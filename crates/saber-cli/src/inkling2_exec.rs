//! INK7: the Inkling 2 executor on ONE card (`saber engine --pack <inkling2 pack> --tp 1
//! --ink-run tf|decode`): the eager 48-visit prefill, the per-visit-state decode and the decode step
//! captured as ONE hipGraph, on Saber's own bodies (`ink_loop` / `ink_glue` / `ink_attn` + the
//! native expert cells `moe_fn148` / `ink_moe`, or the MXFP4 cells of `moe_grouped` on an INK2
//! pack), weights straight from the pack's records.
//!
//! The op sequence, grids, argument values and zero-initialisation are the INK4 / INK5 harness's
//! (`tools/capture/ink4/ink4_prefill.py`, `ink5_decode.py`) one for one, so the rows are gated
//! BYTE for byte against the harness's dumped rows (`tools/capture/ink7/ink7_cmp.py`). Every buffer
//! a visit takes comes zeroed from a bump arena (the harness's `torch.zeros` / `torch.full(0)`),
//! the route's `row_slot` from an all-ones fill (-1). `--ink-split` runs the TP = 2 / EP = 2 split
//! on this one card (each rank's halves as their own launches, the two reduces a visit = rank 0 +
//! rank 1 in that order), the harness's `--tp 2` form. `--tp 2 --ink-run tf` is the TWO-CARD
//! process ([`run_two_card`]): one process, two HIP threads, each rank its halves, the reduces over
//! P2P ([`Peer::reduce`]) -- not the split's bytes (the reduce order differs), gated by the FN80d
//! envelope (`tools/capture/ink7/ink7_envelope.py`).

use std::collections::BTreeMap;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, ensure, Context};
use saber_exec::loader::{FunctionTable, PluginSet};
use saber_exec::LoadCounters;
use saber_hip::{DevicePtr, Event, EventFlags, HipThread, Stream, StreamCaptureMode, StreamFlags};
use serde_json::{json, Value};

use crate::inkling2::{InkFlags, Pack, NATIVE_PACK_FORMAT};

#[path = "inkling2_draft.rs"]
mod draft;
#[path = "inkling2_ngram.rs"]
mod ngram;
#[path = "inkling2_pcache.rs"]
mod pcache;
#[path = "inkling2_serve.rs"]
mod serve;

const D: u64 = 2304;
const I: u64 = 1152;
const HD: u64 = 128;
const PAGE: u64 = 64;
const INK_R: u64 = 8;
const GLOBAL: [u32; 4] = [5, 11, 17, 23];
const CS: u32 = 6;
const CE: u32 = 18;
const NPASS: u32 = 3;
const BLOCK: u32 = 6;
const EPS: f32 = 1e-6;
/// The unpadded vocabulary (the rows the gates read).
pub const VOCAB: usize = 200_058;
const QK_CLIP: f32 = 22.627_417;
const REL_CLIP: f32 = 8.0;
const NFLOOR: f32 = 8192.0;
const ALPHA: f32 = 0.1;
const EXT: u64 = 1024;

/// The code objects this path launches.
pub const PLUGINS: [&str; 29] = [
    "ink_loop",
    "ink_glue",
    "ink_attn",
    "moe_grouped",
    "moe_fn148",
    "ink_moe",
    "ink_p2p",
    "ink_fast",
    "ink_fuse",
    "ink_fuse11",
    "ink_draft",
    "ink_w8",
    "ink_pf",
    "ink_pf2",
    "ink_vt",
    "ink_dr2",
    "ink_xl",
    "ink_mv",
    "ink_dr3",
    "ink_dr4",
    "ink_dr5",
    "ink_i8",
    "ink_fa",
    "ink_r28",
    "ink_p29",
    "ink_dr6",
    "ink_fa35",
    "moe_fn153",
    "ink_p37",
];

/// INK10 `--ink-fuse` bits: the push reduce fused into the next sconv + depth (`x`), the route
/// logits + tail as one launch (`route`).
pub const FUSE_X: u32 = 1;
pub const FUSE_ROUTE: u32 = 2;
/// INK10 task 2: the M <= 8 GEMVs on the narrow column tiles (`ink_gemv_n16` / `n32`, byte-equal).
pub const FUSE_GEMV: u32 = 4;
/// INK10: a visit's message gates as one z + one out launch, the in norms in the out launch.
pub const FUSE_MSG: u32 = 8;
/// INK11: a pass's 48 message depth attends in ONE launch (`ink_depth_mq`).
pub const FUSE_MDEP: u32 = 16;
/// INK11: a visit's q / k / v GEMVs of three input rows in ONE launch (`ink_gemv3_f32_bf16`).
pub const FUSE_QKV: u32 = 32;
/// INK11: the decode rows' kv conv / post / append folded into the attention (`ink_attn_kvc`)
/// under the state parity law (two buffers a kv conv state; needs `x`).
pub const FUSE_KVC: u32 = 64;
/// INK17: the fused form for the chained rows of ONE sequence (a verify chunk / a <= 8-row
/// continuation): `ink_xsdc` (`ink_vt`) forms the rows' reduce + sconv + depth and writes the
/// verify history itself (needs `x`).
pub const FUSE_VT: u32 = 128;
/// INK17: the route's tail per row at > 1 row (`ink_route_fused_rt`, byte-equal; needs `route`).
pub const FUSE_RT: u32 = 256;
/// INK17: the 8-bit trunk GEMVs on the 16-wide tile at EVERY N (`ink_gemv_n16_f32_w8` /
/// `ink_gemv_gate_push_n16_w8`, byte-equal: each output still one sequential sum) -- INK14's
/// 64-wide grid left the local q..v GEMV at 68 workgroups (latency-bound); needs `gemv`.
pub const FUSE_G16: u32 = 512;
/// INK18: the fused consumers' pushed-window loads after ONE system-scope acquire a thread
/// (`ink_xsd_l` / `ink_xsdc_l`, `ink_xl`; byte-equal: the same values, plain loads; needs `x`).
pub const FUSE_XL: u32 = 1024;
/// INK19: the plain 2..8-row GEMVs (batched decode rows, verify chunks, the head) and the q / k / v
/// triple on the column-lane GEMM (`ink_gemm_cl_*_r1n1`, INK16's body: byte-equal, each output one
/// sequential sum over k) -- the b8 census put `ink_gemv_f32_w8` at 97 us (M 8) vs 61 (M 1) and the
/// head half at 972 us; needs `cl` (`--ink-prefill-gemm cl`); one row keeps the GEMV.
pub const FUSE_CL8: u32 = 2048;
/// INK19: the widest GEMV `cl8` moves (the head halves stay on the GEMV).
const CL8_MAX_N: u64 = 8192;
/// INK19: the rows from which `cl8` splits the bf16 q / k / v triple into three launches.
const CL8_GEMV3_MIN_M: u64 = 8;
/// INK28: the decode route on ONE workgroup a row (`ink_route_r1`, `ink_r28`): the 128 lanes are
/// the 128 experts, each logit one lane's sequential sum in the parents' order -> byte-equal to
/// `ink_route_fused` / `_rt` (the 16-workgroup forms: 43 us at one row, 89 at eight); needs `route`.
pub const FUSE_R1: u32 = 4096;
/// INK28: `ink_xsd` with the depth attend's block reductions batched a phase (`ink_xsd2`, `ink_r28`:
/// S + S nq + nq dependent bsum256 rounds -> 3; each value its own tree -> byte-equal); needs `x`.
pub const FUSE_XS2: u32 = 8192;
/// INK30: the latency twins of the single-workgroup train kernels (`ink_r28`: `ink_xsd_p`,
/// `ink_combine_push_p`, `ink_msg_z3_p`, `ink_msg_out3n_p`): the parents' values in the parents'
/// order (byte-equal), the butterflies on DPP / `permlanex16`, one barrier a block sum, every load
/// of a phase issued ahead of its first use; each twin runs exactly where its parent ran.
pub const FUSE_P30: u32 = 16384;
/// INK30 task 4 (MEASUREMENT ONLY, never served): the p30 consumers' `_pw` twins stamp the
/// exchange's spin (the exposed peer wait) and the kernel (`FZ_WT`, read by `--ink-run vprice`);
/// needs `p30`.
pub const FUSE_WT: u32 = 32768;
/// INK10's `all` (the INK11 control).
pub const FUSE_ALL10: u32 = FUSE_X | FUSE_ROUTE | FUSE_GEMV | FUSE_MSG;
/// INK11's `all11` = INK10's `all` + `mdep` + `qkv` (h1 / h2: 712 -> 634 launches, b1 graph
/// 14.21 -> 13.53 ms paired, rows byte-equal); `kvc` stays an ARM (586 launches but the folded
/// attention is 24.5 us vs 8.7 + 12.6: b1 14.01 ms, slower).
pub const FUSE_ALL11: u32 = FUSE_ALL10 | FUSE_MDEP | FUSE_QKV;
/// INK17's `all17` = `all11` + `vt` + `rt` (THE INKLING DEFAULT since INK17; `all11` = the
/// control): verify rounds sha256-equal k1..k4, batch rows byte-equal, served single +15 %,
/// conc-8 +11 % (BetterBench --quick, A B B A).
pub const FUSE_ALL17: u32 = FUSE_ALL11 | FUSE_VT | FUSE_RT;
/// INK10: the fuse words a card (`Card::fz`): two push sites (attention 0, MoE 1), each a
/// `[2 slots][8 rows][D]` f32 receive window + 256 flags (the PEER writes them) + 256 call counters;
/// then the spin diagnostic word and the route's last-arriver counter.
const FZ_WIN: u64 = 2 * 8 * D * 4;
const FZ_FLAGS: u64 = FZ_WIN;
const FZ_CTR: u64 = FZ_WIN + 1024;
const FZ_SITE: u64 = FZ_WIN + 2048;
const FZ_DIAG: u64 = 2 * FZ_SITE;
const FZ_ARRIVE: u64 = 2 * FZ_SITE + 64;
const FZ_MSG: u64 = 2 * FZ_SITE + 128;
/// INK17: `ink_xsdc`'s last-arriver counter (zero between launches).
const FZ_VT: u64 = 2 * FZ_SITE + 256;
/// INK17: `ink_route_fused_rt`'s per-row counters (8 words, zero between launches).
const FZ_RT: u64 = 2 * FZ_SITE + 288;
/// INK30: the `wt` stamps (u64 spin ticks, kernel ticks, launch rows; 100 MHz).
const FZ_WT: u64 = 2 * FZ_SITE + 384;
const FZ_BYTES: u64 = 2 * FZ_SITE + 512;

/// `--ink-fuse` -> the bit set (a comma list of `x`, `route`, `gemv`; `all`; `off`).
/// INK15: `--ink-prefill-gemm`: `pf` (the re-tiles) | `base` (the parents, the control).
/// INK16: `cl` = `pf` + the column-lane GEMM (`ink_pf2`) where [`cl_pick`] names a cell.
/// `clp` = `cl` + the prefill reduce as a push (`ink_p2p_push_v5`). Returns (pf, cl, push).
pub fn prefill_gemm(v: &str) -> anyhow::Result<(bool, bool, bool)> {
    match v {
        "clp" => Ok((true, true, true)),
        "cl" => Ok((true, true, false)),
        "pf" => Ok((true, false, false)),
        "base" => Ok((false, false, false)),
        o => {
            anyhow::bail!("--ink-prefill-gemm {o:?}: clp | cl | pf (INK15's) | base (the control)")
        }
    }
}

/// INK25: `--ink-prefill-trunk`: `iu8` (the matrix-core W8A8 prefill trunk, an arm) | `f32` (the
/// byte-law bodies, the default).
pub fn prefill_trunk(v: &str) -> anyhow::Result<bool> {
    match v {
        "iu8" => Ok(true),
        "f32" => Ok(false),
        o => anyhow::bail!(
            "--ink-prefill-trunk {o:?}: f32 (the default, the control) | iu8 (an arm)"
        ),
    }
}

/// INK27: `--ink-prefill-attn`: `v1` (the default, the control) -> 0 | `rt8` -> 8 | `rt16` -> 16
/// (the rows a workgroup of `ink_fa_rt<R>`); INK31: `rt8t` -> 8 on `ink_fa_rt8t` (the union tile's tail keys
/// spread one (row, key) a thread, byte-equal).
#[cfg(test)]
pub fn prefill_attn(v: &str) -> anyhow::Result<u64> {
    Ok(prefill_attn_sym(v)?.0)
}

/// INK31: `--ink-prefill-attn` -> (rows a workgroup, the symbol; 0 / "" = v1).
pub fn prefill_attn_sym(v: &str) -> anyhow::Result<(u64, &'static str)> {
    match v {
        "v1" => Ok((0, "")),
        "rt8" => Ok((8, "ink_fa_rt8")),
        "rt8t" => Ok((8, "ink_fa_rt8t")),
        "rt16" => Ok((16, "ink_fa_rt16")),
        // INK35: the matrix-core flash attention (64 rows a workgroup; a numerics arm)
        "wm" => Ok((64, "ink_fa35_wm")),
        "wm32" => Ok((32, "ink_fa35_wm32")),
        o => anyhow::bail!(
            "--ink-prefill-attn {o:?}: v1 (the control) | rt8 (the default) | rt8t | rt16 | wm | wm32"
        ),
    }
}

/// INK25: `ink_gemm_i8_w8`'s launch law for an `[m, n]` output: one 16-column n-tile x WV 16-row
/// m-tiles a workgroup (one m-tile a wave, WV <= 16) -> (grid, block threads).
pub fn i8_grid(m: u64, n: u64) -> ([u64; 3], u32) {
    let mt = m.div_ceil(16);
    let wv = mt.min(16);
    ([n.div_ceil(16), mt.div_ceil(wv), 1], 32 * wv as u32)
}

/// INK29 `--ink-p29` items (hip/ink_p29.hip): byte-equal levers and numerics arms.
pub const P29_KX: u32 = 1;
pub const P29_RS: u32 = 2;
pub const P29_FQ: u32 = 4;
pub const P29_LS: u32 = 8;
pub const P29_B16: u32 = 16;
pub const P29_BW: u32 = 32;
pub const P29_I8IN: u32 = 64;
pub const P29_I8O: u32 = 128;
pub const P29_MT: u32 = 256;
pub const P29_MZ: u32 = 512;
pub const P29_MZB: u32 = 1024;

/// INK29 `mt`: the expert tile's rows for a `t`-row prefill chunk on the NVFP4 layers (fn148
/// tm1 / tm2 / tm4 = 16 / 32 / 64 rows; the same per-row bytes): by the mean rows an expert
/// (t x 8 / 64 a rank).
pub fn mt_rows(t: u64) -> u64 {
    match t * 8 / 64 {
        ..=MT32_MIN => 16,
        x if x < MT64_MIN => 32,
        _ => 64,
    }
}
pub const MT32_MIN: u64 = 23;
pub const MT64_MIN: u64 = 56;

/// INK39 `--ink-p39` items: the multi-sequence verify levers (byte-equal; `cat` = the default, `push` an arm).
pub const P39_CAT: u32 = 1;
pub const P39_PUSH: u32 = 2;

/// INK39: `--ink-p39` -> its bits (`none` / empty = 0; `all` = cat + push).
pub fn ink39_bits(v: &str) -> anyhow::Result<u32> {
    let mut b = 0;
    for it in v
        .split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty() && *x != "none")
    {
        b |= match it {
            "cat" => P39_CAT,
            "push" => P39_PUSH,
            "all" => P39_CAT | P39_PUSH,
            o => anyhow::bail!("--ink-p39 item {o:?}: cat | push | all | none"),
        };
    }
    Ok(b)
}

/// INK29: `--ink-p29` -> its bits (`none` / empty = 0).
/// INK35 `--ink-p35` items: `mzs` = the prefill message-gate z on the matrix cores (hi + lo bf16).
pub const P35_MZS: u32 = 1;
pub fn ink35_bits(v: &str) -> anyhow::Result<u32> {
    let mut b = 0;
    for it in v
        .split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty() && *x != "none")
    {
        b |= match it {
            "mzs" => P35_MZS,
            o => anyhow::bail!("--ink-p35 item {o:?}: mzs"),
        };
    }
    Ok(b)
}

/// INK36 `--ink-p36` items (`DEFAULT_INK36`): `ov` = the two-half prefill pipeline, `hg` = the
/// lazy visit zeroing + the per-slot row table, `hgp` = `hg` with the plain takes poisoned (the
/// control), `xw<N>` = the exchange's workgroup cap under `ov` (1..=1024; default 128).
pub const P36_OV: u32 = 1;
pub const P36_HG: u32 = 2;
pub const P36_HGP: u32 = 4;
/// The two-half pipeline's smallest chunk (rows) and the default exchange cap under it.
pub const OV_MIN_ROWS: u64 = 256;
pub const OV_XW: u64 = 128;
/// The per-slot row table's rows (a continued chunk of more rows writes its slot words).
pub const SLOT_ROWS: u64 = 4096;
/// `hg`'s smallest chunk (rows).
pub const HG_MIN_ROWS: u64 = 512;
/// The second half's link words: 64 bytes into the two ranks' 256-byte link allocations.
pub const OV_LINK_OFF: u64 = 64;
pub fn ink36_bits(v: &str) -> anyhow::Result<(u32, u64)> {
    let (mut b, mut xw) = (0, OV_XW);
    for it in v
        .split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty() && *x != "none")
    {
        b |= match it {
            "ov" => P36_OV,
            "hg" => P36_HG,
            "hgp" => P36_HG | P36_HGP,
            o if o.starts_with("xw") => {
                xw = o[2..]
                    .parse()
                    .ok()
                    .filter(|n| (1..=1024).contains(n))
                    .with_context(|| format!("--ink-p36 {o:?}: xw<1..1024>"))?;
                0
            }
            o => anyhow::bail!("--ink-p36 item {o:?}: ov | hg | hgp | xw<N>"),
        };
    }
    Ok((b, xw))
}

/// INK37 `--ink-p37` items (hip/ink_p37.hip + moe_fn153), every one BYTE-EQUAL to its control:
/// `moe` = a prefill chunk's NVFP4 expert GEMMs on FN153's grouped prefill tile `pf_tm4` (64-row
/// expert tiles; h1: 56.4 -> 33.9 ms a 1,024-row chunk); `moe8` = `pf_tm8` (128-row tiles when an
/// expert's mean rows reach `P37_TM8_MIN`; h1: 36.4 ms, slower than tm4 -- an arm); `glue` = the prefill router on 4 rows a workgroup (`ink37_post_norm_route_e128_r4`;
/// `glue8` = 8 rows) + the message out on 8 rows a workgroup (`ink37_msg_out_r8`).
pub const P37_MOE: u32 = 1;
pub const P37_MOE8: u32 = 2;
pub const P37_GLUE: u32 = 4;
pub const P37_GLUE8: u32 = 8;
pub const P37_TM8_MIN: u64 = 96;
pub fn ink37_bits(v: &str) -> anyhow::Result<u32> {
    let mut b = 0;
    for it in v
        .split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty() && *x != "none")
    {
        b |= match it {
            "moe" => P37_MOE,
            "moe8" => P37_MOE8,
            "glue" => P37_GLUE,
            "glue8" => P37_GLUE8,
            o => anyhow::bail!("--ink-p37 item {o:?}: moe | moe8 | glue | glue8"),
        };
    }
    Ok(b)
}
/// CONS40 `--ink-p40` items: `fa128` (global layers), `fa128l` (+ the local layers), `fam<N>` = the
/// smallest chunk start (rows) for either (default `P40_FA_MIN`).
pub const P40_FA128: u32 = 1;
pub const P40_FA128L: u32 = 2;
pub const P40_FA_MIN: u64 = 2048;
/// CONS40 `zd1` / `zd2` (under INK36 `hg`): the visit takes named here are NOT zeroed (their
/// producers write every byte the chunk reads); `zp` POISONS them (0xFF) instead -- the control
/// that the skipped zeroing was dead (byte-equal to A under `zp` = proof).
pub const P40_ZD1: u32 = 4;
pub const P40_ZD2: u32 = 8;
pub const P40_ZP: u32 = 16;
pub const ZD1_NAMES: [&str; 6] = [
    "reduce.out",
    "depth.out",
    "msg.out",
    "rmsnorm.out",
    "sconv.out",
    "u",
];
pub const ZD2_NAMES: [&str; 14] = [
    "attn.out", "o", "og", "gt", "proj.all", "proj.q", "proj.k", "proj.v", "kraw", "vraw", "qraw",
    "rraw", "rp", "sink",
];
pub fn ink40_bits(v: &str) -> anyhow::Result<(u32, u64)> {
    let (mut b, mut fam) = (0, P40_FA_MIN);
    for it in v
        .split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty() && *x != "none")
    {
        b |= match it {
            "fa128" => P40_FA128,
            "fa128l" => P40_FA128 | P40_FA128L,
            "zd1" => P40_ZD1,
            "zd2" => P40_ZD2,
            "zd" => P40_ZD1 | P40_ZD2,
            "zp" => P40_ZP,
            o if o.starts_with("fam") => {
                fam = o[3..]
                    .parse()
                    .ok()
                    .with_context(|| format!("--ink-p40 {o:?}: fam<rows>"))?;
                0
            }
            o => {
                anyhow::bail!("--ink-p40 item {o:?}: fa128 | fa128l | fam<N> | zd | zd1 | zd2 | zp")
            }
        };
    }
    Ok((b, fam))
}

/// INK37 `moe`: the pf tile's rows for a `t`-row chunk (None = the INK29 cells).
pub fn p37_rows(bits: u32, t: u64) -> Option<u64> {
    let mean = t * 8 / 64;
    if bits & P37_MOE8 != 0 && mean >= MT64_MIN {
        Some(if mean >= P37_TM8_MIN { 128 } else { 64 })
    } else if bits & P37_MOE != 0 && mean >= MT64_MIN {
        Some(64)
    } else {
        None
    }
}

pub fn ink29_bits(v: &str) -> anyhow::Result<u32> {
    let mut b = 0;
    for it in v
        .split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty() && *x != "none")
    {
        b |= match it {
            "kx" => P29_KX,
            "rs" => P29_RS,
            "fq" => P29_FQ,
            "ls" => P29_LS,
            "b16" => P29_B16,
            "bw" => P29_BW,
            "i8in" => P29_I8IN,
            "i8o" => P29_I8O,
            "mt" => P29_MT,
            "mz" => P29_MZ,
            "mzb" => P29_MZB,
            o => {
                anyhow::bail!(
                    "--ink-p29 item {o:?}: kx | rs | fq | mt | mz | mzb | ls | b16 | bw | i8in | i8o"
                )
            }
        };
    }
    ensure!(
        b & (P29_LS | P29_B16) != (P29_LS | P29_B16),
        "--ink-p29: ls and b16 are two quant laws; pick one"
    );
    ensure!(
        b & (P29_I8IN | P29_I8O) != (P29_I8IN | P29_I8O),
        "--ink-p29: i8in and i8o are two subsets; pick one"
    );
    Ok(b)
}

/// INK29 `kx`: the K-slice count of the byte-equal iu8 GEMM at an `[m, ., k]` shape, or `None` =
/// INK25's body. Slices only pay where the chunk is latency-bound (few m-tiles); the LDS holds
/// 48 stored blocks ((G - ceil(G / S)) x 4, G = k / 128).
pub fn kx_slices(m: u64, n: u64, k: u64) -> Option<u64> {
    // the twin (INK29 h1, IC-cold): the slices win only where the n-tiles are few (k / v / r /
    // sink); q / gate / o (N >= 1,536) already fill the card and lose at every M
    let max_m = match n {
        ..=32 => 1024,
        33..=192 => 256,
        193..=512 => 64,
        _ => 0,
    };
    if k % 128 != 0 || m > max_m {
        return None;
    }
    let g = k / 128;
    [3u64, 2]
        .into_iter()
        .find(|&s| (g - g.div_ceil(s)) * 4 <= 48 && g >= s)
}

/// INK29 `bw`: the bf16 WMMA GEMM's slice count (a law of K alone: 768 k a slice).
pub fn bw_slices(k: u64) -> Option<u64> {
    (k % 768 == 0 && matches!(k / 768, 1..=3 | 6)).then_some(k / 768)
}

/// INK16: the column-lane cell (rows a lane RM, columns a lane RN) for an `[m, n]` prefill GEMM,
/// or `None` = INK15's `ink_pf` tile. The table is the card-1 twin's IC-cold timing
/// (`tools/capture/ink16/ink16_twin.py`, INK16 h3: every cell below beat the pf tile at M 9..256
/// for N 32 / 192 / 512 / 1,536 / 2,304; above 256 rows the pf tile is kept, unmeasured).
pub fn cl_pick(m: u64, n: u64) -> Option<(u64, u64)> {
    if !(9..=256).contains(&m) {
        return None;
    }
    Some(if n <= 32 {
        (1, 1)
    } else if n <= 192 {
        match m {
            ..=26 => (1, 1),
            27..=52 => (2, 1),
            53..=160 => (2, 2),
            _ => (4, 2),
        }
    } else if n <= 512 {
        match m {
            ..=12 => (1, 1),
            13..=26 => (2, 1),
            27..=52 => (2, 2),
            _ => (4, 2),
        }
    } else {
        (4, 2)
    })
}

/// INK15: the `ink_pf` tile (rows, columns) for an `[m, n]` output: the largest of 64x64, 32x64,
/// 32x32, 16x64, 16x32 whose rows fit `m` (16 always fits) and whose grid has >= 128 workgroups
/// (two a CU), else the smallest-row tile at 16x32.
pub fn pf_tile(m: u64, n: u64) -> (u64, u64) {
    const TILES: [(u64, u64); 5] = [(64, 64), (32, 64), (32, 32), (16, 64), (16, 32)];
    for &(tm, tn) in &TILES {
        if tm > 16 && m < tm * 3 / 4 {
            continue;
        }
        if m.div_ceil(tm) * n.div_ceil(tn) >= 128 {
            return (tm, tn);
        }
    }
    (16, 32)
}

pub fn fuse_bits(v: &str) -> anyhow::Result<u32> {
    let mut b = 0;
    for t in v.split(',') {
        b |= match t {
            "off" => 0,
            "x" => FUSE_X,
            "route" => FUSE_ROUTE,
            "gemv" => FUSE_GEMV,
            "msg" => FUSE_MSG,
            "all" => FUSE_ALL10,
            "mdep" => FUSE_MDEP,
            "qkv" => FUSE_QKV,
            "kvc" => FUSE_KVC,
            "all11" => FUSE_ALL11,
            "all17" => FUSE_ALL17,
            "vt" => FUSE_VT,
            "rt" => FUSE_RT,
            "g16" => FUSE_G16,
            "xl" => FUSE_XL,
            "cl8" => FUSE_CL8,
            "r1" => FUSE_R1,
            "xs2" => FUSE_XS2,
            "p30" => FUSE_P30,
            "wt" => FUSE_WT,
            o => bail!(
                "--ink-fuse {o:?}: off (INK9's chain, the control) | a list of x, route, gemv, msg, \
                 mdep, qkv, kvc, vt, rt, g16, xl, cl8, r1, xs2, p30, wt | all (INK10's set) | all11 | all17"
            ),
        };
    }
    ensure!(
        b & FUSE_VT == 0 || b & FUSE_X != 0,
        "--ink-fuse vt needs x (the fused decode form)"
    );
    ensure!(
        b & FUSE_RT == 0 || b & FUSE_ROUTE != 0,
        "--ink-fuse rt needs route (the fused route)"
    );
    ensure!(
        b & FUSE_XS2 == 0 || b & FUSE_X != 0,
        "--ink-fuse xs2 needs x (the fused decode form)"
    );
    ensure!(
        b & FUSE_WT == 0 || b & FUSE_P30 != 0,
        "--ink-fuse wt needs p30 (the stamped twins)"
    );
    ensure!(
        b & FUSE_R1 == 0 || b & FUSE_ROUTE != 0,
        "--ink-fuse r1 needs route (the fused route)"
    );
    ensure!(
        b & FUSE_XL == 0 || b & FUSE_X != 0,
        "--ink-fuse xl needs x (the fused decode form)"
    );
    ensure!(
        b & FUSE_G16 == 0 || b & FUSE_GEMV != 0,
        "--ink-fuse g16 needs gemv (the narrow tiles)"
    );
    ensure!(
        b & FUSE_KVC == 0 || b & FUSE_X != 0,
        "--ink-fuse kvc needs x (the fused decode form)"
    );
    Ok(b)
}

/// INK12: the drafter's target taps (`mtp_config.target_layer_ids` 5 / 23 / 41 / 48): the
/// partial ENTERING these visits; 48 = the normalized final hidden (read after the loop).
pub const TAP_VISITS: [usize; 3] = [5, 23, 41];

/// The visit schedule (`Inkling2Config::schedule`'s; `noloop` = one pass of the loop).
#[must_use]
pub fn schedule(noloop: bool) -> Vec<u32> {
    let npass = if noloop { 1 } else { NPASS };
    let mut s: Vec<u32> = (0..CS).collect();
    for _ in 0..npass {
        s.extend(CS..CE);
    }
    s.extend(CE..24);
    s
}

/// One explicit kernel argument (packed at the manifest's offsets, its size checked).
#[derive(Debug, Clone)]
pub enum A {
    /// A device address (0 = null).
    P(u64),
    I(i32),
    /// An 8-byte integer (`long`).
    L(i64),
    F(f32),
    /// A by-value struct.
    Blob(Vec<u8>),
}

fn pi(x: u64) -> A {
    A::I(i32::try_from(x).expect("an i32 argument"))
}

/// The function table + the symbol -> plugin map.
pub struct Kernels<'m> {
    table: FunctionTable<'m>,
    plugin_of: BTreeMap<String, String>,
    pub launches: u64,
}

impl<'m> Kernels<'m> {
    fn new(ps: &'m PluginSet, hip: &HipThread) -> anyhow::Result<Self> {
        let mut plugin_of = BTreeMap::new();
        for p in ps.plugins() {
            for k in &p.manifest.kernel {
                plugin_of.insert(k.name.clone(), p.facts.plugin.clone());
            }
        }
        Ok(Self {
            table: ps.functions(hip)?,
            plugin_of,
            launches: 0,
        })
    }

    /// The kernarg image of `sym` (the manifest's explicit args in order, each at its offset).
    pub fn pack(&self, sym: &str, args: &[A]) -> anyhow::Result<Vec<u8>> {
        let plugin = self
            .plugin_of
            .get(sym)
            .with_context(|| format!("{sym}: in none of {PLUGINS:?}"))?;
        let (_, d) = self.table.get(plugin, sym).context("table")?;
        let exp: Vec<_> = d
            .arg
            .iter()
            .filter(|a| !a.value_kind.starts_with("hidden"))
            .collect();
        ensure!(
            exp.len() == args.len(),
            "{sym}: {} explicit args declared, {} given",
            exp.len(),
            args.len()
        );
        let mut img = vec![0u8; d.kernarg_segment_size as usize];
        for (a, v) in exp.iter().zip(args) {
            let o = a.offset as usize;
            let b: Vec<u8> = match v {
                A::P(x) => x.to_le_bytes().to_vec(),
                A::I(x) => x.to_le_bytes().to_vec(),
                A::L(x) => x.to_le_bytes().to_vec(),
                A::F(x) => x.to_le_bytes().to_vec(),
                A::Blob(b) => b.clone(),
            };
            ensure!(
                b.len() == a.size as usize && o + b.len() <= img.len(),
                "{sym}: arg {:?} at {o} is {} B, given {} B ({v:?})",
                a.name,
                a.size,
                b.len()
            );
            img[o..o + b.len()].copy_from_slice(&b);
        }
        Ok(img)
    }
}

// ============================================================================================ pack

/// A device tensor: address, element size, shape (the record's, as the harness reshapes it).
#[derive(Debug, Clone)]
pub struct Tw {
    pub addr: u64,
    pub es: u64,
    pub shape: Vec<u64>,
    /// INK14: the 8-bit trunk record's bf16 scales `[N, K / 32]` (`addr` = the int8 codes
    /// `[N, K]`, `es` 1); 0 = a bf16 weight
    pub sc: u64,
}

impl Tw {
    fn rows(&self) -> u64 {
        self.shape[0]
    }
    /// Row `i` of a tensor viewed as `[*, cols]`.
    fn row(&self, i: u64, cols: u64) -> u64 {
        self.addr + i * cols * self.es
    }
}

struct Rec {
    file: String,
    offset: u64,
    length: u64,
    stride: u64,
    count: u64,
    sha256: String,
}

/// The pack's records + the checkpoint's shapes (host side).
struct PackReader {
    dir: PathBuf,
    ckpt: PathBuf,
    man: Value,
    recs: [BTreeMap<String, Rec>; 3],
    shapes: BTreeMap<String, (String, Vec<u64>)>,
    files: std::cell::RefCell<BTreeMap<String, std::fs::File>>,
    /// INK14: the 8-bit trunk overlay (`--ink-trunk w8 | w8-head`); None = the bf16 trunk
    t8: Option<Trunk8>,
}

/// INK14: the 8-bit dense trunk overlay (`ink14-trunk8/1`, `tools/pack/build_inkling2_pack.py
/// --trunk8`): per (tensor, rank) int8 codes `[rows, cols]` + bf16 scales `[rows, cols / 32]`,
/// each bound to the base pack's bf16 record by that record's sha256.
pub const TRUNK8_FORMAT: &str = "ink14-trunk8/1";

/// INK14: an 8-bit record as read: (int8 codes, bf16 scales, [rows, cols]).
type W8Record = (Vec<u8>, Vec<u8>, Vec<u64>);

struct T8Rec {
    file: String,
    rows: u64,
    cols: u64,
    codes_off: u64,
    scales_off: u64,
}

struct Trunk8 {
    dir: PathBuf,
    /// the head (`lm_head`) from the overlay too (`w8-head`)
    head: bool,
    /// a SUBSET overlay (the manifest's `partial`: a bisection arm) -- a tensor it does not carry
    /// stays bf16; a full overlay refuses a missing record
    partial: bool,
    recs: BTreeMap<(String, u32), T8Rec>,
}

impl Trunk8 {
    /// `mode` = `w8` (attention projections + o_proj) | `w8-head` (+ the head).
    fn open(dir: &Path, mode: &str, base: &[BTreeMap<String, Rec>; 3]) -> anyhow::Result<Self> {
        let head = match mode {
            "w8" => false,
            "w8-head" => true,
            o => bail!("--ink-trunk {o:?}: bf16 (the default, the control) | w8 | w8-head"),
        };
        let mp = dir.join("manifest.json");
        let man: Value = serde_json::from_str(
            &std::fs::read_to_string(&mp).with_context(|| mp.display().to_string())?,
        )?;
        ensure!(
            man["format"].as_str() == Some(TRUNK8_FORMAT),
            "{}: format {:?}, this path reads {TRUNK8_FORMAT}",
            mp.display(),
            man["format"]
        );
        let mut recs = BTreeMap::new();
        for r in man["tensors"].as_array().context("trunk8 tensors")? {
            let name = r["tensor"].as_str().unwrap_or("").to_string();
            let rank = u32::try_from(r["rank"].as_u64().unwrap_or(9))?;
            ensure!(rank < 2, "trunk8 {name}: rank {rank}");
            let src = r["src_sha256"].as_str().unwrap_or("");
            let base_sha = base[1 + rank as usize]
                .get(&name)
                .map(|b| b.sha256.as_str())
                .with_context(|| format!("trunk8 {name}: no rank{rank} record in the base pack"))?;
            ensure!(
                src == base_sha,
                "trunk8 {name} rank {rank}: built from {src}, the base pack's record is {base_sha}"
            );
            let u = |k: &str| r[k].as_u64().unwrap_or(0);
            recs.insert(
                (name, rank),
                T8Rec {
                    file: r["file"].as_str().unwrap_or("").to_string(),
                    rows: u("rows"),
                    cols: u("cols"),
                    codes_off: u("codes_off"),
                    scales_off: u("scales_off"),
                },
            );
        }
        println!(
            "INK14 trunk8: --ink-trunk {mode} from {} ({} records, head {})",
            dir.display(),
            recs.len(),
            if head { "8-bit" } else { "bf16" }
        );
        Ok(Self {
            dir: dir.to_path_buf(),
            head,
            partial: man["partial"].as_bool().unwrap_or(false),
            recs,
        })
    }
}

impl PackReader {
    fn open(p: &Pack) -> anyhow::Result<Self> {
        let mut recs: [BTreeMap<String, Rec>; 3] = Default::default();
        for (i, d) in ["shared", "rank0", "rank1"].iter().enumerate() {
            let f = p.dir.join(d).join("dense.json");
            let v: Value = serde_json::from_str(&std::fs::read_to_string(&f)?)?;
            for r in v.as_array().context("dense.json")? {
                let u = |k: &str| r[k].as_u64().unwrap_or(0);
                recs[i].insert(
                    r["tensor"].as_str().unwrap_or("").to_string(),
                    Rec {
                        file: r["file"].as_str().unwrap_or("").to_string(),
                        offset: u("offset"),
                        length: u("length"),
                        stride: u("stride"),
                        count: u("count"),
                        sha256: r["sha256"].as_str().unwrap_or("").to_string(),
                    },
                );
            }
        }
        let mut shapes = BTreeMap::new();
        for e in std::fs::read_dir(&p.ckpt)? {
            let path = e?.path();
            if path.extension().and_then(|s| s.to_str()) != Some("safetensors") {
                continue;
            }
            let f = std::fs::File::open(&path)?;
            let mut n8 = [0u8; 8];
            f.read_exact_at(&mut n8, 0)?;
            let n = u64::from_le_bytes(n8) as usize;
            let mut h = vec![0u8; n];
            f.read_exact_at(&mut h, 8)?;
            let hv: Value = serde_json::from_slice(&h)?;
            for (k, v) in hv.as_object().context("header")? {
                if k == "__metadata__" {
                    continue;
                }
                let shape = v["shape"]
                    .as_array()
                    .map(|a| a.iter().filter_map(Value::as_u64).collect())
                    .unwrap_or_default();
                shapes.insert(
                    k.clone(),
                    (v["dtype"].as_str().unwrap_or("").to_string(), shape),
                );
            }
        }
        Ok(Self {
            dir: p.dir.clone(),
            ckpt: p.ckpt.clone(),
            man: p.manifest.clone(),
            recs,
            shapes,
            files: std::cell::RefCell::new(BTreeMap::new()),
            t8: None,
        })
    }

    /// INK14: bind the 8-bit trunk overlay (`--ink-trunk`, `--ink-trunk8-dir`). An UNSET
    /// `--ink-trunk` resolves to `DEFAULT_INK_TRUNK` when the overlay is present and the boot is
    /// the two-card form, else STANDS DOWN to bf16 by name; an explicit `w8*` refuses.
    fn with_trunk(mut self, f: &InkFlags, pack_dir: &Path, two_card: bool) -> anyhow::Result<Self> {
        let dir = f
            .ink_trunk8_dir
            .clone()
            .unwrap_or_else(|| pack_dir.join("trunk8"));
        let mode = match f.ink_trunk.as_deref() {
            Some(m) => m.to_string(),
            None if two_card && dir.join("manifest.json").is_file() => {
                crate::inkling2::DEFAULT_INK_TRUNK.to_string()
            }
            None => {
                println!(
                    "INK14 trunk8: STANDS DOWN to bf16 (the default {} needs the two-card form and \
                     an overlay at {}: {})",
                    crate::inkling2::DEFAULT_INK_TRUNK,
                    dir.display(),
                    if two_card { "absent" } else { "a one-card run" }
                );
                "bf16".to_string()
            }
        };
        if mode != "bf16" {
            ensure!(
                two_card,
                "--ink-trunk {mode}: the 8-bit trunk is the two-card form's (rank records)"
            );
            self.t8 = Some(Trunk8::open(&dir, &mode, &self.recs)?);
        }
        Ok(self)
    }

    /// INK14: the 8-bit record of `name` (no `language_model.` prefix) for `rank`, when the
    /// overlay is bound and covers it: (codes, scales, [rows, cols]).
    fn tensor_w8(
        &self,
        name: &str,
        rank: Option<u32>,
        head: bool,
    ) -> anyhow::Result<Option<W8Record>> {
        let Some(t8) = &self.t8 else { return Ok(None) };
        if head && !t8.head {
            return Ok(None);
        }
        let rk = rank.context("the 8-bit trunk is the two-card form's (a rank's records)")?;
        let key = (format!("language_model.{name}"), rk);
        if t8.partial && !t8.recs.contains_key(&key) {
            return Ok(None);
        }
        let r = t8
            .recs
            .get(&key)
            .with_context(|| format!("trunk8: no record {} rank {rk}", key.0))?;
        let f = std::fs::File::open(t8.dir.join(&r.file))
            .with_context(|| t8.dir.join(&r.file).display().to_string())?;
        let mut codes = vec![0u8; (r.rows * r.cols) as usize];
        f.read_exact_at(&mut codes, r.codes_off)?;
        let mut sc = vec![0u8; (r.rows * r.cols / 32 * 2) as usize];
        f.read_exact_at(&mut sc, r.scales_off)?;
        Ok(Some((codes, sc, vec![r.rows, r.cols])))
    }

    fn read_at(&self, file: &str, off: u64, buf: &mut [u8]) -> anyhow::Result<()> {
        let mut fs = self.files.borrow_mut();
        if !fs.contains_key(file) {
            let f = std::fs::File::open(self.ckpt.join(file)).with_context(|| file.to_string())?;
            fs.insert(file.to_string(), f);
        }
        fs[file].read_exact_at(buf, off)?;
        Ok(())
    }

    fn rec_bytes(&self, r: &Rec) -> anyhow::Result<Vec<u8>> {
        let mut out = vec![0u8; (r.length * r.count) as usize];
        if r.count == 1 {
            self.read_at(&r.file, r.offset, &mut out)?;
        } else {
            let span = r.stride * (r.count - 1) + r.length;
            let mut blob = vec![0u8; span as usize];
            self.read_at(&r.file, r.offset, &mut blob)?;
            for i in 0..r.count as usize {
                let (s, l) = (i * r.stride as usize, r.length as usize);
                out[i * l..(i + 1) * l].copy_from_slice(&blob[s..s + l]);
            }
        }
        Ok(out)
    }

    /// `Pack.tensor` of the harness: (bytes, element size, shape). `rank None` = the whole tensor
    /// (shared, or the two rank halves assembled); else the rank's piece.
    fn tensor(&self, name: &str, rank: Option<u32>) -> anyhow::Result<(Vec<u8>, u64, Vec<u64>)> {
        let name = format!("language_model.{name}");
        let (dt, shape) = self
            .shapes
            .get(&name)
            .with_context(|| format!("{name}: not in the checkpoint"))?;
        let es = if dt == "BF16" { 2 } else { 4 };
        if let Some(r) = self.recs[0].get(&name) {
            return Ok((self.rec_bytes(r)?, es, shape.clone()));
        }
        let r0 = self.recs[1].get(&name).with_context(|| name.clone())?;
        let r1 = self.recs[2].get(&name).with_context(|| name.clone())?;
        if let Some(rk) = rank {
            let r = if rk == 0 { r0 } else { r1 };
            let b = self.rec_bytes(r)?;
            let sh = if r.count == 1 {
                let mut s = shape.clone();
                s[0] /= 2;
                s
            } else {
                vec![r.count, r.length / es]
            };
            return Ok((b, es, sh));
        }
        let (a0, a1) = (self.rec_bytes(r0)?, self.rec_bytes(r1)?);
        if r0.count == 1 {
            let mut v = a0;
            v.extend_from_slice(&a1);
            return Ok((v, es, shape.clone()));
        }
        let (l0, l1) = (r0.length as usize, r1.length as usize);
        let mut v = Vec::with_capacity(a0.len() + a1.len());
        for i in 0..r0.count as usize {
            v.extend_from_slice(&a0[i * l0..(i + 1) * l0]);
            v.extend_from_slice(&a1[i * l1..(i + 1) * l1]);
        }
        Ok((v, es, shape.clone()))
    }

    /// The embedding rows of `ids` as f32 (bf16 bits << 16), `[T, D]`.
    fn embed_rows(&self, ids: &[u32]) -> anyhow::Result<Vec<u8>> {
        let r = self.recs[0]
            .get("language_model.model.embed_tokens.weight")
            .context("embed_tokens")?;
        let mut out = Vec::with_capacity(ids.len() * D as usize * 4);
        let mut row = vec![0u8; 2 * D as usize];
        for &i in ids {
            self.read_at(&r.file, r.offset + 2 * D * u64::from(i), &mut row)?;
            for c in row.chunks_exact(2) {
                let b = u32::from(u16::from_le_bytes([c[0], c[1]])) << 16;
                out.extend_from_slice(&b.to_le_bytes());
            }
        }
        Ok(out)
    }
}

// ============================================================================================ card

/// A bump region of one device allocation; every take is 256-aligned and comes ZEROED (the region
/// is zeroed at birth and every reset zeroes what the pass dirtied).
struct Bump {
    off: u64,
    cap: u64,
    top: u64,
    high: u64,
}

impl Bump {
    fn take(&mut self, bytes: u64, what: &str) -> anyhow::Result<u64> {
        let b = bytes.div_ceil(256) * 256;
        ensure!(
            self.top + b <= self.cap,
            "ink arena: {what} {bytes} B past the region's {} B (used {})",
            self.cap,
            self.top
        );
        let o = self.off + self.top;
        self.top += b;
        self.high = self.high.max(self.top);
        Ok(o)
    }
}

/// The card: HIP thread, stream, kernels, the work allocation (pass + visit regions), the weight
/// allocations.
pub struct Card<'h, 'm> {
    pub hip: &'h HipThread,
    pub stream: Stream,
    pub k: Kernels<'m>,
    work: DevicePtr,
    pass: Bump,
    visit: Bump,
    keep: Vec<DevicePtr>,
    /// The visit region's high-water mark over the run (bytes).
    pub visit_high: u64,
    /// INK9: the fast bodies (`--ink-bodies fast`, byte-equal to v0's); false = v0 (the control).
    pub fast: bool,
    /// INK10: the fused decode bodies (`--ink-fuse`, bit set `FUSE_*`; 0 = INK9's chain, the
    /// control) and their words (`fz`: the push windows / flags / counters + the route counter)
    pub fuse: u32,
    fz: u64,
    /// INK15: the prefill GEMMs on `ink_pf`'s re-tiles (`--ink-prefill-gemm pf`)
    pub pf: bool,
    /// INK16: the column-lane cells where [`cl_pick`] names one (`--ink-prefill-gemm cl`); the
    /// path counters (cl launches, pf fallbacks of a picked cell: alignment / K % 64)
    pub cl: bool,
    pub cl_n: [u64; 3],
    /// INK16: the prefill reduce as a push (`ink_p2p_push_v5`, `--ink-prefill-gemm clp`); `cl_n[2]`
    /// counts its launches
    pub push: bool,
    /// INK25: the matrix-core W8A8 prefill trunk (`--ink-prefill-trunk iu8`); `i8_live` = the
    /// current GEMMs are a prefill chunk's attention projections (set by `Forward::attention`);
    /// the quantized inputs of this visit (a, m, k) -> (codes, scales), dropped at every region
    /// reset; counters (iu8 GEMMs, quant launches, law misses -> the f32 body)
    pub i8: bool,
    /// INK27: the rows a workgroup of the row-tiled prefill attention (`ink_fa_rt<R>`; 0 = v1);
    /// `fa_n` counts its launches
    pub fa: u64,
    pub fa_sym: &'static str,
    pub fa_n: u64,
    /// INK31: the routing floor (a chunk's last position >= `fa_floor`)
    pub fa_floor: u64,
    pub i8_live: bool,
    i8_cache: Vec<(u64, u64, u64, u64, u64)>,
    pub i8_n: [u64; 3],
    /// INK29: `--ink-p29` bits; `p29_live` = a prefill chunk's attention projections (set with
    /// `i8_live`, independent of `i8`); counters (kx GEMMs, bw GEMMs, v2 route sorts, fused
    /// out_gate quants, ls / b16 quants, subset skips)
    pub p29: u32,
    pub p29_live: bool,
    /// INK35: `--ink-p35` bits; `p35_live` = a prefill chunk's visit (the p29_live rows, set
    /// for the whole visit); `p35_n` = mzs GEMMs
    pub p35: u32,
    pub p35_live: bool,
    /// INK37: `--ink-p37` bits; `p37_n` = [pf-tile MoE parts, R-row routes]
    pub p37: u32,
    pub p37_n: [u64; 2],
    /// CONS40: `--ink-p40` bits + the chunk start floor; `p40_n` = wm128 launches
    pub p40: u32,
    pub p40_fam: u64,
    pub p40_n: u64,
    pub p40_zd: u64,
    pub p35_n: u64,
    /// INK36: `--ink-p36` bits + the exchange cap under `ov`; `hg_live` = a prefill chunk's
    /// lazy zeroing is on (the visit region's zeroed takes memset at their take, merged into
    /// `zpend`; the plain takes stale, or poisoned under `hgp`); `hg_high` = the visit region's
    /// extent dirtied while live (zeroed once at the chunk's end); `alt` / `visit2` / `link_off` =
    /// the second half's stream, visit region and link words while `ov` swaps them in;
    /// `p36_n` = (ov chunks, hg chunks, lazy memsets, ov stand-downs)
    pub p36: u32,
    pub p36_xw: u64,
    pub hg_live: bool,
    zpend: Option<(u64, u64)>,
    hg_high: u64,
    hg_high2: u64,
    alt: Option<Stream>,
    visit2: Bump,
    pub link_off: u64,
    ovev: Vec<Event>,
    ov_live: bool,
    pub p36_n: [u64; 4],
    pub p29_n: [u64; 6],
    /// INK39: `--ink-p39` bits; `mv_live` = inside a multi-sequence verify pass (`Forward::mrow`);
    /// counters (merged projection GEMMs, push reduces)
    pub p39: u32,
    pub mv_live: bool,
    pub p39_n: [u64; 2],
    /// INK9: per-launch event pairs while `Some` (`--ink-op-timing`).
    timing: Option<OpTiming>,
    tpool: Vec<Event>,
    /// INK12: the drafter's taps of one eager pass while `Some` -- the partial entering visits
    /// 5 / 23 / 41 and the normalized final hidden (visit 48), each `[t, D]` f32, in that order
    pub taps: Option<Vec<Vec<u8>>>,
    /// INK12: the same taps copied on the card into 4 planes (base, plane bytes) of the drafter
    pub tapdev: Option<(u64, u64)>,
    /// INK12 task 3: a verify chunk's conv history (`stash` / `commit`) while `Some`
    pub hist: Option<draft::Hist>,
    pub hist_spare: Option<draft::Hist>,
}

/// INK9: one timed step's event pairs + what each launch was.
#[derive(Default)]
struct OpTiming {
    ev: Vec<Event>,
    log: Vec<(String, [u32; 3])>,
}

/// Which region a take comes from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reg {
    /// Cross-visit tensors (the residual stream, the head).
    Pass,
    /// One visit's temporaries.
    Visit,
}

impl<'h, 'm> Card<'h, 'm> {
    fn new(
        hip: &'h HipThread,
        k: Kernels<'m>,
        pass_bytes: u64,
        visit_bytes: u64,
    ) -> anyhow::Result<Self> {
        let stream = hip.create_stream(StreamFlags::NonBlocking)?;
        let work = hip.malloc(pass_bytes + visit_bytes)?;
        hip.memset(&work, 0, &stream)?;
        hip.sync_stream(&stream)?;
        Ok(Self {
            hip,
            stream,
            k,
            pass: Bump {
                off: 0,
                cap: pass_bytes,
                top: 0,
                high: 0,
            },
            visit: Bump {
                off: pass_bytes,
                cap: visit_bytes,
                top: 0,
                high: 0,
            },
            work,
            keep: Vec::new(),
            visit_high: 0,
            fast: true,
            fuse: 0,
            pf: false,
            cl: false,
            cl_n: [0; 3],
            push: false,
            i8: false,
            fa: 0,
            fa_sym: "",
            fa_n: 0,
            fa_floor: 0,
            i8_live: false,
            i8_cache: Vec::new(),
            i8_n: [0; 3],
            p29: 0,
            p29_live: false,
            p35: 0,
            p35_live: false,
            p37: 0,
            p37_n: [0; 2],
            p40: 0,
            p40_fam: P40_FA_MIN,
            p40_n: 0,
            p40_zd: 0,
            p35_n: 0,
            p36: 0,
            p36_xw: OV_XW,
            hg_live: false,
            zpend: None,
            hg_high: 0,
            hg_high2: 0,
            alt: None,
            visit2: Bump {
                off: pass_bytes + visit_bytes,
                cap: 0,
                top: 0,
                high: 0,
            },
            link_off: 0,
            ovev: Vec::new(),
            ov_live: false,
            p36_n: [0; 4],
            p29_n: [0; 6],
            p39: 0,
            mv_live: false,
            p39_n: [0; 2],
            fz: 0,
            timing: None,
            tpool: Vec::new(),
            taps: None,
            tapdev: None,
            hist: None,
            hist_spare: None,
        })
    }

    fn base(&self) -> u64 {
        self.work.addr().as_u64()
    }

    fn take_raw(&mut self, reg: Reg, bytes: u64, what: &str) -> anyhow::Result<u64> {
        let o = match reg {
            Reg::Pass => self.pass.take(bytes, what)?,
            Reg::Visit => self.visit.take(bytes, what)?,
        };
        Ok(self.base() + o)
    }

    /// A plain take: zeroed (the region law) -- except a visit take while INK36's `hg` is live,
    /// which holds whatever the region held (its producer writes every byte it reads: `hgp`
    /// poisons it to prove that).
    fn take(&mut self, reg: Reg, bytes: u64, what: &str) -> anyhow::Result<u64> {
        let a = self.take_raw(reg, bytes, what)?;
        if self.hg_live && reg == Reg::Visit && self.p36 & P36_HGP != 0 {
            self.flush_z()?;
            let b = bytes.div_ceil(256) * 256;
            self.hip
                .memset_at(&self.work, a - self.base(), b, 0xFF, &self.stream)?;
            self.p36_n[2] += 1;
        }
        Ok(a)
    }

    /// A zeroed f32 / i32 buffer of `n` elements. INK36 `hg`: a visit take is zeroed HERE
    /// (contiguous zeroed takes merged into one memset, issued before the next launch).
    fn z4(&mut self, reg: Reg, n: u64, what: &str) -> anyhow::Result<u64> {
        // CONS40 `zd1` / `zd2`: a named take whose producer writes every byte read is not zeroed
        // (`zp`: poisoned, the control)
        if self.hg_live
            && reg == Reg::Visit
            && ((self.p40 & P40_ZD1 != 0 && ZD1_NAMES.contains(&what))
                || (self.p40 & P40_ZD2 != 0 && ZD2_NAMES.contains(&what)))
        {
            let a = self.take_raw(reg, 4 * n, what)?;
            self.p40_zd += 1;
            if self.p40 & P40_ZP != 0 {
                self.flush_z()?;
                let b = (4 * n).div_ceil(256) * 256;
                self.hip
                    .memset_at(&self.work, a - self.base(), b, 0xFF, &self.stream)?;
            }
            return Ok(a);
        }
        if self.hg_live && reg == Reg::Visit {
            let a = self.take_raw(reg, 4 * n, what)?;
            let e = a + (4 * n).div_ceil(256) * 256;
            match self.zpend {
                Some((s0, e0)) if e0 == a => self.zpend = Some((s0, e)),
                _ => {
                    self.flush_z()?;
                    self.zpend = Some((a, e));
                }
            }
            return Ok(a);
        }
        self.take(reg, 4 * n, what)
    }

    /// INK36 `hg`: issue the pending merged zero run.
    fn flush_z(&mut self) -> anyhow::Result<()> {
        if let Some((a, e)) = self.zpend.take() {
            let off = a - self.base();
            if let Some(tm) = self.timing.as_mut() {
                let i = tm.log.len();
                while tm.ev.len() < 2 * i + 2 {
                    tm.ev.push(self.hip.create_event(EventFlags::Default)?);
                }
                self.hip.record_event(&tm.ev[2 * i], &self.stream)?;
                self.hip
                    .memset_at(&self.work, off, e - a, 0, &self.stream)?;
                self.hip.record_event(&tm.ev[2 * i + 1], &self.stream)?;
                tm.log
                    .push(("memset.z4".into(), [u32::try_from((e - a) >> 10)?, 0, 0]));
            } else {
                self.hip
                    .memset_at(&self.work, off, e - a, 0, &self.stream)?;
            }
            self.p36_n[2] += 1;
        }
        Ok(())
    }

    /// INK36 `hg`: start (a prefill chunk's forward) / end the lazy zeroing. The end zeroes the
    /// visit region's whole extent dirtied while live (stale plain takes included) once, so the
    /// region law holds again for every later user.
    fn hg_begin(&mut self) -> anyhow::Result<()> {
        if self.p36 & P36_HG != 0 && !self.hg_live {
            self.reset(Reg::Visit)?;
            self.hg_live = true;
            self.hg_high = 0;
            self.p36_n[1] += 1;
        }
        Ok(())
    }

    fn hg_end(&mut self) -> anyhow::Result<()> {
        if self.hg_live {
            self.flush_z()?;
            self.hg_high = self.hg_high.max(self.visit.top);
            self.hg_live = false;
            let (off, top) = (self.visit.off, self.hg_high);
            self.i8_cache.clear();
            self.visit_high = self.visit_high.max(top);
            self.hip.memset_at(&self.work, off, top, 0, &self.stream)?;
            // the last stage's takes stay marked dirty (zero now): the next reset zeroes them
            // again, as after the control's pass -- a graph captured next records that reset,
            // and its replays rely on it to re-zero their own last stage (vp / conc gates, h3)
        }
        Ok(())
    }

    /// An all-ones buffer (i32 -1).
    fn neg1(&mut self, n: u64, what: &str) -> anyhow::Result<u64> {
        let a = self.take(Reg::Visit, 4 * n, what)?;
        self.hip
            .memset_at(&self.work, a - self.base(), 4 * n, 0xFF, &self.stream)?;
        Ok(a)
    }

    /// Zero what a region's takes dirtied and restart it.
    fn reset(&mut self, reg: Reg) -> anyhow::Result<()> {
        let b = match reg {
            Reg::Pass => &mut self.pass,
            Reg::Visit => &mut self.visit,
        };
        let (off, top) = (b.off, b.top);
        b.top = 0;
        // INK25: a quantized input lives in the visit region and its source may be re-taken
        self.i8_cache.clear();
        if reg == Reg::Visit {
            self.visit_high = self.visit_high.max(top);
            if self.hg_live {
                // INK36 `hg`: nothing zeroed here (each zeroed take zeroes itself)
                self.flush_z()?;
                self.hg_high = self.hg_high.max(top);
                return Ok(());
            }
        }
        if let Some(tm) = self.timing.as_mut() {
            // INK36: the census names the region resets (`memset.visit` / `.pass`, KiB)
            let i = tm.log.len();
            while tm.ev.len() < 2 * i + 2 {
                tm.ev.push(self.hip.create_event(EventFlags::Default)?);
            }
            self.hip.record_event(&tm.ev[2 * i], &self.stream)?;
            self.hip.memset_at(&self.work, off, top, 0, &self.stream)?;
            self.hip.record_event(&tm.ev[2 * i + 1], &self.stream)?;
            let what = if reg == Reg::Visit {
                "memset.visit"
            } else {
                "memset.pass"
            };
            tm.log
                .push((what.into(), [u32::try_from(top >> 10)?, 0, 0]));
            return Ok(());
        }
        self.hip.memset_at(&self.work, off, top, 0, &self.stream)?;
        Ok(())
    }

    /// INK36 `ov`: an event of the pipeline's pool (dependencies only, no timing).
    fn ov_event(&mut self, i: usize) -> anyhow::Result<()> {
        while self.ovev.len() <= i {
            self.ovev
                .push(self.hip.create_event(EventFlags::DisableTiming)?);
        }
        Ok(())
    }

    /// INK36 `ov`: enter the two-half form -- the second stream ordered behind everything queued
    /// so far (the pass prologue), the visit region split in two halves (the second half's is
    /// `visit2`), half A current.
    fn ov_begin(&mut self) -> anyhow::Result<()> {
        ensure!(!self.ov_live, "ov: re-entered");
        self.reset(Reg::Visit)?;
        self.flush_z()?;
        if self.alt.is_none() {
            self.alt = Some(self.hip.create_stream(StreamFlags::NonBlocking)?);
        }
        self.ov_event(0)?;
        self.hip.record_event(&self.ovev[0], &self.stream)?;
        self.hip
            .stream_wait_event(self.alt.as_ref().context("alt")?, &self.ovev[0])?;
        let half = (self.visit.cap / 2) / 256 * 256;
        self.visit2 = Bump {
            off: self.visit.off + half,
            cap: half,
            top: 0,
            high: 0,
        };
        self.visit.cap = half;
        self.ov_live = true;
        self.p36_n[0] += 1;
        Ok(())
    }

    /// INK36 `ov`: switch halves (stream, visit region, link words, dirty extent, tap rows).
    fn ov_swap(&mut self, tap_rows: u64) -> anyhow::Result<()> {
        self.flush_z()?;
        self.i8_cache.clear();
        let alt = self.alt.as_mut().context("ov: no second stream")?;
        std::mem::swap(&mut self.stream, alt);
        std::mem::swap(&mut self.visit, &mut self.visit2);
        std::mem::swap(&mut self.hg_high, &mut self.hg_high2);
        self.link_off = if self.link_off == 0 { OV_LINK_OFF } else { 0 };
        // the half-B rows of a drafter tap start `tap_rows` rows into each plane
        if let Some((tb, tp)) = self.tapdev {
            let d = tap_rows * D * 4;
            self.tapdev = Some((if self.link_off != 0 { tb + d } else { tb - d }, tp));
        }
        Ok(())
    }

    /// INK36 `ov`: the current half's stream marks visit `vi` done / waits for the other half's.
    fn ov_mark(&mut self, vi: usize) -> anyhow::Result<()> {
        self.flush_z()?;
        self.ov_event(vi + 2)?;
        self.hip.record_event(&self.ovev[vi + 2], &self.stream)?;
        Ok(())
    }

    fn ov_wait(&mut self, vi: usize) -> anyhow::Result<()> {
        self.hip
            .stream_wait_event(&self.stream, &self.ovev[vi + 2])?;
        Ok(())
    }

    /// INK36 `ov`: leave the two-half form from half B: its region zeroed over everything it
    /// dirtied, the main stream ordered behind it, the region whole again, half A's state current.
    fn ov_end(&mut self, tap_rows: u64) -> anyhow::Result<()> {
        ensure!(
            self.ov_live && self.link_off != 0,
            "ov: ended outside half B"
        );
        self.flush_z()?;
        let ext = if self.hg_live {
            self.hg_high.max(self.visit.top)
        } else {
            self.visit.top
        };
        self.visit_high = self.visit_high.max(ext);
        self.hip
            .memset_at(&self.work, self.visit.off, ext, 0, &self.stream)?;
        self.visit.top = 0;
        self.hg_high = 0;
        self.ov_event(1)?;
        self.hip.record_event(&self.ovev[1], &self.stream)?;
        self.ov_swap(tap_rows)?;
        self.hip.stream_wait_event(&self.stream, &self.ovev[1])?;
        self.visit.cap += self.visit2.cap;
        self.visit2.cap = 0;
        self.ov_live = false;
        Ok(())
    }

    fn up(&mut self, bytes: &[u8]) -> anyhow::Result<u64> {
        let d = self.hip.malloc(bytes.len().max(1) as u64)?;
        self.hip.memcpy_h2d_blocking(&d, bytes, &self.stream)?;
        let a = d.addr().as_u64();
        self.keep.push(d);
        Ok(a)
    }

    fn alloc_zero(&mut self, bytes: u64) -> anyhow::Result<u64> {
        let d = self.hip.malloc(bytes.max(256))?;
        self.hip.memset(&d, 0, &self.stream)?;
        let a = d.addr().as_u64();
        self.keep.push(d);
        Ok(a)
    }

    /// Write host bytes at a device address inside one of this card's allocations.
    fn write(&self, addr: u64, bytes: &[u8]) -> anyhow::Result<()> {
        ensure!(self.zpend.is_none(), "ink write under a pending zero run");
        for d in std::iter::once(&self.work).chain(self.keep.iter()) {
            let b = d.addr().as_u64();
            if addr >= b && addr + bytes.len() as u64 <= b + d.len() {
                self.hip
                    .memcpy_h2d_at_blocking(d, addr - b, bytes, &self.stream)?;
                return Ok(());
            }
        }
        bail!("ink write: {addr:#x} in none of the card's allocations")
    }

    fn read(&self, addr: u64, bytes: usize) -> anyhow::Result<Vec<u8>> {
        ensure!(self.zpend.is_none(), "ink read under a pending zero run");
        let mut v = vec![0u8; bytes];
        for d in std::iter::once(&self.work).chain(self.keep.iter()) {
            let b = d.addr().as_u64();
            if addr >= b && addr + bytes as u64 <= b + d.len() {
                self.hip
                    .memcpy_d2h_at_blocking(&mut v, d, addr - b, &self.stream)?;
                return Ok(v);
            }
        }
        bail!("ink read: {addr:#x} in none of the card's allocations")
    }

    fn launch(&mut self, sym: &str, grid: [u64; 3], block: u32, args: &[A]) -> anyhow::Result<()> {
        self.flush_z()?;
        let mut img = self.k.pack(sym, args)?;
        let plugin = self.k.plugin_of[sym].clone();
        let (f, _) = self.k.table.get(&plugin, sym).context("table")?;
        let g = grid.map(|x| u32::try_from(x).expect("grid"));
        if let Some(tm) = self.timing.as_mut() {
            let i = tm.log.len();
            while tm.ev.len() < 2 * i + 2 {
                tm.ev.push(self.hip.create_event(EventFlags::Default)?);
            }
            self.hip.record_event(&tm.ev[2 * i], &self.stream)?;
            self.hip
                .launch(f, g, [block, 1, 1], 0, &self.stream, &mut img)?;
            self.hip.record_event(&tm.ev[2 * i + 1], &self.stream)?;
            tm.log.push((sym.to_string(), g));
        } else {
            self.hip
                .launch(f, g, [block, 1, 1], 0, &self.stream, &mut img)?;
        }
        self.k.launches += 1;
        Ok(())
    }

    /// INK9: start timing every launch (event pairs).
    fn time_begin(&mut self) {
        self.timing = Some(OpTiming {
            ev: std::mem::take(&mut self.tpool),
            log: Vec::new(),
        });
    }

    /// INK9: stop timing; every launch since `time_begin` as (symbol, grid, device ms).
    fn time_end(&mut self) -> anyhow::Result<Vec<(String, [u32; 3], f32)>> {
        self.hip.sync_stream(&self.stream)?;
        let Some(t) = self.timing.take() else {
            return Ok(Vec::new());
        };
        let mut out = Vec::with_capacity(t.log.len());
        for (i, (s, g)) in t.log.iter().enumerate() {
            out.push((
                s.clone(),
                *g,
                self.hip.elapsed_ms(&t.ev[2 * i], &t.ev[2 * i + 1])?,
            ));
        }
        self.tpool = t.ev;
        Ok(out)
    }

    /// INK8: `ink_p2p_xchg` (sum = the reduce, else the head gather), the peer's operand read over
    /// P2P, the call-numbered handshake on the two ranks' link words.
    #[allow(clippy::too_many_arguments)]
    fn xchg(
        &mut self,
        own: (u64, u64),
        peer: (u64, u64),
        out: (u64, u64),
        rows: u64,
        n0: u64,
        n1: u64,
        sum: bool,
        link: (u64, u64),
    ) -> anyhow::Result<()> {
        let cols = if sum { n0 } else { n0 + n1 };
        // INK15: 16 bytes a peer load (`ink_pf`'s ink_p2p_xchg_v4, the parent's protocol and
        // values) when every stride and base allows it
        // (h4: at <= 8 rows -- the verify chain, the decode head gather -- it was not faster)
        let v4 = self.pf
            && rows > 8
            && cols % 4 == 0
            && n0 % 4 == 0
            && [own.1, peer.1, out.1].iter().all(|l| l % 4 == 0)
            && [own.0, peer.0, out.0].iter().all(|a| a % 16 == 0);
        // INK36 `ov`: the exchange beside the other half's compute takes at most `p36_xw`
        // workgroups (a grid-stride loop: the same element, the same one add -- value-neutral)
        let cap = if self.ov_live { self.p36_xw } else { 1024 };
        let wg = if v4 {
            (rows * cols / 4).div_ceil(256).clamp(1, cap)
        } else {
            (rows * cols).div_ceil(256).clamp(1, cap)
        };
        let link = (link.0 + self.link_off, link.1 + self.link_off);
        self.launch(
            if v4 {
                "ink_p2p_xchg_v4"
            } else {
                "ink_p2p_xchg"
            },
            [wg, 1, 1],
            256,
            &[
                A::P(own.0),
                pi(own.1),
                A::P(peer.0),
                pi(peer.1),
                A::P(out.0),
                pi(out.1),
                pi(rows),
                pi(n0),
                pi(n1),
                A::I(i32::from(sum)),
                A::P(link.0),
                A::P(link.1),
            ],
        )
    }

    // ---- bodies (the harness's `Card` methods)

    /// INK16: the column-lane cell for this GEMM, when `cl` is on, [`cl_pick`] names one and the
    /// body's load law holds (K % 64, A 16-byte aligned with lda = K % 4, weight rows aligned to
    /// `walign` bytes: 8 for the 8-bit record's 8-byte loads, 16 for bf16's); a picked cell that
    /// fails the law runs the pf tile and is counted (`cl_n[1]`).
    fn cl_cell(
        &mut self,
        a: u64,
        m: u64,
        n: u64,
        k: u64,
        w: u64,
        walign: u64,
    ) -> Option<(u64, u64)> {
        if !self.cl {
            return None;
        }
        let cell = cl_pick(m, n)?;
        let row_bytes = if walign == 8 { k } else { 2 * k };
        if k % 64 == 0 && k % 4 == 0 && a % 16 == 0 && w % walign == 0 && row_bytes % walign == 0 {
            self.cl_n[0] += 1;
            Some(cell)
        } else {
            self.cl_n[1] += 1;
            None
        }
    }

    /// INK19 (`cl8`): the column-lane cell r1n1 for a 2..8-row GEMV when `cl` is on and the load
    /// law holds (as [`Card::cl_cell`]; a miss keeps the GEMV, counted `cl_n[1]`).
    fn cl8_cell(&mut self, a: u64, k: u64, w: u64, walign: u64) -> Option<(u64, u64)> {
        if !self.cl {
            return None;
        }
        let row_bytes = if walign == 8 { k } else { 2 * k };
        if k % 64 == 0 && a % 16 == 0 && w % walign == 0 && row_bytes % walign == 0 {
            self.cl_n[0] += 1;
            Some((1, 1))
        } else {
            self.cl_n[1] += 1;
            None
        }
    }

    /// INK25: an 8-bit trunk GEMM of a prefill chunk on the matrix cores: the input quantized once
    /// a visit (int8 a 32-k block, `ink_act_q8_b32`; cached by (a, m, k) until the region resets),
    /// then `ink_gemm_i8_w8` (the contract in `hip/ink_i8.hip`). False = the load law fails (K %
    /// 128, 16-byte aligned A / W rows) -> the caller runs the f32 body (counted `i8_n[2]`).
    #[allow(clippy::too_many_arguments)]
    fn i8_gemm(
        &mut self,
        a: u64,
        m: u64,
        k: u64,
        w: &Tw,
        c: u64,
        ldc: u64,
        acc: bool,
    ) -> anyhow::Result<bool> {
        let n = w.rows();
        // INK29 subsets (numerics bisection): o = the [D, h x HD] row-split projection
        let is_o = n == D && k != D;
        if (self.p29 & P29_I8IN != 0 && is_o) || (self.p29 & P29_I8O != 0 && !is_o) {
            self.p29_n[5] += 1;
            return Ok(false);
        }
        if k % 128 != 0 || a % 16 != 0 || w.addr % 16 != 0 || w.sc % 8 != 0 || w.es != 1 {
            self.i8_n[2] += 1;
            return Ok(false);
        }
        let b16 = self.p29 & P29_B16 != 0;
        let hit = self
            .i8_cache
            .iter()
            .find(|e| e.0 == a && e.1 == m && e.2 == k)
            .map(|e| (e.3, e.4));
        let (qa, sa) = match hit {
            Some(x) => x,
            None => {
                let qa = self.take(Reg::Visit, m * k, "i8.qa")?;
                let sa = self.z4(Reg::Visit, m * k / if b16 { 16 } else { 32 }, "i8.sa")?;
                let qk = if b16 {
                    "ink29_q8_b16"
                } else if self.p29 & P29_LS != 0 {
                    "ink29_q8_ls"
                } else {
                    "ink_act_q8_b32_v2"
                };
                if qk != "ink_act_q8_b32_v2" {
                    self.p29_n[4] += 1;
                }
                self.launch(
                    qk,
                    [(m * k / 8).div_ceil(256), 1, 1],
                    256,
                    &[A::P(a), pi(k), A::P(qa), A::P(sa), pi(m), pi(k)],
                )?;
                self.i8_n[1] += 1;
                self.i8_cache.push((a, m, k, qa, sa));
                (qa, sa)
            }
        };
        let kx = if b16 || self.p29 & P29_KX == 0 {
            None
        } else {
            kx_slices(m, n, k)
        };
        let (sym, grid, block) = match kx {
            Some(sl) => {
                self.p29_n[0] += 1;
                (
                    format!("ink29_gemm_i8_kx{sl}"),
                    [n.div_ceil(16), m.div_ceil(16), 1],
                    32 * sl as u32,
                )
            }
            None => {
                let (g, b) = i8_grid(m, n);
                let s = if b16 {
                    "ink29_gemm_i8b16"
                } else {
                    "ink_gemm_i8_w8"
                };
                (s.to_string(), g, b)
            }
        };
        self.launch(
            &sym,
            grid,
            block,
            &[
                A::P(qa),
                A::P(sa),
                A::P(w.addr),
                pi(k),
                A::P(w.sc),
                pi(k / 32),
                A::P(c),
                pi(ldc),
                pi(m),
                pi(n),
                pi(k),
                A::I(i32::from(acc)),
            ],
        )?;
        self.i8_n[0] += 1;
        // the output never aliases a cached input; drop any that it overlaps anyway
        let end = c + 4 * ((m - 1) * ldc + n);
        self.i8_cache
            .retain(|e| e.0 + 4 * e.1 * e.2 <= c || e.0 >= end);
        Ok(true)
    }

    /// INK29 `bw`: a bf16 projection of a prefill chunk (the global layers) on the bf16 matrix
    /// cores (`ink29_gemm_bw_s<S>`, S = k / 768). False = the load law fails -> the f32 bodies.
    #[allow(clippy::too_many_arguments)]
    fn bw_gemm(
        &mut self,
        a: u64,
        m: u64,
        k: u64,
        w: &Tw,
        c: u64,
        ldc: u64,
        acc: bool,
    ) -> anyhow::Result<bool> {
        let n = w.rows();
        let Some(sl) = bw_slices(k) else {
            return Ok(false);
        };
        if a % 16 != 0 || w.addr % 16 != 0 || w.es != 2 {
            return Ok(false);
        }
        self.launch(
            &format!("ink29_gemm_bw_s{sl}"),
            [n.div_ceil(16), m.div_ceil(16), 1],
            32 * sl as u32,
            &[
                A::P(a),
                pi(k),
                A::P(w.addr),
                pi(k),
                A::P(c),
                pi(ldc),
                pi(m),
                pi(n),
                pi(k),
                A::I(i32::from(acc)),
            ],
        )?;
        self.p29_n[1] += 1;
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &mut self,
        a: u64,
        m: u64,
        k: u64,
        w: &Tw,
        c: u64,
        ldc: u64,
        acc: bool,
    ) -> anyhow::Result<()> {
        let n = w.rows();
        ensure!(
            w.shape.iter().skip(1).product::<u64>() == k && k % 32 == 0,
            "gemm K {k} vs {:?}",
            w.shape
        );
        let gemv = self.fast && m <= 8 && k % 256 == 0;
        // INK19 (`cl8`): 2..8 rows on the column-lane cell r1n1 (8 rows a workgroup, 32 columns)
        // below N 8,192 (h7 census: the local q..v GEMV at N 4,300 M 8 97 -> 71 us, M 2 43 -> 36;
        // the head half (N 100k) LOSES on it at every M: 979 -> 1,223 us at M 8)
        let cl8 = gemv && m >= 2 && n <= CL8_MAX_N && self.fuse & FUSE_CL8 != 0;
        ensure!(
            !gemv || w.addr % 16 == 0,
            "gemv: weight at {:#x} not 16-byte aligned",
            w.addr
        );
        if w.sc != 0 && !gemv && self.i8_live && self.i8_gemm(a, m, k, w, c, ldc, acc)? {
            return Ok(());
        }
        if w.sc == 0
            && !gemv
            && self.p29_live
            && self.p29 & P29_BW != 0
            && self.bw_gemm(a, m, k, w, c, ldc, acc)?
        {
            return Ok(());
        }
        if w.sc != 0 {
            // INK14: the 8-bit trunk record (int8 codes + bf16 scales a 32-k block); the same
            // routes as the bf16 weight (the 16-wide tile at N <= 1,024)
            ensure!(
                w.es == 1 && w.sc % 16 == 0,
                "gemm: an 8-bit record with es {}",
                w.es
            );
            let narrow =
                gemv && self.fuse & FUSE_GEMV != 0 && (n <= 1024 || self.fuse & FUSE_G16 != 0);
            let gw = if narrow { 16 } else { 64 };
            let cell8 = if cl8 {
                self.cl8_cell(a, k, w.addr, 8)
            } else {
                None
            };
            if (!gemv && self.pf) || cell8.is_some() {
                if let Some((rm, rn)) = cell8.or_else(|| self.cl_cell(a, m, n, k, w.addr, 8)) {
                    return self.launch(
                        &format!("ink_gemm_cl_f32_w8_r{rm}n{rn}"),
                        [n.div_ceil(32 * rn), m.div_ceil(8 * rm), 1],
                        256,
                        &[
                            A::P(a),
                            pi(k),
                            A::P(w.addr),
                            pi(k),
                            A::P(w.sc),
                            pi(k / 32),
                            A::P(c),
                            pi(ldc),
                            pi(m),
                            pi(n),
                            pi(k),
                            A::I(i32::from(acc)),
                        ],
                    );
                }
                let (tm, tn) = pf_tile(m, n);
                return self.launch(
                    &format!("ink_gemm_pf_f32_w8_t{tm}x{tn}"),
                    [n.div_ceil(tn), m.div_ceil(tm), 1],
                    256,
                    &[
                        A::P(a),
                        pi(k),
                        A::P(w.addr),
                        pi(k),
                        A::P(w.sc),
                        pi(k / 32),
                        A::P(c),
                        pi(ldc),
                        pi(m),
                        pi(n),
                        pi(k),
                        A::I(i32::from(acc)),
                    ],
                );
            }
            return self.launch(
                match (gemv, narrow) {
                    (true, true) => "ink_gemv_n16_f32_w8",
                    (true, false) => "ink_gemv_f32_w8",
                    _ => "ink_gemm_f32_w8",
                },
                [n.div_ceil(gw), if gemv { 1 } else { m.div_ceil(64) }, 1],
                256,
                &[
                    A::P(a),
                    pi(k),
                    A::P(w.addr),
                    pi(k),
                    A::P(w.sc),
                    pi(k / 32),
                    A::P(c),
                    pi(ldc),
                    pi(m),
                    pi(n),
                    pi(k),
                    A::I(i32::from(acc)),
                ],
            );
        }
        // INK10 task 2: the narrow tiles (16 / 32 columns a workgroup, three k-tiles in flight)
        // (h2: the 32-wide tile LOST at N 3,328 / 4,352 -- those GEMVs already sit near the
        // capped read rate; the 16-wide one wins at N 512 / 576, 8 -> 32 workgroups)
        let narrow = gemv && self.fuse & FUSE_GEMV != 0 && (k / 256) % 3 == 0 && n <= 1024;
        let gw = if narrow { 16 } else { 64 };
        let cell8 = if cl8 {
            self.cl8_cell(a, k, w.addr, 16)
        } else {
            None
        };
        if (!gemv && self.pf) || cell8.is_some() {
            if let Some((rm, rn)) = cell8.or_else(|| self.cl_cell(a, m, n, k, w.addr, 16)) {
                return self.launch(
                    &format!("ink_gemm_cl_f32_bf16_r{rm}n{rn}"),
                    [n.div_ceil(32 * rn), m.div_ceil(8 * rm), 1],
                    256,
                    &[
                        A::P(a),
                        pi(k),
                        A::P(w.addr),
                        pi(k),
                        A::P(c),
                        pi(ldc),
                        pi(m),
                        pi(n),
                        pi(k),
                        A::I(i32::from(acc)),
                    ],
                );
            }
            let (tm, tn) = pf_tile(m, n);
            return self.launch(
                &format!("ink_gemm_pf_f32_bf16_t{tm}x{tn}"),
                [n.div_ceil(tn), m.div_ceil(tm), 1],
                256,
                &[
                    A::P(a),
                    pi(k),
                    A::P(w.addr),
                    pi(k),
                    A::P(c),
                    pi(ldc),
                    pi(m),
                    pi(n),
                    pi(k),
                    A::I(i32::from(acc)),
                ],
            );
        }
        self.launch(
            match (gemv, narrow, gw) {
                (true, true, _) => "ink_gemv_n16_f32_bf16",
                (true, false, _) => "ink_gemv_f32_bf16",
                _ => "ink_gemm_f32_bf16",
            },
            [n.div_ceil(gw), if gemv { 1 } else { m.div_ceil(64) }, 1],
            256,
            &[
                A::P(a),
                pi(k),
                A::P(w.addr),
                pi(k),
                A::P(c),
                pi(ldc),
                pi(m),
                pi(n),
                pi(k),
                A::I(i32::from(acc)),
            ],
        )
    }

    /// INK11: `ink_gemv3_f32_bf16` -- up to three `(a, w, c, n)` GEMVs (`a` `[m, D]` f32, `c`
    /// `[m, n]` f32, row stride n) in one launch.
    fn gemv3(&mut self, m: u64, segs: &[(u64, &Tw, u64, u64)]) -> anyhow::Result<()> {
        ensure!(
            (1..=3).contains(&segs.len()) && m <= 8 && self.fast,
            "gemv3: {} segments at m {m}",
            segs.len()
        );
        if m >= CL8_GEMV3_MIN_M && segs[0].1.sc == 0 && self.fuse & FUSE_CL8 != 0 && self.cl {
            // INK19 (`cl8`): each segment its own column-lane launch (each output the same sum) --
            // the bf16 triple at 8 rows only (h7: 136 -> 106 us; the 8-bit triple and fewer rows
            // lose to the one-launch GEMV: three launches)
            for &(a, w, c, n) in segs {
                self.gemm(a, m, D, w, c, n, false)?;
            }
            return Ok(());
        }
        let mut blob = Vec::with_capacity(120);
        let get = |i: usize| segs.get(i).copied();
        // INK14: the 8-bit trunk's form carries the scale pointers after the code pointers
        let w8 = segs[0].1.sc != 0;
        ensure!(
            segs.iter().all(|s| (s.1.sc != 0) == w8),
            "gemv3: mixed bf16 / 8-bit segments"
        );
        for i in 0..3 {
            blob.extend_from_slice(&get(i).map_or(0, |s| s.0).to_le_bytes());
        }
        for i in 0..3 {
            let w = get(i).map_or(0, |s| s.1.addr);
            ensure!(w % 16 == 0, "gemv3: weight at {w:#x} not 16-byte aligned");
            blob.extend_from_slice(&w.to_le_bytes());
        }
        if w8 {
            for i in 0..3 {
                let w = get(i).map_or(0, |s| s.1.sc);
                ensure!(w % 16 == 0, "gemv3: scales at {w:#x} not 16-byte aligned");
                blob.extend_from_slice(&w.to_le_bytes());
            }
        }
        for i in 0..3 {
            blob.extend_from_slice(&get(i).map_or(0, |s| s.2).to_le_bytes());
        }
        let mut tiles = 0;
        for i in 0..3 {
            let n = get(i).map_or(0, |s| s.3);
            if let Some(sg) = get(i) {
                ensure!(
                    sg.1.rows() == n && sg.1.shape.iter().skip(1).product::<u64>() == D,
                    "gemv3 shape"
                );
            }
            tiles += n.div_ceil(64);
            blob.extend_from_slice(&u32::try_from(n)?.to_le_bytes());
        }
        for i in 0..3 {
            blob.extend_from_slice(&u32::try_from(get(i).map_or(0, |s| s.3))?.to_le_bytes());
        }
        self.launch(
            if w8 {
                "ink_gemv3_f32_w8"
            } else {
                "ink_gemv3_f32_bf16"
            },
            [tiles, 1, 1],
            256,
            &[A::Blob(blob), pi(D), pi(m), pi(D)],
        )
    }

    /// INK11: `ink_depth_mq` -- the depth attend over `states` for `4 g` queries `q0 + j D`
    /// (no gains) into `o0 + j t D` (pass region).
    fn depth_mq(&mut self, t: u64, states: &[u64], q0: u64, g: u64) -> anyhow::Result<u64> {
        ensure!(states.len() <= 8 && t <= 8, "depth_mq");
        let o0 = self.z4(Reg::Pass, 4 * g * t * D, "msg.all")?;
        let mut blob = Vec::with_capacity(64);
        for i in 0..8 {
            blob.extend_from_slice(&states.get(i).copied().unwrap_or(0).to_le_bytes());
        }
        self.launch(
            "ink_depth_mq",
            [t, g, 1],
            256,
            &[
                A::Blob(blob),
                pi(states.len() as u64),
                A::P(q0),
                pi(D),
                A::P(o0),
                A::L(i64::try_from(t * D)?),
                pi(D),
                pi(D),
                A::F(EPS),
            ],
        )?;
        Ok(o0)
    }

    /// `depth(states, queries, gains)` -> one `[T, D]` output a query (visit region).
    fn depth(
        &mut self,
        t: u64,
        states: &[u64],
        queries: &[u64],
        gains: Option<&[u64]>,
    ) -> anyhow::Result<Vec<u64>> {
        ensure!(
            states.len() <= 8 && queries.len() <= 4,
            "depth: {} states {} queries",
            states.len(),
            queries.len()
        );
        let nq = queries.len();
        let mut outs = Vec::with_capacity(nq);
        for _ in 0..nq {
            outs.push(self.z4(Reg::Visit, t * D, "depth.out")?);
        }
        let mut blob = Vec::with_capacity(160);
        for i in 0..8 {
            blob.extend_from_slice(&states.get(i).copied().unwrap_or(0).to_le_bytes());
        }
        for i in 0..4 {
            blob.extend_from_slice(&queries.get(i).copied().unwrap_or(0).to_le_bytes());
        }
        for i in 0..4 {
            blob.extend_from_slice(&outs.get(i).copied().unwrap_or(0).to_le_bytes());
        }
        for i in 0..4 {
            let g = gains.and_then(|g| g.get(i).copied()).unwrap_or(0);
            blob.extend_from_slice(&g.to_le_bytes());
        }
        self.launch(
            "ink_depth_attend",
            [t, 1, 1],
            256,
            &[
                A::Blob(blob),
                pi(states.len() as u64),
                pi(nq as u64),
                pi(D),
                pi(D),
                A::F(EPS),
            ],
        )?;
        Ok(outs)
    }

    fn rmsnorm(
        &mut self,
        reg: Reg,
        x: u64,
        rows: u64,
        cols: u64,
        gain: u64,
    ) -> anyhow::Result<u64> {
        let o = self.z4(reg, rows * cols, "rmsnorm.out")?;
        self.launch(
            "ink_rmsnorm",
            [rows, 1, 1],
            256,
            &[
                A::P(x),
                pi(cols),
                A::P(gain),
                A::P(o),
                pi(cols),
                pi(cols),
                A::F(EPS),
            ],
        )?;
        Ok(o)
    }

    fn msg_gate(&mut self, h: u64, rows: u64, m: u64, g: &Gate) -> anyhow::Result<u64> {
        let o = self.z4(Reg::Visit, rows * D, "msg.out")?;
        if self.fast {
            // INK9: ink_msg_z + ink_msg_out (the parent's order, byte-equal)
            let z = self.z4(Reg::Visit, rows * 64, "msg.z")?;
            if rows > 8 && self.p35_live && self.p35 & P35_MZS != 0 {
                // INK35 `mzs`: the normed pair, then the hi + lo bf16 matrix-core GEMM
                let nz = self.take(Reg::Visit, rows * 2 * D * 4, "msg.norm2")?;
                self.launch(
                    "ink29_msg_norm2",
                    [rows, 1, 1],
                    256,
                    &[A::P(h), pi(D), A::P(m), pi(D), A::P(nz), A::F(EPS)],
                )?;
                self.launch(
                    "ink35_gemm_bws_s6",
                    [64 / 16, rows.div_ceil(16), 1],
                    32 * 6,
                    &[
                        A::P(nz),
                        pi(2 * D),
                        A::P(g.down),
                        pi(2 * D),
                        A::P(z),
                        pi(64),
                        pi(rows),
                        pi(64),
                        pi(2 * D),
                        A::I(0),
                    ],
                )?;
                self.p35_n += 1;
            } else if rows > 8 && self.p29 & P29_MZB != 0 {
                // INK29 `mzb`: the normed pair, then the bf16 matrix-core GEMM (K 4,608 = 6 slices)
                let nz = self.take(Reg::Visit, rows * 2 * D * 4, "msg.norm2")?;
                self.launch(
                    "ink29_msg_norm2",
                    [rows, 1, 1],
                    256,
                    &[A::P(h), pi(D), A::P(m), pi(D), A::P(nz), A::F(EPS)],
                )?;
                self.launch(
                    "ink29_gemm_bw_s6",
                    [64 / 16, rows.div_ceil(16), 1],
                    32 * 6,
                    &[
                        A::P(nz),
                        pi(2 * D),
                        A::P(g.down),
                        pi(2 * D),
                        A::P(z),
                        pi(64),
                        pi(rows),
                        pi(64),
                        pi(2 * D),
                        A::I(0),
                    ],
                )?;
                self.p29_n[1] += 1;
            } else {
                // INK29 `mz`: the same bytes, the Wd windows prefetched (prefill rows)
                self.launch(
                    if rows > 8 && self.p29 & P29_MZ != 0 {
                        "ink29_msg_z_p"
                    } else {
                        "ink_msg_z"
                    },
                    [rows, 8, 1],
                    256,
                    &[
                        A::P(h),
                        pi(D),
                        A::P(m),
                        pi(D),
                        A::P(g.down),
                        A::P(z),
                        A::F(EPS),
                    ],
                )?;
            }
            if rows > 8 && self.p35_live && self.p37 & (P37_GLUE | P37_GLUE8) != 0 {
                // INK37 `glue`: the same bytes, the column's weights loaded once per 8 rows
                self.p37_n[1] += 1;
                self.launch(
                    "ink37_msg_out_r8",
                    [D / 64, rows.div_ceil(8), 1],
                    64,
                    &[
                        A::P(h),
                        pi(D),
                        A::P(m),
                        pi(D),
                        A::P(z),
                        A::P(g.forget),
                        A::P(g.gain),
                        A::P(o),
                        pi(D),
                        pi(D),
                        pi(rows),
                    ],
                )?;
                return Ok(o);
            }
            self.launch(
                "ink_msg_out",
                [D / 64, rows, 1],
                64,
                &[
                    A::P(h),
                    pi(D),
                    A::P(m),
                    pi(D),
                    A::P(z),
                    A::P(g.forget),
                    A::P(g.gain),
                    A::P(o),
                    pi(D),
                ],
            )?;
            return Ok(o);
        }
        self.launch(
            "ink_msg_gate",
            [rows, 1, 1],
            256,
            &[
                A::P(h),
                pi(D),
                A::P(m),
                pi(D),
                A::P(g.down),
                A::P(g.forget),
                A::P(g.gain),
                A::P(o),
                pi(D),
                A::F(EPS),
            ],
        )?;
        Ok(o)
    }

    /// INK10: `ink_xsd` -- (the push reduce's consumer: own + the peer's pushed rows from site
    /// `xs = (site, flags, per_row)`) + the sconv of row t at state slot `rseq[t]` (+ resid) into
    /// `o`, then the depth attend over `states` (the last one is `o`) for `queries` into `outs`.
    #[allow(clippy::too_many_arguments)]
    fn xsd(
        &mut self,
        own: u64,
        xs: Option<(u64, u64, bool)>,
        t: u64,
        w: u64,
        st: u64,
        rseq: u64,
        resid: u64,
        o: u64,
        states: &[u64],
        queries: &[u64],
        gains: Option<&[u64]>,
        outs: &[u64],
        chain: Option<(u64, bool)>,
    ) -> anyhow::Result<()> {
        ensure!(
            states.len() <= 8 && queries.len() <= 4 && outs.len() == queries.len() && t <= 8,
            "xsd: {} states {} queries t {t}",
            states.len(),
            queries.len()
        );
        ensure!(self.fz != 0, "xsd: no fuse words");
        let mut blob = Vec::with_capacity(160);
        for i in 0..8 {
            blob.extend_from_slice(&states.get(i).copied().unwrap_or(0).to_le_bytes());
        }
        for i in 0..4 {
            blob.extend_from_slice(&queries.get(i).copied().unwrap_or(0).to_le_bytes());
        }
        for i in 0..4 {
            blob.extend_from_slice(&outs.get(i).copied().unwrap_or(0).to_le_bytes());
        }
        for i in 0..4 {
            let g = gains.and_then(|g| g.get(i).copied()).unwrap_or(0);
            blob.extend_from_slice(&g.to_le_bytes());
        }
        let (site, nfl, per_row) = xs.unwrap_or((0, 0, false));
        let sb = self.fz + site * FZ_SITE;
        let mut args = vec![
            A::P(own),
            pi(D),
            A::P(sb),
            pi(D),
            A::P(sb + FZ_FLAGS),
            A::P(sb + FZ_CTR),
            pi(nfl),
            A::I(i32::from(per_row)),
            A::I(i32::from(xs.is_some())),
            A::P(w),
            A::P(st),
            A::P(rseq),
            A::P(resid),
            pi(D),
            A::P(o),
            pi(D),
            A::Blob(blob),
            pi(states.len() as u64),
            pi(queries.len() as u64),
            pi(D),
            pi(D),
            A::F(EPS),
            A::P(self.fz + FZ_DIAG),
        ];
        let sym = if let Some((hist, commit)) = chain {
            // INK17: the chained rows of one sequence (`ink_xsdc`): + the history entry + the
            // last arriver's counter (none when the verify's commit writes the state)
            args.extend([A::P(hist), A::P(if commit { 0 } else { self.fz + FZ_VT })]);
            if self.fuse & FUSE_XL != 0 {
                "ink_xsdc_l"
            } else if self.fuse & FUSE_WT != 0 {
                args.push(A::P(self.fz + FZ_WT));
                "ink_xsdc_pw"
            } else if self.fuse & FUSE_P30 != 0 {
                "ink_xsdc_p"
            } else {
                "ink_xsdc"
            }
        } else if self.fuse & FUSE_XL != 0 {
            "ink_xsd_l"
        } else if self.fuse & FUSE_WT != 0 {
            args.push(A::P(self.fz + FZ_WT));
            "ink_xsd_pw"
        } else if self.fuse & FUSE_P30 != 0 {
            "ink_xsd_p"
        } else if self.fuse & FUSE_XS2 != 0 {
            "ink_xsd2"
        } else {
            "ink_xsd"
        };
        self.launch(sym, [t, 1, 1], 256, &args)
    }

    /// INK10: up to three message gates `(h, m, gate, gain)` in two launches (`ink_msg_z3` +
    /// `ink_msg_out3n`); a gate with a gain returns its in-normed row, else the raw gate row.
    fn msg_gates(
        &mut self,
        t: u64,
        g: &[(u64, u64, Gate, Option<u64>)],
    ) -> anyhow::Result<Vec<u64>> {
        ensure!(
            !g.is_empty() && g.len() <= 3 && t <= 8 && self.fz != 0,
            "msg_gates"
        );
        let n = g.len();
        let mut zs = Vec::with_capacity(n);
        let mut os = Vec::with_capacity(n);
        let mut nos = Vec::with_capacity(n);
        for (_, _, _, gain) in g {
            zs.push(self.z4(Reg::Visit, t * 64, "msg.z")?);
            os.push(self.z4(Reg::Visit, t * D, "msg.out")?);
            nos.push(if gain.is_some() {
                self.z4(Reg::Visit, t * D, "rmsnorm.out")?
            } else {
                0
            });
        }
        let put = |v: &mut Vec<u8>, f: &dyn Fn(usize) -> u64| {
            for i in 0..3 {
                v.extend_from_slice(&(if i < n { f(i) } else { 0 }).to_le_bytes());
            }
        };
        let mut zb = Vec::with_capacity(96);
        put(&mut zb, &|i| g[i].0);
        put(&mut zb, &|i| g[i].1);
        put(&mut zb, &|i| g[i].2.down);
        put(&mut zb, &|i| zs[i]);
        let p30 = self.fuse & FUSE_P30 != 0;
        self.launch(
            if p30 { "ink_msg_z3_p" } else { "ink_msg_z3" },
            [t, 8, n as u64],
            256,
            &[A::Blob(zb), pi(D), A::F(EPS)],
        )?;
        let mut ob = Vec::with_capacity(192);
        put(&mut ob, &|i| g[i].0);
        put(&mut ob, &|i| g[i].1);
        put(&mut ob, &|i| zs[i]);
        put(&mut ob, &|i| g[i].2.forget);
        put(&mut ob, &|i| g[i].2.gain);
        put(&mut ob, &|i| os[i]);
        put(&mut ob, &|i| g[i].3.unwrap_or(0));
        put(&mut ob, &|i| nos[i]);
        self.launch(
            if p30 {
                "ink_msg_out3n_p"
            } else {
                "ink_msg_out3n"
            },
            [D / 64, t, n as u64],
            64,
            &[A::Blob(ob), pi(D), A::F(EPS), A::P(self.fz + FZ_MSG)],
        )?;
        Ok((0..n)
            .map(|i| if g[i].3.is_some() { nos[i] } else { os[i] })
            .collect())
    }

    /// INK10: the spin-timeout word of the push reduce (0 expected).
    pub fn fuse_diag(&self) -> anyhow::Result<u32> {
        if self.fz == 0 {
            return Ok(0);
        }
        let b = self.read(self.fz + FZ_DIAG, 4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// INK30 task 4: the `wt` stamps since the last call (spin ticks, kernel ticks, rows), zeroed.
    pub fn fuse_wt_take(&self) -> anyhow::Result<[u64; 3]> {
        if self.fz == 0 || self.fuse & FUSE_WT == 0 {
            return Ok([0; 3]);
        }
        let b = self.read(self.fz + FZ_WT, 24)?;
        self.write(self.fz + FZ_WT, &[0u8; 24])?;
        let w = |i: usize| u64::from_le_bytes(b[8 * i..8 * i + 8].try_into().unwrap_or([0; 8]));
        Ok([w(0), w(1), w(2)])
    }

    /// INK9 task 4: `ink_sconv_rows` (one token a row, the row's state slot `rseq[t]`).
    #[allow(clippy::too_many_arguments)]
    fn sconv_rows(
        &mut self,
        x: u64,
        t: u64,
        w: u64,
        st: u64,
        rseq: u64,
        resid: u64,
        reg: Reg,
    ) -> anyhow::Result<u64> {
        let o = self.z4(reg, t * D, "sconv.out")?;
        self.launch(
            "ink_sconv_rows",
            [D.div_ceil(256), t, 1],
            256,
            &[
                A::P(x),
                pi(D),
                A::P(w),
                A::P(st),
                A::P(rseq),
                A::P(resid),
                pi(D),
                A::P(o),
                pi(D),
                pi(t),
                pi(D),
            ],
        )?;
        Ok(o)
    }

    #[allow(clippy::too_many_arguments)]
    fn sconv(
        &mut self,
        reg: Reg,
        x: u64,
        t: u64,
        w: u64,
        st_in: u64,
        resid: u64,
        st_out: u64,
    ) -> anyhow::Result<u64> {
        let c = D;
        let o = self.z4(reg, t * c, "sconv.out")?;
        self.launch(
            "ink_sconv_resid",
            [c.div_ceil(256), t.div_ceil(64), 1],
            256,
            &[
                A::P(x),
                pi(c),
                A::P(w),
                A::P(st_in),
                A::P(st_out),
                A::P(resid),
                pi(c),
                A::P(o),
                pi(c),
                pi(t),
                pi(c),
            ],
        )?;
        Ok(o)
    }
}

impl Card<'_, '_> {
    /// INK20: `ink_mv_sconv` -- B sequences x `w` chained rows (row t = sequence t / w), each
    /// row's conv over its own sequence's rows and its slot's state (no state write).
    #[allow(clippy::too_many_arguments)]
    fn sconv_m(
        &mut self,
        x: u64,
        t: u64,
        w: u64,
        wt: u64,
        st: u64,
        rseq: u64,
        resid: u64,
        reg: Reg,
    ) -> anyhow::Result<u64> {
        ensure!(
            w >= 1 && t % w == 0,
            "a {t}-row multi verify of {w}-row sequences"
        );
        let o = self.z4(reg, t * D, "sconv.out")?;
        self.launch(
            "ink_mv_sconv",
            [D.div_ceil(256), t, 1],
            256,
            &[
                A::P(x),
                pi(D),
                A::P(wt),
                A::P(st),
                A::P(rseq),
                A::P(resid),
                pi(D),
                A::P(o),
                pi(D),
                pi(w),
                pi(D),
            ],
        )?;
        Ok(o)
    }

    /// INK20: log one conv of a multi-sequence verify -- the states `[slots][3][cw]` at `st`
    /// (a sequence's at its slot `rseq[b w]`), the rows of `x`; entry b at `at + b (3 + w) cw`;
    /// the commit (`ink_mv_commit`) writes state base `dst` at each sequence's own keep.
    #[allow(clippy::too_many_arguments)]
    fn stash_m(
        &mut self,
        st: u64,
        rseq: u64,
        x: u64,
        ldx: u64,
        t: u64,
        w: u64,
        cw: u64,
        dst: u64,
    ) -> anyhow::Result<()> {
        let Some(h) = self.hist.as_mut() else {
            bail!("a multi-sequence verify without a history");
        };
        ensure!(
            w >= 1 && t % w == 0,
            "a {t}-row multi verify of {w}-row sequences"
        );
        let b = t / w;
        let bytes = b * (3 + w) * cw * 4;
        ensure!(h.off + bytes <= h.cap, "the verify history is full");
        let at = h.buf + h.off;
        h.off += bytes;
        h.entries.extend_from_slice(&[at, dst, cw]);
        h.max_w = h.max_w.max(cw);
        self.launch(
            "ink_mv_hist",
            [cw.div_ceil(256), 3 + w, b],
            256,
            &[
                A::P(st),
                A::P(rseq),
                A::P(x),
                pi(ldx),
                pi(cw),
                pi(w),
                A::P(at),
            ],
        )
    }

    /// INK20: roll every logged conv of a multi-sequence verify (`b` sequences x `w` rows, the
    /// table `tab` of `n` convs of width <= `max_w`) back to each sequence's own keep (`keeps`
    /// already written at `Cache::keep_m`).
    fn commit_m_launch(
        &mut self,
        cache: &Cache,
        tab: u64,
        n: u64,
        max_w: u64,
        b: u64,
        w: u64,
    ) -> anyhow::Result<()> {
        self.launch(
            "ink_mv_commit",
            [max_w.div_ceil(256), n, b],
            256,
            &[A::P(tab), A::P(cache.keep_m), A::P(cache.rseq_m), pi(w)],
        )
    }

    /// INK20: the eager multi-sequence commit: write the keeps, the log's table, launch, clear.
    fn commit_m(&mut self, cache: &Cache, keeps: &[u32], w: u64) -> anyhow::Result<()> {
        let mut h = self.hist.take().context("no verify history")?;
        ensure!(
            keeps.iter().all(|&k| (1..=w).contains(&u64::from(k))),
            "keeps {keeps:?} of a {w}-row verify"
        );
        let n = h.entries.len() as u64 / 3;
        let tab: Vec<u8> = h.entries.iter().flat_map(|v| v.to_le_bytes()).collect();
        ensure!(
            tab.len() as u64 <= 3 * 8 * 512,
            "the history table holds {n} convs"
        );
        self.write(h.tab, &tab)?;
        let kb: Vec<u8> = keeps.iter().flat_map(|k| k.to_le_bytes()).collect();
        self.write(cache.keep_m, &kb)?;
        if n > 0 {
            self.commit_m_launch(cache, h.tab, n, h.max_w, keeps.len() as u64, w)?;
        }
        h.off = 0;
        h.entries.clear();
        h.max_w = 0;
        self.hist_spare = Some(h);
        Ok(())
    }
}

// ============================================================================================ model

/// A message gate's three projections.
#[derive(Debug, Clone, Copy)]
struct Gate {
    down: u64,
    forget: u64,
    gain: u64,
}

/// One rank's attention weights of a layer.
struct RankW {
    q: Tw,
    k: Tw,
    v: Tw,
    r: Tw,
    sink: Tw,
    gate: Tw,
    o: Tw,
    q_scale: u64,
    rel_scale: u64,
    wk: u64,
    wv: u64,
    pk: Option<Tw>,
    pv: Option<Tw>,
    h: u64,
    kv: u64,
    /// INK9: the projections laid out contiguously ([q r sink gate k (pk) v (pv)] rows, one
    /// upload): `qg` = q..gate (the q-input's), `kg` = k (+ pk), `vg` = v (+ pv), `all` = q..v
    /// (local layers: one GEMV when the three inputs are one row)
    qg: Tw,
    kg: Tw,
    vg: Tw,
    all: Option<Tw>,
}

struct LayerW {
    attn_sconv: u64,
    mlp_sconv: u64,
    post_ln: u64,
    mlp_q: Tw,
    router: u64,
    bias: u64,
    rel_proj: u64,
    attn_q: Option<Tw>,
    ln: [u64; 3],
    gates: Option<[Gate; 4]>,
    r: Vec<RankW>,
}

struct Weights {
    tp: u32,
    embed_norm: u64,
    norm: u64,
    final_q: Tw,
    msg_q: Tw,
    layers: Vec<LayerW>,
    head: Vec<Tw>,
}

/// INK14: a dense weight on the card -- the 8-bit record when the overlay covers it (`head` =
/// the `lm_head`, only under `w8-head`), else the bf16 tensor.
fn put_dense(
    card: &mut Card<'_, '_>,
    pk: &PackReader,
    name: &str,
    rank: Option<u32>,
    head: bool,
) -> anyhow::Result<Tw> {
    if let Some((codes, sc, shape)) = pk.tensor_w8(name, rank, head)? {
        let addr = card.up(&codes)?;
        let sc = card.up(&sc)?;
        ensure!(
            addr % 16 == 0 && sc % 16 == 0,
            "trunk8 {name}: unaligned upload"
        );
        return Ok(Tw {
            addr,
            es: 1,
            shape,
            sc,
        });
    }
    put(card, pk.tensor(name, rank)?)
}

fn put(card: &mut Card<'_, '_>, t: (Vec<u8>, u64, Vec<u64>)) -> anyhow::Result<Tw> {
    let addr = card.up(&t.0)?;
    Ok(Tw {
        addr,
        es: t.1,
        shape: t.2,
        sc: 0,
    })
}

impl Weights {
    /// `only` = the two-card process's rank (its halves only).
    fn load(
        pk: &PackReader,
        card: &mut Card<'_, '_>,
        tp: u32,
        only: Option<u32>,
    ) -> anyhow::Result<Self> {
        let g = |n: &str, card: &mut Card<'_, '_>| -> anyhow::Result<Tw> {
            put(card, pk.tensor(n, None)?)
        };
        let embed_norm = g("model.embed_norm.weight", card)?.addr;
        let norm = g("model.norm.weight", card)?.addr;
        let final_q = g("model.final_input.query", card)?;
        let msg_q = g("model.message_input.query", card)?;
        let mut layers = Vec::new();
        for l in 0..24u32 {
            let n = |s: &str| format!("model.layers.{l}.{s}");
            let glob = GLOBAL.contains(&l);
            let ln = if glob {
                [
                    g(&n("q_input_layernorm.weight"), card)?.addr,
                    g(&n("k_input_layernorm.weight"), card)?.addr,
                    g(&n("v_input_layernorm.weight"), card)?.addr,
                ]
            } else {
                let a = g(&n("input_layernorm.weight"), card)?.addr;
                [a, a, a]
            };
            let gates = if (CS..CE).contains(&l) {
                let mut gs = [Gate {
                    down: 0,
                    forget: 0,
                    gain: 0,
                }; 4];
                for (i, w) in ["q", "k", "v", "mlp"].iter().enumerate() {
                    gs[i] = Gate {
                        down: g(&n(&format!("{w}_message_gate.down_proj.weight")), card)?.addr,
                        forget: g(
                            &n(&format!("{w}_message_gate.forget_gate_proj.weight")),
                            card,
                        )?
                        .addr,
                        gain: g(
                            &n(&format!("{w}_message_gate.message_gate_proj.weight")),
                            card,
                        )?
                        .addr,
                    };
                }
                Some(gs)
            } else {
                None
            };
            let ranks: Vec<Option<u32>> = if tp == 1 {
                vec![None]
            } else if let Some(r) = only {
                vec![Some(r)]
            } else {
                vec![Some(0), Some(1)]
            };
            let mut r = Vec::new();
            for rk in ranks {
                let a = |s: &str, card: &mut Card<'_, '_>| -> anyhow::Result<Tw> {
                    put(
                        card,
                        pk.tensor(&format!("model.layers.{l}.self_attn.{s}"), rk)?,
                    )
                };
                // INK9: q, r, sink, gate, k, (pk), v, (pv) as ONE upload (row blocks of the same
                // K), each tensor a view into it -- the same bytes at new addresses
                let tn = |s: &str| -> anyhow::Result<(Vec<u8>, u64, Vec<u64>)> {
                    pk.tensor(&format!("model.layers.{l}.self_attn.{s}"), rk)
                };
                let mut names = vec![
                    "q_proj.weight",
                    "r_proj.weight",
                    "sink_proj.weight",
                    "gate_proj.weight",
                    "k_proj.weight",
                ];
                if glob {
                    names.push("k_sconv.weight_proj.weight");
                }
                names.push("v_proj.weight");
                if glob {
                    names.push("v_sconv.weight_proj.weight");
                }
                // INK14: the 8-bit trunk -- the same row blocks as int8 codes + bf16 scales (two
                // uploads, the views address both)
                let mut parts = Vec::new();
                let mut w8parts = Vec::new();
                for nm in &names {
                    let full = format!("model.layers.{l}.self_attn.{nm}");
                    match pk.tensor_w8(&full, rk, false)? {
                        Some(t) => w8parts.push(t),
                        None => parts.push(tn(nm)?),
                    }
                }
                ensure!(
                    w8parts.is_empty() || parts.is_empty(),
                    "layer {l}: a partial 8-bit projection set"
                );
                let rows_of: Vec<u64> = if w8parts.is_empty() {
                    parts.iter().map(|p| p.2[0]).collect()
                } else {
                    w8parts.iter().map(|p| p.2[0]).collect()
                };
                let (base, sbase, offs, soffs, es) = if w8parts.is_empty() {
                    let mut blob = Vec::new();
                    let mut offs = Vec::new();
                    for (b, es, shape) in &parts {
                        ensure!(
                            *es == 2 && shape.len() == 2 && shape[1] == D && b.len() % 16 == 0,
                            "layer {l}: an attention projection {shape:?} / {es} B is not a bf16 [n, {D}]"
                        );
                        offs.push(blob.len() as u64);
                        blob.extend_from_slice(b);
                    }
                    (card.up(&blob)?, 0, offs, vec![0; parts.len()], 2)
                } else {
                    let (mut cb, mut sb) = (Vec::new(), Vec::new());
                    let (mut offs, mut soffs) = (Vec::new(), Vec::new());
                    for (codes, sc, shape) in &w8parts {
                        ensure!(
                            shape[1] == D && codes.len() % 16 == 0 && sc.len() % 16 == 0,
                            "layer {l}: an 8-bit projection {shape:?} is not [n, {D}]"
                        );
                        offs.push(cb.len() as u64);
                        soffs.push(sb.len() as u64);
                        cb.extend_from_slice(codes);
                        sb.extend_from_slice(sc);
                    }
                    (card.up(&cb)?, card.up(&sb)?, offs, soffs, 1)
                };
                let view = |i: usize, j: usize| -> Tw {
                    let rows: u64 = rows_of[i..j].iter().sum();
                    Tw {
                        addr: base + offs[i],
                        es,
                        shape: vec![rows, D],
                        sc: if sbase == 0 { 0 } else { sbase + soffs[i] },
                    }
                };
                let q = view(0, 1);
                let rr = view(1, 2);
                let sink = view(2, 3);
                let gate = view(3, 4);
                let k = view(4, 5);
                let (pk_i, v_i) = if glob { (Some(5), 6) } else { (None, 5) };
                let v = view(v_i, v_i + 1);
                let qg = view(0, 4);
                let kg = view(4, v_i);
                let vg = view(v_i, names.len());
                let all = (!glob).then(|| view(0, names.len()));
                let pk_view = pk_i.map(|i| view(i, i + 1));
                let pv_view = glob.then(|| view(7, 8));
                let o = put_dense(
                    card,
                    pk,
                    &format!("model.layers.{l}.self_attn.o_proj.weight"),
                    rk,
                    false,
                )?;
                let q_scale = a("q_scale", card)?.addr;
                let rel_scale = a("rel_scale", card)?.addr;
                let (wk, wv, pkk, pvv) = if glob {
                    (
                        a("k_sconv.static_weight", card)?.addr,
                        a("v_sconv.static_weight", card)?.addr,
                        pk_view,
                        pv_view,
                    )
                } else {
                    (
                        a("k_sconv.conv.weight", card)?.addr,
                        a("v_sconv.conv.weight", card)?.addr,
                        None,
                        None,
                    )
                };
                let (h, kv) = (q.rows() / HD, k.rows() / HD);
                r.push(RankW {
                    q,
                    k,
                    v,
                    r: rr,
                    sink,
                    gate,
                    o,
                    q_scale,
                    rel_scale,
                    wk,
                    wv,
                    pk: pkk,
                    pv: pvv,
                    h,
                    kv,
                    qg,
                    kg,
                    vg,
                    all,
                });
            }
            layers.push(LayerW {
                attn_sconv: g(&n("attn_sconv.conv.weight"), card)?.addr,
                mlp_sconv: g(&n("mlp_sconv.conv.weight"), card)?.addr,
                post_ln: g(&n("post_attention_layernorm.weight"), card)?.addr,
                mlp_q: g(&n("mlp_input.query"), card)?,
                router: g(&n("mlp.gate_weight"), card)?.addr,
                bias: g(&n("mlp.e_score_correction_bias"), card)?.addr,
                rel_proj: g(&n("self_attn.rel_proj"), card)?.addr,
                attn_q: if l > 0 {
                    Some(g(&n("attention_input.query"), card)?)
                } else {
                    None
                },
                ln,
                gates,
                r,
            });
        }
        let head = if tp == 1 {
            vec![put(card, pk.tensor("lm_head.weight", None)?)?]
        } else if let Some(r) = only {
            vec![put_dense(card, pk, "lm_head.weight", Some(r), true)?]
        } else {
            vec![
                put(card, pk.tensor("lm_head.weight", Some(0))?)?,
                put(card, pk.tensor("lm_head.weight", Some(1))?)?,
            ]
        };
        Ok(Self {
            tp,
            embed_norm,
            norm,
            final_q,
            msg_q,
            layers,
            head,
        })
    }
}

/// Both tiers' records on the card + the xptr tables (128 slots; rank r = its 64).
struct Experts {
    kind: BTreeMap<u32, String>,
    /// (layer, part 0 = gate_up / 1 = down) -> the full table's address (128 x 3 pointers).
    xp: BTreeMap<(u32, u32), u64>,
}

impl Experts {
    /// `only` = the two-card process's rank (its 64 records a layer; the other rank's table
    /// entries are null).
    fn load(pk: &PackReader, card: &mut Card<'_, '_>, only: Option<u32>) -> anyhow::Result<Self> {
        let man = &pk.man;
        let native = man["format"] == NATIVE_PACK_FORMAT;
        let layers: Vec<u32> = man["layers"]
            .as_array()
            .context("layers")?
            .iter()
            .filter_map(Value::as_u64)
            .map(|x| x as u32)
            .collect();
        let mut kind = BTreeMap::new();
        for &l in &layers {
            kind.insert(
                l,
                if native {
                    man["expert_kinds"][l.to_string()]
                        .as_str()
                        .context("expert_kinds")?
                        .to_string()
                } else {
                    "mxfp4".to_string()
                },
            );
        }
        // (rank, layer) -> the 64 records' base
        let mut blobs = BTreeMap::new();
        let read_tier = |file: &str,
                         tl: &[u32],
                         stride: u64,
                         r: u32,
                         card: &mut Card<'_, '_>,
                         blobs: &mut BTreeMap<(u32, u32), u64>|
         -> anyhow::Result<()> {
            let f = std::fs::File::open(pk.dir.join(file)).with_context(|| file.to_string())?;
            let mut hdr = vec![0u8; 4096];
            f.read_exact_at(&mut hdr, 0)?;
            let doff = u64::from_le_bytes(hdr[168..176].try_into().expect("8"));
            let mut buf = vec![0u8; (stride * 64) as usize];
            for (slot, &l) in tl.iter().enumerate() {
                f.read_exact_at(&mut buf, doff + slot as u64 * 64 * stride)?;
                let a = card.up(&buf)?;
                blobs.insert((r, l), a);
            }
            Ok(())
        };
        for r in 0..2u32 {
            if only.is_some_and(|o| o != r) {
                continue;
            }
            let t = &man["tiers"][r.to_string()];
            if native {
                for (k, tier) in t.as_object().context("tiers")? {
                    let stride = man["records"][k]["stride"].as_u64().context("stride")?;
                    let tl: Vec<u32> = tier["layers"]
                        .as_array()
                        .context("tier layers")?
                        .iter()
                        .filter_map(Value::as_u64)
                        .map(|x| x as u32)
                        .collect();
                    read_tier(
                        tier["file"].as_str().context("file")?,
                        &tl,
                        stride,
                        r,
                        card,
                        &mut blobs,
                    )?;
                }
            } else {
                let stride = man["record"]["stride"].as_u64().context("stride")?;
                read_tier(
                    t["file"].as_str().context("file")?,
                    &layers,
                    stride,
                    r,
                    card,
                    &mut blobs,
                )?;
            }
        }
        let mut xp = BTreeMap::new();
        for &l in &layers {
            let rec = if native {
                &man["records"][&kind[&l]]
            } else {
                &man["record"]
            };
            let stride = rec["stride"].as_u64().context("stride")?;
            for (pi_, part) in ["gate_up", "down"].iter().enumerate() {
                let mut tab = Vec::with_capacity(128 * 3 * 8);
                for r in 0..2u32 {
                    let base = blobs.get(&(r, l)).copied();
                    for s in 0..64u64 {
                        for k in ["_w", "_ws", "_wref"] {
                            let o = rec[format!("{part}{k}")][0]
                                .as_u64()
                                .context("record offset")?;
                            let a = base.map_or(0, |b| b + s * stride + o);
                            tab.extend_from_slice(&a.to_le_bytes());
                        }
                    }
                }
                xp.insert((l, pi_ as u32), card.up(&tab)?);
            }
        }
        Ok(Self { kind, xp })
    }
}

/// INK5: one sequence's per-visit state, allocated up front (a capture allocates nothing).
struct Cache {
    npg: u64,
    table: u64,
    /// INK9: the local layers' ring (`--ink-local-ring` rows; 0 = full per-visit pools, the
    /// control): page p of a local visit lives at physical page p % ring_pages, so a local pool
    /// holds ring_pages x 64 rows (>= the 1,024 window + a chunk's rows - 1: a chunk writes all
    /// its rows before its attention reads any); `ltable` is that map.
    ring_pages: u64,
    ltable: u64,
    rpos: u64,
    rpos_dec: u64,
    /// INK9 task 4: a batched decode step's per-row slot / position words (8 a step, data)
    rseq_b: u64,
    rpos_b: u64,
    /// INK13: a captured verify chunk's per-row slot / position words (8, data)
    rseq_v: u64,
    rpos_v: u64,
    /// INK20: a multi-sequence verify chunk's per-row slot / position words (MV_ROWS each) and
    /// its per-sequence keep counts (8): ONE buffer, one write a replay (slots, then positions)
    rseq_m: u64,
    rpos_m: u64,
    keep_m: u64,
    /// INK36 `hg`: every slot's row words for a continued chunk of <= `SLOT_ROWS` rows
    /// (`SLOT_ROWS` i32 a slot, the slot's index: the bytes the per-visit write would write)
    rseq_s: u64,
    /// live slots (the pad slot is `seqs`)
    seqs: u64,
    ctl_reset: bool,
    /// (visit, rank) -> (k pool, v pool, k state, v state)
    pools: BTreeMap<(usize, usize), [u64; 4]>,
    /// INK11: (visit, rank) -> the k / v conv states' parity-1 buffers (the state parity law)
    pools2: BTreeMap<(usize, usize), [u64; 2]>,
    /// (visit, 2 | 3) -> the sconv state [3, D]
    convs: BTreeMap<(usize, u32), u64>,
    zeros: BTreeMap<u64, u64>,
}

impl Cache {
    fn new(
        card: &mut Card<'_, '_>,
        w: &Weights,
        sched: &[u32],
        max_ctx: u64,
        ring: u64,
        seqs: u64,
    ) -> anyhow::Result<Self> {
        let npg = max_ctx.div_ceil(PAGE);
        // INK9 task 4: `seqs` live sequence slots + one PAD slot (a padded batch row reads and
        // writes only the pad slot's pages and states); slot s owns pages s * pages + p
        ensure!(
            (1..=8).contains(&seqs),
            "ink cache: {seqs} sequences (1..8)"
        );
        let slots = seqs + 1;
        let ring_pages = if ring == 0 || ring.div_ceil(PAGE) >= npg {
            0
        } else {
            ensure!(
                ring >= EXT + PAGE,
                "--ink-local-ring {ring}: at least the 1,024-row window + a page"
            );
            ring.div_ceil(PAGE)
        };
        let lp = if ring_pages > 0 { ring_pages } else { npg };
        let ltab: Vec<u8> = (0..slots as i32)
            .flat_map(|sl| (0..npg as i32).map(move |p| sl * lp as i32 + p % lp as i32))
            .flat_map(i32::to_le_bytes)
            .collect();
        let ltable = card.up(&ltab)?;
        let tab: Vec<u8> = (0..slots as i32)
            .flat_map(|sl| (0..npg as i32).map(move |p| sl * npg as i32 + p))
            .flat_map(i32::to_le_bytes)
            .collect();
        let pos: Vec<u8> = (0..max_ctx as i32).flat_map(i32::to_le_bytes).collect();
        let table = card.up(&tab)?;
        let rpos = card.up(&pos)?;
        let rpos_dec = card.alloc_zero(4)?;
        let rseq_b = card.alloc_zero(4 * 8)?;
        let rpos_b = card.alloc_zero(4 * 8)?;
        // contiguous: one 64-byte write a verify (slot words, then position words)
        let rseq_v = card.alloc_zero(4 * 16)?;
        let rpos_v = rseq_v + 32;
        let rseq_m = card.alloc_zero(4 * (2 * MV_ROWS + 8))?;
        let rpos_m = rseq_m + 4 * MV_ROWS;
        let keep_m = rpos_m + 4 * MV_ROWS;
        let rs: Vec<u8> = (0..slots as i32)
            .flat_map(|sl| (0..SLOT_ROWS).map(move |_| sl))
            .flat_map(i32::to_le_bytes)
            .collect();
        let rseq_s = card.up(&rs)?;
        let mut pools = BTreeMap::new();
        let mut pools2 = BTreeMap::new();
        let mut convs = BTreeMap::new();
        let mut zeros = BTreeMap::new();
        for (vi, &l) in sched.iter().enumerate() {
            let pages = if ring_pages > 0 && !GLOBAL.contains(&l) {
                ring_pages
            } else {
                npg
            };
            for (ri, rw) in w.layers[l as usize].r.iter().enumerate() {
                let c = rw.kv * HD;
                let kp = card.alloc_zero(slots * pages * PAGE * c * 2)?;
                let vp = card.alloc_zero(slots * pages * PAGE * c * 2)?;
                let sk = card.alloc_zero(slots * 3 * c * 4)?;
                let sv = card.alloc_zero(slots * 3 * c * 4)?;
                pools.insert((vi, ri), [kp, vp, sk, sv]);
                let sk1 = card.alloc_zero(slots * 3 * c * 4)?;
                let sv1 = card.alloc_zero(slots * 3 * c * 4)?;
                pools2.insert((vi, ri), [sk1, sv1]);
                if let std::collections::btree_map::Entry::Vacant(e) = zeros.entry(c) {
                    e.insert(card.alloc_zero(slots * 3 * c * 4)?);
                }
            }
            for which in [2u32, 3] {
                convs.insert((vi, which), card.alloc_zero(slots * 3 * D * 4)?);
            }
        }
        Ok(Self {
            npg,
            table,
            ring_pages,
            ltable,
            rpos,
            rpos_dec,
            rseq_b,
            rpos_b,
            rseq_v,
            rpos_v,
            rseq_m,
            rpos_m,
            keep_m,
            rseq_s,
            seqs,
            ctl_reset: false,
            pools,
            pools2,
            convs,
            zeros,
        })
    }
}

/// INK20: the most rows a multi-sequence verify chunk holds (8 sequences x 4 rows).
pub const MV_ROWS: u64 = 32;

/// The forward (the harness's `Forward`).
struct Forward<'w> {
    w: &'w Weights,
    x: &'w Experts,
    sched: Vec<u32>,
    noloop: bool,
    nomsg: bool,
    pos0: u64,
    /// INK8: the decode form a captured graph replays (the position is the `rpos_dec` word, data);
    /// eager runs read `rpos[pos0..]` (the same values), so a prefill chunk may continue a sequence.
    graph: bool,
    /// INK7 task 3: the two-card process's rank and its peer (None = one card).
    peer: Option<&'w Peer>,
    /// INK9 task 4: the cache slot a single-sequence run (prefill / b1 harness decode) uses
    slot: u64,
    /// INK9 task 4: a batched decode step -- every row its own sequence: the rows' slot and
    /// position words live in the cache (`rseq_b` / `rpos_b`), states by slot, sconv per row
    batch: bool,
    /// INK13: a captured VERIFY chunk (INK9's chain, one sequence): the rows' position / slot
    /// words are data (`Cache::rpos_v` / `rseq_v`, written before every replay), so one graph
    /// a (slot, width) replays at any position; `pos0` only selects the carried state (> 0)
    vrow: bool,
    /// INK20: a multi-sequence verify chunk -- `mrow` chained rows a sequence (0 = off), the rows'
    /// slot / position words in the cache (`rseq_m` / `rpos_m`, data); INK9's chain with the
    /// per-sequence sconv / history (`ink_mv`), the kv conv's state written to scratch (the
    /// commit owns every state)
    mrow: u64,
}

/// INK36: a forward's state between visits ([`Forward::run_visit`]).
struct RunSt {
    x: u64,
    blocks: Vec<u64>,
    pre: Option<(u64, Vec<u64>)>,
    prev: Option<(Vec<u64>, u64)>,
    fused: bool,
    pre_next: Option<Vec<u64>>,
    mdep: bool,
    pass_msgs: Option<u64>,
}

/// INK7: the two-card process's pair link -- the peer's partial addresses (one process, two HIP
/// threads, peer access on), the head rows, and a barrier.
pub struct Peer {
    pub rank: u32,
    tx: std::sync::mpsc::Sender<u64>,
    rx: std::sync::mpsc::Receiver<u64>,
    rows_tx: std::sync::mpsc::Sender<Vec<u8>>,
    rows_rx: std::sync::mpsc::Receiver<Vec<u8>>,
    /// The `[D, D]` bf16 identity: the reduce runs on the GEMM body (exact: one product of 1 a
    /// column, zeros elsewhere; a -0.0 partial reads back as +0.0). INK8: the `--ink-reduce gemm`
    /// control (host-synchronised, host head gather).
    ident: Tw,
    /// INK8 (`--ink-reduce p2p`, the default): the fused on-card reduce + head gather
    /// (`ink_p2p_xchg`); the peer's work-region base and the two ranks' link words.
    dev: bool,
    peer_base: u64,
    link: (u64, u64),
    /// INK10: the PEER's fuse words (`Card::fz` of the other card; 0 = none)
    pfz: u64,
    /// INK16: the push reduce's receive windows (ours, the peer's: two slots of `PWIN_ROWS` x D
    /// f32 each; (0, 0) = none) -- `ink_p2p_push_v5` above 8 rows under `--ink-prefill-gemm clp`
    pwin: (u64, u64),
}

/// INK16: rows a push-reduce window slot holds (the prefill chunk).
const PWIN_ROWS: u64 = 1024;

/// INK10: a visit's reduce -- done (the reduced `[t, D]` rows), or pushed (the own partial; the
/// consumer `ink_xsd` forms own + peer from the site's window).
#[derive(Clone, Copy)]
enum Red {
    Done(u64),
    /// the own partial + the producer's flag count (its workgroups)
    Pushed(u64, u64),
}

impl Peer {
    /// The all-reduce of a `[t, D]` f32 partial: both ranks compute `own + peer` (f32 addition is
    /// commutative, so both hold the same bytes = rank 0 + rank 1), the peer's partial read over
    /// P2P inside the GEMM's A loads.
    /// INK8: the peer's address of one of our work-region buffers (both ranks take the same
    /// sequence of buffers, so the offsets agree -- checked over the host link every call).
    fn peer_addr(&self, c: &Card<'_, '_>, a: u64) -> anyhow::Result<u64> {
        let off = a - c.base();
        self.tx.send(off).context("peer gone")?;
        let theirs = self.rx.recv().context("peer gone")?;
        ensure!(
            theirs == off,
            "ink p2p: rank {} operand at work offset {off:#x}, the peer's at {theirs:#x}",
            self.rank
        );
        Ok(self.peer_base + off)
    }

    fn reduce(&self, c: &mut Card<'_, '_>, partial: u64, t: u64) -> anyhow::Result<u64> {
        // INK39 (`push`): a > 8-row multi verify's reduces take the push too (v4's bytes)
        let mv_push = c.mv_live && c.p39 & P39_PUSH != 0;
        if self.dev
            && (c.push || mv_push)
            && self.pwin.0 != 0
            && (9..=PWIN_ROWS).contains(&t)
            && partial % 16 == 0
        {
            let fin = c.z4(Reg::Visit, t * D, "reduce.out")?;
            let wg = (t * D / 4).div_ceil(256).clamp(1, 128);
            if mv_push && !c.push {
                c.p39_n[1] += 1;
            } else {
                c.cl_n[2] += 1;
            }
            c.launch(
                "ink_p2p_push_v5",
                [wg, 1, 1],
                256,
                &[
                    A::P(partial),
                    pi(D),
                    A::P(self.pwin.0),
                    A::P(self.pwin.1),
                    A::L(i64::try_from(PWIN_ROWS * D)?),
                    A::P(fin),
                    pi(D),
                    pi(t),
                    pi(D),
                    A::P(self.link.0 + c.link_off),
                    A::P(self.link.1 + c.link_off),
                ],
            )?;
            return Ok(fin);
        }
        if self.dev {
            let theirs = self.peer_addr(c, partial)?;
            let fin = c.z4(Reg::Visit, t * D, "reduce.out")?;
            c.xchg(
                (partial, D),
                (theirs, D),
                (fin, D),
                t,
                D,
                0,
                true,
                self.link,
            )?;
            return Ok(fin);
        }
        c.hip.sync_stream(&c.stream)?;
        self.tx.send(partial).context("peer gone")?;
        let theirs = self.rx.recv().context("peer gone")?;
        let fin = c.z4(Reg::Visit, t * D, "reduce.out")?;
        c.gemm(partial, t, D, &self.ident, fin, D, false)?;
        c.gemm(theirs, t, D, &self.ident, fin, D, true)?;
        c.hip.sync_stream(&c.stream)?;
        // both ranks done reading each other's partial (a token each way; a dead peer = an error)
        self.tx.send(0).context("peer gone")?;
        self.rx.recv().context("peer gone")?;
        Ok(fin)
    }

    /// INK27 (`--ink-prefix-cache-sync`): both ranks' streams drained, then a host rendezvous (a
    /// token each way, the same program point on both ranks): neither rank's next launch is
    /// queued before both ranks' prior work is complete.
    pub(super) fn barrier(&self, c: &Card<'_, '_>, tag: u64) -> anyhow::Result<()> {
        c.hip.sync_stream(&c.stream)?;
        self.tx.send(tag).context("peer gone")?;
        let t = self.rx.recv().context("peer gone")?;
        ensure!(
            t == tag,
            "ink barrier: rank {} sent {tag:#x}, the peer {t:#x}",
            self.rank
        );
        Ok(())
    }
}

impl Forward<'_> {
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn attention(
        &self,
        c: &mut Card<'_, '_>,
        cache: Option<&Cache>,
        l: u32,
        vi: usize,
        t: u64,
        qi: u64,
        ki: u64,
        vin: u64,
        push: bool,
    ) -> anyhow::Result<Red> {
        let glob = GLOBAL.contains(&l);
        // INK10: the last head group's o GEMV pushes the partial (two-card, the fused form)
        let push_to = match self.peer {
            Some(pr) if push && pr.dev && pr.pfz != 0 => Some(pr.pfz),
            _ => None,
        };
        let mut pushed = false;
        let mut nfl = 0;
        let ngroups = self.w.layers[l as usize].r.len();
        let lw = &self.w.layers[l as usize];
        let out = c.z4(Reg::Visit, t * D, "attn.out")?;
        let (npg, table, rpos) = match cache {
            None => {
                let npg = t.div_ceil(PAGE);
                let table = c.z4(Reg::Visit, npg, "table")?;
                let tab: Vec<u8> = (0..npg as i32).flat_map(i32::to_le_bytes).collect();
                c.flush_z()?;
                c.write(table, &tab)?;
                let rpos = c.z4(Reg::Visit, t, "rpos")?;
                let pos: Vec<u8> = (0..t as i32).flat_map(i32::to_le_bytes).collect();
                c.flush_z()?;
                c.write(rpos, &pos)?;
                (npg, table, rpos)
            }
            Some(k) => (
                k.npg,
                if k.ring_pages > 0 && !glob {
                    ensure!(
                        t + EXT - 1 <= k.ring_pages * PAGE,
                        "a {t}-row chunk over a {}-row local ring",
                        k.ring_pages * PAGE
                    );
                    k.ltable
                } else {
                    k.table
                },
                if self.batch {
                    k.rpos_b
                } else if self.mrow > 0 {
                    k.rpos_m
                } else if self.graph {
                    k.rpos_dec
                } else if self.vrow {
                    k.rpos_v
                } else {
                    k.rpos + 4 * self.pos0
                },
            ),
        };
        let rseq = match cache {
            Some(k) if self.batch => k.rseq_b,
            Some(k) if self.mrow > 0 => k.rseq_m,
            Some(k) if self.vrow => k.rseq_v,
            // INK36 `hg`: the slot's row words from the cache's table (no blocking write)
            Some(k) if c.hg_live && !self.graph && t <= SLOT_ROWS => {
                k.rseq_s + self.slot * SLOT_ROWS * 4
            }
            _ => {
                let r = c.z4(Reg::Visit, t, "rseq")?;
                if self.slot > 0 {
                    ensure!(!self.graph, "a single-sequence graph runs slot 0");
                    let v: Vec<u8> = (0..t)
                        .flat_map(|_| i32::try_from(self.slot).unwrap_or(0).to_le_bytes())
                        .collect();
                    c.flush_z()?;
                    c.write(r, &v)?;
                }
                r
            }
        };
        for (ri, rw) in lw.r.iter().enumerate() {
            let (h, kv) = (rw.h, rw.kv);
            let cc = kv * HD;
            // INK9 (fast, decode rows): the projections of one input row as ONE GEMV over the
            // contiguous weights (byte-equal: every output its own sequential sum); the views'
            // row strides are the group's width
            // INK39 (`cat`): a > 8-row multi verify takes the merged projections too (the cl
            // GEMM: every output its own sequential sum, so the bytes are the separate GEMMs')
            let cat = c.fast && (t <= 8 || (self.mrow > 0 && c.p39 & P39_CAT != 0));
            let (nq, nr, ns) = (h * HD, h * 16, h);
            let qw = nq + nr + ns + nq;
            let nk = if glob { cc + 4 * kv } else { cc };
            let (cs_, l0, ls) = if glob { (1, 0, cc as i32) } else { (4, 3, -1) };
            let (qraw, rraw, sink, gt, ldq);
            let (kraw, vraw, ldkv, dk, dv, lddyn);
            if cat {
                if let (Some(all), true) = (rw.all.as_ref(), qi == ki && ki == vin) {
                    let wd = qw + 2 * cc;
                    let x = c.z4(Reg::Visit, t * wd, "proj.all")?;
                    c.gemm(qi, t, D, all, x, wd, false)?;
                    if t > 8 {
                        c.p39_n[0] += 1;
                    }
                    (qraw, rraw, sink, gt, ldq) =
                        (x, x + 4 * nq, x + 4 * (nq + nr), x + 4 * (nq + nr + ns), wd);
                    let d = c.z4(Reg::Visit, t * kv * 4, "dkv")?;
                    (kraw, vraw, ldkv, dk, dv, lddyn) =
                        (x + 4 * qw, x + 4 * (qw + cc), wd, d, d, kv * 4);
                } else {
                    let xq = c.z4(Reg::Visit, t * qw, "proj.q")?;
                    (qraw, rraw, sink, gt, ldq) = (
                        xq,
                        xq + 4 * nq,
                        xq + 4 * (nq + nr),
                        xq + 4 * (nq + nr + ns),
                        qw,
                    );
                    let xk = c.z4(Reg::Visit, t * nk, "proj.k")?;
                    let xv = c.z4(Reg::Visit, t * nk, "proj.v")?;
                    if t > 8 {
                        // INK39 (`cat`, > 8 rows): the three group GEMMs (gemv3 is <= 8 rows)
                        c.gemm(qi, t, D, &rw.qg, xq, qw, false)?;
                        c.gemm(ki, t, D, &rw.kg, xk, nk, false)?;
                        c.gemm(vin, t, D, &rw.vg, xv, nk, false)?;
                        c.p39_n[0] += 3;
                    } else if c.fuse & FUSE_QKV != 0 {
                        // INK11: the three GEMVs as one launch (each output its own sum)
                        c.gemv3(
                            t,
                            &[
                                (qi, &rw.qg, xq, qw),
                                (ki, &rw.kg, xk, nk),
                                (vin, &rw.vg, xv, nk),
                            ],
                        )?;
                    } else {
                        c.gemm(qi, t, D, &rw.qg, xq, qw, false)?;
                        c.gemm(ki, t, D, &rw.kg, xk, nk, false)?;
                        c.gemm(vin, t, D, &rw.vg, xv, nk, false)?;
                    }
                    if glob {
                        (kraw, vraw, ldkv, dk, dv, lddyn) =
                            (xk, xv, nk, xk + 4 * cc, xv + 4 * cc, nk);
                    } else {
                        let d = c.z4(Reg::Visit, t * kv * 4, "dkv")?;
                        (kraw, vraw, ldkv, dk, dv, lddyn) = (xk, xv, nk, d, d, kv * 4);
                    }
                }
            } else {
                kraw = c.z4(Reg::Visit, t * cc, "kraw")?;
                vraw = c.z4(Reg::Visit, t * cc, "vraw")?;
                qraw = c.z4(Reg::Visit, t * h * HD, "qraw")?;
                rraw = c.z4(Reg::Visit, t * h * 16, "rraw")?;
                c.gemm(ki, t, D, &rw.k, kraw, cc, false)?;
                c.gemm(vin, t, D, &rw.v, vraw, cc, false)?;
                c.gemm(qi, t, D, &rw.q, qraw, h * HD, false)?;
                c.gemm(qi, t, D, &rw.r, rraw, h * 16, false)?;
                ldkv = cc;
                ldq = h * HD;
                if glob {
                    dk = c.z4(Reg::Visit, t * kv * 4, "dk")?;
                    dv = c.z4(Reg::Visit, t * kv * 4, "dv")?;
                    c.gemm(ki, t, D, rw.pk.as_ref().context("pk")?, dk, kv * 4, false)?;
                    c.gemm(vin, t, D, rw.pv.as_ref().context("pv")?, dv, kv * 4, false)?;
                } else {
                    let d = c.z4(Reg::Visit, t * kv * 4, "dkv")?;
                    (dk, dv) = (d, d);
                }
                lddyn = kv * 4;
                sink = 0;
                gt = 0;
            }
            // INK11: the state parity law (`kvc`): the state before position p lives in buffer
            // p & 1; a decode row's attention reads / writes it in-kernel, a chunk here
            let par = c.fuse & FUSE_KVC != 0 && cache.is_some();
            let kvc = par && push && cat && (self.batch || t == 1);
            ensure!(
                !par || kvc || (!self.batch && !self.graph),
                "the state parity law: a batched / captured row outside the kvc attention"
            );
            let mut st1 = (0, 0);
            let (kpool, vpool, sk, sv, sko, svo) = match cache {
                None => {
                    let sk = c.z4(Reg::Visit, 3 * cc, "sk")?;
                    let sv = c.z4(Reg::Visit, 3 * cc, "sv")?;
                    let sko = c.z4(Reg::Visit, 3 * cc, "sko")?;
                    let svo = c.z4(Reg::Visit, 3 * cc, "svo")?;
                    let kp = c.take(Reg::Visit, npg * PAGE * cc * 2, "kpool")?;
                    let vp = c.take(Reg::Visit, npg * PAGE * cc * 2, "vpool")?;
                    (kp, vp, sk, sv, sko, svo)
                }
                Some(k) => {
                    let [kp, vp, sko, svo] = k.pools[&(vi, ri)];
                    let [sk1, sv1] = k.pools2[&(vi, ri)];
                    st1 = (sk1, sv1);
                    let zero = !self.batch && (self.pos0 == 0 || k.ctl_reset);
                    if par && !kvc {
                        let pick = |p: u64| if p & 1 == 0 { (sko, svo) } else { (sk1, sv1) };
                        let (si, vi_) = if zero {
                            (k.zeros[&cc], k.zeros[&cc])
                        } else {
                            pick(self.pos0)
                        };
                        let (so, vo) = pick(self.pos0 + t);
                        (kp, vp, si, vi_, so, vo)
                    } else {
                        let (sk, sv) = if zero {
                            (k.zeros[&cc], k.zeros[&cc])
                        } else {
                            (sko, svo)
                        };
                        (kp, vp, sk, sv, sko, svo)
                    }
                }
            };
            let (sink, ldsink) = if cat { (sink, ldq) } else { (0, 0) };
            let o = c.z4(Reg::Visit, t * h * HD, "o")?;
            if kvc {
                // INK11: the conv / post / append + the attention in one launch
                c.launch(
                    "ink_attn_kvc",
                    [t, h, 1],
                    256,
                    &[
                        A::P(kraw),
                        A::P(vraw),
                        pi(ldkv),
                        A::P(qraw),
                        pi(ldq),
                        A::P(rraw),
                        pi(ldq),
                        A::P(dk),
                        A::P(dv),
                        pi(lddyn),
                        A::P(rw.wk),
                        A::P(rw.wv),
                        A::I(cs_),
                        A::I(l0),
                        A::I(ls),
                        A::P(rw.q_scale),
                        A::P(rw.rel_scale),
                        A::P(sko),
                        A::P(st1.0),
                        A::P(svo),
                        A::P(st1.1),
                        A::P(kpool),
                        A::P(vpool),
                        A::P(table),
                        pi(npg),
                        A::P(rseq),
                        A::P(rpos),
                        A::P(lw.rel_proj),
                        A::P(sink),
                        pi(ldsink),
                        A::P(o),
                        pi(h * HD),
                        pi(h),
                        pi(kv),
                        pi(EXT),
                        A::I(if glob { 0 } else { 1024 }),
                        A::I(i32::from(glob)),
                        A::F(QK_CLIP),
                        A::F(REL_CLIP),
                        A::F(NFLOOR),
                        A::F(ALPHA),
                    ],
                )?;
            } else {
                let qp = c.take(Reg::Visit, t * h * HD * 2, "qp")?;
                let rp = c.z4(Reg::Visit, t * h * 16, "rp")?;
                let (sko, svo) = match cache {
                    Some(k) if self.mrow > 0 => {
                        // INK20: every sequence's kv state is the commit's (its history entry);
                        // the kernel's own state write goes to scratch, so a sequence's rows
                        // across two 8-row tiles never race its in-place state
                        ensure!(!par, "a multi-sequence verify under the kvc parity law");
                        ensure!(
                            c.hist.is_some(),
                            "a multi-sequence verify without a history"
                        );
                        c.stash_m(sk, rseq, kraw, ldkv, t, self.mrow, cc, sko)?;
                        c.stash_m(sv, rseq, vraw, ldkv, t, self.mrow, cc, svo)?;
                        let sc = c.take(Reg::Visit, 2 * (k.seqs + 1) * 3 * cc * 4, "kv.scratch")?;
                        (sc, sc + (k.seqs + 1) * 3 * cc * 4)
                    }
                    _ => (sko, svo),
                };
                if c.hist.is_some() && self.mrow == 0 {
                    // INK12: a verify chunk is one sequence at slot `self.slot`
                    let so = self.slot * 3 * cc * 4;
                    c.stash(sk + so, kraw, ldkv, t, cc, sko + so)?;
                    c.stash(sv + so, vraw, ldkv, t, cc, svo + so)?;
                }
                c.launch(
                    "ink_kv_conv_post_append",
                    [t.div_ceil(INK_R), 2 * kv + h, 1],
                    128,
                    &[
                        A::P(kraw),
                        A::P(vraw),
                        pi(ldkv),
                        A::P(qraw),
                        pi(ldq),
                        A::P(rraw),
                        pi(if cat { ldq } else { h * 16 }),
                        A::P(dk),
                        A::P(dv),
                        pi(lddyn),
                        A::P(rw.wk),
                        A::P(rw.wv),
                        A::I(cs_),
                        A::I(l0),
                        A::I(ls),
                        A::P(rw.q_scale),
                        A::P(rw.rel_scale),
                        A::P(sk),
                        A::P(sv),
                        A::P(sko),
                        A::P(svo),
                        A::P(kpool),
                        A::P(vpool),
                        A::P(table),
                        pi(npg),
                        A::P(rseq),
                        A::P(rpos),
                        A::P(qp),
                        pi(h * HD),
                        A::P(rp),
                        pi(h * 16),
                        pi(t),
                        pi(kv),
                        A::I(i32::from(glob)),
                        A::F(QK_CLIP),
                        A::F(REL_CLIP),
                        A::F(NFLOOR),
                        A::F(ALPHA),
                    ],
                )?;
                let rel = if c.fast {
                    0
                } else {
                    c.z4(Reg::Visit, t * h * EXT, "rel")?
                };
                if !c.fast {
                    c.launch(
                        "ink_rel_bias",
                        [t, h, 1],
                        256,
                        &[
                            A::P(rp),
                            pi(h * 16),
                            A::P(lw.rel_proj),
                            A::P(rel),
                            pi(h * EXT),
                            pi(EXT),
                        ],
                    )?;
                }
                let (sink, ldsink) = if cat {
                    (sink, ldq)
                } else {
                    let _ = sink;
                    let sk = c.z4(Reg::Visit, t * h, "sink")?;
                    c.gemm(qi, t, D, &rw.sink, sk, h, false)?;
                    (sk, h)
                };
                // INK27: a prefill chunk (one sequence, consecutive positions) on the row-tiled body
                let rt = c.fast
                    && c.fa > 0
                    && t > 8
                    && self.pos0 + t > c.fa_floor
                    && !self.batch
                    && self.mrow == 0
                    && !self.vrow
                    && !self.graph;
                if rt {
                    // CONS40 `fa128`: wm on 128 rows a workgroup past the chunk-start floor
                    let w128 = c.fa_sym == "ink_fa35_wm"
                        && c.p40 & P40_FA128 != 0
                        && self.pos0 >= c.p40_fam
                        && (glob || c.p40 & P40_FA128L != 0);
                    let (sym, fr) = if w128 {
                        c.p40_n += 1;
                        ("ink_fa35_wm128", 128)
                    } else {
                        (c.fa_sym, c.fa)
                    };
                    c.fa_n += 1;
                    c.launch(
                        sym,
                        [t.div_ceil(fr), h, 1],
                        if sym.starts_with("ink_fa35") {
                            // 16 rows a wave (wm32 was launched at 128 threads > its 64-thread bound)
                            2 * fr as u32
                        } else {
                            256
                        },
                        &[
                            A::P(qp),
                            pi(h * HD),
                            A::P(kpool),
                            A::P(vpool),
                            A::P(table),
                            pi(npg),
                            A::P(rseq),
                            A::P(rpos),
                            A::P(rp),
                            pi(h * 16),
                            A::P(lw.rel_proj),
                            A::P(sink),
                            pi(ldsink),
                            A::P(o),
                            pi(h * HD),
                            pi(t),
                            pi(h),
                            pi(kv),
                            pi(EXT),
                            A::I(if glob { 0 } else { 1024 }),
                        ],
                    )?;
                } else if c.fast {
                    c.launch(
                        "ink_attn_bias_sink_v1",
                        [t, h, 1],
                        256,
                        &[
                            A::P(qp),
                            pi(h * HD),
                            A::P(kpool),
                            A::P(vpool),
                            A::P(table),
                            pi(npg),
                            A::P(rseq),
                            A::P(rpos),
                            A::P(rp),
                            pi(h * 16),
                            A::P(lw.rel_proj),
                            A::P(sink),
                            pi(ldsink),
                            A::P(o),
                            pi(h * HD),
                            pi(h),
                            pi(kv),
                            pi(EXT),
                            A::I(if glob { 0 } else { 1024 }),
                        ],
                    )?;
                } else {
                    c.launch(
                        "ink_attn_bias_sink_v0",
                        [t, h, 1],
                        256,
                        &[
                            A::P(qp),
                            pi(h * HD),
                            A::P(kpool),
                            A::P(vpool),
                            A::P(table),
                            pi(npg),
                            A::P(rseq),
                            A::P(rpos),
                            A::P(rel),
                            pi(h * EXT),
                            A::P(sink),
                            pi(ldsink),
                            A::P(o),
                            pi(h * HD),
                            pi(h),
                            pi(kv),
                            pi(EXT),
                            A::I(if glob { 0 } else { 1024 }),
                        ],
                    )?;
                }
            }
            let (gt, ldg) = if cat {
                (gt, ldq)
            } else {
                let _ = gt;
                let g = c.z4(Reg::Visit, t * h * HD, "gt")?;
                c.gemm(qi, t, D, &rw.gate, g, h * HD, false)?;
                (g, h * HD)
            };
            if c.fast && t <= 8 && (h * HD) % 256 == 0 {
                // INK9: o * sigmoid(g) formed in the o GEMV's staging (byte-equal)
                ensure!(rw.o.addr % 16 == 0, "o_proj not 16-byte aligned");
                if let (Some(pf), true) = (push_to, ri + 1 == ngroups) {
                    let ctr = c.fz + FZ_CTR;
                    // INK10 h2: the 16-wide push tile measured slower at N 2,304 (42.5 vs 40 us
                    // a launch): the narrow form serves N <= 1,024 only
                    let w8 = rw.o.sc != 0;
                    // INK17 (`g16`): the 8-bit push on the 16-wide tile at N 2,304 too
                    let narrow = c.fuse & FUSE_GEMV != 0
                        && (h * HD / 256) % 3 == 0
                        && (D <= 1024 || (w8 && c.fuse & FUSE_G16 != 0));
                    let gw = if narrow { 16 } else { 64 };
                    nfl = D / gw;
                    let mut args = vec![
                        A::P(o),
                        pi(h * HD),
                        A::P(gt),
                        pi(ldg),
                        A::P(rw.o.addr),
                        pi(h * HD),
                    ];
                    if w8 {
                        // INK14: the 8-bit o_proj (codes + scales)
                        args.extend([A::P(rw.o.sc), pi(h * HD / 32)]);
                    }
                    args.extend([
                        A::P(out),
                        pi(D),
                        pi(t),
                        pi(D),
                        pi(h * HD),
                        A::I(i32::from(ri > 0)),
                        A::P(pf),
                        pi(D),
                        A::P(pf + FZ_FLAGS),
                        A::P(ctr),
                    ]);
                    c.launch(
                        match (narrow, w8) {
                            (true, false) => "ink_gemv_gate_push_n16",
                            (false, false) => "ink_gemv_gate_push",
                            (true, true) => "ink_gemv_gate_push_n16_w8",
                            (false, true) => "ink_gemv_gate_push_w8",
                        },
                        [D.div_ceil(gw), 1, 1],
                        256,
                        &args,
                    )?;
                    pushed = true;
                    continue;
                }
                if rw.o.sc != 0 {
                    c.launch(
                        "ink_gemv_gate_f32_w8",
                        [D.div_ceil(64), 1, 1],
                        256,
                        &[
                            A::P(o),
                            pi(h * HD),
                            A::P(gt),
                            pi(ldg),
                            A::P(rw.o.addr),
                            pi(h * HD),
                            A::P(rw.o.sc),
                            pi(h * HD / 32),
                            A::P(out),
                            pi(D),
                            pi(t),
                            pi(D),
                            pi(h * HD),
                            A::I(i32::from(ri > 0)),
                        ],
                    )?;
                    continue;
                }
                c.launch(
                    "ink_gemv_gate_f32_bf16",
                    [D.div_ceil(64), 1, 1],
                    256,
                    &[
                        A::P(o),
                        pi(h * HD),
                        A::P(gt),
                        pi(ldg),
                        A::P(rw.o.addr),
                        pi(h * HD),
                        A::P(out),
                        pi(D),
                        pi(t),
                        pi(D),
                        pi(h * HD),
                        A::I(i32::from(ri > 0)),
                    ],
                )?;
                continue;
            }
            let og = c.z4(Reg::Visit, t * h * HD, "og")?;
            // INK29 `fq`: the o input's q8 in the out_gate launch (byte-equal to the pair; only
            // where the iu8 GEMM will read it at INK25's own quant law)
            let fq = c.p29 & P29_FQ != 0
                && c.i8_live
                && rw.o.sc != 0
                && c.p29 & (P29_LS | P29_I8IN) == 0
                && (h * HD) % 128 == 0
                && og % 16 == 0
                && !c.i8_cache.iter().any(|e| e.0 == og);
            if fq {
                let b16 = c.p29 & P29_B16 != 0;
                let qa = c.take(Reg::Visit, t * h * HD, "i8.qa")?;
                let sa = c.z4(Reg::Visit, t * h * HD / if b16 { 16 } else { 32 }, "i8.sa")?;
                c.launch(
                    if b16 {
                        "ink29_out_gate_q8_b16"
                    } else {
                        "ink29_out_gate_q8"
                    },
                    [(t * h * HD / 8).div_ceil(256), 1, 1],
                    256,
                    &[
                        A::P(o),
                        pi(h * HD),
                        A::P(gt),
                        pi(ldg),
                        A::P(og),
                        pi(h * HD),
                        pi(t),
                        pi(h * HD),
                        A::P(qa),
                        A::P(sa),
                    ],
                )?;
                c.i8_cache.push((og, t, h * HD, qa, sa));
                c.p29_n[3] += 1;
            } else {
                c.launch(
                    "ink_out_gate",
                    [1024.min((t * h * HD).div_ceil(256)), 1, 1],
                    256,
                    &[
                        A::P(o),
                        pi(h * HD),
                        A::P(gt),
                        pi(ldg),
                        A::P(og),
                        pi(h * HD),
                        pi(t),
                        pi(h * HD),
                    ],
                )?;
            }
            c.gemm(og, t, h * HD, &rw.o, out, D, ri > 0)?;
        }
        if pushed {
            return Ok(Red::Pushed(out, nfl));
        }
        Ok(Red::Done(match self.peer {
            Some(pr) => pr.reduce(c, out, t)?,
            None => out,
        }))
    }

    fn moe(&self, c: &mut Card<'_, '_>, l: u32, h: u64, t: u64, push: bool) -> anyhow::Result<Red> {
        let lw = &self.w.layers[l as usize];
        let u = c.z4(Reg::Visit, t * D, "u")?;
        let ids = c.z4(Reg::Visit, t * 8, "ids")?;
        let wts = c.z4(Reg::Visit, t * 8, "wts")?;
        let mut tail_sk: Option<u64> = None;
        let parts: Vec<(u64, u64, Option<u64>)> = if self.w.tp == 1 {
            vec![(0, 128, None)]
        } else if let Some(pr) = self.peer {
            let r = u64::from(pr.rank);
            vec![(64 * r, 64, Some(r))]
        } else {
            vec![(0, 64, Some(0)), (64, 64, Some(1))]
        };
        let push_to = match self.peer {
            Some(pr) if push && pr.dev && pr.pfz != 0 && parts.len() == 1 => Some(pr.pfz),
            _ => None,
        };
        let a8 = c.take(Reg::Visit, t * D, "a8")?;
        let as_ = c.z4(Reg::Visit, t, "as")?;
        // INK10: the route's sort tables of part 0, taken before the route when it is fused
        let mut tabs0: Option<[u64; 7]> = None;
        // INK30: `r1` also takes every row count the logits + tail pair ran (<= 32, <= 8 under pf):
        // the 24-row batched verify; the same tables (the tail's stable 256-wide rank)
        let r1_rows = if c.pf { 8 } else { 32 };
        let route_fused = c.fast
            && c.fuse & FUSE_ROUTE != 0
            && (t <= 8 || (c.fuse & FUSE_R1 != 0 && t <= r1_rows))
            && c.fz != 0;
        if route_fused {
            let tb = Self::route_tables(c, t, parts[0].1, 16)?;
            let sk = c.z4(Reg::Visit, t * 256, "route.sk")?;
            let (lo, nloc) = (parts[0].0, parts[0].1);
            // INK17 (`rt`): the tail per row (`ink_route_fused_rt`) at > 1 row
            let rt = c.fuse & FUSE_RT != 0 && t > 1;
            // INK28 (`r1`): one workgroup a row, every row count <= 8 (INK30: <= 32)
            let r1 = c.fuse & FUSE_R1 != 0;
            let mut args = vec![
                A::P(h),
                pi(D),
                A::P(lw.post_ln),
                A::P(lw.router),
                A::P(lw.bias),
                A::P(u),
                pi(D),
                A::P(sk),
                pi(D),
                A::P(ids),
                A::P(wts),
                A::F(8.0),
                A::P(a8),
                pi(D),
                A::P(as_),
                pi(t),
                pi(lo),
                pi(nloc),
                A::P(tb[0]),
                A::P(tb[1]),
                A::P(tb[2]),
                A::P(tb[3]),
                A::P(tb[4]),
                A::P(tb[5]),
                A::P(tb[6]),
                A::P(c.fz + FZ_ARRIVE),
            ];
            if rt || r1 {
                args.push(A::P(c.fz + FZ_RT));
            }
            c.launch(
                if r1 {
                    "ink_route_r1"
                } else if rt {
                    "ink_route_fused_rt"
                } else {
                    "ink_route_fused"
                },
                [t, if r1 { 1 } else { 16 }, 1],
                128,
                &args,
            )?;
            tabs0 = Some(tb);
        } else if c.fast && t <= if c.pf { 8 } else { 32 } {
            // INK9: ink_route_logits + ink_route_tail (the parent's order, byte-equal); INK15
            // (`--ink-prefill-gemm pf`): a prefill chunk of 9..32 rows takes the parent instead
            // (census: logits + tail 340 us vs the parent's ~100 at 26 rows)
            let sk = c.z4(Reg::Visit, t * 256, "route.sk")?;
            c.launch(
                "ink_route_logits",
                [t, 16, 1],
                128,
                &[
                    A::P(h),
                    pi(D),
                    A::P(lw.post_ln),
                    A::P(lw.router),
                    A::P(lw.bias),
                    A::P(u),
                    pi(D),
                    A::P(sk),
                    pi(D),
                ],
            )?;
            // INK9: the top-8 rides ink_route_tail with the act quant and the first route sort
            tail_sk = Some(sk);
        } else if t > 8 && c.p35_live && c.p37 & (P37_GLUE | P37_GLUE8) != 0 {
            // INK37 `glue`: the same bytes, R rows a workgroup (the router rows read once per R)
            let r = if c.p37 & P37_GLUE8 != 0 { 8 } else { 4 };
            c.p37_n[1] += 1;
            c.launch(
                if r == 8 {
                    "ink37_post_norm_route_e128_r8"
                } else {
                    "ink37_post_norm_route_e128_r4"
                },
                [t.div_ceil(r), 1, 1],
                128,
                &[
                    A::P(h),
                    pi(D),
                    A::P(lw.post_ln),
                    A::P(lw.router),
                    A::P(lw.bias),
                    A::P(u),
                    pi(D),
                    A::P(ids),
                    A::P(wts),
                    pi(D),
                    A::F(8.0),
                    pi(t),
                ],
            )?;
        } else {
            c.launch(
                "ink_post_norm_route_e128",
                [t, 1, 1],
                128,
                &[
                    A::P(h),
                    pi(D),
                    A::P(lw.post_ln),
                    A::P(lw.router),
                    A::P(lw.bias),
                    A::P(u),
                    pi(D),
                    A::P(ids),
                    A::P(wts),
                    pi(D),
                    A::F(8.0),
                ],
            )?;
        }
        let kind = self.x.kind[&l].as_str();
        if tail_sk.is_none() && tabs0.is_none() {
            c.launch(
                "ink_act_quant_rows",
                [t, 1, 1],
                256,
                &[A::P(u), pi(D), A::P(a8), pi(D), A::P(as_), pi(D)],
            )?;
        }
        let y = c.z4(Reg::Visit, t * D, "y")?;
        let mut pushed = false;
        for (pidx, &(lo, nloc, rk)) in parts.iter().enumerate() {
            // INK29 `mt`: 32 / 64-row expert tiles on the NVFP4 layers (the sort's own path only)
            let sorted = !(pidx == 0 && (tabs0.is_some() || tail_sk.is_some()));
            // INK37 `moe`: FN153's grouped prefill tile (byte-equal to the fn148 ks1 cells per row)
            let p37r = if sorted && t > 8 && kind == "nvfp4_native" && c.p35_live {
                p37_rows(c.p37, t)
            } else {
                None
            };
            let tr = if let Some(r) = p37r {
                r
            } else if sorted && t > 8 && kind == "nvfp4_native" && c.p29 & P29_MT != 0 {
                mt_rows(t)
            } else {
                16
            };
            let (maxrows, maxt) = (t * 8 + tr * nloc, t * 8 / tr + nloc);
            let [row_tok, row_slot, te, tr0, tn, meta, pos] = match (pidx, tabs0) {
                (0, Some(tb)) => tb,
                _ => Self::route_tables(c, t, nloc, tr)?,
            };
            if pidx == 0 && tabs0.is_some() {
                // INK10: the tail ran inside ink_route_fused
            } else if let (Some(sk), 0) = (tail_sk, pidx) {
                c.launch(
                    "ink_route_tail",
                    [t + 1, 1, 1],
                    256,
                    &[
                        A::P(sk),
                        A::P(ids),
                        A::P(wts),
                        A::F(8.0),
                        A::P(u),
                        pi(D),
                        A::P(a8),
                        pi(D),
                        A::P(as_),
                        pi(D),
                        pi(t),
                        pi(lo),
                        pi(nloc),
                        A::P(row_tok),
                        A::P(row_slot),
                        A::P(te),
                        A::P(tr0),
                        A::P(tn),
                        A::P(meta),
                        A::P(pos),
                    ],
                )?;
            } else {
                // INK29 `rs`: the same outputs, a workgroup an expert (prefill chunks)
                let v2 = (c.p29 & P29_RS != 0 && t > 8) || tr > 16;
                if v2 {
                    c.p29_n[2] += 1;
                }
                c.launch(
                    match (v2, tr) {
                        (_, 128) => "ink37_route_sort_e128_v2_t128",
                        (_, 32) => "ink29_route_sort_e128_v2_t32",
                        (_, 64) => "ink29_route_sort_e128_v2_t64",
                        (true, _) => "ink29_route_sort_e128_v2",
                        _ => "ink_route_sort_e128",
                    },
                    [if v2 { nloc } else { 1 }, 1, 1],
                    256,
                    &[
                        A::P(ids),
                        pi(t),
                        pi(lo),
                        pi(nloc),
                        A::P(row_tok),
                        A::P(row_slot),
                        A::P(te),
                        A::P(tr0),
                        A::P(tn),
                        A::P(meta),
                        A::P(pos),
                    ],
                )?;
            }
            let dummy = c.z4(Reg::Visit, 64, "dummy")?;
            let cnt = c.z4(Reg::Visit, 64, "cnt")?;
            let off = rk.map_or(0, |r| r * 64 * 3 * 8);
            let (xgu, xdn) = (self.x.xp[&(l, 0)] + off, self.x.xp[&(l, 1)] + off);
            let tt = [A::P(te), A::P(tr0), A::P(tn), A::P(row_tok), A::P(meta)];
            let gu = c.take(Reg::Visit, maxrows * 2 * I * 2, "gu")?;
            let dn = c.take(Reg::Visit, maxrows * D * 2, "dn")?;
            let mut a1 = vec![A::P(a8), A::P(xgu)];
            a1.extend(tt.iter().cloned());
            let (s1, s0) = match kind {
                "mxfp4" => (
                    "moe_grouped_w4a8_bk128_ks1_tm1_g1",
                    "moe_grouped_w4a8_bk128_ks1_tm1_g0",
                ),
                "nvfp4_native" if p37r.is_some() => {
                    c.p37_n[0] += 1;
                    if tr == 128 {
                        ("fn153_w4a8b_pf_tm8_g1", "fn153_w4a8b_pf_tm8_g0")
                    } else {
                        ("fn153_w4a8b_pf_tm4_g1", "fn153_w4a8b_pf_tm4_g0")
                    }
                }
                "nvfp4_native" => match tr {
                    32 => (
                        "fn148_w4a8b_bk128_ks1_tm2_g1",
                        "fn148_w4a8b_bk128_ks1_tm2_g0",
                    ),
                    64 => (
                        "fn148_w4a8b_bk128_ks1_tm4_g1",
                        "fn148_w4a8b_bk128_ks1_tm4_g0",
                    ),
                    _ => (
                        "fn148_w4a8b_bk128_ks1_tm1_g1",
                        "fn148_w4a8b_bk128_ks1_tm1_g0",
                    ),
                },
                "mxfp8_native" => ("ink_mx8_w8a8_g1", "ink_mx8_w8a8_g0"),
                k => bail!("layer {l}: expert kind {k:?}"),
            };
            if kind == "mxfp8_native" {
                a1.extend([A::P(as_), A::P(gu), pi(2 * I), pi(D)]);
            } else {
                a1.extend([
                    A::P(as_),
                    A::P(dummy),
                    A::P(cnt),
                    A::P(gu),
                    pi(2 * I),
                    pi(D),
                    A::I(0),
                ]);
                if kind == "nvfp4_native" {
                    a1.push(pi(I));
                }
            }
            c.launch(s1, [2 * I / 128, maxt, 1], 256, &a1)?;
            let q8 = c.take(Reg::Visit, maxrows * I, "q8")?;
            let qs = c.z4(Reg::Visit, maxrows, "qs")?;
            c.launch(
                "ink_swiglu_q8",
                [maxrows, 1, 1],
                256,
                &[
                    A::P(gu),
                    pi(2 * I),
                    A::P(row_slot),
                    A::P(meta),
                    A::P(0),
                    pi(I),
                    A::P(q8),
                    pi(I),
                    A::P(qs),
                    pi(I),
                    A::F(4.0),
                    A::F(25.0),
                ],
            )?;
            let mut a0 = vec![A::P(q8), A::P(xdn)];
            a0.extend(tt.iter().cloned());
            if kind == "mxfp8_native" {
                a0.extend([A::P(qs), A::P(dn), pi(D), pi(I)]);
            } else {
                a0.extend([
                    A::P(qs),
                    A::P(dummy),
                    A::P(cnt),
                    A::P(dn),
                    pi(D),
                    pi(I),
                    A::I(0),
                ]);
                if kind == "nvfp4_native" {
                    a0.push(pi(D));
                }
            }
            c.launch(s0, [D / 128, maxt, 1], 256, &a0)?;
            if let Some(pf) = push_to {
                let ctr = c.fz + FZ_SITE + FZ_CTR;
                c.launch(
                    if c.fuse & FUSE_P30 != 0 {
                        "ink_combine_push_p"
                    } else {
                        "ink_combine_push"
                    },
                    [t, 1, 1],
                    256,
                    &[
                        A::P(dn),
                        pi(D),
                        A::P(pos),
                        A::P(wts),
                        A::P(y),
                        pi(D),
                        pi(D),
                        A::I(i32::from(pidx > 0)),
                        A::P(pf + FZ_SITE),
                        pi(D),
                        A::P(pf + FZ_SITE + FZ_FLAGS),
                        A::P(ctr),
                    ],
                )?;
                pushed = true;
                continue;
            }
            c.launch(
                "ink_moe_combine",
                [t, 1, 1],
                256,
                &[
                    A::P(dn),
                    pi(D),
                    A::P(pos),
                    A::P(wts),
                    A::P(y),
                    pi(D),
                    pi(D),
                    A::I(i32::from(pidx > 0)),
                ],
            )?;
        }
        if pushed {
            return Ok(Red::Pushed(y, t));
        }
        Ok(Red::Done(match self.peer {
            Some(pr) => pr.reduce(c, y, t)?,
            None => y,
        }))
    }

    /// The route sort's tables of one part (`[row_tok, row_slot, te, tr0, tn, meta, pos]`).
    fn route_tables(c: &mut Card<'_, '_>, t: u64, nloc: u64, tr: u64) -> anyhow::Result<[u64; 7]> {
        let (maxrows, maxt) = (t * 8 + tr * nloc, t * 8 / tr + nloc);
        Ok([
            c.z4(Reg::Visit, maxrows, "row_tok")?,
            c.neg1(maxrows, "row_slot")?,
            c.z4(Reg::Visit, maxt, "te")?,
            c.z4(Reg::Visit, maxt, "tr0")?,
            c.z4(Reg::Visit, maxt, "tn")?,
            c.z4(Reg::Visit, 4, "meta")?,
            c.z4(Reg::Visit, t * 8, "pos")?,
        ])
    }

    /// INK10: the fused decode form applies (the fast bodies, `--ink-fuse x`, decode / verify rows
    /// of a carried cache: every sconv reads its carried state).
    fn fused(&self, c: &Card<'_, '_>, cache: Option<&Cache>, t: u64) -> bool {
        c.fast
            && c.fuse & FUSE_X != 0
            && c.fz != 0
            && t <= 8
            && self.mrow == 0
            // INK11: one row a sequence (a <= 8-row continuation chunk of ONE sequence chains its
            // rows' states: INK9's chain) -- INK17 (`vt`): the chained rows on `ink_xsdc`
            && (self.batch || t == 1 || c.fuse & FUSE_VT != 0)
            && cache.is_some_and(|k| self.batch || (self.pos0 > 0 && !k.ctl_reset))
    }

    /// INK17: the history entry (verify) or a scratch `[3 + t, D]` for `ink_xsdc` when the fused
    /// visit's rows chain one sequence's states, and whether the verify's commit (`ink_dr_commit`
    /// from that history) writes the state; `None` = one row a sequence (`ink_xsd`).
    fn chain(
        &self,
        c: &mut Card<'_, '_>,
        cache: &Cache,
        vi: usize,
        which: u32,
        t: u64,
    ) -> anyhow::Result<Option<(u64, bool)>> {
        if self.batch || t == 1 {
            return Ok(None);
        }
        let dst = cache.convs[&(vi, which)] + self.slot * 3 * D * 4;
        if let Some(at) = c.hist_entry(t, D, dst)? {
            return Ok(Some((at, true)));
        }
        Ok(Some((c.z4(Reg::Visit, (3 + t) * D, "xsdc.hist")?, false)))
    }

    /// INK10: the rows' state slots (`rseq`) of a fused visit.
    fn rseq(&self, c: &mut Card<'_, '_>, cache: &Cache, t: u64) -> anyhow::Result<u64> {
        if self.batch {
            return Ok(cache.rseq_b);
        }
        if self.vrow {
            return Ok(cache.rseq_v);
        }
        let r = c.z4(Reg::Visit, t, "rseq.x")?;
        if self.slot > 0 {
            ensure!(!self.graph, "a single-sequence graph runs slot 0");
            let v: Vec<u8> = (0..t)
                .flat_map(|_| i32::try_from(self.slot).unwrap_or(0).to_le_bytes())
                .collect();
            c.flush_z()?;
            c.write(r, &v)?;
        }
        Ok(r)
    }

    /// The visit's input depth attend (queries + gains) as `layer` forms it, for visit `l` with
    /// or without messages: (queries, gains, replicate one output to q / k / v).
    fn in_depth(&self, l: u32, msgs: bool) -> anyhow::Result<(Vec<u64>, Option<Vec<u64>>, bool)> {
        let lw = &self.w.layers[l as usize];
        let qn = if GLOBAL.contains(&l) { 3 } else { 1 };
        let aq = lw.attn_q.as_ref().context("attn_q")?;
        let qs: Vec<u64> = (0..qn).map(|i| aq.row(i, D)).collect();
        if msgs {
            Ok((qs, None, false))
        } else {
            Ok((qs, Some(lw.ln[..qn as usize].to_vec()), qn == 1))
        }
    }

    /// One visit. `pre` = this visit's input depth outputs, computed by the previous visit's fused
    /// tail (INK10); `ahead` = the next visit's input depth (queries, gains) to fuse into this
    /// visit's mlp sconv, its outputs returned with the visit's output.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn layer(
        &self,
        c: &mut Card<'_, '_>,
        cache: Option<&Cache>,
        l: u32,
        vi: usize,
        t: u64,
        x: u64,
        blocks: &mut Vec<u64>,
        msgs: Option<&[u64]>,
        pre: Option<Vec<u64>>,
        ahead: Option<(Vec<u64>, Option<Vec<u64>>, bool)>,
    ) -> anyhow::Result<(u64, Option<Vec<u64>>)> {
        let lw = &self.w.layers[l as usize];
        let glob = GLOBAL.contains(&l);
        let fused = self.fused(c, cache, t);
        let mut mixed_raw: Vec<u64> = Vec::new();
        let mut qkv: Vec<u64> = Vec::new();
        if let Some(p) = pre {
            if msgs.is_none() {
                qkv = if p.len() == 1 { vec![p[0]; 3] } else { p };
            } else {
                mixed_raw = p;
            }
        } else if l == 0 {
            mixed_raw = vec![x];
            if msgs.is_none() {
                let n = c.rmsnorm(Reg::Visit, x, t, D, lw.ln[0])?;
                qkv = vec![n, n, n];
            }
        } else {
            let qn = if glob { 3 } else { 1 };
            let aq = lw.attn_q.as_ref().context("attn_q")?;
            let qs: Vec<u64> = (0..qn).map(|i| aq.row(i, D)).collect();
            let mut st = blocks.clone();
            st.push(x);
            if msgs.is_none() {
                let outs = c.depth(t, &st, &qs, Some(&lw.ln[..qn as usize]))?;
                qkv = if qn == 1 { vec![outs[0]; 3] } else { outs };
            } else {
                mixed_raw = c.depth(t, &st, &qs, None)?;
            }
        }
        c.p35_live = t > 8 && !self.graph && !self.vrow && self.mrow == 0 && !self.batch;
        let msg_fused = c.fast && c.fuse & FUSE_MSG != 0 && t <= 8 && c.fz != 0;
        if let (Some(m), true) = (msgs, msg_fused) {
            let g = lw.gates.as_ref().context("gates")?;
            let items: Vec<(u64, u64, Gate, Option<u64>)> = (0..3)
                .map(|i| {
                    (
                        mixed_raw[i.min(mixed_raw.len() - 1)],
                        m[i],
                        g[i],
                        Some(lw.ln[i]),
                    )
                })
                .collect();
            qkv = c.msg_gates(t, &items)?;
        } else if let Some(m) = msgs {
            let g = lw.gates.as_ref().context("gates")?;
            qkv.clear();
            for i in 0..3 {
                let p = mixed_raw[i.min(mixed_raw.len() - 1)];
                let p = c.msg_gate(p, t, m[i], &g[i])?;
                qkv.push(c.rmsnorm(Reg::Visit, p, t, D, lw.ln[i])?);
            }
        }
        let (qi, ki, vin) = (qkv[0], qkv[1], qkv[2]);
        // INK25: the attention projections + o of a PREFILL chunk (> 8 rows, eager, one sequence)
        // may take the matrix-core trunk; verify / batch / multi-row / captured forms never do
        c.p29_live = t > 8 && !self.graph && !self.vrow && self.mrow == 0 && !self.batch;
        c.i8_live = c.i8 && c.p29_live;
        let attn = self.attention(c, cache, l, vi, t, qi, ki, vin, fused);
        c.i8_live = false;
        c.p29_live = false;
        let attn = attn?;
        let boundary = l % BLOCK == 0;
        if fused {
            // INK10: (push reduce) + attn sconv + mlp depth in one launch; (push reduce) + mlp
            // sconv + the next visit's input depth in one launch
            let k = cache.context("fused without a cache")?;
            let rseq = self.rseq(c, k, t)?;
            let (own, xs) = match attn {
                Red::Done(a) => (a, None),
                Red::Pushed(a, n) => (a, Some((0, n, false))),
            };
            let resid = c.z4(Reg::Visit, t * D, "sconv.out")?;
            if boundary {
                blocks.push(x);
            }
            let mut st = blocks.clone();
            st.push(resid);
            let h = c.z4(Reg::Visit, t * D, "depth.out")?;
            let ch = self.chain(c, k, vi, 2, t)?;
            c.xsd(
                own,
                xs,
                t,
                lw.attn_sconv,
                k.convs[&(vi, 2)],
                rseq,
                if boundary { 0 } else { x },
                resid,
                &st,
                &[lw.mlp_q.addr],
                None,
                &[h],
                ch,
            )?;
            let mut h = h;
            if let Some(m) = msgs {
                let g3 = lw.gates.as_ref().context("gates")?[3];
                h = if msg_fused {
                    c.msg_gates(t, &[(h, m[3], g3, None)])?[0]
                } else {
                    c.msg_gate(h, t, m[3], &g3)?
                };
            }
            let (own, xs) = match self.moe(c, l, h, t, true)? {
                Red::Done(a) => (a, None),
                Red::Pushed(a, _) => (a, Some((1, 0, true))),
            };
            let xo = c.z4(Reg::Pass, t * D, "sconv.out")?;
            let (qs, gs, rep, outs) = match &ahead {
                Some((qs, gs, rep)) => {
                    let mut outs = Vec::with_capacity(qs.len());
                    for _ in 0..qs.len() {
                        outs.push(c.z4(Reg::Pass, t * D, "depth.next")?);
                    }
                    (qs.clone(), gs.clone(), *rep, outs)
                }
                None => (Vec::new(), None, false, Vec::new()),
            };
            let mut st = blocks.clone();
            st.push(xo);
            let ch = self.chain(c, k, vi, 3, t)?;
            c.xsd(
                own,
                xs,
                t,
                lw.mlp_sconv,
                k.convs[&(vi, 3)],
                rseq,
                resid,
                xo,
                if qs.is_empty() { &[] } else { &st },
                &qs,
                gs.as_deref(),
                &outs,
                ch,
            )?;
            let _ = rep;
            return Ok((xo, ahead.map(|_| outs)));
        }
        let attn = match attn {
            Red::Done(a) => a,
            Red::Pushed(..) => bail!("a pushed reduce outside the fused form"),
        };
        let zst = c.z4(Reg::Visit, 3 * D, "zst")?;
        if let (Some(k), true) = (cache, self.mrow > 0) {
            // INK20: B sequences x mrow chained rows: each row's sconv over its own sequence's
            // rows + its slot's state; the history entry per sequence (the commit writes states)
            let (s2, s3) = (k.convs[&(vi, 2)], k.convs[&(vi, 3)]);
            c.stash_m(s2, k.rseq_m, attn, D, t, self.mrow, D, s2)?;
            let resid = c.sconv_m(
                attn,
                t,
                self.mrow,
                lw.attn_sconv,
                s2,
                k.rseq_m,
                if boundary { 0 } else { x },
                Reg::Visit,
            )?;
            if boundary {
                blocks.push(x);
            }
            let mut st = blocks.clone();
            st.push(resid);
            let mut h = c.depth(t, &st, &[lw.mlp_q.addr], None)?[0];
            if let Some(m) = msgs {
                h = c.msg_gate(h, t, m[3], &lw.gates.as_ref().context("gates")?[3])?;
            }
            let Red::Done(m) = self.moe(c, l, h, t, false)? else {
                bail!("a pushed MoE reduce outside the fused form")
            };
            c.stash_m(s3, k.rseq_m, m, D, t, self.mrow, D, s3)?;
            let xo = c.sconv_m(
                m,
                t,
                self.mrow,
                lw.mlp_sconv,
                s3,
                k.rseq_m,
                resid,
                Reg::Pass,
            )?;
            return Ok((xo, None));
        }
        if let (Some(k), true) = (cache, self.batch) {
            // INK9 task 4: every row its own sequence's sconv state (by slot)
            let resid = c.sconv_rows(
                attn,
                t,
                lw.attn_sconv,
                k.convs[&(vi, 2)],
                k.rseq_b,
                if boundary { 0 } else { x },
                Reg::Visit,
            )?;
            if boundary {
                blocks.push(x);
            }
            let mut st = blocks.clone();
            st.push(resid);
            let mut h = c.depth(t, &st, &[lw.mlp_q.addr], None)?[0];
            if let Some(m) = msgs {
                h = c.msg_gate(h, t, m[3], &lw.gates.as_ref().context("gates")?[3])?;
            }
            let Red::Done(m) = self.moe(c, l, h, t, false)? else {
                bail!("a pushed MoE reduce outside the fused form")
            };
            let xo = c.sconv_rows(
                m,
                t,
                lw.mlp_sconv,
                k.convs[&(vi, 3)],
                k.rseq_b,
                resid,
                Reg::Pass,
            )?;
            return Ok((xo, None));
        }
        let (s2i, s3i, s2o, s3o) = match cache {
            None => (zst, zst, 0, 0),
            Some(k) => {
                let so = self.slot * 3 * D * 4;
                let (s2o, s3o) = (k.convs[&(vi, 2)] + so, k.convs[&(vi, 3)] + so);
                if self.pos0 == 0 || k.ctl_reset {
                    (zst, zst, s2o, s3o)
                } else {
                    (s2o, s3o, s2o, s3o)
                }
            }
        };
        c.stash(s2i, attn, D, t, D, s2o)?;
        let resid = c.sconv(
            Reg::Visit,
            attn,
            t,
            lw.attn_sconv,
            s2i,
            if boundary { 0 } else { x },
            s2o,
        )?;
        if boundary {
            blocks.push(x);
        }
        let mut st = blocks.clone();
        st.push(resid);
        let mut h = c.depth(t, &st, &[lw.mlp_q.addr], None)?[0];
        if let Some(m) = msgs {
            h = c.msg_gate(h, t, m[3], &lw.gates.as_ref().context("gates")?[3])?;
        }
        let Red::Done(m) = self.moe(c, l, h, t, false)? else {
            bail!("a pushed MoE reduce outside the fused form")
        };
        c.stash(s3i, m, D, t, D, s3o)?;
        Ok((
            c.sconv(Reg::Pass, m, t, lw.mlp_sconv, s3i, resid, s3o)?,
            None,
        ))
    }

    /// The forward over `t` rows of the embedding at `emb` (`[t, D]` f32 on the card): returns the
    /// softcapped logits' address (rows `head_from..t`, row stride `vpad`) and that stride.
    fn run(
        &self,
        c: &mut Card<'_, '_>,
        cache: Option<&Cache>,
        emb: u64,
        t: u64,
        head_from: u64,
    ) -> anyhow::Result<(u64, u64)> {
        // INK39: the multi verify's levers read `mv_live` (cleared on every exit of the pass)
        c.mv_live = self.mrow > 0;
        let r = self.run_plain(c, cache, emb, t, head_from);
        c.mv_live = false;
        r
    }

    fn run_plain(
        &self,
        c: &mut Card<'_, '_>,
        cache: Option<&Cache>,
        emb: u64,
        t: u64,
        head_from: u64,
    ) -> anyhow::Result<(u64, u64)> {
        let mut rs = self.run_begin(c, cache, emb, t)?;
        for vi in 0..self.sched.len() {
            self.run_visit(c, cache, &mut rs, vi, t)?;
        }
        self.run_end(c, rs, t, Some(head_from))
    }

    /// INK36: the prefill chunk's entry -- the two-half pipeline when `ov` is on and the chunk
    /// qualifies, else [`Forward::run`] (a stand-down by rule is counted `p36_n[3]`).
    fn run_chunk(
        &self,
        c: &mut Card<'_, '_>,
        cache: Option<&Cache>,
        emb: u64,
        t: u64,
        head_from: u64,
    ) -> anyhow::Result<(u64, u64)> {
        if c.p36 & P36_OV == 0 {
            return self.run(c, cache, emb, t, head_from);
        }
        let h = (t / 2) / 64 * 64;
        let form = t >= OV_MIN_ROWS
            && !self.graph
            && !self.batch
            && !self.vrow
            && self.mrow == 0
            && c.taps.is_none()
            && c.hist.is_none()
            && !c.push
            && head_from >= h
            && self.peer.is_some_and(|p| p.dev);
        let k = match cache {
            Some(k) if form && !k.ctl_reset => k,
            _ => {
                if t >= OV_MIN_ROWS {
                    c.p36_n[3] += 1;
                }
                return self.run(c, cache, emb, t, head_from);
            }
        };
        // the attention's routing floor must route both halves as the whole chunk
        if (self.pos0 + h > c.fa_floor) != (self.pos0 + t > c.fa_floor) {
            c.p36_n[3] += 1;
            return self.run(c, cache, emb, t, head_from);
        }
        let fb = Forward {
            pos0: self.pos0 + h,
            sched: self.sched.clone(),
            ..*self
        };
        c.ov_begin()?;
        let mut ra = self.run_begin(c, Some(k), emb, h)?;
        c.ov_swap(h)?;
        let mut rb = fb.run_begin(c, Some(k), emb + h * D * 4, t - h)?;
        c.ov_swap(h)?;
        for vi in 0..self.sched.len() {
            self.run_visit(c, Some(k), &mut ra, vi, h)?;
            c.ov_mark(vi)?;
            c.ov_swap(h)?;
            c.ov_wait(vi)?;
            fb.run_visit(c, Some(k), &mut rb, vi, t - h)?;
            c.ov_swap(h)?;
        }
        if c.tapdev.is_some() {
            self.run_end(c, ra, h, None)?;
        }
        c.ov_swap(h)?;
        let r = fb.run_end(c, rb, t - h, Some(head_from - h))?;
        c.ov_end(h)?;
        Ok(r)
    }

    /// INK36: [`Forward::run`]'s state between visits (so two row halves can interleave).
    fn run_begin(
        &self,
        c: &mut Card<'_, '_>,
        cache: Option<&Cache>,
        emb: u64,
        t: u64,
    ) -> anyhow::Result<RunSt> {
        Ok(RunSt {
            x: c.rmsnorm(Reg::Pass, emb, t, D, self.w.embed_norm)?,
            blocks: Vec::new(),
            pre: None,
            prev: None,
            fused: self.fused(c, cache, t),
            pre_next: None,
            // INK11 (`mdep`): the pass's 48 messages, computed at the pass's first visit
            mdep: c.fast && c.fuse & FUSE_MDEP != 0 && t <= 8,
            pass_msgs: None,
        })
    }

    /// INK36: one visit of [`Forward::run`] (visit index `vi`).
    fn run_visit(
        &self,
        c: &mut Card<'_, '_>,
        cache: Option<&Cache>,
        rs: &mut RunSt,
        vi: usize,
        t: u64,
    ) -> anyhow::Result<()> {
        let l = self.sched[vi];
        let RunSt {
            x,
            blocks,
            pre,
            prev,
            fused,
            pre_next,
            mdep,
            pass_msgs,
        } = rs;
        let (fused, mdep) = (*fused, *mdep);
        {
            c.reset(Reg::Visit)?;
            let mut msgs = None;
            if l == CS && !self.noloop {
                match &*pre {
                    None => *pre = Some((*x, blocks.clone())),
                    Some((px, pb)) => {
                        *prev = Some((blocks[pb.len() + 1..].to_vec(), *x));
                        *x = *px;
                        *blocks = pb.clone();
                    }
                }
            }
            if let Some((pb, px)) = &*prev {
                if l < CE && !self.nomsg {
                    let mut st = pb.clone();
                    st.push(*px);
                    if mdep {
                        if l == CS {
                            *pass_msgs = Some(c.depth_mq(
                                t,
                                &st,
                                self.w.msg_q.row(0, D),
                                u64::from(CE - CS),
                            )?);
                        }
                        let o0 = pass_msgs.context("mdep: no pass messages")?;
                        msgs = Some(
                            (0..4)
                                .map(|i| o0 + ((u64::from(l - CS) * 4 + i) * t * D) * 4)
                                .collect(),
                        );
                    } else {
                        let q: Vec<u64> = (0..4)
                            .map(|i| self.w.msg_q.row(u64::from(l - CS) * 4 + i, D))
                            .collect();
                        msgs = Some(c.depth(t, &st, &q, None)?);
                    }
                }
            }
            // INK10: the next visit's input depth rides this visit's fused mlp sconv unless the
            // next visit restarts the loop (x replaced by the saved pass input) or has no depth
            let ahead = if fused {
                match self.sched.get(vi + 1) {
                    None => Some((vec![self.w.final_q.addr], None, false)),
                    Some(&ln) => {
                        let restart = ln == CS && !self.noloop && pre.is_some();
                        if restart || ln == 0 {
                            None
                        } else {
                            let next_msgs = prev.is_some() && ln < CE && !self.nomsg;
                            Some(self.in_depth(ln, next_msgs)?)
                        }
                    }
                }
            } else {
                None
            };
            if let (Some((tb, tp)), Some(ti)) = (c.tapdev, TAP_VISITS.iter().position(|&v| v == vi))
            {
                c.launch(
                    "ink_dr_copy",
                    [(t * D).div_ceil(256).min(1024), 1, 1],
                    256,
                    &[A::P(*x), A::P(tb + ti as u64 * tp), A::L((t * D) as i64)],
                )?;
            }
            if c.taps.is_some() && TAP_VISITS.contains(&vi) {
                c.hip.sync_stream(&c.stream)?;
                let row = c.read(*x, (t * D * 4) as usize)?;
                c.taps.as_mut().context("taps")?.push(row);
            }
            let (xo, nxt) = self.layer(
                c,
                cache,
                l,
                vi,
                t,
                *x,
                blocks,
                msgs.as_deref(),
                pre_next.take(),
                ahead,
            )?;
            *x = xo;
            *pre_next = nxt;
        }
        Ok(())
    }

    /// INK36: [`Forward::run`]'s tail after the last visit (the final depth, norm, head).
    fn run_end(
        &self,
        c: &mut Card<'_, '_>,
        rs: RunSt,
        t: u64,
        head_from: Option<u64>,
    ) -> anyhow::Result<(u64, u64)> {
        let RunSt {
            x,
            blocks,
            mut pre_next,
            ..
        } = rs;
        c.reset(Reg::Visit)?;
        let mut st = blocks.clone();
        st.push(x);
        let h = match pre_next.take() {
            Some(p) => p[0],
            None => c.depth(t, &st, &[self.w.final_q.addr], None)?[0],
        };
        let hn = c.rmsnorm(Reg::Visit, h, t, D, self.w.norm)?;
        if let Some((tb, tp)) = c.tapdev {
            c.launch(
                "ink_dr_copy",
                [(t * D).div_ceil(256).min(1024), 1, 1],
                256,
                &[A::P(hn), A::P(tb + 3 * tp), A::L((t * D) as i64)],
            )?;
        }
        if c.taps.is_some() {
            c.hip.sync_stream(&c.stream)?;
            let row = c.read(hn, (t * D * 4) as usize)?;
            c.taps.as_mut().context("taps")?.push(row);
        }
        // INK36 `ov`: half A runs no head (its tail only for the drafter's tap)
        let Some(head_from) = head_from else {
            return Ok((0, 0));
        };
        let rows = t - head_from;
        let hs = hn + head_from * D * 4;
        let vpad: u64 = self.w.head.iter().map(Tw::rows).sum();
        let lg = c.z4(Reg::Pass, rows * vpad, "logits")?;
        let mut col = 0;
        for hw in &self.w.head {
            c.gemm(hs, rows, D, hw, lg + col * 4, vpad, false)?;
            col += hw.rows();
        }
        c.launch(
            "ink_softcap",
            [1024, 1, 1],
            256,
            &[A::P(lg), A::L((rows * vpad) as i64), A::F(30.0)],
        )?;
        if let Some(pr) = self.peer.filter(|p| p.dev) {
            // INK8: the head's vocab halves gathered on the card: rank 0 = [its half | the
            // peer's], read over P2P; rank 1 only answers the handshake (rows 0)
            let theirs = pr.peer_addr(c, lg)?;
            let full = 2 * vpad;
            if pr.rank == 0 {
                let out = c.z4(Reg::Pass, rows * full, "logits.full")?;
                c.xchg(
                    (lg, vpad),
                    (theirs, vpad),
                    (out, full),
                    rows,
                    vpad,
                    vpad,
                    false,
                    pr.link,
                )?;
                return Ok((out, full));
            }
            c.xchg(
                (lg, vpad),
                (theirs, vpad),
                (0, full),
                0,
                vpad,
                vpad,
                false,
                pr.link,
            )?;
            return Ok((lg, full));
        }
        Ok((lg, vpad))
    }
}

// ============================================================================================ runs

fn npy_rows(p: &Path) -> anyhow::Result<usize> {
    if !p.is_file() {
        return Ok(0);
    }
    let b = std::fs::read(p)?;
    let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
    let h = String::from_utf8_lossy(&b[10..10 + hl]);
    let s = h.split("'shape': (").nth(1).context("npy shape")?;
    Ok(s.split(',')
        .next()
        .unwrap_or("0")
        .trim()
        .parse()
        .unwrap_or(0))
}

fn ids_of(v: &Value) -> Vec<u32> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(Value::as_u64)
                .map(|x| x as u32)
                .collect()
        })
        .unwrap_or_default()
}

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best as u32
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// A pass: reset the pass region, upload the embedding, run, read back the rows (each `VOCAB`
/// f32, unpadded).
/// INK16: [`pass`]'s forward without the read-back: the logits rows' (address, row stride f32),
/// queued (not synchronised).
fn pass_lg(
    c: &mut Card<'_, '_>,
    fw: &Forward<'_>,
    pk: &PackReader,
    cache: Option<&Cache>,
    ids: &[u32],
    head_from: u64,
) -> anyhow::Result<(u64, u64)> {
    c.reset(Reg::Pass)?;
    let t = ids.len() as u64;
    let emb = c.z4(Reg::Pass, t * D, "emb")?;
    c.write(emb, &pk.embed_rows(ids)?)?;
    let hg = hg_form(c, fw, t);
    if hg {
        c.hg_begin()?;
    }
    let r = fw.run_chunk(c, cache, emb, t, head_from);
    if hg {
        c.hg_end()?;
    }
    r
}

/// INK36 `hg`: the lazy visit zeroing runs on an eager prefill chunk (one sequence) of at least
/// `HG_MIN_ROWS` rows: below it the per-take memsets (~24 a visit) cost more than the small
/// region's one reset (served single TTFT p50 +2 ms at the 8-row floor, s2 / s2b).
fn hg_form(c: &Card<'_, '_>, fw: &Forward<'_>, t: u64) -> bool {
    c.p36 & P36_HG != 0 && t >= HG_MIN_ROWS && !fw.graph && !fw.batch && !fw.vrow && fw.mrow == 0
}

fn pass(
    c: &mut Card<'_, '_>,
    fw: &Forward<'_>,
    pk: &PackReader,
    cache: Option<&Cache>,
    ids: &[u32],
    head_from: u64,
) -> anyhow::Result<Vec<u8>> {
    let t = ids.len() as u64;
    let (lg, vpad) = pass_lg(c, fw, pk, cache, ids, head_from)?;
    c.hip.sync_stream(&c.stream)?;
    if let Some(pr) = fw.peer.filter(|p| p.dev) {
        if pr.rank == 1 {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity((t - head_from) as usize * VOCAB * 4);
        for r in 0..t - head_from {
            out.extend(c.read(lg + r * vpad * 4, VOCAB * 4)?);
        }
        return Ok(out);
    }
    if let Some(pr) = fw.peer {
        // the head's vocab halves gathered on the host: rank 1 sends its rows, rank 0 concatenates
        let rows = (t - head_from) as usize;
        let mine = c.read(lg, rows * vpad as usize * 4)?;
        if pr.rank == 1 {
            pr.rows_tx.send(mine).context("peer gone")?;
            return Ok(Vec::new());
        }
        let theirs = pr.rows_rx.recv().context("peer gone")?;
        let tv = theirs.len() / rows / 4;
        let mut out = Vec::with_capacity(rows * VOCAB * 4);
        for r in 0..rows {
            let mut row = mine[r * vpad as usize * 4..(r + 1) * vpad as usize * 4].to_vec();
            row.extend_from_slice(&theirs[r * tv * 4..(r + 1) * tv * 4]);
            row.truncate(VOCAB * 4);
            out.extend_from_slice(&row);
        }
        return Ok(out);
    }
    let mut out = Vec::with_capacity((t - head_from) as usize * VOCAB * 4);
    for r in 0..t - head_from {
        out.extend(c.read(lg + r * vpad * 4, VOCAB * 4)?);
    }
    Ok(out)
}

/// The executor's entry (`--ink-run tf|decode`).
#[allow(clippy::too_many_lines, clippy::needless_range_loop)]
pub fn run(
    p: &Pack,
    tp: u32,
    prebuilt: &Path,
    hip_lib: &str,
    device: i32,
    f: &InkFlags,
) -> anyhow::Result<()> {
    if f.ink_serve.is_some() {
        ensure!(
            tp == 2 && f.ink_run.is_none() && !f.ink_split,
            "--ink-serve runs the two-card process: --tp 2, no --ink-run / --ink-split"
        );
        return run_two_card(p, prebuilt, hip_lib, f);
    }
    let mode = f.ink_run.clone().unwrap_or_default();
    ensure!(
        mode == "tf"
            || mode == "decode"
            || mode == "taps"
            || (tp == 2
                && (mode == "long"
                    || mode == "pfc"
                    || mode == "batch"
                    || mode == "vprice"
                    || mode == "mtp"
                    || mode == "mverify"
                    || mode == "mtpm")),
        "--ink-run {mode:?}: tf | decode (two cards: also long | batch)"
    );
    if tp == 2 {
        ensure!(
            !f.ink_split,
            "--tp 2 --ink-run: the two-card process (--ink-split is the one-card form)"
        );
        return run_two_card(p, prebuilt, hip_lib, f);
    }
    ensure!(
        tp == 1,
        "--ink-run is the one-card executor: --tp 1 (--ink-split = the TP=2 split on that card)"
    );
    let rf = f
        .ink_ref
        .as_ref()
        .context("--ink-run needs --ink-ref <oracle run dir>")?;
    let out = f
        .ink_out
        .as_ref()
        .context("--ink-run needs --ink-out DIR")?;
    std::fs::create_dir_all(out)?;
    let only: Vec<String> = f
        .ink_only
        .split(',')
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();
    ensure!(!only.is_empty(), "--ink-only p00,...");
    let wtp = if f.ink_split { 2 } else { 1 };
    let t0 = Instant::now();
    let hip = HipThread::spawn(hip_lib)?;
    hip.set_device(device)?;
    let lc = LoadCounters::new();
    let ps = PluginSet::load_named(&hip, prebuilt, &PLUGINS, &lc)?;
    let k = Kernels::new(&ps, &hip)?;
    let mut c = Card::new(&hip, k, f.ink_pass_mib << 20, f.ink_visit_mib << 20)?;
    c.fast = match f.ink_bodies.as_str() {
        "fast" => true,
        "v0" => false,
        o => bail!("--ink-bodies {o:?}: fast (the default) | v0 (the INK3b bodies, the control)"),
    };
    c.fuse = fuse_bits(&f.ink_fuse)?;
    (c.pf, c.cl, c.push) = prefill_gemm(&f.ink_prefill_gemm)?;
    c.i8 = prefill_trunk(&f.ink_prefill_trunk)?;
    (c.fa, c.fa_sym) = prefill_attn_sym(&f.ink_prefill_attn)?;
    c.fa_floor = f.ink_prefill_attn_floor;
    c.p29 = ink29_bits(&f.ink29)?;
    c.p35 = ink35_bits(&f.ink35)?;
    (c.p36, c.p36_xw) = ink36_bits(&f.ink36)?;
    // INK36: the two-half pipeline exchanges over the pair's link words (TP=2 only)
    c.p36 &= !P36_OV;
    c.p37 = ink37_bits(&f.ink37)?;
    (c.p40, c.p40_fam) = ink40_bits(&f.ink40)?;
    c.p39 = ink39_bits(&f.ink_p39)?;
    c.fz = c.alloc_zero(FZ_BYTES)?;
    let pk = PackReader::open(p)?.with_trunk(f, &p.dir, false)?;
    let x = Experts::load(&pk, &mut c, None)?;
    let w = Weights::load(&pk, &mut c, wtp, None)?;
    let (free, total) = hip.mem_info()?;
    println!(
        "ink7: weights on card {device} in {:.1} s (TP={wtp}{}), {:.2} of {:.2} GiB used",
        t0.elapsed().as_secs_f64(),
        if f.ink_split {
            " split on one card"
        } else {
            ""
        },
        (total - free) as f64 / f64::from(1 << 30),
        total as f64 / f64::from(1 << 30)
    );
    let fw = Forward {
        w: &w,
        x: &x,
        sched: schedule(f.ink_control == "noloop"),
        noloop: f.ink_control == "noloop",
        nomsg: f.ink_control == "nomsg",
        pos0: 0,
        graph: false,
        peer: None,
        slot: 0,
        batch: false,
        vrow: false,
        mrow: 0,
    };
    let tag = format!(
        "tp{wtp}{}",
        if f.ink_control.is_empty() {
            String::new()
        } else {
            format!("_{}", f.ink_control)
        }
    );
    let mut rep = serde_json::Map::new();
    let mut ok = true;
    let mut tap_cache: Option<(Cache, u64)> = None;
    for pid in &only {
        let meta: Value =
            serde_json::from_str(&std::fs::read_to_string(rf.join(pid).join("meta.json"))?)?;
        let p_ids = ids_of(&meta["token_ids"]);
        let gen = ids_of(&meta["generated_ids"]);
        let n0 = p_ids.len();
        if mode == "taps" {
            // INK12 task 1: the card's own greedy continuation (eos stops it), then ONE eager
            // teacher-forced pass over prompt + continuation with the drafter's taps read back
            let steps = f.ink_steps as usize;
            // one cache for every prompt (the slot law: a sequence at position 0 reads zero conv
            // states and only the KV rows it wrote); per-prompt allocations ran card 0 out at p25
            if tap_cache.is_none() {
                tap_cache = Some((
                    Cache::new(&mut c, &w, &fw.sched, f.ink_cache_ctx, f.ink_local_ring, 1)?,
                    c.alloc_zero(D * 4)?,
                ));
            }
            let (cache, emb_in) = tap_cache.as_ref().context("cache")?;
            let (cache, emb_in) = (cache, *emb_in);
            let mut fwd = Forward {
                pos0: 0,
                sched: fw.sched.clone(),
                ..fw
            };
            let last = pass(&mut c, &fwd, &pk, Some(cache), &p_ids, n0 as u64 - 1)?;
            let mut nxt = argmax(&f32s(&last[last.len() - VOCAB * 4..]));
            let mut toks = Vec::new();
            for i in 0..steps {
                toks.push(nxt);
                if nxt == 200_006 {
                    break;
                }
                fwd.pos0 = (n0 + i) as u64;
                c.write(cache.rpos_dec, &((n0 + i) as i32).to_le_bytes())?;
                c.write(emb_in, &pk.embed_rows(&[nxt])?)?;
                c.reset(Reg::Pass)?;
                let (lg, _) = fwd.run(&mut c, Some(cache), emb_in, 1, 0)?;
                c.hip.sync_stream(&c.stream)?;
                nxt = argmax(&f32s(&c.read(lg, VOCAB * 4)?));
            }
            fwd.pos0 = 0;
            let mut feed = p_ids.clone();
            feed.extend_from_slice(&toks);
            c.taps = Some(Vec::new());
            let rows = pass(&mut c, &fwd, &pk, None, &feed, n0 as u64 - 1)?;
            let taps = c.taps.take().context("taps")?;
            ensure!(taps.len() == 4, "taps: {} of 4 read", taps.len());
            // the pass's own greedy tokens must be the decode's (decode == tf, INK5's law)
            let tf_toks: Vec<u32> = (0..toks.len())
                .map(|i| argmax(&f32s(&rows[i * VOCAB * 4..(i + 1) * VOCAB * 4])))
                .collect();
            let same = tf_toks.iter().zip(&toks).filter(|(a, b)| a == b).count();
            std::fs::write(out.join(format!("{pid}.taps.f32")), taps.concat())?;
            std::fs::write(
                out.join(format!("{pid}.taps.json")),
                serde_json::to_string(&json!({"prompt": p_ids, "gen": toks, "rows": feed.len(),
                    "tf_greedy_equal": same, "visits": [5, 23, 41, 48]}))?,
            )?;
            println!(
                "TAPS {pid}: n {n0} gen {} rows {} tf==dec {same}/{}",
                toks.len(),
                feed.len(),
                toks.len()
            );
            rep.insert(
                pid.clone(),
                json!({"n": n0, "gen": toks.len(), "tf_greedy_equal": same}),
            );
            continue;
        }
        if mode == "tf" {
            // INK4's kl feed: prompt + the oracle's greedy tokens (as many as it has top rows)
            let nrow = npy_rows(&rf.join(pid).join("gen_top_ids.npy"))?;
            let mut feed = p_ids.clone();
            feed.extend_from_slice(&gen[..nrow.min(gen.len())]);
            let l0 = c.k.launches;
            let ts = Instant::now();
            let rows = pass(&mut c, &fw, &pk, None, &feed, n0 as u64 - 1)?;
            let ms = ts.elapsed().as_secs_f64() * 1e3;
            std::fs::write(out.join(format!("{pid}.{tag}.tf.f32")), &rows)?;
            let r0 = f32s(&rows[..VOCAB * 4]);
            println!(
                "TF {tag} {pid}: n {n0} rows {} first {} launches {} {ms:.0} ms",
                rows.len() / (VOCAB * 4),
                argmax(&r0),
                c.k.launches - l0
            );
            rep.insert(pid.clone(), json!({"n": n0, "rows": rows.len() / (VOCAB * 4), "launches": c.k.launches - l0, "ms": ms}));
            continue;
        }
        // ---- decode (INK5)
        let steps = f.ink_steps as usize;
        let cache = Cache::new(&mut c, &w, &fw.sched, f.ink_cache_ctx, f.ink_local_ring, 1)?;
        let emb_in = c.alloc_zero(D * 4)?;
        let mut fwd = Forward {
            pos0: 0,
            sched: fw.sched.clone(),
            ..fw
        };
        let last = pass(&mut c, &fwd, &pk, Some(&cache), &p_ids, n0 as u64 - 1)?;
        let mut nxt = argmax(&f32s(&last[last.len() - VOCAB * 4..]));
        let mut toks = Vec::new();
        let mut rows: Vec<Vec<u8>> = Vec::new();
        let mut t_eager = Vec::new();
        let step = |c: &mut Card<'_, '_>,
                    fwd: &Forward<'_>,
                    cache: &Cache,
                    tok: u32,
                    pos: u64|
         -> anyhow::Result<Vec<u8>> {
            c.write(cache.rpos_dec, &(pos as i32).to_le_bytes())?;
            c.write(emb_in, &pk.embed_rows(&[tok])?)?;
            c.reset(Reg::Pass)?;
            let (lg, _) = fwd.run(c, Some(cache), emb_in, 1, 0)?;
            c.hip.sync_stream(&c.stream)?;
            c.read(lg, VOCAB * 4)
        };
        let l0 = c.k.launches;
        for i in 0..steps {
            toks.push(nxt);
            fwd.pos0 = (n0 + i) as u64;
            let ts = Instant::now();
            let lg = step(&mut c, &fwd, &cache, nxt, (n0 + i) as u64)?;
            t_eager.push(ts.elapsed().as_secs_f64() * 1e3);
            nxt = argmax(&f32s(&lg));
            rows.push(lg);
        }
        let launches_per_step = (c.k.launches - l0) / steps.max(1) as u64;
        let mut dec = Vec::new();
        for r in &rows {
            dec.extend_from_slice(r);
        }
        std::fs::write(out.join(format!("{pid}.{tag}.dec.f32")), &dec)?;
        std::fs::write(
            out.join(format!("{pid}.{tag}.dec.toks.json")),
            serde_json::to_string(&toks)?,
        )?;
        // A: the teacher-forced prefill of the same tokens (no cache)
        fwd.pos0 = 0;
        let mut feed = p_ids.clone();
        feed.extend_from_slice(&toks);
        let tf = pass(&mut c, &fwd, &pk, None, &feed, n0 as u64)?;
        let a_eq = (0..steps)
            .filter(|&i| rows[i][..] == tf[i * VOCAB * 4..(i + 1) * VOCAB * 4])
            .count();
        // control A: the conv states not carried (reset every step)
        let mut cc = Cache::new(&mut c, &w, &fw.sched, f.ink_cache_ctx, f.ink_local_ring, 1)?;
        fwd.pos0 = 0;
        pass(&mut c, &fwd, &pk, Some(&cc), &p_ids, n0 as u64 - 1)?;
        cc.ctl_reset = true;
        let mut ctl_a = 0;
        for i in 0..steps.min(4) {
            fwd.pos0 = (n0 + i) as u64;
            if step(&mut c, &fwd, &cc, toks[i], (n0 + i) as u64)? == rows[i] {
                ctl_a += 1;
            }
        }
        // B: one captured step, replayed (static inputs: the embedding row + the position word)
        let mut graph_rep = json!(null);
        let mut b_ok = true;
        if f.ink_capture {
            let cg = Cache::new(&mut c, &w, &fw.sched, f.ink_cache_ctx, f.ink_local_ring, 1)?;
            fwd.pos0 = 0;
            pass(&mut c, &fwd, &pk, Some(&cg), &p_ids, n0 as u64 - 1)?;
            c.hip.sync_stream(&c.stream)?;
            fwd.pos0 = n0 as u64; // any pos0 > 0 selects the decode form; the word is data
            fwd.graph = true;
            let lb = c.k.launches;
            c.hip
                .begin_capture(&c.stream, StreamCaptureMode::ThreadLocal)?;
            c.reset(Reg::Pass)?;
            let cap = fwd.run(&mut c, Some(&cg), emb_in, 1, 0);
            let g = c.hip.end_capture(&c.stream)?;
            let (lg_static, _) = cap?;
            let nodes = c.hip.graph_node_count(&g)?;
            let exec = c.hip.instantiate(&g)?;
            let cap_launches = c.k.launches - lb;
            let mut rows_g = Vec::new();
            let mut t_graph = Vec::new();
            for i in 0..steps {
                c.write(emb_in, &pk.embed_rows(&[toks[i]])?)?;
                c.write(cg.rpos_dec, &((n0 + i) as i32).to_le_bytes())?;
                c.hip.sync_stream(&c.stream)?;
                let ts = Instant::now();
                c.hip.graph_launch(&exec, &c.stream)?;
                c.hip.sync_stream(&c.stream)?;
                t_graph.push(ts.elapsed().as_secs_f64() * 1e3);
                rows_g.push(c.read(lg_static, VOCAB * 4)?);
            }
            fwd.graph = false;
            let b_eq = (0..steps).filter(|&i| rows_g[i] == rows[i]).count();
            // control B: one more replay at the STALE position word with the next token
            c.write(emb_in, &pk.embed_rows(&[*toks.last().context("steps")?])?)?;
            c.hip.graph_launch(&exec, &c.stream)?;
            c.hip.sync_stream(&c.stream)?;
            let ctl_b = c.read(lg_static, VOCAB * 4)? == rows[steps - 1];
            let mut gd = Vec::new();
            for r in &rows_g {
                gd.extend_from_slice(r);
            }
            std::fs::write(out.join(format!("{pid}.{tag}.graph.f32")), &gd)?;
            t_graph.sort_by(f64::total_cmp);
            b_ok = b_eq == steps && !ctl_b;
            graph_rep = json!({"B_rows_equal": b_eq, "B_ctl_stale_pos_equal": ctl_b, "nodes": nodes,
                               "capture_launches": cap_launches, "graph_ms_p50": t_graph[t_graph.len() / 2]});
            drop(exec);
            drop(g);
        }
        t_eager.sort_by(f64::total_cmp);
        let good = a_eq == steps && ctl_a == 0 && b_ok;
        ok &= good;
        let r = json!({"steps": steps, "A_rows_equal": a_eq, "A_ctl_reset_equal": ctl_a, "graph": graph_rep,
                       "launches_per_step": launches_per_step, "eager_ms_p50": t_eager[t_eager.len() / 2],
                       "toks": toks, "pass": good});
        println!("DEC {tag} {pid} {r}");
        rep.insert(pid.clone(), r);
        drop(cache);
    }
    rep.insert("visit_high_mib".into(), json!(c.visit_high >> 20));
    std::fs::write(
        out.join(format!("ink7_{mode}_{tag}.json")),
        serde_json::to_string_pretty(&Value::Object(rep))?,
    )?;
    if mode == "decode" {
        println!("INK7 DECODE {}", if ok { "PASS" } else { "FAIL" });
    }
    println!(
        "INK7 {mode} done in {:.1} s, {} launches",
        t0.elapsed().as_secs_f64(),
        c.k.launches
    );
    drop(c);
    drop(ps);
    drop(hip);
    ensure!(ok, "INK7 decode gate FAILED");
    Ok(())
}

/// INK7 task 3: the two-card TP=2 / EP=2 process (one process, two HIP threads, card r = rank r):
/// each rank its halves + its 64 experts a layer, the two reduces a visit as [`Peer::reduce`]
/// (P2P), the head's halves gathered on the host. Rows -> `<id>.tp2x.tf.f32`.
fn run_two_card(p: &Pack, prebuilt: &Path, hip_lib: &str, f: &InkFlags) -> anyhow::Result<()> {
    use std::sync::mpsc::channel;
    let (a_tx, b_rx) = channel();
    let (b_tx, a_rx) = channel();
    let (ra_tx, rb_rx) = channel();
    let (rb_tx, ra_rx) = channel();
    std::thread::scope(|s| {
        let h1 = s.spawn(move || rank_main(p, prebuilt, hip_lib, f, 1, b_tx, b_rx, rb_tx, rb_rx));
        let r0 = rank_main(p, prebuilt, hip_lib, f, 0, a_tx, a_rx, ra_tx, ra_rx);
        let r1 = h1.join().map_err(|_| anyhow::anyhow!("rank 1 panicked"))?;
        r0.and(r1)
    })
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::needless_range_loop
)]
fn rank_main(
    p: &Pack,
    prebuilt: &Path,
    hip_lib: &str,
    f: &InkFlags,
    rank: u32,
    tx: std::sync::mpsc::Sender<u64>,
    rx: std::sync::mpsc::Receiver<u64>,
    rows_tx: std::sync::mpsc::Sender<Vec<u8>>,
    rows_rx: std::sync::mpsc::Receiver<Vec<u8>>,
) -> anyhow::Result<()> {
    let t0 = Instant::now();
    let hip = HipThread::spawn(hip_lib)?;
    let dev_ = i32::try_from(rank)?;
    hip.set_device(dev_)?;
    ensure!(
        hip.can_access_peer(dev_, 1 - dev_)?,
        "card {dev_} cannot reach card {} over P2P",
        1 - dev_
    );
    hip.enable_peer_access(1 - dev_)?;
    let lc = LoadCounters::new();
    let ps = PluginSet::load_named(&hip, prebuilt, &PLUGINS, &lc)?;
    let k = Kernels::new(&ps, &hip)?;
    let mut c = Card::new(&hip, k, f.ink_pass_mib << 20, f.ink_visit_mib << 20)?;
    c.fast = match f.ink_bodies.as_str() {
        "fast" => true,
        "v0" => false,
        o => bail!("--ink-bodies {o:?}: fast (the default) | v0 (the INK3b bodies, the control)"),
    };
    let pk = PackReader::open(p)?.with_trunk(f, &p.dir, true)?;
    let x = Experts::load(&pk, &mut c, Some(rank))?;
    let w = Weights::load(&pk, &mut c, 2, Some(rank))?;
    let mut id = vec![0u8; (D * D * 2) as usize];
    for i in 0..D as usize {
        id[(i * D as usize + i) * 2..(i * D as usize + i) * 2 + 2]
            .copy_from_slice(&0x3F80u16.to_le_bytes());
    }
    let ident = Tw {
        addr: c.up(&id)?,
        es: 2,
        shape: vec![D, D],
        sc: 0,
    };
    let dev = match f.ink_reduce.as_str() {
        "p2p" => true,
        "gemm" => false,
        o => bail!("--ink-reduce {o:?}: p2p (the default) | gemm (the INK7 control)"),
    };
    // INK8: the link words (zeroed once) + the work bases, swapped once over the host link
    let link = c.alloc_zero(256)?;
    c.fuse = fuse_bits(&f.ink_fuse)?;
    (c.pf, c.cl, c.push) = prefill_gemm(&f.ink_prefill_gemm)?;
    c.i8 = prefill_trunk(&f.ink_prefill_trunk)?;
    (c.fa, c.fa_sym) = prefill_attn_sym(&f.ink_prefill_attn)?;
    c.fa_floor = f.ink_prefill_attn_floor;
    c.p29 = ink29_bits(&f.ink29)?;
    c.p35 = ink35_bits(&f.ink35)?;
    (c.p36, c.p36_xw) = ink36_bits(&f.ink36)?;
    c.p37 = ink37_bits(&f.ink37)?;
    (c.p40, c.p40_fam) = ink40_bits(&f.ink40)?;
    c.p39 = ink39_bits(&f.ink_p39)?;
    c.fz = c.alloc_zero(FZ_BYTES)?;
    // INK16: the push reduce's receive window (two slots), only when the arm asks for it
    // INK39 (`--ink-p39 push`): the same window for the multi verify's reduces
    let pwin = if (c.push || c.p39 & P39_PUSH != 0) && dev {
        c.alloc_zero(2 * PWIN_ROWS * D * 4)?
    } else {
        0
    };
    tx.send(c.base()).context("peer gone")?;
    tx.send(link).context("peer gone")?;
    tx.send(c.fz).context("peer gone")?;
    tx.send(pwin).context("peer gone")?;
    tx.send(w.head[0].rows()).context("peer gone")?;
    let peer_base = rx.recv().context("peer gone")?;
    let peer_link = rx.recv().context("peer gone")?;
    let pfz = rx.recv().context("peer gone")?;
    let ppwin = rx.recv().context("peer gone")?;
    ensure!(
        (pwin == 0) == (ppwin == 0),
        "the push windows differ across the ranks"
    );
    ensure!(
        rx.recv().context("peer gone")? == w.head[0].rows(),
        "the head halves differ in rows"
    );
    let peer = Peer {
        rank,
        tx,
        rx,
        rows_tx,
        rows_rx,
        ident,
        dev,
        peer_base,
        link: (link, peer_link),
        pfz,
        pwin: (pwin, ppwin),
    };
    let (free, total) = hip.mem_info()?;
    println!(
        "ink8 rank{rank}: card {dev_} weights in {:.1} s ({:.2} of {:.2} GiB used), peer access to card {}, reduce {}",
        t0.elapsed().as_secs_f64(),
        (total - free) as f64 / f64::from(1 << 30),
        total as f64 / f64::from(1 << 30),
        1 - dev_,
        f.ink_reduce
    );
    let fw = Forward {
        w: &w,
        x: &x,
        sched: schedule(f.ink_control == "noloop"),
        noloop: f.ink_control == "noloop",
        nomsg: f.ink_control == "nomsg",
        pos0: 0,
        graph: false,
        peer: Some(&peer),
        slot: 0,
        batch: false,
        vrow: false,
        mrow: 0,
    };
    if f.ink_serve.is_some() {
        let r = serve::rank_serve(&mut c, &fw, &pk, &w, f, link, &p.dir.join("draft"));
        drop(fw);
        drop(peer);
        drop(c);
        drop(ps);
        drop(hip);
        return r;
    }
    let rf = f.ink_ref.as_ref().context("--ink-ref")?;
    let out = f.ink_out.as_ref().context("--ink-out")?;
    std::fs::create_dir_all(out)?;
    let mode = f.ink_run.clone().unwrap_or_default();
    let tag = if dev { "tp2x" } else { "tp2g" };
    let only: Vec<&str> = f.ink_only.split(',').filter(|s| !s.is_empty()).collect();
    let mut rep = serde_json::Map::new();
    let mut ok = true;
    let emb_in = c.alloc_zero(D * 4)?;
    // the two decode caches a rank, REUSED prompt after prompt (the serve plane's slot law: a
    // new sequence at position 0 reads zero conv states and only the KV rows it wrote)
    let (cache, cg) = if mode == "decode" {
        (
            Some(Cache::new(
                &mut c,
                &w,
                &fw.sched,
                f.ink_cache_ctx,
                f.ink_local_ring,
                1,
            )?),
            Some(Cache::new(
                &mut c,
                &w,
                &fw.sched,
                f.ink_cache_ctx,
                f.ink_local_ring,
                1,
            )?),
        )
    } else {
        (None, None)
    };
    if mode == "batch" {
        let r = batch_gate(&mut c, &fw, &pk, &w, f, rf, out, &only, rank, tag);
        drop(fw);
        drop(peer);
        drop(c);
        drop(ps);
        drop(hip);
        return r;
    }
    if mode == "mtp" {
        let r = mtp_gate(&mut c, &fw, &pk, &w, f, p, rf, out, &only, rank);
        drop(fw);
        drop(peer);
        drop(c);
        drop(ps);
        drop(hip);
        return r;
    }
    if mode == "mtpm" {
        let r = mtpm_gate(&mut c, &fw, &pk, &w, f, p, rf, out, &only, rank);
        drop(fw);
        drop(peer);
        drop(c);
        drop(ps);
        drop(hip);
        return r;
    }
    if mode == "mverify" {
        let r = mverify_gate(&mut c, &fw, &pk, &w, f, rf, out, &only, rank);
        drop(fw);
        drop(peer);
        drop(c);
        drop(ps);
        drop(hip);
        return r;
    }
    if mode == "vprice" {
        let r = vprice(&mut c, &fw, &pk, &w, f, rf, &only, rank);
        drop(fw);
        drop(peer);
        drop(c);
        drop(ps);
        drop(hip);
        return r;
    }
    for pid in &only {
        let meta: Value =
            serde_json::from_str(&std::fs::read_to_string(rf.join(pid).join("meta.json"))?)?;
        let p_ids = ids_of(&meta["token_ids"]);
        if mode == "pfc" {
            // ---- INK27 task 1: the long-prompt prefill census. The prompt in `--ink-chunk`-row
            // chunks through the per-visit caches (the served chunk law), each chunk's host clock
            // and (`--ink-op-timing`) every launch timed -> pfcensus_<pid>_rank<r>.json; no oracle
            // rows are read (any meta.json with `token_ids` serves).
            let cache = Cache::new(&mut c, &w, &fw.sched, f.ink_cache_ctx, f.ink_local_ring, 1)?;
            let mut fwd = Forward {
                pos0: 0,
                graph: false,
                sched: fw.sched.clone(),
                ..fw
            };
            let n0 = p_ids.len();
            let ch = usize::try_from(f.ink_chunk)?;
            let (mut p0, mut k, mut chunks) = (0usize, 0usize, Vec::new());
            let tall = Instant::now();
            while p0 < n0 {
                let e = (p0 + ch).min(n0);
                fwd.pos0 = p0 as u64;
                c.hip.sync_stream(&c.stream)?;
                if f.ink_op_timing {
                    c.time_begin();
                }
                let ts = Instant::now();
                let l0 = c.k.launches;
                let lastrow = pass(
                    &mut c,
                    &fwd,
                    &pk,
                    Some(&cache),
                    &p_ids[p0..e],
                    (e - p0 - 1) as u64,
                )?;
                let ms = ts.elapsed().as_secs_f64() * 1e3;
                if rank == 0 {
                    // every chunk's last logits row (the arms' byte compare)
                    std::fs::write(out.join(format!("pfc_{pid}_c{k}.f32")), &lastrow)?;
                }
                let ops = if f.ink_op_timing {
                    c.time_end()?
                } else {
                    Vec::new()
                };
                let ksum: f32 = ops.iter().map(|o| o.2).sum();
                if rank == 0 {
                    println!(
                        "PFC {pid} chunk {k} pos0 {p0} rows {} host_ms {ms:.2} kernel_ms {ksum:.2} \
                         launches {}",
                        e - p0,
                        c.k.launches - l0
                    );
                }
                let v: Vec<Value> = ops.iter().map(|(s, g, ms)| json!([s, g, ms])).collect();
                chunks.push(
                    json!({"chunk": k, "pos0": p0, "rows": e - p0, "host_ms": ms,
                                   "ops": v}),
                );
                p0 = e;
                k += 1;
            }
            let tot = tall.elapsed().as_secs_f64() * 1e3;
            std::fs::write(
                out.join(format!("pfcensus_{pid}_rank{rank}.json")),
                serde_json::to_string(&json!({"prompt_rows": n0, "chunk": ch,
                                              "total_ms": tot, "chunks": chunks}))?,
            )?;
            if rank == 0 {
                println!("PFC {pid} total rows {n0} chunk {ch} {tot:.1} ms");
            }
            continue;
        }
        let gen = ids_of(&meta["generated_ids"]);
        let n0 = p_ids.len();
        let nrow = npy_rows(&rf.join(pid).join("gen_top_ids.npy"))?.min(gen.len());
        let mut feed = p_ids.clone();
        feed.extend_from_slice(&gen[..nrow]);
        let l0 = c.k.launches;
        // INK15: `--ink-op-timing` on a tf run times every launch of each prompt's prefill pass
        // (the prefill price census, `tools/capture/ink15/pcensus.py`)
        let timed = f.ink_op_timing && mode == "tf";
        if timed {
            c.hip.sync_stream(&c.stream)?;
            c.time_begin();
        }
        let ts = Instant::now();
        let tf = pass(&mut c, &fw, &pk, None, &feed, n0 as u64 - 1)?;
        let ms = ts.elapsed().as_secs_f64() * 1e3;
        if timed {
            let ops = c.time_end()?;
            let v: Vec<Value> = ops.iter().map(|(s, g, ms)| json!([s, g, ms])).collect();
            std::fs::write(
                out.join(format!("poptime_{pid}_rank{rank}.json")),
                serde_json::to_string(&v)?,
            )?;
        }
        if rank == 0 {
            std::fs::write(out.join(format!("{pid}.{tag}.tf.f32")), &tf)?;
            println!(
                "TF {tag} {pid}: n {n0} rows {} first {} launches/rank {} {ms:.0} ms (two cards) \
                 iu8 {:?} p29 {:?}",
                tf.len() / (VOCAB * 4),
                argmax(&f32s(&tf[..VOCAB * 4])),
                c.k.launches - l0,
                c.i8_n,
                c.p29_n
            );
        }
        if mode == "long" {
            // ---- INK9 task 3: chunk continuation (prefill at pos0 > 0) through the per-visit
            // caches with the local rings: the prompt in `--ink-chunk`-row chunks, then the
            // oracle's tokens fed one a step (eager) == the one-pass teacher-forced rows, bytes
            let cache = Cache::new(&mut c, &w, &fw.sched, f.ink_cache_ctx, f.ink_local_ring, 1)?;
            let mut fwd = Forward {
                pos0: 0,
                graph: false,
                sched: fw.sched.clone(),
                ..fw
            };
            let ch = usize::try_from(f.ink_chunk)?;
            let mut rows = Vec::new();
            let mut p0 = 0;
            let mut chunks = 0;
            while p0 < n0 {
                let e = (p0 + ch).min(n0);
                fwd.pos0 = p0 as u64;
                let r = pass(
                    &mut c,
                    &fwd,
                    &pk,
                    Some(&cache),
                    &p_ids[p0..e],
                    (e - p0 - 1) as u64,
                )?;
                if e == n0 {
                    rows = r;
                }
                p0 = e;
                chunks += 1;
            }
            for i in 0..nrow {
                fwd.pos0 = (n0 + i) as u64;
                c.write(emb_in, &pk.embed_rows(&[gen[i]])?)?;
                c.reset(Reg::Pass)?;
                let (lg, _) = fwd.run(&mut c, Some(&cache), emb_in, 1, 0)?;
                c.hip.sync_stream(&c.stream)?;
                if rank == 0 {
                    rows.extend(c.read(lg, VOCAB * 4)?);
                }
            }
            if rank == 0 {
                let n = rows.len() / (VOCAB * 4);
                let eq = (0..n)
                    .filter(|&i| {
                        rows[i * VOCAB * 4..(i + 1) * VOCAB * 4]
                            == tf[i * VOCAB * 4..(i + 1) * VOCAB * 4]
                    })
                    .count();
                let ring = cache.ring_pages * PAGE;
                std::fs::write(out.join(format!("{pid}.{tag}l{ring}.tf.f32")), &rows)?;
                let good = eq == n && n == nrow + 1;
                ok &= good;
                let r = json!({"prompt_rows": n0, "chunk": ch, "chunks": chunks,
                               "local_ring_rows": ring, "rows": n, "equal_to_one_pass": eq,
                               "pass": good});
                println!("LONG {tag} {pid} {r}");
                rep.insert((*pid).to_string(), r);
            }
            continue;
        }
        if mode != "decode" {
            continue;
        }
        // ---- INK8 task 1: the two-card decode on the rank-local per-visit cache, fed the
        // ORACLE's tokens (teacher-forced decode: every row comparable to run1's gen rows)
        let steps = nrow;
        if steps < 2 {
            println!("DEC {tag} {pid}: skipped ({steps} oracle decode rows)");
            continue;
        }
        let mut fwd = Forward {
            pos0: 0,
            graph: false,
            sched: fw.sched.clone(),
            ..fw
        };
        let cache = cache.as_ref().context("cache")?;
        let cg = cg.as_ref().context("cache")?;
        let mut drows = pass(&mut c, &fwd, &pk, Some(cache), &p_ids, n0 as u64 - 1)?;
        let mut t_eager = Vec::new();
        let l1 = c.k.launches;
        for i in 0..steps {
            fwd.pos0 = (n0 + i) as u64;
            c.write(emb_in, &pk.embed_rows(&[gen[i]])?)?;
            let timed = f.ink_op_timing && i == steps / 2;
            if timed {
                c.hip.sync_stream(&c.stream)?;
                c.time_begin();
            }
            let ts = Instant::now();
            c.reset(Reg::Pass)?;
            let (lg, ld) = fwd.run(&mut c, Some(cache), emb_in, 1, 0)?;
            c.hip.sync_stream(&c.stream)?;
            t_eager.push(ts.elapsed().as_secs_f64() * 1e3);
            if timed {
                let ops = c.time_end()?;
                let tot: f32 = ops.iter().map(|o| o.2).sum();
                println!(
                    "OPT {tag} {pid} rank{rank}: step {i} pos {} launches {} sum of kernel ms {tot:.3}",
                    n0 + i,
                    ops.len()
                );
                let v: Vec<Value> = ops.iter().map(|(s, g, ms)| json!([s, g, ms])).collect();
                std::fs::write(
                    out.join(format!("optime_{pid}_rank{rank}.json")),
                    serde_json::to_string(&v)?,
                )?;
            }
            if rank == 0 {
                let _ = ld;
                drows.extend(c.read(lg, VOCAB * 4)?);
            }
        }
        let lps = (c.k.launches - l1) / steps as u64;
        // A: decode == the two-card teacher-forced rows (bytes; every body is row-local and the
        // reduce elementwise)
        let a_eq = if rank == 0 {
            (0..=steps)
                .filter(|&i| {
                    drows[i * VOCAB * 4..(i + 1) * VOCAB * 4]
                        == tf[i * VOCAB * 4..(i + 1) * VOCAB * 4]
                })
                .count()
        } else {
            0
        };
        if rank == 0 {
            std::fs::write(out.join(format!("{pid}.{tag}d.tf.f32")), &drows)?;
        }
        // B: one captured step a rank, replayed (static inputs: the embedding row + the position
        // word); the reduces' handshake replays on the card
        fwd.pos0 = 0;
        pass(&mut c, &fwd, &pk, Some(cg), &p_ids, n0 as u64 - 1)?;
        c.hip.sync_stream(&c.stream)?;
        fwd.pos0 = n0 as u64;
        fwd.graph = true;
        let lb = c.k.launches;
        c.hip
            .begin_capture(&c.stream, StreamCaptureMode::ThreadLocal)?;
        c.reset(Reg::Pass)?;
        let cap = fwd.run(&mut c, Some(cg), emb_in, 1, 0);
        let g = c.hip.end_capture(&c.stream)?;
        let (lg_static, _) = cap?;
        fwd.graph = false;
        let nodes = c.hip.graph_node_count(&g)?;
        let exec = c.hip.instantiate(&g)?;
        let cap_launches = c.k.launches - lb;
        let mut t_graph = Vec::new();
        let mut b_eq = 0;
        for i in 0..steps {
            c.write(emb_in, &pk.embed_rows(&[gen[i]])?)?;
            c.write(cg.rpos_dec, &((n0 + i) as i32).to_le_bytes())?;
            c.hip.sync_stream(&c.stream)?;
            let ts = Instant::now();
            c.hip.graph_launch(&exec, &c.stream)?;
            c.hip.sync_stream(&c.stream)?;
            t_graph.push(ts.elapsed().as_secs_f64() * 1e3);
            if rank == 0
                && c.read(lg_static, VOCAB * 4)?[..]
                    == drows[(i + 1) * VOCAB * 4..(i + 2) * VOCAB * 4]
            {
                b_eq += 1;
            }
        }
        // control B: one more replay at the STALE position word
        c.write(emb_in, &pk.embed_rows(&[gen[steps - 1]])?)?;
        c.hip.graph_launch(&exec, &c.stream)?;
        c.hip.sync_stream(&c.stream)?;
        let ctl_b = rank == 0
            && c.read(lg_static, VOCAB * 4)?[..]
                == drows[steps * VOCAB * 4..(steps + 1) * VOCAB * 4];
        drop(exec);
        drop(g);
        t_eager.sort_by(f64::total_cmp);
        t_graph.sort_by(f64::total_cmp);
        if rank == 0 {
            let good = a_eq == steps + 1 && b_eq == steps && !ctl_b;
            ok &= good;
            let r = json!({"steps": steps, "A_dec_eq_tf": a_eq, "B_replay_eq_eager": b_eq,
                           "B_ctl_stale_pos_equal": ctl_b, "launches_per_step": lps,
                           "capture_launches": cap_launches, "nodes": nodes,
                           "eager_ms_p50": t_eager[t_eager.len() / 2],
                           "graph_ms_p50": t_graph[t_graph.len() / 2], "pass": good});
            println!("DEC {tag} {pid} {r}");
            rep.insert((*pid).to_string(), r);
        }
    }
    if rank == 0 {
        std::fs::write(
            out.join(format!("ink8_{mode}_{tag}.json")),
            serde_json::to_string_pretty(&Value::Object(rep))?,
        )?;
        if mode == "decode" {
            println!("INK8 DECODE {tag} {}", if ok { "PASS" } else { "FAIL" });
        }
        if mode == "long" {
            println!("INK9 LONG {tag} {}", if ok { "PASS" } else { "FAIL" });
        }
    }
    println!(
        "INK8 {tag} rank{rank} done in {:.1} s, {} launches",
        t0.elapsed().as_secs_f64(),
        c.k.launches
    );
    let diag = c.fuse_diag()?;
    println!(
        "INK10 {tag} rank{rank}: fuse {} ({:#x}), push-spin timeouts {diag}",
        f.ink_fuse, c.fuse
    );
    if c.p36 != 0 {
        println!(
            "INK36 {tag} rank{rank}: p36 {:?} ({:#x}, xw {}): ov chunks {} / hg chunks {} / \
             lazy memsets {} / ov stand-downs {}, visit high {} MiB",
            f.ink36,
            c.p36,
            c.p36_xw,
            c.p36_n[0],
            c.p36_n[1],
            c.p36_n[2],
            c.p36_n[3],
            c.visit_high >> 20
        );
    }
    if c.p40 != 0 {
        println!(
            "CONS40 {tag} rank{rank}: p40 {:?} ({:#x}, fam {}): wm128 launches {} / undezeroed takes {}",
            f.ink40, c.p40, c.p40_fam, c.p40_n, c.p40_zd
        );
    }
    ok &= diag == 0;
    drop(fw);
    drop(peer);
    drop(c);
    drop(ps);
    drop(hip);
    ensure!(ok, "INK8 two-card decode gate FAILED");
    Ok(())
}

/// INK12 tasks 2-4: the drafter (rank 0) + the verify chunk (INK9's chain, both ranks) + the
/// conv rollback, greedy, per prompt: the committed tokens must equal the NON-MTP greedy tokens of
/// `--ink-mtp-ref` (the `taps` run's, decode == tf) -- the byte law (a chunk's rows == the decode
/// rows); `--ink-mtp-control nocommit` skips the rollback and must differ. Round 0's k drafter
/// logits rows are dumped for the oracle gate (`dspark_ref.py rows`).
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn mtp_gate(
    c: &mut Card<'_, '_>,
    fw: &Forward<'_>,
    pk: &PackReader,
    w: &Weights,
    f: &InkFlags,
    pack: &Pack,
    rf: &Path,
    out: &Path,
    only: &[&str],
    rank: u32,
) -> anyhow::Result<()> {
    let pr = fw.peer.context("mtp: the two-card process")?;
    let k = f.ink_mtp_k;
    let n_tok = f.ink_mtp_tokens as usize;
    let mref = f.ink_mtp_ref.as_ref().context("--ink-mtp-ref <taps dir>")?;
    // INK13: the rollback controls -- `nocommit` (no rollback), `shift` (the rollback one row
    // off: keep + 1, or keep - 1 at a full accept), `zero` (every logged state zeroed); each must
    // DIFFER from the reference
    let ctl = f.ink_mtp_control.as_str();
    ensure!(
        matches!(ctl, "" | "none" | "nocommit" | "shift" | "zero"),
        "--ink-mtp-control {ctl:?}: none | nocommit | shift | zero"
    );
    let nocommit = !matches!(ctl, "" | "none");
    let graph = f.ink_mtp_graph;
    ensure!(
        graph || matches!(ctl, "" | "none" | "nocommit"),
        "--ink-mtp-control {ctl} needs --ink-mtp-graph"
    );
    let emb_v = c.alloc_zero(8 * D * 4)?;
    let zeros = c.alloc_zero(3 * D * 4)?;
    let mut vgs: BTreeMap<u64, draft::VGraph> = BTreeMap::new();
    let cache = Cache::new(c, w, &fw.sched, f.ink_cache_ctx, f.ink_local_ring, 1)?;
    // INK33 (`--ink-mtp-gate`): the confidence-gated depth (graph form; drafts at k, verifies the
    // confident prefix; every round's p1s recorded in the json)
    // INK34: the harness reads the serve default `auto` as off (the gate needs --ink-mtp-graph)
    let gate = serve::mtp_gate(if f.ink_mtp_gate == "auto" {
        ""
    } else {
        &f.ink_mtp_gate
    })?;
    ensure!(
        gate.is_none() || graph,
        "--ink-mtp-gate needs --ink-mtp-graph"
    );
    let mut dr = if rank == 0 {
        let mut o = draft::DraftOpt::of(f.draft_vocab(), &f.ink_draft_gemv, &f.ink_draft_layers)?;
        o.conf = gate.is_some();
        Some(draft::Drafter::load(
            c,
            pk,
            &pack.dir.join("draft"),
            &w.head[0],
            f.ink_cache_ctx,
            1024,
            &o,
        )?)
    } else {
        None
    };
    let hcap = 3 * (3 + 8) * D * 4 * 48 * 2;
    c.hist_spare = Some(draft::Hist {
        buf: c.alloc_zero(hcap)?,
        cap: hcap,
        tab: c.alloc_zero(3 * 8 * 512)?,
        ..Default::default()
    });
    let send = |v: &[u32]| -> anyhow::Result<()> {
        pr.rows_tx
            .send(v.iter().flat_map(|x| x.to_le_bytes()).collect())
            .context("peer gone")
    };
    let recv = || -> anyhow::Result<Vec<u32>> {
        let b = pr.rows_rx.recv().context("peer gone")?;
        Ok(b.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    };
    let mut rep = serde_json::Map::new();
    let mut ok = true;
    let (mut t_draft, mut t_ver, mut t_post, mut rounds_all, mut toks_all) =
        (0f64, 0f64, 0f64, 0usize, 0usize);
    let (mut ndiff, mut nprompts) = (0usize, 0usize);
    for pid in only {
        let meta: Value =
            serde_json::from_str(&std::fs::read_to_string(rf.join(pid).join("meta.json"))?)?;
        let p_ids = ids_of(&meta["token_ids"]);
        let n0 = p_ids.len();
        // the reference: a `taps` run's dir (TP=1 greedy) or a `--ink-mtp-k 0` run's json (the
        // two-card non-MTP greedy: one row a step, the served decode form)
        let want = if mref.is_file() {
            let v: Value = serde_json::from_str(&std::fs::read_to_string(mref)?)?;
            ids_of(&v[*pid]["got"])
        } else {
            let tj: Value = serde_json::from_str(&std::fs::read_to_string(
                mref.join(format!("{pid}.taps.json")),
            )?)?;
            ids_of(&tj["gen"])
        };
        let mut fwd = Forward {
            pos0: 0,
            graph: false,
            sched: fw.sched.clone(),
            ..*fw
        };
        if let Some(d) = dr.as_mut() {
            d.nctx = 0;
            c.tapdev = Some((d.taps, d.tcap * D * 4));
        }
        let last = pass(c, &fwd, pk, Some(&cache), &p_ids, n0 as u64 - 1)?;
        c.tapdev = None;
        let mut tok = if rank == 0 {
            let t = argmax(&f32s(&last[last.len() - VOCAB * 4..]));
            send(&[t])?;
            dr.as_mut().context("dr")?.append(c, n0 as u64, 0)?;
            t
        } else {
            recv()?[0]
        };
        let mut got = vec![tok];
        let mut p = n0 as u64;
        let mut accs = Vec::new();
        let mut r0 = Vec::new();
        let mut vsha: Vec<String> = Vec::new();
        let mut gp1: Vec<Value> = Vec::new();
        while got.len() < n_tok && tok != 200_006 {
            let kk = k.min((f.ink_cache_ctx - p - 8).min(7));
            let ts = Instant::now();
            let feed: Vec<u32> = if kk == 0 {
                vec![tok]
            } else if rank == 0 {
                let d = dr.as_mut().context("dr")?;
                if accs.is_empty() {
                    // the oracle gate's input: this card's own taps of the prompt rows
                    let mut tp = Vec::new();
                    for i in 0..4 {
                        tp.extend(c.read(d.taps + i * d.tcap * D * 4, n0 * D as usize * 4)?);
                    }
                    std::fs::write(out.join(format!("{pid}.taps.f32")), &tp)?;
                    std::fs::write(
                        out.join(format!("{pid}.taps.json")),
                        serde_json::to_string(&json!({"prompt": p_ids, "gen": [tok], "rows": n0}))?,
                    )?;
                }
                if f.ink_op_timing && graph && rounds_all == 4 {
                    // INK18: the drafter census -- one eager round with per-launch event pairs
                    // (`poptime_draft_k<k>_rank0.json`), then the captured round alone (device
                    // p50 of 24 replays) and its prep (host: embed row + norm + the blocking write)
                    d.issue(c, pk, tok, kk)?;
                    d.drafts(c, kk)?;
                    c.hip.sync_stream(&c.stream)?;
                    c.time_begin();
                    d.body(c, kk, false)?;
                    let ops = c.time_end()?;
                    let v: Vec<Value> = ops.iter().map(|(s, g, ms)| json!([s, g, ms])).collect();
                    std::fs::write(
                        out.join(format!("poptime_draft_k{kk}_rank0.json")),
                        serde_json::to_string(&v)?,
                    )?;
                    let (mut tg, mut tp) = (Vec::new(), Vec::new());
                    for _ in 0..24 {
                        let ts = Instant::now();
                        d.prep(c, pk, tok, kk)?;
                        tp.push(ts.elapsed().as_secs_f64() * 1e3);
                        let ts = Instant::now();
                        d.issue(c, pk, tok, kk)?;
                        c.hip.sync_stream(&c.stream)?;
                        tg.push(ts.elapsed().as_secs_f64() * 1e3);
                    }
                    tg.sort_by(f64::total_cmp);
                    tp.sort_by(f64::total_cmp);
                    println!(
                        "DPRICE k {kk}: eager kernel sum {:.3} ms ({} launches); issue + sync p50 {:.3} ms (p10 {:.3}); prep p50 {:.3} ms ({:?})",
                        ops.iter().map(|o| f64::from(o.2)).sum::<f64>(),
                        ops.len(),
                        tg[12],
                        tg[2],
                        tp[12],
                        d.opt
                    );
                }
                let dd = if let (true, Some((prod, tau, dmin, _))) = (graph, gate) {
                    d.issue(c, pk, tok, kk)?;
                    let (mut dd, p1) = d.drafts_conf(c, kk)?;
                    let g = serve::gated_depth(&p1, prod, tau, dmin.min(kk));
                    dd.truncate(g as usize);
                    gp1.push(json!([p1, g]));
                    dd
                } else if graph {
                    d.issue(c, pk, tok, kk)?;
                    d.drafts(c, kk)?
                } else {
                    let (dd, lg) = d.draft(c, pk, tok, kk, accs.is_empty())?;
                    if accs.is_empty() {
                        r0 = lg;
                    }
                    dd
                };
                let mut fd = vec![tok];
                fd.extend_from_slice(&dd);
                send(&fd)?;
                fd
            } else {
                recv()?
            };
            let td = ts.elapsed().as_secs_f64() * 1e3;
            let ts = Instant::now();
            let t = feed.len() as u64;
            let rows = if graph && t > 1 {
                if let std::collections::btree_map::Entry::Vacant(slot_e) = vgs.entry(t) {
                    let taps = dr.as_ref().map(|d| (d.taps, d.tcap * D * 4));
                    let vg = capture_verify(c, &fwd, &cache, emb_v, t, 0, taps)?;
                    if rank == 0 {
                        println!(
                            "INK13 verify graph T {t}: {} launches, {} nodes, {} convs logged",
                            vg.launches, vg.nodes, vg.n
                        );
                    }
                    slot_e.insert(vg);
                }
                let vg = &vgs[&t];
                replay_verify(c, pk, &cache, emb_v, vg, &feed, p, 0)?;
                let mut out = Vec::new();
                if rank == 0 {
                    for a in 0..t {
                        out.extend(c.read(vg.lg + a * vg.ld * 4, VOCAB * 4)?);
                    }
                }
                out
            } else {
                ensure!(!graph, "a one-row verify in the graph harness (k 0)");
                c.hist = c.hist_spare.take();
                if let Some(d) = dr.as_ref() {
                    c.tapdev = Some((d.taps, d.tcap * D * 4));
                }
                fwd.pos0 = p;
                let rows = pass(c, &fwd, pk, Some(&cache), &feed, 0)?;
                c.tapdev = None;
                rows
            };
            if rank == 0 {
                use sha2::Digest;
                let hx = sha2::Sha256::digest(&rows);
                vsha.push(
                    hx[..8]
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect::<String>(),
                );
            }
            let tv = ts.elapsed().as_secs_f64() * 1e3;
            let ts = Instant::now();
            let (keep, newt) = if rank == 0 {
                let mut a = 0;
                let mut newt = Vec::new();
                loop {
                    let t = argmax(&f32s(&rows[a * VOCAB * 4..(a + 1) * VOCAB * 4]));
                    newt.push(t);
                    if a < feed.len() - 1 && t == feed[a + 1] && t != 200_006 {
                        a += 1;
                    } else {
                        break;
                    }
                }
                let keep = a + 1;
                let mut m = vec![keep as u32];
                m.extend_from_slice(&newt);
                send(&m)?;
                (keep, newt)
            } else {
                let m = recv()?;
                (m[0] as usize, m[1..].to_vec())
            };
            if graph && t > 1 {
                let vg = &vgs[&t];
                match ctl {
                    "nocommit" => {}
                    "shift" => c.commit_tab(
                        vg,
                        if keep < t as usize {
                            keep + 1
                        } else {
                            keep - 1
                        } as u64,
                    )?,
                    "zero" => c.commit_zero(vg, zeros)?,
                    _ => c.commit_tab(vg, keep as u64)?,
                }
            } else if nocommit {
                let mut h = c.hist.take().context("hist")?;
                h.off = 0;
                h.entries.clear();
                h.max_w = 0;
                c.hist_spare = Some(h);
            } else {
                c.commit(keep as u64)?;
            }
            if let Some(d) = dr.as_mut() {
                d.append(c, keep as u64, p)?;
            }
            c.hip.sync_stream(&c.stream)?;
            let tp = ts.elapsed().as_secs_f64() * 1e3;
            accs.push(keep - 1);
            t_draft += td;
            t_ver += tv;
            t_post += tp;
            rounds_all += 1;
            toks_all += newt.len();
            tok = *newt.last().context("walk")?;
            got.extend_from_slice(&newt);
            p += keep as u64;
        }
        got.truncate(n_tok);
        let m = got.len().min(want.len());
        let same = (0..m).take_while(|&i| got[i] == want[i]).count();
        let good = same == m;
        if nocommit {
            ndiff += usize::from(!good);
        } else {
            ok &= good || rank == 1;
        }
        nprompts += 1;
        if rank == 0 {
            if !r0.is_empty() {
                std::fs::write(out.join(format!("{pid}.r0.draft.f32")), &r0)?;
            }
            let mean = accs.iter().sum::<usize>() as f64 / accs.len().max(1) as f64;
            let r = json!({"tokens": got.len(), "equal_prefix": same, "compared": m, "rounds": accs.len(),
                           "accepted_mean": mean, "accs": accs, "got": got, "pass": good, "vsha": vsha,
                           "gate_p1": gp1});
            println!("MTP {pid}: k {k} tokens {} rounds {} accepted/verify {mean:.2} equal prefix {same}/{m} {}",
                     got.len(), accs.len(), if good { "EQUAL" } else { "DIFFER" });
            rep.insert((*pid).to_string(), r);
        }
    }
    if nocommit && rank == 0 {
        // a control passes when it changes the output on a MAJORITY of the prompts
        ok = 2 * ndiff > nprompts;
        println!("INK13 MTP control {ctl}: output differs on {ndiff}/{nprompts} prompts");
    }
    if rank == 0 {
        let n = rounds_all.max(1) as f64;
        let msg = format!(
            "MTPTIME k {k}: rounds {rounds_all} tokens {toks_all} ({:.3}/round) draft {:.2} verify {:.2} post {:.2} ms a round -> {:.2} ms a token (host clock, {})",
            toks_all as f64 / n, t_draft / n, t_ver / n, t_post / n, (t_draft + t_ver + t_post) / toks_all.max(1) as f64,
            if graph { "graph" } else { "eager" }
        );
        println!("{msg}");
        rep.insert("time".into(), json!(msg));
        std::fs::write(
            out.join(format!(
                "mtp_k{k}{}{}.json",
                if graph { "_graph" } else { "" },
                if nocommit {
                    format!("_{ctl}")
                } else {
                    String::new()
                }
            )),
            serde_json::to_string_pretty(&Value::Object(rep))?,
        )?;
        println!(
            "INK12 MTP {} ({})",
            if ok { "PASS" } else { "FAIL" },
            if nocommit {
                "a rollback control: must differ"
            } else {
                "byte law"
            }
        );
    }
    drop(dr);
    drop(cache);
    ensure!(ok, "INK12 mtp gate FAILED");
    Ok(())
}

/// INK13: capture a verify chunk of `t` rows of cache slot `slot` as ONE graph -- INK9's chain
/// with the rows' slot / position words as data (`Cache::rseq_v` / `rpos_v`, [`replay_verify`]),
/// every conv's stash (the rollback log; its table uploaded once: the graph's addresses never
/// move), and the drafter's taps when `taps` (rank 0). Both ranks capture in lockstep (the peer
/// offsets are exchanged on the host at capture; the reduces' handshakes are call-keyed device
/// counters, so a replay needs no host step).
fn capture_verify(
    c: &mut Card<'_, '_>,
    fw: &Forward<'_>,
    cache: &Cache,
    emb: u64,
    t: u64,
    slot: u64,
    taps: Option<(u64, u64)>,
) -> anyhow::Result<draft::VGraph> {
    ensure!((2..=8).contains(&t), "a verify chunk of {t} rows (2..8)");
    let fwd = Forward {
        pos0: 1,
        graph: false,
        vrow: true,
        slot,
        batch: false,
        sched: fw.sched.clone(),
        ..*fw
    };
    c.hip.sync_stream(&c.stream)?;
    let mut h = c.hist_spare.take().context("no verify history buffer")?;
    h.off = 0;
    h.entries.clear();
    h.max_w = 0;
    c.hist = Some(h);
    c.tapdev = taps;
    let l0 = c.k.launches;
    c.hip
        .begin_capture(&c.stream, StreamCaptureMode::ThreadLocal)?;
    c.reset(Reg::Pass)?;
    let r = fwd.run(c, Some(cache), emb, t, 0);
    let g = c.hip.end_capture(&c.stream)?;
    c.tapdev = None;
    let mut h = c.hist.take().context("hist")?;
    let (lg, ld) = r?;
    let launches = c.k.launches - l0;
    let n = h.entries.len() as u64 / 3;
    let ent = h.entries.clone();
    let tab: Vec<u8> = h.entries.iter().flat_map(|v| v.to_le_bytes()).collect();
    let max_w = h.max_w;
    h.off = 0;
    h.entries.clear();
    h.max_w = 0;
    c.hist_spare = Some(h);
    let tab = c.up(&tab)?;
    let e = c.hip.instantiate(&g)?;
    let nodes = c.hip.graph_node_count(&g)?;
    Ok(draft::VGraph {
        _g: g,
        e,
        lg,
        ld,
        tab,
        n,
        max_w,
        t,
        launches,
        nodes,
        ent,
    })
}

/// INK20: capture a multi-sequence verify chunk (`b` sequences x `w` chained rows) as ONE graph:
/// the rows' slot / position words are data (`Cache::rseq_m` / `rpos_m`), every conv's history
/// entry per sequence; the rollback = `ink_mv_commit` on the graph's table at the keeps in
/// `Cache::keep_m`.
fn capture_mverify(
    c: &mut Card<'_, '_>,
    fw: &Forward<'_>,
    cache: &Cache,
    emb: u64,
    b: u64,
    w: u64,
    taps: Option<(u64, u64)>,
) -> anyhow::Result<draft::VGraph> {
    let t = b * w;
    ensure!(
        (1..=8).contains(&b) && (2..=8).contains(&w) && t <= MV_ROWS,
        "a multi verify of {b} x {w} rows"
    );
    let fwd = Forward {
        pos0: 1,
        graph: false,
        vrow: false,
        mrow: w,
        slot: 0,
        batch: false,
        sched: fw.sched.clone(),
        ..*fw
    };
    c.hip.sync_stream(&c.stream)?;
    let mut h = c.hist_spare.take().context("no verify history buffer")?;
    h.off = 0;
    h.entries.clear();
    h.max_w = 0;
    c.hist = Some(h);
    c.tapdev = taps;
    let l0 = c.k.launches;
    c.hip
        .begin_capture(&c.stream, StreamCaptureMode::ThreadLocal)?;
    c.reset(Reg::Pass)?;
    let r = fwd.run(c, Some(cache), emb, t, 0);
    let g = c.hip.end_capture(&c.stream)?;
    c.tapdev = None;
    let mut h = c.hist.take().context("hist")?;
    let (lg, ld) = r?;
    let launches = c.k.launches - l0;
    let n = h.entries.len() as u64 / 3;
    let ent = h.entries.clone();
    let tab: Vec<u8> = h.entries.iter().flat_map(|v| v.to_le_bytes()).collect();
    ensure!(
        tab.len() as u64 <= 3 * 8 * 512,
        "the history table holds {n} convs"
    );
    let max_w = h.max_w;
    h.off = 0;
    h.entries.clear();
    h.max_w = 0;
    c.hist_spare = Some(h);
    let tab = c.up(&tab)?;
    let e = c.hip.instantiate(&g)?;
    let nodes = c.hip.graph_node_count(&g)?;
    Ok(draft::VGraph {
        _g: g,
        e,
        lg,
        ld,
        tab,
        n,
        max_w,
        t,
        launches,
        nodes,
        ent,
    })
}

/// INK20: the rows' slot / position words of a multi-sequence verify (`rows` = (slot, first
/// position) a sequence, `w` rows each) as the one `rseq_m` / `rpos_m` write.
fn mverify_words(rows: &[(u64, u64)], w: u64) -> anyhow::Result<Vec<u8>> {
    let mut sq = Vec::with_capacity(8 * MV_ROWS as usize);
    let mut ps = Vec::with_capacity(4 * MV_ROWS as usize);
    for &(slot, pos) in rows {
        for j in 0..w {
            sq.extend_from_slice(&i32::try_from(slot)?.to_le_bytes());
            ps.extend_from_slice(&i32::try_from(pos + j)?.to_le_bytes());
        }
    }
    ensure!(
        sq.len() as u64 <= 4 * MV_ROWS,
        "a multi verify past {MV_ROWS} rows"
    );
    sq.resize(4 * MV_ROWS as usize, 0);
    ps.resize(4 * MV_ROWS as usize, 0);
    sq.extend_from_slice(&ps);
    Ok(sq)
}

/// INK20: roll a captured multi verify back to each sequence's keep (one write + one launch).
fn commit_mverify(
    c: &mut Card<'_, '_>,
    cache: &Cache,
    vg: &draft::VGraph,
    w: u64,
    keeps: &[u32],
) -> anyhow::Result<()> {
    let b = vg.t / w;
    ensure!(
        keeps.len() as u64 == b && keeps.iter().all(|&k| (1..=w).contains(&u64::from(k))),
        "keeps {keeps:?} of a {b} x {w} verify"
    );
    let kb: Vec<u8> = keeps.iter().flat_map(|k| k.to_le_bytes()).collect();
    c.write(cache.keep_m, &kb)?;
    c.commit_m_launch(cache, vg.tab, vg.n, vg.max_w, b, w)
}

/// INK20 (`--ink-run mverify`): the multi-sequence verify byte law. For every `BxW` cell of
/// `--ink-mseq`: B prompts prefilled into slots 0..B-1, then rounds of a W-row verify a sequence
/// (the anchor + W - 1 drafts taken from the non-MTP greedy reference, some corrupted on a fixed
/// pattern so the keeps vary), sequence s running `--ink-mseq-rounds` - s % 4 rounds (the live
/// count walks down). Arm `single`: each sequence's verify alone (the served b1 verify form,
/// eager), the eager commit. Arm `multi`: every live sequence in ONE chunk (eager) + the
/// per-sequence commit; arm `mgraph`: the same on a captured graph a live count. Law: every row of
/// every round sha256-equal across the three arms (the walk follows the rows, so equal rows =
/// equal schedules), and the committed ids == the reference.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::needless_range_loop
)]
fn mverify_gate(
    c: &mut Card<'_, '_>,
    fw: &Forward<'_>,
    pk: &PackReader,
    w: &Weights,
    f: &InkFlags,
    rf: &Path,
    out: &Path,
    only: &[&str],
    rank: u32,
) -> anyhow::Result<()> {
    use sha2::Digest;
    let pr = fw.peer.context("mverify: the two-card process")?;
    let mref = f.ink_mtp_ref.as_ref().context("--ink-mtp-ref <k0 json>")?;
    let refj: Value = serde_json::from_str(&std::fs::read_to_string(mref)?)?;
    let send = |v: &[u32]| -> anyhow::Result<()> {
        pr.rows_tx
            .send(v.iter().flat_map(|x| x.to_le_bytes()).collect())
            .context("peer gone")
    };
    let recv = || -> anyhow::Result<Vec<u32>> {
        let b = pr.rows_rx.recv().context("peer gone")?;
        Ok(b.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    };
    // the greedy walk a sequence on rank 0 (keep, committed ids), exchanged with rank 1
    let walk_x = |rows: &[Vec<u8>], feeds: &[Vec<u32>]| -> anyhow::Result<Vec<(u32, Vec<u32>)>> {
        if rank == 0 {
            let mut v = Vec::new();
            for (b, fd) in feeds.iter().enumerate() {
                let mut a = 0;
                let mut newt = Vec::new();
                loop {
                    let t = argmax(&f32s(&rows[b][a * VOCAB * 4..(a + 1) * VOCAB * 4]));
                    newt.push(t);
                    if a + 1 < fd.len() && t == fd[a + 1] {
                        a += 1;
                    } else {
                        break;
                    }
                }
                v.push(((a + 1) as u32, newt));
            }
            let mut m = Vec::new();
            for (k, nt) in &v {
                m.push(*k);
                m.push(nt.len() as u32);
                m.extend_from_slice(nt);
            }
            send(&m)?;
            Ok(v)
        } else {
            let m = recv()?;
            let mut v = Vec::new();
            let mut i = 0;
            while i < m.len() {
                let n = m[i + 1] as usize;
                v.push((m[i], m[i + 2..i + 2 + n].to_vec()));
                i += 2 + n;
            }
            Ok(v)
        }
    };
    let hcap = 3 * (3 + 8) * D * 4 * 48 * 2 * 6;
    c.hist_spare = Some(draft::Hist {
        buf: c.alloc_zero(hcap)?,
        cap: hcap,
        tab: c.alloc_zero(3 * 8 * 512)?,
        ..Default::default()
    });
    let emb_m = c.alloc_zero(MV_ROWS * D * 4)?;
    let mut rep = serde_json::Map::new();
    let mut ok = true;
    for cell in f.ink_mseq.split(',').filter(|s| !s.is_empty()) {
        let (bs, ws) = cell.split_once('x').context("--ink-mseq BxW")?;
        let (nb, vw): (usize, u64) = (bs.parse()?, ws.parse()?);
        ensure!(
            (2..=8).contains(&nb)
                && (2..=8).contains(&vw)
                && nb as u64 * vw <= MV_ROWS
                && nb <= only.len(),
            "--ink-mseq {cell}: 2..8 sequences (<= --ink-only) x 2..8 rows (<= 32 rows)"
        );
        let mut seqs: Vec<(String, Vec<u32>, Vec<u32>)> = Vec::new();
        for pid in &only[..nb] {
            let meta: Value =
                serde_json::from_str(&std::fs::read_to_string(rf.join(pid).join("meta.json"))?)?;
            seqs.push((
                (*pid).to_string(),
                ids_of(&meta["token_ids"]),
                ids_of(&refj[*pid]["got"]),
            ));
        }
        let rounds = |s: usize| f.ink_mseq_rounds as usize - s % 4;
        // per arm: [seq][round] -> (row sha256 prefixes, keep); committed ids per seq
        let mut res: BTreeMap<&str, (Vec<Vec<(Vec<String>, u32)>>, Vec<Vec<u32>>)> =
            BTreeMap::new();
        let mut tms: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let mut mv_timed = std::collections::BTreeSet::new();
        for arm in ["single", "multi", "mgraph"] {
            let cache = Cache::new(
                c,
                w,
                &fw.sched,
                f.ink_cache_ctx,
                f.ink_local_ring,
                nb as u64,
            )?;
            let mut toks: Vec<u32> = Vec::new();
            let mut pos: Vec<u64> = Vec::new();
            for (s, (_, p_ids, _)) in seqs.iter().enumerate() {
                let fwd = Forward {
                    pos0: 0,
                    graph: false,
                    slot: s as u64,
                    batch: false,
                    sched: fw.sched.clone(),
                    ..*fw
                };
                let last = pass(c, &fwd, pk, Some(&cache), p_ids, p_ids.len() as u64 - 1)?;
                let t = if rank == 0 {
                    let t = argmax(&f32s(&last[last.len() - VOCAB * 4..]));
                    send(&[t])?;
                    t
                } else {
                    recv()?[0]
                };
                toks.push(t);
                pos.push(p_ids.len() as u64);
            }
            let mut got: Vec<Vec<u32>> = toks.iter().map(|&t| vec![t]).collect();
            let mut log: Vec<Vec<(Vec<String>, u32)>> = vec![Vec::new(); nb];
            let mut vgs: BTreeMap<u64, draft::VGraph> = BTreeMap::new();
            let maxr = (0..nb).map(rounds).max().unwrap_or(0);
            for r in 0..maxr {
                let live: Vec<usize> = (0..nb).filter(|&s| r < rounds(s)).collect();
                // the feeds: the anchor + W - 1 drafts (the reference, corrupted on a pattern)
                let feeds: Vec<Vec<u32>> = live
                    .iter()
                    .map(|&s| {
                        let g = &got[s];
                        let want = &seqs[s].2;
                        let mut fd = vec![*g.last().unwrap_or(&0)];
                        for i in 1..vw as usize {
                            let mut d = want.get(g.len() - 1 + i).copied().unwrap_or(13);
                            if (r + s) % 3 == 1 && i + 1 == vw as usize
                                || (r + s) % 5 == 2 && i == 1
                            {
                                d = (d + 1) % VOCAB as u32;
                            }
                            fd.push(d);
                        }
                        fd
                    })
                    .collect();
                // the rows, one Vec of VOCAB f32 rows a live sequence (rank 0)
                let mut rows: Vec<Vec<u8>> = vec![Vec::new(); live.len()];
                let mut single_kn: Vec<(u32, Vec<u32>)> = Vec::new();
                let mut tdev = 0f64;
                let ts = Instant::now();
                if arm == "single" {
                    for (b, &s) in live.iter().enumerate() {
                        let fwd = Forward {
                            pos0: pos[s],
                            graph: false,
                            slot: s as u64,
                            batch: false,
                            sched: fw.sched.clone(),
                            ..*fw
                        };
                        c.hist = c.hist_spare.take();
                        rows[b] = pass(c, &fwd, pk, Some(&cache), &feeds[b], 0)?;
                        let kn = walk_x(&rows[b..=b], &feeds[b..=b])?;
                        c.commit(u64::from(kn[0].0))?;
                        single_kn.extend(kn);
                    }
                } else {
                    let all: Vec<u32> = feeds.concat();
                    let t = all.len() as u64;
                    let words = mverify_words(
                        &live.iter().map(|&s| (s as u64, pos[s])).collect::<Vec<_>>(),
                        vw,
                    )?;
                    c.write(emb_m, &pk.embed_rows(&all)?)?;
                    c.write(cache.rseq_m, &words)?;
                    let (lg, ld) = if arm == "multi" {
                        let fwd = Forward {
                            pos0: 1,
                            graph: false,
                            slot: 0,
                            batch: false,
                            vrow: false,
                            mrow: vw,
                            sched: fw.sched.clone(),
                            ..*fw
                        };
                        c.hist = c.hist_spare.take();
                        c.reset(Reg::Pass)?;
                        // INK21: the multi verify census (`poptime_mv<t>_rank<r>.json`, one
                        // eager chunk of each row count with per-launch event pairs)
                        let timed = f.ink_op_timing && mv_timed.insert(t);
                        if timed {
                            c.time_begin();
                        }
                        let r = fwd.run(c, Some(&cache), emb_m, t, 0)?;
                        if timed {
                            let ops = c.time_end()?;
                            let v: Vec<Value> =
                                ops.iter().map(|(s, g, ms)| json!([s, g, ms])).collect();
                            std::fs::write(
                                out.join(format!("poptime_mv{t}_rank{rank}.json")),
                                serde_json::to_string(&v)?,
                            )?;
                            if rank == 0 {
                                println!(
                                    "MVPRICE {t} rows ({vw} a sequence): eager kernel sum {:.3} ms ({} launches)",
                                    ops.iter().map(|o| f64::from(o.2)).sum::<f64>(),
                                    ops.len()
                                );
                            }
                        }
                        r
                    } else {
                        let b = live.len() as u64;
                        if let std::collections::btree_map::Entry::Vacant(e) = vgs.entry(b) {
                            let vg = capture_mverify(c, fw, &cache, emb_m, b, vw, None)?;
                            if rank == 0 {
                                println!(
                                    "INK20 mverify graph {b} x {vw}: {} launches, {} nodes, {} convs logged",
                                    vg.launches, vg.nodes, vg.n
                                );
                            }
                            e.insert(vg);
                        }
                        let vg = &vgs[&b];
                        c.hip.graph_launch(&vg.e, &c.stream)?;
                        (vg.lg, vg.ld)
                    };
                    c.hip.sync_stream(&c.stream)?;
                    tdev = ts.elapsed().as_secs_f64() * 1e3;
                    if rank == 0 {
                        let all_rows = c.read(lg, (t * ld * 4) as usize)?;
                        for b in 0..live.len() {
                            for j in 0..vw as usize {
                                let r0 = (b * vw as usize + j) * ld as usize * 4;
                                rows[b].extend_from_slice(&all_rows[r0..r0 + VOCAB * 4]);
                            }
                        }
                    }
                }
                if tdev == 0.0 {
                    tdev = ts.elapsed().as_secs_f64() * 1e3;
                }
                tms.entry(format!("{arm}_b{}", live.len()))
                    .or_default()
                    .push(tdev);
                let keeps_new = if arm == "single" {
                    single_kn
                } else {
                    walk_x(&rows, &feeds)?
                };
                let keeps: Vec<u32> = keeps_new.iter().map(|k| k.0).collect();
                if arm == "single" {
                } else if arm == "multi" {
                    c.commit_m(&cache, &keeps, vw)?;
                } else {
                    let vg = &vgs[&(live.len() as u64)];
                    commit_mverify(c, &cache, vg, vw, &keeps)?;
                }
                c.hip.sync_stream(&c.stream)?;
                for (b, &s) in live.iter().enumerate() {
                    let (k, nt) = &keeps_new[b];
                    if rank == 0 {
                        let shas: Vec<String> = (0..vw as usize)
                            .map(|j| {
                                let hx = sha2::Sha256::digest(
                                    &rows[b][j * VOCAB * 4..(j + 1) * VOCAB * 4],
                                );
                                hx[..8].iter().map(|x| format!("{x:02x}")).collect()
                            })
                            .collect();
                        log[s].push((shas, *k));
                    }
                    got[s].extend_from_slice(nt);
                    pos[s] += u64::from(*k);
                }
            }
            drop(vgs);
            res.insert(arm, (log, got));
        }
        if rank == 0 {
            let (sl, sg) = &res["single"];
            let mut cellrep = serde_json::Map::new();
            for arm in ["multi", "mgraph"] {
                let (ml, mg) = &res[arm];
                let (mut eq, mut n) = (0usize, 0usize);
                let mut first_diff = Value::Null;
                for s in 0..nb {
                    for (r, (a, b)) in sl[s].iter().zip(&ml[s]).enumerate() {
                        for j in 0..a.0.len() {
                            n += 1;
                            if a.0.get(j) == b.0.get(j) {
                                eq += 1;
                            } else if first_diff.is_null() {
                                first_diff = json!({"seq": s, "round": r, "row": j});
                            }
                        }
                    }
                }
                let ids_eq = (0..nb).filter(|&s| sg[s] == mg[s]).count();
                let pass = eq == n && n > 0 && ids_eq == nb;
                ok &= pass;
                println!(
                    "MVERIFY {cell} {arm}: rows sha256-equal {eq}/{n}, committed ids equal {ids_eq}/{nb} {} first diff {first_diff}",
                    if pass { "PASS" } else { "FAIL" }
                );
                cellrep.insert(
                    arm.into(),
                    json!({"rows_equal": eq, "rows": n, "ids_equal": ids_eq, "pass": pass, "first_diff": first_diff}),
                );
            }
            let refeq = (0..nb)
                .filter(|&s| {
                    let m = sg[s].len().min(seqs[s].2.len());
                    sg[s][..m] == seqs[s].2[..m]
                })
                .count();
            let keeps: Vec<u32> = sl.iter().flatten().map(|x| x.1).collect();
            let mean = f64::from(keeps.iter().sum::<u32>()) / keeps.len().max(1) as f64;
            println!(
                "MVERIFY {cell}: single-arm ids == the non-MTP greedy reference {refeq}/{nb} (prefix), keep mean {mean:.2} over {} rounds",
                keeps.len()
            );
            let mut p50 = serde_json::Map::new();
            for (k, v) in &mut tms {
                v.sort_by(f64::total_cmp);
                p50.insert(k.clone(), json!(v[v.len() / 2]));
            }
            println!(
                "MVERIFY {cell} host-clock p50 ms (launch + sync): {}",
                Value::Object(p50.clone())
            );
            cellrep.insert("ref_equal".into(), json!(refeq));
            cellrep.insert("keep_mean".into(), json!(mean));
            cellrep.insert("ms_p50".into(), Value::Object(p50));
            rep.insert(cell.into(), Value::Object(cellrep));
        }
    }
    if rank == 0 {
        std::fs::write(
            out.join("ink20_mverify.json"),
            serde_json::to_string_pretty(&Value::Object(rep))?,
        )?;
        println!(
            "INK39 mv levers {:?}: merged projection GEMMs {} push reduces {}",
            f.ink_p39, c.p39_n[0], c.p39_n[1]
        );
        println!("INK20 MVERIFY {}", if ok { "PASS" } else { "FAIL" });
    }
    Ok(())
}

/// INK20 (`--ink-run mtpm`): batched speculation end to end on the harness. For every `BxW` cell
/// of `--ink-mseq` (W = drafts + 1): B prompts prefilled into slots 0..B-1 (each tapped into its
/// own drafter context), then rounds of: the batched drafter pass (every live sequence's k
/// drafts in ONE pass), the multi-sequence verify graph (taps on), the greedy walk a sequence,
/// the per-sequence commit, the drafter appends. Law: every sequence's committed ids == the
/// non-MTP greedy reference (`--ink-mtp-ref`, the k0 json). Also: the batched drafts vs each
/// slot's own single-sequence round (agreement), and the host-clock phases a round.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn mtpm_gate(
    c: &mut Card<'_, '_>,
    fw: &Forward<'_>,
    pk: &PackReader,
    w: &Weights,
    f: &InkFlags,
    pack: &Pack,
    rf: &Path,
    out: &Path,
    only: &[&str],
    rank: u32,
) -> anyhow::Result<()> {
    let pr = fw.peer.context("mtpm: the two-card process")?;
    let mref = f.ink_mtp_ref.as_ref().context("--ink-mtp-ref <k0 json>")?;
    let refj: Value = serde_json::from_str(&std::fs::read_to_string(mref)?)?;
    let n_tok = f.ink_mtp_tokens as usize;
    let send = |v: &[u32]| -> anyhow::Result<()> {
        pr.rows_tx
            .send(v.iter().flat_map(|x| x.to_le_bytes()).collect())
            .context("peer gone")
    };
    let recv = || -> anyhow::Result<Vec<u32>> {
        let b = pr.rows_rx.recv().context("peer gone")?;
        Ok(b.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    };
    let hcap = 3 * (3 + 8) * D * 4 * 48 * 2 * 6;
    c.hist_spare = Some(draft::Hist {
        buf: c.alloc_zero(hcap)?,
        cap: hcap,
        tab: c.alloc_zero(3 * 8 * 512)?,
        ..Default::default()
    });
    let emb_m = c.alloc_zero(MV_ROWS * D * 4)?;
    let mut opt = draft::DraftOpt::of(f.draft_vocab(), &f.ink_draft_gemv, &f.ink_draft_layers)?
        .with_mm(&f.ink_draft_mm)?;
    opt.slots = 8;
    let mut dr = if rank == 0 {
        Some(draft::Drafter::load(
            c,
            pk,
            &pack.dir.join("draft"),
            &w.head[0],
            f.ink_cache_ctx,
            1024,
            &opt,
        )?)
    } else {
        None
    };
    let mut rep = serde_json::Map::new();
    let mut ok = true;
    for cell in f.ink_mseq.split(',').filter(|s| !s.is_empty()) {
        let (bs, ws) = cell.split_once('x').context("--ink-mseq BxW")?;
        let (nb, vw): (usize, u64) = (bs.parse()?, ws.parse()?);
        let k = vw - 1;
        ensure!(
            (1..=8).contains(&nb)
                && (2..=8).contains(&vw)
                && nb as u64 * vw <= MV_ROWS
                && nb <= only.len(),
            "--ink-mseq {cell}: 1..8 sequences x 2..8 rows (<= 32 rows)"
        );
        let cache = Cache::new(
            c,
            w,
            &fw.sched,
            f.ink_cache_ctx,
            f.ink_local_ring,
            nb as u64,
        )?;
        let mut want: Vec<Vec<u32>> = Vec::new();
        let mut toks: Vec<u32> = Vec::new();
        let mut pos: Vec<u64> = Vec::new();
        for (s, pid) in only[..nb].iter().enumerate() {
            let meta: Value =
                serde_json::from_str(&std::fs::read_to_string(rf.join(pid).join("meta.json"))?)?;
            let p_ids = ids_of(&meta["token_ids"]);
            want.push(ids_of(&refj[*pid]["got"]));
            let fwd = Forward {
                pos0: 0,
                graph: false,
                slot: s as u64,
                batch: false,
                sched: fw.sched.clone(),
                ..*fw
            };
            if let Some(d) = dr.as_ref() {
                c.tapdev = Some((d.taps, d.tcap * D * 4));
            }
            let last = pass(c, &fwd, pk, Some(&cache), &p_ids, p_ids.len() as u64 - 1)?;
            c.tapdev = None;
            let t = if rank == 0 {
                let d = dr.as_mut().context("dr")?;
                d.select(s)?;
                d.reset_ctx();
                d.append(c, p_ids.len() as u64, 0)?;
                let t = argmax(&f32s(&last[last.len() - VOCAB * 4..]));
                send(&[t])?;
                t
            } else {
                recv()?[0]
            };
            toks.push(t);
            pos.push(p_ids.len() as u64);
        }
        let mut got: Vec<Vec<u32>> = toks.iter().map(|&t| vec![t]).collect();
        let mut vgs: BTreeMap<u64, draft::VGraph> = BTreeMap::new();
        let (mut agree, mut ndr, mut rounds, mut ntok) = (0usize, 0usize, 0usize, 0usize);
        let mut tph: Vec<[f64; 5]> = Vec::new();
        let mut censused = false;
        loop {
            let live: Vec<usize> = (0..nb)
                .filter(|&s| got[s].len() < n_tok && !got[s].ends_with(&[200_006]))
                .collect();
            if live.is_empty() {
                break;
            }
            let b = live.len() as u64;
            let t0 = Instant::now();
            // 1. the drafts (rank 0): the batched pass, then each slot's own round (agreement)
            let feeds: Vec<Vec<u32>> = if rank == 0 {
                let d = dr.as_mut().context("dr")?;
                let rows: Vec<(usize, u32)> = live.iter().map(|&s| (s, toks[s])).collect();
                d.issue_multi(c, pk, &rows, k)?;
                let dm = d.drafts_multi(c, b, k)?;
                let t1 = t0.elapsed().as_secs_f64() * 1e3;
                if f.ink_op_timing && b as usize == nb && !censused {
                    // INK21: the batched drafter census (one eager pass with per-launch event
                    // pairs: `poptime_draftm_b<b>_rank0.json`)
                    censused = true;
                    c.hip.sync_stream(&c.stream)?;
                    c.time_begin();
                    d.body_multi(c, b, k)?;
                    let ops = c.time_end()?;
                    let v: Vec<Value> = ops.iter().map(|(s, g, ms)| json!([s, g, ms])).collect();
                    std::fs::write(
                        out.join(format!("poptime_draftm_b{b}_rank0.json")),
                        serde_json::to_string(&v)?,
                    )?;
                    println!(
                        "DPRICEM b {b} k {k}: eager kernel sum {:.3} ms ({} launches, --ink-draft-mm {})",
                        ops.iter().map(|o| f64::from(o.2)).sum::<f64>(),
                        ops.len(),
                        f.ink_draft_mm
                    );
                }
                for (i, &s) in live.iter().enumerate() {
                    d.select(s)?;
                    d.issue(c, pk, toks[s], k)?;
                    let ds = d.drafts(c, k)?;
                    agree += ds.iter().zip(&dm[i]).filter(|(a, b)| a == b).count();
                    ndr += ds.len();
                }
                let fd: Vec<Vec<u32>> = live
                    .iter()
                    .enumerate()
                    .map(|(i, &s)| {
                        let mut v = vec![toks[s]];
                        v.extend_from_slice(&dm[i]);
                        v
                    })
                    .collect();
                send(&fd.concat())?;
                tph.push([t1, 0.0, 0.0, 0.0, 0.0]);
                fd
            } else {
                let v = recv()?;
                tph.push([0.0; 5]);
                v.chunks(vw as usize).map(<[u32]>::to_vec).collect()
            };
            let t2 = Instant::now();
            // 2. the multi verify (taps on rank 0)
            let all: Vec<u32> = feeds.concat();
            let words = mverify_words(
                &live.iter().map(|&s| (s as u64, pos[s])).collect::<Vec<_>>(),
                vw,
            )?;
            c.write(emb_m, &pk.embed_rows(&all)?)?;
            c.write(cache.rseq_m, &words)?;
            if let std::collections::btree_map::Entry::Vacant(e) = vgs.entry(b) {
                let taps = dr.as_ref().map(|d| (d.taps, d.tcap * D * 4));
                let vg = capture_mverify(c, fw, &cache, emb_m, b, vw, taps)?;
                if rank == 0 {
                    println!(
                        "INK20 mtpm verify graph {b} x {vw}: {} launches, {} nodes",
                        vg.launches, vg.nodes
                    );
                }
                e.insert(vg);
            }
            let vg = &vgs[&b];
            c.hip.graph_launch(&vg.e, &c.stream)?;
            c.hip.sync_stream(&c.stream)?;
            let tv = t2.elapsed().as_secs_f64() * 1e3;
            // 3. the walk (rank 0), the keeps to rank 1
            let kn: Vec<(u32, Vec<u32>)> = if rank == 0 {
                let all_rows = c.read(vg.lg, (b * vw * vg.ld * 4) as usize)?;
                let mut v = Vec::new();
                for (i, fd) in feeds.iter().enumerate() {
                    let mut a = 0usize;
                    let mut nt = Vec::new();
                    loop {
                        let r0 = (i * vw as usize + a) * vg.ld as usize * 4;
                        let t = argmax(&f32s(&all_rows[r0..r0 + VOCAB * 4]));
                        nt.push(t);
                        if a + 1 < fd.len() && t == fd[a + 1] && t != 200_006 {
                            a += 1;
                        } else {
                            break;
                        }
                    }
                    v.push(((a + 1) as u32, nt));
                }
                let mut m = Vec::new();
                for (kp, nt) in &v {
                    m.push(*kp);
                    m.push(nt.len() as u32);
                    m.extend_from_slice(nt);
                }
                send(&m)?;
                v
            } else {
                let m = recv()?;
                let mut v = Vec::new();
                let mut i = 0;
                while i < m.len() {
                    let n = m[i + 1] as usize;
                    v.push((m[i], m[i + 2..i + 2 + n].to_vec()));
                    i += 2 + n;
                }
                v
            };
            let t3 = Instant::now();
            let keeps: Vec<u32> = kn.iter().map(|x| x.0).collect();
            commit_mverify(c, &cache, vg, vw, &keeps)?;
            if let Some(d) = dr.as_mut() {
                let rows: Vec<(usize, u64, u64, u64)> = live
                    .iter()
                    .enumerate()
                    .map(|(i, &s)| (s, u64::from(keeps[i]), pos[s], i as u64 * vw))
                    .collect();
                d.append_multi(c, &rows)?;
            }
            c.hip.sync_stream(&c.stream)?;
            let tc = t3.elapsed().as_secs_f64() * 1e3;
            if let Some(l) = tph.last_mut() {
                l[1] = tv;
                l[2] = tc;
                l[3] = t0.elapsed().as_secs_f64() * 1e3;
                l[4] = b as f64;
            }
            for (i, &s) in live.iter().enumerate() {
                let (kp, nt) = &kn[i];
                got[s].extend_from_slice(nt);
                toks[s] = *nt.last().context("walk")?;
                pos[s] += u64::from(*kp);
                ntok += nt.len();
            }
            rounds += 1;
        }
        drop(vgs);
        drop(cache);
        if rank == 0 {
            let mut eq = 0;
            let mut per = Vec::new();
            for s in 0..nb {
                let mut g = got[s].clone();
                g.truncate(n_tok);
                let m = g.len().min(want[s].len());
                let same = (0..m).take_while(|&i| g[i] == want[s][i]).count();
                eq += usize::from(same == m);
                per.push(json!({"tokens": g.len(), "equal_prefix": same, "compared": m}));
            }
            let pass = eq == nb;
            ok &= pass;
            let full: Vec<&[f64; 5]> = tph.iter().filter(|r| r[4] as usize == nb).collect();
            let p50 = |j: usize| -> f64 {
                let mut v: Vec<f64> = full.iter().map(|r| r[j]).collect();
                v.sort_by(f64::total_cmp);
                v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
            };
            let msg = format!(
                "MTPM {cell}: ids == the non-MTP greedy {eq}/{nb} {}; rounds {rounds} tokens {ntok} ({:.3} a sequence-round); batched drafts == each slot's own round {agree}/{ndr}; at {nb} live (n {}) p50 ms: draft {:.2} verify {:.2} commit + append {:.2} round {:.2} -> {:.2} ms a token",
                if pass { "PASS" } else { "FAIL" },
                ntok as f64 / tph.iter().map(|r| r[4]).sum::<f64>().max(1.0),
                full.len(),
                p50(0),
                p50(1),
                p50(2),
                p50(3),
                p50(3) / (ntok as f64 / tph.iter().map(|r| r[4]).sum::<f64>().max(1.0)) / nb as f64
            );
            println!("{msg}");
            rep.insert(cell.into(), json!({"pass": pass, "per": per, "msg": msg}));
        }
    }
    if rank == 0 {
        std::fs::write(
            out.join("ink20_mtpm.json"),
            serde_json::to_string_pretty(&Value::Object(rep))?,
        )?;
        println!("INK20 MTPM {}", if ok { "PASS" } else { "FAIL" });
    }
    drop(dr);
    Ok(())
}

/// INK13: replay a captured verify: the `ids` rows' embeddings + the slot / position words
/// (two blocking writes, ordered behind the stream), then the graph (queued, not synchronised).
#[allow(clippy::too_many_arguments)]
fn replay_verify(
    c: &mut Card<'_, '_>,
    pk: &PackReader,
    cache: &Cache,
    emb: u64,
    vg: &draft::VGraph,
    ids: &[u32],
    pos: u64,
    slot: u64,
) -> anyhow::Result<()> {
    ensure!(
        ids.len() as u64 == vg.t,
        "a {}-row verify on the {}-row graph",
        ids.len(),
        vg.t
    );
    c.write(emb, &pk.embed_rows(ids)?)?;
    let mut w = Vec::with_capacity(64);
    for _ in 0..8 {
        w.extend_from_slice(&i32::try_from(slot)?.to_le_bytes());
    }
    for i in 0..8 {
        w.extend_from_slice(&i32::try_from(pos + i.min(vg.t - 1))?.to_le_bytes());
    }
    c.write(cache.rseq_v, &w)?;
    c.hip.graph_launch(&vg.e, &c.stream)?;
    Ok(())
}

/// INK12 task 4: the price of a verify of T rows of ONE sequence on the two-card process, paired in
/// one window: T 1 = the served b1 decode graph (the fused form), T 2..8 = INK9's chain as a
/// continuation chunk at the same position (a captured graph of it, and eager), every arm
/// interleaved rep by rep; host clock around launch + sync (device-bound forms).
#[allow(clippy::too_many_arguments)]
fn vprice(
    c: &mut Card<'_, '_>,
    fw: &Forward<'_>,
    pk: &PackReader,
    w: &Weights,
    f: &InkFlags,
    rf: &Path,
    only: &[&str],
    rank: u32,
) -> anyhow::Result<()> {
    let pid = only.first().context("--ink-only")?;
    let meta: Value =
        serde_json::from_str(&std::fs::read_to_string(rf.join(pid).join("meta.json"))?)?;
    let p_ids = ids_of(&meta["token_ids"]);
    let n0 = p_ids.len();
    let bats: Vec<u64> = f
        .ink_vprice_batch
        .split(',')
        .filter(|x| !x.is_empty())
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let nseq = if bats.is_empty() { 1 } else { 8 };
    let cache = Cache::new(c, w, &fw.sched, f.ink_cache_ctx, f.ink_local_ring, nseq)?;
    let mut fwd = Forward {
        pos0: 0,
        graph: false,
        sched: fw.sched.clone(),
        ..*fw
    };
    pass(c, &fwd, pk, Some(&cache), &p_ids, n0 as u64 - 1)?;
    // INK19: up to 32 rows (rows 0..8 = the INK12 rows: the prompt's first 8 ids; past 8 the
    // prompt's ids in order, so a wide chunk routes distinct tokens as a batched verify would)
    let emb8 = c.alloc_zero(VPRICE_MAX * D * 4)?;
    let mut eids: Vec<u32> = p_ids[..8.min(n0)].repeat(8)[..8].to_vec();
    eids.extend((8..VPRICE_MAX as usize).map(|i| p_ids[i % n0]));
    c.write(emb8, &pk.embed_rows(&eids)?)?;
    let widths: Vec<u64> = f
        .ink_vprice_widths
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let mut graphs = Vec::new();
    for &t in &widths {
        ensure!(
            (1..=VPRICE_MAX).contains(&t),
            "--ink-vprice-widths: 1..{VPRICE_MAX}"
        );
        fwd.pos0 = n0 as u64;
        fwd.graph = t == 1;
        c.write(cache.rpos_dec, &(n0 as i32).to_le_bytes())?;
        c.hip.sync_stream(&c.stream)?;
        let l0 = c.k.launches;
        c.hip
            .begin_capture(&c.stream, StreamCaptureMode::ThreadLocal)?;
        c.reset(Reg::Pass)?;
        let cap = fwd.run(c, Some(&cache), emb8, t, 0);
        let g = c.hip.end_capture(&c.stream)?;
        cap?;
        let nodes = c.hip.graph_node_count(&g)?;
        let e = c.hip.instantiate(&g)?;
        if rank == 0 {
            println!(
                "VPRICE capture T {t}: {} launches, {nodes} nodes",
                c.k.launches - l0
            );
        }
        graphs.push((g, e));
    }
    fwd.graph = false;
    // INK19: the batched decode graphs (one row a sequence, slots 0..b, every row at position n0;
    // slots past 0 hold no prompt: their attention reads zero pages -- a price, not a row law)
    let mut bgraphs = Vec::new();
    for &b in &bats {
        ensure!(
            [1, 2, 4, 8].contains(&b),
            "--ink-vprice-batch: buckets 1 / 2 / 4 / 8"
        );
        let bf = Forward {
            pos0: 1,
            graph: false,
            slot: 0,
            batch: true,
            sched: fw.sched.clone(),
            ..*fw
        };
        let rs: Vec<u8> = (0..b as i32).flat_map(i32::to_le_bytes).collect();
        let rp: Vec<u8> = (0..b).flat_map(|_| (n0 as i32).to_le_bytes()).collect();
        c.write(cache.rseq_b, &rs)?;
        c.write(cache.rpos_b, &rp)?;
        c.hip.sync_stream(&c.stream)?;
        let l0 = c.k.launches;
        c.hip
            .begin_capture(&c.stream, StreamCaptureMode::ThreadLocal)?;
        c.reset(Reg::Pass)?;
        let cap = bf.run(c, Some(&cache), emb8, b, 0);
        let g = c.hip.end_capture(&c.stream)?;
        cap?;
        let nodes = c.hip.graph_node_count(&g)?;
        let e = c.hip.instantiate(&g)?;
        if rank == 0 {
            println!(
                "VPRICE capture b{b}: {} launches, {nodes} nodes",
                c.k.launches - l0
            );
        }
        bgraphs.push((g, e));
    }
    let mut tb = vec![Vec::new(); bats.len()];
    let reps = f.ink_steps as usize;
    let mut tg = vec![Vec::new(); widths.len()];
    let mut te = vec![Vec::new(); widths.len()];
    // INK30 task 4: the `wt` stamps a graph form (widths, then the batches)
    let mut wts = vec![[0u64; 4]; widths.len() + bats.len()];
    c.fuse_wt_take()?;
    for _ in 0..reps {
        for (i, &t) in widths.iter().enumerate() {
            c.hip.sync_stream(&c.stream)?;
            c.fuse_wt_take()?;
            let ts = Instant::now();
            c.hip.graph_launch(&graphs[i].1, &c.stream)?;
            c.hip.sync_stream(&c.stream)?;
            tg[i].push(ts.elapsed().as_secs_f64() * 1e3);
            let w = c.fuse_wt_take()?;
            for j in 0..3 {
                wts[i][j] += w[j];
            }
            wts[i][3] += 1;
            if f.ink_op_timing && tg[i].len() == 2 {
                // INK17: one eager run of the width with per-launch event pairs, after a replay
                // (the verify census: `poptime_vT<t>_rank<r>.json`, tools/capture/ink15/pcensus.py)
                fwd.pos0 = n0 as u64;
                c.time_begin();
                c.reset(Reg::Pass)?;
                fwd.run(c, Some(&cache), emb8, t, 0)?;
                let ops = c.time_end()?;
                let v: Vec<Value> = ops.iter().map(|(s, g, ms)| json!([s, g, ms])).collect();
                std::fs::write(
                    f.ink_out
                        .as_ref()
                        .context("--ink-out")?
                        .join(format!("poptime_vT{t}_rank{rank}.json")),
                    serde_json::to_string(&v)?,
                )?;
            }
            if t > 1 {
                fwd.pos0 = n0 as u64;
                let ts = Instant::now();
                c.reset(Reg::Pass)?;
                fwd.run(c, Some(&cache), emb8, t, 0)?;
                c.hip.sync_stream(&c.stream)?;
                te[i].push(ts.elapsed().as_secs_f64() * 1e3);
            }
        }
        for (i, (_, e)) in bgraphs.iter().enumerate() {
            c.hip.sync_stream(&c.stream)?;
            c.fuse_wt_take()?;
            let ts = Instant::now();
            c.hip.graph_launch(e, &c.stream)?;
            c.hip.sync_stream(&c.stream)?;
            tb[i].push(ts.elapsed().as_secs_f64() * 1e3);
            let w = c.fuse_wt_take()?;
            for j in 0..3 {
                wts[widths.len() + i][j] += w[j];
            }
            wts[widths.len() + i][3] += 1;
            if f.ink_op_timing && tb[i].len() == 2 {
                // INK19: the batched decode census (`poptime_b<b>_rank<r>.json`)
                let b = bats[i];
                let bf = Forward {
                    pos0: 1,
                    graph: false,
                    slot: 0,
                    batch: true,
                    sched: fw.sched.clone(),
                    ..*fw
                };
                c.time_begin();
                c.reset(Reg::Pass)?;
                bf.run(c, Some(&cache), emb8, b, 0)?;
                let ops = c.time_end()?;
                let v: Vec<Value> = ops.iter().map(|(s, g, ms)| json!([s, g, ms])).collect();
                std::fs::write(
                    f.ink_out
                        .as_ref()
                        .context("--ink-out")?
                        .join(format!("poptime_b{b}_rank{rank}.json")),
                    serde_json::to_string(&v)?,
                )?;
            }
        }
    }
    let p50 = |v: &mut Vec<f64>| {
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    if rank == 0 {
        for (i, &t) in widths.iter().enumerate() {
            let (g, e) = (p50(&mut tg[i]), p50(&mut te[i]));
            let p10 = tg[i][tg[i].len() / 10];
            println!(
                "VPRICE T {t}: graph p50 {g:.2} ms (p10 {p10:.2}) eager p50 {e:.2} ms ({reps} reps, {})",
                if t == 1 {
                    "b1 decode form"
                } else if c.fuse & FUSE_VT != 0 && t <= 8 {
                    "INK17 fused T rows"
                } else {
                    "INK9 chain chunk"
                }
            );
        }
    }
    if rank == 0 {
        for (i, &b) in bats.iter().enumerate() {
            let g = p50(&mut tb[i]);
            println!("VPRICE b{b}: graph p50 {g:.2} ms ({reps} reps, batched decode form)");
        }
    }
    if c.fuse & FUSE_WT != 0 {
        // INK30 task 4: a graph step's summed consumer spin (the exposed wait on the peer) and the
        // consumers' kernel time, both from wall_clock64 (100 MHz = 10 ns a tick)
        for (i, w) in wts.iter().enumerate() {
            let name = if i < widths.len() {
                format!("T {}", widths[i])
            } else {
                format!("b{}", bats[i - widths.len()])
            };
            let n = w[3].max(1) as f64;
            println!(
                "INK30 WT rank{rank} {name}: consumer spin {:.3} ms a step, consumer kernels {:.3} ms a step ({} launch rows a step, {} steps)",
                w[0] as f64 * 1e-5 / n,
                w[1] as f64 * 1e-5 / n,
                w[2] / w[3].max(1),
                w[3]
            );
        }
    }
    drop(bgraphs);
    drop(graphs);
    drop(cache);
    Ok(())
}

/// INK19: rows `--ink-run vprice` prices at most (a b8 x T 3 verify = 24 rows + headroom).
const VPRICE_MAX: u64 = 32;

/// INK9 task 4: batched decode with per-row sequence tables. Up to 8 prompts, each prefilled
/// into its own slot of ONE cache, then decoded in lockstep (the oracle's tokens fed; a sequence
/// whose rows ran out leaves the batch, so the bucket walks 8 -> 4 -> 2 -> 1 with pad rows on
/// the pad slot): eager (`<pid>.<tag>b.tf.f32`) and one captured graph a bucket
/// (`<pid>.<tag>bg.tf.f32`) on a second cache, both laid out as the b1 decode rows (row 0 = the
/// prompt's last row) -- gated BYTE-equal to the b1 rows by `tools/capture/ink9/cmp9.py`.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn batch_gate(
    c: &mut Card<'_, '_>,
    fw: &Forward<'_>,
    pk: &PackReader,
    w: &Weights,
    f: &InkFlags,
    rf: &Path,
    out: &Path,
    only: &[&str],
    rank: u32,
    tag: &str,
) -> anyhow::Result<()> {
    let mut seqs: Vec<(String, Vec<u32>, Vec<u32>)> = Vec::new();
    for pid in only {
        let meta: Value =
            serde_json::from_str(&std::fs::read_to_string(rf.join(pid).join("meta.json"))?)?;
        let p_ids = ids_of(&meta["token_ids"]);
        let gen = ids_of(&meta["generated_ids"]);
        let nrow = npy_rows(&rf.join(pid).join("gen_top_ids.npy"))?.min(gen.len());
        if nrow >= 2 {
            seqs.push(((*pid).to_string(), p_ids, gen[..nrow].to_vec()));
        }
    }
    seqs.truncate(8);
    let nb = seqs.len() as u64;
    ensure!(nb >= 1, "batch gate: no prompt with decode rows");
    let emb_in = c.alloc_zero(8 * D * 4)?;
    let mut rep = serde_json::Map::new();
    let vpad2 = 2 * w.head[0].rows();
    for arm in ["eager", "graph"] {
        let cache = Cache::new(c, w, &fw.sched, f.ink_cache_ctx, f.ink_local_ring, nb)?;
        let mut rows: Vec<Vec<u8>> = vec![Vec::new(); seqs.len()];
        for (s, (_, p_ids, _)) in seqs.iter().enumerate() {
            let fwd = Forward {
                pos0: 0,
                graph: false,
                slot: s as u64,
                batch: false,
                sched: fw.sched.clone(),
                ..*fw
            };
            rows[s] = pass(c, &fwd, pk, Some(&cache), p_ids, p_ids.len() as u64 - 1)?;
        }
        let fwd = Forward {
            pos0: 1,
            graph: false,
            slot: 0,
            batch: true,
            sched: fw.sched.clone(),
            ..*fw
        };
        let mut graphs: Vec<Option<(saber_hip::Graph, saber_hip::GraphExec, u64)>> =
            (0..4).map(|_| None).collect();
        let maxs = seqs.iter().map(|q| q.2.len()).max().unwrap_or(0);
        let mut buckets: BTreeMap<String, u64> = BTreeMap::new();
        let mut ms = Vec::new();
        for i in 0..maxs {
            let live: Vec<usize> = (0..seqs.len()).filter(|&s| i < seqs[s].2.len()).collect();
            let bucket = live.len().next_power_of_two();
            *buckets.entry(format!("b{bucket}")).or_insert(0) += 1;
            let toks: Vec<u32> = live.iter().map(|&s| seqs[s].2[i]).collect();
            let mut emb = pk.embed_rows(&toks)?;
            emb.resize(bucket * D as usize * 4, 0);
            let (mut rs, mut rp) = (Vec::new(), Vec::new());
            for j in 0..bucket {
                let (sl, pos) = live
                    .get(j)
                    .map_or((nb, 0), |&s| (s as u64, (seqs[s].1.len() + i) as u64));
                rs.extend_from_slice(&i32::try_from(sl)?.to_le_bytes());
                rp.extend_from_slice(&i32::try_from(pos)?.to_le_bytes());
            }
            c.write(emb_in, &emb)?;
            c.write(cache.rseq_b, &rs)?;
            c.write(cache.rpos_b, &rp)?;
            let lg = if arm == "eager" {
                let ts = Instant::now();
                c.reset(Reg::Pass)?;
                let (lg, _) = fwd.run(c, Some(&cache), emb_in, bucket as u64, 0)?;
                c.hip.sync_stream(&c.stream)?;
                ms.push((bucket, ts.elapsed().as_secs_f64() * 1e3));
                lg
            } else {
                let bi = bucket.trailing_zeros() as usize;
                if graphs[bi].is_none() {
                    c.hip.sync_stream(&c.stream)?;
                    c.hip
                        .begin_capture(&c.stream, StreamCaptureMode::ThreadLocal)?;
                    c.reset(Reg::Pass)?;
                    let cap = fwd.run(c, Some(&cache), emb_in, bucket as u64, 0);
                    let g = c.hip.end_capture(&c.stream)?;
                    let (lg, _) = cap?;
                    let e = c.hip.instantiate(&g)?;
                    graphs[bi] = Some((g, e, lg));
                }
                let (_, e, lg) = graphs[bi].as_ref().expect("captured");
                let ts = Instant::now();
                c.hip.graph_launch(e, &c.stream)?;
                c.hip.sync_stream(&c.stream)?;
                ms.push((bucket, ts.elapsed().as_secs_f64() * 1e3));
                *lg
            };
            if rank == 0 {
                for (j, &s) in live.iter().enumerate() {
                    rows[s].extend(c.read(lg + j as u64 * vpad2 * 4, VOCAB * 4)?);
                }
            }
        }
        if rank == 0 {
            let suf = if arm == "eager" { "b" } else { "bg" };
            for (s, (pid, _, _)) in seqs.iter().enumerate() {
                std::fs::write(out.join(format!("{pid}.{tag}{suf}.tf.f32")), &rows[s])?;
            }
            let mut by: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
            for (b, m) in ms {
                by.entry(b).or_default().push(m);
            }
            let p50: serde_json::Map<String, Value> = by
                .iter_mut()
                .map(|(b, v)| {
                    v.sort_by(f64::total_cmp);
                    (format!("b{b}"), json!(v[v.len() / 2]))
                })
                .collect();
            let r = json!({"sequences": nb, "steps_by_bucket": buckets, "step_ms_p50": p50});
            println!("BATCH {tag} {arm} {r}");
            rep.insert(arm.to_string(), r);
        }
        drop(graphs);
    }
    if rank == 0 {
        std::fs::write(
            out.join(format!("ink9_batch_{tag}.json")),
            serde_json::to_string_pretty(&Value::Object(rep))?,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ink14_the_trunk_defaults_to_w8_head_with_the_overlay_and_stands_down_without() {
        // INK14: unset = w8-head when <pack>/trunk8 exists (two-card), else bf16 by name; `bf16`
        // is the control; an explicit w8 refuses a missing overlay and a one-card run
        assert_eq!(crate::inkling2::InkFlags::default().ink_trunk, None);
        assert_eq!(crate::inkling2::DEFAULT_INK_TRUNK, "w8-head");
        let scratch = saber_test_scratch::ScratchDir::new("ink14-trunk");
        let pk = || PackReader {
            dir: scratch.path().to_path_buf(),
            ckpt: scratch.path().to_path_buf(),
            man: Value::Null,
            recs: Default::default(),
            shapes: BTreeMap::new(),
            files: std::cell::RefCell::new(BTreeMap::new()),
            t8: None,
        };
        let f = crate::inkling2::InkFlags::default();
        assert!(pk()
            .with_trunk(&f, scratch.path(), true)
            .unwrap()
            .t8
            .is_none());
        assert!(pk()
            .with_trunk(&f, scratch.path(), false)
            .unwrap()
            .t8
            .is_none());
        let w8 = crate::inkling2::InkFlags {
            ink_trunk: Some("w8-head".into()),
            ..Default::default()
        };
        assert!(pk().with_trunk(&w8, scratch.path(), true).is_err());
        assert!(pk().with_trunk(&w8, scratch.path(), false).is_err());
        let bf = crate::inkling2::InkFlags {
            ink_trunk: Some("bf16".into()),
            ..Default::default()
        };
        assert!(pk()
            .with_trunk(&bf, scratch.path(), true)
            .unwrap()
            .t8
            .is_none());
    }

    #[test]
    fn ink31_the_prefill_attention_rt8_is_the_default_v1_the_control_and_rt_names_its_rows() {
        // CONS40: wm at floor 0 is the default; rt8 at 256 the control
        assert_eq!(
            prefill_attn(crate::inkling2::DEFAULT_INK_PREFILL_ATTN).unwrap(),
            64
        );
        assert_eq!(crate::inkling2::DEFAULT_INK_PREFILL_ATTN_FLOOR, 0);
        assert!(crate::inkling2::CONS40_CONTROL
            .contains("--ink-prefill-attn rt8 --ink-prefill-attn-floor 256"));
        assert_eq!(prefill_attn("v1").unwrap(), 0);
        assert_eq!(prefill_attn("rt8").unwrap(), 8);
        assert_eq!(prefill_attn("rt16").unwrap(), 16);
        assert_eq!(prefill_attn_sym("rt8t").unwrap(), (8, "ink_fa_rt8t"));
        assert!(prefill_attn("rt4").is_err());
        assert!(PLUGINS.contains(&"ink_fa"));
        assert_eq!(prefill_attn_sym("wm").unwrap(), (64, "ink_fa35_wm"));
        assert_eq!(prefill_attn_sym("wm32").unwrap(), (32, "ink_fa35_wm32"));
        assert!(PLUGINS.contains(&"ink_fa35"));
    }

    #[test]
    fn ink36_levers_parse_and_default_off() {
        assert_eq!(
            ink36_bits(crate::inkling2::DEFAULT_INK36).unwrap(),
            (P36_HG, OV_XW)
        );
        assert_eq!(ink36_bits("none").unwrap(), (0, OV_XW));
        assert_eq!(ink36_bits("ov").unwrap(), (P36_OV, OV_XW));
        assert_eq!(ink36_bits("hgp").unwrap(), (P36_HG | P36_HGP, OV_XW));
        assert_eq!(ink36_bits("ov,hg,xw256").unwrap(), (P36_OV | P36_HG, 256));
        assert!(ink36_bits("xw0").is_err());
        assert!(ink36_bits("xw2048").is_err());
        assert!(ink36_bits("ovx").is_err());
    }

    #[test]
    fn ink35_levers_parse_and_default_off() {
        assert_eq!(ink35_bits(crate::inkling2::DEFAULT_INK35).unwrap(), P35_MZS);
        assert_eq!(ink35_bits("none").unwrap(), 0);
        assert_eq!(ink35_bits("mzs").unwrap(), P35_MZS);
        assert!(ink35_bits("mzb").is_err());
    }

    #[test]
    fn cons40_levers_parse_default_off() {
        assert_eq!(
            ink40_bits(crate::inkling2::DEFAULT_INK40).unwrap(),
            (P40_FA128 | P40_FA128L | P40_ZD1 | P40_ZD2, P40_FA_MIN)
        );
        assert_eq!(ink40_bits("none").unwrap(), (0, P40_FA_MIN));
        assert_eq!(ink40_bits("fa128").unwrap(), (P40_FA128, P40_FA_MIN));
        assert_eq!(
            ink40_bits("fa128l,fam1024").unwrap(),
            (P40_FA128 | P40_FA128L, 1024)
        );
        assert!(ink40_bits("moe").is_err());
    }

    #[test]
    fn ink37_levers_parse_default_off_and_pick_tiles() {
        assert_eq!(
            ink37_bits(crate::inkling2::DEFAULT_INK37).unwrap(),
            P37_MOE | P37_GLUE
        );
        assert_eq!(ink37_bits("none").unwrap(), 0);
        assert_eq!(ink37_bits("moe,glue").unwrap(), P37_MOE | P37_GLUE);
        assert!(ink37_bits("mzs").is_err());
        assert_eq!(p37_rows(0, 1024), None);
        assert_eq!(p37_rows(P37_MOE, 1024), Some(64));
        assert_eq!(p37_rows(P37_MOE, 256), None);
        assert_eq!(p37_rows(P37_MOE8, 1024), Some(128));
        assert_eq!(p37_rows(P37_MOE8, 512), Some(64));
        assert_eq!(crate::inkling2::DEFAULT_INK_PREFILL_ATTN, "wm");
    }

    #[test]
    fn ink29_levers_parse_and_their_shape_laws() {
        assert_ne!(ink29_bits(crate::inkling2::DEFAULT_INK29).unwrap(), 0);
        assert_eq!(ink29_bits("none").unwrap(), 0);
        assert_eq!(
            ink29_bits("kx,rs,fq,mt").unwrap(),
            P29_KX | P29_RS | P29_FQ | P29_MT
        );
        assert!(ink29_bits("ls,b16").is_err());
        assert!(ink29_bits("i8in,i8o").is_err());
        assert!(ink29_bits("w4").is_err());
        // K slices: 3 at the local shapes (K 2,304: 48 stored blocks), 2 at K 3,072; only the
        // narrow outputs (the twin's law)
        assert_eq!(kx_slices(25, 512, 2304), Some(3));
        assert_eq!(kx_slices(65, 512, 2304), None);
        assert_eq!(kx_slices(256, 192, 2304), Some(3));
        assert_eq!(kx_slices(1024, 12, 2304), Some(3));
        assert_eq!(kx_slices(25, 1536, 2304), None);
        assert_eq!(kx_slices(25, 2304, 1536), None);
        assert_eq!(kx_slices(25, 12, 3072), Some(2));
        assert_eq!(bw_slices(2304), Some(3));
        assert_eq!(bw_slices(1536), Some(2));
        assert_eq!(bw_slices(1000), None);
        assert_eq!(bw_slices(4608), Some(6));
        assert_eq!(bw_slices(3072), None);
        assert_eq!(mt_rows(25), 16);
        assert_eq!(mt_rows(405), 32);
        assert_eq!(mt_rows(1024), 64);
    }

    #[test]
    fn ink26_the_prefill_trunk_iu8_is_the_default_and_f32_is_the_control_with_its_launch_law() {
        assert!(prefill_trunk(crate::inkling2::DEFAULT_INK_PREFILL_TRUNK).unwrap());
        assert!(prefill_trunk("iu8").unwrap());
        assert!(!prefill_trunk("f32").unwrap());
        assert!(prefill_trunk("int8").is_err());
        // one 16-column n-tile x up to 16 m-tiles a workgroup, one m-tile a wave
        assert_eq!(i8_grid(25, 1536), ([96, 1, 1], 64));
        assert_eq!(i8_grid(117, 12), ([1, 1, 1], 256));
        assert_eq!(i8_grid(256, 2304), ([144, 1, 1], 512));
        assert_eq!(i8_grid(405, 512), ([32, 2, 1], 512));
        assert_eq!(i8_grid(1024, 192), ([12, 4, 1], 512));
        for m in 9..=2048u64 {
            let (g, b) = i8_grid(m, 1536);
            assert!(g[1] * u64::from(b / 32) * 16 >= m && b <= 512 && b % 32 == 0);
        }
    }

    #[test]
    fn ink15_the_prefill_gemm_pf_is_the_default_its_tile_fills_the_card_and_base_is_the_control() {
        assert_eq!(prefill_gemm("base").unwrap(), (false, false, false));
        assert_eq!(prefill_gemm("pf").unwrap(), (true, false, false));
        assert_eq!(prefill_gemm("cl").unwrap(), (true, true, false));
        assert_eq!(prefill_gemm("clp").unwrap(), (true, true, true));
        assert!(prefill_gemm("x").is_err());
        assert!(
            prefill_gemm(crate::inkling2::DEFAULT_INK_PREFILL_GEMM)
                .unwrap()
                .0
        );
        // a 26-row chunk at N 1,536 / 512: 16-row tiles (the 64 x 64 parent ran 24 workgroups)
        assert_eq!(pf_tile(26, 1536), (16, 32));
        assert_eq!(pf_tile(26, 512), (16, 32));
        // a 1,024-row chunk keeps the parent's 64 x 64 tile
        assert_eq!(pf_tile(1024, 4352), (64, 64));
        for m in [9u64, 17, 33, 100, 406, 1024] {
            for n in [12u64, 192, 512, 1536, 2304, 4352, 100_512] {
                let (tm, tn) = pf_tile(m, n);
                assert!(tm == 16 || m >= tm * 3 / 4, "{m} x {n}: {tm}");
                assert!([32, 64].contains(&tn));
            }
        }
    }

    #[test]
    fn ink16_the_column_lane_cells_cover_the_short_prompt_band_and_leave_the_rest_to_pf() {
        // the decode GEMV owns m <= 8; past 256 rows the pf tile (unmeasured for cl)
        assert_eq!(cl_pick(8, 1536), None);
        assert_eq!(cl_pick(257, 1536), None);
        assert_eq!(cl_pick(26, 32), Some((1, 1)));
        assert_eq!(cl_pick(26, 512), Some((2, 1)));
        assert_eq!(cl_pick(85, 192), Some((2, 2)));
        assert_eq!(cl_pick(117, 1536), Some((4, 2)));
        for m in 9..=256u64 {
            for n in [12u64, 32, 192, 512, 1536, 2304, 4352, 100_512] {
                let (rm, rn) = cl_pick(m, n).unwrap();
                // a symbol the object carries
                assert!(
                    [(1, 1), (2, 1), (4, 1), (8, 1), (2, 2), (4, 2)].contains(&(rm, rn)),
                    "{m} x {n}"
                );
            }
        }
    }

    #[test]
    fn ink11_the_fuse_flag_defaults_to_all11_and_names_its_parts() {
        // INK11: the Inkling default (the user's authorisation, every gate green); `all` =
        // INK10's set (the control), `off` = INK9's chain
        assert_eq!(
            crate::inkling2::InkFlags::default().ink_fuse,
            "all17,cl8,r1"
        );
        assert_eq!(crate::inkling2::InkFlags::default().ink_sampler, "seam");
        assert_eq!(
            fuse_bits("all11").unwrap(),
            FUSE_ALL10 | FUSE_MDEP | FUSE_QKV
        );
        assert!(fuse_bits("kvc").is_err());
        assert_eq!(fuse_bits("all11,kvc").unwrap() & FUSE_KVC, FUSE_KVC);
        assert_eq!(fuse_bits("off").unwrap(), 0);
        assert_eq!(
            fuse_bits("all").unwrap(),
            FUSE_X | FUSE_ROUTE | FUSE_GEMV | FUSE_MSG
        );
        assert_eq!(fuse_bits("x,route").unwrap(), FUSE_X | FUSE_ROUTE);
        assert!(fuse_bits("everything").is_err());
        // two push sites, the flags / counters inside a site, the route + msg counters after
        const { assert!(FZ_CTR + 1024 <= FZ_SITE && FZ_MSG + 4 * 24 <= FZ_BYTES && FZ_ARRIVE < FZ_MSG) };
    }

    #[test]
    fn ink17_vt_is_an_arm_on_the_fused_form_and_its_counter_has_its_own_word() {
        // INK17: the fused T-row verify is an ARM (`all11` stays the default and the control);
        // it needs the fused form (`x`)
        assert_eq!(fuse_bits("all11,vt").unwrap(), FUSE_ALL11 | FUSE_VT);
        // INK19: `cl8` is an arm, outside every named set
        assert_eq!(fuse_bits("all17,cl8").unwrap(), FUSE_ALL17 | FUSE_CL8);
        assert_eq!(FUSE_ALL17 & FUSE_CL8, 0);
        // INK28: the one-workgroup route and the batched-reduction consumer are arms off the default
        assert_eq!(
            fuse_bits("all17,cl8,r1,xs2").unwrap(),
            FUSE_ALL17 | FUSE_CL8 | FUSE_R1 | FUSE_XS2
        );
        assert_eq!((FUSE_ALL17 | FUSE_CL8) & (FUSE_R1 | FUSE_XS2), 0);
        assert!(fuse_bits("r1").is_err());
        // INK30: the latency twins are an arm off the default, a bit of their own
        assert_eq!(
            fuse_bits("all17,cl8,r1,p30").unwrap(),
            FUSE_ALL17 | FUSE_CL8 | FUSE_R1 | FUSE_P30
        );
        assert_eq!((FUSE_ALL17 | FUSE_CL8 | FUSE_R1 | FUSE_XS2) & FUSE_P30, 0);
        assert!(fuse_bits("xs2").is_err());
        assert_eq!(fuse_bits("all11").unwrap() & FUSE_VT, 0);
        assert_eq!(fuse_bits("all17").unwrap(), FUSE_ALL11 | FUSE_VT | FUSE_RT);
        assert!(fuse_bits("vt").is_err());
        assert!(fuse_bits("route,vt").is_err());
        assert!(PLUGINS.contains(&"ink_vt"));
        assert!(PLUGINS.contains(&"ink_dr2"));
        const { assert!(FZ_MSG + 4 * 24 <= FZ_VT && FZ_VT + 4 <= FZ_RT && FZ_RT + 4 * 8 <= FZ_BYTES) };
        assert_eq!(fuse_bits("all11,rt").unwrap(), FUSE_ALL11 | FUSE_RT);
        assert!(fuse_bits("x,rt").is_err());
        assert_eq!(fuse_bits("all11,g16").unwrap(), FUSE_ALL11 | FUSE_G16);
        assert!(fuse_bits("x,g16").is_err());
    }

    #[test]
    fn ink7_the_schedule_is_48_visits_and_noloop_24() {
        assert_eq!(schedule(false).len(), 48);
        assert_eq!(schedule(true).len(), 24);
        assert_eq!(schedule(false)[6..18], (6..18).collect::<Vec<_>>()[..]);
    }

    #[test]
    fn ink7_the_kernargs_pack_at_the_manifest_offsets() {
        // host-only: the manifests of the frozen objects; every body the executor names is declared
        let pre = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../prebuilt/gfx1201");
        for p in PLUGINS {
            let m = saber_kernels::manifest::Manifest::load(pre.join(format!("{p}.manifest.toml")))
                .unwrap();
            assert!(!m.kernel.is_empty(), "{p}");
        }
        let m =
            saber_kernels::manifest::Manifest::load(pre.join("ink_loop.manifest.toml")).unwrap();
        let d = m
            .kernel
            .iter()
            .find(|k| k.name == "ink_depth_attend")
            .unwrap();
        let ex: Vec<_> = d
            .arg
            .iter()
            .filter(|a| !a.value_kind.starts_with("hidden"))
            .collect();
        assert_eq!(ex[0].size, 160, "8 states + 4 queries + 4 outs + 4 gains");
        let s = m.kernel.iter().find(|k| k.name == "ink_softcap").unwrap();
        let ex: Vec<_> = s
            .arg
            .iter()
            .filter(|a| !a.value_kind.starts_with("hidden"))
            .collect();
        assert_eq!(ex[1].size, 8, "the element count is a long");
    }
}
