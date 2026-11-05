//! `saber gate` — the P0 bit-exact gate, in Rust.
//!
//! This is the replacement for `crates/saber-kernels/hip/harness/run_hsaco.cpp`. It consumes
//! exactly what `the internal reference harness` already writes under `.cache/capture/`, so the
//! reference-capture step — the expensive half, which builds the prod kernel module exactly as
//! `serve-mxfp4.sh:1041` does and runs it — is reused unchanged and the two harnesses are
//! directly comparable.
//!
//! What changed is the second half. The C++ harness re-implemented the launch path: its own JSON
//! reader, its own kernarg packer, its own `hipModuleLaunchKernel` call. This one runs the
//! **engine's** path — [`saber_exec::PluginSet`] loads and validates the plugin, `resolve` picks
//! the kernel, [`saber_exec::Launcher`] packs the kernarg image and assigns the workspaces — so a
//! pass is evidence about the engine rather than about a second transcription of it. It also
//! removes the intermediate `plan.json`: `resolve` is called in-process.
//!
//! It re-derives nothing. No tile choice, no split-K rule, no kernarg offset. If the descriptor is
//! wrong the gate fails, which is the point.
//!
//! GPU rules (CLAUDE.md): prod serves on both cards. Run on HIP device 1 only, under
//! the card's lock (`~/saber/.cache/gpu.lock.<card>`, via `tools/capture/gpu_idle.sh`), with `saber:num_requests_running` at 0, every allocation inside
//! `--budget-mib`, and never touch the power cap.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use saber_exec::{
    ArgBindings, LaunchCounters, LaunchCtx, Launcher, LoadCounters, PluginSet, QkCell, TuneOp,
};
use saber_hip::{DevicePtr, HipThread, Stream, StreamFlags};
use saber_kernels::glue::{self, GlueRules};
use saber_kernels::head::{self, HeadOp, HeadRules, HeadShape};
use saber_kernels::rules::{
    resolve_with_layout, ALayout, ArgValue, LaunchRules, Op, Shape, NO_PARTITIONS,
};
use saber_kernels::saber::{
    self, attn_decode_combine, gdn_decode_pair_recurrent, ArDtype, ArShape, AttnShape,
    GdnPrefillShape, GdnShape, KvDtype, NormShape, SaberRules,
};
use saber_kernels::LaunchDescriptor;

use crate::npy;

/// One case's verdict.
enum Verdict {
    Ok {
        bytes: usize,
    },
    Mismatch {
        bytes: usize,
        bad: usize,
        first: usize,
        buffer: String,
    },
    Refused(String),
    Skipped(String),
}

/// Every non-comment, non-empty case line in a capture directory's `cases.txt`.
///
/// Shared with `saber capture-gate`, which runs the same cases through a captured graph: the two
/// gates must be reading the same list, or "the graph matches eager" is a claim about two
/// different sets of work.
pub(crate) fn read_cases(data: &Path) -> anyhow::Result<Vec<String>> {
    let path = data.join("cases.txt");
    let text = std::fs::read_to_string(&path).map_err(|e| {
        anyhow::anyhow!(
            "{}: {e} — run the internal reference harness step 1 first",
            path.display()
        )
    })?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_owned)
        .collect())
}

/// Resolve one case line to a descriptor, through whichever plugin family owns its op token.
///
/// Two capture scripts write two grammars — `the internal reference harness` for the MXFP4 family and
/// the reference capture script (`the internal reference harness`) for saber — and `saber-kplan` / `saber-coreplan` each read one of
/// them. The op token separates them unambiguously, so the gate reads both and needs no flag: point
/// it at `.cache/capture` or `.cache/capture-saber` and it does the right thing.
///
/// Both arms call the same `resolve` the engine calls. Nothing here re-derives a tile, a split law,
/// a warp count or a kernarg offset.
pub(crate) fn resolve_case(
    line: &str,
    mxfp4: &LaunchRules,
    saber_rules: &SaberRules,
    head_rules: &HeadRules,
    glue_rules: &GlueRules,
) -> anyhow::Result<std::result::Result<(LaunchDescriptor, Shape), String>> {
    let op = line.split_whitespace().next().unwrap_or("");
    if is_head_op(op) {
        // The head plugin's shapes are rows x cols, not (M, N, K), and it declares no workspace,
        // so the GEMM shape that rides along as the Persistent workspace key is empty.
        return Ok(resolve_head(line, head_rules).map(|d| (d, Shape::new(0, 0, 0))));
    }
    if is_glue_op(op) {
        // Like the reference's: the glue shapes are not (M, N, K), and none of these ops declares a
        // Persistent workspace, so the GEMM shape rides along empty.
        return Ok(resolve_glue(line, glue_rules).map(|d| (d, Shape::new(0, 0, 0))));
    }
    if is_saber_op(op) {
        // The reference's shapes are not (M, N, K); the GEMM shape only rides along as the Persistent
        // workspace key, and the reference's ops declare no Persistent workspace.
        return Ok(resolve_saber(line, saber_rules).map(|d| (d, Shape::new(0, 0, 0))));
    }
    // A case this harness has no arm for is a SKIP with a reason, not a crash: the capture's
    // grammar grew `gemm_part`, `gemm_wp0` and `gemm_indirect` after this gate was written (both
    // need something beyond a shape -- partition boundaries, a per-process layout knob, a named
    // instantiation), and the C++ replay still covers them. Erroring out here instead would mean
    // the `--autotune` arm could not reach the cases after them.
    let (op, shape) = match parse_mxfp4_case(line) {
        Ok(pair) => pair,
        Err(e) => {
            return Ok(Err(format!(
                "{e} (no arm in the Rust gate; the C++ replay covers it)"
            )))
        }
    };
    let layout = if line.starts_with("gemm_rm ") {
        ALayout::RowMajor
    } else {
        ALayout::FollowsM
    };
    Ok(resolve_with_layout(op, shape, mxfp4, layout)
        .map(|d| (d, shape))
        .map_err(|e| e.to_string()))
}

/// `the internal reference harness`'s grammar, field for field with `saber-glueplan`'s.
///
/// The fourth plugin's three ops. Its reference is the reference server itself (ATen + Triton), so the capture
/// runs in prod's image and this side only replays -- but until now `saber gate` had no arm for
/// the grammar at all, which meant the 18 glue routes the plan BINDS could not be byte-compared
/// against what the cache chose. `--autotune`'s coverage line read `qk_norm_rope 0/6`.
fn resolve_glue(line: &str, rules: &GlueRules) -> std::result::Result<LaunchDescriptor, String> {
    let f: Vec<&str> = line.split_whitespace().collect();
    let n = |i: usize| -> u32 { f.get(i).and_then(|v| v.parse().ok()).unwrap_or(0) };
    let l = |i: usize| -> i64 { f.get(i).and_then(|v| v.parse().ok()).unwrap_or(0) };
    let (op, shape) =
        glue_op_shape(&f, n, l).ok_or_else(|| format!("unknown glue op in `{line}`"))?;
    glue::resolve(op, shape, rules).map_err(|e| e.to_string())
}

/// The op and shape one glue case line names. Shared by [`resolve_glue`] and the `--autotune`
/// override, so the field indices are read once.
fn glue_op_shape(
    f: &[&str],
    n: impl Fn(usize) -> u32,
    l: impl Fn(usize) -> i64,
) -> Option<(glue::Op, glue::Shape)> {
    Some(match *f.first()? {
        // The trailing position mode is a data label, not a rules input: the planner stops at
        // field 6, and so does this.
        "qk_rope" => (
            glue::Op::QkNormRope {
                interleaved: n(6) != 0,
                // See `saber-glueplan`: a `qk_rope` case line is the target's flavour.
                flavour: glue::QkFlavour::TARGET,
            },
            glue::Shape::Qk(glue::QkShape {
                m: n(1),
                nq: n(2),
                nkv: n(3),
                head_dim: n(4),
                qkv_n: n(5),
            }),
        ),
        "gate_quant" => (
            glue::Op::AttnGateQuant,
            glue::Shape::Gate(glue::GateShape {
                m: n(1),
                n: n(2),
                head_dim: n(3),
                gs: l(4),
            }),
        ),
        "kv_append" => (
            glue::Op::KvAppend,
            glue::Shape::Kv(glue::KvShape {
                tokens: n(1),
                nkv: n(2),
                head_dim: n(3),
                block_size: n(4),
                key_stride: l(5),
                value_stride: l(6),
                block_stride: l(7),
                page_stride: l(8),
                head_stride: l(9),
            }),
        ),
        _ => return None,
    })
}

fn is_glue_op(op: &str) -> bool {
    matches!(op, "qk_rope" | "gate_quant" | "kv_append")
}

fn is_head_op(op: &str) -> bool {
    matches!(op, "embed" | "row_gather" | "argmax")
}

/// `tools/capture/capture_head.py`'s grammar.
///
/// Three lines, all positional, all in elements (never bytes), because that is what
/// `HeadShape` is in:
///
/// ```text
/// embed      <rows> <hidden> <vocab_begin> <vocab_end>
/// row_gather <rows> <cols>   <src_stride>
/// argmax     <rows> <cols>   <stride>
/// ```
///
/// The embedding takes no stride argument at all — its table rows are the hidden size by
/// construction — so the shape's `src_stride` is set to the width it reads, which is what
/// `head::resolve`'s "a stride narrower than the row it carries" check wants to see.
fn resolve_head(line: &str, rules: &HeadRules) -> std::result::Result<LaunchDescriptor, String> {
    let f: Vec<&str> = line.split_whitespace().collect();
    let n = |i: usize| -> u32 { f.get(i).and_then(|v| v.parse().ok()).unwrap_or(0) };
    let l = |i: usize| -> i64 { f.get(i).and_then(|v| v.parse().ok()).unwrap_or(0) };
    let (op, shape) = match f[0] {
        "embed" => (
            HeadOp::Embedding,
            HeadShape {
                rows: n(1),
                cols: n(2),
                src_stride: i64::from(n(2)),
                vocab: (n(3), n(4)),
            },
        ),
        "row_gather" => (
            HeadOp::RowGather,
            HeadShape {
                rows: n(1),
                cols: n(2),
                src_stride: l(3),
                vocab: (0, 0),
            },
        ),
        "argmax" => (
            HeadOp::Argmax,
            HeadShape {
                rows: n(1),
                cols: n(2),
                src_stride: l(3),
                vocab: (0, 0),
            },
        ),
        other => return Err(format!("unknown head op `{other}`")),
    };
    head::resolve(op, shape, rules).map_err(|e| e.to_string())
}

fn is_saber_op(op: &str) -> bool {
    matches!(
        op,
        "attn_prefill"
            | "attn_decode"
            | "attn_combine"
            | "gdn_fused"
            | "gdn_conv_upd"
            | "gdn_recurrent"
            | "gdn_prep"
            | "gdn_kkt"
            | "gdn_scan"
            | "gdn_norm"
            | "ar_wht"
    )
}

/// A saber or attn_glue case line, read into the `TuneOp` the sweeper measures.
///
/// The two sub-op grammars (`attn_combine`, `gdn_recurrent`) name their ROUTE's op and dispatch 1,
/// because they are halves of a route and the autotune record names the route. The field indices
/// are the ones `resolve_saber` and `glue_op_shape` use, and `the_case_grammar_agrees_with_the_tune_op`
/// holds the two against each other so an index that moves in one and not the other fails a test.
fn saber_tune_op(line: &str) -> Option<(TuneOp, usize)> {
    let f: Vec<&str> = line.split_whitespace().collect();
    let n = |i: usize| -> u32 { f.get(i).and_then(|v| v.parse().ok()).unwrap_or(0) };
    let l = |i: usize| -> i64 { f.get(i).and_then(|v| v.parse().ok()).unwrap_or(0) };
    let b = |i: usize| -> u32 {
        f.get(i)
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.0)
            .to_bits()
    };
    let kv_at = |i: usize| {
        if *f.get(i).unwrap_or(&"fp8") == "bf16" {
            KvDtype::Bf16
        } else {
            KvDtype::Fp8E4m3
        }
    };
    let attn = |kv: KvDtype| AttnShape {
        num_seqs: n(1),
        q_len: n(2),
        live_rows: n(2),
        q_heads: n(3),
        kv_heads: n(4),
        head_dim: 256,
        block_size: 16,
        max_blocks: n(5).div_ceil(16),
        max_ctx: n(5),
        splits: 0,
        kv_dtype: kv,
        kv_block_stride: l(6),
        kv_head_stride: l(7),
        scale_bits: b(8),
        has_descales: false,
        ragged_rows: 0,
    };
    let gdn = || GdnShape {
        n_seqs: n(1),
        h: n(2),
        hg: n(3),
        head_k: 128,
        head_v: 128,
        conv_width: 4,
        maxq: n(4),
        o_rows: n(5),
        xpitch: l(6),
        cs_seq: l(7),
        cs_dim: l(8),
        cs_tok: l(9),
        ci_stride: l(10),
        ab_stride: l(11),
        st_slot: l(12),
        st_head: l(13),
        sidx_stride: l(14),
        scale_bits: b(15),
        softplus_thr_bits: b(16),
        has_naccept: true,
        has_bias: true,
        state_f16: false,
    };
    let prefill = || GdnPrefillShape {
        n_seqs: n(1),
        t: n(2),
        h: n(3),
        hg: n(4),
        head_k: 128,
        head_v: 128,
        conv_width: 4,
        chunk: 64,
        xpitch: l(5),
        cs_seq: l(6),
        cs_dim: l(7),
        cs_tok: l(8),
        ci_stride: l(9),
        ab_stride: l(10),
        scale_bits: b(11),
        softplus_thr_bits: b(12),
        has_bias: true,
        has_init: true,
        state_f16: false,
    };
    // The second element is WHICH dispatch of the route this case is. The capture ran each kernel
    // on its own, so the attention decode's split-KV combine and the GDN pair's recurrent update
    // are cases of their own -- and they are the halves whose knobs (`splits`, `vs`) the record
    // also chose, so skipping them would leave half of every two-dispatch binding ungated.
    // The glue trio, from the same parse `resolve_glue` uses. Its shapes carry the strides that
    // describe the caller's buffers, and `tuning_signature` drops them -- so a case and the route
    // it exercises agree on the geometry and disagree on the pitches, which is the point.
    if is_glue_op(f.first().copied().unwrap_or("")) {
        let (gop, shape) = glue_op_shape(&f, n, l)?;
        // `enumerate_saber` offers `qk_norm_rope` at the CONFIG's own `mrope_interleaved`, which
        // for this checkpoint is `true` (`graph.rs`) -- so the chunked symbol is not an
        // instantiation any record can name, and asking the cache for it would report a bit
        // mismatch that is really a grammar mismatch. The chunked cases are gated against the
        // RULES, which read the flag; the interleaved case of the same geometry is what covers
        // the signature, because `tuning_signature` drops the flag.
        if matches!(
            gop,
            glue::Op::QkNormRope {
                interleaved: false,
                ..
            }
        ) {
            return None;
        }
        return Some((
            match shape {
                // A capture case is a call with NO history: `capture_glue.py` ran the
                // kernel standalone over positions `0..m`, and its grammar carries no context
                // band. `QkCell::fresh_chunk` says exactly that, and the band is not in
                // `tuning_signature`, so this still matches the bound route it exercises.
                glue::Shape::Qk(q) => TuneOp::GlueQkNormRope(QkCell::fresh_chunk(q)),
                glue::Shape::Gate(g) => TuneOp::GlueGateQuant(g),
                glue::Shape::Kv(k) => TuneOp::GlueKvAppend(k),
            },
            0,
        ));
    }
    Some(match *f.first()? {
        "attn_prefill" => (TuneOp::AttnPrefill(attn(kv_at(10))), 0),
        "attn_decode" => (TuneOp::AttnDecode(attn(kv_at(9))), 0),
        "attn_combine" => (TuneOp::AttnDecode(attn(kv_at(9))), 1),
        "gdn_fused" => (TuneOp::GdnFused(gdn()), 0),
        "gdn_conv_upd" => (TuneOp::GdnPair(gdn()), 0),
        "gdn_recurrent" => (TuneOp::GdnPair(gdn()), 1),
        "gdn_prep" => (TuneOp::GdnConvPrep(prefill()), 0),
        "gdn_kkt" => (TuneOp::GdnKktSolve(prefill()), 0),
        "gdn_scan" => (TuneOp::GdnChunkScan(prefill()), 0),
        // `act` is the layer's `output_gate_type`, not a variant: the sweeper offers act=0 only
        // (swish, this checkpoint's) and the key carries no act, so a case at any other act is a
        // kernel the plan never binds and must fall through to the rules.
        "gdn_norm" if n(6) == 0 => (
            TuneOp::GdnNorm(NormShape {
                rows: l(1) as u64,
                channels: 128,
                xrow: l(2),
                zrow: l(3),
                orow: l(4),
                eps_bits: b(5),
            }),
            0,
        ),
        _ => return None,
    })
}

/// the reference capture script (`the internal reference harness`)'s grammar, field for field with `saber-coreplan`'s.
fn resolve_saber(line: &str, rules: &SaberRules) -> std::result::Result<LaunchDescriptor, String> {
    let f: Vec<&str> = line.split_whitespace().collect();
    let n = |i: usize| -> u32 { f.get(i).and_then(|v| v.parse().ok()).unwrap_or(0) };
    let l = |i: usize| -> i64 { f.get(i).and_then(|v| v.parse().ok()).unwrap_or(0) };
    let b = |i: usize| -> u32 {
        f.get(i)
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.0)
            .to_bits()
    };
    let attn = |kv: &str| AttnShape {
        num_seqs: n(1),
        q_len: n(2),
        live_rows: n(2),
        q_heads: n(3),
        kv_heads: n(4),
        head_dim: 256,
        block_size: 16,
        max_blocks: n(5).div_ceil(16),
        max_ctx: n(5),
        splits: 0,
        kv_dtype: if kv == "bf16" {
            KvDtype::Bf16
        } else {
            KvDtype::Fp8E4m3
        },
        kv_block_stride: l(6),
        kv_head_stride: l(7),
        scale_bits: b(8),
        // The capture generates a cache whose per-tensor scales are 1.0, which is what prod
        // serves: `_descale_ptrs` then passes 0 and the kernels read "no descale".
        has_descales: false,
        ragged_rows: 0,
    };
    let gdn = || GdnShape {
        n_seqs: n(1),
        h: n(2),
        hg: n(3),
        head_k: 128,
        head_v: 128,
        conv_width: 4,
        maxq: n(4),
        o_rows: n(5),
        xpitch: l(6),
        cs_seq: l(7),
        cs_dim: l(8),
        cs_tok: l(9),
        ci_stride: l(10),
        ab_stride: l(11),
        st_slot: l(12),
        st_head: l(13),
        sidx_stride: l(14),
        scale_bits: b(15),
        softplus_thr_bits: b(16),
        has_naccept: true,
        has_bias: true,
        state_f16: false,
    };
    let prefill = || GdnPrefillShape {
        n_seqs: n(1),
        t: n(2),
        h: n(3),
        hg: n(4),
        head_k: 128,
        head_v: 128,
        conv_width: 4,
        chunk: 64,
        xpitch: l(5),
        cs_seq: l(6),
        cs_dim: l(7),
        cs_tok: l(8),
        ci_stride: l(9),
        ab_stride: l(10),
        scale_bits: b(11),
        softplus_thr_bits: b(12),
        has_bias: true,
        has_init: true,
        state_f16: false,
    };
    let kv_at = |i: usize| *f.get(i).unwrap_or(&"fp8");
    let r = match f[0] {
        "attn_prefill" => saber::resolve(
            saber::Op::AttnPrefill { fp8_mode: n(9) },
            saber::Shape::Attn(attn(kv_at(10))),
            rules,
        ),
        "attn_decode" => saber::resolve(
            saber::Op::AttnDecode,
            saber::Shape::Attn(attn(kv_at(9))),
            rules,
        ),
        "attn_combine" => attn_decode_combine(attn(kv_at(9)), rules),
        "gdn_fused" => saber::resolve(
            saber::Op::GdnFusedUpdate {
                ab_bf16: n(17) == 1,
            },
            saber::Shape::Gdn(gdn()),
            rules,
        ),
        "gdn_conv_upd" => saber::resolve(
            saber::Op::GdnDecodePair {
                ab_bf16: n(17) == 1,
            },
            saber::Shape::Gdn(gdn()),
            rules,
        ),
        // Prod passes no z gate and no norm weight: the gated RMS norm is NOT folded into the
        // epilogue on this unit.
        "gdn_recurrent" => gdn_decode_pair_recurrent(gdn(), n(17) == 1, None, rules),
        "gdn_prep" => saber::resolve(
            saber::Op::GdnConvPrep {
                ab_bf16: n(13) == 1,
            },
            saber::Shape::GdnPrefill(prefill()),
            rules,
        ),
        "gdn_kkt" => saber::resolve(
            saber::Op::GdnKktSolve,
            saber::Shape::GdnPrefill(prefill()),
            rules,
        ),
        "gdn_scan" => saber::resolve(
            saber::Op::GdnChunkScan,
            saber::Shape::GdnPrefill(prefill()),
            rules,
        ),
        "gdn_norm" => saber::resolve(
            saber::Op::GdnGatedRmsNorm { act: n(6) },
            saber::Shape::Norm(NormShape {
                rows: l(1) as u64,
                channels: 128,
                xrow: l(2),
                zrow: l(3),
                orow: l(4),
                eps_bits: b(5),
            }),
            rules,
        ),
        "ar_wht" => saber::resolve(
            saber::Op::AllReduceWht {
                bits: n(3),
                dtype: if f.get(4) == Some(&"fp16") {
                    ArDtype::Fp16
                } else {
                    ArDtype::Bf16
                },
            },
            saber::Shape::Ar(ArShape {
                n_elem: l(1) as u64,
                elem_bytes: l(2) as u64,
                world_size: 2,
            }),
            rules,
        ),
        other => return Err(format!("unknown saber op `{other}`")),
    };
    r.map_err(|e| e.to_string())
}

/// Parse one line of the MXFP4 `cases.txt` into the op and shape `resolve` takes.
///
/// The grammar is `the internal reference harness`'s, and `crates/saber-kernels/src/bin/saber-kplan.rs`
/// reads the same lines. Keeping it identical is what lets `gate.sh`'s step 1 be reused verbatim.
fn parse_mxfp4_case(line: &str) -> anyhow::Result<(Op, Shape)> {
    let f: Vec<&str> = line.split_whitespace().collect();
    let num = |i: usize| -> anyhow::Result<u32> {
        f.get(i)
            .ok_or_else(|| anyhow::anyhow!("`{line}`: missing field {i}"))?
            .parse()
            .map_err(|_| anyhow::anyhow!("`{line}`: field {i} is not an integer"))
    };
    let float = |i: usize| -> anyhow::Result<f32> {
        f.get(i)
            .ok_or_else(|| anyhow::anyhow!("`{line}`: missing field {i}"))?
            .parse()
            .map_err(|_| anyhow::anyhow!("`{line}`: field {i} is not a float"))
    };
    Ok(match f.first().copied() {
        // `gemm_rm` is `gemm` on an edge whose A stays row-major at any M (`o_proj`, whose A is
        // the attention output). The layout is not in the shape, so it is applied at the resolve
        // call below -- see `resolve_case`.
        Some("gemm" | "gemm_rm") => (Op::gemm(), Shape::new(num(1)?, num(2)?, num(3)?)),
        Some("gemm_nw") => (
            Op::Gemm {
                has_wref: false,
                partitions: NO_PARTITIONS,
            },
            Shape::new(num(1)?, num(2)?, num(3)?),
        ),
        Some("add_rms") => (
            Op::AddRmsQuant { eps: float(3)? },
            Shape::new(num(1)?, 0, num(2)?),
        ),
        Some("silu") => (Op::SiluMulQuant, Shape::new(num(1)?, num(2)?, 0)),
        Some("gdn") => (
            Op::GdnNormQuant {
                zs: i64::from(num(3)?),
                eps: float(4)?,
            },
            Shape::new(num(1)?, num(2)?, 0),
        ),
        other => anyhow::bail!("unknown op `{}` in `{line}`", other.unwrap_or("")),
    })
}

/// `shared.txt`: buffers a case shares with others (weights are per shape, not per M).
pub(crate) fn read_shared(data: &Path) -> anyhow::Result<BTreeMap<(String, String), String>> {
    let path = data.join("shared.txt");
    let mut out = BTreeMap::new();
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Ok(out);
    };
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() == 3 {
            out.insert((f[0].to_string(), f[1].to_string()), f[2].to_string());
        }
    }
    Ok(out)
}

/// Arguments of `saber gate`.
pub struct GateArgs {
    /// Where a capture script wrote its `.npy` files and `cases.txt`. Repeatable: the four
    /// grammars live in four directories (`capture`, `capture-saber`, `capture-glue`, `capture-head`)
    /// and the `--autotune` coverage line counts every non-GEMM route the plan binds, so a run
    /// given one directory can only ever reach a fraction of it. Each case is read against the
    /// directory it came from.
    pub data: Vec<PathBuf>,
    /// The committed code objects and their manifests.
    pub prebuilt: PathBuf,
    /// Only run cases whose task line contains this substring.
    pub only: Option<String>,
    /// Refuse a case whose device buffers exceed this. Prod owns the card.
    pub budget_mib: u64,
    /// HIP device ordinal within the visible set.
    pub device: i32,
    /// An autotune cache. With it the gate byte-compares the variant the CACHE binds for a shape
    /// rather than the one `resolve` picks — see [`autotune_overrides`].
    pub autotune: Option<PathBuf>,
    /// The manifest the cache is read against. Required with `--autotune`.
    pub manifest: Option<PathBuf>,
    /// With `--autotune`: write the capture case line for every bound GEMM shape here and stop.
    ///
    /// This is what closes the loop between the plan and the capture. The capture's own grid is
    /// the band ladder's M (1, 8, 24, 40, 64, 128, 2048); the plan binds at the BUCKETS' M
    /// (5, 10, 20, 40 decode, 2048/8192 prefill) and at five widths the grid has no weights for,
    /// so 34 of 41 bound variants had never been byte-compared at their own shape. Rather than
    /// transcribe the list into `tools/capture/shapes.py` and let it drift, the gate emits it.
    pub emit_cases: Option<PathBuf>,
}

/// The variant the autotune cache binds for each GEMM shape, ready to launch.
///
/// The rules and the cache can disagree — that is the autotune plane's whole purpose — and until
/// 2026-09-20 the gate only ever replayed the RULES' answer. So a measurement could promote a
/// variant to `measured` and put it in `explain` without one byte of it ever having been compared
/// against the reference AT THAT SHAPE. (The symbol is gated somewhere; prod's K%128 incident is
/// the standing proof that a symbol can be right at one shape and wrong at another.)
///
/// This is the other half of the pair: `--autotune` gates what the plan binds. Today the executor
/// still resolves from the rules, so the two are the same launch except where a measurement
/// overrode the policy; when the executor starts taking its descriptor from the bound plan, this
/// arm is already covering it.
struct AutotuneOverrides {
    /// (m, n, k) -> the descriptors the cache's variant launches.
    by_shape: BTreeMap<(u32, u32, u32), Vec<LaunchDescriptor>>,
    /// (m, n, k) -> the route names that bind it, for the coverage report.
    routes: BTreeMap<(u32, u32, u32), Vec<String>>,
    /// Shapes a case actually exercised.
    used: std::collections::BTreeSet<(u32, u32, u32)>,
    /// The saber half: `TuneOp::tuning_signature()` -> the VARIANT ID the cache binds, and the
    /// routes that bind it.
    ///
    /// Keyed on the signature rather than on a shape tuple because a saber geometry is not
    /// (m, n, k), and carrying the id rather than a descriptor because the kernarg holds this
    /// CASE's strides: the descriptor is rebuilt by enumerating at the case's own shape and
    /// picking the instantiation the record names. A variant id is an FNV of the SYMBOL, so it
    /// means the same thing at both shapes.
    saber_by_sig: BTreeMap<String, (u32, Vec<String>)>,
    /// Signatures a case actually exercised.
    saber_used: std::collections::BTreeSet<String>,
}

/// Build the override table: bind the plan, then rebuild each bound variant's descriptor by
/// finding it in the sweep's own enumeration, which is the only place a variant id means anything.
fn autotune_overrides(
    cache_path: &Path,
    manifest_path: &Path,
    prebuilt: &Path,
) -> anyhow::Result<AutotuneOverrides> {
    use saber_exec::tune::{enumerate_variants, SymbolIndex, TuneOp};
    use saber_plan::{registry_from_prebuilt, AutotuneCache, DeviceProfile, Manifest, ModelSpec};

    let manifest = Manifest::from_path(manifest_path)?;
    let registry = registry_from_prebuilt(prebuilt)?;
    let device = DeviceProfile::gfx1201_reference();
    let model = ModelSpec::qwen38_27b_dense(manifest.device.tp)?;
    let resolved = saber_plan::resolve(&manifest, &device, &model, &registry)?;
    let cache = AutotuneCache::load(cache_path)?;
    let missing = resolved.missing(&cache);
    if !missing.is_empty() {
        anyhow::bail!(
            "{} of {} routes have no record in {}: a gate of the cache's variants needs a cache that covers the plan",
            missing.len(),
            resolved.requirements().len(),
            cache_path.display()
        );
    }
    let plan = resolved.bind(&cache)?;
    let rules = LaunchRules::prod_mxfp4()?;
    let saber_rules = SaberRules::prod_saber()?;
    let symbols = SymbolIndex::from_manifest_dir(prebuilt)?;

    let mut out = AutotuneOverrides {
        by_shape: BTreeMap::new(),
        routes: BTreeMap::new(),
        used: std::collections::BTreeSet::new(),
        saber_by_sig: BTreeMap::new(),
        saber_used: std::collections::BTreeSet::new(),
    };
    for bucket in &plan.body.buckets {
        for op in bucket.ops.ops() {
            if op.op != "gemm_w4a8" {
                // The saber half: the same question, asked of a geometry that is not (m, n, k).
                // A route the sweep cannot launch (the two-rank all-reduce) has no TuneOp and is
                // skipped here exactly as it is skipped there.
                if let Ok(crate::tune::Work::Sweep(tune_op)) =
                    crate::tune::work_for_keys(&op.op, &op.shape)
                {
                    let e = out
                        .saber_by_sig
                        .entry(tune_op.tuning_signature())
                        .or_insert_with(|| (op.variant.variant_id, Vec::new()));
                    e.1.push(format!("{} {}", bucket.bucket.label(), op.op_name));
                }
                continue;
            }
            let dim = |k: &str| u32::try_from(op.shape.num(k).unwrap_or(0)).unwrap_or(0);
            let key = (dim("m"), dim("n"), dim("k"));
            out.routes.entry(key).or_default().push(format!(
                "{} {}",
                bucket.bucket.label(),
                op.op_name
            ));
            if out.by_shape.contains_key(&key) {
                continue;
            }
            let shape = Shape::new(key.0, key.1, key.2);
            let tune_op = TuneOp::Mxfp4 {
                op: Op::gemm(),
                shape,
            };
            let set = enumerate_variants(&tune_op, &rules, &saber_rules, &symbols);
            let Some(v) = set
                .variants
                .iter()
                .find(|v| v.variant_id() == op.variant.variant_id)
            else {
                anyhow::bail!(
                    "{}: the cache binds variant {} at M={} N={} K={}, and the rules cannot enumerate it — the record outlived the policy that measured it",
                    op.op_name,
                    op.variant.variant_id,
                    key.0,
                    key.1,
                    key.2
                );
            };
            out.by_shape.insert(key, v.descs.clone());
        }
    }
    Ok(out)
}

/// Run the gate. Returns the number of mismatching cases.
pub fn run(hip: &HipThread, args: &GateArgs) -> anyhow::Result<usize> {
    hip.set_device(args.device)?;
    let rules = LaunchRules::prod_mxfp4()?;
    let saber_rules = SaberRules::prod_saber()?;
    let head_rules = HeadRules::prod_head()?;
    let glue_rules = GlueRules::prod()?;
    let load_counters = LoadCounters::new();
    let plugins = PluginSet::load_dir(hip, &args.prebuilt, &load_counters)?;
    let functions = plugins.functions(hip)?;
    let ctx = LaunchCtx::new(hip, &functions);
    println!(
        "plugins {} loaded from {} ({} symbols); {}",
        plugins.plugins().len(),
        args.prebuilt.display(),
        functions.len(),
        load_counters.snapshot().line()
    );
    for p in plugins.plugins() {
        println!(
            " {:<20} {:>6} kernels  wave{}  lds<={}  kernarg<={}",
            p.facts.plugin,
            p.facts.kernels,
            p.facts.wave_size,
            p.facts.max_lds_bytes,
            p.facts.max_kernarg_bytes
        );
    }

    // The symbol index the override's enumeration needs: which instantiations the committed code
    // objects actually ship. Built once, from the same directory the plugins came from.
    let symbols = saber_exec::SymbolIndex::from_manifest_dir(&args.prebuilt)?;
    let mut overrides = match (&args.autotune, &args.manifest) {
        (Some(cache), Some(manifest)) => {
            let o = autotune_overrides(cache, manifest, &args.prebuilt)?;
            if let Some(path) = &args.emit_cases {
                // Sorted by (n, k, m) so consecutive lines share a weight tensor: the capture
                // generates one per (n, k) and walks the M values under it.
                let mut keys: Vec<&(u32, u32, u32)> = o.by_shape.keys().collect();
                keys.sort_by_key(|k| (k.1, k.2, k.0));
                let mut text = String::from(
                    "# GENERATED by `saber gate --autotune --emit-cases`: one line per GEMM shape
                     # the autotune cache BINDS, so the gate can byte-compare the variant the plan
                     # promotes rather than the one the rules resolve.
",
                );
                for k in &keys {
                    text.push_str(&format!(
                        "gemm {} {} {}
",
                        k.0, k.1, k.2
                    ));
                }
                std::fs::write(path, text)?;
                println!(
                    "emitted {} bound GEMM shape(s) to {}",
                    keys.len(),
                    path.display()
                );
                return Ok(0);
            }
            println!(
                "autotune {} shape(s) bound by {} — the gate replays the CACHE's variant \
                 wherever a case has one",
                o.by_shape.len(),
                cache.display()
            );
            Some(o)
        }
        (Some(_), None) => anyhow::bail!(
            "--autotune needs --manifest: the cache is read against \
             a plan, not against a shape list"
        ),
        _ => None,
    };

    type CaseSet<'a> = (&'a PathBuf, String, BTreeMap<(String, String), String>);
    let mut sets: Vec<CaseSet<'_>> = Vec::new();
    for dir in &args.data {
        let cases_path = dir.join("cases.txt");
        let cases = std::fs::read_to_string(&cases_path).map_err(|e| {
            anyhow::anyhow!(
                "{}: {e} — run the internal reference harness step 1 first",
                cases_path.display()
            )
        })?;
        let shared = read_shared(dir)?;
        sets.push((dir, cases, shared));
    }

    let stream = hip.create_stream(StreamFlags::NonBlocking)?;
    let launch_counters = LaunchCounters::new();
    let mut launcher = Launcher::new("gate", &launch_counters);

    let (mut pass, mut fail, mut skipped) = (0usize, 0usize, 0usize);
    println!();
    println!("{:<34} {:<46} {:>12} result", "case", "kernel", "bytes");
    for (dir, cases, shared) in &sets {
        if sets.len() > 1 {
            println!("-- {}", dir.display());
        }
        for line in cases.lines() {
            let task = line.trim();
            if task.is_empty() || task.starts_with('#') {
                continue;
            }
            if let Some(only) = &args.only {
                if !task.contains(only.as_str()) {
                    continue;
                }
            }
            let cid = task.replace(' ', "_");
            // The saber half of the override. The descriptor is rebuilt by enumerating at THIS case's
            // geometry and taking the instantiation the record names: the kernarg carries the case's
            // own strides, and a variant id is an FNV of the symbol, so it names the same
            // instantiation at both geometries.
            if let Some(o) = overrides.as_mut() {
                if let Some((tune_op, which)) = saber_tune_op(task) {
                    let sig = tune_op.tuning_signature();
                    if let Some(variant_id) = o.saber_by_sig.get(&sig).map(|(id, _)| *id) {
                        let set = saber_exec::enumerate_variants(
                            &tune_op,
                            &rules,
                            &saber_rules,
                            &symbols,
                        );
                        let found = set
                            .variants
                            .iter()
                            .find(|v| v.variant_id() == variant_id)
                            .and_then(|v| {
                                v.descs.get(which).map(|d| (v.kernel.clone(), d.clone()))
                            });
                        let Some((kernel, desc)) = found else {
                            // Fail loud: the record names an instantiation the rules cannot enumerate
                            // at this geometry, or a dispatch the route does not have.
                            println!(
                                "{:<34} {:<46} {:>12} NOT ENUMERABLE (the cache binds variant \
                             {variant_id}, dispatch {which})",
                                task, "-", "-"
                            );
                            fail += 1;
                            continue;
                        };
                        o.saber_used.insert(sig);
                        match run_case(
                            &ctx,
                            &mut launcher,
                            &stream,
                            dir,
                            shared,
                            &cid,
                            &desc,
                            Shape::new(0, 0, 0),
                            args.budget_mib << 20,
                        )? {
                            Verdict::Ok { bytes } => {
                                println!("{:<34} {:<46} {bytes:>12} OK [cache]", task, kernel);
                                pass += 1;
                            }
                            Verdict::Mismatch {
                                bytes,
                                bad,
                                first,
                                buffer,
                            } => {
                                println!(
                                    "{:<34} {:<46} {bytes:>12} MISMATCH {bad} bytes (first in \
                                 `{buffer}` at {first}) [cache]",
                                    task, kernel
                                );
                                fail += 1;
                            }
                            Verdict::Refused(why) | Verdict::Skipped(why) => {
                                println!(
                                    "{:<34} {:<46} {:>12} SKIP ({why}) [cache]",
                                    task, kernel, "-"
                                );
                                skipped += 1;
                            }
                        }
                        continue;
                    }
                }
            }
            // The cache's variant wins where it has one: that is the launch the plan promotes to
            // `measured`, and the one this mode exists to byte-compare.
            if let Some(o) = overrides.as_mut() {
                if let Ok((_, shape)) =
                    resolve_case(task, &rules, &saber_rules, &head_rules, &glue_rules)?
                {
                    let key = (shape.m, shape.n, shape.k);
                    if task.starts_with("gemm ") {
                        if let Some(descs) = o.by_shape.get(&key) {
                            o.used.insert(key);
                            let descs = descs.clone();
                            let head = descs[0].clone();
                            match run_case(
                                &ctx,
                                &mut launcher,
                                &stream,
                                dir,
                                shared,
                                &cid,
                                &head,
                                shape,
                                args.budget_mib << 20,
                            )? {
                                Verdict::Ok { bytes } => {
                                    println!(
                                        "{:<34} {:<46} {bytes:>12} OK [cache]",
                                        task, head.kernel
                                    );
                                    pass += 1;
                                }
                                Verdict::Mismatch {
                                    bytes,
                                    bad,
                                    first,
                                    buffer,
                                } => {
                                    println!(
                                        "{:<34} {:<46} {bytes:>12} MISMATCH {bad} bytes (first in \
                                     `{buffer}` at {first}) [cache]",
                                        task, head.kernel
                                    );
                                    fail += 1;
                                }
                                Verdict::Refused(why) => {
                                    println!(
                                        "{:<34} {:<46} {:>12} REFUSED ({why}) [cache]",
                                        task, head.kernel, "-"
                                    );
                                    skipped += 1;
                                }
                                Verdict::Skipped(why) => {
                                    println!(
                                        "{:<34} {:<46} {:>12} SKIP ({why}) [cache]",
                                        task, head.kernel, "-"
                                    );
                                    skipped += 1;
                                }
                            }
                            continue;
                        }
                    }
                }
            }
            let (desc, shape) =
                match resolve_case(task, &rules, &saber_rules, &head_rules, &glue_rules)? {
                    Ok(pair) => pair,
                    Err(e) => {
                        // A refusal is a result, not a crash: the reference launcher refuses these shapes
                        // too, and the gate records that agreement.
                        println!("{:<34} {:<46} {:>12} REFUSED ({e})", task, "-", "-");
                        skipped += 1;
                        continue;
                    }
                };
            match run_case(
                &ctx,
                &mut launcher,
                &stream,
                dir,
                shared,
                &cid,
                &desc,
                shape,
                args.budget_mib << 20,
            )? {
                Verdict::Ok { bytes } => {
                    println!("{:<34} {:<46} {bytes:>12} OK", task, desc.kernel);
                    pass += 1;
                }
                Verdict::Mismatch {
                    bytes,
                    bad,
                    first,
                    buffer,
                } => {
                    println!(
                    "{:<34} {:<46} {bytes:>12} MISMATCH {bad} bytes (first in `{buffer}` at {first})",
                    task, desc.kernel
                );
                    fail += 1;
                }
                Verdict::Refused(why) => {
                    println!(
                        "{:<34} {:<46} {:>12} REFUSED ({why})",
                        task, desc.kernel, "-"
                    );
                    skipped += 1;
                }
                Verdict::Skipped(why) => {
                    println!("{:<34} {:<46} {:>12} SKIP ({why})", task, desc.kernel, "-");
                    skipped += 1;
                }
            }
        }
    }

    if let Some(o) = &overrides {
        let missed: Vec<&(u32, u32, u32)> =
            o.by_shape.keys().filter(|k| !o.used.contains(*k)).collect();
        println!();
        println!(
            "autotune {} of {} bound GEMM shape(s) had a captured case; the cache's variant \
             was replayed for those",
            o.used.len(),
            o.by_shape.len()
        );
        for k in &missed {
            println!(
                " NOT GATED  M={} N={} K={}  ({})",
                k.0,
                k.1,
                k.2,
                o.routes[k].join(", ")
            );
        }
        if !missed.is_empty() {
            println!(
                " a bound variant with no captured case is a launch the engine \
                 would make and nothing has byte-compared AT THAT SHAPE"
            );
        }
        let saber_missed: Vec<&String> = o
            .saber_by_sig
            .keys()
            .filter(|k| !o.saber_used.contains(*k))
            .collect();
        // By family, because the denominator is every non-GEMM route the plan binds and the
        // grammars live in different captures: the reference's in `.cache/capture-saber`, the attn_glue
        // trio's in `.cache/capture-glue`. Both are gated, one `--data` directory at a time, so
        // a single run covers one grammar's families and prints the other's as NOT GATED.
        let mut by_family: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
        for k in o.saber_by_sig.keys() {
            let fam = k.split('/').next().unwrap_or("?");
            let e = by_family.entry(fam).or_insert((0, 0));
            e.1 += 1;
            if o.saber_used.contains(k) {
                e.0 += 1;
            }
        }
        println!(
            "autotune {} of {} bound non-GEMM shape(s) had a captured case: {}",
            o.saber_used.len(),
            o.saber_by_sig.len(),
            by_family
                .iter()
                .map(|(fam, (used, all))| format!("{fam} {used}/{all}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        for k in &saber_missed {
            println!(" NOT GATED  {k}  ({})", o.saber_by_sig[*k].1.join(", "));
        }
    }

    let (scratch, persistent) = launcher.arena_bytes();
    println!();
    println!("{pass} bit-exact, {fail} MISMATCH, {skipped} skipped");
    println!("load counters  {}", load_counters.snapshot().line());
    println!("launch counters {}", launch_counters.snapshot().line());
    println!(
        "workspace arenas scratch {} B, persistent {} B (held across cases, never reallocated)",
        scratch, persistent
    );
    Ok(fail)
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    stream: &Stream,
    data: &Path,
    shared: &BTreeMap<(String, String), String>,
    cid: &str,
    desc: &LaunchDescriptor,
    shape: Shape,
    budget_bytes: u64,
) -> anyhow::Result<Verdict> {
    // ---- resolve every buffer the descriptor names ----------------------------------------
    let mut owned: Vec<DevicePtr> = Vec::new();
    let mut bindings = ArgBindings::new(shape);
    let mut expects: Vec<(String, Vec<u8>, usize)> = Vec::new(); // (name, want, index into `owned`)
    let mut seen: BTreeMap<String, ()> = BTreeMap::new();
    let mut total_bytes = 0u64;

    for slot in &desc.args {
        let ArgValue::Buffer { buffer } = &slot.value else {
            continue;
        };
        if seen.insert(buffer.clone(), ()).is_some() {
            continue;
        }
        // An input captured for this case, possibly shared with others at the same shape (the
        // MXFP4 weights are per shape, not per M), and an expected output. A buffer can be BOTH:
        // saber rewrites the conv-state cache and the fp32 recurrent state IN PLACE, so the
        // capture dumps them twice and the gate has to upload the first and compare the second.
        let input = match npy::read_u8_opt(&data.join(format!("{cid}.in.{buffer}.npy")))? {
            Some(bytes) => Some(bytes),
            None => match shared.get(&(cid.to_string(), buffer.clone())) {
                Some(file) => Some(npy::read_u8(&data.join(format!("{file}.npy")))?),
                None => None,
            },
        };
        let want = npy::read_u8_opt(&data.join(format!("{cid}.out.{buffer}.npy")))?;
        if input.is_none() && want.is_none() {
            // Not captured, so it must be a workspace the descriptor declared — and then the
            // LAUNCHER owns it, by kind. If the descriptor did not declare it, the plan is wrong.
            if desc.workspaces.iter().any(|w| &w.name == buffer) {
                continue;
            }
            return Ok(Verdict::Refused(format!("no data for buffer `{buffer}`")));
        }
        let bytes = input
            .as_ref()
            .map_or(0, Vec::len)
            .max(want.as_ref().map_or(0, Vec::len));
        total_bytes += bytes as u64;
        if total_bytes > budget_bytes {
            return Ok(Verdict::Skipped(format!(
                "{} MiB exceeds the budget",
                total_bytes >> 20
            )));
        }
        let p = ctx.hip.malloc(bytes as u64)?;
        match &input {
            Some(b) => ctx.hip.memcpy_h2d_blocking(&p, b, stream)?,
            // An output with no captured input starts zeroed, so a kernel that fails to write a
            // byte shows up as a mismatch rather than as whatever the allocator handed back.
            None => ctx.hip.memset(&p, 0, stream)?,
        }
        // Binding by name takes priority over the launcher's own workspace of the same name, which
        // is how a case whose expected output IS a workspace (attention decode's split-KV
        // partials) can be read back at all.
        bindings.bind(buffer, p.addr());
        if let Some(w) = want {
            expects.push((buffer.clone(), w, owned.len()));
        }
        owned.push(p);
    }
    if expects.is_empty() {
        return Ok(Verdict::Refused(
            "the capture has no expected output for this case".to_string(),
        ));
    }
    ctx.hip.sync_stream(stream)?;

    // ---- the engine's own launch path -------------------------------------------------------
    // ONE dispatch: the reference ran each kernel on its own and dumped its own buffers, so an
    // op that declares two launches (attention decode, whose split-KV combine is the second) is
    // two SEPARATE cases here, exactly as the capture wrote them.
    launcher.dispatch_one(ctx, desc, &bindings, stream)?;
    ctx.hip.sync_stream(stream)?;

    // ---- byte compare ----------------------------------------------------------------------
    let mut nbytes = 0usize;
    let mut bad = 0usize;
    let mut first = 0usize;
    let mut worst = String::new();
    for (name, want, index) in &expects {
        let mut got = vec![0u8; want.len()];
        ctx.hip
            .memcpy_d2h_blocking(&mut got, &owned[*index], stream)?;
        nbytes += got.len();
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            if g != w {
                if bad == 0 {
                    first = i;
                    worst = name.clone();
                }
                bad += 1;
            }
        }
    }
    // Free before the next case: there is ~1.0 GiB of free VRAM beside a 0.95-utilisation server.
    drop(owned);
    Ok(if bad == 0 {
        Verdict::Ok { bytes: nbytes }
    } else {
        Verdict::Mismatch {
            bytes: nbytes,
            bad,
            first,
            buffer: worst,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_case_grammar_is_the_capture_scripts() {
        // Exactly the five lines `capture_ref.py` writes and `saber-kplan` reads.
        assert!(matches!(
            parse_mxfp4_case("gemm 8 5120 5120").unwrap().0,
            Op::Gemm { has_wref: true, .. }
        ));
        assert!(matches!(
            parse_mxfp4_case("gemm_nw 8 10240 5120").unwrap().0,
            Op::Gemm {
                has_wref: false,
                ..
            }
        ));
        let (op, shape) = parse_mxfp4_case("add_rms 128 8704 1e-06").unwrap();
        assert!(matches!(op, Op::AddRmsQuant { .. }));
        assert_eq!(shape, Shape::new(128, 0, 8704));
        let (_, shape) = parse_mxfp4_case("silu 64 17408").unwrap();
        assert_eq!(shape, Shape::new(64, 17408, 0));
        let (op, shape) = parse_mxfp4_case("gdn 8 3072 9216 1e-06").unwrap();
        assert!(matches!(op, Op::GdnNormQuant { zs: 9216, .. }));
        assert_eq!(shape, Shape::new(8, 3072, 0));
    }

    /// The head grammar, field for field with what `tools/capture/capture_head.py` writes.
    #[test]
    fn the_head_grammar_resolves_the_three_head_ops() {
        let rules = HeadRules::prod_head().expect("the head rules parse");
        let d = resolve_head("embed 7 5120 0 248320", &rules).expect("embed resolves");
        assert_eq!(d.kernel, "saber_embed_bf16");
        assert_eq!(d.grid, [7, 1, 1]);
        // The vocabulary slice reaches the kernarg image as a VALUE, not as a buffer.
        let vend = d
            .args
            .iter()
            .find(|a| a.name == "vocab_end")
            .expect("vocab_end");
        assert!(matches!(vend.value, ArgValue::I32 { ival: 248_320 }));

        let d = resolve_head("row_gather 1 5120 5120", &rules).expect("row_gather resolves");
        assert_eq!(d.kernel, "saber_row_gather_bf16");
        assert_eq!(d.grid, [1, 1, 1]);

        let d = resolve_head("argmax 3 248320 248320", &rules).expect("argmax resolves");
        assert_eq!(d.kernel, "saber_argmax_bf16");
        assert_eq!(d.grid, [3, 1, 1]);
    }

    /// NEGATIVE CONTROL: a shape `head::resolve` refuses is refused HERE too, so the gate records
    /// the refusal as a result instead of launching a grid of zero workgroups.
    #[test]
    fn a_head_shape_the_kernels_cannot_serve_is_refused() {
        let rules = HeadRules::prod_head().expect("the head rules parse");
        let e = resolve_head("embed 7 5120 0 0", &rules).expect_err("an empty vocab slice");
        assert!(e.contains("empty"), "{e}");
        let e = resolve_head("row_gather 1 5120 512", &rules).expect_err("a narrow stride");
        assert!(e.contains("narrower"), "{e}");
        assert!(resolve_head("embed_bf16 1 2 3 4", &rules).is_err());
    }

    /// The glue grammar, field for field with what `the internal reference harness` writes:
    ///
    ///   qk_rope    <m> <nq> <nkv> <head_dim> <qkv_n> <interleaved> <pmode>
    ///   gate_quant <m> <n> <head_dim> <gs>
    ///   kv_append  <t> <nkv> <head_dim> <block_size> <k_stride> <v_stride> <blk> <page> <head>
    ///              <scale label>
    ///
    /// The trailing label on `qk_rope` and `kv_append` is data the capture needs to name two
    /// dumps of one shape; the planner stops before it, and so does this.
    #[test]
    fn the_glue_grammar_resolves_the_three_glue_ops() {
        let rules = GlueRules::prod().expect("the glue rules parse");
        // TP=2's rank shard: 12 q heads, 2 kv heads, 256-wide heads, [q|gate] then k then v.
        let d = resolve_glue("qk_rope 8 12 2 256 7168 1 mm", &rules).expect("qk_rope resolves");
        assert_eq!(d.plugin, "attn_glue");
        assert!(d.kernel.starts_with("qk_norm_rope"), "{}", d.kernel);
        let chunked = resolve_glue("qk_rope 8 12 2 256 7168 0 mm", &rules).expect("chunked too");
        // The flag is a SYMBOL, not a kernarg: the two decompositions are separate instantiations.
        assert_ne!(d.kernel, chunked.kernel);

        let d = resolve_glue("gate_quant 8 3072 256 7168", &rules).expect("gate_quant resolves");
        assert_eq!(d.plugin, "attn_glue");
        assert!(d.kernel.contains("gate"), "{}", d.kernel);

        let d = resolve_glue("kv_append 5 2 256 16 512 7168 16384 512 8192 s1.0", &rules)
            .expect("kv_append resolves");
        assert_eq!(d.plugin, "attn_glue");
        // The strides reach the kernarg image as VALUES: one instantiation serves every pitch.
        assert!(
            d.args
                .iter()
                .any(|a| matches!(a.value, ArgValue::I64 { ival: 7168 })),
            "the projection-width V stride is in the kernarg image"
        );
    }

    /// NEGATIVE CONTROL: the four lines `capture_glue.py` writes as refusal controls are refused
    /// HERE too, so the gate records a refusal instead of launching a kernel that cannot serve
    /// the shape. Each names a necessary condition of a compiled instantiation.
    #[test]
    fn a_glue_shape_the_kernels_cannot_serve_is_refused() {
        let rules = GlueRules::prod().expect("the glue rules parse");
        for line in [
            "qk_rope 8 12 2 128 7168 1",  // head_dim is a template parameter
            "qk_rope 8 12 2 256 4096 1",  // projection too narrow for the heads
            "gate_quant 8 3072 256 3072", // gate stride cannot hold [q|gate]
            "kv_append 8 2 256 32 512 512 16384 512 8192", // block_size is compiled in
        ] {
            assert!(
                resolve_glue(line, &rules).is_err(),
                "`{line}` must be refused, not launched"
            );
        }
        // And an op outside the grammar is an error, never a default.
        assert!(resolve_glue("qk_norm_rope 8 12 2 256 7168 1", &rules).is_err());
    }

    /// The `--autotune` override rebuilds a descriptor by enumerating at the CASE's geometry and
    /// taking the instantiation the record names, so the case grammar and the `TuneOp` the sweeper
    /// measures must read the same fields. The signature is the contract between them.
    #[test]
    fn the_case_grammar_agrees_with_the_tune_op() {
        let (op, which) =
            saber_tune_op("qk_rope 8 12 2 256 7168 1 mm").expect("a glue case names a TuneOp");
        assert_eq!(which, 0, "the glue ops are one launch each");
        assert_eq!(op.tuning_signature(), "qk_norm_rope/8/12/2/256");
        let (op, _) = saber_tune_op("gate_quant 8 3072 256 7168").expect("gate_quant");
        assert_eq!(op.tuning_signature(), "attn_gate_quant/8/3072/256");
        let (op, _) =
            saber_tune_op("kv_append 5 2 256 16 512 7168 16384 512 8192 s1.0").expect("kv_append");
        // The strides are NOT in the signature: they describe the caller's buffers, and the
        // record is about the instantiation.
        assert_eq!(op.tuning_signature(), "kv_append/5/2/256/16");

        // The chunked mrope has no record to compare against -- the sweep offers the config's
        // own `mrope_interleaved` -- so its cases are gated against the rules instead.
        assert!(saber_tune_op("qk_rope 8 12 2 256 7168 0 mm").is_none());
        // A sub-op names its ROUTE's op and the second dispatch, not an op of its own.
        let (route, which) =
            saber_tune_op("attn_combine 1 5 24 4 8192 8192 2048 0.0625 fp8").expect("attn_combine");
        assert_eq!(which, 1);
        assert!(
            route.tuning_signature().starts_with("attn_decode/"),
            "{}",
            route.tuning_signature()
        );
    }

    #[test]
    fn an_unknown_op_is_an_error_not_a_default() {
        assert!(parse_mxfp4_case("matmul 1 2 3").is_err());
        assert!(parse_mxfp4_case("gemm 1 2").is_err());
    }
}
