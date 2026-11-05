//! Emit saber [`LaunchDescriptor`](saber_kernels::LaunchDescriptor)s as JSON for the standalone
//! HIP harness (`crates/saber-kernels/hip/harness/run_hsaco.cpp`) to replay, the same way
//! `saber-kplan` does for the MXFP4 plugin.
//!
//! The harness never re-derives a warp count, a split law or a kernarg offset: it reads the kernel
//! symbol, grid, block, dynamic LDS and the kernarg byte layout from here, which is the same
//! `saber::resolve` the engine will call. That is what makes the bit-exact gate a test of the PORT.
//!
//! Task lines on stdin, a JSON array on stdout. Fields are space-separated and positional:
//!
//!   attn_prefill <num_seqs> <q_len> <q_heads> <kv_heads> <max_ctx> <blk_stride> <hd_stride> <scale> <fp8_mode> <kv>
//!   attn_decode  <num_seqs> <q_len> <q_heads> <kv_heads> <max_ctx> <blk_stride> <hd_stride> <scale> <kv>
//!   attn_combine <same as attn_decode>
//!   gdn_fused    <n_seqs> <h> <hg> <maxq> <o_rows> <xpitch> <cs_seq> <cs_dim> <cs_tok> <ci_stride> <ab_stride> <st_slot> <st_head> <sidx_stride> <scale> <sp_thr> <ab_bf16>
//!   gdn_conv_upd <same 17 fields as gdn_fused>   # the decode PAIR's first launch
//!   gdn_recurrent <same 17 fields as gdn_fused>   # ... and its second
//!   gdn_prep     <n_seqs> <t> <h> <hg> <xpitch> <cs_seq> <cs_dim> <cs_tok> <ci_stride> <ab_stride> <scale> <sp_thr> <ab_bf16>
//!   gdn_kkt      <same as gdn_prep>
//!   gdn_scan     <same as gdn_prep>
//!   gdn_norm     <rows> <xrow> <zrow> <orow> <eps> <act>
//!   ar_wht       <n_elem> <elem_bytes> <bits> <dtype>
//!   ar_exact     <n_elem> <elem_bytes> <dtype>              # dtype: bf16 | fp16 | fp32
//!   ar_pick      <n_elem> <elem_bytes> <dtype>              # the POLICY op: size picks the family
//!   ar_nq        <m_rows> <k_cols> <eps>
//!
//! `<kv>` is `fp8` or `bf16`; `<dtype>` is `bf16` or `fp16`. `--rules PATH` replaces the embedded
//! prod table.

use saber_kernels::saber::{
    attn_decode_combine, gdn_decode_pair_recurrent, resolve, ArDtype, ArNqShape, ArPayload,
    ArShape, AttnShape, GdnPrefillShape, GdnShape, KvDtype, NormShape, Op, SaberRules, Shape,
};

fn main() {
    // `--rules PATH` overrides the embedded prod table. The all-reduce gate uses it to shrink the
    // IPC slot: prod's 84 MiB slot would be 168 MiB of scratch per rank, and there is ~1.0 GiB of
    // free VRAM next to a 0.95-utilisation server. The slot stride is an addressing offset, so
    // the packed bytes at a given BITS are the same either way.
    let mut args = std::env::args().skip(1);
    let mut rules_path: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--rules" => rules_path = args.next(),
            other => {
                eprintln!("saber-coreplan: unknown flag `{other}`");
                std::process::exit(2);
            }
        }
    }
    let loaded = rules_path.as_ref().map(|p| {
        let text = std::fs::read_to_string(p).unwrap_or_else(|e| {
            eprintln!("saber-coreplan: {p}: {e}");
            std::process::exit(2);
        });
        SaberRules::from_toml(&text)
    });
    let rules = match loaded.unwrap_or_else(SaberRules::prod_saber) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("saber-coreplan: {e}");
            std::process::exit(2);
        }
    };
    let mut input = String::new();
    if std::io::Read::read_to_string(&mut std::io::stdin(), &mut input).is_err() {
        eprintln!("saber-coreplan: cannot read stdin");
        std::process::exit(2);
    }
    let mut out: Vec<serde_json::Value> = Vec::new();
    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        let n = |i: usize| -> u32 {
            f[i].parse()
                .unwrap_or_else(|_| panic!("field {i} of `{line}`"))
        };
        let l = |i: usize| -> i64 {
            f[i].parse()
                .unwrap_or_else(|_| panic!("field {i} of `{line}`"))
        };
        let b = |i: usize| -> u32 {
            f[i].parse::<f32>()
                .unwrap_or_else(|_| panic!("field {i} of `{line}`"))
                .to_bits()
        };
        let r = match f[0] {
            "attn_prefill" => resolve(
                Op::AttnPrefill { fp8_mode: n(9) },
                Shape::Attn(attn_shape(&f, f[10])),
                &rules,
            ),
            "attn_decode" => resolve(Op::AttnDecode, Shape::Attn(attn_shape(&f, f[9])), &rules),
            "attn_combine" => attn_decode_combine(attn_shape(&f, f[9]), &rules),
            "gdn_fused" => resolve(
                Op::GdnFusedUpdate {
                    ab_bf16: n(17) == 1,
                },
                Shape::Gdn(gdn_shape(&f)),
                &rules,
            ),
            "gdn_conv_upd" => resolve(
                Op::GdnDecodePair {
                    ab_bf16: n(17) == 1,
                },
                Shape::Gdn(gdn_shape(&f)),
                &rules,
            ),
            "gdn_recurrent" => {
                // Prod passes no z gate and no norm weight: the gated RMS norm is NOT folded into
                // the epilogue on this unit.
                gdn_decode_pair_recurrent(gdn_shape(&f), n(17) == 1, None, &rules)
            }
            "gdn_prep" => resolve(
                Op::GdnConvPrep {
                    ab_bf16: n(13) == 1,
                },
                Shape::GdnPrefill(gdn_prefill_shape(&f)),
                &rules,
            ),
            "gdn_kkt" => resolve(
                Op::GdnKktSolve,
                Shape::GdnPrefill(gdn_prefill_shape(&f)),
                &rules,
            ),
            "gdn_scan" => resolve(
                Op::GdnChunkScan,
                Shape::GdnPrefill(gdn_prefill_shape(&f)),
                &rules,
            ),
            "gdn_norm" => resolve(
                Op::GdnGatedRmsNorm { act: n(6) },
                Shape::Norm(NormShape {
                    rows: l(1) as u64,
                    channels: 128,
                    xrow: l(2),
                    zrow: l(3),
                    orow: l(4),
                    eps_bits: b(5),
                }),
                &rules,
            ),
            "ar_exact" => resolve(
                Op::AllReduceExact {
                    dtype: payload(f[3]),
                },
                Shape::Ar(ArShape {
                    n_elem: l(1) as u64,
                    elem_bytes: l(2) as u64,
                    world_size: 2,
                }),
                &rules,
            ),
            "ar_pick" => resolve(
                Op::AllReduce {
                    dtype: payload(f[3]),
                },
                Shape::Ar(ArShape {
                    n_elem: l(1) as u64,
                    elem_bytes: l(2) as u64,
                    world_size: 2,
                }),
                &rules,
            ),
            "ar_nq" => resolve(
                Op::AllReduceExactNq,
                Shape::ArNq(ArNqShape {
                    m_rows: n(1),
                    k_cols: n(2),
                    eps_bits: b(3),
                    world_size: 2,
                }),
                &rules,
            ),
            "ar_wht" => resolve(
                Op::AllReduceWht {
                    bits: n(3),
                    dtype: if f[4] == "fp16" {
                        ArDtype::Fp16
                    } else {
                        ArDtype::Bf16
                    },
                },
                Shape::Ar(ArShape {
                    n_elem: l(1) as u64,
                    elem_bytes: l(2) as u64,
                    world_size: 2,
                }),
                &rules,
            ),
            other => {
                eprintln!("saber-coreplan: unknown op `{other}`");
                std::process::exit(2);
            }
        };
        match r {
            Ok(d) => {
                let mut v = serde_json::to_value(&d).expect("descriptor serialises");
                v["task"] = serde_json::Value::String(line.to_owned());
                out.push(v);
            }
            // A refusal is a result, not a crash: the harness records it and moves on.
            Err(e) => out.push(serde_json::json!({ "task": line, "error": e.to_string() })),
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&out).expect("array serialises")
    );
}

/// The attention fields, which prefill, decode and combine share position for position.
fn attn_shape(f: &[&str], kv: &str) -> AttnShape {
    let n = |i: usize| -> u32 { f[i].parse().unwrap_or_else(|_| panic!("field {i}")) };
    let l = |i: usize| -> i64 { f[i].parse().unwrap_or_else(|_| panic!("field {i}")) };
    AttnShape {
        num_seqs: n(1),
        q_len: n(2),
        live_rows: n(2),
        q_heads: n(3),
        kv_heads: n(4),
        head_dim: 256,
        block_size: 16,
        max_blocks: n(5).div_ceil(16),
        max_ctx: n(5),
        // 0 = let the split law choose, which is what prod passes.
        splits: 0,
        kv_dtype: if kv == "bf16" {
            KvDtype::Bf16
        } else {
            KvDtype::Fp8E4m3
        },
        kv_block_stride: l(6),
        kv_head_stride: l(7),
        scale_bits: f[8].parse::<f32>().expect("scale").to_bits(),
        // The capture generates a cache whose per-tensor scales are 1.0, which is what prod
        // serves: `_descale_ptrs` then passes 0 and the kernels read "no descale".
        has_descales: false,
        ragged_rows: 0,
    }
}

/// The GDN decode fields, which the fused kernel and both halves of the pair share position for
/// position -- the pair's arguments are the union of the fused kernel's, because the fused kernel
/// IS the pair with a grid barrier where the kernel boundary was.
fn gdn_shape(f: &[&str]) -> GdnShape {
    let n = |i: usize| -> u32 { f[i].parse().unwrap_or_else(|_| panic!("field {i}")) };
    let l = |i: usize| -> i64 { f[i].parse().unwrap_or_else(|_| panic!("field {i}")) };
    let b = |i: usize| -> u32 { f[i].parse::<f32>().expect("float").to_bits() };
    GdnShape {
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
    }
}

/// The GDN prefill fields, shared by conv_prep, kkt_solve and chunk_scan.
fn gdn_prefill_shape(f: &[&str]) -> GdnPrefillShape {
    let n = |i: usize| -> u32 { f[i].parse().unwrap_or_else(|_| panic!("field {i}")) };
    let l = |i: usize| -> i64 { f[i].parse().unwrap_or_else(|_| panic!("field {i}")) };
    let b = |i: usize| -> u32 { f[i].parse::<f32>().expect("float").to_bits() };
    GdnPrefillShape {
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
    }
}

/// The exact reduce's payload spelling.
fn payload(s: &str) -> ArPayload {
    match s {
        "fp16" => ArPayload::Fp16,
        "fp32" => ArPayload::Fp32,
        _ => ArPayload::Bf16,
    }
}
