//! Reference fixtures: an example manifest and a registry describing the routes P0 ports.
//!
//! These stand in until `saber-kernels` emits real plugin manifests from the `.hsaco` metadata.
//! The rows below are transcriptions of what the production launchers already do — the decode
//! W4A8 band, the A-tiled prefill kernel, the split-K reduce, the 2-rank one-shot all-reduce — so
//! that the planner can be exercised end to end against shapes that exist.
//!
//! What is deliberately **not** here: any fabricated measurement. A registry row says a route
//! exists; only a real autotune record says it is fast, and only that record can bind it.

use crate::provenance::{Promotion, PromotionState, Provenance};
use crate::registry::{
    Constraint, Implementation, LaunchModel, Registry, WorkspaceKind, WorkspaceReq, WorkspaceSizing,
};

/// A complete, valid manifest. Also the documentation of the format.
pub const EXAMPLE_MANIFEST_TOML: &str = r#"# saber manifest — the replacement for 137 SABER_* env vars.

[model]
path = "/mnt/nvme/models/Qwen3.8-27B-NVFP4"
tier_id = "tier:qwen38-27b-nvfp4-mxfp4-dflash2-v22.3"

[device]
target_gfx = "gfx1201"
power_cap_w = 210          # required: a record measured uncapped is invalid at the enforced cap
tp = 2

[[plugins]]
name = "mxfp4"
path = "plugins/gfx1201/mxfp4.hsaco"
build_hash = "b7f3a1c0"

[[plugins]]
name = "saber"
path = "plugins/gfx1201/saber.hsaco"
build_hash = "5d91e2f4"

[[plugins]]
name = "attn_glue"
path = "plugins/gfx1201/attn_glue.hsaco"
build_hash = "1fb9de64"

[[buckets]]
batch = 1
spec_width = 5
ctx_band = 8192

[[buckets]]
batch = 8
spec_width = 5
ctx_band = 8192

# Prefill buckets: M = prompt tokens in a chunk, batch 1, no speculative width. 2048 is the
# smallest chunk above `a_tiled_min_m` (513) where the A-tiled family is the one measured, and
# 8192 is `--max-num-batched-tokens` on the prod unit -- the largest chunk the scheduler forms.
[[buckets]]
phase = "prefill"
tokens = 2048
ctx_band = 8192

[[buckets]]
phase = "prefill"
tokens = 8192
ctx_band = 8192

[plan]
capture = true
"#;

/// The reference registry: the routes P0 ports, with their real constraints.
pub fn reference_registry() -> Registry {
    let maintainer = |when: &str, harness: &str| Provenance::new("maintainer", when, harness);

    Registry::new()
        .with(
            Implementation::new("gemm_w4a8_mxfp4_decode", "mxfp4", "gemm_w4a8")
                .describe(
                    "C[M,N] = A[M,K] @ dequant(W)[N,K]^T, MXFP4 weight, int8 activation, int8 WMMA",
                    "M <= 128, N % 16, K % 128 (the decode GEMM stages unmasked 128-byte K slabs)",
                    "w=mxfp4 a=int8 c=bf16",
                )
                .constrain(Constraint::le("m", 128))
                .constrain(Constraint::div("n", 16))
                .constrain(Constraint::div("k", 128))
                .constrain(Constraint::is_in("a_layout", "row_major"))
                .constrain(Constraint::is_in("w_dtype", "mxfp4 nvfp4"))
                .constrain(Constraint::is_in("a_dtype", "int8 fp8_e4m3"))
                .capture(true, false)
                .code_object(32, 33_280)
                .workspace(WorkspaceReq::new(
                    "split-K partials",
                    WorkspaceKind::Scratch,
                    WorkspaceSizing::SplitKPartials { elem_bytes: 4 },
                ))
                .workspace(WorkspaceReq::new(
                    "split-K block counter",
                    WorkspaceKind::Persistent,
                    WorkspaceSizing::Fixed(4096),
                ))
                .launches(LaunchModel::SplitKReduce { base: 1 })
                .promoted(Promotion::with(
                    PromotionState::Default,
                    maintainer("2026-09-15", "bench_decode_ctx"),
                )),
        )
        .with(
            Implementation::new("gemm_w4a8_mxfp4_wperm", "mxfp4", "gemm_w4a8")
                .describe(
                    "the same GEMM in WMMA fragment order with non-temporal weight loads",
                    "M <= 4096, N % 16, K % 128",
                    "w=mxfp4 a=int8 c=bf16",
                )
                .constrain(Constraint::le("m", 4096))
                .constrain(Constraint::div("n", 16))
                .constrain(Constraint::div("k", 128))
                // Row-major A. This is half of the per-edge layout obligation: without it, the
                // A-tiled row below is the only one that covers a prefill M, and `gdn.out_proj`
                // -- whose producer `gdn_norm_quant` is not fused -- would be routed to a kernel
                // that reads its bytes in the wrong order.
                .constrain(Constraint::is_in("a_layout", "row_major"))
                .constrain(Constraint::is_in("w_dtype", "mxfp4 nvfp4"))
                .constrain(Constraint::is_in("a_dtype", "int8 fp8_e4m3"))
                .capture(true, false)
                .code_object(32, 33_280)
                .workspace(WorkspaceReq::new(
                    "split-K partials",
                    WorkspaceKind::Scratch,
                    WorkspaceSizing::SplitKPartials { elem_bytes: 4 },
                ))
                .workspace(WorkspaceReq::new(
                    "split-K block counter",
                    WorkspaceKind::Persistent,
                    WorkspaceSizing::Fixed(4096),
                ))
                .launches(LaunchModel::SplitKReduce { base: 1 })
                .promoted(Promotion::with(
                    PromotionState::Measured,
                    maintainer("2026-09-02", "ab.sh"),
                )),
        )
        .with(
            // The folded-scale GEMM: row-major A, no M ceiling. It is the row that makes a
            // row-major edge routable at a PREFILL M -- `gdn.out_proj`'s A side is written by
            // `gdn_norm_quant`, which has one instantiation and is not fused, so at M=8192 the
            // A-tiled row refuses it on `a_layout` and the decode/wperm rows refuse it on M.
            // Without this row that edge
            // has no kernel at all, which is how the per-edge obligation first showed up here:
            // as a `NoImplementation` rather than as a wrong pick.
            Implementation::new("gemm_w4a8_mxfp4_folded", "mxfp4", "gemm_w4a8")
                .describe(
                    "row-major A, per-row reference exponent folded into the epilogue",
                    "any M, N % 16, K % 64",
                    "w=mxfp4 a=int8 c=bf16",
                )
                .constrain(Constraint::div("n", 16))
                .constrain(Constraint::div("k", 64))
                .constrain(Constraint::is_in("a_layout", "row_major"))
                .constrain(Constraint::is_in("w_dtype", "mxfp4 nvfp4"))
                .constrain(Constraint::is_in("a_dtype", "int8 fp8_e4m3"))
                .capture(true, false)
                .code_object(32, 33_280)
                .workspace(WorkspaceReq::new(
                    "split-K partials",
                    WorkspaceKind::Scratch,
                    WorkspaceSizing::SplitKPartials { elem_bytes: 4 },
                ))
                .launches(LaunchModel::SplitKReduce { base: 1 })
                .promoted(Promotion::with(
                    PromotionState::Measured,
                    maintainer("2026-09-08", "bench_prefill"),
                )),
        )
        .with(
            Implementation::new("gemm_w4a8_mxfp4_atiled", "mxfp4", "gemm_w4a8")
                .describe(
                    "prefill GEMM with activations in fragment-tiled layout, A straight to WMMA",
                    "M >= 2048, N % 16, K % 64",
                    "w=mxfp4 a=int8 c=bf16",
                )
                .constrain(Constraint::ge("m", 2048))
                .constrain(Constraint::div("n", 16))
                .constrain(Constraint::div("k", 64))
                // …and the other half: this kernel reads A already in WMMA fragment order, so a
                // row-major edge must not reach it whatever M says.
                .constrain(Constraint::is_in("a_layout", "a_tiled"))
                .constrain(Constraint::is_in("w_dtype", "mxfp4 nvfp4"))
                .constrain(Constraint::is_in("a_dtype", "int8 fp8_e4m3"))
                .capture(true, false)
                .code_object(32, 49_152)
                .workspace(WorkspaceReq::new(
                    "A-tile staging",
                    WorkspaceKind::Scratch,
                    WorkspaceSizing::PerRow { bytes_per_row: 512 },
                ))
                .launches(LaunchModel::Fixed(1))
                .promoted(Promotion::with(
                    PromotionState::Measured,
                    maintainer("2026-09-02", "bench_prefill"),
                )),
        )
        .with(
            // Paged attention, decode. TWO launches: the split-KV combine is a second dispatch
            // over the partials the first one wrote, which is why the op declares it.
            Implementation::new("attn_decode_h256_gqa6_fp8kv", "saber", "attn_decode_paged")
                .describe(
                    "paged causal attention, split-KV, fp8 KV cache",
                    "head_dim 256, GQA 6, paged block 16, q_len <= 10",
                    "q=bf16 kv=fp8_e4m3 out=bf16",
                )
                .constrain(Constraint::eq("head_dim", 256))
                .constrain(Constraint::eq("gqa", 6))
                .constrain(Constraint::eq("block_size", 16))
                .constrain(Constraint::eq("causal", 1))
                .constrain(Constraint::le("q_len", 10))
                .constrain(Constraint::is_in("q_dtype", "bf16"))
                .constrain(Constraint::is_in("kv_dtype", "fp8_e4m3"))
                .capture(true, false)
                .code_object(32, 33_280)
                .workspace(WorkspaceReq::new(
                    "attn_decode_scratch",
                    WorkspaceKind::Scratch,
                    WorkspaceSizing::Fixed(63_897_600),
                ))
                .launches(LaunchModel::Fixed(2))
                .promoted(Promotion::with(
                    PromotionState::Default,
                    maintainer("2026-09-15", "bench_decode_ctx"),
                )),
        )
        .with(
            // Paged attention, PREFILL: query-tiled, ONE launch per run of equal `q_len`. No
            // `q_len` predicate at all -- the decode row's `q_len <= 10` is what separates the
            // two, and above it the backend sends the run here.
            Implementation::new("attn_prefill_h256_gqa6_fp8kv", "saber", "attn_prefill_paged")
                .describe(
                    "paged causal attention, query-tiled, fp8 KV cache, fp8 QK and PV legs",
                    "head_dim 256, GQA 6, paged block 16; BQ = 64 query rows per block",
                    "q=bf16 kv=fp8_e4m3 out=bf16",
                )
                .constrain(Constraint::eq("head_dim", 256))
                .constrain(Constraint::eq("gqa", 6))
                .constrain(Constraint::eq("block_size", 16))
                .constrain(Constraint::eq("causal", 1))
                .constrain(Constraint::is_in("q_dtype", "bf16"))
                .constrain(Constraint::is_in("kv_dtype", "fp8_e4m3"))
                .capture(true, false)
                .code_object(32, 34_820)
                .launches(LaunchModel::Fixed(1))
                .promoted(Promotion::with(
                    PromotionState::Default,
                    maintainer("2026-09-15", "bench_prefill"),
                )),
        )
        .with(
            // The GDN prefill path, kernel 1 of 3: the conv preamble. Causal conv, qkv split,
            // qk l2 norm, gating and the per-chunk cumsum, all before the scan can start.
            Implementation::new("gdn_conv_prep_w4_h128_bf16", "saber", "gdn_conv_prep")
                .describe(
                    "causal conv + qkv split + qk l2 norm + gating + per-chunk cumsum",
                    "conv width 4, key head 128, value head 128, chunk 64",
                    "bf16 in/out",
                )
                .constrain(Constraint::eq("conv_width", 4))
                .constrain(Constraint::eq("head_k", 128))
                .constrain(Constraint::eq("head_v", 128))
                .constrain(Constraint::eq("chunk", 64))
                .capture(true, false)
                .code_object(32, 8192)
                .launches(LaunchModel::Fixed(1))
                .promoted(Promotion::with(
                    PromotionState::Measured,
                    maintainer("2026-09-18", "bench_prefill"),
                )),
        )
        .with(
            // Kernel 2 of 3: A = (I + strict_lower(diag(beta) K K^T e^dg))^-1 per chunk. The
            // fp32 gram never reaches HBM, which is why this is its own dispatch rather than a
            // prologue of the scan.
            Implementation::new("gdn_kkt_solve_k128_c64_bf16", "saber", "gdn_kkt_solve")
                .describe(
                    "per-chunk triangular solve; the fp32 gram stays in registers and LDS",
                    "key head 128, chunk 64",
                    "bf16 in/out, fp32 gram",
                )
                .constrain(Constraint::eq("head_k", 128))
                .constrain(Constraint::eq("chunk", 64))
                .capture(true, false)
                .code_object(32, 16_640)
                .launches(LaunchModel::Fixed(1))
                .promoted(Promotion::with(
                    PromotionState::Measured,
                    maintainer("2026-09-18", "bench_prefill"),
                )),
        )
        .with(
            // Kernel 3 of 3: the chunked scan. WY recompute, state recurrence and output in one
            // kernel -- and the one that leaves the recurrent state the decode path continues
            // from, which is why a prefill bucket still needs a GDN state slot at all.
            Implementation::new("gdn_chunk_scan_k128_v128_c64_bf16", "saber", "gdn_chunk_scan")
                .describe(
                    "chunked scan: WY recompute, state recurrence and output",
                    "key head 128, value head 128, chunk 64",
                    "bf16 in/out, fp32 recurrent state",
                )
                .constrain(Constraint::eq("head_k", 128))
                .constrain(Constraint::eq("head_v", 128))
                .constrain(Constraint::eq("chunk", 64))
                .capture(true, false)
                .code_object(32, 32_768)
                .launches(LaunchModel::Fixed(1))
                .promoted(Promotion::with(
                    PromotionState::Measured,
                    maintainer("2026-09-18", "bench_prefill"),
                )),
        )
        .with(
            // ---- the attention layer's glue (workstream F, plugin 4) -----------------
            //
            // Three rows the transcription needs because `ModelSpec` names all three: the
            // per-head q/k norm + partial mrope, the output gate's quant, and the paged K/V
            // write. `head_dim` and `rotary_dim` are compiled into the symbols, so they are
            // equalities here exactly as they are in the derived registry.
            Implementation::new("qk_norm_rope_h256_rot64", "attn_glue", "qk_norm_rope")
                .describe(
                    "per-head q/k RMS norm and the partial mrope, ONE launch against prod's five",
                    "head_dim 256, rotary_dim 64",
                    "bf16 in/out, fp32 cos/sin cache, i32 positions",
                )
                .constrain(Constraint::eq("head_dim", 256))
                .constrain(Constraint::eq("rotary_dim", 64))
                .constrain(Constraint::ge("m", 1))
                .capture(true, false)
                .code_object(32, 1024)
                .launches(LaunchModel::Fixed(1))
                .promoted(Promotion::implemented()),
        )
        .with(
            Implementation::new("attn_gate_quant_t256", "attn_glue", "attn_gate_quant")
                .describe(
                    "core * sigmoid(gate) with the per-token e4m3 quant o_proj reads",
                    "head_dim 256",
                    "bf16 in, fp8_e4m3 + per-token fp32 scale out",
                )
                .constrain(Constraint::eq("head_dim", 256))
                .constrain(Constraint::ge("m", 1))
                .capture(true, false)
                .code_object(32, 0)
                .launches(LaunchModel::Fixed(1))
                .promoted(Promotion::implemented()),
        )
        .with(
            Implementation::new("kv_append_h256_b16_fp8", "attn_glue", "kv_append")
                .describe(
                    "the paged fp8 K/V write, both tensors in ONE launch",
                    "head_dim 256, block_size 16",
                    "bf16 in, fp8_e4m3 paged cache out",
                )
                .constrain(Constraint::eq("head_dim", 256))
                .constrain(Constraint::eq("block_size", 16))
                .constrain(Constraint::is_in("kv_dtype", "fp8_e4m3"))
                .constrain(Constraint::ge("m", 1))
                .capture(true, false)
                .code_object(32, 0)
                .launches(LaunchModel::Fixed(1))
                .promoted(Promotion::implemented()),
        )
        .with(
            // The standalone gated RMS norm. PROD binds it and never calls it (its decode kernel
            // folds the arithmetic into its epilogue, and the MXFP4 tier fuses norm+quant into
            // one producer); Saber ported the code object, so the prefill path has a route.
            Implementation::new("gdn_gated_rmsnorm_h128_silu", "saber", "gdn_gated_rmsnorm")
                .describe(
                    "per-head gated RMS norm over the GDN output, silu gate",
                    "channels 128",
                    "bf16 in/out",
                )
                .constrain(Constraint::eq("channels", 128))
                .capture(true, false)
                .code_object(32, 4096)
                .launches(LaunchModel::Fixed(1))
                .promoted(Promotion::with(
                    PromotionState::Measured,
                    maintainer("2026-09-18", "bench_prefill"),
                )),
        )
        .with(
            // The GDN decode step as ONE launch: conv update, grid-wide barrier, recurrent update.
            Implementation::new(
                "gdn_fused_update_w4k128v128_bf16",
                "saber",
                "gdn_fused_update",
            )
            .describe(
                "conv update + grid barrier + recurrent delta-rule update, one launch",
                "conv width 4, key head 128, value head 128",
                "bf16 in/out, fp32 recurrent state",
            )
            .constrain(Constraint::eq("conv_width", 4))
            .constrain(Constraint::eq("head_k", 128))
            .constrain(Constraint::eq("head_v", 128))
            .capture(true, false)
            .code_object(32, 47_104)
            .workspace(WorkspaceReq::new(
                // ONE int32. The last workgroup to arrive resets it in place, so it is zeroed
                // exactly once and graph replay needs no clearing pass -- and `scratch` would
                // be wrong even at four bytes, because an aliased buffer would hand the next
                // launch someone else's count.
                "gdn_fused_barrier",
                WorkspaceKind::Persistent,
                WorkspaceSizing::Fixed(4),
            ))
            .launches(LaunchModel::Fixed(1))
            .promoted(Promotion::with(
                PromotionState::Measured,
                maintainer("2026-09-02", "bench_decode_ctx"),
            )),
        )
        .with(
            // The other GDN decode route: above FUSED_MAX_ITEMS the step is the conv update
            // followed by the recurrent update. TWO launches, declared on the head.
            Implementation::new("gdn_conv_update_w4_h128_bf16", "saber", "gdn_conv_update")
                .describe(
                    "the decode conv update; the recurrent delta-rule update is its SECOND dispatch",
                    "conv width 4, key head 128, value head 128",
                    "bf16 in/out, fp32 recurrent state",
                )
                .constrain(Constraint::eq("conv_width", 4))
                .constrain(Constraint::eq("head_k", 128))
                .constrain(Constraint::eq("head_v", 128))
                .capture(true, false)
                .code_object(32, 8192)
                .launches(LaunchModel::Fixed(2))
                .promoted(Promotion::with(
                    PromotionState::Default,
                    maintainer("2026-09-02", "bench_decode_ctx"),
                )),
        )
        .with(
            // The lossy all-reduce. `exact` is what separates the two families: a request that
            // does not declare it will accept a quantised wire format gets no kernel rather than
            // the quantised one.
            Implementation::new("ar_oneshot_2rank_wht6", "saber", "allreduce")
                .describe(
                    "one-shot 2-rank push all-reduce, Walsh-Hadamard rotated, 6 bits per symbol",
                    "world_size == 2, numel % 64",
                    "bf16 / fp16 payload",
                )
                .constrain(Constraint::eq("world_size", 2))
                .constrain(Constraint::eq("exact", 0))
                .constrain(Constraint::eq("bits", 6))
                .constrain(Constraint::is_in("dtype", "bf16 fp16"))
                .constrain(Constraint::div("numel", 64))
                .capture(true, false)
                .code_object(32, 4)
                .workspace(WorkspaceReq::new(
                    "ar_scratch",
                    WorkspaceKind::IpcPeerVisible,
                    WorkspaceSizing::Fixed(176_160_768),
                ))
                .workspace(WorkspaceReq::new(
                    "ar_flags",
                    WorkspaceKind::IpcPeerVisible,
                    WorkspaceSizing::Fixed(2048),
                ))
                .launches(LaunchModel::Fixed(1))
                .promoted(Promotion::with(
                    PromotionState::Measured,
                    maintainer("2026-09-16", "ar_gate.sh"),
                )),
        )
        .with(
            // The exact kernel is the SAME op, distinguished by `exact == 1`: below the 128 KiB
            // quantisation threshold this is what runs.
            Implementation::new("ar_oneshot_2rank_exact", "saber", "allreduce")
                .describe(
                    "one-shot push all-reduce over P2P, exactly 2 ranks, fp32 accumulate",
                    "world_size == 2, exact",
                    "bf16 / fp16 / fp32 payload",
                )
                .constrain(Constraint::eq("world_size", 2))
                .constrain(Constraint::eq("exact", 1))
                .constrain(Constraint::is_in("dtype", "bf16 fp16 fp32"))
                .capture(true, false)
                .code_object(32, 0)
                .workspace(WorkspaceReq::new(
                    "peer push scratch",
                    WorkspaceKind::IpcPeerVisible,
                    WorkspaceSizing::PerRow {
                        bytes_per_row: 10_240,
                    },
                ))
                .workspace(WorkspaceReq::new(
                    "peer flags",
                    WorkspaceKind::Persistent,
                    WorkspaceSizing::Fixed(4096),
                ))
                .launches(LaunchModel::Fixed(1))
                .promoted(Promotion::with(
                    PromotionState::Certified,
                    maintainer("2026-09-16", "ar_gate.sh"),
                )),
        )
}
