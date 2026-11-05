//! The DFlash2 drafter's forward pass, as a description with a declared launch budget.
//!
//! This is the drafter what [`crate::graph`] is to the target: every launch a decode step makes,
//! named, with the shape it runs at and the kernel-plane op it selects, so that a launch census
//! can be compared against a number that was DECLARED rather than counted afterwards.
//!
//! It is a separate module rather than rows in [`crate::graph::Graph::drafter`] for one concrete
//! reason: that field describes the **MTP** drafter (one bf16 decoder layer plus `fc`), and its ops
//! select on `graph::KernelOp`, an exhaustive enum every match in a 5000-line module covers. The
//! DFlash2 drafter is a different model — five W8A8 layers, a grouped convolution around each
//! sublayer, a candidate selector and a 2-bit head — and adding its eight kernel ids to that enum
//! would put a dead arm in every one of those matches. The same trade the LEDGER records for
//! `saber_kernels::rules::Op`, which broke `saber-exec::tune`'s exhaustive match twice.
//!
//! WHAT IS HERE AND WHAT IS NOT. Every op below either names a `dflash` kernel that exists and is
//! gated, or is marked [`Coverage::Absent`] with what it would need. The budget counts only the
//! former; `uncovered()` returns the latter. A description that quietly omitted the gaps would make
//! the budget agree with the launcher and disagree with prod.

use std::collections::BTreeMap;
use std::fmt;

use saber_kernels::dflash::{DflashOp, DflashRules, DflashShape};

/// The drafter's own configuration, as its `config.json` states it.
///
/// Every field is read from `Qwen3.8-27B-DFlash2-FP8/config.json` rather than inferred, and the
/// three that are easy to guess wrong are noted where they are used: `block_size` comes from the
/// SERVE (`1 + num_speculative_tokens`) and not from the checkpoint's trained value, the selector
/// runs on the MASK positions only (`block_size - 1`), and `target_layers` is the count of the
/// target's hidden states the aux projection concatenates, not a layer index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DflashConfig {
    /// Ranks the shapes below are already sharded for.
    pub tp: u32,
    /// Hidden size, 5120.
    pub hidden: u32,
    /// Decoder layers, 5.
    pub layers: u32,
    /// MLP intermediate size, 17408.
    pub intermediate: u32,
    /// Query heads BEFORE the TP divide, 32.
    pub q_heads: u32,
    /// KV heads before the divide, 8.
    pub kv_heads: u32,
    /// Head dimension, 128.
    pub head_dim: u32,
    /// `1 + num_speculative_tokens` — the query rows per forward. The SERVE sets this, not the
    /// checkpoint: `qwen3_dflash2.py` warns when they disagree and runs anyway.
    pub block_size: u32,
    /// `conv_kernel_size`, 2.
    pub conv_taps: u32,
    /// `conv_group_size`, 16.
    pub conv_group: u32,
    /// `selector_rank`, 256.
    pub selector_rank: u32,
    /// `selector_top_k`, 16.
    pub selector_top_k: u32,
    /// How many of the target's layers feed the aux projection — `len(target_layer_ids)`, 5.
    pub target_layers: u32,
    /// Vocabulary size, 248320.
    pub vocab: u32,
    /// `SABER_DRAFT_RERANK` on the serving unit, 80. Not the module default of 32.
    pub rerank: u32,
    /// `ref_drafthead.BLOCK_N`, 64: the int2 head's column block, and therefore the width of
    /// one block-max entry. It is here beside `rerank` and for the same reason — the SHAPE of
    /// `bm.topk(RERANK)` is `[M, (vocab/BLOCK_N) * KCAND]`, and a plan that could not state that
    /// row width could not declare the selection's launch at all.
    pub head_block_n: u32,
    /// `ref_drafthead.KCAND`, 8: candidates emitted per block.
    pub head_kcand: u32,
}

impl DflashConfig {
    /// `Qwen3.8-27B-DFlash2-FP8` as prod serves it, at `tp` ranks.
    #[must_use]
    pub fn qwen38_dflash2(tp: u32, num_speculative_tokens: u32) -> Self {
        DflashConfig {
            tp: tp.max(1),
            hidden: 5120,
            layers: 5,
            intermediate: 17408,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            block_size: 1 + num_speculative_tokens,
            conv_taps: 2,
            conv_group: 16,
            selector_rank: 256,
            selector_top_k: 16,
            target_layers: 5,
            vocab: 248_320,
            rerank: 80,
            head_block_n: 64,
            head_kcand: 8,
        }
    }

    /// Query heads on one rank.
    #[must_use]
    pub fn q_heads_local(&self) -> u32 {
        self.q_heads / self.tp
    }

    /// KV heads on one rank.
    #[must_use]
    pub fn kv_heads_local(&self) -> u32 {
        self.kv_heads / self.tp
    }

    /// This rank's vocabulary shard.
    #[must_use]
    pub fn vocab_local(&self) -> u32 {
        self.vocab / self.tp
    }

    /// The `qkv_proj` output width on one rank: q + k + v heads, each `head_dim` wide.
    #[must_use]
    pub fn qkv_n(&self) -> u32 {
        (self.q_heads_local() + 2 * self.kv_heads_local()) * self.head_dim
    }

    /// The `kernel_projection` output width: two sides x taps x conv groups.
    #[must_use]
    pub fn kernel_projection_n(&self) -> u32 {
        2 * self.conv_taps * (self.hidden / self.conv_group)
    }

    /// Positions the selector proposes for: the MASK positions, one fewer than the block, because
    /// position 0 carries the bonus token. The census's 7-workgroup `warpMergeSortTopK` grid is
    /// this number.
    #[must_use]
    pub fn selector_positions(&self) -> u32 {
        self.block_size - 1
    }

    /// Rows the CONTEXT STORE runs at: `target_token_ids.shape[0]`, the tokens the target's own
    /// forward covered in this step (`dflash.py:117`). A RUNTIME value, like the attention's
    /// sequence length and not a shape of the model — at batch 1 steady-state decode the target
    /// verifies one block, so it equals `block_size`, and the step right after a prefill runs it
    /// at the whole prompt. The plan declares the decode value and `refusals()` resolves it, which
    /// is what exposes the one real bound here: the fused KV projection is a `bgemm_bf16` and that
    /// kernel's tile stops at 16 rows.
    #[must_use]
    pub fn ctx_rows(&self) -> u32 {
        self.block_size
    }

    /// The row width of `bm`, the int2 head's block maxima: one entry per block of `head_block_n`
    /// vocabulary columns, `head_kcand` of them. 31040 at TP=1, 15520 at TP=2.
    #[must_use]
    pub fn head_block_maxima(&self) -> u32 {
        self.vocab_local() / self.head_block_n * self.head_kcand
    }

    /// The fused context-KV projection's output width: every layer's K and V on one rank.
    #[must_use]
    pub fn context_kv_n(&self) -> u32 {
        self.layers * 2 * self.kv_heads_local() * self.head_dim
    }
}

/// Whether a Saber code object covers an op, and what the gap is when it does not.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Coverage {
    /// A `dflash` op, gated bit-exact. Counts toward the budget.
    Dflash(DflashOp),
    /// Another plugin already covers it (the reference's exact all-reduce, say).
    Other(&'static str),
    /// No code object. The string says what it would take — never "TODO".
    Absent(&'static str),
}

/// One launch site in the drafter's forward pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftOp {
    /// Unique name, `layer.3.mlp.down` style.
    pub name: String,
    /// What covers it.
    pub coverage: Coverage,
    /// The shape it runs at, when a `dflash` op covers it.
    pub shape: Option<DflashShape>,
    /// How many times a decode step launches it.
    pub count: u32,
    /// One line of why it looks the way it does.
    pub note: &'static str,
}

/// Which record the drafter's exact rerank reads the target's `lm_head` from (workstream N6).
///
/// Both are prod's matrix, `bf16(f32(w8) * f32(scale))`, and the rerank's output is the same bit
/// for bit; they differ in BYTES and in who else can read them. [`RerankHead::Bf16`] is
/// `head.rerank.w`, the dequantized bf16 matrix (1.27 GB per TP=2 rank) — sha-identical to the
/// bf16-head target's `lm_head.w`. [`RerankHead::W8`] is `head.rerank.w8` + `head.rerank.ws`,
/// the checkpoint's own e4m3 tensor raw and its scale widened to f32 (0.636 GB) — sha-identical to
/// the fp8-head target's `lm_head.w8` + `lm_head.ws`, and expanded in register by
/// `DflashOp::RerankExactW8`. Which one a pack carries follows the checkpoint's head dtype (the
/// head is served at the checkpoint's own precision, never requantized: the user's law of
/// 2026-09-22), and the engine reads it off the pack, not a flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RerankHead {
    /// `head.rerank.w`, bf16 (the drafter pack before N6; a bf16 checkpoint's only form).
    #[default]
    Bf16,
    /// `head.rerank.w8` + `head.rerank.ws`: the checkpoint's per-channel fp8 head, raw.
    W8,
}

impl RerankHead {
    /// The record names this kind binds, in pack order.
    #[must_use]
    pub fn records(self) -> &'static [&'static str] {
        match self {
            RerankHead::Bf16 => &["head.rerank.w"],
            RerankHead::W8 => &["head.rerank.w8", "head.rerank.ws"],
        }
    }

    /// The rerank op that reads this kind.
    #[must_use]
    pub fn op(self) -> DflashOp {
        match self {
            RerankHead::Bf16 => DflashOp::RerankExactBf16,
            RerankHead::W8 => DflashOp::RerankExactW8,
        }
    }
}

/// The drafter's forward pass at one `M`.
#[derive(Clone, Debug)]
pub struct DraftGraph {
    /// The config it was built from.
    pub cfg: DflashConfig,
    /// The ops, in forward order.
    pub ops: Vec<DraftOp>,
    /// The record the exact rerank reads the head from ([`RerankHead`]; bf16 unless
    /// [`DraftGraph::with_rerank_head`] says otherwise).
    pub rerank_head: RerankHead,
}

impl DraftGraph {
    /// Build the description. `M` is the query row count, `cfg.block_size`.
    #[must_use]
    pub fn build(cfg: DflashConfig) -> Self {
        let m = cfg.block_size;
        let h = cfg.hidden;
        let l = cfg.layers;
        let mut ops: Vec<DraftOp> = Vec::new();
        let mut push = |name: String,
                        coverage: Coverage,
                        shape: Option<DflashShape>,
                        count: u32,
                        note: &'static str| {
            ops.push(DraftOp {
                name,
                coverage,
                shape,
                count,
                note,
            });
        };

        // ---- once per step, before the layers -------------------------------------------------
        // The aux-hidden-state projection: the five target layers' hidden states concatenated, so
        // K is 5 x hidden. This is the census's "234 us outlier", which was simply its largest GEMM.
        let ctx = cfg.ctx_rows();
        // THE ACTIVATION QUANT NOBODY DECLARED. Every `gemm_a8w8_blockscale` in the drafter needs
        // an `e4m3` activation with per-128-group `f32` scales, and for all but two of them a
        // declared op produces it: `attention_conv.prepare` feeds `attn.qkv`, `mlp_conv.prepare`
        // feeds `mlp.gate_up`, `mlp.silu_quant` feeds `mlp.down`. The target's five concatenated
        // hidden states and the attention's output arrive in `bf16` and nothing in this plan
        // quantised them. In prod the quant is inside the layer's own quant method, so a launch
        // census that counts GEMMs never sees it and three rounds of this plan did not either. It
        // showed the first time a replay tried to CHAIN the pass (workstream F round 21) --
        // a staged replay seeds past the gap and never notices, which is worth remembering about
        // every staged gate in this tree.
        push(
            "aux_hidden_projection.quant".into(),
            Coverage::Dflash(DflashOp::ActQuant),
            Some(DflashShape::new(
                ctx,
                cfg.target_layers * h,
                cfg.target_layers * h,
            )),
            1,
            "prod quantises inside the quant method of `model.fc`, which is why no launch census \
             sees it; `dflash_act_quant_g128` is that quant, at the CONTEXT rows",
        );
        push(
            "aux_hidden_projection".into(),
            Coverage::Dflash(DflashOp::GemmBlockscale),
            Some(DflashShape::new(ctx, h, cfg.target_layers * h)),
            1,
            "the target's five tapped layers concatenated: N = hidden, K = 5 x hidden. It runs at \
             the CONTEXT rows, not the query rows -- `combine_hidden_states` folds the target's \
             own hidden states and its output is what the context store then projects",
        );
        // THE CONTEXT-KV STORE, re-read from the source that actually runs it.
        //
        // `precompute_and_store_context_kv` (`qwen3_dflash.py:585`, INHERITED by
        // `DFlash2Qwen3Model` -- `qwen3_dflash2.py` subclasses it and overrides only the decoder
        // layer and the selector) fills the drafter's KV cache from the TARGET's hidden states.
        // The previous declaration had this as five per-layer K norms plus a rope; the source says
        // otherwise, and the difference is not cosmetic:
        //
        //   * `_normalize_context_k` is ONE `ops.rms_norm` over `[L, num_ctx, nkv, head_dim]`
        //     against a `[L, head_dim]` weight -- the five `k_norm.weight`s are STACKED by
        //     `_build_context_kv_buffers` "so the per-layer K-norm runs as a single grouped
        //     kernel". Five launches were declared; prod issues one.
        //   * `_project_context_kv` runs a `ops.rms_norm` of its OWN first -- the drafter's
        //     `hidden_norm` over the context states -- which the plan did not declare at all,
        //     then the fused KV `F.linear`, then a `permute(...).contiguous()`.
        //   * and the store ends with one `reshape_and_cache_flash` PER LAYER, which the plan also
        //     did not declare.
        //
        // So the context store is TEN launches, not six, and four of them were invisible. Both
        // norms and the rope are `upstream::` C++ kernels rather than the inductor-compiled Triton the
        // rest of the drafter runs, because this call site invokes `torch.ops._C` directly and so
        // bypasses the serve's `custom_ops: [..., "none"]` toggle -- which is why they are their
        // own code object (`hip/dflash_ctx.hip`) with their own roundings.
        push(
            "context_kv.hidden_norm".into(),
            Coverage::Dflash(DflashOp::CtxRmsNorm),
            Some(DflashShape::ctx_norm(ctx, h, ctx)),
            1,
            "`_project_context_kv`'s own norm over the context states, ONE group -- the launch \
             the plan had missed entirely",
        );
        // Declared COVERED for one round and taken back by the gate that was supposed to confirm
        // it. `bgemm_bf16`'s reference is `F.linear`, and hipBLASLt is not one kernel: at M = 8,
        // K = 5120 it is identical at N = 1280 and N = 2560 and DIFFERS in 19 of 40960 at
        // N = 5120, which is this shape. The plan carries what the gate measured.
        push(
            "context_kv.kv_projection".into(),
            Coverage::Dflash(DflashOp::BgemmBf16),
            Some(DflashShape::new(ctx, cfg.context_kv_n(), h)),
            1,
            "the fused KV projection over every layer at once: N = L x 2 x kv_size, which is \
             10240 on one rank. The rules' bgemm envelope stopped at 2560 for a round because \
             `F.linear` differs from this kernel in 19 of 40960 at N = 5120 -- an algorithm \
             switch inside hipBLASLt, not a wider launch -- and F26 took the shape back with the \
             K-reduction arms that question needs",
        );
        push(
            "context_kv.layout".into(),
            Coverage::Dflash(DflashOp::CtxKvLayout),
            Some(DflashShape::ctx_layout(
                ctx,
                cfg.layers,
                cfg.kv_heads_local(),
                cfg.head_dim,
            )),
            1,
            "K and V separated and transposed to layer-major, one contiguous copy. Still a \
             fusion to take one day -- a Saber store would have the projection write layer-major \
             -- but a CHAIN cannot skip it, because nothing else stands between the projection \
             and the K norm",
        );
        push(
            "context_kv.k_norm".into(),
            Coverage::Dflash(DflashOp::CtxRmsNorm),
            Some(DflashShape::ctx_norm(
                cfg.layers * ctx * cfg.kv_heads_local(),
                cfg.head_dim,
                ctx * cfg.kv_heads_local(),
            )),
            1,
            "ONE grouped launch over all five layers: the weight row advances every \
             num_ctx x nkv rows",
        );
        push(
            "context_kv.rope".into(),
            Coverage::Dflash(DflashOp::CtxRope),
            Some(DflashShape::ctx_rope(
                cfg.layers * ctx,
                cfg.kv_heads_local(),
                cfg.head_dim,
                cfg.head_dim,
            )),
            1,
            "ONE launch for all five layers, K only and in place, with the positions repeated L \
             times -- rotary_dim == head_dim, so a FULL plain rope",
        );
        push(
            "context_kv.cache_insert".into(),
            Coverage::Dflash(DflashOp::KvAppendFp8),
            Some(DflashShape::new(ctx, cfg.kv_heads_local(), cfg.head_dim)),
            cfg.layers,
            "`do_kv_cache_update` per layer -- the same `reshape_and_cache_flash` the drafter's \
             own query rows take, at the context rows",
        );
        push(
            "input_embedding".into(),
            Coverage::Other("head::embedding"),
            None,
            1,
            "the `head` plugin's vocab-parallel lookup at the drafter's own shard -- already a \
             code object. `input_embedding_scale` is ABSENT from this checkpoint's dflash_config, \
             so it defaults to 1.0 and the multiply prod would apply is a no-op; a checkpoint that \
             set it would need one more elementwise op",
        );
        push(
            "input_embedding.all_reduce".into(),
            Coverage::Other("saber_ar_exact"),
            None,
            1,
            "the ELEVENTH reduce, and the plan did not declare it until ws-s6. The drafter starts \
             its stack from the TARGET's `VocabParallelEmbedding`, whose `forward` all-reduces at \
             `tp_size > 1`, and the lookup writes a ZERO row for every id outside this rank's \
             window -- so without it a two-rank proposal starts from half the tokens' embeddings \
             and zeros for the rest. The target's own graph has always declared one",
        );

        // ---- the five layers -------------------------------------------------------------------
        for i in 0..l {
            let p = |s: &str| format!("layer.{i}.{s}");
            if i == 0 {
                // The ONLY standalone norm in the drafter: every other one follows a convolution's
                // `finish` half and is fused into it. Layer 0 takes the `residual is None` branch
                // (`qwen3_dflash2.py:164`), so it norms its input -- the token EMBEDDING, which is
                // what `DFlashQwen3Model.forward` starts the stack from -- with nothing to add.
                push(
                    p("input_layernorm"),
                    Coverage::Dflash(DflashOp::RmsNorm),
                    Some(DflashShape::new(m, h, h)),
                    1,
                    "the only norm in the drafter that is not the tail of a conv_add_rms",
                );
            }
            // Each sublayer is wrapped by a grouped convolution: `prepare` before it and `finish`
            // after, with the coefficients from a bf16 projection. Two sublayers, so four conv
            // launches and two projections per layer -- and `kernel_projection` x 2 x 5 layers is
            // the census's x10 bf16 GEMM at 38.06 us, which it had attributed to the einsum.
            for (side, tag) in [(0u32, "attention_conv"), (1, "mlp_conv")] {
                push(
                    p(&format!("{tag}.kernel_projection")),
                    Coverage::Dflash(DflashOp::BgemmBf16),
                    Some(DflashShape::new(m, cfg.kernel_projection_n(), h)),
                    1,
                    "bf16: the checkpoint lists kernel_projection in modules_to_not_convert",
                );
                push(
                    p(&format!("{tag}.prepare")),
                    Coverage::Dflash(DflashOp::GroupedConvQuant),
                    Some(DflashShape::conv(
                        m,
                        h,
                        h / cfg.conv_group,
                        cfg.block_size,
                        0,
                    )),
                    1,
                    "the conv and the per-128-group e4m3 quant its consumer needs, in one pass",
                );
                let _ = side;
            }
            push(
                p("attn.qkv"),
                Coverage::Dflash(DflashOp::GemmBlockscale),
                Some(DflashShape::new(m, cfg.qkv_n(), h)),
                1,
                "q + k + v on one rank",
            );
            push(
                p("attn.qk_norm_rope"),
                Coverage::Dflash(DflashOp::QkNormRope),
                Some(DflashShape::qk_rope(
                    m,
                    cfg.q_heads_local(),
                    cfg.kv_heads_local(),
                    cfg.head_dim,
                )),
                1,
                "q_norm and k_norm are in modules_to_not_convert, so bf16, then rope. This is the \
                 drafter's own QUERY tokens; the context K took the batched rope above. It was \
                 `Other(\"attn_glue::qk_norm_rope\")` through F27 and that was the wrong plugin, \
                 not a missing one: `attn_glue` implements the TARGET's GemmaRMSNorm (`1 + w`) \
                 over ATen's reduction with a non-contracting rope, and the drafter's norm is \
                 The reference server's plain `RMSNorm` over Triton's. F28 measured the difference as 44,612 of \
                 65,536 bytes at `chain.attn.l0.q`",
            );
            push(
                p("attn.kv_append"),
                Coverage::Dflash(DflashOp::KvAppendFp8),
                Some({
                    let mut s = DflashShape::new(m, cfg.kv_heads_local(), cfg.head_dim);
                    // k is the rope's contiguous output; **v is a view of the `[q | k | v]`
                    // projection**, so its source rows are `qkv_n` apart and not `nkv * head_dim`.
                    s.kv_src_stride = (cfg.kv_heads_local() * cfg.head_dim, cfg.qkv_n());
                    s
                }),
                1,
                "the drafter's OWN cache, in the reference server's flash layout: separate k and v tensors, and a \
                 v whose source rows are the PROJECTION's",
            );
            push(
                p("attn.attention"),
                Coverage::Dflash(DflashOp::AttnFp8Kv),
                Some(DflashShape::rerank(
                    m,
                    cfg.q_heads_local(),
                    0,
                    cfg.kv_heads_local(),
                )),
                1,
                "non-causal with a SYMMETRIC 2048 window; k is the sequence length at run time",
            );
            // The second undeclared quant, for the same reason and at a different place: the
            // attention writes `bf16` and `o_proj` is a blockscale GEMM, and the conv `prepare`
            // that feeds every other one is on the OTHER side of the attention.
            push(
                p("attn.o_proj.quant"),
                Coverage::Dflash(DflashOp::ActQuantRecip),
                Some(DflashShape::new(
                    m,
                    cfg.q_heads_local() * cfg.head_dim,
                    cfg.q_heads_local() * cfg.head_dim,
                )),
                1,
                "the same SOURCE LINE as `aux_hidden_projection.quant` and NOT the same \
                 program: this one is INSIDE the drafter's compiled region, where inductor \
                 strength-reduces `absmax / 448` into `absmax * 0.002232142857142857` (`_0` of \
                 prod's own generated code), while `model.fc`'s is eager and divides. Prod \
                 quantises inside `o_proj`'s quant method, and the conv `prepare` that feeds \
                 every other blockscale GEMM is on the other side of the attention",
            );
            push(
                p("attn.o_proj"),
                Coverage::Dflash(DflashOp::GemmBlockscale),
                Some(DflashShape::new(m, h, cfg.q_heads_local() * cfg.head_dim)),
                1,
                "row-parallel, so an all-reduce follows",
            );
            push(
                p("attn.all_reduce"),
                Coverage::Other("saber_ar_exact"),
                None,
                1,
                "twelve of the step's fifteen go through the reference's exact kernel; three go to RCCL",
            );
            push(
                p("attention_conv.finish+post_attention_layernorm"),
                Coverage::Dflash(DflashOp::ConvAddRmsOpen),
                Some(DflashShape::conv(
                    m,
                    h,
                    h / cfg.conv_group,
                    cfg.block_size,
                    1,
                )),
                1,
                "the conv and the norm that follows it, in one kernel -- which is how inductor \
                 already runs them -- and the half that OPENS the layer's fp32 residual",
            );
            push(
                p("mlp.gate_up"),
                Coverage::Dflash(DflashOp::GemmBlockscale),
                Some(DflashShape::new(m, 2 * cfg.intermediate / cfg.tp, h)),
                1,
                "gate and up merged, as prod launches them",
            );
            push(
                p("mlp.silu_quant"),
                Coverage::Dflash(DflashOp::SiluQuant),
                // `k` is the INPUT width: gate and up concatenated, so twice the output.
                Some(DflashShape::new(
                    m,
                    cfg.intermediate / cfg.tp,
                    2 * cfg.intermediate / cfg.tp,
                )),
                1,
                "the blockscale GEMM's activation needs GROUP scales, not a token scale",
            );
            push(
                p("mlp.down"),
                Coverage::Dflash(DflashOp::GemmBlockscale),
                Some(DflashShape::new(m, h, cfg.intermediate / cfg.tp)),
                1,
                "row-parallel",
            );
            push(
                p("mlp.all_reduce"),
                Coverage::Other("saber_ar_exact"),
                None,
                1,
                "the second of the layer's two reductions",
            );
            push(
                p("mlp_conv.finish+next_layernorm"),
                Coverage::Dflash(DflashOp::ConvAddRmsClose),
                Some(DflashShape::conv(
                    m,
                    h,
                    h / cfg.conv_group,
                    cfg.block_size,
                    1,
                )),
                1,
                "the conv and the NEXT layer's input norm -- or, in the last layer, the drafter's \
                 own final `hidden_norm`",
            );
        }

        // ---- the head and the selector, once per step -------------------------------------------
        // THE TWO ROW GATHERS `_generate_draft` MAKES, between the drafter's forward and its head
        // (`dflash2/speculator.py:196-206`):
        //
        //     hidden_states  = last_hidden_states[self.sample_indices[:num_sample]]
        //     anchor_token_ids = self.input_buffers.input_ids[self._anchor_indices[:num_reqs]]
        //
        // The first narrows the drafter's `block_size` query rows to the `selector_positions()`
        // the head scores -- which is why every op above runs at 8 rows and every op below runs
        // at 7 -- and the second takes the one token per request the selector anchors on. A
        // STAGED replay cannot see either: it starts from `head.hidden`, which is the first
        // one's OUTPUT. One `DraftOp` at two shapes, because a row gather has a width and no
        // dtype.
        push(
            "head.sample_rows".into(),
            Coverage::Dflash(DflashOp::RowGather),
            Some(DflashShape::new(
                cfg.selector_positions(),
                h * 2,
                cfg.block_size,
            )),
            1,
            "`last_hidden[sample_indices]`: the drafter's query rows narrowed to the positions \
             the head scores, which is where the plan's row count changes from 8 to 7",
        );
        push(
            "selector.anchor_ids".into(),
            Coverage::Dflash(DflashOp::RowGather),
            Some(DflashShape::new(1, 4, cfg.block_size)),
            1,
            "`input_ids[_anchor_indices]`: one i32 token per request, the selector's anchor. The \
             same kernel at a row of four bytes",
        );
        // THE HEAD'S OWN INPUT PREP, and the plan called both of these INPUTS until F27 chained
        // the backbone. `_apply_head_int2` runs
        //
        //     M = _pow2_at_least(m); x = torch.cat([x, x.new_zeros(M - m, k)])
        //     xs = x.reshape(M, ng, GROUP).float().sum(-1).contiguous()
        //
        // before it launches the head, and both lines are launches prod makes. They are declared
        // here at ONE launch each and the notes say what that absorbs, because prod's ATen issues
        // four: a fill and a concatenation for the pad, an fp32 cast and a reduction for the sum.
        // The same absorption `head.int2`'s `_topk` variant already makes of prod's `fill_`, and
        // it is stated rather than left to be counted.
        push(
            "head.row_pad".into(),
            Coverage::Dflash(DflashOp::HeadRowPad),
            Some(DflashShape::rerank(m.max(16), h, h, m)),
            1,
            "`_pow2_at_least`'s pad: `torch.cat([x, x.new_zeros(M - m, k)])` at the drafter's \
             draft rows, in one kernel that writes the live rows and the fill",
        );
        push(
            "head.group_sum".into(),
            Coverage::Dflash(DflashOp::HeadGroupSum),
            Some(DflashShape::new(m.max(16), h, h)),
            1,
            "`xs`, the per-128-group fp32 column sum the int2 head reads as a kernarg -- one \
             kernel over the bf16 input, absorbing the fp32 cast prod materialises first",
        );
        push(
            "head.int2".into(),
            Coverage::Dflash(DflashOp::DraftHeadInt2),
            Some(DflashShape::new(m.max(16), cfg.vocab_local(), h)),
            1,
            "M is padded to 16 by _pow2_at_least; the _topk variant stores -inf, absorbing the \
             fill_ prod runs after it",
        );
        // TWO launches, and they are two SITES here rather than one site of two, because they
        // are two different code objects with two different shapes of work -- a select and a
        // gather -- and a plan that named one of them would be describing half a launch. ATen's
        // own count for this selection is also two (`sbtopk::gatherTopK`), so the declared total
        // is unchanged by covering it.
        let topk_rows = m.max(16);
        let topk_n = cfg.head_block_maxima();
        push(
            "head.candidate_topk.select".into(),
            Coverage::Dflash(DflashOp::HeadTopkThresh),
            Some(DflashShape::rerank(topk_rows, topk_n, h, cfg.rerank)),
            1,
            "`bm.topk(RERANK)` over the block maxima the head emitted, pass one: the radix \
             select for the row's k-th largest (value, ~index) composite",
        );
        push(
            "head.candidate_topk.gather".into(),
            Coverage::Dflash(DflashOp::HeadTopkGather),
            Some(DflashShape::rerank(topk_rows, topk_n, h, cfg.rerank)),
            1,
            "... and pass two, the gather and the rank. The composite key is a TOTAL order, so \
             the tie rule the LEDGER could not pin for ATen is not a question at this site",
        );
        push(
            "head.candidate_topk.gather_ids".into(),
            Coverage::Dflash(DflashOp::HeadCandGather),
            Some(DflashShape::rerank(topk_rows, topk_n, h, cfg.rerank)),
            1,
            "`bi.gather(1, ...)`: the selection returns POSITIONS in the block-maxima row and the \
             rerank scores VOCABULARY COLUMNS -- this is the map. The third launch of the head's \
             selection, and a staged replay cannot see it because it seeds `idx` from prod's own",
        );
        push(
            "head.rerank".into(),
            // `_rerank_exact`'s FP8=False arm, because the live `lm_head` is bf16: see the
            // `head.rerank.w` record below and `DflashOp::RerankExactBf16`.
            Coverage::Dflash(DflashOp::RerankExactBf16),
            Some(DflashShape::rerank(
                m.max(16),
                cfg.vocab_local(),
                h,
                cfg.rerank,
            )),
            1,
            "the _scatter variant writes y at the candidate column, absorbing prod's scatter_",
        );
        push(
            "selector.candidate_topk".into(),
            Coverage::Dflash(DflashOp::SelectorTopk),
            Some(DflashShape::rerank(
                cfg.selector_positions(),
                cfg.vocab_local(),
                h,
                cfg.selector_top_k,
            )),
            1,
            "`torch.topk(logits, 16)` over the vocab shard -- the candidates the selector scores \
             edges between. The kernel and its rules row have existed since the census; what the \
             plan was missing is that `_ref_topk_only` fills every entry the rerank did not \
             touch with -inf, so the 16 come out of 80 finite values and ATen's tie order among \
             the rest is not reachable",
        );
        // THE CANDIDATE SEAM, at TP > 1. `selector.candidate_topk` above scores this rank's
        // vocabulary SHARD, so its 16 are the best of half a row. Prod's `get_top_k_tokens`
        // (`ref/patch_dflash2.py`, edit 4) makes the ids global, ALL-GATHERS the `k`
        // values and the `k` ids and takes the top-k of the concatenation -- four launches,
        // "O(batch * 2k * tp_size) rather than O(batch * vocab_size)". Saber SUPPLIES all four on
        // the host, over the rendezvous socket, exactly as it supplies `logits.gather`: 896 bytes
        // a proposal, and the merge is `merge_candidates`. Declared as one op because it is one
        // seam and no code object implements any of it.
        push(
            "selector.candidate_seam".into(),
            // `Other` and not `Absent`, for `logits.gather`'s own reason: the seam has an
            // implementation and it is not a code object. `saber_exec::layer` calls that
            // `Route::Supplied` -- "no code object implements this op; the caller must produce
            // the edge" -- and a plan that said ABSENT here would be claiming a proposal cannot
            // run at two ranks, which is the opposite of what ws-s6 measured.
            Coverage::Other("host: saber_engine::device::merge_candidates"),
            None,
            4,
            "prod's two `tensor_model_parallel_all_gather`s, the second `topk` and the id gather. \
             At TP=1 it does not exist at all -- the shard IS the row -- which is why the TP=1 \
             proposal is 102 launches and not 106",
        );
        push(
            "selector.hidden_projection".into(),
            Coverage::Dflash(DflashOp::BgemmBf16),
            Some(DflashShape::new(m, cfg.selector_rank, h)),
            1,
            "bf16, in modules_to_not_convert like kernel_projection",
        );
        push(
            "selector.score_edges".into(),
            Coverage::Dflash(DflashOp::SelectorScore),
            Some(DflashShape::selector(
                cfg.selector_positions(),
                cfg.selector_positions(),
                cfg.selector_rank,
                cfg.selector_top_k,
            )),
            1,
            "both codebook gathers, the broadcast multiply, the 16x16x256 einsum and the bias add",
        );
        push(
            "selector.walk".into(),
            Coverage::Dflash(DflashOp::SelectorWalk),
            // `k` is the score matrix's inner width, which is the top-k again: each step reads a
            // [top_k, top_k] block and takes the row the previous winner names.
            Some(DflashShape::rerank(
                1,
                cfg.block_size - 1,
                cfg.selector_top_k,
                cfg.selector_top_k,
            )),
            1,
            "one program per request, one wave: `num_steps` sequential argmaxes, each over the \
             score row the previous step's winner names",
        );

        DraftGraph {
            cfg,
            ops,
            rerank_head: RerankHead::Bf16,
        }
    }

    /// The same graph with its exact rerank reading `head` (N6): the `head.rerank` op's coverage
    /// becomes that kind's op and [`DraftGraph::records`] declares that kind's records. Nothing
    /// else moves — the int2 head is quantised from prod's bf16 matrix either way.
    #[must_use]
    pub fn with_rerank_head(mut self, head: RerankHead) -> Self {
        for o in &mut self.ops {
            if o.name == "head.rerank" {
                o.coverage = Coverage::Dflash(head.op());
            }
        }
        self.rerank_head = head;
        self
    }

    /// Launches a decode step makes that a Saber code object covers. This is the number a launch
    /// census is compared against, and it is DECLARED here rather than counted from a trace.
    #[must_use]
    pub fn budget(&self) -> u32 {
        self.ops
            .iter()
            .filter(|o| matches!(o.coverage, Coverage::Dflash(_)))
            .map(|o| o.count)
            .sum()
    }

    /// Every launch, covered or not.
    #[must_use]
    pub fn total_launches(&self) -> u32 {
        self.ops.iter().map(|o| o.count).sum()
    }

    /// Covered launches per `dflash` op id.
    #[must_use]
    pub fn by_op(&self) -> BTreeMap<&'static str, u32> {
        let mut m = BTreeMap::new();
        for o in &self.ops {
            if let Coverage::Dflash(op) = o.coverage {
                *m.entry(op.as_str()).or_insert(0) += o.count;
            }
        }
        m
    }

    /// What no code object covers, with the reason, in forward order.
    #[must_use]
    pub fn uncovered(&self) -> Vec<(&str, &'static str, u32)> {
        self.ops
            .iter()
            .filter_map(|o| match o.coverage {
                Coverage::Absent(why) => Some((o.name.as_str(), why, o.count)),
                _ => None,
            })
            .collect()
    }

    /// Resolve every covered op through the kernel plane, and return the ones it REFUSES.
    ///
    /// This is the check that the description and the rules agree: a shape this module declares
    /// that `dflash::resolve` will not launch is a description of a program that cannot run.
    ///
    /// # Errors
    /// Never — refusals come back in the vector rather than as an error, so a caller sees all of
    /// them at once.
    #[must_use]
    pub fn refusals(&self, rules: &DflashRules) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for o in &self.ops {
            if let (Coverage::Dflash(op), Some(shape)) = (o.coverage.clone(), o.shape) {
                // The attention's `k` is the sequence length and its `bt_stride` is the step's own
                // block-table padding: both are run-time values rather than shapes of the model,
                // so this asks the question at a REPRESENTATIVE step instead of pretending the
                // plan knows them. A zero `bt_stride` is refused by name (round 22), which is the
                // point — a caller that has not been told the stride cannot launch — so the
                // substitution here is what keeps that refusal from reading as a plan defect.
                let shape = if matches!(op, DflashOp::AttnBf16Kv | DflashOp::AttnFp8Kv) {
                    DflashShape::attn(shape.m, shape.n, 4096, shape.r, 0, 1)
                } else {
                    shape
                };
                if let Err(e) = saber_kernels::dflash::resolve(op, shape, rules) {
                    out.push((o.name.clone(), e.to_string()));
                }
            }
        }
        out
    }
}

impl fmt::Display for DraftGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "DFlash2 drafter, tp {}, M = {} query rows: {} declared launches ({} covered)",
            self.cfg.tp,
            self.cfg.block_size,
            self.total_launches(),
            self.budget()
        )?;
        for (op, n) in self.by_op() {
            writeln!(f, "  {op:<22} {n:>4}")?;
        }
        let un = self.uncovered();
        if !un.is_empty() {
            writeln!(f, "  not a code object yet:")?;
            for (name, why, n) in un {
                writeln!(f, "  {name:<34} x{n:<3} {why}")?;
            }
        }
        Ok(())
    }
}

/// Which of `_score_edges`'s two hidden bf16 roundings an arm of [`score_edges_reference`] keeps.
///
/// `unary_logits[:, :, None] + torch.einsum("blpr,blcr->blpc", predecessors * hidden[:, :, None],
/// successors)` is one line of Python over `bf16` operands and it hides both: the elementwise
/// product is a real `bf16` tensor, and `torch.einsum` over two `bf16` tensors **returns `bf16`**,
/// so the 256-term dot is rounded before the `f32` bias is added. The `f32` output tensor is the
/// last add's dtype and says nothing about either.
///
/// Measured against prod's own captured `walk.scores` (F23, three prompts, host, no device):
/// [`Prod`](ScoreEdgesArm::Prod) is identical of 7168 bytes; [`F32Mul`](ScoreEdgesArm::F32Mul)
/// differs in 775 / 588 / 635 and [`F32`](ScoreEdgesArm::F32) in 3694 / 3725 / 3804. Both are
/// CONTROLS that must fail — a rounding this reference calls load-bearing and which turns out not
/// to be would mean the kernel is free to drop it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScoreEdgesArm {
    /// Prod's own: bf16 product, f32 accumulation, **bf16 round of the dot**, f32 bias add.
    Prod,
    /// Control 1 — the elementwise product formed in f32. The dot is still rounded.
    F32Mul,
    /// Control 2 — f32 throughout, which is what an f32-accumulating kernel writes.
    F32,
}

/// `_score_edges` on the host, gathers included, at prod's own element widths.
///
/// `pred_cb` and `succ_cb` are `[vocab, rank]` bf16 codebooks as raw `u16`; `cand` is
/// `torch.topk`'s **`i64`** indices `[positions, top_k]`; `anchor` is `input_buffers.input_ids`,
/// which is **`i32`** (`torch.cat` promotes it against the i64 candidates); `hidden` is the
/// projection's `[positions, rank]` bf16 output; `unary` is the exact rerank's **`f32`**
/// `[positions, top_k]`. The answer is `[positions, top_k, top_k]` **`f32`**, indexed `[p][c]`
/// with the bias on `c` — `unary_logits[:, :, None]` broadcasts along the PREDECESSOR axis, so the
/// unary score belongs to the token the edge arrives at.
///
/// **Those widths are why this function exists**, exactly as `selector_walk_greedy`'s `&[i64]` is.
/// `dflash_selector_score_r256_k16` read the candidates as `i32`, the bias as `bf16` and wrote
/// `bf16` until 2026-09-21, and was gated bit-exact against a capture built at those same widths
/// while differing in 5007 / 5377 / 5133 of 7168 bytes against prod's own proposal.
///
/// The accumulation ORDER is free: 256 f32 partial sums agree far inside a bf16 ULP, and prod's
/// expression written out with an explicit round is bit-identical to prod whatever the order. So
/// this folds left to right and makes no claim about a kernel's k-order.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn score_edges_reference(
    pred_cb: &[u16],
    succ_cb: &[u16],
    cand: &[i64],
    anchor: &[i32],
    hidden: &[u16],
    unary: &[f32],
    positions: usize,
    l_len: usize,
    top_k: usize,
    rank: usize,
    arm: ScoreEdgesArm,
) -> Vec<f32> {
    let row = |cb: &[u16], id: i64, r: usize| -> f32 {
        let o = (id.max(0) as usize) * rank + r;
        cb.get(o).map_or(0.0, |v| bf16_bits_to_f32(*v))
    };
    let mut out = vec![0f32; positions * top_k * top_k];
    for pos in 0..positions {
        let l = if l_len == 0 { 0 } else { pos % l_len };
        for p in 0..top_k {
            // `predecessor_ids` is the anchor at l == 0 and the PREVIOUS position's candidates
            // otherwise -- prod's `cat(anchor.expand(...), candidate_ids[:, :-1])`.
            let pid = if l == 0 {
                i64::from(anchor.get(pos / l_len.max(1)).copied().unwrap_or(0))
            } else {
                cand.get((pos - 1) * top_k + p).copied().unwrap_or(0)
            };
            for c in 0..top_k {
                let cid = cand.get(pos * top_k + c).copied().unwrap_or(0);
                let mut dot = 0f32;
                for r in 0..rank {
                    let h = hidden
                        .get(pos * rank + r)
                        .map_or(0.0, |v| bf16_bits_to_f32(*v));
                    let a = row(pred_cb, pid, r) * h;
                    let a = if arm == ScoreEdgesArm::F32Mul {
                        a
                    } else {
                        bf16_bits_to_f32(f32_to_bf16_bits(a))
                    };
                    dot += a * row(succ_cb, cid, r);
                }
                if arm != ScoreEdgesArm::F32 {
                    dot = bf16_bits_to_f32(f32_to_bf16_bits(dot));
                }
                out[(pos * top_k + p) * top_k + c] =
                    dot + unary.get(pos * top_k + c).copied().unwrap_or(0.0);
            }
        }
    }
    out
}

/// bf16 bits -> f32: the low sixteen mantissa bits are zero, so this is exact and lossless.
#[must_use]
pub fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// f32 -> bf16 bits, round-to-nearest-EVEN, which is what torch's conversion does.
///
/// A NaN stays a NaN rather than becoming an infinity, which the bias trick alone does not
/// guarantee.
#[must_use]
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7FC0;
    }
    let u = x.to_bits();
    ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16
}

/// The selector's path walk, on the host, at prod's own dtypes.
///
/// `dflash2/speculator.py::_selector_walk_kernel` at `draft_sample_method: greedy` — the two
/// constexprs that fall away are `SAMPLE_PROBABILISTIC` (no `draft_logits`) and `temperature`
/// (`gumbel_noised_argmax` at `temp == 0` is a plain argmax whose ties go to the LOWEST index).
/// What is left is seven sequential steps, each reading the score row the previous step's winner
/// names.
///
/// **The element widths are the point of this function existing.** `candidate_ids` are
/// `torch.topk`'s indices, which ATen types **`i64`**, and the kernel stores `tokens` at the same
/// width. `req_state` is `i32`. `dflash_selector_walk_greedy` read the candidates at four bytes
/// until 2026-09-21 and was bit-exact on every score and every argmax while storing the wrong
/// token at every step — a failure only a width-aware reference can catch, which is why this one
/// takes `&[i64]` and its test puts distinct ids in the ODD slots.
///
/// Returns `(tokens, realized)`: `[reqs * steps]` and `[reqs * steps * top_k]`. An inactive
/// request (`req_state[row * steps] < 0`) writes neither, exactly as the reference's `mask & valid`
/// suppresses both stores.
#[must_use]
pub fn selector_walk_greedy(
    scores: &[f32],
    candidates: &[i64],
    req_state: &[i32],
    reqs: usize,
    steps: usize,
    top_k: usize,
) -> (Vec<i64>, Vec<f32>) {
    let mut tokens = vec![0i64; reqs * steps];
    let mut realized = vec![0f32; reqs * steps * top_k];
    for row in 0..reqs {
        // `valid` is read ONCE, from step 0's slot: it is a property of the request, not of the
        // step, and it gates every store.
        if req_state.get(row * steps).copied().unwrap_or(-1) < 0 {
            continue;
        }
        let mut previous = 0usize;
        for step in 0..steps {
            let flat = row * steps + step;
            let base = (flat * top_k + previous) * top_k;
            let mut best = (f32::NEG_INFINITY, 0usize);
            for i in 0..top_k {
                let s = scores.get(base + i).copied().unwrap_or(f32::NEG_INFINITY);
                realized[flat * top_k + i] = s;
                // Strictly greater wins; an exact tie goes to the LOWER index, which is what
                // Triton's `tl.max(..., return_indices=True)` does.
                if s > best.0 {
                    best = (s, i);
                }
            }
            tokens[flat] = candidates
                .get(flat * top_k + best.1)
                .copied()
                .unwrap_or_default();
            previous = best.1;
        }
    }
    (tokens, realized)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g() -> DraftGraph {
        DraftGraph::build(DflashConfig::qwen38_dflash2(2, 7))
    }

    /// The config's derived widths are the ones the census read out of prod's own kernel names.
    #[test]
    fn the_derived_widths_are_the_traced_ones() {
        let c = DflashConfig::qwen38_dflash2(2, 7);
        assert_eq!(c.block_size, 8, "1 + num_speculative_tokens");
        assert_eq!(c.qkv_n(), 3072, "the census's qkv shape");
        assert_eq!(c.kernel_projection_n(), 1280, "2 x taps x 320 conv groups");
        assert_eq!(c.q_heads_local() * c.head_dim, 2048, "o_proj's K");
        assert_eq!(2 * c.intermediate / c.tp, 17408, "gate_up's N");
        assert_eq!(c.intermediate / c.tp, 8704, "down's K");
        assert_eq!(c.target_layers * c.hidden, 25600, "the aux projection's K");
        assert_eq!(c.vocab_local(), 124_160, "the head's N");
        assert_eq!(
            c.selector_positions(),
            7,
            "the mask positions, not the block"
        );
        assert_eq!(c.ctx_rows(), 8, "one verified block at batch 1");
        assert_eq!(
            c.context_kv_n(),
            5120,
            "L x 2 x kv_size: every layer's K and V in one projection"
        );
    }

    /// Every covered op's shape must be one the kernel plane will actually launch. A description
    /// the rules refuse is a description of a program that cannot run.
    #[test]
    fn every_declared_shape_resolves() {
        let rules = DflashRules::prod_dflash().expect("the committed rules");
        let refusals = g().refusals(&rules);
        assert!(refusals.is_empty(), "the rules refuse: {refusals:?}");
    }

    /// The budget is declared, and the per-op counts are the ones the census traced. These are the
    /// numbers `saber draft-step` compares a launch census against.
    #[test]
    fn the_budget_matches_the_census_per_op() {
        let g = g();
        let by = g.by_op();
        // 21 blockscale GEMMs: qkv, o_proj, gate_up and down once per layer, plus the aux
        // projection once -- exactly the census's 21 launches over five shapes.
        assert_eq!(by["gemm_a8w8_blockscale"], 21);
        // kernel_projection twice per layer plus hidden_projection once = the census's x10 + x1,
        // AND the context store's fused KV projection, which is the twelfth. It was excluded for
        // a round because `bgemm_bf16`'s envelope stopped at N = 2560 while that launch is 10240
        // on one rank; the kernel and the reference were never in question, only the N at which
        // hipBLASLt changes its answer.
        assert_eq!(by["bgemm_bf16"], 12);
        // Four conv sites per layer: two `prepare` (quantised) and two `finish` -- and a `finish`
        // carries the norm that follows it, because that is how inductor already runs the pair.
        assert_eq!(by["grouped_conv_quant"], 10);
        // FIVE AND FIVE, not ten: the two halves are two ops because prod carries the residual
        // between them as fp32 and rounds it to bf16 only where its compiled region ends.
        assert_eq!(
            by["conv_add_rms_open"], 5,
            "one attention-side norm per layer"
        );
        assert_eq!(by["conv_add_rms_close"], 5, "one MLP-side norm per layer");
        assert_eq!(by["silu_quant"], 5, "one MLP activation per layer");
        assert_eq!(by["attn_fp8kv"], 5);
        // Five for the drafter's own query rows and five for the context store's per-layer
        // insert: the same `reshape_and_cache_flash`, at two row counts.
        assert_eq!(by["kv_append_fp8"], 10);
        // The context store's two norms -- the hidden norm at one group and the stacked K norm at
        // L -- and its one batched rope.
        assert_eq!(by["ctx_rms_norm"], 2);
        assert_eq!(by["ctx_rope"], 1);
        assert_eq!(by["draft_head_int2"], 1);
        // The bf16 arm: `SABER_NVFP4_LMHEAD=bf16` dequantises `lm_head` at load, so
        // `_head_matrix` returns a matrix with no scale and `_rerank_exact` takes `FP8=False`.
        assert_eq!(by["rerank_exact_bf16"], 1);
        assert!(
            !by.contains_key("rerank_exact"),
            "the e4m3 arm is not what this serve runs"
        );
        assert_eq!(by["selector_score"], 1);
        assert_eq!(by["selector_walk"], 1, "one program per request");
        assert_eq!(
            by["rms_norm"], 1,
            "layer 0 is the only layer whose residual is None"
        );
        // THE SIX LAUNCHES NOBODY DECLARED, now TWO ops: the activation quant in front of
        // `aux_hidden_projection` and in front of each layer's `attn.o_proj`. Every other
        // blockscale GEMM in the drafter is fed by a conv `prepare` or by `silu_quant`. One
        // source line, two programs -- `model.fc` is outside the compiled region and divides,
        // the five `o_proj`s are inside it and multiply by the reciprocal -- and `DflashRules`
        // binds one symbol per op, so the split is an op and not a plan field.
        assert_eq!(by["act_quant"], 1, "only `model.fc`'s is eager");
        assert_eq!(by["act_quant_recip"], 5, "one per layer, inside the region");
        // The int2 head's own selection, two passes -- which is also ATen's own launch count for
        // it -- and the selector's top-16 over the vocab shard.
        assert_eq!(by["head_topk_thresh"], 1);
        assert_eq!(by["head_topk_gather"], 1);
        assert_eq!(by["selector_topk"], 1);
        // The context store's `permute(...).contiguous()`, which computes nothing and is still a
        // launch: nothing else stands between the projection and the K norm.
        assert_eq!(by["ctx_kv_layout"], 1);
        // The int2 head's OWN input prep, which the plan treated as inputs for six rounds: the
        // row pad and the per-group column sum. One launch each, and the notes on both say what
        // that absorbs of prod's four ATen launches.
        assert_eq!(by["head_row_pad"], 1);
        assert_eq!(by["head_group_sum"], 1);
        assert_eq!(by["head_cand_gather"], 1);
        assert_eq!(by["row_gather"], 2);
        // The drafter's OWN q/k norm + rope, one a layer. It counted as another plugin's through
        // F27 -- `attn_glue::qk_norm_rope`, which is the TARGET's GemmaRMSNorm over ATen's
        // reduction with a rope that does not contract -- and F28 measured what that cost:
        // 44,612 of 65,536 bytes at the first layer's post-rope q, on every prompt.
        assert_eq!(by["qk_norm_rope_draft"], 5);
        assert_eq!(g.budget(), 101, "the declared covered-launch budget");
        // 112, not 107, not 101, not 97. The context store was six launches and is ten -- the
        // hidden norm, the fused KV projection, the layout copy and the five per-layer cache
        // inserts, against five k-norms that are really one. Then round 21's replay found six
        // more: an ACTIVATION QUANT in front of every blockscale GEMM whose producer is not a
        // conv `prepare` or a `silu_quant`, which is `aux_hidden_projection` and every layer's
        // `attn.o_proj`. F27's chain of the BACKBONE found the last five, all of them in the
        // seam between the drafter's forward and its head and none of them reachable from a
        // staged replay, which seeds that seam from prod's own tensors: two row gathers
        // (`last_hidden[sample_indices]`, `input_ids[_anchor_indices]`), `_pow2_at_least`'s row
        // pad, `xs`, and the `bi` gather between the top-80's positions and the rerank's
        // vocabulary columns. The number has now got worse four times running and every time it
        // got RIGHTER, which is the only direction a declared budget may move on evidence.
        // ws-s6 added two: the EMBEDDING's own all-reduce (the drafter shares the target's
        // `VocabParallelEmbedding` and it reduces at `tp_size > 1`) and the four-launch candidate
        // seam prod runs at the head, which Saber supplies on the host. Both are TP > 1 launches
        // and both are declared at every TP, exactly as the ten layer reduces always were.
        assert_eq!(g.total_launches(), 101 + 1 + 5 + 5 + 1 + 4);
        // 101 covered by a `dflash` code object, 11 by another plugin (the head's embedding and
        // the ten all-reduces), and NOTHING absent. The budget went 80 -> 91 -> 96 -> 101; the
        // first move covered launches the plan already declared, the second DECLARED five more
        // and covered them in the same round, and the third took the five q/k norm + rope
        // launches off `attn_glue` -- which was the wrong plugin and not a missing one.
        assert_eq!(
            g.total_launches() - g.budget(),
            16,
            "the ops another plugin covers or the host supplies, and no others"
        );
    }

    /// **There are no gaps.** Every launch this plan declares has a code object.
    ///
    /// This test is the one that changed shape in F26, and the history is the point of keeping it
    /// rather than deleting it. It used to assert that ELEVEN launches were absent and that each
    /// was named with what it would need, because a description that quietly omitted them would
    /// have made the budget agree with the launcher and disagree with prod. The list was:
    ///
    ///   `aux_hidden_projection.quant` x1 and `layer.{0..4}.attn.o_proj.quant` x5 -- one kernel
    ///       at two shapes, the blockscale GEMM's own `bf16 -> e4m3` + per-128-group scales, which
    ///       prod runs inside the layer's quant method where no launch census sees it;
    ///   `context_kv.kv_projection` x1 -- `bgemm_bf16` at an N outside the gated envelope;
    ///   `context_kv.layout` x1 -- a `.contiguous()` that computes nothing;
    ///   `head.candidate_topk` x2 and `selector.candidate_topk` x1 -- ATen selections whose tie
    ///       order the LEDGER could not state as a rule.
    ///
    /// What it asserts now is the same claim from the other side, and it is not vacuous: the
    /// uncovered set is empty AND the budget is the whole of what `dflash` covers AND every op
    /// still carries a shape the committed rules resolve (`every_declared_shape_resolves`). A
    /// coverage added without a rules row fails there; a coverage added without a launch fails
    /// here, because the total would move.
    #[test]
    fn nothing_is_absent_and_the_eleven_that_were_are_named_where_they_went() {
        let graph = g();
        assert!(
            graph.uncovered().is_empty(),
            "still absent: {:?}",
            graph.uncovered()
        );
        // The eleven, by the op that covers each of them now. Named individually rather than
        // counted, so a coverage silently moved to the wrong op is a failure here.
        let by_name: std::collections::BTreeMap<&str, Coverage> = graph
            .ops
            .iter()
            .map(|o| (o.name.as_str(), o.coverage.clone()))
            .collect();
        assert_eq!(
            by_name["aux_hidden_projection.quant"],
            Coverage::Dflash(DflashOp::ActQuant),
            "`model.fc` is eager and its `absmax / 448` is a real division"
        );
        for name in ["layer.0.attn.o_proj.quant", "layer.4.attn.o_proj.quant"] {
            assert_eq!(
                by_name[name],
                Coverage::Dflash(DflashOp::ActQuantRecip),
                "{name} is inside the compiled region, where the divisor is strength-reduced"
            );
        }
        assert_eq!(
            by_name["context_kv.kv_projection"],
            Coverage::Dflash(DflashOp::BgemmBf16)
        );
        assert_eq!(
            by_name["context_kv.layout"],
            Coverage::Dflash(DflashOp::CtxKvLayout)
        );
        assert_eq!(
            by_name["head.candidate_topk.select"],
            Coverage::Dflash(DflashOp::HeadTopkThresh)
        );
        assert_eq!(
            by_name["head.candidate_topk.gather"],
            Coverage::Dflash(DflashOp::HeadTopkGather)
        );
        assert_eq!(
            by_name["selector.candidate_topk"],
            Coverage::Dflash(DflashOp::SelectorTopk)
        );
        // The ops another plugin covers are still `Other` and still named -- covering the eleven
        // must not have quietly re-labelled an all-reduce or the head's embedding.
        let other = graph
            .ops
            .iter()
            .filter(|o| matches!(o.coverage, Coverage::Other(_)))
            .map(|o| o.count)
            .sum::<u32>();
        assert_eq!(
            by_name["layer.0.attn.qk_norm_rope"],
            Coverage::Dflash(DflashOp::QkNormRope),
            "the drafter's q/k norm + rope is `dflash_qk`, not `attn_glue`: F28"
        );
        assert_eq!(
            other, 16,
            "embedding + 11 all_reduce + the 4-launch candidate seam"
        );
    }
}

// =====================================================================================
// The record set a DFlash2 drafter pack holds
// =====================================================================================
//
// [`DraftGraph`] above says which LAUNCHES a decode step makes. This section says which BYTES those
// launches read, which is the other half of the same description and the thing that was missing
// when `saber draft-step --proposal` could run three launches and not the other ninety-three:
// every launch after the context store reads a drafter weight and there was no record map.
//
// Three rules make the set, and each one exists because the alternative was a silent mismatch:
//
//  1. **A record is named after the OP that reads it, plus the kernarg role** -- `layer.3.mlp.
//     gate_up.w`, not `layers.3.mlp.gate_proj.weight`. A merged projection is no longer any one
//     checkpoint tensor, and the launcher binds by the op's own name.
//  2. **The layout id names the transform AND the consumer.** The drafter's fp8 B is preshuffled;
//     the same bytes in checkpoint order compute a wrong answer and nothing about the dtype or the
//     length says which order they are in (`docs/abi/PACK.md` § Layout ids, 7..13).
//  3. **Two checkpoints feed one drafter pack.** The DFlash2 checkpoint has 117 tensors and none
//     of them is an embedding or a head: `qwen3_dflash2.py` starts its stack from the token
//     embedding and scores against `lm_head`, both of which the reference server hands it from the TARGET, and the
//     int2 draft head is quantized from the latter at load. A record map that named only the
//     drafter's own file would be missing 2.1 GiB of what a proposal reads.

use crate::graph::{
    Dtype, ShardKind, LAYOUT_BF16_FROM_FP8_BLOCK, LAYOUT_BF16_FROM_FP8_CHANNEL, LAYOUT_BF16_MERGED,
    LAYOUT_F32_CHANNEL_SCALE, LAYOUT_FP8_BLOCK_SCALE, LAYOUT_FP8_CHANNEL, LAYOUT_FP8_PRESHUFFLE,
    LAYOUT_INT2_HEAD, LAYOUT_INT2_HEAD_SCALE, LAYOUT_INT2_HEAD_ZERO, LAYOUT_RAW,
    LAYOUT_ROPE_COS_SIN,
};

/// Which checkpoint a source tensor lives in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Checkpoint {
    /// `Qwen3.8-27B-DFlash2-FP8/model.safetensors`.
    Drafter,
    /// The target tier's own checkpoint (`Qwen3.8-27B-NVFP4`).
    Target,
}

impl Checkpoint {
    /// The string the plan JSON uses.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Checkpoint::Drafter => "drafter",
            Checkpoint::Target => "target",
        }
    }
}

/// How a source tensor is quantized in its checkpoint — what the builder has to undo.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftQuant {
    /// `e4m3` weight × a `[N/128, K/128]` `bf16` `weight_scale_inv`, which is how the DFlash2
    /// checkpoint stores every linear it converted (`quantization_config.weight_block_size`).
    Fp8Block,
    /// `e4m3` weight × a `[N, 1]` per-output-channel scale: the target's `lm_head`.
    Fp8Channel,
    /// `bf16`, nothing to undo.
    Bf16,
    /// Not a matrix; carried as stored.
    AsStored,
}

impl DraftQuant {
    /// The string the plan JSON uses.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            DraftQuant::Fp8Block => "fp8_block",
            DraftQuant::Fp8Channel => "fp8_channel",
            DraftQuant::Bf16 => "bf16",
            DraftQuant::AsStored => "as_stored",
        }
    }
}

/// One checkpoint tensor that feeds a pack record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftSource {
    /// Which file it lives in.
    pub checkpoint: Checkpoint,
    /// The tensor's own name.
    pub weight: String,
    /// What has to be undone.
    pub quant: DraftQuant,
    /// The companion scale, when there is one.
    pub scale: Option<String>,
}

impl DraftSource {
    fn drafter(weight: impl Into<String>, quant: DraftQuant, scale: Option<String>) -> Self {
        DraftSource {
            checkpoint: Checkpoint::Drafter,
            weight: weight.into(),
            quant,
            scale,
        }
    }

    fn raw(weight: impl Into<String>) -> Self {
        DraftSource::drafter(weight, DraftQuant::AsStored, None)
    }

    fn fp8(weight: &str) -> Self {
        DraftSource::drafter(
            weight.to_owned(),
            DraftQuant::Fp8Block,
            Some(format!("{weight}_scale_inv")),
        )
    }

    fn head() -> Self {
        DraftSource {
            checkpoint: Checkpoint::Target,
            weight: "lm_head.weight".into(),
            quant: DraftQuant::Fp8Channel,
            scale: Some("lm_head.weight_scale".into()),
        }
    }
}

/// What a record is to the launch that reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftRole {
    /// The matrix itself (`w`, `wq`).
    Weight,
    /// Its scales (`ws`, `s`).
    Scale,
    /// Its zero points (`zs`).
    Zero,
    /// A norm weight, a codebook, a convolution base — carried as stored or cast.
    Passthrough,
    /// A table the builder COMPUTES from the config rather than copying (the rope cache).
    Table,
}

impl DraftRole {
    /// The string the plan JSON uses.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            DraftRole::Weight => "weight",
            DraftRole::Scale => "scale",
            DraftRole::Zero => "zero",
            DraftRole::Passthrough => "passthrough",
            DraftRole::Table => "table",
        }
    }
}

/// One record a DFlash2 drafter pack holds, for one rank.
#[derive(Clone, Debug, PartialEq)]
pub struct DraftRecord {
    /// The record's name in the pack: the op that reads it plus the kernarg role.
    pub name: String,
    /// `docs/abi/PACK.md` § Layout ids.
    pub layout_id: u32,
    /// What it is to its consumer.
    pub role: DraftRole,
    /// The [`DraftOp`] that reads it, when one does.
    pub op: Option<String>,
    /// The dtype the pack holds.
    pub dtype: Dtype,
    /// How the record is sharded.
    pub shard: ShardKind,
    /// The **rank-local** shape, as the record entry carries it.
    pub shape: Vec<u64>,
    /// The checkpoint tensors that produced it, in concatenation order.
    pub sources: Vec<DraftSource>,
    /// Numbers the transform needs that the shape does not carry (`group`, `bits`, `theta`).
    pub params: Vec<(&'static str, f64)>,
}

impl DraftRecord {
    /// The kernarg this record binds to: the LAST dot-segment of its name.
    ///
    /// The naming rule is what makes this total rather than a table —
    /// `layer.0.attn.qkv.w` -> `w`, `head.int2.zs` -> `zs`,
    /// `layer.2.mlp_conv.base` -> `base`, `context_kv.rope.cos_sin` -> `cos_sin`,
    /// `norm.weight` -> `weight` — and
    /// `every_records_kernarg_is_one_its_kernel_declares` is the test that says so against the
    /// plugin ABI itself, which is the only place the argument names actually live. A record whose
    /// name did not end in its kernarg would bind nothing, and an unbound buffer is whatever the
    /// arena held.
    #[must_use]
    pub fn kernarg(&self) -> &str {
        self.name.rsplit('.').next().unwrap_or(&self.name)
    }
}

/// The drafter's KV cache geometry, which the plan sizes and no checkpoint tensor carries.
///
/// It is in the plan rather than in a record because there is nothing to load: the cache starts
/// empty. What a loader needs from the pack is its SHAPE, and `dflash_attn`'s `block_size` is
/// load-bearing — the kernel divides a key position by it to index the block table — so it is
/// declared here beside the weights rather than rediscovered at allocation time.
///
/// **This is Saber's geometry, and prod's is one tensor per layer rather than two.**
/// `TritonAttentionBackend::get_kv_cache_shape` allocates
/// `[blocks, kv_heads, block_size, 2 * head_dim]` and `do_kv_cache_update` hands the append kernel
/// two STRIDED views of it (`transpose(1, 2).split(head_size, dim=-1)`), while `dflash_kv` and
/// `dflash_attn` take two separate CONTIGUOUS caches — which is what every capture this plugin was
/// gated against allocated. Same arithmetic, different stride, and a proposal replayed against
/// prod's own cache tensor would have to carry the interleave. Recorded here because a geometry
/// that is only in a capture script is a geometry nobody checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftKvCache {
    /// Decoder layers, each with its own k and v cache.
    pub layers: u32,
    /// Tokens per block; `[attn].block_size` in the committed rules.
    pub block_size: u32,
    /// KV heads on one rank.
    pub kv_heads: u32,
    /// Head dimension.
    pub head_dim: u32,
    /// Bytes per element: 1, because the drafter's cache is `e4m3`
    /// (`dflash_attn_h128_gqa4_fp8kv_t32`, and `capture_dflash_kv.py` reads the same dtype off the
    /// live serve).
    pub elem_bytes: u32,
}

impl DraftKvCache {
    /// Bytes one block of ONE cache (k or v) of ONE layer takes.
    #[must_use]
    pub fn block_bytes(&self) -> u64 {
        u64::from(self.block_size)
            * u64::from(self.kv_heads)
            * u64::from(self.head_dim)
            * u64::from(self.elem_bytes)
    }

    /// Bytes the whole cache takes at `blocks` blocks: both caches, every layer.
    #[must_use]
    pub fn total_bytes(&self, blocks: u64) -> u64 {
        2 * u64::from(self.layers) * blocks * self.block_bytes()
    }
}

/// The rope table's parameters, read off the drafter's own `config.json`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DraftRope {
    /// `rope_parameters.rope_theta`.
    pub theta: f64,
    /// `head_dim`; the drafter's rope is FULL, so `rotary_dim == head_dim`.
    pub rotary_dim: u32,
    /// How many positions the table covers.
    pub max_pos: u32,
}

impl DraftRope {
    /// `Qwen3.8-27B-DFlash2-FP8`'s own numbers at a given context length.
    #[must_use]
    pub fn qwen38_dflash2(max_pos: u32) -> Self {
        DraftRope {
            theta: 1.0e7,
            rotary_dim: 128,
            max_pos,
        }
    }

    /// The record's NAME in a drafter pack (and the plan's binding).
    pub const RECORD: &'static str = "context_kv.rope.cos_sin";

    /// Bytes of one row: `rotary_dim` bf16 values, `[cos(rotary_dim / 2) | sin(rotary_dim / 2)]`.
    #[must_use]
    pub fn row_bytes(&self) -> u64 {
        u64::from(self.rotary_dim) * 2
    }

    /// **P27-9d: the table itself**, `[max_pos, rotary_dim]` bf16 rows of `[cos | sin]`, built on
    /// the host so the engine can size it to the BOOTED context rather than the pack's (the
    /// served drafter pack carries 4,096 rows and the server runs 262,144).
    ///
    /// The law (`build_pack.py`'s `rope_cos_sin`, `RotaryEmbedding._compute_cos_sin_cache`):
    /// `inv_freq = 1 / theta^(2i / rotary_dim)` in f32, `freq = pos * inv_freq` in f32, cos / sin
    /// of that f32 ROUNDED TO f32 (computed in f64 from the exact f32 argument, i.e. correctly
    /// rounded), then bf16 round-to-nearest-even. Measured (2026-10-07): rows `0..4,096` equal
    /// both ranks' pack record bit for bit (0 of 524,288 elements, sha256 `5ad487db...c683`); at
    /// 262,144 rows torch's own f32 libm (not correctly rounded: 5.1 M f32 elements differ by an
    /// ulp) lands on the other bf16 in 115 of 33.5 M elements, first at position 4,498 -- this
    /// table is the correctly rounded one. A row is a function of its position only, so a longer
    /// table's prefix IS the shorter table.
    #[must_use]
    pub fn cos_sin_bf16(&self) -> Vec<u8> {
        let half = (self.rotary_dim / 2) as usize;
        let inv: Vec<f32> = (0..half)
            .map(|i| {
                let e = (2 * i) as f32 / self.rotary_dim as f32;
                1.0f32 / (self.theta.powf(f64::from(e)) as f32)
            })
            .collect();
        let rne = |f: f32| -> [u8; 2] {
            let u = f.to_bits();
            (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16).to_le_bytes()
        };
        let mut out = Vec::with_capacity(self.max_pos as usize * self.row_bytes() as usize);
        let mut fr = vec![0f32; half];
        for p in 0..self.max_pos {
            for (f, &v) in fr.iter_mut().zip(&inv) {
                *f = p as f32 * v;
            }
            for &f in &fr {
                out.extend_from_slice(&rne(f64::from(f).cos() as f32));
            }
            for &f in &fr {
                out.extend_from_slice(&rne(f64::from(f).sin() as f32));
            }
        }
        out
    }
}

impl DraftGraph {
    /// The record set a drafter pack must hold, in forward order.
    ///
    /// `rope` sizes the one COMPUTED table; every other shape comes from [`DflashConfig`], so a
    /// change to the drafter's geometry moves the pack by construction rather than by
    /// transcription.
    #[must_use]
    pub fn records(&self, rope: DraftRope) -> Vec<DraftRecord> {
        let c = &self.cfg;
        let h = u64::from(c.hidden);
        let hd = u64::from(c.head_dim);
        let vloc = u64::from(c.vocab_local());
        let group = 128u64;
        let mut out: Vec<DraftRecord> = Vec::new();

        // A preshuffled fp8 B and its block scales are ONE transform and TWO records, so they are
        // emitted together and can never be half-declared.
        fn gemm(
            out: &mut Vec<DraftRecord>,
            op: &str,
            n: u64,
            k: u64,
            shard: ShardKind,
            sources: Vec<DraftSource>,
        ) {
            out.push(DraftRecord {
                name: format!("{op}.w"),
                layout_id: LAYOUT_FP8_PRESHUFFLE,
                role: DraftRole::Weight,
                op: Some(op.to_owned()),
                dtype: Dtype::F8E4M3,
                shard,
                shape: vec![n / 16, k * 16],
                sources: sources.clone(),
                params: vec![("tile", 16.0)],
            });
            out.push(DraftRecord {
                name: format!("{op}.ws"),
                layout_id: LAYOUT_FP8_BLOCK_SCALE,
                role: DraftRole::Scale,
                op: Some(op.to_owned()),
                dtype: Dtype::F32,
                shard,
                shape: vec![n / 128, k / 128],
                sources,
                params: vec![("group", 128.0)],
            });
        }

        // ---- once per step, before the layers ----------------------------------------------
        gemm(
            &mut out,
            "aux_hidden_projection",
            h,
            u64::from(c.target_layers) * h,
            // N is `hidden` and K is the five target layers' hidden states concatenated, neither
            // of which TP divides: every rank folds the same aux projection.
            ShardKind::Replicated,
            vec![DraftSource::fp8("fc.weight")],
        );
        out.push(DraftRecord {
            name: "context_kv.hidden_norm.weight".into(),
            layout_id: LAYOUT_RAW,
            role: DraftRole::Passthrough,
            op: Some("context_kv.hidden_norm".into()),
            dtype: Dtype::Bf16,
            shard: ShardKind::Replicated,
            shape: vec![h],
            sources: vec![DraftSource::raw("hidden_norm.weight")],
            params: vec![],
        });
        // The fused KV projection: every layer's K and V rows of `qkv_proj`, LAYER-MAJOR with k
        // before v, which is the order `all_kv_flat.view(num_ctx, L, 2, nkv, head_dim)` reads them
        // back in and the reason the store's next launch is a `permute(2, 1, 0, 3, 4)`.
        let mut kv_sources = Vec::new();
        for i in 0..c.layers {
            kv_sources.push(DraftSource::fp8(&format!(
                "layers.{i}.self_attn.k_proj.weight"
            )));
            kv_sources.push(DraftSource::fp8(&format!(
                "layers.{i}.self_attn.v_proj.weight"
            )));
        }
        out.push(DraftRecord {
            name: "context_kv.kv_projection.w".into(),
            layout_id: LAYOUT_BF16_FROM_FP8_BLOCK,
            role: DraftRole::Weight,
            op: Some("context_kv.kv_projection".into()),
            dtype: Dtype::Bf16,
            shard: ShardKind::Dim0Segments,
            shape: vec![u64::from(c.context_kv_n()), h],
            sources: kv_sources,
            params: vec![],
        });
        out.push(DraftRecord {
            name: "context_kv.k_norm.weight".into(),
            layout_id: LAYOUT_BF16_MERGED,
            role: DraftRole::Passthrough,
            op: Some("context_kv.k_norm".into()),
            dtype: Dtype::Bf16,
            shard: ShardKind::Replicated,
            // `_build_context_kv_buffers` stacks the five layers' `k_norm.weight` into one
            // `[L, head_dim]` buffer "so the per-layer K-norm runs as a single grouped kernel".
            shape: vec![u64::from(c.layers), hd],
            sources: (0..c.layers)
                .map(|i| DraftSource::raw(format!("layers.{i}.self_attn.k_norm.weight")))
                .collect(),
            params: vec![],
        });
        out.push(DraftRecord {
            name: "context_kv.rope.cos_sin".into(),
            layout_id: LAYOUT_ROPE_COS_SIN,
            role: DraftRole::Table,
            op: Some("context_kv.rope".into()),
            dtype: Dtype::Bf16,
            shard: ShardKind::Replicated,
            shape: vec![u64::from(rope.max_pos), u64::from(rope.rotary_dim)],
            sources: vec![],
            params: vec![
                ("theta", rope.theta),
                ("rotary_dim", f64::from(rope.rotary_dim)),
                ("max_pos", f64::from(rope.max_pos)),
            ],
        });
        out.push(DraftRecord {
            name: "input_embedding.weight".into(),
            layout_id: LAYOUT_RAW,
            role: DraftRole::Passthrough,
            op: Some("input_embedding".into()),
            dtype: Dtype::Bf16,
            shard: ShardKind::Dim0,
            shape: vec![vloc, h],
            sources: vec![DraftSource {
                checkpoint: Checkpoint::Target,
                weight: "model.language_model.embed_tokens.weight".into(),
                quant: DraftQuant::Bf16,
                scale: None,
            }],
            params: vec![],
        });

        // ---- the five layers ----------------------------------------------------------------
        for i in 0..c.layers {
            let p = |s: &str| format!("layer.{i}.{s}");
            gemm(
                &mut out,
                &p("attn.qkv"),
                u64::from(c.qkv_n()),
                h,
                ShardKind::Dim0Segments,
                vec![
                    DraftSource::fp8(&format!("layers.{i}.self_attn.q_proj.weight")),
                    DraftSource::fp8(&format!("layers.{i}.self_attn.k_proj.weight")),
                    DraftSource::fp8(&format!("layers.{i}.self_attn.v_proj.weight")),
                ],
            );
            for role in ["q_norm", "k_norm"] {
                out.push(DraftRecord {
                    name: p(&format!("attn.{role}.weight")),
                    layout_id: LAYOUT_RAW,
                    role: DraftRole::Passthrough,
                    op: Some(p("attn.qk_norm_rope")),
                    dtype: Dtype::Bf16,
                    shard: ShardKind::Replicated,
                    shape: vec![hd],
                    sources: vec![DraftSource::raw(format!(
                        "layers.{i}.self_attn.{role}.weight"
                    ))],
                    params: vec![],
                });
            }
            gemm(
                &mut out,
                &p("attn.o_proj"),
                h,
                u64::from(c.q_heads_local()) * hd,
                ShardKind::Dim1,
                vec![DraftSource::fp8(&format!(
                    "layers.{i}.self_attn.o_proj.weight"
                ))],
            );
            gemm(
                &mut out,
                &p("mlp.gate_up"),
                2 * u64::from(c.intermediate / c.tp),
                h,
                ShardKind::Dim0Segments,
                vec![
                    DraftSource::fp8(&format!("layers.{i}.mlp.gate_proj.weight")),
                    DraftSource::fp8(&format!("layers.{i}.mlp.up_proj.weight")),
                ],
            );
            gemm(
                &mut out,
                &p("mlp.down"),
                h,
                u64::from(c.intermediate / c.tp),
                ShardKind::Dim1,
                vec![DraftSource::fp8(&format!(
                    "layers.{i}.mlp.down_proj.weight"
                ))],
            );
            for tag in ["attention_conv", "mlp_conv"] {
                out.push(DraftRecord {
                    name: p(&format!("{tag}.kernel_projection.w")),
                    layout_id: LAYOUT_RAW,
                    role: DraftRole::Weight,
                    op: Some(p(&format!("{tag}.kernel_projection"))),
                    dtype: Dtype::Bf16,
                    shard: ShardKind::Replicated,
                    shape: vec![u64::from(c.kernel_projection_n()), h],
                    sources: vec![DraftSource::raw(format!(
                        "layers.{i}.{tag}.kernel_projection.weight"
                    ))],
                    params: vec![],
                });
                out.push(DraftRecord {
                    name: p(&format!("{tag}.base")),
                    layout_id: LAYOUT_RAW,
                    role: DraftRole::Passthrough,
                    op: Some(p(&format!("{tag}.prepare"))),
                    dtype: Dtype::Bf16,
                    shard: ShardKind::Replicated,
                    // `[side, taps, hidden]`: `prepare` reads side 0 and `finish` side 1, which is
                    // why the kernel takes the whole buffer and a side index.
                    shape: vec![2, u64::from(c.conv_taps), h],
                    sources: vec![DraftSource::raw(format!("layers.{i}.{tag}.base_kernel"))],
                    params: vec![],
                });
            }
            // WHICH LAUNCH READS A NORM WEIGHT IS NOT WHICH LAYER OWNS IT, and the join test above
            // is what said so. Only layer 0's `input_layernorm` is a standalone norm; every other
            // layer's is the TAIL of the PREVIOUS layer's `mlp_conv.finish`, because that is how
            // inductor already fuses the pair. So layer i's input norm is read by
            // `layer.{i-1}.mlp_conv.finish+next_layernorm`, and the drafter's own final `norm` is
            // read by the last layer's.
            let input_norm_op = if i == 0 {
                p("input_layernorm")
            } else {
                format!("layer.{}.mlp_conv.finish+next_layernorm", i - 1)
            };
            for (tensor, op) in [
                ("input_layernorm", input_norm_op),
                (
                    "post_attention_layernorm",
                    p("attention_conv.finish+post_attention_layernorm"),
                ),
            ] {
                out.push(DraftRecord {
                    name: p(&format!("{tensor}.weight")),
                    layout_id: LAYOUT_RAW,
                    role: DraftRole::Passthrough,
                    op: Some(op),
                    dtype: Dtype::Bf16,
                    shard: ShardKind::Replicated,
                    shape: vec![h],
                    sources: vec![DraftSource::raw(format!("layers.{i}.{tensor}.weight"))],
                    params: vec![],
                });
            }
        }

        // ---- the tail: the final norm, the head, the selector --------------------------------
        // `mlp_conv.finish` carries the NEXT layer's input norm, and in the last layer that is the
        // drafter's own `norm` -- the one record whose consumer is another layer's op.
        out.push(DraftRecord {
            name: "norm.weight".into(),
            layout_id: LAYOUT_RAW,
            role: DraftRole::Passthrough,
            op: Some(format!(
                "layer.{}.mlp_conv.finish+next_layernorm",
                c.layers - 1
            )),
            dtype: Dtype::Bf16,
            shard: ShardKind::Replicated,
            shape: vec![h],
            sources: vec![DraftSource::raw("norm.weight")],
            params: vec![],
        });
        for (suffix, layout, role, dtype, shape) in [
            (
                "wq",
                LAYOUT_INT2_HEAD,
                DraftRole::Weight,
                Dtype::U8,
                vec![vloc, h / 4],
            ),
            (
                "s",
                LAYOUT_INT2_HEAD_SCALE,
                DraftRole::Scale,
                Dtype::Bf16,
                vec![vloc, h / group],
            ),
            (
                "zs",
                LAYOUT_INT2_HEAD_ZERO,
                DraftRole::Zero,
                Dtype::Bf16,
                vec![vloc, h / group],
            ),
        ] {
            out.push(DraftRecord {
                name: format!("head.int2.{suffix}"),
                layout_id: layout,
                role,
                op: Some("head.int2".into()),
                dtype,
                shard: ShardKind::Dim0,
                shape,
                sources: vec![DraftSource::head()],
                // `from_bf16_head` says the quantiser's INPUT is the live head, not the
                // checkpoint: `_quantize_head_now` runs at first inference against whatever
                // `_head_matrix` returns, and under `SABER_NVFP4_LMHEAD=bf16` that is
                // `bf16(f32(w8) * f32(scale))` rather than the e4m3 bytes times their scale in
                // f32. One extra rounding per element, applied BEFORE the 128-wide min/max, so
                // it moves the group's scale and zero point and therefore every packed nibble.
                params: vec![("group", 128.0), ("bits", 2.0), ("from_bf16_head", 1.0)],
            });
        }
        // THE RERANK SCORES AGAINST THE LIVE HEAD, WHICH IS bf16, NOT THE CHECKPOINT'S e4m3.
        //
        // `SABER_NVFP4_LMHEAD=bf16` — the unit's setting and `ref_nvfp4`'s own default —
        // has `RefFp8ChannelToBf16.process_weights_after_loading` replace the checkpoint's
        // `[N, K]` e4m3 weight and `[N, 1]` f32 `weight_scale` with a single bf16 matrix,
        // `bf16(f32(w8) * f32(scale))` a row chunk at a time, BEFORE the drafter is built. So
        // `ref_drafthead._head_matrix` returns `(bf16 rows, None)`, `_rerank_exact` takes its
        // `FP8=False` arm, and `_quantize_head_now` builds the int2 head from those same bf16
        // bytes.
        //
        // Until 2026-09-21 this pack carried the checkpoint's e4m3 + per-row scale instead, which
        // is a DIFFERENT MATRIX by one bf16 rounding per element. Measured against prod's own
        // captured `unary_logits`, on the host with no card: the bf16 head is identical of 448
        // bytes on all three corpus prompts and the e4m3-times-scale one differs in 6 / 8 / 9.
        // There is no `head.rerank.ws` any more, because the arm prod runs does not read one.
        //
        // N6: under [`RerankHead::W8`] the same matrix is carried as the checkpoint's own e4m3
        // bytes and its f32 scale — the target fp8 head's `lm_head.w8` / `lm_head.ws` layouts
        // (15 / 16), so the two packs' records are sha-identical and the card holds one — and
        // `DflashOp::RerankExactW8` expands it in register to these bf16 bytes.
        match self.rerank_head {
            RerankHead::Bf16 => out.push(DraftRecord {
                name: "head.rerank.w".into(),
                layout_id: LAYOUT_BF16_FROM_FP8_CHANNEL,
                role: DraftRole::Weight,
                op: Some("head.rerank".into()),
                dtype: Dtype::Bf16,
                shard: ShardKind::Dim0,
                shape: vec![vloc, h],
                sources: vec![DraftSource::head()],
                params: vec![],
            }),
            RerankHead::W8 => {
                out.push(DraftRecord {
                    name: "head.rerank.w8".into(),
                    layout_id: LAYOUT_FP8_CHANNEL,
                    role: DraftRole::Weight,
                    op: Some("head.rerank".into()),
                    dtype: Dtype::F8E4M3,
                    shard: ShardKind::Dim0,
                    shape: vec![vloc, h],
                    sources: vec![DraftSource::head()],
                    params: vec![],
                });
                out.push(DraftRecord {
                    name: "head.rerank.ws".into(),
                    layout_id: LAYOUT_F32_CHANNEL_SCALE,
                    role: DraftRole::Scale,
                    op: Some("head.rerank".into()),
                    dtype: Dtype::F32,
                    shard: ShardKind::Dim0,
                    shape: vec![vloc],
                    sources: vec![DraftSource::head()],
                    params: vec![],
                });
            }
        }
        out.push(DraftRecord {
            name: "selector.hidden_projection.w".into(),
            layout_id: LAYOUT_RAW,
            role: DraftRole::Weight,
            op: Some("selector.hidden_projection".into()),
            dtype: Dtype::Bf16,
            shard: ShardKind::Replicated,
            shape: vec![u64::from(c.selector_rank), h],
            sources: vec![DraftSource::raw(
                "candidate_selector.hidden_projection.weight",
            )],
            params: vec![],
        });
        for (role, tensor) in [
            ("pred_cb", "predecessor_codebook"),
            ("succ_cb", "successor_codebook"),
        ] {
            out.push(DraftRecord {
                name: format!("selector.score_edges.{role}"),
                layout_id: LAYOUT_RAW,
                role: DraftRole::Passthrough,
                op: Some("selector.score_edges".into()),
                dtype: Dtype::Bf16,
                // Indexed by GLOBAL token id -- the selector's candidates are token ids, not
                // shard offsets -- so a vocab-parallel split would point every rank at the wrong
                // row. Replicated is a decision, not an omission.
                shard: ShardKind::Replicated,
                shape: vec![u64::from(c.vocab), u64::from(c.selector_rank)],
                sources: vec![DraftSource::raw(format!("candidate_selector.{tensor}"))],
                params: vec![],
            });
        }
        out
    }

    /// The drafter's KV cache geometry, from the plan. `block_size` is `[attn].block_size`.
    #[must_use]
    pub fn kv_cache(&self, block_size: u32) -> DraftKvCache {
        DraftKvCache {
            layers: self.cfg.layers,
            block_size,
            kv_heads: self.cfg.kv_heads_local(),
            head_dim: self.cfg.head_dim,
            elem_bytes: 1,
        }
    }
}

#[cfg(test)]
mod record_tests {
    use super::*;

    fn g() -> DraftGraph {
        DraftGraph::build(DflashConfig::qwen38_dflash2(2, 7))
    }

    fn recs() -> Vec<DraftRecord> {
        g().records(DraftRope::qwen38_dflash2(4096))
    }

    /// Every record's `op` must name an op the plan actually declares. This is the join that makes
    /// "the pack holds what the plan reads" checkable rather than asserted: a record whose op is
    /// misspelled would bind nothing and the launch would read whatever the arena held.
    #[test]
    fn every_record_names_a_declared_op() {
        let graph = g();
        let names: std::collections::BTreeSet<&str> =
            graph.ops.iter().map(|o| o.name.as_str()).collect();
        for r in graph.records(DraftRope::qwen38_dflash2(4096)) {
            let op = r.op.as_deref().expect("every record names its consumer");
            assert!(
                names.contains(op),
                "record `{}` names op `{op}`, which the plan does not declare",
                r.name
            );
        }
    }

    /// And the other direction: every op that READS a weight has its records. The list of ops that
    /// legitimately read none is explicit, because "this op has no weight" is a claim.
    #[test]
    fn every_weight_reading_op_has_its_records() {
        let graph = g();
        let recs = graph.records(DraftRope::qwen38_dflash2(4096));
        let covered: std::collections::BTreeSet<&str> =
            recs.iter().filter_map(|r| r.op.as_deref()).collect();
        // Ops with no weight of their own: the conv `prepare`/`finish` pair reads the coefficients
        // a projection just produced, the activation quant and the silu read activations, the
        // reduces and the selections read nothing stored, the KV appends read the cache, and the
        // head's row pad and group sum read the head's own input.
        let weightless = |n: &str| {
            n.ends_with(".prepare")
                || n.ends_with(".quant")
                || n.contains("all_reduce")
                || n.contains("kv_append")
                || n.contains("cache_insert")
                || n.contains("attn.attention")
                || n.contains("mlp.silu_quant")
                || n.contains("candidate_topk")
                || n == "selector.candidate_seam"
                || n == "head.candidate_topk.gather_ids"
                || n == "head.sample_rows"
                || n == "selector.anchor_ids"
                || n == "context_kv.layout"
                || n == "head.row_pad"
                || n == "head.group_sum"
                || n == "selector.walk"
        };
        for op in &graph.ops {
            if weightless(&op.name) {
                continue;
            }
            assert!(
                covered.contains(op.name.as_str()),
                "op `{}` reads a weight and no record feeds it",
                op.name
            );
        }
    }

    /// The counts, declared. 95 records: 7 before the layers, 16 per layer, 8 in the tail.
    ///
    /// It was 96 until 2026-09-21: `head.rerank.ws` is gone, because the arm prod runs
    /// (`_rerank_exact`'s `FP8=False`) reads a bf16 head and no per-row scale at all.
    /// N6: the fp8 rerank head swaps ONE bf16 record for the checkpoint's two (e4m3 matrix +
    /// f32 scale, the target fp8 head's layouts 15 / 16) and moves the `head.rerank` op's coverage
    /// to `RerankExactW8`; nothing else moves. CONTROL: the default build is the bf16 record set.
    #[test]
    fn the_w8_rerank_carries_the_checkpoints_two_records_and_nothing_else_moves() {
        let bf = DraftGraph::build(DflashConfig::qwen38_dflash2(2, 7));
        let w8 = bf.clone().with_rerank_head(RerankHead::W8);
        let rope = DraftRope::qwen38_dflash2(4096);
        let (rb, r8) = (bf.records(rope), w8.records(rope));
        assert_eq!(r8.len(), rb.len() + 1);
        let by = |v: &[DraftRecord], n: &str| v.iter().find(|r| r.name == n).cloned();
        let m = by(&r8, "head.rerank.w8").expect("w8");
        let sc = by(&r8, "head.rerank.ws").expect("ws");
        assert_eq!(
            (m.layout_id, m.dtype, m.shape.clone()),
            (LAYOUT_FP8_CHANNEL, Dtype::F8E4M3, vec![124_160, 5120])
        );
        assert_eq!(
            (sc.layout_id, sc.dtype, sc.shape.clone()),
            (LAYOUT_F32_CHANNEL_SCALE, Dtype::F32, vec![124_160])
        );
        assert_eq!((m.kernarg(), sc.kernarg()), ("w8", "ws"));
        assert!(by(&r8, "head.rerank.w").is_none());
        assert!(by(&rb, "head.rerank.w").is_some() && by(&rb, "head.rerank.w8").is_none());
        for r in &rb {
            if r.name != "head.rerank.w" {
                assert_eq!(
                    by(&r8, &r.name).map(|x| x.name),
                    Some(r.name.clone()),
                    "{} moved",
                    r.name
                );
            }
        }
        let cov = |g: &DraftGraph| {
            g.ops
                .iter()
                .find(|o| o.name == "head.rerank")
                .expect("op")
                .coverage
                .clone()
        };
        assert_eq!(cov(&w8), Coverage::Dflash(DflashOp::RerankExactW8));
        assert_eq!(cov(&bf), Coverage::Dflash(DflashOp::RerankExactBf16));
        assert_eq!(w8.budget(), bf.budget());
    }

    #[test]
    fn the_record_count_is_declared() {
        let recs = recs();
        assert_eq!(recs.len(), 7 + 5 * 16 + 8, "the declared record count");
        assert_eq!(recs.len(), 95);
        let mut names: Vec<&str> = recs.iter().map(|r| r.name.as_str()).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before, "record names must be unique in a pack");
    }

    /// The shapes the transforms have to produce, at the rank the drafter runs at. Each of these
    /// was a place the layout could have been half a permutation wrong.
    #[test]
    fn the_preshuffled_shapes_are_the_kernels() {
        let recs = recs();
        let by = |n: &str| {
            recs.iter()
                .find(|r| r.name == n)
                .unwrap_or_else(|| panic!("no record {n}"))
                .clone()
        };
        // qkv at TP=2: q 2048 + k 512 + v 512 = 3072 rows over K = 5120, preshuffled to
        // [N/16, K*16] and scaled at [N/128, K/128].
        assert_eq!(by("layer.0.attn.qkv.w").shape, vec![192, 81_920]);
        assert_eq!(by("layer.0.attn.qkv.ws").shape, vec![24, 40]);
        // o_proj is row-parallel: K halves, N does not.
        assert_eq!(by("layer.4.attn.o_proj.w").shape, vec![320, 32_768]);
        assert_eq!(by("layer.4.attn.o_proj.ws").shape, vec![40, 16]);
        assert_eq!(by("layer.2.mlp.gate_up.w").shape, vec![1088, 81_920]);
        assert_eq!(by("layer.2.mlp.down.ws").shape, vec![40, 68]);
        // The aux projection is replicated: five target layers of hidden, folded on every rank.
        assert_eq!(by("aux_hidden_projection.w").shape, vec![320, 409_600]);
        assert_eq!(by("aux_hidden_projection.ws").shape, vec![40, 200]);
        // The int2 head over this rank's vocabulary shard, and its two premultiplied tables.
        assert_eq!(by("head.int2.wq").shape, vec![124_160, 1280]);
        assert_eq!(by("head.int2.s").shape, vec![124_160, 40]);
        assert_eq!(by("head.int2.zs").shape, vec![124_160, 40]);
        // bf16, dequantised from the target's per-channel fp8 head, with no companion scale:
        // that is the matrix `_head_matrix` returns under SABER_NVFP4_LMHEAD=bf16.
        assert_eq!(by("head.rerank.w").shape, vec![124_160, 5120]);
        assert_eq!(by("head.rerank.w").dtype, Dtype::Bf16);
        assert!(
            !recs.iter().any(|r| r.name == "head.rerank.ws"),
            "the FP8=False arm reads no per-row scale, so the pack must not carry one"
        );
        // The fused context-KV projection: every layer's K and V on one rank.
        assert_eq!(by("context_kv.kv_projection.w").shape, vec![5120, 5120]);
        assert_eq!(by("context_kv.k_norm.weight").shape, vec![5, 128]);
        assert_eq!(by("context_kv.rope.cos_sin").shape, vec![4096, 128]);
    }

    /// The two checkpoints, counted. A record map that named only the drafter's own file would be
    /// missing the embedding and the head, which is 2.1 GiB of what a proposal reads.
    #[test]
    fn the_head_and_the_embedding_come_from_the_target() {
        let recs = recs();
        let from_target: Vec<&str> = recs
            .iter()
            .filter(|r| r.sources.iter().any(|s| s.checkpoint == Checkpoint::Target))
            .map(|r| r.name.as_str())
            .collect();
        assert_eq!(
            from_target,
            vec![
                "input_embedding.weight",
                "head.int2.wq",
                "head.int2.s",
                "head.int2.zs",
                "head.rerank.w",
            ]
        );
        // And every drafter source names a tensor of the drafter's own checkpoint prefix set.
        for r in &recs {
            for s in &r.sources {
                if s.checkpoint != Checkpoint::Drafter {
                    continue;
                }
                assert!(
                    s.weight.starts_with("layers.")
                        || s.weight.starts_with("candidate_selector.")
                        || s.weight == "fc.weight"
                        || s.weight == "hidden_norm.weight"
                        || s.weight == "norm.weight",
                    "`{}` is not a DFlash2 checkpoint tensor name",
                    s.weight
                );
            }
        }
    }

    /// A block-fp8 source always carries its `weight_scale_inv`, and nothing else does. A missing
    /// scale is a matrix multiplied by whatever the arena held.
    #[test]
    fn every_block_fp8_source_carries_its_scale() {
        for r in recs() {
            for s in &r.sources {
                match s.quant {
                    DraftQuant::Fp8Block => assert_eq!(
                        s.scale.as_deref(),
                        Some(format!("{}_scale_inv", s.weight).as_str()),
                        "{}: block-fp8 source without its scale",
                        r.name
                    ),
                    DraftQuant::Fp8Channel => {
                        assert!(s.scale.is_some(), "{}: fp8 head without its scale", r.name);
                    }
                    DraftQuant::Bf16 | DraftQuant::AsStored => {
                        assert!(s.scale.is_none(), "{}: a dense source with a scale", r.name);
                    }
                }
            }
        }
    }

    /// The KV cache the plan sizes, and the one number in it that is load-bearing.
    #[test]
    fn the_kv_cache_geometry_is_the_rules() {
        let kv = g().kv_cache(16);
        assert_eq!(kv.layers, 5);
        assert_eq!(kv.kv_heads, 4, "8 kv heads over two ranks");
        assert_eq!(kv.head_dim, 128);
        assert_eq!(kv.elem_bytes, 1, "the drafter's cache is e4m3");
        assert_eq!(kv.block_bytes(), 16 * 4 * 128);
        assert_eq!(kv.total_bytes(1024), 2 * 5 * 1024 * 8192);
    }

    /// THE JOIN THAT MATTERS: every record a `dflash` op reads must name one of that kernel's own
    /// buffer arguments, read out of the plugin ABI rather than out of a second list here. This is
    /// what says the pack and the kernels agree — a record named `…​.weights` instead of `…​.weight`
    /// would build, load, verify and bind nothing.
    #[test]
    fn every_records_kernarg_is_one_its_kernel_declares() {
        use saber_kernels::abi::{signature_for, ArgKind};
        let graph = g();
        let rules = DflashRules::prod_dflash().expect("the committed rules");
        let by_name: std::collections::BTreeMap<&str, &DraftOp> =
            graph.ops.iter().map(|o| (o.name.as_str(), o)).collect();
        let mut checked = 0;
        let all = graph.records(DraftRope::qwen38_dflash2(4096));
        for r in &all {
            let op = by_name[r.op.as_deref().expect("an op")];
            let Coverage::Dflash(kop) = op.coverage else {
                continue; // `attn_glue`'s q/k norms, the head plugin's embedding, the absent ops
            };
            let shape = op.shape.expect("a covered op declares its shape");
            let sym = saber_kernels::dflash::resolve(kop, shape, &rules)
                .unwrap_or_else(|e| panic!("{}: {e}", op.name))
                .kernel;
            let sig = signature_for(&sym).unwrap_or_else(|| panic!("{sym} has no signature"));
            let buffers: Vec<&str> = sig
                .iter()
                .filter(|a| a.kind == ArgKind::Buffer)
                .map(|a| a.name)
                .collect();
            // THE ONE OP WHOSE RECORDS BIND BY POSITION, and it is the checkpoint's doing rather
            // than a convention: the drafter's two per-head norm weights are both called
            // `weight` (`self_attn.q_norm.weight`, `self_attn.k_norm.weight`), so the record
            // map's last-dot-segment key cannot tell them apart and no renaming of the KERNEL
            // can make it. The join this test exists for still holds; it is an ORDER here rather
            // than a name, and it is asserted as one, so a third record, a dropped one or a
            // swapped push still fails.
            if kop == DflashOp::QkNormRope {
                let mine: Vec<&str> = all
                    .iter()
                    .filter(|x| x.op.as_deref() == r.op.as_deref())
                    .map(|x| x.name.as_str())
                    .collect();
                assert_eq!(
                    mine.len(),
                    2,
                    "{}: two weight slots, two records, bound in order",
                    op.name
                );
                assert!(
                    mine[0].ends_with("q_norm.weight") && mine[1].ends_with("k_norm.weight"),
                    "{}: the plan pushes q_norm then k_norm, which is `qw` then `kw`: {mine:?}",
                    op.name
                );
                assert_eq!(
                    buffers
                        .iter()
                        .filter(|b| **b == "qw" || **b == "kw")
                        .count(),
                    2,
                    "{sym} no longer takes two per-head norm weights"
                );
                assert_eq!(r.kernarg(), "weight");
                checked += 1;
                continue;
            }
            assert!(
                buffers.contains(&r.kernarg()),
                "record `{}` binds kernarg `{}`, which {sym} does not declare: {buffers:?}",
                r.name,
                r.kernarg()
            );
            checked += 1;
        }
        // 21 GEMM pairs + 12 bgemm w + 10 conv bases + 11 norms + 3 int2 + 1 rerank + 2 codebooks
        // + the two context norms and the rope table. The rerank is ONE record since the bf16 arm
        // reads no scale. The twelfth bgemm weight is `context_kv.kv_projection.w`: the pack has
        // carried it since F20 and it only reached a `dflash` kernarg in F26, when the op stopped
        // being `Coverage::Absent` -- which is what this count is for.
        // 84 through F27, plus the ten per-head norm weights that stopped being `attn_glue`'s
        // when the drafter's q/k norm + rope became `dflash_qk` (F28).
        assert_eq!(checked, 94, "the records a dflash kernel reads, counted");
    }

    /// Negative control for the join above: a record that names an op nobody declares must be
    /// caught, so the test is not passing because the check is vacuous.
    #[test]
    fn negative_control_a_record_for_an_undeclared_op_is_visible() {
        let graph = g();
        let names: std::collections::BTreeSet<&str> =
            graph.ops.iter().map(|o| o.name.as_str()).collect();
        assert!(
            !names.contains("layer.0.attn.qkv_proj"),
            "the control's misspelling must not be a real op name"
        );
    }

    /// One synthetic step table, built so that the WIDTH of the candidate array is the only thing
    /// that can be got wrong: every slot holds a distinct id above `2^32` in its own right, so an
    /// `int32` reader returns a token from the wrong slot at every even index and zero at every
    /// odd one.
    fn synthetic(steps: usize, top_k: usize) -> (Vec<f32>, Vec<i64>, Vec<i32>, Vec<usize>) {
        let mut scores = vec![0f32; steps * top_k * top_k];
        // Step `s`'s winner, from the row its predecessor names: a fixed walk chosen so that the
        // winners are NOT all even — an `int32` reader zeroes the odd ones, which is the whole
        // control.
        let winners: Vec<usize> = (0..steps).map(|s| (3 * s + 1) % top_k).collect();
        let mut previous = 0usize;
        for (s, w) in winners.iter().copied().enumerate() {
            let base = (s * top_k + previous) * top_k;
            for i in 0..top_k {
                #[allow(clippy::cast_precision_loss)]
                let v = (i as f32).mul_add(0.25, -1.0);
                scores[base + i] = v;
            }
            scores[base + w] = 100.0;
            previous = w;
        }
        // Distinct ids, all of them wider than 32 bits, so a truncating read is visible in the
        // VALUE as well as in the slot.
        let candidates: Vec<i64> = (0..steps * top_k)
            .map(|i| (1i64 << 33) + 7 * i as i64 + 1)
            .collect();
        (scores, candidates, vec![0i32; steps], winners)
    }

    /// The host walk follows the score chain and stores the candidate the winner names, as
    /// `int64`.
    #[test]
    fn the_walk_stores_the_int64_candidate_its_argmax_names() {
        let (steps, top_k) = (7usize, 16usize);
        let (scores, cand, state, winners) = synthetic(steps, top_k);
        let (tokens, realized) = selector_walk_greedy(&scores, &cand, &state, 1, steps, top_k);
        for (s, w) in winners.iter().copied().enumerate() {
            assert_eq!(
                tokens[s],
                cand[s * top_k + w],
                "step {s} must store candidate {w}"
            );
            assert!(
                tokens[s] > i64::from(u32::MAX),
                "the ids are 64-bit on purpose"
            );
        }
        assert_eq!(realized.len(), steps * top_k);
    }

    /// The CONTROL, and it is the defect this function exists to make visible: the same walk with
    /// the candidate table read four bytes at a time — `int32` over prod's `int64` buffer — must
    /// produce a DIFFERENT block. It is what `dflash_selector_walk_greedy` did until 2026-09-21,
    /// and it agrees with the truth on every score and every argmax, so only the tokens can catch
    /// it.
    #[test]
    fn negative_control_reading_the_candidates_as_int32_changes_the_block() {
        let (steps, top_k) = (7usize, 16usize);
        let (scores, cand, state, _) = synthetic(steps, top_k);
        let (tokens, realized) = selector_walk_greedy(&scores, &cand, &state, 1, steps, top_k);
        // The same bytes seen as `int32`, which is what an `int32*` kernarg over that buffer is.
        let as32: Vec<i32> = cand
            .iter()
            .flat_map(|v| [(*v & 0xffff_ffff) as i32, (*v >> 32) as i32])
            .collect();
        let widened: Vec<i64> = as32.iter().map(|v| i64::from(*v)).collect();
        let (bad, bad_real) = selector_walk_greedy(&scores, &widened, &state, 1, steps, top_k);
        assert_eq!(
            realized, bad_real,
            "the width error is invisible in the scores: that is why it survived"
        );
        assert_ne!(
            tokens, bad,
            "an int32 read of an int64 candidate table must change the draft block, or this \
             control has no teeth"
        );
    }

    /// Synthetic codebooks, candidates, hidden rows and f32 unary logits for the scorer, at the
    /// drafter's own rank and top-k.
    ///
    /// The values are deliberately ORDINARY — around 1e-2 per term, 256 of them — because the
    /// roundings this exercises are last-bit ones: a table of exact powers of two would round the
    /// same way in every arm and the controls would not fire.
    #[allow(clippy::type_complexity)]
    fn sel_synthetic(
        positions: usize,
        top_k: usize,
        rank: usize,
        vocab: usize,
    ) -> (Vec<u16>, Vec<u16>, Vec<i64>, Vec<i32>, Vec<u16>, Vec<f32>) {
        // A small deterministic LCG: no dependency, and the same stream on every host.
        let mut s = 0x2026_0921_u64;
        let mut next = || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            #[allow(clippy::cast_precision_loss)]
            let u = ((s >> 33) as u32 as f32) / (u32::MAX >> 1) as f32;
            u - 0.5
        };
        let mut cb = |n: usize| -> Vec<u16> {
            (0..n)
                .map(|_| f32_to_bf16_bits(next() * 0.1))
                .collect::<Vec<_>>()
        };
        let pred_cb = cb(vocab * rank);
        let succ_cb = cb(vocab * rank);
        let hidden: Vec<u16> = (0..positions * rank)
            .map(|_| f32_to_bf16_bits(next() * 0.5))
            .collect();
        let unary: Vec<f32> = (0..positions * top_k).map(|_| next() * 4.0).collect();
        // i64 ids in range of the codebooks, and DISTINCT per slot: an i32 read of this buffer
        // sees element `i` at byte `4i` rather than `8i`, so it gathers a different row for every
        // slot — the same trap `synthetic` sets for the walk, one stage earlier.
        #[allow(clippy::cast_possible_wrap)]
        let cand: Vec<i64> = (0..positions * top_k)
            .map(|i| ((i * 37 + 11) % vocab) as i64)
            .collect();
        (pred_cb, succ_cb, cand, vec![5i32], hidden, unary)
    }

    /// The three rows `the internal reference harness` prints, on the host with no capture:
    /// prod's own expression written out, and the two roundings dropped one at a time.
    ///
    /// Prod's arm is the CASE; the other two are CONTROLS and each must differ from it. Against
    /// prod's own captured `walk.scores` they differ in 775 / 588 / 635 and 3694 / 3725 / 3804 of
    /// 7168 bytes on the three prompts (F23); here the only claim is that each rounding MOVES the
    /// answer, which is what makes "the kernel must round here" a testable statement.
    #[test]
    fn score_edges_both_roundings_are_load_bearing() {
        let (positions, top_k, rank, vocab) = (7usize, 16usize, 256usize, 64usize);
        let (pred, succ, cand, anchor, hidden, unary) =
            sel_synthetic(positions, top_k, rank, vocab);
        let arm = |a| {
            score_edges_reference(
                &pred, &succ, &cand, &anchor, &hidden, &unary, positions, positions, top_k, rank, a,
            )
        };
        let prod = arm(ScoreEdgesArm::Prod);
        let f32mul = arm(ScoreEdgesArm::F32Mul);
        let f32all = arm(ScoreEdgesArm::F32);
        assert_eq!(prod.len(), positions * top_k * top_k);
        // Close in value — these are one-ULP differences, not a different computation ...
        let worst = prod
            .iter()
            .zip(&f32all)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            worst < 0.05,
            "the arms must be the same arithmetic: {worst}"
        );
        // ... and different in bits, which is the whole point.
        assert_ne!(
            prod, f32mul,
            "control: forming `predecessors * hidden` in f32 must change the answer"
        );
        assert_ne!(
            prod, f32all,
            "control: dropping the bf16 round of the 256-term dot must change the answer"
        );
    }

    /// The WIDTHS, one control each, and all three were wrong in the kernel until 2026-09-21.
    ///
    /// `cand` is `torch.topk`'s i64, `unary` is the exact rerank's f32 and the answer is therefore
    /// f32. Reading the candidate table as i32, rounding the bias through bf16, or rounding the
    /// answer through bf16 must each change the bytes — otherwise the gate that compares them
    /// proves nothing about the width.
    #[test]
    fn score_edges_negative_controls_on_every_element_width() {
        let (positions, top_k, rank, vocab) = (7usize, 16usize, 256usize, 64usize);
        let (pred, succ, cand, anchor, hidden, unary) =
            sel_synthetic(positions, top_k, rank, vocab);
        let good = score_edges_reference(
            &pred,
            &succ,
            &cand,
            &anchor,
            &hidden,
            &unary,
            positions,
            positions,
            top_k,
            rank,
            ScoreEdgesArm::Prod,
        );
        // 1. the same bytes read four at a time, which is what an `int*` kernarg over an i64
        //    buffer is.
        let as32: Vec<i64> = cand
            .iter()
            .flat_map(|v| {
                [
                    i64::from((*v & 0xffff_ffff) as i32),
                    i64::from((*v >> 32) as i32),
                ]
            })
            .take(cand.len())
            .collect();
        let misread = score_edges_reference(
            &pred,
            &succ,
            &as32,
            &anchor,
            &hidden,
            &unary,
            positions,
            positions,
            top_k,
            rank,
            ScoreEdgesArm::Prod,
        );
        assert_ne!(
            good, misread,
            "control: an i32 read of the i64 candidate table must gather different codebook rows"
        );
        // 2. the bias buffer READ AS bf16, which is what a `const __bf16 *unary` kernarg over an
        //    f32 buffer does: element `pos * top_k + c` of a u16 array is the low or the high half
        //    of a DIFFERENT f32. Rounding the values through bf16 is the wrong control here --
        //    prod's own `unary_logits` are already bf16-exact (the exact rerank stores through
        //    bf16), so that arm is identical to prod and proves nothing.
        let halves: Vec<u16> = unary
            .iter()
            .flat_map(|v| {
                let b = v.to_bits();
                [(b & 0xffff) as u16, (b >> 16) as u16]
            })
            .collect();
        let bias16: Vec<f32> = halves
            .iter()
            .take(unary.len())
            .map(|b| bf16_bits_to_f32(*b))
            .collect();
        let biased = score_edges_reference(
            &pred,
            &succ,
            &cand,
            &anchor,
            &hidden,
            &bias16,
            positions,
            positions,
            top_k,
            rank,
            ScoreEdgesArm::Prod,
        );
        assert_ne!(
            good, biased,
            "control: reading the f32 unary_logits buffer as bf16 elements must change the answer"
        );
        // 3. the answer, rounded through bf16 on its way out.
        let out16: Vec<f32> = good
            .iter()
            .map(|v| bf16_bits_to_f32(f32_to_bf16_bits(*v)))
            .collect();
        assert_ne!(
            good, out16,
            "control: storing the f32 score tensor as bf16 must change the bytes"
        );
    }

    /// An inactive request writes nothing, which is the reference's `mask & valid` on BOTH stores.
    #[test]
    fn an_inactive_request_stores_neither_token_nor_score() {
        let (steps, top_k) = (7usize, 16usize);
        let (scores, cand, _, _) = synthetic(steps, top_k);
        let dead = vec![-1i32; steps];
        let (tokens, realized) = selector_walk_greedy(&scores, &cand, &dead, 1, steps, top_k);
        assert!(tokens.iter().all(|t| *t == 0));
        assert!(realized.iter().all(|r| *r == 0.0));
    }

    /// P27-9d: the host-built drafter rope table IS the served pack's record below 4,096 rows
    /// (sha256 of `qwen38-dflash2-drafter-fp8head`'s `context_kv.rope.cos_sin`, both ranks, read
    /// 2026-10-07), a longer table's prefix is the shorter table, and its length is the plan's.
    ///
    /// NEGATIVE CONTROL: theta 1e6 (the 27B's text rope is 1e7 too, so a wrong constant is the
    /// plausible mistake) and a truncating bf16 (the target's `rope_cache` form) both miss.
    #[test]
    fn the_host_rope_table_is_the_packs_below_4096_rows_and_grows_by_position() {
        use sha2::{Digest, Sha256};
        const PACK_4096: &str = "5ad487dbac6560677f994e634fe0d283b8567687fbac0853884563af9ff9c683";
        let r = DraftRope::qwen38_dflash2(4096);
        let t = r.cos_sin_bf16();
        assert_eq!(t.len() as u64, 4096 * r.row_bytes());
        assert_eq!(
            hex::encode(Sha256::digest(&t)),
            PACK_4096,
            "the pack's record"
        );
        let long = DraftRope::qwen38_dflash2(8192).cos_sin_bf16();
        assert_eq!(
            &long[..t.len()],
            &t[..],
            "a row is a function of its position only"
        );
        // The plan's record carries exactly these bytes.
        let rec = g()
            .records(DraftRope::qwen38_dflash2(8192))
            .into_iter()
            .find(|x| x.name == DraftRope::RECORD)
            .expect("the rope record");
        assert_eq!(
            rec.shape.iter().product::<u64>() * u64::from(rec.dtype.bits()) / 8,
            long.len() as u64
        );
        // Controls.
        let wrong = DraftRope { theta: 1.0e6, ..r }.cos_sin_bf16();
        assert_ne!(hex::encode(Sha256::digest(&wrong)), PACK_4096);
        let mut trunc = Vec::with_capacity(t.len());
        let half = 64usize;
        let inv: Vec<f32> = (0..half)
            .map(|i| 1.0f32 / (1.0e7f64.powf(f64::from((2 * i) as f32 / 128.0)) as f32))
            .collect();
        for p in 0..4096u32 {
            let fr: Vec<f32> = inv.iter().map(|v| p as f32 * v).collect();
            for f in fr
                .iter()
                .map(|f| f64::from(*f).cos())
                .chain(fr.iter().map(|f| f64::from(*f).sin()))
            {
                trunc.extend_from_slice(&(((f as f32).to_bits() >> 16) as u16).to_le_bytes());
            }
        }
        assert_ne!(
            hex::encode(Sha256::digest(&trunc)),
            PACK_4096,
            "truncation is not RNE"
        );
    }
}
