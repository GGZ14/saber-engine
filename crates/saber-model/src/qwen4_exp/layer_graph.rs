//! The Flash-Next layer graph (FN9, plan §8 F2; `docs/EXEC.md` §15): the op list of every decoder
//! layer kind as the executor runs it, each op's kernel and launch count, and the per-layer / per-step
//! launch totals -- DECLARED here and asserted against the kernel plane's own constants
//! (`saber_kernels::{moe, flashnext, fn6}`), so a count cannot drift away from the round that
//! registered the kernel (structural rule 7).
//!
//! Op order (one decoder layer; `x` = the 10,240-wide 4-stream residual, bf16):
//!
//! | kind | ops |
//! |---|---|
//! | GDN | HC mix (attn side: `down` [+ the previous layer's MoE combine fused] + `up` bf16) -> in_proj (bf16, merged `qkvz`+`ba`, N 16,480 / 8,240) -> conv prep -> KKT solve -> chunk scan (prefill; decode: the fused update at b1, the conv+recurrent pair above) -> gated norm (SIGMOID gate, bf16 out) -> out_proj -> [AR] -> HC mix (MoE side: `down` combine + `up` q8) -> MoE (router GEMM, shared-gate GEMV, route+sort, grouped gate_up, silu-quant, grouped down, combine) -> [AR] |
//! | QSA | HC mix -> merged projection (q\|gate, k, v, index_qk; N 13,952 / 7,296) -> qk_norm_rope -> indexer q + compressed store -> KV append + ring -> attention -> output gate -> o_proj -> [AR] -> HC mix -> MoE -> [AR] |
//! | PLE (layer 1) | key GEMM, value GEMM, `fn_ple_gate_combine` (the previous layer's MoE combine folded in), `fn_ple_conv` (the add into the stream), THEN the attention-side mix as a plain `mix` |
//!
//! The HC inject of a sub-layer is DEFERRED into the next mix's `down` (FN5's combine mode), so a
//! layer's output stream only exists inside the next layer; layer 0's first mix is `expand` (the
//! embedding into 4 streams); after layer 47 the final mixer (`final` mode, no inject rows) makes the
//! head's input -- there is no final norm tensor (FN5).
//!
//! **FN9 findings the graph encodes**: (1) Flash-Next's GDN gated norm gates with SIGMOID
//! (`fn_gdn_norm_bf16_sig_t256`), not the 27B's silu (FN5's `fn_gdn_norm_bf16_t256`); (2) the MoE
//! block is SEVEN launches, not FN4's six: `moe_combine_bf16` reads the shared-expert gate LOGIT per
//! token (`sgate[t]`, stride 1) and nothing in FN4's train produces it (FN4 fed it random) -- one
//! transposed GEMV `C[1,T] = w_sg . X^T` on the bf16 band (N = T rows, the activations as the weight
//! operand) is the seventh (FN9b moved it into `moe::LAUNCHES_PER_MOE_BLOCK` = 7); (3) FN9b: the
//! GDN prefill scan is `flashnext_gdn`'s EXACT-GATE scan -- the certified saber scan splits each
//! intra-chunk decay e^{g_i-g_j} about the chunk midpoint and clamps each half at e^80, which
//! attenuates any 64-token chunk whose gates span > 160 nats (Flash-Next's layer 0 spans 1,706 over
//! 16 tokens: FN9's 43 %).

use saber_kernels::flashnext::{FlashNextRules, MixMode, MixOut};
use saber_kernels::fn19;
use saber_kernels::fn6::{self, AttnForm, PleSource};
use saber_kernels::moe;

use super::config::LayerKind;

/// What an op's time is attributed to in the per-op-group timing (brief (5)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Group {
    /// The hyper-connection mixes (and the standalone combine / final mixer).
    Hc,
    /// Dense bf16 projections (in_proj, out_proj, the QSA projection, o_proj).
    Proj,
    /// The GDN core or the attention core with its glue (norm/rope, indexer, append, gate).
    Core,
    /// The MoE train (router GEMM .. combine).
    Moe,
    /// The PLE at layer 1.
    Ple,
    /// The two-rank all-reduces (TP=2).
    Ar,
    /// Embedding and head.
    Head,
}

impl Group {
    /// Stable name for tables and JSON.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Group::Hc => "hc",
            Group::Proj => "proj",
            Group::Core => "core",
            Group::Moe => "moe",
            Group::Ple => "ple",
            Group::Ar => "ar",
            Group::Head => "head",
        }
    }
}

/// Tensor parallelism of the dense layers (the experts are EP at TP=2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tp {
    /// One card: the whole dense layer, all 512 experts routed locally.
    One,
    /// Two ranks: dense TP-split as FN2 packed it, experts EP (256 per rank), two reduces a layer.
    Two,
}

/// The step form: a prefill chunk, or a decode step at `batch` sequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Form {
    /// A prefill chunk (the GDN chunked triple, the attention prefill kernel).
    Prefill,
    /// A decode step of `batch` sequences; `attn` is the attention form the split law picked
    /// (`fn6::decode_route`).
    Decode { batch: u32, attn: AttnForm },
}

/// One op of the layer graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FnOp {
    /// Graph name (`l3.qsa.proj`, `l0.moe.route`, ...).
    pub name: String,
    /// Timing group.
    pub group: Group,
    /// The kernel symbol (or the family when the route resolves per M: `bgemm:<shape>`).
    pub kernel: String,
    /// Launches this op submits.
    pub launches: u32,
}

/// Where a layer's attention-side stream comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Input {
    /// Layer 0: the embedding, expanded into 4 streams by the first mix's `down`.
    Embed,
    /// The previous layer's residual + its pending MoE output (the deferred inject): the first
    /// mix's `down` runs in combine mode (or, at the PLE layer, `fn_ple_gate_combine` does).
    Pending,
    /// A materialized stream (a single-layer run fed from a capture): the first mix is a plain
    /// `mix` (the PLE layer uses `fn_ple_gate_x`).
    Materialized,
}

/// Everything that decides one layer's op list.
#[derive(Debug, Clone, Copy)]
pub struct LayerSpec {
    pub layer: u32,
    pub kind: LayerKind,
    pub ple: bool,
    pub input: Input,
    pub form: Form,
    pub tp: Tp,
    /// Rows of the step (tokens of the chunk, or batch x verify width).
    pub rows: u32,
    /// FN19: the QSA attention runs the SPARSE pipeline (the indexer's score + top-k per
    /// `saber_kernels::fn19::SEL_ROWS`-row slice, the union key list, the gather attention) instead
    /// of FN6's dense body -- `saber_kernels::fn19::sparse_step` decides (a no-op on GDN layers).
    pub qsa_sparse: bool,
    /// FN30: the MoE block's train (`--fn-moe-train`): FN26's fused back half at a decode step of
    /// <= 64 rows ([`moe_block_fused`]), else FN4's seven launches. The MTP drafter's layer (bf16
    /// experts) is always [`MoeTrain::Legacy`].
    pub moe: MoeTrain,
}

/// FN30: which MoE train a block runs. `Legacy` (THE DEFAULT since FN30's hold: FN4's seven at
/// every form) or `Fused` (the opt-in arm, `--fn-moe-train fused`) at a decode / verify step of
/// <= `moe::BackTrain::MAX_TOKENS` rows: FN31's front half (`moe_front_*`: router GEMM + route) +
/// FN26's `moe_back_*` = 2 launches at T <= `moe::FrontTrain::MAX_TOKENS` (8), router GEMM +
/// `moe::route_symbol` + `moe_back_*` = 3 above; prefill chunks keep the seven. FN30 hold 1: bytes
/// identical, the target step faster at b1 / b8-verify, but the step at b8 SPEC 0 and the MTP
/// drafter round after a fused step SLOWER (LEDGER FN30) -- so not the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MoeTrain {
    Fused,
    #[default]
    Legacy,
}

impl std::str::FromStr for MoeTrain {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim() {
            "fused" => Ok(MoeTrain::Fused),
            "legacy" => Ok(MoeTrain::Legacy),
            t => Err(format!("--fn-moe-train fused | legacy, not {t}")),
        }
    }
}

impl std::fmt::Display for MoeTrain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            MoeTrain::Fused => "fused",
            MoeTrain::Legacy => "legacy",
        })
    }
}

/// FN30: whether a (target) MoE block of a `decode` step over `rows` rows runs the fused back half
/// -- the ONE predicate the executor and the declared counts share.
#[must_use]
pub fn moe_block_fused(train: MoeTrain, decode: bool, rows: u32) -> bool {
    train == MoeTrain::Fused && decode && (1..=moe::BackTrain::MAX_TOKENS).contains(&rows)
}

/// FN30 + FN31: whether a fused block also runs FN31's fused FRONT half (router GEMM + route in
/// one launch) -- at T <= `moe::FrontTrain::MAX_TOKENS` (measured: it loses from T 12).
#[must_use]
pub fn moe_block_front(train: MoeTrain, decode: bool, rows: u32) -> bool {
    moe_block_fused(train, decode, rows) && rows <= moe::FrontTrain::MAX_TOKENS
}

/// The MoE block's launches under `train` at a step form over `rows` rows.
#[must_use]
pub fn moe_block_launches(train: MoeTrain, form: Form, rows: u32) -> u32 {
    if moe_block_front(train, matches!(form, Form::Decode { .. }), rows) {
        MOE_BLOCK_LAUNCHES_FRONT
    } else if moe_block_fused(train, matches!(form, Form::Decode { .. }), rows) {
        MOE_BLOCK_LAUNCHES_FUSED
    } else {
        MOE_BLOCK_LAUNCHES
    }
}

/// The MoE block's launches: FN4's train, which carries the shared-gate GEMV since FN9b (FN9's
/// finding; the constant itself says 7 now, so no `+ 1` here).
pub const MOE_BLOCK_LAUNCHES: u32 = moe::LAUNCHES_PER_MOE_BLOCK as u32;
/// FN30: the fused train's (router GEMM, route, `moe_back_*`).
pub const MOE_BLOCK_LAUNCHES_FUSED: u32 = moe::LAUNCHES_PER_MOE_BLOCK_FUSED as u32;
/// FN30 + FN31: the two-launch train's (`moe_front_*`, `moe_back_*`).
pub const MOE_BLOCK_LAUNCHES_FRONT: u32 = moe::LAUNCHES_PER_MOE_BLOCK_FRONT as u32;
/// The GDN core's launches at a prefill chunk: conv prep, KKT solve, chunk scan (the 27B's).
pub const GDN_PREFILL_CORE_LAUNCHES: u32 = 3;
/// Reduces per layer at TP=2 (post-attention, post-MoE).
pub const AR_PER_LAYER_TP2: u32 = 2;
/// The prior fork's launches per step on this model (plan §4, `KERNEL_HITLIST.md`).
pub const PRIOR_FORK_LAUNCHES_PER_STEP: u32 = 3_490;

fn op(name: String, group: Group, kernel: impl Into<String>, launches: u32) -> FnOp {
    FnOp {
        name,
        group,
        kernel: kernel.into(),
        launches,
    }
}

fn mix_ops(
    r: &FlashNextRules,
    l: u32,
    side: &str,
    mode: MixMode,
    out: MixOut,
    m: u32,
) -> Vec<FnOp> {
    let p = r
        .mix_plan(mode, out, m.max(1))
        .expect("an HC band holds every M <= 2048");
    vec![
        op(format!("l{l}.hc.{side}.down"), Group::Hc, p.down.symbol, 1),
        op(format!("l{l}.hc.{side}.up"), Group::Hc, p.up.symbol, 1),
    ]
}

fn tp_suffix(tp: Tp) -> &'static str {
    match tp {
        Tp::One => "tp1",
        Tp::Two => "tp2",
    }
}

/// The op list of one decoder layer.
#[must_use]
pub fn layer_ops(r: &FlashNextRules, s: &LayerSpec) -> Vec<FnOp> {
    let l = s.layer;
    let m = s.rows.max(1);
    let tp = tp_suffix(s.tp);
    let mut v = Vec::new();
    // ---- the attention-side stream + mix (PLE first at its layer)
    if s.ple {
        v.push(op(
            format!("l{l}.ple.key"),
            Group::Ple,
            "bgemm:ple.key_proj",
            1,
        ));
        v.push(op(
            format!("l{l}.ple.value"),
            Group::Ple,
            "bgemm:ple.value_proj",
            1,
        ));
        let gate = if s.input == Input::Pending {
            "fn_ple_gate_combine"
        } else {
            "fn_ple_gate_x"
        };
        v.push(op(format!("l{l}.ple.gate"), Group::Ple, gate, 1));
        v.push(op(format!("l{l}.ple.conv"), Group::Ple, "fn_ple_conv", 1));
        debug_assert_eq!(4, fn6::ple_launches(PleSource::CkptBf16));
    }
    let first_mode = match (s.input, s.ple) {
        (Input::Embed, _) => MixMode::Expand,
        (Input::Pending, false) => MixMode::Combine,
        _ => MixMode::Mix,
    };
    v.extend(mix_ops(r, l, "attn", first_mode, MixOut::Bf16, m));
    // ---- the mixer
    match s.kind {
        LayerKind::Gdn => {
            v.push(op(
                format!("l{l}.gdn.in_proj"),
                Group::Proj,
                format!("bgemm:gdn.in_proj.{tp}"),
                1,
            ));
            match s.form {
                Form::Prefill => {
                    for (n, k) in [
                        ("conv_prep", "saber_gdn_conv_prep_w4_h128_hpb8_ab1"),
                        ("kkt", "saber_gdn_kkt_solve_k128_c64_r3"),
                        // FN9b: the exact-gate scan (flashnext_gdn). The certified saber scan's
                        // split-and-clamped gate form attenuates Flash-Next's > 160-nat chunks.
                        ("scan", "fn_gdn_chunk_scan_xg_f16s_k128_v128_c64"),
                    ] {
                        v.push(op(format!("l{l}.gdn.{n}"), Group::Core, k, 1));
                    }
                }
                Form::Decode { batch, .. } => {
                    if batch <= 1 {
                        v.push(op(
                            format!("l{l}.gdn.fused_update"),
                            Group::Core,
                            "saber_gdn_fused_update_f16s_w4k128v128_ab1",
                            1,
                        ));
                    } else {
                        v.push(op(
                            format!("l{l}.gdn.conv_update"),
                            Group::Core,
                            "saber_gdn_conv_update",
                            1,
                        ));
                        v.push(op(
                            format!("l{l}.gdn.recurrent"),
                            Group::Core,
                            "saber_gdn_recurrent_update_f16s",
                            1,
                        ));
                    }
                }
            }
            v.push(op(
                format!("l{l}.gdn.norm"),
                Group::Core,
                r.kernels.gdn_norm_bf16_sigmoid.clone(),
                1,
            ));
            v.push(op(
                format!("l{l}.gdn.out_proj"),
                Group::Proj,
                format!("bgemm:gdn.out_proj.{tp}"),
                1,
            ));
        }
        LayerKind::Qsa => {
            let form = match s.form {
                Form::Prefill => AttnForm::Prefill,
                Form::Decode { attn, .. } => attn,
            };
            v.push(op(
                format!("l{l}.qsa.proj"),
                Group::Proj,
                format!("bgemm:qsa.proj.{tp}"),
                1,
            ));
            v.push(op(
                format!("l{l}.qsa.qk_norm_rope"),
                Group::Core,
                // FN21's byte-identical instantiation (the engine's since FN24)
                "fn_qk_norm_rope_h256_rot64_mi1_v2",
                1,
            ));
            v.push(op(
                format!("l{l}.qsa.idx"),
                Group::Core,
                "fn_idx_q_compress_h128_v2",
                1,
            ));
            v.push(op(
                format!("l{l}.qsa.kv_append"),
                Group::Core,
                "fn_kv_append_ring_h256_bf16",
                1,
            ));
            if s.qsa_sparse {
                // FN19: score + top-k per slice, the union, the gather attention (+ FN6's combine at
                // a decode step: the sparse decode is always the PAIR form)
                let sl = fn19::sel_slices(s.rows);
                v.push(op(
                    format!("l{l}.qsa.sel.score"),
                    Group::Core,
                    fn19::SCORE_SYMBOL,
                    sl,
                ));
                v.push(op(
                    format!("l{l}.qsa.sel.topk"),
                    Group::Core,
                    fn19::TOPK_SYMBOL,
                    sl,
                ));
                v.push(op(
                    format!("l{l}.qsa.sel.union"),
                    Group::Core,
                    fn19::UNION_SYMBOL,
                    1,
                ));
                match form {
                    AttnForm::Prefill => v.push(op(
                        format!("l{l}.qsa.attn"),
                        Group::Core,
                        fn19::PREFILL_SYMBOL,
                        1,
                    )),
                    AttnForm::DecodeDeclc | AttnForm::DecodePair => {
                        v.push(op(
                            format!("l{l}.qsa.attn"),
                            Group::Core,
                            fn19::DECODE_SYMBOL,
                            1,
                        ));
                        v.push(op(
                            format!("l{l}.qsa.attn_combine"),
                            Group::Core,
                            fn6::COMBINE_SYMBOL,
                            1,
                        ));
                    }
                }
            } else {
                match form {
                    AttnForm::Prefill => v.push(op(
                        format!("l{l}.qsa.attn"),
                        Group::Core,
                        fn6::PREFILL_SYMBOL,
                        1,
                    )),
                    AttnForm::DecodeDeclc => {
                        v.push(op(
                            format!("l{l}.qsa.attn"),
                            Group::Core,
                            "fn_attn_declc_h256_gqa12_bf16kv",
                            1,
                        ));
                    }
                    AttnForm::DecodePair => {
                        v.push(op(
                            format!("l{l}.qsa.attn"),
                            Group::Core,
                            "fn_attn_decode_h256_gqa12_bf16kv",
                            1,
                        ));
                        v.push(op(
                            format!("l{l}.qsa.attn_combine"),
                            Group::Core,
                            fn6::COMBINE_SYMBOL,
                            1,
                        ));
                    }
                }
            }
            v.push(op(
                format!("l{l}.qsa.gate"),
                Group::Core,
                r.kernels.attn_gate_bf16.clone(),
                1,
            ));
            v.push(op(
                format!("l{l}.qsa.o_proj"),
                Group::Proj,
                format!("bgemm:gdn.out_proj.{tp}"),
                1,
            ));
        }
    }
    if s.tp == Tp::Two {
        v.push(op(
            format!("l{l}.ar.attn"),
            Group::Ar,
            "saber_ar_wht6 (the 27B's band)",
            1,
        ));
    }
    // ---- the MoE side (the attention output's inject deferred into this down)
    v.extend(mix_ops(r, l, "moe", MixMode::Combine, MixOut::Q8, m));
    let gu = moe::grouped_symbol(
        moe::MoeGemm::GateUp,
        moe::measured_cell(moe::MoeGemm::GateUp, m),
    );
    let dn = moe::grouped_symbol(
        moe::MoeGemm::Down,
        moe::measured_cell(moe::MoeGemm::Down, m),
    );
    let fused = moe_block_fused(s.moe, matches!(s.form, Form::Decode { .. }), m);
    let train: Vec<(&str, String)> = if fused {
        // FN30: FN26's back half (gate GEMV + gate_up + silu + down + combine in one launch)
        let back =
            moe::BackTrain::for_tokens(m, moe::BACK_PREFETCH_BODY == moe::DecodeBody::Prefetch)
                .map_or_else(|| "moe_back".to_string(), |b| b.symbol());
        match moe::FrontTrain::for_tokens(m) {
            // FN31: router GEMM + route in one launch at T <= 8
            Some(f) => vec![("front", f.symbol()), ("back", back)],
            None => vec![
                ("router", "bgemm:moe.router".to_string()),
                ("route", moe::route_symbol(m).to_string()),
                ("back", back),
            ],
        }
    } else {
        vec![
            ("router", "bgemm:moe.router".to_string()),
            (
                "shared_gate",
                "fn_bgemm_sk_m16 (C[1,T] = w_sg . X^T)".to_string(),
            ),
            ("route", moe::ROUTE_SYMBOL.to_string()),
            ("gate_up", gu),
            ("silu", moe::SILU_SYMBOL.to_string()),
            ("down", dn),
            ("combine", moe::COMBINE_SYMBOL.to_string()),
        ]
    };
    debug_assert_eq!(train.len() as u32, moe_block_launches(s.moe, s.form, m));
    for (n, k) in train {
        v.push(op(format!("l{l}.moe.{n}"), Group::Moe, k, 1));
    }
    if s.tp == Tp::Two {
        v.push(op(
            format!("l{l}.ar.moe"),
            Group::Ar,
            "saber_ar_wht6 (the 27B's band)",
            1,
        ));
    }
    v
}

/// Launches of one layer, from its op list.
#[must_use]
pub fn launches(ops: &[FnOp]) -> u32 {
    ops.iter().map(|o| o.launches).sum()
}

/// A whole target step: embedding, 48 layers, the final mixer, the head.
#[derive(Debug, Clone)]
pub struct StepGraph {
    pub pre: Vec<FnOp>,
    pub layers: Vec<Vec<FnOp>>,
    pub post: Vec<FnOp>,
}

impl StepGraph {
    /// Every launch of the step.
    #[must_use]
    pub fn launches(&self) -> u32 {
        launches(&self.pre)
            + self.layers.iter().map(|l| launches(l)).sum::<u32>()
            + launches(&self.post)
    }
}

/// The step graph for `layers` (the config's schedule) at `ple_layer`.
#[must_use]
pub fn step_graph(
    r: &FlashNextRules,
    kinds: &[LayerKind],
    ple_layer: u32,
    form: Form,
    tp: Tp,
    rows: u32,
) -> StepGraph {
    let pre = vec![op("embed".into(), Group::Head, "saber_embed_bf16", 1)];
    let layers = kinds
        .iter()
        .enumerate()
        .map(|(l, &kind)| {
            let l = l as u32;
            layer_ops(
                r,
                &LayerSpec {
                    layer: l,
                    kind,
                    ple: l == ple_layer,
                    input: if l == 0 { Input::Embed } else { Input::Pending },
                    form,
                    tp,
                    rows,
                    qsa_sparse: false,
                    moe: MoeTrain::default(),
                },
            )
        })
        .collect();
    let mut post = mix_ops(
        r,
        kinds.len() as u32,
        "final",
        MixMode::Final,
        MixOut::Bf16,
        rows,
    );
    for o in &mut post {
        o.name = o
            .name
            .replacen(&format!("l{}.hc.final", kinds.len()), "final_mixer", 1);
    }
    let head_launches = match tp {
        // the vocab-parallel head: at TP=1 both shards run on the one card (two launches)
        Tp::One => 2,
        Tp::Two => 1,
    };
    post.push(op(
        "head".into(),
        Group::Head,
        "bgemm:head.tp2",
        head_launches,
    ));
    StepGraph { pre, layers, post }
}

/// The graph as JSON for the card harness (`tools/capture/fn9/fn9_layers.py` asserts its own
/// launches per layer against `launches`).
#[must_use]
pub fn to_json(g: &StepGraph) -> serde_json::Value {
    let ops = |v: &[FnOp]| -> Vec<serde_json::Value> {
        v.iter()
            .map(|o| {
                serde_json::json!({"name": o.name, "group": o.group.name(), "kernel": o.kernel,
                                   "launches": o.launches})
            })
            .collect()
    };
    serde_json::json!({
        "pre": ops(&g.pre),
        "layers": g.layers.iter().map(|l| serde_json::json!({"ops": ops(l), "launches": launches(l)})).collect::<Vec<_>>(),
        "post": ops(&g.post),
        "launches": g.launches(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules() -> FlashNextRules {
        FlashNextRules::prod().expect("rules")
    }

    fn kinds() -> Vec<LayerKind> {
        (0..48)
            .map(|l| {
                if l % 4 == 3 {
                    LayerKind::Qsa
                } else {
                    LayerKind::Gdn
                }
            })
            .collect()
    }

    fn spec(layer: u32, kind: LayerKind, ple: bool, form: Form, tp: Tp) -> LayerSpec {
        LayerSpec {
            layer,
            kind,
            ple,
            input: Input::Pending,
            form,
            tp,
            rows: 64,
            qsa_sparse: false,
            moe: MoeTrain::Legacy,
        }
    }

    const DEC1: Form = Form::Decode {
        batch: 1,
        attn: AttnForm::DecodePair,
    };

    #[test]
    fn the_layer_counts_are_the_kernel_planes_constants() {
        let r = rules();
        let hc = r.hc_launches_per_layer(false);
        let moe_b = moe::LAUNCHES_PER_MOE_BLOCK as u32;
        assert_eq!(
            MOE_BLOCK_LAUNCHES, 7,
            "router, shared-gate GEMV, route, gate_up, silu, down, combine"
        );
        for (form, core) in [(Form::Prefill, GDN_PREFILL_CORE_LAUNCHES), (DEC1, 1)] {
            let gdn = layer_ops(&r, &spec(0, LayerKind::Gdn, false, form, Tp::One));
            // mix 2 + in_proj + core + norm + out_proj + mix 2 + MoE
            assert_eq!(launches(&gdn), hc + 1 + core + 1 + 1 + moe_b, "{gdn:#?}");
        }
        let two = layer_ops(
            &r,
            &spec(
                0,
                LayerKind::Gdn,
                false,
                Form::Decode {
                    batch: 4,
                    attn: AttnForm::DecodePair,
                },
                Tp::One,
            ),
        );
        assert_eq!(
            launches(&two),
            hc + 1 + 2 + 1 + 1 + moe_b,
            "the decode pair above b1"
        );
        for form in [
            (Form::Prefill, AttnForm::Prefill),
            (
                Form::Decode {
                    batch: 1,
                    attn: AttnForm::DecodeDeclc,
                },
                AttnForm::DecodeDeclc,
            ),
            (DEC1, AttnForm::DecodePair),
        ] {
            let q = layer_ops(&r, &spec(3, LayerKind::Qsa, false, form.0, Tp::One));
            assert_eq!(
                launches(&q),
                hc + fn6::qsa_attn_launches(form.1) as u32 + moe_b,
                "{q:#?}"
            );
        }
        let p = layer_ops(&r, &spec(1, LayerKind::Gdn, true, Form::Prefill, Tp::One));
        let g = layer_ops(&r, &spec(1, LayerKind::Gdn, false, Form::Prefill, Tp::One));
        assert_eq!(
            launches(&p),
            launches(&g) + fn6::ple_launches(PleSource::CkptBf16) as u32,
            "the PLE's combine rides fn_ple_gate_combine; the mix after it is a plain mix"
        );
        assert!(p.iter().any(|o| o.kernel == "fn_ple_gate_combine"));
        assert!(p.iter().any(|o| o.kernel.starts_with("fn_hc_down_mix_")));
        let t2 = layer_ops(&r, &spec(5, LayerKind::Gdn, false, Form::Prefill, Tp::Two));
        assert_eq!(launches(&t2), launches(&g) + AR_PER_LAYER_TP2);
    }

    #[test]
    fn the_fused_moe_train_is_three_launches_at_decode_steps_of_at_most_64_rows() {
        let r = rules();
        assert_eq!(MOE_BLOCK_LAUNCHES_FUSED, 3, "router, route, moe_back");
        assert_eq!(MOE_BLOCK_LAUNCHES_FRONT, 2, "moe_front, moe_back");
        for (form, rows, want) in [
            (DEC1, 1, 2),
            (DEC1, 4, 2),
            (DEC1, 8, 2),
            (DEC1, 9, 3),
            (DEC1, 40, 3),
            (DEC1, 64, 3),
            (DEC1, 65, 7),
            (Form::Prefill, 1, 7),
            (Form::Prefill, 64, 7),
            (Form::Prefill, 2048, 7),
        ] {
            assert_eq!(
                moe_block_launches(MoeTrain::Fused, form, rows),
                want,
                "{form:?} {rows}"
            );
            assert_eq!(moe_block_launches(MoeTrain::Legacy, form, rows), 7);
            for (l, kind) in [(0, LayerKind::Gdn), (3, LayerKind::Qsa)] {
                let mut f = spec(l, kind, false, form, Tp::Two);
                f.rows = rows;
                f.moe = MoeTrain::Fused;
                let mut g = f;
                g.moe = MoeTrain::Legacy;
                let (fo, go) = (layer_ops(&r, &f), layer_ops(&r, &g));
                assert_eq!(launches(&go) - launches(&fo), 7 - want, "{form:?} {rows}");
                let back: Vec<_> = fo
                    .iter()
                    .filter(|o| o.name.ends_with(".moe.back"))
                    .collect();
                assert_eq!(back.len(), usize::from(want < 7));
                if want == 3 {
                    assert!(back[0].kernel.starts_with("moe_back_gu"), "{back:?}");
                    let route = fo.iter().find(|o| o.name.ends_with(".moe.route")).unwrap();
                    assert_eq!(route.kernel, moe::route_symbol(rows));
                }
                if want == 2 {
                    let front = fo.iter().find(|o| o.name.ends_with(".moe.front")).unwrap();
                    assert!(
                        front.kernel.starts_with("moe_front_e512_nw8_s"),
                        "{front:?}"
                    );
                }
            }
        }
        assert_eq!("legacy".parse::<MoeTrain>(), Ok(MoeTrain::Legacy));
        assert_eq!(MoeTrain::default().to_string(), "legacy");
        assert!("x".parse::<MoeTrain>().is_err());
    }

    #[test]
    fn the_numbers_the_report_states() {
        let r = rules();
        let one = |kind, ple, form, tp| launches(&layer_ops(&r, &spec(1, kind, ple, form, tp)));
        assert_eq!(one(LayerKind::Gdn, false, Form::Prefill, Tp::One), 17);
        assert_eq!(one(LayerKind::Qsa, false, Form::Prefill, Tp::One), 18);
        assert_eq!(one(LayerKind::Gdn, true, Form::Prefill, Tp::One), 21);
        assert_eq!(one(LayerKind::Gdn, false, DEC1, Tp::Two), 17);
        assert_eq!(one(LayerKind::Qsa, false, DEC1, Tp::Two), 21);
        let k = kinds();
        let pre = step_graph(&r, &k, 1, Form::Prefill, Tp::One, 2048);
        // embed 1 + 36 GDN x 17 + 12 QSA x 18 + PLE 4 + final mixer 2 + head 2 (both vocab shards)
        assert_eq!(pre.launches(), 1 + 36 * 17 + 12 * 18 + 4 + 2 + 2);
        assert_eq!(pre.launches(), 837);
        let dec = step_graph(&r, &k, 1, DEC1, Tp::Two, 5);
        // TP=2 decode b1 (verify 5 rows): GDN 15 + 2 AR, QSA 19 + 2 AR (the pair form) with the
        // seven-launch MoE (the step graph's default train; FN30's fused arm takes 5 a layer off at
        // T <= 8)
        assert_eq!(dec.launches(), 1 + 36 * 17 + 12 * 21 + 4 + 2 + 1);
        assert_eq!(dec.launches(), 872);
        assert!(
            dec.launches() * 3 < PRIOR_FORK_LAUNCHES_PER_STEP,
            "under a third of the fork's"
        );
        let hc: u32 = dec
            .layers
            .iter()
            .flatten()
            .chain(&dec.post)
            .filter(|o| o.group == Group::Hc)
            .map(|o| o.launches)
            .sum();
        assert_eq!(
            hc,
            48 * 4 + 2,
            "FN5's HC law minus the combine the PLE absorbed"
        );
    }

    #[test]
    fn the_first_mix_mode_follows_the_input() {
        let r = rules();
        let mut s = spec(0, LayerKind::Gdn, false, Form::Prefill, Tp::One);
        s.input = Input::Embed;
        assert!(layer_ops(&r, &s)[0]
            .kernel
            .starts_with("fn_hc_down_expand_"));
        s.input = Input::Materialized;
        assert!(layer_ops(&r, &s)[0].kernel.starts_with("fn_hc_down_mix_"));
        s.input = Input::Pending;
        assert!(layer_ops(&r, &s)[0]
            .kernel
            .starts_with("fn_hc_down_combine_"));
        let moe_side: Vec<_> = layer_ops(&r, &s)
            .into_iter()
            .filter(|o| o.name.contains(".hc.moe."))
            .collect();
        assert!(moe_side[0].kernel.starts_with("fn_hc_down_combine_"));
        assert!(
            moe_side[1].kernel.starts_with("fn_hc_up_q8_"),
            "{moe_side:?}"
        );
        s.ple = true;
        s.layer = 1;
        s.input = Input::Materialized;
        assert!(layer_ops(&r, &s)
            .iter()
            .any(|o| o.kernel == "fn_ple_gate_x"));
    }

    #[test]
    fn the_gdn_norm_is_the_sigmoid_twin() {
        let r = rules();
        let g = layer_ops(&r, &spec(0, LayerKind::Gdn, false, Form::Prefill, Tp::One));
        let n = g
            .iter()
            .find(|o| o.name.ends_with(".gdn.norm"))
            .expect("norm");
        assert_eq!(n.kernel, "fn_gdn_norm_bf16_sig_t256");
        assert_ne!(
            n.kernel, r.kernels.gdn_norm_bf16,
            "the 27B's silu twin is the control"
        );
    }

    #[test]
    fn every_bgemm_family_the_graph_names_is_a_rule_shape() {
        let r = rules();
        let k = kinds();
        for tp in [Tp::One, Tp::Two] {
            for m in [1, 5, 8, 2048] {
                let g = step_graph(&r, &k, 1, Form::Prefill, tp, m);
                for o in g.layers.iter().flatten().chain(&g.post) {
                    if let Some(shape) = o.kernel.strip_prefix("bgemm:") {
                        assert!(
                            r.bgemm.iter().any(|b| b.shape == shape),
                            "{} names `{shape}`",
                            o.name
                        );
                    }
                }
            }
        }
        let j = to_json(&step_graph(&r, &k, 1, Form::Prefill, Tp::One, 16));
        assert_eq!(j["layers"].as_array().unwrap().len(), 48);
        assert_eq!(j["layers"][1]["launches"], 21);
    }
}
