//! INK2: the op list of one Inkling 2 visit at TP=2 / EP=2 (one rank), its kernel classes and its
//! launch law (`docs/INKLING2.md` §3, §7, §9). A declaration: the engine's emitter walks it, the
//! card harness counts launches against [`visit_launches`], and every op names whether a shipped
//! Saber body serves it (`Reuse`) or a new body is owed (`New`, INK3).

use super::config::{Inkling2Config, Visit};

/// Where an op's body comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// A shipped body (bf16 GEMM, grouped W4A8 MoE GEMM, the `gp` all-reduce, the MoE combine).
    Reuse,
    /// A body INK3 must write.
    New,
}

/// One launch of a visit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Op {
    pub name: &'static str,
    pub class: Class,
}

const fn op(name: &'static str, class: Class) -> Op {
    Op { name, class }
}

/// The ops of visit `v`, in order. `first` = visit 0 (no attention-input depth mix).
#[must_use]
pub fn visit_ops(v: &Visit) -> Vec<Op> {
    use Class::{New, Reuse};
    let mut o = Vec::new();
    // attention side: the input mix (+ messages + gains), or the embedding row at visit 0
    if v.index > 0 {
        o.push(op("depth_attend_in", New)); // rmsnorm states, 1 or 3 queries, softmax over depth
    }
    if v.messages {
        o.push(op("msg_attend", New)); // the previous pass's 4 messages (once a visit)
        o.push(op("msg_gate_qkv", New)); // three gates: concat-norm, 64-rank GEMV, tanh mix
    }
    o.push(op("in_norm", New)); // the gain norm(s) -> q / k / v inputs (fused with the mix later)
    let shared_input = v.local && !v.messages;
    if shared_input {
        o.push(op("proj_qgrskv_bf16", Reuse)); // one merged GEMM: q, gate, r, sink, k, v rows
    } else {
        o.push(op("proj_qgrs_bf16", Reuse)); // from the q input
        o.push(op("proj_k_bf16", Reuse)); // k (+ the dynamic taps' 16 rows when global)
        o.push(op("proj_v_bf16", Reuse));
    }
    o.push(op("kv_conv_post_append", New)); // conv (static | dynamic), l2 clip, v rmsnorm, q clip + tau, KV write
    o.push(op("rel_bias_gemm", New)); // r . P -> [T, H, 1024]
    o.push(op("attn_bias_sink", New)); // GQA 3, band bias, sink in the denominator, window | causal
    o.push(op("out_gate", New)); // o * sigmoid(gate)
    o.push(op("o_proj_bf16", Reuse)); // row-parallel
    o.push(op("ar_attn", Reuse));
    // MoE side
    o.push(op("sconv_resid_mlp_in", New)); // attn sconv, residual / block append, depth mix (1 query)
    if v.messages {
        o.push(op("msg_gate_mlp", New));
    }
    o.push(op("post_norm_route", New)); // gain norm + router GEMV + sigmoid/bias top-8 of 128
    o.push(op("moe_gate_up_w4a8", Reuse));
    o.push(op("moe_swiglu_softcap_quant", New));
    o.push(op("moe_down_w4a8", Reuse));
    o.push(op("moe_combine", Reuse));
    o.push(op("ar_moe", Reuse));
    o.push(op("mlp_sconv_resid", New));
    o
}

/// Launches a visit costs.
#[must_use]
pub fn visit_launches(v: &Visit) -> usize {
    visit_ops(v).len()
}

/// The head after the last visit: final depth mix + norm, head GEMM (vocab half), softcap, gather.
pub const HEAD_LAUNCHES: usize = 4;

/// Launches of a whole step (one forward) and the count of new-body launches.
#[must_use]
pub fn step_launches(cfg: &Inkling2Config) -> (usize, usize) {
    let mut all = HEAD_LAUNCHES + 1; // + the embedding gather + embed norm
    let mut new = 0;
    for v in cfg.visits() {
        let ops = visit_ops(&v);
        all += ops.len();
        new += ops.iter().filter(|o| o.class == Class::New).count();
    }
    (all, new)
}

/// The distinct new bodies INK3 owes (names, sorted).
#[must_use]
pub fn new_bodies(cfg: &Inkling2Config) -> Vec<&'static str> {
    let mut n: Vec<_> = cfg
        .visits()
        .iter()
        .flat_map(visit_ops)
        .filter(|o| o.class == Class::New)
        .map(|o| o.name)
        .collect();
    n.sort_unstable();
    n.dedup();
    n
}
