//! INK2: the engine's per-rank memory law for an `inkling2_loop` boot (`docs/INKLING2.md` §5, §9):
//! the per-visit ARENA (every buffer a visit's op list writes, by name, sized by the rows of the
//! widest pass), the persistent per-sequence STATE (8 paged global KV pools + 40 local rings of
//! `window` rows + 192 short-conv states a sequence, one spare slot for padded rows: FN17's pad
//! law) and the fixed runtime classes. A declaration: it never allocates; the CLI's dry run bills
//! it into a `saber_mem::ledger::Ledger` against the card's budget, and the INK4 executor carves
//! exactly these buffers.
//!
//! The TP=2 split (one rank): query heads and KV heads halve (12 / 4), so do the k / v conv
//! channels and the dynamic-conv taps; the attention and MLP short convs run over the full hidden
//! (after the all-reduce) and are REPLICATED; experts split by EP (64 of 128 a layer).

use super::config::{Inkling2Config, Visit};

const ALIGN: u64 = 256;
const F32: u64 = 4;
const BF16: u64 = 2;

/// The boot's geometry (what the flags choose).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EngineGeom {
    /// Tensor-parallel width (1 or 2).
    pub tp: u32,
    /// Sequences the state pools hold (the decode batch).
    pub max_seqs: u32,
    /// Per-sequence context the global KV pools hold.
    pub max_ctx: u64,
    /// Rows of the widest prefill chunk.
    pub max_chunk: u64,
    /// Rows of a global KV page.
    pub page_rows: u64,
    /// Split-K partitions of a decode attention row (the partials' width).
    pub decode_splits: u64,
}

impl EngineGeom {
    /// Rows of the widest decode pass (one row a sequence at SPEC 0).
    #[must_use]
    pub fn decode_rows(&self) -> u64 {
        u64::from(self.max_seqs)
    }

    /// State slots: one a sequence + the pad law's spare.
    #[must_use]
    pub fn slots(&self) -> u64 {
        u64::from(self.max_seqs) + 1
    }
}

/// One rank's share of the model's widths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RankDims {
    pub hidden: u64,
    /// Query heads on this rank.
    pub heads: u64,
    /// KV heads on this rank.
    pub kv_heads: u64,
    pub head_dim: u64,
    pub d_rel: u64,
    pub rel_extent: u64,
    /// Dynamic-conv taps of one of k / v on this rank (KV heads x lags).
    pub dyn_taps: u64,
    pub experts_total: u64,
    pub top_k: u64,
    pub inter: u64,
    /// Vocabulary rows of this rank's head half (padded vocab / tp).
    pub vocab: u64,
    pub window: u64,
    pub conv_rows: u64,
}

impl RankDims {
    #[must_use]
    pub fn of(cfg: &Inkling2Config, tp: u32) -> Self {
        let t = u64::from(tp);
        Self {
            hidden: u64::from(cfg.hidden_size),
            heads: u64::from(cfg.num_attention_heads) / t,
            kv_heads: u64::from(cfg.num_key_value_heads) / t,
            head_dim: u64::from(cfg.head_dim),
            d_rel: u64::from(cfg.d_rel),
            rel_extent: u64::from(cfg.rel_extent),
            dyn_taps: u64::from(cfg.num_key_value_heads) / t * u64::from(cfg.sconv_kernel_size),
            experts_total: u64::from(cfg.n_routed_experts),
            top_k: u64::from(cfg.num_experts_per_tok),
            inter: u64::from(cfg.intermediate_size),
            vocab: u64::from(cfg.vocab_size) / t,
            window: u64::from(cfg.sliding_window_size),
            conv_rows: u64::from(cfg.sconv_kernel_size - 1),
        }
    }

    /// Query channels on this rank.
    #[must_use]
    pub fn q(&self) -> u64 {
        self.heads * self.head_dim
    }

    /// K (or V) channels on this rank.
    #[must_use]
    pub fn kvc(&self) -> u64 {
        self.kv_heads * self.head_dim
    }
}

/// The pass a buffer is sized for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass {
    Prefill,
    Decode,
}

/// One named arena buffer: `per_row` bytes for every row of the pass + `fixed` bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Buf {
    pub name: &'static str,
    pub per_row: u64,
    pub fixed: u64,
}

impl Buf {
    /// Bytes at `rows`, 256-B aligned.
    #[must_use]
    pub fn bytes(&self, rows: u64) -> u64 {
        (self.per_row * rows + self.fixed).div_ceil(ALIGN) * ALIGN
    }
}

const fn buf(name: &'static str, per_row: u64, fixed: u64) -> Buf {
    Buf {
        name,
        per_row,
        fixed,
    }
}

/// The per-token DEPTH state of one forward: live across every visit of the pass (never across
/// steps -- a token's depth state is rebuilt by its own forward): the partial `x`, the block list
/// (one entry a block boundary: 4 at most), the saved pre-loop pair, the previous pass's summary
/// and the four messages of a core visit.
#[must_use]
pub fn depth_buffers(cfg: &Inkling2Config, d: &RankDims) -> Vec<Buf> {
    let h = d.hidden * F32;
    let blocks = u64::from(cfg.num_hidden_layers.div_ceil(cfg.attn_res_block));
    let saved_blocks = u64::from(cfg.loop_cfg.core_start / cfg.attn_res_block);
    vec![
        buf("depth.x", h, 0),
        buf("depth.blocks", blocks * h, 0),
        buf("depth.saved", (1 + saved_blocks) * h, 0),
        // the previous pass's summary: the blocks it appended after its first (core / block - 1)
        // + its output
        buf(
            "depth.prev",
            u64::from(
                (cfg.loop_cfg.core_end - cfg.loop_cfg.core_start) / cfg.attn_res_block - 1 + 1,
            ) * h,
            0,
        ),
        buf("depth.msgs", 4 * h, 0),
    ]
}

/// The buffers ONE visit of kind `v` writes (reused by the next visit: visits run in order).
#[must_use]
/// `splits` = the decode attention's split-K partitions (prefill: none).
pub fn visit_buffers(d: &RankDims, v: &Visit, pass: Pass, splits: u64) -> Vec<Buf> {
    let h = d.hidden;
    let three_inputs = !v.local || v.messages;
    let n_in = if three_inputs { 3 } else { 1 };
    let mut b = vec![
        // the attention-input depth mix (f32, 1 or 3 queries) and the gain-normed GEMM inputs
        buf("visit.mix_in", n_in * h * F32, 0),
        buf("visit.qkv_in_bf16", n_in * h * BF16, 0),
    ];
    if v.messages {
        // three message gates' 64-rank codes + the gated f32 parts
        b.push(buf("visit.msg_gate", 4 * 64 * F32 + 3 * h * F32, 0));
    }
    let q = d.q();
    let kvc = d.kvc();
    let dyn_taps = if v.local { 0 } else { 2 * d.dyn_taps };
    b.extend([
        // projections (f32 out): q, gate, r, sink | k, v (+ the dynamic taps of global layers)
        buf(
            "visit.proj",
            (q + q + d.heads * d.d_rel + d.heads + 2 * kvc + dyn_taps) * F32,
            0,
        ),
        // the post-conv k / v rows written to the cache are in the pools; q (clipped, tau) here
        buf("visit.q_post", q * BF16, 0),
        // r . P: [rows, H, extent] f32 -- the bias every query row reads by distance
        buf("visit.rel_bias", d.heads * d.rel_extent * F32, 0),
        buf("visit.attn_o", q * F32, 0),
        buf("visit.gated_bf16", q * BF16, 0),
        // the row-parallel o_proj partial and the reduce's result
        buf("visit.oproj", h * F32, 0),
        // attn sconv -> residual; MLP depth mix; gain norm -> e4m3 + per-32 scales
        buf("visit.resid", h * F32, 0),
        buf("visit.mlp_in", h * F32, 0),
        buf("visit.u_q8", h + h / 32 * F32, 0),
        // router: sigmoid scores, top-k ids + weights
        buf(
            "visit.route",
            d.experts_total * F32 + d.top_k * (4 + F32),
            d.experts_total * 4,
        ),
        // EP worst case: every one of a row's top-k experts on this rank
        buf("visit.moe_gate_up", d.top_k * 2 * d.inter * F32, 0),
        buf(
            "visit.moe_act_q8",
            d.top_k * (d.inter + d.inter / 32 * F32),
            0,
        ),
        buf("visit.moe_down", d.top_k * h * F32, 0),
        buf("visit.moe_out", h * F32, 0),
    ]);
    if v.messages {
        b.push(buf("visit.msg_gate_mlp", 64 * F32 + h * F32, 0));
    }
    if pass == Pass::Decode {
        // split-K attention partials: per head per split (o, max, sum)
        b.push(buf(
            "visit.attn_partials",
            splits * d.heads * (d.head_dim + 2) * F32,
            0,
        ));
    }
    b
}

/// The head after the last visit: final mix + norm (f32 / bf16), the vocab half's logits (f32,
/// softcapped in place) and the top-20 a row (the pair's merge).
#[must_use]
pub fn head_buffers(d: &RankDims) -> Vec<Buf> {
    vec![
        buf("head.final", d.hidden * (F32 + BF16), 0),
        buf("head.logits", d.vocab * F32, 0),
        buf("head.top20", 20 * 8, 0),
    ]
}

/// The arena of a pass at `rows`: depth + the widest visit kind + the head (the head runs after
/// the last visit and reuses no visit buffer).
#[derive(Clone, Debug, PartialEq)]
pub struct ArenaPlan {
    pub pass: Pass,
    pub rows: u64,
    /// (name, bytes) of the depth buffers, the widest visit's buffers and the head's.
    pub bufs: Vec<(&'static str, u64)>,
    /// The visit index whose kind set the visit part.
    pub widest_visit: u32,
    pub bytes: u64,
}

#[must_use]
pub fn arena(
    cfg: &Inkling2Config,
    d: &RankDims,
    pass: Pass,
    rows: u64,
    head_rows: u64,
    splits: u64,
) -> ArenaPlan {
    let mut out: Vec<(&'static str, u64)> = depth_buffers(cfg, d)
        .iter()
        .map(|b| (b.name, b.bytes(rows)))
        .collect();
    let (wv, vb) = cfg
        .visits()
        .iter()
        .map(|v| {
            let bs: Vec<_> = visit_buffers(d, v, pass, splits)
                .iter()
                .map(|b| (b.name, b.bytes(rows)))
                .collect();
            (v.index, bs)
        })
        .max_by_key(|(i, bs)| (bs.iter().map(|x| x.1).sum::<u64>(), u32::MAX - *i))
        .expect("visits");
    out.extend(vb);
    out.extend(head_buffers(d).iter().map(|b| (b.name, b.bytes(head_rows))));
    let bytes = out.iter().map(|x| x.1).sum();
    ArenaPlan {
        pass,
        rows,
        bufs: out,
        widest_visit: wv,
        bytes,
    }
}

/// One named class of the rank's runtime bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Class {
    pub name: &'static str,
    pub bytes: u64,
    /// What it is, for the boot line.
    pub what: String,
}

/// The persistent state + runtime classes of one rank (weights excluded: the pack bills those).
#[derive(Clone, Debug, PartialEq)]
pub struct RankPlan {
    pub geom: EngineGeom,
    pub dims: RankDims,
    pub prefill: ArenaPlan,
    pub decode: ArenaPlan,
    pub classes: Vec<Class>,
    /// Global KV visits and local KV visits.
    pub global_visits: u32,
    pub local_visits: u32,
    /// Conv states a sequence (4 a visit).
    pub conv_states: u32,
    pub pages_per_seq: u64,
    pub pages: u64,
}

impl RankPlan {
    /// The plan of one rank.
    #[must_use]
    pub fn new(cfg: &Inkling2Config, g: EngineGeom) -> Self {
        let d = RankDims::of(cfg, g.tp);
        let prefill = arena(cfg, &d, Pass::Prefill, g.max_chunk, 1, 1);
        let decode = arena(
            cfg,
            &d,
            Pass::Decode,
            g.decode_rows(),
            g.decode_rows(),
            g.decode_splits.max(1),
        );
        let visits = cfg.visits();
        let global_visits = visits.iter().filter(|v| !v.local).count() as u32;
        let local_visits = visits.len() as u32 - global_visits;
        let kv_row = 2 * d.kvc() * BF16; // K + V, one row, one visit
        let pages_per_seq = g.max_ctx.div_ceil(g.page_rows);
        let pages = u64::from(g.max_seqs) * pages_per_seq + 1;
        let page_bytes = g.page_rows * kv_row;
        let conv_one = d.conv_rows * (2 * d.kvc() + 2 * d.hidden) * F32; // k, v (split) + attn, mlp (full)
        let arena_bytes = prefill.bytes.max(decode.bytes);
        let mut classes = vec![
            Class {
                name: "ink.kv.global",
                bytes: u64::from(global_visits) * pages * page_bytes,
                what: format!(
                    "{global_visits} paged pools x {pages} pages ({} x {pages_per_seq} + 1 spare) x {} rows x {} B",
                    g.max_seqs, g.page_rows, kv_row
                ),
            },
            Class {
                name: "ink.kv.table",
                bytes: (u64::from(global_visits) * g.slots() * pages_per_seq * 4).div_ceil(ALIGN)
                    * ALIGN,
                what: format!("block tables: {global_visits} x {} slots x {pages_per_seq} pages x 4 B", g.slots()),
            },
            Class {
                name: "ink.kv.local",
                bytes: u64::from(local_visits) * g.slots() * d.window * kv_row,
                what: format!(
                    "{local_visits} rings x {} slots x {} rows x {kv_row} B",
                    g.slots(),
                    d.window
                ),
            },
            Class {
                name: "ink.conv",
                bytes: visits.len() as u64 * g.slots() * conv_one,
                what: format!(
                    "{} states a sequence ({} visits x 4) x {} slots, f32 {} rows",
                    visits.len() * 4,
                    visits.len(),
                    g.slots(),
                    d.conv_rows
                ),
            },
            Class {
                name: "ink.arena",
                bytes: arena_bytes,
                what: format!(
                    "max(prefill {} rows {:.1} MiB, decode {} rows {:.1} MiB)",
                    prefill.rows,
                    mib(prefill.bytes),
                    decode.rows,
                    mib(decode.bytes)
                ),
            },
        ];
        if g.tp > 1 {
            // the gp reduce: this rank's scratch (the peer writes it) at the widest pass + flags
            classes.push(Class {
                name: "ink.ar",
                bytes: 2 * g.max_chunk.max(g.decode_rows()) * d.hidden * F32 + (64 << 10),
                what: "gp reduce scratch: 2 x rows x hidden x f32 + 64 KiB flags/seq".into(),
            });
        }
        Self {
            geom: g,
            dims: d,
            prefill,
            decode,
            classes,
            global_visits,
            local_visits,
            conv_states: visits.len() as u32 * 4,
            pages_per_seq,
            pages,
        }
    }

    /// Sum of the runtime classes.
    #[must_use]
    pub fn runtime_bytes(&self) -> u64 {
        self.classes.iter().map(|c| c.bytes).sum()
    }
}

/// Bytes as MiB.
#[must_use]
pub fn mib(b: u64) -> f64 {
    b as f64 / f64::from(1u32 << 20)
}
