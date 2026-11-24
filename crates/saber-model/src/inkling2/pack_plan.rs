//! The INK1 TP=2 / EP=2 pack plan of an `inkling2_loop` checkpoint: the record SET, the source
//! kind that builds each record, and the per-rank byte ledger (`docs/INKLING2.md` §5, §7).
//!
//! `saber-inkling2-packplan --checkpoint DIR` prints it as JSON; `tools/pack/build_inkling2_pack.py`
//! does the bytes and `tools/pack/check_inkling2_pack.py` checks what landed.
//!
//! * Experts: rank `r` owns experts `[64 r, 64 r + 64)` of every layer (EP=2), one fixed-size
//!   record per (layer, expert) in the Flash-Next xtier record layout (gate rows first in gate_up).
//!   The source kind per layer is the plan's choice: `nvfp4_requant_mxfp4` / `mxfp8_requant_mxfp4`
//!   (the 27B's requant: exact f32 dequant -> MXFP4 at the `mse` block exponent; one record format
//!   for all 24 layers, the served 4-bit class) or `nvfp4_native` (FN144's v2 record, no rounding;
//!   FN148's bodies read it). MXFP8 has no native record yet (no body reads e8m0 per 32).
//! * Dense: bf16 / f32 identity records REFERENCED in place (extents into the shards), split per
//!   rank on the head / KV-head axis (column-parallel projections, the per-head scales, the k / v
//!   convs), on K for `o_proj` (row-parallel: one all-reduce a visit), on vocab rows for `lm_head`;
//!   everything else replicated under `shared/`.

use super::config::{ExpertQuant, Inkling2Config};

/// Expert parallel width.
pub const EP: u32 = 2;
/// Tensor parallel width.
pub const TP: u32 = 2;

/// How the expert records are built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertForm {
    /// Every layer requantized to MXFP4 (one record format, the shipped grouped W4A8 bodies).
    Mxfp4,
    /// NVFP4 layers kept native (FN144 v2 record); MXFP8 layers requantized to MXFP4.
    Nvfp4Native,
    /// INK6: every layer native, no rounding -- NVFP4 layers in FN144's v2 record (FN148's form (b)
    /// reads it), MXFP8 layers in the INK6 record ([`mxfp8_layout`]: e4m3 `[N, K]` row-major +
    /// e8m0 `[K/32, N]`; `ink_mx8_w8a8_*` reads it).
    Native,
}

impl ExpertForm {
    /// The source kind of a layer with `q` under this form.
    #[must_use]
    pub fn kind(self, q: ExpertQuant) -> &'static str {
        match (self, q) {
            (Self::Mxfp4, ExpertQuant::Nvfp4G16) => "nvfp4_requant_mxfp4",
            (Self::Native, ExpertQuant::Mxfp8G32) => "mxfp8_native",
            (_, ExpertQuant::Mxfp8G32) => "mxfp8_requant_mxfp4",
            (Self::Nvfp4Native | Self::Native, ExpertQuant::Nvfp4G16) => "nvfp4_native",
        }
    }
}

/// How a dense tensor lands on the ranks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    /// One copy under `shared/`.
    Replicated,
    /// Rank `r` takes the `r`-th half of dim `dim`.
    Half { dim: u32 },
}

/// One dense record.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseRecord {
    /// The checkpoint tensor (full name).
    pub tensor: String,
    pub dtype: &'static str,
    pub shape: Vec<u64>,
    pub split: Split,
}

impl DenseRecord {
    /// Bytes one rank holds of it (a replicated record is billed to every rank: it lives in each
    /// card's VRAM).
    #[must_use]
    pub fn rank_bytes(&self) -> u64 {
        let es = if self.dtype == "F32" { 4 } else { 2 };
        let n: u64 = self.shape.iter().product::<u64>() * es;
        match self.split {
            Split::Replicated => n,
            Split::Half { .. } => n / 2,
        }
    }
}

/// The expert record geometry (bytes; offsets are the builder's, from
/// `qwen4_exp::pack_plan::ExpertRecordLayout`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertGeometry {
    pub hidden: u64,
    pub inter: u64,
    pub mxfp4_stride: u64,
    pub nvfp4_stride: u64,
    /// INK6's native MXFP8 record ([`mxfp8_layout`]).
    pub mxfp8_stride: u64,
}

/// INK6: the native MXFP8 e8m0-g32 expert record (no rounding; the checkpoint's bytes): gate_up
/// `w` `[2I, H]` e4m3 row-major (gate rows then up rows), `ws` e8m0 `[H/32, 2I]` (transposed, one
/// row a 32-k group); down `w` `[H, I]`, `ws` `[I/32, H]`; the `*_wref` slots are empty (no global).
/// Sub-records 256-aligned, the stride rounded to 4 KiB (the xtier law). Format `MXFP8_E8M0G32`.
#[must_use]
pub fn mxfp8_layout(hidden: u64, inter: u64) -> crate::qwen4_exp::pack_plan::ExpertRecordLayout {
    use crate::qwen4_exp::pack_plan::{RECORD_ALIGN, SUB_ALIGN};
    let gu_n = 2 * inter;
    let sizes = [
        gu_n * hidden,
        (hidden / 32) * gu_n,
        0,
        hidden * inter,
        (inter / 32) * hidden,
        0,
    ];
    let mut off = 0u64;
    let mut at = [(0u64, 0u64); 6];
    for (i, s) in sizes.iter().enumerate() {
        at[i] = (off, *s);
        off = (off + s).div_ceil(SUB_ALIGN) * SUB_ALIGN;
    }
    let end = at[4].0 + at[4].1;
    crate::qwen4_exp::pack_plan::ExpertRecordLayout {
        gate_up_w: at[0],
        gate_up_ws: at[1],
        gate_up_wref: (at[2].0, 0),
        down_w: at[3],
        down_ws: at[4],
        down_wref: (end, 0),
        payload: end,
        stride: end.div_ceil(RECORD_ALIGN) * RECORD_ALIGN,
        format: "MXFP8_E8M0G32",
    }
}

/// The plan.
#[derive(Clone, Debug, PartialEq)]
pub struct PackPlan {
    pub form: ExpertForm,
    pub experts_per_rank: u32,
    pub layers: u32,
    pub kinds: Vec<&'static str>,
    pub geometry: ExpertGeometry,
    pub dense: Vec<DenseRecord>,
}

const ATTN_HEAD_ROWS: [&str; 5] = [
    "q_proj.weight",
    "gate_proj.weight",
    "r_proj.weight",
    "sink_proj.weight",
    "q_scale",
];

impl PackPlan {
    /// The plan of `cfg` under `form`.
    #[must_use]
    pub fn new(cfg: &Inkling2Config, form: ExpertForm) -> Self {
        let h = u64::from(cfg.hidden_size);
        let i = u64::from(cfg.intermediate_size);
        let mx = crate::qwen4_exp::pack_plan::ExpertRecordLayout::new(h, i);
        let nv = crate::qwen4_exp::pack_plan::ExpertRecordLayout::nvfp4(h, i);
        let geometry = ExpertGeometry {
            hidden: h,
            inter: i,
            mxfp4_stride: mx.stride,
            nvfp4_stride: nv.stride,
            mxfp8_stride: mxfp8_layout(h, i).stride,
        };
        let kinds = cfg.expert_quant.iter().map(|&q| form.kind(q)).collect();
        let mut dense = Vec::new();
        let p = "language_model.model";
        let hd = u64::from(cfg.num_attention_heads * cfg.head_dim);
        let kvd = u64::from(cfg.num_key_value_heads * cfg.head_dim);
        let nh = u64::from(cfg.num_attention_heads);
        let mut push = |t: String, dtype: &'static str, shape: Vec<u64>, split: Split| {
            dense.push(DenseRecord {
                tensor: t,
                dtype,
                shape,
                split,
            });
        };
        push(
            format!("{p}.embed_tokens.weight"),
            "BF16",
            vec![u64::from(cfg.vocab_size), h],
            Split::Replicated,
        );
        push(
            format!("{p}.embed_norm.weight"),
            "BF16",
            vec![h],
            Split::Replicated,
        );
        push(
            format!("{p}.norm.weight"),
            "BF16",
            vec![h],
            Split::Replicated,
        );
        push(
            format!("{p}.final_input.query"),
            "F32",
            vec![h],
            Split::Replicated,
        );
        let core = u64::from(cfg.loop_cfg.core_end - cfg.loop_cfg.core_start);
        push(
            format!("{p}.message_input.query"),
            "F32",
            vec![core, 4, h],
            Split::Replicated,
        );
        push(
            "language_model.lm_head.weight".into(),
            "BF16",
            vec![u64::from(cfg.vocab_size), h],
            Split::Half { dim: 0 },
        );
        for l in 0..cfg.num_hidden_layers {
            let lp = format!("{p}.layers.{l}");
            let local = cfg.is_local(l);
            let a = format!("{lp}.self_attn");
            for name in ATTN_HEAD_ROWS {
                let shape = match name {
                    "q_proj.weight" | "gate_proj.weight" => vec![hd, h],
                    "r_proj.weight" => vec![nh * u64::from(cfg.d_rel), h],
                    "sink_proj.weight" => vec![nh, h],
                    _ => vec![nh, u64::from(cfg.head_dim)],
                };
                push(format!("{a}.{name}"), "BF16", shape, Split::Half { dim: 0 });
            }
            push(
                format!("{a}.rel_scale"),
                "BF16",
                vec![nh, u64::from(cfg.d_rel)],
                Split::Half { dim: 0 },
            );
            push(
                format!("{a}.k_proj.weight"),
                "BF16",
                vec![kvd, h],
                Split::Half { dim: 0 },
            );
            push(
                format!("{a}.v_proj.weight"),
                "BF16",
                vec![kvd, h],
                Split::Half { dim: 0 },
            );
            push(
                format!("{a}.o_proj.weight"),
                "BF16",
                vec![h, hd],
                Split::Half { dim: 1 },
            );
            push(
                format!("{a}.rel_proj"),
                "BF16",
                vec![u64::from(cfg.d_rel), u64::from(cfg.rel_extent)],
                Split::Replicated,
            );
            let k = u64::from(cfg.sconv_kernel_size);
            for c in ["k_sconv", "v_sconv"] {
                if local {
                    push(
                        format!("{a}.{c}.conv.weight"),
                        "BF16",
                        vec![kvd, k, 1],
                        Split::Half { dim: 0 },
                    );
                } else {
                    push(
                        format!("{a}.{c}.static_weight"),
                        "BF16",
                        vec![k, kvd],
                        Split::Half { dim: 1 },
                    );
                    // rows = kv head x lag: the head-major half.
                    push(
                        format!("{a}.{c}.weight_proj.weight"),
                        "BF16",
                        vec![u64::from(cfg.num_key_value_heads) * k, h],
                        Split::Half { dim: 0 },
                    );
                }
            }
            for c in ["attn_sconv", "mlp_sconv"] {
                push(
                    format!("{lp}.{c}.conv.weight"),
                    "BF16",
                    vec![h, k, 1],
                    Split::Replicated,
                );
            }
            if local {
                push(
                    format!("{lp}.input_layernorm.weight"),
                    "BF16",
                    vec![h],
                    Split::Replicated,
                );
            } else {
                for n in ["q", "k", "v"] {
                    push(
                        format!("{lp}.{n}_input_layernorm.weight"),
                        "BF16",
                        vec![h],
                        Split::Replicated,
                    );
                }
            }
            push(
                format!("{lp}.post_attention_layernorm.weight"),
                "BF16",
                vec![h],
                Split::Replicated,
            );
            if l > 0 {
                let nq = if local { vec![h] } else { vec![3, h] };
                push(
                    format!("{lp}.attention_input.query"),
                    "F32",
                    nq,
                    Split::Replicated,
                );
            }
            push(
                format!("{lp}.mlp_input.query"),
                "F32",
                vec![h],
                Split::Replicated,
            );
            push(
                format!("{lp}.mlp.gate_weight"),
                "BF16",
                vec![u64::from(cfg.n_routed_experts), h],
                Split::Replicated,
            );
            push(
                format!("{lp}.mlp.e_score_correction_bias"),
                "F32",
                vec![u64::from(cfg.n_routed_experts)],
                Split::Replicated,
            );
            if (cfg.loop_cfg.core_start..cfg.loop_cfg.core_end).contains(&l) {
                let r = u64::from(cfg.loop_cfg.gate_rank);
                for g in ["q", "k", "v", "mlp"] {
                    push(
                        format!("{lp}.{g}_message_gate.down_proj.weight"),
                        "BF16",
                        vec![r, 2 * h],
                        Split::Replicated,
                    );
                    push(
                        format!("{lp}.{g}_message_gate.forget_gate_proj.weight"),
                        "BF16",
                        vec![h, r],
                        Split::Replicated,
                    );
                    push(
                        format!("{lp}.{g}_message_gate.message_gate_proj.weight"),
                        "BF16",
                        vec![h, r],
                        Split::Replicated,
                    );
                }
            }
        }
        Self {
            form,
            experts_per_rank: cfg.n_routed_experts / EP,
            layers: cfg.num_hidden_layers,
            kinds,
            geometry,
            dense,
        }
    }

    /// The record stride of a layer's experts.
    #[must_use]
    pub fn stride(&self, layer: u32) -> u64 {
        match self.kinds[layer as usize] {
            "nvfp4_native" => self.geometry.nvfp4_stride,
            "mxfp8_native" => self.geometry.mxfp8_stride,
            _ => self.geometry.mxfp4_stride,
        }
    }

    /// Per-rank bytes: (experts, dense split, dense replicated).
    #[must_use]
    pub fn rank_bytes(&self) -> (u64, u64, u64) {
        let ex: u64 = (0..self.layers)
            .map(|l| self.stride(l) * u64::from(self.experts_per_rank))
            .sum();
        let mut split = 0;
        let mut rep = 0;
        for d in &self.dense {
            match d.split {
                Split::Replicated => rep += d.rank_bytes(),
                Split::Half { .. } => split += d.rank_bytes(),
            }
        }
        (ex, split, rep)
    }

    /// The builder's input file byte for byte (`saber-inkling2-packplan`'s stdout): the plan JSON
    /// with the checkpoint path and the visit list, pretty-printed, one trailing newline. Its
    /// sha256 is the pack manifest's `plan_sha256` (the INK2 dry run re-derives it).
    #[must_use]
    pub fn plan_text(&self, cfg: &Inkling2Config, checkpoint: &str) -> String {
        let mut v = self.to_json();
        v["checkpoint"] = checkpoint.into();
        let visits: Vec<_> = cfg
            .visits()
            .iter()
            .map(|x| {
                serde_json::json!({"visit": x.index, "layer": x.layer, "pass": x.pass, "local": x.local,
                                    "block_boundary": x.block_boundary, "messages": x.messages,
                                    "resets_to_saved": x.resets_to_saved})
            })
            .collect();
        v["visits"] = visits.into();
        format!("{}\n", serde_json::to_string_pretty(&v).expect("json"))
    }

    /// The plan as JSON (the builder's input).
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        let dense: Vec<_> = self
            .dense
            .iter()
            .map(|d| {
                let (placement, dim) = match d.split {
                    Split::Replicated => ("shared", serde_json::Value::Null),
                    Split::Half { dim } => ("rank", serde_json::Value::from(dim)),
                };
                serde_json::json!({"tensor": d.tensor, "dtype": d.dtype, "shape": d.shape,
                                   "placement": placement, "split_dim": dim})
            })
            .collect();
        let (ex, split, rep) = self.rank_bytes();
        let mut v = serde_json::json!({
            "model": "inkling2_loop",
            "tp": TP, "ep": EP,
            "form": format!("{:?}", self.form),
            "experts_per_rank": self.experts_per_rank,
            "layers": self.layers,
            "expert_kinds": self.kinds,
            "geometry": {"hidden": self.geometry.hidden, "inter": self.geometry.inter,
                         "mxfp4_stride": self.geometry.mxfp4_stride, "nvfp4_stride": self.geometry.nvfp4_stride},
            "records": {"mxfp4": record_json(&crate::qwen4_exp::pack_plan::ExpertRecordLayout::new(self.geometry.hidden, self.geometry.inter)),
                        "nvfp4": record_json(&crate::qwen4_exp::pack_plan::ExpertRecordLayout::nvfp4(self.geometry.hidden, self.geometry.inter))},
            "dense": dense,
            "rank_bytes": {"experts": ex, "dense_split": split, "dense_replicated": rep},
        });
        // INK6: the native MXFP8 record only in a Native plan (the older plans' text, and so the
        // built packs' plan_sha256, stay byte for byte).
        if self.form == ExpertForm::Native {
            v["geometry"]["mxfp8_stride"] = self.geometry.mxfp8_stride.into();
            v["records"]["mxfp8"] =
                record_json(&mxfp8_layout(self.geometry.hidden, self.geometry.inter));
        }
        v
    }
}

fn record_json(r: &crate::qwen4_exp::pack_plan::ExpertRecordLayout) -> serde_json::Value {
    serde_json::json!({
        "format": r.format, "payload": r.payload, "stride": r.stride,
        "gate_up_w": [r.gate_up_w.0, r.gate_up_w.1], "gate_up_ws": [r.gate_up_ws.0, r.gate_up_ws.1],
        "gate_up_wref": [r.gate_up_wref.0, r.gate_up_wref.1], "down_w": [r.down_w.0, r.down_w.1],
        "down_ws": [r.down_ws.0, r.down_ws.1], "down_wref": [r.down_wref.0, r.down_wref.1],
    })
}
