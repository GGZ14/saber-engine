//! `config.json` of an `inkling2_loop` checkpoint as typed data (INK1, `docs/INKLING2.md`).
//!
//! Every key is taken, acknowledged, or refused by its dotted path (the `qwen4_exp` reader's law,
//! [`crate::qwen4_exp::fields`]). Values the forward cannot run are refused by name: a loop whose
//! bounds are not attention-residual block boundaries, a variable loop, a global layer that is not
//! a block's last layer, an expert quantization other than NVFP4 g16 / MXFP8 g32, a drafter tap
//! past the visit count.

use serde_json::Value;

use crate::qwen4_exp::fields::Fields;
use crate::ModelError;

fn err(msg: String) -> ModelError {
    ModelError::Config(msg)
}

/// How one layer's routed experts are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertQuant {
    /// e2m1 codes (8 a u32, low nibble first) + E4M3 scales per 16 + an f32 global per expert per
    /// projection, MULTIPLIED (`weight_scale_2`).
    Nvfp4G16,
    /// e4m3 bytes (4 a u32) + e8m0 scales per 32; no global.
    Mxfp8G32,
}

impl ExpertQuant {
    /// Bytes of one projection `[n, k]` (codes + scales + global).
    #[must_use]
    pub fn projection_bytes(self, n: u64, k: u64) -> u64 {
        match self {
            Self::Nvfp4G16 => n * k / 2 + n * k / 16 + 4,
            Self::Mxfp8G32 => n * k + n * k / 32,
        }
    }
}

/// The middle loop (`text_config.loop_config`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopConfig {
    pub core_start: u32,
    pub core_end: u32,
    pub num_passes: u32,
    pub gate_rank: u32,
}

/// The drafter's config (`mtp_config`; stage 6).
#[derive(Debug, Clone, PartialEq)]
pub struct DrafterConfig {
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub intermediate_size: u32,
    pub rope_theta: f64,
    pub local_layer_ids: Vec<u32>,
    pub sliding_window_size: u32,
    pub block_size: u32,
    pub markov_rank: u32,
    /// VISIT indices of the target's schedule (the last = the final normalized hidden).
    pub target_layer_ids: Vec<u32>,
}

/// The vision tower's config (`vision_config`).
#[derive(Debug, Clone, PartialEq)]
pub struct VisionConfig {
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub mlp_ratio: f64,
    pub native_patch_size: u32,
    pub temporal_patch_size: u32,
    pub adapter_merge_size: u32,
    pub projection_hidden_dim: u32,
    pub posemb_grid_size: u32,
    pub layer_norm_eps: f64,
}

/// The whole model's facts.
#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2Config {
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub vocab_size: u32,
    pub unpadded_vocab_size: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub d_rel: u32,
    pub rel_extent: u32,
    pub sliding_window_size: u32,
    pub local_layer_ids: Vec<u32>,
    pub sconv_kernel_size: u32,
    pub attn_res_block: u32,
    pub n_routed_experts: u32,
    pub num_experts_per_tok: u32,
    pub intermediate_size: u32,
    pub route_scale: f64,
    pub mlp_gate_softcap: f64,
    pub mlp_up_softcap: f64,
    pub final_logit_softcap: f64,
    pub qk_l2_clip: f64,
    pub rel_l2_clip: f64,
    pub log_scaling_n_floor: f64,
    pub log_scaling_alpha: f64,
    pub rms_norm_eps: f64,
    pub max_context: u64,
    pub eos_token_id: u32,
    pub loop_cfg: LoopConfig,
    /// One entry per physical layer.
    pub expert_quant: Vec<ExpertQuant>,
    pub drafter: DrafterConfig,
    pub vision: VisionConfig,
}

fn parse_quant(path: &str, v: &Value, layers: u32) -> Result<Vec<ExpertQuant>, ModelError> {
    let mut q = Fields::new(path, v)?;
    // The default entry: NVFP4 g16 (what the layers 0..19 entries repeat).
    let (gs, bits, mode) = (q.u32("group_size")?, q.u32("bits")?, q.str("mode")?);
    let default = quant_of(path, gs, bits, mode)?;
    let mut out = vec![None; layers as usize];
    for l in 0..layers {
        let mut seen = None;
        for p in ["gate_proj", "up_proj", "down_proj"] {
            let key = format!("language_model.model.layers.{l}.mlp.switch_mlp.{p}");
            let kind = match q.optional(&key) {
                Some(e) => {
                    let sub = format!("{path}.{key}");
                    let mut f = Fields::new(&sub, e)?;
                    let k = quant_of(&sub, f.u32("group_size")?, f.u32("bits")?, f.str("mode")?)?;
                    f.finish()?;
                    k
                }
                None => default,
            };
            if seen.is_some_and(|s| s != kind) {
                return Err(err(format!(
                    "{path}: layer {l}'s three projections differ in mode"
                )));
            }
            seen = Some(kind);
        }
        out[l as usize] = seen;
    }
    q.finish()?;
    Ok(out.into_iter().map(|k| k.expect("set above")).collect())
}

fn quant_of(path: &str, gs: u32, bits: u32, mode: &str) -> Result<ExpertQuant, ModelError> {
    match (mode, bits, gs) {
        ("nvfp4", 4, 16) => Ok(ExpertQuant::Nvfp4G16),
        ("mxfp8", 8, 32) => Ok(ExpertQuant::Mxfp8G32),
        _ => Err(err(format!(
            "{path}: expert quant {mode} bits {bits} group {gs} is not NVFP4 g16 or MXFP8 g32"
        ))),
    }
}

impl Inkling2Config {
    /// Parse `config.json`.
    ///
    /// # Errors
    /// [`ModelError::Config`] naming the offending field, or a JSON parse error.
    pub fn from_config_json(json: &str) -> Result<Self, ModelError> {
        let root: Value = serde_json::from_str(json)?;
        let mut top = Fields::new("", &root)?;
        let arch = top.array("architectures")?;
        if arch.len() != 1 || arch[0].as_str() != Some("Inkling2LoopForConditionalGeneration") {
            return Err(err(format!(
                "architectures: expected [\"Inkling2LoopForConditionalGeneration\"], got {}",
                Value::Array(arch.clone())
            )));
        }
        top.expect("model_type", &Value::from("inkling2_loop"))?;
        top.ack("do_sample");
        top.ack("temperature");
        top.ack("generation_config");
        let eos_token_id = top.u32("eos_token_id")?;

        let tv = top.value("text_config")?;
        let mut t = Fields::new("text_config", tv)?;
        t.expect("model_type", &Value::from("inkling2_loop_text"))?;
        t.expect("torch_dtype", &Value::from("bfloat16"))?;
        let max_context = t.u64("model_max_length")?;
        let hidden_size = t.u32("hidden_size")?;
        let num_hidden_layers = t.u32("num_hidden_layers")?;
        let vocab_size = t.u32("vocab_size")?;
        let unpadded_vocab_size = t.u32("unpadded_vocab_size")?;
        let num_attention_heads = t.u32("num_attention_heads")?;
        let num_key_value_heads = t.u32("num_key_value_heads")?;
        let head_dim = t.u32("head_dim")?;
        // The sliding layers' attention shape: Saber's attention body is one shape for both kinds.
        for (k, want) in [
            ("swa_head_dim", head_dim),
            ("swa_num_attention_heads", num_attention_heads),
            ("swa_num_key_value_heads", num_key_value_heads),
        ] {
            if t.u32(k)? != want {
                return Err(err(format!(
                    "text_config.{k}: a sliding shape unlike the global one is not supported"
                )));
            }
        }
        let d_rel = t.u32("d_rel")?;
        let rel_extent = t.u32("rel_extent")?;
        t.expect("q_bias", &Value::from(false))?;
        t.expect("o_bias", &Value::from(false))?;
        let log_scaling_n_floor = t.f64("log_scaling_n_floor")?;
        let log_scaling_alpha = t.f64("log_scaling_alpha")?;
        let rms_norm_eps = t.f64("rms_norm_eps")?;
        t.expect("use_value_norm_no_gamma", &Value::from(true))?;
        let qk_l2_clip = t.f64("qk_l2_clip_threshold")?;
        let rel_l2_clip = t.f64("rel_l2_clip_threshold")?;
        let local_layer_ids = t.u32_array("local_layer_ids")?;
        let sconv_kernel_size = t.u32("sconv_kernel_size")?;
        let final_logit_softcap = t.f64("final_logit_softcapping")?;
        let sliding_window_size = t.u32("sliding_window_size")?;
        let attn_res_block = t.u32("n_layers_per_attn_res_block")?;
        t.expect("attn_res_normalize_values", &Value::from(true))?;
        let n_routed_experts = t.u32("n_routed_experts")?;
        let num_experts_per_tok = t.u32("num_experts_per_tok")?;
        t.expect("n_shared_experts", &Value::from(0))?;
        let intermediate_size = t.u32("intermediate_size")?;
        let mlp_gate_softcap = t.f64("mlp_gate_softcap")?;
        let mlp_up_softcap = t.f64("mlp_up_softcap")?;
        let route_scale = t.f64("route_scale")?;
        t.expect("use_gate_bias", &Value::from(true))?;
        let lv = t.value("loop_config")?;
        let mut lf = Fields::new("text_config.loop_config", lv)?;
        let loop_cfg = LoopConfig {
            core_start: lf.u32("core_start")?,
            core_end: lf.u32("core_end")?,
            num_passes: lf.u32("num_passes")?,
            gate_rank: lf.u32("gate_rank")?,
        };
        lf.expect("is_varloop", &Value::from(false))?;
        lf.finish()?;
        t.finish()?;

        let qv = top.value("quantization")?;
        let expert_quant = parse_quant("quantization", qv, num_hidden_layers)?;
        if top.value("quantization_config")? != qv {
            return Err(err("quantization_config: differs from quantization".into()));
        }

        let mv = top.value("mtp_config")?;
        let mut m = Fields::new("mtp_config", mv)?;
        let drafter = DrafterConfig {
            hidden_size: m.u32("hidden_size")?,
            num_hidden_layers: m.u32("num_hidden_layers")?,
            num_attention_heads: m.u32("num_attention_heads")?,
            num_key_value_heads: m.u32("num_key_value_heads")?,
            head_dim: m.u32("head_dim")?,
            intermediate_size: m.u32("intermediate_size")?,
            rope_theta: m.f64("rope_theta")?,
            local_layer_ids: m.u32_array("local_layer_ids")?,
            sliding_window_size: m.u32("sliding_window_size")?,
            block_size: m.u32("block_size")?,
            markov_rank: m.u32("markov_rank")?,
            target_layer_ids: m.u32_array("target_layer_ids")?,
        };
        m.ack("max_position_embeddings");
        m.expect("torch_dtype", &Value::from("bfloat16"))?;
        m.expect("q_bias", &Value::from(false))?;
        m.expect("o_bias", &Value::from(false))?;
        m.ack("rms_norm_eps");
        m.expect("markov_head_type", &Value::from("rnn_hidden"))?;
        m.finish()?;

        let vv = top.value("vision_config")?;
        let mut v = Fields::new("vision_config", vv)?;
        v.expect("model_type", &Value::from("inkling2_vision"))?;
        if v.u32("text_hidden_size")? != hidden_size {
            return Err(err(
                "vision_config.text_hidden_size: differs from the text hidden size".into(),
            ));
        }
        v.expect("projector_bias", &Value::from(false))?;
        let vision = VisionConfig {
            hidden_size: v.u32("hidden_size")?,
            num_hidden_layers: v.u32("num_hidden_layers")?,
            num_attention_heads: v.u32("num_attention_heads")?,
            mlp_ratio: v.f64("mlp_ratio")?,
            native_patch_size: v.u32("native_patch_size")?,
            temporal_patch_size: v.u32("temporal_patch_size")?,
            adapter_merge_size: v.u32("adapter_merge_size")?,
            projection_hidden_dim: v.u32("projection_hidden_dim")?,
            posemb_grid_size: v.u32("posemb_grid_size")?,
            layer_norm_eps: v.f64("layer_norm_eps")?,
        };
        v.finish()?;
        top.finish()?;

        let cfg = Self {
            hidden_size,
            num_hidden_layers,
            vocab_size,
            unpadded_vocab_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            d_rel,
            rel_extent,
            sliding_window_size,
            local_layer_ids,
            sconv_kernel_size,
            attn_res_block,
            n_routed_experts,
            num_experts_per_tok,
            intermediate_size,
            route_scale,
            mlp_gate_softcap,
            mlp_up_softcap,
            final_logit_softcap,
            qk_l2_clip,
            rel_l2_clip,
            log_scaling_n_floor,
            log_scaling_alpha,
            rms_norm_eps,
            max_context,
            eos_token_id,
            loop_cfg,
            expert_quant,
            drafter,
            vision,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<(), ModelError> {
        let l = &self.loop_cfg;
        let b = self.attn_res_block;
        if !(0 < l.core_start && l.core_start < l.core_end && l.core_end <= self.num_hidden_layers)
            || l.core_start % b != 0
            || l.core_end % b != 0
            || l.num_passes < 1
            || l.gate_rank < 1
        {
            return Err(err(format!(
                "text_config.loop_config: {l:?} is not a block-aligned loop inside {} layers",
                self.num_hidden_layers
            )));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(err("text_config: heads not a multiple of KV heads".into()));
        }
        if self
            .local_layer_ids
            .iter()
            .any(|&i| i >= self.num_hidden_layers)
        {
            return Err(err(
                "text_config.local_layer_ids: a layer out of range".into()
            ));
        }
        if self.rel_extent != self.sliding_window_size {
            return Err(err(
                "text_config: rel_extent != sliding_window_size (the bias table is one width)"
                    .into(),
            ));
        }
        let visits = self.schedule().len() as u32;
        let taps = &self.drafter.target_layer_ids;
        if taps.len() < 2
            || taps.windows(2).any(|w| w[0] >= w[1])
            || *taps.last().unwrap_or(&0) != visits
        {
            return Err(err(format!("mtp_config.target_layer_ids: {taps:?} must be strictly increasing and end at the visit count {visits}")));
        }
        Ok(())
    }

    /// Whether physical layer `l` is a sliding-window (local) layer.
    #[must_use]
    pub fn is_local(&self, l: u32) -> bool {
        self.local_layer_ids.contains(&l)
    }

    /// The visit list: visit `v` runs physical layer `schedule()[v]`.
    #[must_use]
    pub fn schedule(&self) -> Vec<u32> {
        let l = &self.loop_cfg;
        let mut s: Vec<u32> = (0..l.core_start).collect();
        for _ in 0..l.num_passes {
            s.extend(l.core_start..l.core_end);
        }
        s.extend(l.core_end..self.num_hidden_layers);
        s
    }

    /// Facts of every visit, in order.
    #[must_use]
    pub fn visits(&self) -> Vec<Visit> {
        let l = &self.loop_cfg;
        let per_pass = l.core_end - l.core_start;
        self.schedule()
            .into_iter()
            .enumerate()
            .map(|(v, layer)| {
                let v = v as u32;
                let pass = if v < l.core_start {
                    0
                } else if v < l.core_start + l.num_passes * per_pass {
                    (v - l.core_start) / per_pass
                } else {
                    l.num_passes - 1
                };
                let in_core = (l.core_start..l.core_end).contains(&layer);
                Visit {
                    index: v,
                    layer,
                    pass,
                    local: self.is_local(layer),
                    block_boundary: layer % self.attn_res_block == 0,
                    messages: in_core && pass > 0,
                    resets_to_saved: in_core && layer == l.core_start && pass > 0,
                }
            })
            .collect()
    }

    /// Per-sequence state bytes at `ctx` tokens (bf16 KV, f32 conv states): (global KV, local KV
    /// rings, conv states). The local ring is `min(ctx, window)` rows.
    #[must_use]
    pub fn state_bytes(&self, ctx: u64) -> (u64, u64, u64) {
        let kv_row = 2 * u64::from(self.num_key_value_heads) * u64::from(self.head_dim) * 2;
        let k1 = u64::from(self.sconv_kernel_size - 1);
        let kvc = u64::from(self.num_key_value_heads * self.head_dim);
        let conv = k1 * (2 * kvc + 2 * u64::from(self.hidden_size)) * 4;
        let mut g = 0;
        let mut loc = 0;
        let mut c = 0;
        for v in self.visits() {
            c += conv;
            if v.local {
                loc += kv_row * ctx.min(u64::from(self.sliding_window_size));
            } else {
                g += kv_row * ctx;
            }
        }
        (g, loc, c)
    }
}

/// One layer visit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Visit {
    pub index: u32,
    pub layer: u32,
    /// The loop pass (0-based; 0 for the prologue, `num_passes - 1` for the epilogue).
    pub pass: u32,
    pub local: bool,
    /// `layer % block == 0`: the visit appends its input to the block list and its residual
    /// starts from the attention output alone.
    pub block_boundary: bool,
    /// The visit reads the previous pass's messages (core layers, passes 1..).
    pub messages: bool,
    /// The visit first resets the partial and the block list to the saved pre-loop pair.
    pub resets_to_saved: bool,
}
