//! The memory-budget estimator: what a config leaves for the KV cache, in bytes per
//! card and in tokens, and how many sequences fit at the chosen context.
//!
//! This is an ESTIMATE, labelled so everywhere it is shown: the engine's own ledger
//! (`engledger` / `fnbudget` boot lines) is the truth and refuses a boot that does not fit. The
//! inputs are read from the pack's `manifest.json` (per-rank weight bytes; Flash-Next's ledger
//! classes and expert record stride) and the model's `config.json` (attention layers, KV heads,
//! head dim, the GDN state shape) where readable; what is not readable is named in `notes`.
//!
//! Two laws, one per serving path:
//! - **27B** (`saber serve --engine real`): the device budget per rank (the engine's ledger law:
//!   `lean` replaces `--engine-budget-mib` / `--engine-kv-headroom-mib` with its measured pair,
//!   [`LEAN_BUDGET_MIB`] / [`LEAN_HEADROOM_MIB`]; capped by the card's VRAM) less the target and
//!   drafter weights, the GDN state slots (`--engine-max-seqs` of them), a fixed arena allowance
//!   and the headroom is the KV pool; its tokens / the context = the sequences that fit at full
//!   context. At **TP=1** the one card is the only rank: it holds every weight (the `-tp1`
//!   packs' single `rank0.pack`), every KV head and the whole GDN state of each sequence, so the
//!   same law runs with the unsplit shapes and the pool is far smaller than TP=2's.
//! - **Flash-Next** (`saber engine --fn-serve`): the KV lives in pinned host RAM behind a VRAM
//!   hot pool (the tiered KV); the admission slab (`--fn-kv-admit-slab-tokens`, else one
//!   max-context request + one 10,240-token request) is the token capacity, the engine holds 8
//!   sequence slots, and the VRAM left after the dense weights and the hot pool goes to expert
//!   slots (shown, since it is what a bigger context costs).
//!
//! [`estimate`] is pure; [`read_pack`] / [`read_model`] do the file reads.

use std::path::{Path, PathBuf};

use serde::Serialize;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const MIB: u64 = 1024 * 1024;

/// What the estimator knows about a model (from its `config.json`).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ModelInfo {
    /// The file it was read from.
    pub source: String,
    pub layers: u32,
    /// Layers that keep a KV cache (`full_attention`).
    pub attn_layers: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// GDN (`linear_attention`) layers and their recurrent state shape.
    pub linear_layers: u32,
    pub linear_v_heads: u32,
    pub linear_k_dim: u32,
    pub linear_v_dim: u32,
    pub max_position: Option<u64>,
}

impl ModelInfo {
    /// Parse a Hugging Face `config.json` (its `text_config` when present).
    pub fn from_config_json(text: &str, source: &str) -> anyhow::Result<ModelInfo> {
        let v: serde_json::Value = serde_json::from_str(text)?;
        let c = v.get("text_config").unwrap_or(&v);
        let u = |k: &str| c.get(k).and_then(serde_json::Value::as_u64);
        let layers =
            u("num_hidden_layers").ok_or_else(|| anyhow::anyhow!("no num_hidden_layers"))? as u32;
        let types: Vec<&str> = c
            .get("layer_types")
            .and_then(|t| t.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
            .unwrap_or_default();
        let (attn, linear) = if types.is_empty() {
            match u("full_attention_interval") {
                Some(i) if i > 0 => {
                    let a = layers / i as u32;
                    (a, layers - a)
                }
                _ => (layers, 0),
            }
        } else {
            let a = types.iter().filter(|t| **t == "full_attention").count() as u32;
            let l = types.iter().filter(|t| **t == "linear_attention").count() as u32;
            (a, l)
        };
        let heads = u("num_attention_heads").unwrap_or(0);
        let head_dim = u("head_dim").or_else(|| {
            let h = u("hidden_size")?;
            (heads > 0).then(|| h / heads)
        });
        Ok(ModelInfo {
            source: source.to_string(),
            layers,
            attn_layers: attn,
            kv_heads: u("num_key_value_heads").unwrap_or(heads) as u32,
            head_dim: head_dim.unwrap_or(0) as u32,
            linear_layers: linear,
            linear_v_heads: u("linear_num_value_heads").unwrap_or(0) as u32,
            linear_k_dim: u("linear_key_head_dim").unwrap_or(0) as u32,
            linear_v_dim: u("linear_value_head_dim").unwrap_or(0) as u32,
            max_position: u("max_position_embeddings"),
        })
    }

    /// KV bytes a token on ONE rank (K and V, every attention layer, the rank's KV heads).
    #[must_use]
    pub fn kv_bytes_per_token(&self, tp: u32, dtype_bytes: u32) -> u64 {
        let heads = (self.kv_heads / tp.max(1)).max(1);
        u64::from(self.attn_layers)
            * u64::from(heads)
            * u64::from(self.head_dim)
            * 2
            * u64::from(dtype_bytes)
    }

    /// GDN recurrent state bytes a sequence on ONE rank.
    #[must_use]
    pub fn state_bytes_per_seq(&self, tp: u32, state_bytes: u32) -> u64 {
        let heads = (self.linear_v_heads / tp.max(1)).max(u32::from(self.linear_v_heads > 0));
        u64::from(self.linear_layers)
            * u64::from(heads)
            * u64::from(self.linear_k_dim)
            * u64::from(self.linear_v_dim)
            * u64::from(state_bytes)
    }
}

/// An installed pack, as far as the estimator and the model manager read it.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct PackInfo {
    pub name: String,
    pub path: String,
    /// Bytes of the pack's own files.
    pub size_bytes: u64,
    /// Checkpoint bytes the pack references in place (Flash-Next packs reference their
    /// checkpoint's shards instead of copying them).
    pub referenced_bytes: u64,
    /// `qwen4_exp` (Flash-Next) or the builder's model descriptor.
    pub model: Option<String>,
    /// `target` or `drafter`.
    pub role: &'static str,
    pub tp_size: Option<u32>,
    /// Device-resident weight bytes on rank 0 (27B: `rank0.pack`; Flash-Next: the ledger's
    /// dense classes, the embedding excluded: it lives in pinned RAM by default).
    pub rank_weight_bytes: Option<u64>,
    /// Flash-Next: the expert records' bytes a rank and one record's stride.
    pub expert_bytes: Option<u64>,
    pub expert_stride: Option<u64>,
    /// Flash-Next: expert records a rank owns (`expert_tiers[0].records`).
    pub expert_records: Option<u64>,
    /// The checkpoint dir the manifest names (Flash-Next), for its `config.json`.
    pub checkpoint_dir: Option<String>,
    pub has_vision: bool,
    /// The manifest did not parse (listed anyway).
    pub error: Option<String>,
    pub built_unix: Option<u64>,
}

fn dir_bytes(p: &Path, depth: u32) -> u64 {
    let Ok(rd) = std::fs::read_dir(p) else {
        return 0;
    };
    let mut n = 0;
    for e in rd.flatten() {
        let Ok(m) = e.metadata() else { continue };
        if m.is_dir() && depth < 6 {
            n += dir_bytes(&e.path(), depth + 1);
        } else if m.is_file() {
            n += m.len();
        }
    }
    n
}

/// Parse a pack's `manifest.json` text (pure; [`read_pack`] adds the file sizes).
#[must_use]
pub fn parse_manifest(name: &str, text: &str) -> PackInfo {
    let mut p = PackInfo {
        name: name.to_string(),
        role: if name.contains("drafter") {
            "drafter"
        } else {
            "target"
        },
        ..PackInfo::default()
    };
    let m: serde_json::Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(e) => {
            p.error = Some(format!("manifest.json: {e}"));
            return p;
        }
    };
    p.built_unix = m["built_unix"].as_u64();
    p.tp_size = m["tp_size"].as_u64().map(|v| v as u32);
    if let Some(b) = m["builder"].as_str() {
        if b.contains("drafter") {
            p.role = "drafter";
        }
    }
    if let Some(model) = m["model"].as_str() {
        p.model = Some(model.to_string());
    } else if let Some(model) = m["tier"]["descriptor"]["model"].as_str() {
        p.model = Some(model.to_string());
    } else if let Some(model) =
        m["tier"]["descriptor"]["derived_from"]["tier"]["descriptor"]["model"].as_str()
    {
        p.model = Some(model.to_string());
    }
    // 27B form: per-rank pack files
    if let Some(ranks) = m["ranks"].as_array() {
        p.rank_weight_bytes = ranks
            .iter()
            .find(|r| r["rank"].as_u64() == Some(0))
            .and_then(|r| r["file_bytes"].as_u64());
        if p.tp_size.is_none() {
            p.tp_size = Some(ranks.len() as u32);
        }
    }
    // Flash-Next form: the ledger + the expert tiers + the checkpoint it references
    if let Some(ledger) = m["ledger"].as_array() {
        let mut dense = 0u64;
        let mut experts = 0u64;
        for e in ledger {
            let place = e["placement"].as_str().unwrap_or("");
            let class = e["class"].as_str().unwrap_or("");
            let b = e["bytes"].as_u64().unwrap_or(0);
            if place != "rank0" && place != "shared" {
                continue;
            }
            match class {
                "experts" => experts += b,
                "embed" => {}
                _ => dense += b,
            }
        }
        p.rank_weight_bytes = Some(dense);
        p.expert_bytes = Some(experts);
        if let Some(l) = m["layout"]["tp"].as_u64() {
            p.tp_size = Some(l as u32);
        }
    }
    p.expert_stride = m["expert_tiers"][0]["record"]["stride"].as_u64();
    p.expert_records = m["expert_tiers"][0]["records"].as_u64();
    if let Some(dir) = m["checkpoint"]["dir"].as_str() {
        p.checkpoint_dir = Some(dir.to_string());
        p.referenced_bytes = m["checkpoint"]["shards"]
            .as_array()
            .map(|a| a.iter().filter_map(|s| s["size_bytes"].as_u64()).sum())
            .unwrap_or(0);
    }
    p
}

/// Read a pack directory (its manifest and its own files' sizes).
#[must_use]
pub fn read_pack(dir: &Path) -> PackInfo {
    let name = dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut p = match std::fs::read_to_string(dir.join("manifest.json")) {
        Ok(t) => parse_manifest(&name, &t),
        Err(e) => PackInfo {
            name: name.clone(),
            role: if name.contains("drafter") {
                "drafter"
            } else {
                "target"
            },
            error: Some(format!("manifest.json: {e}")),
            ..PackInfo::default()
        },
    };
    p.path = dir.display().to_string();
    p.size_bytes = dir_bytes(dir, 0);
    p.has_vision = dir.join("vision").is_dir();
    p
}

/// The model's `config.json`: the pack's own, else its checkpoint's, else the first of
/// `fallbacks` that reads.
#[must_use]
pub fn read_model(pack: Option<&PackInfo>, fallbacks: &[PathBuf]) -> Option<ModelInfo> {
    let mut cands: Vec<PathBuf> = Vec::new();
    if let Some(p) = pack {
        cands.push(Path::new(&p.path).join("config.json"));
        if let Some(c) = &p.checkpoint_dir {
            cands.push(Path::new(c).join("config.json"));
        }
    }
    cands.extend(fallbacks.iter().cloned());
    cands.iter().find_map(|c| {
        let t = std::fs::read_to_string(c).ok()?;
        ModelInfo::from_config_json(&t, &c.display().to_string()).ok()
    })
}

/// Which law applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Law {
    Dense27b,
    Flashnext,
    Mock,
}

/// Everything [`estimate`] reads (resolved from the config's effective flags + the files).
#[derive(Clone, Debug)]
pub struct Inputs {
    pub law: Law,
    pub tp: u32,
    /// The card's VRAM (from sysfs), when known.
    pub vram_per_card: Option<u64>,
    /// The engine's per-rank budget flag (MiB).
    pub budget_mib: Option<u64>,
    pub max_ctx: u64,
    /// 27B: `--engine-max-seqs`; Flash-Next: the engine's 8 slots.
    pub max_seqs: u64,
    pub kv_dtype_bytes: u32,
    pub state_dtype_bytes: u32,
    pub target: Option<PackInfo>,
    pub drafter: Option<PackInfo>,
    pub model: Option<ModelInfo>,
    /// 27B: `--engine-kv-headroom-mib`.
    pub kv_headroom_mib: u64,
    /// Flash-Next: the hot pool (GiB), the admission slab (tokens), host RAM (bytes).
    pub fn_kv_hot_gib: f64,
    pub fn_admit_slab_tokens: Option<u64>,
    pub fn_kv_tier_on: bool,
    pub host_ram: Option<u64>,
    /// Mock: `--kv-gib` / `--state-gib`.
    pub mock_kv_gib: u64,
}

/// The estimate the UI shows.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Estimate {
    /// Always true: the UI labels this an estimate.
    pub is_estimate: bool,
    pub law: Option<Law>,
    /// The tensor-parallel width estimated and the cards it uses (1 or 2).
    pub tp: u32,
    /// What the figures are, in words (shown with them): per card of two, or the one card.
    pub label: String,
    /// Per card (rank 0).
    pub vram_bytes: Option<u64>,
    pub budget_bytes: Option<u64>,
    pub weights_bytes: Option<u64>,
    pub drafter_bytes: Option<u64>,
    pub state_bytes: Option<u64>,
    pub reserved_bytes: Option<u64>,
    /// The KV pool a rank (27B: VRAM; Flash-Next: the VRAM hot pool, its RAM tier below).
    pub kv_bytes: Option<i64>,
    pub kv_ram_bytes: Option<u64>,
    pub kv_bytes_per_token: Option<u64>,
    pub kv_capacity_tokens: Option<u64>,
    pub max_ctx: u64,
    pub requested_seqs: u64,
    /// Sequences that fit at `max_ctx` (capped by the engine's slots).
    pub seqs_at_ctx: Option<u64>,
    /// Flash-Next: VRAM expert slots left a rank, and how many records a rank owns.
    pub expert_slots: Option<i64>,
    pub expert_records: Option<u64>,
    /// `false` when the estimate says the config does not fit.
    pub fits: bool,
    pub notes: Vec<String>,
    pub warnings: Vec<String>,
}

/// The engine's `lean` ledger law (`saber_cli::ledger_law`, the default): the
/// fence and the KV headroom it uses in place of the two flags. A `saber-cli` test keeps these
/// equal to the engine's.
pub const LEAN_BUDGET_MIB: u64 = 31_744;
pub const LEAN_HEADROOM_MIB: u64 = 848;

/// Fixed per-rank allowance for the arenas, graph pools and heads the estimate does not model.
pub const ARENA_ALLOWANCE_MIB: u64 = 2_048;
/// The Flash-Next engine's sequence slots.
pub const FN_SEQ_SLOTS: u64 = 8;
/// Flash-Next's default admission slab beyond one max-context request.
pub const FN_SLAB_EXTRA_TOKENS: u64 = 10_240;

fn gib(b: f64) -> String {
    format!("{:.2} GiB", b / GIB)
}

/// The estimate for `i` (pure).
#[must_use]
pub fn estimate(i: &Inputs) -> Estimate {
    let mut e = Estimate {
        is_estimate: true,
        law: Some(i.law),
        tp: i.tp.max(1),
        label: match (i.law, i.tp) {
            (Law::Mock, _) => "Estimate: the mock engine (no GPU memory is used)".into(),
            (Law::Dense27b, 1) => {
                "Estimate for TP=1: ONE card holds the whole model (all of its weights, the KV \
                 cache and every sequence's GDN state)"
                    .into()
            }
            (_, tp) => format!(
                "Estimate per card, TP={tp}: each of the {tp} cards holds 1/{tp} of the weights and \
                 of the KV cache"
            ),
        },
        max_ctx: i.max_ctx,
        requested_seqs: i.max_seqs,
        fits: true,
        vram_bytes: i.vram_per_card,
        ..Estimate::default()
    };
    if i.max_ctx == 0 {
        e.warnings.push("max context is 0".into());
        e.fits = false;
        return e;
    }
    match i.law {
        Law::Mock => {
            // the mock's arenas are plain host buffers: `--kv-gib` of KV at a nominal 16 KiB a token
            let per_token = 16 * 1024;
            let kv = i.mock_kv_gib * 1024 * MIB;
            e.vram_bytes = None;
            e.kv_bytes = Some(kv as i64);
            e.kv_bytes_per_token = Some(per_token);
            e.kv_capacity_tokens = Some(kv / per_token);
            e.seqs_at_ctx = Some((kv / per_token / i.max_ctx).min(i.max_seqs.max(1)));
            e.notes.push(
                "the mock engine: no GPU memory is used; the figures are the mock's `--kv-gib` \
                 arena at a nominal 16 KiB a token"
                    .into(),
            );
            e.fits = e.seqs_at_ctx.unwrap_or(0) >= 1;
            return e;
        }
        Law::Dense27b | Law::Flashnext => {}
    }
    let budget = match (i.budget_mib.map(|m| m * MIB), i.vram_per_card) {
        (Some(b), Some(v)) => {
            if b > v {
                e.warnings.push(format!(
                    "the budget ({}) is above the card's VRAM ({}): the VRAM binds",
                    gib(b as f64),
                    gib(v as f64)
                ));
            }
            Some(b.min(v))
        }
        (Some(b), None) => {
            e.notes
                .push("the card's VRAM is not readable here; using the budget flag".into());
            Some(b)
        }
        (None, v) => v,
    };
    e.budget_bytes = budget;
    let Some(budget) = budget else {
        e.notes
            .push("no budget and no VRAM reading: the estimate needs one".into());
        return e;
    };
    let Some(model) = &i.model else {
        e.notes.push(
            "the model's config.json is not readable (no KV shape): token figures unavailable"
                .into(),
        );
        return e;
    };
    e.notes.push(format!("model shape from {}", model.source));
    let per_token = model.kv_bytes_per_token(i.tp, i.kv_dtype_bytes);
    e.kv_bytes_per_token = Some(per_token);
    if let Some(mp) = model.max_position {
        if i.max_ctx > mp {
            e.warnings.push(format!(
                "max context {} is past the model's max_position_embeddings {mp}",
                i.max_ctx
            ));
        }
    }
    let weights = i.target.as_ref().and_then(|p| p.rank_weight_bytes);
    if weights.is_none() {
        e.notes
            .push("the pack's per-rank weight bytes are not readable: counted as 0".into());
    }
    let weights = weights.unwrap_or(0);
    e.weights_bytes = Some(weights);
    let drafter = i
        .drafter
        .as_ref()
        .and_then(|p| p.rank_weight_bytes)
        .unwrap_or(0);
    if i.drafter.is_some() {
        e.drafter_bytes = Some(drafter);
    }
    let per_seq_state = model.state_bytes_per_seq(i.tp, i.state_dtype_bytes);
    match i.law {
        Law::Dense27b => {
            let state = per_seq_state * i.max_seqs;
            let reserved = (ARENA_ALLOWANCE_MIB + i.kv_headroom_mib) * MIB;
            e.state_bytes = Some(state);
            e.reserved_bytes = Some(reserved);
            let kv = budget as i64 - (weights + drafter + state + reserved) as i64;
            e.kv_bytes = Some(kv);
            e.notes.push(format!(
                "reserved = {ARENA_ALLOWANCE_MIB} MiB arena allowance + the KV headroom ({} MiB)",
                i.kv_headroom_mib
            ));
            if i.tp == 1 {
                e.notes.push(
                    "TP=1: one card, so the single-card packs' whole weights, every KV head and \
                     each sequence's full GDN state come out of one budget"
                        .into(),
                );
            }
            if kv <= 0 {
                e.fits = false;
                e.kv_capacity_tokens = Some(0);
                e.seqs_at_ctx = Some(0);
                e.warnings
                    .push("the weights, state and allowance exceed the budget: no KV pool".into());
                return e;
            }
            let tokens = kv as u64 / per_token.max(1);
            e.kv_capacity_tokens = Some(tokens);
            let at_ctx = (tokens / i.max_ctx).min(i.max_seqs);
            e.seqs_at_ctx = Some(at_ctx);
            if at_ctx == 0 {
                e.fits = false;
                e.warnings.push(format!(
                    "one sequence at {} tokens needs {} of KV, the pool is {}",
                    i.max_ctx,
                    gib((i.max_ctx * per_token) as f64),
                    gib(kv as f64)
                ));
            } else if at_ctx < i.max_seqs {
                e.warnings.push(format!(
                    "{} of {} sequences fit at the full {}-token context (shorter ones share the pool)",
                    at_ctx, i.max_seqs, i.max_ctx
                ));
            }
        }
        Law::Flashnext => {
            let hot = (i.fn_kv_hot_gib * GIB) as u64;
            let state = per_seq_state * FN_SEQ_SLOTS;
            let reserved = ARENA_ALLOWANCE_MIB * MIB;
            e.state_bytes = Some(state);
            e.reserved_bytes = Some(reserved);
            e.kv_bytes = Some(hot as i64);
            let left = budget as i64 - (weights + state + reserved + hot) as i64;
            if let Some(stride) = i.target.as_ref().and_then(|p| p.expert_stride) {
                e.expert_slots = Some(left / stride as i64);
                e.expert_records = i.target.as_ref().and_then(|p| {
                    p.expert_records
                        .or_else(|| p.expert_bytes.map(|b| b / stride))
                });
            }
            if left <= 0 {
                e.fits = false;
                e.warnings
                    .push("the dense weights, state and hot pool exceed the budget".into());
            }
            let slab = i
                .fn_admit_slab_tokens
                .unwrap_or(i.max_ctx + FN_SLAB_EXTRA_TOKENS);
            e.kv_capacity_tokens = Some(slab);
            let ram = slab * per_token;
            e.kv_ram_bytes = Some(ram);
            e.seqs_at_ctx = Some((slab / i.max_ctx).min(FN_SEQ_SLOTS));
            e.requested_seqs = FN_SEQ_SLOTS;
            e.notes.push(
                "Flash-Next: the KV is tiered (a VRAM hot pool over pinned host RAM); capacity = \
                 the admission slab; the VRAM left after the dense weights goes to expert slots"
                    .into(),
            );
            if !i.fn_kv_tier_on {
                e.notes.push(
                    "the KV tier is off: the flat pools hold 8 x the max context in VRAM".into(),
                );
                let flat = FN_SEQ_SLOTS * i.max_ctx * per_token;
                e.kv_bytes = Some(flat as i64);
                let left = budget as i64 - (weights + state + reserved + flat) as i64;
                if let Some(stride) = i.target.as_ref().and_then(|p| p.expert_stride) {
                    e.expert_slots = Some(left / stride as i64);
                }
                if left <= 0 {
                    e.fits = false;
                    e.warnings
                        .push("the flat KV pools do not fit the budget".into());
                }
            } else if let Some(host) = i.host_ram {
                // both ranks pin their slab on the one host
                let both = ram * u64::from(i.tp.max(1));
                if both as f64 > host as f64 * 0.8 {
                    e.fits = false;
                    e.warnings.push(format!(
                        "the KV slab pins {} of host RAM ({} total): past the pinnable bytes the boot is refused",
                        gib(both as f64),
                        gib(host as f64)
                    ));
                }
            }
        }
        Law::Mock => unreachable!(),
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    const CFG_27B: &str = r#"{"text_config": {"num_hidden_layers": 64, "num_key_value_heads": 4,
        "head_dim": 256, "num_attention_heads": 24, "full_attention_interval": 4,
        "linear_num_value_heads": 48, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "max_position_embeddings": 262144}}"#;

    fn model() -> ModelInfo {
        ModelInfo::from_config_json(CFG_27B, "cfg").unwrap()
    }

    fn inputs() -> Inputs {
        Inputs {
            law: Law::Dense27b,
            tp: 2,
            vram_per_card: Some(32 * 1024 * MIB),
            budget_mib: Some(30_500),
            max_ctx: 262_144,
            max_seqs: 8,
            kv_dtype_bytes: 1,
            state_dtype_bytes: 2,
            target: Some(PackInfo {
                rank_weight_bytes: Some(10_631_356_416),
                ..PackInfo::default()
            }),
            drafter: Some(PackInfo {
                rank_weight_bytes: Some(3_800_000_000),
                ..PackInfo::default()
            }),
            model: Some(model()),
            kv_headroom_mib: 1024,
            fn_kv_hot_gib: 0.916,
            fn_admit_slab_tokens: None,
            fn_kv_tier_on: true,
            host_ram: Some(60 * 1024 * MIB),
            mock_kv_gib: 8,
        }
    }

    #[test]
    fn model_shape_from_config() {
        let m = model();
        assert_eq!((m.layers, m.attn_layers, m.linear_layers), (64, 16, 48));
        // 16 layers x 2 heads a rank x 256 x K,V x fp8
        assert_eq!(m.kv_bytes_per_token(2, 1), 16 * 2 * 256 * 2);
        assert_eq!(m.state_bytes_per_seq(2, 2), 48 * 24 * 128 * 128 * 2);
        // a config without layer_types or head_dim
        let m = ModelInfo::from_config_json(
            r#"{"num_hidden_layers": 8, "num_attention_heads": 8, "hidden_size": 1024}"#,
            "x",
        )
        .unwrap();
        assert_eq!((m.attn_layers, m.kv_heads, m.head_dim), (8, 8, 128));
        assert!(ModelInfo::from_config_json("{}", "x").is_err());
    }

    #[test]
    fn dense_law_counts_down_from_the_budget() {
        let e = estimate(&inputs());
        assert!(e.is_estimate);
        let per_token = 16 * 2 * 256 * 2;
        assert_eq!(e.kv_bytes_per_token, Some(per_token));
        let state = 8 * 48 * 24 * 128 * 128 * 2;
        let kv = 30_500 * MIB - 10_631_356_416 - 3_800_000_000 - state - (2048 + 1024) * MIB;
        assert_eq!(e.kv_bytes, Some(kv as i64));
        assert_eq!(e.kv_capacity_tokens, Some(kv / per_token));
        assert_eq!(e.seqs_at_ctx, Some((kv / per_token / 262_144).min(8)));
        assert!(e.fits);
        // a context the pool cannot hold once does not fit
        let mut i = inputs();
        i.max_ctx = 1_048_576 * 4;
        let e = estimate(&i);
        assert_eq!(e.seqs_at_ctx, Some(0));
        assert!(!e.fits);
        assert!(e.warnings.iter().any(|w| w.contains("max_position")));
        // weights past the budget
        let mut i = inputs();
        i.budget_mib = Some(8_000);
        assert!(!estimate(&i).fits);
        // the VRAM binds below the budget flag
        let mut i = inputs();
        i.vram_per_card = Some(16 * 1024 * MIB);
        let e = estimate(&i);
        assert_eq!(e.budget_bytes, Some(16 * 1024 * MIB));
        assert!(e.warnings.iter().any(|w| w.contains("VRAM binds")));
    }

    #[test]
    fn tp1_one_card_holds_everything() {
        // the NVFP4 TP=1 pair (rank0.pack bytes of the -tp1 packs) at the launcher's TP=1 defaults
        let mut i = inputs();
        i.tp = 1;
        i.max_ctx = 65_536;
        i.max_seqs = 4;
        i.budget_mib = Some(LEAN_BUDGET_MIB);
        i.kv_headroom_mib = LEAN_HEADROOM_MIB;
        i.target = Some(PackInfo {
            rank_weight_bytes: Some(18_534_289_408),
            ..PackInfo::default()
        });
        i.drafter = Some(PackInfo {
            rank_weight_bytes: Some(6_397_935_616),
            ..PackInfo::default()
        });
        let e = estimate(&i);
        assert_eq!(e.tp, 1);
        assert!(e.label.contains("ONE card"), "{}", e.label);
        // every KV head on the one card: twice TP=2's bytes a token; the whole GDN state
        assert_eq!(e.kv_bytes_per_token, Some(16 * 4 * 256 * 2));
        assert_eq!(e.state_bytes, Some(4 * 48 * 48 * 128 * 128 * 2));
        let kv = LEAN_BUDGET_MIB * MIB
            - 18_534_289_408
            - 6_397_935_616
            - 4 * 48 * 48 * 128 * 128 * 2
            - (ARENA_ALLOWANCE_MIB + LEAN_HEADROOM_MIB) * MIB;
        assert_eq!(e.kv_bytes, Some(kv as i64));
        // (the measured pool of this pair at these defaults: 168,768 tokens, README "One card")
        let tokens = e.kv_capacity_tokens.unwrap();
        assert!((120_000..=180_000).contains(&tokens), "{tokens}");
        assert_eq!(e.seqs_at_ctx, Some(tokens / 65_536));
        assert!(e.fits);
        assert!(e.notes.iter().any(|n| n.starts_with("TP=1")));
        // TP=2 reads per card
        let e2 = estimate(&inputs());
        assert_eq!(e2.tp, 2);
        assert!(e2.label.contains("per card"), "{}", e2.label);
    }

    #[test]
    fn seqs_shrink_as_ctx_grows() {
        let mut last = u64::MAX;
        for ctx in [4_096u64, 32_768, 131_072, 262_144] {
            let mut i = inputs();
            i.max_ctx = ctx;
            i.max_seqs = 64;
            let s = estimate(&i).seqs_at_ctx.unwrap();
            assert!(s <= last, "{ctx}: {s} > {last}");
            last = s;
        }
    }

    #[test]
    fn flashnext_law_slab_and_expert_slots() {
        let mut i = inputs();
        i.law = Law::Flashnext;
        i.max_ctx = 262_144;
        i.target = Some(PackInfo {
            rank_weight_bytes: Some(5_000_000_000),
            expert_bytes: Some(12_288 * 2_617_344),
            expert_stride: Some(2_617_344),
            ..PackInfo::default()
        });
        i.drafter = None;
        let e = estimate(&i);
        assert_eq!(e.kv_capacity_tokens, Some(262_144 + FN_SLAB_EXTRA_TOKENS));
        assert_eq!(e.seqs_at_ctx, Some(1));
        assert_eq!(e.requested_seqs, 8);
        assert_eq!(e.expert_records, Some(12_288));
        assert!(e.expert_slots.unwrap() > 0);
        assert!(e.fits);
        // the explicit slab
        i.fn_admit_slab_tokens = Some(338_592);
        i.max_ctx = 32_768;
        let e = estimate(&i);
        assert_eq!(e.seqs_at_ctx, Some(8));
        // a slab past the host's RAM
        i.fn_admit_slab_tokens = Some(40_000_000);
        assert!(!estimate(&i).fits);
        // tier off: the flat pools come out of the expert slots
        let mut j = i.clone();
        j.fn_admit_slab_tokens = None;
        j.fn_kv_tier_on = false;
        j.max_ctx = 8_192;
        let flat = estimate(&j);
        let mut k = j.clone();
        k.max_ctx = 16_384;
        assert!(estimate(&k).expert_slots < flat.expert_slots);
    }

    #[test]
    fn missing_inputs_are_named() {
        let mut i = inputs();
        i.model = None;
        let e = estimate(&i);
        assert!(e.kv_capacity_tokens.is_none());
        assert!(e.notes.iter().any(|n| n.contains("config.json")));
        let mut i = inputs();
        i.law = Law::Mock;
        let e = estimate(&i);
        assert_eq!(e.kv_capacity_tokens, Some(8 * 1024 * MIB / (16 * 1024)));
    }

    #[test]
    fn manifests_of_both_forms() {
        let m27 = r#"{"tp_size": 2, "builder": "tools/pack/build_pack.py",
            "tier": {"descriptor": {"model": "ckpt-w8"}},
            "ranks": [{"rank": 0, "file_bytes": 100}, {"rank": 1, "file_bytes": 101}]}"#;
        let p = parse_manifest("qwen38-27b-x", m27);
        assert_eq!(p.rank_weight_bytes, Some(100));
        assert_eq!(p.tp_size, Some(2));
        assert_eq!(p.role, "target");
        assert_eq!(p.model.as_deref(), Some("ckpt-w8"));
        let fnm = r#"{"model": "qwen4_exp", "layout": {"ep": 2, "tp": 2},
            "checkpoint": {"dir": "/m/ck", "shards": [{"size_bytes": 5}, {"size_bytes": 7}]},
            "ledger": [{"placement": "rank0", "class": "experts", "bytes": 1000},
                       {"placement": "rank0", "class": "gdn", "bytes": 10},
                       {"placement": "rank0", "class": "embed", "bytes": 99},
                       {"placement": "rank1", "class": "gdn", "bytes": 10},
                       {"placement": "shared", "class": "hc", "bytes": 3}],
            "expert_tiers": [{"records": 20, "record": {"stride": 50}}]}"#;
        let p = parse_manifest("qwen38-flashnext-x", fnm);
        assert_eq!(p.rank_weight_bytes, Some(13));
        assert_eq!(p.expert_bytes, Some(1000));
        assert_eq!(p.expert_stride, Some(50));
        assert_eq!(p.expert_records, Some(20));
        assert_eq!(p.referenced_bytes, 12);
        assert_eq!(p.checkpoint_dir.as_deref(), Some("/m/ck"));
        let d = parse_manifest(
            "qwen38-dflash2-drafter-q",
            r#"{"builder": "tools/pack/drafter_head_pair.py", "ranks": []}"#,
        );
        assert_eq!(d.role, "drafter");
        assert!(parse_manifest("x", "not json").error.is_some());
    }
}
