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
    /// 27B: rank 0's records without the token embedding (it is not a device weight).
    pub device_weight_bytes: Option<u64>,
    /// A drafter pack: its head records shared with the target's head (billed once).
    pub shared_head_bytes: u64,
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
        let r0 = ranks.iter().find(|r| r["rank"].as_u64() == Some(0));
        p.rank_weight_bytes = r0.and_then(|r| r["file_bytes"].as_u64());
        if let Some(recs) = r0.and_then(|r| r["records"].as_array()) {
            let len = |r: &serde_json::Value| r["length"].as_u64().unwrap_or(0);
            let name = |r: &serde_json::Value| r["name"].as_str().unwrap_or("").to_string();
            p.device_weight_bytes = Some(
                recs.iter()
                    .filter(|r| !name(r).contains("embed"))
                    .map(len)
                    .sum(),
            );
            p.shared_head_bytes = recs
                .iter()
                .filter(|r| name(r).starts_with("head.rerank."))
                .map(len)
                .sum();
        }
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
    /// 27B: `--engine-kv-headroom-mib` (the lean ledger's when that law is on).
    pub kv_headroom_mib: u64,
    /// 27B: the kernel's P2PDMA pool a card (MiB), which the engine's ledger gives up.
    pub p2p_pool_mib: u64,
    /// 27B: the drafter's block (`--engine-draft-block`; 0 = no drafter tier).
    pub draft_block: u64,
    /// 27B: the decode batches the engine runs.
    pub decode_batches: Vec<u64>,
    /// 27B: the GDN replay (its stash) is on.
    pub gdn_replay: bool,
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
    /// 27B: the drafter's KV pages (paged by admission) and the target's KV blocks.
    pub draft_kv_bytes: Option<u64>,
    pub kv_blocks: Option<u64>,
    /// 27B: KV blocks left after the sequences that fit at the full context are admitted (the
    /// prefix cache's room; the engine's `admit ... left over`).
    pub spare_blocks: Option<u64>,
    /// 27B: the longest context the engine boots at this many sequences.
    pub ctx_ceiling: Option<u64>,
    /// 27B: why the engine will refuse to start (its boot check: one sequence at the max
    /// context must fit the ledger's blocks). A validation error.
    pub refusal: Option<String>,
    /// 27B: the KV the same config would get with fewer sequences or a shorter context (the
    /// "KV vs context vs sequences" tip; computed by the same law).
    pub alternatives: Vec<Alternative>,
}

/// One "what if" of the KV tip.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Alternative {
    /// `seqs` (fewer sequences) or `ctx` (a shorter context).
    pub kind: &'static str,
    pub max_ctx: u64,
    pub max_seqs: u64,
    /// The decode batches it would run (the configured ones that fit the sequences).
    pub decode_batches: Vec<u64>,
    pub kv_capacity_tokens: u64,
    pub seqs_at_ctx: u64,
    /// The longest context the engine boots at these sequences.
    pub ctx_ceiling: Option<u64>,
}

/// The engine's `lean` ledger law (`saber_cli::ledger_law`, the default): the
/// fence and the KV headroom it uses in place of the two flags. A `saber-cli` test keeps these
/// equal to the engine's.
pub const LEAN_BUDGET_MIB: u64 = 31_744;
pub const LEAN_HEADROOM_MIB: u64 = 848;

/// Flash-Next: a fixed per-rank allowance for the arenas, graph pools and heads the estimate
/// does not model.
pub const ARENA_ALLOWANCE_MIB: u64 = 2_048;

/// The 27B ledger's constants, fitted to five TP=1 boots of the MACaQ `-tp1` pair (2026-11-14,
/// ROCm 10.1, frozen main 03416d1f; contexts 65,536 / 32,768 / 16,384 and 4 / 2 / 1 sequences:
/// every KV block count within 0.1 %) and checked against the NVFP4 `-tp1` pair's measured
/// 168,768 tokens (-1 %):
/// - the arenas billed before the pool that do not scale (the shared prefill arena 641 MiB, the
///   drafter's aux / store buffers, the indices), MiB a rank;
pub const FIXED_ARENA_MIB: u64 = 681;
/// - the weights the engine bills beyond the packs' records (embedding excluded, the shared head
///   counted once), MiB at TP=1 (divided by the TP);
pub const WEIGHT_EXTRA_MIB: u64 = 761;
/// - the GDN replay stash, bytes a GDN layer a sequence at TP=1;
pub const REPLAY_STASH_BYTES_PER_LAYER_SEQ: u64 = 405_504;
/// - the DFlash2 drafter's context KV, bytes a token at TP=1 (pages of 16 tokens);
pub const DRAFT_KV_BYTES_PER_TOKEN: u64 = 10_240;
/// - the drafter's rope table, bytes a context position (built at load at `max_ctx`);
pub const DRAFT_ROPE_BYTES_PER_POS: u64 = 256;
/// - tokens a KV block / drafter page.
pub const BLOCK_TOKENS: u64 = 16;

/// The decode arenas, MiB, by the largest decode batch (fitted: 1 -> 0, 2 -> 21, 4 -> 75).
#[must_use]
pub fn decode_arena_mib(largest_batch: u64) -> u64 {
    let m = largest_batch.max(1) - 1;
    19 * m + 2 * m * m
}

/// The configured decode batches the engine can run with `seqs` sequences (it refuses a batch
/// larger than its sequences); never empty.
#[must_use]
pub fn trim_batches(batches: &[u64], seqs: u64) -> Vec<u64> {
    let mut b: Vec<u64> = batches
        .iter()
        .copied()
        .filter(|&x| x <= seqs.max(1))
        .collect();
    if b.is_empty() {
        b.push(1);
    }
    b
}

/// One configuration's 27B ledger figures (bytes a rank).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DenseFigures {
    pub budget: u64,
    pub weights: u64,
    pub drafter_weights: u64,
    pub state: u64,
    pub reserved: u64,
    pub draft_kv: u64,
    pub pool: i64,
    pub blocks: u64,
    pub tokens: u64,
    /// The blocks the engine's boot check counts (the KV budget over a target block plus a
    /// drafter page, before the drafter is sized): one sequence at `max_ctx` needs
    /// `max_ctx / 16 + 1` of them or the engine refuses to start.
    pub ledger_blocks: u64,
}

impl DenseFigures {
    /// The blocks one sequence at `max_ctx` needs (the engine's boot check).
    #[must_use]
    pub fn needed(max_ctx: u64) -> u64 {
        max_ctx.div_ceil(BLOCK_TOKENS) + 1
    }
}

/// The longest context (a multiple of 1,024) the engine boots at `seqs` sequences, by its own
/// boot check (`None` = not even 4,096).
#[must_use]
pub fn ctx_ceiling(i: &Inputs, model: &ModelInfo, seqs: u64, batches: &[u64]) -> Option<u64> {
    let ok =
        |c: u64| dense_figures(i, model, c, seqs, batches).ledger_blocks >= DenseFigures::needed(c);
    if !ok(4_096) {
        return None;
    }
    let (mut lo, mut hi) = (4u64, 4_096u64);
    while hi < 4_096 && ok(hi * 1_024) {
        lo = hi;
        hi *= 2;
    }
    // ok(lo * 1024), and hi is past it or the 4M-token cap
    if ok(hi * 1_024) {
        return Some(hi * 1_024);
    }
    while hi - lo > 1 {
        let m = (lo + hi) / 2;
        if ok(m * 1_024) {
            lo = m;
        } else {
            hi = m;
        }
    }
    Some(lo * 1_024)
}

/// The 27B ledger law (see [`FIXED_ARENA_MIB`]) for `max_ctx` / `seqs` / `batches`.
#[must_use]
pub fn dense_figures(
    i: &Inputs,
    model: &ModelInfo,
    max_ctx: u64,
    seqs: u64,
    batches: &[u64],
) -> DenseFigures {
    let tp = u64::from(i.tp.max(1));
    let budget = match (i.budget_mib.map(|m| m * MIB), i.vram_per_card) {
        (Some(b), Some(v)) => b.min(v),
        (Some(b), None) => b,
        (None, v) => v.unwrap_or(0),
    }
    .saturating_sub(i.p2p_pool_mib * MIB);
    let dev = |p: &Option<PackInfo>| {
        p.as_ref()
            .and_then(|p| p.device_weight_bytes.or(p.rank_weight_bytes))
            .unwrap_or(0)
    };
    let weights = dev(&i.target) + WEIGHT_EXTRA_MIB * MIB / tp;
    let has_drafter = i.drafter.is_some() && i.draft_block > 0;
    let drafter_weights = if has_drafter {
        let d = i.drafter.as_ref().unwrap();
        dev(&i.drafter).saturating_sub(d.shared_head_bytes) + max_ctx * DRAFT_ROPE_BYTES_PER_POS
    } else {
        0
    };
    let largest = batches.iter().copied().max().unwrap_or(1);
    let fixed = (FIXED_ARENA_MIB + decode_arena_mib(largest)) * MIB;
    let kv_budget = budget as i64 - (weights + drafter_weights + fixed) as i64;
    // the GDN state slots: (draft block + 1) snapshots a sequence + the null slot
    let slot = model.state_bytes_per_seq(i.tp, i.state_dtype_bytes) * 10_046 / 10_000;
    let snaps = if has_drafter { i.draft_block + 1 } else { 1 };
    let state = (seqs * snaps + 1) * slot + 9 * MIB / tp;
    let stash = if i.gdn_replay {
        seqs * u64::from(model.linear_layers) * REPLAY_STASH_BYTES_PER_LAYER_SEQ / tp
    } else {
        0
    };
    let reserved = stash + i.kv_headroom_mib * MIB;
    let rest = kv_budget - (state + reserved) as i64;
    let block = BLOCK_TOKENS * model.kv_bytes_per_token(i.tp, i.kv_dtype_bytes);
    let page = BLOCK_TOKENS * DRAFT_KV_BYTES_PER_TOKEN / tp;
    let ledger_blocks = if rest <= 0 || block == 0 {
        0
    } else if has_drafter {
        rest as u64 / (block + page)
    } else {
        rest as u64 / block
    };
    let (blocks, draft_kv) = if rest <= 0 || block == 0 {
        (0, 0)
    } else if !has_drafter {
        (rest as u64 / block, 0)
    } else {
        // the drafter's pages follow the target's blocks, at most its dense size
        let dense = seqs * max_ctx.div_ceil(BLOCK_TOKENS) * page;
        let shared = rest as u64 / (block + page);
        if shared * page <= dense {
            (shared, shared * page)
        } else {
            ((rest as u64).saturating_sub(dense) / block, dense)
        }
    };
    DenseFigures {
        budget,
        weights,
        drafter_weights,
        state,
        reserved,
        draft_kv,
        pool: (blocks * block) as i64,
        blocks,
        tokens: blocks * BLOCK_TOKENS,
        ledger_blocks,
    }
}
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
            let f = dense_figures(i, model, i.max_ctx, i.max_seqs, &i.decode_batches);
            e.budget_bytes = Some(f.budget);
            e.weights_bytes = Some(f.weights);
            if i.drafter.is_some() {
                e.drafter_bytes = Some(f.drafter_weights);
            }
            e.state_bytes = Some(f.state);
            e.reserved_bytes = Some(f.reserved);
            e.draft_kv_bytes = Some(f.draft_kv);
            e.kv_bytes = Some(f.pool);
            e.kv_bytes_per_token = Some(per_token);
            e.kv_blocks = Some(f.blocks);
            e.notes.push(format!(
                "the 27B ledger law: the budget less the kernel's P2P pool ({} MiB), the weights the \
                 engine bills (the packs' records without the token embedding, the drafter's head \
                 shared with the target's), {} MiB of fixed arenas, the drafter's rope table and \
                 the decode arenas leave the KV budget; the GDN state slots ((block + 1) a \
                 sequence + 1), the replay stash and the {} MiB headroom come off it, and the \
                 target's KV blocks share the rest with the drafter's KV pages (paged by \
                 admission, at most the drafter's dense size); fitted to five measured TP=1 boots \
                 within 0.1 %",
                i.p2p_pool_mib,
                FIXED_ARENA_MIB + WEIGHT_EXTRA_MIB / u64::from(i.tp.max(1)),
                i.kv_headroom_mib
            ));
            if i.tp == 1 {
                e.notes.push(
                    "TP=1: one card, so the single-card packs' whole weights, every KV head and \
                     each sequence's full GDN state come out of one budget"
                        .into(),
                );
            } else {
                e.notes.push(
                    "TP=2: the same law per card (the TP=2 constants are scaled, not measured \
                     on this law: expect a few % of error)"
                        .into(),
                );
            }
            let ceiling = ctx_ceiling(i, model, i.max_seqs, &i.decode_batches);
            e.ctx_ceiling = ceiling;
            let needed = DenseFigures::needed(i.max_ctx);
            if f.ledger_blocks < needed {
                let why = format!(
                    "the engine will refuse to start: one sequence at {} tokens needs {needed} KV \
                     blocks and the ledger leaves {} at {} sequence(s); the longest context at {} \
                     sequence(s) is about {}",
                    i.max_ctx,
                    f.ledger_blocks,
                    i.max_seqs,
                    i.max_seqs,
                    ceiling.map_or("under 4,096".to_string(), |c| c.to_string())
                );
                e.warnings.push(why.clone());
                e.refusal = Some(why);
                e.fits = false;
            }
            if f.blocks == 0 {
                e.fits = false;
                e.kv_capacity_tokens = Some(0);
                e.seqs_at_ctx = Some(0);
                e.warnings
                    .push("the weights, state and arenas exceed the budget: no KV pool".into());
                return e;
            }
            let tokens = f.tokens;
            e.kv_capacity_tokens = Some(tokens);
            let at_ctx = (tokens / i.max_ctx).min(i.max_seqs);
            e.seqs_at_ctx = Some(at_ctx);
            e.spare_blocks = Some(
                f.blocks
                    .saturating_sub(at_ctx * i.max_ctx.div_ceil(BLOCK_TOKENS)),
            );
            if at_ctx == 0 {
                e.fits = false;
                if e.refusal.is_none() {
                    e.warnings.push(format!(
                        "one sequence at {} tokens needs {} of KV, the pool is {}",
                        i.max_ctx,
                        gib((i.max_ctx * per_token) as f64),
                        gib(f.pool as f64)
                    ));
                }
            } else if at_ctx < i.max_seqs {
                e.warnings.push(format!(
                    "{} of {} sequences fit at the full {}-token context (shorter ones share the pool)",
                    at_ctx, i.max_seqs, i.max_ctx
                ));
            }
            // what other sequence counts or a shorter context would give (the KV tip); a
            // sequence count runs at the current context when it boots there, else at its ceiling
            let mut alts = Vec::new();
            let mut push = |kind: &'static str, ctx: u64, seqs: u64| {
                if (ctx, seqs) == (i.max_ctx, i.max_seqs) || seqs == 0 || ctx < 4_096 {
                    return;
                }
                // the console derives the batches from the sequences (the TP's ladder trimmed)
                let ladder: &[u64] = if i.tp == 1 {
                    &[1, 2, 4, 8]
                } else {
                    &[1, 2, 3, 4, 8]
                };
                let b = trim_batches(ladder, seqs);
                let ceil = ctx_ceiling(i, model, seqs, &b);
                let ctx = match ceil {
                    Some(c) if c < ctx => c / 4_096 * 4_096,
                    _ => ctx,
                };
                let g = dense_figures(i, model, ctx, seqs, &b);
                alts.push(Alternative {
                    kind,
                    max_ctx: ctx,
                    max_seqs: seqs,
                    decode_batches: b,
                    kv_capacity_tokens: g.tokens,
                    seqs_at_ctx: (g.tokens / ctx).min(seqs),
                    ctx_ceiling: ceil,
                });
            };
            for s in [1, 2, 4] {
                if s != i.max_seqs {
                    // several sequences are shown at a shared context (at most 65,536)
                    let c = if s == 1 {
                        i.max_ctx
                    } else {
                        i.max_ctx.min(65_536)
                    };
                    push("seqs", c, s);
                }
            }
            for c in [i.max_ctx / 2, i.max_ctx / 4] {
                push("ctx", c, i.max_seqs);
            }
            e.alternatives = alts;
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
            p2p_pool_mib: 0,
            draft_block: 8,
            decode_batches: vec![1, 2, 3, 4, 8],
            gdn_replay: true,
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

    /// The MACaQ `-tp1` pair's records (rank 0 without the embedding; the drafter's head shared).
    fn macaq_tp1() -> (PackInfo, PackInfo) {
        (
            PackInfo {
                device_weight_bytes: Some(17_674_337_312),
                ..PackInfo::default()
            },
            PackInfo {
                device_weight_bytes: Some(5_125_366_016),
                shared_head_bytes: 2_542_796_800,
                ..PackInfo::default()
            },
        )
    }

    /// The engine's TP=1 boot on a two-card R9700 host: the lean ledger, the 2 GiB P2P pool, block 8.
    fn tp1(target: PackInfo, drafter: PackInfo, ctx: u64, seqs: u64, batches: &[u64]) -> Inputs {
        let mut i = inputs();
        i.tp = 1;
        i.vram_per_card = Some(32_624 * MIB);
        i.budget_mib = Some(LEAN_BUDGET_MIB);
        i.kv_headroom_mib = LEAN_HEADROOM_MIB;
        i.p2p_pool_mib = 2_048;
        i.max_ctx = ctx;
        i.max_seqs = seqs;
        i.decode_batches = batches.to_vec();
        i.target = Some(target);
        i.drafter = Some(drafter);
        i
    }

    /// Five measured TP=1 boots of the MACaQ `-tp1` pair (card 0, 210 W, 2026-11-14, frozen
    /// main 03416d1f; `kvpool` boot lines): the law reproduces every KV block count within 0.1 %.
    #[test]
    fn tp1_macaq_matches_the_five_measured_boots() {
        let measured = [
            (65_536, 4, vec![1, 2, 4], 7_982u64),
            (32_768, 4, vec![1, 2, 4], 7_994),
            (16_384, 4, vec![1, 2, 4], 9_222),
            (65_536, 1, vec![1], 13_364),
            (65_536, 2, vec![1, 2], 10_703),
        ];
        let (t, d) = macaq_tp1();
        for (ctx, seqs, b, blocks) in measured {
            let e = estimate(&tp1(t.clone(), d.clone(), ctx, seqs, &b));
            let got = e.kv_blocks.unwrap();
            let err = (got as f64 - blocks as f64).abs() / blocks as f64;
            eprintln!("ctx {ctx} seqs {seqs}: law {got} blocks, measured {blocks} ({err:.5})");
            assert!(
                err < 0.001,
                "ctx {ctx} seqs {seqs}: {got} vs {blocks} blocks"
            );
            assert_eq!(e.kv_capacity_tokens, Some(got * 16));
            assert!(e.fits, "ctx {ctx} seqs {seqs}: {:?}", e.warnings);
        }
        // the default fits: 1 full 65,536 session of 4 (the engine admitted exactly 1)
        let a = estimate(&tp1(t.clone(), d.clone(), 65_536, 4, &[1, 2, 4]));
        assert_eq!(a.seqs_at_ctx, Some(1));
        assert!(a.label.contains("ONE card"));
        assert!(
            a.warnings.iter().any(|w| w.contains("1 of 4")),
            "{:?}",
            a.warnings
        );
        // GDN slots: (8 + 1) x 4 + 1 of the per-sequence state + 9 MiB = 2,685 MiB (the boot's)
        let gdn = a.state_bytes.unwrap() / MIB;
        assert!((2_683..=2_687).contains(&gdn), "{gdn}");
        // the tip: fewer sequences give the most KV, a shorter context little until 16k
        let alt = |kind: &str, ctx: u64, seqs: u64| {
            a.alternatives
                .iter()
                .find(|x| x.kind == kind && x.max_ctx == ctx && x.max_seqs == seqs)
                .unwrap_or_else(|| panic!("{kind} {ctx} {seqs}: {:?}", a.alternatives))
                .clone()
        };
        let s2 = alt("seqs", 65_536, 2);
        assert_eq!(s2.decode_batches, [1, 2]);
        assert!(
            (171_000..=171_500).contains(&s2.kv_capacity_tokens),
            "{s2:?}"
        );
        let s1 = alt("seqs", 65_536, 1);
        assert_eq!(s1.decode_batches, [1]);
        assert!(
            (213_600..=214_100).contains(&s1.kv_capacity_tokens),
            "{s1:?}"
        );
        let c32 = alt("ctx", 32_768, 4);
        assert!(c32.kv_capacity_tokens < a.kv_capacity_tokens.unwrap() + 1_000);
        eprintln!("alternatives: {:?}", a.alternatives);
        let c16 = alt("ctx", 16_384, 4);
        assert!(
            (147_300..=147_800).contains(&c16.kv_capacity_tokens),
            "{c16:?}"
        );
    }

    /// Seven more measured TP=1 boots at 1 sequence (batches 1): the ledger's KV blocks and the
    /// spare blocks after the one admitted session within 0.1 %, 176,128 boots and 180,224 /
    /// 196,608 / 262,144 are refused with the engine's own block counts.
    #[test]
    fn tp1_macaq_one_sequence_ceiling_matches_the_engine() {
        let (t, d) = macaq_tp1();
        // (ctx, the kvpool blocks, the blocks left after the one session)
        let boots = [
            (65_536u64, 13_364u64, 9_268u64),
            (131_072, 12_052, 3_860),
            (163_840, 11_396, 1_156),
            (172_032, 11_232, 480),
            (176_128, 11_150, 142),
        ];
        for (ctx, blocks, spare) in boots {
            let e = estimate(&tp1(t.clone(), d.clone(), ctx, 1, &[1]));
            let got = e.kv_blocks.unwrap();
            eprintln!(
                "1 seq ctx {ctx}: law {got} blocks (measured {blocks}), spare {:?} ({spare})",
                e.spare_blocks
            );
            assert!(
                got.abs_diff(blocks) * 1_000 <= blocks,
                "ctx {ctx}: {got} vs {blocks}"
            );
            assert!(
                e.spare_blocks.unwrap().abs_diff(spare) <= 10,
                "ctx {ctx}: {:?}",
                e.spare_blocks
            );
            assert!(e.refusal.is_none() && e.fits, "ctx {ctx}: {:?}", e.warnings);
            assert_eq!(e.seqs_at_ctx, Some(1));
        }
        // the engine's refusals: (ctx, its ledger blocks)
        for (ctx, ledger) in [
            (180_224u64, 11_114u64),
            (196_608, 11_108),
            (262_144, 11_084),
        ] {
            let i = tp1(t.clone(), d.clone(), ctx, 1, &[1]);
            let f = dense_figures(&i, &model(), ctx, 1, &[1]);
            eprintln!(
                "1 seq ctx {ctx}: law ledger {} blocks (engine {ledger})",
                f.ledger_blocks
            );
            assert!(
                f.ledger_blocks.abs_diff(ledger) <= 3,
                "{ctx}: {}",
                f.ledger_blocks
            );
            let e = estimate(&i);
            assert!(!e.fits);
            let r = e.refusal.as_deref().unwrap();
            assert!(r.contains("will refuse to start"), "{r}");
        }
        // the ceiling at 1 sequence lies between the last boot and the first refusal
        let c = ctx_ceiling(
            &tp1(t.clone(), d.clone(), 131_072, 1, &[1]),
            &model(),
            1,
            &[1],
        )
        .unwrap();
        eprintln!("ceiling at 1 sequence: {c}");
        assert!((176_128..180_224).contains(&c), "{c}");
        // the new TP=1 default: 1 sequence at 131,072 fits with ~62k tokens spare; the tip
        // offers 2 / 4 sequences at 65,536 and names each one's ceiling
        let e = estimate(&tp1(t.clone(), d.clone(), 131_072, 1, &[1]));
        assert_eq!(e.ctx_ceiling, Some(c));
        assert!(e.spare_blocks.unwrap() * 16 > 60_000);
        let two = e
            .alternatives
            .iter()
            .find(|a| a.kind == "seqs" && a.max_seqs == 2)
            .unwrap();
        assert_eq!(
            (two.max_ctx, two.decode_batches.clone()),
            (65_536, vec![1, 2])
        );
        assert!(
            (170_000..=172_500).contains(&two.kv_capacity_tokens),
            "{two:?}"
        );
        let four = e
            .alternatives
            .iter()
            .find(|a| a.kind == "seqs" && a.max_seqs == 4)
            .unwrap();
        assert_eq!(four.decode_batches, [1, 2, 4]);
        assert!(
            (126_500..=129_000).contains(&four.kv_capacity_tokens),
            "{four:?}"
        );
        assert!(four.ctx_ceiling.unwrap() < c);
    }

    /// The NVFP4 `-tp1` pair at the launcher's TP=1 defaults: the README's measured 168,768 KV
    /// tokens (an earlier binary) within 2 %.
    #[test]
    fn tp1_nvfp4_matches_its_measured_pool() {
        let t = PackInfo {
            device_weight_bytes: Some(15_979_828_000),
            ..PackInfo::default()
        };
        let d = PackInfo {
            device_weight_bytes: Some(3_854_960_896),
            shared_head_bytes: 1_272_391_680,
            ..PackInfo::default()
        };
        let e = estimate(&tp1(t, d, 65_536, 4, &[1, 2, 4]));
        let tokens = e.kv_capacity_tokens.unwrap() as f64;
        eprintln!("NVFP4 TP=1 defaults: law {tokens} tokens, measured 168,768");
        assert!((tokens - 168_768.0).abs() / 168_768.0 < 0.02, "{tokens}");
        assert_eq!(e.seqs_at_ctx, Some(2));
    }

    #[test]
    fn the_dense_law_without_a_drafter_and_past_the_budget() {
        let (t, _) = macaq_tp1();
        let mut i = tp1(t.clone(), PackInfo::default(), 65_536, 4, &[1, 2, 4]);
        i.drafter = None;
        i.draft_block = 0;
        let e = estimate(&i);
        // no drafter: one GDN slot a sequence and every byte to the target's blocks
        assert_eq!(e.draft_kv_bytes, Some(0));
        assert!(e.kv_capacity_tokens.unwrap() > 300_000);
        // weights past the budget
        i.budget_mib = Some(20_000);
        let e = estimate(&i);
        assert!(!e.fits);
        assert_eq!(e.kv_capacity_tokens, Some(0));
        // the batches the engine can run with fewer sequences
        assert_eq!(trim_batches(&[1, 2, 4], 2), [1, 2]);
        assert_eq!(trim_batches(&[1, 2, 3, 4, 8], 3), [1, 2, 3]);
        assert_eq!(trim_batches(&[2, 4], 1), [1]);
        assert_eq!(decode_arena_mib(1), 0);
        assert_eq!(decode_arena_mib(2), 21);
        assert_eq!(decode_arena_mib(4), 75);
    }

    /// TP=2 (not measured on this law): the MACaQ TP=2 pair at the served 262,144 x 8 lands in
    /// the README's ~740k-token pool's neighbourhood.
    #[test]
    fn tp2_macaq_is_in_the_measured_neighbourhood() {
        let mut i = tp1(
            PackInfo {
                device_weight_bytes: Some(8_915 * MIB),
                ..PackInfo::default()
            },
            PackInfo {
                device_weight_bytes: Some(2_692 * MIB),
                shared_head_bytes: 1_212 * MIB,
                ..PackInfo::default()
            },
            262_144,
            8,
            &[1, 2, 3, 4, 8],
        );
        i.tp = 2;
        let e = estimate(&i);
        let t = e.kv_capacity_tokens.unwrap() as f64;
        eprintln!("MACaQ TP=2 262k x 8: law {t} tokens");
        assert!((640_000.0..=800_000.0).contains(&t), "{t}");
        assert!(e.label.contains("per card"));
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
