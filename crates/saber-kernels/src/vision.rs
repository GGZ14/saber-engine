//! The vision tower (workstream D1): Qwen3.8-27B's `model.visual`, planned as a list of launches
//! over the pack's 333 `model.visual.*` records.
//!
//! Prod's implementation is `Qwen3_VisionTransformer` (the reference server `qwen3_vl.py`), run in ATen. This
//! module is the host half of Saber's: the rules table (`rules/vision_gfx1201.toml`), the tower's
//! launch list for a set of image grids ([`plan_tower`]), and the two pieces of per-image host
//! metadata the kernels read -- the pos-embed bilinear gather table ([`pos_table`]) and the 2-D
//! rope cos/sin rows ([`rope_table`]) -- each written to reproduce the float32 arithmetic of the
//! torch expression prod evaluates, because a table that is off by one ulp is an input the gate
//! would blame on a kernel.
//!
//! The launch count is static per grid (I7): `3 + 8 * depth + 3`, asserted by [`plan_tower`].
//! Nothing here touches HIP; `tools/capture/vision_gate.py` executes the plan.

use serde::{Deserialize, Serialize};

use crate::abi::{self, SigArg};
use crate::rules::{ArgValue, LaunchDescriptor, Slot};
use crate::{Error, Result};

pub const PROD_RULES_TEXT: &str = include_str!("../rules/vision_gfx1201.toml");
pub const PLUGIN: &str = "vision";
/// FN55: Flash-Next's `vision_config.out_hidden_size` (its text hidden).
pub const FLASHNEXT_OUT_HIDDEN: u32 = 2560;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct VisionModel {
    pub depth: u32,
    pub hidden: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub intermediate: u32,
    pub out_hidden: u32,
    pub patch: u32,
    pub temporal_patch: u32,
    pub in_channels: u32,
    pub spatial_merge: u32,
    pub num_position_embeddings: u32,
    pub eps: f32,
    pub rotary_dim: u32,
    pub rope_theta: f32,
    pub deepstack_visual_indexes: Vec<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct VisionKernels {
    pub cast: String,
    pub gemm: String,
    pub gemm_gelu_tanh: String,
    pub gemm_gelu_erf: String,
    /// The block MLP fc1 + act the plan binds (prod on ROCm: the erf GELU, see the rules).
    pub gemm_block_fc1: String,
    pub gemm_residual: String,
    pub layernorm: String,
    pub pos_add: String,
    pub rope: String,
    /// The plan's attention: D2's tiled kernel (no per-image cap).
    pub attn: String,
    /// The tiled kernel with P rounded once to bf16 -- the class arm gated beside it.
    pub attn_p16: String,
    /// D1's correctness-first kernel (the whole score row in LDS), the gate's second reference.
    pub attn_reference: String,
    pub gemm_tile: u32,
    pub gemm_threads: u32,
    /// **D6: the LDS-staged GEMM tiles**, in order. A launch takes the FIRST tile whose grid is at
    /// least `min_wgs` workgroups (the last entry's `min_wgs` is 0, so one always matches) and binds
    /// its epilogue's symbol with the tile's `suffix`. Every tile is the shipped kernel's
    /// arithmetic bit for bit (`tools/capture/d6_gemm_bench.py` gates it per shape), so the choice
    /// is a TIME fact, held here as data. Empty: every GEMM binds the shipped symbol at
    /// `gemm_tile` / `gemm_threads` (the control).
    #[serde(default)]
    pub gemm_tiles: Vec<GemmTile>,
    pub ew_threads: u32,
    pub ln_threads: u32,
    pub attn_threads: u32,
    pub attn_q_rows: u32,
    pub attn_reference_threads: u32,
    pub attn_reference_q_rows: u32,
    /// The longest sequence `attn_reference` holds in LDS; refused past it by name.
    pub attn_reference_max_patches: u32,
    /// The largest image (patches in one frame) the planner admits: prod's processor ceiling.
    pub max_image_patches: u32,
}

/// One LDS-staged GEMM tile (D6): `{epilogue symbol}{suffix}`, `m` x `n` outputs per workgroup of
/// `threads`, taken when the launch has at least `min_wgs` workgroups at this tile.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct GemmTile {
    pub suffix: String,
    pub m: u32,
    pub n: u32,
    pub threads: u32,
    pub min_wgs: u32,
}

/// Which attention kernel a plan binds. [`AttnArm::Tiled`] is the plan; the other two exist for
/// the gate (a class arm and the D1 reference) and are never what an engine runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttnArm {
    /// `kernels.attn`: flash-tiled, P as hi + lo bf16.
    Tiled,
    /// `kernels.attn_p16`: flash-tiled, P rounded once.
    TiledP16,
    /// `kernels.attn_reference`: D1's kernel, capped at `attn_reference_max_patches`.
    Reference,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct VisionRules {
    pub schema_version: u32,
    pub name: String,
    pub model: VisionModel,
    pub kernels: VisionKernels,
}

impl VisionRules {
    pub fn prod() -> Result<Self> {
        Self::from_toml(PROD_RULES_TEXT)
    }

    /// FN55: Qwen3.8 Flash-Next's tower -- the SAME architecture and kernels as the 27B's
    /// (`vision_config` equal key for key) with ONE difference, the merger's output width
    /// [`FLASHNEXT_OUT_HIDDEN`] (Flash-Next's hidden, 2,560; the 27B's 5,120). The weights are
    /// the Flash-Next checkpoint's own (they are NOT the 27B's bytes: measured FN55), so only the
    /// arithmetic is shared; the merger's fc2 is a GEMM at N 2,560 (the same symbol at a smaller
    /// grid).
    pub fn flashnext() -> Result<Self> {
        let mut r = Self::prod()?;
        r.name = "vision-gfx1201-flashnext".to_owned();
        r.model.out_hidden = FLASHNEXT_OUT_HIDDEN;
        Ok(r)
    }

    pub fn from_toml(text: &str) -> Result<Self> {
        let r: VisionRules = toml::from_str(text).map_err(|e| Error::Parse(e.to_string()))?;
        if r.schema_version != 2 {
            return Err(Error::Rule(format!(
                "vision rules schema_version {} (this build reads 2)",
                r.schema_version
            )));
        }
        let m = &r.model;
        if m.hidden != m.heads * m.head_dim {
            return Err(Error::Rule(format!(
                "vision hidden {} != heads {} x head_dim {}",
                m.hidden, m.heads, m.head_dim
            )));
        }
        if m.rotary_dim * 2 != m.head_dim || m.rotary_dim % 4 != 0 {
            return Err(Error::Rule(format!(
                "vision rotary_dim {} must be head_dim/2 and split evenly over two axes",
                m.rotary_dim
            )));
        }
        let side = (m.num_position_embeddings as f64).sqrt() as u32;
        if side * side != m.num_position_embeddings {
            return Err(Error::Rule(format!(
                "num_position_embeddings {} is not a square grid",
                m.num_position_embeddings
            )));
        }
        if let Some(last) = r.kernels.gemm_tiles.last() {
            if last.min_wgs != 0 {
                return Err(Error::Rule(format!(
                    "vision gemm_tiles: the last tile (`{}`) has min_wgs {} -- a launch with fewer \
                     workgroups would match no tile",
                    last.suffix, last.min_wgs
                )));
            }
        }
        for t in &r.kernels.gemm_tiles {
            if t.m == 0 || t.n == 0 || t.m % 16 != 0 || t.n % 16 != 0 || t.threads % 32 != 0 {
                return Err(Error::Rule(format!(
                    "vision gemm tile `{}`: {}x{} at {} threads is not whole WMMA fragments and \
                     whole waves",
                    t.suffix, t.m, t.n, t.threads
                )));
            }
        }
        Ok(r)
    }

    /// The GEMM symbol, grid and threads a launch of `[m, n]` outputs binds for epilogue symbol
    /// `sym` (D6): the first of `gemm_tiles` with at least its `min_wgs` workgroups, or the shipped
    /// geometry when the table is empty.
    pub fn gemm_route(&self, sym: &str, m: u32, n: u32) -> (String, [u32; 3], u32) {
        let k = &self.kernels;
        for t in &k.gemm_tiles {
            let gx = cdiv(n as u64, t.n as u64);
            let gy = cdiv(m as u64, t.m as u64);
            if gx as u64 * gy as u64 >= t.min_wgs as u64 {
                return (format!("{sym}{}", t.suffix), [gx, gy, 1], t.threads);
            }
        }
        let gt = k.gemm_tile as u64;
        (
            sym.to_owned(),
            [cdiv(n as u64, gt), cdiv(m as u64, gt), 1],
            k.gemm_threads,
        )
    }

    /// Every kernel symbol the table names (D6: every GEMM epilogue at every tile too).
    pub fn symbols(&self) -> Vec<String> {
        let k = &self.kernels;
        let mut v: Vec<String> = [
            &k.cast,
            &k.gemm,
            &k.gemm_gelu_tanh,
            &k.gemm_block_fc1,
            &k.gemm_gelu_erf,
            &k.gemm_residual,
            &k.layernorm,
            &k.pos_add,
            &k.rope,
            &k.attn,
            &k.attn_p16,
            &k.attn_reference,
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        for t in &k.gemm_tiles {
            for g in [
                &k.gemm,
                &k.gemm_gelu_tanh,
                &k.gemm_block_fc1,
                &k.gemm_gelu_erf,
                &k.gemm_residual,
            ] {
                v.push(format!("{g}{}", t.suffix));
            }
        }
        v
    }

    /// The 48 of a 48x48 pos-embed table.
    pub fn grid_side(&self) -> u32 {
        (self.model.num_position_embeddings as f64).sqrt() as u32
    }

    /// Launches for any grid set: cast + patch GEMM + pos add, eight per block, three in the
    /// merger. Declared, then asserted against the list [`plan_tower`] builds (I7).
    pub fn n_launches(&self) -> u32 {
        3 + 8 * self.model.depth + 3
    }

    /// Every pack record the tower binds, in the order the plan first reads it. For this
    /// checkpoint that is all 333 `model.visual.*` records, each exactly once; the gate holds it
    /// against the pack's own manifest.
    pub fn records(&self) -> Vec<String> {
        let mut v = vec![
            "model.visual.patch_embed.proj.weight".to_owned(),
            "model.visual.patch_embed.proj.bias".to_owned(),
            "model.visual.pos_embed.weight".to_owned(),
        ];
        for i in 0..self.model.depth {
            for s in [
                "norm1.weight",
                "norm1.bias",
                "attn.qkv.weight",
                "attn.qkv.bias",
                "attn.proj.weight",
                "attn.proj.bias",
                "norm2.weight",
                "norm2.bias",
                "mlp.linear_fc1.weight",
                "mlp.linear_fc1.bias",
                "mlp.linear_fc2.weight",
                "mlp.linear_fc2.bias",
            ] {
                v.push(format!("model.visual.blocks.{i}.{s}"));
            }
        }
        for s in [
            "norm.weight",
            "norm.bias",
            "linear_fc1.weight",
            "linear_fc1.bias",
            "linear_fc2.weight",
            "linear_fc2.bias",
        ] {
            v.push(format!("model.visual.merger.{s}"));
        }
        v
    }
}

/// One image (or video) grid in patches: `image_grid_thw`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Grid {
    pub t: u32,
    pub h: u32,
    pub w: u32,
}

impl Grid {
    pub fn patches(&self) -> u32 {
        self.t * self.h * self.w
    }
}

/// A device buffer the plan names. `record` is set for a pack record (a weight); otherwise it is
/// an activation or a metadata table the runner allocates at `bytes`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BufferReq {
    pub name: String,
    pub bytes: u64,
    pub record: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TowerPlan {
    pub grids: Vec<Grid>,
    pub patches: u32,
    pub tokens: u32,
    pub launches: Vec<LaunchDescriptor>,
    pub buffers: Vec<BufferReq>,
    /// cu_seqlens over the attention's sequences (one per frame), i32.
    pub cu_seqlens: Vec<i32>,
}

fn slots(sig: &'static [SigArg], values: Vec<ArgValue>) -> Result<(Vec<Slot>, u32)> {
    if sig.len() != values.len() {
        return Err(Error::Rule(format!(
            "vision kernarg: {} values for a {}-argument signature",
            values.len(),
            sig.len()
        )));
    }
    let (pk, size) = abi::pack(sig);
    let out = sig
        .iter()
        .zip(pk)
        .zip(values)
        .map(|((s, (off, sz)), v)| Slot {
            name: s.name.to_owned(),
            offset: off,
            size: sz,
            value: v,
        })
        .collect();
    Ok((out, size))
}

fn buf(n: &str) -> ArgValue {
    ArgValue::Buffer {
        buffer: n.to_owned(),
    }
}
fn null(n: &str) -> ArgValue {
    ArgValue::NullBuffer {
        buffer: n.to_owned(),
    }
}
fn i32v(v: u32) -> ArgValue {
    ArgValue::I32 { ival: v as i64 }
}

fn launch(
    op: String,
    kernel: &str,
    grid: [u32; 3],
    block: u32,
    values: Vec<ArgValue>,
) -> Result<LaunchDescriptor> {
    let sig = abi::signature_for(kernel)
        .ok_or_else(|| Error::Rule(format!("vision: no ABI signature for `{kernel}`")))?;
    let (args, kernarg_bytes) = slots(sig, values)?;
    Ok(LaunchDescriptor {
        op,
        plugin: PLUGIN.to_owned(),
        kernel: kernel.to_owned(),
        grid,
        block: [block, 1, 1],
        shared_bytes: 0,
        kernarg_bytes,
        args,
        workspaces: Vec::new(),
        n_launches: 1,
    })
}

fn cdiv(a: u64, b: u64) -> u32 {
    a.div_ceil(b) as u32
}

/// The tower's launch list over `grids` (one or more images, concatenated as prod does), with the
/// plan's own attention ([`AttnArm::Tiled`]).
pub fn plan_tower(r: &VisionRules, grids: &[Grid]) -> Result<TowerPlan> {
    plan_tower_with(r, grids, AttnArm::Tiled)
}

/// [`plan_tower`] with the attention kernel named -- the gate's arms.
pub fn plan_tower_with(r: &VisionRules, grids: &[Grid], arm: AttnArm) -> Result<TowerPlan> {
    let m = &r.model;
    let k = &r.kernels;
    if !m.deepstack_visual_indexes.is_empty() {
        return Err(Error::Rule(format!(
            "vision: deepstack_visual_indexes {:?} -- the deepstack mergers and the decoder-side \
             injection are not built (this checkpoint's list is empty)",
            m.deepstack_visual_indexes
        )));
    }
    if grids.is_empty() {
        return Err(Error::Rule("vision: no image grid".into()));
    }
    let ms = m.spatial_merge;
    let mut cu = vec![0i32];
    for g in grids {
        if g.t == 0 || g.h == 0 || g.w == 0 || g.h % ms != 0 || g.w % ms != 0 {
            return Err(Error::Rule(format!(
                "vision: grid {g:?} is not a positive multiple of spatial_merge {ms}"
            )));
        }
        let per_frame = g.h * g.w;
        if per_frame > k.max_image_patches {
            return Err(Error::Rule(format!(
                "vision: an image of {per_frame} patches ({}x{}) exceeds max_image_patches {} \
                 (prod's processor ceiling, max_pixels / 256)",
                g.h, g.w, k.max_image_patches
            )));
        }
        if arm == AttnArm::Reference && per_frame > k.attn_reference_max_patches {
            return Err(Error::Rule(format!(
                "vision: an image of {per_frame} patches ({}x{}) exceeds `{}`'s {} -- the score \
                 rows live in LDS; the tiled `{}` serves it",
                g.h, g.w, k.attn_reference, k.attn_reference_max_patches, k.attn
            )));
        }
        for _ in 0..g.t {
            let last = *cu.last().expect("cu starts at 0");
            cu.push(last + per_frame as i32);
        }
    }
    let l: u32 = grids.iter().map(Grid::patches).sum();
    let unit = ms * ms;
    let tokens = l / unit;
    let c = m.hidden;
    let kin = m.in_channels * m.temporal_patch * m.patch * m.patch;
    let qkv_n = 3 * c;
    let mw = c * unit;
    let max_seq = grids.iter().map(|g| g.h * g.w).max().unwrap_or(0);

    let (l64, c64) = (l as u64, c as u64);
    let mut buffers = vec![
        BufferReq {
            name: "pixels".into(),
            bytes: l64 * kin as u64 * 4,
            record: false,
        },
        BufferReq {
            name: "x0".into(),
            bytes: l64 * kin as u64 * 2,
            record: false,
        },
        BufferReq {
            name: "patch".into(),
            bytes: l64 * c64 * 2,
            record: false,
        },
        BufferReq {
            name: "h".into(),
            bytes: l64 * c64 * 2,
            record: false,
        },
        BufferReq {
            name: "ln".into(),
            bytes: l64 * c64 * 2,
            record: false,
        },
        BufferReq {
            name: "qkv".into(),
            bytes: l64 * qkv_n as u64 * 2,
            record: false,
        },
        BufferReq {
            name: "ctx".into(),
            bytes: l64 * c64 * 2,
            record: false,
        },
        BufferReq {
            name: "mlp".into(),
            bytes: l64 * m.intermediate as u64 * 2,
            record: false,
        },
        BufferReq {
            name: "mfc1".into(),
            bytes: tokens as u64 * mw as u64 * 2,
            record: false,
        },
        BufferReq {
            name: "out".into(),
            bytes: tokens as u64 * m.out_hidden as u64 * 2,
            record: false,
        },
        BufferReq {
            name: "pos_idx".into(),
            bytes: 4 * l64 * 4,
            record: false,
        },
        BufferReq {
            name: "pos_wts".into(),
            bytes: 4 * l64 * 4,
            record: false,
        },
        BufferReq {
            name: "rope_cos".into(),
            bytes: l64 * (m.rotary_dim as u64) * 2,
            record: false,
        },
        BufferReq {
            name: "rope_sin".into(),
            bytes: l64 * (m.rotary_dim as u64) * 2,
            record: false,
        },
        BufferReq {
            name: "cu_seqlens".into(),
            bytes: cu.len() as u64 * 4,
            record: false,
        },
    ];
    for rec in r.records() {
        buffers.push(BufferReq {
            name: rec,
            bytes: 0,
            record: true,
        });
    }

    let gemm = |op: String,
                sym: &str,
                a: &str,
                w: &str,
                b: &str,
                res: Option<&str>,
                out: &str,
                mm: u32,
                nn: u32,
                kk: u32|
     -> Result<LaunchDescriptor> {
        if kk % 16 != 0 || nn % 16 != 0 {
            return Err(Error::Rule(format!(
                "vision: `{op}` k={kk} n={nn} -- the GEMM walks 16-wide WMMA steps"
            )));
        }
        if !k.gemm_tiles.is_empty() && kk < 32 {
            return Err(Error::Rule(format!(
                "vision: `{op}` k={kk} -- the LDS tiles stage 32-k tiles and clamp inside one"
            )));
        }
        let (sym, grid, threads) = r.gemm_route(sym, mm, nn);
        launch(
            op,
            &sym,
            grid,
            threads,
            vec![
                buf(a),
                buf(w),
                buf(b),
                res.map_or_else(|| null("res"), buf),
                buf(out),
                i32v(mm),
                i32v(nn),
                i32v(kk),
            ],
        )
    };
    let ln = |op: String, x: &str, w: &str, b: &str, y: &str| -> Result<LaunchDescriptor> {
        launch(
            op,
            &k.layernorm,
            [l, 1, 1],
            k.ln_threads,
            vec![
                buf(x),
                buf(w),
                buf(b),
                buf(y),
                i32v(l),
                i32v(c),
                ArgValue::F32 { fval: m.eps },
            ],
        )
    };
    let ew = k.ew_threads as u64;
    let mut v = Vec::with_capacity(r.n_launches() as usize);
    v.push(launch(
        "vision.cast".into(),
        &k.cast,
        [cdiv(l64 * kin as u64, ew), 1, 1],
        k.ew_threads,
        vec![
            buf("pixels"),
            buf("x0"),
            ArgValue::I64 {
                ival: (l64 * kin as u64) as i64,
            },
        ],
    )?);
    v.push(gemm(
        "vision.patch_embed".into(),
        &k.gemm,
        "x0",
        "model.visual.patch_embed.proj.weight",
        "model.visual.patch_embed.proj.bias",
        None,
        "patch",
        l,
        c,
        kin,
    )?);
    v.push(launch(
        "vision.pos_add".into(),
        &k.pos_add,
        [cdiv(l64 * c64, ew), 1, 1],
        k.ew_threads,
        vec![
            buf("patch"),
            buf("model.visual.pos_embed.weight"),
            buf("pos_idx"),
            buf("pos_wts"),
            buf("h"),
            i32v(l),
            i32v(c),
        ],
    )?);
    let scale = 1.0f32 / (m.head_dim as f32).sqrt();
    for i in 0..m.depth {
        let p = format!("model.visual.blocks.{i}.");
        let o = |s: &str| format!("vision.blocks.{i}.{s}");
        v.push(ln(
            o("norm1"),
            "h",
            &format!("{p}norm1.weight"),
            &format!("{p}norm1.bias"),
            "ln",
        )?);
        v.push(gemm(
            o("qkv"),
            &k.gemm,
            "ln",
            &format!("{p}attn.qkv.weight"),
            &format!("{p}attn.qkv.bias"),
            None,
            "qkv",
            l,
            qkv_n,
            c,
        )?);
        v.push(launch(
            o("rope"),
            &k.rope,
            [
                cdiv(l64 * 2 * m.heads as u64 * (m.head_dim / 2) as u64, ew),
                1,
                1,
            ],
            k.ew_threads,
            vec![
                buf("qkv"),
                buf("rope_cos"),
                buf("rope_sin"),
                i32v(l),
                i32v(m.heads),
                i32v(m.head_dim),
            ],
        )?);
        let (attn_sym, attn_rows, attn_thr) = match arm {
            AttnArm::Tiled => (&k.attn, k.attn_q_rows, k.attn_threads),
            AttnArm::TiledP16 => (&k.attn_p16, k.attn_q_rows, k.attn_threads),
            AttnArm::Reference => (
                &k.attn_reference,
                k.attn_reference_q_rows,
                k.attn_reference_threads,
            ),
        };
        v.push(launch(
            o("attn"),
            attn_sym,
            [
                cdiv(max_seq as u64, attn_rows as u64),
                m.heads,
                (cu.len() - 1) as u32,
            ],
            attn_thr,
            vec![
                buf("qkv"),
                buf("ctx"),
                buf("cu_seqlens"),
                i32v(m.heads),
                ArgValue::F32 { fval: scale },
            ],
        )?);
        v.push(gemm(
            o("proj"),
            &k.gemm_residual,
            "ctx",
            &format!("{p}attn.proj.weight"),
            &format!("{p}attn.proj.bias"),
            Some("h"),
            "h",
            l,
            c,
            c,
        )?);
        v.push(ln(
            o("norm2"),
            "h",
            &format!("{p}norm2.weight"),
            &format!("{p}norm2.bias"),
            "ln",
        )?);
        v.push(gemm(
            o("fc1"),
            &k.gemm_block_fc1,
            "ln",
            &format!("{p}mlp.linear_fc1.weight"),
            &format!("{p}mlp.linear_fc1.bias"),
            None,
            "mlp",
            l,
            m.intermediate,
            c,
        )?);
        v.push(gemm(
            o("fc2"),
            &k.gemm_residual,
            "mlp",
            &format!("{p}mlp.linear_fc2.weight"),
            &format!("{p}mlp.linear_fc2.bias"),
            Some("h"),
            "h",
            l,
            c,
            m.intermediate,
        )?);
    }
    v.push(ln(
        "vision.merger.norm".into(),
        "h",
        "model.visual.merger.norm.weight",
        "model.visual.merger.norm.bias",
        "ln",
    )?);
    v.push(gemm(
        "vision.merger.fc1".into(),
        &k.gemm_gelu_erf,
        "ln",
        "model.visual.merger.linear_fc1.weight",
        "model.visual.merger.linear_fc1.bias",
        None,
        "mfc1",
        tokens,
        mw,
        mw,
    )?);
    v.push(gemm(
        "vision.merger.fc2".into(),
        &k.gemm,
        "mfc1",
        "model.visual.merger.linear_fc2.weight",
        "model.visual.merger.linear_fc2.bias",
        None,
        "out",
        tokens,
        m.out_hidden,
        mw,
    )?);
    if v.len() as u32 != r.n_launches() {
        return Err(Error::Rule(format!(
            "vision: planned {} launches, declared {}",
            v.len(),
            r.n_launches()
        )));
    }
    Ok(TowerPlan {
        grids: grids.to_vec(),
        patches: l,
        tokens,
        launches: v,
        buffers,
        cu_seqlens: cu,
    })
}

// =====================================================================================
// The tower's arena (D2): one allocation, sized for the largest image, buffers aliased by phase
// =====================================================================================

/// Buffers that share bytes because they are never live together, as DATA. Each region is a list
/// of ALTERNATIVES; an alternative is a list of buffers laid back to back from the region's
/// start. The region is as large as its largest alternative. [`check_arena`] replays the launch
/// list against the layout and refuses one where a launch reads a buffer another write clobbered.
///
///   * `patch` lives from the patch GEMM to the pos add; `ln` is first written by block 0's norm1.
///   * the WORK region: `pixels`+`x0` before block 0; per block `qkv`+`ctx` (qkv -> rope -> attn
///     -> proj) and then `mlp` (fc1 -> fc2), which are never live together; `mfc1` in the merger.
///   * `h` (the residual stream), `out` (the rows D4's scatter reads) and the host tables have
///     regions of their own.
pub const ARENA_REGIONS: &[&[&[&str]]] = &[
    &[&["h"]],
    &[&["patch"], &["ln"]],
    &[&["pixels", "x0"], &["qkv", "ctx"], &["mlp"], &["mfc1"]],
    &[&["out"]],
    &[&["pos_idx"]],
    &[&["pos_wts"]],
    &[&["rope_cos"]],
    &[&["rope_sin"]],
    &[&["cu_seqlens"]],
];

/// Alignment of every buffer in the tower's arena (the text arena's `ARENA_ALIGN`).
pub const ARENA_ALIGN: u64 = 256;

/// Buffers the host writes before the first launch.
pub const HOST_INPUTS: &[&str] = &[
    "pixels",
    "pos_idx",
    "pos_wts",
    "rope_cos",
    "rope_sin",
    "cu_seqlens",
];

/// The buffer the tower's output rows land in: `[tokens, out_hidden]` bf16 at the arena offset
/// [`TowerArena::slot`] names. D4's embed scatter reads it.
pub const OUT_BUFFER: &str = "out";

/// One buffer's place in the tower's arena.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ArenaSlot {
    pub name: String,
    pub offset: u64,
    /// Capacity at `max_patches` (a smaller image uses a prefix).
    pub bytes: u64,
}

/// The tower's arena for images of up to `max_patches` patches.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TowerArena {
    pub max_patches: u32,
    pub slots: Vec<ArenaSlot>,
    /// Total bytes, aligned.
    pub bytes: u64,
    /// What the same buffers would take with no aliasing -- the saving stated, not assumed.
    pub unaliased_bytes: u64,
}

impl TowerArena {
    pub fn slot(&self, name: &str) -> Option<&ArenaSlot> {
        self.slots.iter().find(|s| s.name == name)
    }
}

fn align(x: u64) -> u64 {
    x.div_ceil(ARENA_ALIGN) * ARENA_ALIGN
}

/// The arena for one image of up to `max_patches` patches (a multiple of `spatial_merge^2`, at
/// most `max_image_patches`): every non-record buffer [`plan_tower`] names, placed by
/// [`ARENA_REGIONS`], and checked by [`check_arena`] against that plan's launch list.
pub fn tower_arena(r: &VisionRules, max_patches: u32) -> Result<TowerArena> {
    let ms = r.model.spatial_merge;
    if max_patches == 0 || max_patches % (ms * ms) != 0 {
        return Err(Error::Rule(format!(
            "vision arena: max_patches {max_patches} is not a positive multiple of {}",
            ms * ms
        )));
    }
    if max_patches > r.kernels.max_image_patches {
        return Err(Error::Rule(format!(
            "vision arena: max_patches {max_patches} exceeds max_image_patches {}",
            r.kernels.max_image_patches
        )));
    }
    // any grid of max_patches sizes every buffer: the byte counts depend on L and L/4 only
    let plan = plan_tower(
        r,
        &[Grid {
            t: 1,
            h: ms,
            w: max_patches / ms,
        }],
    )?;
    let mut sizes: std::collections::BTreeMap<&str, u64> = plan
        .buffers
        .iter()
        .filter(|b| !b.record)
        .map(|b| (b.name.as_str(), b.bytes))
        .collect();
    // D5: `cu_seqlens` is the ONE buffer sized by the FRAME count, not by L. A request's images
    // run in one tower pass (D4), one attention sequence per frame, so the slot holds the most
    // frames `max_patches` can carry -- every frame is at least one merge window
    // (`spatial_merge^2` patches) -- plus the leading 0. The single-grid plan above sized it for
    // ONE frame (8 B), and a two-image request was refused on the card ("12 B of `cu_seqlens`
    // into a 8 B slot").
    let max_frames = u64::from(max_patches / (ms * ms));
    sizes.insert("cu_seqlens", (max_frames + 1) * 4);
    let mut placed: Vec<ArenaSlot> = Vec::new();
    let mut base = 0u64;
    let mut unaliased = 0u64;
    for region in ARENA_REGIONS {
        let mut region_bytes = 0u64;
        for alt in *region {
            let mut off = 0u64;
            for name in *alt {
                let b = *sizes.get(name).ok_or_else(|| {
                    Error::Rule(format!(
                        "vision arena: region names `{name}`, the plan does not"
                    ))
                })?;
                if placed.iter().any(|s| s.name == *name) {
                    return Err(Error::Rule(format!("vision arena: `{name}` placed twice")));
                }
                placed.push(ArenaSlot {
                    name: (*name).to_owned(),
                    offset: base + off,
                    bytes: b,
                });
                off += align(b);
                unaliased += align(b);
            }
            region_bytes = region_bytes.max(off);
        }
        base += region_bytes;
    }
    for name in sizes.keys() {
        if !placed.iter().any(|s| s.name == *name) {
            return Err(Error::Rule(format!(
                "vision arena: the plan's buffer `{name}` is in no region"
            )));
        }
    }
    let arena = TowerArena {
        max_patches,
        slots: placed,
        bytes: base,
        unaliased_bytes: unaliased,
    };
    check_arena(&plan, &arena)?;
    Ok(arena)
}

/// Which of a vision kernel's buffer arguments it WRITES (the rest it reads). `rope` reads and
/// writes `qkv` in place.
fn writes(kernel: &str, arg: &str) -> bool {
    matches!(arg, "y" | "c" | "out") || (kernel.starts_with("vis_rope") && arg == "qkv")
}

/// Replay `plan`'s launches over `arena`'s byte ranges: every buffer a launch reads must hold the
/// value its last writer left (no aliased write since), and `out` must survive to the end. The
/// check a layout is built with, and the negative control's target.
pub fn check_arena(plan: &TowerPlan, arena: &TowerArena) -> Result<()> {
    use std::collections::BTreeSet;
    let range = |n: &str| arena.slot(n).map(|s| (s.offset, s.offset + s.bytes));
    let mut valid: BTreeSet<String> = HOST_INPUTS.iter().map(|s| (*s).to_owned()).collect();
    for (i, ld) in plan.launches.iter().enumerate() {
        let bufs: Vec<(&str, &str)> = ld
            .args
            .iter()
            .filter_map(|s| match &s.value {
                ArgValue::Buffer { buffer } => Some((s.name.as_str(), buffer.as_str())),
                _ => None,
            })
            .collect();
        for (arg, b) in &bufs {
            if range(b).is_none() {
                continue; // a pack record
            }
            let rw = writes(&ld.kernel, arg);
            if (!rw || ld.kernel.starts_with("vis_rope")) && !valid.contains(*b) {
                return Err(Error::Rule(format!(
                    "vision arena: launch {i} (`{}`) reads `{b}` after an aliased write \
                     clobbered it",
                    ld.op
                )));
            }
        }
        // a launch may not read one buffer while writing ANOTHER that shares its bytes (the
        // in-place rope and the residual GEMM name the same buffer twice, which is allowed)
        for (warg, wb) in &bufs {
            let Some((wl, wh)) = range(wb) else { continue };
            if !writes(&ld.kernel, warg) {
                continue;
            }
            for (_, rb) in &bufs {
                if rb == wb {
                    continue;
                }
                if let Some((rl, rh)) = range(rb) {
                    if rl < wh && wl < rh {
                        return Err(Error::Rule(format!(
                            "vision arena: launch {i} (`{}`) writes `{wb}` over the bytes of \
                             `{rb}`, which it also names (overlapping)",
                            ld.op
                        )));
                    }
                }
            }
        }
        for (arg, b) in &bufs {
            let Some((lo, hi)) = range(b) else { continue };
            if !writes(&ld.kernel, arg) {
                continue;
            }
            valid.retain(|o| o == *b || range(o).is_none_or(|(ol, oh)| oh <= lo || hi <= ol));
            valid.insert((*b).to_owned());
        }
    }
    if !valid.contains(OUT_BUFFER) {
        return Err(Error::Rule(
            "vision arena: `out` does not survive the plan".into(),
        ));
    }
    Ok(())
}

/// `torch.linspace(start, end, steps, dtype=float32)` as prod's CPU build evaluates it: the step
/// in float32, the first half counted up from `start` and the second half DOWN from `end`
/// (ATen's `RangeFactoriesKernel`), each as ONE fused multiply-add -- the AVX-512 build
/// contracts `end - step * k`, so the upper half is rounded once, not twice. Measured, not
/// assumed: 0 mismatches of 239,397 values against torch 2.11 in prod's image over
/// `end` in {31, 47, 100} and every `steps` in 2..400, where the unfused form misses at
/// 1-4 indices of most lengths (24 steps: index 17). Only `start = 0` is exercised here, where
/// the lower half is exact either way.
pub fn linspace_f32(start: f32, end: f32, steps: u32) -> Vec<f32> {
    if steps == 1 {
        return vec![start];
    }
    let step = (end - start) / (steps - 1) as f32;
    let half = steps / 2;
    (0..steps)
        .map(|i| {
            if i < half {
                step.mul_add(i as f32, start)
            } else {
                (-step).mul_add((steps - 1 - i) as f32, end)
            }
        })
        .collect()
}

/// The pos-embed gather table for `grids`: `(idx [4][L], wts [4][L])`, rows in spatial-merge
/// order and repeated per frame -- `pos_embed_interpolate_native`'s indices and float32 weights
/// (the kernel rounds the weights to bf16 itself, where the native path does).
pub fn pos_table(r: &VisionRules, grids: &[Grid]) -> (Vec<i32>, Vec<f32>) {
    let ng = r.grid_side();
    let ms = r.model.spatial_merge;
    let l: usize = grids.iter().map(|g| g.patches() as usize).sum();
    let mut idx = vec![vec![0i32; l]; 4];
    let mut wts = vec![vec![0f32; l]; 4];
    let mut row = 0usize;
    for g in grids {
        let hi = linspace_f32(0.0, (ng - 1) as f32, g.h);
        let wi = linspace_f32(0.0, (ng - 1) as f32, g.w);
        let mut frame_idx = Vec::with_capacity((g.h * g.w) as usize);
        let mut frame_wts = Vec::with_capacity((g.h * g.w) as usize);
        for bh in 0..g.h / ms {
            for bw in 0..g.w / ms {
                for i in 0..ms {
                    for j in 0..ms {
                        let (y, x) = ((bh * ms + i) as usize, (bw * ms + j) as usize);
                        let (hf, wf) = (hi[y] as i32, wi[x] as i32);
                        let hc = (hf + 1).min(ng as i32 - 1);
                        let wc = (wf + 1).min(ng as i32 - 1);
                        let dh = hi[y] - hf as f32;
                        let dw = wi[x] - wf as f32;
                        let w11 = dh * dw;
                        let w10 = dh - w11;
                        let w01 = dw - w11;
                        let w00 = 1.0 - dh - w01;
                        let n = ng as i32;
                        frame_idx.push([hf * n + wf, hf * n + wc, hc * n + wf, hc * n + wc]);
                        frame_wts.push([w00, w01, w10, w11]);
                    }
                }
            }
        }
        for _ in 0..g.t {
            for (a, b) in frame_idx.iter().zip(&frame_wts) {
                for j in 0..4 {
                    idx[j][row] = a[j];
                    wts[j][row] = b[j];
                }
                row += 1;
            }
        }
    }
    (idx.concat(), wts.concat())
}

/// fp32 -> bf16 bits, round to nearest even (no NaNs reach here).
pub fn bf16_bits(x: f32) -> u16 {
    let u = x.to_bits();
    let r = ((u >> 16) & 1) + 0x7FFF;
    (u.wrapping_add(r) >> 16) as u16
}

/// The 2-D rope rows for `grids`: `(cos, sin)`, each `[L][rotary_dim]` bf16 bits, the first
/// `rotary_dim/2` columns the h-axis frequencies and the rest the w-axis ones -- the reference server's
/// `cos_sin_cache` (float32 `inv_freq = 1 / base^(2i/rotary_dim)`, `t * inv_freq`, cos/sin,
/// then `.to(bf16)`) gathered by `rot_pos_ids` in spatial-merge order.
pub fn rope_table(r: &VisionRules, grids: &[Grid]) -> (Vec<u16>, Vec<u16>) {
    let m = &r.model;
    let nf = (m.rotary_dim / 2) as usize;
    let inv: Vec<f32> = (0..nf)
        .map(|i| {
            let e = (2 * i) as f32 / m.rotary_dim as f32;
            1.0f32 / m.rope_theta.powf(e)
        })
        .collect();
    let cs = |p: u32| -> (Vec<u16>, Vec<u16>) {
        inv.iter()
            .map(|&f| {
                let x = p as f32 * f;
                (
                    bf16_bits((x as f64).cos() as f32),
                    bf16_bits((x as f64).sin() as f32),
                )
            })
            .unzip()
    };
    let ms = m.spatial_merge;
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for g in grids {
        for _ in 0..g.t {
            for bh in 0..g.h / ms {
                for bw in 0..g.w / ms {
                    for i in 0..ms {
                        for j in 0..ms {
                            let (ch, sh) = cs(bh * ms + i);
                            let (cw, sw) = cs(bw * ms + j);
                            cos.extend(ch);
                            cos.extend(cw);
                            sin.extend(sh);
                            sin.extend(sw);
                        }
                    }
                }
            }
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(t: u32, h: u32, w: u32) -> Grid {
        Grid { t, h, w }
    }

    #[test]
    fn prod_rules_parse_and_name_signed_symbols() {
        let r = VisionRules::prod().expect("rules");
        assert_eq!(r.model.depth, 27);
        assert!(r.model.deepstack_visual_indexes.is_empty());
        for s in r.symbols() {
            assert!(abi::signature_for(&s).is_some(), "{s} has no signature");
        }
    }

    /// D6: the LDS tile a launch binds is the first with enough workgroups (one tile today); an empty table is the
    /// shipped kernel (the control); a table whose last tile can refuse a launch is refused.
    #[test]
    fn gemm_routes_follow_the_tile_table() {
        let r = VisionRules::prod().expect("rules");
        let names = r.symbols();
        assert!(names
            .iter()
            .any(|s| s == "vis_gemm_bf16_bias_residual_l128x128"));
        // 768 patches: every GEMM at the one measured tile, even proj at 54 workgroups
        let (s, gr, t) = r.gemm_route("vis_gemm_bf16_bias", 768, 3456);
        assert_eq!(
            (s.as_str(), gr, t),
            ("vis_gemm_bf16_bias_l128x128", [27, 6, 1], 256)
        );
        let (s, gr, t) = r.gemm_route("vis_gemm_bf16_bias_residual", 768, 1152);
        assert_eq!(
            (s.as_str(), gr, t),
            ("vis_gemm_bf16_bias_residual_l128x128", [9, 6, 1], 256)
        );
        // 8,064 patches: everything at 128x128
        let (s, _, _) = r.gemm_route("vis_gemm_bf16_bias_residual", 8064, 1152);
        assert_eq!(s, "vis_gemm_bf16_bias_residual_l128x128");
        // the control: an empty table binds the shipped symbol at the shipped geometry
        let mut c = r.clone();
        c.kernels.gemm_tiles.clear();
        let (s, gr, t) = c.gemm_route("vis_gemm_bf16_bias", 768, 3456);
        assert_eq!(
            (s.as_str(), gr, t),
            ("vis_gemm_bf16_bias", [54, 12, 1], 128)
        );
        let plan = plan_tower(&c, &[g(1, 24, 32)]).unwrap();
        assert!(plan
            .launches
            .iter()
            .filter(|l| l.kernel.starts_with("vis_gemm"))
            .all(|l| !l.kernel.contains("_l")));
        // negative: a last tile with a floor could match nothing
        let text = PROD_RULES_TEXT.replace("min_wgs = 0", "min_wgs = 1");
        let e = VisionRules::from_toml(&text).unwrap_err().to_string();
        assert!(e.contains("would match no tile"), "{e}");
    }

    /// FN55: Flash-Next's tower is the 27B's plan launch for launch -- same symbols, same
    /// geometry -- except the merger's fc2 (N 2,560: a narrower grid of the same GEMM) and the
    /// `out` rows it writes; the control (the 27B's rules) differs exactly there.
    #[test]
    fn fn55_the_flashnext_tower_differs_from_the_27b_only_at_the_merger_out() {
        let a = VisionRules::prod().unwrap();
        let b = VisionRules::flashnext().unwrap();
        assert_eq!(b.model.out_hidden, 2560);
        assert_eq!(a.records(), b.records());
        assert_eq!(a.symbols(), b.symbols());
        assert_eq!(a.n_launches(), b.n_launches());
        let grids = [g(1, 24, 32)];
        let (pa, pb) = (
            plan_tower(&a, &grids).unwrap(),
            plan_tower(&b, &grids).unwrap(),
        );
        assert_eq!(pa.launches.len(), pb.launches.len());
        let differ: Vec<usize> = (0..pa.launches.len())
            .filter(|&i| pa.launches[i] != pb.launches[i])
            .collect();
        let last = pa.launches.len() - 1;
        assert_eq!(differ, vec![last], "only the merger fc2 moves");
        assert_eq!(pa.launches[last].kernel, pb.launches[last].kernel);
        assert_eq!(pb.launches[last].grid[0] * 2, pa.launches[last].grid[0]);
        let (aa, ab) = (
            tower_arena(&a, 16_384).unwrap(),
            tower_arena(&b, 16_384).unwrap(),
        );
        check_arena(&pb, &ab).unwrap();
        assert!(ab.bytes <= aa.bytes, "{} > {}", ab.bytes, aa.bytes);
        let out = |t: &TowerArena| t.slot(OUT_BUFFER).unwrap().bytes;
        assert!(
            out(&ab) * 2 <= out(&aa) + 4096,
            "{} vs {}",
            out(&ab),
            out(&aa)
        );
    }

    #[test]
    fn the_tower_binds_all_333_records_once() {
        let r = VisionRules::prod().unwrap();
        let recs = r.records();
        assert_eq!(recs.len(), 333);
        let set: std::collections::BTreeSet<_> = recs.iter().collect();
        assert_eq!(set.len(), 333);
        let plan = plan_tower(&r, &[g(1, 24, 32)]).unwrap();
        // every record is read by at least one launch; control: one that is not a record
        let read: std::collections::BTreeSet<String> = plan
            .launches
            .iter()
            .flat_map(|l| l.args.iter())
            .filter_map(|s| match &s.value {
                ArgValue::Buffer { buffer } => Some(buffer.clone()),
                _ => None,
            })
            .collect();
        for rcd in &recs {
            assert!(read.contains(rcd), "{rcd} is never read");
        }
        assert!(!read.contains("model.visual.deepstack_merger_list.0.norm.weight"));
    }

    #[test]
    fn launch_count_is_declared_and_shapes_follow_the_grid() {
        let r = VisionRules::prod().unwrap();
        let plan = plan_tower(&r, &[g(1, 24, 32)]).unwrap();
        assert_eq!(plan.launches.len(), 222);
        assert_eq!(plan.patches, 768);
        assert_eq!(plan.tokens, 192);
        assert_eq!(plan.cu_seqlens, vec![0, 768]);
        let fc1 = plan
            .launches
            .iter()
            .find(|l| l.op == "vision.merger.fc1")
            .unwrap();
        assert_eq!(fc1.kernel, "vis_gemm_bf16_bias_gelu_erf_l128x128");
        // D6: 192 rows x 4608 at the 128x128 LDS tile.
        assert_eq!(fc1.grid, [36, 2, 1]);
        assert_eq!(fc1.block, [256, 1, 1]);
        let blk_fc1 = plan
            .launches
            .iter()
            .find(|l| l.op == "vision.blocks.0.fc1")
            .unwrap();
        assert_eq!(blk_fc1.kernel, "vis_gemm_bf16_bias_gelu_erf_l128x128");
        // two images, two attention sequences
        let two = plan_tower(&r, &[g(1, 24, 32), g(1, 16, 16)]).unwrap();
        assert_eq!(two.cu_seqlens, vec![0, 768, 1024]);
        assert_eq!(two.tokens, 256);
    }

    #[test]
    fn refusals_are_by_name() {
        let r = VisionRules::prod().unwrap();
        let e = plan_tower(&r, &[g(1, 23, 32)]).unwrap_err().to_string();
        assert!(e.contains("spatial_merge"), "{e}");
        // D2: the tiled attention serves 64x64 (4,096 patches); D1's kernel is refused there
        plan_tower(&r, &[g(1, 64, 64)]).expect("the tiled attention has no LDS cap");
        let e = plan_tower_with(&r, &[g(1, 64, 64)], AttnArm::Reference)
            .unwrap_err()
            .to_string();
        assert!(e.contains("vis_attn_bf16_hd72"), "{e}");
        // prod's ceiling: 256 x 256 = 65,536 patches plans, one more row of merges does not
        plan_tower(&r, &[g(1, 256, 256)]).expect("prod's largest image");
        let e = plan_tower(&r, &[g(1, 258, 256)]).unwrap_err().to_string();
        assert!(e.contains("max_image_patches"), "{e}");
        let mut ds = r.clone();
        ds.model.deepstack_visual_indexes = vec![8, 16, 24];
        let e = plan_tower(&ds, &[g(1, 24, 32)]).unwrap_err().to_string();
        assert!(e.contains("deepstack"), "{e}");
    }

    #[test]
    fn the_arena_aliases_by_phase_and_the_replay_holds() {
        let r = VisionRules::prod().unwrap();
        let a = tower_arena(&r, 65_536).expect("prod's largest image");
        // per-buffer capacity at 65,536 patches: h/ln/ctx/patch 151 MB, qkv 453 MB, mlp 564 MB,
        // pixels 403 MB + x0 201 MB, out 168 MB (16,384 x 5120 bf16)
        assert_eq!(a.slot("out").unwrap().bytes, 16_384 * 5120 * 2);
        assert_eq!(a.slot("qkv").unwrap().bytes, 65_536 * 3456 * 2);
        assert!(
            a.bytes < a.unaliased_bytes * 6 / 10,
            "{} vs {}",
            a.bytes,
            a.unaliased_bytes
        );
        // the work region is its largest alternative: pixels + x0 == qkv + ctx here
        let w = a.slot("pixels").unwrap();
        assert_eq!(a.slot("qkv").unwrap().offset, w.offset);
        assert_eq!(a.slot("mlp").unwrap().offset, w.offset);
        // every smaller image fits in the same slots
        let small = plan_tower(&r, &[g(1, 24, 32)]).unwrap();
        for b in small.buffers.iter().filter(|b| !b.record) {
            assert!(b.bytes <= a.slot(&b.name).unwrap().bytes, "{}", b.name);
        }
        check_arena(&small, &a).expect("the replay at 768 patches");
    }

    /// D5: a request's images share ONE tower pass, so the frame-count buffer must hold every
    /// frame the arena's patches can carry. The card refused a two-image request with the slot
    /// sized for one; the control is that slot (8 B) against a three-image plan.
    #[test]
    fn the_arena_holds_every_frame_its_patches_can_carry() {
        let r = VisionRules::prod().unwrap();
        let a = tower_arena(&r, 16_384).expect("arena");
        let cu = a.slot("cu_seqlens").unwrap().bytes;
        assert_eq!(cu, (16_384 / 4 + 1) * 4);
        let three = plan_tower(&r, &[g(1, 8, 8), g(1, 2, 2), g(1, 30, 6)]).unwrap();
        for b in three.buffers.iter().filter(|b| !b.record) {
            assert!(b.bytes <= a.slot(&b.name).unwrap().bytes, "{}", b.name);
        }
        let need = three
            .buffers
            .iter()
            .find(|b| b.name == "cu_seqlens")
            .unwrap()
            .bytes;
        assert_eq!(need, 16);
        assert!(need > 8, "control: the pre-D5 one-frame slot refuses it");
        // the worst case: every frame one merge window
        let tiny: Vec<Grid> = (0..16_384 / 4).map(|_| g(1, 2, 2)).collect();
        let worst = plan_tower(&r, &tiny).unwrap();
        let w = worst
            .buffers
            .iter()
            .find(|b| b.name == "cu_seqlens")
            .unwrap();
        assert_eq!(w.bytes, cu);
        check_arena(&three, &a).expect("the replay over three frames");
    }

    #[test]
    fn a_layout_that_clobbers_a_live_buffer_is_refused() {
        // NEGATIVE CONTROL: put `mlp` over the residual stream -- fc1 then writes the bytes `h`
        // holds and fc2's residual read (and the next norm1) would read garbage.
        let r = VisionRules::prod().unwrap();
        let mut a = tower_arena(&r, 4096).unwrap();
        let h = a.slot("h").unwrap().offset;
        a.slots.iter_mut().find(|s| s.name == "mlp").unwrap().offset = h;
        let plan = plan_tower(&r, &[g(1, 64, 64)]).unwrap();
        let e = check_arena(&plan, &a).unwrap_err().to_string();
        assert!(e.contains("`mlp`"), "{e}");
        // and `ln` over `qkv`: the qkv GEMM would write the rows it is still reading
        let mut b = tower_arena(&r, 4096).unwrap();
        let q = b.slot("qkv").unwrap().offset;
        b.slots.iter_mut().find(|s| s.name == "ln").unwrap().offset = q;
        let e = check_arena(&plan, &b).unwrap_err().to_string();
        assert!(e.contains("overlapping"), "{e}");
    }

    #[test]
    fn arena_refusals_are_by_name() {
        let r = VisionRules::prod().unwrap();
        assert!(tower_arena(&r, 65_537)
            .unwrap_err()
            .to_string()
            .contains("multiple"));
        assert!(tower_arena(&r, 65_540)
            .unwrap_err()
            .to_string()
            .contains("max_image_patches"));
    }

    #[test]
    fn linspace_counts_the_upper_half_down_from_end() {
        let v = linspace_f32(0.0, 47.0, 24);
        assert_eq!(v[0], 0.0);
        assert_eq!(v[23], 47.0);
        let step = 47.0f32 / 23.0;
        // index 17, where torch's fused `end - step * k` and the two-rounding form part: prod's
        // table has 34.739132 (the fused value); the unfused 47 - step*6 is 34.73913.
        assert_eq!(v[17], 34.739_132_f32);
        assert_ne!(v[17], 47.0 - std::hint::black_box(step * 6.0));
    }

    #[test]
    fn pos_table_weights_sum_to_one_and_corners_are_exact() {
        let r = VisionRules::prod().unwrap();
        let (idx, wts) = pos_table(&r, &[g(1, 24, 32)]);
        let l = 768;
        assert_eq!(idx.len(), 4 * l);
        // row 0 is grid (0,0): all weight on table entry 0
        assert_eq!(idx[0], 0);
        assert_eq!(wts[0], 1.0);
        for i in 0..l {
            let s: f32 = (0..4).map(|j| wts[j * l + i]).sum();
            assert!((s - 1.0).abs() < 1e-6, "row {i} sums to {s}");
        }
    }

    #[test]
    fn rope_rows_start_at_position_zero() {
        let r = VisionRules::prod().unwrap();
        let (cos, sin) = rope_table(&r, &[g(1, 24, 32)]);
        assert_eq!(cos.len(), 768 * 36);
        assert!(cos[..36].iter().all(|&b| b == bf16_bits(1.0)));
        assert!(sin[..36].iter().all(|&b| b == 0));
        // row 1 is (h=0, w=1): h half unchanged, w half rotated
        assert_eq!(cos[36], bf16_bits(1.0));
        assert_ne!(cos[36 + 18], bf16_bits(1.0));
    }
}
