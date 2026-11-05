//! The WHOLE k=7 proposal, replayed against prod's own captured one.
//!
//! `the internal reference harness` boots prod's configuration at TP=1 on one card and dumps
//! ONE decode step of `DFlash2Speculator.propose`: the drafter's input hidden states, the index
//! tables `prepare_dflash_inputs` wrote, the drafter's KV cache, the int2 head's candidates, the
//! candidate selector's score tensor and the k=7 draft block. This module is what consumes it.
//!
//! **What it claims, and what it does not.** A proposal is 107 launches. Until ws-f26 the plan
//! covered 80 of them and the other eleven were ABSENT — three ATen selections, the fused KV
//! projection, its layout copy and six activation quants — of which five were SUPPLIED by the
//! capture's own tensors and six had no kernel at all. The comparison below is the one that shape
//! forced: STAGED, every stage but the last seeded from prod's own tensors at its input.
//!
//! All eleven are covered now (91 of 107; the other 16 are another plugin's), so the staged table
//! is no longer the only thing this file can claim. `--sites` byte-compares each of the eleven
//! against prod's own captured intermediate with a control arm, and `--no-seed` runs the whole
//! chain from prod's INPUTS — the target's concatenated hidden states, the drafter's query ids and
//! positions, and the KV cache as it stood BEFORE the context store — to the k=7 block, with
//! nothing seeded in between. The staged comparison stays because it is what localises a
//! difference: a chain says which BLOCK is wrong and a stage says which OP is.
//!
//! | stage | seeded from | compared against |
//! |---|---|---|
//! | the exact rerank | prod's sampled hidden rows + prod's candidate ids | prod's `unary_logits` |
//! | the selector | prod's candidate ids, unary logits and hidden rows | prod's score tensor |
//! | the walk | **Saber's own** scores | prod's k=7 draft block |
//!
//! Seeding each stage rather than chaining the whole pass is the honest construction while two
//! activation quants are undeclared ([`UNDECLARED_QUANTS`]): a chain through a launch the plan does
//! not declare would be a measurement of this file's guess at it. The WALK is the one stage
//! deliberately chained — it reads the scores Saber's own selector produced, so a draft block that
//! matches is a statement about both kernels and not only about the walk.
//!
//! Beside the comparison there is a TIME, behind `--pack-launches`: every covered launch of the
//! proposal, in forward order, at the shapes this captured step actually ran at, with its weights
//! bound by record name out of the drafter pack and the conv / attention / KV / selector launches
//! bound to the capture's own index tables and KV cache. It is measured against prod's own
//! **5030 µs median** over 24 decode steps at TP=1 on card 0, cudagraphs as prod configures them.
//!
//! It FAULTED on 2026-09-21 with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` and a 6.5 GiB
//! coredump, and round 21 called that "a scratch buffer this file sizes is smaller than what some
//! kernel writes". It was not a write. `_rope_context_k` calls
//! `ops.rotary_embedding(positions.repeat(num_layers), ...)`, so the context rope's grid is
//! `L * num_ctx` rows and it READS a position per row; this file uploaded the capture's `num_ctx`
//! positions unrepeated, and rows past the first `num_ctx` took whatever `hipMalloc` had left
//! behind. A garbage `int64` position times `rot_dim` indexes far outside a `[4096, 128]` rope
//! cache. The repeat is the reference's own and is restored here; the first pass now syncs after
//! every launch so the next one of these names an op instead of a set.
//!
//! **The KV cache is de-interleaved here, not in the capture.** Prod's `TritonAttentionBackend`
//! allocates ONE tensor per layer, `[blocks, kv_heads, block_size, 2 * head_dim]`, and hands the
//! append kernel two strided views of it; `dflash_attn` and `dflash_kv` take two SEPARATE
//! contiguous caches. Same arithmetic, different stride ([`saber_model::dflash_graph::DraftKvCache`]
//! says so in the plan). [`deinterleave`] is that transform, and the block table is renumbered with
//! it: the capture holds only the blocks the request's table names, so block `b` becomes its index
//! in that list and every slot follows.

use std::collections::BTreeMap;
use std::path::Path;

use saber_exec::{ArgBindings, LaunchCounters, LaunchCtx, Launcher, LoadCounters, PluginSet};
use saber_hip::{DeviceAddr, DevicePtr, HipThread, Stream, StreamFlags};
use saber_kernels::dflash::{DflashOp, DflashRules, DflashShape};
use saber_kernels::head::{self, HeadOp, HeadRules, HeadShape};
use saber_kernels::rules::{ArgValue, Shape};
use saber_mem::dio::OpenMode;
use saber_mem::ledger::{Ledger, Pool};
use saber_mem::upload::{HipSink, PinnedRing, Upload, UploadConfig};
use saber_model::dflash_graph::{Coverage, DflashConfig, DraftGraph, DraftRecord, DraftRope};

use crate::draft_step::DraftStepArgs;
use crate::npy;

/// The launches a proposal makes that read an ACTIVATION the plan declares no producer for.
///
/// Both are quantisations. `aux_hidden_projection` and `attn.o_proj` are `gemm_a8w8_blockscale`, so
/// each needs an `e4m3` activation with per-128-group `f32` scales — and unlike every other
/// blockscale GEMM in the drafter, neither is preceded by an op that emits one:
/// `attention_conv.prepare` feeds `attn.qkv`, `mlp_conv.prepare` feeds `mlp.gate_up` and
/// `mlp.silu_quant` feeds `mlp.down`, but the target's concatenated hidden states and the
/// attention's output arrive in `bf16`. In prod the quant is inside the layer's own quant method,
/// which is why a launch census that counts GEMMs does not see it and why the plan does not declare
/// it. Found by trying to CHAIN this replay, which is the only way a missing producer announces
/// itself — a staged replay seeds past it and never notices.
pub const UNDECLARED_QUANTS: &[(&str, &str)] = &[
    (
        "aux_hidden_projection",
        "the five tapped target layers arrive bf16; the blockscale GEMM needs e4m3 + per-128-group \
         f32 scales and no declared op produces them",
    ),
    (
        "attn.o_proj",
        "the attention writes bf16 and o_proj is a blockscale GEMM; the conv `prepare` that feeds \
         every other one is on the OTHER side of the attention",
    ),
];

/// One op's output against the compiled REGION's own bytes (F32).
///
/// Returns the differing ELEMENT count and the widest distance between two raw patterns, which
/// for two bf16s of the same sign and exponent is the number of ULPs between them. Elements and
/// not bytes, because the question these rows answer is how far a value moved and a byte count
/// cannot tell one ULP from a different exponent.
fn region_diff(got: &[u8], want: &[u8], width: usize) -> (usize, i64) {
    let n = got.len().min(want.len()) / width;
    let val = |b: &[u8], i: usize| -> i64 {
        match width {
            1 => i64::from(b[i]),
            2 => i64::from(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])),
            _ => i64::from(u32::from_le_bytes([
                b[4 * i],
                b[4 * i + 1],
                b[4 * i + 2],
                b[4 * i + 3],
            ])),
        }
    };
    let (mut d, mut ulp) = (0usize, 0i64);
    for i in 0..n {
        let (a, b) = (val(got, i), val(want, i));
        if a != b {
            d += 1;
            ulp = ulp.max((a - b).abs());
        }
    }
    (d, ulp)
}

/// One captured proposal: the manifest and the flat `.npy` beside it.
struct Capture {
    dir: std::path::PathBuf,
    tag: String,
    man: serde_json::Value,
}

impl Capture {
    fn open(dir: &Path, tag: &str) -> anyhow::Result<Self> {
        let man: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join(format!("{tag}.json")))?)?;
        Ok(Capture {
            dir: dir.to_path_buf(),
            tag: tag.to_string(),
            man,
        })
    }

    fn bytes(&self, key: &str) -> anyhow::Result<Vec<u8>> {
        npy::read_u8(&self.dir.join(format!("{}.{key}.npy", self.tag)))
    }

    fn shape(&self, key: &str) -> anyhow::Result<Vec<usize>> {
        let v = self
            .man
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("the capture has no `{key}`"))?;
        Ok(v["shape"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("`{key}` has no shape"))?
            .iter()
            .filter_map(serde_json::Value::as_u64)
            .map(|x| x as usize)
            .collect())
    }

    fn dtype(&self, key: &str) -> String {
        self.man
            .get(key)
            .and_then(|v| v["dtype"].as_str())
            .unwrap_or("?")
            .to_string()
    }

    /// The COMPILED REGION's own GEMM calls, named by their own shapes (F32, r32 and later).
    ///
    /// The drafter's layer body is one inductor region and `@support_torch_compile` leaves no
    /// module seam inside it, so through F31 its only anchors were its two ends and every number
    /// between them was a chained one. The region's GEMMs are not fused into Triton, though:
    /// inductor emits them as `torch.ops.<ns>.<op>.default(...)` and looks the packet up at every
    /// call, so the probe swaps the packet for the length of one forward and keeps every operand
    /// (`dflash_proposal_probe.py::_region_call`). This is the reader for those taps.
    ///
    /// TWO RULES, both of them measurements rather than assumptions (`evidence-f32/`):
    ///   * a `rocm_unquantized_gemm` arrives TWICE at every kernel projection, because prod's
    ///     `ref_gemm` wrapper takes the call, finds no `gemm_nt` kernel for M <= 64 bf16 and
    ///     calls the op again. Both dispatches carry the same operands and return the same bytes
    ///     -- compared on the host, byte for byte -- so consecutive records with the same op and
    ///     the same operand shapes are ONE call;
    ///   * the four `preshuffle_gemm` sites are distinct at this checkpoint's widths, and the qkv
    ///     calls count the layers: everything after a layer's qkv and before the next one belongs
    ///     to that layer's body.
    fn region_map(&self) -> std::collections::HashMap<(u32, String), usize> {
        let mut out = std::collections::HashMap::new();
        let Some(calls) = self.man.get("region.calls").and_then(|v| v.as_array()) else {
            return out;
        };
        let sig = |rec: &serde_json::Value| -> String {
            let op = rec["op"].as_str().unwrap_or("?").to_string();
            let args = rec["in"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|e| {
                            e.get("shape")
                                .map(|s| s.to_string())
                                .or_else(|| e.get("value").map(|v| v.to_string()))
                                .unwrap_or_default()
                        })
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .unwrap_or_default();
            format!("{op}|{args}")
        };
        let (mut qkv_seen, mut last, mut prev) = (0u32, String::new(), String::new());
        for rec in calls {
            let this = sig(rec);
            if this == prev {
                continue;
            }
            prev = this;
            let op = rec["op"].as_str().unwrap_or("");
            let scal: Vec<i64> = rec["in"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|e| e.get("value"))
                        .filter_map(|v| v.as_str().and_then(|s| s.parse::<i64>().ok()))
                        .collect()
                })
                .unwrap_or_default();
            let owidth = rec["out"]["shape"]
                .as_array()
                .and_then(|a| a.last())
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(0);
            let name = if op.ends_with("preshuffle_gemm") && scal.len() >= 2 {
                match (scal[scal.len() - 2], scal[scal.len() - 1]) {
                    (5120, 4096) => "o_proj",
                    (34816, 5120) => "gate_up",
                    (5120, 17408) => "down",
                    (6144, 5120) => "qkv",
                    _ => "",
                }
            } else if op.ends_with("rocm_unquantized_gemm") && owidth == 1280 {
                if last == "o_proj" {
                    "kp_mlp"
                } else {
                    "kp_attn"
                }
            } else {
                ""
            };
            if name.is_empty() {
                continue;
            }
            let layer = if name == "kp_attn" || name == "qkv" {
                qkv_seen
            } else {
                qkv_seen.saturating_sub(1)
            };
            if name == "qkv" {
                qkv_seen += 1;
            }
            last = name.to_string();
            let idx = rec["i"].as_u64().unwrap_or(0) as usize;
            out.entry((layer, name.to_string())).or_insert(idx);
        }
        out
    }

    fn int(&self, key: &str) -> anyhow::Result<u32> {
        let v = self
            .man
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| anyhow::anyhow!("the capture has no scalar `{key}`"))?;
        Ok(u32::try_from(v).unwrap_or(u32::MAX))
    }

    fn i64_table(&self, key: &str) -> anyhow::Result<Vec<i64>> {
        Ok(self
            .bytes(key)?
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap_or_default()))
            .collect())
    }

    fn i32_table(&self, key: &str) -> anyhow::Result<Vec<i32>> {
        Ok(self
            .bytes(key)?
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap_or_default()))
            .collect())
    }
}

/// Prod's one interleaved cache per layer, `[blocks, kv_heads, block_size, 2 * head_dim]`, split
/// into the two contiguous `[blocks, block_size, kv_heads, head_dim]` caches Saber's kernels take.
///
/// A pure permutation of bytes, so `elem` (the element width) is all it needs to know about the
/// dtype: it is correct for a `bf16` cache and an `e4m3` one alike.
fn deinterleave(
    src: &[u8],
    blocks: usize,
    kv_heads: usize,
    block_size: usize,
    head_dim: usize,
    elem: usize,
) -> (Vec<u8>, Vec<u8>) {
    let row = head_dim * elem;
    let mut k = vec![0u8; blocks * block_size * kv_heads * row];
    let mut v = vec![0u8; k.len()];
    for b in 0..blocks {
        for h in 0..kv_heads {
            for t in 0..block_size {
                let s = ((b * kv_heads + h) * block_size + t) * 2 * row;
                let d = ((b * block_size + t) * kv_heads + h) * row;
                if s + 2 * row <= src.len() && d + row <= k.len() {
                    k[d..d + row].copy_from_slice(&src[s..s + row]);
                    v[d..d + row].copy_from_slice(&src[s + row..s + 2 * row]);
                }
            }
        }
    }
    (k, v)
}

/// The element width of a captured tensor, from its dtype NAME.
fn width_of(dtype: &str) -> usize {
    match dtype {
        "uint8" | "int8" | "float8_e4m3fn" | "float8_e4m3fnuz" | "bool" => 1,
        "bfloat16" | "float16" | "int16" => 2,
        "int64" | "float64" => 8,
        // `float32`, `int32` and anything else four bytes wide.
        _ => 4,
    }
}

/// The element width of prod's KV CACHE, which its dtype does not state.
///
/// The reference server allocates an fp8 cache as `uint8` and `.view(fp8_dtype)`s it at the point of use
/// (`TritonAttentionImpl.forward` does exactly that), so the capture's manifest says `uint8` for a
/// cache whose elements are one byte AND for one whose elements are two. The last dimension is
/// what says which: it is `2 * head_dim` ELEMENTS of the cache's own dtype, so the width is
/// `itemsize * shape[3] / (2 * head_dim)`. Reading `float8` out of the dtype string — which this
/// file did until round 23 — takes an fp8 cache for a bf16 one, de-interleaves at twice the row
/// width, and silently copies half the bytes.
fn kv_elem(cap: &Capture, key: &str, head_dim: usize) -> anyhow::Result<usize> {
    let shape = cap.shape(key)?;
    let last = shape
        .last()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("`{key}` has no last dimension"))?;
    let bytes = width_of(&cap.dtype(key)) * last;
    let per = 2 * head_dim.max(1);
    if bytes % per != 0 {
        anyhow::bail!(
            "`{key}`'s last dimension is {bytes} bytes, which is not a whole number of \
             2 x head_dim = {per} elements"
        );
    }
    Ok(bytes / per)
}

/// `f32` -> `bf16` with round-to-nearest-even, and back. The rerank writes `f32`; prod stores the
/// same number as `bf16` (`y.scatter_`) before its top-k reads it as `f32`, so a comparison has to
/// round where prod rounds and nowhere else.
fn f32_to_bf16_rne(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let rounding = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding) >> 16) as u16
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// Every record a weight-reading op binds, by op name and kernarg.
fn weight_map<'a>(
    want: &'a [DraftRecord],
    addr: &BTreeMap<String, DeviceAddr>,
) -> BTreeMap<&'a str, Vec<(&'a str, DeviceAddr)>> {
    let mut m: BTreeMap<&str, Vec<(&str, DeviceAddr)>> = BTreeMap::new();
    for r in want {
        let (Some(op), Some(a)) = (r.op.as_deref(), addr.get(&r.name)) else {
            continue;
        };
        m.entry(op).or_default().push((r.kernarg(), *a));
    }
    m
}

/// A device allocation plus the length that was put in it.
struct Buf {
    p: DevicePtr,
    len: usize,
}

impl Buf {
    fn addr(&self) -> DeviceAddr {
        self.p.addr()
    }
}

/// Everything the replay allocates, kept alive for the whole prompt.
struct Pool2 {
    bufs: Vec<Buf>,
}

impl Pool2 {
    fn new() -> Self {
        Pool2 { bufs: Vec::new() }
    }

    fn upload(
        &mut self,
        ctx: &LaunchCtx<'_>,
        stream: &Stream,
        bytes: &[u8],
    ) -> anyhow::Result<DeviceAddr> {
        let p = ctx.hip.malloc(bytes.len().max(4096) as u64)?;
        ctx.hip.memcpy_h2d_blocking(&p, bytes, stream)?;
        let a = p.addr();
        self.bufs.push(Buf {
            p,
            len: bytes.len(),
        });
        Ok(a)
    }

    fn zeroed(
        &mut self,
        ctx: &LaunchCtx<'_>,
        stream: &Stream,
        len: usize,
    ) -> anyhow::Result<DeviceAddr> {
        let p = ctx.hip.malloc(len.max(4096) as u64)?;
        ctx.hip.memset(&p, 0, stream)?;
        let a = p.addr();
        self.bufs.push(Buf { p, len });
        Ok(a)
    }

    fn read(
        &self,
        ctx: &LaunchCtx<'_>,
        stream: &Stream,
        a: DeviceAddr,
        len: usize,
    ) -> anyhow::Result<Vec<u8>> {
        let b = self
            .bufs
            .iter()
            .find(|b| b.addr() == a)
            .ok_or_else(|| anyhow::anyhow!("no allocation at that address"))?;
        let n = len.min(b.len);
        let mut out = vec![0u8; n];
        ctx.hip.memcpy_d2h_blocking(&mut out, &b.p, stream)?;
        Ok(out)
    }
}

/// The blocks this capture carries, and the map from prod's block ids onto their index in it.
///
/// The capture holds only the pages the request's block table names, so a replay's cache is
/// `blocks.len()` pages long and every table and slot that points into it has to be renumbered.
fn captured_blocks(cap: &Capture) -> (Vec<i64>, BTreeMap<i64, i32>) {
    let blocks: Vec<i64> = cap.man["kv_blocks_captured"]
        .as_array()
        .map(|a| a.iter().filter_map(serde_json::Value::as_i64).collect())
        .unwrap_or_default();
    let renum = blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (*b, i32::try_from(i).unwrap_or(0)))
        .collect();
    (blocks, renum)
}

/// One dispatch of a dflash op, with every `Buffer` slot bound by name from `binds`.
fn dispatch(
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    stream: &Stream,
    rules: &DflashRules,
    op: DflashOp,
    shape: DflashShape,
    binds: &[(&str, DeviceAddr)],
) -> anyhow::Result<String> {
    let d = saber_kernels::dflash::resolve(op, shape, rules)?;
    let mut b = ArgBindings::new(Shape::new(shape.m, shape.n, shape.k));
    for slot in &d.args {
        if let ArgValue::Buffer { buffer } = &slot.value {
            let (_, p) = binds
                .iter()
                .find(|(n, _)| n == buffer)
                .ok_or_else(|| anyhow::anyhow!("{}: nothing bound for `{buffer}`", op.as_str()))?;
            b.bind(buffer, *p);
        }
    }
    launcher.dispatch_one(ctx, &d, &b, stream)?;
    Ok(d.kernel.clone())
}

/// Dispatch one op, optionally through a symbol OTHER than the one the rules route to.
///
/// The control arms of [`compare_sites`] are separate symbols in the same code object — one
/// arithmetic choice changed apiece — so they take the routed descriptor with its kernel name
/// replaced. Everything else about the launch (grid, block, kernarg layout) is the route's,
/// which is the point: a control that also changed the launch would not be a control for the
/// arithmetic. The signature is shared by construction (`abi::signature_for` matches on the
/// symbol PREFIX), and `dispatch_one` refuses a symbol the plugin does not carry.
#[allow(clippy::too_many_arguments)]
fn dispatch_as(
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    stream: &Stream,
    rules: &DflashRules,
    op: DflashOp,
    shape: DflashShape,
    binds: &[(&str, DeviceAddr)],
    symbol: Option<&str>,
) -> anyhow::Result<String> {
    let mut d = saber_kernels::dflash::resolve(op, shape, rules)?;
    if let Some(sym) = symbol {
        d.kernel = sym.to_string();
    }
    let mut b = ArgBindings::new(Shape::new(shape.m, shape.n, shape.k));
    for slot in &d.args {
        if let ArgValue::Buffer { buffer } = &slot.value {
            let (_, p) = binds
                .iter()
                .find(|(n, _)| n == buffer)
                .ok_or_else(|| anyhow::anyhow!("{}: nothing bound for `{buffer}`", op.as_str()))?;
            b.bind(buffer, *p);
        }
    }
    launcher.dispatch_one(ctx, &d, &b, stream)?;
    Ok(d.kernel.clone())
}

/// How a comparison reports one tensor: a name, what Saber wrote, what prod wrote, and the count
/// of differing bytes back. A `type` rather than a closure written out twice, because it crosses
/// two function boundaries and the signature is the contract between them.
type Verdict<'a> = dyn FnMut(&str, &[u8], &[u8]) -> usize + 'a;

/// `bf16` bytes widened to `f32` bytes, element for element.
///
/// ATen's `topk` returns the INPUT dtype, so the selector's selection comes back `bf16` and prod
/// converts it (`values.float()` in `get_top_k_tokens`) before anything reads it. The kernel
/// writes the fp32 value directly, so the comparison widens prod's where prod widens it — which
/// is exact, because bf16 -> f32 is.
fn widen_bf16(b: &[u8]) -> Vec<u8> {
    b.chunks_exact(2)
        .flat_map(|c| {
            let v = u16::from_le_bytes([c[0], c[1]]);
            bf16_to_f32(v).to_le_bytes()
        })
        .collect()
}

/// `i32` index bytes widened to `i64`, which is what `torch.topk` returns.
fn widen_i32(b: &[u8]) -> Vec<u8> {
    b.chunks_exact(4)
        .flat_map(|c| i64::from(i32::from_le_bytes([c[0], c[1], c[2], c[3]])).to_le_bytes())
        .collect()
}

/// `i64` index bytes narrowed to `i32`, which is what every drafter kernel's index slot reads.
///
/// The direction `widen_i32` does not go, and it is needed for the same reason: prod's tensors are
/// ATen's `int64` at every boundary this replay crosses (`torch.topk`'s indices, the drafter's own
/// `input_ids`) and the kernels take `int`. Doing it HERE rather than declaring the kernarg wider
/// is the F22/F24 lesson from the other side -- a gate that converts its SUBJECT's dtypes up is
/// hiding a defect; a replay that converts its REFERENCE's indices down is feeding the kernel the
/// tensor prod's own launch was handed.
fn narrow_i64(b: &[u8]) -> Vec<u8> {
    b.chunks_exact(8)
        .flat_map(|c| {
            let v = i64::from_le_bytes(c.try_into().unwrap_or([0; 8]));
            i32::try_from(v).unwrap_or(0).to_le_bytes()
        })
        .collect()
}

/// **The eleven launch sites that kept the drafter out of the engine**, each against prod's own
/// captured intermediate, each with a control that must differ.
///
/// These were `Coverage::Absent` through F24 and none of them could be gated at all, because the
/// r23 capture did not carry their intermediates: the quant's output has no module boundary at
/// all (prod runs it inside the layer's quant method), `all_kv_flat` is a local, and the two
/// selections happen inside one Python call with nothing between them. The r26 capture taps each
/// one where it happens, and this is the comparison that tap exists for.
///
/// The controls are chosen the way `CLAUDE.md` asks — each differs from its case the way the case
/// could differ from the truth:
///
/// | site | control | what it would mean if the control passed |
/// |---|---|---|
/// | the activation quant, EAGER (`model.fc`) | `absmax * (1/448)` | the inductor lowering is live outside the compiled region too |
/// | ... | the scale APPLIED as a reciprocal multiply | prod's `x_grouped / scales` is not a true divide |
/// | the activation quant, COMPILED (`o_proj`) | — it is a NOTE, not a route | the capture carries no compiled-region quant: its reference `quant.l{i}.*` is the EAGER module, so the route (`act_quant_recip`) MUST differ from it and the CHAIN is what gates it. The division arm is printed beside it as the thing that does match the eager reference |
/// | the KV projection | two and four fp32 accumulators over K | the reference's K order is this kernel's, and the N = 5120 residual was something else |
/// | the layout copy | the layer and context axes swapped | the permutation is not the one `view(...).permute(...)` performs |
/// | the head's top-80 | the composite key without its index half | ATen's tie order is not reachable on this data |
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn compare_sites(
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    stream: &Stream,
    cfg: &DflashConfig,
    rules: &DflashRules,
    cap: &Capture,
    weights_of: &BTreeMap<&str, Vec<(&str, DeviceAddr)>>,
    pool: &mut Pool2,
    verdict: &mut Verdict<'_>,
) -> anyhow::Result<()> {
    let h = cfg.hidden;
    let hd = cfg.head_dim;
    let nkv = cfg.kv_heads_local();
    let num_ctx = cap.int("num_target_tokens")?;
    let m = cap.int("num_query_per_req")?;

    // A CONTROL THAT FIRES MUST NOT FAIL THE GATE, and the first cut of this function let it.
    // `verdict` accumulates into the caller's defect count, which is what decides the exit code;
    // a control arm differing is the control WORKING, and an arm that is a declared residual
    // (the selector's tie order) is a documented fact rather than a defect. Both print through
    // `note`, which counts nothing. The route rows, and only the route rows, go through
    // `verdict` -- so `saber draft-step --proposal` exits non-zero for a wrong kernel and zero
    // for a firing control, which is the distinction the exit code exists to make.
    let note = |name: &str, got: &[u8], want: &[u8]| -> usize {
        let d =
            got.iter().zip(want).filter(|(a, b)| a != b).count() + got.len().abs_diff(want.len());
        println!(
            "   {name:<26} {} of {} bytes",
            if d == 0 {
                "identical".to_string()
            } else {
                format!("DIFFERS in {d}")
            },
            want.len()
        );
        d
    };

    // ---- 1 + 2. the activation quant, at both its shapes ---------------------------------------
    // `op` is the ROUTE at this site and the arms beside it are the other program: the drafter's
    // six `act_quant` launches are TWO programs at one source line (`model.fc` is eager and
    // divides; the five `o_proj`s are inside the compiled region, where inductor strength-reduces
    // the divisor), and `route_is_gated` says whether this site's captured reference is the
    // program the route runs. It is for the eager one and it is NOT for the compiled ones — the
    // capture has no compiled-region quant — so those five print as NOTES and the chain
    // (`chain.l{1..4}.v`) is their gate.
    let quant_site = |launcher: &mut Launcher,
                      pool: &mut Pool2,
                      label: &str,
                      key: &str,
                      rows: u32,
                      n: u32,
                      op: DflashOp,
                      route_is_gated: bool,
                      controls: bool,
                      verdict: &mut Verdict<'_>|
     -> anyhow::Result<()> {
        let note = &note;
        let Ok(x) = cap.bytes(&format!("{key}.x")) else {
            println!(
                "   {label:<26} the capture carries no `{key}.x` (re-capture with r26's probe)"
            );
            return Ok(());
        };
        let wq = cap.bytes(&format!("{key}.q"))?;
        let ws = cap.bytes(&format!("{key}.s"))?;
        let xa = pool.upload(ctx, stream, &x)?;
        let shape = DflashShape::new(rows, n, n);
        let qlen = (rows * n) as usize;
        let slen = (rows * n.div_ceil(128) * 4) as usize;
        for (arm, sym) in [
            ("", None),
            ("  arm div", Some("dflash_act_quant_g128")),
            ("  control recip", Some("dflash_act_quant_g128_recip")),
            ("  control scmul", Some("dflash_act_quant_g128_scmul")),
        ] {
            // The `div` arm is only interesting where the ROUTE is not the division: at the
            // compiled site it is the arm that reproduces the EAGER reference, which is what says
            // the two programs differ and by how much.
            if arm == "  arm div" && route_is_gated {
                continue;
            }
            if !controls && !arm.is_empty() {
                continue;
            }
            let q = pool.zeroed(ctx, stream, qlen)?;
            let sc = pool.zeroed(ctx, stream, slen)?;
            dispatch_as(
                ctx,
                launcher,
                stream,
                rules,
                op,
                shape,
                &[("x", xa), ("q", q), ("scales", sc)],
                sym,
            )?;
            ctx.hip.sync_stream(stream)?;
            let got_q = pool.read(ctx, stream, q, qlen)?;
            let got_s = pool.read(ctx, stream, sc, slen)?;
            if arm.is_empty() && route_is_gated {
                verdict(&format!("{label}.q"), &got_q, &wq);
                verdict(&format!("{label}.s"), &got_s, &ws);
            } else if arm.is_empty() {
                note(&format!("{label}.q  route (recip, NOTE)"), &got_q, &wq);
                note(&format!("{label}.s  route (recip, NOTE)"), &got_s, &ws);
            } else {
                note(&format!("{label}.q{arm}"), &got_q, &wq);
                note(&format!("{label}.s{arm}"), &got_s, &ws);
            }
        }
        Ok(())
    };
    quant_site(
        launcher,
        pool,
        "aux.quant",
        "quant.aux",
        num_ctx,
        cfg.target_layers * h,
        DflashOp::ActQuant,
        true,
        true,
        verdict,
    )?;
    for l in 0..cfg.layers {
        quant_site(
            launcher,
            pool,
            &format!("l{l}.o_proj.quant"),
            &format!("quant.l{l}"),
            m,
            cfg.q_heads_local() * hd,
            DflashOp::ActQuantRecip,
            false,
            l == 0,
            verdict,
        )?;
    }

    // ---- 3. the fused context-KV projection, at the N the envelope stopped at -------------------
    if let (Ok(a), Ok(want)) = (cap.bytes("ctx.kv_norm_x"), cap.bytes("ctx.kv_flat")) {
        let n = cfg.context_kv_n();
        let aa = pool.upload(ctx, stream, &a)?;
        let w = weights_of
            .get("context_kv.kv_projection")
            .and_then(|v| v.iter().find(|(nm, _)| *nm == "w"))
            .map(|(_, a)| *a);
        match w {
            None => println!(
                "   ctx.kv_projection          the pack carries no `context_kv.kv_projection.w`"
            ),
            Some(wa) => {
                let shape = DflashShape::new(num_ctx, n, h);
                for (arm, sym) in [
                    ("", None),
                    ("  arm acc2", Some("dflash_bgemm_bf16_m16n16_acc2")),
                    ("  arm acc4", Some("dflash_bgemm_bf16_m16n16_acc4")),
                ] {
                    let c = pool.zeroed(ctx, stream, (num_ctx * n * 2) as usize)?;
                    dispatch_as(
                        ctx,
                        launcher,
                        stream,
                        rules,
                        DflashOp::BgemmBf16,
                        shape,
                        &[("a", aa), ("w", wa), ("c", c)],
                        sym,
                    )?;
                    ctx.hip.sync_stream(stream)?;
                    let got = pool.read(ctx, stream, c, want.len())?;
                    if arm.is_empty() {
                        verdict("ctx.kv_projection", &got, &want);
                    } else {
                        note(&format!("ctx.kv_projection{arm}"), &got, &want);
                    }
                }
            }
        }
    } else {
        println!("   ctx.kv_projection          the capture carries no `ctx.kv_flat`");
    }

    // ---- 4. the layout copy ---------------------------------------------------------------------
    if let (Ok(flat), Ok(k), Ok(v)) = (
        cap.bytes("ctx.kv_flat"),
        cap.bytes("ctx.all_k"),
        cap.bytes("ctx.all_v"),
    ) {
        let mut want = k;
        want.extend_from_slice(&v);
        let ina = pool.upload(ctx, stream, &flat)?;
        let shape = DflashShape::ctx_layout(num_ctx, cfg.layers, nkv, hd);
        for (arm, sym) in [
            ("", None),
            ("  control swap", Some("dflash_ctx_kv_layout_swapped")),
        ] {
            let out = pool.zeroed(ctx, stream, want.len())?;
            dispatch_as(
                ctx,
                launcher,
                stream,
                rules,
                DflashOp::CtxKvLayout,
                shape,
                &[("in", ina), ("out", out)],
                sym,
            )?;
            ctx.hip.sync_stream(stream)?;
            let got = pool.read(ctx, stream, out, want.len())?;
            if arm.is_empty() {
                verdict("ctx.layout", &got, &want);
            } else {
                note(&format!("ctx.layout{arm}"), &got, &want);
            }
        }
    } else {
        println!("   ctx.layout                 the capture carries no `ctx.all_k`/`ctx.all_v`");
    }

    // ---- 5 + 6. the two ATen selections ----------------------------------------------------------
    //
    // Which capture slot is which is read off the manifest's own `topk.{j}.k` rather than assumed
    // from the order: both happen inside one `compute_candidates` call and the int2 head's is
    // first only because `_apply_head_int2` runs before `get_top_k_tokens`'s own selection.
    let calls = cap.int("topk.calls").unwrap_or(0);
    for j in 0..calls {
        let k = cap.int(&format!("topk.{j}.k"))?;
        let x = cap.bytes(&format!("topk.{j}.x"))?;
        let want_v = cap.bytes(&format!("topk.{j}.values"))?;
        let want_i = cap.bytes(&format!("topk.{j}.indices"))?;
        let sh = cap.shape(&format!("topk.{j}.x"))?;
        let (rows, n) = (
            u32::try_from(sh[0]).unwrap_or(1),
            u32::try_from(sh[1]).unwrap_or(1),
        );
        let xa = pool.upload(ctx, stream, &x)?;
        let vlen = (rows * k * 4) as usize;
        // The two selections do NOT write the same index width, and the difference is their
        // consumers'. The selector's top-16 feeds `dflash_selector_score`, whose `cand` is
        // `const long *`, so it writes i64 (F28); the head's top-80 feeds `_rerank_exact`, which
        // indexes head ROWS with an `int` and is handed a converted table (the `Narrowed`
        // binding in `tests/capture_widths.rs`), so it writes i32 and the comparison widens it.
        let sel_i64 = k == cfg.selector_top_k;
        let ilen = (rows * k * if sel_i64 { 8 } else { 4 }) as usize;
        if sel_i64 {
            // THE SELECTOR'S TOP-16, and its indices are the one thing in this gate that is
            // DECLARED rather than matched.
            //
            // Measured on the host from this capture before any of it was launched: the selected
            // SET is identical, the VALUE sequence is identical, and the indices differ in
            // 11 / 8 / 13 of 112 — every one a pair of EXACTLY EQUAL bf16 logits. Ascending index
            // differs in 11 and DESCENDING in 13 on the same row, so ATen's order at this shape is
            // not one rule: `mbtopk` merges per-block results and the order among equal values
            // falls out of that merge. The LEDGER's tie-order entry says the rule depends on which
            // algorithm ATen picked; this shape says it is not always a rule at all.
            //
            // So the kernel keeps the total order it can defend (value descending, index
            // ascending) and this gate states the residual instead of hiding it. The
            // `hightie` arm is the control: it must move exactly the tied positions.
            let shape = DflashShape::rerank(rows, n, h, k);
            for (arm, sym) in [
                ("", None),
                ("  control hightie", Some("dflash_topk_bf16_k16_hightie")),
            ] {
                let vals = pool.zeroed(ctx, stream, vlen)?;
                let idx = pool.zeroed(ctx, stream, ilen)?;
                dispatch_as(
                    ctx,
                    launcher,
                    stream,
                    rules,
                    DflashOp::SelectorTopk,
                    shape,
                    &[("x", xa), ("vals", vals), ("idx", idx)],
                    sym,
                )?;
                ctx.hip.sync_stream(stream)?;
                let gv = pool.read(ctx, stream, vals, vlen)?;
                let gi = pool.read(ctx, stream, idx, ilen)?;
                let dv = if arm.is_empty() {
                    verdict("selector.candidate_topk.v", &gv, &widen_bf16(&want_v))
                } else {
                    note(
                        &format!("selector.candidate_topk.v{arm}"),
                        &gv,
                        &widen_bf16(&want_v),
                    )
                };
                // No widening: this kernel writes prod's own width now.
                let wide = gi.clone();
                // DECLARED, not matched: see the comment above. It prints and it does not count.
                let d = note(&format!("selector.candidate_topk.i{arm}"), &wide, &want_i);
                // The SET, which is what everything downstream reads: the scorer gathers the
                // codebook rows by id and the walk maps its argmax back to one. A different
                // ORDER of the same sixteen permutes the score matrix's columns and the walk's
                // answer with it, so this is a real residual — it is just not a wrong selection.
                let mut a: Vec<i64> = wide
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap_or([0; 8])))
                    .collect();
                let mut b: Vec<i64> = want_i
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap_or([0; 8])))
                    .collect();
                for r in 0..rows as usize {
                    let (lo, hi) = (r * k as usize, (r + 1) * k as usize);
                    a[lo..hi].sort_unstable();
                    b[lo..hi].sort_unstable();
                }
                let same_set = a == b;
                println!(
                    "     the SET is {}, the value sequence differs in {dv} byte(s), and the \
                     index residual is {d} byte(s) -- DECLARED as ties (the set and the values \
                     are what everything downstream reads), or a wrong selection if the first \
                     two are not zero",
                    if same_set { "identical" } else { "DIFFERENT" }
                );
                // The set and the values ARE a defect if they move, and they are the half of
                // this site that the exit code should still see.
                if arm.is_empty() && (!same_set || dv != 0) {
                    verdict("selector.candidate_topk.set", &[0u8], &[1u8]);
                }
            }
        } else {
            // The int2 head's top-80 over the block maxima, fp32 in, two passes.
            let shape = DflashShape::rerank(rows, n, h, k);
            for (arm, sym) in [
                ("", None),
                ("  control notie", Some("dflash_topk_f32_gather_k80_notie")),
            ] {
                let th = pool.zeroed(ctx, stream, (rows * 8) as usize)?;
                let vals = pool.zeroed(ctx, stream, vlen)?;
                let idx = pool.zeroed(ctx, stream, ilen)?;
                dispatch_as(
                    ctx,
                    launcher,
                    stream,
                    rules,
                    DflashOp::HeadTopkThresh,
                    shape,
                    &[("x", xa), ("thresh", th)],
                    None,
                )?;
                dispatch_as(
                    ctx,
                    launcher,
                    stream,
                    rules,
                    DflashOp::HeadTopkGather,
                    shape,
                    &[("x", xa), ("thresh", th), ("vals", vals), ("idx", idx)],
                    sym,
                )?;
                ctx.hip.sync_stream(stream)?;
                let gv = pool.read(ctx, stream, vals, vlen)?;
                let gi = pool.read(ctx, stream, idx, ilen)?;
                if arm.is_empty() {
                    verdict("head.candidate_topk.v", &gv, &want_v);
                    verdict("head.candidate_topk.i", &widen_i32(&gi), &want_i);
                } else {
                    note(&format!("head.candidate_topk.v{arm}"), &gv, &want_v);
                    note(
                        &format!("head.candidate_topk.i{arm}"),
                        &widen_i32(&gi),
                        &want_i,
                    );
                }
            }
        }
    }
    if calls == 0 {
        println!("   head/selector topk         the capture carries no `topk.*` (r23 or older)");
    }

    // ---- 7 + 8 + 9. the three launches INSIDE `_apply_head_int2` -------------------------------
    //
    // None of them crosses a module boundary, an ATen binding or a quant method: all three are
    // LOCALS of one Python function, which is why six rounds of this gate did not see them and
    // why the r26 probe taps the head's own Triton call to get them.
    if let (Ok(live), Ok(want_pad)) = (cap.bytes("head.hidden"), cap.bytes("head.x_pad")) {
        let pad_rows = u32::try_from(cap.shape("head.x_pad")?[0]).unwrap_or(16);
        let live_rows = u32::try_from(cap.shape("head.hidden")?[0]).unwrap_or(m);
        let xa = pool.upload(ctx, stream, &live)?;
        let shape = DflashShape::rerank(pad_rows, h, h, live_rows);
        for (arm, sym) in [
            ("", None),
            (
                "  control fillfirst",
                Some("dflash_head_row_pad_bf16_fillfirst"),
            ),
        ] {
            let out = pool.zeroed(ctx, stream, (pad_rows * h * 2) as usize)?;
            dispatch_as(
                ctx,
                launcher,
                stream,
                rules,
                DflashOp::HeadRowPad,
                shape,
                &[("x", xa), ("out", out)],
                sym,
            )?;
            ctx.hip.sync_stream(stream)?;
            let got = pool.read(ctx, stream, out, want_pad.len())?;
            if arm.is_empty() {
                verdict("head.row_pad", &got, &want_pad);
            } else {
                note(&format!("head.row_pad{arm}"), &got, &want_pad);
            }
        }

        // `xs`, and its two controls are the two OTHER readings of ATen's own reduction: the
        // non-vectorized lane layout and no lane decomposition at all.
        if let Ok(want_xs) = cap.bytes("head.xs") {
            let pa = pool.upload(ctx, stream, &want_pad)?;
            let shape = DflashShape::new(pad_rows, h, h);
            for (arm, sym) in [
                ("", None),
                (
                    "  control strided",
                    Some("dflash_head_group_sum_g128_strided"),
                ),
                ("  control seq", Some("dflash_head_group_sum_g128_seq")),
            ] {
                let xs = pool.zeroed(ctx, stream, want_xs.len())?;
                dispatch_as(
                    ctx,
                    launcher,
                    stream,
                    rules,
                    DflashOp::HeadGroupSum,
                    shape,
                    &[("x", pa), ("xs", xs)],
                    sym,
                )?;
                ctx.hip.sync_stream(stream)?;
                let got = pool.read(ctx, stream, xs, want_xs.len())?;
                if arm.is_empty() {
                    verdict("head.group_sum", &got, &want_xs);
                } else {
                    note(&format!("head.group_sum{arm}"), &got, &want_xs);
                }
            }
        } else {
            println!("   head.group_sum             the capture carries no `head.xs`");
        }

        // THE CONTROL THE CAPTURED ROWS CANNOT CARRY (F27 recorded it as bad, F28 fixes it).
        //
        // `strided` and `seq` ARE different fold orders -- `thread_reduce_impl`'s lane layout,
        // and no lane decomposition at all -- but on prod's own rows they differed in 0 to 4
        // bytes of 2,560 and were IDENTICAL on p86. 128 well-conditioned bf16 values summed in
        // fp32 rarely disagree, so the arm proved the route was *a* plausible order rather than
        // *the* order, and rule 8 says a control that cannot fail is not one.
        //
        // The data is the problem, not the arms, so this gives them data. Each group of 128 is a
        // deterministic wide-dynamic-range row: +/- 2^e with e swept over [-14, 14], which is
        // bf16-exact in every element and whose fp32 SUM depends on the order at the top bits
        // rather than the bottom one. The reference here is the ROUTE'S OWN output on the same
        // row -- there is no prod tensor for a synthetic row and none is wanted: what this arm
        // asserts is that the three orders are DISTINGUISHABLE, which is the property the
        // captured comparison needs and cannot supply. A control that does not fire is counted
        // as a defect, by name.
        {
            let groups = h / 128;
            let rows = 16u32;
            let mut syn = Vec::with_capacity((rows * h * 2) as usize);
            let mut lcg: u32 = 0x9e37_79b9;
            for _ in 0..rows * h {
                lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                // Sign from one bit, exponent from five: 2^-14 .. 2^14, exactly representable in
                // bf16 and spanning 28 binades, so a re-association moves the sum's top bits.
                let e = i32::try_from((lcg >> 8) & 0x1f).unwrap_or(0) - 14;
                let v = (if lcg & 1 == 0 { 1.0f32 } else { -1.0f32 }) * 2f32.powi(e);
                syn.extend_from_slice(&f32_to_bf16_rne(v).to_le_bytes());
            }
            let sa = pool.upload(ctx, stream, &syn)?;
            let shape = DflashShape::new(rows, h, h);
            let xs_len = (rows * groups * 4) as usize;
            let mut route: Vec<u8> = Vec::new();
            for (arm, sym) in [
                ("", None),
                (
                    "  control strided",
                    Some("dflash_head_group_sum_g128_strided"),
                ),
                ("  control seq", Some("dflash_head_group_sum_g128_seq")),
            ] {
                let xs = pool.zeroed(ctx, stream, xs_len)?;
                dispatch_as(
                    ctx,
                    launcher,
                    stream,
                    rules,
                    DflashOp::HeadGroupSum,
                    shape,
                    &[("x", sa), ("xs", xs)],
                    sym,
                )?;
                ctx.hip.sync_stream(stream)?;
                let got = pool.read(ctx, stream, xs, xs_len)?;
                if arm.is_empty() {
                    route = got;
                    println!(
                        "   head.group_sum.synthetic   the route's own {xs_len} bytes over a \
                         wide-dynamic-range row"
                    );
                } else {
                    let d = note(&format!("head.group_sum.synthetic{arm}"), &got, &route);
                    if d == 0 {
                        verdict(
                            &format!(
                                "head.group_sum.synthetic{arm} DID NOT FIRE -- a fold order this \
                                 row cannot see is not a control"
                            ),
                            &[0],
                            &[1],
                        );
                    }
                }
            }
        }
    } else {
        println!("   head.row_pad               the capture carries no `head.x_pad` (pre-r27)");
    }

    // `bi.gather(1, bm.topk(80).indices)`. The positions come from the capture's own selection
    // rather than from Saber's, so this site asks about the GATHER and nothing else.
    if let (Ok(bidx), Ok(want_idx)) = (cap.bytes("head.bidx"), cap.bytes("head.rerank_idx")) {
        let pos_key = (0..calls)
            .find(|j| cap.int(&format!("topk.{j}.k")).unwrap_or(0) == cfg.rerank)
            .map(|j| format!("topk.{j}.indices"));
        if let Some(key) = pos_key {
            let sh = cap.shape("head.bidx")?;
            let (rows, n) = (
                u32::try_from(sh[0]).unwrap_or(16),
                u32::try_from(sh[1]).unwrap_or(1),
            );
            let pos = narrow_i64(&cap.bytes(&key)?);
            let ba = pool.upload(ctx, stream, &bidx)?;
            let pa = pool.upload(ctx, stream, &pos)?;
            let shape = DflashShape::rerank(rows, n, h, cfg.rerank);
            for (arm, sym) in [
                ("", None),
                ("  control pos", Some("dflash_head_cand_gather_pos")),
            ] {
                let out = pool.zeroed(ctx, stream, want_idx.len())?;
                dispatch_as(
                    ctx,
                    launcher,
                    stream,
                    rules,
                    DflashOp::HeadCandGather,
                    shape,
                    &[("bidx", ba), ("pos", pa), ("out", out)],
                    sym,
                )?;
                ctx.hip.sync_stream(stream)?;
                let got = pool.read(ctx, stream, out, want_idx.len())?;
                if arm.is_empty() {
                    verdict("head.candidate_gather", &got, &want_idx);
                } else {
                    note(&format!("head.candidate_gather{arm}"), &got, &want_idx);
                }
            }
        }
    } else {
        println!("   head.candidate_gather      the capture carries no `head.bidx`/`rerank_idx`");
    }

    // ---- 12. the drafter's OWN q/k norm + rope, STAGED from prod's own projection -------------
    //
    // The ARITHMETIC arm for `dflash_qk`. What it is NOT is the bit-exact claim: that is the
    // chain's `chain.attn.l{l}.q` against the LIVE attention tap, because the only tensor that
    // can be fed to this kernel outside a chain is r28's eager SHADOW of the projection, and
    // that capture measures its own shadow at 21,750-22,417 differing bytes of 65,536 against
    // the live post-rope q. Eager and inductor are not the same program here -- `ir.ops.rms_norm`
    // rounds the scaled value to bf16 before the weight multiply and the compiled kernel does
    // not, and the conv in front of it is fused the same way -- so a staged arm fed the shadow
    // cannot be identical to prod's bytes and a gate that demanded it would be measuring the
    // capture, not the kernel.
    //
    // What it DOES ask is the one thing a chain cannot: whether the two arithmetic choices that
    // are invisible in the source are choices this DATA can see. Both controls run on the same
    // staged input as the route and are held against the ROUTE'S OWN output, which is what makes
    // "these are different orders" falsifiable rather than asserted.
    //
    // THE CONTROLS ARE THE TWO CHOICES THAT ARE NOT VISIBLE IN THE SOURCE. `ord1` sums the four
    // elements a lane owns in index order instead of the reference's (second, first, third,
    // fourth -- the order its ISA opens the accumulator in), and `rsqf64` computes the reciprocal
    // square root the way ATen does, correctly rounded, instead of the `v_rsq_f32` Triton emits.
    // Each is one arithmetic choice and nothing else; both must differ, and the second is the one
    // the LEDGER already prices at 15 of 40 rope cases.
    if cap.bytes("chain.l0.qkv").is_ok() {
        let nq = cfg.q_heads_local();
        let nkv = cfg.kv_heads_local();
        let pos = pool.upload(ctx, stream, &cap.bytes("q.positions")?)?;
        let cs = weights_of
            .get("context_kv.rope")
            .and_then(|v| v.iter().find(|(n, _)| *n == "cos_sin"))
            .map(|(_, a)| *a)
            .ok_or_else(|| anyhow::anyhow!("the pack carries no `context_kv.rope.cos_sin`"))?;
        for l in 0..cfg.layers {
            let (Ok(qkv), Ok(want_q), Ok(want_k)) = (
                cap.bytes(&format!("chain.l{l}.qkv")),
                cap.bytes(&format!("attn.l{l}.q")),
                cap.bytes(&format!("attn.l{l}.k")),
            ) else {
                continue;
            };
            let qa = pool.upload(ctx, stream, &qkv)?;
            let op = format!("layer.{l}.attn.qk_norm_rope");
            let wq = weights_of
                .get(op.as_str())
                .and_then(|v| v.first())
                .map(|(_, a)| *a)
                .ok_or_else(|| anyhow::anyhow!("the pack carries no `{op}` q weight"))?;
            let wk = weights_of
                .get(op.as_str())
                .and_then(|v| v.get(1))
                .map(|(_, a)| *a)
                .ok_or_else(|| anyhow::anyhow!("the pack carries no `{op}` k weight"))?;
            let qo = pool.zeroed(ctx, stream, want_q.len())?;
            let ko = pool.zeroed(ctx, stream, want_k.len())?;
            dispatch(
                ctx,
                launcher,
                stream,
                rules,
                DflashOp::QkNormRope,
                DflashShape::qk_rope(m, nq, nkv, hd),
                &[
                    ("qkv", qa),
                    ("qw", wq),
                    ("kw", wk),
                    ("cos_sin", cs),
                    ("pos", pos),
                    ("q_out", qo),
                    ("k_out", ko),
                ],
            )?;
            ctx.hip.sync_stream(stream)?;
            // Against prod's own post-rope q and k, from a staged projection the capture does
            // NOT vouch for: a distance, printed, not a defect. The CONTROLS are below, on data
            // that can see them.
            note(
                &format!("qk_norm_rope.l{l}.q"),
                &pool.read(ctx, stream, qo, want_q.len())?,
                &want_q,
            );
            note(
                &format!("qk_norm_rope.l{l}.k"),
                &pool.read(ctx, stream, ko, want_k.len())?,
                &want_k,
            );
        }
        // THE CONTROLS, ON DATA THAT CAN SEE THEM. Run per layer on prod's own rows they were
        // both IDENTICAL to the route -- 128 well-conditioned bf16 values summed in fp32 agree
        // whatever the order, and `v_rsq_f32` is within an ULP of the correctly-rounded one on
        // arguments that are all near 1 -- so as arms over the captured shapes they proved
        // nothing, which is the same failure F27 recorded for `head.group_sum` and rule 8's
        // whole point. The data is what is wrong with them, so this gives them data: a synthetic
        // projection whose every head spans 28 binades, where a re-association of the variance and a
        // one-ULP `inv` both move the bf16 output. The reference is the ROUTE'S OWN output on the
        // same rows; what is asserted is that the three arithmetics are DISTINGUISHABLE.
        {
            let nq = cfg.q_heads_local();
            let nkv = cfg.kv_heads_local();
            let qkv_n = (nq + 2 * nkv) * hd;
            let mut syn = Vec::with_capacity((m * qkv_n * 2) as usize);
            let mut lcg: u32 = 0x1234_5678;
            for _ in 0..m * qkv_n {
                lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let e = i32::try_from((lcg >> 8) & 0x1f).unwrap_or(0) - 14;
                let v = (if lcg & 1 == 0 { 1.0f32 } else { -1.0f32 }) * 2f32.powi(e);
                syn.extend_from_slice(&f32_to_bf16_rne(v).to_le_bytes());
            }
            let sa = pool.upload(ctx, stream, &syn)?;
            let zero = pool.zeroed(ctx, stream, (m * 8) as usize)?;
            let op = "layer.0.attn.qk_norm_rope";
            let wq = weights_of
                .get(op)
                .and_then(|v| v.first())
                .map(|(_, a)| *a)
                .ok_or_else(|| anyhow::anyhow!("the pack carries no `{op}` q weight"))?;
            let wk = weights_of
                .get(op)
                .and_then(|v| v.get(1))
                .map(|(_, a)| *a)
                .ok_or_else(|| anyhow::anyhow!("the pack carries no `{op}` k weight"))?;
            let (qlen, klen) = ((m * nq * hd * 2) as usize, (m * nkv * hd * 2) as usize);
            let (mut rq, mut rk) = (Vec::new(), Vec::new());
            for (arm, sym, must_fire) in [
                ("", None, false),
                (
                    "  control bf16n",
                    Some("dflash_qk_norm_rope_h128_bf16n"),
                    true,
                ),
                (
                    "  control ord1",
                    Some("dflash_qk_norm_rope_h128_ord1"),
                    true,
                ),
                (
                    "  control rsqf64",
                    Some("dflash_qk_norm_rope_h128_rsqf64"),
                    false,
                ),
            ] {
                let qo = pool.zeroed(ctx, stream, qlen)?;
                let ko = pool.zeroed(ctx, stream, klen)?;
                dispatch_as(
                    ctx,
                    launcher,
                    stream,
                    rules,
                    DflashOp::QkNormRope,
                    DflashShape::qk_rope(m, nq, nkv, hd),
                    &[
                        ("qkv", sa),
                        ("qw", wq),
                        ("kw", wk),
                        ("cos_sin", cs),
                        ("pos", zero),
                        ("q_out", qo),
                        ("k_out", ko),
                    ],
                    sym,
                )?;
                ctx.hip.sync_stream(stream)?;
                let gq = pool.read(ctx, stream, qo, qlen)?;
                let gk = pool.read(ctx, stream, ko, klen)?;
                if arm.is_empty() {
                    println!(
                        "   qk_norm_rope.synthetic     the route's own {qlen} + {klen} bytes \
                         over a wide-dynamic-range projection"
                    );
                    rq = gq;
                    rk = gk;
                } else {
                    let d = note(&format!("qk_norm_rope.synthetic.q{arm}"), &gq, &rq)
                        + note(&format!("qk_norm_rope.synthetic.k{arm}"), &gk, &rk);
                    // `rsqf64` is not required to fire and says so: `v_rsq_f32` is within an ULP
                    // of the correctly-rounded reciprocal square root, and one ULP of `inv`
                    // reaches a bf16 output only when the product lands on a rounding boundary.
                    // It is a real alternative arithmetic that THIS data cannot see, which is a
                    // different thing from a control that is the same function -- and the
                    // difference is exactly what `ord1`'s first draft got wrong.
                    if d == 0 && must_fire {
                        verdict(
                            &format!(
                                "qk_norm_rope.synthetic{arm} DID NOT FIRE -- an arithmetic \
                                 choice no data can see is not a control"
                            ),
                            &[0],
                            &[1],
                        );
                    }
                }
            }
        }
    } else {
        println!("   qk_norm_rope               the capture carries no `chain.l0.qkv` (pre-r28)");
    }

    // ---- 10 + 11. the two row gathers `_generate_draft` makes ----------------------------------
    //
    // Seeded, because what they ask is which ROWS and not what arithmetic: prod's own
    // `last_hidden` and prod's own `input_ids`, gathered by prod's own index tables. The
    // `_noidx` control is the identity, which is what a replay that never looked at
    // `sample_indices` would compute -- and it is the whole defect this op exists to prevent,
    // because a chain that takes the FIRST 7 of 8 rows produces a plausible block from the wrong
    // positions.
    let gather_site = |launcher: &mut Launcher,
                       pool: &mut Pool2,
                       label: &str,
                       src: &[u8],
                       idx: &[u8],
                       want: &[u8],
                       src_rows: u32,
                       verdict: &mut Verdict<'_>|
     -> anyhow::Result<()> {
        let note = &note;
        let rows = u32::try_from(idx.len() / 8).unwrap_or(1).max(1);
        let row_bytes = u32::try_from(want.len() / rows.max(1) as usize).unwrap_or(1);
        let ia = pool.upload(ctx, stream, src)?;
        let xa = pool.upload(ctx, stream, idx)?;
        for (arm, sym) in [
            ("", None),
            ("  control noidx", Some("dflash_row_gather_noidx")),
        ] {
            let out = pool.zeroed(ctx, stream, want.len())?;
            dispatch_as(
                ctx,
                launcher,
                stream,
                rules,
                DflashOp::RowGather,
                DflashShape::new(rows, row_bytes, src_rows),
                &[("in", ia), ("idx", xa), ("out", out)],
                sym,
            )?;
            ctx.hip.sync_stream(stream)?;
            let got = pool.read(ctx, stream, out, want.len())?;
            if arm.is_empty() {
                verdict(label, &got, want);
            } else {
                note(&format!("{label}{arm}"), &got, want);
            }
        }
        Ok(())
    };
    if let (Ok(lh), Ok(si), Ok(hh)) = (
        cap.bytes("q.last_hidden"),
        cap.bytes("idx.sample_indices"),
        cap.bytes("head.hidden"),
    ) {
        let src_rows = u32::try_from(cap.shape("q.last_hidden")?[0]).unwrap_or(m);
        gather_site(
            launcher,
            pool,
            "head.sample_rows",
            &lh,
            &si,
            &hh,
            src_rows,
            verdict,
        )?;
    } else {
        println!("   head.sample_rows           the capture carries no `q.last_hidden`");
    }
    // `_anchor_indices` is `arange(max_num_reqs) * num_query_per_req`, so the request's anchor is
    // element `req * num_query_per_req` of the query ids. Derived rather than captured, the same
    // way the staged selector stage derives it.
    if let (Ok(ids), Ok(want)) = (cap.bytes("q.input_ids"), cap.bytes("sel.anchor_token_ids")) {
        let reqs = cap.int("num_reqs").unwrap_or(1).max(1);
        let q = cap.int("num_query_per_req").unwrap_or(1).max(1);
        let src_rows = u32::try_from(cap.shape("q.input_ids")?[0]).unwrap_or(q);
        let idx: Vec<u8> = (0..reqs)
            .flat_map(|r| i64::from(r * q).to_le_bytes())
            .collect();
        gather_site(
            launcher,
            pool,
            "selector.anchor_ids",
            &ids,
            &idx,
            &want,
            src_rows,
            verdict,
        )?;
    } else {
        println!("   selector.anchor_ids        the capture carries no `sel.anchor_token_ids`");
    }

    // THE CONTROL THE CAPTURED SHAPE CANNOT CARRY (F27 recorded it as vacuous, F28 fixes it).
    //
    // At ONE request `_anchor_indices` is `[0]` and the identity IS the answer, so the `noidx`
    // arm above agrees with the route for a reason that has nothing to do with the route being
    // right. The capture is of a one-request step and re-taking it at two is a different serve,
    // so the control gets its own SHAPE instead: two requests of eight query rows, ids that are
    // distinct per row, and `_anchor_indices = [0, 8]` -- prod's own
    // `arange(max_num_reqs) * num_query_per_req`. The route must gather rows 0 and 8; the
    // identity gathers rows 0 and 1 and therefore differs in exactly the second element.
    //
    // A gather computes nothing, so a synthetic reference is the arithmetic itself rather than a
    // model of it: `want` is built here by indexing, which is the whole definition of the op.
    {
        let q = 8u32;
        let reqs = 2u32;
        let ids: Vec<u8> = (0..reqs * q)
            .flat_map(|i| (1000 + i32::try_from(i).unwrap_or(0)).to_le_bytes())
            .collect();
        let idx: Vec<u8> = (0..reqs)
            .flat_map(|r| i64::from(r * q).to_le_bytes())
            .collect();
        let want: Vec<u8> = (0..reqs)
            .flat_map(|r| (1000 + i32::try_from(r * q).unwrap_or(0)).to_le_bytes())
            .collect();
        let ia = pool.upload(ctx, stream, &ids)?;
        let xa = pool.upload(ctx, stream, &idx)?;
        for (arm, sym) in [
            ("", None),
            ("  control noidx", Some("dflash_row_gather_noidx")),
        ] {
            let out = pool.zeroed(ctx, stream, want.len())?;
            dispatch_as(
                ctx,
                launcher,
                stream,
                rules,
                DflashOp::RowGather,
                DflashShape::new(reqs, 4, reqs * q),
                &[("in", ia), ("idx", xa), ("out", out)],
                sym,
            )?;
            ctx.hip.sync_stream(stream)?;
            let got = pool.read(ctx, stream, out, want.len())?;
            if arm.is_empty() {
                verdict("selector.anchor_ids.2req", &got, &want);
            } else {
                let d = note(&format!("selector.anchor_ids.2req{arm}"), &got, &want);
                if d == 0 {
                    verdict(
                        "selector.anchor_ids.2req control noidx DID NOT FIRE -- at two requests \
                         the identity is not the answer",
                        &[0],
                        &[1],
                    );
                }
            }
        }
    }
    Ok(())
}

/// THE DRAFTER'S ATTENTION AT EIGHT KV HEADS, against prod's own per-layer output.
///
/// F22 gave `dflash_attn` a y grid extent of `ceil(num_kv_heads / waves)` because `kvh = wave`
/// half-computes at TP=1: the checkpoint's 8 kv heads meet four waves and heads 4..7 were never
/// visited, which `resolve`'s GQA check (`r * gqa == n`, satisfied by 8 x 4 = 32) cannot see. That
/// extent is **1** at the drafter's TP=2 shape, which is where every earlier attention capture was
/// taken — so `gate_dflash.sh --only attn` compares byte-for-byte the same program before and
/// after the fix and **cannot** gate it. This is the arm that can: prod's own five drafter layers
/// at TP=1, each seeded with prod's query rows and prod's own KV cache.
///
/// Seeded, not chained: `q` is the capture's, because the rope and the per-head norms upstream of
/// it are `attn_glue`'s and are gated elsewhere. What this asks is the attention alone.
#[allow(clippy::too_many_arguments)]
fn compare_attention(
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    stream: &Stream,
    cfg: &DflashConfig,
    rules: &DflashRules,
    cap: &Capture,
    pool: &mut Pool2,
    explain: bool,
) -> anyhow::Result<(usize, usize)> {
    let hd = cfg.head_dim as usize;
    let nkv = cfg.kv_heads_local();
    let nq = cfg.q_heads_local();
    let kv_block = cap.int("kv_block_size")?;
    if cap.bytes("attn.l0.q").is_err() {
        println!(
            "   attention                  SKIPPED: this capture carries no per-layer attention \
             tensors (taken before round 23's impl tap). Nothing gates the kernel at 8 kv heads \
             without them"
        );
        return Ok((0, 0));
    }
    let (blocks, renum) = captured_blocks(cap);
    let nb = blocks.len().max(1);
    let mut differ = 0usize;
    let mut done = 0usize;
    for l in 0..cfg.layers {
        let (qk, ok, kvk) = (
            format!("attn.l{l}.q"),
            format!("attn.l{l}.out"),
            format!("kv.after_fwd.l{l}"),
        );
        let (Ok(q), Ok(mut want)) = (cap.bytes(&qk), cap.bytes(&ok)) else {
            println!("   attn l{l}                   MISSING from the capture");
            continue;
        };
        let q_shape = cap.shape(&qk)?;
        // ROWS IS `num_actual_tokens`, NOT THE BUFFER'S FIRST DIMENSION. The reference computes
        // `output[:num_actual_tokens]` and leaves the rest of a `torch.empty` untouched, so a
        // comparison over the whole buffer would be against uninitialised memory — the padded
        // rows would read as a mismatch on a kernel that is right, or, worse, agree by accident.
        // At `disable_padded_drafter_batch` and one request the two are equal; the rule is what
        // keeps that a fact rather than an assumption.
        let rows = u32::try_from(
            cap.man
                .get(format!("attn.l{l}.meta"))
                .and_then(|m| m["num_actual_tokens"].as_u64())
                .map_or_else(|| q_shape.first().copied().unwrap_or(0), |v| v as usize),
        )
        .unwrap_or(0);
        want.truncate((rows as usize) * (nq as usize) * hd * 2);
        // PROD'S OWN CONSTANTS, asserted rather than assumed. `window` and `scale` are rules
        // values (`1 + window_size[0]` and `head_dim ** -0.5`) and the kernel is non-causal by
        // construction; the capture records what prod's metadata actually said, so a checkpoint
        // that changes any of the three fails here instead of quietly gating a different
        // computation.
        if let Some(meta) = cap.man.get(format!("attn.l{l}.meta")) {
            let win = meta["sliding_window"][0].as_i64().map(|w| w + 1);
            // `causal` is a bool in the metadata today and the field's type is
            // `bool | torch.Tensor`, so the capture records a list when it is a tensor. Only a
            // DEFINITE yes is a refusal; an unreadable one is said out loud and the comparison
            // still runs, because a gate that aborts on a field it cannot parse gates nothing.
            let causal = meta["causal"].as_bool().or_else(|| {
                meta["causal"]
                    .as_array()
                    .map(|a| a.iter().any(|v| v.as_bool().unwrap_or(false)))
            });
            let heads = meta["num_kv_heads"].as_u64();
            if explain {
                println!(
                    "   attn l{l}    prod: {:?} kv heads, window {:?}, causal {:?}, \
                     scale {:.6}, {}",
                    heads,
                    win,
                    causal,
                    meta["scale"].as_f64().unwrap_or(0.0),
                    meta["kv_cache_dtype"].as_str().unwrap_or("?")
                );
            }
            match causal {
                Some(true) => anyhow::bail!(
                    "layer {l}: prod's attention metadata says CAUSAL — `dflash_attn` is \
                     non-causal by construction and this gate would be comparing two different \
                     computations"
                ),
                None => println!(
                    "     ^ this capture does not state `causal`; the kernel is non-causal and \
                     the comparison below assumes prod's was too"
                ),
                Some(false) => {}
            }
            if let Some(w) = win {
                if w != i64::from(rules.attn.window) {
                    anyhow::bail!(
                        "layer {l}: prod's window is {w} and the rules' is {}",
                        rules.attn.window
                    );
                }
            }
            if let Some(h) = heads {
                if u64::from(nkv) != h {
                    anyhow::bail!(
                        "layer {l}: prod ran {h} kv heads and the plan is sharded for {nkv}"
                    );
                }
            }
        }
        let elem = kv_elem(cap, &kvk, hd)?;
        let src = cap.bytes(&kvk)?;
        let (k, v) = deinterleave(&src, nb, nkv as usize, kv_block as usize, hd, elem);
        let (kc, vc) = (pool.upload(ctx, stream, &k)?, pool.upload(ctx, stream, &v)?);
        let raw_bt = cap.i32_table(&format!("attn.l{l}.block_table"))?;
        let bt_stride = u32::try_from(
            cap.shape(&format!("attn.l{l}.block_table"))?
                .get(1)
                .copied()
                .unwrap_or(1),
        )
        .unwrap_or(1)
        .max(1);
        let seq_tab = cap.i32_table(&format!("attn.l{l}.seq_lens"))?;
        let seq_len = u32::try_from(seq_tab.iter().copied().max().unwrap_or(0)).unwrap_or(0);
        // EVERY PAGE THIS ATTENTION WILL READ HAS TO BE IN THE CAPTURE, and a renumber that
        // defaults to 0 cannot say so. The capture holds only the pages `propose` named, and a
        // page outside that set silently becomes page 0 of the replay's cache — a real page, in
        // range, belonging to another position. The kernel reads `ceil(seq_len / block_size)`
        // entries per sequence, so those are the ones that must map.
        let pages = (seq_len as usize)
            .div_ceil(kv_block.max(1) as usize)
            .max(1)
            .min(bt_stride as usize);
        for (s, _) in seq_tab.iter().enumerate() {
            for j in 0..pages {
                let b = raw_bt
                    .get(s * bt_stride as usize + j)
                    .copied()
                    .unwrap_or_default();
                if !renum.contains_key(&i64::from(b)) {
                    anyhow::bail!(
                        "layer {l}: the attention reads page {b} (sequence {s}, entry {j} of a \
                         {seq_len}-key sequence at block {kv_block}) and the capture carries only \
                         {blocks:?}"
                    );
                }
            }
        }
        let bt: Vec<u8> = raw_bt
            .iter()
            .flat_map(|b| {
                renum
                    .get(&i64::from(*b))
                    .copied()
                    .unwrap_or(0)
                    .to_le_bytes()
            })
            .collect();
        let binds = [
            ("q", pool.upload(ctx, stream, &q)?),
            ("kc", kc),
            ("vc", vc),
            ("block_table", pool.upload(ctx, stream, &bt)?),
            (
                "seq_lens",
                pool.upload(ctx, stream, &cap.bytes(&format!("attn.l{l}.seq_lens"))?)?,
            ),
            (
                "k_scale",
                pool.upload(ctx, stream, &cap.bytes(&format!("attn.l{l}.k_scale"))?)?,
            ),
            (
                "v_scale",
                pool.upload(ctx, stream, &cap.bytes(&format!("attn.l{l}.v_scale"))?)?,
            ),
            ("out", pool.zeroed(ctx, stream, want.len())?),
        ];
        // The KV dtype is the CACHE's, read off the manifest: one byte per element is the fp8
        // instantiation and two is the bf16 one. `AttnFp8Kv` is what the plan declares and a
        // capture that disagrees is a different serve, so it is named rather than defaulted.
        let op = if elem == 1 {
            DflashOp::AttnFp8Kv
        } else {
            DflashOp::AttnBf16Kv
        };
        let shape = DflashShape::attn(rows, nq, seq_len, nkv, kv_block, bt_stride);
        let sym = dispatch(ctx, launcher, stream, rules, op, shape, &binds)?;
        let got = pool.read(ctx, stream, binds[7].1, want.len())?;
        let d =
            got.iter().zip(&want).filter(|(a, b)| a != b).count() + got.len().abs_diff(want.len());
        differ += d;
        done += 1;
        println!(
            "   attn l{l} ({rows}x{nq}x{hd}, seq {seq_len})  {} of {} bytes{}",
            if d == 0 {
                "identical".to_string()
            } else {
                format!("DIFFERS in {d}")
            },
            want.len(),
            if explain {
                format!("   {sym} grid y = {}", nkv.div_ceil(rules.attn.block / 32))
            } else {
                String::new()
            }
        );
    }
    Ok((differ, done))
}

/// Run the staged proposal gate over every prompt in the capture.
///
/// # Errors
/// A missing capture file, a rules refusal, a pack that does not carry a record the plan declares,
/// or a HIP failure.
pub fn run(
    hip: &HipThread,
    args: &DraftStepArgs,
    cfg: &DflashConfig,
    graph: &DraftGraph,
    rules: &DflashRules,
    dir: &Path,
) -> anyhow::Result<usize> {
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("index.json"))?)?;
    let tags: Vec<String> = index["prompts"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("the capture's index.json has no `prompts`"))?
        .iter()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    let pack_dir = args.pack.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "a whole-proposal replay needs --pack: every launch after the context store reads a \
             drafter weight"
        )
    })?;
    println!("\n== the WHOLE proposal, against prod's own captured one");
    println!(
        "   capture   {} ({} prompt(s), decode step {}, tp {})",
        dir.display(),
        tags.len(),
        index["step"].as_u64().unwrap_or(0),
        index["tp"].as_u64().unwrap_or(0),
    );

    // WHICH DRAFTER THE CAPTURE IS OF, and it is not always the plan's.
    //
    // `ref_w4`'s load hook converts every drafter linear to 4 bits when the loaded reference library
    // carries a `w4a16 gemm_nt` kernel. The image's baked reference `.so` does; the build prod's unit
    // pins (`REF_KERNELS_SO=ref/kernel-builds/b9e42ab-rx6`) does not, and `serve-mxfp4.sh` copies
    // the latter over the former at container start. `prod_prelude.sh` omitted that copy until
    // 2026-09-21, so the round-21 capture is of a drafter whose `selector.hidden_projection` and
    // ten `kernel_projection`s are 4-bit and whose 21 blockscale GEMMs are too — a different
    // program from the one this plan describes, which declares them `bgemm_bf16` and
    // `gemm_a8w8_blockscale` because that is what the UNIT runs.
    //
    // Measured on the host with no card, from this capture's own tensors and the TP=1 pack:
    // prod's `walk.scores` are reproduced to 0 / 2 / 1 bytes of 7168 by a 4-bit
    // `hidden_projection` through `saber_gemm_w4a16_nt_m64`'s arithmetic, and to 3122 / 2768 / 2945
    // by the bf16 `F.linear` the plan declares. So the selector row of this gate is an
    // attribution and not a defect in `bgemm_bf16`, and it is SAID here rather than left for the
    // reader to infer from a red number.
    let w4 = index.get("drafter").and_then(|d| d.get("ref_w4"));
    match w4.and_then(serde_json::Value::as_bool) {
        Some(true) => println!(
            "   drafter   CONVERTED: this capture's drafter ran ref_w4's 4-bit weights, \
             which prod's unit does NOT (its pinned reference build has no w4a16 gemm). The selector's \
             projection rows below are that, not a kernel disagreement -- re-capture with \
             prod_prelude.sh's reference .so substitution"
        ),
        Some(false) => println!("   drafter   prod's own: ref_w4 disabled, block-fp8 + bf16"),
        None => println!(
            "   drafter   UNSTATED (a capture taken before 2026-09-21). Round 21's was taken \
             without prod_prelude.sh's reference .so substitution, so its drafter is ref_w4's \
             4-bit one and its selector rows are attributed rather than gating `bgemm_bf16`"
        ),
    }

    // The pack, uploaded once for every prompt. The reservation, the staging ring and the sink all
    // own device memory, so they stay in THIS scope: a helper that returned only the address map
    // would hand back addresses whose arena had just been freed.
    let ucfg = UploadConfig {
        mode: OpenMode::Direct,
        ..UploadConfig::default()
    };
    let upload = Upload::open(pack_dir, args.pack_rank, ucfg)?;
    let uplan = upload.plan();
    if uplan.arena_bytes > args.budget_mib * (1 << 20) {
        anyhow::bail!(
            "the drafter's arena is {} MiB and --budget-mib is {}",
            uplan.arena_bytes >> 20,
            args.budget_mib
        );
    }
    let mut ledger = Ledger::new();
    ledger.declare(Pool::Vram(args.pack_rank), uplan.arena_bytes.max(1))?;
    ledger.declare(Pool::HostPinned, uplan.staging_bytes.max(1))?;
    let reserved = upload.reserve(&mut ledger)?;
    let mut staging = hip.host_malloc(reserved.staging_bytes())?;
    let ring = PinnedRing::new(&mut staging, &reserved)?;
    let mut sink = HipSink::new(hip, &reserved)?;
    let mut addr: BTreeMap<String, DeviceAddr> = BTreeMap::new();
    let stats = upload.run(&mut sink, &ring, |r| {
        addr.insert(r.name.to_string(), r.addr);
    })?;
    println!(
        "   pack      {} records, {} MiB into VRAM, {:.2} GB/s",
        addr.len(),
        uplan.arena_bytes >> 20,
        stats.end_to_end_gbps()
    );

    let want = graph.records(DraftRope::qwen38_dflash2(args.rope_max_pos));
    let weights_of = weight_map(&want, &addr);

    let load = LoadCounters::new();
    let plugins = PluginSet::load_dir(hip, &args.prebuilt, &load)?;
    let functions = plugins.functions(hip)?;
    let ctx = LaunchCtx::new(hip, &functions);
    let counters = LaunchCounters::new();
    let mut launcher = Launcher::new("draft-step-proposal", &counters);
    let stream = hip.create_stream(StreamFlags::NonBlocking)?;

    let mut bad = 0usize;
    for tag in &tags {
        let cap = Capture::open(dir, tag)?;
        println!("\n-- {tag}");
        bad += compare(
            &ctx,
            &mut launcher,
            &stream,
            cfg,
            rules,
            &cap,
            &weights_of,
            args.explain,
        )?;
        // THE TIMED LAUNCH SET IS OPT-IN, and `--pack-launches` is the flag because it already
        // means "upload the rank and dispatch" rather than "check the records". It is a different
        // claim from the three staged comparisons above -- a time over the real shapes with the
        // real weights, not a bit-exactness -- and it costs a warm-up plus `--iters` passes over
        // the whole plan, so the comparisons do not carry it.
        if args.no_seed {
            let mut d = 0usize;
            let mut v = |name: &str, got: &[u8], want: &[u8]| -> usize {
                let n = got.iter().zip(want).filter(|(a, b)| a != b).count()
                    + got.len().abs_diff(want.len());
                d += n;
                println!(
                    "   {name:<26} {} of {} bytes",
                    if n == 0 {
                        "identical".to_string()
                    } else {
                        format!("DIFFERS in {n}")
                    },
                    want.len()
                );
                n
            };
            println!("   -- the CHAIN (--no-seed): nothing between the launches is prod's");
            if args.empty_draft_cache {
                println!(
                    "   -- CONTROL (--empty-draft-cache): the drafter's KV cache is ZEROS, not \n      \
                     prod's `kv.before_ctx` — the engine's own state at its first decode step"
                );
            }
            chain_context_store(
                &ctx,
                &mut launcher,
                &stream,
                cfg,
                rules,
                &cap,
                &weights_of,
                args.pack_rank,
                args.dump_chain.as_deref(),
                args.empty_draft_cache,
                &mut v,
            )?;
            bad += d;
        }
        if args.pack_launches {
            time_whole_proposal(
                &ctx,
                &mut launcher,
                &stream,
                cfg,
                graph,
                rules,
                &cap,
                &weights_of,
                args.iters.max(1),
            )?;
        } else {
            println!(
                "   (the staged comparisons only; --pack-launches adds the timed whole-proposal \
                 launch set)"
            );
        }
    }

    println!("\n== the producers the plan did not declare until the chain asked for them");
    for (name, why) in UNDECLARED_QUANTS {
        println!("   {name:<24} {why}");
    }
    println!(
        "   Both are `act_quant` now (six launches, one kernel at two shapes) and both are \n            compared above. The note stays because the WAY they were found is the durable part: a \n            staged replay seeds past a missing producer and only a chain notices."
    );
    Ok(bad)
}

/// The three staged comparisons.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn compare(
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    stream: &Stream,
    cfg: &DflashConfig,
    rules: &DflashRules,
    cap: &Capture,
    weights_of: &BTreeMap<&str, Vec<(&str, DeviceAddr)>>,
    explain: bool,
) -> anyhow::Result<usize> {
    let h = cfg.hidden;
    let steps = cap.int("num_speculative_steps")?;
    let top_k = cfg.selector_top_k;
    println!(
        "   step      {} target tokens, {} query rows/req, {} steps, kv block {}, head.hidden {}",
        cap.int("num_target_tokens")?,
        cap.int("num_query_per_req")?,
        steps,
        cap.int("kv_block_size")?,
        cap.dtype("head.hidden"),
    );

    let mut pool = Pool2::new();
    let mut differ = 0usize;
    let verdict = |name: &str, got: &[u8], want: &[u8], differ: &mut usize| -> usize {
        let d =
            got.iter().zip(want).filter(|(a, b)| a != b).count() + got.len().abs_diff(want.len());
        *differ += d;
        println!(
            "   {name:<26} {} of {} bytes",
            if d == 0 {
                "identical".to_string()
            } else {
                format!("DIFFERS in {d}")
            },
            want.len()
        );
        d
    };

    let run = |launcher: &mut Launcher,
               op: DflashOp,
               shape: DflashShape,
               binds: &[(&str, DeviceAddr)]|
     -> anyhow::Result<String> {
        dispatch(ctx, launcher, stream, rules, op, shape, binds)
    };

    // ---- 0. the five attention layers, at TP=1's eight kv heads ---------------------------------
    let (attn_bad, attn_layers) =
        compare_attention(ctx, launcher, stream, cfg, rules, cap, &mut pool, explain)?;
    differ += attn_bad;
    if attn_layers > 0 {
        println!(
            "     {attn_layers} of {} drafter attention layers against prod's own output, at {} \
             kv heads over {} waves (grid y = {})",
            cfg.layers,
            cfg.kv_heads_local(),
            rules.attn.block / 32,
            cfg.kv_heads_local().div_ceil(rules.attn.block / 32)
        );
    }

    // ---- 0b. the ELEVEN sites that were absent, each against prod's own intermediate ----------
    {
        let mut v = |name: &str, got: &[u8], want: &[u8]| -> usize {
            verdict(name, got, want, &mut differ)
        };
        compare_sites(
            ctx, launcher, stream, cfg, rules, cap, weights_of, &mut pool, &mut v,
        )?;
    }

    // ---- 1. the exact rerank, on prod's own candidate columns ----------------------------------
    //
    // `head.candidate_topk` is `Coverage::Absent` — ATen's radix select, whose index order among
    // ties is not a rule — so the candidate SET is handed in from the capture rather than
    // recomputed. That is what "the capture supplies the absent op" means here. What is asked is
    // the other half: given prod's own candidates, does `rerank_exact` reproduce the logit prod
    // scored them with? Every rounding in the coarse int2 pass is upstream and none of it survives
    // the rerank, which is what makes this an EXACT comparison rather than a bounded one.
    let hid = cap.bytes("head.hidden")?;
    let hid_shape = cap.shape("head.hidden")?;
    let rows = u32::try_from(hid_shape.first().copied().unwrap_or(0)).unwrap_or(0);
    let m_pad = rows.max(16).next_power_of_two();
    let mut x_pad = vec![0u8; (m_pad as usize) * (h as usize) * 2];
    let n = hid.len().min(x_pad.len());
    x_pad[..n].copy_from_slice(&hid[..n]);
    let x_addr = pool.upload(ctx, stream, &x_pad)?;

    let cand = cap.i64_table("head.candidate_ids")?;
    let mut idx32: Vec<i32> = vec![0; (m_pad as usize) * (top_k as usize)];
    for (i, c) in cand.iter().enumerate().take(idx32.len()) {
        idx32[i] = i32::try_from(*c).unwrap_or(0);
    }
    let idx_bytes: Vec<u8> = idx32.iter().flat_map(|v| v.to_le_bytes()).collect();
    let idx_addr = pool.upload(ctx, stream, &idx_bytes)?;

    let rr_shape = DflashShape::rerank(m_pad, cfg.vocab_local(), h, top_k);
    // The bf16 arm, because that is the head prod scores with: `SABER_NVFP4_LMHEAD=bf16`
    // dequantises `lm_head` at load, so `_head_matrix` returns a matrix with no scale and
    // `_rerank_exact` runs its `FP8=False` path. Measured on the host: the bf16 arm is
    // identical of 448 bytes on all three prompts, the e4m3 one differs in 6 / 8 / 9.
    // N6: or the same arithmetic over the fp8 record, when the pack carries `head.rerank.w8`.
    let rr_op = rerank_op(weights_of.get("head.rerank").map(Vec::as_slice));
    let rr = saber_kernels::dflash::resolve(rr_op, rr_shape, rules)?;
    let scatter = rr.kernel.contains("scatter");
    let out_len = if scatter {
        (m_pad as usize) * (cfg.vocab_local() as usize) * 2
    } else {
        (m_pad as usize) * (top_k as usize) * 4
    };
    let out_addr = pool.zeroed(ctx, stream, out_len)?;
    let ws = weights_of
        .get("head.rerank")
        .ok_or_else(|| anyhow::anyhow!("the pack carries no head.rerank records"))?;
    let mut binds: Vec<(&str, DeviceAddr)> =
        vec![("x", x_addr), ("idx", idx_addr), ("out", out_addr)];
    binds.extend(ws.iter().copied());
    let sym = run(launcher, rr_op, rr_shape, &binds)?;
    if explain {
        println!("   rerank    {sym} (r = {top_k}, prod's own candidate columns)");
    }
    let got = pool.read(ctx, stream, out_addr, out_len)?;
    let unary = cap.bytes("head.unary_logits")?;
    let mut ours = Vec::with_capacity(unary.len());
    for (i, c) in cand.iter().enumerate().take(unary.len() / 4) {
        let v = if scatter {
            let o = (i / top_k as usize) * cfg.vocab_local() as usize
                + usize::try_from(*c).unwrap_or(0);
            if o * 2 + 2 <= got.len() {
                bf16_to_f32(u16::from_le_bytes([got[o * 2], got[o * 2 + 1]]))
            } else {
                f32::NAN
            }
        } else {
            let f = f32::from_le_bytes(got[i * 4..i * 4 + 4].try_into().unwrap_or_default());
            bf16_to_f32(f32_to_bf16_rne(f))
        };
        ours.extend_from_slice(&v.to_le_bytes());
    }
    let rr_diff = verdict("head.rerank -> unary", &ours, &unary, &mut differ);
    if rr_diff > 0 {
        // Read this before reading it as a kernel bug. `head.rerank.w` is the TARGET checkpoint's
        // `lm_head.weight` as stored -- `e4m3` with a per-row scale -- and `dflash::resolve` picks
        // the fp8 variant for it. What prod reranks against is whatever tensor is bound at first
        // inference, and `SABER_NVFP4_LMHEAD=bf16` (which the unit sets, and which the capture
        // probe reports) dequantises that head to `bf16` at load, so `ref_drafthead`'s
        // `_head_matrix` returns a bf16 matrix with no scale and `_rerank_exact` runs its
        // `FP8=False` arm. Different bytes, different arithmetic, same numbers to about a ULP.
        println!(
            "     ^ the pack's head.rerank.w is the checkpoint's e4m3 + per-row scale; prod's \
             LIVE lm_head is bf16 (SABER_NVFP4_LMHEAD), so this compares two arithmetics and \
             not two implementations of one. The record map is what has to change, not the kernel"
        );
    }

    // ---- 2. the selector, from prod's own candidates -------------------------------------------
    //
    // THE SELECTOR'S FOUR INPUTS DO NOT COME FROM A HOOK ON IT. `CandidateSelector` carries
    // `@support_torch_compile`, which installs its own `__call__` on the class, so
    // `nn.Module._call_impl` never runs and a forward hook registered on that module never fires
    // -- it reports itself installed and writes nothing (GATES.md blind spot 6, in a second
    // place). Three of the four are already captured where they are ordinary Python values, at
    // `compute_candidates` and at `_sample_path`, and the fourth is the ANCHOR:
    // `_generate_draft` forms it as `input_buffers.input_ids[_anchor_indices[:num_reqs]]`, and
    // `_anchor_indices` is `arange(max_num_reqs) * num_query_per_req` -- so it is element
    // `req * num_query_per_req` of the captured query ids, derived here when the capture does not
    // carry it directly.
    let sel_hidden = pool.upload(ctx, stream, &cap.bytes("head.hidden")?)?;
    let sel_cand = pool.upload(ctx, stream, &cap.bytes("walk.candidate_ids")?)?;
    let sel_unary = pool.upload(ctx, stream, &cap.bytes("head.unary_logits")?)?;
    let anchor_bytes = match cap.bytes("sel.anchor_token_ids") {
        Ok(b) => b,
        Err(_) => {
            let ids = cap.bytes("q.input_ids")?;
            let q = cap.int("num_query_per_req")? as usize;
            let reqs = cap.int("num_reqs")? as usize;
            let w = ids.len()
                / cap
                    .shape("q.input_ids")?
                    .first()
                    .copied()
                    .unwrap_or(1)
                    .max(1);
            let mut out = Vec::with_capacity(reqs * w);
            for r in 0..reqs {
                let o = r * q * w;
                out.extend_from_slice(&ids[o..o + w]);
            }
            out
        }
    };
    let sel_anchor = pool.upload(ctx, stream, &anchor_bytes)?;
    let proj = pool.zeroed(
        ctx,
        stream,
        (steps as usize) * (cfg.selector_rank as usize) * 2,
    )?;
    let pw = weights_of
        .get("selector.hidden_projection")
        .ok_or_else(|| anyhow::anyhow!("the pack carries no selector.hidden_projection.w"))?;
    let mut pb: Vec<(&str, DeviceAddr)> = vec![("a", sel_hidden), ("c", proj)];
    pb.extend(pw.iter().copied());
    run(
        launcher,
        DflashOp::BgemmBf16,
        DflashShape::new(steps, cfg.selector_rank, h),
        &pb,
    )?;

    let sc_ref = cap.bytes("walk.scores")?;
    let scores = pool.zeroed(ctx, stream, sc_ref.len())?;
    let cb = weights_of
        .get("selector.score_edges")
        .ok_or_else(|| anyhow::anyhow!("the pack carries no selector codebooks"))?;
    let mut sb: Vec<(&str, DeviceAddr)> = vec![
        ("cand", sel_cand),
        ("anchor", sel_anchor),
        ("unary", sel_unary),
        ("hidden", proj),
        ("out", scores),
    ];
    sb.extend(cb.iter().copied());
    run(
        launcher,
        DflashOp::SelectorScore,
        DflashShape::selector(steps, steps, cfg.selector_rank, top_k),
        &sb,
    )?;
    let got_sc = pool.read(ctx, stream, scores, sc_ref.len())?;
    verdict("selector.score_edges", &got_sc, &sc_ref, &mut differ);

    // ---- 3. the walk, chained on OUR scores ----------------------------------------------------
    let req_state = pool.upload(ctx, stream, &cap.bytes("walk.idx_mapping")?)?;
    let tok_ref = cap.bytes("walk.draft_tokens")?;
    let real_ref = cap.bytes("walk.realized_scores")?;
    let tokens = pool.zeroed(ctx, stream, tok_ref.len())?;
    let realized = pool.zeroed(ctx, stream, real_ref.len())?;
    run(
        launcher,
        DflashOp::SelectorWalk,
        DflashShape::rerank(1, steps, top_k, top_k),
        &[
            ("scores", scores),
            ("candidates", sel_cand),
            ("req_state", req_state),
            ("tokens", tokens),
            ("realized", realized),
        ],
    )?;
    let got_real = pool.read(ctx, stream, realized, real_ref.len())?;
    let got_tok = pool.read(ctx, stream, tokens, tok_ref.len())?;
    verdict("selector.walk realized", &got_real, &real_ref, &mut differ);
    let tok_diff = verdict("draft block (our scores)", &got_tok, &tok_ref, &mut differ);

    // ---- 3b. the walk ALONE, on prod's own score tensor -----------------------------------------
    //
    // The chained arm above puts the selector and the walk in one verdict, which is the right
    // claim when the selector is right and useless when it is not: a walk that reads different
    // scores cannot produce the same block and the reading says nothing about the walk. So the
    // same kernel runs again on PROD's score tensor, which is the only arm of this gate that is a
    // statement about the walk by itself. Two arms of one kernel, and the pair is the diagnosis:
    // if this one matches and the chained one does not, the disagreement is upstream.
    let prod_scores = pool.upload(ctx, stream, &sc_ref)?;
    let tokens2 = pool.zeroed(ctx, stream, tok_ref.len())?;
    let realized2 = pool.zeroed(ctx, stream, real_ref.len())?;
    run(
        launcher,
        DflashOp::SelectorWalk,
        DflashShape::rerank(1, steps, top_k, top_k),
        &[
            ("scores", prod_scores),
            ("candidates", sel_cand),
            ("req_state", req_state),
            ("tokens", tokens2),
            ("realized", realized2),
        ],
    )?;
    let got_tok2 = pool.read(ctx, stream, tokens2, tok_ref.len())?;
    let got_real2 = pool.read(ctx, stream, realized2, real_ref.len())?;
    verdict(
        "selector.walk realized (prod's scores)",
        &got_real2,
        &real_ref,
        &mut differ,
    );
    let tok_diff2 = verdict(
        "THE k=7 DRAFT BLOCK (prod's scores)",
        &got_tok2,
        &tok_ref,
        &mut differ,
    );
    let as_ids = |b: &[u8]| -> Vec<i64> {
        b.chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap_or_default()))
            .collect()
    };
    println!("     prod                {:?}", as_ids(&tok_ref));
    println!("     saber, our scores   {:?}", as_ids(&got_tok));
    println!("     saber, prod's scores{:?}", as_ids(&got_tok2));
    println!(
        "     the k={steps} tokens are {} from Saber's own scores and {} from prod's",
        if tok_diff == 0 {
            "IDENTICAL"
        } else {
            "NOT identical"
        },
        if tok_diff2 == 0 {
            "IDENTICAL"
        } else {
            "NOT identical"
        }
    );
    Ok(differ)
}

/// **The context store as a CHAIN**, from the target's own hidden states to the drafter's KV
/// cache, with nothing seeded in between.
///
/// Ten launches — the activation quant, the aux projection, the hidden norm, the fused KV
/// projection, the layout copy, the grouped K norm, the batched rope and five per-layer cache
/// inserts — every one of them reading what the previous one wrote. The only inputs are prod's
/// own: `aux.cat` (the five tapped target layers, concatenated), `ctx.positions`, the per-layer
/// slot tables, and `kv.before_ctx` — **the cache as it stood BEFORE the store ran**. That last
/// one is why the r26 capture exists at all: seeding from `kv.after_ctx` hands the replay the
/// answer, and a gate that is handed the answer is not a gate.
///
/// Three of these ten were `Coverage::Absent` until this round, and this is the claim the staged
/// comparison could not make about them. A staged gate seeds each op from prod's tensor at its
/// input, so a producer that is subtly wrong is invisible to the op after it; here the quant's
/// e4m3 bytes and its per-128-group scales go straight into the blockscale GEMM, whose output is
/// normed, projected, permuted, normed again, rotated, and finally quantised into the cache.
/// **`kv.after_ctx` is the only thing compared**, so every one of the ten has to be right for it
/// to be identical.
///
/// It DOES chain the drafter's five-layer backbone as well, since F27: embedding -> five layers
/// -> the sample gather -> the head -> the selector -> the walk, from `q.input_ids` and the
/// caches the store itself wrote. `head::embedding` is resolved here out of the head plugin's
/// rules; the q/k norm + rope is `dflash_qk`, the drafter's own kernel, because `attn_glue`'s is
/// the TARGET's arithmetic (see the dispatch below).
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn chain_context_store(
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    stream: &Stream,
    cfg: &DflashConfig,
    rules: &DflashRules,
    cap: &Capture,
    weights_of: &BTreeMap<&str, Vec<(&str, DeviceAddr)>>,
    rank: u32,
    dump: Option<&Path>,
    // ws-s8's control: the drafter's KV cache as ZEROS rather than prod's. See
    // `DraftStepArgs::empty_draft_cache` — it is the engine's own cache at its first decode step,
    // because prod stores the drafter's context K/V at every target step and Saber's engine
    // stores it only at a decode step.
    empty_cache: bool,
    verdict: &mut Verdict<'_>,
) -> anyhow::Result<()> {
    // THE CHAIN'S OWN INTERMEDIATES, under the ENGINE's arena-region names (ws-s7).
    //
    // Two implementations of one program are only worth having if something compares them, and
    // prod's taps cannot do it: the body's `chain.*` rows are an eager SHADOW (F30) and the
    // region calls are anchored but noisy at the ULP. Engine against chain is neither — it is two
    // runs of the same kernels over the same bytes, so the only honest verdict is IDENTICAL, and
    // the first name in this list that is not identical is the operand the engine's plumbing
    // departs at. The names are `saber_engine::draft_plan::DraftPlan::arena_for`'s, so the reader
    // is `arena:<name>` and needs no map.
    let mut dumps: Vec<(String, DeviceAddr, usize)> = Vec::new();
    let h = cfg.hidden;
    let hd = cfg.head_dim;
    let nkv = cfg.kv_heads_local();
    let num_ctx = cap.int("num_target_tokens")?;
    let kv_block = cap.int("kv_block_size")?;
    let mut pool = Pool2::new();

    let w = |op: &str, slot: &str| -> anyhow::Result<DeviceAddr> {
        weights_of
            .get(op)
            .and_then(|v| v.iter().find(|(nm, _)| *nm == slot))
            .map(|(_, a)| *a)
            .ok_or_else(|| anyhow::anyhow!("the pack carries no `{op}.{slot}`"))
    };

    // ---- the inputs, and nothing else ----------------------------------------------------------
    let aux = pool.upload(ctx, stream, &cap.bytes("aux.cat")?)?;
    let (blocks, renum) = captured_blocks(cap);
    let nb = blocks.len().max(1);
    let elem = kv_elem(cap, "kv.before_ctx.l0", hd as usize)?;
    let cache_bytes = nb * kv_block as usize * nkv as usize * hd as usize * elem;
    let mut caches: Vec<(DeviceAddr, DeviceAddr)> = Vec::new();
    for l in 0..cfg.layers {
        let mut src = cap.bytes(&format!("kv.before_ctx.l{l}"))?;
        if empty_cache {
            src.iter_mut().for_each(|b| *b = 0);
        }
        let (k, v) = deinterleave(&src, nb, nkv as usize, kv_block as usize, hd as usize, elem);
        caches.push((pool.upload(ctx, stream, &k)?, pool.upload(ctx, stream, &v)?));
    }
    let positions = {
        let p = cap.bytes("ctx.positions")?;
        let mut rep = Vec::with_capacity(p.len() * cfg.layers as usize);
        for _ in 0..cfg.layers.max(1) {
            rep.extend_from_slice(&p);
        }
        pool.upload(ctx, stream, &rep)?
    };
    let mut ctx_slots: Vec<DeviceAddr> = Vec::new();
    for l in 0..cfg.layers {
        let s: Vec<i64> = cap
            .i64_table(&format!("ctx.slots.l{l}"))
            .or_else(|_| cap.i64_table("ctx.slots"))?
            .iter()
            .map(|s| {
                let b = s / i64::from(kv_block);
                let o = s % i64::from(kv_block);
                i64::from(renum.get(&b).copied().unwrap_or(0)) * i64::from(kv_block) + o
            })
            .collect();
        let b: Vec<u8> = s.iter().flat_map(|v| v.to_le_bytes()).collect();
        ctx_slots.push(pool.upload(ctx, stream, &b)?);
    }
    let one_f32 = pool.upload(ctx, stream, &1.0f32.to_le_bytes().repeat(nkv as usize))?;

    // ---- the ten launches ------------------------------------------------------------------------
    let aux_k = cfg.target_layers * h;
    let aq = pool.zeroed(ctx, stream, (num_ctx * aux_k) as usize)?;
    let aqs = pool.zeroed(ctx, stream, (num_ctx * aux_k.div_ceil(128) * 4) as usize)?;
    let states = pool.zeroed(ctx, stream, (num_ctx * h * 2) as usize)?;
    let normed = pool.zeroed(ctx, stream, (num_ctx * h * 2) as usize)?;
    let kv_n = cfg.context_kv_n();
    let kv_flat = pool.zeroed(ctx, stream, (num_ctx * kv_n * 2) as usize)?;
    let half = (cfg.layers * num_ctx * nkv * hd * 2) as usize;
    let all_kv = pool.zeroed(ctx, stream, half * 2)?;
    let all_k_normed = pool.zeroed(ctx, stream, half)?;

    dispatch(
        ctx,
        launcher,
        stream,
        rules,
        DflashOp::ActQuant,
        DflashShape::new(num_ctx, aux_k, aux_k),
        &[("x", aux), ("q", aq), ("scales", aqs)],
    )?;
    dispatch(
        ctx,
        launcher,
        stream,
        rules,
        DflashOp::GemmBlockscale,
        DflashShape::new(num_ctx, h, aux_k),
        &[
            ("a", aq),
            ("as", aqs),
            ("w", w("aux_hidden_projection", "w")?),
            ("ws", w("aux_hidden_projection", "ws")?),
            ("c", states),
        ],
    )?;
    dispatch(
        ctx,
        launcher,
        stream,
        rules,
        DflashOp::CtxRmsNorm,
        DflashShape::ctx_norm(num_ctx, h, num_ctx),
        &[
            ("x", states),
            ("weight", w("context_kv.hidden_norm", "weight")?),
            ("out", normed),
        ],
    )?;
    dispatch(
        ctx,
        launcher,
        stream,
        rules,
        DflashOp::BgemmBf16,
        DflashShape::new(num_ctx, kv_n, h),
        &[
            ("a", normed),
            ("w", w("context_kv.kv_projection", "w")?),
            ("c", kv_flat),
        ],
    )?;
    dispatch(
        ctx,
        launcher,
        stream,
        rules,
        DflashOp::CtxKvLayout,
        DflashShape::ctx_layout(num_ctx, cfg.layers, nkv, hd),
        &[("in", kv_flat), ("out", all_kv)],
    )?;
    // `all_k` is `all_kv[0]` and `all_v` is `all_kv[1]` — the two halves of one contiguous
    // `[2, L, num_ctx, nkv, hd]`, which is exactly what the layout copy wrote.
    let off = |a: DeviceAddr, b: u64| a.offset(b).expect("a device address past its own arena");
    let all_v = off(all_kv, half as u64);
    dispatch(
        ctx,
        launcher,
        stream,
        rules,
        DflashOp::CtxRmsNorm,
        DflashShape::ctx_norm(cfg.layers * num_ctx * nkv, hd, num_ctx * nkv),
        &[
            ("x", all_kv),
            ("weight", w("context_kv.k_norm", "weight")?),
            ("out", all_k_normed),
        ],
    )?;
    dispatch(
        ctx,
        launcher,
        stream,
        rules,
        DflashOp::CtxRope,
        DflashShape::ctx_rope(cfg.layers * num_ctx, nkv, hd, hd),
        &[
            ("positions", positions),
            ("k", all_k_normed),
            ("cos_sin", w("context_kv.rope", "cos_sin")?),
        ],
    )?;
    let per_layer = (num_ctx * nkv * hd * 2) as u64;
    for l in 0..cfg.layers {
        let mut shape = DflashShape::new(num_ctx, nkv, hd);
        shape.block_size = kv_block;
        dispatch(
            ctx,
            launcher,
            stream,
            rules,
            DflashOp::KvAppendFp8,
            shape,
            &[
                ("key", off(all_k_normed, u64::from(l) * per_layer)),
                ("value", off(all_v, u64::from(l) * per_layer)),
                ("kc", caches[l as usize].0),
                ("vc", caches[l as usize].1),
                ("slots", ctx_slots[l as usize]),
                ("scales", one_f32),
            ],
        )?;
    }
    ctx.hip.sync_stream(stream)?;

    // ---- the only comparison: prod's own cache after the store -------------------------------
    for l in 0..cfg.layers {
        let want_inter = cap.bytes(&format!("kv.after_ctx.l{l}"))?;
        let (wk, wv) = deinterleave(
            &want_inter,
            nb,
            nkv as usize,
            kv_block as usize,
            hd as usize,
            elem,
        );
        let gk = pool.read(ctx, stream, caches[l as usize].0, cache_bytes)?;
        let gv = pool.read(ctx, stream, caches[l as usize].1, cache_bytes)?;
        verdict(&format!("chain.ctx.kv.l{l}.k"), &gk, &wk);
        verdict(&format!("chain.ctx.kv.l{l}.v"), &gv, &wv);
    }
    println!(
        "     ten launches chained from `aux.cat` and `kv.before_ctx`: quant -> aux GEMM -> \n     \
         hidden norm -> KV projection -> layout -> K norm -> rope -> five cache inserts. \n     \
         Nothing between them is seeded, so a difference anywhere lands in the cache."
    );

    for (name, a, n) in [
        ("ctx.aux.q", aq, (num_ctx * aux_k) as usize),
        (
            "ctx.aux.scales",
            aqs,
            (num_ctx * aux_k.div_ceil(128) * 4) as usize,
        ),
        ("ctx.states", states, (num_ctx * h * 2) as usize),
        ("ctx.normed", normed, (num_ctx * h * 2) as usize),
        ("ctx.kv_flat", kv_flat, (num_ctx * kv_n * 2) as usize),
        ("ctx.all_kv", all_kv, half * 2),
        ("ctx.all_k_normed", all_k_normed, half),
    ] {
        dumps.push((name.to_owned(), a, n));
    }

    // ---- and the BACKBONE, from the query ids into the same caches the store just wrote --------
    //
    // This is the half a staged replay cannot reach at all: every stage of it starts from a
    // tensor the capture supplies, so a missing producer is seeded past rather than noticed. Five
    // launches were found by writing THIS loop, all of them in the seam between the drafter's
    // forward and its head, and each one is a `DraftOp` now (`dflash_graph.rs`): two row gathers,
    // `_pow2_at_least`'s row pad, `xs`, and the `bi` gather.
    //
    // The KV caches are the SAME device buffers the context store above wrote, which is the
    // point: the attention reads what the store put there and not what prod's `kv.after_ctx`
    // says should be there.
    let head_rules = HeadRules::prod_head()?;
    let mrows = u32::try_from(cap.shape("q.input_ids")?.first().copied().unwrap_or(8)).unwrap_or(8);
    let steps = cap.int("num_speculative_steps")?;
    let top_k = cfg.selector_top_k;
    let nq = cfg.q_heads_local();
    let qkv_n = cfg.qkv_n();
    let kp_n = cfg.kernel_projection_n();
    let inter = cfg.intermediate / cfg.tp.max(1);
    let vloc = cfg.vocab_local();
    let groups = h / cfg.conv_group;
    let qn = nq * hd;
    // THE ATTENTION'S SEQUENCE LENGTH COMES FROM THE SLOT TABLE, not from `idx.seq_lens`.
    //
    // `idx.seq_lens` is the scheduler's own per-request length and it is NOT where the drafter's
    // query rows land: on p86 it reads 282 while prod's own attention ran at 286 and the first
    // query slot sits at offset 278 of its block. 278 + 8 IS 286, and the same identity holds on
    // p54 (262 + 8 = 270) and p11 (265 + 8 = 273). The slot table is the tensor the scheduler
    // hands the append, so deriving the length from it keeps the chain unseeded AND right; taking
    // `idx.seq_lens + mrows` made the layer-0 attention read four positions of stale cache on one
    // prompt of three, which is what `chain.attn.l0.out` was (62,386 of 65,536 with q, k and v
    // all bit-identical to prod's own).
    //
    // Checked against prod's own tap below rather than trusted: `attn.l0.seq_lens` is in the
    // capture and a disagreement is printed, because a chain that derives a table wrongly and
    // agrees with itself is exactly what this row exists to catch.
    let slot0 = cap
        .i64_table("idx.query_slots")
        .ok()
        .and_then(|v| v.first().copied());
    let seq_len = match slot0 {
        Some(s) => u32::try_from(s % i64::from(kv_block)).unwrap_or(0) + mrows,
        None => {
            cap.i32_table("idx.seq_lens")
                .ok()
                .and_then(|v| v.first().copied())
                .map_or(4096, |v| u32::try_from(v).unwrap_or(4096))
                + mrows
        }
    };
    if let Ok(want) = cap.i32_table("attn.l0.seq_lens") {
        if let Some(w) = want.first().copied() {
            let w = u32::try_from(w).unwrap_or(0);
            if w == seq_len {
                println!("   chain.attn.seq_len          {seq_len}, and prod's own tap agrees");
            } else {
                println!(
                    "   chain.attn.seq_len          {seq_len} from the slot table, but prod's \
                     own `attn.l0.seq_lens` is {w} -- the attention below is reading a different \
                     window from the one prod read"
                );
            }
        }
    }
    let bt_stride = u32::try_from(cap.shape("idx.block_table")?.get(1).copied().unwrap_or(1))
        .unwrap_or(1)
        .max(1);
    let bt: Vec<u8> = cap
        .i32_table("idx.block_table")
        .unwrap_or_default()
        .iter()
        .flat_map(|b| {
            renum
                .get(&i64::from(*b))
                .copied()
                .unwrap_or(0)
                .to_le_bytes()
        })
        .collect();
    let block_table = pool.upload(ctx, stream, if bt.is_empty() { &[0u8; 4] } else { &bt })?;
    let seq_lens = pool.upload(
        ctx,
        stream,
        &i32::try_from(seq_len).unwrap_or(0).to_le_bytes(),
    )?;
    let q_slots: Vec<u8> = cap
        .i64_table("idx.query_slots")
        .unwrap_or_default()
        .iter()
        .flat_map(|s| {
            let b = s / i64::from(kv_block);
            let o = s % i64::from(kv_block);
            (i64::from(renum.get(&b).copied().unwrap_or(0)) * i64::from(kv_block) + o).to_le_bytes()
        })
        .collect();
    let query_slots = pool.upload(ctx, stream, &q_slots)?;
    let ids = pool.upload(ctx, stream, &cap.bytes("q.input_ids")?)?;
    // THE ROPE'S POSITION ROW, singular. `attn_glue`'s kernel reads a `[3, M]` mrope table and
    // wanted the row replicated three times (its section map reaches rows 1 and 2 at
    // `rot_dim = 128` even for a plain rope); `dflash_qk` takes the drafter's own `[M]` i64
    // positions, which is the tensor prod hands its rope.
    let positions_q = pool.upload(ctx, stream, &cap.bytes("q.positions")?)?;
    let one_f32_1 = pool.upload(ctx, stream, &1.0f32.to_le_bytes())?;

    let wpos = |op: &str, i: usize| -> anyhow::Result<DeviceAddr> {
        weights_of
            .get(op)
            .and_then(|v| v.get(i))
            .map(|(_, a)| *a)
            .ok_or_else(|| anyhow::anyhow!("the pack carries no `{op}` record #{i}"))
    };

    let hidden = pool.zeroed(ctx, stream, (mrows * h * 2) as usize)?;
    let mut normed = pool.zeroed(ctx, stream, (mrows * h * 2) as usize)?;
    let mut resid = hidden;

    // THE BACKBONE'S BISECT ROWS, and what makes them trustworthy is a number the CAPTURE carries.
    //
    // Every tensor between `q.input_ids` and `attn.l0.q` is an intermediate of an inductor-compiled
    // piece, so the r28 capture takes them from an eager SHADOW of prod's own modules and then
    // CHECKS that shadow where the live step can be seen: its post-rope q and k against
    // `attn.l{i}.q` / `attn.l{i}.k`, which the attention impl tapped from prod's own launch. A
    // layer whose `chain.l{i}.selfcheck.q` is 0 was reached through prod's own arithmetic at every
    // step before it -- a wrong qkv, a wrong norm weight or a wrong position cannot produce the
    // right post-rope q -- so its intermediates are prod's and a difference against them COUNTS.
    // A layer whose self-check is not 0 is a reference for the STRUCTURE and prints without
    // counting, with its own drift beside it, because eager and inductor are not the same program
    // (the LEDGER's `silu_quant` entry, from the other side).
    let exact_shadow = |l: u32| -> bool {
        cap.int(&format!("chain.l{l}.selfcheck.q")).unwrap_or(1) == 0
            && cap.int(&format!("chain.l{l}.selfcheck.k")).unwrap_or(1) == 0
    };
    // F32: the COMPILED REGION's own operands, call by call. Every op of the body has an anchor
    // in prod's own arithmetic now, and these rows are where the chain says WHICH op parts company
    // with it. They are NOTES: each one is reached through the ops before it, so only the FIRST
    // one that fires names an op rather than an inheritance.
    let rmap = cap.region_map();
    if rmap.is_empty() {
        println!(
            "   (this capture carries no `region.*` taps, so the body's ops are unanchored -- \
             that is r28 and earlier; re-take it with the internal reference harness)"
        );
    }
    // Set at the END of each layer's body and read at the TOP of the next one, where
    // `chain.l{l}.v` is compared: the body that produced that tensor is the previous layer's.
    let mut body_class = false;

    // The printing half of that rule: a row against a shadow this capture does NOT vouch for,
    // shown with the shadow's own drift beside it so the number can be read for what it is.
    let note_chain = |name: &str, got: &[u8], want: &[u8]| {
        let d =
            got.iter().zip(want).filter(|(a, b)| a != b).count() + got.len().abs_diff(want.len());
        println!(
            "   {name:<26} {} of {} bytes -- against an eager shadow this capture does not vouch \
             for (see `chain.l*.selfcheck.*`); NOT counted",
            if d == 0 {
                "identical".to_string()
            } else {
                format!("DIFFERS in {d}")
            },
            want.len()
        );
    };

    // 1. the embedding. `head::embedding` is the vocab-parallel lookup the plan already names;
    // at TP=1 the window is the whole vocabulary and the zero-row rule never fires.
    {
        let d = head::resolve(
            HeadOp::Embedding,
            HeadShape {
                rows: mrows,
                cols: h,
                src_stride: i64::from(h),
                vocab: (rank * vloc, (rank + 1) * vloc),
            },
            &head_rules,
        )?;
        let mut b = ArgBindings::new(Shape::new(mrows, h, 0));
        b.bind("table", w("input_embedding", "weight")?);
        b.bind("ids", ids);
        b.bind("out", hidden);
        launcher.dispatch_one(ctx, &d, &b, stream)?;
    }
    // The embedding is a GATHER: exact in eager and compiled alike, so this row counts whatever
    // the shadow's self-check says. It is also where the mask substitution would be if this
    // checkpoint shipped one (`chain.mask_embedding` in the manifest says whether it does).
    if let Ok(want) = cap.bytes("chain.embed") {
        ctx.hip.sync_stream(stream)?;
        verdict(
            "chain.embed",
            &pool.read(ctx, stream, hidden, want.len())?,
            &want,
        );
    }
    dispatch(
        ctx,
        launcher,
        stream,
        rules,
        DflashOp::RmsNorm,
        DflashShape::new(mrows, h, h),
        &[
            ("x", hidden),
            ("weight", w("layer.0.input_layernorm", "weight")?),
            ("out", normed),
        ],
    )?;
    dumps.push(("bb.hidden".to_owned(), hidden, (mrows * h * 2) as usize));
    dumps.push(("bb.normed0".to_owned(), normed, (mrows * h * 2) as usize));
    if let Ok(want) = cap.bytes("chain.l0.input_norm") {
        ctx.hip.sync_stream(stream)?;
        let got = pool.read(ctx, stream, normed, want.len())?;
        if exact_shadow(0) {
            verdict("chain.l0.input_norm", &got, &want);
        } else {
            note_chain("chain.l0.input_norm", &got, &want);
        }
    }

    for l in 0..cfg.layers {
        let p = |t: &str| format!("layer.{l}.{t}");
        // The residual STREAM at the top of layer `l`. Layer 0's is the embedding; 1..4's are the
        // only statement of what the previous layer produced, and a chain that drifts drifts here
        // first.
        if let Ok(want) = cap.bytes(&format!("chain.l{l}.resid")) {
            ctx.hip.sync_stream(stream)?;
            let got = pool.read(ctx, stream, resid, want.len())?;
            if exact_shadow(l) {
                verdict(&format!("chain.l{l}.resid"), &got, &want);
            } else {
                note_chain(&format!("chain.l{l}.resid"), &got, &want);
            }
        }
        let coeff_a = pool.zeroed(ctx, stream, (mrows * kp_n * 2) as usize)?;
        let coeff_m = pool.zeroed(ctx, stream, (mrows * kp_n * 2) as usize)?;
        let aq = pool.zeroed(ctx, stream, (mrows * h) as usize)?;
        let as_ = pool.zeroed(ctx, stream, (mrows * h.div_ceil(128) * 4) as usize)?;
        let qkv = pool.zeroed(ctx, stream, (mrows * qkv_n * 2) as usize)?;
        let q_rot = pool.zeroed(ctx, stream, (mrows * qn * 2) as usize)?;
        let k_rot = pool.zeroed(ctx, stream, (mrows * nkv * hd * 2) as usize)?;
        let attn_out = pool.zeroed(ctx, stream, (mrows * qn * 2) as usize)?;
        let oq = pool.zeroed(ctx, stream, (mrows * qn) as usize)?;
        let os = pool.zeroed(ctx, stream, (mrows * qn.div_ceil(128) * 4) as usize)?;
        let o = pool.zeroed(ctx, stream, (mrows * h * 2) as usize)?;
        let normed2 = pool.zeroed(ctx, stream, (mrows * h * 2) as usize)?;
        // FOUR BYTES A CHANNEL, and it is prod's own buffer rather than a precision choice:
        // the drafter's compiled region allocates the residual between its two `conv_add_rms`
        // halves as fp32 (`buf6 = empty_strided_cuda((s72, 5120), (5120, 1), torch.float32)`) and
        // hands it to the second as `in_out_ptr0`. Only `res_out` below -- the tensor that
        // crosses the eager attention seam into the next layer -- is bf16. Saber rounded at both
        // until F29, which is one rounding a layer more than prod.
        let resid2 = pool.zeroed(ctx, stream, (mrows * h * 4) as usize)?;
        let mq = pool.zeroed(ctx, stream, (mrows * h) as usize)?;
        let ms = pool.zeroed(ctx, stream, (mrows * h.div_ceil(128) * 4) as usize)?;
        let gate_up = pool.zeroed(ctx, stream, (mrows * 2 * inter * 2) as usize)?;
        let dq = pool.zeroed(ctx, stream, (mrows * inter) as usize)?;
        let ds = pool.zeroed(ctx, stream, (mrows * inter.div_ceil(128) * 4) as usize)?;
        let down = pool.zeroed(ctx, stream, (mrows * h * 2) as usize)?;
        let out_norm = pool.zeroed(ctx, stream, (mrows * h * 2) as usize)?;
        let res_out = pool.zeroed(ctx, stream, (mrows * h * 2) as usize)?;

        dispatch(
            ctx,
            launcher,
            stream,
            rules,
            DflashOp::BgemmBf16,
            DflashShape::new(mrows, kp_n, h),
            &[
                ("a", normed),
                ("w", w(&p("attention_conv.kernel_projection"), "w")?),
                ("c", coeff_a),
            ],
        )?;
        dispatch(
            ctx,
            launcher,
            stream,
            rules,
            DflashOp::GroupedConvQuant,
            DflashShape::conv(mrows, h, groups, cfg.block_size, 0),
            &[
                ("x", normed),
                ("coeff", coeff_a),
                ("base", w(&p("attention_conv.prepare"), "base")?),
                ("q", aq),
                ("scales", as_),
            ],
        )?;
        dispatch(
            ctx,
            launcher,
            stream,
            rules,
            DflashOp::GemmBlockscale,
            DflashShape::new(mrows, qkv_n, h),
            &[
                ("a", aq),
                ("as", as_),
                ("w", w(&p("attn.qkv"), "w")?),
                ("ws", w(&p("attn.qkv"), "ws")?),
                ("c", qkv),
            ],
        )?;
        // The projection, BEFORE the norm and the rope: the tap that splits this prefix in two.
        // If `qkv` is identical and `q` is not, the defect is the norm or the rope; if `qkv` is
        // not, it is the embedding, the norm, the kernel projection, the conv or the quant.
        if let Ok(want) = cap.bytes(&format!("chain.l{l}.qkv")) {
            ctx.hip.sync_stream(stream)?;
            let got = pool.read(ctx, stream, qkv, want.len())?;
            if exact_shadow(l) {
                verdict(&format!("chain.l{l}.qkv"), &got, &want);
            } else {
                note_chain(&format!("chain.l{l}.qkv"), &got, &want);
            }
        }

        // The drafter's OWN q/k norm + rope (`dflash_qk`), and not `attn_glue`'s.
        //
        // F28's finding, and the whole of the backbone's red: `attn_glue::qk_norm_rope` is the
        // TARGET's kernel. Its norm is a GemmaRMSNorm -- `x * rsqrt(var + eps) * (f32(w) + 1)` --
        // and the drafter's `q_norm`/`k_norm` are the reference server's PLAIN `RMSNorm`, so this chain was
        // multiplying every element of q and k by `1 + w` where prod multiplies by `w`. Three
        // more differences ride with it (the variance's reduction ORDER, the bf16 round that
        // inductor does not do between the scale and the weight, and a rope that CONTRACTS where
        // `attn_glue` is compiled `-ffp-contract=off`), so it is a different kernel and not a
        // different template parameter.
        //
        // Two records under ONE op name and both called `weight` (the record map keys on the last
        // dot-segment), so they are taken BY POSITION -- the plan pushes q_norm then k_norm,
        // which is the order the kernel's `qw`/`kw` want.
        dispatch(
            ctx,
            launcher,
            stream,
            rules,
            DflashOp::QkNormRope,
            DflashShape::qk_rope(mrows, nq, nkv, hd),
            &[
                ("qkv", qkv),
                ("qw", wpos(&p("attn.qk_norm_rope"), 0)?),
                ("kw", wpos(&p("attn.qk_norm_rope"), 1)?),
                ("cos_sin", w("context_kv.rope", "cos_sin")?),
                ("pos", positions_q),
                ("q_out", q_rot),
                ("k_out", k_rot),
            ],
        )?;
        ctx.hip.sync_stream(stream)?;
        // V IS THE ONLY BIT-EXACT REFERENCE FOR THE PROJECTION, and it was in the capture all
        // along. `attn.l{l}.v` is prod's own tensor, tapped at the attention impl -- and it is a
        // VIEW of prod's own qkv (the manifest records its stride as 6144, the projection's row),
        // so it is the last third of the tensor this chain has just computed, with nothing
        // between. It splits the prefix in two, exactly, with no shadow in it:
        //
        //   v identical, q not  -> the norm + rope is what differs
        //   v not identical     -> the conv, its quant, or one of the two GEMMs in front of it,
        //                          and any claim about the norm + rope is untested
        let v_off = off(qkv, u64::from(qn + nkv * hd) * 2);
        if let Ok(want_v) = cap.bytes(&format!("attn.l{l}.v")) {
            // The WHOLE projection is read and V is sliced ROW BY ROW on the host. Two things
            // this cost a lock hold apiece: `Pool2::read` finds an allocation by its BASE
            // address, so `v_off` (an offset into one) is not readable; and the v BLOCK is
            // `[5120, 6144)` of EVERY row, not the tail of the tensor -- taking the last
            // `want_v.len()` bytes reads rows 6 and 7 of `[q | k | v]` and compares them against
            // eight rows of v, which is a number that means nothing and does not look wrong.
            let all = pool.read(ctx, stream, qkv, (mrows * qkv_n * 2) as usize)?;
            let row = (qkv_n * 2) as usize;
            let vb = (nkv * hd * 2) as usize;
            let off_v = (qn + nkv * hd) as usize * 2;
            let mut got_v = Vec::with_capacity(want_v.len());
            for t in 0..mrows as usize {
                let b = t * row + off_v;
                got_v.extend_from_slice(&all[b..(b + vb).min(all.len())]);
            }
            // THE DECLARED CLASS (F32). When the first divergence in the body that produced this
            // tensor is a KERNEL PROJECTION -- the bf16 `rocm_unquantized_gemm` whose 1,280
            // outputs are the conv's per-group coefficients -- and it is a few ULPs on a handful
            // of them, this row is that GEMM's accumulation order carried forward and amplified,
            // not a kernel of ours: one wrong coefficient multiplies a whole 16-channel group at
            // every row it reaches, and the four convs' tap-1 look-back spreads it to every row
            // above the first. Measured on the host BEFORE it was declared here: substituting the
            // region's own three projections into the emulated body takes `attn.l1.v` from
            // 868/919/786/876/703/910/888 differing elements at rows 1-7 to ZERO on all eight
            // (p86), and p54's whole tensor from 4,779 bytes to one element
            // (`evidence-f32/coeff_arm.txt`). The row prints either way; the class decides only
            // whether it FAILS the gate.
            if body_class {
                let d = got_v.iter().zip(&want_v).filter(|(a, b)| a != b).count()
                    + got_v.len().abs_diff(want_v.len());
                println!(
                    "   {:<26} {} of {} bytes -- the kernel projection's bf16 GEMM order, carried \
                     by the conv look-back (the F32 class); NOT counted",
                    format!("chain.l{l}.v (prod's own)"),
                    if d == 0 {
                        "identical".to_string()
                    } else {
                        format!("DIFFERS in {d}")
                    },
                    want_v.len()
                );
            } else {
                verdict(&format!("chain.l{l}.v (prod's own)"), &got_v, &want_v);
            }
        }
        verdict(
            &format!("chain.attn.l{l}.q"),
            &pool.read(ctx, stream, q_rot, (mrows * qn * 2) as usize)?,
            &cap.bytes(&format!("attn.l{l}.q"))?,
        );
        if let Ok(want_k) = cap.bytes(&format!("attn.l{l}.k")) {
            verdict(
                &format!("chain.attn.l{l}.k"),
                &pool.read(ctx, stream, k_rot, want_k.len())?,
                &want_k,
            );
        }
        let mut ap = DflashShape::new(mrows, nkv, hd);
        ap.block_size = kv_block;
        // `value` is `qkv + (qn + nkv * hd)` -- the v BLOCK of every row of the projection, not a
        // contiguous tensor. Its source stride is the projection's row, which is the one fact the
        // append had been assuming wrongly (F29: 7,121 of 3,375,104 cache bytes).
        ap.kv_src_stride = (nkv * hd, qkv_n);
        dispatch(
            ctx,
            launcher,
            stream,
            rules,
            DflashOp::KvAppendFp8,
            ap,
            &[
                ("key", k_rot),
                ("value", v_off),
                ("kc", caches[l as usize].0),
                ("vc", caches[l as usize].1),
                ("slots", query_slots),
                ("scales", one_f32),
            ],
        )?;
        // THE CACHE THE ATTENTION IS ABOUT TO READ, against prod's own after its forward.
        //
        // The store's rows above stop at `kv.after_ctx`; this is the other half of the cache and
        // it was never gated. It is the row that separates "our q is wrong" from "our attention
        // is reading a different cache", and F29 needed exactly that separation: on p86 q, k and
        // v were all bit-identical to prod's own and `chain.attn.l0.out` still differed in 95% of
        // its bytes. `kv.after_fwd` is prod's cache after the whole drafter forward, and the only
        // thing that writes to it after the store is this append, so at layer `l` it is what our
        // buffer must hold.
        if let Ok(want_inter) = cap.bytes(&format!("kv.after_fwd.l{l}")) {
            ctx.hip.sync_stream(stream)?;
            let (wk, wv) = deinterleave(
                &want_inter,
                nb,
                nkv as usize,
                kv_block as usize,
                hd as usize,
                elem,
            );
            verdict(
                &format!("chain.kv.l{l}.k (after the query append)"),
                &pool.read(ctx, stream, caches[l as usize].0, cache_bytes)?,
                &wk,
            );
            verdict(
                &format!("chain.kv.l{l}.v (after the query append)"),
                &pool.read(ctx, stream, caches[l as usize].1, cache_bytes)?,
                &wv,
            );
        }
        dispatch(
            ctx,
            launcher,
            stream,
            rules,
            DflashOp::AttnFp8Kv,
            DflashShape::attn(mrows, nq, seq_len, nkv, kv_block, bt_stride),
            &[
                ("q", q_rot),
                ("kc", caches[l as usize].0),
                ("vc", caches[l as usize].1),
                ("block_table", block_table),
                ("seq_lens", seq_lens),
                ("k_scale", one_f32_1),
                ("v_scale", one_f32_1),
                ("out", attn_out),
            ],
        )?;
        ctx.hip.sync_stream(stream)?;
        verdict(
            &format!("chain.attn.l{l}.out"),
            &pool.read(ctx, stream, attn_out, (mrows * qn * 2) as usize)?,
            &cap.bytes(&format!("attn.l{l}.out"))?,
        );
        dispatch(
            ctx,
            launcher,
            stream,
            rules,
            // The COMPILED program, not the eager one: this quant is inside the drafter's
            // region, where inductor strength-reduces `absmax / 448` (F31).
            DflashOp::ActQuantRecip,
            DflashShape::new(mrows, qn, qn),
            &[("x", attn_out), ("q", oq), ("scales", os)],
        )?;
        dispatch(
            ctx,
            launcher,
            stream,
            rules,
            DflashOp::GemmBlockscale,
            DflashShape::new(mrows, h, qn),
            &[
                ("a", oq),
                ("as", os),
                ("w", w(&p("attn.o_proj"), "w")?),
                ("ws", w(&p("attn.o_proj"), "ws")?),
                ("c", o),
            ],
        )?;
        dispatch(
            ctx,
            launcher,
            stream,
            rules,
            DflashOp::ConvAddRmsOpen,
            DflashShape::conv(mrows, h, groups, cfg.block_size, 1),
            &[
                ("x", o),
                ("coeff", coeff_a),
                ("base", w(&p("attention_conv.prepare"), "base")?),
                ("resid", resid),
                (
                    "weight",
                    w(
                        &p("attention_conv.finish+post_attention_layernorm"),
                        "weight",
                    )?,
                ),
                ("out", normed2),
                ("res_out", resid2),
            ],
        )?;
        dispatch(
            ctx,
            launcher,
            stream,
            rules,
            DflashOp::BgemmBf16,
            DflashShape::new(mrows, kp_n, h),
            &[
                ("a", normed2),
                ("w", w(&p("mlp_conv.kernel_projection"), "w")?),
                ("c", coeff_m),
            ],
        )?;
        dispatch(
            ctx,
            launcher,
            stream,
            rules,
            DflashOp::GroupedConvQuant,
            DflashShape::conv(mrows, h, groups, cfg.block_size, 0),
            &[
                ("x", normed2),
                ("coeff", coeff_m),
                ("base", w(&p("mlp_conv.prepare"), "base")?),
                ("q", mq),
                ("scales", ms),
            ],
        )?;
        dispatch(
            ctx,
            launcher,
            stream,
            rules,
            DflashOp::GemmBlockscale,
            DflashShape::new(mrows, 2 * inter, h),
            &[
                ("a", mq),
                ("as", ms),
                ("w", w(&p("mlp.gate_up"), "w")?),
                ("ws", w(&p("mlp.gate_up"), "ws")?),
                ("c", gate_up),
            ],
        )?;
        dispatch(
            ctx,
            launcher,
            stream,
            rules,
            DflashOp::SiluQuant,
            DflashShape::new(mrows, inter, 2 * inter),
            &[("gate_up", gate_up), ("q", dq), ("scales", ds)],
        )?;
        dispatch(
            ctx,
            launcher,
            stream,
            rules,
            DflashOp::GemmBlockscale,
            DflashShape::new(mrows, h, inter),
            &[
                ("a", dq),
                ("as", ds),
                ("w", w(&p("mlp.down"), "w")?),
                ("ws", w(&p("mlp.down"), "ws")?),
                ("c", down),
            ],
        )?;
        dispatch(
            ctx,
            launcher,
            stream,
            rules,
            DflashOp::ConvAddRmsClose,
            DflashShape::conv(mrows, h, groups, cfg.block_size, 1),
            &[
                ("x", down),
                ("coeff", coeff_m),
                ("base", w(&p("mlp_conv.prepare"), "base")?),
                ("resid", resid2),
                ("weight", w(&p("mlp_conv.finish+next_layernorm"), "weight")?),
                ("out", out_norm),
                ("res_out", res_out),
            ],
        )?;

        // The layer's own regions, in the order the launches above wrote them.
        for (nm, a, n) in [
            ("coeff_a", coeff_a, (mrows * kp_n * 2) as usize),
            ("aq", aq, (mrows * h) as usize),
            ("as", as_, (mrows * h.div_ceil(128) * 4) as usize),
            ("qkv", qkv, (mrows * qkv_n * 2) as usize),
            ("q_rot", q_rot, (mrows * qn * 2) as usize),
            ("k_rot", k_rot, (mrows * nkv * hd * 2) as usize),
            ("attn_out", attn_out, (mrows * qn * 2) as usize),
            ("oq", oq, (mrows * qn) as usize),
            ("os", os, (mrows * qn.div_ceil(128) * 4) as usize),
            ("o", o, (mrows * h * 2) as usize),
            ("normed2", normed2, (mrows * h * 2) as usize),
            ("resid2", resid2, (mrows * h * 4) as usize),
            ("coeff_m", coeff_m, (mrows * kp_n * 2) as usize),
            ("mq", mq, (mrows * h) as usize),
            ("ms", ms, (mrows * h.div_ceil(128) * 4) as usize),
            ("gate_up", gate_up, (mrows * 2 * inter * 2) as usize),
            ("dq", dq, (mrows * inter) as usize),
            ("ds", ds, (mrows * inter.div_ceil(128) * 4) as usize),
            ("down", down, (mrows * h * 2) as usize),
            ("out_norm", out_norm, (mrows * h * 2) as usize),
            ("res_out", res_out, (mrows * h * 2) as usize),
        ] {
            dumps.push((format!("bb.l{l}.{nm}"), a, n));
        }

        // ---- F32: THE BODY, OP BY OP, AGAINST THE REGION'S OWN OPERANDS -----------------------
        //
        // Every buffer of this layer's body is still live here, and every one of them now has a
        // prod-true anchor: the compiled region's GEMM calls carry the quantised activations and
        // their scales, both kernel projections (`buf10` among them, whose finish half is what the
        // second `conv_add_rms` multiplies by), the two normed outputs `_2` and `_8` write, and
        // every GEMM result. The rows are NOTES -- op N is reached through ops 1..N-1 -- and what
        // they are for is the FIRST one that fires.
        {
            let sites: [(u32, &str, &str, &str, DeviceAddr, usize); 16] = [
                (l, "o_proj", "in0", "_0/_1 act_quant   q", oq, 1),
                (l, "o_proj", "in2", "_0/_1 act_quant   s", os, 4),
                (l, "o_proj", "out", "o_proj GEMM", o, 2),
                (l, "kp_mlp", "in0", "_2 conv_add_rms OPEN", normed2, 2),
                (l, "kp_mlp", "out", "kp_mlp bgemm (buf10)", coeff_m, 2),
                (l, "gate_up", "in0", "_3/_4/_5 conv+quant q", mq, 1),
                (l, "gate_up", "in2", "_3/_4/_5 conv+quant s", ms, 4),
                (l, "gate_up", "out", "gate_up GEMM", gate_up, 2),
                (l, "down", "in0", "_6/_7 silu+quant  q", dq, 1),
                (l, "down", "in2", "_6/_7 silu+quant  s", ds, 4),
                (l, "down", "out", "down GEMM", down, 2),
                (
                    l + 1,
                    "kp_attn",
                    "in0",
                    "_8 conv_add_rms CLOSE",
                    out_norm,
                    2,
                ),
                (l, "kp_attn", "out", "kp_attn bgemm (buf29)", coeff_a, 2),
                (l, "qkv", "in0", "_3/_4/_5 aconv+quant q", aq, 1),
                (l, "qkv", "in2", "_3/_4/_5 aconv+quant s", as_, 4),
                (l, "qkv", "out", "qkv GEMM", qkv, 2),
            ];
            // The order the region RUNS them in, which is the order a first divergence has to be
            // read in: the two that belong to this layer's own top (`kp_attn`, the attention
            // conv's quant and the qkv GEMM) ran before its attention did.
            let order = [12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
            let mut first: Option<(String, usize, i64)> = None;
            for i in order {
                let (layer, call, which, label, buf, width) = sites[i];
                let Some(idx) = rmap.get(&(layer, call.to_string())) else {
                    continue;
                };
                let Ok(want) = cap.bytes(&format!("region.c{idx:02}.{which}")) else {
                    continue;
                };
                ctx.hip.sync_stream(stream)?;
                let got = pool.read(ctx, stream, buf, want.len())?;
                let (d, ulp) = region_diff(&got, &want, width);
                println!(
                    "   region l{l} {label:<22} {} of {} elements -- the region's own operand; \
                     NOT counted",
                    if d == 0 {
                        "identical".to_string()
                    } else {
                        format!("DIFFERS in {d} (max {ulp} ULP)")
                    },
                    want.len() / width
                );
                if d > 0 && first.is_none() {
                    first = Some((call.to_string(), d, ulp));
                }
            }
            // THE CLASS, and it is a rule over the measurement rather than a name for it: the
            // body's FIRST divergence is one of the two kernel projections -- the bf16 GEMM whose
            // order is prod's hipblaslt's and not ours -- and it is at most sixteen of the 1,280
            // coefficients by at most four ULPs. Anything else, anywhere else, and the next
            // layer's `v` is counted as it always was.
            body_class = match &first {
                None => true,
                Some((call, d, ulp)) => {
                    (call == "kp_attn" || call == "kp_mlp") && *d <= 16 && *ulp <= 4
                }
            };
            if let Some((call, d, ulp)) = &first {
                println!(
                    "   region l{l} FIRST divergence: {call} in {d} elements, max {ulp} ULP -- \
                     {}",
                    if body_class {
                        "the declared bf16 kernel-projection class"
                    } else {
                        "NOT the declared class; the next layer's `v` is counted"
                    }
                );
            }
        }

        normed = out_norm;
        resid = res_out;
    }

    // 3. the seam the chain was written to find: the drafter's `block_size` rows narrowed to the
    // `steps` the head scores, then the head's own pad and group sum.
    let head_hidden = pool.zeroed(ctx, stream, (steps * h * 2) as usize)?;
    let sample_idx = pool.upload(ctx, stream, &cap.bytes("idx.sample_indices")?)?;
    dispatch(
        ctx,
        launcher,
        stream,
        rules,
        DflashOp::RowGather,
        DflashShape::new(steps, h * 2, mrows),
        &[("in", normed), ("idx", sample_idx), ("out", head_hidden)],
    )?;
    ctx.hip.sync_stream(stream)?;
    verdict(
        "chain.head.hidden",
        &pool.read(ctx, stream, head_hidden, (steps * h * 2) as usize)?,
        &cap.bytes("head.hidden")?,
    );

    let m_pad = steps.max(16).next_power_of_two();
    let x_pad = pool.zeroed(ctx, stream, (m_pad * h * 2) as usize)?;
    let xs = pool.zeroed(ctx, stream, (m_pad * h.div_ceil(128) * 4) as usize)?;
    dispatch(
        ctx,
        launcher,
        stream,
        rules,
        DflashOp::HeadRowPad,
        DflashShape::rerank(m_pad, h, h, steps),
        &[("x", head_hidden), ("out", x_pad)],
    )?;
    dispatch(
        ctx,
        launcher,
        stream,
        rules,
        DflashOp::HeadGroupSum,
        DflashShape::new(m_pad, h, h),
        &[("x", x_pad), ("xs", xs)],
    )?;
    ctx.hip.sync_stream(stream)?;
    if let Ok(want) = cap.bytes("head.xs") {
        verdict(
            "chain.head.xs",
            &pool.read(ctx, stream, xs, want.len())?,
            &want,
        );
    }

    // 4. the head, its selection, the gather the selection needs, and the exact rerank.
    let bm_w = cfg.head_block_maxima();
    let y = pool.zeroed(ctx, stream, (m_pad * vloc * 2) as usize)?;
    let bmax = pool.zeroed(ctx, stream, (m_pad * bm_w * 4) as usize)?;
    let bidx = pool.zeroed(ctx, stream, (m_pad * bm_w * 4) as usize)?;
    dispatch(
        ctx,
        launcher,
        stream,
        rules,
        DflashOp::DraftHeadInt2,
        DflashShape::new(m_pad, vloc, h),
        &[
            ("x", x_pad),
            ("xs", xs),
            ("wq", w("head.int2", "wq")?),
            ("s", w("head.int2", "s")?),
            ("zs", w("head.int2", "zs")?),
            ("y", y),
            ("bmax", bmax),
            ("bidx", bidx),
        ],
    )?;
    let sel_shape = DflashShape::rerank(m_pad, bm_w, h, cfg.rerank);
    let thresh = pool.zeroed(ctx, stream, (m_pad * 8) as usize)?;
    let tvals = pool.zeroed(ctx, stream, (m_pad * cfg.rerank * 4) as usize)?;
    let tpos = pool.zeroed(ctx, stream, (m_pad * cfg.rerank * 4) as usize)?;
    let cand80 = pool.zeroed(ctx, stream, (m_pad * cfg.rerank * 4) as usize)?;
    dispatch(
        ctx,
        launcher,
        stream,
        rules,
        DflashOp::HeadTopkThresh,
        sel_shape,
        &[("x", bmax), ("thresh", thresh)],
    )?;
    dispatch(
        ctx,
        launcher,
        stream,
        rules,
        DflashOp::HeadTopkGather,
        sel_shape,
        &[
            ("x", bmax),
            ("thresh", thresh),
            ("vals", tvals),
            ("idx", tpos),
        ],
    )?;
    dispatch(
        ctx,
        launcher,
        stream,
        rules,
        DflashOp::HeadCandGather,
        sel_shape,
        &[("bidx", bidx), ("pos", tpos), ("out", cand80)],
    )?;
    ctx.hip.sync_stream(stream)?;
    if let Ok(want) = cap.bytes("head.rerank_idx") {
        verdict(
            "chain.head.rerank_idx",
            &pool.read(ctx, stream, cand80, want.len())?,
            &want,
        );
    }
    let rr_op = rerank_op(weights_of.get("head.rerank").map(Vec::as_slice));
    let rr = saber_kernels::dflash::resolve(
        rr_op,
        DflashShape::rerank(m_pad, vloc, h, cfg.rerank),
        rules,
    )?;
    let scatter = rr.kernel.contains("scatter");
    let rr_out = if scatter {
        y
    } else {
        pool.zeroed(ctx, stream, (m_pad * cfg.rerank * 4) as usize)?
    };
    let mut rb: Vec<(&str, DeviceAddr)> = vec![("x", x_pad), ("idx", cand80), ("out", rr_out)];
    if let Some(ws) = weights_of.get("head.rerank") {
        rb.extend(ws.iter().copied());
    }
    dispatch(
        ctx,
        launcher,
        stream,
        rules,
        rr_op,
        DflashShape::rerank(m_pad, vloc, h, cfg.rerank),
        &rb,
    )?;
    ctx.hip.sync_stream(stream)?;

    // 5. the selector's own top-16 over the reranked vocabulary row, and the anchor gather.
    //
    // `sidx` is EIGHT bytes an element: `torch.topk` returns int64 and the selector's `cand`
    // kernarg is `const long *`, so the selection writes what its consumer reads. It wrote i32
    // through F27 and this chain re-widened on the host between the two launches -- a round trip
    // in a proposal whose whole claim is `host_round_trips() == 0`.
    let svals = pool.zeroed(ctx, stream, (steps * top_k * 4) as usize)?;
    let sidx = pool.zeroed(ctx, stream, (steps * top_k * 8) as usize)?;
    dispatch(
        ctx,
        launcher,
        stream,
        rules,
        DflashOp::SelectorTopk,
        DflashShape::rerank(steps, vloc, h, top_k),
        &[("x", y), ("vals", svals), ("idx", sidx)],
    )?;
    ctx.hip.sync_stream(stream)?;
    let got_ids = pool.read(ctx, stream, sidx, (steps * top_k * 8) as usize)?;
    if let Ok(want) = cap.bytes("head.candidate_ids") {
        verdict("chain.head.candidate_ids", &got_ids, &want);
    }
    if let Ok(want) = cap.bytes("head.unary_logits") {
        let got_v = pool.read(ctx, stream, svals, want.len())?;
        verdict("chain.head.unary_logits", &got_v, &want);
    }

    // THE EDGE THAT USED TO NEED A HOST ROUND TRIP, closed in F28. `dflash_topk_bf16_k16` writes
    // the i64 its only consumer declares (`dflash_selector_score_r256_k16`'s `cand` is
    // `const long *`, because prod's candidate ids are `torch.topk`'s int64 and F24's finding was
    // that a narrower declaration reads the right addresses and the wrong bytes), so the
    // selection's output buffer goes straight into the scoring. Nothing is re-widened and nothing
    // is uploaded: `sidx` IS `cand`.
    let cand_i64 = sidx;
    let anchor = pool.zeroed(ctx, stream, 4)?;
    let anchor_idx: Vec<u8> = (0..cap.int("num_reqs").unwrap_or(1).max(1))
        .flat_map(|r| i64::from(r * mrows).to_le_bytes())
        .collect();
    let anchor_ia = pool.upload(ctx, stream, &anchor_idx)?;
    dispatch(
        ctx,
        launcher,
        stream,
        rules,
        DflashOp::RowGather,
        DflashShape::new(1, 4, mrows),
        &[("in", ids), ("idx", anchor_ia), ("out", anchor)],
    )?;
    let proj = pool.zeroed(ctx, stream, (steps * cfg.selector_rank * 2) as usize)?;
    dispatch(
        ctx,
        launcher,
        stream,
        rules,
        DflashOp::BgemmBf16,
        DflashShape::new(steps, cfg.selector_rank, h),
        &[
            ("a", head_hidden),
            ("w", w("selector.hidden_projection", "w")?),
            ("c", proj),
        ],
    )?;
    let sc_len = cap
        .bytes("walk.scores")
        .map(|b| b.len())
        .unwrap_or((steps * top_k * top_k * 4) as usize);
    let scores = pool.zeroed(ctx, stream, sc_len)?;
    let mut sb: Vec<(&str, DeviceAddr)> = vec![
        ("cand", cand_i64),
        ("anchor", anchor),
        ("unary", svals),
        ("hidden", proj),
        ("out", scores),
    ];
    if let Some(cb) = weights_of.get("selector.score_edges") {
        sb.extend(cb.iter().copied());
    }
    dispatch(
        ctx,
        launcher,
        stream,
        rules,
        DflashOp::SelectorScore,
        DflashShape::selector(steps, steps, cfg.selector_rank, top_k),
        &sb,
    )?;
    ctx.hip.sync_stream(stream)?;
    if let Ok(want) = cap.bytes("walk.scores") {
        verdict(
            "chain.selector.scores",
            &pool.read(ctx, stream, scores, want.len())?,
            &want,
        );
    }

    // 6. the walk, and the 56 bytes this whole chain exists to produce.
    let tok_ref = cap.bytes("out.draft_tokens")?;
    let real_len = cap
        .bytes("walk.realized_scores")
        .map(|b| b.len())
        .unwrap_or((steps * top_k * 4) as usize);
    let tokens = pool.zeroed(ctx, stream, tok_ref.len())?;
    let realized = pool.zeroed(ctx, stream, real_len)?;
    let req_state = pool.upload(ctx, stream, &cap.bytes("walk.idx_mapping")?)?;
    dispatch(
        ctx,
        launcher,
        stream,
        rules,
        DflashOp::SelectorWalk,
        DflashShape::rerank(1, steps, top_k, top_k),
        &[
            ("scores", scores),
            ("candidates", cand_i64),
            ("req_state", req_state),
            ("tokens", tokens),
            ("realized", realized),
        ],
    )?;
    ctx.hip.sync_stream(stream)?;
    for (nm, a, n) in [
        ("head.hidden", head_hidden, (steps * h * 2) as usize),
        ("head.x_pad", x_pad, (m_pad * h * 2) as usize),
        ("head.xs", xs, (m_pad * h.div_ceil(128) * 4) as usize),
        ("head.y", y, (m_pad * vloc * 2) as usize),
        ("head.bmax", bmax, (m_pad * bm_w * 4) as usize),
        ("head.bidx", bidx, (m_pad * bm_w * 4) as usize),
        ("head.thresh", thresh, (m_pad * 8) as usize),
        ("head.tvals", tvals, (m_pad * cfg.rerank * 4) as usize),
        ("head.tpos", tpos, (m_pad * cfg.rerank * 4) as usize),
        ("head.cand80", cand80, (m_pad * cfg.rerank * 4) as usize),
        ("sel.vals", svals, (steps * top_k * 4) as usize),
        ("sel.cand64", cand_i64, (steps * top_k * 8) as usize),
        ("sel.anchor", anchor, 4),
        ("sel.proj", proj, (steps * cfg.selector_rank * 2) as usize),
        ("sel.scores", scores, sc_len),
        ("out.tokens", tokens, tok_ref.len()),
        ("out.realized", realized, real_len),
    ] {
        dumps.push((nm.to_owned(), a, n));
    }
    if let Some(d) = dump {
        write_chain_dump(ctx, stream, &pool, d, &cap.tag, &dumps)?;
    }
    verdict(
        "chain.draft_block",
        &pool.read(ctx, stream, tokens, tok_ref.len())?,
        &tok_ref,
    );
    println!(
        "     the BACKBONE chained from `q.input_ids` and the caches the store above wrote: \n     \
         embedding -> five layers -> the sample gather -> pad -> xs -> int2 head -> top-80 -> \n     \
         the bi gather -> exact rerank -> top-16 -> anchor -> projection -> score_edges -> walk. \n     \
         Nothing between them is prod's except the index tables the scheduler wrote."
    );
    Ok(())
}

/// Write the chain's own intermediates, in launch order, for the engine's replay to compare with.
///
/// Raw little-endian bytes per region plus an `order.json` that carries the ORDER — which is the
/// half that matters, because "the first operand that differs" is only a statement about a
/// program if the list is the program's. One directory per prompt tag.
fn write_chain_dump(
    ctx: &LaunchCtx<'_>,
    stream: &Stream,
    pool: &Pool2,
    dir: &Path,
    tag: &str,
    dumps: &[(String, DeviceAddr, usize)],
) -> anyhow::Result<()> {
    let out = dir.join(tag);
    std::fs::create_dir_all(&out)?;
    ctx.hip.sync_stream(stream)?;
    let mut order: Vec<serde_json::Value> = Vec::with_capacity(dumps.len());
    let mut bytes = 0usize;
    for (name, addr, len) in dumps {
        let b = pool.read(ctx, stream, *addr, *len)?;
        std::fs::write(out.join(format!("{name}.bin")), &b)?;
        order.push(serde_json::json!({ "name": name, "bytes": b.len() }));
        bytes += b.len();
    }
    std::fs::write(
        out.join("order.json"),
        serde_json::to_vec_pretty(&serde_json::json!({ "regions": order }))?,
    )?;
    println!(
        "   chain dump  {} region(s), {} KiB -> {}",
        dumps.len(),
        bytes >> 10,
        out.display()
    );
    Ok(())
}

/// Every covered launch of the proposal, in forward order, at the shapes THIS captured step ran
/// at, timed.
///
/// The weights come from the pack by record name; the conv / attention / KV-append / selector
/// launches are bound to the capture's own index tables and KV cache, which is the half a pack
/// alone could not supply. What it is NOT is a chain: the five absent ops are absent and the two
/// undeclared quants ([`UNDECLARED_QUANTS`]) have no producer, so the intermediate activations are
/// scratch. Every launch runs at its real shape with its real weights, which is what a time needs;
/// the bit-exactness claims are the staged ones above.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn time_whole_proposal(
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    stream: &Stream,
    cfg: &DflashConfig,
    graph: &DraftGraph,
    rules: &DflashRules,
    cap: &Capture,
    weights_of: &BTreeMap<&str, Vec<(&str, DeviceAddr)>>,
    iters: u32,
) -> anyhow::Result<()> {
    let h = cfg.hidden;
    let hd = cfg.head_dim;
    let nkv = cfg.kv_heads_local();
    let num_ctx = cap.int("num_target_tokens")?;
    let kv_block = cap.int("kv_block_size")?;
    let seq_len = cap
        .i32_table("idx.seq_lens")
        .ok()
        .and_then(|v| v.first().copied())
        .map_or(4096, |v| u32::try_from(v).unwrap_or(4096))
        + cap.int("num_query_per_req")?;

    let mut pool = Pool2::new();

    // The KV cache, de-interleaved out of prod's own, and the block table renumbered onto the
    // blocks the capture carries.
    let (blocks, renum) = captured_blocks(cap);
    let nb = blocks.len().max(1);
    let elem = kv_elem(cap, "kv.after_ctx.l0", hd as usize)?;
    let mut caches: Vec<(DeviceAddr, DeviceAddr)> = Vec::new();
    for l in 0..cfg.layers {
        let key = format!("kv.after_ctx.l{l}");
        let src = cap.bytes(&key).unwrap_or_default();
        let (k, v) = if src.is_empty() {
            let n = nb * kv_block as usize * nkv as usize * hd as usize * elem;
            (vec![0u8; n], vec![0u8; n])
        } else {
            deinterleave(&src, nb, nkv as usize, kv_block as usize, hd as usize, elem)
        };
        caches.push((pool.upload(ctx, stream, &k)?, pool.upload(ctx, stream, &v)?));
    }
    let bt: Vec<i32> = cap
        .i32_table("idx.block_table")
        .unwrap_or_default()
        .iter()
        .map(|b| renum.get(&i64::from(*b)).copied().unwrap_or(0))
        .collect();
    let bt_bytes: Vec<u8> = if bt.is_empty() {
        vec![0u8; 4]
    } else {
        bt.iter().flat_map(|v| v.to_le_bytes()).collect()
    };
    let block_table = pool.upload(ctx, stream, &bt_bytes)?;
    // THE BLOCK TABLE'S ROW STRIDE, from the capture's own table rather than from an expression.
    // It is `[num_seqs, max_blocks]` and the stride is dimension 1: the scheduler padded every
    // sequence's page list to the widest in the step, so nothing but this step knows it. Until it
    // had a `DflashShape` field the five attention launches were REFUSED by name (round 21).
    let bt_stride = u32::try_from(
        cap.shape("idx.block_table")?
            .get(1)
            .copied()
            .unwrap_or(bt.len()),
    )
    .unwrap_or(1)
    .max(1);
    let seq_lens = pool.upload(
        ctx,
        stream,
        &i32::try_from(seq_len).unwrap_or(0).to_le_bytes(),
    )?;
    let slots: Vec<i64> = cap
        .i64_table("idx.query_slots")
        .unwrap_or_default()
        .iter()
        .map(|s| {
            let b = s / i64::from(kv_block);
            let o = s % i64::from(kv_block);
            i64::from(renum.get(&b).copied().unwrap_or(0)) * i64::from(kv_block) + o
        })
        .collect();
    let slot_bytes: Vec<u8> = if slots.is_empty() {
        vec![0u8; 8]
    } else {
        slots.iter().flat_map(|v| v.to_le_bytes()).collect()
    };
    let query_slots = pool.upload(ctx, stream, &slot_bytes)?;
    // ONE SLOT TABLE PER LAYER. `precompute_and_store_context_kv` calls `do_kv_cache_update` once
    // per layer with that layer's own `slot_mapping`, and the capture carries all five
    // (`ctx.slots.l0..l4`). Taking layer 0's for all five writes the right bytes to the wrong
    // rows in four caches, which is in range and therefore silent.
    let mut ctx_slots: Vec<DeviceAddr> = Vec::new();
    for l in 0..cfg.layers {
        let s: Vec<i64> = cap
            .i64_table(&format!("ctx.slots.l{l}"))
            .or_else(|_| cap.i64_table("ctx.slots"))
            .unwrap_or_default()
            .iter()
            .map(|s| {
                let b = s / i64::from(kv_block);
                let o = s % i64::from(kv_block);
                i64::from(renum.get(&b).copied().unwrap_or(0)) * i64::from(kv_block) + o
            })
            .collect();
        let b: Vec<u8> = if s.is_empty() {
            vec![0u8; 8]
        } else {
            s.iter().flat_map(|v| v.to_le_bytes()).collect()
        };
        ctx_slots.push(pool.upload(ctx, stream, &b)?);
    }
    // THE ROPE'S POSITIONS ARE REPEATED PER LAYER, and this is the 2026-09-21 aperture violation.
    //
    // `_rope_context_k` calls `ops.rotary_embedding(positions.repeat(num_layers), ...)` over the
    // `[L * num_ctx, ...]` flattened K, so the kernel's grid is `L * num_ctx` rows and it reads a
    // position for EACH of them. This file uploaded `ctx.positions` as it stands -- `num_ctx`
    // entries -- so rows `num_ctx..L*num_ctx` read whatever `hipMalloc` last left in those bytes,
    // and a garbage `int64` position times `rot_dim` is an index far outside the `[4096, 128]`
    // rope cache: `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`, and a 6.5 GiB coredump on a
    // shared card. Not a buffer that was too small for what a kernel WRITES, which is how round
    // 21 described it, but one too small for what a kernel READS -- and the repeat is the
    // reference's, not a pad.
    let ctx_positions = {
        let p = cap.bytes("ctx.positions").unwrap_or_else(|_| vec![0u8; 8]);
        let mut rep = Vec::with_capacity(p.len() * cfg.layers as usize);
        for _ in 0..cfg.layers.max(1) {
            rep.extend_from_slice(&p);
        }
        pool.upload(ctx, stream, &rep)?
    };
    let one_f32 = pool.upload(ctx, stream, &1.0f32.to_le_bytes().repeat(nkv as usize))?;

    // Everything else is scratch, sized from the shape each op declares.
    let mut scratch: BTreeMap<String, DeviceAddr> = BTreeMap::new();
    let mut get = |pool: &mut Pool2, key: &str, len: usize| -> anyhow::Result<DeviceAddr> {
        if let Some(a) = scratch.get(key) {
            return Ok(*a);
        }
        let a = pool.zeroed(ctx, stream, len)?;
        scratch.insert(key.to_string(), a);
        Ok(a)
    };

    struct Planned {
        /// The PLAN's name for this launch, carried so a fault can be attributed to an op
        /// instead of to the set. A HIP fault poisons the context, so the first pass syncs after
        /// every launch and this is what it prints; the timed passes do not sync at all.
        name: String,
        desc: saber_kernels::rules::LaunchDescriptor,
        shape: DflashShape,
        binds: Vec<(String, DeviceAddr)>,
    }
    let mut planned: Vec<Planned> = Vec::new();
    let mut refused = 0u32;
    let mut layer_of_insert = 0u32;
    for op in &graph.ops {
        let (Coverage::Dflash(kop), Some(declared)) = (op.coverage.clone(), op.shape) else {
            continue;
        };
        // The runtime shapes: the context store's rows and the attention's sequence length and
        // page size are properties of the STEP, not of the model, and the capture is what knows
        // them. Everything else is the plan's own declaration.
        let shape = match op.name.as_str() {
            "aux_hidden_projection" => DflashShape::new(num_ctx, h, cfg.target_layers * h),
            // The quant in front of it runs at the same rows over the SAME axis it reduces:
            // `[ctx, 5 x hidden]`, N = K, because a quant has one width and not two.
            "aux_hidden_projection.quant" => {
                DflashShape::new(num_ctx, cfg.target_layers * h, cfg.target_layers * h)
            }
            "context_kv.kv_projection" => DflashShape::new(num_ctx, cfg.context_kv_n(), h),
            "context_kv.layout" => DflashShape::ctx_layout(num_ctx, cfg.layers, nkv, hd),
            "context_kv.hidden_norm" => DflashShape::ctx_norm(num_ctx, h, num_ctx),
            "context_kv.k_norm" => {
                DflashShape::ctx_norm(cfg.layers * num_ctx * nkv, hd, num_ctx * nkv)
            }
            "context_kv.rope" => DflashShape::ctx_rope(cfg.layers * num_ctx, nkv, hd, hd),
            "context_kv.cache_insert" => {
                let mut s = DflashShape::new(num_ctx, nkv, hd);
                s.block_size = kv_block;
                s
            }
            _ if op.name.ends_with("attn.kv_append") => {
                let mut s = declared;
                s.block_size = kv_block;
                s
            }
            _ if op.name.ends_with("attn.attention") => DflashShape::attn(
                declared.m, declared.n, seq_len, declared.r, kv_block, bt_stride,
            ),
            _ => declared,
        };
        for _ in 0..op.count {
            let desc = saber_kernels::dflash::resolve(kop, shape, rules)?;
            let (m, n, k, r) = (
                u64::from(shape.m.max(1)),
                u64::from(shape.n.max(1)),
                u64::from(shape.k.max(1)),
                u64::from(shape.r.max(1)),
            );
            let layer = if op.name == "context_kv.cache_insert" {
                let l = layer_of_insert;
                layer_of_insert += 1;
                l as usize
            } else {
                op.name
                    .strip_prefix("layer.")
                    .and_then(|s| s.split('.').next())
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0)
            };
            let ws = weights_of
                .get(op.name.as_str())
                .cloned()
                .unwrap_or_default();
            // A SCALAR SLOT LEFT AS A BUFFER IS A REFUSAL, not a scratch allocation.
            //
            // `resolve` fills any signature slot its shape table does not name with
            // `ArgValue::Buffer`, so an `i32`/`f32` kernarg the rules forgot arrives here looking
            // exactly like a pointer -- and binding scratch to it would write eight bytes of
            // address into four bytes of kernarg and launch anyway. Four were found this way
            // (`conv_add_rms`/`rms_norm`'s `eps` and `attn`'s `scale` in round 21, `attn`'s
            // `bt_stride` in round 22). All four are values now -- `eps` and `scale` derived in
            // the rules, `bt_stride` a plan field the caller fills from the step's own table --
            // and `every_f32_kernarg_is_a_value_and_not_a_pointer` asks the SIGNATURE which slots
            // are scalars rather than carrying a name list. This stays as the launcher-side
            // backstop for a slot nobody has thought of yet, and it asks the same question.
            let scalar_leak: Vec<&str> = desc
                .args
                .iter()
                .filter_map(|s| match &s.value {
                    ArgValue::Buffer { buffer }
                        if saber_kernels::abi::signature_for(&desc.kernel).is_some_and(|sig| {
                            sig.iter().any(|a| {
                                a.name == s.name
                                    && !matches!(a.kind, saber_kernels::abi::ArgKind::Buffer)
                            })
                        }) =>
                    {
                        Some(buffer.as_str())
                    }
                    _ => None,
                })
                .collect();
            if !scalar_leak.is_empty() {
                println!(
                    "   REFUSED   {:<34} scalar kernarg(s) {:?} left unbound by the rules; \
                     dropped from the timed set rather than launched with a pointer in them",
                    op.name, scalar_leak
                );
                refused += op.count;
                break;
            }
            let mut binds: Vec<(String, DeviceAddr)> = Vec::new();
            for slot in &desc.args {
                let ArgValue::Buffer { buffer } = &slot.value else {
                    continue;
                };
                if let Some((_, a)) = ws.iter().find(|(nm, _)| nm == buffer) {
                    binds.push((buffer.clone(), *a));
                    continue;
                }
                // ONE RECORD, ONE OP -- but the conv's `base_kernel` is read TWICE, once by
                // `prepare` (side 0) and once by `finish` (side 1), and the record map names a
                // single op per record. `layer.i.<conv>.base` belongs to the prepare, so the
                // finish asks for a `base` nothing hands it. Taken from the sibling here rather
                // than duplicated in the map, because a pack with the same bytes under two names
                // would be 12.5 MiB per layer of the same weight and a second thing to keep in
                // step.
                if buffer == "base" {
                    if let Some(prefix) = op.name.split('+').next().and_then(|n| n.rsplit_once('.'))
                    {
                        let sibling = format!("{}.prepare", prefix.0);
                        if let Some((_, a)) = weights_of
                            .get(sibling.as_str())
                            .and_then(|w| w.iter().find(|(nm, _)| *nm == "base"))
                        {
                            binds.push((buffer.clone(), *a));
                            continue;
                        }
                    }
                }
                let a = match (kop, buffer.as_str()) {
                    (_, "kc") => caches[layer.min(caches.len() - 1)].0,
                    (_, "vc") => caches[layer.min(caches.len() - 1)].1,
                    (_, "block_table") => block_table,
                    (_, "seq_lens") => seq_lens,
                    (_, "k_scale" | "v_scale") => one_f32,
                    (DflashOp::KvAppendBf16 | DflashOp::KvAppendFp8, "scales") => one_f32,
                    (_, "slots") => {
                        if op.name == "context_kv.cache_insert" {
                            ctx_slots[layer.min(ctx_slots.len() - 1)]
                        } else {
                            query_slots
                        }
                    }
                    (DflashOp::CtxRope, "positions") => ctx_positions,
                    (DflashOp::GemmBlockscale, "a") => {
                        get(&mut pool, &format!("{}.a", op.name), (m * k) as usize)?
                    }
                    (DflashOp::GemmBlockscale, "as") => get(
                        &mut pool,
                        &format!("{}.as", op.name),
                        (m * k.div_ceil(128) * 4) as usize,
                    )?,
                    (DflashOp::GemmBlockscale | DflashOp::BgemmBf16, "c") => {
                        get(&mut pool, &format!("{}.c", op.name), (m * n * 2) as usize)?
                    }
                    (DflashOp::BgemmBf16, "a") => {
                        get(&mut pool, &format!("{}.a", op.name), (m * k * 2) as usize)?
                    }
                    (DflashOp::DraftHeadInt2, "x" | "y") => get(
                        &mut pool,
                        &format!("{}.{buffer}", op.name),
                        (m * if buffer == "x" { k } else { n } * 2) as usize,
                    )?,
                    (DflashOp::DraftHeadInt2, "xs") => get(
                        &mut pool,
                        &format!("{}.xs", op.name),
                        (m * k.div_ceil(128) * 4) as usize,
                    )?,
                    (DflashOp::DraftHeadInt2, "bmax" | "bidx") => get(
                        &mut pool,
                        &format!("{}.{buffer}", op.name),
                        (m * n.div_ceil(64) * 8 * 4) as usize,
                    )?,
                    (DflashOp::RerankExact | DflashOp::RerankExactBf16, "x") => {
                        get(&mut pool, &format!("{}.x", op.name), (m * k * 2) as usize)?
                    }
                    (DflashOp::RerankExact | DflashOp::RerankExactBf16, "idx") => {
                        get(&mut pool, &format!("{}.idx", op.name), (m * r * 4) as usize)?
                    }
                    (DflashOp::RerankExact | DflashOp::RerankExactBf16, "out") => get(
                        &mut pool,
                        &format!("{}.out", op.name),
                        ((r * 4).max(n * 2) * m) as usize,
                    )?,
                    (DflashOp::GroupedConv | DflashOp::GroupedConvQuant, "x") => {
                        get(&mut pool, &format!("{}.x", op.name), (m * n * 2) as usize)?
                    }
                    (DflashOp::GroupedConvQuant, "q") => {
                        get(&mut pool, &format!("{}.q", op.name), (m * n) as usize)?
                    }
                    (DflashOp::GroupedConvQuant, "scales") => get(
                        &mut pool,
                        &format!("{}.qs", op.name),
                        (m * n.div_ceil(128) * 4) as usize,
                    )?,
                    (DflashOp::GroupedConv | DflashOp::GroupedConvQuant, "coeff")
                    | (DflashOp::ConvAddRmsOpen | DflashOp::ConvAddRmsClose, "coeff") => get(
                        &mut pool,
                        &format!("{}.coeff", op.name),
                        (m * u64::from(cfg.kernel_projection_n()) * 2) as usize,
                    )?,
                    // FOUR BYTES for the fp32 end of the layer's residual and two for the bf16
                    // one: `_open` writes it and `_close` reads it. A slot sized at the wrong
                    // width here is the class `capture_widths.rs` exists to refuse, one layer out.
                    (DflashOp::ConvAddRmsOpen, "res_out")
                    | (DflashOp::ConvAddRmsClose, "resid") => get(
                        &mut pool,
                        &format!("{}.{buffer}", op.name),
                        (m * n * 4) as usize,
                    )?,
                    (
                        DflashOp::ConvAddRmsOpen | DflashOp::ConvAddRmsClose,
                        "x" | "resid" | "out" | "res_out",
                    ) => get(
                        &mut pool,
                        &format!("{}.{buffer}", op.name),
                        (m * n * 2) as usize,
                    )?,
                    // The activation quant: `[m, n]` bf16 in, e4m3 out, one fp32 scale per
                    // 128-wide group. The same three slots `silu_quant` has at half the input.
                    (DflashOp::ActQuant | DflashOp::ActQuantRecip, "x") => {
                        get(&mut pool, &format!("{}.x", op.name), (m * n * 2) as usize)?
                    }
                    (DflashOp::ActQuant | DflashOp::ActQuantRecip, "q") => {
                        get(&mut pool, &format!("{}.q", op.name), (m * n) as usize)?
                    }
                    (DflashOp::ActQuant | DflashOp::ActQuantRecip, "scales") => get(
                        &mut pool,
                        &format!("{}.s", op.name),
                        (m * n.div_ceil(128) * 4) as usize,
                    )?,
                    // The context store's layout copy: `[num_ctx, L, 2, nkv, hd]` in and the same
                    // element count out. `n` is the LAYER count here and `k`/`r` are the kv-head
                    // and head-dim extents, so the byte count is the product and a 2 for K-and-V.
                    (DflashOp::CtxKvLayout, "in" | "out") => get(
                        &mut pool,
                        &format!("{}.{buffer}", op.name),
                        (m * n * 2 * k * r * 2) as usize,
                    )?,
                    // The int2 head's own selection. `thresh` is ONE u64 per row and it is the
                    // only thing the two passes share.
                    (DflashOp::HeadTopkThresh | DflashOp::HeadTopkGather, "x") => {
                        get(&mut pool, "head.topk.x", (m * n * 4) as usize)?
                    }
                    (DflashOp::HeadTopkThresh | DflashOp::HeadTopkGather, "thresh") => {
                        get(&mut pool, "head.topk.thresh", (m * 8) as usize)?
                    }
                    (DflashOp::HeadTopkGather, "vals") => {
                        get(&mut pool, "head.topk.vals", (m * r * 4) as usize)?
                    }
                    (DflashOp::HeadTopkGather, "idx") => {
                        get(&mut pool, "head.topk.idx", (m * r * 4) as usize)?
                    }
                    // The selector's top-16 over the vocab shard: bf16 logits in, fp32 values and
                    // i32 indices out (ATen returns the input dtype and the caller converts).
                    (DflashOp::SelectorTopk, "x") => {
                        get(&mut pool, "selector.topk.x", (m * n * 2) as usize)?
                    }
                    (DflashOp::SelectorTopk, "vals") => {
                        get(&mut pool, "selector.topk.vals", (m * r * 4) as usize)?
                    }
                    (DflashOp::SelectorTopk, "idx") => {
                        get(&mut pool, "selector.topk.idx", (m * r * 4) as usize)?
                    }
                    (DflashOp::SiluQuant, "gate_up") => get(
                        &mut pool,
                        &format!("{}.gate_up", op.name),
                        (m * k * 2) as usize,
                    )?,
                    (DflashOp::SiluQuant, "q") => {
                        get(&mut pool, &format!("{}.q", op.name), (m * n) as usize)?
                    }
                    (DflashOp::SiluQuant, "scales") => get(
                        &mut pool,
                        &format!("{}.s", op.name),
                        (m * n.div_ceil(128) * 4) as usize,
                    )?,
                    (DflashOp::RmsNorm | DflashOp::CtxRmsNorm, "x" | "out") => get(
                        &mut pool,
                        &format!("{}.{buffer}", op.name),
                        (m * n * 2) as usize,
                    )?,
                    (DflashOp::CtxRope, "k") => get(
                        &mut pool,
                        &format!("{}.k", op.name),
                        (m * n * k * 2) as usize,
                    )?,
                    (DflashOp::KvAppendBf16 | DflashOp::KvAppendFp8, "key" | "value") => {
                        // SIZED BY THE STRIDE. A `value` that is a view of the projection is read
                        // at `i * value_stride`, so a buffer sized at `m * nkv * head_dim` is
                        // short by a factor of six and the last row reads off the end of it.
                        let st = u64::from(if buffer == "key" {
                            op.shape.map_or(0, |s| s.kv_src_stride.0)
                        } else {
                            op.shape.map_or(0, |s| s.kv_src_stride.1)
                        });
                        let row = if st == 0 { n * k } else { st };
                        get(
                            &mut pool,
                            &format!("{}.{buffer}", op.name),
                            ((m.saturating_sub(1) * row + n * k) * 2) as usize,
                        )?
                    }
                    (DflashOp::AttnBf16Kv | DflashOp::AttnFp8Kv, "q" | "out") => get(
                        &mut pool,
                        &format!("{}.{buffer}", op.name),
                        (m * n * u64::from(hd) * 2) as usize,
                    )?,
                    (DflashOp::SelectorScore, "cand" | "anchor") => get(
                        &mut pool,
                        &format!("{}.{buffer}", op.name),
                        (m * r * 8) as usize,
                    )?,
                    (DflashOp::SelectorScore, "unary") => get(
                        &mut pool,
                        &format!("{}.unary", op.name),
                        (m * r * 4) as usize,
                    )?,
                    (DflashOp::SelectorScore, "hidden") => get(
                        &mut pool,
                        &format!("{}.hidden", op.name),
                        (m * k * 2) as usize,
                    )?,
                    (DflashOp::SelectorScore, "out") | (DflashOp::SelectorWalk, "scores") => get(
                        &mut pool,
                        "selector.scores",
                        (u64::from(cfg.selector_positions()) * r * r * 4) as usize,
                    )?,
                    (DflashOp::SelectorWalk, "candidates") => {
                        get(&mut pool, "walk.candidates", (n * r * 8) as usize)?
                    }
                    (DflashOp::SelectorWalk, "req_state") => {
                        get(&mut pool, "walk.req_state", (m * n * 4) as usize)?
                    }
                    (DflashOp::SelectorWalk, "tokens") => {
                        get(&mut pool, "walk.tokens", (m * n * 8) as usize)?
                    }
                    (DflashOp::SelectorWalk, "realized") => {
                        get(&mut pool, "walk.realized", (m * n * r * 4) as usize)?
                    }
                    // The head's own three: the row pad reads the LIVE rows and writes the
                    // padded ones, the group sum reads the padded ones, and the candidate
                    // gather maps the selection's positions onto the head's `bidx`.
                    (DflashOp::HeadRowPad, "x") => {
                        get(&mut pool, "head.pad.x", (r * n * 2) as usize)?
                    }
                    (DflashOp::HeadRowPad, "out") => {
                        get(&mut pool, "head.int2.x", (m * n * 2) as usize)?
                    }
                    (DflashOp::HeadGroupSum, "x") => {
                        get(&mut pool, "head.int2.x", (m * n * 2) as usize)?
                    }
                    (DflashOp::HeadGroupSum, "xs") => get(
                        &mut pool,
                        "head.int2.xs",
                        (m * n.div_ceil(128) * 4) as usize,
                    )?,
                    // The two row gathers: `k` source rows of `n` BYTES in, `m` rows out.
                    (DflashOp::RowGather, "in") => {
                        get(&mut pool, &format!("{}.in", op.name), (k * n) as usize)?
                    }
                    (DflashOp::RowGather, "idx") => {
                        get(&mut pool, &format!("{}.idx", op.name), (m * 8) as usize)?
                    }
                    (DflashOp::RowGather, "out") => {
                        get(&mut pool, &format!("{}.out", op.name), (m * n) as usize)?
                    }
                    (DflashOp::HeadCandGather, "bidx") => {
                        get(&mut pool, "head.topk.bidx", (m * n * 4) as usize)?
                    }
                    (DflashOp::HeadCandGather, "pos") => {
                        get(&mut pool, "head.topk.idx", (m * r * 4) as usize)?
                    }
                    (DflashOp::HeadCandGather, "out") => {
                        get(&mut pool, "head.topk.ids", (m * r * 4) as usize)?
                    }
                    // The drafter's q/k norm + rope. `r` is the head width here, `n` the q heads
                    // and `k` the kv heads, so the projection is `(n + 2k) * r` wide -- the same
                    // derivation `DflashShape::qk_rope` makes, because the drafter's qkv is a
                    // plain [q | k | v]. The cos/sin table is the CONTEXT store's record (one
                    // rope cache per model, and `_build_fused_kv_buffers` aliases layer 0's), so
                    // it is taken from that op's weights the way the conv's `base` is taken from
                    // its sibling's.
                    (DflashOp::QkNormRope, "qkv") => get(
                        &mut pool,
                        &format!("{}.qkv", op.name),
                        (m * (n + 2 * k) * r * 2) as usize,
                    )?,
                    (DflashOp::QkNormRope, "pos") => get(&mut pool, "qk.pos", (m * 8) as usize)?,
                    (DflashOp::QkNormRope, "q_out") => get(
                        &mut pool,
                        &format!("{}.q_out", op.name),
                        (m * n * r * 2) as usize,
                    )?,
                    (DflashOp::QkNormRope, "k_out") => get(
                        &mut pool,
                        &format!("{}.k_out", op.name),
                        (m * k * r * 2) as usize,
                    )?,
                    // The two per-head norm weights bind BY POSITION, because the checkpoint
                    // calls both of them `weight` and the record map keys on the last
                    // dot-segment: `layer.<i>.attn.q_norm.weight` and `...k_norm.weight` are one
                    // kernarg name for two slots. The plan pushes q_norm then k_norm, which is
                    // `qw` then `kw`, and `dflash_graph`'s record test pins that order.
                    (DflashOp::QkNormRope, "qw" | "kw") => ws
                        .get(usize::from(buffer == "kw"))
                        .map(|(_, a)| *a)
                        .ok_or_else(|| {
                            anyhow::anyhow!("{}: no record #{buffer} for the q/k norm", op.name)
                        })?,
                    (DflashOp::QkNormRope, "cos_sin") => weights_of
                        .get("context_kv.rope")
                        .and_then(|w| w.iter().find(|(nm, _)| *nm == "cos_sin"))
                        .map(|(_, a)| *a)
                        .ok_or_else(|| {
                            anyhow::anyhow!("the pack carries no `context_kv.rope.cos_sin`")
                        })?,
                    _ => anyhow::bail!("{}: no buffer for `{buffer}`", op.name),
                };
                binds.push((buffer.clone(), a));
            }
            planned.push(Planned {
                name: op.name.clone(),
                desc,
                shape,
                binds,
            });
        }
    }

    let issue = |launcher: &mut Launcher, p: &Planned| -> anyhow::Result<()> {
        let mut b = ArgBindings::new(Shape::new(p.shape.m, p.shape.n, p.shape.k));
        for (nm, a) in &p.binds {
            b.bind(nm, *a);
        }
        launcher.dispatch_one(ctx, &p.desc, &b, stream)?;
        Ok(())
    };
    // THE FIRST PASS SYNCS AFTER EVERY LAUNCH, and that is the whole reason it is a separate
    // loop. A memory fault takes the HIP context with it, so a set issued back to back reports
    // `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` for the SET and names nothing -- which is what
    // round 21 had, and it is why the buffer that was actually wrong (the context rope's
    // positions, read 5x past their upload) took a host reading of the reference to find rather
    // than a run. One sync per launch costs ~5 us x 96 once and turns that into an op name.
    for p in &planned {
        issue(launcher, p)
            .and_then(|()| {
                ctx.hip.sync_stream(stream)?;
                Ok(())
            })
            .map_err(|e| {
                anyhow::anyhow!(
                    "{}: {e}\n   descriptor {} grid {:?} block {:?}, shape m{} n{} k{} r{} \
                 block_size {} bt_stride {}\n   bound: {:?}",
                    p.name,
                    p.desc.kernel,
                    p.desc.grid,
                    p.desc.block,
                    p.shape.m,
                    p.shape.n,
                    p.shape.k,
                    p.shape.r,
                    p.shape.block_size,
                    p.shape.bt_stride,
                    p.binds.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>()
                )
            })?;
    }
    // ---- the whole set, and every launch INSIDE it ------------------------------------------
    //
    // PER-LAUNCH TIMING IS DONE IN PLACE, not by timing each kernel on its own, and the
    // difference is the LEDGER's oldest measurement rule: a solo repeated launch reads its
    // weight out of the 64 MB Infinity Cache and reports a number the proposal never sees. So
    // the events bracket each launch WHERE IT SITS in the set -- same order, same cache state,
    // same clocks -- and the whole-set wall time is taken by the same loop. The per-launch sum
    // is therefore slightly under the wall figure (the gaps between launches are the host's) and
    // both are printed, because the difference between them is the launch gap and it is a
    // number somebody will want.
    let ev: Vec<(saber_hip::Event, saber_hip::Event)> = planned
        .iter()
        .map(|_| {
            Ok((
                ctx.hip.create_event(saber_hip::EventFlags::Default)?,
                ctx.hip.create_event(saber_hip::EventFlags::Default)?,
            ))
        })
        .collect::<anyhow::Result<_>>()?;
    let mut per: Vec<Vec<f64>> = vec![Vec::with_capacity(iters as usize); planned.len()];
    // TWO PASS SETS, and the first one is the number to put beside prod's.
    //
    // The instrumented set records two events per launch, and on this box that is ~4 us of host
    // time per launch: F28 measured the whole set at 5890.8 us of WALL with the events in and
    // 5481.8 us of summed DEVICE time, against F27's 5137.3 us for the same set with no events
    // at 96 launches. An arm that changes what it measures is not a comparison, so the clean
    // passes run first and own `med`, and the instrumented ones only own the per-launch table.
    let mut samples = Vec::new();
    for _ in 0..iters {
        let t0 = std::time::Instant::now();
        for p in &planned {
            issue(launcher, p)?;
        }
        ctx.hip.sync_stream(stream)?;
        samples.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    let mut inst = Vec::new();
    for _ in 0..iters {
        let t0 = std::time::Instant::now();
        for (i, p) in planned.iter().enumerate() {
            ctx.hip.record_event(&ev[i].0, stream)?;
            issue(launcher, p)?;
            ctx.hip.record_event(&ev[i].1, stream)?;
        }
        ctx.hip.sync_stream(stream)?;
        inst.push(t0.elapsed().as_secs_f64() * 1e6);
        for (i, e) in ev.iter().enumerate() {
            per[i].push(f64::from(ctx.hip.elapsed_ms(&e.0, &e.1)?) * 1000.0);
        }
    }
    samples.sort_by(f64::total_cmp);
    inst.sort_by(f64::total_cmp);
    let med = samples[samples.len() / 2];
    let med_inst = inst[inst.len() / 2];
    let uncovered: u32 = graph.uncovered().iter().map(|(_, _, n)| *n).sum();
    println!(
        "   proposal  {} of the plan's {} launches ({} absent, {} refused for an unbound scalar          kernarg), {:.1} us (median of {})",
        planned.len(),
        graph.total_launches(),
        uncovered,
        refused,
        med,
        samples.len()
    );
    println!(
        "     every launch of the proposal that a `dflash` code object covers, at its real shape \
         with its real weights. The activations between stages are still SCRATCH, so this is a \
         TIME and not a chain -- `--no-seed` is the chain, and it is a different claim"
    );

    // ---- the table, sorted by us ---------------------------------------------------------------
    let mut rows: Vec<(f64, usize)> = per
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let mut v = v.clone();
            v.sort_by(f64::total_cmp);
            (v[v.len() / 2], i)
        })
        .collect();
    let total: f64 = rows.iter().map(|(u, _)| u).sum();
    rows.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!(
        "   per-launch (median of {iters}, events around each launch in its place in the set): \
         {total:.1} us of device time in {} launches; that instrumented pass walls at \
         {med_inst:.1} us against the clean {med:.1}, and the difference is the events",
        planned.len()
    );
    println!(
        "     {:<4} {:<34} {:<38} {:>9} {:>6}",
        "#", "op", "kernel", "us", "% "
    );
    let mut over: Vec<String> = Vec::new();
    for (rank, (us, i)) in rows.iter().enumerate() {
        let p = &planned[*i];
        if *us > 100.0 {
            over.push(format!("{} ({}) {us:.1} us", p.name, p.desc.kernel));
        }
        // The whole table is printed, not a head: a launch that is not in it is a launch nobody
        // can ask about later, and 112 lines once a round is cheaper than a second lock hold.
        println!(
            "     {:<4} {:<34} {:<38} {us:>9.1} {:>5.1}%",
            rank,
            p.name,
            p.desc.kernel,
            100.0 * us / total.max(1e-9)
        );
    }
    if over.is_empty() {
        println!("     no single launch is over 100 us");
    } else {
        println!("     OVER 100 us, for the rules agent: {}", over.join("; "));
    }
    Ok(())
}

/// The rerank op the pack's `head.rerank` records select (N6): `w8` bound -> the fp8 record's
/// [`DflashOp::RerankExactW8`], else the bf16 arm. The records decide, never a flag.
fn rerank_op(ws: Option<&[(&str, DeviceAddr)]>) -> DflashOp {
    if ws.is_some_and(|ws| ws.iter().any(|(k, _)| *k == "w8")) {
        DflashOp::RerankExactW8
    } else {
        DflashOp::RerankExactBf16
    }
}

#[cfg(test)]
mod tests {
    use super::{bf16_to_f32, deinterleave, f32_to_bf16_rne, kv_elem, Capture};

    /// The KV cache's element width comes from the LAST DIMENSION, not from the dtype name.
    /// The reference server allocates an fp8 cache as `uint8` and views it as `fp8` at the point of use, so the
    /// capture's manifest says `uint8` for a one-byte cache — and the reading this replaced
    /// (`dtype.contains("float8")`) calls that two bytes, de-interleaves at twice the row width
    /// and copies half the bytes in range and in silence. The CONTROL is that old reading, and it
    /// must get this case wrong or the fix is not a fix.
    #[test]
    fn an_fp8_kv_cache_stored_as_uint8_is_one_byte_per_element() {
        let man = serde_json::json!({
            "kv.fp8":  {"kind": "tensor", "dtype": "uint8",    "shape": [2, 8, 1648, 256]},
            "kv.bf16": {"kind": "tensor", "dtype": "bfloat16", "shape": [2, 8, 16, 256]},
            "kv.odd":  {"kind": "tensor", "dtype": "uint8",    "shape": [2, 8, 16, 100]},
        });
        let cap = Capture {
            dir: std::path::PathBuf::from("."),
            tag: "t".to_string(),
            man,
        };
        assert_eq!(
            kv_elem(&cap, "kv.fp8", 128).unwrap(),
            1,
            "256 uint8 = 2 x 128 fp8"
        );
        assert_eq!(
            kv_elem(&cap, "kv.bf16", 128).unwrap(),
            2,
            "256 bf16 = 2 x 128 bf16"
        );
        assert!(
            kv_elem(&cap, "kv.odd", 128).is_err(),
            "a last dimension that is not a whole number of 2 x head_dim elements is a refusal, \
             not a rounded guess"
        );
        let by_dtype_name = |d: &str| usize::from(!d.contains("float8")) + 1;
        assert_ne!(
            by_dtype_name("uint8"),
            1,
            "the control must get the fp8-as-uint8 cache wrong, or it proves nothing"
        );
    }

    /// The de-interleave is a permutation and nothing else: every byte of prod's cache lands
    /// exactly once, K before V within a head's row. The CONTROL is the same call with `kv_heads`
    /// and `block_size` transposed — the mistake a reader of `[blocks, kv_heads, block_size,
    /// 2*head_dim]` makes — and it must NOT produce the same bytes, or this test would pass on a
    /// layout that silently takes another head's keys.
    #[test]
    fn deinterleave_is_a_permutation_and_its_transpose_is_not() {
        let (blocks, heads, bs, hd, elem) = (2usize, 3usize, 4usize, 2usize, 1usize);
        let n = blocks * heads * bs * 2 * hd * elem;
        let src: Vec<u8> = (0..n).map(|i| u8::try_from(i % 251).unwrap_or(0)).collect();
        let (k, v) = deinterleave(&src, blocks, heads, bs, hd, elem);
        assert_eq!(k.len() + v.len(), src.len());
        let mut seen = k.clone();
        seen.extend_from_slice(&v);
        seen.sort_unstable();
        let mut want = src.clone();
        want.sort_unstable();
        assert_eq!(seen, want, "every source byte lands exactly once");
        for b in 0..blocks {
            for h in 0..heads {
                for t in 0..bs {
                    for e in 0..hd {
                        let s = ((b * heads + h) * bs + t) * 2 * hd + e;
                        let d = ((b * bs + t) * heads + h) * hd + e;
                        assert_eq!(k[d], src[s], "K at b{b} h{h} t{t} e{e}");
                        assert_eq!(v[d], src[s + hd], "V at b{b} h{h} t{t} e{e}");
                    }
                }
            }
        }
        let (k2, _) = deinterleave(&src, blocks, bs, heads, hd, elem);
        assert_ne!(
            k2, k,
            "the transposed read must differ, or this gate has no teeth"
        );
    }

    /// `f32` → `bf16` rounds to nearest EVEN, which is where prod rounds: the rerank writes `f32`
    /// and prod's `y.scatter_` stores `bf16` before its own top-k reads it back as `f32`. The
    /// control is truncation, which agrees on every exactly-representable case and differs on the
    /// tie — so a comparison built on it would look right until it was not.
    #[test]
    fn bf16_rounding_is_nearest_even_not_truncation() {
        for x in [0.0f32, 1.0, -2.5, 65536.0] {
            assert!(
                (bf16_to_f32(f32_to_bf16_rne(x)) - x).abs() < f32::EPSILON,
                "{x} is exact in bf16"
            );
        }
        let even_tie = f32::from_bits(0x3f80_8000);
        assert_eq!(
            f32_to_bf16_rne(even_tie),
            0x3f80,
            "a tie at an even mantissa rounds to even"
        );
        let odd_tie = f32::from_bits(0x3f81_8000);
        assert_eq!(
            f32_to_bf16_rne(odd_tie),
            0x3f82,
            "a tie at an odd mantissa rounds up"
        );
        let truncated = u16::try_from(odd_tie.to_bits() >> 16).unwrap_or(0);
        assert_ne!(
            truncated,
            f32_to_bf16_rne(odd_tie),
            "truncation must differ on a tie, or the control cannot fail"
        );
    }
}
