//! `saber step-capture` — the whole STEP as one captured graph, and what it costs.
//!
//! This is P3's backend at full scale. `saber block-capture` captured five layers; this captures
//! the step the engine will actually replay: **the embedding, all 64 decoder layers, the row
//! selection, the head and the argmax, at TP=1, in one graph** — then replays it and times it
//! against the same step submitted op by op.
//!
//! # Two plans, because a step is two plans
//!
//! The body runs at the bucket's rows and the head runs at the rows a sampler READS — one per
//! sequence at prefill, every row at decode. That is `saber-engine`'s own shape (`build_head(1)`
//! for prefill) and since 2026-09-20 it is this command's too. It was one plan over every site
//! until then, which gave the head the chunk's row count: 2048 × 248320 logits at `p2048`,
//! 970 MiB, of which one row is read, and ~26 ms of a 630 ms step spent computing the rest.
//! `--head-on-every-row` restores that shape, because the figures it produced are in the ledger
//! and have to stay reproducible.
//!
//! > Plan → captured HIP graphs per bucket (batch × spec width × ctx band).
//! >
//! > — ARCHITECTURE.md, *4. Execution plane*
//!
//! # The bucket
//!
//! `b1s5` is prod's own decode bucket: one sequence, five verify rows (`SPEC=4`+1), context band
//! 8192. `b8s5` is the same step at eight sequences (M = 40). Both are planned as a
//! [`PlanOptions::BODY`] and a [`PlanOptions::HEAD`], and at a decode bucket the head's rows are
//! the body's — every verify row is a sampled position — so the split costs an arena of one row
//! of logits and the three launches between the two plans, and nothing else.
//!
//! # What is real and what is not
//!
//! Real: every weight out of the device-final pack, every launch the graph declares, the GDN conv
//! and recurrent caches and the paged KV cache as actual device state the kernels rewrite in
//! place, the arena, the node count, and the microseconds.
//!
//! Not real: **the numerics**. Three things have no code object on this branch — `qk_norm_rope`,
//! `attn_gate_quant`, and the append of K/V into the paged cache, which the graph does not model
//! as an op at all — so the 16 attention layers do not compute their real output and the KV cache
//! is read as it was seeded. `saber forward` says the same thing about the same three, and this
//! run prints it every time. **The launch structure and the timing are what this measures**, and
//! the comparison the P-1b comparator asks for is a comparison of those.
//!
//! # The three claims
//!
//! 1. **The node count is the budget**: `hipGraphGetNodes` = `BlockPlan::expected_launches()`.
//! 2. **graph == eager**, byte for byte, over the WHOLE arena and every state buffer, on three
//!    replays, each from a freshly reset state.
//! 3. **Every Persistent region was prepared before the capture opened**
//!    (`saber_exec::check_persistent_ready`), which for a 675-node graph is 64 layers' worth of
//!    split-K counters and GDN barriers rather than the six of a five-layer block.
//!
//! GPU rules (CLAUDE.md): one card under the card's lock (`~/saber/.cache/gpu.lock.<card>`, via `tools/capture/gpu_idle.sh`), **GPU idle** (prod is
//! down for this session; a refused `localhost:8080/metrics` is the idle check and the run prints
//! it), every wait deadline-bounded, power cap untouched.

use std::collections::BTreeMap;
use std::path::PathBuf;

use saber_exec::block_capture::{bind_steps, check_capturable, check_persistent_ready};
use saber_exec::forward::i32_bytes;
use saber_exec::layer::{
    BlockExecutor, BlockInputs, BlockPlan, BoundWeight, PlanOptions, Route, RuntimeParams, Step,
};
use saber_exec::tp2::wait_bounded;
use saber_exec::{
    ArgBindings, CapturableOp, CaptureCounters, CapturedBlock, CapturedBucket, GraphSafe,
    LaunchCounters, LaunchCtx, Launcher, LoadCounters, PluginSet,
};
use saber_hip::{DeviceAddr, DevicePtr, EventFlags, HipThread, Stream, StreamFlags};
use saber_kernels::rules::LaunchRules;
use saber_kernels::saber::SaberRules;
use saber_kernels::Shape;
use saber_mem::dio::OpenMode;
use saber_mem::ledger::{Ledger, Pool};
use saber_mem::pack::PackReader;
use saber_mem::upload::{HipSink, PinnedRing, Upload, UploadConfig};
use saber_model::graph::{Bucket, Geometry, Graph, KernelOp, ModelConfig, Op, Phase, Site};

use crate::layer::{pack_file, to_device};

/// The comparator this whole subcommand exists to be read against.
///
/// P-1b: prod's stack (the reference server + the reference, NVFP4 tier) at SPEC=0, greedy, one stream, GPU idle.
/// It is a **different step** — no speculation, so one row instead of five — and it computes the
/// model's real output, which this does not yet. It is printed beside the measurement because a
/// number with no comparator is not a result, and hidden because a comparison that is not stated
/// is a comparison someone will make wrongly later.
const PROD_MS_PER_STEP: f64 = 18.7;

/// Arguments of `saber step-capture`.
pub struct StepCaptureArgs {
    /// The TP=1 device-final tier pack (directory or `rank<r>.pack`).
    pub pack: PathBuf,
    /// P27-16: the verify width a decode `--rows` is cut by (`None` = the graph's own).
    pub verify_width: Option<u32>,
    /// TP rank of the pack.
    pub rank: u32,
    /// A JSON array of token ids, or a `.jsonl` corpus whose rows carry `token_ids`.
    pub prompt_ids: PathBuf,
    /// Row of the corpus to take.
    pub prompt_index: usize,
    /// Directory of committed `.hsaco` + `*.manifest.toml`.
    pub prebuilt: PathBuf,
    /// Rows per step: 5 is b1s5, 40 is b8s5. Both are decode buckets at verify width 5.
    pub rows: Vec<u32>,
    /// Prefill buckets to run instead, by chunk length: 2048 is `p2048`, 8192 is `p8192`.
    ///
    /// A prefill step is a DIFFERENT op set — the query-tiled attention kernel, the three-kernel
    /// GDN chunk path and the standalone gated norm, none of which a decode step launches — at one
    /// sequence and no speculative width. Everything else here is shared: the same `BlockPlan`,
    /// the same index-table law (`saber_exec::forward` has carried `Phase::Prefill` from the
    /// start), the same capture, replay, byte-compare and per-op timing. Added by I15 (the
    /// autotune plane) to measure a prefill op IN CHAIN for the first time; ws-h18 owns this file
    /// and is told.
    pub prefill: Vec<u32>,
    /// Sequences whose chunks share ONE prefill step.
    ///
    /// `--prefill 1024 --prefill-seqs 2` is two sequences of 1024 tokens in a single 2048-row
    /// step: the L9 rule picks the bucket by TOTAL rows, the `cu` prefix says where one chunk ends
    /// and the next begins, and the head runs on the last row of each chunk. `1` is the shape the
    /// engine runs today, one chunk per step.
    pub prefill_seqs: u32,

    /// Plan the head on every row of the chunk, in ONE combined plan, instead of the engine's
    /// body-plus-head split on the sampled rows.
    ///
    /// This command's original shape, kept because the figure it produces is in the ledger and
    /// has to stay reproducible. It is not what the engine runs: `saber-engine` plans the head at
    /// `build_head(1)` for prefill, so a p2048 step measured this way carries ~26 ms of logits
    /// nobody reads (`docs/LEDGER.md`, workstream N5).
    pub head_on_every_row: bool,

    /// Context band the paged KV cache is sized for.
    pub ctx: u32,
    /// Refuse to start if the step's device buffers would exceed this.
    pub budget_mib: u64,
    /// HIP device ordinal within the visible set.
    pub device: i32,
    /// Replays to byte-compare against the eager step.
    pub replays: usize,
    /// Timed graph/eager pairs.
    pub iters: usize,
    /// Untimed pairs first.
    pub warmup: usize,
    /// Capture the step as a chain of per-layer sub-graphs and time each layer IN PLACE.
    pub per_layer: bool,
    /// Rebuild the step with one family's ops omitted, per family, and time it.
    pub ablate: bool,
    /// Time every op as its own sub-graph, in place, and report achieved GB/s per GEMM shape.
    pub per_op: bool,
    /// With `--per-op`: detail every link whose OP NAME contains this, with the ops either side
    /// and the same op timed eagerly in place. The link-cost column is what it is for.
    pub detail: Option<String>,
    /// The DRAM peak this box actually has, for the weight-stream floor.
    ///
    /// 635 GB/s measured, not the 576 the startup sweep reports
    /// (`docs/LEDGER.md`, `rdna4-dram-peak-635`).
    pub dram_gbs: f64,
    /// The arithmetic peak the W4A8 path can reach, for the compute floor.
    ///
    /// 325 TFLOP/s: RDNA4's fp8 WMMA is 2x f16's 160 (`docs/LEDGER.md`, `rdna4-fp8-wmma-2x`), and
    /// the MXFP4 kernels upconvert into that same fp8 path. Which of the two floors is higher is
    /// what the `bound` column names, and at a prefill M it is this one.
    pub peak_tflops: f64,
    /// FN63: replay the step through ROCr AQL packets beside the HIP graph, then stop.
    pub fn63: Option<crate::fn63_hsa::Fn63Opts>,
}

/// The prompt's token ids, from a bare JSON array or a corpus row.
pub(crate) fn read_prompt_ids(path: &std::path::Path, index: usize) -> anyhow::Result<Vec<i32>> {
    let text =
        std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let ids: Vec<i64> = if text.trim_start().starts_with('[') {
        serde_json::from_str(&text)?
    } else {
        let line = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .nth(index)
            .ok_or_else(|| anyhow::anyhow!("{}: no row {index}", path.display()))?;
        let row: serde_json::Value = serde_json::from_str(line)?;
        row.get("token_ids")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("{} row {index}: no `token_ids`", path.display()))?
            .iter()
            .filter_map(serde_json::Value::as_i64)
            .collect()
    };
    if ids.is_empty() {
        anyhow::bail!("{}: the prompt is empty", path.display());
    }
    Ok(ids.into_iter().map(|v| v as i32).collect())
}

/// The index tables a DECODE step needs — the shared laws (`saber_exec::forward`).
///
/// They were stated here until ws-h15, and they were the weaker copy: this one handed the GDN
/// kernels state slot 0, which is their NULL slot, so every step measured before ws-h15 launched
/// all 48 GDN layers and updated no recurrent state. The launch structure and the microseconds are
/// unaffected — the kernels ran, read their caches and wrote their outputs — but the state traffic
/// was one slot short of real. The shared law numbers slots from
/// [`saber_exec::forward::FIRST_STATE_SLOT`], and the caches this command sizes
/// (`batch * (q + 2)`) already cover `1 + batch * q`.
pub(crate) fn index_table(
    key: &str,
    phase: Phase,
    batch: u32,
    q: u32,
    kv_blocks_per_seq: u32,
    ctx: u32,
    ids: &[i32],
) -> anyhow::Result<Vec<u8>> {
    use saber_exec::forward::{SeqFacts, StepIndex, FIRST_STATE_SLOT};
    let rows = (batch * q) as usize;
    let cycled: Vec<i32> = ids.iter().copied().cycle().take(rows).collect();
    // Each sequence holds its own pages, and its own `q` consecutive state slots from the first
    // real one — the recurrent kernel reads `sidx[n * stride + t]` for every candidate `t`.
    let pages: Vec<Vec<u32>> = (0..batch)
        .map(|n| (n * kv_blocks_per_seq..(n + 1) * kv_blocks_per_seq).collect())
        .collect();
    let seqs: Vec<SeqFacts<'_>> = (0..batch)
        .map(|n| SeqFacts {
            // A prefill chunk scans through ONE state per sequence, not one per row (the 09-14
            // Infinity-Cache cliff is about speculative state, and a chunk has no candidates), so
            // the slots are not strided by `q` there.
            state_slot: FIRST_STATE_SLOT
                + match phase {
                    Phase::Prefill => n,
                    Phase::Decode => n * q,
                },
            // A decode step reads a context that is already resident; `seqused_k` is that plus this
            // step's rows, which is what the band was sized for. A prefill chunk is the first
            // thing in its sequence: nothing is done before it.
            done_before: match phase {
                Phase::Prefill => 0,
                Phase::Decode => ctx.saturating_sub(q),
            },
            kv_blocks: &pages[n as usize],
            live_rows: q,
            rope: None,
        })
        .collect();
    let step = StepIndex {
        phase,
        rows: batch * q,
        max_blocks: kv_blocks_per_seq,
        seqs: &seqs,
        ids: &cycled,
        band_width: None,
    };
    step.check().map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(saber_exec::forward::index_table(key, &step)?)
}

/// What one instance of an op must move, and the arithmetic it must do, derived from the graph's
/// OWN geometry -- never from the measurement.
///
/// The GEMM table gets its bytes from the pack's record lengths, which is exact because a linear's
/// traffic IS its weight. A non-linear op's traffic is its activations, and nothing in the pack
/// describes those, so until now the non-linear table reported time with no floor beside it: the
/// reader could see that `silu_mul_quant` costs 34 ms of a `p2048` step and not whether that is
/// 40% or 99% of what moving the bytes costs.
///
/// Every row here is an arithmetic identity on the op's declared shape -- an elementwise producer
/// reads its inputs once and writes its outputs once, and attention does `4 * head_dim` flops per
/// (query, key) pair per head. **`None` is the honest answer for the rest** (the GDN core's conv,
/// KKT and chunk scan, whose traffic is a state-blocking strategy inside saber, and the gathers):
/// a modelled number nobody can derive from the graph is worse than a blank, because it would be
/// read as a floor. Those rows print `?` and are named as kernel work rather than rules work.
///
/// The count is a LOWER bound in one direction only: an op that re-reads an input (attention's
/// K/V per query tile) moves more than this, never less. So "at or near the floor" is a real
/// verdict and "far above it" is a question, which is the asymmetry a floor is for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Traffic {
    /// Bytes read plus bytes written, each counted once.
    pub bytes: u64,
    /// Flops, where the op is a matmul. 0 otherwise.
    pub flops: u64,
}

/// [`Traffic`] for one op at `bucket`, `weight_bytes` being what its weight records hold (the
/// GEMM case; 0 elsewhere).
#[allow(clippy::match_same_arms)]
pub(crate) fn op_traffic(op: &Op, bucket: Bucket, weight_bytes: u64) -> Option<Traffic> {
    let m = u64::from(bucket.m());
    // Rows per sequence and how many sequences: attention counts (query, key) pairs per sequence.
    let (t, nseq) = match bucket {
        Bucket::Prefill { tokens, seqs } => (u64::from(tokens / seqs.max(1)), u64::from(seqs)),
        Bucket::Decode {
            batch,
            verify_width,
        } => (u64::from(verify_width), u64::from(batch)),
        // A row band (C2): the mean rows per sequence over the band, which is what its attention
        // reads in aggregate (the padding rows are skipped by `cu`).
        Bucket::DecodeBand { batch, rows, .. } => {
            (u64::from(rows / batch.max(1)).max(1), u64::from(batch))
        }
        // P27-15d: one sequence's tree rows (each row reads its ancestors' keys + the context).
        Bucket::DecodeTree { rows } => (u64::from(rows), 1),
    };
    let bytes = |b: u64| Some(Traffic { bytes: b, flops: 0 });
    match op.geometry {
        // A linear: the weight is the traffic and `2 * m * n * k` is the arithmetic. Only the head
        // reaches here -- a layer linear is in the GEMM table.
        Geometry::Gemm { n, k, .. } => Some(Traffic {
            bytes: weight_bytes + m * u64::from(k) + m * u64::from(n) * 2,
            flops: 2 * m * u64::from(n) * u64::from(k),
        }),
        // y and residual in (bf16), residual and the e4m3 quant out, one fp32 scale per row, the
        // weight row once.
        Geometry::Rms { k } => bytes(m * u64::from(k) * 7 + u64::from(k) * 2 + m * 4),
        // `gate_up`'s 2n bf16 columns in, n e4m3 out, one scale per row.
        Geometry::SiluMul { n } => bytes(m * u64::from(n) * 5 + m * 4),
        // x and z are both bf16 and n wide (z is strided out of the projection -- the same bytes
        // are touched), q is e4m3, and there is one fp32 scale per HEAD row.
        Geometry::GdnNorm { n, channels, .. } => {
            let (n, c) = (u64::from(n), u64::from(channels).max(1));
            bytes(m * n * 5 + c * 2 + m * (n / c) * 4)
        }
        // q and k read out of the projection and written compact, both bf16; the two norm weight
        // rows; one cos/sin row per position.
        Geometry::QkNorm {
            head_dim,
            q_heads,
            kv_heads,
            rotary_dim,
        } => {
            let hd = u64::from(head_dim);
            let w = (u64::from(q_heads) + u64::from(kv_heads)) * hd;
            bytes(m * w * 4 + 2 * hd * 2 + m * u64::from(rotary_dim) * 2)
        }
        // The attention output and the gate, both bf16 and n wide; q e4m3; a scale per row.
        Geometry::GateQuant { n, .. } => bytes(m * u64::from(n) * 5 + m * 4),
        Geometry::Attn {
            head_dim,
            q_heads,
            kv_heads,
            ..
        } => {
            let hd = u64::from(head_dim);
            match op.kernel {
                // K compact and V strided in, both bf16; both written as e4m3 into the page.
                KernelOp::KvAppend => bytes(m * u64::from(kv_heads) * hd * 6),
                // `4 * head_dim` flops per (query, key) pair per query head: QK^T and PV, two
                // flops each. A prefill chunk is the whole sequence (`done_before = 0`), so the
                // causal pair count is `t * (t + 1) / 2`; a decode row sees the resident context.
                KernelOp::AttnPrefillPaged => Some(Traffic {
                    bytes: 0,
                    flops: 4 * hd * u64::from(q_heads) * nseq * (t * (t + 1) / 2),
                }),
                _ => None,
            }
        }
        // The gather reads a row of the table and writes it.
        Geometry::Embedding { hidden, .. } => bytes(m * u64::from(hidden) * 4),
        // Not derivable from the graph's geometry: see the type's own doc.
        _ => None,
    }
}

/// One line of the non-linear table: a kernel at a shape, with what the graph says it must move.
pub(crate) struct OtherRow {
    /// `producers`, `GDN`, `attention`, `embed+head`.
    pub fam: &'static str,
    /// The symbol the rules chose.
    pub kernel: String,
    /// How many of them are in a step.
    pub count: usize,
    /// Median in-chain microseconds for one, instrument cost already removed.
    pub us: f64,
    /// Bytes ONE of them must move, 0 where the graph cannot say.
    pub bytes: u64,
    /// Flops ONE of them must do, 0 where it is not a matmul.
    pub flops: u64,
}

impl OtherRow {
    /// The floor one instance cannot go below: its bytes at the DRAM peak or its flops at the
    /// compute roof, whichever is larger. `(mem, cmp, the floor)`, microseconds.
    pub fn floor_us(&self, dram_gbs: f64, peak_tflops: f64) -> (f64, f64, f64) {
        let mem = self.bytes as f64 / (dram_gbs * 1e9) * 1e6;
        let cmp = self.flops as f64 / (peak_tflops * 1e12) * 1e6;
        (mem, cmp, mem.max(cmp))
    }
    /// Milliseconds of the step all of them spend above that floor. A row with no floor spends
    /// all of its time above zero, which is exactly the claim being made about it.
    pub fn above_ms(&self, dram_gbs: f64, peak_tflops: f64) -> f64 {
        let (_, _, f) = self.floor_us(dram_gbs, peak_tflops);
        (self.us - f).max(0.0) * self.count as f64 / 1000.0
    }
}

/// Group the non-GEMM links by (family, kernel, traffic) and attach the median in-chain µs.
///
/// `links` are indices into `plan.steps`, `us` the matching per-link microseconds with the
/// instrument cost already removed. Shared by `step-capture` and `step-tp2` so the two planes
/// report the same table with the same floors.
pub(crate) fn other_rows(
    plan: &BlockPlan<'_>,
    links: &[usize],
    us: &[f64],
    record_len: &BTreeMap<String, u64>,
) -> Vec<OtherRow> {
    /// `(count, bytes, flops, per-instance µs)`.
    type Group = (usize, u64, u64, Vec<f64>);
    let mut groups: BTreeMap<(&'static str, String, u64, u64), Group> = BTreeMap::new();
    for (j, i) in links.iter().enumerate() {
        let step = &plan.steps[*i];
        if family(step) == "GEMMs" {
            continue;
        }
        let Route::Kernel { descs, .. } = &step.route else {
            continue;
        };
        // The head linear's weight is a pack record like any other; nothing else here has one.
        let weight_bytes: u64 = step
            .bindings
            .iter()
            .filter_map(|(_, src)| match src {
                saber_exec::Source::Weight { key, .. } => record_len.get(key).copied(),
                _ => None,
            })
            .sum();
        let t = op_traffic(step.op, plan.bucket, weight_bytes).unwrap_or_default();
        let e = groups
            .entry((family(step), descs[0].kernel.clone(), t.bytes, t.flops))
            .or_insert((0, t.bytes, t.flops, Vec::new()));
        e.0 += 1;
        e.3.push(us[j]);
    }
    groups
        .into_iter()
        .map(|((fam, kernel, _, _), (count, bytes, flops, mut v))| {
            v.sort_by(f64::total_cmp);
            OtherRow {
                fam,
                kernel,
                count,
                us: v[v.len() / 2],
                bytes,
                flops,
            }
        })
        .collect()
}

/// Print the non-linear table, sorted by milliseconds above the floor.
pub(crate) fn print_others(
    rows: &[OtherRow],
    dram_gbs: f64,
    peak_tflops: f64,
    linears_ms: f64,
    whole_ms: f64,
) {
    let mut rows: Vec<&OtherRow> = rows.iter().collect();
    rows.sort_by(|a, b| {
        b.above_ms(dram_gbs, peak_tflops)
            .total_cmp(&a.above_ms(dram_gbs, peak_tflops))
    });
    println!(
        "          {:<10} {:<40} {:>4} {:>9} {:>8} {:>8} {:>5} {:>8} {:>8} {:>8}",
        "family",
        "kernel",
        "n",
        "µs/op",
        "GB/s",
        "TFLOP/s",
        "bound",
        "total ms",
        "floor ms",
        "above ms"
    );
    let (mut total, mut floor_total) = (0.0, 0.0);
    for r in &rows {
        let all = r.us * r.count as f64 / 1000.0;
        let (mem_us, cmp_us, f_us) = r.floor_us(dram_gbs, peak_tflops);
        let floor = f_us * r.count as f64 / 1000.0;
        total += all;
        floor_total += floor;
        let gbs = if r.bytes == 0 {
            String::from("-")
        } else {
            format!("{:.0}", r.bytes as f64 / (r.us * 1e3))
        };
        let tfs = if r.flops == 0 {
            String::from("-")
        } else {
            format!("{:.1}", r.flops as f64 / (r.us * 1e-6) / 1e12)
        };
        let bound = match (r.bytes, r.flops) {
            (0, 0) => "?",
            _ if cmp_us > mem_us => "cmp",
            _ => "mem",
        };
        println!(
            "          {:<10} {:<40} {:>4} {:>9.1} {:>8} {:>8} {:>5} {:>8.3} {:>8.3} {:>8.3}",
            r.fam,
            r.kernel,
            r.count,
            r.us,
            gbs,
            tfs,
            bound,
            all,
            floor,
            r.above_ms(dram_gbs, peak_tflops)
        );
    }
    println!(
        "          {} kernel(s), {total:.3} ms in chain over a {floor_total:.3} ms floor \
         ({:.3} ms above it); with the linears' {linears_ms:.3} ms that is {:.3} ms of the \
         step's {whole_ms:.3}",
        rows.len(),
        total - floor_total,
        total + linears_ms
    );
    println!(
        "          a `?` row has no floor the GRAPH can state (see `op_traffic`): its traffic is \
         a blocking strategy inside the kernel, so it is kernel work, not rules work"
    );
}

/// Which family an op belongs to, for the breakdown.
///
/// By SITE first and kernel second: `lm_head` is a `gemm_w4a8` like every linear in a layer, and
/// counting it with them would hide the one GEMM whose N is the vocabulary.
pub(crate) fn family(step: &Step<'_>) -> &'static str {
    match step.op.site {
        Site::Embedding | Site::Head => "embed+head",
        Site::Layer(_) => match step.op.kernel.id() {
            "gemm_w4a8" => "GEMMs",
            id if id.starts_with("gdn_") => "GDN",
            id if id.starts_with("attn_") => "attention",
            // Its own family, not a producer: at TP=2 it is 129 dispatches of a two-rank protocol
            // and lumping it in with the norms and the quantisers is how an attribution table
            // loses the one op whose cost is a WIRE (ws-t1).
            "allreduce" => "all-reduce",
            _ => "producers",
        },
        _ => "other",
    }
}

/// The per-family split of one step, as a share of the graph's own wall.
///
/// **This is the attribution table.** `other_rows`/`print_others` answer "which op is furthest
/// above its own floor"; this answers "where does the step's time go", which is the question a
/// width or a batch sweep asks. It is printed from the same `(link -> µs)` vector, so the two
/// cannot disagree, and the residual against the graph's wall is shown rather than folded into
/// the biggest family.
pub(crate) fn print_family_summary(
    plan: &BlockPlan<'_>,
    links: &[usize],
    us: &[f64],
    whole_ms: f64,
    label: &str,
) {
    let mut by: BTreeMap<&'static str, (usize, f64)> = BTreeMap::new();
    for (j, i) in links.iter().enumerate() {
        let e = by.entry(family(&plan.steps[*i])).or_insert((0, 0.0));
        e.0 += 1;
        e.1 += us[j];
    }
    let sum_ms: f64 = by.values().map(|(_, u)| u / 1000.0).sum();
    println!();
    println!(
        "families  {label}: per-op medians grouped, instrument removed, against the captured \
         step's own {whole_ms:.3} ms"
    );
    println!(
        "          {:<12} {:>6} {:>10} {:>8}",
        "family", "ops", "ms", "% step"
    );
    let mut rows: Vec<(&str, usize, f64)> =
        by.iter().map(|(k, (n, u))| (*k, *n, u / 1000.0)).collect();
    rows.sort_by(|a, b| b.2.total_cmp(&a.2));
    for (name, n, ms) in &rows {
        println!(
            "          {name:<12} {n:>6} {ms:>10.3} {:>7.1}%",
            ms / whole_ms.max(1e-9) * 100.0
        );
    }
    println!(
        "          {:<12} {:>6} {:>10.3} {:>7.1}%   (sum of the families)",
        "TOTAL",
        links.len(),
        sum_ms,
        sum_ms / whole_ms.max(1e-9) * 100.0
    );
    println!(
        "          {:<12} {:>6} {:>10.3} {:>7.1}%   (ZERO BY CONSTRUCTION: the per-link instrument \
         cost is calibrated so the links sum to the graph's own wall, so this line is an \
         arithmetic check and NOT a measurement of drain)",
        "residual",
        "",
        whole_ms - sum_ms,
        (whole_ms - sum_ms) / whole_ms.max(1e-9) * 100.0
    );
}

/// Run the capture. Returns the number of failures.
pub fn run(hip: &HipThread, args: &StepCaptureArgs) -> anyhow::Result<usize> {
    hip.set_device(args.device)?;
    let mut graph = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 1)?;
    if let Some(w) = args.verify_width {
        graph.verify_width = w;
    }
    let mxfp4 = LaunchRules::prod_mxfp4()?;
    let saber = SaberRules::prod_saber()?;
    let ids = read_prompt_ids(&args.prompt_ids, args.prompt_index)?;

    let load_counters = LoadCounters::new();
    let plugins = PluginSet::load_dir(hip, &args.prebuilt, &load_counters)?;
    let functions = plugins.functions(hip)?;
    let ctx = LaunchCtx::new(hip, &functions);
    let stream = hip.create_stream(StreamFlags::NonBlocking)?;
    let layers: Vec<u32> = (0..graph.config.num_hidden_layers).collect();

    println!(
        "model     Qwen3.8-27B at TP=1: {} layers, hidden {}, ctx band {}",
        graph.config.num_hidden_layers, graph.config.hidden_size, args.ctx
    );
    println!(
        "plugins   {} loaded ({} symbols); {}",
        plugins.plugins().len(),
        functions.len(),
        load_counters.snapshot().line()
    );

    // The shared idle check: VRAM first, /metrics second (`crate::gpu_idle`). `need` is not known
    // until the weight window has been measured, so this call reports and refuses on the request
    // count, and the weight window is checked against the free bytes it returns.
    let idle = crate::gpu_idle::require(hip, args.device, 0, 0, crate::gpu_idle::DEFAULT_METRICS)?;
    let free = idle.free;

    // ---- every weight of the whole model, once ------------------------------------------------
    // The probe only needs A plan, to learn which weight records the step binds; the op SET
    // differs between the phases but the weights do not, so the first bucket asked for will do.
    let (probe_rows, probe_phase) = match (args.rows.first(), args.prefill.first()) {
        (Some(r), _) => (*r, Phase::Decode),
        (None, Some(t)) => (*t, Phase::Prefill),
        (None, None) => anyhow::bail!("nothing to run: pass --rows or --prefill"),
    };
    // The COMBINED plan, whichever shape the run will use: it names every weight the step binds,
    // the head's included, in one list. Asking the body alone would leave `lm_head.weight` out of
    // the upload and the head would then bind nothing.
    let probe = plan_combined(
        &graph,
        &layers,
        graph.bucket_for(probe_rows, probe_phase),
        args.ctx,
        probe_phase,
        &mxfp4,
        &saber,
    )?;
    let want = probe.weight_keys();
    let reader = PackReader::open(&pack_file(&args.pack, args.rank))?;
    let mut layout_of: BTreeMap<String, u32> = BTreeMap::new();
    let mut idx: Vec<usize> = Vec::new();
    for (i, rec) in reader.records().iter().enumerate() {
        let name = reader.name(i);
        if want.contains(name) {
            idx.push(i);
            layout_of.insert(name.to_owned(), rec.layout_id);
        }
    }
    if layout_of.len() != want.len() {
        let missing: Vec<&String> = want
            .iter()
            .filter(|k| !layout_of.contains_key(*k))
            .collect();
        anyhow::bail!(
            "{} record(s) this step binds are not in {}: {:?} — a decode step needs a TP=1 \
             device-final pack",
            missing.len(),
            args.pack.display(),
            missing.iter().take(3).collect::<Vec<_>>()
        );
    }
    let record_len: BTreeMap<String, u64> = reader
        .records()
        .iter()
        .enumerate()
        .map(|(i, r)| (reader.name(i).to_owned(), r.length))
        .collect();
    let (lo, hi) = (idx[0], *idx.last().expect("non-empty"));
    let last = &reader.records()[hi];
    let span =
        saber_mem::pack::align_up(last.data_off + last.length) - reader.records()[lo].data_off;
    let mut budget = args.budget_mib << 20;
    // A whole-model step needs its weights, its caches and its arena; the weights alone are the
    // biggest single claim and the one that fails first. Refuse before the upload rather than in
    // the middle of it, and name what is missing.
    if span + (1 << 30) > free {
        anyhow::bail!(
            "this step needs {} MiB of weights (plus caches and arena) and the card has {} MiB \
             free of {} MiB. Something else is holding it — prod, or another agent's container. \
             Nothing here will stop prod; run this when the cards are free.",
            span >> 20,
            free >> 20,
            idle.total >> 20
        );
    }
    if span > budget {
        anyhow::bail!(
            "the weight window is {} MiB and the budget is {} MiB",
            span >> 20,
            budget >> 20
        );
    }
    budget -= span;
    let upload = Upload::open(
        &args.pack,
        args.rank,
        UploadConfig {
            start_record: lo,
            limit_bytes: Some(span),
            mode: OpenMode::Direct,
            ..UploadConfig::default()
        },
    )?;
    let uplan = upload.plan();
    let mut ledger = Ledger::new();
    ledger.declare(Pool::Vram(args.rank), uplan.arena_bytes.max(1))?;
    ledger.declare(Pool::HostPinned, uplan.staging_bytes.max(1))?;
    let reserved = upload.reserve(&mut ledger)?;
    let mut staging = hip.host_malloc(reserved.staging_bytes())?;
    let ring = PinnedRing::new(&mut staging, &reserved)?;
    let mut sink = HipSink::new(hip, &reserved)?;
    let mut weights: BTreeMap<String, BoundWeight> = BTreeMap::new();
    let stats = upload.run(&mut sink, &ring, |r| {
        if let Some(layout) = layout_of.get(r.name) {
            weights.insert(r.name.to_string(), BoundWeight::new(r.addr, *layout));
        }
    })?;
    println!(
        "weights   {} records, {} MiB resident, {:.2} GB/s verified",
        weights.len(),
        span >> 20,
        stats.end_to_end_gbps()
    );

    let mut failures = 0usize;
    // Each entry is the BUCKET, built once here so the row count and the sequence count travel
    // together: a prefill entry is `--prefill-seqs` chunks of `--prefill` tokens, and the L9 rule
    // sizes the step by their total.
    let seqs = args.prefill_seqs.max(1);
    let buckets: Vec<Bucket> = args
        .rows
        .iter()
        .map(|r| graph.bucket_for(*r, Phase::Decode))
        .chain(args.prefill.iter().map(|t| Bucket::prefill_batch(seqs, *t)))
        .collect();
    for bucket in buckets {
        failures += one_bucket(
            hip,
            &ctx,
            &stream,
            args,
            &graph,
            &layers,
            &mxfp4,
            &saber,
            &weights,
            &record_len,
            bucket,
            &ids,
            budget,
        )?;
    }
    drop(sink);
    Ok(failures)
}

/// How many GDN state slots the bucket needs.
///
/// A decode step snapshots one state per candidate row (`batch * q`), plus the batch's conv slots
/// and the null slot at 0 — `batch * (q + 2)` is what the P1 capture recorded at b1s5 and it is
/// kept. A PREFILL chunk has no candidates: it scans the chunk through one state per sequence, so
/// it is the sequences plus the null slot. Sizing prefill by `q` asks for 2050 slots x 3 MiB per
/// layer at p2048 (6,150 MiB, which is how this was found) and refuses the bucket.
///
/// One function because the plan and the allocation must agree: they are two statements of the
/// same fact and they were two copies of it.
pub(crate) fn gdn_slot_count(phase: Phase, batch: u32, q: u32) -> u32 {
    match phase {
        Phase::Prefill => batch + 1,
        Phase::Decode => batch * (q + 2),
    }
}

/// The runtime params a bucket's plans share.
fn params_for(ctx: u32, phase: Phase, batch: u32, q: u32) -> RuntimeParams {
    RuntimeParams {
        max_ctx: ctx,
        // Slots for every candidate row's state snapshot, plus the batch's conv slots. The
        // recurrent kernel indexes `state` by `sidx[n*q + t]`, so `batch * q` is the floor;
        // `batch * (q + 2)` is what the P1 capture recorded at b1s5 and it is kept.
        //
        // A PREFILL chunk has no candidates: it scans the chunk through one state per sequence,
        // so its slots are the sequences plus the null one. Sizing it by `q` there would ask for
        // 2050 slots x 1.5 MiB per layer at p2048 and refuse the bucket.
        gdn_slots: gdn_slot_count(phase, batch, q),
    }
}

/// The step's BODY: the embedding and every layer, at the bucket's row count.
///
/// The head is not here. It runs on the rows a sampler reads -- one per sequence at prefill --
/// and it is [`plan_head`]'s job, exactly as `saber-engine` splits them. The two together are
/// what a step is; neither alone is.
fn plan_body<'g>(
    graph: &'g Graph,
    layers: &[u32],
    bucket: Bucket,
    ctx: u32,
    phase: Phase,
    mxfp4: &LaunchRules,
    saber: &SaberRules,
) -> anyhow::Result<BlockPlan<'g>> {
    let q = bucket.q_len().max(1);
    let mut plan = BlockPlan::with_options(
        graph,
        bucket,
        layers,
        params_for(ctx, phase, bucket.m() / q, q),
        mxfp4,
        saber,
        PlanOptions::BODY,
    )?;
    // STATE THE CHUNK'S LIVE ROWS, like the engine does (ws-h27). This command feeds every row of
    // its bucket -- `index_table` above gives each sequence `live_rows = q` -- so the law is the
    // identity here and the descriptors are the ones this gate captured before. It is called
    // anyway, because the alternative is a second caller that pads differently from the engine and
    // nothing that says so.
    if bucket.phase() == Phase::Prefill {
        plan.retarget_prefill_rows(q, saber)?;
        // AND THE RECURRENT STATE AT THE FIRST REAL SLOT (ws-h30), which is where `index_table`
        // above puts sequence 0 of a prefill step; the rest follow it consecutively, which is
        // what the scan's `blockIdx.z` indexing requires and what `StepIndex::check` enforces.
        plan.retarget_prefill_state(saber_exec::forward::FIRST_STATE_SLOT)?;
    }
    Ok(plan)
}

/// The step's HEAD: the final norm and `lm_head`, at the bucket's SAMPLED rows.
///
/// `Graph::head_bucket` is the row-count rule and this is the only place this command states it.
fn plan_head<'g>(
    graph: &'g Graph,
    body: &BlockPlan<'g>,
    ctx: u32,
    phase: Phase,
    mxfp4: &LaunchRules,
    saber: &SaberRules,
) -> anyhow::Result<BlockPlan<'g>> {
    let bucket = body.bucket;
    let q = bucket.q_len().max(1);
    Ok(BlockPlan::with_options(
        graph,
        graph.head_bucket(bucket),
        &[],
        params_for(ctx, phase, bucket.m() / q, q),
        mxfp4,
        saber,
        PlanOptions::HEAD,
    )?)
}

/// The OLD shape of this command: every site in one plan, the head on every row of the chunk.
///
/// Kept because the figures it produced are in the ledger and have to stay reproducible, and
/// because comparing the two is the only way to see what the every-row head costs. `PlanOptions`
/// refuses this shape unless a caller says `with_head_on_every_row()`, which is what makes it a
/// stated choice rather than an accident of which constant was reached for.
fn plan_combined<'g>(
    graph: &'g Graph,
    layers: &[u32],
    bucket: Bucket,
    ctx: u32,
    phase: Phase,
    mxfp4: &LaunchRules,
    saber: &SaberRules,
) -> anyhow::Result<BlockPlan<'g>> {
    let q = bucket.q_len().max(1);
    Ok(BlockPlan::with_options(
        graph,
        bucket,
        layers,
        params_for(ctx, phase, bucket.m() / q, q),
        mxfp4,
        saber,
        PlanOptions::FORWARD.with_head_on_every_row(),
    )?)
}
/// The launches between the body and the head that no plan models: the row selections and the
/// argmax. `saber_exec::forward::Head` is the same three in the shape `saber forward` uses; this
/// carries them as plain descriptors because this command also has to CAPTURE them.
pub(crate) struct HeadExtra {
    pub(crate) gather: saber_kernels::LaunchDescriptor,
    pub(crate) argmax: saber_kernels::LaunchDescriptor,
    /// The edges the head reads and the body wrote, named by the head plan.
    pub(crate) carried: Vec<String>,
    pub(crate) sel: DeviceAddr,
    pub(crate) token: DeviceAddr,
}

/// Where `edge` lives in `plan`'s arena.
pub(crate) fn edge_addr(
    plan: &BlockPlan<'_>,
    base: DeviceAddr,
    edge: &str,
) -> anyhow::Result<DeviceAddr> {
    let b = plan
        .locate(edge)
        .ok_or_else(|| anyhow::anyhow!("`{edge}` is not a planned buffer"))?;
    base.offset(b.offset)
        .ok_or_else(|| anyhow::anyhow!("`{edge}` runs past its arena"))
}

/// The bindings for one row selection: `src` in the body's arena, `out` in the head's.
pub(crate) fn gather_args(
    body: &BlockPlan<'_>,
    body_base: DeviceAddr,
    head: &BlockPlan<'_>,
    head_base: DeviceAddr,
    edge: &str,
    sel: DeviceAddr,
) -> anyhow::Result<ArgBindings> {
    let mut a = ArgBindings::new(Shape::new(0, 0, 0));
    a.bind("src", edge_addr(body, body_base, edge)?);
    a.bind("idx", sel);
    a.bind("out", edge_addr(head, head_base, edge)?);
    Ok(a)
}

/// The bindings for the argmax over the head's logits.
pub(crate) fn argmax_args(
    head: &BlockPlan<'_>,
    head_base: DeviceAddr,
    token: DeviceAddr,
) -> anyhow::Result<ArgBindings> {
    let mut a = ArgBindings::new(Shape::new(0, 0, 0));
    a.bind("logits", edge_addr(head, head_base, "logits.partial")?);
    a.bind("out", token);
    Ok(a)
}

/// The whole step, eagerly: body, the row selections, head, argmax. Returns the dispatch count.
///
/// With no head plan this is the old one-plan step and the count is the body's alone, which is
/// what `--head-on-every-row` asks for.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_whole_step(
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    stream: &Stream,
    exec: &mut BlockExecutor,
    plan: &BlockPlan<'_>,
    head_exec: Option<&mut BlockExecutor>,
    head_plan: Option<&BlockPlan<'_>>,
    extra: Option<&HeadExtra>,
    inputs: &BlockInputs,
) -> anyhow::Result<u32> {
    let mut total = exec
        .run_steps(ctx, launcher, stream, &plan.steps, inputs, false)?
        .launches;
    let (Some(head_exec), Some(head_plan), Some(extra)) = (head_exec, head_plan, extra) else {
        return Ok(total);
    };
    let body_base = exec.arena_base();
    let head_base = head_exec.arena_base();
    for edge in &extra.carried {
        let a = gather_args(plan, body_base, head_plan, head_base, edge, extra.sel)?;
        total += head_exec.launch_extra(ctx, launcher, stream, &extra.gather, &a)?;
    }
    total += head_exec
        .run_steps(ctx, launcher, stream, &head_plan.steps, inputs, false)?
        .launches;
    let a = argmax_args(head_plan, head_base, extra.token)?;
    total += head_exec.launch_extra(ctx, launcher, stream, &extra.argmax, &a)?;
    Ok(total)
}

/// One bucket: plan, seed, run eagerly, capture, replay, compare, time.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn one_bucket(
    hip: &HipThread,
    ctx: &LaunchCtx<'_>,
    stream: &Stream,
    args: &StepCaptureArgs,
    graph: &Graph,
    layers: &[u32],
    mxfp4: &LaunchRules,
    saber: &SaberRules,
    weights: &BTreeMap<String, BoundWeight>,
    record_len: &BTreeMap<String, u64>,
    bucket: Bucket,
    ids: &[i32],
    budget: u64,
) -> anyhow::Result<usize> {
    let phase = bucket.phase();
    let rows = bucket.m();
    // A step is a BODY and a HEAD, which is the shape `saber-engine` runs: the body at the
    // chunk's rows, the head at the rows a sampler reads. `--head-on-every-row` restores this
    // command's original one-plan shape, whose figures are in the ledger.
    let split = !args.head_on_every_row;
    let plan = if split {
        plan_body(graph, layers, bucket, args.ctx, phase, mxfp4, saber)?
    } else {
        plan_combined(graph, layers, bucket, args.ctx, phase, mxfp4, saber)?
    };
    let head_plan = if split {
        Some(plan_head(graph, &plan, args.ctx, phase, mxfp4, saber)?)
    } else {
        None
    };
    // Both plans, for everything that has to cover the whole step: the state it allocates, the
    // index tables it builds, the arena it compares and the ops it times.
    let all_plans: Vec<&BlockPlan<'_>> = std::iter::once(&plan).chain(head_plan.iter()).collect();
    let q = bucket.q_len().max(1);
    let batch = rows / q;
    let label = match phase {
        // `p1024x2` is two sequences' 1024-token chunks in one 2048-row step; `p2048` is one.
        Phase::Prefill if bucket.num_seqs() > 1 => format!("p{q}x{}", bucket.num_seqs()),
        Phase::Prefill => format!("p{rows}"),
        Phase::Decode => format!("b{batch}s{q}"),
    };
    // The three launches no plan models: one row selection per edge the head carries, and the
    // argmax. `saber-engine` issues the same three; they are counted here for the same reason it
    // counts them, which is that the graph does not.
    let extras: u32 = if split {
        saber_exec::forward::Head::EXTRA_LAUNCHES
    } else {
        0
    };
    let declared: u32 = all_plans.iter().map(|p| p.expected_launches()).sum::<u32>() + extras;
    let arena_total: u64 = all_plans.iter().map(|p| p.arena_bytes).sum();
    println!();
    if let Some(h) = &head_plan {
        println!(
            "step      {label}: {:?}, M={rows}, body {} ops / {} launches, head {} ops / {} \
             launches at {} row(s), + {extras} this command adds (2 row selections, 1 argmax) = \
             {declared} declared, arena {} MiB",
            bucket,
            plan.steps.len(),
            plan.expected_launches(),
            h.steps.len(),
            h.expected_launches(),
            h.bucket.m(),
            arena_total >> 20
        );
    } else {
        println!(
            "step      {label}: {:?}, M={rows}, {} ops, {declared} launches declared, arena {} \
             MiB   (ONE plan, head on every row -- not the engine's shape)",
            bucket,
            plan.steps.len(),
            arena_total >> 20
        );
    }
    // The banner only when there IS something to declare, and it no longer claims the K/V write
    // is missing: `kv_append` is an op with a code object as of the attention glue (workstream F,
    // plugin 4), so a step that contains it appends K/V. Until then this always fired, and
    // printing "0 op(s) have no code object ()" beside a complete step would be worse than
    // silence -- a banner nobody can trigger is a banner nobody reads.
    // Over BOTH plans: at TP>1 the head's `logits.gather` is `Supplied`, and a banner that only
    // looked at the body would call that step complete.
    let mut supplied_ops: BTreeMap<&str, u32> = BTreeMap::new();
    for pl in &all_plans {
        for (op, n) in &pl.supplied {
            *supplied_ops.entry(op).or_default() += n;
        }
    }
    let supplied: u32 = supplied_ops.values().sum();
    if supplied > 0 {
        println!(
            "PARTIAL   {supplied} op(s) have no code object ({}), so the attention layers do not \
             compute the model's output. The launch structure, the arena, the state traffic and \
             the microseconds are real.",
            supplied_ops.keys().copied().collect::<Vec<_>>().join(" + ")
        );
    }

    // ---- state: the caches a decode step reads and rewrites in place ---------------------------
    let mut budget = budget;
    let mut owned: Vec<DevicePtr> = Vec::new();
    let mut inputs = BlockInputs {
        weights: weights.clone(),
        ..BlockInputs::default()
    };
    let kv_blocks_per_seq = args.ctx.div_ceil(16).max(1);
    let mut state_ptrs: Vec<(String, usize, u64)> = Vec::new();
    let mut state_bytes = 0u64;
    for step in all_plans.iter().flat_map(|p| p.steps.iter()) {
        for st in &step.op.state {
            if inputs.states.contains_key(&st.name) {
                continue;
            }
            let bytes = match st.bytes_per_slot {
                Some(b) => b * u64::from(gdn_slot_count(phase, batch, q)),
                None => match step.op.geometry {
                    Geometry::Attn {
                        head_dim,
                        block_size,
                        kv_heads,
                        ..
                    } => {
                        u64::from(batch * kv_blocks_per_seq)
                            * u64::from(kv_heads)
                            * u64::from(block_size)
                            * 2
                            * u64::from(head_dim)
                    }
                    _ => anyhow::bail!("state `{}` states no size", st.name),
                },
            };
            budget = budget.checked_sub(bytes).ok_or_else(|| {
                anyhow::anyhow!(
                    "state `{}` ({} MiB) exceeds the budget",
                    st.name,
                    bytes >> 20
                )
            })?;
            state_bytes += bytes;
            let p = hip.malloc(bytes.max(1))?;
            hip.memset(&p, 0, stream)?;
            inputs.states.insert(st.name.clone(), p.addr());
            state_ptrs.push((st.name.clone(), owned.len(), bytes));
            owned.push(p);
        }
    }
    let mut index_keys: Vec<String> = all_plans.iter().flat_map(|p| p.index_keys()).collect();
    index_keys.sort();
    index_keys.dedup();
    for key in index_keys {
        let bytes = index_table(&key, phase, batch, q, kv_blocks_per_seq, args.ctx, ids)?;
        let p = to_device(hip, stream, &bytes, &mut budget)?;
        inputs.indices.insert(key.clone(), p.addr());
        owned.push(p);
    }
    hip.sync_stream(stream)?;
    println!(
        "state     {} caches, {} MiB ({} KV blocks per sequence at ctx {}), {} index tables",
        inputs.states.len(),
        state_bytes >> 20,
        kv_blocks_per_seq,
        args.ctx,
        inputs.indices.len()
    );

    if arena_total > budget {
        anyhow::bail!(
            "the arenas are {} MiB and {} MiB is left",
            arena_total >> 20,
            budget >> 20
        );
    }
    let mut exec = BlockExecutor::new(hip, &plan, 8usize << 20)?;
    let mut head_exec = match &head_plan {
        Some(h) => Some(BlockExecutor::new(hip, h, 8usize << 20)?),
        None => None,
    };

    // A reset is the whole of "a fresh step": zero the arena, so an op that writes nothing shows
    // as a difference rather than as the last run's answer, and re-zero every state the step
    // rewrites in place.
    let reset = |hip: &HipThread,
                 exec: &mut BlockExecutor,
                 head_exec: &mut Option<BlockExecutor>,
                 owned: &[DevicePtr]|
     -> anyhow::Result<()> {
        exec.zero(hip, stream)?;
        if let Some(h) = head_exec.as_mut() {
            h.zero(hip, stream)?;
        }
        for (_, i, _) in &state_ptrs {
            hip.memset(&owned[*i], 0, stream)?;
        }
        wait_bounded(ctx, stream, "resetting the step's state")?;
        Ok(())
    };
    // Everything the comparison watches: the whole arena and every state buffer. Stronger than a
    // probe list, and at 3 MiB (b1s5) or 30 MiB (b8s5) plus the caches it is affordable.
    // The comparison watches BOTH arenas, concatenated: a head that replayed differently would
    // otherwise be invisible, and the head is the half this command was changed to get right.
    let snapshot = |hip: &HipThread,
                    exec: &BlockExecutor,
                    head_exec: &Option<BlockExecutor>,
                    owned: &[DevicePtr]|
     -> anyhow::Result<(Vec<u8>, Vec<Vec<u8>>)> {
        let mut arena = vec![0u8; usize::try_from(plan.arena_bytes)?];
        exec.download(hip, 0, &mut arena, stream)?;
        if let (Some(h), Some(hp)) = (head_exec.as_ref(), head_plan.as_ref()) {
            let mut harena = vec![0u8; usize::try_from(hp.arena_bytes)?];
            h.download(hip, 0, &mut harena, stream)?;
            arena.extend_from_slice(&harena);
        }
        let mut states = Vec::with_capacity(state_ptrs.len());
        for (_, i, bytes) in &state_ptrs {
            let mut b = vec![0u8; usize::try_from(*bytes)?];
            hip.memcpy_d2h_blocking(&mut b, &owned[*i], stream)?;
            states.push(b);
        }
        Ok((arena, states))
    };

    let counters = LaunchCounters::new();
    let capture_counters = CaptureCounters::new();
    let mut launcher = Launcher::new(format!("step.{label}"), &counters);

    // ---- the three launches no plan models -----------------------------------------------------
    //
    // The row selection, once per edge the head carries, and the argmax. `saber-engine` issues
    // exactly these between its body and its head; this command issues them for the same reason
    // and counts them in its own budget, because the graph does not model them.
    let head_rows = head_plan.as_ref().map_or(0, |h| h.bucket.m());
    let sel_rows: Vec<i32> = match (split, phase) {
        // A prefill chunk samples its LAST row: that is the position whose logits become the
        // sequence's first generated token.
        // The last row of EACH chunk: with one sequence that is `rows - 1`, and with several
        // it is one index per chunk, which is what makes `sampled_rows` the sequence count.
        (true, Phase::Prefill) => (1..=bucket.num_seqs())
            .map(|i| i32::try_from(i * q).map(|r| r - 1))
            .collect::<Result<Vec<i32>, _>>()?,
        // Every decode row is a sampled position, so the selection is the identity -- and it is
        // still issued, because the engine issues it and this command measures the engine's step.
        (true, Phase::Decode) => (0..i32::try_from(head_rows)?).collect(),
        (false, _) => Vec::new(),
    };
    let head_extra = match (&head_plan, head_exec.as_ref()) {
        (Some(hp), Some(_)) => {
            let hidden = graph.config.hidden_size;
            let vocab = match graph
                .ops
                .iter()
                .find(|o| o.name == "lm_head")
                .map(|o| &o.geometry)
            {
                Some(Geometry::Gemm { n, .. }) => *n,
                _ => anyhow::bail!("the graph has no `lm_head` GEMM"),
            };
            let rules = saber_kernels::head::HeadRules::prod_head()?;
            let gather = saber_kernels::head::resolve(
                saber_kernels::head::HeadOp::RowGather,
                saber_kernels::head::HeadShape {
                    rows: head_rows,
                    cols: hidden,
                    src_stride: i64::from(hidden),
                    vocab: (0, 0),
                },
                &rules,
            )?;
            let argmax = saber_kernels::head::resolve(
                saber_kernels::head::HeadOp::Argmax,
                saber_kernels::head::HeadShape {
                    rows: head_rows,
                    cols: vocab,
                    src_stride: i64::from(vocab),
                    vocab: (0, 0),
                },
                &rules,
            )?;
            // The edges the head reads and the body wrote. Asked of the HEAD PLAN, so the names
            // are the graph's and never spelled out here.
            let carried: Vec<String> = hp.inputs.clone();
            if carried.len() != 2 {
                anyhow::bail!(
                    "the head reads {} edges from outside its plan ({carried:?}); this command \
                     selects rows for exactly the hidden state and the residual",
                    carried.len()
                );
            }
            let sel = to_device(hip, stream, &i32_bytes(&sel_rows), &mut budget)?;
            let token = hip.malloc(4 * u64::from(head_rows).max(1))?;
            hip.memset(&token, 0, stream)?;
            owned.push(sel);
            owned.push(token);
            Some(HeadExtra {
                gather,
                argmax,
                carried,
                sel: owned[owned.len() - 2].addr(),
                token: owned[owned.len() - 1].addr(),
            })
        }
        _ => None,
    };

    // ---- the eager step ------------------------------------------------------------------------
    reset(hip, &mut exec, &mut head_exec, &owned)?;
    // FN63: the eager step's dispatches, recorded as issued (the HSA replay's node list).
    launcher.record_dispatches(args.fn63.is_some());
    let eager_launches = run_whole_step(
        ctx,
        &mut launcher,
        stream,
        &mut exec,
        &plan,
        head_exec.as_mut(),
        head_plan.as_ref(),
        head_extra.as_ref(),
        &inputs,
    )?;
    wait_bounded(ctx, stream, "the eager step")?;
    let fn63_records = launcher.take_dispatch_records();
    launcher.record_dispatches(false);
    let (eager_arena, eager_states) = snapshot(hip, &exec, &head_exec, &owned)?;
    // THE SAMPLED TOKENS, one per sequence, which is the only thing that makes two chunks in one
    // step comparable with the same two chunks run one at a time. The argmax already wrote them;
    // printing them costs a 4-byte-per-sequence read and turns "the step ran" into "the step
    // produced these tokens". At TP>1 this would be a shard's winner and not a token -- this
    // command is TP=1.
    if let Some(x) = head_extra.as_ref() {
        let mut got = vec![0u8; 4 * head_rows as usize];
        hip.memcpy_d2h_blocking(&mut got, &owned[owned.len() - 1], stream)?;
        let toks: Vec<i32> = got
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let _ = x;
        println!(
            "tokens    argmax per sequence: {toks:?}   (one per sampled row; the last row of each \
             chunk)"
        );
    }
    let mut failures = 0usize;
    if eager_launches != declared {
        println!(
            "LAUNCH BUDGET the eager step issued {eager_launches} dispatches, the plans declare \
             {declared}"
        );
        failures += 1;
    }

    // ---- capture ---------------------------------------------------------------------------------
    reset(hip, &mut exec, &mut head_exec, &owned)?;
    let proof = GraphSafe::from_flags(true, false).expect("graph_safe && !may_sync");
    // ONE graph over the WHOLE step: the body's ops, the row selections, the head's ops and the
    // argmax. The command's claim is still "the step as one captured graph" -- what changed is
    // that the step is now two plans and three launches between them, which is the engine's shape
    // rather than one plan covering every site.
    let body_bound = bind_steps(&exec, &plan.steps, &inputs)?;
    let head_bound = match (&head_plan, head_exec.as_ref()) {
        (Some(hp), Some(he)) => bind_steps(he, &hp.steps, &inputs)?,
        _ => Vec::new(),
    };
    check_capturable(&body_bound)?;
    check_capturable(&head_bound)?;
    // The extras' bindings have to outlive the op list that borrows them.
    let extra_args: Vec<(&saber_kernels::LaunchDescriptor, ArgBindings)> =
        match (&head_plan, head_exec.as_ref(), head_extra.as_ref()) {
            (Some(hp), Some(he), Some(x)) => {
                let (bb, hb) = (exec.arena_base(), he.arena_base());
                let mut v: Vec<(&saber_kernels::LaunchDescriptor, ArgBindings)> = Vec::new();
                for edge in &x.carried {
                    v.push((&x.gather, gather_args(&plan, bb, hp, hb, edge, x.sel)?));
                }
                v.push((&x.argmax, argmax_args(hp, hb, x.token)?));
                v
            }
            _ => Vec::new(),
        };
    // Every descriptor's workspaces, before the capture opens -- the extras included, or the
    // first row selection would allocate inside it (docs/EXEC.md section 4).
    for st in body_bound.iter().chain(head_bound.iter()) {
        for desc in st.descs {
            launcher.prepare_workspaces(ctx, desc, &st.args, stream)?;
        }
    }
    for (desc, args) in &extra_args {
        launcher.prepare_workspaces(ctx, desc, args, stream)?;
    }
    check_persistent_ready(&launcher, &body_bound)?;
    check_persistent_ready(&launcher, &head_bound)?;
    let mut ops: Vec<CapturableOp<'_>> = Vec::new();
    for st in &body_bound {
        ops.push(CapturableOp::sequence(st.descs, &st.args, proof)?);
    }
    // ORDER IS THE STEP: body, then the selections, then the head, then the argmax. A graph that
    // recorded the head before the selection would replay the previous step's rows.
    let n_gathers = extra_args.len().saturating_sub(1);
    for (desc, args) in extra_args.iter().take(n_gathers) {
        ops.push(CapturableOp::new(desc, args, proof));
    }
    for st in &head_bound {
        ops.push(CapturableOp::sequence(st.descs, &st.args, proof)?);
    }
    if let Some((desc, args)) = extra_args.last() {
        ops.push(CapturableOp::new(desc, args, proof));
    }
    let captured = CapturedBucket::capture(
        ctx,
        &mut launcher,
        &ops,
        stream,
        format!("step.{label}"),
        &capture_counters,
    )?;
    wait_bounded(ctx, stream, "the capture")?;
    let want_nodes = usize::try_from(declared)?;
    println!(
        "captured  {} node(s) for {} op(s); the step's budget is {want_nodes} — {}",
        captured.node_count(),
        ops.len(),
        if captured.node_count() == want_nodes {
            "MATCH"
        } else {
            "MISMATCH"
        }
    );
    if captured.node_count() != want_nodes {
        anyhow::bail!(
            "the graph has {} nodes, not {want_nodes}",
            captured.node_count()
        );
    }
    drop(body_bound);
    drop(head_bound);
    let persistent = launcher.persistent_regions();
    println!(
        "persistent {} region(s) prepared and zeroed before the capture opened (split-K counters \
         and GDN barriers, one per distinct shape)",
        persistent.len()
    );

    // ---- replays, each from a fresh state --------------------------------------------------------
    for r in 1..=args.replays {
        reset(hip, &mut exec, &mut head_exec, &owned)?;
        captured.replay(ctx, stream)?;
        wait_bounded(ctx, stream, &format!("graph replay {r}"))?;
        let (arena, states) = snapshot(hip, &exec, &head_exec, &owned)?;
        let arena_bad = arena
            .iter()
            .zip(&eager_arena)
            .filter(|(a, b)| a != b)
            .count();
        let state_bad: usize = states
            .iter()
            .zip(&eager_states)
            .map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x != y).count())
            .sum();
        println!(
            "replay {r}  arena {} of {} bytes differ from the eager step, state {} of {} — {}",
            arena_bad,
            arena.len(),
            state_bad,
            eager_states.iter().map(Vec::len).sum::<usize>(),
            if arena_bad == 0 && state_bad == 0 {
                "BIT-EXACT"
            } else {
                "MISMATCH"
            }
        );
        if arena_bad != 0 || state_bad != 0 {
            failures += 1;
        }
    }

    // ---- FN63: the same step through ROCr AQL packets, then stop --------------------------------
    if let Some(opts) = args.fn63.as_ref() {
        let mut extents: Vec<(u64, u64)> = Vec::new();
        let base = exec.arena_base().as_u64();
        extents.extend(
            plan.buffers
                .iter()
                .map(|b| (base + b.offset, b.bytes.max(1))),
        );
        if let (Some(hp), Some(he)) = (head_plan.as_ref(), head_exec.as_ref()) {
            let hb = he.arena_base().as_u64();
            extents.extend(hp.buffers.iter().map(|b| (hb + b.offset, b.bytes.max(1))));
        }
        extents.extend(owned.iter().map(|d| (d.addr().as_u64(), d.len().max(1))));
        let input = crate::fn63_hsa::Fn63Input {
            hip,
            stream,
            device: args.device,
            prebuilt: &args.prebuilt,
            records: fn63_records,
            graph_nodes: captured.node_count(),
            eager_arena: &eager_arena,
            eager_states: &eager_states,
            extents,
        };
        let mut state = |reset_now: bool| -> anyhow::Result<Option<crate::fn63_hsa::Snapshot>> {
            if reset_now {
                reset(hip, &mut exec, &mut head_exec, &owned)?;
                Ok(None)
            } else {
                Ok(Some(snapshot(hip, &exec, &head_exec, &owned)?))
            }
        };
        let mut replay = || -> anyhow::Result<()> {
            captured.replay(ctx, stream)?;
            Ok(())
        };
        failures += crate::fn63_hsa::run(input, opts, &mut state, &mut replay)?;
        return Ok(failures);
    }

    // ---- timing: interleaved pairs ----------------------------------------------------------------
    let start = hip.create_event(EventFlags::Default)?;
    let end = hip.create_event(EventFlags::Default)?;
    let mut g_us: Vec<f64> = Vec::with_capacity(args.iters);
    let mut e_us: Vec<f64> = Vec::with_capacity(args.iters);
    for i in 0..(args.warmup + args.iters) {
        hip.record_event(&start, stream)?;
        captured.replay(ctx, stream)?;
        hip.record_event(&end, stream)?;
        wait_bounded(ctx, stream, "a timed replay")?;
        let g = f64::from(hip.elapsed_ms(&start, &end)?);

        hip.record_event(&start, stream)?;
        // The WHOLE step, not just the body: the eager arm has to be the same work the graph
        // replays or the pair measures two different things.
        run_whole_step(
            ctx,
            &mut launcher,
            stream,
            &mut exec,
            &plan,
            head_exec.as_mut(),
            head_plan.as_ref(),
            head_extra.as_ref(),
            &inputs,
        )?;
        hip.record_event(&end, stream)?;
        wait_bounded(ctx, stream, "a timed eager step")?;
        let e = f64::from(hip.elapsed_ms(&start, &end)?);
        if i >= args.warmup {
            g_us.push(g);
            e_us.push(e);
        }
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let (g, e) = (med(&mut g_us), med(&mut e_us));
    let nodes = f64::from(declared);
    println!();
    println!(
        "time      {label}: {} interleaved pairs, hipEvents around the WHOLE step, one stream, \
         GPU idle",
        args.iters
    );
    println!(
        "          graph replay  {g:>8.3} ms/step   {:>6.2} µs/node",
        g * 1000.0 / nodes
    );
    println!(
        "          eager step    {e:>8.3} ms/step   {:>6.2} µs/node",
        e * 1000.0 / nodes
    );
    println!(
        "          graph saves   {:>8.3} ms/step   {:>6.2} µs/node ({:+.1}%)",
        e - g,
        (e - g) * 1000.0 / nodes,
        100.0 * (e - g) / e
    );
    // The comparator's caveat comes from the SAME fact as the banner — `plan.supplied` — because a
    // sentence that says "attention is PARTIAL" while every op has a code object is worse than no
    // sentence: it tells a reader to discount a number that is now real (I15 caught it still
    // printing after P4's glue landed).
    let caveat = if supplied_ops.is_empty() {
        "every op has a code object, so the difference is rows and nothing else"
    } else {
        "and this step has op(s) with no code object, so the ratio is structural"
    };
    println!(
        "          P-1b comparator: prod at SPEC=0, greedy, one stream, GPU idle = {PROD_MS_PER_STEP} \
         ms/step. This step is {rows} rows to prod's 1 — {caveat} — so the ratio {:.2}x is a \
         structural reading, not a speedup.",
        g / PROD_MS_PER_STEP
    );

    // ---- where the time is, per family -----------------------------------------------------------
    //
    // Solo medians: each distinct op bracketed alone. ws-h7 established that a lone launch on an
    // idle queue OVER-states device time (it pays a cold start the step amortises), so these are
    // RELATIVE ONLY: they say which family dominates, never what the step costs.
    let mut solo: BTreeMap<String, f64> = BTreeMap::new();
    let mut per_family: BTreeMap<&str, (f64, usize)> = BTreeMap::new();
    // Both plans: the head is two ops and one of them is `lm_head`, which is the op this whole
    // change is about. Each step is addressed against the executor whose arena holds it.
    let timed_steps: Vec<(&Step<'_>, bool)> = plan
        .steps
        .iter()
        .map(|s| (s, false))
        .chain(
            head_plan
                .iter()
                .flat_map(|h| h.steps.iter().map(|s| (s, true))),
        )
        .collect();
    for (step, in_head) in timed_steps {
        let Route::Kernel { descs, shape } = &step.route else {
            continue;
        };
        // The cache key is the symbol AND the shape. The symbol alone silently aliases two
        // different GEMMs onto one measurement the moment a rule routes them to the same
        // instantiation -- which is exactly what happened to `lm_head` (2026-09-19): once the
        // vocabulary GEMM took the decode tile it shared `..._bk64_ks1_tm1_...` with a layer
        // linear 24x smaller, and the head's family read 97.6 µs for 676 MB of weights, a number
        // physically impossible at 635 GB/s. A solo median is per (kernel, shape) or it is not a
        // measurement.
        let key = format!(
            "{}|{}x{}x{}",
            descs
                .iter()
                .map(|d| d.kernel.clone())
                .collect::<Vec<_>>()
                .join("+"),
            shape.m,
            shape.n,
            shape.k
        );
        let us = if let Some(v) = solo.get(&key) {
            *v
        } else {
            let mut args_b = saber_exec::ArgBindings::new(*shape);
            for (name, src) in &step.bindings {
                let addr = match (in_head, head_exec.as_ref()) {
                    (true, Some(h)) => h.address(src, &inputs)?,
                    _ => exec.address(src, &inputs)?,
                };
                args_b.bind(name, addr);
            }
            let refs: Vec<&saber_kernels::LaunchDescriptor> = descs.iter().collect();
            for _ in 0..4 {
                launcher.launch_op(ctx, &refs, &args_b, stream)?;
            }
            wait_bounded(ctx, stream, "warming an op")?;
            let mut s: Vec<f64> = Vec::with_capacity(9);
            for _ in 0..9 {
                hip.record_event(&start, stream)?;
                launcher.launch_op(ctx, &refs, &args_b, stream)?;
                hip.record_event(&end, stream)?;
                wait_bounded(ctx, stream, "timing an op")?;
                s.push(f64::from(hip.elapsed_ms(&start, &end)?) * 1000.0);
            }
            s.sort_by(f64::total_cmp);
            let v = s[s.len() / 2];
            solo.insert(key, v);
            v
        };
        let slot = per_family.entry(family(step)).or_insert((0.0, 0));
        slot.0 += us;
        slot.1 += 1;
    }
    let total: f64 = per_family.values().map(|(us, _)| us).sum();
    println!();
    println!(
        "families  solo medians summed per family — RELATIVE ONLY (a lone launch over-states \
         device time; the step's own number is above)"
    );
    let mut fams: Vec<(&&str, &(f64, usize))> = per_family.iter().collect();
    fams.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
    for (name, (us, n)) in fams {
        println!(
            "          {name:<12} {:>9.1} µs over {n:>4} op(s)   {:>5.1}%",
            us,
            100.0 * us / total
        );
    }
    println!(
        "          (sum {total:.1} µs vs the step's {:.1} µs: the gap is what the solo method \
         over-states)",
        g * 1000.0
    );
    println!("counters  {}", counters.snapshot().line());
    println!("          {}", capture_counters.snapshot().line());

    // ---- in-chain per-layer timing -----------------------------------------------------------
    if args.per_layer {
        failures += per_layer_chain(
            hip,
            ctx,
            stream,
            &mut exec,
            &mut launcher,
            &capture_counters,
            &plan,
            &inputs,
            &owned,
            &state_ptrs,
            args,
            &label,
            g,
            &eager_arena,
            &eager_states,
        )?;
    }

    // ---- family ablation ----------------------------------------------------------------------
    if args.ablate {
        ablate_families(
            hip,
            ctx,
            stream,
            &mut exec,
            &mut launcher,
            &capture_counters,
            &plan,
            &inputs,
            &owned,
            &state_ptrs,
            args,
            &label,
            g,
            record_len,
        )?;
    }

    // ---- per-op, in chain: achieved GB/s per GEMM shape ---------------------------------------
    if args.per_op {
        per_op_gemms(
            hip,
            ctx,
            stream,
            &mut exec,
            &mut launcher,
            &capture_counters,
            &plan,
            &inputs,
            args,
            &label,
            g,
            record_len,
        )?;
    }

    drop(captured);
    drop(exec);
    drop(owned);
    Ok(failures)
}

/// The step as a chain of per-layer sub-graphs, each timed IN ITS REAL POSITION.
///
/// Why sub-graphs and not events inside one graph: a `hipEventRecord` under capture becomes an
/// event-record NODE, and this engine asserts a graph's node count against the launch budget (rule
/// 7), so 65 event nodes would either break that assertion or force it to be loosened for a
/// measurement. A chain of sub-graphs replayed back to back in the plan's own order costs 66 graph
/// launches instead of 1 — which this function measures against the single-graph number, so the
/// reader can see what the instrument costs.
///
/// **The point is the cache state.** A solo launch of one GEMM re-reads its ~44 MiB of weights from
/// the 64 MB Infinity Cache on every repeat, so a solo median under-states it. In the chain, layer
/// `n`'s weights were evicted by layers `n+1…63` before layer `n` runs again, which is the state a
/// real step runs in.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn per_layer_chain(
    hip: &HipThread,
    ctx: &LaunchCtx<'_>,
    stream: &Stream,
    exec: &mut BlockExecutor,
    launcher: &mut Launcher,
    capture_counters: &std::sync::Arc<CaptureCounters>,
    plan: &BlockPlan<'_>,
    inputs: &BlockInputs,
    owned: &[DevicePtr],
    state_ptrs: &[(String, usize, u64)],
    args: &StepCaptureArgs,
    label: &str,
    whole_ms: f64,
    eager_arena: &[u8],
    eager_states: &[Vec<u8>],
) -> anyhow::Result<usize> {
    // Slice the plan at site boundaries: the embedding, each layer, the head. The steps are in
    // forward order, so a slice is a contiguous range and the ranges tile the step exactly.
    let mut slices: Vec<(String, std::ops::Range<usize>)> = Vec::new();
    let mut start = 0usize;
    let mut current = site_label(&plan.steps[0]);
    for (i, step) in plan.steps.iter().enumerate() {
        let l = site_label(step);
        if l != current {
            slices.push((current, start..i));
            current = l;
            start = i;
        }
    }
    slices.push((current, start..plan.steps.len()));

    let proof = GraphSafe::from_flags(true, false).expect("graph_safe && !may_sync");
    let mut subs: Vec<(String, CapturedBlock)> = Vec::new();
    let mut declared = 0u32;
    for (name, range) in &slices {
        // A slice with no launched op (a site that is all supplied) is not a graph; skip it and say
        // so by its absence from the table.
        if !plan.steps[range.clone()]
            .iter()
            .any(|s| matches!(s.route, Route::Kernel { .. }))
        {
            continue;
        }
        let sub = CapturedBlock::capture_steps(
            ctx,
            launcher,
            exec,
            &plan.steps[range.clone()],
            inputs,
            stream,
            format!("step.{label}.{name}"),
            proof,
            capture_counters,
        )?;
        declared += sub.declared_launches();
        subs.push((name.clone(), sub));
    }
    let want = plan.expected_launches();
    println!();
    println!(
        "per-layer {label}: {} sub-graph(s), {declared} nodes in total against the step's {want} — {}",
        subs.len(),
        if declared == want { "MATCH" } else { "MISMATCH" }
    );
    let mut failures = 0usize;
    if declared != want {
        failures += 1;
    }

    // The chain must still compute what the eager step computes: same addresses, same order.
    reset_state(hip, exec, owned, state_ptrs, ctx, stream)?;
    for (_, sub) in &subs {
        sub.replay(ctx, stream)?;
    }
    wait_bounded(ctx, stream, "the sub-graph chain")?;
    let mut arena = vec![0u8; eager_arena.len()];
    exec.download(hip, 0, &mut arena, stream)?;
    let arena_bad = arena
        .iter()
        .zip(eager_arena)
        .filter(|(a, b)| a != b)
        .count();
    let mut state_bad = 0usize;
    for ((_, i, bytes), want) in state_ptrs.iter().zip(eager_states) {
        let mut b = vec![0u8; usize::try_from(*bytes)?];
        hip.memcpy_d2h_blocking(&mut b, &owned[*i], stream)?;
        state_bad += b.iter().zip(want).filter(|(x, y)| x != y).count();
    }
    println!(
        "          the chain vs the eager step: arena {arena_bad} bytes differ, state {state_bad} — {}",
        if arena_bad == 0 && state_bad == 0 {
            "BIT-EXACT"
        } else {
            "MISMATCH"
        }
    );
    if arena_bad != 0 || state_bad != 0 {
        failures += 1;
    }

    // Time each sub-graph in place: one pass over the chain per sample, events around each link, so
    // every layer is timed with the cache state its predecessors left.
    let start_ev = hip.create_event(EventFlags::Default)?;
    let end_ev = hip.create_event(EventFlags::Default)?;
    let mut per: Vec<Vec<f64>> = vec![Vec::with_capacity(args.iters); subs.len()];
    let mut chain_total: Vec<f64> = Vec::with_capacity(args.iters);
    for i in 0..(args.warmup + args.iters) {
        let t = std::time::Instant::now();
        for (j, (_, sub)) in subs.iter().enumerate() {
            hip.record_event(&start_ev, stream)?;
            sub.replay(ctx, stream)?;
            hip.record_event(&end_ev, stream)?;
            wait_bounded(ctx, stream, "a sub-graph")?;
            let us = f64::from(hip.elapsed_ms(&start_ev, &end_ev)?) * 1000.0;
            if i >= args.warmup {
                per[j].push(us);
            }
        }
        if i >= args.warmup {
            chain_total.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let mut rows: Vec<(String, f64)> = subs
        .iter()
        .enumerate()
        .map(|(j, (name, _))| (name.clone(), med(&mut per[j])))
        .collect();
    let sum_us: f64 = rows.iter().map(|(_, us)| us).sum();
    let chain_ms = med(&mut chain_total);
    println!(
        "          sum of the links {:.3} ms vs the single graph's {whole_ms:.3} ms; the chain's \
         own wall is {chain_ms:.3} ms ({} launches instead of 1, and a host wait per link)",
        sum_us / 1000.0,
        subs.len()
    );

    // Per family, in the chain: a layer's site says which family its ops are, so group the links.
    let mut fam: BTreeMap<&str, (f64, usize)> = BTreeMap::new();
    for (name, us) in &rows {
        let key = if name == "embedding" || name == "head" {
            "embed+head"
        } else if name.starts_with("layer") {
            "layers"
        } else {
            "other"
        };
        let e = fam.entry(key).or_insert((0.0, 0));
        e.0 += us;
        e.1 += 1;
    }
    println!("          per-link medians, in place (ms), the ten costliest:");
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (name, us) in rows.iter().take(10) {
        println!(
            "            {name:<16} {:>8.3} ms   {:>5.1}%",
            us / 1000.0,
            100.0 * us / sum_us
        );
    }
    let layer_rows: Vec<&(String, f64)> = rows
        .iter()
        .filter(|(n, _)| n.starts_with("layer"))
        .collect();
    if !layer_rows.is_empty() {
        let lo = layer_rows
            .iter()
            .map(|(_, us)| *us)
            .fold(f64::MAX, f64::min);
        let hi = layer_rows.iter().map(|(_, us)| *us).fold(0.0, f64::max);
        let mean = layer_rows.iter().map(|(_, us)| *us).sum::<f64>() / layer_rows.len() as f64;
        println!(
            "          {} layer links: mean {:.3} ms, min {:.3}, max {:.3} (the max is the \
             attention layer or the first GDN layer to miss the cache)",
            layer_rows.len(),
            mean / 1000.0,
            lo / 1000.0,
            hi / 1000.0
        );
    }
    for (k, (us, n)) in &fam {
        println!(
            "            {k:<16} {:>8.3} ms over {n:>3} link(s)   {:>5.1}%",
            us / 1000.0,
            100.0 * us / sum_us
        );
    }
    Ok(failures)
}

/// The site a step belongs to, as a link name.
fn site_label(step: &Step<'_>) -> String {
    match step.op.site {
        Site::Embedding => "embedding".to_string(),
        Site::Head => "head".to_string(),
        Site::Layer(n) => format!("layer{n:02}"),
        _ => "other".to_string(),
    }
}

/// Re-zero the arena and every state buffer: one fresh step's worth of state.
pub(crate) fn reset_state(
    hip: &HipThread,
    exec: &mut BlockExecutor,
    owned: &[DevicePtr],
    state_ptrs: &[(String, usize, u64)],
    ctx: &LaunchCtx<'_>,
    stream: &Stream,
) -> anyhow::Result<()> {
    exec.zero(hip, stream)?;
    for (_, i, _) in state_ptrs {
        hip.memset(&owned[*i], 0, stream)?;
    }
    wait_bounded(ctx, stream, "resetting the step's state")?;
    Ok(())
}

/// Rebuild the step with one family's ops omitted, and time it.
///
/// The marginal cost of a family is what the step loses when its ops are not there — measured in the
/// real chain, with every other family's cache traffic still happening. That is a different and more
/// honest number than a solo median: it includes the family's effect on everyone else's cache
/// residency, which is most of what a decode step is.
///
/// **The outputs of an ablated step are meaningless** — a layer whose GDN never ran feeds the next
/// layer whatever the arena held. Nothing is compared here; only the wall time is read, and the
/// report says so.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn ablate_families(
    hip: &HipThread,
    ctx: &LaunchCtx<'_>,
    stream: &Stream,
    exec: &mut BlockExecutor,
    launcher: &mut Launcher,
    capture_counters: &std::sync::Arc<CaptureCounters>,
    plan: &BlockPlan<'_>,
    inputs: &BlockInputs,
    owned: &[DevicePtr],
    state_ptrs: &[(String, usize, u64)],
    args: &StepCaptureArgs,
    label: &str,
    whole_ms: f64,
    record_len: &BTreeMap<String, u64>,
) -> anyhow::Result<()> {
    let proof = GraphSafe::from_flags(true, false).expect("graph_safe && !may_sync");
    println!();
    println!(
        "ablation  {label}: the step rebuilt with one family omitted. OUTPUTS ARE MEANINGLESS — \
         nothing is compared, only the wall time is read."
    );

    // The families to drop, in a fixed order so two runs are comparable. GEMMs stay: a step with no
    // linears is not a step, and every other family's inputs come from one.
    let drops = ["GDN", "producers", "attention", "embed+head"];
    let mut full: Vec<f64> = Vec::with_capacity(args.iters);
    let mut results: Vec<(&str, usize, f64)> = Vec::new();

    // A fresh capture of the FULL step through the same path, so the baseline and the ablations
    // differ in nothing but the omitted ops.
    let keep_all: Vec<Step<'_>> = plan.steps.to_vec();
    let base = CapturedBlock::capture_steps(
        ctx,
        launcher,
        exec,
        &keep_all,
        inputs,
        stream,
        format!("ablate.{label}.full"),
        proof,
        capture_counters,
    )?;
    let start_ev = hip.create_event(EventFlags::Default)?;
    let end_ev = hip.create_event(EventFlags::Default)?;

    let mut captured: Vec<(&str, usize, CapturedBlock)> = Vec::new();
    for drop_family in drops {
        let kept: Vec<Step<'_>> = plan
            .steps
            .iter()
            .filter(|s| family(s) != drop_family)
            .cloned()
            .collect();
        let dropped = plan.steps.len() - kept.len();
        if dropped == 0 {
            continue;
        }
        let sub = CapturedBlock::capture_steps(
            ctx,
            launcher,
            exec,
            &kept,
            inputs,
            stream,
            format!("ablate.{label}.no-{drop_family}"),
            proof,
            capture_counters,
        )?;
        captured.push((drop_family, dropped, sub));
    }

    // Interleaved: the full step and every ablation in one pass, so clock and cache state apply
    // equally to all of them.
    let mut per: Vec<Vec<f64>> = vec![Vec::with_capacity(args.iters); captured.len()];
    for i in 0..(args.warmup + args.iters) {
        reset_state(hip, exec, owned, state_ptrs, ctx, stream)?;
        hip.record_event(&start_ev, stream)?;
        base.replay(ctx, stream)?;
        hip.record_event(&end_ev, stream)?;
        wait_bounded(ctx, stream, "the full step")?;
        let f = f64::from(hip.elapsed_ms(&start_ev, &end_ev)?);
        if i >= args.warmup {
            full.push(f);
        }
        for (j, (_, _, sub)) in captured.iter().enumerate() {
            reset_state(hip, exec, owned, state_ptrs, ctx, stream)?;
            hip.record_event(&start_ev, stream)?;
            sub.replay(ctx, stream)?;
            hip.record_event(&end_ev, stream)?;
            wait_bounded(ctx, stream, "an ablated step")?;
            let v = f64::from(hip.elapsed_ms(&start_ev, &end_ev)?);
            if i >= args.warmup {
                per[j].push(v);
            }
        }
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let base_ms = med(&mut full);
    for (j, (name, dropped, sub)) in captured.iter().enumerate() {
        let ms = med(&mut per[j]);
        results.push((name, *dropped, base_ms - ms));
        println!(
            "          without {name:<12} {} of {} ops ({} nodes): {ms:>8.3} ms — marginal \
             {:>7.3} ms ({:>5.1}% of the step)",
            dropped,
            plan.steps.len(),
            sub.node_count(),
            base_ms - ms,
            100.0 * (base_ms - ms) / base_ms
        );
    }
    println!(
        "          full step through the same path: {base_ms:.3} ms (the single-graph number above \
         is {whole_ms:.3})"
    );
    let marginal_sum: f64 = results.iter().map(|(_, _, ms)| ms).sum();
    println!(
        "          the four marginals sum to {marginal_sum:.3} ms of {base_ms:.3}; what is left, \
         {:.3} ms ({:.1}%), is the GEMM family plus whatever the omitted families were sharing \
         with it",
        base_ms - marginal_sum,
        100.0 * (base_ms - marginal_sum) / base_ms
    );

    // ---- the number that matters: the GEMM family against the DRAM floor ----------------------
    // The bytes the GEMM family streams once per step: every weight record its ops bind, at the
    // PACK's own packed length. Taken from the pack's record table rather than from a shape, so a
    // layout's padding and its scale/reference vectors are counted the way the loader reads them.
    let mut gemm_keys: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for step in &plan.steps {
        if family(step) != "GEMMs" {
            continue;
        }
        for (_, src) in &step.bindings {
            if let saber_exec::Source::Weight { key, .. } = src {
                gemm_keys.insert(key.as_str());
            }
        }
    }
    let gemm_records = gemm_keys.len();
    let gemm_bytes: u64 = gemm_keys
        .iter()
        .filter_map(|k| record_len.get(*k).copied())
        .sum();
    let floor_ms = (gemm_bytes as f64 / (args.dram_gbs * 1e9)) * 1e3;
    println!();
    println!(
        "floor     the {gemm_records} weight record(s) the 304 layer GEMMs bind are {} MiB; at \
         {} GB/s that is {floor_ms:.3} ms of pure weight stream",
        gemm_bytes >> 20,
        args.dram_gbs
    );
    println!(
        "          the step is {base_ms:.3} ms, so the weight stream is {:.1}% of it and \
         everything else — producers, GDN state, attention, the head, and every launch — is the \
         other {:.1}%",
        100.0 * floor_ms / base_ms,
        100.0 * (base_ms - floor_ms) / base_ms
    );
    println!(
        "          read with the ablation above: the GEMM family's own marginal cost is what is \
         left of the step after the four marginals ({:.3} ms), which is {:.2}x its DRAM floor",
        base_ms - marginal_sum,
        (base_ms - marginal_sum) / floor_ms
    );
    Ok(())
}

/// One GEMM shape, measured in the chain: what it binds, what it achieves, and how far that is from
/// the floor.
struct ShapeRow {
    /// `n x k`.
    n: u32,
    k: u32,
    /// The variant the rules chose.
    kernel: String,
    /// How many of them are in a step.
    count: usize,
    /// Weight bytes ONE of them reads.
    bytes: u64,
    /// Median in-chain microseconds for one, with the per-link instrument cost removed.
    us: f64,
}

impl ShapeRow {
    /// Achieved bandwidth for the bytes it binds.
    fn gbs(&self) -> f64 {
        self.bytes as f64 / (self.us * 1e3)
    }
    /// What all of them cost in a step, ms.
    fn total_ms(&self) -> f64 {
        self.us * self.count as f64 / 1000.0
    }
    /// What all of them would cost at the DRAM peak, ms.
    fn floor_ms(&self, dram_gbs: f64) -> f64 {
        (self.bytes * self.count as u64) as f64 / (dram_gbs * 1e9) * 1e3
    }
    /// Milliseconds of the step this shape spends ABOVE its own weight stream.
    fn above_ms(&self, dram_gbs: f64) -> f64 {
        self.total_ms() - self.floor_ms(dram_gbs)
    }
}

/// Every layer linear of `plan`, grouped by `(n, k, variant)`: median in-chain µs, achieved GB/s
/// and TFLOP/s, and which roof the shape is under. `indices[j]` is the plan step link `j` timed and
/// `per_us[j]` its microseconds with the instrument already removed. Returns the linears' total
/// ms in chain.
///
/// One function for `step-capture --per-op` and `step-tp2 --per-op` (ws-h34), so a TP=1 and a TP=2
/// table of the same bucket are the same arithmetic over the same grouping.
pub(crate) fn print_gemm_shapes(
    plan: &BlockPlan<'_>,
    indices: &[usize],
    per_us: &[f64],
    record_len: &BTreeMap<String, u64>,
    dram_gbs: f64,
    peak_tflops: f64,
    label: &str,
) -> f64 {
    // Group the GEMM family by (n, k, variant).
    /// `(count, bytes one op binds, per-instance in-chain µs)`, keyed by shape and variant.
    type Group = (usize, u64, Vec<f64>);
    let mut groups: BTreeMap<(u32, u32, String), Group> = BTreeMap::new();
    for (j, i) in indices.iter().enumerate() {
        let step = &plan.steps[*i];
        if family(step) != "GEMMs" {
            continue;
        }
        let Route::Kernel { descs, shape } = &step.route else {
            continue;
        };
        let bytes: u64 = step
            .bindings
            .iter()
            .filter_map(|(_, src)| match src {
                saber_exec::Source::Weight { key, .. } => record_len.get(key).copied(),
                _ => None,
            })
            .sum();
        let e = groups
            .entry((shape.n, shape.k, descs[0].kernel.clone()))
            .or_insert((0, bytes, Vec::new()));
        e.0 += 1;
        e.2.push(per_us[j]);
    }
    let mut rows: Vec<ShapeRow> = groups
        .into_iter()
        .map(|((n, k, kernel), (count, bytes, mut us))| {
            us.sort_by(f64::total_cmp);
            ShapeRow {
                n,
                k,
                kernel,
                count,
                bytes,
                us: us[us.len() / 2],
            }
        })
        .collect();
    rows.sort_by(|a, b| b.above_ms(dram_gbs).total_cmp(&a.above_ms(dram_gbs)));

    // The bucket's M: what every layer linear runs at, and the row count the FLOPs are counted
    // over. `Bucket` knows it for both phases.
    let rows_m = plan.bucket.m();
    let step_total: f64 = rows.iter().map(ShapeRow::total_ms).sum();
    let floor_total: f64 = rows.iter().map(|r| r.floor_ms(dram_gbs)).sum();
    println!();
    println!(
        "gemms     {label}: every layer linear in chain, sorted by milliseconds ABOVE its own \
         weight stream at {} GB/s",
        dram_gbs
    );
    println!(
        "          {:<14} {:<44} {:>4} {:>8} {:>8} {:>7} {:>8} {:>9} {:>9} {:>6}",
        "n x k",
        "variant",
        "n",
        "MiB/op",
        "µs/op",
        "GB/s",
        "TFLOP/s",
        "total ms",
        "above ms",
        "bound"
    );
    for r in &rows {
        // Which roof this shape is under. A GEMM at M rows does 2*M*N*K FLOP against a weight
        // stream of `bytes`: at a decode M the stream dominates by orders of magnitude, and at a
        // prefill M the arithmetic does. Saying which is what keeps "above the floor" honest --
        // the DRAM floor is not the floor for a compute-bound shape.
        let flops = 2.0 * f64::from(rows_m) * f64::from(r.n) * f64::from(r.k);
        let tflops = flops / (r.us * 1e-6) / 1e12;
        let mem_us = r.bytes as f64 / (dram_gbs * 1e9) * 1e6;
        let cmp_us = flops / (peak_tflops * 1e12) * 1e6;
        println!(
            "          {:<14} {:<44} {:>4} {:>8.2} {:>8.1} {:>7.0} {:>8.1} {:>9.3} {:>9.3} {:>6}",
            format!("{} x {}", r.n, r.k),
            r.kernel,
            r.count,
            r.bytes as f64 / (1024.0 * 1024.0),
            r.us,
            r.gbs(),
            tflops,
            r.total_ms(),
            r.above_ms(dram_gbs),
            if cmp_us > mem_us { "cmp" } else { "mem" }
        );
    }
    let above_total = step_total - floor_total;
    println!(
        "          {} shape(s), {:.3} ms in chain against a {:.3} ms floor: {:.3} ms above it",
        rows.len(),
        step_total,
        floor_total,
        above_total
    );
    let mut cum = 0.0;
    let mut named = 0usize;
    for r in &rows {
        if cum >= 0.8 * above_total {
            break;
        }
        cum += r.above_ms(dram_gbs);
        named += 1;
    }
    println!(
        "          {named} shape(s) account for {:.0}% of the {above_total:.3} ms above the floor: \
         {}",
        100.0 * cum / above_total,
        rows.iter()
            .take(named)
            .map(|r| format!("{}x{} @{:.0} GB/s", r.n, r.k, r.gbs()))
            .collect::<Vec<_>>()
            .join(", ")
    );
    step_total
}

/// Every op as its own sub-graph, timed in place: achieved GB/s per GEMM shape.
///
/// ws-h12 put the GEMM family at 27.0 ms against a 20.4 ms weight-stream floor, i.e. **6.7 ms above
/// it** at b1s5 — and said nothing about which shapes. This slices the step at OP boundaries (675
/// links) so every GEMM is timed with the cache state its position gives it, and divides the bytes
/// its records hold by that time.
///
/// **The instrument cost is measured, not assumed.** 675 links pay a graph launch and a host wait
/// each; the sum of the links against the single graph's wall gives the per-link cost, and it is
/// subtracted from every link before the bandwidth is computed. The correction is printed so the
/// reader can undo it.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn per_op_gemms(
    hip: &HipThread,
    ctx: &LaunchCtx<'_>,
    stream: &Stream,
    exec: &mut BlockExecutor,
    launcher: &mut Launcher,
    capture_counters: &std::sync::Arc<CaptureCounters>,
    plan: &BlockPlan<'_>,
    inputs: &BlockInputs,
    args: &StepCaptureArgs,
    label: &str,
    whole_ms: f64,
    record_len: &BTreeMap<String, u64>,
) -> anyhow::Result<()> {
    let proof = GraphSafe::from_flags(true, false).expect("graph_safe && !may_sync");
    // One sub-graph per launched op, in the plan's order.
    let mut links: Vec<(usize, CapturedBlock)> = Vec::new();
    for (i, step) in plan.steps.iter().enumerate() {
        if !matches!(step.route, Route::Kernel { .. }) {
            continue;
        }
        let sub = CapturedBlock::capture_steps(
            ctx,
            launcher,
            exec,
            &plan.steps[i..=i],
            inputs,
            stream,
            format!("op.{label}.{i}"),
            proof,
            capture_counters,
        )?;
        links.push((i, sub));
    }
    let nodes: u32 = links.iter().map(|(_, s)| s.declared_launches()).sum();
    println!();
    println!(
        "per-op    {label}: {} link(s), {nodes} nodes against the step's {} — {}",
        links.len(),
        plan.expected_launches(),
        if nodes == plan.expected_launches() {
            "MATCH"
        } else {
            "MISMATCH"
        }
    );

    let start_ev = hip.create_event(EventFlags::Default)?;
    let end_ev = hip.create_event(EventFlags::Default)?;
    let mut per: Vec<Vec<f64>> = vec![Vec::with_capacity(args.iters); links.len()];
    for pass in 0..(args.warmup + args.iters) {
        for (j, (_, sub)) in links.iter().enumerate() {
            hip.record_event(&start_ev, stream)?;
            sub.replay(ctx, stream)?;
            hip.record_event(&end_ev, stream)?;
            wait_bounded(ctx, stream, "an op link")?;
            let us = f64::from(hip.elapsed_ms(&start_ev, &end_ev)?) * 1000.0;
            if pass >= args.warmup {
                per[j].push(us);
            }
        }
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let raw: Vec<f64> = per.iter_mut().map(med).collect();
    let sum_us: f64 = raw.iter().sum();
    // The per-link instrument cost: what the chain costs over the single graph, spread evenly. It is
    // a cold start per link (ws-h7's finding), so it inflates every link by about the same amount
    // whatever the link does.
    let overhead_us = ((sum_us / 1000.0) - whole_ms).max(0.0) * 1000.0 / links.len() as f64;
    println!(
        "          the links sum to {:.3} ms against the single graph's {whole_ms:.3} ms: \
         {:+.1}% instrument, {overhead_us:.2} µs/link, subtracted below",
        sum_us / 1000.0,
        100.0 * ((sum_us / 1000.0) - whole_ms) / whole_ms
    );

    // The GEMM table, shared with `step-tp2 --per-op` (ws-h34: the TP=2 prefill had no per-shape
    // TFLOP/s until then).
    let indices: Vec<usize> = links.iter().map(|(i, _)| *i).collect();
    let per_us: Vec<f64> = raw.iter().map(|u| (u - overhead_us).max(0.0)).collect();
    let step_total = print_gemm_shapes(
        plan,
        &indices,
        &per_us,
        record_len,
        args.dram_gbs,
        args.peak_tflops,
        label,
    );
    // Everything that is not a layer linear, in chain, by kernel -- with the floor its own
    // geometry implies beside it (`op_traffic`), because "34 ms of silu" is not a work order
    // until it says how much of the 34 is the bytes.
    // `--detail SUBSTR`: one line per matching link, with the ops either side and the SAME op run
    // eagerly in its own position.
    //
    // The question this exists to answer: what is a sub-graph link's cost a function of? H13
    // subtracted 6.2 us/link at `b1s5` over 659 links and H12 measured +1.0% over 66; I18 saw
    // ~350 us on a 1 ms prefill op at TP=2. Those reconcile only if the cost scales with the op
    // rather than being per launch -- and if it does, the eager column is the primary number for
    // every prefill table. One op, three bucket sizes, both columns, is the measurement.
    if let Some(want) = args.detail.as_deref() {
        let mut rows: Vec<(usize, String, String, String, f64, f64)> = Vec::new();
        for (j, (i, _)) in links.iter().enumerate() {
            let step = &plan.steps[*i];
            if !step.name().contains(want) {
                continue;
            }
            let prev = j.checked_sub(1).map_or_else(
                || String::from("-"),
                |k| plan.steps[links[k].0].name().to_owned(),
            );
            let next = links.get(j + 1).map_or_else(
                || String::from("-"),
                |(n, _)| plan.steps[*n].name().to_owned(),
            );
            let mut eager: Vec<f64> = Vec::with_capacity(args.iters);
            for pass in 0..(args.warmup + args.iters) {
                hip.record_event(&start_ev, stream)?;
                exec.run_steps(ctx, launcher, stream, &plan.steps[*i..=*i], inputs, false)?;
                hip.record_event(&end_ev, stream)?;
                wait_bounded(ctx, stream, "an eager op link")?;
                if pass >= args.warmup {
                    eager.push(f64::from(hip.elapsed_ms(&start_ev, &end_ev)?) * 1000.0);
                }
            }
            eager.sort_by(f64::total_cmp);
            rows.push((
                *i,
                prev,
                step.name().to_owned(),
                next,
                raw[j],
                eager[eager.len() / 2],
            ));
        }
        if !rows.is_empty() {
            let (mut sg, mut se) = (0.0, 0.0);
            println!();
            println!(
                "detail    {label}: every link whose op contains `{want}`, with the ops either \
                 side and the same op run eagerly in place"
            );
            println!(
                "          {:>4} {:<30} {:<30} {:<30} {:>10} {:>10} {:>9}",
                "step", "previous op", "op", "next op", "graph µs", "eager µs", "link µs"
            );
            for (i, prev, op, next, g, e) in &rows {
                sg += g;
                se += e;
                println!(
                    "          {i:>4} {prev:<30} {op:<30} {next:<30} {g:>10.1} {e:>10.1} \
                     {:>9.1}",
                    g - e
                );
            }
            let (g, e) = (sg / rows.len() as f64, se / rows.len() as f64);
            println!(
                "          mean over {} link(s): graph {g:.1} µs, eager {e:.1} µs, link cost \
                 {:.1} µs = {:.0}% of the op",
                rows.len(),
                g - e,
                100.0 * (g - e) / e
            );
        }
    }

    let indices: Vec<usize> = links.iter().map(|(i, _)| *i).collect();
    let per_us: Vec<f64> = raw.iter().map(|u| (u - overhead_us).max(0.0)).collect();
    let rows = other_rows(plan, &indices, &per_us, record_len);
    println!();
    println!(
        "others    {label}: every op that is NOT a layer linear, in chain, against the floor its \
         own geometry implies"
    );
    print_others(&rows, args.dram_gbs, args.peak_tflops, step_total, whole_ms);
    print_family_summary(plan, &indices, &per_us, whole_ms, label);
    println!();
    println!(
        "          (in-chain µs have the {overhead_us:.2} µs/link instrument cost removed; a \
         LINEAR's bytes are the pack's own record lengths, so padding and the scale vectors \
         count, and a non-linear op's are `op_traffic`'s identity on its declared shape)"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use saber_kernels::rules::LaunchRules;
    use saber_kernels::saber::SaberRules;

    fn graph() -> Graph {
        Graph::qwen38_27b(ModelConfig::qwen38_27b(), 1).expect("graph")
    }

    /// The index laws a decode step needs, by width and by dtype. A wrong one is a kernel walking
    /// off a buffer, and `sidx` in particular is a STATE SLOT index: out of range is an OOB write.
    #[test]
    fn the_decode_index_tables_are_the_ones_the_pair_kernels_read() {
        let ids = [11i32, 12, 13];
        let t = |k: &str, batch: u32, q: u32| {
            index_table(k, Phase::Decode, batch, q, 2, 8192, &ids).expect(k)
        };
        // One id per ROW, cycling the prompt: batch 2 x q 5 is ten rows.
        assert_eq!(t("embed_tokens.ids", 2, 5).len(), 2 * 5 * 4);
        // cu_seqlens over the batch, batch + 1 entries.
        assert_eq!(
            t("layer.0.gdn.conv_update.cu", 2, 5),
            i32_bytes(&[0, 5, 10])
        );
        // One conv slot per sequence, from ONE and spaced by the verify width: each sequence owns
        // `q` consecutive snapshot slots, so sequence n starts at `1 + n * q`. This read
        // `[0, 1, 2]` until ws-h15, and slot 0 is the kernels' null slot — the first sequence of
        // every step measured before then updated no GDN state at all.
        assert_eq!(
            t("layer.0.gdn.conv_update.cache_idx", 3, 5),
            i32_bytes(&[1, 6, 11])
        );
        // Both kernels read `naccept[n] - 1` as an offset, so it is q, not 0.
        assert_eq!(
            t("layer.0.gdn.conv_update.naccept", 2, 5),
            i32_bytes(&[5, 5])
        );
        // One state-snapshot slot per candidate ROW, and the slot count is batch * (q + 2), so the
        // largest index here must be inside it.
        let (batch, q) = (2u32, 5u32);
        let sidx = t("layer.0.gdn.recurrent_update.sidx", batch, q);
        assert_eq!(
            sidx,
            i32_bytes(&(1..=(batch * q) as i32).collect::<Vec<_>>())
        );
        // The slot law `plan_for` uses must cover the largest index this table hands the kernel,
        // and the largest is now `batch * q` rather than `batch * q - 1`.
        let slots = batch * (q + 2);
        let max_sidx = batch * q;
        assert!(
            max_sidx < slots,
            "sidx {max_sidx} must index inside the {slots} slots the caches were sized for"
        );
        // has_init is a BYTE per sequence, and a decode step HAS carried state.
        assert_eq!(t("layer.0.gdn.conv_update.has_init", 4, 5), vec![1u8; 4]);
        // NEGATIVE CONTROL: an index this command has no law for is a refusal, not a zero buffer.
        assert!(index_table(
            "layer.0.gdn.decode.whatever",
            Phase::Decode,
            1,
            5,
            2,
            8192,
            &ids
        )
        .is_err());
    }

    /// The split is the engine's shape, and the combined plan is what it costs.
    ///
    /// No GPU: both plans are pure functions of the graph, so the arena and the launch counts can
    /// be asserted on the host. The arena is the part worth asserting — 970 MiB of logits that a
    /// sampler reads one row of is the whole finding, and it is visible here as a number.
    #[test]
    fn the_split_head_is_one_row_at_prefill_and_every_row_at_decode() {
        let g = graph();
        let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
        let mx = LaunchRules::prod_mxfp4().expect("rules");
        let saber = SaberRules::prod_saber().expect("saber");

        // ---- a prefill chunk: the head runs on ONE row, the body on 2048 -------------------
        let body = plan_body(
            &g,
            &layers,
            g.bucket_for(2048, Phase::Prefill),
            8192,
            Phase::Prefill,
            &mx,
            &saber,
        )
        .expect("body");
        let head = plan_head(&g, &body, 8192, Phase::Prefill, &mx, &saber).expect("head");
        assert_eq!(body.bucket.m(), 2048);
        assert_eq!(head.bucket.m(), 1, "the head runs on the sampled row");
        // The head reads exactly the two edges the row selection carries across.
        assert_eq!(
            head.inputs.len(),
            2,
            "the head carries the hidden state and the residual: {:?}",
            head.inputs
        );

        // ---- the same step in ONE plan: the head on every row -----------------------------
        let combined = plan_combined(
            &g,
            &layers,
            g.bucket_for(2048, Phase::Prefill),
            8192,
            Phase::Prefill,
            &mx,
            &saber,
        )
        .expect("combined");
        // 2048 rows x 248320 logits x 2 bytes = 970 MiB that the combined plan allocates and the
        // split one does not. Allow slack for the rest of the arena differing; the point is the
        // order of magnitude, and it is three of them.
        let split_bytes = body.arena_bytes + head.arena_bytes;
        assert!(
            combined.arena_bytes > split_bytes + (768 << 20),
            "the every-row head should cost ~970 MiB of logits: combined {} MiB vs split {} MiB",
            combined.arena_bytes >> 20,
            split_bytes >> 20
        );
        assert!(
            head.arena_bytes < (64 << 20),
            "a one-row head's arena is {} MiB",
            head.arena_bytes >> 20
        );

        // ---- a decode step: every row is sampled, so the two agree on rows ------------------
        let dbody = plan_body(
            &g,
            &layers,
            g.bucket_for(5, Phase::Decode),
            8192,
            Phase::Decode,
            &mx,
            &saber,
        )
        .expect("body");
        let dhead = plan_head(&g, &dbody, 8192, Phase::Decode, &mx, &saber).expect("head");
        assert_eq!(dbody.bucket.m(), 5);
        assert_eq!(
            dhead.bucket.m(),
            5,
            "every decode row is a sampled position"
        );
        // And there the combined plan is the same work, so its arena is not blown up.
        let dcombined = plan_combined(
            &g,
            &layers,
            g.bucket_for(5, Phase::Decode),
            8192,
            Phase::Decode,
            &mx,
            &saber,
        )
        .expect("combined");
        assert!(
            dcombined.arena_bytes <= dbody.arena_bytes + dhead.arena_bytes + (4 << 20),
            "at decode the split costs no extra arena: combined {} vs split {}",
            dcombined.arena_bytes,
            dbody.arena_bytes + dhead.arena_bytes
        );

        // ---- the budget a split step declares ---------------------------------------------
        // Body + head + the three this command issues (two row selections, one argmax), which is
        // the same arithmetic `saber_exec::forward::Plans::expected_launches` does.
        let declared = body.expected_launches()
            + head.expected_launches()
            + saber_exec::forward::Head::EXTRA_LAUNCHES;
        assert!(declared > body.expected_launches() + 2);
        assert_eq!(saber_exec::forward::Head::EXTRA_LAUNCHES, 3);
    }

    /// The per-layer slicing must TILE the step: every step in exactly one link, in order. A gap
    /// would silently drop ops from the chain and make the per-layer numbers add up to less than
    /// the step.
    #[test]
    fn the_site_slices_tile_the_step_exactly() {
        let g = graph();
        let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
        let plan = plan_combined(
            &g,
            &layers,
            g.bucket_for(5, Phase::Decode),
            8192,
            Phase::Decode,
            &LaunchRules::prod_mxfp4().expect("rules"),
            &SaberRules::prod_saber().expect("saber"),
        )
        .expect("plan");
        let mut slices: Vec<(String, std::ops::Range<usize>)> = Vec::new();
        let mut start = 0usize;
        let mut current = site_label(&plan.steps[0]);
        for (i, step) in plan.steps.iter().enumerate() {
            let l = site_label(step);
            if l != current {
                slices.push((current, start..i));
                current = l;
                start = i;
            }
        }
        slices.push((current, start..plan.steps.len()));
        // 1 embedding + 64 layers + 1 head.
        assert_eq!(
            slices.len(),
            66,
            "{:?}",
            slices.iter().map(|(n, _)| n).collect::<Vec<_>>()
        );
        assert_eq!(slices[0].0, "embedding");
        assert_eq!(slices[65].0, "head");
        // Contiguous, non-overlapping, covering everything.
        let mut next = 0usize;
        for (_, r) in &slices {
            assert_eq!(r.start, next);
            next = r.end;
        }
        assert_eq!(next, plan.steps.len());
        // And every launched op is in some slice with a family.
        for step in &plan.steps {
            if matches!(step.route, Route::Kernel { .. }) {
                assert_ne!(family(step), "other", "{}", step.name());
            }
        }
    }

    /// The family map is by SITE first: `lm_head` is a `gemm_w4a8` and must not be counted with the
    /// layer linears, because its N is the vocabulary.
    #[test]
    fn the_head_gemm_is_not_a_layer_gemm() {
        let g = graph();
        let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
        let plan = plan_combined(
            &g,
            &layers,
            g.bucket_for(5, Phase::Decode),
            8192,
            Phase::Decode,
            &LaunchRules::prod_mxfp4().expect("rules"),
            &SaberRules::prod_saber().expect("saber"),
        )
        .expect("plan");
        let head = plan
            .steps
            .iter()
            .find(|s| s.name() == "lm_head")
            .expect("the head GEMM");
        assert_eq!(family(head), "embed+head");
        let linear = plan
            .steps
            .iter()
            .find(|s| s.name() == "layer.0.mlp.gate_up")
            .expect("a layer linear");
        assert_eq!(family(linear), "GEMMs");
        assert_eq!(
            plan.steps.iter().filter(|s| family(s) == "GEMMs").count(),
            304,
            "the 304 layer linears"
        );
    }

    /// The non-linear floor is an arithmetic identity on the op's own shape, so it is checked as
    /// one: the numbers below are written out longhand from the graph's declared widths, not read
    /// back from the function.
    #[test]
    fn a_non_linear_ops_floor_is_its_own_geometry() {
        let g = graph();
        let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
        // `plan_combined` is what `plan_for` was when this test was written: every site in one
        // plan. The assertions are per-op geometry, so the head's row count does not enter them.
        let plan = plan_combined(
            &g,
            &layers,
            g.bucket_for(2048, Phase::Prefill),
            8192,
            Phase::Prefill,
            &LaunchRules::prod_mxfp4().expect("rules"),
            &SaberRules::prod_saber().expect("saber"),
        )
        .expect("plan");
        let at = |name: &str| {
            let step = plan
                .steps
                .iter()
                .find(|s| s.name() == name)
                .unwrap_or_else(|| panic!("{name} is in the prefill plan"));
            op_traffic(step.op, plan.bucket, 0)
        };
        let m = 2048u64;

        // `silu_mul_quant`: 2n bf16 in, n e4m3 out, one fp32 scale per row. n is HALF the
        // `gate_up` width: 34816 / 2 at TP=1, which is also `down_proj`'s K.
        let n = 17408u64;
        assert_eq!(
            at("layer.0.mlp.act").expect("derivable"),
            Traffic {
                bytes: m * n * 5 + m * 4,
                flops: 0
            }
        );
        // `add_rms_quant`: y + residual in, residual + e4m3 out, a scale per row, the weight once.
        let k = 5120u64;
        assert_eq!(
            at("layer.0.input_layernorm").expect("derivable"),
            Traffic {
                bytes: m * k * 7 + k * 2 + m * 4,
                flops: 0
            }
        );
        // Every 4th layer is full attention (`full_attention_interval`), so layer 3 is the
        // first one with a paged attention op at all.
        // Attention prefill is COMPUTE: `4 * head_dim` flops per causal (query, key) pair per
        // query head, and a chunk is the whole sequence.
        let t = at("layer.3.attn.prefill").expect("derivable");
        assert_eq!(
            t.bytes, 0,
            "the paged re-reads are not a number the graph holds"
        );
        assert_eq!(t.flops, 4 * 256 * 24 * (m * (m + 1) / 2));
        // 51.6 GFLOP against the 325 TFLOP/s roof is a 159 us floor; I15 measured 606 us in
        // chain, i.e. 85 TFLOP/s. The floor is a floor, not a prediction.
        assert!(
            (t.flops as f64 / 325e12 * 1e6 - 158.6).abs() < 1.0,
            "{} us",
            t.flops as f64 / 325e12 * 1e6
        );

        // NEGATIVE CONTROL: the GDN core's ops have no floor the graph can state, and the
        // function must say so rather than invent one. If this ever starts returning `Some`, the
        // table's `?` rows become claims and the doc above `Traffic` is wrong.
        assert_eq!(at("layer.0.gdn.chunk_scan"), None);
        assert_eq!(at("layer.0.gdn.conv_prep"), None);
    }
}
