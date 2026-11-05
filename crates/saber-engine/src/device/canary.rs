//! **T9: the prefill ordering canary** (under `--engine-race-check shadow`, prefill chunks only).
//!
//! T9's harness (`saber gdn-race`) ran the GDN prefill path on fixed inputs tens of thousands of
//! times per rung and saw every op's output bytes identical; what a solo repeat cannot see is the
//! ENGINE's launch train -- a write of op K that lands (or is changed) after op K+1 has read it.
//! This canary checks exactly that seam, inside the served chain, with no new code object: for
//! every op of a GDN layer's recurrent path (`in_proj_qkvz`, `in_proj_ba`, `conv_prep`,
//! `kkt_solve`, `chunk_scan`) the T6 digest kernel digests every arena range and state range the
//! op binds twice -- A right after the op, B right after the NEXT op in the chunk (its consumer) --
//! and the host compares A with B once the chunk has run. A pair that differs names a binding that
//! moved between its producer's end and its consumer's end: a late write, a stray writer or a
//! stale read's cause. Stream order is untouched (every digest is one more dispatch on the step's
//! own stream after the op it reads), and the host reads the record once per chunk, after every
//! dispatch of the chunk was enqueued.

use saber_exec::layer::{BlockInputs, BlockPlan, Source, StateAt};
use saber_exec::{ArgBindings, BoundStep, LaunchCtx, Launcher};
use saber_hip::{DeviceAddr, DevicePtr, HipThread, Stream};
use saber_kernels::race_digest::{self, KvWalk, KV_PARTS, PARTS};
use saber_kernels::rules::{ArgValue, LaunchDescriptor, Shape, WorkspaceKind};

use super::device_err;
use crate::forward::ForwardError;

/// The ops of a GDN layer's recurrent path the canary brackets (suffixes of the op names).
pub(super) const WATCHED: [&str; 5] = [
    ".gdn.in_proj_qkvz",
    ".gdn.in_proj_ba",
    ".gdn.conv_prep",
    ".gdn.kkt_solve",
    ".gdn.chunk_scan",
];

/// Descriptor table bytes and record bytes reserved once at boot.
const DESC_BYTES: u64 = 1 << 20;
const REC_BYTES: u64 = 4 << 20;

pub(super) struct PrefillCanary {
    rank: u32,
    desc: DevicePtr,
    rec: DevicePtr,
    /// Prefill chunks checked, pairs compared, pairs that moved.
    chunks: u64,
    pairs: u64,
    moved: u64,
    /// The NEGATIVE CONTROL (`--poison-boot canary:0`): B reads its first binding 8 bytes on.
    control: bool,
    /// T10: the target's K/V page bytes (one attention layer's page) and the pool's page count,
    /// set once the KV pool is allocated ([`PrefillCanary::set_kv`]); 0 = the K/V digest is off.
    kv_page_bytes: u64,
    kv_blocks: u32,
    /// T10: the layer record's online checks (see [`LayerTally`]).
    tally: LayerTally,
}

/// **T10: the layer record's online checks**, per rank, printed at the end of the run.
#[derive(Debug, Default)]
struct LayerTally {
    /// All-reduce calls self-checked.
    ar_calls: u64,
    /// Calls whose `ar_seq` counters after the call were not the ones predicted from the chunk's
    /// start (`c0[b] + i + 1`): something else advanced this rank's protocol counters.
    seq_wrong: u64,
    /// Calls whose flag words (the PEER's stores) read the predicted counters exactly.
    flag_ok: u64,
    /// ... read the predicted counters plus one on every block: the peer already started its next
    /// call on this region (legal: the double buffer is there for it).
    flag_ahead: u64,
    /// ... read anything else (a protocol fault, named per call).
    flag_odd: u64,
    /// Chunks whose counters at the start were NOT uniform over the call's blocks (the region is
    /// shared by launches of several grids; see the slot audit, `tests/ar_slot_audit.rs`).
    desync_chunks: u64,
    /// Attention layers whose K/V pages were digested.
    kv_layers: u64,
    /// Lines printed for anomalies (capped).
    printed: u64,
}

/// T10: one digest the layer record takes right after step `after` of the chunk.
struct Post {
    after: usize,
    /// First record word.
    word: usize,
    kind: PostKind,
}

enum PostKind {
    /// A regions launch over the table at `table`: stage names, region bytes; the self-check.
    Regions {
        table: u64,
        names: Vec<String>,
        bytes: Vec<u64>,
        launch: LaunchDescriptor,
        check: Option<ArCheck>,
    },
    /// The K/V pages the chunk's attention reads (the walk of `race_digest::resolve_kv`).
    Kv {
        name: String,
        launch: LaunchDescriptor,
        args: ArgBindings,
    },
}

/// T10: what the host predicts a reduce leaves in its protocol words: the counters `c0[b] + i + 1`
/// of blocks `0..nb`, at region indices `seq` and `flag` of the launch.
struct ArCheck {
    op: String,
    seq: usize,
    flag: usize,
    want: Vec<u32>,
}

/// T10: the stage prefix of a step (`layer.7.attn.all_reduce` -> `p.l7.attn.ar`,
/// `embed_tokens.all_reduce` -> `p.embed.ar`, `layer.3.attn.prefill` -> `p.l3.attn`).
pub(super) fn stage_of(step: &str) -> String {
    let s = step.strip_prefix("layer.").map_or_else(
        || step.replace("embed_tokens", "embed"),
        |r| format!("l{r}"),
    );
    let s = s
        .strip_suffix(".all_reduce")
        .map_or_else(|| s.clone(), |b| format!("{b}.ar"));
    let s = s.strip_suffix(".prefill").map_or(s.clone(), str::to_owned);
    format!("p.{s}")
}

fn ival(d: &LaunchDescriptor, n: &str) -> Option<u64> {
    match d.arg(n).map(|s| &s.value) {
        Some(ArgValue::I32 { ival } | ArgValue::I64 { ival }) => u64::try_from(*ival).ok(),
        _ => None,
    }
}

/// T10: the byte range of the one-shot reduce's push that block `b` of `nb` owns, for a message
/// of `units` units of `unit` bytes (the lossy kernel: chunks; the exact one: 16-byte words).
pub(super) fn block_range(b: u64, nb: u64, units: u64, unit: u64) -> (u64, u64) {
    let per = units.div_ceil(nb.max(1));
    let lo = (b * per).min(units);
    let hi = ((b + 1) * per).min(units);
    (lo * unit, hi * unit)
}

/// T10: consecutive blocks with the same scratch slot (`(c0[b] + call + 1) & 1`) merged into
/// runs `(start byte, end byte, slot)`; one run when the counters are uniform.
pub(super) fn slot_runs(c0: &[u32], call: u32, units: u64, unit: u64) -> Vec<(u64, u64, u32)> {
    let nb = c0.len() as u64;
    let mut runs: Vec<(u64, u64, u32)> = Vec::new();
    for (b, c) in c0.iter().enumerate() {
        let (lo, hi) = block_range(b as u64, nb, units, unit);
        if hi <= lo {
            continue;
        }
        let slot = c.wrapping_add(call).wrapping_add(1) & 1;
        match runs.last_mut() {
            Some(r) if r.2 == slot && r.1 == lo => r.1 = hi,
            _ => runs.push((lo, hi, slot)),
        }
    }
    runs
}

struct Pair {
    name: String,
    regions: Vec<(String, u64, u64)>,
    a_word: usize,
    b_word: usize,
}

impl PrefillCanary {
    pub(super) fn new(hip: &HipThread, rank: u32, control: bool) -> Result<Self, ForwardError> {
        let desc = hip
            .malloc(DESC_BYTES)
            .map_err(device_err("the prefill canary's tables"))?;
        let rec = hip
            .malloc(REC_BYTES)
            .map_err(device_err("the prefill canary's record"))?;
        println!(
            "racecheck canary rank {rank}: armed (T9) -- every prefill chunk's GDN recurrent-path \
             ops ({}) digested after the op and again after its consumer",
            WATCHED.join(" ")
        );
        if control {
            println!(
                "racecheck canary rank {rank}: NEGATIVE CONTROL -- every pair's second digest reads \
                 its first binding 8 bytes on; every pair must report MOVED"
            );
        }
        Ok(Self {
            rank,
            desc,
            rec,
            chunks: 0,
            pairs: 0,
            moved: 0,
            control,
            kv_page_bytes: 0,
            kv_blocks: 0,
            tally: LayerTally::default(),
        })
    }

    /// T10: the target's K/V geometry, once the pool exists: one attention layer's page bytes and
    /// the pool's page count (a page id at or past it is skipped by the walk).
    pub(super) fn set_kv(&mut self, page_bytes: u64, blocks: u32) {
        self.kv_page_bytes = page_bytes;
        self.kv_blocks = blocks;
    }

    /// The regions a bound step names: every arena range and state range of its plan step.
    fn regions(
        plan: &BlockPlan<'_>,
        inputs: &BlockInputs,
        name: &str,
        args: &ArgBindings,
    ) -> Vec<(String, u64, u64)> {
        let Some(step) = plan.steps.iter().find(|s| s.op.name == name) else {
            return Vec::new();
        };
        let mut out: Vec<(String, u64, u64)> = Vec::new();
        for (b, src) in &step.bindings {
            let bytes = match src {
                Source::Arena { bytes, .. } => *bytes,
                Source::State { key, at } => {
                    let per = step
                        .op
                        .state
                        .iter()
                        .find(|s| &s.name == key)
                        .and_then(|s| s.bytes_per_slot);
                    match (at, per) {
                        (StateAt::Slot { .. }, Some(p)) => p,
                        // a pool binding: the kernel finds its slot itself; the canary digests
                        // the slot this chunk writes only through `h0`/`ht`, so skip the pool
                        _ => continue,
                    }
                }
                _ => continue,
            };
            let _ = inputs;
            if let Some(a) = args.get(b) {
                if bytes > 0 && !out.iter().any(|r| r.1 == a.as_u64() && r.2 == bytes) {
                    out.push((b.clone(), a.as_u64(), bytes));
                }
            }
        }
        out
    }

    /// Run `steps` (a prefill chunk's bound ops, in order) with the canary's digests between them,
    /// then compare. Returns the dispatches of `steps` alone (the canary's are not the plan's) and
    /// T10's LAYER RECORD of the chunk: the stages `p.*` in forward order (every reduce's input,
    /// output, the peer's push as this rank read it and this rank's own push; every attention
    /// layer's K/V pages), for the shadow's next FULL proposal line.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub(super) fn run(
        &mut self,
        hip: &HipThread,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        stream: &Stream,
        plan: &BlockPlan<'_>,
        inputs: &BlockInputs,
        steps: &[BoundStep<'_>],
    ) -> Result<(u32, Vec<(String, u64)>), ForwardError> {
        // ---- the tables: a pair per watched op whose next step exists ----
        let mut pairs: Vec<(usize, Pair)> = Vec::new();
        let mut desc: Vec<u8> = Vec::new();
        let mut words = 0usize;
        for (i, s) in steps.iter().enumerate() {
            if i + 1 >= steps.len() || !WATCHED.iter().any(|w| s.name.ends_with(w)) {
                continue;
            }
            let regions = Self::regions(plan, inputs, s.name, &s.args);
            if regions.is_empty() {
                continue;
            }
            let n = regions.len() + 1;
            let a_word = words;
            let b_word = words + n * PARTS as usize;
            words += 2 * n * PARTS as usize;
            pairs.push((
                i,
                Pair {
                    name: s.name.to_owned(),
                    regions,
                    a_word,
                    b_word,
                },
            ));
        }
        // Two tables a pair: A's, then B's (the same, except under the control).
        let mut offs: Vec<(u64, u64)> = Vec::with_capacity(pairs.len());
        for (_, p) in &pairs {
            let a_off = desc.len() as u64;
            for (_, a, b) in &p.regions {
                desc.extend_from_slice(&a.to_le_bytes());
                desc.extend_from_slice(&b.to_le_bytes());
            }
            let b_off = desc.len() as u64;
            for (j, (_, a, b)) in p.regions.iter().enumerate() {
                let shift = if self.control && j == 0 && *b > 8 {
                    8
                } else {
                    0
                };
                desc.extend_from_slice(&(a + shift).to_le_bytes());
                desc.extend_from_slice(&(b - shift).to_le_bytes());
            }
            offs.push((a_off, b_off));
        }
        // ---- T10: the layer record's digests, after the canary's words ----
        let posts = self.layer_posts(hip, launcher, stream, plan, steps, &mut desc, &mut words)?;
        if desc.len() as u64 > DESC_BYTES || (words * 8) as u64 > REC_BYTES {
            return Err(ForwardError::Device(format!(
                "the prefill canary's tables ({} B) or record ({} B) outgrow the boot's",
                desc.len(),
                words * 8
            )));
        }
        if !desc.is_empty() {
            hip.memcpy_h2d_blocking(&self.desc, &desc, stream)
                .map_err(device_err("uploading the prefill canary's tables"))?;
        }
        let launches: Vec<LaunchDescriptor> = pairs
            .iter()
            .map(|(_, p)| race_digest::resolve_regions(p.regions.len() as u32, 0))
            .collect::<Result<_, _>>()
            .map_err(|e| ForwardError::Device(format!("the prefill canary's launch: {e}")))?;
        let at = |base: &DevicePtr, off: u64| -> Result<DeviceAddr, ForwardError> {
            base.addr().offset(off).ok_or_else(|| {
                ForwardError::Device("the prefill canary's address overflows".to_owned())
            })
        };
        let digest =
            |launcher: &mut Launcher, k: usize, second: bool| -> Result<(), ForwardError> {
                let (table, word) = if second {
                    (offs[k].1, pairs[k].1.b_word)
                } else {
                    (offs[k].0, pairs[k].1.a_word)
                };
                let args = ArgBindings::new(Shape::new(0, 0, 0))
                    .with("desc", at(&self.desc, table)?)
                    .with("extra", self.desc.addr())
                    .with("out", at(&self.rec, word as u64 * 8)?);
                launcher
                    .launch_op(ctx, &[&launches[k]], &args, stream)
                    .map_err(device_err("a prefill canary digest"))?;
                Ok(())
            };
        // ---- the chunk, with A after each watched op and B after its consumer ----
        let mut total = 0u32;
        let mut pending: Option<usize> = None;
        let mut next = 0usize;
        for (i, s) in steps.iter().enumerate() {
            let refs: Vec<&LaunchDescriptor> = s.descs.iter().collect();
            total += launcher
                .launch_op(ctx, &refs, &s.args, stream)
                .map_err(device_err("a dispatch"))?
                .launches;
            if let Some(k) = pending.take() {
                digest(launcher, k, true)?;
            }
            if next < pairs.len() && pairs[next].0 == i {
                digest(launcher, next, false)?;
                pending = Some(next);
                next += 1;
            }
            for p in posts.iter().filter(|p| p.after == i) {
                let out = at(&self.rec, p.word as u64 * 8)?;
                match &p.kind {
                    PostKind::Regions { table, launch, .. } => {
                        let args = ArgBindings::new(Shape::new(0, 0, 0))
                            .with("desc", at(&self.desc, *table)?)
                            .with("extra", self.desc.addr())
                            .with("out", out);
                        launcher
                            .launch_op(ctx, &[launch], &args, stream)
                            .map_err(device_err("a layer-record digest"))?;
                    }
                    PostKind::Kv { launch, args, .. } => {
                        let args = args.clone().with("out", out);
                        launcher
                            .launch_op(ctx, &[launch], &args, stream)
                            .map_err(device_err("a layer-record K/V digest"))?;
                    }
                }
            }
        }
        // ---- compare: once, after every dispatch of the chunk ----
        let mut stages = Vec::new();
        if words > 0 {
            let mut host = vec![0u8; words * 8];
            hip.memcpy_d2h_blocking(&mut host, &self.rec, stream)
                .map_err(device_err("reading the prefill canary's record"))?;
            stages = self.layer_stages(&posts, &host);
            for (_, p) in &pairs {
                self.pairs += 1;
                for (ri, (b, _, bytes)) in p.regions.iter().enumerate() {
                    let o = ri * PARTS as usize;
                    let (a0, b0) = ((p.a_word + o) * 8, (p.b_word + o) * 8);
                    let len = PARTS as usize * 8;
                    if host[a0..a0 + len] != host[b0..b0 + len] {
                        self.moved += 1;
                        println!(
                            "racecheck canary rank {}: MOVED prefill chunk {} op `{}` binding \
                             `{b}` ({bytes} B) differs between its end and its consumer's end",
                            self.rank, self.chunks, p.name
                        );
                        break;
                    }
                }
            }
        }
        self.chunks += 1;
        Ok((total, stages))
    }

    /// **T10: plan the layer record of one chunk**: after every all-reduce, one regions digest of
    /// its input and output (the live rows), the peer's push as this rank read it out of its own
    /// scratch slot and this rank's own push (paired post hoc across the ranks: my read of you =
    /// your send), and the protocol words (`ar_seq` = my counters, `ar_flags` = the peer's stores,
    /// both predicted on the host from the counters read at the chunk's start); after every
    /// attention layer's prefill, its K/V pages. Tables go into `desc`, record words after `words`.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn layer_posts(
        &mut self,
        hip: &HipThread,
        launcher: &Launcher,
        stream: &Stream,
        plan: &BlockPlan<'_>,
        steps: &[BoundStep<'_>],
        desc: &mut Vec<u8>,
        words: &mut usize,
    ) -> Result<Vec<Post>, ForwardError> {
        let live_rows = u64::from(plan.prefill_q_len());
        let (in_name, out_name) = saber_kernels::saber::ar_band_names(0);
        let mut posts = Vec::new();
        // Per reduce family: (calls so far in this chunk, the counters at the chunk's start).
        let mut fam: Vec<(String, u32, Vec<u32>)> = Vec::new();
        let mut desync = false;
        for (i, s) in steps.iter().enumerate() {
            let Some(d) = s.descs.first() else {
                continue;
            };
            let wht = d.op == "allreduce_wht";
            if (wht || d.op == "allreduce_exact") && s.descs.len() == 1 {
                let stage = stage_of(s.name);
                let row_elems = plan
                    .steps
                    .iter()
                    .find(|p| p.op.name == s.name)
                    .and_then(|p| match p.op.geometry {
                        saber_model::graph::Geometry::AllReduce {
                            elems_per_token, ..
                        } => Some(u64::from(elems_per_token)),
                        _ => None,
                    })
                    .unwrap_or(0);
                let geo = if wht {
                    let bits: u64 = if d.kernel.contains("wht4") {
                        4
                    } else if d.kernel.contains("wht5") {
                        5
                    } else {
                        6
                    };
                    let cg = if bits % 2 == 0 { 32 } else { 64 };
                    ival(d, "n_groups")
                        .zip(ival(d, "n_chunks"))
                        .map(|(ng, nc)| {
                            (
                                ng * 64 * 2,
                                nc,
                                cg * 64 * bits / 8,
                                // `ar_loc_pack` is a SCRATCH slab (not peer-visible): its index
                                // is its rank among the descriptor's Scratch workspaces, and
                                // right after the reduce it still holds this call's own push.
                                d.workspaces
                                    .iter()
                                    .filter(|w| w.kind == WorkspaceKind::Scratch)
                                    .position(|w| w.name == "ar_loc_pack")
                                    .and_then(|i| launcher.scratch_slab_addr(i)),
                                ival(d, "slot_stride_bytes").unwrap_or(0),
                            )
                        })
                } else {
                    ival(d, "n16").map(|n16| {
                        (
                            n16 * 16,
                            n16,
                            16,
                            s.args.get(&in_name),
                            ival(d, "slot_stride16").unwrap_or(0) * 16,
                        )
                    })
                };
                let Some((msg, units, unit, sent, stride)) = geo else {
                    continue;
                };
                let live = (live_rows * row_elems * 2).min(msg);
                let mut names = Vec::new();
                let mut regions: Vec<(u64, u64)> = Vec::new();
                for (sfx, b) in [("in", &in_name), ("out", &out_name)] {
                    if let Some(a) = s.args.get(b) {
                        names.push(format!("{stage}.{sfx}"));
                        regions.push((a.as_u64(), live));
                    }
                }
                // The self-check: the region set this call's key resolves to.
                let nb = d.grid[0] as usize;
                let scratch = launcher.ipc_region_addr(d, &s.args, "ar_scratch");
                let seq = launcher.ipc_region_addr(d, &s.args, "ar_seq");
                let flags = launcher.ipc_region_addr(d, &s.args, "ar_flags");
                let mut check = None;
                if let (Some(scratch), Some(seq), Some(flags), Some(sent)) =
                    (scratch, seq, flags, sent)
                {
                    let k = match fam.iter().position(|f| f.0 == d.op) {
                        Some(k) => k,
                        None => {
                            // The counters at the chunk's start: the stream is idle here (the
                            // previous chunk's record read synchronised it), so no reduce of
                            // this rank is in flight.
                            let mut raw = vec![0u8; nb * 4];
                            hip.memcpy_d2h_addr_blocking(&mut raw, seq, stream)
                                .map_err(device_err("reading the reduce counters"))?;
                            let c0: Vec<u32> = raw
                                .chunks_exact(4)
                                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                .collect();
                            desync |= c0.iter().any(|c| *c != c0[0]);
                            fam.push((d.op.clone(), 0, c0));
                            fam.len() - 1
                        }
                    };
                    let (_, call, c0) = &fam[k];
                    if c0.len() == nb {
                        let runs = slot_runs(c0, *call, units, unit);
                        let many = runs.len() > 1;
                        for (r, (lo, hi, slot)) in runs.iter().enumerate() {
                            let sfx = if many {
                                format!(".r{r}")
                            } else {
                                String::new()
                            };
                            names.push(format!("{stage}.peer{sfx}"));
                            regions
                                .push((scratch.as_u64() + u64::from(*slot) * stride + lo, hi - lo));
                            names.push(format!("{stage}.sent{sfx}"));
                            regions.push((sent.as_u64() + lo, hi - lo));
                        }
                        let want: Vec<u32> = c0
                            .iter()
                            .map(|c| c.wrapping_add(*call).wrapping_add(1))
                            .collect();
                        names.push(format!("{stage}.seq"));
                        regions.push((seq.as_u64(), nb as u64 * 4));
                        names.push(format!("{stage}.flag"));
                        regions.push((flags.as_u64(), nb as u64 * 4));
                        check = Some(ArCheck {
                            op: s.name.to_owned(),
                            seq: names.len() - 2,
                            flag: names.len() - 1,
                            want,
                        });
                    }
                    fam[k].1 += 1;
                }
                let table = desc.len() as u64;
                for (a, b) in &regions {
                    desc.extend_from_slice(&a.to_le_bytes());
                    desc.extend_from_slice(&b.to_le_bytes());
                }
                let launch = race_digest::resolve_regions(regions.len() as u32, 0)
                    .map_err(|e| ForwardError::Device(format!("a layer-record launch: {e}")))?;
                posts.push(Post {
                    after: i,
                    word: *words,
                    kind: PostKind::Regions {
                        table,
                        names,
                        bytes: regions.iter().map(|r| r.1).collect(),
                        launch,
                        check,
                    },
                });
                *words += (regions.len() + 1) * PARTS as usize;
            } else if s.name.ends_with(".attn.prefill") && self.kv_page_bytes > 0 {
                let (Some(bt), Some(sl), Some(kv)) = (
                    s.args.get("block_table"),
                    s.args.get("seqused_k"),
                    s.args.get("kv"),
                ) else {
                    continue;
                };
                let (Some(reqs), Some(stride), Some(page)) = (
                    ival(d, "num_seqs"),
                    ival(d, "max_blocks"),
                    ival(d, "block_size"),
                ) else {
                    continue;
                };
                let walk = KvWalk {
                    reqs: reqs as u32,
                    bt_stride: stride as u32,
                    page_tokens: page as u32,
                    layers: 1,
                    blocks: self.kv_blocks,
                    per_layer: 0,
                    block_bytes: self.kv_page_bytes,
                };
                let launch = race_digest::resolve_kv(walk)
                    .map_err(|e| ForwardError::Device(format!("a layer-record K/V launch: {e}")))?;
                let args = ArgBindings::new(Shape::new(0, 0, 0))
                    .with("block_table", bt)
                    .with("seq_lens", sl)
                    .with("kbase", kv)
                    .with("vbase", kv);
                posts.push(Post {
                    after: i,
                    word: *words,
                    kind: PostKind::Kv {
                        name: stage_of(s.name),
                        launch,
                        args,
                    },
                });
                *words += walk.partials() as usize;
            }
        }
        if desync {
            self.tally.desync_chunks += 1;
        }
        Ok(posts)
    }

    /// **T10: the layer record's stages** from the chunk's record, in forward order, and its
    /// online checks: every reduce's counters must be the predicted ones, and its flag words
    /// (the peer's stores) the predicted ones or one ahead on every block.
    fn layer_stages(&mut self, posts: &[Post], host: &[u8]) -> Vec<(String, u64)> {
        let word = |i: usize| -> u64 {
            let o = i * 8;
            host.get(o..o + 8)
                .map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap_or_default()))
        };
        let p = PARTS as usize;
        let mut out = Vec::new();
        for post in posts {
            match &post.kind {
                PostKind::Regions {
                    names,
                    bytes,
                    check,
                    ..
                } => {
                    let digs: Vec<u64> = (0..names.len())
                        .map(|j| {
                            let parts: Vec<u64> =
                                (0..p).map(|q| word(post.word + j * p + q)).collect();
                            race_digest::fold(&parts, bytes[j])
                        })
                        .collect();
                    if let Some(c) = check {
                        self.tally.ar_calls += 1;
                        let le = |v: &[u32], add: u32| -> Vec<u8> {
                            v.iter()
                                .flat_map(|x| x.wrapping_add(add).to_le_bytes())
                                .collect()
                        };
                        let want = race_digest::host_digest(&le(&c.want, 0));
                        let ahead = race_digest::host_digest(&le(&c.want, 1));
                        let seq_ok = digs[c.seq] == want;
                        let flag = digs[c.flag];
                        let verdict = if flag == want {
                            self.tally.flag_ok += 1;
                            None
                        } else if flag == ahead {
                            self.tally.flag_ahead += 1;
                            None
                        } else {
                            self.tally.flag_odd += 1;
                            Some("flag words neither the counters nor all one ahead (a partial lead or a fault)")
                        };
                        if !seq_ok {
                            self.tally.seq_wrong += 1;
                        }
                        if (!seq_ok || verdict.is_some()) && self.tally.printed < 64 {
                            self.tally.printed += 1;
                            println!(
                                "racecheck ld rank {}: PROTOCOL chunk {} `{}` counters {} | {} \
                                 (c0[0] {} want[0] {})",
                                self.rank,
                                self.chunks,
                                c.op,
                                if seq_ok {
                                    "as predicted"
                                } else {
                                    "NOT as predicted"
                                },
                                verdict.unwrap_or("flags ok"),
                                c.want.first().map_or(0, |w| w.wrapping_sub(1)),
                                c.want.first().copied().unwrap_or(0),
                            );
                        }
                    }
                    for (j, n) in names.iter().enumerate() {
                        if n.ends_with(".seq") || n.ends_with(".flag") {
                            continue;
                        }
                        out.push((n.clone(), digs[j]));
                    }
                }
                PostKind::Kv { name, .. } => {
                    self.tally.kv_layers += 1;
                    let parts: Vec<u64> = (0..KV_PARTS as usize)
                        .map(|q| word(post.word + q))
                        .collect();
                    out.push((format!("{name}.kv"), race_digest::fold_kv(&parts)));
                }
            }
        }
        out
    }
}

impl Drop for PrefillCanary {
    fn drop(&mut self) {
        println!(
            "racecheck canary rank {}: prefill chunks {} | pairs checked {} | moved {}",
            self.rank, self.chunks, self.pairs, self.moved
        );
        let t = &self.tally;
        println!(
            "racecheck ld rank {}: reduces self-checked {} | counters not as predicted {} | flags \
             = counters {} | flags one ahead {} | flags otherwise {} | chunks with non-uniform \
             counters {} | K/V layers digested {}",
            self.rank,
            t.ar_calls,
            t.seq_wrong,
            t.flag_ok,
            t.flag_ahead,
            t.flag_odd,
            t.desync_chunks,
            t.kv_layers
        );
    }
}

#[cfg(test)]
mod tests {
    use super::WATCHED;
    use saber_model::graph::{Graph, ModelConfig};

    /// T10: the layer record's stage names, and the slot runs the self-check digests: uniform
    /// counters give ONE run over the whole push; the control (counters that differ in parity
    /// across the blocks, the shared-region state) splits it where the parity changes.
    #[test]
    fn layer_record_stages_and_slot_runs() {
        use super::{block_range, slot_runs, stage_of};
        assert_eq!(stage_of("layer.7.attn.all_reduce"), "p.l7.attn.ar");
        assert_eq!(stage_of("layer.12.mlp.all_reduce"), "p.l12.mlp.ar");
        assert_eq!(stage_of("layer.0.gdn.all_reduce"), "p.l0.gdn.ar");
        assert_eq!(stage_of("embed_tokens.all_reduce"), "p.embed.ar");
        assert_eq!(stage_of("layer.3.attn.prefill"), "p.l3.attn");
        assert_eq!(block_range(3, 8, 100, 1536), (39 * 1536, 52 * 1536));
        assert_eq!(block_range(7, 8, 100, 1536), (91 * 1536, 100 * 1536));
        assert_eq!(slot_runs(&[4; 8], 0, 100, 1536), vec![(0, 100 * 1536, 1)]);
        assert_eq!(slot_runs(&[4; 8], 1, 100, 1536), vec![(0, 100 * 1536, 0)]);
        let r = slot_runs(&[4, 4, 4, 5, 5, 5, 5, 5], 0, 100, 1536);
        assert_eq!(r, vec![(0, 39 * 1536, 1), (39 * 1536, 100 * 1536, 0)]);
        // a block with nothing to push (more blocks than units) is skipped, not a zero-length run
        assert_eq!(slot_runs(&[0; 4], 0, 2, 16), vec![(0, 32, 1)]);
    }

    /// Every watched suffix names one op in each of the 48 GDN layers of the served graph, so the
    /// canary cannot go silently blind to an op renamed in the graph.
    #[test]
    fn every_watched_op_is_in_every_gdn_layer() {
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("graph");
        for w in WATCHED {
            let n = g.ops.iter().filter(|o| o.name.ends_with(w)).count();
            assert_eq!(n, 48, "`{w}` names {n} ops, not one per GDN layer");
        }
        // the negative control: a suffix the graph does not carry names nothing
        assert_eq!(
            g.ops
                .iter()
                .filter(|o| o.name.ends_with(".gdn.not_an_op"))
                .count(),
            0
        );
    }
}
