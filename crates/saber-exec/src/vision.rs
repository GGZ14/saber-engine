//! The vision tower on the HipThread (workstream D2).
//!
//! D1 planned the tower (`saber_kernels::vision::plan_tower`: 222 launches over the pack's 333
//! `model.visual.*` records) and ran it from a Python gate. This is the same plan as an EXECUTOR
//! of the execution plane, beside [`crate::layer::BlockExecutor`]:
//!
//! * **one arena**, allocated once at construction and never reallocated (I1), laid out by
//!   `saber_kernels::vision::tower_arena` for the LARGEST image the engine admits, with buffers
//!   that are never live together sharing bytes (the layout is replayed against the launch list
//!   by `check_arena` before it is used; 1.011 GiB at prod's 65,536-patch ceiling instead of 2.380
//!   unaliased);
//! * **records bound by name** from the addresses the caller's upload placed them at; a record
//!   the plan reads and the caller did not bind is a refusal naming it, and so is a
//!   `model.visual.*` record the plan does NOT read (an unread weight is how a missing op hides);
//! * **per image**: the host tables (pos-embed gather, 2-D rope, cu_seqlens) and the pixel rows
//!   go up through the pinned staging buffer as `Job::MemcpyH2d` batches (`RawJob::lower` checks
//!   every range before anything is posted), then the plan's launches go out through the same
//!   [`Launcher`] every text op takes, and the launch count is asserted against the rules'
//!   declaration (rule 7). Eager, or captured once per grid set as a [`CapturedBucket`] and
//!   replayed.
//!
//! The output rows `[tokens, out_hidden]` bf16 are left at the arena's `out` slot
//! ([`VisionOut`]), which is what D4's embed-row scatter reads.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use std::sync::Arc;

use saber_hip::{DeviceAddr, DevicePtr, HipThread, Job, PinnedBuf, Stream};
use saber_kernels::rules::{ArgValue, Shape};
use saber_kernels::vision::{
    plan_tower, pos_table, rope_table, tower_arena, Grid, TowerArena, TowerPlan, VisionRules,
    OUT_BUFFER,
};

use crate::capture::{CapturableOp, CapturedBucket, GraphSafe};
use crate::counters::CaptureCounters;
use crate::error::{Error, Result};
use crate::launch::{ArgBindings, LaunchCtx, Launcher};

/// The record-name prefix of every tower weight in the pack.
pub const VISION_PREFIX: &str = "model.visual.";

/// Where one buffer name of a tower launch resolves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisionSource {
    /// A pack record, by name.
    Record(String),
    /// A byte offset in the tower's arena.
    Arena(u64),
}

/// Resolve every buffer a plan's launches name: records against `records` (the names the caller
/// holds addresses for), everything else against the arena layout. Host-only, so the binding is
/// tested without a card.
///
/// # Errors
/// A record the plan reads that is not in `records`, a `model.visual.*` record in `records` that
/// no launch reads, or a buffer that is neither.
pub fn bind_names(
    plan: &TowerPlan,
    arena: &TowerArena,
    records: &BTreeSet<String>,
) -> Result<Vec<Vec<(String, VisionSource)>>> {
    let mut read: BTreeSet<&str> = BTreeSet::new();
    let mut out = Vec::with_capacity(plan.launches.len());
    for ld in &plan.launches {
        let mut per = Vec::new();
        for s in &ld.args {
            let ArgValue::Buffer { buffer } = &s.value else {
                continue;
            };
            let src = if buffer.starts_with(VISION_PREFIX) {
                if !records.contains(buffer) {
                    return Err(Error::Refused(format!(
                        "vision: `{}` reads the record `{buffer}`, which is not resident",
                        ld.op
                    )));
                }
                read.insert(buffer);
                VisionSource::Record(buffer.clone())
            } else {
                let slot = arena.slot(buffer).ok_or_else(|| {
                    Error::Refused(format!(
                        "vision: `{}` names `{buffer}`, which is neither a record nor an arena slot",
                        ld.op
                    ))
                })?;
                VisionSource::Arena(slot.offset)
            };
            per.push((buffer.clone(), src));
        }
        out.push(per);
    }
    let unread: Vec<&String> = records
        .iter()
        .filter(|r| r.starts_with(VISION_PREFIX) && !read.contains(r.as_str()))
        .collect();
    if !unread.is_empty() {
        return Err(Error::Refused(format!(
            "vision: {} resident `model.visual.*` record(s) no launch reads: {:?}{} -- an unread \
             weight is an op the plan lost",
            unread.len(),
            unread.iter().take(3).collect::<Vec<_>>(),
            if unread.len() > 3 { ", …" } else { "" }
        )));
    }
    Ok(out)
}

/// The host bytes one image set uploads before its launches: `(arena slot, bytes)` pairs, pixel
/// rows last. Host-only.
///
/// # Errors
/// `pixels` not `patches x in_features` long.
pub fn host_inputs(
    rules: &VisionRules,
    plan: &TowerPlan,
    pixels: &[f32],
) -> Result<Vec<(&'static str, Vec<u8>)>> {
    let m = &rules.model;
    let kin = (m.in_channels * m.temporal_patch * m.patch * m.patch) as usize;
    let want = plan.patches as usize * kin;
    if pixels.len() != want {
        return Err(Error::Refused(format!(
            "vision: {} pixel values for {} patches x {kin} (want {want})",
            pixels.len(),
            plan.patches
        )));
    }
    let (idx, wts) = pos_table(rules, &plan.grids);
    let (cos, sin) = rope_table(rules, &plan.grids);
    let le32 = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    Ok(vec![
        ("pos_idx", le32(&idx)),
        (
            "pos_wts",
            wts.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ),
        (
            "rope_cos",
            cos.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ),
        (
            "rope_sin",
            sin.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ),
        ("cu_seqlens", le32(&plan.cu_seqlens)),
        (
            "pixels",
            pixels.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ),
    ])
}

/// Where a tower run left its output rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionOut {
    /// Device address of row 0.
    pub addr: DeviceAddr,
    /// The same, as an arena offset.
    pub offset: u64,
    /// Rows: one per `<|image_pad|>` token.
    pub tokens: u32,
    /// `tokens x out_hidden x 2`.
    pub bytes: u64,
    /// Dispatches this run issued (eager) or the graph holds (captured).
    pub launches: u32,
}

/// The tower's executor: the arena, the bound records, the staging buffer, the captured graphs.
pub struct VisionExecutor {
    rules: VisionRules,
    arena_plan: TowerArena,
    /// FN55c: shared, so a lazy owner (`--fn-vision lazy`) can hand the tower memory it lends.
    /// The allocation the tower's arena lives in: its own (`new`), a lent region (FN55c), or -- the
    /// STREAMED form (P27-11, [`VisionExecutor::new_streamed`]) -- the engine's shared prefill arena,
    /// idle while the tower runs (the image forward precedes the chunk that scatters its rows).
    arena: Rc<DevicePtr>,
    /// The tower arena's offset inside `arena` (0 for an own arena; past the weight ring when lent).
    base: u64,
    /// P27-11: the weights in pinned host memory, streamed per image through a two-slot ring.
    streamed: Option<Streamed>,
    staging: PinnedBuf,
    weights: BTreeMap<String, DeviceAddr>,
    graphs: BTreeMap<Vec<Grid>, CapturedBucket>,
    counters: Arc<CaptureCounters>,
}

impl VisionExecutor {
    /// Lay out and allocate the arena for images of up to `max_patches` patches, and take the
    /// records' addresses. Every record the tower reads must be here and every `model.visual.*`
    /// record here must be read (checked on the host before the allocation).
    ///
    /// # Errors
    /// The layout (a cap past `max_image_patches`, a layout the replay refuses), a missing or
    /// unread record, or the allocation.
    pub fn new(
        hip: &HipThread,
        rules: VisionRules,
        max_patches: u32,
        weights: BTreeMap<String, DeviceAddr>,
        staging_bytes: usize,
    ) -> Result<Self> {
        Self::new_in(hip, rules, max_patches, weights, staging_bytes, None)
    }

    /// FN55c: [`Self::new`] with the arena in `arena` when given (memory the caller holds: the
    /// lazy tower's region, lent to the expert tier while no tower is resident); refused when it
    /// is smaller than the layout.
    ///
    /// # Errors
    /// As [`Self::new`], or a given arena smaller than the layout.
    pub fn new_in(
        hip: &HipThread,
        rules: VisionRules,
        max_patches: u32,
        weights: BTreeMap<String, DeviceAddr>,
        staging_bytes: usize,
        arena: Option<std::rc::Rc<DevicePtr>>,
    ) -> Result<Self> {
        let arena_plan =
            tower_arena(&rules, max_patches).map_err(|e| Error::Refused(e.to_string()))?;
        let probe = plan_tower(
            &rules,
            &[Grid {
                t: 1,
                h: rules.model.spatial_merge,
                w: rules.model.spatial_merge,
            }],
        )
        .map_err(|e| Error::Refused(e.to_string()))?;
        let names: BTreeSet<String> = weights.keys().cloned().collect();
        bind_names(&probe, &arena_plan, &names)?;
        let arena = match arena {
            Some(a) => {
                if a.len() < arena_plan.bytes.max(1) {
                    return Err(Error::Refused(format!(
                        "vision: the given arena is {} B, the tower's layout needs {} B",
                        a.len(),
                        arena_plan.bytes
                    )));
                }
                a
            }
            None => std::rc::Rc::new(hip.malloc(arena_plan.bytes.max(1))?),
        };
        let staging = hip.host_malloc(staging_bytes.max(1 << 20) as u64)?;
        Ok(VisionExecutor {
            rules,
            arena_plan,
            arena,
            base: 0,
            streamed: None,
            staging,
            weights,
            graphs: BTreeMap::new(),
            counters: Arc::new(CaptureCounters::default()),
        })
    }

    /// The arena itself (D4: the embed-row scatter copies out of its `out` slot).
    #[must_use]
    pub fn arena(&self) -> &DevicePtr {
        self.arena.as_ref()
    }

    /// The arena layout.
    #[must_use]
    pub fn layout(&self) -> &TowerArena {
        &self.arena_plan
    }

    /// Arena bytes allocated.
    #[must_use]
    pub fn arena_bytes(&self) -> u64 {
        if self.streamed.is_some() {
            self.base + self.arena_plan.bytes
        } else {
            self.arena.len()
        }
    }

    /// P27-11: the streamed form's facts, `None` for a resident tower.
    #[must_use]
    pub fn streamed(&self) -> Option<&Streamed> {
        self.streamed.as_ref()
    }

    /// Where the last run's output rows are: the arena's `out` slot (resident), or the pinned
    /// host copy the streamed form takes before the prefill arena is reused.
    ///
    /// # Errors
    /// No `out` slot.
    pub fn out_source(&self) -> Result<OutSource<'_>> {
        if let Some(st) = &self.streamed {
            return Ok(OutSource::Host(&st.host_out));
        }
        let slot = self
            .arena_plan
            .slot(OUT_BUFFER)
            .ok_or_else(|| Error::Refused("vision: no `out` slot".to_owned()))?;
        Ok(OutSource::Device {
            arena: &self.arena,
            offset: self.base + slot.offset,
            bytes: slot.bytes,
        })
    }

    /// The plan for `grids`, refused by name past this executor's arena.
    ///
    /// # Errors
    /// The planner's refusals, or an image larger than the arena was sized for.
    pub fn plan(&self, grids: &[Grid]) -> Result<TowerPlan> {
        let l: u64 = grids.iter().map(|g| u64::from(g.patches())).sum();
        if l > u64::from(self.arena_plan.max_patches) {
            return Err(Error::Refused(format!(
                "vision: {l} patches exceed this engine's tower arena of {} patches \
                 (`--vision-max-patches`; prod's ceiling is {})",
                self.arena_plan.max_patches, self.rules.kernels.max_image_patches
            )));
        }
        plan_tower(&self.rules, grids).map_err(|e| Error::Refused(e.to_string()))
    }

    fn bindings(&self, plan: &TowerPlan) -> Result<Vec<ArgBindings>> {
        let names: BTreeSet<String> = self.weights.keys().cloned().collect();
        let resolved = bind_names(plan, &self.arena_plan, &names)?;
        let mut out = Vec::with_capacity(resolved.len());
        for per in resolved {
            let mut b = ArgBindings::new(Shape::new(plan.patches, self.rules.model.hidden, 0));
            for (name, src) in per {
                let addr = match src {
                    VisionSource::Record(r) => self.weights[&r],
                    VisionSource::Arena(off) => {
                        self.arena.sub_addr(self.base + off).ok_or_else(|| {
                            Error::Refused(format!("vision: `{name}` at {off} is past the arena"))
                        })?
                    }
                };
                b.bind(&name, addr);
            }
            out.push(b);
        }
        Ok(out)
    }

    /// Upload the host tables and the pixel rows for `plan`, through the pinned staging buffer:
    /// one `run_batch` of `MemcpyH2d` jobs + a stream sync per staging-sized chunk.
    ///
    /// # Errors
    /// The pixel count, a range past a slot, or the copies.
    pub fn upload(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        plan: &TowerPlan,
        pixels: &[f32],
    ) -> Result<()> {
        let inputs = host_inputs(&self.rules, plan, pixels)?;
        let cap = self.staging.len();
        // pack as many whole inputs (or chunks of one) into the staging buffer as fit, then post
        // them as one batch
        let mut pending: Vec<(u64, usize, usize)> = Vec::new(); // (arena off, staging off, len)
        let mut used = 0usize;
        let flush = |me: &mut Self, pending: &mut Vec<(u64, usize, usize)>| -> Result<()> {
            if pending.is_empty() {
                return Ok(());
            }
            let mut jobs: Vec<Job<'_>> = Vec::with_capacity(pending.len() + 1);
            for (off, soff, len) in pending.iter() {
                let src = me.staging.slice(*soff, *len).ok_or_else(|| {
                    Error::Refused("vision: staging slice out of range".to_owned())
                })?;
                jobs.push(Job::MemcpyH2d {
                    dst: &me.arena,
                    dst_offset: me.base + *off,
                    src,
                    stream,
                });
            }
            jobs.push(Job::SyncStream { stream });
            hip.run_batch(&jobs)?;
            pending.clear();
            Ok(())
        };
        for (name, bytes) in &inputs {
            let slot = self.arena_plan.slot(name).cloned().ok_or_else(|| {
                Error::Refused(format!("vision: no arena slot for the host input `{name}`"))
            })?;
            if bytes.len() as u64 > slot.bytes {
                return Err(Error::Refused(format!(
                    "vision: {} B of `{name}` into a {} B slot",
                    bytes.len(),
                    slot.bytes
                )));
            }
            let mut done = 0usize;
            while done < bytes.len() {
                if used == cap {
                    flush(self, &mut pending)?;
                    used = 0;
                }
                let n = (bytes.len() - done).min(cap - used);
                {
                    let mut dst = self.staging.slice_mut(used, n).ok_or_else(|| {
                        Error::Refused("vision: staging slice out of range".to_owned())
                    })?;
                    dst.as_mut_slice().copy_from_slice(&bytes[done..done + n]);
                }
                pending.push((slot.offset + done as u64, used, n));
                used += n;
                done += n;
            }
        }
        flush(self, &mut pending)?;
        Ok(())
    }

    fn out_of(&self, plan: &TowerPlan, launches: u32) -> Result<VisionOut> {
        let slot = self
            .arena_plan
            .slot(OUT_BUFFER)
            .ok_or_else(|| Error::Refused("vision: no `out` slot".to_owned()))?;
        Ok(VisionOut {
            addr: self
                .arena
                .sub_addr(self.base + slot.offset)
                .ok_or_else(|| Error::Refused("vision: `out` is past the arena".to_owned()))?,
            offset: self.base + slot.offset,
            tokens: plan.tokens,
            bytes: u64::from(plan.tokens) * u64::from(self.rules.model.out_hidden) * 2,
            launches,
        })
    }

    /// Run the tower EAGERLY over an already-uploaded `plan`: every launch through `launcher`,
    /// the count asserted against [`VisionRules::n_launches`].
    ///
    /// # Errors
    /// A binding or launch failure, or a launch count off the declaration.
    pub fn launch_eager(
        &mut self,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        stream: &Stream,
        plan: &TowerPlan,
    ) -> Result<VisionOut> {
        let binds = self.bindings(plan)?;
        let reads = match &self.streamed {
            Some(st) => Some(st.stage_reads(plan, &self.arena_plan)?),
            None => None,
        };
        let mut ring = RingState::default();
        let mut n = 0u32;
        for (j, (ld, b)) in plan.launches.iter().zip(&binds).enumerate() {
            if let (Some(st), Some(reads)) = (&self.streamed, &reads) {
                st.stage_in(ctx.hip, stream, &self.arena, &reads[j], &mut ring)?;
            }
            n += launcher.launch(ctx, ld, b, stream)?.launches;
        }
        if n != self.rules.n_launches() {
            return Err(Error::Refused(format!(
                "vision: issued {n} launches, the rules declare {}",
                self.rules.n_launches()
            )));
        }
        let out = self.out_of(plan, n)?;
        if let Some(st) = &mut self.streamed {
            if ring.next != st.stages.len() {
                return Err(Error::Refused(format!(
                    "vision: the streamed run read {} of the tower's {} weight stages",
                    ring.next,
                    st.stages.len()
                )));
            }
            // The rows leave the prefill arena before the step loop reuses it.
            let dst = st
                .host_out
                .slice_mut(0, out.bytes as usize)
                .ok_or_else(|| {
                    Error::Refused("vision: the host out buffer is smaller than the run".to_owned())
                })?;
            ctx.hip.run_batch(&[
                Job::MemcpyD2h {
                    src: &self.arena,
                    src_offset: out.offset,
                    dst,
                    stream,
                },
                Job::SyncStream { stream },
            ])?;
            st.runs += 1;
            st.bytes_streamed += ring.bytes;
        }
        Ok(out)
    }

    /// [`VisionExecutor::launch_eager`] with a `hipEvent` pair around every launch: per-launch
    /// `(op, kernel, ms)`. It serialises the stream at launch granularity, so the sum is the
    /// kernels' time without the host's launch cost -- a profile, not the tower's wall.
    ///
    /// # Errors
    /// As [`VisionExecutor::launch_eager`], and the events.
    pub fn launch_profiled(
        &mut self,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        stream: &Stream,
        plan: &TowerPlan,
    ) -> Result<Vec<(String, String, f64)>> {
        let binds = self.bindings(plan)?;
        let mut out = Vec::with_capacity(plan.launches.len());
        let a = ctx.hip.create_event(saber_hip::EventFlags::Default)?;
        let b = ctx.hip.create_event(saber_hip::EventFlags::Default)?;
        for (ld, bd) in plan.launches.iter().zip(&binds) {
            ctx.hip.record_event(&a, stream)?;
            launcher.launch(ctx, ld, bd, stream)?;
            ctx.hip.record_event(&b, stream)?;
            ctx.hip.sync_event(&b)?;
            out.push((
                ld.op.clone(),
                ld.kernel.clone(),
                f64::from(ctx.hip.elapsed_ms(&a, &b)?),
            ));
        }
        Ok(out)
    }

    /// Replay the graph captured for `plan`'s grid set, capturing it on first use. The capture
    /// asserts its node count against the declaration (rule 7).
    ///
    /// # Errors
    /// The capture or the replay.
    pub fn launch_graph(
        &mut self,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        stream: &Stream,
        plan: &TowerPlan,
    ) -> Result<VisionOut> {
        if self.streamed.is_some() {
            return Err(Error::Refused(
                "vision: the streamed tower runs eager (its weight copies sit between launches)"
                    .to_owned(),
            ));
        }
        if !self.graphs.contains_key(&plan.grids) {
            let binds = self.bindings(plan)?;
            let proof = || {
                GraphSafe::from_flags(true, false)
                    .ok_or_else(|| Error::Refused("vision: graph proof".to_owned()))
            };
            let mut ops = Vec::with_capacity(plan.launches.len());
            for (ld, b) in plan.launches.iter().zip(&binds) {
                ops.push(CapturableOp::new(ld, b, proof()?));
            }
            let label = format!("vision.{}p", plan.patches);
            let bucket =
                CapturedBucket::capture(ctx, launcher, &ops, stream, label, &self.counters)?;
            self.graphs.insert(plan.grids.clone(), bucket);
        }
        let g = &self.graphs[&plan.grids];
        g.replay(ctx, stream)?;
        let n = g.node_count() as u32;
        self.out_of(plan, n)
    }

    /// Plan, upload and run one image set; eager or from its captured graph.
    ///
    /// # Errors
    /// Any of [`VisionExecutor::plan`], [`VisionExecutor::upload`] and the launch.
    pub fn run(
        &mut self,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        stream: &Stream,
        grids: &[Grid],
        pixels: &[f32],
        graph: bool,
    ) -> Result<VisionOut> {
        let plan = self.plan(grids)?;
        self.upload(ctx.hip, stream, &plan, pixels)?;
        if graph {
            self.launch_graph(ctx, launcher, stream, &plan)
        } else {
            self.launch_eager(ctx, launcher, stream, &plan)
        }
    }

    /// Zero the whole arena (a gate's way to prove a run wrote what it is compared on).
    ///
    /// # Errors
    /// The memset.
    pub fn zero(&self, hip: &HipThread, stream: &Stream) -> Result<()> {
        if self.streamed.is_some() {
            return Err(Error::Refused(
                "vision: zero() is a gate of the resident tower".to_owned(),
            ));
        }
        hip.memset(self.arena.as_ref(), 0, stream)?;
        hip.sync_stream(stream)?;
        Ok(())
    }

    /// Read `bytes` of the arena at `offset` back to the host (blocking) -- the gate's view of
    /// the output rows or an intermediate.
    ///
    /// # Errors
    /// The copy.
    pub fn download(
        &self,
        hip: &HipThread,
        stream: &Stream,
        offset: u64,
        out: &mut [u8],
    ) -> Result<()> {
        hip.memcpy_d2h_at_blocking(out, self.arena.as_ref(), offset, stream)?;
        Ok(())
    }
}

/// Where a run's output rows are ([`VisionExecutor::out_source`]).
#[derive(Debug)]
pub enum OutSource<'a> {
    /// The resident tower's `out` slot.
    Device {
        /// The allocation.
        arena: &'a DevicePtr,
        /// The slot's offset in it.
        offset: u64,
        /// The slot's bytes.
        bytes: u64,
    },
    /// The streamed tower's pinned host copy (row 0 at offset 0).
    Host(&'a PinnedBuf),
}

/// One weight record of the streamed tower: where it sits in the pinned slab, and its bytes.
#[derive(Debug, Clone)]
pub struct SlabRecord {
    /// The record's name (`model.visual.*`).
    pub name: String,
    /// Byte offset in the slab.
    pub offset: usize,
    /// Bytes.
    pub len: usize,
}

/// The stage a record streams in with: one per `blocks.<i>`, the merger, and the rest (`pre`).
fn stage_key(name: &str) -> String {
    let rest = name.strip_prefix(VISION_PREFIX).unwrap_or(name);
    if let Some(b) = rest.strip_prefix("blocks.") {
        let i = b.split('.').next().unwrap_or("");
        return format!("blocks.{i}");
    }
    if rest.starts_with("merger") {
        return "merger".to_owned();
    }
    "pre".to_owned()
}

/// The copies of one stage: `(slab offset, len, arena offset)`.
type StageCopies = Vec<(usize, usize, u64)>;

/// P27-11: the STREAMED tower's weights -- the 333 records in a pinned host slab, copied per
/// image, stage by stage, into a two-slot ring at the front of the lent arena (stage `s` into
/// slot `s % 2`, on the step's stream, right before the first launch that reads it: the
/// side-stream H2D law keeps the copies on the compute stream).
pub struct Streamed {
    slab: PinnedBuf,
    /// Per stage, in launch order.
    stages: Vec<StageCopies>,
    /// Record -> stage index.
    stage_of: BTreeMap<String, usize>,
    host_out: PinnedBuf,
    /// The two ring slots' bytes.
    pub ring_bytes: [u64; 2],
    /// Tower runs streamed so far.
    pub runs: u64,
    /// Weight bytes streamed so far.
    pub bytes_streamed: u64,
}

impl std::fmt::Debug for Streamed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Streamed({} stages, ring {:?} B, slab {} B, {} runs)",
            self.stages.len(),
            self.ring_bytes,
            self.slab.len(),
            self.runs
        )
    }
}

/// The ring's state during one run.
#[derive(Debug, Default)]
struct RingState {
    /// Stages copied in so far (they arrive in order).
    next: usize,
    /// Bytes copied this run.
    bytes: u64,
}

impl Streamed {
    /// Pinned host bytes of the out rows' copy.
    #[must_use]
    pub fn host_out_bytes(&self) -> usize {
        self.host_out.len()
    }

    /// Weight bytes in the slab.
    #[must_use]
    pub fn slab_bytes(&self) -> usize {
        self.slab.len()
    }

    /// Stages (one copy batch each per image).
    #[must_use]
    pub fn stages(&self) -> usize {
        self.stages.len()
    }

    /// The stages each launch of `plan` reads, launch by launch.
    fn stage_reads(&self, plan: &TowerPlan, arena: &TowerArena) -> Result<Vec<Vec<usize>>> {
        let names: BTreeSet<String> = self.stage_of.keys().cloned().collect();
        let resolved = bind_names(plan, arena, &names)?;
        Ok(resolved
            .into_iter()
            .map(|per| {
                let mut v: Vec<usize> = per
                    .iter()
                    .filter_map(|(_, src)| match src {
                        VisionSource::Record(r) => self.stage_of.get(r).copied(),
                        VisionSource::Arena(_) => None,
                    })
                    .collect();
                v.sort_unstable();
                v.dedup();
                v
            })
            .collect())
    }

    /// Copy in every stage up to the newest one `need` names, in order, then refuse the launch if
    /// it reads a stage whose slot a newer stage has overwritten (the ring law: a launch reads
    /// only the newest two stages).
    fn stage_in(
        &self,
        hip: &HipThread,
        stream: &Stream,
        arena: &DevicePtr,
        need: &[usize],
        ring: &mut RingState,
    ) -> Result<()> {
        let Some(&top) = need.last() else {
            return Ok(());
        };
        while ring.next <= top {
            let st = &self.stages[ring.next];
            let mut jobs: Vec<Job<'_>> = Vec::with_capacity(st.len());
            for &(off, len, dst) in st {
                let src = self
                    .slab
                    .slice(off, len)
                    .ok_or_else(|| Error::Refused("vision: slab slice out of range".to_owned()))?;
                jobs.push(Job::MemcpyH2d {
                    dst: arena,
                    dst_offset: dst,
                    src,
                    stream,
                });
                ring.bytes += len as u64;
            }
            hip.run_batch(&jobs)?;
            ring.next += 1;
        }
        if let Some(&s) = need.iter().find(|&&s| s + 2 < ring.next) {
            return Err(Error::Refused(format!(
                "vision: a launch reads weight stage {s} after stage {} overwrote its ring slot",
                s + 2
            )));
        }
        Ok(())
    }
}

/// [`VisionExecutor::streamed_layout`]'s answer: the arena, the ring slots, the arena's base past
/// the ring, the stages' copies and each record's (stage, lent-arena offset).
type StreamedLayout = (
    TowerArena,
    [u64; 2],
    u64,
    Vec<StageCopies>,
    BTreeMap<String, (usize, u64)>,
);

impl VisionExecutor {
    /// The STREAMED tower (P27-11): no VRAM of its own. `lent` is an allocation idle whenever the
    /// tower runs (the engine's shared prefill arena); its front holds the two-slot weight ring,
    /// the tower arena for `max_patches` follows. `slab` holds `records` (every `model.visual.*`
    /// record the tower reads). The output rows are copied to a pinned host buffer at the end of
    /// every run, so nothing of the tower outlives the run in VRAM.
    ///
    /// # Errors
    /// The layout, a record the plan reads and the slab does not hold, a lent allocation too
    /// small for ring + arena (named, with both numbers), or the pinned allocations.
    pub fn new_streamed(
        hip: &HipThread,
        rules: VisionRules,
        max_patches: u32,
        lent: Rc<DevicePtr>,
        slab: PinnedBuf,
        records: &[SlabRecord],
        staging_bytes: usize,
    ) -> Result<Self> {
        let (arena_plan, ring_bytes, base, stages, placed) =
            Self::streamed_layout(&rules, max_patches, records)?;
        if base + arena_plan.bytes > lent.len() {
            return Err(Error::Refused(format!(
                "vision: the streamed tower needs {} B (ring {base} + arena {} for {max_patches} \
                 patches) and the lent arena is {} B",
                base + arena_plan.bytes,
                arena_plan.bytes,
                lent.len()
            )));
        }
        if let Some(r) = records.iter().find(|r| r.offset + r.len > slab.len()) {
            return Err(Error::Refused(format!(
                "vision: record `{}` runs past the slab",
                r.name
            )));
        }
        let mut weights = BTreeMap::new();
        for (name, (_, dst)) in &placed {
            let a = lent.sub_addr(*dst).ok_or_else(|| {
                Error::Refused(format!("vision: `{name}` is past the lent arena"))
            })?;
            weights.insert(name.clone(), a);
        }
        let stage_of = placed.iter().map(|(n, (s, _))| (n.clone(), *s)).collect();
        let out = arena_plan
            .slot(OUT_BUFFER)
            .ok_or_else(|| Error::Refused("vision: no `out` slot".to_owned()))?
            .bytes;
        let host_out = hip.host_malloc(out.max(1))?;
        let staging = hip.host_malloc(staging_bytes.max(1 << 20) as u64)?;
        Ok(VisionExecutor {
            rules,
            arena_plan,
            arena: lent,
            base,
            streamed: Some(Streamed {
                slab,
                stages,
                stage_of,
                host_out,
                ring_bytes,
                runs: 0,
                bytes_streamed: 0,
            }),
            staging,
            weights,
            graphs: BTreeMap::new(),
            counters: Arc::new(CaptureCounters::default()),
        })
    }

    /// The bytes the streamed form lays over the lent arena for `max_patches` (ring + arena), on
    /// the host -- the engine picks the largest image the prefill arena holds with it.
    ///
    /// # Errors
    /// As [`VisionExecutor::new_streamed`]'s layout.
    pub fn streamed_bytes(
        rules: &VisionRules,
        max_patches: u32,
        records: &[SlabRecord],
    ) -> Result<u64> {
        let (a, _, base, _, _) = Self::streamed_layout(rules, max_patches, records)?;
        Ok(base + a.bytes)
    }

    fn streamed_layout(
        rules: &VisionRules,
        max_patches: u32,
        records: &[SlabRecord],
    ) -> Result<StreamedLayout> {
        let arena_plan =
            tower_arena(rules, max_patches).map_err(|e| Error::Refused(e.to_string()))?;
        let probe = plan_tower(
            rules,
            &[Grid {
                t: 1,
                h: rules.model.spatial_merge,
                w: rules.model.spatial_merge,
            }],
        )
        .map_err(|e| Error::Refused(e.to_string()))?;
        let names: BTreeSet<String> = records.iter().map(|r| r.name.clone()).collect();
        let resolved = bind_names(&probe, &arena_plan, &names)?;
        // The stage order is the order of each stage's first read.
        let mut order: Vec<String> = Vec::new();
        for per in &resolved {
            for (_, src) in per {
                if let VisionSource::Record(r) = src {
                    let k = stage_key(r);
                    if !order.contains(&k) {
                        order.push(k);
                    }
                }
            }
        }
        let mut stages: Vec<StageCopies> = vec![Vec::new(); order.len()];
        let mut fill = vec![0u64; order.len()];
        let mut placed: BTreeMap<String, (usize, u64)> = BTreeMap::new();
        for r in records {
            let k = stage_key(&r.name);
            let s = order.iter().position(|x| *x == k).ok_or_else(|| {
                Error::Refused(format!("vision: record `{}` is read by no launch", r.name))
            })?;
            let off = fill[s];
            stages[s].push((r.offset, r.len, off));
            placed.insert(r.name.clone(), (s, off));
            fill[s] = (off + r.len as u64).div_ceil(256) * 256;
        }
        let mut ring = [0u64; 2];
        for (s, f) in fill.iter().enumerate() {
            ring[s % 2] = ring[s % 2].max(*f);
        }
        let slot1 = ring[0].div_ceil(4096) * 4096;
        let base = (slot1 + ring[1]).div_ceil(4096) * 4096;
        for (s, st) in stages.iter_mut().enumerate() {
            if s % 2 == 1 {
                for c in st.iter_mut() {
                    c.2 += slot1;
                }
            }
        }
        for v in placed.values_mut() {
            if v.0 % 2 == 1 {
                v.1 += slot1;
            }
        }
        Ok((arena_plan, ring, base, stages, placed))
    }
}

#[cfg(test)]
mod streamed_tests {
    use super::*;

    /// The ring law on the real tower plan: every record lands in a stage, the stages follow the
    /// launches, each block is one stage, and the ring is two block-or-merger slots, not the
    /// tower. NEGATIVE CONTROL: a slab missing a record the plan reads is refused by name.
    #[test]
    fn the_streamed_layout_stages_every_record_in_launch_order() {
        let r = VisionRules::prod().unwrap();
        let names: Vec<String> = r.records().into_iter().collect();
        let mut off = 0usize;
        let recs: Vec<SlabRecord> = names
            .iter()
            .map(|n| {
                let len = 1 << 20;
                let x = SlabRecord {
                    name: n.clone(),
                    offset: off,
                    len,
                };
                off += len;
                x
            })
            .collect();
        let (a, ring, base, stages, placed) =
            VisionExecutor::streamed_layout(&r, 16_384, &recs).unwrap();
        assert_eq!(placed.len(), recs.len());
        assert_eq!(
            stages.len(),
            r.model.depth as usize + 2,
            "pre + blocks + merger"
        );
        assert!(ring[0] > 0 && ring[1] > 0);
        assert!(base >= ring[0] + ring[1] && a.bytes > 0);
        let b0 = placed[&format!("{VISION_PREFIX}blocks.0.attn.qkv.weight")];
        let b1 = placed[&format!("{VISION_PREFIX}blocks.1.attn.qkv.weight")];
        assert_eq!(b1.0, b0.0 + 1, "blocks stream in order");
        assert_ne!(b0.1, b1.1, "consecutive stages take the two slots");
        let short: Vec<SlabRecord> = recs[1..].to_vec();
        assert!(VisionExecutor::streamed_layout(&r, 16_384, &short).is_err());
    }
}

impl std::fmt::Debug for VisionExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "VisionExecutor(arena {} B for {} patches, {} records, {} graphs)",
            self.arena.len(),
            self.arena_plan.max_patches,
            self.weights.len(),
            self.graphs.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_records(r: &VisionRules) -> BTreeSet<String> {
        r.records().into_iter().collect()
    }

    #[test]
    fn the_plan_binds_every_record_and_every_arena_buffer() {
        let r = VisionRules::prod().unwrap();
        let arena = tower_arena(&r, 65_536).unwrap();
        let plan = plan_tower(&r, &[Grid { t: 1, h: 24, w: 32 }]).unwrap();
        let b = bind_names(&plan, &arena, &all_records(&r)).unwrap();
        assert_eq!(b.len(), 222);
        let recs: BTreeSet<&str> = b
            .iter()
            .flatten()
            .filter_map(|(_, s)| match s {
                VisionSource::Record(n) => Some(n.as_str()),
                VisionSource::Arena(_) => None,
            })
            .collect();
        assert_eq!(recs.len(), 333);
        // the tiled attention is what the engine binds
        assert_eq!(plan.launches[6].kernel, "vis_attn_tiled_bf16_hd72");
        assert_eq!(plan.launches[6].grid, [12, 16, 1]);
    }

    #[test]
    fn a_missing_or_an_unread_record_is_refused_by_name() {
        let r = VisionRules::prod().unwrap();
        let arena = tower_arena(&r, 4096).unwrap();
        let plan = plan_tower(&r, &[Grid { t: 1, h: 24, w: 32 }]).unwrap();
        let mut recs = all_records(&r);
        recs.remove("model.visual.blocks.13.attn.proj.bias");
        let e = bind_names(&plan, &arena, &recs).unwrap_err().to_string();
        assert!(e.contains("blocks.13.attn.proj.bias"), "{e}");
        let mut recs = all_records(&r);
        recs.insert("model.visual.deepstack_merger_list.0.norm.weight".to_owned());
        let e = bind_names(&plan, &arena, &recs).unwrap_err().to_string();
        assert!(e.contains("deepstack_merger_list"), "{e}");
    }

    #[test]
    fn host_inputs_are_sized_by_the_plan() {
        let r = VisionRules::prod().unwrap();
        let plan = plan_tower(&r, &[Grid { t: 1, h: 24, w: 32 }]).unwrap();
        let px = vec![0.25f32; 768 * 1536];
        let ins = host_inputs(&r, &plan, &px).unwrap();
        let by: BTreeMap<&str, usize> = ins.iter().map(|(n, b)| (*n, b.len())).collect();
        assert_eq!(by["pixels"], 768 * 1536 * 4);
        assert_eq!(by["pos_idx"], 4 * 768 * 4);
        assert_eq!(by["rope_cos"], 768 * 36 * 2);
        assert_eq!(by["cu_seqlens"], 8);
        let e = host_inputs(&r, &plan, &px[1..]).unwrap_err().to_string();
        assert!(e.contains("pixel values"), "{e}");
    }
}
