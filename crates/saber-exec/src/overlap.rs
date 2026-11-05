//! **The prefill all-reduce, overlapped with the GEMM that produces it** (workstream H36).
//!
//! # What can overlap what
//!
//! A decoder layer at TP=2 is a strict chain: `o_proj` (or the GDN `out_proj`) → reduce →
//! residual add + norm + quant → `gate_up` → `silu` → `down_proj` → reduce → the next layer's add +
//! norm + quant. Every reduce's output is the NEXT op's input, and layer `L + 1`'s first GEMM reads
//! `norm(residual + reduce(L))`, so **no reduce can overlap another layer's work at whole-chunk
//! granularity**: there is no independent op to put beside it. What IS independent is the other
//! ROWS. A reduce is elementwise over rows (`saber::ar_bands` already cuts a message into row bands
//! for exactly this reason), and a GEMM row does not read another row. So the producing GEMM is cut
//! into row SLICES, and slice `s`'s reduce runs on a second stream while slice `s + 1`'s GEMM runs
//! on the first:
//!
//! ```text
//!   main  : G0 ─ G1 ─ G2 ─ G3 ───────────────┐ wait(done) ─ add+norm ─ ...
//!             ╲    ╲    ╲    ╲               │
//!   side  :    A0 ─ A1 ─ A2 ─ A3 ─ record(done)
//! ```
//!
//! The consumer (the residual add + norm, the first op after the reduce) waits for the LAST slice's
//! reduce, because it reads every row; so the exposed cost of a pair is one slice's GEMM plus one
//! slice's reduce, not the whole reduce. The attention and the GDN scan are causal over rows and
//! are never sliced — they are what makes the chunk the unit of work in the first place.
//!
//! # Why the arithmetic cannot move
//!
//! * **The GEMM slice is the SAME descriptor** — the same kernel symbol (TN, LBK, WPERM, the fold)
//!   the full-M plan resolved — with `m` and the row-tile grid cut to the slice and `a`, `as`, `c`
//!   advanced by the slice's first row. Slices start on a row-tile boundary (`rows / grid.y`), so
//!   every workgroup computes exactly the tile it computed unsliced, over the same K order. The
//!   fragment-tiled A layout is 16-row-tile-major (`mt * K * 16`), so `r0 * K` is its offset too.
//! * **The reduce slice is the same FAMILY** — resolved through the same rules at the slice's
//!   size, and refused if it lands in another one (the wht6 kernel above `quant_min_bytes`, the exact
//!   one below; two families in one message would be two roundings). Both families are row-local:
//!   the wht groups are 64 elements and a row is 5120, so a group never spans a band boundary.
//! * Only the reduces' PLACEMENT moves. They are still issued in plan order on ONE stream per rank,
//!   so the two ranks' protocol sequence (`ar_seq`, the scratch double-buffer) is what it was.
//!
//! # What it costs
//!
//! Events, not spin-waits: the side stream waits on an event the main stream records after each
//! GEMM slice, and the main stream waits on the side's `done` before the consumer — one
//! `run_batch` of two jobs per slice. Nothing is allocated per call on the device: the events and
//! the side stream are created at boot, and the reduce slices resolve to the IPC regions the full
//! reduce already prepared (same `(plugin, op, shape)` key; a slice declaring MORE bytes than the
//! full message is refused by name). The memory fact that decode's overlap LOST (spin-wait
//! contention, −3.5 to −5.6 %) is why this is a PREFILL-only path gated by the chunk's rows.
//!
//! # Measured (H36, 2026-09-23): it LOSES here too
//!
//! Bit-exact on both ranks at every slice, and slower at every slice: the body +4.6 / +1.9 % at a
//! 2048-row chunk (512 / 1024-row slices), +4.2 / +3.7 / +1.7 % at 8192 (512 / 1024 / 2048). The
//! one-shot reduce's blocks spin on CUs its producer wants and a slice is a partial wave of the
//! tile grid. Off by default; kept as the measured control (LEDGER H36).
//!
//! # The split form (P27-1d)
//!
//! `--ar-form split --ar-push-blocks N` replaces each slice's one-shot reduce with P27-1b/1c's
//! SPLIT pair on the side stream: the PUSH (N physical blocks walking the one-shot's virtual
//! blocks -- few enough to co-reside with the GEMM slice on the main stream), a COMMAND-PROCESSOR
//! wait (`hipStreamWaitValue32 >= v` on this rank's `ar_flags[511]`, which the peer's last push
//! block raises to `v`), then the REDUCE at the one-shot's grid. No CU spins on the peer; the
//! arithmetic is the one-shot's (the halves are its code sliced; P27-1c's byte law). `v` is one
//! counter per rank ([`OverlapCounters::next_split_flag`]), advanced once per split call in issue
//! order -- both ranks issue the same reduce sequence, so their values agree, and every protocol
//! region (all reduce routes share one per family, LEDGER T10) only ever sees it grow. A prefill
//! chunk runs eagerly (never captured), so no graph bakes a value. Decode keeps the one-shot.

use std::sync::atomic::{AtomicU64, Ordering};

use saber_hip::{Event, Job, Stream};
use saber_kernels::rules::ArgValue;
use saber_kernels::saber::{ar_wht_split, AR_SPLIT_FLAG_WORD};
use saber_kernels::LaunchDescriptor;

use crate::block_capture::BoundStep;
use crate::error::{Error, Result};
use crate::launch::{ArgBindings, LaunchCtx, Launcher};

/// The law: which prefill chunks overlap their reduces, and how finely the producer is cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OverlapLaw {
    /// Rows per slice. The bucket must be a whole number of slices and a slice a whole number of the
    /// producer's row tiles; a bucket that is not is planned WITHOUT the overlap, by name.
    pub slice_rows: u32,
    /// The smallest bucket that overlaps. Below it a chunk is one slice or two, and the reduce is
    /// small enough that the events cost more than they hide.
    pub min_rows: u32,
}

impl OverlapLaw {
    /// Whether a prefill bucket of `rows` takes the overlapped path at all.
    #[must_use]
    pub fn applies(&self, rows: u32) -> bool {
        self.slice_rows > 0
            && rows >= self.min_rows
            && rows > self.slice_rows
            && rows % self.slice_rows == 0
    }

    /// Slices a bucket of `rows` is cut into (1 = not overlapped).
    #[must_use]
    pub fn slices(&self, rows: u32) -> u32 {
        if self.applies(rows) {
            rows / self.slice_rows
        } else {
            1
        }
    }
}

/// Counted on the path (structural rule 5): every reduce an overlapped step issues is one or the
/// other, and a plan that should be all `overlapped` and reads `sequential` is visible.
#[derive(Debug, Default)]
pub struct OverlapCounters {
    /// Producer + reduce pairs issued sliced across the two streams.
    pub overlapped: AtomicU64,
    /// Reduces an overlapped step issued in place on the main stream (the embedding's, whose
    /// producer is not a GEMM).
    pub sequential: AtomicU64,
    /// Slices issued (GEMM + reduce each).
    pub slices: AtomicU64,
    /// P27-1d: split calls issued (push + CP wait + reduce). Also the source of the split flag's
    /// value: this rank's `n`-th split call raises the peer's flag to `n` (1-based).
    pub split_calls: AtomicU64,
}

impl OverlapCounters {
    /// One line for a census. The one-shot form's line is H36's, unchanged; the split form adds
    /// its call count.
    #[must_use]
    pub fn line(&self) -> String {
        let split = self.split_calls.load(Ordering::Relaxed);
        format!(
            "ar_overlap(pairs={} sequential={} slices={}{})",
            self.overlapped.load(Ordering::Relaxed),
            self.sequential.load(Ordering::Relaxed),
            self.slices.load(Ordering::Relaxed),
            if split > 0 {
                format!(" split={split}")
            } else {
                String::new()
            }
        )
    }

    /// The next split call's flag value: 1, 2, 3, ... in issue order on this rank (THE FLAG LAW:
    /// monotone per rank, the same sequence on both ranks because both issue the same reduces).
    ///
    /// # Errors
    /// Past `i32::MAX` calls (the kernarg is a 32-bit int and the wait an unsigned compare; a
    /// wrapped value would pass a stale flag). ~16 M 2k chunks; refused, never wrapped.
    pub fn next_split_flag(&self) -> Result<u32> {
        let n = self.split_calls.fetch_add(1, Ordering::Relaxed) + 1;
        u32::try_from(n)
            .ok()
            .filter(|v| i32::try_from(*v).is_ok())
            .ok_or_else(|| {
                Error::Refused(format!(
                    "the split all-reduce's flag would pass i32::MAX at call {n}; restart the \
                     engine (both ranks re-zero the flag region at boot)"
                ))
            })
    }
}

/// P27-1d: the split form of the slice reduce -- the push and the reduce derived from the ONE-SHOT
/// slice descriptor the rules resolved (`saber::ar_wht_split`), kept beside it.
#[derive(Clone, Debug)]
pub struct SplitForm {
    /// The one-shot symbol it was derived from; [`Schedule::with_split`] refuses any other.
    pub source: String,
    /// The fewer-blocks push (`flag_val` 0 here; set per call).
    pub push: LaunchDescriptor,
    /// The reduce half at the one-shot's grid.
    pub reduce: LaunchDescriptor,
}

impl SplitForm {
    /// Derive the split pair from the one-band slice reduce.
    ///
    /// # Errors
    /// A slice that is not one band, or `ar_wht_split`'s refusals (not the wht6 bf16 one-shot,
    /// `push_blocks` outside `1..=nbv`).
    pub fn derive(slice: &[LaunchDescriptor], push_blocks: u32) -> Result<Self> {
        let [one] = slice else {
            return Err(Error::Refused(format!(
                "--ar-form split: the slice reduce resolved to {} bands; a slice must be one",
                slice.len()
            )));
        };
        let (push, reduce) =
            ar_wht_split(one, push_blocks).map_err(|e| Error::Refused(e.to_string()))?;
        Ok(SplitForm {
            source: one.kernel.clone(),
            push,
            reduce,
        })
    }
}

/// What one slice of an overlapped pair issues, in order (structural rule 7: the sequence is data,
/// and the host test pins it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SliceOp {
    /// The GEMM slice, on the main stream.
    Gemm,
    /// Record `produced[j]` on the main stream; the side stream waits on it.
    Handoff,
    /// The one-shot reduce slice, on the side stream (H36's form).
    OneShot,
    /// The split push (N blocks), on the side stream, raising the PEER's flag to `v`.
    Push,
    /// `hipStreamWaitValue32(side, my ar_flags[511] >= v)`: the command processor waits, no CU.
    CpWait,
    /// The split reduce, on the side stream.
    Reduce,
}

/// The per-slice sequence of a form. The one-shot's is H36's, unchanged.
#[must_use]
pub fn slice_ops(split: bool) -> &'static [SliceOp] {
    if split {
        &[
            SliceOp::Gemm,
            SliceOp::Handoff,
            SliceOp::Push,
            SliceOp::CpWait,
            SliceOp::Reduce,
        ]
    } else {
        &[SliceOp::Gemm, SliceOp::Handoff, SliceOp::OneShot]
    }
}

/// The GEMM families whose rows are independent and whose arguments this module knows how to
/// advance. Anything else in front of a reduce is left in place.
const SLICEABLE: &[&str] = &["gemm_atiled", "gemm_folded"];

/// Is this step a reduce (any family, any band count)?
#[must_use]
pub fn is_reduce(step: &BoundStep<'_>) -> bool {
    step.descs
        .first()
        .is_some_and(|d| d.op.starts_with("allreduce"))
}

fn slot_i64(desc: &LaunchDescriptor, name: &str) -> Result<i64> {
    match desc.arg(name).map(|s| &s.value) {
        Some(ArgValue::I32 { ival } | ArgValue::I64 { ival }) => Ok(*ival),
        _ => Err(Error::Refused(format!(
            "{}: no scalar `{name}` to slice on",
            desc.kernel
        ))),
    }
}

fn set_slot(desc: &mut LaunchDescriptor, name: &str, v: i64) -> Result<()> {
    let slot = desc
        .args
        .iter_mut()
        .find(|s| s.name == name)
        .ok_or_else(|| Error::Refused(format!("{}: no slot `{name}`", desc.kernel)))?;
    match &mut slot.value {
        ArgValue::I32 { ival } | ArgValue::I64 { ival } => {
            *ival = v;
            Ok(())
        }
        _ => Err(Error::Refused(format!(
            "{}: `{name}` is not a scalar",
            desc.kernel
        ))),
    }
}

/// One slice of a producer GEMM: the full descriptor with `m` and the row-tile grid cut.
///
/// # Errors
/// A descriptor this module cannot slice exactly: not a sliceable family, partitions (a merged
/// projection's `pb1 < n`), a bucket that is not a whole number of row tiles, or a slice that
/// does not start and end on one.
pub fn gemm_slice(
    full: &LaunchDescriptor,
    rows: u32,
    first_row: u32,
    slice_rows: u32,
) -> Result<LaunchDescriptor> {
    if !SLICEABLE.contains(&full.op.as_str()) {
        return Err(Error::Refused(format!(
            "{}: `{}` is not a row-independent GEMM family this module slices",
            full.kernel, full.op
        )));
    }
    let m = slot_i64(full, "m")?;
    let n = slot_i64(full, "n")?;
    if m != i64::from(rows) {
        return Err(Error::Refused(format!(
            "{}: the descriptor's m is {m} and the bucket is {rows}",
            full.kernel
        )));
    }
    if slot_i64(full, "pb1")? < n || slot_i64(full, "pb2")? < n {
        return Err(Error::Refused(format!(
            "{}: a partitioned GEMM (a merged projection) is not a reduce's producer",
            full.kernel
        )));
    }
    let tiles = full.grid[1];
    if tiles == 0 || rows % tiles != 0 {
        return Err(Error::Refused(format!(
            "{}: {rows} rows over {tiles} row tile(s) is not a whole tile each",
            full.kernel
        )));
    }
    let tile_rows = rows / tiles;
    if first_row % tile_rows != 0 || slice_rows % tile_rows != 0 || first_row + slice_rows > rows {
        return Err(Error::Refused(format!(
            "{}: a slice of {slice_rows} rows at {first_row} is not on the {tile_rows}-row tile \
             grid of a {rows}-row GEMM",
            full.kernel
        )));
    }
    let mut d = full.clone();
    set_slot(&mut d, "m", i64::from(slice_rows))?;
    d.grid[1] = slice_rows / tile_rows;
    Ok(d)
}

/// The GEMM slice's bindings: `a`, `as` and `c` advanced to `first_row`, everything else shared.
///
/// # Errors
/// A binding missing, or an address that would overflow.
pub fn gemm_slice_args(
    full: &LaunchDescriptor,
    args: &ArgBindings,
    first_row: u32,
) -> Result<ArgBindings> {
    let k = u64::try_from(slot_i64(full, "k")?).unwrap_or(0);
    let n = u64::try_from(slot_i64(full, "n")?).unwrap_or(0);
    let r0 = u64::from(first_row);
    let mut out = args.clone();
    for (name, stride) in [("a", k), ("as", 4u64), ("c", n * 2)] {
        let base = args
            .get(name)
            .ok_or_else(|| Error::Refused(format!("{}: `{name}` is not bound", full.kernel)))?;
        let at = base.offset(r0 * stride).ok_or_else(|| {
            Error::Refused(format!(
                "{}: `{name}` + {} overflows",
                full.kernel,
                r0 * stride
            ))
        })?;
        out.bind(name, at);
    }
    Ok(out)
}

/// One producer + reduce pair of a step, cut into slices.
#[derive(Debug)]
pub struct Pair {
    /// Index of the producer GEMM in the bound step list; the reduce is the next one.
    pub producer: usize,
    /// Per slice: the GEMM and its bindings, then the reduce's bindings.
    pub slices: Vec<(LaunchDescriptor, ArgBindings, ArgBindings)>,
}

/// The overlapped schedule of one prefill step: which (producer, reduce) pairs are sliced.
#[derive(Debug)]
pub struct Schedule<'r> {
    /// The pairs, in step order.
    pub pairs: Vec<Pair>,
    /// Reduces left in place (their producer is not a sliceable GEMM).
    pub sequential: usize,
    /// The reduce descriptor every slice issues (one band of `slice_rows`).
    pub reduce: &'r [LaunchDescriptor],
    /// P27-1d: the split form, when `--ar-form split`; `None` = the one-shot (the control).
    pub split: Option<&'r SplitForm>,
}

impl<'r> Schedule<'r> {
    /// The same schedule with every slice's reduce issued in the split form.
    ///
    /// # Errors
    /// A split form derived from another one-shot than the one the slices resolved to.
    pub fn with_split(mut self, split: &'r SplitForm) -> Result<Self> {
        match self.reduce {
            [one] if one.kernel == split.source => {
                self.split = Some(split);
                Ok(self)
            }
            [] if self.pairs.is_empty() => Ok(self),
            _ => Err(Error::Refused(format!(
                "--ar-form split: the form was derived from `{}` and the slices resolve to {:?}",
                split.source,
                self.reduce.iter().map(|d| &d.kernel).collect::<Vec<_>>()
            ))),
        }
    }
}

/// Derive the schedule for `steps`, a prefill step of `rows` bucket rows, with `reduce` the
/// resolved reduce for ONE slice of `law.slice_rows` rows.
///
/// A reduce qualifies when its producer is the step right before it, is a sliceable GEMM whose `c`
/// IS the reduce's `in` (the planner aliases the reduce's edge onto the producer's buffer), and
/// the slice reduce is the same family and fits the prepared regions. Anything else runs in place
/// and is counted.
///
/// # Errors
/// A slice reduce of another family, or wider than the full one — a rules edit that would change
/// the arithmetic, refused rather than served.
pub fn derive<'r>(
    steps: &[BoundStep<'_>],
    rows: u32,
    law: OverlapLaw,
    reduce: &'r [LaunchDescriptor],
) -> Result<Schedule<'r>> {
    let mut sched = Schedule {
        pairs: Vec::new(),
        sequential: 0,
        reduce,
        split: None,
    };
    if !law.applies(rows) {
        return Ok(sched);
    }
    let [slice_ar] = reduce else {
        return Err(Error::Refused(format!(
            "the {}-row reduce slice resolved to {} bands; a slice must be one",
            law.slice_rows,
            reduce.len()
        )));
    };
    let s = law.slices(rows);
    let mut i = 0usize;
    while i < steps.len() {
        let step = &steps[i];
        if !is_reduce(step) {
            i += 1;
            continue;
        }
        let full_ar = &step.descs[0];
        let producer = i.checked_sub(1).map(|p| &steps[p]);
        let aliased = producer.is_some_and(|p| {
            p.descs.len() == 1
                && SLICEABLE.contains(&p.descs[0].op.as_str())
                && p.args.get("c").is_some()
                && p.args.get("c") == step.args.get("in")
        });
        if !aliased || full_ar.op.ends_with("_nq") {
            sched.sequential += 1;
            i += 1;
            continue;
        }
        if slice_ar.kernel != full_ar.kernel {
            return Err(Error::Refused(format!(
                "{}: the {}-row slice resolves to `{}` and the whole message to `{}` -- two \
                 families in one message is two roundings",
                step.name, law.slice_rows, slice_ar.kernel, full_ar.kernel
            )));
        }
        for w in &slice_ar.workspaces {
            let full_bytes = full_ar
                .workspaces
                .iter()
                .find(|f| f.name == w.name)
                .map(|f| f.bytes);
            if full_bytes.is_none_or(|b| w.bytes > b) {
                return Err(Error::Refused(format!(
                    "{}: the slice's `{}` wants {} B and the prepared region is {:?}",
                    step.name, w.name, w.bytes, full_bytes
                )));
            }
        }
        let p = producer.expect("aliased implies a producer");
        let gemm = &p.descs[0];
        let base = step
            .args
            .get("in")
            .expect("aliased implies the reduce's `in` is bound");
        let n = u64::try_from(slot_i64(gemm, "n")?).unwrap_or(0);
        let mut slices = Vec::with_capacity(s as usize);
        for j in 0..s {
            let r0 = j * law.slice_rows;
            let d = gemm_slice(gemm, rows, r0, law.slice_rows)?;
            let ga = gemm_slice_args(gemm, &p.args, r0)?;
            let at = base.offset(u64::from(r0) * n * 2).ok_or_else(|| {
                Error::Refused(format!("{}: the slice address overflows", step.name))
            })?;
            // The reduce's bindings keep the FULL reduce's shape: that is the IPC key, so the
            // slice lands on the regions the whole message prepared.
            let ra = ArgBindings::new(step.args.shape())
                .with("in", at)
                .with("out", at);
            slices.push((d, ga, ra));
        }
        sched.pairs.push(Pair {
            producer: i - 1,
            slices,
        });
        i += 1;
    }
    Ok(sched)
}

/// The two streams' fixed resources: the side stream and the events, created once at boot.
pub struct OverlapStreams {
    /// Where the reduces run.
    pub side: Stream,
    /// One per slice: recorded on the main stream after the slice's GEMM.
    pub produced: Vec<Event>,
    /// Recorded on the side stream after the last slice's reduce.
    pub done: Event,
}

/// Issue `steps` with `sched`'s pairs overlapped. Returns the dispatches issued.
///
/// # Errors
/// A launch or an event call failing; the schedule's producer indices not matching `steps`.
pub fn run(
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    main: &Stream,
    streams: &OverlapStreams,
    steps: &[BoundStep<'_>],
    sched: &Schedule<'_>,
    counters: &OverlapCounters,
) -> Result<u32> {
    let mut total = 0u32;
    let mut next = sched.pairs.iter().peekable();
    let mut i = 0usize;
    while i < steps.len() {
        if let Some(pair) = next.next_if(|p| p.producer == i) {
            if i + 1 >= steps.len() || !is_reduce(&steps[i + 1]) {
                return Err(Error::Refused(format!(
                    "the overlap schedule names step {i} as a producer and step {} is not its \
                     reduce",
                    i + 1
                )));
            }
            if pair.slices.len() > streams.produced.len() {
                return Err(Error::Refused(format!(
                    "{} slices and {} events",
                    pair.slices.len(),
                    streams.produced.len()
                )));
            }
            for (j, (gemm, ga, ra)) in pair.slices.iter().enumerate() {
                let mut flag = 0u32;
                for op in slice_ops(sched.split.is_some()) {
                    match (op, sched.split) {
                        (SliceOp::Gemm, _) => {
                            total += launcher.launch_op(ctx, &[gemm], ga, main)?.launches;
                        }
                        (SliceOp::Handoff, _) => {
                            let ev = &streams.produced[j];
                            ctx.hip.run_batch(&[
                                Job::RecordEvent {
                                    event: ev,
                                    stream: main,
                                },
                                Job::StreamWaitEvent {
                                    stream: &streams.side,
                                    event: ev,
                                },
                            ])?;
                        }
                        (SliceOp::OneShot, _) => {
                            let refs: Vec<&LaunchDescriptor> = sched.reduce.iter().collect();
                            total += launcher.launch_op(ctx, &refs, ra, &streams.side)?.launches;
                        }
                        (SliceOp::Push, Some(sp)) => {
                            flag = counters.next_split_flag()?;
                            let mut push = sp.push.clone();
                            set_slot(&mut push, "flag_val", i64::from(flag))?;
                            total += launcher
                                .launch_op(ctx, &[&push], ra, &streams.side)?
                                .launches;
                        }
                        (SliceOp::CpWait, Some(sp)) => {
                            let word = launcher
                                .ipc_region_addr(&sp.reduce, ra, "ar_flags")
                                .and_then(|a| a.offset(AR_SPLIT_FLAG_WORD * 4))
                                .ok_or_else(|| {
                                    Error::Refused(
                                        "--ar-form split: this rank's `ar_flags` region is not \
                                         prepared; the CP wait has no word to wait on"
                                            .to_owned(),
                                    )
                                })?;
                            ctx.hip.run_batch(&[Job::StreamWaitValueGte {
                                stream: &streams.side,
                                word,
                                value: flag,
                                mask: u32::MAX,
                            }])?;
                        }
                        (SliceOp::Reduce, Some(sp)) => {
                            total += launcher
                                .launch_op(ctx, &[&sp.reduce], ra, &streams.side)?
                                .launches;
                        }
                        (op, None) => {
                            return Err(Error::Refused(format!(
                                "the one-shot overlap has no `{op:?}` step"
                            )))
                        }
                    }
                }
                counters.slices.fetch_add(1, Ordering::Relaxed);
            }
            ctx.hip.run_batch(&[
                Job::RecordEvent {
                    event: &streams.done,
                    stream: &streams.side,
                },
                Job::StreamWaitEvent {
                    stream: main,
                    event: &streams.done,
                },
            ])?;
            counters.overlapped.fetch_add(1, Ordering::Relaxed);
            i += 2;
            continue;
        }
        let step = &steps[i];
        if is_reduce(step) && !sched.pairs.is_empty() {
            counters.sequential.fetch_add(1, Ordering::Relaxed);
        }
        let refs: Vec<&LaunchDescriptor> = step.descs.iter().collect();
        total += launcher.launch_op(ctx, &refs, &step.args, main)?.launches;
        i += 1;
    }
    Ok(total)
}

/// Dispatches an overlapped step issues beyond the plan's own: each pair's one GEMM and one reduce
/// become `slices` of each -- and in the split form each slice's reduce is TWO dispatches (push +
/// reduce; the CP wait is a stream op, not a dispatch), so a pair is `3 * slices` for the plan's 2.
#[must_use]
pub fn extra_launches(sched: &Schedule<'_>) -> u32 {
    let per_slice = if sched.split.is_some() { 3 } else { 2 };
    sched
        .pairs
        .iter()
        .map(|p| (per_slice * p.slices.len() as u32).saturating_sub(2))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use saber_hip::DeviceAddr;
    use saber_kernels::rules::{Shape, Slot, WorkspaceKind, WorkspaceReq};

    fn scalar(name: &str, v: i64) -> Slot {
        Slot {
            name: name.to_owned(),
            offset: 0,
            size: 4,
            value: ArgValue::I32 { ival: v },
        }
    }

    fn gemm(op: &str, m: u32, n: u32, k: u32, pb: i64) -> LaunchDescriptor {
        LaunchDescriptor {
            op: op.into(),
            plugin: "mxfp4_gemm_atiled".into(),
            kernel: "mxfp4_gemm_atiled_tn4_lbk64_wperm1".into(),
            grid: [n.div_ceil(128), m.div_ceil(256), 1],
            block: [256, 1, 1],
            shared_bytes: 0,
            kernarg_bytes: 0,
            args: vec![
                scalar("m", i64::from(m)),
                scalar("n", i64::from(n)),
                scalar("k", i64::from(k)),
                scalar("pb1", pb),
                scalar("pb2", pb),
            ],
            workspaces: vec![],
            n_launches: 1,
        }
    }

    fn reduce(kernel: &str, scratch: u64) -> LaunchDescriptor {
        LaunchDescriptor {
            op: "allreduce_wht".into(),
            plugin: "saber_ar_wht".into(),
            kernel: kernel.into(),
            grid: [96, 1, 1],
            block: [512, 1, 1],
            shared_bytes: 0,
            kernarg_bytes: 0,
            args: vec![],
            workspaces: vec![WorkspaceReq {
                name: "ar_scratch".into(),
                kind: WorkspaceKind::IpcPeerVisible,
                bytes: scratch,
                zero_once: true,
            }],
            n_launches: 1,
        }
    }

    fn addr(v: u64) -> DeviceAddr {
        DeviceAddr::NULL.offset(v).expect("a test address")
    }

    #[test]
    fn the_law_cuts_only_whole_buckets_at_or_above_its_floor() {
        let law = OverlapLaw {
            slice_rows: 1024,
            min_rows: 2048,
        };
        assert_eq!(law.slices(256), 1);
        assert_eq!(law.slices(1024), 1, "one slice is no overlap");
        assert_eq!(law.slices(2048), 2);
        assert_eq!(law.slices(8192), 8);
        assert_eq!(law.slices(3000), 1, "not a whole number of slices");
        let off = OverlapLaw {
            slice_rows: 0,
            min_rows: 0,
        };
        assert_eq!(off.slices(8192), 1);
    }

    #[test]
    fn a_gemm_slice_is_the_same_kernel_on_a_cut_row_grid() {
        let full = gemm("gemm_atiled", 2048, 5120, 3072, 5120);
        let s = gemm_slice(&full, 2048, 1024, 1024).unwrap();
        assert_eq!(
            s.kernel, full.kernel,
            "the symbol the full M resolved, never re-resolved"
        );
        assert_eq!(s.grid, [40, 4, 1]);
        assert_eq!(slot_i64(&s, "m").unwrap(), 1024);
        assert_eq!(slot_i64(&s, "n").unwrap(), 5120);
        let args = ArgBindings::new(Shape::new(2048, 5120, 3072))
            .with("a", addr(0x10_0000))
            .with("as", addr(0x20_0000))
            .with("c", addr(0x30_0000))
            .with("w", addr(0x40_0000));
        let sa = gemm_slice_args(&full, &args, 1024).unwrap();
        assert_eq!(sa.get("a"), Some(addr(0x10_0000 + 1024 * 3072)));
        assert_eq!(sa.get("as"), Some(addr(0x20_0000 + 1024 * 4)));
        assert_eq!(sa.get("c"), Some(addr(0x30_0000 + 1024 * 5120 * 2)));
        assert_eq!(
            sa.get("w"),
            Some(addr(0x40_0000)),
            "the weights are not sliced"
        );
    }

    #[test]
    fn a_slice_off_the_tile_grid_or_a_partitioned_gemm_is_refused() {
        let full = gemm("gemm_atiled", 2048, 5120, 3072, 5120);
        assert!(
            gemm_slice(&full, 2048, 100, 1024).is_err(),
            "off the 256-row grid"
        );
        assert!(
            gemm_slice(&full, 2048, 1024, 2048).is_err(),
            "past the bucket"
        );
        assert!(
            gemm_slice(&full, 4096, 0, 1024).is_err(),
            "m is not the bucket"
        );
        let merged = gemm("gemm_atiled", 2048, 8192, 5120, 4096);
        assert!(
            gemm_slice(&merged, 2048, 0, 1024).is_err(),
            "a merged projection"
        );
        let decode = gemm("gemm_decode", 2048, 5120, 3072, 5120);
        assert!(
            gemm_slice(&decode, 2048, 0, 1024).is_err(),
            "a split-K family"
        );
    }

    fn steps<'a>(
        g: &'a [LaunchDescriptor],
        ar: &'a [LaunchDescriptor],
        emb: &'a [LaunchDescriptor],
        c: DeviceAddr,
    ) -> Vec<BoundStep<'a>> {
        let gargs = ArgBindings::new(Shape::new(2048, 5120, 3072))
            .with("a", addr(0x1000))
            .with("as", addr(0x2000))
            .with("c", c);
        let rargs = ArgBindings::new(Shape::new(0, 0, 0))
            .with("in", c)
            .with("out", c);
        let eargs = ArgBindings::new(Shape::new(0, 0, 0))
            .with("in", addr(0x9000))
            .with("out", addr(0x9000));
        vec![
            BoundStep {
                name: "embed",
                descs: g,
                args: gargs.clone().with("c", addr(0x9000)),
            },
            BoundStep {
                name: "embed.all_reduce",
                descs: emb,
                args: eargs,
            },
            BoundStep {
                name: "layer.0.o_proj",
                descs: g,
                args: gargs,
            },
            BoundStep {
                name: "layer.0.all_reduce",
                descs: ar,
                args: rargs,
            },
        ]
    }

    #[test]
    fn only_a_reduce_over_its_producers_own_buffer_is_sliced() {
        let g = [gemm("gemm_atiled", 2048, 5120, 3072, 5120)];
        let ar = [reduce("saber_ar_oneshot_2rank_wht6_bf16", 1 << 26)];
        let emb = [reduce("saber_ar_oneshot_2rank_wht6_bf16", 1 << 26)];
        let slice = [reduce("saber_ar_oneshot_2rank_wht6_bf16", 1 << 24)];
        let c = addr(0x100_0000);
        let mut st = steps(&g, &ar, &emb, c);
        // The embedding's reduce reads 0x9000 and its "producer" wrote 0x9000 too -- make the
        // first pair a NON-alias to prove the alias is what qualifies a pair.
        st[0].args = st[0].args.clone().with("c", addr(0x8000));
        let law = OverlapLaw {
            slice_rows: 1024,
            min_rows: 2048,
        };
        let s = derive(&st, 2048, law, &slice).unwrap();
        assert_eq!(s.pairs.len(), 1);
        assert_eq!(
            s.sequential, 1,
            "the reduce whose input its predecessor did not write"
        );
        assert_eq!(s.pairs[0].producer, 2);
        let sl = &s.pairs[0].slices;
        assert_eq!(sl.len(), 2);
        assert_eq!(sl[1].2.get("in"), Some(addr(0x100_0000 + 1024 * 5120 * 2)));
        assert_eq!(
            sl[1].2.get("out"),
            sl[1].2.get("in"),
            "in place, as the full reduce"
        );
        assert_eq!(
            sl[1].1.get("c"),
            sl[1].2.get("in"),
            "the slice reduces what the slice wrote"
        );
        assert_eq!(extra_launches(&s), 2);
        // Below the floor: nothing is sliced and nothing is counted.
        let none = derive(&st, 1024, law, &slice).unwrap();
        assert!(none.pairs.is_empty());
    }

    fn served_slice_reduce(rows: u64) -> Vec<LaunchDescriptor> {
        let r = saber_kernels::saber::SaberRules::prod_saber().expect("rules");
        saber_kernels::saber::resolve_ar_banded(
            rows,
            5120,
            2,
            saber_kernels::saber::ArPayload::Bf16,
            2,
            &r,
            &saber_kernels::saber::Registry::new(&r),
        )
        .expect("the served slice reduce")
    }

    #[test]
    fn the_split_arm_issues_push_cp_wait_reduce_per_slice_and_the_one_shot_is_unchanged() {
        assert_eq!(
            slice_ops(false),
            &[SliceOp::Gemm, SliceOp::Handoff, SliceOp::OneShot],
            "H36's one-shot sequence, the control"
        );
        assert_eq!(
            slice_ops(true),
            &[
                SliceOp::Gemm,
                SliceOp::Handoff,
                SliceOp::Push,
                SliceOp::CpWait,
                SliceOp::Reduce
            ],
            "the push, THEN the command-processor wait, THEN the reduce, all after the handoff"
        );
        // The schedule: the same pairs, the split form attached, the budget 3 per slice.
        let g = [gemm("gemm_atiled", 2048, 5120, 3072, 5120)];
        let slice = served_slice_reduce(1024);
        let full = served_slice_reduce(2048);
        assert_eq!(slice[0].kernel, full[0].kernel);
        let st = steps(&g, &full, &full, addr(0x100_0000));
        let law = OverlapLaw {
            slice_rows: 1024,
            min_rows: 2048,
        };
        let one = derive(&st, 2048, law, &slice).unwrap();
        assert!(one.split.is_none(), "derive is the one-shot form");
        let ones = extra_launches(&one);
        let form = SplitForm::derive(&slice, 16).unwrap();
        assert_eq!(form.push.grid[0], 16);
        assert_eq!(form.reduce.grid, slice[0].grid);
        let split = derive(&st, 2048, law, &slice)
            .unwrap()
            .with_split(&form)
            .unwrap();
        // 2 pairs (embed's reduce aliases too in this fixture) x 2 slices.
        let pairs = u32::try_from(split.pairs.len()).unwrap();
        assert_eq!(ones, pairs * 2, "one-shot: 2 x 2 - 2 per pair");
        assert_eq!(
            extra_launches(&split),
            pairs * 4,
            "split: 3 x 2 - 2 per pair"
        );
        // A form derived from another one-shot is refused.
        let mut other = form.clone();
        other.source = "saber_ar_oneshot_2rank_wht4_bf16".into();
        assert!(derive(&st, 2048, law, &slice)
            .unwrap()
            .with_split(&other)
            .is_err());
        // The exact family has no split twin.
        assert!(SplitForm::derive(&served_slice_reduce(8), 4).is_err());
        assert!(SplitForm::derive(&slice, 0).is_err());
    }

    #[test]
    fn the_flag_law_is_monotone_from_one_and_the_same_on_both_ranks() {
        let (r0, r1) = (OverlapCounters::default(), OverlapCounters::default());
        let a: Vec<u32> = (0..300).map(|_| r0.next_split_flag().unwrap()).collect();
        let b: Vec<u32> = (0..300).map(|_| r1.next_split_flag().unwrap()).collect();
        assert_eq!(
            a, b,
            "two ranks issuing the same calls raise the same values"
        );
        assert_eq!(
            a[0], 1,
            "the boot-zeroed flag never satisfies the first wait"
        );
        assert!(a.windows(2).all(|w| w[1] == w[0] + 1), "strictly monotone");
        assert!(r0.line().ends_with("split=300)"));
        assert_eq!(
            OverlapCounters::default().line(),
            "ar_overlap(pairs=0 sequential=0 slices=0)",
            "the one-shot form's census line is H36's"
        );
        let edge = OverlapCounters::default();
        edge.split_calls
            .store(u64::from(i32::MAX.unsigned_abs()) - 1, Ordering::Relaxed);
        assert_eq!(edge.next_split_flag().unwrap(), i32::MAX.unsigned_abs());
        assert!(edge.next_split_flag().is_err(), "refused, never wrapped");
    }

    #[test]
    fn a_slice_in_another_family_or_wider_than_the_prepared_region_is_refused() {
        let g = [gemm("gemm_atiled", 2048, 5120, 3072, 5120)];
        let ar = [reduce("saber_ar_oneshot_2rank_wht6_bf16", 1 << 26)];
        let emb = ar.clone();
        let st = steps(&g, &ar, &emb, addr(0x100_0000));
        let law = OverlapLaw {
            slice_rows: 1024,
            min_rows: 2048,
        };
        let exact = [reduce("saber_ar_oneshot_2rank_exact_bf16", 1 << 24)];
        assert!(derive(&st, 2048, law, &exact).is_err(), "two roundings");
        let wide = [reduce("saber_ar_oneshot_2rank_wht6_bf16", 1 << 27)];
        assert!(
            derive(&st, 2048, law, &wide).is_err(),
            "an unprepared region"
        );
        let two = [ar[0].clone(), ar[0].clone()];
        assert!(
            derive(&st, 2048, law, &two).is_err(),
            "a slice must be one band"
        );
    }
}
