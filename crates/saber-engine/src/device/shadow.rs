//! **T6: the shadow digest's device side** (`--engine-race-check shadow`; [`crate::race::ShadowCheck`]).
//!
//! Built ONCE at boot beside the drafter resident: for every planned shape, two digest tables --
//! the FULL list (every seeded table, every arena region of the shape) and the LIGHT list (every
//! seeded table, the head's selection, the selector's inputs and the tail's outputs) -- uploaded
//! into one small device buffer, and a record buffer the digest kernels write their partials to.
//! A proposal then costs the host two enqueues after its last replay (the regions launch, and on a
//! FULL proposal the K/V pages launch) and one more `Job::MemcpyD2h` in the pinned batch that
//! already reads its tokens: no sync, no read-back, no frame on the wire, and no upload after boot.
//!
//! The aux window is the regions launch's per-launch `extra` region (its row moves with the
//! proposal). The drafter pages are walked on the card from the proposal's own seeded block table
//! and sequence lengths ([`saber_kernels::race_digest::resolve_kv`]).

use saber_exec::{ArgBindings, LaunchCtx, Launcher};
use saber_hip::{DeviceAddr, HipThread, Stream};
use saber_kernels::race_digest::{self, KvWalk, KV_PARTS, PARTS};
use saber_kernels::rules::{LaunchDescriptor, Shape};

use super::{device_err, AuxTap, DraftBuf, DraftResident};
use crate::forward::ForwardError;

/// A LIGHT proposal's arena regions: T5's light set (the head's selection and the selector's
/// inputs) plus the tail's outputs T5 read on every proposal.
const LIGHT: [&str; 12] = [
    "head.hidden",
    "head.thresh",
    "head.tvals",
    "head.tpos",
    "head.cand80",
    "sel.vals",
    "sel.cand64",
    "sel.anchor",
    "sel.proj",
    "sel.scores",
    "out.realized",
    "out.tokens",
];

/// **T7: the bytes of table `tb.<name>` one proposal of this shape SEEDS** (the frame's own table
/// widths, `DraftWire::unpack`; `ctx.positions` is the base replicated per drafter layer and
/// `ctx.slots` one copy per layer), or `None` for a table the boot seeds once (a constant), which a
/// LIGHT proposal does not digest.
fn seeded_span(name: &str, w: &crate::draft_inputs::DraftWire, layers: u32) -> Option<u64> {
    let ctx = u64::from(w.ctx_rows);
    let q = u64::from(w.query_rows);
    let r = u64::from(w.reqs);
    let st = u64::from(w.steps);
    let name = name.strip_prefix("tb.").unwrap_or(name);
    Some(match name {
        "ctx.positions" => ctx * 8 * u64::from(layers),
        n if n.starts_with("ctx.slots.l") => ctx * 8,
        "q.input_ids" => q * 4,
        "q.pos" | "idx.query_slots" => q * 8,
        "idx.block_table" => r * u64::from(w.bt_stride) * 4,
        "idx.seq_lens" => r * 4,
        "idx.sample_indices" => st * 8,
        "idx.anchor_indices" => r * 8,
        "walk.idx_mapping" => st * 4,
        _ => return None,
    })
}

/// Named stage digests, in order.
type Stages = Vec<(String, u64)>;

/// One digest table: `names[i]` / `bytes[i]` per region, the aux window LAST (the launch's
/// `extra`), its offset in the table buffer, and its launch.
#[derive(Debug)]
pub(super) struct ShadowList {
    names: Vec<String>,
    bytes: Vec<u64>,
    desc_off: u64,
    launch: LaunchDescriptor,
}

impl ShadowList {
    /// Partials the regions launch writes.
    fn partials(&self) -> usize {
        self.names.len() * PARTS as usize
    }
}

/// One shape's two tables and its K/V launch.
#[derive(Debug)]
pub(super) struct ShadowShape {
    full: ShadowList,
    light: ShadowList,
    kv: LaunchDescriptor,
    kv_partials: usize,
}

/// The shadow's device state (on [`DraftResident::shadow`]).
#[derive(Debug)]
pub(super) struct ShadowDev {
    /// [`DraftResident::bufs`] index of the digest tables.
    desc: usize,
    /// ... of the record the kernels write.
    rec: usize,
    /// Bytes of the record.
    rec_bytes: u64,
    shapes: Vec<ShadowShape>,
    /// T8: the TARGET's state, digested on a FULL proposal ([`attach_target`]).
    target: Option<TargetDigest>,
}

/// **T8: the target's state at a FULL proposal**: for every GDN state slot, one table of every
/// slotted state buffer's slice at that slot (per layer: the recurrent state and the conv cache),
/// then the prefill head's arena -- uploaded once at boot, one table per slot, so a proposal picks
/// its slot's table by offset and uploads nothing. On a first proposal the slot is the one the
/// prefill chunk just wrote (`ShadowCheck::prefill_slot`), so a moved target prefill names its
/// first differing LAYER here, upstream of `aux`.
#[derive(Debug)]
struct TargetDigest {
    /// [`DraftResident::bufs`] index of the per-slot tables.
    desc: usize,
    /// Bytes of one slot's table.
    stride: u64,
    slots: u32,
    names: Vec<String>,
    bytes: Vec<u64>,
    launch: LaunchDescriptor,
}

/// T8: what a proposal hands the shadow beyond T6's: the aux window's LIVE bytes (the rows a
/// store writes; the rest are the prompt's padded tile rows) and the target's GDN slot.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ShadowExtra {
    /// Bytes of the aux window's live rows, when fewer than the window.
    pub live_bytes: Option<u64>,
    /// The GDN slot whose target state a FULL proposal digests.
    pub slot: Option<u32>,
}

/// What one proposal's shadow launches wrote: which list, and how many record bytes to read.
#[derive(Clone, Copy, Debug)]
pub(super) struct ShadowTicket {
    ix: usize,
    full: bool,
    bytes: usize,
    /// T8: the aux window was split (live rows, padded rows).
    split: bool,
    /// T8: the target's state was digested.
    target: bool,
}

impl ShadowTicket {
    /// Record bytes the launches wrote.
    pub(super) fn bytes(self) -> usize {
        self.bytes
    }
}

impl ShadowDev {
    /// The tables of every shape of `res`, uploaded once; the record allocated. Every buffer the
    /// resident holds is digested whole unless it is the arena (cut by region) or a K/V cache
    /// (walked by page).
    pub(super) fn build(
        hip: &HipThread,
        stream: &Stream,
        res: &DraftResident,
        tap: &AuxTap,
    ) -> Result<(ShadowDev, saber_hip::DevicePtr, saber_hip::DevicePtr), ForwardError> {
        let addr = |b: DraftBuf| -> Result<DeviceAddr, ForwardError> {
            res.bufs[b.idx].addr().offset(b.off).ok_or_else(|| {
                ForwardError::Device("the shadow digest's table address overflows".to_owned())
            })
        };
        let mut tables: Vec<(String, DeviceAddr, u64)> = Vec::new();
        for (name, b) in &res.named {
            if name.starts_with("arena") || name.starts_with("kv.k") || name.starts_with("kv.v") {
                continue;
            }
            tables.push((format!("tb.{name}"), addr(*b)?, b.bytes));
        }
        let row_bytes = tap.plan.row_elems() * 2;
        let geom = res.kv_geom();
        let mut words: Vec<u64> = Vec::new();
        let mut shapes = Vec::with_capacity(res.shapes.len());
        let mut rec_bytes = 0u64;
        for ix in 0..res.shapes.len() {
            let e = res.shape(ix)?;
            let aux_bytes = u64::from(e.plan.shape.ctx_rows) * row_bytes;
            let wire = res.wire_at(ix)?;
            let layers = res.cfg.layers;
            let mut regions: Vec<(String, crate::draft_plan::Region)> = e
                .plan
                .arena
                .regions()
                .map(|(n, r)| (n.to_owned(), r))
                .collect();
            regions.sort_by_key(|(_, r)| r.at);
            let mut list = |light: bool| -> Result<ShadowList, ForwardError> {
                let mut names = Vec::new();
                let mut bytes = Vec::new();
                let desc_off = words.len() as u64 * 8;
                for (n, a, b) in &tables {
                    // T7's trim: a LIGHT proposal digests each table's span THIS shape's frame
                    // seeds (`seeded_span`), not the buffer sized for the widest shape, and skips
                    // the boot-seeded constants (a FULL proposal still digests every table whole).
                    let b = if light {
                        match seeded_span(n, &wire, layers) {
                            Some(span) => span.min(*b),
                            None => continue,
                        }
                    } else {
                        *b
                    };
                    names.push(n.clone());
                    bytes.push(b);
                    words.extend_from_slice(&[a.as_u64(), b]);
                }
                for (n, r) in &regions {
                    if light && !LIGHT.contains(&n.as_str()) {
                        continue;
                    }
                    let b = res.arena_buf(ix, n)?;
                    names.push(format!("a.{n}"));
                    bytes.push(b.bytes);
                    words.extend_from_slice(&[addr(b)?.as_u64(), r.bytes]);
                }
                let n = u32::try_from(names.len()).map_err(|_| {
                    ForwardError::Device("the shadow digest's table is past u32".to_owned())
                })?;
                names.push("aux".to_owned());
                bytes.push(aux_bytes);
                let launch = race_digest::resolve_regions(n, aux_bytes)
                    .map_err(device_err("the shadow digest's regions launch"))?;
                Ok(ShadowList {
                    names,
                    bytes,
                    desc_off,
                    launch,
                })
            };
            let full = list(false)?;
            let light = list(true)?;
            let walk = KvWalk {
                reqs: e.plan.shape.reqs,
                bt_stride: e.plan.shape.bt_stride,
                page_tokens: res.kv_block,
                layers: res.cfg.layers,
                blocks: res.blocks,
                per_layer: geom.per_layer(),
                block_bytes: res.block_bytes,
            };
            let kv = race_digest::resolve_kv(walk)
                .map_err(device_err("the shadow digest's K/V launch"))?;
            let kv_partials = walk.partials() as usize;
            // T8: + the aux window's live / padded split (two regions).
            rec_bytes =
                rec_bytes.max(((full.partials() + kv_partials + 2 * PARTS as usize) * 8) as u64);
            shapes.push(ShadowShape {
                full,
                light,
                kv,
                kv_partials,
            });
        }
        let desc_bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let desc = hip
            .malloc(desc_bytes.len().max(8) as u64)
            .map_err(device_err("the shadow digest's tables"))?;
        hip.memcpy_h2d_blocking(&desc, &desc_bytes, stream)
            .map_err(device_err("uploading the shadow digest's tables"))?;
        let rec = hip
            .malloc(rec_bytes.max(8))
            .map_err(device_err("the shadow digest's record"))?;
        hip.memset(&rec, 0, stream)
            .map_err(device_err("zeroing the shadow digest's record"))?;
        tracing::info!(
            shapes = shapes.len(),
            tables = tables.len(),
            desc_bytes = desc_bytes.len(),
            rec_bytes,
            "racecheck shadow ARMED (T6 diagnostic): every proposal's stages digested on the \
             card after its last kernel, read in its own pinned batch; no check of T4/T5 runs"
        );
        // The resident owns every allocation for the life of the process ([`attach`] pushes
        // these two and sets their indices).
        Ok((
            ShadowDev {
                desc: usize::MAX,
                rec: usize::MAX,
                rec_bytes,
                shapes,
                target: None,
            },
            desc,
            rec,
        ))
    }

    /// The record's buffer, for the proposal's pinned read.
    pub(super) fn record(&self, t: ShadowTicket) -> DraftBuf {
        DraftBuf {
            idx: self.rec,
            off: 0,
            bytes: self.rec_bytes.min(t.bytes as u64),
        }
    }

    /// **Enqueue the digests** of proposal shape `ix` on `stream`, after its last kernel: the
    /// regions launch (the aux window at `aux_at`) and, `full`, the K/V pages launch. No wait.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch(
        &self,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        stream: &Stream,
        res: &DraftResident,
        ix: usize,
        full: bool,
        aux_at: DeviceAddr,
        x: ShadowExtra,
    ) -> Result<ShadowTicket, ForwardError> {
        let off = |a: DeviceAddr, by: u64| -> Result<DeviceAddr, ForwardError> {
            a.offset(by).ok_or_else(|| {
                ForwardError::Device("the shadow digest's address overflows".to_owned())
            })
        };
        let sh = self
            .shapes
            .get(ix)
            .ok_or_else(|| ForwardError::Device(format!("the shadow digest has no shape {ix}")))?;
        let list = if full { &sh.full } else { &sh.light };
        let base = |i: usize, off: u64| -> Result<DeviceAddr, ForwardError> {
            res.bufs[i].addr().offset(off).ok_or_else(|| {
                ForwardError::Device("the shadow digest's address overflows".to_owned())
            })
        };
        let rec = base(self.rec, 0)?;
        let args = ArgBindings::new(Shape::new(0, 0, 0))
            .with("desc", base(self.desc, list.desc_off)?)
            .with("extra", aux_at)
            .with("out", rec);
        launcher
            .launch_op(ctx, &[&list.launch], &args, stream)
            .map_err(|e| ForwardError::Device(format!("the shadow digest's regions: {e}")))?;
        let mut bytes = list.partials() * 8;
        if full {
            let buf = |n: &str| -> Result<DeviceAddr, ForwardError> {
                let b = res.buf(n)?;
                base(b.idx, b.off)
            };
            let args = ArgBindings::new(Shape::new(0, 0, 0))
                .with("block_table", buf("idx.block_table")?)
                .with("seq_lens", buf("idx.seq_lens")?)
                .with("kbase", buf("kv.k")?)
                .with("vbase", buf("kv.v")?)
                .with("out", base(self.rec, bytes as u64)?);
            launcher
                .launch_op(ctx, &[&sh.kv], &args, stream)
                .map_err(|e| ForwardError::Device(format!("the shadow digest's K/V: {e}")))?;
            bytes += sh.kv_partials * 8;
        }
        // T8: THE AUX WINDOW, LIVE ROWS AND PADDED ROWS APART (two more regions launches over the
        // same window, each one region, no table): T6's `aux` spans the prompt's padded tile rows,
        // which no live row reads, so only `aux.live` says whether the target's live rows moved.
        let aux_bytes = list.bytes.last().copied().unwrap_or(0);
        let split = match x.live_bytes {
            Some(live) if live > 0 && live < aux_bytes => {
                for (at, n) in [(aux_at, live), (off(aux_at, live)?, aux_bytes - live)] {
                    let d = race_digest::resolve_regions(0, n)
                        .map_err(|e| ForwardError::Device(format!("the aux split: {e}")))?;
                    let args = ArgBindings::new(Shape::new(0, 0, 0))
                        .with("desc", rec)
                        .with("extra", at)
                        .with("out", base(self.rec, bytes as u64)?);
                    launcher
                        .launch_op(ctx, &[&d], &args, stream)
                        .map_err(|e| ForwardError::Device(format!("the aux split: {e}")))?;
                    bytes += PARTS as usize * 8;
                }
                true
            }
            _ => false,
        };
        // T8: THE TARGET'S STATE at the slot the last prefill chunk wrote (FULL only).
        let target = match (full, &self.target, x.slot) {
            (true, Some(t), Some(slot)) if slot < t.slots => {
                let args = ArgBindings::new(Shape::new(0, 0, 0))
                    .with("desc", base(t.desc, u64::from(slot) * t.stride)?)
                    .with("extra", rec)
                    .with("out", base(self.rec, bytes as u64)?);
                launcher
                    .launch_op(ctx, &[&t.launch], &args, stream)
                    .map_err(|e| ForwardError::Device(format!("the target digest: {e}")))?;
                bytes += (t.names.len() + 1) * PARTS as usize * 8;
                true
            }
            _ => false,
        };
        Ok(ShadowTicket {
            ix,
            full,
            bytes,
            split,
            target,
        })
    }

    /// The device stages of one proposal from its record: `(inputs, arena)` -- the seeded tables,
    /// the aux window and (full) the K/V pages per layer and cache; then the arena regions in layout
    /// order. `None` when the record is not the ticket's length.
    pub(super) fn stages(
        &self,
        t: ShadowTicket,
        rec: &[u8],
        layers: u32,
    ) -> Option<(Stages, Stages)> {
        if rec.len() != t.bytes {
            return None;
        }
        let sh = self.shapes.get(t.ix)?;
        let list = if t.full { &sh.full } else { &sh.light };
        let words: Vec<u64> = rec
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap_or_default()))
            .collect();
        let p = PARTS as usize;
        let mut inputs = Vec::new();
        let mut aux = Vec::new();
        let mut arena = Vec::new();
        let n = list.names.len();
        for (i, name) in list.names.iter().enumerate() {
            let d = race_digest::fold(&words[i * p..(i + 1) * p], list.bytes[i]);
            if name.starts_with("a.") {
                arena.push((name.clone(), d));
            } else if name == "aux" {
                aux.push((name.clone(), d));
            } else {
                inputs.push((name.clone(), d));
            }
        }
        let mut at = n * p;
        let mut kv = Vec::new();
        if t.full {
            let kp = KV_PARTS as usize;
            for l in 0..layers as usize {
                for (c, half) in ["k", "v"].iter().enumerate() {
                    let o = at + (l * 2 + c) * kp;
                    kv.push((
                        format!("kv.{half}.l{l}"),
                        race_digest::fold_kv(words.get(o..o + kp)?),
                    ));
                }
            }
            at += sh.kv_partials;
        }
        // T8: the split's two regions come right after, then the target's; each folded with its
        // own length (the split's live length is the ticket's; the pad is the rest).
        let mut split = Vec::new();
        if t.split {
            for name in ["aux.live", "aux.pad"] {
                let w = words.get(at..at + p)?;
                // The partials' XOR is the unkeyed-by-length digest; the length is folded in as
                // `fold` does, with the region's own byte count unknown here: 0 keeps it a pure
                // function of the bytes and their positions (the lengths are fixed per shape).
                split.push((name.to_owned(), race_digest::fold(w, 0)));
                at += p;
            }
        }
        let mut target = Vec::new();
        if let (true, Some(tg)) = (t.target, self.target.as_ref()) {
            for (i, name) in tg.names.iter().enumerate() {
                let o = at + i * p;
                target.push((
                    format!("t.{name}"),
                    race_digest::fold(words.get(o..o + p)?, tg.bytes[i]),
                ));
            }
        }
        // Order: the host's inputs and the seeded tables, the TARGET's state (upstream of
        // everything the drafter reads), the aux window (live, padded, whole), the pages.
        inputs.extend(target);
        inputs.extend(split);
        inputs.extend(aux);
        inputs.extend(kv);
        Some((inputs, arena))
    }
}

/// Arm the shadow on `res`: build its tables and give the resident its two allocations.
///
/// # Errors
/// An allocation, an upload or a launch that does not resolve.
pub(super) fn attach(
    hip: &HipThread,
    stream: &Stream,
    res: &mut DraftResident,
    tap: &AuxTap,
) -> Result<(), ForwardError> {
    let (mut dev, desc, rec) = ShadowDev::build(hip, stream, res, tap)?;
    dev.desc = res.bufs.len();
    res.bufs.push(desc);
    dev.rec = res.bufs.len();
    res.bufs.push(rec);
    res.shadow = Some(dev);
    Ok(())
}

/// One slotted target state buffer the T8 target digest reads: its name, base, and bytes a slot.
pub(super) struct TargetState<'a> {
    pub name: &'a str,
    pub base: DeviceAddr,
    pub per_slot: u64,
    pub slots: u32,
}

/// **T8: arm the target digest** on a resident that already carries the shadow: one table per GDN
/// slot (every slotted state buffer's slice at that slot, then `head` = the prefill head's arena),
/// uploaded once; the record grown to hold it. Nothing is uploaded after boot.
///
/// # Errors
/// An allocation, an upload or a launch that does not resolve.
pub(super) fn attach_target(
    hip: &HipThread,
    stream: &Stream,
    res: &mut DraftResident,
    states: &[TargetState<'_>],
    head: (DeviceAddr, u64),
) -> Result<(), ForwardError> {
    let Some(dev) = res.shadow.as_ref() else {
        return Ok(());
    };
    let slots = states.iter().map(|s| s.slots).min().unwrap_or(0);
    let mut names: Vec<String> = states.iter().map(|s| s.name.to_owned()).collect();
    names.push("prefill_head".to_owned());
    let mut bytes: Vec<u64> = states.iter().map(|s| s.per_slot).collect();
    bytes.push(head.1);
    let mut words: Vec<u64> = Vec::with_capacity(slots as usize * names.len() * 2);
    for slot in 0..slots {
        for s in states {
            let a = s
                .base
                .offset(u64::from(slot) * s.per_slot)
                .ok_or_else(|| ForwardError::Device("a target slot past its state".to_owned()))?;
            words.extend_from_slice(&[a.as_u64(), s.per_slot]);
        }
        words.extend_from_slice(&[head.0.as_u64(), head.1]);
    }
    let n = u32::try_from(names.len())
        .map_err(|_| ForwardError::Device("the target digest's table is past u32".to_owned()))?;
    let launch = race_digest::resolve_regions(n, 0)
        .map_err(device_err("the shadow's target digest launch"))?;
    let stride = names.len() as u64 * 16;
    let desc_bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let desc = hip
        .malloc(desc_bytes.len().max(8) as u64)
        .map_err(device_err("the shadow's target tables"))?;
    hip.memcpy_h2d_blocking(&desc, &desc_bytes, stream)
        .map_err(device_err("uploading the shadow's target tables"))?;
    let rec_bytes = dev.rec_bytes + (names.len() as u64 + 1) * u64::from(PARTS) * 8;
    let rec = hip
        .malloc(rec_bytes)
        .map_err(device_err("the shadow digest's record"))?;
    hip.memset(&rec, 0, stream)
        .map_err(device_err("zeroing the shadow digest's record"))?;
    tracing::info!(
        slots,
        regions = names.len(),
        desc_bytes = desc_bytes.len(),
        rec_bytes,
        "racecheck shadow TARGET digest armed (T8): a FULL proposal digests the target's GDN \
         state at the last prefill's slot, per layer, and the prefill head's arena"
    );
    let desc_idx = res.bufs.len();
    res.bufs.push(desc);
    let rec_idx = res.bufs.len();
    res.bufs.push(rec);
    if let Some(dev) = res.shadow.as_mut() {
        dev.rec = rec_idx;
        dev.rec_bytes = rec_bytes;
        dev.target = Some(TargetDigest {
            desc: desc_idx,
            stride,
            slots,
            names,
            bytes,
            launch,
        });
    }
    Ok(())
}
