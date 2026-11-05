//! The device mover for the Flash-Next expert tier (workstream FN8; docs/EXEC.md §14, "the
//! mover's ordering law").
//!
//! [`DeviceMover`] is the [`ExpertMover`] the engine runs: the tier's promotions (RAM → VRAM) and
//! demotions (VRAM → RAM) become asynchronous copies on ONE ordered copy stream per rank, the NVMe
//! demand reads stay on FN7's reader threads (one aligned `O_DIRECT` read per record straight into
//! the pinned slot), and a per-layer "landed" mark is what a consumer of that layer's slots waits
//! on. The device side is a [`CopyQueue`]: [`HipCopyQueue`] (saber-hip: one `hipMalloc` slot
//! arena from the byte ledger, one `hipHostMalloc` RAM slab from the host budget, one copy stream,
//! one event per layer) or [`FakeQueue`] (host memory, copies applied only when a consumer waits
//! or the host drains — the tests' model of an asynchronous device, so a missing wait READS STALE
//! BYTES instead of passing by accident).
//!
//! **The ordering law** (what makes the swap through one spare RAM slot safe while the MoE kernels
//! keep running):
//!
//! 1. **Fence.** A step's copies start only after the consumer's reads of that step: the step loop
//!    calls [`DeviceMover::after`] (the copy stream waits on the compute stream's work so far)
//!    before [`ExpertTier::end_step`](crate::expert_tier::ExpertTier::end_step). A copy queued
//!    without that fence is refused ([`TierError::Mover`]): the spare the D2H writes is the RAM slot
//!    the step's UVA read of the promoted record just used.
//! 2. **One stream, call order.** The tier queues `D2H victim → spare` before `H2D promoted →
//!    victim's slot`; one in-order stream keeps it so, and the next swap's D2H into the new spare
//!    (the promoted record's old RAM slot) runs after the H2D that read it.
//! 3. **Landed per layer.** After a layer's copies the mover records that layer's mark; before a
//!    layer's grouped launches read its `xptr` placements the consumer calls
//!    [`DeviceMover::gate`], which makes the consumer stream wait on the mark when (and only when)
//!    copies for that layer were issued since its last gate. `fill_xptr` may run on the host at any
//!    time: what it publishes is only READ after the gate.
//! 4. **Host writes drain first.** An NVMe read into a pinned slot (the demand path, the prefetch)
//!    drains the copy stream before it writes; the caller has already synchronised the consumer
//!    (the router's readback does, before `resolve`).
//!
//! **FN17, the deferred publication law** ([`DeviceMover::with_defer`], the tier's
//! `defer_spares`): a step's copies write only slots no placement points at (a promotion into a
//! HOLE of its layer, a demotion into a free RAM spare), so the NEXT step keeps reading the old
//! placements while they fly; at that step's fence [`DeviceMover::after`] also makes the consumer
//! wait for every copy issued so far (a step old by then), and the tier publishes them in its
//! `end_step`. The per-layer gate is then free: no step waits for the copies of its own end.
//!
//! **FN54b, the two-queue mover** (`--xtier-mover two-queue`; [`HipCopyQueue::enable_two_queue`],
//! [`FakeQueue::two_queue`]): the copies split over TWO in-order lanes -- [`Lane::Urgent`] (the
//! promotions H2D and the layer marks a gate waits on) and [`Lane::Background`] (the demotions
//! D2H) -- so the link's two directions run at once instead of in call order. Law 2 (one stream,
//! call order) becomes a SLOT law kept by events ([`LanePlanner`]): an op waits on the other
//! lane's mark after every earlier op that wrote a slot it touches or read a slot it writes (a
//! promotion into a victim's VRAM slot waits on that victim's D2H; a D2H into a RAM slot waits on
//! the H2D that last read it), and a layer's mark also waits on every demotion of its batch, so
//! law 3 is unchanged. The bytes at every read are the single lane's; only the copies' overlap
//! moves. With FN8's one swap spare the slot law is a chain (each D2H writes the RAM slot the
//! previous H2D read): the tier's `spare_ring` rotates extra spares so the chain breaks.

#![forbid(unsafe_code)]

use std::collections::VecDeque;

use saber_hip::{DevicePtr, Event, EventFlags, HipThread, Job, MemcpyKind, PinnedBuf, Stream};

use crate::dio::AlignedBuf;
use crate::expert_tier::{ExpertMover, TierError};
use crate::xtier::XtierFile;

/// One operation on the copy stream, in issue order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyOp {
    /// RAM slot → VRAM slot (one stride).
    H2d {
        /// Source RAM slot.
        ram: u32,
        /// Destination VRAM slot.
        vram: u32,
    },
    /// VRAM slot → RAM slot (one stride).
    D2h {
        /// Source VRAM slot.
        vram: u32,
        /// Destination RAM slot.
        ram: u32,
    },
    /// Every copy before this one has landed for `layer`'s consumers.
    Landed {
        /// The layer whose slots the preceding copies changed.
        layer: u32,
    },
}

/// FN54b: the copy lane of an op under the two-queue mover.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lane {
    /// Promotions (H2D) and the layer marks a gate waits on.
    Urgent = 0,
    /// Demotions (D2H): deferrable, the link's other direction.
    Background = 1,
}

impl Lane {
    fn other(self) -> Lane {
        match self {
            Lane::Urgent => Lane::Background,
            Lane::Background => Lane::Urgent,
        }
    }
}

/// FN54b: one planned job of a two-lane queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneOp {
    /// A copy (or a layer mark) on `lane`.
    Op {
        /// The lane.
        lane: Lane,
        /// The op.
        op: CopyOp,
    },
    /// `lane` marks its copy number `seq` (1-based, in lane order) done.
    Mark {
        /// The lane.
        lane: Lane,
        /// The lane's copy count so far.
        seq: u64,
    },
    /// `lane` waits for the OTHER lane's mark `seq`.
    Wait {
        /// The waiting lane.
        lane: Lane,
        /// The other lane's mark.
        seq: u64,
    },
}

#[derive(Debug, Clone, Copy, Default)]
struct SlotUse {
    /// The last writer: (lane index, seq); seq 0 = none.
    w: (usize, u64),
    /// The last reader on each lane (seq; 0 = none).
    r: [u64; 2],
}

/// FN54c: copies per lane between two marks under the batch-mark law (`LanePlanner::group` = 0).
/// The spare ring's FIFO lets a demotion depend on the promotion `spares` before it; a group of
/// `g` keeps a lane's next group free of the other lane's current one while `spares >= 2 g` (the
/// served ring: 1 + 4 spares -> 2).
pub const DEFAULT_LANE_GROUP: usize = 2;

/// FN54b / FN54c: the slot law of the two-queue mover: splits a call-ordered op list over the two
/// lanes with the cross-lane waits that keep every slot's reads and writes in call order (see the
/// module docs).
///
/// FN54c (the batch-mark law): FN54b marked EVERY copy and waited per op (~90 extra HIP calls a
/// b8 step on the engine thread). Now a batch is planned in two passes: (1) each op's dependency
/// on the other lane (the same slot tables, in call order); (2) a list schedule that emits each
/// lane's ops in their own order, `group` copies at a time, alternating lanes; a lane MARKS only
/// after a group some later op of the other lane depends on (or lazily at its tail, for a
/// dependency on an earlier batch), and an op WAITS only when its dependency is past the lane's
/// last wait -- on the other lane's latest mark (a superset: conservative, still exact). Every
/// read sees the call-order bytes; the device overlap is one group's.
#[derive(Debug, Default)]
pub struct LanePlanner {
    seq: [u64; 2],
    /// `waited[a]` = the other lane's highest mark lane `a` has waited on.
    waited: [u64; 2],
    /// FN54c: each lane's latest recorded mark (0 = none).
    marked: [u64; 2],
    vram: std::collections::HashMap<u32, SlotUse>,
    ram: std::collections::HashMap<u32, SlotUse>,
    /// The latest demotion since the last layer mark (0 = none).
    bg_open: u64,
    /// Cross-lane waits planned (the counter of the slot law).
    pub waits: u64,
    /// FN54c: lane marks planned.
    pub marks: u64,
    /// FN54c: copies per lane between marks (0 = [`DEFAULT_LANE_GROUP`]).
    pub group: usize,
    /// A CONTROL (tests only): plan no cross-lane wait at all.
    pub no_waits: bool,
}

/// One op of a batch with its dependency on the other lane (FN54c's pass 1).
#[derive(Debug, Clone, Copy)]
struct Planned {
    op: CopyOp,
    /// The other lane's copy this op must follow (0 = none).
    dep: u64,
}

impl LanePlanner {
    /// The lane `op` runs on.
    #[must_use]
    pub fn lane(op: CopyOp) -> Lane {
        match op {
            CopyOp::D2h { .. } => Lane::Background,
            CopyOp::H2d { .. } | CopyOp::Landed { .. } => Lane::Urgent,
        }
    }

    /// Pass 1: `op`'s dependency on the other lane; updates the slot tables and the lane counts.
    fn depend(&mut self, op: CopyOp) -> u64 {
        let lane = Self::lane(op);
        let (a, b) = (lane as usize, lane.other() as usize);
        let (rd, wr) = match op {
            CopyOp::Landed { .. } => {
                let s = self.bg_open;
                self.bg_open = 0;
                return s;
            }
            CopyOp::H2d { ram, vram } => ((false, ram), (true, vram)),
            CopyOp::D2h { vram, ram } => ((true, vram), (false, ram)),
        };
        let mut dep = 0u64;
        {
            let t = if rd.0 { &self.vram } else { &self.ram };
            if let Some(u) = t.get(&rd.1) {
                if u.w.0 == b {
                    dep = dep.max(u.w.1);
                }
            }
            let t = if wr.0 { &self.vram } else { &self.ram };
            if let Some(u) = t.get(&wr.1) {
                if u.w.0 == b {
                    dep = dep.max(u.w.1);
                }
                dep = dep.max(u.r[b]);
            }
        }
        self.seq[a] += 1;
        let seq = self.seq[a];
        let t = if rd.0 { &mut self.vram } else { &mut self.ram };
        t.entry(rd.1).or_default().r[a] = seq;
        let t = if wr.0 { &mut self.vram } else { &mut self.ram };
        t.entry(wr.1).or_default().w = (a, seq);
        if lane == Lane::Background {
            self.bg_open = seq;
        }
        dep
    }

    /// The jobs of `ops` (call order) on the two lanes, appended to `out` in a valid host order
    /// (every wait after the mark it waits on; each lane's ops in call order).
    pub fn plan(&mut self, ops: &[CopyOp], out: &mut Vec<LaneOp>) {
        let mut emitted = self.seq;
        let mut pending: [VecDeque<Planned>; 2] = [VecDeque::new(), VecDeque::new()];
        for &op in ops {
            let dep = self.depend(op);
            pending[Self::lane(op) as usize].push_back(Planned { op, dep });
        }
        let g = if self.group == 0 {
            DEFAULT_LANE_GROUP
        } else {
            self.group
        };
        let is_copy = |p: &Planned| !matches!(p.op, CopyOp::Landed { .. });
        while !pending[0].is_empty() || !pending[1].is_empty() {
            let mut progressed = false;
            for lane in [Lane::Background, Lane::Urgent] {
                let (a, b) = (lane as usize, lane.other() as usize);
                let mut n = 0usize;
                while let Some(&p) = pending[a].front() {
                    if (is_copy(&p) && n == g) || p.dep > emitted[b] {
                        break;
                    }
                    if p.dep > self.waited[a] && !self.no_waits {
                        if p.dep > self.marked[b] {
                            // lazily, at the other lane's tail (an earlier batch's copy)
                            self.marked[b] = emitted[b];
                            self.marks += 1;
                            out.push(LaneOp::Mark {
                                lane: lane.other(),
                                seq: emitted[b],
                            });
                        }
                        self.waited[a] = self.marked[b];
                        self.waits += 1;
                        out.push(LaneOp::Wait {
                            lane,
                            seq: self.marked[b],
                        });
                    }
                    pending[a].pop_front();
                    out.push(LaneOp::Op { lane, op: p.op });
                    if is_copy(&p) {
                        emitted[a] += 1;
                        n += 1;
                    }
                    progressed = true;
                }
                // mark the group only when a later op of the other lane depends on it
                if emitted[a] > self.marked[a]
                    && pending[b]
                        .iter()
                        .any(|q| q.dep > self.marked[a] && q.dep <= emitted[a])
                {
                    self.marked[a] = emitted[a];
                    self.marks += 1;
                    out.push(LaneOp::Mark {
                        lane,
                        seq: emitted[a],
                    });
                }
            }
            // the earliest pending op in call order depends only on emitted ops: never stuck
            assert!(
                progressed,
                "the lane planner stalled (a dependency on a later op)"
            );
        }
        debug_assert_eq!(emitted, self.seq);
    }
}

/// FN63b: the word a [`SigWhat::Mark`] writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarkWord {
    /// A layer's landed word: a consumer of the layer's slots waits until it reaches the mark.
    Layer(u32),
    /// The ALL word: every op issued before the mark has landed.
    All,
}

/// FN63b: what one lane job does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SigWhat {
    /// A copy (H2D on the lane's host-to-device engine, D2H on its device-to-host engine).
    Copy(CopyOp),
    /// Write `value` into `word` (a 4-byte H2D on the host-to-device engine).
    Mark {
        /// The word.
        word: MarkWord,
        /// Its new value (per word: 1, 2, 3, ..; a consumer waits `word >= value`).
        value: u32,
    },
}

/// FN63b: one job of the SDMA lane: run `what` once every ticket in `deps` has landed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SigJob {
    /// The job's ticket (1, 2, 3, .. in issue order).
    pub ticket: u64,
    /// What it does.
    pub what: SigWhat,
    /// The tickets it waits for (none landed when planned, by the caller's view).
    pub deps: Vec<u64>,
}

/// FN63b, THE SIGNAL LAW of the HSA copy lane (`--xtier-mover hsa-lane`): no stream order at all
/// -- every copy is a ticket with its own completion signal, and order is ONLY what a dependency
/// names:
///
/// 1. **Slots.** A copy waits for the last earlier op that touched (read or wrote) either of its
///    two slots (its VRAM slot and its RAM slot). Each op depends on the previous toucher, so the
///    touchers of a slot form a chain: every read sees the bytes of the call order (a promotion
///    into a victim's VRAM slot after the victim's demotion read it; a demotion into a spare after
///    the promotion that last read it), and the link's two directions overlap wherever the slots
///    allow (FN54b's spare ring breaks FN8's one-spare chain).
/// 2. **Layer marks.** `Landed { layer }` writes the layer's word to its next value after every
///    copy of its segment (the copies since the batch's previous mark: the layer's promotions AND
///    demotions -- a demoted record is read from its RAM slot by the next step) and after the
///    layer's previous mark (the word only grows: a consumer's `>=` wait never sees it fall).
/// 3. **The all mark** ([`Self::all`]): every op still open (marks and unmarked copies) -> the ALL
///    word; what the staging ring's copies and the deferred law's publication wait on.
///
/// Laws 1 + 2 are what the one stream gave by call order; nothing else is ordered. The planner is
/// pure (no device): the real lane ([`HipCopyQueue`] over `saber_hsa::SdmaLane`) and the host
/// model ([`FakeQueue::signal_lanes`]: jobs run in ANY order their dependencies allow) share it.
#[derive(Debug)]
pub struct SignalPlanner {
    next: u64,
    vram: std::collections::HashMap<u32, u64>,
    ram: std::collections::HashMap<u32, u64>,
    marks: std::collections::HashMap<u32, (u64, u32)>,
    all_mark: (u64, u32),
    /// Marks and unmarked copies not yet known to have landed (an ALL mark / a drain waits these).
    open: Vec<u64>,
    /// Dependencies planned.
    pub deps: u64,
    /// A CONTROL (tests only): plan no slot dependency (law 1 off).
    pub no_slot_deps: bool,
}

impl Default for SignalPlanner {
    fn default() -> SignalPlanner {
        SignalPlanner {
            next: 1,
            vram: std::collections::HashMap::new(),
            ram: std::collections::HashMap::new(),
            marks: std::collections::HashMap::new(),
            all_mark: (0, 0),
            open: Vec::new(),
            deps: 0,
            no_slot_deps: false,
        }
    }
}

impl SignalPlanner {
    fn take(&mut self) -> u64 {
        let t = self.next;
        self.next += 1;
        t
    }

    /// The next ticket the planner will hand out.
    #[must_use]
    pub fn next_ticket(&self) -> u64 {
        self.next
    }

    /// Drop the open tickets `landed` says have landed.
    pub fn prune(&mut self, landed: &dyn Fn(u64) -> bool) {
        self.open.retain(|&t| !landed(t));
    }

    /// The open tickets (marks + unmarked copies in flight, as far as the last prune knows).
    #[must_use]
    pub fn open(&self) -> &[u64] {
        &self.open
    }

    /// Layer `l`'s latest mark: (ticket, value).
    #[must_use]
    pub fn layer_mark(&self, l: u32) -> Option<(u64, u32)> {
        self.marks.get(&l).copied()
    }

    fn deps_of(&mut self, cand: &[u64], landed: &dyn Fn(u64) -> bool) -> Vec<u64> {
        let mut d: Vec<u64> = cand
            .iter()
            .copied()
            .filter(|&t| t != 0 && !landed(t))
            .collect();
        d.sort_unstable();
        d.dedup();
        self.deps += d.len() as u64;
        d
    }

    /// The jobs of one batch (`ops` in call order), appended to `out` in ticket order.
    pub fn plan(&mut self, ops: &[CopyOp], landed: &dyn Fn(u64) -> bool, out: &mut Vec<SigJob>) {
        self.prune(landed);
        let mut seg: Vec<u64> = Vec::new();
        for &op in ops {
            match op {
                CopyOp::H2d { ram, vram } | CopyOp::D2h { vram, ram } => {
                    let t = self.take();
                    let cand = if self.no_slot_deps {
                        Vec::new()
                    } else {
                        vec![
                            self.vram.get(&vram).copied().unwrap_or(0),
                            self.ram.get(&ram).copied().unwrap_or(0),
                        ]
                    };
                    let deps = self.deps_of(&cand, landed);
                    self.vram.insert(vram, t);
                    self.ram.insert(ram, t);
                    seg.push(t);
                    out.push(SigJob {
                        ticket: t,
                        what: SigWhat::Copy(op),
                        deps,
                    });
                }
                CopyOp::Landed { layer } => {
                    let t = self.take();
                    let (prev, v) = self.marks.get(&layer).copied().unwrap_or((0, 0));
                    let value = match v.wrapping_add(1) {
                        0 => 1,
                        x => x,
                    };
                    seg.push(prev);
                    let deps = self.deps_of(&seg, landed);
                    seg.clear();
                    self.marks.insert(layer, (t, value));
                    // the new mark covers the layer's previous one (its dependency)
                    self.open.retain(|&o| o != prev);
                    self.open.push(t);
                    out.push(SigJob {
                        ticket: t,
                        what: SigWhat::Mark {
                            word: MarkWord::Layer(layer),
                            value,
                        },
                        deps,
                    });
                }
            }
        }
        // copies no mark covers (a batch flushed mid-layer): open until they land
        self.open.extend(seg);
    }

    /// Law 3: an ALL mark over every open ticket (and the previous ALL mark), or `None` when
    /// nothing is open (every issued op has landed: the ALL word needs no wait).
    pub fn all(&mut self, landed: &dyn Fn(u64) -> bool) -> Option<SigJob> {
        self.prune(landed);
        if self.open.is_empty() {
            return None;
        }
        let t = self.take();
        let mut cand = std::mem::take(&mut self.open);
        cand.push(self.all_mark.0);
        let deps = self.deps_of(&cand, landed);
        let value = match self.all_mark.1.wrapping_add(1) {
            0 => 1,
            x => x,
        };
        self.all_mark = (t, value);
        self.open.push(t);
        Some(SigJob {
            ticket: t,
            what: SigWhat::Mark {
                word: MarkWord::All,
                value,
            },
            deps,
        })
    }
}

/// The device side of a [`DeviceMover`].
pub trait CopyQueue {
    /// What reads the slots (the compute stream).
    type Consumer: ?Sized;
    /// Issue `ops` on the one copy stream, in order, without waiting.
    fn issue(&mut self, ops: &[CopyOp]) -> Result<(), TierError>;
    /// The host waits for every issued op.
    fn drain(&mut self) -> Result<(), TierError>;
    /// The pinned RAM slab (`(ram_slots + 1) · stride`), for host writes after [`Self::drain`].
    fn ram_mut(&mut self) -> &mut [u8];
    /// FN36 (law 4b): the slab for a host write into `slots` WITHOUT a drain -- the caller
    /// guarantees no issued op reads or writes them (the fake queue asserts it).
    fn ram_mut_disjoint(&mut self, _slots: &[u32]) -> &mut [u8] {
        self.ram_mut()
    }
    /// FN53b: every host region of the RAM slots as `(first slot, bytes)` -- the slab, then the
    /// valve band's open segments and the bounce (their own allocations). Default: the slab.
    fn ram_regions_mut(&mut self) -> Vec<(u32, &mut [u8])> {
        vec![(0, self.ram_mut())]
    }
    /// FN71c (`--xtier-nvme-p2p on`): serve the reads whose slot lives in the card's P2PDMA pool
    /// (the valve's bounce) drive -> pool, no host DRAM; returns the reads it did NOT serve (the
    /// host route reads them). Called after the drain (law 4). Default: none served.
    fn pool_reads(
        &mut self,
        _xt: &XtierFile,
        reads: &[(u64, u32)],
        _qd: usize,
    ) -> Result<Vec<(u64, u32)>, TierError> {
        Ok(reads.to_vec())
    }
    /// `c` waits (on the device) for `layer`'s latest [`CopyOp::Landed`].
    fn consumer_waits(&mut self, layer: u32, c: &Self::Consumer) -> Result<(), TierError>;
    /// The copy stream waits (on the device) for everything `c` has been given so far.
    fn after(&mut self, c: &Self::Consumer) -> Result<(), TierError>;
    /// FN17: `c` waits (on the device) for every op issued so far.
    fn consumer_waits_all(&mut self, c: &Self::Consumer) -> Result<(), TierError>;
    /// FN98 (`--fn-graph-form one`): `layer`'s landed-mark word, the value its latest mark
    /// writes and that mark's position in the queue's (in-order) mark sequence `(device address,
    /// value, seq)`, for a wait captured INSIDE a graph: `None` = this queue has no such word
    /// (only the one-stream HIP mover under the word law has one).
    fn mark_word(&self, _layer: u32) -> Option<(saber_hip::DeviceAddr, u32, u64)> {
        None
    }
}

/// What the mover did (the which-path counters of the copy path).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MoverCounts {
    /// H2D copies issued (promotions).
    pub h2d: u64,
    /// D2H copies issued (demotions).
    pub d2h: u64,
    /// Records read from NVMe into RAM slots (the host route: drive -> pinned RAM).
    pub nvme_reads: u64,
    /// FN71c: records read from NVMe into the card's P2PDMA pool (drive -> VRAM, no host DRAM).
    pub nvme_pool_reads: u64,
    /// Batches issued to the copy stream.
    pub batches: u64,
    /// Layer marks recorded.
    pub landed: u64,
    /// Gates that made the consumer wait (copies were in flight for the layer).
    pub gate_waits: u64,
    /// Gates that did not (nothing issued for the layer since its last gate).
    pub gate_free: u64,
    /// Host drains (before NVMe writes, or asked for).
    pub drains: u64,
    /// Consumer fences (`after`).
    pub fences: u64,
    /// FN17: consumer waits for every issued copy at a fence (the deferred law's publication).
    pub publish_waits: u64,
    /// FN30c: held batches released (the copy stream re-fenced behind the consumer's later work).
    pub releases: u64,
}

/// The engine's expert mover over a [`CopyQueue`] (see the module docs).
#[derive(Debug)]
pub struct DeviceMover<'x, Q: CopyQueue> {
    xt: &'x XtierFile,
    q: Q,
    stride: usize,
    qd: usize,
    pending: Vec<CopyOp>,
    /// Copies were issued for the layer since its last gate.
    dirty: Vec<bool>,
    /// The consumer's current step has been fenced; copies may be queued.
    fenced: bool,
    /// FN17: the deferred law is armed (the fence publishes every issued copy).
    defer: bool,
    /// FN17: the law of the copies being queued (set by the tier per `end_step`).
    deferred: bool,
    /// FN30c: the HOLD law armed for the current step (see [`Self::set_hold`]).
    hold: bool,
    /// The counters.
    pub counts: MoverCounts,
}

impl<'x, Q: CopyQueue> DeviceMover<'x, Q> {
    /// A mover over `q` for `layers` layers, reading NVMe with up to `qd` threads.
    pub fn new(xt: &'x XtierFile, q: Q, qd: usize) -> DeviceMover<'x, Q> {
        let layers = xt.header().layers as usize;
        DeviceMover {
            stride: xt.header().stride as usize,
            xt,
            q,
            qd: qd.max(1),
            pending: Vec::new(),
            dirty: vec![false; layers],
            fenced: false,
            defer: false,
            deferred: false,
            hold: false,
            counts: MoverCounts::default(),
        }
    }

    /// FN17: arm the deferred publication law (the tier must carry `defer_spares`): every fence
    /// also makes the consumer wait for every copy issued so far.
    #[must_use]
    pub fn with_defer(mut self, on: bool) -> Self {
        self.defer = on;
        self
    }

    /// FN17: whether the deferred law is armed.
    #[must_use]
    pub fn defers(&self) -> bool {
        self.defer
    }

    /// FN30c: the HOLD law for the step about to end (call before `end_step`). While held, the
    /// step's copies (promotions, victims, the per-layer landed events) are QUEUED on the host, not
    /// issued; [`Self::release`] then re-fences the copy stream behind the consumer's work so far
    /// (e.g. the MTP drafter round enqueued after the step) and issues them, so no copy overlaps
    /// that work on the device. A held batch is released at the latest by the next [`Self::gate`],
    /// [`Self::after`] or [`Self::drain`], so every consumer read still waits for its layer's
    /// copies (law 3) and every copy still runs after the step it was fenced behind (law 1): the
    /// bytes at every read are unchanged, only the copies' start moves later.
    pub fn set_hold(&mut self, on: bool) {
        self.hold = on;
    }

    /// FN30c: whether copies are held (queued, not issued).
    #[must_use]
    pub fn held(&self) -> bool {
        self.hold && !self.pending.is_empty()
    }

    /// FN30c: release a held batch: the copy stream waits for the consumer's work so far (a new
    /// fence), then the held copies are issued in their order. Returns whether a batch was held.
    /// The hold is disarmed.
    pub fn release(&mut self, c: &Q::Consumer) -> Result<bool, TierError> {
        let was = self.hold;
        self.hold = false;
        if !was || self.pending.is_empty() {
            return Ok(false);
        }
        self.q.after(c)?;
        self.issue_pending()?;
        self.counts.releases += 1;
        Ok(true)
    }

    /// FN42b: the tier file the mover reads (the placement byte law's reference).
    #[must_use]
    pub fn xtier(&self) -> &'x XtierFile {
        self.xt
    }

    /// The queue.
    pub fn queue(&self) -> &Q {
        &self.q
    }
    /// The queue, mutably.
    pub fn queue_mut(&mut self) -> &mut Q {
        &mut self.q
    }

    /// Law 1: the copy stream waits for the consumer's work so far (call at the end of a step's
    /// launches, before `end_step`).
    pub fn after(&mut self, c: &Q::Consumer) -> Result<(), TierError> {
        // FN30c: a batch still held from an earlier step goes first; this step's hold (armed by
        // `set_hold` before its fence) stays armed
        if self.held() {
            self.release(c)?;
            self.hold = true;
        }
        if self.defer {
            // FN17: the copies issued at the LAST end_step are published at this one: the
            // consumer's next reads come after them (a step later, they have landed)
            self.issue_pending()?;
            self.q.consumer_waits_all(c)?;
            self.counts.publish_waits += 1;
        }
        self.q.after(c)?;
        self.fenced = true;
        self.counts.fences += 1;
        Ok(())
    }

    /// Law 3: before `layer`'s launches read its placements. Returns whether the consumer waits.
    pub fn gate(&mut self, layer: u32, c: &Q::Consumer) -> Result<bool, TierError> {
        // FN30c: a held batch is issued before the first read of the next step
        if self.held() {
            self.release(c)?;
        }
        // A new step's reads have begun: its copies need a new fence.
        self.fenced = false;
        let d = self
            .dirty
            .get_mut(layer as usize)
            .ok_or_else(|| TierError::Mover(format!("gate of layer {layer}: no such layer")))?;
        if *d {
            *d = false;
            self.q.consumer_waits(layer, c)?;
            self.counts.gate_waits += 1;
            Ok(true)
        } else {
            self.counts.gate_free += 1;
            Ok(false)
        }
    }

    /// Whether copies for `layer` are in flight past its last gate (a consumer reading the layer
    /// now without [`Self::gate`] could read a slot mid-copy).
    #[must_use]
    pub fn in_flight(&self, layer: u32) -> bool {
        self.dirty.get(layer as usize).copied().unwrap_or(false)
    }

    /// FN98 (`--fn-graph-form one`): law 3 for a gate captured INSIDE the step's graph -- the
    /// same bookkeeping as [`Self::gate`] (a held batch released first, the fence reopened, the
    /// layer's dirty bit cleared and counted), but instead of queueing the wait it returns the
    /// `(word address, value, seq)` the captured wait node must hold for this launch (`None` = no
    /// copy in flight for the layer: any value it held before has landed). Refuses a queue with
    /// no mark word (only the one-stream word law captures).
    pub fn gate_mark(
        &mut self,
        layer: u32,
        c: &Q::Consumer,
    ) -> Result<Option<(saber_hip::DeviceAddr, u32, u64)>, TierError> {
        if self.held() {
            self.release(c)?;
        }
        self.fenced = false;
        let d = self
            .dirty
            .get_mut(layer as usize)
            .ok_or_else(|| TierError::Mover(format!("gate of layer {layer}: no such layer")))?;
        if !*d {
            self.counts.gate_free += 1;
            return Ok(None);
        }
        *d = false;
        self.counts.gate_waits += 1;
        self.q.mark_word(layer).map(Some).ok_or_else(|| {
            TierError::Mover(format!(
                "gate of layer {layer}: this mover has no mark word (FN98's in-graph gate needs \
                 the one-stream mover under --xtier-mover-marks word)"
            ))
        })
    }

    /// FN98: the device address of `layer`'s mark word (`None` without the word law).
    #[must_use]
    pub fn mark_word_addr(&self, layer: u32) -> Option<saber_hip::DeviceAddr> {
        self.q.mark_word(layer).map(|w| w.0)
    }

    /// FN53c: the device miss flag's reads -- `(record, bounce slot)` drive -> the bounce (the
    /// P2PDMA pool's slots drive -> VRAM, the rest into their pinned region) WITHOUT draining the
    /// copy stream: the step's compute stream is parked in `fn_xmiss` while this runs, and a
    /// drain could wait on a copy ordered after it. Sound because no queued copy reads or writes
    /// a bounce slot under the miss flag: misses never enter the L2 (no promotion's source is a
    /// bounce slot) and the engine drains once before the first miss-flag step after a prefill
    /// pass (whose bounced records may have been promoted). Complete on return (the pool route's
    /// host fence included).
    pub fn nvme_to_bounce(&mut self, reads: &[(u64, u32)]) -> Result<(), TierError> {
        let (xt, st, qd) = (self.xt, self.stride, self.qd);
        let n0 = reads.len();
        let reads = self.q.pool_reads(xt, reads, qd)?;
        self.counts.nvme_pool_reads += (n0 - reads.len()) as u64;
        let mut done = 0usize;
        for (first, buf) in self.q.ram_regions_mut() {
            let n = (buf.len() / st) as u32;
            let sub: Vec<(u64, u32)> = reads
                .iter()
                .filter(|r| r.1 >= first && r.1 - first < n)
                .map(|&(i, s)| (i, s - first))
                .collect();
            if !sub.is_empty() {
                crate::expert_mover::parallel_reads(xt, buf, st, &sub, qd)?;
                done += sub.len();
            }
        }
        if done != reads.len() {
            return Err(TierError::Mover(format!(
                "{} of {} miss reads name a slot no RAM region holds",
                reads.len() - done,
                reads.len()
            )));
        }
        self.counts.nvme_reads += reads.len() as u64;
        Ok(())
    }

    /// The host waits for every issued copy.
    pub fn drain(&mut self) -> Result<(), TierError> {
        self.hold = false;
        self.issue_pending()?;
        self.q.drain()?;
        self.counts.drains += 1;
        Ok(())
    }

    fn issue_pending(&mut self) -> Result<(), TierError> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let ops = std::mem::take(&mut self.pending);
        self.q.issue(&ops)?;
        self.counts.batches += 1;
        Ok(())
    }

    fn check_fenced(&self, what: &str) -> Result<(), TierError> {
        if self.fenced {
            Ok(())
        } else {
            Err(TierError::Mover(format!(
                "{what} queued before the consumer's step was fenced: call DeviceMover::after \
                 (the ordering law, rule 1) before end_step"
            )))
        }
    }
}

impl<Q: CopyQueue> ExpertMover for DeviceMover<'_, Q> {
    fn nvme_to_ram(&mut self, reads: &[(u64, u32)]) -> Result<(), TierError> {
        let reads_len = reads.len();
        // Law 4: nothing on the copy stream may still read or write a slot the host writes.
        self.drain()?;
        let (xt, st, qd) = (self.xt, self.stride, self.qd);
        // FN71c: the pool's slots first (drive -> VRAM), the rest by the host route
        let reads = self.q.pool_reads(xt, reads, qd)?;
        let pooled = reads_len - reads.len();
        self.counts.nvme_pool_reads += pooled as u64;
        // FN53b: each read lands in the region holding its slot (one region without a valve)
        let mut done = 0usize;
        for (first, buf) in self.q.ram_regions_mut() {
            let n = (buf.len() / st) as u32;
            let sub: Vec<(u64, u32)> = reads
                .iter()
                .filter(|r| r.1 >= first && r.1 - first < n)
                .map(|&(i, s)| (i, s - first))
                .collect();
            if !sub.is_empty() {
                crate::expert_mover::parallel_reads(xt, buf, st, &sub, qd)?;
                done += sub.len();
            }
        }
        if done != reads.len() {
            return Err(TierError::Mover(format!(
                "{} of {} reads name a slot no RAM region holds",
                reads.len() - done,
                reads.len()
            )));
        }
        self.counts.nvme_reads += reads.len() as u64;
        Ok(())
    }
    fn nvme_refill(&mut self, reads: &[(u64, u32)]) -> Result<(), TierError> {
        // FN36 (law 4b): the refill pool's slots are pointed at by no placement and read or
        // written by no issued copy (a promoted record's old slot re-enters the pool only at the
        // second end_step after its H2D, after a whole step read past its gate), so the host
        // writes them WITHOUT draining the copy stream, under the step's device work
        let slots: Vec<u32> = reads.iter().map(|r| r.1).collect();
        crate::expert_mover::parallel_reads(
            self.xt,
            self.q.ram_mut_disjoint(&slots),
            self.stride,
            reads,
            self.qd,
        )?;
        self.counts.nvme_reads += reads.len() as u64;
        Ok(())
    }
    fn ram_to_vram(&mut self, ram: u32, vram: u32) -> Result<(), TierError> {
        self.check_fenced("a promotion")?;
        self.pending.push(CopyOp::H2d { ram, vram });
        self.counts.h2d += 1;
        Ok(())
    }
    fn vram_to_ram(&mut self, vram: u32, ram: u32) -> Result<(), TierError> {
        self.check_fenced("a demotion")?;
        self.pending.push(CopyOp::D2h { vram, ram });
        self.counts.d2h += 1;
        Ok(())
    }
    fn end_layer(&mut self, layer: u32) -> Result<(), TierError> {
        // no copy queued for this layer (FN30c: a held queue ends with the last layer's marker)
        if matches!(self.pending.last(), None | Some(CopyOp::Landed { .. })) {
            return Ok(());
        }
        self.pending.push(CopyOp::Landed { layer });
        if !self.hold {
            self.issue_pending()?;
        }
        // FN17: a deferred copy is read only after its publication (the fence's wait): the
        // layer's gate stays free
        if !self.deferred {
            if let Some(d) = self.dirty.get_mut(layer as usize) {
                *d = true;
            }
        }
        self.counts.landed += 1;
        Ok(())
    }
    fn flush(&mut self) -> Result<(), TierError> {
        // Asynchronous: every queued copy is ISSUED (end_layer already did, per layer); device
        // consumers order themselves by `gate`, host writers by `drain`. FN30c: a held batch
        // waits for `release` (or the next gate / fence / drain).
        if self.hold {
            return Ok(());
        }
        self.issue_pending()
    }
    fn set_deferred(&mut self, on: bool) {
        self.deferred = on && self.defer;
    }
}

/// One logged event of the [`FakeQueue`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FakeEvent {
    /// An op was issued.
    Issue(CopyOp),
    /// A consumer waited on a layer's mark.
    Wait(u32),
    /// The copy stream was fenced behind the consumer.
    After,
    /// The host drained.
    Drain,
    /// FN17: a consumer waited for every issued op (the deferred law's fence).
    WaitAll,
}

/// A host model of an asynchronous copy stream (the tests): issued ops sit in flight and are
/// APPLIED only when a consumer waits on a layer mark behind them or the host drains — so a
/// consumer that skips the gate reads the stale bytes a real device could give it.
#[derive(Debug)]
pub struct FakeQueue {
    stride: usize,
    /// The fake VRAM slot arena.
    pub vram: AlignedBuf,
    /// The RAM slab (`ram_slots + 1` slots).
    pub ram: AlignedBuf,
    inflight: VecDeque<CopyOp>,
    /// Everything that happened, in order.
    pub log: Vec<FakeEvent>,
    /// FN17: apply every op AT ISSUE (the other extreme of an asynchronous device: a copy lands
    /// at once, so a copy that writes a slot a reader still reads is caught).
    pub eager: bool,
    /// FN54b: the two-lane model ([`Self::two_queue`]); `None` = the one in-order stream.
    pub lanes: Option<FakeLanes>,
    /// FN63b: the signal-lane model ([`Self::signal_lanes`]); `None` = not armed.
    pub sig: Option<FakeSig>,
    /// FN71c: the P2PDMA pool model ([`Self::with_pool`]): RAM slot ids `lo .. lo + n` live in
    /// this buffer instead of the slab (`None` = no pool).
    pub pool: Option<(u32, u32, AlignedBuf)>,
}

/// FN63b: the [`FakeQueue`]'s model of the HSA copy lane: the [`SignalPlanner`]'s jobs run in ANY
/// order their dependencies allow (a seeded choice among the runnable jobs, across both engines
/// and out of issue order -- stronger than two in-order SDMA rings), copies applied as they run,
/// a mark sets its word (so an unordered pair of marks could make a word FALL: the `>=` wait
/// would then hang, which the model reports).
#[derive(Debug, Default)]
pub struct FakeSig {
    /// The signal law.
    pub planner: SignalPlanner,
    pending: Vec<SigJob>,
    done: std::collections::HashSet<u64>,
    words: std::collections::HashMap<MarkWord, u32>,
    rng: u64,
    /// Demotions run while a promotion issued before them was still pending (the directions
    /// overlapped).
    pub overlaps: u64,
}

/// FN54b: the [`FakeQueue`]'s two-lane model: the planner's jobs sit on two FIFOs and the model
/// advances a pseudo-randomly chosen unblocked lane (a wait blocks until the other lane's mark ran),
/// so any interleaving the slot law allows can happen.
#[derive(Debug, Default)]
pub struct FakeLanes {
    /// The slot law.
    pub planner: LanePlanner,
    lanes: [VecDeque<(u64, LaneOp)>; 2],
    done: [u64; 2],
    rng: u64,
    calls: u64,
    landed_issued: std::collections::HashMap<u32, u64>,
    landed_done: std::collections::HashMap<u32, u64>,
    /// Demotions applied while a promotion issued BEFORE them was still queued (the two
    /// directions overlapped; 0 on one stream).
    pub overlaps: u64,
}

impl FakeQueue {
    /// `vram_slots` VRAM slots and `ram_slots + 1` RAM slots of `stride` bytes.
    #[must_use]
    pub fn new(stride: u64, vram_slots: u32, ram_slots: u32) -> FakeQueue {
        let s = stride as usize;
        FakeQueue {
            stride: s,
            vram: AlignedBuf::zeroed(s * vram_slots as usize),
            ram: AlignedBuf::zeroed(s * (ram_slots as usize + 1)),
            inflight: VecDeque::new(),
            log: Vec::new(),
            eager: false,
            lanes: None,
            sig: None,
            pool: None,
        }
    }

    /// FN71c: model the pool bounce -- RAM slot ids `lo .. lo + n` are read drive -> "pool" (a
    /// buffer of their own) by [`CopyQueue::pool_reads`], never through the slab.
    #[must_use]
    pub fn with_pool(mut self, lo: u32, n: u32) -> FakeQueue {
        self.pool = Some((lo, n, AlignedBuf::zeroed(self.stride * n as usize)));
        self
    }

    /// FN71c: the bytes of pool slot `lo + k`.
    #[must_use]
    pub fn pool_slot(&self, k: u32) -> &[u8] {
        let p = &self.pool.as_ref().expect("with_pool").2;
        &p[k as usize * self.stride..(k as usize + 1) * self.stride]
    }

    /// FN63b: model the HSA signal lane (jobs run in a `seed`-driven order their dependencies
    /// allow).
    #[must_use]
    pub fn signal_lanes(mut self, seed: u64) -> FakeQueue {
        self.sig = Some(FakeSig {
            rng: seed | 1,
            ..FakeSig::default()
        });
        self
    }

    /// FN63b: run one runnable job of the signal lane; false when none is pending.
    fn sig_step(&mut self) -> bool {
        let eager = self.eager;
        let Some(fs) = self.sig.as_mut() else {
            return false;
        };
        if fs.pending.is_empty() {
            return false;
        }
        let runnable: Vec<usize> = (0..fs.pending.len())
            .filter(|&i| fs.pending[i].deps.iter().all(|d| fs.done.contains(d)))
            .collect();
        assert!(
            !runnable.is_empty(),
            "the signal lane deadlocked (a dependency on a job never issued)"
        );
        let i = if eager {
            runnable[0]
        } else {
            fs.rng = fs
                .rng
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            runnable[((fs.rng >> 33) as usize) % runnable.len()]
        };
        let job = fs.pending.remove(i);
        fs.done.insert(job.ticket);
        let mut apply = None;
        match job.what {
            SigWhat::Copy(op) => {
                if matches!(op, CopyOp::D2h { .. })
                    && fs.pending.iter().any(|j| {
                        j.ticket < job.ticket && matches!(j.what, SigWhat::Copy(CopyOp::H2d { .. }))
                    })
                {
                    fs.overlaps += 1;
                }
                apply = Some(op);
            }
            SigWhat::Mark { word, value } => {
                fs.words.insert(word, value);
            }
        }
        if let Some(op) = apply {
            self.apply(op);
        }
        true
    }

    /// FN63b: run jobs until `word >= value` (the consumer's CP wait).
    fn sig_wait_word(&mut self, word: MarkWord, value: u32) {
        loop {
            let fs = self.sig.as_ref().expect("signal lane");
            if fs.words.get(&word).copied().unwrap_or(0) >= value {
                return;
            }
            assert!(
                self.sig_step(),
                "{word:?} never reached {value} (a mark fell back or never ran)"
            );
        }
    }

    /// FN54b: model the two-queue mover (lanes advanced in a `seed`-driven order).
    #[must_use]
    pub fn two_queue(mut self, seed: u64) -> FakeQueue {
        self.lanes = Some(FakeLanes {
            rng: seed | 1,
            ..FakeLanes::default()
        });
        self
    }

    /// FN54b: advance one unblocked lane by one job; false when nothing can run.
    fn lane_step(&mut self) -> bool {
        let Some(fl) = self.lanes.as_mut() else {
            return false;
        };
        let runnable = |fl: &FakeLanes, i: usize| match fl.lanes[i].front() {
            None => false,
            Some((_, LaneOp::Wait { seq, .. })) => fl.done[1 - i] >= *seq,
            Some(_) => true,
        };
        let (r0, r1) = (runnable(fl, 0), runnable(fl, 1));
        let i = match (r0, r1) {
            (false, false) => {
                assert!(
                    fl.lanes[0].is_empty() && fl.lanes[1].is_empty(),
                    "the two lanes deadlocked (a wait on a mark never recorded)"
                );
                return false;
            }
            (true, false) => 0,
            (false, true) => 1,
            (true, true) => {
                fl.rng = fl
                    .rng
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((fl.rng >> 33) & 1) as usize
            }
        };
        let (call, job) = fl.lanes[i].pop_front().expect("runnable");
        let mut apply = None;
        match job {
            LaneOp::Mark { seq, .. } => fl.done[i] = seq,
            LaneOp::Wait { .. } => {}
            LaneOp::Op { op, .. } => {
                if let CopyOp::Landed { layer } = op {
                    *fl.landed_done.entry(layer).or_default() += 1;
                } else {
                    if matches!(op, CopyOp::D2h { .. })
                        && fl.lanes[0].iter().any(|(c, j)| {
                            *c < call
                                && matches!(
                                    j,
                                    LaneOp::Op {
                                        op: CopyOp::H2d { .. },
                                        ..
                                    }
                                )
                        })
                    {
                        fl.overlaps += 1;
                    }
                    apply = Some(op);
                }
            }
        }
        if let Some(op) = apply {
            self.apply(op);
        }
        true
    }

    fn lanes_run_all(&mut self) {
        while self.lane_step() {}
        while self.sig_step() {}
    }

    /// The bytes a consumer reads from VRAM slot `s` NOW.
    #[must_use]
    pub fn vram_slot(&self, s: u32) -> &[u8] {
        &self.vram[s as usize * self.stride..(s as usize + 1) * self.stride]
    }
    /// The bytes of RAM slot `s` NOW.
    #[must_use]
    pub fn ram_slot(&self, s: u32) -> &[u8] {
        &self.ram[s as usize * self.stride..(s as usize + 1) * self.stride]
    }
    /// FN42b: the lendable memory's OWNER writes VRAM slot `s` (the prefill arena of a wide
    /// chunk scribbling over a reclaimed slot): every byte becomes `byte`.
    pub fn scribble(&mut self, s: u32, byte: u8) {
        let st = self.stride;
        self.vram[s as usize * st..(s as usize + 1) * st].fill(byte);
    }
    /// Ops issued and not yet applied.
    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.inflight.len()
            + self.lanes.as_ref().map_or(0, |fl| {
                fl.lanes
                    .iter()
                    .flatten()
                    .filter(|(_, j)| matches!(j, LaneOp::Op { .. }))
                    .count()
            })
            + self.sig.as_ref().map_or(0, |fs| {
                fs.pending
                    .iter()
                    .filter(|j| matches!(j.what, SigWhat::Copy(_)))
                    .count()
            })
    }
    fn apply(&mut self, op: CopyOp) {
        let s = self.stride;
        match op {
            CopyOp::H2d { ram, vram } => {
                let (r, v) = (ram as usize * s, vram as usize * s);
                self.vram[v..v + s].copy_from_slice(&self.ram[r..r + s]);
            }
            CopyOp::D2h { vram, ram } => {
                let (r, v) = (ram as usize * s, vram as usize * s);
                self.ram[r..r + s].copy_from_slice(&self.vram[v..v + s]);
            }
            CopyOp::Landed { .. } => {}
        }
    }
    /// Realise a warm plan directly (the boot's reads, not the step path).
    pub fn warm(
        &mut self,
        xt: &XtierFile,
        plan: &[(u64, crate::expert_tier::Placement)],
    ) -> Result<(), TierError> {
        use crate::expert_tier::Placement;
        let s = self.stride;
        for &(i, p) in plan {
            match p {
                Placement::Vram(v) => {
                    xt.read_record(i, &mut self.vram[v as usize * s..(v as usize + 1) * s])?;
                }
                Placement::Ram(r) => {
                    xt.read_record(i, &mut self.ram[r as usize * s..(r as usize + 1) * s])?;
                }
                Placement::Fetch => {}
            }
        }
        Ok(())
    }
}

impl CopyQueue for FakeQueue {
    type Consumer = ();
    fn issue(&mut self, ops: &[CopyOp]) -> Result<(), TierError> {
        if let Some(fs) = self.sig.as_mut() {
            let mut jobs = Vec::new();
            let done = &fs.done;
            fs.planner.plan(ops, &|t| done.contains(&t), &mut jobs);
            fs.pending.extend(jobs);
            for &op in ops {
                self.log.push(FakeEvent::Issue(op));
            }
            if self.eager {
                self.lanes_run_all();
            }
            return Ok(());
        }
        if let Some(fl) = self.lanes.as_mut() {
            let mut jobs = Vec::new();
            fl.planner.plan(ops, &mut jobs);
            let mut k = 0;
            for j in jobs {
                let lane = match j {
                    LaneOp::Op { lane, op } => {
                        if let CopyOp::Landed { layer } = op {
                            *fl.landed_issued.entry(layer).or_default() += 1;
                        }
                        k += 1;
                        lane
                    }
                    LaneOp::Mark { lane, .. } | LaneOp::Wait { lane, .. } => lane,
                };
                let c = fl.calls + k;
                fl.lanes[lane as usize].push_back((c, j));
            }
            fl.calls += k;
            for &op in ops {
                self.log.push(FakeEvent::Issue(op));
            }
            if self.eager {
                self.lanes_run_all();
            }
            return Ok(());
        }
        for &op in ops {
            if self.eager {
                self.apply(op);
            } else {
                self.inflight.push_back(op);
            }
            self.log.push(FakeEvent::Issue(op));
        }
        Ok(())
    }
    fn drain(&mut self) -> Result<(), TierError> {
        self.lanes_run_all();
        while let Some(op) = self.inflight.pop_front() {
            self.apply(op);
        }
        self.log.push(FakeEvent::Drain);
        Ok(())
    }
    fn pool_reads(
        &mut self,
        xt: &XtierFile,
        reads: &[(u64, u32)],
        _qd: usize,
    ) -> Result<Vec<(u64, u32)>, TierError> {
        let st = self.stride;
        let Some((lo, n, buf)) = self.pool.as_mut() else {
            return Ok(reads.to_vec());
        };
        let mut rest = Vec::new();
        for &(i, s) in reads {
            if s >= *lo && s - *lo < *n {
                let k = (s - *lo) as usize;
                xt.read_record(i, &mut buf[k * st..(k + 1) * st])?;
            } else {
                rest.push((i, s));
            }
        }
        Ok(rest)
    }
    fn ram_mut(&mut self) -> &mut [u8] {
        assert!(
            self.in_flight() == 0,
            "a host write into the RAM slab with copies in flight (law 4)"
        );
        &mut self.ram
    }
    fn ram_mut_disjoint(&mut self, slots: &[u32]) -> &mut [u8] {
        let laned = self.lanes.iter().flat_map(|fl| {
            fl.lanes.iter().flatten().filter_map(|(_, j)| match j {
                LaneOp::Op { op, .. } => Some(op),
                _ => None,
            })
        });
        let signalled = self.sig.iter().flat_map(|fs| {
            fs.pending.iter().filter_map(|j| match &j.what {
                SigWhat::Copy(op) => Some(op),
                SigWhat::Mark { .. } => None,
            })
        });
        for op in self.inflight.iter().chain(laned).chain(signalled) {
            let touched = match *op {
                CopyOp::H2d { ram, .. } | CopyOp::D2h { ram, .. } => Some(ram),
                _ => None,
            };
            assert!(
                touched.is_none_or(|r| !slots.contains(&r)),
                "a refill into RAM slot {touched:?} with a copy of it in flight (law 4b)"
            );
        }
        &mut self.ram
    }
    fn consumer_waits(&mut self, layer: u32, _c: &()) -> Result<(), TierError> {
        if let Some(fs) = self.sig.as_ref() {
            // FN63b: the CP's `word >= value` wait on the layer's latest mark
            if let Some((_, v)) = fs.planner.layer_mark(layer) {
                self.sig_wait_word(MarkWord::Layer(layer), v);
            }
            self.log.push(FakeEvent::Wait(layer));
            return Ok(());
        }
        if self.lanes.is_some() {
            // FN54b: run lanes (any allowed order) until the layer's latest mark ran
            loop {
                let fl = self.lanes.as_ref().expect("lanes");
                let want = fl.landed_issued.get(&layer).copied().unwrap_or(0);
                if fl.landed_done.get(&layer).copied().unwrap_or(0) >= want {
                    break;
                }
                assert!(self.lane_step(), "layer {layer}'s mark never ran");
            }
            self.log.push(FakeEvent::Wait(layer));
            return Ok(());
        }
        // Apply up to and including the LAST mark of this layer in flight.
        let last = self
            .inflight
            .iter()
            .rposition(|op| *op == CopyOp::Landed { layer });
        if let Some(k) = last {
            for _ in 0..=k {
                let op = self.inflight.pop_front().expect("in flight");
                self.apply(op);
            }
        }
        self.log.push(FakeEvent::Wait(layer));
        Ok(())
    }
    fn after(&mut self, _c: &()) -> Result<(), TierError> {
        self.log.push(FakeEvent::After);
        Ok(())
    }
    fn consumer_waits_all(&mut self, _c: &()) -> Result<(), TierError> {
        if let Some(fs) = self.sig.as_mut() {
            // FN63b law 3: the ALL mark over everything open, then the CP's wait on it
            let done = &fs.done;
            let job = fs.planner.all(&|t| done.contains(&t));
            if let Some(j) = job {
                let SigWhat::Mark { value, .. } = j.what else {
                    unreachable!("the all mark is a mark")
                };
                fs.pending.push(j);
                if self.eager {
                    self.lanes_run_all();
                }
                self.sig_wait_word(MarkWord::All, value);
            }
            self.log.push(FakeEvent::WaitAll);
            return Ok(());
        }
        self.lanes_run_all();
        while let Some(op) = self.inflight.pop_front() {
            self.apply(op);
        }
        self.log.push(FakeEvent::WaitAll);
        Ok(())
    }
}

/// The HIP copy queue: the slot arena, the RAM slab, the one copy stream and one event per layer.
/// Everything is allocated ONCE, here (I1); the tier only changes which record a slot holds.
pub struct HipCopyQueue<'h> {
    /// FN63b: the HSA SDMA copy lane (`--xtier-mover hsa-lane`). Declared FIRST: it drops (and
    /// its `Drop` waits for every copy in flight) before the slot arena, the slab and the lent
    /// memory it copies between (the lane's safety law).
    hsa: Option<std::cell::RefCell<HsaLane>>,
    /// FN71c (`--xtier-nvme-p2p on`): the card's P2PDMA pool reader holding the valve's bounce
    /// (drops early: its lane drains before anything it copied into goes away).
    pool: Option<PoolBounce>,
    /// FN63b: the copy-batch census (`--xtier-mover-census`, either mover).
    census: Option<Census>,
    hip: &'h HipThread,
    stride: u64,
    /// The VRAM slot arena (`vram_slots · stride`, one `hipMalloc`).
    pub vram: DevicePtr,
    /// The pinned RAM slab (`(ram_slots + 1) · stride`, one `hipHostMalloc`).
    pub ram: PinnedBuf,
    /// The slab's device-visible (UVA) address, the tier's `ram_base`.
    pub ram_device_addr: u64,
    /// The one ordered copy stream.
    pub stream: Stream,
    landed: Vec<Event>,
    fence: Event,
    /// FN17: every op issued so far (the deferred law's publication wait).
    all: Event,
    /// FN42b: the LENDABLE slots' memory (another client's allocation, shared): slot
    /// `first + i` lives at byte `offset + i * stride` of it (see [`Self::set_lend`]). FN43: one
    /// region per client (the wide arena's tail, the staging ring), ascending `first`.
    lend: Vec<(std::rc::Rc<DevicePtr>, u64, u32)>,
    /// FN54b: the background lane (`None` = the one in-order stream, today's law).
    bg: Option<BgLane>,
    /// FN59b (`--xtier-plan async`): the one-stream copy batches go to the HIP thread's
    /// background lane ([`HipThread::submit_batch`]) instead of blocking in `run_batch`.
    submit: bool,
    /// FN59b: the submitted batches not yet known to be issued (oldest first).
    tickets: std::collections::VecDeque<saber_hip::BatchTicket>,
    /// FN59b: host ns spent inside [`CopyQueue::issue`] (the census' `te_issue`).
    issue_ns: u64,
    /// FN53b: the valve band's segments + the bounce (`None` = no valve: the slab only).
    valve: Option<ValveMem>,
    /// FN72b (`--xtier-mover-marks`): how a layer's landed mark is signalled (`None` = a HIP
    /// event per layer, today's law).
    marks: Option<MarkWords>,
}

/// FN72b: the landed-mark laws of the one-stream HIP mover (`--xtier-mover-marks`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkLaw {
    /// A HIP event recorded per layer batch (today's law, the control): ROCclr puts a barrier
    /// packet on the copy stream's hardware queue behind every SDMA batch.
    Layer,
    /// The layer's mark = a 4-byte SDMA write (H2D from a pinned ring) of the layer's
    /// fine-grained device word, after its copies on the same stream; the gate =
    /// `hipStreamWaitValue32(word >= mark)` on the consumer. No HIP event on the copy stream.
    Word,
    /// The discriminator: no per-layer mark at all; the first gate after a copy records ONE HIP
    /// event over everything issued so far (a superset: exact, but a layer waits every copy).
    Step,
}

impl std::str::FromStr for MarkLaw {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "layer" => Ok(MarkLaw::Layer),
            "word" => Ok(MarkLaw::Word),
            "step" => Ok(MarkLaw::Step),
            x => Err(format!("{x:?}: one of layer|word|step")),
        }
    }
}

/// FN72b: the pinned value ring's length (u32 words); its quarters are guarded by one HIP event
/// each (one event per `MARK_WORD_RING / 4` marks: the host never rewrites a value an issued
/// copy has not read yet).
pub const MARK_WORD_RING: usize = 65_536;
/// FN72b: the bytes of one mark write. ROCclr runs a pinned H2D of a few bytes as a BLIT KERNEL
/// on the copy stream's hardware queue (h1: a 4-byte mark = `__amd_rocclr_copyBuffer` on the
/// mover's queue behind the SDMA batch = the same parked packet as an event); 20 KiB copies go to
/// the SDMA engine (h1 copy log). The mark copies a 20 KiB window of the ring starting at the
/// value (the bytes past the first word are don't-care) into the layer's own 20 KiB region; the
/// gate reads the region's first word.
pub const MARK_WORD_BYTES: usize = 20_480;

/// FN72b: the word / step mark state.
struct MarkWords {
    law: MarkLaw,
    /// `layers` fine-grained device words (word law).
    words: Option<DevicePtr>,
    /// The pinned value ring (word law).
    ring: Option<PinnedBuf>,
    /// Each layer's latest mark value (word law).
    values: Vec<u32>,
    /// FN98: each layer's latest mark's position in the stream's mark order (`seq` at its write):
    /// the one-stream mover lands its marks in this order, so one wait on the block's LAST
    /// issued mark covers every earlier one.
    seqs: Vec<u64>,
    /// Marks written so far (the ring position).
    seq: u64,
    /// One event per ring quarter, recorded after the quarter's last word copy.
    guards: Vec<Event>,
    /// Whether `guards[q]` has been recorded (the quarter was used once).
    guard_live: [bool; 4],
    /// Step law: one event over everything issued; `step_dirty` = a copy since its last record.
    step_ev: Event,
    step_dirty: bool,
    /// Counters: (word marks, gate waits, guard host waits, step records).
    counts: [u64; 4],
}

/// FN71c: the valve's bounce in the card's P2PDMA pool: slot `bounce_lo + k` at GPU address
/// `gpu + k * stride` (pool offset `k * stride`), read drive -> pool by [`saber_hsa::NvmeVram`].
#[derive(Debug)]
struct PoolBounce {
    nv: saber_hsa::NvmeVram,
    gpu: u64,
    slots: u32,
}

/// FN53b: the valve's pinned memory -- band segment `k` holds slots `band_lo + k * VALVE_SEG ..`
/// (`None` = closed: its allocation freed), the bounce slots `bounce_lo ..` (allocated on the
/// first close). Each `(buffer, UVA address)`.
#[derive(Debug, Default)]
struct ValveMem {
    band_lo: u32,
    segs: Vec<Option<(PinnedBuf, u64)>>,
    bounce_lo: u32,
    bounce: Option<(PinnedBuf, u64)>,
}

/// FN54b: marks per lane in each event ring (a wait on a mark re-recorded later on the SAME lane
/// waits longer, never shorter: conservative, still exact).
const MARK_RING: usize = 256;

/// FN63b: the HSA lane's ticket ring (signals + word sources): ~25-90 b8 steps of copies.
pub const HSA_LANE_RING: usize = 8192;

/// FN63b: the HSA copy lane's state (see [`HipCopyQueue::enable_hsa_lane`]).
struct HsaLane {
    /// The SDMA lane (first: dropped -- drained -- before `words`).
    lane: saber_hsa::SdmaLane,
    /// `layers + 1` fine-grained device words: `[l]` = layer `l`'s latest landed mark, `[layers]`
    /// = the ALL word. Written only by the lane's SDMA word copies; read by `hipStreamWaitValue32`.
    words: DevicePtr,
    layers: u32,
    planner: SignalPlanner,
    /// A fence was recorded (`after`) and not yet checked by an issue.
    fence_pending: bool,
    counts: HsaCounts,
    /// Census batches not yet landed: (issue timestamp ticks, first ticket, last ticket).
    pending: VecDeque<(u64, u64, u64)>,
}

/// FN63b: the HSA lane's which-path counters (rule 5).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HsaCounts {
    /// Batches issued.
    pub batches: u64,
    /// Issues whose fence (the consumer's reads) had already completed (no host wait).
    pub fence_free: u64,
    /// Issues that waited on the host for the fence (the consumer had not reached it).
    pub fence_waits: u64,
    /// Gates that put a `hipStreamWaitValue32` on the consumer stream.
    pub gate_waits: u64,
    /// Gates whose mark had already landed (no wait issued: the word already holds it).
    pub gate_skips: u64,
    /// ALL marks issued (the ring's stage, the deferred law's publication).
    pub all_marks: u64,
    /// ALL waits that found nothing open (no mark, no wait).
    pub all_free: u64,
}

/// FN63b: the copy-batch census of either mover: each batch's device span (first copy start ->
/// last copy end) and, on the HSA lane, its landing time from the issue (issue -> last end).
#[derive(Debug, Default)]
struct Census {
    spans_ms: Vec<f32>,
    lands_ms: Vec<f32>,
    ops: u64,
    batches: u64,
    unread: u64,
    /// The HIP mover's two timing-event pairs (ping-pong) and which hold an unread batch.
    hip_ev: Vec<(Event, Event)>,
    hip_pending: [bool; 2],
    hip_k: usize,
}

fn pct(v: &[f32], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s: Vec<f32> = v.to_vec();
    s.sort_by(f32::total_cmp);
    f64::from(s[((s.len() - 1) as f64 * q).round() as usize])
}

/// FN54b: the two-queue mover's second stream and its events.
struct BgLane {
    stream: Stream,
    planner: LanePlanner,
    /// `marks[lane][seq % MARK_RING]`.
    marks: [Vec<Event>; 2],
    /// The background lane's publication event (FN17's `consumer_waits_all`).
    all: Event,
    /// The background lane's tail (the ring's copies wait on it).
    tail: Event,
}

impl std::fmt::Debug for HipCopyQueue<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HipCopyQueue({} B vram, {} B pinned, {} layer events)",
            self.vram.len(),
            self.ram.len(),
            self.landed.len()
        )
    }
}

impl<'h> HipCopyQueue<'h> {
    /// Allocate the arena, the slab, the stream and the events. `vram_slots` comes from the
    /// device ledger ([`crate::flashnext_budget::Ep2Budget`]), `ram_slots` from the host budget.
    pub fn new(
        hip: &'h HipThread,
        stride: u64,
        vram_slots: u32,
        ram_slots: u32,
        layers: u32,
    ) -> Result<HipCopyQueue<'h>, TierError> {
        let e = |w: &str, err: saber_hip::Error| TierError::Mover(format!("{w}: {err}"));
        let vram = hip
            .malloc(stride * u64::from(vram_slots.max(1)))
            .map_err(|x| e("the VRAM slot arena", x))?;
        let ram = hip
            .host_malloc(stride * (u64::from(ram_slots) + 1))
            .map_err(|x| e("the pinned RAM slab", x))?;
        let ram_device_addr = hip
            .host_device_addr(&ram)
            .map_err(|x| e("hipHostGetDevicePointer", x))?;
        let stream = hip
            .create_stream(saber_hip::StreamFlags::NonBlocking)
            .map_err(|x| e("the copy stream", x))?;
        let mut landed = Vec::with_capacity(layers as usize);
        for _ in 0..layers {
            landed.push(
                hip.create_event(EventFlags::DisableTiming)
                    .map_err(|x| e("a layer event", x))?,
            );
        }
        let fence = hip
            .create_event(EventFlags::DisableTiming)
            .map_err(|x| e("the fence event", x))?;
        let all = hip
            .create_event(EventFlags::DisableTiming)
            .map_err(|x| e("the publication event", x))?;
        Ok(HipCopyQueue {
            hsa: None,
            pool: None,
            census: None,
            hip,
            stride,
            vram,
            ram,
            ram_device_addr,
            stream,
            landed,
            fence,
            all,
            lend: Vec::new(),
            bg: None,
            submit: false,
            tickets: std::collections::VecDeque::new(),
            issue_ns: 0,
            valve: None,
            marks: None,
        })
    }

    /// FN72b (`--xtier-mover-marks word|step`): replace the per-layer HIP event marks (see
    /// [`MarkLaw`]). The one-stream mover only (not two-queue / hsa-lane). Call once, before any
    /// copy. `Layer` leaves today's law.
    pub fn set_marks(&mut self, law: MarkLaw, layers: u32) -> Result<(), TierError> {
        if law == MarkLaw::Layer {
            return Ok(());
        }
        if self.bg.is_some() || self.hsa.is_some() {
            return Err(TierError::Mover(
                "--xtier-mover-marks word|step: the one-stream mover only".into(),
            ));
        }
        let e = |w: &str, err: saber_hip::Error| TierError::Mover(format!("marks: {w}: {err}"));
        let ev = || {
            self.hip
                .create_event(EventFlags::DisableTiming)
                .map_err(|x| e("an event", x))
        };
        let (mut words, mut ring) = (None, None);
        if law == MarkLaw::Word {
            let w = self
                .hip
                .malloc_with_flags(
                    u64::from(layers.max(1)) * MARK_WORD_BYTES as u64,
                    saber_hip::MallocFlags::FineGrained,
                )
                .map_err(|x| e("the mark words", x))?;
            self.hip
                .memset(&w, 0, &self.stream)
                .and_then(|()| self.hip.sync_stream(&self.stream))
                .map_err(|x| e("zero the mark words", x))?;
            words = Some(w);
            ring = Some(
                self.hip
                    .host_malloc((MARK_WORD_RING * 4 + MARK_WORD_BYTES) as u64)
                    .map_err(|x| e("the value ring", x))?,
            );
        }
        let mut guards = Vec::with_capacity(4);
        for _ in 0..4 {
            guards.push(ev()?);
        }
        self.marks = Some(MarkWords {
            law,
            words,
            ring,
            values: vec![0; layers as usize],
            seqs: vec![0; layers as usize],
            seq: 0,
            guards,
            guard_live: [false; 4],
            step_ev: ev()?,
            step_dirty: false,
            counts: [0; 4],
        });
        Ok(())
    }

    /// FN72b: the mark law and its counters `(word marks, gate waits, guard host waits, step
    /// records)`; `Layer` with zeros when no other law is armed.
    #[must_use]
    pub fn marks(&self) -> (MarkLaw, [u64; 4]) {
        self.marks
            .as_ref()
            .map_or((MarkLaw::Layer, [0; 4]), |m| (m.law, m.counts))
    }

    /// FN72b (word law): the ring slots for `ops`' landed marks -- each mark's value is written
    /// into its slot now (a quarter is reused only after its guard event completed). Returns,
    /// per op, `Some((layer, slot, guard quarter to record after it))` for a landed mark.
    #[allow(clippy::type_complexity)]
    fn word_slots(
        &mut self,
        ops: &[CopyOp],
    ) -> Result<Vec<Option<(u32, usize, Option<usize>)>>, TierError> {
        let quarter = MARK_WORD_RING / 4;
        let mut out = Vec::with_capacity(ops.len());
        for op in ops {
            let CopyOp::Landed { layer } = *op else {
                out.push(None);
                continue;
            };
            let m = self.marks.as_ref().expect("marks");
            let slot = (m.seq % MARK_WORD_RING as u64) as usize;
            let q = slot / quarter;
            if slot % quarter == 0 && m.guard_live[q] {
                // the quarter's previous values: their copies must have run before the rewrite
                // (normally long done: one guard per 16k marks)
                self.reap(true)?;
                let m = self.marks.as_ref().expect("marks");
                if !self
                    .hip
                    .query_event(&m.guards[q])
                    .map_err(|x| TierError::Mover(format!("marks guard: {x}")))?
                {
                    self.hip
                        .sync_event(&m.guards[q])
                        .map_err(|x| TierError::Mover(format!("marks guard wait: {x}")))?;
                    self.marks.as_mut().expect("marks").counts[2] += 1;
                }
            }
            let m = self.marks.as_mut().expect("marks");
            let v = m
                .values
                .get_mut(layer as usize)
                .ok_or_else(|| TierError::Mover(format!("no mark word for layer {layer}")))?;
            *v = v
                .checked_add(1)
                .ok_or_else(|| TierError::Mover(format!("layer {layer}'s mark word wrapped")))?;
            let val = *v;
            m.seqs[layer as usize] = m.seq;
            let ring = m.ring.as_mut().expect("word law ring");
            ring.as_mut_slice()[4 * slot..4 * slot + 4].copy_from_slice(&val.to_le_bytes());
            m.seq += 1;
            m.counts[0] += 1;
            let last = (slot + 1) % quarter == 0;
            if last {
                m.guard_live[q] = true;
            }
            out.push(Some((layer, slot, last.then_some(q))));
        }
        Ok(out)
    }

    /// FN72 (`--fn-sdma-lanes split`): move the ordered copy stream onto a HARDWARE QUEUE OF ITS
    /// OWN ([`HipThread::create_stream_own_queue`], every one of `cu_count` CUs in its mask) so no
    /// other stream's packets queue behind its copies' barriers. The same copies, events and order:
    /// only the queue the stream's packets land on changes. Call once, before any copy.
    pub fn own_queue(&mut self, cu_count: u32) -> Result<(), TierError> {
        if self.bg.is_some() || self.hsa.is_some() {
            return Err(TierError::Mover(
                "--fn-sdma-lanes split moves the one-stream mover: not with two-queue / hsa-lane"
                    .into(),
            ));
        }
        self.stream = self
            .hip
            .create_stream_own_queue(cu_count)
            .map_err(|x| TierError::Mover(format!("the copy stream on its own queue: {x}")))?;
        Ok(())
    }

    /// FN54b: arm the two-queue mover: a second (background) copy stream for the demotions, the
    /// cross-lane mark rings and the slot law ([`LanePlanner`]). Call once, before any copy.
    pub fn enable_two_queue(&mut self) -> Result<(), TierError> {
        if self.hsa.is_some() {
            return Err(TierError::Mover(
                "two-queue and the HSA lane are two movers: pick one".into(),
            ));
        }
        let e = |w: &str, err: saber_hip::Error| TierError::Mover(format!("{w}: {err}"));
        let stream = self
            .hip
            .create_stream(saber_hip::StreamFlags::NonBlocking)
            .map_err(|x| e("the background copy stream", x))?;
        let ring = || -> Result<Vec<Event>, TierError> {
            (0..MARK_RING)
                .map(|_| {
                    self.hip
                        .create_event(EventFlags::DisableTiming)
                        .map_err(|x| e("a lane mark", x))
                })
                .collect()
        };
        let marks = [ring()?, ring()?];
        let ev = || {
            self.hip
                .create_event(EventFlags::DisableTiming)
                .map_err(|x| e("a background event", x))
        };
        self.bg = Some(BgLane {
            stream,
            planner: LanePlanner::default(),
            marks,
            all: ev()?,
            tail: ev()?,
        });
        Ok(())
    }

    /// FN63b: arm the HSA SDMA copy lane (`--xtier-mover hsa-lane`, see [`SignalPlanner`]): the
    /// tier's copies go to `saber_hsa::SdmaLane` (ROCr opened beside HIP in this process, the
    /// GPU agent at PCI `bdfid`), H2D and D2H on two SDMA engines (`engines`), ordered only by
    /// the signal law; a layer's landed mark is a 4-byte SDMA write of a fine-grained device word
    /// the consumer's gate waits on with `hipStreamWaitValue32` (no host wait, no extra HIP
    /// stream). The fence stays a HIP event on the consumer stream: an issue checks it (the
    /// engine's route-log read has normally drained the stream already) and waits on the host
    /// only when it has not completed (counted). Boot self-test: one word H2D + D2H round trip
    /// on the chosen engines. Returns the boot line's description. Call once, before any copy.
    pub fn enable_hsa_lane(
        &mut self,
        bdfid: u32,
        engines: saber_hsa::EngineChoice,
        layers: u32,
    ) -> Result<String, TierError> {
        if self.bg.is_some() {
            return Err(TierError::Mover(
                "two-queue and the HSA lane are two movers: pick one".into(),
            ));
        }
        let e = |w: &str, err: String| TierError::Mover(format!("hsa lane: {w}: {err}"));
        let words = self
            .hip
            .malloc_with_flags(
                u64::from(layers + 1) * 4,
                saber_hip::MallocFlags::FineGrained,
            )
            .map_err(|x| e("the mark words", x.to_string()))?;
        let mut lane =
            saber_hsa::SdmaLane::open(saber_hsa::DEFAULT_HSA_SONAME, bdfid, HSA_LANE_RING, engines)
                .map_err(|x| e("open", x.to_string()))?;
        lane.register_device(&self.vram);
        lane.register_host(&self.ram);
        lane.register_device(&words);
        for (i, (p, _, _)) in self.lend.iter().enumerate() {
            if !self.lend[..i].iter().any(|r| std::rc::Rc::ptr_eq(&r.0, p)) {
                lane.register_device(p);
            }
        }
        let all = words.addr().as_u64() + 4 * u64::from(layers);
        let (magic, fallback) = match lane.self_test(all) {
            Ok(m) => (m, String::new()),
            // `auto`'s explicit pair failed (an engine this agent does not have): the runtime's
            // choice, named on the boot line (the which-path)
            Err(x)
                if engines == saber_hsa::EngineChoice::Auto
                    && lane.engines() != saber_hsa::Engines::Runtime =>
            {
                let why = format!(
                    " [auto: {:?} failed its self-test ({x}) -> Runtime]",
                    lane.engines()
                );
                lane.set_runtime_engines()
                    .map_err(|x| e("engines", x.to_string()))?;
                let m = lane
                    .self_test(all)
                    .map_err(|x| e("boot self-test (runtime engines)", x.to_string()))?;
                (m, why)
            }
            Err(x) => return Err(e("boot self-test", x.to_string())),
        };
        // every word starts at 0 (the self-test wrote the ALL word)
        self.hip
            .memset(&words, 0, &self.stream)
            .and_then(|()| self.hip.sync_stream(&self.stream))
            .map_err(|x| e("zero the mark words", x.to_string()))?;
        let desc = format!(
            "{:?} (H2D mask {:?}, D2H mask {:?}), ring {}, self-test {magic:#x} ok{fallback}",
            lane.engines(),
            lane.masks().0,
            lane.masks().1,
            lane.ring()
        );
        self.hsa = Some(std::cell::RefCell::new(HsaLane {
            lane,
            words,
            layers,
            planner: SignalPlanner::default(),
            fence_pending: false,
            counts: HsaCounts::default(),
            pending: VecDeque::new(),
        }));
        Ok(desc)
    }

    /// FN63b: whether the HSA lane is armed.
    #[must_use]
    pub fn hsa_lane(&self) -> bool {
        self.hsa.is_some()
    }

    /// FN63b: arm the copy-batch census (either mover; the HIP mover records two timing events
    /// a batch, the HSA lane turns ROCr's copy profiling on). A DIAGNOSTIC: off in served cells.
    pub fn set_census(&mut self) -> Result<(), TierError> {
        let mut c = Census::default();
        if let Some(h) = &self.hsa {
            h.borrow_mut()
                .lane
                .set_profiling(true)
                .map_err(|x| TierError::Mover(format!("census profiling: {x}")))?;
        } else {
            for _ in 0..2 {
                let ev = || {
                    self.hip
                        .create_event(EventFlags::Default)
                        .map_err(|x| TierError::Mover(format!("census event: {x}")))
                };
                c.hip_ev.push((ev()?, ev()?));
            }
        }
        self.census = Some(c);
        Ok(())
    }

    /// FN63b: harvest the HSA census batches that have landed (all of their tickets).
    fn hsa_harvest(&mut self) {
        let (Some(h), Some(c)) = (&self.hsa, self.census.as_mut()) else {
            return;
        };
        let mut h = h.borrow_mut();
        let hz = h.lane.hz();
        while let Some(&(t0, a, b)) = h.pending.front() {
            if !(a..=b).all(|t| h.lane.done(t)) {
                break;
            }
            h.pending.pop_front();
            let ts: Vec<(u64, u64)> = (a..=b).filter_map(|t| h.lane.times(t)).collect();
            if ts.len() as u64 != b - a + 1 {
                c.unread += 1;
                continue;
            }
            let s = ts.iter().map(|x| x.0).min().unwrap_or(0);
            let e = ts.iter().map(|x| x.1).max().unwrap_or(0);
            c.spans_ms
                .push((e.saturating_sub(s) as f64 / hz * 1e3) as f32);
            c.lands_ms
                .push((e.saturating_sub(t0) as f64 / hz * 1e3) as f32);
        }
    }

    /// FN63b: the HIP mover's census -- read the pair `k` if its batch landed.
    fn hip_harvest(&mut self, k: usize) {
        let Some(c) = self.census.as_mut() else {
            return;
        };
        if !c.hip_pending[k] {
            return;
        }
        let (s, e) = &c.hip_ev[k];
        match self.hip.query_event(e) {
            Ok(true) => match self.hip.elapsed_ms(s, e) {
                Ok(ms) => c.spans_ms.push(ms),
                Err(_) => c.unread += 1,
            },
            _ => c.unread += 1,
        }
        c.hip_pending[k] = false;
    }

    /// FN63b: the `fnmover63b` census line (the copy batches + the HSA lane's counters).
    #[must_use]
    pub fn census_line(&self, rank: u32) -> String {
        let mut out = format!(
            "fnmover63b rank{rank}: mover {}",
            if self.hsa.is_some() {
                "hsa-lane"
            } else if self.bg.is_some() {
                "two-queue"
            } else {
                "single"
            }
        );
        if let Some(c) = &self.census {
            out.push_str(&format!(
                " | census batches {} ({} unread) ops/batch {:.1} | device span p50 {:.3} p90 {:.3} p99 {:.3} ms",
                c.batches,
                c.unread,
                c.ops as f64 / c.batches.max(1) as f64,
                pct(&c.spans_ms, 0.5),
                pct(&c.spans_ms, 0.9),
                pct(&c.spans_ms, 0.99)
            ));
            if !c.lands_ms.is_empty() {
                out.push_str(&format!(
                    " | issue->landed p50 {:.3} p90 {:.3} p99 {:.3} ms",
                    pct(&c.lands_ms, 0.5),
                    pct(&c.lands_ms, 0.9),
                    pct(&c.lands_ms, 0.99)
                ));
            }
        }
        if let Some(h) = &self.hsa {
            let h = h.borrow();
            let l = h.lane.counts;
            let k = h.counts;
            out.push_str(&format!(
                " | lane {:?} h2d {} d2h {} words {} deps {} (landed at issue {}) ring waits {} host waits {} | batches {} fence free {} waits {} | gates waited {} skipped {} | all marks {} free {}",
                h.lane.engines(),
                l.h2d,
                l.d2h,
                l.words,
                l.deps,
                l.deps_landed,
                l.ring_waits,
                l.host_waits,
                k.batches,
                k.fence_free,
                k.fence_waits,
                k.gate_waits,
                k.gate_skips,
                k.all_marks,
                k.all_free
            ));
        }
        out
    }

    /// FN63b: an ALL mark over everything open on the HSA lane; `Some((word, value))` for the
    /// sleeper (`None`: nothing open, or no lane).
    fn hsa_all(&self) -> Result<Option<(saber_hip::DeviceAddr, u32)>, TierError> {
        let Some(h) = &self.hsa else {
            return Ok(None);
        };
        let mut h = h.borrow_mut();
        let HsaLane {
            lane,
            planner,
            words,
            layers,
            counts,
            ..
        } = &mut *h;
        let Some(j) = planner.all(&|t| lane.done(t)) else {
            counts.all_free += 1;
            return Ok(None);
        };
        let SigWhat::Mark { value, .. } = j.what else {
            return Err(TierError::Mover("the ALL job is not a mark".into()));
        };
        let w = words
            .addr()
            .offset(4 * u64::from(*layers))
            .ok_or_else(|| TierError::Mover("the ALL word's address".into()))?;
        lane.write_word(j.ticket, w.as_u64(), value, &j.deps)
            .map_err(|x| TierError::Mover(format!("hsa lane ALL mark: {x}")))?;
        counts.all_marks += 1;
        Ok(Some((w, value)))
    }

    /// FN59b: issue the one-stream copy batches through the HIP thread's background lane (the
    /// same jobs, streams and events in the same order; only the host time of the issue moves).
    /// FN54c: the two-queue mover's batches too (its lane jobs in the same order).
    pub fn set_submit(&mut self, on: bool) {
        self.submit = on;
    }

    /// FN59b: host ns spent inside `issue` so far.
    #[must_use]
    pub fn issue_ns(&self) -> u64 {
        self.issue_ns
    }

    /// FN59b: drop the tickets whose batch has been issued (all of them with `wait`), failing on
    /// the first batch that failed, by name.
    fn reap(&mut self, wait: bool) -> Result<(), TierError> {
        while let Some(t) = self.tickets.front_mut() {
            if !wait && !t.is_issued() {
                break;
            }
            let t = self.tickets.pop_front().expect("front");
            t.wait()
                .map_err(|x| TierError::Mover(format!("copy batch (submitted): {x}")))?;
        }
        Ok(())
    }

    /// FN54b: whether the two-queue mover is armed; the slot law's cross-lane waits so far.
    #[must_use]
    pub fn two_queue_waits(&self) -> Option<u64> {
        self.bg.as_ref().map(|b| b.planner.waits)
    }

    /// FN42b: the tier's lendable slots `first ..` live in `owner` (an allocation another client
    /// owns -- the prefill arena's tail), slot `first + i` at byte `offset + i * stride`. The
    /// copies into a lent slot and the slot read-back address it there.
    /// FN43: every call adds a region (a later client's ids start past the earlier's).
    pub fn set_lend(&mut self, owner: std::rc::Rc<DevicePtr>, offset: u64, first: u32) {
        // FN63b: the lane copies into lent slots too (the owner outlives the lane: field order)
        // FN51f-b: one owner may carry many regions (the elastic KV pool, a region a segment):
        // the lane registers each allocation once
        if let Some(h) = &self.hsa {
            if !self.lend.iter().any(|r| std::rc::Rc::ptr_eq(&r.0, &owner)) {
                h.borrow_mut().lane.register_device(&owner);
            }
        }
        self.lend.push((owner, offset, first));
        self.lend.sort_by_key(|r| r.2);
    }

    /// FN42b: VRAM slot `s`'s allocation and byte offset (a lendable slot: the owner's memory).
    /// FN51f-b: a binary search over the regions (ascending `first`; the elastic pool registers
    /// one region a segment).
    #[must_use]
    pub fn slot_at(&self, s: u32) -> (&DevicePtr, u64) {
        let i = self.lend.partition_point(|r| r.2 <= s);
        match i.checked_sub(1).map(|i| &self.lend[i]) {
            Some((p, off, first)) => (p.as_ref(), off + u64::from(s - first) * self.stride),
            None => (&self.vram, u64::from(s) * self.stride),
        }
    }

    /// FN43: THE RING'S COPIES of one layer on the ONE copy stream (see [`crate::stage_ring`]):
    /// wait for `after` (the buffer's previous user ran its experts), mark `start` (timing), copy
    /// each `(RAM slot, byte offset in ring)` one stride, then mark `arrived` (the layer's
    /// consumer waits on it before its grouped launches). The stream's order keeps every earlier
    /// tier copy (and the pinned slab's host writes, which drain first) ahead of these reads.
    pub fn stage(
        &self,
        ring: &DevicePtr,
        copies: &[(u32, u64)],
        after: Option<&Event>,
        start: Option<&Event>,
        arrived: &Event,
    ) -> Result<(), TierError> {
        let mut jobs = Vec::with_capacity(copies.len() + 6);
        // FN63b: the ring reads RAM slots an HSA demotion may have just written: the copy stream
        // waits the lane's ALL mark (nothing open: no wait)
        let all = self.hsa_all()?;
        if let Some((word, value)) = all {
            jobs.push(Job::StreamWaitValueGte {
                stream: &self.stream,
                word,
                value,
                mask: u32::MAX,
            });
        }
        // FN54b: the ring reads RAM slots a demotion may have just written: the background
        // lane's tail first
        if let Some(bg) = &self.bg {
            jobs.push(Job::RecordEvent {
                event: &bg.tail,
                stream: &bg.stream,
            });
            jobs.push(Job::StreamWaitEvent {
                stream: &self.stream,
                event: &bg.tail,
            });
        }
        if let Some(ev) = after {
            jobs.push(Job::StreamWaitEvent {
                stream: &self.stream,
                event: ev,
            });
        }
        if let Some(ev) = start {
            jobs.push(Job::RecordEvent {
                event: ev,
                stream: &self.stream,
            });
        }
        for &(r, off) in copies {
            jobs.push(Job::MemcpyPinned {
                dev: ring,
                dev_offset: off,
                host: self.ram_job(r)?.0,
                host_offset: self.ram_job(r)?.1,
                bytes: self.stride,
                kind: MemcpyKind::HostToDevice,
                stream: &self.stream,
            });
        }
        jobs.push(Job::RecordEvent {
            event: arrived,
            stream: &self.stream,
        });
        self.hip
            .run_batch(&jobs)
            .map(|_| ())
            .map_err(|x| TierError::Mover(format!("stage batch: {x}")))
    }

    /// FN43: the copy stream waits (on the device) for everything `c` has been given so far
    /// (the ring's first copies of a pass run after every earlier read of its slots).
    pub fn fence_after(&self, c: &Stream) -> Result<(), TierError> {
        self.hip
            .run_batch(&[
                Job::RecordEvent {
                    event: &self.fence,
                    stream: c,
                },
                Job::StreamWaitEvent {
                    stream: &self.stream,
                    event: &self.fence,
                },
            ])
            .map(|_| ())
            .map_err(|x| TierError::Mover(format!("ring fence: {x}")))
    }

    /// FN53b: the host buffer and byte offset of RAM slot `s` (the slab, a band segment or the
    /// bounce); `None` for a closed segment's slot or a slot past every region.
    fn ram_at(&self, s: u32) -> Option<(&PinnedBuf, u64)> {
        let st = self.stride;
        if let Some(v) = &self.valve {
            if s >= v.bounce_lo {
                let (b, _) = v.bounce.as_ref()?;
                return Some((b, u64::from(s - v.bounce_lo) * st));
            }
            if s >= v.band_lo {
                let k = (s - v.band_lo) / crate::expert_tier::VALVE_SEG;
                let (b, _) = v.segs.get(k as usize)?.as_ref()?;
                return Some((
                    b,
                    u64::from((s - v.band_lo) % crate::expert_tier::VALVE_SEG) * st,
                ));
            }
        }
        (u64::from(s) * st < self.ram.len() as u64).then(|| (&self.ram, u64::from(s) * st))
    }

    fn ram_job(&self, s: u32) -> Result<(&PinnedBuf, u64), TierError> {
        self.ram_at(s)
            .ok_or_else(|| TierError::Mover(format!("RAM slot {s} is in no open region")))
    }

    /// FN53b: arm the valve's memory map -- `segs` band segments at slot `band_lo` (allocated
    /// here, [`crate::expert_tier::VALVE_SEG`] slots each), the bounce at `bounce_lo` (allocated
    /// by [`Self::valve_bounce`]). Returns each segment's UVA address.
    pub fn valve_arm(
        &mut self,
        band_lo: u32,
        segs: u32,
        bounce_lo: u32,
    ) -> Result<Vec<u64>, TierError> {
        if self.hsa.is_some() || self.bg.is_some() {
            return Err(TierError::Config(
                "--fn-mem-valve runs on the single-stream mover only".into(),
            ));
        }
        self.valve = Some(ValveMem {
            band_lo,
            segs: (0..segs).map(|_| None).collect(),
            bounce_lo,
            bounce: None,
        });
        (0..segs).map(|k| self.valve_alloc(k)).collect()
    }

    /// FN53b: allocate band segment `k` (pinned); returns its UVA address.
    pub fn valve_alloc(&mut self, k: u32) -> Result<u64, TierError> {
        let bytes = self.stride * u64::from(crate::expert_tier::VALVE_SEG);
        let e = |w: &str, x: saber_hip::Error| TierError::Mover(format!("valve {w}: {x}"));
        let b = self.hip.host_malloc(bytes).map_err(|x| e("segment", x))?;
        let a = self
            .hip
            .host_device_addr(&b)
            .map_err(|x| e("segment address", x))?;
        let v = self
            .valve
            .as_mut()
            .ok_or_else(|| TierError::Config("valve not armed".into()))?;
        let slot = v
            .segs
            .get_mut(k as usize)
            .ok_or_else(|| TierError::Config(format!("valve segment {k}")))?;
        *slot = Some((b, a));
        Ok(a)
    }

    /// FN53b: free band segment `k` (its pinned pages go back to the host). The caller drained
    /// the copy stream and the consumer and moved every record out of it.
    pub fn valve_free(&mut self, k: u32) {
        if let Some(v) = self.valve.as_mut() {
            if let Some(s) = v.segs.get_mut(k as usize) {
                *s = None;
            }
        }
    }

    /// FN71c (`--xtier-nvme-p2p on`): put the valve's bounce in the card's P2PDMA pool -- the
    /// reader `nv` (route Pool) owns the pool; the bounce's `slots` slots are its first
    /// `slots * stride` bytes. The valve must be armed and the bounce not yet allocated; a reader
    /// on another route or a pool too small for the bounce is refused by name.
    pub fn set_pool_bounce(
        &mut self,
        nv: saber_hsa::NvmeVram,
        slots: u32,
    ) -> Result<(u64, u64), TierError> {
        let v = self.valve.as_ref().ok_or_else(|| {
            TierError::Config("--xtier-nvme-p2p on: the valve is not armed".into())
        })?;
        if v.bounce.is_some() {
            return Err(TierError::Config(
                "--xtier-nvme-p2p on: the bounce is already in pinned RAM".into(),
            ));
        }
        if nv.route() != saber_hsa::NvmeRoute::Pool {
            return Err(TierError::Config(format!(
                "--xtier-nvme-p2p on: the reader runs the {:?} route ({})",
                nv.route(),
                nv.p2p_refusal().unwrap_or("no refusal recorded")
            )));
        }
        let m = nv
            .pool()
            .ok_or_else(|| TierError::Config("--xtier-nvme-p2p on: no pool".into()))?;
        let need = self.stride * u64::from(slots.max(1));
        if m.bytes < need {
            return Err(TierError::Config(format!(
                "--xtier-nvme-p2p on: the pool holds {} B, the bounce needs {need} B ({slots} slots)",
                m.bytes
            )));
        }
        self.pool = Some(PoolBounce {
            nv,
            gpu: m.gpu,
            slots,
        });
        Ok((m.gpu, m.bytes))
    }

    /// FN71c: the pool route's counters (None = the bounce is in pinned RAM).
    #[must_use]
    pub fn pool_counts(&self) -> Option<saber_hsa::NvmeCounts> {
        self.pool.as_ref().map(|p| p.nv.counts)
    }

    /// FN53b: the bounce (`slots` pinned slots, allocated once); returns its UVA address.
    /// FN71c: with the pool bounce set, the pool's GPU address (nothing allocated).
    pub fn valve_bounce(&mut self, slots: u32) -> Result<u64, TierError> {
        let v = self
            .valve
            .as_ref()
            .ok_or_else(|| TierError::Config("valve not armed".into()))?;
        if let Some(p) = &self.pool {
            if slots > p.slots {
                return Err(TierError::Config(format!(
                    "the pool bounce holds {} slots, the tier asks {slots}",
                    p.slots
                )));
            }
            return Ok(p.gpu);
        }
        if let Some((_, a)) = &v.bounce {
            return Ok(*a);
        }
        let e = |w: &str, x: saber_hip::Error| TierError::Mover(format!("valve {w}: {x}"));
        let b = self
            .hip
            .host_malloc(self.stride * u64::from(slots.max(1)))
            .map_err(|x| e("bounce", x))?;
        let a = self
            .hip
            .host_device_addr(&b)
            .map_err(|x| e("bounce address", x))?;
        if let Some(v) = self.valve.as_mut() {
            v.bounce = Some((b, a));
        }
        Ok(a)
    }

    /// FN53b: copy RAM slot `from`'s bytes into slot `to` on the host (a valve close's
    /// relocations; the caller drained every stream).
    pub fn ram_copy(&mut self, from: u32, to: u32) -> Result<(), TierError> {
        let st = self.stride as usize;
        let src = {
            let (b, o) = self.ram_job(from)?;
            b.as_slice()[o as usize..o as usize + st].to_vec()
        };
        for (first, buf) in self.ram_regions_mut() {
            let n = (buf.len() / st) as u32;
            if to >= first && to - first < n {
                let o = (to - first) as usize * st;
                buf[o..o + st].copy_from_slice(&src);
                return Ok(());
            }
        }
        Err(TierError::Mover(format!(
            "RAM slot {to} is in no open region"
        )))
    }

    /// The VRAM arena's base address (the tier's `vram_base`).
    #[must_use]
    pub fn vram_base(&self) -> u64 {
        self.vram.addr().as_u64()
    }

    /// Copy one record's bytes (host) into VRAM slot `s` through RAM slot `staging` (the boot's
    /// warm path; blocking).
    pub fn warm_vram(&mut self, staging: u32, s: u32) -> Result<(), TierError> {
        let (dev, dev_offset) = self.slot_at(s);
        let jobs = [
            Job::MemcpyPinned {
                dev,
                dev_offset,
                host: self.ram_job(staging)?.0,
                host_offset: self.ram_job(staging)?.1,
                bytes: self.stride,
                kind: MemcpyKind::HostToDevice,
                stream: &self.stream,
            },
            Job::SyncStream {
                stream: &self.stream,
            },
        ];
        self.hip
            .run_batch(&jobs)
            .map(|_| ())
            .map_err(|x| TierError::Mover(format!("warm H2D: {x}")))
    }
}

impl HipCopyQueue<'_> {
    /// FN54b: one batch over the two lanes (the slot law's waits and marks between the copies).
    fn issue_two(&mut self, ops: &[CopyOp]) -> Result<(), TierError> {
        let mut plan = Vec::with_capacity(ops.len() * 3);
        self.bg
            .as_mut()
            .expect("two-queue armed")
            .planner
            .plan(ops, &mut plan);
        let bg = self.bg.as_ref().expect("two-queue armed");
        let st = self.stride;
        let stream = |l: Lane| match l {
            Lane::Urgent => &self.stream,
            Lane::Background => &bg.stream,
        };
        let mark = |l: Lane, seq: u64| &bg.marks[l as usize][(seq % MARK_RING as u64) as usize];
        let mut jobs = Vec::with_capacity(plan.len());
        for j in plan {
            jobs.push(match j {
                LaneOp::Op {
                    lane,
                    op: CopyOp::H2d { ram, vram },
                } => Job::MemcpyPinned {
                    dev: self.slot_at(vram).0,
                    dev_offset: self.slot_at(vram).1,
                    host: self.ram_job(ram)?.0,
                    host_offset: self.ram_job(ram)?.1,
                    bytes: st,
                    kind: MemcpyKind::HostToDevice,
                    stream: stream(lane),
                },
                LaneOp::Op {
                    lane,
                    op: CopyOp::D2h { vram, ram },
                } => Job::MemcpyPinned {
                    dev: self.slot_at(vram).0,
                    dev_offset: self.slot_at(vram).1,
                    host: self.ram_job(ram)?.0,
                    host_offset: self.ram_job(ram)?.1,
                    bytes: st,
                    kind: MemcpyKind::DeviceToHost,
                    stream: stream(lane),
                },
                LaneOp::Op {
                    lane,
                    op: CopyOp::Landed { layer },
                } => Job::RecordEvent {
                    event: self
                        .landed
                        .get(layer as usize)
                        .ok_or_else(|| TierError::Mover(format!("no event for layer {layer}")))?,
                    stream: stream(lane),
                },
                LaneOp::Mark { lane, seq } => Job::RecordEvent {
                    event: mark(lane, seq),
                    stream: stream(lane),
                },
                LaneOp::Wait { lane, seq } => Job::StreamWaitEvent {
                    stream: stream(lane),
                    event: mark(lane.other(), seq),
                },
            });
        }
        // FN54c: submitted like the one stream's under `--xtier-plan async` (FN59's lane orders
        // every later job naming these streams / events after the batch)
        if self.submit {
            let t = self
                .hip
                .submit_batch(&jobs)
                .map_err(|x| TierError::Mover(format!("two-queue copy batch (submit): {x}")))?;
            self.tickets.push_back(t);
            return self.reap(false);
        }
        self.hip
            .run_batch(&jobs)
            .map(|_| ())
            .map_err(|x| TierError::Mover(format!("two-queue copy batch: {x}")))
    }
    /// FN63b: one batch on the HSA lane (the signal law; see [`Self::enable_hsa_lane`]).
    fn issue_hsa(&mut self, ops: &[CopyOp]) -> Result<(), TierError> {
        let e = |w: &str, x: String| TierError::Mover(format!("hsa lane: {w}: {x}"));
        self.hsa_harvest();
        let census = self.census.is_some();
        let st = self.stride;
        let ram_host = self.ram.addr() as u64;
        let h = self.hsa.as_ref().expect("hsa armed");
        // the fence (law 1): every copy starts after the consumer's reads of the step. The
        // engine has normally drained the consumer stream past its fence already (the route
        // log's read); otherwise the host waits for it here (counted)
        let pending = h.borrow().fence_pending;
        if pending && ops.iter().any(|o| !matches!(o, CopyOp::Landed { .. })) {
            let done = self
                .hip
                .query_event(&self.fence)
                .map_err(|x| e("fence query", x.to_string()))?;
            let mut hb = h.borrow_mut();
            if done {
                hb.counts.fence_free += 1;
            } else {
                drop(hb);
                self.hip
                    .sync_event(&self.fence)
                    .map_err(|x| e("fence wait", x.to_string()))?;
                hb = h.borrow_mut();
                hb.counts.fence_waits += 1;
            }
            hb.fence_pending = false;
        }
        let mut hb = h.borrow_mut();
        let HsaLane {
            lane,
            planner,
            words,
            layers,
            counts,
            pending,
            ..
        } = &mut *hb;
        let mut jobs = Vec::with_capacity(ops.len());
        planner.plan(ops, &|t| lane.done(t), &mut jobs);
        if jobs.is_empty() {
            return Ok(());
        }
        let t0 = if census {
            lane.now().map_err(|x| e("timestamp", x.to_string()))?
        } else {
            0
        };
        let wbase = words.addr().as_u64();
        for j in &jobs {
            let r = match j.what {
                SigWhat::Copy(CopyOp::H2d { ram, vram }) => {
                    let (d, off) = self.slot_at(vram);
                    lane.copy(
                        j.ticket,
                        d.addr().as_u64() + off,
                        ram_host + u64::from(ram) * st,
                        st,
                        &j.deps,
                    )
                }
                SigWhat::Copy(CopyOp::D2h { vram, ram }) => {
                    let (d, off) = self.slot_at(vram);
                    lane.copy(
                        j.ticket,
                        ram_host + u64::from(ram) * st,
                        d.addr().as_u64() + off,
                        st,
                        &j.deps,
                    )
                }
                SigWhat::Mark {
                    word: MarkWord::Layer(l),
                    value,
                } => {
                    if l >= *layers {
                        return Err(TierError::Mover(format!("no word for layer {l}")));
                    }
                    lane.write_word(j.ticket, wbase + 4 * u64::from(l), value, &j.deps)
                }
                SigWhat::Mark {
                    word: MarkWord::All,
                    value,
                } => lane.write_word(j.ticket, wbase + 4 * u64::from(*layers), value, &j.deps),
                SigWhat::Copy(CopyOp::Landed { .. }) => {
                    return Err(TierError::Mover("a landed op planned as a copy".into()))
                }
            };
            r.map_err(|x| e("issue", x.to_string()))?;
        }
        counts.batches += 1;
        if census {
            pending.push_back((t0, jobs[0].ticket, jobs[jobs.len() - 1].ticket));
            drop(hb);
            if let Some(c) = self.census.as_mut() {
                c.batches += 1;
                c.ops += jobs.len() as u64;
            }
        }
        Ok(())
    }

    /// One batch on the one copy stream (today's law; FN59b: submitted when armed).
    fn issue_one(&mut self, ops: &[CopyOp]) -> Result<(), TierError> {
        if self.hsa.is_some() {
            return self.issue_hsa(ops);
        }
        if self.bg.is_some() {
            return self.issue_two(ops);
        }
        // FN63b: the census' timing pair around the batch (the pair two batches back is read)
        let ck = self.census.as_ref().map(|c| c.hip_k);
        if let Some(k) = ck {
            self.hip_harvest(k);
        }
        let law = self.marks.as_ref().map_or(MarkLaw::Layer, |m| m.law);
        let slots = if law == MarkLaw::Word {
            self.word_slots(ops)?
        } else {
            Vec::new()
        };
        let st = self.stride;
        let mut jobs = Vec::with_capacity(ops.len() + 2);
        if let Some(k) = ck {
            let c = self.census.as_ref().expect("census");
            jobs.push(Job::RecordEvent {
                event: &c.hip_ev[k].0,
                stream: &self.stream,
            });
        }
        for (i, op) in ops.iter().enumerate() {
            let job = match *op {
                CopyOp::H2d { ram, vram } => Job::MemcpyPinned {
                    dev: self.slot_at(vram).0,
                    dev_offset: self.slot_at(vram).1,
                    host: self.ram_job(ram)?.0,
                    host_offset: self.ram_job(ram)?.1,
                    bytes: st,
                    kind: MemcpyKind::HostToDevice,
                    stream: &self.stream,
                },
                CopyOp::D2h { vram, ram } => Job::MemcpyPinned {
                    dev: self.slot_at(vram).0,
                    dev_offset: self.slot_at(vram).1,
                    host: self.ram_job(ram)?.0,
                    host_offset: self.ram_job(ram)?.1,
                    bytes: st,
                    kind: MemcpyKind::DeviceToHost,
                    stream: &self.stream,
                },
                // FN72b step law: no per-layer mark (the gate records one over everything)
                CopyOp::Landed { .. } if law == MarkLaw::Step => continue,
                // FN72b word law: the layer's word <- its new value, after its copies (stream
                // order), an SDMA write; a ring quarter's last value also records its guard
                CopyOp::Landed { .. } if law == MarkLaw::Word => {
                    let (layer, slot, guard) = slots[i].expect("a landed op's slot");
                    let m = self.marks.as_ref().expect("marks");
                    jobs.push(Job::MemcpyPinned {
                        dev: m.words.as_ref().expect("word law words"),
                        dev_offset: MARK_WORD_BYTES as u64 * u64::from(layer),
                        host: m.ring.as_ref().expect("word law ring"),
                        host_offset: 4 * slot as u64,
                        bytes: MARK_WORD_BYTES as u64,
                        kind: MemcpyKind::HostToDevice,
                        stream: &self.stream,
                    });
                    match guard {
                        Some(q) => Job::RecordEvent {
                            event: &m.guards[q],
                            stream: &self.stream,
                        },
                        None => continue,
                    }
                }
                CopyOp::Landed { layer } => Job::RecordEvent {
                    event: self
                        .landed
                        .get(layer as usize)
                        .ok_or_else(|| TierError::Mover(format!("no event for layer {layer}")))?,
                    stream: &self.stream,
                },
            };
            jobs.push(job);
        }
        if let Some(k) = ck {
            let c = self.census.as_ref().expect("census");
            jobs.push(Job::RecordEvent {
                event: &c.hip_ev[k].1,
                stream: &self.stream,
            });
        }
        let copies = ops
            .iter()
            .any(|o| matches!(o, CopyOp::H2d { .. } | CopyOp::D2h { .. }));
        let r = if self.submit {
            match self.hip.submit_batch(&jobs) {
                Ok(t) => {
                    self.tickets.push_back(t);
                    self.reap(false)
                }
                Err(x) => Err(TierError::Mover(format!("copy batch (submit): {x}"))),
            }
        } else {
            self.hip
                .run_batch(&jobs)
                .map(|_| ())
                .map_err(|x| TierError::Mover(format!("copy batch: {x}")))
        };
        if let Some(m) = self.marks.as_mut() {
            m.step_dirty |= copies;
        }
        if let (Some(k), Some(c)) = (ck, self.census.as_mut()) {
            c.hip_pending[k] = true;
            c.hip_k = 1 - k;
            c.batches += 1;
            c.ops += ops.len() as u64;
        }
        r
    }
}

impl CopyQueue for HipCopyQueue<'_> {
    fn mark_word(&self, layer: u32) -> Option<(saber_hip::DeviceAddr, u32, u64)> {
        // FN98: the one-stream mover's word law only (the HSA lane / two-queue marks are other
        // words or events)
        if self.hsa.is_some() || self.bg.is_some() {
            return None;
        }
        let m = self.marks.as_ref().filter(|m| m.law == MarkLaw::Word)?;
        let value = *m.values.get(layer as usize)?;
        let seq = *m.seqs.get(layer as usize)?;
        let word = m
            .words
            .as_ref()?
            .addr()
            .offset(MARK_WORD_BYTES as u64 * u64::from(layer))?;
        Some((word, value, seq))
    }
    type Consumer = Stream;
    fn issue(&mut self, ops: &[CopyOp]) -> Result<(), TierError> {
        let t0 = std::time::Instant::now();
        let r = self.issue_one(ops);
        self.issue_ns += t0.elapsed().as_nanos() as u64;
        r
    }
    fn drain(&mut self) -> Result<(), TierError> {
        if let Some(h) = &self.hsa {
            // FN63b: every open ticket (the marks cover their copies; unmarked copies are open)
            let mut hb = h.borrow_mut();
            let HsaLane { lane, planner, .. } = &mut *hb;
            planner.prune(&|t| lane.done(t));
            for &t in planner.open() {
                lane.wait(t)
                    .map_err(|x| TierError::Mover(format!("drain (hsa lane): {x}")))?;
            }
            planner.prune(&|t| lane.done(t));
            drop(hb);
            self.hsa_harvest();
        }
        if let Some(bg) = &self.bg {
            self.hip
                .sync_stream(&bg.stream)
                .map_err(|x| TierError::Mover(format!("drain (background): {x}")))?;
        }
        // a host-blocking sync drains the HIP thread's background lane first (FN59's law)
        self.hip
            .sync_stream(&self.stream)
            .map_err(|x| TierError::Mover(format!("drain: {x}")))?;
        self.reap(true)
    }
    fn ram_mut(&mut self) -> &mut [u8] {
        self.ram.as_mut_slice()
    }
    fn pool_reads(
        &mut self,
        xt: &XtierFile,
        reads: &[(u64, u32)],
        qd: usize,
    ) -> Result<Vec<(u64, u32)>, TierError> {
        let (Some(p), Some(v)) = (self.pool.as_mut(), self.valve.as_ref()) else {
            return Ok(reads.to_vec());
        };
        let (lo, n) = (v.bounce_lo, p.slots);
        let (mine, rest): (Vec<_>, Vec<_>) =
            reads.iter().partition(|&&(_, s)| s >= lo && s - lo < n);
        if !mine.is_empty() {
            let h = xt.header();
            let items: Vec<(u64, u64)> = mine
                .iter()
                .map(|&(i, s)| (h.record_offset(i), p.gpu + u64::from(s - lo) * self.stride))
                .collect();
            p.nv.read_pool_many(xt.file(), &items, self.stride, qd)
                .map_err(|e| TierError::Mover(format!("pool read: {e}")))?;
        }
        Ok(rest)
    }
    fn ram_regions_mut(&mut self) -> Vec<(u32, &mut [u8])> {
        let mut v = vec![(0, self.ram.as_mut_slice())];
        if let Some(m) = self.valve.as_mut() {
            for (k, sg) in m.segs.iter_mut().enumerate() {
                if let Some((b, _)) = sg.as_mut() {
                    v.push((
                        m.band_lo + k as u32 * crate::expert_tier::VALVE_SEG,
                        b.as_mut_slice(),
                    ));
                }
            }
            if let Some((b, _)) = m.bounce.as_mut() {
                v.push((m.bounce_lo, b.as_mut_slice()));
            }
        }
        v
    }
    fn consumer_waits(&mut self, layer: u32, c: &Stream) -> Result<(), TierError> {
        if let Some(h) = &self.hsa {
            // FN63b: the CP waits the layer's word >= its latest mark -- or nothing, when that
            // mark has already landed (the word holds it: the wait would pass at once)
            let mut hb = h.borrow_mut();
            let mark = hb.planner.layer_mark(layer);
            let Some((t, value)) = mark.filter(|&(t, _)| !hb.lane.done(t)) else {
                hb.counts.gate_skips += 1;
                return Ok(());
            };
            let _ = t;
            hb.counts.gate_waits += 1;
            let word = hb
                .words
                .addr()
                .offset(4 * u64::from(layer))
                .ok_or_else(|| TierError::Mover("a layer word's address".into()))?;
            drop(hb);
            return self
                .hip
                .run_batch(&[Job::StreamWaitValueGte {
                    stream: c,
                    word,
                    value,
                    mask: u32::MAX,
                }])
                .map(|_| ())
                .map_err(|x| TierError::Mover(format!("gate wait (hsa lane): {x}")));
        }
        if let Some(m) = self.marks.as_mut() {
            m.counts[1] += 1;
            if m.law == MarkLaw::Word {
                // FN72b: the CP waits the layer's word >= its latest mark (written by the SDMA
                // after the layer's copies)
                let value = *m
                    .values
                    .get(layer as usize)
                    .ok_or_else(|| TierError::Mover(format!("no mark word for layer {layer}")))?;
                let word = m
                    .words
                    .as_ref()
                    .expect("word law words")
                    .addr()
                    .offset(MARK_WORD_BYTES as u64 * u64::from(layer))
                    .ok_or_else(|| TierError::Mover("a layer word's address".into()))?;
                return self
                    .hip
                    .run_batch(&[Job::StreamWaitValueGte {
                        stream: c,
                        word,
                        value,
                        mask: u32::MAX,
                    }])
                    .map(|_| ())
                    .map_err(|x| TierError::Mover(format!("gate wait (mark word): {x}")));
            }
            // FN72b step law: ONE event over every op issued so far, recorded at the first gate
            // after a copy (a superset of the layer's mark: exact)
            let record = std::mem::replace(&mut m.step_dirty, false);
            if record {
                m.counts[3] += 1;
            }
            let m = self.marks.as_ref().expect("marks");
            let mut jobs = Vec::with_capacity(2);
            if record {
                jobs.push(Job::RecordEvent {
                    event: &m.step_ev,
                    stream: &self.stream,
                });
            }
            jobs.push(Job::StreamWaitEvent {
                stream: c,
                event: &m.step_ev,
            });
            return self
                .hip
                .run_batch(&jobs)
                .map(|_| ())
                .map_err(|x| TierError::Mover(format!("gate wait (step mark): {x}")));
        }
        let ev = self
            .landed
            .get(layer as usize)
            .ok_or_else(|| TierError::Mover(format!("no event for layer {layer}")))?;
        self.hip
            .stream_wait_event(c, ev)
            .map_err(|x| TierError::Mover(format!("gate wait: {x}")))
    }
    fn after(&mut self, c: &Stream) -> Result<(), TierError> {
        let mut jobs = vec![
            Job::RecordEvent {
                event: &self.fence,
                stream: c,
            },
            Job::StreamWaitEvent {
                stream: &self.stream,
                event: &self.fence,
            },
        ];
        // FN54b: both lanes start after the consumer's reads (law 1)
        if let Some(bg) = &self.bg {
            jobs.push(Job::StreamWaitEvent {
                stream: &bg.stream,
                event: &self.fence,
            });
        }
        // FN63b: the lane's next issue checks this fence (see `issue_hsa`)
        if let Some(h) = &self.hsa {
            h.borrow_mut().fence_pending = true;
        }
        self.hip
            .run_batch(&jobs)
            .map(|_| ())
            .map_err(|x| TierError::Mover(format!("fence: {x}")))
    }
    fn consumer_waits_all(&mut self, c: &Stream) -> Result<(), TierError> {
        if self.hsa.is_some() {
            // FN63b law 3: the consumer waits the ALL word (nothing open: no wait)
            return match self.hsa_all()? {
                Some((word, value)) => self
                    .hip
                    .run_batch(&[Job::StreamWaitValueGte {
                        stream: c,
                        word,
                        value,
                        mask: u32::MAX,
                    }])
                    .map(|_| ())
                    .map_err(|x| TierError::Mover(format!("publication wait (hsa lane): {x}"))),
                None => Ok(()),
            };
        }
        let mut jobs = vec![
            Job::RecordEvent {
                event: &self.all,
                stream: &self.stream,
            },
            Job::StreamWaitEvent {
                stream: c,
                event: &self.all,
            },
        ];
        if let Some(bg) = &self.bg {
            jobs.push(Job::RecordEvent {
                event: &bg.all,
                stream: &bg.stream,
            });
            jobs.push(Job::StreamWaitEvent {
                stream: c,
                event: &bg.all,
            });
        }
        self.hip
            .run_batch(&jobs)
            .map(|_| ())
            .map_err(|x| TierError::Mover(format!("publication wait: {x}")))
    }
}
