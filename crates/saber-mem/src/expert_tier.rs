//! The Flash-Next expert tier: per rank, VRAM expert slots ← a pinned host-RAM L2 ← the NVMe
//! `experts.xtier` records (docs/EXEC.md §14, the memory-plane law; plan §6 "Residency on two
//! cards").
//!
//! ```text
//!   VRAM slots (N, allocated once)      pinned RAM slots (M + 1 spare)       NVMe experts.xtier
//!  ┌───────────────────────────┐  H2D  ┌────────────────────────────┐ pread ┌───────────────────┐
//!  │ per-layer LRU partitions  │ ◄──── │ global LRU, UVA-readable   │ ◄──── │ 12,288 records    │
//!  │ (profile water-fill)      │ ────► │ by the MoE kernel in place │       │ one O_DIRECT read │
//!  └───────────────────────────┘  D2H  └────────────────────────────┘ drop  └───────────────────┘
//! ```
//!
//! **What is structural** (the invariants of ARCHITECTURE.md):
//!
//! * **I1.** The VRAM slot arena and the RAM slab are allocated ONCE, by the caller, from the byte
//!   ledger ([`crate::flashnext_budget`]); the tier only ever changes which record a slot holds.
//!   Every address it hands out is `base + slot · stride + sub-record offset`.
//! * **I2.** Every decision is a pure function of the routes the rank already holds and the tier's
//!   own state: distinct experts are taken in expert-id order, VRAM victims by the total order
//!   `(stamp, slot)`, RAM victims by `(stamp, slot)`, prefetch candidates by `(score, record)`. No
//!   clock, no hash-map iteration order, no atomics decide a byte.
//! * **I3.** Every routed expert is counted on exactly one path — `resident` (a VRAM slot), `ram`
//!   (the kernel reads the pinned RAM slot over UVA in place), `nvme` (a demand read this step) —
//!   and `prefetched` counts the `ram` hits whose record a prefetch brought in. A layer whose
//!   routed expert is still `Fetch` after [`ExpertTier::resolve`] is an error, never a zero read.
//!
//! **The step law.** `begin_step` → for every MoE layer, after its router: `resolve(layer,
//! routed ids)` (a `Fetch` expert is read from NVMe into a RAM slot NOW, the step waits for that
//! read only; an expert already in RAM or VRAM never waits) → the grouped GEMM reads the
//! placements [`ExpertTier::fill_xptr`] wrote → `end_step` runs the LRU over the step's routes
//! (promotions RAM → VRAM with the victim demoted VRAM → RAM through the one spare RAM slot,
//! capped at `max_inserts` per layer, no inserts on a layer whose distinct routed experts exceed
//! `read_through · slots`), then the next-step prefetch (NVMe → RAM from the per-expert access
//! history). When `vram_slots + ram_slots` covers every record ([`ExpertTier::complete`]) no expert
//! is ever `Fetch` after boot: the NVMe is off the step path and `resolve` never blocks.

#![forbid(unsafe_code)]

use std::fmt::Write as _;

use thiserror::Error;

use crate::stage_ring::{StageKey, PF_SEEN_DECAY};
use crate::xtier::{XtierError, XtierHeader};

/// FN117b (S2): the inputs of every `xptr` table ([`ExpertTier::xptr_key`]); equal keys = equal
/// tables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XptrKey {
    place: Vec<Placement>,
    lend_addr: Vec<u64>,
    vram_base: u64,
    ram_base: u64,
    bounce: (u32, u64),
    band: (u32, Vec<u64>),
}

/// Where one (layer, expert) lives this step — the value FN4's `xptr` entries are filled from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// In VRAM slot `n` of the rank's slot arena.
    Vram(u32),
    /// In pinned RAM slot `n`; the MoE kernel reads it over UVA in place.
    Ram(u32),
    /// Only on NVMe: must be read before a kernel may touch it.
    Fetch,
}

/// Why the tier refused.
#[derive(Debug, Error)]
pub enum TierError {
    /// The configuration cannot work (numbers named).
    #[error("expert tier config: {0}")]
    Config(String),
    /// Every RAM slot holds an expert routed this step: a demand read has nowhere to land.
    #[error("rank {rank} layer {layer}: RAM L2 exhausted ({slots} slots, all routed this step)")]
    RamExhausted {
        /// Rank.
        rank: u32,
        /// Layer being resolved.
        layer: u32,
        /// RAM slots.
        slots: u32,
    },
    /// A routed expert is still `Fetch` at dispatch (I3).
    #[error("rank {rank} layer {layer} expert {expert}: dispatched while Fetch")]
    Unresolved {
        /// Rank.
        rank: u32,
        /// Layer.
        layer: u32,
        /// Global expert id.
        expert: u32,
    },
    /// The mover failed.
    #[error("expert mover: {0}")]
    Mover(String),
    /// A record read failed.
    #[error(transparent)]
    Xtier(#[from] XtierError),
    /// A profile file was refused.
    #[error("route profile: {0}")]
    Profile(String),
}

/// The tier's policy parameters. Defaults are the plan §6 port of `q38fn-lru`.
#[derive(Debug, Clone, PartialEq)]
pub struct TierParams {
    /// VRAM expert slots on this rank (the ledger's remainder; `--xtier-vram-slots`).
    pub vram_slots: u32,
    /// Pinned RAM L2 slots on this rank (`--xtier-ram-slots`); one more is allocated as the swap
    /// spare.
    pub ram_slots: u32,
    /// A layer routing more than this fraction of its VRAM slots in one step does no inserts
    /// (reads through): prefill chunks and wide batches do not thrash the cache. Fork: 0.5.
    pub read_through: f64,
    /// Inserts (RAM → VRAM promotions) per layer per step. Fork: 64.
    pub max_inserts: u32,
    /// Next-step prefetch reads (NVMe → RAM) per step; 0 = the demand path only.
    pub prefetch_per_step: u32,
    /// Per-step decay of the per-expert access score the prefetch ranks by.
    pub history_decay: f32,
    /// Floor of VRAM slots per layer in the profile water-fill (every layer stays adaptive).
    pub min_layer_slots: u32,
    /// FN17: THE DEFERRED PUBLICATION LAW when > 0 -- this many EXTRA pinned RAM slots (beyond
    /// `ram_slots` + the FN8 spare) carry the demotions in flight. A step's promotions are copied
    /// into a HOLE (an empty VRAM slot of the layer) and its demotions into a free RAM spare, so no
    /// copy writes a byte any placement points at; the placements move (are PUBLISHED) at the NEXT
    /// `end_step`, after the consumer's fence waited for the copies -- no step waits for the
    /// copies of its own end. 0 = FN8's law (swap through the one spare, publish at once, the
    /// consumer gates per layer). Only a complete tier defers (the demand path is FN8's).
    pub defer_spares: u32,
    /// FN18 (the user's RLFU question): LRFU victim ranking -- a VRAM victim is the record with the
    /// smallest decayed access score `crf = crf * lambda^(steps since its last use) + 1` (on use),
    /// then `(stamp, slot)`. 0 = FN8's recency-only `(stamp, slot)` ranking (the control).
    pub lrfu_lambda: f64,
    /// FN18: TinyLFU-style admission -- per-record route counts over a recent window (every count
    /// halved each `admit_window` steps); a RAM-placed route is promoted only if its count exceeds
    /// the victim's (an empty slot always admits); otherwise it stays in RAM (read over UVA). 0 = off.
    pub admit_window: u32,
    /// FN39: THE ADMISSION MARGIN (`--xtier-admit-margin`) -- a RAM-placed route is promoted over
    /// a resident victim only when its LRFU score exceeds the victim's by at least this much
    /// (hysteresis on the score; an empty slot always admits). Needs `lrfu_lambda > 0`. 0 = off
    /// (today's law: every capped miss is promoted).
    pub admit_margin: f64,
    /// FN39: `--xtier-admit-uses` -- a RAM-placed route is promoted only once it was routed at
    /// least this many steps since it entered RAM (the step it arrived counts). 0 = off.
    pub admit_uses: u32,
    /// FN36: where a promotion's VRAM victim goes (`--xtier-demote`). `Ram` = FN8's law (a D2H
    /// copy of the victim into the spare RAM slot); `None` = THE DEMOTION-FREE LAW: the victim's
    /// record is re-read from NVMe (immutable weights, the same bytes) into a refill pool slot
    /// under the NEXT step, and the swap (victim -> that RAM slot, promotion H2D into its VRAM
    /// slot) is published at the next `end_step` -- the GPU link carries the H2D only, no step
    /// reads a byte in flight, and the tier stays complete. Needs `refill_spares` and a complete
    /// tier (otherwise FN8's law runs).
    pub demote: Demote,
    /// FN36: the refill pool (extra pinned RAM slots after the FN17 spares): victims in flight +
    /// the promoted records' old RAM slots retiring for one step. `2 x max_inserts x layers`.
    pub refill_spares: u32,
    /// FN42: LENT slots activated at construction (the host replay's arm: `lend_boot` lendable
    /// slots at synthetic addresses, all lent at once; the engine registers its real addresses
    /// with [`ExpertTier::register_lendable`] and keeps this 0).
    pub lend_boot: u32,
    /// FN48: the VRAM victim policy (`--xtier-policy`); the default is FN18's LRFU, byte for byte.
    pub policy: VictimPolicy,
    /// FN54: SECOND-CHANCE insertion (`--xtier-second-chance W`) -- a RAM-placed route is a
    /// promotion candidate only when the same record was also a RAM-placed route on an earlier
    /// step at most `W` steps ago (its first miss in a window only marks it). 0 = off (today's
    /// law: every miss is a candidate). Placement only: the bytes the kernels read never change.
    pub second_chance: u32,
    /// FN54b: extra swap spares for FN8's law (`--xtier-mover two-queue`): the D2H spares rotate
    /// FIFO through `1 + spare_ring` RAM slots (the slab's last), so a demotion's D2H writes a RAM
    /// slot whose promotion H2D read it `spare_ring` swaps earlier -- not the one just read --
    /// and a second copy queue can run the D2H direction beside the H2D one. 0 = today's one
    /// spare (byte for byte). Placement only: which RAM slot holds a demoted record.
    pub spare_ring: u32,
    /// FN53b: THE VALVE BAND (`--fn-mem-valve on`) -- this many EXTRA L2 slots (a multiple of
    /// [`VALVE_SEG`]) past every spare, each [`VALVE_SEG`]-slot segment its own pinned
    /// allocation, so the memory-plane valve can DEMOTE a segment (its coldest records to the
    /// drive, the allocation freed) and reopen it live. 0 = today's slab, byte for byte.
    pub band_slots: u32,
    /// FN53b: the prefill BOUNCE -- slots past the band (two halves, alternating by layer) a
    /// prefill pass reads its `Fetch` records into (the layer-ahead tail reader) instead of the
    /// L2: a prefill chunk routes every record, so inserting its misses would evict the decode
    /// working set. 0 = no bypass (the demand path inserts into the L2).
    pub bounce_slots: u32,
}

/// FN53b: the valve's move unit: one band segment = this many L2 slots (one pinned allocation).
pub const VALVE_SEG: u32 = 256;

/// FN53b: the valve's which-path counters (rule 5) and moves.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ValveCounts {
    /// Segments closed (records demoted to the drive) / reopened.
    pub closes: u64,
    /// Segments reopened.
    pub opens: u64,
    /// Records dropped to `Fetch` by a close.
    pub dropped: u64,
    /// Band records relocated into freed lower slots by a close (host copies).
    pub relocated: u64,
    /// Records read back from the drive after an open.
    pub refilled: u64,
    /// Prefill tail records read blind one layer ahead into the bounce.
    pub bounce_prefetch: u64,
    /// ... of which the layer then routed.
    pub bounce_hits: u64,
    /// Prefill `Fetch` routes read on demand into the bounce (not prefetched).
    pub bounce_demand: u64,
    /// Prefill `Fetch` routes that found no bounce room and entered the L2.
    pub bounce_full: u64,
    /// FN53c: the device miss flag's requests (layers of a graph step that missed) ...
    pub xmiss_requests: u64,
    /// ... and the records they read drive -> bounce.
    pub xmiss_records: u64,
    /// FN53c: `Fetch` routes a miss-flag pass resolved as served (kept `Fetch`: the open set is
    /// static under the miss flag).
    pub xmiss_routes: u64,
}

/// FN53c: one record the device miss flag asked for -- its record index, the bounce slot it is
/// read into and the six sub-record addresses (gate_up x3, down x3) the kernel patches in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XmissSlot {
    /// The record index.
    pub record: u32,
    /// The bounce slot (a RAM slot id).
    pub slot: u32,
    /// The xptr entries: `[gu0, gu1, gu2, dn0, dn1, dn2]`.
    pub entries: [u64; 6],
}

/// FN53b: what one valve close did (the engine copies `relocs` host-side, then frees the
/// segment's allocation).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ValveClose {
    /// The segment index closed.
    pub seg: u32,
    /// `(from band slot, to slot)` copies of records (or lent shadows / the spare) that stay.
    pub relocs: Vec<(u32, u32)>,
    /// Records dropped to `Fetch` (the coldest, layer-balanced).
    pub dropped: Vec<u32>,
}

/// FN48: the VRAM victim ranking's form. [`VictimKind::Lrfu`] with the neutral knobs (the
/// [`Default`]) is FN18's LRFU exactly; every other setting is placement only (bit-exact bytes).
#[derive(Debug, Clone, PartialEq)]
pub struct VictimPolicy {
    /// The ranking.
    pub kind: VictimKind,
    /// LRFU: the score a route adds on a READ-THROUGH layer (a prefill chunk's wide routing);
    /// 1.0 = today. Reuse: 0.0 = a read-through layer's routes do not update the reuse estimate.
    pub rt_weight: f64,
    /// LRFU: lambda per layer (empty = `lrfu_lambda` everywhere).
    pub lambda_by_layer: Vec<f64>,
    /// LRFU: + beta * ln(1 + the warm profile's route count) on the victim score (0 = off).
    pub prior_beta: f64,
    /// The capped misses fill the cap in the policy's order (most valuable first) instead of
    /// expert-id order.
    pub miss_rank: bool,
    /// Reuse: the EWMA weight of the newest inter-reference gap.
    pub reuse_alpha: f64,
    /// Reuse: an overdue record's predicted remaining time = this x its age (0 = the gap only).
    pub reuse_age: f64,
    /// Reuse: the gap assumed for a record seen once (steps).
    pub reuse_first_gap: f64,
    /// LRFU: a route adds (rows routing it)^gamma instead of 1 (0 = today).
    pub row_gamma: f64,
    /// Table: the learned value per (class, age bucket) ([`table_index`]; higher = kept), fit on
    /// another trace by `route_trace::learn_table`.
    pub table: Vec<f64>,
    /// Table: the decay of the class counter (a record's class = its decayed use count at its
    /// last use + whether that use was a read-through layer's).
    pub table_lambda: f64,
}

/// FN48: the table's class count (decayed-count half-octaves x read-through) and age buckets.
pub const TABLE_CLASSES: usize = 24;
/// FN48: see [`TABLE_CLASSES`].
pub const TABLE_AGES: usize = 80;

/// FN48: the class of a record whose decayed use count is `lc` right after a use (`rt` = that
/// use was a read-through layer's).
#[must_use]
pub fn table_class(lc: f64, rt: bool) -> usize {
    let c = ((2.0 * lc.max(1.0).log2()) as usize).min(TABLE_CLASSES / 2 - 1);
    c * 2 + usize::from(rt)
}

/// FN48: the age bucket (exact below 8 steps, then quarter octaves).
#[must_use]
pub fn table_age(age: u64) -> usize {
    if age < 8 {
        return age as usize;
    }
    let b = 8 + (4.0 * ((age as f64).log2() - 3.0)) as usize;
    b.min(TABLE_AGES - 1)
}

/// FN48: the table's flat index.
#[must_use]
pub fn table_index(class: usize, age: usize) -> usize {
    class * TABLE_AGES + age
}

/// FN48: see [`VictimPolicy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VictimKind {
    /// FN18: the smallest decayed access score (needs `lrfu_lambda > 0`; at 0 FN8's LRU).
    #[default]
    Lrfu,
    /// The farthest PREDICTED next use (last use + the EWMA inter-reference gap; Belady's rule
    /// with the future estimated).
    Reuse,
    /// A LEARNED value per (class, age) -- the conditional expected time to next use (or reuse
    /// probability) measured on another trace; the smallest value leaves.
    Table,
}

impl Default for VictimPolicy {
    fn default() -> Self {
        VictimPolicy {
            kind: VictimKind::Lrfu,
            rt_weight: 1.0,
            lambda_by_layer: Vec::new(),
            prior_beta: 0.0,
            miss_rank: false,
            reuse_alpha: 0.3,
            reuse_age: 1.0,
            reuse_first_gap: 64.0,
            row_gamma: 0.0,
            table: Vec::new(),
            table_lambda: 0.98,
        }
    }
}

/// FN36: where a VRAM victim goes (see [`TierParams::demote`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Demote {
    /// FN8's law: D2H the victim into the spare RAM slot.
    #[default]
    Ram,
    /// The demotion-free law: re-read the victim from NVMe, publish the swap a step later.
    None,
}

impl Default for TierParams {
    fn default() -> Self {
        TierParams {
            vram_slots: 0,
            ram_slots: 0,
            read_through: 0.5,
            max_inserts: 64,
            prefetch_per_step: 0,
            history_decay: 0.9,
            min_layer_slots: 16,
            defer_spares: 0,
            lrfu_lambda: 0.0,
            admit_window: 0,
            admit_margin: 0.0,
            admit_uses: 0,
            demote: Demote::Ram,
            refill_spares: 0,
            lend_boot: 0,
            policy: VictimPolicy::default(),
            second_chance: 0,
            spare_ring: 0,
            band_slots: 0,
            bounce_slots: 0,
        }
    }
}

/// A routing-frequency profile: per layer, per GLOBAL expert, how often it was routed. The
/// pack-side file `profile/route_profile.json` (format `fn7-route-profile/1`), built by
/// `tools/capture/fn7/route_profile.py` from captured routes.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteProfile {
    /// Where the counts came from.
    pub source: String,
    /// Tokens counted.
    pub tokens: u64,
    /// `counts[layer][global expert]`.
    pub counts: Vec<Vec<u64>>,
}

impl RouteProfile {
    /// Parse a `fn7-route-profile/1` JSON document.
    pub fn from_json(text: &str) -> Result<RouteProfile, TierError> {
        let v: serde_json::Value =
            serde_json::from_str(text).map_err(|e| TierError::Profile(e.to_string()))?;
        let bad = |w: &str| TierError::Profile(w.to_string());
        if v["format"] != "fn7-route-profile/1" {
            return Err(bad("format is not fn7-route-profile/1"));
        }
        let experts = v["experts"].as_u64().ok_or_else(|| bad("no experts"))?;
        let rows = v["counts"].as_array().ok_or_else(|| bad("no counts"))?;
        let mut counts = Vec::with_capacity(rows.len());
        for (l, r) in rows.iter().enumerate() {
            let r = r
                .as_array()
                .ok_or_else(|| bad("counts row is not an array"))?;
            if r.len() as u64 != experts {
                return Err(TierError::Profile(format!(
                    "layer {l}: {} counts, {experts} experts",
                    r.len()
                )));
            }
            counts.push(
                r.iter()
                    .map(|c| c.as_u64().ok_or_else(|| bad("a count is not an integer")))
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
        if v["layers"].as_u64() != Some(counts.len() as u64) {
            return Err(bad("layers does not match the counts rows"));
        }
        Ok(RouteProfile {
            source: v["source"].as_str().unwrap_or("").to_string(),
            tokens: v["tokens"].as_u64().unwrap_or(0),
            counts,
        })
    }
}

/// Which-path counts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PathCounts {
    /// Routed experts served from a VRAM slot.
    pub resident: u64,
    /// Routed experts the kernel read from pinned RAM over UVA (includes `prefetched`).
    pub ram: u64,
    /// Routed experts read from NVMe on demand this step.
    pub nvme: u64,
    /// Of `ram`: the record arrived through a prefetch and this is its first hit.
    pub prefetched: u64,
}

impl PathCounts {
    fn add(&mut self, o: &PathCounts) {
        self.resident += o.resident;
        self.ram += o.ram;
        self.nvme += o.nvme;
        self.prefetched += o.prefetched;
    }
    /// All routed (distinct per layer per step) experts.
    #[must_use]
    pub fn routed(&self) -> u64 {
        self.resident + self.ram + self.nvme
    }
}

/// What one step moved.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MoveCounts {
    /// RAM → VRAM promotions.
    pub inserts: u64,
    /// VRAM → RAM demotions (a promotion's victim).
    pub demotions: u64,
    /// RAM records dropped (their placement became `Fetch`).
    pub ram_drops: u64,
    /// NVMe reads issued by the prefetch.
    pub prefetch_reads: u64,
    /// Layers that read through (no inserts) this step.
    pub read_through_layers: u64,
    /// Bytes copied host → device (promotions).
    pub bytes_h2d: u64,
    /// Bytes copied device → host (demotions).
    pub bytes_d2h: u64,
    /// Bytes read from NVMe (demand + prefetch).
    pub bytes_nvme: u64,
    /// FN17: copies issued under the deferred law and published at this `end_step` (promotions
    /// + demotions of the previous step).
    pub published: u64,
    /// FN17: misses a deferred step could not promote for want of a hole (a demotion makes one
    /// for the next step).
    pub hole_waits: u64,
    /// FN18: misses the admission filter refused (served from RAM, not promoted).
    pub refused: u64,
    /// FN36: victims re-read from NVMe into a refill pool slot (the demotion-free law: each one
    /// replaces a D2H victim copy on the GPU link).
    pub refills: u64,
    /// FN36: refill batches `end_step` had to read itself (the worker did not call
    /// [`ExpertTier::refill`] under the step): the host waited on the NVMe between steps.
    pub refill_late: u64,
    /// FN42: records dropped from a LENT slot (a victim or a reclaim) -- no copy: the record's
    /// shadow RAM slot still holds it.
    pub lent_drops: u64,
    /// FN44b: EMPTY lent slots filled at a prefill pass ([`ExpertTier::fill_lent`]): one H2D
    /// each, the record keeps its RAM slot (inclusive).
    pub lent_fills: u64,
    /// FN54: capped misses the step's promotion BYTE budget left in RAM.
    pub budget_cut: u64,
    /// FN54: misses held by second-chance insertion (a first miss in its window).
    pub sc_held: u64,
    /// FN107 (X1): victims of an EXCLUSIVE lent slot (the P2P pool's under
    /// `--xtier-pool-exclusive`) WRITTEN BACK to RAM through the spare (a D2H each, as a static
    /// slot's demotion; also counted in `demotions`).
    pub excl_writebacks: u64,
}

impl MoveCounts {
    fn add(&mut self, o: &MoveCounts) {
        self.inserts += o.inserts;
        self.demotions += o.demotions;
        self.ram_drops += o.ram_drops;
        self.prefetch_reads += o.prefetch_reads;
        self.read_through_layers += o.read_through_layers;
        self.bytes_h2d += o.bytes_h2d;
        self.bytes_d2h += o.bytes_d2h;
        self.bytes_nvme += o.bytes_nvme;
        self.published += o.published;
        self.hole_waits += o.hole_waits;
        self.refused += o.refused;
        self.refills += o.refills;
        self.refill_late += o.refill_late;
        self.lent_drops += o.lent_drops;
        self.lent_fills += o.lent_fills;
        self.budget_cut += o.budget_cut;
        self.sc_held += o.sc_held;
        self.excl_writebacks += o.excl_writebacks;
    }
}

/// The byte mover the tier drives. [`HostMover`](crate::expert_mover::HostMover) is the host
/// implementation (fake VRAM, real `O_DIRECT` reads) the tests and the replay bench run on; the
/// engine's HIP mover queues the copies on ONE copy stream in call order, so a demotion's D2H out
/// of slot `s` is ordered before the promotion's H2D into `s`.
pub trait ExpertMover {
    /// Read records from NVMe into RAM slots: one aligned `O_DIRECT` read per record, all in
    /// flight at once up to the mover's queue depth. Complete when it returns.
    fn nvme_to_ram(&mut self, reads: &[(u64, u32)]) -> Result<(), TierError>;
    /// FN36: read victims' records from NVMe into refill pool slots that NO placement points at
    /// and NO queued copy reads or writes (the demotion-free law), so no drain is needed. Complete
    /// when it returns. The default is [`Self::nvme_to_ram`].
    fn nvme_refill(&mut self, reads: &[(u64, u32)]) -> Result<(), TierError> {
        self.nvme_to_ram(reads)
    }
    /// Queue a RAM slot → VRAM slot copy (one stride).
    fn ram_to_vram(&mut self, ram: u32, vram: u32) -> Result<(), TierError>;
    /// Queue a VRAM slot → RAM slot copy (one stride).
    fn vram_to_ram(&mut self, vram: u32, ram: u32) -> Result<(), TierError>;
    /// `layer`'s copies are all queued (called by `end_step` after each layer that queued any):
    /// an asynchronous mover issues them and records the layer's "landed" mark here, so the copies
    /// of layer 0 are on the link while the host is still ranking layer 1's victims (FN8).
    fn end_layer(&mut self, _layer: u32) -> Result<(), TierError> {
        Ok(())
    }
    /// FN17: the law of the copies queued from now until the next call -- `true` = deferred
    /// (nothing reads them before their publication, which the consumer's fence orders; the
    /// device mover's per-layer gate stays free), `false` = FN8's (published at once, gated per
    /// layer). The tier calls it at the start of every `end_step`.
    fn set_deferred(&mut self, _on: bool) {}
    /// Every queued copy is issued, in order. A synchronous mover (the host ones) has completed
    /// them; the device mover has not — its device consumers order themselves by the per-layer
    /// gate and its host writes drain first ([`crate::device_mover`], the ordering law).
    fn flush(&mut self) -> Result<(), TierError>;
}

const EMPTY: u32 = u32::MAX;

/// FN17: one copy of the deferred law, between its issue and its publication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pend {
    /// Record `m` (placed `Ram(r)`) copied into the HOLE `h` of its layer.
    Promote { m: u32, r: u32, h: u32 },
    /// Record `v` (placed `Vram(vs)`) copied into the free RAM spare `sp`.
    Demote { v: u32, vs: u32, sp: u32 },
}

/// FN36: one swap of the demotion-free law, between its choice and its publication: record `m`
/// (placed `Ram(r)`) is promoted into slot `v`, whose record `w` is re-read from NVMe into the
/// pool slot `sp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Swap {
    m: u32,
    r: u32,
    v: u32,
    w: u32,
    sp: u32,
}

#[derive(Debug, Clone, Copy)]
struct Slot {
    record: u32,
    stamp: u64,
}

/// One rank's expert tier (see the module docs).
#[derive(Debug)]
pub struct ExpertTier {
    rank: u32,
    first_expert: u32,
    experts: u32,
    layers: u32,
    stride: u64,
    sub: [(u64, u64); 6],
    params: TierParams,
    vram_base: u64,
    ram_base: u64,
    /// `(first slot, slots)` of each layer's VRAM partition.
    layer_slots: Vec<(u32, u32)>,
    place: Vec<Placement>,
    vslot: Vec<Slot>,
    /// RAM slots `0..ram_slots` plus the spare at index `ram_slots` initially.
    rslot: Vec<Slot>,
    spare: u32,
    /// FN54b: the extra swap spares, oldest first (empty = FN8's one spare).
    spare_q: std::collections::VecDeque<u32>,
    /// FN59b: the cheap victim planning (`--xtier-plan fast|async`): each candidate slot's score
    /// is computed ONCE and only the first `misses` keys are selected, instead of a full sort that
    /// re-derives two LRFU scores (`powi`) per comparison. The same total order, so the same
    /// victims (placement-identical by construction). `false` = today's full sort.
    fast_plan: bool,
    ram_free: Vec<u32>,
    /// FN17: free RAM slots for the deferred demotions (`ram_slots + 1 ..`).
    rspares: Vec<u32>,
    /// FN17: the copies issued by the last deferred `end_step`, published by the next one.
    pending: Vec<Pend>,
    /// FN36: the refill pool's free slots, the old RAM slots of the last publication (free once
    /// the step after it has read past its gates), the swaps awaiting publication, and the NVMe
    /// reads not yet issued.
    pool: Vec<u32>,
    retiring: Vec<u32>,
    swaps: Vec<Swap>,
    refill_reads: Vec<(u64, u32)>,
    now: u64,
    touched: Vec<u64>,
    step_routes: Vec<Vec<u32>>,
    score: Vec<f32>,
    prefetched: Vec<bool>,
    per_layer: Vec<PathCounts>,
    /// FN51f-b (the residency view): H2D bytes promoted into VRAM a layer (inserts + fills).
    layer_h2d: Vec<u64>,
    total: PathCounts,
    step: PathCounts,
    moves_total: MoveCounts,
    moves_step: MoveCounts,
    steps: u64,
    profile: Option<(String, u64)>,
    /// FN18: LRFU score and the step it was last updated at, per record.
    crf: Vec<f64>,
    crf_at: Vec<u64>,
    /// FN18: admission counts per record (halved every `admit_window` steps).
    freq: Vec<u32>,
    /// FN39: steps routed since the record entered RAM (the admission-uses rule).
    ram_uses: Vec<u32>,
    /// FN42: THE LENDING LAW -- slots `base_slots ..` are LENDABLE: memory another client owns
    /// (the KV pool's free tail) that the tier may cache records in while it is lent. A lent slot
    /// is INCLUSIVE: the record keeps its RAM slot (its `shadow`), so a reclaim is a placement
    /// flip with no copy. `lend_addr[i]` is lendable slot `base_slots + i`'s device address.
    base_slots: u32,
    lend_addr: Vec<u64>,
    lend_on: Vec<bool>,
    shadow: Vec<u32>,
    /// FN107 (X1): lendable slots made EXCLUSIVE ([`Self::make_exclusive`]): never reclaimed, so
    /// their records keep NO RAM shadow -- a victim is written back through the spare like a
    /// static slot's (FN8's swap), and they count as static for the RAM L2's complement.
    excl: Vec<bool>,
    /// FN42: each layer's lent slots (beside its static partition).
    layer_lent: Vec<Vec<u32>>,
    /// FN42: the warm profile's route count per record (the lend's layer choice at boot).
    prof_count: Vec<u64>,
    /// LR-H: the decayed SERVED prefill route count per record (one per staged pass that routed
    /// it, x [`PF_SEEN_DECAY`] a pass): the `served` ring order key (host-only; no placement or
    /// byte reads it).
    pf_seen: Vec<f32>,
    /// FN48: the reuse estimate per record -- last use step, EWMA gap, uses seen.
    ru_last: Vec<u64>,
    ru_gap: Vec<f64>,
    ru_n: Vec<u32>,
    /// FN48: the table policy's decayed use count and class at the last use.
    tb_lc: Vec<f64>,
    tb_cls: Vec<u8>,
    /// FN54: the promotion budget of the steps from now on, in records (`None` = no budget:
    /// today's law), set per step like the insert cap ([`Self::set_promote_budget`]).
    budget: Option<u32>,
    /// FN54: this step's per-layer promotion quota under the budget (`u32::MAX` = unbounded).
    quota: Vec<u32>,
    /// FN103: the admission planner's per-layer insert caps for the next `end_step` (empty =
    /// none: today's law, byte for byte; each layer's cap is `min(max_inserts, quota, this)`).
    layer_cap: Vec<u32>,
    /// FN54: the step of each record's last RAM-placed route (0 = never) and the step at which
    /// it last qualified for promotion under second-chance insertion.
    sc_last: Vec<u64>,
    sc_ok: Vec<u64>,
    /// FN53b: the valve band's first slot, its OPEN slots (segments close from the top), each
    /// segment's UVA address (0 = closed), the bounce's first slot / address / free slots, the
    /// records placed in the bounce this step, the layer-ahead reads not yet routed, the pass
    /// kind, and the counters.
    band_lo: u32,
    band_open: u32,
    band_addr: Vec<u64>,
    bounce_lo: u32,
    bounce_addr: u64,
    bounced: Vec<(u32, u32)>,
    bounce_ready: Vec<(u32, u32)>,
    pass_prefill: bool,
    /// FN53c: the pass that follows runs the device miss flag (its `Fetch` routes are served by
    /// the bounce during the step and stay `Fetch`).
    pass_xmiss: bool,
    valve: ValveCounts,
    /// FN104: the scheduler's PROTECTION set (empty = off, today's law): a protected record's
    /// victim value is raised by [`SCHED_PROTECT`], so it leaves only after every unprotected one.
    sched_prot: Vec<bool>,
}

/// FN104: the victim-value lift of a record the scheduler protects (above any LRFU score).
pub const SCHED_PROTECT: f64 = 1e9;

impl ExpertTier {
    /// Build a rank's tier over its `experts.xtier` geometry. `vram_base` is the device address of
    /// the slot arena (`vram_slots · stride` bytes), `ram_base` the UVA address of the RAM slab
    /// (`(ram_slots + 1) · stride` bytes, the last slot the swap spare). The warm start places the
    /// profile's most-routed experts in VRAM (a global water-fill over the layers, at least
    /// `min_layer_slots` a layer), then RAM, the rest `Fetch`. Without a profile the split is
    /// uniform and the warm order is expert id. Nothing is copied here: [`Self::warm_plan`] lists
    /// the reads the boot must do.
    pub fn new(
        h: &XtierHeader,
        params: TierParams,
        profile: Option<&RouteProfile>,
        vram_base: u64,
        ram_base: u64,
    ) -> Result<ExpertTier, TierError> {
        let (layers, experts) = (h.layers, h.experts);
        let records = u64::from(layers) * u64::from(experts);
        if u64::from(params.vram_slots) > records {
            return Err(TierError::Config(format!(
                "{} VRAM slots for {records} records",
                params.vram_slots
            )));
        }
        if !(0.0..=1.0).contains(&params.read_through) || params.max_inserts == 0 {
            return Err(TierError::Config(format!(
                "read_through {} / max_inserts {}",
                params.read_through, params.max_inserts
            )));
        }
        let floor = params.min_layer_slots.min(experts);
        if params.vram_slots < floor * layers && params.vram_slots > 0 {
            return Err(TierError::Config(format!(
                "{} VRAM slots cannot give {layers} layers {floor} each",
                params.vram_slots
            )));
        }
        // The global ranking: (count desc, layer asc, expert asc) over this rank's records.
        let mut ranked: Vec<(u64, u32, u32)> = Vec::with_capacity(records as usize);
        for l in 0..layers {
            for e in 0..experts {
                let c = profile
                    .and_then(|p| p.counts.get(l as usize))
                    .and_then(|row| row.get((h.first_expert + e) as usize))
                    .copied()
                    .unwrap_or(0);
                ranked.push((c, l, e));
            }
        }
        if let Some(p) = profile {
            if p.counts.len() != layers as usize {
                return Err(TierError::Profile(format!(
                    "{} profile layers for a {layers}-layer tier",
                    p.counts.len()
                )));
            }
        }
        let mut prof_count = vec![0u64; records as usize];
        for &(c, l, e) in &ranked {
            prof_count[(l * experts + e) as usize] = c;
        }
        ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        // Water-fill: every layer gets its floor, the rest of the slots go down the ranking.
        let mut per = vec![0u32; layers as usize];
        let mut in_vram = vec![false; records as usize];
        if params.vram_slots > 0 {
            if profile.is_some() {
                let mut taken = vec![0u32; layers as usize];
                // Floors first, each layer's own most-routed experts.
                for &(_, l, e) in &ranked {
                    if taken[l as usize] < floor {
                        taken[l as usize] += 1;
                        in_vram[(l * experts + e) as usize] = true;
                    }
                }
                let mut left = params.vram_slots - floor * layers;
                for &(_, l, e) in &ranked {
                    if left == 0 {
                        break;
                    }
                    let i = (l * experts + e) as usize;
                    if !in_vram[i] {
                        in_vram[i] = true;
                        taken[l as usize] += 1;
                        left -= 1;
                    }
                }
                per = taken;
            } else {
                let base = params.vram_slots / layers;
                let extra = params.vram_slots % layers;
                for l in 0..layers {
                    per[l as usize] = base + u32::from(l < extra);
                    for e in 0..per[l as usize].min(experts) {
                        in_vram[(l * experts + e) as usize] = true;
                    }
                }
            }
        }
        let mut layer_slots = Vec::with_capacity(layers as usize);
        let mut acc = 0u32;
        for &n in &per {
            layer_slots.push((acc, n));
            acc += n;
        }
        let mut place = vec![Placement::Fetch; records as usize];
        let mut vslot = vec![
            Slot {
                record: EMPTY,
                stamp: 0
            };
            params.vram_slots as usize
        ];
        let mut fill = per.iter().map(|_| 0u32).collect::<Vec<_>>();
        for l in 0..layers {
            for e in 0..experts {
                let i = l * experts + e;
                if in_vram[i as usize] {
                    let s = layer_slots[l as usize].0 + fill[l as usize];
                    fill[l as usize] += 1;
                    vslot[s as usize].record = i;
                    place[i as usize] = Placement::Vram(s);
                }
            }
        }
        // RAM: the next-ranked records not in VRAM, in ranking order.
        let mut rslot = vec![
            Slot {
                record: EMPTY,
                stamp: 0
            };
            params.ram_slots as usize
                + 1
                + params.defer_spares as usize
                + params.refill_spares as usize
                + params.spare_ring as usize
                + params.band_slots as usize
                + params.bounce_slots as usize
        ];
        if params.band_slots % VALVE_SEG != 0 {
            return Err(TierError::Config(format!(
                "valve band {} slots is not a multiple of {VALVE_SEG}",
                params.band_slots
            )));
        }
        // FN53b: the L2's slots in fill order -- the slab's, then the valve band's (past every
        // spare); without a band exactly `0..ram_slots`
        let band_lo = Self::slab_slots(&params);
        let l2: Vec<u32> = (0..params.ram_slots)
            .chain(band_lo..band_lo + params.band_slots)
            .collect();
        let mut r = 0u32;
        let ranking: Vec<(u64, u32, u32)> = if profile.is_some() {
            ranked
        } else {
            let mut v: Vec<_> = (0..layers)
                .flat_map(|l| (0..experts).map(move |e| (0u64, l, e)))
                .collect();
            v.sort_by(|a, b| a.2.cmp(&b.2).then(a.1.cmp(&b.1)));
            v
        };
        for &(_, l, e) in &ranking {
            if r as usize >= l2.len() {
                break;
            }
            let i = l * experts + e;
            if place[i as usize] == Placement::Fetch {
                let s = l2[r as usize];
                rslot[s as usize].record = i;
                place[i as usize] = Placement::Ram(s);
                r += 1;
            }
        }
        let ram_free = l2[r as usize..].iter().rev().copied().collect();
        let bounce_lo = band_lo + params.band_slots;
        let base_slots = params.vram_slots;
        let lend_boot = params.lend_boot;
        let mut t = ExpertTier {
            rank: h.rank,
            first_expert: h.first_expert,
            experts,
            layers,
            stride: h.stride,
            sub: h.sub,
            vram_base,
            ram_base,
            layer_slots,
            place,
            vslot,
            rslot,
            spare: params.ram_slots,
            spare_q: Self::ring_slots(&params),
            fast_plan: false,
            ram_free,
            rspares: (params.ram_slots + 1..params.ram_slots + 1 + params.defer_spares)
                .rev()
                .collect(),
            pending: Vec::new(),
            pool: Self::pool_slots(&params),
            retiring: Vec::new(),
            swaps: Vec::new(),
            refill_reads: Vec::new(),
            now: 0,
            touched: vec![0; records as usize],
            step_routes: vec![Vec::new(); layers as usize],
            score: vec![0.0; records as usize],
            prefetched: vec![false; records as usize],
            per_layer: vec![PathCounts::default(); layers as usize],
            layer_h2d: vec![0; layers as usize],
            total: PathCounts::default(),
            step: PathCounts::default(),
            moves_total: MoveCounts::default(),
            moves_step: MoveCounts::default(),
            steps: 0,
            profile: profile.map(|p| (p.source.clone(), p.tokens)),
            crf: vec![0.0; records as usize],
            crf_at: vec![0; records as usize],
            freq: vec![0; records as usize],
            ram_uses: vec![0; records as usize],
            params,
            base_slots,
            lend_addr: Vec::new(),
            lend_on: Vec::new(),
            shadow: Vec::new(),
            excl: Vec::new(),
            layer_lent: vec![Vec::new(); layers as usize],
            prof_count,
            pf_seen: vec![0.0; records as usize],
            ru_last: vec![0; records as usize],
            ru_gap: vec![0.0; records as usize],
            ru_n: vec![0; records as usize],
            tb_lc: vec![0.0; records as usize],
            tb_cls: vec![0; records as usize],
            budget: None,
            quota: vec![u32::MAX; layers as usize],
            layer_cap: Vec::new(),
            sc_last: vec![0; records as usize],
            sc_ok: vec![0; records as usize],
            band_lo,
            band_open: 0,
            band_addr: Vec::new(),
            bounce_lo,
            bounce_addr: 0,
            bounced: Vec::new(),
            bounce_ready: Vec::new(),
            pass_xmiss: false,
            pass_prefill: false,
            valve: ValveCounts::default(),
            sched_prot: Vec::new(),
        };
        t.band_open = t.params.band_slots;
        t.band_addr = (0..t.params.band_slots / VALVE_SEG)
            .map(|k| ram_base + (u64::from(band_lo) + u64::from(k * VALVE_SEG)) * h.stride)
            .collect();
        t.bounce_addr = ram_base + u64::from(bounce_lo) * h.stride;
        if lend_boot > 0 {
            let addrs: Vec<u64> = (0..u64::from(lend_boot))
                .map(|i| vram_base + (u64::from(base_slots) + i) * h.stride)
                .collect();
            let first = t.register_lendable(&addrs)?;
            let ids: Vec<u32> = (first..first + lend_boot).collect();
            t.lend(&ids)?;
        }
        Ok(t)
    }

    /// FN42: register lendable slots at `addrs` (each one stride of memory another client owns);
    /// they start RECLAIMED (not lent). Returns the first new slot id (`base_slots + i`). Only the
    /// FN8 law lends (the deferred and demotion-free laws keep copies in flight across steps).
    pub fn register_lendable(&mut self, addrs: &[u64]) -> Result<u32, TierError> {
        if self.params.defer_spares > 0 || self.demotion_free_law() {
            return Err(TierError::Config(
                "lent slots run under FN8's law only (no --xtier-defer, --xtier-demote ram)".into(),
            ));
        }
        let first = self.vslot.len() as u32;
        for &a in addrs {
            self.lend_addr.push(a);
            self.lend_on.push(false);
            self.shadow.push(EMPTY);
            self.excl.push(false);
            self.vslot.push(Slot {
                record: EMPTY,
                stamp: 0,
            });
        }
        Ok(first)
    }

    /// FN42: lend slots `ids` to the tier. Each joins the layer of the next most valuable record
    /// not in VRAM (RAM records by LRFU score now, then the warm profile's count, then index; one
    /// record per slot), EMPTY: the normal insert law fills it (an empty slot is every layer's
    /// first victim). Ids already lent are skipped. Returns the slots lent.
    pub fn lend(&mut self, ids: &[u32]) -> Result<usize, TierError> {
        if let Some(&bad) = ids
            .iter()
            .find(|&&s| s < self.base_slots || (s - self.base_slots) as usize >= self.lend_on.len())
        {
            return Err(TierError::Config(format!(
                "lend: slot {bad} is not lendable"
            )));
        }
        let want: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|&s| !self.lend_on[(s - self.base_slots) as usize])
            .collect();
        if want.is_empty() {
            return Ok(0);
        }
        let mut cand: Vec<(f64, u64, u32)> = self
            .place
            .iter()
            .enumerate()
            .filter(|(_, p)| matches!(p, Placement::Ram(_)))
            .map(|(i, _)| {
                let c = if self.params.lrfu_lambda > 0.0 {
                    self.crf_now(i as u32)
                } else {
                    self.score[i] as f64
                };
                (c, self.prof_count[i], i as u32)
            })
            .collect();
        cand.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.cmp(&a.1)).then(a.2.cmp(&b.2)));
        // FN108b: a layer's EMPTY lent slots (an earlier lend not filled yet) will take that
        // layer's RAM records at the next fill -- skip as many of its candidates, or the layer
        // gets more lent slots than records to fill them (rank 1's 2 empty pool slots: the
        // exclusive pool's refill then left 2 records on the drive, an incomplete tier)
        let mut reserved: Vec<u32> = self
            .layer_lent
            .iter()
            .map(|ll| {
                ll.iter()
                    .filter(|&&s| self.vslot[s as usize].record == EMPTY && !self.is_exclusive(s))
                    .count() as u32
            })
            .collect();
        let mut it = cand.iter().filter(|&&(_, _, i)| {
            let r = &mut reserved[(i / self.experts) as usize];
            if *r > 0 {
                *r -= 1;
                false
            } else {
                true
            }
        });
        for &s in &want {
            // no RAM record left: the slot joins the layers round-robin
            let l = it
                .next()
                .map_or(s % self.layers, |&(_, _, i)| i / self.experts);
            self.layer_lent[l as usize].push(s);
            self.lend_on[(s - self.base_slots) as usize] = true;
        }
        Ok(want.len())
    }

    /// FN42: RECLAIM slots `ids` (their owner needs the memory): each lent slot's record, if any,
    /// falls back to its shadow RAM slot (a placement flip -- no copy), in LRFU victim order
    /// (lowest score first). The caller must have drained the copy stream (no promotion into the
    /// slot in flight) and must rebuild the xptr tables before the next launch (every step does).
    /// Returns the records dropped.
    pub fn reclaim(&mut self, ids: &[u32]) -> usize {
        let mut v: Vec<(f64, u32)> = ids
            .iter()
            .copied()
            .filter(|&s| {
                s >= self.base_slots
                    && self
                        .lend_on
                        .get((s - self.base_slots) as usize)
                        .copied()
                        .unwrap_or(false)
                    // FN107: an exclusive slot is never reclaimed (its record has no shadow)
                    && !self.excl[(s - self.base_slots) as usize]
            })
            .map(|s| {
                let r = self.vslot[s as usize].record;
                (if r == EMPTY { -1.0 } else { self.crf_now(r) }, s)
            })
            .collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut dropped = 0;
        for (_, s) in v {
            let i = (s - self.base_slots) as usize;
            let r = self.vslot[s as usize].record;
            if r != EMPTY {
                self.place[r as usize] = Placement::Ram(self.shadow[i]);
                self.ram_uses[r as usize] = 0;
                dropped += 1;
            }
            self.vslot[s as usize] = Slot {
                record: EMPTY,
                stamp: 0,
            };
            self.shadow[i] = EMPTY;
            self.lend_on[i] = false;
            for ll in &mut self.layer_lent {
                ll.retain(|&x| x != s);
            }
        }
        self.moves_total.lent_drops += dropped as u64;
        dropped
    }

    /// LR-H: the ring's fill order under `key` -- [`StageKey::Profile`] = [`Self::stage_order`]
    /// (today, byte for byte); [`StageKey::Served`] = every RAM-resident record of the layer by
    /// its decayed served prefill route count ([`Self::note_prefill_routes`]) first, then the warm
    /// profile's count, then the LRFU (or LRU) score, then the expert index. Both orders hold the
    /// SAME records (the layer's RAM set) -- only their sequence differs.
    #[must_use]
    pub fn stage_order_by(&self, layer: u32, key: StageKey) -> Vec<(u32, u32)> {
        if key == StageKey::Profile {
            return self.stage_order(layer);
        }
        let n = self.experts as usize;
        let lo = layer as usize * n;
        let mut v: Vec<(f32, u64, f64, u32, u32)> = (0..n)
            .filter_map(|e| match self.place[lo + e] {
                Placement::Ram(r) => {
                    let i = (lo + e) as u32;
                    let c = if self.params.lrfu_lambda > 0.0 {
                        self.crf_now(i)
                    } else {
                        f64::from(self.score[i as usize])
                    };
                    Some((
                        self.pf_seen[i as usize],
                        self.prof_count[i as usize],
                        c,
                        e as u32,
                        r,
                    ))
                }
                _ => None,
            })
            .collect();
        v.sort_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then(b.1.cmp(&a.1))
                .then(b.2.total_cmp(&a.2))
                .then(a.3.cmp(&b.3))
        });
        v.into_iter().map(|(_, _, _, e, r)| (e, r)).collect()
    }

    /// LR-H: a staged prefill pass's routes of `layer` (`routed`: GLOBAL expert ids, any rank's;
    /// another rank's are skipped, repeats count once) bump the `served` key's count by one each.
    /// Call [`Self::decay_prefill_seen`] once per staged pass before its layers' notes.
    pub fn note_prefill_routes(&mut self, layer: u32, routed: &[u32]) {
        let seen = crate::stage_ring::routed_seen(self, routed);
        self.note_prefill_seen(layer, &seen);
    }

    /// [`Self::note_prefill_routes`] from the layer's `stage_ring::routed_seen` map.
    pub fn note_prefill_seen(&mut self, layer: u32, seen: &[bool]) {
        let lo = (layer * self.experts) as usize;
        for (e, &r) in seen.iter().enumerate().take(self.experts as usize) {
            if r {
                self.pf_seen[lo + e] += 1.0;
            }
        }
    }

    /// LR-H: one staged pass's decay of the `served` key ([`PF_SEEN_DECAY`]).
    pub fn decay_prefill_seen(&mut self) {
        for x in &mut self.pf_seen {
            *x *= PF_SEEN_DECAY;
        }
    }

    /// LR-H: the `served` key of (`layer`, local expert `e`).
    #[must_use]
    pub fn prefill_seen(&self, layer: u32, e: u32) -> f32 {
        self.pf_seen[(layer * self.experts + e) as usize]
    }

    /// FN43: THE RING'S FILL ORDER for `layer` -- every RAM-resident record of the layer as
    /// `(local expert, RAM slot)`, most likely routed first: the warm profile's route count (the
    /// long-run frequency a wide prefill chunk samples), then the LRFU score now (or the LRU
    /// score), then the expert index. A prefill chunk stages a prefix of this list (a layer's
    /// routes are unknown until its router runs).
    #[must_use]
    pub fn stage_order(&self, layer: u32) -> Vec<(u32, u32)> {
        let n = self.experts as usize;
        let lo = layer as usize * n;
        let mut v: Vec<(u64, f64, u32, u32)> = (0..n)
            .filter_map(|e| match self.place[lo + e] {
                Placement::Ram(r) => {
                    let i = (lo + e) as u32;
                    let c = if self.params.lrfu_lambda > 0.0 {
                        self.crf_now(i)
                    } else {
                        f64::from(self.score[i as usize])
                    };
                    Some((self.prof_count[i as usize], c, e as u32, r))
                }
                _ => None,
            })
            .collect();
        v.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.total_cmp(&a.1)).then(a.2.cmp(&b.2)));
        v.into_iter().map(|(_, _, e, r)| (e, r)).collect()
    }

    /// FN44b: the EMPTY lent slots (a lend hands them over empty; a prefill chunk's layers are
    /// read-through, so nothing fills them until decode misses do).
    #[must_use]
    pub fn lent_empty(&self) -> u32 {
        self.layer_lent
            .iter()
            .flatten()
            .filter(|&&s| self.vslot[s as usize].record == EMPTY && !self.is_exclusive(s))
            .count() as u32
    }

    /// FN44b: FILL every EMPTY lent slot now, layer by layer: each takes the next record of its
    /// layer's [`Self::stage_order`] (the warm profile's route count first -- what a prefill chunk
    /// samples -- then the LRFU score) that is still RAM-placed; one H2D per slot, the record
    /// KEEPS its RAM slot as the shadow (a lent slot is inclusive), each layer's copies close with
    /// `end_layer` (law 3: the consumer's gate of that layer waits for them). The caller must have
    /// fenced the copy stream after every earlier use of the slots' memory (`DeviceMover::after`)
    /// and must build the step's tables after it. Returns the slots filled.
    pub fn fill_lent(&mut self, mover: &mut dyn ExpertMover) -> Result<usize, TierError> {
        let mut filled = 0usize;
        for l in 0..self.layers {
            let empty: Vec<u32> = self.layer_lent[l as usize]
                .iter()
                .copied()
                .filter(|&s| self.vslot[s as usize].record == EMPTY && !self.is_exclusive(s))
                .collect();
            if empty.is_empty() {
                continue;
            }
            let order = self.stage_order(l);
            let lo = l * self.experts;
            let mut any = false;
            for (&s, &(e, r)) in empty.iter().zip(order.iter()) {
                let m = lo + e;
                let i = self
                    .lent_slot(s)
                    .ok_or_else(|| TierError::Config(format!("fill: slot {s} is not lent")))?;
                debug_assert_eq!(self.place[m as usize], Placement::Ram(r));
                mover.ram_to_vram(r, s)?;
                self.shadow[i] = r;
                self.vslot[s as usize] = Slot {
                    record: m,
                    stamp: self.now,
                };
                self.place[m as usize] = Placement::Vram(s);
                self.prefetched[m as usize] = false;
                self.moves_total.lent_fills += 1;
                self.moves_total.bytes_h2d += self.stride;
                self.layer_h2d[l as usize] += self.stride;
                filled += 1;
                any = true;
            }
            if any {
                mover.end_layer(l)?;
            }
        }
        if filled > 0 {
            mover.flush()?;
        }
        Ok(filled)
    }

    /// FN51f-c: THE FILL ON LEND for the elastic region -- the EMPTY lent slots among `ids` (a
    /// segment just handed back by the KV) each take the HOTTEST record of its layer still
    /// RAM-placed, by the LRFU score NOW (the decode phase's recency, not the warm profile a
    /// prefill chunk samples; `lend` already picked each slot's layer by the same global LRFU
    /// order), then the profile count, then the index. One H2D a slot, the record keeps its RAM
    /// slot as the shadow (inclusive, as [`Self::fill_lent`]), each layer's copies close with
    /// `end_layer` (law 3). Bounded by `ids` (the caller passes one tick's segments). The caller
    /// must have fenced the copy stream after every earlier use of the slots' memory
    /// (`DeviceMover::after`) and builds the step's tables after it. Returns the slots filled.
    pub fn fill_lent_hot(
        &mut self,
        ids: &[u32],
        mover: &mut dyn ExpertMover,
    ) -> Result<usize, TierError> {
        let mut filled = 0usize;
        for l in 0..self.layers {
            let empty: Vec<u32> = self.layer_lent[l as usize]
                .iter()
                .copied()
                .filter(|s| {
                    ids.contains(s)
                        && self.vslot[*s as usize].record == EMPTY
                        && !self.is_exclusive(*s)
                })
                .collect();
            if empty.is_empty() {
                continue;
            }
            let n = self.experts as usize;
            let lo = l * self.experts;
            let mut order: Vec<(f64, u64, u32, u32)> = (0..n)
                .filter_map(|e| match self.place[lo as usize + e] {
                    Placement::Ram(r) => {
                        let i = lo + e as u32;
                        let c = if self.params.lrfu_lambda > 0.0 {
                            self.crf_now(i)
                        } else {
                            f64::from(self.score[i as usize])
                        };
                        Some((c, self.prof_count[i as usize], e as u32, r))
                    }
                    _ => None,
                })
                .collect();
            order.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.cmp(&a.1)).then(a.2.cmp(&b.2)));
            let mut any = false;
            for (&s, &(_, _, e, r)) in empty.iter().zip(order.iter()) {
                let m = lo + e;
                let i = self
                    .lent_slot(s)
                    .ok_or_else(|| TierError::Config(format!("fill: slot {s} is not lent")))?;
                mover.ram_to_vram(r, s)?;
                self.shadow[i] = r;
                self.vslot[s as usize] = Slot {
                    record: m,
                    stamp: self.now,
                };
                self.place[m as usize] = Placement::Vram(s);
                self.prefetched[m as usize] = false;
                self.moves_total.lent_fills += 1;
                self.moves_total.bytes_h2d += self.stride;
                self.layer_h2d[l as usize] += self.stride;
                filled += 1;
                any = true;
            }
            if any {
                mover.end_layer(l)?;
            }
        }
        if filled > 0 {
            mover.flush()?;
        }
        Ok(filled)
    }

    /// FN43: the local experts this rank owns per layer and the first one's global id.
    #[must_use]
    pub fn experts_geometry(&self) -> (u32, u32) {
        (self.first_expert, self.experts)
    }

    /// FN51f: whether lendable slot `s` is lent now.
    #[must_use]
    pub fn is_lent(&self, s: u32) -> bool {
        s >= self.base_slots
            && self
                .lend_on
                .get((s - self.base_slots) as usize)
                .copied()
                .unwrap_or(false)
    }

    /// FN42: `(lent, lendable)` slot counts.
    #[must_use]
    pub fn lent(&self) -> (u32, u32) {
        (
            self.lend_on.iter().filter(|&&b| b).count() as u32,
            self.lend_on.len() as u32,
        )
    }

    /// FN42: the static partition's slot count (`vram_slots`); lendable slot ids start here.
    #[must_use]
    pub fn base_slots(&self) -> u32 {
        self.base_slots
    }

    /// FN104: scale the LRFU scores of `recs` by `factor` (decayed to this step first): the
    /// FINISH-DECAY law (`factor` < 1 for the records only a finished sequence used). Placement
    /// only; a no-op without LRFU.
    pub fn sched_scale(&mut self, recs: &[u32], factor: f64) {
        if self.params.lrfu_lambda <= 0.0 {
            return;
        }
        for &i in recs {
            let c = self.crf_now(i);
            self.crf[i as usize] = c * factor;
            self.crf_at[i as usize] = self.now;
        }
    }

    /// FN104: add `add` to record `i`'s LRFU score (the ADMISSION PRIOR). Placement only.
    pub fn sched_bump(&mut self, i: u32, add: f64) {
        if self.params.lrfu_lambda <= 0.0 {
            return;
        }
        let c = self.crf_now(i);
        self.crf[i as usize] = c + add;
        self.crf_at[i as usize] = self.now;
    }

    /// FN104: the record's current victim value (the LRFU score decayed to now, + the protection
    /// lift); `None` without LRFU.
    #[must_use]
    pub fn sched_value(&self, i: u32) -> Option<f64> {
        (self.params.lrfu_lambda > 0.0).then(|| self.vscore(i))
    }

    /// FN104: the scheduler's protection set (`prot[record]`; empty = off). Placement only.
    pub fn set_sched_protect(&mut self, prot: Vec<bool>) {
        debug_assert!(prot.is_empty() || prot.len() == self.place.len());
        self.sched_prot = prot;
    }

    /// FN104: take the protection set back (to update it in place).
    pub fn take_sched_protect(&mut self) -> Vec<bool> {
        std::mem::take(&mut self.sched_prot)
    }

    /// FN104: the PRE-WARM law (admission prior with copies): promote each RAM-placed record of
    /// `cands` (one layer's, most valuable first) into a slot of its layer whose record's victim
    /// value is below `dead` (or empty), never a record routed this step, at most `budget`
    /// copies. FN8's swap (D2H victim through the spare, H2D the record); a lent slot drops its
    /// victim to its shadow. Only under FN8's immediate law (no deferral, no demotion-free).
    /// Returns the copies made.
    pub fn sched_prewarm(
        &mut self,
        layer: u32,
        cands: &[u32],
        dead: f64,
        budget: usize,
        mover: &mut dyn ExpertMover,
    ) -> Result<usize, TierError> {
        if self.params.defer_spares > 0 || self.demotion_free() || budget == 0 {
            return Ok(0);
        }
        let (base, n0) = self.layer_slots[layer as usize];
        let lent = self.layer_lent[layer as usize].clone();
        let mut keys: Vec<(f64, u32)> = (base..base + n0)
            .chain(lent.iter().copied())
            .filter_map(|s| {
                let r = self.vslot[s as usize].record;
                if r == EMPTY {
                    return Some((f64::NEG_INFINITY, s));
                }
                if self.touched[r as usize] == self.now {
                    return None;
                }
                let v = self.vscore(r);
                (v < dead).then_some((v, s))
            })
            .collect();
        keys.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut made = 0;
        let mut vic = keys.into_iter();
        for &m in cands {
            if made >= budget {
                break;
            }
            let Placement::Ram(r) = self.place[m as usize] else {
                continue;
            };
            let Some((_, v)) = vic.next() else { break };
            let victim = self.vslot[v as usize].record;
            if let Some(i) = self.lent_slot(v) {
                if victim != EMPTY {
                    self.place[victim as usize] = Placement::Ram(self.shadow[i]);
                    self.ram_uses[victim as usize] = 0;
                    self.moves_step.lent_drops += 1;
                }
                mover.ram_to_vram(r, v)?;
                self.shadow[i] = r;
            } else {
                if victim != EMPTY {
                    if self.is_exclusive(v) {
                        self.moves_step.excl_writebacks += 1;
                    }
                    let sp = self.spare;
                    mover.vram_to_ram(v, sp)?;
                    self.rslot[sp as usize] = Slot {
                        record: victim,
                        stamp: self.vslot[v as usize].stamp,
                    };
                    self.place[victim as usize] = Placement::Ram(sp);
                    self.ram_uses[victim as usize] = 0;
                    self.moves_step.demotions += 1;
                    self.moves_step.bytes_d2h += self.stride;
                    mover.ram_to_vram(r, v)?;
                    if let Some(next) = self.spare_q.pop_front() {
                        self.spare_q.push_back(r);
                        self.spare = next;
                    } else {
                        self.spare = r;
                    }
                } else {
                    mover.ram_to_vram(r, v)?;
                    self.ram_free.push(r);
                }
                self.rslot[r as usize] = Slot {
                    record: EMPTY,
                    stamp: 0,
                };
            }
            self.vslot[v as usize] = Slot {
                record: m,
                stamp: self.now,
            };
            self.place[m as usize] = Placement::Vram(v);
            self.prefetched[m as usize] = false;
            self.moves_step.inserts += 1;
            self.moves_step.bytes_h2d += self.stride;
            self.layer_h2d[layer as usize] += self.stride;
            made += 1;
        }
        mover.end_layer(layer)?;
        Ok(made)
    }

    /// FN42: whether slot `s` is an INCLUSIVE lent slot (its record keeps a RAM shadow). FN107:
    /// an exclusive slot is not one -- it takes the static slots' paths (FN8's swap).
    fn lent_slot(&self, s: u32) -> Option<usize> {
        self.lent_any(s).filter(|&i| !self.excl[i])
    }

    /// FN42 / FN107: whether slot `s` is lent at all (inclusive or exclusive).
    fn lent_any(&self, s: u32) -> Option<usize> {
        (s >= self.base_slots)
            .then(|| (s - self.base_slots) as usize)
            .filter(|&i| self.lend_on.get(i).copied().unwrap_or(false))
    }

    /// FN107: whether slot `s` is an exclusive lent slot.
    #[must_use]
    pub fn is_exclusive(&self, s: u32) -> bool {
        self.lent_any(s).is_some_and(|i| self.excl[i])
    }

    /// FN107: the exclusive lent slots (`--xtier-pool-exclusive`).
    #[must_use]
    pub fn exclusive_slots(&self) -> u32 {
        self.excl.iter().filter(|&&b| b).count() as u32
    }

    /// FN107 (X1): make the LENT slots `ids` EXCLUSIVE -- slots their owner never reclaims (the
    /// P2P pool after the boot fill): each one's record DROPS its RAM shadow (the shadow slot is
    /// freed), and from now on a victim leaves the slot by FN8's swap (D2H through the spare, a
    /// write-back) instead of falling back to a shadow. The caller refills the freed RAM slots
    /// ([`Self::refill_ram`]). Refused under the deferred / demotion-free laws (as lending) and
    /// for an id that is not lent. Returns the shadows dropped.
    pub fn make_exclusive(&mut self, ids: &[u32]) -> Result<usize, TierError> {
        if self.params.defer_spares > 0 || self.demotion_free_law() {
            return Err(TierError::Config(
                "exclusive slots run under FN8's law only (no --xtier-defer, --xtier-demote ram)"
                    .into(),
            ));
        }
        if let Some(&bad) = ids.iter().find(|&&s| self.lent_any(s).is_none()) {
            return Err(TierError::Config(format!(
                "make_exclusive: slot {bad} is not lent"
            )));
        }
        let mut dropped = 0;
        for &s in ids {
            let i = (s - self.base_slots) as usize;
            if self.excl[i] {
                continue;
            }
            let sh = self.shadow[i];
            if sh != EMPTY {
                self.rslot[sh as usize] = Slot {
                    record: EMPTY,
                    stamp: 0,
                };
                self.ram_free.push(sh);
                self.shadow[i] = EMPTY;
                dropped += 1;
            }
            self.excl[i] = true;
        }
        Ok(dropped)
    }

    /// FN107 (X1): fill the FREE RAM slots with `Fetch` records (NVMe -> RAM, one batch), most
    /// valuable first (the warm profile's route count, then the LRFU score now, then the index):
    /// after [`Self::make_exclusive`] an L2 sized to the complement of the static AND exclusive
    /// slots is complete again. Returns the records read.
    pub fn refill_ram(&mut self, mover: &mut dyn ExpertMover) -> Result<usize, TierError> {
        let mut cand: Vec<(u64, f64, u32)> = self
            .place
            .iter()
            .enumerate()
            .filter(|(_, p)| **p == Placement::Fetch)
            .map(|(i, _)| (self.prof_count[i], self.crf_now(i as u32), i as u32))
            .collect();
        cand.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.total_cmp(&a.1)).then(a.2.cmp(&b.2)));
        let mut reads = Vec::new();
        for (_, _, rec) in cand {
            let Some(slot) = self.ram_free.pop() else {
                break;
            };
            self.rslot[slot as usize] = Slot {
                record: rec,
                stamp: self.now,
            };
            self.place[rec as usize] = Placement::Ram(slot);
            self.ram_uses[rec as usize] = 0;
            reads.push((u64::from(rec), slot));
        }
        if !reads.is_empty() {
            mover.nvme_to_ram(&reads)?;
            self.moves_total.bytes_nvme += reads.len() as u64 * self.stride;
        }
        Ok(reads.len())
    }

    /// The reads a boot does to realise the warm start: `(record, Placement)` for every record
    /// not `Fetch`, VRAM first. The boot reads each record from NVMe into its RAM slot (or a
    /// staging slot then H2D for VRAM).
    #[must_use]
    pub fn warm_plan(&self) -> Vec<(u64, Placement)> {
        let mut v: Vec<(u64, Placement)> = self
            .place
            .iter()
            .enumerate()
            .filter(|(_, p)| **p != Placement::Fetch)
            .map(|(i, p)| (i as u64, *p))
            .collect();
        v.sort_by_key(|(i, p)| (matches!(p, Placement::Ram(_)), *i));
        v
    }

    /// Rank.
    #[must_use]
    pub fn rank(&self) -> u32 {
        self.rank
    }
    /// Records this rank owns.
    #[must_use]
    pub fn records(&self) -> u64 {
        u64::from(self.layers) * u64::from(self.experts)
    }
    /// The parameters.
    #[must_use]
    pub fn params(&self) -> &TierParams {
        &self.params
    }
    /// Record stride.
    #[must_use]
    pub fn stride(&self) -> u64 {
        self.stride
    }
    /// `(first slot, slots)` of a layer's VRAM partition.
    #[must_use]
    pub fn layer_slots(&self, layer: u32) -> (u32, u32) {
        self.layer_slots[layer as usize]
    }
    /// Whether VRAM + RAM hold every record (the NVMe is off the step path).
    #[must_use]
    pub fn complete(&self) -> bool {
        self.place.iter().all(|p| *p != Placement::Fetch)
    }
    /// FN59b: arm the cheap victim planning (see the field; placement-identical).
    pub fn set_fast_plan(&mut self, on: bool) {
        self.fast_plan = on;
    }
    /// FN59b: every record's placement (the host proof compares two tiers step by step).
    #[must_use]
    pub fn placements(&self) -> &[Placement] {
        &self.place
    }
    /// The placement of (layer, GLOBAL expert), `None` for another rank's expert.
    #[must_use]
    pub fn placement(&self, layer: u32, expert: u32) -> Option<Placement> {
        self.index(layer, expert).map(|i| self.place[i as usize])
    }
    /// Record index of (layer, GLOBAL expert).
    #[must_use]
    pub fn index(&self, layer: u32, expert: u32) -> Option<u32> {
        if layer >= self.layers
            || expert < self.first_expert
            || expert >= self.first_expert + self.experts
        {
            return None;
        }
        Some(layer * self.experts + (expert - self.first_expert))
    }
    /// The record held by a VRAM slot.
    #[must_use]
    pub fn vram_record(&self, slot: u32) -> Option<u32> {
        self.vslot
            .get(slot as usize)
            .map(|s| s.record)
            .filter(|&r| r != EMPTY)
    }
    /// The record held by a RAM slot (the spare holds none).
    #[must_use]
    pub fn ram_record(&self, slot: u32) -> Option<u32> {
        self.rslot
            .get(slot as usize)
            .map(|s| s.record)
            .filter(|&r| r != EMPTY)
    }

    /// Base address of a placement's record (VRAM slot or the RAM slot's UVA address).
    #[must_use]
    pub fn record_addr(&self, p: Placement) -> Option<u64> {
        match p {
            Placement::Vram(s) if s >= self.base_slots => {
                self.lend_addr.get((s - self.base_slots) as usize).copied()
            }
            Placement::Vram(s) => Some(self.vram_base + u64::from(s) * self.stride),
            Placement::Ram(s) if s >= self.bounce_lo => {
                Some(self.bounce_addr + u64::from(s - self.bounce_lo) * self.stride)
            }
            Placement::Ram(s) if s >= self.band_lo => {
                let k = (s - self.band_lo) / VALVE_SEG;
                let a = *self.band_addr.get(k as usize)?;
                (a != 0).then(|| a + u64::from((s - self.band_lo) % VALVE_SEG) * self.stride)
            }
            Placement::Ram(s) => Some(self.ram_base + u64::from(s) * self.stride),
            Placement::Fetch => None,
        }
    }

    /// Fill one layer's two FN4 `xptr` tables for this rank's `experts` local experts:
    /// `gate_up[3e + k]` / `down[3e + k]` = the address of sub-record `k` (w, ws, wref) of local
    /// expert `e`, from its placement; a `Fetch` expert's entries are 0. The shared expert's
    /// entry (index `experts`, FN4's `e_loc`) is the engine's (a `dense.pack` record) and is left
    /// untouched. Returns the number of `Fetch` experts (their entries must never be read: a
    /// routed `Fetch` expert is resolved before the launch).
    pub fn fill_xptr(&self, layer: u32, gate_up: &mut [u64], down: &mut [u64]) -> usize {
        let n = self.experts as usize;
        assert!(
            gate_up.len() >= 3 * n && down.len() >= 3 * n,
            "xptr tables too short"
        );
        let mut fetch = 0;
        for e in 0..n {
            let i = layer as usize * n + e;
            match self.record_addr(self.place[i]) {
                Some(base) => {
                    for k in 0..3 {
                        gate_up[3 * e + k] = base + self.sub[k].0;
                        down[3 * e + k] = base + self.sub[3 + k].0;
                    }
                }
                None => {
                    fetch += 1;
                    for k in 0..3 {
                        gate_up[3 * e + k] = 0;
                        down[3 * e + k] = 0;
                    }
                }
            }
        }
        fetch
    }

    /// FN117b (S2): EVERY input of [`Self::fill_xptr`] that can change after construction -- the
    /// placements and the addresses [`Self::record_addr`] reads (the lent slots', the RAM slab's,
    /// the valve band's and the bounce's). Two equal keys fill byte-identical tables at every
    /// layer (the rest -- experts, stride, sub-record offsets -- is fixed at construction), so an
    /// engine may build the next step's tables early and keep them iff the key it took then equals
    /// the key at the step's begin.
    #[must_use]
    pub fn xptr_key(&self) -> XptrKey {
        XptrKey {
            place: self.place.clone(),
            lend_addr: self.lend_addr.clone(),
            vram_base: self.vram_base,
            ram_base: self.ram_base,
            bounce: (self.bounce_lo, self.bounce_addr),
            band: (self.band_lo, self.band_addr.clone()),
        }
    }

    /// FN18: the insert cap for the steps from now on (the per-bucket law,
    /// `--xtier-max-inserts-by-bucket`: the step's bucket picks it before `end_step`). 0 = no
    /// promotions this step (a static placement for it); the constructor still refuses 0 as the
    /// DEFAULT cap (FN8: a static placement loses held out).
    pub fn set_max_inserts(&mut self, n: u32) {
        self.params.max_inserts = n;
    }

    /// FN54: the PROMOTION BYTE BUDGET for the steps from now on (`--xtier-promote-budget-mib-by-
    /// bucket`: the step's bucket picks it before `end_step`, as the insert cap). `None` = no
    /// budget (today's law, byte for byte); `Some(b)` = at most `b / stride` promotions in the
    /// step over all layers, shared round-robin from a layer that rotates every step (no layer
    /// starves), each layer still under its insert cap. Placement only.
    pub fn set_promote_budget(&mut self, bytes: Option<u64>) {
        self.budget = bytes.map(|b| u32::try_from(b / self.stride).unwrap_or(u32::MAX));
    }

    /// FN54: the second-chance window for the steps from now on (the step's bucket picks it
    /// before `end_step`, as the insert cap; 0 = off). The misses are marked at every step
    /// whatever the window, so a bucket that turns it on sees the history.
    pub fn set_second_chance(&mut self, steps: u32) {
        self.params.second_chance = steps;
    }

    /// FN103: the admission planner's per-layer insert caps for the steps from now on (`None` /
    /// empty = today's law: `max_inserts` a layer). A layer's cap is `min(max_inserts, the
    /// byte-budget quota, caps[layer])`, so a planner that RAISES a layer above today's cap sets
    /// `max_inserts` to its largest cap. Placement only: the bytes the kernels read never change.
    pub fn set_layer_caps(&mut self, caps: Option<&[u32]>) {
        self.layer_cap.clear();
        if let Some(c) = caps {
            self.layer_cap.extend_from_slice(c);
        }
    }

    /// FN103: the step a record was last a RAM-placed route (the second chance's memory; 0 =
    /// never). Read-only (the census classifies a miss's reuse distance by it).
    #[must_use]
    pub fn last_ram_route(&self, i: u32) -> u64 {
        self.sc_last[i as usize]
    }

    /// FN54: second-chance marking + the step's per-layer quotas under the byte budget, before
    /// the layers' promotions (placement-neutral at the defaults: no budget, second chance 0).
    fn plan_promotions(&mut self) {
        let w = u64::from(self.params.second_chance);
        for l in 0..self.layers as usize {
            for &i in &self.step_routes[l] {
                let i = i as usize;
                if !matches!(self.place[i], Placement::Ram(_)) {
                    continue;
                }
                let last = self.sc_last[i];
                if w > 0 && last > 0 && self.now - last <= w {
                    self.sc_ok[i] = self.now;
                }
                self.sc_last[i] = self.now;
            }
        }
        let Some(mut left) = self.budget else {
            self.quota.iter_mut().for_each(|q| *q = u32::MAX);
            return;
        };
        let thresh = self.params.read_through;
        let cap = self.params.max_inserts;
        // each layer's candidates (its capped misses), as `end_layer_fn8` will count them
        let want: Vec<u32> = (0..self.layers as usize)
            .map(|l| {
                let n = self.layer_slots[l].1 + self.layer_lent[l].len() as u32;
                let routes = &self.step_routes[l];
                if n == 0 || routes.is_empty() || routes.len() as f64 > thresh * f64::from(n) {
                    return 0;
                }
                let c = routes.iter().filter(|&&i| self.candidate(i)).count() as u32;
                c.min(cap)
            })
            .collect();
        self.quota.iter_mut().for_each(|q| *q = 0);
        let layers = self.layers as usize;
        let start = (self.now % u64::from(self.layers.max(1))) as usize;
        let mut more = true;
        while left > 0 && more {
            more = false;
            for k in 0..layers {
                let l = (start + k) % layers;
                if self.quota[l] < want[l] && left > 0 {
                    self.quota[l] += 1;
                    left -= 1;
                    more = true;
                }
            }
        }
        let cut: u32 = want.iter().zip(&self.quota).map(|(w, q)| w - q).sum();
        self.moves_step.budget_cut += u64::from(cut);
    }

    /// FN54: a RAM-placed route that may be promoted this step (second chance when armed).
    fn candidate(&self, i: u32) -> bool {
        matches!(self.place[i as usize], Placement::Ram(_))
            && (self.params.second_chance == 0 || self.sc_ok[i as usize] == self.now)
    }

    /// Start a step.
    pub fn begin_step(&mut self) {
        self.unbounce();
        self.now += 1;
        self.step = PathCounts::default();
        self.moves_step = MoveCounts::default();
        for r in &mut self.step_routes {
            r.clear();
        }
    }

    /// The demand path for one layer, after its router: count every distinct routed expert of
    /// this rank on its path, and read every `Fetch` one from NVMe into a RAM slot NOW (all of
    /// them in one batch, queue depth = the batch). `routed` holds GLOBAL expert ids (any order,
    /// repeats and other ranks' experts allowed). After it returns every routed expert of this
    /// rank is `Vram` or `Ram`.
    pub fn resolve(
        &mut self,
        layer: u32,
        routed: &[u32],
        mover: &mut dyn ExpertMover,
    ) -> Result<PathCounts, TierError> {
        let mut local: Vec<u32> = routed
            .iter()
            .filter_map(|&e| self.index(layer, e))
            .collect();
        local.sort_unstable();
        // FN48: the rows routing each record (the row-weighted LRFU bump; 1 a record = today)
        let gamma = self.params.policy.row_gamma;
        let mut mult: Vec<u32> = Vec::new();
        if gamma != 0.0 {
            for w in local.chunk_by(|a, b| a == b) {
                mult.push(w.len() as u32);
            }
        }
        local.dedup();
        let mut c = PathCounts::default();
        let mut fetch = Vec::new();
        for (k, &i) in local.iter().enumerate() {
            if self.touched[i as usize] == self.now {
                continue; // resolved earlier this step (a second call for the layer)
            }
            self.touched[i as usize] = self.now;
            self.step_routes[layer as usize].push(i);
            if self.params.lrfu_lambda > 0.0 {
                let bump = if gamma == 0.0 {
                    1.0
                } else {
                    f64::from(mult[k]).powf(gamma)
                };
                self.crf[i as usize] = self.crf_now(i) + bump;
                self.crf_at[i as usize] = self.now;
            }
            if self.params.admit_window > 0 {
                self.freq[i as usize] = self.freq[i as usize].saturating_add(1);
            }
            match self.place[i as usize] {
                Placement::Vram(s) => {
                    c.resident += 1;
                    self.vslot[s as usize].stamp = self.now;
                }
                Placement::Ram(s) => {
                    c.ram += 1;
                    self.rslot[s as usize].stamp = self.now;
                    self.ram_uses[i as usize] = self.ram_uses[i as usize].saturating_add(1);
                    if self.prefetched[i as usize] {
                        c.prefetched += 1;
                        self.prefetched[i as usize] = false;
                    }
                }
                Placement::Fetch if self.pass_xmiss => {
                    // FN53c: the miss flag served it from the bounce during the step
                    c.nvme += 1;
                    self.valve.xmiss_routes += 1;
                }
                Placement::Fetch => {
                    c.nvme += 1;
                    fetch.push(i);
                }
            }
        }
        // FN53b: a PREFILL pass's tail records go to the bounce (prefetched one layer ahead, or
        // read now into the layer's half), never into the L2
        let fetch = self.bounce_resolve(layer, fetch, mover)?;
        if !fetch.is_empty() {
            let mut reads = Vec::with_capacity(fetch.len());
            for &i in &fetch {
                let s = self.take_ram_slot(layer)?;
                self.rslot[s as usize] = Slot {
                    record: i,
                    stamp: self.now,
                };
                self.place[i as usize] = Placement::Ram(s);
                self.ram_uses[i as usize] = 1;
                reads.push((u64::from(i), s));
            }
            mover.nvme_to_ram(&reads)?;
            self.moves_step.bytes_nvme += reads.len() as u64 * self.stride;
        }
        self.per_layer[layer as usize].add(&c);
        self.step.add(&c);
        self.total.add(&c);
        Ok(c)
    }

    /// FN53b: the pass that follows is a PREFILL chunk (`true`: its `Fetch` routes take the
    /// bounce) or a decode / verify step (`false`: misses enter the L2 under the policy).
    pub fn set_pass_prefill(&mut self, on: bool) {
        self.pass_prefill = on;
    }

    /// FN53b: the bounce's records leave the placements (back to `Fetch`; their bytes stay in
    /// the bounce until the next read overwrites them, after the reader's step synchronized).
    fn unbounce(&mut self) {
        for (i, s) in std::mem::take(&mut self.bounced) {
            self.place[i as usize] = Placement::Fetch;
            self.rslot[s as usize].record = EMPTY;
        }
        self.bounce_ready.clear();
    }

    /// FN53b: the bounce half of `layer` (two halves alternate by layer: layer `l + 1`'s reads
    /// may run while layer `l`'s experts read the other half). Taking a half drops the records
    /// an earlier layer placed in it (that layer's experts ran: the engine synchronized on this
    /// layer's routes, or on the previous layer's, before it writes the half).
    fn bounce_half(&mut self, layer: u32) -> (u32, u32) {
        let n = self.params.bounce_slots / 2;
        let lo = self.bounce_lo + (layer % 2) * n;
        let mut keep = Vec::with_capacity(self.bounced.len());
        for (i, s) in std::mem::take(&mut self.bounced) {
            if s >= lo && s < lo + n && i / self.experts != layer {
                self.place[i as usize] = Placement::Fetch;
                self.rslot[s as usize].record = EMPTY;
            } else {
                keep.push((i, s));
            }
        }
        self.bounced = keep;
        (lo, n)
    }

    /// FN53b: THE LAYER-AHEAD TAIL READER -- read `layer`'s `Fetch` records (expert order, as
    /// many as its bounce half holds) from the drive into the half, BLIND (a wide prefill chunk
    /// routes ~every record). The engine calls it after it issued the previous layer's experts
    /// and before it waits for this layer's routes, so the reads run under that layer's device
    /// work. Only on a prefill pass with a bounce; returns the records read.
    pub fn bounce_prefetch(
        &mut self,
        layer: u32,
        mover: &mut dyn ExpertMover,
    ) -> Result<usize, TierError> {
        self.bounce_ready.clear();
        if !self.pass_prefill || self.params.bounce_slots == 0 || layer >= self.layers {
            return Ok(0);
        }
        let (lo, n) = self.bounce_half(layer);
        let first = layer * self.experts;
        let recs: Vec<u32> = (first..first + self.experts)
            .filter(|&i| self.place[i as usize] == Placement::Fetch)
            .take(n as usize)
            .collect();
        if recs.is_empty() {
            return Ok(0);
        }
        self.bounce_ready = recs.iter().zip(lo..).map(|(&i, s)| (i, s)).collect();
        let reads: Vec<(u64, u32)> = self
            .bounce_ready
            .iter()
            .map(|&(i, s)| (u64::from(i), s))
            .collect();
        mover.nvme_to_ram(&reads)?;
        self.valve.bounce_prefetch += reads.len() as u64;
        self.moves_step.bytes_nvme += reads.len() as u64 * self.stride;
        Ok(reads.len())
    }

    /// FN53b: place a prefill pass's `Fetch` routes of `layer` in the bounce (prefetched ones
    /// first, then demand reads into the half's free slots); returns the routes that found no
    /// bounce room (they take the L2 path).
    fn bounce_resolve(
        &mut self,
        layer: u32,
        fetch: Vec<u32>,
        mover: &mut dyn ExpertMover,
    ) -> Result<Vec<u32>, TierError> {
        if fetch.is_empty() || !self.pass_prefill || self.params.bounce_slots == 0 {
            self.bounce_ready.clear();
            return Ok(fetch);
        }
        let (lo, n) = self.bounce_half(layer);
        let ready: Vec<(u32, u32)> = std::mem::take(&mut self.bounce_ready)
            .into_iter()
            .filter(|&(i, s)| i / self.experts == layer && s >= lo && s < lo + n)
            .collect();
        let held: Vec<u32> = ready.iter().map(|&(_, s)| s).collect();
        let mut free: Vec<u32> = (lo..lo + n).filter(|s| !held.contains(s)).collect();
        free.reverse();
        let mut rest = Vec::new();
        let mut reads = Vec::new();
        for i in fetch {
            let slot = if let Some(&(_, s)) = ready.iter().find(|r| r.0 == i) {
                self.valve.bounce_hits += 1;
                s
            } else if let Some(s) = free.pop() {
                self.valve.bounce_demand += 1;
                reads.push((u64::from(i), s));
                s
            } else {
                self.valve.bounce_full += 1;
                rest.push(i);
                continue;
            };
            self.place[i as usize] = Placement::Ram(slot);
            self.rslot[slot as usize] = Slot {
                record: i,
                stamp: self.now,
            };
            self.bounced.push((i, slot));
        }
        if !reads.is_empty() {
            mover.nvme_to_ram(&reads)?;
            self.moves_step.bytes_nvme += reads.len() as u64 * self.stride;
        }
        Ok(rest)
    }

    /// FN53c: the pass that follows runs the DEVICE MISS FLAG (`true`): its `Fetch` routes are
    /// read into the bounce while the step runs ([`Self::xmiss_slots`]) and, at the tier end,
    /// count as served and stay `Fetch` (the open set is static: no miss enters the L2, so the
    /// valve's freed RAM stays freed). Never with a prefill pass.
    pub fn set_pass_xmiss(&mut self, on: bool) {
        self.pass_xmiss = on && !self.pass_prefill;
    }

    /// FN53c: the device miss flag's request for `layer`: `experts` = GLOBAL expert ids the
    /// kernel found with a 0 xptr entry. Each must be this rank's and `Fetch` (else the tables
    /// and the tier disagree: refused by name); they take the layer's bounce half in order (a
    /// half holds `bounce_slots / 2`; more misses than that are refused -- the layer-balanced
    /// close leaves at most `ceil(band / layers)` `Fetch` records a layer). The caller reads each
    /// record into its slot, then answers the kernel with the entries.
    pub fn xmiss_slots(
        &mut self,
        layer: u32,
        experts: &[u32],
    ) -> Result<Vec<XmissSlot>, TierError> {
        let n = self.params.bounce_slots / 2;
        if n == 0 {
            return Err(TierError::Config(
                "the miss flag needs the valve's bounce".into(),
            ));
        }
        if experts.len() > n as usize {
            return Err(TierError::Config(format!(
                "layer {layer}: {} misses, the bounce half holds {n}",
                experts.len()
            )));
        }
        let lo = self.bounce_lo + (layer % 2) * n;
        let mut out = Vec::with_capacity(experts.len());
        for (j, &e) in experts.iter().enumerate() {
            let i = self.index(layer, e).ok_or_else(|| {
                TierError::Config(format!("layer {layer}: expert {e} is not this rank's"))
            })?;
            if self.place[i as usize] != Placement::Fetch {
                return Err(TierError::Config(format!(
                    "layer {layer}: expert {e} missed on the card but is placed {:?}",
                    self.place[i as usize]
                )));
            }
            let slot = lo + j as u32;
            let base = self
                .record_addr(Placement::Ram(slot))
                .ok_or_else(|| TierError::Config("the bounce has no address".into()))?;
            let mut entries = [0u64; 6];
            for (k, x) in entries.iter_mut().enumerate() {
                *x = base + self.sub[k].0;
            }
            out.push(XmissSlot {
                record: i,
                slot,
                entries,
            });
        }
        self.valve.xmiss_requests += 1;
        self.valve.xmiss_records += out.len() as u64;
        self.moves_step.bytes_nvme += out.len() as u64 * self.stride;
        Ok(out)
    }

    /// FN53b: the valve band's geometry -- (first slot, slots, open slots).
    #[must_use]
    pub fn band(&self) -> (u32, u32, u32) {
        (self.band_lo, self.params.band_slots, self.band_open)
    }

    /// FN53b: the bounce's first slot and slots.
    #[must_use]
    pub fn bounce(&self) -> (u32, u32) {
        (self.bounce_lo, self.params.bounce_slots)
    }

    /// FN53b: the valve's counters.
    #[must_use]
    pub fn valve_counts(&self) -> ValveCounts {
        self.valve
    }

    /// FN53b: segment `seg`'s UVA address (its own pinned allocation).
    pub fn set_band_addr(&mut self, seg: u32, addr: u64) {
        if let Some(a) = self.band_addr.get_mut(seg as usize) {
            *a = addr;
        }
    }

    /// FN53b: the bounce's UVA address.
    pub fn set_bounce_addr(&mut self, addr: u64) {
        self.bounce_addr = addr;
    }

    fn valve_score(&self, i: u32) -> f64 {
        if self.params.lrfu_lambda > 0.0 {
            self.crf_now(i)
        } else {
            f64::from(self.score[i as usize])
        }
    }

    /// FN53b: THE VALVE'S DEMOTION -- close the band's top open segment: the [`VALVE_SEG`]
    /// COLDEST RAM-placed records (LRFU score, then the stamp, then the record) drop to `Fetch`
    /// (their bytes stay on the drive: no copy), LAYER-BALANCED (no layer's `Fetch` count past
    /// `ceil(closed band slots / layers)` while another layer has room), and every record, lent
    /// shadow or spare still in the segment moves into a freed slot below it (`relocs`: host
    /// copies the engine makes before it frees the segment's allocation). Placement only: no
    /// routed byte changes. The caller drained every stream; FN8's law only (no pending copy).
    pub fn valve_close(&mut self) -> Result<ValveClose, TierError> {
        if self.band_open == 0 {
            return Err(TierError::Config(
                "valve: every band segment is closed".into(),
            ));
        }
        if !self.pending.is_empty() || !self.swaps.is_empty() {
            return Err(TierError::Config(
                "valve: a close runs between steps under FN8's law only".into(),
            ));
        }
        self.unbounce();
        let seg = self.band_open / VALVE_SEG - 1;
        let (a, b) = (
            self.band_lo + seg * VALVE_SEG,
            self.band_lo + self.band_open,
        );
        let closed_after = self.params.band_slots - self.band_open + VALVE_SEG;
        let cap = closed_after.div_ceil(self.layers);
        let mut per = vec![0u32; self.layers as usize];
        for (i, p) in self.place.iter().enumerate() {
            if *p == Placement::Fetch {
                per[i / self.experts as usize] += 1;
            }
        }
        let mut cand: Vec<(f64, u64, u32)> = self
            .place
            .iter()
            .enumerate()
            .filter_map(|(i, p)| match *p {
                Placement::Ram(s) if s < self.bounce_lo => Some((
                    self.valve_score(i as u32),
                    self.rslot[s as usize].stamp,
                    i as u32,
                )),
                _ => None,
            })
            .collect();
        cand.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));
        let mut pick: Vec<u32> = Vec::with_capacity(VALVE_SEG as usize);
        let mut taken = vec![false; cand.len()];
        for pass in 0..2 {
            for (k, &(_, _, i)) in cand.iter().enumerate() {
                if pick.len() == VALVE_SEG as usize {
                    break;
                }
                let l = (i / self.experts) as usize;
                if taken[k] || (pass == 0 && per[l] >= cap) {
                    continue;
                }
                taken[k] = true;
                per[l] += 1;
                pick.push(i);
            }
        }
        // drop the picks; their slots outside the segment become relocation targets
        let mut freed: Vec<u32> = Vec::new();
        for &i in &pick {
            let Placement::Ram(s) = self.place[i as usize] else {
                unreachable!("picked RAM-placed records")
            };
            self.place[i as usize] = Placement::Fetch;
            self.prefetched[i as usize] = false;
            self.rslot[s as usize] = Slot {
                record: EMPTY,
                stamp: 0,
            };
            if s < a || s >= b {
                freed.push(s);
            }
        }
        // the free list's slots outside the segment are targets too; the segment's leave it
        freed.extend(self.ram_free.iter().copied().filter(|&s| s < a || s >= b));
        freed.sort_unstable_by(|x, y| y.cmp(x));
        let mut relocs = Vec::new();
        for s in a..b {
            let rec = self.rslot[s as usize].record;
            let is_spare = s == self.spare;
            let ring = self.spare_q.iter().position(|&q| q == s);
            if rec == EMPTY && !is_spare && ring.is_none() {
                continue;
            }
            let to = freed.pop().ok_or_else(|| {
                TierError::Config("valve: no free slot below the segment (accounting)".into())
            })?;
            if rec != EMPTY {
                self.rslot[to as usize] = self.rslot[s as usize];
                self.rslot[s as usize] = Slot {
                    record: EMPTY,
                    stamp: 0,
                };
                if self.place[rec as usize] == Placement::Ram(s) {
                    self.place[rec as usize] = Placement::Ram(to);
                } else if let Some(k) = self.shadow.iter().position(|&x| x == s) {
                    self.shadow[k] = to;
                }
                relocs.push((s, to));
            } else if is_spare {
                self.spare = to;
            } else if let Some(k) = ring {
                self.spare_q[k] = to;
            }
        }
        self.ram_free = freed;
        self.band_open -= VALVE_SEG;
        if let Some(x) = self.band_addr.get_mut(seg as usize) {
            *x = 0;
        }
        self.valve.closes += 1;
        self.valve.dropped += pick.len() as u64;
        self.valve.relocated += relocs.len() as u64;
        Ok(ValveClose {
            seg,
            relocs,
            dropped: pick,
        })
    }

    /// FN53b: REOPEN the band's lowest closed segment at `addr` (its new pinned allocation) and
    /// return the drive reads that refill the L2 -- the hottest `Fetch` records first (LRFU
    /// score, then the record), one a free slot; the caller issues them (`nvme_to_ram`) before
    /// the next step. With every segment open again the tier is complete again.
    pub fn valve_open(&mut self, addr: u64) -> Result<Vec<(u64, u32)>, TierError> {
        if self.band_open == self.params.band_slots {
            return Err(TierError::Config(
                "valve: every band segment is open".into(),
            ));
        }
        self.unbounce();
        let seg = self.band_open / VALVE_SEG;
        let a = self.band_lo + self.band_open;
        self.band_open += VALVE_SEG;
        if let Some(x) = self.band_addr.get_mut(seg as usize) {
            *x = addr;
        }
        self.ram_free.extend((a..a + VALVE_SEG).rev());
        let mut want: Vec<(f64, u32)> = self
            .place
            .iter()
            .enumerate()
            .filter(|(_, p)| **p == Placement::Fetch)
            .map(|(i, _)| (self.valve_score(i as u32), i as u32))
            .collect();
        want.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
        let mut reads = Vec::new();
        for (_, i) in want {
            let Some(s) = self.ram_free.pop() else { break };
            self.rslot[s as usize] = Slot {
                record: i,
                stamp: self.now,
            };
            self.place[i as usize] = Placement::Ram(s);
            self.ram_uses[i as usize] = 0;
            reads.push((u64::from(i), s));
        }
        self.valve.opens += 1;
        self.valve.refilled += reads.len() as u64;
        Ok(reads)
    }

    /// Check (I3) that every routed expert of this rank at `layer` is dispatchable.
    pub fn check_dispatch(&self, layer: u32, routed: &[u32]) -> Result<(), TierError> {
        for &e in routed {
            if self.placement(layer, e) == Some(Placement::Fetch) {
                return Err(TierError::Unresolved {
                    rank: self.rank,
                    layer,
                    expert: e,
                });
            }
        }
        Ok(())
    }

    /// A free RAM slot, or the LRU `(stamp, slot)` RAM slot whose record is not routed at this
    /// layer or a later one this step (its record is dropped to `Fetch`). An EARLIER layer's
    /// records are evictable: the demand path reads layer L's routes after layer L's router ran,
    /// which the in-order stream runs after every earlier layer's MoE kernels completed.
    fn take_ram_slot(&mut self, layer: u32) -> Result<u32, TierError> {
        if let Some(s) = self.ram_free.pop() {
            return Ok(s);
        }
        let mut best: Option<(u64, u32)> = None;
        for (s, slot) in self.rslot.iter().enumerate() {
            // FN53b: never the bounce, and never a lent record's shadow (its record is placed in
            // VRAM: dropping it would leave the lent slot's record unplaced)
            if s as u32 >= self.bounce_lo
                || slot.record == EMPTY
                || self.place[slot.record as usize] != Placement::Ram(s as u32)
                || (self.touched[slot.record as usize] == self.now
                    && slot.record / self.experts >= layer)
            {
                continue;
            }
            let k = (slot.stamp, s as u32);
            if best.is_none_or(|b| k < b) {
                best = Some(k);
            }
        }
        let (_, s) = best.ok_or(TierError::RamExhausted {
            rank: self.rank,
            layer,
            slots: self.params.ram_slots,
        })?;
        let rec = self.rslot[s as usize].record;
        self.place[rec as usize] = Placement::Fetch;
        self.prefetched[rec as usize] = false;
        self.rslot[s as usize].record = EMPTY;
        self.moves_step.ram_drops += 1;
        Ok(s)
    }

    /// The end of a step: the LRU over the step's routes, then the prefetch. Returns the step's
    /// counters. The copies are queued on the mover in order and flushed before it returns.
    pub fn end_step(&mut self, mover: &mut dyn ExpertMover) -> Result<StepReport, TierError> {
        // FN53b: the bounce's records leave the placements before the LRU (never promoted)
        self.unbounce();
        let thresh = self.params.read_through;
        // FN17: the deferred law (a complete tier with spares): publish the last step's copies
        // (the consumer's fence waited for them), then this step's copies go to holes / spares
        let defer = self.params.defer_spares > 0 && self.complete();
        mover.set_deferred(defer);
        if defer {
            self.publish();
        } else {
            debug_assert!(self.pending.is_empty(), "pending copies under FN8's law");
        }
        // FN36: the demotion-free law -- the last step's refills (read under this step), then
        // their swaps' publication (H2D promotions, gated per layer as FN8's)
        let nd = self.demotion_free();
        if nd {
            if !self.refill_reads.is_empty() {
                self.moves_step.refill_late += 1;
                self.refill(mover)?;
            }
            let freed = std::mem::take(&mut self.retiring);
            self.pool.extend(freed);
            self.publish_swaps(mover)?;
        }
        self.plan_promotions();
        for l in 0..self.layers {
            let (base, n0) = self.layer_slots[l as usize];
            // FN42: the layer's lent slots join its victim set (0 without lending: today's law)
            let lent = std::mem::take(&mut self.layer_lent[l as usize]);
            let n = n0 + lent.len() as u32;
            let routes = std::mem::take(&mut self.step_routes[l as usize]);
            let res = self.end_layer_fn8(l, base, n0, n, &lent, routes, thresh, defer, nd, mover);
            self.layer_lent[l as usize] = lent;
            res?;
        }
        mover.flush()?;
        if self.params.admit_window > 0 && (self.now % u64::from(self.params.admit_window)) == 0 {
            self.freq.iter_mut().for_each(|f| *f /= 2);
        }
        // The access history: an exponentially decayed touch count per record.
        let d = self.params.history_decay;
        for (i, s) in self.score.iter_mut().enumerate() {
            *s = *s * d
                + if self.touched[i] == self.now {
                    1.0
                } else {
                    0.0
                };
        }
        self.prefetch(mover)?;
        self.steps += 1;
        self.moves_total.add(&self.moves_step);
        Ok(StepReport {
            paths: self.step,
            moves: self.moves_step,
        })
    }

    /// One layer of an FN8 / demotion-free `end_step` (FN42: `n0` static slots from `base` plus
    /// the layer's `lent` slots, `n` in all).
    #[allow(clippy::too_many_arguments)]
    fn end_layer_fn8(
        &mut self,
        l: u32,
        base: u32,
        n0: u32,
        n: u32,
        lent: &[u32],
        routes: Vec<u32>,
        thresh: f64,
        defer: bool,
        nd: bool,
        mover: &mut dyn ExpertMover,
    ) -> Result<(), TierError> {
        let rt = n > 0 && routes.len() as f64 > thresh * f64::from(n);
        // FN48: the reuse estimate learns from the step's routes (a read-through layer's only
        // when `rt_weight` > 0)
        if self.params.policy.kind == VictimKind::Table {
            let lam = self.params.policy.table_lambda;
            for &i in &routes {
                let i = i as usize;
                let lc = if self.ru_n[i] == 0 {
                    1.0
                } else {
                    let dt = (self.now - self.ru_last[i]).min(1 << 20) as i32;
                    self.tb_lc[i] * lam.powi(dt) + 1.0
                };
                self.tb_lc[i] = lc;
                self.tb_cls[i] = table_class(lc, rt) as u8;
                self.ru_last[i] = self.now;
                self.ru_n[i] = self.ru_n[i].saturating_add(1);
            }
        }
        if self.params.policy.kind == VictimKind::Reuse
            && !(rt && self.params.policy.rt_weight == 0.0)
        {
            let a = self.params.policy.reuse_alpha;
            for &i in &routes {
                let i = i as usize;
                if self.ru_n[i] > 0 {
                    let g = (self.now - self.ru_last[i]) as f64;
                    self.ru_gap[i] = if self.ru_n[i] == 1 {
                        g
                    } else {
                        a * g + (1.0 - a) * self.ru_gap[i]
                    };
                }
                self.ru_last[i] = self.now;
                self.ru_n[i] = self.ru_n[i].saturating_add(1);
            }
        }
        if n == 0 || routes.is_empty() {
            self.step_routes[l as usize] = routes;
            return Ok(());
        }
        if rt {
            self.moves_step.read_through_layers += 1;
            // FN48: a read-through layer's routes add `rt_weight` to the LRFU score (1 = today)
            let w = self.params.policy.rt_weight;
            if self.params.lrfu_lambda > 0.0 && w != 1.0 {
                for &i in &routes {
                    self.crf[i as usize] -= 1.0 - w;
                }
            }
            self.step_routes[l as usize] = routes;
            return Ok(());
        }
        // Misses in expert-id order (routes were pushed in id order per resolve call).
        let mut misses: Vec<u32> = routes
            .iter()
            .copied()
            .filter(|&i| matches!(self.place[i as usize], Placement::Ram(_)))
            .collect();
        if self.params.second_chance > 0 {
            // FN54: a first miss in its window only marks the record
            let all = misses.len();
            misses.retain(|&i| self.candidate(i));
            self.moves_step.sc_held += (all - misses.len()) as u64;
        }
        misses.sort_unstable();
        if self.params.policy.miss_rank {
            // FN48: the most valuable misses fill the cap
            misses.sort_by(|&a, &b| self.vscore(b).total_cmp(&self.vscore(a)).then(a.cmp(&b)));
        }
        // FN54: the step's byte budget shares out the promotions (u32::MAX = no budget)
        let cap = self.params.max_inserts.min(self.quota[l as usize]);
        // FN103: the admission planner's per-layer cap (none = today's law)
        let cap = self.layer_cap.get(l as usize).map_or(cap, |&c| cap.min(c));
        misses.truncate(cap as usize);
        if defer {
            self.defer_layer(l, &misses, mover)?;
            self.step_routes[l as usize] = routes;
            return Ok(());
        }
        if !misses.is_empty() {
            // Batched victim ranking: the smallest (stamp, slot) keys among slots whose record
            // is not routed this step; an empty slot has stamp 0.
            let mut keys: Vec<(u64, u32)> = (base..base + n0)
                .chain(lent.iter().copied())
                .filter(|&s| {
                    let r = self.vslot[s as usize].record;
                    r == EMPTY || self.touched[r as usize] != self.now
                })
                .map(|s| {
                    let v = &self.vslot[s as usize];
                    (if v.record == EMPTY { 0 } else { v.stamp }, s)
                })
                .collect();
            if self.fast_plan {
                // FN59b: the same order as below, the score computed once a slot, and only the
                // first `misses.len()` keys ranked (a pair consumes at most one victim a miss)
                self.select_victims(&mut keys, misses.len());
            } else if self.params.lrfu_lambda > 0.0 || self.params.policy.kind != VictimKind::Lrfu {
                // LRFU: the smallest decayed score first, then FN8's (stamp, slot); FN48: the
                // policy's score (an empty slot always first)
                let score = |s: u32| {
                    let r = self.vslot[s as usize].record;
                    if r == EMPTY {
                        f64::NEG_INFINITY
                    } else {
                        self.vscore(r)
                    }
                };
                keys.sort_by(|a, b| score(a.1).total_cmp(&score(b.1)).then(a.cmp(b)));
            } else {
                keys.sort_unstable();
            }
            let mut victims = keys.iter();
            let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(misses.len());
            let mut next = victims.next();
            for &m in &misses {
                let Some(&(_, v)) = next else { break };
                let vr = self.vslot[v as usize].record;
                if self.params.admit_window > 0
                    && vr != EMPTY
                    && self.freq[m as usize] <= self.freq[vr as usize]
                {
                    self.moves_step.refused += 1;
                    continue; // not admitted: it stays in RAM, the victim stays
                }
                // FN39: the margin + uses admission (off at 0 = today's law, byte for byte)
                if (self.params.admit_uses > 0
                    && self.ram_uses[m as usize] < self.params.admit_uses)
                    || (self.params.admit_margin > 0.0
                        && self.params.lrfu_lambda > 0.0
                        && vr != EMPTY
                        && self.crf_now(m) < self.crf_now(vr) + self.params.admit_margin)
                {
                    self.moves_step.refused += 1;
                    continue; // not admitted: it stays in RAM, the victim stays
                }
                pairs.push((m, v));
                next = victims.next();
            }
            for (m, v) in pairs {
                let Placement::Ram(r) = self.place[m as usize] else {
                    unreachable!("a miss is RAM-placed by construction")
                };
                let victim = self.vslot[v as usize].record;
                if nd && victim != EMPTY {
                    // FN36: nothing moves now; the victim is re-read from NVMe under the next
                    // step and the swap is published at the next end_step
                    let Some(sp) = self.pool.pop() else {
                        self.moves_step.hole_waits += 1;
                        continue;
                    };
                    self.swaps.push(Swap {
                        m,
                        r,
                        v,
                        w: victim,
                        sp,
                    });
                    self.refill_reads.push((u64::from(victim), sp));
                    self.moves_step.refills += 1;
                    self.moves_step.bytes_nvme += self.stride;
                    continue;
                }
                if let Some(i) = self.lent_slot(v) {
                    // FN42: a LENT slot is inclusive -- its victim falls back to its shadow
                    // RAM slot (no copy) and the promoted record keeps its RAM slot `r`
                    if victim != EMPTY {
                        self.place[victim as usize] = Placement::Ram(self.shadow[i]);
                        self.ram_uses[victim as usize] = 0;
                        self.moves_step.lent_drops += 1;
                    }
                    mover.ram_to_vram(r, v)?;
                    self.shadow[i] = r;
                    self.vslot[v as usize] = Slot {
                        record: m,
                        stamp: self.now,
                    };
                    self.place[m as usize] = Placement::Vram(v);
                    self.prefetched[m as usize] = false;
                    self.moves_step.inserts += 1;
                    self.moves_step.bytes_h2d += self.stride;
                    self.layer_h2d[(m / self.experts) as usize] += self.stride;
                    continue;
                }
                if victim != EMPTY {
                    // Demote through the spare: D2H victim -> spare, then H2D r -> v; r becomes
                    // the spare. One copy stream in this order is what makes it safe.
                    // FN107: an exclusive slot's victim takes this path (its write-back)
                    if self.is_exclusive(v) {
                        self.moves_step.excl_writebacks += 1;
                    }
                    let sp = self.spare;
                    mover.vram_to_ram(v, sp)?;
                    self.rslot[sp as usize] = Slot {
                        record: victim,
                        stamp: self.vslot[v as usize].stamp,
                    };
                    self.place[victim as usize] = Placement::Ram(sp);
                    self.ram_uses[victim as usize] = 0;
                    self.moves_step.demotions += 1;
                    self.moves_step.bytes_d2h += self.stride;
                    mover.ram_to_vram(r, v)?;
                    // FN54b: with a spare ring, r waits `spare_ring` swaps before its next D2H
                    if let Some(next) = self.spare_q.pop_front() {
                        self.spare_q.push_back(r);
                        self.spare = next;
                    } else {
                        self.spare = r;
                    }
                } else {
                    mover.ram_to_vram(r, v)?;
                    self.ram_free.push(r);
                }
                self.rslot[r as usize] = Slot {
                    record: EMPTY,
                    stamp: 0,
                };
                self.vslot[v as usize] = Slot {
                    record: m,
                    stamp: self.now,
                };
                self.place[m as usize] = Placement::Vram(v);
                self.prefetched[m as usize] = false;
                self.moves_step.inserts += 1;
                self.moves_step.bytes_h2d += self.stride;
                self.layer_h2d[(m / self.experts) as usize] += self.stride;
            }
            mover.end_layer(l)?;
        }
        self.step_routes[l as usize] = routes;
        Ok(())
    }

    /// FN17: publish the deferred copies of the last `end_step` -- the promoted records move to
    /// their holes, the demoted ones to their spares; each promoted record's old RAM slot and each
    /// demoted record's old VRAM slot become free (a RAM spare / a hole). Called only after the
    /// consumer's fence (the device mover's `after` waits for every issued copy under this law)
    /// and after every read of the old placements (the step that just ended).
    fn publish(&mut self) {
        for p in std::mem::take(&mut self.pending) {
            match p {
                Pend::Promote { m, r, h } => {
                    self.vslot[h as usize] = Slot {
                        record: m,
                        stamp: self.now,
                    };
                    self.place[m as usize] = Placement::Vram(h);
                    self.prefetched[m as usize] = false;
                    self.rslot[r as usize] = Slot {
                        record: EMPTY,
                        stamp: 0,
                    };
                    self.rspares.push(r);
                }
                Pend::Demote { v, vs, sp } => {
                    self.rslot[sp as usize] = Slot {
                        record: v,
                        stamp: self.vslot[vs as usize].stamp,
                    };
                    self.place[v as usize] = Placement::Ram(sp);
                    self.ram_uses[v as usize] = 0;
                    self.vslot[vs as usize] = Slot {
                        record: EMPTY,
                        stamp: 0,
                    };
                }
            }
            self.moves_step.published += 1;
        }
    }

    /// FN17: one layer of a deferred `end_step`: each miss (up to the layer's holes) is copied
    /// RAM -> a hole; then victims (the LRU, not routed this step) are copied VRAM -> a free RAM
    /// spare until the layer has as many holes ready as it had misses. Nothing is published: every
    /// copy writes a slot no placement points at.
    fn defer_layer(
        &mut self,
        l: u32,
        misses: &[u32],
        mover: &mut dyn ExpertMover,
    ) -> Result<(), TierError> {
        if misses.is_empty() {
            return Ok(());
        }
        let (base, n) = self.layer_slots[l as usize];
        let mut holes: Vec<u32> = (base..base + n)
            .rev()
            .filter(|&s| self.vslot[s as usize].record == EMPTY)
            .collect();
        let mut issued = false;
        for &m in misses {
            let Some(h) = holes.pop() else {
                self.moves_step.hole_waits += 1;
                continue;
            };
            let Placement::Ram(r) = self.place[m as usize] else {
                unreachable!("a miss is RAM-placed by construction")
            };
            mover.ram_to_vram(r, h)?;
            self.pending.push(Pend::Promote { m, r, h });
            self.moves_step.inserts += 1;
            self.moves_step.bytes_h2d += self.stride;
            self.layer_h2d[(m / self.experts) as usize] += self.stride;
            issued = true;
        }
        let need = misses.len().saturating_sub(holes.len());
        if need > 0 {
            let mut keys: Vec<(u64, u32)> = (base..base + n)
                .filter(|&s| {
                    let r = self.vslot[s as usize].record;
                    r != EMPTY && self.touched[r as usize] != self.now
                })
                .map(|s| (self.vslot[s as usize].stamp, s))
                .collect();
            keys.sort_unstable();
            for &(_, vs) in keys.iter().take(need) {
                let Some(sp) = self.rspares.pop() else {
                    break;
                };
                let v = self.vslot[vs as usize].record;
                mover.vram_to_ram(vs, sp)?;
                self.pending.push(Pend::Demote { v, vs, sp });
                self.moves_step.demotions += 1;
                self.moves_step.bytes_d2h += self.stride;
                issued = true;
            }
        }
        if issued {
            mover.end_layer(l)?;
        }
        Ok(())
    }

    /// FN54b: the extra swap spares (after the refill pool).
    fn ring_slots(p: &TierParams) -> std::collections::VecDeque<u32> {
        let b = p.ram_slots + 1 + p.defer_spares + p.refill_spares;
        (b..b + p.spare_ring).collect()
    }

    /// FN54b: pinned RAM slab slots this law needs (`ram_slots` + the spare + every pool).
    #[must_use]
    pub fn slab_slots(p: &TierParams) -> u32 {
        p.ram_slots + 1 + p.defer_spares + p.refill_spares + p.spare_ring
    }

    fn pool_slots(p: &TierParams) -> Vec<u32> {
        let b = p.ram_slots + 1 + p.defer_spares;
        (b..b + p.refill_spares).rev().collect()
    }

    /// FN36: whether the boot armed the demotion-free law (FN42: lent slots refuse it).
    fn demotion_free_law(&self) -> bool {
        self.params.demote == Demote::None && self.params.refill_spares > 0
    }

    /// FN36: whether the demotion-free law runs this step.
    fn demotion_free(&self) -> bool {
        self.params.demote == Demote::None
            && self.params.refill_spares > 0
            && self.params.defer_spares == 0
            && self.complete()
    }

    /// FN36: issue the refill reads chosen by the last `end_step` (the victims' records, NVMe ->
    /// pool slots no placement points at and no copy touches). The worker calls it under the
    /// step (after its launches, while the host waits for the device); `end_step` calls it itself
    /// otherwise (`refill_late`). Returns the reads issued.
    pub fn refill(&mut self, mover: &mut dyn ExpertMover) -> Result<usize, TierError> {
        if self.refill_reads.is_empty() {
            return Ok(0);
        }
        let reads = std::mem::take(&mut self.refill_reads);
        mover.nvme_refill(&reads)?;
        Ok(reads.len())
    }

    /// FN36: `(pool slot, record)` of every swap awaiting publication whose refill has landed
    /// (the byte gate checks them as it checks the placed slots).
    #[must_use]
    pub fn refill_targets(&self) -> Vec<(u32, u32)> {
        self.swaps
            .iter()
            .filter(|s| !self.refill_reads.iter().any(|&(_, sp)| sp == s.sp))
            .map(|s| (s.sp, s.w))
            .collect()
    }

    /// FN36: publish the demotion-free swaps: each victim moves to its refilled pool slot, the
    /// promoted record is copied H2D into the victim's VRAM slot (FN8's per-layer gate orders
    /// the next step's reads after it) and its old RAM slot retires for one step.
    fn publish_swaps(&mut self, mover: &mut dyn ExpertMover) -> Result<(), TierError> {
        let swaps = std::mem::take(&mut self.swaps);
        let mut layer: Option<u32> = None;
        for Swap { m, r, v, w, sp } in swaps {
            let l = m / self.experts;
            if let Some(x) = layer.filter(|&x| x != l) {
                mover.end_layer(x)?;
            }
            layer = Some(l);
            debug_assert_eq!(self.vslot[v as usize].record, w);
            debug_assert_eq!(self.place[m as usize], Placement::Ram(r));
            self.rslot[sp as usize] = Slot {
                record: w,
                stamp: self.vslot[v as usize].stamp,
            };
            self.place[w as usize] = Placement::Ram(sp);
            self.ram_uses[w as usize] = 0;
            mover.ram_to_vram(r, v)?;
            self.rslot[r as usize] = Slot {
                record: EMPTY,
                stamp: 0,
            };
            self.retiring.push(r);
            self.vslot[v as usize] = Slot {
                record: m,
                stamp: self.now,
            };
            self.place[m as usize] = Placement::Vram(v);
            self.prefetched[m as usize] = false;
            self.moves_step.inserts += 1;
            self.moves_step.bytes_h2d += self.stride;
            self.layer_h2d[(m / self.experts) as usize] += self.stride;
            self.moves_step.published += 1;
        }
        if let Some(l) = layer {
            mover.end_layer(l)?;
        }
        Ok(())
    }

    /// FN59b: rank `keys` (`(stamp, slot)`) exactly as the full sort of [`Self::end_layer_fn8`]
    /// does, keeping only the first `k`: LRFU / a policy = (score, stamp, slot) with an empty
    /// slot's score -inf; FN8's LRU = (stamp, slot). Keys are distinct (the slot), so the first
    /// `k` of the total order are unique and come out in the same order.
    fn select_victims(&self, keys: &mut Vec<(u64, u32)>, k: usize) {
        if k == 0 {
            keys.clear();
            return;
        }
        if self.params.lrfu_lambda > 0.0 || self.params.policy.kind != VictimKind::Lrfu {
            let mut sk: Vec<(f64, u64, u32)> = keys
                .iter()
                .map(|&(st, s)| {
                    let r = self.vslot[s as usize].record;
                    let sc = if r == EMPTY {
                        f64::NEG_INFINITY
                    } else {
                        self.vscore(r)
                    };
                    (sc, st, s)
                })
                .collect();
            let cmp = |a: &(f64, u64, u32), b: &(f64, u64, u32)| {
                a.0.total_cmp(&b.0).then((a.1, a.2).cmp(&(b.1, b.2)))
            };
            if k < sk.len() {
                sk.select_nth_unstable_by(k - 1, cmp);
                sk.truncate(k);
            }
            sk.sort_unstable_by(cmp);
            keys.clear();
            keys.extend(sk.iter().map(|&(_, st, s)| (st, s)));
        } else {
            if k < keys.len() {
                keys.select_nth_unstable(k - 1);
                keys.truncate(k);
            }
            keys.sort_unstable();
        }
    }

    /// FN18: record `i`'s LRFU score decayed to this step.
    fn crf_now(&self, i: u32) -> f64 {
        let dt = self.now.saturating_sub(self.crf_at[i as usize]);
        let lam = if self.params.policy.lambda_by_layer.is_empty() {
            self.params.lrfu_lambda
        } else {
            self.params.policy.lambda_by_layer[(i / self.experts) as usize]
        };
        self.crf[i as usize] * lam.powi(dt.min(1 << 20) as i32)
    }

    /// FN48: record `i`'s value under the victim policy (the smallest leaves first): LRFU =
    /// its decayed score (+ the profile prior); Reuse = minus its predicted next use.
    fn vscore(&self, i: u32) -> f64 {
        let v = self.vscore_policy(i);
        if self.sched_prot.get(i as usize).copied().unwrap_or(false) {
            v + SCHED_PROTECT
        } else {
            v
        }
    }

    /// FN48: the policy's own value (see [`Self::vscore`]).
    fn vscore_policy(&self, i: u32) -> f64 {
        let p = &self.params.policy;
        match p.kind {
            VictimKind::Lrfu => {
                let c = self.crf_now(i);
                if p.prior_beta > 0.0 {
                    c + p.prior_beta * (1.0 + self.prof_count[i as usize] as f64).ln()
                } else {
                    c
                }
            }
            VictimKind::Reuse => {
                let i = i as usize;
                if self.ru_n[i] == 0 {
                    return -1e18;
                }
                let last = self.ru_last[i] as f64;
                let age = self.now as f64 - last;
                let gap = if self.ru_n[i] >= 2 {
                    self.ru_gap[i]
                } else {
                    p.reuse_first_gap
                };
                let next = (last + gap).max(self.now as f64 + p.reuse_age * age);
                -next
            }
            VictimKind::Table => {
                let i = i as usize;
                if self.ru_n[i] == 0 {
                    return -1e18;
                }
                let age = table_age(self.now - self.ru_last[i]);
                p.table[table_index(usize::from(self.tb_cls[i]), age)]
            }
        }
    }

    /// FN17: RAM slab slots (the L2 + FN8's spare + the deferred law's spares).
    #[must_use]
    pub fn ram_slab_slots(&self) -> u32 {
        self.rslot.len() as u32
    }

    /// Byte offset of sub-record `k` (w, ws, wref of gate_up; then down) inside a record.
    #[must_use]
    pub fn sub_offset(&self, k: usize) -> u64 {
        self.sub[k].0
    }

    /// FN17: the deferred copies still in flight (issued by the last `end_step`, unpublished).
    #[must_use]
    pub fn pending_copies(&self) -> usize {
        self.pending.len()
    }

    /// FN17: publish the deferred copies NOW (the end of a run: the caller has drained the copy
    /// stream), so every slot's bytes can be checked against its record.
    pub fn settle(&mut self) {
        self.publish();
    }

    /// The next-step prefetch: the highest-score `Fetch` records replace the lowest-score RAM
    /// records (not routed this step) when their score is strictly higher, up to
    /// `prefetch_per_step` reads in one batch.
    fn prefetch(&mut self, mover: &mut dyn ExpertMover) -> Result<(), TierError> {
        let budget = self.params.prefetch_per_step as usize;
        if budget == 0 || self.complete() {
            return Ok(());
        }
        let mut cand: Vec<(f32, u32)> = self
            .place
            .iter()
            .enumerate()
            .filter(|(i, p)| **p == Placement::Fetch && self.score[*i] > 0.0)
            .map(|(i, _)| (self.score[i], i as u32))
            .collect();
        cand.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        cand.truncate(budget);
        if cand.is_empty() {
            return Ok(());
        }
        let mut victims: Vec<(f32, u64, u32)> = self
            .rslot
            .iter()
            .enumerate()
            .filter(|(_, s)| s.record != EMPTY && self.touched[s.record as usize] != self.now)
            .map(|(s, slot)| (self.score[slot.record as usize], slot.stamp, s as u32))
            .collect();
        victims.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        let mut reads = Vec::new();
        let mut vi = victims.into_iter();
        for (sc, rec) in cand {
            let slot = if let Some(s) = self.ram_free.pop() {
                s
            } else {
                match vi.next() {
                    Some((vs, _, s)) if vs < sc => {
                        let old = self.rslot[s as usize].record;
                        self.place[old as usize] = Placement::Fetch;
                        self.prefetched[old as usize] = false;
                        self.moves_step.ram_drops += 1;
                        s
                    }
                    _ => break,
                }
            };
            self.rslot[slot as usize] = Slot {
                record: rec,
                stamp: self.now,
            };
            self.place[rec as usize] = Placement::Ram(slot);
            self.ram_uses[rec as usize] = 0;
            self.prefetched[rec as usize] = true;
            reads.push((u64::from(rec), slot));
        }
        if !reads.is_empty() {
            mover.nvme_to_ram(&reads)?;
            self.moves_step.prefetch_reads += reads.len() as u64;
            self.moves_step.bytes_nvme += reads.len() as u64 * self.stride;
        }
        Ok(())
    }

    /// THE FORCED-MISS CONTROL: forget every placement (VRAM and RAM emptied, everything
    /// `Fetch`). The next touches of every record must count as `nvme`, and the LRU refills from
    /// there. Counters are not reset.
    pub fn evict_all(&mut self) {
        self.place.fill(Placement::Fetch);
        self.vslot.fill(Slot {
            record: EMPTY,
            stamp: 0,
        });
        self.rslot.fill(Slot {
            record: EMPTY,
            stamp: 0,
        });
        self.spare = self.params.ram_slots;
        self.spare_q = Self::ring_slots(&self.params);
        self.ram_free = (0..self.params.ram_slots)
            .chain(self.band_lo..self.band_lo + self.band_open)
            .rev()
            .collect();
        self.bounced.clear();
        self.bounce_ready.clear();
        self.rspares = (self.params.ram_slots + 1
            ..self.params.ram_slots + 1 + self.params.defer_spares)
            .rev()
            .collect();
        self.pending.clear();
        self.pool = Self::pool_slots(&self.params);
        self.retiring.clear();
        self.swaps.clear();
        self.refill_reads.clear();
        self.prefetched.fill(false);
        self.shadow.fill(EMPTY);
    }

    /// Totals since boot.
    #[must_use]
    pub fn totals(&self) -> (PathCounts, MoveCounts) {
        (self.total, self.moves_total)
    }
    /// Per-layer path totals.
    #[must_use]
    pub fn per_layer(&self) -> &[PathCounts] {
        &self.per_layer
    }

    /// FN51f-b (the residency view): layer `l`'s VRAM-resident local expert ids (ascending).
    #[must_use]
    pub fn resident_ids(&self, l: u32) -> Vec<u32> {
        let lo = (l * self.experts) as usize;
        (0..self.experts)
            .filter(|&e| matches!(self.place[lo + e as usize], Placement::Vram(_)))
            .collect()
    }

    /// FN51f-b: H2D bytes promoted into VRAM per layer since boot (inserts + lent fills).
    #[must_use]
    pub fn layer_h2d(&self) -> &[u64] {
        &self.layer_h2d
    }

    /// FN51f-b: the `fnplane` census -- one line a layer: resident records of the rank's
    /// `experts`, the routed hit rate (resident / all routed) by path, the H2D bytes moved.
    #[must_use]
    pub fn plane_lines(&self) -> Vec<String> {
        (0..self.layers)
            .map(|l| {
                let c = &self.per_layer[l as usize];
                let n = c.resident + c.ram + c.nvme;
                let res = self.resident_ids(l).len();
                format!(
                    "fnplane rank{} layer {l:2}: resident {res:3}/{} ({:5.1} %) | routed {n} hit \
                     {:5.1} % (resident {} ram {} nvme {}) | h2d {:.1} MiB",
                    self.rank,
                    self.experts,
                    100.0 * res as f64 / f64::from(self.experts),
                    if n == 0 {
                        0.0
                    } else {
                        100.0 * c.resident as f64 / n as f64
                    },
                    c.resident,
                    c.ram,
                    c.nvme,
                    self.layer_h2d[l as usize] as f64 / 1_048_576.0
                )
            })
            .collect()
    }

    /// FN51f-b: the per-layer residency as JSON (`--fn-plane-dump`).
    #[must_use]
    pub fn plane_json(&self) -> String {
        let mut o = format!(
            "{{\"rank\":{},\"experts\":{},\"first_expert\":{},\"layers\":[",
            self.rank, self.experts, self.first_expert
        );
        for l in 0..self.layers {
            let c = &self.per_layer[l as usize];
            let ids: Vec<String> = self
                .resident_ids(l)
                .iter()
                .map(ToString::to_string)
                .collect();
            if l > 0 {
                o.push(',');
            }
            o.push_str(&format!(
                "{{\"layer\":{l},\"resident\":[{}],\"routed\":{{\"resident\":{},\"ram\":{},\"nvme\":{}}},\"h2d_bytes\":{}}}",
                ids.join(","),
                c.resident,
                c.ram,
                c.nvme,
                self.layer_h2d[l as usize]
            ));
        }
        o.push_str("]}");
        o
    }
    /// Steps ended.
    #[must_use]
    pub fn steps(&self) -> u64 {
        self.steps
    }
    /// Records currently in VRAM / RAM / only on NVMe.
    #[must_use]
    pub fn occupancy(&self) -> (u64, u64, u64) {
        let mut o = (0, 0, 0);
        for p in &self.place {
            match p {
                Placement::Vram(_) => o.0 += 1,
                Placement::Ram(_) => o.1 += 1,
                Placement::Fetch => o.2 += 1,
            }
        }
        o
    }

    /// The boot line.
    #[must_use]
    pub fn boot_line(&self) -> String {
        let gib = |slots: u64| slots as f64 * self.stride as f64 / f64::from(1u32 << 30);
        let (v, r, f) = self.occupancy();
        let p = &self.params;
        format!(
            "xtier rank{}: experts [{}, {}) x {} layers = {} records x {} B | vram {} slots ({:.2} GiB, {:.1}%) \
             ram {} slots + 1 spare ({:.2} GiB) nvme {} | warm vram {v} ram {r} fetch {f} ({}) | lru read-through {:.2} \
             max-inserts {} prefetch {}/step decay {:.2} min-layer {} | profile {}",
            self.rank,
            self.first_expert,
            self.first_expert + self.experts,
            self.layers,
            self.records(),
            self.stride,
            p.vram_slots,
            gib(u64::from(p.vram_slots)),
            100.0 * f64::from(p.vram_slots) / self.records() as f64,
            p.ram_slots,
            gib(u64::from(p.ram_slots) + 1),
            self.records(),
            if f == 0 {
                "complete: NVMe off the step path"
            } else {
                "demand: a Fetch expert is read on its layer"
            },
            p.read_through,
            p.max_inserts,
            p.prefetch_per_step,
            p.history_decay,
            p.min_layer_slots,
            match &self.profile {
                Some((s, t)) => format!("{s} ({t} tokens)"),
                None => "none (uniform split, id order)".into(),
            }
        )
    }

    /// Prometheus text for the metrics plane.
    #[must_use]
    pub fn render_prometheus(&self) -> String {
        let mut o = String::new();
        let r = self.rank;
        let _ = writeln!(
            o,
            "# HELP saber_xtier_routed_total Routed experts by the path that served them."
        );
        let _ = writeln!(o, "# TYPE saber_xtier_routed_total counter");
        for (path, v) in [
            ("resident", self.total.resident),
            ("ram", self.total.ram),
            ("nvme", self.total.nvme),
            ("prefetched", self.total.prefetched),
        ] {
            let _ = writeln!(
                o,
                "saber_xtier_routed_total{{rank=\"{r}\",path=\"{path}\"}} {v}"
            );
        }
        for (l, c) in self.per_layer.iter().enumerate() {
            for (path, v) in [("resident", c.resident), ("ram", c.ram), ("nvme", c.nvme)] {
                let _ = writeln!(
                    o,
                    "saber_xtier_layer_routed_total{{rank=\"{r}\",layer=\"{l}\",path=\"{path}\"}} {v}"
                );
            }
        }
        let m = &self.moves_total;
        let _ = writeln!(o, "# TYPE saber_xtier_moves_total counter");
        for (k, v) in [
            ("insert", m.inserts),
            ("demote", m.demotions),
            ("ram_drop", m.ram_drops),
            ("prefetch_read", m.prefetch_reads),
            ("read_through_layer", m.read_through_layers),
            ("budget_cut", m.budget_cut),
            ("sc_held", m.sc_held),
        ] {
            let _ = writeln!(
                o,
                "saber_xtier_moves_total{{rank=\"{r}\",kind=\"{k}\"}} {v}"
            );
        }
        let _ = writeln!(o, "# TYPE saber_xtier_bytes_total counter");
        for (k, v) in [
            ("h2d", m.bytes_h2d),
            ("d2h", m.bytes_d2h),
            ("nvme", m.bytes_nvme),
        ] {
            let _ = writeln!(o, "saber_xtier_bytes_total{{rank=\"{r}\",dir=\"{k}\"}} {v}");
        }
        // FN51f-b: the residency view (resident records and H2D bytes a layer)
        let _ = writeln!(
            o,
            "# HELP saber_xtier_plane_resident Resident (VRAM) records of the rank's experts a layer."
        );
        let _ = writeln!(o, "# TYPE saber_xtier_plane_resident gauge");
        for l in 0..self.layers {
            let _ = writeln!(
                o,
                "saber_xtier_plane_resident{{rank=\"{r}\",layer=\"{l}\"}} {}",
                self.resident_ids(l).len()
            );
        }
        let _ = writeln!(o, "# TYPE saber_xtier_plane_h2d_bytes_total counter");
        for (l, b) in self.layer_h2d.iter().enumerate() {
            let _ = writeln!(
                o,
                "saber_xtier_plane_h2d_bytes_total{{rank=\"{r}\",layer=\"{l}\"}} {b}"
            );
        }
        let (v, ram, f) = self.occupancy();
        let _ = writeln!(o, "# TYPE saber_xtier_records gauge");
        for (k, x) in [("vram", v), ("ram", ram), ("fetch", f)] {
            let _ = writeln!(o, "saber_xtier_records{{rank=\"{r}\",tier=\"{k}\"}} {x}");
        }
        o
    }

    /// Internal consistency (a test oracle): placements, slots and the spare agree.
    pub fn check_invariants(&self) -> Result<(), String> {
        let mut seen_ram = vec![false; self.rslot.len()];
        let mut seen_vram = vec![false; self.vslot.len()];
        for (i, p) in self.place.iter().enumerate() {
            match *p {
                Placement::Vram(s) => {
                    if self.vslot[s as usize].record != i as u32 || seen_vram[s as usize] {
                        return Err(format!("record {i} -> vram {s} disagrees"));
                    }
                    let l = i as u32 / self.experts;
                    let (b, n) = self.layer_slots[l as usize];
                    let lent_here =
                        self.lent_any(s).is_some() && self.layer_lent[l as usize].contains(&s);
                    if (s < b || s >= b + n) && !lent_here {
                        return Err(format!(
                            "record {i} (layer {l}) in another layer's slot {s}"
                        ));
                    }
                    seen_vram[s as usize] = true;
                }
                Placement::Ram(s) => {
                    if self.rslot[s as usize].record != i as u32 || seen_ram[s as usize] {
                        return Err(format!("record {i} -> ram {s} disagrees"));
                    }
                    seen_ram[s as usize] = true;
                }
                Placement::Fetch => {}
            }
        }
        for (s, v) in self.vslot.iter().enumerate() {
            if v.record != EMPTY && !seen_vram[s] {
                return Err(format!("vram slot {s} holds {} unplaced", v.record));
            }
        }
        // FN42: a lent slot's shadow RAM slot holds the same record (placed in VRAM)
        let mut shadows = 0usize;
        for (i, &sh) in self.shadow.iter().enumerate() {
            let s = self.base_slots + i as u32;
            let r = self.vslot[s as usize].record;
            if !self.lend_on[i] && (r != EMPTY || sh != EMPTY) {
                return Err(format!("reclaimed slot {s} holds {r} / shadow {sh}"));
            }
            // FN107: an exclusive slot's record has NO RAM copy
            if self.excl[i] {
                if !self.lend_on[i] || sh != EMPTY {
                    return Err(format!(
                        "exclusive slot {s}: lent {} shadow {sh}",
                        self.lend_on[i]
                    ));
                }
                continue;
            }
            if (r == EMPTY) != (sh == EMPTY) {
                return Err(format!("lent slot {s}: record {r} shadow {sh}"));
            }
            if r != EMPTY {
                if self.rslot[sh as usize].record != r || seen_ram[sh as usize] {
                    return Err(format!("lent slot {s}: shadow {sh} does not hold {r}"));
                }
                seen_ram[sh as usize] = true;
                shadows += 1;
            }
        }
        for (s, v) in self.rslot.iter().enumerate() {
            if v.record != EMPTY && !seen_ram[s] {
                return Err(format!("ram slot {s} holds {} unplaced", v.record));
            }
        }
        for &sp in std::iter::once(&self.spare).chain(self.spare_q.iter()) {
            if self.rslot[sp as usize].record != EMPTY {
                return Err(format!("the spare {sp} holds a record"));
            }
        }
        let used = seen_ram.iter().filter(|&&b| b).count();
        debug_assert!(shadows <= used);
        // FN17: a deferred demotion's spare holds its copy, unpublished (not a free slot)
        let demoting = self
            .pending
            .iter()
            .filter(|p| matches!(p, Pend::Demote { .. }))
            .count();
        // FN36: the refill pool's free, retiring and in-flight slots
        let refill = self.pool.len() + self.retiring.len() + self.swaps.len();
        // FN53b: the band's closed slots and the bounce's slots not holding a record this step
        let closed = (self.params.band_slots - self.band_open) as usize;
        let bounce_idle = self.params.bounce_slots as usize - self.bounced.len();
        for s in self.band_lo + self.band_open..self.bounce_lo {
            if self.rslot[s as usize].record != EMPTY || self.ram_free.contains(&s) {
                return Err(format!("closed band slot {s} in use"));
            }
        }
        if used
            + closed
            + bounce_idle
            + self.ram_free.len()
            + 1
            + self.spare_q.len()
            + self.rspares.len()
            + demoting
            + refill
            != self.rslot.len()
        {
            return Err(format!(
                "ram accounting: {used} used + {} free + 1 spare + {} deferred spares + {demoting} \
                 in flight != {}",
                self.ram_free.len(),
                self.rspares.len(),
                self.rslot.len()
            ));
        }
        for p in &self.pending {
            match *p {
                Pend::Promote { m, r, h } => {
                    if self.place[m as usize] != Placement::Ram(r)
                        || self.vslot[h as usize].record != EMPTY
                    {
                        return Err(format!("pending promotion {m} ram {r} -> hole {h} moved"));
                    }
                }
                Pend::Demote { v, vs, sp } => {
                    if self.place[v as usize] != Placement::Vram(vs)
                        || self.rslot[sp as usize].record != EMPTY
                    {
                        return Err(format!(
                            "pending demotion {v} vram {vs} -> spare {sp} moved"
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

/// One step's counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StepReport {
    /// Routed experts by path.
    pub paths: PathCounts,
    /// What moved.
    pub moves: MoveCounts,
}
