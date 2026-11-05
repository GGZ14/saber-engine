//! FN43: THE PREFILL STAGING RING (docs/EXEC.md §12, "the ring law"; LEDGER FN43).
//!
//! A prefill chunk's MoE reads its RAM-resident experts over UVA in place, in-kernel, so the host
//! link idles while the layer's attention / GDN / HC work runs. The ring moves those bytes into the
//! idle window: a VRAM ring of `buffers x slots` expert strides (the memory is the tier's LENDABLE
//! slots -- FN42's inclusive lending law -- lent to the tier while no staged pass runs, reclaimed
//! by a flip for the pass), filled on the tier's ONE ordered copy stream: layer `L`'s RAM records,
//! in [`ExpertTier::stage_order`] (every one up to `slots`; a layer's routes are unknown until its
//! router runs), into buffer `L % buffers`, the copies waiting until that buffer's previous user
//! (layer `L - buffers`) has run its experts. Layer `L`'s grouped GEMMs read the ring through its
//! `xptr` entries ([`StagePlan::retarget`]) after the layer's ARRIVAL mark (the consumer waits
//! between the route half and the expert half); a RAM record past the ring's `slots` keeps its UVA
//! entry (the fallback, counted: [`StageCounts::uva`]).
//!
//! **The staging byte law:** a ring slot is a byte copy of the record's RAM slot, and a retargeted
//! entry is the same sub-record offset in it -- so every grouped GEMM reads the bytes it read over
//! UVA and every layer's output is the UVA run's (host: `tests/stage_ring.rs`; card: the engine's
//! row sha256 / probe gates, LEDGER FN43).
//!
//! This module is the host half: the plan (which record goes to which ring slot), the table
//! rewrite and the which-path counters. The engine (saber-cli `flashnext::worker`) issues the
//! copies and the marks.

#![forbid(unsafe_code)]

use crate::expert_tier::{ExpertTier, Placement};

/// LR-H: the ring's ORDER KEY (`--xtier-stage-order`): which RAM records of a layer take its
/// `slots` ring slots. It moves WHICH records are copies in VRAM, never a byte a GEMM reads (the
/// staging byte law above: a staged entry and its UVA entry point at identical bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StageKey {
    /// FN43 (the default, today's bytes and schedule): the warm profile's route count, then the
    /// LRFU score ([`ExpertTier::stage_order`]).
    #[default]
    Profile,
    /// LR-H (an ARM): the decayed count of SERVED staged prefill passes that routed the record
    /// ([`ExpertTier::note_prefill_routes`]), then the profile, then the LRFU score.
    Served,
}

impl StageKey {
    /// The flag's name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            StageKey::Profile => "profile",
            StageKey::Served => "served",
        }
    }
    /// The other key (the counterfactual a pass prices on its `fnringpass` line).
    #[must_use]
    pub fn other(self) -> StageKey {
        match self {
            StageKey::Profile => StageKey::Served,
            StageKey::Served => StageKey::Profile,
        }
    }
    /// Parse `profile` | `served`.
    pub fn parse(s: &str) -> Result<StageKey, String> {
        match s {
            "profile" => Ok(StageKey::Profile),
            "served" => Ok(StageKey::Served),
            o => Err(format!("stage order {o:?}: profile | served")),
        }
    }
}

/// LR-H: the `served` key's decay per staged pass.
pub const PF_SEEN_DECAY: f32 = 0.9;

/// The ring's shape: `buffers` layer buffers of `slots` expert strides each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingShape {
    /// Layer buffers (1 = the next layer's copies wait for this layer's experts; 2 = double
    /// buffered: layer `L + 1` fills while layer `L` computes).
    pub buffers: u32,
    /// Expert strides per buffer.
    pub slots: u32,
}

impl RingShape {
    /// Ring slots in all.
    #[must_use]
    pub fn total(&self) -> u32 {
        self.buffers * self.slots
    }
    /// The buffer layer `layer` stages into.
    #[must_use]
    pub fn buffer(&self, layer: u32) -> u32 {
        layer % self.buffers
    }
    /// Ring slot of layer `layer`'s `k`-th staged record.
    #[must_use]
    pub fn slot(&self, layer: u32, k: u32) -> u32 {
        self.buffer(layer) * self.slots + k
    }
}

/// One layer's staging.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LayerStage {
    /// `(RAM slot, ring slot)` per staged record, in copy order.
    pub copies: Vec<(u32, u32)>,
    /// The staged local experts (same order as `copies`).
    pub experts: Vec<u32>,
    /// RAM records of the layer NOT staged (past the ring's slots): their entries stay UVA.
    pub overflow: u32,
}

/// Which path each routed (layer, local expert) of a staged pass took, plus the ring's waste.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StageCounts {
    /// Routed and read from the ring (a staged RAM record).
    pub ring: u64,
    /// Routed RAM records past the ring's slots: read over UVA (the fallback).
    pub uva: u64,
    /// Routed VRAM records (the tier's slots; the ring does not touch them).
    pub vram: u64,
    /// Staged records no row routed (link bytes spent for nothing).
    pub wasted: u64,
    /// Records staged.
    pub staged: u64,
    /// LR-H: the swaps an ideal order in the same slots would make: per layer
    /// `min(wasted, uva)` (a staged-unrouted record beside a routed record left on UVA).
    pub misordered: u64,
}

impl std::ops::AddAssign for StageCounts {
    fn add_assign(&mut self, o: StageCounts) {
        self.ring += o.ring;
        self.uva += o.uva;
        self.vram += o.vram;
        self.wasted += o.wasted;
        self.staged += o.staged;
        self.misordered += o.misordered;
    }
}

/// A staged pass's plan: per layer, the records copied into the ring (placements are frozen for
/// the pass: the tier changes only at `end_step`, after the pass's fence).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagePlan {
    /// The ring's shape.
    pub shape: RingShape,
    /// Per layer.
    pub layers: Vec<LayerStage>,
}

impl StagePlan {
    /// Plan every layer of `tier` at its current placements: the first `shape.slots` records of
    /// [`ExpertTier::stage_order`] per layer.
    #[must_use]
    pub fn plan(tier: &ExpertTier, layers: u32, shape: RingShape) -> StagePlan {
        Self::plan_by(tier, layers, shape, StageKey::Profile)
    }

    /// LR-H: [`Self::plan`] with the ring's order key: the first `shape.slots` records of
    /// [`ExpertTier::stage_order_by`] per layer (`Profile` = [`Self::plan`] exactly).
    #[must_use]
    pub fn plan_by(tier: &ExpertTier, layers: u32, shape: RingShape, key: StageKey) -> StagePlan {
        let layers = (0..layers)
            .map(|l| {
                let order = tier.stage_order_by(l, key);
                let take = order.len().min(shape.slots as usize);
                LayerStage {
                    copies: order[..take]
                        .iter()
                        .enumerate()
                        .map(|(k, &(_, r))| (r, shape.slot(l, k as u32)))
                        .collect(),
                    experts: order[..take].iter().map(|&(e, _)| e).collect(),
                    overflow: (order.len() - take) as u32,
                }
            })
            .collect();
        StagePlan { shape, layers }
    }

    /// Records staged over the pass.
    #[must_use]
    pub fn copies(&self) -> u64 {
        self.layers.iter().map(|s| s.copies.len() as u64).sum()
    }

    /// The largest layer's RAM record count (staged + overflow): the slots a buffer needs to
    /// stage every RAM record of every layer.
    #[must_use]
    pub fn max_layer_ram(&self) -> u32 {
        self.layers
            .iter()
            .map(|s| s.copies.len() as u32 + s.overflow)
            .max()
            .unwrap_or(0)
    }

    /// Point layer `layer`'s staged experts' entries of the FN4 `xptr` tables (as
    /// [`ExpertTier::fill_xptr`] filled them) at their ring slots: the same sub-record offsets in
    /// ring slot `s` at `ring_base + s * stride`. Returns the entries' experts rewritten.
    pub fn retarget(
        &self,
        tier: &ExpertTier,
        layer: u32,
        ring_base: u64,
        gate_up: &mut [u64],
        down: &mut [u64],
    ) -> usize {
        let st = tier.stride();
        let ls = &self.layers[layer as usize];
        for (&e, &(_, s)) in ls.experts.iter().zip(&ls.copies) {
            let base = ring_base + u64::from(s) * st;
            let e = e as usize;
            for k in 0..3 {
                gate_up[3 * e + k] = base + tier.sub_offset(k);
                down[3 * e + k] = base + tier.sub_offset(3 + k);
            }
        }
        ls.experts.len()
    }

    /// The which-path counts of layer `layer` from its logged routes (`routed`: GLOBAL expert ids,
    /// any rank's; another rank's are skipped), at the pass's (frozen) placements.
    #[must_use]
    pub fn classify(&self, tier: &ExpertTier, layer: u32, routed: &[u32]) -> StageCounts {
        self.classify_seen(tier, layer, &routed_seen(tier, routed))
    }

    /// [`Self::classify`] from the layer's [`routed_seen`] map (LR-H: one map prices the plan,
    /// the other key's counterfactual plan and the served key's note).
    #[must_use]
    pub fn classify_seen(&self, tier: &ExpertTier, layer: u32, seen: &[bool]) -> StageCounts {
        let (first, n) = tier.experts_geometry();
        let ls = &self.layers[layer as usize];
        let mut staged = vec![false; n as usize];
        for &e in &ls.experts {
            staged[e as usize] = true;
        }
        let mut c = StageCounts {
            staged: ls.experts.len() as u64,
            ..StageCounts::default()
        };
        for e in 0..n {
            let (s, r) = (staged[e as usize], seen[e as usize]);
            if s && !r {
                c.wasted += 1;
            }
            if !r {
                continue;
            }
            if s {
                c.ring += 1;
            } else {
                match tier.placement(layer, first + e) {
                    Some(Placement::Ram(_)) => c.uva += 1,
                    _ => c.vram += 1,
                }
            }
        }
        c.misordered = c.wasted.min(c.uva);
        c
    }
}

/// LR-H: which of this rank's local experts `routed` (GLOBAL expert ids, any rank's; another
/// rank's are skipped) names, as a map over the tier's local experts.
#[must_use]
pub fn routed_seen(tier: &ExpertTier, routed: &[u32]) -> Vec<bool> {
    let (first, n) = tier.experts_geometry();
    let mut seen = vec![false; n as usize];
    for &g in routed {
        if g >= first && g < first + n {
            seen[(g - first) as usize] = true;
        }
    }
    seen
}
