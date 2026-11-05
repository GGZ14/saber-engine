//! The Flash-Next EP=2 byte ledger: every resident class of a rank, the runtime state at the
//! planned maximum sequences, the arenas — and the expert slots as the REMAINDER (plan §3, LEDGER
//! FN1 §3 / FN2 §2, docs/EXEC.md §14).
//!
//! The weights come from the pack's own `plan.json` `ledger` (the bytes the loader will place:
//! this rank's placement plus `shared/`), minus the `experts` class, which is the tier's. The
//! runtime classes are computed from the model geometry (FN1 §3's per-token / per-sequence
//! numbers) and the serving parameters. The remainder of the per-card budget, in whole
//! `experts.xtier` strides, is the VRAM slot count; [`Ep2Budget::declare`] reserves every class in
//! the [`Ledger`], which REFUSES an over-commit by name. The host half sizes the pinned RAM L2 per
//! rank beside the box's own use and the 27B's pinned reservation.

#![forbid(unsafe_code)]

use crate::ledger::{Ledger, LedgerError, Pool};

const MIB: u64 = 1 << 20;
/// FN42: the ledger line of the KV bytes lent to the expert tier.
pub const KV_LENT: &str = "xtier.kv_lent";
const GIB: u64 = 1 << 30;

/// One named class of bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Class {
    /// Ledger name.
    pub name: String,
    /// Bytes.
    pub bytes: u64,
}

impl Class {
    fn new(name: &str, bytes: u64) -> Class {
        Class {
            name: name.to_string(),
            bytes,
        }
    }
}

/// Why the budget refused.
#[derive(Debug, thiserror::Error)]
pub enum BudgetError {
    /// The pack's plan ledger is not what the pack law says.
    #[error("plan ledger: {0}")]
    Plan(String),
    /// The fixed classes alone exceed the budget.
    #[error(
        "rank {rank}: weights {weights} B + runtime {runtime} B exceed the {budget} B budget by {over} B \
         before a single expert slot"
    )]
    NoRoom {
        /// Rank.
        rank: u32,
        /// Weight bytes.
        weights: u64,
        /// Runtime bytes.
        runtime: u64,
        /// Budget.
        budget: u64,
        /// Shortfall.
        over: u64,
    },
    /// The ledger refused a reservation.
    #[error(transparent)]
    Ledger(#[from] LedgerError),
}

/// The GDN recurrent state's stored dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateDtype {
    /// f16 (the 27B's stored-state default since 2026-09-23; 27.0 MiB/sequence-slot/rank).
    F16,
    /// f32 (upstream's `mamba_ssm_dtype`; 54.0 MiB).
    F32,
}

/// The runtime parameters the budget is planned at.
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeParams {
    /// Maximum live sequences.
    pub max_seqs: u64,
    /// KV pool tokens per rank.
    pub kv_tokens: u64,
    /// KV bytes per token per rank (12 QSA layers + the MTP layer; 13,312 bf16, 6,656 fp8 — fp8 KV
    /// is a numerics choice, not a default).
    pub kv_bytes_per_token: u64,
    /// QSA indexer compressed-key bytes per token (replicated): 832.
    pub indexer_bytes_per_token: u64,
    /// GDN state slots per sequence (MTP-k verify: 1 + k).
    pub gdn_slots_per_seq: u64,
    /// GDN state dtype.
    pub gdn_state: StateDtype,
    /// Prefill activation arena (one, at the widest rung; the C7 law).
    pub prefill_arena: u64,
    /// Decode + head arena.
    pub decode_arena: u64,
}

impl Default for RuntimeParams {
    fn default() -> Self {
        RuntimeParams {
            max_seqs: 8,
            kv_tokens: 131_072,
            kv_bytes_per_token: 13_312,
            indexer_bytes_per_token: 832,
            gdn_slots_per_seq: 5,
            gdn_state: StateDtype::F16,
            prefill_arena: 1024 * MIB,
            decode_arena: 512 * MIB,
        }
    }
}

/// Model geometry constants of the runtime classes (FN1 §3, TP=2 rank).
pub mod geometry {
    /// GDN layers.
    pub const GDN_LAYERS: u64 = 36;
    /// GDN v-heads per rank × k dim × v dim.
    pub const GDN_STATE_ELEMS_PER_LAYER: u64 = 24 * 128 * 128;
    /// GDN conv channels per rank × (kernel − 1), bf16.
    pub const GDN_CONV_BYTES_PER_LAYER: u64 = 5120 * 3 * 2;
    /// PLE short-conv state rows at spec 0: (kernel − 1) × dilation = 9 (the history's minimum).
    pub const PLE_CONV_STATE_ROWS: u64 = 9;
    /// One PLE conv history row: 10,240 × bf16 (replicated).
    pub const PLE_CONV_ROW_BYTES: u64 = 10_240 * 2;
    /// FN7's line (9 rows, spec 0, ONE slot = 180 KiB/sequence): the pre-FN9 CONTROL value only.
    pub const PLE_CONV_BYTES_FN7: u64 = PLE_CONV_STATE_ROWS * PLE_CONV_ROW_BYTES;
    /// FN6's conv-state geometry (EXEC §13, `saber_kernels::fn6::PleConvState`): a history of
    /// `9 + spec` rows in TWO ping-pong slots (the step reads one, writes the other) --
    /// 532,480 B/sequence at MTP-4 (FN9's correction of FN7's 180 KiB).
    #[must_use]
    pub const fn ple_conv_bytes_per_seq(spec: u64) -> u64 {
        2 * (PLE_CONV_STATE_ROWS + spec) * PLE_CONV_ROW_BYTES
    }
    /// One `experts.xtier` record stride.
    pub const EXPERT_STRIDE: u64 = 2_617_344;
    /// Records per rank (48 layers × 256 experts).
    pub const EXPERT_RECORDS: u64 = 12_288;
    /// The xptr tables: 48 layers × 2 GEMMs × 257 entries × 3 pointers × 8 B.
    pub const XPTR_BYTES: u64 = 48 * 2 * 257 * 3 * 8;
}

impl RuntimeParams {
    /// The runtime classes of one rank.
    #[must_use]
    pub fn classes(&self) -> Vec<Class> {
        use geometry::*;
        let state_b = match self.gdn_state {
            StateDtype::F16 => 2,
            StateDtype::F32 => 4,
        };
        let slots = self.max_seqs * self.gdn_slots_per_seq;
        vec![
            Class::new("fn.kv", self.kv_tokens * self.kv_bytes_per_token),
            Class::new(
                "fn.indexer_keys",
                self.kv_tokens * self.indexer_bytes_per_token,
            ),
            Class::new(
                "fn.gdn_state",
                slots * GDN_LAYERS * GDN_STATE_ELEMS_PER_LAYER * state_b,
            ),
            Class::new("fn.gdn_conv", slots * GDN_LAYERS * GDN_CONV_BYTES_PER_LAYER),
            Class::new(
                "fn.ple_conv",
                self.max_seqs * ple_conv_bytes_per_seq(self.gdn_slots_per_seq.saturating_sub(1)),
            ),
            Class::new("fn.arena.prefill", self.prefill_arena),
            Class::new("fn.arena.decode", self.decode_arena),
            Class::new("xtier.xptr", XPTR_BYTES),
        ]
    }
}

impl RuntimeParams {
    /// FN45a: the runtime classes under the TIERED KV cache (`saber_mem::kv_tier`): `fn.kv` and
    /// `fn.indexer_keys` give way to `kv.hot` (the plan's VRAM hot pool) and `kv.index` (the
    /// compressed index cache at the plan's `max_seqs x max_ctx`, VRAM-resident); `kv.ram` is a
    /// HOST line ([`crate::kv_tier::KvTierPlan::declare`] reserves it on `Pool::HostPinned`).
    #[must_use]
    pub fn classes_tiered(&self, kv: &crate::kv_tier::KvTierPlan) -> Vec<Class> {
        let mut out: Vec<Class> = self
            .classes()
            .into_iter()
            .filter(|c| c.name != "fn.kv" && c.name != "fn.indexer_keys")
            .collect();
        out.insert(0, Class::new(crate::kv_tier::KV_INDEX, kv.index_bytes()));
        out.insert(0, Class::new(crate::kv_tier::KV_HOT, kv.hot_bytes()));
        out
    }
}

/// The weight classes of `rank` from the pack's `plan.json` (its `ledger` array): this rank's
/// placement and `shared/`, the `experts` class excluded (the tier owns it). Returns the classes
/// and the rank's expert-class bytes (for the check against `records × payload`).
pub fn weights_from_plan(plan_json: &str, rank: u32) -> Result<(Vec<Class>, u64), BudgetError> {
    let v: serde_json::Value =
        serde_json::from_str(plan_json).map_err(|e| BudgetError::Plan(e.to_string()))?;
    let rows = v["ledger"]
        .as_array()
        .ok_or_else(|| BudgetError::Plan("no ledger array".into()))?;
    let me = format!("rank{rank}");
    let mut out = Vec::new();
    let mut experts = 0;
    for r in rows {
        let (Some(p), Some(c), Some(b)) = (
            r["placement"].as_str(),
            r["class"].as_str(),
            r["bytes"].as_u64(),
        ) else {
            return Err(BudgetError::Plan(format!("malformed ledger row {r}")));
        };
        if p != me && p != "shared" {
            continue;
        }
        if c == "experts" {
            if p != me {
                return Err(BudgetError::Plan("experts placed in shared/".into()));
            }
            experts += b;
            continue;
        }
        out.push(Class::new(&format!("fn.w.{p}.{c}"), b));
    }
    if experts == 0 {
        return Err(BudgetError::Plan(format!("no experts class for {me}")));
    }
    Ok((out, experts))
}

/// One rank's VRAM plan.
#[derive(Debug, Clone)]
pub struct Ep2Budget {
    /// Rank.
    pub rank: u32,
    /// The per-card budget.
    pub budget: u64,
    /// Weight classes.
    pub weights: Vec<Class>,
    /// Runtime classes.
    pub runtime: Vec<Class>,
    /// Expert record stride.
    pub stride: u64,
    /// Records the rank owns.
    pub records: u64,
    /// VRAM expert slots (the remainder, in whole strides, at most `records`).
    pub slots: u64,
    /// VRAM STAGING slots (FN8's prefill staging ring, two halves), taken out of the remainder
    /// before the LRU's slots: ledger line `xtier.stage_ring`. 0 = no ring.
    pub stage_slots: u64,
}

impl Ep2Budget {
    /// Plan a rank: the slots are what is left after the weights and the runtime.
    pub fn plan(
        rank: u32,
        budget: u64,
        weights: Vec<Class>,
        runtime: Vec<Class>,
        stride: u64,
        records: u64,
    ) -> Result<Ep2Budget, BudgetError> {
        let w: u64 = weights.iter().map(|c| c.bytes).sum();
        let r: u64 = runtime.iter().map(|c| c.bytes).sum();
        if w + r > budget {
            return Err(BudgetError::NoRoom {
                rank,
                weights: w,
                runtime: r,
                budget,
                over: w + r - budget,
            });
        }
        let slots = ((budget - w - r) / stride).min(records);
        Ok(Ep2Budget {
            rank,
            budget,
            weights,
            runtime,
            stride,
            records,
            slots,
            stage_slots: 0,
        })
    }

    /// Reserve `stage_slots` of the remainder for the prefill staging ring (FN8): the LRU keeps
    /// the rest. Refused when the ring would take every slot.
    pub fn with_stage_ring(mut self, stage_slots: u64) -> Result<Ep2Budget, BudgetError> {
        let total = self.slots + self.stage_slots;
        if stage_slots >= total {
            return Err(BudgetError::NoRoom {
                rank: self.rank,
                weights: self.weight_bytes(),
                runtime: self.runtime_bytes() + stage_slots * self.stride,
                budget: self.budget,
                over: (stage_slots + 1 - total) * self.stride,
            });
        }
        self.slots = total - stage_slots;
        self.stage_slots = stage_slots;
        Ok(self)
    }

    /// Weight bytes.
    #[must_use]
    pub fn weight_bytes(&self) -> u64 {
        self.weights.iter().map(|c| c.bytes).sum()
    }
    /// Runtime bytes.
    #[must_use]
    pub fn runtime_bytes(&self) -> u64 {
        self.runtime.iter().map(|c| c.bytes).sum()
    }
    /// VRAM slot bytes.
    #[must_use]
    pub fn slot_bytes(&self) -> u64 {
        self.slots * self.stride
    }
    /// Records that cannot be VRAM-resident.
    #[must_use]
    pub fn complement(&self) -> u64 {
        self.records - self.slots
    }
    /// The staging ring's bytes.
    #[must_use]
    pub fn stage_bytes(&self) -> u64 {
        self.stage_slots * self.stride
    }

    /// Declare `Pool::Vram(rank)` at the budget and reserve every class, the expert slots last.
    /// `extra_slots` asks for more slots than planned: the over-commit control.
    pub fn declare(&self, ledger: &mut Ledger, extra_slots: u64) -> Result<(), BudgetError> {
        let pool = Pool::Vram(self.rank);
        ledger.declare(pool, self.budget)?;
        for c in self.weights.iter().chain(self.runtime.iter()) {
            ledger.reserve(pool, &c.name, c.bytes)?;
        }
        if self.stage_slots > 0 {
            ledger.reserve(pool, "xtier.stage_ring", self.stage_slots * self.stride)?;
        }
        ledger.reserve(
            pool,
            "xtier.vram_slots",
            (self.slots + extra_slots) * self.stride,
        )?;
        Ok(())
    }

    /// FN42: THE LENDING LINE -- `lent` bytes of the KV class `kv_class` (declared at `kv_bytes`)
    /// are lent to the expert tier as slots: the ledger holds `kv_class` at `kv_bytes - lent` and
    /// `xtier.kv_lent` at `lent` (no line at 0), so the pool's total never moves and every byte
    /// stays declared once. Called at every lend / reclaim; refuses a lend past the class.
    pub fn relend(
        &self,
        ledger: &mut Ledger,
        kv_class: &str,
        kv_bytes: u64,
        lent: u64,
    ) -> Result<(), BudgetError> {
        self.relend_as(ledger, kv_class, kv_bytes, lent, KV_LENT)
    }

    /// FN43: [`Self::relend`] with the lent bytes on the line `lent_class` (one line per lending
    /// client: the staging ring lends `xtier.stage_ring` as `xtier.ring_lent`).
    pub fn relend_as(
        &self,
        ledger: &mut Ledger,
        kv_class: &str,
        kv_bytes: u64,
        lent: u64,
        lent_class: &str,
    ) -> Result<(), BudgetError> {
        if lent > kv_bytes {
            return Err(BudgetError::Plan(format!(
                "rank {}: lending {lent} B of the {kv_bytes} B {kv_class}",
                self.rank
            )));
        }
        let pool = Pool::Vram(self.rank);
        let had = ledger.release(pool, kv_class)?;
        let was = ledger.release(pool, lent_class).unwrap_or(0);
        if had + was != kv_bytes {
            return Err(BudgetError::Plan(format!(
                "rank {}: {kv_class} {had} B + {lent_class} {was} B != {kv_bytes} B",
                self.rank
            )));
        }
        ledger.reserve(pool, kv_class, kv_bytes - lent)?;
        if lent > 0 {
            ledger.reserve(pool, lent_class, lent)?;
        }
        Ok(())
    }

    /// The boot line.
    #[must_use]
    pub fn boot_line(&self) -> String {
        let g = |b: u64| b as f64 / GIB as f64;
        let rt: Vec<String> = self
            .runtime
            .iter()
            .map(|c| format!("{} {:.3}", c.name.trim_start_matches("fn."), g(c.bytes)))
            .collect();
        format!(
            "fnbudget rank{}: budget {:.3} GiB = weights {:.3} GiB + runtime {:.3} GiB ({}) + {}expert slots {} x {} B \
             = {:.3} GiB ({:.1}% of {} records resident; {} not) + {} B unassigned",
            self.rank,
            g(self.budget),
            g(self.weight_bytes()),
            g(self.runtime_bytes()),
            rt.join(", "),
            if self.stage_slots > 0 {
                format!(
                    "stage ring {} x {} B = {:.3} GiB + ",
                    self.stage_slots,
                    self.stride,
                    g(self.stage_bytes())
                )
            } else {
                String::new()
            },
            self.slots,
            self.stride,
            g(self.slot_bytes()),
            100.0 * self.slots as f64 / self.records as f64,
            self.records,
            self.complement(),
            self.budget - self.weight_bytes() - self.runtime_bytes() - self.slot_bytes() - self.stage_bytes()
        )
    }
}

/// The host side: the pinned RAM L2 per rank beside the box and the 27B's pinned use.
#[derive(Debug, Clone, PartialEq)]
pub struct HostBudget {
    /// `MemTotal` of the box.
    pub mem_total: u64,
    /// What the box itself uses (OS, services): 8 GiB on this box (`free -g`).
    pub box_used: u64,
    /// RAM left free for the OS page cache and the other workstreams: 16 GiB.
    pub free_floor: u64,
    /// The serve plane's own `Pool::HostPinned` (park staging; the 27B's pinned use): 4 GiB.
    pub serve_pinned: u64,
    /// Ranks on this host.
    pub ranks: u64,
    /// PLE RAM row cache per rank.
    pub ple_cache_per_rank: u64,
    /// FN45b4: the tiered KV's pinned RAM slab per rank (`kv.ram`: every block of 8 x the max
    /// context; 0 on the flat pools).
    pub kv_ram_per_rank: u64,
    /// FN99: the token embedding half in pinned RAM per rank (`--fn-embed-placement ram`, read in
    /// place by the drafter chain; 0 = the VRAM half, FN22's).
    pub embed_per_rank: u64,
}

impl HostBudget {
    /// This box: 60 GiB, the brief's reserves.
    #[must_use]
    pub fn this_box(mem_total: u64) -> HostBudget {
        HostBudget {
            mem_total,
            box_used: 8 * GIB,
            free_floor: 16 * GIB,
            serve_pinned: 4 * GIB,
            ranks: 2,
            ple_cache_per_rank: GIB,
            kv_ram_per_rank: 0,
            embed_per_rank: 0,
        }
    }

    /// FN99: with the embedding half of `bytes` a rank in pinned RAM.
    #[must_use]
    pub fn with_embed(mut self, bytes: u64) -> HostBudget {
        self.embed_per_rank = bytes;
        self
    }

    /// FN45b4: with the tiered KV's pinned slab of `bytes` a rank.
    #[must_use]
    pub fn with_kv_ram(mut self, bytes: u64) -> HostBudget {
        self.kv_ram_per_rank = bytes;
        self
    }

    /// FN45b4: the bytes the boot pins: the serve plane + every rank's L2 (+ spare), PLE cache
    /// and KV slab. `l2_slots` per rank.
    #[must_use]
    pub fn pinned(&self, l2_slots: &[u64], stride: u64) -> u64 {
        self.serve_pinned
            + l2_slots
                .iter()
                .map(|&l| {
                    (l + 1) * stride
                        + self.ple_cache_per_rank
                        + self.kv_ram_per_rank
                        + self.embed_per_rank
                })
                .sum::<u64>()
    }

    /// FN45b4: refuse a boot whose pinned bytes ([`Self::pinned`]) exceed [`Self::pinnable`]
    /// (the expert RAM cache is NOT shrunk to make room: that needs a replay number).
    pub fn check(&self, l2_slots: &[u64], stride: u64) -> Result<(), BudgetError> {
        let (want, have) = (self.pinned(l2_slots, stride), self.pinnable());
        if want > have {
            let g = |b: u64| b as f64 / GIB as f64;
            return Err(BudgetError::Plan(format!(
                "the host pins {:.2} GiB (serve {:.2} + L2 + PLE + KV RAM {} x {:.2}) past the \
                 {:.2} GiB pinnable (MemTotal {:.2} - box {:.0} - free floor {:.0})",
                g(want),
                g(self.serve_pinned),
                l2_slots.len(),
                g(self.kv_ram_per_rank),
                g(have),
                g(self.mem_total),
                g(self.box_used),
                g(self.free_floor)
            )));
        }
        Ok(())
    }

    /// FN45b4: the largest max context (a multiple of `step` tokens) whose KV slab (8 x ctx x
    /// `kv_token_bytes` a rank, + `kv_align` bytes) fits beside `l2_slots` -- the tiered KV's
    /// ceiling on this host. 0 when none fits.
    #[must_use]
    pub fn kv_ceiling(
        &self,
        l2_slots: &[u64],
        stride: u64,
        kv_token_bytes: u64,
        kv_align: u64,
        step: u64,
    ) -> u64 {
        let base = self.clone().with_kv_ram(0).pinned(l2_slots, stride);
        let left = self.pinnable().saturating_sub(base) / l2_slots.len().max(1) as u64;
        let per = left.saturating_sub(kv_align) / (8 * kv_token_bytes);
        per / step * step
    }

    /// Bytes the engine may pin on this host.
    #[must_use]
    pub fn pinnable(&self) -> u64 {
        self.mem_total
            .saturating_sub(self.box_used + self.free_floor)
    }

    /// The default RAM L2 slots per rank: the complement (every expert in VRAM or RAM: the NVMe
    /// off the step path) when it fits beside the serve pinned bytes and the PLE caches, else
    /// what fits. One more slot per rank is the swap spare.
    #[must_use]
    pub fn default_l2_slots(&self, complement: u64, stride: u64) -> u64 {
        let avail = self.pinnable().saturating_sub(
            self.serve_pinned + self.ranks * (self.ple_cache_per_rank + self.embed_per_rank),
        );
        let fit = (avail / self.ranks / stride).saturating_sub(1);
        complement.min(fit)
    }

    /// Declare `Pool::HostPinned` at the pinnable bytes and reserve the serve plane's bytes, every
    /// rank's L2 (+ spare) and PLE cache. `l2_slots` per rank.
    pub fn declare(
        &self,
        ledger: &mut Ledger,
        l2_slots: u64,
        stride: u64,
    ) -> Result<(), BudgetError> {
        ledger.declare(Pool::HostPinned, self.pinnable())?;
        ledger.reserve(Pool::HostPinned, "serve.pinned", self.serve_pinned)?;
        for r in 0..self.ranks {
            ledger.reserve(
                Pool::HostPinned,
                &format!("xtier.l2.rank{r}"),
                (l2_slots + 1) * stride,
            )?;
            ledger.reserve(
                Pool::HostPinned,
                &format!("ple.cache.rank{r}"),
                self.ple_cache_per_rank,
            )?;
            if self.kv_ram_per_rank > 0 {
                ledger.reserve(
                    Pool::HostPinned,
                    &format!("kv.ram.rank{r}"),
                    self.kv_ram_per_rank,
                )?;
            }
            if self.embed_per_rank > 0 {
                ledger.reserve(
                    Pool::HostPinned,
                    &format!("fn.embed.rank{r}"),
                    self.embed_per_rank,
                )?;
            }
        }
        Ok(())
    }

    /// The host boot line.
    #[must_use]
    pub fn boot_line(&self, l2_slots: u64, complement: u64, stride: u64) -> String {
        let g = |b: u64| b as f64 / GIB as f64;
        let l2 = (l2_slots + 1) * stride;
        format!(
            "fnhost: MemTotal {:.2} GiB - box {:.0} - free floor {:.0} = {:.2} GiB pinnable: serve {:.2} + {} x (L2 {} slots + 1 spare = {:.3} GiB + PLE cache {:.2} GiB{}{}) = {:.2} GiB | L2 {} the {} non-resident records per rank: {}",
            g(self.mem_total),
            g(self.box_used),
            g(self.free_floor),
            g(self.pinnable()),
            g(self.serve_pinned),
            self.ranks,
            l2_slots,
            g(l2),
            g(self.ple_cache_per_rank),
            if self.kv_ram_per_rank > 0 {
                format!(" + KV RAM {:.2} GiB", g(self.kv_ram_per_rank))
            } else {
                String::new()
            },
            if self.embed_per_rank > 0 {
                format!(" + embed {:.3} GiB", g(self.embed_per_rank))
            } else {
                String::new()
            },
            g(self.serve_pinned
                + self.ranks
                    * (l2 + self.ple_cache_per_rank + self.kv_ram_per_rank + self.embed_per_rank)),
            if l2_slots >= complement { "covers" } else { "does NOT cover" },
            complement,
            if l2_slots >= complement {
                "complete (NVMe off the expert step path)"
            } else {
                "demand (misses read from NVMe on their layer)"
            }
        )
    }
}
