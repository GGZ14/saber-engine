//! The two allocators admission has to satisfy, and the ledger reservation that sizes them.
//!
//! They are **separate on purpose**. `ARCHITECTURE.md` § "What exploration corrected" item 3:
//! The reference server couples the mamba page to the attention block size, the fp32 GDN page forces a 1648-token
//! attention block, and TP=1 concurrency caps at 3 (09-16). Saber never inherits that coupling, so
//! a KV page size and a state slot size are two unrelated numbers here and a sequence needs one
//! allocation from each.
//!
//! Both are **reserved up front, for the whole sequence**: KV blocks for `prompt + max_tokens`, and
//! the full `max_spec_width + 1` state slot set. That is what makes "admission refuses" the only
//! failure mode; there is no path on which a running sequence discovers at token 4000 that the
//! arena is full.
//!
//! Free lists hand back the **lowest** free index. That is not cosmetic either: deterministic reuse
//! is what gives the multi-turn gate its teeth, because turn 2 lands on exactly the blocks and slots
//! turn 1 used, and any state that survived the release shows up as a different token stream.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use saber_mem::ledger::{Ledger, LedgerError, Pool};

use crate::config::{KvGeometry, SchedulerConfig};

/// Ledger reservation name for the paged KV arena.
pub const KV_ARENA_RESERVATION: &str = "kv_arena";
/// Ledger reservation name for the GDN state arena.
pub const GDN_ARENA_RESERVATION: &str = "gdn_state_arena";

/// One sequence's KV pages.
///
/// With `n > 1` the branches of a sequence group **share the prompt's whole blocks**: the leader
/// allocates them and every other branch takes a reference, so a group of three costs one prompt
/// and three tails rather than three prompts. A block goes back to the free list when the last
/// holder releases it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvAllocation {
    /// The block table, in sequence order: the shared prefix first, then this branch's own tail.
    pub blocks: Vec<u32>,
    /// How many leading blocks are shared with the group's other branches. `0` for a request with
    /// `n == 1`, which owns everything it holds.
    pub shared_prefix: u32,
    /// The **copy-on-write boundary**: the leader's partially-filled block whose contents this
    /// branch must copy into its first private block before it writes there.
    ///
    /// Only whole blocks can be shared — a block the branches will both write into cannot be —
    /// so the prompt's last, partial block is copied rather than shared. `None` when the prompt
    /// ends exactly on a block boundary, or for a branch that shares nothing.
    pub cow_source: Option<u32>,
    /// Tokens the reservation covers: `prompt + max_tokens`, rounded up to whole blocks.
    pub capacity_tokens: u64,
}

impl KvAllocation {
    /// Bytes this allocation holds on one rank.
    pub fn bytes(&self, kv: &KvGeometry) -> u64 {
        self.blocks.len() as u64 * kv.block_bytes()
    }
}

/// One sequence's GDN state slot set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnAllocation {
    /// Index of the slot set.
    pub set: u32,
    /// Slots in it: `max_spec_width + 1`.
    pub slots: u32,
}

/// A paged KV allocator over a fixed arena, with **refcounted blocks**.
///
/// A block is free when nothing references it. Sharing is what makes a sequence group cost one
/// prompt instead of `n`, and refcounting is what makes it safe: a branch that finishes, aborts or
/// parks decrements, and the block returns to the arena on the last release — never before, or a
/// live branch would be reading pages the next admission had been given.
#[derive(Debug, Clone)]
pub struct PagedKvAllocator {
    geometry: KvGeometry,
    total_blocks: u32,
    free: BinaryHeap<Reverse<u32>>,
    /// One refcount per block id. `0` means the block is in `free`.
    refs: Vec<u32>,
    live_blocks: u32,
}

impl PagedKvAllocator {
    /// An allocator over `arena_bytes` on one rank, at `geometry`.
    pub fn new(geometry: KvGeometry, arena_bytes: u64) -> Self {
        let total_blocks = u32::try_from(arena_bytes / geometry.block_bytes()).unwrap_or(u32::MAX);
        Self {
            geometry,
            total_blocks,
            free: (0..total_blocks).map(Reverse).collect(),
            refs: vec![0; total_blocks as usize],
            live_blocks: 0,
        }
    }

    /// Blocks in the arena.
    pub fn total_blocks(&self) -> u32 {
        self.total_blocks
    }

    /// Blocks not currently held by a sequence.
    pub fn free_blocks(&self) -> u32 {
        self.total_blocks - self.live_blocks
    }

    /// Blocks a sequence of `tokens` tokens needs.
    pub fn blocks_for(&self, tokens: u64) -> u32 {
        u32::try_from(self.geometry.blocks_for(tokens)).unwrap_or(u32::MAX)
    }

    /// How many blocks reference a block id. `0` is free, `>1` is shared by a sequence group.
    pub fn refs(&self, block: u32) -> u32 {
        self.refs.get(block as usize).copied().unwrap_or(0)
    }

    /// Whole blocks a prompt of `tokens` fills. Only whole blocks can be shared: the partial one
    /// is where the branches diverge, and it is copied rather than shared.
    pub fn whole_blocks(&self, tokens: u64) -> u32 {
        u32::try_from(tokens / u64::from(self.geometry.block_tokens)).unwrap_or(u32::MAX)
    }

    fn take(&mut self, n: u32) -> Vec<u32> {
        let mut blocks = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let Reverse(b) = self.free.pop().expect("caller checked free_blocks()");
            self.refs[b as usize] = 1;
            blocks.push(b);
        }
        self.live_blocks += n;
        blocks
    }

    /// Reserves the blocks for a whole sequence, or returns how many were free.
    pub fn allocate(&mut self, tokens: u64) -> Result<KvAllocation, u32> {
        let want = self.blocks_for(tokens);
        if want > self.free_blocks() {
            return Err(self.free_blocks());
        }
        let blocks = self.take(want);
        Ok(KvAllocation {
            blocks,
            shared_prefix: 0,
            cow_source: None,
            capacity_tokens: u64::from(want) * u64::from(self.geometry.block_tokens),
        })
    }

    /// LR-G: `n` more blocks at the END of `alloc` (lazy growth), or how many were free. The
    /// lowest free ids, like every other take; nothing else of the allocation moves.
    pub fn extend(&mut self, alloc: &mut KvAllocation, n: u32) -> Result<(), u32> {
        if n > self.free_blocks() {
            return Err(self.free_blocks());
        }
        let more = self.take(n);
        alloc.blocks.extend(more);
        alloc.capacity_tokens += u64::from(n) * u64::from(self.geometry.block_tokens);
        Ok(())
    }

    /// A branch of an existing sequence: the first `shared` blocks are **referenced**, not copied,
    /// and the rest are this branch's own.
    ///
    /// The block after the shared prefix is the copy-on-write boundary and is recorded in
    /// [`KvAllocation::cow_source`]: the branch gets a private block there, and whoever owns the
    /// device copies the parent's partial page into it before writing.
    pub fn branch(
        &mut self,
        parent: &KvAllocation,
        shared: u32,
        tokens: u64,
    ) -> Result<KvAllocation, u32> {
        let want = self.blocks_for(tokens);
        let shared = shared.min(want).min(parent.blocks.len() as u32);
        let own = want - shared;
        if own > self.free_blocks() {
            return Err(self.free_blocks());
        }
        let mut blocks: Vec<u32> = parent.blocks[..shared as usize].to_vec();
        for b in &blocks {
            self.refs[*b as usize] += 1;
        }
        let cow_source = parent.blocks.get(shared as usize).copied();
        blocks.extend(self.take(own));
        Ok(KvAllocation {
            blocks,
            shared_prefix: shared,
            cow_source,
            capacity_tokens: u64::from(want) * u64::from(self.geometry.block_tokens),
        })
    }

    /// FN52: one more reference on each of `blocks` (a retained prefix entry). Every block must
    /// already be held — an entry retains pages a live sequence wrote, never free ones.
    pub fn retain(&mut self, blocks: &[u32]) {
        for b in blocks {
            let r = &mut self.refs[*b as usize];
            debug_assert!(*r > 0, "retaining block {b} that nothing holds");
            *r += 1;
        }
    }

    /// FN52: drop one reference on each of `blocks`; a block whose last holder this was returns
    /// to the arena.
    pub fn release(&mut self, blocks: &[u32]) {
        for b in blocks {
            let r = &mut self.refs[*b as usize];
            debug_assert!(*r > 0, "releasing block {b} that nothing holds");
            *r -= 1;
            if *r == 0 {
                self.live_blocks -= 1;
                self.free.push(Reverse(*b));
            }
        }
    }

    /// FN52: a sequence that ADOPTS a retained prefix: `prefix` is referenced (no copy) and the
    /// rest of a `tokens`-token reservation is this sequence's own.
    pub fn allocate_prefixed(&mut self, prefix: &[u32], tokens: u64) -> Result<KvAllocation, u32> {
        let want = self.blocks_for(tokens);
        let shared = (prefix.len() as u32).min(want);
        let own = want - shared;
        if own > self.free_blocks() {
            return Err(self.free_blocks());
        }
        let mut blocks: Vec<u32> = prefix[..shared as usize].to_vec();
        self.retain(&blocks);
        blocks.extend(self.take(own));
        Ok(KvAllocation {
            blocks,
            shared_prefix: shared,
            cow_source: None,
            capacity_tokens: u64::from(want) * u64::from(self.geometry.block_tokens),
        })
    }

    /// Hands a sequence's blocks back. A shared block returns to the arena only when its last
    /// holder releases it.
    pub fn free(&mut self, alloc: KvAllocation) {
        for b in alloc.blocks {
            let r = &mut self.refs[b as usize];
            debug_assert!(*r > 0, "freeing block {b} that nothing holds");
            *r -= 1;
            if *r == 0 {
                self.live_blocks -= 1;
                self.free.push(Reverse(b));
            }
        }
    }
}

/// A GDN state allocator: whole slot sets, one per sequence.
#[derive(Debug, Clone)]
pub struct GdnStateAllocator {
    slots_per_set: u32,
    bytes_per_slot: u64,
    total_sets: u32,
    free: BinaryHeap<Reverse<u32>>,
    live_sets: u32,
}

impl GdnStateAllocator {
    /// An allocator over `arena_bytes` on one rank, with `slots_per_set` slots of `bytes_per_slot`
    /// each.
    pub fn new(slots_per_set: u32, bytes_per_slot: u64, arena_bytes: u64) -> Self {
        let set_bytes = u64::from(slots_per_set) * bytes_per_slot;
        let total_sets = arena_bytes
            .checked_div(set_bytes)
            .map_or(0, |sets| u32::try_from(sets).unwrap_or(u32::MAX));
        Self {
            slots_per_set,
            bytes_per_slot,
            total_sets,
            free: (0..total_sets).map(Reverse).collect(),
            live_sets: 0,
        }
    }

    /// Slot sets in the arena — the hard concurrency cap below the KV crossover.
    pub fn total_sets(&self) -> u32 {
        self.total_sets
    }

    /// Slot sets not currently held.
    pub fn free_sets(&self) -> u32 {
        self.total_sets - self.live_sets
    }

    /// Slots per set.
    pub fn slots_per_set(&self) -> u32 {
        self.slots_per_set
    }

    /// Bytes one sequence's slot set holds on one rank.
    pub fn bytes_per_set(&self) -> u64 {
        u64::from(self.slots_per_set) * self.bytes_per_slot
    }

    /// Takes one slot set, or reports that there are none.
    pub fn allocate(&mut self) -> Result<GdnAllocation, u32> {
        let Some(Reverse(set)) = self.free.pop() else {
            return Err(0);
        };
        self.live_sets += 1;
        Ok(GdnAllocation {
            set,
            slots: self.slots_per_set,
        })
    }

    /// Hands a slot set back.
    pub fn free(&mut self, alloc: GdnAllocation) {
        self.live_sets -= 1;
        self.free.push(Reverse(alloc.set));
    }
}

/// Why admission refused a request this step. Every variant carries the arithmetic, because the
/// question a refusal has to answer is *which* allocator ran out — the whole point of keeping the
/// two separate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The running set is already at the widest captured bucket.
    BatchFull {
        /// Sequences running.
        running: u32,
        /// The widest captured batch.
        max_batch: u32,
    },
    /// Not enough KV pages for `prompt + max_tokens`.
    KvBlocks {
        /// Blocks needed.
        want: u32,
        /// Blocks free.
        free: u32,
    },
    /// No GDN state slot set left. Below the ~4.6k crossover this is what runs out first, and
    /// for a sequence group it is what caps `n`: the recurrent state is not addressable per
    /// token, so **every branch needs its own slot set** however much KV the group shares.
    GdnStateSlots {
        /// Slots one sequence needs.
        want_slots: u32,
        /// Slot sets the group needs: one per branch, i.e. `n`.
        want_sets: u32,
        /// Sets free.
        free_sets: u32,
    },
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::BatchFull { running, max_batch } => write!(
                f,
                "batch full: {running} running, widest captured bucket batch is {max_batch}"
            ),
            Refusal::KvBlocks { want, free } => {
                write!(f, "kv arena: wants {want} blocks, {free} free")
            }
            Refusal::GdnStateSlots {
                want_slots,
                want_sets,
                free_sets,
            } => write!(
                f,
                "gdn state arena: wants {want_sets} set(s) of {want_slots} slots (one per branch, \
                 n = {want_sets}), {free_sets} free -- the state arena is what caps n, because \
                 recurrent state cannot be shared the way KV pages can"
            ),
        }
    }
}

/// Which-path counters for admission (`CLAUDE.md` rule 5: a route with a fallback branch declares a
/// counter at construction).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdmissionCounters {
    /// Sequences admitted.
    pub admitted: u64,
    /// Refusals because the widest captured batch was full.
    pub refused_batch: u64,
    /// Refusals for want of KV pages.
    pub refused_kv: u64,
    /// Refusals for want of a GDN state slot set.
    pub refused_gdn: u64,
    /// KV allocations rolled back because the state allocation then failed.
    pub kv_rollbacks: u64,
}

// The admission half of the shutdown census, live on `/metrics` (ws-v8).
saber_serve::census_counters!(
    AdmissionCounters,
    CENSUS,
    census_values,
    [
        (
            admitted,
            "saber:admission_admitted_total",
            "Sequences admitted.",
            "admission: admitted {},"
        ),
        (
            refused_kv,
            "saber:admission_refused_kv_total",
            "Admissions refused for want of KV pages.",
            "admission: refused kv {} /"
        ),
        (
            refused_gdn,
            "saber:admission_refused_gdn_total",
            "Admissions refused for want of a GDN state slot set.",
            "admission: gdn {} /"
        ),
        (
            refused_batch,
            "saber:admission_refused_batch_total",
            "Admissions refused because the widest batch was full.",
            "admission: batch {},"
        ),
        (
            kv_rollbacks,
            "saber:admission_kv_rollbacks_total",
            "KV allocations rolled back after a state refusal.",
            "admission: kv rollbacks {}"
        ),
    ]
);

impl AdmissionCounters {
    /// Records a refusal.
    pub fn refuse(&mut self, refusal: Refusal) {
        match refusal {
            Refusal::BatchFull { .. } => self.refused_batch += 1,
            Refusal::KvBlocks { .. } => self.refused_kv += 1,
            Refusal::GdnStateSlots { .. } => self.refused_gdn += 1,
        }
    }
}

/// The two arenas, sized from the config and reserved through the byte ledger on every rank.
#[derive(Debug)]
pub struct Arenas {
    kv: PagedKvAllocator,
    gdn: GdnStateAllocator,
    counters: AdmissionCounters,
}

impl Arenas {
    /// Reserves both arenas on every rank, then builds the allocators.
    ///
    /// The reservation is the startup error the ledger exists for: if the KV and state arenas do
    /// not fit beside the weights, that is known here, before a request is accepted.
    pub fn reserve(config: &SchedulerConfig, ledger: &mut Ledger) -> Result<Self, LedgerError> {
        for rank in 0..config.tp {
            ledger.reserve(
                Pool::Vram(rank),
                KV_ARENA_RESERVATION,
                config.arenas.kv_per_rank,
            )?;
            ledger.reserve(
                Pool::Vram(rank),
                GDN_ARENA_RESERVATION,
                config.arenas.gdn_state_per_rank,
            )?;
        }
        Ok(Self {
            kv: PagedKvAllocator::new(config.kv, config.arenas.kv_per_rank),
            gdn: GdnStateAllocator::new(
                config.gdn_slots_per_seq(),
                config.gdn_bytes_per_slot,
                config.arenas.gdn_state_per_rank,
            ),
            counters: AdmissionCounters::default(),
        })
    }

    /// The KV allocator.
    pub fn kv(&self) -> &PagedKvAllocator {
        &self.kv
    }

    /// The state allocator.
    pub fn gdn(&self) -> &GdnStateAllocator {
        &self.gdn
    }

    /// Admission counters.
    pub fn counters(&self) -> AdmissionCounters {
        self.counters
    }

    /// Takes a whole **sequence group**: one KV allocation per branch — the first owns the
    /// prompt's whole blocks and the rest reference them — and one GDN state slot set each.
    ///
    /// All or nothing. A group that got half its branches would hold arena that can never produce
    /// an answer, and at the batch cap it would deadlock against itself; so a refusal leaves both
    /// arenas exactly as they were, and says which one ran out.
    ///
    /// `shared_tokens` is the prompt: its **whole** blocks are shared and its partial one is the
    /// copy-on-write boundary. The state arena is not shareable at all — recurrent state is not
    /// addressable per token — so `n` slot sets are taken whatever the prompt costs, which is why
    /// a group's refusal is usually [`Refusal::GdnStateSlots`].
    pub fn allocate_group(
        &mut self,
        n: u32,
        tokens: u64,
        shared_tokens: u64,
    ) -> Result<(Vec<KvAllocation>, Vec<GdnAllocation>), Refusal> {
        let n = n.max(1);
        let mut kvs: Vec<KvAllocation> = Vec::with_capacity(n as usize);
        let mut gdns: Vec<GdnAllocation> = Vec::with_capacity(n as usize);

        let rollback = |kvs: Vec<KvAllocation>, gdns: Vec<GdnAllocation>, me: &mut Self| {
            for kv in kvs {
                me.kv.free(kv);
            }
            for gdn in gdns {
                me.gdn.free(gdn);
            }
        };

        let shared = self.kv.whole_blocks(shared_tokens);
        for branch in 0..n {
            let taken = if branch == 0 {
                self.kv.allocate(tokens)
            } else {
                self.kv.branch(&kvs[0], shared, tokens)
            };
            match taken {
                Ok(kv) => kvs.push(kv),
                Err(free) => {
                    let want = self.kv.blocks_for(tokens);
                    rollback(kvs, gdns, self);
                    self.counters.kv_rollbacks += u64::from(branch);
                    let refusal = Refusal::KvBlocks { want, free };
                    self.counters.refuse(refusal);
                    return Err(refusal);
                }
            }
        }
        for branch in 0..n {
            match self.gdn.allocate() {
                Ok(gdn) => gdns.push(gdn),
                Err(free_sets) => {
                    let want_slots = self.gdn.slots_per_set();
                    rollback(kvs, gdns, self);
                    self.counters.kv_rollbacks += u64::from(n);
                    let _ = branch;
                    let refusal = Refusal::GdnStateSlots {
                        want_slots,
                        want_sets: n,
                        free_sets,
                    };
                    self.counters.refuse(refusal);
                    return Err(refusal);
                }
            }
        }
        self.counters.admitted += u64::from(n);
        Ok((kvs, gdns))
    }

    /// Takes one KV allocation **and** one state slot set, or refuses and leaves both arenas
    /// exactly as they were. A KV allocation that survives a failed state allocation would be a
    /// leak that only shows up as a capacity loss hours later, so it is rolled back and counted.
    pub fn allocate(&mut self, tokens: u64) -> Result<(KvAllocation, GdnAllocation), Refusal> {
        let kv = match self.kv.allocate(tokens) {
            Ok(kv) => kv,
            Err(free) => {
                let refusal = Refusal::KvBlocks {
                    want: self.kv.blocks_for(tokens),
                    free,
                };
                self.counters.refuse(refusal);
                return Err(refusal);
            }
        };
        match self.gdn.allocate() {
            Ok(gdn) => {
                self.counters.admitted += 1;
                Ok((kv, gdn))
            }
            Err(free_sets) => {
                self.kv.free(kv);
                self.counters.kv_rollbacks += 1;
                let refusal = Refusal::GdnStateSlots {
                    want_slots: self.gdn.slots_per_set(),
                    want_sets: 1,
                    free_sets,
                };
                self.counters.refuse(refusal);
                Err(refusal)
            }
        }
    }

    /// FN52: one KV allocation that adopts `prefix` **and** one state slot set, or a refusal that
    /// leaves both arenas as they were ([`PagedKvAllocator::allocate_prefixed`]).
    pub fn allocate_prefixed(
        &mut self,
        tokens: u64,
        prefix: &[u32],
    ) -> Result<(KvAllocation, GdnAllocation), Refusal> {
        let kv = match self.kv.allocate_prefixed(prefix, tokens) {
            Ok(kv) => kv,
            Err(free) => {
                let want = self
                    .kv
                    .blocks_for(tokens)
                    .saturating_sub(prefix.len() as u32);
                let refusal = Refusal::KvBlocks { want, free };
                self.counters.refuse(refusal);
                return Err(refusal);
            }
        };
        match self.gdn.allocate() {
            Ok(gdn) => {
                self.counters.admitted += 1;
                Ok((kv, gdn))
            }
            Err(free_sets) => {
                self.kv.free(kv);
                self.counters.kv_rollbacks += 1;
                let refusal = Refusal::GdnStateSlots {
                    want_slots: self.gdn.slots_per_set(),
                    want_sets: 1,
                    free_sets,
                };
                self.counters.refuse(refusal);
                Err(refusal)
            }
        }
    }

    /// LR-G: grow a running sequence's KV by `n` blocks, or refuse (counted as a KV refusal of
    /// its own kind: `lazy_grow_refused`, not an admission refusal) and leave the arena as it was.
    pub fn extend_kv(&mut self, alloc: &mut KvAllocation, n: u32) -> Result<(), u32> {
        self.kv.extend(alloc, n)
    }

    /// FN52: a prefix entry's reference on its blocks.
    pub fn retain_kv(&mut self, blocks: &[u32]) {
        self.kv.retain(blocks);
    }

    /// FN52: a prefix entry dropped.
    pub fn release_kv(&mut self, blocks: &[u32]) {
        self.kv.release(blocks);
    }

    /// Frees both allocations. Called on finish, on abort and on park — the three ways a sequence
    /// leaves the running set.
    pub fn free(&mut self, kv: KvAllocation, gdn: GdnAllocation) {
        self.kv.free(kv);
        self.gdn.free(gdn);
    }

    /// Records a refusal the arenas did not produce (the batch cap).
    pub fn refuse(&mut self, refusal: Refusal) {
        self.counters.refuse(refusal);
    }
}
