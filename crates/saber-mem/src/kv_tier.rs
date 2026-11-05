//! FN45a: the TIERED KV cache for Flash-Next's QSA layers -- VRAM hot <- pinned RAM warm <- NVMe
//! cold (docs/LEDGER.md FN45a; no engine wiring yet -- that is FN45b).
//!
//! **Why it fits the model.** Above 2,048 tokens a QSA layer attends each query to an exact top-512
//! of 4-token groups chosen over the COMPRESSED index cache (`kv.index`, one 128-d row per group,
//! VRAM-resident, 1/16 of the K/V bytes) plus the row's open group tail. A decode step therefore
//! reads the full K/V of at most 2,048 + 3 tokens per layer per row WHATEVER the context, and
//! FN45a's locality study (`tools/capture/fn45/kv_locality.py`) shows those groups are mostly the
//! ones the previous steps read. So the full K/V becomes a block tier:
//!
//! * **Blocks** of [`KvTierPlan::block_tokens`] tokens (a multiple of the 16-token page, so of the
//!   4-token group; [`DEFAULT_BLOCK_TOKENS`] = 16 = the pools' page: FN45a measured the hot set at
//!   95 % hits 33 % smaller than 64-token blocks at 16k, and the gap widens with the context);
//!   1,024 B per token per layer per rank (K + V, one KV head per rank, bf16, head 256).
//! * **Write-through to the lowest tier.** A block gets its VRAM slot and a RAM staging slot when
//!   an append opens it; the commit that completes it issues the D2H "flush" into the RAM slot
//!   and, with an SSD tier, the RAM -> NVMe write into the block's `kv.ssd` slot (13.3 KB/token/
//!   rank, ~93 MB/s a rank at 7,000 tok/s). Without an SSD tier (`ssd_slots == 0`) RAM is the
//!   lowest tier and holds every block.
//! * **VRAM hot** (`kv.hot`): the newest-W-token WINDOW of each sequence and the open append's
//!   blocks (never evicted: the compute writes them), and whatever else was selected while slots
//!   last -- a (layer, sequence) holding [`KvTierPlan::quota`] out-of-window hot blocks recycles
//!   its OWN least recently used block for a miss; below its quota it takes a free slot, then the
//!   global LRU victim that its own layer's latest gather did not select (no cross-layer thrash).
//! * **RAM warm** (`kv.ram`): the staging slots of the open blocks plus an LRU of everything else
//!   that is already on the SSD (a RAM copy is dropped without a copy: the SSD has the bytes).
//! * **[`KvTier::gather`]** turns one layer's step selection into the exact lists for the mover:
//!   SSD reads (block -> a RAM slot; the latency risk, counted apart) first, then one H2D per
//!   selected block absent from VRAM, then the layer's `Landed` mark. A block that finds no VRAM
//!   slot stays [`Place::Ram`] and is a counted UVA read -- exact, slower, never a failure.
//!
//! **The copy-order law** (FN8's mover: one in-order copy stream per rank; FN7's reader for the
//! drive):
//!
//! * K1. A commit's flushes (D2H) are issued after the step's compute wrote the rows (the mover's
//!   `after` fence); its SSD writes are issued after those flushes landed (the host drains).
//! * K2. An eviction copies nothing (every block outside the window and the open append is
//!   flushed, and on the SSD when there is one); its VRAM slot is reused by later ops on the same
//!   stream or by a later step's compute, which waits for every issued copy first.
//! * K3. A layer's attention reads its placements only after the gate on its `Landed` mark; the
//!   gather's H2Ds are issued only after its SSD reads landed in RAM.
//! * K4. The drive's queue keeps issue order per slot: a RAM slot's SSD write is read out before
//!   a later SSD read lands in the same (recycled) slot.
//!
//! The scheduler admits a sequence only when its whole `max_tokens` fits the lowest tier's
//! reservation ([`KvTier::admit`]); a plan whose `max_seqs x max_ctx` does not fit it refuses to
//! boot ([`KvTierPlan::new`]).
//!
//! **The prompt / conversation cache** (the SSD tier's second use; FN45b wires it): a finished
//! sequence's blocks are already on the drive; persisting them with its GDN state, conv tails, PLE
//! history and compressed index rows under [`prefix_keys`] (a chained sha256 at every chunk
//! boundary of its token ids) lets a returning conversation reload the longest cached chunk-aligned
//! prefix instead of re-prefilling it. The exactness law: the reloaded bytes are the bytes that
//! prefill wrote (each record carries its sha256), and the reload stops at a chunk boundary, so
//! FN13's chunk law (the cut decides the bytes) is unaffected. Ledger line `kv.pcache`
//! (`Pool::Nvme`), a budget apart from `kv.ssd`.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use sha2::{Digest, Sha256};

use crate::device_mover::CopyOp;
use crate::ledger::{Ledger, LedgerError, Pool};

/// Tokens per QSA group (the indexer's pooling unit).
pub const GROUP_TOKENS: u32 = 4;
/// Tokens per page of today's sequence-major pools (`plan.rs`: `nb = max_ctx / 16 + 1`).
pub const PAGE_TOKENS: u32 = 16;
/// The selection's width in groups (exact top-512).
pub const TOPK_GROUPS: u32 = 512;
/// The widest verify step's rows (each selects its own 512 groups).
pub const MAX_STEP_ROWS: u32 = 5;
/// The default block = the page (FN45a's locality study; LEDGER FN45a).
pub const DEFAULT_BLOCK_TOKENS: u32 = 16;
/// QSA attention layers with a KV cache: 12 target + the MTP layer.
pub const QSA_LAYERS: u32 = 13;
/// K + V bytes per token per attention layer per rank (one KV head, bf16, head 256).
pub const KV_TOKEN_LAYER_BYTES: u64 = 1024;

/// FN68: the main K/V pages' element type (`--fn-kv-dtype`). `Bf16` is upstream QSA's only mode
/// (the default); `Fp8` stores e4m3 bytes at one per-tensor scale -- half the bytes a token, so
/// every tier's page (and the slab, the hot pool, the drive copy) halves at the same tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvDtype {
    #[default]
    /// bf16 pages (upstream QSA's only mode; the default, the control).
    Bf16,
    /// e4m3 pages at one per-tensor scale (FN68, the arm).
    Fp8,
}

impl KvDtype {
    /// Bytes of one K/V element.
    #[must_use]
    pub fn elem_bytes(self) -> u64 {
        match self {
            KvDtype::Bf16 => 2,
            KvDtype::Fp8 => 1,
        }
    }

    /// K + V bytes per token per attention layer per rank (head 256, one KV head).
    #[must_use]
    pub fn token_layer_bytes(self) -> u64 {
        KV_TOKEN_LAYER_BYTES / 2 * self.elem_bytes()
    }

    /// The flag's spelling.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            KvDtype::Bf16 => "bf16",
            KvDtype::Fp8 => "fp8",
        }
    }
}
/// LR-G: blocks one step may open past a lazy reservation (the widest verify's rows + the MTP
/// chain's pages ahead + a page of rounding, in 16-token blocks, rounded up): the floor's
/// allowance beside a lazy sequence.
pub const LAZY_STEP_BLOCKS: u32 = 4;
/// The compressed index cache per token (13 layers x 128 bf16 / 4 tokens), replicated.
pub const INDEX_TOKEN_BYTES: u64 = 832;
/// Ledger line: the VRAM hot pool.
pub const KV_HOT: &str = "kv.hot";
/// Ledger line: the pinned RAM warm pool (every block, without an SSD tier).
pub const KV_RAM: &str = "kv.ram";
/// Ledger line: the NVMe copy of every block.
pub const KV_SSD: &str = "kv.ssd";
/// Ledger line: the compressed index cache (VRAM).
pub const KV_INDEX: &str = "kv.index";
/// Ledger line: the persisted prompt / conversation cache (NVMe, FN45b).
pub const KV_PCACHE: &str = "kv.pcache";

const GIB: u64 = 1 << 30;

/// Why the tier refused.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum KvTierError {
    /// A geometry the tier cannot serve.
    #[error("kv tier geometry: {0}")]
    Geometry(String),
    /// `max_seqs x max_ctx` does not fit the lowest tier (`kv.ram` or `kv.ssd`), or a pool is
    /// below its floor.
    #[error(
        "{line}: {max_seqs} sequences x {max_ctx} tokens need {need} B, the budget is {have} B (short {short} B)"
    )]
    Short {
        /// The ledger line that is short.
        line: &'static str,
        /// Sequences.
        max_seqs: u32,
        /// Context per sequence.
        max_ctx: u32,
        /// Bytes needed.
        need: u64,
        /// Bytes budgeted.
        have: u64,
        /// Shortfall.
        short: u64,
    },
    /// The scheduler asked for something the tier's state forbids.
    #[error("kv tier: {0}")]
    State(String),
    /// Admission refused: the lowest tier's reservation is full.
    #[error("kv tier: admitting {want} blocks leaves {free} unreserved slots")]
    Admission {
        /// Blocks the sequence needs.
        want: u64,
        /// Unreserved slots.
        free: u64,
    },
    /// No slot for a block that MUST be resident (the window, the open append, an SSD read's
    /// RAM landing).
    #[error("kv tier: no {pool} slot for a block that must be resident (pool of {slots})")]
    NoRoom {
        /// `vram` or `ram`.
        pool: &'static str,
        /// Pool size.
        slots: u32,
    },
    /// The ledger refused.
    #[error(transparent)]
    Ledger(#[from] LedgerError),
}

/// The tier's sizes (one rank).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvTierPlan {
    /// Attention layers with a KV cache.
    pub layers: u32,
    /// Tokens per block.
    pub block_tokens: u32,
    /// The newest tokens of each sequence that stay hot.
    pub window_tokens: u32,
    /// The widest append (a prefill chunk) the pools must hold beside every window.
    pub chunk_tokens: u32,
    /// Sequence slots.
    pub max_seqs: u32,
    /// Context per sequence.
    pub max_ctx: u32,
    /// VRAM block slots.
    pub hot_slots: u32,
    /// Pinned RAM block slots.
    pub ram_slots: u32,
    /// NVMe block slots (0 = two tiers: RAM holds every block).
    pub ssd_slots: u32,
    /// Out-of-window hot blocks per (layer, sequence) beyond which a miss recycles the sequence's
    /// own LRU block (the even share of what the floor leaves).
    pub quota: u32,
    /// FN68: K + V bytes per token per layer (the page dtype's; bf16 = [`KV_TOKEN_LAYER_BYTES`]).
    pub token_layer_bytes: u64,
}

impl KvTierPlan {
    /// Plan a rank from the budgets (`--fn-kv-hot-gib`, `--fn-kv-ram-gib`, `--fn-kv-ssd-gib`, in
    /// bytes; `ssd_bytes` 0 = no SSD tier). Refuses a boot whose `max_seqs x max_ctx` does not
    /// fit the lowest tier, whose hot pool cannot hold every window plus one open append, or
    /// whose RAM cannot stage those plus one widest gather.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        layers: u32,
        block_tokens: u32,
        window_tokens: u32,
        chunk_tokens: u32,
        max_seqs: u32,
        max_ctx: u32,
        hot_bytes: u64,
        ram_bytes: u64,
        ssd_bytes: u64,
    ) -> Result<KvTierPlan, KvTierError> {
        Self::build(
            layers,
            block_tokens,
            window_tokens,
            chunk_tokens,
            max_seqs,
            max_ctx,
            hot_bytes,
            ram_bytes,
            ssd_bytes,
            false,
            KV_TOKEN_LAYER_BYTES,
        )
    }

    /// FN68: [`KvTierPlan::new`] at the page dtype `dt` (every byte budget is divided by its block
    /// bytes; `Bf16` = `new`).
    #[allow(clippy::too_many_arguments)]
    pub fn new_dt(
        layers: u32,
        block_tokens: u32,
        window_tokens: u32,
        chunk_tokens: u32,
        max_seqs: u32,
        max_ctx: u32,
        hot_bytes: u64,
        ram_bytes: u64,
        ssd_bytes: u64,
        dt: KvDtype,
    ) -> Result<KvTierPlan, KvTierError> {
        Self::build(
            layers,
            block_tokens,
            window_tokens,
            chunk_tokens,
            max_seqs,
            max_ctx,
            hot_bytes,
            ram_bytes,
            ssd_bytes,
            false,
            dt.token_layer_bytes(),
        )
    }

    /// FN68: [`KvTierPlan::new_admitted`] at the page dtype `dt`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_admitted_dt(
        layers: u32,
        block_tokens: u32,
        window_tokens: u32,
        chunk_tokens: u32,
        max_seqs: u32,
        max_ctx: u32,
        hot_bytes: u64,
        ram_bytes: u64,
        dt: KvDtype,
    ) -> Result<KvTierPlan, KvTierError> {
        Self::build(
            layers,
            block_tokens,
            window_tokens,
            chunk_tokens,
            max_seqs,
            max_ctx,
            hot_bytes,
            ram_bytes,
            0,
            true,
            dt.token_layer_bytes(),
        )
    }

    /// FN51: a TWO-tier plan under ADMISSION BY RESERVATION: the RAM slab need not hold
    /// `max_seqs x max_ctx` -- [`KvTier::admit`] reserves every sequence's blocks out of the RAM
    /// slots at its request length and refuses what does not fit (the scheduler queues it) -- but
    /// it must hold ONE sequence at the max context (else a request the context allows could never
    /// run: refused here, the ceiling named).
    #[allow(clippy::too_many_arguments)]
    pub fn new_admitted(
        layers: u32,
        block_tokens: u32,
        window_tokens: u32,
        chunk_tokens: u32,
        max_seqs: u32,
        max_ctx: u32,
        hot_bytes: u64,
        ram_bytes: u64,
    ) -> Result<KvTierPlan, KvTierError> {
        Self::build(
            layers,
            block_tokens,
            window_tokens,
            chunk_tokens,
            max_seqs,
            max_ctx,
            hot_bytes,
            ram_bytes,
            0,
            true,
            KV_TOKEN_LAYER_BYTES,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        layers: u32,
        block_tokens: u32,
        window_tokens: u32,
        chunk_tokens: u32,
        max_seqs: u32,
        max_ctx: u32,
        hot_bytes: u64,
        ram_bytes: u64,
        ssd_bytes: u64,
        admitted: bool,
        token_layer_bytes: u64,
    ) -> Result<KvTierPlan, KvTierError> {
        let g = |m: String| Err(KvTierError::Geometry(m));
        if block_tokens == 0 || block_tokens % PAGE_TOKENS != 0 {
            return g(format!(
                "block {block_tokens} is not a multiple of the {PAGE_TOKENS}-token page"
            ));
        }
        if window_tokens < block_tokens || window_tokens % block_tokens != 0 {
            return g(format!(
                "window {window_tokens} is not a positive multiple of the {block_tokens}-token block"
            ));
        }
        if layers == 0 || max_seqs == 0 || max_ctx == 0 || chunk_tokens == 0 {
            return g("zero layers / sequences / context / chunk".into());
        }
        if token_layer_bytes == 0 {
            return g("zero bytes a token".into());
        }
        let bb = u64::from(block_tokens) * token_layer_bytes;
        let blocks = u64::from(max_ctx.div_ceil(block_tokens));
        // FN51: under admission by reservation the slab holds one max-context sequence at least
        let need = u64::from(if admitted { 1 } else { max_seqs }) * u64::from(layers) * blocks * bb;
        let short = |line, need: u64, have: u64| KvTierError::Short {
            line,
            max_seqs,
            max_ctx,
            need,
            have,
            short: need - have,
        };
        let (wb, cb) = (
            u64::from(window_tokens / block_tokens) + 1,
            u64::from(chunk_tokens.div_ceil(block_tokens)) + 1,
        );
        let floor = u64::from(layers) * (u64::from(max_seqs) * wb + cb);
        if ssd_bytes == 0 {
            if need > ram_bytes {
                return Err(short(KV_RAM, need, ram_bytes));
            }
        } else {
            if need > ssd_bytes {
                return Err(short(KV_SSD, need, ssd_bytes));
            }
            let gather = u64::from(MAX_STEP_ROWS * (TOPK_GROUPS + 1)).min(blocks);
            // FN45c3: under the RESIDENT law every active block owns a RAM slot inside its
            // sequence's reservation (admission by request length), so the hot windows need no
            // staging of their own: the floor is one widest admission's chunk + one gather.
            let ram_floor = (u64::from(layers) * cb + gather) * bb;
            if ram_bytes < ram_floor {
                return Err(short(KV_RAM, ram_floor, ram_bytes));
            }
        }
        if hot_bytes / bb < floor {
            return Err(short(KV_HOT, floor * bb, hot_bytes));
        }
        let slots = |b: u64, what: &str| {
            u32::try_from(b / bb)
                .map_err(|_| KvTierError::Geometry(format!("{what} slots overflow")))
        };
        let hot_slots = slots(hot_bytes, "hot")?;
        let quota = u32::try_from(
            (u64::from(hot_slots) - floor) / (u64::from(layers) * u64::from(max_seqs)),
        )
        .unwrap_or(u32::MAX);
        Ok(KvTierPlan {
            layers,
            block_tokens,
            window_tokens,
            chunk_tokens,
            max_seqs,
            max_ctx,
            hot_slots,
            ram_slots: slots(ram_bytes, "RAM")?,
            ssd_slots: slots(ssd_bytes, "SSD")?,
            quota,
            token_layer_bytes,
        })
    }

    /// Bytes of one block of one layer.
    #[must_use]
    pub fn block_bytes(&self) -> u64 {
        u64::from(self.block_tokens) * self.token_layer_bytes
    }

    /// Whether the NVMe tier is on.
    #[must_use]
    pub fn has_ssd(&self) -> bool {
        self.ssd_slots > 0
    }

    /// `kv.hot` bytes.
    #[must_use]
    pub fn hot_bytes(&self) -> u64 {
        u64::from(self.hot_slots) * self.block_bytes()
    }

    /// `kv.ram` bytes.
    #[must_use]
    pub fn ram_bytes(&self) -> u64 {
        u64::from(self.ram_slots) * self.block_bytes()
    }

    /// `kv.ssd` bytes.
    #[must_use]
    pub fn ssd_bytes(&self) -> u64 {
        u64::from(self.ssd_slots) * self.block_bytes()
    }

    /// `kv.index` bytes: every sequence's compressed index rows at `max_ctx` (VRAM-resident).
    #[must_use]
    pub fn index_bytes(&self) -> u64 {
        u64::from(self.max_seqs) * u64::from(self.max_ctx) * INDEX_TOKEN_BYTES
    }

    /// Blocks per layer of a `tokens`-token sequence.
    #[must_use]
    pub fn blocks(&self, tokens: u32) -> u32 {
        tokens.div_ceil(self.block_tokens)
    }

    /// The lowest tier's slots (what admission reserves).
    fn backing_slots(&self) -> u32 {
        if self.has_ssd() {
            self.ssd_slots
        } else {
            self.ram_slots
        }
    }

    /// Reserve the lines: `kv.hot` + `kv.index` on `Pool::Vram(rank)`, `kv.ram` on
    /// `Pool::HostPinned`, `kv.ssd` on `Pool::Nvme` when the tier is on (the pools must be
    /// declared).
    pub fn declare(&self, ledger: &mut Ledger, rank: u32) -> Result<(), KvTierError> {
        ledger.reserve(Pool::Vram(rank), KV_HOT, self.hot_bytes())?;
        ledger.reserve(Pool::Vram(rank), KV_INDEX, self.index_bytes())?;
        ledger.reserve(Pool::HostPinned, KV_RAM, self.ram_bytes())?;
        if self.has_ssd() {
            ledger.reserve(Pool::Nvme, KV_SSD, self.ssd_bytes())?;
        }
        Ok(())
    }

    /// The boot line.
    #[must_use]
    pub fn boot_line(&self) -> String {
        let gib = |b: u64| b as f64 / GIB as f64;
        format!(
            "kv tier: block {} tok, window {} tok, {} seqs x {} ctx; kv.hot {:.2} GiB ({} slots, quota {}/layer/seq), \
             kv.index {:.2} GiB, kv.ram {:.2} GiB ({} slots), kv.ssd {:.2} GiB ({} slots)",
            self.block_tokens,
            self.window_tokens,
            self.max_seqs,
            self.max_ctx,
            gib(self.hot_bytes()),
            self.hot_slots,
            self.quota,
            gib(self.index_bytes()),
            gib(self.ram_bytes()),
            self.ram_slots,
            gib(self.ssd_bytes()),
            self.ssd_slots
        )
    }
}

/// Where a block's bytes are read from now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Place {
    /// A VRAM slot of `kv.hot`.
    Vram(u32),
    /// A pinned RAM slot of `kv.ram` (a UVA read).
    Ram(u32),
    /// Its NVMe slot of `kv.ssd` only.
    Ssd(u32),
}

/// One layer's gather.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Gather {
    /// SSD reads `(ssd slot, RAM slot)`, issued and landed BEFORE `ops` (K3).
    pub ssd_reads: Vec<(u32, u32)>,
    /// The copies for the mover, in issue order (H2Ds, then the layer's `Landed` when any).
    pub ops: Vec<CopyOp>,
    /// Selected blocks already hot.
    pub hits: u32,
    /// Selected blocks copied in (one H2D each).
    pub copied: u32,
    /// Selected blocks left in RAM (read through UVA).
    pub uva: Vec<u32>,
}

/// One commit's write-through.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Commit {
    /// D2H flushes (VRAM -> the RAM staging slot), in issue order.
    pub flush: Vec<CopyOp>,
    /// Then the SSD writes `(RAM slot, ssd slot)`, after the flushes landed (K1).
    pub ssd_writes: Vec<(u32, u32)>,
}

/// The tier's which-path counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvCounts {
    /// Selected blocks found hot (window included).
    pub hits: u64,
    /// H2D copies (gathers).
    pub h2d: u64,
    /// D2H flushes (write-through).
    pub d2h: u64,
    /// Selected blocks read from RAM (UVA fallback).
    pub uva: u64,
    /// Selected blocks read from the SSD into RAM.
    pub ssd_reads: u64,
    /// Blocks written to the SSD.
    pub ssd_writes: u64,
    /// Hot blocks evicted.
    pub evictions: u64,
    /// RAM copies dropped (warm evictions).
    pub ram_drops: u64,
    /// Gathers.
    pub gathers: u64,
}

#[derive(Debug, Clone)]
struct Blk {
    ram: Option<u32>,
    vram: Option<u32>,
    ssd: Option<u32>,
    /// The SSD copy is written (issued).
    on_ssd: bool,
    last: u64,
    /// In the open append (or not yet complete): never a victim of either pool.
    pinned: bool,
}

#[derive(Debug, Clone)]
struct Seq {
    len: u32,
    max_tokens: u32,
    /// Blocks of each layer.
    layers: Vec<Vec<Blk>>,
    /// Per layer: the evictable hot blocks `(last use, block)` (outside the window, unpinned).
    lru: Vec<BTreeSet<(u64, u32)>>,
    /// Lowest-tier slots reserved at admission and not yet taken.
    left: u64,
    /// Blocks `0..flushed` are complete and flushed.
    flushed: u32,
    /// The window start the indexes last saw.
    ws_indexed: u32,
    /// Per layer: the clock at the start of its latest gather (`u64::MAX` before the first) --
    /// the blocks it selected are no other gather's VRAM victims.
    epoch: Vec<u64>,
    /// FN45b4: the sequence is mid-prompt ([`KvTier::protect`]): its evictable blocks sit in its
    /// `lru` only (not the global LRU), so every other sequence's block is a victim first.
    protected: bool,
    /// FN45c: parked ([`KvTier::park`]): no hot block but its open ones, its RAM copies are the
    /// warm cache's victims; no step may touch it before [`KvTier::unpark`].
    parked: bool,
    /// FN45c (resident mode): RAM slots this sequence holds reserved.
    ram_res: u64,
}

/// The tier of one rank (host bookkeeping; the bytes live in the mover's slabs and the file).
#[derive(Debug)]
pub struct KvTier {
    plan: KvTierPlan,
    seqs: Vec<Option<Seq>>,
    vram_free: Vec<u32>,
    ram_free: Vec<u32>,
    ssd_free: Vec<u32>,
    reserved: u64,
    /// Every evictable hot block: `(last use, seq, layer, block)`.
    global: BTreeSet<(u64, u32, u32, u32)>,
    /// Every droppable RAM copy (on the SSD, unpinned): `(last use, seq, layer, block)`.
    warm: BTreeSet<(u64, u32, u32, u32)>,
    clock: u64,
    counts: KvCounts,
    /// FN45b: the `(seq, layer)` pairs whose placements changed since [`Self::take_dirty`] (the
    /// device page table's rows to rewrite).
    dirty: BTreeSet<(u32, u32)>,
    /// FN45c: the ENGINE's law ([`KvTier::resident`]): an active sequence's every block keeps a
    /// RAM (or hot) copy -- the kernels read RAM over UVA, never the drive -- so only PARKED
    /// sequences' RAM copies are warm victims, and admission reserves RAM for active sequences
    /// (by request length) beside the drive's reservation for every sequence.
    resident: bool,
    /// FN45c: RAM slots reserved by the resident law.
    ram_reserved: u64,
    /// FN51f: the ELASTIC ceiling -- the KV places blocks in hot slots `0..cap` only; slots
    /// `cap..hot_slots` are LENT to another client (the expert tier). `hot_slots` without
    /// `--fn-vram-elastic` (today's law, never moved).
    cap: u32,
    /// LR-G (`--fn-kv-lazy`): a sequence's reservation GROWS on demand -- an append / open past
    /// its admitted tokens reserves the missing blocks ([`KvTier::grow`]) instead of refusing --
    /// and the elastic want counts WRITTEN blocks ([`KvTier::hot_want`]). The scheduler's arena
    /// grows first (the same tokens, block by block), so a growth here fits the slab by
    /// construction; one that does not is the loud `Admission` refusal, never a silent clamp.
    lazy: bool,
}

impl KvTier {
    /// An empty tier over `plan`'s slots.
    #[must_use]
    pub fn new(plan: KvTierPlan) -> KvTier {
        let cap = plan.hot_slots;
        KvTier {
            seqs: vec![None; plan.max_seqs as usize + 1],
            vram_free: (0..plan.hot_slots).rev().collect(),
            ram_free: (0..plan.ram_slots).rev().collect(),
            ssd_free: (0..plan.ssd_slots).rev().collect(),
            plan,
            reserved: 0,
            global: BTreeSet::new(),
            warm: BTreeSet::new(),
            clock: 0,
            counts: KvCounts::default(),
            dirty: BTreeSet::new(),
            resident: false,
            ram_reserved: 0,
            cap,
            lazy: false,
        }
    }

    /// LR-G: switch an EMPTY tier to lazy growth (see the field).
    pub fn lazy(mut self) -> Result<KvTier, KvTierError> {
        if self.seqs.iter().any(Option::is_some) {
            return Err(KvTierError::State("lazy growth needs an empty tier".into()));
        }
        self.lazy = true;
        Ok(self)
    }

    /// LR-G: the tier grows reservations on demand.
    #[must_use]
    pub fn is_lazy(&self) -> bool {
        self.lazy
    }

    /// LR-G: grow `seq`'s reservation to `tokens` (never past the max context): reserves the
    /// lowest tier's slots (and, under the resident law, the RAM) of the new blocks, or refuses
    /// with `Admission` and changes nothing. A `tokens` at or below today's reservation is a no-op.
    /// No byte moves: the blocks are opened by the next append / open exactly as an admitted
    /// sequence's are (the copy-order law K1-K4 is unchanged).
    pub fn grow(&mut self, seq: u32, tokens: u32) -> Result<(), KvTierError> {
        let s = self.seq(seq)?;
        let tokens = tokens.min(self.plan.max_ctx);
        if tokens <= s.max_tokens {
            return Ok(());
        }
        let l = u64::from(self.plan.layers);
        let more = l * u64::from(self.plan.blocks(tokens) - self.plan.blocks(s.max_tokens));
        let free = u64::from(self.plan.backing_slots()) - self.reserved;
        if more > free {
            return Err(KvTierError::Admission { want: more, free });
        }
        let ram_more = if self.resident && !s.parked {
            let rfree = u64::from(self.plan.ram_slots).saturating_sub(self.ram_reserved);
            if more > rfree {
                return Err(KvTierError::Admission {
                    want: more,
                    free: rfree,
                });
            }
            more
        } else {
            0
        };
        self.reserved += more;
        self.ram_reserved += ram_more;
        let s = self.seqs[seq as usize].as_mut().expect("live");
        s.max_tokens = tokens;
        s.left += more;
        s.ram_res += ram_more;
        Ok(())
    }

    /// FN51f: the elastic ceiling (hot slots `0..cap` are the KV's; the rest are lent).
    #[must_use]
    pub fn hot_cap(&self) -> u32 {
        self.cap
    }

    /// FN51f: the hot slots the LIVE sequences need placed now (the floor's law over the live set
    /// instead of `max_seqs`): per live, unparked sequence and layer its window's blocks (or all
    /// of its reservation's when shorter) + one widest chunk's open blocks. Below this the KV
    /// could fail an append (`NoRoom`); the elastic controller never lends into it.
    #[must_use]
    pub fn hot_floor(&self) -> u64 {
        let p = &self.plan;
        let wb = p.window_tokens / p.block_tokens + 1;
        let cb = u64::from(p.chunk_tokens.div_ceil(p.block_tokens)) + 1;
        let live: u64 = self
            .seqs
            .iter()
            .flatten()
            .filter(|s| !s.parked)
            .map(|s| {
                // LR-G: a lazy reservation may grow inside the step (the open past it), so the
                // floor holds the growth one step can take beside it
                let grow = if self.lazy { LAZY_STEP_BLOCKS } else { 0 };
                u64::from((p.blocks(s.max_tokens) + grow).min(wb))
            })
            .sum();
        u64::from(p.layers) * (live + cb)
    }

    /// FN51f: the hot slots the live sequences could USE: every block of every reservation
    /// (a long request's out-of-window blocks hot instead of read over UVA) + the chunk.
    #[must_use]
    pub fn hot_want(&self) -> u64 {
        let p = &self.plan;
        let cb = u64::from(p.chunk_tokens.div_ceil(p.block_tokens)) + 1;
        let live: u64 = self
            .seqs
            .iter()
            .flatten()
            .filter(|s| !s.parked)
            .map(|s| {
                let reserved = p.blocks(s.max_tokens) + 1;
                if self.lazy {
                    // LR-G: the WRITTEN blocks (+ the open append's: every opened block) + one,
                    // never more than the reservation -- unwritten max_tokens pull no lent
                    // segment back
                    (s.layers[0].len() as u32 + 1).min(reserved)
                } else {
                    reserved
                }
            })
            .map(u64::from)
            .sum();
        u64::from(p.layers) * (live + cb)
    }

    /// FN130: the admitted token reservation of `seq` (`None` when not live).
    #[must_use]
    pub fn max_tokens(&self, seq: u32) -> Option<u32> {
        self.seqs
            .get(seq as usize)
            .and_then(Option::as_ref)
            .map(|s| s.max_tokens)
    }

    /// FN130: the hot slots an append could take WITHOUT growing the cap: the free list + every
    /// block a VRAM victim may be (the global LRU = the unprotected sequences' evictable blocks,
    /// + every protected sequence's own evictable blocks: the guarded victims of `take_vram`).
    #[must_use]
    pub fn hot_takeable(&self) -> u64 {
        let guarded: usize = self
            .seqs
            .iter()
            .flatten()
            .filter(|s| s.protected)
            .map(|s| s.lru.iter().map(BTreeSet::len).sum::<usize>())
            .sum();
        (self.vram_free.len() + self.global.len() + guarded) as u64
    }

    /// FN130 (THE DEMAND LAW's page count): hot slots the opens of a step need past what the pool
    /// can give now ([`Self::hot_takeable`]) -- `opens` = `(seq, blocks the sequence will hold per
    /// layer after the step)`; a sequence not live counts from 0. Conservative: the blocks the
    /// step's own commit lets out of the window are not counted as takeable (a demand reclaim may
    /// grow the cap a segment early, never late). 0 = the step fits under the cap.
    #[must_use]
    pub fn open_shortfall(&self, opens: &[(u32, u32)]) -> u64 {
        let new: u64 = opens
            .iter()
            .map(|&(s, nb)| {
                let have = self.blocks_of(s, 0);
                u64::from(nb.saturating_sub(have)) * u64::from(self.plan.layers)
            })
            .sum();
        new.saturating_sub(self.hot_takeable())
    }

    /// FN51f: whether hot slots `new_cap..cap` can be LENT now: each is free or holds an
    /// evictable block (complete, flushed, outside its window -- K2: dropping it copies nothing).
    #[must_use]
    pub fn can_shrink_hot(&self, new_cap: u32) -> bool {
        new_cap <= self.cap && self.shrink_victims(new_cap).is_some()
    }

    fn shrink_victims(&self, new_cap: u32) -> Option<Vec<(u32, u32, u32)>> {
        let mut out = Vec::new();
        for (si, s) in self.seqs.iter().enumerate() {
            let Some(s) = s else { continue };
            let ws = self.win_start(s.len);
            for (l, blocks) in s.layers.iter().enumerate() {
                for (b, blk) in blocks.iter().enumerate() {
                    let Some(v) = blk.vram else { continue };
                    if v < new_cap || v >= self.cap {
                        continue;
                    }
                    if blk.pinned || (b as u32) >= ws {
                        return None;
                    }
                    out.push((si as u32, l as u32, b as u32));
                }
            }
        }
        Some(out)
    }

    /// FN51f: LEND hot slots `new_cap..cap` (the top of the pool): every evictable block placed
    /// there is evicted (no copy: K2 -- its RAM copy is current; the page table's rows turn
    /// dirty and the next step's upload points them at RAM before any kernel runs), every free
    /// one leaves the free list. Refuses with no change when a window or open block sits there.
    /// Returns the blocks evicted.
    pub fn shrink_hot(&mut self, new_cap: u32) -> Result<u32, KvTierError> {
        if new_cap > self.cap {
            return Err(KvTierError::State(format!(
                "shrink the hot pool to {new_cap} above its cap {}",
                self.cap
            )));
        }
        let v = self.shrink_victims(new_cap).ok_or_else(|| {
            KvTierError::State(format!(
                "hot slots {new_cap}..{} hold a window or open block",
                self.cap
            ))
        })?;
        for &(s, l, b) in &v {
            self.evict(s, l, b);
        }
        self.vram_free.retain(|&x| x < new_cap);
        self.cap = new_cap;
        Ok(v.len() as u32)
    }

    /// FN51f-b (the lend's ALIAS CHECK): the KV blocks placed in hot slots `lo..hi` now, and
    /// whether any of those slots is still on the free list (a lend needs both zero: the range
    /// past the cap).
    #[must_use]
    pub fn hot_held_in(&self, lo: u32, hi: u32) -> (u32, bool) {
        let mut n = 0;
        for s in self.seqs.iter().flatten() {
            for blocks in &s.layers {
                n += blocks
                    .iter()
                    .filter(|b| b.vram.is_some_and(|v| v >= lo && v < hi))
                    .count() as u32;
            }
        }
        (n, self.vram_free.iter().any(|&v| v >= lo && v < hi))
    }

    /// FN51f: RECLAIM hot slots `cap..new_cap` (the other client gave them back): they join the
    /// free list (kept descending: pops ascend).
    pub fn grow_hot(&mut self, new_cap: u32) -> Result<(), KvTierError> {
        if new_cap < self.cap || new_cap > self.plan.hot_slots {
            return Err(KvTierError::State(format!(
                "grow the hot pool from {} to {new_cap} (of {})",
                self.cap, self.plan.hot_slots
            )));
        }
        self.vram_free.extend(self.cap..new_cap);
        self.vram_free.sort_unstable_by(|a, b| b.cmp(a));
        self.cap = new_cap;
        Ok(())
    }

    /// The plan.
    #[must_use]
    pub fn plan(&self) -> &KvTierPlan {
        &self.plan
    }

    /// The counters.
    #[must_use]
    pub fn counts(&self) -> KvCounts {
        self.counts
    }

    /// Free VRAM slots.
    #[must_use]
    pub fn vram_free(&self) -> usize {
        self.vram_free.len()
    }

    /// Free RAM slots.
    #[must_use]
    pub fn ram_free(&self) -> usize {
        self.ram_free.len()
    }

    /// Free SSD slots.
    #[must_use]
    pub fn ssd_free(&self) -> usize {
        self.ssd_free.len()
    }

    /// FN45b: the `(seq, layer)` pairs whose placements changed since the last call (append,
    /// eviction, gather copy, RAM drop, finish) -- the rows [`KvPageTable::sync`] rewrites.
    pub fn take_dirty(&mut self) -> BTreeSet<(u32, u32)> {
        std::mem::take(&mut self.dirty)
    }

    /// FN45b: blocks of `(seq, layer)` (0 when not live).
    #[must_use]
    pub fn blocks_of(&self, seq: u32, layer: u32) -> u32 {
        self.seqs
            .get(seq as usize)
            .and_then(Option::as_ref)
            .and_then(|s| s.layers.get(layer as usize))
            .map_or(0, |l| l.len() as u32)
    }

    /// Tokens of `seq` (0 when not live).
    #[must_use]
    pub fn len(&self, seq: u32) -> u32 {
        self.seqs
            .get(seq as usize)
            .and_then(Option::as_ref)
            .map_or(0, |s| s.len)
    }

    fn win_start(&self, len: u32) -> u32 {
        len.saturating_sub(self.plan.window_tokens) / self.plan.block_tokens
    }

    fn seq(&self, seq: u32) -> Result<&Seq, KvTierError> {
        self.seqs
            .get(seq as usize)
            .and_then(Option::as_ref)
            .ok_or_else(|| KvTierError::State(format!("sequence {seq} is not live")))
    }

    fn blk(&mut self, seq: u32, layer: u32, block: u32) -> &mut Blk {
        &mut self.seqs[seq as usize].as_mut().expect("live").layers[layer as usize][block as usize]
    }

    /// Where `(seq, layer, block)` is read from now.
    pub fn place(&self, seq: u32, layer: u32, block: u32) -> Result<Place, KvTierError> {
        let s = self.seq(seq)?;
        let b = s
            .layers
            .get(layer as usize)
            .and_then(|l| l.get(block as usize))
            .ok_or_else(|| KvTierError::State(format!("no block {seq}/{layer}/{block}")))?;
        match (b.vram, b.ram, b.ssd) {
            (Some(v), _, _) => Ok(Place::Vram(v)),
            (None, Some(r), _) => Ok(Place::Ram(r)),
            (None, None, Some(d)) => Ok(Place::Ssd(d)),
            (None, None, None) => Err(KvTierError::State(format!(
                "block {seq}/{layer}/{block} has no copy"
            ))),
        }
    }

    /// Hot blocks of `(seq, layer)` outside the window.
    pub fn hot_outside(&self, seq: u32, layer: u32) -> Result<u32, KvTierError> {
        let s = self.seq(seq)?;
        let l = &s.layers[layer as usize];
        let ws = (self.win_start(s.len) as usize).min(l.len());
        Ok(l[..ws].iter().filter(|b| b.vram.is_some()).count() as u32)
    }

    /// Admit `seq` for up to `max_tokens` (the scheduler's admission by the lowest tier's
    /// budget): reserves every block it may create, or refuses.
    pub fn admit(&mut self, seq: u32, max_tokens: u32) -> Result<(), KvTierError> {
        if seq > self.plan.max_seqs || max_tokens > self.plan.max_ctx {
            return Err(KvTierError::State(format!(
                "admit {seq} at {max_tokens} tokens: {} slots x {} ctx",
                self.plan.max_seqs, self.plan.max_ctx
            )));
        }
        if self.seqs[seq as usize].is_some() {
            return Err(KvTierError::State(format!("sequence {seq} already live")));
        }
        let want = u64::from(self.plan.layers) * u64::from(self.plan.blocks(max_tokens));
        let free = u64::from(self.plan.backing_slots()) - self.reserved;
        if want > free {
            return Err(KvTierError::Admission { want, free });
        }
        let ram_res = if self.resident {
            let rfree = u64::from(self.plan.ram_slots).saturating_sub(self.ram_reserved);
            if want > rfree {
                return Err(KvTierError::Admission { want, free: rfree });
            }
            want
        } else {
            0
        };
        self.ram_reserved += ram_res;
        self.reserved += want;
        let nl = self.plan.layers as usize;
        self.seqs[seq as usize] = Some(Seq {
            len: 0,
            max_tokens,
            layers: vec![Vec::new(); nl],
            lru: vec![BTreeSet::new(); nl],
            left: want,
            flushed: 0,
            ws_indexed: 0,
            epoch: vec![u64::MAX; nl],
            protected: false,
            parked: false,
            ram_res,
        });
        Ok(())
    }

    fn unindex(&mut self, seq: u32, layer: u32, block: u32) {
        let last = self.blk(seq, layer, block).last;
        let s = self.seqs[seq as usize].as_mut().expect("live");
        s.lru[layer as usize].remove(&(last, block));
        self.global.remove(&(last, seq, layer, block));
        self.warm.remove(&(last, seq, layer, block));
    }

    fn index(&mut self, seq: u32, layer: u32, block: u32) {
        let ws = self.win_start(self.seqs[seq as usize].as_ref().expect("live").len);
        let b = self.blk(seq, layer, block).clone();
        if b.pinned {
            return;
        }
        if b.vram.is_some() && block < ws {
            let s = self.seqs[seq as usize].as_mut().expect("live");
            s.lru[layer as usize].insert((b.last, block));
            if !s.protected {
                self.global.insert((b.last, seq, layer, block));
            }
        }
        if b.ram.is_some() && b.on_ssd && self.warm_ok(seq) {
            self.warm.insert((b.last, seq, layer, block));
        }
    }

    /// A RAM copy of `seq` may be a warm victim (the inclusive law; the resident law: parked only).
    fn warm_ok(&self, seq: u32) -> bool {
        !self.resident || self.seqs[seq as usize].as_ref().is_some_and(|s| s.parked)
    }

    fn touch(&mut self, seq: u32, layer: u32, block: u32) {
        self.unindex(seq, layer, block);
        self.clock += 1;
        self.blk(seq, layer, block).last = self.clock;
        self.index(seq, layer, block);
    }

    /// Evict one hot block: its VRAM slot returns to the free list (no copy: K2).
    fn evict(&mut self, seq: u32, layer: u32, block: u32) {
        self.unindex(seq, layer, block);
        let slot = self.blk(seq, layer, block).vram.take().expect("hot");
        self.vram_free.push(slot);
        self.counts.evictions += 1;
        self.dirty.insert((seq, layer));
        self.index(seq, layer, block);
    }

    /// Drop one RAM copy (the SSD has the bytes: no copy).
    fn drop_ram(&mut self, seq: u32, layer: u32, block: u32) {
        self.unindex(seq, layer, block);
        let slot = self.blk(seq, layer, block).ram.take().expect("warm");
        self.ram_free.push(slot);
        self.counts.ram_drops += 1;
        self.dirty.insert((seq, layer));
        self.index(seq, layer, block);
    }

    /// A VRAM slot: the free list, else the global LRU victim that is not `(seq, layer, b)` for
    /// `b` in `spare` and (with `protect`: a gather's miss) not in its own layer's latest
    /// selection. FN45b4: the global LRU holds the UNPROTECTED sequences' blocks only; a gather
    /// (`protect`) never evicts a prefilling sequence's block; an append (the window must be hot)
    /// then takes the coldest block of another prefilling sequence, and last its own.
    fn take_vram(
        &mut self,
        seq: u32,
        layer: u32,
        spare: &BTreeSet<u32>,
        protect: bool,
    ) -> Option<u32> {
        if let Some(s) = self.vram_free.pop() {
            return Some(s);
        }
        let seqs = &self.seqs;
        let v = self
            .global
            .iter()
            .find(|&&(last, s, l, b)| {
                let protected =
                    protect && last >= seqs[s as usize].as_ref().expect("live").epoch[l as usize];
                !protected && !(s == seq && l == layer && spare.contains(&b))
            })
            .copied();
        let v = match v {
            Some(v) => v,
            None if protect => return None,
            None => self.guarded_victim(seq, true)?,
        };
        self.evict(v.1, v.2, v.3);
        self.vram_free.pop()
    }

    /// FN45b4: the coldest evictable block of a protected (prefilling) sequence other than `seq`
    /// (`own`: then `seq`'s own coldest when it is protected and no other has one).
    fn guarded_victim(&self, seq: u32, own: bool) -> Option<(u64, u32, u32, u32)> {
        let coldest = |si: usize, s: &Seq| {
            s.lru
                .iter()
                .enumerate()
                .filter_map(|(l, x)| x.first().map(|&(last, b)| (last, si as u32, l as u32, b)))
                .min()
        };
        let other = self
            .seqs
            .iter()
            .enumerate()
            .filter_map(|(si, s)| s.as_ref().map(|s| (si, s)))
            .filter(|(si, s)| s.protected && *si as u32 != seq)
            .filter_map(|(si, s)| coldest(si, s))
            .min();
        if other.is_some() || !own {
            return other;
        }
        let s = self.seqs.get(seq as usize)?.as_ref()?;
        if s.protected {
            coldest(seq as usize, s)
        } else {
            None
        }
    }

    /// FN45b4: protect `seq` while its prompt prefills (`on`) or release it (its first decode
    /// step): a protected sequence's blocks leave the global LRU, so appends and gathers take every
    /// other sequence's coldest block first and a decode gather never takes one of them.
    pub fn protect(&mut self, seq: u32, on: bool) -> Result<(), KvTierError> {
        let s = self
            .seqs
            .get_mut(seq as usize)
            .and_then(Option::as_mut)
            .ok_or_else(|| KvTierError::State(format!("sequence {seq} is not live")))?;
        if s.protected == on {
            return Ok(());
        }
        s.protected = on;
        for (l, set) in s.lru.iter().enumerate() {
            for &(last, b) in set {
                let key = (last, seq, l as u32, b);
                if on {
                    self.global.remove(&key);
                } else {
                    self.global.insert(key);
                }
            }
        }
        Ok(())
    }

    /// FN45b4: whether `seq` is protected (mid-prompt).
    #[must_use]
    pub fn is_protected(&self, seq: u32) -> bool {
        self.seqs
            .get(seq as usize)
            .and_then(Option::as_ref)
            .is_some_and(|s| s.protected)
    }

    /// FN45b4: THE PREFILL GATHER -- before a prompt chunk of `seq`, bring every block of the
    /// sequence that is not hot back into VRAM (the chunk's sparse rows may select any earlier
    /// group; the indexer's selection is not known on the host before the chunk). Slots come from
    /// the free list, then the unprotected sequences' coldest blocks, then the other prefilling
    /// sequences' coldest -- never `seq`'s own. The freed slots are taken in ascending order
    /// (layer-major, block-ascending) so the H2Ds coalesce into runs. A block that finds no slot
    /// stays in RAM (a counted UVA read: exact). No `Landed` mark: the engine issues the copies on
    /// the compute stream before the chunk (stream order is the fence).
    pub fn gather_seq(&mut self, seq: u32) -> Result<Gather, KvTierError> {
        self.active(seq)?;
        let s = self.seq(seq)?;
        let mut miss = Vec::new();
        for (l, blocks) in s.layers.iter().enumerate() {
            for (b, blk) in blocks.iter().enumerate() {
                if blk.vram.is_none() {
                    if blk.ram.is_none() {
                        return Err(KvTierError::State(format!(
                            "gather_seq {seq}/{l}/{b}: an SSD-only block (no RAM copy)"
                        )));
                    }
                    miss.push((l as u32, b as u32));
                }
            }
        }
        let mut out = Gather::default();
        while self.vram_free.len() < miss.len() {
            let v = self
                .global
                .iter()
                .find(|&&(_, s, _, _)| s != seq)
                .copied()
                .or_else(|| self.guarded_victim(seq, false));
            let Some(v) = v else { break };
            self.evict(v.1, v.2, v.3);
        }
        self.vram_free.sort_unstable_by(|a, b| b.cmp(a));
        for (l, b) in miss {
            let Some(v) = self.vram_free.pop() else {
                out.uva.push(b);
                continue;
            };
            self.unindex(seq, l, b);
            let blk = self.blk(seq, l, b);
            blk.vram = Some(v);
            let r = blk.ram.expect("checked above");
            out.ops.push(CopyOp::H2d { ram: r, vram: v });
            out.copied += 1;
            self.dirty.insert((seq, l));
            self.touch(seq, l, b);
        }
        self.counts.gathers += 1;
        self.counts.h2d += u64::from(out.copied);
        self.counts.uva += out.uva.len() as u64;
        Ok(out)
    }

    /// A RAM slot: the free list, else the warm LRU copy that is not `(seq, layer, b)` for `b` in
    /// `spare` (the copies a gather is about to read).
    fn take_ram(&mut self, seq: u32, layer: u32, spare: &BTreeSet<u32>) -> Option<u32> {
        if let Some(s) = self.ram_free.pop() {
            return Some(s);
        }
        let v = self
            .warm
            .iter()
            .find(|&&(_, s, l, b)| !(s == seq && l == layer && spare.contains(&b)))
            .copied()?;
        self.drop_ram(v.1, v.2, v.3);
        self.ram_free.pop()
    }

    /// Open an append of `n` (at most `chunk_tokens`) tokens to `seq` (a prefill chunk or a
    /// decode step's rows): every new block gets its lowest-tier slot, a RAM staging slot and a
    /// VRAM slot, pinned until [`Self::commit`] completes it. Returns no copies (an eviction
    /// copies nothing: K2); the compute that writes the rows waits for every issued copy first.
    pub fn append(&mut self, seq: u32, n: u32) -> Result<(), KvTierError> {
        self.active(seq)?;
        if n > self.plan.chunk_tokens {
            return Err(KvTierError::State(format!(
                "append of {n} tokens past the {}-token chunk",
                self.plan.chunk_tokens
            )));
        }
        let s = self.seq(seq)?;
        let new = s.len + n;
        if new > s.max_tokens && self.lazy {
            self.grow(seq, new)?;
        }
        let s = self.seq(seq)?;
        if new > s.max_tokens {
            return Err(KvTierError::State(format!(
                "sequence {seq}: {new} tokens past its admitted {}",
                s.max_tokens
            )));
        }
        self.open_blocks(seq, new.div_ceil(self.plan.block_tokens))?;
        self.seqs[seq as usize].as_mut().expect("live").len = new;
        Ok(())
    }

    /// FN45b2: open (pinned, hot, with a staging slot) every block of `seq` below `tokens`
    /// WITHOUT moving its length: the speculative rows a decode / verify step writes past the
    /// committed length (and the MTP chain's) land on open pages; [`Self::append`] later moves the
    /// length over them and [`Self::commit`] flushes them once complete. Never past `max_tokens`.
    pub fn open(&mut self, seq: u32, tokens: u32) -> Result<(), KvTierError> {
        self.active(seq)?;
        if self.lazy {
            self.grow(seq, tokens)?;
        }
        let s = self.seq(seq)?;
        let tokens = tokens.min(s.max_tokens);
        self.open_blocks(seq, tokens.div_ceil(self.plan.block_tokens))
    }

    fn open_blocks(&mut self, seq: u32, nb_new: u32) -> Result<(), KvTierError> {
        let nb_old = self.seq(seq)?.layers[0].len() as u32;
        let none = BTreeSet::new();
        for layer in 0..self.plan.layers {
            for _ in nb_old..nb_new {
                let vram = self
                    .take_vram(seq, layer, &none, false)
                    .ok_or(KvTierError::NoRoom {
                        pool: "vram",
                        slots: self.plan.hot_slots,
                    })?;
                let ram = self
                    .take_ram(seq, layer, &none)
                    .ok_or(KvTierError::NoRoom {
                        pool: "ram",
                        slots: self.plan.ram_slots,
                    })?;
                let ssd = if self.plan.has_ssd() {
                    Some(self.ssd_free.pop().expect("admission reserved it"))
                } else {
                    None
                };
                self.dirty.insert((seq, layer));
                let s = self.seqs[seq as usize].as_mut().expect("live");
                s.left -= 1;
                s.layers[layer as usize].push(Blk {
                    ram: Some(ram),
                    vram: Some(vram),
                    ssd,
                    on_ssd: false,
                    last: 0,
                    pinned: true,
                });
            }
        }
        Ok(())
    }

    /// FN45b2: move sequence `from` to slot `to` (the pools' compaction / swap through the
    /// spare slot `max_seqs`): a sequence still live at `to` is finished first; no byte moves --
    /// the page table's rows of both slots are rewritten.
    pub fn move_seq(&mut self, from: u32, to: u32) -> Result<(), KvTierError> {
        if from == to {
            return Ok(());
        }
        if to as usize >= self.seqs.len() {
            return Err(KvTierError::State(format!("move to {to}: no such slot")));
        }
        self.seq(from)?;
        if self.seqs[to as usize].is_some() {
            self.finish(to)?;
        }
        let s = self.seqs[from as usize].take().expect("live");
        let nl = self.plan.layers;
        for l in 0..nl {
            for b in 0..s.layers[l as usize].len() as u32 {
                let key = (s.layers[l as usize][b as usize].last, from, l, b);
                if self.global.remove(&key) {
                    self.global.insert((key.0, to, l, b));
                }
                if self.warm.remove(&key) {
                    self.warm.insert((key.0, to, l, b));
                }
            }
        }
        self.seqs[to as usize] = Some(s);
        self.dirty
            .extend((0..nl).flat_map(|l| [(from, l), (to, l)]));
        Ok(())
    }

    /// FN45b2: whether `seq` is live.
    #[must_use]
    pub fn is_live(&self, seq: u32) -> bool {
        self.seqs.get(seq as usize).is_some_and(Option::is_some)
    }

    /// FN45c: switch an EMPTY tier with an SSD tier to the engine's resident law (an active
    /// sequence never loses its RAM copy; only parked sequences' copies are warm victims; RAM is
    /// reserved per active sequence at admission by its request length).
    pub fn resident(mut self) -> Result<KvTier, KvTierError> {
        if !self.plan.has_ssd() || self.seqs.iter().any(Option::is_some) {
            return Err(KvTierError::State(
                "the resident law needs an SSD tier and an empty tier".into(),
            ));
        }
        self.resident = true;
        Ok(self)
    }

    /// FN45c: RAM slots reserved by active sequences (the resident law).
    #[must_use]
    pub fn ram_reserved(&self) -> u64 {
        self.ram_reserved
    }

    /// FN45c: `seq` is parked.
    #[must_use]
    pub fn is_parked(&self, seq: u32) -> bool {
        self.seqs
            .get(seq as usize)
            .and_then(Option::as_ref)
            .is_some_and(|s| s.parked)
    }

    fn active(&self, seq: u32) -> Result<(), KvTierError> {
        if self.seq(seq)?.parked {
            return Err(KvTierError::State(format!("sequence {seq} is parked")));
        }
        Ok(())
    }

    /// FN45c: PARK `seq` (idle): every complete block must already be on the drive (the commit's
    /// write-through; `commit` first); its hot blocks but the open ones leave VRAM (no copy: RAM
    /// holds them), its RAM copies become the warm cache's victims and its RAM reservation is
    /// released down to the open blocks it still holds. Returns the blocks evicted from VRAM.
    pub fn park(&mut self, seq: u32) -> Result<u32, KvTierError> {
        self.active(seq)?;
        self.protect(seq, false)?;
        let s = self.seq(seq)?;
        let mut hot = Vec::new();
        let mut open = 0u64;
        for (l, blocks) in s.layers.iter().enumerate() {
            for (b, blk) in blocks.iter().enumerate() {
                if blk.pinned {
                    open += 1;
                    continue;
                }
                if !blk.on_ssd {
                    return Err(KvTierError::State(format!(
                        "park {seq}: block {l}/{b} is complete but not on the drive"
                    )));
                }
                if blk.vram.is_some() {
                    hot.push((l as u32, b as u32));
                }
            }
        }
        for &(l, b) in &hot {
            self.unindex(seq, l, b);
            let v = self.blk(seq, l, b).vram.take().expect("hot");
            self.vram_free.push(v);
            self.counts.evictions += 1;
            self.dirty.insert((seq, l));
        }
        let s = self.seqs[seq as usize].as_mut().expect("live");
        s.parked = true;
        if self.resident {
            let keep = open.min(s.ram_res);
            self.ram_reserved -= s.ram_res - keep;
            s.ram_res = keep;
        }
        let nl = self.plan.layers;
        for l in 0..nl {
            for b in 0..self.blocks_of(seq, l) {
                self.unindex(seq, l, b);
                self.index(seq, l, b);
            }
        }
        self.vram_free.sort_unstable_by(|a, b| b.cmp(a));
        Ok(hot.len() as u32)
    }

    /// FN45c: UNPARK `seq` before its next step: re-reserve its RAM (refused, still parked, if
    /// the active sequences' reservations leave no room = the admission law), read every block
    /// whose RAM copy the warm cache dropped back from the drive (`ssd_reads`, landed first),
    /// then copy its window back into VRAM (`ops`, H2D after the reads).
    pub fn unpark(&mut self, seq: u32) -> Result<Gather, KvTierError> {
        let s = self.seq(seq)?;
        if !s.parked {
            return Err(KvTierError::State(format!("unpark {seq}: not parked")));
        }
        if self.resident {
            let full = u64::from(self.plan.layers) * u64::from(self.plan.blocks(s.max_tokens));
            let need = full - s.ram_res;
            let rfree = u64::from(self.plan.ram_slots).saturating_sub(self.ram_reserved);
            if need > rfree {
                return Err(KvTierError::Admission {
                    want: need,
                    free: rfree,
                });
            }
            self.ram_reserved += need;
            self.seqs[seq as usize].as_mut().expect("live").ram_res = full;
        }
        let nl = self.plan.layers;
        self.seqs[seq as usize].as_mut().expect("live").parked = false;
        for l in 0..nl {
            for b in 0..self.blocks_of(seq, l) {
                self.unindex(seq, l, b);
                self.index(seq, l, b);
            }
        }
        let mut out = Gather::default();
        let none = BTreeSet::new();
        for l in 0..nl {
            for b in 0..self.blocks_of(seq, l) {
                if self.blk(seq, l, b).ram.is_some() {
                    continue;
                }
                let r = self.take_ram(seq, l, &none).ok_or(KvTierError::NoRoom {
                    pool: "ram",
                    slots: self.plan.ram_slots,
                })?;
                self.unindex(seq, l, b);
                let blk = self.blk(seq, l, b);
                blk.ram = Some(r);
                let d = blk.ssd.expect("a block off RAM is on the SSD");
                out.ssd_reads.push((d, r));
                self.dirty.insert((seq, l));
                self.index(seq, l, b);
            }
        }
        let ws = self.win_start(self.seq(seq)?.len);
        for l in 0..nl {
            for b in ws..self.blocks_of(seq, l) {
                if self.blk(seq, l, b).vram.is_some() {
                    continue;
                }
                let v = self
                    .take_vram(seq, l, &none, false)
                    .ok_or(KvTierError::NoRoom {
                        pool: "vram",
                        slots: self.plan.hot_slots,
                    })?;
                self.unindex(seq, l, b);
                let blk = self.blk(seq, l, b);
                blk.vram = Some(v);
                let r = blk.ram.expect("read above");
                out.ops.push(CopyOp::H2d { ram: r, vram: v });
                out.copied += 1;
                self.dirty.insert((seq, l));
                self.touch(seq, l, b);
            }
        }
        self.counts.ssd_reads += out.ssd_reads.len() as u64;
        self.counts.h2d += u64::from(out.copied);
        Ok(out)
    }

    /// Close the open append of `seq` after the compute wrote its rows (K1): one D2H flush per
    /// block the append completed (then its SSD write), unpin, index the blocks the window let go
    /// as evictable (they stay hot until a slot is needed).
    pub fn commit(&mut self, seq: u32) -> Result<Commit, KvTierError> {
        self.active(seq)?;
        let s = self.seq(seq)?;
        let (len, from, ws_old) = (s.len, s.flushed, s.ws_indexed);
        let full = len / self.plan.block_tokens; // blocks 0..full are complete
        let ws = self.win_start(len);
        let mut out = Commit::default();
        for layer in 0..self.plan.layers {
            for b in from..full {
                let blk = self.blk(seq, layer, b);
                let (v, r) = (
                    blk.vram.expect("a pinned block is hot"),
                    blk.ram.expect("a pinned block has its staging slot"),
                );
                out.flush.push(CopyOp::D2h { vram: v, ram: r });
                let ssd = blk.ssd;
                if let Some(d) = ssd {
                    out.ssd_writes.push((r, d));
                    blk.on_ssd = true;
                }
                blk.pinned = false;
                self.counts.ssd_writes += u64::from(ssd.is_some());
                self.counts.d2h += 1;
                self.touch(seq, layer, b);
            }
            for b in ws_old..ws {
                self.unindex(seq, layer, b);
                self.index(seq, layer, b);
            }
        }
        let s = self.seqs[seq as usize].as_mut().expect("live");
        s.flushed = full;
        s.ws_indexed = ws;
        Ok(out)
    }

    /// One layer's gather for `seq`: `groups` = the step's selected groups (one row's top-512 or
    /// the union of a verify's rows; any order, duplicates allowed). Every selected block is
    /// touched; one on the SSD only is read into a RAM slot first (a counted SSD read); each one
    /// absent from VRAM gets an H2D into a free or evicted slot (never a selected, pinned or
    /// window block), or stays in RAM when no slot exists (UVA). Ends with the layer's `Landed`
    /// mark when any copy was queued.
    pub fn gather(&mut self, seq: u32, layer: u32, groups: &[u32]) -> Result<Gather, KvTierError> {
        self.active(seq)?;
        let s = self.seq(seq)?;
        if layer >= self.plan.layers {
            return Err(KvTierError::State(format!("layer {layer}")));
        }
        let gpb = self.plan.block_tokens / GROUP_TOKENS;
        let nbl = s.layers[layer as usize].len() as u32;
        let mut sel = BTreeSet::new();
        for &g in groups {
            let b = g / gpb;
            if b >= nbl {
                return Err(KvTierError::State(format!(
                    "group {g} past sequence {seq}'s {} tokens",
                    s.len
                )));
            }
            sel.insert(b);
        }
        let mut out = Gather::default();
        let mut miss = Vec::new();
        self.seqs[seq as usize].as_mut().expect("live").epoch[layer as usize] = self.clock + 1;
        for &b in &sel {
            if self.blk(seq, layer, b).vram.is_some() {
                out.hits += 1;
            } else {
                miss.push(b);
            }
            self.touch(seq, layer, b);
        }
        for b in miss {
            if self.blk(seq, layer, b).ram.is_none() {
                let r = self.take_ram(seq, layer, &sel).ok_or(KvTierError::NoRoom {
                    pool: "ram",
                    slots: self.plan.ram_slots,
                })?;
                self.unindex(seq, layer, b);
                let blk = self.blk(seq, layer, b);
                blk.ram = Some(r);
                let d = blk.ssd.expect("a block off RAM is on the SSD");
                out.ssd_reads.push((d, r));
                self.index(seq, layer, b);
            }
            // a (layer, sequence) at its quota recycles its own LRU block; below it, a free
            // slot, then the global LRU
            let at_quota = self.seqs[seq as usize].as_ref().expect("live").lru[layer as usize]
                .len()
                >= self.plan.quota as usize;
            let mut slot = None;
            if at_quota {
                let own = self.seqs[seq as usize].as_ref().expect("live").lru[layer as usize]
                    .iter()
                    .map(|&(_, x)| x)
                    .find(|x| !sel.contains(x));
                if let Some(v) = own {
                    self.evict(seq, layer, v);
                    slot = self.vram_free.pop();
                }
            }
            if slot.is_none() {
                slot = self.take_vram(seq, layer, &sel, true);
            }
            match slot {
                Some(v) => {
                    self.unindex(seq, layer, b);
                    let blk = self.blk(seq, layer, b);
                    blk.vram = Some(v);
                    let r = blk.ram.expect("landed above");
                    out.ops.push(CopyOp::H2d { ram: r, vram: v });
                    out.copied += 1;
                    self.dirty.insert((seq, layer));
                    self.index(seq, layer, b);
                }
                None => out.uva.push(b),
            }
        }
        if !out.ops.is_empty() {
            out.ops.push(CopyOp::Landed { layer });
        }
        self.counts.gathers += 1;
        self.counts.hits += u64::from(out.hits);
        self.counts.h2d += u64::from(out.copied);
        self.counts.uva += out.uva.len() as u64;
        self.counts.ssd_reads += out.ssd_reads.len() as u64;
        Ok(out)
    }

    /// Reclaim a finished sequence: every slot returns to its pool, the reservation is released.
    pub fn finish(&mut self, seq: u32) -> Result<(), KvTierError> {
        let s = self
            .seqs
            .get_mut(seq as usize)
            .and_then(Option::take)
            .ok_or_else(|| KvTierError::State(format!("finish {seq}: not live")))?;
        for (layer, blocks) in s.layers.iter().enumerate() {
            for (b, blk) in blocks.iter().enumerate() {
                let key = (blk.last, seq, layer as u32, b as u32);
                self.global.remove(&key);
                self.warm.remove(&key);
                self.vram_free.extend(blk.vram);
                self.ram_free.extend(blk.ram);
                self.ssd_free.extend(blk.ssd);
            }
        }
        self.reserved -= u64::from(self.plan.layers) * u64::from(self.plan.blocks(s.max_tokens));
        self.ram_reserved -= s.ram_res;
        self.dirty.extend((0..self.plan.layers).map(|l| (seq, l)));
        // FN45b2: keep the free lists descending (pops ascend) so a later append's blocks take
        // consecutive VRAM and RAM slots and its flush coalesces into a few runs (hold 1: an 8k
        // prefill after reuse flushed as ~6,700 single-page copies on one step's host wall)
        self.vram_free.sort_unstable_by(|a, b| b.cmp(a));
        self.ram_free.sort_unstable_by(|a, b| b.cmp(a));
        Ok(())
    }

    /// The bookkeeping law (tests): every slot of every pool is free or owned exactly once; the
    /// window and the open blocks are hot with a staging slot; every block has its lowest-tier
    /// slot and a valid copy; the two indexes hold exactly the evictable blocks; the reservation
    /// balances.
    pub fn check(&self) -> Result<(), String> {
        let mut vown = vec![0u8; self.plan.hot_slots as usize];
        // FN51f: the lent top is owned by the other client (counted once here)
        vown[self.cap as usize..].iter_mut().for_each(|c| *c = 1);
        if self.vram_free.iter().any(|&v| v >= self.cap) {
            return Err(format!("a free hot slot at or past the cap {}", self.cap));
        }
        let mut rown = vec![0u8; self.plan.ram_slots as usize];
        let mut down = vec![0u8; self.plan.ssd_slots as usize];
        self.vram_free.iter().for_each(|&v| vown[v as usize] += 1);
        self.ram_free.iter().for_each(|&r| rown[r as usize] += 1);
        self.ssd_free.iter().for_each(|&d| down[d as usize] += 1);
        let (mut hot_ix, mut warm_ix, mut reserved, mut ram_res) = (0usize, 0usize, 0u64, 0u64);
        let ssd = self.plan.has_ssd();
        for (si, s) in self.seqs.iter().enumerate() {
            let Some(s) = s else { continue };
            let ws = self.win_start(s.len);
            let mut taken = 0u64;
            for (l, blocks) in s.layers.iter().enumerate() {
                if (blocks.len() as u32) < self.plan.blocks(s.len)
                    || blocks[self.plan.blocks(s.len) as usize..]
                        .iter()
                        .any(|b| !b.pinned)
                {
                    return Err(format!(
                        "seq {si} layer {l}: {} blocks at {}",
                        blocks.len(),
                        s.len
                    ));
                }
                for (b, blk) in blocks.iter().enumerate() {
                    let b = b as u32;
                    let at = format!("seq {si} layer {l} block {b}");
                    taken += 1;
                    if let Some(v) = blk.vram {
                        vown[v as usize] += 1;
                    }
                    if let Some(r) = blk.ram {
                        rown[r as usize] += 1;
                    }
                    if let Some(d) = blk.ssd {
                        down[d as usize] += 1;
                    }
                    if (blk.pinned || (b >= ws && !s.parked)) && blk.vram.is_none() {
                        return Err(format!("{at}: window/open block not hot"));
                    }
                    if blk.pinned && blk.ram.is_none() {
                        return Err(format!("{at}: open block without its staging slot"));
                    }
                    if ssd != blk.ssd.is_some() || (!ssd && blk.ram.is_none()) {
                        return Err(format!("{at}: no lowest-tier slot"));
                    }
                    if !blk.pinned && !blk.on_ssd && blk.ram.is_none() {
                        return Err(format!("{at}: a complete block with no valid copy"));
                    }
                    let key = (blk.last, si as u32, l as u32, b);
                    let ev = blk.vram.is_some() && !blk.pinned && b < ws;
                    let inl = s.lru[l].contains(&(blk.last, b));
                    let ing = self.global.contains(&key);
                    // FN45b4: a protected sequence's evictable blocks are in its LRU only
                    let evg = ev && !s.protected;
                    if ev != inl || evg != ing {
                        return Err(format!("{at}: hot index {ev} {inl} {ing}"));
                    }
                    let wv = blk.ram.is_some()
                        && blk.on_ssd
                        && !blk.pinned
                        && (!self.resident || s.parked);
                    if self.resident && !s.parked && !blk.pinned && blk.ram.is_none() {
                        return Err(format!("{at}: an active sequence's block off RAM"));
                    }
                    if s.parked && blk.vram.is_some() && !blk.pinned {
                        return Err(format!("{at}: a parked block hot"));
                    }
                    if wv != self.warm.contains(&key) {
                        return Err(format!("{at}: warm index {wv}"));
                    }
                    hot_ix += usize::from(evg);
                    warm_ix += usize::from(wv);
                }
            }
            let all = u64::from(self.plan.layers) * u64::from(self.plan.blocks(s.max_tokens));
            if taken + s.left != all {
                return Err(format!("seq {si}: reservation does not balance"));
            }
            reserved += all;
            ram_res += s.ram_res;
        }
        if self.resident && ram_res != self.ram_reserved {
            return Err(format!("RAM reserved {ram_res} != {}", self.ram_reserved));
        }
        if self.resident && self.ram_reserved > u64::from(self.plan.ram_slots) {
            return Err(format!("RAM reserved {} past its slots", self.ram_reserved));
        }
        if reserved != self.reserved {
            return Err(format!("reserved {reserved} != {}", self.reserved));
        }
        if hot_ix != self.global.len() || warm_ix != self.warm.len() {
            return Err(format!(
                "indexes {} / {} != {hot_ix} / {warm_ix}",
                self.global.len(),
                self.warm.len()
            ));
        }
        for (name, own) in [("VRAM", &vown), ("RAM", &rown), ("SSD", &down)] {
            if let Some(i) = own.iter().position(|&c| c != 1) {
                return Err(format!("{name} slot {i} owned {} times", own[i]));
            }
        }
        Ok(())
    }
}

/// The prompt-cache keys of `ids`: one `(tokens, key)` per complete `chunk`-token prefix, the key
/// a chained sha256 (`key_k = sha256(key_{k-1} || ids of chunk k as u32 LE)`, `key_0 =
/// sha256(model)`), so a conversation's keys extend its earlier turns' keys and a lookup walks
/// back from the longest prefix. `model` names the pack + geometry the bytes belong to.
#[must_use]
pub fn prefix_keys(model: &[u8], ids: &[u32], chunk: u32) -> Vec<(u32, [u8; 32])> {
    let mut key: [u8; 32] = Sha256::digest(model).into();
    let mut out = Vec::new();
    for (k, c) in ids.chunks_exact(chunk as usize).enumerate() {
        let mut h = Sha256::new();
        h.update(key);
        for t in c {
            h.update(t.to_le_bytes());
        }
        key = h.finalize().into();
        out.push(((k as u32 + 1) * chunk, key));
    }
    out
}

/// What a replay measured.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ReplayStats {
    /// Decode steps replayed.
    pub steps: u64,
    /// Selected blocks (all layers, all steps).
    pub selected: u64,
    /// ... found hot.
    pub hits: u64,
    /// ... copied in.
    pub copied: u64,
    /// ... read through UVA.
    pub uva: u64,
    /// ... read from the SSD first.
    pub ssd_reads: u64,
    /// Flush bytes (prefill + decode write-through).
    pub flush_bytes: u64,
    /// Out-of-window hot blocks per layer at the end (mean over layers).
    pub hot_outside: f64,
}

impl ReplayStats {
    /// Hit share of the selected blocks.
    #[must_use]
    pub fn hit(&self) -> f64 {
        self.hits as f64 / self.selected.max(1) as f64
    }
    /// Gather bytes per step per sequence over the link (copies + UVA reads).
    #[must_use]
    pub fn gather_bytes_per_step(&self, block_bytes: u64) -> f64 {
        ((self.copied + self.uva) * block_bytes) as f64 / self.steps.max(1) as f64
    }
}

/// THE HOST SIMULATION: replay one sequence's real QSA selections through a [`KvTier`]. `sel(layer,
/// row)` = the groups row `row` (position `p0 + row`) selected at `layer` (negative = padding);
/// the prompt's first `p0` tokens are prefilled in the plan's chunk appends, then every `step`
/// rows are one decode step: append `step` tokens, gather the union of the rows' selections per
/// layer, commit. Layers past the selection's count reuse layer `layer % sel_layers`.
pub fn replay<'a, F>(
    tier: &mut KvTier,
    seq: u32,
    p0: u32,
    rows: u32,
    step: u32,
    sel_layers: u32,
    sel: F,
) -> Result<ReplayStats, KvTierError>
where
    F: Fn(u32, u32) -> &'a [i32],
{
    let mut st = ReplayStats::default();
    tier.admit(seq, p0 + rows)?;
    let bb = tier.plan().block_bytes();
    let d2h0 = tier.counts().d2h;
    let mut done = 0;
    while done < p0 {
        let n = tier.plan().chunk_tokens.min(p0 - done);
        tier.append(seq, n)?;
        tier.commit(seq)?;
        done += n;
    }
    let mut r = 0;
    let mut groups = Vec::new();
    while r + step <= rows {
        tier.append(seq, step)?;
        for layer in 0..tier.plan().layers {
            groups.clear();
            for k in r..r + step {
                groups.extend(
                    sel(layer % sel_layers, k)
                        .iter()
                        .filter(|&&g| g >= 0)
                        .map(|&g| g as u32),
                );
            }
            let g = tier.gather(seq, layer, &groups)?;
            st.selected += u64::from(g.hits + g.copied) + g.uva.len() as u64;
            st.hits += u64::from(g.hits);
            st.copied += u64::from(g.copied);
            st.uva += g.uva.len() as u64;
            st.ssd_reads += g.ssd_reads.len() as u64;
        }
        tier.commit(seq)?;
        st.steps += 1;
        r += step;
    }
    st.flush_bytes = (tier.counts().d2h - d2h0) * bb;
    let nl = tier.plan().layers;
    let mut h = 0u64;
    for l in 0..nl {
        h += u64::from(tier.hot_outside(seq, l)?);
    }
    st.hot_outside = h as f64 / f64::from(nl);
    tier.finish(seq)?;
    Ok(st)
}

// ---- FN45b: the device page table ---------------------------------------------------------------
//
// The Flash-Next attention kernels ALREADY read K/V through a block table: FN6's prefill and
// decode bodies take `block_table` (i32 `[seqs, max_blocks]`) and `kv + (size_t)page *
// kv_block_stride`, FN19's score/union turn a group into `page * 16 + slot` (an i32 key slot, the
// gather attention's `kslot`), and FN6's KV append writes row `r` at the i64 slot `slot[r]`. Today
// the tables are the identity (`s * nb + j`) over each layer's flat `Pools` slab, so the tier needs
// NO new attention kernel: it hands the KV readers / writers a PER-LAYER table whose entries name
// a hot slot or the block's pinned-RAM copy (a UVA read, FN45a's counted overflow) as page indexes
// from ONE base address. The indexer's compressed store (`kv.index`, VRAM at max context) keeps the
// identity table: the score kernel and the store glue read `bt` for `comp`, never for K/V.
//
// The limit the kernels set: the union writes `page * 16 + slot` as an i32, so every page index
// (hot, RAM, the pad page) must stay below 2^27 = [`MAX_PAGE_INDEX`] -- the hot slab, the RAM slab
// and the pad page must sit within 2 TiB of the lowest of them, each a whole number of blocks
// from it. [`KvAddr::new`] refuses a boot that breaks either.

/// The page-index ceiling: FN19's union stores `page * 16 + slot` as an i32 (`flashnext_qsa.hip`).
pub const MAX_PAGE_INDEX: u64 = 1 << 27;

/// Where the kernels' page indexes point (one rank): the base address the kernels add
/// `page * block_bytes` to, and the page index of the hot slab's slot 0, the RAM slab's slot 0
/// (the DEVICE-visible address of the pinned slab) and the pad page (the pad law's spare: every
/// entry of a sequence that is not live, and every row a pad sequence writes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvAddr {
    /// The kernels' `kv` argument.
    pub base: u64,
    /// Bytes of one block of one layer (the kernels' `kv_block_stride`).
    pub block_bytes: u64,
    /// Page index of hot slot 0.
    pub hot: u32,
    /// Page index of RAM slot 0.
    pub ram: u32,
    /// Page index of the pad page.
    pub pad: u32,
    hot_slots: u32,
    ram_slots: u32,
}

impl KvAddr {
    /// The address law of a boot: refuses regions that are not a whole number of blocks apart,
    /// overlap, or reach past [`MAX_PAGE_INDEX`] pages from the lowest of them.
    pub fn new(
        block_bytes: u64,
        hot_addr: u64,
        hot_slots: u32,
        ram_addr: u64,
        ram_slots: u32,
        pad_addr: u64,
    ) -> Result<KvAddr, KvTierError> {
        let g = |m: String| Err(KvTierError::Geometry(m));
        if block_bytes == 0 {
            return g("zero block bytes".into());
        }
        let base = hot_addr.min(ram_addr).min(pad_addr);
        let mut ix = [0u64; 3];
        for (i, (what, a)) in [("hot", hot_addr), ("ram", ram_addr), ("pad", pad_addr)]
            .into_iter()
            .enumerate()
        {
            if (a - base) % block_bytes != 0 {
                return g(format!(
                    "the {what} region 0x{a:x} is not a whole number of {block_bytes}-byte blocks \
                     from the base 0x{base:x}"
                ));
            }
            ix[i] = (a - base) / block_bytes;
        }
        let spans = [
            (ix[0], u64::from(hot_slots), "hot"),
            (ix[1], u64::from(ram_slots), "ram"),
            (ix[2], 1, "pad"),
        ];
        for &(s, n, what) in &spans {
            if s + n > MAX_PAGE_INDEX {
                return g(format!(
                    "the {what} region ends at page {} past the kernels' {MAX_PAGE_INDEX}-page \
                     index (the union's i32 `page * 16 + slot`): the regions span more than 2 TiB",
                    s + n
                ));
            }
        }
        for (i, a) in spans.iter().enumerate() {
            for b in &spans[i + 1..] {
                if a.0 < b.0 + b.1 && b.0 < a.0 + a.1 {
                    return g(format!("the {} and {} regions overlap", a.2, b.2));
                }
            }
        }
        let t = |x: u64| x as u32; // < 2^27 above
        Ok(KvAddr {
            base,
            block_bytes,
            hot: t(ix[0]),
            ram: t(ix[1]),
            pad: t(ix[2]),
            hot_slots,
            ram_slots,
        })
    }

    /// The page index a placement reads (the SSD is never read by a kernel: a gather lands it).
    pub fn page(&self, p: Place) -> Result<i32, KvTierError> {
        match p {
            Place::Vram(v) if v < self.hot_slots => Ok((self.hot + v) as i32),
            Place::Ram(r) if r < self.ram_slots => Ok((self.ram + r) as i32),
            Place::Ssd(_) => Err(KvTierError::State(
                "a kernel cannot read an SSD-only block (gather it first)".into(),
            )),
            _ => Err(KvTierError::State(format!("{p:?} outside the slabs"))),
        }
    }

    /// What a page index names (the inverse of [`Self::page`]; `None` = the pad page).
    #[must_use]
    pub fn resolve(&self, page: i32) -> Option<Place> {
        let p = page as u32;
        if p >= self.hot && p < self.hot + self.hot_slots {
            Some(Place::Vram(p - self.hot))
        } else if p >= self.ram && p < self.ram + self.ram_slots {
            Some(Place::Ram(p - self.ram))
        } else {
            None
        }
    }
}

/// The per-layer device page tables of one rank: i32 `[layers, slots, nbs]`, layer-major so one
/// layer's `[slots, nbs]` is the `block_table` its KV readers take (pool slot `s`'s row at `s *
/// nbs`, the decode step's batch index == its pool slot, pads at the spare slot -- FN17's law).
/// `nbs` = the pools' blocks per sequence (`max_ctx / 16 + 1`). A live sequence's block `b < its
/// blocks` holds its placement's page; every other entry holds the pad page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvPageTable {
    addr: KvAddr,
    layers: u32,
    slots: u32,
    nbs: u32,
    table: Vec<i32>,
}

impl KvPageTable {
    /// The table of `plan` over `slots` pool slots (`max_seqs + 1`: the spare), all pad.
    pub fn new(plan: &KvTierPlan, slots: u32, addr: KvAddr) -> Result<KvPageTable, KvTierError> {
        if plan.block_tokens != PAGE_TOKENS {
            return Err(KvTierError::Geometry(format!(
                "the kernels' page is {PAGE_TOKENS} tokens, the plan's block {}",
                plan.block_tokens
            )));
        }
        if slots < plan.max_seqs {
            return Err(KvTierError::Geometry(format!(
                "{slots} pool slots under {} sequences",
                plan.max_seqs
            )));
        }
        let nbs = plan.max_ctx.div_ceil(PAGE_TOKENS) + 1;
        let n = plan.layers as usize * slots as usize * nbs as usize;
        Ok(KvPageTable {
            addr,
            layers: plan.layers,
            slots,
            nbs,
            table: vec![addr.pad as i32; n],
        })
    }

    /// The address law.
    #[must_use]
    pub fn addr(&self) -> KvAddr {
        self.addr
    }

    /// FN68d: the layers this table holds (13, or 12 + a one-layer MTP twin's 1).
    #[must_use]
    pub fn layers(&self) -> u32 {
        self.layers
    }

    /// Entries per row (the kernels' `max_blocks`).
    #[must_use]
    pub fn nbs(&self) -> u32 {
        self.nbs
    }

    /// Bytes of the whole table (the device copy's size).
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.table.len() as u64 * 4
    }

    /// Byte offset of `layer`'s `[slots, nbs]` table in the device copy.
    #[must_use]
    pub fn layer_offset(&self, layer: u32) -> u64 {
        u64::from(layer) * u64::from(self.slots) * u64::from(self.nbs) * 4
    }

    /// Row `(layer, slot)`.
    #[must_use]
    pub fn row(&self, layer: u32, slot: u32) -> &[i32] {
        let o = (layer as usize * self.slots as usize + slot as usize) * self.nbs as usize;
        &self.table[o..o + self.nbs as usize]
    }

    /// The whole table, little-endian (the upload image).
    #[must_use]
    pub fn le_bytes(&self) -> Vec<u8> {
        self.table.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// Rewrite row `(layer, seq)` from the tier's placements.
    pub fn rewrite(&mut self, tier: &KvTier, seq: u32, layer: u32) -> Result<(), KvTierError> {
        if seq >= self.slots || layer >= self.layers {
            return Err(KvTierError::State(format!(
                "row {seq}/{layer} outside the table"
            )));
        }
        let n = tier.blocks_of(seq, layer);
        if n > self.nbs {
            return Err(KvTierError::State(format!(
                "sequence {seq}: {n} blocks past the table's {}",
                self.nbs
            )));
        }
        let mut row = vec![self.addr.pad as i32; self.nbs as usize];
        for (b, e) in row.iter_mut().enumerate().take(n as usize) {
            *e = self.addr.page(tier.place(seq, layer, b as u32)?)?;
        }
        let o = (layer as usize * self.slots as usize + seq as usize) * self.nbs as usize;
        self.table[o..o + self.nbs as usize].copy_from_slice(&row);
        Ok(())
    }

    /// Rewrite every row the tier changed since the last sync; returns the rewritten rows as
    /// `(layer, slot)`, layer-major (the order of [`Self::upload_spans`]).
    pub fn sync(&mut self, tier: &mut KvTier) -> Result<Vec<(u32, u32)>, KvTierError> {
        let mut rows: Vec<(u32, u32)> =
            tier.take_dirty().into_iter().map(|(s, l)| (l, s)).collect();
        rows.sort_unstable();
        for &(l, s) in &rows {
            self.rewrite(tier, s, l)?;
        }
        Ok(rows)
    }

    /// The device copy's byte spans covering `rows` (`(offset, bytes)`, adjacent rows merged):
    /// what a step uploads before its first kernel (the in-order stream orders it before every
    /// read and after every earlier flush).
    #[must_use]
    pub fn upload_spans(&self, rows: &[(u32, u32)]) -> Vec<(u64, u64)> {
        let rb = u64::from(self.nbs) * 4;
        let mut out: Vec<(u64, u64)> = Vec::new();
        for &(l, s) in rows {
            let o = (u64::from(l) * u64::from(self.slots) + u64::from(s)) * rb;
            match out.last_mut() {
                Some(last) if last.0 + last.1 == o => last.1 += rb,
                _ => out.push((o, rb)),
            }
        }
        out
    }

    /// The append slot of a row of `slot` at position `pos` in `layer` (the KV append's i64
    /// `slot[r]`): `page * 16 + pos % 16` -- a hot page for every open block.
    #[must_use]
    pub fn append_slot(&self, layer: u32, slot: u32, pos: u32) -> i64 {
        let b = (pos / PAGE_TOKENS).min(self.nbs - 1);
        i64::from(self.row(layer, slot)[b as usize]) * i64::from(PAGE_TOKENS)
            + i64::from(pos % PAGE_TOKENS)
    }
}

/// One memcpy of a run of consecutive slots (`n` blocks from `vram` / `ram`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CopyRun {
    /// RAM -> VRAM (a gather) when true, VRAM -> RAM (a flush) when false.
    pub h2d: bool,
    /// First VRAM slot.
    pub vram: u32,
    /// First RAM slot.
    pub ram: u32,
    /// Blocks.
    pub n: u32,
}

/// The copies of `ops` as runs: adjacent same-direction copies whose VRAM AND RAM slots both
/// continue merge into one memcpy (a prefill chunk's flush on a fresh tier is one run a layer);
/// issue order is kept and `Landed` marks are dropped (the caller fences per layer).
#[must_use]
pub fn coalesce(ops: &[CopyOp]) -> Vec<CopyRun> {
    let mut out: Vec<CopyRun> = Vec::new();
    for op in ops {
        let (h2d, vram, ram) = match *op {
            CopyOp::H2d { ram, vram } => (true, vram, ram),
            CopyOp::D2h { vram, ram } => (false, vram, ram),
            CopyOp::Landed { .. } => continue,
        };
        match out.last_mut() {
            Some(r) if r.h2d == h2d && r.vram + r.n == vram && r.ram + r.n == ram => r.n += 1,
            _ => out.push(CopyRun {
                h2d,
                vram,
                ram,
                n: 1,
            }),
        }
    }
    out
}
