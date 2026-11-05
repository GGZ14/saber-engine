//! FN45b2: the TIERED KV (VRAM hot <- pinned RAM) wired into the Flash-Next engine on FN45b's page
//! table (`saber_mem::kv_tier`; docs/LEDGER.md FN45a / FN45b / FN45b2).
//!
//! **The wiring law.** Every K/V reader and writer of a QSA layer already addresses through a
//! table (FN6's bodies `kv + bt[seq * nb + j] * stride`, FN19's union `bt * 16 + slot`, the append's
//! i64 `slot[r]`). Under the tier the 13 QSA layers (12 target + the MTP layer) share ONE base
//! ([`KvAddr::base`]: the hot slab, the pad page and the pinned RAM slab's device address are page
//! indexes from it) and each layer reads its OWN `[slots, nbs]` table out of one device copy of the
//! [`KvPageTable`] (rows = pool slots, so a decode step's batch index == its pool slot; pads read
//! the pad rows of the slots past the live set); the append writes the per-layer `kslot` table the
//! step's constants carry (`[13, m]` i64 = `page * 16 + pos % 16`, a pad row at the pad page). A
//! verify step's per-row table (FN24's width-1 class, dense rows only: every row <= 2,050) is the
//! per-layer `kbtr` (`[13, m, KBTR]` i32). The score kernel and the indexer store keep the identity
//! table (they index the compressed store, `kv.index`, VRAM at max context). No kernel changes.
//!
//! **The step law.** Before a step's first kernel ([`KvDev::begin`], same stream): every live row
//! set `(slot, first position, end)` admits its slot (at the boot's max context: the RAM budget is
//! refused at boot, so admission never fails at run time), moves the sequence's FINAL length to
//! `first - SLACK` (a step rewrites at most the rows it feeds, the MTP chain's and its own verify
//! rows; nothing below `first - SLACK` is ever written again), commits it (one D2H flush per newly
//! complete block, coalesced into runs: K1 -- ordered after the steps that wrote them, before any
//! later step reuses a slot or reads the RAM copy), opens every block up to `end + AHEAD` (pinned,
//! hot), syncs the page table's changed rows into a pinned mirror and uploads only the changed
//! entries. One `run_batch` round trip a step. A block evicted from VRAM is read in place from RAM
//! (UVA, exact; the counted overflow). <= the hot window (8,192 - 128 tokens at the default hot
//! pool, today's flat pools' bytes) nothing is evicted.

use std::cell::RefCell;
use std::path::Path;

use anyhow::{ensure, Context};
use saber_hip::{DevicePtr, HipThread, Job, MemcpyKind, PinnedBuf, Stream};
use saber_mem::device_mover::CopyOp;
use saber_mem::dio::OpenMode;
use saber_mem::kv_ssd::KvSsd;
use saber_mem::kv_tier::{
    coalesce, KvAddr, KvDtype, KvPageTable, KvTier, KvTierPlan, Place, KV_TOKEN_LAYER_BYTES,
    PAGE_TOKENS, QSA_LAYERS,
};

/// Bytes of one 16-token page of one layer at bf16 (one KV head a rank: TP=2 only).
pub const BB: u64 = PAGE_TOKENS as u64 * KV_TOKEN_LAYER_BYTES;

/// FN68: bytes of one 16-token page of one layer at the page dtype `dt` (fp8 = [`BB`] / 2).
#[must_use]
pub fn page_bytes(dt: KvDtype) -> u64 {
    u64::from(PAGE_TOKENS) * dt.token_layer_bytes()
}

/// FN68: the slab's bytes per token per rank at `dt` (13 QSA layers; FN68d: a split boot's is
/// [`KvGeom::token_bytes_all`]).
#[must_use]
#[cfg(test)]
pub fn token_bytes(dt: KvDtype) -> u64 {
    u64::from(QSA_LAYERS) * dt.token_layer_bytes()
}
/// Tokens below a step's first row that may still be rewritten (the MTP chain's fed row, the
/// verify width): the final length lags the first row by this.
pub const SLACK: u32 = 32;
/// Tokens opened past a step's last row (the MTP chain writes up to k past it).
pub const AHEAD: u32 = 16;
/// Entries of a row of the per-row verify table (`kbtr`): a dense row sits at <= 2,050, so its
/// context is <= 2,051 tokens = 129 pages.
pub const KBTR: u32 = 132;
/// The hot window's ceiling (FN45a: the plan's window; 128 tokens under the pools' per-sequence
/// pages so the open blocks of every sequence fit today's bytes).
pub const WINDOW_MAX: u32 = 8192;
/// The default (`--fn-kv-tier` absent; FN45b4 flip constant): ON where the tier can run (the
/// worker form at TP=2, sparse QSA); `--fn-kv-tier off` = the flat pools (the control).
pub const DEFAULT_FN_KV_TIER: bool = true;
/// FN45b4: the served max context under the tier when `--fn-max-ctx` is absent = the PINNED-RAM
/// CEILING today (a 2,048 rung): the slab pins 8 x ctx x [`TOKEN_BYTES`] (+ a page) a rank; the
/// served boot's expert RAM cache covers its complement (5,750 slots a rank: serve + L2 + PLE =
/// 34.04 GiB) on this 60.4 GiB box (36.40 GiB pinnable), so 10,240 fits (+2.03 GiB = 36.07) and
/// 12,288 does not (36.48). 16k+ is RAM-bound (the 192 GB decision) or needs admission by request
/// length (the same slab, sequences past 8 x ctx / 8 while the aggregate fits) / the SSD tier.
pub const DEFAULT_FN_KV_MAX_CTX: u32 = 10_240;
/// FN51b: the SERVED max context when `--fn-max-ctx` is absent on a `--fn-serve` boot under the
/// tier = the checkpoint's 262,144 positions, served by ADMISSION BY RESERVATION over the same
/// admission slab ([`admit_slab_bytes`]): one pool of active tokens (272,528 a rank) shared by
/// whatever mix arrives, each request reserving prompt + `max_tokens` (a request without
/// `max_tokens` keeps the 10,240 clamp, or 16,384 headroom past it). [`DEFAULT_FN_KV_MAX_CTX`]
/// stays the slab's sizing law and the admission threshold; `--fn-max-ctx 10240` = the
/// pre-FN51b served boot (the control). Harness and scripted runs are unchanged.
pub const DEFAULT_FN_SERVED_MAX_CTX: u32 = 262_144; // FN51g flip (2026-10-02): every gate PASS once the leaked /tmp test scratch is cleared (MemAvailable law, LEDGER FN51g)

/// FN51f: THE DECODE CONTEXT BAND -- a decode-form step's attention segmentation context (the
/// split law's `max_ctx`, the sparse score / top-k extent, the index table's row length) follows
/// the LIVE context instead of the boot's: the smallest of these bands at or above the step's
/// longest row + [`BAND_MARGIN`], capped at the boot's max context. A boot at or under
/// [`DEFAULT_FN_KV_MAX_CTX`] keeps its own max context (byte-for-byte today's); on a longer boot
/// every step whose rows sit under the first band runs the 10,240 boot's launches (FN51e: the
/// 262k boot's b8 step paid +1.36 ms device for the boot-wide band). Bit-exact by the split law's
/// definition: the band is an input of the class, so equal bands give equal bytes.
pub const CTX_BANDS: [u32; 5] = [DEFAULT_FN_KV_MAX_CTX, 32_768, 65_536, 131_072, 262_144];
/// FN51f: positions past a step's first rows the band must cover (a verify width, the MTP chain's
/// passes ahead of their first one, the open pages): one band serves a whole chain.
pub const BAND_MARGIN: u32 = 64;

/// FN51f: the decode band at boot context `max_ctx` for rows ending below `need` positions.
#[must_use]
pub fn ctx_band(max_ctx: u32, need: u32) -> u32 {
    if max_ctx <= DEFAULT_FN_KV_MAX_CTX {
        return max_ctx;
    }
    CTX_BANDS
        .iter()
        .copied()
        .find(|&b| b >= need)
        .unwrap_or(max_ctx)
        .min(max_ctx)
}

/// FN51f: the band of a step whose rows start at `pos` (pads at 0 included).
#[must_use]
pub fn band_of(max_ctx: u32, pos: &[u32]) -> u32 {
    ctx_band(
        max_ctx,
        pos.iter().copied().max().unwrap_or(0) + BAND_MARGIN,
    )
}

/// FN51f: the index table's row length at `band` (the blocks a row of the band can address: the
/// pools' `nbs` law at the band's context).
#[must_use]
pub fn band_nb(band: u32) -> u32 {
    band.div_ceil(PAGE_TOKENS) + 1
}

/// The slab's bytes per token per rank (13 QSA layers x 1,024 B) at bf16 ([`token_bytes`]).
#[cfg(test)]
pub const TOKEN_BYTES: u64 = QSA_LAYERS as u64 * KV_TOKEN_LAYER_BYTES;
/// FN45c2: the drive slab's ceiling a rank (`--fn-kv-ssd-gib`; both ranks share /mnt/nvme).
pub const SSD_GIB_MAX: f64 = 64.0;
/// FN45c2: tokens a sequence's admission adds over the scheduler's reservation (the step's
/// [`AHEAD`] pages + a page of rounding + the [`SLACK`] lag).
pub const ADMIT_EXTRA: u32 = AHEAD + PAGE_TOKENS + SLACK;
/// FN51: the RAM slab a rank under ADMISSION BY RESERVATION (`--fn-max-ctx` past
/// [`DEFAULT_FN_KV_MAX_CTX`], `--fn-kv-ram-gib` absent) was 3.65 GiB (294,400 tokens a rank: one
/// 262,144-token sequence + ~31k of others). FN51e sizes it by [`admit_slab_tokens`] instead: one
/// max-context request + ONE request at the pre-FN51 implicit context beside it (3.386 GiB at
/// 262,144: -0.53 GiB pinned over both ranks; 26 x 10,240 without a long one). `--fn-kv-ram-gib
/// 3.65` = the FN51 slab (the control). The host law refuses the boot past the pinnable bytes
/// (beside the COMPLETE expert RAM cache).
///
/// The tokens the admission slab holds a rank at `max_ctx`: one `max_ctx` request + one
/// [`DEFAULT_FN_KV_MAX_CTX`] request beside it + [`ADMIT_EXTRA`] and a block of rounding for
/// every slot (the scheduler's budget, [`KvGeom::active_tokens`], is this less the extras).
#[must_use]
pub fn admit_slab_tokens(max_ctx: u32) -> u64 {
    u64::from(max_ctx) + u64::from(DEFAULT_FN_KV_MAX_CTX) + 9 * u64::from(ADMIT_EXTRA + PAGE_TOKENS)
}

/// FN51e: the admission slab's bytes a rank at `max_ctx` (every QSA layer's blocks of
/// [`admit_slab_tokens`]).
#[must_use]
#[cfg(test)]
pub fn admit_slab_bytes(max_ctx: u32) -> u64 {
    admit_slab_bytes_dt(max_ctx, KvDtype::Bf16)
}

/// FN68: [`admit_slab_bytes`] at the page dtype `dt` (the same tokens; fp8 = half the bytes).
#[must_use]
pub fn admit_slab_bytes_dt(max_ctx: u32, dt: KvDtype) -> u64 {
    admit_slab_bytes_tokens(admit_slab_tokens(max_ctx), dt)
}

/// LR-K: an admission slab of `tokens` a rank at the page dtype `dt` (every QSA layer's blocks).
#[must_use]
pub fn admit_slab_bytes_tokens(tokens: u64, dt: KvDtype) -> u64 {
    u64::from(QSA_LAYERS) * tokens.div_ceil(u64::from(PAGE_TOKENS)) * page_bytes(dt)
}

/// The QSA layer's tier index: decoder layer `l` (`l % 4 == 3`) -> `l / 4`; the MTP layer (48) ->
/// 12.
#[must_use]
pub fn tier_layer(l: i32) -> Option<u32> {
    if l == 48 {
        Some(QSA_LAYERS - 1)
    } else if l >= 0 && l % 4 == 3 && l < 48 {
        Some(l as u32 / 4)
    } else {
        None
    }
}

/// A boot's tier sizes (one rank).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvGeom {
    pub max_ctx: u32,
    pub window: u32,
    pub chunk: u32,
    pub hot_bytes: u64,
    pub ram_bytes: u64,
    /// FN45b3 (`--fn-kv-pages`): count, every step, the live (layer, block) entries the table
    /// points at the RAM slab (the kernels' UVA reads' ceiling; host time, a gate arm only).
    pub count_ram: bool,
    /// FN45b3 (`--fn-kv-drop-flush`, the eviction cell's NEGATIVE CONTROL): skip every D2H
    /// flush, so an evicted page read from RAM is stale -- the bytes must then move.
    pub drop_flush: bool,
    /// FN45b4: the prefill gather (default on; `--no-fn-kv-gather` = FN45b3's in-place reads,
    /// the control). The protection order holds either way.
    pub gather: bool,
    /// FN45c2 (`--fn-kv-ssd-gib`): the drive slab's bytes (0 = two tiers). With it the tier runs
    /// FN45c's RESIDENT law: the RAM slab holds the ACTIVE sequences' every block (admitted by
    /// request length, not 8 x the max context) and every complete block is written through to
    /// the drive after its flush lands.
    pub ssd_bytes: u64,
    /// FN51: ADMISSION BY RESERVATION on the two tiers -- the RAM slab ([`admit_slab_bytes`]) holds
    /// the ACTIVE sequences' every block, each admitted at its request length (`start`; the
    /// scheduler's KV budget = [`KvGeom::active_tokens`]), the index store paged out of one pool
    /// sized at the slab ([`KvGeom::ix_pages`]) instead of 9 x the max context.
    pub admit: bool,
    /// FN51: a harness sequence's admission (no scheduler reservation): its prompt + decode.
    pub first_admit: u32,
    /// FN68 (`--fn-kv-dtype`): the pages' element type. Every budget is in TOKENS first: the
    /// hot window, the slab and the drive copy hold the same tokens and fp8 pays half the bytes.
    pub dt: KvDtype,
    /// FN68d: the QSA layers THIS tier holds -- [`QSA_LAYERS`] (one tier), or 12 (the target's)
    /// when the MTP layer has its own one-layer twin ([`KvGeom::mtp`]) at another page dtype.
    pub layers: u32,
    /// FN68d (`--fn-kv-mtp-dtype`): the MTP layer's (tier layer 12's) own one-layer tier at
    /// another page dtype -- its budgets at the SAME pages a layer as this tier (so every
    /// admission, window and quota of the two is the same in tokens). None = one tier.
    pub mtp: Option<MtpTwin>,
    /// LR-G (`--fn-kv-lazy`, under admission only): the tier GROWS a sequence's reservation on
    /// demand (`KvTier::lazy`) after the scheduler's arena grew it, and the elastic want counts
    /// written blocks. False = today's whole reservation at `start`.
    pub lazy: bool,
}

/// FN68d: the MTP layer's one-layer twin tier ([`KvGeom::split_mtp`]): its page dtype and byte
/// budgets (the main tier's pages a layer at the twin's page bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MtpTwin {
    pub dt: KvDtype,
    pub hot_bytes: u64,
    pub ram_bytes: u64,
    pub ssd_bytes: u64,
}

impl KvGeom {
    /// The sizes at `max_ctx` / `max_chunk`: hot = today's flat pools at `min(max_ctx, 8,192)`
    /// (13 layers x 9 slots x pages) unless `hot_gib`; window = those pages less 128 tokens; RAM =
    /// every block of 8 x `max_ctx` unless `ram_gib`.
    #[must_use]
    #[cfg(test)]
    pub fn new(max_ctx: u32, max_chunk: u32, hot_gib: Option<f64>, ram_gib: Option<f64>) -> KvGeom {
        Self::new_dt(max_ctx, max_chunk, hot_gib, ram_gib, KvDtype::Bf16)
    }

    /// FN68: [`KvGeom::new`] at the page dtype `dt` (the same tokens; `--fn-kv-hot-gib` /
    /// `--fn-kv-ram-gib` stay byte budgets).
    #[must_use]
    pub fn new_dt(
        max_ctx: u32,
        max_chunk: u32,
        hot_gib: Option<f64>,
        ram_gib: Option<f64>,
        dt: KvDtype,
    ) -> KvGeom {
        let bb = page_bytes(dt);
        let flat_nb = u64::from(max_ctx.min(WINDOW_MAX).div_ceil(PAGE_TOKENS) + 1);
        let hot = u64::from(QSA_LAYERS) * 9 * flat_nb * bb;
        let window = ((flat_nb as u32 - 1) * PAGE_TOKENS)
            .saturating_sub(128)
            .max(PAGE_TOKENS);
        let ram = 8 * u64::from(QSA_LAYERS) * u64::from(max_ctx.div_ceil(PAGE_TOKENS)) * bb;
        let gib = |g: f64| ((g * (1u64 << 30) as f64) as u64 / bb) * bb;
        KvGeom {
            max_ctx,
            window,
            chunk: max_chunk.min(max_ctx).max(PAGE_TOKENS),
            hot_bytes: hot_gib.map_or(hot, gib),
            ram_bytes: ram_gib.map_or(ram, gib),
            count_ram: false,
            drop_flush: false,
            gather: true,
            ssd_bytes: 0,
            admit: false,
            first_admit: max_ctx,
            dt,
            layers: QSA_LAYERS,
            mtp: None,
            lazy: false,
        }
    }

    /// FN68d: the MTP layer (tier layer 12) in its OWN one-layer tier at the page dtype `mtp_dt`
    /// (`--fn-kv-mtp-dtype bf16` under fp8 pages: its ~750-nat logits cannot survive K at e4m3,
    /// LEDGER FN68c): this tier keeps the 12 target layers at the same pages a layer (`extra_hot`
    /// = the elastic borrow, which stays this tier's alone), the twin gets one layer's pages at
    /// its own page bytes. `mtp_dt == self.dt` = one tier (unchanged). Call last (after the
    /// admission / drive / elastic sizes).
    #[must_use]
    pub fn split_mtp(mut self, mtp_dt: KvDtype, extra_hot: u64) -> KvGeom {
        if mtp_dt == self.dt || self.mtp.is_some() {
            return self;
        }
        let (bb, l) = (self.bb(), u64::from(self.layers));
        let per = |b: u64| b / bb / l;
        let (ph, pr, ps) = (
            per(self.hot_bytes.saturating_sub(extra_hot)),
            per(self.ram_bytes),
            per(self.ssd_bytes),
        );
        let tb = page_bytes(mtp_dt);
        self.layers -= 1;
        let lm = u64::from(self.layers);
        self.hot_bytes = ph * lm * bb + extra_hot.min(self.hot_bytes);
        self.ram_bytes = pr * lm * bb;
        self.ssd_bytes = ps * lm * bb;
        self.mtp = Some(MtpTwin {
            dt: mtp_dt,
            hot_bytes: ph * tb,
            ram_bytes: pr * tb,
            ssd_bytes: ps * tb,
        });
        self
    }

    /// FN68d: the MTP layer's twin tier's geometry (one layer at its own dtype), if split.
    #[must_use]
    pub fn twin(&self) -> Option<KvGeom> {
        self.mtp.map(|t| KvGeom {
            dt: t.dt,
            hot_bytes: t.hot_bytes,
            ram_bytes: t.ram_bytes,
            ssd_bytes: t.ssd_bytes,
            layers: 1,
            mtp: None,
            ..*self
        })
    }

    /// FN68d: the MTP layer's page dtype (the twin's, else this tier's).
    #[must_use]
    pub fn mtp_dt(&self) -> KvDtype {
        self.mtp.map_or(self.dt, |t| t.dt)
    }

    /// FN68d: K + V bytes a token a rank over every QSA layer (both tiers).
    #[must_use]
    pub fn token_bytes_all(&self) -> u64 {
        u64::from(self.layers) * self.dt.token_layer_bytes()
            + self.mtp.map_or(0, |t| {
                u64::from(QSA_LAYERS - self.layers) * t.dt.token_layer_bytes()
            })
    }

    /// FN68d: `kv.hot` of both tiers (this tier's [`KvGeom::hot_alloc`] + the twin's).
    #[must_use]
    pub fn hot_alloc_all(&self) -> u64 {
        self.hot_alloc() + self.twin().map_or(0, |t| t.hot_alloc())
    }

    /// FN68d: the device page tables of both tiers.
    #[must_use]
    pub fn table_bytes_all(&self) -> u64 {
        self.table_bytes() + self.twin().map_or(0, |t| t.table_bytes())
    }

    /// FN68d: `kv.ram` of both tiers (the pinned slabs).
    #[must_use]
    pub fn ram_alloc_all(&self) -> u64 {
        self.ram_alloc() + self.twin().map_or(0, |t| t.ram_alloc())
    }

    /// FN68: bytes of one 16-token page of one layer at this boot's dtype.
    #[must_use]
    pub fn bb(&self) -> u64 {
        page_bytes(self.dt)
    }

    /// FN51: ADMISSION BY RESERVATION (module [`KvGeom::admit`]): the RAM slab `ram_gib` (absent:
    /// [`admit_slab_bytes`]); `first` = a harness sequence's admission (its prompt + decode).
    #[must_use]
    pub fn with_admission(mut self, ram_gib: Option<f64>, first: u32) -> KvGeom {
        let bb = self.bb();
        let gib = |g: f64| ((g * (1u64 << 30) as f64) as u64 / bb) * bb;
        self.admit = true;
        self.ram_bytes = ram_gib.map_or_else(|| admit_slab_bytes_dt(self.max_ctx, self.dt), gib);
        self.first_admit = first.clamp(PAGE_TOKENS, self.max_ctx);
        self
    }

    /// LR-K (`--fn-kv-admit-slab-tokens`): the admission slab at `tokens` a rank instead of
    /// [`admit_slab_tokens`] (the same law: every QSA layer's blocks at this tier's page dtype;
    /// call after [`KvGeom::with_admission`], before [`KvGeom::split_mtp`]).
    #[must_use]
    pub fn with_admit_slab_tokens(mut self, tokens: u64) -> KvGeom {
        self.ram_bytes = admit_slab_bytes_tokens(tokens, self.dt);
        self
    }

    /// LR-K: the largest admission slab in tokens (a multiple of a page) whose pinned bytes over
    /// both tiers ([`KvGeom::ram_alloc_all`] at this geometry's dtypes) fit `left` bytes.
    #[must_use]
    pub fn admit_slab_tokens_in(&self, left: u64) -> u64 {
        let align = self.bb() + self.twin().map_or(0, |t| t.bb());
        left.saturating_sub(align) / (self.token_bytes_all() * u64::from(PAGE_TOKENS))
            * u64::from(PAGE_TOKENS)
    }

    /// FN51: pages of each QSA layer's paged index store (one 16-token block of compressed rows =
    /// 4 groups x 256 B a page) under admission -- the slab's blocks a layer + 2 a slot (the open
    /// pages past a reservation) -- or 0 (the identity layout: 9 x the max context, today's).
    #[must_use]
    pub fn ix_pages(&self) -> u32 {
        if self.admit {
            (self.ram_bytes / self.bb() / u64::from(self.layers) + 9 * 2) as u32
        } else {
            0
        }
    }

    /// FN51: the sequence's admission at its first step without a reservation (the harness).
    #[must_use]
    pub fn first_tokens(&self, end: u32) -> u32 {
        if self.admit {
            self.first_admit.max(end + ADMIT_EXTRA).min(self.max_ctx)
        } else {
            self.max_ctx
        }
    }

    /// FN45c2: the drive tier (`--fn-kv-ssd-gib`): `ssd_gib` of drive a rank (bounded by
    /// [`SSD_GIB_MAX`]); the RAM slab defaults to today's served slab (8 x
    /// [`DEFAULT_FN_KV_MAX_CTX`] tokens: the pinned-RAM ceiling) unless `ram_gib`, never more than
    /// 8 x the max context.
    pub fn with_ssd(mut self, ssd_gib: f64, ram_gib: Option<f64>) -> anyhow::Result<KvGeom> {
        ensure!(
            ssd_gib > 0.0 && ssd_gib <= SSD_GIB_MAX,
            "--fn-kv-ssd-gib {ssd_gib}: 0 < GiB <= {SSD_GIB_MAX} a rank"
        );
        let bb = self.bb();
        self.ssd_bytes = ((ssd_gib * (1u64 << 30) as f64) as u64 / bb) * bb;
        if ram_gib.is_none() {
            let served =
                8 * u64::from(self.layers) * u64::from(DEFAULT_FN_KV_MAX_CTX / PAGE_TOKENS) * bb;
            self.ram_bytes = self.ram_bytes.min(served);
        }
        Ok(self)
    }

    /// FN45c2: tokens the RAM slab holds for ACTIVE sequences under the resident law (the
    /// scheduler's KV budget: admission by request length), less [`ADMIT_EXTRA`] a slot.
    #[must_use]
    pub fn active_tokens(&self) -> u64 {
        let pages = self.ram_bytes / self.bb() / u64::from(self.layers);
        (pages * u64::from(PAGE_TOKENS)).saturating_sub(9 * u64::from(ADMIT_EXTRA))
    }

    /// The tier plan (refuses a RAM budget under 8 x max context, a hot pool under the floor).
    pub fn plan(&self) -> anyhow::Result<KvTierPlan> {
        if self.admit {
            return KvTierPlan::new_admitted_dt(
                self.layers,
                PAGE_TOKENS,
                self.window,
                self.chunk,
                8,
                self.max_ctx,
                self.hot_bytes,
                self.ram_bytes,
                self.dt,
            )
            .map_err(|e| {
                anyhow::anyhow!(
                    "--fn-kv-tier: {e}; under admission by reservation the RAM slab holds one \
                     {}-token sequence at least",
                    self.max_ctx
                )
            });
        }
        KvTierPlan::new_dt(
            self.layers,
            PAGE_TOKENS,
            self.window,
            self.chunk,
            8,
            self.max_ctx,
            self.hot_bytes,
            self.ram_bytes,
            self.ssd_bytes,
            self.dt,
        )
        .map_err(|e| anyhow::anyhow!("--fn-kv-tier: {e}"))
    }

    /// VRAM the tier allocates: the hot slab + the pad page + one page of alignment (`kv.hot`).
    #[must_use]
    pub fn hot_alloc(&self) -> u64 {
        (self.hot_bytes / self.bb() + 2) * self.bb()
    }

    /// The device page table's bytes (`kv.index`'s table part).
    #[must_use]
    pub fn table_bytes(&self) -> u64 {
        u64::from(self.layers) * 9 * u64::from(self.max_ctx.div_ceil(PAGE_TOKENS) + 1) * 4
    }

    /// Pinned RAM (`kv.ram`): the slab + one page of alignment.
    #[must_use]
    pub fn ram_alloc(&self) -> u64 {
        (self.ram_bytes / self.bb() + 1) * self.bb()
    }
}

/// The step's which-path counters (`fnkv` lines).
#[derive(Debug, Clone, Copy, Default)]
pub struct KvStats {
    pub steps: u64,
    pub flush_runs: u64,
    pub flush_blocks: u64,
    pub upload_jobs: u64,
    pub upload_bytes: u64,
    /// FN45b3: live RAM-placed entries summed over steps / steps with any (`count_ram`).
    pub ram_live: u64,
    pub ram_steps: u64,
    /// FN45b4: prefill gathers -- H2D runs, blocks copied, blocks left in RAM (UVA).
    pub gather_runs: u64,
    pub gather_blocks: u64,
    pub gather_uva: u64,
    /// FN45c2: drive write-through -- blocks, runs, host ms.
    pub ssd_blocks: u64,
    pub ssd_runs: u64,
    pub ssd_ms: f64,
}

/// The tier on one rank: the bookkeeping, the page table and its device copy, the slabs.
pub struct KvDev {
    pub geom: KvGeom,
    pub tier: KvTier,
    pub table: KvPageTable,
    /// FN51f-b: shared with the copy queue (the elastic region registers its segments in it).
    hot: std::rc::Rc<DevicePtr>,
    hot_off: u64,
    ram: PinnedBuf,
    ram_off: u64,
    pub tdev: DevicePtr,
    mirror: PinnedBuf,
    pub stats: KvStats,
    /// FN45c2: the drive slab (`--fn-kv-ssd-gib`) and the last step's drive writes (issued at
    /// the next step's begin, after the flushes they copy have landed).
    ssd: Option<KvSsd>,
    pending: Vec<(u32, u32)>,
    /// FN51: the paged index store's allocator (admission by reservation; None = identity).
    pub ix: Option<IxPool>,
    /// FN68: bytes of one page of one layer at this boot's dtype (`geom.bb()`).
    pub bb: u64,
    /// FN68d: the MTP layer's one-layer twin tier at its own page dtype (`--fn-kv-mtp-dtype`;
    /// [`KvGeom::split_mtp`]): every sequence call reaches both; None = one tier.
    pub mtp: Option<Box<KvDev>>,
}

/// FN51: the PAGED INDEX STORE. Every QSA layer's compressed keys (`kv.index`: 4 groups x 256 B a
/// 16-token block) are addressed by the store and the score kernels through the step's block
/// table `bt[seq * nb + j]` -- today the identity (`slot * nb + j`, 9 x the max context of VRAM a
/// layer: 1.96 GiB a rank at 262,144). Under admission the table rows come from this pool instead
/// (one page numbering shared by the 13 layers, each layer's store `pages + 1` pages; page
/// `pages` = the PAD page every unallocated entry and the spare slot's row point at), pages
/// taken as a sequence grows ([`IxPool::ensure`]) and returned when it leaves -- no kernel
/// change: the kernels read the same bytes at other addresses. No page is read before the store
/// wrote it (a row's score reads only groups complete at or before it, stored first).
#[derive(Debug, Clone)]
pub struct IxPool {
    pub pages: u32,
    pub nbs: u32,
    free: Vec<u32>,
    rows: Vec<Vec<i32>>,
}

impl IxPool {
    /// `pages` usable pages, `slots` table rows of `nbs` entries.
    #[must_use]
    pub fn new(pages: u32, slots: u32, nbs: u32) -> IxPool {
        IxPool {
            pages,
            nbs,
            free: (0..pages).rev().collect(),
            rows: vec![vec![pages as i32; nbs as usize]; slots as usize],
        }
    }

    /// The pad page (every unallocated entry).
    #[must_use]
    pub fn pad(&self) -> i32 {
        self.pages as i32
    }

    /// Slot `slot`'s row.
    #[must_use]
    pub fn row(&self, slot: u32) -> &[i32] {
        &self.rows[slot as usize]
    }

    /// Pages held by `slot`.
    #[must_use]
    pub fn held(&self, slot: u32) -> u32 {
        self.rows[slot as usize]
            .iter()
            .take_while(|&&p| p != self.pad())
            .count() as u32
    }

    /// Pages free (tests).
    #[cfg(test)]
    #[must_use]
    pub fn free(&self) -> u32 {
        self.free.len() as u32
    }

    /// Give `slot` its first `blocks` entries (lowest free page first: a deterministic layout).
    pub fn ensure(&mut self, slot: u32, blocks: u32) -> anyhow::Result<()> {
        let blocks = blocks.min(self.nbs);
        let have = self.held(slot);
        for j in have..blocks {
            let p = self.free.pop().with_context(|| {
                format!(
                    "kv tier: the paged index store is exhausted ({} pages) at slot {slot} block \
                     {j} (admission reserves the slab's blocks: it cannot run out)",
                    self.pages
                )
            })?;
            self.rows[slot as usize][j as usize] = p as i32;
        }
        Ok(())
    }

    /// `slot` left: its pages return (kept sorted so the next take is the lowest).
    pub fn release(&mut self, slot: u32) {
        let pad = self.pad();
        let mut back: Vec<u32> = Vec::new();
        for e in &mut self.rows[slot as usize] {
            if *e == pad {
                break;
            }
            back.push(*e as u32);
            *e = pad;
        }
        if !back.is_empty() {
            self.free.extend(back);
            self.free.sort_unstable_by(|a, b| b.cmp(a));
        }
    }

    /// The compaction / swap: `from`'s row moves to `to` (whose pages return first).
    pub fn move_row(&mut self, from: u32, to: u32) {
        if from == to {
            return;
        }
        self.release(to);
        let pads = vec![self.pad(); self.nbs as usize];
        let r = std::mem::replace(&mut self.rows[from as usize], pads);
        self.rows[to as usize] = r;
    }

    /// Every page is either free or held exactly once, each row a held prefix then pads (tests).
    #[cfg(test)]
    pub fn check(&self) -> Result<(), String> {
        let mut seen = vec![false; self.pages as usize];
        for &p in &self.free {
            if std::mem::replace(&mut seen[p as usize], true) {
                return Err(format!("page {p} free twice"));
            }
        }
        for (s, r) in self.rows.iter().enumerate() {
            let h = self.held(s as u32) as usize;
            if r[h..].iter().any(|&p| p != self.pad()) {
                return Err(format!("slot {s}: a page past its prefix"));
            }
            for &p in &r[..h] {
                if std::mem::replace(&mut seen[p as usize], true) {
                    return Err(format!("page {p} held twice"));
                }
            }
        }
        if seen.iter().all(|&x| x) {
            Ok(())
        } else {
            Err("a page lost".into())
        }
    }
}

/// The per-layer append slots and (width > 1) per-row tables of `rows` = `(pool slot, position)`.
#[must_use]
#[cfg(test)]
pub fn step_tables(t: &KvPageTable, rows: &[(u32, u32)], per_row: bool) -> (Vec<u8>, Vec<u8>) {
    step_tables_of(&[t], rows, per_row)
}

/// FN68d: [`step_tables`] over the tiers' tables in tier-layer order (the main tier's 12 target
/// layers, then the MTP twin's one): the step constants keep the `[13, m]` layout whichever tier
/// holds a layer (each entry is a page index from ITS tier's base).
#[must_use]
pub fn step_tables_of(
    tables: &[&KvPageTable],
    rows: &[(u32, u32)],
    per_row: bool,
) -> (Vec<u8>, Vec<u8>) {
    let mut ks = Vec::with_capacity(QSA_LAYERS as usize * rows.len() * 8);
    let mut br = Vec::new();
    for t in tables {
        for l in 0..t.layers() {
            for &(s, p) in rows {
                ks.extend_from_slice(&t.append_slot(l, s, p).to_le_bytes());
            }
            if per_row {
                let pad = t.addr().pad as i32;
                for &(s, _) in rows {
                    let row = t.row(l, s);
                    for j in 0..KBTR as usize {
                        br.extend_from_slice(&row.get(j).copied().unwrap_or(pad).to_le_bytes());
                    }
                }
            }
        }
    }
    (ks, br)
}

/// One step's host plan ([`plan_step`]).
#[derive(Debug, Default)]
pub struct StepPlan {
    /// The commits' D2H flushes, in issue order (first on the stream).
    pub flush: Vec<CopyOp>,
    /// FN45b4: a prefill chunk's gather H2Ds (after the flushes, before the chunk's kernels).
    pub gather: Vec<CopyOp>,
    /// FN45b4: earlier blocks the gather found no hot slot for (read in place from RAM).
    pub uva: u64,
    /// The rewritten `(layer, slot)` table rows.
    pub rows: Vec<(u32, u32)>,
    /// FN45c2: the commits' drive writes `(RAM slot, ssd slot)` -- issued only after this
    /// step's flushes landed (the next step's begin).
    pub ssd_writes: Vec<(u32, u32)>,
}

/// The host half of the step law ([`KvDev::begin`]): admit, move every final length, commit
/// (the flushes, in issue order), open the step's pages, sync the table. FN45b4: a PREFILL step
/// (`prefill`: a prompt chunk or the drafter's prefill-form piece) PROTECTS its sequence (its
/// blocks are every other sequence's victims' last resort, never a decode gather's) and GATHERS
/// every earlier block of it that is not hot back into VRAM (`KvTier::gather_seq`: a chunk's sparse
/// rows may select any earlier group, and the selection is not on the host before the chunk); a
/// decode step releases the protection of every sequence in it (its prompt is complete).
pub fn plan_step(
    tier: &mut KvTier,
    table: &mut KvPageTable,
    geom: &KvGeom,
    seqs: &[(u32, u32, u32)],
    prefill: bool,
) -> anyhow::Result<StepPlan> {
    let e = |x: saber_mem::kv_tier::KvTierError| anyhow::anyhow!("kv tier: {x}");
    let (max_ctx, chunk) = (geom.max_ctx, geom.chunk);
    let mut out = StepPlan::default();
    for &(s, p0, pend) in seqs {
        ensure!(
            pend <= max_ctx + PAGE_TOKENS,
            "kv tier: slot {s} writes position {} past the {max_ctx}-token context",
            pend - 1
        );
        if !tier.is_live(s) {
            // FN51: under admission a harness sequence takes its prompt + decode, not the context
            tier.admit(s, geom.first_tokens(pend)).map_err(e)?;
        }
        tier.protect(s, prefill).map_err(e)?;
        let fin = p0.saturating_sub(SLACK);
        let mut len = tier.len(s);
        if fin > len {
            while len < fin {
                let n = (fin - len).min(chunk);
                tier.append(s, n).map_err(e)?;
                len += n;
            }
            let c = tier.commit(s).map_err(e)?;
            out.flush.extend(c.flush);
            out.ssd_writes.extend(c.ssd_writes);
        }
        tier.open(s, pend + AHEAD).map_err(e)?;
    }
    if prefill && geom.gather {
        for &(s, _, _) in seqs {
            let g = tier.gather_seq(s).map_err(e)?;
            out.uva += g.uva.len() as u64;
            out.gather.extend(g.ops);
        }
    }
    out.rows = table.sync(tier).map_err(e)?;
    Ok(out)
}

impl KvDev {
    /// Allocate the slabs, check the address law, upload the all-pad table.
    pub fn new(
        hip: &HipThread,
        stream: &Stream,
        geom: KvGeom,
        ssd_path: Option<&Path>,
    ) -> anyhow::Result<KvDev> {
        let plan = geom.plan()?;
        let bb = geom.bb();
        ensure!(plan.block_bytes() == bb, "the tier plan's page bytes");
        ensure!(
            geom.mtp.is_none() || geom.ssd_bytes == 0,
            "--fn-kv-mtp-dtype: the MTP layer's twin tier runs on the two tiers only (no drive)"
        );
        ensure!(
            geom.layers + u32::from(geom.mtp.is_some()) == QSA_LAYERS || geom.layers == 1,
            "the tiers hold {} + {} of the {QSA_LAYERS} QSA layers",
            geom.layers,
            u32::from(geom.mtp.is_some())
        );
        let mtp = match geom.twin() {
            Some(t) => {
                let mut d = KvDev::new(hip, stream, t, None)?;
                // the paged index store is one pool for every layer: the main tier's
                d.ix = None;
                Some(Box::new(d))
            }
            None => None,
        };
        let ssd = match (plan.has_ssd(), ssd_path) {
            (false, _) => None,
            (true, Some(p)) => Some(
                KvSsd::create(p, plan.ssd_slots, bb, OpenMode::Direct)
                    .with_context(|| format!("--fn-kv-ssd-gib: the drive slab {}", p.display()))?,
            ),
            (true, None) => anyhow::bail!("--fn-kv-ssd-gib needs a drive path"),
        };
        let tier = if ssd.is_some() {
            KvTier::new(plan.clone())
                .resident()
                .map_err(|e| anyhow::anyhow!("{e}"))?
        } else {
            KvTier::new(plan.clone())
        };
        // LR-G: lazy growth (both tiers: the twin is built from this geometry)
        let tier = if geom.lazy {
            ensure!(
                geom.admit,
                "--fn-kv-lazy runs under admission by reservation only"
            );
            tier.lazy().map_err(|e| anyhow::anyhow!("{e}"))?
        } else {
            tier
        };
        let hot = std::rc::Rc::new(hip.malloc(geom.hot_alloc())?);
        let ha = hot.addr().as_u64();
        let hot_off = ha.div_ceil(bb) * bb - ha;
        let ram = hip.host_malloc(geom.ram_alloc())?;
        let rd = hip.host_device_addr(&ram)?;
        let ram_off = rd.div_ceil(bb) * bb - rd;
        let pad = ha + hot_off + u64::from(plan.hot_slots) * bb;
        let addr = KvAddr::new(
            bb,
            ha + hot_off,
            plan.hot_slots,
            rd + ram_off,
            plan.ram_slots,
            pad,
        )
        .map_err(|e| {
            anyhow::anyhow!(
                "--fn-kv-tier: the address law refuses this boot (hot 0x{ha:x}, RAM device \
                 0x{rd:x}): {e}"
            )
        })?;
        let table = KvPageTable::new(&plan, 9, addr).map_err(|e| anyhow::anyhow!("{e}"))?;
        let nbs = table.nbs();
        ensure!(table.bytes() == geom.table_bytes(), "the table's bytes");
        let tdev = hip.malloc(table.bytes())?;
        let mut mirror = hip.host_malloc(table.bytes())?;
        mirror.as_mut_slice().copy_from_slice(&table.le_bytes());
        hip.run_batch(&[Job::MemcpyPinned {
            dev: &tdev,
            dev_offset: 0,
            host: &mirror,
            host_offset: 0,
            bytes: table.bytes(),
            kind: MemcpyKind::HostToDevice,
            stream,
        }])?;
        hip.sync_stream(stream)?;
        Ok(KvDev {
            geom,
            tier,
            ssd,
            pending: Vec::new(),
            table,
            hot,
            hot_off,
            ram,
            ram_off,
            tdev,
            mirror,
            stats: KvStats::default(),
            ix: (geom.ix_pages() > 0).then(|| IxPool::new(geom.ix_pages(), 9, nbs)),
            bb,
            mtp,
        })
    }

    /// FN51f-b: the tier's counters on `/metrics` (`saber_fn_kv_*`): steps, the live RAM-placed
    /// entries summed over steps (`--fn-kv-pages` only: the kernels' UVA reads' ceiling), the
    /// prefill gather's blocks copied / left in RAM, the flushes, the hot cap and pool.
    #[must_use]
    pub fn render_prometheus(&self, rank: u32) -> String {
        let s = &self.stats;
        let mut o = String::from("# TYPE saber_fn_kv_total counter\n");
        for (k, v) in [
            ("steps", s.steps),
            ("ram_entries", s.ram_live),
            ("ram_steps", s.ram_steps),
            ("gather_blocks", s.gather_blocks),
            ("gather_uva", s.gather_uva),
            ("flush_blocks", s.flush_blocks),
        ] {
            o.push_str(&format!(
                "saber_fn_kv_total{{rank=\"{rank}\",kind=\"{k}\"}} {v}\n"
            ));
        }
        o.push_str("# TYPE saber_fn_kv_hot_pages gauge\n");
        for (k, v) in [
            ("cap", self.tier.hot_cap()),
            ("pool", self.tier.plan().hot_slots),
        ] {
            o.push_str(&format!(
                "saber_fn_kv_hot_pages{{rank=\"{rank}\",kind=\"{k}\"}} {v}\n"
            ));
        }
        o
    }

    /// FN51f-b: the hot pool's allocation and the byte offset of its page 0 (the elastic region:
    /// segment `i`'s expert slots live at `off + geom.slot_offset(i, k, bb)` of it).
    #[must_use]
    pub fn hot_region(&self) -> (std::rc::Rc<DevicePtr>, u64) {
        (std::rc::Rc::clone(&self.hot), self.hot_off)
    }

    /// The kernels' `kv` base (every QSA layer's).
    #[must_use]
    pub fn base(&self) -> u64 {
        self.table.addr().base
    }

    /// Bytes of one table row (`nbs * 4`).
    #[must_use]
    pub fn row_bytes(&self) -> u64 {
        u64::from(self.table.nbs()) * 4
    }

    /// Bytes of one layer's `[slots, nbs]` table.
    #[must_use]
    pub fn layer_bytes(&self) -> u64 {
        self.table.layer_offset(1)
    }

    /// A sequence starts at pool slot `slot` (the pools' zero): its old occupant is reclaimed.
    /// FN45c2: under the drive tier it is ADMITTED here by its request length (`tokens` = the
    /// scheduler's reservation, + [`ADMIT_EXTRA`]) -- the resident law reserves its RAM; with
    /// `tokens` 0 (the harness) or without the drive its first step admits it at the max context
    /// (FN45b2).
    pub fn start(&mut self, slot: u32, tokens: u32) -> anyhow::Result<()> {
        if let Some(t) = self.mtp.as_mut() {
            t.start(slot, tokens)?;
        }
        self.release_own(slot)?;
        if (self.ssd.is_some() || self.geom.admit) && tokens > 0 {
            let t = (tokens + ADMIT_EXTRA).min(self.geom.max_ctx);
            self.tier.admit(slot, t).map_err(|e| {
                anyhow::anyhow!(
                    "kv tier: admission of slot {slot} at {t} tokens refused ({e}); the \
                     scheduler's KV budget is the RAM slab's active tokens"
                )
            })?;
        }
        Ok(())
    }

    /// FN45c2: the sequence in `slot` left (its reservation returns at once: the scheduler has
    /// already released its KV).
    pub fn release(&mut self, slot: u32) -> anyhow::Result<()> {
        if let Some(t) = self.mtp.as_mut() {
            t.release(slot)?;
        }
        self.release_own(slot)
    }

    fn release_own(&mut self, slot: u32) -> anyhow::Result<()> {
        if let Some(ix) = self.ix.as_mut() {
            ix.release(slot);
        }
        if self.tier.is_live(slot) {
            self.tier.finish(slot).map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        Ok(())
    }

    /// FN45c2: the last step's drive writes (its flushes landed: the stream is synchronized).
    pub fn drain(&mut self, hip: &HipThread, stream: &Stream) -> anyhow::Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let ssd = self.ssd.as_ref().context("drive writes without a drive")?;
        hip.sync_stream(stream)?;
        let t0 = std::time::Instant::now();
        let runs = ssd
            .write(self.ram.as_slice(), self.ram_off, &self.pending)
            .context("kv.ssd write")?;
        self.stats.ssd_blocks += self.pending.len() as u64;
        self.stats.ssd_runs += runs as u64;
        self.stats.ssd_ms += t0.elapsed().as_secs_f64() * 1e3;
        self.pending.clear();
        Ok(())
    }

    /// FN45c2 (the prompt cache's snapshot): the bytes of `slot`'s blocks `b0..b1` of every QSA
    /// layer, layer-major, into `buf` at `at` (VRAM pages D2H in one batch, RAM pages copied).
    /// Synchronizes `stream`. Returns the bytes written.
    pub fn dump(
        &self,
        hip: &HipThread,
        stream: &Stream,
        slot: u32,
        (b0, b1): (u32, u32),
        buf: &mut PinnedBuf,
        at: usize,
    ) -> anyhow::Result<usize> {
        hip.sync_stream(stream)?;
        let n = (b1 - b0) as usize * self.geom.layers as usize * self.bb as usize;
        ensure!(at + n <= buf.len(), "the snapshot buffer");
        let mut ram = Vec::new();
        let mut d2h = Vec::new();
        for l in 0..self.geom.layers {
            for b in b0..b1 {
                let o = at + ((l * (b1 - b0) + b - b0) as usize) * self.bb as usize;
                match self
                    .tier
                    .place(slot, l, b)
                    .map_err(|x| anyhow::anyhow!("{x}"))?
                {
                    Place::Vram(v) => d2h.push((self.hot_off + u64::from(v) * self.bb, o as u64)),
                    Place::Ram(r) => {
                        ram.push(((self.ram_off + u64::from(r) * self.bb) as usize, o))
                    }
                    Place::Ssd(_) => anyhow::bail!("snapshot of a parked block"),
                }
            }
        }
        {
            let dst = buf.as_mut_slice();
            for &(r, o) in &ram {
                dst[o..o + self.bb as usize]
                    .copy_from_slice(&self.ram.as_slice()[r..r + self.bb as usize]);
            }
        }
        let jobs: Vec<Job<'_>> = d2h
            .iter()
            .map(|&(d, o)| Job::MemcpyPinned {
                dev: &self.hot,
                dev_offset: d,
                host: buf,
                host_offset: o,
                bytes: self.bb,
                kind: MemcpyKind::DeviceToHost,
                stream,
            })
            .collect();
        hip.run_batch(&jobs)?;
        hip.sync_stream(stream)?;
        // FN68d: the MTP twin's layer after the main tier's (layer-major over the 13)
        let t = match self.mtp.as_ref() {
            Some(t) => t.dump(hip, stream, slot, (b0, b1), buf, at + n)?,
            None => 0,
        };
        Ok(n + t)
    }

    /// FN113b (the END node's store off the step): THIS tier's part of [`KvDev::dump`] (not the
    /// MTP twin's: the caller issues it apart, at `at` + this call's bytes) -- the RAM pages
    /// copied now, the VRAM pages' D2H issued on `stream` WITHOUT a synchronization. The caller
    /// has drained every writer of these pages and fences the copies before the next one.
    pub fn dump_async(
        &self,
        hip: &HipThread,
        stream: &Stream,
        slot: u32,
        (b0, b1): (u32, u32),
        buf: &mut PinnedBuf,
        at: usize,
    ) -> anyhow::Result<usize> {
        let n = (b1 - b0) as usize * self.geom.layers as usize * self.bb as usize;
        ensure!(at + n <= buf.len(), "the snapshot buffer");
        let mut d2h = Vec::new();
        for l in 0..self.geom.layers {
            for b in b0..b1 {
                let o = at + ((l * (b1 - b0) + b - b0) as usize) * self.bb as usize;
                match self
                    .tier
                    .place(slot, l, b)
                    .map_err(|x| anyhow::anyhow!("{x}"))?
                {
                    Place::Vram(v) => {
                        let d = self.hot_off + u64::from(v) * self.bb;
                        // COALESCED: a page right after the previous one on the card and in
                        // the buffer extends its copy
                        match d2h.last_mut() {
                            Some((pd, po, pn)) if *pd + *pn == d && *po + *pn == o as u64 => {
                                *pn += self.bb;
                            }
                            _ => d2h.push((d, o as u64, self.bb)),
                        }
                    }
                    Place::Ram(r) => {
                        let r = (self.ram_off + u64::from(r) * self.bb) as usize;
                        buf.as_mut_slice()[o..o + self.bb as usize]
                            .copy_from_slice(&self.ram.as_slice()[r..r + self.bb as usize]);
                    }
                    Place::Ssd(_) => anyhow::bail!("snapshot of a parked block"),
                }
            }
        }
        let jobs: Vec<Job<'_>> = d2h
            .iter()
            .map(|&(d, o, n)| Job::MemcpyPinned {
                dev: &self.hot,
                dev_offset: d,
                host: buf,
                host_offset: o,
                bytes: n,
                kind: MemcpyKind::DeviceToHost,
                stream,
            })
            .collect();
        hip.run_batch(&jobs)?;
        Ok(n)
    }

    /// FN45c2 (the prompt cache's restore): write `dump`'s bytes back into `slot`'s blocks
    /// `b0..b1` wherever the table places them now (after the step's `begin` opened them).
    pub fn load(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        slot: u32,
        (b0, b1): (u32, u32),
        buf: &PinnedBuf,
        at: usize,
    ) -> anyhow::Result<usize> {
        hip.sync_stream(stream)?;
        let n = (b1 - b0) as usize * self.geom.layers as usize * self.bb as usize;
        ensure!(at + n <= buf.len(), "the snapshot buffer");
        let mut h2d = Vec::new();
        for l in 0..self.geom.layers {
            for b in b0..b1 {
                let o = at + ((l * (b1 - b0) + b - b0) as usize) * self.bb as usize;
                match self
                    .tier
                    .place(slot, l, b)
                    .map_err(|x| anyhow::anyhow!("{x}"))?
                {
                    Place::Vram(v) => h2d.push((self.hot_off + u64::from(v) * self.bb, o as u64)),
                    Place::Ram(r) => {
                        let r = (self.ram_off + u64::from(r) * self.bb) as usize;
                        self.ram.as_mut_slice()[r..r + self.bb as usize]
                            .copy_from_slice(&buf.as_slice()[o..o + self.bb as usize]);
                    }
                    Place::Ssd(_) => anyhow::bail!("restore into a parked block"),
                }
            }
        }
        let jobs: Vec<Job<'_>> = h2d
            .iter()
            .map(|&(d, o)| Job::MemcpyPinned {
                dev: &self.hot,
                dev_offset: d,
                host: buf,
                host_offset: o,
                bytes: self.bb,
                kind: MemcpyKind::HostToDevice,
                stream,
            })
            .collect();
        hip.run_batch(&jobs)?;
        hip.sync_stream(stream)?;
        let t = match self.mtp.as_mut() {
            Some(t) => t.load(hip, stream, slot, (b0, b1), buf, at + n)?,
            None => 0,
        };
        Ok(n + t)
    }

    /// FN68d: bytes of one page of tier layer `l` (0..13): the main tier's, or the MTP twin's
    /// for the layers past it.
    #[must_use]
    pub fn layer_bb(&self, l: u32) -> u64 {
        match self.mtp.as_ref() {
            Some(t) if l >= self.geom.layers => t.layer_bb(l - self.geom.layers),
            _ => self.bb,
        }
    }

    /// FN68d: bytes of one block of every QSA layer (a snapshot's block unit, both tiers).
    #[must_use]
    pub fn block_set_bytes(&self) -> u64 {
        u64::from(self.geom.layers) * self.bb + self.mtp.as_ref().map_or(0, |t| t.block_set_bytes())
    }

    /// FN68d: the tiers' page tables in tier-layer order ([`step_tables_of`]).
    #[must_use]
    pub fn tables(&self) -> Vec<&KvPageTable> {
        let mut v = vec![&self.table];
        if let Some(t) = self.mtp.as_ref() {
            v.extend(t.tables());
        }
        v
    }

    /// The pools' compaction / swap.
    pub fn move_seq(&mut self, from: u32, to: u32) -> anyhow::Result<()> {
        if let Some(t) = self.mtp.as_mut() {
            t.move_seq(from, to)?;
        }
        if let Some(ix) = self.ix.as_mut() {
            ix.move_row(from, to);
        }
        if !self.tier.is_live(from) {
            if self.tier.is_live(to) {
                self.tier.finish(to).map_err(|e| anyhow::anyhow!("{e}"))?;
            }
            return Ok(());
        }
        self.tier
            .move_seq(from, to)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// FN130: the hot pages the step law's opens over `seqs` (`(pool slot, first position, end
    /// position)`, as [`Self::begin`]) need past what the main tier's pool gives now (the elastic
    /// cap): every sequence holds `ceil(min(end + AHEAD, its reservation) / page)` blocks a layer
    /// after the step (`plan_step`'s `open`; a sequence not live is admitted at
    /// [`KvGeom::first_tokens`]). Pure (no tier state moves). The MTP twin is never lent.
    #[must_use]
    pub fn shortfall(&self, seqs: &[(u32, u32, u32)]) -> u64 {
        let opens: Vec<(u32, u32)> = seqs
            .iter()
            .map(|&(s, _, pend)| {
                let res = self
                    .tier
                    .max_tokens(s)
                    .unwrap_or_else(|| self.geom.first_tokens(pend));
                (
                    s,
                    (pend + AHEAD)
                        .min(res)
                        .div_ceil(self.tier.plan().block_tokens),
                )
            })
            .collect();
        self.tier.open_shortfall(&opens)
    }

    /// The step law (module doc): `seqs` = `(pool slot, first position, end position)` of every
    /// live sequence of the step. Issues the flushes and the table upload on `stream`.
    pub fn begin(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        seqs: &[(u32, u32, u32)],
        prefill: bool,
    ) -> anyhow::Result<()> {
        if let Some(t) = self.mtp.as_mut() {
            t.begin(hip, stream, seqs, prefill)?;
        }
        self.drain(hip, stream)?;
        if let Some(ix) = self.ix.as_mut() {
            // FN51: the index pages every row of the step may store or read (+ the open pages)
            for &(s, _, pend) in seqs {
                ix.ensure(s, (pend + AHEAD).div_ceil(PAGE_TOKENS))?;
            }
        }
        let StepPlan {
            flush,
            gather,
            uva,
            rows,
            ssd_writes,
        } = plan_step(&mut self.tier, &mut self.table, &self.geom, seqs, prefill)?;
        if self.ssd.is_some() && !self.geom.drop_flush {
            self.pending = ssd_writes;
        }
        // the changed entries of each rewritten row: [first, last] differing i32
        let mut ups: Vec<(u64, u64)> = Vec::new();
        let rb = self.row_bytes() as usize;
        for &(l, s) in &rows {
            let o = self.table.layer_offset(l) as usize + s as usize * rb;
            let new = self.table.row(l, s);
            let m = &mut self.mirror.as_mut_slice()[o..o + rb];
            let mut lo = usize::MAX;
            let mut hi = 0;
            for (i, v) in new.iter().enumerate() {
                let b = v.to_le_bytes();
                if m[i * 4..i * 4 + 4] != b {
                    m[i * 4..i * 4 + 4].copy_from_slice(&b);
                    lo = lo.min(i);
                    hi = i;
                }
            }
            if lo != usize::MAX {
                ups.push(((o + lo * 4) as u64, ((hi - lo + 1) * 4) as u64));
            }
        }
        let runs = if self.geom.drop_flush {
            Vec::new()
        } else {
            coalesce(&flush)
        };
        if self.geom.count_ram {
            let mut n = 0u64;
            for &(s, _, _) in seqs {
                for l in 0..self.geom.layers {
                    for b in 0..self.tier.blocks_of(s, l) {
                        n += u64::from(matches!(self.tier.place(s, l, b), Ok(Place::Ram(_))));
                    }
                }
            }
            self.stats.ram_live += n;
            self.stats.ram_steps += u64::from(n > 0);
        }
        // FN45b4: the gather's H2Ds after the flushes (K1: a block's RAM copy is written before
        // it is read back; an evicted slot's flush precedes its reuse) and before the chunk's
        // kernels (the same in-order stream: the copies land before any attention reads them)
        let gruns = coalesce(&gather);
        let mut jobs = Vec::with_capacity(runs.len() + gruns.len() + ups.len());
        ensure!(
            runs.iter().all(|r| !r.h2d) && gruns.iter().all(|r| r.h2d),
            "a commit issues flushes only, a gather H2Ds only"
        );
        for r in runs.iter().chain(&gruns) {
            jobs.push(Job::MemcpyPinned {
                dev: &self.hot,
                dev_offset: self.hot_off + u64::from(r.vram) * self.bb,
                host: &self.ram,
                host_offset: self.ram_off + u64::from(r.ram) * self.bb,
                bytes: u64::from(r.n) * self.bb,
                kind: if r.h2d {
                    MemcpyKind::HostToDevice
                } else {
                    MemcpyKind::DeviceToHost
                },
                stream,
            });
        }
        for &(o, n) in &ups {
            jobs.push(Job::MemcpyPinned {
                dev: &self.tdev,
                dev_offset: o,
                host: &self.mirror,
                host_offset: o,
                bytes: n,
                kind: MemcpyKind::HostToDevice,
                stream,
            });
        }
        hip.run_batch(&jobs)?;
        let st = &mut self.stats;
        st.steps += 1;
        st.flush_runs += runs.len() as u64;
        st.flush_blocks += runs.iter().map(|r| u64::from(r.n)).sum::<u64>();
        st.upload_jobs += ups.len() as u64;
        st.gather_runs += gruns.len() as u64;
        st.gather_blocks += gruns.iter().map(|r| u64::from(r.n)).sum::<u64>();
        st.gather_uva += uva;
        st.upload_bytes += ups.iter().map(|x| x.1).sum::<u64>();
        Ok(())
    }

    /// The bytes of `(slot, layer)`'s first `blocks` pages in logical order (VRAM pages read back,
    /// RAM pages from the slab) -- the page byte law's dump. Synchronizes `stream`.
    pub fn read_pages(
        &self,
        hip: &HipThread,
        stream: &Stream,
        slot: u32,
        layer: u32,
        blocks: u32,
    ) -> anyhow::Result<Vec<u8>> {
        if let Some(t) = self.mtp.as_ref().filter(|_| layer >= self.geom.layers) {
            return t.read_pages(hip, stream, slot, layer - self.geom.layers, blocks);
        }
        hip.sync_stream(stream)?;
        let mut out = Vec::with_capacity(blocks as usize * self.bb as usize);
        let mut tmp = hip.host_malloc(self.bb)?;
        for b in 0..blocks {
            match self
                .tier
                .place(slot, layer, b)
                .map_err(|x| anyhow::anyhow!("{x}"))?
            {
                Place::Vram(v) => {
                    hip.run_batch(&[Job::MemcpyPinned {
                        dev: &self.hot,
                        dev_offset: self.hot_off + u64::from(v) * self.bb,
                        host: &tmp,
                        host_offset: 0,
                        bytes: self.bb,
                        kind: MemcpyKind::DeviceToHost,
                        stream,
                    }])?;
                    hip.sync_stream(stream)?;
                    out.extend_from_slice(tmp.as_mut_slice());
                }
                Place::Ram(r) => {
                    let o = (self.ram_off + u64::from(r) * self.bb) as usize;
                    out.extend_from_slice(&self.ram.as_slice()[o..o + self.bb as usize]);
                }
                Place::Ssd(_) => anyhow::bail!("an SSD block with no SSD tier"),
            }
        }
        Ok(out)
    }

    /// The boot line.
    #[must_use]
    pub fn boot_line(&self, rank: u32) -> String {
        let own = self.boot_line_own(rank);
        match self.mtp.as_ref() {
            // FN68d: the MTP layer's twin tier on its own line
            Some(t) => format!(
                "{own} [{} layers]\n{}",
                self.geom.layers,
                t.boot_line_own(rank).replace(
                    &format!("fnkv rank{rank}:"),
                    &format!("fnkv rank{rank} MTP layer:")
                )
            ),
            None => own,
        }
    }

    fn boot_line_own(&self, rank: u32) -> String {
        let p = self.tier.plan();
        let a = self.table.addr();
        format!(
            "fnkv rank{rank}: TIERED KV on ({} pages, {} B a page): {} hot pages ({:.3} GiB, window {} tokens, quota {}) + \
             {} pinned RAM pages ({:.2} GiB) for 8 x {} tokens; base 0x{:x}, hot @{} ram @{} pad \
             @{}; table {:.2} MiB ({} entries a row)",
            self.geom.dt.name(),
            self.bb,
            p.hot_slots,
            p.hot_bytes() as f64 / (1u64 << 30) as f64,
            p.window_tokens,
            p.quota,
            p.ram_slots,
            p.ram_bytes() as f64 / (1u64 << 30) as f64,
            p.max_ctx,
            a.base,
            a.hot,
            a.ram,
            a.pad,
            self.table.bytes() as f64 / (1u64 << 20) as f64,
            self.table.nbs()
        )
    }

    /// Live (layer, block) entries read from RAM now (the UVA overflow ceiling of a step).
    #[must_use]
    pub fn ram_entries(&self) -> u64 {
        let mut n = 0u64;
        for s in 0..=8 {
            for l in 0..self.geom.layers {
                for b in 0..self.tier.blocks_of(s, l) {
                    n += u64::from(matches!(self.tier.place(s, l, b), Ok(Place::Ram(_))));
                }
            }
        }
        n
    }

    /// The end-of-run line.
    #[must_use]
    pub fn end_line(&self, rank: u32) -> String {
        let own = self.end_line_own(rank);
        match self.mtp.as_ref() {
            Some(t) => format!(
                "{own}\n{}",
                t.end_line_own(rank).replace(
                    &format!("fnkv rank{rank}:"),
                    &format!("fnkv rank{rank} MTP layer:")
                )
            ),
            None => own,
        }
    }

    fn end_line_own(&self, rank: u32) -> String {
        let c = self.tier.counts();
        let s = self.stats;
        let ram = self.ram_entries();
        format!(
            "fnkv rank{rank}: steps {} flush runs {} blocks {} ({:.1} MiB) gather {} runs {} blocks \
             {} ({:.1} MiB) {} UVA evictions {} uploads {} ({:.1} KiB) ram-resident entries now {} hot \
             free {}{}{}",
            s.steps,
            s.flush_runs,
            s.flush_blocks,
            (s.flush_blocks * self.bb) as f64 / (1u64 << 20) as f64,
            if self.geom.gather { "on" } else { "OFF" },
            s.gather_runs,
            s.gather_blocks,
            (s.gather_blocks * self.bb) as f64 / (1u64 << 20) as f64,
            s.gather_uva,
            c.evictions,
            s.upload_jobs,
            s.upload_bytes as f64 / 1024.0,
            ram,
            self.tier.vram_free(),
            if self.ssd.is_some() {
                format!(
                    " | DRIVE (resident law): {} blocks in {} runs ({:.1} MiB, {:.1} ms host), RAM \
                     reserved {} of {} slots, {} drive reads",
                    s.ssd_blocks,
                    s.ssd_runs,
                    (s.ssd_blocks * self.bb) as f64 / (1u64 << 20) as f64,
                    s.ssd_ms,
                    self.tier.ram_reserved(),
                    self.tier.plan().ram_slots,
                    c.ssd_reads
                )
            } else {
                String::new()
            },
            if self.geom.count_ram {
                format!(
                    " | RAM-addressed live entries {} over {} steps{}",
                    s.ram_live,
                    s.ram_steps,
                    if self.geom.drop_flush {
                        " (FLUSHES DROPPED: negative control)"
                    } else {
                        ""
                    }
                )
            } else {
                String::new()
            }
        )
    }
}

/// The tier as the pools hold it (the step builders borrow the rank immutably).
pub type KvCell = Option<RefCell<KvDev>>;

#[cfg(test)]
mod tests {
    /// FN51f: the decode band -- a boot at or under 10,240 keeps its own context (today's bytes);
    /// a 262,144 boot runs every step whose rows (+ the margin) sit under 10,240 at the 10,240
    /// boot's band (and its index row length), longer rows at the smallest band covering them,
    /// never past the boot; a chain's band (its first pass + the margin) covers k <= 4 passes.
    #[test]
    fn fn51f_the_decode_band_follows_the_live_context_and_equals_the_short_boot_below_it() {
        use super::*;
        for m in [2_048u32, 8_192, 10_240] {
            for need in [0u32, 5_000, 10_240, 70_000] {
                assert_eq!(ctx_band(m, need), m, "a short boot keeps its own band");
            }
        }
        let long = 262_144;
        assert_eq!(band_of(long, &[0, 300, 9_000]), 10_240);
        assert_eq!(band_of(long, &[10_176]), 10_240);
        assert_eq!(band_of(long, &[10_177]), 32_768);
        assert_eq!(band_of(long, &[40_000, 12]), 65_536);
        assert_eq!(band_of(long, &[262_000]), 262_144);
        assert_eq!(band_of(65_536, &[60_000]), 65_536);
        assert_eq!(band_of(65_536, &[65_535]), 65_536, "capped at the boot");
        assert_eq!(band_nb(10_240), 641, "the 10,240 boot's nbs");
        assert_eq!(band_nb(262_144), 16_385);
        // every row a band serves sits inside it (decode w <= 5, chain k <= 4 ahead of pass 1)
        for p in (0..262_144u32 - BAND_MARGIN).step_by(997) {
            let b = band_of(long, &[p]);
            assert!(p + 5 + 4 <= b && b <= long, "{p} -> {b}");
        }
    }

    /// FN51e: the admission slab = one max-context request + one 10,240-token request beside it
    /// (both fit the scheduler's budget, a third short does not), 0.53 GiB under FN51's 3.65 GiB
    /// over both ranks at 262,144; `--fn-kv-ram-gib 3.65` (the control) is FN51's slab.
    #[test]
    fn fn51e_the_admission_slab_holds_one_full_request_and_one_short_beside_it() {
        use super::*;
        let g = KvGeom::new(262_144, 8192, None, None).with_admission(None, 4096);
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        assert_eq!(g.ram_bytes, admit_slab_bytes(262_144));
        assert!(
            (gib(g.ram_bytes) - 3.386).abs() < 0.001,
            "{}",
            gib(g.ram_bytes)
        );
        let fn51 = KvGeom::new(262_144, 8192, None, None).with_admission(Some(3.65), 4096);
        assert!(gib(2 * (fn51.ram_bytes - g.ram_bytes)) > 0.52);
        let t = g.active_tokens();
        assert!(t >= 262_144 + u64::from(DEFAULT_FN_KV_MAX_CTX), "{t}");
        assert!(t < 262_144 + 2 * u64::from(DEFAULT_FN_KV_MAX_CTX));
        g.plan().unwrap();
        crate::flashnext::serve::scheduler_config(262_144, 5, false, 8192, 0, Some(t)).unwrap();
        // the slab follows the boot's context (a 64k boot pins its own)
        assert!(admit_slab_bytes(65_536) < admit_slab_bytes(262_144) / 3);
    }

    /// FN68: `--fn-kv-dtype fp8` keeps every TOKEN budget and halves every byte: the same hot
    /// window, quota and slots, the page and the slab half (262k admission slab 3.386 -> 1.693
    /// GiB a rank), the same scheduler budget; bf16 is today's geometry exactly.
    #[test]
    fn fn68_fp8_pages_halve_every_byte_at_the_same_tokens() {
        use super::*;
        assert_eq!(page_bytes(KvDtype::Bf16), BB);
        assert_eq!(page_bytes(KvDtype::Fp8) * 2, BB);
        assert_eq!(token_bytes(KvDtype::Bf16), TOKEN_BYTES);
        assert_eq!(token_bytes(KvDtype::Fp8), 13 * 512);
        for ctx in [10_240u32, 65_536, 262_144] {
            let b = KvGeom::new(ctx, 8192, None, None);
            assert_eq!(b, KvGeom::new_dt(ctx, 8192, None, None, KvDtype::Bf16));
            let f = KvGeom::new_dt(ctx, 8192, None, None, KvDtype::Fp8);
            assert_eq!((f.window, f.chunk), (b.window, b.chunk));
            assert_eq!(f.hot_bytes * 2, b.hot_bytes);
            assert_eq!(f.ram_bytes * 2, b.ram_bytes);
            assert_eq!(f.hot_alloc() * 2, b.hot_alloc());
            assert_eq!(f.ram_alloc() * 2, b.ram_alloc());
            let (pb, pf) = (b.plan().unwrap(), f.plan().unwrap());
            assert_eq!(
                (pf.hot_slots, pf.ram_slots, pf.quota, pf.window_tokens),
                (pb.hot_slots, pb.ram_slots, pb.quota, pb.window_tokens)
            );
            assert_eq!(pf.block_bytes(), page_bytes(KvDtype::Fp8));
            let (ab, af) = (b.with_admission(None, 4096), f.with_admission(None, 4096));
            assert_eq!(af.ram_bytes * 2, ab.ram_bytes);
            assert_eq!(af.active_tokens(), ab.active_tokens());
            assert_eq!(af.ix_pages(), ab.ix_pages());
            let (qb, qf) = (ab.plan().unwrap(), af.plan().unwrap());
            assert_eq!((qf.hot_slots, qf.ram_slots), (qb.hot_slots, qb.ram_slots));
        }
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        let s = admit_slab_bytes_dt(262_144, KvDtype::Fp8);
        assert_eq!(s * 2, admit_slab_bytes(262_144));
        assert!((gib(s) - 1.693).abs() < 0.001, "{}", gib(s));
        // an explicit byte budget stays bytes: twice the pages at fp8
        let (b, f) = (
            KvGeom::new(65_536, 8192, Some(0.5), Some(8.0)),
            KvGeom::new_dt(65_536, 8192, Some(0.5), Some(8.0), KvDtype::Fp8),
        );
        assert_eq!(f.hot_bytes / f.bb(), 2 * (b.hot_bytes / b.bb()));
    }

    /// FN51: the paged index store -- every page free or held once through random grow /
    /// release / move, rows a held prefix then pads, exhaustion refused by name.
    #[test]
    fn fn51_the_index_pool_holds_every_page_once() {
        let mut ix = super::IxPool::new(64, 9, 20);
        let mut x: u64 = 7;
        for _ in 0..4000 {
            x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let (s, t) = (((x >> 33) % 9) as u32, ((x >> 40) % 9) as u32);
            match (x >> 50) % 4 {
                0 | 1 => {
                    let want = ix.held(s) + ((x >> 20) % 4) as u32;
                    if want - ix.held(s) <= ix.free() {
                        ix.ensure(s, want).unwrap();
                    }
                }
                2 => ix.release(s),
                _ => ix.move_row(s, t),
            }
            ix.check().unwrap();
        }
        let mut full = super::IxPool::new(4, 2, 8);
        full.ensure(0, 4).unwrap();
        assert!(full
            .ensure(1, 1)
            .unwrap_err()
            .to_string()
            .contains("exhausted"));
        full.release(0);
        full.ensure(1, 2).unwrap();
        assert_eq!(
            &full.row(1)[..3],
            &[0, 1, 4],
            "lowest pages first, then the pad"
        );
    }

    use super::*;

    #[test]
    fn tier_layers_are_the_qsa_layers_and_the_mtp_layer() {
        let v: Vec<u32> = (0..49).filter_map(tier_layer).collect();
        assert_eq!(v, (0..13).collect::<Vec<_>>());
        assert_eq!(tier_layer(3), Some(0));
        assert_eq!(tier_layer(47), Some(11));
        assert_eq!(tier_layer(48), Some(12));
        assert_eq!(tier_layer(4), None);
        assert_eq!(tier_layer(-1), None);
    }

    #[test]
    fn the_default_hot_pool_is_todays_flat_pools_and_the_plan_holds_at_64k() {
        let g = KvGeom::new(65_536, 8192, None, None);
        // today's flat pools at 8,192: 13 x 9 x 513 pages of 16 KiB = 0.916 GiB
        assert_eq!(g.hot_bytes, 13 * 9 * 513 * BB);
        assert_eq!(g.window, 8192 - 128);
        assert_eq!(g.ram_bytes, 8 * 13 * 4096 * BB);
        let p = g.plan().unwrap();
        assert!(p.quota >= 8, "quota {}", p.quota);
        // short boots: the window shrinks with the pools
        let s = KvGeom::new(2304, 8192, None, None);
        assert_eq!(s.chunk, 2304);
        assert_eq!(s.window, 2304 - 128);
        s.plan().unwrap();
        // a RAM budget under 8 x 64k is refused at boot
        assert!(KvGeom::new(65_536, 8192, None, Some(4.0)).plan().is_err());
    }

    /// The device model: hot pages, RAM pages, the pad page (16 values a page).
    struct Model {
        hot: Vec<[u64; 16]>,
        ram: Vec<[u64; 16]>,
        pad: [u64; 16],
    }

    impl Model {
        fn page(&mut self, a: &KvAddr, page: i32) -> &mut [u64; 16] {
            match a.resolve(page) {
                Some(Place::Vram(v)) => &mut self.hot[v as usize],
                Some(Place::Ram(r)) => &mut self.ram[r as usize],
                _ => &mut self.pad,
            }
        }
    }

    struct Rng(u64);
    impl Rng {
        fn below(&mut self, n: u64) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 % n.max(1)
        }
    }

    type Expect = Vec<Option<(u32, Vec<Vec<u64>>)>>;

    /// The wiring's round trip on the host: random prefill chunks, verify steps whose rejected
    /// rows are rewritten later, MTP rows past the step and the fed row before it, slot moves and
    /// restarts, a hot pool far under the contexts (evictions, RAM reads) -- every K/V write goes
    /// through the step's per-layer append slots, every flush lands in issue order BEFORE the
    /// step's writes (the stream), and after every step every position below each sequence's
    /// context reads, through its layer's table row, the value its LAST write left (the page byte
    /// law). The negative control (flushes dropped) must fail.
    fn round_trip(seed: u64, drop_flushes: bool) -> Result<u64, String> {
        round_trip_g(seed, drop_flushes).map(|x| x.0)
    }

    /// [`round_trip`] with FN45b4's prefill steps (one sequence, a prompt chunk: protected +
    /// gathered): after the gather every earlier page of the chunk's sequence resolves to VRAM
    /// except exactly the blocks the plan counted as UVA. Returns (reads, chunks, RAM pages a
    /// chunk read in place, gathered blocks).
    fn round_trip_g(seed: u64, drop_flushes: bool) -> Result<(u64, u64, u64, u64), String> {
        let geom = KvGeom {
            max_ctx: 1024,
            window: 128,
            chunk: 256,
            hot_bytes: 13 * (8 * 9 + 17 + 8 * 8 + 24) * BB,
            ram_bytes: 8 * 13 * 64 * BB,
            count_ram: false,
            drop_flush: false,
            gather: true,
            ssd_bytes: 0,
            admit: false,
            first_admit: 1024,
            dt: KvDtype::Bf16,
            layers: QSA_LAYERS,
            mtp: None,
            lazy: false,
        };
        let plan = geom.plan().unwrap();
        let mut tier = KvTier::new(plan.clone());
        let hot = plan.hot_slots;
        let addr = KvAddr::new(
            BB,
            0x10_0000_0000,
            hot,
            0x20_0000_0000,
            plan.ram_slots,
            0x10_0000_0000 + u64::from(hot) * BB,
        )
        .unwrap();
        let mut table = KvPageTable::new(&plan, 9, addr).unwrap();
        let mut dm = Model {
            hot: vec![[0; 16]; hot as usize],
            ram: vec![[0; 16]; plan.ram_slots as usize],
            pad: [0; 16],
        };
        let mut rng = Rng(seed);
        let mut seqs: Expect = vec![None; 9];
        let mut stamp = 1u64;
        let mut reads = 0u64;
        let (mut chunks, mut in_place) = (0u64, 0u64);
        for _ in 0..400 {
            let op = rng.below(10);
            if op == 0 {
                let s = rng.below(8) as u32;
                if tier.is_live(s) {
                    tier.finish(s).unwrap();
                }
                seqs[s as usize] = Some((0, vec![vec![0; 1024 + 64]; 13]));
                continue;
            }
            if op == 1 {
                let (a, b) = (rng.below(8) as u32, rng.below(9) as u32);
                if a != b && seqs[a as usize].is_some() && tier.is_live(a) {
                    if b == 8 {
                        // the swap through the spare slot (the spare never stays live: the
                        // reservation holds 8 sequences)
                        let c = (a + 1 + rng.below(7) as u32) % 8;
                        tier.move_seq(a, 8).unwrap();
                        if seqs[c as usize].is_some() && tier.is_live(c) {
                            tier.move_seq(c, a).unwrap();
                        }
                        tier.move_seq(8, c).unwrap();
                        seqs.swap(a as usize, c as usize);
                        if !tier.is_live(a) {
                            seqs[a as usize] = None;
                        }
                    } else {
                        tier.move_seq(a, b).unwrap();
                        seqs[b as usize] = seqs[a as usize].take();
                    }
                }
                continue;
            }
            let live: Vec<u32> = (0..8).filter(|&s| seqs[s as usize].is_some()).collect();
            // slot, first row, rows written, rows accepted
            let mut rows: Vec<(u32, u32, u32, u32)> = Vec::new();
            for &s in &live {
                let c = seqs[s as usize].as_ref().unwrap().0;
                let w = if c == 0 || rng.below(4) == 0 {
                    1 + rng.below(200) as u32
                } else {
                    1 + rng.below(5) as u32
                };
                if c + w + 8 > 1024 {
                    continue;
                }
                let acc = if w > 5 {
                    w
                } else {
                    1 + rng.below(u64::from(w)) as u32
                };
                rows.push((s, c, w, acc));
            }
            // FN45b4: a third of the steps are one sequence's prompt chunk
            let pf = rng.below(3) == 0 && !rows.is_empty();
            if pf {
                let i = rng.below(rows.len() as u64) as usize;
                let (s, p0, _, _) = rows[i];
                let w = (1 + rng.below(200) as u32).min(1024 - 8 - p0);
                rows = vec![(s, p0, w, w)];
            }
            let steps: Vec<(u32, u32, u32)> =
                rows.iter().map(|&(s, p0, w, _)| (s, p0, p0 + w)).collect();
            let sp = plan_step(&mut tier, &mut table, &geom, &steps, pf).unwrap();
            if !drop_flushes {
                for op in &sp.flush {
                    if let CopyOp::D2h { vram, ram } = *op {
                        dm.ram[ram as usize] = dm.hot[vram as usize];
                    }
                }
            }
            for op in &sp.gather {
                let CopyOp::H2d { ram, vram } = *op else {
                    return Err("a gather issues H2Ds only".into());
                };
                dm.hot[vram as usize] = dm.ram[ram as usize];
            }
            if pf {
                let s = rows[0].0;
                let mut ram = 0u64;
                for l in 0..13u32 {
                    for b in 0..tier.blocks_of(s, l) {
                        let page = table.row(l, s)[b as usize];
                        ram += u64::from(matches!(addr.resolve(page), Some(Place::Ram(_))));
                    }
                }
                if ram != sp.uva {
                    return Err(format!(
                        "seed {seed}: a chunk reads {ram} RAM pages, the gather counted {}",
                        sp.uva
                    ));
                }
                chunks += 1;
                in_place += ram;
            } else if !sp.gather.is_empty() {
                return Err("a decode step gathered".into());
            }
            for &(s, p0, w, acc) in &rows {
                let mtp = (p0.saturating_sub(1)..p0 + w + 3).map(|p| (12u32, p));
                let tgt = (0..12u32).flat_map(|l| (p0..p0 + w).map(move |p| (l, p)));
                for (l, p) in tgt.chain(mtp) {
                    stamp += 1;
                    let at = table.append_slot(l, s, p);
                    let (page, k) = ((at / 16) as i32, (at % 16) as usize);
                    dm.page(&addr, page)[k] = stamp;
                    seqs[s as usize].as_mut().unwrap().1[l as usize][p as usize] = stamp;
                }
                seqs[s as usize].as_mut().unwrap().0 = p0 + acc;
            }
            for &s in &live {
                let (c, want) = seqs[s as usize].as_ref().unwrap();
                for l in 0..13u32 {
                    let row = table.row(l, s).to_vec();
                    for p in 0..*c {
                        let got = dm.page(&addr, row[(p / 16) as usize])[(p % 16) as usize];
                        reads += 1;
                        if got != want[l as usize][p as usize] {
                            return Err(format!("seed {seed}: slot {s} layer {l} pos {p}"));
                        }
                    }
                }
            }
            tier.check()?;
        }
        if tier.counts().evictions == 0 {
            return Err("no eviction: the round trip did not reach RAM".into());
        }
        Ok((reads, chunks, in_place, tier.counts().h2d))
    }

    #[test]
    fn every_read_through_the_layer_tables_is_the_last_write_across_flushes_and_evictions() {
        for seed in [7u64, 99, 1234, 0xfeed] {
            let n = round_trip(seed, false).unwrap();
            assert!(n > 100_000, "{n} reads");
        }
        // the negative control: a tier whose flushes never land reads stale RAM pages
        assert!((0..4u64).any(|i| round_trip(31 + i, true).is_err()));
    }

    #[test]
    fn a_prompt_chunk_reads_its_gathered_pages_hot_and_the_gather_moves_bytes() {
        let (mut chunks, mut h2d) = (0, 0);
        for seed in [5u64, 77, 4321] {
            let (n, c, _, g) = round_trip_g(seed, false).unwrap();
            assert!(n > 50_000 && c > 50, "{n} reads {c} chunks");
            chunks += c;
            h2d += g;
        }
        assert!(
            chunks > 150 && h2d > 0,
            "{chunks} chunks {h2d} gathered blocks"
        );
    }

    #[test]
    fn the_host_law_counts_the_kv_slab_and_the_default_is_the_ceiling() {
        use saber_mem::flashnext_budget::{geometry::EXPERT_STRIDE, HostBudget};
        // this box: MemTotal 63,336,184 kB; the served boot's L2 (the complement: fn49b hold 2)
        let host = HostBudget::this_box(63_336_184 * 1024);
        let l2 = [5750u64, 5750];
        host.check(&l2, EXPERT_STRIDE).unwrap();
        let at = |ctx: u32| KvGeom::new(ctx, 8192, None, None).ram_alloc();
        assert_eq!(at(10_240), 8 * 10_240 * TOKEN_BYTES + BB);
        // the default fits, the next 2,048 rung and FN45b3's 64k are refused
        host.clone()
            .with_kv_ram(at(DEFAULT_FN_KV_MAX_CTX))
            .check(&l2, EXPERT_STRIDE)
            .unwrap();
        assert!(host
            .clone()
            .with_kv_ram(at(DEFAULT_FN_KV_MAX_CTX + 2048))
            .check(&l2, EXPERT_STRIDE)
            .is_err());
        assert!(host
            .clone()
            .with_kv_ram(at(65_536))
            .check(&l2, EXPERT_STRIDE)
            .is_err());
        let ceil = host.kv_ceiling(&l2, EXPERT_STRIDE, TOKEN_BYTES, BB, 2048);
        assert_eq!(ceil, u64::from(DEFAULT_FN_KV_MAX_CTX));
        // the flat pools pin no slab: the line is today's
        assert!(!host.boot_line(5750, 5750, EXPERT_STRIDE).contains("KV RAM"));
        assert!(host
            .with_kv_ram(at(10_240))
            .boot_line(5750, 5750, EXPERT_STRIDE)
            .contains("KV RAM 1.02 GiB"));
    }

    fn small_geom() -> KvGeom {
        KvGeom {
            max_ctx: 1024,
            window: 128,
            chunk: 256,
            hot_bytes: 13 * (8 * 9 + 17 + 8 * 8 + 24) * BB,
            ram_bytes: 8 * 13 * 64 * BB,
            count_ram: false,
            drop_flush: false,
            gather: true,
            ssd_bytes: 0,
            admit: false,
            first_admit: 1024,
            dt: KvDtype::Bf16,
            layers: QSA_LAYERS,
            mtp: None,
            lazy: false,
        }
    }

    fn apply_ops(m: &mut Model, ops: &[CopyOp]) {
        for op in ops {
            match *op {
                CopyOp::D2h { vram, ram } => m.ram[ram as usize] = m.hot[vram as usize],
                CopyOp::H2d { ram, vram } => m.hot[vram as usize] = m.ram[ram as usize],
                _ => {}
            }
        }
    }

    fn val(l: u32, p: u32) -> u64 {
        (u64::from(l) << 32) | u64::from(p) | 0x5a00_0000_0000_0000
    }

    /// FN45c2, the prompt cache's page law on the host model: a chunk's pages dumped from one
    /// slot (`KvDev::dump`'s walk: every block of `chunk_blocks` through `place`) and loaded into
    /// another slot after the same step plan (`pcache::reload`'s `begin` + `KvDev::load`) read,
    /// through the new slot's table, exactly what the prefilled slot reads -- before and after
    /// later steps flush and evict them.
    #[test]
    fn a_reloaded_chunk_reads_every_position_as_the_prefilled_one() {
        let mut geom = small_geom();
        // a pool barely over the floor: the later steps evict the reloaded pages to RAM
        geom.hot_bytes = 13 * (8 * 9 + 17 + 8) * BB;
        let plan = geom.plan().unwrap();
        let mut tier = KvTier::new(plan.clone());
        let addr = KvAddr::new(
            BB,
            0x10_0000_0000,
            plan.hot_slots,
            0x20_0000_0000,
            plan.ram_slots,
            0x10_0000_0000 + u64::from(plan.hot_slots) * BB,
        )
        .unwrap();
        let mut table = KvPageTable::new(&plan, 9, addr).unwrap();
        let mut m = Model {
            hot: vec![[0; 16]; plan.hot_slots as usize],
            ram: vec![[0; 16]; plan.ram_slots as usize],
            pad: [0; 16],
        };
        let write = |m: &mut Model, t: &KvPageTable, s: u32, a: u32, b: u32| {
            for l in 0..QSA_LAYERS {
                for p in a..b {
                    let page = t.row(l, s)[(p / 16) as usize];
                    m.page(&addr, page)[(p % 16) as usize] = val(l, p);
                }
            }
        };
        let step = |tier: &mut KvTier, t: &mut KvPageTable, m: &mut Model, s, a, b, pf| {
            let sp = plan_step(tier, t, &geom, &[(s, a, b)], pf).unwrap();
            apply_ops(m, &sp.flush);
            apply_ops(m, &sp.gather);
        };
        // A (slot 3): two chunks prefilled
        for (a, b) in [(0, 256), (256, 512)] {
            step(&mut tier, &mut table, &mut m, 3, a, b, true);
            write(&mut m, &table, 3, a, b);
        }
        let (b0, b1) = super::super::worker::pcache_blocks(256, 512);
        let mut snap = Vec::new();
        for l in 0..QSA_LAYERS {
            for b in b0..b1 {
                let page = addr.page(tier.place(3, l, b).unwrap()).unwrap();
                snap.push(*m.page(&addr, page));
            }
        }
        // B (slot 5): the first chunk prefilled, the second RELOADED (no compute writes)
        step(&mut tier, &mut table, &mut m, 5, 0, 256, true);
        write(&mut m, &table, 5, 0, 256);
        step(&mut tier, &mut table, &mut m, 5, 256, 512, true);
        let mut i = 0;
        for l in 0..QSA_LAYERS {
            for b in b0..b1 {
                let page = addr.page(tier.place(5, l, b).unwrap()).unwrap();
                *m.page(&addr, page) = snap[i];
                i += 1;
            }
        }
        // later decode steps on both (flushes, window moves, evictions under a small pool)
        for p in 512..1000 {
            for s in [3, 5] {
                step(&mut tier, &mut table, &mut m, s, p, p + 1, false);
                write(&mut m, &table, s, p, p + 1);
            }
            if p % 37 != 0 && p != 999 {
                continue;
            }
            for s in [3, 5] {
                for l in 0..QSA_LAYERS {
                    for q in 0..=p {
                        let page = table.row(l, s)[(q / 16) as usize];
                        assert_eq!(
                            m.page(&addr, page)[(q % 16) as usize],
                            val(l, q),
                            "slot {s} layer {l} position {q} after step {p}"
                        );
                    }
                }
            }
        }
        assert!(tier.counts().evictions > 0, "the pool never evicted");
    }

    /// FN45c2: the drive tier admits by REQUEST LENGTH (today's served RAM slab: one 64k
    /// request beside seven short ones, a second 64k refused), the scheduler's budget is that
    /// slab less the per-slot extra, and a commit's drive writes copy exactly the RAM slots its
    /// flushes filled (issued a step later, after they land).
    #[test]
    fn the_drive_tier_admits_by_request_length_and_writes_through() {
        let g = KvGeom::new(65_536, 8192, None, None)
            .with_ssd(8.0, None)
            .unwrap();
        assert_eq!(
            g.ram_bytes,
            8 * 13 * u64::from(DEFAULT_FN_KV_MAX_CTX / 16) * BB
        );
        assert!(KvGeom::new(65_536, 8192, None, None)
            .with_ssd(SSD_GIB_MAX + 1.0, None)
            .is_err());
        // 8 x 64k on a drive under it is refused at boot
        assert!(KvGeom::new(65_536, 8192, None, None)
            .with_ssd(4.0, None)
            .unwrap()
            .plan()
            .is_err());
        // FN45c3: the 64k drive-tier boot at a 0.75 GiB slab (FN45c2's refused cell) plans: the
        // RAM floor is one widest admission's chunk + a gather, not 8 hot windows; one 49k
        // request is admitted on it, a second waits
        let small = KvGeom::new(65_536, 8192, None, Some(0.75))
            .with_ssd(8.0, Some(0.75))
            .unwrap();
        let sp = small.plan().unwrap();
        assert!(small.active_tokens() >= 49_152 && small.active_tokens() < 2 * 49_152);
        let mut st = KvTier::new(sp).resident().unwrap();
        st.admit(0, 49_152).unwrap();
        assert!(st.admit(1, 49_152).is_err());
        let plan = g.plan().unwrap();
        let mut tier = KvTier::new(plan.clone()).resident().unwrap();
        let active = g.active_tokens();
        assert_eq!(
            active,
            8 * u64::from(DEFAULT_FN_KV_MAX_CTX) - 9 * u64::from(ADMIT_EXTRA)
        );
        let big = 65_536 - 1024;
        tier.admit(0, big + ADMIT_EXTRA).unwrap();
        for s in 1..8 {
            tier.admit(s, 2048 + ADMIT_EXTRA).unwrap();
        }
        assert!(u64::from(big) + 7 * 2048 <= active);
        tier.finish(1).unwrap();
        assert!(
            tier.admit(1, big + ADMIT_EXTRA).is_err(),
            "a second 64k fits"
        );
        // the scheduler's budget is the slab's active tokens
        let c = super::super::serve::scheduler_config(65_536, 1, false, 2048, 0, Some(active));
        assert!(c.is_ok(), "{c:?}");
        // write-through: a 2,048 prompt then a decode step
        let addr = KvAddr::new(
            BB,
            0x10_0000_0000,
            plan.hot_slots,
            0x20_0000_0000,
            plan.ram_slots,
            0x10_0000_0000 + u64::from(plan.hot_slots) * BB,
        )
        .unwrap();
        let mut table = KvPageTable::new(&plan, 9, addr).unwrap();
        let p1 = plan_step(&mut tier, &mut table, &g, &[(2, 0, 2048)], true).unwrap();
        assert!(p1.ssd_writes.is_empty());
        let p2 = plan_step(&mut tier, &mut table, &g, &[(2, 2048, 2049)], false).unwrap();
        let flushed: Vec<u32> = p2
            .flush
            .iter()
            .filter_map(|o| match *o {
                CopyOp::D2h { ram, .. } => Some(ram),
                _ => None,
            })
            .collect();
        let written: Vec<u32> = p2.ssd_writes.iter().map(|w| w.0).collect();
        assert!(!written.is_empty());
        assert_eq!(flushed, written);
    }

    #[test]
    fn a_prefill_step_protects_and_a_decode_step_releases() {
        let geom = KvGeom::new(4096, 2048, None, None);
        let plan = geom.plan().unwrap();
        let mut tier = KvTier::new(plan.clone());
        let a = KvAddr::new(
            BB,
            0x10_0000_0000,
            plan.hot_slots,
            0x20_0000_0000,
            plan.ram_slots,
            0x10_0000_0000 + u64::from(plan.hot_slots) * BB,
        )
        .unwrap();
        let mut table = KvPageTable::new(&plan, 9, a).unwrap();
        plan_step(&mut tier, &mut table, &geom, &[(3, 0, 2048)], true).unwrap();
        assert!(tier.is_protected(3));
        plan_step(&mut tier, &mut table, &geom, &[(3, 2048, 2049)], false).unwrap();
        assert!(!tier.is_protected(3));
    }

    /// FN68d: `--fn-kv-mtp-dtype bf16` -- the MTP layer's one-layer bf16 TWIN beside the 12
    /// target layers' fp8 tier: the same pages a layer in both (so the same hot window, quota,
    /// admission, scheduler budget and index pool), the bytes = 12/13 of the fp8 tier + one bf16
    /// layer (the 262k slab 1.693 -> 1.823 GiB a rank, +0.13), the elastic borrow stays the main
    /// tier's, `fp8` is the one-tier boot exactly; the twin runs the step law beside the main tier
    /// (same admissions and final lengths) and the step constants' row 12 is the twin's.
    #[test]
    fn fn68d_the_mtp_twin_holds_one_bf16_layer_at_the_main_tiers_pages() {
        use super::*;
        let (f8, b16) = (KvDtype::Fp8, KvDtype::Bf16);
        for (ctx, admit) in [(10_240u32, false), (65_536, false), (262_144, true)] {
            let mut base = KvGeom::new_dt(ctx, 8192, None, None, f8);
            if admit {
                base = base.with_admission(None, 4096);
            }
            assert_eq!(base.split_mtp(f8, 0), base, "fp8 = the one-tier boot");
            assert_eq!(base.mtp_dt(), f8);
            let s = base.split_mtp(b16, 0);
            let t = s.twin().unwrap();
            assert_eq!((s.layers, s.dt, t.layers, t.dt), (12, f8, 1, b16));
            assert_eq!(s.mtp_dt(), b16);
            assert!(t.twin().is_none() && base.twin().is_none());
            let per = |g: &KvGeom, b: u64| b / g.bb() / u64::from(g.layers);
            for (g, what) in [(&s, "main"), (&t, "twin")] {
                assert_eq!(
                    per(g, g.hot_bytes),
                    per(&base, base.hot_bytes),
                    "{ctx} {what} hot"
                );
                assert_eq!(
                    per(g, g.ram_bytes),
                    per(&base, base.ram_bytes),
                    "{ctx} {what} ram"
                );
                assert_eq!(g.active_tokens(), base.active_tokens(), "{ctx} {what}");
                assert_eq!(
                    (g.window, g.chunk, g.admit),
                    (base.window, base.chunk, base.admit)
                );
            }
            assert_eq!(s.ix_pages(), base.ix_pages());
            let (pb, ps, pt) = (base.plan().unwrap(), s.plan().unwrap(), t.plan().unwrap());
            assert_eq!((ps.quota, ps.window_tokens), (pb.quota, pb.window_tokens));
            assert_eq!((pt.quota, pt.window_tokens), (pb.quota, pb.window_tokens));
            assert_eq!(ps.hot_slots / 12, pb.hot_slots / 13, "{ctx}");
            assert_eq!(pt.hot_slots, pb.hot_slots / 13, "{ctx}");
            assert_eq!(pt.ram_slots, pb.ram_slots / 13, "{ctx}");
            assert_eq!(pt.block_bytes(), page_bytes(b16));
            assert_eq!(
                (s.ram_bytes + t.ram_bytes) * 13,
                base.ram_bytes * 14,
                "{ctx}"
            );
            assert_eq!(
                (s.hot_bytes + t.hot_bytes) * 13,
                base.hot_bytes * 14,
                "{ctx}"
            );
            assert_eq!(s.ram_alloc_all(), s.ram_alloc() + t.ram_alloc());
            assert_eq!(s.hot_alloc_all(), s.hot_alloc() + t.hot_alloc());
            assert_eq!(s.table_bytes_all(), base.table_bytes(), "13 layers' tables");
            assert_eq!(s.token_bytes_all(), 12 * 512 + 1024);
            assert_eq!(base.token_bytes_all(), token_bytes(f8));
            // bf16 pages need no twin
            let b = KvGeom::new_dt(ctx, 8192, None, None, b16);
            assert_eq!(b.split_mtp(b16, 0), b);
        }
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        let base = KvGeom::new_dt(262_144, 8192, None, None, f8).with_admission(None, 4096);
        let s = base.split_mtp(b16, 0);
        let more = gib(s.ram_alloc_all()) - gib(base.ram_alloc());
        assert!((0.12..0.14).contains(&more), "{more} GiB a rank");
        assert!((gib(s.ram_alloc_all()) - 1.823).abs() < 0.002);
        // the elastic borrow (added before the split) stays the main tier's
        let lent = 4096 * page_bytes(f8);
        let mut g = KvGeom::new_dt(262_144, 8192, None, None, f8);
        let plain = g.split_mtp(b16, 0);
        g.hot_bytes += lent;
        let e = g.with_admission(None, 4096).split_mtp(b16, lent);
        assert_eq!(e.hot_bytes, plain.hot_bytes + lent);
        assert_eq!(e.twin().unwrap().hot_bytes, plain.twin().unwrap().hot_bytes);
        // the twin's step law beside the main tier's
        let s = KvGeom::new_dt(4096, 2048, None, None, f8).split_mtp(b16, 0);
        let t = s.twin().unwrap();
        let mk = |g: &KvGeom| {
            let plan = g.plan().unwrap();
            let bb = g.bb();
            let a = KvAddr::new(
                bb,
                0x10_0000_0000,
                plan.hot_slots,
                0x20_0000_0000,
                plan.ram_slots,
                0x10_0000_0000 + u64::from(plan.hot_slots) * bb,
            )
            .unwrap();
            (
                KvTier::new(plan.clone()),
                KvPageTable::new(&plan, 9, a).unwrap(),
            )
        };
        let ((mut tm, mut qm), (mut tt, mut qt)) = (mk(&s), mk(&t));
        type Step = (Vec<(u32, u32, u32)>, bool);
        let steps: Vec<Step> = vec![
            (vec![(0, 0, 2048)], true),
            (vec![(0, 2048, 3000)], true),
            (vec![(2, 0, 700)], true),
            (vec![(0, 3000, 3004), (2, 700, 704)], false),
            (vec![(0, 3004, 3008), (2, 704, 708)], false),
        ];
        for (seqs, pre) in &steps {
            let a = plan_step(&mut tm, &mut qm, &s, seqs, *pre).unwrap();
            let b = plan_step(&mut tt, &mut qt, &t, seqs, *pre).unwrap();
            assert_eq!(
                a.flush.len() / 12,
                b.flush.len(),
                "the twin flushes its one layer"
            );
            for &(slot, _, _) in seqs {
                assert_eq!(tm.len(slot), tt.len(slot));
                assert_eq!(tm.blocks_of(slot, 0), tt.blocks_of(slot, 0));
            }
            let rows: Vec<(u32, u32)> = seqs.iter().map(|&(sl, p, _)| (sl, p)).collect();
            let (ks, br) = step_tables_of(&[&qm, &qt], &rows, true);
            let n = rows.len();
            assert_eq!(ks.len(), 13 * n * 8);
            assert_eq!(br.len(), 13 * n * KBTR as usize * 4);
            for (i, &(sl, p)) in rows.iter().enumerate() {
                let at = (12 * n + i) * 8;
                let v = i64::from_le_bytes(ks[at..at + 8].try_into().unwrap());
                assert_eq!(v, qt.append_slot(0, sl, p), "row 12 is the twin's");
                let at = (3 * n + i) * 8;
                let v = i64::from_le_bytes(ks[at..at + 8].try_into().unwrap());
                assert_eq!(v, qm.append_slot(3, sl, p));
            }
            // a one-tier boot's constants are FN45b2's
            assert_eq!(
                step_tables_of(&[&qm], &rows, false),
                step_tables(&qm, &rows, false)
            );
        }
    }
}
