//! The Flash-Next memory plane's parameters as flags (FN8): the tier's laws are `DEFAULT_*`
//! constants here, every one with a `--...` override or a `--no-...` control, in the pattern of
//! the 27B's default laws (`on_unless_off`). The engine does not boot Flash-Next yet; the flags are
//! parsed by `saber flashnext-moe-bench` today and are what the engine's boot will flatten in.
//!
//! What each law is and where its number came from: docs/EXEC.md §14 and LEDGER FN7 / FN8.

use std::path::PathBuf;

use saber_mem::expert_tier::{Demote, TierParams};

/// Inserts (RAM → VRAM promotions) per layer per step. FN8's sweep (LEDGER FN8 §3): 0 (a static
/// placement) wins in-sample but loses 1.9-3.9 ms/step on a held-out fold (resident 79.7 % vs
/// 93.7 %); every cap from 1 up is within the noise floor of the others; 2 is best-or-tied on all
/// three paired traces and moves 12-17 % fewer bytes than the non-binding 64 (FN7's) for 0.7-0.8 %
/// residency. `--xtier-max-inserts 64` is the pre-FN8 control.
pub const DEFAULT_XTIER_MAX_INSERTS: u32 = 2;
/// FN17: the deferred publication law (`--xtier-defer`: a step's promotions go to holes, its
/// demotions to spare RAM slots, published at the next step's fence -- no step waits for the
/// copies of its own end) is byte-exact (the mover gate 480/480, ids 320/320) but LOSES, so it is
/// an arm, OFF by default (LEDGER FN17 hold 3: harness b8 53.2 / 55.9 vs FN8's 52.7 / 50.1 ms/step;
/// BetterBench --quick conc-8 112.1 / 92.1 vs 120.6 / 131.6 tok/s): the b8 cost is the copies'
/// contention with the step's UVA reads on the link, not the gate waits.
pub const DEFAULT_XTIER_DEFER: bool = false;
/// FN17: the expert layers of the tier (the deferred law's spares are 2 x max-inserts per layer).
pub const XTIER_LAYERS: u32 = 48;
/// A layer routing more than this fraction of its VRAM slots in a step reads through (no inserts).
/// FN8's conc-8 sweep (LEDGER FN8 §3).
pub const DEFAULT_XTIER_READ_THROUGH: f64 = 0.5;
/// Next-step NVMe → RAM prefetch reads per step: OFF (FN7: 4 % of prefetch reads ever hit).
pub const DEFAULT_XTIER_PREFETCH: u32 = 0;
/// Whether prefill chunks stage their RAM-resident experts into a VRAM ring before the layer's
/// grouped launches. OFF: FN8 measured NO crossover at the measured grouped cell (tm4 above 64
/// rows reads each routed expert once over UVA): staging the layer's RAM experts loses 16-109 %
/// at 512..2,048 rows and even an ORACLE ring (only the routed experts, routes known in advance)
/// loses 2.7-4.2 % (LEDGER FN8 §4). UVA in place at every chunk size is the law. FN43 wired the
/// ring into the engine (lent slots, per-layer arrivals): byte-exact, loses 4-11 % at c2048 with
/// 128 slots; FN43b: a 48-SLOT ring of the likelier records wins 9-13 % at every length.
/// FN43c: THE ONE FLIP CONSTANT -- the 48-slot ring at the default chunk width is the engine's
/// default iff its card byte law holds (LEDGER FN43c). The default stands down (no refusal) where
/// the ring cannot run ([`XtierFlags::stage_on`]); `--no-xtier-stage` is the control (UVA in place).
pub const DEFAULT_XTIER_STAGE: bool = true;
/// FN43c, the stage-min-rows rule (`--xtier-stage-min-rows R`): only prefill chunks of at least R
/// rows stage; below R the ring is INERT (lent to the tier, no copies). 2,048 = every full prefill
/// chunk at the narrow width (c2048) stages; decode and short tails never do. FN44: the default is
/// a function of the boot's widest chunk ([`STAGE_BY_WIDTH`]); this is the narrow row.
pub const DEFAULT_XTIER_STAGE_MIN_ROWS: u32 = 2048;
/// Staging slots per ring half (the ring is two halves: layer L stages while layer L-1 computes);
/// the ledger line `xtier.stage_ring` = 2 × this × one stride, out of the LRU's remainder.
/// FN43b: 48 (fewer, likelier records: 497 staged-never-routed vs 128 slots' 29 %). FN44: the
/// narrow row of [`STAGE_BY_WIDTH`].
pub const DEFAULT_XTIER_STAGE_SLOTS: u32 = 48;
/// The narrow chunk width (the serve plane's 2,048-row rung; `flashnext::run::MAX_ROWS`).
pub const NARROW_WIDTH: u32 = 2048;
/// FN44: THE ONE TABLE -- the staging ring's default `(min rows, slots per half)` by the boot's
/// widest chunk (`--fn-max-chunk`), first row whose width bound covers it. c2048 (FN43c): every
/// 2,048-row chunk stages into 48 slots. c4096 / c8192 (FN43b's WR128, the user's wide default):
/// only chunks ABOVE 2,048 rows stage (a 2,048-row pass has no window to hide the copies: WR at 2k
/// lost 670-768 vs 624 ms) into 128 slots (8k 1,169 ms = 7,009 tok/s). `--xtier-stage-min-rows` /
/// `--xtier-stage-slots` override both columns at any width.
pub const STAGE_BY_WIDTH: [(u32, u32, u32); 2] = [
    (
        NARROW_WIDTH,
        DEFAULT_XTIER_STAGE_MIN_ROWS,
        DEFAULT_XTIER_STAGE_SLOTS,
    ),
    (u32::MAX, NARROW_WIDTH + 1, 128),
];

/// FN122e: `--fn-prefill-prefetch` -- the staging ring's next-layer prefetch for the wide boot's
/// narrow passes. `off` (today and the control): a wide boot stages only chunks above 2,048 rows
/// ([`STAGE_BY_WIDTH`]); `band`: chunks of >= [`PREFETCH_BAND_MIN_ROWS`] rows stage too (at >= 1,024
/// rows a layer touches every local expert, so the ring's next-layer set is the layer's RAM set).
/// The same ring, the same records, the same bytes (FN43c's law); only which chunks stage moves.
pub const DEFAULT_FN_PREFILL_PREFETCH: &str = "off";
/// FN122e: the band's floor (h11, 2026-10-14: a 1,500-row chunk 327.9 -> 297.0 ms, 2,048 rows
/// 378.3 -> 321.8, bytes 58/58).
pub const PREFETCH_BAND_MIN_ROWS: u32 = 1024;

/// FN44: the ring's default `(min rows, slots per half)` at a boot whose widest chunk is `width`.
#[must_use]
pub fn stage_default(width: u32) -> (u32, u32) {
    let &(_, rows, slots) = STAGE_BY_WIDTH
        .iter()
        .find(|&&(w, _, _)| width <= w)
        .expect("the last row covers every width");
    (rows, slots)
}
/// FN43: the ring's layer buffers (2 = double buffered: layer L + 1 fills while layer L computes;
/// 1 = the next layer's copies wait for this layer's experts).
pub const DEFAULT_XTIER_STAGE_BUFFERS: u32 = 2;
/// The PLE row cache per rank (FN7: 20 MiB serves 98 % of the template corpus). FN51e: 512 MiB
/// (was 1,024; the host law counts the flag's bytes): on FN18's two served route traces
/// (`flashnext-profile ple`) 320, 512 and 1,024 MiB read the SAME drive rows (BetterBench +
/// templates: 666,050; GSM8K + multi-turn: 1,019,336 = 27.11 rows a decode step), and a long
/// prompt fills the cache (4.2 M rows at 262k) -- the cap is 1 GiB of host RAM over both ranks.
/// `--ple-cache-mib 1024` = the pre-FN51e cache (the control); the rows' bytes never change.
pub const DEFAULT_PLE_CACHE_MIB: u64 = 512;
/// PLE reader queue depth for a prefill chunk (FN7's drive sweep: 128 ≈ 850 k rows/s).
pub const DEFAULT_PLE_QD_PREFILL: usize = 128;
/// PLE reader queue depth for decode tokens (16 rows a token).
pub const DEFAULT_PLE_QD_DECODE: usize = 16;

/// FN18: the per-bucket insert cap (`--xtier-max-inserts-by-bucket`, `bucket:cap,..`): decode buckets
/// not listed (and every prefill chunk) take `--xtier-max-inserts`. Empty = FN8's one cap at every
/// bucket (the control is `--xtier-max-inserts-by-bucket ''`).
pub const DEFAULT_XTIER_MAX_INSERTS_BY_BUCKET: &str = "";
/// FN54: the per-step promotion BYTE budget by decode bucket (`--xtier-promote-budget-mib-by-bucket`,
/// `bucket:MiB,..`): empty = none (today's law: only the per-layer insert cap binds).
pub const DEFAULT_XTIER_PROMOTE_BUDGET_MIB_BY_BUCKET: &str = "";
/// FN54: second-chance insertion (`--xtier-second-chance W` every step, or `4:W,8:W` by decode
/// bucket; "0" = off). **`4:8,8:8` is the DEFAULT since FN54c (2026-10-05, hold h1, 210 W, keep-awake
/// in every arm, 4 + 4 interleaved boots on the one-stream mover + FN59b's async plan): a b4/b8
/// record is promoted on its 2nd RAM miss within 8 steps; b1 and prefill chunks keep the old
/// insertion. Bit-exact (placement only: harness 16/16 row sha, API ids + logprobs 11/11 boots);
/// conc-8 link bytes -20 % (rank 0 H2D/D2H ~170/144 -> ~136/115 MiB a step), VRAM hit +0.6 pt,
/// BetterBench --quick conc-8 175.7 -> 184.8 tok/s, single 94.5 -> 98.4. `0` = the control.**
pub const DEFAULT_XTIER_SECOND_CHANCE: &str = "4:8,8:8";
/// FN54b: the device mover's copy queues (`--xtier-mover`): `single` = one in-order copy stream
/// (FN8's law, today) or `two-queue` = promotions + layer marks on an urgent stream, demotions on a
/// background stream, ordered by the slot law's events (`saber_mem::device_mover::LanePlanner`).
pub const DEFAULT_XTIER_MOVER: &str = "single";
/// FN54b: extra swap spares under `--xtier-mover two-queue` (FN8's one spare makes every D2H wait
/// the H2D before it; 4 more pinned RAM slots a rank let the demotions run ahead).
pub const TWO_QUEUE_SPARES: u32 = 4;
/// FN72b: the one-stream mover's landed marks (`--xtier-mover-marks`): `layer` = a HIP event per
/// layer batch (today's law: ROCclr parks a barrier packet on the copy stream's hardware queue
/// behind each SDMA batch); `word` = a 4-byte SDMA write of the layer's device word + the gate's
/// `hipStreamWaitValue32` (no HIP event on the copy stream); `step` = one event a step at the
/// first gate (the discriminator; a layer waits every copy). **`word` is the DEFAULT since FN72b
/// (2026-10-08, hold h1, 210 W): the per-layer events' barrier packets (alive ~5 ms behind each
/// SDMA batch) cost every drafter-round dispatch ~+10 us; `word` = fast 13/13 boots vs `layer`
/// slow 12/12, bit-exact (rows 16/16, kla 25/25, API 25/25 x 8 boots, inserts identical),
/// BetterBench --quick 4 + 4 single 112.6 -> 116.1, conc-8 259.5 -> 269.0. `layer` = the control.**
/// `auto` = `word` on the one-stream mover, the mover's own marks on two-queue / hsa-lane.
pub const DEFAULT_XTIER_MOVER_MARKS: &str = "auto";
/// FN63b: the HSA lane's engine choice (`--xtier-hsa-engines`).
pub const DEFAULT_XTIER_HSA_ENGINES: &str = "auto";
/// FN71c: the boot VRAM fill's route (`--xtier-boot-fill`): `pool` = drive -> the card's P2PDMA
/// pool (a ring sized to `p2pmem/size`) -> an SDMA copy into each VRAM slot, no host DRAM on the
/// path, refused by name when the card has no pool; `host` = drive -> a pinned stage (sha256
/// checked) -> H2D, the pre-FN71c law and THE CONTROL; `auto` = `pool` when the card's pool opens,
/// else `host` with the reason on the `fnfill` line. The RAM L2 half is drive -> RAM in every
/// form (host-bound by design). **`auto` IS THE DEFAULT since FN71c** (2026-10-08, hold h1, both
/// cards, 210 W read, one served boot per arm): the VRAM half 4.25 / 4.74 s -> 3.00 / 3.00 s a rank
/// (3.4-3.8 -> 5.35 GB/s), the host DRAM probe beside it 27.2 -> 47.2 GB/s (alone 51.0), every
/// filled slot read back equal (6,133 / 6,133 a rank) + `fnslots` 0 differ + API ids 25/25;
/// `host` = the control.
pub const DEFAULT_XTIER_BOOT_FILL: &str = "auto";
/// FN71c: where a container finds each card's `p2pmem/allocate` (sysfs is read-only there): the
/// launcher bind-mounts it rw at `<dir>/<bdf>`; absent = sysfs in place (a host process).
pub const DEFAULT_XTIER_P2P_DIR: &str = "/saber-p2p";
/// FN76: THE POOL AS SLOTS -- after the boot fill the card's P2PDMA pool (minus
/// `--xtier-pool-slots-reserve-mib`) becomes VRAM expert slots, addressed through the pool BO
/// imported into HIP (`HipThread::import_dmabuf`); lent at boot, never reclaimed. `auto` = on
/// when the pool opens (the reason printed otherwise), `on` = refused without the pool, `off` =
/// the control (the pool idle after the fill). Bit-exact (placement only). **`auto` IS THE
/// DEFAULT since FN76** (2026-10-08, hold h1 + h2, both cards, 210 W read): +820 VRAM slots a
/// rank at a 2 GiB pool, row sha 16/16 + kla 25/25 + every slot's sha256; `off` = the control.
pub const DEFAULT_XTIER_POOL_SLOTS: &str = "auto";
/// FN107 (X1, FN97's row, FLASHNEXT-PLAN §15 E6): THE POOL SLOTS EXCLUSIVE
/// (`--xtier-pool-exclusive auto|on|off`) -- the pool's slots are never reclaimed, so they count
/// as STATIC for the RAM L2's complement: their records keep NO RAM copy (the L2 shrinks by the
/// pool's slots, -2.0 GiB pinned a rank at a 2 GiB pool) and a victim leaving a pool slot is
/// WRITTEN BACK to RAM through the spare (FN8's swap, a static slot's path; `excl_writebacks`).
/// `auto` = exclusive whenever the pool slots are planned (the pool present at plan time, the
/// slots on, not the valve's bounce), `on` = refused otherwise, `off` = FN76's inclusive pool
/// (every record keeps its RAM shadow: the control). The other lent holders (arena tail / top,
/// sequence pools, vision, KV elastic, buffers) are reclaimed and KEEP their shadows. **`auto` IS
/// THE DEFAULT since FN107** (2026-10-09, the user's word "yes to flip 1, 4").
pub const DEFAULT_XTIER_POOL_EXCLUSIVE: &str = "auto";
/// FN76b: THE NARROW ARENA'S TOP LENT AT DECODE (`--xtier-lend-arena-top on|off`, an ARM): under
/// the two-layout arena borrow every pass of <= `--xtier-lend-arena-top-rows` rows (every decode /
/// verify / drafter pass and a short prefill chunk) runs its scratch below a boundary B (the
/// largest such pass's host-emitted scratch + `--xtier-lend-arena-top-margin-mib`) and the narrow
/// scratch past B is the tier's lendable slots; a prefill pass of more rows RECLAIMS them first
/// (copy stream drained, every lent record flipped to its RAM shadow: dropped, no copy) and the
/// next pass of <= the rows lends them again. `auto` = on where it applies (the two-layout borrow
/// or none; it stands down BY NAME under `--fn-borrow-cut`, `--fn-mixed`, the drafter's wide-capture
/// control, the one-process form and the poison arms), `on` = those refuse, `off` = the control.
/// **`auto` IS THE DEFAULT since FN76b** (2026-10-09, hold h1, both cards, 210 W read): +197 VRAM
/// slots a rank at the served boot, bytes identical (row sha 18/18 + 2/2, kla 25/25, API greedy
/// 25/25 at conc 1 and 8 on 8 boots, seeded 20/20, every slot's sha256), BetterBench --quick 4 + 4:
/// single TTFT p50 110.5 -> 107.0 ms, conc-8 279.2 -> 287.3 tok/s, single tied (120.6 / 120.4).
pub const DEFAULT_XTIER_LEND_ARENA_TOP: &str = "auto";
/// FN76b: the largest prefill chunk (rows) that keeps the arena's top lent (the census's pick).
pub const DEFAULT_XTIER_LEND_ARENA_TOP_ROWS: u32 = 256;
/// FN76b: MiB of margin above the host-emitted boundary (the drafter chain's tail and the pick
/// take scratch the dry runs do not emit).
pub const DEFAULT_XTIER_LEND_ARENA_TOP_MARGIN_MIB: u64 = 32;
/// FN96: the idle sequence slots' state regions lent to the tier at decode (`auto|on|off`): `auto`
/// = THE DEFAULT since FN96 (2026-10-09; `off` = the control).
pub const DEFAULT_XTIER_LEND_SEQ_POOLS: &str = "auto";
/// FN100: the row-scaled buffers' tails (`fn.buffers`: the embedding / PLE staging rows, the
/// route-id log and the MTP hidden past the heads of every pass of at most
/// `--xtier-lend-arena-top-rows` rows) lent to the tier at decode (`auto|on|off`): `off` = THE
/// CONTROL and the default until FN100's gate passes.
pub const DEFAULT_XTIER_LEND_BUFFERS_TOP: &str = "off";
/// FN96: decode steps a sequence slot stays free before its region is lent (the churn guard: a
/// slot a queued request takes at once is never lent and reclaimed).
pub const DEFAULT_XTIER_LEND_SEQ_POOLS_HOLD: u32 = 32;
/// FN59b: the tier's end-of-step planning (`--xtier-plan`): `sync` = the pre-FN59b law and THE
/// CONTROL (the full LRFU victim sort, the copy batch issued by a blocking `run_batch`); `fast` = the cheap victim selection
/// (each slot scored once, only the first `misses` keys ranked: the same victims, placement-
/// identical); `async` = `fast` + the copy batch submitted to the HIP thread's background lane
/// (FN59's `submit_batch`; FN54c: both movers). **`async` is the DEFAULT since FN59b
/// (2026-10-05, hold h1: bytes 16/16 + 25/25 + API 8/8 boots, served tier end b1 0.84 -> 0.26 ms /
/// b8 2.15 -> 0.69, BetterBench --quick 4 + 4 interleaved conc-8 190.3 -> 193.8, single tied).**
pub const DEFAULT_XTIER_PLAN: &str = "async";
/// FN18: the LRFU decay of the VRAM victim ranking (0 = FN8's recency-only LRU). FN18c (2026-09-29,
/// hold 2's paired win, with the exact ledger and the pack's real-traffic profile): 0.98 is the
/// default -- the host sweep's held-out winner (LEDGER FN18); `--xtier-policy-lru` (or
/// `--xtier-lrfu-lambda 0`) = FN8's LRU, the control.
pub const DEFAULT_XTIER_LRFU_LAMBDA: f64 = 0.98;
/// FN36: where a promotion's VRAM victim goes: `ram` = FN8's D2H demotion (the default until the
/// paired law), `none` = the demotion-free law (the victim re-read from NVMe under the next step,
/// the swap published a step later; `saber_mem::expert_tier::Demote`).
pub const DEFAULT_XTIER_DEMOTE: &str = "ram";
/// FN18: the admission window in steps (0 = every RAM route of a non-read-through layer is a
/// promotion candidate, FN8's law).
pub const DEFAULT_XTIER_ADMIT_WINDOW: u32 = 0;
/// FN39: the LRFU admission margin (a RAM route is promoted only when its LRFU score beats the
/// victim's by this much; 0 = off, every capped miss is promoted -- today's law).
pub const DEFAULT_XTIER_ADMIT_MARGIN: f64 = 0.0;
/// FN39: RAM uses before a promotion (0 = off).
pub const DEFAULT_XTIER_ADMIT_USES: u32 = 0;

/// The memory plane's flags (flatten into a subcommand).
#[derive(clap::Args, Debug, Clone, PartialEq)]
pub struct XtierFlags {
    /// VRAM expert slots per rank. Default: the byte ledger's remainder after the weights, the
    /// runtime and the staging ring (`fnbudget` boot line; 7,301 at the defaults, less the ring).
    #[arg(long)]
    pub xtier_vram_slots: Option<u32>,
    /// Pinned RAM L2 slots per rank. Default: the host budget's (the complement when it fits,
    /// which makes the NVMe leave the expert step path; `fnhost` boot line).
    #[arg(long)]
    pub xtier_ram_slots: Option<u32>,
    /// Inserts per layer per step (the LRU's promotion cap).
    #[arg(long, default_value_t = DEFAULT_XTIER_MAX_INSERTS)]
    pub xtier_max_inserts: u32,
    /// FN18: per decode bucket insert caps, `bucket:cap,..` (0 = no promotions at that bucket).
    #[arg(long, default_value = DEFAULT_XTIER_MAX_INSERTS_BY_BUCKET)]
    pub xtier_max_inserts_by_bucket: String,
    /// FN54: per decode bucket promotion BYTE budget a step, `bucket:MiB,..` (shared round-robin
    /// over the layers under each layer's insert cap; empty = none).
    #[arg(long, default_value = DEFAULT_XTIER_PROMOTE_BUDGET_MIB_BY_BUCKET)]
    pub xtier_promote_budget_mib_by_bucket: String,
    /// FN54: second-chance insertion -- promote a record only on its second RAM miss within W
    /// steps: `W` (every step) or `bucket:W,..` (those decode buckets); 0 = off.
    #[arg(long, default_value = DEFAULT_XTIER_SECOND_CHANCE)]
    pub xtier_second_chance: String,
    /// FN54b: `single` (one in-order copy stream) or `two-queue` (promotions on an urgent stream,
    /// demotions on a background stream; the same bytes, the link's two directions at once).
    /// FN63b: `hsa-lane` = the copies on ROCr's SDMA engines (H2D and D2H on two engines),
    /// ordered by the signal law (`saber_mem::device_mover::SignalPlanner`).
    #[arg(long, default_value = DEFAULT_XTIER_MOVER, value_parser = ["single", "two-queue", "hsa-lane"])]
    pub xtier_mover: String,
    /// FN72b: the one-stream mover's landed marks -- `layer` (a HIP event per layer batch,
    /// today), `word` (an SDMA write of the layer's device word, the gate waits the word: no
    /// HIP event on the copy stream) or `step` (one event a step at the first gate). Bit-exact
    /// (ordering only).
    #[arg(long, default_value = DEFAULT_XTIER_MOVER_MARKS, value_parser = ["auto", "layer", "word", "step"])]
    pub xtier_mover_marks: String,
    /// FN71c: the boot VRAM fill's route -- `auto` (the pool when it opens, else `host`),
    /// `pool` (drive -> the P2PDMA pool -> the slot; refused without a pool) or `host` (drive ->
    /// pinned RAM -> H2D: the control). Bit-exact (bytes).
    #[arg(long, default_value = DEFAULT_XTIER_BOOT_FILL, value_parser = ["auto", "pool", "host"])]
    pub xtier_boot_fill: String,
    /// FN71c (an ARM): `on` = the valve's bounce (FN53b: a prefill pass's `Fetch` tail, read
    /// ahead and on demand) lives in the card's P2PDMA pool -- drive -> VRAM, the step's kernels
    /// read it in place (no host DRAM, no UVA); needs `--fn-mem-valve on` and the pool. `off` =
    /// the pinned-RAM bounce (the control). Counters: the `fnp2p` end line.
    #[arg(long, default_value = "off", value_parser = ["on", "off"])]
    pub xtier_nvme_p2p: String,
    /// FN71c: the directory holding each card's `p2pmem/allocate` as `<dir>/<bdf>` (a container's
    /// bind mount); a missing file = sysfs in place.
    #[arg(long, default_value = DEFAULT_XTIER_P2P_DIR)]
    pub xtier_p2p_dir: std::path::PathBuf,
    /// FN76: the P2PDMA pool as VRAM expert slots after the boot fill (`auto|on|off`; see
    /// [`DEFAULT_XTIER_POOL_SLOTS`]); refused with `--xtier-nvme-p2p on` (the valve's bounce
    /// owns the pool).
    #[arg(long, default_value = DEFAULT_XTIER_POOL_SLOTS, value_parser = ["auto", "on", "off"])]
    pub xtier_pool_slots: String,
    /// FN107: the pool's slots EXCLUSIVE (no RAM copy; counted static for the L2's complement):
    /// `auto|on|off`, see [`DEFAULT_XTIER_POOL_EXCLUSIVE`]; `off` = FN76's inclusive pool.
    #[arg(long, default_value = DEFAULT_XTIER_POOL_EXCLUSIVE, value_parser = ["auto", "on", "off"])]
    pub xtier_pool_exclusive: String,
    /// FN76: MiB at the pool's top kept OUT of the slots (a landing zone for a later reader).
    #[arg(long, default_value_t = 0)]
    pub xtier_pool_slots_reserve_mib: u64,
    /// FN76b: the narrow arena's top lent to the expert tier at decode (`on|off`; see
    /// [`DEFAULT_XTIER_LEND_ARENA_TOP`]); needs the two-layout arena borrow.
    #[arg(long, default_value = DEFAULT_XTIER_LEND_ARENA_TOP, value_parser = ["auto", "on", "off"])]
    pub xtier_lend_arena_top: String,
    /// FN76b: a prefill chunk of more rows reclaims the arena's top; at most this many keep it
    /// lent (the boundary is sized for this chunk).
    #[arg(long, default_value_t = DEFAULT_XTIER_LEND_ARENA_TOP_ROWS)]
    pub xtier_lend_arena_top_rows: u32,
    /// FN76b: MiB of margin above the host-emitted boundary.
    #[arg(long, default_value_t = DEFAULT_XTIER_LEND_ARENA_TOP_MARGIN_MIB)]
    pub xtier_lend_arena_top_margin_mib: u64,
    /// FN96: the state regions of the sequence slots no request holds, lent to the expert tier
    /// at decode as whole slots (`auto|on|off`; see [`DEFAULT_XTIER_LEND_SEQ_POOLS`]); an
    /// admission reclaims its slot's region first (records dropped to their RAM shadows).
    #[arg(long, default_value = DEFAULT_XTIER_LEND_SEQ_POOLS, value_parser = ["auto", "on", "off"])]
    pub xtier_lend_seq_pools: String,
    /// FN100: the row-scaled buffers' tails (`fn.buffers` past the heads of every pass of at most
    /// `--xtier-lend-arena-top-rows` rows) lent to the expert tier at decode as whole slots
    /// (`auto|on|off`; see [`DEFAULT_XTIER_LEND_BUFFERS_TOP`]); a write past the heads (a wider
    /// pass's staging) reclaims them first (records dropped to their RAM shadows).
    #[arg(long, default_value = DEFAULT_XTIER_LEND_BUFFERS_TOP, value_parser = ["auto", "on", "off"])]
    pub xtier_lend_buffers_top: String,
    /// FN96: decode steps a sequence slot must stay free before its region is lent.
    #[arg(long, default_value_t = DEFAULT_XTIER_LEND_SEQ_POOLS_HOLD)]
    pub xtier_lend_seq_pools_hold: u32,
    /// FN63b: the HSA lane's engines -- `auto` (the lowest available engine for H2D, the lowest
    /// other for D2H), `runtime` (ROCr picks: FN63's unchained form) or `H2D:D2H` indices.
    #[arg(long, default_value = DEFAULT_XTIER_HSA_ENGINES)]
    pub xtier_hsa_engines: String,
    /// FN63b (DIAGNOSTIC): the copy-batch census (`fnmover63b` line: device span per batch, the
    /// HSA lane's issue->landed time); two timing events a batch on the HIP mover, ROCr's copy
    /// profiling on the lane. Off in served cells.
    #[arg(long)]
    pub xtier_mover_census: bool,
    /// FN59b: `sync` (today), `fast` (the cheap victim selection, placement-identical) or `async`
    /// (`fast` + the copy batches submitted to the HIP thread's background lane).
    #[arg(long, default_value = DEFAULT_XTIER_PLAN, value_parser = ["sync", "fast", "async"])]
    pub xtier_plan: String,
    /// FN18: LRFU victim ranking decay (0 = recency-only LRU).
    #[arg(long, default_value_t = DEFAULT_XTIER_LRFU_LAMBDA)]
    pub xtier_lrfu_lambda: f64,
    /// FN18: TinyLFU-style admission window in steps (0 = off).
    #[arg(long, default_value_t = DEFAULT_XTIER_ADMIT_WINDOW)]
    pub xtier_admit_window: u32,
    /// FN39: LRFU admission margin -- promote only when the route's LRFU score exceeds the
    /// victim's by at least M (0 = off; needs LRFU).
    #[arg(long, default_value_t = DEFAULT_XTIER_ADMIT_MARGIN)]
    pub xtier_admit_margin: f64,
    /// FN39: promote only after U routed steps since the record entered RAM (0 = off).
    #[arg(long, default_value_t = DEFAULT_XTIER_ADMIT_USES)]
    pub xtier_admit_uses: u32,
    /// FN36: `ram` (FN8's D2H victim copy) or `none` (the demotion-free law: NVMe refill).
    #[arg(long, default_value = DEFAULT_XTIER_DEMOTE, value_parser = ["ram", "none"])]
    pub xtier_demote: String,
    /// FN18: the control policy -- FN8's LRU (recency-only victims, no admission filter).
    #[arg(long)]
    pub xtier_policy_lru: bool,
    /// FN17, the measured-losing ARM: the deferred publication law (2 x max-inserts spare RAM slots
    /// a layer; nothing a placement points at is written by a copy; published at the next step's
    /// fence). The default is FN8's law (the swap through one spare, published at once, each
    /// layer's graph gated on its copies).
    #[arg(long, conflicts_with = "no_xtier_defer")]
    pub xtier_defer: bool,
    /// FN17: FN8's publication law (the default; this flag names it).
    #[arg(long)]
    pub no_xtier_defer: bool,
    /// The read-through fraction of a layer's VRAM slots.
    #[arg(long, default_value_t = DEFAULT_XTIER_READ_THROUGH)]
    pub xtier_read_through: f64,
    /// NVMe → RAM prefetch reads per step (0 = off, the default).
    #[arg(long, default_value_t = DEFAULT_XTIER_PREFETCH)]
    pub xtier_prefetch: u32,
    /// The routing profile the warm start ranks by (`fn7-route-profile/1`). Default: the pack's.
    #[arg(long)]
    pub xtier_profile: Option<PathBuf>,
    /// No profile: the uniform split in expert-id order (the warm-start control).
    #[arg(long, conflicts_with = "xtier_profile")]
    pub no_xtier_profile: bool,
    /// FN43c: prefill chunks of at least this many rows stage their RAM-resident experts in VRAM;
    /// below it the ring is inert (`--xtier-stage-above-rows` is the old name).
    /// FN44: default by the boot's widest chunk ([`STAGE_BY_WIDTH`]: 2,048 at c2048, 2,049 wider).
    #[arg(long, alias = "xtier-stage-above-rows", value_parser = clap::value_parser!(u32).range(1..))]
    pub xtier_stage_min_rows: Option<u32>,
    /// FN122e: the ring's next-layer prefetch for a wide boot's narrow passes -- `off` (stage only
    /// chunks above 2,048 rows: today, the control) or `band` (stage chunks of >= 1,024 rows too).
    /// `--xtier-stage-min-rows` overrides it.
    #[arg(long = "fn-prefill-prefetch", default_value = DEFAULT_FN_PREFILL_PREFETCH, value_parser = ["off", "band"])]
    pub fn_prefill_prefetch: String,
    /// LR-E (diagnostic ARM, default off): per staged layer, two timing events around the compute
    /// stream's wait for the layer's ring arrivals; `fnringpass` then prints the pass's compute
    /// wait (the compute stream idle on the link) and its worst layer. Off = today's line, no
    /// event recorded.
    #[arg(long = "fn-ring-idle")]
    pub fn_ring_idle: bool,
    /// LR-H: the staging ring's ORDER KEY -- `profile` (the warm profile's route count, then the
    /// LRFU score: today, the default and the control) or `served` (an ARM: the decayed count of
    /// served staged passes that routed the record first, then the profile key). It moves which
    /// RAM records take the ring's slots, never a byte a GEMM reads; every `fnringpass` line
    /// prices BOTH keys (the other one counterfactually, host-only).
    #[arg(long = "xtier-stage-order", default_value = "profile", value_parser = ["profile", "served"])]
    pub xtier_stage_order: String,
    /// Staging slots per ring half. FN44: default by the boot's widest chunk ([`STAGE_BY_WIDTH`]:
    /// 48 at c2048, 128 wider).
    #[arg(long)]
    pub xtier_stage_slots: Option<u32>,
    /// FN43: the ring's layer buffers (1 or 2).
    #[arg(long, default_value_t = DEFAULT_XTIER_STAGE_BUFFERS, value_parser = clap::value_parser!(u32).range(1..=2))]
    pub xtier_stage_buffers: u32,
    /// The staging ring (`--xtier-stage-buffers` x `--xtier-stage-slots`, taken out of the LRU's
    /// slots) for chunks of at least `--xtier-stage-min-rows` rows. Explicit = refused where it
    /// cannot run (the default stands down there instead).
    #[arg(long, conflicts_with = "no_xtier_stage")]
    pub xtier_stage: bool,
    /// THE CONTROL: no staging ring -- every RAM-resident expert is read over UVA in place at
    /// every chunk size (the pre-FN43c law).
    #[arg(long)]
    pub no_xtier_stage: bool,
    /// PLE RAM row cache per rank, MiB.
    #[arg(long, default_value_t = DEFAULT_PLE_CACHE_MIB)]
    pub ple_cache_mib: u64,
    /// PLE reader queue depth for a prefill chunk.
    #[arg(long, default_value_t = DEFAULT_PLE_QD_PREFILL)]
    pub ple_qd_prefill: usize,
    /// PLE reader queue depth for decode tokens.
    #[arg(long, default_value_t = DEFAULT_PLE_QD_DECODE)]
    pub ple_qd_decode: usize,
}

/// The staging law, resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageLaw {
    /// Chunks of at least this many rows stage.
    pub above_rows: u32,
    /// Slots per ring half.
    pub slots: u32,
    /// FN43: layer buffers (1 or 2).
    pub buffers: u32,
}

impl StageLaw {
    /// Whether a chunk of `rows` rows stages (the which-path decision; counted by the caller).
    #[must_use]
    pub fn stages(&self, rows: u32) -> bool {
        rows >= self.above_rows
    }
    /// Ring slots the ledger reserves (both halves).
    #[must_use]
    pub fn ring_slots(&self) -> u32 {
        self.buffers * self.slots
    }
}

/// The memory plane's parameters after the flags.
#[derive(Debug, Clone, PartialEq)]
pub struct TierLaw {
    /// The tier's parameters (slots filled by [`XtierFlags::resolve`]).
    pub tier: TierParams,
    /// The staging law; `None` under `--no-xtier-stage`.
    pub stage: Option<StageLaw>,
    /// Whether the warm start uses a profile.
    pub profile: bool,
    /// PLE cache rows (320 B bf16 rows).
    pub ple_cache_rows: u32,
    /// PLE queue depths (prefill, decode).
    pub ple_qd: (usize, usize),
    /// FN18: the insert cap by decode bucket.
    pub inserts: saber_mem::route_trace::InsertLaw,
}

impl XtierFlags {
    /// FN54b: whether the two-queue mover is armed.
    #[must_use]
    pub fn two_queue(&self) -> bool {
        self.xtier_mover == "two-queue"
    }

    /// FN63b: whether the HSA SDMA copy lane is armed (`--xtier-mover hsa-lane`).
    #[must_use]
    pub fn hsa_lane(&self) -> bool {
        self.xtier_mover == "hsa-lane"
    }

    /// FN72b: the landed-mark law, parsed (`single` mover only; refused by name otherwise).
    pub fn mover_marks(&self) -> anyhow::Result<saber_mem::device_mover::MarkLaw> {
        // FN72b: `auto` (the default) = `word` on the one-stream mover, the mover's own marks
        // on the others (an explicit word/step there is refused)
        if self.xtier_mover_marks == "auto" {
            return Ok(if self.xtier_mover == "single" {
                saber_mem::device_mover::MarkLaw::Word
            } else {
                saber_mem::device_mover::MarkLaw::Layer
            });
        }
        let law: saber_mem::device_mover::MarkLaw = self
            .xtier_mover_marks
            .parse()
            .map_err(|e: String| anyhow::anyhow!("--xtier-mover-marks: {e}"))?;
        anyhow::ensure!(
            law == saber_mem::device_mover::MarkLaw::Layer || self.xtier_mover == "single",
            "--xtier-mover-marks {}: the one-stream mover only (not --xtier-mover {})",
            self.xtier_mover_marks,
            self.xtier_mover
        );
        Ok(law)
    }

    /// FN71c: the valve's bounce in the P2PDMA pool (`--xtier-nvme-p2p on`).
    #[must_use]
    pub fn nvme_p2p(&self) -> bool {
        self.xtier_nvme_p2p == "on"
    }

    /// FN76b: whether the narrow arena's top is lent at decode (`auto` / `on`), and whether a
    /// form it does not model refuses the boot (`on`) instead of standing it down (`auto`).
    #[must_use]
    pub fn lend_arena_top(&self) -> (bool, bool) {
        match self.xtier_lend_arena_top.as_str() {
            "on" => (true, true),
            "auto" => (true, false),
            _ => (false, false),
        }
    }

    /// FN96: whether the idle sequence slots' regions are lent (`auto` / `on`), and whether a
    /// form it does not model refuses the boot (`on`) instead of standing it down (`auto`).
    #[must_use]
    pub fn lend_seq_pools(&self) -> (bool, bool) {
        match self.xtier_lend_seq_pools.as_str() {
            "on" => (true, true),
            "auto" => (true, false),
            _ => (false, false),
        }
    }

    /// FN100: whether the buffers' tails are lent (`auto` / `on`), and whether a form it does not
    /// model refuses the boot (`on`) instead of standing it down (`auto`).
    #[must_use]
    pub fn lend_buffers_top(&self) -> (bool, bool) {
        match self.xtier_lend_buffers_top.as_str() {
            "on" => (true, true),
            "auto" => (true, false),
            _ => (false, false),
        }
    }

    /// FN76: whether the pool becomes expert slots (`auto` / `on`), and whether a refusal is
    /// fatal (`on`).
    #[must_use]
    pub fn pool_slots(&self) -> (bool, bool) {
        match self.xtier_pool_slots.as_str() {
            "on" => (true, true),
            "auto" => (true, false),
            _ => (false, false),
        }
    }

    /// FN107: whether the pool slots are made exclusive (`auto` / `on`), and whether a boot that
    /// cannot make them so is refused (`on`).
    #[must_use]
    pub fn pool_exclusive(&self) -> (bool, bool) {
        match self.xtier_pool_exclusive.as_str() {
            "on" => (true, true),
            "auto" => (true, false),
            _ => (false, false),
        }
    }

    /// FN71c: whether the boot fill tries the pool (`auto` / `pool`), and whether a refusal is
    /// fatal (`pool`).
    #[must_use]
    pub fn boot_fill_pool(&self) -> (bool, bool) {
        match self.xtier_boot_fill.as_str() {
            "pool" => (true, true),
            "auto" => (true, false),
            _ => (false, false),
        }
    }

    /// FN63b: the HSA lane's engine choice, parsed (refused by name).
    pub fn hsa_engines(&self) -> anyhow::Result<saber_hsa::EngineChoice> {
        self.xtier_hsa_engines
            .parse()
            .map_err(|e: String| anyhow::anyhow!("--xtier-hsa-engines: {e}"))
    }

    /// FN59b: the cheap victim selection (`--xtier-plan fast|async`).
    #[must_use]
    pub fn plan_fast(&self) -> bool {
        self.xtier_plan != "sync"
    }

    /// FN59b: the copy batches go through the HIP thread's background lane (`--xtier-plan async`).
    #[must_use]
    pub fn plan_submit(&self) -> bool {
        self.xtier_plan == "async"
    }

    /// FN43c: whether the staging ring is on. `--xtier-stage` / `--no-xtier-stage` decide when
    /// given; otherwise [`DEFAULT_XTIER_STAGE`], which STANDS DOWN where the ring cannot run:
    /// `default_ok` false (the caller's: TP=1, the one-process form, the poison arms, the FN30b
    /// alias, the FN8 bench) or the tier's deferred / demotion-free publication laws (the ring
    /// lends to FN8's law only). An explicit `--xtier-stage` there is refused by the boot instead.
    #[must_use]
    pub fn stage_on(&self, default_ok: bool) -> bool {
        if self.xtier_stage {
            true
        } else if self.no_xtier_stage {
            false
        } else {
            DEFAULT_XTIER_STAGE
                && default_ok
                && !self.xtier_defer
                && !DEFAULT_XTIER_DEFER
                && self.xtier_demote != "none"
        }
    }

    /// FN44: the stage-min-rows rule at a boot whose widest chunk is `width` (the flag, else
    /// [`stage_default`]).
    #[must_use]
    pub fn stage_min_rows(&self, width: u32) -> u32 {
        self.xtier_stage_min_rows.unwrap_or_else(|| {
            let (rows, _) = stage_default(width);
            // FN122e: the band form stages the wide boot's narrow chunks of >= 1,024 rows too
            if self.fn_prefill_prefetch == "band" && width > NARROW_WIDTH {
                rows.min(PREFETCH_BAND_MIN_ROWS)
            } else {
                rows
            }
        })
    }

    /// FN44: slots per ring half at a boot whose widest chunk is `width` (the flag, else
    /// [`stage_default`]).
    #[must_use]
    pub fn stage_slots(&self, width: u32) -> u32 {
        self.xtier_stage_slots
            .unwrap_or_else(|| stage_default(width).1)
    }

    /// LR-H: the ring's order key (`--xtier-stage-order`).
    #[must_use]
    pub fn stage_key(&self) -> saber_mem::stage_ring::StageKey {
        saber_mem::stage_ring::StageKey::parse(&self.xtier_stage_order).unwrap_or_default()
    }

    /// FN43: the staging ring's slots (0 when [`Self::stage_on`] is false) at a boot whose widest
    /// chunk is `width`.
    #[must_use]
    pub fn stage_ring_slots(&self, default_ok: bool, width: u32) -> u32 {
        if self.stage_on(default_ok) {
            self.xtier_stage_buffers * self.stage_slots(width)
        } else {
            0
        }
    }

    /// Resolve against the ledger's remainder (`ledger_slots`, BEFORE the ring) and the host
    /// budget's default L2 (`host_ram_slots`). The ring comes out of the remainder; `default_ok`
    /// as in [`Self::stage_on`]; `width` = the boot's widest chunk ([`STAGE_BY_WIDTH`]).
    pub fn resolve(
        &self,
        ledger_slots: u32,
        host_ram_slots: u32,
        default_ok: bool,
        width: u32,
    ) -> Result<TierLaw, String> {
        let stage = self.stage_on(default_ok).then_some(StageLaw {
            above_rows: self.stage_min_rows(width),
            slots: self.stage_slots(width),
            buffers: self.xtier_stage_buffers,
        });
        let ring = stage.map_or(0, |s| s.ring_slots());
        if ring >= ledger_slots {
            return Err(format!(
                "the staging ring ({ring} slots) would take all {ledger_slots} ledger slots"
            ));
        }
        let vram = self.xtier_vram_slots.unwrap_or(ledger_slots - ring);
        if vram + ring > ledger_slots {
            return Err(format!(
                "--xtier-vram-slots {vram} + ring {ring} > the ledger's {ledger_slots}"
            ));
        }
        if !(self.xtier_admit_margin >= 0.0 && self.xtier_admit_margin.is_finite()) {
            return Err(format!(
                "--xtier-admit-margin {} (a finite score gap >= 0)",
                self.xtier_admit_margin
            ));
        }
        if !(0.0..=1.0).contains(&self.xtier_read_through) || self.xtier_max_inserts == 0 {
            return Err(format!(
                "--xtier-read-through {} / --xtier-max-inserts {}",
                self.xtier_read_through, self.xtier_max_inserts
            ));
        }
        Ok(TierLaw {
            tier: TierParams {
                vram_slots: vram,
                ram_slots: self.xtier_ram_slots.unwrap_or(host_ram_slots),
                read_through: self.xtier_read_through,
                max_inserts: self.xtier_max_inserts,
                prefetch_per_step: self.xtier_prefetch,
                defer_spares: if self.no_xtier_defer || !(self.xtier_defer || DEFAULT_XTIER_DEFER) {
                    0
                } else {
                    2 * self.xtier_max_inserts * XTIER_LAYERS
                },
                lrfu_lambda: if self.xtier_policy_lru {
                    0.0
                } else {
                    self.xtier_lrfu_lambda
                },
                admit_window: if self.xtier_policy_lru {
                    0
                } else {
                    self.xtier_admit_window
                },
                admit_margin: if self.xtier_policy_lru {
                    0.0
                } else {
                    self.xtier_admit_margin
                },
                admit_uses: self.xtier_admit_uses,
                // FN63b: the HSA lane's slot law is FN54b's: the ring lets a demotion run ahead
                spare_ring: if self.two_queue() || self.hsa_lane() {
                    TWO_QUEUE_SPARES
                } else {
                    0
                },
                demote: if self.xtier_demote == "none" {
                    Demote::None
                } else {
                    Demote::Ram
                },
                refill_spares: if self.xtier_demote == "none" {
                    2 * self.xtier_max_inserts * XTIER_LAYERS
                } else {
                    0
                },
                ..TierParams::default()
            },
            stage,
            profile: !self.no_xtier_profile,
            ple_cache_rows: ((self.ple_cache_mib << 20) / 320) as u32,
            ple_qd: (self.ple_qd_prefill, self.ple_qd_decode),
            inserts: saber_mem::route_trace::InsertLaw::parse(
                self.xtier_max_inserts,
                &self.xtier_max_inserts_by_bucket,
            )
            .map_err(|e| format!("--xtier-max-inserts-by-bucket: {e}"))?
            .with_budget_mib(&self.xtier_promote_budget_mib_by_bucket)
            .map_err(|e| format!("--xtier-promote-budget-mib-by-bucket: {e}"))?
            .with_second_chance(&self.xtier_second_chance)
            .map_err(|e| format!("--xtier-second-chance: {e}"))?,
        })
    }
}

impl TierLaw {
    /// The law's boot line (beside FN7's `fnbudget` / `fnhost` / `xtier` / `pletier` lines).
    #[must_use]
    pub fn boot_line(&self, rank: u32) -> String {
        let t = &self.tier;
        format!(
            "xtierlaw rank{rank}: vram {} ram {} | policy {} | max-inserts {}{} read-through {:.2} prefetch {} | {} | \
             profile {} | ple cache {} rows qd {}/{} | mover: one ordered copy stream, per-layer \
             landed marks, consumer fence | publication {}",
            t.vram_slots,
            t.ram_slots,
            match (t.lrfu_lambda > 0.0, t.admit_window > 0) {
                (false, false) => "LRU (FN8)".to_string(),
                (l, a) => format!(
                    "{}{}",
                    if l { format!("LRFU lambda {}", t.lrfu_lambda) } else { "LRU".into() },
                    if a { format!(" + admission window {}", t.admit_window) } else { String::new() }
                ),
            } + &if t.admit_margin > 0.0 || t.admit_uses > 0 {
                format!(" + admit margin {} uses {}", t.admit_margin, t.admit_uses)
            } else {
                String::new()
            } + &{
                let l = &self.inserts;
                if l.sc_default == 0 && l.sc_by_bucket.is_empty() {
                    String::new()
                } else {
                    format!(
                        " + second chance {}{}",
                        l.sc_default,
                        l.sc_by_bucket
                            .iter()
                            .map(|(b, w)| format!(" {b}:{w}"))
                            .collect::<String>()
                    )
                }
            },
            t.max_inserts,
            if self.inserts.by_bucket.is_empty() {
                String::new()
            } else {
                format!(
                    " (by bucket {})",
                    self.inserts
                        .by_bucket
                        .iter()
                        .map(|(b, c)| format!("{b}:{c}"))
                        .collect::<Vec<_>>()
                        .join(",")
                )
            } + &if self.inserts.budget_by_bucket.is_empty() {
                String::new()
            } else {
                format!(
                    " promote budget MiB/step {}",
                    self.inserts
                        .budget_by_bucket
                        .iter()
                        .map(|(b, c)| format!("{b}:{}", c >> 20))
                        .collect::<Vec<_>>()
                        .join(",")
                )
            },
            t.read_through,
            t.prefetch_per_step,
            match self.stage {
                Some(s) => format!(
                    "stage >= {} rows into a ring of {} x {} slots",
                    s.above_rows, s.buffers, s.slots
                ),
                None => "stage off (UVA in place at every chunk)".into(),
            },
            if self.profile { "on" } else { "OFF (uniform)" },
            self.ple_cache_rows,
            self.ple_qd.0,
            self.ple_qd.1,
            if t.refill_spares > 0 {
                format!(
                    "DEMOTION-FREE (FN36: victims re-read from NVMe into {} refill slots, swaps \
                     published a step later; --xtier-demote ram = FN8's D2H)",
                    t.refill_spares
                )
            } else if t.defer_spares > 0 {
                format!(
                    "DEFERRED ({} spare RAM slots: holes + spares, published at the next fence)",
                    t.defer_spares
                )
            } else {
                "at once (FN8's law, per-layer gates; --no-xtier-defer)".to_string()
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Probe {
        #[command(flatten)]
        x: XtierFlags,
    }

    fn parse(args: &[&str]) -> XtierFlags {
        Probe::try_parse_from(std::iter::once("probe").chain(args.iter().copied()))
            .expect("parses")
            .x
    }

    #[test]
    fn lrh_the_served_stage_order_is_an_arm_and_profile_is_the_default() {
        use saber_mem::stage_ring::StageKey;
        assert_eq!(parse(&[]).stage_key(), StageKey::Profile);
        assert_eq!(
            parse(&["--xtier-stage-order", "served"]).stage_key(),
            StageKey::Served
        );
        assert_eq!(
            parse(&["--xtier-stage-order", "profile"]).stage_key(),
            StageKey::Profile
        );
        assert!(Probe::try_parse_from(["probe", "--xtier-stage-order", "lrfu"]).is_err());
    }

    #[test]
    fn lre_the_ring_idle_timing_is_an_arm_off_by_default() {
        assert!(
            !parse(&[]).fn_ring_idle,
            "today's fnringpass line, no events"
        );
        assert!(parse(&["--fn-ring-idle"]).fn_ring_idle);
    }

    #[test]
    fn second_chance_at_b4_b8_is_the_default_and_0_is_the_control() {
        // FN54c: the default promotes a b4 / b8 record on its 2nd RAM miss within 8 steps; b1 and
        // prefill chunks (bucket 0) keep the old insertion; `--xtier-second-chance 0` = the control
        let d = parse(&[])
            .resolve(7_301, 4_987, true, NARROW_WIDTH)
            .unwrap();
        assert_eq!(
            (1..=8)
                .map(|b| d.inserts.second_chance(b))
                .collect::<Vec<_>>(),
            vec![0, 0, 0, 8, 0, 0, 0, 8]
        );
        assert_eq!(d.inserts.second_chance(0), 0, "prefill keeps today's law");
        let c = parse(&["--xtier-second-chance", "0"])
            .resolve(7_301, 4_987, true, NARROW_WIDTH)
            .unwrap();
        assert!((0..=8).all(|b| c.inserts.second_chance(b) == 0));
    }

    #[test]
    fn the_tier_laws_are_the_default_and_the_off_flags_are_the_controls() {
        // FN43c: the default boot runs the 48-slot ring, so the controls below start from
        // `--no-xtier-stage` (UVA in place, the LRU keeps every slot).
        let d = parse(&["--no-xtier-stage"])
            .resolve(7_301, 4_987, true, NARROW_WIDTH)
            .unwrap();
        assert_eq!(
            d.tier.vram_slots, 7_301,
            "no ring under the control: the LRU keeps every slot"
        );
        assert_eq!(d.tier.ram_slots, 4_987, "the host budget's L2");
        assert_eq!(d.tier.max_inserts, DEFAULT_XTIER_MAX_INSERTS);
        assert_eq!(d.tier.read_through, DEFAULT_XTIER_READ_THROUGH);
        assert_eq!(d.tier.prefetch_per_step, 0, "prefetch is off by default");
        // FN18c: LRFU victims at lambda 0.98 are the default; FN8's LRU is the control, named
        // by `--xtier-policy-lru` (or `--xtier-lrfu-lambda 0`).
        assert_eq!(d.tier.lrfu_lambda, 0.98);
        assert_eq!(
            d.tier.admit_window, 0,
            "admission stays off (rejected, LEDGER FN18)"
        );
        let lru = parse(&["--xtier-policy-lru"])
            .resolve(7_301, 4_987, true, NARROW_WIDTH)
            .unwrap();
        assert_eq!(lru.tier.lrfu_lambda, 0.0);
        assert_eq!(
            parse(&["--xtier-lrfu-lambda", "0"])
                .resolve(7_301, 4_987, true, NARROW_WIDTH)
                .unwrap(),
            lru
        );
        assert!(d.boot_line(0).contains("LRFU lambda 0.98"));
        // FN39: the admission margin is off by default (today's law); the flags name it.
        assert_eq!((d.tier.admit_margin, d.tier.admit_uses), (0.0, 0));
        assert!(!d.boot_line(0).contains("admit margin"));
        let m = parse(&["--xtier-admit-margin", "1.5", "--xtier-admit-uses", "2"])
            .resolve(7_301, 4_987, true, NARROW_WIDTH)
            .unwrap();
        assert_eq!((m.tier.admit_margin, m.tier.admit_uses), (1.5, 2));
        assert!(m.boot_line(0).contains("admit margin 1.5 uses 2"));
        assert!(parse(&["--xtier-admit-margin=-1"])
            .resolve(7_301, 4_987, true, NARROW_WIDTH)
            .is_err());
        assert!(lru.boot_line(0).contains("LRU"));
        assert_eq!(d.stage, None, "UVA in place at every chunk is the control");
        assert!(d.profile);
        assert_eq!(
            d.ple_cache_rows,
            ((DEFAULT_PLE_CACHE_MIB << 20) / 320) as u32
        );
        // The explicit flag names the default: the ring comes out of the remainder.
        let on = parse(&["--xtier-stage"])
            .resolve(7_301, 4_987, true, NARROW_WIDTH)
            .unwrap();
        assert_eq!(
            parse(&[])
                .resolve(7_301, 4_987, true, NARROW_WIDTH)
                .unwrap(),
            on
        );
        assert_eq!(on.tier.vram_slots, 7_301 - 2 * DEFAULT_XTIER_STAGE_SLOTS);
        assert_eq!(
            on.stage,
            Some(StageLaw {
                above_rows: DEFAULT_XTIER_STAGE_MIN_ROWS,
                slots: DEFAULT_XTIER_STAGE_SLOTS,
                buffers: DEFAULT_XTIER_STAGE_BUFFERS
            })
        );
        assert!(on.stage.unwrap().stages(DEFAULT_XTIER_STAGE_MIN_ROWS));
        assert!(!on.stage.unwrap().stages(DEFAULT_XTIER_STAGE_MIN_ROWS - 1));
        assert!(
            !parse(&["--no-xtier-profile"])
                .resolve(7_301, 4_987, true, NARROW_WIDTH)
                .unwrap()
                .profile
        );
        assert!(
            Probe::try_parse_from(["probe", "--xtier-stage", "--no-xtier-stage"]).is_err(),
            "the law and its control together are refused"
        );
        // The overrides.
        let o = parse(&[
            "--xtier-max-inserts",
            "16",
            "--xtier-read-through",
            "0.25",
            "--xtier-ram-slots",
            "1024",
            "--xtier-stage",
            "--xtier-stage-min-rows",
            "4096",
        ])
        .resolve(7_301, 4_987, true, NARROW_WIDTH)
        .unwrap();
        assert_eq!(
            (o.tier.max_inserts, o.tier.read_through, o.tier.ram_slots),
            (16, 0.25, 1024)
        );
        assert!(!o.stage.unwrap().stages(2048));
        // Refusals, by name.
        assert!(parse(&["--xtier-vram-slots", "7302"])
            .resolve(7_301, 4_987, true, NARROW_WIDTH)
            .is_err());
        assert!(parse(&["--xtier-read-through", "1.5"])
            .resolve(7_301, 4_987, true, NARROW_WIDTH)
            .is_err());
        assert!(d
            .boot_line(0)
            .starts_with("xtierlaw rank0: vram 7301 ram 4987"));
    }

    #[test]
    fn the_48_slot_ring_is_the_default_and_stages_only_chunks_of_min_rows() {
        // FN43c: a plain boot = `--xtier-stage --xtier-stage-slots 48 --xtier-stage-buffers 2
        // --xtier-stage-min-rows 2048`.
        const { assert!(DEFAULT_XTIER_STAGE) };
        let d = parse(&[]);
        assert!(d.stage_on(true));
        assert_eq!(d.stage_ring_slots(true, NARROW_WIDTH), 2 * 48);
        let law = d.resolve(7_768, 4_520, true, NARROW_WIDTH).unwrap();
        let s = law.stage.expect("the ring is the default");
        assert_eq!((s.slots, s.buffers, s.above_rows), (48, 2, 2_048));
        assert_eq!(
            law.tier.vram_slots,
            7_768 - 96,
            "the ring comes out of the remainder"
        );
        assert!(law
            .boot_line(0)
            .contains("stage >= 2048 rows into a ring of 2 x 48 slots"));
        // The min-rows rule: the ring is inert below R (a 2,047-row tail, every decode step).
        assert!(s.stages(2_048) && s.stages(8_192));
        assert!(!s.stages(2_047) && !s.stages(1) && !s.stages(8));
        let r = parse(&["--xtier-stage-min-rows", "4096"])
            .resolve(7_768, 4_520, true, NARROW_WIDTH)
            .unwrap()
            .stage
            .unwrap();
        assert!(!r.stages(2_048) && r.stages(4_096));
        // The old name is an alias of the rule; 0 rows is refused (every chunk = 1).
        assert_eq!(
            parse(&["--xtier-stage-above-rows", "1"]),
            parse(&["--xtier-stage-min-rows", "1"])
        );
        assert!(Probe::try_parse_from(["probe", "--xtier-stage-min-rows", "0"]).is_err());
        // The default STANDS DOWN where the ring cannot run (no refusal) ...
        assert!(
            !d.stage_on(false),
            "TP=1 / one-process / poison / alias: UVA in place"
        );
        assert_eq!(
            d.resolve(7_768, 4_520, false, NARROW_WIDTH).unwrap().stage,
            None
        );
        assert!(!parse(&["--xtier-defer"]).stage_on(true));
        assert!(!parse(&["--xtier-demote", "none"]).stage_on(true));
        // ... while an explicit `--xtier-stage` stays on (the boot refuses it there by name).
        assert!(parse(&["--xtier-stage"]).stage_on(false));
        assert!(parse(&["--xtier-stage", "--xtier-defer"]).stage_on(true));
        // The control.
        let c = parse(&["--no-xtier-stage"]);
        assert!(!c.stage_on(true));
        assert_eq!(c.stage_ring_slots(true, NARROW_WIDTH), 0);
        assert_eq!(
            c.resolve(7_768, 4_520, true, NARROW_WIDTH)
                .unwrap()
                .tier
                .vram_slots,
            7_768
        );
    }

    #[test]
    fn fn44_the_ring_slots_and_min_rows_are_one_table_by_the_chunk_width() {
        // FN44: c2048 keeps FN43c's ring (48 slots, every 2,048-row chunk stages) ...
        assert_eq!(stage_default(2_048), (2_048, 48));
        assert_eq!(stage_default(1), (2_048, 48));
        // ... a wide rung stages only chunks ABOVE 2,048 rows, into 128 slots (FN43b's WR128).
        assert_eq!(stage_default(4_096), (2_049, 128));
        assert_eq!(stage_default(8_192), (2_049, 128));
        assert_eq!(stage_default(2_049), (2_049, 128));
        assert_eq!(stage_default(u32::MAX), (2_049, 128));
        // FN122e: the band form lowers only a wide boot's floor; the narrow boot keeps 2,048
        assert_eq!(parse(&[]).stage_min_rows(8_192), 2_049);
        let b = parse(&["--fn-prefill-prefetch", "band"]);
        assert_eq!(b.stage_min_rows(8_192), PREFETCH_BAND_MIN_ROWS);
        assert_eq!(b.stage_min_rows(2_048), 2_048);
        let b = parse(&[
            "--fn-prefill-prefetch",
            "band",
            "--xtier-stage-min-rows",
            "4096",
        ]);
        assert_eq!(b.stage_min_rows(8_192), 4_096);
        let d = parse(&[]);
        assert_eq!(d.stage_ring_slots(true, 8_192), 2 * 128);
        assert_eq!(d.stage_ring_slots(true, 2_048), 2 * 48);
        assert_eq!(
            d.stage_ring_slots(false, 8_192),
            0,
            "the default stands down"
        );
        let w = d.resolve(7_553, 4_520, true, 8_192).unwrap();
        let s = w.stage.unwrap();
        assert_eq!((s.slots, s.buffers, s.above_rows), (128, 2, 2_049));
        assert_eq!(w.tier.vram_slots, 7_553 - 256);
        assert!(!s.stages(2_048) && s.stages(2_049) && s.stages(8_192));
        assert!(!s.stages(8), "decode never stages");
        // the flags override both columns at any width
        let o = parse(&[
            "--xtier-stage-slots",
            "64",
            "--xtier-stage-min-rows",
            "4096",
        ]);
        assert_eq!((o.stage_slots(8_192), o.stage_min_rows(8_192)), (64, 4_096));
        assert_eq!((o.stage_slots(2_048), o.stage_min_rows(2_048)), (64, 4_096));
        // the table's rows are ordered and the last covers every width
        assert!(STAGE_BY_WIDTH.windows(2).all(|p| p[0].0 < p[1].0));
        assert_eq!(STAGE_BY_WIDTH[STAGE_BY_WIDTH.len() - 1].0, u32::MAX);
    }

    #[test]
    fn fn76b_the_arena_top_lend_is_the_default_and_off_is_the_control() {
        // FN76b: `--xtier-lend-arena-top` auto | on | off, `auto` by default (stands down by name
        // where it does not apply), `off` = the control; rows / margin
        let d = parse(&[]);
        assert_eq!(d.xtier_lend_arena_top, DEFAULT_XTIER_LEND_ARENA_TOP);
        assert_eq!(d.lend_arena_top(), (true, false), "auto = the default");
        assert_eq!(
            parse(&["--xtier-lend-arena-top", "off"]).lend_arena_top(),
            (false, false)
        );
        assert_eq!(
            d.xtier_lend_arena_top_rows,
            DEFAULT_XTIER_LEND_ARENA_TOP_ROWS
        );
        assert_eq!(
            d.xtier_lend_arena_top_margin_mib,
            DEFAULT_XTIER_LEND_ARENA_TOP_MARGIN_MIB
        );
        let o = parse(&[
            "--xtier-lend-arena-top",
            "on",
            "--xtier-lend-arena-top-rows",
            "512",
        ]);
        assert_eq!(o.lend_arena_top(), (true, true));
        assert_eq!(o.xtier_lend_arena_top_rows, 512);
    }

    #[test]
    fn fn96_the_seq_pool_lend_is_the_default_and_off_is_the_control() {
        let d = parse(&[]);
        assert_eq!(d.xtier_lend_seq_pools, DEFAULT_XTIER_LEND_SEQ_POOLS);
        assert_eq!(d.lend_seq_pools(), (true, false), "auto = the default");
        assert_eq!(
            d.xtier_lend_seq_pools_hold,
            DEFAULT_XTIER_LEND_SEQ_POOLS_HOLD
        );
        assert_eq!(
            parse(&["--xtier-lend-seq-pools", "off"]).lend_seq_pools(),
            (false, false)
        );
        let o = parse(&[
            "--xtier-lend-seq-pools",
            "on",
            "--xtier-lend-seq-pools-hold",
            "4",
        ]);
        assert_eq!(o.lend_seq_pools(), (true, true));
        assert_eq!(o.xtier_lend_seq_pools_hold, 4);
    }

    #[test]
    fn fn100_the_buffers_top_lend_parses_by_name_and_off_is_the_default() {
        let d = parse(&[]);
        assert_eq!(d.xtier_lend_buffers_top, DEFAULT_XTIER_LEND_BUFFERS_TOP);
        assert_eq!(d.lend_buffers_top(), (false, false), "off until the gate");
        assert_eq!(
            parse(&["--xtier-lend-buffers-top", "auto"]).lend_buffers_top(),
            (true, false)
        );
        assert_eq!(
            parse(&["--xtier-lend-buffers-top", "on"]).lend_buffers_top(),
            (true, true)
        );
    }

    #[test]
    fn fn76_the_pool_slots_parse_by_name_and_the_default_is_the_constant() {
        // FN76: `--xtier-pool-slots` auto | on | off (`on` = a refusal is fatal); the default is
        // DEFAULT_XTIER_POOL_SLOTS; the reserve defaults to 0 MiB
        let d = parse(&[]);
        assert_eq!(d.xtier_pool_slots, DEFAULT_XTIER_POOL_SLOTS);
        assert_eq!(d.xtier_pool_slots_reserve_mib, 0);
        assert_eq!(
            parse(&["--xtier-pool-slots", "on"]).pool_slots(),
            (true, true)
        );
        assert_eq!(
            parse(&["--xtier-pool-slots", "auto"]).pool_slots(),
            (true, false)
        );
        assert_eq!(
            parse(&["--xtier-pool-slots", "off"]).pool_slots(),
            (false, false)
        );
        assert_eq!(
            parse(&["--xtier-pool-slots-reserve-mib", "256"]).xtier_pool_slots_reserve_mib,
            256
        );
    }

    #[test]
    fn fn107_the_pool_exclusive_parses_by_name_and_auto_is_the_default() {
        // FN107 (X1): `--xtier-pool-exclusive` auto | on | off; `auto` is the default (the user's
        // word 2026-10-09), `off` = FN76's inclusive pool, the control
        let d = parse(&[]);
        assert_eq!(d.xtier_pool_exclusive, DEFAULT_XTIER_POOL_EXCLUSIVE);
        assert_eq!(DEFAULT_XTIER_POOL_EXCLUSIVE, "auto");
        assert_eq!(d.pool_exclusive(), (true, false));
        assert_eq!(
            parse(&["--xtier-pool-exclusive", "on"]).pool_exclusive(),
            (true, true)
        );
        assert_eq!(
            parse(&["--xtier-pool-exclusive", "off"]).pool_exclusive(),
            (false, false)
        );
    }
}
