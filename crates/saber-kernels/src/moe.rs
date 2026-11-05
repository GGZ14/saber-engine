//! Flash-Next's MoE family (workstream FN4): the symbols, the tile law and the launch train.
//!
//! One MoE block of `qwen4_exp` (512 experts, top-10, `norm_topk_prob`, a shared expert gated by
//! `sigmoid(shared_expert_gate)`) runs as SIX launches (docs/EXEC.md section 11):
//!
//! | # | launch | code object | what |
//! |---|---|---|---|
//! | 1 | router GEMM (bf16, N 512, K 2,560) | FN5's bf16 family | logits (+ the shared gate's logit) |
//! | 2 | `moe_route_topk10_sort_e512` | `moe_route` | softmax, top-10, renorm, the expert-major sort + tiles |
//! | 3 | `moe_grouped_w4a8_*_g1` | `moe_grouped` | gate_up (N 1,280, K 2,560), A rows gathered by token |
//! | 4 | `moe_silu_mul_quant_t256` | `moe_grouped` | silu(gate) * up -> e4m3, expert-major rows |
//! | 5 | `moe_grouped_w4a8_*_g0` | `moe_grouped` | down (N 2,560, K 640), compact rows |
//! | 6 | `moe_combine_bf16` | `moe_route` | sum over routes in top-k order + shared * sigmoid(g) |
//!
//! The SHARED expert rides launches 3-5 as one more "expert" (tile table index `e_loc`, pointer
//! table entry `e_loc`, rows = every token), so it costs no launch of its own; it is the dense
//! W4A8 GEMM on its own bytes by construction (the grouped rows ARE the dense kernel's rows). The
//! combine is to be folded into the post-MoE all-reduce's epilogue (a later round).
//!
//! THE TILE LAW. The router/sort kernel writes a rank's routes to its LOCAL experts
//! `[e_lo, e_lo + e_loc)` (EP: rank r owns `[256r, 256r + 256)`) expert-major: expert e's routes in
//! token order at rows `off[e] .. off[e] + cnt[e]`, `off` the exclusive scan of
//! `ceil(cnt / 16) * 16` (a 16-row block owns ONE expert, the saber `mixed_kernel` invariant), then
//! cuts each expert's segment into tiles of `tile_blocks` blocks. A grouped GEMM instantiation at
//! `tm` (DTM, 16-row fragments per workgroup) reads tiles of `tm` blocks, so the route kernel runs
//! at `tile_blocks == tm`. The grids are STATIC (`rows_cap`, `tiles_cap`); `meta[0]` / `meta[1]`
//! carry the live counts and the tail exits.

use crate::{Error, Result};

/// Routed experts in one layer (config `num_experts`); the router kernel's width.
pub const EXPERTS: u32 = 512;
/// `num_experts_per_tok`.
pub const TOPK: u32 = 10;
/// `hidden_size`.
pub const HIDDEN: u32 = 2_560;
/// `moe_intermediate_size` (and `shared_expert_intermediate_size`).
pub const INTER: u32 = 640;
/// Rows per expert block: one 16-row WMMA fragment owns one expert.
pub const BLOCK_ROWS: u32 = 16;
/// Tokens one routing workgroup routes (`MOE_TPB` in `moe_route.hip`).
pub const ROUTE_TOKENS_PER_WG: u32 = 32;
/// The decode body's N tile (`DEC_DWN * 16`).
pub const N_TILE: u32 = 128;
/// Launches per MoE block, declared (structural rule 7): router GEMM, the shared-expert gate GEMV,
/// route+sort, gate_up, silu-quant, down, combine. SEVEN, not FN4's six (FN9 found it, FN9b fixed the
/// constant): `moe_combine_bf16` reads the shared expert's gate LOGIT per token (`sigmoid(g)` weights
/// the shared row), and FN4's train never computed it -- its bench fed a random one. The GEMV is a
/// transposed bf16 GEMM `C[1, T] = w_sg . X^T` on FN5's split-K band (`fn_bgemm_sk_m16`).
pub const LAUNCHES_PER_MOE_BLOCK: usize = 7;
/// The fused `moe_align_block_size` + MoE of the prior fork, per layer (1,104 over 48 layers,
/// `KERNEL_HITLIST.md`) -- the number this family replaces.
pub const PRIOR_FORK_LAUNCHES_PER_LAYER: usize = 23;

/// Code objects.
pub const PLUGIN_GROUPED: &str = "moe_grouped";
pub const PLUGIN_ROUTE: &str = "moe_route";
/// Symbols with no tile parameter.
pub const ROUTE_SYMBOL: &str = "moe_route_topk10_sort_e512";
/// FN26: FN4's route with phase 1's tokens interleaved over the 8 waves (token tl on wave tl % 8;
/// FN4's map put a decode step's T <= 4 tokens all on wave 0, serially). Same tables by
/// construction (`tools/gen_moe_route_v2.py`), gated by `fn26_bench --mode oracle`.
pub const ROUTE_SYMBOL_V2: &str = "moe_route_topk10_sort_e512_v2";
/// FN31: FN4's route with the whole step (T <= 64) in ONE workgroup and every table in LDS (the
/// per-expert token mask replaces the per-workgroup counts + ranks; phase 1 is FN4's text on v2's
/// map; `tools/gen_moe_front.py`). Drop-in: [`ROUTE_SYMBOL`]'s signature and grid, workgroups past
/// the first exit, `blk_cnt` / `rank` / `arrive` untouched. The same body runs inside the fused
/// front half ([`FrontTrain`]).
pub const ROUTE_SYMBOL_LDS: &str = "moe_route_topk10_sort_e512_lds";
/// The route symbol by token count (FN26 hold 2, 2026-09-29, card 1, 210 W, IC-cold paired, real
/// routes; tables byte-identical in 800 oracle cases + synthetic T 33 / 64 / 300): v2 wins where
/// FN4's map serialized a step's tokens on few waves -- T 2 22.8 -> 17.2 us, T 4 34.5 -> 17.7,
/// T 8 35.6 -> 18.4, T 12 36.5 -> 24.9, T 16 37.4 -> 25.5 -- ties at T 1 (16.9 / 17.0) and LOSES at
/// T 40 (42.0 -> 68.8, two workgroups): the band is the measured one. The engine names
/// [`ROUTE_SYMBOL`] today; switching it to this function is FN24's (exec.rs) edit.
pub const ROUTE_V2_MAX_TOKENS: u32 = 16;
/// FN31 hold 1 (2026-09-29, card 1, 210 W, IC-cold paired graphs, router GEMM + route, real
/// routes; the LDS route byte-identical to FN4's in every oracle case + the tie set): the LDS route
/// wins at every measured T <= 16 -- T 1 32.1 -> 30.8 us, T 2 32.0 -> 30.8, T 4 33.0 -> 31.4,
/// T 6 33.6 -> 31.8, T 8 34.1 -> 32.2, T 12 41.0 -> 38.6, T 16 42.0 -> 39.4 (vs v1 / v2 as routed)
/// -- and ties FN4's at T 40 (60.7 / 60.5): banded to T 1..16.
pub const ROUTE_LDS_MAX_TOKENS: u32 = 16;
/// FN49a: [`ROUTE_SYMBOL_LDS`] with phase 1's 110 xor-butterfly hops on DPP16 `row_xmask` +
/// `v_permlanex16` instead of `ds_bpermute` (the same partner lane, the same add order: identical
/// values by construction; `fn31_bench --mode oracle --fn49a` 4,548 cases, 0 bytes) -- -1.0..-1.2 us
/// a launch at T 1..8, -2 at T 12..16. ROUTED since FN49b ([`route_symbol`]).
pub const ROUTE_SYMBOL_LDS_DPP: &str = "moe_route_topk10_sort_e512_lds_dpp";
/// FN49b: which shuffle lowering the routed route / front half takes -- `Dpp` (FN49a's symbols,
/// THE DEFAULT) or `Shipped` (the pre-FN49b symbols, `--fn-route-control`: the byte control).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RouteForm {
    #[default]
    Dpp,
    Shipped,
}
/// The route symbol by token count (the three-launch train's; the two-launch train routes inside
/// [`FrontTrain`]) at the default [`RouteForm`].
#[must_use]
pub fn route_symbol(tokens: u32) -> &'static str {
    route_symbol_form(tokens, RouteForm::Dpp)
}
/// [`route_symbol`] at `form`: the LDS route (DPP or shipped lowering) at T 1..16, FN4's above.
#[must_use]
pub fn route_symbol_form(tokens: u32, form: RouteForm) -> &'static str {
    // FN26's v2 band (T 2..=ROUTE_V2_MAX_TOKENS) lies inside the LDS band: v2 is a bench arm now.
    if (1..=ROUTE_LDS_MAX_TOKENS).contains(&tokens) {
        match form {
            RouteForm::Dpp => ROUTE_SYMBOL_LDS_DPP,
            RouteForm::Shipped => ROUTE_SYMBOL_LDS,
        }
    } else {
        ROUTE_SYMBOL
    }
}
pub const SILU_SYMBOL: &str = "moe_silu_mul_quant_t256";
pub const COMBINE_SYMBOL: &str = "moe_combine_bf16";

/// The two expert GEMMs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeGemm {
    /// gate + up MERGED (FN2's record: gate rows first), A = the token rows, gathered.
    GateUp,
    /// down, A = the silu-quant rows, compact.
    Down,
}

impl MoeGemm {
    #[must_use]
    pub const fn n(self) -> u32 {
        match self {
            Self::GateUp => 2 * INTER,
            Self::Down => HIDDEN,
        }
    }
    #[must_use]
    pub const fn k(self) -> u32 {
        match self {
            Self::GateUp => HIDDEN,
            Self::Down => INTER,
        }
    }
    /// Whether the A row is read through `row_tok` (the token) or is the tile row itself.
    #[must_use]
    pub const fn gather(self) -> bool {
        matches!(self, Self::GateUp)
    }
    /// Bytes of one expert's `w` + `ws` + `wref` for this GEMM (what a tile of it streams).
    #[must_use]
    pub const fn expert_bytes(self) -> u64 {
        let (n, k) = (self.n() as u64, self.k() as u64);
        n * k / 2 + (k / 32) * n + n
    }
}

/// One grouped instantiation: split-K `ks` and 16-row fragments per workgroup `tm`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cell {
    pub ks: u32,
    pub tm: u32,
}

/// Every committed `(ks, tm)` cell (both A modes each), DBK 128, WPERM + NT.
pub const CELLS: [Cell; 9] = [
    Cell { ks: 1, tm: 1 },
    Cell { ks: 1, tm: 2 },
    Cell { ks: 1, tm: 4 },
    Cell { ks: 2, tm: 1 },
    Cell { ks: 2, tm: 2 },
    Cell { ks: 2, tm: 4 },
    Cell { ks: 4, tm: 1 },
    Cell { ks: 4, tm: 2 },
    Cell { ks: 4, tm: 4 },
];

/// FN20's grouped PREFILL body (`moe_grouped_w4a8_pf_tm{8,16}_g{0,1}`, DERIVED by
/// `tools/gen_moe_grouped.py` from G5's certified mid-M folded tile, `mxfp4_folded_mt.h`): one
/// 128-wide N tile x a 128- or 256-row tile of ONE expert's rows (the route's `tile_blocks` 8 / 16),
/// each 64-deep W slab staged once for every row of the tile, no split-K (the parent has none).
/// A cell here is `ks 1` at `tm` = `tile_blocks` 4 / 8 / 16 (64 / 128 / 256-row tiles): the same
/// launch law as the decode cells (grid `[N / 128, tiles_cap, 1]`, 256 threads, `SIG_MOE_GROUPED`),
/// so the launcher runs it unchanged. **`(ks 1, tm 4)` NAMES THE PREFILL BODY since FN20**: FN20's
/// oracle found it byte-identical to FN4's decode cell `..._bk128_ks1_tm4_g*` row for row (38
/// cases, T 64..2,048, EP-realistic and all-local) and faster at every T (paired IC-cold), so the
/// cell's symbol moved and no byte did; the decode symbol stays in the object as the oracle's
/// control. `(ks 2 | 4, tm 4)` stay decode cells.
pub const PREFILL_CELLS: [Cell; 3] = [
    Cell { ks: 1, tm: 4 },
    Cell { ks: 1, tm: 8 },
    Cell { ks: 1, tm: 16 },
];
/// The certified dense body the prefill cells equal byte for byte per expert (the FN20 oracle):
/// G5's folded mid-M tile (any tile height is the same per-16x16-tile arithmetic).
pub const PREFILL_PARENT_SYMBOL: &str = "mxfp4_gemm_folded_mt_tn2_wperm1_t5w4";

/// Whether `c` names the prefill body.
#[must_use]
pub const fn is_prefill_cell(c: Cell) -> bool {
    c.ks == 1 && c.tm >= 4
}

/// The MEASURED cell per token band (FN4 hold 1, 2026-09-28, card 1, 210 W, IC-cold, EP=2 rank's
/// 256 experts of layer 0, LEDGER FN4): `(max tokens, gate_up cell, down cell)`, first row whose
/// bound holds. Decode: `tm1` everywhere (one 16-row block per expert; taller tiles only compute
/// rows nobody asked for); split-K pays only at T = 1 (gate_up 52.9 -> 50.0 us at ks4, down
/// 33.1 -> 31.5 at ks2). Prefill (2,048): `tm4 ks1` (gate_up 4,119 -> 2,559 us) -- the decode body is
/// NOT the prefill body (26 TF/s); a grouped folded / A-tiled body is the open item.
///
/// FN20 (holds 1-2, 2026-09-29, card 1, 210 W, paired IC-cold A B A B, EP-realistic rank-1 routing,
/// gate_up + down us): T 64 tm1 646 + 322 vs pf_tm4 681 + 412; T 256 tm1 929 + 481 vs pf4 909 + 559
/// vs the old tm4 row 1,216 + 664; T 512 tm1 987 + 536 vs pf4 938 + 580 (a tie, tm1 kept); T 1,024
/// pf4 984 + 618 vs tm1 1,509 + 852; T 2,048 pf4 1,058 + 687 vs the old decode tm4 1,486 + 898.
/// Every row is the same bytes (decode tm1 == tm4 == pf_tm4 per row, FN4 + FN20 oracles). At a
/// skewed routing past 64 rows an expert (all-local 2,048) the 128-row tile wins (pf_tm8 1,306 +
/// 860 vs pf4 1,532 + 924): not routed -- the tile is chosen before the routing exists.
pub const MEASURED_CELLS: [(u32, Cell, Cell); 3] = [
    (1, Cell { ks: 4, tm: 1 }, Cell { ks: 2, tm: 1 }),
    (512, Cell { ks: 1, tm: 1 }, Cell { ks: 1, tm: 1 }),
    (u32::MAX, Cell { ks: 1, tm: 4 }, Cell { ks: 1, tm: 4 }),
];

/// The measured cell for `tokens` rows (data, not a ladder: the first row of [`MEASURED_CELLS`]
/// whose bound holds).
#[must_use]
pub fn measured_cell(g: MoeGemm, tokens: u32) -> Cell {
    let row = MEASURED_CELLS
        .iter()
        .find(|(max, _, _)| tokens <= *max)
        .unwrap_or(&MEASURED_CELLS[MEASURED_CELLS.len() - 1]);
    match g {
        MoeGemm::GateUp => row.1,
        MoeGemm::Down => row.2,
    }
}

/// The grouped W4A8 symbol for a GEMM at a cell -- the ROUTED body ([`DECODE_BODY`] for a decode
/// `tm 1` cell, FN4's body otherwise). The engine names its cells through this function, so the
/// body is picked up from the table without an engine edit; [`grouped_symbol_body`] names any
/// body (the control arm is [`DecodeBody::Shipped`]).
#[must_use]
pub fn grouped_symbol(g: MoeGemm, c: Cell) -> String {
    grouped_symbol_body(g, c, DECODE_BODY)
}

/// The grouped W4A8 symbol for a GEMM at a cell and a decode body. Only `tm 1` decode cells have
/// FN26's bodies; every other cell names FN4's / FN20's symbol whatever the body.
#[must_use]
pub fn grouped_symbol_body(g: MoeGemm, c: Cell, body: DecodeBody) -> String {
    if is_prefill_cell(c) {
        return format!("moe_grouped_w4a8_pf_tm{}_g{}", c.tm, u8::from(g.gather()));
    }
    let suffix = if c.tm == 1 { body.suffix() } else { "" };
    format!(
        "moe_grouped_w4a8_bk128_ks{}_tm{}_g{}{suffix}",
        c.ks,
        c.tm,
        u8::from(g.gather())
    )
}

/// FN26: the decode GEMM BODY of a `tm 1` cell. Every body computes the same bytes as FN4's (the
/// same slabs in the same order into the same WMMA chain, the same split-K partition and reduce --
/// gated on the card by `tools/capture/fn26/fn26_bench.cpp --mode oracle`); they differ in how the
/// weight stream is issued. A body is a BYTE-EXACT lever; a different split count `ks` is NOT (the
/// reduce sums a different set of partials: numerics, FN4's class).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeBody {
    /// FN4's derived decode body (one slab in flight per workgroup): the control arm.
    Shipped,
    /// Slab s + 1's W words, scale bytes and A bytes loaded into registers before slab s folds
    /// (a register double buffer; `_p1`, 126-127 VGPR, occupancy 10).
    Prefetch,
    /// 64-wide n-blocks (4 waves; `_n64_p0`): twice the workgroups at the same bytes. A bench arm:
    /// its grid is `N / 64` (the launch law's `N / 128` would change), so no table names it.
    N64,
    /// 64-wide n-blocks + the prefetch (`_n64_p1`): a bench arm, as [`DecodeBody::N64`].
    N64Prefetch,
    /// FN31 (`_p2`): the `_p1` register double buffer with the three things that kept one slab in
    /// flight removed -- LDS-scope barriers (a `__syncthreads()` is a workgroup release fence =
    /// `s_wait_loadcnt 0` on gfx12, so `_p1`'s slab loop drained its prefetch twice a slab), the
    /// fold table `kMag` read from an LDS copy (its `__constant__` lookups queued behind the
    /// prefetch), global (not flat) weight loads, unconditional prefetches at a clamped slab --
    /// and <= 120 VGPR (100-101, occupancy 12, 6 workgroups a WGP; `_p1` 126-127, occupancy 10).
    LdsPrefetch,
    /// FN31 (`_p3`): [`DecodeBody::LdsPrefetch`] with a THREE-slab register ring (two slabs in
    /// flight while one folds), 140-141 VGPR at occupancy 10 (it spills under the 120 cap).
    Ring3,
}

impl DecodeBody {
    /// The symbol suffix.
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::Shipped => "",
            Self::Prefetch => "_p1",
            Self::N64 => "_n64_p0",
            Self::N64Prefetch => "_n64_p1",
            Self::LdsPrefetch => "_p2",
            Self::Ring3 => "_p3",
        }
    }
    /// The n-block width (the grid's x is `N / width`, the block `width / 16` waves).
    #[must_use]
    pub const fn n_width(self) -> u32 {
        match self {
            Self::Shipped | Self::Prefetch | Self::LdsPrefetch | Self::Ring3 => N_TILE,
            Self::N64 | Self::N64Prefetch => 64,
        }
    }
    /// The fused back half's loop-body suffix (`moe_back_*_p{n}`), or `None` for a body the back
    /// half is not instantiated with (the 64-wide arms).
    #[must_use]
    pub const fn back_suffix(self) -> Option<&'static str> {
        match self {
            Self::Shipped => Some("_p0"),
            Self::Prefetch => Some("_p1"),
            Self::LdsPrefetch => Some("_p2"),
            Self::Ring3 => Some("_p3"),
            Self::N64 | Self::N64Prefetch => None,
        }
    }
}

/// The ROUTED decode body (FN26). Only a body with FN4's launch law (`N / 128` x 256 threads) can be
/// routed without an engine edit. **`Prefetch` since FN26 hold 1** (2026-09-29, card 1, 210 W,
/// IC-cold paired, real routes): the `_p1` cells equal the shipped cells byte for byte (720 oracle
/// cases, 0 rows differ) and win at every decode T (gate_up T 1 38.5 -> 33.7 us, T 4 97.2 -> 76.2,
/// T 8 131.2 -> 119.6, T 40 402.9 -> 381.2; down T 1 23.7 -> 22.6, T 4 47.0 -> 42.4, T 8 71.4 ->
/// 63.8). `DecodeBody::Shipped` is the control arm ([`grouped_symbol_body`]).
///
/// FN32 (hold 1, 2026-09-29, card 1, 210 W, IC-cold paired, real routes, LEDGER FN32): `_p1` IS AT
/// ITS WALL -- each launch costs ~9.5 us fixed + its bytes at ~540 GB/s (the card's sustained read
/// rate under the cap, fclk 2016 of 2400), within 0.2-1.1 us + 3 % of a load-only stream of the same
/// grid and addresses; occupancy 4..12, 32/64-wide n-blocks and dead-tiles-last orders are flat
/// (bit-exact, 9,792 oracle cases 0 rows), so no body change is routed.
pub const DECODE_BODY: DecodeBody = DecodeBody::Prefetch;

/// FN26: the MoE block's launches in the FUSED train: the router GEMM (FN5's bf16 band), the
/// route and sort, and ONE launch for everything after the route ([`BackTrain`]: the shared-expert
/// gate GEMV, gate_up, silu-quant, down and the combine, chained by device counters).
pub const LAUNCHES_PER_MOE_BLOCK_FUSED: usize = 3;

/// The fused train, declared (structural rule 7).
#[must_use]
pub const fn launch_train_fused() -> [TrainOp; LAUNCHES_PER_MOE_BLOCK_FUSED] {
    [
        TrainOp {
            name: "router_gemm_bf16",
            plugin: None,
        },
        TrainOp {
            name: ROUTE_SYMBOL,
            plugin: Some(PLUGIN_ROUTE),
        },
        TrainOp {
            name: "moe_back_gu{ks}_dn{ks}_p{pfd} (gate GEMV + gate_up + silu + down + combine)",
            plugin: Some(PLUGIN_GROUPED),
        },
    ]
}

/// FN26's fused back half (`moe_back_gu{KSG}_dn{KSD}_p{PFD}` in `moe_grouped`) -- its launch law.
///
/// Arguments (`abi::SIG_MOE_BACK`, in order): the block input's e4m3 rows `q8` + per-row scale (the
/// gate_up A, as launch 4), `xptr` gate_up / down, the route's tile tables + `row_tok` + `meta`,
/// the two split-K partial slabs (`[ks][pcap][N]` f32, null at ks 1) and FN4's two arrival-counter
/// arrays (`tc * 10` / `tc * 20` i32, zeroed once), the gate_up rows `[pcap][1280]` bf16, the silu
/// rows `[pcap][640]` e4m3 + scales `[pcap]` f32, the down rows `[pcap][2560]` bf16, `topk_w`,
/// `route_pos`, the shared gate weight `[2560]` bf16, the block input `bi` `[T][2560]` bf16 (the
/// GEMV's operand, as launch 2), the gate logit `[T]` bf16 (written), the MoE output `[T][2560]`
/// bf16, the `sync` words ([`BackTrain::sync_words`] i32, ZEROED ONCE at allocation: every word is
/// back at zero when the launch ends), then `T`, `pcap`, `tc`. Grid `[grid(tc), 1, 1]` x 256.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackTrain {
    /// gate_up split count.
    pub gu_ks: u32,
    /// down split count.
    pub dn_ks: u32,
    /// The register double buffer (FN26's lever) on both GEMMs.
    pub prefetch: bool,
}

impl BackTrain {
    /// Tokens the fused half serves: its gate GEMV runs the split-K band's one-workgroup form (64
    /// columns); a wider step keeps the seven-launch train.
    pub const MAX_TOKENS: u32 = 64;
    /// The (gate_up, down) split pairs instantiated: FN4's T = 1 cell and its T >= 2 cell.
    pub const CELLS: [(u32, u32); 2] = [(4, 2), (1, 1)];
    /// The fused half at the measured cells for `tokens` (the same split class as the seven-launch
    /// train's [`measured_cell`]s, so the same bytes), or `None` above [`Self::MAX_TOKENS`] / at a
    /// prefill cell.
    #[must_use]
    pub fn for_tokens(tokens: u32, prefetch: bool) -> Option<Self> {
        let (gu, dn) = (
            measured_cell(MoeGemm::GateUp, tokens),
            measured_cell(MoeGemm::Down, tokens),
        );
        if tokens == 0 || tokens > Self::MAX_TOKENS || gu.tm != 1 || dn.tm != 1 {
            return None;
        }
        let b = Self {
            gu_ks: gu.ks,
            dn_ks: dn.ks,
            prefetch,
        };
        Self::CELLS.contains(&(b.gu_ks, b.dn_ks)).then_some(b)
    }
    /// The symbol: `prefetch` names the ROUTED prefetching body ([`BACK_PREFETCH_BODY`]), so the
    /// engine picks a byte-exact body change up with no edit; `false` = FN26's `_p0`.
    #[must_use]
    pub fn symbol(&self) -> String {
        self.symbol_body(if self.prefetch {
            BACK_PREFETCH_BODY
        } else {
            DecodeBody::Shipped
        })
    }
    /// The symbol at an explicit loop body (the bench's arms); `_p0` for a body the back half is
    /// not instantiated with.
    #[must_use]
    pub fn symbol_body(&self, body: DecodeBody) -> String {
        format!(
            "moe_back_gu{}_dn{}{}",
            self.gu_ks,
            self.dn_ks,
            body.back_suffix().unwrap_or("_p0")
        )
    }
    /// Workgroups: one for the gate GEMV + every (tile, n-block, split) of gate_up, then of down.
    #[must_use]
    pub const fn grid(&self, tiles_cap: u32) -> u32 {
        1 + tiles_cap * (MoeGemm::GateUp.n() / N_TILE) * self.gu_ks
            + tiles_cap * (MoeGemm::Down.n() / N_TILE) * self.dn_ks
    }
    pub const THREADS: u32 = 256;
    /// `sync` words (i32): ticket, gate flag, combine count, 20 per-n-block down counts, then per
    /// tile the gate_up count, the silu flag and the down count.
    #[must_use]
    pub const fn sync_words(tiles_cap: u32) -> u64 {
        3 + (MoeGemm::Down.n() / N_TILE) as u64 + 3 * tiles_cap as u64
    }
}

/// The loop body [`BackTrain::symbol`] names when `prefetch` is set: FN26's `_p1` until a FN31
/// body is gated byte-exact AND measured faster on the card (the bench arm is
/// [`BackTrain::symbol_body`]).
pub const BACK_PREFETCH_BODY: DecodeBody = DecodeBody::Prefetch;

/// FN31: the MoE block in TWO launches -- [`FrontTrain`] (router GEMM + route) and [`BackTrain`].
pub const LAUNCHES_PER_MOE_BLOCK_FRONT: usize = 2;

/// The two-launch train, declared (structural rule 7).
#[must_use]
pub const fn launch_train_front() -> [TrainOp; LAUNCHES_PER_MOE_BLOCK_FRONT] {
    [
        TrainOp {
            name: "moe_front_e512_nw{w}_s{split} (router GEMM + route + sort)",
            plugin: Some(PLUGIN_ROUTE),
        },
        TrainOp {
            name: "moe_back_gu{ks}_dn{ks}_p{pfd} (gate GEMV + gate_up + silu + down + combine)",
            plugin: Some(PLUGIN_GROUPED),
        },
    ]
}

/// FN84: the fused back half's work ORDER (`--fn-moe-miss-order`). Every order runs the same
/// (tile, n-block, split) work with the same body, the split-K reduce in split order and the
/// combine in top-k order, so every order is the shipped symbol's bytes (gated, LEDGER FN84); only
/// which workgroup takes which ticket moves. `Id` = FN26's symbol (the default until the gate
/// passes); `First` = the RAM-tier tiles' gate_up and down workgroups take the first tickets;
/// `Capped` = one RAM workgroup per `k` resident ones (the rules' `[[moe_miss]]` row by T);
/// `CappedPre` = `Capped` + a RAM tile's down workgroup streams its second W slab before it waits
/// for silu (`moe_back_mop_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MissOrder {
    #[default]
    Id,
    First,
    Capped,
    CappedPre,
}

impl MissOrder {
    pub const ALL: [Self; 4] = [Self::Id, Self::First, Self::Capped, Self::CappedPre];
    /// The flag's spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Id => "id",
            Self::First => "first",
            Self::Capped => "capped",
            Self::CappedPre => "cappre",
        }
    }
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|o| o.name() == s)
    }
    /// The `which_path` counter's slot.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
    /// The order-law symbol for `b` (`None` = FN26's shipped symbol, [`BackTrain::symbol`]).
    /// Instantiated at the prefetch body only.
    #[must_use]
    pub fn symbol(self, b: &BackTrain) -> Option<String> {
        let stem = match self {
            Self::Id => return None,
            Self::First | Self::Capped => "mo",
            Self::CappedPre => "mop",
        };
        Some(format!("moe_back_{stem}_gu{}_dn{}_p1", b.gu_ks, b.dn_ks))
    }
    /// The kernel's `kint`: -1 = id order, 0 = every RAM workgroup first, k >= 1 = the rules'
    /// interleave at `tokens` rows (`None` when the table has no row: the caller refuses).
    #[must_use]
    pub fn kint(self, rules: &crate::flashnext::FlashNextRules, tokens: u32) -> Option<i32> {
        match self {
            Self::Id => Some(-1),
            Self::First => Some(0),
            Self::Capped | Self::CappedPre => rules.miss_k(tokens).filter(|&k| k >= 1),
        }
    }
}

/// The router GEMM's split count at `tokens` rows: the rules' `moe.router` row (FN5's measured
/// bands: `fn_bgemm_sk_m16` at 4 / 2 / 1 splits to 1 / 16 / 32 rows, then the drafter's
/// `dflash_bgemm_bf16_m32n16`, whose arithmetic is the split-K band's at ONE split). The fused
/// front half runs the SAME split (a different split is a different set of fp32 partials:
/// numerics). A test holds this against `rules/flashnext_gfx1201.toml`.
#[must_use]
pub const fn router_split(tokens: u32) -> u32 {
    if tokens <= 1 {
        4
    } else if tokens <= 16 {
        2
    } else {
        1
    }
}

/// FN31's fused FRONT half (`moe_front_e512_*` in `moe_route`) -- its launch law.
///
/// Arguments (`abi::SIG_MOE_FRONT`, in order): `bi`, the router weight, the partial slices
/// `[split][T][512]` f32 ([`FrontTrain::part_floats`]), ONE `sync` i32 (ZEROED ONCE at allocation;
/// zero again when the launch ends), the router logits `[T][512]` bf16 (written), `k` = 2,560,
/// then `T`, `e_lo`, `e_loc`, `tile_blocks`, `with_shared` and the route's tables in
/// `SIG_MOE_ROUTE`'s order (no `blk_cnt` / `rank` / `arrive`: the route runs in one workgroup's LDS).
/// Grid [`FrontTrain::grid`] x [`FrontTrain::threads`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrontTrain {
    /// The router GEMM's split count ([`router_split`]).
    pub split: u32,
    /// Waves a workgroup (16 router columns each).
    pub waves: u32,
    /// The slice fully unrolled at `split` (`_s{split}`) or the band's own loop (`_u8`, the
    /// shipped schedule: a bench arm).
    pub unrolled: bool,
    /// FN49b: the route's butterflies on DPP (`_dpp`, FN49a's symbols, the default; built for the
    /// unrolled nw8 cells only) or `ds_bpermute` (the shipped text, `--fn-route-control`).
    pub route: RouteForm,
}

impl FrontTrain {
    /// Tokens the kernels hold (`MF_TMAX`; = [`BackTrain::MAX_TOKENS`]; byte-gated at T 1..64).
    pub const KERNEL_MAX_TOKENS: u32 = 64;
    /// Tokens the front half is ROUTED for (FN31 hold 2, 2026-09-29, card 1, 210 W, IC-cold paired
    /// graphs, real routes, per MoE block vs FN26's three-launch train): it wins while every token
    /// has its own wave of the last workgroup (nw8: T <= 8) -- T 1 74.0 -> 70.1 us, T 2 97.2 -> 93.7,
    /// T 4 134.7 -> 130.2, T 6 182.2 -> 177.8, T 8 197.9 -> 192.1 -- and loses from the second token
    /// a wave (T 12 283.8 -> 302.2, T 16 365.0 -> 382.1, T 40 634.9 -> 655.3: ~30 us a serial token
    /// round, not the slice reads). Above 8 the three-launch train with [`route_symbol`] runs.
    pub const MAX_TOKENS: u32 = 8;
    /// Waves a workgroup at the routed cell.
    pub const WAVES: u32 = 8;
    /// The `sync` words (i32).
    pub const SYNC_WORDS: u64 = 1;
    /// The routed front half at `tokens`, or `None` above [`Self::MAX_TOKENS`].
    #[must_use]
    pub const fn for_tokens(tokens: u32) -> Option<Self> {
        if tokens == 0 || tokens > Self::MAX_TOKENS {
            return None;
        }
        Some(Self {
            split: router_split(tokens),
            waves: Self::WAVES,
            unrolled: true,
            route: RouteForm::Dpp,
        })
    }
    #[must_use]
    pub fn symbol(&self) -> String {
        if self.unrolled {
            let dpp = if self.route == RouteForm::Dpp && self.waves == 8 {
                "_dpp"
            } else {
                ""
            };
            format!("moe_front_e512_nw{}_s{}{dpp}", self.waves, self.split)
        } else {
            format!("moe_front_e512_nw{}_u8", self.waves)
        }
    }
    /// `[512 / (16 waves), cdiv(tokens, 16), split]`.
    #[must_use]
    pub const fn grid(&self, tokens: u32) -> [u32; 3] {
        [EXPERTS / (16 * self.waves), tokens.div_ceil(16), self.split]
    }
    #[must_use]
    pub const fn threads(&self) -> u32 {
        self.waves * 32
    }
    /// The partial slices (f32 entries).
    #[must_use]
    pub const fn part_floats(&self, tokens: u32) -> u64 {
        self.split as u64 * tokens as u64 * EXPERTS as u64
    }
}

/// The dense certified kernel the grouped one must equal byte for byte at the same cell (the
/// byte oracle; its own DTM is a function of the rows it is handed and does not enter the answer).
#[must_use]
pub fn dense_oracle_symbol(c: Cell, rows: u32) -> String {
    if is_prefill_cell(c) {
        return PREFILL_PARENT_SYMBOL.to_string();
    }
    format!(
        "mxfp4_gemm_decode_bk128_ks{}_tm{}_wperm1_nt1",
        c.ks,
        rows.clamp(1, 128).div_ceil(16)
    )
}

/// The A16 control arm's symbol.
#[must_use]
pub fn a16_symbol(g: MoeGemm) -> &'static str {
    if g.gather() {
        "moe_grouped_a16_ref_g1"
    } else {
        "moe_grouped_a16_ref_g0"
    }
}

/// The geometry of one MoE block on one rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    /// Token rows in the step (decode rows or a prefill chunk).
    pub tokens: u32,
    /// First local expert (EP rank r: 256r).
    pub e_lo: u32,
    /// Local experts (EP=2: 256; one card holding all: 512).
    pub e_loc: u32,
    /// 16-row blocks per tile == the grouped cell's `tm`.
    pub tile_blocks: u32,
    /// Whether this rank computes the shared expert (one rank does).
    pub shared: bool,
}

impl Geometry {
    /// Refuses what the kernels cannot run.
    pub fn validate(&self) -> Result<()> {
        let bad = |why: String| Err(Error::Validate(vec![why]));
        if self.tokens == 0 {
            return bad("moe: zero tokens".into());
        }
        if self.e_loc == 0 || self.e_loc > EXPERTS || self.e_lo + self.e_loc > EXPERTS {
            return bad(format!(
                "moe: local experts [{}, {}) outside 0..{EXPERTS} (the sort scans two experts a \
                 thread over 256 threads)",
                self.e_lo,
                self.e_lo + self.e_loc
            ));
        }
        if !matches!(self.tile_blocks, 1 | 2 | 4 | 8 | 16) {
            return bad(format!(
                "moe: tile_blocks {} has no grouped cell (decode 1, 2, 4; prefill 8, 16)",
                self.tile_blocks
            ));
        }
        Ok(())
    }
    #[must_use]
    pub const fn routes(&self) -> u32 {
        self.tokens * TOPK
    }
    /// Rows of the expert-major arrays (H, the silu-quant output, Y) at the worst routing:
    /// every local route + the per-expert pad (<= 15 rows for each expert that has a route) + the
    /// shared expert's rows padded to 16.
    #[must_use]
    pub fn rows_cap(&self) -> u32 {
        let r = self.routes();
        let sh = if self.shared {
            self.tokens.div_ceil(BLOCK_ROWS) * BLOCK_ROWS
        } else {
            0
        };
        r + (BLOCK_ROWS - 1) * r.min(self.e_loc) + sh
    }
    /// Tiles at the worst routing (the grouped grid's y).
    #[must_use]
    pub fn tiles_cap(&self) -> u32 {
        let r = self.routes();
        let sh = if self.shared {
            self.tokens.div_ceil(BLOCK_ROWS).div_ceil(self.tile_blocks)
        } else {
            0
        };
        r.min(self.e_loc) + r / (BLOCK_ROWS * self.tile_blocks) + 1 + sh
    }
    /// Routing workgroups (the route kernel's grid).
    #[must_use]
    pub const fn route_workgroups(&self) -> u32 {
        self.tokens.div_ceil(ROUTE_TOKENS_PER_WG)
    }
    /// The route kernel's per-workgroup count table (i32 entries).
    #[must_use]
    pub const fn blk_cnt_len(&self) -> u64 {
        self.route_workgroups() as u64 * self.e_loc as u64
    }
    /// The grouped GEMM's split-K partial slab (f32 entries); zero at `ks == 1` (DIRECT).
    #[must_use]
    pub fn partial_floats(&self, g: MoeGemm, ks: u32) -> u64 {
        if ks <= 1 {
            0
        } else {
            u64::from(ks) * u64::from(self.rows_cap()) * u64::from(g.n())
        }
    }
    /// Arrival counters of a split grouped GEMM (i32, zeroed once; the kernel re-zeroes).
    #[must_use]
    pub fn counters(&self, g: MoeGemm) -> u64 {
        u64::from(self.tiles_cap()) * u64::from(g.n() / N_TILE)
    }
}

/// One launch of the train.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainOp {
    pub name: &'static str,
    /// The code object, or `None` for the router GEMM (FN5's bf16 family).
    pub plugin: Option<&'static str>,
}

/// The MoE block's launch train, declared (structural rule 7).
#[must_use]
pub const fn launch_train() -> [TrainOp; LAUNCHES_PER_MOE_BLOCK] {
    [
        TrainOp {
            name: "router_gemm_bf16",
            plugin: None,
        },
        TrainOp {
            name: "shared_gate_gemv_bf16 (C[1,T] = w_sg . X^T)",
            plugin: None,
        },
        TrainOp {
            name: ROUTE_SYMBOL,
            plugin: Some(PLUGIN_ROUTE),
        },
        TrainOp {
            name: "moe_grouped_w4a8 (gate_up, g1)",
            plugin: Some(PLUGIN_GROUPED),
        },
        TrainOp {
            name: SILU_SYMBOL,
            plugin: Some(PLUGIN_GROUPED),
        },
        TrainOp {
            name: "moe_grouped_w4a8 (down, g0)",
            plugin: Some(PLUGIN_GROUPED),
        },
        TrainOp {
            name: COMBINE_SYMBOL,
            plugin: Some(PLUGIN_ROUTE),
        },
    ]
}

/// The expert-major layout the route kernel writes -- the Rust twin of `moe_ref.py`'s
/// `sort_law`, for sizing and for the planner's host checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    /// Row -> token (`-1` = pad).
    pub row_tok: Vec<i32>,
    /// Route id (`t * 10 + k`) -> row (`-1` = another rank's expert).
    pub route_pos: Vec<i32>,
    /// Tiles: (local expert or `e_loc` for the shared expert, first row, live rows).
    pub tiles: Vec<(u32, u32, u32)>,
    /// Routes per local expert.
    pub expert_cnt: Vec<u32>,
    /// Live padded rows (routed + shared).
    pub rows: u32,
}

/// The layout for `ids` (`[tokens * 10]` global expert ids, top-k order per token).
pub fn sort_layout(ids: &[u32], g: &Geometry) -> Result<Layout> {
    g.validate()?;
    if ids.len() != g.routes() as usize {
        return Err(Error::Validate(vec![format!(
            "moe: {} ids for {} routes",
            ids.len(),
            g.routes()
        )]));
    }
    let local = |e: u32| e >= g.e_lo && e < g.e_lo + g.e_loc;
    let mut cnt = vec![0u32; g.e_loc as usize];
    for &e in ids {
        if e >= EXPERTS {
            return Err(Error::Validate(vec![format!(
                "moe: expert id {e} >= {EXPERTS}"
            )]));
        }
        if local(e) {
            cnt[(e - g.e_lo) as usize] += 1;
        }
    }
    let mut off = Vec::with_capacity(cnt.len());
    let mut rows = 0u32;
    for &c in &cnt {
        off.push(rows);
        rows += c.div_ceil(BLOCK_ROWS) * BLOCK_ROWS;
    }
    let routed = rows;
    if g.shared {
        rows += g.tokens.div_ceil(BLOCK_ROWS) * BLOCK_ROWS;
    }
    let mut row_tok = vec![-1i32; rows as usize];
    let mut route_pos = vec![-1i32; ids.len()];
    let mut fill = vec![0u32; cnt.len()];
    for (r, &e) in ids.iter().enumerate() {
        if !local(e) {
            continue;
        }
        let le = (e - g.e_lo) as usize;
        let p = off[le] + fill[le];
        fill[le] += 1;
        row_tok[p as usize] = (r as u32 / TOPK) as i32;
        route_pos[r] = p as i32;
    }
    let tr = BLOCK_ROWS * g.tile_blocks;
    let mut tiles = Vec::new();
    for (le, &c) in cnt.iter().enumerate() {
        let mut j = 0;
        while j * tr < c {
            tiles.push((le as u32, off[le] + j * tr, tr.min(c - j * tr)));
            j += 1;
        }
    }
    if g.shared {
        let mut j = 0;
        while j * tr < g.tokens {
            tiles.push((g.e_loc, routed + j * tr, tr.min(g.tokens - j * tr)));
            j += 1;
        }
        for t in 0..g.tokens {
            row_tok[(routed + t) as usize] = t as i32;
        }
    }
    Ok(Layout {
        row_tok,
        route_pos,
        tiles,
        expert_cnt: cnt,
        rows,
    })
}

/// FN80b-1: the code object of the TP=2 half-width cells (`tools/gen_moe_grouped.py`
/// `derive_tp2`): the down half at DBK 64 + the half-width fused back half. Its own object, so
/// `moe_grouped` keeps its bytes.
pub const PLUGIN_GROUPED_TP2: &str = "moe_grouped_tp2";

/// FN80b-1: the SHAPE KEY of a rank's routed-expert cells (FLASHNEXT-PLAN section 13, LEDGER
/// FN80a / FN80b-1). `Whole` = EP=2 (rank r owns whole experts: gate_up N 1,280, down K 640 at
/// DBK 128) -- every symbol the engine routes today. `TpHalf` = TP=2 (every rank holds half of
/// EVERY expert along the intermediate dimension: INTER 320, gate_up N 640 on the COMMITTED cells
/// since n is a kernel argument and 640 = 5 x 128, down K 320 = 2.5 slabs of 128 -> the DBK-64
/// cells of [`PLUGIN_GROUPED_TP2`]). NOTHING ROUTES `TpHalf` YET (FN80b-2 owns the engine's
/// `--fn-experts ep|tp2`); the symbols are named here so the engine picks them up by key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExpertShape {
    #[default]
    Whole,
    TpHalf,
}

impl ExpertShape {
    /// The intermediate rows a rank holds per expert (silu-quant's `n`, the down GEMM's K).
    #[must_use]
    pub const fn inter(self) -> u32 {
        match self {
            Self::Whole => INTER,
            Self::TpHalf => INTER / 2,
        }
    }
    /// gate_up's N (gate rows then up rows).
    #[must_use]
    pub const fn gate_up_n(self) -> u32 {
        2 * self.inter()
    }
    /// The down GEMM's K slab: the decode body stages UNMASKED `dbk`-byte K slabs, so K % dbk == 0
    /// is the legality law (memory `mxfp4-decode-kernel-k128`).
    #[must_use]
    pub const fn down_dbk(self) -> u32 {
        match self {
            Self::Whole => 128,
            Self::TpHalf => 64,
        }
    }
    /// The code object the down cells live in.
    #[must_use]
    pub const fn down_plugin(self) -> &'static str {
        match self {
            Self::Whole => PLUGIN_GROUPED,
            Self::TpHalf => PLUGIN_GROUPED_TP2,
        }
    }
    /// The code object of the fused back half.
    #[must_use]
    pub const fn back_plugin(self) -> &'static str {
        self.down_plugin()
    }
    /// Bytes of one expert (part) for a GEMM -- what a tile of it streams: `w` + `ws` + `wref`
    /// (the half's `down.wref` is the WHOLE row's, 2,560 B, so the down bytes are N + ...).
    #[must_use]
    pub const fn expert_bytes(self, g: MoeGemm) -> u64 {
        let i = self.inter() as u64;
        let h = HIDDEN as u64;
        match g {
            MoeGemm::GateUp => 2 * i * h / 2 + (h / 32) * 2 * i + 2 * i,
            MoeGemm::Down => h * i / 2 + (i / 32) * h + h,
        }
    }
    /// The gate_up symbol at a cell and body: the COMMITTED `moe_grouped` cells at either shape
    /// (N is an argument; 640 = 5 n-blocks of 128).
    #[must_use]
    pub fn gate_up_symbol(self, c: Cell, body: DecodeBody) -> String {
        grouped_symbol_body(MoeGemm::GateUp, c, body)
    }
    /// The down symbol at a cell and body. `Whole` = [`grouped_symbol_body`]. `TpHalf`: the
    /// prefill cells (`pf_tm*`, BK 64 staging: 320 = 5 x 64, legal as committed) or the DBK-64
    /// decode cells `moe_grouped_w4a8_bk64_ks{1,2}_tm1_g0[_p1]` (FN4's body / FN26's `_p1`); any
    /// other cell or body is refused by name (no DBK-64 instantiation exists).
    pub fn down_symbol(self, c: Cell, body: DecodeBody) -> Result<String> {
        if self == Self::Whole || is_prefill_cell(c) {
            return Ok(grouped_symbol_body(MoeGemm::Down, c, body));
        }
        let suffix = match body {
            DecodeBody::Shipped => "",
            DecodeBody::Prefetch => "_p1",
            other => {
                return Err(Error::Rule(format!(
                    "the TP=2 down half has no {other:?} body at DBK 64 (shipped and _p1 only)"
                )))
            }
        };
        if c.tm != 1 || !TP2_DOWN_KS.contains(&c.ks) {
            return Err(Error::Rule(format!(
                "the TP=2 down half (K 320, DBK 64) has no cell ks {} tm {} (tm 1 at ks {:?})",
                c.ks, c.tm, TP2_DOWN_KS
            )));
        }
        Ok(format!("moe_grouped_w4a8_bk64_ks{}_tm1_g0{suffix}", c.ks))
    }
    /// The fused back half's symbol for `b` at this shape. `TpHalf` instantiates the prefetching
    /// body only (`moe_back_h2_gu{ks}_dn{ks}_p1`, [`BackTrain::CELLS`]' split pairs).
    pub fn back_symbol(self, b: &BackTrain) -> Result<String> {
        match self {
            Self::Whole => Ok(b.symbol()),
            Self::TpHalf if b.prefetch && BACK_PREFETCH_BODY == DecodeBody::Prefetch => {
                Ok(format!("moe_back_h2_gu{}_dn{}_p1", b.gu_ks, b.dn_ks))
            }
            Self::TpHalf => Err(Error::Rule(
                "the half-width back half is instantiated with the _p1 body only".into(),
            )),
        }
    }
    /// The fused back half's 1-D grid: one gate-GEMV workgroup + every (tile, n-block, split) of
    /// gate_up (N / 128 n-blocks: 10 whole, 5 half) then of down (20 at either shape).
    #[must_use]
    pub const fn back_grid(self, b: &BackTrain, tiles_cap: u32) -> u32 {
        1 + tiles_cap * (self.gate_up_n() / N_TILE) * b.gu_ks
            + tiles_cap * (MoeGemm::Down.n() / N_TILE) * b.dn_ks
    }
}

/// FN80b-1: the down half's instantiated split counts (EP's T 1 down cell is ks 2, T >= 2 ks 1).
pub const TP2_DOWN_KS: [u32; 2] = [1, 2];

/// FN150: the NVFP4 (FP4-Local, xtier v2 `NVFP4_E4M3G16`) expert cells the engine routes under
/// `--fn-experts-fmt nvfp4`: FN148's FORM (b) (`fn148_w4a8b_*` in `moe_fn148`: exact e2m1 x E4M3
/// g16 weights, A8 activations, the record's f32 globals in the epilogue split at `gsplit`;
/// byte-equal to its A8-domain twin on all 15 cells, LEDGER FN148). Only FN148's plain decode
/// body exists (no `_p1`, no fused back half, no `far`, no prefill tile), so: gate_up ks 1 / 2 / 4
/// x tm 1 / 2 / 4, down ks 1 / 2 x tm 1 / 2 / 4 at the whole shape (DBK 128) and ks 1 / 2 tm 1 at
/// the TP half (DBK 64); a PREFILL cell ([`is_prefill_cell`]: ks 1, tm >= 4) is refused by name.
pub const NVFP4_GATE_UP_KS: [u32; 3] = [1, 2, 4];
/// FN150: the NVFP4 down cells' split counts at the whole shape (the TP half: [`TP2_DOWN_KS`]).
pub const NVFP4_DOWN_KS: [u32; 2] = [1, 2];
/// FN150: the code object of the NVFP4 cells.
pub const PLUGIN_FN148: &str = "moe_fn148";

/// FN150: the NVFP4 cell symbol for `g` at `c` and `shape` (see [`NVFP4_GATE_UP_KS`]).
pub fn nvfp4_symbol(shape: ExpertShape, g: MoeGemm, c: Cell) -> Result<String> {
    if is_prefill_cell(c) {
        return Err(Error::Rule(format!(
            "--fn-experts-fmt nvfp4: no NVFP4 prefill tile exists (cell ks {} tm {} = the pf_tm{} \
             body; FN148 built the decode cells only -- the prefill tiles are FN133 P3's next item)",
            c.ks, c.tm, c.tm
        )));
    }
    let tm124 = matches!(c.tm, 1 | 2 | 4);
    let (ok, dbk) = match (shape, g) {
        (_, MoeGemm::GateUp) => (NVFP4_GATE_UP_KS.contains(&c.ks) && tm124, 128),
        (ExpertShape::Whole, MoeGemm::Down) => (NVFP4_DOWN_KS.contains(&c.ks) && tm124, 128),
        (ExpertShape::TpHalf, MoeGemm::Down) => (TP2_DOWN_KS.contains(&c.ks) && c.tm == 1, 64),
    };
    if !ok {
        return Err(Error::Rule(format!(
            "--fn-experts-fmt nvfp4: FN148 has no {g:?} cell ks {} tm {} at the {shape:?} shape",
            c.ks, c.tm
        )));
    }
    Ok(format!(
        "fn148_w4a8b_bk{dbk}_ks{}_tm{}_g{}",
        c.ks,
        c.tm,
        u8::from(g.gather())
    ))
}

/// FN152: the code object of FN151's `_p1`-class twins of the `fn148_w4a8b` cells.
pub const PLUGIN_FN151: &str = "moe_fn151";
/// FN152: the code object of the NVFP4 fused back halves (`fn152_moe_back_*`).
pub const PLUGIN_FN152: &str = "moe_fn152";

/// FN152: which NVFP4 grouped body the engine names under `--fn-experts-fmt nvfp4`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Nvfp4Body {
    /// FN151's `_p1`-class twins (`fn151_w4a8b_*`, byte-equal to FN148's cells, LEDGER FN151)
    /// where one exists, FN148's cell elsewhere (gate_up tm 2 / 4). The default.
    #[default]
    P1,
    /// FN148's plain form (b) cells only (`fn148_w4a8b_*`): FN150's route, the byte control.
    Plain,
}

/// FN152: the NVFP4 cell symbol at `body` (see [`Nvfp4Body`]); refusals = [`nvfp4_symbol`]'s.
pub fn nvfp4_symbol_body(
    shape: ExpertShape,
    g: MoeGemm,
    c: Cell,
    body: Nvfp4Body,
) -> Result<String> {
    let s = nvfp4_symbol(shape, g, c)?;
    // FN151's cells (tools/gen_fn151_nvfp4p1.py CELLS, default SC): gate_up bk128 ks 1/2/4 tm 1;
    // down bk128 ks 1/2 tm 1/2/4 and bk64 ks 1/2 tm 1 -- every down cell FN148 has
    let twin = match g {
        MoeGemm::GateUp => c.tm == 1,
        MoeGemm::Down => true,
    };
    Ok(if body == Nvfp4Body::P1 && twin {
        s.replacen("fn148_", "fn151_", 1)
    } else {
        s
    })
}

/// FN152: the NVFP4 fused back half for `b` at `shape` (`fn152_moe_back_gu*_dn*` at the EP whole
/// shape, `fn152_moe_back_h2_*` at the TP half, `fn152_moe_back_h2_far_*` with FN137's fused
/// reduce), `_lr` form: FN151's `_p1` body, so the prefetching body and [`BackTrain::CELLS`] only.
pub fn nvfp4_back_symbol(shape: ExpertShape, b: &BackTrain, far: bool) -> Result<String> {
    if !b.prefetch || !BackTrain::CELLS.contains(&(b.gu_ks, b.dn_ks)) {
        return Err(Error::Rule(format!(
            "--fn-experts-fmt nvfp4: no fused back half at ks {} / {} (FN152: the _p1 body at {:?})",
            b.gu_ks,
            b.dn_ks,
            BackTrain::CELLS
        )));
    }
    let stem = match (shape, far) {
        (ExpertShape::Whole, false) => "moe_back",
        (ExpertShape::TpHalf, false) => "moe_back_h2",
        (ExpertShape::TpHalf, true) => "moe_back_h2_far",
        (ExpertShape::Whole, true) => {
            return Err(Error::Rule(
                "--fn-experts-fmt nvfp4: the fused reduce (far) rides the TP half only".into(),
            ))
        }
    };
    // `_lr` = FN151's SC 7 body (global loads, LDS-scope slab barriers, register scales): byte-equal
    // to the default SC 1 form and faster in every back half (LEDGER FN152, probe p2)
    Ok(format!("fn152_{stem}_gu{}_dn{}_lr", b.gu_ks, b.dn_ks))
}

/// FN150: the NVFP4 cells' `gsplit` (the first output row of the record's SECOND f32 global):
/// gate_up = its N / 2 (gate rows take `gs[0]`, up rows `gs[1]`), down = its N (one global).
#[must_use]
pub const fn nvfp4_gsplit(g: MoeGemm, n: u32) -> u32 {
    match g {
        MoeGemm::GateUp => n / 2,
        MoeGemm::Down => n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FN152: every symbol the P1 route names is in the committed `moe_fn151` / `moe_fn148`
    /// manifest, and every NVFP4 back half in `moe_fn152`'s.
    #[test]
    fn fn152_the_nvfp4_p1_route_names_committed_symbols() {
        let load = |p: &str| {
            crate::manifest::Manifest::load(
                crate::prebuilt::dir().join(format!("{p}.manifest.toml")),
            )
            .expect("manifest")
        };
        let (m148, m151, m152) = (load(PLUGIN_FN148), load(PLUGIN_FN151), load(PLUGIN_FN152));
        let has = |m: &crate::manifest::Manifest, s: &str| m.kernel.iter().any(|k| k.name == s);
        let mut n151 = 0;
        for shape in [ExpertShape::Whole, ExpertShape::TpHalf] {
            for g in [MoeGemm::GateUp, MoeGemm::Down] {
                for ks in [1, 2, 4] {
                    for tm in [1, 2, 4] {
                        let c = Cell { ks, tm };
                        let Ok(s) = nvfp4_symbol_body(shape, g, c, Nvfp4Body::P1) else {
                            continue;
                        };
                        let plain = nvfp4_symbol_body(shape, g, c, Nvfp4Body::Plain).unwrap();
                        assert!(has(&m148, &plain), "{plain}");
                        if s.starts_with("fn151_") {
                            n151 += 1;
                            assert!(has(&m151, &s), "{s}");
                        } else {
                            assert_eq!(s, plain);
                            assert!(g == MoeGemm::GateUp && tm != 1, "{s}");
                        }
                    }
                }
            }
        }
        // gate_up tm 1 ks 1/2/4 at both shapes (6), whole down ks 1/2 x tm 1/2/4 less the prefill
        // cell ks 1 tm 4 (5), the half's bk64 ks 1/2 (2)
        assert_eq!(n151, 13, "FN151's default-SC cells");
        for (gu, dn) in BackTrain::CELLS {
            let b = BackTrain {
                gu_ks: gu,
                dn_ks: dn,
                prefetch: true,
            };
            for (shape, far) in [
                (ExpertShape::Whole, false),
                (ExpertShape::TpHalf, false),
                (ExpertShape::TpHalf, true),
            ] {
                let s = nvfp4_back_symbol(shape, &b, far).unwrap();
                assert!(has(&m152, &s), "{s}");
            }
            assert!(nvfp4_back_symbol(ExpertShape::Whole, &b, true).is_err());
            let p0 = BackTrain {
                prefetch: false,
                ..b
            };
            assert!(nvfp4_back_symbol(ExpertShape::Whole, &p0, false).is_err());
        }
    }

    /// FN150: every NVFP4 cell the engine can name is in `moe_fn148`'s committed manifest, a
    /// prefill cell and a missing instantiation are refused by name, and `gsplit` is FN148's.
    #[test]
    fn fn150_the_nvfp4_cells_name_committed_fn148_symbols() {
        let m = crate::manifest::Manifest::load(
            crate::prebuilt::dir().join(format!("{PLUGIN_FN148}.manifest.toml")),
        )
        .expect("moe_fn148 manifest");
        let has = |s: &str| m.kernel.iter().any(|k| k.name == s);
        let mut n = 0;
        for shape in [ExpertShape::Whole, ExpertShape::TpHalf] {
            for g in [MoeGemm::GateUp, MoeGemm::Down] {
                for ks in [1, 2, 4] {
                    for tm in [1, 2, 4, 8, 16] {
                        let c = Cell { ks, tm };
                        match nvfp4_symbol(shape, g, c) {
                            Ok(s) => {
                                assert!(has(&s), "{s} not in moe_fn148");
                                assert!(!is_prefill_cell(c));
                                n += 1;
                            }
                            Err(e) => {
                                let e = e.to_string();
                                assert!(e.contains("--fn-experts-fmt nvfp4"), "{e}");
                                if is_prefill_cell(c) {
                                    assert!(e.contains("prefill"), "{e}");
                                }
                            }
                        }
                    }
                }
            }
        }
        // whole: gate_up 9 - 1 (ks 1 tm 4 = the prefill cell) + down 6 - 1; half: 8 + 2
        assert_eq!(n, 8 + 5 + 8 + 2);
        // every cell the measured table routes at T <= 512 has an NVFP4 twin; above, refused
        for t in [1, 2, 8, 64, 512] {
            for g in [MoeGemm::GateUp, MoeGemm::Down] {
                assert!(nvfp4_symbol(ExpertShape::Whole, g, measured_cell(g, t)).is_ok());
            }
        }
        let big = measured_cell(MoeGemm::GateUp, 513);
        assert!(nvfp4_symbol(ExpertShape::Whole, MoeGemm::GateUp, big).is_err());
        let c41 = Cell { ks: 4, tm: 1 };
        assert_eq!(
            nvfp4_symbol(ExpertShape::Whole, MoeGemm::GateUp, c41).unwrap(),
            "fn148_w4a8b_bk128_ks4_tm1_g1"
        );
        let c21 = Cell { ks: 2, tm: 1 };
        assert_eq!(
            nvfp4_symbol(ExpertShape::TpHalf, MoeGemm::Down, c21).unwrap(),
            "fn148_w4a8b_bk64_ks2_tm1_g0"
        );
        assert_eq!(
            (
                nvfp4_gsplit(MoeGemm::GateUp, 1280),
                nvfp4_gsplit(MoeGemm::Down, 2560)
            ),
            (640, 2560)
        );
    }

    #[test]
    fn the_tp2_shape_key_names_committed_legal_cells() {
        let dir = crate::prebuilt::dir();
        let grouped = crate::manifest::Manifest::load(dir.join("moe_grouped.manifest.toml"))
            .expect("moe_grouped manifest");
        let tp2 = crate::manifest::Manifest::load(dir.join("moe_grouped_tp2.manifest.toml"))
            .expect("moe_grouped_tp2 manifest");
        let has = |m: &crate::manifest::Manifest, s: &str| m.kernel.iter().any(|k| k.name == s);
        let h = ExpertShape::TpHalf;
        assert_eq!((h.inter(), h.gate_up_n(), h.down_dbk()), (320, 640, 64));
        // the K law at each shape: whole down K 640 % 128, half K 320 % 64 (and NOT % 128).
        assert_eq!(
            ExpertShape::Whole.inter() % ExpertShape::Whole.down_dbk(),
            0
        );
        assert_eq!(h.inter() % h.down_dbk(), 0);
        assert_ne!(
            h.inter() % 128,
            0,
            "the control: K 320 is illegal at DBK 128"
        );
        assert_eq!(h.gate_up_n() % N_TILE, 0, "gate_up's half = whole n-blocks");
        // FN80a's half record: gate_up 819,200 + 51,200 + 640, down 409,600 + 25,600 + 2,560.
        assert_eq!(h.expert_bytes(MoeGemm::GateUp), 819_200 + 51_200 + 640);
        assert_eq!(h.expert_bytes(MoeGemm::Down), 409_600 + 25_600 + 2_560);
        assert_eq!(
            ExpertShape::Whole.expert_bytes(MoeGemm::Down),
            MoeGemm::Down.expert_bytes()
        );
        for ks in TP2_DOWN_KS {
            for body in [DecodeBody::Shipped, DecodeBody::Prefetch] {
                let c = Cell { ks, tm: 1 };
                let s = h.down_symbol(c, body).unwrap();
                assert!(has(&tp2, &s), "{s}");
                assert!(has(
                    &grouped,
                    &h.gate_up_symbol(Cell { ks: 4, tm: 1 }, body)
                ));
            }
        }
        for c in PREFILL_CELLS {
            assert!(has(
                &grouped,
                &h.down_symbol(c, DecodeBody::Prefetch).unwrap()
            ));
        }
        assert!(h
            .down_symbol(Cell { ks: 4, tm: 1 }, DecodeBody::Prefetch)
            .is_err());
        assert!(h
            .down_symbol(Cell { ks: 1, tm: 2 }, DecodeBody::Prefetch)
            .is_err());
        assert!(h
            .down_symbol(Cell { ks: 1, tm: 1 }, DecodeBody::LdsPrefetch)
            .is_err());
        for (gu, dn) in BackTrain::CELLS {
            let b = BackTrain {
                gu_ks: gu,
                dn_ks: dn,
                prefetch: true,
            };
            let s = h.back_symbol(&b).unwrap();
            assert!(has(&tp2, &s), "{s}");
            assert_eq!(
                ExpertShape::Whole.back_grid(&b, 7),
                b.grid(7),
                "the whole shape's grid is FN26's"
            );
            assert_eq!(h.back_grid(&b, 7), 1 + 7 * 5 * gu + 7 * 20 * dn);
            assert!(h
                .back_symbol(&BackTrain {
                    prefetch: false,
                    ..b
                })
                .is_err());
        }
        // nothing routes the half: the default shape is the whole expert's.
        assert_eq!(ExpertShape::default(), ExpertShape::Whole);
    }

    fn lcg(s: &mut u64) -> u64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *s >> 33
    }

    /// Top-10 distinct ids per token from a seeded stream, optionally concentrated on few experts.
    fn ids_for(tokens: u32, seed: u64, span: u32, base: u32) -> Vec<u32> {
        let mut s = seed;
        let mut out = Vec::new();
        for _ in 0..tokens {
            let mut pick: Vec<u32> = Vec::new();
            while pick.len() < TOPK as usize {
                let e = base + (lcg(&mut s) as u32 % span);
                if !pick.contains(&e) {
                    pick.push(e);
                }
            }
            out.extend(pick);
        }
        out
    }

    #[test]
    fn the_k128_law_holds_for_both_shapes() {
        // The decode body stages unmasked 128-byte K slabs (memory `mxfp4-decode-kernel-k128`):
        // 2,560 = 20 x 128 and 640 = 5 x 128. The TP-split down (K 320) would not be legal.
        for g in [MoeGemm::GateUp, MoeGemm::Down] {
            assert_eq!(g.k() % 128, 0, "{g:?}");
            assert_eq!(g.n() % N_TILE, 0, "{g:?}");
            assert_eq!(g.n() % 16, 0, "{g:?}: WPERM needs N % 16");
        }
        assert_ne!(
            (INTER / 2) % 128,
            0,
            "the control: a TP-split down's K = 320 breaks the law"
        );
        // FN2's record sub-sizes (docs/EXEC.md section 10): 1,741,080 + 872,960 = 2,615,040.
        assert_eq!(MoeGemm::GateUp.expert_bytes(), 1_638_400 + 102_400 + 1_280);
        assert_eq!(MoeGemm::Down.expert_bytes(), 819_200 + 51_200 + 2_560);
        assert_eq!(
            MoeGemm::GateUp.expert_bytes() + MoeGemm::Down.expert_bytes(),
            2_615_040
        );
    }

    #[test]
    fn the_train_is_seven_launches_and_the_shared_expert_adds_only_its_gate() {
        let t = launch_train();
        assert_eq!(t.len(), LAUNCHES_PER_MOE_BLOCK);
        // FN9: six was the brief's target, and wrong -- the combine reads a gate logit per token.
        const {
            assert!(
                LAUNCHES_PER_MOE_BLOCK == 7,
                "router, shared gate, route, 3 expert ops, combine"
            )
        };
        const { assert!(LAUNCHES_PER_MOE_BLOCK * 3 < PRIOR_FORK_LAUNCHES_PER_LAYER) };
        // The two bf16 GEMMs (router, shared-gate GEMV) are FN5's family, not a MoE code object.
        assert_eq!(t.iter().filter(|o| o.plugin.is_none()).count(), 2);
        assert!(t[1].name.starts_with("shared_gate_gemv"));
        assert_eq!(t[t.len() - 1].name, COMBINE_SYMBOL);
        // Shared on vs off: the same train (the shared expert is a tile, not a launch).
        let a = Geometry {
            tokens: 8,
            e_lo: 0,
            e_loc: 256,
            tile_blocks: 1,
            shared: true,
        };
        let b = Geometry { shared: false, ..a };
        assert!(a.tiles_cap() > b.tiles_cap());
        assert_eq!(launch_train(), launch_train());
    }

    #[test]
    fn the_layout_stays_inside_its_static_capacities() {
        let mut worst_rows = 0.0f64;
        for (tokens, span, base) in [
            (1, 512, 0),
            (8, 512, 0),
            (64, 512, 0),
            (2048, 512, 0),
            (2048, 10, 256), // every token on the same ten experts
            (33, 256, 256),
            (1000, 20, 0),
        ] {
            for (e_lo, e_loc) in [(0, 512), (0, 256), (256, 256)] {
                for tb in [1, 2, 4, 8, 16] {
                    for shared in [false, true] {
                        let g = Geometry {
                            tokens,
                            e_lo,
                            e_loc,
                            tile_blocks: tb,
                            shared,
                        };
                        let ids = ids_for(tokens, 0x5eed + u64::from(tokens), span, base);
                        let l = sort_layout(&ids, &g).expect("layout");
                        assert!(
                            l.rows <= g.rows_cap(),
                            "{g:?}: rows {} > {}",
                            l.rows,
                            g.rows_cap()
                        );
                        assert!(
                            l.tiles.len() as u32 <= g.tiles_cap(),
                            "{g:?}: tiles {} > {}",
                            l.tiles.len(),
                            g.tiles_cap()
                        );
                        worst_rows = worst_rows.max(f64::from(l.rows) / f64::from(g.rows_cap()));
                        // Every live row is covered by exactly one tile; tiles are block-aligned
                        // and own one expert; pad rows are -1.
                        let mut cover = vec![0u8; l.rows as usize];
                        for &(e, r0, n) in &l.tiles {
                            assert_eq!(r0 % BLOCK_ROWS, 0);
                            assert!(n >= 1 && n <= BLOCK_ROWS * tb);
                            assert!(e <= e_loc);
                            for r in r0..r0 + n {
                                cover[r as usize] += 1;
                            }
                        }
                        for (r, &c) in cover.iter().enumerate() {
                            assert_eq!(c == 1, l.row_tok[r] >= 0, "{g:?} row {r}");
                            assert!(c <= 1);
                        }
                        // Routes of one expert in token order.
                        for (r, &p) in l.route_pos.iter().enumerate() {
                            if p >= 0 {
                                assert_eq!(l.row_tok[p as usize], (r as u32 / TOPK) as i32);
                            }
                        }
                    }
                }
            }
        }
        assert!(worst_rows <= 1.0);
    }

    #[test]
    fn a_tile_count_over_the_capacity_is_what_the_capacity_prevents() {
        // NEGATIVE CONTROL: the capacity without the per-expert pad term is too small for a
        // decode step whose ten routes land on ten distinct experts -- the case it exists for.
        let g = Geometry {
            tokens: 1,
            e_lo: 0,
            e_loc: 512,
            tile_blocks: 1,
            shared: false,
        };
        let l = sort_layout(&ids_for(1, 3, 512, 0), &g).expect("layout");
        assert_eq!(l.rows, 160, "ten experts x one 16-row block");
        assert!(
            l.rows > g.routes(),
            "routes alone would under-size the arrays"
        );
        assert_eq!(l.rows, g.rows_cap());
    }

    #[test]
    fn every_measured_cell_is_committed_and_bands_are_ordered() {
        for w in MEASURED_CELLS.windows(2) {
            assert!(w[0].0 < w[1].0);
        }
        for (_, a, b) in MEASURED_CELLS {
            let known = |c: Cell| CELLS.contains(&c) || PREFILL_CELLS.contains(&c);
            assert!(known(a) && known(b));
            assert_eq!(
                a.tm, b.tm,
                "one route per chunk: both GEMMs share tile_blocks"
            );
        }
        assert_eq!(measured_cell(MoeGemm::GateUp, 1), Cell { ks: 4, tm: 1 });
        assert_eq!(measured_cell(MoeGemm::Down, 8), Cell { ks: 1, tm: 1 });
        assert_eq!(measured_cell(MoeGemm::GateUp, 2048), Cell { ks: 1, tm: 4 });
    }

    #[test]
    fn the_fused_train_is_three_launches_and_its_cells_are_the_measured_split_class() {
        // FN26: 7 -> 3 launches; the fused half takes the SAME split pair as the seven-launch
        // train at every token count it serves (a different pair would be numerics).
        let t = launch_train_fused();
        assert_eq!(t.len(), LAUNCHES_PER_MOE_BLOCK_FUSED);
        const { assert!(LAUNCHES_PER_MOE_BLOCK_FUSED <= 3) };
        assert_eq!(t[0].name, launch_train()[0].name, "the router GEMM stays");
        assert_eq!(t[1].name, ROUTE_SYMBOL);
        assert!(t[2].name.starts_with("moe_back_"));
        for tokens in 1..=BackTrain::MAX_TOKENS {
            let b = BackTrain::for_tokens(tokens, true).expect("a fused cell");
            assert_eq!(b.gu_ks, measured_cell(MoeGemm::GateUp, tokens).ks);
            assert_eq!(b.dn_ks, measured_cell(MoeGemm::Down, tokens).ks);
        }
        assert_eq!(
            BackTrain::for_tokens(1, false)
                .map(|b| b.symbol())
                .as_deref(),
            Some("moe_back_gu4_dn2_p0")
        );
        assert_eq!(
            BackTrain::for_tokens(4, true)
                .map(|b| b.symbol())
                .as_deref(),
            Some("moe_back_gu1_dn1_p1")
        );
        assert!(BackTrain::for_tokens(BackTrain::MAX_TOKENS + 1, true).is_none());
        assert!(BackTrain::for_tokens(0, true).is_none());
        // The grid and the sync words at T = 1 (tiles_cap 12 with the shared expert).
        let g = Geometry {
            tokens: 1,
            e_lo: 0,
            e_loc: 256,
            tile_blocks: 1,
            shared: true,
        };
        let b = BackTrain::for_tokens(1, false).unwrap();
        assert_eq!(g.tiles_cap(), 12);
        assert_eq!(b.grid(g.tiles_cap()), 1 + 12 * 10 * 4 + 12 * 20 * 2);
        assert_eq!(BackTrain::sync_words(12), 3 + 20 + 36);
    }

    #[test]
    fn the_front_half_is_two_launches_at_the_rules_router_split() {
        // FN31: router GEMM + route in one launch, then FN26's back half: 2 launches a block.
        let t = launch_train_front();
        assert_eq!(t.len(), LAUNCHES_PER_MOE_BLOCK_FRONT);
        assert!(t[0].name.starts_with("moe_front_") && t[1].name.starts_with("moe_back_"));
        // The split class is the rules' `moe.router` band at every T the front half serves (the
        // 33..64 band is the drafter's m32n16 = the split-K band's arithmetic at ONE split).
        let rules = crate::flashnext::FlashNextRules::prod().expect("rules");
        for tokens in 1..=FrontTrain::KERNEL_MAX_TOKENS {
            let (launch, splits) = rules
                .bgemm_route("moe.router", tokens, EXPERTS)
                .expect("a router band");
            assert_eq!(
                router_split(tokens),
                splits,
                "T {tokens}: {}",
                launch.symbol
            );
            let Some(f) = FrontTrain::for_tokens(tokens) else {
                assert!(tokens > FrontTrain::MAX_TOKENS);
                continue;
            };
            assert_eq!(f.split, splits, "T {tokens}: {}", launch.symbol);
            assert!(
                launch.symbol == "fn_bgemm_sk_m16" || launch.symbol == "dflash_bgemm_bf16_m32n16",
                "T {tokens}: the front half derives FN5's split-K band"
            );
            assert_eq!(f.grid(tokens), [4, tokens.div_ceil(16), splits]);
            assert_eq!(f.threads(), 256);
            assert_eq!(f.part_floats(tokens), u64::from(splits * tokens * 512));
        }
        assert_eq!(
            FrontTrain::for_tokens(1).unwrap().symbol(),
            "moe_front_e512_nw8_s4_dpp"
        );
        assert_eq!(
            FrontTrain::for_tokens(8).unwrap().symbol(),
            "moe_front_e512_nw8_s2_dpp"
        );
        // FN49b: the control names the shipped (ds_bpermute) text
        let ctl = FrontTrain {
            route: RouteForm::Shipped,
            ..FrontTrain::for_tokens(1).unwrap()
        };
        assert_eq!(ctl.symbol(), "moe_front_e512_nw8_s4");
        assert!(FrontTrain::for_tokens(0).is_none());
        assert!(
            FrontTrain::for_tokens(12).is_none(),
            "measured losing above 8"
        );
        const { assert!(FrontTrain::MAX_TOKENS <= FrontTrain::KERNEL_MAX_TOKENS) };
        assert_eq!(FrontTrain::KERNEL_MAX_TOKENS, BackTrain::MAX_TOKENS);
        // The back half's routed prefetching body is a suffix of the body table.
        let b = BackTrain::for_tokens(4, true).unwrap();
        assert_eq!(b.symbol(), b.symbol_body(BACK_PREFETCH_BODY));
        assert_eq!(
            b.symbol_body(DecodeBody::LdsPrefetch),
            "moe_back_gu1_dn1_p2"
        );
        assert_eq!(b.symbol_body(DecodeBody::Ring3), "moe_back_gu1_dn1_p3");
    }

    #[test]
    fn the_route_band_is_the_measured_one() {
        // FN31: the LDS route at T 1..16 (measured winning, byte-identical); v2 is now an arm.
        // FN49b: the DPP lowering is the routed form; the shipped text is the control.
        for t in 1..=ROUTE_LDS_MAX_TOKENS {
            assert_eq!(route_symbol(t), ROUTE_SYMBOL_LDS_DPP);
            assert_eq!(route_symbol_form(t, RouteForm::Shipped), ROUTE_SYMBOL_LDS);
        }
        assert_eq!(route_symbol_form(17, RouteForm::Shipped), ROUTE_SYMBOL);
        assert_eq!(route_symbol(17), ROUTE_SYMBOL, "unmeasured 17..32: FN4's");
        assert_eq!(
            route_symbol(40),
            ROUTE_SYMBOL,
            "v2 measured losing at 40, LDS ties"
        );
        assert_eq!(route_symbol(2048), ROUTE_SYMBOL);
    }

    #[test]
    fn a_body_is_a_suffix_and_the_routed_body_keeps_the_launch_law() {
        let c = Cell { ks: 4, tm: 1 };
        assert_eq!(
            grouped_symbol_body(MoeGemm::GateUp, c, DecodeBody::Shipped),
            "moe_grouped_w4a8_bk128_ks4_tm1_g1"
        );
        assert_eq!(
            grouped_symbol_body(MoeGemm::Down, c, DecodeBody::Prefetch),
            "moe_grouped_w4a8_bk128_ks4_tm1_g0_p1"
        );
        // tm > 1 cells and the prefill body have no FN26 body.
        assert_eq!(
            grouped_symbol_body(MoeGemm::GateUp, Cell { ks: 2, tm: 2 }, DecodeBody::Prefetch),
            "moe_grouped_w4a8_bk128_ks2_tm2_g1"
        );
        assert_eq!(
            grouped_symbol_body(MoeGemm::GateUp, Cell { ks: 1, tm: 4 }, DecodeBody::Prefetch),
            "moe_grouped_w4a8_pf_tm4_g1"
        );
        // Only a 128-wide body keeps FN4's grid (the engine's launch law): the routed one must.
        assert_eq!(DECODE_BODY.n_width(), N_TILE);
        assert_eq!(DecodeBody::N64.n_width(), 64);
        assert_eq!(
            grouped_symbol(MoeGemm::GateUp, c),
            grouped_symbol_body(MoeGemm::GateUp, c, DECODE_BODY)
        );
    }

    #[test]
    fn geometry_refuses_what_the_kernels_cannot_run() {
        let ok = Geometry {
            tokens: 4,
            e_lo: 256,
            e_loc: 256,
            tile_blocks: 4,
            shared: true,
        };
        assert!(ok.validate().is_ok());
        for bad in [
            Geometry { tokens: 0, ..ok },
            Geometry { e_loc: 0, ..ok },
            Geometry { e_lo: 300, ..ok },
            Geometry {
                tile_blocks: 3,
                ..ok
            },
            Geometry {
                tile_blocks: 32,
                ..ok
            },
            Geometry {
                e_lo: 0,
                e_loc: 513,
                ..ok
            },
        ] {
            assert!(bad.validate().is_err(), "{bad:?} accepted");
        }
        assert!(sort_layout(&[1, 2, 3], &ok).is_err(), "a short id list");
    }

    #[test]
    fn fn84_miss_order_names_kint_and_rules_rows() {
        let r = crate::flashnext::FlashNextRules::prod().expect("prod rules");
        for o in MissOrder::ALL {
            assert_eq!(MissOrder::parse(o.name()), Some(o));
        }
        assert_eq!(MissOrder::default(), MissOrder::Id);
        assert_eq!(MissOrder::parse("nope"), None);
        for t in 1..=BackTrain::MAX_TOKENS {
            assert_eq!(MissOrder::Id.kint(&r, t), Some(-1));
            assert_eq!(MissOrder::First.kint(&r, t), Some(0));
            for o in [MissOrder::Capped, MissOrder::CappedPre] {
                let k = o
                    .kint(&r, t)
                    .expect("a [[moe_miss]] row holds every fused T");
                assert!(k >= 1, "T {t}: k {k}");
            }
        }
    }

    #[test]
    fn symbols_name_the_committed_code_objects() {
        let dir = crate::prebuilt::dir();
        let grouped = crate::manifest::Manifest::load(dir.join("moe_grouped.manifest.toml"))
            .expect("moe_grouped manifest");
        let route = crate::manifest::Manifest::load(dir.join("moe_route.manifest.toml"))
            .expect("moe_route manifest");
        let has = |m: &crate::manifest::Manifest, s: &str| m.kernel.iter().any(|k| k.name == s);
        for c in CELLS.into_iter().chain(PREFILL_CELLS) {
            if is_prefill_cell(c) {
                assert!(grouped_symbol(MoeGemm::GateUp, c).contains("_pf_"));
            }
            for g in [MoeGemm::GateUp, MoeGemm::Down] {
                assert!(
                    has(&grouped, &grouped_symbol(g, c)),
                    "{}",
                    grouped_symbol(g, c)
                );
            }
        }
        assert!(has(&grouped, SILU_SYMBOL));
        for s in [
            ROUTE_SYMBOL,
            COMBINE_SYMBOL,
            a16_symbol(MoeGemm::GateUp),
            a16_symbol(MoeGemm::Down),
        ] {
            assert!(has(&route, s), "{s}");
        }
        // FN4's decode ks1_tm4 symbols stay in the object (the FN20 oracle's control).
        for g in [1, 0] {
            assert!(has(
                &grouped,
                &format!("moe_grouped_w4a8_bk128_ks1_tm4_g{g}")
            ));
        }
        // FN26: 3 split counts x 2 A modes x 3 bodies (`_p1`, `_n64_p0`, `_n64_p1`) + 2 x 2 fused
        // back halves; FN31: 3 x 2 x 2 bodies (`_p2`, `_p3`) + 2 x 2 back halves; FN84: 2 orders
        // (`mo`, `mop`) x 2 back-half cells.
        assert_eq!(
            grouped.kernel.len(),
            2 * CELLS.len()
                + 1
                + 2 * (PREFILL_CELLS.len() - 1)
                + 2
                + 3 * 2 * 3
                + 4
                + 3 * 2 * 2
                + 4
                + 4
        );
        for (gu, dn) in BackTrain::CELLS {
            let b = BackTrain {
                gu_ks: gu,
                dn_ks: dn,
                prefetch: true,
            };
            for o in MissOrder::ALL {
                match o.symbol(&b) {
                    None => assert_eq!(o, MissOrder::Id),
                    Some(s) => {
                        assert!(has(&grouped, &s), "{s}");
                        assert_eq!(
                            crate::abi::signature_for(&s).map(<[_]>::len),
                            Some(crate::abi::SIG_MOE_BACK.len() + 3),
                            "{s}"
                        );
                    }
                }
            }
        }
        for c in CELLS.into_iter().filter(|c| c.tm == 1) {
            for g in [MoeGemm::GateUp, MoeGemm::Down] {
                for b in [
                    DecodeBody::Shipped,
                    DecodeBody::Prefetch,
                    DecodeBody::N64,
                    DecodeBody::N64Prefetch,
                    DecodeBody::LdsPrefetch,
                    DecodeBody::Ring3,
                ] {
                    let s = grouped_symbol_body(g, c, b);
                    assert!(has(&grouped, &s), "{s}");
                }
            }
        }
        for (gu, dn) in BackTrain::CELLS {
            for prefetch in [false, true] {
                let b = BackTrain {
                    gu_ks: gu,
                    dn_ks: dn,
                    prefetch,
                };
                assert!(has(&grouped, &b.symbol()), "{}", b.symbol());
                for body in [DecodeBody::LdsPrefetch, DecodeBody::Ring3] {
                    assert!(
                        has(&grouped, &b.symbol_body(body)),
                        "{}",
                        b.symbol_body(body)
                    );
                }
            }
        }
        for c in PREFILL_CELLS {
            assert!(is_prefill_cell(c));
            assert_eq!(dense_oracle_symbol(c, 64), PREFILL_PARENT_SYMBOL);
        }
        // Only (ks 1, tm 4) is shared with the decode grid; the decode rows (tm 1) never are.
        assert_eq!(CELLS.iter().filter(|c| is_prefill_cell(**c)).count(), 1);
        assert_eq!(
            route.kernel.len(),
            5 + 1 + 8 + 4 + 1,
            "FN26 added the v2 route; FN31 the LDS route + 8 front halves; FN49a the DPP route + 3 DPP fronts; FN111 the 16-wave route"
        );
        assert!(has(&route, ROUTE_SYMBOL_LDS_DPP));
        for sp in [1, 2, 4] {
            assert!(has(&route, &format!("moe_front_e512_nw8_s{sp}_dpp")));
        }
        assert!(has(&route, ROUTE_SYMBOL_V2));
        assert!(has(&route, ROUTE_SYMBOL_LDS));
        for tokens in 1..=FrontTrain::MAX_TOKENS {
            let f = FrontTrain::for_tokens(tokens).unwrap();
            assert!(has(&route, &f.symbol()), "{}", f.symbol());
            // The control arm (the band's own loop and schedule) exists at 4 waves.
            let control = FrontTrain {
                unrolled: false,
                waves: 4,
                ..f
            };
            assert!(has(&route, &control.symbol()), "{}", control.symbol());
            // FN49b: the route control's front half exists too
            let shipped = FrontTrain {
                route: RouteForm::Shipped,
                ..f
            };
            assert!(has(&route, &shipped.symbol()), "{}", shipped.symbol());
        }
        // The flag table: the route object is contraction-free, the grouped one is the decode
        // body's own flags (and so contracts exactly where it does).
        assert!(
            route.flags.ends_with("-ffp-contract=off"),
            "{}",
            route.flags
        );
        assert!(!grouped.flags.contains("ffp-contract"), "{}", grouped.flags);
        // The oracle's dense symbols exist for every row count up to 64 at every split.
        let dec = crate::manifest::Manifest::load(dir.join("mxfp4_gemm_decode.manifest.toml"))
            .expect("decode manifest");
        let folded =
            crate::manifest::Manifest::load(dir.join("mxfp4_gemm_folded_mt.manifest.toml"))
                .expect("folded_mt manifest");
        assert!(has(&folded, PREFILL_PARENT_SYMBOL));
        for c in CELLS.into_iter().filter(|c| !is_prefill_cell(*c)) {
            for rows in [1, 16, 17, 64] {
                assert!(
                    has(&dec, &dense_oracle_symbol(c, rows)),
                    "{}",
                    dense_oracle_symbol(c, rows)
                );
            }
        }
    }
}
