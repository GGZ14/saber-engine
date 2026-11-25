//! The Inkling 2 Mini engine path (INK2, `docs/INKLING2.md` §9): `saber engine --pack <inkling2
//! pack>` boots this module when the pack's manifest names `"model": "inkling2_loop"` (the FN12
//! model-selection law; the 27B and Flash-Next paths are untouched and refuse `--ink-*`).
//!
//! What exists (host only, no card touched): the boot reads the pack (manifest, per-rank and shared
//! dense records, the two expert tiers), re-derives the pack's plan and checks its sha256, and runs
//! the WORKER-PROCESS form (rank 0 spawns rank 1 with `--ink-rank 1 --ink-peer-port P`; the pair
//! meet over `saber_exec::TcpExchange`, Flash-Next's transport; `--ink-one-process` = two threads
//! over `ThreadPairExchange`, the control). Each rank bills its weights + the engine plan's runtime
//! classes (`saber_model::inkling2::engine_plan`: the per-visit arena, 8 paged global KV pools, 40
//! local rings, 192 conv states a sequence, the reduce scratch) into a `saber_mem::ledger::Ledger`
//! against `--ink-budget-mib` less the kernel's P2PDMA pool; the pair swaps the geometry digest and
//! rank 1's ledger, and rank 0 prints both (`inkbudget rank<r>:`).
//!
//! `--ink-dry-run` is the host form; INK7's executor (`saber_cli::inkling2_exec`) runs with
//! `--ink-run tf|decode` (`--tp 1` on one card, `--tp 2 --ink-run tf` = the two-card process); any
//! other boot is refused by name (no serve plane yet).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context};
use saber_exec::{IpcExchange, RankPeer, TcpExchange, ThreadPairExchange};
use saber_mem::ledger::{Ledger, Pool};
use saber_model::inkling2::engine_plan::{mib, EngineGeom, RankPlan};
use saber_model::inkling2::{layer_graph, ExpertForm, Inkling2Config, PackPlan};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

/// The pack format this path reads.
pub const PACK_FORMAT: &str = "ink1-pack/1";
/// INK6: the native pack's manifest format (`build_inkling2_pack.py` on a `--form native` plan).
pub const NATIVE_PACK_FORMAT: &str = "ink6-pack/1";
/// INK41: the manifest `layout` of the one-card view (`tools/pack/build_ink2_tp1.py`).
pub const TP1_VIEW_LAYOUT: &str = "tp1-view";
/// INK41: the view's directory inside a published pack (its `blobs` = "..").
pub const TP1_VIEW_DIR: &str = "tp1";
/// INK41: the view's 8-bit trunk index (`ink41-trunk8-view/1`), in the view's directory.
pub const TRUNK8_VIEW_FORMAT: &str = "ink41-trunk8-view/1";

/// INK41: true when `dir` holds the one-card view's manifest (no other check).
pub fn is_tp1_view(dir: &Path) -> bool {
    std::fs::read_to_string(dir.join("manifest.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .is_some_and(|m| m["layout"].as_str() == Some(TP1_VIEW_LAYOUT))
}
/// Flash-Next's default card fence (MiB), the same 32 GB card.
pub const DEFAULT_INK_BUDGET_MIB: u64 = 30_500;

/// INK13: the served drafter's depth (`--ink-mtp`; 0 = off, the control).
/// INK15: `--ink-prefill-gemm` -- `pf` IS THE DEFAULT (BetterBench --quick TTFT p50 184 / 185 ->
/// 125 / 125 ms, conc-8 198.4 / 197.0 -> 202.9 / 202.7 t/s, rows byte-equal); `base` = the control.
/// INK16: `cl` IS THE DEFAULT (the column-lane GEMM at 9..256 rows; BetterBench --quick TTFT p50
/// 127 / 126 -> 114 / 113 ms, conc-8 TTFT p50 205 / 207 -> 195 / 196, rows byte-equal); `pf` = the
/// control, `clp` = + the push reduce (an arm).
pub const DEFAULT_INK_PREFILL_GEMM: &str = "cl";
/// INK25: `--ink-prefill-trunk` -- `iu8` = the 8-bit trunk's prefill GEMMs (local q / k / v / r /
/// sink / gate / o, > 8 rows) on the matrix cores with int8 activations a 32-k block (`ink_i8`; a
/// numerics change: prefill rows move, decode rows do not). INK26 (the user's decision): **`iu8` IS
/// THE DEFAULT** (TTFT p50 -17 % single; first-token KL max 0.0245 at p04 vs the 0.0227 bar, mean
/// unchanged); `f32` (the byte-law column-lane / pf bodies) = the control.
pub const DEFAULT_INK_PREFILL_TRUNK: &str = "iu8";
/// INK27 / INK31: `--ink-prefill-attn` -- `rt8` (the row-tiled body, byte-equal to v1) IS THE
/// DEFAULT since INK31 (under the routing floor below); `v1` (one workgroup a row and head) = the
/// control and the off-switch; `rt16` / `rt8t` = arms.
/// CONS40: `wm` (INK35's matrix-core flash attention, a numerics arm KL-gated by INK35) IS THE
/// DEFAULT at floor 0; `rt8` at floor 256 = the pre-CONS40 default (in `CONS40_CONTROL`).
pub const DEFAULT_INK_PREFILL_ATTN: &str = "wm";
/// INK31: the row-tiled prefill attention's routing floor -- a chunk routes `ink_fa_rt<R>` only when
/// its last position is >= this (short chunks keep v1's launch shape; 0 = INK27's no floor).
pub const DEFAULT_INK_PREFILL_ATTN_FLOOR: u64 = 0;
/// INK29: `--ink-p29` -- the prefill levers of `hip/ink_p29.hip` (a comma list; empty = INK26's
/// bodies): `kx` (the iu8 GEMM in K slices, byte-equal), `rs` (the route sort a workgroup an
/// expert, byte-equal), `fq` (out_gate + the o input's q8 in one launch, byte-equal), `mt` (32 / 64-row expert
/// tiles on fn148 tm2 / tm4, byte-equal), `mz` (the message-gate z with its windows prefetched,
/// byte-equal), and the
/// NUMERICS arms `ls` (least-squares activation scales), `b16` (16-k activation blocks), `bw` (the
/// global layers' bf16 projections on the bf16 matrix cores), `i8in` / `i8o` (iu8 on the input
/// projections only / on o only).
/// INK29 DEFAULT (the gated set): the byte-equal levers `kx,rs,fq,mt,mz` + the numerics pair
/// `b16,bw` (first-token KL vs the oracle mean 0.00108 / max 0.0135, 0 of 40 over 0.0227 -- the
/// iu8 default before it 0.00133 / 0.0245, the f32 control 0.00130 / 0.0223; FN80d envelope PASS);
/// `INK29_CONTROL` (`--ink-p29 none`) = INK26's bodies.
pub const DEFAULT_INK29: &str = "kx,rs,fq,mt,mz,b16,bw";
pub const INK29_CONTROL: &str = "--ink-p29 none";
/// INK35: `--ink-p35` -- the prefill levers of `hip/ink_fa35.hip` beyond the attention (a comma
/// list; empty = off, the control): `mzs` = the message-gate z of a prefill chunk on the bf16
/// matrix cores with the normed activations split hi + lo (~f32 accuracy; a NUMERICS arm).
/// CONS40: `mzs` IS THE DEFAULT (`--ink-p35 none` = the control).
pub const DEFAULT_INK35: &str = "mzs";
/// INK36: `--ink-p36` -- the prefill chunk's schedule levers (a comma list; empty = off, the
/// control; every item is byte-equal by construction and gated as such): `ov` = the chunk as two
/// row halves in flight on two streams (half B one visit behind half A, each half's reduces on its
/// own link words), `hg` = the visit region zeroed per buffer at its take (only the zeroed takes,
/// merged runs) instead of the whole dirty region at every visit, + the slot rows of a continued
/// chunk from a per-slot device table (no blocking write a visit); `hgp` = `hg` with every plain
/// take POISONED (0xFF: the gate's control that the skipped zeroing was dead); `xw<N>` = the
/// exchange's workgroup cap under `ov` (default 128).
/// CONS40: `hg` IS THE DEFAULT (byte-equal; `--ink-p36 none` = the control).
pub const DEFAULT_INK36: &str = "hg";
/// INK37: `--ink-p37` -- byte-equal prefill levers (`hip/ink_p37.hip` + `moe_fn153`; empty = off,
/// the control): `moe` (FN153's grouped prefill tile `pf_tm4` for the NVFP4 experts), `moe8` (its 128-row
/// tile at 1,024-row chunks, an arm), `glue` / `glue8` (the router on 4 / 8 rows a workgroup).
/// CONS40: `moe,glue` IS THE DEFAULT (byte-equal; `--ink-p37 none` = the control).
pub const DEFAULT_INK37: &str = "moe,glue";
/// CONS40: `--ink-p40` -- byte-equal prefill levers on the consolidated stack (empty = off, the
/// control): `fa128` = the matrix-core prefill attention (`wm`) on 128 rows a workgroup
/// (`ink_fa35_wm128`) in the GLOBAL layers of a chunk starting at >= 2,048 (`fam<N>` sets the start),
/// `fa128l` = also the local-window layers; `zd1` / `zd2` / `zd` = under `--ink-p36 hg`, the named
/// visit takes whose producers write every byte read are not zeroed (`zp` poisons them: the control).
/// CONS40: `fa128l,zd` IS THE DEFAULT (byte-equal, zd's poison control `zp` equal; `--ink-p40 none` = off).
pub const DEFAULT_INK40: &str = "fa128l,zd";
/// CONS40: the rollback to the pre-CONS40 prefill (rt8 at floor 256, every INK35..CONS40 lever off).
pub const CONS40_CONTROL: &str =
    "--ink-prefill-attn rt8 --ink-prefill-attn-floor 256 --ink-p35 none \
     --ink-p36 none --ink-p37 none --ink-p40 none";
/// INK39: `--ink-p39` (the multi-verify levers): `cat` IS THE DEFAULT (the merged projection GEMMs
/// of a > 8-row multi verify, byte-equal; conc-8 +4.2 % paired); `push` stays an arm (measured
/// losing). `INK39_CONTROL` (`--ink-p39 none`) = the pre-INK39 multi verify, the rollback.
pub const DEFAULT_INK_P39: &str = "cat";
pub const INK39_CONTROL: &str = "--ink-p39 none";
/// INK34: the single-stream drafter IS THE GATED 7-DEEP DRAFT -- `--ink-mtp 7 --ink-mtp-gate
/// prod:0.6:2:3` (INK33's recommendation: served single +12..14 %, conc-8 / TTFT tied; re-gated on
/// the consolidated stack, LEDGER INK34). It needs the pruned draft vocabulary (`ink_dr6` reads the
/// drafts' top-1 probabilities over it): without one (or with `--ink-mtp-gate off`) an unset
/// `--ink-mtp` is INK13's 2 (`DEFAULT_INK_MTP_UNGATED`). `INK34_MTP_CONTROL` = the pre-INK34
/// single-stream drafter. The batched table (`--ink-mtp-batch`) is unchanged.
pub const DEFAULT_INK_MTP: u64 = 7;
pub const DEFAULT_INK_MTP_UNGATED: u64 = 2;
pub const DEFAULT_INK_MTP_GATE: &str = "prod:0.6:2:3";
pub const INK34_MTP_CONTROL: &str = "--ink-mtp 2 --ink-mtp-gate off";
/// INK18: the drafter's defaults -- the pruned vocabulary (32,768 ids), the split-K GEMV and the
/// 8-bit block layers ARE THE DEFAULT (BetterBench --quick A B B A: see LEDGER INK18; the
/// committed tokens are the control's by the drawn-row walk); `--ink-draft-vocab 0
/// --ink-draft-gemv base --ink-draft-layers bf16` = the control.
pub const DEFAULT_INK_DRAFT_VOCAB: u64 = 32_768;
pub const DEFAULT_INK_DRAFT_GEMV: &str = "sk";
/// INK22: INK21's stacked arm C as the serve defaults (INK26: the user's flip; the control =
/// [`INK22_CONTROL`]).
pub const DEFAULT_INK_MTP_BATCH: &str = "2:2,4:2,8:2";
pub const DEFAULT_INK_DRAFT_MM: &str = "wmma";
/// INK30: + `r1` (INK28's one-workgroup route, widened to <= 32 rows by INK30; flipped on the
/// h3-h6 BetterBench pairs; the control = `--ink-fuse all17,cl8`).
pub const DEFAULT_INK_FUSE: &str = "all17,cl8,r1";
pub const DEFAULT_INK_BATCH_PLAIN: &str = "rel,first3";
pub const DEFAULT_INK_BATCH_APPEND: &str = "graph,pfmm";
/// INK26: the pre-flip serve defaults as flags (the rollback / the paired control E): INK22's five
/// constants at their controls + INK25's trunk at `f32`.
pub const INK22_CONTROL: &str = "--ink-mtp-batch off --ink-draft-mm base --ink-fuse all17 \
     --ink-batch-plain first2 --ink-batch-append each";
pub const INK26_CONTROL: &str = "--ink-mtp-batch off --ink-draft-mm base --ink-fuse all17 \
     --ink-batch-plain first2 --ink-batch-append each --ink-prefill-trunk f32";
pub const DEFAULT_INK_DRAFT_LAYERS: &str = "w8";
/// INK16: `--ink-first-token` -- `seam` IS THE DEFAULT (BetterBench --quick TTFT p50 116 / 115 ->
/// 110 / 107 ms, conc-8 TTFT p50 198 / 195 -> 192 / 185; served ids / seeds / max_tokens equal to
/// `dense`); `dense` = the control.
pub const DEFAULT_INK_FIRST_TOKEN: &str = "seam";
/// INK14: the trunk an unset `--ink-trunk` resolves to when the pack carries the overlay.
pub const DEFAULT_INK_TRUNK: &str = "w8-head";

/// The Inkling 2 path's flags (`--ink-*`); a 27B or Flash-Next pack refuses any of them off its
/// default.
#[derive(clap::Args, Clone, Debug, PartialEq, Eq)]
pub struct InkFlags {
    /// INK2: size everything against the pack and the budget on the host, touch no card, exit
    #[arg(long)]
    pub ink_dry_run: bool,
    /// INK2: the two ranks as two threads of this process (the worker form's control)
    #[arg(long)]
    pub ink_one_process: bool,
    /// Sequences the state pools hold
    #[arg(long, default_value_t = 8)]
    pub ink_max_seqs: u32,
    /// Per-sequence context the global KV pools hold
    #[arg(long, default_value_t = 32_768)]
    pub ink_max_ctx: u64,
    /// Rows of the widest prefill chunk (the arena's prefill form)
    #[arg(long, default_value_t = 2_048)]
    pub ink_max_chunk: u64,
    /// Rows of a global KV page
    #[arg(long, default_value_t = 64)]
    pub ink_page_rows: u64,
    /// Split-K partitions a decode attention row's partials are sized for
    #[arg(long, default_value_t = 16)]
    pub ink_decode_splits: u64,
    /// The card fence (MiB) before the P2PDMA pool's discount
    #[arg(long, default_value_t = DEFAULT_INK_BUDGET_MIB)]
    pub ink_budget_mib: u64,
    /// Worker form: this process's rank (rank 0 spawns rank 1)
    #[arg(long, default_value_t = 0, hide = true)]
    pub ink_rank: u32,
    /// Worker form: the port rank 0 listens on (rank 1 only)
    #[arg(long, default_value_t = 0, hide = true)]
    pub ink_peer_port: u16,
    /// INK7: run the one-card executor (`saber_cli::inkling2_exec`): `tf` = the teacher-forced
    /// prefill rows of every `--ink-only` prompt (prompt + the oracle's greedy tokens), `decode` =
    /// the INK5 gates (eager decode == teacher-forced rows, `--ink-capture` replay == eager)
    #[arg(long)]
    pub ink_run: Option<String>,
    /// INK7: the oracle run dir (`<id>/meta.json`, `gen_top_ids.npy`)
    #[arg(long)]
    pub ink_ref: Option<PathBuf>,
    /// INK7: where the rows (`<id>.<tag>.{tf,dec,graph}.f32`) and the report go
    #[arg(long)]
    pub ink_out: Option<PathBuf>,
    /// INK7: the prompts (comma-separated ids)
    #[arg(long, default_value = "")]
    pub ink_only: String,
    /// INK7: decode steps a prompt
    #[arg(long, default_value_t = 24)]
    pub ink_steps: u32,
    /// INK12: `--ink-run vprice`'s verify widths (rows of one sequence; 1 = the b1 decode form)
    #[arg(long, default_value = "1,2,3,4,5,6,8")]
    pub ink_vprice_widths: String,
    /// INK19: `--ink-run vprice`'s batched decode buckets (one row a sequence, the served b > 1
    /// decode graph, `batch` form) priced beside the widths; empty = none
    #[arg(long, default_value = "")]
    pub ink_vprice_batch: String,
    /// INK20: `--ink-run mverify`'s multi-sequence verify cells (`BxW` = B sequences x W chained
    /// rows; each sequence's rows gated byte-equal to its own single-sequence verify)
    #[arg(long, default_value = "2x3,4x3,8x3")]
    pub ink_mseq: String,
    /// INK20: `--ink-run mverify`'s verify rounds a sequence (sequence s runs rounds - s % 4, so
    /// the live count walks down)
    #[arg(long, default_value_t = 12)]
    pub ink_mseq_rounds: u32,
    /// INK20: the serve plane's batched speculation (`--ink-serve`): drafts a verify by live
    /// count, `B:K,...` (e.g. `2:2,4:2,8:2`; a live count takes the largest entry <= it, K 0 =
    /// plain decode); every live sequence keeps its own drafter context and the step runs ONE
    /// multi-sequence verify. INK22: **`2:2,4:2,8:2` IS THE DEFAULT** (stands down by name at
    /// `--ink-serve-seqs 1`); `off` (or empty) = OFF, the control (drafts only at one live)
    #[arg(long, default_value = DEFAULT_INK_MTP_BATCH)]
    pub ink_mtp_batch: String,
    /// INK33: the confidence-gated draft depth at ONE live sequence (an ARM; empty = OFF = the
    /// fixed depth `--ink-mtp K`, the control): `prod:TAU:DMIN` / `each:TAU:DMIN` -- the drafter
    /// drafts at `--ink-mtp K` (up to 7) and reports each draft's top-1 probability over its
    /// scored vocabulary (`ink_dr6`); the round verifies the leading drafts whose running product
    /// (`prod`) or own probability (`each`) is >= TAU, at least DMIN. The walk is unchanged, so
    /// the committed tokens are the non-speculative stream's either way (the MTP law).
    /// INK34: `auto` (THE DEFAULT) = `DEFAULT_INK_MTP_GATE` in the serve plane when `--ink-mtp` is
    /// unset and the pruned draft vocabulary is on, else off; the `--ink-run` harness reads `auto`
    /// as off; `off` = INK33's control.
    #[arg(long, default_value = "auto")]
    pub ink_mtp_gate: String,
    /// INK38: the confidence-gated depth of a BATCHED round (an ARM; `off` = THE DEFAULT = the
    /// fixed depth of `--ink-mtp-batch`, the control): `prod:TAU:DMIN:AGG` / `each:TAU:DMIN:AGG`,
    /// AGG `max` | `min` | `mean` -- the batched drafter drafts at the table's K and reports every
    /// sequence's drafts' top-1 probabilities (`ink_dr6` a sequence); each sequence's gated depth
    /// is INK33's rule, and the round verifies ONE depth for all = AGG of them (clamped to the
    /// table's K and to 32 multi-verify rows). The walk is unchanged (the MTP law)
    #[arg(long, default_value = "off")]
    pub ink_mtp_bgate: String,
    /// INK23: the serve plane's prompt-prefix cache: N nodes a rank (1..16), each one sequence's
    /// whole cross-token state at a prompt cut (every visit's KV pages + conv states, the
    /// drafter's context); a prompt reloads the longest stored proper prefix and stores its cut
    /// before its LAST user turn. 0 = OFF (THE DEFAULT, the control)
    #[arg(long, default_value_t = 0)]
    pub ink_prefix_cache: u64,
    /// INK23: the shortest cut `--ink-prefix-cache` stores (tokens)
    #[arg(long, default_value_t = 64)]
    pub ink_prefix_min: u64,
    /// INK26: the prefix cache's verify arm -- after every node store / reload, every copy's
    /// destination is read back and compared with its source and a reloaded node's regions with
    /// their fingerprints at the store (`INK26 pcverify` lines), and rank 0 prints every request's
    /// schedule at its release (`INK26 trace`: prompt length + hash, reload, the step kinds that
    /// emitted its outputs). Allowed with `--ink-prefix-cache 0` (the trace alone: the control's
    /// schedules)
    #[arg(long)]
    pub ink_prefix_cache_verify: bool,
    /// INK31: rank 0 prints every request's schedule at its release (`INK26 trace`, + the first
    /// three verify rounds' input ids and picked rows' logprob bits) WITHOUT the copy read-backs of
    /// `--ink-prefix-cache-verify` (a diagnostic; the served bytes are unchanged)
    #[arg(long)]
    pub ink_serve_trace: bool,
    /// INK27: after every prefix node copy (store / load) both ranks drain their streams and meet
    /// on the host before either queues the next forward (the INK23 miss probe; an arm)
    #[arg(long)]
    pub ink_prefix_cache_sync: bool,
    /// INK21: the batched speculation's plain-decode rules (`--ink-mtp-batch`), joined by `,`:
    /// `first<N>` = a step holding a sequence before its N-th output decodes plainly (`first2` =
    /// INK20's rule: its first visible text rides that step), `rel` = the step after a sequence left decodes
    /// plainly (a request arriving then waits one ~28 ms step, not a ~50 ms round); `none` = no
    /// rule. INK22: **`rel,first3` IS THE DEFAULT** (INK21's arm C); `first2` = INK20's (the
    /// control)
    #[arg(long, default_value = DEFAULT_INK_BATCH_PLAIN)]
    pub ink_batch_plain: String,
    /// INK22: the batched speculation's plain-step drafter append (`--ink-mtp-batch`): `each` =
    /// INK20's per-slot eager append after the step (the control); `graph` = the step's rows
    /// appended inside its captured decode graph (`ink_dr5`: the same bytes a row);
    /// `check` = `graph` + the eager append over the same rows, every row compared byte for byte;
    /// `,pfmm` (with `--ink-draft-mm wmma`; stands down by name otherwise) = a prefill chunk's
    /// context rows on the matrix cores (moves drafts only). **`graph,pfmm` IS THE DEFAULT (INK22)**
    #[arg(long, default_value = DEFAULT_INK_BATCH_APPEND)]
    pub ink_batch_append: String,
    /// INK22: when a batched round queues the NEXT round (`--ink-mtp-batch`): `on` = at every
    /// round's end (INK20's, THE DEFAULT, the control); `full` = only while every sequence slot is
    /// taken (a free slot means an arrival's prefill would wait behind the queued drafter pass);
    /// `off` = never (each round drafts at its own start)
    #[arg(long, default_value = "on")]
    pub ink_batch_ahead: String,
    /// INK32: copy drafting (`inkling2_ngram.rs`): a round whose sequence's trailing n-gram
    /// occurred earlier in its own tokens (prompt + outputs) verifies the tokens that followed it
    /// (one-hot drafts through the drawn-row walk: committed tokens unchanged), the drafter
    /// otherwise; `off` = THE DEFAULT (the control)
    #[arg(long, default_value = "off")]
    pub ink_ngram: String,
    /// INK32: matched tokens a copy proposal needs (the census: 5; 3 loses on structured output)
    #[arg(long, default_value_t = 5)]
    pub ink_ngram_min: u64,
    /// INK32: the longest match compared (the longest earlier match wins)
    #[arg(long, default_value_t = 8)]
    pub ink_ngram_max: u64,
    /// INK32: copied drafts a one-sequence round may verify (1..7; the depth by match length is
    /// capped by it); a batched round takes its table's K
    #[arg(long, default_value_t = 7)]
    pub ink_ngram_depth: u64,
    /// INK32: copies also replace a batched round's K drafts (`--ink-mtp-batch`, >= 2 live); OFF by
    /// default: BetterBench --quick conc-8 lost 3 % with it (b1 hold, A N A N 280.5 / 275.9 vs
    /// 269.4 / 269.9 t/s) -- at temperature 0.7 the drafter's K 2 drafts beat a copy's
    #[arg(long)]
    pub ink_ngram_batch: bool,
    /// INK21: the batched drafter's M > 8 GEMMs -- `wmma` = the matrix cores (`ink_dr4_mm_*`, A
    /// rounded to bf16: moves acceptance only; **THE DEFAULT since INK22**); `base` = the target's
    /// f32 GEMM (the control)
    #[arg(long, default_value = DEFAULT_INK_DRAFT_MM)]
    pub ink_draft_mm: String,
    /// INK12: `--ink-run mtp` -- drafts a verify (1..7), tokens a prompt, the non-MTP greedy
    /// reference (a `taps` run's dir), the rollback control (`nocommit`)
    #[arg(long, default_value_t = 2)]
    pub ink_mtp_k: u64,
    /// INK12: `--ink-run mtp`'s tokens a prompt
    #[arg(long, default_value_t = 64)]
    pub ink_mtp_tokens: u32,
    /// INK12: `--ink-run mtp`'s non-MTP greedy reference (a `taps` dir or a `--ink-mtp-k 0` json)
    #[arg(long)]
    pub ink_mtp_ref: Option<PathBuf>,
    /// INK12 / INK13: `--ink-run mtp`'s rollback control (`nocommit`; with `--ink-mtp-graph` also
    /// `shift` | `zero`): must change the output
    #[arg(long, default_value = "")]
    pub ink_mtp_control: String,
    /// INK13: `--ink-run mtp` on the captured drafter round + the captured verify chunk a width
    /// (`--ink-mtp-control shift|zero|nocommit` = the rollback controls)
    #[arg(long)]
    pub ink_mtp_graph: bool,
    /// INK13: the serve plane's drafter (`--ink-serve`): K = the bundled block drafter at K drafts
    /// a verify (1..4) whenever ONE sequence is live -- **2 IS THE DEFAULT** (INK13: BetterBench
    /// --quick single 69.2 -> 77.1 / 77.2 t/s, conc-8 tied); `--ink-mtp 0` = off, the control
    /// INK19: unset = `DEFAULT_INK_MTP` when the pack carries the drafter (`<pack>/draft/
    /// manifest.json`), else STANDS DOWN to 0 by name (`INK19 drafter:`); an explicit K refuses a
    /// pack without it
    #[arg(long)]
    pub ink_mtp: Option<u64>,
    /// INK18: the drafter's vocabulary -- N = draft over the N most frequent token ids of
    /// `<pack>/draft/vocab_rank.u32` (`tools/pack/build_ink2_draft_vocab.py`; every special id
    /// first); lossless under the drawn-row walk (a draft only moves acceptance). **32,768 IS THE
    /// DEFAULT (INK18)**; 0 = the full head (the control)
    /// INK19: unset = `DEFAULT_INK_DRAFT_VOCAB` when `<pack>/draft/vocab_rank.u32` exists, else
    /// STANDS DOWN to the full head by name (`INK19 draft vocab:`); an explicit N refuses a pack
    /// without the rank file
    #[arg(long)]
    pub ink_draft_vocab: Option<u64>,
    /// INK18: the drafter's GEMVs -- `sk` = the split-K body with a last-arriver reduce
    /// (`ink_dr_gemv_sk_*`, the drafter's own numerics; `sk1` = one column pass a workgroup) --
    /// **`sk` IS THE DEFAULT (INK18)**; `base` = the target's exact GEMV (the control)
    #[arg(long, default_value = DEFAULT_INK_DRAFT_GEMV)]
    pub ink_draft_gemv: String,
    /// INK18: the drafter's block layers -- `w8` = int8 per-32 RTN at load (INK14's record law)
    /// **IS THE DEFAULT (INK18)**; `bf16` = the checkpoint's bytes (the control)
    #[arg(long, default_value = DEFAULT_INK_DRAFT_LAYERS)]
    pub ink_draft_layers: String,
    /// INK7: the decode cache's per-sequence rows (the INK5 harness's `--max-ctx`)
    #[arg(long, default_value_t = 512)]
    pub ink_cache_ctx: u64,
    /// INK7: the TP=2 / EP=2 split on ONE card (each rank's halves their own launches)
    #[arg(long)]
    pub ink_split: bool,
    /// INK7: capture the decode step as one hipGraph and gate replay == eager
    #[arg(long)]
    pub ink_capture: bool,
    /// INK7: a control arm (`noloop` | `nomsg`)
    #[arg(long, default_value = "")]
    pub ink_control: String,
    /// INK8: the two-card reduce + head gather: `p2p` = the fused on-card body (`ink_p2p_xchg`),
    /// `gemm` = INK7's identity-GEMM reduce + host gather (the control)
    #[arg(long, default_value = "p2p")]
    pub ink_reduce: String,
    /// INK8: the OpenAI-compatible serve plane on this port (the two-card process, `--tp 2`)
    #[arg(long)]
    pub ink_serve: Option<u16>,
    /// INK8: the served model name
    #[arg(long, default_value = "Inkling-2-Mini")]
    pub ink_served_name: String,
    /// INK8: sequences the serve plane holds at once (each its own per-visit cache; a step runs
    /// them one after another)
    #[arg(long, default_value_t = 1)]
    pub ink_serve_seqs: u32,
    /// INK8: the served context (tokens a sequence's per-visit cache holds)
    #[arg(long, default_value_t = 4_096)]
    pub ink_serve_ctx: u64,
    /// INK8: decode steps replay one captured graph a sequence (`--no-ink-serve-graph` = eager)
    #[arg(long = "no-ink-serve-graph", action = clap::ArgAction::SetFalse, default_value_t = true)]
    pub ink_serve_graph: bool,
    /// INK9: the decode bodies: `fast` = the byte-equal re-tiles (`ink_fast`: the M <= 8 GEMV, ...),
    /// `v0` = the INK3b correctness-first bodies (the control)
    #[arg(long, default_value = "fast")]
    pub ink_bodies: String,
    /// INK10: the fused decode bodies over the fast ones (decode / verify rows <= 8 with a carried
    /// cache): `x` = the push reduce fused into the next sconv + depth attend, `route` = route
    /// logits + tail in one launch, `all` = both; `off` = INK9's launch chain (the control)
    /// INK11: the default is `all11` (= `all` + `mdep` (a pass's 48 messages in one launch) +
    /// `qkv` (the q / k / v GEMVs in one launch)); `all` = INK10's set (the control); `kvc` = the
    /// kv conv folded into the attention (an arm)
    /// INK17: the default is `all17` (= `all11` + `vt` (a verify chunk's chained rows on the
    /// fused form, `ink_xsdc`) + `rt` (the route tail per row, `ink_route_fused_rt`)); `all11` =
    /// the control; `g16` = the 8-bit GEMVs on the 16-wide tile at every N (an arm)
    /// INK22: the default is `all17,cl8` (+ the 2..8-row GEMVs on the column-lane GEMM); `all17`
    /// = the control
    #[arg(long, default_value = DEFAULT_INK_FUSE)]
    pub ink_fuse: String,
    /// INK15: the prefill GEMMs (M > 8 rows): `pf` = `ink_pf`'s re-tiles (the parents' order, a
    /// tile sized to the shape: every output byte-equal) + the 16-byte peer reduce and the parent
    /// route above 8 rows; `base` = the 64 x 64 parents (the control); INK16: `cl` =
    /// `pf` + the column-lane GEMM (`ink_pf2`, a lane = one column's rows; byte-equal) at 9..256 rows
    /// -- THE DEFAULT (`pf` = the control); `clp` = `cl` + the prefill reduce as a push (an arm)
    #[arg(long, default_value = DEFAULT_INK_PREFILL_GEMM)]
    pub ink_prefill_gemm: String,
    /// INK25: the 8-bit trunk's prefill GEMMs: `iu8` = the matrix-core W8A8 GEMM
    /// (`ink_gemm_i8_w8`, int8 activations a 32-k block quantized once an input by
    /// `ink_act_q8_b32`) on the attention projections + o at > 8 rows of a prefill chunk (the
    /// prefill rows' numerics change; verify / batch / decode rows never route here) -- THE
    /// DEFAULT since INK26 (the user's flip); `f32` = the byte-law bodies of `--ink-prefill-gemm`
    /// (the control)
    #[arg(long, default_value = DEFAULT_INK_PREFILL_TRUNK)]
    pub ink_prefill_trunk: String,
    /// INK27: a prefill chunk's attention body: `v1` = `ink_attn_bias_sink_v1`, one workgroup a
    /// (row, head) (the default, the control); `rt8` / `rt16` = `ink_fa_rt{8,16}`, R consecutive
    /// rows of the chunk a workgroup sharing every key row's loads, v1's order op for op (byte-equal;
    /// an arm). Verify / batch / decode rows never route here.
    #[arg(long, default_value = DEFAULT_INK_PREFILL_ATTN)]
    pub ink_prefill_attn: String,
    /// INK31: route `--ink-prefill-attn rt8|rt16` only at a chunk whose last position is >= N
    /// (0 = every eligible chunk, INK27's arm)
    #[arg(long, default_value_t = DEFAULT_INK_PREFILL_ATTN_FLOOR)]
    pub ink_prefill_attn_floor: u64,
    /// INK29: the prefill levers (`DEFAULT_INK29`): kx, rs, fq (byte-equal); ls, b16, bw, i8in,
    /// i8o (numerics arms); comma-separated, `none` = empty
    #[arg(long = "ink-p29", default_value = DEFAULT_INK29)]
    pub ink29: String,
    /// INK35: the prefill levers (`DEFAULT_INK35`, empty = off): mzs (numerics arm)
    #[arg(long = "ink-p35", default_value = DEFAULT_INK35)]
    pub ink35: String,
    /// INK36: the prefill schedule levers (`DEFAULT_INK36`, empty = off): ov, hg, hgp, xw<N>
    #[arg(long = "ink-p36", default_value = DEFAULT_INK36)]
    pub ink36: String,
    /// INK37: the byte-equal prefill levers (`DEFAULT_INK37`, empty = off): moe, moe8, glue, glue8
    #[arg(long = "ink-p37", default_value = DEFAULT_INK37)]
    pub ink37: String,
    /// CONS40: the byte-equal prefill levers (`DEFAULT_INK40`, empty = off): fa128, fa128l, fam<N>, zd, zd1, zd2, zp
    #[arg(long = "ink-p40", default_value = DEFAULT_INK40)]
    pub ink40: String,
    /// INK39: the multi-sequence verify levers (comma list, `cat` = THE DEFAULT, `none` = the control):
    /// `cat` = a > 8-row multi verify's attention projections as the merged group GEMMs (byte-equal:
    /// every output its own sequential sum), `push` = its reduces as `ink_p2p_push_v5` (byte-equal
    /// to v4), `all` = both
    #[arg(long = "ink-p39", default_value = DEFAULT_INK_P39)]
    pub ink_p39: String,
    /// INK11: the serve plane's decode sampler: `seam` = the device seam (`ink_seam_p1/p2`: the
    /// top 64 + log-sum-exps a row, `Sampler::sample`; a refused row falls back to the dense row),
    /// `dense` = the whole row read back (`Sampler::sample_dense`, the control), `check` = both,
    /// every seeded / greedy row's tokens compared (counted)
    #[arg(long, default_value = "seam")]
    pub ink_sampler: String,
    /// INK16: a prompt's FIRST token: `seam` = the last chunk's row through the device seam (the
    /// decode sampler's law; a refused row falls back to the dense row), `dense` = the whole row
    /// read back and sampled on the host (~5.6-7 ms at BetterBench's sampling; the control). `seam`
    /// IS THE DEFAULT (INK16)
    #[arg(long, default_value = DEFAULT_INK_FIRST_TOKEN)]
    pub ink_first_token: String,
    /// INK9: time every launch of one eager decode step a prompt (event pairs; per-rank JSON
    /// `optime_rank<r>.json` in `--ink-out`) -- a diagnostic, the step it times is eager; on a
    /// `--ink-run tf` run (INK15) every prompt's prefill pass (`poptime_<id>_rank<r>.json`)
    #[arg(long)]
    pub ink_op_timing: bool,
    /// INK9: the local layers' KV ring (rows; >= 1,024 + the widest chunk - 1; 0 = full
    /// per-visit pools, the control)
    #[arg(long, default_value_t = 2_048)]
    pub ink_local_ring: u64,
    /// INK9: `--ink-run long`'s prefill chunk (rows)
    #[arg(long, default_value_t = 1_024)]
    pub ink_chunk: u64,
    /// INK14: the dense trunk's weights. **Default (unset) = the 8-bit trunk `w8-head` when the
    /// pack carries the overlay at `<pack>/trunk8` (INK14: single +3.6 %, conc-8 +14 %, the FN80d
    /// envelope PASS), else the bf16 trunk -- the boot says which (`INK14 trunk8:`)**; `bf16` = the
    /// checkpoint's weights (the control), `w8` = the attention projections + o_proj from the
    /// overlay (int8 RTN per-32 records, `ink_w8` bodies, the dequant fused into the GEMV),
    /// `w8-head` = also the `lm_head`; an explicit `w8*` refuses a missing overlay
    #[arg(long)]
    pub ink_trunk: Option<String>,
    /// INK14: the 8-bit trunk overlay (`ink14-trunk8/1`; default `<pack>/trunk8`)
    #[arg(long)]
    pub ink_trunk8_dir: Option<PathBuf>,
    /// INK7: the cross-visit arena (MiB)
    #[arg(long, default_value_t = 1_024)]
    pub ink_pass_mib: u64,
    /// INK7: one visit's arena (MiB)
    #[arg(long, default_value_t = 1_024)]
    pub ink_visit_mib: u64,
}

impl InkFlags {
    /// The drafter's depth (`--ink-mtp`; unset = the default -- see [`InkFlags::resolved`]).
    pub fn mtp(&self) -> u64 {
        self.ink_mtp.unwrap_or(if self.default_gate_on() {
            DEFAULT_INK_MTP
        } else {
            DEFAULT_INK_MTP_UNGATED
        })
    }

    /// INK34: the default gate applies (unset depth, `--ink-mtp-gate auto`, pruned vocabulary).
    fn default_gate_on(&self) -> bool {
        self.ink_mtp.is_none() && self.ink_mtp_gate == "auto" && self.draft_vocab() > 0
    }

    /// INK34: the serve plane's `--ink-mtp-gate` after the `auto` rule (empty = off).
    pub fn mtp_gate_spec(&self) -> &str {
        match self.ink_mtp_gate.as_str() {
            "auto" if self.default_gate_on() => DEFAULT_INK_MTP_GATE,
            "auto" => "",
            s => s,
        }
    }

    /// The drafter's vocabulary (`--ink-draft-vocab`; unset = the default).
    pub fn draft_vocab(&self) -> u64 {
        self.ink_draft_vocab.unwrap_or(DEFAULT_INK_DRAFT_VOCAB)
    }

    /// INK19: the pack-data defaults resolved against the pack, each stand-down logged by name
    /// (INK14's trunk8 law): an UNSET `--ink-mtp` without `<pack>/draft/manifest.json` = 0 (no
    /// drafter), an UNSET `--ink-draft-vocab` without a `<pack>/draft/vocab_rank.u32` of
    /// `VOCAB` u32 = 0 (the full head); an explicit value is kept (its load refuses by name).
    pub fn resolved(&self, pack_dir: &Path) -> (Self, Vec<String>) {
        let mut f = self.clone();
        let mut said = Vec::new();
        let draft = pack_dir.join("draft");
        let has_draft = draft.join("manifest.json").is_file();
        if f.ink_mtp.is_none() && !has_draft {
            said.push(format!(
                "INK19 drafter: STANDS DOWN to --ink-mtp 0 (the default {DEFAULT_INK_MTP} needs \
                 {}: absent)",
                draft.join("manifest.json").display()
            ));
            f.ink_mtp = Some(0);
        }
        let rf = draft.join("vocab_rank.u32");
        let rank_ok = std::fs::metadata(&rf)
            .is_ok_and(|m| m.is_file() && m.len() == crate::inkling2_exec::VOCAB as u64 * 4);
        if f.ink_draft_vocab.is_none() && !rank_ok {
            if f.mtp() > 0 {
                said.push(format!(
                    "INK19 draft vocab: STANDS DOWN to the full head (the default \
                     --ink-draft-vocab {DEFAULT_INK_DRAFT_VOCAB} needs {} of {} ids: {})",
                    rf.display(),
                    crate::inkling2_exec::VOCAB,
                    if rf.exists() { "wrong size" } else { "absent" }
                ));
            }
            f.ink_draft_vocab = Some(0);
        }
        (f, said)
    }
}

impl Default for InkFlags {
    fn default() -> Self {
        Self {
            ink_dry_run: false,
            ink_one_process: false,
            ink_max_seqs: 8,
            ink_max_ctx: 32_768,
            ink_max_chunk: 2_048,
            ink_page_rows: 64,
            ink_decode_splits: 16,
            ink_budget_mib: DEFAULT_INK_BUDGET_MIB,
            ink_rank: 0,
            ink_peer_port: 0,
            ink_run: None,
            ink_ref: None,
            ink_out: None,
            ink_only: String::new(),
            ink_steps: 24,
            ink_vprice_widths: "1,2,3,4,5,6,8".into(),
            ink_vprice_batch: String::new(),
            ink_mseq: "2x3,4x3,8x3".into(),
            ink_mseq_rounds: 12,
            ink_mtp_batch: DEFAULT_INK_MTP_BATCH.into(),
            ink_mtp_gate: "auto".into(),
            ink_mtp_bgate: "off".into(),
            ink_prefix_cache: 0,
            ink_prefix_min: 64,
            ink_prefix_cache_verify: false,
            ink_serve_trace: false,
            ink_prefix_cache_sync: false,
            ink_batch_plain: DEFAULT_INK_BATCH_PLAIN.into(),
            ink_batch_append: DEFAULT_INK_BATCH_APPEND.into(),
            ink_batch_ahead: "on".into(),
            ink_ngram: "off".into(),
            ink_ngram_min: 5,
            ink_ngram_max: 8,
            ink_ngram_depth: 7,
            ink_ngram_batch: false,
            ink_draft_mm: DEFAULT_INK_DRAFT_MM.into(),
            ink_mtp_k: 2,
            ink_mtp_tokens: 64,
            ink_mtp_ref: None,
            ink_mtp_control: String::new(),
            ink_mtp_graph: false,
            ink_mtp: None,
            ink_draft_vocab: None,
            ink_draft_gemv: DEFAULT_INK_DRAFT_GEMV.into(),
            ink_draft_layers: DEFAULT_INK_DRAFT_LAYERS.into(),
            ink_cache_ctx: 512,
            ink_split: false,
            ink_capture: false,
            ink_control: String::new(),
            ink_pass_mib: 1_024,
            ink_visit_mib: 1_024,
            ink_reduce: "p2p".into(),
            ink_serve: None,
            ink_served_name: "Inkling-2-Mini".into(),
            ink_serve_seqs: 1,
            ink_serve_ctx: 4_096,
            ink_serve_graph: true,
            ink_bodies: "fast".into(),
            ink_fuse: DEFAULT_INK_FUSE.into(),
            ink_prefill_gemm: DEFAULT_INK_PREFILL_GEMM.into(),
            ink_prefill_trunk: DEFAULT_INK_PREFILL_TRUNK.into(),
            ink_prefill_attn: DEFAULT_INK_PREFILL_ATTN.into(),
            ink_prefill_attn_floor: DEFAULT_INK_PREFILL_ATTN_FLOOR,
            ink29: DEFAULT_INK29.into(),
            ink35: DEFAULT_INK35.into(),
            ink36: DEFAULT_INK36.into(),
            ink37: DEFAULT_INK37.into(),
            ink40: DEFAULT_INK40.into(),
            ink_p39: DEFAULT_INK_P39.into(),
            ink_sampler: "seam".into(),
            ink_first_token: DEFAULT_INK_FIRST_TOKEN.into(),
            ink_op_timing: false,
            ink_local_ring: 2_048,
            ink_chunk: 1_024,
            ink_trunk: None,
            ink_trunk8_dir: None,
        }
    }
}

impl InkFlags {
    /// A non-Inkling pack refuses every `--ink-*` flag off its default.
    pub fn refuse_elsewhere(&self) -> anyhow::Result<()> {
        ensure!(
            *self == Self::default(),
            "--ink-* flags are the Inkling 2 path's (a pack whose manifest names inkling2_loop); \
             this pack is not one"
        );
        Ok(())
    }

    fn geom(&self, tp: u32) -> EngineGeom {
        EngineGeom {
            tp,
            max_seqs: self.ink_max_seqs,
            max_ctx: self.ink_max_ctx,
            max_chunk: self.ink_max_chunk,
            page_rows: self.ink_page_rows,
            decode_splits: self.ink_decode_splits,
        }
    }
}

/// What the boot read from the pack (rank-independent).
#[derive(Debug)]
pub struct Pack {
    pub dir: PathBuf,
    /// INK41: where the pack's BYTES live -- `dir` for a TP=2 pack; for the one-card view
    /// (`layout: "tp1-view"`) the TP=2 pack its manifest's `blobs` names (relative to `dir`)
    pub blobs: PathBuf,
    /// INK41: the one-card (TP=1) view of a TP=2 pack (`tools/pack/build_ink2_tp1.py`)
    pub view: bool,
    pub manifest: Value,
    pub ckpt: PathBuf,
    pub cfg: Inkling2Config,
    pub plan_sha256: String,
}

/// HF3: the directory a downloaded pack ships its checkpoint subset in (the dense records'
/// tensors byte for byte + the small model files; `tools/pack/build_ink2_base.py`).
pub const PACK_BASE_DIR: &str = "base";
/// The file that marks a checkpoint directory (the full checkpoint's or `base/`'s).
pub const CKPT_MARKER: &str = "model.safetensors.index.json";

/// HF3: where the pack's dense records live on THIS machine. `<pack>/base` when the pack ships
/// its own checkpoint subset (a downloaded pack: its `dense.json` records name that subset's
/// file); else the manifest's recorded absolute path -- the control, the maintainer's pack is
/// unchanged. The plan sha256 is always derived from the RECORDED path string, so a pack's
/// manifest is never rewritten to move it.
pub fn resolve_ckpt(pack: &Path, recorded: &Path) -> anyhow::Result<(PathBuf, &'static str)> {
    let base = pack.join(PACK_BASE_DIR);
    if base.join(CKPT_MARKER).is_file() && base.join("config.json").is_file() {
        return Ok((base, "the pack's own base/"));
    }
    if recorded.join("config.json").is_file() {
        return Ok((recorded.to_path_buf(), "the manifest's path"));
    }
    bail!(
        "{}: the pack's checkpoint {} is not on this machine and the pack ships no {}/ subset \
         (download the published pack: tools/fetch_packs.sh inkling)",
        pack.display(),
        recorded.display(),
        PACK_BASE_DIR
    )
}

/// Read the pack's manifest + its checkpoint's config; re-derive the plan and check its sha256.
pub fn open_pack(dir: &Path, tp: u32) -> anyhow::Result<Pack> {
    let mp = dir.join("manifest.json");
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(&mp).with_context(|| mp.display().to_string())?,
    )
    .with_context(|| format!("{}: not JSON", mp.display()))?;
    let s = |k: &str| manifest.get(k).and_then(Value::as_str).unwrap_or("");
    ensure!(
        s("model") == "inkling2_loop",
        "{}: model {:?}",
        mp.display(),
        s("model")
    );
    // INK6: the native pack (NVFP4 v2 + MXFP8 records, one tier file a format a rank)
    let native = s("format") == NATIVE_PACK_FORMAT;
    ensure!(
        s("format") == PACK_FORMAT || native,
        "{}: format {:?}, this path reads {PACK_FORMAT} or {NATIVE_PACK_FORMAT}",
        mp.display(),
        s("format")
    );
    let ptp = manifest.get("tp").and_then(Value::as_u64).unwrap_or(0);
    let pep = manifest.get("ep").and_then(Value::as_u64).unwrap_or(0);
    // INK41: the one-card layout is a VIEW over a TP=2 pack's bytes (the same files)
    let view = s("layout") == TP1_VIEW_LAYOUT;
    let blobs = if view {
        ensure!(
            native && (ptp, pep) == (1, 1),
            "{}: a {TP1_VIEW_LAYOUT} manifest of TP={ptp} / EP={pep} ({:?})",
            mp.display(),
            s("format")
        );
        ensure!(
            tp == 1,
            "--tp {tp}: {} is the ONE-CARD (TP=1) view of a TP=2 pack; a two-card boot reads \
             the TP=2 pack itself ({})",
            dir.display(),
            dir.join(s("blobs")).display()
        );
        let b = dir.join(s("blobs"));
        let braw = std::fs::read(b.join("manifest.json"))
            .with_context(|| format!("{}: the view's blobs", b.join("manifest.json").display()))?;
        let got = hex::encode(Sha256::digest(&braw));
        ensure!(
            got == s("blobs_manifest_sha256"),
            "{}: the view pins the TP=2 manifest {} but {} is {got} (a view of another pack: \
             rebuild it with tools/pack/build_ink2_tp1.py)",
            mp.display(),
            s("blobs_manifest_sha256"),
            b.display()
        );
        let bm: Value = serde_json::from_slice(&braw)?;
        ensure!(
            (bm["tp"].as_u64(), bm["ep"].as_u64()) == (Some(2), Some(2))
                && bm["plan_sha256"] == manifest["plan_sha256"]
                && bm["checkpoint"] == manifest["checkpoint"],
            "{}: the view's blobs are not its TP=2 pack (tp / plan / checkpoint)",
            mp.display()
        );
        b
    } else {
        ensure!(
            (ptp, pep) == (2, 2),
            "{}: a TP={ptp} / EP={pep} pack; the Inkling 2 path reads TP=2 / EP=2 packs (or their \
             {TP1_VIEW_LAYOUT})",
            mp.display()
        );
        ensure!(
            tp == 2,
            "--tp {tp}: {} is the TWO-CARD (TP=2 / EP=2) pack; a one-card boot reads its TP=1 view \
             ({}/{TP1_VIEW_DIR} in the published pack; tools/pack/build_ink2_tp1.py)",
            dir.display(),
            dir.display()
        );
        let ranks: Vec<u64> = manifest["ranks"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_u64).collect())
            .unwrap_or_default();
        ensure!(
            ranks == [0, 1],
            "{}: ranks {ranks:?}: a fixture pack (both ranks needed)",
            mp.display()
        );
        dir.to_path_buf()
    };
    let (ckpt, ckpt_from) = resolve_ckpt(&blobs, Path::new(s("checkpoint")))?;
    println!("inkpaths: checkpoint {} ({ckpt_from})", ckpt.display());
    let cj = ckpt.join("config.json");
    let cfg = Inkling2Config::from_config_json(
        &std::fs::read_to_string(&cj).with_context(|| cj.display().to_string())?,
    )
    .map_err(|e| anyhow::anyhow!("{}: {e}", cj.display()))?;
    let layers = manifest["layers"].as_array().map_or(0, Vec::len);
    ensure!(
        layers == cfg.num_hidden_layers as usize,
        "{}: {layers} layers of {}: a fixture pack",
        mp.display(),
        cfg.num_hidden_layers
    );
    // the pack's plan: re-derive it (the MXFP4 form, or INK6's Native form) and match the sha
    let form = if native {
        ExpertForm::Native
    } else {
        ExpertForm::Mxfp4
    };
    let text = PackPlan::new(&cfg, form).plan_text(&cfg, s("checkpoint"));
    let got = hex::encode(Sha256::digest(text.as_bytes()));
    ensure!(
        got == s("plan_sha256"),
        "{}: plan_sha256 {} but this binary's plan of {} is {got} (a pack of another plan or \
         form; rebuild it or this binary)",
        mp.display(),
        s("plan_sha256"),
        ckpt.display()
    );
    if view {
        println!(
            "inkpaths: the one-card (TP=1) view {} over the TP=2 pack {}",
            dir.display(),
            blobs.display()
        );
    }
    Ok(Pack {
        dir: dir.to_path_buf(),
        blobs,
        view,
        manifest,
        ckpt,
        cfg,
        plan_sha256: got,
    })
}

/// A record of `dense.json`.
fn dense_bytes(p: &Pack, file: &Path) -> anyhow::Result<Vec<(String, u64)>> {
    let v: Value = serde_json::from_str(
        &std::fs::read_to_string(file).with_context(|| file.display().to_string())?,
    )?;
    let recs = v
        .as_array()
        .with_context(|| format!("{}: not a record list", file.display()))?;
    let mut sizes = std::collections::BTreeMap::new();
    let mut out = Vec::with_capacity(recs.len());
    for r in recs {
        let name = r["tensor"].as_str().unwrap_or("").to_string();
        let f = r["file"].as_str().unwrap_or("");
        let (off, stride, count, len) = (
            r["offset"].as_u64().unwrap_or(0),
            r["stride"].as_u64().unwrap_or(0),
            r["count"].as_u64().unwrap_or(0),
            r["length"].as_u64().unwrap_or(0),
        );
        ensure!(
            !name.is_empty() && !f.is_empty() && len > 0 && count > 0 && len <= stride,
            "{}: a malformed record {r}",
            file.display()
        );
        let size = match sizes.get(f) {
            Some(s) => *s,
            None => {
                let s = std::fs::metadata(p.ckpt.join(f))
                    .with_context(|| format!("{}: record {name}'s shard {f}", file.display()))?
                    .len();
                sizes.insert(f.to_string(), s);
                s
            }
        };
        let end = off + stride * (count - 1) + len;
        ensure!(
            end <= size,
            "{}: record {name} ends at {end} past {f}'s {size} B",
            file.display()
        );
        out.push((name, len * count));
    }
    Ok(out)
}

/// One rank's bill.
#[derive(Debug, Clone, PartialEq)]
pub struct RankBill {
    pub rank: u32,
    /// (class, bytes, what), in reservation order.
    pub classes: Vec<(String, u64, String)>,
    pub capacity: u64,
    pub used: u64,
    pub p2pmem_mib: u64,
    /// The geometry digest both ranks must agree on.
    pub geom_digest: u64,
    pub launches: (usize, usize),
}

impl RankBill {
    fn to_json(&self) -> Value {
        json!({"rank": self.rank, "capacity": self.capacity, "used": self.used,
               "p2pmem_mib": self.p2pmem_mib, "geom_digest": self.geom_digest,
               "launches": [self.launches.0, self.launches.1],
               "classes": self.classes.iter().map(|(n, b, w)| json!([n, b, w])).collect::<Vec<_>>()})
    }

    fn from_json(v: &Value) -> anyhow::Result<Self> {
        let u = |k: &str| v[k].as_u64().with_context(|| format!("rank bill: {k}"));
        Ok(Self {
            rank: u32::try_from(u("rank")?)?,
            capacity: u("capacity")?,
            used: u("used")?,
            p2pmem_mib: u("p2pmem_mib")?,
            geom_digest: u("geom_digest")?,
            launches: (
                v["launches"][0].as_u64().unwrap_or(0) as usize,
                v["launches"][1].as_u64().unwrap_or(0) as usize,
            ),
            classes: v["classes"]
                .as_array()
                .context("rank bill: classes")?
                .iter()
                .map(|c| {
                    (
                        c[0].as_str().unwrap_or("").to_string(),
                        c[1].as_u64().unwrap_or(0),
                        c[2].as_str().unwrap_or("").to_string(),
                    )
                })
                .collect(),
        })
    }

    /// The boot lines.
    #[must_use]
    pub fn lines(&self) -> Vec<String> {
        let r = self.rank;
        let mut l: Vec<String> = self
            .classes
            .iter()
            .map(|(n, b, w)| format!("inkbudget rank{r}: {n:<20} {:>10.1} MiB  {w}", mib(*b)))
            .collect();
        let free = self.capacity.saturating_sub(self.used);
        l.push(format!(
            "inkbudget rank{r}: TOTAL {:.1} MiB of {:.1} MiB (budget less kernel.p2pmem {} MiB) -> free {:.1} MiB",
            mib(self.used),
            mib(self.capacity),
            self.p2pmem_mib,
            mib(free)
        ));
        l
    }
}

/// Bill one rank: the pack's weight classes (read from its records) + the engine plan's runtime
/// classes, reserved by name in a ledger whose capacity is the budget less the P2PDMA pool. A
/// class that does not fit is refused with its name and both numbers.
pub fn bill_rank(
    p: &Pack,
    rank: u32,
    f: &InkFlags,
    tp: u32,
    p2pmem_mib: u64,
) -> anyhow::Result<RankBill> {
    ensure!(rank < 2, "--ink-rank {rank}: 0 or 1");
    let geom = f.geom(tp);
    ensure!(
        geom.max_seqs >= 1 && geom.max_ctx >= 1 && geom.max_chunk >= 1 && geom.page_rows >= 1,
        "--ink-max-seqs / --ink-max-ctx / --ink-max-chunk / --ink-page-rows must be >= 1"
    );
    ensure!(
        geom.max_ctx <= p.cfg.max_context,
        "--ink-max-ctx {} past the model's advertised {}",
        geom.max_ctx,
        p.cfg.max_context
    );
    let mut classes: Vec<(String, u64, String)> = Vec::new();
    // experts: this rank's tier, every record VRAM-resident (no expert tier on the step path)
    let per = p.manifest["experts_per_rank"].as_u64().unwrap_or(0);
    let t = &p.manifest["tiers"][rank.to_string()];
    // an INK2 pack: one tier ({file, bytes, records}, the manifest's `record`); an INK6 native pack:
    // one tier a format ({kind: {file, bytes, records, layers}}, `records[kind]`)
    let tiers: Vec<(&Value, u64)> = if p.manifest["format"] == NATIVE_PACK_FORMAT {
        t.as_object()
            .map(|o| {
                o.iter()
                    .map(|(k, v)| (v, p.manifest["records"][k]["stride"].as_u64().unwrap_or(0)))
                    .collect()
            })
            .unwrap_or_default()
    } else {
        vec![(t, p.manifest["record"]["stride"].as_u64().unwrap_or(0))]
    };
    let mut ex_bytes = 0u64;
    let mut ex_records = 0u64;
    let mut strides = Vec::new();
    for (t, stride) in tiers {
        let tier = p.dir.join(t["file"].as_str().unwrap_or(""));
        let tier_len = std::fs::metadata(&tier)
            .with_context(|| format!("rank {rank}'s expert tier {}", tier.display()))?
            .len();
        ensure!(
            Some(tier_len) == t["bytes"].as_u64(),
            "{}: {tier_len} B, the manifest says {}",
            tier.display(),
            t["bytes"]
        );
        let records = t["records"].as_u64().unwrap_or(0);
        ensure!(
            records * stride <= tier_len && records % per.max(1) == 0,
            "{}: {records} records x {stride} B do not fit the tier ({tier_len} B)",
            tier.display()
        );
        ex_bytes += records * stride;
        ex_records += records;
        strides.push(stride.to_string());
    }
    ensure!(
        ex_records == u64::from(p.cfg.num_hidden_layers) * per,
        "rank {rank}: {ex_records} expert records, the model has {} layers x {per}",
        p.cfg.num_hidden_layers
    );
    classes.push((
        "ink.w.experts".into(),
        ex_bytes,
        format!(
            "{ex_records} records ({} layers x {per}) x {} B, VRAM-resident",
            p.cfg.num_hidden_layers,
            strides.join(" / ")
        ),
    ));
    // dense: this rank's split records, then the shared ones (embedding + vision named apart)
    let mine = dense_bytes(p, &p.dir.join(format!("rank{rank}/dense.json")))?;
    let shared = dense_bytes(p, &p.dir.join("shared/dense.json"))?;
    let want = |k: &str| p.manifest["dense"][k].as_u64().unwrap_or(u64::MAX);
    ensure!(
        mine.len() as u64 == want(&format!("rank{rank}")) && shared.len() as u64 == want("shared"),
        "dense record counts {} / {} differ from the manifest's {}",
        mine.len(),
        shared.len(),
        p.manifest["dense"]
    );
    let sum = |v: &[(String, u64)], pred: &dyn Fn(&str) -> bool| -> (u64, usize) {
        let s: Vec<_> = v.iter().filter(|(n, _)| pred(n)).collect();
        (s.iter().map(|x| x.1).sum(), s.len())
    };
    let (d_rank, n_rank) = sum(&mine, &|_| true);
    classes.push((
        "ink.w.dense.split".into(),
        d_rank,
        format!("{n_rank} rank records (q/gate/r/sink/k/v/o halves, lm_head half)"),
    ));
    let is_vis =
        |n: &str| n.starts_with("vision") || n.contains(".vision") || n.contains("adapter");
    let is_emb = |n: &str| n.ends_with("embed_tokens.weight");
    let (emb, _) = sum(&shared, &|n| is_emb(n));
    let (vis, n_vis) = sum(&shared, &|n| is_vis(n));
    let (rest, n_rest) = sum(&shared, &|n| !is_emb(n) && !is_vis(n));
    classes.push((
        "ink.w.embed".into(),
        emb,
        "embed_tokens, whole on each card".into(),
    ));
    classes.push((
        "ink.w.dense.shared".into(),
        rest,
        format!("{n_rest} replicated records (norms, queries, convs, router, gates)"),
    ));
    classes.push((
        "ink.w.vision".into(),
        vis,
        if n_vis == 0 {
            "0 records: this pack carries no tower (INK7 adds it, ~0.54 GB)".to_string()
        } else {
            format!("{n_vis} vision tower + adapter records")
        },
    ));
    // the runtime classes
    let plan = RankPlan::new(&p.cfg, geom);
    for c in &plan.classes {
        classes.push((c.name.to_string(), c.bytes, c.what.clone()));
    }
    // the fence
    let capacity = f.ink_budget_mib.saturating_sub(p2pmem_mib) << 20;
    let mut ledger = Ledger::new();
    let pool = Pool::Vram(rank);
    ledger.declare(pool, capacity)?;
    for (n, b, _) in &classes {
        ledger.reserve(pool, n, *b).map_err(|e| {
            anyhow::anyhow!(
                "inkbudget rank{rank}: {e} -- class {n} ({:.1} MiB) does not fit after {:.1} MiB \
                 of {:.1} MiB (lower --ink-max-ctx / --ink-max-seqs / --ink-max-chunk)",
                mib(*b),
                mib(ledger.used(pool)),
                mib(capacity)
            )
        })?;
    }
    // the digest the pair must agree on: the plan + every runtime class + the arena tables
    let mut h = Sha256::new();
    h.update(p.plan_sha256.as_bytes());
    h.update(format!("{geom:?}").as_bytes());
    for c in &plan.classes {
        h.update(format!("{}={};", c.name, c.bytes).as_bytes());
    }
    for a in [&plan.prefill, &plan.decode] {
        h.update(format!("{:?}", a.bufs).as_bytes());
    }
    let d = h.finalize();
    let geom_digest = u64::from_le_bytes(d[..8].try_into().expect("8 bytes"));
    Ok(RankBill {
        rank,
        used: ledger.used(pool),
        classes,
        capacity,
        p2pmem_mib,
        geom_digest,
        launches: layer_graph::step_launches(&p.cfg),
    })
}

/// The arena lines of a rank plan (both forms, buffer by buffer).
fn arena_lines(p: &Pack, f: &InkFlags, tp: u32) -> Vec<String> {
    let plan = RankPlan::new(&p.cfg, f.geom(tp));
    let mut l = Vec::new();
    for a in [&plan.prefill, &plan.decode] {
        l.push(format!(
            "inkarena: {:?} {} rows: {:.1} MiB (widest visit {} = layer {}){}",
            a.pass,
            a.rows,
            mib(a.bytes),
            a.widest_visit,
            p.cfg.schedule()[a.widest_visit as usize],
            a.bufs
                .iter()
                .map(|(n, b)| format!(" {n} {:.2}", mib(*b)))
                .collect::<String>()
        ));
    }
    l.push(format!(
        "inkstate: per sequence a rank: {} global visits x {} pages of {} rows, {} local rings x {} rows, {} conv states; {} slots (+1 spare, the pad law)",
        plan.global_visits,
        plan.pages_per_seq,
        plan.geom.page_rows,
        plan.local_visits,
        plan.dims.window,
        plan.conv_states,
        plan.geom.slots()
    ));
    l
}

/// The rank's part of the pair: bill, swap the digest, send / take the bills.
fn run_rank(
    p: &Pack,
    rank: u32,
    f: &InkFlags,
    tp: u32,
    p2pmem: u64,
    ex: Arc<dyn IpcExchange>,
) -> anyhow::Result<Option<(RankBill, RankBill)>> {
    let peer = RankPeer::new(ex);
    let bill = bill_rank(p, rank, f, tp, p2pmem)?;
    let theirs = peer
        .exchange()
        .swap_digest("ink.geom", bill.geom_digest)
        .map_err(|e| anyhow::anyhow!("rank {rank}: {e}"))?;
    ensure!(
        theirs == bill.geom_digest,
        "rank {rank}: geometry digest {:016x} vs the peer's {theirs:016x}: the ranks planned \
         different engines",
        bill.geom_digest
    );
    let got = peer
        .exchange()
        .xchg_bytes("ink.bill", bill.to_json().to_string().as_bytes())
        .map_err(|e| anyhow::anyhow!("rank {rank}: {e}"))?;
    let other = RankBill::from_json(&serde_json::from_slice(&got)?)?;
    ensure!(
        other.rank == 1 - rank,
        "rank {rank}: the peer says rank {}",
        other.rank
    );
    Ok((rank == 0).then_some((bill, other)))
}

/// Kills an un-reaped rank 1 on every exit path.
struct Child(Option<std::process::Child>);

impl Drop for Child {
    fn drop(&mut self) {
        if let Some(c) = self.0.as_mut() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

/// The path's entry: `saber engine --pack <inkling2 pack> --tp 2 --ink-dry-run`, or INK7's
/// one-card executor `--tp 1 --ink-run tf|decode` (the TP=2 pack's halves assembled on one card).
pub fn main(
    pack: &Path,
    tp: u32,
    prebuilt: &Path,
    hip_lib: &str,
    device: i32,
    f: &InkFlags,
) -> anyhow::Result<()> {
    let t0 = Instant::now();
    if f.ink_run.is_some() || f.ink_serve.is_some() {
        ensure!(
            !f.ink_dry_run,
            "--ink-run and --ink-dry-run exclude each other"
        );
        // the one-card harness reads the TP=2 pack's halves (INK7's form) or its TP=1 view; the
        // one-card serve plane reads the view only
        let ptp = if tp == 1 && (is_tp1_view(pack) || f.ink_serve.is_some()) {
            1
        } else {
            2
        };
        let p = open_pack(pack, ptp)?;
        let (f, said) = f.resolved(&p.blobs);
        for l in said {
            println!("{l}");
        }
        return crate::inkling2_exec::run(&p, tp, prebuilt, hip_lib, device, &f);
    }
    let p = open_pack(pack, tp)?;
    let rank = f.ink_rank;
    let (all, new) = layer_graph::step_launches(&p.cfg);
    if rank == 0 {
        println!(
            "inkmodel: {} manifest model inkling2_loop -> the Inkling 2 path ({} layers, {} visits, \
             hidden {}, {} experts top-{}, checkpoint {}); plan sha256 {} re-derived; the 27B and \
             Flash-Next paths are not entered",
            p.dir.display(),
            p.cfg.num_hidden_layers,
            p.cfg.schedule().len(),
            p.cfg.hidden_size,
            p.cfg.n_routed_experts,
            p.cfg.num_experts_per_tok,
            p.ckpt.display(),
            &p.plan_sha256[..16]
        );
        println!(
            "inklaunch: {all} launches a TP=2 step ({new} of them new-body launches; bodies owed: {})",
            layer_graph::new_bodies(&p.cfg).join(" ")
        );
    }
    if !f.ink_dry_run {
        bail!(
            "{}: an inkling2_loop pack boots with --ink-dry-run (INK2) or the one-card executor \
             --tp 1 --ink-run tf|decode (INK7); the serve plane is not built (docs/INKLING2.md \
             section 9)",
            pack.display()
        );
    }
    let p2pmem = crate::ledger_law::box_p2pmem_mib();
    let pair = if f.ink_one_process {
        ensure!(rank == 0, "--ink-one-process is rank 0's form");
        let (a, b) = ThreadPairExchange::pair();
        let p1 = &p;
        std::thread::scope(|s| -> anyhow::Result<_> {
            let h = s.spawn(move || run_rank(p1, 1, f, tp, p2pmem, Arc::new(b)));
            let r0 = run_rank(&p, 0, f, tp, p2pmem, Arc::new(a));
            let r1 = h.join().map_err(|_| anyhow::anyhow!("rank 1 panicked"))?;
            r1?;
            r0
        })?
    } else if rank == 0 {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let mut cmd = std::process::Command::new(std::env::current_exe()?);
        cmd.args(std::env::args().skip(1))
            .arg("--ink-rank")
            .arg("1")
            .arg("--ink-peer-port")
            .arg(port.to_string());
        let mut child = Child(Some(cmd.spawn().context("spawning rank 1")?));
        let ex = TcpExchange::accept(&listener, Duration::from_secs(120))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let r = run_rank(&p, 0, f, tp, p2pmem, Arc::new(ex))?;
        let st = child.0.take().expect("child").wait()?;
        ensure!(st.success(), "rank 1 exited {st}");
        r
    } else {
        let ex = TcpExchange::connect(f.ink_peer_port, Duration::from_secs(120))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        run_rank(&p, 1, f, tp, p2pmem, Arc::new(ex))?;
        return Ok(());
    };
    let (b0, b1) = pair.context("rank 0 holds both bills")?;
    for l in arena_lines(&p, f, tp) {
        println!("{l}");
    }
    for b in [&b0, &b1] {
        for l in b.lines() {
            println!("{l}");
        }
    }
    println!(
        "inkworker: {} form, geometry digest {:016x} on both ranks; dry run done in {:.2} s, no card touched",
        if f.ink_one_process {
            "one-process (control)"
        } else {
            "worker-process"
        },
        b0.geom_digest,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PACK: &str = "/mnt/nvme/models/ink2-pack";

    fn pack() -> Option<Pack> {
        if !Path::new(PACK).join("manifest.json").is_file() {
            eprintln!("skip: {PACK} absent");
            return None;
        }
        Some(open_pack(Path::new(PACK), 2).expect("the full pack opens"))
    }

    #[test]
    fn ink2_the_dry_run_bills_the_full_pack_under_the_budget() {
        let Some(p) = pack() else { return };
        let f = InkFlags {
            ink_dry_run: true,
            ..InkFlags::default()
        };
        let b0 = bill_rank(&p, 0, &f, 2, 0).unwrap();
        let b1 = bill_rank(&p, 1, &f, 2, 0).unwrap();
        assert_eq!(b0.geom_digest, b1.geom_digest);
        let by = |b: &RankBill, n: &str| b.classes.iter().find(|c| c.0 == n).unwrap().1;
        // experts 6.51 GB a rank (§9's memory plan), every record resident
        assert_eq!(by(&b0, "ink.w.experts"), 1_536 * 4_235_264);
        assert_eq!(by(&b0, "ink.w.experts"), by(&b1, "ink.w.experts"));
        // the pack's dense bytes are the plan's (split + replicated)
        let (_, split, rep) = PackPlan::new(&p.cfg, ExpertForm::Mxfp4).rank_bytes();
        assert_eq!(by(&b0, "ink.w.dense.split"), split);
        let shared =
            by(&b0, "ink.w.embed") + by(&b0, "ink.w.dense.shared") + by(&b0, "ink.w.vision");
        assert_eq!(shared, rep);
        // the INK1 pack carries no vision tower (INK7); its class is billed and named, empty
        assert_eq!(by(&b0, "ink.w.vision"), 0);
        assert!(
            by(&b0, "ink.w.embed") == 201_024 * 2_304 * 2,
            "the whole table on each card"
        );
        assert!(b0.used < b0.capacity, "{:?}", b0.lines());
        assert_eq!(b0.launches, (948, b0.launches.1));
        // a bill round-trips through the pair's frame
        assert_eq!(RankBill::from_json(&b0.to_json()).unwrap(), b0);
    }

    #[test]
    fn ink2_a_bill_past_the_budget_is_refused_by_the_class_name() {
        let Some(p) = pack() else { return };
        let f = InkFlags {
            ink_dry_run: true,
            ink_max_ctx: 262_144,
            ..InkFlags::default()
        };
        let e = bill_rank(&p, 0, &f, 2, 0).unwrap_err().to_string();
        assert!(e.contains("ink.kv.global"), "{e}");
        // the P2PDMA pool is discounted from the fence
        let ok = InkFlags {
            ink_dry_run: true,
            ..InkFlags::default()
        };
        let a = bill_rank(&p, 0, &ok, 2, 0).unwrap();
        let b = bill_rank(&p, 0, &ok, 2, 2_048).unwrap();
        assert_eq!(a.capacity - b.capacity, 2_048 << 20);
    }

    #[test]
    fn ink2_the_worker_pair_agrees_in_one_process() {
        let Some(p) = pack() else { return };
        let f = InkFlags {
            ink_dry_run: true,
            ..InkFlags::default()
        };
        let (a, b) = ThreadPairExchange::pair();
        let (r0, r1) = std::thread::scope(|s| {
            let h = s.spawn(|| run_rank(&p, 1, &f, 2, 0, Arc::new(b)));
            (run_rank(&p, 0, &f, 2, 0, Arc::new(a)), h.join().unwrap())
        });
        assert!(r1.unwrap().is_none());
        let (b0, b1) = r0.unwrap().unwrap();
        assert_eq!((b0.rank, b1.rank), (0, 1));
        // a control: a rank planned with another geometry is refused by the digest
        let g = InkFlags {
            ink_max_ctx: 16_384,
            ..f.clone()
        };
        let (a, b) = ThreadPairExchange::pair();
        let (r0, r1) = std::thread::scope(|s| {
            let h = s.spawn(|| run_rank(&p, 1, &g, 2, 0, Arc::new(b)));
            (run_rank(&p, 0, &f, 2, 0, Arc::new(a)), h.join().unwrap())
        });
        assert!(r0.unwrap_err().to_string().contains("geometry digest"));
        assert!(r1.is_err());
    }

    #[test]
    fn ink2_the_pack_refusals_name_their_cause() {
        let Some(_) = pack() else { return };
        let e = open_pack(Path::new(PACK), 1).unwrap_err().to_string();
        assert!(e.contains("--tp 1") && e.contains("TWO-CARD"), "{e}");
        let fx = Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../saber-cache/ws-ink1/packs/ink2-l0l20"
        ));
        if fx.join("manifest.json").is_file() {
            let e = open_pack(fx, 2).unwrap_err().to_string();
            assert!(e.contains("fixture") || e.contains("plan_sha256"), "{e}");
        }
        assert!(InkFlags::default().refuse_elsewhere().is_ok());
        let on = InkFlags {
            ink_dry_run: true,
            ..InkFlags::default()
        };
        assert!(on.refuse_elsewhere().is_err());
    }

    /// INK41: a one-card view boots only at --tp 1, its TP=2 pack only at --tp 2 (each refusal
    /// names the other layout); a view whose blobs are not the pinned TP=2 manifest is refused.
    #[test]
    fn ink41_each_layout_refuses_the_other_by_name() {
        let (vdir, tdir) = (
            Path::new("/mnt/nvme/models/ink2-pack-nv-tp1"),
            Path::new("/mnt/nvme/models/ink2-pack-nv"),
        );
        if !vdir.join("manifest.json").is_file() {
            eprintln!("skip: {} absent", vdir.display());
            return;
        }
        assert!(is_tp1_view(vdir) && !is_tp1_view(tdir));
        let v = open_pack(vdir, 1).unwrap();
        assert!(v.view && v.manifest["experts_per_rank"] == 128);
        let e = open_pack(vdir, 2).unwrap_err().to_string();
        assert!(e.contains("ONE-CARD") && e.contains("--tp 2"), "{e}");
        let e = open_pack(tdir, 1).unwrap_err().to_string();
        assert!(e.contains("TWO-CARD") && e.contains("TP=1 view"), "{e}");
        // a wrong pin: the view copied with its blobs absolute and the TP=2 manifest pin moved
        let scratch = saber_test_scratch::ScratchDir::new("ink41-pin");
        let mut m: Value =
            serde_json::from_str(&std::fs::read_to_string(vdir.join("manifest.json")).unwrap())
                .unwrap();
        m["blobs"] = json!(tdir.display().to_string());
        std::fs::write(scratch.path().join("manifest.json"), m.to_string()).unwrap();
        assert!(open_pack(scratch.path(), 1).unwrap().view);
        m["blobs_manifest_sha256"] = json!("0".repeat(64));
        std::fs::write(scratch.path().join("manifest.json"), m.to_string()).unwrap();
        let e = open_pack(scratch.path(), 1).unwrap_err().to_string();
        assert!(e.contains("pins the TP=2 manifest"), "{e}");
    }

    /// HF3: a pack's own `base/` wins, else the recorded path (the control), else a refusal by name.
    #[test]
    fn hf3_the_checkpoint_resolves_to_the_packs_base_then_the_recorded_path() {
        let scratch = saber_test_scratch::ScratchDir::new("hf3-inkpaths");
        let (pack, rec) = (scratch.path().join("pack"), scratch.path().join("ckpt"));
        std::fs::create_dir_all(&pack).unwrap();
        let e = resolve_ckpt(&pack, &rec).unwrap_err().to_string();
        assert!(e.contains("fetch_packs.sh inkling"), "{e}");
        std::fs::create_dir_all(&rec).unwrap();
        std::fs::write(rec.join("config.json"), b"{}").unwrap();
        assert_eq!(
            resolve_ckpt(&pack, &rec).unwrap(),
            (rec.clone(), "the manifest's path")
        );
        // a base/ without the marker is not a checkpoint: the recorded path stays
        let base = pack.join(PACK_BASE_DIR);
        std::fs::create_dir_all(&base).unwrap();
        std::fs::write(base.join("config.json"), b"{}").unwrap();
        assert_eq!(resolve_ckpt(&pack, &rec).unwrap().0, rec);
        std::fs::write(base.join(CKPT_MARKER), b"{}").unwrap();
        assert_eq!(
            resolve_ckpt(&pack, &rec).unwrap(),
            (base, "the pack's own base/")
        );
    }

    /// INK19: the pack-data defaults stand down by name on an older pack (no `draft/`, or a
    /// `draft/` without `vocab_rank.u32`, or a rank file of the wrong size); explicit values stay.
    #[test]
    fn ink19_the_drafter_defaults_stand_down_on_an_older_pack() {
        let scratch = saber_test_scratch::ScratchDir::new("ink19-standdown");
        let pack = scratch.path();
        let f = InkFlags::default();
        // no draft/ at all: no drafter, the full head (one line: the vocab is moot at mtp 0)
        let (r, said) = f.resolved(pack);
        assert_eq!((r.mtp(), r.draft_vocab()), (0, 0));
        assert_eq!(said.len(), 1);
        assert!(
            said[0].starts_with("INK19 drafter: STANDS DOWN"),
            "{said:?}"
        );
        // draft/ without the rank file: the drafter at the default depth on the full head
        std::fs::create_dir_all(pack.join("draft")).unwrap();
        std::fs::write(pack.join("draft/manifest.json"), b"{}").unwrap();
        let (r, said) = f.resolved(pack);
        assert_eq!((r.mtp(), r.draft_vocab()), (DEFAULT_INK_MTP_UNGATED, 0));
        assert_eq!(r.mtp_gate_spec(), "");
        assert!(said.len() == 1 && said[0].contains("absent"), "{said:?}");
        // a rank file of the wrong size: the full head, said so
        std::fs::write(pack.join("draft/vocab_rank.u32"), [0u8; 16]).unwrap();
        let (r, said) = f.resolved(pack);
        assert_eq!(r.draft_vocab(), 0);
        assert!(
            said.len() == 1 && said[0].contains("wrong size"),
            "{said:?}"
        );
        // the full pack: the defaults, silent
        std::fs::write(
            pack.join("draft/vocab_rank.u32"),
            vec![0u8; crate::inkling2_exec::VOCAB * 4],
        )
        .unwrap();
        let (r, said) = f.resolved(pack);
        assert_eq!(
            (r.mtp(), r.draft_vocab()),
            (DEFAULT_INK_MTP, DEFAULT_INK_DRAFT_VOCAB)
        );
        assert!(said.is_empty());
        // explicit values are kept whatever the pack holds (their loads refuse by name)
        std::fs::remove_dir_all(pack.join("draft")).unwrap();
        let x = InkFlags {
            ink_mtp: Some(3),
            ink_draft_vocab: Some(16_384),
            ..InkFlags::default()
        };
        let (r, said) = x.resolved(pack);
        assert_eq!((r.mtp(), r.draft_vocab()), (3, 16_384));
        assert!(said.is_empty());
    }
}
