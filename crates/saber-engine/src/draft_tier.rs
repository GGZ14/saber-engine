//! The DFlash2 drafter as a **second model in the tier**: what it costs, what it would launch, and
//! — since ws-f26 — that every one of those launches has a code object.
//!
//! The target's half of speculation has been in the engine since ws-l14 — a verify step runs
//! `w = k + 1` rows, the acceptance law is `saber_exec::forward::accept_prefix`, the `Accepted` law
//! picks each row's recurrent snapshot, and the whole thing is gated against a SPEC=0 run at every
//! acceptance rate with mocks whose acceptance is known in advance ([`crate::draft`]). What is
//! missing is the half that *proposes*: a second forward pass, over a second checkpoint, with a
//! second KV cache, run inside the target's own decode step.
//!
//! This module is the accounting for that pass. It answers four questions on the HOST, with no
//! card and no pack open:
//!
//! 1. **What would it launch?** [`saber_model::dflash_graph::DraftGraph`] declares every launch site
//!    of a proposal with the shape it runs at, and [`DraftTier::budget`] is the covered subset —
//!    the number a capture's node count is asserted against, per `CLAUDE.md` rule 7.
//! 2. **What would it cost?** [`DraftTier::demand`] — weights from the record set's own shapes, the
//!    drafter KV cache from its own geometry, and the per-sequence context store. Every byte is
//!    derived from the plan rather than measured off an allocation, because the engine's byte
//!    ledger has to be able to refuse a tier that does not fit **before** it allocates 7.6 GiB
//!    (ws-k13: one byte, one reservation, named for whoever allocates it).
//! 3. **Which sessions does it follow?** The drafter's KV pages and its per-sequence context state
//!    are a session's, exactly as the target's are: [`DraftTier::admit`], [`DraftTier::release`].
//!    A drafter whose state outlived a session would propose from another request's history —
//!    accepted at 0 and therefore invisible in the ids, which is the failure mode ws-l14 named.
//! 4. **Can it propose?** [`DraftTier::readiness`] — **yes, since ws-f26**, and the answer is a
//!    check rather than a claim: every launch site the plan declares resolves through the
//!    committed rules to a code object, and the two questions "covered" and "resolvable" are both
//!    asked here. `CLAUDE.md` rule 5 still applies in the other direction — a site that loses its
//!    coverage, or a rules edit that refuses one, turns this back into a named refusal.
//!
//! # What this refused, for four rounds, and what closed it
//!
//! `saber draft-step --proposal` reproduced prod's own k=7 draft block **bit-exactly on prod's
//! scores** at 4399 µs against prod's 5511 (ws-f23, TP=1, card 0) — and that comparison was
//! **staged**: each stage seeded from prod's captured tensors at its input, because eleven of the
//! proposal's 107 launch sites had no Saber code object. Six activation quants (the blockscale
//! GEMM's own `bf16 -> e4m3` + per-128-group scales, which prod runs inside the layer's quant
//! method where no launch census sees it), the fused context-KV projection, its layout copy, and
//! three top-k selections.
//!
//! **A staged replay seeds past a gap and never notices it; a CHAIN cannot.** A drafter wired into
//! the engine over that plan would have run its 80 covered launches over buffers nothing wrote, and
//! the tokens it proposed would have been a function of the arena — which the acceptance law would
//! reject, correctly and silently, at the cost of the whole block. **Acceptance would collapse and
//! the ids would not move**, which is exactly the fault the ids cannot see. That is why this module
//! refused rather than proposed, and why the refusal named every site instead of saying `todo!()`.
//!
//! ws-f26 covered all eleven and ws-f27 declared and covered five more in the seam between the
//! drafter's forward and its head (two row gathers, `_pow2_at_least`'s row pad, `xs`, and the
//! `bi` gather), so the budget is 101 of 112 (the other 11 are another plugin's — the
//! head's embedding and the ten all-reduces) and the uncovered set is empty. The five
//! q/k-norm+rope launches moved INTO this budget in ws-f28: they were routed to
//! `attn_glue::qk_norm_rope`, which is the TARGET's GemmaRMSNorm (`1 + w`) over ATen's reduction
//! with a non-contracting rope, and the drafter's norm is the reference server's plain `RMSNorm` over Triton's.
//! They are `dflash_qk` now. What this module still owns is the ACCOUNTING; the proposal's own bit-exactness is
//! `gate_proposal.sh`'s claim and lives in `docs/GATES.md`.
//!
//! # The block is not the verify width, and the rules said so
//!
//! The first cut of this module took ONE width and used it for both the drafter's `block_size` and
//! the target's verify width, on the reading that "one number crosses the seam". The committed
//! dflash rules refuted it in the first test that asked: `grouped_conv_quant` and `conv_add_rms`
//! refuse any `block_size` that is not a POWER OF TWO, because prod's `_grouped_conv` indexes the
//! conv ring with `position & (block_size - 1)` and the modulo branch is a different kernel —
//! twenty covered launch sites, four per layer, refused at block 5 or 6 and resolving at 8.
//!
//! So `num_speculative_tokens` ∈ {1, 3, 7, 15} for this drafter, which is why prod runs **7**, and
//! the target's verify width is a SEPARATE number: the scheduler's acceptance EMA caps it between
//! steps and the drafter's block does not move. [`DraftTier::check_verify_width`] is the relation,
//! and this tree's `VERIFY_WIDTH = 5` was never a DFlash2 block — it is a verify width, and a
//! legal one.
//!
//! **Correction (ws-t1).** This paragraph used to read `SABER_DRAFT_SCHEDULE`'s `8:5` as "a
//! block of eight verified five rows at a time". It is not. The knob's value is
//! `1:8,2:7,4:6,8:5,16:4` — a `batch:max_serial_MTP_forwards` carry-forward table read by
//! `ref_draft.py::_batch_ceil` ([`saber_model::draft::DraftSchedule`]) — and it hooks the
//! **V1** MTP proposer, which `SPEC_METHOD=dflash` does not run at all
//! (`docs/abi/DFLASH2-SEMANTICS.md` § 6, and the ledger's "Dynamic draft inert under dflash").
//! prod's verify width at concurrency comes from `patch_dynwidth.py` alone; that it lands near
//! five at batch eight is a MEASUREMENT, not a setting.

use std::collections::{BTreeMap, BTreeSet};

use saber_model::dflash_graph::{DflashConfig, DraftGraph, DraftKvCache, DraftRope};
use saber_sched::SeqId;

use crate::forward::ForwardError;

/// What one drafter tier asks of the byte ledger, in bytes, derived from its own plan.
///
/// Separate fields rather than a total, because the three are owned by different allocators and
/// `Pool`s: the weights are an [`saber_mem::upload::Upload`]'s window, the KV cache is an arena the
/// engine sizes against the pool it has left, and the context store is per-session state that the
/// scheduler's capacity accounting has to see (ws-k13's `KvGeometry` is built from the engine's own
/// per-block figure for exactly this reason — the CLI once built it from a per-LAYER number and
/// under-counted the pool by 16x).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DraftDemand {
    /// Bytes of drafter weight the pack holds, summed over the record set's own shapes.
    pub weights: u64,
    /// Bytes of drafter K/V for ONE block across every drafter layer, both caches.
    ///
    /// The stride the drafter's own pool is indexed by, which is the number ws-k13 says to carry —
    /// not the per-layer, per-cache figure.
    pub kv_block_stride: u64,
    /// Bytes of drafter K/V at the block count this tier was sized for.
    pub kv_pool: u64,
    /// Sequences this tier's KV pool and session table were sized for.
    pub max_seqs: u32,
}

impl DraftDemand {
    /// Every byte this tier would hold at its sizing.
    ///
    /// # There is no context-store term, and there never was a buffer (ws-k15)
    ///
    /// This demand carried a `context_per_seq` line — `ctx_rows x hidden` bf16 per sequence, for
    /// "the drafter's projected context hidden states" — and the engine BILLED it
    /// (`engine.draft.context`). [`crate::device`]'s `build_drafter` allocates no such buffer and
    /// never did: the context store writes the target's tapped rows straight into the drafter's
    /// own **KV pages**, at `ctx_slot = block_id * block_size + pos % block_size`, and
    /// `DraftSession::slot` reaches the card only as `walk.idx_mapping`'s validity mask (ws-k14
    /// § 1, confirming it for the park image, which is complete without it).
    ///
    /// A reservation with nothing behind it is not a harmless over-count here: the KV pool is
    /// derived from what the byte ledger has LEFT (ws-k13), so every billed byte that holds
    /// nothing is a byte of context the target does not get. Removing it grows the pool by
    /// exactly `ctx_rows * hidden * 2 * max_seqs` per rank.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.weights + self.kv_pool
    }

    /// One line for the boot log and the report.
    #[must_use]
    pub fn line(&self) -> String {
        let mib = |b: u64| (b as f64) / (1024.0 * 1024.0);
        format!(
            "weights {:.0} MiB, kv {:.0} MiB ({} B/block, {} seq) = {:.0} MiB",
            mib(self.weights),
            mib(self.kv_pool),
            self.kv_block_stride,
            self.max_seqs,
            mib(self.total()),
        )
    }
}

/// What the drafter holds for one live sequence.
///
/// It mirrors the target's [`crate::device`] session and is the reason this is a type rather than a
/// count: the drafter's KV pages and its context slot are allocated at admission, rewritten in
/// place every step, and returned at release. A drafter that kept them would propose from a history
/// the request no longer has.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftSession {
    /// The drafter's context-store slot for this sequence.
    pub slot: u32,
    /// The drafter's own KV blocks, in order. One table per sequence, as the target has.
    pub kv_blocks: Vec<u32>,
    /// Tokens of this sequence the drafter's KV holds.
    pub done: u32,
    /// D4: the sequence's M-RoPE table, when its prompt has images. The drafter's rope reads its
    /// TEMPORAL axis (prod hands `copy_and_expand_dflash_inputs_kernel` the target's `[3, N]`
    /// positions, whose row 0 is what a 1-D load sees); `None` is the index.
    pub rope: Option<std::sync::Arc<saber_exec::mrope::MropePositions>>,
}

/// The drafter tier: one plan, one record set, one sizing, and one refusal.
#[derive(Clone, Debug)]
pub struct DraftTier {
    graph: DraftGraph,
    kv: DraftKvCache,
    demand: DraftDemand,
    blocks: u32,
    sessions: BTreeMap<SeqId, DraftSession>,
    free_slots: Vec<u32>,
    max_seqs: u32,
    /// The dense run a session gets: `ceil(max_ctx / page)` pages.
    per_seq: u32,
    /// P27-6: `Some` when the pool is paged by admission (`--engine-draft-kv paged`).
    runs: Option<crate::draft_runs::RunAlloc>,
}

impl DraftTier {
    /// Plan a drafter tier whose block is `block` query rows.
    ///
    /// **`block` is the DRAFTER's `block_size` = `1 + num_speculative_tokens`, and it is NOT the
    /// target's verify width.** The two were one number in the first cut of this module and the
    /// committed rules refuted it within the hour: `grouped_conv_quant` and `conv_add_rms` refuse
    /// any block that is not a power of two, because prod's `_grouped_conv` indexes its ring with
    /// `position & (block_size - 1)` and the modulo branch is a different kernel. Twenty of the
    /// drafter's covered launch sites — four per layer — are refused at block 5 or 6 and resolve at
    /// 8.
    ///
    /// So a DFlash2 drafter can only be built at `num_speculative_tokens` ∈ {1, 3, 7, 15}, which is
    /// why prod runs 7, and the target's verify width is a SEPARATE, narrower number that the
    /// scheduler's acceptance EMA caps between steps (`patch_dynwidth.py`:
    /// `w = max(ceil(ema) + 2, 2)`, applied only at three or more running requests, and the
    /// positions past `w` are DELETED — `del request.spec_token_ids[w:]` — not carried to the
    /// next step). [`DraftTier::check_verify_width`] is the relation between them.
    ///
    /// `SABER_DRAFT_SCHEDULE`'s `8:5` is NOT that relation and this comment said it was until
    /// ws-t1: that string is `1:8,2:7,4:6,8:5,16:4`, a `batch:max_serial_MTP_forwards`
    /// carry-forward table for the **V1** proposer, and `SPEC_METHOD=dflash` never reads it
    /// (`docs/abi/DFLASH2-SEMANTICS.md` § 6).
    ///
    /// # Errors
    /// A block of 0, and a block that is not a power of two — refused HERE, with the reason, rather
    /// than twenty launches into a proposal.
    pub fn plan(tp: u32, block: u32, max_ctx: u32, max_seqs: u32) -> Result<Self, ForwardError> {
        Self::plan_with(
            tp,
            block,
            max_ctx,
            max_seqs,
            saber_model::dflash_graph::RerankHead::Bf16,
        )
    }

    /// [`DraftTier::plan`] with the exact rerank reading `rerank` (N6): the drafter PACK's record
    /// kind (`head_share::drafter_rerank_kind`), so the plan, the record set and the billed weights
    /// are the pack's — `head.rerank.w8` + `.ws` (0.636 GB per TP=2 rank) instead of the bf16
    /// `head.rerank.w` (1.27 GB).
    ///
    /// # Errors
    /// As [`DraftTier::plan`].
    pub fn plan_with(
        tp: u32,
        block: u32,
        max_ctx: u32,
        max_seqs: u32,
        rerank: saber_model::dflash_graph::RerankHead,
    ) -> Result<Self, ForwardError> {
        if block < 2 {
            return Err(ForwardError::Device(format!(
                "a drafter tier at block {block}: the DFlash2 block is \
                 `1 + num_speculative_tokens` query rows and the selector proposes for the \
                 `block - 1` MASK positions, so a block of 1 proposes NOTHING. That is SPEC=0, \
                 which is a tier with no drafter in it rather than a drafter with no proposals — \
                 the selector's launches are refused as empty (`1 x 0 x 16`) and rightly so"
            )));
        }
        if block & (block - 1) != 0 {
            return Err(ForwardError::Device(format!(
                "a DFlash2 block of {block} is not a power of two: `grouped_conv_quant` and \
                 `conv_add_rms` index the conv ring with `position & (block_size - 1)` (prod's \
                 `_grouped_conv`), and the modulo branch is a different kernel. Four launches per \
                 layer are refused at this block. num_speculative_tokens must be one of 1, 3, 7, \
                 15 — prod runs 7"
            )));
        }
        let cfg = DflashConfig::qwen38_dflash2(tp.max(1), block.saturating_sub(1));
        let graph = DraftGraph::build(cfg.clone()).with_rerank_head(rerank);
        let kv = graph.kv_cache(block_size());
        let per_seq_blocks = max_ctx.div_ceil(kv.block_size);
        let blocks = per_seq_blocks * max_seqs.max(1);
        // The record set's own shapes: no pack is opened, so this is what the plan SAYS the tier
        // weighs and a pack that disagrees is a pack that does not match the plan.
        let records = graph.records(DraftRope::qwen38_dflash2(max_ctx));
        let weights: u64 = records
            .iter()
            .map(|r| {
                let elems: u64 = r.shape.iter().product();
                elems * u64::from(r.dtype.bits()) / 8
            })
            .sum();
        let demand = DraftDemand {
            weights,
            kv_block_stride: 2 * u64::from(kv.layers) * kv.block_bytes(),
            kv_pool: kv.total_bytes(u64::from(blocks)),
            max_seqs: max_seqs.max(1),
        };
        Ok(DraftTier {
            graph,
            kv,
            demand,
            blocks,
            sessions: BTreeMap::new(),
            // Slot 0 is handed out: unlike the GDN state arena the drafter's context store has no
            // null slot, because no drafter kernel takes a padded row. A padded verify row never
            // reaches this tier at all — the proposal runs per LIVE sequence.
            free_slots: (0..max_seqs.max(1)).rev().collect(),
            max_seqs: max_seqs.max(1),
            per_seq: per_seq_blocks,
            runs: None,
        })
    }

    /// **P27-6: page this tier's KV by admission** over a pool of `pages` (the engine passes the
    /// target pool's block count plus one page a sequence), instead of `max_seqs` dense runs of
    /// [`DraftTier::per_seq_pages`]. Called once at boot, after the KV pool is derived and before
    /// the drafter's caches are allocated; the demand's `kv_pool` follows.
    pub fn page_by_admission(&mut self, pages: u32) {
        self.blocks = pages;
        self.demand.kv_pool = self.kv.total_bytes(u64::from(pages));
        self.runs = Some(crate::draft_runs::RunAlloc::new(pages));
    }

    /// Whether this tier's KV is paged by admission (P27-6).
    #[must_use]
    pub fn is_paged(&self) -> bool {
        self.runs.is_some()
    }

    /// The pages of a DENSE session's run (`ceil(max_ctx / page)`), which is also the longest run
    /// a paged session is granted.
    #[must_use]
    pub fn per_seq_pages(&self) -> u32 {
        self.per_seq
    }

    /// The run a paged session is granted for a target reservation of `target_blocks` pages: the
    /// positions the reservation holds plus the drafter's own block of query rows past them (a
    /// proposal writes `last_valid_pos + 1 ..= last_valid_pos + block`), capped at the dense run
    /// (positions at or past `max_ctx` are the drafter's PAD_SLOT and need no page).
    #[must_use]
    pub fn run_pages_for(&self, target_blocks: u32) -> u32 {
        let page = self.kv.block_size.max(1);
        let tokens =
            u64::from(target_blocks) * u64::from(page) + u64::from(self.graph.cfg.block_size);
        let pages = u32::try_from(tokens.div_ceil(u64::from(page))).unwrap_or(u32::MAX);
        pages.min(self.per_seq).max(1)
    }

    /// P27-6: grant `seq` a contiguous run of `len` pages (`compact`: slide live runs to make room).
    /// The moves are applied to the tier's own sessions here; the CALLER copies the bytes on both
    /// ranks ([`crate::draft_runs::RunMove::copies`]) before anything reads them.
    ///
    /// # Errors
    /// A dense tier, or no room (named by [`crate::draft_runs::RunAlloc::grant`]).
    pub fn grant_run(
        &mut self,
        seq: SeqId,
        len: u32,
        compact: bool,
    ) -> Result<crate::draft_runs::RunGrant, ForwardError> {
        let runs = self.runs.as_mut().ok_or_else(|| {
            ForwardError::Device(format!("{seq}: a page-run grant on a DENSE drafter tier"))
        })?;
        let g = runs
            .grant(seq, len, compact)
            .map_err(ForwardError::Device)?;
        for m in &g.moves {
            if let Some(s) = self.sessions.get_mut(&m.seq) {
                s.kv_blocks = (m.to..m.to + m.len).collect();
            }
        }
        Ok(g)
    }

    /// P27-6: pages held by live runs, of the pool (`None` on a dense tier).
    #[must_use]
    pub fn paged_use(&self) -> Option<(u32, u32)> {
        self.runs.as_ref().map(|r| (r.used(), r.pages()))
    }

    /// The relation between the drafter's block and the target's verify width.
    ///
    /// A verify step of `w` rows feeds row 0 the last committed token and rows `1..w` the draft, so
    /// it needs `w - 1` draft tokens and the drafter has exactly `block - 1` to give. Equal is the
    /// full block; narrower is what the scheduler's acceptance EMA does between steps and is legal
    /// — the extra proposals are simply not verified. WIDER is a step that would feed rows nothing
    /// proposed, which the engine pads with a token the model cannot want: legal, counted, and a
    /// silent waste of the widest rows of every step, so it is refused by name instead.
    ///
    /// # Errors
    /// A verify width above the drafter's block.
    pub fn check_verify_width(&self, width: u32) -> Result<(), ForwardError> {
        let block = self.graph.cfg.block_size;
        if width > block {
            return Err(ForwardError::Device(format!(
                "a verify width of {width} against a DFlash2 block of {block}: the drafter has \
                 {} token(s) to propose and this step would feed {} row(s) from them, so \
                 {} row(s) of every step would carry a token the model cannot want",
                block - 1,
                width - 1,
                width - block,
            )));
        }
        Ok(())
    }

    /// The plan this tier would run.
    #[must_use]
    pub fn graph(&self) -> &DraftGraph {
        &self.graph
    }

    /// The plan, mutably — for a TEST that has to construct a tier whose plan is wrong.
    ///
    /// `readiness()`'s coverage arm cannot be exercised any other way now that nothing is absent,
    /// and a check for gaps that has never been shown to find one is a line of output. Marked
    /// `#[doc(hidden)]` rather than `#[cfg(test)]` because the control lives in this crate's own
    /// test module and a `cfg(test)` accessor would not be visible to an integration test that
    /// wanted the same control later.
    #[doc(hidden)]
    #[must_use]
    pub fn graph_mut(&mut self) -> &mut DraftGraph {
        &mut self.graph
    }

    /// The drafter's KV geometry.
    #[must_use]
    pub fn kv_cache(&self) -> DraftKvCache {
        self.kv
    }

    /// KV blocks the tier was sized for.
    #[must_use]
    pub fn kv_blocks(&self) -> u32 {
        self.blocks
    }

    /// What this tier asks of the byte ledger.
    #[must_use]
    pub fn demand(&self) -> DraftDemand {
        self.demand
    }

    /// Launches one proposal makes that a Saber code object covers — the number a capture asserts
    /// its node count against.
    #[must_use]
    pub fn budget(&self) -> u32 {
        self.graph.budget()
    }

    /// Launch sites a proposal makes that nothing covers, with what each would need.
    #[must_use]
    pub fn uncovered(&self) -> Vec<(String, &'static str, u32)> {
        self.graph
            .uncovered()
            .into_iter()
            .map(|(n, why, c)| (n.to_owned(), why, c))
            .collect()
    }

    /// Launches of a proposal that a **host round trip** would have to carry, rather than a node in
    /// the step's graph.
    ///
    /// The answer is the uncovered set and nothing else, and that is worth stating precisely
    /// because it decides whether the proposal can join the step's captured graph at all
    /// (`docs/EXEC.md` § 4). Everything the plan covers is an ordinary dispatch over addresses this
    /// tier owns — including the walk, whose sequential argmax chain is ON DEVICE
    /// (`dflash_selector_walk`, one program, one wave, `block_size - 1` dependent argmaxes), and
    /// including the two ops a reader would expect to be host round trips: the int2 head's
    /// `bm.topk(RERANK)` and the selector's `torch.topk(logits, 16)` are `dflash_topk` code
    /// objects, not ATen calls a host would have to make.
    ///
    /// Since ws-f26 the uncovered set is EMPTY, so this is zero — the whole proposal is graphable.
    /// It is still computed rather than returned as a constant, because the thing that would make
    /// it non-zero again is a plan edit, and a plan edit is exactly what this should catch.
    #[must_use]
    pub fn host_round_trips(&self) -> u32 {
        self.graph.uncovered().iter().map(|(_, _, n)| n).sum()
    }

    /// Launch sites the plan COVERS and the committed rules then refuse — a different failure from
    /// an absent op and a worse one, because the plan says it is covered.
    ///
    /// `DflashShape` carries fields a descriptor cannot be resolved without (ws-f21 found five
    /// attention launches refused by name for want of `bt_stride`), so "covered" and "resolvable"
    /// are two questions. Asking the second one here means a tier reports both in one place rather
    /// than the second one surfacing as a launch that never happens.
    #[must_use]
    pub fn refusals(&self) -> Vec<(String, String)> {
        match saber_kernels::dflash::DflashRules::prod_dflash() {
            Ok(rules) => self.graph.refusals(&rules),
            Err(e) => vec![(
                "rules/dflash_gfx1201.toml".to_owned(),
                format!("the committed dflash rules do not parse: {e}"),
            )],
        }
    }

    /// Whether this tier can propose, and everything stopping it if it cannot.
    ///
    /// # Errors
    /// A launch site with no code object, or a covered site the committed rules refuse. Both were
    /// live until ws-f26 — eleven absent launches, and before ws-f22 five attention launches that
    /// the plan called covered and `resolve` would not serve for want of a `bt_stride` — and the
    /// message names every one of them with its count, because a drafter that ran its covered
    /// launches over buffers nothing wrote would propose a function of the arena and the
    /// acceptance law would reject it silently.
    pub fn readiness(&self) -> Result<(), ForwardError> {
        let refused = self.refusals();
        let missing = self.uncovered();
        if missing.is_empty() && refused.is_empty() {
            return Ok(());
        }
        if !refused.is_empty() {
            let names: Vec<String> = refused
                .iter()
                .map(|(op, why)| format!("{op} ({why})"))
                .collect();
            return Err(ForwardError::Device(format!(
                "the DFlash2 drafter cannot propose: {} covered launch site(s) are REFUSED by the \
                 committed rules, which is worse than an absent op because the plan says they are \
                 covered: {}",
                refused.len(),
                names.join("; "),
            )));
        }
        let total: u32 = missing.iter().map(|(_, _, n)| n).sum();
        let names: Vec<String> = missing
            .iter()
            .map(|(n, why, c)| format!("{n} x{c} ({why})"))
            .collect();
        Err(ForwardError::Device(format!(
            "the DFlash2 drafter cannot propose: {} of this proposal's {} launch site(s) have no \
             code object ({total} launch(es)), and a proposal is a CHAIN — the covered {} would \
             run over buffers nothing wrote and the block would be a function of the arena. The \
             acceptance law would reject it, so the ids would not move and the acceptance would \
             collapse, which is the one failure the ids cannot see. Missing: {}",
            missing.len(),
            self.graph.total_launches(),
            self.budget(),
            names.join("; "),
        )))
    }

    /// Give a sequence the drafter's own session state: a context slot and its KV pages.
    ///
    /// # Errors
    /// No context slot left, or a KV block outside the pool this tier was sized for — the same two
    /// refusals the target's side makes, for the same reason: a block id the pool does not contain
    /// is another session's pages.
    pub fn admit(&mut self, seq: SeqId, kv_blocks: &[u32]) -> Result<(), ForwardError> {
        if self.sessions.contains_key(&seq) {
            return Err(ForwardError::Device(format!(
                "{seq}: the drafter tier already holds a session for this sequence"
            )));
        }
        if let Some(bad) = kv_blocks.iter().find(|b| **b >= self.blocks) {
            return Err(ForwardError::Device(format!(
                "{seq}: drafter KV block {bad} is outside a pool of {} block(s)",
                self.blocks
            )));
        }
        let slot = self.free_slots.pop().ok_or(ForwardError::AtCapacity {
            live: self.sessions.len() as u32,
            max: self.max_seqs,
        })?;
        self.sessions.insert(
            seq,
            DraftSession {
                slot,
                kv_blocks: kv_blocks.to_vec(),
                done: 0,
                rope: None,
            },
        );
        Ok(())
    }

    /// D4: place this session's rope by an M-RoPE table (`None`: the index).
    ///
    /// # Errors
    /// A sequence the tier holds no session for.
    pub fn set_rope(
        &mut self,
        seq: SeqId,
        rope: Option<std::sync::Arc<saber_exec::mrope::MropePositions>>,
    ) -> Result<(), ForwardError> {
        let s = self.sessions.get_mut(&seq).ok_or_else(|| {
            ForwardError::Device(format!("{seq}: no drafter session to place a rope on"))
        })?;
        s.rope = rope;
        Ok(())
    }

    /// Drop a sequence's drafter state. Its context slot goes back to the free list.
    ///
    /// Idempotent, because release is called on paths that may not have admitted — an abort during
    /// admission, a park that failed after the target's half was released.
    pub fn release(&mut self, seq: SeqId) {
        if let Some(s) = self.sessions.remove(&seq) {
            self.free_slots.push(s.slot);
        }
        // P27-6: a paged run goes back with the session (granted runs whose admission failed too).
        if let Some(r) = self.runs.as_mut() {
            r.free(seq);
        }
    }

    /// **Give a BRANCH of `leader` its own drafter session, as a copy of the leader's** (ws-f38).
    ///
    /// The drafter half of an `n > 1` fork. The target's half (`device::fork_session`) shares the
    /// prompt's whole KV pages read-only, because a target page is only ever APPENDED to past the
    /// prompt. A drafter page is not like that: every proposal rewrites the positions of its own
    /// block (`last_valid_pos + 1 ..`), which on the prompt's last page are positions both branches
    /// would write — and a drafter run is one CONTIGUOUS piece per session (the park images it as
    /// one region per layer). So the branch gets a run of its own, `kv_blocks`, and the plan this
    /// returns copies the leader's pages that hold anything into it and zeroes the rest, which is
    /// exactly what an independent admission (`clear_pages`) followed by the leader's own context
    /// stores would have left there.
    ///
    /// `resident` is the positions the leader's pages hold: the prompt, plus the first block the
    /// leader's prefill proposal wrote past it. Positions past that are zeros in both runs.
    ///
    /// # Errors
    /// Every refusal names the branch and leaves the tier as it was: a leader with no session, a
    /// branch that already has one, no free context slot, a run outside the pool or not one
    /// contiguous piece, runs of different lengths, and — [`branch_is_distinct`] — a branch whose
    /// slot or pages are the leader's.
    pub fn fork(
        &mut self,
        leader: SeqId,
        branch: SeqId,
        kv_blocks: &[u32],
        resident: u32,
    ) -> Result<DraftForkCopy, ForwardError> {
        let lead = self.sessions.get(&leader).cloned().ok_or_else(|| {
            ForwardError::Device(format!(
                "{branch}: forked from {leader}, which has no drafter session"
            ))
        })?;
        // The slot the branch WILL get (`admit` pops the same end), checked before anything is
        // taken, so a refusal leaves the free list untouched.
        let slot = *self.free_slots.last().ok_or(ForwardError::AtCapacity {
            live: self.sessions.len() as u32,
            max: self.max_seqs,
        })?;
        let candidate = DraftSession {
            slot,
            kv_blocks: kv_blocks.to_vec(),
            done: lead.done,
            rope: lead.rope.clone(),
        };
        let copy = self.fork_plan(branch, &lead, &candidate, resident)?;
        self.admit(branch, kv_blocks)?;
        if let Some(s) = self.sessions.get_mut(&branch) {
            s.done = lead.done;
            // D4: a branch is the leader's prompt, so it is the leader's rope.
            s.rope = lead.rope.clone();
        }
        Ok(copy)
    }

    /// The copy a fork makes, for a branch session the CALLER chose — the law [`DraftTier::fork`]
    /// applies, callable directly, which is how the control hands it the leader's own slot or
    /// pages.
    ///
    /// # Errors
    /// See [`DraftTier::fork`].
    pub fn fork_plan(
        &self,
        branch: SeqId,
        leader: &DraftSession,
        candidate: &DraftSession,
        resident: u32,
    ) -> Result<DraftForkCopy, ForwardError> {
        if self.sessions.contains_key(&branch) {
            return Err(ForwardError::Device(format!(
                "{branch}: the drafter tier already holds a session for this branch"
            )));
        }
        branch_is_distinct(branch, leader, candidate)?;
        if let Some(bad) = candidate.kv_blocks.iter().find(|b| **b >= self.blocks) {
            return Err(ForwardError::Device(format!(
                "{branch}: drafter KV block {bad} is outside a pool of {} block(s)",
                self.blocks
            )));
        }
        let (from_first, from_run) = one_run(branch, "the leader's", &leader.kv_blocks)?;
        let (to_first, run) = one_run(branch, "its own", &candidate.kv_blocks)?;
        if from_run != run {
            return Err(ForwardError::Device(format!(
                "{branch}: a drafter run of {run} page(s) forked from a leader holding {from_run}"
            )));
        }
        let copy = resident.div_ceil(self.kv.block_size.max(1)).min(run);
        Ok(DraftForkCopy {
            from_first,
            to_first,
            run,
            copy,
        })
    }

    /// A sequence's drafter session, if it has one.
    #[must_use]
    pub fn session(&self, seq: SeqId) -> Option<&DraftSession> {
        self.sessions.get(&seq)
    }

    /// Live drafter sessions.
    #[must_use]
    pub fn live(&self) -> usize {
        self.sessions.len()
    }

    /// The record names this tier binds, for a pack check that does not open a pack.
    #[must_use]
    pub fn record_names(&self, max_ctx: u32) -> BTreeSet<String> {
        self.graph
            .records(DraftRope::qwen38_dflash2(max_ctx))
            .into_iter()
            .map(|r| r.name)
            .collect()
    }
}

/// The device copy a drafter FORK makes (ws-f38), in the drafter pool's own page numbers.
///
/// Both ranks run exactly this over their own half of the drafter's K/V (`Command::ForkDraft`):
/// for every drafter layer and both caches, pages `from_first .. from_first + copy` are copied to
/// `to_first ..`, and the rest of the branch's run, `to_first + copy .. to_first + run`, is zeroed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftForkCopy {
    /// The leader's first page.
    pub from_first: u32,
    /// The branch's first page.
    pub to_first: u32,
    /// Pages in each run.
    pub run: u32,
    /// Pages copied from the front of the leader's run; the rest of the branch's is zeroed.
    pub copy: u32,
}

/// **A branch's drafter session is not its leader's** (ws-f38): a different context slot and not
/// one page in common.
///
/// A branch that shared the leader's slot would write its `sample_idx_mapping` entry over the
/// leader's in a batched proposal; one that shared a page would rewrite the leader's block
/// positions with its own and the leader would propose from them. Neither changes a token
/// (`draft::Greedy` commits the target's answer); both change the ACCEPTANCE, which no gate on the
/// ids can see — so this is a refusal by name.
///
/// # Errors
/// The shared slot, or the first shared page, named.
pub fn branch_is_distinct(
    branch: SeqId,
    leader: &DraftSession,
    candidate: &DraftSession,
) -> Result<(), ForwardError> {
    if candidate.slot == leader.slot {
        return Err(ForwardError::Device(format!(
            "{branch}: a branch that REUSES its leader's drafter context slot {} is refused — two \
             sessions of one group would write one slot",
            leader.slot
        )));
    }
    if let Some(p) = candidate
        .kv_blocks
        .iter()
        .find(|p| leader.kv_blocks.contains(p))
    {
        return Err(ForwardError::Device(format!(
            "{branch}: a branch whose drafter pages include its leader's page {p} is refused — \
             every proposal rewrites its block's positions, so a shared drafter page is two \
             sessions writing one page"
        )));
    }
    Ok(())
}

/// A drafter run as `(first, pages)`, or a refusal naming whose run was not one piece.
fn one_run(branch: SeqId, whose: &str, pages: &[u32]) -> Result<(u32, u32), ForwardError> {
    let Some(&first) = pages.first() else {
        return Err(ForwardError::Device(format!(
            "{branch}: {whose} drafter run holds no pages"
        )));
    };
    if pages
        .iter()
        .enumerate()
        .any(|(i, p)| u64::from(*p) != u64::from(first) + i as u64)
    {
        return Err(ForwardError::Device(format!(
            "{branch}: {whose} drafter pages are not one contiguous run from {first}"
        )));
    }
    Ok((first, pages.len() as u32))
}

/// The drafter's KV block size, from the committed saber rules.
///
/// Read rather than restated: `dflash_attn` divides a key position by it to index the block table,
/// so a number written down twice is a block table that indexes the wrong page — in range, and
/// therefore silent.
fn block_size() -> u32 {
    saber_kernels::saber::SaberRules::prod_saber()
        .map(|r| r.knobs.attn_block_size)
        .unwrap_or(16)
}

/// **P27-9d: the drafter's rope table covers the booted context, or the boot is refused.**
///
/// `dflash_rope_neox_k_inplace` (the context store's K rope) and `dflash_qk` (the block's query
/// rope) read `cos_sin + position * rotary_dim` with NO bound. Until P27-9d the bound table was the
/// pack's record, `[4096, 128]`, while the served default runs 262,144: every drafter row past
/// position 4,096 read its cos / sin from the NEXT record (the drafter's own embedding) -- wrong
/// rope, no fault -- and with the embedding share on (that record not uploaded) the read left the
/// allocation and faulted (P27-9c). The engine now builds the table for `max_ctx` at load
/// ([`DraftRope::cos_sin_bf16`]); this is the law it is held to, named with both numbers.
///
/// # Errors
/// A table of fewer rows than `max_ctx`.
pub fn check_rope_rows(table_rows: u64, max_ctx: u32, source: &str) -> Result<(), ForwardError> {
    if table_rows < u64::from(max_ctx) {
        return Err(ForwardError::Device(format!(
            "the drafter's rope table `{}` ({source}) holds {table_rows} position(s) and this \
             boot's max_ctx is {max_ctx}: the drafter's rope kernels index it by position with no \
             bound, so every position past {table_rows} would read the bytes after the table \
             (P27-9d). Build it for max_ctx (the engine does, at load) or rebuild the pack",
            DraftRope::RECORD
        )));
    }
    Ok(())
}

/// **P27-9d: the launch-side check** -- the highest position an `i64` position table (as the host
/// seeds it) hands the drafter's rope kernels is inside the bound table. Host-side, before the
/// table's copy is posted: no kernel changes, and a refusal by name instead of a read past the end.
///
/// # Errors
/// A position at or past `table_rows`, or a negative one, named with the table.
pub fn check_rope_positions(
    table: &str,
    bytes: &[u8],
    table_rows: u64,
) -> Result<(), ForwardError> {
    let mut top = 0i64;
    for c in bytes.chunks_exact(8) {
        let p = i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
        if p < 0 {
            return Err(ForwardError::Device(format!(
                "the drafter's `{table}` carries position {p}: the rope kernels read `cos_sin + \
                 position * rotary_dim` with no bound (P27-9d)"
            )));
        }
        top = top.max(p);
    }
    if !bytes.is_empty() && top.unsigned_abs() >= table_rows {
        return Err(ForwardError::Device(format!(
            "the drafter's `{table}` reaches position {top} and the bound rope table `{}` holds \
             {table_rows} row(s): the launch would read past the table (P27-9d)",
            DraftRope::RECORD
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use saber_model::dflash_graph::Coverage;

    const S: SeqId = SeqId(1);

    /// The tier can propose, and the check that says so is still able to say the opposite.
    ///
    /// Two halves, and the second is what keeps the first from being a line of output. The first
    /// is the claim: at prod's own geometry `readiness()` is `Ok`, the uncovered set is empty, and
    /// the whole proposal is graphable (`host_round_trips() == 0`). The second is the CONTROL —
    /// a tier whose plan carries one absent op must refuse and must name it — because "no gaps"
    /// asserted by a function that has forgotten how to find one is not an assertion.
    ///
    /// The control is built by hand rather than by breaking the rules file: a rules edit would be
    /// a refusal (the other arm of `readiness`), and this arm is about COVERAGE.
    #[test]
    fn the_drafter_tier_can_propose_and_still_knows_how_to_refuse() {
        let tier = DraftTier::plan(1, 8, 4096, 8).expect("a legal tier");
        assert!(
            tier.uncovered().is_empty(),
            "still absent: {:?}",
            tier.uncovered()
        );
        assert_eq!(tier.refusals(), Vec::new(), "the rules refuse a covered op");
        tier.readiness()
            .expect("every launch site has a code object");
        assert_eq!(
            tier.host_round_trips(),
            0,
            "the whole proposal is device-side, the two top-k selections included"
        );
        assert_eq!(tier.budget(), 101);
        // 117 since ws-s6: the embedding's own all-reduce and the four-launch candidate seam,
        // both of which exist only at TP > 1 and both of which the plan declares at every TP, as
        // the ten layer reduces always were.
        assert_eq!(tier.graph().total_launches(), 117);

        // THE CONTROL. One op put back to `Absent`, and the refusal has to name it, count it, and
        // say why a chain cannot be seeded past it.
        let mut broken = DraftTier::plan(1, 8, 4096, 8).expect("a legal tier");
        let why = "the control: a producer with no code object";
        broken
            .graph_mut()
            .ops
            .iter_mut()
            .find(|o| o.name == "aux_hidden_projection.quant")
            .expect("the site the control needs")
            .coverage = Coverage::Absent(why);
        let err = broken
            .readiness()
            .expect_err("one absent op is one refusal");
        let text = err.to_string();
        assert!(text.contains("aux_hidden_projection.quant"), "{text}");
        assert!(text.contains(why), "{text}");
        assert!(text.contains("CHAIN"), "{text}");
        assert_eq!(broken.host_round_trips(), 1);
    }

    /// **The block must be a power of two, and the target's verify width is a different number.**
    ///
    /// This is the finding that cost the first cut of this module: the two were one parameter, and
    /// the committed rules refuse `grouped_conv_quant` and `conv_add_rms` at any block that is not
    /// a power of two — four launches per layer, twenty in all — because prod's `_grouped_conv`
    /// indexes its ring with `position & (block_size - 1)`. So `num_speculative_tokens` ∈
    /// {1, 3, 7, 15}, prod runs 7, and the target's verify width is a different number again
    /// (the scheduler's acceptance-EMA cap, never `SABER_DRAFT_SCHEDULE` — see this module's
    /// header for the correction).
    ///
    /// NEGATIVE CONTROL in the same test: blocks 5 and 6 — the two this tree would have reached
    /// for, being "SPEC=4" and "SPEC=5" — are refused with the reason.
    #[test]
    fn the_block_is_a_power_of_two_and_the_verify_width_is_not_it() {
        for block in [2u32, 4, 8] {
            let tier = DraftTier::plan(1, block, 4096, 8).expect("a legal block");
            assert_eq!(tier.graph().cfg.block_size, block, "block {block}");
            // The selector proposes for the MASK positions: one fewer than the block.
            assert_eq!(
                tier.graph().cfg.selector_positions(),
                block.saturating_sub(1)
            );
            assert_eq!(tier.refusals(), Vec::<(String, String)>::new());
        }
        for bad in [3u32, 5, 6, 7, 9] {
            let err = DraftTier::plan(1, bad, 4096, 8).expect_err("not a power of two");
            assert!(err.to_string().contains("power of two"), "{err}");
        }
        // A block of 1 is a power of two and is still not a drafter: it proposes nothing, and the
        // selector's launches would be refused as empty.
        for none in [0u32, 1] {
            let err = DraftTier::plan(1, none, 4096, 8).expect_err("no proposals");
            assert!(err.to_string().contains("SPEC=0"), "{err}");
        }
        // The width relation: up to the block is legal, past it is refused by name.
        let tier = DraftTier::plan(1, 8, 4096, 8).expect("prod's block");
        for w in 1..=8 {
            tier.check_verify_width(w).expect("within the block");
        }
        let err = tier.check_verify_width(9).expect_err("past the block");
        assert!(err.to_string().contains("cannot want"), "{err}");
    }

    /// The demand is derived from the plan, and every part of it moves for a reason.
    ///
    /// The absolute numbers are not asserted — they would be a transcription of the plan and would
    /// go stale with it. What is asserted is that each term responds to the input that owns it,
    /// which is what makes this an accounting rather than a constant: more context is more KV, more
    /// sequences is more KV, and the weights are the same at every sizing because a checkpoint does
    /// not grow with the session table.
    ///
    /// **And the demand is WEIGHTS PLUS KV and nothing else** (ws-k15): the context-store term this
    /// carried was a buffer `build_drafter` never allocated, so the total is asserted against the
    /// two terms that are real rather than against a sum that included a third.
    #[test]
    fn the_demand_is_derived_and_each_term_moves_with_its_own_input() {
        let small = DraftTier::plan(1, 8, 4096, 8).expect("a tier").demand();
        let long = DraftTier::plan(1, 8, 8192, 8).expect("a tier").demand();
        let many = DraftTier::plan(1, 8, 4096, 16).expect("a tier").demand();
        assert!(small.weights > 0);
        // A checkpoint does not grow with the session table.
        assert_eq!(small.weights, many.weights, "weights are the checkpoint's");
        // It DOES grow with the context, by exactly one record: the rope table is the only
        // COMPUTED record in the set and `max_pos` is its first dimension. Asserted as "the whole
        // difference is that record" rather than as a number, because a number here would be a
        // transcription of the plan and would go stale with it.
        let rope_bytes = |ctx: u32| -> u64 {
            let g = DraftTier::plan(1, 8, ctx, 8).expect("a tier");
            g.graph
                .records(DraftRope::qwen38_dflash2(ctx))
                .iter()
                .filter(|r| r.name.ends_with("cos_sin"))
                .map(|r| r.shape.iter().product::<u64>() * u64::from(r.dtype.bits()) / 8)
                .sum()
        };
        assert!(rope_bytes(4096) > 0, "the rope table is a record");
        assert_eq!(
            long.weights - small.weights,
            rope_bytes(8192) - rope_bytes(4096),
            "only the rope table grows with the context"
        );
        assert_eq!(long.kv_pool, 2 * small.kv_pool, "twice the context");
        assert_eq!(many.kv_pool, 2 * small.kv_pool, "twice the sequences");
        assert_eq!(many.max_seqs, 16);
        assert!(small.total() > small.weights, "the session state is in it");
        assert_eq!(
            small.total(),
            small.weights + small.kv_pool,
            "the demand is the weights and the KV pool: there is no context-store buffer"
        );
        // TP=2 halves the shards, so a rank holds less of everything but the replicated records.
        let tp2 = DraftTier::plan(2, 8, 4096, 8).expect("a tier").demand();
        assert!(tp2.weights < small.weights, "a rank holds its own half");
        assert_eq!(tp2.kv_block_stride * 2, small.kv_block_stride);
        assert!(small.line().contains("MiB"));
    }

    /// The drafter's session follows the target's: admitted with it, released with it, and its
    /// context slot re-used.
    ///
    /// NEGATIVE CONTROL in the same test: a KV block outside the pool is refused by name, and the
    /// slot table runs out rather than handing two sequences one slot. A drafter that kept a
    /// released session's state would propose from another request's history — accepted at 0, so
    /// the ids would not move and nothing would ever say so.
    #[test]
    fn a_drafter_session_is_admitted_and_released_with_the_target_and_its_slots_return() {
        let mut tier = DraftTier::plan(1, 8, 4096, 2).expect("a tier");
        tier.admit(S, &[0, 1]).expect("admitted");
        assert_eq!(tier.live(), 1);
        let first = tier.session(S).expect("a session").slot;
        // The same sequence twice is a protocol error, not a second slot.
        assert!(tier.admit(S, &[0, 1]).is_err());
        tier.admit(SeqId(2), &[2]).expect("admitted");
        // Sized for two: the third is refused, by capacity and by name.
        assert!(matches!(
            tier.admit(SeqId(3), &[3]),
            Err(ForwardError::AtCapacity { live: 2, max: 2 })
        ));
        tier.release(S);
        assert_eq!(tier.live(), 1);
        assert!(tier.session(S).is_none());
        // The slot came back and the next admission gets it.
        tier.admit(SeqId(3), &[3]).expect("admitted");
        assert_eq!(tier.session(SeqId(3)).expect("a session").slot, first);
        // Idempotent.
        tier.release(S);
        // NEGATIVE CONTROL: a block outside the pool this tier was sized for.
        let mut small = DraftTier::plan(1, 8, 4096, 1).expect("a tier");
        let past = small.kv_blocks();
        let err = small
            .admit(SeqId(9), &[past])
            .expect_err("a block outside the pool");
        assert!(err.to_string().contains("outside a pool"), "{err}");
    }

    /// **The fork law, host side** (ws-f38): a branch's drafter session is its OWN — a context
    /// slot that is not the leader's and a run that shares no page — the copy covers exactly the
    /// pages the leader has written, and the branch starts where the leader is.
    ///
    /// CONTROLS, each a refusal by name that leaves the tier as it was: the branch REUSING the
    /// leader's context slot, a branch run that includes a leader page, a run of another length,
    /// a run that is not one piece, a leader with no session, and a branch forked twice.
    #[test]
    fn a_forked_branch_gets_its_own_slot_and_run_and_a_copy_of_what_the_leader_wrote() {
        let mut tier = DraftTier::plan(1, 8, 4096, 4).expect("a tier");
        let page = tier.kv_cache().block_size;
        let per = tier.kv_blocks() / 4;
        let leader_run: Vec<u32> = (0..per).collect();
        let branch_run: Vec<u32> = (per..2 * per).collect();
        tier.admit(S, &leader_run).expect("the leader");
        // A 234-token prompt and the first block of 8 written past it: 242 positions.
        let resident = 234 + 8;
        let copy = tier
            .fork(S, SeqId(2), &branch_run, resident)
            .expect("the fork");
        let lead = tier.session(S).expect("leader").clone();
        let br = tier.session(SeqId(2)).expect("branch").clone();
        assert_ne!(br.slot, lead.slot, "its own context slot");
        assert!(br.kv_blocks.iter().all(|p| !lead.kv_blocks.contains(p)));
        assert_eq!(br.done, lead.done);
        assert_eq!(
            copy,
            DraftForkCopy {
                from_first: 0,
                to_first: per,
                run: per,
                copy: resident.div_ceil(page),
            }
        );
        // A fork of a branch is not a thing, and a second fork of one branch is refused.
        let again = tier
            .fork(S, SeqId(2), &branch_run, resident)
            .expect_err("forked twice");
        assert!(again.to_string().contains("already holds"), "{again}");

        // CONTROL 1: the branch handed the LEADER'S context slot.
        let reuse = DraftSession {
            slot: lead.slot,
            kv_blocks: (2 * per..3 * per).collect(),
            done: 0,
            rope: None,
        };
        let err = tier
            .fork_plan(SeqId(3), &lead, &reuse, resident)
            .expect_err("the leader's slot");
        assert!(
            err.to_string()
                .contains("REUSES its leader's drafter context slot"),
            "{err}"
        );
        // CONTROL 2: a run that includes one of the leader's pages.
        let overlap = DraftSession {
            slot: 99,
            kv_blocks: (per - 1..2 * per - 1).collect(),
            done: 0,
            rope: None,
        };
        let err = tier
            .fork_plan(SeqId(3), &lead, &overlap, resident)
            .expect_err("a shared page");
        assert!(
            err.to_string().contains("include its leader's page"),
            "{err}"
        );
        // CONTROL 3: a run of another length, and one that is not one piece.
        let short = DraftSession {
            slot: 99,
            kv_blocks: (2 * per..3 * per - 1).collect(),
            done: 0,
            rope: None,
        };
        assert!(tier.fork_plan(SeqId(3), &lead, &short, resident).is_err());
        let mut gappy: Vec<u32> = (2 * per..3 * per).collect();
        gappy.swap(0, 1);
        let gappy = DraftSession {
            slot: 99,
            kv_blocks: gappy,
            done: 0,
            rope: None,
        };
        assert!(tier.fork_plan(SeqId(3), &lead, &gappy, resident).is_err());
        // CONTROL 4: a leader nobody admitted. The tier is exactly as it was after each refusal.
        let live = tier.live();
        assert!(tier
            .fork(
                SeqId(77),
                SeqId(3),
                &(2 * per..3 * per).collect::<Vec<_>>(),
                8
            )
            .is_err());
        assert_eq!(tier.live(), live);
        // The copy never reaches past the run, however long the leader's context is.
        let long = tier
            .fork(
                S,
                SeqId(3),
                &(2 * per..3 * per).collect::<Vec<_>>(),
                1 << 30,
            )
            .expect("a long leader");
        assert_eq!(long.copy, per);
        // Released like any session: its slot comes back.
        tier.release(SeqId(2));
        tier.release(SeqId(3));
        assert_eq!(tier.live(), 1);
    }

    /// Every COVERED launch of a proposal resolves against the committed rules.
    ///
    /// "Covered" and "resolvable" are two questions, and ws-f21 found five attention launches that
    /// answered yes to the first and no to the second — a `DflashShape` field the descriptor could
    /// not be built without. A tier that reported only the absent set would have called those five
    /// ready.
    #[test]
    fn no_covered_launch_of_a_proposal_is_refused_by_the_rules() {
        for block in [2u32, 4, 8] {
            let tier = DraftTier::plan(1, block, 4096, 8).expect("a tier");
            assert_eq!(
                tier.refusals(),
                Vec::<(String, String)>::new(),
                "block {block}: a covered op the rules refuse"
            );
        }
    }

    /// The plan's byte count IS the pack's, to the byte.
    ///
    /// `DraftDemand::weights` is derived from the record set's own shapes and nothing else — no
    /// pack is opened — so it can be wrong in a way no host test would notice. This pins it
    /// against the pack ws-f20/f23 actually built: `qwen38-dflash2-drafter-tp1/manifest.json`
    /// holds **95 records summing to 7,668,162,816 bytes** at `max_pos = 4096`, read on
    /// 2026-09-21. A plan that stopped agreeing with it would be a plan the engine could not size
    /// a tier from.
    ///
    /// It also pins the ROPE TABLE's context, which is the one number in that pack that is not a
    /// checkpoint tensor: `context_kv.rope.cos_sin` is `[4096, 128]` bf16, so a tier planned above
    /// `max_ctx = 4096` needs the pack REBUILT and not merely re-read. That is a property of the
    /// artefact, so the engine has to be able to state it before it boots one. **P27-9d**: it
    /// does -- a pack table shorter than `max_ctx` is REPLACED at load by the host-built one
    /// ([`DraftRope::cos_sin_bf16`], its first 4,096 rows asserted equal to the pack's), and any
    /// bound table shorter than `max_ctx` is refused by name ([`check_rope_rows`]). The plan's
    /// `weights` already billed the `max_ctx` table; before P27-9d those bytes were billed and
    /// never allocated.
    #[test]
    fn the_plans_weight_bytes_are_the_built_packs() {
        let tier = DraftTier::plan(1, 8, 4096, 8).expect("a tier");
        let recs = tier.graph().records(DraftRope::qwen38_dflash2(4096));
        assert_eq!(recs.len(), 95, "the pack's record count");
        assert_eq!(
            tier.demand().weights,
            7_668_162_816,
            "the TP=1 drafter pack's summed record lengths"
        );
        assert_eq!(tier.record_names(4096).len(), 95, "names are unique");
    }

    /// Every covered launch of a proposal is a DISPATCH, so nothing in the covered set forces a
    /// host round trip — including the walk, whose dependent argmax chain runs on the device.
    ///
    /// This is the fact that decides whether a proposal can join the step's captured graph
    /// (`docs/EXEC.md` § 4), and it is asserted rather than asserted-in-prose because the walk is
    /// the op a reader expects to be on the host.
    #[test]
    fn the_only_host_round_trips_a_proposal_needs_are_the_uncovered_ops() {
        let tier = DraftTier::plan(1, 8, 4096, 8).expect("a tier");
        let uncovered: u32 = tier.uncovered().iter().map(|(_, _, n)| n).sum();
        assert_eq!(tier.host_round_trips(), uncovered);
        // The walk is covered, and it is the sequential argmax chain.
        let walk = tier
            .graph()
            .ops
            .iter()
            .find(|o| o.name == "selector.walk")
            .expect("the walk is a declared op");
        assert!(
            matches!(walk.coverage, Coverage::Dflash(_)),
            "the walk is on the device"
        );
    }

    /// P27-6: a PAGED tier's runs follow the admission — length from the target reservation,
    /// granted and returned with the session, and a compaction rewrites the moved sessions' pages.
    ///
    /// NEGATIVE CONTROL: the dense tier refuses a grant, and a pool too small for the run is a
    /// refusal by name rather than a run that overlaps another session's.
    #[test]
    fn a_paged_tier_grants_runs_by_admission_and_compaction_moves_the_sessions() {
        // max_ctx 262,144 at the drafter's 16-token page: the dense run is 16,384 pages.
        let mut dense = DraftTier::plan(2, 8, 262_144, 8).expect("a tier");
        assert_eq!(dense.per_seq_pages(), 16_384);
        assert_eq!(dense.kv_blocks(), 8 * 16_384);
        assert!(!dense.is_paged());
        assert!(
            dense.grant_run(S, 4, true).is_err(),
            "a dense tier grants no runs"
        );
        let dense_kv = dense.demand().kv_pool;
        let stride = dense.demand().kv_block_stride;
        assert_eq!(
            dense_kv,
            8 * 16_384 * stride,
            "the dense pool is max_seqs x max_ctx"
        );

        let mut t = dense;
        t.page_by_admission(100);
        assert!(t.is_paged());
        assert_eq!(t.kv_blocks(), 100);
        assert_eq!(
            t.demand().kv_pool,
            100 * stride,
            "the demand follows the pool"
        );
        // A reservation of 10 target pages (160 tokens) + the drafter's block of 8 rows = 11.
        assert_eq!(t.run_pages_for(10), 11);
        // Capped at the dense run.
        assert_eq!(t.run_pages_for(16_385), 16_384);

        let a = t.grant_run(SeqId(1), 30, true).expect("a run");
        t.admit(SeqId(1), &(a.first..a.first + a.len).collect::<Vec<_>>())
            .expect("admitted");
        let b = t.grant_run(SeqId(2), 30, true).expect("a run");
        t.admit(SeqId(2), &(b.first..b.first + b.len).collect::<Vec<_>>())
            .expect("admitted");
        let c = t.grant_run(SeqId(3), 30, true).expect("a run");
        t.admit(SeqId(3), &(c.first..c.first + c.len).collect::<Vec<_>>())
            .expect("admitted");
        assert_eq!((a.first, b.first, c.first), (0, 30, 60));
        assert_eq!(t.paged_use(), Some((90, 100)));
        t.release(SeqId(2));
        assert_eq!(
            t.paged_use(),
            Some((60, 100)),
            "the run went back with the session"
        );
        // 40 pages free in two holes of 30 and 10: a 35-page run compacts.
        let d = t.grant_run(SeqId(4), 35, true).expect("compacted");
        assert_eq!(d.first, 60);
        assert_eq!(d.moves.len(), 1);
        assert_eq!(
            t.session(SeqId(3)).expect("moved").kv_blocks,
            (30..60).collect::<Vec<u32>>(),
            "the moved session's pages follow its run"
        );
        assert_eq!(t.session(SeqId(1)).expect("kept").kv_blocks[0], 0);
        // No room: refused by name.
        let e = t.grant_run(SeqId(5), 10, true).unwrap_err();
        assert!(format!("{e:?}").contains("no drafter page run"), "{e:?}");
    }

    /// P27-9d: the rope-table law and its launch-side twin, with their negative controls.
    #[test]
    fn p27_9d_a_rope_table_shorter_than_max_ctx_is_refused_by_name() {
        // The served pack's 4,096 rows at the served 262,144: refused, both numbers named.
        let e = check_rope_rows(4096, 262_144, "the pack's record").expect_err("refused");
        let msg = format!("{e:?}");
        assert!(msg.contains("context_kv.rope.cos_sin"), "{msg}");
        assert!(msg.contains("4096") && msg.contains("262144"), "{msg}");
        // Equal and longer tables pass (the engine gate's 4,096 at max_ctx 4,096; the built one).
        check_rope_rows(4096, 4096, "the pack's record").expect("equal");
        check_rope_rows(262_144, 262_144, "built at load").expect("built");
        // The launch check: every position < rows passes, the first one at rows fails.
        let ok: Vec<u8> = (0..4096i64).flat_map(i64::to_le_bytes).collect();
        check_rope_positions("ctx.positions", &ok, 4096).expect("in range");
        let at: Vec<u8> = 4096i64.to_le_bytes().to_vec();
        assert!(check_rope_positions("ctx.positions", &at, 4096).is_err());
        let past: Vec<u8> = [0i64, 12_096, 3]
            .iter()
            .flat_map(|p| p.to_le_bytes())
            .collect();
        let e = check_rope_positions("ctx.positions", &past, 4096).expect_err("past the table");
        assert!(format!("{e:?}").contains("12096"));
        check_rope_positions("ctx.positions", &past, 262_144).expect("a 262k table holds it");
        assert!(check_rope_positions("q.pos", &(-1i64).to_le_bytes(), 4096).is_err());
        check_rope_positions("q.pos", &[], 1).expect("an empty table reads nothing");
    }
}
