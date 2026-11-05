//! The DFlash2 drafter's **per-step index tables**, and the five target layers they read.
//!
//! [`crate::draft_tier`] answers what a proposal would launch and what it would cost.
//! [`crate::draft`] is the mailbox that carries a proposal to the next step's verify. This module
//! is the third thing a proposal inside the decode step needs and the only one nothing in this
//! tree had: **where every row of it goes.**
//!
//! # Why this is the piece that was missing
//!
//! `saber draft-step --proposal` replays prod's proposal against prod's own captured tensors, and
//! the index tables are among the tensors it is HANDED (`idx.query_slots`, `ctx.positions`,
//! `ctx.slots.l{0..4}`). They are computed by a Triton kernel — `prepare_dflash_inputs`, in
//! `upstream/v1/worker/gpu/spec_decode/dflash/speculator.py` — from the scheduler's own state, and the
//! capture's own probe says of them: *"a replay cannot invent them"*. That is exactly right, and it
//! is why the replay is a replay: the engine has no capture to be handed, so a proposal in the
//! decode step has to DERIVE them, and a derivation that is wrong writes the drafter's K/V to
//! in-range pages belonging to another position — silent, and rejected by the acceptance law at the
//! cost of the whole block (`docs/LEDGER.md`, workstream S1: the one failure the ids cannot see).
//!
//! So the law is written here, once, and held against **prod's own tables** for the three prompts
//! of the r26 proposal capture (`crates/saber-engine/tests/draft_inputs_gate.rs`) — an EXTERNAL
//! reference, on the host, with no card (`docs/GATES.md`, blind spot 1).
//!
//! # The law, and the one place the reference source is not what prod runs
//!
//! Per request, given the target step's query rows and how many of them it rejected:
//!
//! * the **context** rows are the target's own query rows. Row `j` goes at the sequence's absolute
//!   position `positions[j]`, in the DRAFTER's page at `block_table[pos / P] * P + pos % P`.
//! * only the **VALID** rows are addressed: `valid = rows - rejected`. The rows after a rejection
//!   carry hidden states the model produced for a token it then refused, and their slot is
//!   [`PAD_SLOT`], which is the K/V append's own early-out. The store still RUNS over all `rows` —
//!   prod hands it `hidden_states[:num_target_tokens]` — and the pad slot is what drops them.
//! * `last_valid_pos` is the last valid row's position, and the drafter's own `block` query rows
//!   follow it: positions `last_valid_pos + 1 ..= last_valid_pos + block`, row 0 fed the **anchor**
//!   (the token the step just committed) and rows `1..block` fed the **mask** token.
//! * the `block - 1` mask rows are the ones sampled, each at its own position.
//!
//! **The reference source in the container overlay masks the context on `j < num_ctx`; prod's own
//! tables say `j < num_ctx - num_rejected`.** All three captured prompts agree and none of them is
//! ambiguous: the target ran eight rows in every one (`idx.query_start_loc` is `[0, 8]`), and
//! `ctx.positions` carries **1, 4 and 6** written entries with the rest of the buffer stale (`0`)
//! and `ctx.slots` at `-1`. Read with `q.positions` — whose first entry is `last_valid_pos + 1` —
//! the three say `valid` is 1, 4 and 6, i.e. the step accepted 0, 3 and 5 of its seven draft
//! tokens. A law that wrote all eight would put five rows of post-rejection K/V into the drafter's
//! cache at positions the sequence never committed. The captured tables are the authority here and
//! the overlay is not prod's image.
//!
//! # What this module does not do
//!
//! It does not launch anything. The proposal's 91 covered launches are still `saber-cli`'s
//! (`draft_proposal.rs`), eager and seeded; this is the half that a chain in the engine cannot be
//! seeded past. [`AUX_TAPS`] names the five target edges the proposal's first op reads, for the
//! same reason: the concatenation is the one input to a proposal that no pack can produce and only
//! the target's own step has.

use saber_serve::engine::TokenId;

use crate::forward::ForwardError;

/// The slot id that means "do not write this row's K/V".
///
/// The reference server's `PAD_SLOT_ID`, and both fp8 K/V appends' own early-out. Read off prod's captured tables
/// rather than restated from a header: `ctx.slots.l{0..4}` is `-1` in every entry past `valid` on
/// all three prompts of the r26 capture.
pub const PAD_SLOT: i64 = -1;

/// The target layers whose residual stream the drafter's aux projection reads, as the drafter's own
/// checkpoint names them (`dflash_config.target_layer_ids`).
///
/// The reference server collects an aux hidden state as `hidden_states + residual` **after** layer `id - 1`
/// (its aux-hidden-state hook, called with `layer_idx + 1`), which is exactly the
/// value `layer.{id}.input_layernorm` adds and publishes as `l{id}.resid.a` — the same edge
/// `saber engine`'s per-layer probe digests and `saber forward --dump-layers` prints. So the tap
/// points need no new law and no new name; see [`aux_tap_edges`].
///
/// Held against `/mnt/nvme/models/Qwen3.8-27B-DFlash2-FP8/config.json` by the host gate, because a
/// tap list written down twice is a drafter reading five layers the checkpoint was not trained on —
/// which produces a plausible projection and an acceptance of zero.
pub const AUX_TAPS: [u32; 5] = [5, 19, 33, 47, 61];

/// The id the drafter feeds every query row but the first: `dflash_config.mask_token_id`.
///
/// It is prod's `parallel_drafting_token_id`, and it is a real row of the drafter's embedding
/// table, not a sentinel — a wrong value here is a proposal from the wrong prompt.
pub const MASK_TOKEN: TokenId = 248_070;

/// The arena edge each of [`AUX_TAPS`] is carried by.
#[must_use]
pub fn aux_tap_edges() -> Vec<String> {
    AUX_TAPS.iter().map(|l| format!("l{l}.resid.a")).collect()
}

/// One request's half of a step, as the drafter's index tables see it.
///
/// Every field is a fact the engine already holds when `decode_step` has committed: the session's
/// `done` before the step, the rows it ran, what the acceptance law committed, and the drafter's own
/// pages. Nothing here is a plan fact, which is why this type takes them rather than a plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftSeqStep {
    /// Absolute positions of the target's query rows this step, in row order.
    ///
    /// A `Vec` rather than `(first, len)` because the law reads `positions[valid - 1]` and a
    /// chunked prefill's rows are not this step's `done_before + i`; the engine's decode path
    /// supplies the contiguous run and [`DraftSeqStep::decode`] builds it.
    pub positions: Vec<u32>,
    /// Rows of the tail the step did NOT commit (`num_rejected`). `rows - accepted - 1`.
    pub rejected: u32,
    /// The token the drafter's row 0 is fed: the last token this step committed.
    pub anchor: TokenId,
    /// The drafter's KV pages for this sequence, in order, indexed by `position / page`.
    pub kv_blocks: Vec<u32>,
    /// This request's slot in whatever persistent per-request state the walk writes
    /// (`sample_idx_mapping`). The engine's drafter context slot.
    pub state_idx: i32,
    /// **A PADDING request: rows the shape has and the step does not** (ws-s10).
    ///
    /// A batched proposal runs at a planned shape and a step may have fewer live sequences than
    /// the shape it is padded up to — the int2 head refuses a partial tile, so a batch is padded
    /// UP rather than trimmed down. A padding request's rows are computed and stored NOWHERE: its
    /// context and query slots are [`PAD_SLOT`], which is the K/V append's own early-out; its
    /// `seq_lens` entry is 0, which makes the attention's tile loop empty; and its
    /// `sample_idx_mapping` is −1, which is `dflash_selector_walk_greedy`'s own `valid` gate, so
    /// the walk writes neither a token nor a realized score for it.
    ///
    /// It still carries a LIVE request's positions and pages, because every table it appears in
    /// has to hold a legal value: a block-table row of zeros points at page 0, which belongs to
    /// another session (`docs/EXEC.md` § 2, the target's own law).
    pub pad: bool,
    /// **Where the drafter's ROPE puts each row** (D4). `None`: the row's index, as for every
    /// text-only sequence. `Some`: the sequence's M-RoPE table, of which the drafter reads the
    /// TEMPORAL axis -- prod's `copy_and_expand_dflash_inputs_kernel` is handed the target's
    /// `[3, N]` `mrope_positions` and loads `target_positions_ptr + i`, i.e. row 0 -- so a prompt
    /// row is at its temporal position and every row past the prompt at `index + delta`. The K/V
    /// SLOT stays the index, and so does `seq_lens`. **D6 correction:** that kernel is the reference server's V1
    /// runner; prod runs Model Runner V2, whose DFlash speculator builds rope, slot and `seq_lens`
    /// all from the token INDEX (`docs/VISION.md` § 11). `--drafter-rope-index` hands no table here,
    /// which is prod's law; `Some` (D4's temporal rope) stays the default until the user decides.
    pub rope: Option<std::sync::Arc<saber_exec::mrope::MropePositions>>,
}

/// The drafter's rope position of the token at `index` (D4): the temporal M-RoPE axis, or the
/// index itself for a text-only sequence.
#[must_use]
pub fn draft_rope_pos(rope: Option<&saber_exec::mrope::MropePositions>, index: u32) -> i64 {
    match rope {
        Some(r) => r.at(index)[0],
        None => i64::from(index),
    }
}

impl DraftSeqStep {
    /// The same step, its rope placed by `rope` (D4). `None` leaves it the index.
    #[must_use]
    pub fn with_rope(
        mut self,
        rope: Option<std::sync::Arc<saber_exec::mrope::MropePositions>>,
    ) -> Self {
        self.rope = rope;
        self
    }

    /// The decode case: `rows` contiguous positions from `done_before`, of which the step committed
    /// `committed`.
    ///
    /// # Errors
    /// A step that committed nothing, or committed more rows than it ran — both are a verdict the
    /// acceptance law cannot have produced (`committed == accepted + 1`, always).
    pub fn decode(
        done_before: u32,
        rows: u32,
        committed: u32,
        anchor: TokenId,
        kv_blocks: &[u32],
        state_idx: i32,
    ) -> Result<Self, ForwardError> {
        if committed == 0 || committed > rows {
            return Err(ForwardError::Device(format!(
                "a verify step of {rows} row(s) that committed {committed}: the acceptance law \
                 commits the accepted prefix plus exactly one, so a step commits 1..={rows}"
            )));
        }
        Ok(DraftSeqStep {
            positions: (done_before..done_before + rows).collect(),
            rejected: rows - committed,
            anchor,
            kv_blocks: kv_blocks.to_vec(),
            state_idx,
            pad: false,
            rope: None,
        })
    }

    /// A PADDING request shaped like `like`: the same row count and the same pages, storing
    /// nothing and proposing nothing.
    ///
    /// It is built FROM a live request rather than from zeros for the reason in the field's own
    /// doc — every table it appears in must carry a legal value — and it is the live request's
    /// tables that are known to be legal.
    #[must_use]
    pub fn padding(like: &DraftSeqStep) -> Self {
        DraftSeqStep {
            positions: like.positions.clone(),
            rejected: 0,
            anchor: like.anchor,
            kv_blocks: like.kv_blocks.clone(),
            state_idx: -1,
            pad: true,
            rope: like.rope.clone(),
        }
    }

    /// Rows the target ran for this request this step.
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.positions.len() as u32
    }

    /// Rows whose hidden state the drafter may store: the ones before the first rejection.
    #[must_use]
    pub fn valid(&self) -> u32 {
        self.rows().saturating_sub(self.rejected)
    }
}

/// The fixed half: what the drafter is, rather than what this step is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftGeometry {
    /// `1 + num_speculative_tokens` — query rows per proposal. Prod's 8.
    pub block: u32,
    /// Tokens of the DRAFTER's KV cache one page holds.
    pub page: u32,
    /// Requests the per-request tables are padded out to, so a captured replay never reads past
    /// the batch it was captured at.
    pub max_reqs: u32,
    /// Positions a query row may be clamped to. `max_model_len`.
    pub max_pos: u32,
    /// The id every query row but the first is fed.
    pub mask_token: TokenId,
}

impl DraftGeometry {
    /// Positions the selector proposes for: `block - 1`.
    #[must_use]
    pub fn steps(&self) -> u32 {
        self.block.saturating_sub(1)
    }
}

/// Every table one proposal binds, in the engine's own units.
///
/// The widths are prod's: `i64` for anything a Triton kernel indexes a cache with, `i32` for the
/// two the attention metadata carries. That is not cosmetic — ws-f24 lost three rounds to three
/// kernarg ELEMENT WIDTHS, and `crates/saber-kernels/tests/capture_widths.rs` is the rule that
/// came out of it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DraftStepIndex {
    /// One per target row, in row order: the sequence position that row's hidden state belongs to.
    /// `0` where the row is past the acceptance — the value is never read, because the slot is
    /// [`PAD_SLOT`].
    pub context_positions: Vec<i64>,
    /// One per target row: where that row's context K/V lands, or [`PAD_SLOT`].
    pub context_slots: Vec<i64>,
    /// `max_reqs * block` ids: the anchor at each request's row 0, the mask token elsewhere.
    pub input_ids: Vec<i32>,
    /// `max_reqs * block` positions for the drafter's own query rows.
    pub query_positions: Vec<i64>,
    /// `max_reqs * block` slots, [`PAD_SLOT`] past the live requests.
    pub query_slots: Vec<i64>,
    /// `max_reqs + 1` entries: request `i`'s query rows start at `i * block`.
    pub query_start_loc: Vec<i32>,
    /// `max_reqs` entries: the absolute length the drafter's attention reads up to.
    pub seq_lens: Vec<i32>,
    /// `max_reqs * steps`: which query row each proposal position samples from.
    pub sample_indices: Vec<i64>,
    /// `max_reqs * steps`: the absolute position each proposal is for.
    pub sample_pos: Vec<i64>,
    /// `max_reqs * steps`: the per-request state slot, `-1` for padding.
    pub sample_idx_mapping: Vec<i32>,
}

impl DraftStepIndex {
    /// Derive every table from the step's requests.
    ///
    /// The order of the outputs is prod's: context rows are laid out at the TARGET's row offsets
    /// (request `i` starts where its target rows start, which is the running sum), and query rows
    /// at `i * block` whatever the target did — because the drafter's forward is exactly `block`
    /// rows per request and prod pads to that for the captured graph's sake.
    ///
    /// # Errors
    /// More requests than the tables are padded for; a request whose position run is empty; a
    /// position with no page in that request's block table. Each is a table that would be written
    /// somewhere in range and read as another sequence's.
    pub fn build(reqs: &[DraftSeqStep], geom: DraftGeometry) -> Result<Self, ForwardError> {
        if geom.block < 2 {
            return Err(ForwardError::Device(format!(
                "a DFlash2 block of {}: a block of 1 proposes nothing",
                geom.block
            )));
        }
        if geom.page == 0 {
            return Err(ForwardError::Device(
                "a drafter KV page of 0 tokens".to_owned(),
            ));
        }
        if reqs.len() as u32 > geom.max_reqs {
            return Err(ForwardError::Device(format!(
                "{} request(s) in a step whose drafter tables are padded for {}: the padding is \
                 what keeps a replayed graph from reading past the batch it was captured at",
                reqs.len(),
                geom.max_reqs
            )));
        }
        let block = geom.block as usize;
        let steps = geom.steps() as usize;
        let max_reqs = geom.max_reqs as usize;
        let ctx_rows: usize = reqs.iter().map(|r| r.positions.len()).sum();

        let mut out = DraftStepIndex {
            context_positions: vec![0; ctx_rows],
            context_slots: vec![PAD_SLOT; ctx_rows],
            input_ids: vec![geom.mask_token as i32; max_reqs * block],
            query_positions: vec![0; max_reqs * block],
            query_slots: vec![PAD_SLOT; max_reqs * block],
            query_start_loc: vec![0; max_reqs + 1],
            seq_lens: vec![0; max_reqs],
            sample_indices: vec![0; max_reqs * steps],
            sample_pos: vec![0; max_reqs * steps],
            sample_idx_mapping: vec![-1; max_reqs * steps],
        };
        // Every request's query rows start at `i * block`, live or not: prod pads the whole table
        // to `max_reqs` with the last live end, so a replay at a wider batch reads a legal start.
        for (i, e) in out.query_start_loc.iter_mut().enumerate() {
            *e = (i.min(reqs.len()) * block) as i32;
        }

        let mut ctx_at = 0usize;
        for (i, r) in reqs.iter().enumerate() {
            let rows = r.positions.len();
            if rows == 0 {
                return Err(ForwardError::Device(format!(
                    "request {i} ran no rows: a step with no query row has no hidden state for the \
                     drafter to store and no token to anchor its block"
                )));
            }
            let valid = r.valid() as usize;
            if valid == 0 {
                return Err(ForwardError::Device(format!(
                    "request {i} rejected all {rows} of its rows: the acceptance law commits the \
                     accepted prefix plus exactly one, so at least row 0 is always valid"
                )));
            }
            let slot_of = |pos: u32| -> Result<i64, ForwardError> {
                // A PADDING request writes nowhere, and that is the whole of what makes a batched
                // proposal at a shape wider than the step safe: the K/V appends take their own
                // early-out on this slot, so the rows are computed and discarded.
                if r.pad {
                    return Ok(PAD_SLOT);
                }
                // AT THE TOP OF A CONTEXT, prod's own law: PAD_SLOT (ws-t2, S11's reading).
                //
                // A block's query rows run `last_valid_pos + 1 .. + block` whatever the context
                // ceiling is, so the last block of a session that has reached `max_model_len`
                // asks for positions the session may not hold. Prod clamps the BLOCK INDEX at the
                // block table's stride and maps the null block to `PAD_SLOT_ID`
                // (`patch_dflash_base.py`: `ctx_block_num = tl.minimum(ctx_block_num,
                // block_table_stride - 1)` and `ctx_resident = … & (ctx_block_id != 0)`); the
                // rows are computed out of the running softmax's initial state and write nothing.
                //
                // **The clamp is expressed on the POSITION here and not on the page index**, and
                // that is a deliberate deviation with a reason: Saber's drafter block table is
                // padded with the sequence's own last page (`block_table`, the target's law),
                // where prod's is zero-padded, so clamping the index would point an overrun row's
                // K/V write at a real position's slot instead of at a null block. Same effect as
                // prod's — nothing is written past the context — reached through the one table
                // this tree pads by its own rule. `query_positions` is already clamped to
                // `max_pos - 1` for the same rows, and neither number is read without the other.
                if geom.max_pos > 0 && pos >= geom.max_pos {
                    return Ok(PAD_SLOT);
                }
                let page = (pos / geom.page) as usize;
                // BELOW the ceiling this stays a refusal, because there it is a DEFECT and not a
                // context that ran out: a page run too short for the context the geometry says
                // this session may reach would otherwise turn into silent PAD_SLOTs and a
                // proposal drafted out of an empty cache.
                let blk = r.kv_blocks.get(page).ok_or_else(|| {
                    ForwardError::Device(format!(
                        "request {i}: position {pos} needs drafter page {page} and this session \
                         holds {} (max_pos {}): a clamped page id is another session's K/V, in \
                         range and silent",
                        r.kv_blocks.len(),
                        geom.max_pos
                    ))
                })?;
                Ok(i64::from(*blk) * i64::from(geom.page) + i64::from(pos % geom.page))
            };

            // --- the context: the target's own rows, valid ones only ---------------------------
            let rope = r.rope.as_deref();
            let top = i64::from(geom.max_pos.saturating_sub(1));
            for j in 0..valid {
                let pos = r.positions[j];
                out.context_positions[ctx_at + j] = draft_rope_pos(rope, pos);
                out.context_slots[ctx_at + j] = slot_of(pos)?;
            }
            // Rows `valid..rows` keep position 0 and PAD_SLOT: the store runs over them and writes
            // nothing, which is prod's own arrangement rather than a shortening of the call.
            ctx_at += rows;

            // --- the drafter's own query rows --------------------------------------------------
            let last_valid_pos = r.positions[valid - 1];
            let base = i * block;
            for off in 0..block {
                let pos = last_valid_pos + 1 + off as u32;
                out.input_ids[base + off] = if off == 0 {
                    r.anchor as i32
                } else {
                    geom.mask_token as i32
                };
                // Prod: `query_pos = last_pos + 1 + query_off`, `last_pos` the last valid
                // context row's (rope) position. Text-only: the index, clamped as before.
                out.query_positions[base + off] = match rope {
                    None => i64::from(pos.min(geom.max_pos.saturating_sub(1))),
                    Some(_) => {
                        (draft_rope_pos(rope, last_valid_pos) + 1 + off as i64).min(top.max(0))
                    }
                };
                out.query_slots[base + off] = slot_of(pos)?;
                // The mask rows are the proposals: one per speculative step, each AT its own
                // position. Row 0 is the anchor and is never sampled from — that is the DFlash
                // default, and `sample_from_anchor` (DSpark) is the other model, not this one.
                if off >= 1 {
                    let s = i * steps + (off - 1);
                    out.sample_indices[s] = (base + off) as i64;
                    out.sample_pos[s] = i64::from(pos);
                    out.sample_idx_mapping[s] = r.state_idx;
                }
            }
            // 0 FOR A PADDING REQUEST, and it is not a sentinel: `dflash_attn` computes
            // `num_tiles = (seq_len + TILE - 1) / TILE`, so a length of 0 gives an empty tile
            // loop and the rows come out of the running softmax's initial state. Nothing is read
            // out of the block table, which is why a padding request never reaches another
            // session's pages even though it carries a live one's row.
            //
            // AND CLAMPED AT `max_model_len` FOR A LIVE ONE, which is prod's own law:
            // `tl.minimum(last_valid_pos + 1 + num_query_per_req, max_model_len)`
            // (`patch_dflash_base.py`, "dflash: clamp seq_lens"). Without it a session at the top
            // of its context tells the attention to read `block` positions past the end of the
            // pages it holds — tiles whose block-table entry is this sequence's own last page
            // repeated, i.e. its own K/V read at the wrong positions. It binds only in the last
            // `block` positions of a context, which is why this went unnoticed at `max_ctx` 4096
            // with 512-token generations (ws-s11 read it; ws-t2 patched it).
            out.seq_lens[i] = if r.pad {
                0
            } else {
                let want = last_valid_pos + 1 + geom.block;
                if geom.max_pos > 0 {
                    want.min(geom.max_pos) as i32
                } else {
                    want as i32
                }
            };
        }
        Ok(out)
    }

    /// **Pad the context tables out to `ctx_rows`** (C2): a ROW BAND's requests carry their own
    /// row counts, so their context rows are packed by the target's `cu` and fall short of the
    /// shape's `reqs x ctx_per_req` window. The rows past them get position 0 and `PAD_SLOT` —
    /// the K/V append's own early-out, the arrangement a rejected row already has — so the store
    /// runs over them and writes nothing.
    ///
    /// # Errors
    /// A table already longer than the window: the requests hold more rows than the shape.
    pub fn pad_context_to(&mut self, ctx_rows: u32) -> Result<(), ForwardError> {
        let n = ctx_rows as usize;
        if self.context_positions.len() > n || self.context_slots.len() > n {
            return Err(ForwardError::Device(format!(
                "{} context row(s) in a proposal window of {n}",
                self.context_positions.len()
            )));
        }
        self.context_positions.resize(n, 0);
        self.context_slots.resize(n, PAD_SLOT);
        Ok(())
    }

    /// The `[num_reqs, max_blocks]` drafter block table, padded the way the TARGET's is: each
    /// sequence's own pages, and **its own last page** where it runs out.
    ///
    /// Padding with 0 points a stray read at page 0, which belongs to another session
    /// (`docs/EXEC.md` § 2, the same law as the target's `block_table`). **This is a deliberate
    /// deviation from prod**, whose drafter table is zero-padded (`[64, 0, 0, 0, 0, 0, 0, 0]` in
    /// every captured prompt) — safely, because prod's drafter page holds 1648 tokens and no
    /// captured sequence reaches a second one. Saber's drafter page is the reference's 16, so the pages
    /// past a sequence's own are reachable here in a way they are not there, and the captures
    /// cannot say which padding prod would have used. The target's law is the one this tree
    /// already states in one place, so it is the one used.
    ///
    /// # Errors
    /// A sequence with no pages at all.
    pub fn block_table(reqs: &[DraftSeqStep], max_blocks: u32) -> Result<Vec<i32>, ForwardError> {
        let mut out = vec![0i32; reqs.len() * max_blocks as usize];
        for (i, r) in reqs.iter().enumerate() {
            let last = *r.kv_blocks.last().ok_or_else(|| {
                ForwardError::Device(format!("request {i}: no drafter KV pages at all"))
            })?;
            for b in 0..max_blocks as usize {
                out[i * max_blocks as usize + b] =
                    r.kv_blocks.get(b).copied().unwrap_or(last) as i32;
            }
        }
        Ok(out)
    }
}

// =====================================================================================
// THE PROMPT'S CONTEXT: the drafter's context store on a PREFILL chunk (ws-s8)
// =====================================================================================

/// One TILE of a prefill chunk's context store: `block` rows of the target's own prompt, at their
/// absolute positions, addressed into this session's drafter pages.
///
/// # Why this type exists at all
///
/// **Prod stores the drafter's context K/V at EVERY target step, prefill chunks included.**
/// `GPUModelRunner.execute_model` calls `speculator.propose(...)` unconditionally after the
/// sampler, and `DFlashSpeculator.propose` runs `precompute_and_store_context_kv` over
/// `hidden_states[:num_target_tokens]` before it runs the drafter's backbone — so on a prefill
/// step the drafter's cache takes one row per PROMPT token (`num_sampled == 0` there, which
/// changes only which token the anchor row is fed: `next_prefill_tokens` rather than
/// `last_sampled`). By the first decode step prod's drafter has the whole prompt in its cache.
///
/// Saber's engine proposed on DECODE steps only, so its drafter's window
/// (`seq_lens = last_valid_pos + 1 + block`, the same law at both ends) covered a prompt's worth
/// of pages that nothing had ever written — zeros, since `DraftResident::clear_pages` zeroes a
/// session's run at admission. At the corpus's median 274-token prompt that is **80% of the
/// drafter's attention window reading zeros**, and no capture of ONE step could see it: the
/// standalone chain and `--draft-replay` both seed the cache from prod's own `kv.before_ctx`,
/// which already holds the prompt.
///
/// A tile is exactly one STORE SHAPE's rows — `block`, or the wide store's (H35, [`StoreTiling`])
/// — because the store's descriptors are resolved at that `ctx_rows` and their grids are baked; a
/// tile with fewer LIVE rows carries [`PAD_SLOT`] past them, which is the K/V append's own
/// early-out — the same arrangement the rows after a rejection use. Every other op of the store
/// is per row, so a pad row computes on whatever `aux.cat` holds there and lands nowhere.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DraftCtxTile {
    /// `block` absolute positions; `0` where the row is not live (never read — the slot pads).
    pub positions: Vec<i64>,
    /// `block` slots, [`PAD_SLOT`] past the live rows.
    pub slots: Vec<i64>,
    /// This tile's first row in `aux.cat`.
    pub row0: u32,
    /// Live rows in this tile, `1..=block`.
    pub live: u32,
}

/// What a prompt-context frame on the `engine.draft` stream says (ws-s8, widened by H35).
///
/// The header's third word. It was a `more` flag (0 = the terminator, 1 = a tile); the wide store
/// adds the STORE CHECK's two markers, which the leader sends only under `--draft-store-check`
/// and which make the follower do exactly what the leader does at the same point of the stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CtxFrame {
    /// No tile follows.
    End,
    /// A tile to store, at its width's store shape.
    Tile,
    /// A `block`-row tile stored at the CHUNK's norm class (H35): a tile of a chunk of at least
    /// [`StoreTiling::class_rows`] rows, whose hidden norm prod launches with the wide-row block.
    TileChunkClass,
    /// STORE CHECK, first marker: read back the rows every tile since the chunk began stored,
    /// then POISON them — so the second store has to rewrite every byte the check compares.
    CheckSnap,
    /// STORE CHECK, second marker: read the same rows back again and compare them to the first
    /// read, byte for byte — THE GATE (the re-store is class-matched: identical, or it fails).
    CheckCompare,
    /// STORE CHECK, third marker: poison the same rows again, keeping the first read.
    CheckPoison,
    /// STORE CHECK, fourth marker: read them back after a re-store at MAIN's class and compare —
    /// the class residual, reported; the control that the comparison sees a one-ULP norm change.
    CheckResidual,
    /// A WIDE tile of a chunk BELOW [`StoreTiling::class_rows`], stored at the chunk's own
    /// (narrow-row) norm class (C6): the rows prod stores in one call at the chunk's M, which is
    /// below the boundary, so its hidden norm takes the wide block — the class every one of main's
    /// 8-row tiles of that chunk already had. One tile of twelve launches instead of one per
    /// `block` rows.
    TileWideNatural,
}

impl CtxFrame {
    fn code(self) -> u32 {
        match self {
            CtxFrame::End => 0,
            CtxFrame::Tile => 1,
            CtxFrame::CheckSnap => 2,
            CtxFrame::CheckCompare => 3,
            CtxFrame::TileChunkClass => 4,
            CtxFrame::CheckPoison => 5,
            CtxFrame::CheckResidual => 6,
            CtxFrame::TileWideNatural => 7,
        }
    }

    fn from_code(c: u32) -> Option<Self> {
        match c {
            0 => Some(CtxFrame::End),
            1 => Some(CtxFrame::Tile),
            2 => Some(CtxFrame::CheckSnap),
            3 => Some(CtxFrame::CheckCompare),
            4 => Some(CtxFrame::TileChunkClass),
            5 => Some(CtxFrame::CheckPoison),
            6 => Some(CtxFrame::CheckResidual),
            7 => Some(CtxFrame::TileWideNatural),
            _ => None,
        }
    }
}

/// **How a prefill chunk's context store is TILED** (H35).
///
/// ws-s8 stored every chunk as `block`-row tiles because the store's descriptors were resolved at
/// `ctx_rows = block`: a 2048-row chunk was 256 tiles of twelve eager launches at M = 8, each one
/// re-reading the drafter's projection weights for eight rows — 149 to 156 ms a chunk, 28 % of a
/// served one (H34). The wide store is a SECOND store shape, `wide` rows tall, over the same twelve
/// ops: `gemm_a8w8_blockscale_mt` (to `max_m` 256), `bgemm_bf16_m32` (to `max_m` 256) and five
/// per-row ops, which F33/F34 gated bit-identical per 16-row slice to the symbols the narrow store
/// runs. A chunk is `wide`-row tiles while whole ones fit, then the remainder.
///
/// `wide == 0` is main's law — every tile `block` rows — and is the pairing control.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoreTiling {
    /// Rows of the wide tile, a multiple of `block`; 0 for no wide shape.
    pub wide: u32,
    /// The REMAINDER, in rows, from which ONE partial wide tile replaces its narrow tiles.
    ///
    /// A wide tile costs the same whether 8 or 256 of its rows are live (its grids are baked), so
    /// below some remainder the narrow tiles are cheaper. It is data, not a branch (`CLAUDE.md`
    /// rule 6): 0 disables the partial tile, and the remainder is always narrow.
    pub min_partial: u32,
    /// Rows `aux.cat` holds. A wide tile reads `wide` rows of it from `row0` whatever its live
    /// count, so a partial one is taken only where that window is inside the buffer.
    pub aux_rows: u32,
    /// **The norm CLASS boundary**, in chunk rows: `[ctx] narrow_rows` (256). `ctx_rms_norm`'s
    /// block is a function of the launch's rows — prod's launcher narrows it from 640 to 256
    /// threads at 256 rows, and at hidden 5120 that changes how many vectors one thread chains
    /// into its accumulator, i.e. the association of the norm's sum (the one op of the store
    /// whose arithmetic moves with M). Prod stores a chunk in ONE call at the chunk's rows, so a
    /// chunk of at least this many rows is prod's wide-row class for every row; a wide tile (256
    /// rows) already is, and the chunk's narrow tiles are stored at it too
    /// ([`CtxFrame::TileChunkClass`]). A chunk below it is stored at its natural class — prod's
    /// for it — narrow, or (C6) as wide tiles of that class ([`StoreTiling::natural_min`]).
    /// 0: no class rule (every narrow tile natural).
    pub class_rows: u32,
    /// **C6: the rows from which a chunk BELOW `class_rows` is stored as wide tiles at its own
    /// narrow-row class** ([`CtxFrame::TileWideNatural`]) instead of `block`-row tiles: every 8-row
    /// tile re-reads the drafter's projection weights (~630 µs a tile, C5: 11.5 tiles and 7.3 ms a
    /// chunk), and a wide tile's cost does not depend on its live rows. 0 = off (main's law for
    /// such a chunk). Data, not a branch: the value is the measured crossover.
    pub natural_min: u32,
}

impl StoreTiling {
    /// Main's law: every tile is `block` rows, at its natural class.
    #[must_use]
    pub fn narrow() -> Self {
        StoreTiling {
            wide: 0,
            min_partial: 0,
            aux_rows: 0,
            class_rows: 0,
            natural_min: 0,
        }
    }
}

/// One chunk's context store, in the order the engine runs it (H35).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StoreSchedule {
    /// The tiles the chunk's store loop runs, in row order: wide tiles first, then the remainder.
    pub stored: Vec<DraftCtxTile>,
    /// The chunk's LAST `block`-row tile, held back for the first proposal (ws-s9, F37): the
    /// proposal's eager phase IS the context store, at the one-request store shape, so these rows
    /// are its context rows. Exactly the tile main's schedule ends with — the wide store changes
    /// how the rows BEFORE it are tiled and never which rows the proposal owns.
    pub held: Option<DraftCtxTile>,
    /// Whether this schedule's `block`-row tiles are stored at the CHUNK's norm class
    /// ([`CtxFrame::TileChunkClass`]) rather than their natural one: true for a chunk of at least
    /// [`StoreTiling::class_rows`] rows under the wide store. The held tile is not in `stored` and
    /// runs in the first proposal's eager phase at its natural class (as on main).
    pub chunk_class: bool,
    /// C6: this schedule's WIDE tiles are a chunk below [`StoreTiling::class_rows`] stored at its
    /// own narrow-row class ([`CtxFrame::TileWideNatural`]), not the wide-row class.
    pub natural_wide: bool,
}

impl StoreSchedule {
    /// The frame kind a stored tile of this schedule travels as.
    #[must_use]
    pub fn kind_of(&self, t: &DraftCtxTile, block: u32) -> CtxFrame {
        if t.width() == block {
            if self.chunk_class {
                CtxFrame::TileChunkClass
            } else {
                CtxFrame::Tile
            }
        } else if self.natural_wide {
            CtxFrame::TileWideNatural
        } else {
            CtxFrame::Tile
        }
    }

    /// Tiles the loop stores at a width other than `block`.
    #[must_use]
    pub fn wide_tiles(&self, block: u32) -> usize {
        self.stored.iter().filter(|t| t.width() != block).count()
    }

    /// Eager launches the chunk's store loop issues at `per_tile` launches a tile.
    #[must_use]
    pub fn launches(&self, per_tile: u32) -> u64 {
        self.stored.len() as u64 * u64::from(per_tile)
    }

    /// Live rows the loop stores.
    #[must_use]
    pub fn stored_rows(&self) -> u32 {
        self.stored.iter().map(|t| t.live).sum()
    }
}

impl DraftCtxTile {
    /// The tiles one chunk of `rows` target rows, whose first row is at absolute position
    /// `first_pos` and at `aux.cat` row `aux0`, stores — main's law, every tile `block` rows.
    ///
    /// # Errors
    /// A geometry a store cannot run (`block < 2`, `page == 0`), or a position with no drafter
    /// page — the same refusal [`DraftStepIndex::build`] makes, and for the same reason: a clamped
    /// page id is another session's K/V, in range and silent.
    pub fn chunk(
        first_pos: u32,
        rows: u32,
        aux0: u32,
        kv_blocks: &[u32],
        geom: DraftGeometry,
    ) -> Result<Vec<Self>, ForwardError> {
        Self::chunk_rope(first_pos, rows, aux0, kv_blocks, geom, None)
    }

    /// [`DraftCtxTile::chunk`] with the rope placed by a sequence's M-RoPE table (D4): each row's
    /// position is the table's TEMPORAL axis ([`draft_rope_pos`]); the slot stays the index.
    ///
    /// # Errors
    /// [`DraftCtxTile::chunk`]'s.
    pub fn chunk_rope(
        first_pos: u32,
        rows: u32,
        aux0: u32,
        kv_blocks: &[u32],
        geom: DraftGeometry,
        rope: Option<&saber_exec::mrope::MropePositions>,
    ) -> Result<Vec<Self>, ForwardError> {
        let s = Self::schedule(
            first_pos,
            rows,
            aux0,
            kv_blocks,
            geom,
            rope,
            false,
            StoreTiling::narrow(),
        )?;
        Ok(s.stored)
    }

    /// **The chunk's store, tiled** (H35): `wide`-row tiles while whole ones fit, the remainder
    /// as ONE partial wide tile when it is at least `min_partial` rows and its window fits
    /// `aux.cat`, `block`-row tiles otherwise — and, when `hold_last`, the chunk's last
    /// `block`-row tile held back for the first proposal, exactly as main's schedule holds it.
    ///
    /// Every row is stored once, at the slot and position main's schedule gives it: a row's
    /// entries are a function of its position alone, so which tile carries it changes nothing
    /// the kernels read about it. The narrow tiles of the remainder start at a multiple of
    /// `block` from the chunk's first row, so they are main's own tiles over those rows.
    ///
    /// # Errors
    /// [`DraftCtxTile::chunk`]'s, and a wide width that is not a whole number of blocks above one.
    #[allow(clippy::too_many_arguments)]
    pub fn schedule(
        first_pos: u32,
        rows: u32,
        aux0: u32,
        kv_blocks: &[u32],
        geom: DraftGeometry,
        rope: Option<&saber_exec::mrope::MropePositions>,
        hold_last: bool,
        tiling: StoreTiling,
    ) -> Result<StoreSchedule, ForwardError> {
        Self::check_geom(rows, geom)?;
        let block = geom.block;
        if tiling.wide != 0 && (tiling.wide % block != 0 || tiling.wide <= block) {
            return Err(ForwardError::Device(format!(
                "a wide context store of {} rows at a block of {block}: the wide tile is a whole \
                 number of blocks above one, or the narrow tiles of its remainder would not be \
                 main's own",
                tiling.wide
            )));
        }
        let at_ = |at: u32, live: u32, width: u32| {
            Self::tile(first_pos, at, live, width, aux0, kv_blocks, geom, rope)
        };
        // The held tile is main's LAST tile: rows `block * floor((rows - 1) / block) .. rows`.
        let (limit, held) = if hold_last {
            let row0 = (rows - 1) / block * block;
            (row0, Some(at_(row0, rows - row0, block)?))
        } else {
            (rows, None)
        };
        let mut stored = Vec::new();
        let mut at = 0u32;
        // THE CLASS (see `StoreTiling::class_rows`): a chunk below the boundary is prod's
        // narrow-row class for every row, so it is stored narrow at its natural class and never
        // wide — a wide tile would launch its norm at 256 rows, the other class.
        let wide_on = tiling.wide != 0 && rows >= tiling.class_rows;
        let chunk_class = wide_on && tiling.class_rows != 0;
        // C6: a chunk BELOW the class boundary as wide tiles at ITS class — whole ones, then one
        // partial when the rest is at least `natural_min` rows and its window fits `aux.cat`.
        let natural_wide = !wide_on && tiling.wide != 0 && tiling.natural_min > 0;
        if natural_wide {
            while limit - at >= tiling.wide {
                stored.push(at_(at, tiling.wide, tiling.wide)?);
                at += tiling.wide;
            }
            let rest = limit - at;
            let window_fits = u64::from(aux0) + u64::from(at) + u64::from(tiling.wide)
                <= u64::from(tiling.aux_rows);
            if rest > 0 && rest >= tiling.natural_min && window_fits {
                stored.push(at_(at, rest, tiling.wide)?);
                at = limit;
            }
        }
        if wide_on {
            while limit - at >= tiling.wide {
                stored.push(at_(at, tiling.wide, tiling.wide)?);
                at += tiling.wide;
            }
            let rest = limit - at;
            let window_fits = u64::from(aux0) + u64::from(at) + u64::from(tiling.wide)
                <= u64::from(tiling.aux_rows);
            if rest > 0 && tiling.min_partial > 0 && rest >= tiling.min_partial && window_fits {
                stored.push(at_(at, rest, tiling.wide)?);
                at = limit;
            }
        }
        while at < limit {
            let live = (limit - at).min(block);
            stored.push(at_(at, live, block)?);
            at += live;
        }
        let natural_wide = natural_wide && stored.iter().any(|t| t.width() != block);
        Ok(StoreSchedule {
            stored,
            held,
            chunk_class,
            natural_wide,
        })
    }

    fn check_geom(rows: u32, geom: DraftGeometry) -> Result<(), ForwardError> {
        if geom.block < 2 {
            return Err(ForwardError::Device(format!(
                "a DFlash2 block of {}: a context store tiled at a block of 1 is not this plan's",
                geom.block
            )));
        }
        if geom.page == 0 {
            return Err(ForwardError::Device(
                "a drafter KV page of 0 tokens".to_owned(),
            ));
        }
        if rows == 0 {
            return Err(ForwardError::Device(
                "a prefill chunk of no rows has no context to store".to_owned(),
            ));
        }
        Ok(())
    }

    /// One tile `width` rows wide whose first `live` rows are the chunk's rows `at..at + live`.
    #[allow(clippy::too_many_arguments)]
    fn tile(
        first_pos: u32,
        at: u32,
        live: u32,
        width: u32,
        aux0: u32,
        kv_blocks: &[u32],
        geom: DraftGeometry,
        rope: Option<&saber_exec::mrope::MropePositions>,
    ) -> Result<Self, ForwardError> {
        let mut t = DraftCtxTile {
            positions: vec![0; width as usize],
            slots: vec![PAD_SLOT; width as usize],
            row0: aux0 + at,
            live,
        };
        for j in 0..live as usize {
            let pos = first_pos + at + j as u32;
            let page = (pos / geom.page) as usize;
            let blk = kv_blocks.get(page).ok_or_else(|| {
                ForwardError::Device(format!(
                    "the prompt's context at position {pos} needs drafter page {page} and \
                     this session holds {}: a clamped page id is another session's K/V, in \
                     range and silent",
                    kv_blocks.len()
                ))
            })?;
            t.positions[j] = draft_rope_pos(rope, pos);
            t.slots[j] = i64::from(*blk) * i64::from(geom.page) + i64::from(pos % geom.page);
        }
        Ok(t)
    }

    /// Rows this tile is WIDE — the store shape it runs at (`block`, or the wide store's).
    #[must_use]
    pub fn width(&self) -> u32 {
        self.slots.len() as u32
    }

    /// Bytes one tile's frame occupies on a wire of `rows` rows.
    #[must_use]
    pub fn frame_bytes(rows: u32) -> usize {
        16 + rows as usize * 16
    }

    /// This tile as ONE frame for the rank that holds no sessions and therefore no pages, at its
    /// own width.
    ///
    /// The follower runs exactly this rank's tiles over its own half of the drafter's K/V, so the
    /// slots are the leader's and the arithmetic is not repeated on the other card. `more` is what
    /// makes the NUMBER of tiles a fact both ranks agree on frame by frame: a prefill chunk's row
    /// count is the leader's alone.
    #[must_use]
    pub fn pack(&self, more: bool) -> Vec<u8> {
        let kind = if more { CtxFrame::Tile } else { CtxFrame::End };
        self.encode(kind, self.width())
    }

    /// This tile as one frame on a wire `wire` rows wide (H35): the header carries the tile's own
    /// WIDTH and the tables are padded to the wire's, so a follower reads one length whatever
    /// store shape the frame is for and learns the shape from the header.
    ///
    /// # Errors
    /// A tile wider than the wire.
    pub fn pack_wire(&self, kind: CtxFrame, wire: u32) -> Result<Vec<u8>, ForwardError> {
        let width = self.width();
        if width > wire || self.positions.len() != self.slots.len() {
            return Err(ForwardError::Device(format!(
                "a {width}-row context tile on a {wire}-row wire"
            )));
        }
        Ok(self.encode(kind, wire))
    }

    fn encode(&self, kind: CtxFrame, wire: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity(Self::frame_bytes(wire));
        out.extend_from_slice(&self.row0.to_le_bytes());
        out.extend_from_slice(&self.live.to_le_bytes());
        out.extend_from_slice(&kind.code().to_le_bytes());
        out.extend_from_slice(&self.width().to_le_bytes());
        for j in 0..wire as usize {
            out.extend_from_slice(&self.positions.get(j).copied().unwrap_or(0).to_le_bytes());
        }
        for j in 0..wire as usize {
            out.extend_from_slice(&self.slots.get(j).copied().unwrap_or(PAD_SLOT).to_le_bytes());
        }
        out
    }

    /// The terminator: a frame that says no tile follows.
    #[must_use]
    pub fn terminator(block: u32) -> Vec<u8> {
        Self::marker(CtxFrame::End, block, block)
    }

    /// A frame that carries no tile — the terminator or a store-check marker — `block` wide on a
    /// `wire`-row wire (a `block` above the wire is clamped to it).
    #[must_use]
    pub fn marker(kind: CtxFrame, block: u32, wire: u32) -> Vec<u8> {
        let w = block.min(wire) as usize;
        DraftCtxTile {
            positions: vec![0; w],
            slots: vec![PAD_SLOT; w],
            row0: 0,
            live: 0,
        }
        .encode(kind, wire)
    }

    /// Read a frame back, with `more`.
    ///
    /// # Errors
    /// A frame of the wrong length, or one whose declared block is not this rank's — both of
    /// which are a pair of binaries that disagree about the drafter rather than a short table.
    pub fn unpack(frame: &[u8], block: u32) -> Result<(Self, bool), ForwardError> {
        let (t, kind) = Self::unpack_wire(frame, block, &[block])?;
        match kind {
            CtxFrame::End => Ok((t, false)),
            CtxFrame::Tile => Ok((t, true)),
            other => Err(ForwardError::Peer(format!(
                "a prefill context frame of kind {other:?} on a wire that carries tiles only"
            ))),
        }
    }

    /// Read a frame off a `wire`-row wire whose tiles may be any of `widths` (H35): the store
    /// shapes this rank planned.
    ///
    /// # Errors
    /// A frame of the wrong length, a width this rank has no store shape for, a live count past
    /// the width, or a kind this protocol does not have — each a pair of binaries that disagree.
    pub fn unpack_wire(
        frame: &[u8],
        wire: u32,
        widths: &[u32],
    ) -> Result<(Self, CtxFrame), ForwardError> {
        let want = Self::frame_bytes(wire);
        if frame.len() != want {
            return Err(ForwardError::Peer(format!(
                "a prefill context frame of {} byte(s) against this rank's {want}",
                frame.len()
            )));
        }
        let u32_at = |i: usize| -> u32 {
            u32::from_le_bytes(frame[i..i + 4].try_into().unwrap_or_default())
        };
        let width = u32_at(12);
        if !widths.contains(&width) || width > wire {
            return Err(ForwardError::Peer(format!(
                "a prefill context frame at a width of {width} and this rank's store shapes are \
                 {widths:?} on a {wire}-row wire"
            )));
        }
        let kind = CtxFrame::from_code(u32_at(8)).ok_or_else(|| {
            ForwardError::Peer(format!("a prefill context frame of kind {}", u32_at(8)))
        })?;
        let live = u32_at(4);
        if live > width {
            return Err(ForwardError::Peer(format!(
                "a prefill context frame with {live} live row(s) in a {width}-row tile"
            )));
        }
        let mut t = DraftCtxTile {
            positions: Vec::with_capacity(width as usize),
            slots: Vec::with_capacity(width as usize),
            row0: u32_at(0),
            live,
        };
        let at = 16;
        for j in 0..width as usize {
            let o = at + j * 8;
            t.positions.push(i64::from_le_bytes(
                frame[o..o + 8].try_into().unwrap_or_default(),
            ));
        }
        let at = 16 + wire as usize * 8;
        for j in 0..width as usize {
            let o = at + j * 8;
            t.slots.push(i64::from_le_bytes(
                frame[o..o + 8].try_into().unwrap_or_default(),
            ));
        }
        Ok((t, kind))
    }

    /// `ctx.positions` as the batched rope reads it: one copy of the table per drafter layer.
    #[must_use]
    pub fn positions_bytes(&self, layers: u32) -> Vec<u8> {
        let one: Vec<u8> = self
            .positions
            .iter()
            .flat_map(|p| p.to_le_bytes())
            .collect();
        let mut out = Vec::with_capacity(one.len() * layers.max(1) as usize);
        for _ in 0..layers.max(1) {
            out.extend_from_slice(&one);
        }
        out
    }

    /// `ctx.slots.l{l}` — one table, written once per layer.
    #[must_use]
    pub fn slots_bytes(&self) -> Vec<u8> {
        self.slots.iter().flat_map(|s| s.to_le_bytes()).collect()
    }
}

// =====================================================================================
// The proposal's tables ON THE WIRE (ws-s6)
// =====================================================================================

/// One proposal's index tables as ONE frame, for the rank that cannot derive them.
///
/// **Why anything at all crosses.** `docs/EXEC.md` section 6: rank 1 keeps no session state, owns
/// no slot table and has no scheduler, and every one of these tables is a function of exactly the
/// facts it does not have — the sequence's drafter PAGES, its context SLOT, how many tokens the
/// step committed and which positions they hold. The rule the rest of this wire follows is "put
/// nothing on it the follower can compute"; these are the residue of that rule rather than an
/// exception to it, and they are the same residue the TARGET's index blob already is.
///
/// **The widths are both ranks' own.** Every field's length is a function of the plan's shape,
/// which both ranks built from the same `DflashConfig` and the same `DraftShape`, so a frame that
/// is not [`DraftWire::bytes`] long is a pair of binaries that disagree about the drafter — a
/// refusal by name here, not a silently short table on one card.
///
/// `ctx.positions` is NOT carried: it is the context positions repeated once per drafter layer
/// (the batched rope reads one position per row over all five layers at once), so the follower
/// expands it exactly as the leader does. The same goes for the per-layer copies of
/// `ctx.slots.l{l}` — one table on the wire, five writes on each card.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftWire {
    /// Context rows the store covers: the plan's `ctx_rows`.
    pub ctx_rows: u32,
    /// Query rows: `reqs * block`.
    pub query_rows: u32,
    /// Requests the tables are padded for.
    pub reqs: u32,
    /// Proposal positions: `reqs * (block - 1)`.
    pub steps: u32,
    /// Entries per sequence in the drafter's block table.
    pub bt_stride: u32,
}

/// One frame's tables, by the drafter's own buffer NAME — so the side that receives them seeds by
/// name and never by offset.
pub type DraftTables = Vec<(&'static str, Vec<u8>)>;

/// The header every proposal frame carries, and the only part a terminator has.
///
/// Three of its four fields exist because the follower cannot derive them and one of them is the
/// refusal that makes the pair checkable:
///
/// * `row0` — which rows of `aux.cat` this proposal's context store reads. Both cards hold the
///   same (post-reduce, therefore identical) residual stream; WHICH slot owns which rows is the
///   leader's own bookkeeping.
/// * `more` — whether a proposal follows this frame. The step's command header does not carry a
///   proposal count, so this is what makes the number of proposals a fact both ranks agree on
///   frame by frame rather than a number one of them guessed.
/// * `prev_digest` — the leader's own digest of the block it proposed for the PREVIOUS frame,
///   checked by the follower against its own. **This is the "the two ranks proposed the same
///   block" claim, on every proposal of every step, not only under a gate**: the drafter is
///   sharded, the two halves meet at one host seam, and a block that differed would change no
///   token and change the acceptance — the one thing an id list cannot see (ws-s5 part B).
/// * `count` — proposals the leader ran this step, on the terminator. A pair that disagreed would
///   advance the all-reduce sequence counters by different amounts and the next collective would
///   never retire.
/// * `shape` — **which of the drafter's planned shapes this proposal runs** (ws-s10). The drafter
///   is a batch of shapes now, exactly as the target is a batch of decode buckets, and the batch
///   is the leader's own scheduling decision: the follower holds no sessions and cannot know how
///   many were live. It rides in the header and not in the payload because the payload's WIDTH is
///   a function of it — a follower that read the frame before it knew the shape would be reading a
///   length it had guessed. Every frame is therefore padded to the widest shape's width and the
///   header is read first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DraftFrameHead {
    /// The first row of `aux.cat` this proposal reads.
    pub row0: u32,
    /// Does a proposal follow this frame?
    pub more: bool,
    /// The sender's digest of the block it last proposed, or 0 before the first one.
    pub prev_digest: u64,
    /// Proposals the sender ran this step. Only meaningful on a terminator.
    pub count: u32,
    /// Which of this rank's planned drafter shapes the frame's tables are laid out for.
    pub shape: u32,
    /// **Live requests in this proposal**: how many of the shape's `reqs` carry a real sequence.
    ///
    /// The rest are padding ([`DraftSeqStep::padding`]) and neither rank may read a block out of
    /// them — the walk stores nothing for a padded row, so what is there is the PREVIOUS
    /// proposal's. It is on the wire rather than derived because the follower holds no sessions:
    /// it is the same class of fact as `row0`, and the block digest the two ranks compare is
    /// taken over exactly these requests' ids on both cards.
    pub live: u32,
}

impl DraftFrameHead {
    /// The 24 bytes at the front of a frame.
    #[must_use]
    pub fn encode(&self) -> [u8; DraftWire::HEAD] {
        let mut h = [0u8; DraftWire::HEAD];
        h[0..4].copy_from_slice(&self.row0.to_le_bytes());
        h[4..8].copy_from_slice(&u32::from(self.more).to_le_bytes());
        h[8..16].copy_from_slice(&self.prev_digest.to_le_bytes());
        h[16..20].copy_from_slice(&self.count.to_le_bytes());
        h[20..24].copy_from_slice(&self.shape.to_le_bytes());
        h[24..28].copy_from_slice(&self.live.to_le_bytes());
        h
    }

    /// Read them back. A frame shorter than the header is a caller bug the width check catches
    /// first, so this is total.
    #[must_use]
    pub fn decode(frame: &[u8]) -> Self {
        let w = |at: usize| {
            u32::from_le_bytes([frame[at], frame[at + 1], frame[at + 2], frame[at + 3]])
        };
        let mut d = [0u8; 8];
        d.copy_from_slice(&frame[8..16]);
        DraftFrameHead {
            row0: w(0),
            more: w(4) != 0,
            prev_digest: u64::from_le_bytes(d),
            count: w(16),
            shape: w(20),
            live: w(24),
        }
    }
}

/// FNV-1a over a whole PROPOSAL's blocks — one per live request — which is what the two ranks
/// compare once a proposal is a batch (ws-s10).
///
/// Each block's LENGTH is mixed in before its ids, so two ranks that agreed on every id and
/// disagreed about where one block ended do not collide. A dropped proposal is every block empty,
/// which is a digest of its own and not zero.
#[must_use]
pub fn blocks_digest(blocks: &[Vec<TokenId>]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |b: u8| {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for block in blocks {
        for b in (block.len() as u64).to_le_bytes() {
            mix(b);
        }
        for t in block {
            for b in (*t as u64).to_le_bytes() {
                mix(b);
            }
        }
    }
    h | 1
}

/// FNV-1a over a block's token ids, which is what the two ranks compare.
#[must_use]
pub fn block_digest(block: &[TokenId]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for t in block {
        for b in (*t as u64).to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    // A zero digest means "no block yet", so an empty block must not collide with it.
    h | 1
}

impl DraftWire {
    /// Bytes of frame header before the first table.
    pub const HEAD: usize = 32;

    /// The frame's width, in bytes.
    #[must_use]
    pub fn bytes(&self) -> usize {
        let ctx = self.ctx_rows as usize;
        let q = self.query_rows as usize;
        let r = self.reqs as usize;
        let st = self.steps as usize;
        DraftWire::HEAD                // the frame's own header
            + ctx * 8                 // context positions
            + ctx * 8                 // context slots
            + q * 4                   // input ids
            + q * 8                   // query positions
            + q * 8                   // query slots
            + r * self.bt_stride as usize * 4
            + r * 4                   // seq lens
            + st * 8                  // sample indices
            + r * 8                   // anchor indices
            + st * 4 // the walk's request map
    }

    /// Pack one step's tables into a frame, in the order [`DraftWire::unpack`] reads them.
    ///
    /// # Errors
    /// A table whose length is not this geometry's — which is a caller that built the tables at
    /// one shape and the frame at another.
    pub fn pack(
        &self,
        ix: &DraftStepIndex,
        block_table: &[i32],
        head: DraftFrameHead,
    ) -> Result<Vec<u8>, ForwardError> {
        let want = |what: &str, got: usize, n: usize| -> Result<(), ForwardError> {
            if got == n {
                Ok(())
            } else {
                Err(ForwardError::Device(format!(
                    "the proposal's `{what}` is {got} entries and this frame's geometry says {n}"
                )))
            }
        };
        want(
            "context_positions",
            ix.context_positions.len(),
            self.ctx_rows as usize,
        )?;
        want(
            "context_slots",
            ix.context_slots.len(),
            self.ctx_rows as usize,
        )?;
        want("input_ids", ix.input_ids.len(), self.query_rows as usize)?;
        want(
            "query_positions",
            ix.query_positions.len(),
            self.query_rows as usize,
        )?;
        want(
            "query_slots",
            ix.query_slots.len(),
            self.query_rows as usize,
        )?;
        want(
            "block_table",
            block_table.len(),
            self.reqs as usize * self.bt_stride as usize,
        )?;
        want("seq_lens", ix.seq_lens.len(), self.reqs as usize)?;
        want(
            "sample_indices",
            ix.sample_indices.len(),
            self.steps as usize,
        )?;
        want(
            "sample_idx_mapping",
            ix.sample_idx_mapping.len(),
            self.steps as usize,
        )?;
        let mut out = Vec::with_capacity(self.bytes());
        out.extend_from_slice(&head.encode());
        for v in &ix.context_positions {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in &ix.context_slots {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in &ix.input_ids {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in &ix.query_positions {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in &ix.query_slots {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in block_table {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in &ix.seq_lens {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in &ix.sample_indices {
            out.extend_from_slice(&v.to_le_bytes());
        }
        // The anchor gather reads row 0 of each request's own query rows.
        for i in 0..self.reqs as usize {
            out.extend_from_slice(
                &((i as i64) * i64::from(self.query_rows / self.reqs.max(1))).to_le_bytes(),
            );
        }
        for v in &ix.sample_idx_mapping {
            out.extend_from_slice(&v.to_le_bytes());
        }
        if out.len() != self.bytes() {
            return Err(ForwardError::Device(format!(
                "a proposal frame of {} byte(s) against a geometry that says {}",
                out.len(),
                self.bytes()
            )));
        }
        Ok(out)
    }

    /// The frame's fields, in the order the device buffers take them: `(name, bytes)` pairs whose
    /// names are the drafter's own buffer names, so the receiving side seeds by name and never by
    /// offset.
    ///
    /// # Errors
    /// A frame that is not [`DraftWire::bytes`] long.
    pub fn unpack(&self, frame: &[u8]) -> Result<(DraftFrameHead, DraftTables), ForwardError> {
        // AT LEAST this shape's width, because a frame is padded to the WIDEST shape the drafter
        // planned (ws-s10): the follower has to read the header before it knows which shape's
        // tables follow, so every frame on this wire is one length and the tables are its prefix.
        if frame.len() < self.bytes() {
            return Err(ForwardError::Peer(format!(
                "a proposal frame of {} byte(s); this rank's plan says {}. The two ranks do not                  agree about the drafter's shape",
                frame.len(),
                self.bytes()
            )));
        }
        let ctx = self.ctx_rows as usize;
        let q = self.query_rows as usize;
        let r = self.reqs as usize;
        let st = self.steps as usize;
        let widths: [(&'static str, usize); 10] = [
            ("ctx.positions.base", ctx * 8),
            ("ctx.slots", ctx * 8),
            ("q.input_ids", q * 4),
            ("q.pos", q * 8),
            ("idx.query_slots", q * 8),
            ("idx.block_table", r * self.bt_stride as usize * 4),
            ("idx.seq_lens", r * 4),
            ("idx.sample_indices", st * 8),
            ("idx.anchor_indices", r * 8),
            ("walk.idx_mapping", st * 4),
        ];
        let head = DraftFrameHead::decode(frame);
        let mut at = DraftWire::HEAD;
        let mut out = Vec::with_capacity(widths.len());
        for (name, n) in widths {
            out.push((name, frame[at..at + n].to_vec()));
            at += n;
        }
        Ok((head, out))
    }

    /// **The anchor each live request of a packed frame is FED** (ws-f37): `q.input_ids` at the
    /// request's own row 0, read back out of the bytes both ranks seed from.
    ///
    /// It is read off the FRAME rather than taken from the caller's list because that is the
    /// claim the executor checks: that the block it is offered continues the token the slot
    /// committed. A frame is what the drafter consumes on both cards, so an anchor echoed from it
    /// is the drafter's own input and not a second copy of the caller's intention.
    ///
    /// # Errors
    /// A frame shorter than this geometry, or more live requests than the shape has.
    pub fn anchors(&self, frame: &[u8], live: usize) -> Result<Vec<TokenId>, ForwardError> {
        if frame.len() < self.bytes() || live > self.reqs as usize {
            return Err(ForwardError::Device(format!(
                "reading {live} anchor(s) out of a frame of {} byte(s) at a geometry of {} \
                 request(s) and {} byte(s)",
                frame.len(),
                self.reqs,
                self.bytes()
            )));
        }
        let per = (self.query_rows / self.reqs.max(1)) as usize;
        let ids_at = DraftWire::HEAD + self.ctx_rows as usize * 16;
        Ok((0..live)
            .map(|i| {
                let at = ids_at + i * per * 4;
                i32::from_le_bytes([frame[at], frame[at + 1], frame[at + 2], frame[at + 3]])
                    as TokenId
            })
            .collect())
    }

    /// The frame that says "no more proposals this step", with the count and the last digest.
    #[must_use]
    pub fn terminator(&self, head: DraftFrameHead) -> Vec<u8> {
        let mut out = vec![0u8; self.bytes()];
        out[..DraftWire::HEAD].copy_from_slice(&head.encode());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geom() -> DraftGeometry {
        DraftGeometry {
            block: 8,
            page: 16,
            max_reqs: 4,
            max_pos: 4096,
            mask_token: MASK_TOKEN,
        }
    }

    /// The five taps are the edges the engine's own probe digests, and they are the residual
    /// stream AFTER the layer the drafter's config names minus one.
    #[test]
    fn the_aux_taps_name_the_engines_own_dump_points() {
        assert_eq!(
            aux_tap_edges(),
            vec![
                "l5.resid.a".to_owned(),
                "l19.resid.a".to_owned(),
                "l33.resid.a".to_owned(),
                "l47.resid.a".to_owned(),
                "l61.resid.a".to_owned(),
            ]
        );
        // One per target layer the plan declares the aux projection reads: the projection's K is
        // `target_layers * hidden`, so a list of another length cannot be bound at all.
        let cfg = saber_model::dflash_graph::DflashConfig::qwen38_dflash2(1, 7);
        assert_eq!(AUX_TAPS.len() as u32, cfg.target_layers);
    }

    /// **The five taps are edges of the engine's OWN decode plan**, and each is the output of the
    /// residual add at the layer the drafter's checkpoint names.
    ///
    /// Asked of the real graph and the real planner rather than of a name list, because the tap is
    /// the one input to a proposal that no pack can produce: if `l19.resid.a` stopped being an edge
    /// — a fused norm, a renamed residual — a drafter wired to these names would read whatever the
    /// arena held and propose a function of it, which the acceptance law rejects silently.
    ///
    /// NEGATIVE CONTROL in the same test: a tap one past the model's last layer does NOT resolve,
    /// so this is a lookup that can fail.
    ///
    /// **It also states what a tap costs**, which is the next round's problem and not this one's:
    /// the body plan runs with slot reuse ON, so these five edges are aliased onto a handful of
    /// buffers and only the last of them holds its own value at the end of a step. Reading them
    /// needs five copies issued AT the tap points inside the step's op list — five more nodes, five
    /// more in the declared budget — or a body plan with reuse off, which is an arena the card does
    /// not have.
    #[test]
    fn every_aux_tap_is_an_edge_of_the_engines_own_plan() {
        use saber_exec::layer::{BlockPlan, PlanOptions, RuntimeParams};
        use saber_kernels::rules::LaunchRules;
        use saber_kernels::saber::SaberRules;
        use saber_model::graph::{Bucket, Graph, ModelConfig};

        let graph = Graph::qwen38_27b_at_width(ModelConfig::qwen38_27b(), 1, 8).expect("graph");
        let layers: Vec<u32> = (0..graph.config.num_hidden_layers).collect();
        let mx = LaunchRules::prod_mxfp4().expect("mxfp4 rules");
        let r4 = SaberRules::prod_saber().expect("saber rules");
        let plan = BlockPlan::with_options(
            &graph,
            Bucket::Decode {
                batch: 1,
                verify_width: 8,
            },
            &layers,
            RuntimeParams {
                max_ctx: 4096,
                gdn_slots: 9,
            },
            &mx,
            &r4,
            PlanOptions::BODY,
        )
        .expect("a decode plan");

        for (tap, edge) in AUX_TAPS.iter().zip(aux_tap_edges()) {
            assert!(
                *tap < graph.config.num_hidden_layers,
                "tap {tap} is past the model's {} layers",
                graph.config.num_hidden_layers
            );
            let at = plan
                .locate(&edge)
                .unwrap_or_else(|| panic!("{edge} is not an edge of the engine's decode plan"));
            assert_eq!(
                at.bytes,
                u64::from(graph.config.hidden_size) * 2 * 8,
                "{edge} is not 8 rows of bf16 hidden"
            );
            // The edge is the output of THAT layer's input layernorm, which is the residual add
            // The reference server's aux hook reads. Named, so a rename is a failure here rather than a silent
            // change of which layer the drafter is fed.
            let producer = format!("layer.{tap}.input_layernorm");
            assert!(
                plan.steps.iter().any(|st| st.op.name == producer),
                "{producer} is not a step of this plan"
            );
        }
        // THE CONTROL: one past the last layer is not an edge.
        let past = format!("l{}.resid.a", graph.config.num_hidden_layers);
        assert!(plan.locate(&past).is_none(), "{past} resolved");
    }

    /// D4: an image sequence's drafter rope is the TEMPORAL M-RoPE axis (prod's 1-D load of the
    /// target's `[3, N]` positions): past the prompt `index + delta` for the context rows and
    /// `last_valid + delta + 1 + off` for the query rows; the SLOTS stay the index. A text-only
    /// sequence (`None`) is unchanged, byte for byte.
    #[test]
    fn an_image_sequences_drafter_rope_is_the_temporal_axis_and_its_slots_the_index() {
        let pages: Vec<u32> = (0..8).collect();
        let mut prompt = vec![1i32; 4];
        prompt.extend([9; 48]); // a (1, 12, 16) grid: 6 x 8 = 48 tokens, spanning 8 positions
        prompt.extend([2; 4]);
        let rope = std::sync::Arc::new(
            saber_exec::mrope::for_prompt(&prompt, 9, &[[1, 12, 16]]).expect("layout"),
        );
        assert_eq!(rope.delta, -40);
        let text = DraftSeqStep::decode(100, 8, 4, 77, &pages, 0).expect("legal");
        let img = text.clone().with_rope(Some(rope.clone()));
        let (a, b) = (
            DraftStepIndex::build(std::slice::from_ref(&text), geom()).expect("text"),
            DraftStepIndex::build(std::slice::from_ref(&img), geom()).expect("image"),
        );
        assert_eq!(&a.context_positions[..4], &[100, 101, 102, 103]);
        assert_eq!(&b.context_positions[..4], &[60, 61, 62, 63]);
        assert_eq!(a.context_slots, b.context_slots, "the slots are the index");
        assert_eq!(a.query_slots, b.query_slots);
        assert_eq!(a.seq_lens, b.seq_lens);
        assert_eq!(b.query_positions[0], 64);
        assert_eq!(b.query_positions[7], 71);
        // A prompt tile: the temporal axis inside the prompt (the image's 48 tokens all at 4).
        let t = DraftCtxTile::chunk_rope(0, 56, 0, &pages, geom(), Some(&rope)).expect("tiles");
        let flat: Vec<i64> = t
            .iter()
            .flat_map(|x| x.positions[..x.live as usize].to_vec())
            .collect();
        assert_eq!(&flat[..6], &[0, 1, 2, 3, 4, 4]);
        assert_eq!(&flat[52..], &[12, 13, 14, 15]);
        let plain = DraftCtxTile::chunk(0, 56, 0, &pages, geom()).expect("tiles");
        assert_eq!(
            t.iter().map(|x| x.slots.clone()).collect::<Vec<_>>(),
            plain.iter().map(|x| x.slots.clone()).collect::<Vec<_>>()
        );
    }

    /// D6: PROD'S law as prod runs it (the reference server's Model Runner V2: the DFlash speculator's
    /// `_prepare_dflash_inputs_kernel` reads `input_batch.positions`, i.e. `num_computed + i`, for
    /// every table). Under `--drafter-rope-index` an image sequence's drafter session carries no rope
    /// table, so its tables ARE the text law's, byte for byte: context rows at their index, slots on
    /// the index, query rows at `last_valid + 1 + off`, `seq_lens = last_valid + 1 + block`. A
    /// text-only sequence is unchanged by the switch (the delta is 0 and no table is handed). The
    /// CONTROL is D4's temporal law on the same image, which must differ.
    #[test]
    fn under_the_index_law_an_image_sequences_drafter_reads_at_prods_positions() {
        use crate::device::drafter_rope_table;
        let pages: Vec<u32> = (0..8).collect();
        let mut prompt = vec![1i32; 4];
        prompt.extend([9; 48]);
        prompt.extend([2; 4]);
        let rope = std::sync::Arc::new(
            saber_exec::mrope::for_prompt(&prompt, 9, &[[1, 12, 16]]).expect("layout"),
        );
        let imgs = crate::forward::SeqImages {
            rope: rope.clone(),
            epoch: 1,
        };
        assert!(
            drafter_rope_table(false, Some(&imgs)).is_some(),
            "D4's default"
        );
        assert!(drafter_rope_table(true, Some(&imgs)).is_none());
        assert!(
            drafter_rope_table(true, None).is_none() && drafter_rope_table(false, None).is_none()
        );

        let g = geom();
        let text = DraftSeqStep::decode(100, 8, 4, 77, &pages, 0).expect("legal");
        let index_law = text
            .clone()
            .with_rope(drafter_rope_table(true, Some(&imgs)));
        let temporal = text
            .clone()
            .with_rope(drafter_rope_table(false, Some(&imgs)));
        let a = DraftStepIndex::build(std::slice::from_ref(&text), g).expect("text");
        let b = DraftStepIndex::build(std::slice::from_ref(&index_law), g).expect("index");
        let c = DraftStepIndex::build(std::slice::from_ref(&temporal), g).expect("temporal");
        assert_eq!(
            a, b,
            "the index law's tables are the text law's, byte for byte"
        );
        // prod's V2 kernel, restated over input_batch.positions = 100..108, 4 valid:
        assert_eq!(&b.context_positions[..4], &[100, 101, 102, 103]);
        assert_eq!(
            b.context_slots[0],
            100 / i64::from(g.page) * i64::from(g.page) + 100 % i64::from(g.page)
        );
        assert_eq!(b.query_positions[0], 104);
        assert_eq!(b.seq_lens[0], (103 + 1 + g.block) as i32);
        // THE CONTROL: D4's temporal law moves the rope (index + delta) and nothing else.
        assert_ne!(c.context_positions, b.context_positions);
        assert_eq!(c.context_slots, b.context_slots);
        // The prompt's tiles: no table -> the plain index chunk.
        let t = DraftCtxTile::chunk_rope(0, 56, 0, &pages, g, index_law.rope.as_deref())
            .expect("tiles");
        let plain = DraftCtxTile::chunk(0, 56, 0, &pages, g).expect("tiles");
        assert_eq!(
            t.iter().map(|x| x.positions.clone()).collect::<Vec<_>>(),
            plain
                .iter()
                .map(|x| x.positions.clone())
                .collect::<Vec<_>>()
        );
    }

    /// The shape of every table, and the padding that makes a replay safe.
    #[test]
    fn every_table_is_padded_to_the_shape_a_replay_was_captured_at() {
        let g = geom();
        let r = DraftSeqStep::decode(100, 8, 4, 77, &[0, 1, 2, 3, 4, 5, 6, 7], 0).expect("legal");
        let ix = DraftStepIndex::build(&[r], g).expect("legal");
        assert_eq!(ix.context_positions.len(), 8);
        assert_eq!(ix.context_slots.len(), 8);
        assert_eq!(ix.input_ids.len(), 32);
        assert_eq!(ix.query_positions.len(), 32);
        assert_eq!(ix.query_slots.len(), 32);
        assert_eq!(ix.query_start_loc, vec![0, 8, 8, 8, 8]);
        assert_eq!(ix.seq_lens.len(), 4);
        assert_eq!(ix.sample_indices.len(), 28);
        // Past the one live request: no slot, no state, and the mask token rather than a stale id.
        assert!(ix.query_slots[8..].iter().all(|s| *s == PAD_SLOT));
        assert!(ix.sample_idx_mapping[7..].iter().all(|m| *m == -1));
        assert!(ix.input_ids[8..].iter().all(|t| *t == MASK_TOKEN as i32));
        assert_eq!(ix.seq_lens[1..], [0, 0, 0]);
    }

    /// **The rejection is what the pad slot is for.** Four of eight rows committed means rows
    /// 4..8 carry hidden states for tokens the model refused, and they must not reach the cache.
    ///
    /// NEGATIVE CONTROL in the same test: a step that committed everything writes all eight, so
    /// the pad is a consequence of the verdict and not a constant.
    #[test]
    fn the_rows_after_a_rejection_have_no_slot_and_the_full_accept_has_eight() {
        let g = geom();
        let part = DraftSeqStep::decode(100, 8, 4, 77, &[0, 1, 2, 3, 4, 5, 6, 7], 0).expect("ok");
        let ix = DraftStepIndex::build(&[part], g).expect("legal");
        assert_eq!(ix.context_positions[..4], [100, 101, 102, 103]);
        assert!(ix.context_slots[..4].iter().all(|s| *s != PAD_SLOT));
        assert!(ix.context_slots[4..].iter().all(|s| *s == PAD_SLOT));
        assert!(ix.context_positions[4..].iter().all(|p| *p == 0));
        // The query rows follow the LAST VALID row, not the last row.
        assert_eq!(
            ix.query_positions[..8],
            [104, 105, 106, 107, 108, 109, 110, 111]
        );
        assert_eq!(ix.seq_lens[0], 111 + 1);

        let all = DraftSeqStep::decode(100, 8, 8, 77, &[0, 1, 2, 3, 4, 5, 6, 7], 0).expect("ok");
        let full = DraftStepIndex::build(&[all], g).expect("legal");
        assert!(full.context_slots.iter().all(|s| *s != PAD_SLOT));
        assert_eq!(full.query_positions[0], 108);
    }

    /// A slot is a page of the DRAFTER's own table times the page size, plus the offset — and a
    /// position past the table is a refusal, not a clamp.
    ///
    /// The refusal is the point: `prepare_dflash_inputs` CLAMPS (`tl.minimum(block_num,
    /// block_table_stride - 1)`) because a Triton kernel has nowhere to report, and a clamp writes
    /// this sequence's K/V into the last page it owns — in range, wrong position, silent. The
    /// engine has somewhere to report.
    #[test]
    fn a_position_with_no_page_is_refused_rather_than_clamped() {
        let g = geom();
        // Two pages = 32 tokens. Row 30 is fine, and the eight query rows from 31 are not.
        let r = DraftSeqStep::decode(24, 8, 8, 77, &[9, 10], 0).expect("ok");
        let err = DraftStepIndex::build(&[r], g).expect_err("past the table");
        assert!(err.to_string().contains("drafter page"), "{err}");
        assert!(err.to_string().contains("in range and silent"), "{err}");
        // With the pages for it, the same step resolves and the slots are the pages'.
        let ok = DraftSeqStep::decode(24, 8, 8, 77, &[9, 10, 11, 12], 0).expect("ok");
        let ix = DraftStepIndex::build(&[ok], g).expect("legal");
        // position 24 -> page 1 of the sequence -> block 10, offset 8
        assert_eq!(ix.context_slots[0], 10 * 16 + 8);
        // position 32 -> page 2 -> block 11, offset 0
        assert_eq!(ix.query_positions[0], 32);
        assert_eq!(ix.query_slots[0], 11 * 16);
    }

    /// Two requests: the context rows are laid out at the TARGET's offsets and the query rows at
    /// `i * block`, which are two different strides and stay two.
    #[test]
    fn two_requests_use_the_targets_row_offsets_and_the_drafters_block_offsets() {
        let g = geom();
        let a = DraftSeqStep::decode(10, 8, 2, 11, &[0, 1, 2], 0).expect("ok");
        let b = DraftSeqStep::decode(50, 8, 8, 22, &[3, 4, 5, 6, 7], 1).expect("ok");
        let ix = DraftStepIndex::build(&[a, b], g).expect("legal");
        assert_eq!(ix.context_positions.len(), 16);
        assert_eq!(ix.context_positions[..2], [10, 11]);
        assert_eq!(
            ix.context_positions[8..16],
            [50, 51, 52, 53, 54, 55, 56, 57]
        );
        assert_eq!(ix.query_start_loc, vec![0, 8, 16, 16, 16]);
        assert_eq!(ix.input_ids[0], 11);
        assert_eq!(ix.input_ids[8], 22);
        assert_eq!(ix.query_positions[8], 58);
        // Each request's proposals carry ITS state slot.
        assert!(ix.sample_idx_mapping[..7].iter().all(|m| *m == 0));
        assert!(ix.sample_idx_mapping[7..14].iter().all(|m| *m == 1));
        assert!(ix.sample_idx_mapping[14..].iter().all(|m| *m == -1));
    }

    /// **THE TP=2 FRAME CARRIES THE COMMITTED ANCHOR** (ws-f37).
    ///
    /// After the reorder the leader packs the proposal's tables from the token the host sampler
    /// COMMITTED, and the follower seeds its half of the drafter from the same bytes — so the
    /// anchor has to be in the frame, at each live request's row 0, and nowhere else. The anchors
    /// here are deliberately NOT what an argmax would have been (`11`/`22` against a device that
    /// "wanted" `10`/`21`), and they come back out of the packed frame, both through
    /// [`DraftWire::anchors`] — the echo the executor checks — and through the follower's own
    /// [`DraftWire::unpack`]. The CONTROL: the argmax the device would have anchored on is not
    /// what the frame carries, and a padding request's row 0 is not read as an anchor.
    #[test]
    fn the_tp2_frame_carries_the_committed_anchor() {
        let g = geom();
        let (argmax_a, argmax_b) = (10u32, 21u32);
        let (committed_a, committed_b) = (11u32, 22u32);
        let a = DraftSeqStep::decode(10, 8, 2, committed_a, &[0, 1, 2], 0).expect("ok");
        let b = DraftSeqStep::decode(50, 8, 8, committed_b, &[3, 4, 5, 6, 7], 1).expect("ok");
        let mut steps = vec![a, b];
        while steps.len() < g.max_reqs as usize {
            let like = steps.last().expect("one").clone();
            steps.push(DraftSeqStep::padding(&like));
        }
        let ix = DraftStepIndex::build(&steps, g).expect("legal");
        let bt_stride = 8u32;
        let bt = DraftStepIndex::block_table(&steps, bt_stride).expect("pages");
        let wire = DraftWire {
            ctx_rows: ix.context_positions.len() as u32,
            query_rows: ix.input_ids.len() as u32,
            reqs: g.max_reqs,
            steps: ix.sample_indices.len() as u32,
            bt_stride,
        };
        let head = DraftFrameHead {
            row0: 0,
            more: true,
            prev_digest: 0,
            count: 0,
            shape: 0,
            live: 2,
        };
        let frame = wire.pack(&ix, &bt, head).expect("packs");
        let fed = wire.anchors(&frame, 2).expect("two anchors");
        assert_eq!(fed, vec![committed_a, committed_b], "the committed anchors");
        assert_ne!(fed, vec![argmax_a, argmax_b], "the control: not the argmax");
        // The follower's side: the same bytes, seeded by name.
        let (_, tables) = wire.unpack(&frame).expect("unpacks");
        let ids = &tables
            .iter()
            .find(|(n, _)| *n == "q.input_ids")
            .expect("the ids table")
            .1;
        let id_at = |row: usize| {
            i32::from_le_bytes([
                ids[row * 4],
                ids[row * 4 + 1],
                ids[row * 4 + 2],
                ids[row * 4 + 3],
            ])
        };
        assert_eq!(id_at(0), committed_a as i32);
        assert_eq!(id_at(8), committed_b as i32);
        // Every other query row is the mask token: the anchor is row 0 and only row 0.
        assert!((1..8).all(|r| id_at(r) == MASK_TOKEN as i32));
        // More anchors than the shape has requests is a refusal, not a read past the table.
        assert!(wire.anchors(&frame, 5).is_err());
        assert!(wire.anchors(&frame[..10], 1).is_err());
    }

    /// **The context tables are built at the PLAN's `block`, not at the step's width**, and this is
    /// the law that makes that legal.
    ///
    /// `crate::device`'s proposal binds a context store whose descriptors were resolved at
    /// `ctx_rows = block`, so the tables it uploads have to be that long whatever width the target
    /// ran. Building the step with `rows = block` and the verdict's own `committed` gives the same
    /// `valid`, the same positions for every valid row and the same `last_valid_pos` as building it
    /// with `rows = width` — the extra rows are `PAD_SLOT`, which is exactly what the rows after a
    /// rejection already are. If that stopped being true the store would write K/V for a row the
    /// target never ran, in range and silent.
    ///
    /// NEGATIVE CONTROL in the same test: padding to the block while claiming `committed = rows`
    /// (i.e. forgetting that the pad rows are not accepted) moves `last_valid_pos` and therefore
    /// every query position — so the padding is only sound because `valid` is the VERDICT's.
    #[test]
    fn padding_the_context_to_the_block_moves_nothing_the_law_reads() {
        let g = geom();
        let pages: Vec<u32> = (0..8).collect();
        for width in 1u32..=8 {
            for committed in 1..=width {
                let narrow =
                    DraftSeqStep::decode(100, width, committed, 77, &pages, 0).expect("ok");
                let padded =
                    DraftSeqStep::decode(100, g.block, committed, 77, &pages, 0).expect("ok");
                assert_eq!(narrow.valid(), padded.valid(), "width {width}/{committed}");
                let a = DraftStepIndex::build(&[narrow], g).expect("legal");
                let b = DraftStepIndex::build(&[padded], g).expect("legal");
                let v = committed as usize;
                assert_eq!(a.context_positions[..v], b.context_positions[..v]);
                assert_eq!(a.context_slots[..v], b.context_slots[..v]);
                // Everything the DRAFTER's own rows read is a function of `last_valid_pos`, and it
                // does not move: the query positions, the slots they land in, and the window.
                assert_eq!(
                    a.query_positions, b.query_positions,
                    "w{width} c{committed}"
                );
                assert_eq!(a.query_slots, b.query_slots);
                assert_eq!(a.seq_lens, b.seq_lens);
                assert_eq!(a.input_ids, b.input_ids);
                assert_eq!(a.sample_indices, b.sample_indices);
                // And the pad rows carry no slot, so the store runs over them and writes nothing.
                assert!(b.context_slots[v..].iter().all(|s| *s == PAD_SLOT));
            }
        }
        // THE CONTROL: the same padding with the pad rows claimed as committed.
        let wrong = DraftSeqStep::decode(100, g.block, g.block, 77, &pages, 0).expect("ok");
        let right = DraftSeqStep::decode(100, g.block, 2, 77, &pages, 0).expect("ok");
        let w = DraftStepIndex::build(&[wrong], g).expect("legal");
        let r = DraftStepIndex::build(&[right], g).expect("legal");
        assert_ne!(w.query_positions, r.query_positions);
        assert_ne!(w.seq_lens, r.seq_lens);
    }

    /// The refusals, each of which is a table that would otherwise be written somewhere in range.
    #[test]
    fn the_refusals_name_what_they_refuse() {
        let g = geom();
        // A verdict the acceptance law cannot have produced.
        assert!(DraftSeqStep::decode(0, 8, 0, 1, &[0], 0).is_err());
        assert!(DraftSeqStep::decode(0, 8, 9, 1, &[0], 0).is_err());
        // More requests than the tables are padded for.
        let r = || DraftSeqStep::decode(0, 8, 1, 1, &[0, 1], 0).expect("ok");
        let many: Vec<DraftSeqStep> = (0..5).map(|_| r()).collect();
        let err = DraftStepIndex::build(&many, g).expect_err("past the padding");
        assert!(err.to_string().contains("padded for 4"), "{err}");
        // A block of 1 proposes nothing.
        let one = DraftGeometry { block: 1, ..g };
        assert!(DraftStepIndex::build(&[r()], one).is_err());
        // A request with no rows.
        let empty = DraftSeqStep {
            positions: Vec::new(),
            rejected: 0,
            anchor: 1,
            kv_blocks: vec![0],
            state_idx: 0,
            pad: false,
            rope: None,
        };
        let err = DraftStepIndex::build(&[empty], g).expect_err("no rows");
        assert!(err.to_string().contains("ran no rows"), "{err}");
    }

    /// **THE TOP OF A CONTEXT IS PROD'S LAW, NOT A REFUSAL** (ws-t2; ws-s11 read the difference).
    ///
    /// A proposal's query rows run `last_valid_pos + 1 .. + block` whatever the context ceiling
    /// is, so the last `block` positions of a session are where the two stacks used to part:
    /// prod clamps `seq_lens` at `max_model_len` and maps a non-resident block to `PAD_SLOT_ID`
    /// (`patch_dflash_base.py`), and Saber clamped the query POSITIONS only — and then refused by
    /// name in `slot_of` for the rows past the pages, which at `max_ctx` and a full context is
    /// every step, on a session prod keeps serving.
    ///
    /// The walk is over `max_ctx - block + 1 ..= max_ctx` as the next position, which is exactly
    /// the band where the clamp starts to bind, and it asserts three things per position: the
    /// clamp on `seq_lens`, `PAD_SLOT` on the query rows at or past the ceiling, and a REAL slot
    /// on the rows below it — the last being the control, because a test that only looked for
    /// PAD_SLOTs would pass on a build that padded the whole block.
    #[test]
    fn at_the_top_of_a_context_seq_lens_clamps_and_the_overrun_rows_are_pad_slot() {
        // A small context so the whole page run is spelled out: 64 positions, 16 to a page.
        const MAX_CTX: u32 = 64;
        let g = DraftGeometry {
            block: 8,
            page: 16,
            max_reqs: 1,
            max_pos: MAX_CTX,
            mask_token: MASK_TOKEN,
        };
        let pages: Vec<u32> = (0..MAX_CTX / g.page).collect();
        for next in (MAX_CTX - g.block + 1)..=MAX_CTX {
            // One row, committed: `last_valid_pos` is `next - 1` and the block runs from `next`.
            let last_valid = next - 1;
            let req = DraftSeqStep::decode(last_valid, 1, 1, 7, &pages, 0).expect("a live step");
            let ix = DraftStepIndex::build(&[req], g).expect("the top of a context is servable");

            assert_eq!(
                ix.seq_lens[0], MAX_CTX as i32,
                "next {next}: `min(last_valid + 1 + block, max_model_len)` — prod's own clamp"
            );
            for off in 0..g.block as usize {
                let pos = next + off as u32;
                let slot = ix.query_slots[off];
                if pos >= MAX_CTX {
                    assert_eq!(
                        slot, PAD_SLOT,
                        "next {next} row {off}: position {pos} is at or past the ceiling"
                    );
                } else {
                    let want = i64::from(pages[(pos / g.page) as usize]) * i64::from(g.page)
                        + i64::from(pos % g.page);
                    assert_eq!(
                        slot, want,
                        "next {next} row {off}: position {pos} is still inside the context and \
                         must write its own slot — the control"
                    );
                }
                assert!(
                    ix.query_positions[off] < i64::from(MAX_CTX),
                    "next {next} row {off}: a position past the rope table is never fed"
                );
            }
        }

        // AND THE REFUSAL IS STILL THERE where it means a defect: a page run too short for the
        // context this geometry says the session may reach is not a context that ran out, and it
        // is named rather than padded away.
        let short = DraftSeqStep::decode(20, 1, 1, 7, &[0], 0).expect("a live step");
        let err = DraftStepIndex::build(&[short], g).expect_err("a page run that is too short");
        assert!(err.to_string().contains("needs drafter page"), "{err}");
    }

    /// **A PADDING REQUEST STORES NOWHERE, ATTENDS NOTHING AND PROPOSES NOTHING** (ws-s10).
    ///
    /// A batched proposal runs at a planned shape and a step may be narrower than it, so the rows
    /// the shape has and the step does not are a request of their own. Three facts make that safe
    /// and each is a different kernel's own early-out:
    ///
    /// * every context and query slot is [`PAD_SLOT`] — the K/V appends' own early-out, so the
    ///   drafter's cache takes nothing from it;
    /// * `seq_lens` is 0 — `dflash_attn`'s tile count is `ceil(seq_len / TILE)`, so its attention
    ///   reads no page at all, which is why carrying a LIVE request's block-table row is safe;
    /// * `sample_idx_mapping` is −1 — `dflash_selector_walk_greedy` gates every store on
    ///   `req_state >= 0`, so it writes neither a token nor a realized score.
    ///
    /// THE CONTROL is the live request it was built from: the same positions and the same pages
    /// must produce a real slot, or this test is asserting that a table is empty for some reason
    /// other than the padding.
    #[test]
    fn a_padding_request_writes_nowhere() {
        let g = DraftGeometry {
            block: 8,
            page: 16,
            max_reqs: 2,
            max_pos: 4096,
            mask_token: MASK_TOKEN,
        };
        let live = DraftSeqStep::decode(24, 8, 3, 99, &[4, 5, 6], 7).expect("a live request");
        let pad = DraftSeqStep::padding(&live);
        let ix = DraftStepIndex::build(&[live.clone(), pad.clone()], g).expect("a padded batch");

        // Request 0 is untouched by the padding beside it.
        assert_eq!(
            ix.context_slots[0],
            5 * 16 + 8,
            "the live request still stores: position 24 is page 1, which is block 5"
        );
        assert_eq!(ix.seq_lens[0], 24 + 3 - 1 + 1 + 8);
        assert_eq!(ix.sample_idx_mapping[0], 7);
        // Request 1 writes nowhere.
        assert!(
            ix.context_slots[8..16].iter().all(|s| *s == PAD_SLOT),
            "{:?}",
            &ix.context_slots[8..16]
        );
        assert!(
            ix.query_slots[8..16].iter().all(|s| *s == PAD_SLOT),
            "{:?}",
            &ix.query_slots[8..16]
        );
        assert_eq!(
            ix.seq_lens[1], 0,
            "a padding request's attention reads nothing"
        );
        assert!(ix.sample_idx_mapping[7..14].iter().all(|m| *m == -1));
        // ... and its block-table row is the live request's, which is a LEGAL page and not 0.
        let bt = DraftStepIndex::block_table(&[live, pad], 4).expect("a legal table");
        assert_eq!(&bt[4..8], &[4, 5, 6, 6]);
    }

    /// The frame header carries the SHAPE and the LIVE count, and both survive the wire (ws-s10).
    ///
    /// They are in the header rather than the payload because the payload's WIDTH is a function of
    /// the shape: a follower that read the tables before it knew which shape they were for would
    /// be reading a length it had guessed.
    #[test]
    fn the_frame_head_carries_the_shape_and_the_live_count() {
        let h = DraftFrameHead {
            row0: 24,
            more: true,
            prev_digest: 0xdead_beef_1234_5678,
            count: 3,
            shape: 2,
            live: 3,
        };
        let mut frame = h.encode().to_vec();
        frame.resize(DraftWire::HEAD + 64, 0);
        assert_eq!(DraftFrameHead::decode(&frame), h);
        // A terminator's header is the only part of it that is read, and it round-trips too.
        let t = DraftFrameHead {
            more: false,
            count: 4,
            ..h
        };
        let mut frame = t.encode().to_vec();
        frame.resize(DraftWire::HEAD + 64, 0);
        assert_eq!(DraftFrameHead::decode(&frame), t);
    }

    /// Two proposals of different blocks digest differently, and a LENGTH difference alone is
    /// enough (ws-s10): the two ranks compare one number over a whole batch now.
    #[test]
    fn the_batch_digest_separates_the_blocks() {
        let a = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let b = vec![vec![1, 2, 3], vec![4, 5, 7]];
        let c = vec![vec![1, 2, 3, 4, 5, 6]];
        let d = vec![vec![1, 2, 3], vec![4, 5, 6], Vec::new()];
        assert_ne!(blocks_digest(&a), blocks_digest(&b));
        assert_ne!(
            blocks_digest(&a),
            blocks_digest(&c),
            "the lengths are mixed in"
        );
        assert_ne!(blocks_digest(&a), blocks_digest(&d));
        assert_eq!(blocks_digest(&a), blocks_digest(&a.clone()));
        assert_ne!(blocks_digest(&[]), 0, "no block yet is not a zero digest");
    }

    /// The block table pads with the sequence's OWN last page, never with 0.
    ///
    /// NEGATIVE CONTROL: page 0 belongs to whichever session holds it, so a table that padded with
    /// it would point a stray read at another request's K/V. The assertion is that the padding is
    /// this sequence's own page even when that page IS 0 for another sequence.
    #[test]
    fn the_block_table_pads_with_the_sequences_own_last_page() {
        let a = DraftSeqStep::decode(0, 8, 1, 1, &[7, 8], 0).expect("ok");
        let b = DraftSeqStep::decode(0, 8, 1, 1, &[0], 1).expect("ok");
        let t = DraftStepIndex::block_table(&[a, b], 4).expect("legal");
        assert_eq!(t, vec![7, 8, 8, 8, 0, 0, 0, 0]);
        let none = DraftSeqStep {
            positions: vec![0],
            rejected: 0,
            anchor: 1,
            kv_blocks: Vec::new(),
            state_idx: 0,
            pad: false,
            rope: None,
        };
        assert!(DraftStepIndex::block_table(&[none], 4).is_err());
    }

    // =================================================================================
    // THE PROMPT'S CONTEXT (ws-s8)
    // =================================================================================

    /// A whole prompt's pages, one per 16 positions, in the order a session holds them.
    fn pages(n: u32) -> Vec<u32> {
        (64..64 + n).collect()
    }

    /// The law, stated over a chunk: every prompt row is stored EXACTLY ONCE, at its own absolute
    /// position, in this session's own page — and the tail of the last tile is [`PAD_SLOT`].
    ///
    /// This is `_prepare_dflash_inputs_kernel`'s context half read at a prefill step: prod's
    /// `num_ctx` there is the chunk's scheduled tokens, `num_rejected` is 0 (a prompt rejects
    /// nothing), and `ctx_slot = block_id * block_size + pos % block_size` for every one of them.
    #[test]
    fn the_prompts_context_stores_every_row_once_at_its_own_position() {
        let g = geom();
        let blocks = pages(32);
        let tiles = DraftCtxTile::chunk(0, 274, 0, &blocks, g).expect("a 274-row prompt tiles");
        assert_eq!(tiles.len(), 35, "ceil(274 / 8)");
        let mut seen: Vec<i64> = Vec::new();
        for (i, t) in tiles.iter().enumerate() {
            assert_eq!(t.positions.len(), 8, "a tile is the plan's block, always");
            assert_eq!(t.slots.len(), 8);
            assert_eq!(
                t.row0,
                i as u32 * 8,
                "tile {i} reads its own rows of `aux.cat`"
            );
            let live = if i == 34 { 2 } else { 8 };
            assert_eq!(t.live, live);
            for j in 0..8usize {
                if j < live as usize {
                    let pos = (i * 8 + j) as i64;
                    assert_eq!(t.positions[j], pos);
                    let page = i64::from(blocks[(pos / 16) as usize]);
                    assert_eq!(t.slots[j], page * 16 + pos % 16);
                    seen.push(pos);
                } else {
                    // THE CONTROL IS THE OTHER LAW: a tile that wrote all eight rows live would
                    // give row 2 of the last tile position 274 and a real slot, and store the
                    // drafter's K/V for a token the prompt does not contain.
                    assert_eq!(t.slots[j], PAD_SLOT, "tile {i} row {j}");
                    let would_be = 274i64 + (j as i64 - live as i64);
                    assert_ne!(
                        t.slots[j],
                        i64::from(blocks[(would_be / 16) as usize]) * 16 + would_be % 16
                    );
                }
            }
        }
        assert_eq!(seen, (0..274).collect::<Vec<i64>>());
    }

    /// A second chunk of the same prompt starts where the first ended, and its `aux.cat` rows
    /// start at 0 again — the tap writes one chunk at a time.
    #[test]
    fn a_later_chunk_addresses_its_own_absolute_positions_from_row_zero() {
        let g = geom();
        let blocks = pages(32);
        let tiles = DraftCtxTile::chunk(256, 18, 0, &blocks, g).expect("chunk 2 tiles");
        assert_eq!(tiles.len(), 3);
        assert_eq!(
            tiles[0].row0, 0,
            "`aux.cat` is this chunk's, not the prompt's"
        );
        assert_eq!(tiles[0].positions[0], 256);
        assert_eq!(tiles[0].slots[0], i64::from(blocks[16]) * 16);
        assert_eq!(tiles[2].live, 2);
        assert_eq!(tiles[2].positions[1], 273);
    }

    /// The drafter's context store and the DECODE law meet at the seam: the first decode step's
    /// context row 0 is the position after the prompt's last.
    #[test]
    fn the_prompt_and_the_first_decode_step_meet_with_no_gap_and_no_overlap() {
        let g = geom();
        let blocks = pages(32);
        let tiles = DraftCtxTile::chunk(0, 274, 0, &blocks, g).expect("the prompt tiles");
        let last = tiles
            .iter()
            .flat_map(|t| t.positions[..t.live as usize].iter().copied())
            .max()
            .expect("a prompt row");
        let step = DraftSeqStep::decode(274, 8, 1, 7, &blocks, 0).expect("the first decode step");
        let ix = DraftStepIndex::build(std::slice::from_ref(&step), g).expect("the tables");
        assert_eq!(last, 273);
        assert_eq!(ix.context_positions[0], 274);
        assert_eq!(ix.context_slots[0], i64::from(blocks[17]) * 16 + 2);
    }

    /// **The FIRST proposal of a request (ws-s9), and the split that makes it free.**
    ///
    /// Prod proposes after every target step, the last prefill chunk included, so its first decode
    /// step already verifies a block. The engine reaches the same state by handing the chunk's
    /// LAST TILE to the proposal instead of storing it in the tile loop — `propose_block`'s eager
    /// phase IS the context store — and that is only sound if the tables the proposal builds for
    /// those rows are the tile's own, entry for entry. This is that claim.
    #[test]
    fn the_first_proposal_stores_the_chunks_last_tile_and_anchors_on_its_sampled_token() {
        let g = geom();
        let blocks = pages(32);
        let tiles = DraftCtxTile::chunk(0, 274, 0, &blocks, g).expect("the prompt tiles");
        let tail = tiles.last().expect("a tile");
        assert_eq!(tail.row0, 272);
        assert_eq!(tail.live, 2);
        // The step the engine builds out of that tile: `done + row0` rows, of which `live` are
        // valid. `anchor` is the token the chunk's head sampled — prod's `last_sampled`.
        let step = DraftSeqStep::decode(tail.row0, g.block, tail.live, 9764, &blocks, 0)
            .expect("the first proposal");
        let ix = DraftStepIndex::build(std::slice::from_ref(&step), g).expect("the tables");
        // THE CONTEXT ROWS ARE THE TILE'S, entry for entry — positions AND slots, pad included.
        assert_eq!(ix.context_positions, tail.positions);
        assert_eq!(ix.context_slots, tail.slots);
        // THE QUERY ROWS FOLLOW THE PROMPT'S LAST TOKEN, and row 0 carries the sampled token.
        assert_eq!(ix.query_positions[0], 274);
        assert_eq!(ix.input_ids[0], 9764);
        assert!(ix.input_ids[1..g.block as usize]
            .iter()
            .all(|t| *t == MASK_TOKEN as i32));
        // `seq_lens` is `last_valid_pos + 1 + block`, the same law a decode step runs.
        assert_eq!(ix.seq_lens[0], 274 + g.block as i32);
        assert_eq!(ix.sample_pos[0], 275);
        assert_eq!(ix.sample_pos[g.steps() as usize - 1], 281);
        // THE CONTROL: the chunk's FIRST position instead of the last tile's, which is the
        // arithmetic a caller that read `done` rather than `done + row0` would run. It addresses
        // the prompt's opening rows and drafts from position 8, not 274.
        let wrong = DraftSeqStep::decode(0, g.block, tail.live, 9764, &blocks, 0).expect("control");
        let bad = DraftStepIndex::build(std::slice::from_ref(&wrong), g).expect("the tables");
        assert_ne!(bad.context_positions, tail.positions);
        assert_ne!(bad.query_positions[0], 274);
    }

    /// A chunk whose rows do not fill one tile still proposes, and the tile loop stores nothing.
    #[test]
    fn a_prompt_shorter_than_one_tile_is_the_whole_of_its_own_first_proposal() {
        let g = geom();
        let blocks = pages(32);
        let tiles = DraftCtxTile::chunk(0, 3, 0, &blocks, g).expect("a three-row prompt");
        assert_eq!(tiles.len(), 1);
        let tail = &tiles[0];
        assert_eq!((tail.row0, tail.live), (0, 3));
        let step =
            DraftSeqStep::decode(tail.row0, g.block, tail.live, 42, &blocks, 0).expect("the step");
        let ix = DraftStepIndex::build(std::slice::from_ref(&step), g).expect("the tables");
        assert_eq!(ix.context_positions, tail.positions);
        assert_eq!(ix.context_slots, tail.slots);
        assert_eq!(ix.query_positions[0], 3);
        assert_eq!(ix.seq_lens[0], 3 + g.block as i32);
    }

    /// A position with no drafter page is a REFUSAL, not a clamp: a clamped page id is another
    /// session's K/V, in range and silent.
    #[test]
    fn a_prompt_longer_than_its_drafter_pages_is_refused_by_name() {
        let g = geom();
        // Four pages hold 64 positions; a 274-token prompt needs eighteen.
        let e = DraftCtxTile::chunk(0, 274, 0, &pages(4), g).expect_err("a short page run");
        assert!(e.to_string().contains("drafter page"), "{e}");
        assert!(DraftCtxTile::chunk(0, 0, 0, &pages(4), g).is_err());
        let narrow = DraftGeometry { block: 1, ..g };
        assert!(DraftCtxTile::chunk(0, 8, 0, &pages(4), narrow).is_err());
        let nopage = DraftGeometry { page: 0, ..g };
        assert!(DraftCtxTile::chunk(0, 8, 0, &pages(4), nopage).is_err());
    }

    /// The frame the follower is handed: a round trip, and the two controls that make it a pair of
    /// binaries rather than a short table.
    #[test]
    fn a_context_frame_round_trips_and_a_frame_of_another_shape_is_refused() {
        let g = geom();
        let blocks = pages(32);
        let tiles = DraftCtxTile::chunk(0, 274, 0, &blocks, g).expect("the prompt tiles");
        let f = tiles[34].pack(true);
        assert_eq!(f.len(), DraftCtxTile::frame_bytes(8));
        let (got, more) = DraftCtxTile::unpack(&f, 8).expect("the frame reads back");
        assert!(more);
        assert_eq!(got.positions, tiles[34].positions);
        assert_eq!(got.slots, tiles[34].slots);
        assert_eq!(got.row0, tiles[34].row0);
        assert_eq!(got.live, tiles[34].live);
        // The terminator says no tile follows, and it says so with a table that stores nothing.
        let (end, more) = DraftCtxTile::unpack(&DraftCtxTile::terminator(8), 8).expect("the end");
        assert!(!more);
        assert_eq!(end.live, 0);
        assert!(end.slots.iter().all(|s| *s == PAD_SLOT));
        // CONTROL 1: a rank whose drafter is a different block reads its own width and refuses.
        assert!(DraftCtxTile::unpack(&f, 4).is_err());
        // CONTROL 2: a frame the right length for a block of 4 but written for a block of 8.
        let short = DraftCtxTile {
            positions: vec![0; 4],
            slots: vec![PAD_SLOT; 4],
            row0: 0,
            live: 4,
        }
        .pack(true);
        assert_eq!(short.len(), DraftCtxTile::frame_bytes(4));
        assert!(DraftCtxTile::unpack(&short, 8).is_err());
    }

    fn wide() -> StoreTiling {
        StoreTiling {
            wide: 256,
            min_partial: 64,
            aux_rows: 2048,
            class_rows: 256,
            natural_min: 0,
        }
    }

    /// **C6: a chunk below the class boundary as ONE wide tile at its own class.** Every row once,
    /// at main's position and slot, the held tile main's last; the tile travels as
    /// `TileWideNatural`; below `natural_min` (and with the option off) the chunk is main's
    /// narrow tiles; a chunk at or above the boundary is H35's schedule unchanged.
    #[test]
    fn a_chunk_below_the_class_boundary_is_one_wide_tile_at_its_own_class() {
        let g = geom();
        let blocks = pages(256);
        let nat = StoreTiling {
            natural_min: 32,
            ..wide()
        };
        for (rows, want_wide) in [
            (72u32, 1usize),
            (255, 1),
            (41, 1),
            (39, 1),
            (20, 0),
            (31, 0),
        ] {
            let main = DraftCtxTile::chunk(0, rows, 0, &blocks, g).expect("main");
            let s = DraftCtxTile::schedule(0, rows, 0, &blocks, g, None, true, nat).expect("s");
            let off =
                DraftCtxTile::schedule(0, rows, 0, &blocks, g, None, true, wide()).expect("o");
            assert_eq!(s.wide_tiles(8), want_wide, "p{rows}");
            assert_eq!(s.natural_wide, want_wide > 0, "p{rows}");
            assert!(!s.chunk_class, "p{rows}: below the boundary");
            assert_eq!(
                off.wide_tiles(8),
                0,
                "p{rows}: the option off is main's law"
            );
            assert_eq!(s.held, off.held, "p{rows}: the held tile does not move");
            assert_eq!(
                s.held.as_ref(),
                main.last(),
                "p{rows}: main's last tile is held"
            );
            assert_eq!(
                rows_of(&s),
                rows_of(&off),
                "p{rows}: every row once, main's slots"
            );
            for t in &s.stored {
                let k = s.kind_of(t, 8);
                if t.width() == 256 {
                    assert_eq!(k, CtxFrame::TileWideNatural, "p{rows}");
                    assert!(t.slots[t.live as usize..].iter().all(|x| *x == PAD_SLOT));
                    let f = t.pack_wire(k, 256).expect("fits");
                    let (back, kind) =
                        DraftCtxTile::unpack_wire(&f, 256, &[8, 256]).expect("reads");
                    assert_eq!((&back, kind), (t, CtxFrame::TileWideNatural));
                } else {
                    assert_eq!(k, CtxFrame::Tile, "p{rows}");
                }
            }
        }
        // AT OR ABOVE THE BOUNDARY nothing moves: H35's chunk-class schedule, byte for byte.
        for rows in [256u32, 300, 1300, 2048] {
            let a = DraftCtxTile::schedule(0, rows, 0, &blocks, g, None, true, nat).expect("a");
            let b = DraftCtxTile::schedule(0, rows, 0, &blocks, g, None, true, wide()).expect("b");
            assert_eq!(a, b, "p{rows}");
        }
        // The window: a wide tile past `aux.cat` is never taken.
        let tight = StoreTiling {
            aux_rows: 200,
            ..nat
        };
        let t = DraftCtxTile::schedule(0, 72, 0, &blocks, g, None, true, tight).expect("t");
        assert_eq!(t.wide_tiles(8), 0);
    }

    /// Every (position, slot) a schedule stores, in row order, with the held tile's LAST.
    fn rows_of(s: &StoreSchedule) -> Vec<(i64, i64)> {
        s.stored
            .iter()
            .chain(s.held.iter())
            .flat_map(|t| (0..t.live as usize).map(move |j| (t.positions[j], t.slots[j])))
            .collect()
    }

    /// **The wide store's schedule (H35)** over the chunk sizes the brief names — 2048, 1300, 8
    /// and 1 rows, plus 1400 (the partial wide tile) — with and without the held-back last tile.
    ///
    /// The law, at every size: every row is stored EXACTLY ONCE at main's own position and slot;
    /// the held tile IS main's last tile; the narrow tiles of a remainder ARE main's tiles over
    /// those rows; a wide tile is 256 rows wide whatever its live count.
    #[test]
    fn the_wide_schedule_stores_main_s_rows_once_and_holds_main_s_last_tile() {
        let g = geom();
        let blocks = pages(512);
        // (rows, hold, wide tiles, narrow tiles, partial wide live, held (row0, live))
        #[allow(clippy::type_complexity)]
        let cases: [(u32, bool, usize, usize, Option<u32>, Option<(u32, u32)>); 12] = [
            (2048, false, 8, 0, None, None),
            (2048, true, 8, 0, Some(248), Some((2040, 8))),
            (1300, false, 5, 3, None, None),
            (1300, true, 5, 2, None, Some((1296, 4))),
            (1400, true, 6, 0, Some(112), Some((1392, 8))),
            (1400, false, 6, 0, Some(120), None),
            (8, false, 0, 1, None, None),
            (8, true, 0, 0, None, Some((0, 8))),
            (1, false, 0, 1, None, None),
            (1, true, 0, 0, None, Some((0, 1))),
            (200, false, 0, 25, None, None),
            (200, true, 0, 24, None, Some((192, 8))),
        ];
        for (rows, hold, nw, nn, partial, held) in cases {
            let first = 4096u32; // a later chunk: absolute positions from 4096
            let s = DraftCtxTile::schedule(first, rows, 0, &blocks, g, None, hold, wide())
                .expect("the schedule");
            let main = DraftCtxTile::chunk(first, rows, 0, &blocks, g).expect("main's tiles");
            let tag = format!("{rows} rows, hold {hold}");
            assert_eq!(s.wide_tiles(8), nw, "{tag}: wide tiles");
            assert_eq!(s.stored.len() - nw, nn, "{tag}: narrow tiles");
            assert!(
                s.stored.iter().all(|t| t.width() == 8 || t.width() == 256),
                "{tag}"
            );
            let partials: Vec<u32> = s
                .stored
                .iter()
                .filter(|t| t.width() == 256 && t.live < 256)
                .map(|t| t.live)
                .collect();
            assert_eq!(partials, partial.into_iter().collect::<Vec<_>>(), "{tag}");
            assert_eq!(s.held.as_ref().map(|t| (t.row0, t.live)), held, "{tag}");
            // THE CLASS: a chunk of 256 rows or more is stored at the chunk's norm class, a
            // shorter one narrow at its natural class with no wide tile at all.
            assert_eq!(s.chunk_class, rows >= 256, "{tag}: the chunk class");
            for t in &s.stored {
                let k = s.kind_of(t, 8);
                let want = if rows >= 256 && t.width() == 8 {
                    CtxFrame::TileChunkClass
                } else {
                    CtxFrame::Tile
                };
                assert_eq!(k, want, "{tag}: row0 {}", t.row0);
            }
            // THE HELD TILE IS MAIN'S LAST, entry for entry — the first proposal's context rows.
            if let Some(h) = &s.held {
                assert_eq!(h, main.last().expect("a tile"), "{tag}: the held tile");
            }
            // EVERY ROW ONCE, at main's position and slot.
            let want: Vec<(i64, i64)> = main
                .iter()
                .flat_map(|t| (0..t.live as usize).map(move |j| (t.positions[j], t.slots[j])))
                .collect();
            assert_eq!(rows_of(&s), want, "{tag}: the rows");
            assert_eq!(
                s.stored_rows() + s.held.as_ref().map_or(0, |h| h.live),
                rows
            );
            // A NARROW tile of the schedule is one of main's, whole.
            for t in s.stored.iter().filter(|t| t.width() == 8) {
                assert!(
                    main.contains(t),
                    "{tag}: narrow tile at {} is main's",
                    t.row0
                );
            }
            // The pad rows of a wide tile land nowhere.
            for t in s.stored.iter().filter(|t| t.width() == 256) {
                assert!(t.slots[t.live as usize..].iter().all(|x| *x == PAD_SLOT));
            }
        }
        // THE LAUNCHES a 2048-row chunk costs: 12 a tile, 256 tiles on main's law, 8 here.
        let narrow =
            DraftCtxTile::schedule(0, 2048, 0, &blocks, g, None, false, StoreTiling::narrow())
                .expect("main's schedule");
        let w = DraftCtxTile::schedule(0, 2048, 0, &blocks, g, None, false, wide()).expect("wide");
        assert_eq!((narrow.launches(12), w.launches(12)), (3072, 96));
        let last =
            DraftCtxTile::schedule(0, 2048, 0, &blocks, g, None, true, wide()).expect("last");
        assert_eq!(
            last.launches(12),
            96,
            "7 whole + 1 partial wide tile, the last 8 rows held"
        );
    }

    /// THE CONTROLS of the partial tile: below `min_partial` the remainder stays narrow, and a
    /// wide window that would run past `aux.cat` is never taken — the remainder goes narrow.
    #[test]
    fn a_partial_wide_tile_needs_its_threshold_and_a_window_inside_aux_cat() {
        let g = geom();
        let blocks = pages(256);
        let fits = DraftCtxTile::schedule(0, 1400, 0, &blocks, g, None, true, wide()).expect("ok");
        assert_eq!(fits.wide_tiles(8), 6);
        let tight = StoreTiling {
            aux_rows: 1400,
            ..wide()
        };
        let past = DraftCtxTile::schedule(0, 1400, 0, &blocks, g, None, true, tight).expect("ok");
        assert_eq!(
            past.wide_tiles(8),
            5,
            "the 1280..1536 window is past a 1400-row aux.cat"
        );
        assert_eq!(past.stored.len(), 5 + 14, "the remainder's 112 rows narrow");
        let off = StoreTiling {
            min_partial: 0,
            ..wide()
        };
        let none = DraftCtxTile::schedule(0, 1400, 0, &blocks, g, None, true, off).expect("ok");
        assert_eq!(none.wide_tiles(8), 5);
        assert_eq!(rows_of(&fits), rows_of(&past));
        assert_eq!(rows_of(&fits), rows_of(&none));
        // A wide width that is not whole blocks is refused by name.
        let bad = StoreTiling {
            wide: 100,
            ..wide()
        };
        assert!(DraftCtxTile::schedule(0, 1400, 0, &blocks, g, None, true, bad).is_err());
    }

    /// **The frame on the wide wire (H35)**: every frame is the widest shape's length, the header
    /// names the tile's own width and kind, and a narrow tile, a wide one and each marker round
    /// trip — with a width, a live count and a kind this rank does not plan refused by name.
    #[test]
    fn the_wide_wire_round_trips_every_tile_and_marker_and_refuses_the_rest() {
        let g = geom();
        let blocks = pages(256);
        let s = DraftCtxTile::schedule(0, 1300, 0, &blocks, g, None, true, wide()).expect("s");
        let widths = [8u32, 256];
        for t in s.stored.iter().chain(s.held.iter()) {
            let k = s.kind_of(t, 8);
            let f = t.pack_wire(k, 256).expect("fits");
            assert_eq!(f.len(), DraftCtxTile::frame_bytes(256));
            let (back, kind) = DraftCtxTile::unpack_wire(&f, 256, &widths).expect("reads");
            assert_eq!(kind, k);
            assert_eq!(&back, t, "row0 {} width {}", t.row0, t.width());
        }
        assert!(
            s.stored
                .iter()
                .any(|t| s.kind_of(t, 8) == CtxFrame::TileChunkClass),
            "a 1300-row chunk's narrow tiles travel at the chunk's class"
        );
        for k in [
            CtxFrame::End,
            CtxFrame::CheckSnap,
            CtxFrame::CheckCompare,
            CtxFrame::CheckPoison,
            CtxFrame::CheckResidual,
        ] {
            let f = DraftCtxTile::marker(k, 8, 256);
            let (back, kind) = DraftCtxTile::unpack_wire(&f, 256, &widths).expect("a marker");
            assert_eq!((kind, back.live), (k, 0));
        }
        // CONTROL 1: a width this rank did not plan.
        let odd = DraftCtxTile {
            positions: vec![0; 128],
            slots: vec![PAD_SLOT; 128],
            row0: 0,
            live: 1,
        };
        let f = odd.pack_wire(CtxFrame::Tile, 256).expect("fits the wire");
        assert!(DraftCtxTile::unpack_wire(&f, 256, &widths).is_err());
        // CONTROL 2: a tile wider than the wire is not packed at all.
        assert!(s.stored[0].pack_wire(CtxFrame::Tile, 8).is_err());
        // CONTROL 3: a live count past the width, and a kind the protocol does not have.
        let mut f = s.stored[0].pack_wire(CtxFrame::Tile, 256).expect("fits");
        f[4..8].copy_from_slice(&257u32.to_le_bytes());
        assert!(DraftCtxTile::unpack_wire(&f, 256, &widths).is_err());
        let mut f = s.stored[0].pack_wire(CtxFrame::Tile, 256).expect("fits");
        f[8..12].copy_from_slice(&9u32.to_le_bytes());
        assert!(DraftCtxTile::unpack_wire(&f, 256, &widths).is_err());
        // CONTROL 4: a narrow-wire rank (main's) reading a wide-wire frame refuses on length, and
        // the tiles-only reader refuses a check marker.
        let f = s.stored[0].pack_wire(CtxFrame::Tile, 256).expect("fits");
        assert!(DraftCtxTile::unpack(&f, 8).is_err());
        let m = DraftCtxTile::marker(CtxFrame::CheckSnap, 8, 8);
        assert!(DraftCtxTile::unpack(&m, 8).is_err());
    }
}

#[cfg(test)]
mod window_law {
    use super::*;

    fn geom() -> DraftGeometry {
        DraftGeometry {
            block: 8,
            page: 16,
            max_reqs: 4,
            max_pos: 4096,
            mask_token: MASK_TOKEN,
        }
    }

    /// Walk one request's life and return the positions the drafter READ without this request
    /// having written them first.
    ///
    /// `store_prompt` is ws-s8's context store over the prompt; turning it off is that defect back
    /// again and is this law's control.
    fn holes(prompt: u32, accepts: &[u32], store_prompt: bool) -> Vec<u32> {
        let g = geom();
        let blocks: Vec<u32> = (0..256).collect();
        let mut written: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        let mut holes = Vec::new();
        // THE PROMPT, stored in `block`-row tiles by the prefill's own phase (ws-s8).
        if store_prompt {
            for t in DraftCtxTile::chunk(0, prompt, 0, &blocks, g).expect("tiles") {
                for j in 0..t.live as usize {
                    written.insert(t.positions[j] as u32);
                }
            }
        }
        // The prompt's own proposal (one row, the chunk's last), then a decode step per entry.
        let mut done = prompt;
        for (n, committed) in accepts.iter().copied().enumerate() {
            let rows = if n == 0 { 1 } else { g.block };
            let committed = committed.min(rows).max(1);
            let step = DraftSeqStep::decode(done - 1, rows, committed, 7 as TokenId, &blocks, 0)
                .expect("a step");
            let ix = DraftStepIndex::build(std::slice::from_ref(&step), g).expect("tables");
            let last_valid_pos = step.positions[step.valid() as usize - 1];
            let seq_len = ix.seq_lens[0] as u32;
            assert_eq!(
                seq_len,
                last_valid_pos + 1 + g.block,
                "the window is the same law at both ends"
            );
            // What this proposal writes before its backbone reads anything: its own context rows,
            // and then its `block` query rows.
            let mut writes = written.clone();
            for j in 0..step.valid() as usize {
                writes.insert(step.positions[j]);
            }
            for off in 0..g.block {
                writes.insert(last_valid_pos + 1 + off);
            }
            for pos in 0..seq_len {
                if !writes.contains(&pos) {
                    holes.push(pos);
                }
            }
            written = writes;
            // The next step feeds the token this one committed last and runs `block` rows from
            // there — the engine's own accounting.
            done = last_valid_pos + 2;
        }
        holes
    }

    /// **WHY `clear_pages`' ARM IS RETIRED** (ws-p9, over ws-s8's prefill context store).
    ///
    /// The memset exists so that a proposal is not a function of what the PREVIOUS session left in
    /// this session's drafter pages. The only place that can happen is a position the drafter
    /// READS and this request never WRITES — and there is none: the window is
    /// `last_valid_pos + 1 + block`, the context store covers every position up to
    /// `last_valid_pos` (the prompt's on the prefill steps, each committed token's on the step
    /// that committed it) and the proposal's own `block` query rows cover the rest, all before the
    /// backbone reads any of them. Nothing at or past `seq_lens` is read at all, so a predecessor
    /// that wrote THERE cannot be seen either.
    ///
    /// So the gate arm's control could not fail (ws-s8 measured exactly that on the card) and the
    /// claim lives here, as a law over the tables rather than two runs that agree.
    #[test]
    fn every_position_the_drafter_reads_is_written_by_this_request_first() {
        for prompt in [1u32, 7, 8, 9, 41, 274, 1000] {
            for accepts in [&[1u32, 1, 1, 1][..], &[1, 8, 3, 8][..], &[1, 2, 2, 2][..]] {
                let h = holes(prompt, accepts, true);
                assert!(
                    h.is_empty(),
                    "prompt {prompt}, accepts {accepts:?}: the drafter read {h:?} unwritten"
                );
            }
        }
    }

    /// THE CONTROL, and it is ws-s8's own defect: with the prompt's context store gone the drafter
    /// reads a prompt's worth of positions nothing wrote — which is what `clear_pages` turned into
    /// zeros and what the arm's control was trying to catch.
    #[test]
    fn the_decode_only_store_leaves_the_prompt_unwritten() {
        let h = holes(274, &[1, 8, 3, 8], false);
        assert!(
            h.len() >= 200,
            "the decode-only store must leave the prompt's positions unwritten, not {}",
            h.len()
        );
        assert_eq!(h[0], 0, "the hole starts at the prompt's first position");
    }
}
