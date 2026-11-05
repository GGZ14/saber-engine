//! What the scheduler is allowed to do, derived from the resolved plan.
//!
//! Rule 4 of `CLAUDE.md` is that behaviour comes from the [`ExecutionPlan`] and nowhere else. The
//! scheduler needs four things from it — the captured bucket set, the tensor-parallel width, the
//! GDN state bytes one sequence slot costs per rank, and the context bands — and every one of them
//! is read here, once, at construction.
//!
//! The KV **geometry** (page size in tokens, bytes per token per rank) is not a plan field today:
//! the plan sizes the GDN state budget and binds the routes, but it states KV only as the
//! `block_size` selection key on the paged-attention row. It is therefore passed in as
//! [`KvGeometry`], typed and documented, with prod's numbers available as [`KvGeometry::PROD_FP8`]
//! — and **cross-checked**: a page size that disagrees with the bound attention route is
//! [`ConfigError::KvBlockMismatch`], because the block table this scheduler builds is the one that
//! kernel indexes. When `saber-plan` grows a KV section this struct becomes a projection of it and
//! nothing else in this crate changes.

use saber_plan::{Bucket, ExecutionPlan, Phase, PlannedBucket};

use crate::park::ParkPolicy;
use crate::width::VerifyWidthPolicy;

/// 16 KiB per token per rank — fp8 KV for Qwen3.8-27B at TP=2, the number the 4.6k crossover in
/// `ARCHITECTURE.md` § "What exploration corrected" item 3 is computed from.
pub const PROD_KV_BYTES_PER_TOKEN_PER_RANK: u64 = 16 * 1024;

/// Paged-KV geometry: the page size the allocator hands out and what a token costs in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvGeometry {
    /// Tokens per KV block (page). Prod serves 16.
    pub block_tokens: u32,
    /// KV bytes one token holds on one rank, both K and V, at the configured KV dtype.
    pub bytes_per_token_per_rank: u64,
}

impl KvGeometry {
    /// Production's geometry: 16-token pages, fp8 KV, 16 KiB/token/rank.
    pub const PROD_FP8: KvGeometry = KvGeometry {
        block_tokens: 16,
        bytes_per_token_per_rank: PROD_KV_BYTES_PER_TOKEN_PER_RANK,
    };

    /// Bytes one block holds on one rank.
    pub fn block_bytes(&self) -> u64 {
        u64::from(self.block_tokens) * self.bytes_per_token_per_rank
    }

    /// Blocks a sequence of `tokens` tokens needs.
    pub fn blocks_for(&self, tokens: u64) -> u64 {
        tokens.div_ceil(u64::from(self.block_tokens))
    }
}

/// Bytes the scheduler's two arenas hold on **each** rank. Reserved through the byte ledger at
/// construction, so an over-commit is a startup error rather than an OOM at token 4000.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaBytes {
    /// The paged KV arena, per rank.
    pub kv_per_rank: u64,
    /// The GDN state arena, per rank. Separate from KV on purpose: coupling the mamba page to the
    /// attention block size is what capped TP=1 concurrency at 3 in the reference server (09-16).
    pub gdn_state_per_rank: u64,
}

/// Why a plan could not be turned into a scheduler configuration.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ConfigError {
    /// The plan has no captured bucket. A scheduler that replays uncaptured shapes is exactly the
    /// bug the bucket set exists to prevent.
    #[error("plan {plan_hash} has no captured bucket; the scheduler only replays captured shapes")]
    NoCapturedBucket {
        /// The plan.
        plan_hash: String,
    },
    /// Two buckets disagree about what one GDN state slot costs. The allocator is sized once, so
    /// this has to hold across the whole bucket set.
    #[error(
        "buckets {a} and {b} disagree on GDN state bytes per slot ({a_bytes} vs {b_bytes}); \
         the state arena is sized once for the whole bucket set"
    )]
    InconsistentGdnState {
        /// First bucket.
        a: String,
        /// Its bytes per slot.
        a_bytes: u64,
        /// Second bucket.
        b: String,
        /// Its bytes per slot.
        b_bytes: u64,
    },
    /// A bucket reports zero slots or zero state bytes.
    #[error(
        "bucket {bucket} reports a degenerate GDN state budget ({slots} slots, {bytes} bytes)"
    )]
    DegenerateGdnState {
        /// The bucket.
        bucket: String,
        /// Slots it reported.
        slots: u32,
        /// Bytes it reported.
        bytes: u64,
    },
    /// The KV geometry is unusable.
    #[error("kv geometry is degenerate: block_tokens={block_tokens}, bytes_per_token={bytes}")]
    DegenerateKv {
        /// Tokens per block.
        block_tokens: u32,
        /// Bytes per token per rank.
        bytes: u64,
    },
    /// The KV page size disagrees with the one the plan's attention route is bound at. The block
    /// table the scheduler builds is read by that kernel; two page sizes is a wrong-address bug,
    /// not a configuration preference.
    #[error(
        "kv geometry says {kv_block_tokens} tokens per page, but op {op_name} is bound at \
         block_size={plan_block_size}"
    )]
    KvBlockMismatch {
        /// The op that declares the page size.
        op_name: String,
        /// What the plan says.
        plan_block_size: i64,
        /// What the caller passed.
        kv_block_tokens: u32,
    },
}

/// How a prompt is cut into prefill steps — **the rule, as data**.
///
/// The sizes are the plan's captured prefill buckets, ascending, and the rule over them is one
/// sentence: **a chunk is as much of what is left as the smallest fitting bucket will take.**
///
/// ```text
///     chunk   = min(remaining, largest captured prefill bucket)
///     bucket  = the smallest captured prefill bucket >= chunk
/// ```
///
/// With prod's `p2048` and `p8192` that means a 234-token prompt is one chunk in `p2048`, a
/// 5000-token prompt is one chunk in `p8192` (nothing is gained by cutting a prompt that already
/// fits a captured shape), and a 12000-token prompt is `8192 + 3808` — two steps, with the KV
/// pages and the GDN state carried from the first to the second.
///
/// The alternative — always cut at the *smallest* bucket — was rejected here rather than in a
/// comment: it turns one 5000-row GEMM into three, and prefill is the phase where `M` is the
/// whole point (`docs/LEDGER.md`, A-tiled prefill: the win is −12..−16% at `M >= 2048`).
///
/// # The fill rule (C7, `fill_above`; OFF = 0, the pairing control)
///
/// G4 measured the padding law's price above 2048 rows: with a `4096`/`8192` rung a 4,000-row
/// prompt is ONE 8192-row chunk (+76 % TTFT) where the 2048 law runs two. With `fill_above = F`
/// every rung ABOVE `F` is **fill-only**: it is taken only by a chunk that fills it, and a prompt
/// is cut into the largest rungs it fills before the padded law takes the tail:
///
/// ```text
///     P = the largest rung <= F                          (the padded law's own maximum)
///     remaining >  P : chunk = bucket = the largest rung <= remaining   (filled, no padding)
///     remaining <= P : chunk = remaining, bucket = the smallest rung >= chunk   (padded, as before)
/// ```
///
/// So at `…,2048,4096,8192` with `F = 2048`: 6,000 = 4096 + 1904 (p2048), 16,384 = 8192 × 2,
/// 32,000 = 8192 × 3 + 4096 + 2048 + 1280 (p1536). On a rung list whose top is `<= F` the rule
/// IS the padded law (`P` = the top), which is the control the tests pin. A chunk boundary is not
/// bit-neutral on the sums (G4), so a rung list + fill rule that re-cuts a prompt is a numerics
/// change for every prompt above `P` rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefillChunking {
    /// Captured prefill chunk sizes, ascending.
    pub sizes: Vec<u32>,
    /// C7: rungs above this many rows are fill-only (see the type's doc). 0 = off: every rung is
    /// padded into, the pre-C7 law.
    pub fill_above: u32,
    /// FN38: the chunk cap while some sequence DECODES (0 = off: every chunk may take the top
    /// rung). Flash-Next's wide rung (4,096 / 8,192 rows) pays the link's RAM-expert set once a
    /// chunk; beside decode a chunk keeps this many rows so a decode step never waits behind a
    /// wide one (`--fn-wide-when idle`). The cut then depends on the arrival pattern -- a
    /// numerics statement for the rows past the first cut (the chunk law), not a correctness one.
    pub busy_max: u32,
    /// LR-B (`--fn-prefill-yield`, an ARM, 0 = off = today's prefill-first schedule): decode steps
    /// the scheduler runs between two prefill chunks while some sequence is ready to decode. With
    /// 0 a prompt's chunks run back to back and every decoder waits the WHOLE prompt; with `n` a
    /// decoder waits at most one chunk. The chunk sequence (the cut) is unchanged, and each step is
    /// an ordinary unmixed step (disjoint sequences), so no row's bytes move.
    pub yield_steps: u32,
}

impl PrefillChunking {
    /// The rule over a plan's captured prefill buckets (the padded law; `fill_above` 0).
    pub fn from_sizes(mut sizes: Vec<u32>) -> Self {
        sizes.sort_unstable();
        sizes.dedup();
        Self {
            sizes,
            fill_above: 0,
            busy_max: 0,
            yield_steps: 0,
        }
    }

    /// LR-B: `n` decode steps between two prefill chunks while something decodes (0 = off).
    #[must_use]
    pub fn with_yield_steps(mut self, n: u32) -> Self {
        self.yield_steps = n;
        self
    }

    /// FN38: the same rungs with a chunk cap while some sequence decodes (0 = off).
    #[must_use]
    pub fn with_busy_max(mut self, rows: u32) -> Self {
        self.busy_max = rows;
        self
    }

    /// FN38: the most rows a chunk may feed this step: the busy cap when something decodes (and
    /// the cap is on), else no cap.
    #[must_use]
    pub fn room(&self, decoding: bool) -> u32 {
        if decoding && self.busy_max > 0 {
            self.busy_max
        } else {
            u32::MAX
        }
    }

    /// The same rungs under the fill rule (C7): rungs above `rows` are taken only when filled.
    /// `0` turns it off.
    #[must_use]
    pub fn with_fill_above(mut self, rows: u32) -> Self {
        self.fill_above = rows;
        self
    }

    /// Whether the plan captured any prefill shape at all.
    pub fn is_empty(&self) -> bool {
        self.sizes.is_empty()
    }

    /// The largest chunk a step may feed.
    pub fn max_chunk(&self) -> u32 {
        self.sizes.last().copied().unwrap_or(0)
    }

    /// The padded law's maximum: the largest rung a chunk may be PADDED into. The top rung with
    /// the fill rule off; with it on, the largest rung at or below `fill_above` (the smallest rung
    /// when every rung is above it, so a short tail still has a bucket).
    pub fn padded_max(&self) -> u32 {
        if self.fill_above == 0 {
            return self.max_chunk();
        }
        self.sizes
            .iter()
            .copied()
            .filter(|t| *t <= self.fill_above)
            .max()
            .or_else(|| self.sizes.first().copied())
            .unwrap_or(0)
    }

    /// `(chunk, bucket)` for `remaining` prompt tokens: how many this step feeds, and the bucket
    /// it runs in. `None` when the plan captured no prefill shape.
    pub fn chunk_for(&self, remaining: u32) -> Option<(u32, u32)> {
        if self.sizes.is_empty() || remaining == 0 {
            return None;
        }
        let padded = self.padded_max();
        if self.fill_above > 0 && remaining > padded {
            // A FILLED chunk: the largest rung the rest of the prompt fills. `padded` itself is
            // one (it is <= remaining), so this always finds a rung.
            let b = self
                .sizes
                .iter()
                .copied()
                .rev()
                .find(|t| *t <= remaining)
                .unwrap_or(padded);
            return Some((b, b));
        }
        let chunk = remaining.min(padded);
        let bucket = self
            .sizes
            .iter()
            .copied()
            .find(|t| *t >= chunk)
            .unwrap_or(padded);
        Some((chunk, bucket))
    }

    /// Every `(chunk, bucket)` a prompt of `tokens` is cut into, in order.
    pub fn cut(&self, tokens: u32) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        let mut n = tokens;
        while let Some((chunk, bucket)) = self.chunk_for(n) {
            out.push((chunk, bucket));
            n -= chunk;
        }
        out
    }

    /// How many steps a prompt of `tokens` takes. Used by tests and by `explain`-style output.
    pub fn steps_for(&self, tokens: u32) -> u32 {
        if self.max_chunk() == 0 || tokens == 0 {
            return 0;
        }
        u32::try_from(self.cut(tokens).len()).unwrap_or(u32::MAX)
    }
}

/// Everything the scheduler reads from the plan, plus the KV geometry and the arena sizes.
#[derive(Clone, Debug, PartialEq)]
pub struct SchedulerConfig {
    /// The plan this came from, for provenance in logs and `/metrics`.
    pub plan_hash: String,
    /// Captured **decode** buckets only, sorted by `(batch, spec_width, ctx_band)` — the order
    /// bucket selection walks. Prefill buckets are not replay candidates and are dropped at
    /// construction.
    pub buckets: Vec<Bucket>,
    /// Tensor-parallel ranks. Both arenas are reserved on every rank.
    pub tp: u32,
    /// GDN state bytes one sequence slot costs on one rank, all GDN layers:
    /// `3.0 MiB/layer/slot / tp * gdn_layers` = 72 MiB at TP=2.
    pub gdn_bytes_per_slot: u64,
    /// Paged KV geometry.
    pub kv: KvGeometry,
    /// Arena sizes, per rank.
    pub arenas: ArenaBytes,
    /// Captured **prefill** buckets, ascending by `tokens`. A prefill step replays one of these;
    /// they are kept apart from `buckets` because selection asks a different question of them
    /// (which chunk size, for one sequence) and because a prefill bucket would otherwise win the
    /// decode ladder's own ordering outright.
    pub prefill_buckets: Vec<Bucket>,
    /// How a prompt is cut into prefill steps.
    pub prefill: PrefillChunking,
    /// The dynamic-verify-width policy. Data, not a ladder — see [`crate::width`].
    pub width: VerifyWidthPolicy,
    /// Longest context the scheduler will serve: the largest ctx band in the bucket set, capped by
    /// the caller's `--max-model-len`.
    pub max_context_tokens: usize,
    /// FN51 (Flash-Next admission by reservation): the context a request WITHOUT `max_tokens` is
    /// clamped to (None = `max_context_tokens`, the reference server's rule). A prompt shorter than it generates
    /// up to it (the pre-FN51 served context's exact clamp); a longer prompt gets
    /// `implicit_headroom` tokens (within `max_context_tokens`). An explicit `max_tokens` keeps
    /// The reference server's rule. The reservation (prompt + max_tokens) is what admission holds, so this keeps a
    /// long-context boot's short requests reserving what they did before.
    pub implicit_context: Option<usize>,
    /// FN51: see `implicit_context`.
    pub implicit_headroom: usize,
    /// When the scheduler may park a running session by itself. Off by default: a tier gives the
    /// operator an explicit hook, and this is what turns it into an eviction policy.
    pub park: ParkPolicy,
    /// Whether a decode step may carry the queue head's next prefill chunk, and the step's token
    /// budget (C4, [`MixedPrefill`]). Off by default: the unmixed schedule is the pairing control.
    pub mixed: MixedPrefill,
    /// FN52: prefix (conversation) caching. Off by default — today's engine, the control.
    pub prefix_cache: crate::prefix_cache::PrefixCachePolicy,
    /// E4-3: the speculative tail a TREE verify writes past the chain's widest width (the 16-row
    /// form writes rows `anchor..anchor + 15`): admission reserves `max(max_spec_width,
    /// tree_reserve) + 1` tokens past `prompt + max_tokens`. 0 = no tree form (the chain's law).
    pub tree_reserve: u32,
    /// LR-F: the admission-side arms (all off by default = today's head-of-line schedule).
    pub admit: AdmitPolicy,
}

/// **LR-F: the admission-side arms.** Every field's zero / `false` is today's schedule, bit for
/// bit (the replay `examples/sched_replay` and `tests/lrf_admit_slack.rs` prove it).
///
/// * `slack_ms` (replay only -- LR-F NO-GO, no served flag, the step loop never holds): after a
///   step that FINISHED a sequence, with free seats,
///   an empty queue, nothing owing prefill and at least one decoder, the step loop may HOLD the next
///   step up to `min(slack_ms, the predicted decode step)` for an arrival to join it -- only when
///   the predicted gain beats the stall it costs the decoders (`Scheduler::admit_slack_hold`).
///   The hold is a hard cap per finish (one hold per finishing step, never chained), so nothing
///   can starve: decoders wait at most `slack_ms` a finish.
/// * `slack_force`: the hold without the gain test (the unconditional form) -- the replay's
///   diagnostic form, never a served default.
/// * `skip_kv_blocked`: a KV-refused queue head may be passed by up to this many later requests
///   whose reservation fits now (0 = strict head-of-line); a head is passed at most
///   [`AdmitPolicy::SKIP_LIMIT`] times, then the queue is strict again until it is admitted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdmitPolicy {
    /// The hold cap, ms (0 = off).
    pub slack_ms: u32,
    /// Hold without the gain test (diagnostic).
    pub slack_force: bool,
    /// Queue depth scanned past a KV-refused head (0 = strict head-of-line).
    pub skip_kv_blocked: u32,
    /// **LR-G: lazy KV growth** (`--fn-kv-lazy <headroom>`; 0 = off = today's whole reservation).
    /// A plain request (n 1, no images, no prompt logprobs, not a parked session) reserves
    /// `prompt + min(max_tokens, lazy_headroom) + spec + 1` at admission; a decoding sequence
    /// whose next step would write past its blocks takes the next block(s) BEFORE the step is
    /// built (`Core::grow_lazy`); a growth the arena refuses preempts the YOUNGEST lazy sequence
    /// (recompute: it goes back to the queue's front with its committed tokens as its prompt), so
    /// a running sequence never meets a full arena mid-step.
    pub lazy_headroom: u32,
    /// LR-G: tokens a lazy admission must leave free for the running sequences' growth (0 = none;
    /// waived when nothing runs).
    pub lazy_margin: u32,
}

impl AdmitPolicy {
    /// Times one queue head may be passed before the queue is strict again (the starvation cap).
    pub const SKIP_LIMIT: u32 = 4;

    /// LR-G: the lazy-growth arm is on.
    #[must_use]
    pub fn lazy(&self) -> bool {
        self.lazy_headroom > 0
    }
}

/// **The mixed step (C4): a decode step may carry ONE prefill chunk**, prod's continuous batching
/// reduced to the one rule this engine can serve.
///
/// Unmixed, a sequence that still owes prompt rows takes the step ALONE ("prefill first, and
/// alone") and every decoding sequence waits it out. Mixed, the step keeps its decode half and
/// ALSO feeds the chunk of the first prefilling sequence in admission order, so admission never
/// costs the running sequences a step of their own.
///
/// **The budget law** is prod's `max_num_batched_tokens` (`upstream/v1/core/sched/scheduler.py`:
/// running requests are scheduled first, then waiting prompts take what is left of the token
/// budget, chunked to fit):
///
/// ```text
///     room  = max_batched_tokens - decode rows (the decode bucket's M, padding included)
///     chunk = min(remaining prompt, largest captured prefill bucket, room)
///     bucket = the smallest captured prefill bucket >= chunk
/// ```
///
/// and when `room` is zero the step runs unmixed (its decode half only; the chunk waits for the
/// next step, which is prod's behaviour at a full budget too). Prod serves `8192`
/// (`max_num_batched_tokens=8192`, H34's side-container boot log), so the default is 8192, and
/// with this engine's `2048`-row largest chunk and 64-row widest decode bucket the budget binds
/// only when an operator lowers it. **One chunk, not prod's several**: prod packs every waiting
/// prompt that fits (C2's corpus: 1.30 chunks per prefill-carrying step, at most 7); a step here
/// carries the one at the queue head, and a second prompt waits one step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MixedPrefill {
    /// Whether a decode step may carry a chunk at all.
    pub enabled: bool,
    /// Rows one step may run: the decode bucket's rows plus the chunk's.
    pub max_batched_tokens: u32,
    /// **C9: the FUSED form** (`--engine-mixed-prefill fused`): the executor runs the chunk and
    /// the decode half as ONE forward at the smallest prefill rung holding both. **C11: its chunk
    /// comes from the packing law** ([`crate::mixed_law::pack`] at `max_chunks` = 1): the head's
    /// UNPACKED chunk, whole, so a long prompt's 2048-row chunk rides beside the decode rows at
    /// 2048 and the executor runs that step in the serial form (no in-class rung) — the chunk
    /// boundary never moves ([`crate::mixed_law::chunk_sequence_holds`]). `false` is C4's serial
    /// form (the bit-exact control), whose room is prod's law alone.
    pub fused: bool,
    /// **C10: chunks one fused step may carry** (`--engine-mixed-prefill packed`): 1 is C9's
    /// one-chunk law (the control), more is the packing law ([`crate::mixed_law::pack`]: every
    /// queued prompt's unpacked chunk that fits WHOLE, in arrival order, in class).
    pub max_chunks: u32,
    /// **The PRE-C11 control, host tests only** (no flag reaches it): C9's fused room
    /// ([`crate::mixed_law::fused_room`], `largest rung − decode rows`) cuts the head's chunk to
    /// fit the widest rung — which moves a long prompt's chunk boundary (1992..2047 rows where
    /// the unpacked schedule cuts 2048: the C9 defect C10 found). Built by
    /// [`MixedPrefill::fused_rung_cut_control`].
    pub rung_cut: bool,
}

impl MixedPrefill {
    /// Prod's `max_num_batched_tokens`.
    pub const PROD_MAX_BATCHED_TOKENS: u32 = 8192;

    /// Off: every chunk is a step of its own (the pre-C4 schedule and the pairing control).
    pub const OFF: MixedPrefill = MixedPrefill {
        enabled: false,
        max_batched_tokens: Self::PROD_MAX_BATCHED_TOKENS,
        fused: false,
        max_chunks: 1,
        rung_cut: false,
    };

    /// On, at `max_batched_tokens`.
    #[must_use]
    pub fn on(max_batched_tokens: u32) -> Self {
        Self {
            enabled: true,
            max_batched_tokens,
            fused: false,
            max_chunks: 1,
            rung_cut: false,
        }
    }

    /// C9: on, in the FUSED form, at `max_batched_tokens`.
    #[must_use]
    pub fn fused(max_batched_tokens: u32) -> Self {
        Self {
            fused: true,
            ..Self::on(max_batched_tokens)
        }
    }

    /// C10: the fused form with MULTI-CHUNK PACKING (`--engine-mixed-prefill packed`): up to
    /// [`crate::mixed_law::MAX_PACKED_CHUNKS`] queued prompts' chunks per step, whole, under
    /// [`crate::mixed_law::pack`].
    #[must_use]
    pub fn packed(max_batched_tokens: u32) -> Self {
        Self {
            max_chunks: crate::mixed_law::MAX_PACKED_CHUNKS,
            ..Self::fused(max_batched_tokens)
        }
    }

    /// **The pre-C11 control** (host tests only): C9's one-chunk fused form with its rung cut
    /// ([`crate::mixed_law::fused_room`]) — a long prompt's first chunk is cut to
    /// `2048 − decode rows`, so [`crate::mixed_law::chunk_sequence_holds`] fails on it.
    #[must_use]
    pub fn fused_rung_cut_control(max_batched_tokens: u32) -> Self {
        Self {
            rung_cut: true,
            ..Self::fused(max_batched_tokens)
        }
    }

    /// Whether a step's chunks come from the packing law ([`crate::mixed_law::pack`]): every
    /// fused form (C11: `fused` is the law at one chunk), never the serial form and never the
    /// pre-C11 control.
    #[must_use]
    pub fn uses_pack_law(&self) -> bool {
        self.enabled && self.fused && !self.rung_cut
    }

    /// Whether this form packs several chunks a step (C10).
    #[must_use]
    pub fn packs(&self) -> bool {
        self.enabled && self.fused && self.max_chunks > 1
    }

    /// The rows a chunk may take beside `decode_rows` decode rows, before the plan's chunk law.
    /// Zero means the budget is spent: this step runs its decode half only.
    #[must_use]
    pub fn room(&self, decode_rows: u32) -> u32 {
        if !self.enabled {
            return 0;
        }
        self.max_batched_tokens.saturating_sub(decode_rows)
    }

    /// The room under this form: prod's law ([`MixedPrefill::room`]) for the serial form and for
    /// every fused form (C11: a fused chunk is never cut to fit the rung — a chunk that does not
    /// fit one in class runs the serial form), and for the pre-C11 control that law capped by the
    /// widest rung (`largest_rung − decode_rows`, [`crate::mixed_law::fused_room`]).
    #[must_use]
    pub fn room_for(&self, decode_rows: u32, largest_rung: u32) -> u32 {
        if !self.enabled {
            return 0;
        }
        if self.rung_cut {
            crate::mixed_law::fused_room(self.max_batched_tokens, decode_rows, largest_rung)
        } else {
            self.room(decode_rows)
        }
    }
}

impl Default for MixedPrefill {
    fn default() -> Self {
        Self::OFF
    }
}

impl SchedulerConfig {
    /// Derives the configuration from a resolved plan.
    ///
    /// Eager buckets in the plan are dropped here, not filtered at selection time: a shape that was
    /// never captured is not a shape this scheduler may replay.
    pub fn from_plan(
        plan: &ExecutionPlan,
        kv: KvGeometry,
        arenas: ArenaBytes,
        max_model_len: usize,
    ) -> Result<Self, ConfigError> {
        if kv.block_tokens == 0 || kv.bytes_per_token_per_rank == 0 {
            return Err(ConfigError::DegenerateKv {
                block_tokens: kv.block_tokens,
                bytes: kv.bytes_per_token_per_rank,
            });
        }

        // Captured *decode* buckets. A prefill bucket is in the plan for its launch budget and
        // its GDN state, not as a replay candidate: selection answers "which captured shape does
        // this decode step take", and a prefill bucket would win it outright -- batch 1 and
        // `spec_width` 0 sort ahead of every decode bucket in the ladder's own order.
        let captured: Vec<&PlannedBucket> = plan
            .buckets()
            .iter()
            .filter(|b| b.bucket.captured && b.bucket.phase == Phase::Decode)
            .collect();
        if captured.is_empty() {
            return Err(ConfigError::NoCapturedBucket {
                plan_hash: plan.plan_hash().to_string(),
            });
        }

        // `resident_bytes / slots` is what one slot costs on one rank across every GDN layer:
        // `bytes_per_slot_per_layer` is already divided by tp, and `resident_bytes` multiplies it
        // by the layer count. Deriving it rather than restating 72 MiB keeps the fp16-state policy
        // knob (09-14/09-16) in one place — the plan.
        let mut per_slot: Option<(String, u64)> = None;
        for pb in &captured {
            if pb.gdn_state.slots == 0 || pb.gdn_state.resident_bytes == 0 {
                return Err(ConfigError::DegenerateGdnState {
                    bucket: pb.bucket.label(),
                    slots: pb.gdn_state.slots,
                    bytes: pb.gdn_state.resident_bytes,
                });
            }
            let bytes = pb.gdn_state.resident_bytes / u64::from(pb.gdn_state.slots);
            match &per_slot {
                None => per_slot = Some((pb.bucket.label(), bytes)),
                Some((label, first)) if *first != bytes => {
                    return Err(ConfigError::InconsistentGdnState {
                        a: label.clone(),
                        a_bytes: *first,
                        b: pb.bucket.label(),
                        b_bytes: bytes,
                    })
                }
                Some(_) => {}
            }
        }
        let (_, gdn_bytes_per_slot) = per_slot.expect("captured is non-empty");

        // The paged-attention route is bound at a `block_size`, and the block tables this
        // scheduler hands the executor are what that kernel indexes with. If the caller's geometry
        // disagrees with the plan, refuse here rather than serve a block table the kernel reads at
        // the wrong stride.
        for pb in &captured {
            for op in pb.ops.ops() {
                if let Some(block_size) = op.shape.num("block_size") {
                    if block_size != i64::from(kv.block_tokens) {
                        return Err(ConfigError::KvBlockMismatch {
                            op_name: op.op_name.clone(),
                            plan_block_size: block_size,
                            kv_block_tokens: kv.block_tokens,
                        });
                    }
                }
            }
        }

        let mut buckets: Vec<Bucket> = captured.iter().map(|pb| pb.bucket).collect();
        buckets.sort_by_key(|b| (b.batch, b.spec_width, b.ctx_band));

        // The prefill half of the plan. Absent, a prompt is fed in one decode-shaped step, which
        // is what this scheduler did before P2 — and is still what happens when a manifest
        // captures no prefill bucket, rather than refusing to serve.
        let mut prefill_buckets: Vec<Bucket> = plan
            .buckets()
            .iter()
            .filter(|b| b.bucket.captured && b.bucket.phase == Phase::Prefill)
            .map(|pb| pb.bucket)
            .collect();
        prefill_buckets.sort_by_key(|b| (b.tokens, b.ctx_band));
        let prefill =
            PrefillChunking::from_sizes(prefill_buckets.iter().map(|b| b.tokens).collect());

        let mut widths: Vec<u32> = buckets.iter().map(|b| b.spec_width).collect();
        widths.sort_unstable();
        widths.dedup();
        let width = VerifyWidthPolicy::from_widths(widths);

        let max_band = buckets
            .iter()
            .map(|b| b.ctx_band)
            .max()
            .expect("captured is non-empty") as usize;

        Ok(Self {
            plan_hash: plan.plan_hash().to_string(),
            buckets,
            prefill_buckets,
            prefill,
            tp: plan.body.tp.max(1),
            gdn_bytes_per_slot,
            kv,
            arenas,
            width,
            max_context_tokens: max_band.min(max_model_len),
            implicit_context: None,
            implicit_headroom: 0,
            park: ParkPolicy::default(),
            mixed: MixedPrefill::OFF,
            prefix_cache: crate::prefix_cache::PrefixCachePolicy::OFF,
            tree_reserve: 0,
            admit: AdmitPolicy::default(),
        })
    }

    /// The configuration a **mock** serve needs, from a *resolved* bucket set rather than a bound
    /// plan.
    ///
    /// `from_plan` is the production path and stays the only one: behaviour comes from the
    /// resolved [`ExecutionPlan`] (`CLAUDE.md` rule 4). This exists for the one case where there
    /// is no kernel to select — `saber serve --mock`, whose executor launches nothing — and where
    /// refusing to start because the autotune cache is cold would mean the serving plane could not
    /// be exercised at all until the GPU work lands.
    ///
    /// It is deliberately awkward to reach: the caller must already have run
    /// [`saber_plan::resolve`] (which is what proves the bucket set passes the Infinity-Cache gate
    /// and that every plugin is declared) and must pass a `provenance` string that says where the
    /// numbers came from, which is what [`SchedulerConfig::plan_hash`] then carries into the logs.
    pub fn for_mock(
        buckets: Vec<Bucket>,
        tp: u32,
        gdn_bytes_per_slot: u64,
        kv: KvGeometry,
        arenas: ArenaBytes,
        max_model_len: usize,
        provenance: String,
    ) -> Result<Self, ConfigError> {
        if kv.block_tokens == 0 || kv.bytes_per_token_per_rank == 0 {
            return Err(ConfigError::DegenerateKv {
                block_tokens: kv.block_tokens,
                bytes: kv.bytes_per_token_per_rank,
            });
        }
        // Captured prefill buckets, then captured decode buckets; see `from_plan`.
        let mut prefill_buckets: Vec<Bucket> = buckets
            .iter()
            .copied()
            .filter(|b| b.captured && b.phase == Phase::Prefill)
            .collect();
        prefill_buckets.sort_by_key(|b| (b.tokens, b.ctx_band));
        let prefill =
            PrefillChunking::from_sizes(prefill_buckets.iter().map(|b| b.tokens).collect());
        let mut buckets: Vec<Bucket> = buckets
            .into_iter()
            .filter(|b| b.captured && b.phase == Phase::Decode)
            .collect();
        if buckets.is_empty() {
            return Err(ConfigError::NoCapturedBucket {
                plan_hash: provenance,
            });
        }
        buckets.sort_by_key(|b| (b.batch, b.spec_width, b.ctx_band));
        let mut widths: Vec<u32> = buckets.iter().map(|b| b.spec_width).collect();
        widths.sort_unstable();
        widths.dedup();
        let max_band = buckets.iter().map(|b| b.ctx_band).max().expect("non-empty") as usize;
        Ok(Self {
            plan_hash: provenance,
            buckets,
            prefill_buckets,
            prefill,
            tp: tp.max(1),
            gdn_bytes_per_slot,
            kv,
            arenas,
            width: VerifyWidthPolicy::from_widths(widths),
            max_context_tokens: max_band.min(max_model_len),
            implicit_context: None,
            implicit_headroom: 0,
            park: ParkPolicy::default(),
            mixed: MixedPrefill::OFF,
            prefix_cache: crate::prefix_cache::PrefixCachePolicy::OFF,
            tree_reserve: 0,
            admit: AdmitPolicy::default(),
        })
    }

    /// Set the eviction policy.
    #[must_use]
    pub fn with_park_policy(mut self, park: ParkPolicy) -> Self {
        self.park = park;
        self
    }

    /// The widest speculative width the bucket set captures. One GDN state slot per draft token
    /// plus one for the committed state is what a sequence reserves, so this sets the slot-set
    /// size and therefore the state arena's unit.
    pub fn max_spec_width(&self) -> u32 {
        self.buckets.iter().map(|b| b.spec_width).max().unwrap_or(1)
    }

    /// State slots one sequence reserves: `max_spec_width + 1`.
    pub fn gdn_slots_per_seq(&self) -> u32 {
        self.max_spec_width() + 1
    }

    /// GDN state bytes one sequence costs on one rank.
    pub fn gdn_bytes_per_seq(&self) -> u64 {
        u64::from(self.gdn_slots_per_seq()) * self.gdn_bytes_per_slot
    }

    /// The largest batch the bucket set captures — the concurrency cap. A running set larger than
    /// this has no captured shape to replay, so admission stops there rather than replaying an
    /// uncaptured one.
    pub fn max_batch(&self) -> u32 {
        self.buckets.iter().map(|b| b.batch).max().unwrap_or(1)
    }

    /// Context at which one sequence's KV bytes equal **one** GDN state slot — the architecture's
    /// ~4.6k crossover (72 MiB / 16 KiB = 4608 tokens at TP=2, fp32 state, fp8 KV). Below it,
    /// state outweighs KV and the state allocator is what admission runs out of first.
    pub fn single_slot_crossover_tokens(&self) -> u64 {
        self.gdn_bytes_per_slot / self.kv.bytes_per_token_per_rank
    }

    /// The same crossover for a whole reserved slot set, which is what admission actually compares:
    /// a sequence reserves `max_spec_width + 1` slots whether or not it drafts that wide.
    pub fn crossover_tokens(&self) -> u64 {
        self.gdn_bytes_per_seq() / self.kv.bytes_per_token_per_rank
    }
}

#[cfg(test)]
mod chunk_law_tests {
    use super::PrefillChunking;

    /// **H36: the law at the 8192 arm's buckets (`256,2048,8192`)**, written down with the cost
    /// it carries. The paired depths are whole chunks (2048 one `p2048`, 8192 one `p8192`, 16384
    /// two, 32000 three and a 7424-row `p8192`), and a prompt between two buckets is ONE chunk
    /// padded into the larger one -- L9's deliberate choice (one M, not three), which at 5000
    /// tokens runs 8192 rows where the 2048 law runs 3 x 2048.
    #[test]
    fn the_8192_arm_cuts_by_the_same_law_and_pads_the_middle() {
        let law = PrefillChunking::from_sizes(vec![8192, 256, 2048]);
        let cut = |mut n: u32| {
            let mut out = Vec::new();
            while let Some((chunk, bucket)) = law.chunk_for(n) {
                out.push((chunk, bucket));
                n -= chunk;
            }
            out
        };
        assert_eq!(cut(2048), vec![(2048, 2048)]);
        assert_eq!(cut(8192), vec![(8192, 8192)]);
        assert_eq!(cut(16384), vec![(8192, 8192), (8192, 8192)]);
        assert_eq!(
            cut(32000),
            vec![(8192, 8192), (8192, 8192), (8192, 8192), (7424, 8192)]
        );
        assert_eq!(cut(5000), vec![(5000, 8192)], "one chunk, padded");
        assert_eq!(cut(200), vec![(200, 256)]);
        assert_eq!(law.steps_for(32000), 4);
        // The CONTROL: main's buckets cut the same prompts into 2048-row chunks.
        let main = PrefillChunking::from_sizes(vec![256, 2048]);
        assert_eq!(main.steps_for(32000), 16);
        assert_eq!(main.chunk_for(5000), Some((2048, 2048)));
    }
}
