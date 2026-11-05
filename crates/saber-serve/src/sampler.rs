//! The sampler behind the head — the reference server's V1 sampler, step for step.
//!
//! The device head produces an **argmax** (`saber_argmax_bf16` in `hip/head.hip`), which
//! is the whole of greedy sampling and nothing of anything else. This module is the rest:
//! penalties, temperature, `min_p`, `top_k`, `top_p` and the draw, on the host, in
//! **prod's order of operations**.
//!
//! # The order, and why it is not the one the docstring implies
//!
//! `upstream/v1/sample/sampler.py`'s class docstring lists "apply top_k and/or top_p" and
//! then "min_p processor", which reads as `top_k → top_p → min_p`. The code does the
//! opposite, and the code is what prod runs: `Sampler.sample` applies temperature, then
//! **every `argmax_invariant` logits processor** — which is where `MinPLogitsProcessor`
//! lives (`is_argmax_invariant() -> True`) — and only then calls `TopKTopPSampler`. So
//! the real order, per row, is
//!
//! ```text
//!   raw logprobs (log_softmax of the RAW logits — before anything below)
//!   penalties: repetition, then frequency, then presence
//!   greedy argmax          <- taken HERE, on post-penalty / pre-temperature logits
//!   temperature            <- logits /= T
//!   min_p                  <- mask probs < min_p * max_prob
//!   top_k                  <- mask logits < the k-th largest (ties KEPT)
//!   top_p                  <- ascending sort, ascending cumsum, mask cumsum <= 1 - p
//!   sample                 <- argmax(softmax(logits) / Exp(1) noise)
//!   temperature < 1e-5 ? greedy : sampled
//! ```
//!
//! Every one of those is reproduced below in the same arithmetic order, because the
//! order is the only thing a parity gate can actually see: `top_p` before `min_p` gives
//! a different surviving set on the same logits whenever both bind.
//!
//! # What leaves the device
//!
//! Running this over a dense 248,320-wide bf16 row would mean reading the whole logit
//! vector back every step. It does not have to: everything above is a function of
//!
//! * the **K largest raw logits and their ids** (same total order as the argmax kernel:
//!   value descending, id ascending),
//! * the row's **log-sum-exp**, once at T = 1 (logprobs are raw-logit logprobs) and once
//!   at the row's own temperature (the only thing `top_p`'s cumulative sum needs that
//!   the top-K does not carry),
//! * the raw logits at the **ids the penalties touch** — the host knows that set, it is
//!   the prompt's and the output's tokens, and it is the only set penalties can move.
//!
//! [`DeviceRow`] is that seam. [`Sampler::sample`] runs the identical pipeline over it,
//! and **refuses by name** when K is too narrow to decide the row ([`SamplerError`]),
//! rather than quietly sampling from a truncated distribution. The kernel that fills a
//! `DeviceRow` is not in this tree yet — see `docs/GATES.md`, "what is left for a card".
//!
//! # What this module is gated against
//!
//! `the internal reference harness` runs **prod's own sampler code** inside the prod image
//! on the CPU over fixed logit vectors and records, per case, the surviving mask, the
//! processed logits, the greedy id, prod's logprob schema and — for seeded rows — a
//! histogram of ≥ 10,000 draws. `tests/sampler_corpus.rs` replays it.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::engine::{SampledLogprobs, TokenId, TokenLogprob};
use crate::sampling::SamplingParams;

/// The reference server's `_SAMPLING_EPS`: at or below this, a request is greedy and nothing else runs.
pub const SAMPLING_EPS: f64 = 1e-5;

/// The reference server's `_MAX_TEMP`: a temperature in `(0, 1e-2)` is raised to `1e-2` with a warning,
/// because `logits / 1e-8` overflows to inf and softmax returns NaN.
pub const MAX_TEMP: f32 = 1e-2;

/// The reference server clamps every logprob it reports at this floor
/// (`_get_top_logprobs`: `logprob=max(p[1].logprob, -9999.0)`).
pub const LOGPROB_FLOOR: f32 = -9999.0;

// ── the knobs, after the reference server's own normalisation ───────────────────────────────────

/// One row's sampling knobs **after** `SamplingParams.__post_init__` and the input
/// batch's own normalisation. Three rewrites happen before a logit is touched and all
/// three change the answer:
///
/// * `0 < temperature < 1e-2` is raised to `1e-2`;
/// * `temperature < 1e-5` forces `top_p = 1`, `top_k = 0`, `min_p = 0` — a greedy
///   request cannot be given a truncation, whatever it sent;
/// * `top_k` outside `1..vocab` becomes `vocab`, which is the disabled form
///   (`gpu_input_batch.py`: `if 0 < top_k < vocab_size … else top_k = vocab_size`).
///   `-1` is accepted as "disabled" and only `< -1` is a rejection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RowParams {
    pub temperature: f32,
    pub top_p: f32,
    /// Already normalised: `vocab` means disabled.
    pub top_k: usize,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repetition_penalty: f32,
    /// `None` = no logprobs; `Some(k)` = the top `k` plus the sampled token, prod's
    /// schema. `Some(-1)` is "all of them" and is resolved to the vocabulary here.
    pub logprobs: Option<usize>,
    pub vocab_size: usize,
}

impl RowParams {
    /// Resolve a request's parameters for a vocabulary of `vocab_size`.
    pub fn resolve(params: &SamplingParams, vocab_size: usize) -> Self {
        let mut temperature = params.temperature;
        if temperature > 0.0 && temperature < MAX_TEMP {
            temperature = MAX_TEMP;
        }
        let greedy = f64::from(temperature) < SAMPLING_EPS;
        let (top_p, top_k, min_p) = if greedy {
            (1.0, vocab_size, 0.0)
        } else {
            let k = params.top_k;
            let k = if k > 0 && (k as usize) < vocab_size {
                k as usize
            } else {
                vocab_size
            };
            (params.top_p, k, params.min_p)
        };
        let logprobs = params
            .logprobs
            .map(|k| if k < 0 { vocab_size } else { k as usize });
        Self {
            temperature,
            top_p,
            top_k,
            min_p,
            presence_penalty: params.presence_penalty,
            frequency_penalty: params.frequency_penalty,
            repetition_penalty: params.repetition_penalty,
            logprobs,
            vocab_size,
        }
    }

    /// `SamplingType.GREEDY`: the argmax is the answer and nothing below it runs.
    pub fn is_greedy(&self) -> bool {
        f64::from(self.temperature) < SAMPLING_EPS
    }

    /// `SamplingMetadata.no_penalties` for this row.
    pub fn no_penalties(&self) -> bool {
        self.presence_penalty == 0.0
            && self.frequency_penalty == 0.0
            && self.repetition_penalty == 1.0
    }

    /// The ids a penalised row can move: the prompt's and the output's tokens. Nothing
    /// else changes, which is what makes a top-K seam sufficient.
    pub fn penalty_ids(&self, ctx: &PenaltyContext<'_>) -> Vec<TokenId> {
        if self.no_penalties() {
            return Vec::new();
        }
        let mut ids: Vec<TokenId> = Vec::with_capacity(ctx.prompt.len() + ctx.output.len());
        ids.extend_from_slice(ctx.prompt);
        ids.extend_from_slice(ctx.output);
        ids.sort_unstable();
        ids.dedup();
        ids
    }
}

/// The token history the penalties read. `prompt` contributes to the repetition mask
/// only; `output` contributes to all three, and to the frequency penalty **by count**.
#[derive(Debug, Clone, Copy, Default)]
pub struct PenaltyContext<'a> {
    pub prompt: &'a [TokenId],
    pub output: &'a [TokenId],
}

// ── the seam ────────────────────────────────────────────────────────────────────

/// What one row's head hands the host when the row is not greedy.
///
/// Ordering of `top_values` / `top_ids` is the argmax kernel's total order — value
/// descending, id ascending on a tie — so a `K = 1` row is byte-identical to what
/// `saber_argmax_bf16` writes today.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceRow {
    /// The `K` largest raw logits, descending.
    pub top_values: Vec<f32>,
    /// Their ids, in the same order.
    pub top_ids: Vec<TokenId>,
    /// `logsumexp` of the whole raw row at T = 1. The reference server's logprobs are the **raw**
    /// logits' `log_softmax`, taken before penalties and temperature, so this is what
    /// the reported numbers are normalised by.
    pub logsumexp: f32,
    /// `logsumexp` of the whole raw row divided by the row's temperature. Only `top_p`
    /// needs it (its cumulative sum is over the full distribution); `None` is legal and
    /// is refused, by name, exactly when `top_p < 1`.
    pub logsumexp_at_temperature: Option<f32>,
    /// The raw logits at [`RowParams::penalty_ids`], in that order. Empty when the row
    /// has no penalties.
    pub penalised: Vec<(TokenId, f32)>,
    pub vocab_size: usize,
}

// ── the seam at TP > 1: one shard's half of a row, and the merge ────────────────

/// One vocabulary shard's half of a row's seam, exactly as that rank's head wrote it.
///
/// At TP = 2 `lm_head` is column-parallel: rank *r* holds `[base, base + cols)` and neither
/// rank's top-K is the row's top-K. `saber_device_row_bf16` runs on each rank over its own
/// shard — the same kernel, the same K, the same total order — and this is what comes back,
/// with the ids still LOCAL because that is what the kernel writes and converting them twice
/// is how a base gets added twice.
///
/// [`merge_shards`] is the law that turns a set of these into the [`DeviceRow`] one undivided
/// row would have produced.
#[derive(Debug, Clone, PartialEq)]
pub struct SeamShard {
    /// The first GLOBAL token id this shard holds: `rank * cols`.
    pub base: u32,
    /// Ids in this shard — the head's own `cols`, not the model's vocabulary.
    pub cols: u32,
    /// The `k` largest raw logits of THIS shard, value descending.
    pub top_values: Vec<f32>,
    /// Their columns **within the shard**, in the same order (column ascending on a tie).
    pub top_cols: Vec<u32>,
    /// `logsumexp` of this shard's raw logits at T = 1.
    pub logsumexp: f32,
    /// `logsumexp` of this shard's raw logits divided by the row's temperature.
    pub logsumexp_at_temperature: Option<f32>,
    /// The raw logits at the penalty ids, in the CALLER's order, and `None` at every id this
    /// shard does not hold. A shard has no opinion about the other shard's logits, and saying
    /// `None` rather than `0.0` is what lets the merge refuse an id neither shard claimed.
    pub penalised: Vec<Option<f32>>,
}

/// Why a set of shards is not a row.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SeamMergeError {
    #[error("a row merged from no shards at all")]
    NoShards,
    #[error(
        "shard {at} covers [{base}, {base}+{cols}) and the shards before it end at {expected}: \
         the shards must tile the vocabulary in rank order, with no gap and no overlap"
    )]
    NotATiling {
        /// Which shard, in the order they were handed over.
        at: usize,
        /// Its declared base.
        base: u32,
        /// Its declared width.
        cols: u32,
        /// Where the shards before it ended.
        expected: u32,
    },
    #[error("the shards cover {covered} ids and the vocabulary is {vocab}")]
    WrongVocabulary {
        /// Ids the shards tile between them.
        covered: u32,
        /// Ids the model has.
        vocab: usize,
    },
    #[error("shard {at} sent {values} value(s) and {ids} id(s): the head writes one id per value")]
    Ragged {
        /// Which shard.
        at: usize,
        /// Values it sent.
        values: usize,
        /// Ids it sent.
        ids: usize,
    },
    #[error("shard {at} sent column {col} of a {cols}-wide shard")]
    ColumnPastShard {
        /// Which shard.
        at: usize,
        /// The column it named.
        col: u32,
        /// How wide it said it was.
        cols: u32,
    },
    #[error(
        "shard {at} carries {got} penalty slot(s) and the row asked for {want}: every rank \
         gathers the SAME id list and answers `None` where the id is not its own"
    )]
    PenaltyWidth {
        /// Which shard.
        at: usize,
        /// Slots it sent.
        got: usize,
        /// Slots the row asked for.
        want: usize,
    },
    #[error("penalty id {id} is in no shard's range, so no rank gathered its logit")]
    PenaltyUnclaimed {
        /// The id nobody claimed.
        id: TokenId,
    },
    #[error(
        "shard {at} carries a log-sum-exp at the row's temperature and another does not; \
         top_p needs all of them or none"
    )]
    MixedTemperedLogsumexp {
        /// Which shard disagreed with the first.
        at: usize,
    },
}

/// `log(exp(a) + exp(b))`, the only arithmetic two shards' log-sum-exps can be combined by.
///
/// `logsumexp` over a concatenation IS `log-add-exp` of the parts — an identity, not an
/// approximation — so the only question is rounding. It is computed in **f64** and rounded to
/// f32 **once**, at the return: `m + ln(exp(a-m) + exp(b-m))` with `m = max(a, b)`, so neither
/// term overflows and the smaller one is the only thing that can underflow, to the right answer.
/// The same expression in f32 would round three times.
///
/// Nothing here can be tighter than what it combines: each shard's log-sum-exp is the kernel's
/// own, which `tools/capture/gate_device_row.sh` measures at **1 ulp** against an f64 reference.
/// The measured distance of a merged pair from the TP = 1 kernel's log-sum-exp over the
/// undivided row is in `docs/GATES.md`.
#[must_use]
pub fn log_add_exp(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let (a, b) = (f64::from(a), f64::from(b));
    let m = if a > b { a } else { b };
    if !m.is_finite() {
        return m as f32;
    }
    (m + ((a - m).exp() + (b - m).exp()).ln()) as f32
}

/// The row a set of vocabulary shards describes: the merge law of the TP > 1 seam.
///
/// Three things happen and each is exact for a stated reason:
///
/// 1. **the top-K.** Each shard's list is already `(value descending, column ascending)`, and a
///    shard's columns map to global ids by adding a constant, which preserves that order. So the
///    row's list is the ordinary merge of the sorted lists by `(value descending, GLOBAL id
///    ascending)`, truncated to `k`. It is the row's true top-K and not an approximation: an id
///    in the row's top-K has fewer than `k` ids above it in the whole row, hence fewer than `k`
///    above it inside its own shard, hence it is in that shard's top-K and was sent. The tie rule
///    survives the same way — a member of a tie group its shard truncated has a HIGHER column
///    than the `k` that shard sent, so it ranks below all of them in the merged order too.
/// 2. **the log-sum-exps**, by [`log_add_exp`], which is what `logsumexp` over a concatenation
///    means.
/// 3. **the penalty gather.** Every rank is handed the same id list and gathers the ids inside
///    its own range, so exactly one shard answers for each id and the merge is a selection, not
///    a sum. An id no shard claimed is a refusal: it would otherwise be sampled as if nothing had
///    moved it.
///
/// `penalty_ids` is the list every shard was handed, in the order it was handed.
///
/// # Errors
/// Shards that do not tile `0..vocab` in order, a ragged shard, a column outside its shard, a
/// penalty list of the wrong width, an id no shard holds, or shards that disagree about whether
/// they carry the tempered log-sum-exp.
pub fn merge_shards(
    shards: &[SeamShard],
    penalty_ids: &[TokenId],
    k: usize,
    vocab: usize,
) -> Result<DeviceRow, SeamMergeError> {
    let Some(first) = shards.first() else {
        return Err(SeamMergeError::NoShards);
    };
    let mut expected = 0u32;
    for (at, s) in shards.iter().enumerate() {
        if s.base != expected {
            return Err(SeamMergeError::NotATiling {
                at,
                base: s.base,
                cols: s.cols,
                expected,
            });
        }
        if s.top_values.len() != s.top_cols.len() {
            return Err(SeamMergeError::Ragged {
                at,
                values: s.top_values.len(),
                ids: s.top_cols.len(),
            });
        }
        if let Some(col) = s.top_cols.iter().copied().find(|c| *c >= s.cols) {
            return Err(SeamMergeError::ColumnPastShard {
                at,
                col,
                cols: s.cols,
            });
        }
        if s.penalised.len() != penalty_ids.len() {
            return Err(SeamMergeError::PenaltyWidth {
                at,
                got: s.penalised.len(),
                want: penalty_ids.len(),
            });
        }
        if s.logsumexp_at_temperature.is_some() != first.logsumexp_at_temperature.is_some() {
            return Err(SeamMergeError::MixedTemperedLogsumexp { at });
        }
        expected = expected.saturating_add(s.cols);
    }
    if expected as usize != vocab {
        return Err(SeamMergeError::WrongVocabulary {
            covered: expected,
            vocab,
        });
    }

    // The merge itself: one cursor per shard, always take the best remaining head. `k` is small
    // and the shard count is two, so this is a loop rather than a heap on purpose — the order is
    // the whole point, and a comparator hidden inside a data structure is a comparator nobody
    // reads.
    let mut at = vec![0usize; shards.len()];
    let mut top_values: Vec<f32> = Vec::with_capacity(k);
    let mut top_ids: Vec<TokenId> = Vec::with_capacity(k);
    while top_values.len() < k {
        let mut best: Option<(usize, f32, TokenId)> = None;
        for (i, s) in shards.iter().enumerate() {
            let Some(v) = s.top_values.get(at[i]).copied() else {
                continue;
            };
            let id = s.base + s.top_cols[at[i]];
            // VALUE DESCENDING, then GLOBAL ID ASCENDING — `saber_argmax_bf16`'s own total
            // order, and the one the TP = 1 kernel breaks its ties by.
            let better = best.is_none_or(|(_, bv, bid)| v > bv || (v == bv && id < bid));
            if better {
                best = Some((i, v, id));
            }
        }
        let Some((i, v, id)) = best else { break };
        at[i] += 1;
        top_values.push(v);
        top_ids.push(id);
    }

    let mut logsumexp = f32::NEG_INFINITY;
    let mut logsumexp_at_temperature = first.logsumexp_at_temperature.map(|_| f32::NEG_INFINITY);
    for s in shards {
        logsumexp = log_add_exp(logsumexp, s.logsumexp);
        if let (Some(acc), Some(v)) = (
            logsumexp_at_temperature.as_mut(),
            s.logsumexp_at_temperature,
        ) {
            *acc = log_add_exp(*acc, v);
        }
    }

    let mut penalised = Vec::with_capacity(penalty_ids.len());
    for (j, id) in penalty_ids.iter().enumerate() {
        let value = shards.iter().find_map(|s| s.penalised[j]);
        match value {
            Some(v) => penalised.push((*id, v)),
            None => return Err(SeamMergeError::PenaltyUnclaimed { id: *id }),
        }
    }

    Ok(DeviceRow {
        top_values,
        top_ids,
        logsumexp,
        logsumexp_at_temperature,
        penalised,
        vocab_size: vocab,
    })
}

/// Why a row could not be decided from what the device sent.
///
/// Structural rule 5: the seam has a fallback (re-issue the row with a wider K) so it
/// declares the refusal rather than sampling from a truncated distribution and calling
/// it a token.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SamplerError {
    #[error(
        "top-K seam too narrow: K={k} covers {covered_mass} of the distribution but \
         top_p={top_p} needs the cut above {needed_below}; re-issue the row with a wider K"
    )]
    SeamTooNarrow {
        k: usize,
        top_p: f32,
        covered_mass: f32,
        needed_below: f32,
    },
    #[error(
        "top-K seam carries no logsumexp at the row's temperature, which top_p={top_p} \
         needs (the cumulative sum is over the WHOLE distribution, not the top {k})"
    )]
    SeamNoTemperedLogsumexp { k: usize, top_p: f32 },
    #[error(
        "top-K seam sent {k} candidates but top_k={top_k} would cut below them; \
         re-issue the row with a wider K"
    )]
    SeamTooNarrowForTopK { k: usize, top_k: usize },
    #[error(
        "top-K seam sent {k} candidates but the request asked for {requested} logprobs; \
         re-issue the row with a wider K"
    )]
    SeamTooNarrowForLogprobs { k: usize, requested: usize },
    #[error(
        "the row has no truncation that fits inside a {k}-wide seam \
         (top_p={top_p}, top_k={top_k}, min_p={min_p}): after masking, an id the device \
         never sent could still have won the draw. Read the row densely or widen K"
    )]
    SeamUntruncated {
        k: usize,
        top_p: f32,
        top_k: usize,
        min_p: f32,
    },
    #[error("top-K seam is empty: the head sent no candidates for a non-greedy row")]
    SeamEmpty,
    #[error("logits row is empty")]
    EmptyRow,
}

/// Which branch decided a row. A route with a fallback declares a counter at
/// construction (CLAUDE.md, structural rule 5) — these are it.
#[derive(Debug, Default)]
pub struct SamplerCounters {
    /// Rows answered by the argmax, `temperature < 1e-5`.
    pub greedy_rows: AtomicU64,
    /// Rows answered by the draw.
    pub random_rows: AtomicU64,
    /// Rows where the penalties moved at least one logit.
    pub penalised_rows: AtomicU64,
    /// Rows where `top_k` masked something.
    pub top_k_rows: AtomicU64,
    /// Rows where `top_p` masked something.
    pub top_p_rows: AtomicU64,
    /// Rows where `min_p` masked something.
    pub min_p_rows: AtomicU64,
    /// Rows the seam refused — see [`SamplerError`]. Never silent.
    pub seam_refusals: AtomicU64,
    /// Drawn rows whose LARGEST logit (after the penalties) is held by two or more ids — an exact
    /// tie at the top, which the bf16 logit row makes common (spacing 0.125 at a logit of 16..32).
    /// `top_k` keeps every member of a tie (`logits < the k-th largest` is the mask), so at such a
    /// row even `top_k = 1` leaves more than one candidate, exactly as prod's `apply_top_k_only`
    /// does (ws-f39).
    pub top_ties: AtomicU64,
    /// Of those, the rows whose draw was a tied id OTHER than the argmax's (the lowest id, which is
    /// what temperature 0 commits): the token a greedy request would not have been given, at a
    /// margin of exactly zero. At `top_k = 1` every divergence from the temperature-0 answer is
    /// one of these (ws-f39).
    pub top_tie_draws: AtomicU64,
}

impl SamplerCounters {
    fn bump(counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// A snapshot, for `/metrics` and for a test that asserts a branch was taken.
    pub fn snapshot(&self) -> [(&'static str, u64); 9] {
        [
            ("greedy_rows", self.greedy_rows.load(Ordering::Relaxed)),
            ("random_rows", self.random_rows.load(Ordering::Relaxed)),
            (
                "penalised_rows",
                self.penalised_rows.load(Ordering::Relaxed),
            ),
            ("top_k_rows", self.top_k_rows.load(Ordering::Relaxed)),
            ("top_p_rows", self.top_p_rows.load(Ordering::Relaxed)),
            ("min_p_rows", self.min_p_rows.load(Ordering::Relaxed)),
            ("seam_refusals", self.seam_refusals.load(Ordering::Relaxed)),
            ("top_ties", self.top_ties.load(Ordering::Relaxed)),
            ("top_tie_draws", self.top_tie_draws.load(Ordering::Relaxed)),
        ]
    }
}

// ── noise ───────────────────────────────────────────────────────────────────────

/// The `Exp(1)` draws `random_sample` divides the probabilities by.
///
/// The reference server draws them with `torch.Tensor.exponential_`, whose CPU path is ATen's own
/// MT19937 — reproducing that bit for bit in Rust would be reproducing a specific
/// PyTorch build, not a specification. Saber therefore draws its own, from a
/// **counter-based** generator keyed on `(seed, output position, choice, token id)`: the
/// draw that decides a request's `p`-th output token is a function of those four numbers
/// and of that row's logits ALONE — not of the step it ran in, the verify width, the batch,
/// a park or a fork (ws-f40; ws-f39 keyed it by the sequence's STEP, and a park or a crowd
/// moves step boundaries). The reference server's per-request `torch.Generator` is sequential and is
/// consumed per call — the bonus row, the rejection sampler's uniforms (one per draft
/// token) and its recovered-token row — so prod's seeded stream depends on how many steps
/// and drafts produced it (`docs/LEDGER.md`, F40). What the gate compares is therefore the
/// distribution (a chi-square over ≥ 10k draws) and, on every row where the surviving set
/// has one member, the exact id.
pub trait NoiseSource {
    /// One `Exp(1)` draw for candidate `index` of this row.
    fn exponential(&mut self, index: usize) -> f32;
}

/// `splitmix64`, the mixing function behind the seeded source. Not a cryptographic
/// primitive; it is here because it is one round, has no state to carry and is exactly
/// specifiable in a sentence, which is what makes a seeded sampler reproducible.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A reproducible `Exp(1)` source: `-ln(U)` with `U` uniform in `(0, 1]`.
#[derive(Debug, Clone, Copy)]
pub struct SeededNoise {
    key: u64,
}

impl SeededNoise {
    /// Key the row: the request's seed, the absolute OUTPUT POSITION of the token this row
    /// decides (0 = the first generated token), and which of the request's `n` choices this
    /// row is. The candidate within the row is the fourth coordinate, taken per draw
    /// ([`NoiseSource::exponential`], by token id).
    ///
    /// The position — not the step — is what makes a seeded request schedule-invariant
    /// (ws-f40): the same decision gets the same value whatever step, verify width, batch,
    /// park or fork it happens in. `new(seed, step, choice)` is the pre-F40 key and survives
    /// only as the executor's negative control.
    pub fn new(seed: u64, position: u64, choice: u32) -> Self {
        let key = splitmix64(seed)
            ^ splitmix64(position.wrapping_mul(0x00FF_51AF_D7ED_558D))
            ^ splitmix64(u64::from(choice).wrapping_mul(0xC4CE_B9FE_1A85_EC53));
        Self { key }
    }
}

impl NoiseSource for SeededNoise {
    fn exponential(&mut self, index: usize) -> f32 {
        let bits = splitmix64(self.key ^ splitmix64(index as u64));
        // 53 bits into (0, 1]: the +1 keeps it off zero so -ln never returns inf.
        let u = (((bits >> 11) + 1) as f64) * (1.0 / 9_007_199_254_740_993.0);
        -(u.ln()) as f32
    }
}

/// The unseeded source: a request without `seed` still has to draw *something*, and
/// what it draws must not be a function of anything reproducible, or two identical
/// requests would return identical samples and a client would read that as caching.
#[derive(Debug)]
pub struct EntropyNoise {
    inner: SeededNoise,
}

impl Default for EntropyNoise {
    fn default() -> Self {
        Self::new()
    }
}

impl EntropyNoise {
    pub fn new() -> Self {
        static TICK: AtomicU64 = AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);
        let tick = TICK.fetch_add(1, Ordering::Relaxed);
        Self {
            inner: SeededNoise {
                key: splitmix64(nanos ^ splitmix64(tick)),
            },
        }
    }
}

impl NoiseSource for EntropyNoise {
    fn exponential(&mut self, index: usize) -> f32 {
        self.inner.exponential(index)
    }
}

// ── the row the pipeline runs on ────────────────────────────────────────────────

/// The ids behind a logit slice: the whole vocabulary, or the seam's candidates.
#[derive(Debug, Clone, Copy)]
enum Ids<'a> {
    Identity,
    Explicit(&'a [TokenId]),
}

impl Ids<'_> {
    fn at(&self, i: usize) -> TokenId {
        match self {
            Ids::Identity => i as TokenId,
            Ids::Explicit(ids) => ids[i],
        }
    }
}

/// The mass this row does **not** carry: the vocabulary beyond the top-K, minus
/// whatever the penalties pulled into the candidate list. Zero for a dense row.
///
/// `mass` is `sum(exp(l_i / T))` over the tail divided by the same sum over everything
/// — exact, because it is the difference of two log-sum-exps the device took over the
/// whole row. Every tail element is at or below the K-th top value, which is what lets
/// it be treated as one lump at the bottom of `top_p`'s ascending cumulative sum.
#[derive(Debug, Clone, Copy, Default)]
struct Tail {
    mass: f32,
}

/// One row's result.
#[derive(Debug, Clone, PartialEq)]
pub struct Sampled {
    pub token_id: TokenId,
    pub logprobs: Option<SampledLogprobs>,
    pub greedy: bool,
}

/// The pipeline. Stateless apart from its counters.
#[derive(Debug, Default)]
pub struct Sampler {
    pub counters: SamplerCounters,
}

impl Sampler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sample one row from a **dense** logit vector — the reference path, and the one
    /// `the internal reference harness` compares against prod. `logits` is consumed:
    /// The reference server's sampler is in-place too, and a caller that wants the raw row keeps a copy.
    pub fn sample_dense(
        &self,
        logits: &mut [f32],
        params: &RowParams,
        ctx: &PenaltyContext<'_>,
        noise: &mut dyn NoiseSource,
    ) -> Result<Sampled, SamplerError> {
        if logits.is_empty() {
            return Err(SamplerError::EmptyRow);
        }
        let raw_logprobs = params.logprobs.map(|_| log_softmax(logits));
        self.run(logits, Ids::Identity, None, params, ctx, noise)
            .map(|(token_id, greedy)| Sampled {
                logprobs: raw_logprobs
                    .map(|lp| gather_logprobs(&lp, Ids::Identity, token_id, params, None)),
                token_id,
                greedy,
            })
    }

    /// Sample one row from what the head sent — the production path.
    ///
    /// The seam is legitimate exactly when the row's own truncation is **narrower than
    /// K**: `top_k <= K`, or a `top_p`/`min_p` cut that lands above the K-th logit. A
    /// row with no truncation at all samples from the whole vocabulary and a top-K head
    /// simply cannot serve it — that is refused here by name rather than answered from a
    /// silently truncated distribution. Prod's own defaults (`top_k = 20`,
    /// `top_p = 0.95`) are inside the seam with K = 64 by a wide margin.
    pub fn sample(
        &self,
        row: &DeviceRow,
        params: &RowParams,
        ctx: &PenaltyContext<'_>,
        noise: &mut dyn NoiseSource,
    ) -> Result<Sampled, SamplerError> {
        if row.top_ids.is_empty() {
            SamplerCounters::bump(&self.counters.seam_refusals);
            return Err(SamplerError::SeamEmpty);
        }
        let k = row.top_ids.len();
        let complete = k >= row.vocab_size;
        // The seam's width also bounds the reported logprobs: the top `requested` of the
        // WHOLE row has to be inside the top-K, or the reply would name the wrong tokens.
        if let Some(requested) = params.logprobs {
            if requested > k && !complete {
                SamplerCounters::bump(&self.counters.seam_refusals);
                return Err(SamplerError::SeamTooNarrowForLogprobs { k, requested });
            }
        }
        if !params.is_greedy() && !complete && params.top_k < row.vocab_size && params.top_k > k {
            SamplerCounters::bump(&self.counters.seam_refusals);
            return Err(SamplerError::SeamTooNarrowForTopK {
                k,
                top_k: params.top_k,
            });
        }
        // The candidate list is the top-K plus every id the penalties can move. Those
        // are the only two sets that can hold the answer: an unpenalised id outside the
        // top-K is, by construction, below K logits that are all still there.
        let (ids, mut logits) = candidates(row);
        let floor = *row.top_values.last().expect("non-empty");
        // Raw logprobs come off the device's own log-sum-exp over the whole row, so each
        // candidate's number is exact. A RANK is not: it counts the whole vocabulary,
        // and the seam can only count what it sent. Above the K-th value the two agree
        // by construction, which is the condition `gather_logprobs` is handed.
        let raw_logprobs = params.logprobs.map(|_| {
            logits
                .iter()
                .map(|l| l - row.logsumexp)
                .collect::<Vec<f32>>()
        });
        let rank_floor = (!complete).then_some(floor - row.logsumexp);

        let seam = (!complete).then(|| Seam {
            k,
            floor,
            lse_t: row.logsumexp_at_temperature,
            raw: logits.clone(),
        });
        let result = self.run(
            &mut logits,
            Ids::Explicit(&ids),
            seam.as_ref(),
            params,
            ctx,
            noise,
        );
        match result {
            Ok((token_id, greedy)) => Ok(Sampled {
                logprobs: raw_logprobs.map(|lp| {
                    gather_logprobs(&lp, Ids::Explicit(&ids), token_id, params, rank_floor)
                }),
                token_id,
                greedy,
            }),
            Err(err) => {
                SamplerCounters::bump(&self.counters.seam_refusals);
                Err(err)
            }
        }
    }

    /// The logprobs a row reports for a token **somebody else chose** — prod's schema, no draw.
    ///
    /// This is the accepted prefix of a speculative step. A verify step commits `accepted + 1`
    /// tokens: the last one is the recovery or bonus token and the sampler draws it, and the ones
    /// before it are draft tokens the acceptance law already decided. Their distribution is a real
    /// number the seam already holds — the model's answer at that position — and a client asking
    /// for logprobs wants one entry per token of the text it received, not one for the last.
    ///
    /// Nothing about the pipeline runs here. The reported numbers are the **raw** logits'
    /// `log_softmax` (`SampledLogprobs` is prod's schema: the token first, then the top-k, every
    /// value clamped), taken off the device's own log-sum-exp, and penalties, temperature and the
    /// three masks decide a DRAW rather than a report.
    ///
    /// # Errors
    /// [`SamplerError::SeamEmpty`] for a row the head skipped, and
    /// [`SamplerError::SeamTooNarrowForLogprobs`] when the request asked for more than the seam
    /// carries. A committed token the seam does not hold is not an error: its own logprob is
    /// unknown, so it is reported at the floor with no rank, which is what
    /// `LogprobsProcessor` does for a sampled token outside the top-k.
    pub fn logprobs_at(
        &self,
        row: &DeviceRow,
        params: &RowParams,
        token: TokenId,
    ) -> Result<SampledLogprobs, SamplerError> {
        if row.top_ids.is_empty() {
            SamplerCounters::bump(&self.counters.seam_refusals);
            return Err(SamplerError::SeamEmpty);
        }
        let k = row.top_ids.len();
        let complete = k >= row.vocab_size;
        if let Some(requested) = params.logprobs {
            if requested > k && !complete {
                SamplerCounters::bump(&self.counters.seam_refusals);
                return Err(SamplerError::SeamTooNarrowForLogprobs { k, requested });
            }
        }
        let (mut ids, mut logits) = candidates(row);
        // A committed token the seam did not carry still has to appear FIRST in the reply, which
        // is prod's schema. Its own logprob is below the K-th value and the seam cannot count it,
        // so it enters the list at the floor and `rank_floor` makes the rank `None`.
        if !ids.contains(&token) {
            ids.push(token);
            logits.push(f32::NEG_INFINITY);
        }
        let floor = *row.top_values.last().expect("non-empty");
        let raw: Vec<f32> = logits.iter().map(|l| l - row.logsumexp).collect();
        let rank_floor = (!complete).then_some(floor - row.logsumexp);
        Ok(gather_logprobs(
            &raw,
            Ids::Explicit(&ids),
            token,
            params,
            rank_floor,
        ))
    }

    /// The TRUNCATED distribution a row draws from (ws-f41): prod's `p'` — penalties, temperature,
    /// `min_p`, `top_k`, `top_p`, renormalised over the survivors — as `(id, probability)` over the
    /// ids that survived, in candidate order. `None` for a greedy row (it has no distribution: it
    /// is the argmax). The seam's refusals are the draw's refusals, so a row this answers is a row
    /// [`Sampler::sample`] draws from exactly this.
    ///
    /// This is the number the acceptance law's rate is made of (`p'(draft)`), and what
    /// `saber sample-check --law-out` writes for the prod oracle to compare row for row.
    ///
    /// # Errors
    /// Every refusal [`Sampler::sample`] makes.
    pub fn distribution(
        &self,
        row: &DeviceRow,
        params: &RowParams,
        ctx: &PenaltyContext<'_>,
    ) -> Result<Option<Vec<(TokenId, f32)>>, SamplerError> {
        if row.top_ids.is_empty() {
            return Err(SamplerError::SeamEmpty);
        }
        let k = row.top_ids.len();
        let complete = k >= row.vocab_size;
        if !params.is_greedy() && !complete && params.top_k < row.vocab_size && params.top_k > k {
            return Err(SamplerError::SeamTooNarrowForTopK {
                k,
                top_k: params.top_k,
            });
        }
        let (ids, mut logits) = candidates(row);
        let floor = *row.top_values.last().expect("non-empty");
        let seam = (!complete).then(|| Seam {
            k,
            floor,
            lse_t: row.logsumexp_at_temperature,
            raw: logits.clone(),
        });
        match self.truncate(&mut logits, Ids::Explicit(&ids), seam.as_ref(), params, ctx)? {
            Truncated::Greedy(_) => Ok(None),
            Truncated::Masked { .. } => {
                let probs = softmax(&logits);
                Ok(Some(
                    probs
                        .iter()
                        .enumerate()
                        .filter(|(_, p)| **p > 0.0)
                        .map(|(i, p)| (ids[i], *p))
                        .collect(),
                ))
            }
        }
    }

    /// The pipeline itself, shared by both entry points so the two can never drift.
    ///
    /// `seam` is `None` for a dense row and `Some` for the top-K one, and every place it
    /// is consulted is a place where the answer depends on what the device did **not**
    /// send.
    fn run(
        &self,
        logits: &mut [f32],
        ids: Ids<'_>,
        seam: Option<&Seam>,
        params: &RowParams,
        ctx: &PenaltyContext<'_>,
        noise: &mut dyn NoiseSource,
    ) -> Result<(TokenId, bool), SamplerError> {
        let (greedy_at, tied_at_top) = match self.truncate(logits, ids, seam, params, ctx)? {
            Truncated::Greedy(at) => {
                SamplerCounters::bump(&self.counters.greedy_rows);
                return Ok((ids.at(at), true));
            }
            Truncated::Masked {
                greedy_at,
                tied_at_top,
            } => (greedy_at, tied_at_top),
        };

        // 7. the draw: argmax of probs / Exp(1).
        let probs = softmax(logits);
        let mut best = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (i, p) in probs.iter().enumerate() {
            // Keyed by TOKEN ID, not by position in this row. A seam row and the dense
            // row it came from hold the same tokens in different slots, and noise keyed
            // by slot would make the two draw differently from the same distribution —
            // which is exactly the disagreement `the_seam_agrees_with_the_dense_row`
            // caught the first time it ran.
            let score = p / noise.exponential(ids.at(i) as usize);
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        SamplerCounters::bump(&self.counters.random_rows);
        if tied_at_top.len() > 1 {
            SamplerCounters::bump(&self.counters.top_ties);
            if best != greedy_at && tied_at_top.contains(&best) {
                SamplerCounters::bump(&self.counters.top_tie_draws);
            }
        }
        Ok((ids.at(best), false))
    }

    /// Steps 1–6 of the pipeline: `logits` leaves as the temperature-scaled row with every id the
    /// masks removed at `-inf` — prod's processed logits, whose softmax is `p'`.
    fn truncate(
        &self,
        logits: &mut [f32],
        ids: Ids<'_>,
        seam: Option<&Seam>,
        params: &RowParams,
        ctx: &PenaltyContext<'_>,
    ) -> Result<Truncated, SamplerError> {
        // 1. penalties — repetition, then frequency, then presence, in that order.
        if !params.no_penalties() {
            apply_penalties(logits, ids, params, ctx);
            SamplerCounters::bump(&self.counters.penalised_rows);
        }

        // 2. the greedy pick, taken HERE: on post-penalty, PRE-temperature logits.
        //    `Sampler.sample` computes `greedy_sampled` before it touches temperature,
        //    and a `temperature < eps` row returns it unchanged.
        let greedy_at = argmax(logits);
        if params.is_greedy() {
            // Penalties only ever move ids the host named, and every one of those is a
            // candidate — so the tail is untouched and sits at or below `floor`. The
            // seam's argmax is the row's argmax unless the penalties pushed EVERY
            // candidate under the tail's ceiling, which is a refusal, not a token.
            if let Some(seam) = seam {
                if logits[greedy_at] < seam.floor {
                    return Err(SamplerError::SeamTooNarrow {
                        k: seam.k,
                        top_p: params.top_p,
                        covered_mass: f32::NAN,
                        needed_below: f32::NAN,
                    });
                }
            }
            return Ok(Truncated::Greedy(greedy_at));
        }

        // THE TIE AT THE TOP (ws-f39), recorded before anything rescales the row. A bf16 logit
        // row ties its two best ids far more often than an fp32 one would, and every mask below
        // keeps a tie whole, so a drawn row with a tie at the top can commit an id temperature 0
        // would not — `top_k = 1` included. That is prod's arithmetic, not a defect, and these
        // two counters are what let a served stream's divergence from the greedy one be
        // attributed to it rather than argued about.
        //
        // Both are bumped only once the row is DECIDED (ws-f40): a seam the pipeline refuses below
        // is read again densely and sampled a second time, and a tie counted on both attempts would
        // be one row counted twice.
        let top = logits[greedy_at];
        let tied_at_top: Vec<usize> = (0..logits.len()).filter(|&i| logits[i] == top).collect();

        // 3. temperature.
        for l in logits.iter_mut() {
            *l /= params.temperature;
        }

        // THE TAIL — the ids the seam did not send, every one at or below `floor / T` — enters the
        // pipeline WHOLE and leaves it the way prod's masks leave it (ws-f41). While it is whole
        // its mass is exact (the device's own log-sum-exp minus the candidates'); `min_p` and
        // `top_k` each cut at one threshold, and a cut ABOVE `floor / T` removes every tail id —
        // the common case: prod's own `top_k = 20` inside a 64-wide seam kills the tail outright,
        // and `top_p` then sums over the 20 survivors renormalised, exactly as
        // `apply_top_k_top_p_pytorch` does. A cut AT or below the ceiling leaves an unknown part
        // of the tail alive, and a `top_p` over that is a refusal (the dense row answers). Before
        // F41 the tail's mass entered `top_p` whatever `top_k` had done, so the cumulative sum ran
        // over `20 + tail` and the cut kept ids prod masks.
        let ceiling = seam.map(|s| s.floor / params.temperature);
        let mut tail = match seam {
            None => TailLife::Dead,
            Some(_) => TailLife::Whole,
        };

        // 4. min_p — an argmax-invariant logits processor, and the reference server runs those BEFORE
        //    top_k/top_p. `prob_i < min_p * max_prob`; the shared denominator cancels,
        //    so the tail cannot change the answer and no logsumexp is needed.
        if params.min_p > 0.0 {
            if let Some(ceiling) = ceiling {
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                // A tail id at the ceiling is masked iff `exp(ceiling - max) < min_p`; every
                // other tail id is below it.
                tail = if (ceiling - max).exp() < params.min_p {
                    TailLife::Dead
                } else {
                    tail.cut()
                };
            }
            if apply_min_p(logits, params.min_p) {
                SamplerCounters::bump(&self.counters.min_p_rows);
            }
        }

        // 5. top_k — mask everything strictly below the k-th largest. Ties at the
        //    threshold survive, which is what `logits_sort < top_k_value` means.
        if params.top_k < params.vocab_size {
            if let Some(ceiling) = ceiling {
                // Prod's k-th largest is over the WHOLE row; the candidates hold every id above
                // the ceiling, so when theirs is above it the two are the same number and every
                // tail id is strictly below it. A `top_k` as wide as the candidate list cuts at
                // its smallest, which is at or below the ceiling: the tail ids tied there live.
                let threshold = kth_largest(logits, params.top_k)
                    .unwrap_or_else(|| logits.iter().copied().fold(f32::INFINITY, f32::min));
                tail = if threshold > ceiling {
                    TailLife::Dead
                } else {
                    tail.cut()
                };
            }
            if apply_top_k(logits, params.top_k) {
                SamplerCounters::bump(&self.counters.top_k_rows);
            }
        }

        // 6. top_p — ascending sort, ascending cumulative softmax, mask `cumsum <= 1-p`,
        //    always keep the last (largest) entry. Over the SURVIVORS, renormalised: the tail
        //    enters the sum only while it survived whole.
        if params.top_p < 1.0 {
            let tail_mass = match (tail, seam) {
                (TailLife::Dead, _) | (_, None) => Tail::default(),
                (TailLife::Whole, Some(seam)) => seam.tail(logits, params)?,
                (TailLife::Partial, Some(seam)) => {
                    // Some of the tail lives — the ids tied at the cut — and the seam cannot say
                    // how much. The surviving mass lies between none of it and all of it, and the
                    // cut is monotone in it: when both bounds give the same surviving set, that
                    // set is prod's whatever the tail held (and `p'` over it does not read the
                    // tail at all). When they differ the row is refused and read densely.
                    let whole = Tail {
                        mass: seam.mass(logits, params)?,
                    };
                    let mut none = logits.to_vec();
                    let mut all = logits.to_vec();
                    apply_top_p(&mut none, params.top_p, Tail::default());
                    apply_top_p(&mut all, params.top_p, whole);
                    let alive = |v: &[f32]| v.iter().map(|l| l.is_finite()).collect::<Vec<_>>();
                    if alive(&none) != alive(&all) {
                        return Err(SamplerError::SeamUntruncated {
                            k: seam.k,
                            top_p: params.top_p,
                            top_k: params.top_k,
                            min_p: params.min_p,
                        });
                    }
                    Tail::default()
                }
            };
            if apply_top_p(logits, params.top_p, tail_mass) {
                SamplerCounters::bump(&self.counters.top_p_rows);
            }
        }

        // Every one of those three masks is `value < threshold` for one threshold, so the
        // surviving set is a suffix of the value order. The tail is entirely below
        // `floor / T`; it is therefore entirely masked precisely when nothing at or below
        // `floor / T` survived. If something did, a token the device never sent could
        // have won the draw, and the only honest answer is a refusal.
        if let (Some(seam), Some(ceiling)) = (seam, ceiling) {
            let survivor_under_ceiling = logits.iter().any(|l| l.is_finite() && *l <= ceiling);
            if survivor_under_ceiling {
                return Err(SamplerError::SeamUntruncated {
                    k: seam.k,
                    top_p: params.top_p,
                    top_k: params.top_k,
                    min_p: params.min_p,
                });
            }
        }
        Ok(Truncated::Masked {
            greedy_at,
            tied_at_top,
        })
    }
}

/// What [`Sampler::truncate`] made of a row.
enum Truncated {
    /// A greedy row: its argmax, and nothing was masked.
    Greedy(usize),
    /// A drawn row: the masked, temperature-scaled logits are in place.
    Masked {
        greedy_at: usize,
        tied_at_top: Vec<usize>,
    },
}

/// How much of a seam's tail survives the masks so far (ws-f41).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TailLife {
    /// Every tail id survives: its mass is the device's log-sum-exp minus the candidates'.
    Whole,
    /// A mask cut at or below the ceiling: some tail ids may survive and the seam cannot say which.
    Partial,
    /// Every tail id is masked (or there is no tail: a dense row).
    Dead,
}

impl TailLife {
    /// A mask whose threshold is at or below the ceiling: a live tail becomes partial.
    fn cut(self) -> Self {
        match self {
            TailLife::Dead => TailLife::Dead,
            TailLife::Whole | TailLife::Partial => TailLife::Partial,
        }
    }
}

/// The `k`-th largest value of `logits` (1-based) — `apply_top_k`'s threshold — or `None` when
/// `k` is 0 or covers the row (nothing is cut).
fn kth_largest(logits: &[f32], k: usize) -> Option<f32> {
    if k == 0 || k >= logits.len() {
        return None;
    }
    let mut sorted: Vec<f32> = logits.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    Some(sorted[k - 1])
}

// ── the steps ───────────────────────────────────────────────────────────────────

/// `upstream/model_executor/layers/utils.py::apply_penalties`, in its order.
///
/// The repetition penalty is the CPU form: `scaling = 1/p` for a positive logit and `p`
/// for a non-positive one, then **multiply**. That is not the same float as `logit / p`
/// — the reciprocal rounds first — and prod's CUDA kernel takes the division instead,
/// so the two differ in the last ulp. The gate runs prod on the CPU, so this is the
/// arithmetic it compares; the divergence is recorded in `docs/LEDGER.md`.
fn apply_penalties(logits: &mut [f32], ids: Ids<'_>, params: &RowParams, ctx: &PenaltyContext<'_>) {
    let mut prompt: HashMap<TokenId, ()> = HashMap::new();
    for id in ctx.prompt {
        prompt.insert(*id, ());
    }
    let mut output: HashMap<TokenId, u32> = HashMap::new();
    for id in ctx.output {
        *output.entry(*id).or_insert(0) += 1;
    }
    for (i, l) in logits.iter_mut().enumerate() {
        let id = ids.at(i);
        let count = output.get(&id).copied().unwrap_or(0);
        let seen = prompt.contains_key(&id) || count > 0;
        if params.repetition_penalty != 1.0 {
            let penalty = if seen { params.repetition_penalty } else { 1.0 };
            let scaling = if *l > 0.0 { 1.0 / penalty } else { penalty };
            *l *= scaling;
        }
        *l -= params.frequency_penalty * (count as f32);
        *l -= params.presence_penalty * f32::from(count > 0);
    }
}

/// `torch.argmax`: the first index of the largest value. NaN never wins, which matches
/// `saber_argmax_bf16` — a row of NaN is a broken forward pass, not a token.
fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (i, l) in logits.iter().enumerate() {
        if *l > best_value {
            best_value = *l;
            best = i;
        }
    }
    best
}

/// `torch.softmax(x, -1)` = `exp(x - max) / sum`, in that association.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        // Every entry masked, or a NaN row: prod would produce NaN here too; returning
        // zeros keeps the caller's argmax deterministic and the counters honest.
        return vec![0.0; logits.len()];
    }
    let shifted: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = shifted.iter().sum();
    shifted.iter().map(|e| e / sum).collect()
}

/// `torch.log_softmax(x, -1, dtype=float32)` = `(x - max) - log(sum(exp(x - max)))`.
/// The association matters: `x - (max + log(sum))` is a different float.
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|l| (l - max).exp()).sum();
    let log_sum = sum.ln();
    logits.iter().map(|l| (l - max) - log_sum).collect()
}

/// `MinPLogitsProcessor.apply`. Returns whether anything was masked.
fn apply_min_p(logits: &mut [f32], min_p: f32) -> bool {
    let probs = softmax(logits);
    let max = probs.iter().copied().fold(0.0f32, f32::max);
    let threshold = max * min_p;
    let mut masked = false;
    for (l, p) in logits.iter_mut().zip(probs.iter()) {
        if *p < threshold {
            *l = f32::NEG_INFINITY;
            masked = true;
        }
    }
    masked
}

/// `apply_top_k_only` / the sorted form's top-k half: mask everything **strictly** below
/// the k-th largest logit, so a tie at the threshold keeps every member.
fn apply_top_k(logits: &mut [f32], k: usize) -> bool {
    if k == 0 || k >= logits.len() {
        return false;
    }
    let mut sorted: Vec<f32> = logits.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted[k - 1];
    let mut masked = false;
    for l in logits.iter_mut() {
        if *l < threshold {
            *l = f32::NEG_INFINITY;
            masked = true;
        }
    }
    masked
}

/// `apply_top_k_top_p_pytorch`'s top-p half.
///
/// Sorted **ascending**, softmax over the sorted row, cumulative sum **ascending**,
/// masked where `cumsum <= 1 - p`, with the last (largest) entry forced to survive.
/// Summing from the bottom is not a detail: it is why the cut is decided by the tail's
/// mass and why a top-K seam has to know that mass exactly ([`Tail`]).
fn apply_top_p(logits: &mut [f32], top_p: f32, tail: Tail) -> bool {
    let n = logits.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| {
        logits[*a]
            .partial_cmp(&logits[*b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    });
    // `logits_sort.softmax(-1)` — the sum runs over the SORTED row, ascending, which is
    // a different float from the same sum in vocabulary order and is the one prod takes.
    let max = order.last().map_or(f32::NEG_INFINITY, |i| logits[*i]);
    let mut sum = 0.0f32;
    for idx in &order {
        sum += (logits[*idx] - max).exp();
    }
    let mut probs = vec![0.0f32; n];
    for idx in &order {
        probs[*idx] = (logits[*idx] - max).exp() / sum;
    }
    // The probabilities are over the WHOLE distribution: the candidates carry
    // `1 - tail.mass` of it and the tail carries the rest, entirely below the smallest
    // candidate that the seam kept.
    let scale = 1.0 - tail.mass;
    let mut cumulative = tail.mass;
    let cut = 1.0 - top_p;
    let mut masked = false;
    for (rank, idx) in order.iter().enumerate() {
        let is_last = rank + 1 == n;
        cumulative += probs[*idx] * scale;
        if !is_last && cumulative <= cut {
            logits[*idx] = f32::NEG_INFINITY;
            masked = true;
        }
    }
    masked
}

/// The reference server's `Sampler.gather_logprobs` followed by the OpenAI layer's own ordering.
///
/// Two things here are schema, not arithmetic, and neither is guessable from the
/// OpenAI spec:
///
/// * the **sampled token comes first**, whether or not it is in the top-k. `LogprobsProcessor`
///   builds a Python dict keyed by token id with the sampled id inserted first, and the
///   OpenAI layer takes the first `top_logprobs` entries of it **in insertion order**. So
///   a sampled token outside the top-k pushes the k-th entry out of the reply.
/// * every logprob is clamped at `-9999.0`.
///
/// `rank_floor` is the seam's honesty knob: above it the candidate list holds every
/// logprob that could outrank the sampled token, so the rank is exact; below it the
/// device did not send enough to count and the rank is `None` rather than wrong.
fn gather_logprobs(
    raw_logprobs: &[f32],
    ids: Ids<'_>,
    sampled: TokenId,
    params: &RowParams,
    rank_floor: Option<f32>,
) -> SampledLogprobs {
    let k = params.logprobs.unwrap_or(0).min(raw_logprobs.len());
    let mut order: Vec<usize> = (0..raw_logprobs.len()).collect();
    order.sort_by(|a, b| {
        raw_logprobs[*b]
            .partial_cmp(&raw_logprobs[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    });
    let at = (0..raw_logprobs.len())
        .find(|i| ids.at(*i) == sampled)
        .unwrap_or(0);
    let sampled_logprob = raw_logprobs[at];
    // `batched_count_greater_than`: the rank is how many logprobs are >= this one, so
    // it is 1-based and counts the token itself.
    let rank_is_exact = rank_floor.is_none_or(|floor| sampled_logprob >= floor);
    let rank = rank_is_exact.then(|| {
        raw_logprobs
            .iter()
            .filter(|lp| **lp >= sampled_logprob)
            .count() as u32
    });
    let sampled_entry = TokenLogprob {
        token_id: sampled,
        logprob: sampled_logprob.max(LOGPROB_FLOOR),
        rank,
    };
    let top = order
        .iter()
        .take(k)
        .filter(|i| ids.at(**i) != sampled)
        .map(|i| TokenLogprob {
            token_id: ids.at(*i),
            logprob: raw_logprobs[*i].max(LOGPROB_FLOOR),
            rank: None,
        })
        .collect();
    SampledLogprobs {
        sampled: sampled_entry,
        top,
    }
}

// ── seam reconstruction ─────────────────────────────────────────────────────────

/// The candidate list: the top-K, then every penalised id the top-K does not already
/// carry, with its raw logit.
fn candidates(row: &DeviceRow) -> (Vec<TokenId>, Vec<f32>) {
    let mut ids = row.top_ids.clone();
    let mut logits = row.top_values.clone();
    for (id, logit) in &row.penalised {
        if !ids.contains(id) {
            ids.push(*id);
            logits.push(*logit);
        }
    }
    (ids, logits)
}

/// What the device did **not** send, and everything the pipeline needs to know about it.
#[derive(Debug, Clone)]
struct Seam {
    /// How many candidates the head sent.
    k: usize,
    /// The K-th largest RAW logit. Every id the seam left out is at or below it.
    floor: f32,
    /// `logsumexp` of the whole RAW row divided by the row's temperature.
    lse_t: Option<f32>,
    /// The candidates' raw logits, parallel to the working row — the penalties move the
    /// working copy and the tail's share has to be measured against what was there before.
    raw: Vec<f32>,
}

impl Seam {
    /// The tail's share of the temperature-scaled distribution, **after** the penalties.
    ///
    /// Penalties move only candidates, so the tail's `sum(exp(l/T))` is unchanged: it is
    /// the device's own total minus the candidates' raw contribution. The new total is
    /// that plus the candidates' post-penalty contribution, and the ratio is exact.
    ///
    /// Read only while the tail is WHOLE (ws-f41): once `top_k` or `min_p` has cut above the
    /// ceiling there is no tail left to weigh, and `top_p` sums over the survivors alone.
    fn tail(&self, scaled: &[f32], params: &RowParams) -> Result<Tail, SamplerError> {
        if params.top_p >= 1.0 {
            // Nothing below reads the tail: `min_p`'s threshold is relative (the
            // denominator cancels) and `top_k`'s is a rank. Only `top_p` sums.
            return Ok(Tail::default());
        }
        let mass = self.mass(scaled, params)?;
        let cut = 1.0 - params.top_p;
        if mass > cut {
            return Err(SamplerError::SeamTooNarrow {
                k: self.k,
                top_p: params.top_p,
                covered_mass: 1.0 - mass,
                needed_below: cut,
            });
        }
        Ok(Tail { mass })
    }

    /// The WHOLE tail's share of the distribution whose surviving candidates are `scaled` (a
    /// masked candidate is `-inf` and weighs nothing).
    fn mass(&self, scaled: &[f32], params: &RowParams) -> Result<f32, SamplerError> {
        let Some(lse_t) = self.lse_t else {
            return Err(SamplerError::SeamNoTemperedLogsumexp {
                k: self.k,
                top_p: params.top_p,
            });
        };
        // Everything in one shifted space, so the subtraction cannot overflow.
        let mut raw_candidates = 0.0f64;
        for l in &self.raw {
            raw_candidates += f64::from(l / params.temperature - lse_t).exp();
        }
        let tail_sum = (1.0f64 - raw_candidates).max(0.0);
        let mut new_candidates = 0.0f64;
        for l in scaled {
            new_candidates += f64::from(*l - lse_t).exp();
        }
        let total = tail_sum + new_candidates;
        Ok(if total > 0.0 {
            (tail_sum / total) as f32
        } else {
            0.0
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(temperature: f32) -> RowParams {
        RowParams::resolve(
            &SamplingParams {
                temperature,
                ..Default::default()
            },
            8,
        )
    }

    #[test]
    fn temperature_zero_is_the_argmax_and_nothing_else() {
        let sampler = Sampler::new();
        let mut logits = vec![0.1, 5.0, -3.0, 5.0, 0.0, 1.0, 2.0, -1.0];
        let out = sampler
            .sample_dense(
                &mut logits,
                &params(0.0),
                &PenaltyContext::default(),
                &mut SeededNoise::new(1, 0, 0),
            )
            .expect("greedy");
        // Ties go to the LOWEST id, exactly as `saber_argmax_bf16` reduces.
        assert_eq!(out.token_id, 1);
        assert!(out.greedy);
        assert_eq!(sampler.counters.greedy_rows.load(Ordering::Relaxed), 1);
        assert_eq!(sampler.counters.random_rows.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn a_greedy_request_cannot_be_given_a_truncation() {
        let p = RowParams::resolve(
            &SamplingParams {
                temperature: 0.0,
                top_p: 0.1,
                top_k: 2,
                min_p: 0.5,
                ..Default::default()
            },
            8,
        );
        assert_eq!((p.top_p, p.top_k, p.min_p), (1.0, 8, 0.0));
    }

    #[test]
    fn a_tiny_temperature_is_raised_to_the_floor() {
        let p = RowParams::resolve(
            &SamplingParams {
                temperature: 1e-4,
                ..Default::default()
            },
            8,
        );
        assert_eq!(p.temperature, MAX_TEMP);
        assert!(!p.is_greedy(), "1e-2 is above the greedy epsilon");
    }

    #[test]
    fn top_k_keeps_ties_at_the_threshold() {
        let mut logits = vec![1.0, 3.0, 3.0, 2.0];
        assert!(apply_top_k(&mut logits, 2));
        assert_eq!(logits, vec![f32::NEG_INFINITY, 3.0, 3.0, f32::NEG_INFINITY]);
    }

    /// **`top_k = 1` IS THE ARGMAX EXCEPT AT A TIE** (ws-f39). The mask is `logits < the k-th
    /// largest`, prod's `apply_top_k_only`, so an exact tie at the top survives whole and the draw
    /// chooses between its members — where temperature 0 takes the lowest id. With no tie the
    /// draw can only return the argmax, over hundreds of keys; with one, it returns the higher id
    /// on a fair share of them, and `top_tie_draws` counts exactly those. The control (no tie) is
    /// what fails if the counter or the law were a property of `top_k = 1` rather than of the tie.
    #[test]
    fn top_k_one_is_the_argmax_except_at_an_exact_tie_at_the_top() {
        let p = RowParams::resolve(
            &SamplingParams {
                temperature: 0.7,
                top_k: 1,
                ..Default::default()
            },
            8,
        );
        let draw = |logits: &[f32], key: u64, sampler: &Sampler| -> TokenId {
            let mut row = logits.to_vec();
            sampler
                .sample_dense(
                    &mut row,
                    &p,
                    &PenaltyContext::default(),
                    &mut SeededNoise::new(key, 0, 0),
                )
                .expect("a top_k 1 row is always decidable")
                .token_id
        };
        // The control: a clear winner at id 3, one bf16 step (0.03125 at 5.0) above id 1. Every
        // key gives the argmax; nothing is counted.
        let clear = [0.1, 5.0, -3.0, 5.03125, 0.0, 1.0, 2.0, -1.0];
        let s = Sampler::new();
        for key in 0..400 {
            assert_eq!(
                draw(&clear, key, &s),
                3,
                "key {key}: a top_k 1 draw left the argmax"
            );
        }
        assert_eq!(s.counters.top_ties.load(Ordering::Relaxed), 0);
        assert_eq!(s.counters.top_tie_draws.load(Ordering::Relaxed), 0);

        // The case: ids 1 and 3 tie at 5.0.
        let tied = [0.1, 5.0, -3.0, 5.0, 0.0, 1.0, 2.0, -1.0];
        let s = Sampler::new();
        let mut high = 0u64;
        for key in 0..400 {
            let t = draw(&tied, key, &s);
            assert!(
                t == 1 || t == 3,
                "key {key}: drew {t}, which is not one of the tied ids"
            );
            high += u64::from(t == 3);
        }
        assert!(
            (120..=280).contains(&high),
            "the tie was not split fairly: the higher id was drawn {high} times in 400"
        );
        assert_eq!(s.counters.top_ties.load(Ordering::Relaxed), 400);
        assert_eq!(
            s.counters.top_tie_draws.load(Ordering::Relaxed),
            high,
            "every draw that left the lowest id is a tie draw, and only those"
        );
        // Temperature 0 over the same row: the lowest id, always, and it is not a tie DRAW.
        let g = Sampler::new();
        let mut row = tied.to_vec();
        let greedy = g
            .sample_dense(
                &mut row,
                &params(0.0),
                &PenaltyContext::default(),
                &mut SeededNoise::new(1, 0, 0),
            )
            .expect("greedy");
        assert_eq!(greedy.token_id, 1);
        assert_eq!(g.counters.top_tie_draws.load(Ordering::Relaxed), 0);
    }

    /// A TIE IS COUNTED ONCE PER DECIDED ROW (ws-f40). A seam the pipeline refuses is read again
    /// densely and sampled a second time; before F40 `top_ties` was bumped on the refused attempt
    /// too, so such a row counted its tie twice. The refusal counts nothing tie-shaped; the dense
    /// decision counts one tie, and one tie draw exactly when it left the lowest id.
    #[test]
    fn a_refused_seam_counts_no_tie_and_the_dense_decision_counts_one() {
        let vocab = 16usize;
        // Ids 2 and 5 tie at the top; the rest are well below. No truncation at all (T = 1, no
        // top_k / top_p / min_p), so the 4-wide seam cannot decide the row and refuses it.
        let mut dense = vec![-4.0f32; vocab];
        dense[2] = 3.0;
        dense[5] = 3.0;
        dense[7] = 1.0;
        let max = 3.0f32;
        let lse = max + dense.iter().map(|l| (l - max).exp()).sum::<f32>().ln();
        let row = DeviceRow {
            top_values: vec![3.0, 3.0, 1.0, -4.0],
            top_ids: vec![2, 5, 7, 0],
            logsumexp: lse,
            logsumexp_at_temperature: Some(lse),
            penalised: Vec::new(),
            vocab_size: vocab,
        };
        let p = RowParams::resolve(
            &SamplingParams {
                temperature: 1.0,
                ..Default::default()
            },
            vocab,
        );
        let s = Sampler::new();
        let (mut ties, mut off) = (0u64, 0u64);
        for key in 0..64u64 {
            let mut noise = SeededNoise::new(key, 3, 0);
            let err = s
                .sample(&row, &p, &PenaltyContext::default(), &mut noise)
                .expect_err("an untruncated row is not the seam's to decide");
            assert!(matches!(err, SamplerError::SeamUntruncated { .. }), "{err}");
            assert_eq!(
                s.counters.top_ties.load(Ordering::Relaxed),
                ties,
                "key {key}: the REFUSED attempt counted a tie"
            );
            let mut work = dense.clone();
            let out = s
                .sample_dense(&mut work, &p, &PenaltyContext::default(), &mut noise)
                .expect("the dense row decides it");
            ties += 1;
            off += u64::from(out.token_id != 2 && (out.token_id == 5));
        }
        assert_eq!(s.counters.top_ties.load(Ordering::Relaxed), ties);
        assert_eq!(s.counters.top_tie_draws.load(Ordering::Relaxed), off);
        assert_eq!(s.counters.seam_refusals.load(Ordering::Relaxed), 64);
        assert_eq!(s.counters.random_rows.load(Ordering::Relaxed), 64);
    }

    #[test]
    fn top_p_always_keeps_at_least_one() {
        let mut logits = vec![0.0, 0.0, 0.0, 20.0];
        assert!(apply_top_p(&mut logits, 0.0001, Tail::default()));
        assert_eq!(logits.iter().filter(|l| l.is_finite()).count(), 1);
    }

    #[test]
    fn min_p_is_relative_to_the_maximum() {
        // p = [~1, ~0, ~0, ~0]; min_p 0.5 keeps only the peak.
        let mut logits = vec![10.0, 0.0, 0.0, 0.0];
        assert!(apply_min_p(&mut logits, 0.5));
        assert_eq!(logits.iter().filter(|l| l.is_finite()).count(), 1);
    }

    #[test]
    fn repetition_penalty_divides_positive_and_multiplies_negative() {
        let p = RowParams::resolve(
            &SamplingParams {
                temperature: 1.0,
                repetition_penalty: 2.0,
                ..Default::default()
            },
            4,
        );
        let mut logits = vec![4.0, -4.0, 4.0, -4.0];
        apply_penalties(
            &mut logits,
            Ids::Identity,
            &p,
            &PenaltyContext {
                prompt: &[0, 1],
                output: &[],
            },
        );
        assert_eq!(logits, vec![4.0 * 0.5, -4.0 * 2.0, 4.0, -4.0]);
    }

    #[test]
    fn frequency_penalty_counts_the_output_and_presence_does_not() {
        let p = RowParams::resolve(
            &SamplingParams {
                temperature: 1.0,
                frequency_penalty: 0.5,
                presence_penalty: 1.0,
                ..Default::default()
            },
            4,
        );
        let mut logits = vec![0.0, 0.0, 0.0, 0.0];
        apply_penalties(
            &mut logits,
            Ids::Identity,
            &p,
            &PenaltyContext {
                prompt: &[3],
                output: &[1, 1, 1, 2],
            },
        );
        // id 1 appears three times: -0.5*3 - 1.0. id 2 once: -0.5 - 1.0. id 3 is in the
        // prompt only, so neither penalty touches it.
        assert_eq!(logits, vec![0.0, -2.5, -1.5, 0.0]);
    }

    #[test]
    fn logprobs_put_the_sampled_token_first_and_drop_the_last_top() {
        // Sampled id 3 is NOT in the top 2 (ids 1, 0) — prod's dict is
        // {3: .., 1: .., 0: ..} and the OpenAI layer takes the first 2.
        let raw = vec![1.0f32, 2.0, -5.0, -1.0];
        let p = RowParams {
            logprobs: Some(2),
            ..params(1.0)
        };
        let out = gather_logprobs(&raw, Ids::Identity, 3, &p, None);
        assert_eq!(out.sampled.token_id, 3);
        assert_eq!(out.sampled.rank, Some(3));
        assert_eq!(
            out.top.iter().map(|t| t.token_id).collect::<Vec<_>>(),
            vec![1, 0],
            "the sampled token is first, so only k-1 of the top-k follow it"
        );
    }

    #[test]
    fn logprobs_are_clamped_at_prods_floor() {
        let raw = vec![0.0f32, -1e30];
        let p = RowParams {
            logprobs: Some(2),
            ..params(1.0)
        };
        let out = gather_logprobs(&raw, Ids::Identity, 1, &p, None);
        assert_eq!(out.sampled.logprob, LOGPROB_FLOOR);
    }

    /// The seam has to answer what the dense row answers. This is the property that
    /// makes a top-K head legitimate, and it is checked rather than asserted.
    #[test]
    fn the_seam_agrees_with_the_dense_row() {
        let vocab = 512usize;
        let mut state = 12345u64;
        let mut next = || {
            // Three uniforms summed: an approximately normal row with a sharp peak,
            // which is the shape a language-model logit row actually has. A FLAT row
            // (one uniform) puts 11% of its mass beyond the 64th token and is refused
            // by the seam, correctly — that case is the negative control below.
            let mut sum = 0.0f32;
            for _ in 0..3 {
                state = splitmix64(state);
                sum += (state >> 8) as f32 / (1u64 << 56) as f32;
            }
            (sum - 1.5) * 12.0
        };
        let dense: Vec<f32> = (0..vocab).map(|_| next()).collect();
        for (temperature, top_p, top_k, min_p) in [
            // Prod's own defaults, and three more that all truncate inside K = 64.
            (0.7f32, 0.95f32, 20i32, 0.0f32),
            (0.7, 0.95, 0, 0.0),
            (1.0, 1.0, 16, 0.0),
            (0.7, 1.0, 0, 0.30),
            (1.3, 0.8, 50, 0.01),
        ] {
            let sp = SamplingParams {
                temperature,
                top_p,
                top_k,
                min_p,
                ..Default::default()
            };
            let p = RowParams::resolve(&sp, vocab);
            let sampler = Sampler::new();
            let mut copy = dense.clone();
            let want = sampler
                .sample_dense(
                    &mut copy,
                    &p,
                    &PenaltyContext::default(),
                    &mut SeededNoise::new(7, 0, 0),
                )
                .expect("dense");

            // Build the seam the way a head would: the 64 largest, both log-sum-exps.
            let k = 64usize;
            let mut order: Vec<usize> = (0..vocab).collect();
            order.sort_by(|a, b| {
                dense[*b]
                    .partial_cmp(&dense[*a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(b))
            });
            let max = dense.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lse = max + dense.iter().map(|l| (l - max).exp()).sum::<f32>().ln();
            let scaled: Vec<f32> = dense.iter().map(|l| l / temperature).collect();
            let max_t = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lse_t = max_t + scaled.iter().map(|l| (l - max_t).exp()).sum::<f32>().ln();
            let row = DeviceRow {
                top_values: order[..k].iter().map(|i| dense[*i]).collect(),
                top_ids: order[..k].iter().map(|i| *i as TokenId).collect(),
                logsumexp: lse,
                logsumexp_at_temperature: Some(lse_t),
                penalised: Vec::new(),
                vocab_size: vocab,
            };
            let got = sampler
                .sample(
                    &row,
                    &p,
                    &PenaltyContext::default(),
                    &mut SeededNoise::new(7, 0, 0),
                )
                .expect("seam");
            assert_eq!(
                got.token_id, want.token_id,
                "seam and dense disagree at T={temperature} top_p={top_p} top_k={top_k} min_p={min_p}"
            );
        }
    }

    /// The negative control for the test above (structural rule 8): a row with **no**
    /// truncation samples from the whole vocabulary, and a 64-wide seam has to say so.
    #[test]
    fn an_untruncated_row_is_refused_rather_than_truncated_silently() {
        let vocab = 512usize;
        let mut state = 999u64;
        let mut next = || {
            // Three uniforms summed: an approximately normal row with a sharp peak,
            // which is the shape a language-model logit row actually has. A FLAT row
            // (one uniform) puts 11% of its mass beyond the 64th token and is refused
            // by the seam, correctly — that case is the negative control below.
            let mut sum = 0.0f32;
            for _ in 0..3 {
                state = splitmix64(state);
                sum += (state >> 8) as f32 / (1u64 << 56) as f32;
            }
            (sum - 1.5) * 12.0
        };
        let dense: Vec<f32> = (0..vocab).map(|_| next()).collect();
        let mut order: Vec<usize> = (0..vocab).collect();
        order.sort_by(|a, b| {
            dense[*b]
                .partial_cmp(&dense[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(b))
        });
        let max = dense.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse = max + dense.iter().map(|l| (l - max).exp()).sum::<f32>().ln();
        let row = DeviceRow {
            top_values: order[..64].iter().map(|i| dense[*i]).collect(),
            top_ids: order[..64].iter().map(|i| *i as TokenId).collect(),
            logsumexp: lse,
            logsumexp_at_temperature: Some(lse),
            penalised: Vec::new(),
            vocab_size: vocab,
        };
        let p = RowParams::resolve(
            &SamplingParams {
                temperature: 1.0,
                ..Default::default()
            },
            vocab,
        );
        let sampler = Sampler::new();
        let err = sampler
            .sample(
                &row,
                &p,
                &PenaltyContext::default(),
                &mut SeededNoise::new(7, 0, 0),
            )
            .expect_err("no truncation at all");
        assert!(matches!(err, SamplerError::SeamUntruncated { .. }), "{err}");
        assert_eq!(sampler.counters.seam_refusals.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn a_seam_too_narrow_for_top_p_refuses_by_name() {
        // A flat row: 4 of 4096 candidates cannot hold 99% of a uniform distribution.
        let vocab = 4096usize;
        let row = DeviceRow {
            top_values: vec![0.0; 4],
            top_ids: vec![0, 1, 2, 3],
            logsumexp: (vocab as f32).ln(),
            logsumexp_at_temperature: Some((vocab as f32).ln()),
            penalised: Vec::new(),
            vocab_size: vocab,
        };
        let p = RowParams::resolve(
            &SamplingParams {
                temperature: 1.0,
                top_p: 0.9,
                ..Default::default()
            },
            vocab,
        );
        let sampler = Sampler::new();
        let err = sampler
            .sample(
                &row,
                &p,
                &PenaltyContext::default(),
                &mut SeededNoise::new(1, 0, 0),
            )
            .expect_err("a 4-wide seam cannot decide a uniform 4096 row");
        assert!(matches!(err, SamplerError::SeamTooNarrow { .. }));
        assert_eq!(sampler.counters.seam_refusals.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn top_p_without_a_tempered_logsumexp_refuses_by_name() {
        let row = DeviceRow {
            top_values: vec![3.0, 2.0],
            top_ids: vec![5, 6],
            logsumexp: 4.0,
            logsumexp_at_temperature: None,
            penalised: Vec::new(),
            vocab_size: 1000,
        };
        let p = RowParams::resolve(
            &SamplingParams {
                temperature: 0.7,
                top_p: 0.95,
                ..Default::default()
            },
            1000,
        );
        let sampler = Sampler::new();
        let err = sampler
            .sample(
                &row,
                &p,
                &PenaltyContext::default(),
                &mut SeededNoise::new(1, 0, 0),
            )
            .expect_err("no tempered logsumexp");
        assert!(matches!(err, SamplerError::SeamNoTemperedLogsumexp { .. }));
    }

    /// The draw has to be the distribution it was handed. This is the one property the
    /// corpus gate cannot check on its own — it compares two samplers, and two samplers
    /// biased the same way agree (blind spot 1).
    #[test]
    fn the_draw_reproduces_the_distribution_it_was_given() {
        let vocab = 4usize;
        // ln(1/2), ln(0.3), ln(0.2): a distribution with a clear shape, written as
        // logits so the pipeline does the softmax rather than the test.
        let logits: Vec<f32> = vec![0.0, -std::f32::consts::LN_2, -1.203_972_8, -1.609_438];
        let want = softmax(&logits);
        let draws = 200_000usize;
        let p = RowParams::resolve(
            &SamplingParams {
                temperature: 1.0,
                ..Default::default()
            },
            vocab,
        );
        let sampler = Sampler::new();
        let mut counts = vec![0u64; vocab];
        for step in 0..draws {
            let mut row = logits.clone();
            let out = sampler
                .sample_dense(
                    &mut row,
                    &p,
                    &PenaltyContext::default(),
                    &mut SeededNoise::new(4242, step as u64, 0),
                )
                .expect("dense");
            counts[out.token_id as usize] += 1;
        }
        for (i, expected_p) in want.iter().enumerate() {
            let expected = f64::from(*expected_p) * draws as f64;
            let sd = (expected * (1.0 - f64::from(*expected_p))).sqrt();
            let z = (counts[i] as f64 - expected) / sd;
            assert!(
                z.abs() < 5.0,
                "token {i}: drew {} where {expected:.0} was expected ({z:.2} sd)",
                counts[i]
            );
        }
    }

    #[test]
    fn seeded_noise_is_a_function_of_the_key_alone() {
        let a: Vec<f32> = (0..16)
            .map(|i| SeededNoise::new(9, 3, 1).exponential(i))
            .collect();
        let b: Vec<f32> = (0..16)
            .map(|i| SeededNoise::new(9, 3, 1).exponential(i))
            .collect();
        let c: Vec<f32> = (0..16)
            .map(|i| SeededNoise::new(9, 4, 1).exponential(i))
            .collect();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a.iter().all(|x| x.is_finite() && *x > 0.0));
    }

    // ── the TP > 1 merge ────────────────────────────────────────────────────

    /// A plausible logit row: three uniforms summed, which is peaked the way a real one is.
    fn synthetic_row(seed: u64, n: usize) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                let mut sum = 0.0f32;
                for _ in 0..3 {
                    state = splitmix64(state);
                    sum += (state >> 8) as f32 / (1u64 << 56) as f32;
                }
                (sum - 1.5) * 12.0
            })
            .collect()
    }

    /// The seam a head would write over `logits[base..base + cols]`, in the kernel's own order.
    fn shard_of(
        dense: &[f32],
        base: u32,
        cols: u32,
        k: usize,
        temperature: f32,
        penalty_ids: &[TokenId],
    ) -> SeamShard {
        let lo = base as usize;
        let hi = lo + cols as usize;
        let mine = &dense[lo..hi];
        let mut order: Vec<usize> = (0..mine.len()).collect();
        order.sort_by(|a, b| {
            mine[*b]
                .partial_cmp(&mine[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(b))
        });
        let lse = |scale: f32| {
            let max = mine
                .iter()
                .map(|l| l / scale)
                .fold(f32::NEG_INFINITY, f32::max);
            max + mine
                .iter()
                .map(|l| (l / scale - max).exp())
                .sum::<f32>()
                .ln()
        };
        SeamShard {
            base,
            cols,
            top_values: order[..k].iter().map(|i| mine[*i]).collect(),
            top_cols: order[..k].iter().map(|i| *i as u32).collect(),
            logsumexp: lse(1.0),
            logsumexp_at_temperature: Some(lse(temperature)),
            penalised: penalty_ids
                .iter()
                .map(|id| (*id >= base && *id < base + cols).then(|| dense[*id as usize]))
                .collect(),
        }
    }

    /// TWO SHARDS ARE ONE ROW. The merged seam decides the same token as the dense row it was
    /// split from, at five parameter sets, and its top-K is the dense row's own top-K by value
    /// and by id.
    #[test]
    fn two_shards_merge_into_the_row_they_were_split_from() {
        let vocab = 512usize;
        let k = 64usize;
        let dense = synthetic_row(31_337, vocab);
        for (temperature, top_p, top_k, min_p) in [
            (0.7f32, 0.95f32, 20i32, 0.0f32),
            (0.7, 0.95, 0, 0.0),
            (1.0, 1.0, 16, 0.0),
            (0.7, 1.0, 0, 0.30),
            (1.3, 0.8, 50, 0.01),
        ] {
            let sp = SamplingParams {
                temperature,
                top_p,
                top_k,
                min_p,
                ..Default::default()
            };
            let p = RowParams::resolve(&sp, vocab);
            let half = (vocab / 2) as u32;
            let shards = [
                shard_of(&dense, 0, half, k, temperature, &[]),
                shard_of(&dense, half, half, k, temperature, &[]),
            ];
            let merged = merge_shards(&shards, &[], k, vocab).expect("merge");

            // The top-K by construction, against the whole row sorted.
            let mut order: Vec<usize> = (0..vocab).collect();
            order.sort_by(|a, b| {
                dense[*b]
                    .partial_cmp(&dense[*a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(b))
            });
            let want_ids: Vec<TokenId> = order[..k].iter().map(|i| *i as TokenId).collect();
            let want_values: Vec<f32> = order[..k].iter().map(|i| dense[*i]).collect();
            assert_eq!(merged.top_ids, want_ids, "T={temperature}");
            assert_eq!(merged.top_values, want_values, "T={temperature}");
            assert_eq!(merged.vocab_size, vocab);

            // And the same token as the dense pipeline, with the same noise key.
            let sampler = Sampler::new();
            let mut copy = dense.clone();
            let want = sampler
                .sample_dense(
                    &mut copy,
                    &p,
                    &PenaltyContext::default(),
                    &mut SeededNoise::new(7, 0, 0),
                )
                .expect("dense");
            let got = sampler
                .sample(
                    &merged,
                    &p,
                    &PenaltyContext::default(),
                    &mut SeededNoise::new(7, 0, 0),
                )
                .expect("merged seam");
            assert_eq!(
                got.token_id, want.token_id,
                "merged seam and dense disagree at T={temperature} top_p={top_p} top_k={top_k} \
                 min_p={min_p}"
            );
        }
    }

    /// NEGATIVE CONTROL for the test above (structural rule 8), and it differs from the case in
    /// the way the case differs from the truth: each rank's own top-K taken as if it were the
    /// row's — the exact mistake "a top-K over a shard is not the row's top-K" names. It must
    /// produce a different top-K, and with it a different token on at least one parameter set.
    #[test]
    fn one_shards_top_k_is_not_the_rows_top_k() {
        let vocab = 512usize;
        let k = 64usize;
        let dense = synthetic_row(31_337, vocab);
        let half = (vocab / 2) as u32;
        let a = shard_of(&dense, 0, half, k, 0.7, &[]);
        let shards = [a.clone(), shard_of(&dense, half, half, k, 0.7, &[])];
        let merged = merge_shards(&shards, &[], k, vocab).expect("merge");
        // Rank 0's own list, read as if the vocabulary were its shard alone.
        let alone = DeviceRow {
            top_values: a.top_values.clone(),
            top_ids: a.top_cols.iter().map(|c| a.base + c).collect(),
            logsumexp: a.logsumexp,
            logsumexp_at_temperature: a.logsumexp_at_temperature,
            penalised: Vec::new(),
            vocab_size: vocab,
        };
        assert_ne!(
            alone.top_ids, merged.top_ids,
            "one shard's top-K must not be the row's; the control cannot fail"
        );
        assert!(
            alone.logsumexp < merged.logsumexp,
            "half a row's mass must be below the whole row's"
        );
        let sp = SamplingParams {
            temperature: 0.7,
            top_p: 0.95,
            top_k: 20,
            ..Default::default()
        };
        let p = RowParams::resolve(&sp, vocab);
        // Over a sweep of noise keys, not one: at prod's own knobs a peaked row often truncates
        // to a handful of survivors that both halves share, so ONE key can agree by luck. What
        // must not happen is that the two agree on every key — that would mean the shard's own
        // top-K had been a sound seam all along.
        let sampler = Sampler::new();
        let mut differed = 0usize;
        for seed in 0..64u64 {
            let one = sampler
                .sample(
                    &alone,
                    &p,
                    &PenaltyContext::default(),
                    &mut SeededNoise::new(seed, 0, 0),
                )
                .expect("half a row");
            let both = sampler
                .sample(
                    &merged,
                    &p,
                    &PenaltyContext::default(),
                    &mut SeededNoise::new(seed, 0, 0),
                )
                .expect("the row");
            differed += usize::from(one.token_id != both.token_id);
        }
        assert!(
            differed > 0,
            "sampling from half a distribution agreed with the row on all 64 noise keys; the \
             control cannot fail"
        );
    }

    /// `log(exp(a) + exp(b))` is the log-sum-exp of the concatenation, and the merge of two
    /// shards' log-sum-exps is within one f32 ulp of the whole row's own.
    #[test]
    fn the_merged_logsumexp_is_the_rows() {
        let vocab = 4096usize;
        let dense = synthetic_row(77, vocab);
        let lse_of = |xs: &[f32]| {
            let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            max + xs.iter().map(|l| (l - max).exp()).sum::<f32>().ln()
        };
        let whole = lse_of(&dense);
        let merged = log_add_exp(lse_of(&dense[..vocab / 2]), lse_of(&dense[vocab / 2..]));
        let ulps = (whole.to_bits() as i64 - merged.to_bits() as i64).unsigned_abs();
        assert!(
            ulps <= 1,
            "merged log-sum-exp {merged} is {ulps} ulp(s) from the row's {whole}"
        );
        // The identities at the edges: an empty shard contributes nothing.
        assert_eq!(log_add_exp(f32::NEG_INFINITY, 2.5), 2.5);
        assert_eq!(log_add_exp(2.5, f32::NEG_INFINITY), 2.5);
    }

    /// A tie at the K-th value is broken by the LOWEST GLOBAL id, across the shard boundary —
    /// which is the one place the merge can disagree with the undivided kernel and does not.
    #[test]
    fn a_tie_across_the_shard_boundary_goes_to_the_lower_global_id() {
        let vocab = 8usize;
        // Both shards hold the same four values, so every rank of the merge is a tie.
        let dense = vec![1.0f32, 3.0, 2.0, 0.0, 1.0, 3.0, 2.0, 0.0];
        let a = shard_of(&dense, 0, 4, 4, 1.0, &[]);
        let b = shard_of(&dense, 4, 4, 4, 1.0, &[]);
        let merged = merge_shards(&[a, b], &[], 8, vocab).expect("merge");
        assert_eq!(merged.top_ids, vec![1, 5, 2, 6, 0, 4, 3, 7]);
        assert_eq!(
            merged.top_values,
            vec![3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0]
        );
    }

    /// The penalty gather is a SELECTION: each id is answered by the one shard that holds it,
    /// and an id no shard claimed is a refusal rather than a zero.
    #[test]
    fn the_penalty_gather_is_answered_by_the_shard_that_owns_the_id() {
        let vocab = 512usize;
        let k = 8usize;
        let dense = synthetic_row(5, vocab);
        let ids: Vec<TokenId> = vec![3, 200, 256, 500];
        let half = 256u32;
        let a = shard_of(&dense, 0, half, k, 0.7, &ids);
        let b = shard_of(&dense, half, half, k, 0.7, &ids);
        let merged = merge_shards(&[a.clone(), b.clone()], &ids, k, vocab).expect("merge");
        assert_eq!(
            merged.penalised,
            ids.iter()
                .map(|id| (*id, dense[*id as usize]))
                .collect::<Vec<_>>()
        );
        // NEGATIVE CONTROL: an id neither shard gathered.
        let mut a2 = a;
        let mut b2 = b;
        a2.penalised[1] = None;
        b2.penalised[1] = None;
        let e = merge_shards(&[a2, b2], &ids, k, vocab).expect_err("unclaimed");
        assert_eq!(e, SeamMergeError::PenaltyUnclaimed { id: 200 });
    }

    /// NEGATIVE CONTROLS on the tiling: shards that overlap, leave a gap, or do not cover the
    /// vocabulary are refused by name rather than merged into a row with holes in it.
    #[test]
    fn shards_that_do_not_tile_the_vocabulary_are_refused() {
        let dense = synthetic_row(11, 512);
        let k = 8usize;
        let a = shard_of(&dense, 0, 256, k, 0.7, &[]);
        let b = shard_of(&dense, 256, 256, k, 0.7, &[]);
        assert!(merge_shards(&[a.clone(), b.clone()], &[], k, 512).is_ok());
        // Overlap: the second shard restates the first's base.
        let mut overlap = b.clone();
        overlap.base = 128;
        assert!(matches!(
            merge_shards(&[a.clone(), overlap], &[], k, 512),
            Err(SeamMergeError::NotATiling { .. })
        ));
        // A gap.
        let mut gap = b.clone();
        gap.base = 300;
        assert!(matches!(
            merge_shards(&[a.clone(), gap], &[], k, 512),
            Err(SeamMergeError::NotATiling { .. })
        ));
        // Half the vocabulary, all by itself.
        assert!(matches!(
            merge_shards(std::slice::from_ref(&a), &[], k, 512),
            Err(SeamMergeError::WrongVocabulary { covered: 256, .. })
        ));
        // A column the shard does not hold.
        let mut wide = b.clone();
        wide.top_cols[0] = 256;
        assert!(matches!(
            merge_shards(&[a.clone(), wide], &[], k, 512),
            Err(SeamMergeError::ColumnPastShard { col: 256, .. })
        ));
        // One shard carries the tempered log-sum-exp and the other does not.
        let mut untempered = b;
        untempered.logsumexp_at_temperature = None;
        assert!(matches!(
            merge_shards(&[a, untempered], &[], k, 512),
            Err(SeamMergeError::MixedTemperedLogsumexp { at: 1 })
        ));
        assert_eq!(
            merge_shards(&[], &[], k, 512).expect_err("empty"),
            SeamMergeError::NoShards
        );
    }
}
