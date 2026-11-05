//! `prompt_logprobs`: the reference server's per-prompt-position answer, built from the sampler's seam.
//!
//! # What prod returns, exactly
//!
//! `GPUModelRunner._get_prompt_logprobs_dict` (prod image `the reference image`, the reference 0.27.1)
//! runs `compute_logits` over every prompt row but the last, takes `log_softmax` in f32
//! (`logprobs_mode = raw_logprobs`), and hands `Sampler.gather_logprobs` the row, `k` and the
//! TARGET — the prompt token at the NEXT position. That returns, per row,
//!
//! * ids `[target, topk_0 .. topk_{k-1}]` and their logprobs, in that order, and
//! * ONE rank: the target's, `(logprobs >= logprobs[target]).sum()` (`batched_count_greater_than`).
//!
//! `append_logprobs_for_next_position` then zips the ids with the ranks
//! `chain((rank,), range(1, k + 1))` into a `dict`. Two consequences a client can see:
//!
//! 1. the target is ALWAYS the first key, and
//! 2. when the target is also in the top-k, the dict comprehension writes its key twice: the key
//!    keeps its FIRST position and takes the LATER value, so the entry reads the top-k POSITION
//!    as its rank (the counted rank is thrown away) and the dict has `k` entries, not `k + 1`.
//!
//! Position 0 has no conditional distribution and is `None` (`create_prompt_logprobs` appends it
//! before any row arrives). [`position`] is that law; [`rank_from_seam`] and [`rank_from_dense`] are
//! the counted rank, from whichever of the two the engine has.
//!
//! # From the seam, not from a dense row
//!
//! The seam ([`DeviceRow`]) carries the K = 64 largest raw logits, the row's log-sum-exp at T = 1
//! and the raw logit at up to 256 named ids — the target is named there. So the target's logprob
//! and the top-k (k <= `--max-logprobs` = 20 < K) are exact from the seam. The target's RANK is
//! exact from the seam only when every id at or above the target's logit is inside the top-K; when
//! it is not (the target is below the K-th value, or tied with it) the rank needs the whole row,
//! and [`rank_from_seam`] says so by returning `None` rather than guessing.

use crate::engine::{TokenId, TokenLogprob};
use crate::sampler::DeviceRow;

/// Why one prompt row could not be turned into the reference server's entry.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum PromptLogprobError {
    /// The seam carried no logit for the target: it was not gathered and is not in the top-K.
    #[error("the seam carries no logit for prompt token {target}")]
    NoTargetLogit {
        /// The prompt token this row was asked about.
        target: TokenId,
    },
    /// The request asked for more top entries than the seam carries.
    #[error("the request asked for {requested} prompt logprobs and the seam carries {k}")]
    TooNarrow {
        /// Candidates the seam carried.
        k: usize,
        /// What the request asked for.
        requested: usize,
    },
    /// The head skipped the row (a zero temperature): nothing to report.
    #[error("the seam is empty for this prompt row")]
    Empty,
}

/// The reference server's `clamp_prompt_logprobs`: a `-inf` logprob goes on the wire as `-9999.0`.
pub const PROMPT_LOGPROB_FLOOR: f32 = -9999.0;

/// The raw logit of `target`, from the penalty gather first (it is where the engine NAMES the
/// target) and the top-K second.
#[must_use]
pub fn target_logit(row: &DeviceRow, target: TokenId) -> Option<f32> {
    row.penalised
        .iter()
        .find(|(id, _)| *id == target)
        .map(|(_, v)| *v)
        .or_else(|| {
            row.top_ids
                .iter()
                .position(|id| *id == target)
                .map(|i| row.top_values[i])
        })
}

/// The target's rank, `count(logit >= x)` over the WHOLE row, when the top-K alone decides it.
///
/// It does when the K-th value is strictly below `x`: every id at or above `x` is then inside the
/// top-K (an id outside it is at most the K-th value). It also does when the seam holds the whole
/// vocabulary. Otherwise — `x` at or below the K-th value — ids outside the seam can tie or beat
/// it, and the answer is `None`: the caller reads the row densely ([`rank_from_dense`]).
#[must_use]
pub fn rank_from_seam(row: &DeviceRow, x: f32) -> Option<u32> {
    let last = *row.top_values.last()?;
    let complete = row.top_values.len() >= row.vocab_size;
    if !complete && x <= last {
        return None;
    }
    Some(row.top_values.iter().filter(|v| **v >= x).count() as u32)
}

/// `batched_count_greater_than` over a dense row: how many logits are `>= x`. The comparison is
/// on raw logits, which is the same order as the reference server's on logprobs (one row, one subtrahend).
#[must_use]
pub fn rank_from_dense(dense: &[f32], x: f32) -> u32 {
    dense.iter().filter(|v| **v >= x).count() as u32
}

/// One prompt position, in the reference server's order and with the reference server's ranks.
///
/// `rank` is the target's counted rank ([`rank_from_seam`] / [`rank_from_dense`]); `num` is the
/// request's `prompt_logprobs` (0 is legal: the target alone).
///
/// # Errors
/// An empty seam, a seam narrower than `num`, or a seam that carries no logit for the target.
pub fn position(
    row: &DeviceRow,
    target: TokenId,
    rank: u32,
    num: usize,
) -> Result<Vec<TokenLogprob>, PromptLogprobError> {
    if row.top_ids.is_empty() {
        return Err(PromptLogprobError::Empty);
    }
    let k = row.top_ids.len().min(row.top_values.len());
    if num > k {
        return Err(PromptLogprobError::TooNarrow { k, requested: num });
    }
    let x = target_logit(row, target).ok_or(PromptLogprobError::NoTargetLogit { target })?;
    let lse = row.logsumexp;
    let mut out = Vec::with_capacity(num + 1);
    out.push(TokenLogprob {
        token_id: target,
        logprob: finite_or_floor(x - lse),
        rank: Some(rank),
    });
    for j in 0..num {
        let id = row.top_ids[j];
        let lp = finite_or_floor(row.top_values[j] - lse);
        let entry = TokenLogprob {
            token_id: id,
            logprob: lp,
            rank: Some(j as u32 + 1),
        };
        // THE DICT LAW: a key written twice keeps its first position and takes the later value.
        match out.iter_mut().find(|e| e.token_id == id) {
            Some(first) => *first = entry,
            None => out.push(entry),
        }
    }
    Ok(out)
}

fn finite_or_floor(v: f32) -> f32 {
    if v == f32::NEG_INFINITY {
        PROMPT_LOGPROB_FLOOR
    } else {
        v
    }
}

/// The reference server's `LogprobsProcessor._correct_decoded_token`, over the codec's `decode`.
///
/// A piece that ends in U+FFFD is part of a multi-byte character split across byte-fallback
/// tokens, and prod re-decodes it WITH up to four preceding prompt tokens of context (the first
/// key of each earlier position, i.e. the prompt itself), attributing the text the context tokens
/// could not show on their own to this one. Returns `""` when the bytes are still incomplete.
pub fn correct_decoded_token(
    decode: &dyn Fn(&[TokenId]) -> String,
    token: TokenId,
    context: &[TokenId],
) -> String {
    let max_ctx = context.len().min(4);
    for num_ctx in 1..=max_ctx {
        let ctx = &context[context.len() - num_ctx..];
        let mut ids = ctx.to_vec();
        ids.push(token);
        let full = decode(&ids);
        if full.ends_with('\u{FFFD}') {
            continue;
        }
        let mut clean_end = ctx.len();
        for j in (0..ctx.len()).rev() {
            if decode(&ctx[j..=j]).ends_with('\u{FFFD}') {
                clean_end = j;
            } else {
                break;
            }
        }
        let clean_prefix = if clean_end > 0 {
            decode(&ctx[..clean_end])
        } else {
            String::new()
        };
        if let Some(rest) = full.strip_prefix(clean_prefix.as_str()) {
            return rest.to_owned();
        }
        // Tokenizer normalisation moved the prefix: cut at the longest common CHARACTER prefix.
        let common: usize = clean_prefix
            .chars()
            .zip(full.chars())
            .take_while(|(a, b)| a == b)
            .map(|(a, _)| a.len_utf8())
            .sum();
        return full[common..].to_owned();
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(
        values: &[f32],
        ids: &[TokenId],
        lse: f32,
        pen: &[(TokenId, f32)],
        vocab: usize,
    ) -> DeviceRow {
        DeviceRow {
            top_values: values.to_vec(),
            top_ids: ids.to_vec(),
            logsumexp: lse,
            logsumexp_at_temperature: Some(lse),
            penalised: pen.to_vec(),
            vocab_size: vocab,
        }
    }

    #[test]
    fn a_target_outside_the_top_k_comes_first_with_its_counted_rank() {
        let r = row(&[5.0, 4.0, 3.0], &[7, 8, 9], 6.0, &[(42, 1.0)], 1000);
        let p = position(&r, 42, 17, 2).unwrap();
        let ids: Vec<TokenId> = p.iter().map(|e| e.token_id).collect();
        assert_eq!(ids, vec![42, 7, 8]);
        assert_eq!(p[0].rank, Some(17));
        assert!((p[0].logprob - (1.0 - 6.0)).abs() < 1e-6);
        assert_eq!(p[1].rank, Some(1));
        assert_eq!(p[2].rank, Some(2));
    }

    #[test]
    fn a_target_inside_the_top_k_keeps_first_place_and_takes_its_positional_rank() {
        // The reference server writes the target twice; the dict keeps the first position and the later value.
        let r = row(&[5.0, 5.0, 3.0], &[7, 8, 9], 6.0, &[(8, 5.0)], 1000);
        let p = position(&r, 8, 2, 3).unwrap();
        let ids: Vec<TokenId> = p.iter().map(|e| e.token_id).collect();
        assert_eq!(ids, vec![8, 7, 9]);
        assert_eq!(p[0].rank, Some(2));
        assert_eq!(p.len(), 3, "k entries, not k + 1");
    }

    #[test]
    fn zero_asks_for_the_target_alone() {
        let r = row(&[5.0], &[7], 6.0, &[(7, 5.0)], 1000);
        let p = position(&r, 7, 1, 0).unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].rank, Some(1));
    }

    #[test]
    fn the_seam_decides_the_rank_only_above_its_floor() {
        let r = row(&[5.0, 4.0, 3.0], &[7, 8, 9], 6.0, &[], 1000);
        assert_eq!(rank_from_seam(&r, 4.0), Some(2));
        assert_eq!(rank_from_seam(&r, 3.5), Some(2));
        // AT the floor: ids outside the seam can tie it, so the seam cannot count.
        assert_eq!(rank_from_seam(&r, 3.0), None);
        assert_eq!(rank_from_seam(&r, 1.0), None);
        // A seam that holds the whole vocabulary always can.
        let whole = row(&[5.0, 4.0, 3.0], &[7, 8, 9], 6.0, &[], 3);
        assert_eq!(rank_from_seam(&whole, 1.0), Some(3));
        assert_eq!(rank_from_dense(&[5.0, 4.0, 3.0, 3.0, 1.0], 3.0), 4);
    }

    #[test]
    fn a_target_with_no_logit_is_refused_not_guessed() {
        let r = row(&[5.0], &[7], 6.0, &[], 1000);
        assert_eq!(
            position(&r, 42, 1, 1),
            Err(PromptLogprobError::NoTargetLogit { target: 42 })
        );
        assert!(matches!(
            position(&r, 7, 1, 2),
            Err(PromptLogprobError::TooNarrow { k: 1, requested: 2 })
        ));
    }

    /// A dense row with distinct values, and the seam a device would write over `[base, base+cols)`
    /// of it: K largest in the argmax's order (ids LOCAL), the shard's log-sum-exp, and the
    /// target's logit where the shard holds it.
    fn dense_row(vocab: usize, seed: u64) -> Vec<f32> {
        (0..vocab as u64)
            .map(|v| {
                let h = (v * 2_654_435_761 + seed * 40_503) % (1 << 32);
                ((h >> 12) as f64 / 65_536.0 - 8.0) as f32
            })
            .collect()
    }

    fn shard_of(
        dense: &[f32],
        base: u32,
        cols: u32,
        k: usize,
        target: TokenId,
    ) -> crate::sampler::SeamShard {
        let slice = &dense[base as usize..(base + cols) as usize];
        let mut order: Vec<u32> = (0..cols).collect();
        order.sort_by(|a, b| {
            slice[*b as usize]
                .partial_cmp(&slice[*a as usize])
                .unwrap()
                .then(a.cmp(b))
        });
        order.truncate(k);
        let max = f64::from(slice.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        let lse = (max
            + slice
                .iter()
                .map(|x| (f64::from(*x) - max).exp())
                .sum::<f64>()
                .ln()) as f32;
        crate::sampler::SeamShard {
            base,
            cols,
            top_values: order.iter().map(|c| slice[*c as usize]).collect(),
            top_cols: order,
            logsumexp: lse,
            logsumexp_at_temperature: Some(lse),
            penalised: vec![
                (target >= base && target < base + cols).then(|| dense[target as usize])
            ],
        }
    }

    /// THE TP = 2 SEAM MERGE, for a prompt row: two half-vocabulary shards, merged by the
    /// sampler's own law, give the same the reference server entry as the undivided row — for a target in either
    /// shard, inside and outside the top-k. The negative control hands the merge the shards with
    /// the target's logit claimed by NEITHER rank, and it must refuse rather than guess.
    #[test]
    fn two_vocabulary_shards_merge_into_the_undivided_rows_answer() {
        let (vocab, k, num) = (1000usize, 8usize, 5usize);
        for (seed, target) in [(1u64, 3u32), (2, 777), (3, 499), (4, 500)] {
            let mut dense = dense_row(vocab, seed);
            // Seed 1 and 2: the target is a spike, inside the top-k; 3 and 4: background.
            if seed <= 2 {
                dense[target as usize] = 20.0;
            }
            let whole = shard_of(&dense, 0, vocab as u32, k, target);
            let undivided = DeviceRow {
                top_values: whole.top_values.clone(),
                top_ids: whole.top_cols.clone(),
                logsumexp: whole.logsumexp,
                logsumexp_at_temperature: whole.logsumexp_at_temperature,
                penalised: vec![(target, dense[target as usize])],
                vocab_size: vocab,
            };
            let half = (vocab / 2) as u32;
            let shards = [
                shard_of(&dense, 0, half, k, target),
                shard_of(&dense, half, half, k, target),
            ];
            let merged = crate::sampler::merge_shards(&shards, &[target], k, vocab).unwrap();
            let x = target_logit(&merged, target).unwrap();
            let rank = rank_from_seam(&merged, x).unwrap_or_else(|| rank_from_dense(&dense, x));
            assert_eq!(
                rank,
                rank_from_dense(&dense, x),
                "seed {seed}: the counted rank"
            );
            let a = position(&merged, target, rank, num).unwrap();
            let b = position(&undivided, target, rank, num).unwrap();
            assert_eq!(a.len(), b.len());
            for (p, q) in a.iter().zip(&b) {
                assert_eq!((p.token_id, p.rank), (q.token_id, q.rank), "seed {seed}");
                assert!((p.logprob - q.logprob).abs() < 1e-5, "seed {seed}");
            }
        }
        // NEGATIVE CONTROL: neither shard gathered the target.
        let dense = dense_row(vocab, 9);
        let mut shards = [
            shard_of(&dense, 0, 500, k, 700),
            shard_of(&dense, 500, 500, k, 700),
        ];
        shards[1].penalised = vec![None];
        assert!(crate::sampler::merge_shards(&shards, &[700], k, vocab).is_err());
    }

    #[test]
    fn a_split_character_is_attributed_to_the_token_that_completes_it() {
        // Two byte-fallback tokens make "é" (0xC3 0xA9); decoding either alone is U+FFFD.
        let decode = |ids: &[TokenId]| -> String {
            let mut bytes = Vec::new();
            for id in ids {
                match id {
                    1 => bytes.extend_from_slice(b"ab"),
                    2 => bytes.push(0xC3),
                    3 => bytes.push(0xA9),
                    _ => bytes.push(b'?'),
                }
            }
            String::from_utf8_lossy(&bytes).into_owned()
        };
        assert_eq!(correct_decoded_token(&decode, 3, &[1, 2]), "é");
        assert_eq!(correct_decoded_token(&decode, 2, &[1]), "");
    }
}
