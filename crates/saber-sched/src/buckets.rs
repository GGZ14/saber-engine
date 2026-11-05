//! Bucket selection: which captured shape this step replays.
//!
//! The rule is one sentence — **the smallest captured bucket that fits the running set** — and both
//! halves of it are load-bearing:
//!
//! * *Captured.* An uncaptured shape is not a shape the engine may replay; dropping capture sizes
//!   1/2/4 alone cost **8% of single-stream decode** (`ARCHITECTURE.md` § 4. Execution plane), so
//!   the small buckets exist precisely to be selected. Eager buckets never reach here: the config
//!   drops them at construction.
//! * *Fits.* The batch is **rounded up** to the bucket's batch and the extra slots are padding the
//!   step runs and discards, because a captured graph has a fixed M. Never rounded down, which
//!   would drop a sequence from the step.
//!
//! "Smallest" is the lexicographic order `(batch, spec_width, ctx_band)`. Batch first because an
//! unused slot costs a full sequence's worth of GEMM rows *and* a full GDN state slot against the
//! 64 MB Infinity Cache (the 09-14 cliff); width second because an unused draft token costs a state
//! slot too; the context band last because it only widens attention's iteration bound.

use saber_plan::{Bucket, Phase};

/// What a **prefill** step needs a bucket to cover: a chunk of `tokens` rows for one sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillRequest {
    /// Rows the chunk runs at — the chunk size the chunking rule picked, which is already one of
    /// the captured sizes.
    pub tokens: u32,
    /// Prompt tokens resident after this chunk, which the context band has to cover.
    pub ctx_tokens: u32,
}

/// The smallest captured prefill bucket that covers `req`.
///
/// Prefill selection is a different question from decode selection and gets a different function:
/// there is no batch and no speculative width to round up, only a chunk size and a band. Mixing
/// the two in one call is what [`NoBucket::MixedPhase`] exists to refuse.
pub fn select_prefill(buckets: &[Bucket], req: PrefillRequest) -> Result<Bucket, NoBucket> {
    let captured = || {
        buckets
            .iter()
            .filter(|b| b.captured && b.phase == Phase::Prefill)
    };
    if let Some(b) = captured()
        .filter(|b| b.tokens >= req.tokens && b.ctx_band >= req.ctx_tokens)
        .min_by_key(|b| (b.tokens, b.ctx_band))
        .copied()
    {
        return Ok(b);
    }
    let max_tokens = captured().map(|b| b.tokens).max().unwrap_or(0);
    let max_band = captured().map(|b| b.ctx_band).max().unwrap_or(0);
    if req.tokens > max_tokens {
        Err(NoBucket::PrefillTokens {
            tokens: req.tokens,
            max_tokens,
        })
    } else {
        Err(NoBucket::Context {
            ctx_tokens: req.ctx_tokens,
            max_band,
        })
    }
}

/// What a step needs a bucket to cover.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BucketRequest {
    /// Live sequences in the step.
    pub running: u32,
    /// The verify width the width policy asked for.
    pub width: u32,
    /// Longest context any live sequence will hold **after** this step.
    pub ctx_tokens: u32,
}

/// Why no captured bucket covers a step. Each variant names the dimension that did not fit and the
/// widest captured value in it, because that is the number the manifest has to change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum NoBucket {
    /// More sequences than the widest captured batch.
    #[error("no captured bucket with batch >= {running} (widest captured batch is {max_batch})")]
    Batch {
        /// Sequences in the step.
        running: u32,
        /// The widest captured batch.
        max_batch: u32,
    },
    /// A wider verify width than any captured bucket has.
    #[error(
        "no captured bucket with spec_width >= {width} (widest captured width is {max_width})"
    )]
    Width {
        /// Width asked for.
        width: u32,
        /// The widest captured width.
        max_width: u32,
    },
    /// A longer context than any captured band.
    #[error("no captured bucket with ctx_band >= {ctx_tokens} (widest band is {max_band})")]
    Context {
        /// Context needed.
        ctx_tokens: u32,
        /// The widest captured band.
        max_band: u32,
    },
    /// A chunk wider than any captured prefill bucket.
    #[error("no captured prefill bucket with tokens >= {tokens} (widest captured chunk is {max_tokens})")]
    PrefillTokens {
        /// Rows the chunk needs.
        tokens: u32,
        /// The widest captured chunk.
        max_tokens: u32,
    },
    /// A batch that mixed prefill chunks with decode rows.
    ///
    /// Not a thing the scheduler builds — `prepare` runs one phase per step — and not a thing the
    /// plan can express: the two phases run different ops and budget GDN state differently, so a
    /// mixed batch has no captured shape. The variant exists so that a future caller that tries
    /// gets told what it did rather than an empty candidate list.
    #[error(
        "a step cannot mix {prefill} prefill chunk(s) with {decode} decode row(s): the two \
             phases are different captured shapes"
    )]
    MixedPhase {
        /// Prefill chunks asked for.
        prefill: u32,
        /// Decode rows asked for.
        decode: u32,
    },
    /// The bucket set has a bucket in each dimension but none that covers all three at once.
    #[error(
        "no captured bucket covers batch {running} x width {width} x ctx {ctx_tokens} together"
    )]
    Combination {
        /// Sequences in the step.
        running: u32,
        /// Width asked for.
        width: u32,
        /// Context needed.
        ctx_tokens: u32,
    },
    /// A RAGGED step whose rows no captured band covers (C2). The bands are clipped to `batch x
    /// max width` and that product is always one of them ([`band_rows_for`]), so this names a
    /// plan built without its top band, or a step wider than its ceiling allows.
    #[error(
        "no captured row band covers {rows} row(s) of {running} sequence(s) at ctx {ctx_tokens} \
         (widest band is {max_rows} rows)"
    )]
    Rows {
        /// Rows the step feeds.
        rows: u32,
        /// Sequences in the step.
        running: u32,
        /// Context needed.
        ctx_tokens: u32,
        /// The widest captured band.
        max_rows: u32,
    },
}

/// **The default decode ROW BANDS (workstream C2)**, in rows: data, not a ladder (`CLAUDE.md`
/// rule 6) — every multiple of 8 from 16 to 64, which at batch 8 x max width 8 covers every step
/// `Σ w_i` can be to within 7 padding rows. `--engine-row-bands` replaces it.
pub const DEFAULT_ROW_BANDS: &[u32] = &[16, 24, 32, 40, 48, 56, 64];

/// The bands a plan of `batch` sequences at `max_width` captures, from the `asked` list: each
/// clipped to `batch x max_width` (a band wider than that has rows no step can feed and a GDN /
/// attention layout that cannot hold them), de-duplicated, ascending — and ALWAYS including
/// `batch x max_width` itself, so that every step the scheduler can form has a band (a step's
/// rows are at most `running x ceiling <= batch x max_width`).
#[must_use]
pub fn band_rows_for(asked: &[u32], batch: u32, max_width: u32) -> Vec<u32> {
    let top = batch.max(1) * max_width.max(1);
    let mut out: Vec<u32> = asked
        .iter()
        .copied()
        .filter(|r| *r > 0)
        .map(|r| r.min(top))
        .collect();
    out.push(top);
    out.sort_unstable();
    out.dedup();
    out
}

/// What a RAGGED decode step needs (C2): its live rows `Σ w_i`, how many sequences, and the
/// context its widest slot reaches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BandRequest {
    /// Rows the running slots feed this step, `Σ w_i`.
    pub rows: u32,
    /// Running sequences.
    pub running: u32,
    /// Largest context a slot reaches after this step.
    pub ctx_tokens: u32,
}

/// **The smallest captured row band that holds a ragged step** (C2): rows `>= Σ w_i`, batch `>=`
/// the running count, context band `>=` the step's. Rectangles are never returned, and
/// [`select`] never returns a band.
///
/// # Errors
/// [`NoBucket::Rows`] when no band is wide enough, [`NoBucket::Batch`] / [`NoBucket::Context`]
/// when the other two facts are what fails.
pub fn select_band(buckets: &[Bucket], req: BandRequest) -> Result<Bucket, NoBucket> {
    let bands = || {
        buckets
            .iter()
            .filter(|b| b.captured && b.phase == Phase::Decode && b.is_band())
    };
    if let Some(b) = bands()
        .filter(|b| {
            b.tokens >= req.rows && b.batch >= req.running.max(1) && b.ctx_band >= req.ctx_tokens
        })
        .min_by_key(|b| (b.tokens, b.batch, b.ctx_band))
        .copied()
    {
        return Ok(b);
    }
    let max_batch = bands().map(|b| b.batch).max().unwrap_or(0);
    let max_rows = bands().map(|b| b.tokens).max().unwrap_or(0);
    let max_band = bands().map(|b| b.ctx_band).max().unwrap_or(0);
    if req.running > max_batch {
        Err(NoBucket::Batch {
            running: req.running,
            max_batch,
        })
    } else if req.ctx_tokens > max_band {
        Err(NoBucket::Context {
            ctx_tokens: req.ctx_tokens,
            max_band,
        })
    } else {
        Err(NoBucket::Rows {
            rows: req.rows,
            running: req.running,
            ctx_tokens: req.ctx_tokens,
            max_rows,
        })
    }
}

/// The smallest captured bucket that covers `req`.
///
/// `buckets` is expected to be the config's captured set; a non-captured bucket is refused here as
/// well, so this function is safe to call with any slice.
pub fn select(buckets: &[Bucket], req: BucketRequest) -> Result<Bucket, NoBucket> {
    // Rectangles only: a row band (C2) is a different shape with the same (batch, width) and is
    // chosen by `select_band`, never here.
    let captured = || {
        buckets
            .iter()
            .filter(|b| b.captured && b.phase == Phase::Decode && !b.is_band())
    };

    let best = captured()
        .filter(|b| {
            b.batch >= req.running.max(1)
                && b.spec_width >= req.width
                && b.ctx_band >= req.ctx_tokens
        })
        .min_by_key(|b| (b.batch, b.spec_width, b.ctx_band))
        .copied();
    if let Some(b) = best {
        return Ok(b);
    }

    // Name the dimension that did not fit. Fail loud: a "no bucket" that does not say which
    // manifest number is too small is a bug report nobody can act on.
    let max_batch = captured().map(|b| b.batch).max().unwrap_or(0);
    let max_width = captured().map(|b| b.spec_width).max().unwrap_or(0);
    let max_band = captured().map(|b| b.ctx_band).max().unwrap_or(0);
    if req.running > max_batch {
        Err(NoBucket::Batch {
            running: req.running,
            max_batch,
        })
    } else if req.width > max_width {
        Err(NoBucket::Width {
            width: req.width,
            max_width,
        })
    } else if req.ctx_tokens > max_band {
        Err(NoBucket::Context {
            ctx_tokens: req.ctx_tokens,
            max_band,
        })
    } else {
        Err(NoBucket::Combination {
            running: req.running,
            width: req.width,
            ctx_tokens: req.ctx_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(tokens: u32, ctx_band: u32) -> Bucket {
        Bucket::prefill(tokens, ctx_band, true)
    }

    fn b(batch: u32, spec_width: u32, ctx_band: u32, captured: bool) -> Bucket {
        Bucket::decode(batch, spec_width, ctx_band, captured)
    }

    fn ladder() -> Vec<Bucket> {
        vec![
            b(1, 1, 8192, true),
            b(1, 5, 8192, true),
            b(1, 5, 65536, true),
            b(4, 1, 8192, true),
            b(4, 5, 8192, true),
            b(8, 5, 8192, true),
            b(8, 5, 65536, true),
        ]
    }

    /// **C2 host test (c): band choice.** `Σ w_i` rounds up to the smallest captured band; a band
    /// list is clipped to `batch x max width` and always carries that product; a rectangle is never
    /// a band's answer nor a band a rectangle's; past the widest band is a refusal that names it.
    #[test]
    fn a_ragged_step_takes_the_smallest_band_that_holds_its_rows() {
        assert_eq!(
            band_rows_for(DEFAULT_ROW_BANDS, 8, 8),
            vec![16, 24, 32, 40, 48, 56, 64]
        );
        // Clipped: at batch 8 x width 5 the top is 40, and the wider asks collapse into it.
        assert_eq!(band_rows_for(DEFAULT_ROW_BANDS, 8, 5), vec![16, 24, 32, 40]);
        // The top is always there, whatever was asked.
        assert_eq!(band_rows_for(&[16, 24], 8, 8), vec![16, 24, 64]);
        assert_eq!(band_rows_for(&[], 4, 8), vec![32]);
        let mut set: Vec<Bucket> = band_rows_for(DEFAULT_ROW_BANDS, 8, 8)
            .into_iter()
            .map(|r| Bucket::band(8, 8, r, 8192, true))
            .collect();
        set.push(b(8, 8, 8192, true));
        set.push(b(8, 5, 8192, true));
        let pick = |rows: u32, running: u32| {
            select_band(
                &set,
                BandRequest {
                    rows,
                    running,
                    ctx_tokens: 100,
                },
            )
        };
        for (rows, want) in [
            (1, 16),
            (9, 16),
            (16, 16),
            (17, 24),
            (33, 40),
            (40, 40),
            (41, 48),
            (64, 64),
        ] {
            let got = pick(rows, 8).expect("a band");
            assert!(got.is_band());
            assert_eq!(got.tokens, want, "{rows} rows");
            assert_eq!(got.m(), want);
            assert_eq!(got.spec_width, 8);
            assert_eq!(
                got.gdn_state_slots(),
                64,
                "the arena stays batch x max width"
            );
            assert_eq!(got.label(), format!("b8r{want}c8192"));
        }
        assert_eq!(
            pick(65, 8),
            Err(NoBucket::Rows {
                rows: 65,
                running: 8,
                ctx_tokens: 100,
                max_rows: 64
            })
        );
        assert!(matches!(pick(16, 9), Err(NoBucket::Batch { .. })));
        // NEGATIVE CONTROL: the rectangle selector never answers with a band, even where a band
        // has the same (batch, width) and the rectangle is missing.
        let only_bands: Vec<Bucket> = set.iter().copied().filter(Bucket::is_band).collect();
        assert!(select(
            &only_bands,
            BucketRequest {
                running: 8,
                width: 8,
                ctx_tokens: 100
            }
        )
        .is_err());
        assert_eq!(
            select(
                &set,
                BucketRequest {
                    running: 8,
                    width: 8,
                    ctx_tokens: 100
                }
            ),
            Ok(b(8, 8, 8192, true))
        );
    }

    #[test]
    fn smallest_fitting_bucket() {
        let set = ladder();
        assert_eq!(
            select(
                &set,
                BucketRequest {
                    running: 1,
                    width: 1,
                    ctx_tokens: 100
                }
            )
            .unwrap(),
            b(1, 1, 8192, true)
        );
        assert_eq!(
            select(
                &set,
                BucketRequest {
                    running: 1,
                    width: 2,
                    ctx_tokens: 100
                }
            )
            .unwrap(),
            b(1, 5, 8192, true),
            "width rounds up to the next captured width"
        );
        assert_eq!(
            select(
                &set,
                BucketRequest {
                    running: 3,
                    width: 1,
                    ctx_tokens: 100
                }
            )
            .unwrap(),
            b(4, 1, 8192, true),
            "batch rounds up, the fourth slot is padding"
        );
        assert_eq!(
            select(
                &set,
                BucketRequest {
                    running: 1,
                    width: 5,
                    ctx_tokens: 40000
                }
            )
            .unwrap(),
            b(1, 5, 65536, true),
            "the narrow band does not cover 40k"
        );
    }

    #[test]
    fn prefill_selection_takes_the_smallest_fitting_chunk() {
        let set = vec![p(2048, 8192), p(8192, 8192), p(8192, 65536)];
        assert_eq!(
            select_prefill(
                &set,
                PrefillRequest {
                    tokens: 2048,
                    ctx_tokens: 234
                }
            )
            .unwrap(),
            p(2048, 8192)
        );
        assert_eq!(
            select_prefill(
                &set,
                PrefillRequest {
                    tokens: 8192,
                    ctx_tokens: 40000
                }
            )
            .unwrap(),
            p(8192, 65536),
            "the band has to cover the prompt so far, not just the chunk"
        );
        assert_eq!(
            select_prefill(
                &set,
                PrefillRequest {
                    tokens: 16384,
                    ctx_tokens: 16384
                }
            ),
            Err(NoBucket::PrefillTokens {
                tokens: 16384,
                max_tokens: 8192
            })
        );
        assert_eq!(
            select_prefill(
                &set,
                PrefillRequest {
                    tokens: 2048,
                    ctx_tokens: 100_000
                }
            ),
            Err(NoBucket::Context {
                ctx_tokens: 100_000,
                max_band: 65536
            })
        );
    }

    #[test]
    fn the_two_phases_never_answer_each_other_s_question() {
        // A prefill bucket sorts ahead of every decode bucket in the decode ladder's own order
        // (batch 1, spec_width 0), so a selection that did not filter by phase would hand a
        // decode step a prompt shape. Neither function sees the other's buckets.
        let mixed = vec![p(2048, 8192), b(1, 1, 8192, true), b(8, 5, 8192, true)];
        assert_eq!(
            select(
                &mixed,
                BucketRequest {
                    running: 1,
                    width: 1,
                    ctx_tokens: 10
                }
            )
            .unwrap(),
            b(1, 1, 8192, true)
        );
        assert_eq!(
            select_prefill(
                &mixed,
                PrefillRequest {
                    tokens: 2048,
                    ctx_tokens: 10
                }
            )
            .unwrap(),
            p(2048, 8192)
        );
        // And the refusal for a batch that tried to be both says so by name.
        let mixed_err = NoBucket::MixedPhase {
            prefill: 1,
            decode: 3,
        };
        assert!(mixed_err.to_string().contains("cannot mix"), "{mixed_err}");
    }

    #[test]
    fn an_eager_bucket_is_never_selected() {
        let mut set = ladder();
        // A bucket that would win on every axis if capture were not required.
        set.push(b(1, 1, 1024, false));
        let chosen = select(
            &set,
            BucketRequest {
                running: 1,
                width: 1,
                ctx_tokens: 100,
            },
        )
        .unwrap();
        assert!(chosen.captured);
        assert_eq!(chosen, b(1, 1, 8192, true));
    }

    #[test]
    fn refusals_name_the_dimension() {
        let set = ladder();
        assert_eq!(
            select(
                &set,
                BucketRequest {
                    running: 9,
                    width: 1,
                    ctx_tokens: 10
                }
            ),
            Err(NoBucket::Batch {
                running: 9,
                max_batch: 8
            })
        );
        assert_eq!(
            select(
                &set,
                BucketRequest {
                    running: 1,
                    width: 7,
                    ctx_tokens: 10
                }
            ),
            Err(NoBucket::Width {
                width: 7,
                max_width: 5
            })
        );
        assert_eq!(
            select(
                &set,
                BucketRequest {
                    running: 1,
                    width: 1,
                    ctx_tokens: 100_000
                }
            ),
            Err(NoBucket::Context {
                ctx_tokens: 100_000,
                max_band: 65536
            })
        );
        // Each dimension has a bucket that covers it, but no single bucket covers all three.
        assert_eq!(
            select(
                &set,
                BucketRequest {
                    running: 8,
                    width: 1,
                    ctx_tokens: 40000
                }
            )
            .unwrap(),
            b(8, 5, 65536, true),
            "rounding up in width is how the combination is covered"
        );
        assert_eq!(
            select(
                &[b(1, 5, 8192, true), b(8, 1, 8192, true)],
                BucketRequest {
                    running: 8,
                    width: 5,
                    ctx_tokens: 10
                }
            ),
            Err(NoBucket::Combination {
                running: 8,
                width: 5,
                ctx_tokens: 10
            })
        );
    }
}
