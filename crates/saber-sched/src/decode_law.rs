//! **The decode batch law (workstream B1)**: which captured decode batch a step of `live`
//! sequences replays, how many of its slots are padding, and what the plan's decode buckets cost.
//!
//! A decode step replays ONE captured `(batch, width)` graph, and a captured graph has a fixed M,
//! so a step of `live` sequences runs at the smallest captured batch `>= live`
//! ([`crate::buckets::select`], batch first) and the spare slots are PADDING the step runs and
//! discards ([`crate::executor::StepBatch::padded_slots`]). With the batches `1,2,4,8`
//! ([`PRE_B3_DECODE_BATCHES`], the default until CMP27AF added b3) a live count of 3 runs as 4 and 5 / 6 / 7 run as 8: at Q4's conc-8
//! census that was 1,655 padded rows of 6,424 sequence-steps' worth. prod runs its drafter at the
//! live count (`disable_padded_drafter_batch: true`); Saber's equivalent is to CAPTURE every batch
//! the scheduler can form — [`DECODE_BATCH_LADDER`], `1..=8` — so that the pick is the live count
//! itself and `padded_slots` reads 0 whenever the live count is a bucket.
//!
//! The ladder is data (`CLAUDE.md` rule 6): `--engine-decode-batches` names it, the scheduler's
//! pick is the unchanged `select`, and every new `(batch, width)` pair is planned, billed
//! (`engine.arena.b{b}s{w}`, `engine.arena.head.r{rows}`) and captured by the same loops the
//! power-of-two set always went through. What it costs is stated at boot by [`DecodeBucketLaw`].
//!
//! # What the ladder does NOT change
//!
//! A row's bytes depend on the step's M only through the decode split band and the all-reduce's
//! exact / lossy size rule (`tests/batch_ladder.rs` in `saber-engine` is the census). Against the
//! bucket the power-of-two set PADS each live count into, every new bucket keeps the class
//! except one: **b3s8** (24 rows) sums `gdn.out_proj`, `attn.o_proj` and `mlp.down_proj` at
//! split 4 (the `m_max = 24` fill row) where b4s8 sums them at split 2 — the class b1s8 and b2s8
//! already use. So a 3-sequence width-8 step moves to the solo side's split class; nothing else
//! moves.

use saber_plan::Bucket;

use crate::buckets::{select, BucketRequest, NoBucket};

/// The decode batches every launcher plans by default: `1,2,3,4,8` since CMP27AF (2026-10-29) --
/// the power-of-two set plus a b3 bucket, so conc 3 runs its own 24-row verify instead of padding
/// into b4's 32 rows (served conc-3 step -1.5 %, temp-0 bytes identical; +5 MiB of decode arena
/// a rank). [`PRE_B3_DECODE_BATCHES`] is the control and the rollback.
pub const DEFAULT_DECODE_BATCHES: &[u32] = &[1, 2, 3, 4, 8];

/// The same, as the text a `--engine-decode-batches` flag takes.
pub const DEFAULT_DECODE_BATCHES_ARG: &str = "1,2,3,4,8";

/// **The control** (the pre-CMP27AF default and B1's): `1,2,4,8`, a live count of 3 pads into b4.
pub const PRE_B3_DECODE_BATCHES: &[u32] = &[1, 2, 4, 8];

/// The same, as flag text.
pub const PRE_B3_DECODE_BATCHES_ARG: &str = "1,2,4,8";

/// **The batch ladder (B1)**: every batch the scheduler can form at `max_seqs 8`.
pub const DECODE_BATCH_LADDER: &[u32] = &[1, 2, 3, 4, 5, 6, 7, 8];

/// The same, as flag text.
pub const DECODE_BATCH_LADDER_ARG: &str = "1,2,3,4,5,6,7,8";

/// One live count's pick: the captured bucket and its padding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchPick {
    /// Sequences in the step.
    pub live: u32,
    /// The captured batch it replays.
    pub batch: u32,
    /// `batch - live`: the slots the step runs and discards.
    pub padded_slots: u32,
}

/// The scheduler's pick for every live count `1..=max_live` at verify width `width`, through the
/// scheduler's own [`select`] (so this is the law the step loop runs, not a copy of it).
///
/// # Errors
/// The first live count no captured bucket holds.
pub fn pick_table(
    buckets: &[Bucket],
    max_live: u32,
    width: u32,
    ctx_tokens: u32,
) -> Result<Vec<BatchPick>, NoBucket> {
    (1..=max_live)
        .map(|live| {
            let b = select(
                buckets,
                BucketRequest {
                    running: live,
                    width,
                    ctx_tokens,
                },
            )?;
            Ok(BatchPick {
                live,
                batch: b.batch,
                padded_slots: b.batch - live,
            })
        })
        .collect()
}

/// The smallest batch in `batches` holding `live` sequences — [`select`]'s batch axis over a
/// bare batch list (a boot line has the list, not the buckets). `None` past the widest.
#[must_use]
pub fn pick_batch(batches: &[u32], live: u32) -> Option<u32> {
    batches.iter().copied().filter(|b| *b >= live.max(1)).min()
}

/// **What the decode bucket set costs, and what it picks** — the boot line both ranks print
/// beside the ledger's (`decodebuckets rank N: …`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeBucketLaw {
    /// The planned batches, ascending.
    pub batches: Vec<u32>,
    /// The planned verify widths, ascending.
    pub widths: Vec<u32>,
    /// Body arena bytes billed per `(batch, width)` (`engine.arena.b{b}s{w}`).
    pub body: Vec<(u32, u32, u64)>,
    /// Head arena bytes billed per decode ROW count (`engine.arena.head.r{rows}`; one head per
    /// row count, so `b1s5` and `b5s1` share one).
    pub heads: Vec<(u32, u64)>,
    /// H46: bytes of the ONE shared decode-head arena ([`SharedHeadArena`]), 0 under the
    /// separate-arena law (`heads` then carries each head's arena; under the shared law it
    /// carries only each head's own `sel` / pick / seam buffers).
    pub shared_heads: u64,
}

impl DecodeBucketLaw {
    /// H46: the shared decode-head arena's bytes, billed once beside the per-row lines.
    #[must_use]
    pub fn with_shared_heads(mut self, bytes: u64) -> Self {
        self.shared_heads = bytes;
        self
    }

    /// The law over the planned batches and widths and what the ledger billed for them.
    #[must_use]
    pub fn of(
        batches: &[u32],
        widths: &[u32],
        body: Vec<(u32, u32, u64)>,
        heads: Vec<(u32, u64)>,
    ) -> Self {
        let mut b = batches.to_vec();
        b.sort_unstable();
        b.dedup();
        let mut w = widths.to_vec();
        w.sort_unstable();
        w.dedup();
        Self {
            batches: b,
            widths: w,
            body,
            heads,
            shared_heads: 0,
        }
    }

    /// Bytes of every decode body arena, summed (each shape has its own; none is shared).
    #[must_use]
    pub fn body_bytes(&self) -> u64 {
        self.body.iter().map(|(_, _, b)| *b).sum()
    }

    /// Bytes billed for the decode heads: every per-row line plus the shared arena (H46), if any.
    #[must_use]
    pub fn head_bytes(&self) -> u64 {
        self.heads.iter().map(|(_, b)| *b).sum::<u64>() + self.shared_heads
    }

    /// The pick for every live count up to the widest batch: `(live, batch)`.
    #[must_use]
    pub fn picks(&self) -> Vec<(u32, u32)> {
        let top = self.batches.last().copied().unwrap_or(1);
        (1..=top)
            .filter_map(|live| pick_batch(&self.batches, live).map(|b| (live, b)))
            .collect()
    }

    /// Padded slots summed over the live counts `1..=top` (0 exactly when every count is a bucket).
    #[must_use]
    pub fn padded_over_live_counts(&self) -> u32 {
        self.picks().iter().map(|(l, b)| b - l).sum()
    }

    /// The boot line.
    #[must_use]
    pub fn line(&self, rank: u32) -> String {
        let mib = |b: u64| b as f64 / f64::from(1u32 << 20);
        let picks: Vec<String> = self
            .picks()
            .iter()
            .map(|(l, b)| {
                if l == b {
                    format!("{l}")
                } else {
                    format!("{l}->{b}")
                }
            })
            .collect();
        format!(
            "decodebuckets rank {rank}: batches {:?} x widths {:?} = {} shape(s); pick {} \
             (padded slots over live 1..{}: {}); arenas body {:.1} MiB + heads {:.1} MiB over {} \
             row count(s){} = {:.1} MiB",
            self.batches,
            self.widths,
            self.body.len(),
            picks.join(" "),
            self.batches.last().copied().unwrap_or(1),
            self.padded_over_live_counts(),
            mib(self.body_bytes()),
            mib(self.head_bytes()),
            self.heads.len(),
            if self.shared_heads > 0 {
                format!(" (SHARED head arena {:.1} MiB)", mib(self.shared_heads))
            } else {
                String::new()
            },
            mib(self.body_bytes() + self.head_bytes()),
        )
    }
}

/// The ledger line the shared decode-head arena is billed under (H46; one line a rank, before the
/// KV pool is derived). The separate-arena control bills each head's arena on its
/// `engine.arena.head.r{rows}` line; under the shared law those lines carry only the head's own
/// row index, local pick, stagings and seam.
pub const SHARED_HEAD_ARENA_LINE: &str = "engine.arena.head.decode";

/// **H46: the sizing law of the shared decode-head arena** — C7's
/// [`crate::prefill_law::SharedPrefillArena`] applied to the decode HEADS: one allocation sized to
/// the widest head's arena, every decode row count's head laid over it.
///
/// Legal because a decode step runs ONE decode head (a fused mixed step runs one decode head plus
/// the prefill head, which keeps its own arena), every head buffer but the two gathered inputs is
/// written before it is read inside the head (`BlockPlan::stale_reads` is empty, asserted at boot),
/// and what is read after the step — the TP=2 winning logits, the dense row behind the seam — is
/// read from the head that ran LAST before any later step runs a head.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SharedHeadArena {
    /// `(rows, arena bytes)` per decode head, in planning order.
    pub heads: Vec<(u32, u64)>,
}

impl SharedHeadArena {
    /// The law over `(rows, arena bytes)` per planned decode head.
    #[must_use]
    pub fn of(heads: Vec<(u32, u64)>) -> Self {
        Self { heads }
    }

    /// Bytes the one allocation needs: the widest head's arena (0 with no head).
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.heads.iter().map(|(_, b)| *b).max().unwrap_or(0)
    }

    /// The row count that sizes it (the widest arena; the largest rows on a tie).
    #[must_use]
    pub fn widest_rows(&self) -> u32 {
        self.heads
            .iter()
            .max_by_key(|(r, b)| (*b, *r))
            .map_or(0, |(r, _)| *r)
    }

    /// What the separate-arena law bills for the same heads: the SUM.
    #[must_use]
    pub fn separate_bytes(&self) -> u64 {
        self.heads.iter().map(|(_, b)| *b).sum()
    }

    /// What sharing returns to the KV pool.
    #[must_use]
    pub fn refund_bytes(&self) -> u64 {
        self.separate_bytes() - self.bytes()
    }

    /// The boot line both ranks print beside the ledger's.
    #[must_use]
    pub fn line(&self, rank: u32) -> String {
        let mib = |b: u64| b as f64 / f64::from(1u32 << 20);
        format!(
            "headarena rank {rank}: SHARED over {} decode head(s), {} = {:.1} MiB (widest r{}); \
             separate arenas would be {:.1} MiB; refund {:.1} MiB to the KV pool",
            self.heads.len(),
            SHARED_HEAD_ARENA_LINE,
            mib(self.bytes()),
            self.widest_rows(),
            mib(self.separate_bytes()),
            mib(self.refund_bytes()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buckets(batches: &[u32], widths: &[u32]) -> Vec<Bucket> {
        let mut v = Vec::new();
        for b in batches {
            for w in widths {
                v.push(Bucket::decode(*b, *w, 4096, true));
            }
        }
        v
    }

    /// **The pick for every live count 1..8** under the ladder: the live count itself, 0 padding;
    /// the CONTROL (`1,2,4,8`) pads 3 -> 4 and 5 / 6 / 7 -> 8, as the served engine does today.
    #[test]
    fn the_ladder_picks_the_live_count_and_the_control_pads() {
        for w in [1u32, 5, 8] {
            let ladder = pick_table(&buckets(DECODE_BATCH_LADDER, &[1, 5, 8]), 8, w, 4096)
                .expect("the ladder holds 1..8");
            for p in &ladder {
                assert_eq!(p.batch, p.live, "width {w}: {p:?}");
                assert_eq!(p.padded_slots, 0);
            }
            let control = pick_table(&buckets(PRE_B3_DECODE_BATCHES, &[1, 5, 8]), 8, w, 4096)
                .expect("the control holds 1..8");
            let got: Vec<(u32, u32)> = control.iter().map(|p| (p.batch, p.padded_slots)).collect();
            assert_eq!(
                got,
                vec![
                    (1, 0),
                    (2, 0),
                    (4, 1),
                    (4, 0),
                    (8, 3),
                    (8, 2),
                    (8, 1),
                    (8, 0)
                ],
                "width {w}: the power-of-two control"
            );
        }
    }

    /// CMP27AF: the default holds b3 -- a live count of 3 runs at 3 (no padded slot), 5..7 still
    /// pad into b8; the control pads 3 -> 4.
    #[test]
    fn cmp27af_the_default_holds_b3_and_the_control_pads_it() {
        let got: Vec<u32> = (1..=8)
            .map(|l| pick_batch(DEFAULT_DECODE_BATCHES, l).unwrap())
            .collect();
        assert_eq!(got, vec![1, 2, 3, 4, 8, 8, 8, 8]);
        assert_eq!(pick_batch(PRE_B3_DECODE_BATCHES, 3), Some(4));
        let law = DecodeBucketLaw::of(DEFAULT_DECODE_BATCHES, &[1, 8], Vec::new(), Vec::new());
        assert_eq!(law.padded_over_live_counts(), 6);
    }

    /// A subset (e.g. `1,2,3,4,6,8`) pads only the counts it does not hold.
    #[test]
    fn a_subset_pads_only_what_it_does_not_hold() {
        let b = [1u32, 2, 3, 4, 6, 8];
        let got: Vec<u32> = (1..=8).map(|l| pick_batch(&b, l).unwrap()).collect();
        assert_eq!(got, vec![1, 2, 3, 4, 6, 6, 8, 8]);
        assert_eq!(pick_batch(&b, 9), None);
        assert_eq!(pick_batch(&b, 0), Some(1));
    }

    /// The boot line: padding over the live counts is 0 on the ladder and 7 on the control
    /// (1 + 3 + 2 + 1), and the arena sums are the ledger's.
    #[test]
    fn the_boot_line_states_the_padding_and_the_arenas() {
        let law = DecodeBucketLaw::of(
            DECODE_BATCH_LADDER,
            &[8, 1, 5],
            vec![(1, 8, 1 << 20), (3, 8, 3 << 20)],
            vec![(8, 2 << 20), (24, 6 << 20)],
        );
        assert_eq!(law.widths, vec![1, 5, 8]);
        assert_eq!(law.padded_over_live_counts(), 0);
        assert_eq!(law.body_bytes(), 4 << 20);
        assert_eq!(law.head_bytes(), 8 << 20);
        let line = law.line(1);
        assert!(line.starts_with("decodebuckets rank 1: batches [1, 2, 3, 4, 5, 6, 7, 8]"));
        assert!(line.contains("pick 1 2 3 4 5 6 7 8 (padded slots over live 1..8: 0)"));
        assert!(line.contains("= 12.0 MiB"), "{line}");
        let control = DecodeBucketLaw::of(PRE_B3_DECODE_BATCHES, &[8], Vec::new(), Vec::new());
        assert_eq!(control.padded_over_live_counts(), 7);
        assert!(control
            .line(0)
            .contains("pick 1 2 3->4 4 5->8 6->8 7->8 8 (padded slots over live 1..8: 7)"));
    }

    /// H46: the shared decode-head arena bills the WIDEST head, not the sum, and the boot line says
    /// what the separate law would have billed. Control: one head shares with nothing.
    #[test]
    fn the_shared_head_arena_bills_the_widest_head() {
        const MIB: u64 = 1 << 20;
        // ~0.75 MiB a row, as B1's ledger read it; the rows `1,2,4,8` x widths `1,5,8` plan.
        let rows = [1u32, 2, 4, 5, 8, 10, 16, 20, 32, 40, 64];
        let per_row = 3 * MIB / 4;
        let law = SharedHeadArena::of(rows.iter().map(|r| (*r, u64::from(*r) * per_row)).collect());
        assert_eq!(law.bytes(), 64 * per_row);
        assert_eq!(law.widest_rows(), 64);
        let sum: u64 = rows.iter().map(|r| u64::from(*r) * per_row).sum();
        assert_eq!(law.separate_bytes(), sum);
        assert_eq!(law.refund_bytes(), sum - 64 * per_row);
        let line = law.line(0);
        assert!(line.starts_with("headarena rank 0: SHARED over 11 decode head(s)"));
        assert!(line.contains(SHARED_HEAD_ARENA_LINE) && line.contains("(widest r64)"));
        let one = SharedHeadArena::of(vec![(8, 6 * MIB)]);
        assert_eq!(one.refund_bytes(), 0);
        // The bucket line counts the shared arena once, beside the per-row lines.
        let b = DecodeBucketLaw::of(&[1], &[8], Vec::new(), vec![(8, MIB)]).with_shared_heads(MIB);
        assert_eq!(b.head_bytes(), 2 * MIB);
        assert!(
            b.line(0).contains("(SHARED head arena 1.0 MiB)"),
            "{}",
            b.line(0)
        );
    }

    /// The flag texts and the lists agree.
    #[test]
    fn the_flag_texts_are_the_lists() {
        let parse = |s: &str| -> Vec<u32> { s.split(',').map(|t| t.parse().unwrap()).collect() };
        assert_eq!(parse(DEFAULT_DECODE_BATCHES_ARG), DEFAULT_DECODE_BATCHES);
        assert_eq!(parse(PRE_B3_DECODE_BATCHES_ARG), PRE_B3_DECODE_BATCHES);
        assert_eq!(parse(DECODE_BATCH_LADDER_ARG), DECODE_BATCH_LADDER);
    }
}
