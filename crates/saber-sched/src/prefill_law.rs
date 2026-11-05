//! **C7: the prefill rungs' memory and row laws, as data** — host-only, so the byte ledger, the
//! engine and the tests read one statement of each.
//!
//! * [`SharedPrefillArena`] — every prefill rung laid over ONE activation arena sized to the widest
//!   rung, instead of an arena per rung. A prefill chunk is one sequence's rows in one step, every
//!   buffer of the body arena is written before it is read inside the chunk, and nothing in it is
//!   read by a later chunk (the KV pages, the GDN state, the aux tap and the head's own arena live
//!   elsewhere), so two rungs never need their arenas at the same time. The ladder's eleven rungs
//!   cost the SUM of their arenas (C5: +919 MiB a rank over `256,2048`); the shared law costs the
//!   MAX, and the difference goes back to the KV pool, which the ledger derives from what is left.
//! * [`ragged_rows`] — the row count a chunk would be PLANNED at under a ragged law: its live rows
//!   rounded up to the kernels' row granule, never past its rung.
//! * [`PlanLru`] — the bounded cache such a law keys its plans on (host memory only once the arena
//!   is shared: a plan over the shared arena allocates no device bytes).

use std::collections::VecDeque;

/// The ledger line the shared prefill arena is billed under (one line a rank, before the KV pool
/// is derived). The per-rung lines (`engine.arena.p<rows>`) are what the separate-arena control
/// bills.
pub const SHARED_PREFILL_ARENA_LINE: &str = "engine.arena.prefill";

/// The sizing law of the shared prefill arena: the widest rung's arena, over every planned rung.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SharedPrefillArena {
    /// `(rows, arena bytes)` per rung, in the order they were planned.
    pub rungs: Vec<(u32, u64)>,
}

impl SharedPrefillArena {
    /// The law over `(rows, arena bytes)` per planned rung.
    #[must_use]
    pub fn of(rungs: Vec<(u32, u64)>) -> Self {
        Self { rungs }
    }

    /// Bytes the ONE shared allocation needs: the widest rung's arena (0 with no rung).
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.rungs.iter().map(|(_, b)| *b).max().unwrap_or(0)
    }

    /// The rung that sizes it (the widest arena; the largest rows on a tie).
    #[must_use]
    pub fn widest_rows(&self) -> u32 {
        self.rungs
            .iter()
            .max_by_key(|(r, b)| (*b, *r))
            .map_or(0, |(r, _)| *r)
    }

    /// Bytes the separate-arena law bills for the same rungs: the SUM.
    #[must_use]
    pub fn separate_bytes(&self) -> u64 {
        self.rungs.iter().map(|(_, b)| *b).sum()
    }

    /// What sharing returns to the KV pool: separate − shared.
    #[must_use]
    pub fn refund_bytes(&self) -> u64 {
        self.separate_bytes() - self.bytes()
    }

    /// Whether a plan of `bytes` fits the shared allocation (the executor refuses one that does
    /// not; this is the same statement on the host).
    #[must_use]
    pub fn holds(&self, bytes: u64) -> bool {
        bytes <= self.bytes()
    }

    /// The boot line both ranks print beside the ledger's.
    #[must_use]
    pub fn line(&self, rank: u32) -> String {
        let mib = |b: u64| b as f64 / f64::from(1u32 << 20);
        format!(
            "prefillarena rank {rank}: SHARED over {} rung(s), {} = {:.1} MiB (widest p{}); \
             separate arenas would be {:.1} MiB; refund {:.1} MiB to the KV pool",
            self.rungs.len(),
            SHARED_PREFILL_ARENA_LINE,
            mib(self.bytes()),
            self.widest_rows(),
            mib(self.separate_bytes()),
            mib(self.refund_bytes()),
        )
    }
}

/// The kernels' row granule for a ragged prefill plan: the WMMA fragment height (16) — every
/// prefill GEMM, producer and reduce is row-local at this granularity.
pub const RAGGED_ROW_GRANULE: u32 = 16;

/// The rows a chunk of `live` rows would be planned at under the ragged law: `live` rounded up to
/// `granule`, never past its `rung` (a chunk is padded UP into its rung, never truncated), and
/// never below one granule. `granule` 0 is treated as 1.
#[must_use]
pub fn ragged_rows(live: u32, rung: u32, granule: u32) -> u32 {
    let g = granule.max(1);
    let up = live.max(1).div_ceil(g).saturating_mul(g);
    up.min(rung.max(live))
}

/// A bounded least-recently-used cache keyed on a row count, with its own counters.
#[derive(Debug)]
pub struct PlanLru<V> {
    cap: usize,
    entries: VecDeque<(u32, V)>,
    /// Lookups answered from the cache.
    pub hits: u64,
    /// Lookups that built a value.
    pub misses: u64,
    /// Values dropped to stay at `cap`.
    pub evictions: u64,
}

impl<V> PlanLru<V> {
    /// An empty cache that holds at most `cap` values (at least one).
    #[must_use]
    pub fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            entries: VecDeque::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Values held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing is held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The keys held, most recently used first.
    #[must_use]
    pub fn keys(&self) -> Vec<u32> {
        self.entries.iter().map(|(k, _)| *k).collect()
    }

    /// The value for `key`, built by `make` on a miss (evicting the least recently used value when
    /// full), and marked most recently used either way.
    ///
    /// # Errors
    /// Whatever `make` returns; nothing is inserted then.
    pub fn get_or_try_insert_with<E>(
        &mut self,
        key: u32,
        make: impl FnOnce() -> Result<V, E>,
    ) -> Result<&mut V, E> {
        if let Some(i) = self.entries.iter().position(|(k, _)| *k == key) {
            self.hits += 1;
            let e = self.entries.remove(i).expect("position is in range");
            self.entries.push_front(e);
        } else {
            self.misses += 1;
            let v = make()?;
            if self.entries.len() == self.cap {
                self.entries.pop_back();
                self.evictions += 1;
            }
            self.entries.push_front((key, v));
        }
        Ok(&mut self.entries.front_mut().expect("just pushed").1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    /// C5's ladder at C5's per-rung cost (~0.19 MiB a row, linear: `engine.arena.p2048` = 381 MiB):
    /// the separate law bills the SUM, the shared law the MAX, and the refund is the difference.
    #[test]
    fn the_shared_arena_is_the_widest_rung_and_refunds_the_rest() {
        let ladder = [96u32, 128, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048];
        let per_row = 381 * MIB / 2048;
        let law = SharedPrefillArena::of(
            ladder
                .iter()
                .map(|r| (*r, u64::from(*r) * per_row))
                .collect(),
        );
        assert_eq!(law.bytes(), 2048 * per_row);
        assert_eq!(law.widest_rows(), 2048);
        let sum: u64 = ladder.iter().map(|r| u64::from(*r) * per_row).sum();
        assert_eq!(law.separate_bytes(), sum);
        assert_eq!(law.refund_bytes(), sum - 2048 * per_row);
        // C5's +919 MiB is the ladder's arenas over `256,2048`'s; the refund is that plus p256.
        let refund_mib = law.refund_bytes() / MIB;
        assert!((900..1000).contains(&refund_mib), "refund {refund_mib} MiB");
        assert!(law.holds(2048 * per_row) && !law.holds(2048 * per_row + 1));
        assert!(law.line(0).contains(SHARED_PREFILL_ARENA_LINE));
        // CONTROL: one rung shares with nothing, so there is nothing to refund.
        let one = SharedPrefillArena::of(vec![(2048, 381 * MIB)]);
        assert_eq!(one.refund_bytes(), 0);
        assert_eq!(one.bytes(), one.separate_bytes());
    }

    /// G4's rungs above 2048 under the shared law: the widest (p8192, 1,526 MiB) sizes it, so
    /// `+4096,8192` costs 1526 − 381 MiB over the ladder instead of 1526 + 763.
    #[test]
    fn the_rungs_above_2048_cost_their_widest_only() {
        let lad = SharedPrefillArena::of(vec![(1536, 286 * MIB), (2048, 381 * MIB)]);
        let big = SharedPrefillArena::of(vec![
            (1536, 286 * MIB),
            (2048, 381 * MIB),
            (4096, 763 * MIB),
            (8192, 1526 * MIB),
        ]);
        assert_eq!(big.bytes() - lad.bytes(), (1526 - 381) * MIB);
        assert_eq!(
            big.separate_bytes() - lad.separate_bytes(),
            (763 + 1526) * MIB
        );
    }

    #[test]
    fn a_ragged_chunk_rounds_to_the_granule_inside_its_rung() {
        assert_eq!(ragged_rows(72, 96, 16), 80);
        assert_eq!(ragged_rows(80, 96, 16), 80);
        assert_eq!(ragged_rows(1, 96, 16), 16);
        assert_eq!(ragged_rows(282, 320, 16), 288);
        assert_eq!(ragged_rows(319, 320, 16), 320);
        assert_eq!(ragged_rows(2047, 2048, 16), 2048);
        // Never past the rung, never truncating the chunk.
        assert_eq!(ragged_rows(95, 96, 64), 96);
        assert_eq!(ragged_rows(100, 96, 16), 100);
        // CONTROL: granule 1 is the live rows exactly; the rung's own rows are the bucket law.
        assert_eq!(ragged_rows(72, 96, 1), 72);
        assert_eq!(ragged_rows(72, 96, 96), 96);
    }

    #[test]
    fn the_plan_cache_is_lru_and_counts() {
        let mut lru: PlanLru<String> = PlanLru::new(2);
        let mut built = 0;
        for k in [80u32, 96, 80, 112, 96] {
            let v = lru
                .get_or_try_insert_with::<()>(k, || {
                    built += 1;
                    Ok(format!("p{k}"))
                })
                .unwrap();
            assert_eq!(*v, format!("p{k}"));
        }
        // 80 miss, 96 miss, 80 hit, 112 miss (evicts 96), 96 miss (evicts 80).
        assert_eq!((lru.hits, lru.misses, lru.evictions, built), (1, 4, 2, 4));
        assert_eq!(lru.keys(), vec![96, 112]);
        // A failed build inserts nothing.
        assert!(lru
            .get_or_try_insert_with(7, || Err::<String, _>("no"))
            .is_err());
        assert_eq!(lru.len(), 2);
    }
}
