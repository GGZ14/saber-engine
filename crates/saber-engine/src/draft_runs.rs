//! **P27-6: the drafter's KV pages sized by ADMISSION, not by `max_seqs x max_ctx`.**
//!
//! Until P27-6 every drafter session owned a fixed run of `max_ctx / 16` pages at a base derived
//! from its target GDN slot (`(slot - 1) / max_width * per`), so the drafter's KV was billed dense
//! up front: 8 sequences x 16,384 pages x 81,920 B = **10,240 MiB a rank at max-ctx 262,144**, a
//! reservation the target's KV pool was derived AFTER (21,169 blocks = 338,704 tokens).
//!
//! Under `--engine-draft-kv paged` the drafter's pool has exactly as many pages as the target's
//! pool has blocks (plus one page a sequence of slack), the pool is derived from the SAME byte
//! budget at `target block + drafter page` bytes a block, and a session's run is granted at
//! admission at the length its TARGET reservation implies (its block count, plus the drafter's own
//! block of query rows past the context, capped at the dense run). So the drafter's bytes follow
//! the admitted tokens — one admission ledger, released together.
//!
//! **The run stays ONE CONTIGUOUS PIECE per session.** Every consumer of a drafter run — the
//! admission clear (`Command::ClearDraftPages { base, pages }`), the fork copy
//! (`Command::ForkDraft`), the park image (one region per cache and layer), the prefix-cache image
//! (FN52) and both ranks' bookkeeping — addresses `(first, pages)`, and the drafter's attention
//! reads through a block table that is built from the session's page list every step. Nothing is
//! rebuilt: the kernels, the protocol and the images are the dense arm's, only WHERE a run starts
//! and HOW LONG it is moved.
//!
//! A contiguous allocator fragments, so this module also plans a **compaction**: when no hole holds
//! the run but the free pages do, every live run slides toward page 0 in address order (each move
//! a sequence of non-overlapping page copies — [`RunMove::copies`]), and the hole at the top is
//! granted. A move is a byte copy of a run that nothing reads between steps (an admission abandons
//! any stashed proposal first), so it moves no token; it is a fallback path and is counted
//! (`draftkv` end-of-run line, `CLAUDE.md` rule 5).

use std::collections::BTreeMap;

use saber_sched::SeqId;

/// One run slid toward page 0 by a compaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunMove {
    /// The session whose run moved.
    pub seq: SeqId,
    /// Its first page before the move.
    pub from: u32,
    /// Its first page after it (`to < from`).
    pub to: u32,
    /// Pages in the run.
    pub len: u32,
}

impl RunMove {
    /// The move as page copies `(from_first, to_first, pages)`, in the order they must run.
    ///
    /// A run moved by fewer pages than it holds overlaps itself; one `memcpy` over an overlapping
    /// range is undefined, so the run is copied in pieces of `from - to` pages, front first: each
    /// piece's destination is the previous piece's source, which has already been read.
    #[must_use]
    pub fn copies(&self) -> Vec<(u32, u32, u32)> {
        let gap = self.from - self.to;
        let mut out = Vec::new();
        let mut done = 0;
        while done < self.len {
            let n = gap.min(self.len - done);
            out.push((self.from + done, self.to + done, n));
            done += n;
        }
        out
    }
}

/// A run granted to a session: its first page and the compaction that made room for it, if any.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunGrant {
    /// The run's first page.
    pub first: u32,
    /// Pages in the run.
    pub len: u32,
    /// The moves a compaction made, in the order they must run (empty on the first-fit path).
    pub moves: Vec<RunMove>,
}

/// Contiguous page runs over a pool of `pages`, one per session.
#[derive(Clone, Debug, Default)]
pub struct RunAlloc {
    pages: u32,
    /// `first -> (len, seq)`, in address order.
    live: BTreeMap<u32, (u32, SeqId)>,
    /// `seq -> first`.
    by_seq: BTreeMap<SeqId, u32>,
}

impl RunAlloc {
    /// An empty pool of `pages`.
    #[must_use]
    pub fn new(pages: u32) -> Self {
        Self {
            pages,
            ..Self::default()
        }
    }

    /// Pages in the pool.
    #[must_use]
    pub fn pages(&self) -> u32 {
        self.pages
    }

    /// Pages held by live runs.
    #[must_use]
    pub fn used(&self) -> u32 {
        self.live.values().map(|(l, _)| *l).sum()
    }

    /// A session's run, `(first, len)`.
    #[must_use]
    pub fn run_of(&self, seq: SeqId) -> Option<(u32, u32)> {
        let first = *self.by_seq.get(&seq)?;
        self.live.get(&first).map(|(l, _)| (first, *l))
    }

    /// The lowest hole of at least `len` pages, first fit.
    fn hole(&self, len: u32) -> Option<u32> {
        let mut cursor = 0u32;
        for (&first, &(l, _)) in &self.live {
            if first - cursor >= len {
                return Some(cursor);
            }
            cursor = first + l;
        }
        (self.pages - cursor >= len).then_some(cursor)
    }

    /// Grant `seq` a run of `len` pages: first fit, else (when `compact`) slide every live run
    /// toward page 0 and take the hole at the top.
    ///
    /// # Errors
    /// The session already holds a run, a run of zero pages, or `len` pages that the pool's free
    /// pages cannot hold (or a fragmented pool with `compact` off) — named, with the pool's figures.
    pub fn grant(&mut self, seq: SeqId, len: u32, compact: bool) -> Result<RunGrant, String> {
        if self.by_seq.contains_key(&seq) {
            return Err(format!("{seq}: already holds a drafter page run"));
        }
        if len == 0 {
            return Err(format!("{seq}: a drafter page run of 0 pages"));
        }
        let free = self.pages - self.used();
        let mut moves = Vec::new();
        let first = match self.hole(len) {
            Some(f) => f,
            None if compact && free >= len => {
                let mut cursor = 0u32;
                let runs: Vec<(u32, u32, SeqId)> =
                    self.live.iter().map(|(&f, &(l, s))| (f, l, s)).collect();
                self.live.clear();
                for (f, l, s) in runs {
                    if f != cursor {
                        moves.push(RunMove {
                            seq: s,
                            from: f,
                            to: cursor,
                            len: l,
                        });
                    }
                    self.live.insert(cursor, (l, s));
                    self.by_seq.insert(s, cursor);
                    cursor += l;
                }
                cursor
            }
            None => {
                return Err(format!(
                    "{seq}: no drafter page run of {len} page(s): {free} of the pool's {} page(s) \
                     are free{} (P27-6 `--engine-draft-kv paged` sizes the drafter's pool by the \
                     target's; sequences that SHARE target pages — n > 1 branches, prefix-cache \
                     hits — each need a whole drafter run; `--engine-draft-kv dense` is the \
                     control)",
                    self.pages,
                    if free >= len {
                        " but fragmented and this path does not compact"
                    } else {
                        ""
                    }
                ))
            }
        };
        self.live.insert(first, (len, seq));
        self.by_seq.insert(seq, first);
        Ok(RunGrant { first, len, moves })
    }

    /// Return a session's run to the pool. Idempotent.
    pub fn free(&mut self, seq: SeqId) {
        if let Some(first) = self.by_seq.remove(&seq) {
            self.live.remove(&first);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_fit_reuses_the_lowest_hole() {
        let mut a = RunAlloc::new(100);
        assert_eq!(a.grant(SeqId(1), 10, true).unwrap().first, 0);
        assert_eq!(a.grant(SeqId(2), 20, true).unwrap().first, 10);
        assert_eq!(a.grant(SeqId(3), 5, true).unwrap().first, 30);
        a.free(SeqId(2));
        let g = a.grant(SeqId(4), 15, true).unwrap();
        assert_eq!((g.first, g.moves.len()), (10, 0));
        assert_eq!(a.used(), 30);
        assert!(a.grant(SeqId(4), 1, true).is_err(), "one run per session");
    }

    #[test]
    fn a_fragmented_pool_compacts_and_the_moves_are_exact() {
        let mut a = RunAlloc::new(40);
        for (s, l) in [(1, 10), (2, 10), (3, 10), (4, 10)] {
            a.grant(SeqId(s), l, true).unwrap();
        }
        a.free(SeqId(1));
        a.free(SeqId(3));
        // 20 pages free in two holes of 10: a 15-page run needs the compaction.
        assert!(
            a.clone().grant(SeqId(9), 15, false).is_err(),
            "no compaction: refused"
        );
        let g = a.grant(SeqId(9), 15, true).unwrap();
        assert_eq!(g.first, 20);
        assert_eq!(
            g.moves,
            vec![
                RunMove {
                    seq: SeqId(2),
                    from: 10,
                    to: 0,
                    len: 10
                },
                RunMove {
                    seq: SeqId(4),
                    from: 30,
                    to: 10,
                    len: 10
                },
            ]
        );
        assert_eq!(a.run_of(SeqId(2)), Some((0, 10)));
        assert_eq!(a.run_of(SeqId(4)), Some((10, 10)));
        assert_eq!(a.run_of(SeqId(9)), Some((20, 15)));
        // More than the free pages: refused by name.
        let e = a.grant(SeqId(10), 6, true).unwrap_err();
        assert!(e.contains("5 of the pool's 40"), "{e}");
    }

    #[test]
    fn an_overlapping_move_copies_in_non_overlapping_pieces() {
        // The NEGATIVE CONTROL: a single copy of an overlapping move would read pages it wrote.
        let m = RunMove {
            seq: SeqId(1),
            from: 3,
            to: 0,
            len: 10,
        };
        let c = m.copies();
        assert_eq!(c, vec![(3, 0, 3), (6, 3, 3), (9, 6, 3), (12, 9, 1)]);
        // Simulate on a page array: every destination equals the source's original content.
        let mut pages: Vec<u32> = (0..13).collect();
        for (f, t, n) in &c {
            for i in 0..*n {
                assert!(t + i < *f || t + i >= f + n, "a piece overlaps itself");
                pages[(t + i) as usize] = pages[(f + i) as usize];
            }
        }
        assert_eq!(&pages[..10], &(3..13).collect::<Vec<u32>>()[..]);
        let far = RunMove {
            seq: SeqId(1),
            from: 50,
            to: 0,
            len: 10,
        };
        assert_eq!(far.copies(), vec![(50, 0, 10)]);
    }
}
