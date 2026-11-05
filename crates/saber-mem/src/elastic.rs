//! FN51f: ELASTIC VRAM -- the tiered KV's hot pool and the expert slots as ONE lending budget
//! (docs/LEDGER.md FN51f; the reference's "the cache takes the rest and lends what no live request
//! holds to the expert slab").
//!
//! **The region.** The KV hot pool is allocated at the boot window + `--fn-vram-elastic-kv-max-gib`
//! (what a long request may borrow, carved out of the expert static slots by the exact ledger);
//! its top is cut into SEGMENTS of [`SEG_SLOTS`] expert slots each (`seg_pages` = `SEG_SLOTS x
//! ceil(stride / page)` pages: whole slots at either page size -- 16,384 B bf16 or 8,192 B fp8,
//! from the geometry, never a constant). Segment `i` (0 = the topmost) is the expert tier's
//! lendable slots `first + i * SEG_SLOTS ..`; the KV places blocks only below its CAP
//! (`total - lent x seg_pages`, [`crate::kv_tier::KvTier::hot_cap`]).
//!
//! **The law** ([`Elastic::tick`], once a step command on BOTH ranks between jobs -- every input
//! is the replicated KV bookkeeping, so both ranks take the same move with no exchange):
//!
//! * RECLAIM (KV <- experts) when the KV's `need` (the live reservations' hot floor -- windows +
//!   one chunk -- or, for a reservation past the window, its every block up to the region) is
//!   above the cap: every segment needed at once (a reclaim is a placement flip: the lent
//!   records fall back to their RAM shadows, one copy-stream drain whatever the count, no
//!   copy); the floor part is a correctness bound (an append must find a slot), never delayed.
//! * LEND (KV -> experts) when the cap is more than [`BAND_SEGS`] segments above the need for
//!   [`CALM_TICKS`] consecutive ticks: ONE segment a tick (the tier fills it hottest-first through
//!   its normal insert law: the mover's inserts-a-step cap unchanged), and only a segment whose
//!   pages hold no window / open block ([`crate::kv_tier::KvTier::can_shrink_hot`]); never past
//!   `segs` (the region less the always-kept chunk floor).
//! * Hysteresis: a reclaim resets the calm count, so a budget that just grew is not lent back
//!   for [`CALM_TICKS`] ticks; the band keeps a reservation that wobbles by a block from flapping.
//!
//! Placement only: no move changes a byte (each tier's own byte law; the region's ownership law
//! -- a slot is the KV's or the experts', never both -- is [`Elastic::owner_law`]).

#![forbid(unsafe_code)]

/// Expert slots a segment (the lending unit; whole slots at either page size).
pub const SEG_SLOTS: u32 = 4;
/// Ticks with a surplus before a lend (and the minimum gap after a reclaim).
pub const CALM_TICKS: u32 = 32;
/// Segments of surplus the law leaves unlent (the anti-flap band).
pub const BAND_SEGS: u32 = 2;

/// The region's geometry (one rank).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElasticGeom {
    /// KV pages of one segment (`SEG_SLOTS x ceil(stride / page_bytes)`).
    pub seg_pages: u32,
    /// The hot pool's pages (the whole region).
    pub total: u32,
    /// Segments that may be lent (the region less the always-kept floor).
    pub segs: u32,
}

impl ElasticGeom {
    /// The geometry of a hot pool of `total` pages of `page_bytes` beside expert slots of
    /// `stride` bytes, keeping `keep` pages (the chunk's open blocks) never lent.
    #[must_use]
    pub fn new(total: u32, page_bytes: u64, stride: u64, keep: u32) -> ElasticGeom {
        let slot_pages = stride.div_ceil(page_bytes.max(1)) as u32;
        let seg_pages = SEG_SLOTS * slot_pages.max(1);
        ElasticGeom {
            seg_pages,
            total,
            segs: total.saturating_sub(keep) / seg_pages,
        }
    }

    /// The first page of segment `i` (0 = the topmost).
    #[must_use]
    pub fn seg_first_page(&self, i: u32) -> u32 {
        self.total - (i + 1) * self.seg_pages
    }

    /// The KV's cap with `lent` segments lent.
    #[must_use]
    pub fn cap(&self, lent: u32) -> u32 {
        self.total - lent * self.seg_pages
    }

    /// The byte offset from the pool's page 0 of segment `i`'s `k`-th expert slot (slots back to
    /// back from the segment's first page, each `seg_pages / SEG_SLOTS` pages).
    #[must_use]
    pub fn slot_offset(&self, i: u32, k: u32, page_bytes: u64) -> u64 {
        let slot_bytes = u64::from(self.seg_pages / SEG_SLOTS) * page_bytes;
        u64::from(self.seg_first_page(i)) * page_bytes + u64::from(k) * slot_bytes
    }
}

/// One tick's decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    /// Nothing.
    Hold,
    /// Reclaim the `n` lowest lent segments (the KV grows).
    Reclaim(u32),
    /// Lend one more segment (the KV shrinks).
    Lend,
}

/// The controller's counters (`fnelastic` lines, `/metrics`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ElasticCounts {
    /// Ticks run.
    pub ticks: u64,
    /// Segments lent (one a lend).
    pub lends: u64,
    /// Reclaim moves.
    pub reclaims: u64,
    /// Segments those moves reclaimed.
    pub reclaimed_segs: u64,
    /// Lends held because a window / open block sat in the next segment.
    pub blocked: u64,
    /// The most segments lent at once.
    pub max_lent: u32,
    /// The fewest segments lent at once.
    pub min_lent: u32,
    /// FN130: reclaims taken on DEMAND (a page-opening site outside the step law's tick, or a
    /// step whose opens the tick's need did not cover) -- never a lend, never a calm count.
    pub early_reclaims: u64,
    /// FN130: segments those demand reclaims took.
    pub early_segs: u64,
    /// FN130: demand checks run (every site, with or without a move).
    pub demands: u64,
}

/// The controller (pure: the engine applies its moves).
#[derive(Debug, Clone)]
pub struct Elastic {
    /// The region.
    pub geom: ElasticGeom,
    /// Segments lent now.
    pub lent: u32,
    calm: u32,
    /// The counters.
    pub counts: ElasticCounts,
}

impl Elastic {
    /// A controller with `lent` segments lent at boot.
    #[must_use]
    pub fn new(geom: ElasticGeom, lent: u32) -> Elastic {
        Elastic {
            geom,
            lent,
            calm: 0,
            counts: ElasticCounts {
                max_lent: lent,
                min_lent: lent,
                ..ElasticCounts::default()
            },
        }
    }

    /// The segments to lend at boot for a KV `need` (pages): everything above it but the band.
    #[must_use]
    pub fn boot_lent(geom: &ElasticGeom, need: u64) -> u32 {
        let spare = u64::from(geom.total).saturating_sub(need) / u64::from(geom.seg_pages);
        (spare.saturating_sub(u64::from(BAND_SEGS)) as u32).min(geom.segs)
    }

    /// The KV's need in pages: the live hot floor, raised to `want` (a long reservation's every
    /// block) -- capped by the region.
    #[must_use]
    pub fn need(&self, floor: u64, want: u64) -> u64 {
        floor.max(want).min(u64::from(self.geom.total))
    }

    /// One tick: `need` = [`Self::need`]; `can_lend` = whether the next segment's pages hold no
    /// window / open block now (asked only when the law would lend: it walks every hot block).
    /// The caller applies the move, then [`Self::applied`].
    pub fn tick(&mut self, need: u64, can_lend: impl FnOnce() -> bool) -> Move {
        self.counts.ticks += 1;
        let cap = u64::from(self.geom.cap(self.lent));
        let sp = u64::from(self.geom.seg_pages);
        if need > cap {
            self.calm = 0;
            let n = (need - cap).div_ceil(sp).min(u64::from(self.lent)) as u32;
            return if n > 0 { Move::Reclaim(n) } else { Move::Hold };
        }
        let surplus = (cap - need) / sp;
        if surplus <= u64::from(BAND_SEGS) || self.lent >= self.geom.segs {
            self.calm = 0;
            return Move::Hold;
        }
        self.calm += 1;
        if self.calm < CALM_TICKS {
            return Move::Hold;
        }
        if !can_lend() {
            self.counts.blocked += 1;
            return Move::Hold;
        }
        Move::Lend
    }

    /// FN130: THE DEMAND LAW -- before a step opens KV pages at a site the tick did not run for
    /// (the prompt cache / prefix tree's reload of a stored piece), or when a step's exact opens
    /// pass what the cap can give: `need` = the controller's need ([`Self::need`]) and `short` =
    /// the opens' shortfall in pages ([`crate::kv_tier::KvTier::open_shortfall`]). Reclaims at once
    /// the segments both ask for (capped by the lent count; a reclaim is a placement flip, so the
    /// request waits for it inline, < 1 ms), never lends, never counts a tick or a calm step (the
    /// step law's cadence is unchanged for traffic that never races). The caller applies the
    /// move, then [`Self::applied_demand`].
    pub fn demand(&mut self, need: u64, short: u64) -> Move {
        self.counts.demands += 1;
        let cap = u64::from(self.geom.cap(self.lent));
        let sp = u64::from(self.geom.seg_pages);
        let by_need = need.saturating_sub(cap).div_ceil(sp);
        let by_short = short.div_ceil(sp);
        let n = by_need.max(by_short).min(u64::from(self.lent)) as u32;
        if n == 0 {
            return Move::Hold;
        }
        self.calm = 0;
        Move::Reclaim(n)
    }

    /// FN130: record an applied demand move (a reclaim, counted as one and as an early one).
    pub fn applied_demand(&mut self, m: Move) {
        if let Move::Reclaim(n) = m {
            self.counts.early_reclaims += 1;
            self.counts.early_segs += u64::from(n);
        }
        self.applied(m);
    }

    /// Record an applied move.
    pub fn applied(&mut self, m: Move) {
        match m {
            Move::Hold => {}
            Move::Reclaim(n) => {
                self.lent -= n;
                self.counts.reclaims += 1;
                self.counts.reclaimed_segs += u64::from(n);
            }
            Move::Lend => {
                self.lent += 1;
                self.counts.lends += 1;
            }
        }
        self.counts.max_lent = self.counts.max_lent.max(self.lent);
        self.counts.min_lent = self.counts.min_lent.min(self.lent);
    }

    /// The expert tier's lendable slot ids of segment `i` (registered from `first`).
    #[must_use]
    pub fn seg_ids(first: u32, i: u32) -> Vec<u32> {
        (first + i * SEG_SLOTS..first + (i + 1) * SEG_SLOTS).collect()
    }

    /// The region's OWNERSHIP LAW (tests, every move in the engine): the KV's cap is the region
    /// less the lent segments, and `lent_slots` (the expert tier's lent ids among this region's)
    /// are exactly the lent segments' ids.
    pub fn owner_law(&self, kv_cap: u32, first: u32, lent_slots: &[u32]) -> Result<(), String> {
        if kv_cap != self.geom.cap(self.lent) {
            return Err(format!(
                "KV cap {kv_cap} != region {} less {} lent segments of {}",
                self.geom.total, self.lent, self.geom.seg_pages
            ));
        }
        let mut want: Vec<u32> = (0..self.lent)
            .flat_map(|i| Self::seg_ids(first, i))
            .collect();
        let mut have = lent_slots.to_vec();
        want.sort_unstable();
        have.sort_unstable();
        if want != have {
            return Err(format!(
                "the expert tier holds {} of the region's slots lent, the law {}",
                have.len(),
                want.len()
            ));
        }
        Ok(())
    }

    /// FN51f-b: THE LEND'S ALIAS CHECK (before the expert tier takes segment `i`): the KV's cap
    /// is at or below the segment's first page, no KV block sits in its pages and none of them is
    /// on the KV's free list -- else the lend is REFUSED (a KV block in a lent page would be
    /// overwritten by the expert copy into the slot).
    pub fn lend_alias_check(&self, i: u32, kv: &crate::kv_tier::KvTier) -> Result<(), String> {
        let lo = self.geom.seg_first_page(i);
        let hi = lo + self.geom.seg_pages;
        let (held, free) = kv.hot_held_in(lo, hi);
        if kv.hot_cap() > lo || held > 0 || free {
            return Err(format!(
                "lend of segment {i} (pages {lo}..{hi}) refused: KV cap {}, {held} KV blocks in \
                 its pages{}",
                kv.hot_cap(),
                if free {
                    ", its pages on the KV free list"
                } else {
                    ""
                }
            ));
        }
        Ok(())
    }

    /// The `/metrics` exposition (`saber_xtier_elastic_*`, one rank).
    #[must_use]
    pub fn render_prometheus(&self, rank: u32, page_bytes: u64) -> String {
        let c = &self.counts;
        let mut o = String::new();
        o.push_str(
            "# HELP saber_xtier_elastic_lent_slots Expert slots the KV hot pool lends now (FN51f-b).\n\
             # TYPE saber_xtier_elastic_lent_slots gauge\n",
        );
        o.push_str(&format!(
            "saber_xtier_elastic_lent_slots{{rank=\"{rank}\"}} {}\n",
            self.lent * SEG_SLOTS
        ));
        o.push_str("# TYPE saber_xtier_elastic_kv_cap_bytes gauge\n");
        o.push_str(&format!(
            "saber_xtier_elastic_kv_cap_bytes{{rank=\"{rank}\"}} {}\n",
            u64::from(self.geom.cap(self.lent)) * page_bytes
        ));
        o.push_str("# TYPE saber_xtier_elastic_region_bytes gauge\n");
        o.push_str(&format!(
            "saber_xtier_elastic_region_bytes{{rank=\"{rank}\"}} {}\n",
            u64::from(self.geom.total) * page_bytes
        ));
        o.push_str("# TYPE saber_xtier_elastic_moves_total counter\n");
        for (k, v) in [
            ("lend", c.lends),
            ("reclaim", c.reclaims),
            ("reclaimed_segments", c.reclaimed_segs),
            ("blocked", c.blocked),
            ("tick", c.ticks),
            ("early_reclaim", c.early_reclaims),
            ("early_segments", c.early_segs),
            ("demand", c.demands),
        ] {
            o.push_str(&format!(
                "saber_xtier_elastic_moves_total{{rank=\"{rank}\",kind=\"{k}\"}} {v}\n"
            ));
        }
        o
    }

    /// The line printed at boot and at every move.
    #[must_use]
    pub fn line(&self, rank: u32, what: &str, d_pages: i64, d_slots: i64) -> String {
        format!(
            "fnelastic rank{rank}: {what} kv_pages {d_pages:+} slots {d_slots:+} | lent {} of {} \
             segments ({} slots), KV cap {} of {} pages | ticks {} lends {} reclaims {} ({} segs) \
             blocked {}",
            self.lent,
            self.geom.segs,
            self.lent * SEG_SLOTS,
            self.geom.cap(self.lent),
            self.geom.total,
            self.counts.ticks,
            self.counts.lends,
            self.counts.reclaims,
            self.counts.reclaimed_segs,
            self.counts.blocked
        )
    }
}
