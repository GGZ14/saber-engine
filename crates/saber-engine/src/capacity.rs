//! What a card actually holds, in KV tokens — the capacity side of the byte ledger.
//!
//! # Why this exists
//!
//! The byte ledger already refuses loudly when a reservation does not fit. What it does not say is
//! the question a serving decision is actually made on: **after everything the engine allocates, how
//! many resident KV tokens are left, and how many concurrent sessions is that?** Bytes are the
//! ledger's unit; tokens are the operator's. Prod's comparators are in tokens (760k–854k at TP=2 for
//! its MXFP4 units at 0.95 utilisation), so this converts, once, in one place, with the arithmetic
//! visible rather than done in somebody's head from a `MiB` line.
//!
//! # What it is not
//!
//! Not a planner and not a policy. Nothing branches on what this computes (`CLAUDE.md` rule 6); it
//! reads a ledger snapshot and some geometry and renders a table. The admission path's authority is
//! still the ledger's own refusal, which is why the harness that goes with this admits sessions
//! until the ledger says no and counts them, rather than trusting the arithmetic below.

use std::fmt::Write as _;

use saber_mem::ledger::{LedgerSnapshot, Pool};

/// The geometry a KV-token count needs.
#[derive(Clone, Copy, Debug)]
pub struct KvGeometry {
    /// Bytes one KV block occupies on **this rank**, summed over every attention layer.
    pub block_bytes: u64,
    /// Tokens per block.
    pub block_tokens: u32,
    /// Bytes one GDN state slot occupies on this rank.
    pub gdn_bytes_per_slot: u64,
    /// State slots one sequence reserves: `spec_width + 1`.
    pub gdn_slots_per_seq: u32,
}

impl KvGeometry {
    /// Bytes one resident token costs in KV alone.
    #[must_use]
    pub fn bytes_per_token(&self) -> f64 {
        self.block_bytes as f64 / f64::from(self.block_tokens.max(1))
    }

    /// Bytes one sequence costs **besides** its KV: its GDN state slot set.
    #[must_use]
    pub fn bytes_per_sequence(&self) -> u64 {
        u64::from(self.gdn_slots_per_seq) * self.gdn_bytes_per_slot
    }

    /// How many sessions of `ctx` tokens each fit in `bytes`, KV and state together.
    #[must_use]
    pub fn sessions_in(&self, bytes: u64, ctx: u32) -> u64 {
        let per =
            (f64::from(ctx) * self.bytes_per_token()).ceil() as u64 + self.bytes_per_sequence();
        if per == 0 {
            return 0;
        }
        bytes / per
    }
}

/// One line of the waste census: a reservation, and what it costs in KV tokens.
#[derive(Clone, Debug)]
pub struct Waste {
    /// The reservation's name in the ledger.
    pub name: String,
    /// Bytes it holds.
    pub bytes: u64,
    /// Bytes a step actually touches, when that is known and smaller.
    pub touched: Option<u64>,
    /// Why it is bigger, in one clause.
    pub why: String,
}

impl Waste {
    /// Bytes that are held and not touched.
    #[must_use]
    pub fn slack(&self) -> u64 {
        self.touched.map_or(0, |t| self.bytes.saturating_sub(t))
    }
}

/// What one card holds, after everything.
#[derive(Clone, Debug)]
pub struct Capacity {
    /// The rank this describes.
    pub rank: u32,
    /// Declared VRAM for the rank.
    pub capacity: u64,
    /// Reserved by everything that is not the KV pool.
    pub overhead: u64,
    /// The KV pool's own reservation.
    pub kv: u64,
    /// The GDN state arena's reservation.
    pub gdn: u64,
    /// Bytes declared and reserved by nobody.
    pub unreserved: u64,
    /// The geometry the token counts come from.
    pub geom: KvGeometry,
}

impl Capacity {
    /// Read a rank's capacity out of a ledger snapshot.
    ///
    /// `kv_names` and `gdn_names` are the reservation names the arenas use; everything else on the
    /// rank is overhead. Names rather than a guess, because a reservation this does not recognise
    /// must land in **overhead** — counting an unknown reservation as KV would inflate the answer,
    /// which is the direction that gets somebody an OOM at token 4000.
    #[must_use]
    pub fn from_snapshot(
        snap: &LedgerSnapshot,
        rank: u32,
        kv_names: &[&str],
        gdn_names: &[&str],
        geom: KvGeometry,
    ) -> Capacity {
        let pool = snap.pools.iter().find(|p| p.pool == Pool::Vram(rank));
        let (capacity, used, reservations) = pool.map_or((0, 0, Vec::new()), |p| {
            (p.capacity_bytes, p.used_bytes, p.reservations.clone())
        });
        let mut kv = 0;
        let mut gdn = 0;
        for r in &reservations {
            if kv_names.iter().any(|n| r.name.contains(n)) {
                kv += r.bytes;
            } else if gdn_names.iter().any(|n| r.name.contains(n)) {
                gdn += r.bytes;
            }
        }
        Capacity {
            rank,
            capacity,
            overhead: used.saturating_sub(kv + gdn),
            kv,
            gdn,
            unreserved: capacity.saturating_sub(used),
            geom,
        }
    }

    /// Resident KV tokens the pool holds.
    #[must_use]
    pub fn kv_tokens(&self) -> u64 {
        let per = self.geom.bytes_per_token();
        if per <= 0.0 {
            return 0;
        }
        (self.kv as f64 / per) as u64
    }

    /// Sequences the GDN state arena holds.
    #[must_use]
    pub fn state_sequences(&self) -> u64 {
        let per = self.geom.bytes_per_sequence();
        if per == 0 {
            return 0;
        }
        self.gdn / per
    }

    /// Concurrent sessions of `ctx` tokens: the smaller of what KV holds and what state holds.
    ///
    /// **The smaller of the two, always.** A card with KV for forty sessions and state for eight
    /// serves eight, and reporting forty would be reporting a number nothing can reach.
    #[must_use]
    pub fn sessions(&self, ctx: u32) -> u64 {
        let by_kv = self.kv_tokens() / u64::from(ctx.max(1));
        by_kv.min(self.state_sequences())
    }

    /// What the unreserved remainder would add if it were given to KV.
    #[must_use]
    pub fn headroom_tokens(&self) -> u64 {
        let per = self.geom.bytes_per_token();
        if per <= 0.0 {
            return 0;
        }
        (self.unreserved as f64 / per) as u64
    }

    /// The table a boot log prints.
    #[must_use]
    pub fn render(&self, ctx: u32) -> String {
        let mib = |b: u64| b >> 20;
        let mut out = String::new();
        let _ = writeln!(
            out,
            "capacity  rank {}: {} MiB declared, {} MiB overhead, {} MiB KV, {} MiB GDN state, {} \
             MiB unreserved",
            self.rank,
            mib(self.capacity),
            mib(self.overhead),
            mib(self.kv),
            mib(self.gdn),
            mib(self.unreserved),
        );
        let _ = writeln!(
            out,
            "capacity  {:.0} B/token, {} MiB/sequence of state → **{} KV tokens**, {} state \
             sequence(s), {} concurrent {}k session(s)",
            self.geom.bytes_per_token(),
            mib(self.geom.bytes_per_sequence()),
            self.kv_tokens(),
            self.state_sequences(),
            self.sessions(ctx),
            ctx / 1024,
        );
        if self.unreserved > 0 {
            let _ = writeln!(
                out,
                "capacity  {} MiB is declared and reserved by nobody — {} more KV tokens if it were \
                 given to the pool",
                mib(self.unreserved),
                self.headroom_tokens()
            );
        }
        out
    }
}

/// Render a waste census: what is held, what a step touches, and the KV tokens the difference costs.
#[must_use]
pub fn render_waste(rows: &[Waste], geom: KvGeometry) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "waste     reservation                 held MiB   touched MiB   slack MiB   KV tokens"
    );
    let per = geom.bytes_per_token();
    let mut total = 0u64;
    for w in rows {
        total += w.slack();
        let _ = writeln!(
            out,
            "waste     {:<26} {:>9} {:>13} {:>11} {:>11}  {}",
            w.name,
            w.bytes >> 20,
            w.touched
                .map_or_else(|| "?".to_owned(), |t| (t >> 20).to_string()),
            w.slack() >> 20,
            if per > 0.0 {
                (w.slack() as f64 / per) as u64
            } else {
                0
            },
            w.why
        );
    }
    let _ = writeln!(
        out,
        "waste     TOTAL slack {} MiB = {} KV tokens",
        total >> 20,
        if per > 0.0 {
            (total as f64 / per) as u64
        } else {
            0
        }
    );
    out
}

/// Does the arithmetic match what admission grants?
///
/// The census computes KV tokens from bytes; the allocator hands out blocks. If the two disagree the
/// census is decoration. This is the assertion that keeps it honest, and it is run at ready rather
/// than in a test, because the numbers it compares only exist on a booted engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Granted {
    /// Sessions the census predicted at `ctx`.
    pub predicted: u64,
    /// Sessions the allocator actually handed out before refusing.
    pub granted: u64,
}

impl Granted {
    /// Whether the census told the truth.
    ///
    /// Exact equality: the census and the allocator work from the same two numbers, so anything
    /// other than equality is a bug in one of them, not rounding. The one allowance is that the
    /// allocator may grant **more** when the ceiling on admission (`max_seqs`) is not what binds —
    /// the census reports the smaller of KV and state, and state is a ceiling the allocator also
    /// applies, so it cannot exceed it.
    #[must_use]
    pub fn agrees(&self) -> bool {
        self.predicted == self.granted
    }

    /// The line a boot log prints.
    #[must_use]
    pub fn render(&self) -> String {
        if self.agrees() {
            format!(
                "admit     OK: the census said {} session(s) and admission granted {}",
                self.predicted, self.granted
            )
        } else {
            format!(
                "admit     MISMATCH: the census said {} session(s) and admission granted {} — one \
                 of them is wrong and it is not rounding",
                self.predicted, self.granted
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use saber_mem::ledger::{Ledger, Pool};

    fn geom() -> KvGeometry {
        // Prod's shapes at TP=1: 16 attention layers x 32 KiB per block of 16 tokens.
        KvGeometry {
            block_bytes: 512 * 1024,
            block_tokens: 16,
            gdn_bytes_per_slot: 146 << 20,
            gdn_slots_per_seq: 2,
        }
    }

    #[test]
    fn a_reservation_the_census_does_not_recognise_counts_as_overhead() {
        // The direction that matters: an unknown reservation counted as KV would report capacity
        // that is not there, and the operator finds out at token 4000.
        let mut l = Ledger::new();
        l.declare(Pool::Vram(0), 32 << 30).expect("declare");
        l.reserve(Pool::Vram(0), "weights", 18 << 30).expect("w");
        l.reserve(Pool::Vram(0), "scheduler.kv", 4 << 30)
            .expect("kv");
        l.reserve(Pool::Vram(0), "scheduler.gdn_state", 2 << 30)
            .expect("gdn");
        l.reserve(Pool::Vram(0), "something.new", 1 << 30)
            .expect("new");
        let c = Capacity::from_snapshot(
            &l.snapshot(),
            0,
            &["scheduler.kv"],
            &["scheduler.gdn_state"],
            geom(),
        );
        assert_eq!(c.kv, 4 << 30);
        assert_eq!(c.gdn, 2 << 30);
        assert_eq!(c.overhead, 19 << 30, "the unknown MiB are overhead, not KV");
        assert_eq!(c.unreserved, 7 << 30);
    }

    /// **THE DRAFTER TIER IS THREE LINES AND THE THIRD IS THE PROPOSAL'S ARENA** (ws-s11, ws-k15).
    ///
    /// The census reads the ledger, so a byte the engine allocates and never reserves is a byte
    /// the census reports as FREE — and the KV pool is derived from exactly that number
    /// (ws-k13). The drafter's arena was such a byte until this round: a raw `hipMalloc` in
    /// `build_drafter`, 20 MiB at one request and 118 at eight, absorbed by `kv_headroom_mib`.
    /// Here it is a reservation like the other three, and the control is the same census with the
    /// line missing — which reports 118 MiB of capacity that is not there.
    #[test]
    fn the_drafters_arena_is_in_the_census() {
        let mut l = Ledger::new();
        l.declare(Pool::Vram(0), 32 << 30).expect("declare");
        l.reserve(Pool::Vram(0), "weights", 15 << 30).expect("w");
        l.reserve(Pool::Vram(0), "engine.draft.weights", 7134 << 20)
            .expect("draft weights");
        l.reserve(Pool::Vram(0), "engine.draft.kv", 830 << 20)
            .expect("draft kv");
        // (There is no `engine.draft.context` line: ws-k15 removed it, because the buffer it
        // billed is never allocated — `DraftDemand::total`.)
        let arena = 118u64 << 20;
        l.reserve(Pool::Vram(0), "engine.draft.arena", arena)
            .expect("draft arena");
        l.reserve(Pool::Vram(0), "scheduler.kv", 4 << 30)
            .expect("kv");
        let with = Capacity::from_snapshot(&l.snapshot(), 0, &["scheduler.kv"], &[], geom());
        assert_eq!(
            with.kv,
            4 << 30,
            "the drafter's kv is NOT the target's pool"
        );

        // THE CONTROL: the same engine with the arena unbilled. Everything else is equal, so the
        // difference is exactly the arena — and it lands in `unreserved`, which is what the pool
        // derivation spends.
        let mut l2 = Ledger::new();
        l2.declare(Pool::Vram(0), 32 << 30).expect("declare");
        l2.reserve(Pool::Vram(0), "weights", 15 << 30).expect("w");
        l2.reserve(Pool::Vram(0), "engine.draft.weights", 7134 << 20)
            .expect("draft weights");
        l2.reserve(Pool::Vram(0), "engine.draft.kv", 830 << 20)
            .expect("draft kv");
        l2.reserve(Pool::Vram(0), "scheduler.kv", 4 << 30)
            .expect("kv");
        let without = Capacity::from_snapshot(&l2.snapshot(), 0, &["scheduler.kv"], &[], geom());
        assert_eq!(
            without.unreserved - with.unreserved,
            arena,
            "the unbilled arena reads as free capacity"
        );
        assert_eq!(
            with.overhead - without.overhead,
            arena,
            "the arena is OVERHEAD — not KV, and not state"
        );
        // And it is worth KV tokens the operator would otherwise be promised: 118 MiB at this
        // geometry is 3,776 tokens of context that the pool would have handed out twice.
        assert_eq!(
            arena / (geom().block_bytes / u64::from(geom().block_tokens)),
            3_776
        );
    }

    #[test]
    fn concurrency_is_the_smaller_of_kv_and_state() {
        let mut l = Ledger::new();
        l.declare(Pool::Vram(0), 32 << 30).expect("declare");
        // KV for a lot of 8k sessions, state for two.
        l.reserve(Pool::Vram(0), "scheduler.kv", 8 << 30)
            .expect("kv");
        l.reserve(Pool::Vram(0), "scheduler.gdn_state", 584 << 20)
            .expect("gdn");
        let c = Capacity::from_snapshot(
            &l.snapshot(),
            0,
            &["scheduler.kv"],
            &["scheduler.gdn_state"],
            geom(),
        );
        assert_eq!(c.kv_tokens(), 262_144, "8 GiB at 32 KiB/token");
        assert_eq!(c.state_sequences(), 2);
        assert_eq!(
            c.sessions(8192),
            2,
            "KV holds 32 sessions and state holds 2, so the card serves 2"
        );
        // NEGATIVE CONTROL: with state for forty, KV becomes the binding term and the answer moves.
        let mut l2 = Ledger::new();
        l2.declare(Pool::Vram(0), 64 << 30).expect("declare");
        l2.reserve(Pool::Vram(0), "scheduler.kv", 8 << 30)
            .expect("kv");
        l2.reserve(Pool::Vram(0), "scheduler.gdn_state", 40 * (292 << 20))
            .expect("gdn");
        let c2 = Capacity::from_snapshot(
            &l2.snapshot(),
            0,
            &["scheduler.kv"],
            &["scheduler.gdn_state"],
            geom(),
        );
        assert_eq!(c2.sessions(8192), 32, "now KV binds");
    }

    #[test]
    fn the_census_and_the_allocator_must_agree_exactly() {
        assert!(Granted {
            predicted: 8,
            granted: 8
        }
        .agrees());
        // NEGATIVE CONTROL: off by one is a mismatch, not a rounding allowance. A census that
        // tolerated drift would stop being a check the moment the geometry changed.
        let bad = Granted {
            predicted: 9,
            granted: 8,
        };
        assert!(!bad.agrees());
        assert!(bad.render().contains("MISMATCH"), "{}", bad.render());
    }

    /// The pool's EDGE: the census says N, the allocator grants exactly N, and the N+1th is
    /// refused.
    ///
    /// ws-k13 asked for this one by name. The census and the allocator are two readings of the same
    /// two geometry figures, and the failure that matters is not "the census is roughly right" but
    /// "the census is right up to the last block and wrong after it" — a pool derived from the byte
    /// ledger has no slack left over to hide an off-by-one in.
    #[test]
    fn admission_is_refused_at_exactly_the_pool_edge() {
        use saber_plan::Bucket;
        use saber_sched::{
            ArenaBytes, Arenas, KvGeometry as SchedKv, SchedulerConfig, GDN_ARENA_RESERVATION,
            KV_ARENA_RESERVATION,
        };

        const CTX: u32 = 8192;
        const PER_SESSION_KV: u64 = 256 << 20; // 8192 tokens x 32 KiB
        const PER_SLOT: u64 = 1 << 20;

        // Drive the allocator with `kv_bytes` of KV arena and return what the census predicted
        // beside what admission actually handed out.
        let run = |kv_bytes: u64, sessions: u64| -> Granted {
            let cfg = SchedulerConfig::for_mock(
                vec![Bucket::decode(1, 1, CTX, true)],
                1,
                PER_SLOT,
                SchedKv {
                    block_tokens: 16,
                    bytes_per_token_per_rank: 32 << 10,
                },
                ArenaBytes {
                    kv_per_rank: kv_bytes,
                    // Two slots a sequence (`spec_width + 1`), and enough for `sessions` of them,
                    // so that KV is the term on trial here and state never binds first.
                    gdn_state_per_rank: sessions * 2 * PER_SLOT,
                },
                CTX as usize,
                "ws-k13:pool-edge".to_owned(),
            )
            .expect("a mock config");
            let mut ledger = Ledger::new();
            ledger.declare(Pool::Vram(0), 32 << 30).expect("declare");
            let mut arenas = Arenas::reserve(&cfg, &mut ledger).expect("reserve");
            let predicted = Capacity::from_snapshot(
                &ledger.snapshot(),
                0,
                &[KV_ARENA_RESERVATION],
                &[GDN_ARENA_RESERVATION],
                KvGeometry {
                    block_bytes: 512 << 10,
                    block_tokens: 16,
                    gdn_bytes_per_slot: PER_SLOT,
                    gdn_slots_per_seq: 2,
                },
            )
            .sessions(CTX);
            let want = u64::from(CTX);
            let mut n = 0u64;
            while arenas.allocate_group(1, want, want).is_ok() {
                n += 1;
                assert!(n < 64, "the arena never refused");
            }
            Granted {
                predicted,
                granted: n,
            }
        };

        let exact = run(4 * PER_SESSION_KV, 4);
        assert_eq!(exact.predicted, 4, "the census");
        assert_eq!(exact.granted, 4, "the allocator: {}", exact.render());
        assert!(exact.agrees(), "{}", exact.render());

        // NEGATIVE CONTROL: one block less — 512 KiB out of a gigabyte — and both numbers move
        // down by one together. A test that passed for a range of arena sizes would not be testing
        // an edge, and neither reading may drift while the other stays put.
        let short = run(4 * PER_SESSION_KV - (512 << 10), 4);
        assert_eq!(short.predicted, 3, "the census, one block short");
        assert_eq!(short.granted, 3, "the allocator, one block short");
        assert!(short.agrees(), "{}", short.render());
    }

    #[test]
    fn slack_is_bytes_held_and_not_touched() {
        let rows = vec![
            Waste {
                name: "scratch.p2048".to_owned(),
                bytes: 600 << 20,
                touched: Some(100 << 20),
                why: "sized for the widest op".to_owned(),
            },
            Waste {
                name: "engine:park-staging".to_owned(),
                bytes: 1 << 30,
                touched: Some(0),
                why: "parking is off".to_owned(),
            },
            Waste {
                name: "unknown".to_owned(),
                bytes: 1 << 30,
                touched: None,
                why: "not measured".to_owned(),
            },
        ];
        assert_eq!(rows[0].slack(), 500 << 20);
        assert_eq!(rows[1].slack(), 1 << 30);
        assert_eq!(rows[2].slack(), 0, "unmeasured is not the same as wasted");
        let out = render_waste(&rows, geom());
        assert!(out.contains("TOTAL slack 1524 MiB"), "{out}");
    }
}
