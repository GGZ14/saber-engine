//! FN117b: THE BIT-EXACT HOST-HAND SLICES (`--fn-hand <form>`; `all` = THE DEFAULT since the
//! FN117b flip; `none` = the pre-FN117b host hand, the control). LEDGER FN117 measured the host's hand between two served steps (b1 1.66 ms p50) and
//! designed the device-driven step loop in slices; this arm builds the three that move only WHEN
//! host work runs or HOW the two ranks meet -- the same kernels, the same inputs, the same bytes:
//!
//! * **S1 -- the drafter round's inputs in ONE async upload.** Today the round after a verify step
//!   drains the step's stream, then makes three blocking pageable H2Ds (the pass-1 embedding rows,
//!   the pass-1 step tables, the device chain's inputs) before its first launch. Under `s1` the
//!   host builds all three as today, writes them into one pinned staging and queues the three
//!   copies on the step's stream in one batch, no drain, no sync: the copies land before pass 1
//!   (stream order), every other copy of the round (the KV tier's step law) writes other memory,
//!   and the round's own end (the chain's ids read) drains them before the staging is reused.
//! * **S2 -- the next step's expert-pointer tables under the round.** The verify step's tier end
//!   leaves the placements the next step reads; today its tier begin builds and uploads all 48
//!   layers' tables (a blocking copy + sync) after the scheduler's turn. Under `s2` they are built
//!   while the drafter round runs on the card (the host otherwise idles in the chain's read) and
//!   queued behind the round; the next tier begin KEEPS them iff `ExpertTier::xptr_key` -- every
//!   input of the tables -- is unchanged (lend / reclaim / begin all still run where they run
//!   today: the lending order is untouched; a reclaim, an admission, a prefill pass, a valve move
//!   voids the build). Host law: `saber-mem` test `fn117b_hand_tables`.
//! * **S6 -- the pick exchange on the shared line.** The two ranks' pick frames cross through
//!   `saber_hip::shmline::ShmLine` (one shared page pair, the slot by the call's GLOBAL parity,
//!   data before the released header, a torn read refused) instead of the socket; a frame wider
//!   than the slot takes the socket (both ranks decide by its length, which they share).
//!
//! `check` adds each slice's oracle (diagnostic, it times nothing): S1 reads the three regions back
//! after its batch and compares them with the host's bytes; S2 rebuilds the tables at every kept
//! begin and compares them (host bytes and the device's) with the build; S6 runs the socket too and
//! compares the frames. Any difference is the step's error (fail loud). `--fn-hand-skew-us N` (a
//! race-check diagnostic) sleeps a pseudo-random 0..N us before every exchange on each rank.
//! End line `fnhand117b rank<r>:`.

use saber_hip::shmline::ShmLine;
use saber_hip::PinnedBuf;
use saber_mem::expert_tier::XptrKey;

/// `--fn-hand` when absent: `all` IS THE DEFAULT since FN117b (the orchestrator's pooled flip,
/// 2026-10-13: hand b1 1.687 -> 1.560 ms p50 / 2.696 -> 2.536 p99, bytes identical); `none` =
/// today's pre-FN117b host hand, the control.
pub const DEFAULT_FN_HAND: &str = "all";

/// The pick line's slot (bytes): the widest served pick frame (40 rows x (64 x 8 seam + 8 + 64
/// sha + 20 probes x 4)) is ~27 KiB; a wider one takes the socket.
pub const PICK_LINE_CAP: usize = 256 << 10;

/// Which slices run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Hand {
    pub s1: bool,
    pub s2: bool,
    pub s6: bool,
    pub check: bool,
}

impl Hand {
    /// Parse `--fn-hand`: `none`, or a comma list of `s1`, `s2`, `s6`, `all` (= s1,s2,s6),
    /// `check`.
    ///
    /// # Errors
    /// An unknown item, `none` beside another, or `check` alone.
    pub fn parse(s: Option<&str>) -> anyhow::Result<Hand> {
        let s = s.unwrap_or(DEFAULT_FN_HAND);
        let mut h = Hand::default();
        if s == "none" {
            return Ok(h);
        }
        for it in s.split(',') {
            match it {
                "s1" => h.s1 = true,
                "s2" => h.s2 = true,
                "s6" => h.s6 = true,
                "all" => {
                    h.s1 = true;
                    h.s2 = true;
                    h.s6 = true;
                }
                "check" => h.check = true,
                x => anyhow::bail!(
                    "--fn-hand {s}: unknown item {x:?} (none | a list of s1, s2, s6, all, check)"
                ),
            }
        }
        anyhow::ensure!(
            h.s1 || h.s2 || h.s6,
            "--fn-hand {s}: `check` needs a slice (s1, s2, s6 or all)"
        );
        Ok(h)
    }

    /// Any slice on.
    #[must_use]
    pub fn any(&self) -> bool {
        self.s1 || self.s2 || self.s6
    }

    /// The form's name (`none`, or the list).
    #[must_use]
    pub fn form(&self) -> String {
        let mut v = Vec::new();
        for (on, n) in [
            (self.s1, "s1"),
            (self.s2, "s2"),
            (self.s6, "s6"),
            (self.check, "check"),
        ] {
            if on {
                v.push(n);
            }
        }
        if v.is_empty() {
            "none".to_owned()
        } else {
            v.join(",")
        }
    }
}

/// S1's staging: the embedding rows, the pass-1 tables, the chain's inputs (each 4 KiB aligned).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S1Lay {
    pub emb_off: usize,
    pub emb_cap: usize,
    pub cs_off: usize,
    pub cs_cap: usize,
    pub inb_off: usize,
    pub inb_cap: usize,
    pub total: usize,
}

impl S1Lay {
    #[must_use]
    pub fn of(emb_cap: usize, cs_cap: usize, inb_cap: usize) -> S1Lay {
        let al = |x: usize| x.div_ceil(4096) * 4096;
        let emb_off = 0;
        let cs_off = al(emb_off + emb_cap);
        let inb_off = al(cs_off + cs_cap);
        S1Lay {
            emb_off,
            emb_cap,
            cs_off,
            cs_cap,
            inb_off,
            inb_cap,
            total: al(inb_off + inb_cap),
        }
    }

    /// Whether a round's three blobs fit their regions.
    #[must_use]
    pub fn fits(&self, emb: usize, cs: usize, inb: usize) -> bool {
        emb <= self.emb_cap && cs <= self.cs_cap && inb <= self.inb_cap
    }

    /// Lay the three blobs into `stg` at their regions (the bytes the three copies read).
    pub fn stage(&self, stg: &mut [u8], emb: &[u8], cs: &[u8], inb: &[u8]) {
        stg[self.emb_off..self.emb_off + emb.len()].copy_from_slice(emb);
        stg[self.cs_off..self.cs_off + cs.len()].copy_from_slice(cs);
        stg[self.inb_off..self.inb_off + inb.len()].copy_from_slice(inb);
    }
}

/// One rank's hand state and counters.
pub struct HandRt {
    pub hand: Hand,
    pub skew_us: u64,
    /// S1: the staging (None under `s1` off).
    pub s1: Option<(S1Lay, PinnedBuf)>,
    /// S2: the key of the build queued under the last round (None = none to keep).
    pub pre: Option<XptrKey>,
    /// S6: the shared pick line.
    pub line: Option<ShmLine>,
    pub s1_rounds: u64,
    pub s1_wide: u64,
    pub s1_checked: u64,
    pub s1_ms: f64,
    pub s2_built: u64,
    pub s2_skipped: u64,
    pub s2_kept: u64,
    pub s2_void: u64,
    pub s2_checked: u64,
    pub s2_ms: f64,
    pub s6_calls: u64,
    pub s6_wide: u64,
    pub s6_checked: u64,
    pub s6_ms: f64,
}

impl HandRt {
    #[must_use]
    pub fn new(hand: Hand, skew_us: u64) -> HandRt {
        HandRt {
            hand,
            skew_us,
            s1: None,
            pre: None,
            line: None,
            s1_rounds: 0,
            s1_wide: 0,
            s1_checked: 0,
            s1_ms: 0.0,
            s2_built: 0,
            s2_skipped: 0,
            s2_kept: 0,
            s2_void: 0,
            s2_checked: 0,
            s2_ms: 0.0,
            s6_calls: 0,
            s6_wide: 0,
            s6_checked: 0,
            s6_ms: 0.0,
        }
    }

    /// The race-check skew of call `seq` on `rank` (us; 0 without the diagnostic).
    #[must_use]
    pub fn skew(&self, rank: u32, seq: u64) -> u64 {
        if self.skew_us == 0 {
            return 0;
        }
        let mut s = seq
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(u64::from(rank).wrapping_mul(0xbf58_476d_1ce4_e5b9));
        s ^= s >> 31;
        s = s.wrapping_mul(0x94d0_49bb_1331_11eb);
        (s >> 17) % (self.skew_us + 1)
    }

    /// The end line.
    #[must_use]
    pub fn line(&self, rank: u32) -> String {
        let per = |ms: f64, n: u64| if n == 0 { 0.0 } else { ms / n as f64 };
        let lc = self.line.as_ref().map(|l| l.counts).unwrap_or_default();
        format!(
            "fnhand117b rank{rank}: form {} | S1 rounds {} (wide -> blocking {}) checked {} host {:.3} ms a round \
             | S2 built {} kept {} void {} skipped {} checked {} build {:.3} ms | S6 calls {} (wide -> socket {}) \
             checked {} late {} ack-waits {} {:.3} ms a call{}",
            self.hand.form(),
            self.s1_rounds,
            self.s1_wide,
            self.s1_checked,
            per(self.s1_ms, self.s1_rounds),
            self.s2_built,
            self.s2_kept,
            self.s2_void,
            self.s2_skipped,
            self.s2_checked,
            per(self.s2_ms, self.s2_built),
            self.s6_calls,
            self.s6_wide,
            self.s6_checked,
            lc.late,
            lc.ack_waits,
            per(self.s6_ms, self.s6_calls),
            if self.skew_us > 0 {
                format!(" | SKEW 0..{} us (race check)", self.skew_us)
            } else {
                String::new()
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_form_parses_and_refuses_by_name() {
        assert_eq!(
            Hand::parse(None).unwrap(),
            Hand::parse(Some("all")).unwrap()
        );
        assert!(!Hand::parse(Some("none")).unwrap().any());
        let a = Hand::parse(Some("all")).unwrap();
        assert!(a.s1 && a.s2 && a.s6 && !a.check);
        assert_eq!(a.form(), "s1,s2,s6");
        let c = Hand::parse(Some("s6,check")).unwrap();
        assert!(c.s6 && c.check && !c.s1);
        assert!(Hand::parse(Some("check")).is_err());
        assert!(Hand::parse(Some("s3"))
            .unwrap_err()
            .to_string()
            .contains("s3"));
        assert!(Hand::parse(Some("none,s1")).is_err());
    }

    #[test]
    fn the_s1_staging_lays_each_blob_at_its_region() {
        let l = S1Lay::of(5000, 100, 7000);
        assert_eq!(
            (l.emb_off, l.cs_off, l.inb_off, l.total),
            (0, 8192, 12288, 20480)
        );
        let mut stg = vec![0u8; l.total];
        let (e, c, i) = (vec![1u8; 5000], vec![2u8; 100], vec![3u8; 7000]);
        assert!(l.fits(5000, 100, 7000) && !l.fits(5001, 100, 7000));
        l.stage(&mut stg, &e, &c, &i);
        assert_eq!(&stg[l.emb_off..l.emb_off + 5000], &e[..]);
        assert_eq!(&stg[l.cs_off..l.cs_off + 100], &c[..]);
        assert_eq!(&stg[l.inb_off..l.inb_off + 7000], &i[..]);
    }

    #[test]
    fn the_skew_is_per_rank_and_bounded() {
        let h = HandRt::new(Hand::parse(Some("s6")).unwrap(), 200);
        let a: Vec<u64> = (1..100).map(|s| h.skew(0, s)).collect();
        let b: Vec<u64> = (1..100).map(|s| h.skew(1, s)).collect();
        assert!(a.iter().chain(&b).all(|&x| x <= 200));
        assert_ne!(a, b);
        assert_eq!(HandRt::new(Hand::default(), 0).skew(0, 7), 0);
    }
}
