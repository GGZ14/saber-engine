//! P27-14: the 27B's three BIT-EXACT host levers (LEDGER P27-13 names them, P27-14 builds them).
//!
//! `--engine-host-levers picks,table,socket` (serve) / `--host-levers` (engine, worker,
//! `run_engine.sh`), each selectable, `none` = every lever off (the control). Each one changes
//! WHEN or IN WHAT FRAME the two ranks move bytes they already move, never which bytes:
//!
//! * **`picks`** -- a SAMPLED decode step's follower half crosses in ONE frame (its picks AND its
//!   seam shard, `engine.pickseam`) instead of two (`engine.picks` then `engine.seam.out`), and the
//!   leader reads its own seam shard BEFORE it waits for the peer's frame, not after. P27-13's 1.8
//!   ms "swap picks" is the LEADER waiting for rank 1's step to finish (rank 1's card span is 1.8
//!   ms longer under sampling and equal under greedy): this lever removes the host's share of it,
//!   not the card's (see LEDGER P27-14).
//! * **`table`** -- a page-crossing step's block-table frame carries the table the step BUILT
//!   (run-length coded: a slot's row is its pages then a fill of its last page) and the offsets
//!   it was written at, not the whole 20 MiB region; the follower writes the same bytes at the
//!   same offsets. The first paged frame of a link is whole (the induction base).
//! * **`socket`** -- a deferred proposal's `Command::Propose` carries its first drafter frame
//!   (one round trip, not two), and a single-proposal step sends no terminator: the block digest
//!   the terminator carried rides the NEXT `Command::Propose` (or `Stop`) instead.

use std::sync::atomic::{AtomicU64, Ordering};

/// Which of the three levers a run turns on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostLevers {
    /// The sampled step's one pick-and-seam frame + the leader's seam read before the wait.
    pub picks: bool,
    /// The compact block-table frame.
    pub table: bool,
    /// The proposal's frame inside `Command::Propose` + the folded terminator.
    pub socket: bool,
    // ---- CMP27N: the b1 host trims (in the DEFAULT since CMP27O; never in `all`) ----
    /// `seamcache`: the seam's inputs upload is skipped when the bytes the device already holds
    /// are the bytes it would write (both ranks; a sampled stream at one temperature uploads once).
    pub seamcache: bool,
    /// `seamread`: the seam's read-back copies are POSTED behind the step's submit (no sync), so
    /// after the step's wait the bytes are already in the pinned staging (one sync round trip
    /// fewer a sampled step, both ranks).
    pub seamread: bool,
    /// `seedasync`: the drafter's table seed and its candidate re-seed post their copies with no
    /// closing stream sync (stream order carries them to the launches; the second seed stages
    /// past the first's slice, and the proposal's read-back sync drains both).
    pub seedasync: bool,
}

/// THE DEFAULT since CMP27O (2026-10-28; P27-14's `table` + `socket`, plus CMP27N's three b1 host
/// trims `seamcache` + `seamread` + `seedasync` = `trim`, flipped on the stacked hold: bytes
/// identical, served b1 update p50 -0.10 ms in its own arm). `picks` stays an ARM (bytes identical,
/// no measured gain: the wait it targets is rank 1's card, not the host); `table,socket` = the
/// pre-CMP27O default (the trims' control and rollback); `none` = the control.
pub const DEFAULT_HOST_LEVERS: &str = "table,socket,seamcache,seamread,seedasync";

impl HostLevers {
    /// Every lever on.
    pub const ALL: HostLevers = HostLevers {
        picks: true,
        table: true,
        socket: true,
        seamcache: false,
        seamread: false,
        seedasync: false,
    };

    /// Parse `none`, `all` or a comma list of `picks`, `table`, `socket`.
    ///
    /// # Errors
    /// An unknown lever name, or an empty list.
    pub fn parse(s: &str) -> Result<HostLevers, String> {
        let s = s.trim();
        if s == "none" {
            return Ok(HostLevers::default());
        }
        if s == "all" {
            return Ok(HostLevers::ALL);
        }
        let mut out = HostLevers::default();
        let mut any = false;
        for part in s.split(',') {
            match part.trim() {
                "picks" => out.picks = true,
                "table" => out.table = true,
                "socket" => out.socket = true,
                "seamcache" => out.seamcache = true,
                "seamread" => out.seamread = true,
                "seedasync" => out.seedasync = true,
                // CMP27N: every b1 host trim at once.
                "trim" => {
                    out.seamcache = true;
                    out.seamread = true;
                    out.seedasync = true;
                }
                "" => continue,
                o => {
                    return Err(format!(
                        "unknown host lever `{o}`: the levers are picks, table, socket, seamcache, \
                         seamread, seedasync, trim (or none / all)"
                    ))
                }
            }
            any = true;
        }
        if !any {
            return Err("an empty host-lever list: say `none`".to_owned());
        }
        Ok(out)
    }

    /// The canonical spelling, forwarded to rank 1 by value.
    #[must_use]
    pub fn as_arg(self) -> String {
        let mut v = Vec::new();
        if self.picks {
            v.push("picks");
        }
        if self.table {
            v.push("table");
        }
        if self.socket {
            v.push("socket");
        }
        if self.seamcache {
            v.push("seamcache");
        }
        if self.seamread {
            v.push("seamread");
        }
        if self.seedasync {
            v.push("seedasync");
        }
        if v.is_empty() {
            "none".to_owned()
        } else {
            v.join(",")
        }
    }
}

/// The which-path counters (rule 5), one set a rank.
#[derive(Debug, Default)]
pub struct HostLeverCounters {
    /// `picks`: sampled decode steps whose follower half crossed in one `engine.pickseam` frame.
    pub pickseam_frames: AtomicU64,
    /// `picks` (diagnostic, leader): sampled rows whose PEER seam shard held a tied value at its
    /// K-th slot (the seam kernel's ordered tie scan is the only data-dependent path in it).
    pub peer_tie_rows: AtomicU64,
    /// ... and the deepest tied id's position in the shard, summed in parts per thousand (a proxy
    /// for how far that ordered scan walked the row).
    pub peer_tie_depth_permille: AtomicU64,
    /// The same two for the leader's own shard.
    pub own_tie_rows: AtomicU64,
    /// The leader's own shard's depth sum.
    pub own_tie_depth_permille: AtomicU64,
    /// Sampled rows the tie proxy looked at (each shard).
    pub tie_rows_seen: AtomicU64,
    /// `table`: paged frames sent compact.
    pub table_compact: AtomicU64,
    /// `table`: the compact frames' bytes.
    pub table_compact_bytes: AtomicU64,
    /// `table`: paged frames sent whole (the first of a link, no recipe, or no gain).
    pub table_whole: AtomicU64,
    /// `table`: the whole frames' bytes.
    pub table_whole_bytes: AtomicU64,
    /// `socket`: proposal frames that rode their `Command::Propose`.
    pub propose_inline: AtomicU64,
    /// `socket`: terminators not sent (a single proposal's digest carried instead).
    pub terminators_folded: AtomicU64,
    /// `socket`: carried digests checked against this rank's own (the follower counts).
    pub carry_checked: AtomicU64,
    /// CMP27N `seamcache`: seam input uploads skipped (the device already held the bytes).
    pub seam_upload_skipped: AtomicU64,
    /// CMP27N `seamread`: seam reads served from copies posted behind the step.
    pub seam_read_posted: AtomicU64,
    /// CMP27N `seedasync`: drafter seeds posted without a stream sync.
    pub seed_unsynced: AtomicU64,
}

impl HostLeverCounters {
    /// One line for the end of a run.
    #[must_use]
    pub fn line(&self, levers: HostLevers) -> String {
        let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
        let seen = g(&self.tie_rows_seen).max(1);
        let depth = |rows: u64, sum: u64| {
            if rows == 0 {
                0.0
            } else {
                sum as f64 / rows as f64 / 10.0
            }
        };
        format!(
            "levers {}; picks: pickseam frames {}, tie rows peer {} ({:.1} % of {}, mean depth \
             {:.1} %) own {} ({:.1} %, mean depth {:.1} %); table: compact {} ({} B), whole {} \
             ({} B); socket: inline proposals {}, terminators folded {}, carried digests checked \
             {}; trim: seam uploads skipped {}, seam reads posted {}, seeds unsynced {}",
            levers.as_arg(),
            g(&self.pickseam_frames),
            g(&self.peer_tie_rows),
            100.0 * g(&self.peer_tie_rows) as f64 / seen as f64,
            g(&self.tie_rows_seen),
            depth(g(&self.peer_tie_rows), g(&self.peer_tie_depth_permille)),
            g(&self.own_tie_rows),
            100.0 * g(&self.own_tie_rows) as f64 / seen as f64,
            depth(g(&self.own_tie_rows), g(&self.own_tie_depth_permille)),
            g(&self.table_compact),
            g(&self.table_compact_bytes),
            g(&self.table_whole),
            g(&self.table_whole_bytes),
            g(&self.propose_inline),
            g(&self.terminators_folded),
            g(&self.carry_checked),
            g(&self.seam_upload_skipped),
            g(&self.seam_read_posted),
            g(&self.seed_unsynced),
        )
    }
}

/// The tie proxy for one seam row: `values` / `ids` are the shard's K slots (value descending,
/// id ascending on a tie). Returns `Some(depth in parts per thousand of the shard)` when the K-th
/// value is shared by the slot before it (the kernel then MAY have taken its ordered scan, which
/// walks the shard in id order up to the last tied id it keeps), `None` otherwise.
#[must_use]
pub fn tie_depth(values: &[f32], ids: &[i32], cols: u32) -> Option<u64> {
    let k = values.len().min(ids.len());
    if k < 2 || cols == 0 {
        return None;
    }
    let last = values[k - 1];
    if !last.is_finite() || values[k - 2].to_bits() != last.to_bits() {
        return None;
    }
    let deepest = values[..k]
        .iter()
        .zip(&ids[..k])
        .filter(|(v, _)| v.to_bits() == last.to_bits())
        .map(|(_, i)| *i)
        .max()?;
    let deepest = u64::try_from(deepest).ok()?;
    Some(deepest * 1000 / u64::from(cols))
}

// ------------------------------------------------------------------------------------------- //
// The compact block-table frame (`table`)                                                       //
// ------------------------------------------------------------------------------------------- //

/// What a paged staging wrote into the blob's paged region: each distinct table's bytes, and
/// every (blob offset, table) it was copied to. Recorded by the staging loop itself, which IS the
/// set of writes (the same law as the step frame's runs).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PagedRecipe {
    /// The distinct tables, in first-written order.
    pub tables: Vec<Vec<u8>>,
    /// `(absolute blob offset, index into tables)`, in write order.
    pub spans: Vec<(u32, u32)>,
}

/// The compact frame's first word ("PGR1").
pub const PAGED_MAGIC: u32 = 0x5047_5231;

const OP_LIT: u8 = 0;
const OP_FILL: u8 = 1;
const OP_RAMP: u8 = 2;
/// A run shorter than this goes out as literals.
const MIN_RUN: usize = 4;

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn len_u32(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// Run-length code a table of little-endian u32 words: literal runs, fills (one word repeated)
/// and ramps (`w, w + 1, ...`, wrapping). `None` when the table is not whole words.
#[must_use]
pub fn rle_encode(table: &[u8]) -> Option<Vec<u8>> {
    if table.len() % 4 != 0 {
        return None;
    }
    let w: Vec<u32> = table
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut out = Vec::new();
    let flush_lit = |out: &mut Vec<u8>, from: usize, to: usize| {
        if to > from {
            out.push(OP_LIT);
            put_u32(out, len_u32(to - from));
            for v in &w[from..to] {
                put_u32(out, *v);
            }
        }
    };
    let mut lit_from = 0usize;
    let mut i = 0usize;
    while i < w.len() {
        let mut fill = 1usize;
        while i + fill < w.len() && w[i + fill] == w[i] {
            fill += 1;
        }
        let mut ramp = 1usize;
        while i + ramp < w.len() && w[i + ramp] == w[i + ramp - 1].wrapping_add(1) {
            ramp += 1;
        }
        let (op, n) = if fill >= ramp {
            (OP_FILL, fill)
        } else {
            (OP_RAMP, ramp)
        };
        if n >= MIN_RUN {
            flush_lit(&mut out, lit_from, i);
            out.push(op);
            put_u32(&mut out, len_u32(n));
            put_u32(&mut out, w[i]);
            i += n;
            lit_from = i;
        } else {
            i += 1;
        }
    }
    flush_lit(&mut out, lit_from, w.len());
    Some(out)
}

fn take_u32(s: &mut &[u8], what: &str) -> Result<u32, String> {
    if s.len() < 4 {
        return Err(format!("a truncated {what}"));
    }
    let v = u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
    *s = &s[4..];
    Ok(v)
}

/// Decode [`rle_encode`]'s stream into exactly `len` bytes.
///
/// # Errors
/// A malformed stream, or one that does not decode to `len` bytes.
pub fn rle_decode(mut s: &[u8], len: usize) -> Result<Vec<u8>, String> {
    const WHAT: &str = "run-length stream";
    let mut out = Vec::with_capacity(len);
    while !s.is_empty() {
        let op = s[0];
        s = &s[1..];
        let n = take_u32(&mut s, WHAT)? as usize;
        if out.len() + n * 4 > len {
            return Err(format!(
                "a run-length stream that decodes past its {len} byte(s)"
            ));
        }
        match op {
            OP_LIT => {
                for _ in 0..n {
                    let v = take_u32(&mut s, WHAT)?;
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            OP_FILL => {
                let v = take_u32(&mut s, WHAT)?;
                for _ in 0..n {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            OP_RAMP => {
                let v = take_u32(&mut s, WHAT)?;
                for k in 0..n {
                    out.extend_from_slice(&v.wrapping_add(len_u32(k)).to_le_bytes());
                }
            }
            o => return Err(format!("a run-length op {o}; this build speaks 0..=2")),
        }
    }
    if out.len() != len {
        return Err(format!(
            "a run-length stream of {} byte(s) for a {len}-byte table",
            out.len()
        ));
    }
    Ok(out)
}

/// Encode a recipe as the compact paged frame, or `None` when it is not smaller than `whole`
/// bytes (then the whole region goes, as without the lever).
#[must_use]
pub fn encode_paged(recipe: &PagedRecipe, whole: usize) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    put_u32(&mut out, PAGED_MAGIC);
    put_u32(&mut out, len_u32(recipe.tables.len()));
    for t in &recipe.tables {
        let enc = rle_encode(t)?;
        put_u32(&mut out, len_u32(t.len()));
        put_u32(&mut out, len_u32(enc.len()));
        out.extend_from_slice(&enc);
        if out.len() >= whole {
            return None;
        }
    }
    put_u32(&mut out, len_u32(recipe.spans.len()));
    for (off, t) in &recipe.spans {
        put_u32(&mut out, *off);
        put_u32(&mut out, *t);
    }
    (out.len() < whole).then_some(out)
}

/// Whether a paged frame is the compact form (a whole frame is exactly the region's width, and a
/// compact one is always shorter).
#[must_use]
pub fn is_compact(frame: &[u8], whole: usize) -> bool {
    frame.len() < whole && frame.len() >= 4 && frame[..4] == PAGED_MAGIC.to_le_bytes()
}

/// Apply a compact frame to a blob: every span's table at its offset, bounds-checked against the
/// paged region `[paged_at, end)`.
///
/// # Errors
/// A malformed frame, or a span outside the region.
pub fn apply_compact(
    frame: &[u8],
    host: &mut [u8],
    paged_at: usize,
    end: usize,
) -> Result<(), String> {
    const WHAT: &str = "compact block-table frame";
    let mut s = frame;
    if take_u32(&mut s, WHAT)? != PAGED_MAGIC {
        return Err("a compact block-table frame without its magic".to_owned());
    }
    let n = take_u32(&mut s, WHAT)? as usize;
    let mut tables = Vec::with_capacity(n);
    for _ in 0..n {
        let len = take_u32(&mut s, WHAT)? as usize;
        let enc = take_u32(&mut s, WHAT)? as usize;
        if s.len() < enc {
            return Err("a compact block-table frame shorter than its tables".to_owned());
        }
        tables.push(rle_decode(&s[..enc], len)?);
        s = &s[enc..];
    }
    let spans = take_u32(&mut s, WHAT)? as usize;
    for _ in 0..spans {
        let off = take_u32(&mut s, WHAT)? as usize;
        let t = take_u32(&mut s, WHAT)? as usize;
        let table = tables
            .get(t)
            .ok_or_else(|| format!("a span names table {t} of {}", tables.len()))?;
        if off < paged_at || off + table.len() > end || end > host.len() {
            return Err(format!(
                "a span at {off}..{} outside the paged region {paged_at}..{end}",
                off + table.len()
            ));
        }
        host[off..off + table.len()].copy_from_slice(table);
    }
    if !s.is_empty() {
        return Err(format!(
            "{} byte(s) past the compact block-table frame's spans",
            s.len()
        ));
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------- //
// The single-proposal frame (`socket`)                                                          //
// ------------------------------------------------------------------------------------------- //

/// `more` = 2 in a proposal frame's head: a proposal follows and NO terminator does (its digest
/// is carried to the next `Command::Propose`). `DraftFrameHead::decode` reads any non-zero as
/// `more`, so a build without the lever would still run it -- and then wait for a terminator,
/// which is why both ranks take the lever by value.
pub const SOLO_MORE: u32 = 2;

/// Mark an encoded proposal frame as the step's only one.
pub fn mark_solo(frame: &mut [u8]) {
    if frame.len() >= 8 {
        frame[4..8].copy_from_slice(&SOLO_MORE.to_le_bytes());
    }
}

/// Take the solo mark off (back to `more` = 1, the frame the leader ran).
pub fn unmark_solo(frame: &mut [u8]) {
    if is_solo(frame) {
        frame[4..8].copy_from_slice(&1u32.to_le_bytes());
    }
}

/// Whether an encoded proposal frame is marked solo.
#[must_use]
pub fn is_solo(frame: &[u8]) -> bool {
    frame.len() >= 8 && frame[4..8] == SOLO_MORE.to_le_bytes()
}

#[cfg(test)]
mod tests {
    #[test]
    fn cmp27n_trim_parts_parse_round_trip_and_stay_out_of_all() {
        use super::HostLevers;
        let t = HostLevers::parse("table,socket,trim").expect("trim");
        assert!(t.seamcache && t.seamread && t.seedasync && t.table && t.socket && !t.picks);
        assert_eq!(t.as_arg(), "table,socket,seamcache,seamread,seedasync");
        assert_eq!(HostLevers::parse(&t.as_arg()), Ok(t));
        let one = HostLevers::parse("seamread").expect("one part");
        assert!(one.seamread && !one.seamcache && !one.seedasync);
        let all = HostLevers::parse("all").expect("all");
        assert!(!all.seamcache && !all.seamread && !all.seedasync);
        assert!(HostLevers::parse("seamcash").is_err());
    }

    use super::*;

    #[test]
    fn the_lever_list_parses_and_round_trips_and_refuses_unknown_names() {
        assert_eq!(HostLevers::parse("none"), Ok(HostLevers::default()));
        assert_eq!(HostLevers::parse("all"), Ok(HostLevers::ALL));
        assert_eq!(HostLevers::parse("picks,table,socket"), Ok(HostLevers::ALL));
        let t = HostLevers::parse("table").expect("parses");
        assert!(t.table && !t.picks && !t.socket);
        assert_eq!(t.as_arg(), "table");
        assert_eq!(HostLevers::ALL.as_arg(), "picks,table,socket");
        assert_eq!(HostLevers::default().as_arg(), "none");
        let d = HostLevers::parse(DEFAULT_HOST_LEVERS).expect("the default parses");
        assert!(
            d.table && d.socket && !d.picks,
            "P27-14's flip: table + socket"
        );
        assert!(
            d.seamcache && d.seamread && d.seedasync,
            "CMP27O's flip: the three b1 host trims"
        );
        assert_eq!(d.as_arg(), DEFAULT_HOST_LEVERS, "the default round-trips");
        let pre = HostLevers::parse("table,socket").expect("the rollback parses");
        assert!(!pre.seamcache && !pre.seamread && !pre.seedasync && pre.table && pre.socket);
        assert!(
            HostLevers::parse("picks,tree").is_err(),
            "an unknown lever is refused"
        );
        assert!(HostLevers::parse(",").is_err());
    }

    fn table_like(slots: usize, max_blocks: usize, seed: u32) -> Vec<u8> {
        // A block table: each slot's pages (a ramp with a jump) then its last page repeated.
        let mut out = Vec::new();
        for s in 0..slots {
            let n = 3 + (s * 7 + seed as usize) % 40;
            let mut last = 0u32;
            for k in 0..max_blocks {
                if k < n {
                    let base = 1000 * (s as u32 + 1) + seed;
                    last = if k == n / 2 {
                        base + 77 + k as u32
                    } else {
                        base + k as u32
                    };
                }
                out.extend_from_slice(&last.to_le_bytes());
            }
        }
        out
    }

    #[test]
    fn the_run_length_code_is_exact_and_small_on_a_block_table() {
        let t = table_like(19, 16_384, 5);
        let enc = rle_encode(&t).expect("whole words");
        assert!(enc.len() * 100 < t.len(), "{} vs {}", enc.len(), t.len());
        assert_eq!(rle_decode(&enc, t.len()).expect("decodes"), t);
        // Arbitrary words too, including wrap-around ramps.
        let mut r = Vec::new();
        let mut x = 0x1234_5678u32;
        for i in 0..5000u32 {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = if i % 97 < 30 {
                (u32::MAX - 3).wrapping_add(i % 97)
            } else {
                x >> (i % 7)
            };
            r.extend_from_slice(&v.to_le_bytes());
        }
        let enc = rle_encode(&r).expect("whole words");
        assert_eq!(rle_decode(&enc, r.len()).expect("decodes"), r);
        // THE CONTROL: a wrong length is refused, not padded.
        assert!(rle_decode(&enc, r.len() + 4).is_err());
        assert!(rle_encode(&[1, 2, 3]).is_none(), "not whole words");
    }

    /// The law the lever rests on: the follower's region after a compact frame equals the
    /// leader's after the staging that recorded the recipe -- given the two were equal before.
    #[test]
    fn a_compact_frame_leaves_the_follower_region_equal_to_the_leaders() {
        let paged_at = 4096usize;
        let region = 8 * 20_000usize;
        let mut leader = vec![0u8; paged_at + region];
        // Stale content both ranks share (the induction base: a whole frame sent once).
        for (i, b) in leader[paged_at..].iter_mut().enumerate() {
            *b = (i * 31 % 251) as u8;
        }
        let mut follower = leader.clone();
        let t0 = table_like(4, 2_000, 1);
        let t1 = table_like(4, 2_000, 9);
        let stride = t0.len() + 256;
        let spans = vec![
            (paged_at as u32, 0u32),
            ((paged_at + stride) as u32, 0),
            ((paged_at + 2 * stride) as u32, 1),
        ];
        let recipe = PagedRecipe {
            tables: vec![t0, t1],
            spans,
        };
        for (off, t) in &recipe.spans {
            let tb = &recipe.tables[*t as usize];
            leader[*off as usize..*off as usize + tb.len()].copy_from_slice(tb);
        }
        let frame = encode_paged(&recipe, region).expect("smaller than the region");
        assert!(is_compact(&frame, region));
        assert!(
            !is_compact(&leader[paged_at..], region),
            "a whole frame is not compact"
        );
        apply_compact(&frame, &mut follower, paged_at, paged_at + region).expect("applies");
        assert_eq!(follower, leader);
        // THE CONTROL: a span outside the region is refused by name.
        let bad = PagedRecipe {
            tables: vec![vec![0u8; 64]],
            spans: vec![((paged_at - 8) as u32, 0)],
        };
        let f = encode_paged(&bad, region).expect("small");
        assert!(apply_compact(&f, &mut follower, paged_at, paged_at + region).is_err());
        // And a recipe that does not compress below the whole region is sent whole.
        let mut noisy = Vec::new();
        let mut x = 7u32;
        for _ in 0..1000 {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            noisy.extend_from_slice(&x.to_le_bytes());
        }
        let big = PagedRecipe {
            tables: vec![noisy],
            spans: vec![(paged_at as u32, 0)],
        };
        assert!(encode_paged(&big, 4000).is_none());
    }

    #[test]
    fn the_solo_mark_and_the_tie_proxy() {
        let mut f = vec![0u8; 32];
        f[4..8].copy_from_slice(&1u32.to_le_bytes());
        assert!(!is_solo(&f));
        mark_solo(&mut f);
        assert!(is_solo(&f));
        unmark_solo(&mut f);
        assert!(!is_solo(&f));
        assert_eq!(f[4..8], 1u32.to_le_bytes(), "the leader's own `more`");
        // Tie proxy: the K-th value shared with the slot before it, deepest tied id 900 of 1000.
        let v = [5.0f32, 4.0, 3.0, 3.0];
        let i = [10i32, 20, 900, 400];
        assert_eq!(tie_depth(&v, &i, 1000), Some(900));
        // THE CONTROL: a distinct K-th value is no tie.
        assert_eq!(tie_depth(&[5.0, 4.0, 3.5, 3.0], &i, 1000), None);
    }
}
