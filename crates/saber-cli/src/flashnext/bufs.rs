//! FN100: THE ROW-SCALED BUFFERS AS ONE ALLOCATION and their tails lent to the expert tier (host
//! law, no HIP).
//!
//! `fn.buffers` (the exact ledger's class, [`super::run::exact_runtime`]) holds the boot's widest
//! chunk (`--fn-max-chunk`, 8,192 rows at the served boot) in four row-indexed buffers: the
//! embedding staging rows (`emb`, H bf16 a row), the PLE staging rows (`rows`, H bf16 a row), the
//! route-id log (`idlog`, LAYERS x 40 B a row, layer-packed by the pass's own rows: layer `l` at
//! `l x m x 40`) and the MTP drafter's target-stream input (`hid`, W bf16 a row; MTP boots only).
//! Every pass reads and writes them from row 0: a pass of `m` rows touches each buffer's first
//! `m` rows only. So FN100 lays them out as ONE allocation in a fixed order, each buffer's HEAD
//! (its first `keep` rows: every decode / verify form, the drafter's passes, a <= `keep`-row
//! chunk) at its base, its TAIL (rows `keep..max_chunk`) after it; the tails are carved into whole
//! expert strides and LENT at decode. No address moves: every buffer keeps one base for the
//! boot's life (the decode graphs bake the bases; FN98's one exec per form), and a pass of more
//! than `keep` rows reclaims the tails BEFORE its staging writes them (records dropped to their
//! RAM shadows, no copy).
//!
//! The other members of the class are not lent: the rope table (position-indexed: every decode
//! step reads the row of its position, up to the boot's context), the reduce mailbox (IPC-shared,
//! the peer writes it, the producer-rr channel sits at 3/4 of each slot: its offsets are the
//! widest chunk's), the drafter's `prev` / `stash` rows and the 8 MiB slack.

use std::rc::Rc;

use saber_hip::{DeviceAddr, DevicePtr};

/// Each buffer's base inside the allocation is aligned to this (and each tail's first stride).
pub const ALIGN: u64 = 64 << 10;

/// One buffer of the layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Part {
    pub name: &'static str,
    /// Bytes a row (the idlog: LAYERS x 40, layer-packed by the pass's rows).
    pub row_bytes: u64,
    /// The rows it holds (the boot's widest chunk).
    pub rows: u64,
    /// Its base, bytes from the allocation's.
    pub off: u64,
    /// Its head's bytes (`keep` rows, ALIGN-rounded, at most the buffer).
    pub head: u64,
}

impl Part {
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.rows * self.row_bytes
    }
    /// The tail: (offset of its first lendable byte, whole strides in it).
    #[must_use]
    pub fn tail(&self, stride: u64) -> (u64, u64) {
        let lo = self.off + self.head;
        let end = self.off + self.bytes();
        (lo, end.saturating_sub(lo) / stride)
    }
}

/// The one allocation's layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub parts: Vec<Part>,
    pub total: u64,
    /// The rows every head holds.
    pub keep: u32,
}

/// The four row-scaled buffers (`hid` only with the MTP drafter) for a boot of `max_chunk` rows,
/// heads of `keep` rows. `h` / `w` = the hidden / HC widths, `layers` the decoder layers.
#[must_use]
pub fn layout(max_chunk: u32, keep: u32, mtp: bool, h: u64, w: u64, layers: u64) -> Layout {
    let mut parts = Vec::new();
    let mut off = 0u64;
    let rows = u64::from(max_chunk);
    let mut add = |name: &'static str, row_bytes: u64| {
        let keep_b = (u64::from(keep).min(rows) * row_bytes).div_ceil(ALIGN) * ALIGN;
        let p = Part {
            name,
            row_bytes,
            rows,
            off,
            head: keep_b.min(rows * row_bytes),
        };
        off = (off + p.bytes()).div_ceil(ALIGN) * ALIGN;
        parts.push(p);
    };
    // the biggest tail first (fixed order: the census's)
    if mtp {
        add("hid", w * 2);
    }
    add("emb", h * 2);
    add("rows", h * 2);
    add("idlog", layers * 40);
    Layout {
        parts,
        total: off,
        keep,
    }
}

impl Layout {
    #[must_use]
    pub fn part(&self, name: &str) -> Option<&Part> {
        self.parts.iter().find(|p| p.name == name)
    }
    /// Every tail's lendable strides: (part, offset of the first, count).
    #[must_use]
    pub fn tails(&self, stride: u64) -> Vec<(&'static str, u64, u64)> {
        self.parts
            .iter()
            .map(|p| {
                let (lo, n) = p.tail(stride);
                (p.name, lo, n)
            })
            .filter(|x| x.2 > 0)
            .collect()
    }
    #[must_use]
    pub fn slots(&self, stride: u64) -> u64 {
        self.tails(stride).iter().map(|x| x.2).sum()
    }
    /// The bytes the four separate allocations held before FN100 (the census's referee).
    #[must_use]
    pub fn separate_bytes(&self) -> u64 {
        self.parts.iter().map(Part::bytes).sum()
    }
}

/// A buffer of the one allocation: its owner (the lend's region owner too), its offset, its length.
#[derive(Clone)]
pub struct Sub {
    pub own: Rc<DevicePtr>,
    pub off: u64,
    pub len: u64,
    a: DeviceAddr,
}

impl Sub {
    /// # Errors
    /// A part outside the allocation.
    pub fn new(own: &Rc<DevicePtr>, p: &Part) -> anyhow::Result<Sub> {
        anyhow::ensure!(
            p.off + p.bytes() <= own.len(),
            "FN100: {} at {} + {} B past the {} B allocation",
            p.name,
            p.off,
            p.bytes(),
            own.len()
        );
        let a = own
            .addr()
            .offset(p.off)
            .ok_or_else(|| anyhow::anyhow!("FN100: {} offset overflows", p.name))?;
        Ok(Sub {
            own: Rc::clone(own),
            off: p.off,
            len: p.bytes(),
            a,
        })
    }
    /// The buffer's base (what `DevicePtr::addr` gave the separate allocation).
    #[must_use]
    pub fn addr(&self) -> DeviceAddr {
        self.a
    }
}

/// One member of the `fn.buffers` class (the census).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Member {
    pub name: &'static str,
    pub bytes: u64,
    /// The rows it is sized for (0 = not row-scaled).
    pub rows: u64,
    pub writers: &'static str,
    pub baked: &'static str,
    /// Bytes past the heads (0 = not lendable).
    pub tail: u64,
    pub slots: u64,
}

/// The census of the class at a boot of `max_chunk` rows (`rope_b` = the rope table's bytes,
/// `ar_b` the reduce mailbox's, `mtp_small` the drafter's prev + stash), the heads of `keep` rows.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn census(
    max_chunk: u32,
    keep: u32,
    mtp: bool,
    h: u64,
    w: u64,
    layers: u64,
    rope_b: u64,
    ar_b: u64,
    mtp_small: u64,
    stride: u64,
) -> Vec<Member> {
    let lay = layout(max_chunk, keep, mtp, h, w, layers);
    let mut v: Vec<Member> = lay
        .parts
        .iter()
        .map(|p| {
            let (_, n) = p.tail(stride);
            let (writers, baked) = match p.name {
                "hid" => (
                    "every target pass (its pre-final stream, one D2D: decode / verify forms, \
                     <= keep chunks, wide chunks); the drafter's prefill-form passes read it",
                    "every decode / verify exec (the D2D node)",
                ),
                "emb" => (
                    "every pass's embedding H2D (decode forms, chunks, the drafter's passes, the \
                     device chain's tail)",
                    "every decode / verify exec + the drafter's decode-form passes + the chain",
                ),
                "rows" => (
                    "the PLE staging H2D (chunks; decode forms under the PLE overlap)",
                    "every decode / verify exec (the PLE layer reads it)",
                ),
                _ => (
                    "one D2D a layer of a complete pass (the route ids)",
                    "every decode / verify exec (48 D2D nodes)",
                ),
            };
            Member {
                name: p.name,
                bytes: p.bytes(),
                rows: p.rows,
                writers,
                baked,
                tail: p.bytes() - p.head,
                slots: n,
            }
        })
        .collect();
    v.push(Member {
        name: "rope",
        bytes: rope_b,
        rows: 0,
        writers: "boot only (the cos/sin table)",
        baked: "every exec (read at each row's position)",
        tail: 0,
        slots: 0,
    });
    v.push(Member {
        name: "ar mailbox",
        bytes: ar_b,
        rows: u64::from(max_chunk),
        writers: "every reduce (both ranks, IPC)",
        baked: "every exec (offsets keyed by the widest chunk's slot)",
        tail: 0,
        slots: 0,
    });
    if mtp {
        v.push(Member {
            name: "mtp prev+stash",
            bytes: mtp_small,
            rows: 0,
            writers: "the drafter's chain / a chunk boundary",
            baked: "the drafter's decode-form passes",
            tail: 0,
            slots: 0,
        });
    }
    v.push(Member {
        name: "slack",
        bytes: 8 << 20,
        rows: 0,
        writers: "-",
        baked: "-",
        tail: 0,
        slots: 0,
    });
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    const STRIDE: u64 = 2_617_344;

    #[test]
    fn fn100_the_layout_keeps_every_head_at_its_base_and_lends_whole_tails() {
        let l = layout(8192, 256, true, 2560, 10240, 48);
        assert_eq!(
            l.parts.iter().map(|p| p.name).collect::<Vec<_>>(),
            ["hid", "emb", "rows", "idlog"]
        );
        // bases aligned, in order, no overlap, the total covers the last
        for w in l.parts.windows(2) {
            assert!(w[0].off + w[0].bytes() <= w[1].off);
        }
        for p in &l.parts {
            assert_eq!(p.off % ALIGN, 0);
            assert!(
                p.head >= 256 * p.row_bytes,
                "{}: the head holds keep rows",
                p.name
            );
            let (lo, n) = p.tail(STRIDE);
            assert!(lo >= p.off + 256 * p.row_bytes);
            assert!(
                lo + n * STRIDE <= p.off + p.bytes(),
                "{}: the tail inside",
                p.name
            );
        }
        assert!(l.total >= l.separate_bytes());
        assert!(l.total - l.separate_bytes() < 4 * ALIGN);
        // the census's numbers at the served boot: hid 62 + emb 15 + rows 15 + idlog 5
        let t = l.tails(STRIDE);
        assert_eq!(
            t.iter().map(|x| (x.0, x.2)).collect::<Vec<_>>(),
            [("hid", 62), ("emb", 15), ("rows", 15), ("idlog", 5)]
        );
        assert_eq!(l.slots(STRIDE), 97);
        // no MTP: no hid; a boot no wider than the heads: nothing to lend
        assert_eq!(layout(8192, 256, false, 2560, 10240, 48).slots(STRIDE), 35);
        assert_eq!(layout(256, 256, true, 2560, 10240, 48).slots(STRIDE), 0);
        // the idlog's head holds a 256-row pass's 48 layer-packed slices
        let i = l.part("idlog").unwrap();
        assert!(i.head >= 48 * 256 * 40);
    }

    #[test]
    fn fn100_census_prints_the_class() {
        let c = census(
            8192,
            256,
            true,
            2560,
            10240,
            48,
            262_144 * 128,
            2 * (32 << 20) + (16 << 20) + (1 << 20) + (1 << 20),
            9 * 10240 * 2 + 10240 * 2,
            STRIDE,
        );
        let mib = |x: u64| x as f64 / 1048576.0;
        let mut tot = 0;
        let mut lend = 0;
        for m in &c {
            println!(
                "{:>14} | {:>8.1} MiB | rows {:>5} | tail {:>8.1} MiB | slots {:>3} | writers: {} | baked: {}",
                m.name,
                mib(m.bytes),
                m.rows,
                mib(m.tail),
                m.slots,
                m.writers,
                m.baked
            );
            tot += m.bytes;
            lend += m.slots;
        }
        println!("total {:.1} MiB, lendable {lend} slots", mib(tot));
        assert_eq!(lend, 97);
    }
}
