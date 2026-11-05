//! FN96: the SEQUENCE-MAJOR state pools and the idle sequence slots' lend (host law, no HIP).
//!
//! FN13's pools hold one buffer a layer with every sequence slot in it (`layer`): the GDN layer
//! `l`'s snapshot pool is `[1 + n x pool_w] x hb`, its live pool `[n + 1] x hb`, its conv pool
//! `[n + 1] x cb` (slot 0 = the kernels' NULL slot: `slot <= 0` is skipped in every GDN decode
//! body, so no byte of it is ever read or written). `seq` lays the same rows out as ONE region a
//! sequence slot -- per GDN layer `[conv cb][live hb][snapshots pool_w x hb]` -- so an idle
//! sequence is one contiguous run of bytes. The kernels are unchanged: each still takes a
//! per-layer base and a slot index with its per-slot stride (`cs_seq` = cb, `st_slot` = hb, 64-bit
//! products in every body); the region bytes `S` are a multiple of both strides, so sequence `s`'s
//! layer-`l` rows are slot `1 + s x S / stride` of the base `region 0 + offset(l) - stride`. The
//! step tables (`cidx`, `sidx`) carry that index ([`super::exec::PadLaw::idx`]); a prefill chunk
//! addresses its own rows (index 0 of the sequence's bases) and never sees the law.
//!
//! The lend (`--xtier-lend-seq-pools`): the regions of the sequence slots `0..MAX_SEQS` (never the
//! SPARE: the pad rows and the swaps use it every step) are carved into whole expert strides; a
//! stride is lendable while every sequence slot it overlaps is free.

use super::plan::{self, LAYERS};
use saber_model::qwen4_exp::LayerKind;

/// The layout of the per-sequence state pools (`--fn-pools-layout`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolsLayout {
    /// FN13's: one buffer a layer holding every sequence slot (the control).
    Layer,
    /// FN96: one region a sequence slot.
    Seq,
}

impl PoolsLayout {
    /// # Errors
    /// An unknown name.
    pub fn parse(s: &str) -> anyhow::Result<PoolsLayout> {
        match s {
            "layer" => Ok(PoolsLayout::Layer),
            "seq" => Ok(PoolsLayout::Seq),
            _ => anyhow::bail!("--fn-pools-layout {s}: layer | seq"),
        }
    }
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            PoolsLayout::Layer => "layer",
            PoolsLayout::Seq => "seq",
        }
    }
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// The sequence-major geometry: per GDN layer the (conv, live, snapshots) offsets inside a
/// region; the region's bytes `s` (a multiple of `cb` and `hb`); the lead bytes before region 0
/// (every per-layer base `region 0 + offset - stride` stays inside the allocation).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeqGeom {
    pub off: Vec<[u64; 3]>,
    /// The rows a region holds (before the rounding).
    pub content: u64,
    pub s: u64,
    pub lead: u64,
    pub cb: u64,
    pub hb: u64,
}

impl SeqGeom {
    #[must_use]
    pub fn new(cb: u64, hb: u64, pool_w: u32) -> SeqGeom {
        let mut off = vec![[0u64; 3]; LAYERS as usize + 1];
        let mut at = 0u64;
        for (l, o) in off.iter_mut().enumerate().take(LAYERS as usize) {
            if plan::kind(l as u32) != LayerKind::Qsa {
                *o = [at, at + cb, at + cb + hb];
                at += cb + hb + u64::from(pool_w) * hb;
            }
        }
        let q = cb / gcd(cb, hb) * hb;
        SeqGeom {
            off,
            content: at,
            s: at.div_ceil(q) * q,
            lead: cb.max(hb),
            cb,
            hb,
        }
    }
    /// The allocation's bytes for `n` sequence slots.
    #[must_use]
    pub fn bytes(&self, n: u64) -> u64 {
        self.lead + n * self.s
    }
    /// The step tables' index law: (conv slots a region, GDN slots a region).
    #[must_use]
    pub fn idx(&self) -> (u32, u32) {
        (
            u32::try_from(self.s / self.cb).expect("conv index"),
            u32::try_from(self.s / self.hb).expect("GDN index"),
        )
    }
}

/// One lendable span: the byte offset of sequence slot 0's rows inside its owner and the bytes
/// a sequence slot holds there (slots `0..MAX_SEQS` back to back).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub off: u64,
    pub seq_bytes: u64,
}

/// The strides carved from one span of `nseq` sequence slots: (byte offset of the first stride,
/// strides, the sequence-slot mask each overlaps). The strides end at the span's end, so the
/// highest slots' strides come first free (the compaction keeps the live set at the bottom).
#[must_use]
pub fn carve(sp: Span, nseq: u32, stride: u64) -> (u64, Vec<u32>) {
    let total = u64::from(nseq) * sp.seq_bytes;
    let n = total / stride;
    let lo = sp.off + total - n * stride;
    let masks = (0..n)
        .map(|i| {
            let a = lo + i * stride - sp.off;
            let z = a + stride - 1;
            let (s0, s1) = (a / sp.seq_bytes, z / sp.seq_bytes);
            (s0..=s1).fold(0u32, |m, s| m | (1 << s))
        })
        .collect();
    (lo, masks)
}

/// The census: strides lendable when the sequence slots in `free` (a mask) are free.
#[must_use]
pub fn lendable(spans: &[Span], nseq: u32, stride: u64, free: u32) -> u64 {
    spans
        .iter()
        .map(|&sp| {
            carve(sp, nseq, stride)
                .1
                .iter()
                .filter(|&&m| m & !free == 0)
                .count() as u64
        })
        .sum()
}

/// The lendable spans of a layout (`nseq` = MAX_SEQS sequence slots: the SPARE is never lent).
/// `layer`: every GDN layer's snapshot pool from slot 1 and live pool from slot 1 (the conv
/// pools hold 8 x cb, under one stride: none); `seq`: the regions from region 0.
#[must_use]
pub fn spans(layout: PoolsLayout, cb: u64, hb: u64, pool_w: u32) -> Vec<Span> {
    match layout {
        PoolsLayout::Seq => {
            let g = SeqGeom::new(cb, hb, pool_w);
            vec![Span {
                off: g.lead,
                seq_bytes: g.s,
            }]
        }
        PoolsLayout::Layer => (0..LAYERS)
            .filter(|&l| plan::kind(l) != LayerKind::Qsa)
            .flat_map(|_| {
                [
                    Span {
                        off: hb,
                        seq_bytes: u64::from(pool_w) * hb,
                    },
                    Span {
                        off: hb,
                        seq_bytes: hb,
                    },
                    Span {
                        off: cb,
                        seq_bytes: cb,
                    },
                ]
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STRIDE: u64 = 2_617_344;
    // the served dims a rank (TP=2): hv 24, conv 5,120, f16 state, MTP depth 3 (pool_w 4)
    const HB: u64 = 24 * 128 * 128 * 2;
    const CB: u64 = 5120 * 6 * 2;

    #[test]
    fn fn96_the_seq_geometry_keeps_every_kernel_stride_whole() {
        for (cb, hb, pw) in [(CB, HB, 4), (5120 * 3 * 2, HB, 1), (CB, HB * 2, 4)] {
            let g = SeqGeom::new(cb, hb, pw);
            assert_eq!(g.s % cb, 0);
            assert_eq!(g.s % hb, 0);
            assert!(g.s >= g.content && g.s - g.content < cb / gcd(cb, hb) * hb);
            // every per-layer base (region 0 + offset - stride) inside the allocation
            for (l, o) in g.off.iter().enumerate() {
                if plan::kind(l as u32) == LayerKind::Qsa || l == LAYERS as usize {
                    continue;
                }
                assert!(g.lead + o[0] >= cb && g.lead + o[2] >= hb && g.lead + o[1] >= hb);
                // sequence s's rows are slot 1 + s x S / stride of that base
                let (kc, kg) = g.idx();
                for s in 0..9u64 {
                    let base_c = g.lead + o[0] - cb;
                    assert_eq!(
                        base_c + (1 + s * u64::from(kc)) * cb,
                        g.lead + s * g.s + o[0]
                    );
                    let base_g = g.lead + o[2] - hb;
                    for t in 0..u64::from(pw) {
                        assert_eq!(
                            base_g + (1 + s * u64::from(kg) + t) * hb,
                            g.lead + s * g.s + o[2] + t * hb
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn fn96_census_at_the_served_dims() {
        let g = SeqGeom::new(CB, HB, 4);
        let layer = spans(PoolsLayout::Layer, CB, HB, 4);
        let seq = spans(PoolsLayout::Seq, CB, HB, 4);
        let free = |c: u32| (0xffu32 << c) & 0xff;
        let row = |sp: &[Span]| -> Vec<u64> {
            [0, 1, 2, 4, 8]
                .iter()
                .map(|&c| lendable(sp, 8, STRIDE, free(c)))
                .collect()
        };
        let (l, s) = (row(&layer), row(&seq));
        println!(
            "fn96 census: region {} B (content {}) | layer conc 0/1/2/4/8 {l:?} | seq {s:?}",
            g.s, g.content
        );
        // a whole free sequence-major run carves one stride less than its bytes at most
        assert_eq!(s[1], 7 * g.s / STRIDE);
        assert_eq!(s[4], 0);
        assert_eq!(l[4], 0);
        assert!(s[1] > l[1] && s[2] > l[2] && s[3] > l[3]);
        assert_eq!(g.idx(), (2368, 185));
    }

    #[test]
    fn fn96_a_stride_is_lendable_only_while_every_slot_it_overlaps_is_free() {
        let sp = Span {
            off: 10,
            seq_bytes: 7,
        };
        let (lo, m) = carve(sp, 4, 3);
        // 28 bytes -> 9 strides from byte 1 of the span
        assert_eq!(lo, 11);
        assert_eq!(m.len(), 9);
        assert_eq!(m[0], 0b0001);
        assert_eq!(m[2], 0b0010);
        assert_eq!(m[4], 0b0110);
        assert_eq!(m[8], 0b1000);
        assert_eq!(lendable(&[sp], 4, 3, 0b1000), 2);
        assert_eq!(lendable(&[sp], 4, 3, 0b1111), 9);
        assert_eq!(lendable(&[sp], 4, 3, 0), 0);
    }

    #[test]
    fn fn96_the_step_tables_carry_the_region_strided_index() {
        use crate::flashnext::exec::{ChunkConsts, PadLaw};
        let get = |tb: &[u8], offs: &[(&'static str, usize)], name: &str, n: usize| -> Vec<i32> {
            let o = offs.iter().find(|x| x.0 == name).expect("table").1;
            (0..n)
                .map(|i| i32::from_le_bytes(tb[o + 4 * i..o + 4 * i + 4].try_into().unwrap()))
                .collect()
        };
        let law = |idx| PadLaw {
            live: 1,
            pad: Some(8),
            idx,
        };
        // b2 at width 4 (one live sequence, one pad row at the SPARE), pool_w 4
        let (tb, offs) = ChunkConsts::host_verify_at(&[5, 0], 4, &[1, 1], &[0, 0], 4, 4, law(None));
        assert_eq!(get(&tb, &offs, "cidx", 2), vec![1, 9]);
        assert_eq!(get(&tb, &offs, "sidx", 8), vec![1, 2, 3, 4, 33, 34, 35, 36]);
        let (tq, oq) =
            ChunkConsts::host_verify_at(&[5, 0], 4, &[1, 1], &[0, 0], 4, 4, law(Some((2368, 185))));
        assert_eq!(get(&tq, &oq, "cidx", 2), vec![1, 1 + 8 * 2368]);
        assert_eq!(
            get(&tq, &oq, "sidx", 8),
            vec![
                1,
                2,
                3,
                4,
                1 + 8 * 185,
                2 + 8 * 185,
                3 + 8 * 185,
                4 + 8 * 185
            ]
        );
        // every other table byte for byte (only the two index tables move)
        assert_eq!(tb.len(), tq.len());
        assert_eq!(offs, oq);
        let moved: Vec<&str> = offs
            .iter()
            .enumerate()
            .filter(|&(i, &(_, o))| {
                let z = offs.get(i + 1).map_or(tb.len(), |x| x.1);
                tb[o..z] != tq[o..z]
            })
            .map(|(_, x)| x.0)
            .collect();
        assert_eq!(moved, vec!["cidx", "sidx"]);
    }
}
