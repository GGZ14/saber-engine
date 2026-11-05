//! FN22: the Flash-Next MTP drafter's DEVICE-SIDE CHAIN (`flashnext_draft.hip`) -- the declared
//! geometry and a host model of the law, bit for bit.
//!
//! Between two drafter passes the host chain (FN16) copied every row's logits half to the host,
//! took the argmax, swapped 8 bytes a row with the peer over the TCP exchange, gathered the winning
//! token's embedding row from the checkpoint file and uploaded it. The device chain does the same on
//! the card, four launches a pass, no host hop:
//!
//! 1. `fn_draft_argmax_part` -- per selected row, [`SPLITS`] partial argmaxes over the rank's
//!    vocabulary half: `key << 32 | !index` as a u64 (the u64 max IS the law: the larger bf16 key,
//!    then the LOWER index).
//! 2. `fn_draft_pack` -- the row's winner; the exchange payload row `[2 H + LANES]` bf16: this
//!    rank's slot = the winner's embedding row (the checkpoint's bytes, from the rank-resident half
//!    of `embed_tokens`), its [`RANK_LANES`] lanes = (key hi, key lo, gid b0, b1, b2, 0, 0, 0) as
//!    bf16 small integers; the other rank's slot and lanes = `-0.0` (0x8000).
//! 3. the certified 2-rank EXACT all-reduce (`saber_ar_oneshot_2rank_exact_bf16`, the payload is far
//!    under `quant_min_bytes`): `bf16(f32(x) + f32(-0.0)) == x` for every non-NaN bf16 `x` (+0 and
//!    -0 included, in either operand order -- [`exchange_is_exact`] proves it over all 2^16 patterns
//!    on the host), so both ranks hold both candidates' rows and lanes, byte for byte.
//! 4. `fn_draft_select` -- the pair law (the larger key; an exact tie -> rank 0's candidate, the
//!    lower global id), the winner's row -> the next pass's embedding input, the id -> the chain's
//!    id table at the pass counter, the counter + 1.
//!
//! The law is the host chain's `row_argmax` + `merge_argmax` exactly ([`host_pick`] is that law;
//! [`device_pick`] models the kernels, lanes and exchange included). `tie_high` is the tie law's
//! negative control: both ties to the highest index (never served).

/// Partial argmaxes per row (one wave's lanes: `fn_draft_pack` reduces them with one wave).
pub const SPLITS: u32 = 32;
/// Hidden size (the embedding row, bf16).
pub const H: u32 = 2560;
/// Lanes per payload row (8 per rank).
pub const LANES: u32 = 16;
/// Lanes a rank owns in a payload row.
pub const RANK_LANES: u32 = 8;
/// bf16 elements of one payload row: both candidates' rows + the lanes.
pub const PAY_ROW: u32 = 2 * H + LANES;
/// The id table's row stride (the decode buckets' widest, 8).
pub const IDS_STRIDE: u32 = 8;
/// Passes the id table holds (the verify-width law's widest chain is 4).
pub const MAX_PASSES: u32 = 8;
/// Launches the device tail adds to a drafter pass: argmax part, pack, the exact reduce, select.
pub const TAIL_LAUNCHES: u32 = 4;

/// The order-preserving integer key of a bf16 (larger key = larger value; -0 < +0) -- the kernel's
/// `fn22_key` and the host chain's `bf16_key`.
#[must_use]
pub fn key(b: u16) -> u16 {
    if b & 0x8000 != 0 {
        !b
    } else {
        b | 0x8000
    }
}

/// A small integer (<= 255) as a bf16 bit pattern (exact).
#[must_use]
pub fn enc(v: u32) -> u16 {
    debug_assert!(v <= 255);
    ((v as f32).to_bits() >> 16) as u16
}

/// A lane's small integer back.
#[must_use]
pub fn dec(b: u16) -> u32 {
    f32::from_bits(u32::from(b) << 16) as u32
}

/// The exact reduce of one bf16 element pair as the certified kernel computes it:
/// `bf16_rne(f32(a) + f32(b))`.
#[must_use]
pub fn ar_add(a: u16, b: u16) -> u16 {
    let s = f32::from_bits(u32::from(a) << 16) + f32::from_bits(u32::from(b) << 16);
    let x = s.to_bits();
    if s.is_nan() {
        return ((x >> 16) | 0x40) as u16;
    }
    let r = x.wrapping_add(0x7fff + ((x >> 16) & 1));
    (r >> 16) as u16
}

/// The exchange law over every non-NaN bf16 pattern: `x + (-0)` and `(-0) + x` are `x`.
#[must_use]
pub fn exchange_is_exact() -> bool {
    (0..=u16::MAX)
        .filter(|&x| !f32::from_bits(u32::from(x) << 16).is_nan())
        .all(|x| ar_add(x, 0x8000) == x && ar_add(0x8000, x) == x)
}

/// The host chain's law on one row: each rank's argmax over its half (the largest key, the lowest
/// index on a tie), then the pair's (the larger value, the lower id on a tie). `halves[r]` = rank
/// r's bf16 logits. Returns the global id.
#[must_use]
pub fn host_pick(halves: [&[u16]; 2]) -> u32 {
    let half = halves[0].len() as u32;
    let arg = |h: &[u16]| -> (u32, u16) {
        let mut best = (0u32, key(h[0]));
        for (i, &b) in h.iter().enumerate().skip(1) {
            if key(b) > best.1 {
                best = (i as u32, key(b));
            }
        }
        best
    };
    let (i0, k0) = arg(halves[0]);
    let (i1, k1) = arg(halves[1]);
    if k1 > k0 {
        half + i1
    } else {
        i0
    }
}

/// The device kernels' law on one row, modelled step for step (the u64 partials, the lanes, the
/// exact reduce, the select): returns (the global id, the winner's slot `[H]` after the exchange).
/// `embh[r]` = rank r's embedding half `[half, H]`; `pad` = a pad row (token 0).
#[must_use]
pub fn device_pick(
    halves: [&[u16]; 2],
    embh: [&[u16]; 2],
    pad: bool,
    tie_high: bool,
) -> (u32, Vec<u16>) {
    let half = halves[0].len() as u32;
    let hu = H as usize;
    let mut pays: Vec<Vec<u16>> = Vec::new();
    for rank in 0..2u32 {
        let h = halves[rank as usize];
        let per = (h.len() as u32 / 8).div_ceil(SPLITS) * 8;
        let mut best = 0u64;
        for s in 0..SPLITS {
            let mut b = 0u64;
            for i in (s * per)..((s + 1) * per).min(h.len() as u32) {
                let lo = if tie_high { i } else { !i };
                b = b.max((u64::from(key(h[i as usize])) << 32) | u64::from(lo));
            }
            best = best.max(b);
        }
        let (mut k, mut lidx) = (
            (best >> 32) as u32,
            if tie_high {
                best as u32
            } else {
                !(best as u32)
            },
        );
        if pad {
            k = if rank == 0 { 0xffff } else { 0 };
            lidx = 0;
        }
        let gid = rank * half + lidx;
        let mut row = vec![0x8000u16; PAY_ROW as usize];
        let at = rank as usize * hu;
        row[at..at + hu].copy_from_slice(&embh[rank as usize][lidx as usize * hu..][..hu]);
        let bytes = [
            k >> 8,
            k & 0xff,
            gid & 0xff,
            (gid >> 8) & 0xff,
            (gid >> 16) & 0xff,
            0,
            0,
            0,
        ];
        for (j, &v) in bytes.iter().enumerate() {
            row[2 * hu + rank as usize * 8 + j] = enc(v);
        }
        pays.push(row);
    }
    // the exact reduce: out = local + peer (rank 0's view; rank 1's is the same sum reversed)
    let red: Vec<u16> = pays[0]
        .iter()
        .zip(&pays[1])
        .map(|(&a, &b)| ar_add(a, b))
        .collect();
    let ln = &red[2 * hu..];
    let k0 = (dec(ln[0]) << 8) | dec(ln[1]);
    let k1 = (dec(ln[8]) << 8) | dec(ln[9]);
    let w = usize::from(if tie_high { k1 >= k0 } else { k1 > k0 });
    let l = &ln[8 * w..];
    let gid = dec(l[2]) | (dec(l[3]) << 8) | (dec(l[4]) << 16);
    (gid, red[w * hu..(w + 1) * hu].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_exchange_is_exact_for_every_non_nan_bf16() {
        assert!(exchange_is_exact());
        // the control: +0 is NOT the identity (-0 + +0 = +0 moves a -0.0 embedding element)
        assert_ne!(ar_add(0x8000, 0x0000), 0x8000);
    }

    #[test]
    fn lanes_carry_every_byte_exactly() {
        for v in 0..=255 {
            assert_eq!(dec(enc(v)), v);
            assert_eq!(ar_add(enc(v), 0x8000), enc(v));
        }
    }

    fn lcg(seed: &mut u64) -> u16 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*seed >> 33) as u16
    }

    fn check(h0: &[u16], h1: &[u16], e: [&[u16]; 2]) -> (bool, bool) {
        let want = host_pick([h0, h1]);
        let (got, row) = device_pick([h0, h1], e, false, false);
        let hu = H as usize;
        let half = h0.len() as u32;
        let src = if want < half {
            &e[0][want as usize * hu..][..hu]
        } else {
            &e[1][(want - half) as usize * hu..][..hu]
        };
        let (ctl, _) = device_pick([h0, h1], e, false, true);
        (got == want && row == src, ctl != want)
    }

    #[test]
    fn the_device_law_is_the_host_law_and_the_tie_high_control_differs_on_ties() {
        // a small vocabulary half (the law does not depend on its size; 8-aligned as the kernel's)
        let half = 1024usize;
        let hu = H as usize;
        let mut s = 7u64;
        let e0: Vec<u16> = (0..half * hu).map(|_| lcg(&mut s) & 0xbfff).collect();
        let e1: Vec<u16> = (0..half * hu).map(|_| lcg(&mut s) & 0xbfff).collect();
        let e = [&e0[..], &e1[..]];
        // random rows (finite: the exponent never all ones)
        for _ in 0..200 {
            let h0: Vec<u16> = (0..half).map(|_| lcg(&mut s) & 0xbfff).collect();
            let h1: Vec<u16> = (0..half).map(|_| lcg(&mut s) & 0xbfff).collect();
            assert!(check(&h0, &h1, e).0);
        }
        // ties: inside rank 0, across the ranks, the whole vocabulary, -0 vs +0
        let one = 0x3f80u16;
        let mut h0 = vec![0x3f00u16; half];
        let mut h1 = vec![0x3f00u16; half];
        h0[5] = one;
        h0[700] = one;
        assert_eq!(check(&h0, &h1, e), (true, true), "a tie inside rank 0");
        h1[3] = one;
        assert_eq!(check(&h0, &h1, e), (true, true), "a tie across the pair");
        let flat = vec![one; half];
        assert_eq!(check(&flat, &flat, e), (true, true), "every entry equal");
        let neg0 = vec![0x8000u16; half];
        let mut pos0 = neg0.clone();
        pos0[9] = 0x0000;
        let (ok, ctl) = check(&neg0, &pos0, e);
        assert!(ok && !ctl, "+0 beats -0 (no tie): the control agrees");
        assert_eq!(host_pick([&neg0, &pos0]), half as u32 + 9);
        // a pad row is token 0, rank 0's row 0
        let (g, row) = device_pick([&h0, &h1], e, true, false);
        assert_eq!((g, &row[..]), (0, &e0[..hu]));
    }
}
