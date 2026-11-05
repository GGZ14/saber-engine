//! LR-N: COUPLED DRAFTING -- `--spec-couple on|off` (absent = `off` = the default and the control).
//!
//! The finding it rests on (LR-L): at b8 our verify step costs what the reference engine's does,
//! but it commits fewer tokens a step -- the conc-8 gap is draft acceptance, not step time. A
//! SAMPLED row accepts a greedy draft only when the target's draw happens to be the drafter's
//! argmax: probability `p(argmax q)`, which falls as the temperature rises.
//!
//! The law (Saber's own; derived from the sampler's exponential race, not from anyone's code):
//!
//! * the target's draw at output position `p` is `t = argmax_i p_i / E_i` with
//!   `E_i = SeededNoise::new(seed, p, choice).exponential(i)` (the host sampler; FN119's device
//!   twin draws the same `E`);
//! * the drafter proposes `x = argmax_{i in C} s_i / T - ln E_i` -- the SAME `E_i` -- over its own
//!   candidates `C` (each rank's top-`top_k` of the int2 head's reranked candidates; the pair's
//!   union by the larger perturbed key), `s` its exact rescored logits, `T` the request's
//!   temperature. Both are exponential races over the same noise, so `x == t` whenever the two
//!   laws put the race's winner on the same token: acceptance `P(x == t)`, which tends to 1 as the
//!   drafter's law approaches the target's (a greedy draft's is `p(argmax s)`).
//! * the walk is UNCHANGED: row `j` commits the target's own draw and continues while that draw
//!   IS the draft. So every committed token is the target's draw, byte for byte, arm on or off:
//!   the output distribution, the seed law ("the same seed gives the same tokens as
//!   non-speculative decoding") and every byte gate hold. Only the drafts move.
//!
//! Why not the textbook rejection coupling (accept `x ~ q` with `min(1, p(x) / q(x))`, else draw
//! the residual `(p - q)+`): it reaches `sum min(p, q)` acceptance -- in a toy census (top-k 20
//! targets, drafter = target + Gaussian logit noise, T 0.3..1.0) 0.5-1.6 points above the shared
//! race -- but it makes the committed token a function of the draft, so a seed no longer names
//! the same tokens as non-speculative decoding, and its verify needs the drafter's `q` on the
//! card. The shared race keeps every law for ~1 point.
//!
//! An UNSEEDED sampled request draws one seed at its first coupled step (`SrvState::couple_seeds`)
//! instead of a fresh key a step: the drafts of a round must know the key the NEXT step's rows
//! will draw with. A key drawn once per request is the same law in distribution.
//!
//! Greedy rows (T = 0), pad rows and a prompt's first round keep FN22's integer argmax: the
//! shipped drafts, bit for bit (the coupled tail's 16-bit-key branch).

use super::dstep::mix;
use saber_serve::sampler::RowParams;

/// One ready slot's coupled-drafting row as the leader sends it with `Cmd::Decode`: key base
/// lo / hi ([`couple_base`]), the output position the verify's row 0 decides, 1 / T as f32 bits
/// (0 = a greedy row: FN22's argmax), top_k (0 = every candidate).
pub type CoupleRow = [u32; 5];

/// The chain input buffer's coupled region: rows of 4 words from this byte offset (base lo,
/// base hi, pass 1's output position, 1 / T bits) ...
pub const IN_COUPLE: u64 = 80;
/// ... and each row's top_k (u32) from this one (`flashnext_couple.hip`'s words 20 / 52).
pub const IN_COUPLE_K: u64 = 208;
/// The region's bytes (rows + top_k words).
pub const REGION_BYTES: usize =
    (IN_COUPLE_K - IN_COUPLE) as usize + 4 * saber_kernels::fn22::IDS_STRIDE as usize;

/// The position coordinate's multiplier in the sampler's key (`SeededNoise::new`).
#[allow(dead_code)] // the host twin (tests; tools/capture/lrn/couple_twin.py mirrors it)
pub const C_POS: u64 = 0x00FF_51AF_D7ED_558D;
/// The choice coordinate's multiplier.
pub const C_CHOICE: u64 = 0xC4CE_B9FE_1A85_EC53;

/// The noise key's request part: `dstep::noise_key(seed, p, choice) == couple_base(seed, choice)
/// ^ mix(p * C_POS)`.
#[must_use]
pub fn couple_base(seed: u64, choice: u32) -> u64 {
    mix(seed) ^ mix(u64::from(choice).wrapping_mul(C_CHOICE))
}

/// The key a draft for output position `p` takes (the kernel's `base ^ fnc_mix(pos * C1)`).
#[must_use]
#[allow(dead_code)] // the host twin (tests; tools/capture/lrn/couple_twin.py mirrors it)
pub fn draft_key(base: u64, p: u64) -> u64 {
    base ^ mix(p.wrapping_mul(C_POS))
}

/// A sampled slot's coupled row: the key base, the output position its verify row 0 decides, 1 / T,
/// and top_k when it cuts inside the vocabulary (0 = every candidate). A greedy row is all zero.
#[must_use]
pub fn couple_row(params: &RowParams, seed: u64, choice: u32, position: u64) -> CoupleRow {
    if params.is_greedy() {
        return [0; 5];
    }
    let base = couple_base(seed, choice);
    let tk = if params.top_k >= 1 && params.top_k < params.vocab_size {
        u32::try_from(params.top_k).unwrap_or(0)
    } else {
        0
    };
    [
        base as u32,
        (base >> 32) as u32,
        position as u32,
        (1.0f32 / params.temperature).to_bits(),
        tk,
    ]
}

/// The chain inputs' coupled region ([`REGION_BYTES`] from [`IN_COUPLE`]) for counts `acc`: row i
/// drafts output positions `cur_pos + acc[i] + 1 + pass`. Zero (every row greedy) when the step
/// carried no rows.
#[must_use]
pub fn couple_region(cur: &[CoupleRow], acc: &[u32], n: usize) -> Vec<u8> {
    let ids = saber_kernels::fn22::IDS_STRIDE as usize;
    let mut b = vec![0u8; REGION_BYTES];
    if cur.len() < n || acc.len() < n {
        return b;
    }
    for i in 0..n.min(ids) {
        let [lo, hi, pos, invt, tk] = cur[i];
        let p1 = pos.wrapping_add(acc[i]).wrapping_add(1);
        for (q, x) in [lo, hi, p1, invt].into_iter().enumerate() {
            let a = 16 * i + 4 * q;
            b[a..a + 4].copy_from_slice(&x.to_le_bytes());
        }
        let a = (IN_COUPLE_K - IN_COUPLE) as usize + 4 * i;
        b[a..a + 4].copy_from_slice(&tk.to_le_bytes());
    }
    b
}

/// The target sampler's `E(id)` under key `key` (`SeededNoise`'s exponential; the kernel's
/// `fnc_noise`).
#[must_use]
#[allow(dead_code)] // the host twin (tests; tools/capture/lrn/couple_twin.py mirrors it)
pub fn noise(key: u64, id: u32) -> f32 {
    let bits = mix(key ^ mix(u64::from(id)));
    let u = (((bits >> 11) + 1) as f64) * (1.0 / 9_007_199_254_740_993.0);
    -(u.ln()) as f32
}

/// The f32 total-order key (`fnc_key32`).
#[must_use]
#[allow(dead_code)] // the host twin (tests; tools/capture/lrn/couple_twin.py mirrors it)
pub fn key32(f: f32) -> u32 {
    let b = f.to_bits();
    if b & 0x8000_0000 != 0 {
        !b
    } else {
        b | 0x8000_0000
    }
}

/// The bf16 total-order key (FN22's `fn22_key`).
#[must_use]
#[allow(dead_code)] // the host twin (tests; tools/capture/lrn/couple_twin.py mirrors it)
pub fn key16(b: u16) -> u32 {
    if b & 0x8000 != 0 {
        u32::from(!b)
    } else {
        u32::from(b | 0x8000)
    }
}

/// The HOST TWIN of one rank's coupled pick (`fn_couple_acp`'s wave 0) over that rank's
/// candidates `(local index, bf16 score bits)` for a row with coupled `row` at pass `pass`:
/// (32-bit key, local index) of the winner -- the coupled race for a sampled row, FN22's argmax
/// (key16, lower index on a tie) for a greedy one. `ln` = the natural log the card uses (the
/// kernel's fast `__logf`; tests pass `f32::ln`).
#[must_use]
#[allow(dead_code)] // the host twin (tests; tools/capture/lrn/couple_twin.py mirrors it)
pub fn rank_pick(
    cands: &[(u32, u16)],
    row: &CoupleRow,
    pass: u32,
    rank: u32,
    half: u32,
    ln: &dyn Fn(f32) -> f32,
) -> (u32, u32) {
    let [lo, hi, p1, invt, tk] = *row;
    let mut best: u64 = 0;
    if invt == 0 {
        for &(c, s) in cands {
            best = best.max((u64::from(key16(s)) << 32) | u64::from(!c));
        }
    } else {
        let base = u64::from(lo) | (u64::from(hi) << 32);
        let key = draft_key(base, u64::from(p1) + u64::from(pass));
        let it = f32::from_bits(invt);
        for &(c, s) in cands {
            let rnk = cands
                .iter()
                .filter(|&&(d, t)| key16(t) > key16(s) || (key16(t) == key16(s) && d < c))
                .count() as u32;
            if tk != 0 && rnk >= tk {
                continue;
            }
            let v = f32::from_bits(u32::from(s) << 16);
            let z = v * it - ln(noise(key, rank * half + c));
            if !z.is_nan() {
                best = best.max((u64::from(key32(z)) << 32) | u64::from(!c));
            }
        }
    }
    ((best >> 32) as u32, !(best as u32))
}

/// The pair's pick (`fn_couple_select`): the larger 32-bit key, a tie to rank 0; the global id.
#[must_use]
#[allow(dead_code)] // the host twin (tests; tools/capture/lrn/couple_twin.py mirrors it)
pub fn pair_pick(r0: (u32, u32), r1: (u32, u32), half: u32) -> u32 {
    if r1.0 > r0.0 {
        half + r1.1
    } else {
        r0.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use saber_serve::sampler::{NoiseSource, PenaltyContext, Sampler, SeededNoise};
    use saber_serve::sampling::SamplingParams;

    fn bf(x: f32) -> u16 {
        (x.to_bits() >> 16) as u16
    }

    #[test]
    fn the_draft_key_is_the_samplers_key_at_the_drafts_position() {
        for (seed, p, ch) in [(0u64, 0u64, 0u32), (42, 17, 0), (u64::MAX, 123_456, 3)] {
            assert_eq!(
                draft_key(couple_base(seed, ch), p),
                super::super::dstep::noise_key(seed, p, ch)
            );
            let mut n = SeededNoise::new(seed, p, ch);
            for id in [0u32, 1, 77, 248_319] {
                assert_eq!(
                    n.exponential(id as usize).to_bits(),
                    noise(draft_key(couple_base(seed, ch), p), id).to_bits()
                );
            }
        }
    }

    #[test]
    fn the_region_is_rows_of_four_words_then_top_k() {
        let cur = vec![
            [1, 2, 100, 0x3fb6_db6e, 20],
            [0; 5],
            [3, 4, 7, 0x3f80_0000, 0],
        ];
        let r = couple_region(&cur, &[2, 0, 4], 3);
        let w = |i: usize| u32::from_le_bytes(r[4 * i..4 * i + 4].try_into().unwrap());
        assert_eq!(r.len(), REGION_BYTES);
        assert_eq!([w(0), w(1), w(2), w(3)], [1, 2, 103, 0x3fb6_db6e]);
        assert_eq!([w(4), w(5), w(6), w(7)], [0, 0, 1, 0]);
        assert_eq!([w(8), w(9), w(10), w(11)], [3, 4, 12, 0x3f80_0000]);
        let k = (IN_COUPLE_K - IN_COUPLE) as usize / 4;
        assert_eq!([w(k), w(k + 1), w(k + 2)], [20, 0, 0]);
        assert_eq!(IN_COUPLE / 4, 20);
        assert_eq!(IN_COUPLE_K / 4, 52);
        assert!(IN_COUPLE_K as usize + 32 <= 256, "inside the chain's stage");
        // no rows -> all greedy
        assert!(couple_region(&[], &[1], 1).iter().all(|&b| b == 0));
    }

    #[test]
    fn a_greedy_row_keeps_fn22s_argmax() {
        let c = [(5u32, bf(3.0)), (2, bf(3.0)), (9, bf(-1.0))];
        // equal keys -> the lower index
        assert_eq!(
            rank_pick(&c, &[0; 5], 0, 0, 100, &f32::ln),
            (key16(bf(3.0)), 2)
        );
    }

    /// The distribution law + the byte law + the acceptance claim on a toy vocabulary, with the
    /// host sampler itself as the target: 40,000 seeded positions at T 0.7 and 1.0 (top_k 20,
    /// top_p 0.95). The committed token is the target's draw in both arms (the walk never reads
    /// the draft's value except to compare), its histogram passes a chi-square against the
    /// target's processed law, and the coupled draft matches the draw far more often than the
    /// greedy one (and nearly always when the drafter IS the target: the drafter cuts by top_k per
    /// rank and never by top_p, so its race can land in the target's cut tail).
    #[test]
    fn coupled_drafts_keep_the_targets_draws_and_raise_acceptance() {
        let v = 48usize;
        let half = 24u32;
        let sampler = Sampler::new();
        let mut g = 0x1234_5678_u64;
        let mut rnd = || {
            g = mix(g);
            (g >> 11) as f64 / (1u64 << 53) as f64
        };
        // target logits (bf16-exact) and a drafter = target + noise
        let lt: Vec<f32> = (0..v)
            .map(|i| f32::from_bits(u32::from(bf((rnd() as f32) * 6.0 - (i as f32) * 0.05)) << 16))
            .collect();
        let ld: Vec<f32> = lt
            .iter()
            .map(|&x| f32::from_bits(u32::from(bf(x + (rnd() as f32 - 0.5) * 1.5)) << 16))
            .collect();
        for &t in &[0.7f32, 1.0] {
            let sp = SamplingParams {
                temperature: t,
                top_k: 20,
                top_p: 0.95,
                ..SamplingParams::default()
            };
            let params = RowParams::resolve(&sp, v);
            // the target's processed law (one dense draw's probabilities, by counting below)
            let n = 40_000u64;
            let mut hist = vec![0u64; v];
            let (mut acc_c, mut acc_g, mut acc_same) = (0u64, 0u64, 0u64);
            let greedy = (0..v)
                .max_by(|&a, &b| ld[a].total_cmp(&ld[b]).then(b.cmp(&a)))
                .unwrap();
            for s in 0..n {
                let (seed, pos) = (s / 8, s % 8);
                let mut row = lt.clone();
                let mut nz = SeededNoise::new(seed, pos, 0);
                let tok = sampler
                    .sample_dense(&mut row, &params, &PenaltyContext::default(), &mut nz)
                    .unwrap()
                    .token_id as usize;
                hist[tok] += 1;
                let cr = couple_row(&params, seed, 0, pos);
                let pick = |l: &[f32]| {
                    let c0: Vec<(u32, u16)> = (0..half).map(|i| (i, bf(l[i as usize]))).collect();
                    let c1: Vec<(u32, u16)> =
                        (0..half).map(|i| (i, bf(l[(half + i) as usize]))).collect();
                    // the round drafts pass `pass` at cur_pos + acc + 1 + pass: here p1 = pos
                    let mut r = cr;
                    r[2] = pos as u32;
                    pair_pick(
                        rank_pick(&c0, &r, 0, 0, half, &f32::ln),
                        rank_pick(&c1, &r, 0, 1, half, &f32::ln),
                        half,
                    ) as usize
                };
                acc_c += u64::from(pick(&ld) == tok);
                acc_same += u64::from(pick(&lt) == tok);
                acc_g += u64::from(greedy == tok);
            }
            // chi-square vs the processed law (probabilities from the target's own filtered softmax)
            let mut l: Vec<(usize, f32)> = lt.iter().map(|&x| x / t).enumerate().collect();
            l.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            l.truncate(20);
            let m = l[0].1;
            let e: Vec<f64> = l.iter().map(|&(_, x)| f64::from((x - m).exp())).collect();
            let z: f64 = e.iter().sum();
            let mut keep = Vec::new();
            let mut cum = 0.0;
            for (k, &(i, _)) in l.iter().enumerate() {
                keep.push((i, e[k] / z));
                cum += e[k] / z;
                if cum >= 0.95 {
                    break;
                }
            }
            let zk: f64 = keep.iter().map(|x| x.1).sum();
            let mut chi = 0.0;
            let mut df = 0usize;
            let mut outside = 0u64;
            let kept: Vec<usize> = keep.iter().map(|x| x.0).collect();
            for (i, &h) in hist.iter().enumerate() {
                if !kept.contains(&i) {
                    outside += h;
                }
            }
            for &(i, p) in &keep {
                let ex = n as f64 * p / zk;
                if ex >= 5.0 {
                    chi += (hist[i] as f64 - ex).powi(2) / ex;
                    df += 1;
                }
            }
            df = df.saturating_sub(1);
            // the 0.999 quantile of chi-square at df <= 20 is below 2 df + 26
            assert!(
                chi < 2.0 * df as f64 + 26.0,
                "T {t}: chi-square {chi:.1} at df {df}"
            );
            assert_eq!(outside, 0, "T {t}: draws outside the processed support");
            let (ac, ag, a1) = (
                acc_c as f64 / n as f64,
                acc_g as f64 / n as f64,
                acc_same as f64 / n as f64,
            );
            eprintln!(
                "T {t}: acceptance greedy {ag:.3} coupled {ac:.3} (drafter == target {a1:.3})"
            );
            assert!(ac > ag + 0.03, "T {t}: coupled {ac:.3} vs greedy {ag:.3}");
            // a drafter EQUAL to the target misses only where its race lands outside the target's
            // support: per-rank top_k (the union is wider than the global top_k) and no top_p
            assert!(
                a1 > 0.8 && a1 > ac,
                "T {t}: a drafter equal to the target accepts {a1:.3}"
            );
        }
    }
}
