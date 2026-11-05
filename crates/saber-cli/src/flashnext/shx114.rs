//! FN114c: the TARGET's shared expert balanced across the two EP ranks (`--fn-shared-expert
//! split|rank0`; LEDGER FN114 / FN114c).
//!
//! `rank0` (THE DEFAULT, today: `Dims::shared = rank == 0`) runs the whole shared expert on rank 0
//! only: rank 0's MoE back half is +8 us a layer at b1 / +12.6 at b8 over rank 1's, and the
//! post-MoE reduce waits for it (FN114's census: the MoE-out skew tracks the back half's rank
//! difference at corr +0.99). `split` (the NUMERICS arm) gives each EP rank HALF h = rank of the
//! shared expert's intermediate dimension -- FN80a's slicer, the TP2 form's
//! [`super::xtp2::shared_expert_half`] (gate_up rows `[320h, +320)` + `[640 + 320h, +320)`, down K
//! `[320h, +320)`, `down.wref` the whole row's) -- computed inside the fused back half
//! `moe_back_ss_*` (`moe_grouped_ss`: FN26's routed tiles, the shared tile at FN80b-1's half shape,
//! one launch as today); the two halves meet in the post-MoE all-reduce the layer already pays.
//!
//! WHY NUMERICS: today rank 0's partial is `bf16(shared + routed_0)` and rank 1's `bf16(routed_1)`;
//! split makes them `bf16(shared_0 + routed_0)` / `bf16(shared_1 + routed_1)` -- the same sum in
//! exact arithmetic ([`tests::the_halves_sum_to_the_whole_exactly`]) but another fp32 grouping,
//! another silu-quant scale (the half's own row amax) and other partials on the 6-bit WHT wire.
//!
//! SCOPE: the TARGET's MoE blocks on the fused decode / verify train (T <= 64, `--fn-moe-train
//! fused`, the default). Prefill chunks (the seven-launch train / the pf tiles) and the MTP
//! drafter's own shared expert (`Dims::mtp_experts`, the slab4 table's entry 256 on rank 0) keep
//! rank 0's whole expert, so rank 0 keeps the whole record AND adds its half
//! ([`rank_extra_bytes`]); rank 1 uploads its half inside the whole's bill (every rank is billed
//! the `shared/` classes; rank 1 uploaded none of it until now).

use anyhow::ensure;
use saber_hip::{HipThread, Stream};

use super::model::Owned;
use super::run::FnFlags;
use super::store::PackRecs;

/// `--fn-shared-expert`: `rank0` = today and the default (the flip is the user's word).
pub const DEFAULT_FN_SHARED_EXPERT: &str = "rank0";

/// Layers with a shared expert (the target's 48).
pub const LAYERS: u32 = 48;

/// The half tables' bytes (48 x [`super::exec::SHX_TABLE_BYTES`], one allocation a rank).
pub const TABLES_BYTES: u64 = LAYERS as u64 * super::exec::SHX_TABLE_BYTES;

/// One half record's payload (FN80a's `record_layout(2560, 320)`).
#[must_use]
pub fn half_bytes() -> u64 {
    super::xsplit::half_layout().1
}

/// Bytes rank `r` adds to the `fn.w.shared.shared_expert` class under `split` (`whole` = the
/// class as planned): rank 0 keeps the whole and adds its half + the tables; rank 1's half + the
/// tables fit inside the whole it is already billed.
#[must_use]
pub fn rank_extra_bytes(r: u32, whole: u64) -> u64 {
    let mine = u64::from(LAYERS) * half_bytes() + TABLES_BYTES;
    if r == 0 {
        mine
    } else {
        mine.saturating_sub(whole)
    }
}

/// The half this rank computes under `split` (`None` = `rank0`: rank 0 whole, rank 1 nothing).
#[must_use]
pub fn half_of(f: &FnFlags, rank: u32) -> Option<u32> {
    f.shared_split().then_some(rank)
}

/// Refuse `split` where it cannot run: TP=2 EP (tp2 already splits every expert), the worker
/// form, the fused train (its back half is the only one built) at the id miss order.
pub fn check(f: &FnFlags, tp: u32) -> anyhow::Result<()> {
    if !f.shared_split() {
        return Ok(());
    }
    ensure!(
        tp == 2,
        "--fn-shared-expert split is a TP=2 EP form (--tp 2)"
    );
    ensure!(
        !f.experts_tp2(),
        "--fn-shared-expert split: --fn-experts tp2 already splits the shared expert"
    );
    ensure!(
        !f.fn_one_process,
        "--fn-shared-expert split runs in the worker form (not --fn-one-process)"
    );
    ensure!(
        f.fn_moe_train == super::exec::MoeTrain::Fused,
        "--fn-shared-expert split needs --fn-moe-train fused (moe_back_ss_* is the fused back half)"
    );
    ensure!(
        f.fn_moe_miss_order == "id",
        "--fn-shared-expert split: the FN84 miss order has no shared-split back half \
         (--fn-moe-miss-order id)"
    );
    Ok(())
}

/// Upload rank `rank`'s half of every layer's shared expert (FN80a's slicer, the join checked)
/// and the 48 half tables (6 u64 a layer, one allocation); returns the tables' device address.
pub fn load(
    recs: &PackRecs,
    owned: &mut Owned,
    hip: &HipThread,
    st: &Stream,
    rank: u32,
) -> anyhow::Result<u64> {
    let mut tbl = Vec::with_capacity(TABLES_BYTES as usize);
    for l in 0..LAYERS {
        let v = super::xtp2::shared_expert_half(recs, owned, hip, st, l, rank)?;
        for a in v {
            tbl.extend_from_slice(&a.to_le_bytes());
        }
    }
    ensure!(tbl.len() as u64 == TABLES_BYTES, "the half tables' bytes");
    owned.up(hip, st, &tbl)
}

#[cfg(test)]
mod tests {
    use super::super::xsplit::{self, HIDDEN, INTER, INTER_H};
    use super::super::xtp2::slice_half;
    use saber_cpux::{e4m3_value, folded_byte, MxGemm};
    use saber_mem::xtier::record_layout;

    fn fnx(args: &[&str]) -> super::FnFlags {
        #[derive(clap::Parser)]
        struct C {
            #[command(flatten)]
            f: super::FnFlags,
        }
        let mut v = vec!["saber"];
        v.extend_from_slice(args);
        <C as clap::Parser>::try_parse_from(v).unwrap().f
    }

    /// The flag resolves per rank: `rank0` = no half anywhere (today), `split` = half r on rank r;
    /// the bill (rank 0 whole + half + tables, rank 1 inside the whole); the refusals.
    #[test]
    fn fn114c_the_flag_resolves_per_rank() {
        let d = fnx(&[]);
        assert_eq!(d.fn_shared_expert, super::DEFAULT_FN_SHARED_EXPERT);
        assert!(!d.shared_split());
        assert_eq!((super::half_of(&d, 0), super::half_of(&d, 1)), (None, None));
        assert!(super::check(&d, 2).is_ok());
        let s = fnx(&["--fn-shared-expert", "split", "--fn-moe-train", "fused"]);
        assert!(s.shared_split());
        assert_eq!(
            (super::half_of(&s, 0), super::half_of(&s, 1)),
            (Some(0), Some(1))
        );
        assert!(super::check(&s, 2).is_ok());
        // the class as planned: 48 whole records (FN2's record_layout(2560, 640) payload)
        let whole = 48 * record_layout(HIDDEN, INTER).1;
        let half = 48 * xsplit::half_layout().1;
        assert_eq!(
            super::rank_extra_bytes(0, whole),
            half + super::TABLES_BYTES
        );
        assert_eq!(super::rank_extra_bytes(1, whole), 0);
        assert!(
            half + super::TABLES_BYTES <= whole,
            "rank 1's half fits the whole's bill"
        );
        for bad in [
            vec!["--fn-shared-expert", "split", "--fn-experts", "tp2"],
            vec!["--fn-shared-expert", "split", "--fn-one-process"],
            vec!["--fn-shared-expert", "split", "--fn-moe-train", "legacy"],
            vec![
                "--fn-shared-expert",
                "split",
                "--fn-moe-miss-order",
                "first",
            ],
        ] {
            assert!(super::check(&fnx(&bad), 2).is_err(), "{bad:?}");
        }
        assert!(super::check(&s, 1).is_err(), "TP=1");
    }

    /// A deterministic byte stream (no rand dependency).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u8 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 56) as u8
        }
    }

    /// A synthetic WHOLE shared-expert record: random nibbles, block scales within 15 binades of
    /// the row's reference (every fold in range), references near 127 (rf in [2^-2, 1]).
    fn synthetic_whole(seed: u64) -> Vec<u8> {
        let (ss, payload, _) = record_layout(HIDDEN, INTER);
        let mut r = Lcg(seed);
        let mut b = vec![0u8; payload as usize];
        let (gu_n, dn_n) = ((2 * INTER) as usize, HIDDEN as usize);
        // wref first (the ws draws read it)
        for (i, n) in [(2usize, gu_n), (5, dn_n)] {
            for j in 0..n {
                b[ss[i].0 as usize + j] = 125 + r.next() % 3;
            }
        }
        for (iw, is, ir, n) in [(0usize, 1usize, 2usize, gu_n), (3, 4, 5, dn_n)] {
            for x in &mut b[ss[iw].0 as usize..(ss[iw].0 + ss[iw].1) as usize] {
                *x = r.next();
            }
            let blocks = ss[is].1 as usize / n;
            for blk in 0..blocks {
                for j in 0..n {
                    let wref = b[ss[ir].0 as usize + j];
                    b[ss[is].0 as usize + blk * n + j] = wref - r.next() % 6;
                }
            }
        }
        b
    }

    fn gemm<'a>(rec: &'a [u8], ss: &[(u64, u64); 6], down: bool, n: usize, k: usize) -> MxGemm<'a> {
        let o = if down { 3 } else { 0 };
        let s = |i: usize| &rec[ss[o + i].0 as usize..(ss[o + i].0 + ss[o + i].1) as usize];
        MxGemm {
            w: s(0),
            ws: s(1),
            wref: s(2),
            n,
            k,
        }
    }

    /// The weight value the card multiplies: the folded e4m3 byte times the epilogue's rf.
    fn wv(g: &MxGemm, n: usize, k: usize) -> f64 {
        e4m3_value(folded_byte(g.nibble(n, k), g.shift(n, k / 32))) * f64::from(g.rf(n))
    }

    /// THE IDENTITY THE ARM RESTS ON: the halves' rows / columns are the whole's (gate_up row j of
    /// half h == whole row `gu_row(h, j)`, down column j of half h == whole column 320h + j, every
    /// fold and rf exact), so in exact arithmetic `down_0 . act_0 + down_1 . act_1 == down . act`
    /// for every output row -- checked BIT FOR BIT in fp64 on activations on the e4m3 grid (every
    /// product on a 2^-20 grid, every partial sum < 2^28: fp64 adds them exactly). Also the fp32
    /// reading (the card's accumulator width): the two rounded partials' rounded sum within 3 fp32
    /// epsilons of the whole, relative to |p0| + |p1|. NEGATIVE CONTROL: half 0's columns paired
    /// with half 1's activations differ.
    #[test]
    fn the_halves_sum_to_the_whole_exactly() {
        let (ss, _, _) = record_layout(HIDDEN, INTER);
        let (ds, _, _) = xsplit::half_layout();
        let whole = synthetic_whole(0x0114c);
        let halves = [
            slice_half(&whole, 0).unwrap().0,
            slice_half(&whole, 1).unwrap().0,
        ];
        let (h_, i_, ih) = (HIDDEN as usize, INTER as usize, INTER_H as usize);
        // gate_up: every half row == its whole row (all 2,560 k)
        let gw = gemm(&whole, &ss, false, 2 * i_, h_);
        for (h, hb) in halves.iter().enumerate() {
            let gh = gemm(hb, &ds, false, 2 * ih, h_);
            for j in (0..2 * ih).step_by(7) {
                let wr = xsplit::gu_row(h as u64, j as u64) as usize;
                for k in (0..h_).step_by(3) {
                    assert_eq!(
                        wv(&gh, j, k).to_bits(),
                        wv(&gw, wr, k).to_bits(),
                        "gu h{h} j{j} k{k}"
                    );
                }
            }
        }
        // the activations (act_i = silu(g_i) * u_i, then the e4m3 grid; equal on both forms
        // because the gate_up rows are): synthetic e4m3 values over the whole intermediate
        let mut r = Lcg(7);
        let act: Vec<f64> = (0..i_)
            .map(|_| {
                let v = e4m3_value(r.next() & 0x7E);
                if r.next() & 1 == 1 {
                    -v
                } else {
                    v
                }
            })
            .collect();
        let dw = gemm(&whole, &ss, true, h_, i_);
        let dh: Vec<MxGemm> = halves
            .iter()
            .map(|hb| gemm(hb, &ds, true, h_, ih))
            .collect();
        let (mut differs_ctl, mut f32_worst) = (0usize, 0f64);
        for n in 0..h_ {
            let y: f64 = (0..i_).map(|i| wv(&dw, n, i) * act[i]).sum();
            let p: Vec<f64> = (0..2)
                .map(|h| (0..ih).map(|j| wv(&dh[h], n, j) * act[h * ih + j]).sum())
                .collect();
            assert_eq!(
                (p[0] + p[1]).to_bits(),
                y.to_bits(),
                "row {n}: halves != whole"
            );
            // fp32: each partial rounded once, summed once vs the whole rounded once
            let y32 = y as f32;
            let s32 = p[0] as f32 + p[1] as f32;
            let mag = (p[0].abs() + p[1].abs()).max(1e-30);
            f32_worst = f32_worst.max(f64::from((y32 - s32).abs()) / mag);
            // the control: half 0's columns against half 1's activations
            let c: f64 = (0..ih).map(|j| wv(&dh[0], n, j) * act[ih + j]).sum::<f64>() + p[1];
            differs_ctl += usize::from(c.to_bits() != y.to_bits());
        }
        assert!(
            f32_worst <= 3.0 * f64::from(f32::EPSILON),
            "fp32 reading {f32_worst}"
        );
        assert!(
            differs_ctl > h_ * 9 / 10,
            "the control must differ: {differs_ctl} of {h_}"
        );
    }
}
