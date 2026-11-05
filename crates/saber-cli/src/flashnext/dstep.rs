//! FN119: THE DEVICE-RESIDENT STEP, round 1 -- `--fn-device-step <form>` (absent = `lazy`, THE DEFAULT
//! since the orchestrator's flip 2026-10-13 on h1 + h2; `off` = the pre-FN119 host law, the control).
//!
//! The finding this module is built on (LEDGER FN119): the host sampler's DRAW never reads a row's
//! log-sum-exp under the served sampling law. `saber_serve::sampler::Sampler::sample` reads the
//! T = 1 log-sum-exp only for the logprobs it reports, and the tempered one (`lse_t`) only when
//! `top_p < 1` meets a tail that survived `top_k` / `min_p` -- with the checkpoint's own
//! `top_k = 20` (and BetterBench's) inside a 64-wide seam the tail is dead before `top_p` unless the
//! 20th..64th values all tie. Yet every sampled row computed BOTH (FN117's 0.50 / 1.27 ms "host
//! log-sum-exps", the largest term of the hand) and read its whole row (248 KB) over the link.
//!
//! Forms:
//!
//! * `off` -- the pre-FN119 host law (the control).
//! * `lazy` (THE DEFAULT) -- BIT-EXACT by construction: a decode row computes the T = 1 log-sum-exp only when its
//!   request asks for logprobs, and DEFERS `lse_t` when the tail can only survive through a tie
//!   (`top_k < 64`, no `min_p`, no penalties); after the merge both quantities are present for every
//!   row whose sampler can read them: a deferred row whose merged seam has `v[k-1] / T <= v[63] / T`
//!   (the sampler's own comparison: the tail lives) is LATE and its `lse_t` is computed from its
//!   dense halves by the same function on the same bytes (`Cmd::Dense`) before the walk. A row the
//!   sampler never reads a log-sum-exp of is never read whole.
//! * `lazy-late` -- the control of `lazy`'s fill: EVERY deferred row is late (the fill path's bytes
//!   gated against `off` on served traffic).
//! * `check` -- `lazy` + the device sampler + acceptance kernel (`fn119_draw_walk`) run on rank 0
//!   beside the host walk on every decode step and COMPARED (a seeded or greedy sequence the kernel
//!   did not refuse must commit the host walk's tokens, or the boot fails loud): the round-2
//!   scaffold's law on served traffic.

use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{ensure, Context};
use saber_serve::sampler::RowParams;

use super::worker::{RowOut, RowReq, SEAM_K};

/// `--fn-device-step` forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceStep {
    /// Today's host path (the control).
    Off,
    /// Log-sum-exps only where the sampler reads them (bit-exact; round 1's default, now `ahead`'s
    /// control).
    Lazy,
    /// `lazy` with every deferred row filled late (the fill path's control).
    LazyLate,
    /// `lazy` + the device draw / walk kernel compared with the host walk every step.
    Check,
    /// `lazy` + THE ISSUE-AHEAD (round 2): at an eligible b1 verify step the device walk runs on
    /// both ranks (the seam halves crossed over the P2P window) and the drafter round's
    /// acc-dependent inputs are written on the card, so the round issues behind the verify before
    /// the host waits; the host's walk checks the card's (fail loud) and a refused / mismatched
    /// step re-runs the round on the host's answer. The default FN119b -> FN119d; now `next`'s
    /// control.
    Ahead,
    /// `ahead` + THE NEXT VERIFY EARLY (round 4, FN119d): the commit does not wait for the round's
    /// ids -- its counts are all the next verify's tables, positions and state indices need, so
    /// the next step's host inputs are built and its layers before the PLE layer issued behind the
    /// round while it runs; the drafts' embedding rows are gathered on the card from the chain's
    /// ids, and the host reads the ids (one async copy behind the round) only before the drafts'
    /// PLE rows. Any other command first resolves the round as `ahead` does. THE DEFAULT since
    /// FN119d (2026-10-13, the pooled 4 + 4 in both orders: bytes identical, b1 hand 0.81 -> 0.52 ms
    /// p50, step wall b1 -0.28 / b8 -0.67 ms p50, update p99 -0.55, conc-8 +1.8 %).
    #[default]
    Next,
}

/// FN119: `--fn-device-step` when absent -- `next` IS THE DEFAULT (FN119d, the pooled 4 + 4 in both
/// orders 2026-10-13: bytes identical, b1 hand 0.81 -> 0.52 ms p50, update p99 -0.55); `ahead` =
/// its control (FN119b's default: b1 hand 1.06 -> 0.80), `lazy` = round 1's, `off` = the pre-FN119
/// host law. A boot without the device chain / pick / FN112 runs `ahead` / `next` as `lazy`.
pub const DEFAULT_FN_DEVICE_STEP: DeviceStep = DeviceStep::Next;

impl FromStr for DeviceStep {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "off" => Ok(Self::Off),
            "lazy" => Ok(Self::Lazy),
            "lazy-late" => Ok(Self::LazyLate),
            "check" => Ok(Self::Check),
            "ahead" => Ok(Self::Ahead),
            "next" => Ok(Self::Next),
            x => Err(format!(
                "--fn-device-step {x}: expected off | lazy | lazy-late | check | ahead | next"
            )),
        }
    }
}

impl std::fmt::Display for DeviceStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Off => "off",
            Self::Lazy => "lazy",
            Self::LazyLate => "lazy-late",
            Self::Check => "check",
            Self::Ahead => "ahead",
            Self::Next => "next",
        })
    }
}

impl DeviceStep {
    /// The lazy log-sum-exp law is on.
    #[must_use]
    pub fn lazy(self) -> bool {
        !matches!(self, Self::Off)
    }

    /// The issue-ahead (round 2's device walk + the round behind the verify) is on.
    #[must_use]
    pub fn ahead(self) -> bool {
        matches!(self, Self::Ahead | Self::Next)
    }

    /// FN119d: the next verify is built and issued under the round (the commit leaves the round's
    /// ids pending).
    #[must_use]
    pub fn next(self) -> bool {
        matches!(self, Self::Next)
    }
}

/// `lazy`: what a decode row's request asks of the pick. `rq` is `serve::row_req`'s; `params` its
/// resolved knobs; `pen` its penalty ids. The T = 1 log-sum-exp only for logprobs; `lse_t` deferred
/// (`rq.tk` = the row's `top_k`) when only a tie at the cut can keep the tail alive.
pub fn shape(rq: &mut RowReq, params: &RowParams, pen: &[u32]) {
    rq.lse = params.logprobs.is_some();
    if rq.temp > 0.0
        && pen.is_empty()
        && params.min_p <= 0.0
        && params.top_k >= 1
        && params.top_k < SEAM_K
    {
        rq.tk = params.top_k as u32;
    }
}

/// `lazy`: whether a pick computes the T = 1 log-sum-exp / `lse_t` of a row (the frame's law;
/// `rq.tk == 0` is today's, byte for byte).
#[must_use]
pub fn wants(rq: &RowReq) -> (bool, bool) {
    let t = rq.temp > 0.0 && rq.tk == 0;
    (rq.lse || t, t)
}

/// `lazy`: the deferred rows whose tail can live after `top_k` (their `lse_t` must be filled
/// before the walk): the sampler's own test `threshold > ceiling` with `threshold = v[k-1] / T`,
/// `ceiling = v[63] / T` on the merged seam (no penalties, no `min_p`: the candidates are the
/// seam, non-increasing). Any NaN in the seam is late. `force` = every deferred row (the control).
#[must_use]
pub fn late_rows(outs: &[RowOut], reqs: &[RowReq], force: bool) -> Vec<usize> {
    let mut v = Vec::new();
    for (r, (o, q)) in outs.iter().zip(reqs).enumerate() {
        if !(q.temp > 0.0 && q.tk > 0) {
            continue;
        }
        let k = q.tk as usize;
        let dead = !force
            && o.top.len() == SEAM_K
            && o.top.iter().all(|x| !x.1.is_nan())
            && o.top[k - 1].1 / q.temp > o.top[SEAM_K - 1].1 / q.temp;
        if !dead {
            v.push(r);
        }
    }
    v
}

/// `lazy`: a late row's `lse_t` from its dense row (both halves, f32 of bf16, rank 0's half first):
/// each half's log-sum-exp by the pick's own function (`row_lse_memo` about the half's seam maximum
/// -- the first entry of `seam_top_k(half, 1)`, the frame's `mx`), merged as `pick2` merges.
#[must_use]
pub fn fill_lse_t(dense: &[f32], temp: f32) -> f32 {
    let half = dense.len() / 2;
    let lse_half = |h: &[f32]| {
        let b: Vec<u8> = h
            .iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let mx = super::worker::seam_top_k(&b, 1)[0].1;
        super::worker::row_lse_memo(&b, mx, temp)
    };
    let a = lse_half(&dense[..half]);
    let b = lse_half(&dense[half..]);
    saber_serve::sampler::log_add_exp(a, b)
}

/// The counters of the forms (rank 0's end line `fndstep119`).
#[derive(Debug, Default, Clone)]
pub struct Counts {
    pub rows: u64,
    pub deferred: u64,
    pub late: u64,
    pub fills: u64,
    pub lse1: u64,
    pub check_steps: u64,
    pub check_seqs: u64,
    pub check_equal: u64,
    pub check_refused: u64,
    pub check_skipped: u64,
    pub check_ms: f64,
}

impl std::fmt::Display for Counts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "decode rows {} (lse at T=1 asked {}), lse_t deferred {}, late {} ({} dense fills); \
             check: steps {}, sequences {}, equal {}, refused by the kernel {}, skipped (unseeded) \
             {}, host ms {:.1}",
            self.rows,
            self.lse1,
            self.deferred,
            self.late,
            self.fills,
            self.check_steps,
            self.check_seqs,
            self.check_equal,
            self.check_refused,
            self.check_skipped,
            self.check_ms
        )
    }
}

// ---------------------------------------------------------------- the device twin's host half

/// glibc 2.39's `expf` (the FMA build, ARM optimized-routines), op for op -- the kernel's
/// `fn119_expf`. Gated exhaustively against the libm this binary links (`saber fn119-oracle
/// --mode expf`).
#[must_use]
pub fn glibc_expf(x: f32) -> f32 {
    const T: [u64; 32] = [
        0x3ff0_0000_0000_0000,
        0x3fef_d9b0_d315_8574,
        0x3fef_b558_6cf9_890f,
        0x3fef_9301_d012_5b51,
        0x3fef_72b8_3c7d_517b,
        0x3fef_5487_3168_b9aa,
        0x3fef_387a_6e75_6238,
        0x3fef_1e9d_f51f_dee1,
        0x3fef_06fe_0a31_b715,
        0x3fee_f1a7_373a_a9cb,
        0x3fee_dea6_4c12_3422,
        0x3fee_ce08_6061_892d,
        0x3fee_bfda_d536_2a27,
        0x3fee_b42b_569d_4f82,
        0x3fee_ab07_dd48_5429,
        0x3fee_a47e_b03a_5585,
        0x3fee_a09e_667f_3bcd,
        0x3fee_9f75_e8ec_5f74,
        0x3fee_a114_73eb_0187,
        0x3fee_a589_994c_ce13,
        0x3fee_ace5_422a_a0db,
        0x3fee_b737_b0cd_c5e5,
        0x3fee_c491_82a3_f090,
        0x3fee_d503_b23e_255d,
        0x3fee_e89f_995a_d3ad,
        0x3fee_ff76_f2fb_5e47,
        0x3fef_199b_dd85_529c,
        0x3fef_3720_dcef_9069,
        0x3fef_5818_dcfb_a487,
        0x3fef_7c97_337b_9b5f,
        0x3fef_a4af_a2a4_90da,
        0x3fef_d076_5b6e_4540,
    ];
    let ux = x.to_bits();
    let abstop = (ux >> 20) & 0x7ff;
    if abstop >= (0x42b0_0000 >> 20) {
        if ux == 0xff80_0000 {
            return 0.0;
        }
        if abstop >= (0x7f80_0000 >> 20) {
            return x + x;
        }
        if x > f32::from_bits(0x42b1_7217) {
            return f32::INFINITY;
        }
        if x < f32::from_bits(0xc2cf_f1b4) {
            return 0.0;
        }
    }
    let n = 32.0f64;
    let inv = f64::from_bits(0x3ff7_1547_652b_82fe) * n;
    let c0 = f64::from_bits(0x3fac_6af8_4b91_2394) / n / n / n;
    let c1 = f64::from_bits(0x3fce_bfce_50fa_c4f3) / n / n;
    let c2 = f64::from_bits(0x3fe6_2e42_ff0c_52d6) / n;
    let shift = f64::from_bits(0x4338_0000_0000_0000);
    let xd = f64::from(x);
    let z = inv * xd;
    let mut kd = z + shift;
    let ki = kd.to_bits();
    kd -= shift;
    // the FMA build contracts `z - kd` with z's product: r = fma(inv, xd, -kd) (found by the
    // exhaustive gate: the plain form differs from the image's libm on 2 of 2^32 patterns)
    let r = inv.mul_add(xd, -kd);
    let s = f64::from_bits(T[(ki & 31) as usize].wrapping_add(ki << 47));
    let zz = c0.mul_add(r, c1);
    let r2 = r * r;
    let y = c2.mul_add(r, 1.0);
    let y = zz.mul_add(r2, y);
    (y * s) as f32
}

/// `splitmix64` (the sampler's mixing function, the kernel's `fn119_mix`).
#[must_use]
pub fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `SeededNoise::new(seed, position, choice)`'s key (the kernel draws `E(id)` from it).
#[must_use]
pub fn noise_key(seed: u64, position: u64, choice: u32) -> u64 {
    mix(seed)
        ^ mix(position.wrapping_mul(0x00FF_51AF_D7ED_558D))
        ^ mix(u64::from(choice).wrapping_mul(0xC4CE_B9FE_1A85_EC53))
}

/// The expf probe's hash of one bit pattern (NaNs canonical): the kernel's per-pattern term.
#[must_use]
pub fn expf_term(u: u32, r: u32) -> u64 {
    let r = if (r & 0x7f80_0000) == 0x7f80_0000 && (r & 0x7f_ffff) != 0 {
        0x7fc0_0000
    } else {
        r
    };
    mix((u64::from(u) << 32) | u64::from(r))
}

/// One device row-parameter block (8 u32): T, top_p, min_p, top_k, flags (bit0 greedy, bit1 the
/// host decides, bit2 top_k cuts), seam row, key lo, key hi.
#[must_use]
pub fn row_block(p: &RowParams, host_only: bool, seam_row: u32, key: u64) -> [u32; 8] {
    let mut fl = 0u32;
    if p.is_greedy() {
        fl |= 1;
    }
    if host_only {
        fl |= 2;
    }
    if p.top_k < p.vocab_size {
        fl |= 4;
    }
    [
        p.temperature.to_bits(),
        p.top_p.to_bits(),
        p.min_p.to_bits(),
        p.top_k.min(u32::MAX as usize) as u32,
        fl,
        seam_row,
        key as u32,
        (key >> 32) as u32,
    ]
}

// ---------------------------------------------------------------- the check form (rank 0)

/// The device twin's code object and symbols.
pub const DSTEP_PLUGIN: &str = "flashnext_dstep";
/// The draw + walk kernel.
pub const DRAW_SYMBOL: &str = "fn119_draw_walk";
/// The kernel's waves a sequence (`exec::WMAX`).
pub const KW: usize = 5;

/// One sequence of a decode step as the check form compares it.
pub struct CheckSeq<'a> {
    /// Its first row in the step's pick (slot x w).
    pub row0: usize,
    pub w: usize,
    /// The drafts the walk compared (none when the slot may not draft).
    pub draft: &'a [u32],
    pub sampling: &'a saber_sched::SlotSampling,
    /// The host walk's committed tokens and whether it read a dense row.
    pub host_tokens: &'a [u32],
    pub host_dense: bool,
}

/// `--fn-device-step check`: rank 0's buffers (one device allocation: the peer half seams, the
/// row blocks, the sequence blocks, the result blocks, the per-row results).
pub struct Check {
    buf: saber_hip::DevicePtr,
    rows_max: usize,
    seqs_max: usize,
}

impl Check {
    const ROW_SEAM: usize = SEAM_K * 8;

    fn offs(rows: usize, seqs: usize) -> [usize; 6] {
        let al = |x: usize| x.div_ceil(256) * 256;
        let b = 0;
        let rp = al(b + rows * Self::ROW_SEAM);
        let sq = al(rp + rows * 32);
        let out = al(sq + seqs * 32);
        let ro = al(out + seqs * 64);
        let end = al(ro + rows * 8);
        [b, rp, sq, out, ro, end]
    }

    /// The buffers for `MAX_SEQS x WMAX` rows; the kernel must be loaded.
    ///
    /// # Errors
    /// The allocation, or `flashnext_dstep::fn119_draw_walk` not loaded.
    pub fn new(
        hip: &saber_hip::HipThread,
        table: &saber_exec::loader::FunctionTable<'_>,
    ) -> anyhow::Result<Self> {
        let seqs_max = super::worker::MAX_SEQS as usize;
        let rows_max = seqs_max * KW;
        table
            .get(DSTEP_PLUGIN, DRAW_SYMBOL)
            .with_context(|| format!("{DSTEP_PLUGIN}::{DRAW_SYMBOL} not loaded"))?;
        let o = Self::offs(rows_max, seqs_max);
        let buf = hip.malloc(o[5] as u64)?;
        Ok(Self {
            buf,
            rows_max,
            seqs_max,
        })
    }

    /// Run the kernel on this step's rows (rank 0's own half seams at `seam_a`, the peer's half
    /// rebuilt from the merged rows: every merged entry of the peer's id range, padded with the
    /// lowest key -- the union's top 64 is unchanged) and compare with the host walks.
    ///
    /// # Errors
    /// A HIP failure, or a sequence the kernel did not refuse whose tokens differ from the host
    /// walk's (FAIL LOUD: the twin's law is broken).
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        hip: &saber_hip::HipThread,
        stream: &saber_hip::Stream,
        table: &saber_exec::loader::FunctionTable<'_>,
        seam_a: u64,
        outs: &[RowOut],
        seqs: &[CheckSeq<'_>],
        c: &mut Counts,
    ) -> anyhow::Result<()> {
        let rows = seqs.iter().map(|s| s.row0 + s.w).max().unwrap_or(0);
        ensure!(
            rows <= self.rows_max && seqs.len() <= self.seqs_max && rows <= outs.len(),
            "FN119 check: {rows} rows / {} sequences past the buffers",
            seqs.len()
        );
        let o = Self::offs(self.rows_max, self.seqs_max);
        let half = super::serve::VOCAB as u32 / 2;
        let mut img = vec![0u8; o[5]];
        // rank 0 holds ids [0, half): the peer's half is every merged id >= half
        for (r, ro) in outs.iter().enumerate().take(rows) {
            let mut k = 0;
            let at = o[0] + r * Self::ROW_SEAM;
            for &(id, v) in &ro.top {
                if id >= half {
                    img[at + 8 * k..at + 8 * k + 4].copy_from_slice(&id.to_le_bytes());
                    img[at + 8 * k + 4..at + 8 * k + 8].copy_from_slice(&v.to_bits().to_le_bytes());
                    k += 1;
                }
            }
            for p in k..SEAM_K {
                // key 0 (bf16 0xFFFF): below every real entry; ids distinct and out of range
                let id = 0xFFFF_0000u32 + p as u32;
                img[at + 8 * p..at + 8 * p + 4].copy_from_slice(&id.to_le_bytes());
                img[at + 8 * p + 4..at + 8 * p + 8].copy_from_slice(&0xFFFF_0000u32.to_le_bytes());
            }
        }
        let mut skipped = vec![false; seqs.len()];
        for (i, s) in seqs.iter().enumerate() {
            let params = RowParams::resolve(&s.sampling.params, super::serve::VOCAB);
            let ctx_pen = s.sampling.penalties.is_some() && !params.no_penalties();
            let seed = s.sampling.params.seed;
            skipped[i] = seed.is_none() && !params.is_greedy();
            for j in 0..s.w {
                let key = seed.map_or(0, |sd| {
                    noise_key(sd, s.sampling.position + j as u64, s.sampling.choice_index)
                });
                let blk = row_block(&params, ctx_pen, (s.row0 + j) as u32, key);
                let at = o[1] + (s.row0 + j) * 32;
                for (q, x) in blk.iter().enumerate() {
                    img[at + 4 * q..at + 4 * q + 4].copy_from_slice(&x.to_le_bytes());
                }
            }
            let nd = s.draft.len().min(s.w.saturating_sub(1));
            let mut sq = [0u32; 8];
            sq[0] = s.row0 as u32;
            sq[1] = s.w as u32;
            sq[2] = nd as u32;
            for (q, &d) in s.draft.iter().take(nd).enumerate() {
                sq[3 + q] = d;
            }
            let at = o[2] + i * 32;
            for (q, x) in sq.iter().enumerate() {
                img[at + 4 * q..at + 4 * q + 4].copy_from_slice(&x.to_le_bytes());
            }
        }
        hip.memcpy_h2d_blocking(&self.buf, &img[..o[3]], stream)?;
        let (func, decl) = table
            .get(DSTEP_PLUGIN, DRAW_SYMBOL)
            .with_context(|| format!("{DSTEP_PLUGIN}::{DRAW_SYMBOL} not loaded"))?;
        ensure!(
            decl.kernarg_segment_size == 52 || decl.kernarg_segment_size == 56,
            "{DRAW_SYMBOL}: the manifest's kernargs are {} B",
            decl.kernarg_segment_size
        );
        let base = self.buf.addr().as_u64();
        let mut args = Vec::with_capacity(56);
        // the kernel's seam row index is the row block's [5]: rows map 1:1 here
        for a in [
            seam_a,
            base + o[0] as u64,
            base + o[1] as u64,
            base + o[2] as u64,
            base + o[3] as u64,
            base + o[4] as u64,
        ] {
            args.extend_from_slice(&a.to_le_bytes());
        }
        args.extend_from_slice(&(seqs.len() as i32).to_le_bytes());
        args.resize(decl.kernarg_segment_size as usize, 0);
        hip.launch(
            func,
            [seqs.len() as u32, 1, 1],
            [(32 * KW) as u32, 1, 1],
            0,
            stream,
            &mut args,
        )?;
        let mut back = vec![0u8; o[5]];
        hip.memcpy_d2h_blocking(&mut back, &self.buf, stream)?;
        let rd = |at: usize| u32::from_le_bytes(back[at..at + 4].try_into().expect("4"));
        c.check_steps += 1;
        for (i, s) in seqs.iter().enumerate() {
            c.check_seqs += 1;
            let at = o[3] + i * 64;
            let fl = rd(at);
            let ntok = rd(at + 8) as usize;
            if skipped[i] {
                c.check_skipped += 1;
                continue;
            }
            if fl != 0 {
                c.check_refused += 1;
                continue;
            }
            let dev: Vec<u32> = (0..ntok).map(|q| rd(at + 12 + 4 * q)).collect();
            ensure!(
                !s.host_dense && dev.as_slice() == s.host_tokens,
                "FN119 check: sequence {i} (rows {}..{}): the device walk committed {dev:?}, the \
                 host {:?} (dense {}): the twin's law is broken",
                s.row0,
                s.row0 + s.w,
                s.host_tokens,
                s.host_dense
            );
            c.check_equal += 1;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------- round 2: the issue-ahead

/// The issue-ahead's code object.
pub const AHEAD_PLUGIN: &str = "flashnext_ahead";
/// FN119c: the b > 1 composition's code object.
pub const AHEAD2_PLUGIN: &str = "flashnext_ahead2";
/// FN119d: the next verify's device embedding rows' code object.
pub const AHEAD3_PLUGIN: &str = "flashnext_ahead3";
/// FN119c: the runs region (u32 triples) after the variants.
pub const RUNS_BYTES: usize = 1 << 20;
/// The window's rows a slot (`MAX_SEQS x WMAX`): every pick row of a step.
pub const AHEAD_ROWS: usize = 40;
/// The flags word's TIMEOUT bit (`F119A_TIMEOUT`): the peer's seam never landed.
pub const AHEAD_TIMEOUT: u32 = 256;

/// The window's offset inside each rank's AR scratch: above FN114b's flag-in-data window (its
/// payload + zero masks + tail end below `ll_scratch_off + 420 KiB`), inside the first slot.
#[must_use]
pub fn win_off(mb: u64) -> u64 {
    super::dev::ll_scratch_off(mb) + (512 << 10)
}

/// The window's bytes (two parity slots of [`AHEAD_ROWS`] seam rows).
pub const WIN_BYTES: u64 = 2 * AHEAD_ROWS as u64 * SEAM_K as u64 * 8;

/// What the leader sends with an issue-ahead step (`Cmd::Decode`): every verify row's device
/// parameter block ([`row_block`], its seam row filled by the rank) and the chain length.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct AheadCmd {
    pub blocks: Vec<[u32; 8]>,
    pub kn: u32,
}

/// One rank's issue-ahead buffers + counters.
pub struct AheadRt {
    /// The call counter (the window slot's parity), zeroed at boot; both ranks advance it once an
    /// issue-ahead step.
    pub ctr: saber_hip::DevicePtr,
    /// rp (rows x 32 B) | sq (8 x 32 B) | out (8 x 64 B) | rowout (rows x 16 B).
    pub dev: saber_hip::DevicePtr,
    /// The upload staging (rp | sq) and the result read-back (out | rowout).
    pub stg: saber_hip::PinnedBuf,
    /// The chain inputs' variants (one an accepted count), read by `fn119a_sel` over UVA.
    pub var: saber_hip::PinnedBuf,
    pub var_stride: usize,
    /// The pass-1 tables' pinned staging (one async copy behind the verify).
    pub cs: saber_hip::PinnedBuf,
    /// Recorded after the walk's result copy (the host waits on it, never on the round).
    pub ev: saber_hip::Event,
    /// The step in flight (set at issue, taken at the commit).
    pub issued: Option<Issued>,
    /// FN119c (b > 1): the composition of the step in flight, checked at the commit.
    pub chk: Option<ComposeChk>,
    /// FN119c: the runs region's offset inside `var` (after the variants).
    pub runs_off: usize,
    /// FN119d (`next`): the chain's ids read back behind the round (pinned) + its event, and the
    /// next step's tables staged for an async upload (pinned, the consts region's size).
    pub early: Option<Early>,
    pub c: AheadCounts,
}

/// FN119d (`next`): the early form's buffers.
pub struct Early {
    pub ids: saber_hip::PinnedBuf,
    pub ids_ev: saber_hip::Event,
    pub tb: saber_hip::PinnedBuf,
    /// FN119e: a first round's result block (the sampled token as the walk's committed one).
    pub out0: saber_hip::PinnedBuf,
    /// FN119f: the prompt's last row's stream read back behind the chunk (the END node's stash) and
    /// its event; the chunk's route log read back behind it (the deferred tier end reads it here:
    /// its rows) and its event; the previous piece's stash staged for the stash row's front.
    pub stash: saber_hip::PinnedBuf,
    pub stash_ev: saber_hip::Event,
    pub plog: saber_hip::PinnedBuf,
    pub plog_ev: saber_hip::Event,
    pub plog_m: Option<u32>,
    pub stin: saber_hip::PinnedBuf,
}

/// FN119d: the device embedding rows checked against the host gather on the first early steps of
/// a boot (each a D2H of the rows: a self-check, not a steady cost).
pub const EARLY_EMB_CHECKS: u64 = 64;

/// FN119d: the round left pending by a commit (both ranks): its chain length, the sequences, the
/// commit's clock (the round's host span closes at the resolve) and its `rd_*` baseline.
#[derive(Debug, Clone)]
pub struct RoundPend {
    pub kn: u32,
    pub n: usize,
    pub bucket: u32,
    pub tm: std::time::Instant,
    pub rd0: std::collections::BTreeMap<&'static str, f64>,
    /// FN119e: a prompt's first round (`Cmd::First`, `--fn-first-step kvnext`), not a commit's.
    pub first: bool,
}

/// An issued step.
#[derive(Debug, Clone, Copy)]
pub struct Issued {
    pub kn: u32,
    /// The drafter round was issued behind the walk (false: the variants' bands / flags differed
    /// -- the commit runs today's round).
    pub round: bool,
}

/// FN119c: the seam rows a step pushes to the peer's window = the rows the peer's walk reads and
/// zeroes (sequence i's rows `i * w .. i * w + w`), never the bucket's pad rows: the window's words
/// are data-as-flag, so a row pushed and not consumed would read as LANDED at the slot's next use.
#[must_use]
pub fn push_rows(n: usize, w: usize) -> usize {
    n * w
}

/// FN119c (b > 1): the chain inputs of every uniform accepted count (variant a = every sequence at
/// a) and the word runs each sequence owns -- what the card composed from, kept for the commit's
/// check (the host's inputs at the walk's counts == this composition, word for word).
#[derive(Debug, Clone, Default)]
pub struct ComposeChk {
    pub pos0: Vec<u32>,
    pub vars: Vec<Vec<u8>>,
    pub runs: Vec<[u32; 3]>,
}

impl ComposeChk {
    /// The composition for per-sequence counts `acc`: variant 0, each sequence's runs from the
    /// variant of its count (the card's `fn119b_copy` + `fn119b_compose`).
    #[must_use]
    pub fn compose(&self, acc: &[u32]) -> Vec<u8> {
        let mut out = self.vars[0].clone();
        for &[s, w0, nw] in &self.runs {
            let v = &self.vars[acc[s as usize] as usize];
            let (a, b) = (w0 as usize * 4, (w0 + nw) as usize * 4);
            out[a..b].copy_from_slice(&v[a..b]);
        }
        out
    }
}

/// FN119c: the word runs each sequence owns, from the uniform variant 0 and the single-sequence
/// variants (`single[s][a - 1]` = sequence s at a, every other at 0): a word one sequence moves is
/// its; a word two sequences move -> `None` (no composition).
#[must_use]
pub fn owner_runs(v0: &[u8], single: &[Vec<Vec<u8>>]) -> Option<Vec<[u32; 3]>> {
    let nw = v0.len() / 4;
    let mut owner: Vec<i32> = vec![-1; nw];
    for (s, vs) in single.iter().enumerate() {
        for v in vs {
            if v.len() != v0.len() {
                return None;
            }
            for (w, o) in owner.iter_mut().enumerate() {
                if v[w * 4..w * 4 + 4] != v0[w * 4..w * 4 + 4] {
                    if *o >= 0 && *o != s as i32 {
                        return None;
                    }
                    *o = s as i32;
                }
            }
        }
    }
    let mut runs: Vec<[u32; 3]> = Vec::new();
    for (w, &o) in owner.iter().enumerate() {
        if o < 0 {
            continue;
        }
        match runs.last_mut() {
            Some(r) if r[0] == o as u32 && r[1] + r[2] == w as u32 => r[2] += 1,
            _ => runs.push([o as u32, w as u32, 1]),
        }
    }
    Some(runs)
}

/// The issue-ahead's counters (`fnahead119 rank<r>:`).
#[derive(Debug, Default, Clone)]
pub struct AheadCounts {
    pub steps: u64,
    pub issued: u64,
    pub rounds: u64,
    pub no_round: u64,
    pub equal: u64,
    pub refused: u64,
    pub redo: u64,
    pub stage_dense: u64,
    pub ineligible: u64,
    /// FN119c: steps of more than one sequence issued ahead (the composed chain inputs), and the
    /// commit's checks of a composition.
    pub multi: u64,
    pub composed_checked: u64,
    /// FN119c: host ms of the commit's composition check, and of the issue-side input builds
    /// (b1 / b > 1).
    pub chk_ms: MsStat,
    pub build_one_ms: MsStat,
    pub build_multi_ms: MsStat,
    /// FN119d (`next`): commits that left the round's ids pending; decode steps built + issued
    /// under the round (`early`); pending rounds resolved before anything was issued, by cause
    /// (another command / a step form the early path does not take / an incomplete tier / a graph
    /// to capture); the host's wait for the ids inside an early step; the device embedding rows
    /// checked against the host gather (the first steps of a boot) and differing.
    pub next_pending: u64,
    pub next_early: u64,
    pub next_res_cmd: u64,
    pub next_res_form: u64,
    pub next_res_tier: u64,
    pub next_res_capture: u64,
    pub next_ids_wait_ms: MsStat,
    pub next_emb_checked: u64,
    /// FN119e: first rounds left pending (`--fn-first-step kvnext`).
    pub first_pending: u64,
    /// FN119f (`--fn-first-step dev`): first tokens walked on the card behind the chunk, equal to
    /// the host's draw, refused (the round re-run on the host's token), first rounds whose END
    /// node rode the device path, deferred prefill tier ends read from the queued log.
    pub dev_first: u64,
    pub dev_equal: u64,
    pub dev_refused: u64,
    pub dev_end: u64,
    pub dev_plog: u64,
}

/// FN119c: a host-time counter (n, mean, p99 by a 0.1 ms histogram to 20 ms, max).
#[derive(Debug, Clone)]
pub struct MsStat {
    pub n: u64,
    pub sum: f64,
    pub max: f64,
    pub hist: Vec<u64>,
}

impl Default for MsStat {
    fn default() -> Self {
        Self {
            n: 0,
            sum: 0.0,
            max: 0.0,
            hist: vec![0; 201],
        }
    }
}

impl MsStat {
    pub fn add(&mut self, ms: f64) {
        self.n += 1;
        self.sum += ms;
        self.max = self.max.max(ms);
        let b = ((ms * 10.0) as usize).min(200);
        self.hist[b] += 1;
    }

    /// The bin's upper edge holding quantile `q`.
    #[must_use]
    pub fn quantile(&self, q: f64) -> f64 {
        let want = (self.n as f64 * q).ceil() as u64;
        let mut c = 0;
        for (b, &k) in self.hist.iter().enumerate() {
            c += k;
            if c >= want.max(1) {
                return (b + 1) as f64 / 10.0;
            }
        }
        self.max
    }
}

impl std::fmt::Display for MsStat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.n == 0 {
            return write!(f, "n 0");
        }
        write!(
            f,
            "n {} mean {:.3} p50 <={:.1} p99 <={:.1} max {:.2} ms",
            self.n,
            self.sum / self.n as f64,
            self.quantile(0.5),
            self.quantile(0.99),
            self.max
        )
    }
}

impl AheadRt {
    pub const RP: usize = 0;
    pub const SQ: usize = AHEAD_ROWS * 32;
    pub const OUT: usize = Self::SQ + 8 * 32;
    pub const ROWOUT: usize = Self::OUT + 8 * 64;
    pub const END: usize = Self::ROWOUT + AHEAD_ROWS * 16;

    /// The result block of sequence `i` read back (flags, acc, ntok, tokens).
    #[must_use]
    pub fn result(&self, i: usize) -> (u32, u32, Vec<u32>) {
        let b = self.stg.as_slice();
        let rd = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().expect("4"));
        let o = Self::OUT + i * 64;
        let n = rd(o + 8) as usize;
        (
            rd(o),
            rd(o + 4),
            (0..n.min(5)).map(|q| rd(o + 12 + 4 * q)).collect(),
        )
    }
}

impl std::fmt::Display for AheadCounts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "decode steps {}, issued ahead {} (round ahead {}, round on the host {}), ineligible {}; \
             walks equal {}, refused {}, re-run rounds {}, dense from the backup {}; b > 1 steps {}, \
             compositions checked {}; commit check {}; issue builds b1 {} / b > 1 {}; FN119d next: \
             commits pending {}, steps issued early {}, resolved first (command {}, form {}, tier \
             {}, capture {}), ids wait {}, device embeddings checked {}; FN119e first rounds \
             pending {}; FN119f first tokens on the card {} (equal {}, refused {}, with an END \
             node {}), prefill tier ends from the queued log {}",
            self.steps,
            self.issued,
            self.rounds,
            self.no_round,
            self.ineligible,
            self.equal,
            self.refused,
            self.redo,
            self.stage_dense,
            self.multi,
            self.composed_checked,
            self.chk_ms,
            self.build_one_ms,
            self.build_multi_ms,
            self.next_pending,
            self.next_early,
            self.next_res_cmd,
            self.next_res_form,
            self.next_res_tier,
            self.next_res_capture,
            self.next_ids_wait_ms,
            self.next_emb_checked,
            self.first_pending,
            self.dev_first,
            self.dev_equal,
            self.dev_refused,
            self.dev_end,
            self.dev_plog
        )
    }
}

/// Whether a slot's sampling lets the device walk decide it without a refusal by construction:
/// no penalties, logprobs inside the seam, a sampled row cut by `top_k` inside the seam with no
/// `min_p` (a live tail or a ceiling survivor is then a tie: rare, handled by the backup).
#[must_use]
pub fn ahead_sampling_ok(p: &RowParams, pen: &[u32]) -> bool {
    pen.is_empty()
        && p.logprobs.is_none_or(|k| k <= SEAM_K)
        && (p.is_greedy() || (p.top_k >= 1 && p.top_k < SEAM_K && p.min_p <= 0.0))
}

// ---------------------------------------------------------------- the oracle (`saber fn119-oracle`)

/// `saber fn119-oracle`: the device twin's host oracle. No GPU.
#[derive(Debug, clap::Args)]
pub struct OracleArgs {
    /// Output directory (cases + expected files, the expf hash table).
    #[arg(long)]
    pub out: PathBuf,
    /// `expf` (exhaustive glibc_expf == libm expf over 2^32 + the probe's hash table), `cases`
    /// (the walk + distribution cases with the real sampler's answers), `all`.
    #[arg(long, default_value = "all")]
    pub mode: String,
    /// Walk sequences per setting.
    #[arg(long, default_value_t = 10_000)]
    pub seqs: usize,
    /// Distribution draws per setting (one seam, this many keys).
    #[arg(long, default_value_t = 100_000)]
    pub draws: usize,
    /// Patterns a probe thread hashes (a group = 256 x this).
    #[arg(long, default_value_t = 64)]
    pub per: u32,
    #[arg(long, default_value_t = 119)]
    pub seed: u64,
}

/// The settings: (name, temperature, top_p, top_k (0 = off), min_p).
pub const SETTINGS: [(&str, f32, f32, i32, f32); 10] = [
    ("greedy", 0.0, 1.0, 0, 0.0),
    ("bb_t07_p95_k20", 0.7, 0.95, 20, 0.0),
    ("gc_t10_p95_k20", 1.0, 0.95, 20, 0.0),
    ("t06_p80_k40", 0.6, 0.8, 40, 0.0),
    ("t10_p100_k20", 1.0, 1.0, 20, 0.0),
    ("t07_p95_k20_m05", 0.7, 0.95, 20, 0.05),
    ("t13_p90_k63", 1.3, 0.9, 63, 0.0),
    ("t10_p95_koff", 1.0, 0.95, 0, 0.0),
    ("t001_p95_k20", 0.01, 0.95, 20, 0.0),
    ("t10_p100_koff_m10", 1.0, 1.0, 0, 0.1),
];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix(self.0)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn bf(v: f32) -> f32 {
    // round to nearest-even bf16
    let b = v.to_bits();
    let r = b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(r)
}

/// One synthetic half seam: 64 (global id, bf16 value), value desc, id asc (the pick's order).
fn half_seam(g: &mut Rng, base: u32, kind: u64, top: f32) -> Vec<(u32, f32)> {
    let half = super::serve::VOCAB as u32 / 2;
    let mut ids = std::collections::BTreeSet::new();
    while ids.len() < SEAM_K {
        ids.insert(base + (g.next() % u64::from(half)) as u32);
    }
    let ids: Vec<u32> = ids.into_iter().collect();
    let mut v: Vec<(u32, f32)> = Vec::with_capacity(SEAM_K);
    let mut x = top;
    for (i, &id) in ids.iter().enumerate() {
        let gap = match kind % 5 {
            0 => (g.unit() * 0.8) as f32,  // ordinary
            1 => (g.unit() * 0.05) as f32, // dense: bf16 ties
            2 => {
                if i == 0 {
                    6.0
                } else {
                    (g.unit() * 0.3) as f32
                }
            } // peaked
            3 => {
                if g.unit() < 0.7 {
                    0.0
                } else {
                    (g.unit() * 0.6) as f32
                }
            } // tie runs
            _ => (g.unit() * 2.0) as f32,  // spread
        };
        x -= gap;
        v.push((id, bf(x)));
    }
    v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    v
}

/// pick2's merge of two halves (total_cmp desc, id asc, top 64).
fn merge(a: &[(u32, f32)], b: &[(u32, f32)]) -> Vec<(u32, f32)> {
    let mut all: Vec<(u32, f32)> = a.iter().chain(b).copied().collect();
    all.sort_by(|x, y| y.1.total_cmp(&x.1).then(x.0.cmp(&y.0)));
    all.truncate(SEAM_K);
    all
}

fn lse_of(m: &[(u32, f32)], t: f32) -> f32 {
    // a plausible whole-row log-sum-exp: the seam's own plus a tail of the same mass
    let mx = m[0].1 / t;
    let s: f64 = m.iter().map(|x| f64::from(x.1 / t - mx).exp()).sum();
    (f64::from(mx) + (2.0 * s).ln()) as f32
}

fn params_of(i: usize) -> saber_serve::sampling::SamplingParams {
    let (_, t, p, k, m) = SETTINGS[i];
    saber_serve::sampling::SamplingParams {
        temperature: t,
        top_p: p,
        top_k: k,
        min_p: m,
        ..Default::default()
    }
}

fn push_u32(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

/// The cases of one setting: `seqs` walk sequences (w 1..5, drafts half the host's own draw) +
/// `draws` one-row sequences on ONE seam with keys 0..draws. Writes `<name>.cases` (header nseq,
/// nrows, nseam; seams a / b; row blocks; sequence blocks) and `<name>.expect` (per sequence: host
/// decided (1) / refused (0), acc, ntok, 5 tokens) + `<name>.dist.json` (p' of the one seam).
fn cases_one(a: &OracleArgs, si: usize) -> anyhow::Result<(usize, usize)> {
    let name = SETTINGS[si].0;
    let sampler = std::sync::Arc::new(saber_serve::sampler::Sampler::new());
    let sp = params_of(si);
    let params = RowParams::resolve(&sp, super::serve::VOCAB);
    let mut g = Rng(a.seed ^ mix(si as u64 + 1));
    let half = super::serve::VOCAB as u32 / 2;
    let (mut sa, mut sb, mut rp, mut sq, mut ex) = (vec![], vec![], vec![], vec![], vec![]);
    let mut nrows = 0u32;
    let mut nseam = 0u32;
    let mk_rowout = |m: &[(u32, f32)]| RowOut {
        top: m.to_vec(),
        lse: lse_of(m, 1.0),
        lse_t: (params.top_p < 1.0 && !params.is_greedy()).then(|| lse_of(m, params.temperature)),
        probe: Vec::new(),
        sha: None,
    };
    let emit_seam = |sa: &mut Vec<u8>, sb: &mut Vec<u8>, ha: &[(u32, f32)], hb: &[(u32, f32)]| {
        for &(id, v) in ha {
            push_u32(sa, id);
            push_u32(sa, v.to_bits());
        }
        for &(id, v) in hb {
            push_u32(sb, id);
            push_u32(sb, v.to_bits());
        }
    };
    let mut nseq = 0usize;
    // walk sequences
    for _ in 0..a.seqs {
        let w = 1 + (g.next() % 5) as usize;
        let kind = g.next();
        let mut rows = Vec::with_capacity(w);
        let row0 = nrows;
        for _ in 0..w {
            let top = (5.0 + g.unit() * 25.0) as f32 * if kind % 7 == 6 { -1.0 } else { 1.0 };
            let ha = half_seam(&mut g, 0, kind, top);
            let tb = top - (g.unit() * 3.0) as f32;
            let hb = half_seam(&mut g, half, kind >> 3, tb);
            emit_seam(&mut sa, &mut sb, &ha, &hb);
            rows.push(mk_rowout(&merge(&ha, &hb)));
            nrows += 1;
            nseam += 1;
        }
        let seed = g.next() >> 1;
        let pos = g.next() % 4000;
        let mut ss = saber_sched::SlotSampling::greedy();
        ss.params = saber_serve::sampling::SamplingParams {
            seed: Some(seed),
            ..sp.clone()
        };
        ss.position = pos;
        // drafts: half the host's own row draw (the walk's target), else a near candidate
        let mut draft = Vec::with_capacity(w.saturating_sub(1));
        for j in 0..w.saturating_sub(1) {
            let one = super::worker::verify::walk_slot(
                &sampler,
                &rows[j..=j],
                &[],
                &super::worker::verify::sampling_at(&ss, j, &draft),
            );
            let t = if g.unit() < 0.6 {
                one.tokens[0]
            } else {
                rows[j].top[(g.next() % 4) as usize].0
            };
            draft.push(t);
        }
        let wk = super::worker::verify::walk_slot(&sampler, &rows, &draft, &ss);
        for j in 0..w {
            let key = noise_key(seed, pos + j as u64, 0);
            for x in row_block(&params, false, row0 + j as u32, key) {
                push_u32(&mut rp, x);
            }
        }
        push_u32(&mut sq, row0);
        push_u32(&mut sq, w as u32);
        push_u32(&mut sq, draft.len() as u32);
        for j in 0..5 {
            push_u32(&mut sq, draft.get(j).copied().unwrap_or(0));
        }
        push_u32(&mut ex, u32::from(wk.dense.is_none()));
        push_u32(&mut ex, wk.accepted());
        push_u32(&mut ex, wk.tokens.len() as u32);
        for j in 0..5 {
            push_u32(&mut ex, wk.tokens.get(j).copied().unwrap_or(0));
        }
        nseq += 1;
    }
    // the distribution sequences: one seam, `draws` keys
    let kind = 0u64;
    let ha = half_seam(&mut g, 0, kind, 18.0);
    let hb = half_seam(&mut g, half, kind, 17.0);
    emit_seam(&mut sa, &mut sb, &ha, &hb);
    let m = merge(&ha, &hb);
    let ro = mk_rowout(&m);
    let dseam = nseam;
    nseam += 1;
    let dist = sampler
        .distribution(
            &super::serve::device_row(&ro, &[]),
            &params,
            &Default::default(),
        )
        .ok()
        .flatten();
    for d in 0..a.draws {
        let seed = d as u64;
        let mut ss = saber_sched::SlotSampling::greedy();
        ss.params = saber_serve::sampling::SamplingParams {
            seed: Some(seed),
            ..sp.clone()
        };
        let wk = super::worker::verify::walk_slot(&sampler, std::slice::from_ref(&ro), &[], &ss);
        for x in row_block(&params, false, dseam, noise_key(seed, 0, 0)) {
            push_u32(&mut rp, x);
        }
        push_u32(&mut sq, nrows);
        push_u32(&mut sq, 1);
        push_u32(&mut sq, 0);
        for _ in 0..5 {
            push_u32(&mut sq, 0);
        }
        nrows += 1;
        push_u32(&mut ex, u32::from(wk.dense.is_none()));
        push_u32(&mut ex, 0);
        push_u32(&mut ex, 1);
        push_u32(&mut ex, wk.tokens[0]);
        for _ in 0..4 {
            push_u32(&mut ex, 0);
        }
        nseq += 1;
    }
    let mut f = Vec::new();
    for x in [0x0119_0119u32, nseq as u32, nrows, nseam] {
        push_u32(&mut f, x);
    }
    f.extend_from_slice(&sa);
    f.extend_from_slice(&sb);
    f.extend_from_slice(&rp);
    f.extend_from_slice(&sq);
    std::fs::write(a.out.join(format!("{name}.cases")), &f)?;
    std::fs::write(a.out.join(format!("{name}.expect")), &ex)?;
    let dj = serde_json::json!({
        "setting": name,
        "walk_seqs": a.seqs,
        "draws": a.draws,
        "dist_first_seq": a.seqs,
        "p": dist,
    });
    std::fs::write(
        a.out.join(format!("{name}.dist.json")),
        serde_json::to_vec(&dj)?,
    )?;
    Ok((nseq, nrows as usize))
}

/// `saber fn119-oracle`.
///
/// # Errors
/// I/O, or (mode `expf`) a pattern where the port differs from the linked libm.
pub fn oracle(a: &OracleArgs) -> anyhow::Result<()> {
    std::fs::create_dir_all(&a.out)?;
    if a.mode == "expf" || a.mode == "all" {
        let t0 = std::time::Instant::now();
        let threads = std::thread::available_parallelism()
            .map_or(8, |n| n.get())
            .min(16) as u64;
        let group = 256 * u64::from(a.per);
        let groups = (1u64 << 32) / group;
        let per_t = groups.div_ceil(threads);
        let parts: Vec<(u64, Option<u32>, Vec<u64>)> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..threads)
                .map(|t| {
                    sc.spawn(move || {
                        let mut bad = 0u64;
                        let mut first = None;
                        let mut hs = Vec::new();
                        for gi in t * per_t..((t + 1) * per_t).min(groups) {
                            let mut h = 0u64;
                            for u in gi * group..(gi + 1) * group {
                                let u = u as u32;
                                let x = f32::from_bits(u);
                                let p = glibc_expf(x);
                                let l = x.exp();
                                if p.to_bits() != l.to_bits() && !(p.is_nan() && l.is_nan()) {
                                    bad += 1;
                                    first.get_or_insert(u);
                                    println!(
                                        "FN119 EXPF differ: x {x:e} ({u:#010x}): port {:#010x} libm {:#010x}",
                                        p.to_bits(),
                                        l.to_bits()
                                    );
                                }
                                h ^= expf_term(u, p.to_bits());
                            }
                            hs.push(h);
                        }
                        (bad, first, hs)
                    })
                })
                .collect();
            hs.into_iter()
                .map(|h| h.join().expect("expf thread"))
                .collect()
        });
        let mut bad = 0;
        let mut first = None;
        let mut table = Vec::with_capacity(groups as usize * 8);
        for (b, f, hs) in parts {
            bad += b;
            if first.is_none() {
                first = f;
            }
            for h in hs {
                table.extend_from_slice(&h.to_le_bytes());
            }
        }
        std::fs::write(a.out.join("expf_hash.bin"), &table)?;
        println!(
            "FN119 EXPF: glibc_expf vs the linked libm expf over all 2^32 patterns: {bad} differ \
             (first {first:?}); probe table {groups} groups x {group} patterns; {:.1} s",
            t0.elapsed().as_secs_f64()
        );
        ensure!(
            bad == 0,
            "FN119: the expf port differs from libm on {bad} patterns"
        );
    }
    if a.mode == "cases" || a.mode == "all" {
        let t0 = std::time::Instant::now();
        let r: Vec<anyhow::Result<(usize, usize)>> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..SETTINGS.len())
                .map(|i| sc.spawn(move || cases_one(a, i)))
                .collect();
            hs.into_iter()
                .map(|h| h.join().expect("case thread"))
                .collect()
        });
        for (i, x) in r.into_iter().enumerate() {
            let (s, rows) = x.with_context(|| SETTINGS[i].0)?;
            println!("FN119 CASES {}: {s} sequences, {rows} rows", SETTINGS[i].0);
        }
        println!("FN119 CASES written in {:.1} s", t0.elapsed().as_secs_f64());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ro(vals: &[f32]) -> RowOut {
        RowOut {
            top: vals
                .iter()
                .enumerate()
                .map(|(i, &v)| (i as u32 + 7, v))
                .collect(),
            lse: f32::NAN,
            lse_t: None,
            probe: Vec::new(),
            sha: None,
        }
    }

    #[test]
    fn the_next_form_is_the_default_and_ahead_is_the_control() {
        assert_eq!(DEFAULT_FN_DEVICE_STEP, DeviceStep::Next);
        assert_eq!(DeviceStep::default(), DeviceStep::Next);
        assert!(DeviceStep::Next.ahead() && DeviceStep::Next.next() && DeviceStep::Next.lazy());
        assert!(DeviceStep::Ahead.ahead() && !DeviceStep::Ahead.next());
        assert!(!DeviceStep::Lazy.ahead());
        for f in ["off", "lazy", "lazy-late", "check", "ahead", "next"] {
            assert_eq!(f.parse::<DeviceStep>().unwrap().to_string(), f);
        }
        assert!("on".parse::<DeviceStep>().is_err());
        assert!(!DeviceStep::Off.lazy() && DeviceStep::Check.lazy());
    }

    #[test]
    fn a_row_at_the_served_knobs_defers_and_today_is_unchanged() {
        let sp = saber_serve::sampling::SamplingParams {
            temperature: 0.7,
            top_p: 0.95,
            top_k: 20,
            ..Default::default()
        };
        let mut s = saber_sched::SlotSampling::greedy();
        s.params = sp;
        let (p, pen, rq0) = super::super::serve::row_req(&s);
        // today: both log-sum-exps
        assert_eq!(wants(&rq0), (true, true));
        let mut rq = rq0.clone();
        shape(&mut rq, &p, &pen);
        assert_eq!(rq.tk, 20);
        assert!(!rq.lse);
        assert_eq!(wants(&rq), (false, false));
        // logprobs keep the T = 1 one
        s.params.logprobs = Some(2);
        let (p, pen, mut rq) = super::super::serve::row_req(&s);
        shape(&mut rq, &p, &pen);
        assert_eq!(wants(&rq), (true, false));
        // top_k off / min_p: nothing deferred
        s.params.top_k = 0;
        let (p, pen, mut rq) = super::super::serve::row_req(&s);
        shape(&mut rq, &p, &pen);
        assert_eq!((rq.tk, wants(&rq).1), (0, true));
        s.params.top_k = 20;
        s.params.min_p = 0.05;
        let (p, pen, mut rq) = super::super::serve::row_req(&s);
        shape(&mut rq, &p, &pen);
        assert_eq!(rq.tk, 0);
    }

    #[test]
    fn a_tie_from_the_cut_to_the_floor_is_late_and_the_control_forces_every_row() {
        let rq = RowReq {
            temp: 0.7,
            tk: 20,
            ..Default::default()
        };
        let mut v: Vec<f32> = (0..64).map(|i| 20.0 - i as f32 * 0.25).collect();
        let outs = vec![ro(&v)];
        assert!(late_rows(&outs, std::slice::from_ref(&rq), false).is_empty());
        assert_eq!(late_rows(&outs, std::slice::from_ref(&rq), true), vec![0]);
        for x in v.iter_mut().skip(19) {
            *x = 3.0;
        }
        assert_eq!(
            late_rows(&[ro(&v)], std::slice::from_ref(&rq), false),
            vec![0]
        );
        v[5] = f32::NAN;
        assert_eq!(late_rows(&[ro(&v)], &[rq], false), vec![0]);
    }

    #[test]
    fn the_late_fill_is_the_picks_own_lse_t() {
        // the same row through the frame's function (each half about its own max, merged)
        let half = 1000;
        let mut g = Rng(5);
        let dense: Vec<f32> = (0..2 * half)
            .map(|_| bf((g.unit() * 20.0 - 10.0) as f32))
            .collect();
        let t = 0.7;
        let hb = |h: &[f32]| -> Vec<u8> {
            h.iter()
                .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
                .collect()
        };
        let (a, b) = (hb(&dense[..half]), hb(&dense[half..]));
        let la =
            super::super::worker::row_lse(&a, super::super::worker::seam_top_k(&a, SEAM_K)[0].1, t);
        let lb =
            super::super::worker::row_lse(&b, super::super::worker::seam_top_k(&b, SEAM_K)[0].1, t);
        let want = saber_serve::sampler::log_add_exp(la, lb);
        assert_eq!(fill_lse_t(&dense, t).to_bits(), want.to_bits());
    }

    #[test]
    fn the_push_covers_exactly_the_rows_the_walk_consumes() {
        // the walk's blocks: sequence i reads rows sq[0] = i * w .. + w (worker::ahead_finish)
        for (n, bucket, w) in [(1, 1, 4), (3, 4, 3), (7, 8, 3), (8, 8, 3), (5, 8, 2)] {
            let read: usize = (0..n).map(|i| (i * w..i * w + w).len()).sum();
            assert_eq!(push_rows(n, w), read);
            assert!(push_rows(n, w) <= bucket * w && push_rows(n, w) <= AHEAD_ROWS);
            if n < bucket {
                // the control: the pre-fix push (bucket x w) leaves pad rows unconsumed
                assert!(bucket * w > read);
            }
        }
    }

    #[test]
    fn owner_runs_compose_and_a_shared_word_refuses() {
        let v0 = vec![0u8; 16];
        let mut s0a1 = v0.clone();
        s0a1[0] = 1;
        let mut s1a1 = v0.clone();
        s1a1[8] = 2;
        s1a1[12] = 3;
        let runs = owner_runs(&v0, &[vec![s0a1.clone()], vec![s1a1.clone()]]).unwrap();
        assert_eq!(runs, vec![[0, 0, 1], [1, 2, 2]]);
        let mut v1 = v0.clone();
        v1[0] = 1;
        v1[8] = 2;
        v1[12] = 3;
        let c = ComposeChk {
            pos0: vec![0, 0],
            vars: vec![v0.clone(), v1.clone()],
            runs,
        };
        assert_eq!(c.compose(&[1, 1]), v1);
        assert_eq!(c.compose(&[0, 1])[0], 0);
        assert_eq!(c.compose(&[1, 0])[0], 1);
        let mut shared = v0.clone();
        shared[0] = 9;
        assert!(owner_runs(&v0, &[vec![s0a1], vec![shared]]).is_none());
    }

    #[test]
    fn the_noise_key_is_the_samplers() {
        use saber_serve::sampler::NoiseSource;
        let k = noise_key(42, 17, 0);
        let mut n = saber_serve::sampler::SeededNoise::new(42, 17, 0);
        for id in [0usize, 5, 248_319] {
            let bits = mix(k ^ mix(id as u64));
            let u = (((bits >> 11) + 1) as f64) * (1.0 / 9_007_199_254_740_993.0);
            assert_eq!((-(u.ln()) as f32).to_bits(), n.exponential(id).to_bits());
        }
    }

    #[test]
    fn the_expf_port_matches_libm_on_a_sweep() {
        // the exhaustive gate is `saber fn119-oracle --mode expf` (in the dev image's libm); a
        // sweep here keeps a host change of the port honest
        let mut bad = 0;
        let mut u = 0u32;
        loop {
            let x = f32::from_bits(u);
            let (p, l) = (glibc_expf(x), x.exp());
            if p.to_bits() != l.to_bits() && !(p.is_nan() && l.is_nan()) {
                bad += 1;
            }
            match u.checked_add(9_973) {
                Some(n) => u = n,
                None => break,
            }
        }
        // the host's libm may be another glibc (CORE-MATH expf since 2.41): count, never fail
        // here; the dev image's exhaustive gate is the law
        println!("glibc_expf vs host libm on a 1-in-9973 sweep: {bad} differ");
    }
}
