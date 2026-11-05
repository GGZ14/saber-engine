//! `saber ragged-attn-gate` — the RAGGED decode attention's card gate (workstream C2).
//!
//! `saber_attn_ragged.hsaco` is rx6's decode kernel and split-KV combine DERIVED by anchored
//! substitutions (the retired source generator RAGGED_SUBS): each sequence's query length is read
//! from `cu_q` instead of the one `q_len` kernarg. Three arms, one card, synthetic inputs (the
//! kernels read nothing a capture has to supply: a paged K/V cache, a block table, `seqused_k`,
//! `cu_q` and bf16 queries):
//!
//! * **(a) UNIFORM WIDTHS, BIT-EXACT against the uniform kernel.** At `cu_q[n] = n * q_len` every
//!   expression the derivation touched is the uniform kernel's own, so the outputs must be
//!   byte-identical — widths 1, 5, 8 over batches 1, 4, 8, fp8 and bf16 K/V.
//! * **(b) RAGGED WIDTHS against a HOST f64 reference** over the same (dequantised) K/V: every
//!   live row within tolerance, every band padding row exactly zero, a zero-row padding sequence
//!   contributing nothing. NEGATIVE CONTROLS, both required to FAIL the same tolerance: the
//!   UNIFORM kernel fed the ragged batch (every sequence right-aligned on the max width, its rows
//!   read at `seq * q_len`), and the ragged kernel with an off-by-one `cu_q`.
//! * **(c) PER-ROW IDENTITY**: every sequence of the ragged batch run ALONE through the ragged
//!   kernel at its own width is byte-identical to its rows in the batch — the kernel half of the
//!   engine-identity law. The split count is pinned (the engine's `batch_invariant_attn`), so
//!   the batch cannot move a row's partition.
//!
//! GPU rules (CLAUDE.md): one HIP device, run under that card's lock by
//! `tools/gates/c2_ragged_attn.sh`, every allocation inside `--budget-mib`, the power cap
//! untouched.

use std::path::PathBuf;

use saber_exec::{ArgBindings, LaunchCounters, LaunchCtx, Launcher, LoadCounters, PluginSet};
use saber_hip::{DevicePtr, HipThread, Stream, StreamFlags};
use saber_kernels::rules::Shape;
use saber_kernels::saber::{self, attn_decode_combine, AttnShape, KvDtype, SaberRules};

/// Arguments, as the binary declares them.
#[derive(Debug, Clone)]
pub struct RaggedAttnGateArgs {
    /// Directory of committed `.hsaco` + `*.manifest.toml`.
    pub prebuilt: PathBuf,
    /// HIP device ordinal within the visible set.
    pub device: i32,
    /// Refuse to run if one case's device buffers exceed this.
    pub budget_mib: u64,
    /// Where the verdict JSON goes (the gate script points it at the shared cache).
    pub out: Option<PathBuf>,
    /// RNG seed for the synthetic inputs.
    pub seed: u64,
}

// ---------------------------------------------------------------------------- the geometry

/// The model at TP=2: 12 q heads / 2 kv heads per rank, head_dim 256, paged block 16.
pub(crate) const Q_HEADS: u32 = 12;
pub(crate) const KV_HEADS: u32 = 2;
pub(crate) const HEAD_DIM: u32 = 256;
pub(crate) const BLOCK: u32 = 16;
/// The context band the plans are built for; the split law and the block-table width read it.
pub(crate) const MAX_CTX: u32 = 4096;

/// The tolerance arm (b) reads: `|got - ref| <= ATOL + RTOL * |ref|` per element. The kernel runs
/// its QK and PV products in f16 WMMA with f16 partials and a bf16 output, so this is a class
/// bound, not a rounding bound; the negative controls must break it by an order of magnitude.
pub(crate) const ATOL: f64 = 2.0e-2;
pub(crate) const RTOL: f64 = 2.0e-2;

/// One decode attention call's inputs, on the host.
#[derive(Clone, Debug)]
pub(crate) struct Case {
    pub kv: KvDtype,
    /// Slots in the call (padding included) — the kernel's `num_seqs`.
    pub batch: u32,
    /// The call's max width — the kernel's `q_len`.
    pub max_w: u32,
    /// Each slot's rows (0 for a padding slot; live slots first).
    pub widths: Vec<u32>,
    /// Rows the call's output covers: `batch * max_w` for a uniform call, the band's for a
    /// ragged one (`>= Σ widths`).
    pub rows: u32,
    /// Each slot's resident context before this step.
    pub done: Vec<u32>,
    /// bf16 queries, `[rows, q_heads, head_dim]`, laid out by `cu` (ragged) or `n * max_w`.
    pub q: Vec<u16>,
    /// The paged cache, `[blocks, kv_heads, block, 2 * head_dim]`, one byte (fp8) or two (bf16)
    /// per element.
    pub kv_bytes: Vec<u8>,
    /// Pages per slot, `max_blocks` wide.
    pub block_table: Vec<i32>,
}

pub(crate) fn max_blocks() -> u32 {
    MAX_CTX / BLOCK
}

impl Case {
    /// `cu_q`: the cumulative live-row prefix, `batch + 1` entries.
    pub fn cu(&self) -> Vec<i32> {
        let mut cu = vec![0i32];
        for w in &self.widths {
            cu.push(cu.last().copied().unwrap_or(0) + *w as i32);
        }
        cu
    }

    /// `seqused_k`: `done + w` for a live slot (the ragged mask's right edge), 0 for padding.
    pub fn seqused(&self) -> Vec<i32> {
        self.widths
            .iter()
            .zip(&self.done)
            .map(|(w, d)| if *w == 0 { 0 } else { (*d + *w) as i32 })
            .collect()
    }

    fn elem_bytes(&self) -> usize {
        match self.kv {
            KvDtype::Fp8E4m3 => 1,
            KvDtype::Bf16 => 2,
        }
    }

    /// Element `e` of the (key or value) row at `pos` of slot `n`, kv head `kvh`, dequantised.
    fn kv_at(&self, n: usize, kvh: u32, pos: u32, value: bool, e: u32) -> f64 {
        let mb = max_blocks() as usize;
        let page = self.block_table[n * mb + (pos / BLOCK) as usize] as u64;
        let at = ((page * u64::from(KV_HEADS) + u64::from(kvh)) * u64::from(BLOCK)
            + u64::from(pos % BLOCK))
            * 2
            * u64::from(HEAD_DIM)
            + u64::from(if value { HEAD_DIM } else { 0 })
            + u64::from(e);
        let at = at as usize * self.elem_bytes();
        match self.kv {
            KvDtype::Fp8E4m3 => fp8_e4m3(self.kv_bytes[at]),
            KvDtype::Bf16 => bf16(u16::from_le_bytes([
                self.kv_bytes[at],
                self.kv_bytes[at + 1],
            ])),
        }
    }

    /// The query row the kernel reads for slot `n`'s row `t`, under a layout (`cu` or strided).
    fn q_row(&self, row: usize, h: u32) -> Vec<f64> {
        let base = (row * Q_HEADS as usize + h as usize) * HEAD_DIM as usize;
        self.q[base..base + HEAD_DIM as usize]
            .iter()
            .map(|b| bf16(*b))
            .collect()
    }
}

/// e4m3fn, as the kernel's `fp8x8_to_16x4w` decodes it (no infinities; 0x7f / 0xff are NaN and
/// this gate never generates them).
pub(crate) fn fp8_e4m3(b: u8) -> f64 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((b >> 3) & 0xf);
    let m = f64::from(b & 7);
    if e == 0 {
        sign * m / 8.0 * 2f64.powi(-6)
    } else {
        sign * (1.0 + m / 8.0) * 2f64.powi(e - 7)
    }
}

pub(crate) fn bf16(b: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(b) << 16))
}

fn to_bf16(x: f32) -> u16 {
    // Round to nearest even, as torch does.
    let u = x.to_bits();
    let r = 0x7fff + ((u >> 16) & 1);
    ((u + r) >> 16) as u16
}

/// A small counter-based generator, so a case is a function of its seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// A case: `widths` over `batch` slots at max width `max_w`, `rows` output rows, each live slot
/// holding between 40 and `MAX_CTX - 64` tokens of context in its own pages.
pub(crate) fn make_case(
    kv: KvDtype,
    batch: u32,
    max_w: u32,
    widths: &[u32],
    rows: u32,
    seed: u64,
) -> Case {
    let mut rng = Rng(seed);
    let mb = max_blocks();
    let done: Vec<u32> = (0..batch)
        .map(|_| 40 + (rng.next() % u64::from(MAX_CTX - 128)) as u32)
        .collect();
    // Every slot its own pages, in a pool of `batch * max_blocks` (page ids shuffled per slot so
    // a layout bug reads another slot's K/V rather than a zero).
    let mut block_table = Vec::with_capacity((batch * mb) as usize);
    for n in 0..batch {
        for b in 0..mb {
            block_table.push((n * mb + ((b * 7 + n) % mb)) as i32);
        }
    }
    let blocks = (batch * mb) as usize;
    let elems = blocks * (KV_HEADS * BLOCK * 2 * HEAD_DIM) as usize;
    let kv_bytes: Vec<u8> = match kv {
        KvDtype::Fp8E4m3 => (0..elems)
            .map(|_| {
                let r = rng.next();
                // Exponents 3..=9 (|x| in [2^-4, 2^2]), any mantissa, either sign: never NaN.
                let e = 3 + (r % 7) as u8;
                let m = ((r >> 8) & 7) as u8;
                let s = ((r >> 16) & 1) as u8;
                (s << 7) | (e << 3) | m
            })
            .collect(),
        KvDtype::Bf16 => (0..elems)
            .flat_map(|_| to_bf16(rng.unit()).to_le_bytes())
            .collect(),
    };
    let q: Vec<u16> = (0..(rows * Q_HEADS * HEAD_DIM) as usize)
        .map(|_| to_bf16(rng.unit() * 2.0))
        .collect();
    Case {
        kv,
        batch,
        max_w,
        widths: widths.to_vec(),
        rows,
        done,
        q,
        kv_bytes,
        block_table,
    }
}

/// The HOST reference: f64 softmax attention for every live row, laid out by `cu`; the rows past
/// `cu[batch]` are zero (the combine's own answer for a band's padding).
#[allow(clippy::needless_range_loop)]
pub(crate) fn reference(c: &Case) -> Vec<f64> {
    let cu = c.cu();
    let scale = 1.0 / f64::from(HEAD_DIM).sqrt();
    let gqa = Q_HEADS / KV_HEADS;
    let mut out = vec![0.0; (c.rows * Q_HEADS * HEAD_DIM) as usize];
    for n in 0..c.batch as usize {
        for t in 0..c.widths[n] {
            let row = cu[n] as usize + t as usize;
            let klimit = c.done[n] + t; // the key the row's own token holds
            for h in 0..Q_HEADS {
                let kvh = h / gqa;
                let q = c.q_row(row, h);
                let scores: Vec<f64> = (0..=klimit)
                    .map(|j| {
                        (0..HEAD_DIM)
                            .map(|e| q[e as usize] * c.kv_at(n, kvh, j, false, e))
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
                let z: f64 = w.iter().sum();
                let base = (row * Q_HEADS as usize + h as usize) * HEAD_DIM as usize;
                for e in 0..HEAD_DIM {
                    let v: f64 = (0..=klimit)
                        .map(|j| w[j as usize] * c.kv_at(n, kvh, j, true, e))
                        .sum();
                    out[base + e as usize] = v / z;
                }
            }
        }
    }
    out
}

/// The same reference as the UNIFORM kernel would compute it on a ragged batch — the host model
/// of arm (b)'s first negative control, so the pre-flight can say the control is a control
/// before any card time: every slot right-aligned on `max_w` (`klimit = ctx - max_w + qrow`), its
/// query rows read at `n * max_w`, its outputs written there.
#[cfg(test)]
#[allow(clippy::needless_range_loop)]
pub(crate) fn uniform_reading(c: &Case) -> Vec<f64> {
    let scale = 1.0 / f64::from(HEAD_DIM).sqrt();
    let gqa = Q_HEADS / KV_HEADS;
    let used = c.seqused();
    let mut out = vec![0.0; (c.rows * Q_HEADS * HEAD_DIM) as usize];
    for n in 0..c.batch as usize {
        if used[n] <= 0 {
            continue;
        }
        for t in 0..c.max_w {
            let row = n * c.max_w as usize + t as usize;
            if row >= c.rows as usize {
                continue;
            }
            let klimit = used[n] - c.max_w as i32 + t as i32;
            if klimit < 0 {
                continue;
            }
            for h in 0..Q_HEADS {
                let kvh = h / gqa;
                let q = c.q_row(row, h);
                let scores: Vec<f64> = (0..=klimit as u32)
                    .map(|j| {
                        (0..HEAD_DIM)
                            .map(|e| q[e as usize] * c.kv_at(n, kvh, j, false, e))
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
                let z: f64 = w.iter().sum();
                let base = (row * Q_HEADS as usize + h as usize) * HEAD_DIM as usize;
                for e in 0..HEAD_DIM {
                    let v: f64 = (0..=klimit as u32)
                        .map(|j| w[j as usize] * c.kv_at(n, kvh, j, true, e))
                        .sum();
                    out[base + e as usize] = v / z;
                }
            }
        }
    }
    out
}

/// `(elements outside the tolerance, the largest |got - ref|, band padding elements not zero)`.
pub(crate) fn compare(c: &Case, got: &[f64], want: &[f64]) -> (usize, f64, usize) {
    let live = *c.cu().last().unwrap_or(&0) as usize * (Q_HEADS * HEAD_DIM) as usize;
    let mut bad = 0usize;
    let mut worst = 0.0f64;
    let mut pad_nonzero = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if i >= live {
            if *g != 0.0 {
                pad_nonzero += 1;
            }
            continue;
        }
        let d = (g - w).abs();
        worst = worst.max(d);
        if d > ATOL + RTOL * w.abs() || !g.is_finite() {
            bad += 1;
        }
    }
    (bad, worst, pad_nonzero)
}

/// The pinned split count: the one-sequence answer of the split law, as the engine pins it.
fn splits(rules: &SaberRules) -> u32 {
    rules.decode_splits(MAX_CTX, KV_HEADS, 1, 0)
}

fn shape(c: &Case, ragged: bool, rules: &SaberRules) -> AttnShape {
    AttnShape {
        num_seqs: c.batch,
        q_len: c.max_w,
        live_rows: c.max_w,
        q_heads: Q_HEADS,
        kv_heads: KV_HEADS,
        head_dim: HEAD_DIM,
        block_size: BLOCK,
        max_blocks: max_blocks(),
        max_ctx: MAX_CTX,
        splits: splits(rules),
        kv_dtype: c.kv,
        kv_block_stride: i64::from(KV_HEADS * BLOCK * 2 * HEAD_DIM),
        kv_head_stride: i64::from(BLOCK * 2 * HEAD_DIM),
        scale_bits: (1.0f32 / 16.0).to_bits(),
        has_descales: false,
        ragged_rows: if ragged { c.rows } else { 0 },
    }
}

/// The device half of one call: upload, the decode kernel and its combine, read the output back.
struct Device<'h> {
    hip: &'h HipThread,
    ctx: LaunchCtx<'h>,
    stream: Stream,
    launcher: Launcher,
    rules: SaberRules,
    budget: u64,
}

impl Device<'_> {
    fn upload(&self, bytes: &[u8]) -> anyhow::Result<DevicePtr> {
        let p = self.hip.malloc(bytes.len().max(4) as u64)?;
        self.hip.memcpy_h2d_blocking(&p, bytes, &self.stream)?;
        Ok(p)
    }

    /// Run `c` through the uniform (`ragged` false) or ragged pair, with `cu` as given (the
    /// off-by-one control passes a shifted one). Returns the output as f64.
    fn run(&mut self, c: &Case, ragged: bool, cu: &[i32]) -> anyhow::Result<Vec<f64>> {
        let s = shape(c, ragged, &self.rules);
        let head = saber::resolve(saber::Op::AttnDecode, saber::Shape::Attn(s), &self.rules)?;
        let combine = attn_decode_combine(s, &self.rules)?;
        let out_bytes = (c.rows * Q_HEADS * HEAD_DIM * 2) as usize;
        let scratch = self.rules.decode_scratch_bytes(&s, splits(&self.rules));
        let total = (c.q.len() * 2 + c.kv_bytes.len() + out_bytes) as u64 + scratch;
        anyhow::ensure!(
            total <= self.budget,
            "a case of {} MiB exceeds the {} MiB budget",
            total >> 20,
            self.budget >> 20
        );
        let q: Vec<u8> = c.q.iter().flat_map(|x| x.to_le_bytes()).collect();
        let d_q = self.upload(&q)?;
        let d_kv = self.upload(&c.kv_bytes)?;
        let bt: Vec<u8> = c.block_table.iter().flat_map(|x| x.to_le_bytes()).collect();
        let d_bt = self.upload(&bt)?;
        let used: Vec<u8> = c.seqused().iter().flat_map(|x| x.to_le_bytes()).collect();
        let d_used = self.upload(&used)?;
        let cu_b: Vec<u8> = cu.iter().flat_map(|x| x.to_le_bytes()).collect();
        let d_cu = self.upload(&cu_b)?;
        // The output starts POISONED (0xff bytes = bf16 NaN), so a row the combine does not write
        // reads as a failure, never as a zero it did not produce.
        let d_out = self.hip.malloc(out_bytes as u64)?;
        self.hip.memset(&d_out, 0xff, &self.stream)?;
        let d_scratch = self.hip.malloc(scratch.max(4))?;
        self.hip.memset(&d_scratch, 0, &self.stream)?;
        let mut b = ArgBindings::new(Shape::new(0, 0, 0));
        b.bind("q", d_q.addr());
        b.bind("kv", d_kv.addr());
        b.bind("block_table", d_bt.addr());
        b.bind("seqused_k", d_used.addr());
        b.bind("out", d_out.addr());
        b.bind("attn_decode_scratch", d_scratch.addr());
        b.bind("cu_q", d_cu.addr());
        self.launcher
            .dispatch_one(&self.ctx, &head, &b, &self.stream)?;
        self.launcher
            .dispatch_one(&self.ctx, &combine, &b, &self.stream)?;
        self.hip.sync_stream(&self.stream)?;
        let mut got = vec![0u8; out_bytes];
        self.hip
            .memcpy_d2h_blocking(&mut got, &d_out, &self.stream)?;
        drop((d_q, d_kv, d_bt, d_used, d_cu, d_out, d_scratch));
        Ok(got
            .chunks_exact(2)
            .map(|x| bf16(u16::from_le_bytes([x[0], x[1]])))
            .collect())
    }
}

fn kv_name(kv: KvDtype) -> &'static str {
    match kv {
        KvDtype::Fp8E4m3 => "fp8",
        KvDtype::Bf16 => "bf16",
    }
}

/// Run the gate. Returns the number of failing checks.
#[allow(clippy::too_many_lines)]
pub fn run(hip: &HipThread, args: &RaggedAttnGateArgs) -> anyhow::Result<usize> {
    hip.set_device(args.device)?;
    let rules = SaberRules::prod_saber()?;
    let load = LoadCounters::new();
    let plugins = PluginSet::load_dir(hip, &args.prebuilt, &load)?;
    let functions = plugins.functions(hip)?;
    let counters = LaunchCounters::new();
    let mut dev = Device {
        hip,
        ctx: LaunchCtx::new(hip, &functions),
        stream: hip.create_stream(StreamFlags::NonBlocking)?,
        launcher: Launcher::new("ragged-attn-gate", &counters),
        rules,
        budget: args.budget_mib << 20,
    };
    let mut failed = 0usize;
    let mut verdicts: Vec<String> = Vec::new();
    let mut note = |line: String, ok: bool, failed: &mut usize| {
        println!("{line}  {}", if ok { "OK" } else { "FAIL" });
        verdicts.push(format!(
            "{{\"check\":{:?},\"ok\":{ok}}}",
            line.split_whitespace().collect::<Vec<_>>().join(" ")
        ));
        if !ok {
            *failed += 1;
        }
    };
    println!(
        "ragged-attn gate: TP=2 geometry (q heads {Q_HEADS}, kv heads {KV_HEADS}, head_dim \
         {HEAD_DIM}), ctx band {MAX_CTX}, splits pinned at {}",
        splits(&dev.rules)
    );

    // ---- (a) uniform widths: the ragged pair IS the uniform pair, byte for byte --------------
    println!("== (a) uniform widths, ragged vs uniform kernel, byte-compare");
    for kv in [KvDtype::Fp8E4m3, KvDtype::Bf16] {
        for w in [1u32, 5, 8] {
            for batch in [1u32, 4, 8] {
                let c = make_case(kv, batch, w, &vec![w; batch as usize], batch * w, args.seed);
                let u = dev.run(&c, false, &c.cu())?;
                let r = dev.run(&c, true, &c.cu())?;
                let differ = u
                    .iter()
                    .zip(&r)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                note(
                    format!(
                        "(a) {} b{batch}s{w}: {differ} of {} output element(s) differ",
                        kv_name(kv),
                        u.len()
                    ),
                    differ == 0,
                    &mut failed,
                );
            }
        }
    }

    // ---- (b) ragged widths against the host reference, with two controls ---------------------
    println!(
        "== (b) ragged widths vs the host f64 reference (|d| <= {ATOL} + {RTOL}|ref|; band \
         padding exactly 0)"
    );
    let ragged_cases: [(&str, Vec<u32>, u32); 3] = [
        ("conc-8 multiset", vec![5, 3, 8, 2, 5, 1, 4, 5], 40),
        (
            "zero-row padding slots + band padding",
            vec![6, 2, 7, 1, 0, 0, 0, 0],
            24,
        ),
        ("one slot per width", vec![1, 2, 3, 4, 5, 6, 7, 8], 40),
    ];
    for kv in [KvDtype::Fp8E4m3, KvDtype::Bf16] {
        for (name, widths, rows) in &ragged_cases {
            let c = make_case(kv, 8, 8, widths, *rows, args.seed ^ 0xC2);
            let want = reference(&c);
            let got = dev.run(&c, true, &c.cu())?;
            let (bad, worst, pad) = compare(&c, &got, &want);
            note(
                format!(
                    "(b) {} {name} {widths:?} in r{rows}: {bad} element(s) out of tolerance, max \
                     |d| {worst:.2e}, {pad} padding element(s) nonzero",
                    kv_name(kv)
                ),
                bad == 0 && pad == 0,
                &mut failed,
            );
            // CONTROL 1: the uniform kernel on the ragged batch must NOT pass. It covers
            // `batch * max_w` rows, so it runs over a rectangle-sized copy of the case.
            let mut rect = c.clone();
            rect.rows = c.batch * c.max_w;
            rect.q.resize((rect.rows * Q_HEADS * HEAD_DIM) as usize, 0);
            let u = dev.run(&rect, false, &rect.cu())?;
            let (ubad, uworst, _) = compare(&c, &u[..got.len()], &want);
            note(
                format!(
                    "(b) CONTROL {} {name}: the uniform kernel fed it — {ubad} element(s) out of \
                     tolerance, max |d| {uworst:.2e} (must fail)",
                    kv_name(kv)
                ),
                ubad > 0,
                &mut failed,
            );
            // CONTROL 2: an off-by-one `cu_q` (every boundary one row late) must NOT pass.
            let mut off = c.cu();
            for x in off.iter_mut().skip(1).take(c.batch as usize - 1) {
                *x = (*x + 1).min(c.rows as i32);
            }
            let o = dev.run(&c, true, &off)?;
            let (obad, oworst, _) = compare(&c, &o, &want);
            note(
                format!(
                    "(b) CONTROL {} {name}: cu off by one — {obad} element(s) out of tolerance, \
                     max |d| {oworst:.2e} (must fail)",
                    kv_name(kv)
                ),
                obad > 0,
                &mut failed,
            );

            // ---- (c) per-row identity: each slot alone at its own width -------------------
            let cu = c.cu();
            let row_elems = (Q_HEADS * HEAD_DIM) as usize;
            let mut differ = 0usize;
            let mut compared = 0usize;
            for (n, w) in widths.iter().enumerate() {
                if *w == 0 {
                    continue;
                }
                let mut solo = c.clone();
                solo.batch = 1;
                solo.max_w = *w;
                solo.widths = vec![*w];
                solo.rows = *w;
                solo.done = vec![c.done[n]];
                let mb = max_blocks() as usize;
                solo.block_table = c.block_table[n * mb..(n + 1) * mb].to_vec();
                let lo = cu[n] as usize * row_elems;
                let hi = lo + *w as usize * row_elems;
                solo.q = c.q[lo..hi].to_vec();
                let alone = dev.run(&solo, true, &solo.cu())?;
                compared += alone.len();
                differ += alone
                    .iter()
                    .zip(&got[lo..hi])
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
            }
            note(
                format!(
                    "(c) {} {name}: every slot alone at its own width vs its rows in the batch — \
                     {differ} of {compared} element(s) differ",
                    kv_name(kv)
                ),
                differ == 0,
                &mut failed,
            );
        }
    }
    println!(
        "ragged-attn gate: {}",
        if failed == 0 {
            "every check OK".to_owned()
        } else {
            format!("{failed} check(s) FAILED")
        }
    );
    if let Some(p) = &args.out {
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(
            p,
            format!(
                "{{\"gate\":\"c2_ragged_attn\",\"failed\":{failed},\"checks\":[{}]}}\n",
                verdicts.join(",")
            ),
        )?;
        println!("wrote {}", p.display());
    }
    Ok(failed)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The e4m3 decoder is the reference's, so it is pinned at its corners: the gate's generated
    /// bytes never reach NaN, and a wrong decoder would make every fp8 case a false red.
    #[test]
    fn the_fp8_decoder_matches_e4m3fn() {
        assert_eq!(fp8_e4m3(0x00), 0.0);
        assert_eq!(fp8_e4m3(0x38), 1.0); // e 7, m 0
        assert_eq!(fp8_e4m3(0xb8), -1.0);
        assert_eq!(fp8_e4m3(0x3c), 1.5);
        assert_eq!(fp8_e4m3(0x7e), 448.0); // the largest finite
        assert_eq!(fp8_e4m3(0x01), 2f64.powi(-9)); // the smallest subnormal
        assert_eq!(bf16(to_bf16(1.0)), 1.0);
        assert_eq!(bf16(to_bf16(-0.5)), -0.5);
    }

    /// **The host pre-flight of arm (b)**: on a SMALL ragged case the reference is self-consistent
    /// (a padding row is zero, a live row is a convex combination of values), and the uniform
    /// kernel's reading of the same batch — the gate's first negative control — is outside the
    /// tolerance, so the control is a control before any card time is spent on it. The off-by-one
    /// `cu` reading is checked the same way.
    #[test]
    fn the_uniform_reading_of_a_ragged_batch_fails_the_tolerance() {
        let c = make_case(KvDtype::Fp8E4m3, 4, 4, &[3, 1, 4, 0], 12, 7);
        // Keep the host reference cheap: short contexts.
        let c = Case {
            done: vec![20, 33, 17, 0],
            ..c
        };
        let want = reference(&c);
        let (bad, worst, pad) = compare(&c, &want, &want);
        assert_eq!((bad, pad), (0, 0));
        assert_eq!(worst, 0.0);
        // Rows 8..12 are the band's padding: zero.
        let row = (Q_HEADS * HEAD_DIM) as usize;
        assert!(want[8 * row..].iter().all(|x| *x == 0.0));
        let u = uniform_reading(&c);
        let (ubad, _, _) = compare(&c, &u, &want);
        assert!(ubad > 0, "the uniform reading must fail the tolerance");
        // And the reference is the uniform reading at UNIFORM widths (the bit-exact arm's
        // premise, on the host).
        let uni = Case {
            widths: vec![4, 4, 4, 0],
            rows: 16,
            q: {
                let mut q = c.q.clone();
                q.resize(16 * row, 0);
                q
            },
            ..c.clone()
        };
        let (b2, w2, _) = compare(&uni, &uniform_reading(&uni), &reference(&uni));
        assert_eq!(b2, 0);
        assert!(w2 < 1e-9, "{w2}");
    }

    #[test]
    fn a_case_is_a_function_of_its_seed() {
        let a = make_case(KvDtype::Bf16, 2, 5, &[5, 2], 8, 3);
        let b = make_case(KvDtype::Bf16, 2, 5, &[5, 2], 8, 3);
        assert_eq!(a.q, b.q);
        assert_eq!(a.kv_bytes, b.kv_bytes);
        assert_eq!(a.cu(), vec![0, 5, 7]);
        assert_eq!(
            a.seqused(),
            vec![(a.done[0] + 5) as i32, (a.done[1] + 2) as i32]
        );
    }
}
