//! INK12: the Inkling 2 block drafter on the card (rank 0; docs/INKLING2.md section 6) and the
//! verify's conv-state rollback. Weights = the pack's `draft/` records (`build_ink2_draft.py`,
//! checked by `check_ink2_draft.py`: the checkpoint's own bf16 bytes + the f64 Markov fold stored
//! bf16). Bodies: `ink_draft` (Saber's own arithmetic, f32) + the executor's exact M <= 8 GEMV.
//! The host oracle is `tools/capture/ink12/dspark_ref.py`.

use super::{Card, PackReader, Reg, Tw, A, D, VOCAB};
use anyhow::{ensure, Context};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

const H: u64 = 24;
const KV: u64 = 8;
const HD: u64 = 128;
const FF: u64 = 9216;
const BLOCK: u64 = 8;
const VPAD: u64 = 201_024;
const EPS: f32 = 1e-6;

struct DL {
    dn: u64,
    ln1: u64,
    ln2: u64,
    qn: u64,
    kn: u64,
    knc: u64,
    q: Tw,
    k: Tw,
    v: Tw,
    kc: Tw,
    vc: Tw,
    o: Tw,
    gate: Tw,
    up: Tw,
    down: Tw,
}

/// INK18: the drafter's options (`--ink-draft-vocab`, `--ink-draft-gemv`, `--ink-draft-layers`). The
/// drafter's numerics are free under the drawn-row walk (a draft moves acceptance, never a committed
/// token): each option is gated on the MTP law + acceptance, not the byte law.
#[derive(Clone, Debug, Default)]
pub struct DraftOpt {
    /// draft over the first N ids of `<draft>/vocab_rank.u32` (0 = the full head, the control)
    pub vocab: u64,
    /// the split-K GEMV (`ink_dr_gemv_sk_*`) for the block layers, the context K / V and the
    /// Markov head's projections (false = the target's exact GEMV, the control)
    pub sk: bool,
    /// the split-K GEMV's most column passes a workgroup (`sk` 4, `sk1` 1)
    pub np_max: u64,
    /// the block layers' q / k / v / o / gate / up / down in 8 bits (int8 per-32 RTN at load)
    pub w8: bool,
    /// INK20 (`--ink-mtp-batch`): context slots (one a served sequence; 0 / 1 = one, the
    /// INK13 owner form) and the batched pass's most sequences
    pub slots: u64,
    /// INK21 (`--ink-draft-mm wmma`): the batched pass's M > 8 GEMMs on the matrix cores
    /// (`ink_dr4_mm_*`, A rounded to bf16; false = the target's f32 GEMM, the control)
    pub wmma: bool,
    /// INK22 (`--ink-batch-append ...,pfmm`): a context append of > 8 rows (a prefill chunk) as
    /// matrix-core GEMMs of <= 64 rows (`ink_dr4_mm_*`; false = GEMVs of 8 rows, the control)
    pub pfmm: bool,
    /// INK33 (`--ink-mtp-gate`): the single-sequence pass also writes each draft's top-1
    /// probability (`ink_dr6`: f32 bits in `ptoks` words 9..15; false = `ink_dr_fold` /
    /// `ink_dr_argmax_map`, the control); needs the pruned vocabulary
    pub conf: bool,
    /// INK38 (`--ink-mtp-bgate`): the BATCHED pass also writes each sequence's drafts' top-1
    /// probabilities (`ink_dr6` per sequence on the batched rows: the same fold + argmax as
    /// `ink_dr3_fold` / `ink_dr3_argmax_map` + the exp sum; false = the batched kernels, the
    /// control); needs the pruned vocabulary
    pub mconf: bool,
}

/// INK21: `ink_dr4_mm_*`'s grid -- (K slices, column workgroups): ~256 workgroups, the slice
/// count the largest divisor of K's 32-blocks at or below 256 / workgroups.
pub fn dr4_split(n: u64, k: u64) -> (u64, u64) {
    let nwg = n.div_ceil(32);
    let target = (256 / nwg).max(1);
    let blocks = (k / 32).max(1);
    let ks = (1..=target.min(blocks))
        .rev()
        .find(|d| blocks % d == 0)
        .unwrap_or(1);
    (ks, nwg)
}

/// INK20: the batched pass's most sequences.
pub const MB: u64 = 8;
/// INK22: the batched append's most rows (a b8 x 4 multi verify).
pub const AR: u64 = 32;

impl DraftOpt {
    /// INK21: `--ink-draft-mm base | wmma`.
    pub fn with_mm(mut self, mm: &str) -> anyhow::Result<Self> {
        ensure!(
            matches!(mm, "base" | "wmma"),
            "--ink-draft-mm {mm:?}: wmma | base (the control)"
        );
        self.wmma = mm == "wmma";
        Ok(self)
    }

    pub fn of(vocab: u64, gemv: &str, layers: &str) -> anyhow::Result<Self> {
        ensure!(
            matches!(layers, "w8" | "bf16"),
            "--ink-draft-layers {layers:?}: w8 | bf16 (the control)"
        );
        ensure!(
            matches!(gemv, "base" | "sk" | "sk1"),
            "--ink-draft-gemv {gemv:?}: sk | sk1 (one column pass a workgroup) | base (the control)"
        );
        ensure!(
            vocab == 0 || (1024..=VOCAB as u64).contains(&vocab),
            "--ink-draft-vocab {vocab}: 0 (the full head) or 1,024..{VOCAB}"
        );
        Ok(Self {
            vocab,
            sk: gemv != "base",
            np_max: if gemv == "sk1" { 1 } else { 4 },
            w8: layers == "w8",
            slots: 1,
            wmma: false,
            pfmm: false,
            conf: false,
            mconf: false,
        })
    }
}

/// INK18: the pruned draft vocabulary: the head rows, fold rows and fold biases of `n` token ids
/// (ascending), gathered at load; `map[i]` = the id of row i.
struct Sub {
    head: Tw,
    fw: u64,
    fb: u64,
    map: u64,
    n: u64,
}

/// INK18: int8 round-half-even codes a 32-k block + one bf16 scale = the smallest bf16 >=
/// amax / 127 (INK14's record law; code x scale exact in f32).
fn quant_w8(b: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let w = bf16s(b);
    let mut codes = vec![0u8; w.len()];
    let mut sc = Vec::with_capacity(w.len() / 16);
    for (bi, blk) in w.chunks_exact(32).enumerate() {
        let t = blk.iter().fold(0f32, |m, v| m.max(v.abs())) / 127.0;
        let mut s16 = (t.to_bits() >> 16) as u16;
        if f32::from_bits(u32::from(s16) << 16) < t {
            s16 += 1;
        }
        let s = f32::from_bits(u32::from(s16) << 16);
        sc.extend_from_slice(&s16.to_le_bytes());
        for (j, v) in blk.iter().enumerate() {
            let q = if s == 0.0 {
                0.0
            } else {
                (v / s).round_ties_even().clamp(-127.0, 127.0)
            };
            codes[bi * 32 + j] = (q as i8) as u8;
        }
    }
    (codes, sc)
}

/// INK18: the drafter's split-K GEMV cell (slices, passes): the K slice a multiple of 256 and at
/// most 768 (the A slice in LDS), the fewest slices that give >= 256 workgroups of 32 columns;
/// then up to 4 passes of 32 columns a workgroup over the one staged slice while >= 192 remain.
fn sk_split(n: u64, k: u64, np_max: u64) -> Option<(u64, u64)> {
    let ks = sk_slices(n, k)?;
    let mut np = 1;
    while np < np_max && n.div_ceil(32 * np * 2) * ks >= 192 {
        np *= 2;
    }
    Some((ks, np))
}

fn sk_slices(n: u64, k: u64) -> Option<u64> {
    let nb = n.div_ceil(32);
    let mut best = None;
    for ks in 1..=64 {
        if k % ks != 0 {
            continue;
        }
        let kl = k / ks;
        if kl % 256 != 0 || kl > 768 {
            continue;
        }
        best = Some(ks);
        if nb * ks >= 256 {
            break;
        }
    }
    best
}

/// The drafter's weights + one sequence's context caches on rank 0's card.
pub struct Drafter {
    l: Vec<DL>,
    norm: u64,
    mdn: u64,
    alpha: [[f32; 4]; 4],
    mask: Vec<u8>,
    embed_norm: Vec<f32>,
    head: Vec<Tw>,
    w1: u64,
    wlo: u64,
    wa: Tw,
    wb: Tw,
    br: u64,
    fw: u64,
    fb: u64,
    /// per layer: the context K / V `[slots][cap][8][128]` f32 by absolute position (INK20: one
    /// region a context slot, `cstride` bytes apart; slot `cur` is the single-sequence pass's)
    kc: Vec<u64>,
    vc: Vec<u64>,
    cstride: u64,
    /// INK20: the context slots, the current one, each slot's appended rows (`nctx` = `cur`'s)
    pub nslots: u64,
    cur: usize,
    ctx_n: Vec<u64>,
    /// INK20: the batched pass's inputs as ONE buffer: x [MB][8][D] ; ppos [MB] ; slots [MB] ;
    /// ptoks [MB][16] (i32); the append positions [slots]; the pass's GRU states [2][MB][256],
    /// latents [MB][256], fold partials [MB][...], the f_rnn rows [MB][D]; graphs by (B, k)
    minb: u64,
    apos: u64,
    ms: [u64; 2],
    mlat: u64,
    mpart: u64,
    mfrow: u64,
    mgraphs: BTreeMap<(u64, u64), (saber_hip::Graph, saber_hip::GraphExec)>,
    /// INK22: the batched append's fixed buffers: the row table (positions [AR] i32 ; slots [AR]
    /// i32, slot < 0 = skip ; f_rnn slots [AR] i32, set on a slot's LAST row) and the rows'
    /// scratch (mix [AR][D], K / V [AR][KV][HD], f_rnn [AR][D]); its graphs by row count
    agraphs: BTreeMap<u64, (saber_hip::Graph, saber_hip::GraphExec)>,
    atab: u64,
    amx: u64,
    ak: u64,
    av: u64,
    af: u64,
    pub cap: u64,
    /// the target's taps of the last pass: 4 planes `[tcap][D]` f32
    pub taps: u64,
    pub tcap: u64,
    f_rnn: u64,
    ppos: u64,
    ptoks: u64,
    /// fixed buffers of a block pass
    x0: u64,
    s: [u64; 2],
    lat: u64,
    part: u64,
    /// INK33: the fold workgroups' sums of exp(lg - cap) (`opt.conf`)
    psum: u64,
    zero: u64,
    /// rows appended to the context so far (= the next anchor's position when in sync)
    pub nctx: u64,
    /// INK13: the round's inputs as ONE buffer (x0 ; ppos 16 B ; ptoks 64 B): one blocking write
    inb: u64,
    /// INK13: the drafter round captured as one graph a depth k (INK20: a (slot, k))
    graphs: BTreeMap<(usize, u64), (saber_hip::Graph, saber_hip::GraphExec)>,
    /// INK18: the options, the pruned vocabulary, the split-K scratch (partials, counters)
    pub opt: DraftOpt,
    sub: Option<Sub>,
    skp: u64,
    skp_n: u64,
    skc: u64,
}

fn bf16s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
        .collect()
}

impl Drafter {
    #[allow(clippy::too_many_lines)]
    pub fn load(
        c: &mut Card<'_, '_>,
        pk: &PackReader,
        dir: &Path,
        head0: &Tw,
        cap: u64,
        tcap: u64,
        opt: &DraftOpt,
    ) -> anyhow::Result<Self> {
        let man: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("manifest.json")).with_context(|| {
                format!(
                    "{}: no drafter records (build_ink2_draft.py)",
                    dir.display()
                )
            })?,
        )?;
        let recs = &man["records"];
        let read = |n: &str| -> anyhow::Result<(Vec<u8>, Vec<u64>)> {
            let r = &recs[n];
            let f = r["file"]
                .as_str()
                .with_context(|| format!("draft record {n}"))?;
            let b = std::fs::read(dir.join(f))?;
            let shape: Vec<u64> = r["shape"]
                .as_array()
                .context("shape")?
                .iter()
                .filter_map(Value::as_u64)
                .collect();
            ensure!(
                b.len() as u64 == shape.iter().product::<u64>() * 2,
                "draft record {n}: {} bytes vs {shape:?}",
                b.len()
            );
            Ok((b, shape))
        };
        let tw = |c: &mut Card<'_, '_>, n: &str| -> anyhow::Result<Tw> {
            let (b, shape) = read(n)?;
            Ok(Tw {
                addr: c.up(&b)?,
                es: 2,
                shape,
                sc: 0,
            })
        };
        // INK18: the block layers' projections in 8 bits (`--ink-draft-w8`)
        let twq = |c: &mut Card<'_, '_>, n: &str| -> anyhow::Result<Tw> {
            if !opt.w8 {
                return tw(c, n);
            }
            let (b, shape) = read(n)?;
            let (codes, sc) = quant_w8(&b);
            let t = Tw {
                addr: c.up(&codes)?,
                es: 1,
                shape,
                sc: c.up(&sc)?,
            };
            ensure!(
                t.addr % 16 == 0 && t.sc % 16 == 0,
                "draft w8 {n}: unaligned upload"
            );
            Ok(t)
        };
        let mut l = Vec::new();
        for i in 0..3 {
            let p = |s: &str| format!("layers.{i}.{s}");
            l.push(DL {
                dn: tw(c, &p("dflare_norm.weight"))?.addr,
                ln1: tw(c, &p("input_layernorm.weight"))?.addr,
                ln2: tw(c, &p("post_attention_layernorm.weight"))?.addr,
                qn: tw(c, &p("self_attn.q_norm.weight"))?.addr,
                kn: tw(c, &p("self_attn.k_norm.weight"))?.addr,
                knc: tw(c, &p("self_attn.k_norm_ctx.weight"))?.addr,
                q: twq(c, &p("self_attn.q_proj.weight"))?,
                k: twq(c, &p("self_attn.k_proj.weight"))?,
                v: twq(c, &p("self_attn.v_proj.weight"))?,
                kc: tw(c, &p("self_attn.k_proj_ctx.weight"))?,
                vc: tw(c, &p("self_attn.v_proj_ctx.weight"))?,
                o: twq(c, &p("self_attn.o_proj.weight"))?,
                gate: twq(c, &p("mlp.gate_proj.weight"))?,
                up: twq(c, &p("mlp.up_proj.weight"))?,
                down: twq(c, &p("mlp.down_proj.weight"))?,
            });
        }
        // softmax(fuse_weight) rows (f32 on the host, as the oracle)
        let fwt = bf16s(&read("fuse_weight")?.0);
        let mut alpha = [[0f32; 4]; 4];
        for (r, row) in alpha.iter_mut().enumerate() {
            let m = (0..4)
                .map(|j| fwt[r * 4 + j])
                .fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = (0..4).map(|j| (fwt[r * 4 + j] - m).exp()).collect();
            let s: f32 = e.iter().sum();
            for j in 0..4 {
                row[j] = e[j] / s;
            }
        }
        let (en, _, _) = pk.tensor("model.embed_norm.weight", None)?;
        // INK14: under `--ink-trunk w8-head` the target head is the 8-bit record (the drafter
        // reads the same kind for both halves)
        let head1 = super::put_dense(c, pk, "lm_head.weight", Some(1), true)?;
        ensure!(
            head0.rows() + head1.rows() == VPAD,
            "the head halves: {} + {} rows",
            head0.rows(),
            head1.rows()
        );
        let head = vec![
            Tw {
                addr: head0.addr,
                es: head0.es,
                shape: head0.shape.clone(),
                sc: head0.sc,
            },
            head1,
        ];
        let norm = tw(c, "norm.weight")?.addr;
        let mdn = tw(c, "markov_head.dflare_norm.weight")?.addr;
        let w1 = tw(c, "markov_head.markov_w1.weight")?.addr;
        let wlo = tw(c, "rnn_lo")?.addr;
        let wa = tw(c, "rnn_a")?;
        let wb = tw(c, "rnn_b")?;
        let br = tw(c, "markov_head.rnn_proj.bias")?.addr;
        let fw = tw(c, "fold_w")?.addr;
        let fb = tw(c, "fold_b")?.addr;
        // INK18: the pruned vocabulary (`--ink-draft-vocab N`): the first N ids of the rank file,
        // ascending; their head rows (the target head's kind, both halves), fold rows and biases
        let sub = if opt.vocab == 0 {
            None
        } else {
            let rf = dir.join("vocab_rank.u32");
            let rb = std::fs::read(&rf).with_context(|| {
                format!("{}: no rank file (build_ink2_draft_vocab.py)", rf.display())
            })?;
            ensure!(
                rb.len() == VOCAB * 4,
                "{}: {} bytes",
                rf.display(),
                rb.len()
            );
            let mut ids: Vec<u32> = rb
                .chunks_exact(4)
                .take(opt.vocab as usize)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            ids.sort_unstable();
            ids.dedup();
            ensure!(
                ids.len() as u64 == opt.vocab && ids.iter().all(|&v| (v as usize) < VOCAB),
                "{}: not a permutation prefix",
                rf.display()
            );
            let mut halves = Vec::new();
            for r in 0..2u32 {
                if let Some((codes, sc, shape)) = pk.tensor_w8("lm_head.weight", Some(r), true)? {
                    halves.push((codes, Some(sc), shape[0]));
                } else {
                    let (b, es, shape) = pk.tensor("lm_head.weight", Some(r))?;
                    ensure!(es == 2, "lm_head rank {r}: es {es}");
                    halves.push((b, None, shape[0]));
                }
            }
            let w8 = halves[0].1.is_some();
            ensure!(
                w8 == (head0.sc != 0) && halves[0].2 == head0.rows(),
                "the pruned head: kind / rows differ from the target's"
            );
            let rbytes = (if w8 { D } else { 2 * D }) as usize;
            let sbytes = (D / 32 * 2) as usize;
            let (mut hb, mut hs) = (Vec::new(), Vec::new());
            let fwt = read("fold_w")?.0;
            let fbt = read("fold_b")?.0;
            let (mut fwb, mut fbb, mut mb) = (Vec::new(), Vec::new(), Vec::new());
            for &v in &ids {
                let v = u64::from(v);
                let (h, i) = if v < halves[0].2 {
                    (&halves[0], v)
                } else {
                    (&halves[1], v - halves[0].2)
                };
                let i = i as usize;
                hb.extend_from_slice(&h.0[i * rbytes..(i + 1) * rbytes]);
                if let Some(sc) = &h.1 {
                    hs.extend_from_slice(&sc[i * sbytes..(i + 1) * sbytes]);
                }
                let v = v as usize;
                fwb.extend_from_slice(&fwt[v * 512..(v + 1) * 512]);
                fbb.extend_from_slice(&fbt[v * 2..v * 2 + 2]);
                mb.extend_from_slice(&(v as u32).to_le_bytes());
            }
            let head = Tw {
                addr: c.up(&hb)?,
                es: if w8 { 1 } else { 2 },
                shape: vec![opt.vocab, D],
                sc: if w8 { c.up(&hs)? } else { 0 },
            };
            ensure!(
                head.addr % 16 == 0 && head.sc % 16 == 0,
                "the pruned head: unaligned upload"
            );
            println!(
                "INK18 draft vocab: {} ids of {} ({} head, {:.1} MB head + {:.1} MB fold)",
                opt.vocab,
                rf.display(),
                if w8 { "8-bit" } else { "bf16" },
                (hb.len() + hs.len()) as f64 / 1e6,
                fwb.len() as f64 / 1e6
            );
            Some(Sub {
                head,
                fw: c.up(&fwb)?,
                fb: c.up(&fbb)?,
                map: c.up(&mb)?,
                n: opt.vocab,
            })
        };
        let skp_n = 3 * 8 * VPAD;
        let (skp, skc) = if opt.sk {
            (
                c.alloc_zero(skp_n * 4)?,
                c.alloc_zero(VPAD.div_ceil(32) * 4)?,
            )
        } else {
            (0, 0)
        };
        let mask = read("mask_embed")?.0;
        let mask: Vec<u8> = bf16s(&mask).iter().flat_map(|v| v.to_le_bytes()).collect();
        let nslots = opt.slots.max(1);
        ensure!(nslots <= 8, "drafter: {nslots} context slots (1..8)");
        let cstride = cap * KV * HD * 4;
        let mut kc = Vec::new();
        let mut vc = Vec::new();
        for _ in 0..3 {
            kc.push(c.alloc_zero(nslots * cstride)?);
            vc.push(c.alloc_zero(nslots * cstride)?);
        }
        let (minb, apos, ms, mlat, mpart, mfrow) = if nslots > 1 {
            (
                c.alloc_zero(MB * BLOCK * D * 4 + MB * 4 * 2 + MB * 16 * 4)?,
                c.alloc_zero(nslots * 4)?,
                [c.alloc_zero(MB * 256 * 4)?, c.alloc_zero(MB * 256 * 4)?],
                c.alloc_zero(MB * 256 * 4)?,
                c.alloc_zero(MB * VPAD.div_ceil(64) * 8)?,
                c.alloc_zero(MB * D * 4)?,
            )
        } else {
            (0, 0, [0, 0], 0, 0, 0)
        };
        let (atab, amx, ak, av, af) = if nslots > 1 {
            (
                c.alloc_zero(AR * 4 * 3)?,
                c.alloc_zero(AR * D * 4)?,
                c.alloc_zero(AR * KV * HD * 4)?,
                c.alloc_zero(AR * KV * HD * 4)?,
                c.alloc_zero(AR * D * 4)?,
            )
        } else {
            (0, 0, 0, 0, 0)
        };
        let taps = c.alloc_zero(4 * tcap * D * 4)?;
        // INK13: x0 [8][D] f32 ; ppos (16 B) ; ptoks (64 B) contiguous
        let inb = c.alloc_zero(BLOCK * D * 4 + 16 + 64)?;
        Ok(Self {
            l,
            norm,
            mdn,
            alpha,
            mask,
            embed_norm: bf16s(&en),
            head,
            w1,
            wlo,
            wa,
            wb,
            br,
            fw,
            fb,
            kc,
            vc,
            cstride,
            nslots,
            cur: 0,
            ctx_n: vec![0; nslots as usize],
            minb,
            apos,
            ms,
            mlat,
            mpart,
            mfrow,
            mgraphs: BTreeMap::new(),
            agraphs: BTreeMap::new(),
            atab,
            amx,
            ak,
            av,
            af,
            cap,
            taps,
            tcap,
            f_rnn: c.alloc_zero(nslots * D * 4)?,
            ppos: inb + BLOCK * D * 4,
            ptoks: inb + BLOCK * D * 4 + 16,
            x0: inb,
            inb,
            graphs: BTreeMap::new(),
            s: [c.alloc_zero(256 * 4)?, c.alloc_zero(256 * 4)?],
            lat: c.alloc_zero(256 * 4)?,
            zero: c.alloc_zero(256 * 4)?,
            part: c.alloc_zero(VPAD.div_ceil(64) * 8)?,
            psum: c.alloc_zero(VPAD.div_ceil(64) * 4)?,
            nctx: 0,
            opt: opt.clone(),
            sub,
            skp,
            skp_n,
            skc,
        })
    }

    /// INK18: a drafter GEMV -- the split-K body under `--ink-draft-gemv sk` (M <= 8), else the
    /// target's exact GEMV / GEMM.
    /// INK21: an M > 8 GEMM of the batched pass on the matrix cores (`ink_dr4_mm_*`): 32
    /// columns a workgroup, K split so the grid holds ~256 workgroups (the slices' partials in the
    /// split-K scratch, summed in slice order by `ink_dr4_ksum`).
    #[allow(clippy::too_many_arguments)]
    fn mmw(
        &self,
        c: &mut Card<'_, '_>,
        a: u64,
        m: u64,
        k: u64,
        w: &Tw,
        out: u64,
        ldc: u64,
        acc: bool,
    ) -> anyhow::Result<()> {
        let n = w.rows();
        let (ks, nwg) = dr4_split(n, k);
        ensure!(
            k % 32 == 0 && m <= 64 && w.addr % 16 == 0 && (ks == 1 || ks * m * n <= self.skp_n),
            "draft wmma: {ks} x {m} x {n} (k {k})"
        );
        ensure!(
            ks == 1 || self.skp != 0,
            "draft wmma: split-K needs the sk scratch"
        );
        let i = |v: u64| -> anyhow::Result<A> { Ok(A::I(i32::try_from(v)?)) };
        c.launch(
            if w.sc != 0 {
                "ink_dr4_mm_w8"
            } else {
                "ink_dr4_mm_bf16"
            },
            [nwg, ks, 1],
            128,
            &[
                A::P(a),
                i(k)?,
                A::P(w.addr),
                A::P(if w.sc != 0 { w.sc } else { self.zero }),
                A::P(out),
                i(ldc)?,
                i(m)?,
                i(n)?,
                i(k)?,
                A::I(i32::from(acc)),
                i(ks)?,
                A::P(if ks > 1 { self.skp } else { self.zero }),
            ],
        )?;
        if ks > 1 {
            c.launch(
                "ink_dr4_ksum",
                [(m * n).div_ceil(256), 1, 1],
                256,
                &[
                    A::P(self.skp),
                    i(ks)?,
                    i(m)?,
                    i(n)?,
                    A::P(out),
                    i(ldc)?,
                    A::I(i32::from(acc)),
                ],
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn mm(
        &self,
        c: &mut Card<'_, '_>,
        a: u64,
        m: u64,
        k: u64,
        w: &Tw,
        out: u64,
        ldc: u64,
        acc: bool,
    ) -> anyhow::Result<()> {
        let n = w.rows();
        let ks = if self.opt.sk && m <= 8 {
            sk_split(n, k, self.opt.np_max)
        } else {
            None
        };
        let Some((ks, np)) = ks else {
            return c.gemm(a, m, k, w, out, ldc, acc);
        };
        ensure!(
            ks * m * n <= self.skp_n && w.addr % 16 == 0,
            "draft sk: {ks} x {m} x {n} past the scratch"
        );
        let i = |v: u64| -> anyhow::Result<A> { Ok(A::I(i32::try_from(v)?)) };
        c.launch(
            if w.sc != 0 {
                "ink_dr_gemv_sk_w8"
            } else {
                "ink_dr_gemv_sk_bf16"
            },
            [n.div_ceil(32 * np), ks, 1],
            256,
            &[
                A::P(a),
                i(k)?,
                A::P(w.addr),
                A::P(if w.sc != 0 { w.sc } else { self.zero }),
                A::P(out),
                i(ldc)?,
                i(m)?,
                i(n)?,
                i(k)?,
                A::I(i32::from(acc)),
                i(ks)?,
                i(np)?,
                A::P(self.skp),
                A::P(self.skc),
            ],
        )
    }

    /// INK13: forget the context (a new owner sequence; the next append starts at 0).
    pub fn reset_ctx(&mut self) {
        self.nctx = 0;
    }

    /// INK23 (`--ink-prefix-cache`): the current slot's context rows `0..rows` (per layer the K
    /// then the V region: address, bytes) and its f_rnn row (address, bytes) -- every byte a
    /// prefix node must hold for the drafter to continue at `rows` (all of it row-local: the K / V
    /// rows of the target's taps, f_rnn = mix_3 of the last appended row).
    pub fn ctx_regions(&self, rows: u64) -> Vec<(u64, u64)> {
        let mut v = Vec::with_capacity(7);
        for r in 0..self.kc.len() {
            v.push((self.kc_of(r), rows * KV * HD * 4));
            v.push((self.vc_of(r), rows * KV * HD * 4));
        }
        v.push((self.f_rnn_of(), D * 4));
        v
    }

    /// INK23: the current slot's context now holds `n` rows (a prefix node was copied in).
    pub fn set_nctx(&mut self, n: u64) {
        self.nctx = n;
    }

    /// INK20: make context slot `slot` the current one (the single-sequence pass, append, prep).
    pub fn select(&mut self, slot: usize) -> anyhow::Result<()> {
        ensure!(
            (slot as u64) < self.nslots,
            "drafter slot {slot} of {}",
            self.nslots
        );
        self.ctx_n[self.cur] = self.nctx;
        self.cur = slot;
        self.nctx = self.ctx_n[slot];
        Ok(())
    }

    /// INK20: a slot's appended rows.
    pub fn nctx_of(&self, slot: usize) -> u64 {
        // INK21: a one-slot drafter (the INK13 owner form) holds ONE context whatever the served
        // slot (INK20 indexed `ctx_n` by the slot: a lone sequence in slot 7 panicked the plain boot)
        if slot == self.cur || self.ctx_n.len() <= 1 {
            self.nctx
        } else {
            self.ctx_n[slot]
        }
    }

    fn kc_of(&self, r: usize) -> u64 {
        self.kc[r] + self.cur as u64 * self.cstride
    }

    fn vc_of(&self, r: usize) -> u64 {
        self.vc[r] + self.cur as u64 * self.cstride
    }

    fn f_rnn_of(&self) -> u64 {
        self.f_rnn + self.cur as u64 * D * 4
    }

    /// Append the taps rows `0..t` of the last pass (the target rows at positions `pos0..`) to
    /// the context: per layer K = rope(k_norm_ctx(k_proj_ctx(mix_r))), V = v_proj_ctx(mix_r); the
    /// head's f_rnn = mix_3 of the last appended row.
    pub fn append(&mut self, c: &mut Card<'_, '_>, t: u64, pos0: u64) -> anyhow::Result<()> {
        c.write(self.ppos, &(pos0 as i32).to_le_bytes())?;
        self.append_at(c, t, pos0, 0, self.ppos)
    }

    /// INK20: append the rows of several slots after ONE multi-sequence pass: `rows` = (slot,
    /// rows t, position, the rows' first tap row); the positions in one write.
    pub fn append_multi(
        &mut self,
        c: &mut Card<'_, '_>,
        rows: &[(usize, u64, u64, u64)],
    ) -> anyhow::Result<()> {
        ensure!(self.apos != 0, "append_multi on a one-slot drafter");
        let mut pb = vec![0u8; self.nslots as usize * 4];
        for &(s, _, p, _) in rows {
            pb[s * 4..s * 4 + 4].copy_from_slice(&i32::try_from(p)?.to_le_bytes());
        }
        c.write(self.apos, &pb)?;
        let cur = self.cur;
        for &(s, t, p, r0) in rows {
            self.select(s)?;
            self.append_at(c, t, p, r0, self.apos + s as u64 * 4)?;
        }
        self.select(cur)
    }

    /// INK22: the plain decode step's append for its `b` rows (the bucket's; taps rows 0..b) as
    /// FIXED-address work, so it rides the step's captured graph: per layer the mix norm of the b
    /// rows, the context K / V GEMVs at M b (`mm`: per row the bytes of an M 1 call), the K
    /// norm + rope at each row's position (`ink_dr3_hnr`, rows 1), the rows scattered to their
    /// slots (`ink_dr5_scatter`); then the head's f_rnn rows. Which rows land (slot, position) is
    /// the table [`Self::write_atab`] writes before the replay; [`Self::appended`] books them.
    /// (INK22 rounds: the same form over a multi verify's `b` = B x W taps rows, GEMVs of 8
    /// rows, the f_rnn rows by their own slot table.)
    pub fn append_batch(&self, c: &mut Card<'_, '_>, b: u64) -> anyhow::Result<()> {
        ensure!(
            self.atab != 0 && (1..=AR).contains(&b) && b <= self.tcap,
            "the batched append on a one-slot drafter / {b} rows"
        );
        let plane = (self.tcap * D) as i64;
        let (tpos, tslot, tfs) = (self.atab, self.atab + AR * 4, self.atab + 2 * AR * 4);
        let scatter = |c: &mut Card<'_, '_>,
                       src: u64,
                       ld: u64,
                       dst: u64,
                       ss: u64,
                       ps: u64,
                       n: u64,
                       slots: u64|
         -> anyhow::Result<()> {
            c.launch(
                "ink_dr5_scatter",
                [b, n.div_ceil(256), 1],
                256,
                &[
                    A::P(src),
                    A::I(i32::try_from(ld)?),
                    A::P(dst),
                    A::L(i64::try_from(ss)?),
                    A::L(i64::try_from(ps)?),
                    A::I(i32::try_from(n)?),
                    A::P(slots),
                    A::P(tpos),
                ],
            )
        };
        for r in 0..3 {
            let a = self.alpha[r];
            c.launch(
                "ink_dr_mixnorm",
                [b, 1, 1],
                256,
                &[
                    A::P(self.taps),
                    A::L(plane),
                    A::I(0),
                    A::F(a[0]),
                    A::F(a[1]),
                    A::F(a[2]),
                    A::F(a[3]),
                    A::P(self.l[r].dn),
                    A::P(self.amx),
                    A::F(EPS),
                ],
            )?;
            let lw = &self.l[r];
            let (kw, vw) = (
                Tw {
                    addr: lw.kc.addr,
                    es: 2,
                    shape: lw.kc.shape.clone(),
                    sc: 0,
                },
                Tw {
                    addr: lw.vc.addr,
                    es: 2,
                    shape: lw.vc.shape.clone(),
                    sc: 0,
                },
            );
            let mut i = 0;
            while i < b {
                let m = (b - i).min(8);
                let (xa, ko, vo) = (
                    self.amx + i * D * 4,
                    self.ak + i * KV * HD * 4,
                    self.av + i * KV * HD * 4,
                );
                self.mm(c, xa, m, D, &kw, ko, KV * HD, false)?;
                self.mm(c, xa, m, D, &vw, vo, KV * HD, false)?;
                i += m;
            }
            c.launch(
                "ink_dr3_hnr",
                [b, KV, 1],
                64,
                &[
                    A::P(self.ak),
                    A::I((KV * HD) as i32),
                    A::P(lw.knc),
                    A::P(tpos),
                    A::I(0),
                    A::I(1),
                    A::F(EPS),
                ],
            )?;
            let row = KV * HD;
            scatter(
                c,
                self.ak,
                row,
                self.kc[r],
                self.cstride / 4,
                row,
                row,
                tslot,
            )?;
            scatter(
                c,
                self.av,
                row,
                self.vc[r],
                self.cstride / 4,
                row,
                row,
                tslot,
            )?;
        }
        let a = self.alpha[3];
        c.launch(
            "ink_dr_mixnorm",
            [b, 1, 1],
            256,
            &[
                A::P(self.taps),
                A::L(plane),
                A::I(0),
                A::F(a[0]),
                A::F(a[1]),
                A::F(a[2]),
                A::F(a[3]),
                A::P(self.mdn),
                A::P(self.af),
                A::F(EPS),
            ],
        )?;
        scatter(c, self.af, D, self.f_rnn, D, 0, D, tfs)
    }

    /// INK22: [`Self::append_batch`] of `b` rows as one captured graph (a round's append after
    /// its commit: the table written first, then one launch).
    pub fn append_batch_graph(&mut self, c: &mut Card<'_, '_>, b: u64) -> anyhow::Result<()> {
        if !self.agraphs.contains_key(&b) {
            c.hip.sync_stream(&c.stream)?;
            c.hip
                .begin_capture(&c.stream, saber_hip::StreamCaptureMode::ThreadLocal)?;
            let r = self.append_batch(c, b);
            let g = c.hip.end_capture(&c.stream)?;
            r?;
            let e = c.hip.instantiate(&g)?;
            self.agraphs.insert(b, (g, e));
        }
        let (_, e) = self.agraphs.get(&b).context("captured")?;
        c.hip.graph_launch(e, &c.stream)?;
        Ok(())
    }

    /// INK22: the batched append's row table: row i = (slot, position, the slot's last row:
    /// its f_rnn), `None` = skipped.
    pub fn write_atab(
        &self,
        c: &mut Card<'_, '_>,
        rows: &[Option<(usize, u64, bool)>],
    ) -> anyhow::Result<()> {
        ensure!(self.atab != 0 && rows.len() as u64 <= AR, "append table");
        let ar = AR as usize;
        let mut tb = vec![0u8; ar * 12];
        for i in 0..ar {
            let (s, p, fs) = match rows.get(i).copied().flatten() {
                Some((s, p, last)) => {
                    let s = i32::try_from(s)?;
                    (s, i32::try_from(p)?, if last { s } else { -1 })
                }
                None => (-1, 0, -1),
            };
            tb[i * 4..i * 4 + 4].copy_from_slice(&p.to_le_bytes());
            tb[(ar + i) * 4..(ar + i) * 4 + 4].copy_from_slice(&s.to_le_bytes());
            tb[(2 * ar + i) * 4..(2 * ar + i) * 4 + 4].copy_from_slice(&fs.to_le_bytes());
        }
        c.write(self.atab, &tb)
    }

    /// INK22: book one row the batched append wrote (slot `slot`, position `pos`).
    pub fn appended(&mut self, slot: usize, pos: u64) -> anyhow::Result<()> {
        ensure!(
            pos < self.cap && self.nctx_of(slot) == pos,
            "batched append at {pos} but slot {slot} holds {}",
            self.nctx_of(slot)
        );
        if slot == self.cur {
            self.nctx = pos + 1;
        } else {
            self.ctx_n[slot] = pos + 1;
        }
        Ok(())
    }

    /// INK22 (`check`): a slot's appended context row at `pos` (3 layers' K and V) + its f_rnn.
    pub fn ctx_rows(&self, c: &mut Card<'_, '_>, slot: usize, pos: u64) -> anyhow::Result<Vec<u8>> {
        let mut o = Vec::new();
        let off = slot as u64 * self.cstride + pos * KV * HD * 4;
        for r in 0..3 {
            o.extend(c.read(self.kc[r] + off, (KV * HD * 4) as usize)?);
            o.extend(c.read(self.vc[r] + off, (KV * HD * 4) as usize)?);
        }
        o.extend(c.read(self.f_rnn + slot as u64 * D * 4, (D * 4) as usize)?);
        Ok(o)
    }

    fn append_at(
        &mut self,
        c: &mut Card<'_, '_>,
        t: u64,
        pos0: u64,
        row0: u64,
        ppos: u64,
    ) -> anyhow::Result<()> {
        ensure!(
            t >= 1 && t <= self.tcap && pos0 + t <= self.cap,
            "draft append {t} rows at {pos0}"
        );
        ensure!(
            pos0 == self.nctx,
            "draft append at {pos0} but the context holds {}",
            self.nctx
        );
        ensure!(row0 + t <= self.tcap, "draft append rows {row0}+{t}");
        let plane = (self.tcap * D) as i64;
        for r in 0..3 {
            let a = self.alpha[r];
            let mx = c.z4(Reg::Visit, t * D, "dr.mix")?;
            c.launch(
                "ink_dr_mixnorm",
                [t, 1, 1],
                256,
                &[
                    A::P(self.taps),
                    A::L(plane),
                    A::I(i32::try_from(row0)?),
                    A::F(a[0]),
                    A::F(a[1]),
                    A::F(a[2]),
                    A::F(a[3]),
                    A::P(self.l[r].dn),
                    A::P(mx),
                    A::F(EPS),
                ],
            )?;
            let kdst = self.kc_of(r) + pos0 * KV * HD * 4;
            let vdst = self.vc_of(r) + pos0 * KV * HD * 4;
            // M <= 8 GEMV a chunk of 8 rows (exact per row), else the tiled GEMM
            let lw = &self.l[r];
            let (kw, vw) = (
                Tw {
                    addr: lw.kc.addr,
                    es: 2,
                    shape: lw.kc.shape.clone(),
                    sc: 0,
                },
                Tw {
                    addr: lw.vc.addr,
                    es: 2,
                    shape: lw.vc.shape.clone(),
                    sc: 0,
                },
            );
            let mut i = 0;
            while self.opt.pfmm && self.opt.wmma && t - i > 8 {
                // INK22 `pfmm`: a prefill chunk's rows on the matrix cores, <= 64 a GEMM
                let m = (t - i).min(64);
                self.mmw(
                    c,
                    mx + i * D * 4,
                    m,
                    D,
                    &kw,
                    kdst + i * KV * HD * 4,
                    KV * HD,
                    false,
                )?;
                self.mmw(
                    c,
                    mx + i * D * 4,
                    m,
                    D,
                    &vw,
                    vdst + i * KV * HD * 4,
                    KV * HD,
                    false,
                )?;
                i += m;
            }
            while i < t {
                let m = (t - i).min(8);
                self.mm(
                    c,
                    mx + i * D * 4,
                    m,
                    D,
                    &kw,
                    kdst + i * KV * HD * 4,
                    KV * HD,
                    false,
                )?;
                self.mm(
                    c,
                    mx + i * D * 4,
                    m,
                    D,
                    &vw,
                    vdst + i * KV * HD * 4,
                    KV * HD,
                    false,
                )?;
                i += m;
            }
            c.launch(
                "ink_dr_hnr",
                [t, KV, 1],
                64,
                &[
                    A::P(kdst),
                    A::I((KV * HD) as i32),
                    A::P(lw.knc),
                    A::P(ppos),
                    A::I(0),
                    A::F(EPS),
                ],
            )?;
            c.reset(Reg::Visit)?;
        }
        let a = self.alpha[3];
        c.launch(
            "ink_dr_mixnorm",
            [1, 1, 1],
            256,
            &[
                A::P(self.taps),
                A::L(plane),
                A::I(i32::try_from(row0 + t - 1)?),
                A::F(a[0]),
                A::F(a[1]),
                A::F(a[2]),
                A::F(a[3]),
                A::P(self.mdn),
                A::P(self.f_rnn_of()),
                A::F(EPS),
            ],
        )?;
        self.nctx = pos0 + t;
        Ok(())
    }

    /// One block for the anchor `tok` at position `p = nctx`: `k` drafts (<= 7). Returns the
    /// drafts and, with `keep`, the k logits rows (f32, `VOCAB` each) read back for the gate.
    pub fn draft(
        &mut self,
        c: &mut Card<'_, '_>,
        pk: &PackReader,
        tok: u32,
        k: u64,
        keep: bool,
    ) -> anyhow::Result<(Vec<u32>, Vec<u8>)> {
        self.prep(c, pk, tok, k)?;
        let keep_lg = self.body(c, k, keep)?;
        let toks = self.drafts(c, k)?;
        let lg = if keep {
            c.read(keep_lg, (k * VOCAB as u64 * 4) as usize)?
        } else {
            Vec::new()
        };
        Ok((toks, lg))
    }

    /// INK13: the round's inputs -- rows `[embed_normed(anchor) ; mask x 7]` (the anchor's norm
    /// on the host, f32), the anchor position `p = nctx`, the anchor token -- in ONE blocking write
    /// (ordered behind everything queued on the stream).
    pub fn prep(
        &mut self,
        c: &mut Card<'_, '_>,
        pk: &PackReader,
        tok: u32,
        k: u64,
    ) -> anyhow::Result<()> {
        ensure!((1..BLOCK).contains(&k), "draft depth {k}: 1..7");
        let p = self.nctx;
        ensure!(
            p + BLOCK <= self.cap,
            "draft at {p}: past the drafter's {} rows",
            self.cap
        );
        let raw: Vec<f32> = pk
            .embed_rows(&[tok])?
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let ss: f32 = raw.iter().map(|v| v * v).sum::<f32>() / D as f32;
        let r = 1.0 / (ss + EPS).sqrt();
        let mut x: Vec<u8> = raw
            .iter()
            .zip(&self.embed_norm)
            .flat_map(|(v, g)| ((v * r) * g).to_le_bytes())
            .collect();
        for _ in 1..BLOCK {
            x.extend_from_slice(&self.mask);
        }
        let mut pp = (p as i32).to_le_bytes().to_vec();
        pp.resize(16, 0);
        x.extend_from_slice(&pp);
        let mut tb = (tok as i32).to_le_bytes().to_vec();
        tb.resize(64, 0);
        x.extend_from_slice(&tb);
        c.write(self.inb, &x)
    }

    /// INK13: the round on the graph (captured at a depth's first use): `prep` first; the drafts
    /// are read by [`Self::drafts`] (the round is only queued here).
    pub fn issue(
        &mut self,
        c: &mut Card<'_, '_>,
        pk: &PackReader,
        tok: u32,
        k: u64,
    ) -> anyhow::Result<()> {
        self.prep(c, pk, tok, k)?;
        let ki = (self.cur, k);
        if !self.graphs.contains_key(&ki) {
            c.hip.sync_stream(&c.stream)?;
            let l0 = c.k.launches;
            c.hip
                .begin_capture(&c.stream, saber_hip::StreamCaptureMode::ThreadLocal)?;
            let r = self.body(c, k, false);
            let g = c.hip.end_capture(&c.stream)?;
            r?;
            let e = c.hip.instantiate(&g)?;
            println!(
                "INK13 drafter graph k {k}: {} launches, {} nodes",
                c.k.launches - l0,
                c.hip.graph_node_count(&g)?
            );
            self.graphs.insert(ki, (g, e));
        }
        let (_, e) = self.graphs.get(&ki).context("captured")?;
        c.hip.graph_launch(e, &c.stream)?;
        Ok(())
    }

    /// The k drafts of the last round (a blocking read: waits for the round).
    /// INK33: the round's drafts + each one's top-1 probability (`opt.conf`; ONE read).
    pub fn drafts_conf(
        &self,
        c: &mut Card<'_, '_>,
        k: u64,
    ) -> anyhow::Result<(Vec<u32>, Vec<f32>)> {
        ensure!(
            self.opt.conf && self.sub.is_some(),
            "draft confidence needs --ink-mtp-gate + the pruned vocabulary"
        );
        let tb = c.read(self.ptoks, 64)?;
        let w =
            |i: usize| u32::from_le_bytes([tb[4 * i], tb[4 * i + 1], tb[4 * i + 2], tb[4 * i + 3]]);
        Ok((
            (1..=k as usize).map(w).collect(),
            (1..=k as usize).map(|i| f32::from_bits(w(8 + i))).collect(),
        ))
    }

    pub fn drafts(&self, c: &mut Card<'_, '_>, k: u64) -> anyhow::Result<Vec<u32>> {
        let tb = c.read(self.ptoks, 64)?;
        Ok((1..=k as usize)
            .map(|i| u32::from_le_bytes([tb[4 * i], tb[4 * i + 1], tb[4 * i + 2], tb[4 * i + 3]]))
            .collect())
    }

    /// INK20: the batched round for `rows` = (context slot, anchor token) of B sequences (B <= 8,
    /// every slot's context at its anchor's position): the inputs in ONE write (x, positions,
    /// slots, anchor tokens), then the captured pass a (B, k) -- queued, not synchronised.
    pub fn issue_multi(
        &mut self,
        c: &mut Card<'_, '_>,
        pk: &PackReader,
        rows: &[(usize, u32)],
        k: u64,
    ) -> anyhow::Result<()> {
        let b = rows.len() as u64;
        ensure!(self.minb != 0, "the batched drafter on a one-slot drafter");
        ensure!(
            (1..=MB).contains(&b) && (1..BLOCK).contains(&k),
            "a {b}-sequence k {k} round"
        );
        ensure!(
            self.sub.is_some(),
            "the batched drafter needs the pruned vocabulary"
        );
        let toks: Vec<u32> = rows.iter().map(|r| r.1).collect();
        let emb = pk.embed_rows(&toks)?;
        let mut x = Vec::with_capacity((MB * BLOCK * D * 4 + 96 + MB * 64) as usize);
        for (i, &(slot, _)) in rows.iter().enumerate() {
            let p = self.nctx_of(slot);
            ensure!(
                p + BLOCK <= self.cap,
                "draft at {p}: past the drafter's {} rows",
                self.cap
            );
            let raw: Vec<f32> = emb[i * D as usize * 4..(i + 1) * D as usize * 4]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let ss: f32 = raw.iter().map(|v| v * v).sum::<f32>() / D as f32;
            let r = 1.0 / (ss + EPS).sqrt();
            x.extend(
                raw.iter()
                    .zip(&self.embed_norm)
                    .flat_map(|(v, g)| ((v * r) * g).to_le_bytes()),
            );
            for _ in 1..BLOCK {
                x.extend_from_slice(&self.mask);
            }
        }
        x.resize((MB * BLOCK * D * 4) as usize, 0);
        let mut pp = vec![0u8; MB as usize * 4];
        let mut sl = vec![0u8; MB as usize * 4];
        let mut tk = vec![0u8; (MB * 64) as usize];
        for (i, &(slot, tok)) in rows.iter().enumerate() {
            pp[i * 4..i * 4 + 4].copy_from_slice(&i32::try_from(self.nctx_of(slot))?.to_le_bytes());
            sl[i * 4..i * 4 + 4].copy_from_slice(&i32::try_from(slot)?.to_le_bytes());
            tk[i * 64..i * 64 + 4].copy_from_slice(&(tok as i32).to_le_bytes());
        }
        x.extend(pp);
        x.extend(sl);
        x.extend(tk);
        c.write(self.minb, &x)?;
        let key = (b, k);
        if !self.mgraphs.contains_key(&key) {
            c.hip.sync_stream(&c.stream)?;
            let l0 = c.k.launches;
            c.hip
                .begin_capture(&c.stream, saber_hip::StreamCaptureMode::ThreadLocal)?;
            let r = self.body_multi(c, b, k);
            let g = c.hip.end_capture(&c.stream)?;
            r?;
            let e = c.hip.instantiate(&g)?;
            println!(
                "INK20 drafter graph b {b} k {k}: {} launches, {} nodes",
                c.k.launches - l0,
                c.hip.graph_node_count(&g)?
            );
            self.mgraphs.insert(key, (g, e));
        }
        let (_, e) = self.mgraphs.get(&key).context("captured")?;
        c.hip.graph_launch(e, &c.stream)?;
        Ok(())
    }

    /// INK38: the k drafts + their top-1 probabilities of each of the last batched round's B
    /// sequences (`opt.mconf`; one blocking read).
    pub fn drafts_multi_conf(
        &self,
        c: &mut Card<'_, '_>,
        b: u64,
        k: u64,
    ) -> anyhow::Result<Vec<(Vec<u32>, Vec<f32>)>> {
        ensure!(
            self.opt.mconf,
            "batched draft confidence needs --ink-mtp-bgate"
        );
        let off = MB * BLOCK * D * 4 + MB * 8;
        let tb = c.read(self.minb + off, (b * 64) as usize)?;
        let w = |o: usize| u32::from_le_bytes([tb[o], tb[o + 1], tb[o + 2], tb[o + 3]]);
        Ok((0..b as usize)
            .map(|s| {
                (
                    (1..=k as usize).map(|i| w(s * 64 + 4 * i)).collect(),
                    (1..=k as usize)
                        .map(|i| f32::from_bits(w(s * 64 + 4 * (8 + i))))
                        .collect(),
                )
            })
            .collect())
    }

    /// INK20: the k drafts of each of the last batched round's B sequences (one blocking read).
    pub fn drafts_multi(
        &self,
        c: &mut Card<'_, '_>,
        b: u64,
        k: u64,
    ) -> anyhow::Result<Vec<Vec<u32>>> {
        let off = MB * BLOCK * D * 4 + MB * 8;
        let tb = c.read(self.minb + off, (b * 64) as usize)?;
        Ok((0..b as usize)
            .map(|s| {
                (1..=k as usize)
                    .map(|i| {
                        let o = s * 64 + 4 * i;
                        u32::from_le_bytes([tb[o], tb[o + 1], tb[o + 2], tb[o + 3]])
                    })
                    .collect()
            })
            .collect())
    }

    /// INK20: the batched pass's launches (graph-safe): B blocks of 8 rows through the three
    /// layers at M = 8 B (the GEMMs on the target's exact GEMV / GEMM: one sequential sum an
    /// output), each block's attention over its own slot's context at its own position, the
    /// pruned head over every row, then the Markov chain k steps a sequence.
    #[allow(clippy::too_many_lines)]
    pub(super) fn body_multi(&self, c: &mut Card<'_, '_>, nb: u64, k: u64) -> anyhow::Result<()> {
        let s = self.sub.as_ref().context("sub")?;
        let x0 = self.minb;
        let ppos = x0 + MB * BLOCK * D * 4;
        let slots = ppos + MB * 4;
        let ptoks = slots + MB * 4;
        let m = nb * BLOCK;
        c.reset(Reg::Pass)?;
        let hres = c.z4(Reg::Pass, m * D, "dr.h")?;
        c.launch(
            "ink_dr_copy",
            [64, 1, 1],
            256,
            &[A::P(x0), A::P(hres), A::L((m * D) as i64)],
        )?;
        let gm = |me: &Self,
                  c: &mut Card<'_, '_>,
                  a: u64,
                  kk: u64,
                  w: &Tw,
                  o: u64,
                  ld: u64,
                  acc: bool| {
            if m <= 8 {
                me.mm(c, a, m, kk, w, o, ld, acc)
            } else if me.opt.wmma {
                me.mmw(c, a, m, kk, w, o, ld, acc)
            } else {
                c.gemm(a, m, kk, w, o, ld, acc)
            }
        };
        for (li, lw) in self.l.iter().enumerate() {
            c.reset(Reg::Visit)?;
            let a = c.rmsnorm(Reg::Visit, hres, m, D, lw.ln1)?;
            let q = c.z4(Reg::Visit, m * H * HD, "dr.q")?;
            let kk = c.z4(Reg::Visit, m * KV * HD, "dr.k")?;
            let vv = c.z4(Reg::Visit, m * KV * HD, "dr.v")?;
            gm(self, c, a, D, &lw.q, q, H * HD, false)?;
            gm(self, c, a, D, &lw.k, kk, KV * HD, false)?;
            gm(self, c, a, D, &lw.v, vv, KV * HD, false)?;
            for (buf, nh, g) in [(q, H, lw.qn), (kk, KV, lw.kn)] {
                c.launch(
                    "ink_dr3_hnr",
                    [m, nh, 1],
                    64,
                    &[
                        A::P(buf),
                        A::I((nh * HD) as i32),
                        A::P(g),
                        A::P(ppos),
                        A::I(0),
                        A::I(BLOCK as i32),
                        A::F(EPS),
                    ],
                )?;
            }
            let o = c.z4(Reg::Visit, m * H * HD, "dr.o")?;
            let window = if li == 1 { 0 } else { 1024 };
            c.launch(
                "ink_dr3_attn",
                [m, H, 1],
                256,
                &[
                    A::P(q),
                    A::I((H * HD) as i32),
                    A::P(self.kc[li]),
                    A::P(self.vc[li]),
                    A::L(i64::try_from(self.cstride / 4)?),
                    A::P(slots),
                    A::P(kk),
                    A::P(vv),
                    A::I((KV * HD) as i32),
                    A::P(ppos),
                    A::I(window),
                    A::P(o),
                    A::I((H * HD) as i32),
                    A::F((HD as f32).powf(-0.5)),
                ],
            )?;
            gm(self, c, o, H * HD, &lw.o, hres, D, true)?;
            let bn = c.rmsnorm(Reg::Visit, hres, m, D, lw.ln2)?;
            let g = c.z4(Reg::Visit, m * FF, "dr.g")?;
            let u = c.z4(Reg::Visit, m * FF, "dr.u")?;
            gm(self, c, bn, D, &lw.gate, g, FF, false)?;
            gm(self, c, bn, D, &lw.up, u, FF, false)?;
            c.launch(
                "ink_dr_silu_mul",
                [256, 1, 1],
                256,
                &[A::P(g), A::P(u), A::L((m * FF) as i64)],
            )?;
            gm(self, c, g, FF, &lw.down, hres, D, true)?;
        }
        c.reset(Reg::Visit)?;
        let out = c.rmsnorm(Reg::Pass, hres, m, D, self.norm)?;
        let (vn, vld) = (s.n, s.n);
        let base = c.z4(Reg::Pass, m * vld, "dr.base")?;
        gm(self, c, out, D, &s.head, base, vld, false)?;
        let ca = c.z4(Reg::Pass, m * 768, "dr.ca")?;
        let cb = c.z4(Reg::Pass, nb * 768, "dr.cb")?;
        gm(self, c, out, D, &self.wa, ca, 768, false)?;
        c.launch(
            "ink_dr3_gather",
            [D.div_ceil(256), nb, 1],
            256,
            &[
                A::P(self.f_rnn),
                A::L(D as i64),
                A::P(slots),
                A::P(self.mfrow),
                A::I(D as i32),
            ],
        )?;
        self.mm(c, self.mfrow, nb, D, &self.wb, cb, 768, false)?;
        c.launch(
            "ink_dr_copy",
            [8, 1, 1],
            256,
            &[A::P(self.zero), A::P(self.ms[0]), A::L(256)],
        )?;
        for i in 1..nb {
            c.launch(
                "ink_dr_copy",
                [1, 1, 1],
                256,
                &[A::P(self.zero), A::P(self.ms[0] + i * 1024), A::L(256)],
            )?;
        }
        let nwg = vn.div_ceil(64);
        for i in 0..k {
            let (si, so) = (self.ms[(i % 2) as usize], self.ms[((i + 1) % 2) as usize]);
            c.launch(
                "ink_dr3_gru",
                [256, nb, 1],
                256,
                &[
                    A::P(si),
                    A::P(self.w1),
                    A::P(ptoks + i * 4),
                    A::P(self.wlo),
                    A::I(512),
                    A::P(ca + (1 + i) * 768 * 4),
                    A::L((BLOCK * 768) as i64),
                    A::P(cb),
                    A::P(self.br),
                    A::P(so),
                    A::P(self.mlat),
                ],
            )?;
            if self.opt.mconf {
                // INK38: per sequence, INK33's fold + argmax with the exp sum (the batched
                // kernels' per-row math verbatim + one exp sum; part / lat at the sequence's own
                // stride, one psum reused in stream order)
                for q in 0..nb {
                    c.launch(
                        "ink_dr6_fold_p",
                        [nwg, 1, 1],
                        256,
                        &[
                            A::P(s.fw),
                            A::P(s.fb),
                            A::P(base + (q * BLOCK + 1 + i) * vld * 4),
                            A::P(self.mlat + q * 256 * 4),
                            A::I(vn as i32),
                            A::F(30.0),
                            A::P(self.mpart + q * 2 * nwg * 4),
                            A::P(self.psum),
                        ],
                    )?;
                    c.launch(
                        "ink_dr6_argmax_map_p",
                        [1, 1, 1],
                        256,
                        &[
                            A::P(self.mpart + q * 2 * nwg * 4),
                            A::I(nwg as i32),
                            A::P(s.map),
                            A::P(ptoks + q * 64 + (i + 1) * 4),
                            A::P(self.psum),
                            A::F(30.0),
                            A::P(ptoks + q * 64 + (8 + i + 1) * 4),
                        ],
                    )?;
                }
                continue;
            }
            c.launch(
                "ink_dr3_fold",
                [nwg, nb, 1],
                256,
                &[
                    A::P(s.fw),
                    A::P(s.fb),
                    A::P(base + (1 + i) * vld * 4),
                    A::L((BLOCK * vld) as i64),
                    A::P(self.mlat),
                    A::I(vn as i32),
                    A::F(30.0),
                    A::P(self.mpart),
                    A::L((2 * nwg) as i64),
                ],
            )?;
            c.launch(
                "ink_dr3_argmax_map",
                [nb, 1, 1],
                256,
                &[
                    A::P(self.mpart),
                    A::L((2 * nwg) as i64),
                    A::I(nwg as i32),
                    A::P(s.map),
                    A::P(ptoks + (i + 1) * 4),
                    A::I(16),
                ],
            )?;
        }
        Ok(())
    }

    /// The round's launches (no host write, no sync, no read: graph-safe). With `keep`, the k
    /// full logits rows land at the returned address (the oracle gate).
    #[allow(clippy::too_many_lines)]
    pub(super) fn body(&self, c: &mut Card<'_, '_>, k: u64, keep: bool) -> anyhow::Result<u64> {
        ensure!(
            !keep || self.sub.is_none(),
            "the oracle rows need the full head (--ink-draft-vocab 0)"
        );
        c.reset(Reg::Pass)?;
        let hres = c.z4(Reg::Pass, BLOCK * D, "dr.h")?;
        c.launch(
            "ink_dr_copy",
            [64, 1, 1],
            256,
            &[A::P(self.x0), A::P(hres), A::L((BLOCK * D) as i64)],
        )?;
        let b = BLOCK;
        for (li, lw) in self.l.iter().enumerate() {
            c.reset(Reg::Visit)?;
            let a = c.rmsnorm(Reg::Visit, hres, b, D, lw.ln1)?;
            let q = c.z4(Reg::Visit, b * H * HD, "dr.q")?;
            let kk = c.z4(Reg::Visit, b * KV * HD, "dr.k")?;
            let vv = c.z4(Reg::Visit, b * KV * HD, "dr.v")?;
            self.mm(c, a, b, D, &lw.q, q, H * HD, false)?;
            self.mm(c, a, b, D, &lw.k, kk, KV * HD, false)?;
            self.mm(c, a, b, D, &lw.v, vv, KV * HD, false)?;
            c.launch(
                "ink_dr_hnr",
                [b, H, 1],
                64,
                &[
                    A::P(q),
                    A::I((H * HD) as i32),
                    A::P(lw.qn),
                    A::P(self.ppos),
                    A::I(0),
                    A::F(EPS),
                ],
            )?;
            c.launch(
                "ink_dr_hnr",
                [b, KV, 1],
                64,
                &[
                    A::P(kk),
                    A::I((KV * HD) as i32),
                    A::P(lw.kn),
                    A::P(self.ppos),
                    A::I(0),
                    A::F(EPS),
                ],
            )?;
            let o = c.z4(Reg::Visit, b * H * HD, "dr.o")?;
            let window = if li == 1 { 0 } else { 1024 };
            c.launch(
                "ink_dr_attn",
                [b, H, 1],
                256,
                &[
                    A::P(q),
                    A::I((H * HD) as i32),
                    A::P(self.kc_of(li)),
                    A::P(self.vc_of(li)),
                    A::P(kk),
                    A::P(vv),
                    A::I((KV * HD) as i32),
                    A::P(self.ppos),
                    A::I(window),
                    A::P(o),
                    A::I((H * HD) as i32),
                    A::F((HD as f32).powf(-0.5)),
                ],
            )?;
            self.mm(c, o, b, H * HD, &lw.o, hres, D, true)?;
            let bn = c.rmsnorm(Reg::Visit, hres, b, D, lw.ln2)?;
            let g = c.z4(Reg::Visit, b * FF, "dr.g")?;
            let u = c.z4(Reg::Visit, b * FF, "dr.u")?;
            self.mm(c, bn, b, D, &lw.gate, g, FF, false)?;
            self.mm(c, bn, b, D, &lw.up, u, FF, false)?;
            c.launch(
                "ink_dr_silu_mul",
                [256, 1, 1],
                256,
                &[A::P(g), A::P(u), A::L((b * FF) as i64)],
            )?;
            self.mm(c, g, b, FF, &lw.down, hres, D, true)?;
        }
        c.reset(Reg::Visit)?;
        let out = c.rmsnorm(Reg::Pass, hres + D * 4, k, D, self.norm)?;
        // INK18: the pruned vocabulary's rows (ld = its n), else both full halves (ld VPAD)
        let (vn, vld, fw, fb) = match &self.sub {
            Some(s) => (s.n, s.n, s.fw, s.fb),
            None => (VOCAB as u64, VPAD, self.fw, self.fb),
        };
        let base = c.z4(Reg::Pass, k * vld, "dr.base")?;
        if let Some(s) = &self.sub {
            self.mm(c, out, k, D, &s.head, base, vld, false)?;
        } else {
            let mut col = 0;
            for hw in &self.head {
                c.gemm(out, k, D, hw, base + col * 4, VPAD, false)?;
                col += hw.rows();
            }
        }
        let ca = c.z4(Reg::Pass, k * 768, "dr.ca")?;
        let cb = c.z4(Reg::Pass, 768, "dr.cb")?;
        self.mm(c, out, k, D, &self.wa, ca, 768, false)?;
        self.mm(c, self.f_rnn_of(), 1, D, &self.wb, cb, 768, false)?;
        c.launch(
            "ink_dr_copy",
            [1, 1, 1],
            256,
            &[A::P(self.zero), A::P(self.s[0]), A::L(256)],
        )?;
        let nwg = vn.div_ceil(64);
        let keep_lg = if keep {
            c.z4(Reg::Pass, k * VOCAB as u64, "dr.keep")?
        } else {
            0
        };
        for i in 0..k {
            let (si, so) = (self.s[(i % 2) as usize], self.s[((i + 1) % 2) as usize]);
            c.launch(
                "ink_dr_gru",
                [256, 1, 1],
                256,
                &[
                    A::P(si),
                    A::P(self.w1),
                    A::P(self.ptoks + i * 4),
                    A::P(self.wlo),
                    A::I(512),
                    A::P(ca + i * 768 * 4),
                    A::P(cb),
                    A::P(self.br),
                    A::P(so),
                    A::P(self.lat),
                ],
            )?;
            let conf = self.opt.conf && self.sub.is_some();
            if conf {
                // INK33: the same fold + the workgroups' exp sums
                c.launch(
                    "ink_dr6_fold_p",
                    [nwg, 1, 1],
                    256,
                    &[
                        A::P(fw),
                        A::P(fb),
                        A::P(base + i * vld * 4),
                        A::P(self.lat),
                        A::I(vn as i32),
                        A::F(30.0),
                        A::P(self.part),
                        A::P(self.psum),
                    ],
                )?;
            } else {
                c.launch(
                    "ink_dr_fold",
                    [nwg, 1, 1],
                    256,
                    &[
                        A::P(fw),
                        A::P(fb),
                        A::P(base + i * vld * 4),
                        A::P(self.lat),
                        A::I(vn as i32),
                        A::F(30.0),
                        A::P(self.part),
                    ],
                )?;
            }
            if let (true, Some(s)) = (conf, &self.sub) {
                c.launch(
                    "ink_dr6_argmax_map_p",
                    [1, 1, 1],
                    256,
                    &[
                        A::P(self.part),
                        A::I(nwg as i32),
                        A::P(s.map),
                        A::P(self.ptoks + (i + 1) * 4),
                        A::P(self.psum),
                        A::F(30.0),
                        A::P(self.ptoks + (8 + i + 1) * 4),
                    ],
                )?;
            } else if let Some(s) = &self.sub {
                c.launch(
                    "ink_dr_argmax_map",
                    [1, 1, 1],
                    256,
                    &[
                        A::P(self.part),
                        A::I(nwg as i32),
                        A::P(s.map),
                        A::P(self.ptoks + (i + 1) * 4),
                    ],
                )?;
            } else {
                c.launch(
                    "ink_dr_argmax",
                    [1, 1, 1],
                    256,
                    &[
                        A::P(self.part),
                        A::I(nwg as i32),
                        A::P(self.ptoks + (i + 1) * 4),
                    ],
                )?;
            }
            if keep {
                // the gate's rows: softcap(base + fold_b + fold_w . latent) for the whole vocab
                c.launch(
                    "ink_dr_fold_rows",
                    [nwg, 1, 1],
                    256,
                    &[
                        A::P(self.fw),
                        A::P(self.fb),
                        A::P(base + i * VPAD * 4),
                        A::P(self.lat),
                        A::I(VOCAB as i32),
                        A::F(30.0),
                        A::P(keep_lg + i * VOCAB as u64 * 4),
                    ],
                )?;
            }
        }
        Ok(keep_lg)
    }
}

/// INK12 task 3: the verify's conv history log (`Card::hist`): every conv of a verify chunk
/// stashes `[state ; inputs]` before it writes its state; `commit(keep)` sets every state to the
/// one after the first `keep` rows (the rejected rows never advance a state).
#[derive(Default)]
pub struct Hist {
    pub buf: u64,
    pub cap: u64,
    pub off: u64,
    pub entries: Vec<u64>,
    pub tab: u64,
    pub max_w: u64,
}

/// INK13: a captured verify chunk (`capture_verify`): the graph, its logits (address, row
/// stride f32), its rollback table (`n` convs of width <= `max_w`), its rows `t`.
pub struct VGraph {
    pub _g: saber_hip::Graph,
    pub e: saber_hip::GraphExec,
    pub lg: u64,
    pub ld: u64,
    pub tab: u64,
    pub n: u64,
    pub max_w: u64,
    pub t: u64,
    pub launches: u64,
    pub nodes: usize,
    /// the log's entries (hist, dst, width) -- the zero control rebuilds a table from them
    pub ent: Vec<u64>,
}

impl Card<'_, '_> {
    /// INK13: roll the convs of a captured verify back to `keep` committed rows (one launch; the
    /// table is the graph's own).
    pub(super) fn commit_tab(&mut self, vg: &VGraph, keep: u64) -> anyhow::Result<()> {
        ensure!(
            (1..=vg.t).contains(&keep),
            "commit {keep} rows of a {}-row verify",
            vg.t
        );
        self.launch(
            "ink_dr_commit",
            [vg.max_w.div_ceil(256), vg.n, 1],
            256,
            &[A::P(vg.tab), A::I(keep as i32)],
        )
    }

    /// INK13 control `zero`: every logged conv state set to ZEROS (a deliberately corrupted
    /// rollback; must change the output).
    pub(super) fn commit_zero(&mut self, vg: &VGraph, zeros: u64) -> anyhow::Result<()> {
        let tab: Vec<u8> = vg
            .ent
            .chunks_exact(3)
            .flat_map(|e| [zeros, e[1], e[2]])
            .flat_map(u64::to_le_bytes)
            .collect();
        let h = self.hist_spare.as_ref().context("hist")?.tab;
        self.write(h, &tab)?;
        self.launch(
            "ink_dr_commit",
            [vg.max_w.div_ceil(256), vg.n, 1],
            256,
            &[A::P(h), A::I(0)],
        )
    }

    /// Log one conv: state `st` ([3, w] f32, oldest first) about to take the `t` rows of `x`
    /// (row stride `ldx`) and be written at `dst`.
    pub(super) fn stash(
        &mut self,
        st: u64,
        x: u64,
        ldx: u64,
        t: u64,
        w: u64,
        dst: u64,
    ) -> anyhow::Result<()> {
        let Some(h) = self.hist.as_mut() else {
            return Ok(());
        };
        let bytes = (3 + t) * w * 4;
        ensure!(h.off + bytes <= h.cap, "the verify history is full");
        let at = h.buf + h.off;
        h.off += bytes;
        h.entries.extend_from_slice(&[at, dst, w]);
        h.max_w = h.max_w.max(w);
        self.launch(
            "ink_dr_hist",
            [w.div_ceil(256), 3 + t, 1],
            256,
            &[
                A::P(st),
                A::P(x),
                A::I(ldx as i32),
                A::I(w as i32),
                A::P(at),
            ],
        )
    }

    /// INK17: reserve one conv's history entry `[state (3 rows) ; x (t rows)]` (width `w`, state
    /// at `dst`) that `ink_xsdc` writes itself; `None` while no verify history is open.
    pub(super) fn hist_entry(&mut self, t: u64, w: u64, dst: u64) -> anyhow::Result<Option<u64>> {
        let Some(h) = self.hist.as_mut() else {
            return Ok(None);
        };
        let bytes = (3 + t) * w * 4;
        ensure!(h.off + bytes <= h.cap, "the verify history is full");
        let at = h.buf + h.off;
        h.off += bytes;
        h.entries.extend_from_slice(&[at, dst, w]);
        h.max_w = h.max_w.max(w);
        Ok(Some(at))
    }

    /// Roll every logged conv state back to `keep` committed rows, then clear the log.
    pub(super) fn commit(&mut self, keep: u64) -> anyhow::Result<usize> {
        let mut h = self.hist.take().context("no verify history")?;
        let n = h.entries.len() / 3;
        let tab: Vec<u8> = h.entries.iter().flat_map(|v| v.to_le_bytes()).collect();
        ensure!(
            tab.len() as u64 <= 3 * 8 * 512,
            "the history table holds {n} convs"
        );
        self.write(h.tab, &tab)?;
        if n > 0 {
            self.launch(
                "ink_dr_commit",
                [h.max_w.div_ceil(256), n as u64, 1],
                256,
                &[A::P(h.tab), A::I(keep as i32)],
            )?;
        }
        h.off = 0;
        h.entries.clear();
        h.max_w = 0;
        self.hist_spare = Some(h);
        Ok(n)
    }
}

#[cfg(test)]
mod ink18_tests {
    use super::{quant_w8, sk_split};

    /// INK18: the drafter's 8-bit record law (INK14's): the scale is the smallest bf16 >= amax /
    /// 127, every code is the round-half-even of w / scale, |code| <= 127; the negative control (a
    /// code + 1) breaks the half-step bound.
    #[test]
    fn ink18_drafter_w8_record_law() {
        let w: Vec<f32> = (0..256)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013)
            .collect();
        let b: Vec<u8> = w
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let wb: Vec<f32> = b
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect();
        let (codes, sc) = quant_w8(&b);
        assert_eq!((codes.len(), sc.len()), (256, 16));
        for blk in 0..8 {
            let s =
                f32::from_bits(u32::from(u16::from_le_bytes([sc[2 * blk], sc[2 * blk + 1]])) << 16);
            let amax = wb[blk * 32..(blk + 1) * 32]
                .iter()
                .fold(0f32, |m, v| m.max(v.abs()));
            assert!(s >= amax / 127.0);
            for j in 0..32 {
                let q = f32::from(codes[blk * 32 + j] as i8);
                assert!((q * s - wb[blk * 32 + j]).abs() <= s / 2.0);
                assert!((q + 1.0) * s - wb[blk * 32 + j] > s / 2.0);
            }
        }
    }

    /// INK18: the split-K cell table (slices a multiple of 256 k and <= 768; >= 256 workgroups
    /// where the shape allows; passes while >= 192 workgroups remain).
    #[test]
    fn ink18_drafter_split_cells() {
        assert_eq!(sk_split(9216, 2304, 4), Some((3, 4)));
        assert_eq!(sk_split(2304, 9216, 4), Some((12, 4)));
        assert_eq!(sk_split(1024, 2304, 4), Some((9, 1)));
        assert_eq!(sk_split(32_768, 2304, 4), Some((3, 4)));
        assert_eq!(sk_split(32_768, 2304, 1), Some((3, 1)));
        assert_eq!(sk_split(64, 100, 4), None);
    }
}
