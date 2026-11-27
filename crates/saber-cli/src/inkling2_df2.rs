//! INK48: the public Inkling-2-Mini DFlash2 block drafter (`incoai/Inkling-2-Mini-DFlash2`) as an
//! ARM of the Inkling drafter (`--ink-drafter dflash2 --ink-dflash2-path DIR`; `dspark` = the
//! default, the rollback). docs/INKLING2.md INK47 (the census + the host oracle
//! `tools/capture/ink47/dflash2_ref.py`) / INK48. Our own 27B DFlash2 op list (`dflash_graph.rs`) at
//! this checkpoint's shapes: the context store fc -> hidden_norm -> each layer's k / v -> k_norm +
//! rope (theta 1e4) into a 2,048-row ring a layer a context slot; the block = [embed_norm'd anchor ;
//! the normed MASK row x 7] through 5 layers (grouped two-tap convs around the attention and the
//! MLP, symmetric window 2,048, no causal mask); the head = the target's (pruned `--ink-draft-vocab`
//! rows or both full halves), its top 16, the 27B selector's edge scores
//! (`dflash_selector_score_r256_k16`) and the greedy walk over cap(unary + edge) (`ink_df2_walk`:
//! ids + each step's 16-way top-1 probability in `ptoks`' DSpark layout, so the INK33 gate reads
//! it unchanged). The drafter changes PROPOSALS only.

use super::super::{Card, PackReader, Reg, Tw, A, D, VOCAB};
use super::{bf16s, DraftOpt, Drafter, AR, BLOCK, EPS, HD, KV, MB, VPAD};
use anyhow::{ensure, Context};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

const H2: u64 = 32;
const FF2: u64 = 6912;
const NL2: usize = 5;
/// the context ring's rows a layer a slot (= the sliding window)
pub const RING: u64 = 2048;
const CP: u64 = 576;
const RANK: u64 = 256;
const TOPK: u64 = 16;
const MASK_ID: u32 = 200_056;
const THETA: f32 = 1e4;
const CAP: f32 = 30.0;
/// the append's fixed scratch rows (a prefill chunk runs through it 64 rows at a time)
const APR: u64 = 64;
/// the most draft rows of a pass (B 8 x k 7)
const PR: u64 = 64;

struct L2 {
    ln1: u64,
    ln2: u64,
    qn: u64,
    kn: u64,
    q: Tw,
    k: Tw,
    v: Tw,
    o: Tw,
    gate: Tw,
    up: Tw,
    down: Tw,
    kpa: Tw,
    kpm: Tw,
    ba: u64,
    bm: u64,
}

/// The DFlash2 weights + fixed buffers (rank 0).
pub struct Df2 {
    l: Vec<L2>,
    fc: Tw,
    hn: u64,
    norm: u64,
    hp: Tw,
    pcb: u64,
    scb: u64,
    /// the pruned head (rows, id map, n) or the target's two full halves
    sub: Option<(Tw, u64, u64)>,
    full: Vec<Tw>,
    acat: u64,
    ax: u64,
    axn: u64,
    ak: u64,
    av: u64,
    cand: u64,
    unary: u64,
    hpf: u64,
    hpb: u64,
    scores: u64,
    anch: u64,
    /// INK50 (`--ink-draft-layers iu8`): the activation codes [PR, 4 D] + scales [PR, 4 D / 32]
    /// of the W8A8 projections (0 = the w8 / bf16 forms)
    iqa: u64,
    isa: u64,
}

/// INK50: the iu8 form's tag in the pack manifest (`build_ink2_dflash2.py`'s `iu8` section).
pub const IU8_LAW: &str = "ink50-iu8/1: ink_act_q8_b32_v2 + ink_df2_i8_w8 on the w8 records";

/// INK50: `ink_df2_i8_w8`'s waves a workgroup for a K -- the largest divisor of K / 128 at or
/// below 16 (each wave sums one contiguous run of 128-k groups; the count is numerics).
pub fn iu8_waves(k: u64) -> u64 {
    let g = (k / 128).max(1);
    (1..=16).rev().find(|d| g % d == 0).unwrap_or(1)
}

/// INK48: the drafter pack's format tag (`tools/pack/build_ink2_dflash2.py`).
pub const PACK_FORMAT: &str = "ink2-dflash2/1";

impl Drafter {
    /// INK48: load the DFlash2 drafter (`opt.df2`): every tensor bf16 from the checkpoint (the
    /// block layers' and fc's projections in 8 bits under `opt.w8`, INK14's record law), the
    /// target's embed_norm / head, the pruned vocabulary from `<pack>/draft/vocab_rank.u32`.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub(super) fn load_df2(
        c: &mut Card<'_, '_>,
        pk: &PackReader,
        dir: &Path,
        ck: &Path,
        head0: &Tw,
        cap: u64,
        tcap: u64,
        opt: &DraftOpt,
    ) -> anyhow::Result<Self> {
        // INK48: PACK-DRIVEN -- the Saber drafter pack (build_ink2_dflash2.py, checked by
        // check_ink2_dflash2.py), never the raw checkpoint: the rank-0 shard serves TP=2 and TP=1
        let man: Value = serde_json::from_str(
            &std::fs::read_to_string(ck.join("manifest.json")).with_context(|| {
                format!(
                    "{}: no DFlash2 drafter pack (tools/pack/build_ink2_dflash2.py)",
                    ck.display()
                )
            })?,
        )?;
        ensure!(
            man["format"].as_str() == Some(PACK_FORMAT)
                && man["ranks"] == serde_json::json!({"tp2": [0], "tp1": [0]}),
            "{}: not a {PACK_FORMAT} pack (format {:?})",
            ck.display(),
            man["format"]
        );
        let cfg = &man["config"];
        let dc = &cfg["dflash_config"];
        ensure!(
            cfg["hidden_size"].as_u64() == Some(D)
                && cfg["num_hidden_layers"].as_u64() == Some(NL2 as u64)
                && cfg["num_attention_heads"].as_u64() == Some(H2)
                && cfg["num_key_value_heads"].as_u64() == Some(KV)
                && cfg["intermediate_size"].as_u64() == Some(FF2)
                && cfg["sliding_window"].as_u64() == Some(RING)
                && dc["block_size"].as_u64() == Some(BLOCK)
                && dc["selector_rank"].as_u64() == Some(RANK)
                && dc["selector_top_k"].as_u64() == Some(TOPK)
                && dc["mask_token_id"].as_u64() == Some(u64::from(MASK_ID))
                && dc["selector_softcap_mode"].as_str() == Some("after_correction")
                && dc["target_layer_ids"] == serde_json::json!([5, 23, 41, 48]),
            "{}: not the INK47 DFlash2 geometry",
            ck.display()
        );
        let recs = &man["records"];
        let read = |n: &str, dt: &str, es: u64| -> anyhow::Result<(Vec<u8>, Vec<u64>)> {
            let r = &recs[n];
            let f = r["file"]
                .as_str()
                .with_context(|| format!("dflash2 pack: no record {n}"))?;
            ensure!(
                r["dtype"].as_str() == Some(dt),
                "dflash2 {n}: dtype {:?}",
                r["dtype"]
            );
            let shape: Vec<u64> = r["shape"]
                .as_array()
                .context("shape")?
                .iter()
                .filter_map(Value::as_u64)
                .collect();
            let b = std::fs::read(ck.join(f))?;
            ensure!(
                b.len() as u64 == shape.iter().product::<u64>() * es,
                "dflash2 {n}: {} bytes vs {shape:?}",
                b.len()
            );
            Ok((b, shape))
        };
        let tw = |c: &mut Card<'_, '_>, n: &str, q8: bool| -> anyhow::Result<Tw> {
            if !q8 {
                let (b, shape) = read(n, "BF16", 2)?;
                return Ok(Tw {
                    addr: c.up(&b)?,
                    es: 2,
                    shape,
                    sc: 0,
                });
            }
            // the pack's 8-bit record (INK14's law, written by the builder): codes + bf16 scales
            let (codes, shape) = read(&format!("{n}.w8"), "I8", 1)?;
            let (sc, ss) = read(&format!("{n}.w8s"), "BF16", 2)?;
            ensure!(
                ss == [shape[0], shape[1] / 32],
                "dflash2 {n}.w8s: shape {ss:?}"
            );
            let t = Tw {
                addr: c.up(&codes)?,
                es: 1,
                shape,
                sc: c.up(&sc)?,
            };
            ensure!(
                t.addr % 16 == 0 && t.sc % 16 == 0,
                "dflash2 w8 {n}: unaligned upload"
            );
            Ok(t)
        };
        let w8 = opt.w8;
        let mut l = Vec::new();
        for i in 0..NL2 {
            let p = |s: &str| format!("layers.{i}.{s}");
            l.push(L2 {
                ln1: tw(c, &p("input_layernorm.weight"), false)?.addr,
                ln2: tw(c, &p("post_attention_layernorm.weight"), false)?.addr,
                qn: tw(c, &p("self_attn.q_norm.weight"), false)?.addr,
                kn: tw(c, &p("self_attn.k_norm.weight"), false)?.addr,
                q: tw(c, &p("self_attn.q_proj.weight"), w8)?,
                k: tw(c, &p("self_attn.k_proj.weight"), w8)?,
                v: tw(c, &p("self_attn.v_proj.weight"), w8)?,
                o: tw(c, &p("self_attn.o_proj.weight"), w8)?,
                gate: tw(c, &p("mlp.gate_proj.weight"), w8)?,
                up: tw(c, &p("mlp.up_proj.weight"), w8)?,
                down: tw(c, &p("mlp.down_proj.weight"), w8)?,
                kpa: tw(c, &p("attention_conv.kernel_projection.weight"), false)?,
                kpm: tw(c, &p("mlp_conv.kernel_projection.weight"), false)?,
                ba: tw(c, &p("attention_conv.base_kernel"), false)?.addr,
                bm: tw(c, &p("mlp_conv.base_kernel"), false)?.addr,
            });
            ensure!(
                l[i].kpa.shape == [CP, D] && l[i].q.shape == [H2 * HD, D],
                "dflash2 layer {i}: shapes"
            );
        }
        let fc = tw(c, "fc.weight", w8)?;
        ensure!(fc.shape == [D, 4 * D], "dflash2 fc {:?}", fc.shape);
        if opt.iu8 {
            // INK50: the pack must declare the iu8 form over exactly these records + geometry
            let t = &man["iu8"];
            ensure!(
                t["law"].as_str() == Some(IU8_LAW),
                "{}: no iu8 section (rebuild with tools/pack/build_ink2_dflash2.py, INK50)",
                ck.display()
            );
            let mut ws: Vec<(String, &Tw)> = vec![("fc.weight".into(), &fc)];
            for (i, lw) in l.iter().enumerate() {
                for (s, w) in [
                    ("self_attn.q_proj", &lw.q),
                    ("self_attn.k_proj", &lw.k),
                    ("self_attn.v_proj", &lw.v),
                    ("self_attn.o_proj", &lw.o),
                    ("mlp.gate_proj", &lw.gate),
                    ("mlp.up_proj", &lw.up),
                    ("mlp.down_proj", &lw.down),
                ] {
                    ws.push((format!("layers.{i}.{s}.weight"), w));
                }
            }
            let tr = t["records"].as_object().context("iu8 records")?;
            ensure!(tr.len() == ws.len(), "iu8: {} records", tr.len());
            for (n, w) in ws {
                let e = &tr.get(&n).with_context(|| format!("iu8: no {n}"))?;
                let (nn, k) = (w.shape[0], w.shape[1]);
                ensure!(
                    w.sc != 0
                        && e["n"].as_u64() == Some(nn)
                        && e["k"].as_u64() == Some(k)
                        && e["waves"].as_u64() == Some(iu8_waves(k))
                        && k % 128 == 0
                        && k <= 4 * D,
                    "iu8 {n}: the pack's {e} vs the engine's n {nn} k {k} waves {}",
                    iu8_waves(k)
                );
            }
        }
        let hn = tw(c, "hidden_norm.weight", false)?.addr;
        let norm = tw(c, "norm.weight", false)?.addr;
        let hp = tw(c, "candidate_selector.hidden_projection.weight", false)?;
        let pcbt = tw(c, "candidate_selector.predecessor_codebook", false)?;
        ensure!(
            pcbt.shape == [VPAD, RANK],
            "dflash2 codebook {:?}",
            pcbt.shape
        );
        let pcb = pcbt.addr;
        let scb = tw(c, "candidate_selector.successor_codebook", false)?.addr;
        // the target head: the two vocab halves (INK41: one card = one whole record, two views)
        let whole = head0.rows() == VPAD;
        let half = VPAD / 2;
        let full = if whole {
            let cols = head0.shape[1];
            let sc = |r: u64| {
                if head0.sc == 0 {
                    0
                } else {
                    head0.sc + r * (cols / 32) * 2
                }
            };
            vec![
                Tw {
                    addr: head0.addr,
                    es: head0.es,
                    shape: vec![half, cols],
                    sc: head0.sc,
                },
                Tw {
                    addr: head0.addr + half * cols * head0.es,
                    es: head0.es,
                    shape: vec![VPAD - half, cols],
                    sc: sc(half),
                },
            ]
        } else if opt.vocab == 0 {
            vec![
                head0.clone(),
                super::super::put_dense(c, pk, "lm_head.weight", Some(1), true)?,
            ]
        } else {
            Vec::new()
        };
        let sub = if opt.vocab == 0 {
            None
        } else {
            let rf = dir.join("vocab_rank.u32");
            let rb =
                std::fs::read(&rf).with_context(|| format!("{}: no rank file", rf.display()))?;
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
            if whole {
                let (rb, sb) = ((D * half) as usize, (D / 32 * 2 * half) as usize);
                if let Some((codes, sc, shape)) = pk.tensor_w8("lm_head.weight", None, true)? {
                    ensure!(shape[0] == VPAD, "lm_head: {} rows", shape[0]);
                    halves.push((codes[..rb].to_vec(), Some(sc[..sb].to_vec()), half));
                    halves.push((codes[rb..].to_vec(), Some(sc[sb..].to_vec()), VPAD - half));
                } else {
                    let (b, es, shape) = pk.tensor("lm_head.weight", None)?;
                    ensure!(es == 2 && shape[0] == VPAD, "lm_head: es {es}");
                    halves.push((b[..2 * rb].to_vec(), None, half));
                    halves.push((b[2 * rb..].to_vec(), None, VPAD - half));
                }
            } else {
                for r in 0..2u32 {
                    if let Some((codes, sc, shape)) =
                        pk.tensor_w8("lm_head.weight", Some(r), true)?
                    {
                        halves.push((codes, Some(sc), shape[0]));
                    } else {
                        let (b, es, shape) = pk.tensor("lm_head.weight", Some(r))?;
                        ensure!(es == 2, "lm_head rank {r}: es {es}");
                        halves.push((b, None, shape[0]));
                    }
                }
            }
            let hw8 = halves[0].1.is_some();
            ensure!(
                hw8 == (head0.sc != 0),
                "the pruned head: kind differs from the target's"
            );
            let rbytes = (if hw8 { D } else { 2 * D }) as usize;
            let sbytes = (D / 32 * 2) as usize;
            let (mut hb, mut hs, mut mb) = (Vec::new(), Vec::new(), Vec::new());
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
                mb.extend_from_slice(&(v as u32).to_le_bytes());
            }
            let head = Tw {
                addr: c.up(&hb)?,
                es: if hw8 { 1 } else { 2 },
                shape: vec![opt.vocab, D],
                sc: if hw8 { c.up(&hs)? } else { 0 },
            };
            ensure!(
                head.addr % 16 == 0 && head.sc % 16 == 0,
                "the pruned head: unaligned"
            );
            Some((head, c.up(&mb)?, opt.vocab))
        };
        // the normed MASK row (f32 bytes) for `prep` (the block's rows 1..7)
        let (en, _, _) = pk.tensor("model.embed_norm.weight", None)?;
        let embed_norm = bf16s(&en);
        let raw: Vec<f32> = pk
            .embed_rows(&[MASK_ID])?
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let ss: f32 = raw.iter().map(|v| v * v).sum::<f32>() / D as f32;
        let r = 1.0 / (ss + EPS).sqrt();
        let mask: Vec<u8> = raw
            .iter()
            .zip(&embed_norm)
            .flat_map(|(v, g)| ((v * r) * g).to_le_bytes())
            .collect();
        let nslots = opt.slots.max(1);
        ensure!(nslots <= 8, "drafter: {nslots} context slots (1..8)");
        let cstride = RING * KV * HD * 4;
        let (mut kc, mut vc) = (Vec::new(), Vec::new());
        for _ in 0..NL2 {
            kc.push(c.alloc_zero(nslots * cstride)?);
            vc.push(c.alloc_zero(nslots * cstride)?);
        }
        let skp_n = 3 * 8 * VPAD;
        let (skp, skc) = if opt.sk || opt.wmma {
            (
                c.alloc_zero(skp_n * 4)?,
                c.alloc_zero(VPAD.div_ceil(32) * 4)?,
            )
        } else {
            (0, 0)
        };
        let multi = nslots > 1;
        let minb = if multi {
            c.alloc_zero(MB * BLOCK * D * 4 + MB * 8 + MB * 64 + MB * 4)?
        } else {
            0
        };
        let (apos, atab) = if multi {
            (c.alloc_zero(nslots * 4)?, c.alloc_zero(AR * 4 * 3)?)
        } else {
            (0, 0)
        };
        let df = Df2 {
            l,
            fc,
            hn,
            norm,
            hp,
            pcb,
            scb,
            sub,
            full,
            acat: c.alloc_zero(APR * 4 * D * 4)?,
            ax: c.alloc_zero(APR * D * 4)?,
            axn: c.alloc_zero(APR * D * 4)?,
            ak: c.alloc_zero(APR * KV * HD * 4)?,
            av: c.alloc_zero(APR * KV * HD * 4)?,
            cand: c.alloc_zero(PR * TOPK * 8)?,
            unary: c.alloc_zero(PR * TOPK * 4)?,
            hpf: c.alloc_zero(PR * RANK * 4)?,
            hpb: c.alloc_zero(PR * RANK * 2)?,
            scores: c.alloc_zero(PR * TOPK * TOPK * 4)?,
            anch: c.alloc_zero(MB * 4)?,
            iqa: if opt.iu8 {
                c.alloc_zero(PR * 4 * D)?
            } else {
                0
            },
            isa: if opt.iu8 {
                c.alloc_zero(PR * 4 * D / 32 * 4)?
            } else {
                0
            },
        };
        let dummy = || Tw {
            addr: 0,
            es: 2,
            shape: vec![0, 0],
            sc: 0,
        };
        let taps = c.alloc_zero(4 * tcap * D * 4)?;
        let inb = c.alloc_zero(BLOCK * D * 4 + 16 + 64)?;
        println!(
            "INK48 drafter dflash2: {} ({} layers {}, head {}, ring {RING} x {nslots} slots)",
            ck.display(),
            NL2,
            if opt.iu8 {
                "iu8 W8A8"
            } else if w8 {
                "8-bit"
            } else {
                "bf16"
            },
            match &df.sub {
                Some((_, _, n)) => format!("pruned {n}"),
                None => "full".into(),
            }
        );
        Ok(Self {
            l: Vec::new(),
            norm: 0,
            mdn: 0,
            alpha: [[0.0; 4]; 4],
            mask,
            embed_norm,
            head: Vec::new(),
            w1: 0,
            wlo: 0,
            wa: dummy(),
            wb: dummy(),
            br: 0,
            fw: 0,
            fb: 0,
            kc,
            vc,
            cstride,
            nslots,
            cur: 0,
            ctx_n: vec![0; nslots as usize],
            minb,
            apos,
            ms: [0, 0],
            mlat: 0,
            mpart: 0,
            mfrow: 0,
            mgraphs: BTreeMap::new(),
            agraphs: BTreeMap::new(),
            atab,
            amx: 0,
            ak: 0,
            av: 0,
            af: 0,
            cap,
            taps,
            tcap,
            f_rnn: 0,
            ppos: inb + BLOCK * D * 4,
            ptoks: inb + BLOCK * D * 4 + 16,
            x0: inb,
            inb,
            graphs: BTreeMap::new(),
            s: [0, 0],
            lat: 0,
            zero: c.alloc_zero(256 * 4)?,
            part: 0,
            psum: 0,
            nctx: 0,
            opt: opt.clone(),
            sub: None,
            skp,
            skp_n,
            skc,
            // INK49: the DFlash2 drafter has no DSpark global layer (its attention is windowed)
            ksp: 0,
            ks_smax: 1,
            df2: Some(Box::new(df)),
        })
    }

    /// A drafter GEMM at any M <= 64: the M <= 8 GEMV (`mm`), the matrix-core body (`--ink-draft-mm
    /// wmma`) or the target's GEMM.
    #[allow(clippy::too_many_arguments)]
    fn gm2(
        &self,
        c: &mut Card<'_, '_>,
        a: u64,
        m: u64,
        k: u64,
        w: &Tw,
        out: u64,
        ldc: u64,
    ) -> anyhow::Result<()> {
        if m <= 8 {
            self.mm(c, a, m, k, w, out, ldc, false)
        } else if self.opt.wmma {
            self.mmw(c, a, m, k, w, out, ldc, false)
        } else {
            c.gemm(a, m, k, w, out, ldc, false)
        }
    }

    /// INK50: the layer / fc projections of one input `a` [m, k] -- under `--ink-draft-layers iu8`
    /// the W8A8 form (the rows' int8 codes a 32-k block ONCE, then `ink_df2_i8_w8` on each 8-bit
    /// record) at every M, else `gm2` each.
    fn gl(
        &self,
        c: &mut Card<'_, '_>,
        a: u64,
        m: u64,
        k: u64,
        ws: &[(&Tw, u64, u64)],
    ) -> anyhow::Result<()> {
        let d = self.df2.as_ref().context("df2")?;
        if !self.opt.iu8 {
            for &(w, out, ldc) in ws {
                self.gm2(c, a, m, k, w, out, ldc)?;
            }
            return Ok(());
        }
        ensure!(
            d.iqa != 0 && m <= PR && k % 128 == 0 && k <= 4 * D,
            "iu8: {m} rows (k {k})"
        );
        let i = |v: u64| -> anyhow::Result<A> { Ok(A::I(i32::try_from(v)?)) };
        c.launch(
            "ink_act_q8_b32_v2",
            [(m * (k / 32) * 4).div_ceil(256), 1, 1],
            256,
            &[A::P(a), i(k)?, A::P(d.iqa), A::P(d.isa), i(m)?, i(k)?],
        )?;
        for &(w, out, ldc) in ws {
            let n = w.rows();
            ensure!(
                w.sc != 0 && w.shape[1] == k && w.addr % 16 == 0,
                "iu8: {m} x {n} (k {k}) not an 8-bit record"
            );
            c.launch(
                "ink_df2_i8_w8",
                [n.div_ceil(16), m.div_ceil(16), 1],
                u32::try_from(32 * iu8_waves(k))?,
                &[
                    A::P(d.iqa),
                    A::P(d.isa),
                    A::P(w.addr),
                    i(k)?,
                    A::P(w.sc),
                    i(k / 32)?,
                    A::P(out),
                    i(ldc)?,
                    i(m)?,
                    i(n)?,
                    i(k)?,
                    A::I(0),
                ],
            )?;
        }
        Ok(())
    }

    /// The context rows of `m` tap rows `row0..` (fixed scratch): fc -> hidden_norm -> per layer
    /// K = rope(k_norm(k x)) / V = v x into the ring at the rows' positions (`ppos` / `div` /
    /// `add`, as `ink_df2_hnr`), slots by `slots` (0 = the current slot's ring).
    #[allow(clippy::too_many_arguments)]
    fn df2_ctx(
        &self,
        c: &mut Card<'_, '_>,
        m: u64,
        row0: u64,
        ppos: u64,
        div: i32,
        add: u64,
        slots: u64,
    ) -> anyhow::Result<()> {
        let d = self.df2.as_ref().context("df2")?;
        let i = |v: u64| -> anyhow::Result<A> { Ok(A::I(i32::try_from(v)?)) };
        c.launch(
            "ink_df2_cat4",
            [m, 4 * D / 256, 1],
            256,
            &[
                A::P(self.taps),
                A::L(i64::try_from(self.tcap * D)?),
                i(row0)?,
                A::P(d.acat),
            ],
        )?;
        self.gl(c, d.acat, m, 4 * D, &[(&d.fc, d.ax, D)])?;
        c.launch(
            "ink_df2_rms",
            [m, 1, 1],
            256,
            &[A::P(d.ax), i(D)?, A::P(d.hn), A::P(d.axn), i(D)?, A::F(EPS)],
        )?;
        let row = KV * HD;
        for (li, lw) in d.l.iter().enumerate() {
            self.gl(c, d.axn, m, D, &[(&lw.k, d.ak, row), (&lw.v, d.av, row)])?;
            c.launch(
                "ink_df2_hnr",
                [m, KV, 1],
                64,
                &[
                    A::P(d.ak),
                    i(row)?,
                    A::P(lw.kn),
                    A::P(ppos),
                    A::I(div),
                    i(add)?,
                    A::F(THETA),
                    A::F(EPS),
                ],
            )?;
            let (kb, vb, ss) = if slots == 0 {
                (self.kc_of(li), self.vc_of(li), 0)
            } else {
                (self.kc[li], self.vc[li], self.cstride / 4)
            };
            for (src, dst) in [(d.ak, kb), (d.av, vb)] {
                c.launch(
                    "ink_df2_ring",
                    [m, row / 256, 1],
                    256,
                    &[
                        A::P(src),
                        i(row)?,
                        A::P(dst),
                        A::L(i64::try_from(ss)?),
                        i(row)?,
                        A::P(ppos),
                        A::I(div),
                        i(add)?,
                        A::P(slots),
                    ],
                )?;
            }
        }
        Ok(())
    }

    /// INK48: the single-slot append of the taps rows `row0..row0 + t` at positions `*ppos..`.
    pub(super) fn df2_append(
        &mut self,
        c: &mut Card<'_, '_>,
        t: u64,
        pos0: u64,
        row0: u64,
        ppos: u64,
    ) -> anyhow::Result<()> {
        let _ = pos0;
        let mut i = 0;
        while i < t {
            let m = (t - i).min(APR);
            self.df2_ctx(c, m, row0 + i, ppos, 1 << 30, i, 0)?;
            i += m;
        }
        Ok(())
    }

    /// INK48: the batched append (`append_batch`'s table: positions, slots; slot < 0 = skipped).
    pub(super) fn df2_append_batch(&self, c: &mut Card<'_, '_>, b: u64) -> anyhow::Result<()> {
        self.df2_ctx(c, b, 0, self.atab, 1, 0, self.atab + AR * 4)
    }

    /// INK48: the block pass -- `nb` 0 = the single-sequence round (`inb`: x0 ; ppos ; ptoks, the
    /// current slot), else `nb` sequences of the batched buffer (`minb`: x ; positions ; slots ;
    /// ptoks [MB][16] ; anchors); the k drafts + top-1 probabilities land in the ptoks words.
    #[allow(clippy::too_many_lines)]
    pub(super) fn df2_body(&self, c: &mut Card<'_, '_>, nb: u64, k: u64) -> anyhow::Result<()> {
        let d = self.df2.as_ref().context("df2")?;
        ensure!((1..BLOCK).contains(&k), "dflash2 depth {k}");
        let i = |v: u64| -> anyhow::Result<A> { Ok(A::I(i32::try_from(v)?)) };
        let (x0, ppos, slots, ptoks, nseq) = if nb == 0 {
            (self.x0, self.ppos, 0, self.ptoks, 1)
        } else {
            ensure!(self.minb != 0 && nb <= MB, "dflash2 batched pass of {nb}");
            let pp = self.minb + MB * BLOCK * D * 4;
            (self.minb, pp, pp + MB * 4, pp + MB * 8, nb)
        };
        let m = nseq * BLOCK;
        let rows = nseq * k;
        c.reset(Reg::Pass)?;
        let hres = c.z4(Reg::Pass, m * D, "df2.h")?;
        c.launch(
            "ink_dr_copy",
            [64, 1, 1],
            256,
            &[A::P(x0), A::P(hres), A::L(i64::try_from(m * D)?)],
        )?;
        let (kstride, scale) = (self.cstride / 4, (HD as f32).powf(-0.5));
        for (li, lw) in d.l.iter().enumerate() {
            c.reset(Reg::Visit)?;
            let n = c.z4(Reg::Visit, m * D, "df2.n")?;
            c.launch(
                "ink_df2_rms",
                [m, 1, 1],
                256,
                &[A::P(hres), i(D)?, A::P(lw.ln1), A::P(n), i(D)?, A::F(EPS)],
            )?;
            let ca = c.z4(Reg::Visit, m * CP, "df2.ca")?;
            self.gm2(c, n, m, D, &lw.kpa, ca, CP)?;
            let a = c.z4(Reg::Visit, m * D, "df2.a")?;
            let conv = |c: &mut Card<'_, '_>,
                        x: u64,
                        cf: u64,
                        base: u64,
                        side: i32,
                        out: u64,
                        acc: i32|
             -> anyhow::Result<()> {
                c.launch(
                    "ink_df2_conv",
                    [m, D / 256, 1],
                    256,
                    &[
                        A::P(x),
                        i(D)?,
                        A::P(cf),
                        A::P(base),
                        A::I(side),
                        A::P(out),
                        i(D)?,
                        A::I(acc),
                    ],
                )
            };
            conv(c, n, ca, lw.ba, 0, a, 0)?;
            let q = c.z4(Reg::Visit, m * H2 * HD, "df2.q")?;
            let kk = c.z4(Reg::Visit, m * KV * HD, "df2.k")?;
            let vv = c.z4(Reg::Visit, m * KV * HD, "df2.v")?;
            self.gl(
                c,
                a,
                m,
                D,
                &[
                    (&lw.q, q, H2 * HD),
                    (&lw.k, kk, KV * HD),
                    (&lw.v, vv, KV * HD),
                ],
            )?;
            for (buf, nh, g) in [(q, H2, lw.qn), (kk, KV, lw.kn)] {
                c.launch(
                    "ink_df2_hnr",
                    [m, nh, 1],
                    64,
                    &[
                        A::P(buf),
                        i(nh * HD)?,
                        A::P(g),
                        A::P(ppos),
                        A::I(BLOCK as i32),
                        A::I(0),
                        A::F(THETA),
                        A::F(EPS),
                    ],
                )?;
            }
            let o = c.z4(Reg::Visit, m * H2 * HD, "df2.o")?;
            let (kr, vr) = if nb == 0 {
                (self.kc_of(li), self.vc_of(li))
            } else {
                (self.kc[li], self.vc[li])
            };
            c.launch(
                "ink_df2_attn",
                [m, H2, 1],
                256,
                &[
                    A::P(q),
                    i(H2 * HD)?,
                    A::P(kr),
                    A::P(vr),
                    A::L(i64::try_from(kstride)?),
                    A::P(kk),
                    A::P(vv),
                    i(KV * HD)?,
                    A::P(ppos),
                    A::P(slots),
                    A::P(o),
                    i(H2 * HD)?,
                    A::F(scale),
                ],
            )?;
            let t = c.z4(Reg::Visit, m * D, "df2.t")?;
            self.gl(c, o, m, H2 * HD, &[(&lw.o, t, D)])?;
            conv(c, t, ca, lw.ba, 1, hres, 1)?;
            let n2 = c.z4(Reg::Visit, m * D, "df2.n2")?;
            c.launch(
                "ink_df2_rms",
                [m, 1, 1],
                256,
                &[A::P(hres), i(D)?, A::P(lw.ln2), A::P(n2), i(D)?, A::F(EPS)],
            )?;
            let cm = c.z4(Reg::Visit, m * CP, "df2.cm")?;
            self.gm2(c, n2, m, D, &lw.kpm, cm, CP)?;
            let mx = c.z4(Reg::Visit, m * D, "df2.m")?;
            conv(c, n2, cm, lw.bm, 0, mx, 0)?;
            let g = c.z4(Reg::Visit, m * FF2, "df2.g")?;
            let u = c.z4(Reg::Visit, m * FF2, "df2.u")?;
            self.gl(c, mx, m, D, &[(&lw.gate, g, FF2), (&lw.up, u, FF2)])?;
            c.launch(
                "ink_dr_silu_mul",
                [256, 1, 1],
                256,
                &[A::P(g), A::P(u), A::L(i64::try_from(m * FF2)?)],
            )?;
            let dn = c.z4(Reg::Visit, m * D, "df2.dn")?;
            self.gl(c, g, m, FF2, &[(&lw.down, dn, D)])?;
            conv(c, dn, cm, lw.bm, 1, hres, 1)?;
        }
        c.reset(Reg::Visit)?;
        // the draft rows 1..k of every block, final norm
        let src = if nb == 0 {
            hres + D * 4
        } else {
            let g = c.z4(Reg::Pass, rows * D, "df2.rows")?;
            c.launch(
                "ink_df2_rows",
                [rows, D / 256, 1],
                256,
                &[A::P(hres), A::P(g), i(k)?],
            )?;
            g
        };
        let out = c.z4(Reg::Pass, rows * D, "df2.out")?;
        c.launch(
            "ink_df2_rms",
            [rows, 1, 1],
            256,
            &[A::P(src), i(D)?, A::P(d.norm), A::P(out), i(D)?, A::F(EPS)],
        )?;
        let (vn, vld, map) = match &d.sub {
            Some((_, map, n)) => (*n, *n, *map),
            None => (VOCAB as u64, VPAD, 0),
        };
        let lg = c.z4(Reg::Pass, rows * vld, "df2.lg")?;
        if let Some((hw, _, _)) = &d.sub {
            self.gm2(c, out, rows, D, hw, lg, vld)?;
        } else {
            let mut col = 0;
            for hw in &d.full {
                c.gemm(out, rows, D, hw, lg + col * 4, VPAD, false)?;
                col += hw.rows();
            }
        }
        c.launch(
            "ink_df2_top16",
            [rows, 1, 1],
            256,
            &[
                A::P(lg),
                i(vld)?,
                i(vn)?,
                A::P(map),
                A::P(d.cand),
                A::P(d.unary),
            ],
        )?;
        self.gm2(c, out, rows, D, &d.hp, d.hpf, RANK)?;
        c.launch(
            "ink_df2_f2bf",
            [(rows * RANK).div_ceil(256), 1, 1],
            256,
            &[A::P(d.hpf), A::P(d.hpb), i(rows * RANK)?],
        )?;
        let anchor = if nb == 0 {
            ptoks
        } else {
            c.launch(
                "ink_df2_pick",
                [1, 1, 1],
                64,
                &[A::P(ptoks), A::I(16), A::P(d.anch), i(nseq)?],
            )?;
            d.anch
        };
        c.launch(
            "dflash_selector_score_r256_k16",
            [rows, 1, 1],
            32,
            &[
                A::P(d.pcb),
                A::P(d.scb),
                A::P(d.cand),
                A::P(anchor),
                A::P(d.unary),
                A::P(d.hpb),
                A::P(d.scores),
                i(rows)?,
                i(k)?,
                i(RANK)?,
                i(TOPK)?,
            ],
        )?;
        c.launch(
            "ink_df2_walk",
            [nseq, 1, 1],
            32,
            &[
                A::P(d.scores),
                A::P(d.cand),
                i(k)?,
                A::F(CAP),
                A::P(ptoks),
                A::I(16),
            ],
        )
    }
}

/// INK48 stage 1 gate (`--ink-run df2gate --tp 1`): the card's DFlash2 drafts vs the host f32
/// oracle (`dflash2_ref.py rec`, its `sel` arm) at EVERY anchor of INK12's taps: per prompt the taps
/// rows appended up to the anchor's position, one depth-7 round on the captured graph, the 7 ids
/// compared; the replay tokens a round (INK33's walk) of both. `--ink-ref` = the taps dir
/// (`pNN.taps.{json,f32}`), `--ink-mtp-ref` = the oracle's rec dir.
#[allow(clippy::too_many_lines)]
pub fn gate(
    c: &mut Card<'_, '_>,
    pk: &PackReader,
    head0: &Tw,
    draft_dir: &Path,
    f: &crate::inkling2::InkFlags,
    out: &Path,
) -> anyhow::Result<()> {
    let taps_dir = f.ink_ref.as_ref().context("--ink-ref <taps dir>")?;
    let rec_dir = f
        .ink_mtp_ref
        .as_ref()
        .context("--ink-mtp-ref <oracle rec dir>")?;
    let mut o = DraftOpt::of(f.draft_vocab(), &f.ink_draft_gemv, f.draft_layers())?;
    o = o.with_mm(&f.ink_draft_mm)?;
    o.df2 = f.dflash2()?;
    ensure!(o.df2.is_some(), "df2gate: --ink-drafter dflash2");
    o.conf = true;
    let tcap = 64;
    let mut dr = Drafter::load(c, pk, draft_dir, head0, 8192, tcap, &o)?;
    let only: Vec<String> = if f.ink_only.is_empty() {
        let mut v: Vec<String> = std::fs::read_dir(rec_dir)?
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter_map(|n| n.strip_suffix(".json").map(String::from))
            .collect();
        v.sort();
        v
    } else {
        f.ink_only.split(',').map(String::from).collect()
    };
    let (mut tok_eq, mut tok_n, mut blk_eq, mut blk_n) = (0u64, 0u64, 0u64, 0u64);
    let mut worst = Vec::new();
    let mut rep = serde_json::Map::new();
    let (mut acc_card, mut acc_ref) = (vec![(0u64, 0u64); 8], vec![(0u64, 0u64); 8]);
    let t0 = std::time::Instant::now();
    for pid in &only {
        let meta: Value = serde_json::from_str(&std::fs::read_to_string(
            taps_dir.join(format!("{pid}.taps.json")),
        )?)?;
        let rec: Value = serde_json::from_str(&std::fs::read_to_string(
            rec_dir.join(format!("{pid}.json")),
        )?)?;
        let ids = |v: &Value| -> Vec<u32> {
            v.as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_u64().map(|x| x as u32))
                        .collect()
                })
                .unwrap_or_default()
        };
        let seq = ids(&rec["seq"]);
        let rows = meta["rows"].as_u64().context("rows")?;
        let tb = std::fs::read(taps_dir.join(format!("{pid}.taps.f32")))?;
        ensure!(
            tb.len() as u64 == 4 * rows * D * 4 && rows as usize == seq.len(),
            "{pid}: taps {} bytes, {rows} rows, seq {}",
            tb.len(),
            seq.len()
        );
        let mut ps: Vec<u64> = rec["pos"]
            .as_object()
            .context("pos")?
            .keys()
            .filter_map(|k| k.parse().ok())
            .collect();
        ps.sort_unstable();
        dr.reset_ctx();
        let mut card: BTreeMap<u64, Vec<u32>> = BTreeMap::new();
        let (mut pe, mut pn) = (0u64, 0u64);
        for &p in &ps {
            while dr.nctx < p {
                let n0 = dr.nctx;
                let t = (p - n0).min(tcap);
                for q in 0..4u64 {
                    let a = ((q * rows + n0) * D * 4) as usize;
                    c.write(dr.taps + q * tcap * D * 4, &tb[a..a + (t * D * 4) as usize])?;
                }
                dr.append(c, t, n0)?;
            }
            dr.issue(c, pk, seq[p as usize], 7)?;
            let (d, _) = dr.drafts_conf(c, 7)?;
            let r = &rec["pos"][p.to_string()]["sel"];
            let rd = ids(&r["d"]);
            let eq = d.iter().zip(&rd).filter(|(a, b)| a == b).count() as u64;
            tok_eq += eq;
            tok_n += 7;
            pe += u64::from(d == rd);
            pn += 1;
            if d != rd {
                let j = d.iter().zip(&rd).position(|(a, b)| a != b).unwrap_or(0);
                let mg = r["mg"][j].as_f64().unwrap_or(f64::NAN);
                worst.push(serde_json::json!({"pid": pid, "pos": p, "at": j, "card": d,
                    "ref": rd, "ref_margin": mg}));
            }
            card.insert(p, d);
        }
        blk_eq += pe;
        blk_n += pn;
        // INK33's replay at depth k for both proposal sets
        for k in 1..=7usize {
            for (src, acc) in [(0, &mut acc_card), (1, &mut acc_ref)] {
                let mut p = ps[0] as usize;
                while p + 1 < seq.len() {
                    let d: Vec<u32> = if src == 0 {
                        card[&(p as u64)].clone()
                    } else {
                        ids(&rec["pos"][p.to_string()]["sel"]["d"])
                    };
                    let dk = k.min(seq.len() - 1 - p);
                    let mut a = 0;
                    while a < dk && d[a] == seq[p + 1 + a] {
                        a += 1;
                    }
                    acc[k].0 += 1;
                    acc[k].1 += a as u64 + 1;
                    if seq[p] == 200_006 {
                        break;
                    }
                    p += a + 1;
                }
            }
        }
        println!(
            "DF2GATE {pid} blocks {pe}/{pn} equal ({:.0} s)",
            t0.elapsed().as_secs_f64()
        );
        rep.insert(
            pid.clone(),
            serde_json::json!({"blocks_equal": pe, "blocks": pn}),
        );
    }
    let tpr = |a: &[(u64, u64)]| -> Vec<f64> {
        (1..=7)
            .map(|k| a[k].1 as f64 / a[k].0.max(1) as f64)
            .collect()
    };
    let (tc, tr) = (tpr(&acc_card), tpr(&acc_ref));
    let frac = tok_eq as f64 / tok_n.max(1) as f64;
    println!(
        "DF2GATE ALL tokens {tok_eq}/{tok_n} = {:.4} equal, blocks {blk_eq}/{blk_n}; tokens/round card {:?} ref {:?} -> {}",
        frac,
        tc.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>(),
        tr.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>(),
        if frac >= 0.99 { "PASS" } else { "FAIL" }
    );
    let r = serde_json::json!({"tokens_equal": tok_eq, "tokens": tok_n, "frac": frac,
        "blocks_equal": blk_eq, "blocks": blk_n, "tpr_card": tc, "tpr_ref": tr,
        "prompts": rep, "mismatch": worst});
    std::fs::write(out.join("df2gate.json"), serde_json::to_string_pretty(&r)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// INK50: the iu8 form's wave split at the drafter's four K -- the same table the pack builder
    /// writes (`build_ink2_dflash2.iu8_waves`: 2304 -> 9, 4096 -> 16, 6912 -> 9, 9216 -> 12) -- and
    /// `--ink-draft-layers iu8` = the 8-bit records + int8 activations.
    #[test]
    fn the_iu8_wave_split_is_the_packs_and_iu8_implies_the_8bit_records() {
        for (k, w) in [(D, 9), (H2 * HD, 16), (FF2, 9), (4 * D, 12)] {
            assert_eq!(iu8_waves(k), w, "k {k}");
            assert_eq!((k / 128) % w, 0);
        }
        let o = DraftOpt::of(32768, "sk", "iu8").unwrap();
        assert!(o.iu8 && o.w8);
        let o = DraftOpt::of(32768, "sk", "w8").unwrap();
        assert!(!o.iu8 && o.w8);
        assert!(DraftOpt::of(32768, "sk", "i8").is_err());
    }
}
