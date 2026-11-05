//! Device weights of the Flash-Next forward, per rank: whole checkpoint tensors at TP=1 (gate A,
//! the harness's own sources) or FN2's pack records at TP=2 (sha256-checked), the experts either
//! as one per-layer buffer of both ranks' 512 records (TP=1, the harness's `Experts`) or through
//! the tier's placements (TP=2, the engine).

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

use anyhow::{ensure, Context};
use saber_hip::{DevicePtr, HipThread, Stream};
use saber_mem::xtier::XtierHeader;

use super::exec::{HcW, LayerW, MixerW, PleW, H};
use super::store::{Ckpt, PackRecs};

pub const P: &str = "model.language_model.";

/// Where dense bytes come from.
pub enum Src<'a> {
    /// TP=1: whole checkpoint tensors (merged by concatenation where the kernel reads one matrix).
    Ckpt(&'a Ckpt),
    /// TP=2: the rank's pack records (then `shared/`).
    Pack(&'a PackRecs, u32),
}

impl Src<'_> {
    fn get(&self, name: &str) -> anyhow::Result<Vec<u8>> {
        match self {
            Src::Ckpt(c) => c.bf16(name),
            Src::Pack(p, r) => p.raw(*r, name),
        }
    }
}

/// Device allocations a weight set owns.
pub struct Owned {
    pub bufs: Vec<DevicePtr>,
    pub bytes: u64,
}

impl Owned {
    #[must_use]
    pub fn new() -> Owned {
        Owned {
            bufs: Vec::new(),
            bytes: 0,
        }
    }
    /// Upload `b` into a new allocation; its address.
    pub fn up(&mut self, hip: &HipThread, st: &Stream, b: &[u8]) -> anyhow::Result<u64> {
        let d = hip.malloc(b.len().max(4) as u64)?;
        hip.memcpy_h2d_blocking(&d, b, st)?;
        let a = d.addr().as_u64();
        self.bytes += b.len() as u64;
        self.bufs.push(d);
        Ok(a)
    }
}

impl Default for Owned {
    fn default() -> Self {
        Self::new()
    }
}

/// One connection's HC weights. FN108: `w8` uploads the mixing matrices' e4m3 records
/// (`<name>.hc8`: codes then f32 row-group scales, the pack's `shared/hc8.*`) INSTEAD of the bf16
/// ones; the mix then launches the planned symbols' `_h8` twins (`HcW::w8`).
fn hc(
    src: &Src<'_>,
    o: &mut Owned,
    hip: &HipThread,
    st: &Stream,
    pre: &str,
    inject: bool,
    w8: bool,
) -> anyhow::Result<HcW> {
    ensure!(
        !w8 || matches!(src, Src::Pack(..)),
        "the fp8 HC mixers come from a PACK's shared/hc8 member"
    );
    let mixer = |m: &str| -> anyhow::Result<Vec<u8>> {
        let n = format!("{pre}input_mix_weight_{m}.weight");
        src.get(&if w8 { super::hc8::rec_name(&n) } else { n })
    };
    Ok(HcW {
        w8,
        w: o.up(hip, st, &src.get(&format!("{pre}hc_norm.weight"))?)?,
        wdown: o.up(hip, st, &mixer("down")?)?,
        wup: o.up(hip, st, &mixer("up")?)?,
        winj: if inject {
            o.up(
                hip,
                st,
                &src.get(&format!("{pre}block_inject_weight.weight"))?,
            )?
        } else {
            0
        },
    })
}

/// A layer's dense weights (everything but the routed experts). `xgu` / `xdn` are left 0: the
/// caller fills the xptr tables (TP=1: [`Experts`]; TP=2: the tier). FN61e: `t8_drop` (`--fn-trunk
/// w8` / `w8-head`, a pack source) uploads NONE of the bf16 trunk records the W8 records replace --
/// the merged projection (GDN `win`, QSA `wproj`) and the out / o projection are left 0 and
/// [`trunk8_layer`] uploads the merged rows' bf16 suffix on its own.
pub fn dense_layer(
    src: &Src<'_>,
    o: &mut Owned,
    hip: &HipThread,
    st: &Stream,
    l: u32,
    t8_drop: bool,
    hc8: bool,
) -> anyhow::Result<LayerW> {
    ensure!(
        !t8_drop || matches!(src, Src::Pack(..)),
        "the W8 trunk drops the bf16 records of a PACK source only"
    );
    // FN61e: an upload the dropped form leaves out
    let up_t8 = |o: &mut Owned, b: &dyn Fn() -> anyhow::Result<Vec<u8>>| -> anyhow::Result<u64> {
        if t8_drop {
            Ok(0)
        } else {
            o.up(hip, st, &b()?)
        }
    };
    let pre = format!("{P}layers.{l}.");
    let ahc = hc(
        src,
        o,
        hip,
        st,
        &format!("{pre}attn_hyper_connection."),
        true,
        hc8,
    )?;
    let mhc = hc(
        src,
        o,
        hip,
        st,
        &format!("{pre}mlp_hyper_connection."),
        true,
        hc8,
    )?;
    let qsa = l % 4 == 3;
    let mixer = if qsa {
        let a = format!("{pre}self_attn.");
        let wproj = || -> anyhow::Result<Vec<u8>> {
            Ok(match src {
                Src::Ckpt(c) => {
                    let mut b = c.bf16(&format!("{a}q_proj.weight"))?;
                    for n in ["k_proj", "v_proj", "indexer.index_qk_proj"] {
                        b.extend(c.bf16(&format!("{a}{n}.weight"))?);
                    }
                    b
                }
                Src::Pack(p, r) => p.raw(*r, &format!("{a}qkv_idx.weight"))?,
            })
        };
        MixerW::Qsa {
            wproj: up_t8(o, &wproj)?,
            wo: up_t8(o, &|| src.get(&format!("{a}o_proj.weight")))?,
            qw: o.up(hip, st, &src.get(&format!("{a}q_norm.weight"))?)?,
            kw: o.up(hip, st, &src.get(&format!("{a}k_norm.weight"))?)?,
            iqw: o.up(
                hip,
                st,
                &src.get(&format!("{a}indexer.q_layernorm.weight"))?,
            )?,
            ikw: o.up(
                hip,
                st,
                &src.get(&format!("{a}indexer.k_layernorm.weight"))?,
            )?,
        }
    } else {
        let a = format!("{pre}linear_attn.");
        let (win, alog, dtb) = match src {
            Src::Ckpt(c) => {
                let mut b = c.bf16(&format!("{a}in_proj_qkv.weight"))?;
                for n in ["in_proj_z", "in_proj_b", "in_proj_a"] {
                    b.extend(c.bf16(&format!("{a}{n}.weight"))?);
                }
                (
                    b,
                    c.f32_from_bf16(&format!("{a}A_log"))?,
                    c.f32_from_bf16(&format!("{a}dt_bias"))?,
                )
            }
            Src::Pack(p, r) => {
                let mut b = Vec::new();
                if !t8_drop {
                    b = p.raw(*r, &format!("layer.{l}.gdn.in_proj_qkvz.w"))?;
                    b.extend(p.raw(*r, &format!("layer.{l}.gdn.in_proj_ba.w"))?);
                }
                (
                    b,
                    p.raw(*r, &format!("{a}A_log"))?,
                    p.raw(*r, &format!("{a}dt_bias"))?,
                )
            }
        };
        MixerW::Gdn {
            win: up_t8(o, &|| Ok(win.clone()))?,
            conv: o.up(hip, st, &src.get(&format!("{a}conv1d.weight"))?)?,
            alog: o.up(hip, st, &alog)?,
            dtb: o.up(hip, st, &dtb)?,
            nw: o.up(hip, st, &src.get(&format!("{a}norm.weight"))?)?,
            wout: up_t8(o, &|| src.get(&format!("{a}out_proj.weight")))?,
        }
    };
    let m_ = format!("{pre}mlp.");
    let router = o.up(hip, st, &src.get(&format!("{m_}gate.weight"))?)?;
    let sg = o.up(
        hip,
        st,
        &src.get(&format!("{m_}shared_expert_gate.weight"))?,
    )?;
    let ple = if l == 1 {
        let pl = format!("{pre}ple.");
        let g = |n: &str| src.get(&format!("{pl}{n}.weight"));
        Some(PleW {
            pkey: o.up(hip, st, &g("key_proj")?)?,
            pval: o.up(hip, st, &g("value_proj")?)?,
            pk: o.up(hip, st, &g("norm_key")?)?,
            pq: o.up(hip, st, &g("norm_query")?)?,
            pc: o.up(hip, st, &g("norm_conv")?)?,
            pcw: o.up(hip, st, &g("conv1d")?)?,
        })
    } else {
        None
    };
    Ok(LayerW {
        l,
        ahc,
        mhc,
        mixer,
        router,
        sg,
        ple,
        xgu: 0,
        xdn: 0,
        mtp: false,
        mtp_fp8: false,
        mtp_x4: false,
        mtp_xtp2: false,
        t8: None,
    })
}

/// FN15: the MTP drafter's weights on `rank` (TP=2 pack records): the pre-fc norms, BOTH column
/// halves of `fc_embedding` / `fc_hidden` (fn6's law: no gather per GEMM), the MTP decoder layer
/// (QSA, the merged projection alias, bf16 router + shared-expert gate), its mixer, and this rank's
/// 256 bf16 routed experts (+ the bf16 shared expert on rank 0 at entry 256) RESIDENT with their
/// grouped-GEMM tables (`xptr[2e]` = weight). `swap` = the gate's control: fc_embedding and
/// fc_hidden exchanged (a wiring bug on purpose).
pub fn mtp_weights(
    p: &PackRecs,
    rank: u32,
    o: &mut Owned,
    hip: &HipThread,
    st: &Stream,
    swap: bool,
    x: super::exec::MtpX,
) -> anyhow::Result<super::exec::MtpW> {
    let fp8 = x == super::exec::MtpX::Fp8;
    let src = Src::Pack(p, rank);
    let both = |n: &str| -> anyhow::Result<Vec<u8>> {
        let mut b = p.raw(0, n)?;
        b.extend(p.raw(1, n)?);
        ensure!(
            b.len() as u64 == H * H * 2,
            "{n}: both halves = [2560, 2560]"
        );
        Ok(b)
    };
    let (mut fce, mut fch) = (
        o.up(hip, st, &both("mtp.fc_embedding.weight")?)?,
        o.up(hip, st, &both("mtp.fc_hidden.weight")?)?,
    );
    if swap {
        std::mem::swap(&mut fce, &mut fch);
    }
    let pre = "mtp.layers.0.";
    // FN108: the MTP layer's mixers stay bf16 under every --fn-trunk (the drafter is gated by
    // acceptance, not by the target's KL)
    let ahc = hc(
        &src,
        o,
        hip,
        st,
        &format!("{pre}attn_hyper_connection."),
        true,
        false,
    )?;
    let mhc = hc(
        &src,
        o,
        hip,
        st,
        &format!("{pre}mlp_hyper_connection."),
        true,
        false,
    )?;
    let a = format!("{pre}self_attn.");
    let mixer = MixerW::Qsa {
        wproj: o.up(hip, st, &p.raw(rank, &format!("{a}qkv_idx.weight"))?)?,
        wo: o.up(hip, st, &src.get(&format!("{a}o_proj.weight"))?)?,
        qw: o.up(hip, st, &src.get(&format!("{a}q_norm.weight"))?)?,
        kw: o.up(hip, st, &src.get(&format!("{a}k_norm.weight"))?)?,
        iqw: o.up(
            hip,
            st,
            &src.get(&format!("{a}indexer.q_layernorm.weight"))?,
        )?,
        ikw: o.up(
            hip,
            st,
            &src.get(&format!("{a}indexer.k_layernorm.weight"))?,
        )?,
    };
    let router = o.up(hip, st, &src.get(&format!("{pre}mlp.gate.weight"))?)?;
    let sg = o.up(
        hip,
        st,
        &src.get(&format!("{pre}mlp.shared_expert_gate.weight"))?,
    )?;
    let eloc = 256u64;
    let b = |v: &[u64]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    if x.slab4() {
        // FN110: the MXFP4 records (the tier's native form) at three pointers an entry, the
        // shared expert at entry 256 on rank 0 (the trunk's convention; rank 1's entry stays 0);
        // FN138 (`--fn-mtp-split tp2`): this rank's HALF of all 513 records, sliced at load
        let (gu, dn) = if x == super::exec::MtpX::Slab4Tp2 {
            super::mtpx4::tables_tp2(p, rank, o, hip, st)?
        } else {
            super::mtpx4::tables(p, rank, o, hip, st)?
        };
        return Ok(super::exec::MtpW {
            ne: o.up(hip, st, &src.get("mtp.pre_fc_norm_embedding.weight")?)?,
            nh: o.up(hip, st, &src.get("mtp.pre_fc_norm_hidden.weight")?)?,
            fce,
            fch,
            lw: LayerW {
                l: 48,
                ahc,
                mhc,
                mixer,
                router,
                sg,
                ple: None,
                xgu: o.up(hip, st, &b(&gu))?,
                xdn: o.up(hip, st, &b(&dn))?,
                mtp: true,
                mtp_fp8: false,
                mtp_x4: true,
                mtp_xtp2: x == super::exec::MtpX::Slab4Tp2,
                t8: None,
            },
            fm: hc(
                &src,
                o,
                hip,
                st,
                "mtp.hyper_connection_mixer.",
                false,
                false,
            )?,
        });
    }
    let (mut gu, mut dn) = (
        vec![0u64; 2 * (eloc as usize + 1)],
        vec![0u64; 2 * (eloc as usize + 1)],
    );
    for i in 0..eloc {
        let e = u64::from(rank) * eloc + i;
        if fp8 {
            // FN50: Qwen's first-party fp8 experts (the pack's additional file): [w8, scales]
            let [gw, gs, dw, ds] = super::mtp8::names(e as u32);
            let k = 2 * i as usize;
            gu[k] = o.up(hip, st, &p.raw(rank, &gw)?)?;
            gu[k + 1] = o.up(hip, st, &p.raw(rank, &gs)?)?;
            dn[k] = o.up(hip, st, &p.raw(rank, &dw)?)?;
            dn[k + 1] = o.up(hip, st, &p.raw(rank, &ds)?)?;
            continue;
        }
        gu[2 * i as usize] = o.up(
            hip,
            st,
            &p.raw(rank, &format!("mtp.layer.0.expert.{e}.gate_up.w"))?,
        )?;
        dn[2 * i as usize] = o.up(
            hip,
            st,
            &p.raw(rank, &format!("{pre}mlp.experts.{e}.down_proj.weight"))?,
        )?;
    }
    if rank == 0 {
        gu[2 * eloc as usize] = o.up(
            hip,
            st,
            &p.raw(0, "mtp.layer.0.mlp.shared_expert.gate_up.w")?,
        )?;
        dn[2 * eloc as usize] = o.up(
            hip,
            st,
            &p.raw(0, &format!("{pre}mlp.shared_expert.down_proj.weight"))?,
        )?;
    }
    let lw = LayerW {
        l: 48,
        ahc,
        mhc,
        mixer,
        router,
        sg,
        ple: None,
        xgu: o.up(hip, st, &b(&gu))?,
        xdn: o.up(hip, st, &b(&dn))?,
        mtp: true,
        mtp_fp8: fp8,
        mtp_x4: false,
        mtp_xtp2: false,
        t8: None,
    };
    Ok(super::exec::MtpW {
        ne: o.up(hip, st, &src.get("mtp.pre_fc_norm_embedding.weight")?)?,
        nh: o.up(hip, st, &src.get("mtp.pre_fc_norm_hidden.weight")?)?,
        fce,
        fch,
        lw,
        fm: hc(
            &src,
            o,
            hip,
            st,
            "mtp.hyper_connection_mixer.",
            false,
            false,
        )?,
    })
}

/// The final mixer (no inject rows).
pub fn final_mixer(
    src: &Src<'_>,
    o: &mut Owned,
    hip: &HipThread,
    st: &Stream,
    hc8: bool,
) -> anyhow::Result<HcW> {
    hc(
        src,
        o,
        hip,
        st,
        &format!("{P}hyper_connection_mixer."),
        false,
        hc8,
    )
}

/// FN99: where the rank's embedding half lives (`--fn-embed-placement`): FN22's VRAM copy, or ONE
/// pinned host allocation the drafter chain's `fn_draft_pack` reads in place over UVA (the same
/// kernel: a pointer, not a new symbol).
pub enum EmbedHalf {
    /// FN22's resident half (`hipMalloc`).
    Vram(DevicePtr),
    /// FN99: `hipHostMalloc` + its device address (`hipHostGetDevicePointer`).
    Ram {
        /// The pinned bytes (never written after the boot's copy).
        buf: saber_hip::PinnedBuf,
        /// The address a kernel reads it at.
        dev: u64,
    },
}

impl EmbedHalf {
    /// The address `fn_draft_pack` reads rows at.
    #[must_use]
    pub fn addr(&self) -> u64 {
        match self {
            EmbedHalf::Vram(d) => d.addr().as_u64(),
            EmbedHalf::Ram { dev, .. } => *dev,
        }
    }
    /// Bytes.
    #[must_use]
    pub fn len(&self) -> u64 {
        match self {
            EmbedHalf::Vram(d) => d.len(),
            EmbedHalf::Ram { buf, .. } => buf.len() as u64,
        }
    }
    /// The placement's name (`fnembed99`).
    #[must_use]
    pub fn placement(&self) -> &'static str {
        match self {
            EmbedHalf::Vram(_) => "vram",
            EmbedHalf::Ram { .. } => "ram",
        }
    }
}

/// FN22: this rank's half of `embed_tokens` (the pack's identity record, sha256-checked: rank 0
/// rows 0..124,160, rank 1 the rest) -- the device chain's embedding gather. The budget always
/// reserved it (the plan ledger's `embed` class); FN12's prefill gathers on the host. Returns the
/// bytes' fact for the exchange law's premise too: (NaN or Inf patterns, subnormal patterns).
/// FN99: `ram` = the half in pinned host RAM (the class billed in `fnhost`), read in place.
pub fn embed_half(
    p: &PackRecs,
    rank: u32,
    hip: &HipThread,
    st: &Stream,
    ram: bool,
) -> anyhow::Result<(EmbedHalf, u64, u64)> {
    let b = p.raw(rank, &format!("{P}embed_tokens.weight"))?;
    ensure!(
        b.len() as u64 == 124_160 * H * 2,
        "embed half: {} B",
        b.len()
    );
    let (mut nan, mut sub) = (0u64, 0u64);
    for c in b.chunks_exact(2) {
        let x = u16::from_le_bytes([c[0], c[1]]);
        let (e, m) = ((x >> 7) & 0xff, x & 0x7f);
        nan += u64::from(e == 0xff);
        sub += u64::from(e == 0 && m != 0);
    }
    if ram {
        let mut buf = hip.host_malloc(b.len() as u64)?;
        buf.as_mut_slice().copy_from_slice(&b);
        let dev = hip.host_device_addr(&buf)?;
        ensure!(
            buf.as_slice() == &b[..],
            "FN99: the pinned embedding half differs from the record after the copy"
        );
        return Ok((EmbedHalf::Ram { buf, dev }, nan, sub));
    }
    let d = hip.malloc(b.len() as u64)?;
    hip.memcpy_h2d_blocking(&d, &b, st)?;
    Ok((EmbedHalf::Vram(d), nan, sub))
}

/// FN61c: layer `l`'s fp8 W8 trunk records on `rank` (the `-trunk8` pack variant's additional
/// `rank<r>/trunk8.*`, FN61b): the merged projection's W8 prefix as ONE upload (GDN qkvz; QSA q|gate,
/// k, v concatenated along N -- the preshuffled `[N/16, K*16]` rows and the `[N/128, K/128]` scales
/// both concatenate) + the out / o projection. The QSA prefix's bf16 source records are checked
/// equal to the merged bf16 record's rows (the order the W8 prefix replaces).
pub fn trunk8_layer(
    p: &PackRecs,
    rank: u32,
    o: &mut Owned,
    hip: &HipThread,
    st: &Stream,
    l: u32,
    drop: bool,
) -> anyhow::Result<super::exec::T8W> {
    use super::trunk8::names;
    let pre = format!("{P}layers.{l}.");
    // (weight, scales, N) of the records `srcs` at depth `k`, concatenated along N
    let up2 = |o: &mut Owned, srcs: &[String], k: u64| -> anyhow::Result<(u64, u64, u64)> {
        let (mut w, mut s, mut n) = (Vec::new(), Vec::new(), 0u64);
        for src in srcs {
            let [wn, sn] = names(src);
            let wb = p.raw(rank, &wn)?;
            let sb = p.raw(rank, &sn)?;
            let rn = wb.len() as u64 / k;
            ensure!(
                rn * k == wb.len() as u64
                    && rn % 128 == 0
                    && sb.len() as u64 == (rn / 128) * (k / 128) * 4,
                "{wn}: [{rn}, {k}] e4m3 + [{}, {}] f32 scales ({} B)",
                rn / 128,
                k / 128,
                sb.len()
            );
            n += rn;
            w.extend(wb);
            s.extend(sb);
        }
        Ok((o.up(hip, st, &w)?, o.up(hip, st, &s)?, n))
    };
    if l % 4 == 3 {
        let a = format!("{pre}self_attn.");
        let qkv: Vec<String> = ["q_proj", "k_proj", "v_proj"]
            .iter()
            .map(|n| format!("{a}{n}.weight"))
            .collect();
        // the merged bf16 record's prefix rows ARE q|gate, k, v (the W8 rows replace them)
        let merged = p.raw(rank, &format!("{a}qkv_idx.weight"))?;
        let mut cat = Vec::new();
        for n in &qkv {
            cat.extend(p.raw(rank, n)?);
        }
        ensure!(
            merged.len() > cat.len() && merged[..cat.len()] == cat[..],
            "layer {l}: qkv_idx's prefix != q|k|v (the W8 prefix order)"
        );
        let (w, s, n) = up2(o, &qkv, H)?;
        let (wo, so, _) = up2(o, &[format!("{a}o_proj.weight")], 3072)?;
        // FN61e: the merged record's rows past q|k|v (the indexer) as their own upload
        let sfx = if drop {
            o.up(hip, st, &merged[cat.len()..])?
        } else {
            0
        };
        Ok(super::exec::T8W {
            w,
            s,
            n,
            wo,
            so,
            sfx,
        })
    } else {
        let (w, s, n) = up2(o, &[format!("layer.{l}.gdn.in_proj_qkvz.w")], H)?;
        let (wo, so, _) = up2(o, &[format!("{pre}linear_attn.out_proj.weight")], 3072)?;
        // FN61e: in_proj_ba (the merged GDN row's bf16 suffix) as its own upload
        let sfx = if drop {
            o.up(
                hip,
                st,
                &p.raw(rank, &format!("layer.{l}.gdn.in_proj_ba.w"))?,
            )?
        } else {
            0
        };
        Ok(super::exec::T8W {
            w,
            s,
            n,
            wo,
            so,
            sfx,
        })
    }
}

/// FN61c: this rank's W8 head (the vocab half): (weight, scales).
pub fn trunk8_head(
    p: &PackRecs,
    rank: u32,
    o: &mut Owned,
    hip: &HipThread,
    st: &Stream,
) -> anyhow::Result<(u64, u64)> {
    let [wn, sn] = super::trunk8::names("lm_head.weight");
    Ok((
        o.up(hip, st, &p.raw(rank, &wn)?)?,
        o.up(hip, st, &p.raw(rank, &sn)?)?,
    ))
}

/// The head rows of this rank (TP=1: all 248,320; TP=2: the rank's vocab half).
pub fn head(src: &Src<'_>, o: &mut Owned, hip: &HipThread, st: &Stream) -> anyhow::Result<u64> {
    let b = src.get("lm_head.weight")?;
    ensure!(b.len() as u64 % (H * 2) == 0, "lm_head rows");
    o.up(hip, st, &b)
}

/// The shared expert's six sub-records of layer `l` (`shared/dense.pack`), uploaded.
pub fn shared_expert(
    p: &PackRecs,
    o: &mut Owned,
    hip: &HipThread,
    st: &Stream,
    l: u32,
) -> anyhow::Result<[u64; 6]> {
    let mut v = [0u64; 6];
    let mut i = 0;
    for g in ["gate_up", "down"] {
        for part in ["w", "ws", "wref"] {
            v[i] = o.up(
                hip,
                st,
                &p.raw(0, &format!("layer.{l}.mlp.shared_expert.{g}.{part}"))?,
            )?;
            i += 1;
        }
    }
    Ok(v)
}

/// Parse an `experts.xtier` header.
pub fn xtier_header(path: &Path) -> anyhow::Result<XtierHeader> {
    let f = File::open(path).with_context(|| path.display().to_string())?;
    let mut b = vec![0u8; 4096];
    f.read_exact_at(&mut b, 0)?;
    XtierHeader::parse(&path.display().to_string(), &b).map_err(|e| anyhow::anyhow!("{e}"))
}

/// TP=1 (gate A): layer `l`'s 512 experts in ONE buffer (both ranks' records, the harness's
/// `Experts`) + the shared expert at entry 512; returns the two xptr tables (device).
pub fn experts_tp1(
    pack: &PackRecs,
    o: &mut Owned,
    hip: &HipThread,
    st: &Stream,
    l: u32,
) -> anyhow::Result<(u64, u64)> {
    let mut blob = Vec::new();
    let mut hdr = None;
    for r in 0..2u32 {
        let path = pack.dir.join(format!("rank{r}/experts.xtier"));
        let h = xtier_header(&path)?;
        let i0 = h.index(l, h.first_expert).context("layer")?;
        let f = File::open(&path)?;
        let mut b = vec![0u8; (u64::from(h.experts) * h.stride) as usize];
        f.read_exact_at(&mut b, h.record_offset(i0))?;
        blob.extend_from_slice(&b);
        hdr = Some(h);
    }
    let h = hdr.context("header")?;
    let base = o.up(hip, st, &blob)?;
    let sh = shared_expert(pack, o, hip, st, l)?;
    let n = 512usize;
    let (mut gu, mut dn) = (vec![0u64; 3 * (n + 1)], vec![0u64; 3 * (n + 1)]);
    for e in 0..n {
        for i in 0..3 {
            gu[3 * e + i] = base + e as u64 * h.stride + h.sub[i].0;
            dn[3 * e + i] = base + e as u64 * h.stride + h.sub[3 + i].0;
        }
    }
    for i in 0..3 {
        gu[3 * n + i] = sh[i];
        dn[3 * n + i] = sh[3 + i];
    }
    let b = |v: &[u64]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    Ok((o.up(hip, st, &b(&gu))?, o.up(hip, st, &b(&dn))?))
}
