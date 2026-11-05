//! FN150: the ENGINE-PATH byte test of the NVFP4 expert block (`saber fn150-moe-check`, one card).
//!
//! The layer-0 FP4-Local smoke pack's rank-0 records (`qwen38-flashnext-fp4local-l0`, FN144) in
//! one VRAM buffer, the xptr tables built as `ExpertTier::fill_xptr` builds them (record base +
//! the HEADER's sub-record offsets: `w` / `ws` / `gs`), and the REAL engine emitter
//! ([`Emitter::moe_route`] + [`Emitter::moe_experts`] at `Dims::experts_nvfp4`: the router GEMM,
//! the shared-gate GEMV, the LDS route, FN148's form (b) gate_up, the engine's silu-quant, form (b)
//! down, the combine -- the seven-launch train every target decode / prefill <= 512 rows runs),
//! executed by [`RankDev::exec`] with the scratch poisoned. The routes are FN3's layer-0 top-10
//! (`rid`), put through the engine's own route kernel by an EXACT router: the block input holds
//! the wanted logits in its first 512 columns and the router weight is the identity there (one
//! nonzero product per logit: exact in f32 and in bf16). The activations are FN148's (`xq` /
//! `xas`: layer 0's MoE input of FN3's 286 tokens, A8).
//!
//! Cases: `chunk` (all 286 tokens: gate_up ks 1, down ks 1), `b1_<t>` (one token: gate_up ks 4,
//! down ks 2), and the controls `ctl_mx` (the SHIPPED MXFP4 symbols over the same records: what
//! the format check prevents) and `ctl_wrong` (each expert's entry points at its neighbour's
//! record). Every buffer the train writes is dumped (`moe.gu` / `.q` / `.sc` / `.y`, the route's
//! `ids` / `pos`, the combine's `out`) for `tools/capture/fn150/fn150_check.py`, which compares
//! them with FN148's twins (`tools/capture/fn148/fn148_ref.py`).

use std::path::Path;

use anyhow::{ensure, Context};
use saber_exec::loader::PluginSet;
use saber_exec::LoadCounters;
use saber_hip::HipThread;
use saber_kernels::flashnext::FlashNextRules;
use serde_json::json;

use super::dev::RankDev;
use super::exec::{Dims, Emitter, HcW, LayerW, MixOutBufs, MixerW};
use super::kern::{Arena, ArenaSizes, Decls, PLUGINS};
use super::run::d2h_in;

const H: u64 = 2560;
const E: u64 = 512;

/// The test's arguments.
pub struct Fn150Args<'a> {
    pub pack: &'a Path,
    pub prebuilt: &'a Path,
    pub input: &'a Path,
    pub out: &'a Path,
    pub device: i32,
    /// Single-token cases (`b1_<t>` for t < this).
    pub singles: u32,
    /// FN152: the rank whose records (`rank<r>/experts.xtier`) and route half run.
    pub rank: u32,
    /// FN152: FN150's plain route (FN148's cells only) instead of the P1 default.
    pub plain: bool,
    /// FN152: the fused train (FN31 front + FN152's NVFP4 back half) on the decode cases.
    pub fused: bool,
    /// FN152: 8-token decode cases (`b8_<k>` = tokens 8k..8k+8) for k < this.
    pub b8: u32,
}

fn read(p: &Path) -> anyhow::Result<Vec<u8>> {
    std::fs::read(p).with_context(|| p.display().to_string())
}

fn bf16(x: f32) -> u16 {
    // exact for the values used here (small dyadic numbers)
    (x.to_bits() >> 16) as u16
}

/// Run the cases; writes `<out>/<case>/*.bin` + `<out>/fn150_engine.json`.
#[allow(clippy::too_many_lines)]
pub fn run(hip: &HipThread, a: &Fn150Args<'_>) -> anyhow::Result<()> {
    hip.set_device(a.device)?;
    std::fs::create_dir_all(a.out)?;
    let decls = Decls::load(a.prebuilt)?;
    let rules = FlashNextRules::prod()?;
    let lc = LoadCounters::new();
    let ps = PluginSet::load_named(hip, a.prebuilt, &PLUGINS, &lc)?;
    let table = ps.functions(hip)?;
    // ---- the inputs (fn150_prep.py: FN148's ref.npz + FN3's layer-0 route ids)
    let xq = read(&a.input.join("xq.bin"))?;
    let xas = read(&a.input.join("xas.bin"))?;
    let rid = read(&a.input.join("rid.bin"))?;
    let t_all = (xas.len() / 4) as u64;
    ensure!(
        xq.len() as u64 == t_all * H && rid.len() as u64 == t_all * 40,
        "inputs: xq {} B, xas {} B, rid {} B are not one token set",
        xq.len(),
        xas.len(),
        rid.len()
    );
    let rid: Vec<u32> = rid
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // ---- the records: rank 0's layer-0 tier, header-checked (the FN150 format check)
    let tier = a.pack.join(format!("rank{}/experts.xtier", a.rank));
    let xt = saber_mem::xtier::XtierFile::open(&tier, saber_mem::dio::OpenMode::Buffered)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let h = xt.header().clone();
    h.require_format(saber_mem::xtier::XtierFormat::Nvfp4E4m3G16)
        .map_err(|e| anyhow::anyhow!(e))?;
    ensure!(h.first_expert == 256 * a.rank && h.experts == 256 && h.layers >= 1);
    let n_loc = u64::from(h.experts);
    let stream = hip.create_stream(saber_hip::StreamFlags::NonBlocking)?;
    let recs = hip.malloc(n_loc * h.stride)?;
    {
        let mut buf = vec![0u8; (n_loc * h.stride) as usize];
        let mut f = std::fs::File::open(&tier)?;
        use std::io::{Read, Seek, SeekFrom};
        f.seek(SeekFrom::Start(h.record_offset(0)))?;
        f.read_exact(&mut buf)?;
        hip.memcpy_h2d_blocking(&recs, &buf, &stream)?;
    }
    let rb = recs.addr().as_u64();
    // xptr: 257 entries (entry 256 = the shared expert: none here, the route is told so)
    let xptr = |shift: u64, gate: bool| -> Vec<u8> {
        let mut v = vec![0u64; 3 * 257];
        for e in 0..n_loc {
            let base = rb + ((e + shift) % n_loc) * h.stride;
            for k in 0..3 {
                v[(3 * e + k) as usize] = base + h.sub[k as usize + if gate { 0 } else { 3 }].0;
            }
        }
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    };
    let mut tabs = Vec::new();
    for (shift, gate) in [(0, true), (0, false), (1, true), (1, false)] {
        let d = hip.malloc(3 * 257 * 8)?;
        hip.memcpy_h2d_blocking(&d, &xptr(shift, gate), &stream)?;
        tabs.push(d);
    }
    // the exact router: identity on the first 512 columns of the block input
    let mut router = vec![0u8; (E * H * 2) as usize];
    for e in 0..E {
        let o = ((e * H + e) * 2) as usize;
        router[o..o + 2].copy_from_slice(&bf16(1.0).to_le_bytes());
    }
    let router_d = hip.malloc(E * H * 2)?;
    hip.memcpy_h2d_blocking(&router_d, &router, &stream)?;
    let sg_w = hip.malloc(H * 2)?;
    hip.memset(&sg_w, 0, &stream)?;
    let sizes = ArenaSizes {
        scratch: 512 << 20,
        zero: 32 << 20,
        ones: 1 << 20,
        carry: 16 << 20,
    };
    let mut dev = RankDev::new(hip, 0, sizes, &[0u8; 4096])?;
    dev.poison = Some(165);
    let mut cases: Vec<(String, Vec<u64>, bool, u64)> = Vec::new();
    if !a.fused {
        cases.push(("chunk".into(), (0..t_all).collect(), true, 0));
    }
    for t in 0..u64::from(a.singles).min(t_all) {
        cases.push((format!("b1_{t}"), vec![t], true, 0));
    }
    // FN152: 8-token decode blocks over every token (each rank's routes of FN3's 286 tokens)
    for k in 0..u64::from(a.b8).min(t_all.div_ceil(8)) {
        cases.push((
            format!("b8_{k}"),
            (8 * k..(8 * k + 8).min(t_all)).collect(),
            true,
            0,
        ));
    }
    if !a.fused {
        cases.push(("ctl_mx".into(), (0..t_all).collect(), false, 0));
    }
    cases.push(("ctl_wrong".into(), (0..t_all.min(8)).collect(), true, 1));
    // FN152: the fused train's sync region (zeroed once; the kernels re-zero their own words)
    let sync_d = hip.malloc(super::exec::moe_sync_bytes(256))?;
    hip.memset(&sync_d, 0, &stream)?;
    let mut report = Vec::new();
    for (name, toks, nv, shift) in &cases {
        let m = toks.len() as u64;
        // the block input: the wanted logits in columns 0..512 (FN3's 10 ids a token, distinct
        // descending values), -8 elsewhere among the 512, 0 past them
        let mut bi = vec![0u8; (m * H * 2) as usize];
        let mut q8 = vec![0u8; (m * H) as usize];
        let mut sc = vec![0u8; (m * 4) as usize];
        for (i, &t) in toks.iter().enumerate() {
            let row = &mut bi[(i as u64 * H * 2) as usize..((i as u64 + 1) * H * 2) as usize];
            for e in 0..E as usize {
                row[2 * e..2 * e + 2].copy_from_slice(&bf16(-8.0).to_le_bytes());
            }
            for k in 0..10u64 {
                let e = rid[(t * 10 + k) as usize] as usize;
                let v = 8.0 - 0.5 * k as f32;
                row[2 * e..2 * e + 2].copy_from_slice(&bf16(v).to_le_bytes());
            }
            q8[(i as u64 * H) as usize..((i as u64 + 1) * H) as usize]
                .copy_from_slice(&xq[(t * H) as usize..((t + 1) * H) as usize]);
            sc[i * 4..i * 4 + 4].copy_from_slice(&xas[(t * 4) as usize..(t * 4 + 4) as usize]);
        }
        let bi_d = hip.malloc(m * H * 2)?;
        hip.memcpy_h2d_blocking(&bi_d, &bi, &stream)?;
        let q8_d = hip.malloc(m * H)?;
        hip.memcpy_h2d_blocking(&q8_d, &q8, &stream)?;
        let sc_d = hip.malloc(m * 4)?;
        hip.memcpy_h2d_blocking(&sc_d, &sc, &stream)?;
        let mut dims = Dims::new(2, a.rank);
        dims.shared = false;
        if *nv {
            dims = if a.plain {
                dims.experts_nvfp4_plain()
            } else {
                dims.experts_nvfp4()
            };
        }
        let mut em = Emitter::new(&decls, &rules, dims, Arena::carve(dev.base(), sizes));
        em.layer = 0;
        em.tap_layer = 0;
        em.tap_moe = true;
        let lw = LayerW {
            l: 0,
            ahc: HcW::default(),
            mhc: HcW::default(),
            mixer: MixerW::Gdn {
                win: 0,
                conv: 0,
                alog: 0,
                dtb: 0,
                nw: 0,
                wout: 0,
            },
            router: router_d.addr().as_u64(),
            sg: sg_w.addr().as_u64(),
            ple: None,
            xgu: 0,
            xdn: 0,
            mtp: false,
            mtp_fp8: false,
            mtp_x4: false,
            mtp_xtp2: false,
            t8: None,
        };
        let mx = MixOutBufs {
            hid: 0,
            injo: 0,
            bi: bi_d.addr().as_u64(),
            q8: q8_d.addr().as_u64(),
            scale: sc_d.addr().as_u64(),
        };
        let (xgu, xdn) = (
            tabs[(2 * shift) as usize].addr().as_u64(),
            tabs[(2 * shift + 1) as usize].addr().as_u64(),
        );
        let out = if a.fused {
            em.moe_train = super::exec::MoeTrain::Fused;
            em.moe_sync = sync_d.addr().as_u64();
            let rt = em.moe_route(&lw, &mx, m, true)?;
            let o = em.moe_back(&rt, &lw, &mx, m, xgu, xdn, false)?;
            (rt, o)
        } else {
            let rt = em.moe_route(&lw, &mx, m, false)?;
            let o = em.moe_experts(&rt, &mx, m, xgu, xdn, false)?;
            (rt, o)
        };
        let (rt, out) = out;
        let syms: Vec<String> = em.recs.iter().map(|r| r.sym.clone()).collect();
        let taps = std::mem::take(&mut em.taps);
        dev.exec(&table, &mut em)?;
        hip.sync_stream(&dev.stream)?;
        let d = a.out.join(name);
        std::fs::create_dir_all(&d)?;
        let arena = dev.arena.clone();
        let dump = |n: &str, adr: u64, bytes: u64| -> anyhow::Result<()> {
            let b = d2h_in(&dev, &arena, adr, bytes as usize)?;
            std::fs::write(d.join(format!("{n}.bin")), b)?;
            Ok(())
        };
        for (n, adr, bytes) in &taps {
            dump(n, *adr, *bytes)?;
        }
        dump("ids", rt.ids, m * 10 * 4)?;
        dump("pos", rt.pos, m * 10 * 4)?;
        dump("out", out, m * H * 2)?;
        let toks_u: Vec<u8> = toks
            .iter()
            .flat_map(|&t| (t as u32).to_le_bytes())
            .collect();
        std::fs::write(d.join("toks.bin"), toks_u)?;
        println!(
            "fn150 {name}: m {m}, pcap {}, tc {}, tm {}, gu_ks {}, dn_ks {}, {} launches: {}",
            rt.pcap,
            rt.tc,
            rt.tm,
            rt.gu_ks,
            rt.dn_ks,
            syms.len(),
            syms.join(" ")
        );
        report.push(
            json!({"case": name, "m": m, "pcap": rt.pcap, "tm": rt.tm, "gu_ks": rt.gu_ks,
                           "dn_ks": rt.dn_ks, "nvfp4": nv, "shift": shift, "symbols": syms,
                           "rank": a.rank, "plain": a.plain, "fused": a.fused}),
        );
    }
    std::fs::write(
        a.out.join("fn150_engine.json"),
        serde_json::to_string_pretty(&json!({"pack": a.pack, "cases": report}))?,
    )?;
    println!("fn150: wrote {}", a.out.display());
    Ok(())
}
