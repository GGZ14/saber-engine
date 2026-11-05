//! FN80b-2: the TP=2 EXPERT form in the engine (`--fn-experts ep|tp2`; FLASHNEXT-PLAN §13, LEDGER
//! FN80a / FN80b-2).
//!
//! `ep` (THE DEFAULT, the rollback) is today's boot: rank r owns experts `[256r, 256r + 256)`
//! whole (`rank<r>/experts.xtier`). `tp2` gives every rank half r of EVERY expert along the
//! intermediate dimension (`rank<r>/experts.tp2.xtier`, FN80a's slicer): both ranks route all 512
//! experts as LOCAL (`e_lo` 0, `e_loc` 512, xptr 3 x 513 entries a GEMM), the shared expert is split
//! the same way (sliced from the pack's whole record at boot, both ranks), the combine is
//! unchanged and the partial sums ride the post-MoE all-reduce the layer already pays.
//!
//! THE TIER AT HALF GEOMETRY: a slot is one half record (the header's stride, 1,310,720 B; 24,576
//! records a rank) -- the VRAM slots, the pinned RAM L2, the boot fill, the valve, the staging ring
//! and the lend law all take [`XGeom`], read from the split marker of the file's header, never from
//! a flag. ONE PLACEMENT, MIRRORED: both ranks see the same routes (the router runs on the reduced
//! hidden state) and run the same deterministic policy over the same profile, so their placements
//! are equal by construction; [`placement_digest`] is checked across the pair (`agree`) and a
//! divergence is a FATAL by name.
//!
//! The cells are FN80b-1's shape key (`moe::ExpertShape::TpHalf`): gate_up's half on the committed
//! cells (N 640), the down half at DBK 64 (`moe_grouped_w4a8_bk64_*`, K 320) at decode / verify /
//! small chunks and the committed `pf_tm*` tiles above 512 rows, the fused back half
//! `moe_back_h2_*`; the split counts are EP's measured cells (the class choice kept).

use std::path::Path;

use anyhow::{ensure, Context};
use saber_hip::{HipThread, Stream};
use saber_mem::expert_tier::Placement;
use saber_mem::flashnext_budget::geometry;
use saber_mem::xtier::{record_layout, record_layout_nvfp4, XtierFormat, XtierHeader};

use super::model::Owned;
use super::run::FnFlags;
use super::store::PackRecs;
use super::xsplit;

/// `--fn-experts`: `ep` = the default and the rollback (FN80b-2).
pub const DEFAULT_FN_EXPERTS: &str = "ep";

/// The expert tier's geometry at this boot (one rank's file; both ranks' are equal).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XGeom {
    /// The TP-split form (`--fn-experts tp2`).
    pub tp2: bool,
    /// One record's stride (the slot size).
    pub stride: u64,
    /// Records a rank (48 x local experts).
    pub records: u64,
    /// xptr entries a GEMM (local experts + the shared expert).
    pub entries: usize,
    /// FN150: the records' format (`--fn-experts-fmt`; MXFP4 = every pre-FN150 boot).
    pub fmt: XtierFormat,
}

/// `--fn-experts-fmt`: `mxfp4` = the default (FN150; `nvfp4` = the FP4-Local v2 tier).
pub const DEFAULT_FN_EXPERTS_FMT: &str = "mxfp4";

/// FN150: the NVFP4 record's stride at Flash-Next's (2,560, 640) = `record_layout_nvfp4`'s (676
/// pages; MXFP4's is 639).
pub const NVFP4_EXPERT_STRIDE: u64 = 2_768_896;

impl XGeom {
    /// Today's EP=2 geometry (FN2's constants: the rollback's ledger is byte-for-byte today's).
    pub const EP: XGeom = XGeom {
        tp2: false,
        stride: geometry::EXPERT_STRIDE,
        records: geometry::EXPERT_RECORDS,
        entries: 257,
        fmt: XtierFormat::Mxfp4E8m0,
    };

    /// FN150: the FP4-Local EP=2 geometry (the v2 NVFP4 tier, 676-page records; the same records
    /// a rank as [`XGeom::EP`]).
    pub const NVFP4_EP: XGeom = XGeom {
        stride: NVFP4_EXPERT_STRIDE,
        fmt: XtierFormat::Nvfp4E4m3G16,
        ..XGeom::EP
    };

    /// FN150: the geometry of rank `r`'s NVFP4 EP header, refused by name unless it is FN144's
    /// whole-expert v2 law at 48 layers (the layer-0 smoke pack is a kernel fixture, not a tier).
    pub fn from_nvfp4_header(h: &XtierHeader, r: u32) -> anyhow::Result<XGeom> {
        let (sub, payload, stride) = record_layout_nvfp4(xsplit::HIDDEN, xsplit::INTER_H * 2);
        let g = XGeom::NVFP4_EP;
        h.require_format(XtierFormat::Nvfp4E4m3G16)
            .map_err(|e| anyhow::anyhow!(e))?;
        ensure!(
            h.split_parts == 0
                && h.first_expert == 256 * r
                && u64::from(h.layers) * u64::from(h.experts) == g.records
                && u64::from(h.layers) == xsplit::LAYERS
                && u64::from(h.hidden) == xsplit::HIDDEN
                && u64::from(h.inter) == 2 * xsplit::INTER_H
                && h.sub == sub
                && h.payload == payload
                && h.stride == stride
                && stride == NVFP4_EXPERT_STRIDE,
            "--fn-experts-fmt nvfp4: rank {r}'s experts.xtier is not FN144's whole-expert v2 NVFP4 \
             tier at 48 layers (layers {}, experts {}+{}, split {}, inter {}, stride {}; the \
             layer-0 smoke pack is a kernel fixture, not a bootable tier)",
            h.layers,
            h.first_expert,
            h.experts,
            h.split_parts,
            h.inter,
            h.stride
        );
        Ok(g)
    }

    /// The tier file of rank `r` under this form.
    #[must_use]
    pub fn file(&self, r: u32) -> String {
        if self.tp2 {
            format!("rank{r}/{}", xsplit::FILE)
        } else {
            format!("rank{r}/experts.xtier")
        }
    }

    /// u64 words of one xptr table (3 pointers an entry).
    #[must_use]
    pub fn xn(&self) -> usize {
        3 * self.entries
    }

    /// The xptr tables' bytes (48 layers x 2 GEMMs) = the ledger class `xtier.xptr`.
    #[must_use]
    pub fn xptr_bytes(&self) -> u64 {
        48 * 2 * self.xn() as u64 * 8
    }

    /// The geometry of a TP-split header (rank `r`'s), refused by name unless it is FN80a's law.
    pub fn from_tp2_header(h: &XtierHeader, r: u32) -> anyhow::Result<XGeom> {
        let (sub, payload, stride) = xsplit::half_layout();
        ensure!(
            h.split_parts == xsplit::PARTS as u32
                && h.split_index == r
                && h.first_expert == 0
                && u64::from(h.experts) == xsplit::EXPERTS
                && u64::from(h.layers) == xsplit::LAYERS
                && u64::from(h.hidden) == xsplit::HIDDEN
                && u64::from(h.inter) == xsplit::INTER_H
                && h.sub == sub
                && h.payload == payload
                && h.stride == stride,
            "--fn-experts tp2: rank {r}'s {} is not FN80a's half-record law (split {} / {}, \
             experts {}+{}, inter {}, stride {})",
            xsplit::FILE,
            h.split_parts,
            h.split_index,
            h.first_expert,
            h.experts,
            h.inter,
            h.stride
        );
        Ok(XGeom {
            tp2: true,
            stride: h.stride,
            records: h.records(),
            entries: h.experts as usize + 1,
            fmt: h.format,
        })
    }

    /// This boot's geometry: `ep` = [`XGeom::EP`]; `tp2` = both ranks' split headers (equal).
    pub fn of(f: &FnFlags, pack: &Path) -> anyhow::Result<XGeom> {
        // FN150: an NVFP4 boot reads both ranks' v2 headers (format, 48 layers, the 676-page
        // stride); the MXFP4 default stays FN2's constants (the worker checks the tag)
        if f.experts_fmt() == XtierFormat::Nvfp4E4m3G16 {
            ensure!(
                !f.experts_tp2(),
                "--fn-experts-fmt nvfp4: no NVFP4 TP-split member exists (FN80a's slicer is \
                 MXFP4's; --fn-experts ep only)"
            );
            for r in 0..2u32 {
                let h = super::model::xtier_header(&pack.join(XGeom::EP.file(r)))?;
                XGeom::from_nvfp4_header(&h, r)?;
            }
            return Ok(XGeom::NVFP4_EP);
        }
        if !f.experts_tp2() {
            return Ok(XGeom::EP);
        }
        let mut g = None;
        for r in 0..2u32 {
            let p = pack.join(XGeom::EP.file(r).replace("experts.xtier", xsplit::FILE));
            ensure!(
                p.exists(),
                "--fn-experts tp2: this pack has no TP-split expert member ({} absent; build it \
                 with `saber flashnext-pack --experts tp2`)",
                p.display()
            );
            let h = super::model::xtier_header(&p)?;
            let x = XGeom::from_tp2_header(&h, r)?;
            ensure!(g.is_none_or(|y| y == x), "the ranks' split files differ");
            g = Some(x);
        }
        g.context("geometry")
    }

    /// The boot line.
    #[must_use]
    pub fn line(&self) -> String {
        format!(
            "fnexperts: {} -- slot = one {} {} record ({} B stride), {} records a rank, xptr {} \
             entries a GEMM{}",
            if self.tp2 { "tp2 (FN80b-2)" } else { "ep" },
            if self.tp2 { "HALF" } else { "whole" },
            self.fmt.key(),
            self.stride,
            self.records,
            self.entries,
            if self.tp2 {
                "; both ranks route all 512 experts as local, the shared expert split, ONE \
                 mirrored placement; the MTP experts stay EP"
            } else if self.fmt == XtierFormat::Nvfp4E4m3G16 {
                "; FN150: FN148's form (b) fn148_w4a8b_* cells, the f32 globals in the epilogue \
                 (gsplit), 676 pages a slot"
            } else {
                ""
            }
        )
    }

    /// FN150: the slot-accounting comparison line (`fnexperts150 rank<r>:`): this boot's VRAM
    /// expert slots against the same ledger remainder at MXFP4's stride.
    #[must_use]
    pub fn slots_line(&self, r: u32, slots: u64, remainder_bytes: u64) -> String {
        let at_mx = remainder_bytes / geometry::EXPERT_STRIDE;
        format!(
            "fnexperts150 rank{r}: {} records {} B a slot -> {slots} VRAM expert slots (MXFP4's \
             {} B stride would give {at_mx}: {:+.2} %)",
            self.fmt.key(),
            self.stride,
            geometry::EXPERT_STRIDE,
            (slots as f64 / at_mx.max(1) as f64 - 1.0) * 100.0
        )
    }
}

/// FN150: refuse an NVFP4 boot by name for every piece with no NVFP4 body yet (so a half-wired
/// boot cannot start): the TP split, the one-process form, the fused / `_p1` back half, MTP
/// experts, and -- always, until FN133 P3 / P4 land -- the prefill tiles (every boot's chunk
/// is >= 2,048 rows; FN148's cells stop at 512) and the shared expert's W8 lane (FP4-Local's
/// shared expert is MXFP8; it rides the grouped table's entry 256 as an MXFP4 record today).
/// Returns the list (empty for an MXFP4 boot).
#[must_use]
pub fn nvfp4_refusals(f: &FnFlags, tp: u32) -> Vec<String> {
    let mut v = Vec::new();
    if f.experts_fmt() != XtierFormat::Nvfp4E4m3G16 {
        return v;
    }
    if tp != 2 {
        v.push("TP=1 (gate A's 512-expert tables are MXFP4; --tp 2 only)".to_string());
    }
    if f.fn_one_process {
        v.push("the one-process form (--fn-one-process; the worker form only)".to_string());
    }
    if f.experts_tp2() {
        v.push(
            "--fn-experts tp2 (no NVFP4 TP-split tier: FN152 built the moe_back_h2 / far \
                NVFP4 back halves, the pack has no half records)"
                .to_string(),
        );
    }
    // FN152: --fn-moe-train fused runs FN152's NVFP4 back half (fn152_moe_back_gu*_dn*)
    if f.mtp_k() > 0 {
        v.push(
            "MTP experts (the slab4 / fp8 MTP records are not FP4-Local's W4A16 NVFP4; \
                --fn-mtp 0 only)"
                .to_string(),
        );
    }
    if f.shared_split() {
        v.push(
            "--fn-shared-expert split (the shared expert has no NVFP4 / W8 grouped lane)"
                .to_string(),
        );
    }
    v.push(format!(
        "prefill tiles: this boot's chunks reach {} rows and FN148's NVFP4 cells stop at 512 \
         (no pf_tm* NVFP4 tile; FN133 P3)",
        f.max_chunk()
    ));
    v.push(
        "the shared expert's W8 lane (FP4-Local's shared expert is MXFP8; the grouped \
            table's entry 256 is an MXFP4 record; FN133 P4)"
            .to_string(),
    );
    v
}

/// Refuse a tp2 boot where it cannot run (FN80b-2's arm: TP=2, the worker form).
pub fn check(f: &FnFlags, tp: u32) -> anyhow::Result<()> {
    if !f.experts_tp2() {
        return Ok(());
    }
    ensure!(tp == 2, "--fn-experts tp2 is a TP=2 form (--tp 2)");
    ensure!(
        !f.fn_one_process,
        "--fn-experts tp2 runs in the worker form (not --fn-one-process)"
    );
    // FN80d: the valve's device miss flag (FN53c's arm) names a miss by EP's rank-local id
    // (`256 * rank + j`); it has no tp2 form
    ensure!(
        !(f.fn_mem_valve == "on" && f.fn_valve_graphs == "on"),
        "--fn-experts tp2: the valve's device miss flag (--fn-mem-valve on --fn-valve-graphs on) \
         reads EP's expert range; refused (--fn-valve-graphs off = the demand path runs)"
    );
    Ok(())
}

/// The shared expert's half `h` of layer `l`: the pack's six whole sub-records joined into the
/// whole record layout, sliced by FN80a's law ([`xsplit::slice_payload`]: `down.wref` = the WHOLE
/// row's), checked (the two halves join back to the whole, byte for byte), uploaded as ONE buffer
/// of the half layout; returns the six sub-record addresses.
pub fn shared_expert_half(
    p: &PackRecs,
    o: &mut Owned,
    hip: &HipThread,
    st: &Stream,
    l: u32,
    h: u32,
) -> anyhow::Result<[u64; 6]> {
    let whole = shared_whole(p, l)?;
    let (half, ds) = slice_half(&whole, h)?;
    let base = o.up(hip, st, &half)?;
    let mut v = [0u64; 6];
    for (i, x) in v.iter_mut().enumerate() {
        *x = base + ds[i].0;
    }
    Ok(v)
}

/// The shared expert's six records of layer `l` as ONE whole-layout payload.
fn shared_whole(p: &PackRecs, l: u32) -> anyhow::Result<Vec<u8>> {
    let (ss, payload, _) = record_layout(xsplit::HIDDEN, xsplit::INTER);
    let mut whole = vec![0u8; payload as usize];
    let mut i = 0;
    for g in ["gate_up", "down"] {
        for part in ["w", "ws", "wref"] {
            let b = p.raw(0, &format!("layer.{l}.mlp.shared_expert.{g}.{part}"))?;
            ensure!(
                b.len() as u64 == ss[i].1,
                "layer {l} shared_expert.{g}.{part}: {} B, the whole record law says {}",
                b.len(),
                ss[i].1
            );
            whole[ss[i].0 as usize..(ss[i].0 + ss[i].1) as usize].copy_from_slice(&b);
            i += 1;
        }
    }
    Ok(whole)
}

/// A record's six sub-records `(offset, bytes)` (`xtier::record_layout`).
pub type SubLayout = [(u64, u64); 6];

/// Half `h` of a whole-layout record, with the byte law: both halves join back to `whole`.
pub fn slice_half(whole: &[u8], h: u32) -> anyhow::Result<(Vec<u8>, SubLayout)> {
    let (ss, payload, _) = record_layout(xsplit::HIDDEN, xsplit::INTER);
    let (ds, hp, _) = xsplit::half_layout();
    ensure!(whole.len() as u64 == payload, "the whole record's bytes");
    let mut halves = [vec![0u8; hp as usize], vec![0u8; hp as usize]];
    for (k, d) in halves.iter_mut().enumerate() {
        xsplit::slice_payload(whole, &ss, k as u64, d, &ds);
    }
    let joined = xsplit::join_payload([&halves[0], &halves[1]], &ds, &ss, payload);
    ensure!(
        joined.as_deref() == Some(whole),
        "the shared expert's halves do not join back to the whole record (FN80a's byte law)"
    );
    Ok((std::mem::take(&mut halves[h as usize]), ds))
}

/// The placement map's digest (FNV-1a 64 over every record's placement, in record order): equal on
/// both ranks by construction under tp2 (the mirrored planner); the pair `agree`s on it.
#[must_use]
pub fn placement_digest(pl: &[Placement]) -> (u64, [u64; 3]) {
    let mut d: u64 = 0xcbf2_9ce4_8422_2325;
    let mut n = [0u64; 3];
    let mut put = |x: u64| {
        for b in x.to_le_bytes() {
            d ^= u64::from(b);
            d = d.wrapping_mul(0x0100_0000_01b3);
        }
    };
    for p in pl {
        let (k, v) = match *p {
            Placement::Vram(s) => (0u64, u64::from(s)),
            Placement::Ram(s) => (1, u64::from(s)),
            Placement::Fetch => (2, 0),
        };
        n[k as usize] += 1;
        put((k << 32) | v);
    }
    (d, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use saber_mem::expert_tier::{ExpertTier, TierParams};
    use saber_mem::route_trace::CountMover;
    use saber_mem::xtier::header_block;

    fn fnx(args: &[&str]) -> FnFlags {
        #[derive(clap::Parser)]
        struct C {
            #[command(flatten)]
            f: FnFlags,
        }
        let mut v = vec!["saber"];
        v.extend_from_slice(args);
        <C as clap::Parser>::try_parse_from(v).unwrap().f
    }

    fn parsed(h: &XtierHeader) -> XtierHeader {
        XtierHeader::parse("synthetic", &header_block(h)).expect("a well-formed header")
    }

    /// The tier's geometry comes from a synthetic MARKED header (FN80a's writer, parsed back): a
    /// half record's stride, twice the records, 513 xptr entries; the controls refuse by name.
    /// FN150: the NVFP4 geometry comes from a v2 header at FN144's law (676 pages, 48 layers,
    /// whole experts); an MXFP4 header, a one-layer (smoke) header and a wrong stride refuse by
    /// name; the real smoke pack's header refuses as a non-bootable fixture.
    #[test]
    fn fn150_the_nvfp4_geometry_comes_from_a_v2_header() {
        let (sub, payload, stride) = record_layout_nvfp4(2560, 640);
        assert_eq!((stride, stride / 4096), (NVFP4_EXPERT_STRIDE, 676));
        for r in 0..2u32 {
            let mut h = tier_header(r, false);
            h.sub = sub;
            h.payload = payload;
            h.stride = stride;
            h.hidden = 2560;
            h.inter = 640;
            h.format = XtierFormat::Nvfp4E4m3G16;
            let g = XGeom::from_nvfp4_header(&h, r).unwrap();
            assert_eq!(g, XGeom::NVFP4_EP);
            assert_eq!((g.records, g.entries, g.tp2), (12_288, 257, false));
            assert!(g.line().contains("NVFP4_E4M3G16") && g.line().contains("2768896"));
            let mut mx = h.clone();
            mx.format = XtierFormat::Mxfp4E8m0;
            let e = XGeom::from_nvfp4_header(&mx, r).unwrap_err().to_string();
            assert!(e.contains("MXFP4_E8M0"), "{e}");
            let mut one = h.clone();
            one.layers = 1;
            let e = XGeom::from_nvfp4_header(&one, r).unwrap_err().to_string();
            assert!(e.contains("smoke pack"), "{e}");
            let mut st = h.clone();
            st.stride = 2_617_344;
            assert!(XGeom::from_nvfp4_header(&st, r).is_err());
            assert!(XGeom::from_nvfp4_header(&h, 1 - r).is_err());
        }
        // MXFP4 = FN2's constants, byte for byte (the default boot's ledger is today's)
        assert_eq!(XGeom::EP.fmt, XtierFormat::Mxfp4E8m0);
        assert_eq!(XGeom::EP.stride, geometry::EXPERT_STRIDE);
        let smoke = Path::new("/mnt/nvme/saber-packs-smoke/qwen38-flashnext-fp4local-l0");
        if smoke.join("rank0/experts.xtier").exists() {
            let f = fnx(&["--fn-experts-fmt", "nvfp4"]);
            let e = XGeom::of(&f, smoke).unwrap_err().to_string();
            assert!(e.contains("smoke pack") && e.contains("layers 1"), "{e}");
        }
        let s = XGeom::NVFP4_EP.slots_line(0, 6_901, 7_301 * geometry::EXPERT_STRIDE);
        assert!(
            s.contains("fnexperts150 rank0") && s.contains("-5.48 %"),
            "{s}"
        );
    }

    /// FN150: an NVFP4 boot names every piece with no NVFP4 body; an MXFP4 boot names none.
    #[test]
    fn fn150_the_nvfp4_boot_refuses_the_unwired_pieces_by_name() {
        assert!(nvfp4_refusals(&fnx(&[]), 2).is_empty());
        assert!(nvfp4_refusals(&fnx(&["--fn-experts", "tp2"]), 2).is_empty());
        // FN152: the served train (`fused`) runs FN152's NVFP4 back half: no refusal either way
        let base = nvfp4_refusals(
            &fnx(&["--fn-experts-fmt", "nvfp4", "--fn-moe-train", "legacy"]),
            2,
        );
        assert_eq!(base.len(), 2, "{base:?}");
        let fused = nvfp4_refusals(&fnx(&["--fn-experts-fmt", "nvfp4"]), 2);
        assert_eq!(fused, base, "FN152: fused is not refused");
        assert!(base[0].contains("prefill tiles") && base[1].contains("shared expert"));
        for (args, what) in [
            (&["--fn-experts", "tp2"][..], "--fn-experts tp2"),
            (&["--fn-mtp", "2"][..], "MTP experts"),
            (&["--fn-serve", "8099"][..], "MTP experts"),
            (&["--fn-one-process"][..], "one-process"),
            (
                &["--fn-shared-expert", "split"][..],
                "--fn-shared-expert split",
            ),
        ] {
            let mut v = vec!["--fn-experts-fmt", "nvfp4"];
            v.extend_from_slice(args);
            let r = nvfp4_refusals(&fnx(&v).resolved(), 2);
            assert!(r.iter().any(|x| x.contains(what)), "{args:?}: {r:?}");
        }
        assert!(nvfp4_refusals(&fnx(&["--fn-experts-fmt", "nvfp4"]), 1)[0].contains("TP=1"));
    }

    #[test]
    fn the_tp2_geometry_comes_from_the_header_split_marker() {
        for r in 0..2u32 {
            let h = parsed(&xsplit::half_header(r, [7; 32]));
            assert_eq!((h.split_parts, h.split_index), (2, r));
            let g = XGeom::from_tp2_header(&h, r).unwrap();
            assert_eq!(
                g,
                XGeom {
                    tp2: true,
                    stride: 1_310_720,
                    records: 24_576,
                    entries: 513,
                    fmt: XtierFormat::Mxfp4E8m0,
                }
            );
            assert_eq!(g.file(r), format!("rank{r}/experts.tp2.xtier"));
            assert_eq!(g.xn(), 1_539);
            assert_eq!(g.xptr_bytes(), 48 * 2 * 1_539 * 8);
            // a slot is one HALF: twice the count; the bytes +0.16 % (the second wref + paging)
            assert_eq!(g.records, 2 * XGeom::EP.records);
            assert_eq!(2 * g.stride - XGeom::EP.stride, 4_096);
            // controls: the other rank's half, a whole-expert (EP) header
            assert!(XGeom::from_tp2_header(&h, 1 - r).is_err());
            let mut whole = h.clone();
            whole.split_parts = 0;
            whole.split_index = 0;
            assert!(XGeom::from_tp2_header(&whole, r).is_err());
        }
        assert_eq!(XGeom::EP.xptr_bytes(), geometry::XPTR_BYTES);
        assert_eq!(XGeom::EP.file(1), "rank1/experts.xtier");
    }

    /// `ep` is the default; `tp2` refuses the one-process form / TP=1 by name, a pack without the
    /// member by name; a 27B pack refuses the flag.
    #[test]
    fn the_experts_flag_defaults_to_ep_and_tp2_refuses_by_name() {
        let d = fnx(&[]);
        assert_eq!(d.fn_experts, DEFAULT_FN_EXPERTS);
        assert!(!d.experts_tp2());
        assert!(check(&d, 1).is_ok());
        assert!(d.refuse_on_27b().is_ok());
        let t = fnx(&["--fn-experts", "tp2"]);
        assert!(t.experts_tp2());
        assert!(check(&t, 2).is_ok());
        assert!(t.refuse_on_27b().is_err());
        assert!(check(&t, 1).unwrap_err().to_string().contains("TP=2"));
        // the decode forms run (FN80b-1's cells on main)
        assert!(check(
            &fnx(&["--fn-experts", "tp2", "--fn-serve", "8099"]).resolved(),
            2
        )
        .is_ok());
        let e = check(&fnx(&["--fn-experts", "tp2", "--fn-one-process"]), 2)
            .unwrap_err()
            .to_string();
        assert!(e.contains("worker form"), "{e}");
        let dir = saber_test_scratch::ScratchDir::new("fn80b2-nopack");
        let e = XGeom::of(&t, dir.path()).unwrap_err().to_string();
        assert!(e.contains("no TP-split expert member"), "{e}");
        assert_eq!(XGeom::of(&d, dir.path()).unwrap(), XGeom::EP);
    }

    fn tier_header(rank: u32, tp2: bool) -> XtierHeader {
        let (experts, first) = if tp2 { (512, 0) } else { (256, rank * 256) };
        XtierHeader {
            rank,
            layers: 48,
            first_expert: first,
            experts,
            stride: if tp2 { 1_310_720 } else { 2_617_344 },
            payload: if tp2 { 1_310_720 } else { 2_617_344 },
            sub: [(0, 0); 6],
            hidden: 0,
            inter: 0,
            table_off: 0,
            table_bytes: 0,
            data_off: 0,
            file_bytes: 0,
            table_sha256: [0; 32],
            plan_sha256: [0; 32],
            split_parts: if tp2 { 2 } else { 0 },
            split_index: if tp2 { rank } else { 0 },
            format: saber_mem::xtier::XtierFormat::Mxfp4E8m0,
        }
    }

    /// A tier at the served law: `vram` slots, the RAM L2 = the complement (no NVMe-only record).
    fn tier(h: &XtierHeader, vram: u32) -> ExpertTier {
        let p = TierParams {
            vram_slots: vram,
            ram_slots: h.layers * h.experts - vram,
            max_inserts: 2,
            lrfu_lambda: 0.98,
            ..TierParams::default()
        };
        ExpertTier::new(h, p, None, 1 << 40, 1 << 41).unwrap()
    }

    /// A replayed route trace: `steps` steps of `t` tokens x top-10 over 512 experts, skewed (a
    /// hot set per layer), deterministic.
    fn trace(steps: u32, t: u32) -> Vec<Vec<Vec<u32>>> {
        let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        (0..steps)
            .map(|_| {
                (0..48u32)
                    .map(|l| {
                        (0..t * 10)
                            .map(|_| {
                                let v = next();
                                // 70 % of routes in a 96-expert hot set that moves with the layer
                                if v % 10 < 7 {
                                    (((v >> 8) % 96 + u64::from(l) * 7) % 512) as u32
                                } else {
                                    ((v >> 8) % 512) as u32
                                }
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    fn run(t: &mut ExpertTier, tr: &[Vec<Vec<u32>>], keep: impl Fn(u32) -> bool) -> Vec<u64> {
        let mut m = CountMover::default();
        let mut out = Vec::new();
        for step in tr {
            t.begin_step();
            for (l, ids) in step.iter().enumerate() {
                let ids: Vec<u32> = ids.iter().copied().filter(|&e| keep(e)).collect();
                t.resolve(l as u32, &ids, &mut m).unwrap();
            }
            t.end_step(&mut m).unwrap();
            out.push(placement_digest(t.placements()).0);
        }
        out
    }

    /// THE MIRRORED PLANNER on a replayed route trace: the two ranks' tp2 tiers (each its own
    /// half file's header) fed the same replicated routes hold EQUAL placements after every step;
    /// the EP replay is unchanged (a rank's tier ignores the other rank's experts: fed only its
    /// own routes it decides the same). NEGATIVE CONTROLS (known to fail the law): EP's two ranks
    /// fed the same trace do NOT hold equal maps (two placements, not one), and a rank fed one
    /// moved trace diverges (what `agree` turns into a FATAL).
    #[test]
    fn the_mirrored_planner_holds_equal_placements_on_a_replayed_trace() {
        let tr = trace(24, 8);
        let vram = 6_000;
        let mut a = tier(&tier_header(0, true), vram);
        let mut b = tier(&tier_header(1, true), vram);
        let (da, db) = (run(&mut a, &tr, |_| true), run(&mut b, &tr, |_| true));
        assert_eq!(da, db, "the tp2 ranks' placements diverged");
        assert_eq!(a.placements(), b.placements());
        assert_eq!(
            format!("{:?}", a.totals()),
            format!("{:?}", b.totals()),
            "saber_xtier_* equal by construction"
        );
        assert!(
            a.totals().1.inserts > 0,
            "the trace must move the placement"
        );
        // EP: unchanged by the shared code (the rank-0 tier fed every route == fed its own)
        let mut e0 = tier(&tier_header(0, false), vram / 2);
        let mut e0own = tier(&tier_header(0, false), vram / 2);
        let d0 = run(&mut e0, &tr, |_| true);
        assert_eq!(d0, run(&mut e0own, &tr, |e| e < 256));
        // the control: EP's two ranks keep two different placements
        let mut e1 = tier(&tier_header(1, false), vram / 2);
        let d1 = run(&mut e1, &tr, |_| true);
        assert_ne!(
            d0, d1,
            "EP's ranks are not mirrored (the control must differ)"
        );
        // a perturbed rank (every route from step 3 on moved by 256 experts) diverges
        let mut tr2 = tr.clone();
        for step in tr2.iter_mut().skip(3) {
            for layer in step.iter_mut() {
                for e in layer.iter_mut() {
                    *e = (*e + 256) % 512;
                }
            }
        }
        let mut c = tier(&tier_header(1, true), vram);
        assert_ne!(run(&mut c, &tr2, |_| true), da);
    }

    /// The route table at e_loc 512 (the dry run's layer-0 MoE launches on BOTH ranks): every
    /// route keeps all 512 experts local with the shared rows on both ranks, the expert GEMMs run
    /// at half width (gate_up N 640 / K 2,560, down N 2,560 / K 320 on FN80b-1's DBK-64 cell or the
    /// pf tile, the silu at n 320) -- EP's routes (rank r: [256r, +256), the shared rows on rank 0) at
    /// N 1,280 / K 640 are the control.
    #[test]
    fn the_route_table_keeps_512_experts_local_on_both_ranks() {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../prebuilt/gfx1201");
        let decls = super::super::kern::Decls::load(&p).expect("manifests");
        let rules = saber_kernels::flashnext::FlashNextRules::prod().unwrap();
        let i32_at = |a: &[u8], o: usize| i32::from_le_bytes(a[o..o + 4].try_into().unwrap());
        for (xtp2, m) in [(true, 64u32), (true, 2_048), (false, 64)] {
            let dr =
                super::super::plan::dry_run_cap_x(&decls, &rules, 2, m, false, false, false, xtp2)
                    .unwrap();
            for r in 0..2usize {
                let mine: Vec<_> = dr.moe_args.iter().filter(|x| x.0 == r).collect();
                let route = mine
                    .iter()
                    .find(|x| x.1.starts_with("moe_route"))
                    .expect("a route launch");
                // (logits, T, e_lo, e_loc, tile_blocks, with_shared)
                let (e_lo, e_loc, tm, sh) = (
                    i32_at(&route.2, 12),
                    i32_at(&route.2, 16),
                    i32_at(&route.2, 20),
                    i32_at(&route.2, 24),
                );
                let gu = mine
                    .iter()
                    .find(|x| x.1.starts_with("moe_grouped") && x.1.contains("_g1"))
                    .expect("gate_up");
                let dn = mine
                    .iter()
                    .find(|x| x.1.starts_with("moe_grouped") && x.1.contains("_g0"))
                    .expect("down");
                let silu = mine
                    .iter()
                    .find(|x| x.1.starts_with("moe_silu"))
                    .expect("silu");
                // grouped: 11 pointers, then (N, K, Pmax)
                let (gn, gk, dn_n, dn_k) = (
                    i32_at(&gu.2, 88),
                    i32_at(&gu.2, 92),
                    i32_at(&dn.2, 88),
                    i32_at(&dn.2, 92),
                );
                if xtp2 {
                    // EP's measured cells: tm 1 at <= 512 rows (the down half at DBK 64), the
                    // committed pf tile above
                    let pf = m > 512;
                    assert_eq!((e_lo, e_loc, sh), (0, 512, 1), "rank {r} m {m}");
                    assert_eq!(tm, if pf { 4 } else { 1 });
                    assert_eq!((gn, gk, dn_n, dn_k), (640, 2560, 2560, 320));
                    if pf {
                        assert_eq!(gu.1, "moe_grouped_w4a8_pf_tm4_g1");
                        assert_eq!(dn.1, "moe_grouped_w4a8_pf_tm4_g0");
                    } else {
                        assert_eq!(gu.1, "moe_grouped_w4a8_bk128_ks1_tm1_g1_p1");
                        assert_eq!(dn.1, "moe_grouped_w4a8_bk64_ks1_tm1_g0_p1");
                    }
                    assert_eq!(i32_at(&silu.2, 24), 320);
                } else {
                    assert_eq!(
                        (e_lo, e_loc, sh),
                        (256 * r as i32, 256, i32::from(r == 0)),
                        "the EP control"
                    );
                    assert_eq!((gn, gk, dn_n, dn_k), (1280, 2560, 2560, 640));
                    assert_eq!(i32_at(&silu.2, 24), 640);
                }
            }
        }
    }

    /// The shared expert's split: each half's `down.wref` is the whole row's, the halves join back
    /// to the whole (checked inside `slice_half`); the control (a moved byte in one half) does not.
    #[test]
    fn the_shared_expert_halves_join_back_to_the_whole() {
        let (ss, payload, _) = record_layout(xsplit::HIDDEN, xsplit::INTER);
        let whole: Vec<u8> = (0..payload)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
            .collect();
        let (h0, ds) = slice_half(&whole, 0).unwrap();
        let (h1, _) = slice_half(&whole, 1).unwrap();
        assert_ne!(h0, h1);
        let w = |b: &[u8], s: (u64, u64)| b[s.0 as usize..(s.0 + s.1) as usize].to_vec();
        assert_eq!(w(&h0, ds[5]), w(&whole, ss[5]));
        assert_eq!(w(&h1, ds[5]), w(&whole, ss[5]));
        let mut bad = h1.clone();
        bad[ds[0].0 as usize] ^= 1;
        let j = xsplit::join_payload([&h0, &bad], &ds, &ss, payload).unwrap();
        assert_ne!(j, whole, "the control must not join back");
    }
}
