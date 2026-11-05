//! FN80a: the TP=2 EXPERT SPLIT of a Flash-Next pack (`saber flashnext-pack --experts tp2`).
//!
//! Today's expert tier is EP=2: rank r holds experts `[256r, 256r + 256)` whole (one 2,617,344-B
//! record each, docs/EXEC.md §10). The TP=2 form gives EVERY rank one half of EVERY expert along
//! the intermediate dimension (the standard TP MLP split, no reduction between gate_up and down):
//!
//! * gate_up (`w` `[1280, 2560]` WPERM, gate rows 0..640 then up rows 640..1280): half `h` keeps
//!   gate rows `[320h, 320h + 320)` then up rows `[640 + 320h, ..)` -> `[640, 2560]`, gate first
//!   (the silu producer's `(gu[j], gu[n + j])` law at n = 320). WPERM n-tiles are 16 rows and
//!   320 = 20 tiles, so each half is two CONTIGUOUS runs of whole tiles; `ws` `[80, 1280]` keeps
//!   the same columns of every block row; `wref` the same rows.
//! * down (`w` `[2560, 640]` WPERM, `ws` `[20, 2560]`): half `h` keeps K `[320h, 320h + 320)` =
//!   e8m0 block rows `[10h, 10h + 10)` (320 = 10 x 32: the cut is ON a block boundary) = WPERM
//!   k-steps `[20h, 20h + 20)` of every n-tile (a k-step is 32 u32 words = 128 B contiguous).
//!   **`wref` stays the WHOLE row's max** (2,560 B copied into both halves): the card folds
//!   `d = clamp(wref - ws, 0, 15)`, so every folded weight byte and the epilogue's `rf` of a half
//!   are EXACTLY the whole expert's (a per-half max would move the fold and the flush-to-zero).
//!
//! A half record is `record_layout(2560, 320)`: 1,308,928 B of payload, stride 1,310,720 B (320
//! pages) -- 2 x 1,310,720 = 2,621,440 vs the whole 2,617,344 (+0.16 %: the second `down.wref`
//! copy + page rounding). Each rank's file `rank<r>/experts.tp2.xtier` holds 48 x 512 records,
//! record `L * 512 + e`, header marker `split_parts 2, split_index r` (bytes 248..256), first
//! expert 0, experts 512. The variant directory links every other file of the source pack
//! (`experts.xtier` too: the EP path boots the variant unchanged = the rollback) and adds
//! `experts_tp2.json` (the split's manifest).
//!
//! THE BYTE GATE ([`check`]): every expert of every layer is re-read from BOTH written files
//! (each half against its own table digest) and joined ([`join_payload`]); the join must equal
//! the source record's payload byte for byte (the source read against its table digest). The
//! controls must NOT equal it: the halves swapped, and half 1 taken from the next expert.

use std::path::{Path, PathBuf};

use anyhow::{ensure, Context};
use saber_mem::dio::{self, AlignedBuf, OpenMode};
use saber_mem::xtier::{
    header_block, record_digest, record_layout, table_digest, XtierFile, XtierHeader,
};
use serde_json::json;

/// Model hidden size (gate_up K, down N).
pub const HIDDEN: u64 = 2560;
/// The routed expert's intermediate size (whole).
pub const INTER: u64 = 640;
/// Parts of the split.
pub const PARTS: u64 = 2;
/// The intermediate rows (gate_up) / K (down) of one part.
pub const INTER_H: u64 = INTER / PARTS;
/// Layers.
pub const LAYERS: u64 = 48;
/// Experts a layer.
pub const EXPERTS: u64 = 512;
/// The split file of rank r.
pub const FILE: &str = "experts.tp2.xtier";

// The law's preconditions, asserted at compile time: whole 16-row WPERM n-tiles, whole 32-k e8m0
// blocks, whole 16-k WPERM k-steps.
const _: () = assert!(INTER_H % 16 == 0 && INTER_H % 32 == 0 && INTER % PARTS == 0);

/// Bytes of one WPERM n-tile (16 rows) at K: (K / 16) k-steps x 32 u32 words.
const fn tile_bytes(k: u64) -> u64 {
    (k / 16) * 128
}

fn sub(b: &[u8], s: (u64, u64)) -> &[u8] {
    &b[s.0 as usize..(s.0 + s.1) as usize]
}

/// gate_up row `j` of half `h` (`j < 2 * INTER_H`) -> the whole record's row.
#[must_use]
pub const fn gu_row(h: u64, j: u64) -> u64 {
    if j < INTER_H {
        h * INTER_H + j
    } else {
        INTER + h * INTER_H + (j - INTER_H)
    }
}

/// Half `h`'s payload from the whole record's payload `src` (layout `ss`) into `dst` (layout
/// `ds`, zeroed by the caller where it pads).
pub fn slice_payload(
    src: &[u8],
    ss: &[(u64, u64); 6],
    h: u64,
    dst: &mut [u8],
    ds: &[(u64, u64); 6],
) {
    assert!(h < PARTS);
    // gate_up w: two runs of whole n-tiles
    let tb = tile_bytes(HIDDEN) as usize;
    let (gw, dw) = (sub(src, ss[0]), ds[0].0 as usize);
    let run = (INTER_H / 16) as usize * tb;
    for (i, row0) in [h * INTER_H, INTER + h * INTER_H].into_iter().enumerate() {
        let s0 = (row0 / 16) as usize * tb;
        dst[dw + i * run..dw + (i + 1) * run].copy_from_slice(&gw[s0..s0 + run]);
    }
    // gate_up ws [K/32][2 * INTER] -> [K/32][2 * INTER_H]; wref [2 * INTER] -> [2 * INTER_H]
    let (gs, dsx) = (sub(src, ss[1]), ds[1].0 as usize);
    let (n_w, n_h) = ((2 * INTER) as usize, (2 * INTER_H) as usize);
    for b in 0..(HIDDEN / 32) as usize {
        for j in 0..n_h {
            dst[dsx + b * n_h + j] = gs[b * n_w + gu_row(h, j as u64) as usize];
        }
    }
    let (gr, dr) = (sub(src, ss[2]), ds[2].0 as usize);
    for j in 0..n_h {
        dst[dr + j] = gr[gu_row(h, j as u64) as usize];
    }
    // down w: per n-tile, the half's k-steps (contiguous 128-B steps)
    let (dwv, ddw) = (sub(src, ss[3]), ds[3].0 as usize);
    let (tw, th) = (tile_bytes(INTER) as usize, tile_bytes(INTER_H) as usize);
    let k0 = (h * INTER_H / 16) as usize * 128;
    for nt in 0..(HIDDEN / 16) as usize {
        dst[ddw + nt * th..ddw + (nt + 1) * th]
            .copy_from_slice(&dwv[nt * tw + k0..nt * tw + k0 + th]);
    }
    // down ws [INTER/32][HIDDEN]: the half's block rows, contiguous
    let (dsv, dds) = (sub(src, ss[4]), ds[4].0 as usize);
    let rb = (INTER_H / 32 * HIDDEN) as usize;
    let b0 = (h * INTER_H / 32 * HIDDEN) as usize;
    dst[dds..dds + rb].copy_from_slice(&dsv[b0..b0 + rb]);
    // down wref: the WHOLE row's max, copied
    let (drv, ddr) = (sub(src, ss[5]), ds[5].0 as usize);
    dst[ddr..ddr + HIDDEN as usize].copy_from_slice(drv);
}

/// The whole record's payload (layout `ss`, `payload` bytes) rebuilt from the two halves (layout
/// `ds`). Every whole byte is written by exactly one half except `down.wref` (both carry it): the
/// join takes half 0's and REQUIRES half 1's to be equal (`None` otherwise).
#[must_use]
pub fn join_payload(
    halves: [&[u8]; 2],
    ds: &[(u64, u64); 6],
    ss: &[(u64, u64); 6],
    payload: u64,
) -> Option<Vec<u8>> {
    if sub(halves[0], ds[5]) != sub(halves[1], ds[5]) {
        return None;
    }
    let mut w = vec![0u8; payload as usize];
    let tb = tile_bytes(HIDDEN) as usize;
    let run = (INTER_H / 16) as usize * tb;
    let (n_w, n_h) = ((2 * INTER) as usize, (2 * INTER_H) as usize);
    let (tw, th) = (tile_bytes(INTER) as usize, tile_bytes(INTER_H) as usize);
    let rb = (INTER_H / 32 * HIDDEN) as usize;
    for (h, src) in halves.iter().enumerate() {
        let h = h as u64;
        let gw = sub(src, ds[0]);
        for (i, row0) in [h * INTER_H, INTER + h * INTER_H].into_iter().enumerate() {
            let s0 = ss[0].0 as usize + (row0 / 16) as usize * tb;
            w[s0..s0 + run].copy_from_slice(&gw[i * run..(i + 1) * run]);
        }
        let gs = sub(src, ds[1]);
        for b in 0..(HIDDEN / 32) as usize {
            for j in 0..n_h {
                w[ss[1].0 as usize + b * n_w + gu_row(h, j as u64) as usize] = gs[b * n_h + j];
            }
        }
        let gr = sub(src, ds[2]);
        for j in 0..n_h {
            w[ss[2].0 as usize + gu_row(h, j as u64) as usize] = gr[j];
        }
        let dwv = sub(src, ds[3]);
        let k0 = (h * INTER_H / 16) as usize * 128;
        for nt in 0..(HIDDEN / 16) as usize {
            let o = ss[3].0 as usize + nt * tw + k0;
            w[o..o + th].copy_from_slice(&dwv[nt * th..(nt + 1) * th]);
        }
        let dsv = sub(src, ds[4]);
        let b0 = ss[4].0 as usize + (h * INTER_H / 32 * HIDDEN) as usize;
        w[b0..b0 + rb].copy_from_slice(dsv);
    }
    let r = ss[5].0 as usize;
    w[r..r + HIDDEN as usize].copy_from_slice(sub(halves[0], ds[5]));
    Some(w)
}

/// The half-record geometry (`record_layout(2560, 320)`): (sub, payload, stride).
#[must_use]
pub fn half_layout() -> ([(u64, u64); 6], u64, u64) {
    record_layout(HIDDEN, INTER_H)
}

fn check_source(h: &XtierHeader, r: u32) -> anyhow::Result<()> {
    let (sub, payload, stride) = record_layout(HIDDEN, INTER);
    ensure!(
        h.split_parts == 0
            && h.hidden as u64 == HIDDEN
            && h.inter as u64 == INTER
            && h.layers as u64 == LAYERS
            && h.experts as u64 * 2 == EXPERTS
            && h.first_expert == r * h.experts
            && h.sub == sub
            && h.payload == payload
            && h.stride == stride,
        "rank {r}: the source tier is not FN2's EP=2 whole-expert law ({h:?})"
    );
    Ok(())
}

pub fn half_header(r: u32, plan_sha256: [u8; 32]) -> XtierHeader {
    let (sub, payload, stride) = half_layout();
    let records = LAYERS * EXPERTS;
    let table_off = 4096;
    let table_bytes = records * 32;
    let data_off = dio::align_up(table_off + table_bytes);
    XtierHeader {
        rank: r,
        layers: LAYERS as u32,
        first_expert: 0,
        experts: EXPERTS as u32,
        stride,
        payload,
        sub,
        hidden: HIDDEN as u32,
        inter: INTER_H as u32,
        table_off,
        table_bytes,
        data_off,
        file_bytes: data_off + records * stride,
        table_sha256: [0; 32],
        plan_sha256,
        split_parts: PARTS as u32,
        split_index: r,
        format: saber_mem::xtier::XtierFormat::Mxfp4E8m0,
    }
}

/// The source record (whole expert `e` of layer `l`) into `buf`, checked against its table.
fn read_source(src: &[XtierFile; 2], l: u64, e: u64, buf: &mut AlignedBuf) -> anyhow::Result<()> {
    let owner = (e / (EXPERTS / 2)) as usize;
    let h = src[owner].header();
    let i = h
        .index(l as u32, e as u32)
        .context("the source tier does not own the expert")?;
    src[owner].read_record(i, buf)?;
    ensure!(
        src[owner].verify(i, buf),
        "source rank {owner} record {i} fails its sha256"
    );
    Ok(())
}

/// What a build wrote.
#[derive(Debug)]
pub struct Built {
    pub out: PathBuf,
    pub records: u64,
    pub bytes: [u64; 2],
}

/// Build the variant `out` from the pack `src` (link every file, then write both ranks'
/// `experts.tp2.xtier` + `experts_tp2.json`). The in-build gate joins every pair of halves back
/// against the source payload before it is written. Refuses an existing split file.
pub fn build(src: &Path, out: &Path) -> anyhow::Result<Built> {
    let (src, out) = super::trunk4::link_variant(src, out)?;
    let open = |r: u32| -> anyhow::Result<XtierFile> {
        let x = XtierFile::open(
            &src.join(format!("rank{r}/experts.xtier")),
            OpenMode::Direct,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;
        check_source(x.header(), r)?;
        Ok(x)
    };
    let xs = [open(0)?, open(1)?];
    let (ss, spay) = (xs[0].header().sub, xs[0].header().payload);
    let (ds, hpay, hstride) = half_layout();
    let hdr: Vec<XtierHeader> = (0..2)
        .map(|r| half_header(r, xs[0].header().plan_sha256))
        .collect();
    let mut files = Vec::new();
    for r in 0..2u32 {
        let p = out.join(format!("rank{r}/{FILE}"));
        ensure!(
            !p.exists(),
            "{} exists: the split is written once",
            p.display()
        );
        let tmp = out.join(format!("rank{r}/{FILE}.part"));
        let _ = std::fs::remove_file(&tmp);
        let f = dio::open_rw_create(&tmp, OpenMode::Direct)?;
        dio::fallocate(&f, hdr[r as usize].file_bytes)?;
        files.push((f, tmp, p));
    }
    let mut tables = [Vec::new(), Vec::new()];
    let mut whole = AlignedBuf::zeroed(xs[0].header().stride as usize);
    let mut half = [
        AlignedBuf::zeroed(hstride as usize),
        AlignedBuf::zeroed(hstride as usize),
    ];
    let t0 = std::time::Instant::now();
    for l in 0..LAYERS {
        for e in 0..EXPERTS {
            read_source(&xs, l, e, &mut whole)?;
            for (h, b) in half.iter_mut().enumerate() {
                b.fill(0);
                slice_payload(
                    &whole[..spay as usize],
                    &ss,
                    h as u64,
                    &mut b[..hpay as usize],
                    &ds,
                );
            }
            let j = join_payload([&half[0][..], &half[1][..]], &ds, &ss, spay)
                .context("down.wref differs between the halves")?;
            ensure!(
                j[..] == whole[..spay as usize],
                "layer {l} expert {e}: the halves do not rebuild the source payload"
            );
            for r in 0..2 {
                let i = l * EXPERTS + e;
                dio::pwritev_exact(&files[r].0, &half[r], hdr[r].data_off + i * hstride)?;
                tables[r].extend_from_slice(&record_digest(&half[r]));
            }
        }
        if l % 8 == 7 {
            eprintln!(
                "xsplit: layer {l} done, {:.0} s",
                t0.elapsed().as_secs_f64()
            );
        }
    }
    let mut bytes = [0u64; 2];
    let mut meta = Vec::new();
    for r in 0..2 {
        let mut h = hdr[r].clone();
        h.table_sha256 = table_digest(&tables[r]);
        let mut hb = AlignedBuf::zeroed(4096);
        hb.copy_from_slice(&header_block(&h));
        dio::pwritev_exact(&files[r].0, &hb, 0)?;
        let mut tb = AlignedBuf::zeroed(dio::align_up(h.table_bytes) as usize);
        tb[..tables[r].len()].copy_from_slice(&tables[r]);
        dio::pwritev_exact(&files[r].0, &tb, h.table_off)?;
        files[r].0.sync_all()?;
        std::fs::rename(&files[r].1, &files[r].2)?;
        bytes[r] = h.file_bytes;
        meta.push(json!({
            "rank": r, "file": format!("rank{r}/{FILE}"), "split_index": r,
            "first_expert": 0, "experts": EXPERTS, "layers": LAYERS, "records": LAYERS * EXPERTS,
            "record": {"sub": h.sub, "payload": h.payload, "stride": h.stride},
            "data_off": h.data_off, "file_bytes": h.file_bytes,
            "table_sha256": hex::encode(h.table_sha256),
        }));
    }
    let doc = json!({
        "producer": "saber flashnext-pack --experts tp2 (FN80a)",
        "source": src.display().to_string(),
        "source_tables_sha256": [hex::encode(xs[0].header().table_sha256), hex::encode(xs[1].header().table_sha256)],
        "layout": {
            "experts": "tp2",
            "law": "every rank holds part r of EVERY expert: gate_up rows gate [320r, 320r+320) then up [640+320r, ..) (gate first); down K [320r, 320r+320) = e8m0 block rows [10r, 10r+10) = WPERM k-steps [20r, 20r+20); down.wref = the WHOLE row's max in both parts (the card's fold is the whole expert's); gate_up.wref = the rows' own (whole-row) values",
            "record": "i = layer * 512 + expert (global id)",
            "header_marker": "split_parts 2 / split_index r at header bytes 248..256",
        },
        "tiers": meta,
        "built_seconds": t0.elapsed().as_secs_f64(),
    });
    std::fs::write(
        out.join("experts_tp2.json"),
        serde_json::to_string_pretty(&doc)?,
    )?;
    Ok(Built {
        out,
        records: LAYERS * EXPERTS,
        bytes,
    })
}

/// The byte gate's result.
#[derive(Debug, Default)]
pub struct Checked {
    pub experts: u64,
    pub rebuilt_equal: u64,
    pub control_swap_equal: u64,
    pub control_neighbour_equal: u64,
}

/// THE BYTE GATE: re-read both written split files (each half against its own table) and rebuild
/// every expert of every layer; the join must equal the source record (against its table). The
/// controls (halves swapped; half 1 of the next expert) must never equal it.
pub fn check(src: &Path, out: &Path) -> anyhow::Result<Checked> {
    let open = |p: PathBuf| {
        XtierFile::open(&p, OpenMode::Direct).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))
    };
    let xs = [
        open(src.join("rank0/experts.xtier"))?,
        open(src.join("rank1/experts.xtier"))?,
    ];
    for (r, x) in xs.iter().enumerate() {
        check_source(x.header(), r as u32)?;
    }
    let ys = [
        open(out.join(format!("rank0/{FILE}")))?,
        open(out.join(format!("rank1/{FILE}")))?,
    ];
    let (ds, hpay, hstride) = half_layout();
    for (r, y) in ys.iter().enumerate() {
        let h = y.header();
        ensure!(
            h.split_parts == PARTS as u32
                && h.split_index == r as u32
                && h.sub == ds
                && h.payload == hpay
                && h.stride == hstride
                && h.experts as u64 == EXPERTS
                && h.first_expert == 0,
            "rank {r}: the split file's header is not the FN80a law ({h:?})"
        );
    }
    let (ss, spay) = (xs[0].header().sub, xs[0].header().payload);
    let mut whole = AlignedBuf::zeroed(xs[0].header().stride as usize);
    let mut hb = [
        AlignedBuf::zeroed(hstride as usize),
        AlignedBuf::zeroed(hstride as usize),
        AlignedBuf::zeroed(hstride as usize),
    ];
    let mut c = Checked::default();
    let t0 = std::time::Instant::now();
    for l in 0..LAYERS {
        for e in 0..EXPERTS {
            read_source(&xs, l, e, &mut whole)?;
            let i = l * EXPERTS + e;
            let inext = l * EXPERTS + (e + 1) % EXPERTS;
            for (k, (r, idx)) in [(0usize, i), (1, i), (1, inext)].into_iter().enumerate() {
                ys[r].read_record(idx, &mut hb[k])?;
                ensure!(
                    ys[r].verify(idx, &hb[k]),
                    "split rank {r} record {idx} fails its sha256"
                );
            }
            let w = &whole[..spay as usize];
            c.experts += 1;
            if join_payload([&hb[0][..], &hb[1][..]], &ds, &ss, spay).as_deref() == Some(w) {
                c.rebuilt_equal += 1;
            }
            if join_payload([&hb[1][..], &hb[0][..]], &ds, &ss, spay).as_deref() == Some(w) {
                c.control_swap_equal += 1;
            }
            if join_payload([&hb[0][..], &hb[2][..]], &ds, &ss, spay).as_deref() == Some(w) {
                c.control_neighbour_equal += 1;
            }
        }
        if l % 8 == 7 {
            eprintln!(
                "xsplit check: layer {l} done, {:.0} s",
                t0.elapsed().as_secs_f64()
            );
        }
    }
    println!(
        "fn80a byte gate: {} experts, rebuilt == source {} ({} differ); controls equal: swapped {}, neighbour {} ({:.0} s)",
        c.experts,
        c.rebuilt_equal,
        c.experts - c.rebuilt_equal,
        c.control_swap_equal,
        c.control_neighbour_equal,
        t0.elapsed().as_secs_f64()
    );
    ensure!(
        c.rebuilt_equal == c.experts && c.control_swap_equal == 0 && c.control_neighbour_equal == 0,
        "FN80a byte gate FAILED"
    );
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use saber_cpux::MxGemm;

    fn whole_record(seed: u64) -> (Vec<u8>, [(u64, u64); 6], u64) {
        let (ss, pay, _) = record_layout(HIDDEN, INTER);
        let mut v = vec![0u8; pay as usize];
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        for b in v.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *b = (x >> 24) as u8;
        }
        // e8m0 / wref bytes in a realistic band so `shift` exercises 0..15 and the clamp
        for i in [1usize, 2, 4, 5] {
            for b in &mut v[ss[i].0 as usize..(ss[i].0 + ss[i].1) as usize] {
                *b = 112 + (*b % 12);
            }
        }
        (v, ss, pay)
    }

    fn gemm<'a>(b: &'a [u8], s: &[(u64, u64); 6], i0: usize, n: u64, k: u64) -> MxGemm<'a> {
        MxGemm {
            w: sub(b, s[i0]),
            ws: sub(b, s[i0 + 1]),
            wref: sub(b, s[i0 + 2]),
            n: n as usize,
            k: k as usize,
        }
    }

    #[test]
    fn fn80a_the_halves_rebuild_the_expert_and_read_as_its_rows_and_k_slices() {
        let (w, ss, pay) = whole_record(7);
        let (ds, hpay, hstride) = half_layout();
        assert_eq!((hpay, hstride), (1_308_928, 1_310_720));
        assert_eq!(hstride % 4096, 0);
        let halves: Vec<Vec<u8>> = (0..2)
            .map(|h| {
                let mut b = vec![0u8; hpay as usize];
                slice_payload(&w, &ss, h, &mut b, &ds);
                b
            })
            .collect();
        assert_eq!(
            join_payload([&halves[0], &halves[1]], &ds, &ss, pay).unwrap(),
            w
        );
        assert_ne!(
            join_payload([&halves[1], &halves[0]], &ds, &ss, pay).unwrap(),
            w
        );
        // a neighbour expert's half: no rebuild (down.wref differs -> None, or bytes differ)
        let (w2, _, _) = whole_record(8);
        let mut n1 = vec![0u8; hpay as usize];
        slice_payload(&w2, &ss, 1, &mut n1, &ds);
        assert_ne!(
            join_payload([&halves[0], &n1], &ds, &ss, pay).as_deref(),
            Some(&w[..])
        );
        // SEMANTICS through the card's own accessors (saber-cpux's WPERM / fold law):
        let gu = gemm(&w, &ss, 0, 2 * INTER, HIDDEN);
        let dn = gemm(&w, &ss, 3, HIDDEN, INTER);
        for (h, hb) in halves.iter().enumerate() {
            let h = h as u64;
            let gh = gemm(hb, &ds, 0, 2 * INTER_H, HIDDEN);
            let dh = gemm(hb, &ds, 3, HIDDEN, INTER_H);
            for j in (0..2 * INTER_H as usize).step_by(7) {
                let n = gu_row(h, j as u64) as usize;
                assert_eq!(gh.rf(j), gu.rf(n));
                for k in (0..HIDDEN as usize).step_by(13) {
                    assert_eq!(
                        gh.nibble(j, k),
                        gu.nibble(n, k),
                        "gate_up h{h} row {j} k {k}"
                    );
                    assert_eq!(gh.shift(j, k / 32), gu.shift(n, k / 32));
                }
            }
            for n in (0..HIDDEN as usize).step_by(11) {
                assert_eq!(dh.rf(n), dn.rf(n), "down rf = the WHOLE row's");
                for k in 0..INTER_H as usize {
                    let kw = (h * INTER_H) as usize + k;
                    assert_eq!(dh.nibble(n, k), dn.nibble(n, kw), "down h{h} n {n} k {k}");
                    assert_eq!(
                        dh.shift(n, k / 32),
                        dn.shift(n, kw / 32),
                        "the fold is the whole's"
                    );
                }
            }
        }
    }

    #[test]
    fn fn80a_the_split_header_marker_round_trips_and_the_law_refuses_a_one_part_split() {
        let mut h = half_header(1, [3; 32]);
        h.table_sha256 = [9; 32];
        let b = header_block(&h);
        let p = XtierHeader::parse("t", &b).unwrap();
        assert_eq!(p, h);
        assert_eq!((p.split_parts, p.split_index, p.inter), (2, 1, 320));
        let mut bad = b.clone();
        bad[248] = 1;
        assert!(
            XtierHeader::parse("t", &bad).is_err(),
            "1 part is not a split"
        );
        let mut bad = b;
        bad[252] = 2;
        assert!(XtierHeader::parse("t", &bad).is_err(), "index 2 of 2");
    }
}
