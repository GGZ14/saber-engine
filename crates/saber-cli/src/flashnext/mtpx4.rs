//! FN110: the MTP drafter's experts in the expert tier's NATIVE record form -- MXFP4 RTN of the
//! checkpoint's bf16 MTP experts (routed + the shared expert), FN2's six sub-records an expert
//! (`gate_up.{w,ws,wref}`, `down.{w,ws,wref}`) -- as an ADDITIONAL pack file per rank,
//! `rank<r>/mtp_x4.{pack,json}`, built by `tools/pack/build_mtp_x4.py` (the host gate's `mx4s`
//! form, `tools/capture/fn110/mtp_accept.py`). Loaded only under `--fn-mtp-experts slab4`: ONE
//! VRAM slab a rank (the file's bytes = the ledger's `mtp.experts` class), the grouped tables at
//! three pointers an entry, read by the trunk's seven-launch W4A8 train.
//!
//! FN138 (`--fn-mtp-split tp2`, under `--fn-experts tp2`): the same files, SLICED AT LOAD by
//! FN80a's law ([`super::xtp2::slice_half`]: gate rows then up rows of the half, down K half,
//! `down.wref` the whole row's) -- every rank reads BOTH ranks' files and keeps ITS half of all
//! 512 routed experts + the shared expert (entry 512 on both ranks): 513 half records, equal bytes
//! on both ranks ([`half_slab_bytes`]), nothing written to the drive.

use anyhow::{ensure, Context};
use saber_hip::{HipThread, Stream};

use super::model::Owned;
use super::store::PackRecs;

/// The additional pack file's stem (per rank directory).
pub const STEM: &str = "mtp_x4";
/// Routed MTP experts per EP rank.
pub const ELOC: u32 = 256;
/// The six sub-records in grouped-table order (gate_up then down; w, ws, wref).
pub const SUBS: [&str; 6] = [
    "gate_up.w",
    "gate_up.ws",
    "gate_up.wref",
    "down.w",
    "down.ws",
    "down.wref",
];

/// The record stem of routed expert `e` (`None` = the shared expert).
#[must_use]
pub fn stem(e: Option<u32>) -> String {
    match e {
        Some(e) => format!("mtp.layer.0.expert.{e}.x4"),
        None => "mtp.layer.0.shared_expert.x4".to_string(),
    }
}

fn doc(p: &PackRecs, rank: u32) -> anyhow::Result<serde_json::Value> {
    let f = p.dir.join(format!("rank{rank}/{STEM}.json"));
    serde_json::from_str(&std::fs::read_to_string(&f).with_context(|| {
        format!(
            "{}: --fn-mtp-experts slab4 needs a pack that carries the mtp_x4 records (use the hosted pack)",
            f.display()
        )
    })?)
    .with_context(|| f.display().to_string())
}

/// The file's bytes on `rank` (the ledger's `mtp.experts` class under the slab4 arm).
pub fn rank_bytes(pack: &std::path::Path, rank: u32) -> anyhow::Result<u64> {
    let f = pack.join(format!("rank{rank}/{STEM}.json"));
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&f).with_context(|| {
            format!(
                "{}: --fn-mtp-experts slab4 needs a pack that carries the mtp_x4 records (use the hosted pack)",
                f.display()
            )
        })?)?;
    v["bytes"].as_u64().context("bytes")
}

/// Upload `rank`'s slab (every record == its sha256) and build the grouped tables: entry `i` (the
/// rank's local expert `256 rank + i`) at `3 i .. 3 i + 3`, the shared expert at entry 256 on rank
/// 0 (rank 1's stays 0, the trunk's EP convention). Returns `(gate_up, down)` host tables.
pub fn tables(
    p: &PackRecs,
    rank: u32,
    o: &mut Owned,
    hip: &HipThread,
    st: &Stream,
) -> anyhow::Result<(Vec<u64>, Vec<u64>)> {
    let d = doc(p, rank)?;
    let bytes = d["bytes"].as_u64().context("bytes")?;
    let mut off = std::collections::HashMap::new();
    for r in d["records"].as_array().context("records")? {
        off.insert(
            r["name"].as_str().context("name")?.to_string(),
            r["storage"]["data_off"].as_u64().context("data_off")?,
        );
    }
    let mut slab = vec![0u8; bytes as usize];
    let mut entries: Vec<(usize, String)> = (0..ELOC)
        .map(|i| (i as usize, stem(Some(rank * ELOC + i))))
        .collect();
    if rank == 0 {
        entries.push((ELOC as usize, stem(None)));
    }
    let mut rel = vec![[0u64; 6]; ELOC as usize + 1];
    for (i, s) in &entries {
        for (k, sub) in SUBS.iter().enumerate() {
            let name = format!("{s}.{sub}");
            let b = p.raw(rank, &name)?;
            let at = *off
                .get(&name)
                .with_context(|| format!("{name}: no record"))?;
            ensure!(
                at % 256 == 0 && at + b.len() as u64 <= bytes,
                "{name}: record at {at} + {} outside the 256-aligned slab of {bytes}",
                b.len()
            );
            slab[at as usize..at as usize + b.len()].copy_from_slice(&b);
            rel[*i][k] = at;
        }
    }
    let base = o.up(hip, st, &slab)?;
    let n = ELOC as usize + 1;
    let (mut gu, mut dn) = (vec![0u64; 3 * n], vec![0u64; 3 * n]);
    for (i, _) in &entries {
        for k in 0..3 {
            gu[3 * i + k] = base + rel[*i][k];
            dn[3 * i + k] = base + rel[*i][3 + k];
        }
    }
    Ok((gu, dn))
}

/// FN138: the half slab's entries (512 routed halves + the shared expert's half at 512).
pub const HALF_ENTRIES: usize = 2 * ELOC as usize + 1;

/// FN138: one half record's slab stride (the half payload, 256-aligned: every sub-record of
/// `record_layout(2560, 320)` starts 256-aligned).
#[must_use]
pub fn half_stride() -> u64 {
    let (_, hp, _) = super::xsplit::half_layout();
    hp.div_ceil(256) * 256
}

/// FN138: the half slab's bytes -- equal on both ranks (both hold the shared expert's half): the
/// `mtp.experts` class under `--fn-mtp-split tp2`.
#[must_use]
pub fn half_slab_bytes() -> u64 {
    HALF_ENTRIES as u64 * half_stride()
}

/// FN138: entry `i` of the half slab -> the expert (`None` = the shared expert at 512).
#[must_use]
pub fn half_entry(i: usize) -> Option<u32> {
    (i < 2 * ELOC as usize).then_some(i as u32)
}

/// FN138: MTP expert `e`'s six sub-records (from its EP home rank's file; the shared expert from
/// rank 0's) as ONE whole-layout payload (`record_layout(2560, 640)`).
pub fn whole(p: &PackRecs, e: Option<u32>) -> anyhow::Result<Vec<u8>> {
    use super::xsplit::{HIDDEN, INTER};
    let home = e.map_or(0, |e| e / ELOC);
    let (ss, payload, _) = saber_mem::xtier::record_layout(HIDDEN, INTER);
    let mut w = vec![0u8; payload as usize];
    for (k, sub) in SUBS.iter().enumerate() {
        let name = format!("{}.{sub}", stem(e));
        let b = p.raw(home, &name)?;
        ensure!(
            b.len() as u64 == ss[k].1,
            "{name}: {} B, the whole record law says {}",
            b.len(),
            ss[k].1
        );
        w[ss[k].0 as usize..(ss[k].0 + ss[k].1) as usize].copy_from_slice(&b);
    }
    Ok(w)
}

/// FN138: rank `rank`'s half slab from the whole records `get` returns (entry order
/// [`half_entry`]; each half checked by FN80a's join law) -> (slab bytes, the six sub-record
/// offsets of every entry relative to the slab).
pub fn half_slab(
    rank: u32,
    get: &(dyn Fn(Option<u32>) -> anyhow::Result<Vec<u8>> + Sync),
) -> anyhow::Result<(Vec<u8>, Vec<[u64; 6]>)> {
    ensure!(rank < 2, "the half slab is a TP=2 form");
    let hs = half_stride() as usize;
    let mut slab = vec![0u8; HALF_ENTRIES * hs];
    let mut ds0 = None;
    let threads = 8usize;
    let per = HALF_ENTRIES.div_ceil(threads);
    let res: Vec<anyhow::Result<Option<super::xtp2::SubLayout>>> = std::thread::scope(|sc| {
        let hs_ = hs;
        let mut hands = Vec::new();
        for (t, chunk) in slab.chunks_mut(per * hs_).enumerate() {
            hands.push(
                sc.spawn(move || -> anyhow::Result<Option<super::xtp2::SubLayout>> {
                    let mut ds = None;
                    for (j, dst) in chunk.chunks_mut(hs_).enumerate() {
                        let i = t * per + j;
                        let w = get(half_entry(i))?;
                        let (h, d) = super::xtp2::slice_half(&w, rank)
                            .with_context(|| format!("MTP half slab entry {i}"))?;
                        dst[..h.len()].copy_from_slice(&h);
                        ds = Some(d);
                    }
                    Ok(ds)
                }),
            );
        }
        hands
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err(anyhow::anyhow!("slicer panicked")))
            })
            .collect()
    });
    for r in res {
        if let Some(d) = r? {
            ds0 = Some(d);
        }
    }
    let ds = ds0.context("no entries")?;
    let rel = (0..HALF_ENTRIES)
        .map(|i| {
            let mut v = [0u64; 6];
            for (k, x) in v.iter_mut().enumerate() {
                *x = (i * hs) as u64 + ds[k].0;
            }
            v
        })
        .collect();
    Ok((slab, rel))
}

/// FN138: upload rank `rank`'s HALF slab (`--fn-mtp-split tp2`; both ranks' files read, every
/// record == its sha256, sliced by FN80a's law) and build the grouped tables at three pointers an
/// entry, 513 entries (the shared half at 512 on BOTH ranks: the target tp2 form's convention).
pub fn tables_tp2(
    p: &PackRecs,
    rank: u32,
    o: &mut Owned,
    hip: &HipThread,
    st: &Stream,
) -> anyhow::Result<(Vec<u64>, Vec<u64>)> {
    for r in 0..2 {
        doc(p, r)?;
    }
    let (slab, rel) = half_slab(rank, &|e| whole(p, e))?;
    ensure!(
        slab.len() as u64 == half_slab_bytes(),
        "the half slab's bytes"
    );
    let base = o.up(hip, st, &slab)?;
    let (mut gu, mut dn) = (vec![0u64; 3 * HALF_ENTRIES], vec![0u64; 3 * HALF_ENTRIES]);
    for (i, r) in rel.iter().enumerate() {
        for k in 0..3 {
            gu[3 * i + k] = base + r[k];
            dn[3 * i + k] = base + r[3 + k];
        }
    }
    Ok((gu, dn))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FN138: the half slab's law on synthetic whole records -- every entry's two halves (rank 0's
    /// slab, rank 1's slab) join back to that expert's whole record byte for byte (FN80a's
    /// `join_payload`), the entries are in expert order with the shared expert at 512, the bytes
    /// are equal on both ranks; controls: a neighbour's half and the swapped halves do not join.
    #[test]
    fn fn138_the_half_slabs_join_back_to_the_whole_mtp_records() {
        use super::super::xsplit::{half_layout, join_payload, HIDDEN, INTER};
        let (ss, payload, _) = saber_mem::xtier::record_layout(HIDDEN, INTER);
        let (ds, hp, _) = half_layout();
        let rec = |e: Option<u32>| -> anyhow::Result<Vec<u8>> {
            let id = u64::from(e.unwrap_or(999));
            Ok((0..payload)
                .map(|o| saber_mem::xtier::synthetic_byte(0, id, o))
                .collect())
        };
        let (s0, r0) = half_slab(0, &rec).unwrap();
        let (s1, r1) = half_slab(1, &rec).unwrap();
        assert_eq!(s0.len() as u64, half_slab_bytes());
        assert_eq!(s0.len(), s1.len(), "equal bytes on both ranks");
        assert_eq!(r0, r1, "the same table offsets on both ranks");
        assert_eq!(HALF_ENTRIES, 513);
        assert_eq!(half_entry(512), None);
        assert_eq!(half_entry(300), Some(300));
        let hs = half_stride() as usize;
        let half = |s: &[u8], i: usize| s[i * hs..i * hs + hp as usize].to_vec();
        for i in [0usize, 1, 255, 256, 257, 511, 512] {
            assert_eq!(r0[i][0], (i * hs) as u64 + ds[0].0);
            assert_eq!(r0[i][3], (i * hs) as u64 + ds[3].0);
            let j = join_payload([&half(&s0, i), &half(&s1, i)], &ds, &ss, payload);
            assert_eq!(j, Some(rec(half_entry(i)).unwrap()), "entry {i}");
            // controls: the neighbour's half 1, the halves swapped
            let n = (i + 1) % HALF_ENTRIES;
            let c = join_payload([&half(&s0, i), &half(&s1, n)], &ds, &ss, payload);
            assert_ne!(
                c,
                Some(rec(half_entry(i)).unwrap()),
                "neighbour control {i}"
            );
            let c = join_payload([&half(&s1, i), &half(&s0, i)], &ds, &ss, payload);
            assert_ne!(c, Some(rec(half_entry(i)).unwrap()), "swap control {i}");
        }
        // the bill: below EP rank 0's slab (257 whole records) and equal on both ranks
        assert_eq!(half_slab_bytes(), 513 * 1_308_928);
    }

    #[test]
    fn the_x4_record_names_follow_the_pack_builder() {
        assert_eq!(stem(Some(300)), "mtp.layer.0.expert.300.x4");
        assert_eq!(stem(None), "mtp.layer.0.shared_expert.x4");
        assert_eq!(SUBS[3], "down.w");
    }
}
