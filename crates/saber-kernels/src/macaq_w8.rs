//! MACaQ (2026-11-11): the 27B form-c W8 promotions' GEMM (`--macaq-w8`, pack-driven).
//!
//! A promoted out_proj / o_proj / down_proj is served W8A8: e4m3 weights preshuffled to `[N/16, K*16]`
//! (pack layout 7) with one f32 scale a 128 x 128 block (layout 8), in the SAME basis as the MACaQ-Hadamard
//! producer that feeds it. Its activation is the W4 path's own: row-major e4m3 `A [M, K]` + ONE f32 scale a
//! row (the producers' per-token law). The bodies are `hip/macaq_w8.hip` (`tools/gen_macaq_w8pt.py`): the
//! shipped Flash-Next W8 bodies with the activation scale read per row. The band by rows is Flash-Next's
//! (`flashnext::exec::T8_TILES` + the Flash-Next <= 16-row cells): <= 16 rows the 64-column tile (the //! register ring when `ring`), 17..32 `d2w4`, 33..64 `d4w4`, past 64 the prefill tile `p4w8`.
//!
//! No split-K, no workspace, one launch: every output row's bytes depend on its own row only (the k walk is
//! fixed), so a promoted GEMM has no class across M and none across the decode buckets.

use crate::abi;
use crate::rules::{ArgValue, LaunchDescriptor, Slot};
use crate::{Error, Result};

/// The code object.
pub const PLUGIN: &str = "macaq_w8";
/// The op name descriptors carry.
pub const OP: &str = "gemm_w8a8_pt";
/// The <= 16-row cell (the 64-column tile).
pub const N64: &str = "macaq_w8pt_n64_ldc";
/// The <= 16-row register ring (the register ring's).
pub const N64_PF4: &str = "macaq_w8pt_n64_pf4_ldc";
/// Whether the <= 16-row cell is the register ring (where it takes the shape) -- decided by the
/// oracle hold's timing, data, not a ladder.
pub const RING: bool = true;
/// The ring's k-tile cap (`KTMAX` in the body) and its prefetch depth.
pub const RING_KTMAX: u32 = 160;
pub const RING_PF: u32 = 4;
/// `(max rows, symbol, rows a tile, columns a tile, rows-fastest grid, threads)` past 16 rows.
pub const TILES: &[(u32, &str, u32, u32, bool, u32)] = &[
    (32, "macaq_w8pt_d2w4_ldc", 32, 64, false, 128),
    (64, "macaq_w8pt_d4w4_ldc", 64, 64, false, 128),
    (u32::MAX, "macaq_w8pt_p4w8_ldc", 64, 128, true, 256),
];
/// Every symbol the code object carries (the prebuilt list, the oracle).
pub const SYMBOLS: &[&str] = &[
    N64,
    N64_PF4,
    "macaq_w8pt_d2w4_ldc",
    "macaq_w8pt_d4w4_ldc",
    "macaq_w8pt_p4w8_ldc",
];

/// Whether the ring takes `(m, k)`: <= 16 rows, a whole number of PF-tile groups, under the cap.
#[must_use]
pub fn ring_takes(m: u32, k: u32) -> bool {
    let kt = k / 128;
    m <= 16 && k % 128 == 0 && kt <= RING_KTMAX && kt % RING_PF == 0
}

/// The (symbol, grid, threads) a promoted GEMM of `m` rows runs.
///
/// # Errors
/// A shape the bodies were not written for: N or K not a multiple of 128 (the block scales and the
/// preshuffled fragments), or an empty launch.
pub fn cell(m: u32, n: u32, k: u32, ring: bool) -> Result<(&'static str, [u32; 3], u32)> {
    if m == 0 || n == 0 || k == 0 {
        return Err(Error::Rule(format!(
            "{OP}: {m} x {n} x {k} has nothing to do; an empty launch is a caller bug"
        )));
    }
    if n % 128 != 0 || k % 128 != 0 {
        return Err(Error::Rule(format!(
            "{OP}: N={n} K={k} must both be multiples of 128 (one weight scale a 128 x 128 block, \
             W preshuffled to [N/16, K*16])"
        )));
    }
    if m <= 16 {
        let sym = if ring && ring_takes(m, k) {
            N64_PF4
        } else {
            N64
        };
        return Ok((sym, [n / 64, 1, 1], 128));
    }
    let &(_, sym, bm, bn, mfast, thr) = TILES
        .iter()
        .find(|t| m <= t.0)
        .expect("the last tile takes any rows");
    let grid = if mfast {
        [m.div_ceil(bm), n / bn, 1]
    } else {
        [n / bn, m.div_ceil(bm), 1]
    };
    Ok((sym, grid, thr))
}

/// The descriptor: `C[m, ldc] (columns 0..n) = A[m, k] . W8[n, k]^T` with the per-token activation scale.
/// Buffers `a`, `w`, `as`, `ws`, `c`; scalars `m`, `n`, `k`, `ldc` (= n: a promoted GEMM writes its own
/// edge).
///
/// # Errors
/// What [`cell`] refuses.
pub fn resolve(m: u32, n: u32, k: u32, ring: bool) -> Result<LaunchDescriptor> {
    let (sym, grid, threads) = cell(m, n, k, ring)?;
    let sig = abi::signature_for(sym)
        .ok_or_else(|| Error::Rule(format!("{sym}: no declared signature")))?;
    let values: [(&str, i64); 4] = [
        ("m", i64::from(m)),
        ("n", i64::from(n)),
        ("k", i64::from(k)),
        ("ldc", i64::from(n)),
    ];
    let (packed, bytes) = abi::pack(sig);
    let args = sig
        .iter()
        .zip(packed)
        .map(|(a, (offset, size))| Slot {
            name: a.name.to_owned(),
            offset,
            size,
            value: match values.iter().find(|(nm, _)| *nm == a.name) {
                Some((_, v)) => ArgValue::I32 { ival: *v },
                None => ArgValue::Buffer {
                    buffer: a.name.to_owned(),
                },
            },
        })
        .collect();
    Ok(LaunchDescriptor {
        op: OP.into(),
        plugin: PLUGIN.into(),
        kernel: sym.into(),
        grid,
        block: [threads, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![],
        n_launches: 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_band_is_flash_nexts_and_the_ring_takes_only_whole_groups() {
        // the 27B promoted shapes: N 5,120; K 3,072 / 8,704 (TP=2), 6,144 / 17,408 (TP=1)
        for k in [3072, 8704, 6144, 17408] {
            assert_eq!(cell(9, 5120, k, true).unwrap().0, N64_PF4, "K={k}");
            assert_eq!(cell(9, 5120, k, false).unwrap(), (N64, [80, 1, 1], 128));
        }
        assert_eq!(
            cell(24, 5120, 3072, true).unwrap(),
            ("macaq_w8pt_d2w4_ldc", [80, 1, 1], 128)
        );
        assert_eq!(
            cell(48, 5120, 3072, true).unwrap(),
            ("macaq_w8pt_d4w4_ldc", [80, 1, 1], 128)
        );
        assert_eq!(
            cell(72, 5120, 3072, true).unwrap(),
            ("macaq_w8pt_p4w8_ldc", [2, 40, 1], 256)
        );
        assert_eq!(
            cell(2048, 5120, 8704, true).unwrap(),
            ("macaq_w8pt_p4w8_ldc", [32, 40, 1], 256)
        );
        // the ring's limits: 21,504 = 168 tiles > the cap; 384 = 3 tiles is not a whole PF group
        assert_eq!(cell(4, 5120, 21_504, true).unwrap().0, N64);
        assert_eq!(cell(4, 5120, 384, true).unwrap().0, N64);
        assert!(cell(4, 5120, 3000, true).is_err());
        assert!(cell(4, 5000, 3072, true).is_err());
        assert!(cell(0, 5120, 3072, true).is_err());
    }

    #[test]
    fn the_descriptor_carries_the_parents_signature_with_ldc_n() {
        let d = resolve(9, 5120, 8704, true).unwrap();
        assert_eq!(d.plugin, PLUGIN);
        let names: Vec<&str> = d.args.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["a", "w", "as", "ws", "c", "m", "n", "k", "ldc"]);
        assert_eq!(d.arg("ldc").unwrap().value, ArgValue::I32 { ival: 5120 });
        assert_eq!(d.arg("k").unwrap().value, ArgValue::I32 { ival: 8704 });
        assert!(d.workspaces.is_empty());
        assert_eq!(d.n_launches, 1);
        for s in SYMBOLS {
            assert_eq!(abi::signature_for(s), Some(abi::SIG_FN_T8_A8W8_LDC), "{s}");
        }
    }
}
