//! **T6's shadow digest** (`hip/race_digest.hip`): the per-proposal stage digests computed on the
//! card, so the determinism monitor adds no read-back and no sync (`--engine-race-check shadow`).
//!
//! An INSTRUMENT, not a serve path, like the null plugin: in no model spec, no registry row
//! (`saber_plan::SEPARATELY_PLANNED`), no autotune key, and keyed to the null plugin's instrument
//! table for the policy digest ([`crate::rule_table_for_plugin`]) because it has no launch policy
//! to state -- its geometry is the digest's own definition ([`BLOCK`], [`PARTS`], [`KV_PARTS`]),
//! not a measured choice, and it is spelled once, here.
//!
//! The hash is ORDER-FREE: region bytes as little-endian u64 words `w_i` (a short tail zero-padded),
//! each contributing `mix(w_i, i) = sm64(w_i ^ sm64(i))` by XOR, so any partition of the words over
//! workgroups XORs to the same value. [`host_digest`] is the whole function on the host; the device
//! writes `parts` partials per region and [`fold`] finishes them with the length. The tests hold a
//! partitioned walk (the device's) against the one-pass host function, and a flipped bit, a swapped
//! pair of words and a moved tail byte against the digest (the negative controls).

use crate::abi;
use crate::rules::{ArgValue, LaunchDescriptor, Slot};
use crate::{Error, Result};

/// The one code object both kernels live in.
pub const PLUGIN: &str = "race_digest";
/// The regions kernel.
pub const SYM_REGIONS: &str = "saber_race_digest_regions";
/// The drafter K/V pages kernel.
pub const SYM_KV: &str = "saber_race_digest_kv";
/// Threads per workgroup (`RD_BLOCK` in the source; the LDS reduction is sized by it).
pub const BLOCK: u32 = 256;
/// Workgroups per region.
pub const PARTS: u32 = 8;
/// Workgroups per (layer, cache) of the K/V digest.
pub const KV_PARTS: u32 = 16;

/// splitmix64's finaliser (the device's `rd_sm64`).
#[must_use]
pub const fn sm64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// One word's contribution at word index `i` (the device's `rd_mix`).
#[must_use]
pub const fn mix(w: u64, i: u64) -> u64 {
    sm64(w ^ sm64(i))
}

fn word(bytes: &[u8]) -> u64 {
    let mut w = [0u8; 8];
    w[..bytes.len()].copy_from_slice(bytes);
    u64::from_le_bytes(w)
}

/// The unfinished XOR of every word of `bytes`, keyed from `key0` (what the partials XOR to).
#[must_use]
pub fn host_partial(bytes: &[u8], key0: u64) -> u64 {
    bytes
        .chunks(8)
        .enumerate()
        .fold(0u64, |h, (i, c)| h ^ mix(word(c), key0 + i as u64))
}

/// A region's digest from its partials and its length.
#[must_use]
pub fn fold(partials: &[u64], bytes: u64) -> u64 {
    sm64(partials.iter().fold(0u64, |h, p| h ^ p) ^ sm64(bytes ^ 0x5bd1_e995_5bd1_e995))
}

/// The digest the device computes for `bytes`, in one pass on the host.
#[must_use]
pub fn host_digest(bytes: &[u8]) -> u64 {
    fold(&[host_partial(bytes, 0)], bytes.len() as u64)
}

/// The K/V stage's digest of one (layer, cache) from its partials (no length: the page count is
/// data, and the pages are keyed by their logical place).
#[must_use]
pub fn fold_kv(partials: &[u64]) -> u64 {
    fold(partials, u64::MAX)
}

fn slots(sig: &[abi::SigArg], scalars: &[(&str, ArgValue)]) -> Result<(Vec<Slot>, u32)> {
    let (packed, bytes) = abi::pack(sig);
    let mut out = Vec::with_capacity(sig.len());
    for (a, (offset, size)) in sig.iter().zip(packed) {
        let value = if a.kind == abi::ArgKind::Buffer {
            ArgValue::Buffer {
                buffer: a.name.to_owned(),
            }
        } else {
            scalars
                .iter()
                .find(|(n, _)| *n == a.name)
                .map(|(_, v)| v.clone())
                .ok_or_else(|| Error::Rule(format!("race_digest: no value for `{}`", a.name)))?
        };
        out.push(Slot {
            name: a.name.to_owned(),
            offset,
            size,
            value,
        });
    }
    Ok((out, bytes))
}

fn i32v(what: &str, v: u64) -> Result<ArgValue> {
    let ival = i32::try_from(v)
        .map_err(|_| Error::Rule(format!("race_digest: `{what}` = {v} is past an i32")))?;
    Ok(ArgValue::I32 {
        ival: i64::from(ival),
    })
}

fn i64v(what: &str, v: u64) -> Result<ArgValue> {
    let ival = i64::try_from(v)
        .map_err(|_| Error::Rule(format!("race_digest: `{what}` = {v} is past an i64")))?;
    Ok(ArgValue::I64 { ival })
}

/// The regions launch: `n` regions in the `desc` table (device address, bytes) plus the per-launch
/// `extra` window of `extra_bytes`. Buffers bound by the caller: `desc`, `extra`, `out`. Writes
/// `(n + 1) * PARTS` partials.
///
/// # Errors
/// A count past the kernel's scalar widths.
pub fn resolve_regions(n: u32, extra_bytes: u64) -> Result<LaunchDescriptor> {
    let (args, bytes) = slots(
        abi::SIG_RACE_DIGEST_REGIONS,
        &[
            ("extra_bytes", i64v("extra_bytes", extra_bytes)?),
            ("n", i32v("n", u64::from(n))?),
            ("parts", i32v("parts", u64::from(PARTS))?),
        ],
    )?;
    Ok(LaunchDescriptor {
        op: "race_digest_regions".into(),
        plugin: PLUGIN.into(),
        kernel: SYM_REGIONS.into(),
        grid: [(n + 1) * PARTS, 1, 1],
        block: [BLOCK, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![],
        n_launches: 1,
    })
}

/// The drafter K/V geometry the pages kernel walks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvWalk {
    /// Request rows of the shape.
    pub reqs: u32,
    /// Block-table row stride.
    pub bt_stride: u32,
    /// Tokens a page holds.
    pub page_tokens: u32,
    /// Drafter layers.
    pub layers: u32,
    /// Pages per layer (a page id at or past it is skipped).
    pub blocks: u32,
    /// Bytes of one layer's cache.
    pub per_layer: u64,
    /// Bytes of one page of one layer of one cache (a multiple of 8).
    pub block_bytes: u64,
}

impl KvWalk {
    /// Partials the launch writes: `layers * 2 * KV_PARTS`.
    #[must_use]
    pub fn partials(&self) -> u32 {
        self.layers * 2 * KV_PARTS
    }
}

/// The K/V pages launch. Buffers bound by the caller: `block_table`, `seq_lens`, `kbase`, `vbase`,
/// `out`. Writes [`KvWalk::partials`] partials, `(layer * 2 + cache) * KV_PARTS + part`.
///
/// # Errors
/// A page that is not whole words, or a count past the kernel's scalar widths.
pub fn resolve_kv(w: KvWalk) -> Result<LaunchDescriptor> {
    if w.block_bytes % 8 != 0 || w.page_tokens == 0 {
        return Err(Error::Rule(format!(
            "race_digest: a {}-byte page of {} token(s) is not whole words",
            w.block_bytes, w.page_tokens
        )));
    }
    let (args, bytes) = slots(
        abi::SIG_RACE_DIGEST_KV,
        &[
            ("per_layer", i64v("per_layer", w.per_layer)?),
            ("block_bytes", i64v("block_bytes", w.block_bytes)?),
            ("reqs", i32v("reqs", u64::from(w.reqs))?),
            ("bt_stride", i32v("bt_stride", u64::from(w.bt_stride))?),
            (
                "page_tokens",
                i32v("page_tokens", u64::from(w.page_tokens))?,
            ),
            ("parts", i32v("parts", u64::from(KV_PARTS))?),
            ("blocks", i32v("blocks", u64::from(w.blocks))?),
        ],
    )?;
    Ok(LaunchDescriptor {
        op: "race_digest_kv".into(),
        plugin: PLUGIN.into(),
        kernel: SYM_KV.into(),
        grid: [w.partials(), 1, 1],
        block: [BLOCK, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![],
        n_launches: 1,
    })
}

/// The K/V digest the device computes, on the host: `pages[r][j]` is request `r`'s `j`-th page's
/// bytes of ONE (layer, cache), `None` for a page the kernel skips.
#[must_use]
pub fn host_kv_digest(pages: &[Vec<Option<Vec<u8>>>], bt_stride: u32, block_bytes: u64) -> u64 {
    let wpp = block_bytes / 8;
    let mut h = 0u64;
    for (r, row) in pages.iter().enumerate() {
        for (j, pg) in row.iter().enumerate() {
            if let Some(b) = pg {
                h ^= host_partial(b, (r as u64 * u64::from(bt_stride) + j as u64) * wpp);
            }
        }
    }
    fold_kv(&[h])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The device's partition: `parts` workgroups of `BLOCK` threads, part `p` thread `t` taking
    /// words `p * BLOCK + t + k * parts * BLOCK`, the tail word (if any) in part 0.
    fn device_walk(bytes: &[u8], parts: u32) -> Vec<u64> {
        let nw = bytes.len() / 8;
        let mut out = vec![0u64; parts as usize];
        for (p, slot) in out.iter_mut().enumerate() {
            for t in 0..BLOCK as usize {
                let mut i = p * BLOCK as usize + t;
                while i < nw {
                    *slot ^= mix(word(&bytes[8 * i..8 * i + 8]), i as u64);
                    i += parts as usize * BLOCK as usize;
                }
            }
        }
        if bytes.len() % 8 != 0 {
            out[0] ^= mix(word(&bytes[8 * nw..]), nw as u64);
        }
        out
    }

    fn sample(n: usize) -> Vec<u8> {
        (0..n).map(|i| (sm64(i as u64) >> 11) as u8).collect()
    }

    #[test]
    fn the_partitioned_walk_is_the_one_pass_digest() {
        for n in [0usize, 1, 7, 8, 9, 4096, 70_003] {
            let b = sample(n);
            let d = fold(&device_walk(&b, PARTS), n as u64);
            assert_eq!(d, host_digest(&b), "{n} bytes");
            // Any other partition too: the digest is a function of the bytes, not of the grid.
            assert_eq!(
                fold(&device_walk(&b, 3), n as u64),
                d,
                "{n} bytes at 3 parts"
            );
        }
    }

    /// NEGATIVE CONTROLS: a flipped bit, a swapped pair of words, a longer region with the same
    /// words (zero tail), and a moved tail byte each change the digest.
    #[test]
    fn a_flipped_bit_a_swap_and_a_length_each_move_the_digest() {
        let b = sample(4099);
        let d = host_digest(&b);
        let mut flip = b.clone();
        flip[1234] ^= 0x10;
        assert_ne!(host_digest(&flip), d);
        let mut swap = b.clone();
        let (x, y) = (swap[0..8].to_vec(), swap[8..16].to_vec());
        swap[0..8].copy_from_slice(&y);
        swap[8..16].copy_from_slice(&x);
        assert_ne!(host_digest(&swap), d);
        let mut longer = b.clone();
        longer.push(0);
        assert_ne!(host_digest(&longer), d);
        let mut moved = b.clone();
        let last = moved.len() - 1;
        moved.swap(last, last - 1);
        assert_ne!(host_digest(&moved), d);
    }

    #[test]
    fn the_kv_digest_is_keyed_by_the_logical_page_and_order_free() {
        let page = |s: u64| -> Vec<u8> { (0..64).map(|i| (sm64(s * 97 + i) >> 5) as u8).collect() };
        let pages = vec![
            vec![Some(page(1)), Some(page(2)), None],
            vec![Some(page(3))],
        ];
        let d = host_kv_digest(&pages, 4, 64);
        // The partition over parts (pages j mod parts) XORs to the same value.
        let mut parts = vec![0u64; KV_PARTS as usize];
        for (r, row) in pages.iter().enumerate() {
            for (j, pg) in row.iter().enumerate() {
                if let Some(b) = pg {
                    parts[j % KV_PARTS as usize] ^= host_partial(b, (r as u64 * 4 + j as u64) * 8);
                }
            }
        }
        assert_eq!(fold_kv(&parts), d);
        // Control: the same pages at other logical places.
        let swapped = vec![
            vec![Some(page(2)), Some(page(1)), None],
            vec![Some(page(3))],
        ];
        assert_ne!(host_kv_digest(&swapped, 4, 64), d);
    }

    #[test]
    fn both_launches_resolve_against_their_declared_signatures() {
        let r = resolve_regions(40, 5 * 5120 * 2 * 8).expect("regions");
        assert_eq!(r.grid, [41 * PARTS, 1, 1]);
        assert_eq!(r.block, [BLOCK, 1, 1]);
        assert_eq!(
            r.args.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            ["desc", "extra", "out", "extra_bytes", "n", "parts"]
        );
        assert_eq!(r.kernarg_bytes, 40);
        let w = KvWalk {
            reqs: 8,
            bt_stride: 256,
            page_tokens: 16,
            layers: 5,
            blocks: 4096,
            per_layer: 4096 * 16384,
            block_bytes: 16384,
        };
        let k = resolve_kv(w).expect("kv");
        assert_eq!(k.grid, [5 * 2 * KV_PARTS, 1, 1]);
        assert_eq!(k.args.len(), 12);
        // Control: a page that is not whole words is refused by name.
        let bad = KvWalk {
            block_bytes: 12,
            ..w
        };
        assert!(resolve_kv(bad).is_err());
    }
}
