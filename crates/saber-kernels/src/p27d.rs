//! P27-W2-2: the 27B drafter's DEVICE CANDIDATE SEAM (`--engine-device-step cand`).
//!
//! `hip/p27_dstep.hip`'s `p27d_cand_seam` merges the two ranks' top-K candidate shards on the card
//! (each rank pushes its shard into the peer's P2P window and merges by the host's own stable
//! descending law), so the drafter's tail graph is enqueued behind the backbone with no host wait.
//! This module is the layout both sides agree on: the window's row stride and flag word, the
//! scratch's arrays and the error bits. Launched raw (one row a workgroup of 32), never planned.

/// The code object.
pub const PLUGIN: &str = "p27_dstep";
/// The one symbol.
pub const SYM: &str = "p27d_cand_seam";
/// Threads a workgroup (one wave; lane `l < K` owns candidate `l`, all 32 merge).
pub const BLOCK: u32 = 32;
/// Candidates a rank's shard holds a row (`P27D_K`): the drafter's `selector_top_k`.
pub const K: usize = 16;
/// Tagged 64-bit words a window row (`P27D_RS`): `K` value words + `2K` id words + padding; each
/// word is `(call << 32) | 32 data bits` (data as the flag).
pub const ROW_WORDS: usize = 64;
/// Error bit: the peer's flag never landed (`P27D_ERR_TIMEOUT`).
pub const ERR_TIMEOUT: u32 = 1;
/// Error bit: a NaN in a row (the host's comparator would be inconsistent; `P27D_ERR_NAN`).
pub const ERR_NAN: u32 = 2;
/// Error bit: the launch's geometry is not the kernel's (`P27D_ERR_GEOM`).
pub const ERR_GEOM: u32 = 4;

/// Bytes of one rank's window: two slots of `rows_max` rows of 64-bit words.
#[must_use]
pub const fn window_bytes(rows_max: usize) -> u64 {
    (2 * rows_max * ROW_WORDS * 8) as u64
}

/// Bytes of the scratch for `rows` rows: the error word (+ one pad word), then mine values (f32),
/// mine LOCAL ids (i64), theirs values (f32), theirs GLOBAL ids (i64), merged values (f32), merged
/// GLOBAL ids (i64).
#[must_use]
pub const fn scratch_bytes(rows: usize) -> u64 {
    (8 + rows * K * 36) as u64
}

/// Byte offset of the scratch's `merged` block for `rows` rows (values then global ids: the bytes
/// the host merge seeded).
#[must_use]
pub const fn merged_off(rows: usize) -> usize {
    8 + rows * K * 24
}

/// Byte offset of the scratch's `mine` block (values then local ids: the host read's bytes).
pub const MINE_OFF: usize = 8;

/// Byte offset of the scratch's `theirs` block for `rows` rows (values then global ids: the
/// socket frame's bytes, `CandShard::encode`).
#[must_use]
pub const fn theirs_off(rows: usize) -> usize {
    8 + rows * K * 12
}

/// Names the error word's bits, or `None` when it is clear.
#[must_use]
pub fn error_text(word: u32) -> Option<String> {
    if word == 0 {
        return None;
    }
    let mut v = Vec::new();
    if word & ERR_TIMEOUT != 0 {
        v.push("the peer's shard never landed (timeout)");
    }
    if word & ERR_NAN != 0 {
        v.push("a NaN candidate (the host merge's order is undefined there)");
    }
    if word & ERR_GEOM != 0 {
        v.push("the launch geometry is not the kernel's");
    }
    if word & !(ERR_TIMEOUT | ERR_NAN | ERR_GEOM) != 0 {
        v.push("an unknown error bit");
    }
    Some(v.join("; "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_layout_matches_the_source() {
        let src = include_str!("../hip/p27_dstep.hip");
        for (name, v) in [
            ("P27D_K", K as u64),
            ("P27D_RS", ROW_WORDS as u64),
            ("P27D_ERR_TIMEOUT", u64::from(ERR_TIMEOUT)),
            ("P27D_ERR_NAN", u64::from(ERR_NAN)),
            ("P27D_ERR_GEOM", u64::from(ERR_GEOM)),
        ] {
            let line = format!("#define {name} {v}");
            assert!(
                src.lines()
                    .any(|l| l.trim_end().trim_end_matches('u') == line),
                "{line} not in the source"
            );
        }
        // the K values and the 2K id words fit a row
        const { assert!(3 * K <= ROW_WORDS) };
        assert!(!src.contains("P27D_FLAG"), "the separate flag is gone (h1)");
        assert_eq!(scratch_bytes(1), 8 + 16 * 36);
        assert_eq!(merged_off(2), 8 + 2 * 16 * 24);
        assert_eq!(theirs_off(2), 8 + 2 * 16 * 12);
        assert_eq!(error_text(0), None);
        assert!(error_text(ERR_NAN).unwrap().contains("NaN"));
        assert!(error_text(8).unwrap().contains("unknown"));
    }
}
