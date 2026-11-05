//! **C9: the fused mixed forward's plan law** — one forward over a prompt chunk's rows and a
//! decode half's rows, built from the two plans the engine already has.
//!
//! The fused step (`docs/EXEC.md` § the mixed step, LEDGER C8/C9) runs at the smallest prefill
//! rung `R` holding `c + D` rows (`saber_sched::mixed_law::FusedRows`): the chunk's rows `0..c`,
//! the decode bucket's `D` rows at `c..c + D`, the rung's padding after them. It is assembled from
//! the PREFILL plan at `R` and the DECODE plan at the bucket, and nothing is planned anew:
//!
//! * **every row-local op** (the GEMMs, their producers, the norms, the reduces, the embedding
//!   gather) is the prefill plan's own step, ONE launch over all `R` rows — so the decode rows
//!   take the rung's class (C8's proof: the class proxy's) and the chunk rows keep theirs;
//! * **rope + KV append** are the prefill plan's `qk_norm_rope` + `kv_append`, one launch each
//!   over all rows, reading a COMBINED `pos` / `slot` table ([`merge_rows`]): the same kernel
//!   symbol runs in both plans and each row reads only its own entry, so a decode row's q, k and
//!   KV bytes are what the decode plan's own launch writes;
//! * **the two-launch ops** ([`is_chunk_side`] / [`is_decode_side`]): the prefill plan's
//!   attention (`attn.prefill`, retargeted to `q_len = c`) and GDN chunk path (`conv_prep`,
//!   `kkt_solve`, `chunk_scan`) over the chunk, then the decode plan's attention (`attn.decode`)
//!   and GDN update (`conv_update` + `recurrent_update`, or `fused_update`) over the decode rows —
//!   the decode steps' arena bindings REBOUND onto the prefill arena at row `c` ([`rebase`]),
//!   their index tables at their own keys. [`fused_schedule`] is the order: each layer's decode
//!   group right after that layer's chunk group, so every input is written before it is read and
//!   the chunk group's padding writes (rows `>= c`) are overwritten by the decode group after it.
//!
//! Everything here is pure: no device, no file. The engine binds the result
//! (`saber_engine::device::fused_step`).

use crate::error::{Error, Result};
use crate::layer::{BlockPlan, Route, RowSpan};

/// Ops the fused step runs over the CHUNK's rows only (the prefill plan's per-sequence ops).
#[must_use]
pub fn is_chunk_side(op: &str) -> bool {
    [
        ".attn.prefill",
        ".gdn.conv_prep",
        ".gdn.kkt_solve",
        ".gdn.chunk_scan",
    ]
    .iter()
    .any(|t| op.ends_with(t))
}

/// Ops the fused step runs over the DECODE rows only (the decode plan's per-sequence ops).
#[must_use]
pub fn is_decode_side(op: &str) -> bool {
    [
        ".attn.decode",
        ".gdn.conv_update",
        ".gdn.recurrent_update",
        ".gdn.fused_update",
    ]
    .iter()
    .any(|t| op.ends_with(t))
}

/// Per-row ops that exist in BOTH plans with one kernel and a per-row index table: the fused step
/// runs the prefill plan's launch over every row with the combined table; the decode plan's own
/// launch is dropped.
#[must_use]
pub fn is_shared_per_row(op: &str) -> bool {
    op.ends_with(".attn.qk_norm_rope") || op.ends_with(".attn.kv_append")
}

/// `"layer.N."` of an op name, or `None` for an op outside the layers.
pub fn layer_of(op: &str) -> Option<&str> {
    let rest = op.strip_prefix("layer.")?;
    let dot = rest.find('.')?;
    Some(&op[..6 + dot + 1])
}

/// One launch of the fused step: a step of the prefill plan or of the decode plan, by index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusedOp {
    /// `prefill.steps[i]`, bound on the prefill arena as planned.
    Chunk(usize),
    /// `decode.steps[i]`, its arena bindings rebased to the prefill arena at the decode row
    /// offset.
    Decode(usize),
}

/// **The fused step's launch order**: every prefill step in plan order, and each layer's decode
/// group (its [`is_decode_side`] steps, in the decode plan's order) right after the last
/// [`is_chunk_side`] step of the same layer.
///
/// Refused by name: a decode plan whose row-local ops are not the prefill plan's (the two must
/// list the same row-local ops, C8's census), a decode-side op with no chunk group in its layer (or
/// the reverse), a decode-side op some other op is FUSED into (fusion-map row 5, `gdn.norm` into
/// `fused_update`: its row-local half would run twice), and a decode-side op that launches nothing
/// (a decode-side op FUSED into a decode-side op of its own layer -- CMP27AH's `cuf` pair -- is
/// carried by that op's launch: CMP27AA).
///
/// # Errors
/// As above.
pub fn fused_schedule(prefill: &BlockPlan<'_>, decode: &BlockPlan<'_>) -> Result<Vec<FusedOp>> {
    let refuse = |m: String| Err(Error::Refused(format!("the fused step: {m}")));
    // The decode groups, by layer, in plan order.
    let mut groups: std::collections::BTreeMap<String, Vec<usize>> = Default::default();
    let prefill_names: std::collections::HashSet<&str> =
        prefill.steps.iter().map(|p| p.name()).collect();
    for (i, s) in decode.steps.iter().enumerate() {
        let name = s.name();
        if let Route::Fused { into, .. } = &s.route {
            if is_decode_side(into) {
                // CMP27AA: a DECODE-SIDE op folded into a decode-side op of its own layer
                // (CMP27AH's `recurrent_update` into `conv_update`, `--engine-step-arms cuf`) has
                // no row-local half: its work runs inside `into`'s launch in the decode group, so
                // it is skipped here. A row-local op folded into a decode-side op (fusion-map
                // row 5) would run its row-local half twice: refused.
                if is_decode_side(name)
                    && layer_of(name).is_some()
                    && layer_of(name) == layer_of(into)
                {
                    continue;
                }
                return refuse(format!(
                    "`{name}` is fused into the decode-side `{into}`: its row-local half would run \
                     twice"
                ));
            }
        }
        if is_decode_side(name) {
            if !matches!(s.route, Route::Kernel { .. }) {
                return refuse(format!("the decode-side `{name}` launches nothing"));
            }
            let Some(l) = layer_of(name) else {
                return refuse(format!("the decode-side `{name}` is outside the layers"));
            };
            groups.entry(l.to_owned()).or_default().push(i);
            continue;
        }
        if is_shared_per_row(name) || is_chunk_side(name) {
            continue;
        }
        if matches!(s.route, Route::Kernel { .. }) && !prefill_names.contains(name) {
            return refuse(format!(
                "the decode plan's row-local `{name}` is not a step of the {}-row prefill plan",
                prefill.bucket.m()
            ));
        }
    }
    let mut out =
        Vec::with_capacity(prefill.steps.len() + groups.values().map(Vec::len).sum::<usize>());
    let n = prefill.steps.len();
    for (i, s) in prefill.steps.iter().enumerate() {
        out.push(FusedOp::Chunk(i));
        let name = s.name();
        if !is_chunk_side(name) {
            continue;
        }
        let Some(l) = layer_of(name) else {
            return refuse(format!("the chunk-side `{name}` is outside the layers"));
        };
        // The LAST chunk-side step of this layer's group closes it.
        let next_is_same_group = i + 1 < n && {
            let nx = prefill.steps[i + 1].name();
            is_chunk_side(nx) && layer_of(nx) == Some(l)
        };
        if next_is_same_group {
            continue;
        }
        match groups.remove(l) {
            Some(g) => out.extend(g.into_iter().map(FusedOp::Decode)),
            None => {
                return refuse(format!(
                    "{l}'s chunk group has no decode group: the decode rows would skip its \
                     attention or GDN"
                ))
            }
        }
    }
    if let Some((l, _)) = groups.into_iter().next() {
        return refuse(format!("{l}'s decode group has no chunk group to follow"));
    }
    Ok(out)
}

/// **The offset rebinding**: where a decode-plan arena binding (`buffer` at absolute `offset` in
/// the DECODE arena) lands in the PREFILL arena when the decode rows start at row `row0`.
///
/// The binding's position inside its buffer is read as (block, row, byte-in-row) under the decode
/// buffer's [`RowSpan`] and re-laid under the prefill buffer's of the SAME name — which must be
/// the same edge with the same per-row layout (a row-major bf16 edge or q/k/v blocks), and wide
/// enough for `row0 + decode rows`.
///
/// # Errors
/// A buffer missing from either plan, two layouts that are not the same rows, or a decode range
/// that runs past the prefill buffer's rows.
pub fn rebase(
    decode: &BlockPlan<'_>,
    prefill: &BlockPlan<'_>,
    buffer: &str,
    offset: u64,
    row0: u64,
) -> Result<u64> {
    let (at0, per) = rebase_strided(decode, prefill, buffer, offset, row0)?;
    Ok(at0 + row0 * per)
}

/// [`rebase`] as `(offset at row 0, bytes per row)`, so a caller that binds the same step at many
/// row offsets computes `at0 + row0 * per` (the rows of one block are evenly strided). `row0` is
/// the largest offset the caller will use, for the range check.
///
/// # Errors
/// As [`rebase`].
pub fn rebase_strided(
    decode: &BlockPlan<'_>,
    prefill: &BlockPlan<'_>,
    buffer: &str,
    offset: u64,
    row0: u64,
) -> Result<(u64, u64)> {
    let refuse = |m: String| Err(Error::Refused(format!("rebasing `{buffer}`: {m}")));
    let Some(d) = decode.buffer(buffer) else {
        return refuse("not a decode-plan buffer".into());
    };
    let Some(p) = prefill.buffer(buffer) else {
        return refuse(format!(
            "not a buffer of the {}-row prefill plan",
            prefill.bucket.m()
        ));
    };
    let delta = offset
        .checked_sub(d.offset)
        .filter(|x| *x <= d.bytes)
        .ok_or_else(|| Error::Refused(format!("rebasing `{buffer}`: offset outside the buffer")))?;
    let d_rows = d.rows.rows(d.bytes);
    let p_rows = p.rows.rows(p.bytes);
    if row0 + d_rows > p_rows {
        return refuse(format!(
            "{d_rows} decode row(s) at row {row0} run past the prefill buffer's {p_rows}"
        ));
    }
    let at = match (&d.rows, &p.rows) {
        (RowSpan::Contiguous { bytes: a }, RowSpan::Contiguous { bytes: b })
            if a == b && *a > 0 =>
        {
            let (r, inner) = (delta / a, delta % a);
            (p.offset + r * a + inner, *a)
        }
        (RowSpan::Blocks { blocks: a }, RowSpan::Blocks { blocks: b })
            if a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.1 == y.1 && x.1 > 0) =>
        {
            // The block the binding starts in: the last whose base is at or below it.
            let k = a
                .iter()
                .rposition(|(base, _)| *base <= delta)
                .ok_or_else(|| Error::Refused(format!("rebasing `{buffer}`: before block 0")))?;
            let (base, per) = a[k];
            let (r, inner) = ((delta - base) / per, (delta - base) % per);
            (p.offset + b[k].0 + r * per + inner, per)
        }
        (x, y) => return refuse(format!("decode rows {x:?} and prefill rows {y:?} differ")),
    };
    Ok(at)
}

/// **C10: a packed step's further chunk** — where a PREFILL-plan arena binding lands when the
/// chunk's rows start at row `row0` of the rung instead of row 0 (chunks laid out per prompt in
/// queue order). The binding's (block, row, byte-in-row) under its buffer's [`RowSpan`] moves by
/// `row0` rows inside the same block; `rows` is the chunk's live rows, checked to stay inside the
/// buffer.
///
/// # Errors
/// A buffer the plan lacks, an offset outside it, a layout that is not rows, or a chunk that runs
/// past the buffer's rows.
pub fn shift_rows(
    plan: &BlockPlan<'_>,
    buffer: &str,
    offset: u64,
    row0: u64,
    rows: u64,
) -> Result<u64> {
    let refuse = |m: String| Err(Error::Refused(format!("shifting `{buffer}`: {m}")));
    let Some(b) = plan.buffer(buffer) else {
        return refuse(format!(
            "not a buffer of the {}-row prefill plan",
            plan.bucket.m()
        ));
    };
    let Some(delta) = offset.checked_sub(b.offset).filter(|x| *x <= b.bytes) else {
        return refuse("offset outside the buffer".into());
    };
    let total = b.rows.rows(b.bytes);
    let (row, per) = match &b.rows {
        RowSpan::Contiguous { bytes: a } if *a > 0 => (delta / a, *a),
        RowSpan::Blocks { blocks } if !blocks.is_empty() && blocks.iter().all(|x| x.1 > 0) => {
            let k = blocks
                .iter()
                .rposition(|(base, _)| *base <= delta)
                .ok_or_else(|| Error::Refused(format!("shifting `{buffer}`: before block 0")))?;
            let (base, per) = blocks[k];
            ((delta - base) / per, per)
        }
        other => return refuse(format!("its rows are {other:?}")),
    };
    if row + row0 + rows > total {
        return refuse(format!(
            "{rows} row(s) at row {} run past the buffer's {total}",
            row + row0
        ));
    }
    Ok(offset + row0 * per)
}

/// **C10: the prefill plan's CHUNK-PRIVATE buffers** — arena buffers only [`is_chunk_side`] steps
/// bind (the GDN chunk path's `q`/`k`/`v`/`g`/`beta` split, `a_mat`, `gcs`, ...). A packed step's
/// further chunks run their chunk groups one after another on the stream, so each reuses these at
/// the plan's own offsets (rows from 0); every other buffer a chunk-side op binds carries rows a
/// row-local op wrote or reads, and is shifted to the chunk's rows ([`shift_rows`]).
#[must_use]
pub fn chunk_private_buffers(prefill: &BlockPlan<'_>) -> std::collections::BTreeSet<String> {
    let arena = |s: &crate::layer::Step<'_>| -> Vec<String> {
        s.bindings
            .iter()
            .filter_map(|(_, src)| match src {
                crate::layer::Source::Arena { buffer, .. } => Some(buffer.clone()),
                _ => None,
            })
            .collect()
    };
    let mut chunk: std::collections::BTreeSet<String> = Default::default();
    let mut other: std::collections::BTreeSet<String> = Default::default();
    for s in &prefill.steps {
        if is_chunk_side(s.name()) {
            chunk.extend(arena(s));
        } else {
            other.extend(arena(s));
        }
    }
    chunk.difference(&other).cloned().collect()
}

/// The blob key of a PACKED step's chunk `k >= 1` for the chunk-side table `suffix` (`cu`,
/// `cache_idx`, `block_table`, ...): the further chunks' tables live beside chunk 0's, one set per
/// chunk, under keys every per-op table shares by suffix (the laws read the suffix alone).
#[must_use]
pub fn packed_key(k: u32, suffix: &str) -> String {
    format!("{PACKED_PREFIX}{k}.{suffix}")
}

/// The prefix of every [`packed_key`].
pub const PACKED_PREFIX: &str = "fused.packed.k";

/// Whether a blob key is a packed chunk's ([`packed_key`]): a normal step never writes one.
#[must_use]
pub fn is_packed_key(key: &str) -> bool {
    key.starts_with(PACKED_PREFIX)
}

/// `(k, suffix)` of a [`packed_key`].
#[must_use]
pub fn parse_packed_key(key: &str) -> Option<(u32, &str)> {
    let rest = key.strip_prefix(PACKED_PREFIX)?;
    let (k, suffix) = rest.split_once('.')?;
    Some((k.parse().ok()?, suffix))
}

/// Where one rebound binding lands: `(on the prefill arena?, offset at row 0, bytes per row)`.
pub type RebasedAt = (bool, u64, u64);

/// **One (prefill rung, decode bucket) pair's fused binding plan**, built once and reused: the
/// launch order and, for every decode-side step's arena binding, where it lands as
/// `(on the prefill arena?, offset at row 0, bytes per row)` — so a step binds at row `c` with
/// one multiply per binding instead of re-deriving the law.
#[derive(Clone, Debug)]
pub struct FusedBinding {
    /// [`fused_schedule`]'s order.
    pub ops: Vec<FusedOp>,
    /// Per decode step index, per binding index: `Some((prefill, at0, per_row))` for an arena
    /// binding, `None` for a weight / state / index binding (resolved as the decode plan binds it).
    pub rebound: std::collections::BTreeMap<usize, Vec<Option<RebasedAt>>>,
    /// The widest row offset every rebound binding was checked at.
    pub max_row0: u64,
    /// C10: [`chunk_private_buffers`] of the prefill plan — a packed step's further chunks bind
    /// these at the plan's offsets and shift every other arena binding to their rows.
    pub chunk_private: std::collections::BTreeSet<String>,
}

impl FusedBinding {
    /// Build the plan for `prefill` x `decode`, checking every rebound range at `max_row0` (the
    /// rung minus the decode rows: the largest chunk the pair can carry).
    ///
    /// # Errors
    /// As [`fused_schedule`] and [`rebind`].
    pub fn new(prefill: &BlockPlan<'_>, decode: &BlockPlan<'_>, max_row0: u64) -> Result<Self> {
        let ops = fused_schedule(prefill, decode)?;
        let mut rebound = std::collections::BTreeMap::new();
        for op in &ops {
            let FusedOp::Decode(i) = *op else { continue };
            let mut per = Vec::with_capacity(decode.steps[i].bindings.len());
            for (_, src) in &decode.steps[i].bindings {
                per.push(match src {
                    crate::layer::Source::Arena { buffer, offset, .. } => {
                        Some(match rebind(decode, prefill, buffer, *offset, max_row0)? {
                            Rebound::Prefill(_) => {
                                let (at0, row) =
                                    rebase_strided(decode, prefill, buffer, *offset, max_row0)?;
                                (true, at0, row)
                            }
                            Rebound::Decode(o) => (false, o, 0),
                        })
                    }
                    _ => None,
                });
            }
            rebound.insert(i, per);
        }
        Ok(Self {
            ops,
            rebound,
            max_row0,
            chunk_private: chunk_private_buffers(prefill),
        })
    }
}

/// Where a decode-side op's arena binding lives in a fused step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rebound {
    /// An edge the fused rows share: this offset in the PREFILL arena ([`rebase`]).
    Prefill(u64),
    /// A scratch buffer PRIVATE to the decode-side ops (an `Internal` buffer no other decode
    /// step binds, e.g. `fused_update`'s own q/k/v split): it stays at this offset in the
    /// DECODE arena, which the fused step does not otherwise touch.
    Decode(u64),
}

/// [`rebase`] for an edge both plans hold, or [`Rebound::Decode`] for a decode-private scratch
/// buffer (refused if any op outside the decode-side set binds it: then it carries rows the
/// fused step's row-local ops must see).
///
/// # Errors
/// As [`rebase`], or a buffer the prefill plan lacks that is not decode-private scratch.
pub fn rebind(
    decode: &BlockPlan<'_>,
    prefill: &BlockPlan<'_>,
    buffer: &str,
    offset: u64,
    row0: u64,
) -> Result<Rebound> {
    if prefill.buffer(buffer).is_some() {
        return rebase(decode, prefill, buffer, offset, row0).map(Rebound::Prefill);
    }
    let private = decode
        .buffer(buffer)
        .is_some_and(|b| b.kind == crate::layer::BufferKind::Internal)
        && decode.steps.iter().all(|s| {
            is_decode_side(s.name())
            || !s.bindings.iter().any(|(_, src)| {
                matches!(src, crate::layer::Source::Arena { buffer: b, .. } if b == buffer)
            })
        });
    if private {
        Ok(Rebound::Decode(offset))
    } else {
        Err(Error::Refused(format!(
            "rebasing `{buffer}`: not a buffer of the {}-row prefill plan, and not scratch private              to the decode-side ops",
            prefill.bucket.m()
        )))
    }
}

/// **The combined per-row table** of a fused step: the chunk's table (`chunk_rows` rows, the
/// rung's) with rows `row0..row0 + decode_rows` replaced by the decode half's table (`decode_rows`
/// rows), plane by plane (`pos` is `[3, rows]`, `slot` `[rows]`), each entry `elem` bytes.
///
/// # Errors
/// A table that is not a whole number of planes of its rows, two tables with different plane
/// counts, or a decode range past the rung.
pub fn merge_rows(
    chunk: &[u8],
    chunk_rows: u32,
    decode: &[u8],
    decode_rows: u32,
    row0: u32,
    elem: usize,
) -> Result<Vec<u8>> {
    let (cr, dr, r0) = (chunk_rows as usize, decode_rows as usize, row0 as usize);
    let refuse = |m: String| Err(Error::Refused(format!("merging a per-row table: {m}")));
    if cr == 0 || dr == 0 || elem == 0 || chunk.len() % (cr * elem) != 0 {
        return refuse(format!(
            "{} bytes is not whole planes of {cr} row(s) x {elem} B",
            chunk.len()
        ));
    }
    let planes = chunk.len() / (cr * elem);
    if decode.len() != planes * dr * elem {
        return refuse(format!(
            "the decode table is {} bytes, {planes} plane(s) of {dr} row(s) is {}",
            decode.len(),
            planes * dr * elem
        ));
    }
    if r0 + dr > cr {
        return refuse(format!("{dr} decode row(s) at row {r0} past {cr}"));
    }
    let mut out = chunk.to_vec();
    for p in 0..planes {
        let dst = (p * cr + r0) * elem;
        let src = p * dr * elem;
        out[dst..dst + dr * elem].copy_from_slice(&decode[src..src + dr * elem]);
    }
    Ok(out)
}

/// Which step an index table of a fused step is built from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeySide {
    /// The chunk's step (the prefill plan's per-sequence ops and every row-local table).
    Chunk,
    /// The decode half's step (the decode-side ops' tables, at their own keys).
    Decode,
    /// Both, row by row ([`merge_rows`]): the shared rope / append ops' `pos` and `slot`.
    PerRow,
}

/// The side an index key (`"<op>.<table>"`) is built from in a fused step.
#[must_use]
pub fn key_side(key: &str) -> KeySide {
    let (op, table) = key.rsplit_once('.').unwrap_or((key, ""));
    if is_decode_side(op) {
        KeySide::Decode
    } else if is_shared_per_row(op) && (table == "pos" || table == "slot") {
        KeySide::PerRow
    } else {
        KeySide::Chunk
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_rows_replaces_the_decode_rows_plane_by_plane() {
        // pos: 3 planes of 6 chunk rows, 2 decode rows at row 3.
        let chunk: Vec<u8> = (0..18u8).collect();
        let decode = vec![100u8, 101, 102, 103, 104, 105];
        let m = merge_rows(&chunk, 6, &decode, 2, 3, 1).expect("merge");
        assert_eq!(
            m,
            vec![0, 1, 2, 100, 101, 5, 6, 7, 8, 102, 103, 11, 12, 13, 14, 104, 105, 17]
        );
        // Refusals: a plane-count mismatch, a range past the rung.
        assert!(merge_rows(&chunk, 6, &decode[..4], 2, 3, 1).is_err());
        assert!(merge_rows(&chunk, 6, &decode, 2, 5, 1).is_err());
    }

    #[test]
    fn key_sides() {
        assert_eq!(key_side("layer.3.attn.decode.block_table"), KeySide::Decode);
        assert_eq!(
            key_side("layer.2.gdn.recurrent_update.sidx"),
            KeySide::Decode
        );
        assert_eq!(key_side("layer.2.gdn.conv_update.naccept"), KeySide::Decode);
        assert_eq!(key_side("layer.3.attn.prefill.block_table"), KeySide::Chunk);
        assert_eq!(key_side("layer.2.gdn.conv_prep.cu"), KeySide::Chunk);
        assert_eq!(key_side("layer.3.attn.qk_norm_rope.pos"), KeySide::PerRow);
        assert_eq!(key_side("layer.3.attn.kv_append.slot"), KeySide::PerRow);
        // The shared ops' constants are the chunk's (and step-invariant anyway).
        assert_eq!(
            key_side("layer.3.attn.qk_norm_rope.cos_sin"),
            KeySide::Chunk
        );
        assert_eq!(layer_of("layer.12.gdn.chunk_scan"), Some("layer.12."));
        // C10: a packed chunk's keys carry the suffix the laws read, and parse back.
        let k = packed_key(3, "block_table");
        assert!(is_packed_key(&k) && k.ends_with(".block_table"));
        assert_eq!(parse_packed_key(&k), Some((3, "block_table")));
        assert!(!is_packed_key("layer.3.attn.prefill.block_table"));
        assert_eq!(parse_packed_key("layer.3.attn.prefill.cu"), None);
        assert_eq!(layer_of("embed_tokens"), None);
    }
}
