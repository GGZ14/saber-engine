//! Which `lm_head` the target serves, and ONE head record on the card when the drafter is
//! resident (workstreams N3, N5, N6).
//!
//! **The law (N6, the user's decision of 2026-09-22 ~22:30, superseding N5's bf16 default):** the
//! head is served at the CHECKPOINT's own precision and never requantized. The default,
//! [`TargetHead::Exact`], is not a dtype: it resolves to the record kind the target PACK carries
//! for the exact head, and the pack carries the checkpoint's (`saber-packplan --head auto` reads the
//! checkpoint's `lm_head.weight` dtype and `weight_scale`, `saber_model::checkpoint_head`):
//!
//! | pack record | resolved head | graph | this checkpoint |
//! |---|---|---|---|
//! | `lm_head.w8` (layout 15) + `lm_head.ws` (16) | [`TargetHead::Fp8`] | `Graph::with_fp8_head` | **the default** (per-channel e4m3) |
//! | `lm_head.w` (layout 14) | [`TargetHead::Bf16`] | `Graph::with_bf16_head` | N5's pack, `--head bf16` |
//! | `lm_head.w/.ws/.wref` (MXFP4) | refused under the default | — | `--mxfp4-head`'s pack |
//!
//! Both exact kinds are prod's matrix, `bf16(f32(w8) * f32(scale))`, bit for bit (N4); the fp8 one
//! is half the bytes and −1.07 ms at `b1s8`.
//!
//! **Sharing (N3's law, generalized by N6).** The drafter's exact rerank reads the SAME matrix: its
//! pack carries it as `head.rerank.w` (bf16) or, since N6, `head.rerank.w8` + `head.rerank.ws` (the
//! checkpoint's fp8 bytes, read by `dflash_rerank_exact_w8`). When the drafter is resident the
//! target does NOT upload its head records; it binds the drafter's at the addresses the drafter's
//! upload gave them. One upload, one residency. Refused, by name, when
//!
//! * the drafter's pack carries the other KIND (a bf16 `head.rerank.w` beside an fp8 target head, or
//!   the reverse) — the shared record's kind must match the target's;
//! * a record is missing on either side, or is another layout, shape or length; or
//! * any pair's `sha256` differ — the target's record (from the target pack's own table) is the
//!   REFERENCE, so a drafter pack built from another checkpoint is a refusal at boot.
//!
//! Pure over the two record tables so the law is a host test; `device::run` calls it before the
//! target's upload window is computed.

use std::collections::BTreeSet;
use std::path::Path;

use saber_mem::pack::PackReader;
use saber_model::graph::{
    LAYOUT_BF16_FROM_FP8_CHANNEL, LAYOUT_F32_CHANNEL_SCALE, LAYOUT_FP8_CHANNEL,
};

/// The target's bf16 head record (`Graph::with_bf16_head`).
pub const TARGET_HEAD: &str = "lm_head.w";
/// The drafter's bf16 exact-rerank head record.
pub const DRAFTER_HEAD: &str = "head.rerank.w";
/// The fp8-weight head's matrix record (`Graph::with_fp8_head`, workstream N4).
pub const TARGET_FP8_HEAD: &str = "lm_head.w8";
/// ... and its per-channel f32 scale.
pub const TARGET_FP8_SCALE: &str = "lm_head.ws";
/// The drafter's fp8 rerank head (N6): the same bytes as [`TARGET_FP8_HEAD`].
pub const DRAFTER_FP8_HEAD: &str = "head.rerank.w8";
/// ... and its scale, the same bytes as [`TARGET_FP8_SCALE`].
pub const DRAFTER_FP8_SCALE: &str = "head.rerank.ws";

/// The default target pack at TP=2 (the checkpoint's fp8 head).
pub const DEFAULT_PACK: &str = "/packs/qwen38-27b-nvfp4-devfinal-fp8head";
/// The drafter pack whose rerank reads the fp8 record (TP=2).
pub const DEFAULT_DRAFTER_PACK: &str = "/packs/qwen38-dflash2-drafter-fp8head";
/// The default target pack at TP=1 (ws-v8: the same fp8 head, `build_devfinal.sh --tp 1`).
pub const DEFAULT_PACK_TP1: &str = "/packs/qwen38-27b-nvfp4-devfinal-tp1-fp8head";
/// The drafter pack whose rerank reads the fp8 record, TP=1 (ws-v8).
pub const DEFAULT_DRAFTER_PACK_TP1: &str = "/packs/qwen38-dflash2-drafter-tp1-fp8head";
/// The MXFP4 control's target pack (TP=2).
pub const MXFP4_PACK: &str = "/packs/qwen38-27b-nvfp4-devfinal";

/// What the law reads of one pack record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeadRecord {
    /// The record's layout id.
    pub layout_id: u32,
    /// Rank-local shape.
    pub shape: Vec<u64>,
    /// Payload bytes.
    pub length: u64,
    /// `sha256` over the payload, from the pack's own table.
    pub sha256: [u8; 32],
}

impl HeadRecord {
    /// Record `name` of `reader`, if the pack has one.
    #[must_use]
    pub fn of(reader: &PackReader, name: &str) -> Option<Self> {
        let r = &reader.records()[reader.index_of(name)?];
        Some(HeadRecord {
            layout_id: r.layout_id,
            shape: r.shape.clone(),
            length: r.length,
            sha256: r.sha256,
        })
    }
}

/// Which `lm_head` the target plans. **THE DEFAULT IS [`TargetHead::Exact`] = the checkpoint's
/// own head, as the pack carries it** (N6). `Bf16` / `Fp8` are the two RESOLVED exact kinds (what
/// the leader tells rank 1, and what the graph is built from); `Mxfp4` is the pairing control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TargetHead {
    /// The default: the exact head in whatever kind the target pack carries — resolved against
    /// the pack ([`resolve_target_head`]) before a graph is built.
    #[default]
    Exact,
    /// The exact head as the bf16 record `lm_head.w` (N2/N5's pack, `--head bf16`).
    Bf16,
    /// The exact head as the checkpoint's fp8 record `lm_head.w8` + `lm_head.ws` (N4; the
    /// default's resolution on this checkpoint since N6).
    Fp8,
    /// `--mxfp4-head`: the MXFP4 trio, the head before N5, kept as the pairing control.
    Mxfp4,
}

impl TargetHead {
    /// The value the leader forwards to rank 1 (`saber engine-worker --head <this>`); the leader
    /// forwards a RESOLVED kind, never `exact`.
    #[must_use]
    pub fn as_arg(self) -> &'static str {
        match self {
            TargetHead::Exact => "exact",
            TargetHead::Bf16 => "bf16",
            TargetHead::Fp8 => "fp8",
            TargetHead::Mxfp4 => "mxfp4",
        }
    }

    /// The inverse of [`TargetHead::as_arg`].
    ///
    /// # Errors
    /// Any other spelling, by name.
    pub fn from_arg(s: &str) -> Result<Self, String> {
        match s {
            "exact" => Ok(TargetHead::Exact),
            "bf16" => Ok(TargetHead::Bf16),
            "fp8" => Ok(TargetHead::Fp8),
            "mxfp4" => Ok(TargetHead::Mxfp4),
            other => Err(format!(
                "`--head {other}`: the head is `exact` (the pack's), `bf16`, `fp8` or `mxfp4`"
            )),
        }
    }

    /// The records this RESOLVED kind's target plan binds for its head matrix.
    #[must_use]
    pub fn target_records(self) -> &'static [&'static str] {
        match self {
            TargetHead::Bf16 => &[TARGET_HEAD],
            TargetHead::Fp8 => &[TARGET_FP8_HEAD, TARGET_FP8_SCALE],
            TargetHead::Exact | TargetHead::Mxfp4 => &[],
        }
    }

    /// The drafter's rerank kind that shares this head's records, when one does.
    #[must_use]
    pub fn rerank(self) -> Option<saber_model::dflash_graph::RerankHead> {
        match self {
            TargetHead::Bf16 => Some(saber_model::dflash_graph::RerankHead::Bf16),
            TargetHead::Fp8 => Some(saber_model::dflash_graph::RerankHead::W8),
            TargetHead::Exact | TargetHead::Mxfp4 => None,
        }
    }
}

/// The head kind a target pack CARRIES, from its `lm_head.w` / `lm_head.w8` records.
///
/// # Errors
/// A pack with neither record.
pub fn pack_head_kind(
    w: Option<&HeadRecord>,
    w8: Option<&HeadRecord>,
    pack: &str,
) -> Result<TargetHead, String> {
    if let Some(r) = w8 {
        if r.layout_id == LAYOUT_FP8_CHANNEL {
            return Ok(TargetHead::Fp8);
        }
        return Err(format!(
            "the target pack `{pack}` has `{TARGET_FP8_HEAD}` in layout {} (the fp8 head is \
             layout {LAYOUT_FP8_CHANNEL})",
            r.layout_id
        ));
    }
    match w {
        Some(r) if r.layout_id == LAYOUT_BF16_FROM_FP8_CHANNEL => Ok(TargetHead::Bf16),
        Some(_) => Ok(TargetHead::Mxfp4),
        None => Err(format!(
            "the target pack `{pack}` has no head record (`{TARGET_HEAD}` or `{TARGET_FP8_HEAD}`)"
        )),
    }
}

/// The head the run plans against the pack's, BEFORE anything is uploaded or planned: the default
/// resolves to the pack's exact kind; an explicit kind must match it.
///
/// # Errors
/// The default over the MXFP4 pack names `--mxfp4-head` and the fp8-head pack; `--mxfp4-head` over
/// an exact-head pack names the flag and the MXFP4 pack; an explicit `bf16`/`fp8` (rank 1's
/// `--head`) over a pack of the other kind names both.
pub fn resolve_target_head(
    head: TargetHead,
    w: Option<&HeadRecord>,
    w8: Option<&HeadRecord>,
    pack: &str,
) -> Result<TargetHead, String> {
    let kind = pack_head_kind(w, w8, pack)?;
    match (head, kind) {
        (TargetHead::Exact, TargetHead::Bf16 | TargetHead::Fp8) => Ok(kind),
        (TargetHead::Exact, _) => Err(format!(
            "the target pack `{pack}` has no exact head record: its `{TARGET_HEAD}` is the MXFP4 \
             head. The default head is the CHECKPOINT's own (the user's law of 2026-09-22: \
             served at its own precision, never requantized; per-channel fp8 for this \
             checkpoint): point the pack at `{DEFAULT_PACK}` (TP=1 `{DEFAULT_PACK_TP1}`; \
             the hosted packs carry it (`tools/fetch_packs.sh`)), or pass \
             `--mxfp4-head` to plan the MXFP4 head"
        )),
        (TargetHead::Mxfp4, TargetHead::Mxfp4) => Ok(TargetHead::Mxfp4),
        (TargetHead::Mxfp4, k) => Err(format!(
            "`--mxfp4-head` was given and the target pack `{pack}` carries the exact {} head: drop \
             the flag (the pack's exact head is the default), or point the pack at the MXFP4 one \
             (`{MXFP4_PACK}`, TP=1 `/packs/qwen38-27b-nvfp4-devfinal-tp1`)",
            k.as_arg()
        )),
        (TargetHead::Bf16, TargetHead::Bf16) | (TargetHead::Fp8, TargetHead::Fp8) => Ok(head),
        (h, k) => Err(format!(
            "the run plans the {} head (`--head {}`) and the target pack `{pack}` carries the {} \
             head: the two ranks must be given the same pack",
            h.as_arg(),
            h.as_arg(),
            k.as_arg()
        )),
    }
}

/// [`resolve_target_head`], returning the resolved kind's name (`"bf16"`, `"fp8"`, `"mxfp4"`).
///
/// # Errors
/// As [`resolve_target_head`].
pub fn check_target_head(
    head: TargetHead,
    w: Option<&HeadRecord>,
    w8: Option<&HeadRecord>,
    pack: &str,
) -> Result<&'static str, String> {
    resolve_target_head(head, w, w8, pack).map(TargetHead::as_arg)
}

/// [`resolve_target_head`] over rank `rank`'s file of the pack directory `pack` (or `pack` itself
/// when it is one file, as `device::pack_file` reads it) — what the CLI and
/// `device::spawn` call to turn the default into a kind before the graph exists.
///
/// # Errors
/// The pack cannot be opened, or the resolution refuses.
pub fn resolve_pack(head: TargetHead, pack: &Path, rank: u32) -> Result<TargetHead, String> {
    let file = if pack.is_dir() {
        pack.join(format!("rank{rank}.pack"))
    } else {
        pack.to_path_buf()
    };
    let reader =
        PackReader::open(&file).map_err(|e| format!("opening `{}`: {e}", file.display()))?;
    resolve_target_head(
        head,
        HeadRecord::of(&reader, TARGET_HEAD).as_ref(),
        HeadRecord::of(&reader, TARGET_FP8_HEAD).as_ref(),
        &pack.display().to_string(),
    )
}

/// The rerank kind a drafter pack carries (`head.rerank.w8` -> fp8, else bf16).
#[must_use]
pub fn drafter_rerank_kind(reader: &PackReader) -> saber_model::dflash_graph::RerankHead {
    if reader.index_of(DRAFTER_FP8_HEAD).is_some() {
        saber_model::dflash_graph::RerankHead::W8
    } else {
        saber_model::dflash_graph::RerankHead::Bf16
    }
}

/// The target graph a run plans at `world` ranks and verify `width`, with the RESOLVED head it
/// asked for: the one place a [`TargetHead`] becomes a graph (`device::engine_graph` calls it).
///
/// # Errors
/// The graph's own construction errors, and an unresolved [`TargetHead::Exact`] (resolve it
/// against the pack first).
pub fn target_graph(
    head: TargetHead,
    world: u32,
    width: u32,
) -> Result<saber_model::Graph, saber_model::ModelError> {
    let g = saber_model::Graph::qwen38_27b_at_width(
        saber_model::ModelConfig::qwen38_27b(),
        world,
        width,
    )?;
    match head {
        TargetHead::Exact => Err(saber_model::ModelError::Config(
            "the default head is the pack's exact head and has not been resolved against the \
             pack (`head_share::resolve_pack`)"
                .into(),
        )),
        TargetHead::Bf16 => g.with_bf16_head(),
        TargetHead::Fp8 => g.with_fp8_head(),
        TargetHead::Mxfp4 => Ok(g),
    }
}

/// The first eight bytes of a `sha256`, in hex (the census and the refusals print it).
#[must_use]
pub fn hex8(s: &[u8; 32]) -> String {
    s.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// One shared record: the target's name, the drafter's name, the layout both carry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedRecord {
    /// The target plan's record name (`lm_head.w8`).
    pub target: &'static str,
    /// The drafter's record it binds (`head.rerank.w8`).
    pub drafter: &'static str,
    /// The layout both carry.
    pub layout_id: u32,
    /// Payload bytes (one rank).
    pub bytes: u64,
    /// The one record's `sha256` (the drafter's, equal to the target's).
    pub sha256: [u8; 32],
}

/// The shared head, once the law has passed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedHead {
    /// The resolved kind (`Bf16` or `Fp8`).
    pub kind: TargetHead,
    /// The bytes the target no longer uploads (one rank's head records).
    pub bytes: u64,
    /// Each shared record, in pack order.
    pub records: Vec<SharedRecord>,
}

/// The (target, drafter, layout) record pairs a resolved exact kind shares.
fn pairs(kind: TargetHead) -> &'static [(&'static str, &'static str, u32)] {
    match kind {
        TargetHead::Bf16 => &[(TARGET_HEAD, DRAFTER_HEAD, LAYOUT_BF16_FROM_FP8_CHANNEL)],
        TargetHead::Fp8 => &[
            (TARGET_FP8_HEAD, DRAFTER_FP8_HEAD, LAYOUT_FP8_CHANNEL),
            (
                TARGET_FP8_SCALE,
                DRAFTER_FP8_SCALE,
                LAYOUT_F32_CHANNEL_SCALE,
            ),
        ],
        TargetHead::Exact | TargetHead::Mxfp4 => &[],
    }
}

/// Apply the law to the target's weight set: on success the target's head records are removed
/// from `want` (they are bound to the drafter's records after the drafter's upload) and the shared
/// records returned. `target` / `drafter` look a record up by name in each pack.
///
/// # Errors
/// Every refusal the module lists, each naming the record and the pack it read. A refusal leaves
/// `want` untouched.
pub fn share_head(
    kind: TargetHead,
    want: &mut BTreeSet<String>,
    target: &dyn Fn(&str) -> Option<HeadRecord>,
    drafter: &dyn Fn(&str) -> Option<HeadRecord>,
) -> Result<SharedHead, String> {
    let ps = pairs(kind);
    if ps.is_empty() {
        return Err(format!(
            "the {} head is not an exact head record and is not shared",
            kind.as_arg()
        ));
    }
    let other = if kind == TargetHead::Fp8 {
        (DRAFTER_HEAD, "bf16")
    } else {
        (DRAFTER_FP8_HEAD, "fp8")
    };
    let mut out = Vec::new();
    for &(t_name, d_name, layout) in ps {
        if !want.contains(t_name) {
            return Err(format!(
                "the {} head is shared with the drafter, but the target's plans bind no \
                 `{t_name}`",
                kind.as_arg()
            ));
        }
        let d = drafter(d_name).ok_or_else(|| {
            if drafter(other.0).is_some() {
                format!(
                    "the target serves the {} head and the drafter pack's rerank reads the {} \
                     record (`{}`), not `{d_name}`: the shared record's kind must match. Point \
                     the drafter at {}",
                    kind.as_arg(),
                    other.1,
                    other.0,
                    if kind == TargetHead::Fp8 {
                        format!(
                            "`{DEFAULT_DRAFTER_PACK}` (TP=1 `{DEFAULT_DRAFTER_PACK_TP1}`; \
                             `the internal reference harness`, `--head auto`)"
                        )
                    } else {
                        "the bf16 drafter pack (`/packs/qwen38-dflash2-drafter`, TP=1 \
                         `/packs/qwen38-dflash2-drafter-tp1`)"
                            .to_owned()
                    }
                )
            } else {
                format!(
                    "the {} head is shared with the resident drafter, and the drafter's pack has \
                     no `{d_name}`",
                    kind.as_arg()
                )
            }
        })?;
        if d.layout_id != layout {
            return Err(format!(
                "the drafter's `{d_name}` is layout {} and the {} head reads layout {layout}",
                d.layout_id,
                kind.as_arg()
            ));
        }
        let t = target(t_name).ok_or_else(|| {
            format!("the target's pack has no `{t_name}` to check the drafter's `{d_name}` against")
        })?;
        if t.layout_id != layout {
            return Err(format!(
                "the target's `{t_name}` is layout {} (not the {} head's {layout}): the shared \
                 head is checked against the target's own record",
                t.layout_id,
                kind.as_arg()
            ));
        }
        if t.shape != d.shape || t.length != d.length {
            return Err(format!(
                "the target's `{t_name}` is {:?} ({} B) and the drafter's `{d_name}` is {:?} ({} \
                 B): not the same rank's shard",
                t.shape, t.length, d.shape, d.length
            ));
        }
        if t.sha256 != d.sha256 {
            return Err(format!(
                "the target's `{t_name}` (sha256 {}…) and the drafter's `{d_name}` (sha256 {}…) \
                 differ: two different matrices, refusing to score the target with the \
                 drafter's",
                hex8(&t.sha256),
                hex8(&d.sha256)
            ));
        }
        out.push(SharedRecord {
            target: t_name,
            drafter: d_name,
            layout_id: layout,
            bytes: t.length,
            sha256: d.sha256,
        });
    }
    for r in &out {
        want.remove(r.target);
    }
    Ok(SharedHead {
        kind,
        bytes: out.iter().map(|r| r.bytes).sum(),
        records: out,
    })
}

/// N3's bf16 law, kept as the one-record form of [`share_head`].
///
/// # Errors
/// As [`share_head`] at [`TargetHead::Bf16`].
pub fn share_bf16_head(
    want: &mut BTreeSet<String>,
    target: Option<&HeadRecord>,
    drafter: Option<&HeadRecord>,
) -> Result<SharedHead, String> {
    let (t, d) = (target.cloned(), drafter.cloned());
    share_head(
        TargetHead::Bf16,
        want,
        &|n| (n == TARGET_HEAD).then(|| t.clone()).flatten(),
        &|n| (n == DRAFTER_HEAD).then(|| d.clone()).flatten(),
    )
}

// ---- THE SHARED INPUT EMBEDDING (P27-9b) -------------------------------------------------------
//
// The drafter's `input_embedding.weight` is the target's `model.language_model.embed_tokens.weight`
// byte for byte (the drafter pack builds it from `target:` that record; P27-9 measured the two
// tables' sha256 equal on both ranks of the TP=2 pair and on the TP=1 pair). Both were uploaded:
// 1,271,398,400 B a rank at TP=2, resident twice. With the share ON the DRAFTER does not upload
// its copy: its embedding launch reads the target's record at the address the target's upload gave
// it, and the drafter's weight line is billed without it. The TARGET's record is the reference (the
// target always uploads its embedding); a drafter pack whose record differs in layout, shape,
// length or sha256 is a refusal at boot, by name. `--engine-draft-embed-share off` = the control
// (two copies, the law before P27-9b).

/// The target's embedding record.
pub const TARGET_EMBED: &str = "model.language_model.embed_tokens.weight";
/// The drafter's embedding record: the same bytes as [`TARGET_EMBED`].
pub const DRAFTER_EMBED: &str = "input_embedding.weight";

/// The shared embedding, once the law has passed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedEmbed {
    /// The layout both records carry.
    pub layout_id: u32,
    /// The bytes the drafter no longer uploads (one rank's record).
    pub bytes: u64,
    /// The one record's `sha256`.
    pub sha256: [u8; 32],
}

/// The embedding law over the two packs' records: `Ok` when the drafter's [`DRAFTER_EMBED`] is
/// the target's [`TARGET_EMBED`] byte for byte (layout, shape, length and the table's sha256).
///
/// # Errors
/// A record missing on either side, or any field differing -- each naming the record, both packs
/// and the control flag.
pub fn share_embed(
    target: Option<&HeadRecord>,
    drafter: Option<&HeadRecord>,
    target_pack: &str,
    drafter_pack: &str,
) -> Result<SharedEmbed, String> {
    let ctl = "`--engine-draft-embed-share off` uploads both copies (the control)";
    let t = target.ok_or_else(|| {
        format!(
            "the shared embedding: the target pack `{target_pack}` has no `{TARGET_EMBED}` to \
             check the drafter's `{DRAFTER_EMBED}` against; {ctl}"
        )
    })?;
    let d = drafter.ok_or_else(|| {
        format!(
            "the shared embedding: the drafter pack `{drafter_pack}` has no `{DRAFTER_EMBED}`; \
             {ctl}"
        )
    })?;
    if t.layout_id != d.layout_id {
        return Err(format!(
            "the shared embedding: the target's `{TARGET_EMBED}` (`{target_pack}`) is layout {} \
             and the drafter's `{DRAFTER_EMBED}` (`{drafter_pack}`) is layout {}: refusing to \
             read one as the other; {ctl}",
            t.layout_id, d.layout_id
        ));
    }
    if t.shape != d.shape || t.length != d.length {
        return Err(format!(
            "the shared embedding: the target's `{TARGET_EMBED}` is {:?} ({} B) and the \
             drafter's `{DRAFTER_EMBED}` is {:?} ({} B): not the same rank's shard \
             (`{target_pack}` vs `{drafter_pack}`); {ctl}",
            t.shape, t.length, d.shape, d.length
        ));
    }
    if t.sha256 != d.sha256 {
        return Err(format!(
            "the shared embedding: the target's `{TARGET_EMBED}` (sha256 {}..., `{target_pack}`) \
             and the drafter's `{DRAFTER_EMBED}` (sha256 {}..., `{drafter_pack}`) differ: two \
             different tables, refusing to feed the drafter the target's; {ctl}",
            hex8(&t.sha256),
            hex8(&d.sha256)
        ));
    }
    Ok(SharedEmbed {
        layout_id: t.layout_id,
        bytes: t.length,
        sha256: t.sha256,
    })
}

/// A pack's upload windows with record `skip` left out: contiguous `(start_record, span_bytes)`
/// runs over the table (ascending by `data_off`), each one `saber_mem::upload` window
/// (`start_record`, `limit_bytes`). `skip = None` is the whole pack in one window.
///
/// `ends[i]` = record `i`'s (`data_off`, aligned end).
#[must_use]
pub fn upload_windows(ends: &[(u64, u64)], skip: Option<usize>) -> Vec<(usize, u64)> {
    let mut out = Vec::new();
    let mut push = |lo: usize, hi: usize| {
        if lo < hi {
            out.push((lo, ends[hi - 1].1 - ends[lo].0));
        }
    };
    match skip {
        Some(e) if e < ends.len() => {
            push(0, e);
            push(e + 1, ends.len());
        }
        _ => push(0, ends.len()),
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use saber_exec::layer::{BlockPlan, PlanOptions, RuntimeParams};
    use saber_kernels::rules::LaunchRules;
    use saber_kernels::saber::SaberRules;
    use saber_model::dflash_graph::RerankHead;
    use saber_model::graph::Bucket;
    use saber_model::{Graph, KernelOp};
    use std::collections::BTreeMap;

    use crate::draft_tier::DraftTier;

    fn erec(sha: u8) -> HeadRecord {
        HeadRecord {
            layout_id: 0,
            shape: vec![124_160, 5_120],
            length: 1_271_398_400,
            sha256: [sha; 32],
        }
    }

    #[test]
    fn p27_9b_embed_share_passes_identical_and_refuses_by_name() {
        let ok = share_embed(Some(&erec(7)), Some(&erec(7)), "T", "D").expect("identical");
        assert_eq!(ok.bytes, 1_271_398_400);
        assert_eq!(ok.layout_id, 0);
        // NEGATIVE CONTROLS: every field that differs is a refusal naming the record.
        let e = share_embed(Some(&erec(7)), Some(&erec(8)), "T", "D").unwrap_err();
        assert!(
            e.contains(DRAFTER_EMBED) && e.contains("differ") && e.contains("share off"),
            "{e}"
        );
        let mut lay = erec(7);
        lay.layout_id = 14;
        let e = share_embed(Some(&erec(7)), Some(&lay), "T", "D").unwrap_err();
        assert!(e.contains("layout 14"), "{e}");
        let mut half = erec(7);
        half.shape = vec![62_080, 5_120];
        half.length /= 2;
        let e = share_embed(Some(&erec(7)), Some(&half), "T", "D").unwrap_err();
        assert!(e.contains("same rank's shard"), "{e}");
        let e = share_embed(None, Some(&erec(7)), "T", "D").unwrap_err();
        assert!(e.contains(TARGET_EMBED) && e.contains("`T`"), "{e}");
        let e = share_embed(Some(&erec(7)), None, "T", "D").unwrap_err();
        assert!(e.contains(DRAFTER_EMBED) && e.contains("`D`"), "{e}");
    }

    /// The served pairs on this box (TP=2 both ranks, TP=1): the embedding law passes, the window
    /// law leaves out exactly the embedding's bytes, and the 0.6 GB head is NOT the embedding
    /// (the negative control: the law refuses the drafter's head record as the target's table).
    #[test]
    fn p27_9b_the_real_pairs_share_one_embedding() {
        let root = std::path::Path::new("/mnt/nvme/saber-packs");
        let pairs = [
            (
                "qwen38-27b-nvfp4-devfinal-fp8head",
                "qwen38-dflash2-drafter-fp8head",
                2u32,
            ),
            (
                "qwen38-27b-nvfp4-devfinal-tp1-fp8head",
                "qwen38-dflash2-drafter-tp1-fp8head",
                1,
            ),
        ];
        for (t, d, ranks) in pairs {
            let (t, d) = (root.join(t), root.join(d));
            if !(t.is_dir() && d.is_dir()) {
                eprintln!("skipped: the packs are not on this box");
                return;
            }
            for r in 0..ranks {
                let tr = PackReader::open(&t.join(format!("rank{r}.pack"))).expect("target");
                let dr = PackReader::open(&d.join(format!("rank{r}.pack"))).expect("drafter");
                let s = share_embed(
                    HeadRecord::of(&tr, TARGET_EMBED).as_ref(),
                    HeadRecord::of(&dr, DRAFTER_EMBED).as_ref(),
                    "t",
                    "d",
                )
                .expect("the served pair shares its embedding");
                assert_eq!(s.bytes, 1_271_398_400 * u64::from(3 - ranks));
                let e = share_embed(
                    HeadRecord::of(&tr, TARGET_EMBED).as_ref(),
                    HeadRecord::of(&dr, DRAFTER_FP8_HEAD).as_ref(),
                    "t",
                    "d",
                )
                .expect_err("the control: another record of the drafter's");
                assert!(e.contains("shared embedding"), "{e}");
                let ends: Vec<(u64, u64)> = dr
                    .records()
                    .iter()
                    .map(|x| (x.data_off, saber_mem::pack::align_up(x.data_off + x.length)))
                    .collect();
                let whole = upload_windows(&ends, None);
                let cut = upload_windows(&ends, dr.index_of(DRAFTER_EMBED));
                assert_eq!(whole.len(), 1);
                assert_eq!(cut.len(), 2);
                let span = |w: &[(usize, u64)]| w.iter().map(|x| x.1).sum::<u64>();
                assert_eq!(
                    span(&whole) - span(&cut),
                    saber_mem::pack::align_up(s.bytes),
                    "the windows leave out the embedding's bytes and nothing else"
                );
            }
        }
    }

    #[test]
    fn p27_9b_upload_windows_skip_one_record() {
        let ends = [(0, 16), (16, 32), (32, 48), (48, 64)];
        assert_eq!(upload_windows(&ends, None), vec![(0, 64)]);
        assert_eq!(upload_windows(&ends, Some(1)), vec![(0, 16), (2, 32)]);
        assert_eq!(upload_windows(&ends, Some(0)), vec![(1, 48)]);
        assert_eq!(upload_windows(&ends, Some(3)), vec![(0, 48)]);
        // A skip past the table is the whole pack (the caller refuses a missing record first).
        assert_eq!(upload_windows(&ends, Some(9)), vec![(0, 64)]);
    }

    /// The target's head weight keys at `rows`, from the real planner (the engine's own `want`).
    fn head_keys(g: &Graph, rows: u32) -> BTreeSet<String> {
        BlockPlan::with_options(
            g,
            Bucket::Decode {
                batch: rows,
                verify_width: 1,
            },
            &[],
            RuntimeParams {
                max_ctx: 4096,
                gdn_slots: 1,
            },
            &LaunchRules::prod_mxfp4().expect("rules"),
            &SaberRules::prod_saber().expect("saber rules"),
            PlanOptions {
                rank: 0,
                ..PlanOptions::HEAD
            },
        )
        .expect("head plan")
        .weight_keys()
        .into_iter()
        .collect()
    }

    fn rec(layout_id: u32, sha: u8) -> HeadRecord {
        HeadRecord {
            layout_id,
            shape: vec![124_160, 5120],
            length: 124_160 * 5120 * 2,
            sha256: [sha; 32],
        }
    }

    /// One TP=2 rank's fp8 records: the e4m3 matrix and its f32 scale.
    fn fp8_recs(sha: u8) -> BTreeMap<&'static str, HeadRecord> {
        [
            (
                "w8",
                HeadRecord {
                    layout_id: LAYOUT_FP8_CHANNEL,
                    shape: vec![124_160, 5120],
                    length: 124_160 * 5120,
                    sha256: [sha; 32],
                },
            ),
            (
                "ws",
                HeadRecord {
                    layout_id: LAYOUT_F32_CHANNEL_SCALE,
                    shape: vec![124_160],
                    length: 124_160 * 4,
                    sha256: [sha ^ 0x55; 32],
                },
            ),
        ]
        .into_iter()
        .collect()
    }

    fn lookup(
        m: BTreeMap<&'static str, HeadRecord>,
        prefix: &'static str,
    ) -> impl Fn(&str) -> Option<HeadRecord> {
        move |n: &str| {
            n.strip_prefix(prefix)
                .and_then(|s| s.strip_prefix('.'))
                .and_then(|s| m.get(s).cloned())
        }
    }

    fn heads(set: &BTreeSet<String>) -> Vec<String> {
        set.iter()
            .filter(|n| n.starts_with("lm_head") || n.starts_with("head.rerank"))
            .cloned()
            .collect()
    }

    /// N3's law over the real record sets: the bf16 target's head plans plus the resident bf16
    /// drafter hold ONE head record once shared — the drafter's. CONTROL: the union unshared.
    #[test]
    fn a_shared_bf16_head_is_one_record_and_it_is_the_drafters() {
        let g = target_graph(TargetHead::Bf16, 2, 8).expect("bf16 graph");
        let tier = DraftTier::plan(2, 8, 4096, 8).expect("drafter tier");
        let drafter_names = tier.record_names(4096);
        assert!(drafter_names.contains(DRAFTER_HEAD));
        for rows in [1, 8, 32, 64] {
            let mut want = head_keys(&g, rows);
            let control: BTreeSet<String> = want.union(&drafter_names).cloned().collect();
            assert_eq!(
                heads(&control),
                vec![DRAFTER_HEAD, TARGET_HEAD],
                "rows {rows}"
            );
            let d = rec(LAYOUT_BF16_FROM_FP8_CHANNEL, 0xd1);
            let shared = share_bf16_head(&mut want, Some(&d.clone()), Some(&d)).expect("shares");
            assert_eq!(shared.records[0].sha256, d.sha256);
            assert_eq!(shared.bytes, 1_271_398_400, "one TP=2 rank's bf16 head");
            let union: BTreeSet<String> = want.union(&drafter_names).cloned().collect();
            assert_eq!(heads(&union), vec![DRAFTER_HEAD], "rows {rows}");
            assert_eq!(control.len(), union.len() + 1);
        }
    }

    /// N6: THE DEFAULT on this checkpoint — the fp8 head's two records and the resident fp8-rerank
    /// drafter's two are ONE pair on the card, the drafter's, and 0.636 GB of it. CONTROL: the
    /// union unshared holds four head records.
    #[test]
    fn a_shared_fp8_head_is_one_record_pair_and_it_is_the_drafters() {
        let g = target_graph(TargetHead::Fp8, 2, 8).expect("fp8 graph");
        let tier = DraftTier::plan_with(2, 8, 4096, 8, RerankHead::W8).expect("fp8 drafter tier");
        let drafter_names = tier.record_names(4096);
        assert!(
            drafter_names.contains(DRAFTER_FP8_HEAD) && drafter_names.contains(DRAFTER_FP8_SCALE)
        );
        assert!(
            !drafter_names.contains(DRAFTER_HEAD),
            "no bf16 record beside the fp8 one"
        );
        let f8 = fp8_recs(0xf8);
        for rows in [1, 8, 32, 64] {
            let mut want = head_keys(&g, rows);
            assert_eq!(
                heads(&want),
                vec![TARGET_FP8_HEAD, TARGET_FP8_SCALE],
                "rows {rows}"
            );
            let control: BTreeSet<String> = want.union(&drafter_names).cloned().collect();
            assert_eq!(heads(&control).len(), 4, "rows {rows}");
            let s = share_head(
                TargetHead::Fp8,
                &mut want,
                &lookup(f8.clone(), "lm_head"),
                &lookup(f8.clone(), "head.rerank"),
            )
            .expect("the default shares");
            assert_eq!(
                s.bytes,
                635_699_200 + 496_640,
                "one TP=2 rank's fp8 head + scale"
            );
            assert_eq!(s.records.len(), 2);
            let union: BTreeSet<String> = want.union(&drafter_names).cloned().collect();
            assert_eq!(
                heads(&union),
                vec![DRAFTER_FP8_HEAD, DRAFTER_FP8_SCALE],
                "rows {rows}"
            );
            assert_eq!(control.len(), union.len() + 2);
        }
        // The drafter tier's billed weights drop by the bf16 record and gain the fp8 pair.
        let bf = DraftTier::plan(2, 8, 4096, 8)
            .expect("bf16 tier")
            .demand()
            .weights;
        let w8 = tier.demand().weights;
        assert_eq!(bf - w8, 1_271_398_400 - 635_699_200 - 496_640);
    }

    /// Every refusal, each by name, and none of them touches the target's set — bf16 and fp8.
    #[test]
    fn the_shared_head_refuses_by_name_and_leaves_the_set_alone() {
        let good = rec(LAYOUT_BF16_FROM_FP8_CHANNEL, 7);
        let base: BTreeSet<String> = [TARGET_HEAD, "model.language_model.norm.weight"]
            .into_iter()
            .map(String::from)
            .collect();
        let cases: [(Option<HeadRecord>, Option<HeadRecord>, &str); 5] = [
            (Some(good.clone()), None, "has no `head.rerank.w`"),
            (
                Some(good.clone()),
                Some(rec(LAYOUT_BF16_FROM_FP8_CHANNEL, 8)),
                "differ",
            ),
            (Some(good.clone()), Some(rec(1, 7)), "is layout 1"),
            (
                Some(rec(1, 7)),
                Some(good.clone()),
                "target's `lm_head.w` is layout 1",
            ),
            (None, Some(good.clone()), "has no `lm_head.w`"),
        ];
        for (t, d, why) in cases {
            let mut want = base.clone();
            let e = share_bf16_head(&mut want, t.as_ref(), d.as_ref()).expect_err(why);
            assert!(e.contains(why), "{e}");
            assert_eq!(want, base, "a refusal moved the set ({why})");
        }
        let mut half = good.clone();
        half.shape = vec![248_320, 5120];
        let mut want = base.clone();
        let e = share_bf16_head(&mut want, Some(&good), Some(&half)).expect_err("a TP=1 drafter");
        assert!(e.contains("not the same rank's shard"), "{e}");
        let mut none: BTreeSet<String> = BTreeSet::new();
        assert!(share_bf16_head(&mut none, Some(&good), Some(&good)).is_err());

        // fp8: a KIND mismatch in both directions names the other kind and the pack to use; a
        // scale whose sha differs is refused even though the matrix matches (both are checked).
        let f8 = fp8_recs(3);
        let fbase: BTreeSet<String> = [TARGET_FP8_HEAD, TARGET_FP8_SCALE]
            .into_iter()
            .map(String::from)
            .collect();
        let bf_drafter = |n: &str| (n == DRAFTER_HEAD).then(|| good.clone());
        let mut want = fbase.clone();
        let e = share_head(
            TargetHead::Fp8,
            &mut want,
            &lookup(f8.clone(), "lm_head"),
            &bf_drafter,
        )
        .expect_err("a bf16 drafter under the fp8 head");
        assert!(
            e.contains("kind must match") && e.contains(DEFAULT_DRAFTER_PACK),
            "{e}"
        );
        assert_eq!(want, fbase);
        let mut want = base.clone();
        let e = share_head(
            TargetHead::Bf16,
            &mut want,
            &|n| (n == TARGET_HEAD).then(|| good.clone()),
            &lookup(f8.clone(), "head.rerank"),
        )
        .expect_err("an fp8 drafter under the bf16 head");
        assert!(e.contains("kind must match") && e.contains("fp8"), "{e}");
        assert_eq!(want, base);
        let mut bad_scale = f8.clone();
        bad_scale.get_mut("ws").expect("ws").sha256 = [0; 32];
        let mut want = fbase.clone();
        let e = share_head(
            TargetHead::Fp8,
            &mut want,
            &lookup(f8.clone(), "lm_head"),
            &lookup(bad_scale, "head.rerank"),
        )
        .expect_err("the scales differ");
        assert!(e.contains("`lm_head.ws`") && e.contains("differ"), "{e}");
        assert_eq!(want, fbase, "a refusal on the second record moved the set");
    }

    /// N6: the DEFAULT resolves to the kind the PACK carries — fp8 on the fp8 pack (the graph
    /// carries `GemmFp8Head`), bf16 on the bf16 pack (`GemmBf16Head`), and the MXFP4 pack is a
    /// refusal naming `--mxfp4-head`. The control is `--mxfp4-head` on each.
    #[test]
    fn the_default_resolves_to_the_packs_head() {
        assert_eq!(TargetHead::default(), TargetHead::Exact);
        for h in [
            TargetHead::Exact,
            TargetHead::Bf16,
            TargetHead::Fp8,
            TargetHead::Mxfp4,
        ] {
            assert_eq!(TargetHead::from_arg(h.as_arg()), Ok(h));
        }
        assert!(TargetHead::from_arg("int4").is_err());
        let bf = rec(LAYOUT_BF16_FROM_FP8_CHANNEL, 1);
        let mx = rec(saber_model::graph::LAYOUT_MXFP4_WPERM, 2);
        let w8 = fp8_recs(3)["w8"].clone();
        let head_op = |h: TargetHead| {
            target_graph(h, 2, 8)
                .expect("graph")
                .ops
                .iter()
                .find(|o| o.name == "lm_head")
                .expect("lm_head")
                .kernel
        };
        // the fp8 pack: `lm_head.w8`, and the MXFP4 `lm_head.ws`-style names beside it do not
        // confuse it (the fp8 pack has no `lm_head.w` at all)
        let r = resolve_target_head(TargetHead::Exact, None, Some(&w8), "fp8").expect("fp8");
        assert_eq!(r, TargetHead::Fp8);
        assert_eq!(head_op(r), KernelOp::GemmFp8Head);
        let r = resolve_target_head(TargetHead::Exact, Some(&bf), None, "bf").expect("bf16");
        assert_eq!(r, TargetHead::Bf16);
        assert_eq!(head_op(r), KernelOp::GemmBf16Head);
        let e = resolve_target_head(TargetHead::Exact, Some(&mx), None, "/packs/mx").unwrap_err();
        for needle in [
            "/packs/mx",
            "no exact head record",
            "--mxfp4-head",
            DEFAULT_PACK,
        ] {
            assert!(e.contains(needle), "{needle}: {e}");
        }
        assert!(target_graph(TargetHead::Exact, 2, 8).is_err(), "unresolved");
        // --mxfp4-head: the control
        assert_eq!(
            resolve_target_head(TargetHead::Mxfp4, Some(&mx), None, "mx"),
            Ok(TargetHead::Mxfp4)
        );
        assert_ne!(head_op(TargetHead::Mxfp4), KernelOp::GemmFp8Head);
        for (w, w8_, what) in [(Some(&bf), None, "bf16"), (None, Some(&w8), "fp8")] {
            let e = resolve_target_head(TargetHead::Mxfp4, w, w8_, "/packs/x").unwrap_err();
            assert!(e.contains("--mxfp4-head") && e.contains(what), "{e}");
            assert!(e.contains(MXFP4_PACK), "{e}");
        }
        // rank 1's explicit kind must match its pack
        let e = resolve_target_head(TargetHead::Bf16, None, Some(&w8), "/packs/f").unwrap_err();
        assert!(e.contains("--head bf16") && e.contains("fp8"), "{e}");
        assert_eq!(
            check_target_head(TargetHead::Fp8, None, Some(&w8), "p"),
            Ok("fp8")
        );
        assert!(check_target_head(TargetHead::Exact, None, None, "p").is_err());
    }

    /// The real packs, when this box has them: the default resolves per pack; the fp8 target pack
    /// shares with the fp8-rerank drafter pack on both ranks (sha-checked, two records), the bf16
    /// target pack with the bf16 drafter pack, and each drafter pack is refused by the other kind.
    #[test]
    fn the_real_packs_resolve_and_share_on_both_ranks() {
        let root = std::path::Path::new("/mnt/nvme/saber-packs");
        let (f8, bf, mx, dr, dr8) = (
            root.join("qwen38-27b-nvfp4-devfinal-fp8head"),
            root.join("qwen38-27b-nvfp4-devfinal-bf16head"),
            root.join("qwen38-27b-nvfp4-devfinal"),
            root.join("qwen38-dflash2-drafter"),
            root.join("qwen38-dflash2-drafter-fp8head"),
        );
        if !(f8.is_dir() && bf.is_dir() && mx.is_dir() && dr.is_dir()) {
            eprintln!("skipped: the packs are not on this box");
            return;
        }
        assert_eq!(resolve_pack(TargetHead::Exact, &f8, 0), Ok(TargetHead::Fp8));
        assert_eq!(
            resolve_pack(TargetHead::Exact, &bf, 1),
            Ok(TargetHead::Bf16)
        );
        let e = resolve_pack(TargetHead::Exact, &mx, 0).expect_err("the MXFP4 pack");
        assert!(e.contains("--mxfp4-head"), "{e}");
        assert_eq!(
            resolve_pack(TargetHead::Mxfp4, &mx, 1),
            Ok(TargetHead::Mxfp4)
        );
        let open = |d: &std::path::Path, r: u32| {
            PackReader::open(&d.join(format!("rank{r}.pack"))).expect("open pack")
        };
        for r in 0..2 {
            let (t, d) = (open(&bf, r), open(&dr, r));
            assert_eq!(drafter_rerank_kind(&d), RerankHead::Bf16);
            let mut want: BTreeSet<String> = [TARGET_HEAD.to_owned()].into();
            let s = share_head(
                TargetHead::Bf16,
                &mut want,
                &|n| HeadRecord::of(&t, n),
                &|n| HeadRecord::of(&d, n),
            )
            .expect("bf16 rank shares");
            assert_eq!(s.bytes, 1_271_398_400);
            if dr8.is_dir() {
                let (t8, d8) = (open(&f8, r), open(&dr8, r));
                assert_eq!(drafter_rerank_kind(&d8), RerankHead::W8);
                let mut want: BTreeSet<String> =
                    [TARGET_FP8_HEAD.to_owned(), TARGET_FP8_SCALE.to_owned()].into();
                let s = share_head(
                    TargetHead::Fp8,
                    &mut want,
                    &|n| HeadRecord::of(&t8, n),
                    &|n| HeadRecord::of(&d8, n),
                )
                .expect("fp8 rank shares");
                assert_eq!(s.bytes, 635_699_200 + 496_640);
                assert!(want.is_empty());
                // the other kinds' drafter packs are refused by name
                let mut want: BTreeSet<String> =
                    [TARGET_FP8_HEAD.to_owned(), TARGET_FP8_SCALE.to_owned()].into();
                let e = share_head(
                    TargetHead::Fp8,
                    &mut want,
                    &|n| HeadRecord::of(&t8, n),
                    &|n| HeadRecord::of(&d, n),
                )
                .expect_err("the bf16 drafter under the fp8 head");
                assert!(e.contains("kind must match"), "{e}");
            }
        }
    }

    /// THE TP=1 DEFAULT PAIR (ws-v8), when this box has it: the fp8 target pack built with
    /// `--head auto` resolves to the fp8 head, its drafter pack's rerank is the w8 kind, and the two
    /// share ONE head record pair — same layout, shape, length and sha256 — which at TP=1 is the
    /// whole head: 1,271,398,400 B of e4m3 rows + 993,280 B of f32 scale. The bf16 TP=1 pair
    /// (the control) shares its bf16 record, and each drafter is refused under the other kind.
    #[test]
    fn the_tp1_fp8_pair_resolves_and_shares_one_head_record() {
        let root = std::path::Path::new("/mnt/nvme/saber-packs");
        let (f8, dr8, bf, dr) = (
            root.join("qwen38-27b-nvfp4-devfinal-tp1-fp8head"),
            root.join("qwen38-dflash2-drafter-tp1-fp8head"),
            root.join("qwen38-27b-nvfp4-devfinal-tp1-bf16head"),
            root.join("qwen38-dflash2-drafter-tp1"),
        );
        if !(f8.is_dir() && dr8.is_dir() && bf.is_dir() && dr.is_dir()) {
            eprintln!("skipped: the TP=1 packs are not on this box");
            return;
        }
        assert!(DEFAULT_PACK_TP1.ends_with("qwen38-27b-nvfp4-devfinal-tp1-fp8head"));
        assert!(DEFAULT_DRAFTER_PACK_TP1.ends_with("qwen38-dflash2-drafter-tp1-fp8head"));
        assert_eq!(resolve_pack(TargetHead::Exact, &f8, 0), Ok(TargetHead::Fp8));
        assert_eq!(
            resolve_pack(TargetHead::Exact, &bf, 0),
            Ok(TargetHead::Bf16)
        );
        let open = |d: &std::path::Path| PackReader::open(&d.join("rank0.pack")).expect("open");
        let (t8, d8, tb, db) = (open(&f8), open(&dr8), open(&bf), open(&dr));
        assert_eq!(drafter_rerank_kind(&d8), RerankHead::W8);
        assert_eq!(drafter_rerank_kind(&db), RerankHead::Bf16);
        let mut want: BTreeSet<String> =
            [TARGET_FP8_HEAD.to_owned(), TARGET_FP8_SCALE.to_owned()].into();
        let s = share_head(
            TargetHead::Fp8,
            &mut want,
            &|n| HeadRecord::of(&t8, n),
            &|n| HeadRecord::of(&d8, n),
        )
        .expect("the TP=1 fp8 pair shares");
        assert_eq!(s.bytes, 1_271_398_400 + 993_280, "the whole head, once");
        assert!(
            want.is_empty(),
            "the target uploads no head record of its own"
        );
        let mut want: BTreeSet<String> = [TARGET_HEAD.to_owned()].into();
        let s = share_head(
            TargetHead::Bf16,
            &mut want,
            &|n| HeadRecord::of(&tb, n),
            &|n| HeadRecord::of(&db, n),
        )
        .expect("the TP=1 bf16 pair shares");
        assert_eq!(s.bytes, 2_542_796_800);
        // CONTROLS: each drafter under the other kind is refused by name.
        let mut want: BTreeSet<String> =
            [TARGET_FP8_HEAD.to_owned(), TARGET_FP8_SCALE.to_owned()].into();
        let e = share_head(
            TargetHead::Fp8,
            &mut want,
            &|n| HeadRecord::of(&t8, n),
            &|n| HeadRecord::of(&db, n),
        )
        .expect_err("the bf16 drafter under the fp8 head");
        assert!(
            e.contains("kind must match") && e.contains("tp1-fp8head"),
            "{e}"
        );
        let mut want: BTreeSet<String> = [TARGET_HEAD.to_owned()].into();
        let e = share_head(
            TargetHead::Bf16,
            &mut want,
            &|n| HeadRecord::of(&tb, n),
            &|n| HeadRecord::of(&d8, n),
        )
        .expect_err("the fp8 drafter under the bf16 head");
        assert!(e.contains("kind must match"), "{e}");
    }
}
