//! MACaQ (2026-11-11): which packs run the MACaQ-W4A8 Hadamard producers (`--macaq-had on|off`).
//!
//! A MACaQ-Hadamard pack stores every target linear in the TRANSFORMED domain (`W' = H32 (W / t)` per
//! 32-block of K) and carries the producers' f32 `t` records (`saber_model::graph::Graph::with_macaq_had`).
//! Serving it with the parent producers -- or serving an ordinary pack with the twins -- is wrong numerics
//! that boots cleanly, so the pair is decided HERE, once, from the pack itself: a pack that carries the
//! layer-0 `in_t` record is a MACaQ pack. An absent flag follows the pack; an explicit flag that disagrees
//! is refused by name.

use std::path::Path;

use saber_mem::pack::PackReader;

/// The record whose presence marks a MACaQ-Hadamard pack (every such pack binds it).
pub const MARK: &str = "model.language_model.layers.0.macaq_had32.in_t";

/// Does this pack (a directory of `rank<r>.pack` or one rank's file) carry the MACaQ records?
pub fn pack_is_macaq(pack: &Path) -> Result<bool, String> {
    let file = if pack.is_dir() {
        pack.join("rank0.pack")
    } else {
        pack.to_path_buf()
    };
    let reader =
        PackReader::open(&file).map_err(|e| format!("opening `{}`: {e}", file.display()))?;
    Ok(reader.index_of(MARK).is_some())
}

/// The arm this boot runs: `flag` (`--macaq-had`) when given and consistent with the pack, else the
/// pack's own answer.
pub fn resolve(flag: Option<bool>, pack: &Path) -> Result<bool, String> {
    let has = pack_is_macaq(pack)?;
    decide(flag, has, &pack.display().to_string())
}

/// [`resolve`]'s rule on its own (host-testable).
pub fn decide(flag: Option<bool>, pack_has: bool, pack: &str) -> Result<bool, String> {
    match (flag, pack_has) {
        (None, h) => Ok(h),
        (Some(true), true) | (Some(false), false) => Ok(pack_has),
        (Some(true), false) => Err(format!(
            "--macaq-had on: pack `{pack}` carries no `macaq_had32` records (`{MARK}`): it is an \
             ordinary MXFP4 pack, and the Hadamard producers would run against original-domain \
             weights. Serve a macaq_had32 pack, or drop the flag"
        )),
        (Some(false), true) => Err(format!(
            "--macaq-had off: pack `{pack}` is a `macaq_had32` pack (Hadamard-domain weights + `t` \
             records): the parent producers would run against transformed weights. Serve it with \
             --macaq-had on (or no flag), or serve an ordinary pack"
        )),
    }
}

/// the record suffix a form-c W8 promotion carries (`<op>.w8`, pack layout 7).
pub const W8_SUFFIX: &str = ".w8";

/// MACaQ (`--macaq-w8`, PACK-DRIVEN, no flag): the ops this pack serves as form-c W8 promotions --
/// every `layer.<L>.{gdn.out_proj,attn.o_proj,mlp.down_proj}.w8` record of rank 0's file (every rank's
/// file carries the same record set; the per-rank bytes are checked at upload by the plan's layouts).
/// An ordinary pack answers the empty set, so the arm is the PACK: Bc3 / Bc6 / Bc10 are three packs.
pub fn pack_w8_ops(pack: &Path) -> Result<std::collections::BTreeSet<String>, String> {
    let file = if pack.is_dir() {
        pack.join("rank0.pack")
    } else {
        pack.to_path_buf()
    };
    let reader =
        PackReader::open(&file).map_err(|e| format!("opening `{}`: {e}", file.display()))?;
    Ok(w8_ops_of(
        (0..reader.records().len()).map(|i| reader.name(i)),
    ))
}

/// [`pack_w8_ops`]'s rule on a list of record names (host-testable).
pub fn w8_ops_of<'a>(names: impl Iterator<Item = &'a str>) -> std::collections::BTreeSet<String> {
    names
        .filter_map(|n| n.strip_suffix(W8_SUFFIX))
        .filter(|op| {
            op.starts_with("layer.")
                && [".gdn.out_proj", ".attn.o_proj", ".mlp.down_proj"]
                    .iter()
                    .any(|r| op.ends_with(r))
        })
        .map(str::to_owned)
        .collect()
}

/// The record suffix of an iu8 pack's W4 ops (`<op>.wi4`, layout 17).
pub const IU8_W4_SUFFIX: &str = ".wi4";
/// The record suffix of an iu8 pack's W8 ops (`<op>.wi8`, layout 18).
pub const IU8_W8_SUFFIX: &str = ".wi8";

/// MACaQ W4A8-iu8 (the iu8 pack, PACK-DRIVEN, no flag): `None` for a pack without iu8 records, else the
/// pack's W8 op set (every other `layer.*` linear is W4). The pack's `manifest.json` (beside the rank
/// files) must name this build's format (`saber_kernels::macaq_iu8::FORMAT`) in its plan transforms; an
/// iu8 pack of another format version, an iu8 tag without the records, iu8 records without the tag, or
/// iu8 records without the MACaQ `t` records is refused by name. (An engine without iu8 support refuses
/// an iu8 pack at upload: its plan asks for the `<op>.w` MXFP4 records the pack does not carry.)
pub fn pack_iu8(pack: &Path) -> Result<Option<std::collections::BTreeSet<String>>, String> {
    let file = if pack.is_dir() {
        pack.join("rank0.pack")
    } else {
        pack.to_path_buf()
    };
    let reader =
        PackReader::open(&file).map_err(|e| format!("opening `{}`: {e}", file.display()))?;
    let manifest = file.with_file_name("manifest.json");
    // The manifest's plan transforms are JSON strings; the iu8 tag is the only `"macaq_iu8/...` string a
    // builder writes (no JSON reader in this crate's dependencies: the tags are read as quoted strings).
    let transforms = std::fs::read_to_string(&manifest)
        .ok()
        .map(|t| iu8_tags_in(&t));
    iu8_decide(
        (0..reader.records().len()).map(|i| reader.name(i)),
        transforms.as_deref(),
        reader.index_of(MARK).is_some(),
        &pack.display().to_string(),
    )
}

/// The `"macaq_iu8/..."` strings of a manifest's text (each a whole JSON string), in order.
pub fn iu8_tags_in(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find("\"macaq_iu8/") {
        let tail = &rest[i + 1..];
        let Some(j) = tail.find('"') else { break };
        out.push(tail[..j].to_owned());
        rest = &tail[j..];
    }
    out
}

/// [`pack_iu8`]'s rule (host-testable): record names, the manifest's transform list (None = no manifest),
/// whether the MACaQ `t` records are present.
pub fn iu8_decide<'a>(
    names: impl Iterator<Item = &'a str>,
    transforms: Option<&[String]>,
    had: bool,
    pack: &str,
) -> Result<Option<std::collections::BTreeSet<String>>, String> {
    let fmt = saber_kernels::macaq_iu8::FORMAT;
    let mut w4 = 0usize;
    let mut w8 = std::collections::BTreeSet::new();
    for n in names {
        if n.starts_with("layer.") && n.ends_with(IU8_W4_SUFFIX) {
            w4 += 1;
        } else if let Some(op) = n
            .strip_suffix(IU8_W8_SUFFIX)
            .filter(|o| o.starts_with("layer."))
        {
            w8.insert(op.to_owned());
        }
    }
    let tag: Vec<&String> = transforms
        .unwrap_or_default()
        .iter()
        .filter(|t| t.starts_with("macaq_iu8/"))
        .collect();
    let has_records = w4 > 0 || !w8.is_empty();
    match (has_records, tag.as_slice()) {
        (false, []) => Ok(None),
        (false, _) => Err(format!(
            "pack `{pack}`: its manifest names an iu8 format ({tag:?}) but it carries no `.wi4` / `.wi8` \
             records"
        )),
        (true, [t]) if t.as_str() == fmt => {
            if !had {
                return Err(format!(
                    "pack `{pack}`: iu8 records without the `macaq_had32` t records (`{MARK}`): the iu8 \
                     codes are in the MACaQ-Hadamard basis"
                ));
            }
            if w4 == 0 {
                return Err(format!("pack `{pack}`: iu8 W8 records but no W4 (`.wi4`) records"));
            }
            Ok(Some(w8))
        }
        (true, []) if transforms.is_none() => Err(format!(
            "pack `{pack}`: iu8 records but no manifest.json beside the rank files to name their format \
             (this engine reads `{fmt}`)"
        )),
        (true, _) => Err(format!(
            "pack `{pack}`: iu8 records of format {tag:?}; this engine serves `{fmt}` only -- rebuild the \
             pack for this engine's format, or serve it with an engine that reads its format"
        )),
    }
}

/// The iu8 DRAFTER pack's marker record (`layer.0.attn.qkv.wi4`).
pub const DRAFT_IU8_MARK: &str = "layer.0.attn.qkv.wi4";

/// Whether a DFlash2 drafter pack (its `rank<r>.pack`, `manifest.json` beside it) is a W4A8-iu8 drafter:
/// `.wi4` records AND the manifest tag [`saber_kernels::dflash_iu8::FORMAT`]. Records without the tag, a tag
/// without records, or another `macaq_iu8d/` version are refused by name; neither = the FP8 drafter (`false`).
///
/// # Errors
/// An unreadable pack, and the three mismatches.
pub fn draft_iu8(file: &Path) -> Result<bool, String> {
    let reader =
        PackReader::open(file).map_err(|e| format!("opening `{}`: {e}", file.display()))?;
    let manifest = file.with_file_name("manifest.json");
    let text = std::fs::read_to_string(&manifest).ok();
    let has = (0..reader.records().len()).any(|i| {
        let n = reader.name(i);
        n.starts_with("layer.") && n.ends_with(IU8_W4_SUFFIX)
    });
    draft_iu8_decide(has, text.as_deref(), &file.display().to_string())
}

/// [`draft_iu8`]'s rule over (records present, the manifest's text).
///
/// # Errors
/// As [`draft_iu8`].
pub fn draft_iu8_decide(
    has_records: bool,
    manifest: Option<&str>,
    pack: &str,
) -> Result<bool, String> {
    let fmt = saber_kernels::dflash_iu8::FORMAT;
    let mut tags = Vec::new();
    if let Some(t) = manifest {
        let mut rest = t;
        while let Some(i) = rest.find("\"macaq_iu8d/") {
            let tail = &rest[i + 1..];
            let Some(j) = tail.find('"') else { break };
            tags.push(tail[..j].to_owned());
            rest = &tail[j..];
        }
    }
    match (has_records, tags.as_slice()) {
        (false, []) => Ok(false),
        (false, _) => Err(format!(
            "drafter pack `{pack}`: its manifest names an iu8 drafter format ({tags:?}) but it carries no `.wi4` \
             records"
        )),
        (true, [t]) if t == fmt => Ok(true),
        (true, []) => Err(format!(
            "drafter pack `{pack}`: `.wi4` drafter records but no `{fmt}` tag in a manifest.json beside the rank \
             files"
        )),
        (true, _) => Err(format!(
            "drafter pack `{pack}`: iu8 drafter records of format {tags:?}; this engine serves `{fmt}` only"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::decide;

    #[test]
    fn the_iu8_drafter_rule_reads_the_records_and_the_tag() {
        use super::draft_iu8_decide as d;
        let fmt = saber_kernels::dflash_iu8::FORMAT;
        let man = format!(r#"{{"plan":{{"descriptor":{{"transforms":["merge_dim0","{fmt}"]}}}}}}"#);
        assert_eq!(d(true, Some(&man), "p"), Ok(true));
        assert_eq!(
            d(false, Some(r#"{"transforms":["merge_dim0"]}"#), "p"),
            Ok(false)
        );
        assert_eq!(d(false, None, "p"), Ok(false));
        assert!(d(false, Some(&man), "p").unwrap_err().contains("no `.wi4`"));
        assert!(d(true, Some("{}"), "p").unwrap_err().contains("tag"));
        assert!(d(true, None, "p").unwrap_err().contains("tag"));
        let other = r#"{"transforms":["macaq_iu8d/2 x"]}"#;
        assert!(d(true, Some(other), "p").unwrap_err().contains("serves"));
    }

    #[test]
    fn the_iu8_pack_rule_reads_the_records_and_the_format_tag() {
        use super::iu8_decide;
        let fmt = saber_kernels::macaq_iu8::FORMAT.to_owned();
        let names = [
            "layer.0.gdn.in_proj_qkvz.wi4",
            "layer.0.gdn.in_proj_qkvz.wi4s",
            "layer.0.gdn.out_proj.wi8",
            "layer.0.gdn.out_proj.wi8s",
            "layer.3.mlp.down_proj.wi8",
            "lm_head.w8",
        ];
        let ok = iu8_decide(
            names.iter().copied(),
            Some(std::slice::from_ref(&fmt)),
            true,
            "p",
        )
        .unwrap();
        assert_eq!(
            ok.unwrap().into_iter().collect::<Vec<_>>(),
            ["layer.0.gdn.out_proj", "layer.3.mlp.down_proj"]
        );
        let man = r#"{"plan":{"descriptor":{"transforms":["macaq_had32","macaq_iu8/1 i4row+i8row a8tok had32","macaq_iu8_w8(19 ops)"]}}}"#;
        assert_eq!(super::iu8_tags_in(man), vec![fmt.clone()]);
        assert!(super::iu8_tags_in(r#"{"transforms":["macaq_had32"]}"#).is_empty());
        // an ordinary pack (no records, no tag): off
        assert_eq!(
            iu8_decide(
                ["layer.0.gdn.in_proj_qkvz.w"].iter().copied(),
                Some(&[]),
                true,
                "p"
            ),
            Ok(None)
        );
        assert_eq!(iu8_decide([].iter().copied(), None, false, "p"), Ok(None));
        // refusals by name
        let e = iu8_decide(
            names.iter().copied(),
            Some(&["macaq_iu8/2 x".into()]),
            true,
            "p",
        )
        .unwrap_err();
        assert!(e.contains("serves") && e.contains("macaq_iu8/2"), "{e}");
        let e = iu8_decide(names.iter().copied(), Some(&[]), true, "p").unwrap_err();
        assert!(e.contains("format"), "{e}");
        let e = iu8_decide(names.iter().copied(), None, true, "p").unwrap_err();
        assert!(e.contains("manifest.json"), "{e}");
        let e = iu8_decide(
            names.iter().copied(),
            Some(std::slice::from_ref(&fmt)),
            false,
            "p",
        )
        .unwrap_err();
        assert!(e.contains("macaq_had32"), "{e}");
        let e = iu8_decide(
            ["layer.0.mlp.gate_up.w"].iter().copied(),
            Some(std::slice::from_ref(&fmt)),
            true,
            "p",
        )
        .unwrap_err();
        assert!(e.contains("no `.wi4`"), "{e}");
        let e = iu8_decide(
            ["layer.0.gdn.out_proj.wi8"].iter().copied(),
            Some(std::slice::from_ref(&fmt)),
            true,
            "p",
        )
        .unwrap_err();
        assert!(e.contains("no W4"), "{e}");
    }

    #[test]
    fn the_w8_ops_are_the_packs_layer_w8_records_only() {
        let names = [
            "layer.3.mlp.down_proj.w8",
            "layer.3.mlp.down_proj.w8s",
            "layer.0.gdn.out_proj.w8",
            "layer.3.attn.o_proj.w",
            "lm_head.w8",
            "layer.4.mlp.gate_up.w8",
        ];
        let got = super::w8_ops_of(names.iter().copied());
        assert_eq!(
            got.into_iter().collect::<Vec<_>>(),
            ["layer.0.gdn.out_proj", "layer.3.mlp.down_proj"]
        );
        assert!(super::w8_ops_of(["layer.0.gdn.out_proj.w"].iter().copied()).is_empty());
    }

    #[test]
    fn the_flag_follows_the_pack_and_a_mismatch_is_refused_by_name() {
        assert_eq!(decide(None, false, "p"), Ok(false));
        assert_eq!(decide(None, true, "p"), Ok(true));
        assert_eq!(decide(Some(true), true, "p"), Ok(true));
        assert_eq!(decide(Some(false), false, "p"), Ok(false));
        let e = decide(Some(true), false, "ef512").unwrap_err();
        assert!(
            e.contains("--macaq-had on") && e.contains("ordinary MXFP4 pack"),
            "{e}"
        );
        let e = decide(Some(false), true, "had").unwrap_err();
        assert!(
            e.contains("--macaq-had off") && e.contains("`macaq_had32` pack"),
            "{e}"
        );
    }
}
