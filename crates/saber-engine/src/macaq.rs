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

#[cfg(test)]
mod tests {
    use super::decide;

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
