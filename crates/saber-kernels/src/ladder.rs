//! The chip -> family -> base source override ladder, borrowed from the prior-art engine
//! (docs/ARCHITECTURE.md, "Prior art: the prior-art engine" item 1): `<k>.gfx1201.hip` beats `<k>.gfx12.hip`
//! beats `<k>.hip`.
//!
//! The resolution is a pure function of (stem, arch, "does this file exist") so it is testable on
//! the host with no filesystem; `resolve_in` is the thin wrapper that hits the disk.

use std::path::{Path, PathBuf};

/// Candidate file names, most specific first. `gfx1201` yields `k.gfx1201.hip`, `k.gfx12.hip`,
/// `k.hip`; an unknown arch string still yields the arch-specific name and the base.
pub fn candidates(stem: &str, arch: &str) -> Vec<String> {
    let mut out = vec![format!("{stem}.{arch}.hip")];
    // Family = the arch with its minor/stepping digits dropped: gfx1201 -> gfx12.
    let digits: String = arch.trim_start_matches("gfx").to_string();
    if arch.starts_with("gfx") && digits.len() > 2 {
        let family = format!("gfx{}", &digits[..2]);
        if family != arch {
            out.push(format!("{stem}.{family}.hip"));
        }
    }
    out.push(format!("{stem}.hip"));
    out
}

/// First candidate for which `exists` is true.
pub fn resolve_with(stem: &str, arch: &str, exists: impl Fn(&str) -> bool) -> Option<String> {
    candidates(stem, arch).into_iter().find(|c| exists(c))
}

/// First candidate that exists under `dir`.
pub fn resolve_in(dir: impl AsRef<Path>, stem: &str, arch: &str) -> Option<PathBuf> {
    let dir = dir.as_ref();
    resolve_with(stem, arch, |c| dir.join(c).is_file()).map(|c| dir.join(c))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chip_beats_family_beats_base() {
        assert_eq!(
            candidates("mxfp4_gemm_decode", "gfx1201"),
            [
                "mxfp4_gemm_decode.gfx1201.hip",
                "mxfp4_gemm_decode.gfx12.hip",
                "mxfp4_gemm_decode.hip"
            ]
        );
        let all = |_: &str| true;
        assert_eq!(
            resolve_with("k", "gfx1201", all).as_deref(),
            Some("k.gfx1201.hip")
        );
        assert_eq!(
            resolve_with("k", "gfx1201", |c| c != "k.gfx1201.hip").as_deref(),
            Some("k.gfx12.hip")
        );
        assert_eq!(
            resolve_with("k", "gfx1201", |c| c == "k.hip").as_deref(),
            Some("k.hip")
        );
        assert_eq!(resolve_with("k", "gfx1201", |_| false), None);
    }

    #[test]
    fn the_committed_sources_resolve_to_the_base_file() {
        let dir = crate::prebuilt::hip_dir();
        for (blob, _) in crate::prebuilt::ALL {
            let stem = blob.trim_end_matches(".hsaco");
            let p = resolve_in(&dir, stem, crate::prebuilt::ARCH)
                .unwrap_or_else(|| panic!("no source for {stem}"));
            assert_eq!(p.file_name().unwrap(), format!("{stem}.hip").as_str());
        }
    }
}
