//! **P27-16: the 27B's launch-fusion arm** (`--engine-fuse none|<families>|all` on the serve
//! plane, `--fuse` on `saber engine`; the counter is the boot line `fuse27`).
//!
//! The arm names FAMILIES of the 27B's decode launch census (LEDGER P27-16) whose fused form is a
//! bit-exact ADDED symbol, and composes them over the served set (`--decode-fusion all`, rows 1-3
//! + 7). The families built today are the two the census left standing on the target:
//!
//! * `smq`   -- fusion-map row 4 (ws-h40): the `gate_up` decode GEMM's last M arriving n-tiles
//!   run `silu_mul_quant` on one row each (`mxfp4_gemm_decode_smq_*`), 64 launches a step at
//!   every decode bucket the rules put in the band (b1s8 and b8s8 both);
//! * `gdnnq` -- fusion-map row 5 (ws-h42): the GDN fused update's last M arriving workgroups run
//!   `gdn_norm_quant` (`saber_gdn_fnq_f16s_*`), 48 launches where the decode GDN is the fused update
//!   (one sequence at TP=2: b1).
//!
//! `none` is the control and THE DEFAULT (today's served step); `all` is every family. The arm is
//! resolved on the leader into the [`DecodeFusion`] set rank 1 is told by value, so a pair can
//! never run two protocols.

use crate::device::DecodeFusion;

/// The default: no family beyond the served set.
pub const DEFAULT_FUSE27: &str = "none";

/// Every family name, in census order.
pub const FAMILIES: [&str; 2] = ["smq", "gdnnq"];

/// The resolved arm.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fuse27 {
    /// Fusion-map row 4.
    pub smq: bool,
    /// Fusion-map row 5.
    pub gdnnq: bool,
}

impl Fuse27 {
    /// `none`, `all`, or a comma list of [`FAMILIES`].
    ///
    /// # Errors
    /// An unknown family, or `none` / `all` inside a list.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "none" => return Ok(Self::default()),
            "all" => {
                return Ok(Self {
                    smq: true,
                    gdnnq: true,
                })
            }
            _ => {}
        }
        let mut f = Self::default();
        for p in s.split(',').map(str::trim) {
            match p {
                "smq" => f.smq = true,
                "gdnnq" => f.gdnnq = true,
                other => {
                    return Err(format!(
                        "--engine-fuse: `{other}` is not a family; `none` (the default), `all`, \
                         or a list of {FAMILIES:?}"
                    ))
                }
            }
        }
        Ok(f)
    }

    /// The canonical spelling (`none`, `all`, or the list).
    #[must_use]
    pub fn as_arg(self) -> String {
        match (self.smq, self.gdnnq) {
            (false, false) => "none".into(),
            (true, true) => "all".into(),
            (true, false) => "smq".into(),
            (false, true) => "gdnnq".into(),
        }
    }

    /// Is any family on?
    #[must_use]
    pub const fn is_on(self) -> bool {
        self.smq || self.gdnnq
    }

    /// The decode fusion set this arm runs over `base` (what `--decode-fusion` resolved to).
    /// `none` returns `base` unchanged; a family composes only over the served set `all`.
    ///
    /// # Errors
    /// A family over any other base set (the families are defined on top of rows 1-3 + 7).
    pub fn resolve(self, base: DecodeFusion) -> Result<DecodeFusion, String> {
        if !self.is_on() {
            return Ok(base);
        }
        if base != DecodeFusion::All {
            return Err(format!(
                "--engine-fuse {}: the families compose over the served decode set `all`, not \
                 `{}`; drop the `--decode-fusion` / `--no-decode-fusion` flag",
                self.as_arg(),
                base.worker_arg()
            ));
        }
        Ok(match (self.smq, self.gdnnq) {
            (true, true) => DecodeFusion::H42,
            (true, false) => DecodeFusion::H40,
            (false, true) => DecodeFusion::GdnNq,
            (false, false) => base,
        })
    }

    /// `gdnnq` beside a FUSED mixed step (`--engine-mixed-prefill packed|fused`, packed = the
    /// served default since C11) is refused BY NAME at the boot: the fused mixed forward splits
    /// every row-local op from its per-sequence half, and row 5 folds the row-local norm into the
    /// per-sequence GDN update, so the fused binding plan refuses the step at run time ("its
    /// row-local half would run twice", P27-16 h1 gate:A, a rank-1 FATAL mid-serve). `smq` is
    /// row-local into row-local and runs.
    ///
    /// # Errors
    /// `gdnnq` with `fused_mixed`.
    pub fn check_mixed(self, fused_mixed: bool) -> Result<Self, String> {
        if self.gdnnq && fused_mixed {
            return Err(format!(
                "--engine-fuse {}: the `gdnnq` family (fusion-map row 5) cannot run beside the \
                 fused mixed step (`--engine-mixed-prefill packed|fused`, packed is the served \
                 default): its norm is row-local and the GDN update per-sequence; use `smq`, or \
                 `--engine-mixed-prefill serial|off`",
                self.as_arg()
            ));
        }
        Ok(self)
    }

    /// The arm a resolved set carries (the worker's view, and the `fuse27` line's).
    #[must_use]
    pub const fn of(set: DecodeFusion) -> Self {
        match set {
            DecodeFusion::H40 => Self {
                smq: true,
                gdnnq: false,
            },
            DecodeFusion::H42 => Self {
                smq: true,
                gdnnq: true,
            },
            DecodeFusion::GdnNq => Self {
                smq: false,
                gdnnq: true,
            },
            _ => Self {
                smq: false,
                gdnnq: false,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p27_16_the_arm_parses_resolves_and_defaults_to_none() {
        assert_eq!(Fuse27::parse(DEFAULT_FUSE27), Ok(Fuse27::default()));
        let all = Fuse27::parse("all").expect("all");
        assert_eq!(Fuse27::parse("smq,gdnnq"), Ok(all));
        assert_eq!(all.as_arg(), "all");
        assert!(Fuse27::parse("smq,none").is_err());
        assert!(Fuse27::parse("rope").is_err());
        let b = DecodeFusion::All;
        assert_eq!(Fuse27::default().resolve(b), Ok(b), "none is today's set");
        assert_eq!(all.resolve(b), Ok(DecodeFusion::H42));
        assert_eq!(
            Fuse27::parse("smq").unwrap().resolve(b),
            Ok(DecodeFusion::H40)
        );
        assert_eq!(
            Fuse27::parse("gdnnq").unwrap().resolve(b),
            Ok(DecodeFusion::GdnNq)
        );
        // h1's FATAL, refused at the boot instead: gdnnq beside the fused mixed step
        assert!(all.check_mixed(true).is_err());
        assert!(Fuse27::parse("gdnnq").unwrap().check_mixed(true).is_err());
        assert_eq!(
            Fuse27::parse("smq")
                .unwrap()
                .check_mixed(true)
                .map(Fuse27::as_arg),
            Ok("smq".into())
        );
        assert_eq!(all.check_mixed(false), Ok(all));
        // the CONTROL: a family over another base set is refused by name
        assert!(all.resolve(DecodeFusion::Off).is_err());
        assert!(all.resolve(DecodeFusion::H37).is_err());
        for s in ["none", "smq", "gdnnq", "all"] {
            let f = Fuse27::parse(s).unwrap();
            let set = f.resolve(b).unwrap();
            assert_eq!(Fuse27::of(set), f, "{s}");
            // the worker round trip
            assert_eq!(DecodeFusion::from_worker_arg(set.worker_arg()), Ok(set));
        }
    }

    /// The families are what the plans fold: at the served b1s8 (f16 GDN state) `all` deletes
    /// 64 + 48 launches from the served set's 483, and `none` deletes none.
    #[test]
    fn p27_16_the_families_fold_what_the_census_names() {
        use saber_exec::layer::{BlockPlan, Collectives, PlanOptions, RuntimeParams};
        use saber_kernels::rules::LaunchRules;
        use saber_kernels::saber::SaberRules;
        use saber_model::graph::{Bucket, Graph, ModelConfig};
        let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2)
            .expect("the TP=2 graph")
            .with_gdn_state_f16()
            .expect("the served f16 state");
        let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
        let saber = SaberRules::prod_saber().expect("saber");
        let rules = LaunchRules::prod_mxfp4().expect("rules");
        let mut seen = Vec::new();
        for s in ["none", "smq", "gdnnq", "all"] {
            let set = Fuse27::parse(s)
                .unwrap()
                .resolve(DecodeFusion::All)
                .unwrap();
            let o = set.apply(
                PlanOptions::BODY
                    .rank(0)
                    .with_collectives(Collectives::Real)
                    .with_batch_invariant_attn(),
            );
            let p = BlockPlan::with_options(
                &g,
                Bucket::Decode {
                    batch: 1,
                    verify_width: 8,
                },
                &layers,
                RuntimeParams {
                    max_ctx: 4096,
                    gdn_slots: 9,
                },
                &rules,
                &saber,
                o,
            )
            .expect("b1s8 plans");
            seen.push((s, p.expected_launches(), p.fused_gemm_smq, p.fused_gdn_nq));
        }
        assert_eq!(
            seen,
            vec![
                ("none", 483, 0, 0),
                ("smq", 419, 64, 0),
                ("gdnnq", 435, 0, 48),
                ("all", 371, 64, 48)
            ]
        );
    }
}
