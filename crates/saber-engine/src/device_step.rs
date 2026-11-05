//! P27-W2-2: the 27B's DEVICE-STEP forms (`--engine-device-step`, serve; `--device-step` on
//! engine / worker / `run_engine.sh`, forwarded to rank 1 by value). The port of the Flash-Next
//! device-resident step (FN119 / FN112 / FN117) to the pieces the 27B can use (LEDGER P27-W2-2's
//! census names every piece GO or NO-GO).
//!
//! * **`auto`** -- THE DEFAULT since CMP27P (2026-10-28): `cand` wherever it can serve (TP = 2,
//!   no T4/T5 race check), `off` elsewhere, resolved once at the engine's config head on EACH
//!   rank from the same forwarded inputs ([`DeviceStep::resolve`]; boot line `devstep27 rank R:
//!   auto -> ...`). An explicit `cand` still refuses by name where it cannot serve.
//! * **`off`** -- the host bridge: the control and the rollback.
//! * **`cand`** -- the drafter's candidate seam on the card: between the DFlash2 backbone and its
//!   tail, each rank's top-16 shard crosses into the PEER's P2P window and `p27d_cand_seam`
//!   (`saber_kernels::p27d`, its own code object) merges the 32 by the host's own stable
//!   descending law, writing the merge over `sel.vals` / `sel.cand64` exactly as the host's seed
//!   did; the tail is enqueued behind it, so the host's backbone wait, the shard read, the socket
//!   frame (`engine.draftcand`), the merge and the re-seed are gone from the proposal. Bit-exact
//!   by construction; a NaN candidate (where the host comparator has no defined order) or a peer
//!   shard that never lands FAILS the proposal by name. TP = 2 only, never with the T4/T5 race
//!   check (its host-side seam digests read the socket frame).

use std::sync::atomic::{AtomicU64, Ordering};

/// The forms a run can take.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeviceStep {
    /// Today's host bridge (the control and the rollback).
    Off,
    /// The drafter's candidate seam merged on the card.
    Cand,
    /// `cand` where it can serve, `off` elsewhere ([`DeviceStep::resolve`]; THE DEFAULT, CMP27P).
    #[default]
    Auto,
}

/// The default: `auto` = `cand` at TP = 2 without the T4/T5 race check (CMP27P's flip; `off` = the
/// control and the rollback).
pub const DEFAULT_DEVICE_STEP: &str = "auto";

impl DeviceStep {
    /// Parse `off` / `cand` / `auto`.
    ///
    /// # Errors
    /// Any other spelling, by name.
    pub fn parse(s: &str) -> Result<DeviceStep, String> {
        match s.trim() {
            "off" | "none" => Ok(DeviceStep::Off),
            "cand" => Ok(DeviceStep::Cand),
            "auto" => Ok(DeviceStep::Auto),
            o => Err(format!(
                "unknown device-step form `{o}`: the 27B's forms are auto, off, cand (LEDGER P27-W2-2: \
                 the Flash-Next forms lazy / ahead / next have no 27B port)"
            )),
        }
    }

    /// The canonical spelling (what rank 1 is handed).
    #[must_use]
    pub fn as_arg(self) -> &'static str {
        match self {
            DeviceStep::Off => "off",
            DeviceStep::Cand => "cand",
            DeviceStep::Auto => "auto",
        }
    }

    /// Resolve `auto` for a run of `world` ranks (`race_check` = the T4/T5 check): `cand` where
    /// [`DeviceStep::check`] accepts it, `off` elsewhere, with the stand-down's reason. `off` /
    /// `cand` resolve to themselves (an explicit `cand` is refused later by `check`, by name).
    #[must_use]
    pub fn resolve(self, world: u32, race_check: bool) -> (DeviceStep, Option<String>) {
        match self {
            DeviceStep::Auto => match DeviceStep::Cand.check(world, race_check) {
                Ok(()) => (DeviceStep::Cand, None),
                Err(why) => (DeviceStep::Off, Some(why)),
            },
            f => (f, None),
        }
    }

    /// Whether the candidate seam runs on the card (a RESOLVED form: `auto` is resolved at the
    /// engine's config head before anything reads this).
    #[must_use]
    pub fn cand(self) -> bool {
        self == DeviceStep::Cand
    }

    /// Refuse the combinations the form cannot serve, by name.
    ///
    /// # Errors
    /// `cand` at TP = 1 (no peer: no seam) or with the T4/T5 race check.
    pub fn check(self, tp: u32, race_check: bool) -> Result<(), String> {
        if self == DeviceStep::Auto {
            return Err(
                "--engine-device-step auto reached the drafter unresolved (a bug: the engine's \
                 config head resolves it)"
                    .to_owned(),
            );
        }
        if !self.cand() {
            return Ok(());
        }
        if tp != 2 {
            return Err(format!(
                "--engine-device-step cand is a TP = 2 form (the candidate seam joins two ranks' \
                 shards); this boot is TP = {tp}"
            ));
        }
        if race_check {
            return Err(
                "--engine-device-step cand with --engine-race-check (T4/T5): the check's \
                        seam digests read the host's socket frame, which this form removes; use \
                        --race-check shadow (supported) or off"
                    .to_owned(),
            );
        }
        Ok(())
    }
}

/// End-of-run counters (`devstep27 rank R:`).
#[derive(Debug, Default)]
pub struct DeviceStepCounters {
    /// Proposals whose seam merged on the card.
    pub cand_merged: AtomicU64,
    /// Rows merged on the card.
    pub cand_rows: AtomicU64,
    /// Proposals the kernel's error word failed (each one an `Err`, so a nonzero count is a boot
    /// that died).
    pub cand_errors: AtomicU64,
    /// Proposals that took the host bridge (no tail, or the form off).
    pub host_bridge: AtomicU64,
}

impl DeviceStepCounters {
    /// The end line's body.
    #[must_use]
    pub fn line(&self, form: DeviceStep) -> String {
        let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
        format!(
            "form {}; cand: merged on the card {} proposal(s) ({} rows), errors {}; host bridge {}",
            form.as_arg(),
            g(&self.cand_merged),
            g(&self.cand_rows),
            g(&self.cand_errors),
            g(&self.host_bridge),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_device_step_is_auto_by_default_and_parses_by_name() {
        // CMP27P: `auto` (= cand at TP = 2 without the T4/T5 check) is the default; `off` = the
        // control and the rollback
        assert_eq!(DeviceStep::default(), DeviceStep::Auto);
        assert_eq!(
            DeviceStep::parse(DEFAULT_DEVICE_STEP).unwrap(),
            DeviceStep::Auto
        );
        assert_eq!(DeviceStep::parse("cand").unwrap(), DeviceStep::Cand);
        assert_eq!(DeviceStep::parse("off").unwrap(), DeviceStep::Off);
        assert_eq!(DeviceStep::parse("none").unwrap(), DeviceStep::Off);
        for f in [DeviceStep::Off, DeviceStep::Cand, DeviceStep::Auto] {
            assert_eq!(DeviceStep::parse(f.as_arg()).unwrap(), f);
        }
        // the Flash-Next forms are refused by name (no 27B port)
        let e = DeviceStep::parse("next").unwrap_err();
        assert!(e.contains("next"), "{e}");
        assert!(DeviceStep::parse("ahead").is_err());
    }

    #[test]
    fn cand_refuses_tp1_and_the_race_check() {
        assert!(DeviceStep::Cand.check(2, false).is_ok());
        assert!(DeviceStep::Cand
            .check(1, false)
            .unwrap_err()
            .contains("TP = 1"));
        assert!(DeviceStep::Cand
            .check(2, true)
            .unwrap_err()
            .contains("race-check"));
        // the control refuses nothing
        assert!(DeviceStep::Off.check(1, true).is_ok());
        // an unresolved auto never reaches the drafter
        assert!(DeviceStep::Auto.check(2, false).is_err());
    }

    #[test]
    fn auto_resolves_to_cand_where_it_serves_and_stands_down_elsewhere() {
        assert_eq!(DeviceStep::Auto.resolve(2, false), (DeviceStep::Cand, None));
        let (f, why) = DeviceStep::Auto.resolve(1, false);
        assert_eq!(f, DeviceStep::Off);
        assert!(why.unwrap().contains("TP = 1"));
        let (f, why) = DeviceStep::Auto.resolve(2, true);
        assert_eq!(f, DeviceStep::Off);
        assert!(why.unwrap().contains("race-check"));
        // explicit forms resolve to themselves (an explicit cand is refused later, by name)
        assert_eq!(DeviceStep::Cand.resolve(1, false), (DeviceStep::Cand, None));
        assert_eq!(DeviceStep::Off.resolve(2, false), (DeviceStep::Off, None));
    }

    #[test]
    fn the_end_line_names_the_form() {
        let c = DeviceStepCounters::default();
        c.cand_merged.fetch_add(3, Ordering::Relaxed);
        let l = c.line(DeviceStep::Cand);
        assert!(
            l.starts_with("form cand; cand: merged on the card 3"),
            "{l}"
        );
    }
}
