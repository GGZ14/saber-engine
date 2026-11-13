//! The setup wizard's state machine. The wizard shows while `[setup] completed` is
//! false (a fresh install: no `config.toml`); every step saves the config, so a reload resumes
//! at the saved step. Steps: System check -> Model -> Profile -> Network and security ->
//! Review and launch -> Done. Going back is always allowed; going forward passes the gate of
//! every step on the way (a model that is not installed cannot be left, and so on). "I'll
//! configure manually" skips to done; Settings can run it again.

use serde::Serialize;

use crate::config::SetupState;

/// The steps in order: (id, label).
pub const STEPS: &[(&str, &str)] = &[
    ("system", "System check"),
    ("model", "Model"),
    ("profile", "Profile"),
    ("network", "Network and security"),
    ("review", "Review and launch"),
    ("done", "Done"),
];

/// What the gates read (computed by the caller from the catalog, the packs and the validator).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Facts {
    /// The picked model is in the catalog.
    pub model_known: bool,
    /// Every component of the picked model is installed (the mock has none).
    pub model_installed: bool,
    /// The picked profile is one of the built-ins for the model's subcommand.
    pub profile_known: bool,
    /// The config's validation errors (the Network step's gate).
    pub config_errors: Vec<String>,
    /// The engine reached ready since the wizard launched it.
    pub engine_ready: bool,
}

#[must_use]
pub fn index(step: &str) -> Option<usize> {
    STEPS.iter().position(|(id, _)| *id == step)
}

/// Why `step` cannot be left forward (`None` = it can).
#[must_use]
pub fn blocker(step: &str, st: &SetupState, f: &Facts) -> Option<String> {
    match step {
        "model" if st.model.is_empty() => Some("Pick a model first.".into()),
        "model" if !f.model_known => Some(format!(
            "The model `{}` is not in the catalog any more: pick another.",
            st.model
        )),
        "model" if !f.model_installed => {
            Some("Download the model first (or pick one that is installed).".into())
        }
        "profile" if st.profile.is_empty() || !f.profile_known => {
            Some("Pick a profile first.".into())
        }
        "network" if !f.config_errors.is_empty() => Some(format!(
            "Fix the settings first: {}",
            f.config_errors.join("; ")
        )),
        "review" if !f.engine_ready => Some(
            "Start the engine and wait for it to be ready, or choose \"Finish without starting\"."
                .into(),
        ),
        _ => None,
    }
}

/// Move to `target`: back always, forward through every gate on the way.
pub fn goto(st: &SetupState, target: &str, f: &Facts) -> Result<SetupState, String> {
    let to = index(target).ok_or_else(|| format!("`{target}` is not a wizard step"))?;
    let from = index(&st.step).unwrap_or(0);
    if to > from {
        for (id, _) in &STEPS[from..to] {
            if let Some(why) = blocker(id, st, f) {
                return Err(why);
            }
        }
    }
    let mut out = st.clone();
    out.step = target.to_string();
    if target == "done" {
        out.completed = true;
    }
    Ok(out)
}

/// Finish from the Review step without waiting for ready ("Finish without starting").
pub fn finish_without_start(st: &SetupState, f: &Facts) -> Result<SetupState, String> {
    let mut g = f.clone();
    g.engine_ready = true;
    goto(st, "done", &g)
}

/// "I'll configure manually".
#[must_use]
pub fn skip(st: &SetupState) -> SetupState {
    SetupState {
        completed: true,
        skipped: true,
        step: "done".into(),
        ..st.clone()
    }
}

/// Run the wizard again (from Settings): the picks are kept as the starting point.
#[must_use]
pub fn rerun(st: &SetupState) -> SetupState {
    SetupState {
        completed: false,
        skipped: false,
        step: "system".into(),
        ..st.clone()
    }
}

/// Picking a new model resets the later picks that depended on it.
#[must_use]
pub fn pick_model(st: &SetupState, model: &str, same_subcommand: bool) -> SetupState {
    SetupState {
        model: model.to_string(),
        profile: if same_subcommand {
            st.profile.clone()
        } else {
            String::new()
        },
        ..st.clone()
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct StepView {
    pub id: &'static str,
    pub label: &'static str,
    /// Reachable from the current step (back, or forward through passing gates).
    pub reachable: bool,
    pub current: bool,
    /// Its gate passes (forward from it is open).
    pub complete: bool,
}

/// The step list as the UI shows it.
#[must_use]
pub fn view(st: &SetupState, f: &Facts) -> Vec<StepView> {
    STEPS
        .iter()
        .map(|(id, label)| StepView {
            id,
            label,
            reachable: goto(st, id, f).is_ok(),
            current: st.step == *id,
            complete: blocker(id, st, f).is_none() && *id != "done",
        })
        .collect()
}

/// A random secret (`<prefix><40 hex>`) from the kernel's CSPRNG.
pub fn random_key(prefix: &str) -> std::io::Result<String> {
    use std::io::Read;
    let mut b = [0u8; 20];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut b)?;
    Ok(format!("{prefix}{}", hex::encode(b)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(step: &str) -> SetupState {
        SetupState {
            step: step.into(),
            ..SetupState::default()
        }
    }

    #[test]
    fn the_happy_path_walks_every_step() {
        let mut s = SetupState::default();
        assert_eq!(s.step, "system");
        let mut f = Facts::default();
        s = goto(&s, "model", &f).unwrap();
        // the model step's gates, in order
        assert!(goto(&s, "profile", &f)
            .unwrap_err()
            .contains("Pick a model"));
        s = pick_model(&s, "27b", true);
        assert!(goto(&s, "profile", &f).unwrap_err().contains("catalog"));
        f.model_known = true;
        assert!(goto(&s, "profile", &f).unwrap_err().contains("Download"));
        f.model_installed = true;
        s = goto(&s, "profile", &f).unwrap();
        assert!(goto(&s, "network", &f).is_err());
        s.profile = "Balanced".into();
        f.profile_known = true;
        s = goto(&s, "network", &f).unwrap();
        f.config_errors = vec!["engine.port must be a real port".into()];
        assert!(goto(&s, "review", &f).unwrap_err().contains("engine.port"));
        f.config_errors.clear();
        s = goto(&s, "review", &f).unwrap();
        assert!(goto(&s, "done", &f).unwrap_err().contains("ready"));
        assert!(!s.completed);
        f.engine_ready = true;
        s = goto(&s, "done", &f).unwrap();
        assert!(s.completed && !s.skipped);
    }

    #[test]
    fn back_is_free_and_forward_jumps_check_every_gate() {
        let f = Facts {
            model_known: true,
            model_installed: true,
            ..Facts::default()
        };
        let mut s = st("network");
        s.model = "mock".into();
        // back to any earlier step
        for t in ["review", "system", "model", "profile"] {
            if t == "review" {
                continue;
            }
            assert_eq!(goto(&s, t, &f).unwrap().step, t);
        }
        // a jump from system to review fails at the first closed gate (profile)
        let s0 = SetupState {
            model: "mock".into(),
            ..SetupState::default()
        };
        assert!(goto(&s0, "review", &f).unwrap_err().contains("profile"));
        assert!(goto(&s0, "nowhere", &f).is_err());
        let v = view(&s0, &f);
        assert!(v[0].current && v[1].reachable && v[2].reachable && !v[3].reachable);
    }

    #[test]
    fn skip_rerun_finish_and_model_change() {
        let s = skip(&st("model"));
        assert!(s.completed && s.skipped && s.step == "done");
        let r = rerun(&s);
        assert!(!r.completed && !r.skipped && r.step == "system");
        let mut s = st("review");
        s.model = "mock".into();
        s.profile = "Balanced".into();
        let f = Facts {
            model_known: true,
            model_installed: true,
            profile_known: true,
            ..Facts::default()
        };
        assert!(goto(&s, "done", &f).is_err());
        assert!(finish_without_start(&s, &f).unwrap().completed);
        // a model with another subcommand clears the profile
        assert_eq!(pick_model(&s, "flashnext", false).profile, "");
        assert_eq!(pick_model(&s, "27b", true).profile, "Balanced");
    }

    #[test]
    fn keys_are_random_and_prefixed() {
        let a = random_key("sk-saber-").unwrap();
        let b = random_key("sk-saber-").unwrap();
        assert!(a.starts_with("sk-saber-") && a.len() == 9 + 40);
        assert_ne!(a, b);
    }
}
