//! Print the INK1 pack plan of an `inkling2_loop` checkpoint as JSON (`docs/INKLING2.md`).
//!
//!   saber-inkling2-packplan --checkpoint DIR [--form mxfp4|nvfp4-native|native]

use std::path::PathBuf;

use saber_model::inkling2::{ExpertForm, Inkling2Config, PackPlan};

fn main() {
    let mut ckpt = None;
    let mut form = ExpertForm::Mxfp4;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--checkpoint" => ckpt = args.next().map(PathBuf::from),
            "--form" => {
                form = match args.next().as_deref() {
                    Some("mxfp4") => ExpertForm::Mxfp4,
                    Some("nvfp4-native") => ExpertForm::Nvfp4Native,
                    Some("native") => ExpertForm::Native,
                    other => {
                        eprintln!("--form {other:?}: mxfp4, nvfp4-native or native");
                        std::process::exit(2);
                    }
                }
            }
            _ => {
                eprintln!(
                    "usage: saber-inkling2-packplan --checkpoint DIR [--form mxfp4|nvfp4-native|native]"
                );
                std::process::exit(2);
            }
        }
    }
    let Some(dir) = ckpt else {
        eprintln!("--checkpoint DIR is required");
        std::process::exit(2);
    };
    let text = match std::fs::read_to_string(dir.join("config.json")) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{}: {e}", dir.display());
            std::process::exit(1);
        }
    };
    let cfg = match Inkling2Config::from_config_json(&text) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    print!(
        "{}",
        PackPlan::new(&cfg, form).plan_text(&cfg, &dir.display().to_string())
    );
}
