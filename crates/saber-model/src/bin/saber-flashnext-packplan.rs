//! Print the Flash-Next (`qwen4_exp`) EP=2 tier-pack plan as JSON on stdout.
//!
//!     saber-flashnext-packplan --checkpoint /mnt/nvme/models/Qwen3.8-Flash-Next-Quark-MXFP4
//!
//! The checkpoint's `config.json` is parsed by `saber_model::qwen4_exp::Qwen4ExpConfig`, which
//! refuses an unknown or missing field by name; the record set, the expert tier layout, the PLE
//! index geometry and the byte ledger come from `saber_model::qwen4_exp::pack_plan`.
//! `tools/pack/build_flashnext_pack.py` consumes the JSON.

use std::process::ExitCode;

use saber_model::qwen4_exp::pack_plan::PackPlan;
use saber_model::qwen4_exp::Qwen4ExpConfig;

fn main() -> ExitCode {
    let mut checkpoint: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--checkpoint" => checkpoint = args.next(),
            _ => {
                eprintln!("usage: saber-flashnext-packplan --checkpoint DIR");
                return ExitCode::from(2);
            }
        }
    }
    let Some(dir) = checkpoint else {
        eprintln!("saber-flashnext-packplan: --checkpoint DIR is required");
        return ExitCode::from(2);
    };
    let path = std::path::Path::new(&dir).join("config.json");
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("saber-flashnext-packplan: {}: {e}", path.display());
            return ExitCode::from(1);
        }
    };
    let cfg = match Qwen4ExpConfig::from_config_json(&text) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("saber-flashnext-packplan: {}: {e}", path.display());
            return ExitCode::from(1);
        }
    };
    let plan = PackPlan::ep2(&cfg);
    for d in cfg.declared_defaults() {
        eprintln!("declared default: {} = {} ({})", d.key, d.value, d.source);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&plan.to_json(&cfg)).expect("plan serializes")
    );
    ExitCode::SUCCESS
}
