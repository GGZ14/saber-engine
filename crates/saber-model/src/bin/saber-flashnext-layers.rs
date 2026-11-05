//! Print the Flash-Next layer graph (FN9, `saber_model::qwen4_exp::layer_graph`) as JSON: the op
//! list per layer with each op's kernel, group and launches, and the step total. The card harness
//! (`tools/capture/fn9/fn9_layers.py`) reads it and asserts its own launches per layer against it.
//!
//!   saber-flashnext-layers --checkpoint DIR [--tp 1|2] [--rows N] [--form prefill|decode:B:pair|decode:B:declc]

use saber_kernels::flashnext::FlashNextRules;
use saber_kernels::fn6::AttnForm;
use saber_model::qwen4_exp::layer_graph::{step_graph, to_json, Form, Tp};
use saber_model::qwen4_exp::Qwen4ExpConfig;

fn fail(msg: &str) -> ! {
    eprintln!("saber-flashnext-layers: {msg}");
    std::process::exit(2)
}

fn main() {
    let mut ckpt = None;
    let (mut tp, mut rows, mut form) = (Tp::One, 16u32, Form::Prefill);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let v = args
            .next()
            .unwrap_or_else(|| fail(&format!("{a} needs a value")));
        match a.as_str() {
            "--checkpoint" => ckpt = Some(v),
            "--tp" => {
                tp = match v.as_str() {
                    "1" => Tp::One,
                    "2" => Tp::Two,
                    _ => fail("--tp is 1 or 2"),
                }
            }
            "--rows" => rows = v.parse().unwrap_or_else(|_| fail("--rows")),
            "--form" => {
                form = if v == "prefill" {
                    Form::Prefill
                } else {
                    let f: Vec<&str> = v.split(':').collect();
                    if f.len() != 3 || f[0] != "decode" {
                        fail("--form prefill | decode:B:pair | decode:B:declc");
                    }
                    Form::Decode {
                        batch: f[1].parse().unwrap_or_else(|_| fail("batch")),
                        attn: match f[2] {
                            "pair" => AttnForm::DecodePair,
                            "declc" => AttnForm::DecodeDeclc,
                            _ => fail("attention form pair | declc"),
                        },
                    }
                }
            }
            _ => fail(&format!("unknown flag {a}")),
        }
    }
    let dir = ckpt.unwrap_or_else(|| fail("--checkpoint DIR"));
    let text = std::fs::read_to_string(format!("{dir}/config.json"))
        .unwrap_or_else(|e| fail(&format!("{dir}/config.json: {e}")));
    let cfg = Qwen4ExpConfig::from_config_json(&text).unwrap_or_else(|e| fail(&e.to_string()));
    let ple = *cfg
        .ple
        .layer_indices()
        .first()
        .unwrap_or_else(|| fail("no PLE layer"));
    let rules = FlashNextRules::prod().unwrap_or_else(|e| fail(&e.to_string()));
    let g = step_graph(&rules, &cfg.layers, ple, form, tp, rows);
    println!(
        "{}",
        serde_json::to_string_pretty(&to_json(&g)).expect("json")
    );
}
