//! Emit Paro producer [`LaunchDescriptor`](saber_kernels::LaunchDescriptor)s as JSON for the
//! standalone HIP harness to replay, the same way `saber-kplan` and `saber-coreplan` do.
//!
//! Task lines on stdin, a JSON array on stdout. Fields are space-separated and positional:
//!
//!   paro_rms      <m> <k> <p> <krot> <eps>
//!   paro_rms_tok  <m> <k> <p> <krot> <eps>
//!   paro_ew       <m> <n> <krot> <ys> <eps> <mode>
//!   paro_ew_tok   <m> <n> <krot> <ys> <eps> <mode>
//!
//! `--rules PATH` replaces the embedded prod table.

use saber_kernels::paro::{resolve, Op, ParoEwShape, ParoRules, ParoShape, Shape};

fn main() {
    let mut argv = std::env::args().skip(1);
    let mut rules_path: Option<String> = None;
    while let Some(a) = argv.next() {
        match a.as_str() {
            "--rules" => rules_path = argv.next(),
            other => {
                eprintln!("saber-paroplan: unknown flag `{other}`");
                std::process::exit(2);
            }
        }
    }
    let loaded = rules_path.as_ref().map(|p| {
        let text = std::fs::read_to_string(p).unwrap_or_else(|e| {
            eprintln!("saber-paroplan: {p}: {e}");
            std::process::exit(2);
        });
        ParoRules::from_toml(&text)
    });
    let rules = match loaded.unwrap_or_else(ParoRules::prod_paro) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("saber-paroplan: {e}");
            std::process::exit(2);
        }
    };
    let mut input = String::new();
    if std::io::Read::read_to_string(&mut std::io::stdin(), &mut input).is_err() {
        eprintln!("saber-paroplan: cannot read stdin");
        std::process::exit(2);
    }
    let mut out: Vec<serde_json::Value> = Vec::new();
    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        let n = |i: usize| -> u32 {
            f[i].parse()
                .unwrap_or_else(|_| panic!("field {i} of `{line}`"))
        };
        let l = |i: usize| -> i64 {
            f[i].parse()
                .unwrap_or_else(|_| panic!("field {i} of `{line}`"))
        };
        let b = |i: usize| -> u32 {
            f[i].parse::<f32>()
                .unwrap_or_else(|_| panic!("field {i} of `{line}`"))
                .to_bits()
        };
        let rms = || ParoShape {
            m: n(1),
            k: n(2),
            p: n(3),
            krot: n(4),
            eps_bits: b(5),
        };
        let ew = || ParoEwShape {
            m: n(1),
            n: n(2),
            krot: n(3),
            ys: l(4),
            eps_bits: b(5),
        };
        let r = match f[0] {
            "paro_rms" => resolve(Op::AddRmsRot, Shape::Rms(rms()), &rules),
            "paro_rms_tok" => resolve(Op::AddRmsRotTok, Shape::Rms(rms()), &rules),
            "paro_ew" => resolve(Op::EwRot { mode: n(6) }, Shape::Ew(ew()), &rules),
            "paro_ew_tok" => resolve(Op::EwRotTok { mode: n(6) }, Shape::Ew(ew()), &rules),
            other => {
                eprintln!("saber-paroplan: unknown op `{other}`");
                std::process::exit(2);
            }
        };
        match r {
            Ok(d) => {
                let mut v = serde_json::to_value(&d).expect("descriptor serialises");
                v["task"] = serde_json::Value::String(line.to_owned());
                out.push(v);
            }
            // A refusal is a result, not a crash: the harness records it and moves on.
            Err(e) => out.push(serde_json::json!({ "task": line, "error": e.to_string() })),
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&out).expect("array serialises")
    );
}
