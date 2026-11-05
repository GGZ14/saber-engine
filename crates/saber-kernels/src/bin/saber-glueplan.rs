//! Emit attention-glue [`LaunchDescriptor`](saber_kernels::LaunchDescriptor)s as JSON for the
//! standalone HIP harness to replay, the same way `saber-kplan`, `saber-coreplan` and
//! `saber-paroplan` do.
//!
//! Task lines on stdin, a JSON array on stdout. Fields are space-separated and positional:
//!
//!   qk_rope    <m> <nq> <nkv> <head_dim> <qkv_n> <interleaved 0|1>
//!   gate_quant <m> <n> <head_dim> <gs>
//!   kv_append  <tokens> <nkv> <head_dim> <block_size> <key_stride> <value_stride>
//!              <block_stride> <page_stride> <head_stride>
//!
//! `--rules PATH` replaces the embedded prod table.

use saber_kernels::glue::{resolve, GateShape, GlueRules, KvShape, Op, QkFlavour, QkShape, Shape};

fn main() {
    let mut argv = std::env::args().skip(1);
    let mut rules_path: Option<String> = None;
    while let Some(a) = argv.next() {
        match a.as_str() {
            "--rules" => rules_path = argv.next(),
            other => {
                eprintln!("saber-glueplan: unknown flag `{other}`");
                std::process::exit(2);
            }
        }
    }
    let loaded = rules_path.as_ref().map(|p| {
        let text = std::fs::read_to_string(p).unwrap_or_else(|e| {
            eprintln!("saber-glueplan: {p}: {e}");
            std::process::exit(2);
        });
        GlueRules::from_toml(&text)
    });
    let rules = match loaded.unwrap_or_else(GlueRules::prod) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("saber-glueplan: {e}");
            std::process::exit(2);
        }
    };
    let mut input = String::new();
    if std::io::Read::read_to_string(&mut std::io::stdin(), &mut input).is_err() {
        eprintln!("saber-glueplan: cannot read stdin");
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
        let r = match f[0] {
            "qk_rope" => resolve(
                Op::QkNormRope {
                    interleaved: n(6) != 0,
                    // The capture's reference is the reference server's TARGET rope, so a `qk_rope` line means
                    // that flavour. A drafter case gets its own op name when F's reference
                    // exists -- a new name, not a seventh positional field, because a line whose
                    // meaning depends on a field nobody reads is how the band bug happened.
                    flavour: QkFlavour::TARGET,
                },
                Shape::Qk(QkShape {
                    m: n(1),
                    nq: n(2),
                    nkv: n(3),
                    head_dim: n(4),
                    qkv_n: n(5),
                }),
                &rules,
            ),
            "gate_quant" => resolve(
                Op::AttnGateQuant,
                Shape::Gate(GateShape {
                    m: n(1),
                    n: n(2),
                    head_dim: n(3),
                    gs: l(4),
                }),
                &rules,
            ),
            "kv_append" => resolve(
                Op::KvAppend,
                Shape::Kv(KvShape {
                    tokens: n(1),
                    nkv: n(2),
                    head_dim: n(3),
                    block_size: n(4),
                    key_stride: l(5),
                    value_stride: l(6),
                    block_stride: l(7),
                    page_stride: l(8),
                    head_stride: l(9),
                }),
                &rules,
            ),
            other => {
                eprintln!("saber-glueplan: unknown op `{other}`");
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
