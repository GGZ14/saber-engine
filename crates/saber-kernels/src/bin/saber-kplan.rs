//! Emit [`saber_kernels::LaunchDescriptor`]s as JSON, one per requested op, for the standalone
//! HIP harness (`crates/saber-kernels/hip/harness/run_hsaco.cpp`) to replay.
//!
//! The harness never re-derives a tile choice: it reads the kernel symbol, grid, block and the
//! kernarg byte layout from here, which is the same `resolve` the engine will call. That is what
//! makes the bit-exact gate a test of the PORT rather than of a second transcription.
//!
//! Reads task lines on stdin, writes a JSON array on stdout:
//!
//!   gemm     <M> <N> <K>            # W4A8 GEMM, one partition, folded path
//!   gemm_nw  <M> <N> <K>            # ... without the per-row reference exponent
//!   add_rms  <M> <K> <eps>
//!   silu     <M> <N>
//!   gdn      <M> <N> <zs> <eps>     # the reference's row-major store (`gdn_norm_tiled` off)
//!   gdn_tiled <M> <N> <zs> <eps>    # Saber's fragment-tiled store (G2, `gdn_norm_tiled` on)
//!   gemm_wp0 <M> <N> <K>            # ... under a rules file whose WPERM knob is 0
//!   gemm_part <M> <N> <K> <pb1> <pb2> <astride>   # merged-linear partition select
//!   gemm_indirect <M> <N> <K>       # the DIRECT=false decode control
//!   gemm_silu <M> <N> <K>           # the decode GEMM with silu_mul_quant folded into A staging
//!   gemm_rm  <M> <N> <K>            # ... on an edge whose A stays row-major at any M
//!
//! `--rules PATH` replaces the embedded prod table (the WPERM=0 and DIRECT=false gate cases use
//! it, because both are decided once per process by the loader, not per call).

use saber_kernels::rules::{
    gemm_decode_fused_silu, gemm_decode_indirect, resolve_with_layout, ALayout, LaunchDescriptor,
    LaunchRules, Op, Partitions, Shape, NO_PARTITIONS,
};

fn main() {
    let mut argv = std::env::args().skip(1);
    let mut rules_path: Option<String> = None;
    while let Some(a) = argv.next() {
        match a.as_str() {
            "--rules" => rules_path = argv.next(),
            other => {
                eprintln!("saber-kplan: unknown flag `{other}`");
                std::process::exit(2);
            }
        }
    }
    let loaded = rules_path.as_ref().map(|p| {
        let text = std::fs::read_to_string(p).unwrap_or_else(|e| {
            eprintln!("saber-kplan: {p}: {e}");
            std::process::exit(2);
        });
        LaunchRules::from_toml(&text)
    });
    let rules = match loaded.unwrap_or_else(LaunchRules::prod_mxfp4) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("saber-kplan: {e}");
            std::process::exit(2);
        }
    };
    let mut out: Vec<serde_json::Value> = Vec::new();
    let mut input = String::new();
    if std::io::Read::read_to_string(&mut std::io::stdin(), &mut input).is_err() {
        eprintln!("saber-kplan: cannot read stdin");
        std::process::exit(2);
    }
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
        let fl = |i: usize| -> f32 {
            f[i].parse()
                .unwrap_or_else(|_| panic!("field {i} of `{line}`"))
        };
        // The DIRECT=false control is not an `Op`: nothing in the engine routes to it, so it is
        // resolved by name here instead of forcing a dead variant into every match on `Op`.
        if f[0] == "gemm_silu" {
            // The fused silu route is selected by the graph EDGE, not by the shape, so it is
            // resolved by name here the same way the DIRECT control is.
            let shape = Shape::new(n(1), n(2), n(3));
            push(&mut out, line, shape, gemm_decode_fused_silu(shape, &rules));
            continue;
        }
        if f[0] == "gemm_indirect" {
            let shape = Shape::new(n(1), n(2), n(3));
            push(&mut out, line, shape, gemm_decode_indirect(shape, &rules));
            continue;
        }
        // The activation layout is a property of the EDGE, so unlike WPERM it is per call and
        // belongs in the task line rather than in the rules file.
        let layout = if f[0] == "gemm_rm" {
            ALayout::RowMajor
        } else {
            ALayout::FollowsM
        };
        // The GDN norm's store is a TABLE decision (`gdn_norm_tiled`, G2), and prod only has the
        // row-major one: `gdn` pins the knob off so it is compared against prod's own bytes at
        // every M, `gdn_tiled` pins it on and is compared against the same bytes tiled.
        let mut task_rules = None;
        if f[0] == "gdn" || f[0] == "gdn_tiled" {
            let mut r = rules.clone();
            r.knobs.gdn_norm_tiled = f[0] == "gdn_tiled";
            task_rules = Some(r);
        }
        let rules = task_rules.as_ref().unwrap_or(&rules);
        let (op, shape) = match f[0] {
            "gemm" | "gemm_rm" => (
                Op::Gemm {
                    has_wref: true,
                    partitions: NO_PARTITIONS,
                },
                Shape::new(n(1), n(2), n(3)),
            ),
            // The layout is a per-PROCESS decision (the loader packs the tensor one way and the
            // kernels read the flag), so it arrives as a rules file, not as a task field.
            "gemm_wp0" => (
                Op::Gemm {
                    has_wref: true,
                    partitions: NO_PARTITIONS,
                },
                Shape::new(n(1), n(2), n(3)),
            ),
            "gemm_part" => (
                Op::Gemm {
                    has_wref: true,
                    partitions: Partitions {
                        pb1: n(4) as i32,
                        pb2: n(5) as i32,
                        astride: f[6].parse().expect("astride"),
                    },
                },
                Shape::new(n(1), n(2), n(3)),
            ),
            "gemm_nw" => (
                Op::Gemm {
                    has_wref: false,
                    partitions: NO_PARTITIONS,
                },
                Shape::new(n(1), n(2), n(3)),
            ),
            "add_rms" => (Op::AddRmsQuant { eps: fl(3) }, Shape::new(n(1), 0, n(2))),
            "silu" => (Op::SiluMulQuant, Shape::new(n(1), n(2), 0)),
            "gdn" | "gdn_tiled" => (
                Op::GdnNormQuant {
                    zs: n(3) as i64,
                    eps: fl(4),
                },
                Shape::new(n(1), n(2), 0),
            ),
            other => {
                eprintln!("saber-kplan: unknown op `{other}`");
                std::process::exit(2);
            }
        };
        push(
            &mut out,
            line,
            shape,
            resolve_with_layout(op, shape, rules, layout),
        );
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&out).expect("array serialises")
    );
}

/// One JSON row per task line, whether the rules resolved it or refused it: a refusal is a
/// RESULT the harness records and moves on from, not a crash.
fn push(
    out: &mut Vec<serde_json::Value>,
    line: &str,
    shape: Shape,
    r: saber_kernels::Result<LaunchDescriptor>,
) {
    match r {
        Ok(d) => {
            let mut v = serde_json::to_value(&d).expect("descriptor serialises");
            v["task"] = serde_json::Value::String(line.to_owned());
            v["m"] = shape.m.into();
            v["n"] = shape.n.into();
            v["k"] = shape.k.into();
            out.push(v);
        }
        Err(e) => out.push(serde_json::json!({
            "task": line, "m": shape.m, "n": shape.n, "k": shape.k, "error": e.to_string(),
        })),
    }
}
