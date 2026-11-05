//! `saber-packplan` — print the device-final record plan the tier pack must hold.
//!
//! `the internal reference harness --layout device-final` does the numerics; **this** says what to build:
//! which records exist, what layout each one is, and which checkpoint tensors feed it. The plan is
//! `saber-model`'s `Graph`, so the builder cannot drift from the model — there is no second list of
//! merges, and adding a layer or changing a merge changes the pack by construction.
//!
//! Same shape as `saber-kplan` / `saber-coreplan`: a tiny bin over a pure function, JSON on stdout.
//!
//! ```text
//! cargo run -q -p saber-model --bin saber-packplan -- --tp 2 > plan.json
//! cargo run -q -p saber-model --bin saber-packplan -- --dflash --tp 2 > drafter.json
//! ```
//!
//! `--dflash` prints the DRAFTER's record set instead of the target's:
//! [`saber_model::dflash_graph::DraftGraph::records`], which is the same idea one model over — the
//! pack builder cannot invent a drafter weight or forget one, because the set is the plan's. It
//! carries two things the target's plan does not need: a `checkpoint` per source (the DFlash2
//! checkpoint has 117 tensors and none of them is an embedding or a head, so six records come from
//! the TARGET's file), and the drafter's KV-cache geometry, which no checkpoint tensor carries and
//! a loader still has to size.
use std::process::ExitCode;

use saber_model::checkpoint_head::read_checkpoint_head;
use saber_model::dflash_graph::{DflashConfig, DraftGraph, DraftRope, RerankHead};
use saber_model::graph::{Graph, ModelConfig};

fn main() -> ExitCode {
    let mut tp = 2u32;
    let mut config: Option<String> = None;
    let mut dflash = false;
    // `auto` is the default since N6 (the user's law, 2026-09-22 ~22:30: the head is served at the
    // CHECKPOINT's own precision, never requantized): the checkpoint's `lm_head.weight` dtype and
    // the presence of its `weight_scale` pick `fp8` or `bf16` (`saber_model::checkpoint_head`).
    // `--head bf16|fp8|mxfp4` are explicit overrides; `mxfp4` is the `--mxfp4-head` control's pack.
    let mut head = "auto";
    let mut checkpoint: Option<String> = None;
    let mut spec = 7u32;
    let mut rope_max_pos = 4096u32;
    let mut kv_block = 16u32;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--tp" => match args.next().and_then(|v| v.parse().ok()) {
                Some(v) => tp = v,
                None => return fail("--tp needs a number"),
            },
            "--dflash" => dflash = true,
            // `bf16` (the default since N5): `lm_head.w` is prod's bf16 matrix (workstream N2).
            // `fp8` (N4): `lm_head.w8` + `lm_head.ws`, the checkpoint's e4m3 matrix and its
            // per-channel scale carried raw. `mxfp4`: the trio, the `--mxfp4-head` control's pack.
            "--head" => match args.next().as_deref() {
                Some("auto") => head = "auto",
                Some("bf16") => head = "bf16",
                Some("fp8") => head = "fp8",
                Some("mxfp4") => head = "mxfp4",
                _ => return fail("--head takes `auto` (the default), `bf16`, `fp8` or `mxfp4`"),
            },
            "--checkpoint" => match args.next() {
                Some(v) => checkpoint = Some(v),
                None => return fail("--checkpoint needs the target checkpoint's directory"),
            },
            "--spec" => match args.next().and_then(|v| v.parse().ok()) {
                Some(v) => spec = v,
                None => return fail("--spec needs a number"),
            },
            "--rope-max-pos" => match args.next().and_then(|v| v.parse().ok()) {
                Some(v) => rope_max_pos = v,
                None => return fail("--rope-max-pos needs a number"),
            },
            "--kv-block-size" => match args.next().and_then(|v| v.parse().ok()) {
                Some(v) => kv_block = v,
                None => return fail("--kv-block-size needs a number"),
            },
            "--config" => match args.next() {
                Some(v) => config = Some(v),
                None => return fail("--config needs a path to a checkpoint config.json"),
            },
            "-h" | "--help" => {
                eprintln!(
                    "usage: saber-packplan [--tp N] [--head auto|bf16|fp8|mxfp4 (default auto)] \
                     [--checkpoint DIR] [--config path/to/config.json]"
                );
                eprintln!(
                    "       saber-packplan --dflash [--tp N] [--spec K] [--rope-max-pos P] \
                     [--kv-block-size B] [--head auto|bf16|fp8] [--checkpoint TARGET_DIR]"
                );
                eprintln!(
                    "       `--head auto` reads the TARGET checkpoint's lm_head.weight dtype (and \
                     its weight_scale) from --checkpoint, or from --config's directory"
                );
                return ExitCode::SUCCESS;
            }
            other => return fail(&format!("unknown argument `{other}`")),
        }
    }

    // ---- `--head auto`: the checkpoint's own head (N6) --------------------------------------
    let head_dir = checkpoint.clone().or_else(|| {
        config.as_ref().and_then(|c| {
            std::path::Path::new(c)
                .parent()
                .map(|p| p.display().to_string())
        })
    });
    let mut head_evidence: Option<String> = None;
    if head == "auto" {
        let Some(dir) = head_dir else {
            return fail(
                "--head auto (the default) reads the target checkpoint's `lm_head.weight`: pass \
                 --checkpoint DIR (or --config DIR/config.json), or an explicit --head",
            );
        };
        match read_checkpoint_head(std::path::Path::new(&dir)) {
            Ok(ev) => {
                eprintln!("{}", ev.line());
                head = ev.head.pack_head();
                head_evidence = Some(ev.line());
            }
            Err(e) => return fail(&e),
        }
    }

    if dflash {
        let rerank = match head {
            "fp8" => RerankHead::W8,
            "bf16" => RerankHead::Bf16,
            other => {
                return fail(&format!(
                    "--dflash --head {other}: the drafter's rerank reads the exact head, `bf16` or \
                     `fp8` (the MXFP4 control's drafter pack is either)"
                ))
            }
        };
        return dflash_plan(tp, spec, rope_max_pos, kv_block, rerank, head_evidence);
    }

    let cfg = match config {
        Some(path) => match std::fs::read_to_string(&path)
            .map_err(|e| e.to_string())
            .and_then(|t| ModelConfig::from_config_json(&t).map_err(|e| e.to_string()))
        {
            Ok(c) => c,
            Err(e) => return fail(&e.to_string()),
        },
        None => ModelConfig::qwen38_27b(),
    };
    let graph = match Graph::qwen38_27b(cfg, tp).and_then(|g| match head {
        "bf16" => g.with_bf16_head(),
        "fp8" => g.with_fp8_head(),
        _ => Ok(g),
    }) {
        Ok(g) => g,
        Err(e) => return fail(&e.to_string()),
    };
    let records = match graph.device_final_records() {
        Ok(r) => r,
        Err(e) => return fail(&e.to_string()),
    };

    let rows: Vec<serde_json::Value> = records
        .iter()
        .map(|r| {
            serde_json::json!({
                "name": r.name,
                "layout_id": r.layout_id,
                "role": r.role.as_str(),
                "op": r.op,
                "dtype": r.dtype.manifest_name(),
                "shard_kind": r.shard.manifest_name(),
                "shape": r.shape,
                "transformed": r.is_transformed(),
                "sources": r.sources.iter().map(|s| serde_json::json!({
                    "weight": s.weight,
                    "quant": s.quant.as_str(),
                    "scale": s.scale,
                    "global_scale": s.global_scale,
                    "input_scale": s.input_scale,
                })).collect::<Vec<_>>(),
            })
        })
        .collect();
    let out = serde_json::json!({
        "producer": "saber-packplan",
        "tp_size": tp,
        "head": head,
        "head_auto": head_evidence,
        "unused_record_prefixes": graph
            .unused_records()
            .iter()
            .map(|u| serde_json::json!({"prefix": u.prefix, "reason": u.reason}))
            .collect::<Vec<_>>(),
        "records": rows,
    });
    println!("{}", serde_json::to_string(&out).expect("serialize"));
    ExitCode::SUCCESS
}

/// The drafter's record set as JSON, in the same shape `build_pack.py` reads for the target.
fn dflash_plan(
    tp: u32,
    spec: u32,
    rope_max_pos: u32,
    kv_block: u32,
    rerank: RerankHead,
    head_evidence: Option<String>,
) -> ExitCode {
    let cfg = DflashConfig::qwen38_dflash2(tp, spec);
    let graph = DraftGraph::build(cfg.clone()).with_rerank_head(rerank);
    let rope = DraftRope::qwen38_dflash2(rope_max_pos);
    let kv = graph.kv_cache(kv_block);
    let rows: Vec<serde_json::Value> = graph
        .records(rope)
        .iter()
        .map(|r| {
            let params: serde_json::Map<String, serde_json::Value> = r
                .params
                .iter()
                .map(|(k, v)| ((*k).to_owned(), serde_json::json!(v)))
                .collect();
            serde_json::json!({
                "name": r.name,
                "layout_id": r.layout_id,
                "role": r.role.as_str(),
                "op": r.op,
                "dtype": r.dtype.manifest_name(),
                "shard_kind": r.shard.manifest_name(),
                "shape": r.shape,
                "params": params,
                "sources": r.sources.iter().map(|s| serde_json::json!({
                    "checkpoint": s.checkpoint.as_str(),
                    "weight": s.weight,
                    "quant": s.quant.as_str(),
                    "scale": s.scale,
                })).collect::<Vec<_>>(),
            })
        })
        .collect();
    let out = serde_json::json!({
        "producer": "saber-packplan --dflash",
        "tp_size": tp,
        "rerank_head": match rerank {
            RerankHead::Bf16 => "bf16",
            RerankHead::W8 => "fp8",
        },
        "head_auto": head_evidence,
        "model": "Qwen3.8-27B-DFlash2-FP8",
        "block_size": cfg.block_size,
        "declared_launches": graph.total_launches(),
        "covered_launches": graph.budget(),
        "uncovered": graph.uncovered().iter().map(|(n, why, c)| serde_json::json!({
            "op": n, "count": c, "why": why,
        })).collect::<Vec<_>>(),
        "kv_cache": {
            "layers": kv.layers,
            "block_size": kv.block_size,
            "kv_heads": kv.kv_heads,
            "head_dim": kv.head_dim,
            "elem_bytes": kv.elem_bytes,
            "block_bytes": kv.block_bytes(),
            "caches_per_layer": 2,
            "note": "two separate contiguous caches per layer, which is what dflash_kv and \
                     dflash_attn take; prod's TRITON_ATTN allocates ONE interleaved tensor per \
                     layer and hands them two strided views of it",
        },
        "records": rows,
    });
    println!("{}", serde_json::to_string(&out).expect("serialize"));
    ExitCode::SUCCESS
}

fn fail(msg: &str) -> ExitCode {
    eprintln!("saber-packplan: {msg}");
    ExitCode::FAILURE
}
