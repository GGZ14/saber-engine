//! Emit the vision tower's plan for a set of image grids, plus the host metadata it reads.
//!
//!   saber-visplan --grid T,H,W [--grid T,H,W ...] --out DIR [--attn tiled|p16|reference]
//!                 [--arena MAX_PATCHES]
//!
//! Writes `DIR/plan.json` (the `TowerPlan`: every launch's symbol, grid, block and kernarg slots,
//! and every buffer by name -- pack records by their record name) and the metadata tables the
//! kernels read: `pos_idx.i32`, `pos_wts.f32`, `rope_cos.bf16`, `rope_sin.bf16`,
//! `cu_seqlens.i32`, little-endian, flat. Host-only; `tools/capture/vision_gate.py` runs it.
//!
//! D2: `--attn` names the attention arm (default the plan's tiled kernel; `p16` and `reference`
//! are the gate's arms), and `--arena N` also writes `arena.json`, the tower arena for images of
//! up to N patches (the layout `saber-exec`'s executor allocates).

use std::io::Write;
use std::path::PathBuf;

use saber_kernels::vision::{
    plan_tower_with, pos_table, rope_table, tower_arena, AttnArm, Grid, VisionRules,
};

fn main() {
    if let Err(e) = run() {
        eprintln!("saber-visplan: {e}");
        std::process::exit(2);
    }
}

fn write(path: PathBuf, bytes: &[u8]) -> Result<(), String> {
    std::fs::File::create(&path)
        .and_then(|mut f| f.write_all(bytes))
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn run() -> Result<(), String> {
    let mut grids = Vec::new();
    let mut out: Option<PathBuf> = None;
    let mut arm = AttnArm::Tiled;
    let mut arena: Option<u32> = None;
    let mut shipped_gemm = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--grid" => {
                let v = args.next().ok_or("--grid needs T,H,W")?;
                let n: Vec<u32> = v
                    .split(',')
                    .map(|x| x.parse::<u32>().map_err(|e| format!("--grid {v}: {e}")))
                    .collect::<Result<_, _>>()?;
                if n.len() != 3 {
                    return Err(format!("--grid {v}: want T,H,W"));
                }
                grids.push(Grid {
                    t: n[0],
                    h: n[1],
                    w: n[2],
                });
            }
            "--out" => out = Some(PathBuf::from(args.next().ok_or("--out needs a dir")?)),
            "--attn" => {
                arm = match args.next().as_deref() {
                    Some("tiled") => AttnArm::Tiled,
                    Some("p16") => AttnArm::TiledP16,
                    Some("reference") => AttnArm::Reference,
                    v => return Err(format!("--attn {v:?}: want tiled|p16|reference")),
                }
            }
            // D6: `shipped` plans every GEMM on the pre-D6 kernel (`gemm_tiles` emptied): the
            // control arm the tower's before/after time is taken against in the same hold.
            "--gemm" => {
                shipped_gemm = match args.next().as_deref() {
                    Some("lds") => false,
                    Some("shipped") => true,
                    v => return Err(format!("--gemm {v:?}: want lds|shipped")),
                }
            }
            "--arena" => {
                let v = args.next().ok_or("--arena needs MAX_PATCHES")?;
                arena = Some(v.parse().map_err(|e| format!("--arena {v}: {e}"))?);
            }
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    let out = out.ok_or("--out is required")?;
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let mut rules = VisionRules::prod().map_err(|e| e.to_string())?;
    if shipped_gemm {
        rules.kernels.gemm_tiles.clear();
    }
    let plan = plan_tower_with(&rules, &grids, arm).map_err(|e| e.to_string())?;
    if let Some(n) = arena {
        let a = tower_arena(&rules, n).map_err(|e| e.to_string())?;
        let json = serde_json::to_string_pretty(&a).map_err(|e| e.to_string())?;
        write(out.join("arena.json"), json.as_bytes())?;
        println!(
            "vision arena: {n} patches -> {} B ({:.3} GiB; unaliased {:.3} GiB)",
            a.bytes,
            a.bytes as f64 / (1u64 << 30) as f64,
            a.unaliased_bytes as f64 / (1u64 << 30) as f64
        );
    }
    let json = serde_json::to_string_pretty(&plan).map_err(|e| e.to_string())?;
    write(out.join("plan.json"), json.as_bytes())?;
    let (idx, wts) = pos_table(&rules, &grids);
    write(
        out.join("pos_idx.i32"),
        &idx.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    )?;
    write(
        out.join("pos_wts.f32"),
        &wts.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    )?;
    let (cos, sin) = rope_table(&rules, &grids);
    write(
        out.join("rope_cos.bf16"),
        &cos.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    )?;
    write(
        out.join("rope_sin.bf16"),
        &sin.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    )?;
    write(
        out.join("cu_seqlens.i32"),
        &plan
            .cu_seqlens
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    println!(
        "vision plan: {} patches -> {} tokens, {} launches, {} buffers -> {}",
        plan.patches,
        plan.tokens,
        plan.launches.len(),
        plan.buffers.len(),
        out.display()
    );
    Ok(())
}
