//! D2: the vision tower through the EXECUTION PATH on one card -- the gate's Saber arm.
//!
//!   saber-vision --pack DIR --prebuilt DIR --pixels FILE --grid T,H,W --out DIR
//!                [--rank R] [--world W] [--device D] [--hip-lib SO] [--max-patches N]
//!                [--iters N] [--rules 27b|flashnext]
//!
//! FN55: `--rules flashnext` runs Flash-Next's tower (merger out 2,560) from the Flash-Next pack's
//! `vision/` member (`--pack <flashnext pack>/vision`).
//!
//! Boots what an image-serving engine boots (`saber_engine::vision::boot_vision`: the pack's 333
//! `model.visual.*` records through `saber_mem::upload::Upload`, the tower arena for
//! `--max-patches`), then runs the image from `--pixels` (prod's `pixel_values`, f32 `[L, 1536]`):
//!
//!   1. eager, once -> `rows.eager.bin` (`[tokens, 5120]` bf16, the arena's `out` slot);
//!   2. from the captured graph for this grid, once -> `rows.graph.bin`, which must be the SAME
//!      BYTES as the eager rows (the graph replays the same launches over the same arena);
//!   3. timing: eager and graph medians of `--iters` (the tower's launches only, tables and pixels
//!      already resident), the host-input upload, and a per-launch profile summed by kernel.
//!
//! `tools/capture/vision_exec_gate.py` holds `rows.*.bin` against the fixtures.

use std::path::PathBuf;
use std::sync::Arc;

use saber_exec::{LaunchCounters, LaunchCtx, Launcher, LoadCounters, PluginSet};
use saber_hip::{EventFlags, HipThread, StreamFlags};
use saber_kernels::vision::Grid;

fn main() {
    if let Err(e) = run() {
        eprintln!("saber-vision: {e}");
        std::process::exit(2);
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn run() -> Result<(), String> {
    let mut pack: Option<PathBuf> = None;
    let mut prebuilt = PathBuf::from("prebuilt/gfx1201");
    let mut pixels: Option<PathBuf> = None;
    let mut grids: Vec<Grid> = Vec::new();
    let mut out: Option<PathBuf> = None;
    let (mut rank, mut world, mut device) = (0u32, 1u32, 0i32);
    let mut hip_lib = saber_hip::DEFAULT_HIP_SONAME.to_owned();
    let mut max_patches = 65_536u32;
    let mut iters = 10usize;
    let mut flashnext = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--pack" => pack = Some(PathBuf::from(val()?)),
            "--prebuilt" => prebuilt = PathBuf::from(val()?),
            "--pixels" => pixels = Some(PathBuf::from(val()?)),
            "--out" => out = Some(PathBuf::from(val()?)),
            "--rank" => rank = val()?.parse().map_err(|e| format!("--rank: {e}"))?,
            "--world" => world = val()?.parse().map_err(|e| format!("--world: {e}"))?,
            "--device" => device = val()?.parse().map_err(|e| format!("--device: {e}"))?,
            "--hip-lib" => hip_lib = val()?,
            "--max-patches" => {
                max_patches = val()?.parse().map_err(|e| format!("--max-patches: {e}"))?
            }
            "--rules" => {
                flashnext = match val()?.as_str() {
                    "27b" => false,
                    "flashnext" => true,
                    v => return Err(format!("--rules 27b|flashnext, not {v}")),
                }
            }
            "--iters" => iters = val()?.parse().map_err(|e| format!("--iters: {e}"))?,
            "--grid" => {
                let v = val()?;
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
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    let pack = pack.ok_or("--pack is required")?;
    let out = out.ok_or("--out is required")?;
    let px_path = pixels.ok_or("--pixels is required")?;
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let raw = std::fs::read(&px_path).map_err(|e| format!("{}: {e}", px_path.display()))?;
    let px: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let hip = HipThread::spawn(&hip_lib).map_err(|e| e.to_string())?;
    let r = body(
        &hip,
        &pack,
        &prebuilt,
        &out,
        rank,
        world,
        device,
        max_patches,
        iters,
        &grids,
        &px,
        flashnext,
    );
    // the HIP thread is dropped before the process exits (CLAUDE.md: exit with a live HIP thread
    // segfaults)
    drop(hip);
    r
}

#[allow(clippy::too_many_arguments)]
fn body(
    hip: &HipThread,
    pack: &std::path::Path,
    prebuilt: &std::path::Path,
    out: &std::path::Path,
    rank: u32,
    world: u32,
    device: i32,
    max_patches: u32,
    iters: usize,
    grids: &[Grid],
    px: &[f32],
    flashnext: bool,
) -> Result<(), String> {
    hip.set_device(device).map_err(|e| e.to_string())?;
    let load = LoadCounters::new();
    let plugins = PluginSet::load_dir(hip, prebuilt, &load).map_err(|e| e.to_string())?;
    let functions = plugins.functions(hip).map_err(|e| e.to_string())?;
    let ctx = LaunchCtx::new(hip, &functions);
    let file = if pack.is_dir() {
        pack.join(format!("rank{rank}.pack"))
    } else {
        pack.to_path_buf()
    };
    let rules = if flashnext {
        saber_kernels::vision::VisionRules::flashnext()
    } else {
        saber_kernels::vision::VisionRules::prod()
    }
    .map_err(|e| e.to_string())?;
    let out_hidden = rules.model.out_hidden;
    let mut tier =
        saber_engine::vision::boot_vision_with(hip, &file, rank, world, max_patches, rules)
            .map_err(|e| e.to_string())?;
    println!(
        "vision resident: {} records, {} B weights, arena {} B for {} patches, upload {:.2} s",
        tier.records,
        tier.weight_bytes,
        tier.exec.arena_bytes(),
        max_patches,
        tier.upload_s
    );
    let stream = hip
        .create_stream(StreamFlags::NonBlocking)
        .map_err(|e| e.to_string())?;
    let counters: Arc<LaunchCounters> = LaunchCounters::new();
    let mut launcher = Launcher::new("vision", &counters);
    let plan = tier.exec.plan(grids).map_err(|e| e.to_string())?;
    let t_up = std::time::Instant::now();
    tier.exec
        .upload(hip, &stream, &plan, px)
        .map_err(|e| e.to_string())?;
    let upload_ms = t_up.elapsed().as_secs_f64() * 1e3;

    let read_rows = |tier: &saber_engine::vision::VisionTier<'_>,
                     o: &saber_exec::vision::VisionOut|
     -> Result<Vec<u8>, String> {
        let mut rows = vec![0u8; o.bytes as usize];
        tier.exec
            .download(hip, &stream, o.offset, &mut rows)
            .map_err(|e| e.to_string())?;
        Ok(rows)
    };
    let eo = tier
        .exec
        .launch_eager(&ctx, &mut launcher, &stream, &plan)
        .map_err(|e| e.to_string())?;
    hip.sync_stream(&stream).map_err(|e| e.to_string())?;
    let eager_rows = read_rows(&tier, &eo)?;
    std::fs::write(out.join("rows.eager.bin"), &eager_rows).map_err(|e| e.to_string())?;
    // zero the WHOLE arena and re-upload the inputs, so the graph run has to write every row it
    // is compared on (a graph that replayed nothing would read back zeros, not the eager rows)
    tier.exec.zero(hip, &stream).map_err(|e| e.to_string())?;
    tier.exec
        .upload(hip, &stream, &plan, px)
        .map_err(|e| e.to_string())?;
    let go = tier
        .exec
        .launch_graph(&ctx, &mut launcher, &stream, &plan)
        .map_err(|e| e.to_string())?;
    hip.sync_stream(&stream).map_err(|e| e.to_string())?;
    let graph_rows = read_rows(&tier, &go)?;
    std::fs::write(out.join("rows.graph.bin"), &graph_rows).map_err(|e| e.to_string())?;
    let differ = eager_rows
        .chunks_exact(2)
        .zip(graph_rows.chunks_exact(2))
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "rows: {} tokens x {out_hidden}, eager launches {}, graph nodes {}, graph vs eager: {} \
         of {} differ",
        eo.tokens,
        eo.launches,
        go.launches,
        differ,
        eager_rows.len() / 2
    );

    let ev0 = hip
        .create_event(EventFlags::Default)
        .map_err(|e| e.to_string())?;
    let ev1 = hip
        .create_event(EventFlags::Default)
        .map_err(|e| e.to_string())?;
    let mut timed =
        |graph: bool, tier: &mut saber_engine::vision::VisionTier<'_>| -> Result<f64, String> {
            let mut v = Vec::with_capacity(iters);
            let mut walls = Vec::with_capacity(iters);
            for _ in 0..iters + 2 {
                let t0 = std::time::Instant::now();
                hip.record_event(&ev0, &stream).map_err(|e| e.to_string())?;
                if graph {
                    tier.exec
                        .launch_graph(&ctx, &mut launcher, &stream, &plan)
                        .map_err(|e| e.to_string())?;
                } else {
                    tier.exec
                        .launch_eager(&ctx, &mut launcher, &stream, &plan)
                        .map_err(|e| e.to_string())?;
                }
                hip.record_event(&ev1, &stream).map_err(|e| e.to_string())?;
                hip.sync_event(&ev1).map_err(|e| e.to_string())?;
                walls.push(t0.elapsed().as_secs_f64() * 1e3);
                v.push(f64::from(
                    hip.elapsed_ms(&ev0, &ev1).map_err(|e| e.to_string())?,
                ));
            }
            let (mut v, mut walls) = (v.split_off(2), walls.split_off(2));
            let w = median(&mut walls);
            let d = median(&mut v);
            println!(
                "  {:5} device {:8.2} ms  host wall {:8.2} ms  (median of {iters})",
                if graph { "graph" } else { "eager" },
                d,
                w
            );
            Ok(d)
        };
    println!(
        "TIMING tower {} patches -> {} tokens, card {device}, host inputs upload {upload_ms:.2} ms",
        plan.patches, plan.tokens
    );
    let e_ms = timed(false, &mut tier)?;
    let g_ms = timed(true, &mut tier)?;
    let prof = tier
        .exec
        .launch_profiled(&ctx, &mut launcher, &stream, &plan)
        .map_err(|e| e.to_string())?;
    let mut fam: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();
    for (_, k, ms) in &prof {
        *fam.entry(k.clone()).or_default() += ms;
    }
    let mut fam: Vec<_> = fam.into_iter().collect();
    fam.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (k, ms) in &fam {
        println!("  profile {k:36} {ms:9.3} ms");
    }
    let tok_s = f64::from(plan.tokens) / (g_ms / 1e3);
    println!(
        "SUMMARY patches {} tokens {} eager_ms {e_ms:.3} graph_ms {g_ms:.3} tokens_per_s {tok_s:.0} \
         graph_vs_eager_differ {differ}",
        plan.patches, plan.tokens
    );
    drop(tier);
    Ok(())
}
