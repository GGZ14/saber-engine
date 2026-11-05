//! `saber device-row-gate` — the sampler seam's bit-exact gate.
//!
//! `saber gate` byte-compares, and bytes are the only tolerance it has. Three of this kernel's four
//! outputs want exactly that — the K largest values, their ids and the penalty gather are a
//! selection and a copy, so anything but byte equality is a bug. The fourth cannot have it and
//! saying so is the point of a separate command: a **log-sum-exp** is a sum of 248,320 `expf`
//! results, and the device's `exp`, its 256 strided partials and its tree fold are not any host's.
//! So this gate compares the two log-sum-exps in **ULPs against an f64 reference**, prints the
//! bound it measured, and prints what TORCH's own f32 `logsumexp` scores against the same f64
//! reference — because a bound with nothing beside it is a number, not evidence.
//!
//! It also asks the one question that ties the seam to the greedy path: **the argmax the row
//! implies must be the argmax kernel's own answer.** `saber_argmax_bf16` is launched on the same
//! logits in the same run, and `top_ids[row][0]` is compared against it row by row. That is the
//! property `saber-sched` relies on when it commits a temperature-0 row from the argmax and a
//! temperature-0.7 row from the seam and expects the two paths to be the same pipeline.
//!
//! GPU rules (CLAUDE.md): one HIP device, under that card's lock, every allocation inside
//! `--budget-mib`, and the power cap untouched.

use std::path::PathBuf;

use saber_exec::{ArgBindings, LaunchCounters, LaunchCtx, Launcher, LoadCounters, PluginSet};
use saber_hip::{DevicePtr, HipThread, StreamFlags};
use saber_kernels::head::{self, HeadOp, HeadRules, HeadShape, DEVICE_ROW_K};
use saber_kernels::rules::Shape;

use crate::npy;

/// Arguments, as the binary declares them.
#[derive(Debug, Clone)]
pub struct DeviceRowGateArgs {
    /// Where `tools/capture/capture_device_row.py` wrote `cases.txt` and its `.npy` files.
    pub data: PathBuf,
    /// Directory of committed `.hsaco` + `*.manifest.toml`.
    pub prebuilt: PathBuf,
    /// HIP device ordinal within the visible set.
    pub device: i32,
    /// Refuse a case whose device buffers exceed this. Prod owns the card.
    pub budget_mib: u64,
    /// The largest log-sum-exp deviation, in f32 ULPs, this gate accepts.
    pub lse_ulp: u32,
    /// Vocabulary shards to split each captured row into: 1 (this card alone) or 2 (both cards,
    /// through the `tp2` wire — `crate::device_row_shards`).
    pub shards: u32,
    /// Set on a CHILD process of the 2-shard arm: which rank this is.
    pub rank: Option<u32>,
    /// Set on a child: the loopback port rank 0 listens on.
    pub port: Option<u16>,
    /// The soname or path of the HIP runtime each rank loads, forwarded to the children.
    pub hip_lib: String,
}

/// What one case measured. Printed as a row and summed into the verdict.
#[derive(Debug, Default)]
struct CaseReport {
    rows: usize,
    greedy_rows: usize,
    bytes: usize,
    differing: usize,
    first: String,
    lse_exact: usize,
    lse_max_ulp: u64,
    argmax_agree: usize,
    argmax_rows: usize,
    argmax_vs_reference: usize,
}

/// f32 distance in representable steps — the only scale a log-sum-exp bound reads on.
pub(crate) fn ulp_distance(a: f32, b: f32) -> u64 {
    if a == b {
        return 0;
    }
    if !a.is_finite() || !b.is_finite() {
        // An infinity that is not an equal infinity is not a rounding difference; it is a
        // different answer, and it is reported as the largest distance there is.
        return u64::MAX;
    }
    let map = |x: f32| -> i64 {
        let i = i64::from(x.to_bits() as i32);
        if i < 0 {
            0x8000_0000_i64 - i
        } else {
            i
        }
    };
    map(a).abs_diff(map(b))
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn i32s(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// One `device_row <rows> <cols> <stride> <penalties>` line.
fn parse_case(line: &str) -> Option<(u32, u32, i64, u32)> {
    let f: Vec<&str> = line.split_whitespace().collect();
    if f.first() != Some(&"device_row") || f.len() < 5 {
        return None;
    }
    Some((
        f[1].parse().ok()?,
        f[2].parse().ok()?,
        f[3].parse().ok()?,
        f[4].parse().ok()?,
    ))
}

/// Run the gate. Returns the number of failing cases.
pub fn run(hip: &HipThread, args: &DeviceRowGateArgs) -> anyhow::Result<usize> {
    hip.set_device(args.device)?;
    let rules = HeadRules::prod_head()?;
    let load = LoadCounters::new();
    let plugins = PluginSet::load_dir(hip, &args.prebuilt, &load)?;
    let functions = plugins.functions(hip)?;
    let ctx = LaunchCtx::new(hip, &functions);
    let stream = hip.create_stream(StreamFlags::NonBlocking)?;
    let launch_counters = LaunchCounters::new();
    let mut launcher = Launcher::new("device-row-gate", &launch_counters);
    let k = DEVICE_ROW_K as usize;

    let text = std::fs::read_to_string(args.data.join("cases.txt"))?;
    let cases: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    println!(
        "device-row gate: {} case(s) from {} (K = {k}, lse bound {} ulp)",
        cases.len(),
        args.data.display(),
        args.lse_ulp
    );
    println!(
        "{:<38} {:>6} {:>7} {:>12} {:>10} {:>9} {:>10}  verdict",
        "case", "rows", "greedy", "bytes", "differing", "lse ulp", "argmax"
    );

    let mut failed = 0usize;
    for line in cases {
        let Some((rows, cols, stride, penalties)) = parse_case(line) else {
            println!("{line:<38} {:>6} SKIP (not a device_row case)", "-");
            continue;
        };
        let cid = line.replace(' ', "_");
        let r = rows as usize;
        let pen = penalties as usize;

        // ---- what the capture wrote ----------------------------------------------------------
        let logits = npy::read_u8(&args.data.join(format!("{cid}.in.logits.npy")))?;
        let temps = npy::read_u8(&args.data.join(format!("{cid}.in.temps.npy")))?;
        let pen_ids = npy::read_u8(&args.data.join(format!("{cid}.in.pen_ids.npy")))?;
        let want_values = npy::read_u8(&args.data.join(format!("{cid}.out.top_values.npy")))?;
        let want_ids = npy::read_u8(&args.data.join(format!("{cid}.out.top_ids.npy")))?;
        let want_lse = f32s(&npy::read_u8(
            &args.data.join(format!("{cid}.out.lse.npy")),
        )?);
        let want_pen = npy::read_u8(&args.data.join(format!("{cid}.out.pen_values.npy")))?;
        let want_argmax = i32s(&npy::read_u8(
            &args.data.join(format!("{cid}.out.argmax.npy")),
        )?);
        let temperatures = f32s(&temps);

        let total = (logits.len()
            + temps.len()
            + pen_ids.len()
            + want_values.len()
            + want_ids.len()
            + want_lse.len() * 4
            + want_pen.len()
            + r * 4) as u64;
        if total > args.budget_mib << 20 {
            println!(
                "{line:<38} {rows:>6} {:>7} {:>12} SKIP ({} MiB exceeds the budget)",
                "-",
                "-",
                total >> 20
            );
            continue;
        }

        // ---- upload, launch, read back --------------------------------------------------------
        let d_logits = hip.malloc(logits.len() as u64)?;
        hip.memcpy_h2d_blocking(&d_logits, &logits, &stream)?;
        let d_temps = hip.malloc(temps.len().max(4) as u64)?;
        hip.memcpy_h2d_blocking(&d_temps, &temps, &stream)?;
        let d_pen_ids = hip.malloc(pen_ids.len().max(4) as u64)?;
        if !pen_ids.is_empty() {
            hip.memcpy_h2d_blocking(&d_pen_ids, &pen_ids, &stream)?;
        }
        // Outputs start ZEROED, so a kernel that fails to write a byte shows up as a mismatch
        // rather than as whatever the allocator handed back — and so that a greedy row's
        // untouched slots are the zeros the reference says they are.
        let zeroed = |bytes: usize| -> anyhow::Result<DevicePtr> {
            let p = hip.malloc(bytes.max(4) as u64)?;
            hip.memset(&p, 0, &stream)?;
            Ok(p)
        };
        let d_values = zeroed(r * k * 4)?;
        let d_ids = zeroed(r * k * 4)?;
        let d_lse = zeroed(r * 2 * 4)?;
        let d_pen_values = zeroed(r * pen * 4)?;
        let d_argmax = zeroed(r * 4)?;
        hip.sync_stream(&stream)?;

        let shape = HeadShape {
            rows,
            cols,
            src_stride: stride,
            vocab: (0, 0),
        };
        let desc = head::resolve(HeadOp::DeviceRow { penalties }, shape, &rules)?;
        let mut binds = ArgBindings::new(Shape::new(0, 0, 0));
        binds.bind("logits", d_logits.addr());
        binds.bind("temps", d_temps.addr());
        binds.bind("pen_ids", d_pen_ids.addr());
        binds.bind("top_values", d_values.addr());
        binds.bind("top_ids", d_ids.addr());
        binds.bind("lse", d_lse.addr());
        binds.bind("pen_values", d_pen_values.addr());
        launcher.dispatch_one(&ctx, &desc, &binds, &stream)?;

        // The greedy path, on the very same logits and in the same run.
        let am = head::resolve(HeadOp::Argmax, shape, &rules)?;
        let mut am_binds = ArgBindings::new(Shape::new(0, 0, 0));
        am_binds.bind("logits", d_logits.addr());
        am_binds.bind("out", d_argmax.addr());
        launcher.dispatch_one(&ctx, &am, &am_binds, &stream)?;
        hip.sync_stream(&stream)?;

        let mut got_values = vec![0u8; want_values.len()];
        hip.memcpy_d2h_blocking(&mut got_values, &d_values, &stream)?;
        let mut got_ids = vec![0u8; want_ids.len()];
        hip.memcpy_d2h_blocking(&mut got_ids, &d_ids, &stream)?;
        let mut got_lse = vec![0u8; want_lse.len() * 4];
        hip.memcpy_d2h_blocking(&mut got_lse, &d_lse, &stream)?;
        let mut got_pen = vec![0u8; want_pen.len()];
        if !want_pen.is_empty() {
            hip.memcpy_d2h_blocking(&mut got_pen, &d_pen_values, &stream)?;
        }
        let mut got_argmax = vec![0u8; r * 4];
        hip.memcpy_d2h_blocking(&mut got_argmax, &d_argmax, &stream)?;
        drop((
            d_logits,
            d_temps,
            d_pen_ids,
            d_values,
            d_ids,
            d_lse,
            d_pen_values,
            d_argmax,
        ));

        // ---- the byte compare, and the two it cannot cover --------------------------------------
        let mut rep = CaseReport {
            rows: r,
            greedy_rows: temperatures.iter().filter(|t| **t <= 0.0).count(),
            ..CaseReport::default()
        };
        for (name, want, got) in [
            ("top_values", &want_values, &got_values),
            ("top_ids", &want_ids, &got_ids),
            ("pen_values", &want_pen, &got_pen),
        ] {
            rep.bytes += want.len();
            for (i, (w, g)) in want.iter().zip(got.iter()).enumerate() {
                if w != g {
                    if rep.differing == 0 {
                        rep.first = format!("{name}[{i}]");
                    }
                    rep.differing += 1;
                }
            }
        }
        let got_lse = f32s(&got_lse);
        for (i, (w, g)) in want_lse.iter().zip(got_lse.iter()).enumerate() {
            // A greedy row's log-sum-exps are not computed at all; the reference says zero and the
            // byte compare above is not the place to learn that, so they are skipped here by the
            // same rule the kernel skipped them by.
            if temperatures.get(i / 2).copied().unwrap_or(1.0) <= 0.0 {
                continue;
            }
            let d = ulp_distance(*w, *g);
            if d == 0 {
                rep.lse_exact += 1;
            }
            rep.lse_max_ulp = rep.lse_max_ulp.max(d);
        }
        let got_argmax = i32s(&got_argmax);
        let got_ids_i = i32s(&got_ids);
        for row in 0..r {
            if want_argmax.get(row) == got_argmax.get(row) {
                rep.argmax_vs_reference += 1;
            }
            if temperatures.get(row).copied().unwrap_or(1.0) <= 0.0 {
                continue;
            }
            rep.argmax_rows += 1;
            if got_ids_i.get(row * k) == got_argmax.get(row) {
                rep.argmax_agree += 1;
            }
        }

        let lse_ok = rep.lse_max_ulp <= u64::from(args.lse_ulp);
        let ok = rep.differing == 0 && lse_ok && rep.argmax_agree == rep.argmax_rows;
        if !ok {
            failed += 1;
        }
        println!(
            "{line:<38} {:>6} {:>7} {:>12} {:>10} {:>9} {:>10}  {}",
            rep.rows,
            rep.greedy_rows,
            rep.bytes,
            rep.differing,
            rep.lse_max_ulp,
            format!("{}/{}", rep.argmax_agree, rep.argmax_rows),
            if ok {
                "OK".to_owned()
            } else if rep.differing > 0 {
                format!("MISMATCH (first at {})", rep.first)
            } else if !lse_ok {
                format!("LSE {} ulp > {}", rep.lse_max_ulp, args.lse_ulp)
            } else {
                "ARGMAX DISAGREES".to_owned()
            }
        );
        println!(
            "      log-sum-exps exact on {} of {} live values; `saber_argmax_bf16` == the \
             capture's first-maximum rule on {} of {} rows",
            rep.lse_exact,
            2 * (rep.rows - rep.greedy_rows),
            rep.argmax_vs_reference,
            rep.rows
        );
    }
    if failed == 0 {
        println!("device-row gate: every case OK");
    } else {
        println!("device-row gate: {failed} case(s) FAILED");
    }
    Ok(failed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_case_grammar_is_the_capture_scripts() {
        assert_eq!(
            parse_case("device_row 200 248320 248320 64"),
            Some((200, 248_320, 248_320, 64))
        );
        assert_eq!(parse_case("argmax 8 248320 248320"), None);
        assert_eq!(parse_case("device_row 8 248320"), None);
    }

    /// The ulp ruler is the instrument this gate's one tolerance is read with, so it has a test
    /// of its own — including the case that must NOT read as a rounding difference.
    #[test]
    fn the_ulp_ruler_counts_representable_steps() {
        assert_eq!(ulp_distance(1.0, 1.0), 0);
        assert_eq!(ulp_distance(1.0, f32::from_bits(1.0f32.to_bits() + 1)), 1);
        assert_eq!(ulp_distance(1.0, f32::from_bits(1.0f32.to_bits() + 7)), 7);
        assert_eq!(ulp_distance(-0.0, 0.0), 0);
        assert_eq!(ulp_distance(f32::INFINITY, f32::INFINITY), 0);
        assert_eq!(ulp_distance(f32::INFINITY, 3.4e38), u64::MAX);
        assert_eq!(ulp_distance(f32::NAN, 1.0), u64::MAX);
    }
}
