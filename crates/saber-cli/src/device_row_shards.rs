//! `saber device-row-gate --shards 2` — the sampler's seam at TWO RANKS, against the one-rank
//! gate's own captured rows.
//!
//! The question this asks is the only one that separates a TP = 2 seam from a TP = 1 one: **the
//! rows are the same logits, split down the middle of the vocabulary.** Each rank runs
//! `saber_device_row_bf16` over its own half, the follower's half crosses the real `tp2` wire
//! ([`saber_engine::tp2::RankLink::send_seam`]), and the leader merges the two with the shipped
//! law (`saber_serve::sampler::merge_shards`). What comes out must be what the undivided kernel
//! wrote — the same reference `tools/capture/capture_device_row.py` computed for the one-rank
//! gate, compared the same two ways:
//!
//! * `top_values`, `top_ids` and the penalty gather **by bytes**. A merge is a selection and a
//!   copy; anything but byte equality is a bug, ties included;
//! * the two log-sum-exps in **ULPs**, because a shard's log-sum-exp is a sum the undivided
//!   kernel never took and `log(exp(a) + exp(b))` rounds once more on top of it.
//!
//! It also exercises the frame in the direction the numbers cannot see: the leader sends the
//! temperatures and the penalty ids with [`saber_engine::tp2::RankLink::lead_seam`] and the
//! follower asserts that what arrived is what the capture says — a wire that dropped a row's
//! temperature would otherwise show up only as a greedy row nobody asked for.
//!
//! GPU rules (`CLAUDE.md`): two PROCESSES on devices 0 and 1, under both cards' locks, every
//! allocation inside `--budget-mib`, and the power cap untouched.

use std::sync::Arc;
use std::time::{Duration, Instant};

use saber_engine::tp2::{RankLink, SeamGeometry, SeamOutputs};
use saber_exec::{ArgBindings, LaunchCounters, LaunchCtx, Launcher, LoadCounters, PluginSet};
use saber_exec::{RankPeer, TcpExchange};
use saber_hip::{DevicePtr, HipThread, StreamFlags};
use saber_kernels::head::{self, HeadOp, HeadRules, HeadShape, DEVICE_ROW_K};
use saber_kernels::rules::Shape;
use saber_serve::sampler::{merge_shards, SeamShard};

use crate::device_row_gate::{ulp_distance, DeviceRowGateArgs};
use crate::npy;

/// How long the parent waits for a rank process before declaring it wedged. Two ranks, a handful
/// of cases, one kernel each: a minute is four orders of magnitude of headroom.
const RANK_DEADLINE: Duration = Duration::from_secs(300);

/// The penalty width the engine's head plans, restated here because this gate binds the same
/// kernarg the engine does and a second number would be a second thing to be wrong.
const PENALTY_IDS: u32 = 256;

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

/// The parent: pick a port, spawn one process per rank, wait with a deadline.
///
/// TWO PROCESSES and not two threads, for the reason `ar-gate` gives: this is the transport the
/// engine uses, and a gate that ran the protocol over a channel would be gating a channel.
pub fn run_parent(args: &DeviceRowGateArgs) -> anyhow::Result<usize> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")
        .map_err(|e| anyhow::anyhow!("binding the rendezvous socket: {e}"))?;
    let port = listener.local_addr()?.port();
    drop(listener);

    println!(
        "device-row gate, 2 SHARDS: the one-rank gate's own rows, split down the middle of the \
         vocabulary, on devices 0 and 1 (K = {} per shard, lse bound {} ulp)",
        DEVICE_ROW_K, args.lse_ulp
    );
    println!("               reference: {}", args.data.display());
    println!();

    let exe = std::env::current_exe()?;
    let mut children = Vec::new();
    for rank in [0u32, 1] {
        let mut cmd = std::process::Command::new(&exe);
        cmd.arg("device-row-gate")
            .arg("--shards")
            .arg("2")
            .arg("--rank")
            .arg(rank.to_string())
            .arg("--port")
            .arg(port.to_string())
            .arg("--data")
            .arg(&args.data)
            .arg("--prebuilt")
            .arg(&args.prebuilt)
            .arg("--budget-mib")
            .arg(args.budget_mib.to_string())
            .arg("--lse-ulp")
            .arg(args.lse_ulp.to_string())
            .arg("--hip-lib")
            .arg(&args.hip_lib);
        children.push((rank, cmd.spawn()?));
    }

    let deadline = Instant::now() + RANK_DEADLINE;
    let mut failed = 0usize;
    for (rank, mut child) in children {
        loop {
            match child.try_wait()? {
                Some(status) => {
                    if !status.success() {
                        println!("rank {rank}: exited {status}");
                        failed += 1;
                    }
                    break;
                }
                None if Instant::now() > deadline => {
                    // A rank that never returns is the failure this deadline exists for. Kill it
                    // so its context — and anything still spinning in it — is torn down rather
                    // than left on a card.
                    println!("rank {rank}: no result within {RANK_DEADLINE:?}; killing it");
                    let _ = child.kill();
                    let _ = child.wait();
                    failed += 1;
                    break;
                }
                None => std::thread::sleep(Duration::from_millis(50)),
            }
        }
    }
    if failed == 0 {
        println!("device-row gate (2 shards): both ranks OK");
    }
    Ok(failed)
}

/// What one case measured on the leader.
#[derive(Debug, Default)]
struct ShardReport {
    rows: usize,
    greedy_rows: usize,
    merged_rows: usize,
    bytes: usize,
    differing: usize,
    first: String,
    lse_exact: usize,
    lse_max_ulp: u64,
    control_differing: usize,
}

/// One rank's half of the gate.
#[allow(clippy::too_many_lines)]
pub fn run_rank(
    hip: &HipThread,
    args: &DeviceRowGateArgs,
    rank: u32,
    port: u16,
) -> anyhow::Result<usize> {
    // Each rank drives its OWN card, and the device ordinal is the rank: this is the two-card
    // shape the serving engine has, not two contexts on one card.
    hip.set_device(rank as i32)?;
    let rules = HeadRules::prod_head()?;
    let load = LoadCounters::new();
    let plugins = PluginSet::load_dir(hip, &args.prebuilt, &load)?;
    let functions = plugins.functions(hip)?;
    let ctx = LaunchCtx::new(hip, &functions);
    let stream = hip.create_stream(StreamFlags::NonBlocking)?;
    let launch_counters = LaunchCounters::new();
    let mut launcher = Launcher::new("device-row-shards", &launch_counters);
    let k = DEVICE_ROW_K as usize;

    // The wire: the same `TcpExchange` the engine's two ranks meet on, wrapped in the same
    // `RankLink`. `payload_bytes` is 0 because this gate sends no step frames — only the two the
    // seam added.
    let timeout = Duration::from_secs(120);
    let exchange = if rank == 0 {
        let listener = std::net::TcpListener::bind(("127.0.0.1", port))?;
        TcpExchange::accept(&listener, timeout)
    } else {
        TcpExchange::connect(port, timeout)
    }
    .map_err(|e| anyhow::anyhow!("rank {rank}: meeting the peer on 127.0.0.1:{port}: {e}"))?;
    let peer = Arc::new(RankPeer::new(Arc::new(exchange)));
    let geom = SeamGeometry {
        k: DEVICE_ROW_K,
        penalties: PENALTY_IDS,
    };
    let link = RankLink::new(Arc::clone(&peer), 0, 0, geom);

    let text = std::fs::read_to_string(args.data.join("cases.txt"))?;
    let cases: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    if rank == 0 {
        println!(
            "{:<38} {:>6} {:>7} {:>8} {:>12} {:>10} {:>9}  verdict",
            "case (2 shards)", "rows", "greedy", "merged", "bytes", "differing", "lse ulp"
        );
    }

    let mut failed = 0usize;
    for line in cases {
        let Some((rows, cols, stride, penalties)) = parse_case(line) else {
            continue;
        };
        if cols % 2 != 0 {
            if rank == 0 {
                println!("{line:<38} SKIP (an odd vocabulary cannot be halved)");
            }
            peer.rendezvous(&format!("case.{line}"))?;
            continue;
        }
        let cid = line.replace(' ', "_");
        let r = rows as usize;
        let pen = penalties as usize;
        let half = cols / 2;
        let base = rank * half;

        // ---- what the capture wrote, read by BOTH ranks --------------------------------------
        let logits = npy::read_u8(&args.data.join(format!("{cid}.in.logits.npy")))?;
        let temps = f32s(&npy::read_u8(
            &args.data.join(format!("{cid}.in.temps.npy")),
        )?);
        let pen_ids = i32s(&npy::read_u8(
            &args.data.join(format!("{cid}.in.pen_ids.npy")),
        )?);

        // This rank's half of every row, repacked contiguous: `[rows, half]` bf16.
        let mut mine = vec![0u8; r * half as usize * 2];
        for row in 0..r {
            let src = (row * stride as usize + base as usize) * 2;
            let dst = row * half as usize * 2;
            let n = half as usize * 2;
            if src + n > logits.len() {
                anyhow::bail!("{line}: the capture's logits are shorter than {rows}x{stride}");
            }
            mine[dst..dst + n].copy_from_slice(&logits[src..src + n]);
        }
        // The penalty ids, as COLUMNS of this rank's shard. `-1` is the kernel's "not mine", and
        // the merge takes that id's logit from the rank that does hold it.
        let local_pen: Vec<i32> = pen_ids
            .iter()
            .map(|id| {
                if *id >= base as i32 && *id < (base + half) as i32 {
                    *id - base as i32
                } else {
                    -1
                }
            })
            .collect();

        let total =
            (mine.len() + temps.len() * 4 + local_pen.len() * 4 + r * (2 * k + 2 + pen) * 4) as u64;
        if total > args.budget_mib << 20 {
            if rank == 0 {
                println!(
                    "{line:<38} {rows:>6} SKIP ({} MiB exceeds the budget)",
                    total >> 20
                );
            }
            peer.rendezvous(&format!("case.{line}"))?;
            continue;
        }

        // ---- THE FRAME, in the direction the numbers cannot see -------------------------------
        // The leader tells the follower what it is sampling; the follower checks that against the
        // capture it read itself. A wire that lost a temperature would otherwise read as a greedy
        // row and pass every byte compare below.
        let penalised = !pen_ids.is_empty() && pen > 0;
        if rank == 0 {
            link.lead_seam(
                rows,
                &saber_engine::tp2::SeamInputs {
                    temps: temps.clone(),
                    pen_ids: if penalised {
                        widen_pen(&pen_ids, r, pen)
                    } else {
                        Vec::new()
                    },
                },
            )?;
        } else {
            let got = link.follow_seam(rows, penalised)?;
            if got.temps != temps {
                anyhow::bail!(
                    "{line}: the temperatures that crossed the wire are not the capture's"
                );
            }
            if penalised && got.pen_ids != widen_pen(&pen_ids, r, pen) {
                anyhow::bail!(
                    "{line}: the penalty ids that crossed the wire are not the capture's"
                );
            }
        }

        // ---- upload, launch, read back --------------------------------------------------------
        let d_logits = hip.malloc(mine.len() as u64)?;
        hip.memcpy_h2d_blocking(&d_logits, &mine, &stream)?;
        let mut tbytes: Vec<u8> = Vec::with_capacity(temps.len() * 4);
        for t in &temps {
            tbytes.extend_from_slice(&t.to_le_bytes());
        }
        let d_temps = hip.malloc(tbytes.len().max(4) as u64)?;
        hip.memcpy_h2d_blocking(&d_temps, &tbytes, &stream)?;
        let mut pbytes: Vec<u8> = Vec::with_capacity(local_pen.len() * 4);
        for id in &local_pen {
            pbytes.extend_from_slice(&id.to_le_bytes());
        }
        let d_pen_ids = hip.malloc(pbytes.len().max(4) as u64)?;
        if !pbytes.is_empty() {
            hip.memcpy_h2d_blocking(&d_pen_ids, &pbytes, &stream)?;
        }
        // Outputs start ZEROED, so a kernel that fails to write a byte shows up as a mismatch and
        // not as whatever the allocator handed back.
        let zeroed = |bytes: usize| -> anyhow::Result<DevicePtr> {
            let p = hip.malloc(bytes.max(4) as u64)?;
            hip.memset(&p, 0, &stream)?;
            Ok(p)
        };
        let d_values = zeroed(r * k * 4)?;
        let d_ids = zeroed(r * k * 4)?;
        let d_lse = zeroed(r * 2 * 4)?;
        let d_pen_values = zeroed(r * pen * 4)?;
        hip.sync_stream(&stream)?;

        let shape = HeadShape {
            rows,
            cols: half,
            src_stride: i64::from(half),
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
        hip.sync_stream(&stream)?;

        let mut got_values = vec![0u8; r * k * 4];
        hip.memcpy_d2h_blocking(&mut got_values, &d_values, &stream)?;
        let mut got_ids = vec![0u8; r * k * 4];
        hip.memcpy_d2h_blocking(&mut got_ids, &d_ids, &stream)?;
        let mut got_lse = vec![0u8; r * 2 * 4];
        hip.memcpy_d2h_blocking(&mut got_lse, &d_lse, &stream)?;
        let mut got_pen = vec![0u8; r * pen * 4];
        if pen > 0 {
            hip.memcpy_d2h_blocking(&mut got_pen, &d_pen_values, &stream)?;
        }
        drop((
            d_logits,
            d_temps,
            d_pen_ids,
            d_values,
            d_ids,
            d_lse,
            d_pen_values,
        ));

        let outputs = SeamOutputs {
            values: f32s(&got_values),
            ids: i32s(&got_ids),
            lse: f32s(&got_lse),
            pen: f32s(&got_pen),
        };

        // ---- the wire, and the merge ----------------------------------------------------------
        // The FRAME is the engine's width, not this case's: the served head gathers
        // `PENALTY_IDS` per row whatever a request asks for, and a gate that sent its own narrower
        // table would be gating a geometry nothing runs. So the gather is widened on the way out
        // and narrowed on the way in, and the capture's own width is what is compared.
        if rank == 1 {
            link.send_seam(
                rows,
                &SeamOutputs {
                    pen: widen_f32(&outputs.pen, r, pen),
                    ..outputs.clone()
                },
            )?;
            peer.rendezvous(&format!("case.{line}"))?;
            continue;
        }
        let mut theirs = link.recv_seam(rows, pen > 0)?;
        theirs.pen = narrow_f32(&theirs.pen, r, pen);

        let want_values = f32s(&npy::read_u8(
            &args.data.join(format!("{cid}.out.top_values.npy")),
        )?);
        let want_ids = i32s(&npy::read_u8(
            &args.data.join(format!("{cid}.out.top_ids.npy")),
        )?);
        let want_lse = f32s(&npy::read_u8(
            &args.data.join(format!("{cid}.out.lse.npy")),
        )?);
        let want_pen = f32s(&npy::read_u8(
            &args.data.join(format!("{cid}.out.pen_values.npy")),
        )?);

        let mut rep = ShardReport {
            rows: r,
            greedy_rows: temps.iter().filter(|t| **t <= 0.0).count(),
            ..ShardReport::default()
        };
        let ids_of_row = |row: usize, pen_list: &[i32]| -> Vec<u32> {
            pen_list[row * pen..row * pen + pen]
                .iter()
                .filter(|id| **id >= 0)
                .map(|id| *id as u32)
                .collect()
        };
        for row in 0..r {
            if temps[row] <= 0.0 {
                // A greedy row is skipped by the kernel on BOTH ranks, and the merge never runs.
                // Asserted rather than assumed: `-1` in the first id is the kernel's refusal.
                if outputs.ids[row * k] != -1 || theirs.ids[row * k] != -1 {
                    rep.differing += 1;
                    if rep.first.is_empty() {
                        rep.first = format!("greedy row {row} was not skipped");
                    }
                }
                continue;
            }
            let row_pen: Vec<u32> = if pen > 0 {
                ids_of_row(row, &pen_ids)
            } else {
                Vec::new()
            };
            let shard = |o: &SeamOutputs, b: u32| -> SeamShard {
                let at = row * k;
                let ids = &o.ids[at..at + k];
                let live = ids
                    .iter()
                    .position(|id| *id < 0 || *id as u32 >= half)
                    .unwrap_or(k);
                SeamShard {
                    base: b,
                    cols: half,
                    top_values: o.values[at..at + live].to_vec(),
                    top_cols: ids[..live].iter().map(|id| *id as u32).collect(),
                    logsumexp: o.lse[row * 2],
                    logsumexp_at_temperature: Some(o.lse[row * 2 + 1]),
                    penalised: row_pen
                        .iter()
                        .enumerate()
                        .map(|(i, id)| (*id >= b && *id < b + half).then(|| o.pen[row * pen + i]))
                        .collect(),
                }
            };
            let shards = [shard(&outputs, 0), shard(&theirs, half)];
            let merged = merge_shards(&shards, &row_pen, k, cols as usize)
                .map_err(|e| anyhow::anyhow!("{line}: row {row}: {e}"))?;
            rep.merged_rows += 1;

            // THE CONTROL, on the same rows and costing no card time: merge the two halves with
            // their BASES SWAPPED — each rank's columns read as the other's tokens. It is still a
            // legal tiling, so `merge_shards` accepts it, and it is exactly the way a two-shard
            // merge goes wrong: the values are the row's and the ids are not. A gate whose control
            // cannot fail is not a gate (CLAUDE.md rule 8), and this one is counted beside the
            // case rather than run as a separate arm nobody remembers to pass a flag for.
            let swapped = [shard(&theirs, 0), shard(&outputs, half)];
            if let Ok(wrong) = merge_shards(&swapped, &row_pen, k, cols as usize) {
                for (i, id) in wrong.top_ids.iter().enumerate() {
                    if *id as i32 != want_ids[row * k + i] {
                        rep.control_differing += 1;
                    }
                }
            }

            // Values and ids BY BYTES against the undivided kernel's own answer.
            for (i, v) in merged.top_values.iter().enumerate() {
                rep.bytes += 4;
                if v.to_bits() != want_values[row * k + i].to_bits() {
                    if rep.first.is_empty() {
                        rep.first = format!("top_values[{row}][{i}]");
                    }
                    rep.differing += 1;
                }
            }
            for (i, id) in merged.top_ids.iter().enumerate() {
                rep.bytes += 4;
                if *id as i32 != want_ids[row * k + i] {
                    if rep.first.is_empty() {
                        rep.first = format!("top_ids[{row}][{i}]");
                    }
                    rep.differing += 1;
                }
            }
            for (i, (_, v)) in merged.penalised.iter().enumerate() {
                rep.bytes += 4;
                if v.to_bits() != want_pen[row * pen + i].to_bits() {
                    if rep.first.is_empty() {
                        rep.first = format!("pen_values[{row}][{i}]");
                    }
                    rep.differing += 1;
                }
            }
            for (i, got) in [
                merged.logsumexp,
                merged.logsumexp_at_temperature.unwrap_or(f32::NAN),
            ]
            .into_iter()
            .enumerate()
            {
                let d = ulp_distance(want_lse[row * 2 + i], got);
                if d == 0 {
                    rep.lse_exact += 1;
                }
                rep.lse_max_ulp = rep.lse_max_ulp.max(d);
            }
        }

        let lse_ok = rep.lse_max_ulp <= u64::from(args.lse_ulp);
        // The control has to FIRE, or the case proves nothing about the merge.
        let control_fired = rep.merged_rows == 0 || rep.control_differing > 0;
        let ok = rep.differing == 0 && lse_ok && control_fired;
        if !ok {
            failed += 1;
        }
        println!(
            "{line:<38} {:>6} {:>7} {:>8} {:>12} {:>10} {:>9}  {}",
            rep.rows,
            rep.greedy_rows,
            rep.merged_rows,
            rep.bytes,
            rep.differing,
            rep.lse_max_ulp,
            if ok {
                "OK".to_owned()
            } else if rep.differing > 0 {
                format!("MISMATCH (first at {})", rep.first)
            } else if !lse_ok {
                format!("LSE {} ulp > {}", rep.lse_max_ulp, args.lse_ulp)
            } else {
                "THE CONTROL DID NOT FIRE".to_owned()
            }
        );
        println!(
            "      merged log-sum-exps exact on {} of {} live values; base-swap control \
             {} of {} ids differ",
            rep.lse_exact,
            2 * rep.merged_rows,
            rep.control_differing,
            rep.merged_rows * k
        );
        peer.rendezvous(&format!("case.{line}"))?;
    }
    if rank == 0 {
        if failed == 0 {
            println!("device-row gate (2 shards): every case OK");
        } else {
            println!("device-row gate (2 shards): {failed} case(s) FAILED");
        }
    }
    Ok(failed)
}

/// The capture's `[rows, pen]` gather widened to the frame's `[rows, PENALTY_IDS]`, zero padded —
/// a slot with no id is a slot the kernel writes 0 into, so this is what the head really holds.
fn widen_f32(values: &[f32], rows: usize, pen: usize) -> Vec<f32> {
    let width = PENALTY_IDS as usize;
    let mut out = vec![0.0f32; rows * width];
    for row in 0..rows {
        for i in 0..pen.min(width) {
            out[row * width + i] = values[row * pen + i];
        }
    }
    out
}

/// The inverse: the frame's `[rows, PENALTY_IDS]` back to the capture's own width.
fn narrow_f32(values: &[f32], rows: usize, pen: usize) -> Vec<f32> {
    let width = PENALTY_IDS as usize;
    let mut out = Vec::with_capacity(rows * pen);
    for row in 0..rows {
        for i in 0..pen {
            out.push(values.get(row * width + i).copied().unwrap_or(0.0));
        }
    }
    out
}

/// The capture's `[rows, pen]` id table widened to the frame's `[rows, PENALTY_IDS]`, `-1` padded.
fn widen_pen(pen_ids: &[i32], rows: usize, pen: usize) -> Vec<i32> {
    let width = PENALTY_IDS as usize;
    let mut out = vec![-1i32; rows * width];
    for row in 0..rows {
        for i in 0..pen.min(width) {
            out[row * width + i] = pen_ids[row * pen + i];
        }
    }
    out
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
    }

    /// The widened table is `-1` past the capture's own width, which is what the kernel reads as
    /// "no id here" — a 0 would be token 0's logit gathered into every unused slot.
    #[test]
    fn the_penalty_table_is_padded_with_minus_one() {
        let out = widen_pen(&[7, 9, 11, 13], 2, 2);
        assert_eq!(out.len(), 2 * PENALTY_IDS as usize);
        assert_eq!(&out[0..3], &[7, 9, -1]);
        assert_eq!(
            &out[PENALTY_IDS as usize..PENALTY_IDS as usize + 3],
            &[11, 13, -1]
        );
    }
}
