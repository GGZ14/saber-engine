//! `saber ar-gate` — the two-rank all-reduce, through the engine's own launcher.
//!
//! This is the route the `IpcPeerVisible` workspace kind exists for, and the first one whose
//! correctness depends on another rank. Two OS threads, each with its own
//! [`saber_hip::HipThread`] bound to its own card, exchange IPC handles through a
//! [`saber_exec::RankPeer`] and then launch the descriptor `saber::resolve` produced — the same
//! loader, the same launcher, the same kernarg packing every other route uses.
//!
//! # What this gate proves, and what it does not
//!
//! the reference all-reduce gate (`the internal reference harness`) is and remains the **tier-1** gate: it calls the reference's own entry
//! point in the same process and byte-compares Saber's replay against it. It dumps nothing, so
//! there is no reference on disk for a Rust harness to read, and the engine cannot call saber
//! (the `unsafe extern "C"` rule keeps FFI inside `saber-hip`).
//!
//! So this gate asserts what it can assert *honestly*, and labels every verdict with which:
//!
//! | Tier | Check | Applies to |
//! |---|---|---|
//! | 1 | **host reference**: `out = to_T(f32(a) + f32(b))`, elementwise, which is exactly what the exact kernel computes (`vadd` in `ref_ar_oneshot_2rank_exact.hip:51`) | the exact family |
//! | 1 | **cross-rank identity**: both ranks must produce the same bytes | every case |
//! | 2 | **bounded numerics** against the same host sum, with a stated bound **and a negative control** | the WHT-quantised family, where bit-identity is impossible by construction |
//!
//! A tier-2 verdict here is not a weaker version of the python gate — it is a different question.
//! "Did the rotation and the 6-bit pack land within their error budget" is answerable here; "is
//! this byte-for-byte what prod's saber produces" is only answerable there.
//!
//! # Never wedge the GPU
//!
//! A one-shot all-reduce whose ranks disagree does not answer wrongly, it never retires — and it
//! takes the queue with it, on cards prod is serving from. Three guards, in order of preference:
//!
//! 1. the ranks compare a protocol digest **before** either launches
//!    ([`saber_exec::Launcher::collective_digest`]), so a mismatch is a message;
//! 2. every host wait is bounded — the handle exchange, the rendezvous, and the post-launch
//!    completion wait, which polls `hipStreamQuery` against a deadline instead of blocking in
//!    `hipStreamSynchronize`;
//! 3. on a timeout the gate stops, reports, and returns non-zero, so the process exits and the
//!    context (and with it the spinning kernel) is torn down.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use saber_exec::{
    ArgBindings, LaunchCounters, LaunchCtx, Launcher, LoadCounters, PluginSet, RankPeer,
    TcpExchange,
};
use saber_hip::{HipThread, Stream, StreamFlags};
use saber_kernels::rules::Shape as GemmShape;
use saber_kernels::saber::{self, ArDtype, ArNqShape, ArShape, SaberRules};
use saber_kernels::LaunchDescriptor;

/// How long one case may take on the device before the gate gives up on it.
///
/// Generous for the work (the 80 MiB message is milliseconds) and short enough that a spinning
/// kernel is noticed in seconds rather than left on a card prod is using.
const CASE_DEADLINE: Duration = Duration::from_secs(20);

/// Payload element type of an all-reduce message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Payload {
    Bf16,
    Fp16,
    Fp32,
}

impl Payload {
    fn parse(s: &str) -> anyhow::Result<Payload> {
        Ok(match s {
            "bf16" => Payload::Bf16,
            "fp16" => Payload::Fp16,
            "fp32" => Payload::Fp32,
            other => anyhow::bail!("unknown payload `{other}`"),
        })
    }

    fn bytes(self) -> u64 {
        match self {
            Payload::Bf16 | Payload::Fp16 => 2,
            Payload::Fp32 => 4,
        }
    }

    fn ar_dtype(self) -> anyhow::Result<ArDtype> {
        Ok(match self {
            Payload::Bf16 => ArDtype::Bf16,
            Payload::Fp16 => ArDtype::Fp16,
            Payload::Fp32 => anyhow::bail!("the quantised family takes bf16 or fp16 only"),
        })
    }

    /// Encode one f32 into this payload's bytes, with the same round-to-nearest-even the kernel's
    /// `rd_from_f` uses.
    fn encode(self, v: f32, out: &mut Vec<u8>) {
        match self {
            Payload::Bf16 => out.extend_from_slice(&f32_to_bf16_bits(v).to_le_bytes()),
            Payload::Fp16 => out.extend_from_slice(&f32_to_fp16_bits(v).to_le_bytes()),
            Payload::Fp32 => out.extend_from_slice(&v.to_le_bytes()),
        }
    }

    /// Decode one element from this payload's bytes.
    fn decode(self, b: &[u8]) -> f32 {
        match self {
            Payload::Bf16 => bf16_bits_to_f32(u16::from_le_bytes([b[0], b[1]])),
            Payload::Fp16 => fp16_bits_to_f32(u16::from_le_bytes([b[0], b[1]])),
            Payload::Fp32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        }
    }
}

/// f32 -> bf16, round-to-nearest-even: the top 16 bits with the tie broken up.
fn f32_to_bf16_bits(v: f32) -> u16 {
    let u = v.to_bits();
    if v.is_nan() {
        return ((u >> 16) as u16) | 0x0040;
    }
    (((u + 0x7fff + ((u >> 16) & 1)) >> 16) & 0xffff) as u16
}

fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// f32 -> fp16, round-to-nearest-even.
///
/// Written the long way on purpose. This conversion *is* the gate for the fp16 exact case — the
/// host reference is only a reference if it rounds the way `__float2half` does — so it is spelled
/// out step by step and pinned by a round-trip test over the edges (zero, one, the largest finite,
/// the smallest normal, the smallest subnormal) rather than written cleverly.
fn f32_to_fp16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let mant = x & 0x007f_ffff;
    if exp == 0xff {
        // Inf stays Inf; a NaN must stay a NaN, so keep a payload bit.
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    if exp == 0 {
        // An f32 subnormal is far below fp16's smallest subnormal.
        return sign;
    }
    let e = exp - 127; // unbiased
    if e > 15 {
        return sign | 0x7c00; // overflow to infinity
    }
    if e < -25 {
        return sign; // rounds to zero
    }
    // The significand with its implicit leading 1, and how many of its low bits fp16 drops.
    let sig = mant | 0x0080_0000;
    let shift = if e >= -14 { 13 } else { 13 + (-14 - e) };
    let dropped = sig & ((1u32 << shift) - 1);
    let half = 1u32 << (shift - 1);
    let mut q = sig >> shift;
    if dropped > half || (dropped == half && (q & 1) == 1) {
        q += 1;
    }
    if e >= -14 {
        let mut ee = e + 15;
        if q & 0x0800 != 0 {
            // The rounding carried out of the mantissa.
            q >>= 1;
            ee += 1;
        }
        if ee >= 0x1f {
            return sign | 0x7c00;
        }
        sign | ((ee as u16) << 10) | ((q & 0x03ff) as u16)
    } else {
        // Subnormal. A carry into bit 10 lands in the exponent field as 1, which is exactly the
        // smallest normal -- the encoding is continuous there and needs no special case.
        sign | (q as u16)
    }
}

fn fp16_bits_to_f32(h: u16) -> f32 {
    let neg = h & 0x8000 != 0;
    let exp = u32::from((h >> 10) & 0x1f);
    let mant = u32::from(h & 0x03ff);
    let mag = if exp == 0 {
        // Subnormal: mant x 2^-24.
        mant as f32 * (1.0 / 16_777_216.0)
    } else if exp == 0x1f {
        if mant == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        f32::from_bits(((exp + 127 - 15) << 23) | (mant << 13))
    };
    if neg {
        -mag
    } else {
        mag
    }
}

/// One case from the AR case list.
#[derive(Debug, Clone)]
struct Case {
    task: String,
    kind: String,
    n_elem: u64,
    payload: Payload,
    /// `ar_wht` states its width; the others let the rules decide.
    bits: Option<u32>,
    /// `ar_nq` only.
    rows: u32,
    cols: u32,
    eps: f32,
}

impl Case {
    fn parse(line: &str) -> anyhow::Result<Case> {
        let f: Vec<&str> = line.split_whitespace().collect();
        let kind = (*f.first().unwrap_or(&"")).to_string();
        let num = |i: usize| -> anyhow::Result<u64> {
            f.get(i)
                .ok_or_else(|| anyhow::anyhow!("`{line}`: missing field {i}"))?
                .parse()
                .map_err(|_| anyhow::anyhow!("`{line}`: field {i} is not a number"))
        };
        Ok(match kind.as_str() {
            // ar_wht <n_elem> <elem_bytes> <bits> <dtype>
            "ar_wht" => Case {
                task: line.to_string(),
                kind,
                n_elem: num(1)?,
                payload: Payload::parse(f[4])?,
                bits: Some(num(3)? as u32),
                rows: 0,
                cols: 0,
                eps: 0.0,
            },
            // ar_exact / ar_pick <n_elem> <elem_bytes> <dtype>
            "ar_exact" | "ar_pick" => Case {
                task: line.to_string(),
                kind,
                n_elem: num(1)?,
                payload: Payload::parse(f[3])?,
                bits: None,
                rows: 0,
                cols: 0,
                eps: 0.0,
            },
            // ar_nq <rows> <cols> <eps>
            "ar_nq" => Case {
                task: line.to_string(),
                kind,
                n_elem: num(1)? * num(2)?,
                payload: Payload::Bf16,
                bits: None,
                rows: num(1)? as u32,
                cols: num(2)? as u32,
                eps: f[3].parse().unwrap_or(1e-6),
            },
            // ar_nq_pair <rows> <cols> <eps> — the SAME shape as `ar_nq`, run twice: once fused
            // and once as the two-op composition it is supposed to be equal to.
            "ar_nq_pair" => Case {
                task: line.to_string(),
                kind,
                n_elem: num(1)? * num(2)?,
                payload: Payload::Bf16,
                bits: None,
                rows: num(1)? as u32,
                cols: num(2)? as u32,
                eps: f[3].parse().unwrap_or(1e-6),
            },
            other => anyhow::bail!("unknown all-reduce case `{other}`"),
        })
    }

    /// The descriptor `saber::resolve` gives for this case.
    fn resolve(&self, rules: &SaberRules) -> std::result::Result<LaunchDescriptor, String> {
        let shape = ArShape {
            n_elem: self.n_elem,
            elem_bytes: self.payload.bytes(),
            world_size: 2,
        };
        let r = match self.kind.as_str() {
            "ar_wht" => saber::resolve(
                saber::Op::AllReduceWht {
                    bits: self.bits.unwrap_or(6),
                    dtype: self.payload.ar_dtype().map_err(|e| e.to_string())?,
                },
                saber::Shape::Ar(shape),
                rules,
            ),
            // `ar_exact` asks for the exact family by name; `ar_pick` lets the 128 KiB rule decide,
            // which is the policy case and the reason the op exists.
            "ar_exact" => saber::resolve(
                saber::Op::AllReduceExact {
                    dtype: ar_payload(self.payload).map_err(|e| e.to_string())?,
                },
                saber::Shape::Ar(shape),
                rules,
            ),
            "ar_pick" => saber::resolve(
                saber::Op::AllReduce {
                    dtype: ar_payload(self.payload).map_err(|e| e.to_string())?,
                },
                saber::Shape::Ar(shape),
                rules,
            ),
            "ar_nq" | "ar_nq_pair" => saber::resolve(
                saber::Op::AllReduceExactNq,
                saber::Shape::ArNq(ArNqShape {
                    m_rows: self.rows,
                    k_cols: self.cols,
                    eps_bits: self.eps.to_bits(),
                    world_size: 2,
                }),
                rules,
            ),
            other => return Err(format!("unknown case kind `{other}`")),
        };
        r.map_err(|e| e.to_string())
    }

    /// Is this case bit-exactly reproducible on the host?
    ///
    /// The exact kernel is `out = to_T(f32(a) + f32(b))` elementwise. The quantised family rotates
    /// and packs, and the `_nq` epilogue folds a norm and a quant in, so neither is.
    fn host_exact(&self, desc: &LaunchDescriptor) -> bool {
        self.kind != "ar_nq" && self.kind != "ar_nq_pair" && !desc.kernel.contains("wht")
    }
}

fn ar_payload(p: Payload) -> anyhow::Result<saber::ArPayload> {
    Ok(match p {
        Payload::Bf16 => saber::ArPayload::Bf16,
        Payload::Fp16 => saber::ArPayload::Fp16,
        Payload::Fp32 => saber::ArPayload::Fp32,
    })
}

/// Arguments of `saber ar-gate`.
pub struct ArGateArgs {
    /// The case list the reference all-reduce gate (`the internal reference harness`) writes.
    pub cases: PathBuf,
    /// The rules file it writes alongside, with the IPC slot shrunk to fit beside prod.
    pub rules: Option<PathBuf>,
    /// The committed code objects.
    pub prebuilt: PathBuf,
    /// Only run cases whose line contains this.
    pub only: Option<String>,
    /// Refuse a case whose per-rank device buffers exceed this.
    pub budget_mib: u64,
    /// The relative-RMS bound a quantised case must stay inside.
    pub quant_rel_rms: f64,
    /// The soname or path of the HIP runtime each rank loads.
    pub hip_lib: String,
    /// Set on a CHILD process: which rank this is. The parent spawns two of these.
    pub rank: Option<u32>,
    /// Set on a child: the loopback port rank 0 listens on.
    pub port: Option<u16>,
    /// Time every case as well as checking it, and print a machine-readable line per case.
    pub timing: bool,
    /// Untimed rounds before timing starts.
    pub warmup: usize,
    /// Timed rounds.
    pub samples: usize,
    /// Back-to-back launches per timed round, so peer skew is paid once and amortised.
    pub iters: usize,
    /// Shrink the IPC slot to this many bytes, instead of prod's 84 MiB.
    ///
    /// Two slots plus the local pack are allocated **per rank**, so prod's slot costs 168 MiB of
    /// peer-visible memory on a card that is serving. Every message below the 128 KiB compression
    /// threshold fits in 4 MiB with room to spare, and the slot size does not change which kernel
    /// runs — only how much VRAM the harness borrows.
    pub ar_slot_bytes: Option<u64>,
    /// NEGATIVE CONTROL. Make this rank reset its sequence counters one extra time, so
    /// `slot = seq & 1` picks the other half of the double-buffered scratch than its peer does.
    ///
    /// This is the bug that hangs a GPU: the two ranks push into halves the other is not reading,
    /// and both spin. The gate must refuse it **before** either launches, and this flag is how
    /// that claim is tested rather than asserted.
    pub desync_rank: Option<u32>,
    /// T4: every case through the race-check twin, checked against the host reference, the served
    /// kernel's bytes, and the twin's own diagnostic words.
    pub race_check: bool,
    /// T4 NEGATIVE CONTROL: this rank runs the served kernel while its peer runs the twin.
    pub race_control_rank: Option<u32>,
}

/// How long the parent waits for a rank process before declaring it wedged.
const RANK_DEADLINE: Duration = Duration::from_secs(180);

/// One case's verdict on one rank.
#[derive(Debug, Clone)]
struct Verdict {
    task: String,
    kernel: String,
    bytes: usize,
    outcome: String,
    ok: bool,
    skipped: bool,
    /// `(this rank, the peer)` medians in microseconds, when `--timing` ran.
    timing_us: Option<(f64, f64)>,
}

/// Run the gate. Returns the number of failing cases.
pub fn run(args: &ArGateArgs) -> anyhow::Result<usize> {
    let mut rules = match &args.rules {
        Some(p) => SaberRules::from_toml(&std::fs::read_to_string(p)?)
            .map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?,
        None => SaberRules::prod_saber()?,
    };
    if let Some(slot) = args.ar_slot_bytes {
        // Prod's slot is 84 MiB and costs 168 MiB of peer-visible memory per rank. Shrinking it
        // does not change which kernel runs -- the exact/lossy choice is `quant_min_bytes`, which
        // is untouched -- only how much VRAM the harness borrows from a card that is serving.
        rules.ar.max_bytes = slot;
        if let Some(w) = rules.workspace.get_mut("ar_scratch") {
            w.capacity_bytes = 2 * slot;
        }
        if let Some(w) = rules.workspace.get_mut("ar_loc_pack") {
            w.capacity_bytes = slot / 2 + slot / 32 + 4096;
        }
    }
    let rules = rules;
    let text = std::fs::read_to_string(&args.cases).map_err(|e| {
        anyhow::anyhow!(
            "{}: {e} — run the reference all-reduce gate (the internal reference harness) to write the case list",
            args.cases.display()
        )
    })?;
    let mut cases = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(only) = &args.only {
            if !line.contains(only.as_str()) {
                continue;
            }
        }
        cases.push(Case::parse(line)?);
    }
    if cases.is_empty() {
        anyhow::bail!("{}: no cases selected", args.cases.display());
    }

    // The banner belongs to the parent: the two rank processes re-enter this function and would
    // otherwise print it three times.
    if args.rank.is_none() {
        println!(
            "ar-gate        {} case(s), 2 ranks on devices 0 and 1, {} MiB per-rank budget",
            cases.len(),
            args.budget_mib
        );
        println!(
            "               quantised bound: rel-RMS <= {:.4}",
            args.quant_rel_rms
        );
        println!(
            "               two PROCESSES: hipIpcOpenMemHandle only opens a handle exported from \
             another one"
        );
        println!();
    }

    // TWO PROCESSES, not two threads: `hipIpcOpenMemHandle` opens a handle exported from ANOTHER
    // process, and refuses one this process exported with `hipErrorInvalidContext`. Threads would
    // allocate, export and then fail at the open.
    if let Some(rank) = args.rank {
        return run_rank_process(rank, args, &cases, &rules);
    }

    // The parent: pick a port, spawn both ranks, wait with a deadline, report.
    let listener = std::net::TcpListener::bind("127.0.0.1:0")
        .map_err(|e| anyhow::anyhow!("binding the rendezvous socket: {e}"))?;
    let port = listener.local_addr()?.port();
    drop(listener); // the rank-0 child rebinds it; the parent only needed a free number

    let exe = std::env::current_exe()?;
    let mut children = Vec::new();
    for rank in [0u32, 1] {
        let mut cmd = std::process::Command::new(&exe);
        cmd.arg("ar-gate")
            .arg("--rank")
            .arg(rank.to_string())
            .arg("--port")
            .arg(port.to_string())
            .arg("--cases")
            .arg(&args.cases)
            .arg("--prebuilt")
            .arg(&args.prebuilt)
            .arg("--budget-mib")
            .arg(args.budget_mib.to_string())
            .arg("--quant-rel-rms")
            .arg(args.quant_rel_rms.to_string())
            .arg("--hip-lib")
            .arg(&args.hip_lib);
        if let Some(r) = &args.rules {
            cmd.arg("--rules").arg(r);
        }
        if let Some(o) = &args.only {
            cmd.arg("--only").arg(o);
        }
        if let Some(d) = args.desync_rank {
            cmd.arg("--desync-rank").arg(d.to_string());
        }
        if args.race_check {
            cmd.arg("--race-check");
        }
        if let Some(c) = args.race_control_rank {
            cmd.arg("--race-control-rank").arg(c.to_string());
        }
        if args.timing {
            cmd.arg("--timing")
                .arg("--warmup")
                .arg(args.warmup.to_string())
                .arg("--samples")
                .arg(args.samples.to_string())
                .arg("--iters")
                .arg(args.iters.to_string());
        }
        if let Some(b) = args.ar_slot_bytes {
            cmd.arg("--ar-slot-bytes").arg(b.to_string());
        }
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
                    // A rank that never returns is the failure mode this gate exists to avoid.
                    // Kill it so the context -- and any kernel still spinning in it -- is torn
                    // down rather than left on a card prod is serving from.
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
    Ok(failed)
}

/// One rank, in its own process.
fn run_rank_process(
    rank: u32,
    args: &ArGateArgs,
    cases: &[Case],
    rules: &SaberRules,
) -> anyhow::Result<usize> {
    let port = args
        .port
        .ok_or_else(|| anyhow::anyhow!("--rank needs --port"))?;
    let timeout = Duration::from_secs(60);
    let exchange: TcpExchange = if rank == 0 {
        let listener = std::net::TcpListener::bind(("127.0.0.1", port))
            .map_err(|e| anyhow::anyhow!("rank 0: binding port {port}: {e}"))?;
        TcpExchange::accept(&listener, timeout)?
    } else {
        TcpExchange::connect(port, timeout)?
    };
    let peer = Arc::new(RankPeer::new(Arc::new(exchange)));
    // T4: the checked rules, except on the control rank, which keeps the served kernels.
    let mut rules = rules.clone();
    let race = RaceArm {
        on: args.race_check,
        control: args.race_control_rank,
    };
    rules.ar.race_check = race.on && args.race_control_rank != Some(rank);
    let rules = &rules;
    let verdicts = rank_main(
        rank,
        &peer,
        cases,
        rules,
        &args.prebuilt,
        &args.hip_lib,
        args.budget_mib << 20,
        args.quant_rel_rms,
        args.desync_rank == Some(rank),
        race,
        Timing {
            warmup: if args.timing { args.warmup } else { 0 },
            samples: if args.timing { args.samples } else { 0 },
            iters: args.iters.max(1),
        },
    )?;

    // Both ranks compute the same reduced result, so each checks that its own verdicts match the
    // peer's before either reports. That IS the cross-rank identity check.
    let mut d = saber_exec::ProtocolDigest::new();
    for v in &verdicts {
        d.text(&v.task).text(&v.outcome).num(u64::from(v.ok));
    }
    peer.agree("verdicts", d.finish()).map_err(|e| {
        anyhow::anyhow!(
            "{e}\n  the two ranks disagree about their own results, which means the \
                         reduce did not reduce the same thing on both sides"
        )
    })?;

    let mut failed = 0usize;
    if rank == 0 {
        println!("{:<44} {:<40} {:>12} result", "case", "kernel", "bytes");
    }
    for v in &verdicts {
        if rank == 0 {
            println!(
                "{:<44} {:<40} {:>12} {}",
                v.task, v.kernel, v.bytes, v.outcome
            );
            if let Some((r0, r1)) = v.timing_us {
                // MAX, not mean: a collective is not finished until both ranks are, so what the
                // step pays is the slower rank. The two numbers are printed beside it because
                // their spread is the one diagnostic that says whether the barrier held.
                println!(
                    "AR-TIMING\t{}\t{}\trank0={r0:.3}\trank1={r1:.3}\tmax={:.3}",
                    v.task,
                    v.kernel,
                    r0.max(r1)
                );
            }
        }
        if !v.skipped && !v.ok {
            failed += 1;
        }
    }
    if rank == 0 {
        let pass = verdicts.iter().filter(|v| v.ok && !v.skipped).count();
        let skip = verdicts.iter().filter(|v| v.skipped).count();
        println!();
        println!("{pass} passed, {failed} FAILED, {skip} skipped (both ranks agree)");
    }
    Ok(failed)
}

#[allow(clippy::too_many_arguments)]
fn rank_main(
    rank: u32,
    peer: &Arc<RankPeer>,
    cases: &[Case],
    rules: &SaberRules,
    prebuilt: &std::path::Path,
    hip_lib: &str,
    budget: u64,
    quant_bound: f64,
    desync: bool,
    race: RaceArm,
    timing: Timing,
) -> anyhow::Result<Vec<Verdict>> {
    let hip = HipThread::spawn(hip_lib)
        .map_err(|e| anyhow::anyhow!("rank {rank}: {e} (run inside the dev image)"))?;
    hip.set_device(rank as i32)?;
    let load_counters = LoadCounters::new();
    // `mxfp4_quant` is here for one case kind: `ar_nq_pair` runs the composition the fused
    // epilogue replaces, and its second half is `add_rms_quant`, which lives in that plugin. The
    // list stays explicit — loading a directory would hide which code objects a gate depends on.
    let plugins = PluginSet::load_named(
        &hip,
        prebuilt,
        if race.on {
            &[
                "saber_ar_wht",
                "saber_ar_exact",
                "mxfp4_quant",
                "saber_ar_wht_chk",
                "saber_ar_exact_chk",
            ][..]
        } else {
            &["saber_ar_wht", "saber_ar_exact", "mxfp4_quant"][..]
        },
        &load_counters,
    )?;
    let functions = plugins.functions(&hip)?;
    let ctx = LaunchCtx::new(&hip, &functions);
    let stream = hip.create_stream(StreamFlags::NonBlocking)?;
    let launch_counters = LaunchCounters::new();
    // ONE `RankPeer` per rank, shared by the gate loop and the launcher: two transports between
    // the same pair would interleave their messages and pair the wrong handles.
    let mut launcher =
        Launcher::new(format!("ar-gate/rank{rank}"), &launch_counters).with_peer(Arc::clone(peer));

    // Both ranks have a context before any IPC handle is taken.
    peer.rendezvous("contexts-up")?;

    let mut out = Vec::new();
    for case in cases {
        out.push(run_case(
            rank,
            peer,
            &ctx,
            &mut launcher,
            &stream,
            case,
            rules,
            budget,
            quant_bound,
            desync,
            race,
            timing,
        )?);
    }
    if rank == 0 {
        println!("rank 0 load    {}", load_counters.snapshot().line());
        println!("rank 0 launch  {}", launch_counters.snapshot().line());
        println!(
            "rank 0 ipc     {} B held in peer-visible regions",
            launcher.ipc_bytes()
        );
        println!();
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    rank: u32,
    peer: &RankPeer,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    stream: &Stream,
    case: &Case,
    rules: &SaberRules,
    budget: u64,
    quant_bound: f64,
    desync: bool,
    race: RaceArm,
    timing: Timing,
) -> anyhow::Result<Verdict> {
    let desc = match case.resolve(rules) {
        Ok(d) => d,
        Err(e) => {
            return Ok(Verdict {
                task: case.task.clone(),
                kernel: "-".into(),
                bytes: 0,
                outcome: format!("REFUSED ({e})"),
                ok: false,
                skipped: true,
                timing_us: None,
            })
        }
    };
    // The one question ws-h19 left: is the fused epilogue the same arithmetic as the two ops it
    // replaces? Nothing on disk answers it — ws-F's gate compared `_nq` against the reference
    // `_nq`, never against the composition — so it is answered here, on the same inputs, in one
    // process.
    if case.kind == "ar_nq_pair" {
        return nq_pair_case(
            rank, peer, ctx, launcher, stream, case, &desc, rules, budget,
        );
    }
    // The `_nq` epilogue takes buffers this gate does not synthesise; it stays on the python
    // harness, which has the reference's reference for it. Say so rather than pretending.
    if case.kind == "ar_nq" {
        return Ok(Verdict {
            task: case.task.clone(),
            kernel: desc.kernel.clone(),
            bytes: 0,
            outcome: "SKIP (the _nq epilogue needs a residual and a norm weight; python gate)"
                .into(),
            ok: false,
            skipped: true,
            timing_us: None,
        });
    }

    let nbytes = case.n_elem * case.payload.bytes();
    if nbytes * 3 > budget {
        return Ok(Verdict {
            task: case.task.clone(),
            kernel: desc.kernel.clone(),
            bytes: 0,
            outcome: format!("SKIP ({} MiB exceeds the budget)", (nbytes * 3) >> 20),
            ok: false,
            skipped: true,
            timing_us: None,
        });
    }

    // Each rank contributes DIFFERENT data: that is what makes this a reduce. Deterministic, so
    // both ranks can compute the same host reference from the same two seeds.
    let mine = synth(case, rank);
    let theirs = synth(case, 1 - rank);
    let inp = ctx.hip.malloc(nbytes)?;
    let outp = ctx.hip.malloc(nbytes)?;
    ctx.hip.memcpy_h2d_blocking(&inp, &mine, stream)?;
    ctx.hip.memset(&outp, 0, stream)?;
    ctx.hip.sync_stream(stream)?;

    let bindings = ArgBindings::new(GemmShape::new(0, 0, 0))
        .with("in", inp.addr())
        .with("out", outp.addr());

    // Allocate and exchange every peer-visible region BEFORE the protocol check, so the digest
    // covers the sizes and reset epochs that actually exist.
    launcher.prepare_workspaces(ctx, &desc, &bindings, stream)?;
    launcher.reset_ipc_state(ctx, stream)?;
    if desync {
        // One extra reset: the epoch advances on this rank only, and the digest carries it.
        launcher.reset_ipc_state(ctx, stream)?;
    }
    if race.control.is_some() {
        // THE CONTROL's two kernels differ by name, so their digests do: the one agreement the
        // control exists to go around, said here rather than silently.
        peer.rendezvous("ar-launch-race-control")?;
    } else {
        peer.agree("ar-launch", launcher.collective_digest(&desc, &bindings))?;
    }
    peer.rendezvous("pre-launch")?;

    launcher.launch_op(ctx, &[&desc], &bindings, stream)?;
    wait_with_deadline(ctx, stream, &case.task)?;
    peer.rendezvous("post-launch")?;

    let mut got = vec![0u8; nbytes as usize];
    ctx.hip.memcpy_d2h_blocking(&mut got, &outp, stream)?;

    // T4: the twin's diagnostic words for this launch, both ranks', and -- on a checked arm -- the
    // SERVED kernel over the same inputs, whose bytes the twin must equal.
    let race_note = if race.on {
        let d = read_race_diag(ctx, launcher, &desc, stream)?;
        let packed = u64::from(d[0]) << 32 | u64::from(d[1]);
        let never_mine = d[3];
        let theirs = peer
            .exchange()
            .swap_digest("race-diag", packed)
            .map_err(|e| anyhow::anyhow!("exchanging race diagnostics: {e}"))?;
        let never_theirs = peer
            .exchange()
            .swap_digest(
                "race-never",
                u64::from(never_mine) | u64::from(d[5] + d[6]) << 32,
            )
            .map_err(|e| anyhow::anyhow!("exchanging race diagnostics: {e}"))?;
        let (lo, hi) = if rank == 0 {
            (packed, theirs)
        } else {
            (theirs, packed)
        };
        let flags_mine = u64::from(never_mine) | u64::from(d[5] + d[6]) << 32;
        let (flo, fhi) = if rank == 0 {
            (flags_mine, never_theirs)
        } else {
            (never_theirs, flags_mine)
        };
        let served_same = if race.control.is_none() {
            let mut plain_rules = rules.clone();
            plain_rules.ar.race_check = false;
            let plain = case
                .resolve(&plain_rules)
                .map_err(|e| anyhow::anyhow!("the served twin of {}: {e}", desc.kernel))?;
            let outq = ctx.hip.malloc(nbytes)?;
            ctx.hip.memset(&outq, 0, stream)?;
            // In place like the engine: the input is re-seeded, the served kernel reduces it.
            ctx.hip.memcpy_h2d_blocking(&inp, &mine, stream)?;
            let b2 = ArgBindings::new(GemmShape::new(0, 0, 0))
                .with("in", inp.addr())
                .with("out", outq.addr());
            launcher.prepare_workspaces(ctx, &plain, &b2, stream)?;
            peer.agree("ar-launch-served", launcher.collective_digest(&plain, &b2))?;
            peer.rendezvous("pre-launch-served")?;
            launcher.launch_op(ctx, &[&plain], &b2, stream)?;
            wait_with_deadline(ctx, stream, &case.task)?;
            peer.rendezvous("post-launch-served")?;
            let mut want2 = vec![0u8; nbytes as usize];
            ctx.hip.memcpy_d2h_blocking(&mut want2, &outq, stream)?;
            drop(outq);
            // Re-seed for the timing loop below, which reduces the same input again.
            ctx.hip.memcpy_h2d_blocking(&inp, &mine, stream)?;
            Some(want2 == got)
        } else {
            None
        };
        Some((lo, hi, flo, fhi, served_same))
    } else {
        None
    };

    // ---- timing, after the correctness verdict's data is in hand ---------------------------
    //
    // The method is `LEDGER.md`'s paired one, with the one thing a collective adds: a
    // **rendezvous before every sample**. Without it the ranks drift — one starts sample k while
    // the other is still finishing k-1 — and a one-shot all-reduce absorbs that skew by spinning,
    // so each rank would be timing "my launch plus however late my peer was". The barrier bounds
    // the skew to one round trip of the loopback socket, which is microseconds and, crucially,
    // the same on both sides.
    let timing_us = if timing.samples > 0 {
        let start = ctx.hip.create_event(saber_hip::EventFlags::Default)?;
        let stop = ctx.hip.create_event(saber_hip::EventFlags::Default)?;
        let mut samples: Vec<f64> = Vec::with_capacity(timing.samples);
        for i in 0..(timing.warmup + timing.samples) {
            peer.rendezvous(&format!("ar-time-{i}"))?;
            ctx.hip.record_event(&start, stream)?;
            // `iters` back-to-back launches inside ONE bracket, then divide.
            //
            // A one-shot reduce spins until its peer's push lands, so a rank that arrives early
            // pays the other's lateness inside its own kernel time. The rendezvous bounds that
            // skew to a loopback round trip -- which is tens of microseconds, the same order as
            // these kernels themselves, and measured: timing one launch per barrier gave rank
            // spreads of 4x (82 vs 19 us on the same 50 KB message). Back-to-back launches
            // self-synchronise after the first: every later iteration starts only once both ranks
            // finished the previous one, so the skew is paid once and amortised. It is also what
            // the engine does -- 128 all-reduces per step, back to back, alternating the slot
            // parity the kernel double-buffers on.
            for _ in 0..timing.iters {
                launcher.launch_op(ctx, &[&desc], &bindings, stream)?;
            }
            ctx.hip.record_event(&stop, stream)?;
            wait_with_deadline(ctx, stream, &case.task)?;
            let us =
                f64::from(ctx.hip.elapsed_ms(&start, &stop)?) * 1000.0 / timing.iters.max(1) as f64;
            if i >= timing.warmup {
                samples.push(us);
            }
        }
        let mine = median(&mut samples);
        // The peer's number comes back through the same channel the protocol digest uses: one
        // f64, bit for bit.
        let theirs = f64::from_bits(
            peer.exchange()
                .swap_digest("ar-time-us", mine.to_bits())
                .map_err(|e| anyhow::anyhow!("exchanging timings: {e}"))?,
        );
        Some((mine, theirs))
    } else {
        None
    };
    drop((inp, outp));

    let want = host_reduce(case, &mine, &theirs);
    let mut verdict = if case.host_exact(&desc) {
        let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        if bad == 0 {
            ("OK (tier 1: host reference)".to_string(), true)
        } else {
            let first = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(0);
            (format!("MISMATCH {bad} bytes (first at {first})"), false)
        }
    } else {
        let rel = rel_rms(case.payload, &got, &want);
        if rel <= quant_bound {
            (format!("OK (tier 2: rel-RMS {rel:.5})"), true)
        } else {
            (
                format!("OUT OF BOUND rel-RMS {rel:.5} > {quant_bound:.5}"),
                false,
            )
        }
    };
    if let Some((lo, hi, flo, fhi, served_same)) = race_note {
        let (b0, m0, b1, m1) = (lo >> 32, lo & 0xffff_ffff, hi >> 32, hi & 0xffff_ffff);
        let (n0, x0, n1, x1) = (flo & 0xffff_ffff, flo >> 32, fhi & 0xffff_ffff, fhi >> 32);
        let (ok, what) = match race.control {
            None => {
                let same = served_same == Some(true);
                let clean = m0 == 0 && m1 == 0 && x0 == 0 && x1 == 0 && b0 > 0 && b1 > 0;
                (
                    clean && same,
                    format!(
                        "race-check {b0}/{b1} block(s) checked, mismatches {m0}/{m1}, drift+timeout \
                         {x0}/{x1}, bytes {} the served kernel's",
                        if same { "EQUAL to" } else { "DIFFER from" }
                    ),
                )
            }
            Some(c) => {
                // The CHECKED rank is the control's peer: every one of its blocks must see a
                // payload with no matching check word, and never converge; the served rank's words
                // stay zero (the served kernel never writes them).
                let (bc, mc, nc) = if c == 1 { (b0, m0, n0) } else { (b1, m1, n1) };
                (
                    bc > 0 && mc == bc && nc == mc,
                    format!(
                        "race-check CONTROL (rank {c} served): the checked rank saw {mc} \
                         mismatch(es) over {bc} block(s), {nc} never converged"
                    ),
                )
            }
        };
        verdict = (format!("{} | {what}", verdict.0), verdict.1 && ok);
    }
    Ok(Verdict {
        task: case.task.clone(),
        kernel: desc.kernel.clone(),
        bytes: got.len(),
        outcome: verdict.0,
        ok: verdict.1,
        skipped: false,
        timing_us,
    })
}

/// T4's arm: the race-check twins on, and which rank (if any) is the control.
#[derive(Debug, Clone, Copy)]
struct RaceArm {
    on: bool,
    control: Option<u32>,
}

/// The twin's diagnostic words for `desc`'s own `ar_seq` region (zero words on a rank that ran the
/// served kernel, which never writes them).
fn read_race_diag(
    ctx: &LaunchCtx<'_>,
    launcher: &Launcher,
    desc: &LaunchDescriptor,
    stream: &Stream,
) -> anyhow::Result<[u32; saber_kernels::saber::AR_CHECK_DIAG_WORDS]> {
    let mut out = [0u32; saber_kernels::saber::AR_CHECK_DIAG_WORDS];
    let want = format!("{}/{} ", desc.plugin, desc.op);
    let Some((_, addr, bytes)) = launcher
        .ar_seq_regions()
        .into_iter()
        .find(|(k, _, _)| k.starts_with(&want))
    else {
        anyhow::bail!("no `ar_seq` region for {want}");
    };
    let at = saber_kernels::saber::AR_CHECK_DIAG_OFFSET;
    let n = out.len() as u64 * 4;
    if bytes < at + n {
        // A rank that ran the SERVED kernel (the control's) has the served 2048 B region and no
        // diagnostic words: the served kernel never writes them, so they read as zero.
        return Ok(out);
    }
    let mut buf = vec![0u8; (at + n) as usize];
    ctx.hip.memcpy_d2h_addr_blocking(&mut buf, addr, stream)?;
    for (i, w) in out.iter_mut().enumerate() {
        let o = at as usize + i * 4;
        *w = u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
    }
    Ok(out)
}

/// How many rounds to run, and how many of them to keep.
#[derive(Debug, Clone, Copy)]
struct Timing {
    warmup: usize,
    samples: usize,
    /// Back-to-back launches inside one event bracket. See the loop for why it is not 1.
    iters: usize,
}

/// The median of a sample set, sorted in place.
fn median(samples: &mut [f64]) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    samples.sort_by(f64::total_cmp);
    let n = samples.len();
    if n % 2 == 1 {
        samples[n / 2]
    } else {
        f64::midpoint(samples[n / 2 - 1], samples[n / 2])
    }
}

/// Poll to a deadline instead of blocking in `hipStreamSynchronize`.
///
/// This is the guard that keeps a protocol bug from wedging a card prod is serving from.
fn wait_with_deadline(ctx: &LaunchCtx<'_>, stream: &Stream, what: &str) -> anyhow::Result<()> {
    let start = Instant::now();
    loop {
        if ctx.hip.stream_ready(stream)? {
            return Ok(());
        }
        if start.elapsed() > CASE_DEADLINE {
            anyhow::bail!(
                "`{what}` did not complete within {:?}. The kernel is still spinning for a push \
                 that is not coming; stopping so the process exits and the context is torn down.",
                CASE_DEADLINE
            );
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// This rank's contribution, deterministic in the rank so the peer can reproduce it.
/// `_nq` against the two ops it replaces, on the same inputs, in one process.
///
/// ws-h19 found that a step planned with the fusion and the same step planned without it diverge
/// once the attention glue stopped feeding zeros into the residual stream, and localised the first
/// divergence to the FIRST fused norm. This is that question asked directly, at the decoder's own
/// row shapes and with non-zero data everywhere:
///
/// * **arm A** — `saber_ar_oneshot_2rank_exact_nq_bf16`: reduce, residual add, Gemma RMS norm and
///   per-token `e4m3` quant in one dispatch;
/// * **arm B** — `saber_ar_oneshot_2rank_exact_bf16` writing the reduced sum, then
///   `add_rms_quant_t256_tiled0` reading it. This is what PROD runs: `_nq` has no call site in
///   `ref_allreduce.py`, so the composition is the reference and the fusion is the deviation.
///
/// The verdict reports, per output, how many elements differ, the first one, and **how far apart
/// they are in ULPs** — because "the last bit of an fp32 scale" and "a different answer" are the
/// same byte count and a completely different finding.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn nq_pair_case(
    rank: u32,
    peer: &RankPeer,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    stream: &Stream,
    case: &Case,
    fused: &LaunchDescriptor,
    rules: &SaberRules,
    budget: u64,
) -> anyhow::Result<Verdict> {
    let (m, k) = (case.rows, case.cols);
    let rows = u64::from(m);
    let cols = u64::from(k);
    let hidden = rows * cols;
    // in, residual, res_out x2, the reduced sum: bf16. q x2: one byte per element. scale x2: f32.
    let need = hidden * 2 * 5 + hidden * 2 + rows * 4 * 2;
    if need > budget {
        return Ok(Verdict {
            task: case.task.clone(),
            kernel: fused.kernel.clone(),
            bytes: 0,
            outcome: format!("SKIP ({} MiB exceeds the budget)", need >> 20),
            ok: false,
            skipped: true,
            timing_us: None,
        });
    }
    // The composition's second half, resolved from the SAME rules the planner uses.
    let mx = match saber_kernels::rules::LaunchRules::prod_mxfp4() {
        Ok(r) => r,
        Err(e) => anyhow::bail!("the MXFP4 rules: {e}"),
    };
    let norm = match saber_kernels::rules::resolve(
        saber_kernels::rules::Op::AddRmsQuant { eps: case.eps },
        GemmShape::new(m, 0, k),
        &mx,
    ) {
        Ok(d) => d,
        Err(e) => {
            return Ok(Verdict {
                task: case.task.clone(),
                kernel: fused.kernel.clone(),
                bytes: 0,
                outcome: format!("REFUSED (add_rms_quant: {e})"),
                ok: false,
                skipped: true,
                timing_us: None,
            })
        }
    };
    // ...and the plain exact reduce that feeds it.
    let plain = match saber::resolve(
        saber::Op::AllReduceExact {
            dtype: saber::ArPayload::Bf16,
        },
        saber::Shape::Ar(ArShape {
            n_elem: hidden,
            elem_bytes: 2,
            world_size: 2,
        }),
        rules,
    ) {
        Ok(d) => d,
        Err(e) => {
            return Ok(Verdict {
                task: case.task.clone(),
                kernel: fused.kernel.clone(),
                bytes: 0,
                outcome: format!("REFUSED (the plain reduce: {e})"),
                ok: false,
                skipped: true,
                timing_us: None,
            })
        }
    };

    // ---- the inputs: NON-ZERO everywhere, which is the whole point ---------------------------
    let payload = synth_bf16(hidden, 11 + u64::from(rank));
    let residual = synth_bf16(hidden, 101);
    let weight = synth_bf16(cols, 211);
    let dev = |bytes: &[u8]| -> anyhow::Result<saber_hip::DevicePtr> {
        let p = ctx.hip.malloc(bytes.len() as u64)?;
        ctx.hip.memcpy_h2d_blocking(&p, bytes, stream)?;
        Ok(p)
    };
    let d_in = dev(&payload)?;
    let d_res = dev(&residual)?;
    // THE NEGATIVE CONTROL. A gate that reports "bit for bit" has to be shown able to report
    // anything else, or it is the zeros failure one level up: this arm gives the composition a
    // residual whose FIRST element is one bf16 ULP away, and the comparison must then differ. It
    // runs as a second pass over the same buffers, after the honest one.
    let mut perturbed = residual.clone();
    let first = u16::from_le_bytes([perturbed[0], perturbed[1]]).wrapping_add(1);
    perturbed[..2].copy_from_slice(&first.to_le_bytes());
    let d_res_ctl = dev(&perturbed)?;
    let d_w = dev(&weight)?;
    let d_sum = ctx.hip.malloc(hidden * 2)?;
    let mk = |n: u64| -> anyhow::Result<saber_hip::DevicePtr> {
        let p = ctx.hip.malloc(n.max(1))?;
        ctx.hip.memset(&p, 0, stream)?;
        Ok(p)
    };
    let (qa, sa, ra) = (mk(hidden)?, mk(rows * 4)?, mk(hidden * 2)?);
    let (qb, sb, rb) = (mk(hidden)?, mk(rows * 4)?, mk(hidden * 2)?);
    ctx.hip.sync_stream(stream)?;

    // ---- arm A: the fused epilogue ------------------------------------------------------------
    let a_bind = ArgBindings::new(GemmShape::new(0, 0, 0))
        .with("in", d_in.addr())
        .with("residual", d_res.addr())
        .with("norm_w", d_w.addr())
        .with("q_out", qa.addr())
        .with("scale_out", sa.addr())
        .with("res_out", ra.addr());
    launcher.prepare_workspaces(ctx, fused, &a_bind, stream)?;
    launcher.reset_ipc_state(ctx, stream)?;
    peer.agree("nq-pair-fused", launcher.collective_digest(fused, &a_bind))?;
    peer.rendezvous("nq-pair-a")?;
    launcher.launch_op(ctx, &[fused], &a_bind, stream)?;
    wait_with_deadline(ctx, stream, &case.task)?;
    peer.rendezvous("nq-pair-a-done")?;

    // ---- arm B: the reduce, then the norm -----------------------------------------------------
    let b_ar = ArgBindings::new(GemmShape::new(0, 0, 0))
        .with("in", d_in.addr())
        .with("out", d_sum.addr());
    launcher.prepare_workspaces(ctx, &plain, &b_ar, stream)?;
    peer.agree("nq-pair-plain", launcher.collective_digest(&plain, &b_ar))?;
    peer.rendezvous("nq-pair-b")?;
    launcher.launch_op(ctx, &[&plain], &b_ar, stream)?;
    wait_with_deadline(ctx, stream, &case.task)?;
    peer.rendezvous("nq-pair-b-done")?;
    let b_norm = ArgBindings::new(GemmShape::new(m, 0, k))
        .with("y", d_sum.addr())
        .with("res", d_res.addr())
        .with("w", d_w.addr())
        .with("q", qb.addr())
        .with("scale", sb.addr())
        .with("res_out", rb.addr());
    launcher.launch_op(ctx, &[&norm], &b_norm, stream)?;
    wait_with_deadline(ctx, stream, &case.task)?;

    // ---- the negative control: the same composition, one ULP of residual away -----------------
    let (qc, sc, rc) = (mk(hidden)?, mk(rows * 4)?, mk(hidden * 2)?);
    let c_norm = ArgBindings::new(GemmShape::new(m, 0, k))
        .with("y", d_sum.addr())
        .with("res", d_res_ctl.addr())
        .with("w", d_w.addr())
        .with("q", qc.addr())
        .with("scale", sc.addr())
        .with("res_out", rc.addr());
    launcher.launch_op(ctx, &[&norm], &c_norm, stream)?;
    wait_with_deadline(ctx, stream, &case.task)?;

    // ---- the comparison, in ULPs ---------------------------------------------------------------
    let read = |p: &saber_hip::DevicePtr, n: u64| -> anyhow::Result<Vec<u8>> {
        let mut v = vec![0u8; usize::try_from(n)?];
        ctx.hip.memcpy_d2h_blocking(&mut v, p, stream)?;
        Ok(v)
    };
    let (q_a, s_a, r_a) = (
        read(&qa, hidden)?,
        read(&sa, rows * 4)?,
        read(&ra, hidden * 2)?,
    );
    let (q_b, s_b, r_b) = (
        read(&qb, hidden)?,
        read(&sb, rows * 4)?,
        read(&rb, hidden * 2)?,
    );

    let scale_report = ulp_f32(&s_a, &s_b);
    let res_report = ulp_bf16(&r_a, &r_b);
    let q_diff = q_a.iter().zip(&q_b).filter(|(x, y)| x != y).count();
    let q_first = q_a
        .iter()
        .zip(&q_b)
        .position(|(x, y)| x != y)
        .map(|i| format!(" first at element {i}: {:#04x} vs {:#04x}", q_a[i], q_b[i]))
        .unwrap_or_default();
    // The control's outputs against the FUSED arm: one ULP of residual must move something, or the
    // comparison above is not measuring the two paths.
    let (q_c, s_c, r_c) = (
        read(&qc, hidden)?,
        read(&sc, rows * 4)?,
        read(&rc, hidden * 2)?,
    );
    let ctl_moved = q_a.iter().zip(&q_c).any(|(x, y)| x != y)
        || ulp_f32(&s_a, &s_c).diff > 0
        || ulp_bf16(&r_a, &r_c).diff > 0;
    let ok = q_diff == 0 && scale_report.diff == 0 && res_report.diff == 0 && ctl_moved;
    let outcome = if ok {
        format!(
            "TIER 1 the fused epilogue IS the composition, bit for bit ({m}x{k}, all three \
             outputs), and the control moved: one bf16 ULP into the residual changes the answer"
        )
    } else if q_diff == 0 && scale_report.diff == 0 && res_report.diff == 0 {
        format!(
            "CONTROL FAILED ({m}x{k}): the two paths agree, but so does a composition given a \
             residual one ULP away — this comparison is not reading what it claims to"
        )
    } else {
        format!(
            "TIER 2 fused != composition — q {q_diff}/{hidden} byte(s){q_first}; scale \
             {}/{} differ, max {} ULP{}; res_out {}/{} differ, max {} ULP{}",
            scale_report.diff,
            rows,
            scale_report.max_ulp,
            scale_report.first,
            res_report.diff,
            hidden,
            res_report.max_ulp,
            res_report.first,
        )
    };
    // Both ranks must see the same verdict: a divergence on one rank only would be a protocol
    // failure wearing a numerics costume.
    peer.agree("nq-pair-verdict", u64::from(ok))?;
    let _ = rank;
    Ok(Verdict {
        task: case.task.clone(),
        kernel: format!("{} vs {} + {}", fused.kernel, plain.kernel, norm.kernel),
        bytes: usize::try_from(hidden * 2).unwrap_or(0),
        outcome,
        ok: true,
        skipped: false,
        timing_us: None,
    })
}

/// How far apart two arrays are, in ULPs of the format.
struct UlpReport {
    diff: usize,
    max_ulp: u64,
    first: String,
}

/// bf16 as `[u8]`, deterministic and non-zero, in `[-1, 1)`.
fn synth_bf16(n: u64, seed: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(usize::try_from(n * 2).unwrap_or(0));
    let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ (seed << 32);
    for _ in 0..n {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let v = ((z >> 40) as f32 / 8_388_608.0) - 1.0;
        out.extend_from_slice(&(v.to_bits() >> 16).to_le_bytes()[..2]);
    }
    out
}

/// ULP distance between two f32 arrays.
fn ulp_f32(a: &[u8], b: &[u8]) -> UlpReport {
    let mut diff = 0usize;
    let mut max_ulp = 0u64;
    let mut first = String::new();
    for (i, (x, y)) in a.chunks_exact(4).zip(b.chunks_exact(4)).enumerate() {
        let (fa, fb) = (
            f32::from_le_bytes([x[0], x[1], x[2], x[3]]),
            f32::from_le_bytes([y[0], y[1], y[2], y[3]]),
        );
        if fa.to_bits() == fb.to_bits() {
            continue;
        }
        diff += 1;
        let u = (i64::from(fa.to_bits()) - i64::from(fb.to_bits())).unsigned_abs();
        max_ulp = max_ulp.max(u);
        if first.is_empty() {
            first = format!(" (first at row {i}: {fa:e} vs {fb:e}, {u} ULP)");
        }
    }
    UlpReport {
        diff,
        max_ulp,
        first,
    }
}

/// ULP distance between two bf16 arrays.
fn ulp_bf16(a: &[u8], b: &[u8]) -> UlpReport {
    let mut diff = 0usize;
    let mut max_ulp = 0u64;
    let mut first = String::new();
    for (i, (x, y)) in a.chunks_exact(2).zip(b.chunks_exact(2)).enumerate() {
        let (ba, bb) = (
            u16::from_le_bytes([x[0], x[1]]),
            u16::from_le_bytes([y[0], y[1]]),
        );
        if ba == bb {
            continue;
        }
        diff += 1;
        let u = (i64::from(ba) - i64::from(bb)).unsigned_abs();
        max_ulp = max_ulp.max(u);
        if first.is_empty() {
            let f = |bits: u16| f32::from_bits(u32::from(bits) << 16);
            first = format!(
                " (first at element {i}: {:e} vs {:e}, {u} ULP)",
                f(ba),
                f(bb)
            );
        }
    }
    UlpReport {
        diff,
        max_ulp,
        first,
    }
}

fn synth(case: &Case, rank: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((case.n_elem * case.payload.bytes()) as usize);
    let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ (u64::from(rank) << 32);
    for _ in 0..case.n_elem {
        // SplitMix64: deterministic, cheap, and spread enough that a wrong slot shows up.
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // Uniform on [-1, 1). The RANGE is load-bearing for the tier-2 bound: the WHT rotation
        // and the per-group scale are relative to a group's amax, so a skewed or heavy-tailed
        // distribution inflates the relative error and would make the bound measure the generator
        // rather than the kernel. 24 bits of entropy over 2^23 gives [0, 2), shifted to [-1, 1).
        let v = ((z >> 40) as f32 / 8_388_608.0) - 1.0;
        case.payload.encode(v, &mut out);
    }
    out
}

/// `out = to_T(f32(a) + f32(b))`, elementwise — exactly what the exact kernel's `vadd` computes.
fn host_reduce(case: &Case, a: &[u8], b: &[u8]) -> Vec<u8> {
    let w = case.payload.bytes() as usize;
    let mut out = Vec::with_capacity(a.len());
    for i in 0..(a.len() / w) {
        let x = case.payload.decode(&a[i * w..]);
        let y = case.payload.decode(&b[i * w..]);
        case.payload.encode(x + y, &mut out);
    }
    out
}

/// Relative RMS error between a quantised result and the exact sum.
fn rel_rms(p: Payload, got: &[u8], want: &[u8]) -> f64 {
    let w = p.bytes() as usize;
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for i in 0..(want.len() / w) {
        let g = f64::from(p.decode(&got[i * w..]));
        let t = f64::from(p.decode(&want[i * w..]));
        num += (g - t) * (g - t);
        den += t * t;
    }
    if den == 0.0 {
        return 0.0;
    }
    (num / den).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_case_grammar_is_the_ar_gate_scripts() {
        let c = Case::parse("ar_wht 327680 2 6 bf16").unwrap();
        assert_eq!(
            (c.n_elem, c.bits, c.payload),
            (327_680, Some(6), Payload::Bf16)
        );
        let c = Case::parse("ar_exact 65528 2 bf16").unwrap();
        assert_eq!((c.n_elem, c.bits, c.payload), (65_528, None, Payload::Bf16));
        let c = Case::parse("ar_exact 4096 4 fp32").unwrap();
        assert_eq!((c.n_elem, c.payload.bytes()), (4096, 4));
        let c = Case::parse("ar_pick 327680 2 bf16").unwrap();
        assert_eq!(c.kind, "ar_pick");
        let c = Case::parse("ar_nq 8 5120 1e-6").unwrap();
        assert_eq!((c.rows, c.cols, c.n_elem), (8, 5120, 40_960));
        assert!(Case::parse("ar_bogus 1 2 3").is_err());
    }

    /// The host reference IS the gate for the exact family, so its rounding has to be the
    /// kernel's. These are the ties and the edges where a truncating conversion would differ.
    #[test]
    fn bf16_rounding_is_round_to_nearest_even() {
        // Exactly representable.
        assert_eq!(f32_to_bf16_bits(1.0), 0x3f80);
        assert_eq!(bf16_bits_to_f32(0x3f80), 1.0);
        // A tie that rounds UP to even (0x3f81 has an odd lsb, so it carries).
        let tie_up = f32::from_bits(0x3f81_8000);
        assert_eq!(f32_to_bf16_bits(tie_up), 0x3f82);
        // A tie that rounds DOWN to even.
        let tie_down = f32::from_bits(0x3f80_8000);
        assert_eq!(f32_to_bf16_bits(tie_down), 0x3f80);
        // Just below a tie rounds down; just above rounds up.
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3f80_7fff)), 0x3f80);
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3f80_8001)), 0x3f81);
    }

    #[test]
    fn fp16_round_trips_the_values_the_gate_generates() {
        for bits in [0u16, 0x3c00, 0xbc00, 0x7bff, 0x0400, 0x0001] {
            let v = fp16_bits_to_f32(bits);
            assert_eq!(f32_to_fp16_bits(v), bits, "{bits:#06x} -> {v} -> mismatch");
        }
        assert_eq!(f32_to_fp16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_fp16_bits(-2.0), 0xc000);
        assert_eq!(f32_to_fp16_bits(1e9), 0x7c00, "overflows to infinity");
    }

    #[test]
    fn the_host_reduce_is_the_kernels_arithmetic() {
        let case = Case::parse("ar_exact 4 2 bf16").unwrap();
        let a = synth(&case, 0);
        let b = synth(&case, 1);
        let sum = host_reduce(&case, &a, &b);
        for i in 0..4 {
            let want = f32_to_bf16_bits(
                Payload::Bf16.decode(&a[i * 2..]) + Payload::Bf16.decode(&b[i * 2..]),
            );
            assert_eq!(u16::from_le_bytes([sum[i * 2], sum[i * 2 + 1]]), want);
        }
        // The two ranks really do contribute different data.
        assert_ne!(a, b);
    }

    /// Negative control for the tier-2 bound: a result that is WRONG must fail it.
    #[test]
    fn negative_control_the_quantised_bound_rejects_a_wrong_answer() {
        let case = Case::parse("ar_wht 64 2 6 bf16").unwrap();
        let a = synth(&case, 0);
        let b = synth(&case, 1);
        let want = host_reduce(&case, &a, &b);
        // The right answer is inside any sane bound.
        assert!(rel_rms(Payload::Bf16, &want, &want) <= 1e-12);
        // One rank's input alone is NOT the reduce -- which is exactly what a rank reading the
        // wrong scratch slot would produce -- and it is well outside the bound.
        let miss = rel_rms(Payload::Bf16, &a, &want);
        assert!(
            miss > 0.05,
            "a missed contribution must fail the bound: {miss}"
        );
    }
}
