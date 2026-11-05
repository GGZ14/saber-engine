//! FN59 microbench (card 1): the expert tier's end-of-step copy batch issued BLOCKING
//! (`run_batch`) vs NON-BLOCKING (`submit_batch`, the background lane).
//!
//! The batch is the mover's b1 shape: 45 pinned copies of one 2.6 MB slot (H2D promotions and D2H
//! victims alternating) on ONE copy stream, each followed by its layer's landed record (the
//! mover's `Landed` marks), plus two timing records (after the first copy, at the end). Per
//! iteration and arm (arms alternate, A B B A ...):
//!   * `call`   = host time the CALLER spends in the issue call;
//!   * `over`   = a call on ANOTHER stream posted right after (a record on the compute stream):
//!     what the rank thread's next HIP call waits;
//!   * `issued` = (async) the host time until the ticket reports every job issued;
//!   * `first` / `last` = device time from a record made just before the call to the first /
//!     last copy's completion (the copies must not land later).
//!
//! Usage: `fn59_submit_bench [iterations] [hip soname or path]` (default 300,
//! `libamdhip64.so.7`; in the dev image pass `/opt/rocm/lib/libamdhip64.so.7`) with ONE card.

use std::time::Instant;

use saber_hip::{EventFlags, HipThread, Job, MemcpyKind, StreamFlags, DEFAULT_HIP_SONAME};

const COPIES: usize = 45;
const STRIDE: u64 = 2_600_960; // 2.6 MB, 4 KiB-aligned
const LAYERS: usize = 48;

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let i = ((v.len() as f64 - 1.0) * p).round() as usize;
    v[i]
}

#[derive(Default)]
struct Arm {
    call: Vec<f64>,
    over: Vec<f64>,
    issued: Vec<f64>,
    first: Vec<f64>,
    last: Vec<f64>,
}

impl Arm {
    fn line(&mut self, name: &str) {
        let f = |v: &mut Vec<f64>| {
            if v.is_empty() {
                "-".to_string()
            } else {
                format!("p50 {:8.1} p99 {:8.1}", pct(v, 0.5), pct(v, 0.99))
            }
        };
        println!(
            "fn59 {name:5} n {:4} | call us {} | over us {} | issued us {} | first us {} | last us {}",
            self.call.len(),
            f(&mut self.call),
            f(&mut self.over),
            f(&mut self.issued),
            f(&mut self.first),
            f(&mut self.last),
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let iters: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(300);
    let soname = std::env::args()
        .nth(2)
        .unwrap_or_else(|| DEFAULT_HIP_SONAME.to_string());
    let hip = HipThread::spawn(&soname)?;
    hip.set_device(0)?;
    println!(
        "fn59 device {} ({:?}) copies {COPIES} x {STRIDE} B iters {iters}",
        hip.device_name(0)?,
        hip.info()
    );
    let dev = hip.malloc(2 * COPIES as u64 * STRIDE)?;
    let host = hip.host_malloc(2 * COPIES as u64 * STRIDE)?;
    let copy = hip.create_stream(StreamFlags::NonBlocking)?;
    let compute = hip.create_stream(StreamFlags::NonBlocking)?;
    let landed: Vec<_> = (0..LAYERS)
        .map(|_| hip.create_event(EventFlags::DisableTiming))
        .collect::<Result<_, _>>()?;
    let e0 = hip.create_event(EventFlags::Default)?;
    let e_first = hip.create_event(EventFlags::Default)?;
    let e_end = hip.create_event(EventFlags::Default)?;
    let e_other = hip.create_event(EventFlags::DisableTiming)?;

    let mut jobs: Vec<Job<'_>> = Vec::new();
    for k in 0..COPIES {
        let kind = if k % 2 == 0 {
            MemcpyKind::HostToDevice
        } else {
            MemcpyKind::DeviceToHost
        };
        jobs.push(Job::MemcpyPinned {
            dev: &dev,
            dev_offset: k as u64 * STRIDE,
            host: &host,
            host_offset: (COPIES + k) as u64 * STRIDE,
            bytes: STRIDE,
            kind,
            stream: &copy,
        });
        if k == 0 {
            jobs.push(Job::RecordEvent {
                event: &e_first,
                stream: &copy,
            });
        }
        jobs.push(Job::RecordEvent {
            event: &landed[k % LAYERS],
            stream: &copy,
        });
    }
    jobs.push(Job::RecordEvent {
        event: &e_end,
        stream: &copy,
    });
    println!("fn59 batch jobs {}", jobs.len());

    // one round trip, for scale
    let mut rt = Vec::new();
    for _ in 0..200 {
        let t = Instant::now();
        hip.record_event(&e_other, &compute)?;
        rt.push(t.elapsed().as_secs_f64() * 1e6);
    }
    println!(
        "fn59 roundtrip us p50 {:.1} p99 {:.1}",
        pct(&mut rt, 0.5),
        pct(&mut rt, 0.99)
    );

    let (mut blk, mut asy) = (Arm::default(), Arm::default());
    let warm = 10;
    for i in 0..(iters + warm) {
        // A B B A ...: interleaved, neither arm always first
        let order: [bool; 2] = if i % 2 == 0 {
            [true, false]
        } else {
            [false, true]
        };
        for is_async in order {
            hip.sync_stream(&copy)?;
            hip.sync_stream(&compute)?;
            hip.record_event(&e0, &copy)?;
            let t0 = Instant::now();
            let (call, over, issued);
            if is_async {
                let mut t = hip.submit_batch(&jobs)?;
                call = t0.elapsed();
                let t1 = Instant::now();
                hip.record_event(&e_other, &compute)?;
                over = t1.elapsed();
                // the host would go on here; for the measurement, poll until issued
                while !t.is_issued() {
                    std::hint::spin_loop();
                }
                issued = Some(t0.elapsed());
                t.wait()?;
            } else {
                hip.run_batch(&jobs)?;
                call = t0.elapsed();
                let t1 = Instant::now();
                hip.record_event(&e_other, &compute)?;
                over = t1.elapsed();
                issued = None;
            }
            hip.sync_stream(&copy)?;
            let first = f64::from(hip.elapsed_ms(&e0, &e_first)?) * 1e3;
            let last = f64::from(hip.elapsed_ms(&e0, &e_end)?) * 1e3;
            if i >= warm {
                let a = if is_async { &mut asy } else { &mut blk };
                a.call.push(call.as_secs_f64() * 1e6);
                a.over.push(over.as_secs_f64() * 1e6);
                if let Some(x) = issued {
                    a.issued.push(x.as_secs_f64() * 1e6);
                }
                a.first.push(first);
                a.last.push(last);
            }
        }
    }
    blk.line("block");
    asy.line("async");
    drop(jobs);
    Ok(())
}
