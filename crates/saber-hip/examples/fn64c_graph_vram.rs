//! FN64c probe: the VRAM a captured + instantiated graph exec holds on the dev runtime (ROCm 10.1).
//!
//! For each config, captures `M` graphs of `n` launches of one probe kernel, instantiates them,
//! launches each once, and reads `hipMemGetInfo` free before / after instantiate / after launch /
//! after destroy. Kernels (`fn64c_graph_vram.hip`): `k_nop` (8 B kernarg), `k_big` (512 B
//! kernarg), `k_scratch` (4 KiB private segment). Runtime knobs are the container's env (one
//! process per env arm). Usage: `fn64c_graph_vram <hsaco> [M] [soname] [kind:n,...]`, ONE card.

use saber_hip::{HipThread, StreamCaptureMode, StreamFlags, DEFAULT_HIP_SONAME};

const MIB: f64 = 1024.0 * 1024.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let hsaco = args
        .next()
        .ok_or("usage: fn64c_graph_vram <hsaco> [M] [soname]")?;
    let m: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(64);
    let soname = args
        .next()
        .unwrap_or_else(|| DEFAULT_HIP_SONAME.to_string());
    let hip = HipThread::spawn(&soname)?;
    hip.set_device(0)?;
    let image = std::fs::read(&hsaco)?;
    let module = hip.load_module(&image)?;
    let k_nop = hip.get_function(&module, "k_nop")?;
    let k_big = hip.get_function(&module, "k_big")?;
    let k_scr = hip.get_function(&module, "k_scratch")?;
    let buf = hip.malloc(1 << 20)?;
    let a = buf.addr().as_u64();
    let stream = hip.create_stream(StreamFlags::NonBlocking)?;
    let free = |h: &HipThread| -> f64 { h.mem_info().map(|(f, _)| f as f64).unwrap_or(0.0) };
    // warm: one eager launch of each kernel (scratch for the eager queue comes up here)
    let mut ka8 = a.to_le_bytes().to_vec();
    let mut ka_big = vec![0u8; 512];
    ka_big[..8].copy_from_slice(&a.to_le_bytes());
    let mut ka_scr = a.to_le_bytes().to_vec();
    ka_scr.extend_from_slice(&3i32.to_le_bytes());
    ka_scr.extend_from_slice(&[0u8; 4]);
    hip.launch(&k_nop, [1, 1, 1], [64, 1, 1], 0, &stream, &mut ka8)?;
    hip.launch(&k_big, [1, 1, 1], [64, 1, 1], 0, &stream, &mut ka_big)?;
    hip.launch(&k_scr, [64, 1, 1], [256, 1, 1], 0, &stream, &mut ka_scr)?;
    hip.sync_stream(&stream)?;
    println!(
        "fn64c device {} M {m} free {:.1} MiB",
        hip.device_name(0)?,
        free(&hip) / MIB
    );

    // optional 4th argument: the configs as `kind:n,kind:n` (kinds nop | big | scratch | empty)
    let custom: Option<Vec<(String, usize)>> = std::env::args().nth(4).map(|a| {
        a.split(',')
            .filter_map(|c| {
                let (k, n) = c.split_once(':')?;
                Some((k.to_string(), n.parse().ok()?))
            })
            .collect()
    });
    let default_configs: &[(&str, usize)] = &[
        ("empty", 0),
        ("nop", 1),
        ("nop", 4),
        ("nop", 16),
        ("nop", 64),
        ("big", 16),
        ("big", 64),
        ("scratch", 1),
        ("scratch", 16),
        ("nop", 1),
        ("big", 1024),
        ("big", 4096),
        ("big", 8192),
    ];
    let configs: Vec<(String, usize)> = custom.unwrap_or_else(|| {
        default_configs
            .iter()
            .map(|&(k, n)| (k.to_string(), n))
            .collect()
    });
    for (kind, n) in configs {
        let kind = kind.as_str();
        // the large graphs at fewer copies (their capture is the slow part)
        let m = if n >= 1024 { m.min(8) } else { m };
        let f0 = free(&hip);
        let mut graphs = Vec::with_capacity(m);
        for _ in 0..m {
            hip.begin_capture(&stream, StreamCaptureMode::ThreadLocal)?;
            for _ in 0..n {
                match kind {
                    "nop" => hip.launch(&k_nop, [1, 1, 1], [64, 1, 1], 0, &stream, &mut ka8)?,
                    "big" => hip.launch(&k_big, [1, 1, 1], [64, 1, 1], 0, &stream, &mut ka_big)?,
                    _ => hip.launch(&k_scr, [64, 1, 1], [256, 1, 1], 0, &stream, &mut ka_scr)?,
                }
            }
            graphs.push(hip.end_capture(&stream)?);
        }
        let f1 = free(&hip);
        let mut execs = Vec::with_capacity(m);
        for g in &graphs {
            execs.push(hip.instantiate(g)?);
        }
        let f2 = free(&hip);
        for e in &execs {
            hip.graph_launch(e, &stream)?;
        }
        hip.sync_stream(&stream)?;
        let f3 = free(&hip);
        drop(execs);
        drop(graphs);
        hip.sync_stream(&stream)?;
        let f4 = free(&hip);
        let per = |x: f64, y: f64| (x - y) / MIB / m as f64;
        println!(
            "fn64c {kind:8} nodes {n:3} | MiB/graph capture {:7.3} instantiate {:7.3} launch {:7.3} | \
             total {:7.3} | after destroy {:+8.1} MiB",
            per(f0, f1),
            per(f1, f2),
            per(f2, f3),
            per(f0, f3),
            (f4 - f0) / MIB
        );
    }
    // issue cost: one 64-node graph launched back to back (host us a launch, device us a graph)
    let e0 = hip.create_event(saber_hip::EventFlags::Default)?;
    let e1 = hip.create_event(saber_hip::EventFlags::Default)?;
    for nodes in [16usize, 64] {
        hip.begin_capture(&stream, StreamCaptureMode::ThreadLocal)?;
        for _ in 0..nodes {
            hip.launch(&k_nop, [1, 1, 1], [64, 1, 1], 0, &stream, &mut ka8)?;
        }
        let g = hip.end_capture(&stream)?;
        let e = hip.instantiate(&g)?;
        for _ in 0..20 {
            hip.graph_launch(&e, &stream)?;
        }
        hip.sync_stream(&stream)?;
        let reps = 400;
        hip.record_event(&e0, &stream)?;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            hip.graph_launch(&e, &stream)?;
        }
        let host = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
        hip.record_event(&e1, &stream)?;
        hip.sync_stream(&stream)?;
        let dev = f64::from(hip.elapsed_ms(&e0, &e1)?) * 1e3 / reps as f64;
        println!(
            "fn64c issue nodes {nodes:3} | host us/launch {host:8.2} | device us/graph {dev:8.2}"
        );
    }
    Ok(())
}
