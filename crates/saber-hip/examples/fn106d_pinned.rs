//! FN106d: the huge pinned allocator's host test (`saber_hip::pinned`, `HipThread::host_malloc`
//! under `PinnedPages::Huge`).
//!
//!   fn106d_pinned host  [GiB ...]            THP only (no HIP): map + MADV_HUGEPAGE + populate per
//!                                            size, the THP-backed share (smaps), the time, unmap;
//!                                            RSS / AnonHugePages back to the baseline
//!   fn106d_pinned hip   SONAME LOOPS [GiB ...]   in the dev image, ONE card: per loop and size the
//!                                            huge form (registered: device ptr == host VA, an H2D
//!                                            + D2H round trip of the head and tail 2 MiB equal),
//!                                            then the shmem form, both dropped; the census line,
//!                                            RSS and /proc/meminfo after every loop (no leak =
//!                                            back to the baseline, freed == allocated, leaked 0)
//!
//! Every line is `fn106d ...`.

use std::time::Instant;

use saber_hip::pinned::{self, PinnedPages};
use saber_hip::{HipThread, StreamFlags};

fn rss_mib() -> f64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    s.lines()
        .find_map(|l| l.strip_prefix("VmRSS:"))
        .and_then(|v| v.split_whitespace().next()?.parse::<f64>().ok())
        .unwrap_or(0.0)
        / 1024.0
}

fn mem() -> String {
    let m = pinned::meminfo_kib(&["MemAvailable", "AnonHugePages", "ShmemHugePages", "Shmem"]);
    format!(
        "RSS {:.0} MiB MemAvailable {:.2} GiB AnonHugePages {} MiB ShmemHugePages {} MiB Shmem {} MiB",
        rss_mib(),
        m[0].unwrap_or(0) as f64 / 1048576.0,
        m[1].unwrap_or(0) / 1024,
        m[2].unwrap_or(0) / 1024,
        m[3].unwrap_or(0) / 1024
    )
}

fn sizes(args: &[String]) -> Vec<f64> {
    let v: Vec<f64> = args.iter().filter_map(|a| a.parse().ok()).collect();
    if v.is_empty() {
        vec![1.0, 4.0, 12.0]
    } else {
        v
    }
}

fn host(args: &[String]) {
    println!("fn106d host baseline: {}", mem());
    for g in sizes(args) {
        let bytes = (g * (1u64 << 30) as f64) as u64;
        let t = Instant::now();
        match pinned::map_huge(bytes) {
            Ok(m) => {
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let b = pinned::smaps_huge_bytes(&[(m.base, m.map_len)]).unwrap_or(0);
                println!(
                    "fn106d host {g} GiB: mapped+populated in {ms:.0} ms ({:.1} GB/s), THP-backed {:.3} GiB = {:.1} % | {}",
                    bytes as f64 / ms / 1e6,
                    b as f64 / (1u64 << 30) as f64,
                    100.0 * b as f64 / m.map_len as f64,
                    mem()
                );
                pinned::unmap(&m);
                println!("fn106d host {g} GiB: unmapped | {}", mem());
            }
            Err(e) => println!("fn106d host {g} GiB: FALLBACK by name: {e}"),
        }
    }
}

fn hip(args: &[String]) -> saber_hip::Result<()> {
    let soname = args.first().map_or("libamdhip64.so.7", String::as_str);
    let loops: u32 = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(3);
    let gibs = sizes(&args[2.min(args.len())..]);
    let hip = HipThread::spawn(soname)?;
    let st = hip.create_stream(StreamFlags::NonBlocking)?;
    let dev = hip.malloc(4 << 20)?;
    println!("fn106d hip baseline (runtime up): {}", mem());
    for l in 0..loops {
        for form in [PinnedPages::Huge, PinnedPages::Shmem] {
            hip.set_pinned_pages(form);
            for &g in &gibs {
                let bytes = (g * (1u64 << 30) as f64) as u64;
                let t = Instant::now();
                let mut b = hip.host_malloc(bytes)?;
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let reg = b.is_registered();
                let dp = hip.host_device_addr(&b)?;
                let ident = dp == b.as_slice().as_ptr() as u64;
                let huge = if reg {
                    pinned::smaps_huge_bytes(&[(b.as_slice().as_ptr() as usize, b.len())])
                        .unwrap_or(0)
                } else {
                    0
                };
                // head + tail 2 MiB through the card and back
                let n = b.len();
                for (i, x) in b.as_mut_slice()[..2 << 20].iter_mut().enumerate() {
                    *x = (i * 7 + 3) as u8;
                }
                for (i, x) in b.as_mut_slice()[n - (2 << 20)..].iter_mut().enumerate() {
                    *x = (i * 13 + 1) as u8;
                }
                let head: Vec<u8> = b.as_slice()[..2 << 20].to_vec();
                let tail: Vec<u8> = b.as_slice()[n - (2 << 20)..].to_vec();
                let mut back = hip.host_malloc(4 << 20)?;
                hip.memcpy_h2d_at_blocking(&dev, 0, &b.as_slice()[..2 << 20], &st)?;
                hip.memcpy_h2d_at_blocking(&dev, 2 << 20, &b.as_slice()[n - (2 << 20)..], &st)?;
                hip.memcpy_d2h(&mut back, &dev, &st)?;
                hip.sync_stream(&st)?;
                let ok = back.as_slice()[..2 << 20] == head[..]
                    && back.as_slice()[2 << 20..] == tail[..];
                // the card reads the registered pages over UVA too: a pinned H2D from the buffer
                hip.memcpy_h2d(&dev, &back, &st)?;
                hip.sync_stream(&st)?;
                println!(
                    "fn106d hip loop {l} {} {g} GiB: alloc {ms:.0} ms registered {reg} identity {ident} \
                     THP {:.3} GiB round trip {} | {}",
                    form.as_str(),
                    huge as f64 / (1u64 << 30) as f64,
                    if ok { "EQUAL" } else { "DIFFERS" },
                    mem()
                );
                drop(back);
                drop(b);
                hip.sync_stream(&st)?; // the destructors run on the HIP thread in order
            }
        }
        hip.device_synchronize()?;
        println!(
            "fn106d hip after loop {l}: {} | census {}",
            mem(),
            pinned::stats().line(PinnedPages::Huge)
        );
    }
    Ok(())
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    match a.first().map(String::as_str) {
        Some("host") => host(&a[1..]),
        Some("hip") => {
            if let Err(e) = hip(&a[1..]) {
                println!("fn106d hip FAILED: {e}");
                std::process::exit(1);
            }
        }
        _ => eprintln!("fn106d_pinned host [GiB...] | hip SONAME LOOPS [GiB...]"),
    }
}
