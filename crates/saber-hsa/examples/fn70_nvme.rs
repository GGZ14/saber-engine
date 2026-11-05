//! FN70 card check of `saber_hsa::NvmeVram` (`nvme_read_to_vram`), ONE card, no HIP:
//!
//!   fn70_nvme FILE BDFID_HEX [POOLFILE]   (FN71b: the pool route first when the boot has one)
//!
//! Opens the reader (P2P probe into a 256 MiB host-visible VRAM window), then on whichever route
//! it picked (and on the bounce when the pick was P2P, as the control):
//!   rec   400 records of 2,621,440 B at pseudo-random 4 KiB-aligned offsets (the first 8 GiB),
//!         one call + a host wait each (per-record latency p50/p99, GB/s), every 4th followed by
//!         a completion word; each record read back (hsa_memory_copy) == a buffered host pread
//!   strm  a 1 GiB stream, 128 x 8 MiB calls back to back (a host wait only at the end), GB/s;
//!         the destination's final 256 MiB read back == the file's [768 MiB, 1 GiB)
//!   word  the completion word's value read back == the last `done` value written
//! Prints `fn70nvme` lines; exit 1 on any byte difference.

use std::fs::{File, OpenOptions};
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::time::Instant;

use saber_hsa::nvme::pool_sysfs_dir;
use saber_hsa::{Hsa, NvmeOptions, NvmeRoute, NvmeVram, PoolAsk, DEFAULT_HSA_SONAME};

const O_DIRECT: i32 = 0o40000; // x86_64 Linux
const REC: u64 = 2_621_440;
const MIB: u64 = 1 << 20;

fn pct(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(((q * v.len() as f64) as usize).min(v.len().saturating_sub(1)))
        .copied()
        .unwrap_or(0.0)
}

fn readback(hsa: &Hsa, dev: u64, n: u64) -> Vec<u8> {
    let mut v = vec![0u8; n as usize];
    // SAFETY: `dev .. dev + n` lies in a VRAM allocation this program made; `v` holds n bytes.
    unsafe { hsa.copy(v.as_mut_ptr(), dev as *const u8, n as usize) }.expect("readback");
    v
}

#[allow(clippy::too_many_arguments)]
fn run(
    tag: &str,
    rd: &mut NvmeVram,
    hsa: &Hsa,
    fd: &File,
    fb: &File,
    dst: u64,
    word: u64,
    span: u64,
) -> bool {
    let mut ok = true;
    // ---- rec
    let slots = (256 * MIB) / REC;
    let mut lat = Vec::new();
    let mut bad = 0u32;
    let mut last_done = 0u32;
    let n = 400u64;
    for k in 0..n {
        let off = ((k.wrapping_mul(0x9E37_79B9_7F4A_7C15)) % (span / REC)) * REC;
        let d = dst + (k % slots) * REC;
        let done = (k % 4 == 3).then(|| {
            last_done = k as u32 + 1;
            (word, last_done)
        });
        let a = Instant::now();
        let t = rd
            .nvme_read_to_vram(fd, off, REC, d, done)
            .expect("nvme_read_to_vram");
        rd.wait(t).expect("wait");
        lat.push(a.elapsed().as_secs_f64() * 1e6);
        let got = readback(hsa, d, REC);
        let mut want = vec![0u8; REC as usize];
        fb.read_exact_at(&mut want, off).expect("truth");
        if got != want {
            bad += 1;
        }
    }
    // the timed part excludes the readbacks: sum of per-call latencies
    let busy: f64 = lat.iter().sum::<f64>() / 1e6;
    println!(
        "fn70nvme {tag} rec: {n} x {REC} B, QD1 call+wait p50 {:.0} us p99 {:.0} us, {:.2} GB/s, \
         bytes {} ({bad} differing)",
        pct(&mut lat, 0.5),
        pct(&mut lat, 0.99),
        (n * REC) as f64 / busy / 1e9,
        if bad == 0 { "EQUAL" } else { "DIFFER" }
    );
    ok &= bad == 0;
    // ---- word
    let w = readback(hsa, word, 4);
    let wv = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
    println!(
        "fn70nvme {tag} word: {wv} (want {last_done}) {}",
        if wv == last_done { "OK" } else { "WRONG" }
    );
    ok &= wv == last_done;
    // ---- strm
    let piece = 8 * MIB;
    let a = Instant::now();
    let mut t = 0;
    for i in 0..128u64 {
        t = rd
            .nvme_read_to_vram(fd, i * piece, piece, dst + (i % 32) * piece, None)
            .expect("stream");
    }
    rd.wait(t).expect("stream wait");
    let dt = a.elapsed().as_secs_f64();
    let got = readback(hsa, dst, 256 * MIB);
    let mut want = vec![0u8; (256 * MIB) as usize];
    fb.read_exact_at(&mut want, 768 * MIB).expect("truth");
    let eq = got == want;
    println!(
        "fn70nvme {tag} strm: 1 GiB in 128 x 8 MiB, {:.3} s = {:.2} GB/s, final 256 MiB {}",
        dt,
        (1u64 << 30) as f64 / dt / 1e9,
        if eq { "EQUAL" } else { "DIFFER" }
    );
    ok &= eq;
    println!(
        "fn70nvme {tag} counts: {:?} lane {:?}",
        rd.counts,
        rd.lane_counts()
    );
    ok
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: fn70_nvme FILE BDFID_HEX");
        std::process::exit(2);
    }
    let bdf = u32::from_str_radix(args[2].trim_start_matches("0x"), 16).expect("bdfid hex");
    let fd = OpenOptions::new()
        .read(true)
        .custom_flags(O_DIRECT)
        .open(&args[1])
        .expect("open O_DIRECT");
    let fb = File::open(&args[1]).expect("open");
    let span = fb.metadata().expect("stat").len().min(8 << 30);
    // FN71b: an optional POOLFILE (the card's p2pmem/allocate, bind-mounted rw into a container
    // where sysfs is read-only) names the pool route's drive side; without it, sysfs in place.
    let pool = match args.get(3) {
        Some(a) => PoolAsk::At {
            sysfs_dir: pool_sysfs_dir(bdf),
            allocate: a.into(),
        },
        None => PoolAsk::Auto,
    };
    let opts = NvmeOptions {
        slots: 8,
        slot_bytes: (8 * MIB) as usize,
        window_bytes: (256 * MIB) as usize,
        pool,
    };
    let mut rd =
        NvmeVram::open_with(DEFAULT_HSA_SONAME, bdf, &opts, (&fd, 0)).expect("open reader");
    println!(
        "fn70nvme open: route {:?}, pool {:?}, window {:?}, refusal: {}",
        rd.route(),
        rd.pool(),
        rd.window(),
        rd.p2p_refusal().unwrap_or("none")
    );
    let hsa = Hsa::open(DEFAULT_HSA_SONAME).expect("hsa");
    let gpu = hsa
        .agents()
        .expect("agents")
        .into_iter()
        .find(|a| a.gpu && a.bdfid == bdf)
        .expect("gpu agent")
        .agent;
    let dst = hsa.alloc_vram(gpu, (256 * MIB) as usize).expect("vram") as u64;
    let word = hsa.alloc_vram(gpu, 4096).expect("word") as u64;
    rd.register_device(dst, 256 * MIB);
    rd.register_device(word, 4096);
    let mut ok = true;
    if rd.route() == NvmeRoute::Pool {
        let m = rd.pool().expect("pool");
        ok &= run("pool", &mut rd, &hsa, &fd, &fb, m.gpu, word, span);
        rd.force_bounce();
    } else if rd.route() == NvmeRoute::P2p {
        let (w, _) = rd.window().expect("window");
        ok &= run("p2p", &mut rd, &hsa, &fd, &fb, w, word, span);
        rd.force_bounce();
    }
    ok &= run("bounce", &mut rd, &hsa, &fd, &fb, dst, word, span);
    drop(rd);
    // SAFETY: allocated above; the reader (and its lane) drained and dropped.
    unsafe {
        let _ = hsa.free(dst as *mut u8);
        let _ = hsa.free(word as *mut u8);
    }
    println!("fn70nvme result: {}", if ok { "PASS" } else { "FAIL" });
    if !ok {
        std::process::exit(1);
    }
}
