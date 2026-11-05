//! `saber d2d-gate` — the device-to-device copy binding, on a card (V4).
//!
//! `HipThread::memcpy_d2d_blocking` is what a sequence-group fork is made of: a branch of an
//! `n > 1` request gets a **copy** of the leader's GDN state slot set and of the leader's one
//! partial KV page, and both are ranges of allocations the engine already holds. Until V4 the
//! engine had no D2D binding at all (ws-h29 said so), so the fork could not be built.
//!
//! A binding that moves bytes needs a gate that reads them back, and this is it. Three arms, each
//! answering something the other two cannot:
//!
//! 1. **A known pattern, copied and read back BYTE-EXACT.** A slot-sized region of one allocation
//!    is filled with a per-byte pattern that is a function of its own offset — so a copy that
//!    lands at the wrong offset, copies the wrong length, or copies the same page twice is a
//!    mismatch rather than a plausible-looking buffer. The destination is *poisoned* first, so
//!    "the copy did nothing" cannot pass.
//! 2. **Two offsets of ONE allocation**, which is the case the engine actually runs: the KV pool
//!    is a single buffer and a page copy is two offsets into it.
//! 3. **The refusals**, by name and with nothing posted: a destination that overruns, a source
//!    that overruns, and two ranges of one allocation that OVERLAP (`hipMemcpyAsync` does not
//!    define that one, and a partly-copied page is a page nothing would report).
//!
//! The refusals are checked on the calling thread, so arm 3 is the same check the host test
//! `a_batch_is_bounds_checked_before_anything_is_posted` runs — it is repeated here because a gate
//! whose bounds check lives in another crate's unit test is a gate that cannot say the binary on
//! this card refuses.

use saber_hip::{HipThread, StreamFlags};

/// Arguments, as the binary declares them.
#[derive(Debug, Clone)]
pub struct D2dArgs {
    /// HIP device ordinal within the visible set.
    pub device: i32,
    /// Bytes per region. The default is a GDN state slot's own order of magnitude.
    pub bytes: u64,
    /// Regions in the buffer. Three is the smallest number that can show a copy landing on its
    /// neighbour.
    pub regions: u64,
}

/// Byte `i` of region `r`: a function of both, so a copy at the wrong offset cannot match.
fn pattern(r: u64, i: u64) -> u8 {
    ((r.wrapping_mul(0x9E37).wrapping_add(i.wrapping_mul(31)) ^ (i >> 3)) & 0xFF) as u8
}

/// Run the gate. Returns the number of failures.
pub fn run(hip: &HipThread, args: &D2dArgs) -> anyhow::Result<usize> {
    hip.set_device(args.device)?;
    crate::gpu_idle::require_default(hip, args.device, 0)?;
    let stream = hip.create_stream(StreamFlags::NonBlocking)?;
    let bytes = args.bytes;
    let regions = args.regions.max(3);
    let total = bytes * regions;
    println!(
        "d2d-gate  device {} — {regions} regions of {bytes} bytes ({} MiB total)",
        args.device,
        total / (1024 * 1024)
    );

    // Two allocations, because arm 1 crosses allocations and arm 2 stays inside one.
    let src = hip.malloc(total)?;
    let dst = hip.malloc(total)?;

    // The source holds the pattern; the destination is POISONED, so "the copy did nothing" fails.
    let mut host = vec![0u8; total as usize];
    for r in 0..regions {
        for i in 0..bytes {
            host[(r * bytes + i) as usize] = pattern(r, i);
        }
    }
    hip.memcpy_h2d_blocking(&src, &host, &stream)?;
    let poison = vec![0xA5u8; total as usize];
    hip.memcpy_h2d_blocking(&dst, &poison, &stream)?;

    let mut failures = 0usize;

    // ---- 1. a known pattern, across two allocations, byte-exact -------------------------------
    // Region 1 of the source into region 2 of the destination: neither offset is 0, so an
    // implementation that ignored an offset would land somewhere this compare can see.
    hip.memcpy_d2d_blocking(&dst, 2 * bytes, &src, bytes, bytes, &stream)?;
    let mut back = vec![0u8; total as usize];
    hip.memcpy_d2h_blocking(&mut back, &dst, &stream)?;
    let want: Vec<u8> = (0..bytes).map(|i| pattern(1, i)).collect();
    let got = &back[(2 * bytes) as usize..(3 * bytes) as usize];
    if got == want.as_slice() {
        println!("OK        {bytes} bytes copied between two allocations, byte-exact");
    } else {
        let first = (0..bytes as usize)
            .find(|&i| got[i] != want[i])
            .unwrap_or(usize::MAX);
        println!("MISMATCH  the copied region differs, first at byte {first}");
        failures += 1;
    }
    // Everything the copy was NOT asked to write is still the poison. A copy that wrote past its
    // length is the failure this catches, and it is the one a length bug looks like.
    let untouched_ok = back[..(2 * bytes) as usize].iter().all(|b| *b == 0xA5)
        && back[(3 * bytes) as usize..].iter().all(|b| *b == 0xA5);
    if untouched_ok {
        println!("OK        every byte outside the copied region is still the poison");
    } else {
        println!("MISMATCH  the copy wrote outside the region it was given");
        failures += 1;
    }

    // ---- 2. two offsets of ONE allocation — what the KV page copy really is --------------------
    hip.memcpy_h2d_blocking(&dst, &host, &stream)?;
    hip.memcpy_d2d_blocking(&dst, 0, &dst, 2 * bytes, bytes, &stream)?;
    hip.memcpy_d2h_blocking(&mut back, &dst, &stream)?;
    let want2: Vec<u8> = (0..bytes).map(|i| pattern(2, i)).collect();
    if back[..bytes as usize] == want2[..] {
        println!("OK        two disjoint offsets of ONE allocation (the KV pool's own shape)");
    } else {
        println!("MISMATCH  the in-allocation copy did not land");
        failures += 1;
    }
    // ...and the region it was copied FROM is unchanged, which is what makes the leader's page
    // read-only through a fork.
    if back[(2 * bytes) as usize..(3 * bytes) as usize] == want2[..] {
        println!("OK        the source region is unchanged (the leader's page is read-only)");
    } else {
        println!("MISMATCH  the copy moved its own source");
        failures += 1;
    }

    // ---- 3. the refusals ----------------------------------------------------------------------
    let cases: [(&str, u64, u64, u64, bool); 4] = [
        ("destination overruns", total - bytes + 1, 0, bytes, false),
        ("source overruns", 0, total - bytes + 1, bytes, false),
        ("in range", 0, bytes, bytes, true),
        ("length overflows a u64", 0, 0, u64::MAX, false),
    ];
    for (name, at_dst, at_src, n, want_ok) in cases {
        let got = hip.memcpy_d2d_blocking(&dst, at_dst, &src, at_src, n, &stream);
        match (got, want_ok) {
            (Err(e), false) => println!("OK        refused ({name}): {e}"),
            (Ok(()), true) => println!("OK        accepted ({name})"),
            (Ok(()), false) => {
                println!("MISMATCH  a copy that must be refused was posted ({name})");
                failures += 1;
            }
            (Err(e), true) => {
                println!("MISMATCH  a legal copy was refused ({name}): {e}");
                failures += 1;
            }
        }
    }
    // The overlap refusal is its own case: both ends are in range, and only the relation between
    // them is wrong.
    match hip.memcpy_d2d_blocking(&dst, bytes / 2, &dst, 0, bytes, &stream) {
        Err(e) => println!("OK        refused (overlapping ranges of one allocation): {e}"),
        Ok(()) => {
            println!("MISMATCH  an overlapping copy inside one allocation was posted");
            failures += 1;
        }
    }

    println!();
    if failures == 0 {
        println!("PASS      the D2D binding copies what it is given and refuses what it is not");
    } else {
        println!("FAIL      {failures} D2D case(s)");
    }
    Ok(failures)
}
