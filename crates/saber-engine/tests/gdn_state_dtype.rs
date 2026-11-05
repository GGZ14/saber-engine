//! C1: the f16 STORED GDN recurrent state (`--gdn-state-dtype f16`), on the host.
//!
//! Two laws, each with its control:
//!
//! 1. **The arena halves, and only the recurrent half of it.** Every slotted state's size comes
//!    from the graph's `bytes_per_slot` (the engine's one walk in `device.rs` allocates
//!    `per_slot x slots` per state), so the f16 graph's recurrent slots are half the fp32 graph's
//!    and its conv slots are the same bytes. Stated at the serving profile's TP=2, width 8, batch 8.
//! 2. **A park image carries the state AS STORED.** A session's image is one region per slotted
//!    state at its live slot (`device::session_regions`), `per_slot` long, so an f16 state parks
//!    and restores byte for byte into a different slot run — every one of the 65,536 f16 bit
//!    patterns, NaN payloads, subnormals and -0 included — through the real pinned tier. The fp32
//!    arm is the same path at the fp32 width, unchanged.

use saber_engine::head_share::{target_graph, TargetHead};
use saber_hip::DeviceAddr;
use saber_mem::dio::AlignedBuf;
use saber_mem::ledger::{Ledger, Pool};
use saber_mem::park::{ParkDevice, ParkError, ParkTier, PinnedHostTier, Region, RegionKind, Slab};
use saber_model::graph::{Layout, StateIndexedBy};
use saber_model::Graph;

/// The serving profile's graph: TP=2, widths up to 8, the checkpoint's fp8 head.
fn serving(f16: bool) -> Graph {
    let g = target_graph(TargetHead::Fp8, 2, 8).expect("the serving graph");
    if f16 {
        g.with_gdn_state_f16().expect("f16 state")
    } else {
        g
    }
}

/// `(name, per_slot, layout, indexed_by)` for every slotted state, deduplicated, in graph order —
/// the set the engine's allocation walk sees.
fn slotted(g: &Graph) -> Vec<(String, u64, Layout, StateIndexedBy)> {
    let mut seen = std::collections::BTreeSet::new();
    g.ops
        .iter()
        .flat_map(|o| o.state.iter())
        .filter(|s| s.bytes_per_slot.is_some() && seen.insert(s.name.clone()))
        .map(|s| {
            (
                s.name.clone(),
                s.bytes_per_slot.unwrap_or(0),
                s.layout,
                s.indexed_by,
            )
        })
        .collect()
}

/// Slot 0 is the kernels' null slot, and each live sequence owns `max_width` consecutive slots
/// from `FIRST_STATE_SLOT` (`StepIndex::state_slots_needed`).
const SLOTS_B8_W8: u64 = saber_exec::forward::FIRST_STATE_SLOT as u64 + 8 * 8;

#[test]
fn the_f16_state_halves_the_recurrent_arena_and_leaves_the_conv_cache() {
    let (a, b) = (slotted(&serving(false)), slotted(&serving(true)));
    assert_eq!(a.len(), b.len());
    assert_eq!(
        a.len(),
        96,
        "48 GDN layers x (conv cache + recurrent state)"
    );
    let (mut rec32, mut rec16, mut conv) = (0u64, 0u64, 0u64);
    for ((na, pa, la, ia), (nb, pb, lb, ib)) in a.iter().zip(&b) {
        assert_eq!((na, ia), (nb, ib));
        if na.ends_with(".recurrent_state") {
            assert_eq!(*la, Layout::Fp32State);
            assert_eq!(*lb, Layout::F16State);
            assert_eq!(*ia, StateIndexedBy::Sidx);
            // 24 v heads x 128 x 128 per rank: 4 bytes, then 2.
            assert_eq!(*pa, 24 * 128 * 128 * 4);
            assert_eq!(*pb, 24 * 128 * 128 * 2);
            rec32 += pa;
            rec16 += pb;
        } else {
            // The CONTROL: the conv cache is bf16 and untouched.
            assert_eq!((pa, la), (pb, lb), "{na}");
            conv += pa;
        }
    }
    // Per slot across the 48 layers: 72 MiB of recurrent state becomes 36.
    assert_eq!(rec32, 72 << 20);
    assert_eq!(rec16, 36 << 20);
    // The arena at batch 8 x width 8 (65 slots): 4,680 MiB of recurrent state per rank becomes
    // 2,340 MiB, and the conv cache is the same bytes in both.
    assert_eq!(rec32 * SLOTS_B8_W8, 4680 << 20);
    assert_eq!(rec16 * SLOTS_B8_W8, 2340 << 20);
    let total = |rec: u64| (rec + conv) * SLOTS_B8_W8;
    assert_eq!(
        total(rec32) - total(rec16),
        2340 << 20,
        "2.29 GiB per rank freed"
    );
}

/// A fake VRAM holding a few slotted buffers at fixed addresses, so a round trip can be compared
/// byte for byte (the same device double `saber-mem`'s own park tests use).
struct FakeVram {
    bufs: Vec<(u64, Vec<u8>)>,
}

impl FakeVram {
    fn at(&mut self, region: &Region) -> (&mut Vec<u8>, usize) {
        let addr = region.addr.as_u64();
        let (base, buf) = self
            .bufs
            .iter_mut()
            .find(|(b, v)| addr >= *b && addr < *b + v.len() as u64)
            .expect("a region inside a buffer");
        let off = (addr - *base) as usize;
        (buf, off)
    }
}

impl ParkDevice for FakeVram {
    fn d2h(&mut self, dst: &mut [u8], region: &Region) -> Result<(), ParkError> {
        let len = region.len as usize;
        let (buf, off) = self.at(region);
        dst[..len].copy_from_slice(&buf[off..off + len]);
        Ok(())
    }
    fn h2d(&mut self, region: &Region, src: &[u8]) -> Result<(), ParkError> {
        let len = region.len as usize;
        let (buf, off) = self.at(region);
        buf[off..off + len].copy_from_slice(&src[..len]);
        Ok(())
    }
}

/// The live slot of a run at `base` after `committed` tokens: `device::live_state_slot`'s law
/// (`docs/EXEC.md` § 9): the conv cache is the run's base, the recurrent snapshot `base + n - 1`.
fn live(indexed_by: StateIndexedBy, base: u32, committed: u32) -> u32 {
    match indexed_by {
        StateIndexedBy::Sidx => base + committed - 1,
        _ => base,
    }
}

fn round_trip(f16: bool) -> u64 {
    // Two GDN layers' slotted states (conv + recurrent each) of the serving graph, 17 slots per
    // buffer: the session parks from the run at slot 1 and restores into the run at slot 9.
    let states: Vec<_> = slotted(&serving(f16)).into_iter().take(4).collect();
    const SLOTS: u64 = 17;
    let (from, to, committed) = (1u32, 9u32, 6u32);
    let mut vram = FakeVram { bufs: vec![] };
    for (i, (_, per_slot, _, _)) in states.iter().enumerate() {
        let base = 0x1_0000_0000u64 * (i as u64 + 1);
        let bytes = per_slot * SLOTS;
        // Every 16-bit pattern, in order, as the stored words: at f16 that is every f16 value
        // (the NaN payloads, the subnormals and -0 among them); at fp32 the same bytes read as
        // fp32 words. Offset per buffer so no two regions carry the same bytes.
        let v: Vec<u8> = (0..bytes / 2)
            .flat_map(|w| ((w + i as u64 * 7919) as u16).to_le_bytes())
            .collect();
        vram.bufs.push((base, v));
    }
    let regions = |base_slot: u32| -> Vec<Region> {
        states
            .iter()
            .enumerate()
            .map(|(i, (_, per_slot, _, by))| {
                let slot = u64::from(live(*by, base_slot, committed));
                Region::new(
                    RegionKind::GdnState,
                    DeviceAddr::NULL
                        .offset(0x1_0000_0000u64 * (i as u64 + 1) + slot * per_slot)
                        .expect("an address"),
                    *per_slot,
                )
            })
            .collect()
    };
    let src = regions(from);
    let image: u64 = src.iter().map(|r| r.len).sum();
    let want: Vec<Vec<u8>> = src
        .iter()
        .map(|r| {
            let (buf, off) = vram.at(r);
            buf[off..off + r.len as usize].to_vec()
        })
        .collect();
    let slot_bytes = image.div_ceil(4096) * 4096;
    let slab: Box<dyn Slab> = Box::new(AlignedBuf::zeroed(slot_bytes as usize));
    let mut tier = PinnedHostTier::new(slab, 1, slot_bytes).expect("tier");
    let mut ledger = Ledger::new();
    ledger
        .declare(Pool::HostPinned, 2 * slot_bytes)
        .expect("pool");
    let ticket = tier.park(7, &src, &mut vram, &mut ledger).expect("park");
    assert_eq!(ticket.bytes, image, "the image is the states AS STORED");
    // The run it left is scribbled, the run it lands in is scribbled differently.
    for (_, v) in &mut vram.bufs {
        v.fill(0xEE);
    }
    let dst = regions(to);
    tier.restore(&ticket, &dst, &mut vram, &mut ledger, &mut |_| {})
        .expect("restore");
    for (r, w) in dst.iter().zip(&want) {
        let (buf, off) = vram.at(r);
        assert!(
            buf[off..off + r.len as usize] == w[..],
            "a state came back different"
        );
        // Nothing outside the region moved.
        assert!(buf[..off].iter().all(|&b| b == 0xEE));
        assert!(buf[off + r.len as usize..].iter().all(|&b| b == 0xEE));
    }
    image
}

#[test]
fn an_f16_state_parks_and_restores_byte_exact_into_another_slot_run() {
    let narrow = round_trip(true);
    // The CONTROL: the fp32 path, unchanged — same law, twice the recurrent bytes.
    let wide = round_trip(false);
    assert_eq!(
        wide - narrow,
        2 * 24 * 128 * 128 * 2,
        "two recurrent slots, half the bytes each; the conv regions are the same size"
    );
}
