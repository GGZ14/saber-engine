//! FN71c: the boot fill's ring through the P2PDMA pool (`saber_mem::p2p_fill`) over a model of an
//! ASYNCHRONOUS copy lane: a copy reads its ring slot only when its ticket is waited on, so a read
//! that overwrote a ring slot before its copy landed would deliver the WRONG record -- the reuse
//! law is what keeps every destination equal to its record. The control drops the law (no wait
//! before a reuse) and must deliver wrong bytes.

use std::collections::HashMap;

use saber_mem::p2p_fill::{fill_through_ring, ring_slots, PoolFillIo, FILL_QD, RING_CAP_BYTES};

const STRIDE: u64 = 2_617_344;

/// Record `off` of the "drive" is the byte pattern of its offset (one u64, enough to tell records
/// apart). The pool and the VRAM are maps of slot / address -> that u64.
struct Model {
    pool: Vec<u64>,
    vram: HashMap<u64, u64>,
    copies: HashMap<u64, (u32, u64)>,
    next: u64,
    /// The control: `wait` is a no-op until the end, so a reused slot's copy reads the new bytes.
    lazy: bool,
    deferred: Vec<u64>,
}

impl Model {
    fn new(ring: u32, lazy: bool) -> Model {
        Model {
            pool: vec![u64::MAX; ring as usize],
            vram: HashMap::new(),
            copies: HashMap::new(),
            next: 1,
            lazy,
            deferred: Vec::new(),
        }
    }
    fn land(&mut self, t: u64) {
        if let Some((slot, dst)) = self.copies.remove(&t) {
            self.vram.insert(dst, self.pool[slot as usize]);
        }
    }
}

impl PoolFillIo for Model {
    type Err = String;
    fn read(&mut self, items: &[(u64, u32)]) -> Result<(), String> {
        for &(off, s) in items {
            self.pool[s as usize] = off;
        }
        Ok(())
    }
    fn copy(&mut self, slot: u32, dst: u64) -> Result<u64, String> {
        let t = self.next;
        self.next += 1;
        self.copies.insert(t, (slot, dst));
        Ok(t)
    }
    fn wait(&mut self, t: u64) -> Result<(), String> {
        if self.lazy {
            self.deferred.push(t);
        } else {
            self.land(t);
        }
        Ok(())
    }
}

/// The law-free fill (the control): reads reuse ring slots with no wait, every copy lands late.
fn fill_without_the_law(m: &mut Model, recs: &[(u64, u64)], ring: u32, qd: usize) {
    let mut next = 0u32;
    for batch in recs.chunks(qd) {
        let items: Vec<(u64, u32)> = batch
            .iter()
            .map(|&(off, _)| {
                let s = next;
                next = (next + 1) % ring;
                (off, s)
            })
            .collect();
        m.read(&items).unwrap();
        for (&(_, s), &(_, dst)) in items.iter().zip(batch) {
            m.copy(s, dst).unwrap();
        }
    }
    let ts: Vec<u64> = m.copies.keys().copied().collect();
    let mut ts = ts;
    ts.sort_unstable();
    for t in ts {
        m.land(t);
    }
}

fn records(n: u64) -> Vec<(u64, u64)> {
    // a scattered placement: record k -> slot (k * 7919) % n (7919 prime: a permutation) of a 0x7f.. arena
    (0..n)
        .map(|k| {
            (
                4096 + k * STRIDE,
                0x7f00_0000_0000 + ((k * 7919) % n) * STRIDE,
            )
        })
        .collect()
}

#[test]
fn the_ring_follows_the_pool_and_refuses_a_missing_or_tiny_pool() {
    // today's 2 GiB pool: the 1.5 GiB cap binds
    assert_eq!(
        ring_slots(2048 << 20, STRIDE, RING_CAP_BYTES, FILL_QD),
        Ok(615)
    );
    // the user's possible 1,024 MiB pool: the pool binds
    assert_eq!(
        ring_slots(1024 << 20, STRIDE, RING_CAP_BYTES, FILL_QD),
        Ok(410)
    );
    // no pool (p2pmem_mb 0 / absent): refused by name
    let e = ring_slots(0, STRIDE, RING_CAP_BYTES, FILL_QD).unwrap_err();
    assert!(e.contains("no P2PDMA pool"), "{e}");
    // a pool below two batches: refused
    assert!(ring_slots(128 << 20, STRIDE, RING_CAP_BYTES, FILL_QD).is_err());
    assert_eq!(
        ring_slots(168 << 20, STRIDE, RING_CAP_BYTES, FILL_QD),
        Ok(67)
    );
}

#[test]
fn every_destination_gets_its_record_and_a_ring_slot_is_reused_only_after_its_copy() {
    for (n, ring, qd) in [
        (7_000u64, 615u32, 32usize),
        (1_000, 410, 32),
        (333, 64, 32),
        (5, 615, 32),
    ] {
        let recs = records(n);
        let mut m = Model::new(ring, false);
        let c = fill_through_ring(&mut m, &recs, ring, qd).unwrap();
        assert_eq!(c.reads, n);
        assert_eq!(c.copies, n);
        assert_eq!(c.batches, n.div_ceil(qd as u64));
        assert_eq!(c.reuse_waits, n.saturating_sub(u64::from(ring)));
        assert!(m.copies.is_empty(), "every copy landed at the end");
        for &(off, dst) in &recs {
            assert_eq!(
                m.vram.get(&dst),
                Some(&off),
                "n {n} ring {ring}: dst {dst:#x}"
            );
        }
    }
}

#[test]
fn the_control_without_the_reuse_wait_delivers_wrong_records() {
    let (n, ring, qd) = (2_000u64, 64u32, 32usize);
    let recs = records(n);
    let mut m = Model::new(ring, true);
    fill_without_the_law(&mut m, &recs, ring, qd);
    let wrong = recs
        .iter()
        .filter(|&&(off, dst)| m.vram.get(&dst) != Some(&off))
        .count();
    assert!(wrong as u64 >= n - u64::from(ring), "wrong {wrong}");
}
