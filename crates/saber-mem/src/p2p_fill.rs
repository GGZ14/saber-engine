//! FN71c: the boot VRAM fill THROUGH the card's P2PDMA pool (`--xtier-boot-fill`).
//!
//! The pool (`amdgpu.p2pmem_mb`, FN71/FN71b) is a FIXED VRAM region the drive can write with no
//! host DRAM on the path (`saber_hsa::NvmeVram`'s Pool route). The expert slots are elsewhere in
//! VRAM, so the boot fill runs the pool as a RING: `qd` records at a time are read drive -> ring
//! slots (parallel `O_DIRECT` `pread`s into the pool's host view), then each ring slot is copied
//! into its expert slot on the reader's SDMA lane (an intra-VRAM copy). A ring slot is read into
//! again only after its previous copy landed (the host waits that copy's ticket first) -- THE
//! REUSE LAW, the one hazard of the ring; [`fill_through_ring`] keeps it for any [`PoolFillIo`],
//! and the host test drives it over a model whose copies land only when waited on.
//!
//! Bit-exact by construction: bytes are bytes (no conversion anywhere on the path); the engine's
//! end-of-boot readback (`fnslots`) compares every slot with the xtier's sha256 table.

/// The ring's byte cap (the brief's "~1.5 GiB in use"): the rest of a 2 GiB pool stays free.
pub const RING_CAP_BYTES: u64 = 1536 << 20;

/// Records in flight a batch (the drive's rate is flat from QD 8 to 32: FN71b 9.63 / 9.48 GB/s).
pub const FILL_QD: usize = 32;

/// The ring's slots for a pool of `pool_bytes` (read from `p2pmem/size` at boot, so the ring
/// follows whatever pool the boot has: 2,048 MiB -> 615 slots of 2.62 MB, 1,024 MiB -> 410):
/// `min(pool, cap) / stride`, refused below two batches (one batch reading while the previous
/// one's copies run).
pub fn ring_slots(pool_bytes: u64, stride: u64, cap: u64, qd: usize) -> Result<u32, String> {
    if stride == 0 {
        return Err("record stride 0".into());
    }
    if pool_bytes == 0 {
        return Err("no P2PDMA pool on this card (p2pmem/size 0 or absent)".into());
    }
    let n = pool_bytes.min(cap) / stride;
    let need = 2 * qd.max(1) as u64;
    if n < need {
        return Err(format!(
            "the pool's {} MiB hold {n} records of {stride} B; the ring needs {need} (2 x QD {qd})",
            pool_bytes >> 20
        ));
    }
    Ok(u32::try_from(n).unwrap_or(u32::MAX))
}

/// What the ring drives (the engine: `saber_hsa::NvmeVram`; the test: a model).
pub trait PoolFillIo {
    /// The error type.
    type Err;
    /// Read each `(file offset, ring slot)` of `stride` bytes into the pool; complete on return.
    fn read(&mut self, items: &[(u64, u32)]) -> Result<(), Self::Err>;
    /// Copy ring slot `slot` into the device address `dst` (asynchronous); returns a ticket.
    fn copy(&mut self, slot: u32, dst: u64) -> Result<u64, Self::Err>;
    /// The host waits until ticket `t` landed.
    fn wait(&mut self, t: u64) -> Result<(), Self::Err>;
}

/// What a fill did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FillCounts {
    /// Records read drive -> pool.
    pub reads: u64,
    /// Pool -> slot copies issued.
    pub copies: u64,
    /// Batches.
    pub batches: u64,
    /// Ring slots whose previous copy the host waited on before reusing them.
    pub reuse_waits: u64,
}

/// Fill every `(file offset, device destination)` of `recs` through a ring of `ring` slots,
/// `qd` records a batch: wait the ring slots' previous copies, read the batch into them, issue
/// one copy each; at the end wait every copy. Every byte landed when it returns `Ok`.
pub fn fill_through_ring<I: PoolFillIo>(
    io: &mut I,
    recs: &[(u64, u64)],
    ring: u32,
    qd: usize,
) -> Result<FillCounts, I::Err> {
    let ring = ring.max(1);
    let qd = qd.clamp(1, ring as usize);
    let mut pending: Vec<u64> = vec![0; ring as usize];
    let mut c = FillCounts::default();
    let mut next = 0u32;
    for batch in recs.chunks(qd) {
        let mut items = Vec::with_capacity(batch.len());
        for &(off, _) in batch {
            let s = next;
            next = (next + 1) % ring;
            let t = std::mem::take(&mut pending[s as usize]);
            if t != 0 {
                io.wait(t)?;
                c.reuse_waits += 1;
            }
            items.push((off, s));
        }
        io.read(&items)?;
        c.reads += items.len() as u64;
        for (&(_, s), &(_, dst)) in items.iter().zip(batch) {
            pending[s as usize] = io.copy(s, dst)?;
            c.copies += 1;
        }
        c.batches += 1;
    }
    for t in pending {
        if t != 0 {
            io.wait(t)?;
        }
    }
    Ok(c)
}
