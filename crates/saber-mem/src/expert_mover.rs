//! Host implementations of [`ExpertMover`]: the byte-true one the tests run and the timing one
//! the replay bench runs.
//!
//! * [`HostMover`] holds a fake VRAM slot arena and a RAM slab as ordinary aligned host memory and
//!   reads records with real `O_DIRECT` (or buffered) reads from the `experts.xtier` — every copy
//!   the tier asks for moves real bytes, so a test can check every slot against the sha256 table
//!   after any number of steps.
//! * [`DrainMover`] reads the records the tier asks for from the drive for real (one aligned read
//!   per record, `qd` in flight) into a small ring of buffers, times them, and COUNTS the copies
//!   instead of making them: the replay of a 48-layer route trace needs the drive's real behaviour
//!   and the link's bill, not 18 GB of fake VRAM.

#![forbid(unsafe_code)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use crate::dio::AlignedBuf;
use crate::expert_tier::{ExpertMover, TierError};
use crate::xtier::XtierFile;

/// Read `reads` (record, destination slot) with up to `qd` threads, each read one aligned stride
/// straight into its destination slot of `slab`.
pub fn parallel_reads(
    xt: &XtierFile,
    slab: &mut [u8],
    stride: usize,
    reads: &[(u64, u32)],
    qd: usize,
) -> Result<(), TierError> {
    let mut slots: Vec<Option<&mut [u8]>> = slab.chunks_mut(stride).map(Some).collect();
    let mut jobs: Vec<(u64, &mut [u8])> = Vec::with_capacity(reads.len());
    for &(rec, s) in reads {
        let dst = slots
            .get_mut(s as usize)
            .and_then(Option::take)
            .ok_or_else(|| {
                TierError::Mover(format!("slot {s} missing or read twice in one batch"))
            })?;
        jobs.push((rec, dst));
    }
    let workers = qd.max(1).min(jobs.len().max(1));
    let next = AtomicUsize::new(0);
    type Job<'a> = std::sync::Mutex<Option<(u64, &'a mut [u8])>>;
    let jobs: Vec<Job<'_>> = jobs
        .into_iter()
        .map(|j| std::sync::Mutex::new(Some(j)))
        .collect();
    let err: std::sync::Mutex<Option<TierError>> = std::sync::Mutex::new(None);
    std::thread::scope(|sc| {
        for _ in 0..workers {
            sc.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                let Some(cell) = jobs.get(k) else { break };
                let job = cell.lock().expect("job lock").take();
                if let Some((rec, dst)) = job {
                    if let Err(e) = xt.read_record(rec, dst) {
                        *err.lock().expect("err lock") = Some(e.into());
                        break;
                    }
                }
            });
        }
    });
    match err.into_inner().expect("err lock") {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// The byte-true host mover (see the module docs).
#[derive(Debug)]
pub struct HostMover<'x> {
    xt: &'x XtierFile,
    stride: usize,
    /// The fake VRAM slot arena.
    pub vram: AlignedBuf,
    /// The RAM slab (`ram_slots + 1` slots, the last the spare).
    pub ram: AlignedBuf,
    qd: usize,
    /// NVMe records read.
    pub nvme_reads: u64,
    /// Copies made (h2d, d2h).
    pub copies: (u64, u64),
}

impl<'x> HostMover<'x> {
    /// A mover over `vram_slots` fake VRAM slots and `ram_slots + 1` RAM slots.
    #[must_use]
    pub fn new(xt: &'x XtierFile, vram_slots: u32, ram_slots: u32, qd: usize) -> HostMover<'x> {
        let stride = xt.header().stride as usize;
        HostMover {
            xt,
            stride,
            vram: AlignedBuf::zeroed(stride * vram_slots as usize),
            ram: AlignedBuf::zeroed(stride * (ram_slots as usize + 1)),
            qd,
            nvme_reads: 0,
            copies: (0, 0),
        }
    }

    /// The bytes of VRAM slot `s`.
    #[must_use]
    pub fn vram_slot(&self, s: u32) -> &[u8] {
        &self.vram[s as usize * self.stride..(s as usize + 1) * self.stride]
    }

    /// The bytes of RAM slot `s`.
    #[must_use]
    pub fn ram_slot(&self, s: u32) -> &[u8] {
        &self.ram[s as usize * self.stride..(s as usize + 1) * self.stride]
    }

    /// Realise a warm plan: read every placed record into its slot (VRAM ones through the RAM
    /// spare, as a boot through a staging slot would).
    pub fn warm(
        &mut self,
        plan: &[(u64, crate::expert_tier::Placement)],
        spare: u32,
    ) -> Result<(), TierError> {
        use crate::expert_tier::Placement;
        let ram: Vec<(u64, u32)> = plan
            .iter()
            .filter_map(|&(i, p)| match p {
                Placement::Ram(s) => Some((i, s)),
                _ => None,
            })
            .collect();
        for chunk in ram.chunks(256) {
            self.nvme_to_ram(chunk)?;
        }
        for &(i, p) in plan {
            if let Placement::Vram(s) = p {
                self.nvme_to_ram(&[(i, spare)])?;
                self.ram_to_vram(spare, s)?;
            }
        }
        self.nvme_reads = 0;
        self.copies = (0, 0);
        Ok(())
    }
}

impl ExpertMover for HostMover<'_> {
    fn nvme_to_ram(&mut self, reads: &[(u64, u32)]) -> Result<(), TierError> {
        parallel_reads(self.xt, &mut self.ram, self.stride, reads, self.qd)?;
        self.nvme_reads += reads.len() as u64;
        Ok(())
    }
    fn ram_to_vram(&mut self, ram: u32, vram: u32) -> Result<(), TierError> {
        let s = self.stride;
        let (r, v) = (ram as usize * s, vram as usize * s);
        self.vram[v..v + s].copy_from_slice(&self.ram[r..r + s]);
        self.copies.0 += 1;
        Ok(())
    }
    fn vram_to_ram(&mut self, vram: u32, ram: u32) -> Result<(), TierError> {
        let s = self.stride;
        let (r, v) = (ram as usize * s, vram as usize * s);
        self.ram[r..r + s].copy_from_slice(&self.vram[v..v + s]);
        self.copies.1 += 1;
        Ok(())
    }
    fn flush(&mut self) -> Result<(), TierError> {
        Ok(())
    }
}

/// The timing mover (see the module docs): real reads, counted copies.
#[derive(Debug)]
pub struct DrainMover<'x> {
    xt: &'x XtierFile,
    stride: usize,
    ring: AlignedBuf,
    qd: usize,
    /// Every NVMe batch: (records, seconds).
    pub batches: Vec<(usize, f64)>,
    /// Copies counted (h2d, d2h).
    pub copies: (u64, u64),
    /// When false the reads are skipped (counting only).
    pub read: bool,
}

impl<'x> DrainMover<'x> {
    /// A timing mover with a `qd`-slot ring.
    #[must_use]
    pub fn new(xt: &'x XtierFile, qd: usize) -> DrainMover<'x> {
        let stride = xt.header().stride as usize;
        DrainMover {
            xt,
            stride,
            ring: AlignedBuf::zeroed(stride * qd.max(1)),
            qd: qd.max(1),
            batches: Vec::new(),
            copies: (0, 0),
            read: true,
        }
    }
}

impl ExpertMover for DrainMover<'_> {
    fn nvme_to_ram(&mut self, reads: &[(u64, u32)]) -> Result<(), TierError> {
        let t = Instant::now();
        if self.read {
            for chunk in reads.chunks(self.qd) {
                let ring: Vec<(u64, u32)> = chunk
                    .iter()
                    .enumerate()
                    .map(|(k, &(r, _))| (r, k as u32))
                    .collect();
                parallel_reads(self.xt, &mut self.ring, self.stride, &ring, self.qd)?;
            }
        }
        self.batches.push((reads.len(), t.elapsed().as_secs_f64()));
        Ok(())
    }
    fn ram_to_vram(&mut self, _ram: u32, _vram: u32) -> Result<(), TierError> {
        self.copies.0 += 1;
        Ok(())
    }
    fn vram_to_ram(&mut self, _vram: u32, _ram: u32) -> Result<(), TierError> {
        self.copies.1 += 1;
        Ok(())
    }
    fn flush(&mut self) -> Result<(), TierError> {
        Ok(())
    }
}
