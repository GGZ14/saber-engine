//! Session parking: an idle session's KV **and** GDN state evicted as ONE unit.
//!
//! `docs/ARCHITECTURE.md` § Memory plane, *Session parking*: *"Evict an idle session's KV and its
//! 72 MiB/rank GDN state as one unit — they are useless apart. ~5 ms restore from RAM at PCIe
//! speed; **~12 ms from the CPU-attached Gen5 drive**, which makes NVMe a real second parking tier
//! rather than a cold store. Park to RAM first, spill to NVMe under pressure. Parking is *write*
//! traffic on NVMe: the byte ledger counts NVMe write bytes."*
//!
//! ```text
//!    VRAM                    pinned RAM slot                  NVMe slot
//!   ┌──────────┐   D2H      ┌───────────────┐   pwrite       ┌──────────────┐
//!   │ KV blocks│ ─────────► │ one contiguous│ ─────demote───►│ O_DIRECT file│
//!   │ GDN state│ ◄───────── │ image + sha256│ ◄────promote───│ fallocate'd  │
//!   └──────────┘   H2D      └───────────────┘   pread        └──────────────┘
//! ```
//!
//! What is structural here rather than policy:
//!
//! * **One unit.** A park takes a *list of regions* and stores them as one image with one ticket.
//!   There is no call that parks half a session, and [`ParkTier::restore`] refuses a destination
//!   list whose region kinds and lengths are not the ones that were parked.
//! * **Allocated once (I1).** [`PinnedHostTier`] splits one `hipHostMalloc` into fixed slots at
//!   construction; a park takes a free slot or is refused. Nothing allocates on the park path.
//! * **Hash on park, verified on restore, before anything reaches VRAM (I3).** A parked image
//!   carries its `sha256`; a restore that does not match refuses and copies **nothing** — the
//!   negative control in `tests/park.rs` and `saber park-bench` (on unless `--no-corrupt`) both
//!   prove it, because a
//!   park that silently perturbs state is the 09-17 lazy-snapshot failure in another costume.
//! * **Billed.** Every image reserves its bytes in the [`Ledger`] before it is stored, and NVMe
//!   parks record their write traffic — the number `/metrics` needs against a 2400 TBW drive.
//!
//! The device half is behind [`ParkDevice`] so the whole thing runs on a host with no ROCm:
//! `tests/park.rs` drives the real tiers with a fake VRAM. [`HipParkDevice`] is the real one.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};

use saber_hip::{DeviceAddr, DevicePtr, HipThread, PinnedBuf, PinnedChunk, Stream, StreamFlags};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::dio::{self, align_up, AlignedBuf, OpenMode, DIO_ALIGN};
use crate::ledger::{Ledger, LedgerError, Pool};

/// Host memory a parking tier hands its slots out of.
///
/// Two implementations, and the difference is the whole reason this is a trait: [`PinnedSlab`] is
/// page-locked memory from `hipHostMalloc`, so a park is a DMA; [`crate::dio::AlignedBuf`] is
/// ordinary 4096-aligned heap memory, which is what `saber serve --mock` and the scheduler's own
/// tests run on — a box with no ROCm can still exercise every decision the tiers make.
///
/// `Send` and no lifetime, because a tier ends up inside the scheduler's `Box<dyn Parkable>`,
/// which is `Send + 'static`.
pub trait Slab: Send + std::fmt::Debug {
    /// Bytes the slab holds.
    fn len(&self) -> usize;
    /// Whether it holds none.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Host address of the first byte. `O_DIRECT` needs it 4096-aligned and the tiers check.
    fn addr(&self) -> usize;
    /// The bytes.
    fn as_slice(&self) -> &[u8];
    /// The bytes, mutably.
    fn as_mut_slice(&mut self) -> &mut [u8];
}

impl Slab for AlignedBuf {
    fn len(&self) -> usize {
        AlignedBuf::len(self)
    }
    fn addr(&self) -> usize {
        self.as_ptr() as usize
    }
    fn as_slice(&self) -> &[u8] {
        self
    }
    fn as_mut_slice(&mut self) -> &mut [u8] {
        self
    }
}

/// A page-locked slab: one `hipHostMalloc`, held for the life of the process.
///
/// The buffer is **leaked** on purpose. A parking pool is allocated once (I1) and its slots must
/// not move while a tier is handing them out; a slab that could free its memory would be a slab
/// whose slots could. Leaking is how "this lives for the process" is said in safe Rust, and the
/// engine allocates exactly one of these.
#[derive(Debug)]
pub struct PinnedSlab {
    chunk: PinnedChunk<'static>,
}

impl PinnedSlab {
    /// Take a pinned buffer and hand its whole length to a tier.
    pub fn new(buf: PinnedBuf) -> Result<PinnedSlab, ParkError> {
        let len = buf.len();
        if len == 0 {
            return Err(ParkError::Config("a pinned slab of 0 bytes".into()));
        }
        let leaked: &'static mut PinnedBuf = Box::leak(Box::new(buf));
        let chunk = leaked
            .send_chunks(len)
            .and_then(|mut v| v.pop())
            .ok_or_else(|| ParkError::Config("the pinned buffer cannot be split".into()))?;
        Ok(PinnedSlab { chunk })
    }
}

impl Slab for PinnedSlab {
    fn len(&self) -> usize {
        self.chunk.len()
    }
    fn addr(&self) -> usize {
        self.chunk.addr()
    }
    fn as_slice(&self) -> &[u8] {
        self.chunk.as_slice()
    }
    fn as_mut_slice(&mut self) -> &mut [u8] {
        self.chunk.as_mut_slice()
    }
}

/// Which half of a session an image region holds.
///
/// Recorded per region because the two are useless apart and a restore must put each back where it
/// came from: a KV block written into a GDN state slot is not a crash, it is a wrong answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RegionKind {
    /// Paged KV blocks.
    Kv,
    /// GDN recurrent state slots — 72 MiB per rank at prod's shapes.
    GdnState,
    /// The DRAFTER's own paged K/V, one run of pages per layer per cache.
    ///
    /// A third kind rather than more `Kv` regions, because it is a different pool with a different
    /// page size in a different allocation: a drafter page restored into the target's pool would
    /// be in range and silent. A session that has a drafter has these and a session that does not
    /// does not, so the image's own layout says which kind of session it is (ws-s5).
    DraftKv,
}

impl std::fmt::Display for RegionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegionKind::Kv => write!(f, "kv"),
            RegionKind::GdnState => write!(f, "gdn_state"),
            RegionKind::DraftKv => write!(f, "draft_kv"),
        }
    }
}

/// One contiguous device range belonging to a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Region {
    /// Which half of the session it is.
    pub kind: RegionKind,
    /// Device address of the first byte.
    pub addr: DeviceAddr,
    /// Length in bytes.
    pub len: u64,
}

impl Region {
    /// A region of `len` bytes at `addr`.
    #[must_use]
    pub fn new(kind: RegionKind, addr: DeviceAddr, len: u64) -> Self {
        Region { kind, addr, len }
    }
}

/// The shape of a parked image: what a restore's destination list has to match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionSpec {
    /// Which half.
    pub kind: RegionKind,
    /// Length in bytes.
    pub len: u64,
}

/// A receipt for a parked session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParkTicket {
    /// Tier-local id.
    pub id: u64,
    /// The session this image belongs to, as the caller named it.
    pub session: u64,
    /// Which pool it is billed to.
    pub pool: Pool,
    /// Payload bytes (unpadded).
    pub bytes: u64,
    /// The image's layout, in order.
    pub layout: Vec<RegionSpec>,
    /// `sha256` over the image bytes, taken on park.
    pub sha256: [u8; 32],
    /// The per-chunk digests [`ParkTicket::sha256`] is folded from, in order.
    ///
    /// They are what lets a restore verify chunk by chunk and send each chunk as soon as it is
    /// known good, instead of hashing the whole image before the first byte moves. The folded
    /// value stays the authority: a restore checks that folding this list reproduces it, so a
    /// tampered list is caught by the single value it has to agree with. Empty when verification
    /// is off, and ~32 bytes per 4 MiB of session (6.7 KiB for an 852 MiB image).
    pub chunks: Vec<[u8; 32]>,
}

impl ParkTicket {
    /// The ledger reservation name — the same `session:{id}` shape the scheduler's mock tier uses,
    /// so a `/metrics` reader sees one naming convention.
    #[must_use]
    pub fn reservation(&self) -> String {
        format!("session:{}", self.session)
    }
}

/// One region that has landed back in VRAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionReady {
    /// Index in the destination list.
    pub index: usize,
    /// Which half of the session.
    pub kind: RegionKind,
    /// Where it landed.
    pub addr: DeviceAddr,
    /// How many bytes.
    pub len: u64,
}

/// Why a park or restore failed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ParkError {
    /// The byte ledger refused: the tier is full in the sense that matters.
    #[error(transparent)]
    Ledger(#[from] LedgerError),
    /// HIP refused a copy.
    #[error(transparent)]
    Hip(#[from] saber_hip::Error),
    /// The parking file.
    #[error("park file {path}: {what}: {source}")]
    Io {
        /// The file.
        path: String,
        /// What was being done.
        what: String,
        /// The OS error.
        source: std::io::Error,
    },
    /// Caller error, caught before anything moved.
    #[error("{0}")]
    Config(String),
    /// Every slot holds an image. The scheduler's answer is to demote one, not to grow the tier.
    #[error("park tier {pool} is full: {slots} slots, all occupied")]
    TierFull {
        /// The pool.
        pool: Pool,
        /// Slots the tier has.
        slots: usize,
    },
    /// The image does not fit one slot.
    #[error("session image is {bytes} bytes; a {pool} slot holds {slot_bytes}")]
    ImageTooLarge {
        /// The pool.
        pool: Pool,
        /// Image bytes.
        bytes: u64,
        /// Slot bytes.
        slot_bytes: u64,
    },
    /// No such ticket in this tier.
    #[error("park tier {pool} has no image {id}")]
    NoSuchImage {
        /// The pool.
        pool: Pool,
        /// The ticket id.
        id: u64,
    },
    /// The destination list is not the shape that was parked.
    #[error("restore of image {id}: {detail}")]
    LayoutMismatch {
        /// The ticket id.
        id: u64,
        /// What differed.
        detail: String,
    },
    /// The stored image does not hash to what was parked. **Nothing was copied to the device.**
    #[error(
        "parked image {id} (session {session}) is corrupt: sha256 {found} != {expected}; \
         nothing was restored to VRAM"
    )]
    ImageCorrupt {
        /// The ticket id.
        id: u64,
        /// The session.
        session: u64,
        /// What the ticket says.
        expected: String,
        /// What the bytes say now.
        found: String,
    },
}

impl ParkError {
    fn io(path: &Path, what: impl Into<String>, source: std::io::Error) -> Self {
        ParkError::Io {
            path: path.display().to_string(),
            what: what.into(),
            source,
        }
    }
}

/// The device half of parking: move a region's bytes out of VRAM and back.
///
/// Byte-slice shaped on purpose. A session image is two or three regions, so the per-call cost of
/// the HIP thread round trip (3.24 µs, workstream H) is 0.03% of a 5–20 ms park — this is not the
/// 1018-record cold-start stream, and batching would buy nothing but indirection. Both calls are
/// **complete when they return**, which is what lets `restore` report a region as ready.
pub trait ParkDevice {
    /// Copy the region out of VRAM into `dst`, which is exactly `region.len` bytes of pinned host
    /// memory.
    fn d2h(&mut self, dst: &mut [u8], region: &Region) -> Result<(), ParkError>;

    /// Copy `src` back into the region.
    fn h2d(&mut self, region: &Region, src: &[u8]) -> Result<(), ParkError>;
}

/// What a tier has done since it was built.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TierStats {
    /// Images parked.
    pub parks: u64,
    /// Images restored.
    pub restores: u64,
    /// Images discarded (their request was aborted).
    pub discards: u64,
    /// Images promoted into this tier from the slower one.
    pub promotions: u64,
    /// Images demoted out of this tier into the slower one.
    pub demotions: u64,
    /// Payload bytes written into the tier.
    pub bytes_in: u64,
    /// Payload bytes read back out.
    pub bytes_out: u64,
    /// Restores refused because the stored image did not hash to its ticket.
    pub corrupt_refusals: u64,
}

/// A tier a session can be parked to: pinned host RAM first, NVMe under pressure.
///
/// The shape mirrors `saber_sched::park::Parkable` (park / restore / discard / len, a ticket, a
/// ledger passed in rather than held) so the scheduler can swap its mock for one of these behind a
/// thin adapter. It cannot be that trait: `saber-sched` depends on `saber-mem`, and the trait's
/// image type is the scheduler's.
pub trait ParkTier {
    /// The byte pool this tier bills.
    fn pool(&self) -> Pool;

    /// Slots the tier has. A park is refused when they are all occupied; the scheduler's answer is
    /// to demote, not to grow.
    fn slots(&self) -> usize;

    /// Bytes one slot holds.
    fn slot_bytes(&self) -> u64;

    /// Images currently held.
    fn len(&self) -> usize;

    /// Whether the tier holds nothing.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Store an image the caller already holds in host memory, under `session`.
    ///
    /// The other shape of the same operation. `park` copies the regions out of VRAM itself, which
    /// is what `saber-exec` will do when it can hand out device addresses; this takes the bytes an
    /// executor has *already* copied out — today's `StepExecutor::park_state` contract, and the
    /// only shape that works with an executor that has no device at all.
    ///
    /// The ticket carries no region layout: the executor packed the image and only it knows where
    /// the KV half ends, so what the tier guarantees is the length and the digest.
    fn park_bytes(
        &mut self,
        session: u64,
        bytes: &[u8],
        ledger: &mut Ledger,
    ) -> Result<ParkTicket, ParkError>;

    /// Copy a parked image back into `dst`, which must be exactly the length that was parked.
    ///
    /// Verified first: a corrupt image is refused and `dst` is left untouched, exactly as the
    /// region path refuses before its first H2D.
    fn restore_bytes(
        &mut self,
        ticket: &ParkTicket,
        dst: &mut [u8],
        ledger: &mut Ledger,
    ) -> Result<(), ParkError>;

    /// Take a session out of VRAM: D2H every region into one slot, hash it, bill it.
    fn park(
        &mut self,
        session: u64,
        regions: &[Region],
        device: &mut dyn ParkDevice,
        ledger: &mut Ledger,
    ) -> Result<ParkTicket, ParkError>;

    /// Put a session back: verify the image, then H2D each region to the **destination** address
    /// the caller gives (the scheduler re-admits into fresh slots), reporting each as it lands.
    fn restore(
        &mut self,
        ticket: &ParkTicket,
        dst: &[Region],
        device: &mut dyn ParkDevice,
        ledger: &mut Ledger,
        on_ready: &mut dyn FnMut(RegionReady),
    ) -> Result<(), ParkError>;

    /// Drop a parked session whose request was aborted, releasing its reservation.
    fn discard(&mut self, ticket: &ParkTicket, ledger: &mut Ledger);

    /// Counters.
    fn stats(&self) -> TierStats;
}

/// Bytes per digest chunk. A **constant**, not a knob: the digest folds one `sha256` per chunk, so
/// changing this changes every digest, and a ticket written by one build has to be readable by the
/// next.
pub const DIGEST_CHUNK: usize = 4 << 20;

/// Bytes per device copy in the overlapped path — a **multiple** of [`DIGEST_CHUNK`], and a
/// different thing from it.
///
/// The hash wants small chunks (finer granularity, more overlap, less tail). The copy wants large
/// ones: every piece is a blocking `hipMemcpyAsync` plus a stream synchronise, which is a round
/// trip to the HIP thread and a DMA that spins down and back up. Measured on this box at 136 MiB
/// (34 digest chunks), copying at digest granularity cost ~2.6 ms on a 5.1 ms transfer — about
/// 75 µs per piece. At 16 MiB there are four times fewer pieces and the digest is unchanged,
/// because the digest is folded over `DIGEST_CHUNK` pieces whatever the copy did.
pub const COPY_CHUNK: usize = 4 * DIGEST_CHUNK;

/// A session image's content digest: `sha256` over the per-chunk `sha256` digests, with the image
/// length and the chunk size folded in.
///
/// **Not `sha256(image)`**, and the difference is the point. A plain `sha256` runs at ~2.3 GB/s on
/// one core with SHA-NI, so a 136 MiB session image costs **58 ms** to hash — measured, and more
/// than ten times the whole transfer it is supposed to protect. Chunking lets the hash use the
/// cores that are idle while the DMA runs, and folding the chunk digests keeps it a single
/// collision-resistant value. The length and the chunk size are folded in so that a different
/// split cannot produce the same digest.
///
/// The value does not depend on `threads`: that only decides how fast it is computed.
#[must_use]
pub fn image_digest(bytes: &[u8], threads: usize) -> [u8; 32] {
    fold_chunk_digests(bytes.len(), &chunk_digests(bytes, threads))
}

/// The per-chunk `sha256` digests of an image, in order.
///
/// Exposed because a restore verifies **chunk by chunk** so that it can start sending the image
/// while it is still checking it: the ticket carries this list and
/// [`fold_chunk_digests`] over it has to reproduce the ticket's single value, which is what keeps
/// one collision-resistant authority rather than a list of independent ones.
///
/// Work is taken from a shared counter rather than sliced up front. The difference is measured:
/// with `ceil(n/threads)` slices, a 136 MiB image (34 chunks) took the same 6.2 ms on 12 threads
/// as on 16 — both land on 3 chunks of critical path — while the queue scales smoothly and the
/// pipelined park hands chunks over one at a time anyway.
#[must_use]
pub fn chunk_digests(bytes: &[u8], threads: usize) -> Vec<[u8; 32]> {
    let chunks: Vec<&[u8]> = bytes.chunks(DIGEST_CHUNK).collect();
    let n = chunks.len();
    let mut digests: Vec<[u8; 32]> = vec![[0u8; 32]; n];
    let threads = threads.max(1).min(n.max(1));
    if threads <= 1 || n <= 1 {
        for (d, c) in digests.iter_mut().zip(&chunks) {
            *d = Sha256::digest(c).into();
        }
        return digests;
    }
    let next = AtomicUsize::new(0);
    // One `Mutex` per worker's output would be a lock per chunk; instead each worker writes the
    // slots it claimed, and the claims are disjoint by construction.
    let out: Vec<Mutex<[u8; 32]>> = (0..n).map(|_| Mutex::new([0u8; 32])).collect();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let (next, out, chunks) = (&next, &out, &chunks);
            scope.spawn(move || loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= chunks.len() {
                    break;
                }
                let d: [u8; 32] = Sha256::digest(chunks[i]).into();
                *out[i].lock().unwrap_or_else(|e| e.into_inner()) = d;
            });
        }
    });
    for (d, slot) in digests.iter_mut().zip(&out) {
        *d = *slot.lock().unwrap_or_else(|e| e.into_inner());
    }
    digests
}

/// Fold a chunk-digest list into the image's single content digest.
///
/// The length and the chunk size are folded in so that a different split cannot produce the same
/// value.
#[must_use]
pub fn fold_chunk_digests(len: usize, digests: &[[u8; 32]]) -> [u8; 32] {
    let mut top = Sha256::new();
    top.update((len as u64).to_le_bytes());
    top.update((DIGEST_CHUNK as u64).to_le_bytes());
    for d in digests {
        top.update(d);
    }
    top.finalize().into()
}

/// Per-chunk digests, filled in any order and waited for in any order.
#[derive(Debug, Default)]
struct DigestSlots {
    slots: Mutex<Vec<Option<[u8; 32]>>>,
    ready: Condvar,
}

impl DigestSlots {
    fn with_capacity(n: usize) -> Self {
        DigestSlots {
            slots: Mutex::new(vec![None; n]),
            ready: Condvar::new(),
        }
    }

    fn put(&self, i: usize, d: [u8; 32]) {
        let mut g = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        g[i] = Some(d);
        drop(g);
        self.ready.notify_all();
    }

    /// Block until chunk `i` has been hashed.
    fn wait(&self, i: usize) -> [u8; 32] {
        let mut g = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(d) = g[i] {
                return d;
            }
            g = self.ready.wait(g).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Every digest, in order. The caller must already have waited for them.
    fn take(&self) -> Vec<[u8; 32]> {
        self.slots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|d| d.unwrap_or([0u8; 32]))
            .collect()
    }
}

/// Merge regions that are **adjacent on the device** into one run, in image order.
///
/// A session's KV pages are handed out by a block allocator, so a long context is very often a
/// contiguous span of the arena — an 8k session at `0..512` is one run. Copying it page by page
/// costs a device round trip per page, and workstream K8 measured that at 0.363 ms per 512 KiB
/// page: 511 of them were 82% of a park and 89% of a restore, which is not bandwidth, it is
/// per-call cost.
///
/// This is a **copy-plan** transform and nothing else. It preserves the image's byte order exactly
/// (a run is contiguous in the image because it is contiguous in the arena), so the chunk digests
/// are bit-identical either way, and the image's own `layout` stays **per page** — that is the
/// shape a restore is checked against, and a restore's pages are fragmented differently in general.
/// A fragmented destination simply coalesces into more runs than a clean one did.
pub fn coalesce(regions: &[Region]) -> Vec<Region> {
    let mut out: Vec<Region> = Vec::with_capacity(regions.len());
    for r in regions {
        if let Some(last) = out.last_mut() {
            let contiguous = last
                .addr
                .as_u64()
                .checked_add(last.len)
                .is_some_and(|end| end == r.addr.as_u64());
            // `checked_add` on the merged length too: a run that would overflow a u64 is left as
            // two runs rather than silently wrapping into a copy of the wrong size.
            if last.kind == r.kind && contiguous && last.len.checked_add(r.len).is_some() {
                last.len += r.len;
                continue;
            }
        }
        out.push(*r);
    }
    out
}

/// Where one `DIGEST_CHUNK` of the image comes from on the device: a chunk can straddle the
/// boundary between two regions, so it is copied as one or two pieces.
fn chunk_pieces(regions: &[Region], lo: u64, hi: u64) -> Result<Vec<Region>, ParkError> {
    let mut out = Vec::with_capacity(2);
    let mut base = 0u64;
    for region in regions {
        let end = base + region.len;
        let (a, b) = (lo.max(base), hi.min(end));
        if a < b {
            let addr = region.addr.offset(a - base).ok_or_else(|| {
                ParkError::Config(format!(
                    "region at {:#x} overflows at offset {}",
                    region.addr.as_u64(),
                    a - base
                ))
            })?;
            out.push(Region::new(region.kind, addr, b - a));
        }
        base = end;
    }
    Ok(out)
}

/// Exclusive end offset of each region within the image.
fn region_ends(regions: &[Region]) -> Vec<u64> {
    let mut out = Vec::with_capacity(regions.len());
    let mut at = 0u64;
    for r in regions {
        at += r.len;
        out.push(at);
    }
    out
}

/// Publish every region that is wholly below `filled`, starting at `next`.
///
/// Attribution by **image offset** rather than by device address: a coalesced copy writes one run
/// that spans many regions, and the copy proceeds in image order, so "everything below the current
/// offset has landed" is the whole rule. The old address-based search also cost `O(regions)` per
/// piece.
fn publish_upto(
    regions: &[Region],
    ends: &[u64],
    filled: u64,
    next: &mut usize,
    on_ready: &mut dyn FnMut(RegionReady),
) {
    while *next < regions.len() && ends[*next] <= filled {
        let r = regions[*next];
        on_ready(RegionReady {
            index: *next,
            kind: r.kind,
            addr: r.addr,
            len: r.len,
        });
        *next += 1;
    }
}

/// Copy `regions` out of the device into `slot`, hashing each 4 MiB chunk **as it lands**.
///
/// This is what turns a verified park from `copy + hash` into `max(copy, hash)`. The copy loop
/// never waits for a digest: it fills one chunk, hands it to the pool, and goes straight on to the
/// next. The pool is joined once at the end, by which time it has usually been finished for a
/// while — at the measured rates the hash is the faster half on this box.
///
/// Returns the per-chunk digests, which the ticket carries so that a restore can verify chunk by
/// chunk instead of all at once.
fn d2h_hashed(
    slot: &mut [u8],
    regions: &[Region],
    device: &mut dyn ParkDevice,
    verify: Verification,
) -> Result<Vec<[u8; 32]>, ParkError> {
    let total = total_bytes(regions) as usize;
    if slot.len() < total {
        return Err(ParkError::Config(format!(
            "slot is {} bytes for a {total} byte image",
            slot.len()
        )));
    }
    // The copy plan: contiguous pages as one run. The image is unchanged — same bytes, same order,
    // same digests — only the number of device calls differs.
    let runs = coalesce(regions);
    if !verify.enabled {
        // Nothing to overlap with. One copy per run, which is also the shape every non-HIP device
        // (the tests' fake VRAM) sees.
        let mut off = 0usize;
        for run in &runs {
            let len = run.len as usize;
            device.d2h(&mut slot[off..off + len], run)?;
            off += len;
        }
        return Ok(Vec::new());
    }
    let threads = verify.threads;
    // Reborrowed here, outside the scope, so each chunk carries the caller's lifetime rather than
    // the closure's: that is what lets a chunk be handed to a worker.
    let groups: Vec<&mut [u8]> = slot[..total].chunks_mut(COPY_CHUNK).collect();
    let n = total.div_ceil(DIGEST_CHUNK);
    let per_group = COPY_CHUNK / DIGEST_CHUNK;
    let slots = DigestSlots::with_capacity(n);
    let (tx, rx) = std::sync::mpsc::channel::<(usize, &[u8])>();
    let rx = Mutex::new(rx);
    let workers = threads.max(1).min(n.max(1));

    let copied = std::thread::scope(|scope| -> Result<(), ParkError> {
        for _ in 0..workers {
            let (rx, slots) = (&rx, &slots);
            scope.spawn(move || hash_worker(rx, slots));
        }
        let mut result = Ok(());
        for (gi, group) in groups.into_iter().enumerate() {
            let lo = (gi * COPY_CHUNK) as u64;
            let hi = lo + group.len() as u64;
            let mut filled = 0usize;
            for piece in match chunk_pieces(&runs, lo, hi) {
                Ok(p) => p,
                Err(e) => {
                    result = Err(e);
                    break;
                }
            } {
                let len = piece.len as usize;
                if let Err(e) = device.d2h(&mut group[filled..filled + len], &piece) {
                    result = Err(e);
                    break;
                }
                filled += len;
            }
            if result.is_err() {
                break;
            }
            // `&mut [u8]` -> `&[u8]`: the group is finished and the workers may read it while the
            // next group is still being copied into its own, disjoint, slice.
            let ro: &[u8] = group;
            for (k, sub) in ro.chunks(DIGEST_CHUNK).enumerate() {
                let _ = tx.send((gi * per_group + k, sub));
            }
        }
        // Dropping the sender is what tells the workers there is nothing more coming.
        drop(tx);
        result
    });
    copied?;
    Ok(slots.take())
}

/// Verify `slot` against `expect` chunk by chunk, sending each chunk to the device as soon as it
/// is known good.
///
/// The mirror of [`d2h_hashed`], and the same trade: the hash of chunk `i+1` runs while chunk `i`
/// crosses the link. What it costs is the strength of the I3 wording — a corrupt chunk part-way
/// through an image means the chunks **before** it have already been written into the destination
/// arena. What is preserved is the part that matters: a region is published to the caller only
/// once every byte of it has been verified, the restore fails loud, and the caller's contract is
/// that a failed restore hands those freshly-admitted allocations straight back without binding
/// them. A corrupt chunk 0 still reaches nothing at all.
fn h2d_verified(
    slot: &[u8],
    regions: &[Region],
    device: &mut dyn ParkDevice,
    verify: Verification,
    expect: &[[u8; 32]],
    who: (u64, u64),
    on_ready: &mut dyn FnMut(RegionReady),
) -> Result<(), ParkError> {
    let (id, session) = who;
    let total = total_bytes(regions) as usize;
    if slot.len() < total {
        return Err(ParkError::Config(format!(
            "slot is {} bytes for a {total} byte image",
            slot.len()
        )));
    }
    // The copy plan, as in `d2h_hashed`. `on_ready` still fires per REGION: a caller binds pages,
    // not runs, and a run that spans forty pages publishes forty of them.
    let runs = coalesce(regions);
    let ends = region_ends(regions);
    if !verify.enabled {
        let mut off = 0usize;
        let mut next = 0usize;
        for run in &runs {
            let len = run.len as usize;
            device.h2d(run, &slot[off..off + len])?;
            off += len;
            publish_upto(regions, &ends, off as u64, &mut next, on_ready);
        }
        return Ok(());
    }
    let threads = verify.threads;
    let chunks: Vec<&[u8]> = slot[..total].chunks(DIGEST_CHUNK).collect();
    let n = chunks.len();
    let per_group = COPY_CHUNK / DIGEST_CHUNK;
    if expect.len() != n {
        return Err(ParkError::LayoutMismatch {
            id,
            detail: format!(
                "the image carries {} chunk digests, its bytes are {n} chunks",
                expect.len()
            ),
        });
    }
    let slots = DigestSlots::with_capacity(n);
    let (tx, rx) = std::sync::mpsc::channel::<(usize, &[u8])>();
    let rx = Mutex::new(rx);
    let workers = threads.max(1).min(n.max(1));

    std::thread::scope(|scope| -> Result<(), ParkError> {
        for _ in 0..workers {
            let (rx, slots) = (&rx, &slots);
            scope.spawn(move || hash_worker(rx, slots));
        }
        // Everything is already in host memory, so every chunk can be queued at once; the workers
        // take them in order and the sender below only ever waits for the one it needs next.
        for (ci, chunk) in chunks.iter().enumerate() {
            let _ = tx.send((ci, chunk));
        }
        drop(tx);

        // The next region that has not been published yet. Regions are published in image order
        // and exactly once, because the copy proceeds in image order: everything below the current
        // image offset has landed.
        let mut next = 0usize;
        // Verified one digest chunk at a time, sent one copy chunk at a time: the copy's
        // granularity is the device's (a round trip and a DMA ramp per piece) and the hash's is
        // the digest's, and there is no reason for them to be the same number.
        for gi in 0..n.div_ceil(per_group) {
            let first = gi * per_group;
            let last = (first + per_group).min(n);
            for (ci, want) in expect.iter().enumerate().take(last).skip(first) {
                let got = slots.wait(ci);
                if got != *want {
                    return Err(ParkError::ImageCorrupt {
                        id,
                        session,
                        expected: hex::encode(want),
                        found: hex::encode(got),
                    });
                }
            }
            let lo = (first * DIGEST_CHUNK) as u64;
            let hi = lo
                + chunks[first..last]
                    .iter()
                    .map(|c| c.len() as u64)
                    .sum::<u64>();
            let mut filled = lo as usize;
            for piece in chunk_pieces(&runs, lo, hi)? {
                let len = piece.len as usize;
                device.h2d(&piece, &slot[filled..filled + len])?;
                filled += len;
                publish_upto(regions, &ends, filled as u64, &mut next, on_ready);
            }
        }
        Ok(())
    })
}

/// One hashing worker: take chunks off the shared queue until it closes.
fn hash_worker(rx: &Mutex<std::sync::mpsc::Receiver<(usize, &[u8])>>, slots: &DigestSlots) {
    loop {
        // The lock is held only for the `recv`: at 4 MiB a chunk the queue is touched a few
        // hundred times per image while each hash is well over a millisecond of work.
        let job = {
            let g = rx.lock().unwrap_or_else(|e| e.into_inner());
            g.recv()
        };
        match job {
            Ok((i, bytes)) => slots.put(i, Sha256::digest(bytes).into()),
            Err(_) => break,
        }
    }
}

/// How an image is hashed, and whether it is.
///
/// Verification is **on** by default and off is a benchmark mode, exactly as it is for the pack
/// stream: I3 is why the digest is on the ticket. The flag exists so that the cost of the guarantee
/// can be *measured* rather than argued about — `saber park-bench` reports both rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Verification {
    /// Hash on park and check on restore.
    pub enabled: bool,
    /// Threads the chunked digest may use.
    pub threads: usize,
}

impl Default for Verification {
    fn default() -> Self {
        let par = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(4);
        // Three quarters of the logical cores, and both ends of that are measured
        // (`docs/LEDGER.md`, workstream K6, on 24 logical / 12 physical):
        //
        // * the clamp that used to sit here at **8** was costing a factor of two — a 136 MiB image
        //   hashes at 17.1 GB/s on 8 threads and 29.4 on 16, an 852 MiB one at 20.7 vs 34.9;
        // * **all** of them is worse than most of them, because the overlapped park has a producer
        //   thread issuing the copies and a HIP thread issuing them to the device, and neither can
        //   do that from a run queue: at 852 MiB the verified park is 39.96 ms on 16 threads and
        //   38.79 on 24, but its restore goes from 37.63 ms to 47.24 and its p90 from 38.7 to 47.9.
        //
        // A park is a rare, millisecond-scale event that is by definition taking a session out of
        // the engine, so it is worth most of the machine — just not the part of it that is doing
        // the copying.
        Verification {
            enabled: true,
            threads: par.saturating_sub(par / 4).max(1),
        }
    }
}

impl Verification {
    /// Hashing with `threads` workers.
    #[must_use]
    pub fn on(threads: usize) -> Self {
        Verification {
            enabled: true,
            threads: threads.max(1),
        }
    }

    /// No hashing. A benchmark mode: a corrupt image would then reach VRAM.
    #[must_use]
    pub fn off() -> Self {
        Verification {
            enabled: false,
            threads: 1,
        }
    }

    /// The digest of `bytes` under this policy — all zeroes when verification is off, which is
    /// what a ticket then carries and compares against.
    fn digest(self, bytes: &[u8]) -> [u8; 32] {
        if self.enabled {
            image_digest(bytes, self.threads)
        } else {
            [0u8; 32]
        }
    }
}

/// A session image captured out of device memory: what it is, and what verifies it.
///
/// This is what [`capture_into`] hands back and what [`publish_from`] needs to put the bytes back.
/// It carries no tier and no ledger: an engine that wants to park a sequence has device state and a
/// buffer, and everything else — which pool the bytes end up in, who bills them — is the
/// scheduler's and the tier's business.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageDigests {
    /// The regions the image was taken from, in image order. A restore's destinations must match
    /// it kind for kind and length for length.
    pub layout: Vec<RegionSpec>,
    /// Payload bytes.
    pub bytes: u64,
    /// Bytes one chunk digest covers ([`DIGEST_CHUNK`]), or 0 when verification was off.
    pub chunk_bytes: u32,
    /// Per-chunk `sha256`, in payload order. Empty when verification was off.
    pub chunks: Vec<[u8; 32]>,
    /// [`fold_chunk_digests`] over `chunks` — the image's single content digest, all zero when
    /// verification was off.
    pub sha256: [u8; 32],
}

impl ImageDigests {
    /// Whether this image is verified at all.
    #[must_use]
    pub fn verified(&self) -> bool {
        !self.chunks.is_empty()
    }

    /// Check the chunk list against the folded value.
    ///
    /// The same rule the parking ticket holds: the per-chunk digests are only trusted because
    /// folding them reproduces the one value, so an image whose list was replaced is refused
    /// before a byte is issued to the device.
    pub fn check_self(&self) -> Result<(), ParkError> {
        if !self.verified() {
            return Ok(());
        }
        let folded = fold_chunk_digests(self.bytes as usize, &self.chunks);
        if folded != self.sha256 {
            return Err(ParkError::ImageCorrupt {
                id: 0,
                session: 0,
                expected: hex::encode(self.sha256),
                found: hex::encode(folded),
            });
        }
        Ok(())
    }
}

/// Copy a session's regions out of device memory into `slot`, hashing each chunk as it lands.
///
/// The public form of the overlapped park (`docs/LEDGER.md`, workstream K6): the copy never waits
/// for a digest, and at the measured rates the hash is the faster half. `slot` must be at least the
/// regions' total length and should be page-locked, or the copy is a staged one.
///
/// Returns what a later [`publish_from`] needs. Nothing here touches a ledger or a tier — the
/// caller decides where the bytes go next.
pub fn capture_into(
    slot: &mut [u8],
    regions: &[Region],
    device: &mut dyn ParkDevice,
    verify: Verification,
) -> Result<ImageDigests, ParkError> {
    let bytes = total_bytes(regions);
    let chunks = d2h_hashed(slot, regions, device, verify)?;
    Ok(ImageDigests {
        layout: layout_of(regions),
        bytes,
        chunk_bytes: if verify.enabled {
            DIGEST_CHUNK as u32
        } else {
            0
        },
        sha256: fold_if(verify, bytes as usize, &chunks),
        chunks,
    })
}

/// Put a captured image back, verifying each chunk **before** it is issued to the device.
///
/// `dst` may be a different set of addresses from the one [`capture_into`] read — that is the whole
/// point of parking — but it must match the captured layout kind for kind and length for length,
/// which is checked here. `on_ready` fires per region, once every byte of it has landed, so a
/// caller can bind a region before the rest of the image arrives.
///
/// What this does and does not guarantee is the chunk-at-a-time trade recorded in the K6 ledger
/// entry: no region is published unless every byte of it was verified, a corrupt first chunk
/// reaches nothing at all, and a corrupt chunk part-way through leaves the chunks before it already
/// in the destination — the restore fails loud and the caller's contract is to hand those
/// allocations back without binding them.
pub fn publish_from(
    slot: &[u8],
    dst: &[Region],
    device: &mut dyn ParkDevice,
    verify: Verification,
    image: &ImageDigests,
    on_ready: &mut dyn FnMut(RegionReady),
) -> Result<(), ParkError> {
    if dst.len() != image.layout.len() {
        return Err(ParkError::LayoutMismatch {
            id: 0,
            detail: format!(
                "{} destination region(s) for an image of {}",
                dst.len(),
                image.layout.len()
            ),
        });
    }
    for (i, (spec, region)) in image.layout.iter().zip(dst).enumerate() {
        if spec.kind != region.kind || spec.len != region.len {
            return Err(ParkError::LayoutMismatch {
                id: 0,
                detail: format!(
                    "region {i} is {} of {} bytes; the image has {} of {}",
                    region.kind, region.len, spec.kind, spec.len
                ),
            });
        }
    }
    image.check_self()?;
    let verify = if image.verified() {
        verify
    } else {
        Verification::off()
    };
    h2d_verified(slot, dst, device, verify, &image.chunks, (0, 0), on_ready)
}

/// The chunk digests of `bytes` under `verify`, or an empty list when it is off.
fn chunks_if(verify: Verification, bytes: &[u8]) -> Vec<[u8; 32]> {
    if verify.enabled {
        chunk_digests(bytes, verify.threads)
    } else {
        Vec::new()
    }
}

/// Fold a chunk list under `verify`, or the all-zero digest when it is off — which is exactly what
/// a ticket then carries and compares against.
fn fold_if(verify: Verification, len: usize, chunks: &[[u8; 32]]) -> [u8; 32] {
    if verify.enabled {
        fold_chunk_digests(len, chunks)
    } else {
        [0u8; 32]
    }
}

/// The ticket's chunk list must fold to the ticket's digest, and must have one entry per chunk.
///
/// This is what keeps the folded value the single authority: the per-chunk digests are only
/// trusted because they reproduce it, so a list that was corrupted alongside the image does not
/// let the image through.
fn check_chunk_list(ticket: &ParkTicket, bytes: usize) -> Result<(), ParkError> {
    if ticket.sha256 == [0u8; 32] && ticket.chunks.is_empty() {
        // Verification was off when this was parked; there is nothing to check against.
        return Ok(());
    }
    let want = bytes.div_ceil(DIGEST_CHUNK);
    if ticket.chunks.len() != want {
        return Err(ParkError::LayoutMismatch {
            id: ticket.id,
            detail: format!(
                "ticket carries {} chunk digests for a {bytes} byte image, which is {want} chunks",
                ticket.chunks.len()
            ),
        });
    }
    let folded = fold_chunk_digests(bytes, &ticket.chunks);
    if folded != ticket.sha256 {
        return Err(ParkError::ImageCorrupt {
            id: ticket.id,
            session: ticket.session,
            expected: hex::encode(ticket.sha256),
            found: hex::encode(folded),
        });
    }
    Ok(())
}

/// Check a byte restore's destination against what was parked.
fn check_bytes_len(ticket: &ParkTicket, dst: &[u8]) -> Result<(), ParkError> {
    if dst.len() as u64 != ticket.bytes {
        return Err(ParkError::LayoutMismatch {
            id: ticket.id,
            detail: format!(
                "restore buffer is {} bytes; the image is {}",
                dst.len(),
                ticket.bytes
            ),
        });
    }
    Ok(())
}

/// Check a restore's destination list against what was parked.
fn check_layout(ticket: &ParkTicket, dst: &[Region]) -> Result<(), ParkError> {
    if dst.len() != ticket.layout.len() {
        return Err(ParkError::LayoutMismatch {
            id: ticket.id,
            detail: format!(
                "{} destination region(s) for an image of {}",
                dst.len(),
                ticket.layout.len()
            ),
        });
    }
    for (i, (spec, region)) in ticket.layout.iter().zip(dst).enumerate() {
        if spec.kind != region.kind || spec.len != region.len {
            return Err(ParkError::LayoutMismatch {
                id: ticket.id,
                detail: format!(
                    "region {i} was parked as {} of {} bytes, restore names {} of {}",
                    spec.kind, spec.len, region.kind, region.len
                ),
            });
        }
    }
    Ok(())
}

fn layout_of(regions: &[Region]) -> Vec<RegionSpec> {
    regions
        .iter()
        .map(|r| RegionSpec {
            kind: r.kind,
            len: r.len,
        })
        .collect()
}

fn total_bytes(regions: &[Region]) -> u64 {
    regions.iter().map(|r| r.len).sum()
}

// ---------------------------------------------------------------------------------------------
// The RAM tier
// ---------------------------------------------------------------------------------------------

/// One parked image in a pinned slot.
#[derive(Debug)]
struct Resident {
    slot: usize,
    bytes: u64,
}

/// The first parking tier: a fixed pool of **pinned** RAM slots, allocated once.
///
/// Pinned rather than pageable for one reason that decides the whole design: the park is a DMA out
/// of VRAM and the restore is a DMA back in, and pinned memory is what makes both a DMA instead of
/// a staged copy. That is also why the tier bills [`Pool::HostPinned`] — "every pinned byte is a
/// byte the kernel cannot reclaim", so the ledger's capacity for that pool IS this tier's size.
#[derive(Debug)]
pub struct PinnedHostTier {
    slab: Box<dyn Slab>,
    slots: usize,
    free: Vec<usize>,
    images: BTreeMap<u64, Resident>,
    slot_bytes: u64,
    next_id: u64,
    verify: Verification,
    stats: TierStats,
}

impl PinnedHostTier {
    /// Split `slab` into `slots` slots of `slot_bytes` and hold them for the process.
    ///
    /// The slab is owned, not borrowed: a tier that lives in the scheduler's `Box<dyn Parkable>`
    /// cannot carry a lifetime, and a pool that is allocated once has nothing to borrow from.
    pub fn new(
        slab: Box<dyn Slab>,
        slots: usize,
        slot_bytes: u64,
    ) -> Result<PinnedHostTier, ParkError> {
        if slots == 0 {
            return Err(ParkError::Config(
                "a park tier needs at least one slot".into(),
            ));
        }
        // O_DIRECT reads and writes the *demoted* copy straight out of these slots, so they have to
        // be 4096-aligned and a multiple of 4096 — refused here rather than discovered as EINVAL on
        // the first spill.
        if slot_bytes == 0 || slot_bytes % DIO_ALIGN as u64 != 0 {
            return Err(ParkError::Config(format!(
                "slot_bytes {slot_bytes} must be a non-zero multiple of {DIO_ALIGN}: a demoted \
                 image is written to NVMe straight out of its slot"
            )));
        }
        if slab.addr() % DIO_ALIGN != 0 {
            return Err(ParkError::Config(format!(
                "the slab starts at {:#x}, which is not {DIO_ALIGN}-aligned",
                slab.addr()
            )));
        }
        if (slab.len() as u64) < slots as u64 * slot_bytes {
            return Err(ParkError::Config(format!(
                "slab is {} bytes; {slots} slots of {slot_bytes} need {}",
                slab.len(),
                slots as u64 * slot_bytes
            )));
        }
        Ok(PinnedHostTier {
            slab,
            slots,
            free: (0..slots).rev().collect(),
            images: BTreeMap::new(),
            slot_bytes,
            next_id: 1,
            verify: Verification::default(),
            stats: TierStats::default(),
        })
    }

    /// The bytes of one slot.
    fn slot(&self, slot: usize) -> &[u8] {
        let lo = slot * self.slot_bytes as usize;
        &self.slab.as_slice()[lo..lo + self.slot_bytes as usize]
    }

    /// The bytes of one slot, mutably.
    fn slot_mut(&mut self, slot: usize) -> &mut [u8] {
        let lo = slot * self.slot_bytes as usize;
        let hi = lo + self.slot_bytes as usize;
        &mut self.slab.as_mut_slice()[lo..hi]
    }

    /// Set the verification policy. On by default; off is a benchmark mode.
    #[must_use]
    pub fn verifying(mut self, verify: Verification) -> Self {
        self.verify = verify;
        self
    }

    /// Host bytes this tier holds for the life of the process.
    #[must_use]
    pub fn pinned_bytes(&self) -> u64 {
        self.slots as u64 * self.slot_bytes
    }

    /// Flip one bit of a parked image's bytes, so a negative control can prove the restore
    /// refuses it.
    ///
    /// Tests only — nothing in the engine calls this, and it is here for the same reason
    /// `pack::writer::WriterFaults` is: "a corrupt image is refused" is a claim, and a claim needs
    /// a control that fails. Returns false if the ticket names nothing this tier holds.
    #[doc(hidden)]
    pub fn corrupt_for_test(&mut self, ticket: &ParkTicket, offset: usize) -> bool {
        let Some(res) = self.images.get(&ticket.id) else {
            return false;
        };
        if offset >= res.bytes as usize {
            return false;
        }
        let slot = res.slot;
        self.slot_mut(slot)[offset] ^= 0x01;
        true
    }

    /// The image's bytes, for a demotion that writes them somewhere else.
    fn image_bytes(&self, id: u64) -> Result<&[u8], ParkError> {
        let res = self.images.get(&id).ok_or(ParkError::NoSuchImage {
            pool: Pool::HostPinned,
            id,
        })?;
        Ok(&self.slot(res.slot)[..res.bytes as usize])
    }

    fn take_slot(&mut self) -> Result<usize, ParkError> {
        self.free.pop().ok_or(ParkError::TierFull {
            pool: Pool::HostPinned,
            slots: self.slots,
        })
    }

    /// Store `bytes` that are already in hand (a promotion from NVMe), under a fresh id.
    fn adopt(&mut self, source: &[u8], ledger: &mut Ledger, name: &str) -> Result<u64, ParkError> {
        let bytes = source.len() as u64;
        if bytes > self.slot_bytes {
            return Err(ParkError::ImageTooLarge {
                pool: Pool::HostPinned,
                bytes,
                slot_bytes: self.slot_bytes,
            });
        }
        ledger.reserve(Pool::HostPinned, name, bytes)?;
        let slot = match self.take_slot() {
            Ok(s) => s,
            Err(e) => {
                let _ = ledger.release(Pool::HostPinned, name);
                return Err(e);
            }
        };
        self.slot_mut(slot)[..source.len()].copy_from_slice(source);
        let id = self.next_id;
        self.next_id += 1;
        self.images.insert(id, Resident { slot, bytes });
        self.stats.bytes_in += bytes;
        Ok(id)
    }

    fn evict(&mut self, id: u64) -> Option<Resident> {
        let res = self.images.remove(&id)?;
        self.free.push(res.slot);
        Some(res)
    }
}

impl ParkTier for PinnedHostTier {
    fn pool(&self) -> Pool {
        Pool::HostPinned
    }

    fn slots(&self) -> usize {
        self.slots
    }

    fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }

    fn len(&self) -> usize {
        self.images.len()
    }

    fn park(
        &mut self,
        session: u64,
        regions: &[Region],
        device: &mut dyn ParkDevice,
        ledger: &mut Ledger,
    ) -> Result<ParkTicket, ParkError> {
        if regions.is_empty() {
            return Err(ParkError::Config("a session image has no regions".into()));
        }
        let bytes = total_bytes(regions);
        if bytes > self.slot_bytes {
            return Err(ParkError::ImageTooLarge {
                pool: Pool::HostPinned,
                bytes,
                slot_bytes: self.slot_bytes,
            });
        }
        let id = self.next_id;
        let name = format!("session:{session}");
        // Billed before it is stored: a full pool refuses instead of over-committing.
        ledger.reserve(Pool::HostPinned, &name, bytes)?;
        let slot = match self.take_slot() {
            Ok(s) => s,
            Err(e) => {
                let _ = ledger.release(Pool::HostPinned, &name);
                return Err(e);
            }
        };

        let verify = self.verify;
        let off = bytes as usize;
        let chunks = match d2h_hashed(self.slot_mut(slot), regions, device, verify) {
            Ok(c) => c,
            Err(e) => {
                // Put the slot and the reservation back: a failed park must leave the tier exactly
                // as it was, or the next one is refused for bytes nobody holds.
                self.free.push(slot);
                let _ = ledger.release(Pool::HostPinned, &name);
                return Err(e);
            }
        };
        let sha = fold_if(verify, off, &chunks);
        // Parking is write traffic even into RAM; on NVMe it is the endurance number.
        ledger.record_written(Pool::HostPinned, bytes)?;

        self.next_id += 1;
        self.images.insert(id, Resident { slot, bytes });
        self.stats.parks += 1;
        self.stats.bytes_in += bytes;
        Ok(ParkTicket {
            id,
            session,
            pool: Pool::HostPinned,
            bytes,
            layout: layout_of(regions),
            sha256: sha,
            chunks,
        })
    }

    fn restore(
        &mut self,
        ticket: &ParkTicket,
        dst: &[Region],
        device: &mut dyn ParkDevice,
        ledger: &mut Ledger,
        on_ready: &mut dyn FnMut(RegionReady),
    ) -> Result<(), ParkError> {
        check_layout(ticket, dst)?;
        let res = self.images.get(&ticket.id).ok_or(ParkError::NoSuchImage {
            pool: Pool::HostPinned,
            id: ticket.id,
        })?;
        let slot = res.slot;
        let bytes = res.bytes as usize;
        let verify = self.verify;
        check_chunk_list(ticket, bytes)?;
        let sent = h2d_verified(
            &self.slot(slot)[..bytes],
            dst,
            device,
            verify,
            &ticket.chunks,
            (ticket.id, ticket.session),
            on_ready,
        );
        if let Err(e) = sent {
            if matches!(e, ParkError::ImageCorrupt { .. }) {
                self.stats.corrupt_refusals += 1;
            }
            return Err(e);
        }

        self.evict(ticket.id);
        ledger.release(Pool::HostPinned, &ticket.reservation())?;
        self.stats.restores += 1;
        self.stats.bytes_out += ticket.bytes;
        Ok(())
    }

    fn park_bytes(
        &mut self,
        session: u64,
        bytes: &[u8],
        ledger: &mut Ledger,
    ) -> Result<ParkTicket, ParkError> {
        let len = bytes.len() as u64;
        if len == 0 {
            return Err(ParkError::Config("a session image of 0 bytes".into()));
        }
        if len > self.slot_bytes {
            return Err(ParkError::ImageTooLarge {
                pool: Pool::HostPinned,
                bytes: len,
                slot_bytes: self.slot_bytes,
            });
        }
        let name = format!("session:{session}");
        let id = self.next_id;
        // Billed before it is stored, and the slot taken only once the bytes are accounted for.
        ledger.reserve(Pool::HostPinned, &name, len)?;
        let slot = match self.take_slot() {
            Ok(s) => s,
            Err(e) => {
                let _ = ledger.release(Pool::HostPinned, &name);
                return Err(e);
            }
        };
        self.slot_mut(slot)[..bytes.len()].copy_from_slice(bytes);
        let verify = self.verify;
        let chunks = chunks_if(verify, &self.slot(slot)[..bytes.len()]);
        let sha = fold_if(verify, bytes.len(), &chunks);
        ledger.record_written(Pool::HostPinned, len)?;
        self.next_id += 1;
        self.images.insert(id, Resident { slot, bytes: len });
        self.stats.parks += 1;
        self.stats.bytes_in += len;
        Ok(ParkTicket {
            id,
            session,
            pool: Pool::HostPinned,
            bytes: len,
            layout: Vec::new(),
            sha256: sha,
            chunks,
        })
    }

    fn restore_bytes(
        &mut self,
        ticket: &ParkTicket,
        dst: &mut [u8],
        ledger: &mut Ledger,
    ) -> Result<(), ParkError> {
        check_bytes_len(ticket, dst)?;
        let res = self.images.get(&ticket.id).ok_or(ParkError::NoSuchImage {
            pool: Pool::HostPinned,
            id: ticket.id,
        })?;
        let (slot, bytes) = (res.slot, res.bytes as usize);
        let sha = self.verify.digest(&self.slot(slot)[..bytes]);
        if sha != ticket.sha256 {
            self.stats.corrupt_refusals += 1;
            return Err(ParkError::ImageCorrupt {
                id: ticket.id,
                session: ticket.session,
                expected: hex::encode(ticket.sha256),
                found: hex::encode(sha),
            });
        }
        dst.copy_from_slice(&self.slot(slot)[..bytes]);
        self.evict(ticket.id);
        ledger.release(Pool::HostPinned, &ticket.reservation())?;
        self.stats.restores += 1;
        self.stats.bytes_out += ticket.bytes;
        Ok(())
    }

    fn discard(&mut self, ticket: &ParkTicket, ledger: &mut Ledger) {
        if self.evict(ticket.id).is_some() {
            self.stats.discards += 1;
            if let Err(e) = ledger.release(Pool::HostPinned, &ticket.reservation()) {
                tracing::error!(id = ticket.id, error = %e, "park tier release failed");
            }
        }
    }

    fn stats(&self) -> TierStats {
        self.stats
    }
}

// ---------------------------------------------------------------------------------------------
// The NVMe tier
// ---------------------------------------------------------------------------------------------

/// One parked image in a file slot.
#[derive(Debug)]
struct Stored {
    slot: usize,
    bytes: u64,
    /// Bytes actually written, padded up to 4096 for `O_DIRECT`.
    padded: u64,
}

/// The spill tier: one preallocated `O_DIRECT` file with a fixed slot table.
///
/// The architecture's estimate is ~12–20 ms per restore from this drive, which is what makes NVMe
/// *a second parking tier* rather than a cold store. One file, `posix_fallocate`d at construction
/// so the slots are contiguous, and one staging slot of pinned memory for the VRAM↔file hop: a
/// park is D2H → `pwritev`, a restore is `preadv` → hash → H2D.
#[derive(Debug)]
pub struct NvmeTier {
    path: PathBuf,
    file: File,
    staging: Box<dyn Slab>,
    slot_bytes: u64,
    free: Vec<usize>,
    images: BTreeMap<u64, Stored>,
    next_id: u64,
    verify: Verification,
    stats: TierStats,
    delete_on_drop: bool,
}

impl NvmeTier {
    /// Create (or re-create) the parking file and split it into `slots` slots.
    ///
    /// `staging` is one pinned slot of at least `slot_bytes`: every byte crossing VRAM↔file goes
    /// through it, so it is a DMA on one side and an `O_DIRECT` transfer on the other.
    pub fn create(
        path: &Path,
        slots: usize,
        slot_bytes: u64,
        staging: Box<dyn Slab>,
        mode: OpenMode,
    ) -> Result<NvmeTier, ParkError> {
        if slots == 0 {
            return Err(ParkError::Config(
                "a park tier needs at least one slot".into(),
            ));
        }
        if slot_bytes == 0 || slot_bytes % DIO_ALIGN as u64 != 0 {
            return Err(ParkError::Config(format!(
                "slot_bytes {slot_bytes} must be a non-zero multiple of {DIO_ALIGN} for O_DIRECT"
            )));
        }
        if (staging.len() as u64) < slot_bytes {
            return Err(ParkError::Config(format!(
                "staging slot is {} bytes; the tier's slots are {slot_bytes}",
                staging.len()
            )));
        }
        if mode == OpenMode::Direct && staging.addr() % DIO_ALIGN != 0 {
            return Err(ParkError::Config(
                "the staging slot is not 4096-aligned; O_DIRECT cannot use it".into(),
            ));
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)
                .map_err(|e| ParkError::io(path, "create the parking directory", e))?;
        }
        let file = dio::open_rw_create(path, mode)
            .map_err(|e| ParkError::io(path, "open O_DIRECT read/write", e))?;
        let len = slots as u64 * slot_bytes;
        dio::fallocate(&file, len).map_err(|e| ParkError::io(path, "posix_fallocate", e))?;
        file.set_len(len)
            .map_err(|e| ParkError::io(path, "set_len", e))?;
        Ok(NvmeTier {
            path: path.to_path_buf(),
            file,
            staging,
            slot_bytes,
            free: (0..slots).rev().collect(),
            images: BTreeMap::new(),
            next_id: 1,
            verify: Verification::default(),
            stats: TierStats::default(),
            delete_on_drop: false,
        })
    }

    /// Set the verification policy. On by default; off is a benchmark mode.
    #[must_use]
    pub fn verifying(mut self, verify: Verification) -> Self {
        self.verify = verify;
        self
    }

    /// Delete the parking file when this tier is dropped. For a benchmark, not for a service.
    #[must_use]
    pub fn deleting_on_drop(mut self) -> Self {
        self.delete_on_drop = true;
        self
    }

    /// The file backing the tier.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Bytes the file holds.
    #[must_use]
    pub fn file_bytes(&self) -> u64 {
        (self.free.len() + self.images.len()) as u64 * self.slot_bytes
    }

    fn take_slot(&mut self) -> Result<usize, ParkError> {
        self.free.pop().ok_or(ParkError::TierFull {
            pool: Pool::Nvme,
            slots: self.free.len() + self.images.len(),
        })
    }

    /// Write the first `bytes` of the staging slot into `slot`, padding to 4096.
    fn write_slot(&mut self, slot: usize, bytes: u64) -> Result<u64, ParkError> {
        let padded = align_up(bytes);
        // Zero the padding rather than writing whatever the slot held: one session's tail must not
        // be readable in another's image.
        let pad_from = bytes as usize;
        let pad_to = padded as usize;
        self.staging.as_mut_slice()[pad_from..pad_to].fill(0);
        let off = slot as u64 * self.slot_bytes;
        dio::pwritev_exact(&self.file, &self.staging.as_slice()[..pad_to], off)
            .map_err(|e| ParkError::io(&self.path, format!("write slot {slot}"), e))?;
        Ok(padded)
    }

    fn read_slot(&mut self, slot: usize, padded: u64) -> Result<(), ParkError> {
        let off = slot as u64 * self.slot_bytes;
        let n = padded as usize;
        dio::preadv_exact(&self.file, &mut self.staging.as_mut_slice()[..n], off)
            .map_err(|e| ParkError::io(&self.path, format!("read slot {slot}"), e))?;
        Ok(())
    }

    /// Store bytes already in hand (a demotion from RAM), under a fresh id.
    fn adopt(&mut self, source: &[u8], ledger: &mut Ledger, name: &str) -> Result<u64, ParkError> {
        let bytes = source.len() as u64;
        if bytes > self.slot_bytes {
            return Err(ParkError::ImageTooLarge {
                pool: Pool::Nvme,
                bytes,
                slot_bytes: self.slot_bytes,
            });
        }
        ledger.reserve(Pool::Nvme, name, bytes)?;
        let slot = match self.take_slot() {
            Ok(s) => s,
            Err(e) => {
                let _ = ledger.release(Pool::Nvme, name);
                return Err(e);
            }
        };
        self.staging.as_mut_slice()[..source.len()].copy_from_slice(source);
        let padded = match self.write_slot(slot, bytes) {
            Ok(p) => p,
            Err(e) => {
                self.free.push(slot);
                let _ = ledger.release(Pool::Nvme, name);
                return Err(e);
            }
        };
        // The endurance number: padded bytes, because that is what the drive wrote.
        ledger.record_written(Pool::Nvme, padded)?;
        let id = self.next_id;
        self.next_id += 1;
        self.images.insert(
            id,
            Stored {
                slot,
                bytes,
                padded,
            },
        );
        self.stats.bytes_in += bytes;
        Ok(id)
    }

    fn evict(&mut self, id: u64) -> Option<Stored> {
        let st = self.images.remove(&id)?;
        self.free.push(st.slot);
        Some(st)
    }
}

impl Drop for NvmeTier {
    fn drop(&mut self) {
        if self.delete_on_drop {
            if let Err(e) = std::fs::remove_file(&self.path) {
                tracing::warn!(path = %self.path.display(), error = %e, "could not delete the parking file");
            }
        }
    }
}

impl ParkTier for NvmeTier {
    fn pool(&self) -> Pool {
        Pool::Nvme
    }

    fn slots(&self) -> usize {
        self.free.len() + self.images.len()
    }

    fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }

    fn len(&self) -> usize {
        self.images.len()
    }

    fn park(
        &mut self,
        session: u64,
        regions: &[Region],
        device: &mut dyn ParkDevice,
        ledger: &mut Ledger,
    ) -> Result<ParkTicket, ParkError> {
        if regions.is_empty() {
            return Err(ParkError::Config("a session image has no regions".into()));
        }
        let bytes = total_bytes(regions);
        if bytes > self.slot_bytes {
            return Err(ParkError::ImageTooLarge {
                pool: Pool::Nvme,
                bytes,
                slot_bytes: self.slot_bytes,
            });
        }
        let name = format!("session:{session}");
        ledger.reserve(Pool::Nvme, &name, bytes)?;
        let slot = match self.take_slot() {
            Ok(s) => s,
            Err(e) => {
                let _ = ledger.release(Pool::Nvme, &name);
                return Err(e);
            }
        };

        let verify = self.verify;
        let off = bytes as usize;
        let chunks = match d2h_hashed(self.staging.as_mut_slice(), regions, device, verify) {
            Ok(c) => c,
            Err(e) => {
                self.free.push(slot);
                let _ = ledger.release(Pool::Nvme, &name);
                return Err(e);
            }
        };
        let sha = fold_if(verify, off, &chunks);
        let padded = match self.write_slot(slot, bytes) {
            Ok(p) => p,
            Err(e) => {
                self.free.push(slot);
                let _ = ledger.release(Pool::Nvme, &name);
                return Err(e);
            }
        };
        ledger.record_written(Pool::Nvme, padded)?;

        let id = self.next_id;
        self.next_id += 1;
        self.images.insert(
            id,
            Stored {
                slot,
                bytes,
                padded,
            },
        );
        self.stats.parks += 1;
        self.stats.bytes_in += bytes;
        Ok(ParkTicket {
            id,
            session,
            pool: Pool::Nvme,
            bytes,
            layout: layout_of(regions),
            sha256: sha,
            chunks,
        })
    }

    fn restore(
        &mut self,
        ticket: &ParkTicket,
        dst: &[Region],
        device: &mut dyn ParkDevice,
        ledger: &mut Ledger,
        on_ready: &mut dyn FnMut(RegionReady),
    ) -> Result<(), ParkError> {
        check_layout(ticket, dst)?;
        let st = self.images.get(&ticket.id).ok_or(ParkError::NoSuchImage {
            pool: Pool::Nvme,
            id: ticket.id,
        })?;
        let (slot, bytes, padded) = (st.slot, st.bytes as usize, st.padded);
        self.read_slot(slot, padded)?;
        let verify = self.verify;
        check_chunk_list(ticket, bytes)?;
        // Verified out of the staging slot the drive just filled, chunk by chunk, each one before
        // it is issued to the device.
        let sent = h2d_verified(
            &self.staging.as_slice()[..bytes],
            dst,
            device,
            verify,
            &ticket.chunks,
            (ticket.id, ticket.session),
            on_ready,
        );
        if let Err(e) = sent {
            if matches!(e, ParkError::ImageCorrupt { .. }) {
                self.stats.corrupt_refusals += 1;
            }
            return Err(e);
        }

        self.evict(ticket.id);
        ledger.release(Pool::Nvme, &ticket.reservation())?;
        self.stats.restores += 1;
        self.stats.bytes_out += ticket.bytes;
        Ok(())
    }

    fn park_bytes(
        &mut self,
        session: u64,
        bytes: &[u8],
        ledger: &mut Ledger,
    ) -> Result<ParkTicket, ParkError> {
        let len = bytes.len() as u64;
        if len == 0 {
            return Err(ParkError::Config("a session image of 0 bytes".into()));
        }
        if len > self.slot_bytes {
            return Err(ParkError::ImageTooLarge {
                pool: Pool::Nvme,
                bytes: len,
                slot_bytes: self.slot_bytes,
            });
        }
        let name = format!("session:{session}");
        ledger.reserve(Pool::Nvme, &name, len)?;
        let slot = match self.take_slot() {
            Ok(s) => s,
            Err(e) => {
                let _ = ledger.release(Pool::Nvme, &name);
                return Err(e);
            }
        };
        self.staging.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
        let verify = self.verify;
        let chunks = chunks_if(verify, &self.staging.as_slice()[..bytes.len()]);
        let sha = fold_if(verify, bytes.len(), &chunks);
        let padded = match self.write_slot(slot, len) {
            Ok(p) => p,
            Err(e) => {
                self.free.push(slot);
                let _ = ledger.release(Pool::Nvme, &name);
                return Err(e);
            }
        };
        ledger.record_written(Pool::Nvme, padded)?;
        let id = self.next_id;
        self.next_id += 1;
        self.images.insert(
            id,
            Stored {
                slot,
                bytes: len,
                padded,
            },
        );
        self.stats.parks += 1;
        self.stats.bytes_in += len;
        Ok(ParkTicket {
            id,
            session,
            pool: Pool::Nvme,
            bytes: len,
            layout: Vec::new(),
            sha256: sha,
            chunks,
        })
    }

    fn restore_bytes(
        &mut self,
        ticket: &ParkTicket,
        dst: &mut [u8],
        ledger: &mut Ledger,
    ) -> Result<(), ParkError> {
        check_bytes_len(ticket, dst)?;
        let st = self.images.get(&ticket.id).ok_or(ParkError::NoSuchImage {
            pool: Pool::Nvme,
            id: ticket.id,
        })?;
        let (slot, bytes, padded) = (st.slot, st.bytes as usize, st.padded);
        self.read_slot(slot, padded)?;
        let sha = self.verify.digest(&self.staging.as_slice()[..bytes]);
        if sha != ticket.sha256 {
            self.stats.corrupt_refusals += 1;
            return Err(ParkError::ImageCorrupt {
                id: ticket.id,
                session: ticket.session,
                expected: hex::encode(ticket.sha256),
                found: hex::encode(sha),
            });
        }
        dst.copy_from_slice(&self.staging.as_slice()[..bytes]);
        self.evict(ticket.id);
        ledger.release(Pool::Nvme, &ticket.reservation())?;
        self.stats.restores += 1;
        self.stats.bytes_out += ticket.bytes;
        Ok(())
    }

    fn discard(&mut self, ticket: &ParkTicket, ledger: &mut Ledger) {
        if self.evict(ticket.id).is_some() {
            self.stats.discards += 1;
            if let Err(e) = ledger.release(Pool::Nvme, &ticket.reservation()) {
                tracing::error!(id = ticket.id, error = %e, "park tier release failed");
            }
        }
    }

    fn stats(&self) -> TierStats {
        self.stats
    }
}

// ---------------------------------------------------------------------------------------------
// Promotion and demotion — RAM ↔ NVMe, without touching the device
// ---------------------------------------------------------------------------------------------

/// Spill a parked image from RAM to NVMe. **The device is not involved**: the bytes are already in
/// pinned host memory, so this is one `pwritev` out of the slot they sit in.
///
/// Returns the new ticket. The old one is spent — its slot and its `HostPinned` reservation are
/// released, and the same session now holds an `Nvme` reservation and write traffic.
pub fn demote(
    ram: &mut PinnedHostTier,
    nvme: &mut NvmeTier,
    ticket: &ParkTicket,
    ledger: &mut Ledger,
) -> Result<ParkTicket, ParkError> {
    if ticket.pool != Pool::HostPinned {
        return Err(ParkError::Config(format!(
            "demote takes a HostPinned ticket, not {}",
            ticket.pool
        )));
    }
    let name = ticket.reservation();
    // Copy first, release second: if NVMe refuses, the image is still in RAM and the caller
    // still holds a valid ticket.
    let (new_id, bytes) = {
        let source = ram.image_bytes(ticket.id)?;
        let bytes = source.len() as u64;
        (nvme.adopt(source, ledger, &name)?, bytes)
    };
    ram.evict(ticket.id);
    ledger.release(Pool::HostPinned, &name)?;
    ram.stats.demotions += 1;
    nvme.stats.promotions += 1;
    Ok(ParkTicket {
        id: new_id,
        session: ticket.session,
        pool: Pool::Nvme,
        bytes,
        layout: ticket.layout.clone(),
        sha256: ticket.sha256,
        chunks: ticket.chunks.clone(),
    })
}

/// Bring a spilled image back into RAM, so the next restore is a DMA rather than a drive read.
///
/// The image is **verified on the way through** (I3): it was written to a drive and read back, and
/// the hash is the only thing that says those are the same bytes.
pub fn promote(
    nvme: &mut NvmeTier,
    ram: &mut PinnedHostTier,
    ticket: &ParkTicket,
    ledger: &mut Ledger,
) -> Result<ParkTicket, ParkError> {
    if ticket.pool != Pool::Nvme {
        return Err(ParkError::Config(format!(
            "promote takes an Nvme ticket, not {}",
            ticket.pool
        )));
    }
    let name = ticket.reservation();
    let st = nvme.images.get(&ticket.id).ok_or(ParkError::NoSuchImage {
        pool: Pool::Nvme,
        id: ticket.id,
    })?;
    let (slot, bytes, padded) = (st.slot, st.bytes as usize, st.padded);
    nvme.read_slot(slot, padded)?;
    let sha = nvme.verify.digest(&nvme.staging.as_slice()[..bytes]);
    if sha != ticket.sha256 {
        nvme.stats.corrupt_refusals += 1;
        return Err(ParkError::ImageCorrupt {
            id: ticket.id,
            session: ticket.session,
            expected: hex::encode(ticket.sha256),
            found: hex::encode(sha),
        });
    }
    let new_id = {
        let source = &nvme.staging.as_slice()[..bytes];
        ram.adopt(source, ledger, &name)?
    };
    nvme.evict(ticket.id);
    ledger.release(Pool::Nvme, &name)?;
    nvme.stats.demotions += 1;
    ram.stats.promotions += 1;
    Ok(ParkTicket {
        id: new_id,
        session: ticket.session,
        pool: Pool::HostPinned,
        bytes: bytes as u64,
        layout: ticket.layout.clone(),
        sha256: ticket.sha256,
        chunks: ticket.chunks.clone(),
    })
}

// ---------------------------------------------------------------------------------------------
// The real device half
// ---------------------------------------------------------------------------------------------

/// The real [`ParkDevice`]: copies on a dedicated stream, bounds-checked against the allocations
/// the caller says the regions live in.
///
/// It takes the *arenas* rather than trusting an address, so a region that is not inside one of
/// them is refused here instead of faulting the GPU three kernels later.
pub struct HipParkDevice<'h> {
    hip: &'h HipThread,
    stream: Stream,
    arenas: Vec<&'h DevicePtr>,
}

impl<'h> HipParkDevice<'h> {
    /// A device half over the allocations a session's regions live in (the KV arena and the GDN
    /// state arena, at least).
    pub fn new(
        hip: &'h HipThread,
        arenas: Vec<&'h DevicePtr>,
    ) -> Result<HipParkDevice<'h>, ParkError> {
        if arenas.is_empty() {
            return Err(ParkError::Config(
                "a park device needs at least one arena to copy within".into(),
            ));
        }
        let stream = hip.create_stream(StreamFlags::NonBlocking)?;
        Ok(HipParkDevice {
            hip,
            stream,
            arenas,
        })
    }

    /// The stream the copies run on.
    #[must_use]
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Which arena holds this region, and at what offset.
    fn locate(&self, region: &Region) -> Result<(&'h DevicePtr, u64), ParkError> {
        let lo = region.addr.as_u64();
        let hi = lo.checked_add(region.len).ok_or_else(|| {
            ParkError::Config(format!(
                "region at {lo:#x} of {} bytes overflows",
                region.len
            ))
        })?;
        for arena in &self.arenas {
            let base = arena.addr().as_u64();
            if lo >= base && hi <= base + arena.len() {
                return Ok((arena, lo - base));
            }
        }
        Err(ParkError::Config(format!(
            "region [{lo:#x}, {hi:#x}) ({}) is not inside any of the {} arena(s) this device was \
             given",
            region.kind,
            self.arenas.len()
        )))
    }
}

impl ParkDevice for HipParkDevice<'_> {
    fn d2h(&mut self, dst: &mut [u8], region: &Region) -> Result<(), ParkError> {
        if dst.len() as u64 != region.len {
            return Err(ParkError::Config(format!(
                "d2h: {} bytes of host buffer for a {} byte region",
                dst.len(),
                region.len
            )));
        }
        let (arena, offset) = self.locate(region)?;
        self.hip
            .memcpy_d2h_at_blocking(dst, arena, offset, &self.stream)?;
        Ok(())
    }

    fn h2d(&mut self, region: &Region, src: &[u8]) -> Result<(), ParkError> {
        if src.len() as u64 != region.len {
            return Err(ParkError::Config(format!(
                "h2d: {} bytes of host buffer for a {} byte region",
                src.len(),
                region.len
            )));
        }
        let (arena, offset) = self.locate(region)?;
        self.hip
            .memcpy_h2d_at_blocking(arena, offset, src, &self.stream)?;
        Ok(())
    }
}

impl std::fmt::Debug for HipParkDevice<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HipParkDevice({} arenas)", self.arenas.len())
    }
}
