//! Where a pack stream's read buffers come from.
//!
//! A pack stream reads `align_up(length)` bytes per record into a 4096-aligned host buffer. *Which*
//! host buffer is a policy, not a fact about the format, and it is the difference between two cold
//! starts:
//!
//! * [`PageAlignedPool`] — ordinary pageable memory, pooled by exact size under a byte budget. This
//!   is what workstream K shipped, and it makes the consumer copy every byte once more to get it
//!   into pinned staging: **8.3 GB/s single-threaded, 66% of the wall** (`docs/LEDGER.md`).
//! * A pinned pool (`saber_mem::upload::PinnedRing`) — the staging ring itself, so the drive lands
//!   the record where the H2D will read it and the host copy disappears.
//!
//! The pool owns the memory and the backpressure: [`BufferPool::acquire`] blocks while the pool is
//! full, which is what stops the readers from running ahead of the consumer. It hands out
//! [`PoolBuf`], a closed two-variant enum rather than a trait object, because there are exactly two
//! kinds of host memory here and a `dyn` buffer would have to be downcast back to its pool's own
//! type to be reclaimed.
//!
//! **Ownership crosses the callback.** A pinned buffer is the source of an *asynchronous* H2D, so
//! releasing it when `on_record` returns would let the next read overwrite bytes the DMA has not
//! finished with. The callback therefore returns the buffer when it is done with it, or `None` to
//! keep it and release it to the pool itself once its event completes.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};

use saber_hip::{PinnedChunk, PinnedSlice};

use crate::dio::AlignedBuf;

/// A record-sized read buffer, owned by its pool.
#[derive(Debug)]
pub enum PoolBuf<'p> {
    /// Ordinary page-aligned host memory. The H2D out of it costs a copy.
    Pageable(AlignedBuf),
    /// Pinned staging: the drive read into it, and the H2D reads it directly.
    Pinned(PinnedView<'p>),
}

impl PoolBuf<'_> {
    /// The visible payload.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        match self {
            PoolBuf::Pageable(b) => b,
            PoolBuf::Pinned(v) => v.as_slice(),
        }
    }

    /// The visible payload, mutably — what `preadv` reads into.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            PoolBuf::Pageable(b) => b,
            PoolBuf::Pinned(v) => v.as_mut_slice(),
        }
    }

    /// Bytes this buffer was handed out for.
    #[must_use]
    pub fn capacity(&self) -> usize {
        match self {
            PoolBuf::Pageable(b) => b.capacity(),
            PoolBuf::Pinned(v) => v.capacity(),
        }
    }

    /// Shrink the visible payload to a record's exact `length`, keeping the padded capacity.
    pub fn truncate(&mut self, len: usize) {
        match self {
            PoolBuf::Pageable(b) => b.truncate(len),
            PoolBuf::Pinned(v) => v.truncate(len),
        }
    }

    /// Restore the visible payload, up to the capacity.
    pub fn resize_visible(&mut self, len: usize) {
        match self {
            PoolBuf::Pageable(b) => b.resize_visible(len),
            PoolBuf::Pinned(v) => v.resize_visible(len),
        }
    }

    /// Whether these bytes are already in pinned memory.
    #[must_use]
    pub fn is_pinned(&self) -> bool {
        matches!(self, PoolBuf::Pinned(_))
    }

    /// A device-copyable view of the first `bytes`, when the buffer is pinned.
    ///
    /// `None` is not a failure: it is the consumer being told, in the type, that this record has to
    /// go the copying route. That is the branch the `records_spilled` counter exists for.
    #[must_use]
    pub fn pinned_head(&self, bytes: usize) -> Option<PinnedSlice<'_>> {
        match self {
            PoolBuf::Pageable(_) => None,
            PoolBuf::Pinned(v) => v.head(bytes),
        }
    }

    /// Host address of the payload, for an alignment check or a log line.
    #[must_use]
    pub fn addr(&self) -> usize {
        match self {
            PoolBuf::Pageable(b) => b.as_ptr() as usize,
            PoolBuf::Pinned(v) => v.addr(),
        }
    }
}

/// A run of staging slots, handed out as one contiguous buffer.
#[derive(Debug)]
pub struct PinnedView<'p> {
    chunk: PinnedChunk<'p>,
    slot: usize,
    slots: usize,
    visible: usize,
}

impl<'p> PinnedView<'p> {
    /// Wrap `slots` joined chunks starting at slot index `slot`.
    #[must_use]
    pub fn new(chunk: PinnedChunk<'p>, slot: usize, slots: usize) -> Self {
        let visible = chunk.len();
        PinnedView {
            chunk,
            slot,
            slots,
            visible,
        }
    }

    /// First slot index in the ring.
    #[must_use]
    pub fn slot(&self) -> usize {
        self.slot
    }

    /// How many slots it spans.
    #[must_use]
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// Give the chunk back, for a pool that is reclaiming the slots.
    #[must_use]
    pub fn into_chunk(self) -> PinnedChunk<'p> {
        self.chunk
    }

    fn as_slice(&self) -> &[u8] {
        &self.chunk.as_slice()[..self.visible]
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        let n = self.visible;
        &mut self.chunk.as_mut_slice()[..n]
    }

    fn capacity(&self) -> usize {
        self.chunk.len()
    }

    fn truncate(&mut self, len: usize) {
        assert!(len <= self.visible, "truncate beyond payload");
        self.visible = len;
    }

    fn resize_visible(&mut self, len: usize) {
        assert!(len <= self.chunk.len(), "resize beyond capacity");
        self.visible = len;
    }

    fn head(&self, bytes: usize) -> Option<PinnedSlice<'_>> {
        self.chunk.head(bytes)
    }

    fn addr(&self) -> usize {
        self.chunk.addr()
    }
}

/// What a pool did, for the load report and for `/metrics`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Records served out of pinned memory (the zero-copy path).
    pub pinned_records: u64,
    /// Bytes served out of pinned memory.
    pub pinned_bytes: u64,
    /// Records served out of pageable memory (the copying path).
    pub pageable_records: u64,
    /// Bytes served out of pageable memory.
    pub pageable_bytes: u64,
    /// Times an acquire had to wait for the pool. This is backpressure, counted.
    pub stalls: u64,
    /// Peak bytes held in flight.
    pub peak_inflight_bytes: u64,
    /// Bytes the pool has allocated (a pinned ring allocates all of it up front, I1).
    pub allocated_bytes: u64,
}

/// A source of pack-stream read buffers.
///
/// `Sync` because every I/O worker acquires from it concurrently, and the consumer releases into it
/// from its own thread.
///
/// The lifetime is the **buffers'**, not the pool's borrow: a pinned ring hands out views into one
/// allocation and has to be able to put them back, which a `&self`-scoped lifetime cannot express
/// (`Mutex<RingState<'a>>` is invariant in `'a`).
pub trait BufferPool<'p>: Sync + std::fmt::Debug {
    /// Take a buffer with `padded` bytes of capacity, blocking while the pool is full.
    ///
    /// Returns `None` only when `abort` says the stream is stopping — a pool that cannot ever serve
    /// a request of this size must serve it some other way (a pinned ring falls back to pageable
    /// memory and counts it) rather than returning `None`, which the stream reads as "we are done".
    fn acquire(&self, padded: usize, abort: &(dyn Fn() -> bool + Sync)) -> Option<PoolBuf<'p>>;

    /// Give a buffer back.
    fn release(&self, buf: PoolBuf<'p>);

    /// Bytes the pool may hold in flight.
    fn capacity_bytes(&self) -> u64;

    /// Make sure a single buffer of `bytes` can be served, raising the budget if that is what this
    /// pool does. The stream calls it once with the largest padded record in its window, because a
    /// budget below that would deadlock a pool that blocks.
    fn accommodate(&self, _bytes: u64) {}

    /// Wake every thread blocked in [`BufferPool::acquire`], so an aborting stream does not leave a
    /// reader asleep on a buffer that is never coming.
    fn wake_sleepers(&self) {}

    /// Counters.
    fn stats(&self) -> PoolStats;
}

/// The default pool: 4096-aligned pageable buffers, pooled by exact size under a byte budget.
///
/// Record sizes repeat heavily in a real tier (168 records of 44,564,480 bytes in the NVFP4 tier),
/// so pooling by exact size turns ~11 GB of allocation into a handful of `mmap`s and keeps resident
/// memory at the budget.
#[derive(Debug)]
pub struct PageAlignedPool {
    capacity: AtomicU64,
    state: Mutex<PageState>,
    cv: Condvar,
    stalls: AtomicU64,
    records: AtomicU64,
    bytes: AtomicU64,
}

#[derive(Debug, Default)]
struct PageState {
    inflight: u64,
    peak: u64,
    allocated: u64,
    free: HashMap<usize, Vec<AlignedBuf>>,
}

impl PageAlignedPool {
    /// A pool that may hold `capacity_bytes` in flight.
    #[must_use]
    pub fn new(capacity_bytes: u64) -> Self {
        PageAlignedPool {
            capacity: AtomicU64::new(capacity_bytes.max(1)),
            state: Mutex::new(PageState::default()),
            cv: Condvar::new(),
            stalls: AtomicU64::new(0),
            records: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
        }
    }
}

impl<'p> BufferPool<'p> for PageAlignedPool {
    fn acquire(&self, padded: usize, abort: &(dyn Fn() -> bool + Sync)) -> Option<PoolBuf<'p>> {
        let need = padded as u64;
        let mut g = lock(&self.state);
        loop {
            if abort() {
                return None;
            }
            if g.inflight + need <= self.capacity.load(Ordering::Acquire) {
                g.inflight += need;
                g.peak = g.peak.max(g.inflight);
                let pooled = g.free.get_mut(&padded).and_then(Vec::pop);
                let mut buf = match pooled {
                    Some(b) => b,
                    None => {
                        g.allocated += need;
                        AlignedBuf::zeroed(padded)
                    }
                };
                drop(g);
                buf.resize_visible(padded);
                self.records.fetch_add(1, Ordering::Relaxed);
                self.bytes.fetch_add(need, Ordering::Relaxed);
                return Some(PoolBuf::Pageable(buf));
            }
            self.stalls.fetch_add(1, Ordering::Relaxed);
            g = self
                .cv
                .wait(g)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    fn release(&self, buf: PoolBuf<'p>) {
        let PoolBuf::Pageable(buf) = buf else {
            // A pool never sees a buffer it did not hand out; this one cannot reclaim pinned slots.
            tracing::error!("a pinned buffer was released into the pageable pool; leaking it");
            return;
        };
        let size = buf.capacity();
        let mut g = lock(&self.state);
        g.inflight -= size as u64;
        g.free.entry(size).or_default().push(buf);
        drop(g);
        self.cv.notify_all();
    }

    fn capacity_bytes(&self) -> u64 {
        self.capacity.load(Ordering::Acquire)
    }

    fn accommodate(&self, bytes: u64) {
        // A budget below the largest single record would block forever, so raise it. The pool holds
        // at most one such buffer at a time anyway.
        let mut cur = self.capacity.load(Ordering::Acquire);
        while cur < bytes {
            match self
                .capacity
                .compare_exchange(cur, bytes, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => break,
                Err(now) => cur = now,
            }
        }
        self.cv.notify_all();
    }

    fn wake_sleepers(&self) {
        self.cv.notify_all();
    }

    fn stats(&self) -> PoolStats {
        let g = lock(&self.state);
        PoolStats {
            pinned_records: 0,
            pinned_bytes: 0,
            pageable_records: self.records.load(Ordering::Relaxed),
            pageable_bytes: self.bytes.load(Ordering::Relaxed),
            stalls: self.stalls.load(Ordering::Relaxed),
            peak_inflight_bytes: g.peak,
            allocated_bytes: g.allocated,
        }
    }
}

/// A poisoned lock means a worker panicked: take the data and let the scope's join surface the
/// original panic rather than deadlocking the other side of the stream.
pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_pool_reuses_buffers_of_the_same_size() {
        let pool = PageAlignedPool::new(1 << 20);
        let never = || false;
        let a = pool.acquire(4096, &never).expect("acquire");
        assert!(!a.is_pinned());
        assert!(a.pinned_head(4096).is_none(), "pageable has no pinned view");
        assert_eq!(a.as_slice().len(), 4096);
        let addr = a.addr();
        assert_eq!(addr % 4096, 0, "O_DIRECT needs the address aligned");
        pool.release(a);

        let b = pool.acquire(4096, &never).expect("acquire");
        assert_eq!(b.addr(), addr, "the same buffer came back out of the pool");
        pool.release(b);
        let s = pool.stats();
        assert_eq!(s.pageable_records, 2);
        assert_eq!(
            s.allocated_bytes, 4096,
            "the second acquire allocated nothing"
        );
        assert_eq!(s.pinned_records, 0);
    }

    #[test]
    fn a_full_pool_blocks_and_counts_the_stall() {
        let pool = PageAlignedPool::new(8192);
        let never = || false;
        let a = pool.acquire(8192, &never).expect("acquire");
        // The pool is full: this thread must wait until `a` goes back.
        std::thread::scope(|s| {
            let h = s.spawn(|| {
                let b = pool.acquire(8192, &never).expect("acquire after release");
                pool.release(b);
            });
            // Give the other thread time to get into the wait, then release.
            std::thread::sleep(std::time::Duration::from_millis(20));
            pool.release(a);
            h.join().expect("no panic");
        });
        assert!(pool.stats().stalls > 0, "the sleeper was not counted");
    }

    #[test]
    fn abort_releases_a_sleeper_instead_of_hanging_the_stream() {
        let pool = PageAlignedPool::new(4096);
        let never = || false;
        let held = pool.acquire(4096, &never).expect("acquire");
        let aborted = || true;
        assert!(
            pool.acquire(4096, &aborted).is_none(),
            "an aborting stream must not block in the pool"
        );
        pool.release(held);
    }

    #[test]
    fn accommodate_raises_the_budget_for_one_oversized_record() {
        let pool = PageAlignedPool::new(4096);
        assert_eq!(pool.capacity_bytes(), 4096);
        pool.accommodate(1 << 20);
        assert_eq!(pool.capacity_bytes(), 1 << 20);
        pool.accommodate(4096);
        assert_eq!(pool.capacity_bytes(), 1 << 20, "never lowered");
        let never = || false;
        let b = pool
            .acquire(1 << 20, &never)
            .expect("the big record fits now");
        pool.release(b);
    }
}
