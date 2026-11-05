//! Direct-I/O primitives: 4096-aligned host buffers and positional `preadv`.
//!
//! This is the only module in `saber-mem` that contains `unsafe`, and the only `unsafe` in it is the
//! `preadv(2)` call: the buffer type below is a safe over-allocated `Vec<u8>` sliced to a
//! page-aligned interior, so nothing here hands out a raw pointer or a lifetime the compiler cannot
//! check.
//!
//! `O_DIRECT` demands that the file offset, the transfer length and the memory address are all
//! multiples of the device's logical block size (512 on the Samsung 9100 PRO in this box; 4096 is
//! used everywhere, which satisfies both 512- and 4096-sector devices). The pack format makes that
//! free: every record starts on a 4096 boundary and the file length is a multiple of 4096, so
//! reading `align_up(length)` bytes at `data_off` is always a legal direct read.
//!
//! `io_uring` is the P5 upgrade. Threads + `preadv` reproduce the QD8 point the drive was measured
//! at (`docs/LEDGER.md`: 8 × 1 MiB in flight, 14.79 GB/s) without a new dependency.
//!
//! FN8 adds [`IoUring`], a minimal ring (raw `io_uring_setup`/`io_uring_enter` + the three
//! mappings, `IORING_OP_READ` only) for the PLE tier's per-token row reads, where the reader-thread
//! hand-off was most of the cost. Its `unsafe` is the syscalls, the mappings and the ring-index
//! loads/stores, each commented; its public surface is safe (`read_batch` borrows every buffer
//! uniquely until every completion is reaped).

use std::fs::{File, OpenOptions};
use std::io;
use std::ops::{Deref, DerefMut};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

/// Alignment required for `O_DIRECT` offsets, lengths and buffer addresses.
pub const DIO_ALIGN: usize = 4096;

/// Round `x` up to the next multiple of [`DIO_ALIGN`].
#[inline]
#[must_use]
pub fn align_up(x: u64) -> u64 {
    let a = DIO_ALIGN as u64;
    x.div_ceil(a) * a
}

/// How a pack file is opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OpenMode {
    /// `O_DIRECT`: bypasses the page cache. The only honest mode for a throughput measurement, and
    /// the mode the loader uses in production.
    #[default]
    Direct,
    /// Ordinary buffered I/O. For filesystems that do not implement `O_DIRECT` (tmpfs) and for
    /// tests that only care about bytes.
    Buffered,
}

/// A heap buffer whose payload starts on a [`DIO_ALIGN`] boundary.
///
/// Implemented as a `Vec<u8>` over-allocated by one alignment unit and sliced to its aligned
/// interior — no manual allocator calls, no `unsafe`, and `Send`/`Sync` come from `Vec`.
#[derive(Debug)]
pub struct AlignedBuf {
    raw: Vec<u8>,
    off: usize,
    len: usize,
    cap: usize,
}

impl AlignedBuf {
    /// Allocate a zeroed buffer with `len` aligned payload bytes.
    #[must_use]
    pub fn zeroed(len: usize) -> Self {
        let raw = vec![0u8; len + DIO_ALIGN];
        let off = raw.as_ptr().align_offset(DIO_ALIGN);
        assert!(
            off != usize::MAX && off + len <= raw.len(),
            "align_offset failed for a u8 allocation; this cannot happen"
        );
        Self {
            raw,
            off,
            len,
            cap: len,
        }
    }

    /// Payload length in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when the payload is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Payload bytes this buffer was allocated for.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Shrink the visible payload. Capacity is kept, so a padded read can be handed out as the
    /// record's exact `length`.
    pub fn truncate(&mut self, len: usize) {
        assert!(len <= self.len, "truncate beyond payload");
        self.len = len;
    }

    /// Restore the visible payload to `len`, up to the buffer's capacity.
    pub fn resize_visible(&mut self, len: usize) {
        assert!(len <= self.capacity(), "resize beyond capacity");
        self.len = len;
    }
}

impl Deref for AlignedBuf {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.raw[self.off..self.off + self.len]
    }
}

impl DerefMut for AlignedBuf {
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut self.raw[self.off..self.off + self.len]
    }
}

/// Open `path` for reading in the requested mode.
pub fn open_read(path: &Path, mode: OpenMode) -> io::Result<File> {
    let mut opts = OpenOptions::new();
    opts.read(true);
    if mode == OpenMode::Direct {
        opts.custom_flags(libc::O_DIRECT);
    }
    opts.open(path)
}

/// One positional vectored read. Returns the number of bytes read (0 at EOF).
///
/// A single `iovec` is used today; the signature is `preadv` rather than `pread` because the
/// scatter form is what the loader will need when a record is delivered into several pinned
/// staging segments, and because it keeps the `unsafe` surface at exactly one call.
pub fn preadv_once(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    let mut iov = libc::iovec {
        iov_base: buf.as_mut_ptr().cast::<libc::c_void>(),
        iov_len: buf.len(),
    };
    // SAFETY: `iov` describes exactly `buf`, which is a live, uniquely borrowed `&mut [u8]` for the
    // duration of the call; `iovcnt` is 1, matching the single `iovec`; `file` owns a valid fd for
    // the duration of the borrow. `preadv` does not retain the pointer past the call and writes at
    // most `iov_len` bytes.
    let n = unsafe { libc::preadv(file.as_raw_fd(), &raw mut iov, 1, offset as libc::off_t) };
    if n < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(n as usize)
    }
}

/// Fill `buf` entirely from `offset`, looping over short reads.
///
/// Fails loudly on a short read that makes no progress: with `O_DIRECT` a partial read that leaves
/// an unaligned remainder cannot be resumed, and the pack format guarantees the range is inside the
/// file, so "0 bytes and no error" is a real fault and never a normal EOF.
pub fn preadv_exact(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    let total = buf.len();
    let mut done = 0usize;
    while done < total {
        let n = preadv_once(file, &mut buf[done..], offset + done as u64)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("short read: {done} of {total} bytes at offset {offset}"),
            ));
        }
        done += n;
    }
    Ok(())
}

/// Open (creating if needed) a file for read **and** write in the requested mode.
///
/// The parking tier needs both directions on one descriptor: it writes an image on park and reads
/// it back on restore, and re-opening between the two would give the page cache a chance to serve
/// a read the drive never saw — which is the measurement lying, not the loader.
pub fn open_rw_create(path: &Path, mode: OpenMode) -> io::Result<File> {
    let mut opts = OpenOptions::new();
    opts.read(true).write(true).create(true).truncate(false);
    if mode == OpenMode::Direct {
        opts.custom_flags(libc::O_DIRECT);
    }
    opts.open(path)
}

/// `posix_fallocate`: give the file `len` bytes of real, contiguous space up front.
///
/// Not a convenience. `docs/LEDGER.md` (tier pack, 2026-09-18): without it two interleaved writers
/// left xfs handing each rank five extents braided through the other's, which cost 3% of the read
/// rate and 7x the metadata read. A parking file is written and re-read constantly; allocating it
/// once is what keeps a restore a sequential read.
pub fn fallocate(file: &File, len: u64) -> io::Result<()> {
    // SAFETY: `file` owns a valid fd for the duration of the borrow; `posix_fallocate` only
    // allocates blocks for the given range and reports its error as a return value.
    let rc = unsafe { libc::posix_fallocate(file.as_raw_fd(), 0, len as libc::off_t) };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::from_raw_os_error(rc))
    }
}

/// One positional vectored write. Returns the number of bytes written.
pub fn pwritev_once(file: &File, buf: &[u8], offset: u64) -> io::Result<usize> {
    let iov = libc::iovec {
        // `pwritev` does not write through this pointer, but the C type is not const.
        iov_base: buf.as_ptr().cast::<libc::c_void>().cast_mut(),
        iov_len: buf.len(),
    };
    // SAFETY: `iov` describes exactly `buf`, which is a live, borrowed `&[u8]` for the duration of
    // the call; `iovcnt` is 1, matching the single `iovec`; `file` owns a valid fd. `pwritev` reads
    // at most `iov_len` bytes and does not retain the pointer.
    let n = unsafe { libc::pwritev(file.as_raw_fd(), &raw const iov, 1, offset as libc::off_t) };
    if n < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(n as usize)
    }
}

/// Write `buf` entirely at `offset`, looping over short writes.
///
/// A short write that makes no progress is a fault, not a retry loop: with `O_DIRECT` the next
/// write would have to start at an unaligned offset, which the kernel refuses.
pub fn pwritev_exact(file: &File, buf: &[u8], offset: u64) -> io::Result<()> {
    let total = buf.len();
    let mut done = 0usize;
    while done < total {
        let n = pwritev_once(file, &buf[done..], offset + done as u64)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                format!("short write: {done} of {total} bytes at offset {offset}"),
            ));
        }
        done += n;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_buf_is_page_aligned() {
        for len in [0usize, 1, 4095, 4096, 4097, 1 << 20] {
            let b = AlignedBuf::zeroed(len);
            assert_eq!(b.as_ptr() as usize % DIO_ALIGN, 0, "len {len}");
            assert_eq!(b.len(), len);
            assert!(b.iter().all(|&x| x == 0));
        }
    }

    #[test]
    fn align_up_rounds() {
        assert_eq!(align_up(0), 0);
        assert_eq!(align_up(1), 4096);
        assert_eq!(align_up(4096), 4096);
        assert_eq!(align_up(4097), 8192);
    }

    #[test]
    fn truncate_keeps_capacity() {
        let mut b = AlignedBuf::zeroed(8192);
        b.truncate(10);
        assert_eq!(b.len(), 10);
        assert_eq!(b.capacity(), 8192);
        b.resize_visible(8192);
        assert_eq!(b.len(), 8192);
    }
}

// ------------------------------------------------------------------------------------ io_uring
//
// FN8's trial for the PLE tier's decode path: a token's 16 row reads are submitted from the
// CALLING thread as one batch and reaped by the same `io_uring_enter` — no reader-thread hand-off
// (FN7: ~85 us of a token's 130 us p50 was two hand-offs). Minimal on purpose: one ring, IORING_OP_READ
// only, no SQPOLL, no registered buffers; the ring is used by one thread at a time (`&mut self`).
// The layouts below are the kernel's UAPI (`include/uapi/linux/io_uring.h`), which is stable.

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct SqOffsets {
    head: u32,
    tail: u32,
    ring_mask: u32,
    ring_entries: u32,
    flags: u32,
    dropped: u32,
    array: u32,
    resv1: u32,
    user_addr: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct CqOffsets {
    head: u32,
    tail: u32,
    ring_mask: u32,
    ring_entries: u32,
    overflow: u32,
    cqes: u32,
    flags: u32,
    resv1: u32,
    user_addr: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct UringParams {
    sq_entries: u32,
    cq_entries: u32,
    flags: u32,
    sq_thread_cpu: u32,
    sq_thread_idle: u32,
    features: u32,
    wq_fd: u32,
    resv: [u32; 3],
    sq_off: SqOffsets,
    cq_off: CqOffsets,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct Sqe {
    opcode: u8,
    flags: u8,
    ioprio: u16,
    fd: i32,
    off: u64,
    addr: u64,
    len: u32,
    rw_flags: u32,
    user_data: u64,
    buf_index: u16,
    personality: u16,
    splice_fd_in: i32,
    addr3: u64,
    pad2: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Cqe {
    user_data: u64,
    res: i32,
    flags: u32,
}

const IORING_OFF_SQ_RING: libc::off_t = 0;
const IORING_OFF_CQ_RING: libc::off_t = 0x0800_0000;
const IORING_OFF_SQES: libc::off_t = 0x1000_0000;
const IORING_OP_READ: u8 = 22;
const IORING_ENTER_GETEVENTS: u32 = 1;

struct Map {
    ptr: *mut u8,
    len: usize,
}

impl Map {
    fn new(fd: i32, len: usize, off: libc::off_t) -> io::Result<Map> {
        // SAFETY: a fresh shared mapping of the ring fd at the offset the kernel documents; the
        // kernel sized the region (`len` comes from its own offsets); nothing aliases it yet.
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_POPULATE,
                fd,
                off,
            )
        };
        if p == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(Map { ptr: p.cast(), len })
    }
    fn at<T>(&self, off: u32) -> *mut T {
        debug_assert!((off as usize) + std::mem::size_of::<T>() <= self.len);
        // SAFETY: `off` is one of the kernel's offsets into this mapping (bounds asserted above).
        unsafe { self.ptr.add(off as usize).cast() }
    }
}

impl Drop for Map {
    fn drop(&mut self) {
        // SAFETY: `ptr`/`len` are exactly what `mmap` returned; nothing borrows the mapping past
        // the ring's lifetime (every accessor goes through `&self`/`&mut self` of the ring).
        unsafe {
            libc::munmap(self.ptr.cast(), self.len);
        }
    }
}

/// One `io_uring` for batches of positional reads (see the section comment).
pub struct IoUring {
    fd: i32,
    sq: Map,
    cq: Map,
    sqes: Map,
    p: UringParams,
}

impl std::fmt::Debug for IoUring {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IoUring(fd {}, {} entries)", self.fd, self.p.sq_entries)
    }
}

impl IoUring {
    /// A ring of `entries` submission slots (a power of two is what the kernel rounds to).
    pub fn new(entries: u32) -> io::Result<IoUring> {
        let mut p = UringParams::default();
        // SAFETY: `p` is a live, correctly laid out `io_uring_params` the kernel fills.
        let fd = unsafe {
            libc::syscall(
                libc::SYS_io_uring_setup,
                entries as libc::c_long,
                &raw mut p as libc::c_long,
            )
        };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        let fd = fd as i32;
        let sq_len = p.sq_off.array as usize + p.sq_entries as usize * 4;
        let cq_len = p.cq_off.cqes as usize + p.cq_entries as usize * std::mem::size_of::<Cqe>();
        let close = |e: io::Error| {
            // SAFETY: `fd` is the ring fd this function just opened and nothing else holds it.
            unsafe {
                libc::close(fd);
            }
            e
        };
        let sq = Map::new(fd, sq_len, IORING_OFF_SQ_RING).map_err(close)?;
        let cq = Map::new(fd, cq_len, IORING_OFF_CQ_RING).map_err(close)?;
        let sqes = Map::new(
            fd,
            p.sq_entries as usize * std::mem::size_of::<Sqe>(),
            IORING_OFF_SQES,
        )
        .map_err(close)?;
        Ok(IoUring {
            fd,
            sq,
            cq,
            sqes,
            p,
        })
    }

    /// Submission slots.
    #[must_use]
    pub fn entries(&self) -> u32 {
        self.p.sq_entries
    }

    /// Read every `(file, offset, buffer)` fully, all in flight at once (in waves of the ring's
    /// size), and return when every one has completed. A short read or an error fails the batch.
    pub fn read_batch(&mut self, reads: &mut [(&File, u64, &mut [u8])]) -> io::Result<()> {
        use std::sync::atomic::{AtomicU32, Ordering};
        let n_ring = self.p.sq_entries as usize;
        for (w, wave) in reads.chunks_mut(n_ring).enumerate() {
            let sq_tail: *mut u32 = self.sq.at(self.p.sq_off.tail);
            let sq_mask = self.p.sq_entries - 1;
            let array: *mut u32 = self.sq.at(self.p.sq_off.array);
            let sqes: *mut Sqe = self.sqes.at(0);
            // SAFETY: the tail is a u32 the kernel only reads; we are the only producer.
            let tail0 = unsafe { AtomicU32::from_ptr(sq_tail) }.load(Ordering::Acquire);
            for (k, (f, off, buf)) in wave.iter_mut().enumerate() {
                let t = tail0.wrapping_add(k as u32);
                let idx = t & sq_mask;
                let sqe = Sqe {
                    opcode: IORING_OP_READ,
                    fd: f.as_raw_fd(),
                    off: *off,
                    addr: buf.as_mut_ptr() as u64,
                    len: buf.len() as u32,
                    user_data: (w * n_ring + k) as u64,
                    ..Sqe::default()
                };
                // SAFETY: `idx < sq_entries`, inside the SQE array and the index array the kernel
                // mapped; the kernel does not read an entry before the tail publishes it.
                unsafe {
                    sqes.add(idx as usize).write(sqe);
                    array.add(idx as usize).write(idx);
                }
            }
            let n = wave.len() as u32;
            // SAFETY: as above; the release store publishes the entries written before it.
            unsafe { AtomicU32::from_ptr(sq_tail) }.store(tail0.wrapping_add(n), Ordering::Release);
            let mut done = 0u32;
            while done < n {
                // SAFETY: submit the new entries (first pass) and wait for completions; the buffers
                // the SQEs name are uniquely borrowed by `reads` for this whole call and are not
                // touched by us until their completions are reaped below.
                let r = unsafe {
                    libc::syscall(
                        libc::SYS_io_uring_enter,
                        self.fd as libc::c_long,
                        if done == 0 { n } else { 0 } as libc::c_long,
                        (n - done) as libc::c_long,
                        IORING_ENTER_GETEVENTS as libc::c_long,
                        0 as libc::c_long,
                        0 as libc::c_long,
                    )
                };
                if r < 0 {
                    let e = io::Error::last_os_error();
                    if e.kind() == io::ErrorKind::Interrupted {
                        continue;
                    }
                    return Err(e);
                }
                let head_p: *mut u32 = self.cq.at(self.p.cq_off.head);
                let tail_p: *mut u32 = self.cq.at(self.p.cq_off.tail);
                let cqes: *const Cqe = self.cq.at(self.p.cq_off.cqes);
                let mask = self.p.cq_entries - 1;
                // SAFETY: the CQ head is ours to advance, the tail is the kernel's (acquire).
                let (head, tail) = unsafe {
                    (
                        AtomicU32::from_ptr(head_p).load(Ordering::Relaxed),
                        AtomicU32::from_ptr(tail_p).load(Ordering::Acquire),
                    )
                };
                let mut h = head;
                while h != tail {
                    // SAFETY: entries between head and tail are completed CQEs the kernel wrote.
                    let c = unsafe { cqes.add((h & mask) as usize).read() };
                    let k = c.user_data as usize - w * n_ring;
                    let want = wave.get(k).map_or(0, |x| x.2.len());
                    if c.res < 0 {
                        return Err(io::Error::from_raw_os_error(-c.res));
                    }
                    if c.res as usize != want {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            format!("io_uring read {k}: {} of {want} bytes", c.res),
                        ));
                    }
                    h = h.wrapping_add(1);
                    done += 1;
                }
                // SAFETY: release the reaped entries back to the kernel.
                unsafe { AtomicU32::from_ptr(head_p) }.store(h, Ordering::Release);
            }
        }
        Ok(())
    }
}

impl Drop for IoUring {
    fn drop(&mut self) {
        // SAFETY: the ring fd is owned by this value; the mappings are dropped after (fields).
        unsafe {
            libc::close(self.fd);
        }
    }
}
