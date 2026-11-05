//! FN117b (S6): THE PICK LINE -- the two rank processes' per-step pick exchange through a shared
//! host page instead of the socket.
//!
//! The pick frame (each rank's seam top-k, its half's log-sum-exps, probes) is produced AND consumed
//! by host threads (the log-sum-exps are the host sampling law, LEDGER FN117 S5), so the landed line
//! is read by the host: the line lives in one `MAP_SHARED` file mapping both processes hold, not in
//! VRAM (a device line would add a D2H poll per check). The laws are the all-reduce workspace's
//! (CLAUDE.md, FN114b / FN114d):
//!
//! * **the slot is the call's GLOBAL parity** -- both ranks count every exchange (`seq` = 1, 2, ...,
//!   in lockstep: the exchange is a rendezvous), and call `seq` writes slot `seq % 2` of its own
//!   outbox. Call `seq + 2` may reuse a slot only after the peer copied call `seq` out: the peer
//!   copies call `seq` before it posts call `seq + 1`, which this rank waits for before it can
//!   reach `seq + 2`; the peer's `ack` word (the last call it copied out) is CHECKED anyway before
//!   a slot is rewritten (a wait there is counted: it is 0 by construction);
//! * **no flag before data** -- the payload words and the trailer are stored first, then the
//!   header's `seq` with RELEASE (the host's form of the system-scope write-back); the reader
//!   ACQUIRES the header, copies every word, then re-reads the header and checks the trailer
//!   (a slot rewritten under the copy is refused as `torn`, never returned);
//! * every word is an atomic (relaxed) access, so even the negative control's race is defined.
//!
//! [`ShmLine::control_one_slot`] + [`ShmLine::post_unchecked`] is the hazard the law closes (one
//! slot for every call, no ack): a rank that posts call `seq + 1` while its peer still copies call
//! `seq` tears the peer's read; the tests reproduce it deterministically and show the two-slot law
//! clean.

//! # The line format (FN119 extends it to a result block; LEDGER FN117b)
//!
//! One mapping of [`ShmLine::map_bytes`]`(cap)` bytes (4 KiB rounded), two OUTBOXES (rank 0's at
//! byte 0, rank 1's at `ACK + 2 * slot_bytes`), each `[ack: u64 | pad to 64][slot 0][slot 1]`,
//! a slot `slot_bytes(cap) = round64(64 + round8(cap) + 8)` bytes, all words little-endian u64:
//!
//! | offset in a slot | word | written |
//! |---|---|---|
//! | 0 | `seq` -- the call that filled the slot (0 = never) | LAST, with Release |
//! | 8 | `len` -- payload bytes | before `seq` |
//! | 16..64 | reserved (0) | -- |
//! | 64 .. 64 + round8(len) | payload, 8-byte words (the tail word zero-padded) | first |
//! | 64 + round8(len) | trailer = `seq` | before `seq` |
//!
//! Call `seq` (1, 2, ... counted by both ranks in lockstep) writes its OWN outbox's slot
//! `seq % 2` (the FN114d global-parity law) and reads the PEER's slot `seq % 2`. The consumer
//! detects a landed line by an Acquire load of the slot's `seq` == the call it expects (any other
//! value but 0 or `seq - 2` is a protocol error), copies `len` bytes, then re-reads `seq` and the
//! trailer: both must still equal the call (else `torn`). After the copy the consumer stores the
//! call into its own outbox's `ack` word (Release); a producer about to rewrite slot `seq % 2`
//! first checks the peer's `ack >= seq - 2`. Host producers make the Release store the
//! "write-back"; a DEVICE producer of the same format must make its payload + trailer visible at
//! system scope (`__threadfence_system()` / a release at system scope) before the `seq` store,
//! per the CLAUDE.md AR protocol law.

use std::path::{Path, PathBuf};
use std::sync::atomic::{fence, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Header words of a slot: `seq` (the call that wrote it), `len` (payload bytes), pad.
const HEAD: usize = 64;
/// The ack word (the last call this rank copied out of its PEER's outbox) sits at the start of
/// each rank's outbox region; the slots follow it.
const ACK: usize = 64;

/// The counters of one rank's line.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LineCounts {
    /// Exchanges completed.
    pub calls: u64,
    /// Payload bytes sent.
    pub bytes: u64,
    /// Posts that found the peer's ack behind the slot's previous call (0 by construction).
    pub ack_waits: u64,
    /// Reads whose first header poll had not landed yet (the peer was later than this rank).
    pub late: u64,
}

/// One rank's view of the shared pick line.
pub struct ShmLine {
    base: *mut u8,
    map_len: usize,
    rank: u32,
    cap: usize,
    slots: u64,
    seq: u64,
    timeout: Duration,
    path: PathBuf,
    owner: bool,
    /// The counters.
    pub counts: LineCounts,
}

fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

impl ShmLine {
    /// The mapping's size for a line of `cap` payload bytes a slot.
    #[must_use]
    pub fn map_bytes(cap: usize) -> usize {
        let outbox = ACK + 2 * Self::slot_bytes(cap);
        (2 * outbox).div_ceil(4096) * 4096
    }

    fn slot_bytes(cap: usize) -> usize {
        // header + payload (8-byte words) + trailer word, 64-aligned
        (HEAD + cap.div_ceil(8) * 8 + 8).div_ceil(64) * 64
    }

    /// Rank 0: create the file at `path` (`O_EXCL`, mode 0600) and map it (zeroed).
    ///
    /// # Errors
    /// The file exists, or open / ftruncate / mmap fail.
    pub fn create(path: &Path, cap: usize, timeout: Duration) -> Result<Self, String> {
        Self::map(path, cap, 0, timeout, true)
    }

    /// Rank 1: map the file rank 0 created.
    ///
    /// # Errors
    /// open / fstat / mmap fail, or the file is not the size this `cap` maps.
    pub fn open(path: &Path, cap: usize, timeout: Duration) -> Result<Self, String> {
        Self::map(path, cap, 1, timeout, false)
    }

    fn map(
        path: &Path,
        cap: usize,
        rank: u32,
        timeout: Duration,
        create: bool,
    ) -> Result<Self, String> {
        use std::os::unix::ffi::OsStrExt;
        let len = Self::map_bytes(cap);
        let c = std::ffi::CString::new(path.as_os_str().as_bytes())
            .map_err(|_| format!("{}: a NUL in the path", path.display()))?;
        let flags = if create {
            libc::O_RDWR | libc::O_CREAT | libc::O_EXCL | libc::O_CLOEXEC
        } else {
            libc::O_RDWR | libc::O_CLOEXEC
        };
        // SAFETY: `c` is a valid NUL-terminated path; the flags and mode are plain ints.
        let fd = unsafe { libc::open(c.as_ptr(), flags, 0o600 as libc::c_uint) };
        if fd < 0 {
            return Err(format!("open {}: errno {}", path.display(), errno()));
        }
        let close = |fd: i32| {
            // SAFETY: `fd` is the descriptor this function opened and owns.
            unsafe { libc::close(fd) };
        };
        if create {
            // SAFETY: `fd` is open for writing; `len` fits an off_t.
            if unsafe { libc::ftruncate(fd, len as libc::off_t) } != 0 {
                let e = errno();
                close(fd);
                return Err(format!("ftruncate {} to {len}: errno {e}", path.display()));
            }
        } else {
            // SAFETY: an all-zero `stat` is a valid out-parameter for fstat.
            let mut st: libc::stat = unsafe { std::mem::zeroed() };
            // SAFETY: `fd` is open and `st` is a valid `stat` to fill.
            if unsafe { libc::fstat(fd, &mut st) } != 0 || st.st_size as usize != len {
                close(fd);
                return Err(format!(
                    "{}: {} B, a line of cap {cap} maps {len} B",
                    path.display(),
                    st.st_size
                ));
            }
        }
        // SAFETY: a fresh shared read-write mapping of `len` bytes of the open file.
        let raw = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        close(fd);
        if raw == libc::MAP_FAILED {
            return Err(format!(
                "mmap {} ({len} B): errno {}",
                path.display(),
                errno()
            ));
        }
        Ok(ShmLine {
            base: raw.cast::<u8>(),
            map_len: len,
            rank,
            cap,
            slots: 2,
            seq: 0,
            timeout,
            path: path.to_path_buf(),
            owner: create,
            counts: LineCounts::default(),
        })
    }

    /// Remove the file's name (both ranks keep their mappings). Rank 0 calls it once rank 1 mapped.
    ///
    /// # Errors
    /// unlink failed (other than the name already gone).
    pub fn unlink(&self) -> Result<(), String> {
        use std::os::unix::ffi::OsStrExt;
        let c = std::ffi::CString::new(self.path.as_os_str().as_bytes())
            .map_err(|_| "a NUL in the path".to_owned())?;
        // SAFETY: `c` is a valid NUL-terminated path.
        if unsafe { libc::unlink(c.as_ptr()) } != 0 && errno() != libc::ENOENT {
            return Err(format!("unlink {}: errno {}", self.path.display(), errno()));
        }
        Ok(())
    }

    /// THE NEGATIVE CONTROL: every call in ONE slot (the hazard the global-parity law closes).
    #[doc(hidden)]
    #[must_use]
    pub fn control_one_slot(mut self) -> Self {
        self.slots = 1;
        self
    }

    /// The payload capacity of a slot (a frame wider than it must take another path; both ranks
    /// decide by the frame's length, which they share).
    #[must_use]
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// The calls made so far (the next call is `seq() + 1`).
    #[must_use]
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// The rank this view posts as.
    #[must_use]
    pub fn rank(&self) -> u32 {
        self.rank
    }

    /// Whether this view created the file.
    #[must_use]
    pub fn owner(&self) -> bool {
        self.owner
    }

    fn word(&self, off: usize) -> &AtomicU64 {
        assert!(
            off % 8 == 0 && off + 8 <= self.map_len,
            "pick line word {off}"
        );
        // SAFETY: `off` is 8-aligned inside the live mapping (page-aligned base), and every access
        // to the shared bytes goes through atomics, so the reference never aliases a plain access.
        unsafe { &*self.base.add(off).cast::<AtomicU64>() }
    }

    fn outbox(&self, rank: u32) -> usize {
        rank as usize * (ACK + 2 * Self::slot_bytes(self.cap))
    }

    fn slot_off(&self, rank: u32, seq: u64) -> usize {
        self.outbox(rank) + ACK + (seq % self.slots) as usize * Self::slot_bytes(self.cap)
    }

    /// Post this rank's frame of call `seq` (the next call). The caller's `post` + `take_with` with
    /// the same `seq` is one [`Self::exchange`].
    ///
    /// # Errors
    /// The frame is wider than the slot, or the peer's ack never reaches the slot's previous call.
    pub fn post(&mut self, seq: u64, mine: &[u8]) -> Result<(), String> {
        self.post_impl(seq, mine, true)
    }

    /// [`Self::post`] without the ack check (the negative control's tear only).
    ///
    /// # Errors
    /// The frame is wider than the slot.
    #[doc(hidden)]
    pub fn post_unchecked(&mut self, seq: u64, mine: &[u8]) -> Result<(), String> {
        self.post_impl(seq, mine, false)
    }

    fn post_impl(&mut self, seq: u64, mine: &[u8], check: bool) -> Result<(), String> {
        if mine.len() > self.cap {
            return Err(format!(
                "pick line: a {} B frame in a {} B slot",
                mine.len(),
                self.cap
            ));
        }
        // the slot's previous call (seq - slots) must have been copied out by the peer
        let prev = seq.saturating_sub(self.slots);
        let ack_off = self.outbox(1 - self.rank);
        if check && self.word(ack_off).load(Ordering::Acquire) < prev {
            self.counts.ack_waits += 1;
            let t = Instant::now();
            while self.word(ack_off).load(Ordering::Acquire) < prev {
                if t.elapsed() > self.timeout {
                    return Err(format!(
                        "pick line rank {}: the peer never copied call {prev} out (ack {})",
                        self.rank,
                        self.word(ack_off).load(Ordering::Acquire)
                    ));
                }
                std::hint::spin_loop();
            }
        }
        let s = self.slot_off(self.rank, seq);
        let n = mine.len();
        for (i, ch) in mine.chunks(8).enumerate() {
            let mut b = [0u8; 8];
            b[..ch.len()].copy_from_slice(ch);
            self.word(s + HEAD + 8 * i)
                .store(u64::from_le_bytes(b), Ordering::Relaxed);
        }
        self.word(s + HEAD + n.div_ceil(8) * 8)
            .store(seq, Ordering::Relaxed);
        self.word(s + 8).store(n as u64, Ordering::Relaxed);
        // no flag before data: the header's seq is the last store, released
        fence(Ordering::Release);
        self.word(s).store(seq, Ordering::Release);
        Ok(())
    }

    /// Take the peer's frame of call `seq` (`len` bytes); `mid` runs between the header's acquire
    /// and the copy (a test hook: the deterministic race of the negative control).
    ///
    /// # Errors
    /// A timeout, a header past `seq` or a stale one other than the slot's previous call, a length
    /// other than `len`, a trailer other than `seq`, or the slot rewritten under the copy (`torn`).
    pub fn take_with(
        &mut self,
        seq: u64,
        len: usize,
        mid: impl FnOnce(),
    ) -> Result<Vec<u8>, String> {
        let peer = 1 - self.rank;
        let s = self.slot_off(peer, seq);
        let mut h = self.word(s).load(Ordering::Acquire);
        if h != seq {
            self.counts.late += 1;
            let t = Instant::now();
            let mut spins = 0u64;
            loop {
                // the slot holds the peer's earlier call (or nothing) until it posts this one
                if h > seq || (h != 0 && h + self.slots != seq) {
                    return Err(format!(
                        "pick line rank {}: call {seq}'s slot holds call {h}",
                        self.rank
                    ));
                }
                if t.elapsed() > self.timeout {
                    return Err(format!(
                        "pick line rank {}: the peer never posted call {seq} (slot at {h})",
                        self.rank
                    ));
                }
                spins += 1;
                if spins > 4096 {
                    std::thread::yield_now();
                } else {
                    std::hint::spin_loop();
                }
                h = self.word(s).load(Ordering::Acquire);
                if h == seq {
                    break;
                }
            }
        }
        let n = self.word(s + 8).load(Ordering::Relaxed) as usize;
        if n != len {
            return Err(format!(
                "pick line rank {}: call {seq} carries {n} B, {len} B expected",
                self.rank
            ));
        }
        mid();
        let words = n.div_ceil(8);
        let mut out = Vec::with_capacity(words * 8);
        for i in 0..words {
            out.extend_from_slice(
                &self
                    .word(s + HEAD + 8 * i)
                    .load(Ordering::Relaxed)
                    .to_le_bytes(),
            );
        }
        out.truncate(n);
        let tr = self.word(s + HEAD + words * 8).load(Ordering::Relaxed);
        fence(Ordering::Acquire);
        let h2 = self.word(s).load(Ordering::Acquire);
        if h2 != seq || tr != seq {
            return Err(format!(
                "pick line rank {}: call {seq} torn under the copy (header {h2}, trailer {tr})",
                self.rank
            ));
        }
        // the ack: this rank is done with the peer's call `seq`
        let mine = self.outbox(self.rank);
        self.word(mine).store(seq, Ordering::Release);
        Ok(out)
    }

    /// One exchange: post this rank's frame, take the peer's (the same length: the pick frames of
    /// the pair are equal by construction), count it.
    ///
    /// # Errors
    /// [`Self::post`] / [`Self::take_with`]'s.
    pub fn exchange(&mut self, mine: &[u8]) -> Result<Vec<u8>, String> {
        let seq = self.seq + 1;
        self.post(seq, mine)?;
        let theirs = self.take_with(seq, mine.len(), || {})?;
        self.seq = seq;
        self.counts.calls += 1;
        self.counts.bytes += mine.len() as u64;
        Ok(theirs)
    }
}

impl Drop for ShmLine {
    fn drop(&mut self) {
        // SAFETY: `base` / `map_len` are this view's live mapping, unmapped exactly once.
        unsafe { libc::munmap(self.base.cast(), self.map_len) };
    }
}

// SAFETY: the mapping is process-wide memory touched only through atomics; a view moves between
// threads with its owner (the engine's main thread; the tests' two threads).
unsafe impl Send for ShmLine {}

#[cfg(test)]
mod tests {
    use super::*;
    use saber_test_scratch::ScratchDir;
    use std::sync::{Arc, Barrier, OnceLock};

    /// The lines' files live in one test-scratch directory (never /tmp: structural check 10).
    fn path(tag: &str) -> PathBuf {
        static DIR: OnceLock<ScratchDir> = OnceLock::new();
        DIR.get_or_init(|| ScratchDir::new("saber-fn117b-shmline"))
            .join(format!("line-{tag}"))
    }

    fn frame(rank: u32, seq: u64, len: usize) -> Vec<u8> {
        let mut s = seq.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (u64::from(rank) << 56);
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (s >> 33) as u8
            })
            .collect()
    }

    fn pair(tag: &str, cap: usize) -> (ShmLine, ShmLine) {
        let p = path(tag);
        let a = ShmLine::create(&p, cap, Duration::from_secs(20)).unwrap();
        let b = ShmLine::open(&p, cap, Duration::from_secs(20)).unwrap();
        a.unlink().unwrap();
        (a, b)
    }

    /// The law: 20,000 exchanges between two threads with random frame lengths (odd lengths
    /// included) and random skew on both sides -- every frame received is the peer's, byte for
    /// byte; the ack never had to be waited for.
    #[test]
    fn the_two_slot_line_carries_every_frame_under_skew() {
        let (a, b) = pair("law", 4096);
        let run = |mut l: ShmLine| {
            std::thread::spawn(move || {
                let r = l.rank();
                let mut s = 0x1234_5678u64 ^ u64::from(r);
                for seq in 1..=20_000u64 {
                    s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    if (s >> 60) == 0 {
                        std::thread::sleep(Duration::from_micros((s >> 40) % 300));
                    }
                    let len = 1 + (seq.wrapping_mul(2_654_435_761) % 4095) as usize;
                    let got = l.exchange(&frame(r, seq, len)).unwrap();
                    assert_eq!(got, frame(1 - r, seq, len), "rank {r} call {seq}");
                }
                l.counts
            })
        };
        let (ha, hb) = (run(a), run(b));
        let (ca, cb) = (ha.join().unwrap(), hb.join().unwrap());
        assert_eq!((ca.calls, cb.calls), (20_000, 20_000));
        assert_eq!((ca.ack_waits, cb.ack_waits), (0, 0));
        assert!(ca.late + cb.late > 0, "the skew made one side wait");
    }

    /// The race both forms are put through: rank 1 has acquired call 1's header and is about to
    /// copy when rank 0 (done with call 1) posts call 2. ONE slot (the control): rank 0's post lands
    /// in the slot under the copy -> rank 1's read is refused as torn (never returned). TWO slots
    /// (the law): call 2 goes to the other slot, rank 1 copies call 1 intact.
    fn race(one_slot: bool) -> Result<Vec<u8>, String> {
        let (mut a, mut b) = pair(if one_slot { "ctl" } else { "lawrace" }, 256);
        if one_slot {
            a = a.control_one_slot();
            b = b.control_one_slot();
        }
        let go = Arc::new(Barrier::new(2));
        let done = Arc::new(Barrier::new(2));
        let (g2, d2) = (go.clone(), done.clone());
        let t = std::thread::spawn(move || {
            // rank 0: call 1 complete (its peer posted), then call 2 posted while rank 1 copies
            a.post(1, &frame(0, 1, 200)).unwrap();
            let got = a.take_with(1, 200, || {}).unwrap();
            assert_eq!(got, frame(1, 1, 200));
            g2.wait();
            // the post without the ack check (the ack alone would make the one-slot form WAIT:
            // the control is the slot law's absence, as FN114d's shipped one-shot)
            let r = a.post_unchecked(2, &frame(0, 2, 200));
            d2.wait();
            r
        });
        b.post(1, &frame(1, 1, 200)).unwrap();
        let r = b.take_with(1, 200, || {
            go.wait();
            done.wait();
        });
        t.join().unwrap().unwrap();
        r
    }

    #[test]
    fn the_one_slot_control_tears_and_the_two_slot_law_does_not() {
        let law = race(false).unwrap();
        assert_eq!(law, frame(0, 1, 200), "the law copies call 1 intact");
        let ctl = race(true);
        assert!(
            ctl.as_ref().is_err_and(|e| e.contains("torn")),
            "the one-slot control must tear: {ctl:?}"
        );
    }

    #[test]
    fn a_frame_wider_than_the_slot_and_a_wrong_size_file_are_refused() {
        let (mut a, _b) = pair("wide", 64);
        assert!(a.post(1, &[0u8; 65]).unwrap_err().contains("slot"));
        let q = path("size");
        let c = ShmLine::create(&q, 64, Duration::from_secs(1)).unwrap();
        // a cap whose mapping is another size (64 and 128 both round to one 4 KiB page)
        assert!(ShmLine::open(&q, 8192, Duration::from_secs(1)).is_err());
        c.unlink().unwrap();
    }

    #[test]
    fn a_peer_that_never_posts_times_out_by_name() {
        let (mut a, _b) = pair("timeout", 64);
        a.timeout = Duration::from_millis(50);
        a.post(1, &[1u8; 8]).unwrap();
        let e = a.take_with(1, 8, || {}).unwrap_err();
        assert!(e.contains("never posted call 1"), "{e}");
    }
}
