//! FN106d: the page form of the pinned host memory (`--fn-pinned-pages shmem|huge`).
//!
//! `hipHostMalloc` (the [`PinnedPages::Shmem`] form, today's) is hsakmt's `MAP_SHARED |
//! MAP_ANONYMOUS` mapping: shmem in 4 KiB pages on this box (`shmem_enabled` = never, a knob only
//! root may move). [`PinnedPages::Huge`] is the no-root 2 MiB form FN106c verified: a PRIVATE
//! anonymous mapping, 2 MiB-aligned, `madvise(MADV_HUGEPAGE)` (THP `enabled` = madvise), faulted
//! in, then `hipHostRegister`ed (portable + mapped + coarse-grained, the class of `hipHostMalloc`'s
//! default) -- the device address is the host address (checked), the kernel maps the contiguous
//! 2 MiB runs with large GPU translation fragments.
//!
//! THE TEARDOWN LAW: registered memory is NOT `hipHostFree`d. Its owner (`PinnedBuf`'s drop, on
//! the HIP thread) calls `hipHostUnregister` FIRST and `munmap`s only after it returned; a failed
//! unregister LEAKS the mapping rather than unmap pages the driver may still hold.
//!
//! The fallback is by name: a huge allocation that cannot be made (mmap / madvise / register /
//! the identity check) is made as `hipHostMalloc` instead and counted with its reason
//! ([`PinnedStats`], the `fnpinned106d` line). Allocations under [`HUGE_MIN_BYTES`] stay
//! `hipHostMalloc` in both forms (a 2 MiB page each would waste the rest of it).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// The page form of the pinned host allocations a [`crate::HipThread`] makes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PinnedPages {
    /// `hipHostMalloc`: hsakmt's shmem mapping, 4 KiB pages on this box (the control; default).
    #[default]
    Shmem,
    /// A private anonymous 2 MiB-aligned mapping + `MADV_HUGEPAGE` + `hipHostRegister`.
    Huge,
    /// FN121: [`PinnedPages::Huge`] + `MADV_COLLAPSE` BEFORE the register -- every 4 KiB page the
    /// THP fault left (fragmentation) collapsed into 2 MiB pages, so no 4 KiB page sits inside a
    /// KFD-registered range for kcompactd / khugepaged to migrate later (a migration there is an
    /// MMU-notifier invalidation = every queue of the process evicted for tens of ms).
    HugeCollapse,
}

impl PinnedPages {
    /// Parse the flag's value (`shmem` | `huge`).
    ///
    /// # Errors
    /// Any other value, by name.
    pub fn parse(s: &str) -> Result<PinnedPages, String> {
        match s {
            "shmem" => Ok(PinnedPages::Shmem),
            "huge" => Ok(PinnedPages::Huge),
            "huge-collapse" => Ok(PinnedPages::HugeCollapse),
            o => Err(format!(
                "--fn-pinned-pages {o}: shmem | huge | huge-collapse"
            )),
        }
    }

    /// The flag's spelling.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            PinnedPages::Shmem => "shmem",
            PinnedPages::Huge => "huge",
            PinnedPages::HugeCollapse => "huge-collapse",
        }
    }

    /// Whether the form maps 2 MiB-aligned private memory (the huge forms).
    #[must_use]
    pub fn is_huge(self) -> bool {
        matches!(self, PinnedPages::Huge | PinnedPages::HugeCollapse)
    }

    pub(crate) fn as_u8(self) -> u8 {
        match self {
            PinnedPages::Shmem => 0,
            PinnedPages::Huge => 1,
            PinnedPages::HugeCollapse => 2,
        }
    }

    pub(crate) fn from_u8(v: u8) -> PinnedPages {
        match v {
            1 => PinnedPages::Huge,
            2 => PinnedPages::HugeCollapse,
            _ => PinnedPages::Shmem,
        }
    }
}

/// The THP page size.
pub const HUGE_PAGE: u64 = 2 << 20;
/// Allocations of at least this many bytes take the huge form under [`PinnedPages::Huge`].
pub const HUGE_MIN_BYTES: u64 = HUGE_PAGE;
/// `hipHostRegister` flags of the huge form: portable (0x1) + mapped (0x2) + coarse-grained
/// (`hipExtHostRegisterCoarseGrained`, 0x8) -- the memory class `hipHostMalloc`'s default gives.
pub const REGISTER_FLAGS: u32 = 0x1 | 0x2 | 0x8;

/// A 2 MiB-aligned private anonymous mapping (not yet registered).
#[derive(Debug)]
pub struct HugeMap {
    /// The aligned base.
    pub base: usize,
    /// The mapped length (the request rounded up to [`HUGE_PAGE`]).
    pub map_len: usize,
}

fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

/// Map `bytes` (rounded up to 2 MiB) of private anonymous memory at a 2 MiB-aligned base, advise
/// THP and fault it all in (`MADV_POPULATE_WRITE`; a write per 4 KiB page where the kernel lacks
/// it). The memory reads as zeros.
///
/// # Errors
/// The failing call by name (the caller's fallback reason).
pub fn map_huge(bytes: u64) -> Result<HugeMap, String> {
    if bytes == 0 {
        return Err("a huge mapping of 0 bytes".into());
    }
    let map_len = bytes.div_ceil(HUGE_PAGE) * HUGE_PAGE;
    let over = map_len + HUGE_PAGE;
    let (map_len, over) = (map_len as usize, over as usize);
    // SAFETY: a fresh anonymous mapping; no existing memory is touched.
    let raw = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            over,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if raw == libc::MAP_FAILED {
        return Err(format!("mmap of {over} B: errno {}", errno()));
    }
    let raw = raw as usize;
    let base = raw.div_ceil(HUGE_PAGE as usize) * HUGE_PAGE as usize;
    let head = base - raw;
    let tail = over - head - map_len;
    // SAFETY: both trims lie inside the mapping made above and outside [base, base + map_len).
    unsafe {
        if head > 0 {
            libc::munmap(raw as *mut libc::c_void, head);
        }
        if tail > 0 {
            libc::munmap((base + map_len) as *mut libc::c_void, tail);
        }
    }
    let m = HugeMap { base, map_len };
    // SAFETY: [base, base + map_len) is the live mapping made above.
    if unsafe { libc::madvise(base as *mut libc::c_void, map_len, libc::MADV_HUGEPAGE) } != 0 {
        let e = errno();
        unmap(&m);
        return Err(format!(
            "madvise(MADV_HUGEPAGE): errno {e} (THP unavailable)"
        ));
    }
    const MADV_POPULATE_WRITE: i32 = 23;
    // SAFETY: as above.
    if unsafe { libc::madvise(base as *mut libc::c_void, map_len, MADV_POPULATE_WRITE) } != 0 {
        // an older kernel: fault it in by hand (one write a 4 KiB page; the THP fault takes the
        // whole 2 MiB page at its first write)
        for off in (0..map_len).step_by(4096) {
            // SAFETY: inside the live, writable mapping.
            unsafe { std::ptr::write_volatile((base + off) as *mut u8, 0) };
        }
    }
    Ok(m)
}

/// FN121: `madvise(MADV_COLLAPSE)` over a [`HugeMap`] (before it is registered): every 4 KiB page
/// the THP fault left is collapsed into 2 MiB pages (synchronous; best effort -- up to `tries`
/// calls while the kernel says `EAGAIN` / `ENOMEM`). The bytes are unchanged (a collapse copies).
/// Returns (calls made, the last errno or 0).
pub fn collapse(m: &HugeMap, tries: u32) -> (u32, i32) {
    const MADV_COLLAPSE: i32 = 25;
    let mut last = 0;
    for k in 1..=tries.max(1) {
        // SAFETY: [base, base + map_len) is the live mapping `map_huge` made; not registered yet.
        if unsafe { libc::madvise(m.base as *mut libc::c_void, m.map_len, MADV_COLLAPSE) } == 0 {
            return (k, 0);
        }
        last = errno();
        if last != libc::EAGAIN && last != libc::ENOMEM {
            return (k, last);
        }
    }
    (tries.max(1), last)
}

/// Unmap a [`HugeMap`] (only after `hipHostUnregister` returned for it, if it was registered).
pub fn unmap(m: &HugeMap) {
    // SAFETY: the mapping [base, base + map_len) was made by `map_huge` and is no longer used.
    unsafe { libc::munmap(m.base as *mut libc::c_void, m.map_len) };
}

/// The process-wide census of pinned allocations (both forms), for the `fnpinned106d` line.
#[derive(Debug, Default)]
pub struct PinnedStats {
    /// `hipHostMalloc` bytes / allocations (live).
    pub shmem_bytes: AtomicU64,
    #[allow(missing_docs)]
    pub shmem_n: AtomicU64,
    /// Huge-form bytes (mapped length) / allocations (live).
    pub huge_bytes: AtomicU64,
    #[allow(missing_docs)]
    pub huge_n: AtomicU64,
    /// Huge-form allocations that FELL BACK to `hipHostMalloc` (bytes / count, cumulative).
    pub fallback_bytes: AtomicU64,
    #[allow(missing_docs)]
    pub fallback_n: AtomicU64,
    /// Time spent mapping + faulting + registering the huge form (ns, cumulative).
    pub huge_alloc_ns: AtomicU64,
    /// FN121 (`huge-collapse`): ranges collapsed, collapses that did not finish (the last errno
    /// kept in the reasons), THP-backed bytes of the collapsed ranges before / after (smaps), and
    /// the collapse's time (ns).
    pub collapse_n: AtomicU64,
    #[allow(missing_docs)]
    pub collapse_short: AtomicU64,
    #[allow(missing_docs)]
    pub collapse_thp_before: AtomicU64,
    #[allow(missing_docs)]
    pub collapse_thp_after: AtomicU64,
    #[allow(missing_docs)]
    pub collapse_bytes: AtomicU64,
    #[allow(missing_docs)]
    pub collapse_ns: AtomicU64,
    /// Huge-form allocations released (unregistered, then unmapped) / unregisters that failed
    /// (their mapping leaked).
    pub huge_freed: AtomicU64,
    #[allow(missing_docs)]
    pub huge_leaked: AtomicU64,
    reasons: Mutex<Vec<String>>,
    live: Mutex<Vec<(usize, usize)>>,
}

static STATS: std::sync::OnceLock<PinnedStats> = std::sync::OnceLock::new();

/// The process's census.
pub fn stats() -> &'static PinnedStats {
    STATS.get_or_init(PinnedStats::default)
}

impl PinnedStats {
    pub(crate) fn add_shmem(&self, bytes: u64) {
        self.shmem_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.shmem_n.fetch_add(1, Ordering::Relaxed);
    }
    pub(crate) fn sub_shmem(&self, bytes: u64) {
        self.shmem_bytes.fetch_sub(bytes, Ordering::Relaxed);
        self.shmem_n.fetch_sub(1, Ordering::Relaxed);
    }
    pub(crate) fn add_huge(&self, base: usize, map_len: usize, ns: u64) {
        self.huge_bytes.fetch_add(map_len as u64, Ordering::Relaxed);
        self.huge_n.fetch_add(1, Ordering::Relaxed);
        self.huge_alloc_ns.fetch_add(ns, Ordering::Relaxed);
        if let Ok(mut l) = self.live.lock() {
            l.push((base, map_len));
        }
    }
    pub(crate) fn sub_huge(&self, base: usize, map_len: usize, leaked: bool) {
        self.huge_bytes.fetch_sub(map_len as u64, Ordering::Relaxed);
        self.huge_n.fetch_sub(1, Ordering::Relaxed);
        if leaked {
            self.huge_leaked.fetch_add(1, Ordering::Relaxed);
        } else {
            self.huge_freed.fetch_add(1, Ordering::Relaxed);
        }
        if let Ok(mut l) = self.live.lock() {
            l.retain(|&(b, _)| b != base);
        }
    }
    pub(crate) fn add_collapse(
        &self,
        bytes: u64,
        before: u64,
        after: u64,
        ns: u64,
        short: Option<String>,
    ) {
        self.collapse_n.fetch_add(1, Ordering::Relaxed);
        self.collapse_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.collapse_thp_before
            .fetch_add(before, Ordering::Relaxed);
        self.collapse_thp_after.fetch_add(after, Ordering::Relaxed);
        self.collapse_ns.fetch_add(ns, Ordering::Relaxed);
        if let Some(r) = short {
            self.collapse_short.fetch_add(1, Ordering::Relaxed);
            if let Ok(mut v) = self.reasons.lock() {
                if !v.contains(&r) {
                    v.push(r);
                }
            }
        }
    }
    pub(crate) fn add_fallback(&self, bytes: u64, reason: String) {
        self.fallback_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.fallback_n.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut r) = self.reasons.lock() {
            if !r.contains(&reason) {
                r.push(reason);
            }
        }
    }

    /// The distinct fallback reasons so far.
    #[must_use]
    pub fn fallback_reasons(&self) -> Vec<String> {
        self.reasons.lock().map(|r| r.clone()).unwrap_or_default()
    }

    /// The live huge ranges `(base, map_len)`.
    #[must_use]
    pub fn live_huge(&self) -> Vec<(usize, usize)> {
        self.live.lock().map(|l| l.clone()).unwrap_or_default()
    }

    /// One line: the huge / 4 KiB split in bytes, how much of the huge form THP actually backs
    /// (this process's smaps), the fallbacks by name, the allocation time and /proc/meminfo's
    /// AnonHugePages / ShmemHugePages.
    #[must_use]
    pub fn line(&self, form: PinnedPages) -> String {
        let live = self.live_huge();
        let backed = smaps_huge_bytes(&live);
        let mi = meminfo_kib(&["AnonHugePages", "ShmemHugePages", "MemAvailable"]);
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        let hb = self.huge_bytes.load(Ordering::Relaxed);
        let reasons = self.fallback_reasons();
        let mut sizes: Vec<usize> = live.iter().map(|&(_, n)| n >> 20).collect();
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        sizes.truncate(10);
        format!(
            "form {} | huge {} allocs {:.3} GiB ({} B) MiB {:?}, THP-backed {} | 4K (hipHostMalloc) {} allocs \
             {:.3} GiB ({} B) | fallbacks {} ({} B){} | huge alloc {:.0} ms | freed {} leaked {} | \
             meminfo AnonHugePages {} kB ShmemHugePages {} kB MemAvailable {:.2} GiB{}",
            form.as_str(),
            self.huge_n.load(Ordering::Relaxed),
            gib(hb),
            hb,
            sizes,
            match backed {
                Some(b) => format!("{:.3} GiB ({:.1} %)", gib(b), 100.0 * b as f64 / hb.max(1) as f64),
                None => "n/a".into(),
            },
            self.shmem_n.load(Ordering::Relaxed),
            gib(self.shmem_bytes.load(Ordering::Relaxed)),
            self.shmem_bytes.load(Ordering::Relaxed),
            self.fallback_n.load(Ordering::Relaxed),
            self.fallback_bytes.load(Ordering::Relaxed),
            if reasons.is_empty() {
                String::new()
            } else {
                format!(" [{}]", reasons.join("; "))
            },
            self.huge_alloc_ns.load(Ordering::Relaxed) as f64 / 1e6,
            self.huge_freed.load(Ordering::Relaxed),
            self.huge_leaked.load(Ordering::Relaxed),
            mi[0].unwrap_or(0),
            mi[1].unwrap_or(0),
            mi[2].unwrap_or(0) as f64 / (1u64 << 20) as f64,
            if self.collapse_n.load(Ordering::Relaxed) > 0 {
                format!(
                    " | FN121 collapse: {} ranges {:.3} GiB, THP-backed {:.3} -> {:.3} GiB, {} short, {:.0} ms",
                    self.collapse_n.load(Ordering::Relaxed),
                    gib(self.collapse_bytes.load(Ordering::Relaxed)),
                    gib(self.collapse_thp_before.load(Ordering::Relaxed)),
                    gib(self.collapse_thp_after.load(Ordering::Relaxed)),
                    self.collapse_short.load(Ordering::Relaxed),
                    self.collapse_ns.load(Ordering::Relaxed) as f64 / 1e6
                )
            } else {
                String::new()
            }
        )
    }
}

/// /proc/meminfo values (kB) by key, in order.
#[must_use]
pub fn meminfo_kib(keys: &[&str]) -> Vec<Option<u64>> {
    let s = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    keys.iter()
        .map(|k| {
            s.lines().find_map(|l| {
                let (a, b) = l.split_once(':')?;
                (a == *k).then(|| b.split_whitespace().next()?.parse().ok())?
            })
        })
        .collect()
}

/// AnonHugePages (bytes) of this process's VMAs that overlap any of `ranges`, each VMA counted
/// once and capped at its overlap (`None` when smaps cannot be read).
#[must_use]
pub fn smaps_huge_bytes(ranges: &[(usize, usize)]) -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/smaps").ok()?;
    Some(smaps_huge_in(&s, ranges))
}

/// [`smaps_huge_bytes`] over smaps text (host-testable).
#[must_use]
pub fn smaps_huge_in(smaps: &str, ranges: &[(usize, usize)]) -> u64 {
    let mut tot = 0u64;
    let mut overlap = 0u64;
    for line in smaps.lines() {
        let head = line.split_whitespace().next().unwrap_or("");
        if let Some((a, b)) = head.split_once('-') {
            if let (Ok(a), Ok(b)) = (usize::from_str_radix(a, 16), usize::from_str_radix(b, 16)) {
                overlap = ranges
                    .iter()
                    .map(|&(r, n)| (b.min(r + n)).saturating_sub(a.max(r)) as u64)
                    .sum();
                continue;
            }
        }
        if let Some(v) = line.strip_prefix("AnonHugePages:") {
            if overlap > 0 {
                let kb: u64 = v
                    .split_whitespace()
                    .next()
                    .and_then(|x| x.parse().ok())
                    .unwrap_or(0);
                tot += (kb * 1024).min(overlap);
            }
        }
    }
    tot
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rss_kib() -> u64 {
        let s = std::fs::read_to_string("/proc/self/status").unwrap();
        s.lines()
            .find_map(|l| l.strip_prefix("VmRSS:"))
            .and_then(|v| v.split_whitespace().next()?.parse().ok())
            .unwrap()
    }

    #[test]
    fn the_flag_parses_by_name_and_shmem_is_the_default() {
        assert_eq!(PinnedPages::default(), PinnedPages::Shmem);
        assert_eq!(PinnedPages::parse("huge").unwrap(), PinnedPages::Huge);
        assert_eq!(PinnedPages::parse("shmem").unwrap(), PinnedPages::Shmem);
        assert!(PinnedPages::parse("2m")
            .unwrap_err()
            .contains("shmem | huge"));
        assert_eq!(
            PinnedPages::parse("huge-collapse").unwrap(),
            PinnedPages::HugeCollapse
        );
        assert!(PinnedPages::HugeCollapse.is_huge() && PinnedPages::Huge.is_huge());
        assert!(!PinnedPages::Shmem.is_huge());
        for p in [
            PinnedPages::Shmem,
            PinnedPages::Huge,
            PinnedPages::HugeCollapse,
        ] {
            assert_eq!(PinnedPages::from_u8(p.as_u8()), p);
            assert_eq!(PinnedPages::parse(p.as_str()).unwrap(), p);
        }
    }

    #[test]
    fn a_huge_mapping_is_aligned_zeroed_and_released() {
        let rss0 = rss_kib();
        for _ in 0..4 {
            let m = map_huge(64 << 20).unwrap();
            assert_eq!(m.base % HUGE_PAGE as usize, 0);
            assert_eq!(m.map_len, 64 << 20);
            // SAFETY: the live mapping just made.
            let s = unsafe { std::slice::from_raw_parts(m.base as *const u8, m.map_len) };
            assert!(s.iter().all(|&b| b == 0));
            assert!(rss_kib() >= rss0 + 60 * 1024, "populated");
            unmap(&m);
        }
        assert!(
            rss_kib() < rss0 + 16 * 1024,
            "RSS back to the baseline after the loop"
        );
        let odd = map_huge(3 << 20).unwrap();
        assert_eq!(odd.map_len, 4 << 20);
        unmap(&odd);
        assert!(map_huge(0).is_err());
        // FN121: a collapse leaves the bytes (zeros, then a pattern written before it) unchanged
        let m = map_huge(8 << 20).unwrap();
        // SAFETY: the live mapping just made.
        let s = unsafe { std::slice::from_raw_parts_mut(m.base as *mut u8, m.map_len) };
        for (i, b) in s.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        let (calls, _err) = collapse(&m, 3);
        assert!(calls >= 1);
        assert!(s.iter().enumerate().all(|(i, &b)| b == (i % 251) as u8));
        unmap(&m);
    }

    #[test]
    fn the_smaps_reader_counts_each_vma_once_capped_at_its_overlap() {
        let s = "7f0000000000-7f0000800000 rw-p 00000000 00:00 0\nRss: 8192 kB\nAnonHugePages:   8192 kB\n\
                 7f0000800000-7f0001000000 rw-p 00000000 00:00 0\nAnonHugePages:   4096 kB\n\
                 7f0002000000-7f0002200000 rw-s 00000000 00:01 5 /dev/zero (deleted)\nAnonHugePages: 0 kB\n";
        let base = 0x7f00_0000_0000usize;
        assert_eq!(smaps_huge_in(s, &[(base, 8 << 20)]), 8 << 20);
        assert_eq!(smaps_huge_in(s, &[(base + (2 << 20), 2 << 20)]), 2 << 20);
        assert_eq!(smaps_huge_in(s, &[(base, 16 << 20)]), 12 << 20);
        assert_eq!(smaps_huge_in(s, &[(0x7f00_0200_0000, 2 << 20)]), 0);
    }

    #[test]
    fn the_line_names_the_split_and_the_fallbacks() {
        let st = PinnedStats::default();
        st.add_shmem(4096);
        st.add_huge(0x1000_0000, 2 << 20, 1_000_000);
        st.add_fallback(8 << 20, "hipHostRegister: refused".into());
        let l = st.line(PinnedPages::Huge);
        assert!(
            l.contains("form huge") && l.contains("huge 1 allocs"),
            "{l}"
        );
        assert!(
            l.contains("4K (hipHostMalloc) 1 allocs") && l.contains("(4096 B)"),
            "{l}"
        );
        assert!(
            l.contains("fallbacks 1 (8388608 B) [hipHostRegister: refused]"),
            "{l}"
        );
        st.sub_huge(0x1000_0000, 2 << 20, false);
        st.sub_shmem(4096);
        assert!(st.live_huge().is_empty());
        assert!(st.line(PinnedPages::Huge).contains("freed 1 leaked 0"));
    }
}
