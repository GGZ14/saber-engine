//! FN45c: the tiered KV's NVMe tier (`kv.ssd`) and the PROMPT / CONVERSATION CACHE (`kv.pcache`).
//!
//! **The drive slab ([`KvSsd`]).** One preallocated file of `ssd_slots` blocks (one 16-token page
//! of one layer, 16 KiB at TP=2) on the model drive, `O_DIRECT`, bounded by `--fn-kv-ssd-gib`
//! (ledger line `kv.ssd`, `Pool::Nvme`). Blocks move between it and the pinned RAM slab only:
//! [`KvTier::commit`](crate::kv_tier::KvTier::commit) names the RAM -> drive writes (the
//! write-through to the lowest tier, issued after the flush that filled the RAM copy landed), and
//! [`KvTier::unpark`](crate::kv_tier::KvTier::unpark) / `gather` name the drive -> RAM reads.
//! Adjacent `(ram, ssd)` pairs coalesce into one transfer ([`ssd_runs`]); the runs spread over a
//! few threads (the drive's 14.8 GB/s needs QD >= 8).
//!
//! **The prompt cache ([`PromptCache`]).** A directory on the drive: one file per entry
//! (`e<id>.bin`, the engine's opaque snapshot of a sequence at a CHUNK BOUNDARY: its KV pages, its
//! GDN / conv / ring / index / PLE state and the drafter's stash -- the bytes the prefill wrote)
//! and an append-only `manifest` of fixed 128-byte records (`put` / `drop`, each with its own
//! checksum). An entry is keyed by [`prefix_keys`](crate::kv_tier::prefix_keys) (the chained
//! sha256 of the token prefix at that boundary). **Crash law:** the data file is written as
//! `.tmp`, synced, renamed, the directory synced, and only THEN is its `put` record appended and
//! synced -- so a half-written entry has no record and is never served; a torn record fails its
//! checksum; a record whose file is missing or of the wrong length, or whose sampled digest
//! differs at load, is dropped. Eviction is LRU on the byte budget (`--fn-kv-pcache-gib`).

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::dio::{self, OpenMode, DIO_ALIGN};

/// Threads a drive batch spreads over (the drive's measured QD for full rate).
pub const SSD_THREADS: usize = 16;

/// The drive slab of `kv.ssd`.
#[derive(Debug)]
pub struct KvSsd {
    file: File,
    path: PathBuf,
    slots: u32,
    block: u64,
}

/// One coalesced drive transfer: `n` blocks from RAM slot `ram` <-> drive slot `ssd`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SsdRun {
    /// First RAM slot.
    pub ram: u32,
    /// First drive slot.
    pub ssd: u32,
    /// Blocks.
    pub n: u32,
}

/// Coalesce `(ram, ssd)` pairs (in any order) into runs where both slots are consecutive.
#[must_use]
pub fn ssd_runs(pairs: &[(u32, u32)]) -> Vec<SsdRun> {
    let mut v = pairs.to_vec();
    v.sort_unstable();
    v.dedup();
    let mut out: Vec<SsdRun> = Vec::new();
    for (r, d) in v {
        if let Some(last) = out.last_mut() {
            if last.ram + last.n == r && last.ssd + last.n == d {
                last.n += 1;
                continue;
            }
        }
        out.push(SsdRun {
            ram: r,
            ssd: d,
            n: 1,
        });
    }
    out
}

impl KvSsd {
    /// Create (or reuse) the slab file at `path`: `slots` blocks of `block` bytes, preallocated.
    /// `mode` = `Direct` on the drive (`Buffered` for a filesystem without `O_DIRECT`).
    pub fn create(path: &Path, slots: u32, block: u64, mode: OpenMode) -> io::Result<KvSsd> {
        if block == 0 || block % DIO_ALIGN as u64 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("kv.ssd block {block} is not a multiple of {DIO_ALIGN}"),
            ));
        }
        let file = dio::open_rw_create(path, mode)?;
        let len = u64::from(slots) * block;
        if len > 0 {
            dio::fallocate(&file, len)?;
        }
        Ok(KvSsd {
            file,
            path: path.to_path_buf(),
            slots,
            block,
        })
    }

    /// The file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Bytes on the drive.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        u64::from(self.slots) * self.block
    }

    fn check(&self, run: &SsdRun, slab_len: usize, off: u64) -> io::Result<(usize, u64, usize)> {
        let n = u64::from(run.n) * self.block;
        let at = off + u64::from(run.ram) * self.block;
        if u64::from(run.ssd) + u64::from(run.n) > u64::from(self.slots) || at + n > slab_len as u64
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("kv.ssd run {run:?} outside the slab / the drive"),
            ));
        }
        Ok((at as usize, u64::from(run.ssd) * self.block, n as usize))
    }

    /// RAM -> drive: `pairs` = `(ram slot, ssd slot)` of the RAM slab `slab` whose slot 0 starts
    /// at byte `off`. Returns the runs written.
    pub fn write(&self, slab: &[u8], off: u64, pairs: &[(u32, u32)]) -> io::Result<usize> {
        let runs = ssd_runs(pairs);
        let mut jobs = Vec::with_capacity(runs.len());
        for r in &runs {
            let (at, d, n) = self.check(r, slab.len(), off)?;
            jobs.push((&slab[at..at + n], d));
        }
        let per = jobs.len().div_ceil(SSD_THREADS).max(1);
        std::thread::scope(|sc| {
            let hs: Vec<_> = jobs
                .chunks(per)
                .map(|c| {
                    sc.spawn(move || {
                        c.iter()
                            .try_for_each(|&(b, d)| dio::pwritev_exact(&self.file, b, d))
                    })
                })
                .collect();
            hs.into_iter()
                .try_for_each(|h| h.join().expect("kv.ssd writer panicked"))
        })?;
        Ok(runs.len())
    }

    /// Drive -> RAM: `pairs` = `(ssd slot, ram slot)` (the tier's `ssd_reads` order).
    pub fn read(&self, slab: &mut [u8], off: u64, pairs: &[(u32, u32)]) -> io::Result<usize> {
        let flipped: Vec<(u32, u32)> = pairs.iter().map(|&(d, r)| (r, d)).collect();
        let runs = ssd_runs(&flipped);
        let len = slab.len();
        let mut spans = Vec::with_capacity(runs.len());
        for r in &runs {
            spans.push(self.check(r, len, off)?);
        }
        // disjoint mutable slices in RAM order (the runs are sorted by RAM slot and deduped)
        let mut jobs: Vec<(&mut [u8], u64)> = Vec::with_capacity(spans.len());
        let mut rest = slab;
        let mut base = 0usize;
        for (at, d, n) in spans {
            if at < base {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "kv.ssd reads overlap in RAM",
                ));
            }
            let (_, tail) = rest.split_at_mut(at - base);
            let (mine, tail) = tail.split_at_mut(n);
            jobs.push((mine, d));
            rest = tail;
            base = at + n;
        }
        let nr = jobs.len();
        let per = nr.div_ceil(SSD_THREADS).max(1);
        std::thread::scope(|sc| {
            let mut hs = Vec::new();
            let mut it = jobs.into_iter().peekable();
            while it.peek().is_some() {
                let c: Vec<(&mut [u8], u64)> = it.by_ref().take(per).collect();
                let f = &self.file;
                hs.push(sc.spawn(move || {
                    c.into_iter()
                        .try_for_each(|(b, d)| dio::preadv_exact(f, b, d))
                }));
            }
            hs.into_iter()
                .try_for_each(|h| h.join().expect("kv.ssd reader panicked"))
        })?;
        Ok(nr)
    }
}

/// Bytes of one manifest record.
pub const RECORD: usize = 128;
const MAGIC: u32 = 0x4350_4e46; // "FNPC"
const VERSION: u32 = 1;
const PUT: u8 = 1;
const DROP: u8 = 2;
/// The sampled digest reads 4 KiB every this many bytes (and the whole tail).
pub const DIGEST_STRIDE: usize = 1 << 20;

/// The sampled digest of an entry: sha256 over its length and 4 KiB of every 1 MiB (cheap at
/// drive rate; the crash law rests on the write order, the digest catches a damaged file).
#[must_use]
pub fn sampled_digest(data: &[u8]) -> [u8; 32] {
    sampled_digest_parts(&[data])
}

/// One live entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheEntry {
    /// File id.
    pub id: u64,
    /// The prefix length (a chunk boundary).
    pub tokens: u32,
    /// Bytes.
    pub len: u64,
    /// The sampled digest.
    pub digest: [u8; 32],
    /// Recency (larger = more recent).
    pub last: u64,
}

/// What a cache operation did (the `fnpc` counters).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheCounts {
    /// Entries stored.
    pub stores: u64,
    /// Lookups that found a prefix.
    pub hits: u64,
    /// Lookups that found none.
    pub misses: u64,
    /// Entries evicted by the budget.
    pub evictions: u64,
    /// Entries dropped as damaged (missing file, wrong length, digest).
    pub damaged: u64,
    /// Tokens of prefill the hits skipped.
    pub hit_tokens: u64,
}

/// The prompt / conversation cache (module doc).
#[derive(Debug)]
pub struct PromptCache {
    dir: PathBuf,
    budget: u64,
    used: u64,
    next_id: u64,
    clock: u64,
    map: BTreeMap<[u8; 32], CacheEntry>,
    /// FN45c4: the manifest file behind its OWN lock (an append is a page-cache write under it;
    /// the fsync of a `put` runs on a clone of the fd with no lock held -- [`write_record`]).
    manifest: std::sync::Arc<std::sync::Mutex<File>>,
    /// FN45c3: keys reserved and being written (not served until committed).
    pending: std::collections::BTreeSet<[u8; 32]>,
    /// Counters.
    pub counts: CacheCounts,
}

/// FN45c3: a budget reservation of the async store ([`PromptCache::reserve`]); the entry's file is
/// written by [`write_entry`] OUTSIDE the cache's lock, then [`PromptCache::commit`]ted.
#[derive(Debug)]
pub struct Reserved {
    key: [u8; 32],
    id: u64,
    tokens: u32,
    len: u64,
    tmp: PathBuf,
    bin: PathBuf,
    dir: PathBuf,
}

/// FN45c3 (the ASYNC store, step 2 of 3, no lock held): write `parts` as the reserved entry's file
/// (whole 4 KiB blocks, `.tmp` -> fsync -> rename -> directory fsync) and return its sampled
/// digest. The parts are written in place (no concatenated copy).
pub fn write_entry(r: &Reserved, parts: &[&[u8]]) -> io::Result<[u8; 32]> {
    let len: u64 = parts.iter().map(|p| p.len() as u64).sum();
    if len != r.len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "a reservation of another length",
        ));
    }
    {
        let mut f = File::create(&r.tmp)?;
        for p in parts {
            f.write_all(p)?;
        }
        // the file is whole 4 KiB blocks (an O_DIRECT load); the record keeps the true length
        f.write_all(&vec![0u8; (dio::align_up(len) - len) as usize])?;
        f.sync_all()?;
        drop_cached(&f);
    }
    let d = sampled_digest_parts(parts);
    fs::rename(&r.tmp, &r.bin)?;
    sync_dir(&r.dir)?;
    Ok(d)
}

/// FN45c4: drop a synced file's pages from the page cache (the entries are read back with
/// `O_DIRECT`; a buffered write left them resident, growing the cache against the engine's
/// anonymous memory on a host at its RAM ceiling). Advisory: an error is ignored.
fn drop_cached(f: &File) {
    use std::os::fd::AsRawFd;
    // SAFETY: a valid open fd; posix_fadvise only advises the kernel.
    unsafe {
        libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
    }
}

/// FN45c4 (the ASYNC store, step 3a, NO cache lock): append and fsync the reserved entry's `put`
/// record (the crash law: the record is durable before [`PromptCache::publish`] makes the entry
/// servable). The manifest's own lock covers the append only; the fsync runs on a clone of the fd.
pub fn write_record(
    manifest: &std::sync::Mutex<File>,
    r: &Reserved,
    digest: [u8; 32],
) -> io::Result<()> {
    let e = r.entry(digest, 0);
    let f = {
        let mut m = manifest
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        m.write_all(&record(PUT, &r.key, &e))?;
        m.try_clone()?
    };
    f.sync_data()
}

impl Reserved {
    fn entry(&self, digest: [u8; 32], last: u64) -> CacheEntry {
        CacheEntry {
            id: self.id,
            tokens: self.tokens,
            len: self.len,
            digest,
            last,
        }
    }
}

/// FN45c4: read and check an entry's bytes (NO cache lock: the path and record come from
/// [`PromptCache::entry_file`]); `None` = missing, short or a digest mismatch.
#[must_use]
pub fn read_checked(path: &Path, e: &CacheEntry) -> Option<dio::AlignedBuf> {
    read_direct(path, e.len)
        .ok()
        .filter(|b| sampled_digest(b) == e.digest)
}

/// [`sampled_digest`] of the concatenation of `parts` (without concatenating).
#[must_use]
pub fn sampled_digest_parts(parts: &[&[u8]]) -> [u8; 32] {
    let len: usize = parts.iter().map(|p| p.len()).sum();
    let feed = |h: &mut Sha256, mut o: usize, mut n: usize| {
        for p in parts {
            if o >= p.len() {
                o -= p.len();
                continue;
            }
            let take = n.min(p.len() - o);
            h.update(&p[o..o + take]);
            n -= take;
            o = 0;
            if n == 0 {
                break;
            }
        }
    };
    let mut h = Sha256::new();
    h.update((len as u64).to_le_bytes());
    let mut at = 0usize;
    while at < len {
        let end = (at + DIO_ALIGN).min(len);
        feed(&mut h, at, end - at);
        at += DIGEST_STRIDE;
    }
    let tail = len.saturating_sub(DIO_ALIGN);
    feed(&mut h, tail, len - tail);
    h.finalize().into()
}

fn record(kind: u8, key: &[u8; 32], e: &CacheEntry) -> [u8; RECORD] {
    let mut r = [0u8; RECORD];
    r[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    r[4..8].copy_from_slice(&VERSION.to_le_bytes());
    r[8] = kind;
    r[16..24].copy_from_slice(&e.id.to_le_bytes());
    r[24..28].copy_from_slice(&e.tokens.to_le_bytes());
    r[32..40].copy_from_slice(&e.len.to_le_bytes());
    r[40..72].copy_from_slice(key);
    r[72..104].copy_from_slice(&e.digest);
    let c: [u8; 32] = Sha256::digest(&r[..104]).into();
    r[104..128].copy_from_slice(&c[..24]);
    r
}

fn parse(r: &[u8]) -> Option<(u8, [u8; 32], CacheEntry)> {
    if r.len() != RECORD || r[0..4] != MAGIC.to_le_bytes() || r[4..8] != VERSION.to_le_bytes() {
        return None;
    }
    let c: [u8; 32] = Sha256::digest(&r[..104]).into();
    if r[104..128] != c[..24] {
        return None;
    }
    let u64at = |o: usize| u64::from_le_bytes(r[o..o + 8].try_into().expect("8"));
    let key: [u8; 32] = r[40..72].try_into().expect("32");
    Some((
        r[8],
        key,
        CacheEntry {
            id: u64at(16),
            tokens: u32::from_le_bytes(r[24..28].try_into().expect("4")),
            len: u64at(32),
            digest: r[72..104].try_into().expect("32"),
            last: 0,
        },
    ))
}

/// Read the first `len` bytes of an entry file of `align_up(len)` bytes: `O_DIRECT` in 8 MiB
/// pieces over [`SSD_THREADS`] threads (buffered where the filesystem refuses `O_DIRECT`).
pub fn read_direct(path: &Path, len: u64) -> io::Result<dio::AlignedBuf> {
    let whole = dio::align_up(len);
    if fs::metadata(path)?.len() != whole {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "entry length"));
    }
    let f = dio::open_read(path, OpenMode::Direct)
        .or_else(|_| dio::open_read(path, OpenMode::Buffered))?;
    let mut buf = dio::AlignedBuf::zeroed(whole as usize);
    const PIECE: usize = 8 << 20;
    let pieces: Vec<(usize, &mut [u8])> = buf.chunks_mut(PIECE).enumerate().collect();
    let per = pieces.len().div_ceil(SSD_THREADS).max(1);
    std::thread::scope(|sc| {
        let mut hs = Vec::new();
        let mut it = pieces.into_iter().peekable();
        while it.peek().is_some() {
            let c: Vec<(usize, &mut [u8])> = it.by_ref().take(per).collect();
            let f = &f;
            hs.push(sc.spawn(move || {
                c.into_iter()
                    .try_for_each(|(i, b)| dio::preadv_exact(f, b, (i * PIECE) as u64))
            }));
        }
        hs.into_iter()
            .try_for_each(|h| h.join().expect("prompt-cache reader panicked"))
    })?;
    buf.truncate(len as usize);
    Ok(buf)
}

fn sync_dir(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}

impl PromptCache {
    /// Open (or create) the cache at `dir` with a byte `budget`: replays the manifest, keeps the
    /// entries whose file is present at its recorded length, deletes every other file, rewrites
    /// the manifest compacted, and evicts down to the budget.
    pub fn open(dir: &Path, budget: u64) -> io::Result<PromptCache> {
        fs::create_dir_all(dir)?;
        let mpath = dir.join("manifest");
        let mut raw = Vec::new();
        if let Ok(mut f) = File::open(&mpath) {
            f.read_to_end(&mut raw)?;
        }
        let mut map: BTreeMap<[u8; 32], CacheEntry> = BTreeMap::new();
        let mut next_id = 0u64;
        let mut clock = 0u64;
        let mut damaged = 0u64;
        for r in raw.chunks(RECORD) {
            let Some((kind, key, mut e)) = parse(r) else {
                continue; // a torn or foreign record is never served
            };
            next_id = next_id.max(e.id + 1);
            match kind {
                PUT => {
                    clock += 1;
                    e.last = clock;
                    map.insert(key, e);
                }
                DROP if map.get(&key).is_some_and(|x| x.id == e.id) => {
                    map.remove(&key);
                }
                _ => {}
            }
        }
        map.retain(|_, e| {
            let ok = fs::metadata(dir.join(format!("e{:016x}.bin", e.id)))
                .is_ok_and(|m| m.len() == dio::align_up(e.len));
            damaged += u64::from(!ok);
            ok
        });
        let live: std::collections::BTreeSet<String> = map
            .values()
            .map(|e| format!("e{:016x}.bin", e.id))
            .collect();
        for d in fs::read_dir(dir)? {
            let name = d?.file_name().to_string_lossy().into_owned();
            if name.starts_with('e') && !live.contains(&name) {
                fs::remove_file(dir.join(&name))?;
            }
        }
        // the compacted manifest: written aside, synced, renamed over
        let tmp = dir.join("manifest.tmp");
        {
            let mut f = File::create(&tmp)?;
            let mut v: Vec<(&[u8; 32], &CacheEntry)> = map.iter().collect();
            v.sort_by_key(|(_, e)| e.last);
            for (k, e) in v {
                f.write_all(&record(PUT, k, e))?;
            }
            f.sync_all()?;
        }
        fs::rename(&tmp, &mpath)?;
        sync_dir(dir)?;
        let manifest = std::sync::Arc::new(std::sync::Mutex::new(
            OpenOptions::new().append(true).open(&mpath)?,
        ));
        let used = map.values().map(|e| e.len).sum();
        let mut pc = PromptCache {
            dir: dir.to_path_buf(),
            budget,
            used,
            next_id,
            clock,
            map,
            manifest,
            pending: std::collections::BTreeSet::new(),
            counts: CacheCounts {
                damaged,
                ..CacheCounts::default()
            },
        };
        pc.evict_to(budget)?;
        Ok(pc)
    }

    /// Bytes stored.
    #[must_use]
    pub fn used(&self) -> u64 {
        self.used
    }

    /// The byte budget.
    #[must_use]
    pub fn budget(&self) -> u64 {
        self.budget
    }

    /// Live entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// No entry.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// The entry stored at `key`.
    #[must_use]
    pub fn get(&self, key: &[u8; 32]) -> Option<&CacheEntry> {
        self.map.get(key)
    }

    fn path(&self, id: u64, ext: &str) -> PathBuf {
        self.dir.join(format!("e{id:016x}.{ext}"))
    }

    /// FN45c4: the manifest's handle (the writer thread's [`write_record`]).
    #[must_use]
    pub fn manifest(&self) -> std::sync::Arc<std::sync::Mutex<File>> {
        std::sync::Arc::clone(&self.manifest)
    }

    /// A `drop` record: appended, NOT synced (FN45c4: a lost `drop` is harmless -- the entry's
    /// file is removed after it, and a `put` whose file is missing is dropped at open).
    fn append(&mut self, r: &[u8; RECORD]) -> io::Result<()> {
        self.manifest
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .write_all(r)
    }

    fn drop_entry(&mut self, key: &[u8; 32]) -> io::Result<()> {
        if let Some(e) = self.map.remove(key) {
            self.append(&record(DROP, key, &e))?;
            self.used -= e.len;
            let _ = fs::remove_file(self.path(e.id, "bin"));
        }
        Ok(())
    }

    fn evict_to(&mut self, cap: u64) -> io::Result<()> {
        while self.used > cap {
            let Some(k) = self.map.iter().min_by_key(|(_, e)| e.last).map(|(k, _)| *k) else {
                break;
            };
            self.drop_entry(&k)?;
            self.counts.evictions += 1;
        }
        Ok(())
    }

    /// Store `parts` (concatenated) at `key` (`tokens` = the prefix length). An entry over the
    /// whole budget is refused (`Ok(false)`); a key already stored is kept (`Ok(false)`: the
    /// bytes are the same by the exactness law) and touched. LRU entries are evicted to fit.
    /// = [`PromptCache::reserve`] + [`write_entry`] + [`PromptCache::commit`] in one call.
    pub fn store(&mut self, key: [u8; 32], tokens: u32, parts: &[&[u8]]) -> io::Result<bool> {
        let len: u64 = parts.iter().map(|p| p.len() as u64).sum();
        let Some(r) = self.reserve(key, tokens, len)? else {
            return Ok(false);
        };
        match write_entry(&r, parts) {
            Ok(d) => self.commit(r, d).map(|()| true),
            Err(e) => {
                self.abort(&r);
                Err(e)
            }
        }
    }

    /// FN45c3 (the ASYNC store, step 1 of 3, under the caller's lock): claim `len` bytes of the
    /// budget for `key` (LRU entries evicted to fit) and a file id. `None` = nothing to write (the
    /// key is stored -- touched -- or already reserved, or `len` exceeds the whole budget).
    pub fn reserve(
        &mut self,
        key: [u8; 32],
        tokens: u32,
        len: u64,
    ) -> io::Result<Option<Reserved>> {
        if len > self.budget || self.pending.contains(&key) {
            return Ok(None);
        }
        if self.map.contains_key(&key) {
            self.clock += 1;
            let c = self.clock;
            self.map.get_mut(&key).expect("present").last = c;
            return Ok(None);
        }
        self.evict_to(self.budget - len)?;
        let id = self.next_id;
        self.next_id += 1;
        self.used += len;
        self.pending.insert(key);
        Ok(Some(Reserved {
            key,
            id,
            tokens,
            len,
            tmp: self.path(id, "tmp"),
            bin: self.path(id, "bin"),
            dir: self.dir.clone(),
        }))
    }

    /// FN45c3 (step 3, under the lock): publish a written entry (its `put` record appended and
    /// synced -- the crash law: only now can it be served).
    /// = [`write_record`] + [`PromptCache::publish`] (FN45c4: the async store calls the two
    /// apart, the fsync outside the cache's lock).
    pub fn commit(&mut self, r: Reserved, digest: [u8; 32]) -> io::Result<()> {
        let m = self.manifest();
        if let Err(x) = write_record(&m, &r, digest) {
            self.abort(&r);
            return Err(x);
        }
        self.publish(r, digest);
        Ok(())
    }

    /// FN45c4 (step 3b, under the lock, NO I/O): make a written entry whose `put` record is durable
    /// servable (the in-memory index the probe reads).
    pub fn publish(&mut self, r: Reserved, digest: [u8; 32]) {
        self.pending.remove(&r.key);
        self.clock += 1;
        let e = r.entry(digest, self.clock);
        self.map.insert(r.key, e);
        self.counts.stores += 1;
    }

    /// FN45c4: the file and record of `key` (the read runs outside the lock: [`read_checked`]).
    #[must_use]
    pub fn entry_file(&self, key: &[u8; 32]) -> Option<(PathBuf, CacheEntry)> {
        self.map
            .get(key)
            .map(|e| (self.path(e.id, "bin"), e.clone()))
    }

    /// FN45c4: after a read outside the lock -- touch the entry (`ok`) or drop it as damaged
    /// (only if it is still the same file).
    pub fn settle(&mut self, key: &[u8; 32], id: u64, ok: bool) -> io::Result<()> {
        if self.map.get(key).is_none_or(|e| e.id != id) {
            return Ok(());
        }
        if ok {
            self.clock += 1;
            let c = self.clock;
            self.map.get_mut(key).expect("present").last = c;
            Ok(())
        } else {
            self.counts.damaged += 1;
            self.drop_entry(key)
        }
    }

    /// FN45c3: give a reservation back (its write failed).
    pub fn abort(&mut self, r: &Reserved) {
        self.pending.remove(&r.key);
        self.used -= r.len;
        let _ = fs::remove_file(&r.tmp);
        let _ = fs::remove_file(&r.bin);
    }

    /// FN45c3: the distinct prefix lengths of the stored entries in `lo..=hi`, longest first (the
    /// candidates of a conversation's longest stored prefix; each is confirmed by its key).
    #[must_use]
    pub fn lengths_in(&self, lo: u32, hi: u32) -> Vec<u32> {
        let mut v: Vec<u32> = self
            .map
            .values()
            .map(|e| e.tokens)
            .filter(|t| (lo..=hi).contains(t))
            .collect();
        v.sort_unstable_by(|a, b| b.cmp(a));
        v.dedup();
        v
    }

    /// The longest stored prefix of a request whose chunk-boundary keys are `keys` (from
    /// [`prefix_keys`](crate::kv_tier::prefix_keys), ascending): `(tokens, key)`.
    pub fn lookup(&mut self, keys: &[(u32, [u8; 32])]) -> Option<(u32, [u8; 32])> {
        let hit = keys
            .iter()
            .rev()
            .find(|(t, k)| self.map.get(k).is_some_and(|e| e.tokens == *t))
            .copied();
        match hit {
            Some((t, _)) => {
                self.counts.hits += 1;
                self.counts.hit_tokens += u64::from(t);
            }
            None => self.counts.misses += 1,
        }
        hit
    }

    /// The bytes stored at `key` (touched). A damaged entry (missing, wrong length, digest) is
    /// dropped and refused: the caller prefills instead.
    pub fn load(&mut self, key: &[u8; 32]) -> io::Result<dio::AlignedBuf> {
        let (path, e) = self
            .entry_file(key)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no prompt-cache entry"))?;
        let got = read_checked(&path, &e);
        self.settle(key, e.id, got.is_some())?;
        got.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "a damaged prompt-cache entry (dropped)",
            )
        })
    }

    /// The `fnpc` line.
    #[must_use]
    pub fn line(&self) -> String {
        let c = self.counts;
        format!(
            "fnpc: {} entries {:.2} of {:.2} GiB; stores {} hits {} ({} tokens skipped) misses {} \
             evictions {} damaged {}",
            self.map.len(),
            self.used as f64 / f64::from(1u32 << 30),
            self.budget as f64 / f64::from(1u32 << 30),
            c.stores,
            c.hits,
            c.hit_tokens,
            c.misses,
            c.evictions,
            c.damaged
        )
    }
}
