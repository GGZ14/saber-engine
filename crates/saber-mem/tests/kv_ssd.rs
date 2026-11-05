//! FN45c: the NVMe KV tier under the engine's RESIDENT law (park / unpark, admission by request
//! length, the RAM slab a warm cache over the drive), the drive slab's round trip, and the prompt
//! cache's crash law.

use std::fs;
use std::path::PathBuf;

use saber_mem::device_mover::CopyOp;
use saber_mem::dio::{AlignedBuf, OpenMode};
use saber_mem::kv_ssd::{
    read_checked, sampled_digest, ssd_runs, write_entry, write_record, KvSsd, PromptCache, SsdRun,
    RECORD,
};
use saber_mem::kv_tier::{prefix_keys, Commit, Gather, KvTier, KvTierError, KvTierPlan, Place};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

const TB: usize = 2;

/// Bytes of every slot of every tier + the flat reference.
struct Model {
    bt: usize,
    vram: Vec<Vec<u8>>,
    ram: Vec<Vec<u8>>,
    ssd: Vec<Vec<u8>>,
    flat: Vec<Vec<Vec<u8>>>,
    lose_ssd_writes: bool,
}

impl Model {
    fn new(p: &KvTierPlan) -> Model {
        let bt = p.block_tokens as usize;
        Model {
            bt,
            vram: vec![vec![0xAA; bt * TB]; p.hot_slots as usize],
            ram: vec![vec![0x55; bt * TB]; p.ram_slots as usize],
            ssd: vec![vec![0x33; bt * TB]; p.ssd_slots as usize],
            flat: vec![vec![Vec::new(); p.layers as usize]; p.max_seqs as usize + 1],
            lose_ssd_writes: false,
        }
    }
    fn ops(&mut self, ops: &[CopyOp]) {
        for op in ops {
            match *op {
                CopyOp::H2d { ram, vram } => {
                    self.vram[vram as usize] = self.ram[ram as usize].clone()
                }
                CopyOp::D2h { vram, ram } => {
                    self.ram[ram as usize] = self.vram[vram as usize].clone()
                }
                CopyOp::Landed { .. } => {}
            }
        }
    }
    fn commit(&mut self, c: &Commit) {
        self.ops(&c.flush);
        if !self.lose_ssd_writes {
            for &(r, d) in &c.ssd_writes {
                self.ssd[d as usize] = self.ram[r as usize].clone();
            }
        }
    }
    fn gather(&mut self, g: &Gather) {
        for &(d, r) in &g.ssd_reads {
            self.ram[r as usize] = self.ssd[d as usize].clone();
        }
        self.ops(&g.ops);
    }
    fn write(&mut self, t: &KvTier, seq: u32, old: u32, new: u32, salt: u64) {
        for layer in 0..t.plan().layers {
            for pos in old..new {
                let blk = pos as usize / self.bt;
                let off = (pos as usize % self.bt) * TB;
                let bytes: Vec<u8> = (0..TB)
                    .map(|i| {
                        (salt.wrapping_mul(31)
                            + u64::from(seq) * 7
                            + u64::from(layer) * 13
                            + u64::from(pos) * 3
                            + i as u64)
                            .rem_euclid(251) as u8
                    })
                    .collect();
                let Place::Vram(v) = t.place(seq, layer, blk as u32).unwrap() else {
                    panic!("a written block is not hot");
                };
                self.vram[v as usize][off..off + TB].copy_from_slice(&bytes);
                self.flat[seq as usize][layer as usize].extend_from_slice(&bytes);
            }
        }
    }
    /// Every live sequence's bytes through its placement == the flat reference; and (the
    /// engine's law) an ACTIVE sequence never reads the drive.
    fn law(&self, t: &KvTier) -> Result<(), String> {
        for seq in 0..=t.plan().max_seqs {
            if !t.is_live(seq) {
                continue;
            }
            let len = t.len(seq) as usize;
            for layer in 0..t.plan().layers {
                let flat = &self.flat[seq as usize][layer as usize];
                for blk in 0..len.div_ceil(self.bt) {
                    let n = (len - blk * self.bt).min(self.bt) * TB;
                    let pl = t.place(seq, layer, blk as u32).unwrap();
                    let got = match pl {
                        Place::Vram(v) => &self.vram[v as usize][..n],
                        Place::Ram(r) => &self.ram[r as usize][..n],
                        Place::Ssd(d) => {
                            if !t.is_parked(seq) {
                                return Err(format!("active {seq}/{layer}/{blk} on the drive"));
                            }
                            &self.ssd[d as usize][..n]
                        }
                    };
                    let at = blk * self.bt * TB;
                    if got != &flat[at..at + n] {
                        return Err(format!("seq {seq} layer {layer} block {blk} {pl:?}"));
                    }
                }
            }
        }
        Ok(())
    }
}

/// 3 layers, 16-token blocks, window 32, chunk 64, 4 x 512; RAM 100 blocks (< the 384 of every
/// context: a warm cache over the drive), drive = every block.
fn plan() -> KvTierPlan {
    let bb = 16 * 1024u64;
    KvTierPlan::new(3, 16, 32, 64, 4, 512, 72 * bb, 100 * bb, 384 * bb).unwrap()
}

struct Stats {
    parks: u64,
    unparks: u64,
    ssd_reads: u64,
    refused: u64,
    ram_drops: u64,
}

/// The engine's life of a sequence at random: admit by request length (refused -> park the
/// coldest active sequence and retry later), prefill chunks, decode steps, park, unpark, finish.
fn drive(seed: u64, rounds: usize, lose: bool) -> Result<Stats, String> {
    let p = plan();
    let mut t = KvTier::new(p.clone()).resident().unwrap();
    let mut m = Model::new(&p);
    m.lose_ssd_writes = lose;
    let mut rng = Rng(seed * 0x9E37_79B9 + 1);
    let mut want = [0u32; 5];
    let mut st = Stats {
        parks: 0,
        unparks: 0,
        ssd_reads: 0,
        refused: 0,
        ram_drops: 0,
    };
    let mut salt = 0u64;
    for _ in 0..rounds {
        let s = rng.below(5) as u32;
        salt += 1;
        if !t.is_live(s) {
            let req = 128 + rng.below(385) as u32;
            match t.admit(s, req) {
                Ok(()) => {
                    want[s as usize] = req;
                    m.flat[s as usize].iter_mut().for_each(Vec::clear);
                }
                Err(KvTierError::Admission { .. }) => st.refused += 1,
                Err(e) => return Err(format!("admit: {e}")),
            }
        } else if t.is_parked(s) {
            match t.unpark(s) {
                Ok(g) => {
                    st.unparks += 1;
                    st.ssd_reads += g.ssd_reads.len() as u64;
                    m.gather(&g);
                }
                Err(KvTierError::Admission { .. }) => st.refused += 1,
                Err(e) => return Err(format!("unpark: {e}")),
            }
        } else {
            let len = t.len(s);
            match rng.below(10) {
                0 => {
                    t.finish(s).map_err(|e| e.to_string())?;
                    m.flat[s as usize].iter_mut().for_each(Vec::clear);
                }
                1 | 2 => {
                    t.park(s).map_err(|e| format!("park: {e}"))?;
                    st.parks += 1;
                }
                k => {
                    let n = if k < 6 {
                        1 + rng.below(64) as u32
                    } else {
                        1 + rng.below(3) as u32
                    };
                    let n = n.min(want[s as usize] - len);
                    if n == 0 {
                        t.finish(s).map_err(|e| e.to_string())?;
                        m.flat[s as usize].iter_mut().for_each(Vec::clear);
                    } else {
                        t.append(s, n).map_err(|e| format!("append: {e}"))?;
                        m.write(&t, s, len, len + n, salt);
                        let c = t.commit(s).map_err(|e| e.to_string())?;
                        m.commit(&c);
                    }
                }
            }
        }
        t.check()?;
        m.law(&t)?;
    }
    st.ram_drops = t.counts().ram_drops;
    Ok(st)
}

#[test]
fn park_unpark_keeps_the_byte_law_and_an_active_sequence_never_reads_the_drive() {
    let (mut parks, mut reads, mut drops, mut refused) = (0, 0, 0, 0);
    for seed in 1..=6 {
        let st = drive(seed, 3_000, false).unwrap();
        parks += st.parks;
        reads += st.ssd_reads;
        drops += st.ram_drops;
        refused += st.refused;
        assert!(st.unparks > 0);
    }
    // the warm cache really dropped parked copies and the unparks really read the drive
    assert!(
        parks > 500 && drops > 300 && reads > 300,
        "{parks} {drops} {reads}"
    );
    // admission by request length refused while RAM was reserved by the active sequences
    assert!(refused > 0);
}

#[test]
fn the_park_byte_law_catches_lost_drive_writes() {
    // negative control: the RAM -> drive writes are dropped; an unparked sequence whose warm copy
    // was dropped then reads stale drive bytes
    let failed = (1..=6).filter(|&s| drive(s, 3_000, true).is_err()).count();
    assert!(failed >= 5, "the control failed only {failed} of 6");
}

#[test]
fn admission_by_request_length_counts_active_sequences_only() {
    let p = plan(); // RAM 100 blocks, 3 layers: a 512-token request reserves 96
    let mut t = KvTier::new(p).resident().unwrap();
    t.admit(0, 512).unwrap();
    assert!(matches!(
        t.admit(1, 512),
        Err(KvTierError::Admission { .. })
    ));
    assert!(matches!(
        t.admit(1, 256),
        Err(KvTierError::Admission { .. })
    ));
    t.admit(1, 16).unwrap(); // 3 more: 99 of 100
    for (s, n) in [(0, 64), (0, 64), (0, 1), (1, 15)] {
        t.append(s, n).unwrap();
        t.commit(s).unwrap();
    }
    t.park(0).unwrap();
    assert!(
        t.ram_reserved() < 60,
        "parking released the RAM: {}",
        t.ram_reserved()
    );
    t.admit(2, 400).unwrap(); // fits now (the parked open pages keep 3), the drive holds every context
                              // unpark is the admission law again: refused while 2 and 1 hold the RAM
    assert!(matches!(t.unpark(0), Err(KvTierError::Admission { .. })));
    assert!(t.is_parked(0));
    t.finish(2).unwrap();
    let g = t.unpark(0).unwrap();
    assert!(!t.is_parked(0));
    assert!(g.copied > 0, "the window comes back hot");
    t.check().unwrap();
    // the 2-tier (inclusive) tier refuses the resident law
    let two = KvTierPlan::new(3, 16, 32, 64, 4, 512, 72 * 16384, 400 * 16384, 0).unwrap();
    assert!(KvTier::new(two).resident().is_err());
}

#[test]
fn a_parked_sequence_refuses_every_step_until_unparked() {
    let mut t = KvTier::new(plan()).resident().unwrap();
    t.admit(0, 200).unwrap();
    t.append(0, 64).unwrap();
    t.commit(0).unwrap();
    t.park(0).unwrap();
    assert!(t.append(0, 1).is_err() && t.commit(0).is_err() && t.gather_seq(0).is_err());
    t.unpark(0).unwrap();
    t.append(0, 1).unwrap();
}

fn tmpdir(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn the_drive_slab_round_trips_coalesced_runs() {
    assert_eq!(
        ssd_runs(&[(3, 10), (1, 8), (2, 9), (7, 2), (8, 4)]),
        vec![
            SsdRun {
                ram: 1,
                ssd: 8,
                n: 3
            },
            SsdRun {
                ram: 7,
                ssd: 2,
                n: 1
            },
            SsdRun {
                ram: 8,
                ssd: 4,
                n: 1
            }
        ]
    );
    let d = tmpdir("kv_ssd_slab");
    let block = 16 * 1024u64;
    let ssd = KvSsd::create(&d.join("kv.ssd"), 64, block, OpenMode::Direct)
        .or_else(|_| KvSsd::create(&d.join("kv.ssd"), 64, block, OpenMode::Buffered))
        .unwrap();
    let mut slab = AlignedBuf::zeroed(40 * block as usize);
    let mut rng = Rng(7);
    for b in slab.iter_mut() {
        *b = rng.next() as u8;
    }
    let want = slab.to_vec();
    let pairs: Vec<(u32, u32)> = (0..32)
        .map(|r| (r, (r * 5 + 3) % 64))
        .chain([(33, 40), (34, 41)])
        .collect();
    ssd.write(&slab, 0, &pairs).unwrap();
    slab.iter_mut().for_each(|b| *b = 0);
    let back: Vec<(u32, u32)> = pairs.iter().map(|&(r, d)| (d, r)).collect();
    ssd.read(&mut slab, 0, &back).unwrap();
    for &(r, _) in &pairs {
        let (a, n) = (r as usize * block as usize, block as usize);
        assert_eq!(slab[a..a + n], want[a..a + n], "block {r}");
    }
    assert!(
        ssd.read(&mut slab, 0, &[(64, 0)]).is_err(),
        "past the drive slab"
    );
}

#[test]
fn the_prompt_cache_serves_the_longest_stored_prefix_and_evicts_lru() {
    let d = tmpdir("kv_pcache_lru");
    let ids: Vec<u32> = (0..10_000).map(|i| i * 7 % 1000).collect();
    let keys = prefix_keys(b"pack", &ids, 2048);
    assert_eq!(keys.len(), 4);
    let mut pc = PromptCache::open(&d, 3 * 300_000).unwrap();
    let blob = |k: u8| vec![k; 300_000];
    assert!(pc.lookup(&keys).is_none());
    pc.store(keys[0].1, 2048, &[&blob(1)]).unwrap();
    pc.store(keys[2].1, 6144, &[&blob(3)[..100_000], &blob(3)[100_000..]])
        .unwrap();
    assert_eq!(pc.lookup(&keys), Some(keys[2]));
    assert_eq!(pc.load(&keys[2].1).unwrap().to_vec(), blob(3));
    // a turn that diverges inside chunk 2 finds chunk 1's key only
    let mut ids2 = ids.clone();
    ids2[5000] += 1;
    let k2 = prefix_keys(b"pack", &ids2, 2048);
    assert_eq!(pc.lookup(&k2), Some(keys[0]));
    // LRU: keys[0] was touched by load? no -- keys[2] was; storing two more evicts keys[0] first
    pc.store(keys[1].1, 4096, &[&blob(2)]).unwrap();
    pc.store(keys[3].1, 8192, &[&blob(4)]).unwrap();
    assert!(pc.get(&keys[0].1).is_none() && pc.get(&keys[2].1).is_some());
    assert!(pc.used() <= pc.budget() && pc.counts.evictions == 1);
    // an entry over the whole budget is refused; another model's keys never match
    assert!(!pc.store([9; 32], 2048, &[&vec![0u8; 1_000_000]]).unwrap());
    assert!(pc.lookup(&prefix_keys(b"other", &ids, 2048)).is_none());
    drop(pc);
    // the cache survives a restart with its entries
    let mut pc = PromptCache::open(&d, 3 * 300_000).unwrap();
    assert_eq!(pc.len(), 3);
    assert_eq!(pc.load(&keys[3].1).unwrap().to_vec(), blob(4));
}

#[test]
fn a_half_written_or_damaged_entry_is_never_served() {
    let d = tmpdir("kv_pcache_crash");
    let ids: Vec<u32> = (0..8192).collect();
    let keys = prefix_keys(b"pack", &ids, 2048);
    let mut pc = PromptCache::open(&d, 1 << 30).unwrap();
    let big = vec![5u8; 3 << 20];
    pc.store(keys[0].1, 2048, &[&big]).unwrap();
    pc.store(keys[1].1, 4096, &[&big]).unwrap();
    pc.store(keys[2].1, 6144, &[&big]).unwrap();
    let id = |pc: &PromptCache, k: usize| pc.get(&keys[k].1).unwrap().id;
    let (i1, i2) = (id(&pc, 1), id(&pc, 2));
    drop(pc);
    // (a) a crash mid-store: the data written, no record -> a stray file, never served
    fs::write(d.join("e00000000000ffff.tmp"), &big).unwrap();
    fs::write(d.join("e00000000000fffe.bin"), &big).unwrap();
    // (b) a torn record at the manifest's tail
    let mut man = fs::read(d.join("manifest")).unwrap();
    man.extend_from_slice(&[0x46, 0x4e, 0x50, 0x43, 1, 0, 0]);
    fs::write(d.join("manifest"), &man).unwrap();
    // (c) entry 1 truncated, (d) entry 2 damaged inside a sampled span (same length)
    let p1 = d.join(format!("e{i1:016x}.bin"));
    fs::write(&p1, &big[..1000]).unwrap();
    let p2 = d.join(format!("e{i2:016x}.bin"));
    let mut b2 = fs::read(&p2).unwrap();
    b2[(1 << 20) + 17] ^= 0xFF;
    fs::write(&p2, &b2).unwrap();
    let mut pc = PromptCache::open(&d, 1 << 30).unwrap();
    assert!(!d.join("e00000000000ffff.tmp").exists() && !d.join("e00000000000fffe.bin").exists());
    assert!(
        pc.get(&keys[1].1).is_none(),
        "the truncated entry is dropped at open"
    );
    assert_eq!(pc.counts.damaged, 1);
    // the damaged one is found but refused at load, dropped, and the lookup falls back
    assert_eq!(pc.lookup(&keys), Some(keys[2]));
    assert!(pc.load(&keys[2].1).is_err());
    assert_eq!(pc.lookup(&keys), Some(keys[0]));
    assert_eq!(pc.load(&keys[0].1).unwrap().to_vec(), big);
    assert_eq!(
        fs::metadata(d.join("manifest")).unwrap().len() as usize % RECORD,
        0
    );
    // the digest really samples the damaged span (the control for (d))
    assert_ne!(sampled_digest(&b2), sampled_digest(&big));
}

/// FN45c3: the async store's three steps == `store`; a reserved key is not served (nor reserved
/// twice) until committed; an aborted reservation gives its bytes back; the parts' digest ==
/// the concatenation's.
#[test]
fn the_async_store_publishes_only_on_commit() {
    use saber_mem::kv_ssd::{sampled_digest_parts, write_entry};
    let dir = tmpdir("fn45c3-async");
    let mut c = PromptCache::open(&dir, 64 << 20).unwrap();
    let a: Vec<u8> = (0..3_000_000u32).map(|i| (i * 7 % 251) as u8).collect();
    let (p, q) = a.split_at(1_234_567);
    assert_eq!(sampled_digest_parts(&[p, q]), sampled_digest(&a));
    let k = [3u8; 32];
    let r = c.reserve(k, 4096, a.len() as u64).unwrap().unwrap();
    assert!(c.get(&k).is_none(), "a reserved entry is not served");
    assert!(c.reserve(k, 4096, a.len() as u64).unwrap().is_none());
    assert_eq!(c.used(), a.len() as u64);
    let d = write_entry(&r, &[p, q]).unwrap();
    assert!(c.get(&k).is_none(), "written but not committed");
    c.commit(r, d).unwrap();
    assert_eq!(c.get(&k).unwrap().tokens, 4096);
    assert_eq!(c.load(&k).unwrap().to_vec(), a);
    assert_eq!(c.lengths_in(1, 8192), vec![4096]);
    assert!(c.lengths_in(4097, 8192).is_empty());
    // an aborted reservation leaves nothing behind
    let k2 = [4u8; 32];
    let r2 = c.reserve(k2, 8192, 1000).unwrap().unwrap();
    c.abort(&r2);
    assert_eq!(c.used(), a.len() as u64);
    assert!(c.get(&k2).is_none());
    // a restart serves the committed entry
    drop(c);
    let mut c = PromptCache::open(&dir, 64 << 20).unwrap();
    assert_eq!(c.load(&k).unwrap().to_vec(), a);
    let _ = std::fs::remove_dir_all(&dir);
}

/// FN45c4 (the returning-load stall): a CONTENDED writer + prober. Two writer threads store 24
/// entries of 8 MiB each (file + fsync + the `put` record's fsync, all with NO cache lock: only
/// `reserve` and `publish` take it) while a prober takes the cache lock, reads the index and reads
/// published entries outside the lock. The prober's longest lock wait stays far below the
/// writers' I/O, every published entry reads back whole, and a reopen serves all 24.
#[test]
fn a_contended_writer_never_holds_the_probers_lock_across_io() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Instant;
    let dir = tmpdir("fn45c4-contended");
    let pc = Arc::new(Mutex::new(PromptCache::open(&dir, 1 << 30).unwrap()));
    let done = Arc::new(AtomicBool::new(false));
    let key = |i: u32| {
        let mut k = [0u8; 32];
        k[..4].copy_from_slice(&i.to_le_bytes());
        k
    };
    let writers: Vec<_> = (0..2u32)
        .map(|w| {
            let pc = Arc::clone(&pc);
            std::thread::spawn(move || {
                let mut io_ms = 0f64;
                for i in (w..24).step_by(2) {
                    let data = vec![(i as u8).wrapping_mul(37); 8 << 20];
                    let r = pc
                        .lock()
                        .unwrap()
                        .reserve(key(i), 64 * (i + 1), data.len() as u64);
                    let r = r.unwrap().expect("a fresh key reserves");
                    let t = Instant::now();
                    let d = write_entry(&r, &[&data]).unwrap();
                    let m = pc.lock().unwrap().manifest();
                    write_record(&m, &r, d).unwrap();
                    io_ms += t.elapsed().as_secs_f64() * 1e3;
                    pc.lock().unwrap().publish(r, d);
                }
                io_ms
            })
        })
        .collect();
    let prober = {
        let (pc, done) = (Arc::clone(&pc), Arc::clone(&done));
        std::thread::spawn(move || {
            let (mut worst, mut reads) = (0f64, 0u32);
            while !done.load(Ordering::Relaxed) {
                let t = Instant::now();
                let f = {
                    let c = pc.lock().unwrap();
                    let _ = c.lengths_in(1, u32::MAX);
                    (0..24).find_map(|i| c.entry_file(&key(i)))
                };
                worst = worst.max(t.elapsed().as_secs_f64() * 1e3);
                if let Some((p, e)) = f {
                    assert!(
                        read_checked(&p, &e).is_some(),
                        "a published entry reads whole"
                    );
                    reads += 1;
                }
            }
            (worst, reads)
        })
    };
    let io: f64 = writers.into_iter().map(|h| h.join().unwrap()).sum();
    done.store(true, Ordering::Relaxed);
    let (worst, reads) = prober.join().unwrap();
    assert!(reads > 0);
    // the writers' I/O (24 x (8 MiB file + 2 fsyncs)) never sits inside the prober's lock wait
    assert!(
        worst < 50.0 && worst < io / 24.0 + 5.0,
        "prober waited {worst:.1} ms (writer I/O {io:.0} ms)"
    );
    drop(pc);
    let pc = PromptCache::open(&dir, 1 << 30).unwrap();
    assert_eq!(pc.len(), 24, "every published put record is durable");
    let _ = fs::remove_dir_all(&dir);
}
