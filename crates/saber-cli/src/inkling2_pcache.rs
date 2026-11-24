//! INK23: the Inkling prompt-prefix cache (`--ink-prefix-cache N`, an ARM; 0 = off = the control).
//!
//! A NODE is one sequence's whole cross-token state at a prompt position P, copied out of its
//! cache slot right after a prefill forward that ended at P: per visit (48) the K / V pages of
//! positions `0..P` (global visits; a local visit's ring pages, all of them once the ring wrapped:
//! the ring is position-addressed, so the same rows land at the same physical rows in any slot),
//! the four k / v conv state buffers (both parity halves) and the two sconv states (attn, mlp);
//! on rank 0, when the drafter held the sequence's context, its context K / V rows `0..P` and its
//! f_rnn row. Nothing else crosses a token: the depth state (block list, the pass summaries, the
//! messages) lives inside one token's 48 visits.
//!
//! The byte law it rests on (INK23 h1, `--ink-run long --ink-chunk 13 | 333 | 1024` on the
//! 2,686-row L00: every compared row byte-equal to the one-pass rows): a prefill cut anywhere
//! leaves exactly the state the uncut pass computes, so a reload at P followed by the suffix's
//! prefill is the cold run byte for byte -- and the cold run that STORES a node (its chunk split
//! at P) is the uncut run byte for byte too.
//!
//! The policy (rank 0, `serve`): at admission, the longest stored node whose ids are a proper
//! prefix of the prompt is reloaded; the prompt's own cut is the position of its LAST user-turn
//! token (`<|message_user|>`: everything before the last user message -- a shared system prompt,
//! the earlier turns), stored when it is at least `--ink-prefix-min` tokens and past the reload.
//! LRU over N nodes; a node copy is stream-ordered D2D (no host wait).

use anyhow::{ensure, Context as _};
use std::collections::BTreeMap;

use super::{Cache, Card, GLOBAL, PAGE};

/// A visit's node pages: (k pages, v pages) bases, page bytes, pages a node, global visit.
type KvNode = ([u64; 2], u64, u64, bool);

/// One rank's node storage: N nodes, each a copy of one cache slot's per-visit state.
pub(super) struct Store {
    pub n: u64,
    /// (visit, rank entry) -> (k pages, v pages, page bytes a node, pages a node, is global)
    kv: BTreeMap<(usize, usize), KvNode>,
    /// (visit, rank entry) -> (sk, sv, sk1, sv1) a node, bytes a node
    st: BTreeMap<(usize, usize), ([u64; 4], u64)>,
    /// (visit, 2 | 3) -> the sconv state a node, bytes a node
    cv: BTreeMap<(usize, u32), (u64, u64)>,
    /// rank 0: the drafter's regions a node (the node's base address, region capacity) and the
    /// node bytes they take
    dr: Vec<u64>,
    dr_cap: Vec<u64>,
    pub bytes: u64,
    /// INK26 `--ink-prefix-cache-verify`: each node region's hash at its store ((node address,
    /// bytes) -> FNV-1a 64) and the counters: [ops checked, copies whose destination != source
    /// after the copy, node regions changed between their store and a load, regions checked]
    pub vhash: BTreeMap<(u64, u64), u64>,
    pub vstat: [u64; 4],
}

/// Where a cache slot's region of one kind starts (the pools are slot-major).
fn slot_at(base: u64, slot: u64, per: u64) -> u64 {
    base + slot * per
}

impl Store {
    /// Allocate `n` nodes mirroring `cache`'s per-slot geometry (+ the drafter's regions of
    /// `dr_cap` bytes each on rank 0).
    pub fn new(
        c: &mut Card<'_, '_>,
        cache: &Cache,
        sched: &[u32],
        n: u64,
        dr_cap: &[u64],
    ) -> anyhow::Result<Self> {
        ensure!((1..=16).contains(&n), "--ink-prefix-cache {n}: 1..16 nodes");
        let lp = if cache.ring_pages > 0 {
            cache.ring_pages
        } else {
            cache.npg
        };
        let mut s = Self {
            n,
            kv: BTreeMap::new(),
            st: BTreeMap::new(),
            cv: BTreeMap::new(),
            dr: Vec::new(),
            dr_cap: dr_cap.to_vec(),
            bytes: 0,
            vhash: BTreeMap::new(),
            vstat: [0; 4],
        };
        for (&(vi, ri), p) in &cache.pools {
            let l = sched[vi];
            let global = GLOBAL.contains(&l);
            let pages = if global { cache.npg } else { lp };
            // the pool's bytes a slot / its pages = the page bytes (the c x bf16 rows of a page)
            let slots = cache.seqs + 1;
            let st_b = c_pool_bytes(c, p[2])? / slots;
            let pool_b = c_pool_bytes(c, p[0])? / slots;
            ensure!(
                pool_b % pages == 0,
                "ink pcache: pool {vi}/{ri} {pool_b} B a slot"
            );
            let pg_b = pool_b / pages;
            let k = c.alloc_zero(n * pool_b)?;
            let v = c.alloc_zero(n * pool_b)?;
            s.kv.insert((vi, ri), ([k, v], pg_b, pages, global));
            let mut a4 = [0u64; 4];
            for a in &mut a4 {
                *a = c.alloc_zero(n * st_b)?;
            }
            s.st.insert((vi, ri), (a4, st_b));
            s.bytes += 2 * n * pool_b + 4 * n * st_b;
        }
        for (&key, &base) in &cache.convs {
            let per = c_pool_bytes(c, base)? / (cache.seqs + 1);
            s.cv.insert(key, (c.alloc_zero(n * per)?, per));
            s.bytes += n * per;
        }
        for &cap in dr_cap {
            s.dr.push(c.alloc_zero(n * cap)?);
            s.bytes += n * cap;
        }
        Ok(s)
    }

    /// The copies (dst, src, bytes) between cache slot `slot` and node `node` for a node of `len`
    /// positions; `out` = slot -> node (store), else node -> slot (load). `dr` = the drafter's
    /// current regions (rank 0, `Drafter::ctx_regions(len)`), empty = no drafter part.
    pub fn copies(
        &self,
        cache: &Cache,
        slot: u64,
        node: u64,
        len: u64,
        out: bool,
        dr: &[(u64, u64)],
    ) -> anyhow::Result<Vec<(u64, u64, u64)>> {
        ensure!(node < self.n, "ink pcache: node {node} of {}", self.n);
        ensure!(slot <= cache.seqs, "ink pcache: slot {slot}");
        let mut v = Vec::new();
        let mut push = |cache_at: u64, node_at: u64, b: u64| {
            if b > 0 {
                v.push(if out {
                    (node_at, cache_at, b)
                } else {
                    (cache_at, node_at, b)
                });
            }
        };
        let used = len.div_ceil(PAGE);
        for (key, &(kv2, pg_b, pages, global)) in &self.kv {
            let p = cache.pools.get(key).context("pools")?;
            let np = if global { used } else { used.min(pages) };
            let per = pg_b * pages;
            for w in 0..2 {
                push(
                    slot_at(p[w], slot, per),
                    slot_at(kv2[w], node, per),
                    np * pg_b,
                );
            }
            let (s4, sb) = self.st[key];
            let p2 = cache.pools2.get(key).context("pools2")?;
            let src = [p[2], p[3], p2[0], p2[1]];
            for w in 0..4 {
                push(slot_at(src[w], slot, sb), slot_at(s4[w], node, sb), sb);
            }
        }
        for (key, &(nb, per)) in &self.cv {
            let base = *cache.convs.get(key).context("convs")?;
            push(slot_at(base, slot, per), slot_at(nb, node, per), per);
        }
        if !dr.is_empty() {
            ensure!(
                dr.len() == self.dr.len(),
                "ink pcache: {} drafter regions, {} allocated",
                dr.len(),
                self.dr.len()
            );
            for (i, &(a, b)) in dr.iter().enumerate() {
                ensure!(b <= self.dr_cap[i], "ink pcache: drafter region {i} {b} B");
                push(a, slot_at(self.dr[i], node, self.dr_cap[i]), b);
            }
        }
        Ok(v)
    }
}

/// The bytes of the card allocation that starts at `addr`.
fn c_pool_bytes(c: &Card<'_, '_>, addr: u64) -> anyhow::Result<u64> {
    let (d, off) = alloc_of(c, addr, 1)?;
    ensure!(
        off == 0,
        "ink pcache: {addr:#x} is not an allocation's start"
    );
    Ok(d.len())
}

/// The allocation holding `addr..addr + bytes` and the offset in it.
fn alloc_of<'c>(
    c: &'c Card<'_, '_>,
    addr: u64,
    bytes: u64,
) -> anyhow::Result<(&'c saber_hip::DevicePtr, u64)> {
    for d in std::iter::once(&c.work).chain(c.keep.iter()) {
        let b = d.addr().as_u64();
        if addr >= b && addr + bytes <= b + d.len() {
            return Ok((d, addr - b));
        }
    }
    anyhow::bail!("ink pcache: {addr:#x} + {bytes} in none of the card's allocations")
}

/// Run `copies` (dst, src, bytes) as stream-ordered D2D copies on the card's stream (no host
/// wait: the next forward on the stream reads them).
pub(super) fn run(c: &Card<'_, '_>, copies: &[(u64, u64, u64)]) -> anyhow::Result<()> {
    let mut jobs = Vec::with_capacity(copies.len());
    for &(dst, src, bytes) in copies {
        let (dp, doff) = alloc_of(c, dst, bytes)?;
        let (sp, soff) = alloc_of(c, src, bytes)?;
        jobs.push(saber_hip::Job::MemcpyD2d {
            dst: dp,
            dst_offset: doff,
            src: sp,
            src_offset: soff,
            bytes,
            stream: &c.stream,
        });
    }
    for ch in jobs.chunks(256) {
        c.hip.run_batch(ch)?;
    }
    Ok(())
}

/// FNV-1a 64 of a byte run (the verify arm's node fingerprint).
fn fnv(b: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &x in b {
        h ^= u64::from(x);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// INK26 `--ink-prefix-cache-verify`: after a store / load's copies (stream-ordered: the blocking
/// reads wait for them) read every copy's destination and source back and compare them byte for
/// byte; a store records each node region's fingerprint, a load first checks the node still holds
/// what its store wrote. Returns the counts of this op: (copies differing, node regions changed).
pub(super) fn verify(
    c: &Card<'_, '_>,
    st: &mut Store,
    copies: &[(u64, u64, u64)],
    out: bool,
) -> anyhow::Result<(u64, u64)> {
    let (mut bad, mut moved) = (0u64, 0u64);
    for &(dst, src, b) in copies {
        let n = usize::try_from(b)?;
        let d = c.read(dst, n)?;
        let s = c.read(src, n)?;
        if d != s {
            bad += 1;
        }
        let node_at = if out { dst } else { src };
        let h = fnv(if out { &d } else { &s });
        if out {
            st.vhash.insert((node_at, b), h);
        } else if let Some(&h0) = st.vhash.get(&(node_at, b)) {
            if h0 != h {
                moved += 1;
            }
        }
        st.vstat[3] += 1;
    }
    st.vstat[0] += 1;
    st.vstat[1] += bad;
    st.vstat[2] += moved;
    Ok((bad, moved))
}

/// Rank 0's node table: the ids each node holds, its drafter part, its last use.
#[derive(Default)]
pub(super) struct Index {
    pub nodes: Vec<Option<Meta>>,
    pub tick: u64,
    pub loads: u64,
    pub load_rows: u64,
    pub stores: u64,
    pub skipped_nodraft: u64,
    /// the admission filter: cut prefixes seen once (a hash each, FIFO); a cut is stored only
    /// at its SECOND sighting (a one-off prompt never pays the split + the copy), counted
    pub seen: std::collections::VecDeque<u64>,
    pub first_sightings: u64,
}

pub(super) struct Meta {
    pub ids: Vec<u32>,
    pub draft: bool,
    pub used: u64,
}

/// What admission decided for one prompt: reload node `.0` of `.1` positions; store a node at
/// position `.2`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Plan {
    pub load: Option<(u64, u64)>,
    pub store: Option<u64>,
}

impl Index {
    pub fn new(n: u64) -> Self {
        Self {
            nodes: (0..n).map(|_| None).collect(),
            ..Self::default()
        }
    }

    /// The plan for `prompt`: the longest node whose ids are a PROPER prefix (at least one row
    /// is left to compute: the first token's logits), and the prompt's cut (its last `user`
    /// token's position) when it is >= `min`, past the reload and not already a node.
    pub fn plan(&self, prompt: &[u32], user: Option<u32>, min: u64) -> Plan {
        let mut load: Option<(u64, u64)> = None;
        for (i, m) in self.nodes.iter().enumerate() {
            if let Some(m) = m {
                let l = m.ids.len();
                if l < prompt.len()
                    && prompt[..l] == m.ids[..]
                    && load.is_none_or(|(_, b)| l as u64 > b)
                {
                    load = Some((i as u64, l as u64));
                }
            }
        }
        let lo = load.map_or(0, |x| x.1);
        let store = user.and_then(|u| {
            let p = prompt.iter().rposition(|&t| t == u)? as u64;
            let have = self
                .nodes
                .iter()
                .flatten()
                .any(|m| m.ids.len() as u64 == p && m.ids[..] == prompt[..p as usize]);
            (p >= min && p > lo && !have).then_some(p)
        });
        Plan { load, store }
    }

    /// The admission plan: [`Index::plan`] with the two-touch filter on the store -- a cut is
    /// stored only when the same cut prefix was seen before (INK23 h3 / h4: storing every cut
    /// cost a one-off chat prompt +35 ms, the split forward's second latency floor).
    pub fn admit(&mut self, prompt: &[u32], user: Option<u32>, min: u64) -> Plan {
        use std::hash::{Hash as _, Hasher as _};
        let mut pl = self.plan(prompt, user, min);
        if let Some(p) = pl.store {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            prompt[..p as usize].hash(&mut h);
            let k = h.finish();
            if !self.seen.contains(&k) {
                self.seen.push_back(k);
                if self.seen.len() > 256 {
                    self.seen.pop_front();
                }
                self.first_sightings += 1;
                pl.store = None;
            }
        }
        pl
    }

    /// The node a new store takes: an empty one, else the least recently used.
    pub fn victim(&self) -> u64 {
        if let Some(i) = self.nodes.iter().position(Option::is_none) {
            return i as u64;
        }
        self.nodes
            .iter()
            .enumerate()
            .min_by_key(|(_, m)| m.as_ref().map_or(0, |m| m.used))
            .map_or(0, |(i, _)| i as u64)
    }

    pub fn touch(&mut self, node: u64) {
        self.tick += 1;
        if let Some(Some(m)) = self.nodes.get_mut(node as usize) {
            m.used = self.tick;
        }
    }

    pub fn put(&mut self, node: u64, ids: Vec<u32>, draft: bool) {
        self.tick += 1;
        self.nodes[node as usize] = Some(Meta {
            ids,
            draft,
            used: self.tick,
        });
        self.stores += 1;
    }
}

/// The user-turn token id of the checkpoint's tokenizer (`<|message_user|>`), if it has one.
pub(super) fn user_token(ckpt: &std::path::Path) -> anyhow::Result<Option<u32>> {
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(ckpt.join("tokenizer.json"))?)?;
    Ok(v["added_tokens"].as_array().and_then(|a| {
        a.iter()
            .find(|t| t["content"] == "<|message_user|>")
            .and_then(|t| t["id"].as_u64())
            .and_then(|x| u32::try_from(x).ok())
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idx(nodes: &[&[u32]]) -> Index {
        let mut x = Index::new(4);
        for (i, n) in nodes.iter().enumerate() {
            x.put(i as u64, n.to_vec(), true);
        }
        x
    }

    #[test]
    fn ink23_the_plan_reloads_the_longest_proper_prefix_and_cuts_at_the_last_user_turn() {
        let u = 9;
        // empty: no reload, the cut at the last user token (position 5) when >= min
        let x = Index::new(2);
        let p = [1, 2, 3, 9, 4, 9, 5, 6];
        assert_eq!(
            x.plan(&p, Some(u), 4),
            Plan {
                load: None,
                store: Some(5)
            }
        );
        assert_eq!(
            x.plan(&p, Some(u), 6),
            Plan {
                load: None,
                store: None
            }
        );
        assert_eq!(
            x.plan(&p, None, 0),
            Plan {
                load: None,
                store: None
            }
        );
        // a stored cut is reloaded, never re-stored; the longest proper prefix wins
        let x = idx(&[&[1, 2, 3], &[1, 2, 3, 9, 4], &[1, 2, 7]]);
        assert_eq!(
            x.plan(&p, Some(u), 0),
            Plan {
                load: Some((1, 5)),
                store: None
            }
        );
        // a node equal to the whole prompt is not a proper prefix (the first token's row); the
        // prompt's own cut is still stored
        let x = idx(&[&p[..]]);
        assert_eq!(
            x.plan(&p, Some(u), 0),
            Plan {
                load: None,
                store: Some(5)
            }
        );
        // a cut at or below the reload is not stored
        let x = idx(&[&[1, 2, 3, 9, 4, 9]]);
        assert_eq!(
            x.plan(&p, Some(u), 0),
            Plan {
                load: Some((0, 6)),
                store: None
            }
        );
    }

    #[test]
    fn ink23_a_cut_is_stored_at_its_second_sighting() {
        let mut x = Index::new(2);
        let p = [1, 2, 3, 9, 4, 9, 5, 6];
        let q = [1, 2, 3, 9, 4, 9, 7];
        assert_eq!(x.admit(&p, Some(9), 4).store, None);
        assert_eq!(x.admit(&q, Some(9), 4).store, Some(5));
        assert_eq!(x.admit(&[8, 8, 8, 8, 8, 9, 1], Some(9), 4).store, None);
        assert_eq!(x.first_sightings, 2);
    }

    #[test]
    fn ink23_the_victim_is_an_empty_node_then_the_least_recently_used() {
        let mut x = Index::new(3);
        assert_eq!(x.victim(), 0);
        x.put(0, vec![1], false);
        x.put(1, vec![2], false);
        assert_eq!(x.victim(), 2);
        x.put(2, vec![3], false);
        assert_eq!(x.victim(), 0);
        x.touch(0);
        assert_eq!(x.victim(), 1);
    }
}
