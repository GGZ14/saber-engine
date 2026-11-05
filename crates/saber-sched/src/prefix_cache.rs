//! **FN52: prefix (conversation) caching for a HYBRID model** — what prod the reference server runs as
//! `--enable-prefix-caching --mamba-cache-mode align`, in this engine's terms.
//!
//! The 27B is GDN recurrent layers plus attention, so a cached prefix is TWO things that are
//! useless apart (`ARCHITECTURE.md` § Memory plane): the attention layers' KV pages for the
//! prefix's whole blocks, and the recurrent state AT the prefix's last position. The pages are
//! addressable per token and are retained IN PLACE — the entry holds one reference on each
//! block in the paged allocator's refcount, exactly as a sequence group's branches hold the
//! leader's prompt pages (`alloc::PagedKvAllocator::branch`), and a sequence that adopts them
//! takes another reference: no copy. The recurrent state is not addressable per token, so the
//! executor SNAPSHOTS it at the cached position (with the drafter's pages over the prefix, which
//! are the drafter's context and move acceptance, never ids) into host memory under the entry's
//! id; an adoption restores it into the new sequence's own slot set. A retained entry never pins
//! a live state slot set.
//!
//! **The cached position** is the prompt's last whole-block boundary STRICTLY before its end,
//! `a = ((len - 1) / block) * block`: block-aligned so the pages are whole (a partial block is
//! never shared, the fork's law), and strictly before the end so a hit always re-prefills at
//! least one token — the row that samples. The scheduler cuts the prompt's prefill at `a` (one
//! extra small chunk at the end), which is the only schedule change the cache makes, and the
//! snapshot is taken after that chunk. The reference server's align mode does the same: states kept at
//! block-aligned positions of the computed tokens.
//!
//! **The key** is [`crate::prefix_key::prefix_block_keys`], chained, so block `i`'s key names the
//! whole prefix — and a hit still compares the FULL token ids (no partial-block hits, no
//! hash-collision hits: multi-tenant safety is an equality, not a probability). Image prompts
//! are never cached (their blocks' keys need the image hashes this module does not carry).
//!
//! **Eviction** is LRU, under two pressures: the host budget (snapshot bytes, per rank) on
//! insert, and the KV arena on admission — a new request is never refused while an entry could
//! be dropped to make room for it.

use std::collections::HashMap;

use saber_serve::engine::TokenId;

use crate::prefix_key::prefix_block_keys;

/// The serve flag's whole policy. [`PrefixCachePolicy::OFF`] is today's engine and the control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixCachePolicy {
    /// `--enable-prefix-caching on`.
    pub enabled: bool,
    /// `--prefix-cache-verify`: a hit is NOT adopted; the prompt is prefilled fresh with a cut at
    /// the hit's position, and the fresh pages + state there are compared byte for byte against
    /// the entry (the executor's [`crate::StepExecutor::prefix_verify`]).
    pub verify: bool,
    /// Host bytes the snapshots may hold, per rank.
    pub host_budget_bytes: u64,
    /// Entries at most.
    pub max_entries: u32,
    /// Prefixes shorter than this are not snapshotted (the extra chunk is not worth it).
    pub min_tokens: u32,
}

impl PrefixCachePolicy {
    /// Off: no cut, no snapshot, no lookup. Every byte is today's.
    pub const OFF: Self = Self {
        enabled: false,
        verify: false,
        host_budget_bytes: 0,
        max_entries: 0,
        min_tokens: 0,
    };

    /// On, at the defaults the serve flag uses.
    #[must_use]
    pub fn on() -> Self {
        Self {
            enabled: true,
            verify: false,
            host_budget_bytes: 6 << 30,
            max_entries: 64,
            min_tokens: 512,
        }
    }
}

impl Default for PrefixCachePolicy {
    fn default() -> Self {
        Self::OFF
    }
}

/// One retained prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixEntry {
    /// The executor's handle for the snapshot.
    pub id: u32,
    /// The chained key of the prefix's last block.
    pub key: u64,
    /// The prefix, in full: a hit compares these ids, not the key.
    pub tokens: Vec<TokenId>,
    /// The prefix's whole KV blocks, one reference each held by this entry.
    pub blocks: Vec<u32>,
    /// Host bytes the executor reported for the snapshot (per rank).
    pub host_bytes: u64,
    last_used: u64,
}

/// A lookup's answer: the longest retained prefix of a prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixHit {
    /// The entry.
    pub entry: u32,
    /// Tokens it covers (block-aligned, `< prompt_len`).
    pub tokens: u32,
    /// Its blocks.
    pub blocks: Vec<u32>,
}

/// Which-path counters (`CLAUDE.md` rule 5).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefixCacheCounters {
    /// Admissions that looked.
    pub lookups: u64,
    /// ...and found a prefix.
    pub hits: u64,
    /// Prompt tokens a hit did not prefill.
    pub hit_tokens: u64,
    /// Hits adopted (not verify).
    pub adopted: u64,
    /// Snapshots taken.
    pub saves: u64,
    /// Snapshots the executor refused.
    pub save_failures: u64,
    /// Adoptions whose restore failed (the request is ended with an error).
    pub load_failures: u64,
    /// Entries dropped for the host budget or the entry cap.
    pub evictions_budget: u64,
    /// Entries dropped so an admission fits the KV arena.
    pub evictions_pressure: u64,
    /// Verify arm: fresh == entry.
    pub verify_match: u64,
    /// Verify arm: fresh != entry.
    pub verify_mismatch: u64,
    /// Verify arm: the entry was gone, or the executor refused.
    pub verify_skipped: u64,
}

/// The retained entries.
#[derive(Debug, Clone)]
pub struct PrefixCache {
    policy: PrefixCachePolicy,
    block_tokens: u32,
    by_key: HashMap<u64, u32>,
    entries: HashMap<u32, PrefixEntry>,
    next_id: u32,
    clock: u64,
    host_bytes: u64,
    /// The counters.
    pub counters: PrefixCacheCounters,
}

impl PrefixCache {
    /// An empty cache over `block_tokens`-token KV blocks.
    #[must_use]
    pub fn new(policy: PrefixCachePolicy, block_tokens: u32) -> Self {
        Self {
            policy,
            block_tokens: block_tokens.max(1),
            by_key: HashMap::new(),
            entries: HashMap::new(),
            next_id: 1,
            clock: 0,
            host_bytes: 0,
            counters: PrefixCacheCounters::default(),
        }
    }

    /// The policy.
    #[must_use]
    pub fn policy(&self) -> PrefixCachePolicy {
        self.policy
    }

    /// Whether the cache does anything.
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.policy.enabled
    }

    /// Entries held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether none are held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Host bytes held (per rank).
    #[must_use]
    pub fn host_bytes(&self) -> u64 {
        self.host_bytes
    }

    /// The position a prompt of `prompt_len` tokens is snapshotted at: its last whole-block
    /// boundary strictly before its end, when that is at least `min_tokens`.
    #[must_use]
    pub fn snapshot_point(&self, prompt_len: usize) -> Option<u32> {
        if !self.policy.enabled || prompt_len < 2 {
            return None;
        }
        let bt = self.block_tokens as usize;
        let a = ((prompt_len - 1) / bt) * bt;
        (a > 0 && a >= self.policy.min_tokens as usize).then(|| u32::try_from(a).ok())?
    }

    /// The chained keys of `prompt`'s whole blocks below its last token.
    fn keys(&self, prompt: &[TokenId]) -> Vec<u64> {
        let usable = prompt.len().saturating_sub(1);
        prefix_block_keys(&prompt[..usable], self.block_tokens, &[])
    }

    /// Whether an entry already holds exactly `prompt[..at]`.
    #[must_use]
    pub fn holds(&self, prompt: &[TokenId], at: u32) -> bool {
        let at = at as usize;
        if at == 0 || at > prompt.len() || at % self.block_tokens as usize != 0 {
            return false;
        }
        let keys = prefix_block_keys(&prompt[..at], self.block_tokens, &[]);
        keys.last()
            .and_then(|k| self.by_key.get(k))
            .and_then(|id| self.entries.get(id))
            .is_some_and(|e| e.tokens[..] == prompt[..at])
    }

    /// The LONGEST retained prefix of `prompt` (strictly shorter than it), touched (LRU). The
    /// scheduler counts the hit when it admits on it.
    pub fn lookup(&mut self, prompt: &[TokenId]) -> Option<PrefixHit> {
        if !self.policy.enabled {
            return None;
        }
        let keys = self.keys(prompt);
        let bt = self.block_tokens as usize;
        for (i, k) in keys.iter().enumerate().rev() {
            let Some(id) = self.by_key.get(k).copied() else {
                continue;
            };
            let at = (i + 1) * bt;
            let Some(e) = self.entries.get_mut(&id) else {
                continue;
            };
            // THE FULL IDS, not the key: a hit is an equality.
            if e.tokens.len() != at || e.tokens[..] != prompt[..at] {
                continue;
            }
            self.clock += 1;
            e.last_used = self.clock;
            return Some(PrefixHit {
                entry: id,
                tokens: at as u32,
                blocks: e.blocks.clone(),
            });
        }
        None
    }

    /// A fresh handle for a snapshot about to be taken.
    pub fn reserve_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1).max(1);
        id
    }

    /// Retain `tokens` (block-aligned) under `id`, whose snapshot the executor holds at
    /// `host_bytes`. Returns the entries evicted to stay inside the budget and the cap (oldest
    /// first) — the caller drops their snapshots and releases their blocks. The new entry is never
    /// among them unless it alone is over the budget, in which case it is returned too.
    pub fn insert(
        &mut self,
        id: u32,
        tokens: Vec<TokenId>,
        blocks: Vec<u32>,
        host_bytes: u64,
    ) -> Vec<PrefixEntry> {
        let key = prefix_block_keys(&tokens, self.block_tokens, &[])
            .last()
            .copied()
            .unwrap_or(0);
        let mut evicted = Vec::new();
        // The same prefix twice (two requests raced to it): the older goes.
        if let Some(old) = self.by_key.get(&key).copied() {
            if let Some(e) = self.remove(old) {
                evicted.push(e);
            }
        }
        self.clock += 1;
        self.by_key.insert(key, id);
        self.host_bytes += host_bytes;
        self.entries.insert(
            id,
            PrefixEntry {
                id,
                key,
                tokens,
                blocks,
                host_bytes,
                last_used: self.clock,
            },
        );
        self.counters.saves += 1;
        while self.host_bytes > self.policy.host_budget_bytes
            || self.entries.len() > self.policy.max_entries.max(1) as usize
        {
            let Some(e) = self.evict_lru(None) else { break };
            self.counters.evictions_budget += 1;
            evicted.push(e);
        }
        evicted
    }

    /// Drop the least recently used entry other than `keep`.
    pub fn evict_lru(&mut self, keep: Option<u32>) -> Option<PrefixEntry> {
        let id = self
            .entries
            .values()
            .filter(|e| Some(e.id) != keep)
            .min_by_key(|e| e.last_used)
            .map(|e| e.id)?;
        self.remove(id)
    }

    /// Drop one entry by id.
    pub fn remove(&mut self, id: u32) -> Option<PrefixEntry> {
        let e = self.entries.remove(&id)?;
        if self.by_key.get(&e.key) == Some(&id) {
            self.by_key.remove(&e.key);
        }
        self.host_bytes = self.host_bytes.saturating_sub(e.host_bytes);
        Some(e)
    }

    /// The entry under `id`, if still held.
    #[must_use]
    pub fn get(&self, id: u32) -> Option<&PrefixEntry> {
        self.entries.get(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache() -> PrefixCache {
        PrefixCache::new(
            PrefixCachePolicy {
                min_tokens: 16,
                host_budget_bytes: 1000,
                max_entries: 3,
                ..PrefixCachePolicy::on()
            },
            16,
        )
    }

    fn ids(n: u32) -> Vec<TokenId> {
        (1..=n).collect()
    }

    #[test]
    fn the_snapshot_point_is_the_last_block_boundary_strictly_before_the_end() {
        let c = cache();
        assert_eq!(c.snapshot_point(48), Some(32));
        assert_eq!(c.snapshot_point(49), Some(48));
        assert_eq!(c.snapshot_point(17), Some(16));
        assert_eq!(c.snapshot_point(16), None);
        assert_eq!(
            PrefixCache::new(PrefixCachePolicy::OFF, 16).snapshot_point(4096),
            None
        );
    }

    #[test]
    fn the_longest_retained_prefix_wins_and_a_one_token_difference_misses() {
        let mut c = cache();
        let a = c.reserve_id();
        assert!(c.insert(a, ids(32), vec![7, 8], 10).is_empty());
        let b = c.reserve_id();
        assert!(c.insert(b, ids(64), vec![7, 8, 9, 10], 10).is_empty());
        let hit = c.lookup(&ids(100)).expect("hit");
        assert_eq!((hit.entry, hit.tokens), (b, 64));
        // Strictly shorter than the prompt: a 64-token prompt adopts 32, never all 64.
        assert_eq!(c.lookup(&ids(64)).map(|h| h.tokens), Some(32));
        // One token different inside the first block: nothing.
        let mut p = ids(100);
        p[3] = 999;
        assert_eq!(c.lookup(&p), None);
        // ...inside the third block: the 32-token entry still holds.
        let mut p = ids(100);
        p[40] = 999;
        assert_eq!(c.lookup(&p).map(|h| h.tokens), Some(32));
        assert!(c.holds(&ids(100), 64) && !c.holds(&ids(100), 48));
    }

    #[test]
    fn the_budget_and_the_cap_evict_the_least_recently_used() {
        let mut c = cache();
        let ids3: Vec<u32> = (0..3)
            .map(|k| {
                let id = c.reserve_id();
                let mut t = ids(32);
                t[0] = 1000 + k;
                assert!(c.insert(id, t, vec![k], 100).is_empty());
                id
            })
            .collect();
        // Touch the oldest, so the second is the LRU.
        let mut t = ids(40);
        t[0] = 1000;
        assert!(c.lookup(&t).is_some());
        let id = c.reserve_id();
        let ev = c.insert(id, ids(32), vec![9], 100);
        assert_eq!(ev.iter().map(|e| e.id).collect::<Vec<_>>(), vec![ids3[1]]);
        // Over the byte budget: the oldest go until it fits.
        let id = c.reserve_id();
        let ev = c.insert(id, ids(48), vec![9, 10, 11], 900);
        assert_eq!(
            ev.len(),
            2,
            "300 + 900 is over 1,000 by 200: two of the 100s go"
        );
        assert_eq!(c.len(), 2);
        assert_eq!(c.host_bytes(), 1000);
    }
}
