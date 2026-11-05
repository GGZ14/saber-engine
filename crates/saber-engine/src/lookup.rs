//! FN69: PROMPT-LOOKUP speculation -- a per-sequence n-gram index over the sequence's prompt and
//! its committed output, and the merge law that puts its continuation in front of the drafter's.
//!
//! Agentic coding traffic repeats its input: an edit re-emits most of the file it was handed, a
//! path or an identifier comes back verbatim. Prompt-lookup decoding finds the last `n` tokens of
//! the sequence earlier in the sequence and proposes what followed them there. The target verifies
//! the proposal exactly as it verifies the drafter's, so the committed ids are the target's own
//! whatever was proposed (output-neutral by construction; the verify step is unchanged).
//!
//! **Built in OUR verify path.** The prod-era `SABER_DYNAMIC_DRAFT` n-gram source hooked
//! The reference server's V1 MTP proposer and was inert under DFlash2; this one is a host-side source both serve
//! planes consult at the point where they already hold the next step's drafts: the 27B through
//! [`LookupDrafter`] (a wrapper over the engine's mailbox drafter), Flash-Next through
//! [`SeqLookup`] directly on each rank's `Live` (both ranks see the same commands, so both make
//! the same choice). The Flash-Next PLE n-gram table is a MODEL feature and is unrelated.
//!
//! The index (rule: O(1) per token on the step path): for every `n` in `nmin..=nmax` a map from
//! the hash of the n-gram ENDING at position `e` to the latest such `e`. An n-gram is indexed only
//! once its continuation token exists (one position of lag), so a query for the sequence's own
//! suffix never finds itself. A hit is checked token by token (hash collisions never propose), the
//! longest `n` wins, its match is extended backwards (bounded) to measure how much context agrees,
//! and the continuation is capped at the step's verify width.
//!
//! The MERGE law ("replace when confident"): if the lookup's match length is at least
//! `min_match` and its continuation is at least as long as the drafter's proposal, the lookup's
//! continuation is the step's draft; otherwise the drafter's. Counted per source.

use std::collections::{BTreeMap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};

use saber_sched::SeqId;
use saber_serve::engine::TokenId;

use crate::draft::Drafter;

/// The lookup's parameters (`--spec-lookup-ngram`, `--spec-lookup-min-match`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LookupConfig {
    /// Shortest n-gram indexed.
    pub nmin: u32,
    /// Longest n-gram indexed (the longest match wins).
    pub nmax: u32,
    /// The merge law's confidence bar: the (backward-extended) match length the lookup needs to
    /// replace the drafter's proposal.
    pub min_match: u32,
    /// How far a hit's match is extended backwards to measure it.
    pub max_extend: u32,
}

impl Default for LookupConfig {
    fn default() -> Self {
        LookupConfig {
            nmin: 3,
            nmax: 4,
            min_match: 3,
            max_extend: 64,
        }
    }
}

impl LookupConfig {
    /// Parse `--spec-lookup-ngram` (`"3..4"`, `"3"`) with the merge bar.
    ///
    /// # Errors
    /// A range outside `1..=16`, `nmin > nmax`, or a bar of 0.
    pub fn parse(ngram: &str, min_match: u32) -> Result<LookupConfig, String> {
        let (a, b) = match ngram.split_once("..") {
            Some((a, b)) => (a, b),
            None => (ngram, ngram),
        };
        let p = |s: &str| {
            s.trim()
                .parse::<u32>()
                .map_err(|e| format!("--spec-lookup-ngram {ngram:?}: {e}"))
        };
        let (nmin, nmax) = (p(a)?, p(b)?);
        if nmin == 0 || nmax > 16 || nmin > nmax {
            return Err(format!(
                "--spec-lookup-ngram {ngram:?}: need 1 <= nmin <= nmax <= 16"
            ));
        }
        if min_match == 0 {
            return Err("--spec-lookup-min-match must be at least 1".into());
        }
        Ok(LookupConfig {
            nmin,
            nmax,
            min_match,
            ..LookupConfig::default()
        })
    }
}

/// The lookup's counters, per source (`saber:engine_lookup_*` / `saber_fn_lookup_*`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LookupCounts {
    /// Queries (one per sequence per drafting point).
    pub lookups: u64,
    /// Queries that found an earlier occurrence of the suffix.
    pub hits: u64,
    /// Drafting points the merge law gave to the lookup.
    pub used: u64,
    /// Draft tokens verified from the lookup / accepted out of them.
    pub lookup_drafted: u64,
    /// See [`LookupCounts::lookup_drafted`].
    pub lookup_accepted: u64,
    /// Draft tokens verified from the drafter / accepted out of them (same steps' law).
    pub drafter_drafted: u64,
    /// See [`LookupCounts::drafter_drafted`].
    pub drafter_accepted: u64,
}

impl LookupCounts {
    /// One census line.
    #[must_use]
    pub fn line(&self) -> String {
        format!(
            "lookups {}, hits {}, used {}, lookup drafted {} / accepted {} (from the lookup), \
             drafter drafted {} / accepted {} (from the drafter)",
            self.lookups,
            self.hits,
            self.used,
            self.lookup_drafted,
            self.lookup_accepted,
            self.drafter_drafted,
            self.drafter_accepted
        )
    }

    /// Tally one verified step's acceptance against the source that drafted it.
    pub fn verified(&mut self, source: Source, drafted: u64, accepted: u64) {
        match source {
            Source::Lookup => {
                self.lookup_drafted += drafted;
                self.lookup_accepted += accepted;
            }
            Source::Drafter => {
                self.drafter_drafted += drafted;
                self.drafter_accepted += accepted;
            }
        }
    }
}

/// Which source a step's draft came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// The prompt-lookup continuation.
    Lookup,
    /// The engine's drafter (DFlash2 / MTP).
    Drafter,
}

/// The keys are already hashes: the map's own hasher passes them through.
#[derive(Default)]
struct PassThrough(u64);

impl Hasher for PassThrough {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 << 8) ^ u64::from(b);
        }
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = v;
    }
}

type Map = HashMap<u64, u32, BuildHasherDefault<PassThrough>>;

fn hash(ids: &[TokenId]) -> u64 {
    let mut h: u64 = 0x9e37_79b9_7f4a_7c15;
    for &t in ids {
        h ^= u64::from(t);
        h = h.wrapping_mul(0x0100_0000_01b3_u64 ^ 0xff51_afd7_ed55_8ccd);
        h ^= h >> 29;
    }
    h
}

/// A hit: the match length (backward-extended) and the continuation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hit {
    /// Tokens of context that agree, ending at the sequence's last token.
    pub matched: u32,
    /// What followed the earlier occurrence, at most the cap.
    pub draft: Vec<TokenId>,
}

/// One sequence's index: its ids (prompt + committed output) and the n-gram maps.
pub struct SeqLookup {
    cfg: LookupConfig,
    hist: Vec<TokenId>,
    maps: Vec<Map>,
    /// The next n-gram END position to index (every `e < hist.len() - 1` is indexed).
    next_end: usize,
}

impl std::fmt::Debug for SeqLookup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SeqLookup")
            .field("tokens", &self.hist.len())
            .field("indexed", &self.next_end)
            .finish()
    }
}

impl SeqLookup {
    /// An empty index.
    #[must_use]
    pub fn new(cfg: LookupConfig) -> SeqLookup {
        let k = (cfg.nmax - cfg.nmin + 1) as usize;
        SeqLookup {
            cfg,
            hist: Vec::new(),
            maps: (0..k).map(|_| Map::default()).collect(),
            next_end: 0,
        }
    }

    /// Tokens held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.hist.len()
    }

    /// Whether nothing is held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hist.is_empty()
    }

    /// The last token held.
    #[must_use]
    pub fn last(&self) -> Option<TokenId> {
        self.hist.last().copied()
    }

    /// Append tokens (the prompt's, or a step's committed ones); indexes every n-gram whose
    /// continuation now exists.
    pub fn extend(&mut self, ids: &[TokenId]) {
        self.hist.extend_from_slice(ids);
        self.index_pending();
    }

    /// Replace the last token (a sampled row whose host draw replaced the argmax the commit
    /// reported). No index entry ends at the last position, so nothing is re-indexed.
    pub fn replace_last(&mut self, t: TokenId) {
        if let Some(x) = self.hist.last_mut() {
            *x = t;
        }
    }

    fn index_pending(&mut self) {
        let nmin = self.cfg.nmin as usize;
        while self.next_end + 1 < self.hist.len() {
            let e = self.next_end;
            for (i, map) in self.maps.iter_mut().enumerate() {
                let n = nmin + i;
                if e + 1 >= n {
                    map.insert(hash(&self.hist[e + 1 - n..=e]), e as u32);
                }
            }
            self.next_end += 1;
        }
    }

    /// The longest n-gram suffix's latest earlier occurrence and up to `cap` tokens that
    /// followed it.
    #[must_use]
    pub fn query(&self, cap: usize) -> Option<Hit> {
        let len = self.hist.len();
        if cap == 0 || len == 0 {
            return None;
        }
        let last = len - 1;
        let nmin = self.cfg.nmin as usize;
        for i in (0..self.maps.len()).rev() {
            let n = nmin + i;
            if len < n + 1 {
                continue;
            }
            let suffix = &self.hist[len - n..];
            let Some(&e) = self.maps[i].get(&hash(suffix)) else {
                continue;
            };
            let e = e as usize;
            if e >= last || self.hist[e + 1 - n..=e] != *suffix {
                continue;
            }
            // extend the match backwards: how much context agrees
            let mut m = n;
            while m < n + self.cfg.max_extend as usize
                && e >= m
                && self.hist[e - m] == self.hist[last - m]
            {
                m += 1;
            }
            let end = (e + 1 + cap).min(len);
            return Some(Hit {
                matched: m as u32,
                draft: self.hist[e + 1..end].to_vec(),
            });
        }
        None
    }

    /// THE MERGE LAW for one drafting point: the lookup's continuation when its match is at least
    /// the bar and it is at least as long as the drafter's proposal (`drafter_len`), else `None`
    /// (the drafter's stands). Counts the query and the hit.
    pub fn merge(
        &self,
        cap: usize,
        drafter_len: usize,
        counts: &mut LookupCounts,
    ) -> Option<Vec<TokenId>> {
        counts.lookups += 1;
        let hit = self.query(cap)?;
        counts.hits += 1;
        if hit.matched >= self.cfg.min_match && hit.draft.len() >= drafter_len.min(cap) {
            counts.used += 1;
            Some(hit.draft)
        } else {
            None
        }
    }
}

/// Per-sequence state the 27B's wrapper keeps beside the index.
#[derive(Debug)]
struct SeqState {
    idx: SeqLookup,
    /// The prompt is in, the first generated token is not: the first `propose` appends its
    /// `last` (the token the prefill committed).
    awaiting_first: bool,
    /// What the step being verified was drafted from, and how many tokens.
    pending: Option<(Source, u64)>,
}

/// The 27B's lookup source: wraps the executor's drafter (the DFlash2 mailbox) and applies the
/// merge law at `propose`. The mailbox is always drained, so a step the lookup drafted leaves no
/// stale block behind.
#[derive(Debug)]
pub struct LookupDrafter {
    inner: Box<dyn Drafter>,
    cfg: LookupConfig,
    seqs: BTreeMap<SeqId, SeqState>,
    counts: LookupCounts,
}

impl LookupDrafter {
    /// `inner` with the lookup in front of it.
    #[must_use]
    pub fn new(inner: Box<dyn Drafter>, cfg: LookupConfig) -> LookupDrafter {
        LookupDrafter {
            inner,
            cfg,
            seqs: BTreeMap::new(),
            counts: LookupCounts::default(),
        }
    }

    fn state(&mut self, seq: SeqId) -> &mut SeqState {
        let cfg = self.cfg;
        self.seqs.entry(seq).or_insert_with(|| SeqState {
            idx: SeqLookup::new(cfg),
            awaiting_first: true,
            pending: None,
        })
    }
}

impl Drafter for LookupDrafter {
    fn name(&self) -> &'static str {
        "lookup+engine"
    }

    fn max_width(&self) -> u32 {
        self.inner.max_width()
    }

    fn propose(&mut self, seq: SeqId, last: TokenId, width: u32) -> Vec<TokenId> {
        let theirs = self.inner.propose(seq, last, width);
        let mut counts = self.counts;
        let cap = width.min(self.inner.max_width()) as usize;
        let st = self.state(seq);
        let (draft, source) = match st.idx.merge(cap, theirs.len(), &mut counts) {
            Some(d) => (d, Source::Lookup),
            None => (theirs, Source::Drafter),
        };
        st.pending = Some((source, draft.len() as u64));
        self.counts = counts;
        draft
    }

    fn commit(&mut self, seq: SeqId, committed: &[TokenId]) {
        self.inner.commit(seq, committed);
        let st = self.state(seq);
        st.awaiting_first = false;
        st.idx.extend(committed);
        if let Some((source, drafted)) = st.pending.take() {
            let accepted = (committed.len() as u64).saturating_sub(1).min(drafted);
            self.counts.verified(source, drafted, accepted);
        }
    }

    fn offer(&mut self, seq: SeqId, draft: &[TokenId]) {
        self.inner.offer(seq, draft);
    }

    fn input(&mut self, seq: SeqId, last: TokenId) {
        self.inner.input(seq, last);
        let st = self.state(seq);
        if st.awaiting_first {
            st.idx.extend(&[last]);
            st.awaiting_first = false;
        } else if st.idx.last() != Some(last) {
            // a sampled row: the commit reported the argmax, the host sampler drew this one
            st.idx.replace_last(last);
        }
    }

    fn prompt(&mut self, seq: SeqId, ids: &[TokenId]) {
        self.inner.prompt(seq, ids);
        let cfg = self.cfg;
        let mut idx = SeqLookup::new(cfg);
        idx.extend(ids);
        self.seqs.insert(
            seq,
            SeqState {
                idx,
                awaiting_first: true,
                pending: None,
            },
        );
    }

    fn release(&mut self, seq: SeqId) {
        self.inner.release(seq);
        self.seqs.remove(&seq);
    }

    fn lookup_counts(&self) -> Option<LookupCounts> {
        Some(self.counts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draft::EngineDrafter;

    fn ids(s: &str) -> Vec<TokenId> {
        s.bytes().map(TokenId::from).collect()
    }

    #[test]
    fn a_repeated_span_is_proposed_from_its_earlier_occurrence() {
        let mut l = SeqLookup::new(LookupConfig::default());
        l.extend(&ids("fn alpha(x) { beta(x) } ... fn alp"));
        let h = l.query(6).expect("a hit");
        assert_eq!(h.draft, ids("ha(x) "));
        assert!(h.matched >= 4, "{h:?}");
    }

    #[test]
    fn the_suffix_never_matches_itself_and_a_miss_is_none() {
        let mut l = SeqLookup::new(LookupConfig::default());
        l.extend(&ids("abcdefgh"));
        assert_eq!(l.query(4), None);
        l.extend(&ids("abc"));
        assert_eq!(l.query(4).expect("hit").draft, ids("defg"));
    }

    #[test]
    fn the_longest_n_wins_and_the_latest_occurrence_is_taken() {
        let mut l = SeqLookup::new(LookupConfig::default());
        // "xbcd" then "abcd" then "ybcd": the 4-gram "abcd" occurs once (followed by 'Q'),
        // the 3-gram "bcd" latest is followed by 'Z'
        l.extend(&ids("abcdQ...ybcdZ...abcd"));
        assert_eq!(l.query(1).expect("hit").draft, ids("Q"));
        let mut l = SeqLookup::new(LookupConfig::default());
        l.extend(&ids("1bcdQ...ybcdZ...abcd"));
        assert_eq!(l.query(1).expect("hit").draft, ids("Z"));
    }

    #[test]
    fn the_continuation_is_capped_by_the_width_and_the_end() {
        let mut l = SeqLookup::new(LookupConfig::default());
        l.extend(&ids("abcabcabc"));
        let h = l.query(8).expect("hit");
        // the latest earlier "abc" ends at index 5: the continuation is "abc" (to the end)
        assert_eq!(h.draft, ids("abc"));
        assert_eq!(l.query(2).expect("hit").draft, ids("ab"));
    }

    #[test]
    fn the_merge_law_needs_the_bar_and_the_length() {
        let cfg = LookupConfig {
            min_match: 6,
            ..LookupConfig::default()
        };
        let mut l = SeqLookup::new(cfg);
        l.extend(&ids("xyabcdXYZW..abcd"));
        let mut c = LookupCounts::default();
        // matched 4 < 6: the drafter's stands
        assert_eq!(l.merge(4, 4, &mut c), None);
        let mut l = SeqLookup::new(LookupConfig::default());
        l.extend(&ids("..abcdXYZW..abcd"));
        assert_eq!(l.merge(4, 4, &mut c), Some(ids("XYZW")));
        // shorter than the drafter's: the drafter's stands
        let mut l = SeqLookup::new(LookupConfig::default());
        l.extend(&ids("..abcdXY"));
        l.extend(&ids("abcd"));
        assert!(l.merge(8, 8, &mut c).is_none());
        assert_eq!(c.lookups, 3);
        assert_eq!(c.used, 1);
    }

    #[test]
    fn the_config_parses_a_range_and_refuses_nonsense() {
        assert_eq!(
            LookupConfig::parse("3..4", 3).unwrap(),
            LookupConfig::default()
        );
        assert_eq!(LookupConfig::parse("5", 2).unwrap().nmin, 5);
        assert!(LookupConfig::parse("4..3", 3).is_err());
        assert!(LookupConfig::parse("0..3", 3).is_err());
        assert!(LookupConfig::parse("3..4", 0).is_err());
    }

    #[test]
    fn the_wrapper_replaces_the_mailbox_when_confident_and_counts_per_source() {
        let s = SeqId(7);
        let mut d = LookupDrafter::new(Box::new(EngineDrafter::new(4)), LookupConfig::default());
        d.prompt(s, &ids("let total = sum(values);\nlet tot"));
        // the drafter's block for the first step
        d.offer(s, &ids("!!!!"));
        // the prefill committed 'a'; the index now ends "...let tota"
        d.input(s, TokenId::from(b'a'));
        let p = d.propose(s, TokenId::from(b'a'), 4);
        assert_eq!(p, ids("l = "));
        d.commit(s, &ids("l = s"));
        d.offer(s, &ids("u!!!"));
        d.input(s, TokenId::from(b's'));
        let p = d.propose(s, TokenId::from(b's'), 4);
        assert_eq!(p, ids("um(v"));
        d.commit(s, &ids("um("));
        let c = d.lookup_counts().unwrap();
        assert_eq!(c.used, 2);
        assert_eq!((c.lookup_drafted, c.lookup_accepted), (8, 6));
        // a miss falls back to the mailbox
        d.commit(s, &[]);
        d.release(s);
        let q = SeqId(8);
        d.prompt(q, &ids("zzzz"));
        d.offer(q, &ids("ab"));
        d.input(q, TokenId::from(b'y'));
        assert_eq!(d.propose(q, TokenId::from(b'y'), 4), ids("ab"));
        d.commit(q, &ids("a!"));
        let c = d.lookup_counts().unwrap();
        assert_eq!((c.drafter_drafted, c.drafter_accepted), (2, 1));
    }

    #[test]
    fn a_sampled_rows_replaced_last_token_moves_the_tail_only() {
        let s = SeqId(1);
        let mut d = LookupDrafter::new(Box::new(EngineDrafter::new(4)), LookupConfig::default());
        d.prompt(s, &ids("abcdefgh"));
        // a first step that drafts nothing still feeds its token
        d.input(s, TokenId::from(b'x'));
        // the commit reported the argmax 'q', the sampler drew 'a'
        d.commit(s, &ids("q"));
        d.offer(s, &ids("!!!!"));
        // the index is now "abcdefghxa": no 3-gram repeats
        d.input(s, TokenId::from(b'a'));
        let p = d.propose(s, TokenId::from(b'a'), 4);
        assert_eq!(p, ids("!!!!"));
        d.commit(s, &ids("bc"));
        d.offer(s, &ids("!!!!"));
        // "...xabc": the 3-gram "abc" occurs at the start, followed by "defg"
        d.input(s, TokenId::from(b'c'));
        assert_eq!(d.propose(s, TokenId::from(b'c'), 4), ids("defg"));
    }
}
