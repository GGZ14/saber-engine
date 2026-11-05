//! FN113: the TIERED PREFIX TREE's host logic (FLASHNEXT-PLAN §16, LEDGER FN113) -- the cut law, the
//! pinned RAM tier's node index (pages, LRFU heat, subtree heat, pins) and the page placement.
//!
//! **The cut law ([`CutLaw`]).** FN13: the cut decides the bytes. A hit is byte-identical to a cold
//! prefill only if the cold prefill cuts where the hit's node ends, so the tree OWNS the cuts: they
//! are a pure, PREFIX-STABLE function of the ids (the cuts of a prefix are the full prompt's cuts
//! restricted to it) -- T (right after every generation prompt of the history, at least `min_gap`
//! tokens after the previous cut), S (after a system message of at least `sys_min` tokens) and the
//! scheduler's chunk ends (C, recorded by the caller). The prompt's END is never a cut (its last
//! piece picks the first token); a prompt ending in a generation prompt that the NEXT turn will cut
//! at is an END node ([`CutLaw::end_node`]), stored after its uncut prefill.
//!
//! **The RAM tier ([`RamTree`]).** Whole nodes in fixed pages of one pinned slab; a node's copy
//! items are placed inside single pages ([`Cursor`]) so every copy is one contiguous host range.
//! Heat is the expert tier's LRFU (`crf = crf x lambda^(ticks since last use) + 1`, FN18), the clock
//! = prompt admissions ([`RamTree::tick`]); a node's EFFECTIVE heat is the max over its RAM subtree
//! (leaves leave first); victims are the coldest unpinned nodes, ties by key (both ranks evict the
//! same nodes). Eviction is a drop: the drive holds the node, or the next request prefills.

#![forbid(unsafe_code)]

use std::collections::HashMap;

/// A node key (FN45a's chained sha256).
pub type Key = [u8; 32];

/// What made a cut.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CutKind {
    /// After an earlier generation prompt of the history (a previous turn's prompt end).
    Turn,
    /// After the system message.
    Sys,
    /// The scheduler's chunk end.
    Chunk,
}

impl CutKind {
    /// The counter's name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            CutKind::Turn => "turn",
            CutKind::Sys => "sys",
            CutKind::Chunk => "chunk",
        }
    }
}

/// The cut law of a checkpoint's chat format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CutLaw {
    /// The generation prompts (each a T cut where it ends, not at the prompt's end).
    pub gens: Vec<Vec<u32>>,
    /// A system message's first ids (`<|im_start|> system \n`).
    pub sys_head: Vec<u32>,
    /// A message's end (`<|im_end|> \n`).
    pub msg_end: Vec<u32>,
    /// The shortest piece a T cut leaves behind it (tokens since the previous cut).
    pub min_gap: u32,
    /// The shortest system prefix that is cut (0 = no S cut).
    pub sys_min: u32,
    /// LR-I: a GRID cut at every multiple of this many tokens (0 = none, today's law): a
    /// checkpoint interval, so a prompt's computed tail is stored every `grid` tokens and the next
    /// turn resumes within `grid` of where it diverges (the reference's interval law).
    pub grid: u32,
}

/// LR-I: which cuts the law makes besides S and the scheduler's chunk ends
/// (`--fn-prefix-tree-cuts`). Any rule but `today` moves a COMPUTED piece's boundary, so a cold
/// prompt's bytes differ from today's (FN13: the cut decides the bytes) -- an ARM, the user's
/// quality call (G4); within a rule a hit is byte-identical to that rule's cold prefill.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CutRule {
    /// The T cut sits right after `<|im_start|> assistant \n` (before the `<think>` part), not
    /// after the whole generation prompt: a client that sends the history without its reasoning
    /// renders `<think> \n\n </think> \n\n` where the prompt had `<think> \n`, so today's END
    /// node never matches and the next turn resumes at the previous cut below it; at the `asst`
    /// cut both forms agree and the next turn resumes 2 tokens before the previous prompt's end.
    pub asst: bool,
    /// The grid interval ([`CutLaw::grid`]; 0 = none).
    pub grid: u32,
}

impl CutRule {
    /// The flag's values: `today` (the default), `asst`, `grid<N>`, `asst+grid<N>` (N a multiple
    /// of 64 in 64..=65536).
    pub fn parse(s: &str) -> Result<CutRule, String> {
        let mut r = CutRule::default();
        if s == "today" {
            return Ok(r);
        }
        for part in s.split('+') {
            if part == "asst" && !r.asst {
                r.asst = true;
            } else if let Some(n) = part.strip_prefix("grid").filter(|_| r.grid == 0) {
                let n: u32 = n
                    .parse()
                    .map_err(|_| format!("--fn-prefix-tree-cuts {s}: grid<N> needs a number"))?;
                if !(64..=65_536).contains(&n) || n % 64 != 0 {
                    return Err(format!(
                        "--fn-prefix-tree-cuts {s}: the grid is a multiple of 64 in 64..=65536"
                    ));
                }
                r.grid = n;
            } else {
                return Err(format!(
                    "--fn-prefix-tree-cuts {s}: today | asst | grid<N> | asst+grid<N>"
                ));
            }
        }
        Ok(r)
    }

    /// The flag's spelling.
    #[must_use]
    pub fn name(self) -> String {
        match (self.asst, self.grid) {
            (false, 0) => "today".into(),
            (true, 0) => "asst".into(),
            (false, g) => format!("grid{g}"),
            (true, g) => format!("asst+grid{g}"),
        }
    }
}

/// Flash-Next's chat format ids (`tokenizer.json`: `<|im_start|>` 248045, `<|im_end|>` 248046,
/// `assistant` 74455, `system` 8678, `\n` 198, `\n\n` 271, `<think>` 248068, `</think>` 248069).
pub mod qwen4 {
    /// `<|im_start|>`.
    pub const IM_START: u32 = 248_045;
    /// `<|im_end|>`.
    pub const IM_END: u32 = 248_046;
    /// `assistant`.
    pub const ASSISTANT: u32 = 74_455;
    /// `system`.
    pub const SYSTEM: u32 = 8678;
    /// `\n`.
    pub const NL: u32 = 198;
    /// `\n\n`.
    pub const NL2: u32 = 271;
    /// `<think>`.
    pub const THINK: u32 = 248_068;
    /// `</think>`.
    pub const THINK_END: u32 = 248_069;
}

/// The default `--fn-prefix-tree-min-gap`.
pub const DEFAULT_MIN_GAP: u32 = 64;
/// The default `--fn-prefix-tree-sys-min`.
pub const DEFAULT_SYS_MIN: u32 = 512;

/// The law's state over one sequence's ids so far (prefix-only: nothing else feeds it).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LawState {
    /// Ids fed so far.
    pub pos: u32,
    /// The last cut's position (0 = none).
    pub last_cut: u32,
    /// The last ids (enough for the longest pattern).
    tail: Vec<u32>,
    /// The first ids (the system check).
    head: Vec<u32>,
    /// The system message's end was seen (or ruled out).
    sys_done: bool,
}

impl CutLaw {
    /// Flash-Next's law (thinking on and off generation prompts).
    #[must_use]
    pub fn qwen4(min_gap: u32, sys_min: u32) -> CutLaw {
        Self::qwen4_rule(min_gap, sys_min, CutRule::default())
    }

    /// LR-I: Flash-Next's law under a [`CutRule`] (`CutRule::default()` = [`CutLaw::qwen4`]).
    #[must_use]
    pub fn qwen4_rule(min_gap: u32, sys_min: u32, rule: CutRule) -> CutLaw {
        use qwen4::*;
        let gens = if rule.asst {
            vec![vec![IM_START, ASSISTANT, NL]]
        } else {
            vec![
                vec![IM_START, ASSISTANT, NL, THINK, NL],
                vec![IM_START, ASSISTANT, NL, THINK, NL2, THINK_END, NL2],
            ]
        };
        CutLaw {
            gens,
            sys_head: vec![IM_START, SYSTEM, NL],
            msg_end: vec![IM_END, NL],
            min_gap,
            sys_min,
            grid: rule.grid,
        }
    }

    fn keep(&self) -> usize {
        self.gens
            .iter()
            .map(Vec::len)
            .chain([self.msg_end.len()])
            .max()
            .unwrap_or(1)
    }

    fn ends_with(tail: &[u32], pat: &[u32]) -> bool {
        !pat.is_empty() && tail.len() >= pat.len() && tail[tail.len() - pat.len()..] == *pat
    }

    /// Feed `ids` (the next ids of the sequence); every cut at a position `q` with `st.pos < q <=
    /// st.pos + ids.len()` in order (the caller drops a cut at the prompt's end: an END node).
    pub fn feed(&self, st: &mut LawState, ids: &[u32]) -> Vec<(u32, CutKind)> {
        let keep = self.keep();
        let mut out = Vec::new();
        for &t in ids {
            st.pos += 1;
            st.tail.push(t);
            if st.tail.len() > keep {
                st.tail.remove(0);
            }
            if st.head.len() < self.sys_head.len() {
                st.head.push(t);
            }
            let q = st.pos;
            if !st.sys_done {
                if st.head.len() >= self.sys_head.len().min(q as usize)
                    && st.head[..st.head.len().min(self.sys_head.len())]
                        != self.sys_head[..st.head.len().min(self.sys_head.len())]
                {
                    st.sys_done = true;
                } else if q as usize > self.sys_head.len()
                    && Self::ends_with(&st.tail, &self.msg_end)
                {
                    st.sys_done = true;
                    if self.sys_min > 0 && q >= self.sys_min {
                        st.last_cut = q;
                        out.push((q, CutKind::Sys));
                        continue;
                    }
                }
            }
            if q - st.last_cut >= self.min_gap.max(1)
                && self.gens.iter().any(|g| Self::ends_with(&st.tail, g))
            {
                st.last_cut = q;
                out.push((q, CutKind::Turn));
            } else if self.grid > 0 && q % self.grid == 0 {
                // LR-I: a checkpoint interval (a pure function of the position: prefix-stable)
                st.last_cut = q;
                out.push((q, CutKind::Chunk));
            }
        }
        out
    }

    /// A non-law cut at the state's position (the scheduler's chunk end: C).
    pub fn chunk_end(st: &mut LawState) {
        st.last_cut = st.pos;
    }

    /// Whether a prompt ending at `st.pos` (after [`CutLaw::feed`] returned a cut AT its end, or
    /// not) is an END node: the next turn's law cuts exactly here.
    #[must_use]
    pub fn end_node(cuts: &[(u32, CutKind)], st: &LawState) -> bool {
        cuts.last()
            .is_some_and(|&(q, k)| q == st.pos && k == CutKind::Turn)
    }
}

/// The page placement cursor: items of at most a page each, never straddling a page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cursor {
    page: u64,
    cur: u32,
    off: u64,
}

impl Cursor {
    /// A cursor over pages of `page` bytes.
    #[must_use]
    pub fn new(page: u64) -> Cursor {
        Cursor {
            page,
            cur: 0,
            off: 0,
        }
    }

    /// Place an item of `n` bytes (`n <= page`): (the node's page index, the offset in it).
    pub fn put(&mut self, n: u64) -> Option<(u32, u64)> {
        if n > self.page {
            return None;
        }
        if self.off + n > self.page {
            self.cur += 1;
            self.off = 0;
        }
        let at = (self.cur, self.off);
        self.off += n;
        Some(at)
    }

    /// Place `nb` blocks of `set` bytes in groups that each fit a page: `(first block, blocks,
    /// page index, offset)` (None: a block exceeds a page).
    pub fn put_blocks(&mut self, nb: u32, set: u64) -> Option<Vec<(u32, u32, u32, u64)>> {
        if set == 0 || set > self.page {
            return None;
        }
        let mut out = Vec::new();
        let mut j = 0;
        while j < nb {
            let room = ((self.page - self.off) / set) as u32;
            if room == 0 {
                self.cur += 1;
                self.off = 0;
                continue;
            }
            let sn = room.min(nb - j);
            out.push((j, sn, self.cur, self.off));
            self.off += u64::from(sn) * set;
            j += sn;
        }
        Some(out)
    }

    /// Split a contiguous region of `n` bytes into page-sized pieces and place each: `(piece offset
    /// in the region, bytes, page index, offset)`.
    pub fn put_split(&mut self, n: u64) -> Vec<(u64, u64, u32, u64)> {
        let mut out = Vec::new();
        let mut o = 0;
        while o < n {
            let room = if self.off >= self.page {
                self.cur += 1;
                self.off = 0;
                self.page
            } else {
                self.page - self.off
            };
            // a piece fills the current page's room (a region is device-contiguous: split freely)
            let k = room.min(n - o);
            out.push((o, k, self.cur, self.off));
            self.off += k;
            o += k;
        }
        out
    }

    /// Pages used so far.
    #[must_use]
    pub fn pages(&self) -> u32 {
        self.cur + u32::from(self.off > 0)
    }
}

/// One RAM node.
#[derive(Debug, Clone)]
pub struct RamNode {
    /// The parent node's key (None: the first piece).
    pub parent: Option<Key>,
    /// The sequence length at the node's end.
    pub tokens: u32,
    /// The slab pages it holds, in placement order.
    pub pages: Vec<u32>,
    /// Bytes stored.
    pub bytes: u64,
    /// In-flight users (a reload or a drive write reading it): never evicted while > 0.
    pub pins: u32,
    crf: f64,
    at: u64,
}

/// What the RAM tier counted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RamCounts {
    /// Nodes inserted.
    pub inserts: u64,
    /// Nodes evicted (dropped).
    pub evictions: u64,
    /// Inserts refused (no room even after evicting every unpinned node).
    pub refused: u64,
    /// Hits served from RAM.
    pub hits: u64,
}

/// The pinned RAM tier's index over a slab of `pages` pages.
#[derive(Debug, Clone)]
pub struct RamTree {
    page: u64,
    pages: u32,
    free: Vec<u32>,
    nodes: HashMap<Key, RamNode>,
    clock: u64,
    lambda: f64,
    /// What happened.
    pub counts: RamCounts,
}

impl RamTree {
    /// A tier of `pages` pages of `page` bytes, LRFU at `lambda` (0.98 = the expert tier's).
    #[must_use]
    pub fn new(page: u64, pages: u32, lambda: f64) -> RamTree {
        RamTree {
            page,
            pages,
            free: (0..pages).rev().collect(),
            nodes: HashMap::new(),
            clock: 0,
            lambda,
            counts: RamCounts::default(),
        }
    }

    /// The page size.
    #[must_use]
    pub fn page(&self) -> u64 {
        self.page
    }

    /// Advance the clock (one prompt admission).
    pub fn tick(&mut self) {
        self.clock += 1;
    }

    fn crf_now(&self, n: &RamNode) -> f64 {
        n.crf
            * self
                .lambda
                .powi((self.clock - n.at).min(i32::MAX as u64) as i32)
    }

    /// The node at `key`.
    #[must_use]
    pub fn get(&self, key: &Key) -> Option<&RamNode> {
        self.nodes.get(key)
    }

    /// A use of `key` (a hit or a re-store): its LRFU score.
    pub fn touch(&mut self, key: &Key) {
        let (clock, lambda) = (self.clock, self.lambda);
        if let Some(n) = self.nodes.get_mut(key) {
            n.crf = n.crf * lambda.powi((clock - n.at).min(i32::MAX as u64) as i32) + 1.0;
            n.at = clock;
        }
    }

    /// Pin / unpin a node (in-flight users).
    pub fn pin(&mut self, key: &Key, on: bool) {
        if let Some(n) = self.nodes.get_mut(key) {
            if on {
                n.pins += 1;
            } else {
                n.pins = n.pins.saturating_sub(1);
            }
        }
    }

    /// Each node's effective heat: the max of its own and its RAM descendants' scores.
    fn effective(&self) -> HashMap<Key, f64> {
        let mut eff: HashMap<Key, f64> = self
            .nodes
            .iter()
            .map(|(k, n)| (*k, self.crf_now(n)))
            .collect();
        for n in self.nodes.values() {
            let h = self.crf_now(n);
            let mut p = n.parent;
            let mut guard = 0;
            while let Some(pk) = p {
                guard += 1;
                if guard > self.nodes.len() {
                    break;
                }
                let Some(pn) = self.nodes.get(&pk) else {
                    break;
                };
                let e = eff.entry(pk).or_insert(0.0);
                if *e < h {
                    *e = h;
                }
                p = pn.parent;
            }
        }
        eff
    }

    fn depth(&self, n: &RamNode) -> u32 {
        let mut d = 0;
        let mut p = n.parent;
        while let Some(pk) = p {
            let Some(pn) = self.nodes.get(&pk) else {
                break;
            };
            d += 1;
            if d as usize > self.nodes.len() {
                break;
            }
            p = pn.parent;
        }
        d
    }

    /// The eviction order: coldest effective heat first; at equal heat the DEEPER node first (a
    /// leaf before its ancestors), then the key (both ranks order alike).
    #[must_use]
    pub fn victims(&self) -> Vec<Key> {
        let eff = self.effective();
        let mut v: Vec<(f64, u32, Key)> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.pins == 0)
            .map(|(k, n)| (eff[k], self.depth(n), *k))
            .collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0).then(b.1.cmp(&a.1)).then(a.2.cmp(&b.2)));
        v.into_iter().map(|x| x.2).collect()
    }

    /// The nodes [`RamTree::alloc`] would evict for `need` pages, in order (None: it cannot fit).
    #[must_use]
    pub fn evict_plan(&self, need: u32) -> Option<Vec<Key>> {
        if need > self.pages {
            return None;
        }
        let mut have = self.free.len() as u32;
        let mut gone = Vec::new();
        if have < need {
            for k in self.victims() {
                if have >= need {
                    break;
                }
                have += self.nodes[&k].pages.len() as u32;
                gone.push(k);
            }
        }
        (have >= need).then_some(gone)
    }

    /// Room for a node of `need` pages: evict the coldest until it fits (None: it cannot, the
    /// insert is refused); then the pages and the evicted keys.
    pub fn alloc(&mut self, need: u32) -> Option<(Vec<u32>, Vec<Key>)> {
        if need > self.pages {
            self.counts.refused += 1;
            return None;
        }
        let mut gone = Vec::new();
        if (self.free.len() as u32) < need {
            let order = self.victims();
            let mut have = self.free.len() as u32;
            for k in order {
                if have >= need {
                    break;
                }
                have += self.nodes[&k].pages.len() as u32;
                gone.push(k);
            }
            if have < need {
                self.counts.refused += 1;
                return None;
            }
            for k in &gone {
                self.remove(k);
                self.counts.evictions += 1;
            }
        }
        let at = self.free.len() - need as usize;
        let mut p = self.free.split_off(at);
        p.reverse();
        Some((p, gone))
    }

    /// Insert a node into pages from [`RamTree::alloc`].
    pub fn insert(
        &mut self,
        key: Key,
        parent: Option<Key>,
        tokens: u32,
        pages: Vec<u32>,
        bytes: u64,
    ) {
        self.counts.inserts += 1;
        let at = self.clock;
        if let Some(old) = self.nodes.insert(
            key,
            RamNode {
                parent,
                tokens,
                pages,
                bytes,
                pins: 0,
                crf: 1.0,
                at,
            },
        ) {
            self.free.extend(old.pages);
        }
    }

    /// Give pages from [`RamTree::alloc`] back unused.
    pub fn release(&mut self, pages: Vec<u32>) {
        self.free.extend(pages);
    }

    /// Drop a node (its pages free).
    pub fn remove(&mut self, key: &Key) -> bool {
        match self.nodes.remove(key) {
            Some(n) => {
                self.free.extend(n.pages);
                true
            }
            None => false,
        }
    }

    /// Nodes held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// No node held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Bytes stored / pages used / pages in the slab.
    #[must_use]
    pub fn usage(&self) -> (u64, u32, u32) {
        (
            self.nodes.values().map(|n| n.bytes).sum(),
            self.pages - self.free.len() as u32,
            self.pages,
        )
    }

    /// Every page owned once (by a node or the free list).
    pub fn check(&self) -> Result<(), String> {
        let mut own = vec![0u32; self.pages as usize];
        for p in self
            .free
            .iter()
            .chain(self.nodes.values().flat_map(|n| n.pages.iter()))
        {
            *own.get_mut(*p as usize)
                .ok_or(format!("page {p} out of range"))? += 1;
        }
        match own.iter().position(|&c| c != 1) {
            Some(i) => Err(format!("page {i} owned {} times", own[i])),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::qwen4::*;
    use super::*;

    fn user(n: usize, salt: u32) -> Vec<u32> {
        let mut v = vec![IM_START, 846, NL];
        v.extend((0..n as u32).map(|i| 1000 + (i * 7 + salt) % 5000));
        v.extend([IM_END, NL]);
        v
    }

    fn gen_on() -> Vec<u32> {
        vec![IM_START, ASSISTANT, NL, THINK, NL]
    }

    fn sys(n: usize) -> Vec<u32> {
        let mut v = vec![IM_START, SYSTEM, NL];
        v.extend((0..n as u32).map(|i| 2000 + i % 3000));
        v.extend([IM_END, NL]);
        v
    }

    fn cuts_all(law: &CutLaw, ids: &[u32], chunks: &[usize]) -> Vec<(u32, CutKind)> {
        let mut st = LawState::default();
        let mut out = Vec::new();
        let mut a = 0;
        for &c in chunks {
            let b = (a + c).min(ids.len());
            out.extend(law.feed(&mut st, &ids[a..b]));
            if b < ids.len() {
                CutLaw::chunk_end(&mut st);
            }
            a = b;
        }
        out
    }

    /// The law the tree's exactness rests on: the cuts of a prefix are the full prompt's cuts
    /// restricted to it, whatever the feeding split.
    #[test]
    fn the_cut_law_is_prefix_stable_and_split_independent() {
        let law = CutLaw::qwen4(DEFAULT_MIN_GAP, DEFAULT_SYS_MIN);
        // a 3-turn conversation with a long system prompt, thinking on (reasoning in history)
        let mut ids = sys(600);
        let mut ends = Vec::new();
        for t in 0..3u32 {
            ids.extend(user(100 + t as usize * 30, t));
            ids.extend(gen_on());
            ends.push(ids.len() as u32);
            ids.extend((0..80).map(|i| 3000 + i + t));
            ids.extend([THINK_END, NL2]);
            ids.extend((0..40).map(|i| 4000 + i));
            ids.extend([IM_END, NL]);
        }
        ids.extend(user(50, 9));
        ids.extend(gen_on());
        let full = cuts_all(&law, &ids, &[ids.len()]);
        // S after the system message, T at every previous generation prompt, then the end (T)
        assert_eq!(full[0], (605, CutKind::Sys));
        let turns: Vec<u32> = full
            .iter()
            .filter(|c| c.1 == CutKind::Turn)
            .map(|c| c.0)
            .collect();
        assert_eq!(&turns[..3], &ends[..]);
        assert_eq!(
            *turns.last().unwrap(),
            ids.len() as u32,
            "the end is an END node"
        );
        // split-independent: any feeding split gives the same cuts
        for split in [vec![1usize; ids.len()], vec![7; ids.len()], vec![333; 64]] {
            let mut st = LawState::default();
            let mut got = Vec::new();
            let mut a = 0;
            for c in split {
                if a >= ids.len() {
                    break;
                }
                let b = (a + c).min(ids.len());
                got.extend(law.feed(&mut st, &ids[a..b]));
                a = b;
            }
            assert_eq!(got, full);
        }
        // prefix-stable: every prefix's cuts are the full cuts below it
        for n in [
            600usize,
            605,
            700,
            ends[0] as usize,
            ends[1] as usize + 3,
            ids.len() - 1,
        ] {
            let p = cuts_all(&law, &ids[..n], &[n]);
            let want: Vec<_> = full.iter().copied().filter(|c| c.0 <= n as u32).collect();
            assert_eq!(p, want, "prefix {n}");
        }
    }

    #[test]
    fn a_single_turn_prompt_is_one_piece_and_turn_zero_is_an_end_node() {
        let law = CutLaw::qwen4(DEFAULT_MIN_GAP, DEFAULT_SYS_MIN);
        // a short system prompt (< 512) + one user turn: no cut inside, the end is an END node
        let mut ids = sys(100);
        ids.extend(user(80, 1));
        ids.extend(gen_on());
        let mut st = LawState::default();
        let c = law.feed(&mut st, &ids);
        assert_eq!(c, vec![(ids.len() as u32, CutKind::Turn)]);
        assert!(CutLaw::end_node(&c, &st));
        // no system message, thinking off: the same
        let mut ids2 = user(90, 2);
        ids2.extend([IM_START, ASSISTANT, NL, THINK, NL2, THINK_END, NL2]);
        let mut st2 = LawState::default();
        let c2 = law.feed(&mut st2, &ids2);
        assert!(CutLaw::end_node(&c2, &st2));
        // a completion prompt without a generation prompt: no cut, no end node (the control)
        let mut st3 = LawState::default();
        let raw: Vec<u32> = (0..500).collect();
        let c3 = law.feed(&mut st3, &raw);
        assert!(c3.is_empty() && !CutLaw::end_node(&c3, &st3));
        // a too-short turn (< min gap) is neither cut nor an end node; with min gap 1 it is
        let mut tiny = user(2, 3);
        tiny.extend(gen_on());
        let mut st4 = LawState::default();
        assert!(!CutLaw::end_node(&law.feed(&mut st4, &tiny), &st4));
        let law1 = CutLaw::qwen4(1, 0);
        let mut st5 = LawState::default();
        assert!(CutLaw::end_node(&law1.feed(&mut st5, &tiny), &st5));
    }

    #[test]
    fn the_sys_cut_needs_a_leading_system_message_and_its_minimum() {
        let law = CutLaw::qwen4(DEFAULT_MIN_GAP, 512);
        let mut ids = user(700, 4);
        ids.extend(gen_on());
        let mut st = LawState::default();
        let c = law.feed(&mut st, &ids);
        assert!(
            c.iter().all(|x| x.1 != CutKind::Sys),
            "a user message is not a system"
        );
        let off = CutLaw::qwen4(DEFAULT_MIN_GAP, 0);
        let mut ids2 = sys(700);
        ids2.extend(user(10, 1));
        let mut st2 = LawState::default();
        assert!(off.feed(&mut st2, &ids2).is_empty(), "sys_min 0 = no S cut");
    }

    #[test]
    fn the_chunk_end_resets_the_gap() {
        let law = CutLaw::qwen4(64, 0);
        let mut ids = user(100, 1);
        let g0 = ids.len();
        ids.extend(gen_on()); // a T cut at g0 + 5
                              // chunk end 10 tokens before it: the gap from the chunk end is < 64 -> no T cut
        let c = cuts_all(&law, &ids, &[g0 - 5, 1000]);
        assert!(c.is_empty(), "{c:?}");
        let c = cuts_all(&law, &ids, &[ids.len()]);
        assert_eq!(c, vec![(ids.len() as u32, CutKind::Turn)]);
    }

    /// LR-I: a stripped-reasoning client's turns: prompt k ends in `<think> \n`; turn k + 1
    /// renders that assistant message as `<think> \n\n </think> \n\n reply <|im_end|> \n`.
    fn stripped_turns(n: usize) -> (Vec<u32>, Vec<usize>) {
        let mut ids = sys(600);
        let mut ends = Vec::new();
        for t in 0..n as u32 {
            ids.extend(user(300 + t as usize * 40, t));
            ids.extend(gen_on());
            ends.push(ids.len());
            // the history form of this turn's assistant message (the next prompt's)
            ids.truncate(ids.len() - 1);
            ids.extend([NL2, THINK_END, NL2]);
            ids.extend((0..90).map(|i| 3000 + i + t));
            ids.extend([IM_END, NL]);
        }
        (ids, ends)
    }

    #[test]
    fn lri_the_rules_parse_and_today_is_the_qwen4_law() {
        for s in ["today", "asst", "grid2048", "asst+grid2048"] {
            assert_eq!(CutRule::parse(s).unwrap().name(), s);
        }
        assert_eq!(
            CutRule::parse("grid2048+asst").unwrap().name(),
            "asst+grid2048"
        );
        for bad in [
            "",
            "grid",
            "grid100",
            "grid0",
            "asst+asst",
            "grid64+grid128",
            "end",
        ] {
            assert!(CutRule::parse(bad).is_err(), "{bad}");
        }
        assert_eq!(
            CutLaw::qwen4_rule(64, 512, CutRule::parse("today").unwrap()),
            CutLaw::qwen4(64, 512)
        );
    }

    #[test]
    fn lri_asst_cut_is_where_a_stripped_next_turn_still_agrees() {
        let today = CutLaw::qwen4(DEFAULT_MIN_GAP, DEFAULT_SYS_MIN);
        let asst = CutLaw::qwen4_rule(
            DEFAULT_MIN_GAP,
            DEFAULT_SYS_MIN,
            CutRule::parse("asst").unwrap(),
        );
        let (hist, ends) = stripped_turns(3);
        for (k, &e) in ends.iter().enumerate() {
            // prompt k = the history before its assistant message + the generation prompt
            let mut prompt: Vec<u32> = hist[..e - 1].to_vec();
            prompt.push(NL);
            assert_eq!(&prompt[e - 5..], &gen_on()[..]);
            let next_end = ends.get(k + 1).copied().unwrap_or(hist.len());
            let next = &hist[..next_end];
            // the two prompts agree up to the `<think>` token: index e - 1 is NL vs NL2
            let common = prompt.iter().zip(next).take_while(|(a, b)| a == b).count();
            assert_eq!(common, e - 1);
            for (law, want) in [(&today, e), (&asst, e - 2)] {
                let pc = cuts_all(law, &prompt, &[prompt.len()]);
                let nc = cuts_all(law, next, &[next.len()]);
                // the deepest cut of prompt k inside the common part, and whether turn k + 1 has it
                let deepest = pc
                    .iter()
                    .map(|c| c.0 as usize)
                    .filter(|&q| q <= common)
                    .max();
                let last = pc.last().unwrap().0 as usize;
                assert_eq!(last, want, "the prompt's last cut");
                if law == &asst {
                    // the cut 2 before the end is a stored piece of prompt k AND a cut of turn k+1
                    assert_eq!(deepest, Some(e - 2));
                    assert!(nc.iter().any(|c| c.0 as usize == e - 2), "{nc:?}");
                    assert!(!CutLaw::end_node(&pc, &{
                        let mut st = LawState::default();
                        law.feed(&mut st, &prompt);
                        st
                    }));
                } else {
                    // today: the END (at e) is past the common part; turn k + 1 cuts at e + 2
                    assert!(deepest.is_none_or(|q| q + 64 < e), "{deepest:?}");
                    assert!(nc.iter().any(|c| c.0 as usize == e + 2), "{nc:?}");
                }
            }
        }
        // the asst law is prefix-stable and split-independent like today's
        let full = cuts_all(&asst, &hist, &[hist.len()]);
        for split in [vec![1usize; hist.len()], vec![13; hist.len()]] {
            let mut st = LawState::default();
            let mut got = Vec::new();
            let mut a = 0;
            for c in split {
                if a >= hist.len() {
                    break;
                }
                let b = (a + c).min(hist.len());
                got.extend(asst.feed(&mut st, &hist[a..b]));
                a = b;
            }
            assert_eq!(got, full);
        }
        for n in (1..hist.len()).step_by(37) {
            let p = cuts_all(&asst, &hist[..n], &[n]);
            let want: Vec<_> = full.iter().copied().filter(|c| c.0 <= n as u32).collect();
            assert_eq!(p, want, "prefix {n}");
        }
    }

    #[test]
    fn lri_grid_cuts_every_interval_and_stays_prefix_stable() {
        let law = CutLaw::qwen4_rule(64, 512, CutRule::parse("grid256").unwrap());
        let (hist, _) = stripped_turns(4);
        let full = cuts_all(&law, &hist, &[hist.len()]);
        for q in (256..hist.len() as u32).step_by(256) {
            assert!(full.iter().any(|c| c.0 == q), "grid {q}: {full:?}");
        }
        // the S and T cuts are still there
        assert_eq!(full.iter().filter(|c| c.1 == CutKind::Sys).count(), 1);
        assert!(full.iter().any(|c| c.1 == CutKind::Turn));
        // the scheduler's chunking (512 = a grid multiple) moves nothing; prefixes agree
        assert_eq!(cuts_all(&law, &hist, &[512; 64]), full);
        for n in (1..hist.len()).step_by(29) {
            let p = cuts_all(&law, &hist[..n], &[n]);
            let want: Vec<_> = full.iter().copied().filter(|c| c.0 <= n as u32).collect();
            assert_eq!(p, want, "prefix {n}");
        }
    }

    #[test]
    fn the_cursor_never_straddles_a_page() {
        let mut c = Cursor::new(100);
        assert_eq!(c.put(60), Some((0, 0)));
        assert_eq!(c.put(60), Some((1, 0)));
        assert_eq!(c.put(101), None);
        let g = c.put_blocks(7, 30).unwrap();
        // page 1 has 40 left: one block; then 3 a page
        assert_eq!(g, vec![(0, 1, 1, 60), (1, 3, 2, 0), (4, 3, 3, 0)]);
        let s = c.put_split(250);
        assert_eq!(
            s,
            vec![
                (0, 10, 3, 90),
                (10, 100, 4, 0),
                (110, 100, 5, 0),
                (210, 40, 6, 0)
            ]
        );
        assert_eq!(c.pages(), 7);
        for (_, n, _, off) in &s {
            assert!(off + n <= 100);
        }
    }

    fn k(i: u8) -> Key {
        [i; 32]
    }

    #[test]
    fn eviction_is_lrfu_by_subtree_heat_and_skips_pins() {
        let mut t = RamTree::new(1 << 20, 10, 0.98);
        // a chain a -> b -> c (a conversation) and a lone d
        for (key, parent, n) in [
            (k(1), None, 3),
            (k(2), Some(k(1)), 2),
            (k(3), Some(k(2)), 2),
        ] {
            let (p, g) = t.alloc(n).unwrap();
            assert!(g.is_empty());
            t.insert(key, parent, 100, p, u64::from(n) << 20);
            t.tick();
        }
        let (p, _) = t.alloc(2).unwrap();
        t.insert(k(4), None, 50, p, 2 << 20);
        t.tick();
        // c is hot: a and b inherit its heat; d is touched once and is colder than the chain
        for _ in 0..3 {
            t.touch(&k(3));
            t.tick();
        }
        assert_eq!(t.victims()[0], k(4), "the lone cold node goes first");
        assert_eq!(t.evict_plan(3), Some(vec![k(4)]));
        assert_eq!(t.evict_plan(1), Some(vec![]));
        assert_eq!(t.evict_plan(11), None);
        // need 3 pages with 1 free: d (2) is enough
        let (p, gone) = t.alloc(3).unwrap();
        assert_eq!(gone, vec![k(4)]);
        t.release(p);
        t.check().unwrap();
        // pinned nodes never leave; an impossible insert is refused
        t.pin(&k(1), true);
        t.pin(&k(2), true);
        t.pin(&k(3), true);
        assert!(t.alloc(4).is_none());
        assert_eq!(t.counts.refused, 1);
        t.pin(&k(3), false);
        // the leaf c leaves before its ancestors (they are pinned anyway); then the chain order
        let (p, gone) = t.alloc(5).unwrap();
        assert_eq!(gone, vec![k(3)]);
        t.release(p);
        t.check().unwrap();
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn a_leaf_leaves_before_its_parent_at_equal_heat() {
        let mut t = RamTree::new(1, 4, 0.98);
        let (p, _) = t.alloc(2).unwrap();
        // the parent's key sorts AFTER the child's: the depth tie-break, not the key, decides
        t.insert(k(9), None, 10, p, 2);
        let (p, _) = t.alloc(2).unwrap();
        t.insert(k(2), Some(k(9)), 20, p, 2);
        assert_eq!(t.victims(), vec![k(2), k(9)]);
        // a hotter parent stays after its child too; a hotter CHILD lifts its parent
        t.tick();
        t.touch(&k(2));
        assert_eq!(t.victims(), vec![k(2), k(9)]);
    }
}
