//! FN45c2: the PROMPT / CONVERSATION CACHE in the serve plane (`--fn-kv-pcache-gib`; FN45c's
//! `saber_mem::kv_ssd::PromptCache` on the drive).
//!
//! **The law.** A prompt chunk's result is a function of the sequence's state before it and the
//! chunk's ids, and the chunk law (EXEC §16, FN13) makes it a function of the CUT as well -- so the
//! cache is keyed by the CHAIN of the chunks themselves: `key_0 = sha256(sha256(model) || ids of
//! chunk 0)`, `key_k = sha256(sha256(key_{k-1}) || ids of chunk k)` ([`next_key`]); two prompts
//! share an entry only when every chunk up to it was cut at the same positions with the same ids,
//! so a reload is the state the full prefill would have produced by construction. After every
//! NON-LAST chunk the sequence's whole state is stored ([`store`]); a later non-last chunk whose
//! key is stored on BOTH ranks reloads it ([`reload`]) instead of running. The last chunk always
//! runs (its logits pick the first token).
//!
//! **What a sequence carries** (= exactly what `Pools::zero` clears, + the host's `Live`): the KV
//! pages of the 13 QSA layers (12 target + the MTP layer) through the tier's table, the indexer
//! ring + its positions + the compressed store of each QSA layer, the conv window and the GDN
//! recurrent state (at its ping-pong side) of each GDN layer, the PLE history (at its side), and
//! the host's length / chunk count / sides / PLE context / the drafter's stash.
//!
//! **FN45c3: the served default.** (a) The store is ASYNC: the D2H of the snapshot stays on the
//! step (the next kernels overwrite the state), the drive write (`write_entry`: file, fsync,
//! rename, the manifest record) runs on its own thread from the pinned staging buffer, which the
//! step waits for only when it needs the buffer again (the next store or reload);
//! `--fn-kv-pcache-sync` = the store inside the step (the control). (b) CONVERSATIONS: a chat
//! prompt's last chunk is cut at its last `<|im_start|>` (the generation prompt) and the part
//! before it is stored like any non-last chunk -- the next turn repeats exactly that prefix (the
//! template re-renders the assistant's reply), so it reloads it and prefills only the new turn.
//! The lookup is the LONGEST stored prefix of the chunk (`probe`: every stored length inside the
//! chunk confirmed by its chained key, both ranks agreeing), so a reload may end mid-chunk; the
//! rest of the chunk then runs from there -- the chunk law makes the bytes those of a prefill CUT
//! at the reload point (`--fn-kv-pcache-verify` computes that cut instead of reloading: the byte
//! law's reference).

use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use anyhow::{ensure, Context};
use saber_hip::{HipThread, Job, MemcpyKind, PinnedBuf};
use saber_mem::kv_ssd::{read_checked, write_entry, write_record, PromptCache};
use saber_mem::kv_tier::{prefix_keys, PAGE_TOKENS};
use saber_mem::prefix_tree::{Cursor, CutLaw, Key, RamTree};
use std::collections::HashMap;

use super::{plan, LayerKind, Live, Pools, Rank, LAYERS, RING};

const MAGIC: u64 = 0x4e46_3435_6332_5043; // "CP2c54FN"
/// The fixed header fields (u64 each) before the stash.
const HDR_WORDS: usize = 16;

/// The chained key of the next chunk (`prev` = the model's bytes before the first chunk).
#[must_use]
pub fn next_key(prev: &[u8], ids: &[u32]) -> [u8; 32] {
    prefix_keys(prev, ids, ids.len().max(1) as u32)
        .first()
        .map_or([0; 32], |k| k.1)
}

/// FN125: `lv`'s image runs + content hashes (empty for a text sequence).
fn image_keys(lv: &Live) -> &[saber_sched::prefix_key::ImageKey] {
    lv.img.as_ref().map_or(&[], |im| im.keys.as_slice())
}

/// FN125: the two words ("FN12" "5img") that open a piece key's image fold.
const IMG_TAG: [u32; 2] = [0x3231_4e46, 0x676d_6935];

/// FN125: the chained key of the piece `ids` at prompt positions `p0..p0 + ids.len()` of a
/// sequence with `images` (`saber_sched::prefix_key`'s law over a PIECE): the text key
/// `t = `[`next_key`]`(prev, ids)`, and -- only when some image run overlaps the piece
/// ([`ImageKey::overlaps`], the one overlap rule) -- `t` folded with each such image's
/// `(start, tokens, content hash)` in prompt order, by the same chain step:
/// `next_key(t, [IMG_TAG, start, tokens, hash lo, hash hi, ...])` = `sha256(sha256(t) || words LE)`.
/// A piece no image touches (every text prompt's every piece) keeps the text key byte for byte;
/// a piece AFTER an image carries it through the chain (`prev`); an image run straddling a
/// piece boundary is folded into every piece it overlaps.
///
/// [`ImageKey::overlaps`]: saber_sched::prefix_key::ImageKey::overlaps
#[must_use]
pub fn piece_key(
    prev: &[u8],
    p0: u32,
    ids: &[u32],
    images: &[saber_sched::prefix_key::ImageKey],
) -> [u8; 32] {
    let text = next_key(prev, ids);
    let hi = p0 + ids.len() as u32;
    let mut words: Vec<u32> = Vec::new();
    for im in images.iter().filter(|im| im.overlaps(p0, hi)) {
        if words.is_empty() {
            words.extend_from_slice(&IMG_TAG);
        }
        words.extend_from_slice(&[im.start, im.tokens, im.hash as u32, (im.hash >> 32) as u32]);
    }
    if words.is_empty() {
        text
    } else {
        next_key(&text, &words)
    }
}

/// The KV blocks a chunk at `p0..len` touches: from the block of `p0 - 1` (the drafter's stash
/// row at a chunk boundary) to the block past `len - 1`.
#[must_use]
pub fn chunk_blocks(p0: u32, len: u32) -> (u32, u32) {
    (
        p0.saturating_sub(1) / PAGE_TOKENS,
        len.div_ceil(PAGE_TOKENS),
    )
}

/// How [`regions`] sees the compressed index store.
#[derive(Clone, Copy)]
enum IxView<'a> {
    /// The identity layout's sizes (an upper bound; the staging buffer's size).
    Sizes,
    /// The live layout: identity, or (FN113) the paged allocator's row + its pad page.
    Rows(Option<(&'a [i32], i32)>),
}

/// The device regions of `slot` in snapshot order at length `len`: each LOGICAL region as its
/// device segments (one segment, except the compressed index store under admission: FN113 reads it
/// through the paged allocator's row -- runs of consecutive pages, FN51c's capture law).
fn regions(
    p: &Pools,
    slot: u32,
    len: u32,
    gdn_at: usize,
    ple_at: usize,
    ix: IxView<'_>,
) -> anyhow::Result<Vec<Vec<(u64, u64)>>> {
    const PAGE_B: u64 = 4 * 256;
    let mut out = Vec::new();
    let comp_blocks = u64::from((len.div_ceil(PAGE_TOKENS) + 1).min(p.nbs));
    // FN113: the paged store's blocks of this length (the pages `begin` opened)
    let paged: Option<Vec<(u64, u64)>> = match ix {
        IxView::Sizes => None,
        IxView::Rows(Some((row, pad))) if p.ix_paged => {
            let nb = len.div_ceil(PAGE_TOKENS) as usize;
            ensure!(
                row.len() >= nb,
                "the index row holds {} of {nb} blocks",
                row.len()
            );
            let mut runs: Vec<(u64, u64)> = Vec::new();
            for (j, &pg) in row[..nb].iter().enumerate() {
                ensure!(
                    pg >= 0 && pg != pad,
                    "index block {j} of slot {slot} has no page"
                );
                let pg = pg as u64;
                match runs.last_mut() {
                    Some(r) if r.0 + r.1 == pg => r.1 += 1,
                    _ => runs.push((pg, 1)),
                }
            }
            Some(runs)
        }
        IxView::Rows(_) => {
            ensure!(!p.ix_paged, "a paged index store without its row");
            None
        }
    };
    for l in 0..=LAYERS as usize {
        let v = p.seq(slot, l);
        if l == LAYERS as usize || plan::kind(l as u32) == LayerKind::Qsa {
            out.push(vec![(v.ring, RING * 256)]);
            out.push(vec![(v.ringp, RING * 24)]);
            match paged.as_ref() {
                Some(runs) => out.push(
                    runs.iter()
                        .map(|&(pg, n)| (v.comp + pg * PAGE_B, n * PAGE_B))
                        .collect(),
                ),
                None => out.push(vec![(v.comp, comp_blocks * PAGE_B)]),
            }
        } else {
            out.push(vec![(v.cst, p.cb)]);
            out.push(vec![(v.h[gdn_at], p.hb)]);
        }
    }
    out.push(vec![(p.seq(slot, 0).ple_st[ple_at], p.pb)]);
    Ok(out)
}

/// The regions of `slot` on this rank (FN113: the paged index row read from the tier).
fn regions_of(
    rk: &Rank<'_, '_>,
    slot: u32,
    len: u32,
    gdn_at: usize,
    ple_at: usize,
) -> anyhow::Result<Vec<Vec<(u64, u64)>>> {
    if !rk.pools.ix_paged {
        return regions(&rk.pools, slot, len, gdn_at, ple_at, IxView::Rows(None));
    }
    let kv = rk
        .pools
        .kvt
        .as_ref()
        .context("a paged store without the tier")?;
    let k = kv.borrow();
    let ix = k.ix.as_ref().context("the paged index pool")?;
    let row = ix.row(slot).to_vec();
    let pad = ix.pad();
    drop(k);
    regions(
        &rk.pools,
        slot,
        len,
        gdn_at,
        ple_at,
        IxView::Rows(Some((&row, pad))),
    )
}

/// The segments of logical regions in byte order (the snapshot's flat region stream).
fn flat(regs: &[Vec<(u64, u64)>]) -> Vec<(u64, u64)> {
    regs.iter().flatten().copied().collect()
}

/// FN45c3: the longest tail (tokens) after a prompt's last `<|im_start|>` that is cut off as its
/// own last chunk (the generation prompt is 3..6 tokens; a longer tail is not a chat turn's).
pub const TURN_TAIL_MAX: usize = 64;

/// FN45c3: the PINNED bounce a rank moves a snapshot through (16 MiB; declared in the host pin
/// law beside `kv.ram`). The snapshot itself is assembled in a PAGEABLE buffer the writer thread
/// owns while it writes (the pinned RAM is at its ceiling beside the expert RAM cache: a whole
/// 8,192-row snapshot, ~0.14 GiB a rank, would not fit).
pub const BOUNCE: usize = 16 << 20;

/// The cache's switches at a boot.
#[derive(Debug, Clone, Copy, Default)]
pub struct PcacheOpts {
    /// `--fn-kv-pcache-sync`: the drive write inside the step (the async store's control).
    pub sync: bool,
    /// `--fn-kv-pcache-verify`: a hit is COMPUTED at the same cut instead of reloaded.
    pub verify: bool,
    /// The `<|im_start|>` id of the checkpoint's tokenizer (None = no conversation cut).
    pub turn_tok: Option<u32>,
    /// The shortest prefix a conversation cut stores (`--fn-kv-pcache-min-tokens`).
    pub min_tokens: u32,
    /// FN45c4: the conversation cut is rounded DOWN to a multiple of this many tokens
    /// (`--fn-kv-pcache-align`, default 1 = FN45c3's cut; 64 = FN13's aligned boundary, worse KL).
    pub align: u32,
    /// FN66 (`--warm-prompts`): the `<|im_end|>` id of the WARM CUT ([`warm_cut`]; None = off).
    pub warm_end: Option<u32>,
    /// FN113b (`--fn-prefix-tree-demote off`, a DIAGNOSTIC): the tree's evicted nodes are only
    /// dropped (no lazy demotion: nothing written to the drive).
    pub no_demote: bool,
}

/// The cache on one rank.
pub(super) struct Pcache {
    cache: Arc<Mutex<PromptCache>>,
    model: Vec<u8>,
    bounce: PinnedBuf,
    /// The pageable snapshot buffer (None while a writer owns it).
    snap: Option<Vec<u8>>,
    /// The writer of the last store (it hands the snapshot buffer back).
    inflight: Option<JoinHandle<(Vec<u8>, anyhow::Result<f64>)>>,
    pub opts: PcacheOpts,
    pub verify: bool,
    pub hits: u64,
    pub stores: u64,
    pub fallbacks: u64,
    pub hit_tokens: u64,
    pub reload_ms: f64,
    /// On-step store time (key check + D2H + the reservation; + the write under `sync`).
    pub store_ms: f64,
    /// The writer threads' time (file + fsync + rename + record).
    pub write_ms: f64,
    /// Time the step waited for a writer to release the buffer.
    pub wait_ms: f64,
    pub write_errors: u64,
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub turn_cuts: u64,
    pub verify_runs: u64,
    /// FN45c4: stores skipped because the last store's writer still held the snapshot buffer
    /// (the step never waits for the drive: a skipped store is a later prefill, never a byte).
    pub busy_skips: u64,
    /// FN45c4: the longest single on-step store / reload (ms) -- the stall's witness.
    pub worst_ms: f64,
    /// FN66: warm cuts run (a prompt ending `<|im_end|>` + one token stored at its last token).
    pub warm_cuts: u64,
    /// FN66: the next [`store`] waits for a busy writer instead of skipping (the warm cut's: a
    /// warm prompt exists to be stored).
    pub force_store: bool,
    /// FN113: the tiered prefix tree (`--fn-prefix-tree on`; None = FN45c's cache alone).
    pub tree: Option<Tree>,
    /// FN113b: the engine's QUIET stamp (ms since `base` the last command left no live sequence;
    /// 0 = a request is in flight): a tree demotion's drive write waits for [`QUIET_MS`] of it --
    /// h8: a demotion's write + fsync beside the step stalled the PLE drive gather 80-180 ms.
    quiet: Arc<std::sync::atomic::AtomicU64>,
    base: Instant,
}

impl Pcache {
    /// Open the cache at `dir` (`budget` bytes) for this boot's `model` identity; the staging
    /// buffer holds the widest chunk's snapshot at `max_ctx`.
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        hip: &HipThread,
        pools: &Pools,
        dir: &std::path::Path,
        budget: u64,
        model: String,
        max_ctx: u32,
        max_chunk: u32,
        opts: PcacheOpts,
        tree: Option<TreeOpts>,
    ) -> anyhow::Result<Pcache> {
        std::fs::create_dir_all(dir).with_context(|| format!("{}", dir.display()))?;
        let cache = PromptCache::open(dir, budget)
            .with_context(|| format!("--fn-kv-pcache-gib: the cache at {}", dir.display()))?;
        let bytes = staging_bytes(pools, max_ctx, max_chunk);
        let bounce = hip.host_malloc(BOUNCE as u64)?;
        Ok(Pcache {
            cache: Arc::new(Mutex::new(cache)),
            model: model.into_bytes(),
            bounce,
            snap: Some(vec![0u8; usize::try_from(bytes)?]),
            inflight: None,
            opts,
            verify: opts.verify,
            hits: 0,
            stores: 0,
            fallbacks: 0,
            hit_tokens: 0,
            reload_ms: 0.0,
            store_ms: 0.0,
            write_ms: 0.0,
            wait_ms: 0.0,
            write_errors: 0,
            bytes_in: 0,
            bytes_out: 0,
            turn_cuts: 0,
            verify_runs: 0,
            busy_skips: 0,
            worst_ms: 0.0,
            warm_cuts: 0,
            force_store: false,
            tree: tree.map(|o| Tree::open(hip, o)).transpose()?,
            quiet: Arc::new(std::sync::atomic::AtomicU64::new(1)),
            base: Instant::now(),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, PromptCache> {
        self.cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Join the last store's writer (the snapshot buffer comes back).
    pub fn wait(&mut self) {
        if let Some(h) = self.inflight.take() {
            let t0 = Instant::now();
            match h.join() {
                Ok((v, r)) => {
                    self.snap = Some(v);
                    match r {
                        Ok(ms) => {
                            self.write_ms += ms;
                            self.stores += 1;
                        }
                        Err(_) => self.write_errors += 1,
                    }
                }
                Err(_) => self.write_errors += 1,
            }
            self.wait_ms += t0.elapsed().as_secs_f64() * 1e3;
        }
    }

    /// The previous key of `lv`'s next chunk (the model's bytes before the first chunk).
    fn prev<'a>(&'a self, lv: &'a Live) -> &'a [u8] {
        if lv.chunks == 0 {
            &self.model
        } else {
            &lv.key
        }
    }

    /// The key of `lv`'s next chunk `ids` (FN125: [`piece_key`] -- the text key unless an image
    /// run of `lv` overlaps the chunk).
    #[must_use]
    pub fn key(&self, lv: &Live, ids: &[u32]) -> [u8; 32] {
        piece_key(self.prev(lv), lv.len, ids, image_keys(lv))
    }

    /// FN45c3: where a LAST chunk `ids` at `len` is cut for the conversation's next turn: before
    /// its last `<|im_start|>` when that leaves at least one token before it, a stored prefix of
    /// `min_tokens` or more, and a tail of at most [`TURN_TAIL_MAX`].
    #[must_use]
    pub fn turn_cut(&self, len: u32, ids: &[u32]) -> Option<usize> {
        turn_cut(
            self.opts.turn_tok?,
            self.opts.min_tokens,
            self.opts.align,
            len,
            ids,
        )
    }

    /// FN66: where a LAST chunk `ids` is cut under `--warm-prompts` ([`warm_cut`]).
    #[must_use]
    pub fn warm_cut(&self, ids: &[u32]) -> Option<usize> {
        warm_cut(self.opts.warm_end?, ids)
    }

    /// FN66: the cache handle (the serve plane's warm runner reads its size).
    #[must_use]
    pub fn handle(&self) -> Arc<Mutex<PromptCache>> {
        Arc::clone(&self.cache)
    }

    /// The `fnpc` line.
    #[must_use]
    pub fn line(&self, rank: u32) -> String {
        let l = self.lock().line();
        format!(
            "fnpc rank{rank}: hits {} ({} tokens, {:.1} ms reload, {:.1} MiB in) stores {} ({:.1} \
             ms on-step, {:.1} ms writer, {:.1} ms waited, {:.1} MiB out, {} errors, {} busy skips) \
             fallbacks {} turn cuts {} warm cuts {} verify runs {} worst {:.1} ms | {}",
            self.hits,
            self.hit_tokens,
            self.reload_ms,
            self.bytes_in as f64 / (1u64 << 20) as f64,
            self.stores,
            self.store_ms,
            self.write_ms,
            self.wait_ms,
            self.bytes_out as f64 / (1u64 << 20) as f64,
            self.write_errors,
            self.busy_skips,
            self.fallbacks,
            self.turn_cuts,
            self.warm_cuts,
            self.verify_runs,
            self.worst_ms,
            l
        )
    }
}

impl Drop for Pcache {
    fn drop(&mut self) {
        // the writer reads the pinned buffer: join it before the buffer is freed
        self.wait();
    }
}

/// FN68d: layer `l`'s span of a bounced group of `sn` blocks starting `j` blocks into a
/// snapshot of `nb` blocks at `at`, the layers' page bytes `lbb` (layer-major, the MTP layer's
/// twin at its own page bytes): `(snapshot offset, bounce offset, bytes)` -- the bounce holds the
/// group as `KvDev::dump` writes it (each layer's `sn` pages after the earlier layers').
#[must_use]
pub(super) fn snap_span(
    lbb: &[usize],
    l: usize,
    at: usize,
    nb: usize,
    j: usize,
    sn: usize,
) -> (usize, usize, usize) {
    let before: usize = lbb[..l].iter().sum();
    (at + before * nb + j * lbb[l], before * sn, sn * lbb[l])
}

/// FN68b: one block of every QSA layer at the tiers' dtypes (the snapshot's block unit; FN68d:
/// the MTP layer's twin at its own page bytes).
fn block_set_bb(rk: &Rank<'_, '_>) -> anyhow::Result<usize> {
    Ok(rk
        .pools
        .kvt
        .as_ref()
        .context("the prompt cache needs the tiered KV")?
        .borrow()
        .block_set_bytes() as usize)
}

/// The pinned staging bytes of one snapshot at `max_ctx` / `max_chunk` (the widest chunk's KV
/// pages + every region + the header with the drafter's stash; the pages at bf16's `BB`, an upper
/// bound under fp8).
#[must_use]
pub(super) fn staging_bytes(pools: &Pools, max_ctx: u32, max_chunk: u32) -> u64 {
    let regs: u64 = regions(pools, 0, max_ctx, 0, 0, IxView::Sizes)
        .map_or(0, |r| flat(&r).iter().map(|x| x.1).sum());
    let kv = u64::from(max_chunk.div_ceil(PAGE_TOKENS) + 2)
        * u64::from(saber_mem::kv_tier::QSA_LAYERS)
        * super::super::kvwire::BB;
    hdr_len(super::W as usize * 2) as u64 + kv + regs
}

/// FN45c3: the `<|im_start|>` id of the checkpoint's `tokenizer.json` (None: no such token, no
/// conversation cut).
#[must_use]
pub fn im_start_id(ckpt: &std::path::Path) -> Option<u32> {
    special_id(ckpt, "<|im_start|>")
}

/// FN66: an added token's id in the checkpoint's `tokenizer.json`.
#[must_use]
pub fn special_id(ckpt: &std::path::Path, tok: &str) -> Option<u32> {
    let raw = std::fs::read(ckpt.join("tokenizer.json")).ok()?;
    let v: serde_json::Value = serde_json::from_slice(&raw).ok()?;
    v.get("added_tokens")?
        .as_array()?
        .iter()
        .find(|t| t.get("content").and_then(|c| c.as_str()) == Some(tok))
        .and_then(|t| t.get("id")?.as_u64())
        .and_then(|i| u32::try_from(i).ok())
}

/// FN45c3: the cache's PINNED bytes a rank (the host pin law counts them beside `kv.ram`): the
/// bounce only, whatever the context and chunk.
#[must_use]
pub fn staging_bound(_max_ctx: u32, _max_chunk: u32) -> u64 {
    BOUNCE as u64
}

/// [`Pcache::turn_cut`] (pure). FN45c4: the cut (`len + j`, the last `<|im_start|>`) is rounded
/// DOWN to a multiple of `align` tokens; the tail is then at most `TURN_TAIL_MAX + align - 1`.
#[must_use]
pub fn turn_cut(tok: u32, min_tokens: u32, align: u32, len: u32, ids: &[u32]) -> Option<usize> {
    let j = ids.iter().rposition(|&t| t == tok)?;
    if ids.len() - j > TURN_TAIL_MAX {
        return None;
    }
    let a = align.max(1);
    let cut = (len + j as u32) / a * a;
    // `then` (lazy), not `then_some`: the subtraction must not be evaluated when `cut <= len`.
    (cut > len && cut >= min_tokens.max(1)).then(|| (cut - len) as usize)
}

/// FN66: the WARM CUT -- a LAST chunk that ends `<|im_end|>` + one token (a conversation head
/// rendered up to the next turn: what `--warm-prompts` submits; a served chat prompt ends in the
/// generation prompt instead) is cut before its last token, and the part before it is STORED like
/// a non-last chunk: a later prompt that begins with the head reloads it. The cut after the
/// `<|im_end|>` keeps one token as the last chunk (its logits pick the token the warm request
/// discards).
#[must_use]
pub fn warm_cut(im_end: u32, ids: &[u32]) -> Option<usize> {
    (ids.len() >= 2 && ids[ids.len() - 2] == im_end).then(|| ids.len() - 1)
}

/// FN45c3: the stored lengths `len < t <= len + hi` whose chained key (from `prev`) is held, the
/// longest first (`held(key, t)` asks the cache).
#[must_use]
pub fn held_lengths(
    prev: &[u8],
    len: u32,
    ids: &[u32],
    images: &[saber_sched::prefix_key::ImageKey],
    lengths: &[u32],
    held: impl Fn(&[u8; 32], u32) -> bool,
) -> Vec<(u32, [u8; 32])> {
    lengths
        .iter()
        .filter(|&&t| t > len && (t - len) as usize <= ids.len())
        .map(|&t| (t, piece_key(prev, len, &ids[..(t - len) as usize], images)))
        .filter(|(t, k)| held(k, *t))
        .collect()
}

fn hdr_len(stash: usize) -> usize {
    (HDR_WORDS * 8 + stash).div_ceil(4096) * 4096
}

/// The host fields of a snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    pub p0: u32,
    pub len: u32,
    pub chunks: u32,
    pub gdn_at: u32,
    pub ple_at: u32,
    pub ctx: [u32; 2],
    pub next: u32,
    pub naccept: u32,
    pub drop: u32,
    pub blocks: (u32, u32),
    pub regions: u64,
    pub total: u64,
    pub stash: Vec<u8>,
}

impl Header {
    /// The header's bytes (`hdr_len` of the stash).
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let w = [
            MAGIC,
            u64::from(self.p0),
            u64::from(self.len),
            u64::from(self.chunks),
            u64::from(self.gdn_at),
            u64::from(self.ple_at),
            u64::from(self.ctx[0]),
            u64::from(self.ctx[1]),
            u64::from(self.next),
            u64::from(self.naccept),
            u64::from(self.drop),
            u64::from(self.blocks.0),
            u64::from(self.blocks.1),
            self.regions,
            self.total,
            self.stash.len() as u64,
        ];
        let mut out: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
        out.extend_from_slice(&self.stash);
        out.resize(hdr_len(self.stash.len()), 0);
        out
    }

    /// Parse (refuses a foreign or short header).
    pub fn decode(b: &[u8]) -> anyhow::Result<Header> {
        ensure!(b.len() >= HDR_WORDS * 8, "a short prompt-cache header");
        let w = |i: usize| u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().expect("8"));
        ensure!(w(0) == MAGIC, "not a prompt-cache snapshot");
        let n = w(15) as usize;
        ensure!(b.len() >= HDR_WORDS * 8 + n, "a short stash");
        let u = |i: usize| u32::try_from(w(i)).unwrap_or(u32::MAX);
        Ok(Header {
            p0: u(1),
            len: u(2),
            chunks: u(3),
            gdn_at: u(4),
            ple_at: u(5),
            ctx: [u(6), u(7)],
            next: u(8),
            naccept: u(9),
            drop: u(10),
            blocks: (u(11), u(12)),
            regions: w(13),
            total: w(14),
            stash: b[HDR_WORDS * 8..HDR_WORDS * 8 + n].to_vec(),
        })
    }
}

/// FN45c3: the device regions between the pools and the pageable `snap` at `at`, through the
/// pinned bounce in groups (D2H when `h2d` is false).
fn move_regions_bounced(
    rk: &Rank<'_, '_>,
    bounce: &mut PinnedBuf,
    snap: &mut [u8],
    at: usize,
    regs: &[(u64, u64)],
    h2d: bool,
) -> anyhow::Result<usize> {
    let mut o = at;
    let mut i = 0;
    while i < regs.len() {
        let mut j = i;
        let mut g = 0usize;
        while j < regs.len() && g + regs[j].1 as usize <= BOUNCE {
            g += regs[j].1 as usize;
            j += 1;
        }
        ensure!(
            j > i,
            "a {} B region passes the {BOUNCE} B bounce",
            regs[i].1
        );
        ensure!(o + g <= snap.len(), "the snapshot buffer");
        if h2d {
            bounce.as_mut_slice()[..g].copy_from_slice(&snap[o..o + g]);
            move_regions(rk, bounce, 0, &regs[i..j], true)?;
        } else {
            move_regions(rk, bounce, 0, &regs[i..j], false)?;
            snap[o..o + g].copy_from_slice(&bounce.as_slice()[..g]);
        }
        o += g;
        i = j;
    }
    Ok(o - at)
}

/// FN45c3: the KV pages of `slot`'s `blocks` between the tier and the pageable `snap` at `at` in
/// the tier's layer-major layout (`KvDev::dump` / `load`), through the bounce in block groups.
fn move_kv_bounced(
    rk: &Rank<'_, '_>,
    bounce: &mut PinnedBuf,
    snap: &mut [u8],
    at: usize,
    slot: u32,
    (b0, b1): (u32, u32),
    h2d: bool,
) -> anyhow::Result<usize> {
    let kv = rk
        .pools
        .kvt
        .as_ref()
        .context("the prompt cache needs the tiered KV")?;
    // FN68b: the tier's page bytes at the boot's dtype (bf16 = `kvwire::BB`, fp8 = half);
    // FN68d: each layer's own (the MTP layer's twin at its dtype), layer-major
    let q = saber_mem::kv_tier::QSA_LAYERS as usize;
    let lbb: Vec<usize> = {
        let k = kv.borrow();
        (0..q as u32).map(|l| k.layer_bb(l) as usize).collect()
    };
    let set: usize = lbb.iter().sum();
    let nb = (b1 - b0) as usize;
    ensure!(at + nb * set <= snap.len(), "the snapshot buffer");
    let step = (BOUNCE / set).max(1) as u32;
    let mut s0 = b0;
    while s0 < b1 {
        let s1 = (s0 + step).min(b1);
        let sn = (s1 - s0) as usize;
        let span = |l: usize| snap_span(&lbb, l, at, nb, (s0 - b0) as usize, sn);
        if h2d {
            for l in 0..q {
                let (d, b, n) = span(l);
                bounce.as_mut_slice()[b..b + n].copy_from_slice(&snap[d..d + n]);
            }
            kv.borrow_mut()
                .load(rk.hip, &rk.dev.stream, slot, (s0, s1), bounce, 0)?;
        } else {
            kv.borrow()
                .dump(rk.hip, &rk.dev.stream, slot, (s0, s1), bounce, 0)?;
            for l in 0..q {
                let (d, b, n) = span(l);
                snap[d..d + n].copy_from_slice(&bounce.as_slice()[b..b + n]);
            }
        }
        s0 = s1;
    }
    Ok(nb * set)
}

/// D2H (`h2d` false) or H2D the device regions between the pools and `buf` at `at`.
fn move_regions(
    rk: &Rank<'_, '_>,
    buf: &PinnedBuf,
    at: usize,
    regs: &[(u64, u64)],
    h2d: bool,
) -> anyhow::Result<usize> {
    let mut jobs = Vec::with_capacity(regs.len());
    let mut o = at;
    for &(a, n) in regs {
        let (d, off) = rk.pools.owner(a)?;
        ensure!(o + n as usize <= buf.len(), "the snapshot buffer");
        jobs.push(Job::MemcpyPinned {
            dev: d,
            dev_offset: off,
            host: buf,
            host_offset: o as u64,
            bytes: n,
            kind: if h2d {
                MemcpyKind::HostToDevice
            } else {
                MemcpyKind::DeviceToHost
            },
            stream: &rk.dev.stream,
        });
        o += n as usize;
    }
    rk.hip.run_batch(&jobs)?;
    rk.hip.sync_stream(&rk.dev.stream)?;
    Ok(o - at)
}

/// After a non-last chunk of `lv` that started at `p0`: store its state at `key` (kept if
/// already stored). Both ranks run it (each its own half). FN45c3: the D2H runs on the step, the
/// drive write on a writer thread (joined before the staging buffer is reused).
pub(super) fn store(
    rk: &Rank<'_, '_>,
    pc: &mut Pcache,
    lv: &Live,
    p0: u32,
    key: [u8; 32],
) -> anyhow::Result<()> {
    // FN113: the tree stores into its RAM tier (+ the drive write-through); a refusal falls back
    if pc.tree.is_some() {
        let parent = (p0 > 0).then_some(lv.key);
        if tree_store(rk, pc, lv, p0, key, parent)? {
            return Ok(());
        }
    }
    if pc.lock().get(&key).is_some() {
        return Ok(());
    }
    // FN45c4: the step never waits for the drive -- a writer still busy = this store is skipped
    // (FN66: except the warm cut's, `force_store`)
    let force = std::mem::take(&mut pc.force_store);
    if pc.inflight.as_ref().is_some_and(|h| !h.is_finished()) && !pc.opts.sync && !force {
        pc.busy_skips += 1;
        return Ok(());
    }
    pc.wait();
    let t0 = Instant::now();
    let mut snap = pc.snap.take().context("the snapshot buffer")?;
    let r = fill_snapshot(rk, pc, &mut snap, lv, p0);
    let total = match r {
        Ok(t) => t,
        Err(e) => {
            pc.snap = Some(snap);
            return Err(e);
        }
    };
    if !spawn_writer(pc, key, lv.len, snap, total, false)? {
        pc.store_ms += t0.elapsed().as_secs_f64() * 1e3;
        return Ok(());
    }
    if pc.opts.sync {
        pc.wait();
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    pc.store_ms += ms;
    pc.worst_ms = pc.worst_ms.max(ms);
    Ok(())
}

/// The tier opens a piece's prefill makes (its cuts): `begin` over `p0..p0 + m` at `fn_chunk`.
/// FN130: the demand law first -- a reload opens a whole piece's pages with no prefill chunk (no
/// elastic tick), so the lent segments the piece needs come back here, before any page opens.
fn open_piece(
    rk: &mut Rank<'_, '_>,
    slot: u32,
    p0: u32,
    m: u32,
    fn_chunk: u32,
) -> anyhow::Result<()> {
    let cut = if fn_chunk > 0 && fn_chunk < m {
        fn_chunk
    } else {
        m
    };
    let mut a = 0;
    while a < m {
        let b = (a + cut).min(m);
        super::elastic_demand(rk, &[(slot, p0 + a, p0 + b)], "reload")?;
        let kv = rk
            .pools
            .kvt
            .as_ref()
            .context("the prompt cache needs the tiered KV")?;
        kv.borrow_mut()
            .begin(rk.hip, &rk.dev.stream, &[(slot, p0 + a, p0 + b)], true)?;
        a = b;
    }
    Ok(())
}

/// Reserve `key` (a `tokens`-long prefix of `total` bytes in `snap`) on the drive and hand the
/// snapshot to a writer thread (false: not reserved -- the buffer is back in `pc.snap`).
/// FN113b: the quiet a tree demotion's drive write waits for (no live sequence this long), and
/// the longest it waits before writing anyway (a boot that never goes idle still demotes).
pub const QUIET_MS: u64 = 50;
pub const QUIET_WAIT_MAX_MS: u64 = 60_000;

/// FN113b: a command is about to run (the engine is busy).
pub(super) fn mark_busy(pc: &Pcache) {
    pc.quiet.store(0, std::sync::atomic::Ordering::Release);
}

/// FN113b: a command ran; `live` = a sequence is still in flight (else the quiet stamp starts).
pub(super) fn mark_after(pc: &Pcache, live: bool) {
    use std::sync::atomic::Ordering;
    if live {
        pc.quiet.store(0, Ordering::Release);
    } else if pc.quiet.load(Ordering::Acquire) == 0 {
        let ms = pc.base.elapsed().as_millis() as u64;
        pc.quiet.store(ms.max(1), Ordering::Release);
    }
}

fn spawn_writer(
    pc: &mut Pcache,
    key: [u8; 32],
    tokens: u32,
    snap: Vec<u8>,
    total: usize,
    quiet_gate: bool,
) -> anyhow::Result<bool> {
    let Some(r) = pc
        .lock()
        .reserve(key, tokens, total as u64)
        .context("prompt-cache reserve")?
    else {
        pc.snap = Some(snap);
        return Ok(false);
    };
    pc.bytes_out += total as u64;
    let cache = Arc::clone(&pc.cache);
    let manifest = pc.lock().manifest();
    let quiet = quiet_gate.then(|| (Arc::clone(&pc.quiet), pc.base));
    pc.inflight = Some(std::thread::spawn(move || {
        // FN113b: a tree demotion's write runs only in a QUIET engine (no request in flight for
        // QUIET_MS): its write + fsync never meets a step's drive reads
        if let Some((q, base)) = quiet {
            let t0 = Instant::now();
            loop {
                let s = q.load(std::sync::atomic::Ordering::Acquire);
                let now = base.elapsed().as_millis() as u64;
                if (s != 0 && now.saturating_sub(s) >= QUIET_MS)
                    || t0.elapsed().as_millis() as u64 >= QUIET_WAIT_MAX_MS
                {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
        let t = Instant::now();
        // FN45c4: file + fsync + the record's fsync with NO cache lock; the lock only publishes
        let w = write_entry(&r, &[&snap[..total]])
            .and_then(|d| write_record(&manifest, &r, d).map(|()| d));
        let mut c = cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let res = match w {
            Ok(d) => {
                c.publish(r, d);
                Ok(())
            }
            Err(e) => {
                c.abort(&r);
                Err(anyhow::Error::from(e))
            }
        };
        drop(c);
        (snap, res.map(|()| t.elapsed().as_secs_f64() * 1e3))
    }));
    Ok(true)
}

/// The snapshot of `lv` (a chunk from `p0`) into `snap`: header, KV pages, regions. Its length.
fn fill_snapshot(
    rk: &Rank<'_, '_>,
    pc: &mut Pcache,
    snap: &mut [u8],
    lv: &Live,
    p0: u32,
) -> anyhow::Result<usize> {
    let blocks = chunk_blocks(p0, lv.len);
    let lregs = regions_of(rk, lv.slot, lv.len, lv.gdn_at, lv.ple_at)?;
    let regs = flat(&lregs);
    let mut h = Header {
        p0,
        len: lv.len,
        chunks: lv.chunks,
        gdn_at: lv.gdn_at as u32,
        ple_at: lv.ple_at as u32,
        ctx: lv.ctx.0,
        next: lv.next,
        naccept: lv.naccept,
        drop: lv.drop,
        blocks,
        regions: lregs.len() as u64,
        total: 0,
        stash: lv.stash.clone(),
    };
    let hl = hdr_len(h.stash.len());
    let nkv = move_kv_bounced(rk, &mut pc.bounce, snap, hl, lv.slot, blocks, false)?;
    let nreg = move_regions_bounced(rk, &mut pc.bounce, snap, hl + nkv, &regs, false)?;
    h.total = (hl + nkv + nreg) as u64;
    let hb = h.encode();
    ensure!(hb.len() == hl, "the header's length");
    snap[..hl].copy_from_slice(&hb);
    Ok(h.total as usize)
}

/// FN45c3: the longest prefix of `lv` + `ids` stored on BOTH ranks (one exchange of each rank's
/// held lengths), at most `ids.len() - 1` tokens into a LAST chunk (its last token always runs:
/// its logits pick the first token). `(length, key)`.
pub(super) fn probe(
    rk: &mut Rank<'_, '_>,
    pc: &mut Pcache,
    lv: &Live,
    ids: &[u32],
    last: bool,
) -> anyhow::Result<Option<(u32, [u8; 32])>> {
    let hi = ids.len().saturating_sub(usize::from(last));
    let mine = {
        let c = pc.lock();
        let lengths = c.lengths_in(lv.len + 1, lv.len + hi as u32);
        held_lengths(
            pc.prev(lv),
            lv.len,
            &ids[..hi],
            image_keys(lv),
            &lengths,
            |k, t| c.get(k).is_some_and(|e| e.tokens == t),
        )
    };
    let mut b = Vec::with_capacity(mine.len() * 4);
    for (t, _) in &mine {
        b.extend_from_slice(&t.to_le_bytes());
    }
    let peer = rk.peer.exchange().xchg_bytes("fn.pcache.t", &b)?;
    let theirs: Vec<u32> = peer
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().expect("4")))
        .collect();
    Ok(mine.into_iter().find(|(t, _)| theirs.contains(t)))
}

/// Before `ids` of `lv` at `key` (a stored prefix `probe` found): when BOTH ranks load and parse
/// it (one exchange), restore it into `lv`'s slot and return true; else false (the caller
/// prefills).
pub(super) fn reload(
    rk: &mut Rank<'_, '_>,
    pc: &mut Pcache,
    lv: &mut Live,
    ids: &[u32],
    key: [u8; 32],
    fn_chunk: u32,
) -> anyhow::Result<bool> {
    let t0 = Instant::now();
    let (had, parsed) = read_drive(pc, lv, ids, &key);
    let ok = u8::from(parsed.is_some());
    let peer = rk.peer.exchange().xchg_bytes("fn.pcache", &[ok])?;
    let Some((h, mine)) = parsed.filter(|_| peer.first() == Some(&1)) else {
        if had || peer.first() == Some(&1) {
            pc.fallbacks += 1;
        }
        return Ok(false);
    };
    apply_drive(rk, pc, lv, ids, key, fn_chunk, h, mine, t0)?;
    Ok(true)
}

/// FN45c4: a drive entry for the piece `ids` after `lv` (read OUTSIDE the cache's lock: the lock
/// covers the index lookup, then touch / drop): (an entry was found, the parsed entry if it fits).
fn read_drive(
    pc: &mut Pcache,
    lv: &Live,
    ids: &[u32],
    key: &[u8; 32],
) -> (bool, Option<(Header, saber_mem::dio::AlignedBuf)>) {
    let found = pc.lock().entry_file(key);
    let mine = found.and_then(|(path, e)| {
        let b = read_checked(&path, &e);
        let _ = pc.lock().settle(key, e.id, b.is_some());
        b
    });
    let had = mine.is_some();
    let parsed = mine.and_then(|b| {
        let h = Header::decode(&b).ok()?;
        (h.p0 == lv.len && h.len == lv.len + ids.len() as u32 && h.total as usize <= b.len())
            .then_some((h, b))
    });
    (had, parsed)
}

/// Restore a parsed drive entry into `lv`'s slot (both ranks agreed).
#[allow(clippy::too_many_arguments)]
fn apply_drive(
    rk: &mut Rank<'_, '_>,
    pc: &mut Pcache,
    lv: &mut Live,
    ids: &[u32],
    key: [u8; 32],
    fn_chunk: u32,
    h: Header,
    mut mine: saber_mem::dio::AlignedBuf,
    t0: Instant,
) -> anyhow::Result<()> {
    let total = h.total as usize;
    // the read buffer itself (no copy: FN45c3 copied every reload into a fresh allocation)
    let snap: &mut [u8] = &mut mine[..total];
    let hl = hdr_len(h.stash.len());
    // the tier: the very opens the chunk's prefill makes (its cuts), then the pages
    let m = ids.len() as u32;
    open_piece(rk, lv.slot, h.p0, m, fn_chunk)?;
    {
        let n = move_kv_bounced(rk, &mut pc.bounce, snap, hl, lv.slot, h.blocks, true)?;
        ensure!(
            n == (h.blocks.1 - h.blocks.0) as usize * block_set_bb(rk)?,
            "the snapshot's pages"
        );
    }
    let (gdn_at, ple_at) = (h.gdn_at as usize, h.ple_at as usize);
    let lregs = regions_of(rk, lv.slot, h.len, gdn_at, ple_at)?;
    ensure!(lregs.len() as u64 == h.regions, "the snapshot's regions");
    let regs = flat(&lregs);
    let kvn = (h.blocks.1 - h.blocks.0) as usize * block_set_bb(rk)?;
    let nreg = move_regions_bounced(rk, &mut pc.bounce, snap, hl + kvn, &regs, true)?;
    ensure!(hl + kvn + nreg == total, "the snapshot's length");
    lv.len = h.len;
    lv.chunks = h.chunks;
    lv.gdn_at = gdn_at;
    lv.ple_at = ple_at;
    lv.ctx = super::PleContext(h.ctx);
    lv.next = h.next;
    lv.naccept = h.naccept;
    lv.drop = h.drop;
    lv.stash = h.stash;
    lv.key = key;
    pc.hits += 1;
    pc.hit_tokens += u64::from(m);
    pc.bytes_in += total as u64;
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    pc.reload_ms += ms;
    pc.worst_ms = pc.worst_ms.max(ms);
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// FN113: the TIERED PREFIX TREE (FLASHNEXT-PLAN §16) -- the pinned RAM tier in front of the drive,
// the cut law's pieces, the END node.
// ---------------------------------------------------------------------------------------------

/// FN113: the RAM tier's page (2 MiB: a THP page of the pinned slab, FN106d).
pub const TREE_PAGE: u64 = 2 << 20;

/// FN113: the tree's switches at a boot (`--fn-prefix-tree on`).
#[derive(Debug, Clone, Copy)]
pub struct TreeOpts {
    /// The pinned RAM tier a rank (bytes; `--fn-prefix-tree-ram-gib`).
    pub ram_bytes: u64,
    /// `--fn-prefix-tree-min-gap`.
    pub min_gap: u32,
    /// `--fn-prefix-tree-sys-min`.
    pub sys_min: u32,
    /// LR-I: `--fn-prefix-tree-cuts` (`CutRule::default()` = today's law).
    pub cuts: saber_mem::prefix_tree::CutRule,
    /// The LRFU lambda (the expert tier's).
    pub lambda: f64,
}

/// FN113: what the tree counted (`fnptree113`).
#[derive(Debug, Clone, Default)]
pub struct TreeCounts {
    pub prompts: u64,
    pub cls_turn: u64,
    pub cls_sys: u64,
    pub cls_chunk: u64,
    pub cls_partial: u64,
    pub cls_miss: u64,
    pub hits_ram: u64,
    pub hits_drive: u64,
    pub hit_tokens: u64,
    pub reload_ram_ms: f64,
    pub reload_drive_ms: f64,
    pub worst_reload_ms: f64,
    pub stores_ram: u64,
    pub store_ram_ms: f64,
    pub end_nodes: u64,
    pub promotes: u64,
    pub drive_writes: u64,
    pub drive_skips: u64,
    pub fallbacks: u64,
    pub pieces_run: u64,
    pub law_cuts: u64,
    pub bytes_in_ram: u64,
    pub bytes_out_ram: u64,
    /// FN113b: the END block in `Cmd::First` (ms): all, the m - 1 row drafter pass + stash read,
    /// the demotions' assembly (`write_through` on the step thread), the worst block.
    pub end_ms: f64,
    pub end_mtp_ms: f64,
    pub demote_ms: f64,
    pub end_worst_ms: f64,
    /// FN113b (`--fn-prefix-tree-end side`): END nodes stored off the step, the host wait for
    /// their drafter part (before the m-row pass) and the fence's residual wait (ms).
    pub side_nodes: u64,
    pub side_a_ms: f64,
    pub side_fence_ms: f64,
    pub side_fence_worst_ms: f64,
    /// FN113b: demotions run under a decode step's device span.
    pub demotes_under: u64,
    /// FN113b: the END nodes' region copies after coalescing.
    pub end_copies: u64,
    /// FN113b: a reload / demotion that found its node's copies still in flight (a miss).
    pub pending_miss: u64,
    /// FN113b: host waits for a node's copies (a ring slot reused, a demotion forced).
    pub gate_host_waits: u64,
    /// LR-D: the prompts' lengths (tokens) the tree saw to their last piece, the rows it
    /// COMPUTED for them (every piece not reloaded, the last included), the prompts with an S
    /// cut, and those whose S node key was not among the last [`SYS_SEEN`] distinct ones.
    pub prompt_tokens: u64,
    pub computed_rows: u64,
    pub sys_prompts: u64,
    pub sys_new: u64,
    /// LR-D: the last [`SYS_SEEN`] distinct S node keys (8-byte prefixes), newest last.
    pub sys_seen: std::collections::VecDeque<[u8; 8]>,
}

/// LR-D: how many distinct S node keys the tree remembers (a system prompt that changes every
/// request -- a timestamp, a reordered tool list -- is `sys_new` on every prompt).
pub const SYS_SEEN: usize = 64;

/// LR-D: one prompt's walk through the tree (reset at its first chunk, read at its last piece).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PtReq {
    /// Tokens reloaded (RAM or drive).
    pub hit: u32,
    /// Rows computed (the last piece included).
    pub rows: u32,
    /// The first computed non-last piece: (start, rows, cut kind at its end).
    pub first_miss: Option<(u32, u32, saber_mem::prefix_tree::CutKind)>,
    /// The S cut: (its position, its node key).
    pub sys: Option<(u32, Key)>,
}

impl TreeCounts {
    /// LR-D: count a prompt's S node key; true when it is new (not among the last
    /// [`SYS_SEEN`] distinct ones).
    pub fn note_sys(&mut self, key: &Key) -> bool {
        let mut k8 = [0u8; 8];
        k8.copy_from_slice(&key[..8]);
        self.sys_prompts += 1;
        if let Some(i) = self.sys_seen.iter().position(|x| *x == k8) {
            self.sys_seen.remove(i);
            self.sys_seen.push_back(k8);
            return false;
        }
        self.sys_new += 1;
        if self.sys_seen.len() == SYS_SEEN {
            self.sys_seen.pop_front();
        }
        self.sys_seen.push_back(k8);
        true
    }

    /// LR-D: the tree's `/metrics` block (`saber_fn_ptree_*`): prompt tokens vs reloaded vs
    /// computed is the live prefix-hit rate; the classes, the RAM tier's churn and `sys_new`
    /// name a miss's cause.
    #[must_use]
    pub fn render(&self, ram: &saber_mem::prefix_tree::RamCounts, used: u32, all: u32) -> String {
        let mut s = String::new();
        let mut last = String::new();
        let mut c = |n: &str, l: &str, v: u64| {
            if last != n {
                let t = if n.ends_with("_total") {
                    "counter"
                } else {
                    "gauge"
                };
                s.push_str(&format!("# TYPE saber_fn_ptree_{n} {t}\n"));
                last = n.to_string();
            }
            s.push_str(&format!("saber_fn_ptree_{n}{l} {v}\n"));
        };
        c("prompts_total", "", self.prompts);
        c("prompt_tokens_total", "", self.prompt_tokens);
        c("reloaded_tokens_total", "", self.hit_tokens);
        c("computed_rows_total", "", self.computed_rows);
        c("hits_total", "{tier=\"ram\"}", self.hits_ram);
        c("hits_total", "{tier=\"drive\"}", self.hits_drive);
        c("pieces_computed_total", "", self.pieces_run);
        c("fallbacks_total", "", self.fallbacks);
        c("pending_miss_total", "", self.pending_miss);
        for (k, v) in [
            ("turn", self.cls_turn),
            ("sys", self.cls_sys),
            ("chunk", self.cls_chunk),
            ("partial", self.cls_partial),
            ("miss", self.cls_miss),
        ] {
            c("class_total", &format!("{{class=\"{k}\"}}"), v);
        }
        c("sys_prompts_total", "", self.sys_prompts);
        c("sys_new_total", "", self.sys_new);
        c("end_nodes_total", "", self.end_nodes);
        c("ram_inserts_total", "", ram.inserts);
        c("ram_evictions_total", "", ram.evictions);
        c("ram_refused_total", "", ram.refused);
        c("ram_pages_used", "", u64::from(used));
        c("ram_pages", "", u64::from(all));
        c("drive_writes_total", "", self.drive_writes);
        c("drive_skips_total", "", self.drive_skips);
        s
    }
}

/// LR-D: the `fnptreereq` line -- one a prompt (rank 0), at its last piece.
#[must_use]
pub fn req_line(rank: u32, end: u32, r: &PtReq, sys_new: bool, ram: (u32, u32, usize)) -> String {
    let miss = r.first_miss.map_or_else(
        || "none".to_string(),
        |(a, n, k)| format!("{a}+{n} ({})", k.name()),
    );
    let sys = r.sys.map_or_else(
        || "none".to_string(),
        |(q, k)| {
            format!(
                "{q} key {} {}",
                k[..4]
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>(),
                if sys_new { "new" } else { "seen" }
            )
        },
    );
    format!(
        "fnptreereq rank{rank}: prompt {end} | reloaded {} ({:.1} %) | computed {} rows | first \
         miss {miss} | sys {sys} | ram {} of {} pages, {} nodes",
        r.hit,
        100.0 * f64::from(r.hit) / f64::from(end.max(1)),
        r.rows,
        ram.0,
        ram.1,
        ram.2
    )
}

/// FN113: the tree on one rank.
pub(super) struct Tree {
    pub law: CutLaw,
    pub ram: RamTree,
    slab: PinnedBuf,
    /// Each RAM node's header and logical region sizes (the layout's inputs).
    hdrs: HashMap<Key, (Header, Vec<u64>)>,
    pub c: TreeCounts,
    /// FN113b: the END gate -- FN72b's SDMA-written word: one fine-grained VRAM region of
    /// [`GATE_BYTES`] (its first u32 the word), a pinned value ring of [`GATE_SLOTS`] windows,
    /// the node events (recorded on the step's stream behind the word's wait).
    words: saber_hip::DevicePtr,
    vring: PinnedBuf,
    mark: u32,
    evs: Vec<saber_hip::Event>,
    slot_key: Vec<Option<Key>>,
    /// FN113b: the PENDING nodes (copies in flight): ring slot, gate enqueued.
    pending: HashMap<Key, (usize, bool)>,
    /// FN113b: nodes whose word is written on the mover but whose gate is not yet on the step's
    /// stream ([`end_gate`] enqueues it before the next command's device work).
    owed: Vec<(Key, usize, u32)>,
    /// FN113b: the victim pinned for a demotion under the next decode step ([`demote_pending`]).
    pub demote: Vec<Key>,
    /// FN113b: decode steps the pending demotion still waits (it never runs under the step right
    /// after a prompt's first token: that step's wall is the update tail, h4).
    pub demote_after: u32,
}

impl Tree {
    /// The tier's pinned slab (`host_malloc`, FN106d) of `ram_bytes` rounded down to pages.
    pub fn open(hip: &HipThread, o: TreeOpts) -> anyhow::Result<Tree> {
        let pages = (o.ram_bytes / TREE_PAGE) as u32;
        ensure!(
            pages >= 16,
            "--fn-prefix-tree-ram-gib: at least 16 pages of 2 MiB"
        );
        let slab = hip.host_malloc(u64::from(pages) * TREE_PAGE)?;
        let words =
            hip.malloc_with_flags(GATE_BYTES as u64, saber_hip::MallocFlags::FineGrained)?;
        let vring = hip.host_malloc((GATE_SLOTS * GATE_BYTES) as u64)?;
        let mut evs = Vec::with_capacity(GATE_SLOTS);
        for _ in 0..GATE_SLOTS {
            evs.push(hip.create_event(saber_hip::EventFlags::DisableTiming)?);
        }
        Ok(Tree {
            law: CutLaw::qwen4_rule(o.min_gap, o.sys_min, o.cuts),
            ram: RamTree::new(TREE_PAGE, pages, o.lambda),
            slab,
            hdrs: HashMap::new(),
            c: TreeCounts::default(),
            words,
            vring,
            mark: 0,
            evs,
            slot_key: vec![None; GATE_SLOTS],
            pending: HashMap::new(),
            owed: Vec::new(),
            demote: Vec::new(),
            demote_after: 0,
        })
    }

    /// LR-D: the `/metrics` block.
    #[must_use]
    pub fn render(&self) -> String {
        let (_, used, all) = self.ram.usage();
        self.c.render(&self.ram.counts, used, all)
    }

    /// The `fnptree113` line.
    #[must_use]
    pub fn line(&self, rank: u32) -> String {
        let c = &self.c;
        let (b, used, all) = self.ram.usage();
        let rc = self.ram.counts;
        format!(
            "fnptree113 rank{rank}: prompts {} | classes turn {} sys {} chunk {} partial {} miss {} | \
             hits ram {} drive {} ({} tokens; reload ram {:.1} ms drive {:.1} ms, worst {:.1} ms) | \
             stores ram {} ({:.1} ms, {:.1} MiB) end nodes {} promotes {} | drive writes {} skips {} \
             | fallbacks {} pieces run {} law cuts {} | ram nodes {} {:.1} MiB in {} of {} pages, \
             inserts {} evictions {} refused {} hits {} | in {:.1} MiB | end block {:.1} ms \
             (mtp {:.1} demote {:.1} worst {:.1}) | side {} (drafter part {:.1} ms, fence {:.1} ms, \
             worst {:.2}) demotes under a step {} region copies {} | pending-miss {} gate host waits {}",
            c.prompts,
            c.cls_turn,
            c.cls_sys,
            c.cls_chunk,
            c.cls_partial,
            c.cls_miss,
            c.hits_ram,
            c.hits_drive,
            c.hit_tokens,
            c.reload_ram_ms,
            c.reload_drive_ms,
            c.worst_reload_ms,
            c.stores_ram,
            c.store_ram_ms,
            c.bytes_out_ram as f64 / f64::from(1u32 << 20),
            c.end_nodes,
            c.promotes,
            c.drive_writes,
            c.drive_skips,
            c.fallbacks,
            c.pieces_run,
            c.law_cuts,
            self.ram.len(),
            b as f64 / f64::from(1u32 << 20),
            used,
            all,
            rc.inserts,
            rc.evictions,
            rc.refused,
            rc.hits,
            c.bytes_in_ram as f64 / f64::from(1u32 << 20),
            c.end_ms,
            c.end_mtp_ms,
            c.demote_ms,
            c.end_worst_ms,
            c.side_nodes,
            c.side_a_ms,
            c.side_fence_ms,
            c.side_fence_worst_ms,
            c.demotes_under,
            c.end_copies,
            c.pending_miss,
            c.gate_host_waits,
        )
    }
}

/// A node's placement in its pages: the KV block groups `(first block, blocks, page, offset)` and
/// each logical region's pieces `(offset in the region, bytes, page, offset)`; the pages used.
type Layout = (
    Vec<(u32, u32, u32, u64)>,
    Vec<Vec<(u64, u64, u32, u64)>>,
    u32,
);

fn layout(nb: u32, set: u64, sizes: &[u64]) -> anyhow::Result<Layout> {
    let mut c = Cursor::new(TREE_PAGE);
    let groups = c
        .put_blocks(nb, set)
        .context("a KV block set passes a tree page")?;
    let pieces: Vec<_> = sizes.iter().map(|&n| c.put_split(n)).collect();
    Ok((groups, pieces, c.pages()))
}

/// The copy jobs' (device address, host offset, bytes) of a logical region's `pieces` against its
/// device `segs` (each piece split where a segment ends).
fn piece_copies(
    segs: &[(u64, u64)],
    pieces: &[(u64, u64, u32, u64)],
    page_off: impl Fn(u32) -> u64,
) -> Vec<(u64, u64, u64)> {
    let mut out = Vec::new();
    for &(o, k, pg, off) in pieces {
        let mut x = o;
        let mut base = 0u64;
        for &(a, n) in segs {
            if x >= o + k {
                break;
            }
            if x < base + n {
                let take = (base + n - x).min(o + k - x);
                out.push((a + (x - base), page_off(pg) + off + (x - o), take));
                x += take;
            }
            base += n;
        }
    }
    out
}

fn kv_lbb(rk: &Rank<'_, '_>) -> anyhow::Result<Vec<usize>> {
    let kv = rk
        .pools
        .kvt
        .as_ref()
        .context("the prefix tree needs the tiered KV")?;
    let k = kv.borrow();
    Ok((0..saber_mem::kv_tier::QSA_LAYERS)
        .map(|l| k.layer_bb(l) as usize)
        .collect())
}

/// FN113: store `lv`'s state after a piece from `p0` as the RAM node `key` (D2H straight into
/// its pages), then the drive write-through when the writer is idle. False: the RAM tier refused
/// it (the caller stores to the drive the FN45c way).
pub(super) fn tree_store(
    rk: &Rank<'_, '_>,
    pc: &mut Pcache,
    lv: &Live,
    p0: u32,
    key: [u8; 32],
    parent: Option<Key>,
) -> anyhow::Result<bool> {
    let t0 = Instant::now();
    if let Some(tr) = pc.tree.as_mut() {
        if tr.ram.get(&key).is_some_and(|n| n.tokens == lv.len) {
            tr.ram.touch(&key);
            return Ok(true);
        }
    }
    let blocks = chunk_blocks(p0, lv.len);
    let nb = blocks.1 - blocks.0;
    let set = block_set_bb(rk)? as u64;
    let lregs = regions_of(rk, lv.slot, lv.len, lv.gdn_at, lv.ple_at)?;
    let sizes: Vec<u64> = lregs.iter().map(|r| r.iter().map(|x| x.1).sum()).collect();
    let (groups, pieces, need) = layout(nb, set, &sizes)?;
    // LAZY DEMOTION: the coldest victim the drive lacks is written there (one NVMe write, the
    // writer's when it is idle); the others are dropped (a later prefill, never a byte)
    let Some(plan) = pc.tree.as_ref().context("the tree")?.ram.evict_plan(need) else {
        pc.tree.as_mut().context("the tree")?.ram.counts.refused += 1;
        return Ok(false);
    };
    let mut demoted = pc.opts.no_demote;
    for k in &plan {
        if pc.lock().get(k).is_some() {
            continue;
        }
        if !demoted {
            let td = Instant::now();
            demoted = write_through(rk, pc, *k)?;
            if let Some(t) = pc.tree.as_mut() {
                t.c.demote_ms += td.elapsed().as_secs_f64() * 1e3;
            }
        }
        if !demoted {
            if let Some(t) = pc.tree.as_mut() {
                t.c.drive_skips += 1;
            }
        }
    }
    let tr = pc.tree.as_mut().context("the tree")?;
    let Some((pages, gone)) = tr.ram.alloc(need) else {
        return Ok(false);
    };
    for k in &gone {
        tr.hdrs.remove(k);
    }
    let po = |pg: u32| u64::from(pages[pg as usize]) * TREE_PAGE;
    {
        let kv = rk
            .pools
            .kvt
            .as_ref()
            .context("the prefix tree needs the tiered KV")?;
        let k = kv.borrow();
        for &(j, sn, pg, off) in &groups {
            k.dump(
                rk.hip,
                &rk.dev.stream,
                lv.slot,
                (blocks.0 + j, blocks.0 + j + sn),
                &mut tr.slab,
                (po(pg) + off) as usize,
            )?;
        }
    }
    let mut copies = Vec::new();
    for (segs, pcs) in lregs.iter().zip(&pieces) {
        copies.extend(piece_copies(segs, pcs, po));
    }
    let mut jobs = Vec::with_capacity(copies.len());
    for &(a, h, n) in &copies {
        let (d, off) = rk.pools.owner(a)?;
        jobs.push(Job::MemcpyPinned {
            dev: d,
            dev_offset: off,
            host: &tr.slab,
            host_offset: h,
            bytes: n,
            kind: MemcpyKind::DeviceToHost,
            stream: &rk.dev.stream,
        });
    }
    rk.hip.run_batch(&jobs)?;
    rk.hip.sync_stream(&rk.dev.stream)?;
    let stash = lv.stash.clone();
    let hl = hdr_len(stash.len());
    let total = hl as u64 + u64::from(nb) * set + sizes.iter().sum::<u64>();
    let h = Header {
        p0,
        len: lv.len,
        chunks: lv.chunks,
        gdn_at: lv.gdn_at as u32,
        ple_at: lv.ple_at as u32,
        ctx: lv.ctx.0,
        next: lv.next,
        naccept: lv.naccept,
        drop: lv.drop,
        blocks,
        regions: lregs.len() as u64,
        total,
        stash,
    };
    tr.ram.insert(key, parent, lv.len, pages, total);
    tr.hdrs.insert(key, (h, sizes));
    tr.c.stores_ram += 1;
    tr.c.bytes_out_ram += total;
    tr.c.store_ram_ms += t0.elapsed().as_secs_f64() * 1e3;
    Ok(true)
}

/// FN113b: an END node's RAM placement, decided BEFORE the drafter's m - 1 row pass (its pages
/// allocated, the lazy demotion of the victim done while the pass runs on the card) and copied
/// AFTER it, off the step ([`end_issue`]).
pub(super) struct EndPlan {
    key: [u8; 32],
    parent: Option<Key>,
    p0: u32,
    len: u32,
    blocks: (u32, u32),
    set: u64,
    lregs: Vec<Vec<(u64, u64)>>,
    sizes: Vec<u64>,
    groups: Vec<(u32, u32, u32, u64)>,
    pieces: Vec<Vec<(u64, u64, u32, u64)>>,
    pages: Vec<u32>,
    host_ms: f64,
}

/// FN113b: what [`end_prepare`] found.
pub(super) enum EndPrep {
    /// The RAM tier already holds the node (touched).
    Held,
    /// The node's pages are allocated.
    Plan(Box<EndPlan>),
    /// The RAM tier refused it: the caller stores on the step (FN113's form, the drive path).
    Refused,
}

/// FN113b: the END node `key` of `lv` (a piece from `p0`, `lv` as `Ready` left it) placed in the
/// RAM tier -- the eviction plan, the ONE lazy demotion (the victim's drive form assembled on the
/// host: run while the drafter's m - 1 row pass is on the card) and the pages. No device work.
pub(super) fn end_prepare(
    rk: &Rank<'_, '_>,
    pc: &mut Pcache,
    lv: &Live,
    p0: u32,
    key: [u8; 32],
) -> anyhow::Result<EndPrep> {
    let t0 = Instant::now();
    let parent = (p0 > 0).then_some(lv.key);
    end_sweep(rk, pc)?;
    if let Some(tr) = pc.tree.as_mut() {
        if tr.ram.get(&key).is_some_and(|n| n.tokens == lv.len) {
            tr.ram.touch(&key);
            return Ok(EndPrep::Held);
        }
    }
    let blocks = chunk_blocks(p0, lv.len);
    let nb = blocks.1 - blocks.0;
    let set = block_set_bb(rk)? as u64;
    let lregs = regions_of(rk, lv.slot, lv.len, lv.gdn_at, lv.ple_at)?;
    let sizes: Vec<u64> = lregs.iter().map(|r| r.iter().map(|x| x.1).sum()).collect();
    let (groups, pieces, need) = layout(nb, set, &sizes)?;
    let Some(plan) = pc.tree.as_ref().context("the tree")?.ram.evict_plan(need) else {
        return Ok(EndPrep::Refused);
    };
    // LAZY DEMOTION OFF THE STEP: the coldest victim the drive lacks is PINNED (kept in its
    // pages) and demoted later, under a decode step's device span ([`demote_pending`]); the node
    // takes other victims' pages (dropped: a later prefill, never a byte). One pending at a time.
    let lacking = plan
        .iter()
        .copied()
        .find(|k| pc.lock().get(k).is_none())
        .filter(|_| !pc.opts.no_demote);
    let tr = pc.tree.as_mut().context("the tree")?;
    let mut pinned = None;
    if let Some(k0) = lacking.filter(|_| tr.demote.is_empty()) {
        tr.ram.pin(&k0, true);
        tr.demote.push(k0);
        tr.demote_after = DEMOTE_AFTER_STEPS;
        pinned = Some(k0);
    }
    let got = match tr.ram.alloc(need) {
        Some(x) => Some(x),
        None => match pinned {
            // without the pinned victim the node does not fit: demote it now (FN113's way)
            Some(k0) => {
                tr.ram.pin(&k0, false);
                tr.demote.clear();
                let td = Instant::now();
                let ok = write_through(rk, pc, k0)?;
                let tr = pc.tree.as_mut().context("the tree")?;
                tr.c.demote_ms += td.elapsed().as_secs_f64() * 1e3;
                if !ok {
                    tr.c.drive_skips += 1;
                }
                tr.ram.alloc(need)
            }
            None => None,
        },
    };
    let Some((pages, gone)) = got else {
        return Ok(EndPrep::Refused);
    };
    let dropped = gone
        .iter()
        .filter(|k| Some(**k) != pinned && pc.lock().get(k).is_none())
        .count() as u64;
    let tr = pc.tree.as_mut().context("the tree")?;
    tr.c.drive_skips += dropped;
    for k in &gone {
        tr.hdrs.remove(k);
    }
    Ok(EndPrep::Plan(Box::new(EndPlan {
        key,
        parent,
        p0,
        len: lv.len,
        blocks,
        set,
        lregs,
        sizes,
        groups,
        pieces,
        pages,
        host_ms: t0.elapsed().as_secs_f64() * 1e3,
    })))
}

/// FN113b: one part of an END node's copies OFF THE STEP, issued on the expert mover's copy
/// stream (no new stream: FN58c) straight into the node's pinned huge pages (FN106d's slab), with
/// NO host wait: the DRAFTER part (`drafter`: the MTP layer's KV pages + its ring, ring positions
/// and compressed keys -- the only bytes the drafter's passes write) or the TARGET part (every
/// other page and region -- what the next target step rewrites). The caller has drained every
/// writer of the part's bytes; [`end_fence`] (or a host wait) lands it before its next writer.
/// Bytes are copied, never computed.
pub(super) fn end_copy(
    rk: &Rank<'_, '_>,
    pc: &mut Pcache,
    slot: u32,
    plan: &EndPlan,
    drafter: bool,
    on_step: bool,
) -> anyhow::Result<()> {
    ensure!(plan.lregs.len() >= 4, "the END node's plan");
    let stream = if on_step {
        &rk.dev.stream
    } else {
        &rk.tier.mover.queue().stream
    };
    let tr = pc.tree.as_mut().context("the tree")?;
    let po = |pg: u32| u64::from(plan.pages[pg as usize]) * TREE_PAGE;
    // the MTP layer's three logical regions (`regions`: ring, ring positions, compressed keys)
    // sit right before the PLE history, the last region
    let nr = plan.lregs.len();
    let mtp = (nr - 4)..(nr - 1);
    let kv = rk
        .pools
        .kvt
        .as_ref()
        .context("the prefix tree needs the tiered KV")?;
    let k = kv.borrow();
    for &(j, sn, pg, off) in &plan.groups {
        let at = (po(pg) + off) as usize;
        let bl = (plan.blocks.0 + j, plan.blocks.0 + j + sn);
        if drafter {
            // the MTP twin's pages follow the main tier's in a block group (`KvDev::dump`)
            if let Some(t) = k.mtp.as_deref() {
                let main = (sn as usize) * k.geom.layers as usize * k.bb as usize;
                t.dump_async(rk.hip, stream, slot, bl, &mut tr.slab, at + main)?;
            }
        } else {
            k.dump_async(rk.hip, stream, slot, bl, &mut tr.slab, at)?;
        }
    }
    drop(k);
    // the region copies, COALESCED (FN113b): a copy that continues the previous one in the same
    // allocation AND in the pages is one SDMA copy (the sequence-major pools lay a slot's layers
    // side by side)
    let mut runs: Vec<(&saber_hip::DevicePtr, u64, u64, u64)> = Vec::new();
    for (i, (segs, pcs)) in plan.lregs.iter().zip(&plan.pieces).enumerate() {
        if mtp.contains(&i) != drafter {
            continue;
        }
        for (a, h, n) in piece_copies(segs, pcs, po) {
            let (d, off) = rk.pools.owner(a)?;
            match runs.last_mut() {
                Some(r) if std::ptr::eq(r.0, d) && r.1 + r.3 == off && r.2 + r.3 == h => {
                    r.3 += n;
                }
                _ => runs.push((d, off, h, n)),
            }
        }
    }
    let jobs: Vec<Job<'_>> = runs
        .iter()
        .map(|&(d, off, h, n)| Job::MemcpyPinned {
            dev: d,
            dev_offset: off,
            host: &tr.slab,
            host_offset: h,
            bytes: n,
            kind: MemcpyKind::DeviceToHost,
            stream,
        })
        .collect();
    tr.c.end_copies += jobs.len() as u64;
    rk.hip.run_batch(&jobs)?;
    Ok(())
}

/// FN113b: a host wait for the END copies issued so far (the drafter part before its next
/// writer); counted.
pub(super) fn end_wait(rk: &Rank<'_, '_>, pc: &mut Pcache, on_step: bool) -> anyhow::Result<()> {
    let ta = Instant::now();
    if on_step {
        rk.hip.sync_stream(&rk.dev.stream)?;
    } else {
        rk.hip.sync_stream(&rk.tier.mover.queue().stream)?;
    }
    if let Some(tr) = pc.tree.as_mut() {
        tr.c.side_a_ms += ta.elapsed().as_secs_f64() * 1e3;
    }
    Ok(())
}

/// FN113b: the END node `plan` published in the RAM tier (its header from `lv` as the store
/// point leaves it; its copies issued by [`end_copy`], fenced before any reader).
pub(super) fn end_commit(
    pc: &mut Pcache,
    lv: &Live,
    plan: EndPlan,
    t0: Instant,
) -> anyhow::Result<()> {
    ensure!(lv.len == plan.len, "the END node's plan");
    let tr = pc.tree.as_mut().context("the tree")?;
    let nb = plan.blocks.1 - plan.blocks.0;
    let stash = lv.stash.clone();
    let hl = hdr_len(stash.len());
    let total = hl as u64 + u64::from(nb) * plan.set + plan.sizes.iter().sum::<u64>();
    let h = Header {
        p0: plan.p0,
        len: lv.len,
        chunks: lv.chunks,
        gdn_at: lv.gdn_at as u32,
        ple_at: lv.ple_at as u32,
        ctx: lv.ctx.0,
        next: lv.next,
        naccept: lv.naccept,
        drop: lv.drop,
        blocks: plan.blocks,
        regions: plan.lregs.len() as u64,
        total,
        stash,
    };
    tr.ram
        .insert(plan.key, plan.parent, lv.len, plan.pages, total);
    tr.hdrs.insert(plan.key, (h, plan.sizes));
    // a pending node is never evicted (its pages are still being written)
    if tr.pending.contains_key(&plan.key) {
        tr.ram.pin(&plan.key, true);
    }
    tr.c.stores_ram += 1;
    tr.c.side_nodes += 1;
    tr.c.bytes_out_ram += total;
    tr.c.store_ram_ms += plan.host_ms + t0.elapsed().as_secs_f64() * 1e3;
    Ok(())
}

/// FN113b: the decode steps a pending demotion waits before it runs under one ([`demote_pending`]).
pub const DEMOTE_AFTER_STEPS: u32 = 4;

/// FN113b: the pending demotion ([`end_prepare`]'s pinned victim) -- its drive form assembled
/// from the pages and handed to the writer -- run on the host UNDER a decode step's device span
/// (`decode_step_with`); waits for an idle writer (a later step). The victim stays in RAM,
/// unpinned, now on the drive too (a later eviction drops it with no write).
pub(super) fn demote_pending(rk: &Rank<'_, '_>, pc: &mut Pcache) -> anyhow::Result<()> {
    let Some(k) = pc.tree.as_ref().and_then(|t| t.demote.first().copied()) else {
        return Ok(());
    };
    if let Some(t) = pc.tree.as_mut().filter(|t| t.demote_after > 0) {
        t.demote_after -= 1;
        return Ok(());
    }
    if pc.inflight.as_ref().is_some_and(|h| !h.is_finished()) && !pc.opts.sync {
        return Ok(());
    }
    if let Some(tr) = pc.tree.as_mut() {
        if !node_ready(rk, tr, &k)? {
            tr.c.pending_miss += 1;
            return Ok(());
        }
    }
    let td = Instant::now();
    let ok = write_through(rk, pc, k)?;
    let tr = pc.tree.as_mut().context("the tree")?;
    tr.demote.retain(|x| *x != k);
    tr.ram.pin(&k, false);
    tr.c.demote_ms += td.elapsed().as_secs_f64() * 1e3;
    tr.c.demotes_under += 1;
    if !ok {
        tr.c.drive_skips += 1;
    }
    Ok(())
}

/// FN113b: the END gate's VRAM word region (the first u32 is the word; 20 KiB = one SDMA copy,
/// FN72b: a smaller pinned copy is a blit on the stream's queue).
pub const GATE_BYTES: usize = 20 << 10;
/// FN113b: the gate's value ring windows = the most END nodes pending at once (a ninth waits
/// for the oldest on the host, counted).
pub const GATE_SLOTS: usize = 8;

/// FN113b: zero the gate's word on `stream` (boot, before any node).
pub(super) fn gate_init(rk: &Rank<'_, '_>, pc: &mut Pcache) -> anyhow::Result<()> {
    if let Some(tr) = pc.tree.as_mut() {
        rk.hip.memset(&tr.words, 0, &rk.dev.stream)?;
        rk.hip.sync_stream(&rk.dev.stream)?;
    }
    Ok(())
}

/// FN113b: the END node `key`'s word, written by the SDMA on the MOVER's stream after the copies
/// issued so far (the target part): the value `v` lands only after them. The gate itself goes on
/// the step's stream later ([`end_gate`]). Never a host wait, except a ring slot whose previous
/// node is still pending (counted).
pub(super) fn end_mark(rk: &Rank<'_, '_>, pc: &mut Pcache, key: Key) -> anyhow::Result<()> {
    let tr = pc.tree.as_mut().context("the tree")?;
    tr.mark = tr.mark.wrapping_add(1).max(1);
    let v = tr.mark;
    let i = v as usize % GATE_SLOTS;
    if let Some(old) = tr.slot_key[i].take() {
        if tr.pending.remove(&old).is_some() {
            tr.c.gate_host_waits += 1;
            rk.hip.sync_event(&tr.evs[i])?;
            tr.ram.pin(&old, false);
        }
    }
    let at = i * GATE_BYTES;
    tr.vring.as_mut_slice()[at..at + 4].copy_from_slice(&v.to_le_bytes());
    rk.hip.run_batch(&[Job::MemcpyPinned {
        dev: &tr.words,
        dev_offset: 0,
        host: &tr.vring,
        host_offset: at as u64,
        bytes: GATE_BYTES as u64,
        kind: MemcpyKind::HostToDevice,
        stream: &rk.tier.mover.queue().stream,
    }])?;
    tr.slot_key[i] = Some(key);
    tr.pending.insert(key, (i, false));
    tr.owed.push((key, i, v));
    Ok(())
}

/// FN113b: every owed gate onto the step's stream -- `hipStreamWaitValue32(word >= v)` (the CP
/// waits; the host never does) and the node's event behind it. Run at the end of `Cmd::First`
/// and before any command's device work; every later writer of the node's source bytes is
/// ordered behind its copies, and the event completing = the node's bytes landed.
pub(super) fn end_gate(rk: &Rank<'_, '_>, pc: &mut Pcache) -> anyhow::Result<()> {
    let Some(tr) = pc.tree.as_mut() else {
        return Ok(());
    };
    if tr.owed.is_empty() {
        return Ok(());
    }
    let t0 = Instant::now();
    let word = tr.words.addr();
    for (key, i, v) in std::mem::take(&mut tr.owed) {
        rk.hip.run_batch(&[
            Job::StreamWaitValueGte {
                stream: &rk.dev.stream,
                word,
                value: v,
                mask: u32::MAX,
            },
            Job::RecordEvent {
                event: &tr.evs[i],
                stream: &rk.dev.stream,
            },
        ])?;
        if let Some(p) = tr.pending.get_mut(&key) {
            p.1 = true;
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    tr.c.side_fence_ms += ms;
    tr.c.side_fence_worst_ms = tr.c.side_fence_worst_ms.max(ms);
    Ok(())
}

/// FN113b: whether the node `key`'s bytes have landed (not pending, or its gated event done; the
/// node is unpinned then). Never waits.
fn node_ready(rk: &Rank<'_, '_>, tr: &mut Tree, key: &Key) -> anyhow::Result<bool> {
    let Some(&(i, gated)) = tr.pending.get(key) else {
        return Ok(true);
    };
    if !gated || !rk.hip.query_event(&tr.evs[i])? {
        return Ok(false);
    }
    tr.pending.remove(key);
    tr.slot_key[i] = None;
    tr.ram.pin(key, false);
    Ok(true)
}

/// FN113b: the node `key`'s bytes landed, waiting on the host if they have not (a demotion that
/// must read them now; counted).
fn node_wait(rk: &Rank<'_, '_>, tr: &mut Tree, key: &Key) -> anyhow::Result<()> {
    if node_ready(rk, tr, key)? {
        return Ok(());
    }
    let (i, gated) = tr.pending[key];
    tr.c.gate_host_waits += 1;
    if gated {
        rk.hip.sync_event(&tr.evs[i])?;
    } else {
        rk.hip.sync_stream(&rk.tier.mover.queue().stream)?;
    }
    tr.pending.remove(key);
    tr.slot_key[i] = None;
    tr.ram.pin(key, false);
    Ok(())
}

/// FN113b: the pending nodes whose events completed are released (unpinned: evictable again).
pub(super) fn end_sweep(rk: &Rank<'_, '_>, pc: &mut Pcache) -> anyhow::Result<()> {
    let Some(tr) = pc.tree.as_mut() else {
        return Ok(());
    };
    let keys: Vec<Key> = tr.pending.keys().copied().collect();
    for k in keys {
        node_ready(rk, tr, &k)?;
    }
    Ok(())
}

/// FN113: the DEMOTION of the RAM node `key` to the drive (FN45c3's file format, assembled from
/// the pages into the pageable snapshot buffer) when the writer is idle and the drive lacks it;
/// true = the drive holds it or a writer took it.
fn write_through(rk: &Rank<'_, '_>, pc: &mut Pcache, key: [u8; 32]) -> anyhow::Result<bool> {
    if pc.lock().get(&key).is_some() {
        return Ok(true);
    }
    // FN113b: an END node's copies landed before its pages are read
    if let Some(tr) = pc.tree.as_mut() {
        node_wait(rk, tr, &key)?;
    }
    if pc.inflight.as_ref().is_some_and(|h| !h.is_finished()) && !pc.opts.sync {
        return Ok(false);
    }
    pc.wait();
    let lbb = kv_lbb(rk)?;
    let mut snap = pc.snap.take().context("the snapshot buffer")?;
    let tr = pc.tree.as_ref().context("the tree")?;
    let (Some((h, sizes)), Some(node)) = (tr.hdrs.get(&key), tr.ram.get(&key)) else {
        pc.snap = Some(snap);
        return Ok(false);
    };
    let total = h.total as usize;
    if total > snap.len() {
        pc.snap = Some(snap);
        return Ok(false);
    }
    let nb = h.blocks.1 - h.blocks.0;
    let set: u64 = lbb.iter().sum::<usize>() as u64;
    let hl = hdr_len(h.stash.len());
    let (groups, pieces, _) = layout(nb, set, sizes)?;
    let po = |pg: u32| u64::from(node.pages[pg as usize]) * TREE_PAGE;
    let slab = tr.slab.as_slice();
    snap[..hl].copy_from_slice(&h.encode());
    for &(j, sn, pg, off) in &groups {
        let g0 = (po(pg) + off) as usize;
        for l in 0..lbb.len() {
            let (d, b, n) = snap_span(&lbb, l, hl, nb as usize, j as usize, sn as usize);
            snap[d..d + n].copy_from_slice(&slab[g0 + b..g0 + b + n]);
        }
    }
    let mut o = hl + nb as usize * set as usize;
    for pcs in &pieces {
        for &(_, k, pg, off) in pcs {
            let a = (po(pg) + off) as usize;
            snap[o..o + k as usize].copy_from_slice(&slab[a..a + k as usize]);
            o += k as usize;
        }
    }
    ensure!(o == total, "the node's drive form");
    let tokens = h.len;
    let w = spawn_writer(pc, key, tokens, snap, total, true)?;
    if w {
        if let Some(t) = pc.tree.as_mut() {
            t.c.drive_writes += 1;
        }
    }
    Ok(w)
}

/// FN113: whether BOTH ranks hold the piece `n` ids long after `lv` at `key` (RAM or drive; one
/// exchange).
pub(super) fn probe_piece(
    rk: &mut Rank<'_, '_>,
    pc: &mut Pcache,
    lv: &Live,
    n: u32,
    key: &[u8; 32],
) -> anyhow::Result<bool> {
    let t = lv.len + n;
    let mut ram = pc
        .tree
        .as_ref()
        .is_some_and(|tr| tr.ram.get(key).is_some_and(|x| x.tokens == t));
    // FN113b: a node whose copies are still in flight is a miss here (counted)
    if ram {
        if let Some(tr) = pc.tree.as_mut() {
            if !node_ready(rk, tr, key)? {
                tr.c.pending_miss += 1;
                ram = false;
            }
        }
    }
    let mine = ram || pc.lock().get(key).is_some_and(|e| e.tokens == t);
    let peer = rk
        .peer
        .exchange()
        .xchg_bytes("fn.ptree.p", &[u8::from(mine)])?;
    Ok(mine && peer.first() == Some(&1))
}

/// FN113: reload the piece `ids` after `lv` from this rank's lowest tier holding it (RAM: H2D
/// straight from the node's pages; drive: FN45c4's read + bounce, then PROMOTED to RAM by one D2H
/// of the reloaded state). Both ranks agree by one exchange before any byte moves; false = the
/// caller prefills.
pub(super) fn tree_reload(
    rk: &mut Rank<'_, '_>,
    pc: &mut Pcache,
    lv: &mut Live,
    ids: &[u32],
    key: [u8; 32],
    fn_chunk: u32,
) -> anyhow::Result<bool> {
    let t0 = Instant::now();
    let m = ids.len() as u32;
    // FN113b: a node whose copies are still in flight is not read (the drive, else a miss)
    let landed = match pc.tree.as_mut() {
        Some(tr) => node_ready(rk, tr, &key)?,
        None => true,
    };
    let ram = pc
        .tree
        .as_ref()
        .filter(|_| landed)
        .and_then(|tr| tr.hdrs.get(&key))
        .filter(|(h, _)| h.p0 == lv.len && h.len == lv.len + m)
        .cloned();
    let drive = if ram.is_none() {
        read_drive(pc, lv, ids, &key).1
    } else {
        None
    };
    let ok = ram.is_some() || drive.is_some();
    let peer = rk
        .peer
        .exchange()
        .xchg_bytes("fn.ptree.r", &[u8::from(ok)])?;
    if !(ok && peer.first() == Some(&1)) {
        if let Some(t) = pc.tree.as_mut() {
            t.c.fallbacks += 1;
        }
        return Ok(false);
    }
    let parent = (lv.len > 0).then_some(lv.key);
    if let Some((h, sizes)) = ram {
        open_piece(rk, lv.slot, h.p0, m, fn_chunk)?;
        let set = block_set_bb(rk)? as u64;
        let (gdn_at, ple_at) = (h.gdn_at as usize, h.ple_at as usize);
        let lregs = regions_of(rk, lv.slot, h.len, gdn_at, ple_at)?;
        let now: Vec<u64> = lregs.iter().map(|r| r.iter().map(|x| x.1).sum()).collect();
        ensure!(now == sizes, "the node's regions");
        let tr = pc.tree.as_mut().context("the tree")?;
        let pages = tr.ram.get(&key).context("the node")?.pages.clone();
        let po = |pg: u32| u64::from(pages[pg as usize]) * TREE_PAGE;
        let nb = h.blocks.1 - h.blocks.0;
        let (groups, pieces, need) = layout(nb, set, &sizes)?;
        ensure!(need as usize == pages.len(), "the node's pages");
        {
            let kv = rk
                .pools
                .kvt
                .as_ref()
                .context("the prefix tree needs the tiered KV")?;
            let mut k = kv.borrow_mut();
            for &(j, sn, pg, off) in &groups {
                k.load(
                    rk.hip,
                    &rk.dev.stream,
                    lv.slot,
                    (h.blocks.0 + j, h.blocks.0 + j + sn),
                    &tr.slab,
                    (po(pg) + off) as usize,
                )?;
            }
        }
        let mut copies = Vec::new();
        for (segs, pcs) in lregs.iter().zip(&pieces) {
            copies.extend(piece_copies(segs, pcs, po));
        }
        {
            let mut jobs = Vec::with_capacity(copies.len());
            for &(a, hoff, n) in &copies {
                let (d, off) = rk.pools.owner(a)?;
                jobs.push(Job::MemcpyPinned {
                    dev: d,
                    dev_offset: off,
                    host: &tr.slab,
                    host_offset: hoff,
                    bytes: n,
                    kind: MemcpyKind::HostToDevice,
                    stream: &rk.dev.stream,
                });
            }
            rk.hip.run_batch(&jobs)?;
            rk.hip.sync_stream(&rk.dev.stream)?;
        }
        tr.ram.touch(&key);
        tr.ram.counts.hits += 1;
        lv.len = h.len;
        lv.chunks = h.chunks;
        lv.gdn_at = gdn_at;
        lv.ple_at = ple_at;
        lv.ctx = super::PleContext(h.ctx);
        lv.next = h.next;
        lv.naccept = h.naccept;
        lv.drop = h.drop;
        lv.stash = h.stash.clone();
        lv.key = key;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        tr.c.hits_ram += 1;
        tr.c.hit_tokens += u64::from(m);
        tr.c.reload_ram_ms += ms;
        tr.c.worst_reload_ms = tr.c.worst_reload_ms.max(ms);
        tr.c.bytes_in_ram += h.total;
        return Ok(true);
    }
    let (h, buf) = drive.context("the drive entry")?;
    let p0 = h.p0;
    apply_drive(rk, pc, lv, ids, key, fn_chunk, h, buf, t0)?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    {
        let tr = pc.tree.as_mut().context("the tree")?;
        tr.c.hits_drive += 1;
        tr.c.hit_tokens += u64::from(m);
        tr.c.reload_drive_ms += ms;
        tr.c.worst_reload_ms = tr.c.worst_reload_ms.max(ms);
    }
    if tree_store(rk, pc, lv, p0, key, parent)? {
        if let Some(t) = pc.tree.as_mut() {
            t.c.promotes += 1;
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    #[test]
    fn lrd_a_system_prompt_that_changes_every_request_is_new_every_time() {
        use super::{TreeCounts, SYS_SEEN};
        let mut c = TreeCounts::default();
        let k = |i: u8| {
            let mut x = [0u8; 32];
            x[0] = i;
            x
        };
        // one client's stable system prompt: new once, then seen
        assert!(c.note_sys(&k(1)));
        for _ in 0..10 {
            assert!(!c.note_sys(&k(1)));
        }
        // a per-request system prompt (a timestamp): new every prompt; the stable one stays
        // remembered while it keeps recurring (recency order, not insertion order)
        for i in 0..200u32 {
            assert!(c.note_sys(&k(2 + (i % 250) as u8)));
            if i % 10 == 0 {
                assert!(!c.note_sys(&k(1)));
            }
        }
        assert_eq!(c.sys_seen.len(), SYS_SEEN);
        assert_eq!(c.sys_prompts, 1 + 10 + 200 + 20);
        assert_eq!(c.sys_new, 1 + 200);
    }

    #[test]
    fn lrd_the_tree_metrics_and_the_request_line_read_the_counts() {
        use super::{req_line, PtReq, TreeCounts};
        use saber_mem::prefix_tree::{CutKind, RamCounts};
        let c = TreeCounts {
            prompts: 3,
            prompt_tokens: 60_000,
            hit_tokens: 20_000,
            computed_rows: 40_000,
            hits_ram: 4,
            cls_miss: 2,
            cls_partial: 1,
            sys_prompts: 3,
            sys_new: 2,
            ..TreeCounts::default()
        };
        let ram = RamCounts {
            inserts: 9,
            evictions: 5,
            refused: 0,
            hits: 4,
        };
        let m = c.render(&ram, 700, 768);
        for want in [
            "# TYPE saber_fn_ptree_prompt_tokens_total counter\n",
            "saber_fn_ptree_prompt_tokens_total 60000\n",
            "saber_fn_ptree_reloaded_tokens_total 20000\n",
            "saber_fn_ptree_computed_rows_total 40000\n",
            "saber_fn_ptree_hits_total{tier=\"ram\"} 4\n",
            "saber_fn_ptree_class_total{class=\"miss\"} 2\n",
            "saber_fn_ptree_sys_new_total 2\n",
            "saber_fn_ptree_ram_evictions_total 5\n",
            "# TYPE saber_fn_ptree_ram_pages_used gauge\n",
            "saber_fn_ptree_ram_pages 768\n",
        ] {
            assert!(m.contains(want), "{want} in\n{m}");
        }
        // one TYPE line a family (the labelled families share it)
        assert_eq!(m.matches("# TYPE saber_fn_ptree_class_total").count(), 1);
        // the 7,801 shape: S at 7,801 computed (new key), then the rest
        let mut k = [0u8; 32];
        k[..4].copy_from_slice(&[0xde, 0xad, 0xbe, 0xef]);
        let r = PtReq {
            hit: 0,
            rows: 14_000,
            first_miss: Some((0, 7801, CutKind::Sys)),
            sys: Some((7801, k)),
        };
        assert_eq!(
            req_line(0, 14_000, &r, true, (768, 768, 22)),
            "fnptreereq rank0: prompt 14000 | reloaded 0 (0.0 %) | computed 14000 rows | first \
             miss 0+7801 (sys) | sys 7801 key deadbeef new | ram 768 of 768 pages, 22 nodes"
        );
        let h = PtReq {
            hit: 30_000,
            rows: 2_112,
            first_miss: None,
            sys: Some((7801, k)),
        };
        assert!(req_line(0, 32_112, &h, false, (1, 2, 3))
            .contains("reloaded 30000 (93.4 %) | computed 2112 rows | first miss none | sys 7801 key deadbeef seen"));
    }

    #[test]
    fn fn66_the_warm_cut_stores_a_head_up_to_its_last_token() {
        const END: u32 = 248_046;
        // a head rendered up to the next turn: `... <|im_end|> \n` -> stored before the "\n"
        assert_eq!(super::warm_cut(END, &[1, 2, END, 198]), Some(3));
        assert_eq!(super::warm_cut(END, &[END, 198]), Some(1));
        // a served chat prompt ends in the generation prompt: no warm cut (the CONTROL)
        let chat = [1, END, 198, 248_045, 77_091, 198, 248_068, 198];
        assert_eq!(super::warm_cut(END, &chat), None);
        assert_eq!(
            super::warm_cut(END, &[198]),
            None,
            "one token: nothing to store"
        );
        assert_eq!(super::warm_cut(END, &[1, 2, 198, END]), None);
    }

    use super::*;

    #[test]
    fn the_key_chains_the_cuts_and_the_ids() {
        let ids: Vec<u32> = (0..64).collect();
        let k1 = next_key(b"m", &ids[..32]);
        let k2 = next_key(&k1, &ids[32..]);
        // the same ids cut elsewhere, or one chunk, never share a key
        assert_ne!(next_key(b"m", &ids), k2);
        assert_ne!(next_key(&next_key(b"m", &ids[..16]), &ids[16..]), k2);
        // deterministic, model-scoped
        assert_eq!(next_key(&next_key(b"m", &ids[..32]), &ids[32..]), k2);
        assert_ne!(next_key(b"n", &ids[..32]), k1);
    }

    #[test]
    fn the_header_round_trips_and_refuses_foreign_bytes() {
        let h = Header {
            p0: 8192,
            len: 16_384,
            chunks: 2,
            gdn_at: 0,
            ple_at: 0,
            ctx: [7, 9],
            next: 3,
            naccept: 1,
            drop: 0,
            blocks: chunk_blocks(8192, 16_384),
            regions: 99,
            total: 123_456,
            stash: vec![5; 20_480],
        };
        let b = h.encode();
        assert_eq!(b.len() % 4096, 0);
        assert_eq!(Header::decode(&b).unwrap(), h);
        let mut bad = b.clone();
        bad[0] ^= 1;
        assert!(Header::decode(&bad).is_err());
        assert!(Header::decode(&b[..64]).is_err());
    }

    #[test]
    fn the_conversation_cut_is_before_the_last_im_start_with_a_short_tail() {
        const S: u32 = 9;
        let mut ids: Vec<u32> = (100..700).collect();
        ids[10] = S; // an earlier turn
        ids[596] = S; // the generation prompt: <|im_start|> assistant \n <think> \n
        assert_eq!(turn_cut(S, 512, 1, 0, &ids), Some(596));
        // too short a prefix to store
        assert_eq!(turn_cut(S, 1024, 1, 0, &ids), None);
        // ... unless the chunk starts later in the sequence
        assert_eq!(turn_cut(S, 1024, 1, 600, &ids), Some(596));
        // a tail past TURN_TAIL_MAX is not a chat turn's
        let mut long = ids.clone();
        long.extend(std::iter::repeat_n(1, TURN_TAIL_MAX));
        assert_eq!(turn_cut(S, 512, 1, 0, &long), None);
        // never an empty first part, never without the token
        assert_eq!(turn_cut(S, 0, 1, 0, &[S, 1, 2]), None);
        assert_eq!(turn_cut(S, 0, 1, 0, &[1, 2, 3]), None);
        // FN45c4: 64-ALIGNED -- the cut lands on a multiple of 64 of the SEQUENCE position
        assert_eq!(turn_cut(S, 512, 64, 0, &ids), Some(576));
        assert_eq!(turn_cut(S, 512, 64, 600, &ids), Some(1152 - 600));
        assert_eq!(turn_cut(S, 512, 64, 1150, &ids), Some(1728 - 1150));
        // the aligned cut leaves nothing before it in this chunk / under the minimum: no cut
        assert_eq!(turn_cut(S, 0, 64, 1, &[1, 2, S, 3]), None);
        let short: Vec<u32> = (0..530).map(|i| if i == 520 { S } else { 1 }).collect();
        assert_eq!(turn_cut(S, 512, 64, 0, &short), Some(512));
        let under: Vec<u32> = (0..510).map(|i| if i == 500 { S } else { 1 }).collect();
        assert_eq!(
            turn_cut(S, 512, 64, 0, &under),
            None,
            "448 < the 512 minimum"
        );
    }

    #[test]
    fn the_longest_held_prefix_is_confirmed_by_its_chained_key() {
        let ids: Vec<u32> = (0..100).collect();
        let prev = next_key(b"m", &[7, 8]);
        // stored: the prefixes of 40 and 90 tokens past len 10 (= 30 and 80 ids of this chunk)
        let k30 = next_key(&prev, &ids[..30]);
        let k80 = next_key(&prev, &ids[..80]);
        let held = |k: &[u8; 32], t: u32| (t == 40 && *k == k30) || (t == 90 && *k == k80);
        let got = held_lengths(&prev, 10, &ids, &[], &[120, 90, 60, 40, 5], held);
        assert_eq!(got, vec![(90, k80), (40, k30)]);
        // another conversation with the same lengths does not hit
        let other: Vec<u32> = (1..101).collect();
        assert!(held_lengths(&prev, 10, &other, &[], &[90, 40], held).is_empty());
        // a LAST chunk's probe sees one id fewer: a prefix of the whole chunk is not offered
        assert!(held_lengths(&prev, 10, &ids[..79], &[], &[90], held).is_empty());
    }

    use saber_sched::prefix_key::ImageKey;

    const PAD: u32 = saber_vision_input::IMAGE_PAD_ID;

    /// An agent turn's prompt: `pre` text ids, then per image `(text, run)`: a vision-start id,
    /// the image's pad run, a vision-end id, `text` ids. Returns the ids and the image runs at
    /// `hashes`.
    fn agent_prompt(
        pre: u32,
        runs: &[u32],
        text: u32,
        hashes: &[u64],
    ) -> (Vec<u32>, Vec<ImageKey>) {
        let mut ids: Vec<u32> = (0..pre).map(|i| 1000 + i).collect();
        let mut ims = Vec::new();
        for (k, (&r, &h)) in runs.iter().zip(hashes).enumerate() {
            ids.push(248_053);
            ims.push(ImageKey {
                start: ids.len() as u32,
                tokens: r,
                hash: h,
            });
            ids.extend(std::iter::repeat_n(PAD, r as usize));
            ids.push(248_054);
            ids.extend((0..text).map(|i| 5000 + 100 * k as u32 + i));
        }
        (ids, ims)
    }

    /// The chained piece keys of `ids` cut at `cuts` (piece ends).
    fn chain(ids: &[u32], cuts: &[u32], ims: &[ImageKey]) -> Vec<[u8; 32]> {
        let mut prev: Vec<u8> = b"model".to_vec();
        let mut a = 0u32;
        let mut out = Vec::new();
        for &b in cuts {
            let k = piece_key(&prev, a, &ids[a as usize..b as usize], ims);
            out.push(k);
            prev = k.to_vec();
            a = b;
        }
        out
    }

    /// FN125: a TEXT prompt's keys are the pre-FN125 keys byte for byte -- `piece_key` with no
    /// image (or images that overlap no piece) IS `next_key`, whose bytes are pinned here
    /// (sha256(sha256("m") || ids 1..=40 u32 LE), computed off the code).
    #[test]
    fn fn125_a_text_prompts_keys_are_unchanged_byte_for_byte() {
        let ids: Vec<u32> = (1..=40).collect();
        let k = next_key(b"m", &ids);
        assert_eq!(
            hex::encode(k),
            "224a70c452401e2bb9104cc3d58b1f18c13514d6341ca50bee7558d3eb11ac6c"
        );
        assert_eq!(piece_key(b"m", 0, &ids, &[]), k);
        // images of the same sequence that the piece does not touch leave its key alone
        let far = [ImageKey {
            start: 40,
            tokens: 64,
            hash: 0xAAAA,
        }];
        assert_eq!(piece_key(b"m", 0, &ids, &far), k);
        let before = [ImageKey {
            start: 0,
            tokens: 7,
            hash: 0xAAAA,
        }];
        assert_eq!(
            piece_key(b"m", 7, &ids[7..], &before),
            next_key(b"m", &ids[7..])
        );
        // a whole text conversation: every piece and every held length the text law's
        let (t, _) = agent_prompt(3000, &[], 0, &[]);
        let cuts = [512, 2048, 2900, 3000];
        let mut prev: Vec<u8> = b"model".to_vec();
        let mut a = 0u32;
        for (i, k) in chain(&t, &cuts, &[]).into_iter().enumerate() {
            assert_eq!(k, next_key(&prev, &t[a as usize..cuts[i] as usize]));
            prev = k.to_vec();
            a = cuts[i];
        }
        let prev = next_key(b"m", &t[..10]);
        let h = |_: &[u8; 32], _: u32| true;
        assert_eq!(
            held_lengths(&prev, 10, &t[10..500], &[], &[400, 100], h),
            vec![
                (400, next_key(&prev, &t[10..400])),
                (100, next_key(&prev, &t[10..100]))
            ]
        );
    }

    /// FN125: two prompts with the same ids whose images differ in content share the pieces
    /// BEFORE the first differing image and NO piece from the one that overlaps it on (the chain
    /// carries it); the same images give the same key in every piece (the hit); an image run
    /// straddling a cut is folded into both pieces; the control -- the key over the ids alone --
    /// collides the two prompts in every piece.
    #[test]
    fn fn125_different_images_differ_from_the_image_on_and_the_same_images_hit() {
        let runs = [900, 1200, 640, 1500, 700];
        let a_h = [11, 22, 33, 44, 55];
        let mut b_h = a_h;
        b_h[2] = 0x3333; // the third image differs (one pixel would do: the plane's hash)
        let (ids, ia) = agent_prompt(2000, &runs, 300, &a_h);
        let (ids_b, ib) = agent_prompt(2000, &runs, 300, &b_h);
        assert_eq!(
            ids, ids_b,
            "the same token ids: the pads do not tell the images apart"
        );
        let n = ids.len() as u32;
        // cuts: a system piece, then chunk ends at 2,048 (one cut lands INSIDE image 0's run)
        let mut cuts: Vec<u32> = vec![512];
        cuts.extend((1..).map(|k| k * 2048).take_while(|&c| c < n));
        cuts.push(n);
        assert!(
            ia.iter().any(|im| cuts
                .iter()
                .any(|&c| c > im.start && c < im.start + im.tokens)),
            "some image straddles a cut"
        );
        let ka = chain(&ids, &cuts, &ia);
        let kb = chain(&ids, &cuts, &ib);
        assert_eq!(
            ka,
            chain(&ids, &cuts, &ia),
            "the same images: every piece hits"
        );
        let third = ia[2].start;
        let mut a = 0;
        for (i, &b) in cuts.iter().enumerate() {
            if b <= third {
                assert_eq!(
                    ka[i], kb[i],
                    "piece {i} ({a}..{b}) ends before the third image"
                );
            } else {
                assert_ne!(ka[i], kb[i], "piece {i} ({a}..{b}) holds or follows it");
            }
            a = b;
        }
        // a piece an image run straddles carries the image in both pieces
        let s = ia[0];
        let c = *cuts
            .iter()
            .find(|&&c| c > s.start && c < s.start + s.tokens)
            .unwrap_or(&0);
        if c > 0 {
            let other0 = [ImageKey { hash: 0x9999, ..s }];
            let lo = *cuts.iter().rev().find(|&&x| x < c).unwrap_or(&0);
            let hi = *cuts.iter().find(|&&x| x > c).unwrap();
            for (x, y) in [(lo, c), (c, hi)] {
                let p = b"p".as_slice();
                assert_ne!(
                    piece_key(p, x, &ids[x as usize..y as usize], &[s]),
                    piece_key(p, x, &ids[x as usize..y as usize], &other0),
                    "{x}..{y} holds part of image 0"
                );
            }
        }
        // the control: ids alone collide the two prompts everywhere
        assert_eq!(chain(&ids, &cuts, &[]), chain(&ids_b, &cuts, &[]));
        assert_ne!(chain(&ids, &cuts, &[]), ka);
        // the probe's held lengths fold the same law: B's stored prefix is not A's hit
        let prev = b"model".as_slice();
        let ka_t = piece_key(prev, 0, &ids[..4000], &ia);
        let held = |k: &[u8; 32], t: u32| t == 4000 && *k == ka_t;
        assert_eq!(
            held_lengths(prev, 0, &ids, &ia, &[4000], held),
            vec![(4000, ka_t)]
        );
        let kb_t = piece_key(prev, 0, &ids[..4000], &ib);
        assert_eq!(ka_t, kb_t, "4,000 ends before the third image: shared");
        let ka_8 = piece_key(prev, 0, &ids[..8000], &ia);
        let held8 = |k: &[u8; 32], t: u32| t == 8000 && *k == ka_8;
        assert!(held_lengths(prev, 0, &ids, &ib, &[8000], held8).is_empty());
        assert_eq!(held_lengths(prev, 0, &ids, &ia, &[8000], held8).len(), 1);
    }

    /// FN125: the agent shape -- turn 2 = turn 1's prompt + the reply + a new user turn, the five
    /// images all in turn 1's part. Every piece turn 1 stored is turn 2's key (a hit), and turn
    /// 2's computed suffix overlaps no image run: the deferred tower is not due (the full hit
    /// skips the tower); a turn whose new suffix carries an image needs it.
    #[test]
    fn fn125_the_agent_turn_reloads_its_images_and_the_tower_is_not_due() {
        let runs = [900, 1200, 640, 1500, 700];
        let hs = [1, 2, 3, 4, 5];
        let (t1, ims) = agent_prompt(3000, &runs, 400, &hs);
        let n1 = t1.len() as u32;
        let mut t2 = t1.clone();
        t2.extend(7000..7300); // the reply + the next user turn
        let n2 = t2.len() as u32;
        let cuts1 = [512, 2048, 4096, 6144, n1];
        let k1 = chain(&t1, &cuts1, &ims);
        let mut cuts2 = cuts1.to_vec();
        cuts2.push(n2);
        let k2 = chain(&t2, &cuts2, &ims);
        assert_eq!(
            k1[..],
            k2[..cuts1.len()],
            "turn 2's prefix pieces are turn 1's nodes"
        );
        let grids: Vec<[u32; 3]> = runs
            .iter()
            .map(|&r| {
                // r tokens = (h/2)(w/2): pick w = 2 x 20 merged columns where it divides
                let w = if r % 20 == 0 { 20 } else { 10 };
                [1, 2 * (r / w), 2 * w]
            })
            .collect();
        let mut f = super::super::super::vision::ImgFacts::keyed(&t2, &grids, 0, &hs, 9).unwrap();
        assert_eq!(f.image_keys(), ims);
        let s = super::super::super::vision::ImgSeq::new(&f);
        assert!(!s.tower_due(n1, n2), "the computed suffix holds no image");
        assert!(
            s.tower_due(2048, 4096),
            "a cold prefill over an image needs the tower"
        );
        assert!(
            !s.tower_due(0, 2048),
            "text before the first image does not"
        );
        assert!(!s.ran(4).tower_due(0, n2), "once run, never again");
        // the eager form (the tower ran at encode, no ticket) is never due
        f.defer = 0;
        f.epoch = 3;
        assert!(!super::super::super::vision::ImgSeq::new(&f).tower_due(0, n2));
        // a turn whose new suffix carries a sixth image: the computed suffix needs the tower
        let (t3, ims3) = agent_prompt(
            3000,
            &[900, 1200, 640, 1500, 700, 400],
            400,
            &[1, 2, 3, 4, 5, 6],
        );
        let mut g3 = grids.clone();
        g3.push([1, 40, 40]);
        let f3 = super::super::super::vision::ImgFacts::keyed(&t3, &g3, 0, &[1, 2, 3, 4, 5, 6], 10)
            .unwrap();
        let s3 = super::super::super::vision::ImgSeq::new(&f3);
        assert_eq!(f3.image_keys(), ims3);
        assert!(s3.tower_due(ims3[5].start - 10, t3.len() as u32));
        assert!(!s3.tower_due(ims3[4].start + ims3[4].tokens, ims3[5].start));
    }

    /// FN68d: the snapshot's KV spans with the MTP layer's twin at bf16 beside 12 fp8 layers --
    /// every bounced group's layer spans tile the snapshot exactly once (no gap, no overlap) and
    /// the bounce exactly once, in `KvDev::dump`'s layer-major order; one dtype = FN68b's law.
    #[test]
    fn fn68d_the_snapshot_spans_tile_a_mixed_dtype_block_set() {
        for lbb in [
            [8192usize; 13].to_vec(),
            [vec![8192usize; 12], vec![16_384]].concat(),
            [16_384usize; 13].to_vec(),
        ] {
            let set: usize = lbb.iter().sum();
            for (nb, step) in [(1usize, 1usize), (7, 3), (64, 64), (100, 9)] {
                let at = 4096;
                let mut cover = vec![0u8; nb * set];
                let mut j = 0;
                while j < nb {
                    let sn = step.min(nb - j);
                    let mut bounce = vec![0u8; sn * set];
                    for l in 0..lbb.len() {
                        let (d, b, n) = super::snap_span(&lbb, l, at, nb, j, sn);
                        assert_eq!(n, sn * lbb[l]);
                        for x in &mut cover[d - at..d - at + n] {
                            *x += 1;
                        }
                        for x in &mut bounce[b..b + n] {
                            *x += 1;
                        }
                        if lbb.iter().all(|&x| x == lbb[0]) {
                            // FN68b's one-dtype law
                            let bb = lbb[0];
                            assert_eq!((d, b), (at + (l * nb + j) * bb, l * sn * bb));
                        }
                    }
                    assert!(bounce.iter().all(|&x| x == 1), "the bounce tiles once");
                    j += sn;
                }
                assert!(
                    cover.iter().all(|&x| x == 1),
                    "{nb} blocks tile the snapshot once"
                );
            }
        }
    }

    #[test]
    fn the_pinned_bounce_is_all_the_cache_pins_and_holds_a_block_group() {
        assert_eq!(staging_bound(65_536, 8192), BOUNCE as u64);
        assert_eq!(staging_bound(10_240, 2048), BOUNCE as u64);
        let one =
            saber_mem::kv_tier::QSA_LAYERS as usize * super::super::super::kvwire::BB as usize;
        assert!(
            BOUNCE / one >= 64,
            "a bounce moves at least 64 blocks of every layer"
        );
    }

    /// FN113: a logical region's host pieces map onto its device segments byte for byte (a
    /// paged index store's runs and a page-split region at once), and the node layout tiles its
    /// pages without straddling.
    #[test]
    fn fn113_piece_copies_follow_the_segments_and_the_layout_tiles_pages() {
        let segs = [(1000u64, 300u64), (5000, 200)];
        let pieces = [(0u64, 250u64, 0u32, 0u64), (250, 250, 1, 0)];
        let c = piece_copies(&segs, &pieces, |pg| u64::from(pg) * 10_000);
        assert_eq!(
            c,
            vec![(1000, 0, 250), (1250, 10_000, 50), (5000, 10_050, 200)]
        );
        assert_eq!(c.iter().map(|x| x.2).sum::<u64>(), 500);
        // a 125-row node: 9 blocks of a 13-layer fp8 + bf16-MTP set, the 36 GDN layers' regions
        let set = 12 * 8192 + 16_384;
        let mut sizes = Vec::new();
        for _ in 0..36 {
            sizes.extend([71_680u64, 786_432]);
        }
        for _ in 0..13 {
            sizes.extend([2048u64, 192, 10 * 1024]);
        }
        sizes.push(13 * 10_240 * 2);
        let (groups, pieces, pages) = layout(9, set, &sizes).unwrap();
        assert_eq!(groups.iter().map(|g| g.1).sum::<u32>(), 9);
        let bytes: u64 = 9 * set + sizes.iter().sum::<u64>();
        assert!(u64::from(pages) * TREE_PAGE >= bytes);
        assert!(
            u64::from(pages) * TREE_PAGE < bytes + 2 * TREE_PAGE * 2,
            "{pages} pages"
        );
        let mut used: Vec<(u32, u64, u64)> = groups
            .iter()
            .map(|g| (g.2, g.3, u64::from(g.1) * set))
            .chain(pieces.iter().flatten().map(|p| (p.2, p.3, p.1)))
            .collect();
        used.sort_unstable();
        for w in used.windows(2) {
            assert!(w[0].2 + w[0].1 <= TREE_PAGE);
            if w[0].0 == w[1].0 {
                assert!(w[0].1 + w[0].2 <= w[1].1, "overlap {w:?}");
            }
        }
    }

    #[test]
    fn a_chunks_blocks_cover_the_stash_row_and_the_open_tail() {
        assert_eq!(chunk_blocks(0, 2048), (0, 128));
        // the drafter's stash row at p0 - 1 lives in the previous block
        assert_eq!(chunk_blocks(8192, 16_384), (511, 1024));
        assert_eq!(chunk_blocks(8200, 8300), (512, 519));
    }
}
