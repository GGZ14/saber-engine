//! `saber engine --draft-replay <capture>`: the drafter's proposal run through the ENGINE's own
//! plumbing, against a captured decode step's inputs.
//!
//! # What this arm is for, and what it is not
//!
//! `saber draft-step --proposal` (ws-f) replays prod's proposal through a launcher that
//! `hipMalloc`s every intermediate and makes 18 host round trips before its first launch. That is
//! right for a command and wrong for a step loop, so ws-s3 built the proposal a second time
//! against the engine's plumbing: one arena laid out on the host, weights through the same
//! `Upload` the target's take, workspaces sized once before the first capture, and the drafter's
//! forward as a CAPTURED GRAPH replayed per step.
//!
//! Two implementations of one program are only worth having if something compares them. This arm
//! is that comparison: the same capture's inputs, seeded at the same points, and every
//! intermediate read back and held against **prod's own captured tensors** — so a row the staged
//! arm of `gate_proposal.sh` reports identical must be identical here too. A row that is not is
//! the engine's plumbing differing from the standalone chain's, which is exactly the class of
//! defect neither one can find alone.
//!
//! **It makes no correctness claim about the proposal.** The chain from `aux.cat` to a draft block
//! is ws-f's and its unseeded backbone half is red as of F27; the backbone rows below inherit
//! whatever that chain does. What is claimed is that the engine's launches are the same launches.
//!
//! # The discriminating instrument (ws-s7)
//!
//! Every row above is against PROD, and prod is the wrong comparator for a plumbing question: the
//! body's `chain.*` taps are an eager shadow (F30), the region calls are anchored but carry the
//! bf16 kernel projection's own ULPs (F32), and the standalone chain is itself a few bytes off
//! prod on two of the three prompts. So a red row here could be the engine's plumbing or ws-f's
//! chain and no number could say which.
//!
//! `--draft-chain-dump` is the comparator that can. `saber draft-step --proposal --no-seed
//! --dump-chain <dir>` writes the STANDALONE chain's own intermediates under these same arena
//! region names, in launch order; this arm reads that list, reads the engine's arena at every one
//! of them, and compares. Two runs of the same kernels over the same bytes, so the only honest
//! verdict is IDENTICAL and the first name in the chain's own order that is not identical is the
//! operand where the engine's plumbing departs. It found ws-s7's defect at `sel.cand64`.
//!
//! # The seams, and why the renumbering is here
//!
//! A capture holds only the KV pages the request's block table names, so every block id, slot and
//! table entry in it has to be renumbered onto the pages the ENGINE allocated — which start at 0.
//! `draft_proposal.rs` does the same thing for its own replay; the arithmetic is duplicated rather
//! than shared because that file is another workstream's and is being bisected. The duplication is
//! named here so that whoever unifies them knows there are two.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{bail, Context};
use saber_engine::{ForwardRunner, HipForward};

use crate::npy;

/// A proposal capture, read by name.
pub struct Capture {
    dir: std::path::PathBuf,
    tag: String,
    man: serde_json::Value,
}

impl Capture {
    pub fn open(dir: &Path, tag: &str) -> anyhow::Result<Self> {
        let man: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join(format!("{tag}.json")))?)?;
        Ok(Capture {
            dir: dir.to_path_buf(),
            tag: tag.to_string(),
            man,
        })
    }

    pub fn bytes(&self, key: &str) -> anyhow::Result<Vec<u8>> {
        npy::read_u8(&self.dir.join(format!("{}.{key}.npy", self.tag)))
    }

    pub fn int(&self, key: &str) -> anyhow::Result<u32> {
        let v = self
            .man
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| anyhow::anyhow!("the capture has no scalar `{key}`"))?;
        Ok(u32::try_from(v).unwrap_or(u32::MAX))
    }

    pub fn shape(&self, key: &str) -> Vec<usize> {
        self.man
            .get(key)
            .and_then(|v| v["shape"].as_array())
            .map(|a| {
                a.iter()
                    .filter_map(serde_json::Value::as_u64)
                    .map(|x| x as usize)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Which of prod's pages this tensor's file holds, in file order. A capture carries only the
    /// pages the request's block table names.
    pub fn blocks(&self, key: &str) -> Vec<i64> {
        self.man
            .get(key)
            .and_then(|v| v["blocks"].as_array())
            .map(|a| a.iter().filter_map(serde_json::Value::as_i64).collect())
            .unwrap_or_default()
    }

    pub fn i64_table(&self, key: &str) -> anyhow::Result<Vec<i64>> {
        Ok(self
            .bytes(key)?
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap_or_default()))
            .collect())
    }

    pub fn i32_table(&self, key: &str) -> anyhow::Result<Vec<i32>> {
        Ok(self
            .bytes(key)?
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap_or_default()))
            .collect())
    }
}

/// Prod's one interleaved cache per layer, `[blocks, kv_heads, prod_page, 2 * head_dim]`, re-laid
/// as the TOKEN-MAJOR `[token, kv_heads, head_dim]` the engine's own cache is when a request holds
/// its pages in order.
///
/// **THE TWO DRAFTER PAGE SIZES ARE NOT THE SAME NUMBER, and that is the seam this function is.**
/// Prod's drafter cache holds **1,648 tokens per page** (so a captured sequence never reaches a
/// second one, which is also why prod can zero-pad its drafter block table safely); Saber's is
/// The reference's **16**, because `dflash_attn` divides a key position by the committed
/// `attn_block_size` to index the table. A byte-for-byte seed of one into the other would be a
/// cache whose every token is at the wrong address — in range, silent, and rejected by the
/// acceptance law at the cost of the whole block, which is `draft_inputs`' own warning arriving
/// from the other direction.
///
/// So the replay gives the request the engine's pages **in order**, which makes
/// `block_table[i] = i` and therefore `slot(pos) = pos`: the engine's `[blocks, 16, kv_heads,
/// head_dim]` cache is exactly token-major, and this walks the capture's tokens into it one
/// position at a time.
#[allow(clippy::too_many_arguments)]
pub fn repage(
    src: &[u8],
    table: &[i32],
    file_blocks: &[i64],
    prod_page: usize,
    page_rows: usize,
    kv_heads: usize,
    head_dim: usize,
    elem: usize,
    tokens: usize,
    out_bytes: usize,
) -> (Vec<u8>, Vec<u8>) {
    let n = head_dim * elem;
    let mut k = vec![0u8; out_bytes];
    let mut v = vec![0u8; out_bytes];
    for p in 0..tokens {
        let Some(&blk) = table.get(p / prod_page) else {
            continue;
        };
        // THE CAPTURE HOLDS ONLY THE PAGES THE REQUEST NAMED, so the block id in the table is not
        // an index into the FILE: `kv.before_ctx.l0`'s own `blocks` list says which of prod's 123
        // pages the two in this file are. Reading the raw id instead runs off the end of the file
        // and the guard below silently leaves those tokens zero — which reads as "the store wrote
        // every byte wrong" rather than "the reference was never loaded", because the SEED goes
        // through this same function and a zero seeded against a zero agrees.
        let Some(fi) = file_blocks.iter().position(|b| *b == i64::from(blk)) else {
            continue;
        };
        let off = p % prod_page;
        // THE FILE MAY HOLD A PREFIX OF EACH PAGE and not the whole of it (ws-f36's
        // `--pages`, which slices a 1,648-row page to the step's own window): the slot
        // arithmetic stays prod's — `p / prod_page`, `p % prod_page` — and only the SOURCE
        // stride is the file's. A row past what the file carries is skipped rather than read
        // out of the next page's bytes.
        if off >= page_rows {
            continue;
        }
        for h in 0..kv_heads {
            let si = (((fi * kv_heads + h) * page_rows + off) * 2 * head_dim) * elem;
            let di = ((p * kv_heads) + h) * n;
            if si + 2 * n <= src.len() && di + n <= out_bytes {
                k[di..di + n].copy_from_slice(&src[si..si + n]);
                v[di..di + n].copy_from_slice(&src[si + n..si + 2 * n]);
            }
        }
    }
    (k, v)
}

/// One comparison row.
struct Row {
    site: String,
    bytes: usize,
    differing: usize,
}

fn diff(got: &[u8], want: &[u8]) -> usize {
    got.iter().zip(want).filter(|(a, b)| a != b).count() + got.len().abs_diff(want.len())
}

/// Run every prompt of `dir` through the engine's proposal and print the table.
///
/// Returns the number of rows that differ where the staged arm is identical — the gate's own
/// failure count.
#[allow(clippy::too_many_lines)]
pub fn run(forward: &mut HipForward, dir: &Path, chain: Option<&Path>) -> anyhow::Result<usize> {
    let facts = forward.facts().clone();
    let res = facts.draft_resident.as_ref().context(
        "`--draft-replay` needs a RESIDENT drafter: pass `--draft-pack` as well as `--draft-block`",
    )?;
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("index.json"))?)?;
    let tags: Vec<String> = index["prompts"]
        .as_array()
        .context("the capture's index.json has no `prompts`")?
        .iter()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    println!(
        "\n== the PROPOSAL through the engine's own launcher, against {} ({} prompt(s), step {})",
        dir.display(),
        tags.len(),
        index["step"].as_u64().unwrap_or(0)
    );
    println!(
        "   plan      {} eager launch(es) + {} captured ({} node(s)), {} host round trip(s) \
         inside the captured phase; arena {} MiB in {} region(s)",
        res.eager_launches,
        res.captured_launches,
        res.captured_nodes,
        res.host_round_trips,
        res.arena_bytes >> 20,
        res.arena_regions,
    );
    if res.host_round_trips != 0 {
        bail!(
            "the captured phase declares {} host round trip(s): it cannot be a graph",
            res.host_round_trips
        );
    }

    let mut bad = 0usize;
    // `draft_block_vs_prod`: the k=7 block the engine's proposal produced against the one prod's
    // own walk wrote, 56 bytes of i64. It is a row of the table below AND a tally, because it is
    // the only row that is a claim about the PROPOSAL rather than about a plumbing seam.
    let (mut blocks_ok, mut blocks_seen) = (0usize, 0usize);
    for tag in &tags {
        let cap = Capture::open(dir, tag)?;
        println!("\n-- {tag}");
        let layers = 5usize;
        let kv_block = cap.int("kv_block_size")? as usize;
        let table = cap.i32_table("idx.block_table").unwrap_or_default();
        // THE CACHE'S OWN SHAPE, from the capture's manifest rather than from a constant:
        // `[blocks, kv_heads, prod_page, 2 * head_dim]`, and the element width is the file's
        // length divided by that product -- 1 for the e4m3 cache prod runs.
        let shape = cap.shape("kv.before_ctx.l0");
        if shape.len() != 4 {
            bail!(
                "`kv.before_ctx.l0` has shape {shape:?}: this replay expects \
                 [blocks, kv_heads, prod_page, 2 * head_dim]"
            );
        }
        let nkv = shape[1];
        let head_dim = shape[3] / 2;
        let k0 = cap.bytes("kv.before_ctx.l0")?;
        let elem = (k0.len() / shape.iter().product::<usize>().max(1)).max(1);
        // THE ENGINE'S PAGES, IN ORDER. See `repage`: with `block_table[i] = i` the engine's
        // `[blocks, 16, kv_heads, head_dim]` cache is token-major and every slot is the position
        // itself, which is what lets a capture taken at prod's 1,648-token page be seeded at all.
        let mrows = cap.shape("q.input_ids").first().copied().unwrap_or(8) as u32;
        // THE DRAFTER'S OWN WINDOW, and it is not the target's `idx.seq_lens`.
        //
        // `saber_engine::draft_inputs` states the law — `last_valid_pos + 1 + block` — and ws-f29
        // found the same thing from the other end: the chain read `idx.seq_lens` (the scheduler's
        // per-request length, 282 on p86) where prod's own `attn.l0.seq_lens` reads 286, because
        // four of that step's eight rows were rejected and the drafter's queries follow the last
        // VALID one. The two agree only when the step accepted everything, which is why p54 and
        // p11 never saw it. So it is derived from the context tables — the last row the store
        // addressed — and held against prod's own tap when the capture carries it.
        let ctx_pos = cap.i64_table("ctx.positions")?;
        let ctx_slot0 = cap
            .i64_table("ctx.slots.l0")
            .or_else(|_| cap.i64_table("ctx.slots"))?;
        let valid = ctx_slot0.iter().take_while(|s| **s >= 0).count();
        if valid == 0 || valid > ctx_pos.len() {
            bail!(
                "{tag}: the context store addressed {valid} of {} rows — the acceptance law \
                 commits the accepted prefix plus exactly one, so row 0 is always valid",
                ctx_pos.len()
            );
        }
        let last_valid_pos = u32::try_from(ctx_pos[valid - 1]).unwrap_or(0);
        let seq_len = last_valid_pos + 1 + mrows;
        let target_len = cap
            .i32_table("idx.seq_lens")
            .ok()
            .and_then(|v| v.first().copied())
            .map_or(0, |v| u32::try_from(v).unwrap_or(0));
        // Prod's own drafter-attention window, if the capture taps it. An EXTERNAL reference for
        // the one number a replay cannot check against itself.
        let prod_window = cap
            .i32_table("attn.l0.seq_lens")
            .ok()
            .and_then(|v| v.first().copied())
            .map(|v| u32::try_from(v).unwrap_or(0));
        println!(
            "   window    {seq_len} = last valid position {last_valid_pos} + 1 + {mrows} \
             ({valid} of {} context rows addressed; the TARGET's seq_lens is {target_len}{})",
            ctx_pos.len(),
            match prod_window {
                Some(w) if w == seq_len => ", prod's own tap agrees".to_owned(),
                Some(w) => format!(", PROD'S OWN TAP SAYS {w}"),
                None => String::new(),
            }
        );
        if let Some(w) = prod_window {
            if w != seq_len {
                bail!(
                    "{tag}: the drafter's attention window derives to {seq_len} and prod's own \
                     `attn.l0.seq_lens` is {w}: every key past the shorter one is read out of a \
                     cache position the sequence never wrote"
                );
            }
        }
        let abs_of = |s: i64| -> i64 {
            let b = s / kv_block as i64;
            let o = s % kv_block as i64;
            let idx = table.iter().position(|e| i64::from(*e) == b).unwrap_or(0);
            idx as i64 * kv_block as i64 + o
        };
        let mut seeds: Vec<(String, Vec<u8>)> = Vec::new();

        // ---- the context store's inputs ---------------------------------------------------
        seeds.push(("aux.cat".to_owned(), cap.bytes("aux.cat")?));
        // `ctx.positions` is REPLICATED per layer: the batched rope runs over
        // `[layers * num_ctx]` rows at once and reads one position per row.
        let pos = cap.bytes("ctx.positions")?;
        let mut rep = Vec::with_capacity(pos.len() * layers);
        for _ in 0..layers {
            rep.extend_from_slice(&pos);
        }
        seeds.push(("ctx.positions".to_owned(), rep));
        for l in 0..layers {
            let t = cap
                .i64_table(&format!("ctx.slots.l{l}"))
                .or_else(|_| cap.i64_table("ctx.slots"))?;
            // A NEGATIVE slot is `PAD_SLOT` and stays one: it is the K/V append's own early-out,
            // and re-addressing it would point it at page 0.
            let b: Vec<u8> = t
                .iter()
                .flat_map(|s| if *s < 0 { *s } else { abs_of(*s) }.to_le_bytes())
                .collect();
            seeds.push((format!("ctx.slots.l{l}"), b));
        }
        // Whole pages of the engine's cache, so the region past the sequence is whatever it was --
        // zero, since the engine zeroes it at allocation and nothing has written it.
        let out_bytes = (seq_len as usize).div_ceil(16) * 16 * nkv * head_dim * elem;
        for l in 0..layers {
            let src = cap.bytes(&format!("kv.before_ctx.l{l}"))?;
            let (k, v) = repage(
                &src,
                &table,
                &cap.blocks(&format!("kv.before_ctx.l{l}")),
                kv_block,
                kv_block,
                nkv,
                head_dim,
                elem,
                seq_len as usize,
                out_bytes,
            );
            seeds.push((format!("kv.k.l{l}"), k));
            seeds.push((format!("kv.v.l{l}"), v));
        }

        // ---- the backbone's ----------------------------------------------------------------
        seeds.push(("q.input_ids".to_owned(), cap.bytes("q.input_ids")?));
        seeds.push(("q.pos".to_owned(), cap.bytes("q.positions")?));
        let qs: Vec<u8> = cap
            .i64_table("idx.query_slots")
            .unwrap_or_default()
            .iter()
            .flat_map(|s| if *s < 0 { *s } else { abs_of(*s) }.to_le_bytes())
            .collect();
        seeds.push(("idx.query_slots".to_owned(), qs));
        // The engine's own block table: its pages, in order.
        let bt: Vec<u8> = (0..(seq_len as usize).div_ceil(16).max(1) as i32)
            .flat_map(i32::to_le_bytes)
            .collect();
        seeds.push(("idx.block_table".to_owned(), bt));
        // The DRAFTER's own `seq_lens`, which is `last_valid_pos + 1 + block` and is NOT the
        // target's (`saber_engine::draft_inputs`, gated tier 1 against prod's own tables). The
        // capture carries the target's under `idx.seq_lens`.
        seeds.push((
            "idx.seq_lens".to_owned(),
            i32::try_from(seq_len).unwrap_or(0).to_le_bytes().to_vec(),
        ));
        seeds.push((
            "idx.sample_indices".to_owned(),
            cap.bytes("idx.sample_indices")?,
        ));
        let reqs = cap.int("num_reqs").unwrap_or(1).max(1);
        let anchors: Vec<u8> = (0..reqs)
            .flat_map(|r| i64::from(r * mrows).to_le_bytes())
            .collect();
        seeds.push(("idx.anchor_indices".to_owned(), anchors));
        seeds.push((
            "walk.idx_mapping".to_owned(),
            cap.bytes("walk.idx_mapping")?,
        ));

        // ---- what comes back ---------------------------------------------------------------
        // THE CACHES ARE READ AT THE PHASE BOUNDARY, not at the end. The context store writes
        // the target's accepted rows and the backbone's own K/V append then writes the drafter's
        // eight query rows into the SAME cache, so a read after both carries sixteen changed
        // positions and cannot say which phase wrote which. `kv.after_ctx` is prod's cache after
        // the store alone, so the comparison has to be taken there.
        let cache_bytes = out_bytes;
        let mut mid_reads: Vec<(String, usize)> = Vec::new();
        for l in 0..layers {
            mid_reads.push((format!("kv.k.l{l}"), cache_bytes));
            mid_reads.push((format!("kv.v.l{l}"), cache_bytes));
        }
        let mut reads: Vec<(String, usize)> = Vec::new();
        let want_of = |k: &str| cap.bytes(k).ok();
        let mut arena_reads: Vec<(String, String)> = Vec::new();
        // THE BACKBONE'S OWN PREFIX, so a red row names WHERE rather than only THAT. Every one of
        // these is prod's own tap or its eager shadow's, and the order is the forward's: the
        // embedding, layer 0's norm, then per layer the projection, the post-rope q and k, and the
        // attention's output. ws-f29's table is the comparator — an engine row that is worse than
        // the standalone chain's at the same site is the ENGINE's plumbing, which is the one thing
        // neither the chain nor a staged replay can tell on its own.
        arena_reads.push(("arena:bb.hidden".to_owned(), "chain.embed".to_owned()));
        arena_reads.push((
            "arena:bb.normed0".to_owned(),
            "chain.l0.input_norm".to_owned(),
        ));
        for l in 0..layers {
            arena_reads.push((format!("arena:bb.l{l}.qkv"), format!("chain.l{l}.qkv")));
            arena_reads.push((format!("arena:bb.l{l}.q_rot"), format!("attn.l{l}.q")));
            arena_reads.push((format!("arena:bb.l{l}.k_rot"), format!("attn.l{l}.k")));
            arena_reads.push((format!("arena:bb.l{l}.attn_out"), format!("attn.l{l}.out")));
        }
        for (buf, key) in [
            ("arena:head.hidden", "head.hidden"),
            ("arena:head.xs", "head.xs"),
            ("arena:head.cand80", "head.rerank_idx"),
            ("arena:sel.vals", "head.unary_logits"),
            // THE CANDIDATE IDS, and their absence from this list is how ws-s7's defect survived:
            // `head.candidate_ids` is a LIVE prod tap (`torch.topk`'s int64 indices) and the
            // engine wrote that region through a widen that read the selection's i64 output as
            // i32. Every row above it was identical and the block was wrong.
            ("arena:sel.cand64", "head.candidate_ids"),
            ("arena:sel.scores", "walk.scores"),
            ("arena:out.tokens", "out.draft_tokens"),
            // THE BLOCK, against prod's own WALK rather than against what the proposer returned.
            // The two are byte-identical in this capture and they are two different claims: one is
            // the kernel's output, the other is what prod handed the scheduler.
            ("arena:out.tokens", "walk.draft_tokens"),
        ] {
            arena_reads.push((buf.to_owned(), key.to_owned()));
        }
        for (buf, key) in &arena_reads {
            if let Some(w) = want_of(key) {
                reads.push((buf.clone(), w.len()));
            }
        }
        // THE CACHE AFTER THE BACKBONE'S OWN APPEND, against prod's `kv.after_fwd`. The mid-read
        // above is the context store's half; this is the drafter's own eight query rows, and it is
        // the row ws-f29 had to invent to find an append reading `value` — a VIEW of
        // `[q | k | v]` — at the contiguous stride. A staged replay can never see it, because it
        // seeds the attention from prod's own cache.
        for l in 0..layers {
            reads.push((format!("kv.k.l{l}"), cache_bytes));
            reads.push((format!("kv.v.l{l}"), cache_bytes));
        }
        // THE CHAIN'S OWN ORDER, and it is the chain's rather than this file's on purpose: "the
        // first operand that differs" is a statement about a PROGRAM, so the list has to be the
        // one the program wrote.
        let chain_order: Vec<(String, usize)> = match chain {
            None => Vec::new(),
            Some(c) => {
                let man: serde_json::Value = serde_json::from_slice(
                    &std::fs::read(c.join(tag).join("order.json")).with_context(|| {
                        format!("{tag}: no chain dump under {}", c.join(tag).display())
                    })?,
                )?;
                man["regions"]
                    .as_array()
                    .context("the chain dump's order.json has no `regions`")?
                    .iter()
                    .filter_map(|v| {
                        Some((
                            v["name"].as_str()?.to_owned(),
                            usize::try_from(v["bytes"].as_u64()?).ok()?,
                        ))
                    })
                    .collect()
            }
        };
        for (name, bytes) in &chain_order {
            let key = format!("arena:{name}");
            if !reads.iter().any(|(n, _)| *n == key) {
                reads.push((key, *bytes));
            }
        }

        let out = forward.draft_replay(seeds, mid_reads, reads)?;
        let by: BTreeMap<&str, &Vec<u8>> = out.reads.iter().map(|(n, b)| (n.as_str(), b)).collect();
        let at_boundary: BTreeMap<&str, &Vec<u8>> =
            out.mid_reads.iter().map(|(n, b)| (n.as_str(), b)).collect();
        let mut rows: Vec<Row> = Vec::new();
        for l in 0..layers {
            let want = cap.bytes(&format!("kv.after_ctx.l{l}"))?;
            let (wk, wv) = repage(
                &want,
                &table,
                &cap.blocks(&format!("kv.after_ctx.l{l}")),
                kv_block,
                kv_block,
                nkv,
                head_dim,
                elem,
                seq_len as usize,
                out_bytes,
            );
            for (half, w) in [("k", wk), ("v", wv)] {
                let name = format!("kv.{half}.l{l}");
                if let Some(g) = at_boundary.get(name.as_str()) {
                    rows.push(Row {
                        site: format!("ctx.kv.l{l}.{half}"),
                        bytes: w.len(),
                        differing: diff(g, &w),
                    });
                }
            }
        }
        for (buf, key) in &arena_reads {
            let (Some(g), Some(w)) = (by.get(buf.as_str()), want_of(key)) else {
                continue;
            };
            rows.push(Row {
                // The capture's own key, with one `chain.` prefix and not two: the shadow rows
                // already carry it and the live attention taps do not.
                site: if key.starts_with("chain.") {
                    key.clone()
                } else {
                    format!("chain.{key}")
                },
                bytes: w.len(),
                differing: diff(g, &w),
            });
        }
        for l in 0..layers {
            let Ok(want) = cap.bytes(&format!("kv.after_fwd.l{l}")) else {
                continue;
            };
            let (wk, wv) = repage(
                &want,
                &table,
                &cap.blocks(&format!("kv.after_fwd.l{l}")),
                kv_block,
                kv_block,
                nkv,
                head_dim,
                elem,
                seq_len as usize,
                out_bytes,
            );
            for (half, w) in [("k", wk), ("v", wv)] {
                let name = format!("kv.{half}.l{l}");
                if let Some(g) = by.get(name.as_str()) {
                    rows.push(Row {
                        site: format!("fwd.kv.l{l}.{half}"),
                        bytes: w.len(),
                        differing: diff(g, &w),
                    });
                }
            }
        }
        println!(
            "   ran       {} eager launch(es), {} replayed node(s), {} host round trip(s)",
            out.eager_launches, out.captured_nodes, out.host_round_trips
        );
        for r in &rows {
            println!(
                "   {:<22} {} of {} bytes",
                r.site,
                if r.differing == 0 {
                    "identical".to_owned()
                } else {
                    format!("DIFFERS in {}", r.differing)
                },
                r.bytes
            );
        }
        // ---- THE ENGINE AGAINST THE STANDALONE CHAIN, operand by operand -------------------
        if let Some(c) = chain {
            let mut first: Option<(String, usize, usize)> = None;
            let (mut same, mut moved) = (0usize, 0usize);
            for (name, bytes) in &chain_order {
                let want = std::fs::read(c.join(tag).join(format!("{name}.bin")))
                    .with_context(|| format!("{tag}: the chain dumped no `{name}`"))?;
                let Some(got) = by.get(format!("arena:{name}").as_str()) else {
                    continue;
                };
                let d = diff(got, &want);
                if d == 0 {
                    same += 1;
                } else {
                    moved += 1;
                    if first.is_none() {
                        first = Some((name.clone(), d, *bytes));
                    }
                    // Every departing operand is printed, because the first one names the defect
                    // and the rest say how far it reached.
                    println!("   engine-vs-chain {name:<22} DIFFERS in {d} of {bytes} bytes");
                }
            }
            match &first {
                None => println!(
                    "   engine-vs-chain  {same} of {} operand(s) IDENTICAL: the engine's \
                     plumbing is the standalone chain's program",
                    chain_order.len()
                ),
                Some((name, d, bytes)) => {
                    println!(
                        "   engine-vs-chain  FIRST DEPARTURE at `{name}`: {d} of {bytes} bytes \
                         ({same} identical, {moved} moved)"
                    );
                    bad += moved;
                }
            }
        }
        if let Some(r) = rows.iter().find(|r| r.site == "chain.walk.draft_tokens") {
            blocks_seen += 1;
            if r.differing == 0 {
                blocks_ok += 1;
            }
            println!(
                "   block     draft_block_vs_prod {} of {} bytes",
                if r.differing == 0 {
                    "identical".to_owned()
                } else {
                    format!("DIFFERS in {}", r.differing)
                },
                r.bytes
            );
        }
        // THE VERDICT IS THE CONTEXT STORE'S, and the backbone's is REPORTED and not gated. The
        // staged arm of `gate_proposal.sh` is identical on the context store's caches (1 / 0 / 2
        // bytes of 3,375,104 as of ws-f27, which is the drafter pack's own 2 ppm), and the
        // engine's plumbing must reproduce that. The backbone rows carry ws-f's unseeded chain,
        // which is red at or before `attn.l0.q` and is F28's to close — so a number here is a
        // COMPARISON with that chain, not a claim about the model.
        let ctx_bad = rows
            .iter()
            .filter(|r| r.site.starts_with("ctx.kv.") && r.differing > 8)
            .count();
        if ctx_bad > 0 {
            println!(
                "   FAIL      {ctx_bad} context-store cache(s) differ by more than the drafter \
                 pack's own residual: the ENGINE's plumbing does not reproduce the staged arm"
            );
            bad += ctx_bad;
        } else {
            println!(
                "   OK        every context-store cache is within the pack's own 2 ppm, which is \
                 what the staged arm reports"
            );
        }
    }
    // THE BLOCK IS GATED, and it is the row that moved: ws-f29 closed the unseeded standalone
    // chain at `identical of 56` on 3/3, so the engine's own plumbing must reach the same block
    // from the same inputs. A block that differs where the standalone chain's does not is the
    // engine's, which is the whole reason two implementations of one program are worth having.
    println!(
        "\n   draft_block_vs_prod  {blocks_ok}/{blocks_seen} prompt(s) identical of 56 bytes \
         against prod's own `walk.draft_tokens`"
    );
    if blocks_seen > 0 && blocks_ok < blocks_seen {
        bad += blocks_seen - blocks_ok;
    }
    Ok(bad)
}

#[cfg(test)]
mod tests {
    use super::repage;

    /// One page of `prod_page` rows, `kv_heads` heads, `head_dim` elements a half, filled so that
    /// every (token, head, half, element) byte is distinguishable.
    fn page(prod_page: usize, rows: usize, kv_heads: usize, head_dim: usize) -> Vec<u8> {
        let mut v = vec![0u8; rows * kv_heads * 2 * head_dim];
        for t in 0..rows {
            for h in 0..kv_heads {
                for e in 0..2 * head_dim {
                    // The head axis is OUTSIDE the row axis in prod's layout, which is the whole
                    // point of the stride this test is about.
                    v[(h * rows + t) * 2 * head_dim + e] = ((t * 7 + h * 3 + e) % 251 + 1) as u8;
                }
            }
        }
        let _ = prod_page;
        v
    }

    /// **A SLICED page and a WHOLE page must repage to the same bytes.** ws-f36's sequence
    /// capture stores the first `page_rows` rows of a 1,648-row page, because the window is a few
    /// hundred and the page is 3.4 MiB a layer; the slot arithmetic stays prod's (`slot / 1648`,
    /// `slot % 1648`) and only the SOURCE stride is the file's. Getting that wrong reads every
    /// token but the first head's at the wrong address — in range, silent, and a cache whose
    /// every token is somewhere else is exactly the failure `draft_replay`'s own doc warns about.
    #[test]
    fn a_sliced_page_repages_to_what_the_whole_page_does() {
        let (prod_page, kv_heads, head_dim, tokens) = (1648usize, 8usize, 4usize, 40usize);
        let whole = page(prod_page, prod_page, kv_heads, head_dim);
        let rows = 64usize;
        // The same bytes, kept only for the first `rows` rows of the page.
        let mut sliced = vec![0u8; rows * kv_heads * 2 * head_dim];
        for h in 0..kv_heads {
            for t in 0..rows {
                let src = (h * prod_page + t) * 2 * head_dim;
                let dst = (h * rows + t) * 2 * head_dim;
                sliced[dst..dst + 2 * head_dim].copy_from_slice(&whole[src..src + 2 * head_dim]);
            }
        }
        let table = [0i32];
        let blocks = [0i64];
        let out_bytes = tokens.div_ceil(16) * 16 * kv_heads * head_dim;
        let a = repage(
            &whole, &table, &blocks, prod_page, prod_page, kv_heads, head_dim, 1, tokens, out_bytes,
        );
        let b = repage(
            &sliced, &table, &blocks, prod_page, rows, kv_heads, head_dim, 1, tokens, out_bytes,
        );
        assert_eq!(a.0, b.0, "K halves");
        assert_eq!(a.1, b.1, "V halves");
        assert!(a.0.iter().any(|x| *x != 0), "the fixture wrote nothing");

        // THE NEGATIVE CONTROL: reading the sliced file at the WHOLE page's stride — the bug this
        // test exists to catch — must NOT agree. It is the same call with one argument wrong.
        let wrong = repage(
            &sliced, &table, &blocks, prod_page, prod_page, kv_heads, head_dim, 1, tokens,
            out_bytes,
        );
        assert_ne!(
            a.0, wrong.0,
            "reading a sliced page at the whole page's stride must not agree with the truth"
        );
    }

    /// A token past what the file carries contributes NOTHING rather than the next page's bytes.
    #[test]
    fn a_token_past_the_slice_is_left_alone() {
        let (prod_page, kv_heads, head_dim) = (1648usize, 2usize, 4usize);
        let rows = 8usize;
        let src = page(prod_page, rows, kv_heads, head_dim);
        let out_bytes = 32 * kv_heads * head_dim;
        let (k, _v) = repage(
            &src,
            &[0i32],
            &[0i64],
            prod_page,
            rows,
            kv_heads,
            head_dim,
            1,
            20,
            out_bytes,
        );
        for t in rows..20 {
            for h in 0..kv_heads {
                let at = ((t * kv_heads) + h) * head_dim;
                assert!(
                    k[at..at + head_dim].iter().all(|x| *x == 0),
                    "token {t} is past the {rows} rows the file carries and was written anyway"
                );
            }
        }
        assert!(
            k[..head_dim].iter().any(|x| *x != 0),
            "token 0 was not written"
        );
    }
}
