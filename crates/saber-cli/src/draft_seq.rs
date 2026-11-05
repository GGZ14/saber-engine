//! `saber engine --draft-seq <capture>`: the drafter's proposal replayed at EVERY step of a
//! captured prod SEQUENCE, so that "does Saber's walk pick prod's token" is a rate and not an
//! anecdote.
//!
//! # Why this arm exists
//!
//! ws-f35 closed the shape of the temperature-0 acceptance gap: it is **one uniform number**,
//! `r = 0.957` per accepted token, flat over the block's seven positions (χ² 12.3 on 6 dof) and
//! flat over the 160 held-out prompts. Its falsifiable prediction is that **4–9 % of the walk's
//! positions pick a different token from prod's** — and every instrument this tree had could see
//! 21 of them (`chain.draft_block` identical of 56 on 3 prompts, ws-f32), which cannot
//! discriminate 0 % from 9 %.
//!
//! The arm that can is this one: prod's own per-step inputs, every step of a real generation,
//! through the engine's own drafter. **The target is prod's on both sides** — `aux.cat` is prod's
//! hidden states and the cache is prod's cache — so a difference here is the DRAFTER's (the
//! backbone, the candidate head, the selector or the walk) and nothing else. The target's own
//! contribution is the other half of the question and is measured by comparing committed ids
//! (`tools/gates/target_divergence.py`), not here.
//!
//! # What it needs from the capture, and what it does with the prefill step
//!
//! A decode step attends ~280 cache positions and writes 16 of them, so the row dump
//! `dflash_seq_probe.py` took for ws-s9 cannot seed a replay (GATES.md blind spot 14 from the
//! other end). `proposal_capture.sh --pages` adds `kv.pages.before_ctx.l{n}` — the pages the
//! step's block table names, sliced to the step's own window — and `walk.scores` /
//! `walk.candidate_ids`, which is what turns a differing token into an attribution.
//!
//! **The prefill-shaped step is seeded AFTER prod's context store and says so.** Its context is
//! the whole prompt and this engine's drafter plan is bound at `ctx_rows = block`; Saber's own
//! engine stores exactly the last tile there (ws-s9), so the replay does the same — the last
//! `block` context rows through Saber's store, on top of prod's cache for the rest. On that one
//! step the store is prod's and the arm measures the backbone, the head, the selector and the
//! walk; every decode step is seeded BEFORE the store and runs Saber's own.
//!
//! # Output
//!
//! One `.bin` triple per step — the block, the selector's scores and its candidate ids, at the
//! same widths prod's own tensors carry — plus an `index.json`. The classification (a TIE among
//! equal scores, a near-tie the declared ULP classes flip, or a wide disagreement) is
//! `tools/gates/f36_walk_report.py`, on the host, because it is arithmetic over two files and
//! wants to be re-run without a card.

use std::path::Path;

use anyhow::{bail, Context};
use saber_engine::{ForwardRunner, HipForward};

use crate::draft_replay::{repage, Capture};

/// One step's verdict, as this arm prints it.
struct StepRow {
    tag: String,
    step: u64,
    window: u32,
    ctx_rows: usize,
    prefill_shaped: bool,
    first_diff: Option<usize>,
    positions: usize,
}

fn write_bin(dir: &Path, name: &str, bytes: &[u8]) -> anyhow::Result<()> {
    std::fs::write(dir.join(name), bytes).with_context(|| format!("writing {name}"))
}

/// Replay every captured step of a sequence capture and write what the walk produced.
///
/// Returns the number of steps whose block differs from prod's.
#[allow(clippy::too_many_lines)]
pub fn run(forward: &mut HipForward, dir: &Path, out: &Path) -> anyhow::Result<usize> {
    let facts = forward.facts().clone();
    let res = facts.draft_resident.as_ref().context(
        "`--draft-seq` needs a RESIDENT drafter: pass `--draft-pack` as well as `--draft-block`",
    )?;
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("index.json"))?)?;
    if !index["pages"].as_bool().unwrap_or(false) {
        bail!(
            "{}: this capture carries no drafter PAGES, so a step's ~280 attended positions \
             cannot be seeded — retake it with `proposal_capture.sh --pages`",
            dir.display()
        );
    }
    let scored = index["scores"].as_bool().unwrap_or(false);
    let prompts = index["prompts"]
        .as_array()
        .context("the capture's index.json has no `prompts`")?;
    std::fs::create_dir_all(out)?;
    println!(
        "\n== the SEQUENCE through the engine's own drafter, against {} ({} prompt(s), \
         scores {}, plan {} eager + {} captured node(s))",
        dir.display(),
        prompts.len(),
        if scored { "captured" } else { "ABSENT" },
        res.eager_launches,
        res.captured_nodes,
    );

    let mut rows: Vec<StepRow> = Vec::new();
    let mut bad = 0usize;
    let mut manifest: Vec<serde_json::Value> = Vec::new();
    for p in prompts {
        let tag = p["tag"]
            .as_str()
            .context("a prompt with no `tag`")?
            .to_owned();
        let steps: Vec<u64> = p["steps"]
            .as_array()
            .map(|a| a.iter().filter_map(serde_json::Value::as_u64).collect())
            .unwrap_or_default();
        for step in steps {
            let stag = format!("{tag}.s{step}");
            let cap = Capture::open(dir, &stag)?;
            let layers = 5usize;
            let kv_block = cap.int("kv_block_size")? as usize;
            let table = cap.i32_table("idx.block_table").unwrap_or_default();
            let mrows = cap.shape("q.input_ids").first().copied().unwrap_or(8);
            let ctx_pos = cap.i64_table("ctx.positions")?;
            let ctx_slot0 = cap
                .i64_table("ctx.slots.l0")
                .or_else(|_| cap.i64_table("ctx.slots"))?;
            let valid = ctx_slot0.iter().take_while(|s| **s >= 0).count();
            if valid == 0 || valid > ctx_pos.len() {
                bail!(
                    "{stag}: the context store addressed {valid} of {} rows",
                    ctx_pos.len()
                );
            }
            let last_valid_pos = u32::try_from(ctx_pos[valid - 1]).unwrap_or(0);
            let seq_len = last_valid_pos + 1 + u32::try_from(mrows).unwrap_or(8);
            // THE PREFILL SHAPE: more live context rows than the plan's tile. Saber's engine
            // stores the chunk's LAST TILE there (ws-s9), so the replay seeds the cache after
            // prod's own store and runs that one tile. Declared in the row, not hidden.
            let prefill_shaped = valid > mrows;
            let when = if prefill_shaped {
                "after_ctx"
            } else {
                "before_ctx"
            };
            let shape = cap.shape(&format!("kv.pages.{when}.l0"));
            if shape.len() != 4 {
                bail!(
                    "{stag}: `kv.pages.{when}.l0` has shape {shape:?}; this arm expects \
                     [blocks, kv_heads, page_rows, 2 * head_dim]"
                );
            }
            let nkv = shape[1];
            let page_rows = shape[2];
            let head_dim = shape[3] / 2;
            let k0 = cap.bytes(&format!("kv.pages.{when}.l0"))?;
            let elem = (k0.len() / shape.iter().product::<usize>().max(1)).max(1);
            if u64::from(seq_len) > page_rows as u64 {
                bail!(
                    "{stag}: the window is {seq_len} rows and the capture carries {page_rows} \
                     of each page — every key past it would be read out of a cache position \
                     nothing wrote"
                );
            }
            let abs_of = |s: i64| -> i64 {
                let b = s / kv_block as i64;
                let o = s % kv_block as i64;
                let idx = table.iter().position(|e| i64::from(*e) == b).unwrap_or(0);
                idx as i64 * kv_block as i64 + o
            };
            // ---- the seeds ----------------------------------------------------------------
            //
            // THE LAST `mrows` CONTEXT ROWS, always. On a decode step that is every live row;
            // on the prefill-shaped step it is the tile Saber's own engine stores. The rows
            // are taken from the END of the live run, never from the padded tail.
            let take0 = valid.saturating_sub(mrows);
            let tile: Vec<usize> = (take0..valid).collect();
            let mut seeds: Vec<(String, Vec<u8>)> = Vec::new();
            let aux = cap.bytes("aux.cat")?;
            let aux_rows = cap.shape("aux.cat").first().copied().unwrap_or(1).max(1);
            let aux_stride = aux.len() / aux_rows;
            let mut aux_tile = Vec::with_capacity(mrows * aux_stride);
            for r in &tile {
                aux_tile.extend_from_slice(&aux[r * aux_stride..(r + 1) * aux_stride]);
            }
            seeds.push(("aux.cat".to_owned(), aux_tile));
            let pos_tile: Vec<u8> = tile
                .iter()
                .flat_map(|r| ctx_pos[*r].to_le_bytes())
                .collect();
            let mut rep = Vec::with_capacity(pos_tile.len() * layers);
            for _ in 0..layers {
                rep.extend_from_slice(&pos_tile);
            }
            seeds.push(("ctx.positions".to_owned(), rep));
            for l in 0..layers {
                let t = cap
                    .i64_table(&format!("ctx.slots.l{l}"))
                    .or_else(|_| cap.i64_table("ctx.slots"))?;
                let b: Vec<u8> = tile
                    .iter()
                    .flat_map(|r| {
                        let s = t[*r];
                        if s < 0 { s } else { abs_of(s) }.to_le_bytes()
                    })
                    .collect();
                seeds.push((format!("ctx.slots.l{l}"), b));
            }
            let out_bytes = (seq_len as usize).div_ceil(16) * 16 * nkv * head_dim * elem;
            for l in 0..layers {
                let src = cap.bytes(&format!("kv.pages.{when}.l{l}"))?;
                let (k, v) = repage(
                    &src,
                    &table,
                    &cap.blocks(&format!("kv.pages.{when}.l{l}")),
                    kv_block,
                    page_rows,
                    nkv,
                    head_dim,
                    elem,
                    seq_len as usize,
                    out_bytes,
                );
                seeds.push((format!("kv.k.l{l}"), k));
                seeds.push((format!("kv.v.l{l}"), v));
            }
            seeds.push(("q.input_ids".to_owned(), cap.bytes("q.input_ids")?));
            seeds.push(("q.pos".to_owned(), cap.bytes("q.positions")?));
            let qs: Vec<u8> = cap
                .i64_table("q.slots")
                .or_else(|_| cap.i64_table("idx.query_slots"))?
                .iter()
                .flat_map(|s| if *s < 0 { *s } else { abs_of(*s) }.to_le_bytes())
                .collect();
            seeds.push(("idx.query_slots".to_owned(), qs));
            let bt: Vec<u8> = (0..(seq_len as usize).div_ceil(16).max(1) as i32)
                .flat_map(i32::to_le_bytes)
                .collect();
            seeds.push(("idx.block_table".to_owned(), bt));
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
                .flat_map(|r| i64::from(r * u32::try_from(mrows).unwrap_or(8)).to_le_bytes())
                .collect();
            seeds.push(("idx.anchor_indices".to_owned(), anchors));
            seeds.push((
                "walk.idx_mapping".to_owned(),
                cap.bytes("walk.idx_mapping")
                    .or_else(|_| cap.bytes("idx.sample_idx_mapping"))?,
            ));

            // ---- what comes back ----------------------------------------------------------
            let want_block = cap
                .bytes("walk.draft_tokens")
                .or_else(|_| cap.bytes("out.draft_tokens"))?;
            let mut reads: Vec<(String, usize)> =
                vec![("arena:out.tokens".to_owned(), want_block.len())];
            if let Ok(w) = cap.bytes("walk.scores") {
                reads.push(("arena:sel.scores".to_owned(), w.len()));
            }
            if let Ok(w) = cap.bytes("walk.candidate_ids") {
                reads.push(("arena:sel.cand64".to_owned(), w.len()));
            }
            let got = forward.draft_replay(seeds, Vec::new(), reads)?;
            let by: std::collections::BTreeMap<&str, &Vec<u8>> =
                got.reads.iter().map(|(n, b)| (n.as_str(), b)).collect();
            let block = by
                .get("arena:out.tokens")
                .map(|b| (*b).clone())
                .unwrap_or_default();
            write_bin(out, &format!("{stag}.saber.tokens.bin"), &block)?;
            if let Some(b) = by.get("arena:sel.scores") {
                write_bin(out, &format!("{stag}.saber.scores.bin"), b)?;
            }
            if let Some(b) = by.get("arena:sel.cand64") {
                write_bin(out, &format!("{stag}.saber.cand.bin"), b)?;
            }
            let ours: Vec<i64> = block
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap_or_default()))
                .collect();
            let theirs: Vec<i64> = want_block
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap_or_default()))
                .collect();
            let first_diff = ours.iter().zip(&theirs).position(|(a, b)| a != b);
            if first_diff.is_some() {
                bad += 1;
            }
            manifest.push(serde_json::json!({
                "tag": tag,
                "step": step,
                "window": seq_len,
                "ctx_rows": valid,
                "prefill_shaped": prefill_shaped,
                "saber": ours,
                "prod": theirs,
                "first_diff": first_diff,
            }));
            rows.push(StepRow {
                tag: tag.clone(),
                step,
                window: seq_len,
                ctx_rows: valid,
                prefill_shaped,
                first_diff,
                positions: theirs.len(),
            });
        }
    }

    // ---- the table, and the rate the prediction is against ---------------------------------
    //
    // The per-POSITION rate is a hazard and not a mean: a position after the walk has already
    // diverged is conditioned on a different predecessor, so it is not a second independent
    // draw. The denominator is therefore the positions reached with the prefix still agreeing,
    // which is the same normalisation the acceptance conditional uses (GATES.md blind spot 18).
    let mut reached = vec![0usize; 8];
    let mut diverged = vec![0usize; 8];
    for r in &rows {
        for k in 0..r.positions {
            if r.first_diff.is_some_and(|d| d < k) {
                break;
            }
            reached[k] += 1;
            if r.first_diff == Some(k) {
                diverged[k] += 1;
            }
        }
    }
    let total_reached: usize = reached.iter().sum();
    let total_diverged: usize = diverged.iter().sum();
    println!(
        "   steps     {} over {} prompt(s); {} with a block that differs from prod's",
        rows.len(),
        prompts.len(),
        bad
    );
    for r in &rows {
        println!(
            "   {:<10} step {:>3}  window {:>5}  ctx {:>4}{}  {}",
            r.tag,
            r.step,
            r.window,
            r.ctx_rows,
            if r.prefill_shaped {
                " (PREFILL shape, seeded after prod's store)"
            } else {
                ""
            },
            match r.first_diff {
                None => "block identical".to_owned(),
                Some(d) => format!("FIRST DIFFERENCE at position {d}"),
            }
        );
    }
    println!(
        "   walk      {total_diverged} of {total_reached} position(s) reached with the prefix \
         still agreeing pick a different token = {:.2}%",
        if total_reached == 0 {
            0.0
        } else {
            100.0 * total_diverged as f64 / total_reached as f64
        }
    );
    for k in 0..7 {
        if reached[k] == 0 {
            continue;
        }
        println!(
            "   walk.p{k}    {:>5} reached, {:>4} differ = {:.2}%",
            reached[k],
            diverged[k],
            100.0 * diverged[k] as f64 / reached[k] as f64
        );
    }
    let idx = serde_json::json!({
        "capture": dir.display().to_string(),
        "steps": manifest,
        "reached": reached,
        "diverged": diverged,
    });
    std::fs::write(out.join("index.json"), serde_json::to_vec_pretty(&idx)?)?;
    println!("   wrote     {}", out.display());
    Ok(bad)
}
