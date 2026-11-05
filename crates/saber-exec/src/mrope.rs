//! M-RoPE positions for a prompt that carries images (workstream D4), and the image spans the
//! embedding scatter reads.
//!
//! **The law is prod's own** — the reference 0.27.1's `Qwen3VLForConditionalGeneration.
//! _get_mrope_input_positions` (`qwen3_vl.py`), which `Qwen3_5ForConditionalGeneration` inherits:
//!
//! * a text run of `n` tokens after positions whose maximum is `m` gets `(m + 1 + i)` on all three
//!   axes (the first run starts at 0);
//! * an image of grid `(1, H, W)` patches is an `(H / 2) x (W / 2)` grid of MERGED tokens, row-major,
//!   and its token `(r, c)` gets `(p0, p0 + r, p0 + c)`, where `p0` is the next text position —
//!   `np.indices((1, h, w)) + text_len + st_idx`, i.e. the temporal axis is `p0` for EVERY token of
//!   the image;
//! * the text after an image resumes at `max(every position so far) + 1`, which is
//!   `p0 + max(H/2, W/2)`;
//! * `mrope_position_delta = max(position) + 1 - len(prompt)` — 0 for a text-only prompt, negative
//!   after an image (the D1 image's 192 tokens span 16 positions: delta -176).
//!
//! Every token AFTER the prompt (decode rows, the verify tail, the bonus row) is at
//! `index + delta` on all three axes (the reference server's `MRotaryEmbedding.get_next_input_positions`). The KV
//! slot and the causal mask stay the token INDEX; only the rope position moves.
//!
//! A text-only prompt produces `(i, i, i)` and delta 0, which is exactly the position law the
//! engine had before this module ([`crate::forward`]'s `.pos` table), byte for byte.

use std::fmt;

/// `spatial_merge_size` of the checkpoint's `vision_config`: 2x2 patches make one token.
pub const SPATIAL_MERGE: u32 = 2;

/// Why a prompt's image layout was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MropeError(pub String);

impl fmt::Display for MropeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "m-rope: {}", self.0)
    }
}

impl std::error::Error for MropeError {}

/// One image's run of `<|image_pad|>` tokens in a prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSpan {
    /// Prompt index of the image's first token (the reference server's `mm_position.offset`).
    pub start: u32,
    /// Tokens in the run: `(H / 2) * (W / 2)`.
    pub tokens: u32,
    /// Merged grid rows, `H / 2`.
    pub grid_h: u32,
    /// Merged grid columns, `W / 2`.
    pub grid_w: u32,
    /// The run's first row in the tower's output: the tokens of every earlier image of the
    /// request, because the tower writes a request's images one after another.
    pub src_row: u32,
}

/// The M-RoPE facts of one sequence with images: every prompt token's three positions, the delta
/// that places everything after the prompt, and the image runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MropePositions {
    /// Temporal axis, one per prompt token.
    pub t: Vec<i64>,
    /// Height axis.
    pub h: Vec<i64>,
    /// Width axis.
    pub w: Vec<i64>,
    /// `max(position) + 1 - prompt_len`.
    pub delta: i64,
    /// The image runs, in prompt order.
    pub spans: Vec<ImageSpan>,
}

impl MropePositions {
    /// Prompt tokens described.
    #[must_use]
    pub fn prompt_len(&self) -> u32 {
        self.t.len() as u32
    }

    /// The three positions of the token at `index`: the table inside the prompt,
    /// `index + delta` on every axis past it.
    #[must_use]
    pub fn at(&self, index: u32) -> [i64; 3] {
        let i = index as usize;
        if i < self.t.len() {
            [self.t[i], self.h[i], self.w[i]]
        } else {
            let p = i64::from(index) + self.delta;
            [p, p, p]
        }
    }

    /// Rows `[first, first + n)` of the prompt that are image tokens, as
    /// `(row within the chunk, tower row, rows)` runs — the scatter of one prefill chunk.
    ///
    /// An image may straddle a chunk boundary: each chunk takes the part of the run it covers, at
    /// the tower row that part starts at, so the union over the prompt's chunks is every image
    /// row exactly once.
    #[must_use]
    pub fn scatter_runs(&self, first: u32, n: u32) -> Vec<ScatterRun> {
        let end = first.saturating_add(n);
        let mut out = Vec::new();
        for s in &self.spans {
            let lo = s.start.max(first);
            let hi = (s.start + s.tokens).min(end);
            if lo < hi {
                out.push(ScatterRun {
                    dst_row: lo - first,
                    src_row: s.src_row + (lo - s.start),
                    rows: hi - lo,
                });
            }
        }
        out
    }

    /// One past the last image token of the prompt (0 when there is none): a decode step that
    /// starts before it would feed image rows as text.
    #[must_use]
    pub fn image_end(&self) -> u32 {
        self.spans
            .iter()
            .map(|s| s.start + s.tokens)
            .max()
            .unwrap_or(0)
    }

    /// Tower rows the request's images produce, summed.
    #[must_use]
    pub fn image_tokens(&self) -> u32 {
        self.spans.iter().map(|s| s.tokens).sum()
    }
}

/// One contiguous copy of the scatter: `rows` tower rows from `src_row` into the chunk's embedding
/// rows from `dst_row`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScatterRun {
    /// First row within the chunk.
    pub dst_row: u32,
    /// First row within the tower's output.
    pub src_row: u32,
    /// Rows.
    pub rows: u32,
}

/// Find the image runs of an EXPANDED prompt: the `k`-th maximal run of `pad` ids is image `k`,
/// and its length must be that image's token count.
///
/// # Errors
/// A run count or a run length the grids do not account for, a grid that is not one frame, or a
/// grid whose sides are not a multiple of the merge size — each by name, never a guess.
pub fn image_spans<T: Copy + PartialEq>(
    ids: &[T],
    pad: T,
    grids: &[[u32; 3]],
) -> Result<Vec<ImageSpan>, MropeError> {
    let mut runs: Vec<(u32, u32)> = Vec::new();
    let mut i = 0usize;
    while i < ids.len() {
        if ids[i] == pad {
            let s = i;
            while i < ids.len() && ids[i] == pad {
                i += 1;
            }
            runs.push((s as u32, (i - s) as u32));
        } else {
            i += 1;
        }
    }
    if runs.len() != grids.len() {
        return Err(MropeError(format!(
            "the prompt has {} run(s) of image-pad tokens for {} image(s)",
            runs.len(),
            grids.len()
        )));
    }
    let mut spans = Vec::with_capacity(runs.len());
    let mut src = 0u32;
    for (k, ((start, len), g)) in runs.iter().zip(grids).enumerate() {
        if g[0] != 1 {
            return Err(MropeError(format!(
                "image {k} has {} temporal frames: an image is one frame (prod asserts `t == 1`)",
                g[0]
            )));
        }
        if g[1] % SPATIAL_MERGE != 0 || g[2] % SPATIAL_MERGE != 0 {
            return Err(MropeError(format!(
                "image {k}'s grid {g:?} is not a multiple of the merge size {SPATIAL_MERGE}"
            )));
        }
        let (gh, gw) = (g[1] / SPATIAL_MERGE, g[2] / SPATIAL_MERGE);
        if gh * gw != *len {
            return Err(MropeError(format!(
                "image {k}'s run is {len} token(s) and its grid {g:?} makes {}",
                gh * gw
            )));
        }
        spans.push(ImageSpan {
            start: *start,
            tokens: *len,
            grid_h: gh,
            grid_w: gw,
            src_row: src,
        });
        src += len;
    }
    Ok(spans)
}

/// Prod's `_get_mrope_input_positions` over a prompt whose image runs are `spans`.
#[must_use]
pub fn mrope_positions(prompt_len: u32, spans: &[ImageSpan]) -> MropePositions {
    let n = prompt_len as usize;
    let (mut t, mut h, mut w) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    // `st_idx`: the max over everything placed so far, plus one (0 before anything).
    let mut next: i64 = 0;
    let mut st = 0u32;
    let text = |t: &mut Vec<i64>, h: &mut Vec<i64>, w: &mut Vec<i64>, len: u32, next: &mut i64| {
        for i in 0..i64::from(len) {
            t.push(*next + i);
            h.push(*next + i);
            w.push(*next + i);
        }
        *next += i64::from(len);
    };
    for s in spans {
        text(&mut t, &mut h, &mut w, s.start - st, &mut next);
        let p0 = next;
        for r in 0..i64::from(s.grid_h) {
            for c in 0..i64::from(s.grid_w) {
                t.push(p0);
                h.push(p0 + r);
                w.push(p0 + c);
            }
        }
        next = p0 + i64::from(s.grid_h.max(s.grid_w));
        st = s.start + s.tokens;
    }
    if st < prompt_len {
        text(&mut t, &mut h, &mut w, prompt_len - st, &mut next);
    }
    let max = t.iter().chain(&h).chain(&w).copied().max().unwrap_or(-1);
    MropePositions {
        t,
        h,
        w,
        delta: max + 1 - i64::from(prompt_len),
        spans: spans.to_vec(),
    }
}

/// [`image_spans`] then [`mrope_positions`]: the whole per-sequence fact from the expanded prompt
/// and the request's grids.
///
/// # Errors
/// [`image_spans`]'s.
pub fn for_prompt<T: Copy + PartialEq>(
    ids: &[T],
    pad: T,
    grids: &[[u32; 3]],
) -> Result<MropePositions, MropeError> {
    let spans = image_spans(ids, pad, grids)?;
    Ok(mrope_positions(ids.len() as u32, &spans))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAD: i32 = 9;

    /// A prompt of `pre` text tokens, an image run of `tok` pads, and `post` text tokens.
    fn prompt(parts: &[(bool, u32)]) -> Vec<i32> {
        let mut v = Vec::new();
        for &(img, n) in parts {
            for _ in 0..n {
                v.push(if img { PAD } else { 1 });
            }
        }
        v
    }

    #[test]
    fn a_text_only_prompt_is_the_index_on_every_axis_and_delta_zero() {
        let p = for_prompt(&[1, 2, 3, 4, 5], PAD, &[]).unwrap();
        assert_eq!(p.delta, 0);
        for i in 0..8 {
            assert_eq!(p.at(i), [i64::from(i); 3]);
        }
    }

    #[test]
    fn an_image_is_a_grid_at_one_temporal_position_and_the_text_resumes_past_its_max() {
        // 3 text, a (1, 4, 6) grid -> 2 x 3 = 6 tokens, 2 text.
        let ids = prompt(&[(false, 3), (true, 6), (false, 2)]);
        let p = for_prompt(&ids, PAD, &[[1, 4, 6]]).unwrap();
        assert_eq!(p.t, vec![0, 1, 2, 3, 3, 3, 3, 3, 3, 6, 7]);
        assert_eq!(p.h, vec![0, 1, 2, 3, 3, 3, 4, 4, 4, 6, 7]);
        assert_eq!(p.w, vec![0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7]);
        // max 7, 11 tokens: delta -3, and the first decode token (index 11) is at 8.
        assert_eq!(p.delta, -3);
        assert_eq!(p.at(11), [8, 8, 8]);
    }

    #[test]
    fn a_run_that_does_not_match_its_grid_is_refused_by_name() {
        let ids = prompt(&[(false, 1), (true, 5)]);
        assert!(for_prompt(&ids, PAD, &[[1, 4, 6]]).is_err());
        assert!(for_prompt(&ids, PAD, &[]).is_err());
        assert!(for_prompt(&prompt(&[(true, 6)]), PAD, &[[2, 4, 6]]).is_err());
    }

    #[test]
    fn the_scatter_covers_every_image_row_once_across_any_chunking() {
        let ids = prompt(&[(false, 5), (true, 12), (false, 3), (true, 6), (false, 4)]);
        let p = for_prompt(&ids, PAD, &[[1, 6, 8], [1, 4, 6]]).unwrap();
        for chunk in [1u32, 2, 3, 7, 16, 64] {
            let mut seen = vec![None; ids.len()];
            let mut first = 0u32;
            while first < ids.len() as u32 {
                let n = chunk.min(ids.len() as u32 - first);
                for r in p.scatter_runs(first, n) {
                    for k in 0..r.rows {
                        let at = (first + r.dst_row + k) as usize;
                        assert!(seen[at].is_none(), "row {at} scattered twice");
                        seen[at] = Some(r.src_row + k);
                    }
                }
                first += n;
            }
            // Image rows in order, tower rows 0..18 in order; text rows untouched.
            let want: Vec<Option<u32>> = {
                let mut k = 0;
                ids.iter()
                    .map(|&t| {
                        (t == PAD).then(|| {
                            k += 1;
                            k - 1
                        })
                    })
                    .collect()
            };
            assert_eq!(seen, want, "chunk {chunk}");
        }
    }

    /// The control: a scatter that forgot the straddle (tower row restarting at the chunk) must
    /// disagree with the law on a chunking that splits an image.
    #[test]
    fn a_scatter_that_restarts_the_tower_row_at_a_chunk_boundary_is_caught() {
        let ids = prompt(&[(false, 2), (true, 12), (false, 1)]);
        let p = for_prompt(&ids, PAD, &[[1, 6, 8]]).unwrap();
        let second = p.scatter_runs(8, 7);
        assert_eq!(
            second[0].src_row, 6,
            "rows 8.. of the prompt are tower rows 6.."
        );
        assert_ne!(second[0].src_row, 0);
    }
}
