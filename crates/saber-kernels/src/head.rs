//! The head plugin: the two ends of a forward pass, as launch rules.
//!
//! `embedding` starts a pass, `row_gather` + `argmax` finish it. Prod has none of the three as a
//! kernel — the reference server's `VocabParallelEmbedding` is `index_select`, the row selection before `lm_head`
//! is `index_select` inside `LogitsProcessor`, and greedy sampling is `torch.argmax` — so unlike
//! every other rules module in this crate there is no reference launcher to port. What IS ported
//! is the discipline: geometry out of `rules/head_gfx1201.toml`, the kernarg layout out of
//! [`crate::abi::signature_for`], nothing a literal at a call site, and every shape this cannot
//! launch refused by name.
//!
//! The three shapes are the same shape — `rows` output rows of `cols` elements — which is why one
//! [`HeadShape`] serves all of them.

use serde::Deserialize;

use crate::abi;
use crate::rules::{ArgValue, LaunchDescriptor, Slot};
use crate::{Error, Result};

const PROD_RULES: &str = include_str!("../rules/head_gfx1201.toml");

/// The same text, for the plan input hash: the policy this module enforces.
pub const PROD_RULES_TEXT: &str = PROD_RULES;

/// The one code object the three kernels live in.
const PLUGIN: &str = "head";

/// `ARGMAX_T` in `hip/head.hip`: the argmax kernel's LDS arrays are compiled for this width, so a
/// rules file that named another one would launch a kernel whose shared memory is the wrong size.
const ARGMAX_BLOCK: u32 = 256;

/// `DEVROW_T` in `hip/head.hip`, for the same reason as [`ARGMAX_BLOCK`]: the seam kernel's eight
/// per-wave histograms and its reduction array are compiled for this width.
const DEVICE_ROW_BLOCK: u32 = 256;

/// `DEVROW_K` in `hip/head.hip`: the seam's width. The kernel writes exactly this many values and
/// ids per row and the host's `DeviceRow` reads exactly this many, so a rules file that said
/// anything else would describe a different seam from the one the kernel writes.
pub const DEVICE_ROW_K: u32 = 64;

/// Geometry knobs (`[knobs]`).
#[derive(Debug, Clone, Deserialize)]
pub struct Knobs {
    /// Threads per workgroup; one workgroup per output row.
    pub block: u32,
    /// Elements per 16-byte group in the copy kernels.
    pub vector: u32,
    /// The largest row count a single launch may be asked for.
    pub max_rows: u32,
    /// Threads per workgroup for the sampler's seam. Separate from `block` because the two
    /// kernels' LDS is compiled for their own width, and one number for both would tie a change
    /// in either to the other.
    pub device_row_block: u32,
    /// Candidates the seam carries per row. `DEVROW_K` in the kernel.
    pub device_row_k: u32,
    /// The longest penalty-id list one row may gather. The list is the prompt's and the output's
    /// tokens; a request whose set is wider than this is refused by the caller **by name** rather
    /// than gathered short, because a penalty the seam did not carry is a logit the host would
    /// sample as if nothing had moved it.
    pub device_row_penalty_max: u32,
}

/// One row: an op name and the symbol it launches.
#[derive(Debug, Clone, Deserialize)]
pub struct HeadRow {
    /// The op id `resolve` is asked with.
    pub op: String,
    /// The symbol in the code object.
    pub symbol: String,
}

/// The head plugin's rules.
#[derive(Debug, Clone, Deserialize)]
pub struct HeadRules {
    /// Schema version; 1 for this build.
    pub schema_version: u32,
    /// Human name.
    pub name: String,
    /// Geometry.
    pub knobs: Knobs,
    /// The op → symbol table.
    #[serde(default)]
    pub kernel: Vec<HeadRow>,
}

impl HeadRules {
    /// The committed rules.
    pub fn prod_head() -> Result<Self> {
        Self::from_toml(PROD_RULES)
    }

    /// Parse and check.
    pub fn from_toml(text: &str) -> Result<Self> {
        let r: HeadRules = toml::from_str(text).map_err(|e| Error::Parse(e.to_string()))?;
        r.check()?;
        Ok(r)
    }

    fn check(&self) -> Result<()> {
        if self.schema_version != 1 {
            return Err(Error::Rule(format!(
                "head rules schema {} (this build speaks 1)",
                self.schema_version
            )));
        }
        if self.knobs.block != ARGMAX_BLOCK {
            return Err(Error::Rule(format!(
                "head rules say block {} but `saber_argmax_bf16` reduces in LDS arrays compiled \
                 for {ARGMAX_BLOCK}",
                self.knobs.block
            )));
        }
        if self.knobs.vector == 0 || self.knobs.max_rows == 0 {
            return Err(Error::Rule(
                "head knobs `vector`/`max_rows` must be > 0".into(),
            ));
        }
        if self.knobs.device_row_block != DEVICE_ROW_BLOCK {
            return Err(Error::Rule(format!(
                "head rules say device_row_block {} but `saber_device_row_bf16` histograms in LDS \
                 arrays compiled for {DEVICE_ROW_BLOCK}",
                self.knobs.device_row_block
            )));
        }
        if self.knobs.device_row_k != DEVICE_ROW_K {
            return Err(Error::Rule(format!(
                "head rules say device_row_k {} but `saber_device_row_bf16` writes \
                 {DEVICE_ROW_K} candidates per row",
                self.knobs.device_row_k
            )));
        }
        if self.knobs.device_row_penalty_max == 0 {
            return Err(Error::Rule(
                "head knob `device_row_penalty_max` must be > 0; a seam that gathers no penalty \
                 logit cannot serve a penalised row at all"
                    .into(),
            ));
        }
        for (i, row) in self.kernel.iter().enumerate() {
            if self.kernel[..i].iter().any(|r| r.op == row.op) {
                return Err(Error::Rule(format!(
                    "head rules state op `{}` twice",
                    row.op
                )));
            }
            abi::signature_for(&row.symbol)
                .ok_or_else(|| Error::Rule(format!("{}: no declared signature", row.symbol)))?;
        }
        for op in [
            HeadOp::Embedding,
            HeadOp::RowGather,
            HeadOp::Argmax,
            HeadOp::DeviceRow { penalties: 0 },
            HeadOp::RowCopy { dst_stride: 0 },
            HeadOp::WidenI64 { dst_stride: 0 },
        ] {
            if !self.kernel.iter().any(|r| r.op == op.as_str()) {
                return Err(Error::Rule(format!(
                    "head rules describe no op `{}`; the forward pass has no other way to do it",
                    op.as_str()
                )));
            }
        }
        Ok(())
    }

    /// Every symbol these rules name.
    pub fn symbols(&self) -> Vec<&str> {
        self.kernel.iter().map(|r| r.symbol.as_str()).collect()
    }

    /// The plugin these symbols live in — asked of the rules, never guessed from a file stem.
    pub fn owner(&self) -> &'static str {
        PLUGIN
    }

    fn row(&self, op: HeadOp) -> Result<&HeadRow> {
        self.kernel
            .iter()
            .find(|r| r.op == op.as_str())
            .ok_or_else(|| Error::Rule(format!("the head rules describe no op `{}`", op.as_str())))
    }
}

/// The four ops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadOp {
    /// Vocab-parallel embedding lookup; a token outside `[begin, end)` writes a zero row.
    Embedding,
    /// Select rows of a `[rows, cols]` bf16 tensor by index.
    RowGather,
    /// Greedy sampling: the largest logit per row, ties to the lowest index.
    Argmax,
    /// The sampler's seam: the K largest logits and ids in the argmax's own total order, both
    /// log-sum-exps, and the raw logits at the penalty ids.
    ///
    /// `penalties` is how many ids this launch's list carries per row — a property of the STEP
    /// (the requests in it), not of the model, which is why it rides on the op rather than on
    /// [`HeadShape`]: two steps of the same shape can want different list widths and each is a
    /// different kernarg.
    DeviceRow {
        /// Penalty ids per row, `0..=Knobs::device_row_penalty_max`.
        penalties: u32,
    },
    /// Copy `rows x cols` bf16 into a **column block** of a wider destination: the DFlash2 aux
    /// tap.
    ///
    /// `dst_stride` rides on the OP rather than on [`HeadShape`] for the same reason
    /// [`HeadOp::DeviceRow`]'s `penalties` does: [`HeadShape`] describes the SOURCE view this
    /// launch reads, and where the answer lands is the caller's layout, not a property of the
    /// tensor being read. Two taps of the same source shape write into two different column
    /// blocks of the same buffer and differ in nothing else.
    RowCopy {
        /// Elements between consecutive DESTINATION rows. `5 * hidden` for the five-layer tap.
        dst_stride: i64,
    },
    /// Sign-extend an `i32` table to `i64`, rows `src_stride` / `dst_stride` apart.
    ///
    /// One launch of the drafter's captured proposal that prod does not make, and it is declared
    /// as its own op so that it is counted: `dflash_topk_bf16_k16` writes the candidate ids as
    /// `int` and the selector's score and walk both read them as `const long *`.
    WidenI64 {
        /// Elements between consecutive DESTINATION rows, in i64 units.
        dst_stride: i64,
    },
    /// The bf16 head's final norm (workstream N2): residual add + Gemma RMS norm with the bf16
    /// row as the output, for a `lm_head` that is a bf16 GEMM. `cols` is the hidden size.
    ///
    /// `eps_bits` is the `f32` epsilon's bit pattern, so the op stays `Eq`.
    AddRmsBf16 {
        /// `f32::to_bits` of the norm's epsilon.
        eps_bits: u32,
    },
}

impl HeadOp {
    /// The op id.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            HeadOp::Embedding => "embedding",
            HeadOp::RowGather => "row_gather",
            HeadOp::Argmax => "argmax",
            HeadOp::DeviceRow { .. } => "device_row",
            HeadOp::RowCopy { .. } => "row_copy",
            HeadOp::WidenI64 { .. } => "widen_i64",
            HeadOp::AddRmsBf16 { .. } => "add_rms_bf16",
        }
    }
}

/// `saber_add_rms_bf16`'s register budget: `ARB_MAXG` groups of 8 per thread, so a row is at
/// most `8 * 8 * block` wide. A wider one would be silently truncated by the kernel's loop bound,
/// so [`resolve`] refuses it by name.
const ADD_RMS_BF16_MAXG: u32 = 4;

/// What all three take: `rows` output rows of `cols` bf16 elements, read `stride` apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadShape {
    /// Output rows — the grid.
    pub rows: u32,
    /// Elements per row.
    pub cols: u32,
    /// Elements between consecutive SOURCE rows. Equal to `cols` for a packed tensor; different
    /// when the source is a view (the row gather reads the residual stream, whose rows are the
    /// hidden size, and the argmax reads a `[rows, vocab]` logit tile).
    pub src_stride: i64,
    /// This rank's vocabulary slice, for the embedding. `(0, vocab)` at TP = 1.
    pub vocab: (u32, u32),
}

impl HeadShape {
    /// A packed `[rows, cols]` tensor with no vocab slicing.
    #[must_use]
    pub fn packed(rows: u32, cols: u32) -> Self {
        HeadShape {
            rows,
            cols,
            src_stride: i64::from(cols),
            vocab: (0, 0),
        }
    }
}

/// The one decision function: (op, shape, rules) in, descriptor out. Pure, no environment.
///
/// # Errors
/// A shape the kernels cannot serve: no rows or no columns, a grid past `max_rows`, a source
/// stride narrower than the row it has to carry, or — for the embedding — an empty vocabulary
/// slice.
pub fn resolve(op: HeadOp, shape: HeadShape, rules: &HeadRules) -> Result<LaunchDescriptor> {
    let row = rules.row(op)?;
    let kn = &rules.knobs;
    if shape.rows == 0 || shape.cols == 0 {
        return Err(Error::Rule(format!(
            "{}: {} rows x {} cols has nothing to do; an empty launch is a caller bug, not a \
             no-op",
            op.as_str(),
            shape.rows,
            shape.cols
        )));
    }
    if shape.rows > kn.max_rows {
        return Err(Error::Rule(format!(
            "{}: {} rows exceeds max_rows {}",
            op.as_str(),
            shape.rows,
            kn.max_rows
        )));
    }
    if shape.src_stride < i64::from(shape.cols) {
        return Err(Error::Rule(format!(
            "{}: source stride {} is narrower than the {} columns it has to carry",
            op.as_str(),
            shape.src_stride,
            shape.cols
        )));
    }
    // A destination stride narrower than the row it has to carry overlaps the next row — silently,
    // since every element written is in range of the buffer. Refused with the same words the
    // source stride is refused with.
    if let HeadOp::RowCopy { dst_stride } | HeadOp::WidenI64 { dst_stride } = op {
        if dst_stride < i64::from(shape.cols) {
            return Err(Error::Rule(format!(
                "{}: destination stride {dst_stride} is narrower than the {} columns it has to \
                 carry, so row r + 1 would overwrite the tail of row r",
                op.as_str(),
                shape.cols
            )));
        }
    }
    if let HeadOp::DeviceRow { penalties } = op {
        if penalties > kn.device_row_penalty_max {
            return Err(Error::Rule(format!(
                "device_row: {penalties} penalty ids per row is wider than the seam's {}; a row \
                 whose penalties move more ids than this cannot be sampled through it",
                kn.device_row_penalty_max
            )));
        }
        if shape.cols < kn.device_row_k {
            return Err(Error::Rule(format!(
                "device_row: a {}-column row cannot fill a {}-wide seam",
                shape.cols, kn.device_row_k
            )));
        }
    }
    let sig = abi::signature_for(&row.symbol)
        .ok_or_else(|| Error::Rule(format!("{}: no declared signature", row.symbol)))?;
    let (packed, bytes) = abi::pack(sig);
    let value = |name: &str| -> Result<ArgValue> {
        Ok(match name {
            "hidden" | "cols" | "k" => ArgValue::I32 {
                ival: i64::from(shape.cols),
            },
            "eps" => ArgValue::F32 {
                fval: match op {
                    HeadOp::AddRmsBf16 { eps_bits } => f32::from_bits(eps_bits),
                    _ => 0.0,
                },
            },
            "pen_max" => ArgValue::I32 {
                ival: match op {
                    HeadOp::DeviceRow { penalties } => i64::from(penalties),
                    _ => 0,
                },
            },
            "vocab_begin" => ArgValue::I32 {
                ival: i64::from(shape.vocab.0),
            },
            "vocab_end" => ArgValue::I32 {
                ival: i64::from(shape.vocab.1),
            },
            "src_stride" | "stride" => ArgValue::I64 {
                ival: shape.src_stride,
            },
            "dst_stride" => ArgValue::I64 {
                ival: match op {
                    HeadOp::RowCopy { dst_stride } | HeadOp::WidenI64 { dst_stride } => dst_stride,
                    _ => i64::from(shape.cols),
                },
            },
            other => ArgValue::Buffer {
                buffer: other.to_owned(),
            },
        })
    };
    if let HeadOp::AddRmsBf16 { .. } = op {
        if shape.cols % 8 != 0 || shape.cols > ADD_RMS_BF16_MAXG * 8 * kn.block {
            return Err(Error::Rule(format!(
                "add_rms_bf16: K={} must be 8-aligned and at most {} (the kernel holds {} groups \
                 of 8 per thread at {} threads)",
                shape.cols,
                ADD_RMS_BF16_MAXG * 8 * kn.block,
                ADD_RMS_BF16_MAXG,
                kn.block
            )));
        }
        if shape.src_stride != i64::from(shape.cols) {
            return Err(Error::Rule(
                "add_rms_bf16: the norm reads packed rows; a strided source is not its ABI".into(),
            ));
        }
    }
    if op == HeadOp::Embedding && shape.vocab.1 <= shape.vocab.0 {
        return Err(Error::Rule(format!(
            "embedding: this rank's vocabulary slice [{}, {}) is empty",
            shape.vocab.0, shape.vocab.1
        )));
    }
    let mut args = Vec::with_capacity(sig.len());
    for (a, (offset, size)) in sig.iter().zip(packed) {
        args.push(Slot {
            name: a.name.to_owned(),
            offset,
            size,
            value: value(a.name)?,
        });
    }
    let block = match op {
        HeadOp::DeviceRow { .. } => kn.device_row_block,
        _ => kn.block,
    };
    Ok(LaunchDescriptor {
        op: row.op.clone(),
        plugin: PLUGIN.into(),
        kernel: row.symbol.clone(),
        // One workgroup per output row. Not a tile choice: the rows are independent and the work
        // per row is a copy or a reduction, so this is what fills the machine at prefill and is
        // still one launch at M = 1.
        grid: [shape.rows, 1, 1],
        block: [block, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![],
        n_launches: 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules() -> HeadRules {
        HeadRules::prod_head().expect("the head rules parse")
    }

    #[test]
    fn the_committed_rules_resolve_all_three_ops() {
        let r = rules();
        assert_eq!(r.owner(), "head");
        assert_eq!(
            r.symbols(),
            vec![
                "saber_embed_bf16",
                "saber_row_gather_bf16",
                "saber_argmax_bf16",
                "saber_device_row_bf16",
                "saber_row_copy_bf16",
                "saber_widen_i32_i64",
                "saber_add_rms_bf16"
            ]
        );

        // The embedding at a 2048-token prefill, TP = 1: the whole vocabulary is this rank's.
        let d = resolve(
            HeadOp::Embedding,
            HeadShape {
                rows: 2048,
                cols: 5120,
                src_stride: 5120,
                vocab: (0, 248_320),
            },
            &r,
        )
        .expect("embedding resolves");
        assert_eq!(d.kernel, "saber_embed_bf16");
        assert_eq!(d.grid, [2048, 1, 1]);
        assert_eq!(d.block, [256, 1, 1]);
        assert_eq!(d.n_launches, 1);
        assert_eq!(
            d.arg("hidden").map(|s| s.value.clone()),
            Some(ArgValue::I32 { ival: 5120 })
        );
        assert_eq!(
            d.arg("vocab_end").map(|s| s.value.clone()),
            Some(ArgValue::I32 { ival: 248_320 })
        );
        assert!(d.workspaces.is_empty());

        // The row selection before lm_head: one row out of a 2048-row residual stream.
        let g = resolve(
            HeadOp::RowGather,
            HeadShape {
                rows: 1,
                cols: 5120,
                src_stride: 5120,
                vocab: (0, 0),
            },
            &r,
        )
        .expect("row_gather resolves");
        assert_eq!(g.grid, [1, 1, 1]);
        assert_eq!(
            g.arg("src_stride").map(|s| s.value.clone()),
            Some(ArgValue::I64 { ival: 5120 })
        );

        // Greedy sampling over the full vocabulary.
        let a = resolve(HeadOp::Argmax, HeadShape::packed(1, 248_320), &r).expect("argmax");
        assert_eq!(a.grid, [1, 1, 1]);
        assert_eq!(
            a.arg("cols").map(|s| s.value.clone()),
            Some(ArgValue::I32 { ival: 248_320 })
        );
    }

    #[test]
    fn every_buffer_argument_binds_to_its_own_name() {
        // What lets a caller write `ArgBindings::with("table", addr)` and nothing in between
        // invent a name.
        let r = rules();
        let d = resolve(
            HeadOp::Embedding,
            HeadShape {
                rows: 4,
                cols: 5120,
                src_stride: 5120,
                vocab: (0, 1024),
            },
            &r,
        )
        .unwrap();
        for name in ["table", "ids", "out"] {
            assert_eq!(
                d.arg(name).map(|s| s.value.clone()),
                Some(ArgValue::Buffer {
                    buffer: name.to_owned()
                }),
                "{name}"
            );
        }
    }

    #[test]
    fn shapes_the_kernels_cannot_serve_are_refused_by_name() {
        let r = rules();
        // An empty launch.
        assert!(resolve(HeadOp::Argmax, HeadShape::packed(0, 10), &r).is_err());
        assert!(resolve(HeadOp::Argmax, HeadShape::packed(1, 0), &r).is_err());
        // A grid past the declared bound.
        assert!(resolve(HeadOp::RowGather, HeadShape::packed(1 << 20, 8), &r).is_err());
        // A stride that cannot carry the row.
        assert!(resolve(
            HeadOp::RowGather,
            HeadShape {
                rows: 1,
                cols: 5120,
                src_stride: 4096,
                vocab: (0, 0)
            },
            &r
        )
        .is_err());
        // An empty vocabulary slice: the embedding would write zeros for every token and the
        // all-reduce would sum nothing.
        assert!(resolve(
            HeadOp::Embedding,
            HeadShape {
                rows: 4,
                cols: 5120,
                src_stride: 5120,
                vocab: (100, 100)
            },
            &r
        )
        .is_err());
    }

    /// The seam's own resolve: the grid is the rows, the block is ITS knob, and `pen_max` is the
    /// step's list width rather than anything the model knows.
    #[test]
    fn the_device_row_seam_resolves_at_its_own_width() {
        let r = rules();
        let d = resolve(
            HeadOp::DeviceRow { penalties: 0 },
            HeadShape::packed(5, 248_320),
            &r,
        )
        .expect("device_row resolves");
        assert_eq!(d.kernel, "saber_device_row_bf16");
        assert_eq!(d.grid, [5, 1, 1]);
        assert_eq!(d.block, [256, 1, 1]);
        assert_eq!(d.n_launches, 1);
        assert_eq!(
            d.arg("pen_max").map(|s| s.value.clone()),
            Some(ArgValue::I32 { ival: 0 })
        );
        assert_eq!(
            d.arg("cols").map(|s| s.value.clone()),
            Some(ArgValue::I32 { ival: 248_320 })
        );
        for name in [
            "logits",
            "temps",
            "pen_ids",
            "top_values",
            "top_ids",
            "lse",
            "pen_values",
        ] {
            assert_eq!(
                d.arg(name).map(|s| s.value.clone()),
                Some(ArgValue::Buffer {
                    buffer: name.to_owned()
                }),
                "{name}"
            );
        }
        // A penalised row carries its list width, and a list wider than the seam is refused.
        let d = resolve(
            HeadOp::DeviceRow { penalties: 128 },
            HeadShape::packed(1, 248_320),
            &r,
        )
        .expect("a penalised row resolves");
        assert_eq!(
            d.arg("pen_max").map(|s| s.value.clone()),
            Some(ArgValue::I32 { ival: 128 })
        );
        let e = resolve(
            HeadOp::DeviceRow { penalties: 4096 },
            HeadShape::packed(1, 248_320),
            &r,
        )
        .expect_err("a list wider than the seam is refused");
        assert!(e.to_string().contains("wider than the seam's"), "{e}");
        // And a row narrower than K cannot fill the seam at all.
        assert!(resolve(
            HeadOp::DeviceRow { penalties: 0 },
            HeadShape::packed(1, 32),
            &r
        )
        .is_err());
    }

    /// The aux tap's two strides are INDEPENDENT, and a destination stride that cannot carry the
    /// row is refused by name.
    ///
    /// This is the whole reason the tap is not `row_gather`: the gather's destination is packed
    /// (`out + r * cols`), so five taps into one `rows x 5 x hidden` buffer would need five
    /// separate destinations and a sixth launch to concatenate them. NEGATIVE CONTROL in the same
    /// test: `dst_stride` below `cols` overlaps the next row — every element still in range of the
    /// buffer, which is what makes it worth refusing rather than trusting.
    #[test]
    fn the_aux_tap_carries_two_strides_and_refuses_a_destination_that_overlaps() {
        let r = rules();
        // Tap 3 of 5 at b1s8: eight rows of 5120 out of a 5120-wide residual stream, into the
        // third column block of a 25600-wide concatenation. The DESTINATION BASE carries the
        // block offset; the stride carries the row pitch.
        let d = resolve(
            HeadOp::RowCopy { dst_stride: 25_600 },
            HeadShape {
                rows: 8,
                cols: 5120,
                src_stride: 5120,
                vocab: (0, 0),
            },
            &r,
        )
        .expect("the tap resolves");
        assert_eq!(d.kernel, "saber_row_copy_bf16");
        assert_eq!(d.grid, [8, 1, 1]);
        assert_eq!(d.n_launches, 1);
        assert_eq!(
            d.arg("src_stride").map(|s| s.value.clone()),
            Some(ArgValue::I64 { ival: 5120 })
        );
        assert_eq!(
            d.arg("dst_stride").map(|s| s.value.clone()),
            Some(ArgValue::I64 { ival: 25_600 })
        );
        for name in ["src", "out"] {
            assert_eq!(
                d.arg(name).map(|s| s.value.clone()),
                Some(ArgValue::Buffer {
                    buffer: name.to_owned()
                }),
                "{name}"
            );
        }
        // THE CONTROL: a destination stride under the row width.
        let e = resolve(
            HeadOp::RowCopy { dst_stride: 4096 },
            HeadShape {
                rows: 8,
                cols: 5120,
                src_stride: 5120,
                vocab: (0, 0),
            },
            &r,
        )
        .expect_err("a destination that overlaps is refused");
        assert!(e.to_string().contains("destination stride"), "{e}");
        // And the widen, at the selector's own shape: 7 rows of 16 candidates.
        let w = resolve(
            HeadOp::WidenI64 { dst_stride: 16 },
            HeadShape {
                rows: 7,
                cols: 16,
                src_stride: 16,
                vocab: (0, 0),
            },
            &r,
        )
        .expect("the widen resolves");
        assert_eq!(w.kernel, "saber_widen_i32_i64");
        assert_eq!(w.grid, [7, 1, 1]);
        assert!(resolve(
            HeadOp::WidenI64 { dst_stride: 8 },
            HeadShape {
                rows: 7,
                cols: 16,
                src_stride: 16,
                vocab: (0, 0)
            },
            &r
        )
        .is_err());
    }

    /// NEGATIVE CONTROL: the argmax's LDS width is compiled in, so a rules file that disagreed
    /// would launch a kernel whose shared arrays are the wrong size. It is refused at parse.
    #[test]
    fn a_rules_file_that_moves_the_block_width_is_refused() {
        let bad = PROD_RULES.replace("block = 256", "block = 512");
        let e = HeadRules::from_toml(&bad).expect_err("must refuse");
        assert!(e.to_string().contains("compiled for 256"), "{e}");
        // And one that drops an op the forward pass needs.
        let missing = PROD_RULES.replace("op = \"argmax\"", "op = \"argmax_disabled\"");
        assert!(HeadRules::from_toml(&missing).is_err());
        // The seam's width is compiled in twice over: the block it histograms with and the number
        // of candidates it writes. Either one moved in the rules is a refusal.
        let bad = PROD_RULES.replace("device_row_block = 256", "device_row_block = 512");
        let e = HeadRules::from_toml(&bad).expect_err("must refuse");
        assert!(e.to_string().contains("compiled for 256"), "{e}");
        let bad = PROD_RULES.replace("device_row_k = 64", "device_row_k = 32");
        let e = HeadRules::from_toml(&bad).expect_err("must refuse");
        assert!(e.to_string().contains("64 candidates per row"), "{e}");
    }

    /// The bf16 head's norm (N2): one workgroup per row at the rules' block, the epsilon carried
    /// as its bits, and a row wider than the kernel's register budget refused by name (the kernel
    /// would silently stop at `ARB_MAXG` groups per thread).
    #[test]
    fn the_bf16_heads_norm_resolves_and_refuses_a_row_it_would_truncate() {
        let r = rules();
        let eps = 1e-6f32;
        let op = HeadOp::AddRmsBf16 {
            eps_bits: eps.to_bits(),
        };
        let d = resolve(op, HeadShape::packed(64, 5120), &r).expect("hidden 5120 resolves");
        assert_eq!(d.kernel, "saber_add_rms_bf16");
        assert_eq!(
            (d.grid, d.block, d.n_launches),
            ([64, 1, 1], [256, 1, 1], 1)
        );
        let k = d.arg("k").expect("k");
        assert_eq!(k.value, ArgValue::I32 { ival: 5120 });
        assert_eq!(
            d.arg("eps").expect("eps").value,
            ArgValue::F32 { fval: eps }
        );
        // The budget is 4 groups of 8 per thread at 256 threads: 8192 fits, 8200 does not.
        resolve(op, HeadShape::packed(1, 8192), &r).expect("the budget's own edge");
        let e = resolve(op, HeadShape::packed(1, 8200), &r).expect_err("past the budget");
        assert!(e.to_string().contains("at most 8192"), "{e}");
        let e = resolve(op, HeadShape::packed(1, 5124), &r).expect_err("not 8-aligned");
        assert!(e.to_string().contains("8-aligned"), "{e}");
    }
}
