//! **What a BATCHED proposal asks `dflash::resolve` for**, at the shapes a batch of four would
//! use, as a census rather than as a claim.
//!
//! The ledger said from workstream S3 to S5 that "one proposal per request" had exactly one cause:
//! "the fused context-KV projection is a `bgemm_bf16` whose WMMA M tile stops at 16 rows, so a
//! batched proposal over `max_seqs * block` context rows is refused by the committed rules". S6's
//! census found it was three, not one — `bgemm_bf16`, `gemm_a8w8_blockscale` (21 of the proposal's
//! launches) and `draft_head_int2` — and closed the first of them.
//!
//! **All three are M-tiled now** (F33), and the shape of the fix is the same in each: the tile does
//! not change, `blockIdx.y` picks which sixteen rows a workgroup owns, and everything inside is the
//! committed kernel on those rows — the same k order into the same accumulator. So the census this
//! file keeps is no longer a list of refusals but the other statement: **at four requests every op
//! of the proposal resolves, with the batch in its grid.**
//!
//! Each M-tiled op is a SECOND SYMBOL rather than a widened first one, so the shape the whole
//! proposal was gated at keeps the code object it was gated with. The untiled symbols therefore
//! still refuse a taller M BY NAME, and each refusal names its M-tiled op — which is the other
//! half of this file.

use saber_kernels::dflash::{self, DflashOp, DflashRules, DflashShape};

/// The drafter's own geometry at TP=1, which is where these shapes come from.
const H: u32 = 5120;
const VLOC: u32 = 248_320;

/// Four requests at prod's block of 8: 32 context rows, 32 query rows, 28 selector positions
/// padded to 32, and a head over `_pow2_at_least(4 * 7)` = 32 sampled rows.
const M4: u32 = 32;

/// `max_seqs` 32 × the block of 8 — the widest batched proposal this engine can plan, and the
/// `max_m` every one of the three M-tiled ops declares.
const MAX_M: u32 = 256;

fn rules() -> DflashRules {
    DflashRules::prod_dflash().expect("the committed dflash rules")
}

/// Every site of the proposal at four requests, with the op a batched caller asks for.
fn batched_sites() -> Vec<(&'static str, DflashOp, DflashShape)> {
    vec![
        // The context store's fused KV projection, the ten conv kernel projections and the
        // selector's hidden projection: one op, three sites.
        (
            "context_kv.kv_projection",
            DflashOp::BgemmBf16M32,
            DflashShape::new(M4, 10240, H),
        ),
        (
            "layer.0.attention_conv.kernel_projection",
            DflashOp::BgemmBf16M32,
            DflashShape::new(M4, 1280, H),
        ),
        // Every blockscale GEMM in the proposal: the aux projection at the context rows and the
        // four per layer at the query rows. Twenty-one launches, one op.
        (
            "aux_hidden_projection",
            DflashOp::GemmBlockscaleMt,
            DflashShape::new(M4, H, 5 * H),
        ),
        (
            "layer.0.attn.qkv",
            DflashOp::GemmBlockscaleMt,
            DflashShape::new(M4, 3072, H),
        ),
        (
            "layer.0.attn.o_proj",
            DflashOp::GemmBlockscaleMt,
            DflashShape::new(M4, H, 2048),
        ),
        (
            "layer.0.mlp.gate_up",
            DflashOp::GemmBlockscaleMt,
            DflashShape::new(M4, 17408, H),
        ),
        (
            "layer.0.mlp.down",
            DflashOp::GemmBlockscaleMt,
            DflashShape::new(M4, H, 8704),
        ),
        // The int2 head over this rank's vocabulary shard.
        (
            "head.int2",
            DflashOp::DraftHeadInt2M32,
            DflashShape::new(M4, VLOC, H),
        ),
        // ... and the fifteen that resolved with the batch in their grids all along.
        (
            "aux_hidden_projection.quant",
            DflashOp::ActQuant,
            DflashShape::new(M4, 5 * H, 5 * H),
        ),
        (
            "context_kv.hidden_norm",
            DflashOp::CtxRmsNorm,
            DflashShape::ctx_norm(M4, H, M4),
        ),
        (
            "layer.0.attention_conv.prepare",
            DflashOp::GroupedConvQuant,
            DflashShape::conv(M4, H, H / 16, 8, 0),
        ),
        (
            "layer.0.attention_conv.finish",
            DflashOp::ConvAddRmsOpen,
            DflashShape::conv(M4, H, H / 16, 8, 1),
        ),
        (
            "layer.0.mlp.silu_quant",
            DflashOp::SiluQuant,
            DflashShape::new(M4, 8704, 17408),
        ),
        (
            "layer.0.input_layernorm",
            DflashOp::RmsNorm,
            DflashShape::new(M4, H, H),
        ),
        (
            "layer.0.attn.qk_norm_rope",
            DflashOp::QkNormRope,
            DflashShape::qk_rope(M4, 32, 8, 128),
        ),
        (
            "layer.0.attn.kv_append",
            DflashOp::KvAppendFp8,
            DflashShape::new(M4, 8, 128),
        ),
        (
            "head.row_pad",
            DflashOp::HeadRowPad,
            DflashShape::rerank(M4, H, H, 28),
        ),
        (
            "head.group_sum",
            DflashOp::HeadGroupSum,
            DflashShape::new(M4, H, H),
        ),
        (
            "head.rerank",
            DflashOp::RerankExactBf16,
            DflashShape::rerank(M4, VLOC, H, 80),
        ),
    ]
}

/// **THE CENSUS: at four requests, nothing refuses.**
///
/// Asked of `dflash::resolve` at `reqs = 4`, every site of the proposal resolves and every one of
/// them has the batch in its grid rather than in a refusal. The count is asserted so that a site
/// deleted from the list is a failure rather than a shorter census.
#[test]
fn a_batched_proposal_resolves_at_four_requests_with_no_refusals() {
    let r = rules();
    let sites = batched_sites();
    assert_eq!(sites.len(), 19, "the census lost a site");
    let mut refused: Vec<String> = Vec::new();
    for (site, op, shape) in &sites {
        match dflash::resolve(*op, *shape, &r) {
            Ok(d) => assert!(
                d.grid.iter().product::<u32>() > 0,
                "{site}: a legal launch with an empty grid"
            ),
            Err(e) => refused.push(format!("{site} ({}): {e}", op.as_str())),
        }
    }
    assert!(
        refused.is_empty(),
        "a batched proposal at 4 requests is refused by {} op(s):\n  {}",
        refused.len(),
        refused.join("\n  ")
    );
}

/// The three M-tiled ops put the batch on the grid's **y** axis and nowhere else: the x extent and
/// the block are what one 16-row launch already used, and the y extent is the tile count.
#[test]
fn the_batch_is_the_grids_y_extent_and_the_tile_is_unchanged() {
    let r = rules();
    let cases: [(&str, DflashOp, DflashOp, DflashShape, [u32; 3]); 3] = [
        (
            "context_kv.kv_projection",
            DflashOp::BgemmBf16,
            DflashOp::BgemmBf16Mt,
            DflashShape::new(M4, 10240, H),
            [10240 / 16 / 4, M4 / 16, 1],
        ),
        (
            "layer.0.mlp.gate_up",
            DflashOp::GemmBlockscale,
            DflashOp::GemmBlockscaleMt,
            DflashShape::new(M4, 17408, H),
            [17408 / 128, M4 / 16, 1],
        ),
        (
            "head.int2",
            DflashOp::DraftHeadInt2,
            DflashOp::DraftHeadInt2Mt,
            DflashShape::new(M4, VLOC, H),
            [VLOC / 64, M4 / 16, 1],
        ),
    ];
    for (site, plain, tiled, wide, grid) in cases {
        // The untiled symbol still refuses this shape, BY NAME and naming its M-tiled op.
        let e = dflash::resolve(plain, wide, &r)
            .expect_err(&format!(
                "{site}: the untiled symbol must still refuse M = {M4}"
            ))
            .to_string();
        assert!(
            e.contains("16"),
            "{site}: the refusal must name the tile it stops at, and it says {e}"
        );
        assert!(
            e.contains(tiled.as_str()),
            "{site}: the refusal must name the op that serves the batch, and it says {e}"
        );

        let d = dflash::resolve(tiled, wide, &r).unwrap_or_else(|err| panic!("{site}: {err}"));
        assert_eq!(d.grid, grid, "{site}: the batch belongs on the y axis");

        // ... and a batch past what any plan may ask for is a refusal by name rather than a grid
        // nobody sized. `max_m` is `max_seqs` 32 x the DFlash2 block of 8 for all three.
        let past = DflashShape {
            m: MAX_M + 16,
            ..wide
        };
        let e = dflash::resolve(tiled, past, &r)
            .expect_err(&format!("{site}: past max_m"))
            .to_string();
        assert!(e.contains("max_m"), "{site}: {e}");
    }
}

/// **AT ONE TILE THE TWO ARE THE SAME LAUNCH** — the same grid, the same block, the same kernargs,
/// differing only in which symbol they name. That is what makes "bit-identical on each 16-row
/// slice" a claim about the SOURCE (one tile, one k order, one accumulator) as well as a gate: the
/// tiled kernel's `blockIdx.y` extent is 1 there, so it *is* the untiled kernel.
#[test]
fn every_m_tiled_op_is_the_untiled_one_at_a_single_tile() {
    let r = rules();
    let cases: [(&str, DflashOp, DflashOp, DflashShape, &str); 3] = [
        (
            "bgemm_bf16",
            DflashOp::BgemmBf16,
            DflashOp::BgemmBf16Mt,
            DflashShape::new(8, 1280, H),
            "dflash_bgemm_bf16_mtn16",
        ),
        (
            "gemm_a8w8_blockscale",
            DflashOp::GemmBlockscale,
            DflashOp::GemmBlockscaleMt,
            DflashShape::new(8, 3072, H),
            "dflash_gemm_a8w8_bs128_mtn128k128",
        ),
        (
            "draft_head_int2",
            DflashOp::DraftHeadInt2,
            DflashOp::DraftHeadInt2Mt,
            DflashShape::new(16, VLOC, H),
            "dflash_draft_head_int2_mtn64g128",
        ),
    ];
    for (name, plain, tiled, narrow, symbol) in cases {
        let a = dflash::resolve(plain, narrow, &r).unwrap_or_else(|e| panic!("{name}: {e}"));
        let b = dflash::resolve(tiled, narrow, &r).unwrap_or_else(|e| panic!("{name}_mt: {e}"));
        // ws-h43: the traced head runs two 64-column blocks a workgroup (`wg_blocks = 2`, x staged
        // once in LDS) and the row slice one, so the head's GRID differs by exactly that factor;
        // its arguments do not.
        if name == "draft_head_int2" {
            assert_eq!(
                a.grid[0],
                b.grid[0].div_ceil(2),
                "{name}: two blocks a workgroup"
            );
            assert_eq!(a.block[0], 2 * b.block[0], "{name}: four waves");
        } else {
            assert_eq!(a.grid, b.grid, "{name}: one M tile, one grid");
            assert_eq!(a.block, b.block, "{name}");
        }
        assert_eq!(a.args, b.args, "{name}");
        assert_eq!(a.plugin, b.plugin, "{name}");
        assert_eq!(a.kernarg_bytes, b.kernarg_bytes, "{name}");
        assert_ne!(
            a.kernel, b.kernel,
            "{name}: two symbols, and the route names one"
        );
        assert_eq!(b.kernel, symbol, "{name}");
    }
}

/// The head's M-tiled op takes WHOLE TILES and nothing else, and the refusal says why.
///
/// Prod pads the draft rows with `_pow2_at_least`, allocates `y` / `bmax` / `bidx` at that padded
/// height and the emit writes all sixteen rows of its tile — so a partial tile would run past the
/// buffers rather than compute sixteen rows of which some are ignored. Every M the padding
/// produces is a multiple of 16; an M that is not is a caller bug.
#[test]
fn the_m_tiled_head_refuses_a_partial_tile() {
    let r = rules();
    let e = dflash::resolve(DflashOp::DraftHeadInt2Mt, DflashShape::new(40, VLOC, H), &r)
        .expect_err("40 rows is two and a half tiles")
        .to_string();
    assert!(e.contains("whole number"), "{e}");
    assert!(e.contains("16"), "{e}");

    // The CONTROL: the same M rounded up to the next whole tile resolves, so the refusal above is
    // the partial tile and not the shape.
    let d = dflash::resolve(DflashOp::DraftHeadInt2Mt, DflashShape::new(48, VLOC, H), &r)
        .expect("three whole tiles");
    assert_eq!(d.grid, [VLOC / 64, 3, 1]);
}

/// **THE BAND IS WHAT THE RULES PICK ABOVE THE TILE, and the row slice is the fallback they name.**
///
/// F33 measured the row slice at M = 32 and it LOSES: the head's is 868.54 µs against 743.20 for two
/// launches of the 16-row symbol, because the two m tiles of one column block are two workgroups and
/// each reads the whole weight; the projection's is 334.38 against hipBLASLt's 184.56 for the same
/// reason (two weight passes, and one pass is the 165 µs floor). Both bands put the second m tile in
/// the SAME workgroup against the same weight fragment, so this is the op a batched caller asks for
/// and the y extent is `M / 32`.
#[test]
fn the_thirty_two_row_band_is_one_workgroup_per_thirty_two_rows() {
    let r = rules();
    let cases: [(&str, DflashOp, DflashShape, [u32; 3], &str); 2] = [
        (
            "head.int2",
            DflashOp::DraftHeadInt2M32,
            DflashShape::new(M4, VLOC, H),
            [(VLOC / 64).div_ceil(2), M4 / 32, 1],
            "dflash_draft_head_int2_m32n64x2g128_lx",
        ),
        (
            "context_kv.kv_projection",
            DflashOp::BgemmBf16M32,
            DflashShape::new(M4, 10240, H),
            [10240 / 16 / 4, M4 / 32, 1],
            "dflash_bgemm_bf16_m32n16",
        ),
    ];
    for (site, band, wide, grid, symbol) in cases {
        let d = dflash::resolve(band, wide, &r).unwrap_or_else(|e| panic!("{site}: {e}"));
        assert_eq!(d.grid, grid, "{site}: the band is 32 rows of the batch");
        assert_eq!(d.kernel, symbol, "{site}");
        // ... and the widest batch a plan may ask for is eight of them.
        let d = dflash::resolve(band, DflashShape { m: MAX_M, ..wide }, &r)
            .unwrap_or_else(|e| panic!("{site} at max_m: {e}"));
        assert_eq!(d.grid[1], MAX_M / 32, "{site}");
        let e = dflash::resolve(
            band,
            DflashShape {
                m: MAX_M + 32,
                ..wide
            },
            &r,
        )
        .expect_err(&format!("{site}: past max_m"))
        .to_string();
        assert!(e.contains("max_m"), "{site}: {e}");
    }

    // The 16-row symbols refuse a taller M naming the BAND first and the row slice as the fallback,
    // because which of the two serves an M depends on whether it is whole bands.
    for (site, plain, wide, band_op, slice_op) in [
        (
            "head.int2",
            DflashOp::DraftHeadInt2,
            DflashShape::new(M4, VLOC, H),
            "draft_head_int2_m32",
            "draft_head_int2_mt",
        ),
        (
            "context_kv.kv_projection",
            DflashOp::BgemmBf16,
            DflashShape::new(M4, 10240, H),
            "bgemm_bf16_m32",
            "bgemm_bf16_mt",
        ),
    ] {
        let e = dflash::resolve(plain, wide, &r)
            .expect_err(&format!("{site}: the 16-row symbol must refuse M = {M4}"))
            .to_string();
        assert!(e.contains(band_op), "{site}: {e}");
        assert!(e.contains(slice_op), "{site}: {e}");
    }
}

/// The head's band refuses an M that is whole 16-row TILES and not whole 32-row BANDS, and names
/// the row slice that serves it — which is what "the fallback the rules name" means as a test.
#[test]
fn the_band_refuses_a_partial_band_and_names_the_row_slice() {
    let r = rules();
    let e = dflash::resolve(
        DflashOp::DraftHeadInt2M32,
        DflashShape::new(48, VLOC, H),
        &r,
    )
    .expect_err("48 rows is one and a half bands")
    .to_string();
    assert!(e.contains("whole number"), "{e}");
    assert!(e.contains("draft_head_int2_mt"), "{e}");

    // The CONTROL, and it is the point of the pair: the SAME 48 rows resolve under the row slice,
    // so the refusal above is the band's geometry and not the shape.
    let d = dflash::resolve(DflashOp::DraftHeadInt2Mt, DflashShape::new(48, VLOC, H), &r)
        .expect("three whole 16-row tiles");
    assert_eq!(d.grid, [VLOC / 64, 3, 1]);
    assert_eq!(d.kernel, "dflash_draft_head_int2_mtn64g128");

    // The projection's band CLAMPS its rows rather than predicating them, exactly as its 16-row
    // tile does, so a partial last band is legal there and the rows it produces are not stored.
    let d = dflash::resolve(DflashOp::BgemmBf16M32, DflashShape::new(48, 10240, H), &r)
        .expect("48 rows is two bands, the last one half full");
    assert_eq!(d.grid, [10240 / 16 / 4, 2, 1]);
}

/// The blockscale GEMM's M tile clamps its A rows rather than predicating them, so a partial last
/// tile is legal there and the rows it produces are simply not stored — which is the same thing
/// the untiled symbol already does for an M below 16. Stated as a test so the asymmetry with the
/// head is a decision rather than an oversight.
#[test]
fn the_m_tiled_blockscale_gemm_takes_a_partial_last_tile() {
    let r = rules();
    let d = dflash::resolve(
        DflashOp::GemmBlockscaleMt,
        DflashShape::new(40, 3072, H),
        &r,
    )
    .expect("40 rows is three tiles, the last one partial");
    assert_eq!(d.grid, [3072 / 128, 3, 1]);
}
