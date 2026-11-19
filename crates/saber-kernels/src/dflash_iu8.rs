//! MACaQ (2026-11-18): the W4A8-iu8 DFlash2 drafter's kernels (pack-driven: only an iu8 drafter pack routes here).
//!
//! An iu8 drafter pack (`.wi4` / `.wi4s` / `.t` records + the manifest tag [`FORMAT`]) runs every drafter W8A8
//! blockscale GEMM -- the aux projection, and each layer's `attn.qkv`, `attn.o_proj`, `mlp.gate_up`, `mlp.down` --
//! as an iu8 W4 GEMM on the target's existing bodies ([`crate::macaq_iu8::resolve`], `WKind::I4`: per-row int4
//! weights, per-token int8 activations, int32 accumulation, `y = bf16((f32(acc) * as[m]) * (ws[n] / 16))`), and
//! each of their four activation producers as its per-token INT8 twin (`hip/dflash_hadi8.hip`, one launch a site,
//! as the parent): the parent's pre-quant values, `* t`, the block-32 Hadamard, absmax / 127 over the row, the
//! A-tiled store. Nothing else in the drafter moves: the context-KV path, the kernel projections, the norms, the
//! attention, the int2 head, the rerank and the selector keep their kernels and their bytes.

use crate::abi;
use crate::rules::{ArgValue, LaunchDescriptor, Slot};
use crate::{Error, Result};

/// The producers' code object.
pub const PLUGIN: &str = "dflash_hadi8";
/// The drafter pack's manifest tag (`plan.descriptor.transforms`). An engine that does not know it refuses.
pub const FORMAT: &str = "macaq_iu8d/1 i4row a8tok had32 draft";
/// Every symbol the code object carries (the cells [`cell`] can name).
pub const SYMBOLS: &[&str] = &[
    "dflash_hadi8_act_t256g1",
    "dflash_hadi8_act_t256g2",
    "dflash_hadi8_act_t256g3",
    "dflash_hadi8_act_t256g4",
    "dflash_hadi8_act_t256g5",
    "dflash_hadi8_act_t512g7",
    "dflash_hadi8_conv_t256g3",
    "dflash_hadi8_silu_t256g5",
    "dflash_hadi8_silu_t512g5",
];

/// Which producer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Producer {
    /// The parent `dflash_act_quant_g128[_recip]`: v = f32(x).
    Act,
    /// The parent `dflash_grouped_conv_quant_t2g16`: v = bf16(conv).
    Conv,
    /// The parent `dflash_silu_quant_g128`: v = silu(gate) * up (f32).
    Silu,
}

impl Producer {
    fn tag(self) -> &'static str {
        match self {
            Producer::Act => "act",
            Producer::Conv => "conv",
            Producer::Silu => "silu",
        }
    }
}

/// The cell a producer over rows of `k` values runs: (threads, groups a thread). T 256 while every thread holds at
/// most 5 8-element groups, else T 512.
#[must_use]
pub fn cell(k: u32) -> (u32, u32) {
    let nv = k / 8;
    let g256 = nv.div_ceil(256);
    if g256 <= 5 {
        (256, g256)
    } else {
        (512, nv.div_ceil(512))
    }
}

/// The symbol a producer over `k`-wide rows runs, refused when the code object does not carry it.
///
/// # Errors
/// K not a multiple of 32 (the Hadamard's blocks; the quad lane map), or a cell the object does not carry.
pub fn symbol(p: Producer, k: u32) -> Result<String> {
    if k == 0 || k % 32 != 0 {
        return Err(Error::Rule(format!(
            "dflash_hadi8 {}: K={k} must be a positive multiple of 32 (the block-32 Hadamard)",
            p.tag()
        )));
    }
    let (t, g) = cell(k);
    let s = format!("dflash_hadi8_{}_t{t}g{g}", p.tag());
    if !SYMBOLS.contains(&s.as_str()) {
        return Err(Error::Rule(format!(
            "dflash_hadi8: no `{s}` for K={k} (the object carries {SYMBOLS:?})"
        )));
    }
    Ok(s)
}

fn descriptor(
    op: &str,
    kernel: String,
    rows: u32,
    threads: u32,
    ints: &[(&str, i64)],
) -> Result<LaunchDescriptor> {
    let sig = abi::signature_for(&kernel)
        .ok_or_else(|| Error::Rule(format!("{kernel}: no declared signature")))?;
    let (packed, bytes) = abi::pack(sig);
    let args = sig
        .iter()
        .zip(packed)
        .map(|(a, (offset, size))| Slot {
            name: a.name.to_owned(),
            offset,
            size,
            value: match ints.iter().find(|(n, _)| *n == a.name) {
                Some((_, v)) => ArgValue::I32 { ival: *v },
                None => ArgValue::Buffer {
                    buffer: a.name.to_owned(),
                },
            },
        })
        .collect();
    Ok(LaunchDescriptor {
        op: op.into(),
        plugin: PLUGIN.into(),
        kernel,
        grid: [rows, 1, 1],
        block: [threads, 1, 1],
        shared_bytes: 0,
        kernarg_bytes: bytes,
        args,
        workspaces: vec![],
        n_launches: 1,
    })
}

fn check_rows(rows: u32, what: &str) -> Result<()> {
    if rows == 0 {
        return Err(Error::Rule(format!(
            "dflash_hadi8 {what}: 0 rows is a caller bug"
        )));
    }
    Ok(())
}

/// `x` bf16 `[rows, k]` -> `q` A-tiled int8 (`ceil(rows/16) * 16 * k` bytes), `scales` f32 `[rows]`; `had_t` f32 `[k]`.
///
/// # Errors
/// [`symbol`]'s, and 0 rows.
pub fn resolve_act(rows: u32, k: u32) -> Result<LaunchDescriptor> {
    check_rows(rows, "act")?;
    let s = symbol(Producer::Act, k)?;
    descriptor(
        "act_quant_hadi8",
        s,
        rows,
        cell(k).0,
        &[("n", i64::from(k))],
    )
}

/// The conv `prepare` twin over `rows` query rows of `h` (`groups` conv groups, the drafter's power-of-two
/// `block_size`, `side` 0 = prepare).
///
/// # Errors
/// [`symbol`]'s, 0 rows, a block that is not a power of two.
pub fn resolve_conv(
    rows: u32,
    h: u32,
    groups: u32,
    block_size: u32,
    side: u32,
) -> Result<LaunchDescriptor> {
    check_rows(rows, "conv")?;
    if block_size == 0 || block_size & (block_size - 1) != 0 {
        return Err(Error::Rule(format!(
            "dflash_hadi8 conv: block {block_size} is not a power of two (the parent's `t & (block - 1)` ring)"
        )));
    }
    let s = symbol(Producer::Conv, h)?;
    descriptor(
        "grouped_conv_quant_hadi8",
        s,
        rows,
        cell(h).0,
        &[
            ("h", i64::from(h)),
            ("groups", i64::from(groups)),
            ("block_size", i64::from(block_size)),
            ("side", i64::from(side)),
        ],
    )
}

/// The silu twin: `gate_up` bf16 `[rows, 2n]` -> int8 A-tiled `[rows, n]` + `scales` `[rows]`.
///
/// # Errors
/// [`symbol`]'s, and 0 rows.
pub fn resolve_silu(rows: u32, n: u32) -> Result<LaunchDescriptor> {
    check_rows(rows, "silu")?;
    let s = symbol(Producer::Silu, n)?;
    descriptor(
        "silu_quant_hadi8",
        s,
        rows,
        cell(n).0,
        &[("n", i64::from(n))],
    )
}

/// Bytes an A-tiled int8 edge of `rows x k` takes (16-row tiles).
#[must_use]
pub fn tiled_bytes(rows: u64, k: u64) -> u64 {
    rows.div_ceil(16) * 16 * k
}

/// The drafter's iu8 GEMM shapes `(tp, name, N, K)` on one rank (the five linear sites; the aux
/// projection is replicated, qkv / gate_up column-parallel, o_proj / down row-parallel).
pub const DRAFT_SHAPES: &[(u32, &str, u32, u32)] = &[
    (2, "aux", 5120, 25_600),
    (2, "qkv", 3072, 5120),
    (2, "o", 5120, 2048),
    (2, "gate_up", 17_408, 5120),
    (2, "down", 5120, 8704),
    (1, "aux", 5120, 25_600),
    (1, "qkv", 6144, 5120),
    (1, "o", 5120, 4096),
    (1, "gate_up", 34_816, 5120),
    (1, "down", 5120, 17_408),
];

/// The rows a drafter iu8 GEMM runs at -- the backbone at `reqs x 8` (batches 1, 2, 3, 4, 8), the
/// context store at `reqs x width` (widths 1, 5, 8 and the tree's 10 / 9) and the wide / chunk-class stores
/// (256 / 8 rows).
pub const DRAFT_ROWS: &[u32] = &[
    1, 2, 3, 4, 5, 8, 9, 10, 15, 16, 18, 20, 24, 27, 30, 32, 36, 40, 64, 72, 80, 256,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// The cells the engine routes at every drafter GEMM shape and row count, printed for an on-card kernel check
    /// (one JSON object a line): `cargo test -p saber-kernels --lib dflash_iu8 -- --ignored --nocapture`.
    #[test]
    #[ignore = "prints the drafter cell list"]
    fn print_the_drafter_cells() {
        for &(tp, name, n, k) in DRAFT_SHAPES {
            for &m in DRAFT_ROWS {
                let c = crate::macaq_iu8::draft_cell(crate::macaq_iu8::WKind::I4, m, n, k).unwrap();
                println!(
                    "CELL {{\"tp\": {tp}, \"op\": \"{name}\", \"m\": {m}, \"n\": {n}, \"k\": {k}, \"plugin\": \"{}\", \"symbol\": \"{}\", \"grid\": [{}, {}, {}], \"threads\": {}, \"ks\": {}}}",
                    c.plugin, c.symbol, c.grid[0], c.grid[1], c.grid[2], c.threads, c.ks
                );
            }
        }
    }

    /// EXACTLY these drafter cells moved off the target's [`crate::macaq_iu8::cell`]: aux / qkv / o_proj /
    /// down at M 33..64 (both TPs) onto `macaq_iu8f_pf_w4_t1w4n2k128p`; every other drafter row (gate_up at every M,
    /// every op at M <= 32 and > 64) and every W8 cell routes as before, and the moved cell is a shipped symbol.
    #[test]
    fn the_drafter_rows_move_exactly_the_m33_64_small_n_cells() {
        use crate::macaq_iu8::{cell, draft_cell, WKind, FUSED_SYMBOLS};
        let mut moved = Vec::new();
        for &(tp, name, n, k) in DRAFT_SHAPES {
            for m in 1..=2048u32 {
                let (t, d) = (
                    cell(WKind::I4, m, n, k).unwrap(),
                    draft_cell(WKind::I4, m, n, k).unwrap(),
                );
                assert_eq!(
                    cell(WKind::I8, m, n, k).unwrap(),
                    draft_cell(WKind::I8, m, n, k).unwrap()
                );
                if t != d {
                    assert_eq!(d.symbol, "macaq_iu8f_pf_w4_t1w4n2k128p", "{tp} {name} {m}");
                    assert!(FUSED_SYMBOLS.contains(&d.symbol.as_str()));
                    assert_eq!((d.grid, d.threads, d.ks), ([n / 64, 1, 1], 256, 1));
                    assert!(t.symbol.starts_with("macaq_iu8_dec_w4_tm"), "{}", t.symbol);
                    moved.push((tp, name, m));
                }
            }
        }
        let mut want = Vec::new();
        for &(tp, name, _, _) in DRAFT_SHAPES {
            if name != "gate_up" {
                want.extend((33..=64).map(|m| (tp, name, m)));
            }
        }
        assert_eq!(moved, want);
        assert_eq!(moved.len(), 8 * 32);
    }

    #[test]
    fn every_drafter_k_has_a_cell_and_every_symbol_is_reachable() {
        // TP=2 / TP=1: aux 25,600; o_proj 2,048 / 4,096; conv 5,120; silu 8,704 / 17,408.
        for (p, k, want) in [
            (Producer::Act, 25_600, "dflash_hadi8_act_t512g7"),
            (Producer::Act, 2_048, "dflash_hadi8_act_t256g1"),
            (Producer::Act, 4_096, "dflash_hadi8_act_t256g2"),
            (Producer::Conv, 5_120, "dflash_hadi8_conv_t256g3"),
            (Producer::Silu, 8_704, "dflash_hadi8_silu_t256g5"),
            (Producer::Silu, 17_408, "dflash_hadi8_silu_t512g5"),
        ] {
            assert_eq!(symbol(p, k).unwrap(), want);
            let (t, g) = cell(k);
            assert!(t * g * 8 >= k && t * (g - 1) * 8 < k, "{k}: {t} x {g}");
        }
        assert!(symbol(Producer::Conv, 1000).is_err());
        assert!(symbol(Producer::Silu, 5_120).is_err());
        let d = resolve_act(10, 25_600).unwrap();
        assert_eq!((d.grid, d.block), ([10, 1, 1], [512, 1, 1]));
        assert!(d.workspaces.is_empty() && d.n_launches == 1);
        assert!(resolve_conv(8, 5_120, 320, 6, 0).is_err());
        assert_eq!(tiled_bytes(8, 5_120), 16 * 5_120);
        assert_eq!(tiled_bytes(17, 32), 32 * 32);
    }
}
