//! Flash-Next (qwen4_exp) hyper-connections and the bf16 dense GEMM family (workstream FN5).
//!
//! The launch POLICY for `hip/flashnext_hc.hip` (code object `flashnext_hc`), read from
//! `rules/flashnext_gfx1201.toml`: which HC variant a mix of M rows takes, the geometry and scratch
//! of each launch, the HC launch law (two launches per mix, four per decoder layer), and the bf16
//! GEMM route per Flash-Next shape and row band. No engine binds these yet (F2 builds the layers);
//! the tables and their host laws are what the card gate (`tools/capture/fn5/fn5_gate.py`) and the
//! next rounds read.

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

pub const PROD_RULES_TEXT: &str = include_str!("../rules/flashnext_gfx1201.toml");
pub const PLUGIN: &str = "flashnext_hc";

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct HcModel {
    pub hc_count: u32,
    pub hidden: u32,
    pub lowrank: u32,
    pub eps: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct HcBand {
    pub max_m: u32,
    pub sps: u32,
    pub down_mf: u32,
    pub up_mf: u32,
    /// FN11: the v2 down (`hc.down_v2` cfg) this band routes instead of `fn_hc_down_*_mf{down_mf}`
    /// -- byte-identical to the shipped down of the same `sps` (gated), so the class is unchanged.
    #[serde(default)]
    pub down_v2: Option<String>,
    /// FN11: the v2 up (`hc.up_v2` cfg) this band routes instead of `fn_hc_up_*_mf{up_mf}`.
    #[serde(default)]
    pub up_v2: Option<String>,
}

/// FN11 v2 down geometry: `fn_hc_down_<mode>_v2_<cfg>`, grid `(4 sps, ceil(M / 16 mf), groups)`,
/// `waves` waves each owning `tiles` of the 21 column tiles; the A slab chunk `kc` must divide the
/// K slice (`2560 / sps`). Counters: one per (row block, column group).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct HcDownV2 {
    pub cfg: String,
    pub mf: u32,
    pub waves: u32,
    pub tiles: u32,
    pub kc: u32,
    pub groups: u32,
}

/// FN11 v2 up geometry: `fn_hc_up_<kind>_v2_<cfg>`, grid `(grid_x, ceil(M / 16 mf))`, `waves`
/// waves (4 unless the row says; FN21's prefill `p*` bodies run 8 over grid_x 20); the partial
/// amax is `[grid_x, M]`.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct HcUpV2 {
    pub cfg: String,
    pub mf: u32,
    pub grid_x: u32,
    #[serde(default = "four_waves")]
    pub waves: u32,
}

fn four_waves() -> u32 {
    4
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct HcPolicy {
    pub launches_per_mix: u32,
    pub launches_per_layer: u32,
    pub down_waves: u32,
    pub up_waves: u32,
    pub partial_stride: u32,
    pub sps_allowed: Vec<u32>,
    pub band: Vec<HcBand>,
    #[serde(default)]
    pub down_v2: Vec<HcDownV2>,
    #[serde(default)]
    pub up_v2: Vec<HcUpV2>,
    /// FN37 (the producer-rr law): the TP = 2 exact reduce that produces a block output also
    /// writes the NEXT mix's combined stream x' and its norm factor rr (`fn_ar_exact_hcrr_bf16`,
    /// the down's own B2 + B1 text), and that mix's down runs its norm-free twin
    /// (`fn_hc_down_{combine,final}rr_v2_*`) -- the same bytes one launch earlier. Not a rules-table
    /// row: the boot sets it from `--fn-hc-rr-producer` / `--no-fn-hc-rr-producer`.
    #[serde(default)]
    pub rr_producer: bool,
    /// FN85 (the prologue reduce): at a TP = 2 target decode / verify form the exact reduce of a
    /// block output whose consumer is a decode-band combine down runs inside that down's prologue
    /// (`fn_hc_down_combine_v2_<cfg>_rp`). Not a rules-table row: the boot sets it from
    /// `--fn-ar-prologue` (every emitter starts from it).
    #[serde(default)]
    pub ar_prologue: bool,
    /// FN86b (launch fusion): the families whose fused symbol runs at their sites (a bit per
    /// family, [`fuse86`]). Not a rules-table row: the boot sets it from `--fn-fuse` (every
    /// emitter starts from it); 0 = every shipped chain (the default and the control).
    #[serde(default)]
    pub fuse86: u32,
    /// FN111 (the roofline re-tiles): the re-tiled cells whose ADDED symbol runs at their sites
    /// (a bit per tile, [`tiles111`]). Not a rules-table row: the boot sets it from `--fn-tiles`;
    /// 0 = every shipped cell (`default`, the control).
    #[serde(default)]
    pub tiles111: u32,
    /// FN122b (the prefill tiles): the prefill families whose ADDED symbols run at their sites (a
    /// bit per family, [`prefill122`]). Not a rules-table row: the boot sets it from
    /// `--fn-prefill-tiles`; 0 = every shipped cell (`default`, the control).
    #[serde(default)]
    pub prefill122: u32,
    /// FN128 (the QSA selection at the decode / verify forms): the selection launches whose ADDED
    /// symbol runs (a bit per kernel, [`sel128`]). Not a rules-table row: the boot sets it from
    /// `--fn-sel128`; 0 = every shipped symbol (`default`, the control).
    #[serde(default)]
    pub sel128: u32,
    /// FN131 (the decode HC down re-tiled by phase): the fp8-mixer decode downs whose ADDED symbol
    /// runs ([`down131`]). Not a rules-table row: the boot sets it from `--fn-hc-down`; 0 = every
    /// shipped down (`default`, the control).
    #[serde(default)]
    pub down131: u32,
    /// FN141 (the drained-ring twins): the families whose ADDED symbol runs ([`k141`]). Not a
    /// rules-table row: the boot sets it from `--fn-k141`; 0 = every shipped symbol (the control).
    #[serde(default)]
    pub k141: u32,
    /// FN163 (the parallel route / re-tiled front): the bands whose ADDED symbol runs
    /// ([`route163`]). Not a rules-table row: the boot sets it from `--fn-route163`; 0 = every
    /// shipped symbol (`none`, the control).
    #[serde(default)]
    pub route163: u32,
}

/// FN131: the fp8-mixer HC DOWN at the decode forms RE-TILED BY PHASE (LEDGER FN131). The NEW object
/// `flashnext_hc131` (generated by `tools/gen_fn131_hc131.py`) holds `fn131_down_<mode>_v2_<cfg>_h8`
/// for the two routed decode cfgs: the parent's arguments and grid, (NW + NP) x 32 threads -- NW
/// tile waves issue the weight ring at entry while NP phase waves run the stream norm (a row a
/// wave) and form the whole K slice's A slab; one LDS-only barrier; a branch-free ring; the last
/// arrival's slice sum in one round. Byte-equal to the parent by the card-1 oracle
/// (`tools/capture/fn131/fn131_probe.py`; the dequant's RNE drops the NaN select: equal for every
/// record without a NaN SCALE -- the hc8 records' scales are amax / 448 of finite rows).
pub mod down131 {
    /// split: the decode downs on the phase-split body.
    pub const SPLIT: u32 = 1 << 0;
    pub const ALL: u32 = SPLIT;
    pub const NAMES: [&str; 1] = ["split"];
    /// The routed cfgs: `(cfg, sps, parent waves, added waves)` -- the cfg's band (rules
    /// `hc.band`: `m1w3t1d` = M <= 1 at sps 4, `m1w7t1` = M 2..64 at sps 2) and the phase waves
    /// (NP) the generator built (`tools/gen_fn131_hc131.py` SHIP). MEASURED (LEDGER FN131).
    pub const ROUTE: [(&str, u32, u32, u32); 2] = [("m1w3t1d", 4, 3, 4), ("m1w7t1", 2, 7, 8)];
    pub const MODES: [&str; 4] = ["mix", "expand", "combine", "final"];

    /// The added symbol + its threads for a planned fp8-mixer down `sym`
    /// (`fn_hc_down_<mode>_v2_<cfg>_h8`) at `sps` slices; `None` = keep the shipped down (another
    /// cfg / mode / sps, an rr / rp twin).
    #[must_use]
    pub fn route(sym: &str, sps: u32) -> Option<(String, u32)> {
        let rest = sym.strip_prefix("fn_hc_down_")?.strip_suffix("_h8")?;
        let (mode, cfg) = rest.split_once("_v2_")?;
        if !MODES.contains(&mode) {
            return None;
        }
        let r = ROUTE.iter().find(|r| r.0 == cfg && r.1 == sps)?;
        Some((format!("fn131_down_{mode}_v2_{cfg}_h8"), 32 * (r.2 + r.3)))
    }

    /// The shipped parent of an added symbol and the parent's threads (its class: byte-equal by
    /// the oracle; the phase waves are not a class axis).
    #[must_use]
    pub fn parent(sym: &str) -> Option<(String, u32)> {
        let rest = sym.strip_prefix("fn131_down_")?.strip_suffix("_h8")?;
        let (mode, cfg) = rest.split_once("_v2_")?;
        let r = ROUTE.iter().find(|r| r.0 == cfg)?;
        MODES
            .contains(&mode)
            .then(|| (format!("fn_hc_down_{mode}_v2_{cfg}_h8"), 32 * r.2))
    }

    /// `--fn-hc-down default|none|all|split`.
    ///
    /// # Errors
    /// An unknown name.
    pub fn parse(s: &str) -> Result<u32, String> {
        match s.trim() {
            "default" | "none" | "" => return Ok(0),
            "all" => return Ok(ALL),
            _ => {}
        }
        let mut m = 0;
        for t in s.split(',').map(str::trim) {
            let i = NAMES.iter().position(|n| *n == t).ok_or_else(|| {
                format!("--fn-hc-down: unknown family {t:?} (split, default, all)")
            })?;
            m |= 1 << i;
        }
        Ok(m)
    }

    #[must_use]
    pub fn names(m: u32) -> String {
        if m == 0 {
            return "default".into();
        }
        NAMES
            .iter()
            .enumerate()
            .filter(|(i, _)| m & (1 << i) != 0)
            .map(|(_, n)| *n)
            .collect::<Vec<_>>()
            .join(",")
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn fn131_the_route_takes_the_decode_downs_only_and_parent_inverts_it() {
            for (cfg, sps, nw, np) in ROUTE {
                for mode in MODES {
                    let ship = format!("fn_hc_down_{mode}_v2_{cfg}_h8");
                    let (new, thr) = route(&ship, sps).unwrap();
                    assert_eq!(thr, 32 * (nw + np));
                    assert_eq!(parent(&new), Some((ship.clone(), 32 * nw)));
                    // another sps is another class: never routed
                    assert!(route(&ship, sps * 2).is_none());
                }
            }
            // NEGATIVE CONTROLS: the rr / rp twins, the bf16 (non-h8) down, the prefill cfg, the
            // DPP control's cfg and the up keep the shipped symbol
            for s in [
                "fn_hc_down_combinerr_v2_m1w3t1_h8",
                "fn_hc_down_combine_v2_m1w3t1d",
                "fn_hc_down_combine_v2_m1w3t1d_rp",
                "fn_hc_down_combine_v2_m2w21t1_h8",
                "fn_hc_down_combine_v2_m1w3t1_h8",
                "fn_hc_up_q8_v2_s1d_h8",
            ] {
                assert!(route(s, 4).is_none() && route(s, 2).is_none(), "{s}");
            }
            assert_eq!(parse("default"), Ok(0));
            assert_eq!(parse("all"), Ok(SPLIT));
            assert_eq!(parse("split"), Ok(SPLIT));
            assert!(parse("t1").is_err());
            assert_eq!(names(SPLIT), "split");
            assert_eq!(names(0), "default");
        }
    }
}

/// FN122b: the PREFILL tiles (LEDGER FN122 / FN122b). Each family is a set of ADDED symbols in a
/// NEW code object whose every output byte equals the shipped prefill cell it replaces (card twin,
/// `tools/capture/fn122/fn122_twin.py`); the route table is the twin's measured winners.
pub mod prefill122 {
    /// hc: the fp8-mixer HC down at prefill rows on `flashnext_hc8p`'s geometry (the shipped
    /// `fn_hc8_down2_body` at another row block / waves / A-slab chunk; sps 1 = the class, kept).
    pub const HC: u32 = 1 << 0;
    /// dq (FN122c): at >= [`DQ_MIN_ROWS`] the mix's two fp8 matrices are dequantized ONCE a mix
    /// (`fn122_hc8_dequant`: bf16(code x scale), FN108's `_h8` law) and the BF16 bodies run on
    /// them -- the down on `flashnext_hc8p`'s `m8w21t1k64qb` (the q down on the shipped bf16
    /// body), the up on the shipped `p4s` (FN21's; its `_h8` twin spilled, FN108). A producer-rr
    /// mix keeps `hc`.
    pub const DQ: u32 = 1 << 1;
    /// dqb (FN122d): the dq family from [`DQB_MIN_ROWS`] (the band below 1,024 rows; needs `dq`).
    pub const DQB: u32 = 1 << 2;
    /// qsa (FN124): the fp8-KV QSA gather prefill on `flashnext_qsa124`'s body, which skips a
    /// 16-key sub-tile every row of a wave masks (the identity: a fully masked sub-tile adds +0 to
    /// every accumulator and cannot fire the rescale). The bf16-KV path keeps the shipped symbol.
    pub const QSA: u32 = 1 << 3;
    /// qsatwin (FN124, a DIAGNOSTIC, never in `all`): the shipped gather runs as today and a
    /// qsa124 body runs beside it into scratch (the lever with `qsa`, the CONTROL without);
    /// `fn124_cmp_u32` counts every differing output word (end line `fnqsa124 rank<r>:`) -- the
    /// in-situ twin over every real chunk and row.
    pub const QSATWIN: u32 = 1 << 4;
    pub const ALL: u32 = HC | DQ | DQB | QSA;
    pub const NAMES: [&str; 5] = ["hc", "dq", "dqb", "qsa", "qsatwin"];
    /// The qsa symbols (`flashnext_qsa124`): the lever, the twin's control, the comparator.
    pub const QSA_SYMBOL: &str = "fn124_qsa_attn_prefill_h256_gqa12_fp8kv_skip";
    pub const QSA_CONTROL: &str = "fn124_qsa_attn_prefill_h256_gqa12_fp8kv_skipctl";
    pub const QSA_CMP: &str = "fn124_cmp_u32";

    /// The dq family's row floor for a family mask.
    #[must_use]
    pub fn dq_min_rows(mask: u32) -> u32 {
        if mask & DQB != 0 {
            DQB_MIN_ROWS
        } else {
            DQ_MIN_ROWS
        }
    }
    /// FN122c twin (card 1, IC-cold, the dequant launch included, A = the shipped `_h8`): the qb
    /// down 0.680 / 0.623 / 0.604 / 0.539 x m2w21t1_h8 at 1,024 / 2,048 / 4,096 / 8,192 rows (the
    /// q h8 down 0.709 / 0.688 / 0.676 / 0.633); the p4s up 0.746 / 0.683 / 0.667 / 0.666 x p2s_h8.
    /// dqb (FN122d): `dq` down to [`DQB_MIN_ROWS`] -- the band 513..1,023 rows (the prefill `p4s` band's floor) measured too (t5, card 1,
    /// IC-cold): qb 0.881 / 0.789 / 0.837 / 0.726 / 0.718 x and p4s 0.894 / 0.882 / 0.815 / 0.800 /
    /// 0.774 x at 513 / 640 / 768 / 896 / 1,023 rows (bar <= 1.00 below 2,048) -> routed from 513.
    pub const DQB_MIN_ROWS: u32 = 513;
    /// The dq family's floor without `dqb` (FN122c's twin rows start at 1,024).
    pub const DQ_MIN_ROWS: u32 = 1024;
    /// The dq down: `(cfg, mf, waves, groups)` (the bf16 symbol, no `_h8`).
    pub const DQ_DOWN: (&str, u32, u32, u32) = ("m8w21t1k64qb", 8, 21, 1);
    /// The dq up: the planned bf16 prefill up cfg it runs (rules band > 512 rows).
    pub const DQ_UP_CFG: &str = "p4s";
    /// The dequant symbol (`flashnext_hc8p`), its threads and elements a thread.
    pub const DEQUANT: &str = "fn122_hc8_dequant";
    /// The shipped prefill down cfg the hc family replaces (sps 1, rules bands max_m 512 / 8,192).
    pub const HC_SHIPPED_CFG: &str = "m2w21t1";
    /// The hc route: `(min rows, cfg, mf, waves, groups)`, the first row whose `min rows` <= M
    /// wins (descending); below the last row the shipped cfg runs. MEASURED (FN122b twin).
    /// FN122b twin (card 1, IC-cold, combine, A = m2w21t1_h8): m8w21t1k64q = 0.711 x at 1,024 rows,
    /// 0.687 at 2,048, 0.674 at 4,096, 0.642 at 8,192 (bar 0.75 at >= 2,048); 2.22 x at 125 (not routed).
    pub const HC_ROUTE: [(u32, &str, u32, u32, u32); 1] = [(1024, "m8w21t1k64q", 8, 21, 1)];

    /// The hc route's cfg at `m` rows: `(cfg, mf, waves, groups)`, `None` = the shipped cfg.
    #[must_use]
    pub fn hc_cfg(m: u32) -> Option<(&'static str, u32, u32, u32)> {
        HC_ROUTE
            .iter()
            .find(|r| m >= r.0)
            .map(|r| (r.1, r.2, r.3, r.4))
    }

    /// `--fn-prefill-tiles default|none|all|hc`.
    ///
    /// # Errors
    /// An unknown name.
    pub fn parse(s: &str) -> Result<u32, String> {
        match s.trim() {
            "default" | "none" | "" => return Ok(0),
            "all" => return Ok(ALL),
            _ => {}
        }
        let mut m = 0;
        for t in s.split(',').map(str::trim) {
            let i = NAMES.iter().position(|n| *n == t).ok_or_else(|| {
                format!("--fn-prefill-tiles: unknown family {t:?} (hc, dq, dqb, qsa, qsatwin, default, all)")
            })?;
            m |= 1 << i;
        }
        Ok(m)
    }

    #[must_use]
    pub fn names(m: u32) -> String {
        if m == 0 {
            return "default".into();
        }
        NAMES
            .iter()
            .enumerate()
            .filter(|(i, _)| m & (1 << i) != 0)
            .map(|(_, n)| *n)
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// FN128: the QSA selection RE-TILED for the decode / verify forms (LEDGER FN128). Each is an
/// ADDED symbol in the NEW object `flashnext_qsa128` whose every output word equals its shipped
/// parent in `flashnext_qsa` (card-1 oracle, `tools/capture/fn128/fn128_twin.py`, ties at the 512th
/// value + controls); a decode-form step of a CTA of <= [`MAX_ROWS`] rows names them, every other
/// step (prefill, wider CTAs) keeps the parents.
pub mod sel128 {
    /// topk: the exact top-512 with the row's keys in registers, one 512-thread workgroup a row
    /// (`fn128_qsa_topk512_k<K>`, K keys a thread from the row stride).
    pub const TOPK: u32 = 1 << 0;
    /// union: the per-CTA key list by rank over the rows' ascending lists (`fn128_qsa_union_s`).
    pub const UNION: u32 = 1 << 1;
    /// score: the block scores with one thread a group for every row of the tile
    /// (`fn128_qsa_score_s1`).
    pub const SCORE: u32 = 1 << 2;
    pub const NAMES: [&str; 3] = ["topk", "union", "score"];
    /// The kernels `--fn-sel128 all` runs (the twins that passed the card-1 bar) -- THE DEFAULT
    /// since FN128's holds h1 + h4 (`run::DEFAULT_FN_SEL128`).
    pub const ALL: u32 = TOPK | UNION;
    /// The decode CTA's widest row count the added symbols take (the union's LDS lists).
    pub const MAX_ROWS: u32 = 8;
    /// Threads a workgroup.
    pub const UNION_THREADS: u32 = 1024;
    pub const SCORE_THREADS: u32 = 64;
    pub const UNION_SYMBOL: &str = "fn128_qsa_union_s1k";
    pub const SCORE_SYMBOL: &str = "fn128_qsa_score_s1";
    /// The top-k's instantiations: (keys a thread, threads, symbol), smallest first.
    /// (the 262,144 band = 128 keys a thread at 512: not built, that band keeps the parent; 64
    /// keys at 512 spilled, so the 131,072 band runs 32 keys at 1,024 threads.)
    pub const TOPK_CELLS: [(u32, u32, &str); 4] = [
        (8, 512, "fn128_qsa_topk512_k8"),
        (16, 512, "fn128_qsa_topk512_k16"),
        (32, 512, "fn128_qsa_topk512_k32"),
        (32, 1024, "fn128_qsa_topk512_k32_t1024"),
    ];

    /// The top-k symbol and its threads for a score row stride of `sstride` groups: the smallest
    /// cell with `K x threads >= sstride` at a multiple of 4 keys a thread (`None` past 32,768).
    #[must_use]
    pub fn topk_cell(sstride: u64) -> Option<(&'static str, u32)> {
        TOPK_CELLS
            .iter()
            .find(|(k, nt, _)| u64::from(*k) >= sstride.div_ceil(u64::from(*nt)).div_ceil(4) * 4)
            .map(|(_, nt, s)| (*s, *nt))
    }

    /// Is `sym` one of the added symbols (its parent's class: byte-equal by the oracle)?
    #[must_use]
    pub fn parent(sym: &str) -> Option<&'static str> {
        if TOPK_CELLS.iter().any(|c| c.2 == sym) {
            Some(crate::fn19::TOPK_SYMBOL)
        } else if sym.starts_with("fn128_qsa_union_s") {
            Some(crate::fn19::UNION_SYMBOL)
        } else if sym == SCORE_SYMBOL {
            Some(crate::fn19::SCORE_SYMBOL)
        } else {
            None
        }
    }

    /// `--fn-sel128 default|none|all|<list>` (a list names `topk`, `union`, `score`).
    ///
    /// # Errors
    /// An unknown name.
    pub fn parse(s: &str) -> Result<u32, String> {
        match s.trim() {
            "default" | "none" | "" => return Ok(0),
            "all" => return Ok(ALL),
            _ => {}
        }
        let mut m = 0;
        for t in s.split(',').map(str::trim) {
            let i = NAMES.iter().position(|n| *n == t).ok_or_else(|| {
                format!("--fn-sel128: unknown kernel {t:?} (topk, union, score, default, all)")
            })?;
            m |= 1 << i;
        }
        Ok(m)
    }

    #[must_use]
    pub fn names(m: u32) -> String {
        if m == 0 {
            return "default".into();
        }
        NAMES
            .iter()
            .enumerate()
            .filter(|(i, _)| m & (1 << i) != 0)
            .map(|(_, n)| *n)
            .collect::<Vec<_>>()
            .join(",")
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn fn128_the_topk_k_covers_every_band_and_the_names_round_trip() {
            // FN51f's bands: 10,240 -> 2,560 groups ... 262,144 -> 65,536
            assert_eq!(topk_cell(2560), Some(("fn128_qsa_topk512_k8", 512)));
            assert_eq!(topk_cell(4096), Some(("fn128_qsa_topk512_k8", 512)));
            assert_eq!(topk_cell(4160), Some(("fn128_qsa_topk512_k16", 512)));
            assert_eq!(topk_cell(8192), Some(("fn128_qsa_topk512_k16", 512)));
            assert_eq!(topk_cell(16384), Some(("fn128_qsa_topk512_k32", 512)));
            assert_eq!(
                topk_cell(32768),
                Some(("fn128_qsa_topk512_k32_t1024", 1024))
            );
            assert_eq!(topk_cell(32832), None);
            assert_eq!(topk_cell(65536), None);
            assert_eq!(parse("default"), Ok(0));
            assert_eq!(parse("all"), Ok(ALL));
            assert_eq!(parse("topk,union,score"), Ok(TOPK | UNION | SCORE));
            assert!(parse("x").is_err());
            assert_eq!(names(TOPK | UNION), "topk,union");
            assert_eq!(parent("fn128_qsa_topk512_k16"), Some("fn_qsa_topk512"));
            assert_eq!(
                parent("fn128_qsa_topk512_k32_t1024"),
                Some("fn_qsa_topk512")
            );
            assert_eq!(parent("fn128_qsa_union_s1k"), Some("fn_qsa_union"));
            assert_eq!(parent("x"), None);
        }
    }
}

/// FN141: the drained-ring twins of the served decode kernels (LEDGER FN141). ADDED symbols in two
/// NEW objects, each byte-equal to its shipped parent by the card-1 oracle
/// (`tools/capture/fn141/fn141_probe.py`):
///   * `up` -- `flashnext_hc141` `fn141_up_{bf16,q8}_v2_s1d_h8` = `fn_hc_up_{bf16,q8}_v2_s1d_h8`
///     (every load at entry, the one-round q8 tail, the select-free RNE: equal for every record
///     without a NaN SCALE -- `tools/capture/fn141/hc8_up_scale_law.py`, the served pack PASS);
///   * `t8` -- `flashnext_t8141` `fn141_t8_a8w8_n64_pf4_ldc` (the W8 GEMM's k loop as a 4-deep
///     register ring) for the out / o projection's `fn111_t8_a8w8_n64_ldc` (M <= 16): N 2,560 K
///     3,072 only (the oracle's shape). NOT the 17..32-row `fn_t8p_a8w8_d2w4_ldc`: byte-equal and
///     -2.3 us IC-cold, but +6.4 us IN SITU (FN141 t1: 18.5 -> 24.9 us at b8 w3, the 2 row slices
///     re-read the weights under the step's traffic);
///   * `router` -- `flashnext_hc141` `fn141_bgemm_sk_m16_d16k80` = `fn_bgemm_sk_m16` for the
///     router (N 512) at 80-step slices (K 2,560 over 2 splits): a 16-step ring, the chain unrolled.
pub mod k141 {
    pub const UP: u32 = 1 << 0;
    pub const T8: u32 = 1 << 1;
    pub const ROUTER: u32 = 1 << 2;
    pub const ALL: u32 = UP | T8 | ROUTER;
    pub const NAMES: [&str; 3] = ["up", "t8", "router"];
    pub const UP_SHIP: [(&str, &str); 2] = [
        ("fn_hc_up_bf16_v2_s1d_h8", "fn141_up_bf16_v2_s1d_h8"),
        ("fn_hc_up_q8_v2_s1d_h8", "fn141_up_q8_v2_s1d_h8"),
    ];
    pub const T8_SYM: &str = "fn141_t8_a8w8_n64_pf4_ldc";
    /// the parent the ring replaces (M <= 16) + the 17..32-row tile it does NOT (in-situ loss)
    pub const T8_PARENTS: [&str; 2] = ["fn111_t8_a8w8_n64_ldc", "fn_t8p_a8w8_d2w4_ldc"];
    /// the one oracle-gated (N, K): the out / o projection a rank
    pub const T8_NK: (u64, u64) = (2560, 3072);
    pub const ROUTER_SYM: &str = "fn141_bgemm_sk_m16_d16k80";
    pub const ROUTER_PARENT: &str = "fn_bgemm_sk_m16";

    /// The added up of a planned fp8-mixer up (`None` = keep the shipped one).
    #[must_use]
    pub fn up_route(sym: &str) -> Option<&'static str> {
        UP_SHIP.iter().find(|r| r.0 == sym).map(|r| r.1)
    }

    /// Whether the t8 ring takes this W8 GEMM site (the parent symbol, N, K, rows).
    #[must_use]
    pub fn t8_takes(parent: &str, n: u64, k: u64, m: u64) -> bool {
        parent == T8_PARENTS[0] && m <= 16 && (n, k) == T8_NK
    }

    /// Whether the router ring takes this bf16 split-K site (symbol, N, K, splits).
    #[must_use]
    pub fn router_takes(sym: &str, n: u64, k: u64, sp: u64) -> bool {
        sym == ROUTER_PARENT && n == 512 && sp > 0 && k % sp == 0 && k / sp == 80 * 16
    }

    /// The shipped parent (its class: byte-equal by the oracle) of an added symbol.
    #[must_use]
    pub fn parent(sym: &str) -> Option<&'static str> {
        if let Some(r) = UP_SHIP.iter().find(|r| r.1 == sym) {
            return Some(r.0);
        }
        if sym == T8_SYM {
            // the class of the m16 cell the n64 tile re-tiles (FN111's mapping)
            return Some("dflash_gemm_a8w8_bs128_m16n128k128");
        }
        (sym == ROUTER_SYM).then_some(ROUTER_PARENT)
    }

    /// `--fn-k141 none|default|all|<list of up,t8,router>`.
    ///
    /// # Errors
    /// An unknown name.
    pub fn parse(s: &str) -> Result<u32, String> {
        match s.trim() {
            "default" | "none" | "" => return Ok(0),
            "all" => return Ok(ALL),
            _ => {}
        }
        let mut m = 0;
        for t in s.split(',').map(str::trim) {
            let i = NAMES.iter().position(|n| *n == t).ok_or_else(|| {
                format!("--fn-k141: unknown family {t:?} (up, t8, router, none, all)")
            })?;
            m |= 1 << i;
        }
        Ok(m)
    }

    #[must_use]
    pub fn names(m: u32) -> String {
        if m == 0 {
            return "none".into();
        }
        NAMES
            .iter()
            .enumerate()
            .filter(|(i, _)| m & (1 << i) != 0)
            .map(|(_, n)| *n)
            .collect::<Vec<_>>()
            .join(",")
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn fn141_parse_and_names_round_trip() {
            assert_eq!(parse("none"), Ok(0));
            assert_eq!(parse("default"), Ok(0));
            assert_eq!(parse("all"), Ok(ALL));
            assert_eq!(parse("up"), Ok(UP));
            assert_eq!(parse("t8,router"), Ok(T8 | ROUTER));
            assert!(parse("sfx").is_err());
            assert_eq!(names(ALL), "up,t8,router");
            assert_eq!(names(0), "none");
            assert_eq!(parse(&names(T8 | UP)), Ok(T8 | UP));
        }

        #[test]
        fn fn141_routes_take_the_gated_sites_only_and_parent_inverts_them() {
            for (ship, new) in UP_SHIP {
                assert_eq!(up_route(ship), Some(new));
                assert_eq!(parent(new), Some(ship));
            }
            // NEGATIVE CONTROLS: other up cfgs / kinds keep the shipped symbol
            for s in [
                "fn_hc_up_q8t_v2_s1d_h8",
                "fn_hc_up_q8_v2_p2s_h8",
                "fn_hc_up_q8_v2_s1d",
            ] {
                assert_eq!(up_route(s), None, "{s}");
            }
            assert!(t8_takes(T8_PARENTS[0], 2560, 3072, 4));
            assert!(
                !t8_takes(T8_PARENTS[1], 2560, 3072, 24),
                "d2w4 loses in situ (FN141 t1)"
            );
            assert!(!t8_takes(T8_PARENTS[0], 8192, 2560, 4), "another shape");
            assert!(!t8_takes("fn_t8p_a8w8_d4w4_ldc", 2560, 3072, 24));
            assert!(router_takes(ROUTER_PARENT, 512, 2560, 2));
            assert!(
                !router_takes(ROUTER_PARENT, 512, 2560, 4),
                "another slice length"
            );
            assert!(
                !router_takes(ROUTER_PARENT, 2560, 2560, 2),
                "the PLE value projection"
            );
            assert_eq!(parent(ROUTER_SYM), Some(ROUTER_PARENT));
            assert!(parent(T8_SYM).is_some());
            assert_eq!(parent("fn_hc_up_q8_v2_s1d_h8"), None);
        }
    }
}

/// FN163: the MoE route twins in the NEW object `moe_route163` (LEDGER FN163), each byte-equal to its
/// shipped parent by the card-1 oracle (`tools/capture/fn163/fn163_probe.py`: FN3's real routes +
/// random / ties / wide / NaN-Inf, every table byte, controls caught):
///   * `b8` -- `fn163_route_nt1024` for the three-launch train's route at 9..64 rows (the shipped
///     `moe_route_topk10_sort_e512_lds_dpp` at 9..16, `fn111_moe_route_lds_dpp_nt512` / FN4's at
///     17..64): the shipped selection text at 32 waves (T <= 32 in one round) + one packed scan;
///   * `b1` -- `fn163_front_e512_nw8_s4` for the fused front at router split 4 (the b1 class rows,
///     T 1..8): the shipped GEMM text + a register-rank top-10 (exact by construction, the shipped
///     rounds as its fallback). The split-2 twin is built and gated, NOT routed (mixed timing).
pub mod route163 {
    pub const B8: u32 = 1 << 0;
    pub const B1: u32 = 1 << 1;
    pub const ALL: u32 = B8 | B1;
    pub const NAMES: [&str; 2] = ["b8", "b1"];
    pub const ROUTE_SYM: &str = "fn163_route_nt1024";
    pub const ROUTE_THREADS: u64 = 1024;
    pub const ROUTE_MIN_T: u32 = 9;
    pub const ROUTE_MAX_T: u32 = 64;
    /// the shipped routes the b8 twin replaces (each byte-equal to FN4's at T <= 64)
    pub const ROUTE_PARENTS: [&str; 3] = [
        "moe_route_topk10_sort_e512_lds_dpp",
        "fn111_moe_route_lds_dpp_nt512",
        "moe_route_topk10_sort_e512",
    ];
    pub const FRONT: (&str, &str) = ("moe_front_e512_nw8_s4_dpp", "fn163_front_e512_nw8_s4");

    /// Whether the b8 twin takes this route site (the planned symbol, rows).
    #[must_use]
    pub fn route_takes(sym: &str, m: u32) -> bool {
        ROUTE_PARENTS.contains(&sym) && (ROUTE_MIN_T..=ROUTE_MAX_T).contains(&m)
    }

    /// The b1 twin of a planned front symbol (`None` = keep the shipped one).
    #[must_use]
    pub fn front_route(sym: &str) -> Option<&'static str> {
        (sym == FRONT.0).then_some(FRONT.1)
    }

    /// The shipped parent (its class: byte-equal by the oracle) of an added symbol.
    #[must_use]
    pub fn parent(sym: &str) -> Option<&'static str> {
        if sym == ROUTE_SYM {
            return Some(ROUTE_PARENTS[2]);
        }
        (sym == FRONT.1).then_some(FRONT.0)
    }

    /// `--fn-route163 none|default|all|<list of b8,b1>`.
    ///
    /// # Errors
    /// An unknown name.
    pub fn parse(s: &str) -> Result<u32, String> {
        match s.trim() {
            "default" | "none" | "" => return Ok(0),
            "all" => return Ok(ALL),
            _ => {}
        }
        let mut m = 0;
        for t in s.split(',').map(str::trim) {
            let i = NAMES
                .iter()
                .position(|n| *n == t)
                .ok_or_else(|| format!("--fn-route163: unknown band {t:?} (b8, b1, none, all)"))?;
            m |= 1 << i;
        }
        Ok(m)
    }

    #[must_use]
    pub fn names(m: u32) -> String {
        if m == 0 {
            return "none".into();
        }
        NAMES
            .iter()
            .enumerate()
            .filter(|(i, _)| m & (1 << i) != 0)
            .map(|(_, n)| *n)
            .collect::<Vec<_>>()
            .join(",")
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn fn163_parse_names_and_the_gated_sites() {
            assert_eq!(parse("none"), Ok(0));
            assert_eq!(parse("all"), Ok(ALL));
            assert_eq!(parse("b8"), Ok(B8));
            assert_eq!(parse("b1,b8"), Ok(ALL));
            assert!(parse("b4").is_err());
            assert_eq!(names(ALL), "b8,b1");
            assert_eq!(parse(&names(B1)), Ok(B1));
            for p in ROUTE_PARENTS {
                assert!(route_takes(p, 24));
                assert!(route_takes(p, 9));
                assert!(route_takes(p, 64));
                // NEGATIVE CONTROLS: the front band and past the LDS route's 64 tokens
                assert!(!route_takes(p, 8));
                assert!(!route_takes(p, 65));
            }
            assert!(
                !route_takes("moe_route_topk10_sort_e512_lds", 24),
                "the route control"
            );
            assert_eq!(front_route(FRONT.0), Some(FRONT.1));
            assert_eq!(
                front_route("moe_front_e512_nw8_s2_dpp"),
                None,
                "s2 is not routed"
            );
            assert_eq!(
                front_route("moe_front_e512_nw8_s4"),
                None,
                "the route control"
            );
            assert_eq!(parent(ROUTE_SYM), Some(ROUTE_PARENTS[2]));
            assert_eq!(parent(FRONT.1), Some(FRONT.0));
            assert_eq!(parent("fn163_front_e512_nw8_s2"), None);
        }
    }
}

/// FN111: the roofline census's re-tiles (LEDGER FN111). Each is an ADDED symbol in its parent's
/// code object whose every output byte equals the shipped cell it replaces (card-1 oracle,
/// `tools/capture/fn111/fn111_twin.py`, >= 1,000 cases + a control); a tile that missed the timing
/// bar (>= 10 % of the shipped kernel's own duration) is built and nameable but not in `all`.
pub mod tiles111 {
    /// t1: the W8A8 GEMM at <= 16 rows on 64-column tiles (one fragment a wave) where the shipped
    /// cell is `dflash_gemm_a8w8_bs128_m16n128k128` (the out / o projections, N <= 8,192).
    pub const T1: u32 = 1 << 0;
    /// t2: family 2's merged projection on 64-column W8 tiles (built; MISSED the bar: not in `all`).
    pub const T2: u32 = 1 << 1;
    /// t3: the MoE route at 17..64 tokens on FN49a's LDS + DPP body at 16 waves (FN4's route
    /// above T 16 today).
    pub const T3: u32 = 1 << 2;
    pub const BUILT: u32 = T1 | T2 | T3;
    /// The tiles `--fn-tiles all` runs: the built tiles that passed the card-1 bar.
    pub const ALL: u32 = T1 | T3;
    pub const NAMES: [&str; 3] = ["t1", "t2", "t3"];
    /// The added symbols, in tile order.
    pub const SYMS: [&str; 3] = [
        "fn111_t8_a8w8_n64_ldc",
        "fn111_t8_q8w8_sfx_n64_ldc",
        "fn111_moe_route_lds_dpp_nt512",
    ];
    /// t1's largest N (the head's 124,160-column GEMM keeps its cell: at its roof already).
    pub const T1_MAX_N: u64 = 8192;
    /// t3's token band and threads.
    pub const T3_MIN_T: u32 = 17;
    pub const T3_MAX_T: u32 = 64;
    pub const T3_THREADS: u64 = 512;

    /// `--fn-tiles default|none|all|<list>` (a list names tiles `t1` or their symbols).
    ///
    /// # Errors
    /// An unknown name.
    pub fn parse(s: &str) -> Result<u32, String> {
        match s.trim() {
            "default" | "none" | "" => return Ok(0),
            "all" => return Ok(ALL),
            _ => {}
        }
        let mut m = 0;
        for t in s.split(',').map(str::trim) {
            let i = NAMES
                .iter()
                .position(|n| *n == t)
                .or_else(|| SYMS.iter().position(|n| *n == t))
                .ok_or_else(|| {
                    format!(
                        "--fn-tiles: unknown tile {t:?} (t1..t3 or an fn111_ symbol, default, all)"
                    )
                })?;
            m |= 1 << i;
        }
        Ok(m)
    }

    #[must_use]
    pub fn names(m: u32) -> String {
        if m == 0 {
            return "default".into();
        }
        NAMES
            .iter()
            .enumerate()
            .filter(|(i, _)| m & (1 << i) != 0)
            .map(|(_, n)| *n)
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// FN86b: the launch-fusion families (LEDGER FN86's census numbering). A fused family is an ADDED
/// symbol whose every output byte equals the shipped chain it replaces (oracle-gated on card 1);
/// a kernel change at a fused site re-runs that family's oracle.
pub mod fuse86 {
    /// Family 1: a producer + the W8 GEMM's activation quant as one launch -- the GDN gated norm
    /// (`fn_gdn_norm_q8_sig_t256`) and the QSA attention gate (`fn_attn_gate_q8_t256`).
    pub const F1: u32 = 1 << 0;
    /// Family 4: a merged projection's bf16 suffix GEMM (`fn_t8_bgemm_m16_ldc`) as extra
    /// workgroups of the W8 prefix's activation-quant launch (`fn86_q8_bgemm_m16_ldc`).
    pub const F4: u32 = 1 << 3;
    /// Family 5 (FN86c): the f16-carry GDN decode update with family 1's gated norm + act quant
    /// in its tail (`fn86_gdn_fused_update_nq8_fc_*` at b1, `fn86_gdn_recurrent_nq8_fc_*` in the
    /// pair form); composes on family 1 (its site is family 1's), so `f5` needs `f1`.
    pub const F5: u32 = 1 << 4;
    /// Family 2 (FN86c): the act quant inside the W8 GEMM's A staging, with family 4's bf16
    /// suffix as the same launch's extra workgroups (`fn86_t8_q8w8_sfx_ldc`): a merged projection
    /// in ONE launch; composes on family 4 (its site is family 4's), so `f2` needs `f4`. The
    /// out_proj quants are family 1's / 5's already (no launch left to take there).
    pub const F2: u32 = 1 << 1;
    /// Family 2's rows: the m16 body runs every row tile at <= this M (above it the chain's
    /// multi-row W8 tiles stay; FN86c's card-1 timing).
    pub const F2_MAX_ROWS: u64 = 64;
    /// Family 2's symbol.
    pub const Q8W8_SFX: &str = "fn86_t8_q8w8_sfx_ldc";
    /// Family 10 (FN86d): the drafter pass's int2-tail glue (`--fn-draft-head int2` under the W8
    /// head) in four launches instead of eight -- `fn86_draft_xg_sum_q8` (row gather + group
    /// sums + the rerank's act quant), `fn86_topk_gather_cand` (the top-80 gather + candidate
    /// columns; the threshold stays), `fn86_draft_argmax_cand_pack` (the candidates' argmax +
    /// FN22's pack),
    /// each byte-equal to its chain (`tools/gates/fn86d_twin_hold.sh`). Not a target-layer site:
    /// it lives in the drafter pass's tail (`worker::emit_tail`).
    pub const F10: u32 = 1 << 9;
    /// Launches family 10 removes from a drafter pass (8 -> 4).
    pub const F10_SAVED: u32 = 4;
    /// Family 10's symbols.
    pub const F10_XG: &str = "fn86_draft_xg_sum_q8";
    pub const F10_TOPK: &str = "fn86_topk_gather_cand";
    pub const F10_ACP: &str = "fn86_draft_argmax_cand_pack";
    /// Family 6 (ATTN-OPT-3): a TARGET QSA layer's front at a decode / verify step --
    /// `fn_qk_norm_rope_h256_rot64_mi1_v2` + `fn_idx_q_compress_h128_v2` + the KV / ring append as
    /// ONE launch (`fn_attnfuse_front_h256_{bf16,fp8}`, 3 -> 1; byte-equal to the chain,
    /// `tools/capture/attn3/attn3_twin.py`). The indexer reads the ring only before the step's first
    /// row and the append writes only the step's rows: disjoint while a sequence's rows + 3 <= the
    /// ring's rows ([`F6_MAX_W`]); prefill chunks and the drafter keep the chain. In [`ALL`] since
    /// the f6/f7 default flip (branch fn-fuse-f6f7-default; ATTN-OPT-3's arm before it).
    pub const F6: u32 = 1 << 5;
    /// Family 6's widest verify width (a sequence's rows): the ring's 8 rows - 3.
    pub const F6_MAX_W: u32 = 5;
    /// Launches family 6 removes at a site.
    pub const F6_SAVED: u32 = 2;
    /// Family 6's symbols (bf16 / e4m3 KV pages).
    pub const F6_FRONT_BF16: &str = "fn_attnfuse_front_h256_bf16";
    pub const F6_FRONT_FP8: &str = "fn_attnfuse_front_h256_fp8";
    /// Family 7 (ATTN-OPT-3): a TARGET QSA layer's split-KV combine + family 1's gate + act quant at a
    /// decode / verify step as ONE launch (`fn_attnfuse_combine_gate_q8_h256`, 2 -> 1; byte-equal to
    /// the pair); its site is family 1's (the q8 gate), so `f7` needs `f1`. In [`ALL`] since the
    /// f6/f7 default flip (ATTN-OPT-3's arm before it).
    pub const F7: u32 = 1 << 6;
    /// Family 7's symbol.
    pub const F7_COMBINE_GATE_Q8: &str = "fn_attnfuse_combine_gate_q8_h256";
    /// The families this tree builds (any may be named in `--fn-fuse`).
    pub const BUILT: u32 = F1 | F2 | F4 | F5 | F6 | F7 | F10;
    /// The families `--fn-fuse all` runs (the default): the built families the bit-exact rule kept
    /// (a family that costs served time is dropped from `all` by name, and stays nameable). Since
    /// the f6/f7 flip every built family (`== BUILT`).
    pub const ALL: u32 = F1 | F2 | F4 | F5 | F6 | F7 | F10;
    /// The f6/f7 flip's predecessor set: FN86d's `all` = `f1,f2,f4,f5,f10` (the flip hold's control
    /// arm).
    pub const ALL_FN86D: u32 = F1 | F2 | F4 | F5 | F10;
    /// Family 10's predecessor set: FN86c's `all` (the FN86d hold's control arm).
    pub const ALL_FN86C: u32 = F1 | F2 | F4 | F5;
    /// Family names in bit order (`f1` .. `f10`; f9 = FN85's reduce-in-prologue, closed).
    pub const NAMES: [&str; 10] = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"];
    /// The GDN norm's fused twin.
    pub const GDN_NORM_Q8: &str = "fn_gdn_norm_q8_sig_t256";
    /// The shipped GDN norm the twin replaces (with the act quant after it).
    pub const GDN_NORM: &str = "fn_gdn_norm_bf16_sig_t256";
    /// The attention gate's fused twin.
    pub const ATTN_GATE_Q8: &str = "fn_attn_gate_q8_t256";
    /// The shipped gate the twin replaces.
    pub const ATTN_GATE: &str = "fn_attn_gate_bf16_t256";
    /// Family 4's symbol.
    pub const Q8_BGEMM_LDC: &str = "fn86_q8_bgemm_m16_ldc";
    /// Family 5's fused-update symbol (b1: the update's one-launch form).
    pub const F5_FUSED: &str = "fn86_gdn_fused_update_nq8_fc_f16s_w4k128v128_ab1";
    /// Family 5's recurrent symbol (the pair form, after the conv update).
    pub const F5_RECUR: &str = "fn86_gdn_recurrent_nq8_fc_f16s_k128_v128_ab1_vs1";

    /// `--fn-fuse none | all | f1,f4,...` -> the family mask (`all` in a list = the default set
    /// plus any named arm; `all,f6,f7` == `all` since the f6/f7 flip); a family this tree does not
    /// build is refused by name.
    pub fn parse(s: &str) -> Result<u32, String> {
        match s.trim() {
            "none" | "" => return Ok(0),
            "all" => return Ok(ALL),
            _ => {}
        }
        let mut m = 0;
        for t in s.split(',').map(str::trim) {
            if t == "all" {
                m |= ALL;
                continue;
            }
            let i = NAMES
                .iter()
                .position(|n| *n == t)
                .ok_or_else(|| format!("--fn-fuse: unknown family {t:?} (f1..f10, none, all)"))?;
            if BUILT & (1 << i) == 0 {
                return Err(format!(
                    "--fn-fuse: family {t} is not built (built: {})",
                    names(BUILT)
                ));
            }
            m |= 1 << i;
        }
        if m & F2 != 0 && m & F4 == 0 {
            return Err(
                "--fn-fuse: family f2 composes on f4 (its site is family 4's): name both".into(),
            );
        }
        if m & F5 != 0 && m & F1 == 0 {
            return Err(
                "--fn-fuse: family f5 composes on f1 (its site is family 1's): name both".into(),
            );
        }
        if m & F7 != 0 && m & F1 == 0 {
            return Err(
                "--fn-fuse: family f7 composes on f1 (its site is family 1's): name both".into(),
            );
        }
        Ok(m)
    }

    /// The mask's family names (`none` when empty).
    pub fn names(m: u32) -> String {
        if m == 0 {
            return "none".into();
        }
        NAMES
            .iter()
            .enumerate()
            .filter(|(i, _)| m & (1 << i) != 0)
            .map(|(_, n)| *n)
            .collect::<Vec<_>>()
            .join(",")
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct HcKernels {
    pub down_mix: String,
    pub down_expand: String,
    pub down_combine: String,
    pub down_final: String,
    pub up_bf16: String,
    pub up_q8: String,
    pub up_q8_tiled: String,
    pub combine: String,
    pub gdn_norm_bf16: String,
    /// FN9: Flash-Next's gated norm (sigmoid gate); `gdn_norm_bf16` is the 27B's silu twin.
    pub gdn_norm_bf16_sigmoid: String,
    pub attn_gate_bf16: String,
    /// FN11: `fn_hc_down_{mode}_v2_{cfg}` / `fn_hc_up_{kind}_v2_{cfg}` templates.
    pub down2: String,
    pub up2: String,
    pub gate_only: Vec<String>,
}

/// One bf16 GEMM shape: `bands` = `[max_m, symbol, splits]`, first band holding M wins.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct BgemmShape {
    pub shape: String,
    pub n: Vec<u32>,
    pub k: u32,
    pub bands: Vec<(u32, String, u32)>,
}

/// FN84: one row of the MoE miss-overlap order law (`[[moe_miss]]`): at decode / verify steps of
/// at most `t` rows the `capped` arms interleave ONE RAM-tier workgroup per `k` resident ones
/// (`moe_back_mo{,p}_*`'s `kint`; measured per T, LEDGER FN84). No byte depends on it.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct MissBand {
    pub t: u32,
    pub k: i32,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct FlashNextRules {
    pub schema_version: u32,
    pub name: String,
    pub model: HcModel,
    pub hc: HcPolicy,
    pub kernels: HcKernels,
    pub bgemm: Vec<BgemmShape>,
    /// FN84: the miss-overlap order law's interleave by rows (empty = no capped arm).
    #[serde(default)]
    pub moe_miss: Vec<MissBand>,
}

/// Which HC operation a mix performs on its input stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixMode {
    /// The input is a materialized stream (a mix with no pending combine, e.g. after layer 1's PLE).
    Mix,
    /// Layer 0's attention mix: the stream is `embed.repeat(1, 4)`, written out by the down.
    Expand,
    /// `combine_and_mix`: x' = bf16(res + blk * 2 sigmoid(inj / 4)), written out, then the mix.
    Combine,
    /// The final `hyper_connection_mixer`: a combine and a mix with no injection (Wdown only).
    Final,
}

/// What the up writes besides the bf16 block input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixOut {
    /// bf16 only (the attention side: GDN in_proj / QSA q,k,v are bf16 GEMMs; the final mixer).
    Bf16,
    /// + e4m3 per-token quant, row-major (the MoE side: the W4A8 experts' and shared expert's A).
    Q8,
    /// + e4m3 per-token quant, fragment-tiled (the A-tiled prefill GEMM's layout).
    Q8Tiled,
}

/// One launch: symbol, grid, workgroup threads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Launch {
    pub symbol: String,
    pub grid: [u32; 3],
    pub threads: u32,
}

/// A mix of `m` rows: its two launches and the scratch they need (bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixPlan {
    pub down: Launch,
    pub up: Launch,
    pub sps: u32,
    /// `part` f32 [4 * sps, m, partial_stride].
    pub part_bytes: u64,
    /// The down's and the up's arrival counters, i32 [row blocks] each: ZEROED once at
    /// allocation, reset by each launch's last arrival (the split-K decode GEMM's protocol).
    pub down_counters: u32,
    pub up_counters: u32,
    /// The up's per-row partial amax, f32 [column groups, m] (quant variants only).
    pub amax_bytes: u64,
    /// e4m3 output bytes: row-major m x hidden, or whole 16-row fragments when tiled.
    pub q8_bytes: u64,
}

impl FlashNextRules {
    /// FN84: the `capped` arms' interleave `k` (>= 1) at `tokens` rows: the first `[[moe_miss]]`
    /// row holding it.
    #[must_use]
    pub fn miss_k(&self, tokens: u32) -> Option<i32> {
        self.moe_miss.iter().find(|b| tokens <= b.t).map(|b| b.k)
    }

    pub fn prod() -> Result<Self> {
        Self::from_toml(PROD_RULES_TEXT)
    }

    pub fn from_toml(text: &str) -> Result<Self> {
        let r: FlashNextRules = toml::from_str(text).map_err(|e| Error::Parse(e.to_string()))?;
        if r.schema_version != 1 {
            return Err(Error::Rule(format!(
                "flashnext rules schema_version {} (this build reads 1)",
                r.schema_version
            )));
        }
        let (m, hc) = (&r.model, &r.hc);
        if (m.hc_count, m.hidden, m.lowrank) != (4, 2560, 320) {
            return Err(Error::Rule(format!(
                "flashnext HC geometry {} x {} rank {} is not the compiled 4 x 2560 rank 320",
                m.hc_count, m.hidden, m.lowrank
            )));
        }
        if hc.launches_per_mix != 2 || hc.launches_per_layer != 2 * hc.launches_per_mix {
            return Err(Error::Rule(format!(
                "the HC launch law is 2 per mix, 4 per layer; the table says {} / {}",
                hc.launches_per_mix, hc.launches_per_layer
            )));
        }
        if hc.partial_stride < m.lowrank + m.hc_count || hc.partial_stride % 16 != 0 {
            return Err(Error::Rule(format!(
                "partial_stride {} must hold {} columns in whole 16-column tiles",
                hc.partial_stride,
                m.lowrank + m.hc_count
            )));
        }
        for &s in &hc.sps_allowed {
            let kw = m.hidden / s.max(1);
            if s == 0 || m.hidden % s != 0 || kw % 64 != 0 {
                return Err(Error::Rule(format!(
                    "sps {s}: a K slice of {kw} is not whole 64-k groups"
                )));
            }
        }
        let mut prev = 0;
        for b in &hc.band {
            if b.max_m <= prev {
                return Err(Error::Rule("hc.band max_m must increase".into()));
            }
            prev = b.max_m;
            if !hc.sps_allowed.contains(&b.sps)
                || !(1..=2).contains(&b.down_mf)
                || !(1..=2).contains(&b.up_mf)
            {
                return Err(Error::Rule(format!(
                    "hc.band max_m {}: sps {} / mf {} / {} is not a compiled variant",
                    b.max_m, b.sps, b.down_mf, b.up_mf
                )));
            }
            if let Some(c) = &b.down_v2 {
                let d = hc.down_v2.iter().find(|d| &d.cfg == c).ok_or_else(|| {
                    Error::Rule(format!("hc.band max_m {}: no hc.down_v2 `{c}`", b.max_m))
                })?;
                if (m.hidden / b.sps) % d.kc != 0 {
                    return Err(Error::Rule(format!(
                        "hc.band max_m {}: down_v2 `{c}` slab chunk {} does not divide the K slice {}",
                        b.max_m,
                        d.kc,
                        m.hidden / b.sps
                    )));
                }
            }
            if let Some(c) = &b.up_v2 {
                if !hc.up_v2.iter().any(|u| &u.cfg == c) {
                    return Err(Error::Rule(format!(
                        "hc.band max_m {}: no hc.up_v2 `{c}`",
                        b.max_m
                    )));
                }
            }
        }
        let tiles = (m.lowrank + m.hc_count).div_ceil(16);
        for d in &hc.down_v2 {
            if !(1..=2).contains(&d.mf)
                || d.waves == 0
                || d.groups * d.waves * d.tiles < tiles
                || d.kc % 16 != 0
            {
                return Err(Error::Rule(format!(
                    "hc.down_v2 `{}`: mf {} / {} waves x {} tiles x {} groups / kc {} cannot cover {tiles} tiles",
                    d.cfg, d.mf, d.waves, d.tiles, d.groups, d.kc
                )));
            }
        }
        for u in &hc.up_v2 {
            // FN21: the prefill bodies own 16 x waves block-input columns a workgroup
            // (grid_x = hidden / (16 waves)) and up to 4 row fragments.
            let wide = u.waves == 8 && u.grid_x == m.hidden / 128 && (1..=4).contains(&u.mf);
            let v2 = u.waves == 4
                && (1..=2).contains(&u.mf)
                && [m.hidden / 16, m.hidden / 64].contains(&u.grid_x);
            if !(wide || v2) {
                return Err(Error::Rule(format!(
                    "hc.up_v2 `{}`: mf {} / grid_x {} / {} waves is not a compiled layout",
                    u.cfg, u.mf, u.grid_x, u.waves
                )));
            }
        }
        for g in &r.bgemm {
            let mut prev = 0;
            for (max_m, sym, splits) in &g.bands {
                if *max_m <= prev {
                    return Err(Error::Rule(format!(
                        "bgemm {}: bands must increase",
                        g.shape
                    )));
                }
                prev = *max_m;
                if g.k % (16 * splits.max(&1)) != 0 {
                    return Err(Error::Rule(format!(
                        "bgemm {}: K {} is not whole 16-k steps per split ({splits})",
                        g.shape, g.k
                    )));
                }
                if *splits != 1 && !sym.starts_with("fn_bgemm_sk_m") {
                    return Err(Error::Rule(format!(
                        "bgemm {}: `{sym}` does not split K",
                        g.shape
                    )));
                }
                if band_rows(sym).is_none() {
                    return Err(Error::Rule(format!(
                        "bgemm {}: unknown symbol `{sym}`",
                        g.shape
                    )));
                }
            }
        }
        Ok(r)
    }

    /// The band row a mix of `m` rows takes (the first whose `max_m` holds it).
    pub fn hc_band(&self, m: u32) -> Result<&HcBand> {
        self.hc
            .band
            .iter()
            .find(|b| m >= 1 && m <= b.max_m)
            .ok_or_else(|| Error::Rule(format!("no hc.band holds a mix of {m} rows")))
    }

    /// The two launches and the scratch of one mix.
    pub fn mix_plan(&self, mode: MixMode, out: MixOut, m: u32) -> Result<MixPlan> {
        self.mix_plan_at(mode, out, m, m)
    }

    /// FN24: a mix of `m` rows in the CELL of `class_m` rows (the band -- `sps`, the down / up
    /// symbols -- is `class_m`'s; the grids, counters and scratch are `m`'s). Every v2 body tiles
    /// any M in 16-row fragments and the `sps` K slices are a kernarg, so any band runs any M; a
    /// row's bytes depend on the band only (FN11: `sps` is the class). `class_m == m` is
    /// [`Self::mix_plan`].
    pub fn mix_plan_at(&self, mode: MixMode, out: MixOut, class_m: u32, m: u32) -> Result<MixPlan> {
        if m == 0 {
            return Err(Error::Rule("a mix of 0 rows".into()));
        }
        let b = self.hc_band(class_m)?;
        let k = &self.kernels;
        let down_t = match mode {
            MixMode::Mix => &k.down_mix,
            MixMode::Expand => &k.down_expand,
            MixMode::Combine => &k.down_combine,
            MixMode::Final => &k.down_final,
        };
        if mode == MixMode::Final && out != MixOut::Bf16 {
            return Err(Error::Rule(
                "the final mixer feeds the bf16 head: it has no e4m3 output".into(),
            ));
        }
        let up_t = match out {
            MixOut::Bf16 => &k.up_bf16,
            MixOut::Q8 => &k.up_q8,
            MixOut::Q8Tiled => &k.up_q8_tiled,
        };
        let hidden = self.model.hidden;
        let down_rows = 16 * b.down_mf;
        let up_rows = 16 * b.up_mf;
        let dv2 = b.down_v2.as_ref().map(|c| {
            self.hc
                .down_v2
                .iter()
                .find(|d| &d.cfg == c)
                .expect("validated")
        });
        let uv2 = b.up_v2.as_ref().map(|c| {
            self.hc
                .up_v2
                .iter()
                .find(|u| &u.cfg == c)
                .expect("validated")
        });
        let down_rows = dv2.map_or(down_rows, |d| 16 * d.mf);
        let up_rows = uv2.map_or(up_rows, |u| 16 * u.mf);
        let down_blocks = m.div_ceil(down_rows);
        let up_blocks = m.div_ceil(up_rows);
        let cgroups = uv2.map_or(hidden / (16 * self.hc.up_waves), |u| u.grid_x);
        let mode_name = match mode {
            MixMode::Mix => "mix",
            MixMode::Expand => "expand",
            MixMode::Combine => "combine",
            MixMode::Final => "final",
        };
        let kind_name = match out {
            MixOut::Bf16 => "bf16",
            MixOut::Q8 => "q8",
            MixOut::Q8Tiled => "q8t",
        };
        let q8 = match out {
            MixOut::Bf16 => 0,
            MixOut::Q8 => u64::from(m) * u64::from(hidden),
            MixOut::Q8Tiled => u64::from(m.div_ceil(16) * 16) * u64::from(hidden),
        };
        Ok(MixPlan {
            down: match dv2 {
                None => Launch {
                    symbol: down_t.replace("{mf}", &b.down_mf.to_string()),
                    grid: [self.model.hc_count * b.sps, down_blocks, 1],
                    threads: 32 * self.hc.down_waves,
                },
                Some(d) => Launch {
                    symbol: k
                        .down2
                        .replace("{mode}", mode_name)
                        .replace("{cfg}", &d.cfg),
                    grid: [self.model.hc_count * b.sps, down_blocks, d.groups],
                    threads: 32 * d.waves,
                },
            },
            up: Launch {
                symbol: match uv2 {
                    None => up_t.replace("{mf}", &b.up_mf.to_string()),
                    Some(u) => k.up2.replace("{kind}", kind_name).replace("{cfg}", &u.cfg),
                },
                grid: [cgroups, up_blocks, 1],
                threads: 32 * uv2.map_or(self.hc.up_waves, |u| u.waves),
            },
            sps: b.sps,
            part_bytes: 4
                * u64::from(self.model.hc_count * b.sps)
                * u64::from(m)
                * u64::from(self.hc.partial_stride),
            down_counters: down_blocks * dv2.map_or(1, |d| d.groups),
            up_counters: if out == MixOut::Bf16 { 0 } else { up_blocks },
            amax_bytes: if out == MixOut::Bf16 {
                0
            } else {
                4 * u64::from(cgroups) * u64::from(m)
            },
            q8_bytes: q8,
        })
    }

    /// HC launches in one decoder layer at `m` rows: the attention-side and the MoE-side mix.
    /// `ple` = layer 1, whose attention side runs the standalone combine before the PLE add and
    /// then a plain mix.
    pub fn hc_launches_per_layer(&self, ple: bool) -> u32 {
        self.hc.launches_per_layer + u32::from(ple)
    }

    /// Every routable symbol this plugin owns (gate-only symbols listed separately).
    pub fn symbols(&self) -> Vec<String> {
        let k = &self.kernels;
        let mut v = Vec::new();
        for t in [
            &k.down_mix,
            &k.down_expand,
            &k.down_combine,
            &k.down_final,
            &k.up_bf16,
            &k.up_q8,
            &k.up_q8_tiled,
        ] {
            for mf in 1..=2 {
                v.push(t.replace("{mf}", &mf.to_string()));
            }
        }
        for d in &self.hc.down_v2 {
            for mode in ["mix", "expand", "combine", "final"] {
                v.push(k.down2.replace("{mode}", mode).replace("{cfg}", &d.cfg));
            }
        }
        for u in &self.hc.up_v2 {
            for kind in ["bf16", "q8", "q8t"] {
                v.push(k.up2.replace("{kind}", kind).replace("{cfg}", &u.cfg));
            }
        }
        v.push(k.combine.clone());
        v.push(k.gdn_norm_bf16.clone());
        v.push(k.gdn_norm_bf16_sigmoid.clone());
        v.push(k.attn_gate_bf16.clone());
        for g in &self.bgemm {
            for (_, sym, _) in &g.bands {
                if sym.starts_with("fn_") && !v.contains(sym) {
                    v.push(sym.clone());
                }
            }
        }
        v
    }

    /// The plugin a routed symbol lives in: this one, or the drafter's committed bands.
    pub fn plugin_for(&self, sym: &str) -> Option<&'static str> {
        if sym.starts_with("fn_") {
            Some(PLUGIN)
        } else if sym.starts_with("dflash_bgemm_") {
            Some("dflash_bgemm")
        } else {
            None
        }
    }

    /// The bf16 GEMM launch for `shape` at `m` rows and `n` columns: (symbol, grid, threads,
    /// splits). The split-K band needs `splits * m * n` f32 of partials and one i32 counter per
    /// output tile when `splits > 1`.
    pub fn bgemm_route(&self, shape: &str, m: u32, n: u32) -> Result<(Launch, u32)> {
        let g = self
            .bgemm
            .iter()
            .find(|g| g.shape == shape)
            .ok_or_else(|| Error::Rule(format!("no bgemm shape `{shape}`")))?;
        if !g.n.contains(&n) {
            return Err(Error::Rule(format!(
                "bgemm {shape}: N {n} is not one of {:?}",
                g.n
            )));
        }
        let (_, sym, splits) = g
            .bands
            .iter()
            .find(|(max_m, _, _)| m >= 1 && m <= *max_m)
            .ok_or_else(|| Error::Rule(format!("bgemm {shape}: no band holds M {m}")))?;
        let (rows, cols, threads) = band_rows(sym).expect("validated at parse");
        Ok((
            Launch {
                symbol: sym.clone(),
                grid: [n.div_ceil(cols), m.div_ceil(rows), *splits],
                threads,
            },
            *splits,
        ))
    }
}

/// FN24: the symbol that runs `sym`'s CELL at `m` rows. Every routed bf16 symbol tiles any M on
/// grid y but ONE: `dflash_bgemm_bf16_m16n16` computes rows 0..16 only (it ignores `blockIdx.y`;
/// the rules never route it above 16). Its row-slice twin `dflash_bgemm_bf16_mtn16` is the same
/// tile per 16-row slice -- the same k-steps in the same order into the same one fp32 accumulator,
/// "bit-identical to the m16 kernel on each 16-row slice" (dflash_bgemm.hip, S3 / S4) -- so a
/// width-1-class verify above 16 rows runs the twin and keeps the class (FN24 hold 1 found the cap:
/// b8 w3's rows 16..23 were never written).
#[must_use]
pub fn class_symbol_at(sym: &str, m: u32) -> &str {
    if sym == "dflash_bgemm_bf16_m16n16" && m > 16 {
        "dflash_bgemm_bf16_mtn16"
    } else {
        sym
    }
}

/// (rows per workgroup, columns per workgroup, threads) of a bf16 GEMM symbol, from its name.
pub fn band_rows(sym: &str) -> Option<(u32, u32, u32)> {
    Some(match sym {
        "fn_bgemm_sk_m16" | "dflash_bgemm_bf16_m16n16" | "dflash_bgemm_bf16_mtn16" => (16, 64, 128),
        "fn_bgemm_sk_m32" | "dflash_bgemm_bf16_m32n16" => (32, 64, 128),
        "fn_bgemm_sk_m64" => (64, 64, 128),
        "fn_bgemm_lds_l128x128" => (128, 128, 256),
        "fn_bgemm_lds_l128x64" => (128, 64, 128),
        "fn_bgemm_lds_l64x64" => (64, 64, 128),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_prod_table_parses_and_covers_every_row_count() {
        let r = FlashNextRules::prod().expect("rules");
        // FN38: the widest prefill rung is 8,192 rows (`--fn-max-chunk`)
        for m in 1..=8192 {
            let b = r.hc_band(m).expect("a band");
            assert!(m <= b.max_m);
        }
        assert!(r.hc_band(0).is_err());
        assert!(
            r.hc_band(8193).is_err(),
            "above the widest prefill rung is a refusal"
        );
        // and a row of a 2,049..8,192-row chunk takes the 2,048-row chunk's class
        assert_eq!(r.hc_band(2048).unwrap(), r.hc_band(8192).unwrap());
    }

    #[test]
    fn the_hc_launch_law_is_two_per_mix_four_per_layer() {
        let r = FlashNextRules::prod().expect("rules");
        assert_eq!(r.hc_launches_per_layer(false), 4);
        assert_eq!(r.hc_launches_per_layer(true), 5);
        // 48 layers + the final mixer: the prior fork's 486 HC launches a step become 195.
        let step = 47 * r.hc_launches_per_layer(false) + r.hc_launches_per_layer(true) + 2;
        assert_eq!(step, 195);
        // NEGATIVE CONTROL: a table that claims three launches a mix is refused.
        let bad = PROD_RULES_TEXT.replace("launches_per_mix = 2", "launches_per_mix = 3");
        assert!(FlashNextRules::from_toml(&bad).is_err());
    }

    #[test]
    fn a_mix_plan_sizes_its_scratch_from_the_band() {
        let r = FlashNextRules::prod().expect("rules");
        for m in [1u32, 5, 16, 17, 64, 2048] {
            let b = r.hc_band(m).unwrap().clone();
            let p = r.mix_plan(MixMode::Combine, MixOut::Q8, m).unwrap();
            assert_eq!(p.down.grid[0], 4 * b.sps);
            let d =
                r.hc.down_v2
                    .iter()
                    .find(|d| Some(&d.cfg) == b.down_v2.as_ref());
            let u = r.hc.up_v2.iter().find(|u| Some(&u.cfg) == b.up_v2.as_ref());
            let dmf = d.map_or(b.down_mf, |d| d.mf);
            let umf = u.map_or(b.up_mf, |u| u.mf);
            let gx = u.map_or(40, |u| u.grid_x);
            assert_eq!(p.down.grid[1], m.div_ceil(16 * dmf));
            assert_eq!(p.down.grid[2], d.map_or(1, |d| d.groups));
            assert_eq!(p.down_counters, p.down.grid[1] * p.down.grid[2]);
            assert_eq!(p.up.grid, [gx, m.div_ceil(16 * umf), 1]);
            assert_eq!(p.part_bytes, 4 * u64::from(4 * b.sps) * u64::from(m) * 336);
            assert_eq!(p.amax_bytes, 4 * u64::from(gx) * u64::from(m));
            assert_eq!(p.q8_bytes, u64::from(m) * 2560);
            match d {
                Some(d) => assert_eq!(p.down.symbol, format!("fn_hc_down_combine_v2_{}", d.cfg)),
                None => assert!(p.down.symbol.ends_with(&format!("_mf{}", b.down_mf))),
            }
            match u {
                Some(u) => assert_eq!(p.up.symbol, format!("fn_hc_up_q8_v2_{}", u.cfg)),
                None => assert!(p.up.symbol.ends_with(&format!("_mf{}", b.up_mf))),
            }
            let t = r.mix_plan(MixMode::Combine, MixOut::Q8Tiled, m).unwrap();
            assert_eq!(t.q8_bytes, u64::from(m.div_ceil(16) * 16) * 2560);
            let f = r.mix_plan(MixMode::Final, MixOut::Bf16, m).unwrap();
            assert!(f.down.symbol.contains("_final_"));
            assert_eq!((f.up_counters, f.amax_bytes, f.q8_bytes), (0, 0, 0));
        }
        assert!(r.mix_plan(MixMode::Final, MixOut::Q8, 1).is_err());
    }

    #[test]
    fn a_v2_route_must_divide_its_k_slice() {
        let r = FlashNextRules::prod().expect("rules");
        assert!(r
            .hc
            .band
            .iter()
            .all(|b| b.down_v2.is_some() && b.up_v2.is_some()));
        // NEGATIVE CONTROL: a slab chunk that does not divide the band's K slice is refused, and
        // an unknown cfg is refused.
        let bad = PROD_RULES_TEXT.replace("kc = 640", "kc = 768");
        assert!(FlashNextRules::from_toml(&bad).is_err());
        let bad = PROD_RULES_TEXT.replacen("down_v2 = \"", "down_v2 = \"nope", 1);
        assert!(FlashNextRules::from_toml(&bad).is_err());
        // the shipped symbols stay routable: a table without v2 routes plans the FN5 pair
        let mut t = r.clone();
        for b in &mut t.hc.band {
            b.down_v2 = None;
            b.up_v2 = None;
        }
        let p = t.mix_plan(MixMode::Combine, MixOut::Q8, 1).unwrap();
        assert_eq!(p.down.symbol, "fn_hc_down_combine_mf1");
        assert_eq!(p.up.grid[0], 40);
    }

    #[test]
    fn every_symbol_is_in_the_committed_code_object() {
        let r = FlashNextRules::prod().expect("rules");
        let man =
            std::fs::read_to_string(crate::prebuilt::dir().join("flashnext_hc.manifest.toml"))
                .expect("the flashnext_hc manifest");
        let dflash =
            std::fs::read_to_string(crate::prebuilt::dir().join("dflash_bgemm.manifest.toml"))
                .expect("the dflash_bgemm manifest");
        let mut syms = r.symbols();
        syms.extend(r.kernels.gate_only.iter().cloned());
        for g in &r.bgemm {
            for (_, s, _) in &g.bands {
                syms.push(s.clone());
            }
        }
        for s in syms {
            let hay = match r.plugin_for(&s) {
                Some("flashnext_hc") => &man,
                Some("dflash_bgemm") => &dflash,
                other => panic!("{s}: plugin {other:?}"),
            };
            assert!(
                hay.contains(&format!("\"{s}\"")),
                "{s} is not in its code object"
            );
        }
    }

    /// FN24: the one bf16 symbol with a row cap (`dflash_bgemm_bf16_m16n16` ignores grid y) runs
    /// its class above 16 rows as the row-slice twin; every other symbol is its own at any M.
    #[test]
    fn a_class_cell_above_its_row_cap_runs_its_row_slice_twin() {
        assert_eq!(
            class_symbol_at("dflash_bgemm_bf16_m16n16", 16),
            "dflash_bgemm_bf16_m16n16"
        );
        assert_eq!(
            class_symbol_at("dflash_bgemm_bf16_m16n16", 17),
            "dflash_bgemm_bf16_mtn16"
        );
        assert_eq!(
            band_rows("dflash_bgemm_bf16_mtn16"),
            band_rows("dflash_bgemm_bf16_m16n16")
        );
        for s in [
            "fn_bgemm_sk_m16",
            "dflash_bgemm_bf16_m32n16",
            "fn_bgemm_sk_m64",
        ] {
            assert_eq!(class_symbol_at(s, 40), s);
        }
    }

    #[test]
    fn bgemm_routes_resolve_and_refuse() {
        let r = FlashNextRules::prod().expect("rules");
        let (l, sp) = r.bgemm_route("gdn.out_proj.tp1", 1, 2560).unwrap();
        assert_eq!(l.grid, [40, 1, sp]);
        let (l, _) = r.bgemm_route("gdn.in_proj.tp1", 2048, 16480).unwrap();
        assert!(l.symbol.starts_with("fn_bgemm_lds_"));
        assert!(
            r.bgemm_route("gdn.in_proj.tp1", 1, 16000).is_err(),
            "an N off the table"
        );
        assert!(r.bgemm_route("nope", 1, 2560).is_err());
        assert!(
            r.bgemm_route("head.tp2", 65, 124160).is_err(),
            "the head runs <= 64 rows"
        );
        // NEGATIVE CONTROL: a split on a symbol that cannot split K is refused at parse.
        let bad = PROD_RULES_TEXT.replace(
            "[16, \"dflash_bgemm_bf16_m16n16\", 1]",
            "[16, \"dflash_bgemm_bf16_m16n16\", 2]",
        );
        assert!(FlashNextRules::from_toml(&bad).is_err());
    }
}
