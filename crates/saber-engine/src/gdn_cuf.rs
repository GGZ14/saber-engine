//! **CMP27AH: the batched GDN decode PAIR as ONE launch** (`--engine-step-arms cuf`, an ARM).
//!
//! Under the replay arm (`--engine-gdn-replay on`) every chain decode plan runs each GDN layer as
//! `saber_gdn_conv_update_*` then the replay update (`p27w2_gdn_replay[_lds]_*`). This module folds
//! the pair of every decode plan of `>= 2` sequences into `p27w2_gdn_cuf_*` (`hip/p27w2_gdn_cuf.hip`):
//! the conv step carries the fused descriptor, the recurrent step becomes [`Route::Fused`] into it
//! (one launch a layer fewer, `fused_launches` + 1 each), at the batches [`saber::CUF_VS_TABLE`]
//! fuses (measured: b4; b2 tied and b8 slower keep the pair, and the b1 plans -- the tree form and
//! a b1 chain plan -- keep it too).
//!
//! The race the fusion opens (CMP27L / CMP27Z): the q / k channels of a k head are read by
//! `H / Hg x VS` workgroups and the conv rewrites their window in place. The fused kernel's last
//! arriver rewrites it, counted in a per-layer zeroed region `l<L>.gdn.cuf_cnt` (`[max seqs][hg]`
//! u32, ledger class [`CNT_CLASS`]) that every launch leaves zero. The kernel's byte law (the pair's
//! bytes in every output) and its race stress are LEDGER CMP27AH.

use std::collections::{BTreeMap, BTreeSet};

use saber_exec::layer::{BlockPlan, Route, Source, StateAt};
use saber_kernels::saber;

/// The counter regions' ledger class (one region a fused GDN layer).
pub const CNT_CLASS: &str = "engine.gdn.cuf_counter";

/// The state key of layer `l`'s counter region.
#[must_use]
pub fn cnt_key(l: u32) -> String {
    format!("l{l}.gdn.cuf_cnt")
}

/// Bytes of one layer's counter region: one u32 a (sequence, k head).
#[must_use]
pub fn cnt_bytes(max_seqs: u32, key_heads: u32) -> u64 {
    u64::from(max_seqs.max(1)) * u64::from(key_heads.max(1)) * 4
}

/// Fold every GDN decode pair of `plan` whose launch runs at least `min_seqs` sequences, at a batch
/// [`saber::CUF_VS_TABLE`] fuses (b4 today), into the fused launch. Returns the GDN layers fused (empty: nothing in this plan qualified).
///
/// # Errors
/// A pair the fused form cannot carry, by name: halves that are not adjacent, a recurrent half not
/// already in the replay form, a conv `cache_idx` that is not the replay's `sidx`, or q / k / v bound
/// to different buffers by the two halves.
pub fn fuse_plan(plan: &mut BlockPlan<'_>, min_seqs: u32) -> Result<BTreeSet<u32>, String> {
    let mut layers = BTreeSet::new();
    let mut i = 0;
    while i < plan.steps.len() {
        let name = plan.steps[i].op.name.clone();
        let Some(stem) = name.strip_suffix(".conv_update") else {
            i += 1;
            continue;
        };
        let Route::Kernel { descs, .. } = &plan.steps[i].route else {
            i += 1;
            continue;
        };
        let Some(conv) = descs.first().cloned() else {
            i += 1;
            continue;
        };
        if conv.grid[0] < min_seqs || saber::cuf_vs(conv.grid[0]).is_none() {
            i += 1;
            continue;
        }
        let rec_name = format!("{stem}.recurrent_update");
        let Some(next) = plan.steps.get(i + 1) else {
            return Err(format!("{name}: no recurrent half after it (CMP27AH)"));
        };
        if next.op.name != rec_name {
            return Err(format!(
                "{name}: the step after it is `{}`, not its recurrent half `{rec_name}`: the fused \
                 route needs the pair adjacent (CMP27AH)",
                next.op.name
            ));
        }
        let Route::Kernel { descs: rdescs, .. } = &next.route else {
            return Err(format!("{rec_name}: not a kernel route (CMP27AH)"));
        };
        if rdescs.len() != 1 {
            return Err(format!(
                "{rec_name}: {} descriptors (CMP27AH)",
                rdescs.len()
            ));
        }
        let fused =
            saber::gdn_cuf_from_pair(&conv, &rdescs[0]).map_err(|e| format!("{name}: {e}"))?;
        let saber_model::graph::Site::Layer(layer) = plan.steps[i].op.site else {
            return Err(format!(
                "{name}: a GDN conv update outside a decoder layer (CMP27AH)"
            ));
        };
        let cb: BTreeMap<String, Source> = plan.steps[i].bindings.iter().cloned().collect();
        let rb: BTreeMap<String, Source> = plan.steps[i + 1].bindings.iter().cloned().collect();
        if cb.get("cache_idx") != rb.get("sidx") {
            return Err(format!(
                "{name}: the conv's `cache_idx` ({:?}) is not the replay's `sidx` ({:?}): the fused \
                 form reads one slot law (CMP27AH)",
                cb.get("cache_idx"),
                rb.get("sidx")
            ));
        }
        for b in ["q", "k", "v"] {
            if cb.get(b) != rb.get(b) {
                return Err(format!(
                    "{name}: the two halves bind `{b}` to different buffers (CMP27AH)"
                ));
            }
        }
        let mut bindings = Vec::new();
        for slot in &fused.args {
            let saber_kernels::rules::ArgValue::Buffer { buffer } = &slot.value else {
                continue;
            };
            let src = match buffer.as_str() {
                "cnt" => Source::State {
                    key: cnt_key(layer),
                    at: StateAt::Pool,
                },
                "x" | "wgt" | "bias" | "cstate" | "naccept" | "q" | "k" | "v" | "cu" => {
                    cb.get(buffer).cloned().ok_or_else(|| {
                        format!("{name}: the fused form's `{buffer}` has no conv binding (CMP27AH)")
                    })?
                }
                b => rb.get(b).cloned().ok_or_else(|| {
                    format!("{rec_name}: the fused form's `{b}` has no replay binding (CMP27AH)")
                })?,
            };
            bindings.push((buffer.clone(), src));
        }
        if let Route::Kernel { descs, .. } = &mut plan.steps[i].route {
            descs[0] = fused;
        }
        plan.steps[i].bindings = bindings;
        plan.steps[i + 1].route = Route::Fused {
            into: name.clone(),
            reason: "the conv update before it carries this recurrent update (CMP27AH, \
                     `--engine-step-arms cuf`): one launch reads each window, runs the replay \
                     update, and the last arriver of a k head's workgroups rewrites the shared \
                     q / k window",
        };
        plan.steps[i + 1].bindings = Vec::new();
        plan.fused_launches += 1;
        layers.insert(layer);
        i += 2;
    }
    Ok(layers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use saber_kernels::rules::{ArgValue, LaunchDescriptor, Slot};

    fn desc(kernel: &str, grid: [u32; 3], args: &[(&str, ArgValue)]) -> LaunchDescriptor {
        LaunchDescriptor {
            op: "x".into(),
            plugin: "x".into(),
            kernel: kernel.into(),
            grid,
            block: [256, 1, 1],
            shared_bytes: 0,
            kernarg_bytes: 0,
            args: args
                .iter()
                .map(|(n, v)| Slot {
                    name: (*n).into(),
                    offset: 0,
                    size: 8,
                    value: v.clone(),
                })
                .collect(),
            workspaces: vec![],
            n_launches: 1,
        }
    }

    fn i(v: i64) -> ArgValue {
        ArgValue::I64 { ival: v }
    }

    fn buf(n: &str) -> ArgValue {
        ArgValue::Buffer { buffer: n.into() }
    }

    fn pair(n: u32, vs: u32, lds: bool) -> (LaunchDescriptor, LaunchDescriptor) {
        let conv = desc(
            "saber_gdn_conv_update_w4_h128_hpb1",
            [n, 40, 1],
            &[
                ("x", buf("x")),
                ("xpitch", i(8192)),
                ("wgt", buf("wgt")),
                (
                    "bias",
                    ArgValue::NullBuffer {
                        buffer: "bias".into(),
                    },
                ),
                ("cstate", buf("cstate")),
                ("cs_seq", i(5120 * 12)),
                ("cs_dim", i(12)),
                ("cs_tok", i(1)),
                ("state_len_max", ArgValue::I32 { ival: 10 }),
                ("cache_idx", buf("cache_idx")),
                ("ci_stride", i(8)),
                ("naccept", buf("naccept")),
                ("h", ArgValue::I32 { ival: 24 }),
                ("hg", ArgValue::I32 { ival: 8 }),
                ("maxq", ArgValue::I32 { ival: 8 }),
            ],
        );
        let prefix = if lds {
            saber::REPLAY_LDS_PREFIX
        } else {
            saber::REPLAY_PREFIX
        };
        let rep = desc(
            &format!("{prefix}{vs}"),
            [n, 24, vs],
            &[
                ("ab_stride", i(48)),
                ("st_slot", i(24 * 128 * 128)),
                ("st_head", i(128 * 128)),
                ("sidx_stride", i(8)),
                ("sx_div", i(8)),
                ("sx_head", i(4224)),
                ("h", ArgValue::I32 { ival: 24 }),
                ("hg", ArgValue::I32 { ival: 8 }),
                ("scale", ArgValue::F32 { fval: 0.088 }),
                ("sp_thr", ArgValue::F32 { fval: 20.0 }),
            ],
        );
        (conv, rep)
    }

    /// The fused descriptor keeps both halves' scalars, takes the table's VS by sequences, the
    /// conv's absent bias stays absent, and anything that is not the pair is refused by name.
    #[test]
    fn cmp27ah_the_fused_descriptor_is_built_from_the_pair_only() {
        for (n, want_vs) in [(3u32, 1u32), (4, 1)] {
            for (vs, lds) in [(1, false), (2, true), (1, true)] {
                let (c, r) = pair(n, vs, lds);
                let f = saber::gdn_cuf_from_pair(&c, &r).expect("fused");
                assert_eq!(f.kernel, format!("{}{want_vs}", saber::CUF_PREFIX));
                assert_eq!(f.plugin, saber::CUF_PLUGIN);
                assert_eq!((f.grid, f.block), ([n, 24, want_vs], [256 / want_vs, 1, 1]));
                let g = |a: &str| f.arg(a).map(|s| s.value.clone());
                assert_eq!(g("cs_dim"), Some(i(12)));
                assert_eq!(g("state_len_max"), Some(ArgValue::I32 { ival: 10 }));
                assert_eq!(g("maxq"), Some(ArgValue::I32 { ival: 8 }));
                assert_eq!(g("sx_head"), Some(i(4224)));
                assert_eq!(g("scale"), Some(ArgValue::F32 { fval: 0.088 }));
                assert!(matches!(g("bias"), Some(ArgValue::NullBuffer { .. })));
                assert!(matches!(g("naccept"), Some(ArgValue::Buffer { .. })));
                assert!(matches!(g("cnt"), Some(ArgValue::Buffer { .. })));
                assert_eq!(f.n_launches, 1);
            }
        }
        let (c, r) = pair(4, 1, true);
        // the conv half as the replay half, and the reverse
        assert!(saber::gdn_cuf_from_pair(&r, &r).is_err());
        assert!(saber::gdn_cuf_from_pair(&c, &c).is_err());
        // the batches that keep the pair (measured: tied at 2, slower at 8)
        for n in [1u32, 2, 8] {
            let (c, r) = pair(n, 1, false);
            assert!(saber::cuf_vs(n).is_none());
            assert!(saber::gdn_cuf_from_pair(&c, &r).is_err());
        }
        // halves of different grids
        let (c2, _) = pair(2, 1, true);
        assert!(saber::gdn_cuf_from_pair(&c2, &r).is_err());
        // a conv cache stride that is not the replay's sidx stride
        let mut c3 = c.clone();
        for s in &mut c3.args {
            if s.name == "ci_stride" {
                s.value = i(1);
            }
        }
        assert!(saber::gdn_cuf_from_pair(&c3, &r)
            .unwrap_err()
            .to_string()
            .contains("sidx_stride"));
        // the shipped (non-replay) recurrent update is not a half the fused form takes
        let mut r4 = r.clone();
        r4.kernel = "saber_gdn_recurrent_update_f16s_k128_v128_ab1_norm0_vs1".into();
        assert!(saber::gdn_cuf_from_pair(&c, &r4).is_err());
    }

    #[test]
    fn cmp27ah_the_counter_region_is_one_word_a_sequence_and_k_head() {
        assert_eq!(cnt_bytes(8, 8), 256);
        assert_eq!(cnt_bytes(0, 8), 32);
        assert_eq!(cnt_key(3), "l3.gdn.cuf_cnt");
    }
}
