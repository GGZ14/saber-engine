//! **P27-W2-5: the GDN decode update as BASE + STASH + REPLAY** (`--engine-gdn-replay on`, an ARM).
//!
//! The plan law of the arm, shared by the engine (`device::open`) and the `saber verify-step`
//! chain gate so the two cannot drift: every decode plan's recurrent update
//! (`saber_gdn_recurrent_update_f16s_k128_v128_ab1_norm0_vs*`) becomes
//! `p27w2_gdn_replay_f16s_k128_v128_ab1_vs*` at the plan's own VS and scalars
//! ([`saber_kernels::saber::gdn_replay_from_recurrent`]); `naccept` gives way to the session table
//! `rp` = {rows to replay, tag, stash parity}, the layer's stash pool and the refusal word. The
//! kernel's byte law and its fail-closed refusals are LEDGER / GATES P27-W2-5.

use std::collections::{BTreeMap, BTreeSet};

use saber_exec::layer::{BlockPlan, Route, Source, StateAt};
use saber_kernels::saber;

/// The stash pools' ledger class (one pool a GDN layer, `max_seqs` slot sets each).
pub const STASH_CLASS: &str = "engine.gdn.replay_stash";
/// The refusal word's state key (one word, every layer's launches).
pub const ERR_KEY: &str = "gdn.replay_err";

/// The state key of layer `l`'s stash pool.
#[must_use]
pub fn stash_key(l: u32) -> String {
    format!("l{l}.gdn.replay_stash")
}

/// Rewrite every GDN recurrent update in `plan` to the replay form, by name. `sx_div` is the slot
/// set width the plan's bases are spaced by. Returns the GDN layers rewritten.
///
/// A plan that carries a snapshot-law GDN decode form the replay cannot carry -- the fused update,
/// a fused-norm recurrent, a tree update, a FOLDED recurrent -- is refused: the two laws cannot
/// share a session.
///
/// # Errors
/// The refusals above, by name, and a planned step whose bindings the replay form does not name.
pub fn rewrite_plan(plan: &mut BlockPlan<'_>, sx_div: u32) -> Result<BTreeSet<u32>, String> {
    rewrite_plan_as(plan, sx_div, false)
}

/// [`rewrite_plan`], with `lds` = CMP27T's conc-4 form of the replay launch
/// (`saber::gdn_replay_c4_form`: the LDS-staged twin and / or another VS by batch; the same
/// kernargs, the shipped replay's bytes).
///
/// # Errors
/// As [`rewrite_plan`].
pub fn rewrite_plan_as(
    plan: &mut BlockPlan<'_>,
    sx_div: u32,
    lds: bool,
) -> Result<BTreeSet<u32>, String> {
    let mut layers = BTreeSet::new();
    for st in &mut plan.steps {
        let name = st.op.name.clone();
        let Route::Kernel { descs, .. } = &mut st.route else {
            if name.ends_with(".gdn.recurrent_update") || name.ends_with(".gdn.fused_update") {
                return Err(format!(
                    "{name}: a FOLDED GDN update under --engine-gdn-replay on: refused by name"
                ));
            }
            continue;
        };
        let Some(first) = descs.first() else {
            continue;
        };
        let k = first.kernel.clone();
        if k.contains("gdn_fused_update") || k.contains("gdn_tree") {
            return Err(format!(
                "{name}: `{k}` writes per-row snapshots and has no replay form; under \
                 --engine-gdn-replay on every decode bucket runs the pair"
            ));
        }
        if !name.ends_with(".gdn.recurrent_update") {
            continue;
        }
        if descs.len() != 1 || !k.starts_with(saber::REPLAY_FROM_PREFIX) {
            return Err(format!(
                "{name}: `{k}` is not the f16 / bf16-ab / norm-free recurrent update the replay \
                 form is built from: refused by name"
            ));
        }
        let saber_model::graph::Site::Layer(layer) = st.op.site else {
            return Err(format!(
                "{name}: a GDN recurrent update outside a decoder layer"
            ));
        };
        let d = saber::gdn_replay_from_recurrent(&descs[0], sx_div, false)
            .map_err(|e| format!("{name}: {e}"))?;
        let d = if lds {
            saber::gdn_replay_c4_form(&d).map_err(|e| format!("{name}: {e}"))?
        } else {
            d
        };
        let mut old: BTreeMap<String, Source> = st.bindings.drain(..).collect();
        old.remove("naccept");
        let mut bindings = Vec::new();
        for slot in &d.args {
            let saber_kernels::rules::ArgValue::Buffer { buffer } = &slot.value else {
                continue;
            };
            let src = match buffer.as_str() {
                "rp" => Source::Index(format!("{name}.rp")),
                "stash" => Source::State {
                    key: stash_key(layer),
                    at: StateAt::Pool,
                },
                "err" => Source::State {
                    key: ERR_KEY.to_owned(),
                    at: StateAt::Pool,
                },
                b => old.remove(b).ok_or_else(|| {
                    format!("{name}: the replay form's `{b}` has no binding in the planned step")
                })?,
            };
            bindings.push((buffer.clone(), src));
        }
        if let Some((b, _)) = old.iter().next() {
            return Err(format!(
                "{name}: the planned step binds `{b}`, which the replay form does not name"
            ));
        }
        descs[0] = d;
        st.bindings = bindings;
        layers.insert(layer);
    }
    Ok(layers)
}

/// The `rp` table bytes for one step: `{rows to replay, tag, parity}` a slot, `slots` slots.
#[must_use]
pub fn rp_bytes(rows: &[(u32, u32, u32)], slots: usize) -> Vec<u8> {
    let mut v = vec![0i32; 3 * slots.max(rows.len())];
    for (i, (n, tag, par)) in rows.iter().enumerate() {
        v[3 * i] = *n as i32;
        v[3 * i + 1] = *tag as i32;
        v[3 * i + 2] = (*par & 1) as i32;
    }
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rp_table_is_three_ints_a_slot_padded_with_nothing_to_replay() {
        let b = rp_bytes(&[(3, 7, 1), (0, 9, 2)], 4);
        let ints: Vec<i32> = b
            .chunks(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(ints, vec![3, 7, 1, 0, 9, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn the_replay_form_is_built_from_the_shipped_recurrent_descriptor_only() {
        use saber_kernels::rules::LaunchDescriptor;
        let bad = LaunchDescriptor {
            op: "gdn_recurrent_update".into(),
            plugin: "saber_gdn_recurrent_update".into(),
            kernel: "saber_gdn_recurrent_update_k128_v128_ab1_norm0_vs1".into(),
            grid: [8, 24, 1],
            block: [256, 1, 1],
            shared_bytes: 0,
            kernarg_bytes: 0,
            args: vec![],
            workspaces: vec![],
            n_launches: 1,
        };
        // the fp32-state symbol is refused by name (the replay form is the f16 state's)
        assert!(saber::gdn_replay_from_recurrent(&bad, 8, false)
            .unwrap_err()
            .to_string()
            .contains("f16-state"));
    }

    /// CMP27G: the materialize twin keeps the replay descriptor's signature and immediates
    /// (byte for byte), runs one sequence at VS 1 over the same heads, and refuses anything that is
    /// not the replay form (the shipped recurrent update is not).
    #[test]
    fn cmp27g_the_materialize_is_the_replay_descriptors_twin() {
        use saber_kernels::rules::{ArgValue, LaunchDescriptor, Slot};
        let rec = LaunchDescriptor {
            op: "gdn_recurrent_update".into(),
            plugin: "saber_gdn_recurrent_update".into(),
            kernel: format!("{}2", saber::REPLAY_FROM_PREFIX),
            grid: [8, 24, 2],
            block: [128, 1, 1],
            shared_bytes: 0,
            kernarg_bytes: 0,
            args: ["ab_stride", "st_slot", "st_head", "sidx_stride", "h", "hg"]
                .iter()
                .map(|n| Slot {
                    name: (*n).into(),
                    offset: 0,
                    size: 8,
                    value: ArgValue::I64 { ival: 48 },
                })
                .chain(["scale", "sp_thr"].iter().map(|n| Slot {
                    name: (*n).into(),
                    offset: 0,
                    size: 4,
                    value: ArgValue::F32 { fval: 0.5 },
                }))
                .collect(),
            workspaces: vec![],
            n_launches: 1,
        };
        assert!(saber::gdn_replay_materialize_like(&rec).is_err());
        let rp = saber::gdn_replay_from_recurrent(&rec, 8, false).unwrap();
        let m = saber::gdn_replay_materialize_like(&rp).unwrap();
        let want = saber::gdn_replay_from_recurrent(&rec, 8, true).unwrap();
        assert_eq!(m.kernel, saber::REPLAY_MATERIALIZE);
        assert_eq!(m.grid, [1, 24, 1]);
        assert_eq!(m.block, [256, 1, 1]);
        assert_eq!(m.args, rp.args);
        assert_eq!(m.kernarg_bytes, rp.kernarg_bytes);
        // == the materialize the oracle's builder makes, but at one sequence
        assert_eq!((m.kernel.clone(), m.args.clone()), (want.kernel, want.args));
        assert_eq!(want.grid, [8, 24, 1]);
        // CMP27T: the conc-4 form keeps the replay descriptor's kernargs; at 8 sequences it is the
        // shipped body at VS 1 (REPLAY_C4_TABLE's last row: unchanged), at 1 / 2 / 4 the LDS twin; it
        // materializes exactly as the shipped replay does and refuses a second pass or the
        // shipped recurrent update.
        let l = saber::gdn_replay_c4_form(&rp).unwrap();
        assert_eq!(l.kernel, format!("{}1", saber::REPLAY_PREFIX));
        assert_eq!((l.grid, l.block), ([8, 24, 1], [256, 1, 1]));
        assert_eq!((&l.args, l.kernarg_bytes), (&rp.args, rp.kernarg_bytes));
        let lm = saber::gdn_replay_materialize_like(&l).unwrap();
        assert_eq!(
            (lm.kernel.clone(), lm.plugin.clone(), lm.grid, lm.block),
            (m.kernel.clone(), m.plugin.clone(), m.grid, m.block)
        );
        assert_eq!(lm.args, m.args);
        for (n, lds, vs) in [(1, true, 2), (2, true, 2), (4, true, 1)] {
            let mut d4 = rp.clone();
            d4.grid[0] = n;
            let f = saber::gdn_replay_c4_form(&d4).unwrap();
            let want = if lds {
                saber::REPLAY_LDS_PREFIX
            } else {
                saber::REPLAY_PREFIX
            };
            assert_eq!(f.kernel, format!("{want}{vs}"));
            assert_eq!(f.plugin, saber::REPLAY_LDS_PLUGIN);
            assert_eq!((f.grid, f.block), ([n, 24, vs], [256 / vs, 1, 1]));
            assert_eq!(
                saber::gdn_replay_materialize_like(&f).unwrap().plugin,
                "p27w2_gdn_replay"
            );
            assert!(saber::gdn_replay_c4_form(&f).is_err());
        }
        assert!(saber::gdn_replay_c4_form(&rec).is_err());
    }
}
