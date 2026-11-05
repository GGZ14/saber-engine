//! P27-1d: the SPLIT rotated all-reduce's descriptors, on the host -- what `--ar-form split`
//! derives from the one-shot the rules resolved, and what it refuses. The card half (the byte law
//! in the engine, ids / blocks identical to the one-shot control) is the engine gate under
//! `--ar-form split`; P27-1c's lab is the op-level one.

use saber_kernels::manifest::Manifest;
use saber_kernels::prebuilt;
use saber_kernels::rules::ArgValue;
use saber_kernels::saber::{
    self, ArPayload, ArShape, Op, SaberRules, Shape, AR_SPLIT_FLAG_WORD, AR_SPLIT_PUSH_THREADS,
};
use saber_kernels::LaunchDescriptor;

fn oneshot(rows: u64) -> LaunchDescriptor {
    let r = SaberRules::prod_saber().expect("saber rules");
    saber::resolve(
        Op::AllReduce {
            dtype: ArPayload::Bf16,
        },
        Shape::Ar(ArShape {
            n_elem: rows * 5120,
            elem_bytes: 2,
            world_size: 2,
        }),
        &r,
    )
    .expect("the one-shot")
}

fn scalar(d: &LaunchDescriptor, name: &str) -> i64 {
    match d.arg(name).map(|s| &s.value) {
        Some(ArgValue::I32 { ival } | ArgValue::I64 { ival }) => *ival,
        other => panic!("{}: `{name}` is {other:?}", d.kernel),
    }
}

#[test]
fn the_split_carries_every_one_shot_argument_and_walks_its_virtual_blocks() {
    let one = oneshot(1024);
    assert_eq!(one.kernel, "saber_ar_oneshot_2rank_wht6_bf16");
    let (push, red) = saber::ar_wht_split(&one, 16).expect("the split");
    assert_eq!(push.kernel, "saber_ar_split_2rank_wht6_bf16_push_nb");
    assert_eq!(red.kernel, "saber_ar_split_2rank_wht6_bf16_reduce");
    assert_eq!(push.plugin, "saber_ar_wht_split");
    assert_eq!(red.plugin, "saber_ar_wht_split");
    assert_eq!(push.op, one.op, "the same op: the same protocol regions");
    assert_eq!(push.grid, [16, 1, 1]);
    assert_eq!(push.block, [AR_SPLIT_PUSH_THREADS, 1, 1]);
    assert_eq!(
        red.grid, one.grid,
        "the reduce runs at the one-shot's blocks"
    );
    assert_eq!(red.block, one.block);
    assert_eq!(scalar(&push, "nbv"), i64::from(one.grid[0]));
    assert_eq!(scalar(&push, "flag_val"), 0, "set per call by the executor");
    for d in [&push, &red] {
        assert_eq!(d.workspaces, one.workspaces);
        assert_eq!(d.n_launches, 1);
        for s in &one.args {
            let got = d.arg(&s.name).expect("carried");
            assert_eq!(got.value, s.value, "{}: `{}`", d.kernel, s.name);
            assert_eq!(got.offset, s.offset, "the one-shot's leading ABI");
        }
    }
    // The descriptors agree with the committed code object's own metadata.
    let m = Manifest::load(prebuilt::dir().join("saber_ar_wht_split.manifest.toml")).expect("m");
    for d in [&push, &red] {
        let k = m.kernel(&d.kernel).expect("in the object");
        assert!(d.kernarg_bytes <= k.kernarg_segment_size);
        assert!(d.block[0] <= k.max_flat_workgroup_size);
        for s in &d.args {
            let a = k
                .arg
                .iter()
                .find(|a| a.name.as_deref() == Some(s.name.as_str()))
                .expect("declared");
            assert_eq!((a.offset, a.size), (s.offset, s.size), "{}", s.name);
        }
    }
    assert_eq!(
        saber::plugin_for("saber_ar_split_2rank_wht6_bf16_push_nb").unwrap(),
        "saber_ar_wht_split"
    );
    assert_eq!(
        saber::plugin_for("saber_ar_oneshot_2rank_wht6_bf16").unwrap(),
        "saber_ar_wht",
        "the served one-shot's owner is unchanged"
    );
    let rules = SaberRules::prod_saber().expect("saber rules");
    assert!(
        AR_SPLIT_FLAG_WORD >= u64::from(rules.ar.max_nb)
            && AR_SPLIT_FLAG_WORD < u64::from(rules.ar.max_blocks),
        "the split flag word is outside every one-shot block's flag"
    );
}

#[test]
fn the_split_refuses_what_it_cannot_reproduce() {
    let one = oneshot(1024);
    assert!(saber::ar_wht_split(&one, 0).is_err(), "no physical block");
    assert!(
        saber::ar_wht_split(&one, one.grid[0] + 1).is_err(),
        "more physical than virtual blocks"
    );
    assert!(saber::ar_wht_split(&one, one.grid[0]).is_ok());
    // The exact family (a decode-sized message) has no split twin.
    let small = oneshot(8);
    assert!(small.kernel.contains("exact"));
    assert!(saber::ar_wht_split(&small, 4).is_err());
}
