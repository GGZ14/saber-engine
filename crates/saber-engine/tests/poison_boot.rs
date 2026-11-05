//! T6's boot poison (`saber engine --poison-boot SET[+SET..]:BYTE`): the argument's law, on the
//! host. The card-side use is `tools/gates/t5_stress.sh --hold poison` (LEDGER T6).

use saber_engine::device::PoisonBoot;

#[test]
fn the_poison_sets_parse_and_compose() {
    let p = PoisonBoot::parse("arenas:0x3c").expect("arenas");
    assert!(p.prefill && p.decode && p.heads && p.aux && !p.kv);
    assert_eq!(p.byte, 0x3c);
    let a = PoisonBoot::parse("all:165").expect("all");
    assert!(a.kv && a.prefill && a.byte == 165);
    let d = PoisonBoot::parse("decode+aux:0xff").expect("decode+aux");
    assert!(d.decode && d.aux && !d.prefill && !d.heads);
    let b = PoisonBoot::parse("buf=layer.0.gdn.conv_prep.gates:0x3c").expect("buf");
    assert_eq!(b.buffer.as_deref(), Some("layer.0.gdn.conv_prep.gates"));
    assert!(!b.prefill && !b.decode);
    // T7: substring sets (repeatable) and the rows restriction.
    let m = PoisonBoot::parse("match=.gdn.+match=resid@256:0x3c").expect("match");
    assert_eq!(m.matching, vec![".gdn.".to_owned(), "resid".to_owned()]);
    assert_eq!(m.rows, Some(256));
    assert!(m.buffer.is_none() && !m.prefill);
    let r = PoisonBoot::parse("buf=l0.gdn.conv@96:0").expect("buf@rows");
    assert_eq!(
        (r.buffer.as_deref(), r.rows),
        (Some("l0.gdn.conv"), Some(96))
    );
    assert!(PoisonBoot::parse("match=:1").is_err());
    assert!(PoisonBoot::parse("buf=x@many:1").is_err());
    // Controls: an unknown set, a missing byte, a byte past u8.
    assert!(PoisonBoot::parse("arena:0x3c").is_err());
    assert!(PoisonBoot::parse("prefill").is_err());
    assert!(PoisonBoot::parse("prefill:256").is_err());
    assert!(PoisonBoot::parse("buf=:1").is_err());
}

/// T8: the sets OUTSIDE the target's arenas parse alone, compose, and `outside` is all eight --
/// and none of them turns an arena set on (the control: `arenas` turns none of them on).
#[test]
fn the_outside_sets_parse_and_stay_outside() {
    let o = PoisonBoot::parse("outside:0x3c").expect("outside");
    assert!(o.drafter && o.dkv && o.dtables && o.gdn && o.ws && o.scratch && o.headbufs && o.index);
    assert!(!o.prefill && !o.decode && !o.heads && !o.aux && !o.kv);
    let g = PoisonBoot::parse("gdn+dkv:0xff").expect("gdn+dkv");
    assert!(g.gdn && g.dkv && !g.drafter && !g.ws && g.byte == 0xff);
    for (set, pick) in [
        ("drafter", 0usize),
        ("dkv", 1),
        ("dtables", 2),
        ("gdn", 3),
        ("ws", 4),
        ("scratch", 5),
        ("headbufs", 6),
        ("index", 7),
    ] {
        let p = PoisonBoot::parse(&format!("{set}:0x3c")).expect(set);
        let on = [
            p.drafter, p.dkv, p.dtables, p.gdn, p.ws, p.scratch, p.headbufs, p.index,
        ];
        assert_eq!(on.iter().filter(|b| **b).count(), 1, "{set} alone");
        assert!(on[pick], "{set} turns itself on");
    }
    let a = PoisonBoot::parse("all:0x3c").expect("all");
    assert!(
        !(a.drafter || a.dkv || a.dtables || a.gdn || a.ws || a.scratch || a.headbufs || a.index)
    );
    assert!(PoisonBoot::parse("outsides:0x3c").is_err());
}
