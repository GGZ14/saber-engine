//! Every `Buffer` kernarg's ELEMENT WIDTH against the dtype the capture binds to that slot.
//!
//! `every_f32_kernarg_is_a_value_and_not_a_pointer` asks a signature which slots are SCALARS and
//! stops there. Nothing asked what one element of a pointer slot is, and the last two kernel
//! defects in this tree were exactly that, one per round:
//!
//! * **F22, the walk**: `candidates` was `const int *` over `torch.topk`'s **i64** table. Every
//!   address was right and every token was wrong.
//! * **F24, the selector**: `cand` `const int *` over an i64 buffer, `unary` `const __bf16 *`
//!   over the exact rerank's **f32**, and `out` `__bf16 *` into an **f32** tensor — 5007 / 5377 /
//!   5133 differing bytes of 7168, reproduced to the byte from the widths alone.
//!
//! Both read the right addresses and the wrong bytes, and both survived three gate rounds,
//! because the gates CONVERTED the reference's dtypes down to the kernel's and said so in a NOTE.
//! A gate that converts its reference's dtypes to its subject's is not gating the dtypes.
//!
//! So this test takes the other side: for every capture in `.cache/` that this file knows how to
//! read, each named tensor is held against the element width [`abi::SigArg::elem`] declares for
//! the slot it is bound to, and a disagreement is refused BY OP AND SLOT. It needs no device, no
//! container and no lock — it is JSON and a table — and it is inert (with a printed inventory)
//! wherever the captures are absent, which is every checkout but this box.
//!
//! The width comes from the dtype's ITEMSIZE, never from its name: prod's fp8 KV cache is
//! allocated `uint8` and `.view(fp8)`d at the point of use, so `"float8" in dtype` is false for
//! the very tensor that is one byte per element (`an_fp8_kv_cache_stored_as_uint8_is_one_byte_per
//! _element`, F23). `uint8` and `float8_e4m3fn` are both 1 here, which is the fact that matters.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use saber_kernels::abi::{self, ArgKind, SigArg};
use saber_kernels::dflash::{DflashOp, DflashRules};
use serde_json::Value;

/// How a captured tensor reaches a kernarg slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bind {
    /// The slot is bound at the capture's own element width. The two must be EQUAL.
    Same,
    /// The launcher CONVERTS an index table on the way in, i64 → i32, and the reason says where.
    ///
    /// Allowed only from an 8-byte INTEGER dtype to a 4-byte slot. A float narrowing cannot be
    /// declared here at all, which is the point: two of the selector's three defects were f32
    /// buffers read or written as bf16, and this escape hatch must not cover them.
    Narrowed(&'static str),
    /// Saber's own output is WIDER than prod's tensor and the comparison widens prod's before
    /// comparing. Allowed only float → wider float, and the reason says who converts.
    Widened(&'static str),
}

/// One captured tensor bound to one kernarg slot.
struct Slot {
    /// The kernarg name, as the signature declares it.
    slot: &'static str,
    /// The manifest key, with `{l}` expanded over `layers`.
    key: &'static str,
    bind: Bind,
}

const fn s(slot: &'static str, key: &'static str, bind: Bind) -> Slot {
    Slot { slot, key, bind }
}

/// One op's bindings inside one capture.
struct Case {
    /// The capture directory under `<repo>/.cache/`.
    dir: &'static str,
    /// The manifest inside it. `{tag}` is expanded over `index.json`'s `prompts`, if it has any.
    manifest: &'static str,
    op: DflashOp,
    /// How many layers `{l}` ranges over; 0 when the keys carry no layer.
    layers: u32,
    slots: &'static [Slot],
}

use Bind::{Narrowed, Same, Widened};

/// THE TABLE. Every row is a claim about what the launcher binds, and every row is checkable.
///
/// What is deliberately NOT here: a slot whose buffer is Saber's own (the selector's `hidden`, the
/// context rope's `k`) or comes from the PACK rather than the capture (the two codebooks, every
/// weight). Those are covered by the pack gate, which compares them by record name and dtype.
const CASES: &[Case] = &[
    // ---- the proposal capture: prod's own drafter, one whole decode step -------------------
    Case {
        dir: "capture-dflash-proposal-r28",
        manifest: "{tag}.json",
        op: DflashOp::AttnFp8Kv,
        layers: 5,
        slots: &[
            s("q", "attn.l{l}.q", Same),
            // The interleaved `[blocks, kv_heads, block_size, 2 * head_size]` cache prod
            // allocates as uint8. ONE byte per element, and the dtype NAME does not say so.
            s("kc", "kv.after_fwd.l{l}", Same),
            s("vc", "kv.after_fwd.l{l}", Same),
            s("block_table", "attn.l{l}.block_table", Same),
            s("seq_lens", "attn.l{l}.seq_lens", Same),
            s("k_scale", "attn.l{l}.k_scale", Same),
            s("v_scale", "attn.l{l}.v_scale", Same),
            s("out", "attn.l{l}.out", Same),
        ],
    },
    Case {
        dir: "capture-dflash-proposal-r28",
        manifest: "{tag}.json",
        op: DflashOp::SelectorScore,
        layers: 0,
        slots: &[
            // The three that were wrong until F24, each against prod's own tensor.
            s("cand", "head.candidate_ids", Same),
            s("unary", "head.unary_logits", Same),
            s("out", "walk.scores", Same),
            s("anchor", "sel.anchor_token_ids", Same),
        ],
    },
    Case {
        dir: "capture-dflash-proposal-r28",
        manifest: "{tag}.json",
        op: DflashOp::SelectorWalk,
        layers: 0,
        slots: &[
            s("scores", "walk.scores", Same),
            // The one that was wrong until F22.
            s("candidates", "walk.candidate_ids", Same),
            s("tokens", "walk.draft_tokens", Same),
            s("realized", "walk.realized_scores", Same),
            s("req_state", "walk.idx_mapping", Same),
        ],
    },
    Case {
        dir: "capture-dflash-proposal-r28",
        manifest: "{tag}.json",
        op: DflashOp::RerankExactBf16,
        layers: 0,
        slots: &[
            s("x", "head.hidden", Same),
            // `compute_candidates` hands `_rerank_exact` ATen's i64 topk indices and this kernel
            // indexes head rows with an `int`. The LAUNCHER converts -- `draft_proposal.rs`
            // builds an `idx32` table element by element -- rather than reinterpreting the
            // buffer, and at a 248,320-row head the conversion is value-preserving. That is the
            // whole difference between this row and F22's defect, which reinterpreted.
            s(
                "idx",
                "head.candidate_ids",
                Narrowed(
                    "draft_proposal.rs builds idx32 from the \
               i64 candidate table; vocab-local row indices fit in an i32",
                ),
            ),
            s("out", "head.unary_logits", Same),
        ],
    },
    Case {
        dir: "capture-dflash-proposal-r28",
        manifest: "{tag}.json",
        op: DflashOp::CtxRmsNorm,
        layers: 0,
        slots: &[
            s("x", "ctx.states", Same),
            s("weight", "ctx.hidden_norm_w", Same),
        ],
    },
    Case {
        dir: "capture-dflash-proposal-r28",
        manifest: "{tag}.json",
        op: DflashOp::CtxRope,
        layers: 0,
        slots: &[
            // The reference server stores positions as i64 and the kernel takes `const long *`.
            s("positions", "ctx.positions", Same),
            s("cos_sin", "ctx.rope_cos_sin", Same),
        ],
    },
    // ---- the standalone attention capture, both KV widths -----------------------------------
    Case {
        dir: "capture-dflash-attn",
        manifest: "attn.json",
        op: DflashOp::AttnFp8Kv,
        layers: 0,
        slots: &[
            s("q", "q", Same),
            s("kc", "fp8.k", Same),
            s("vc", "fp8.v", Same),
            s("block_table", "block_table", Same),
            s("seq_lens", "seqused_k", Same),
            s("out", "fp8.out", Same),
        ],
    },
    Case {
        dir: "capture-dflash-attn",
        manifest: "attn.json",
        op: DflashOp::AttnBf16Kv,
        layers: 0,
        slots: &[
            s("q", "q", Same),
            s("kc", "bf16.k", Same),
            s("vc", "bf16.v", Same),
            s("block_table", "block_table", Same),
            s("seq_lens", "seqused_k", Same),
            s("out", "bf16.out", Same),
        ],
    },
    // ---- the KV append and the selector's top-k ---------------------------------------------
    Case {
        dir: "capture-dflash-kv",
        manifest: "kv.json",
        op: DflashOp::KvAppendFp8,
        layers: 0,
        slots: &[
            s("key", "kv.fp8.key", Same),
            s("value", "kv.fp8.value", Same),
            s("kc", "kv.fp8.kc", Same),
            s("vc", "kv.fp8.vc", Same),
            s("slots", "kv.fp8.slots", Same),
        ],
    },
    Case {
        dir: "capture-dflash-kv",
        manifest: "kv.json",
        op: DflashOp::KvAppendBf16,
        layers: 0,
        slots: &[
            s("key", "kv.bf16.key", Same),
            s("value", "kv.bf16.value", Same),
            s("kc", "kv.bf16.kc", Same),
            s("vc", "kv.bf16.vc", Same),
            s("slots", "kv.bf16.slots", Same),
        ],
    },
    Case {
        dir: "capture-dflash-kv",
        manifest: "kv.json",
        op: DflashOp::SelectorTopk,
        layers: 0,
        slots: &[
            s("x", "topk.sel16.x", Same),
            // `torch.topk` returns the INPUT dtype, so prod's values are bf16 and this kernel
            // writes f32; `dflash_attn_bench.py` widens prod's before comparing, which is lossless
            // in that direction. The kernel's own comment says the same thing.
            s(
                "vals",
                "topk.sel16.v",
                Widened(
                    "ATen returns the input dtype; the bench widens \
               prod's bf16 to f32 and compares there",
                ),
            ),
            // `idx` IS NOT BOUND HERE, and that is the point of the row below it. This capture is
            // SYNTHETIC -- random rows with ties planted -- and it stored `i.int()`, so its index
            // dtype is the bench's own convention rather than a fact about prod. Binding a
            // pointer's element width against a tensor the harness narrowed is the exact mistake
            // this file exists to catch, so the claim moved to the proposal capture, where the
            // tensor is `torch.topk`'s own i64 out of prod's own drafter.
        ],
    },
    // ---- ... and the selection's OUTPUT WIDTH, against prod's own indices --------------------
    //
    // The two ATen selections were first tapped in r26 (F26 added them with the eleven absent
    // sites); r23 carries neither, and BOTH of those captures are gone -- main's tracked
    // `.cache` symlink took the shared cache with it on 2026-09-21 (see the LEDGER). Every case
    // in this table reads r28 now. A case whose keys are absent contributes no pairs and refuses
    // nothing, which is how this table goes quietly inert: the printed per-directory counts are
    // what make that visible, and they are the first thing to read when it says 0 refused.
    Case {
        dir: "capture-dflash-proposal-r28",
        manifest: "{tag}.json",
        op: DflashOp::SelectorTopk,
        layers: 0,
        slots: &[
            s("x", "topk.1.x", Same),
            // The edge F28 closed. `dflash_topk_bf16_k16` wrote i32 through F27 while its only
            // consumer, `dflash_selector_score_r256_k16`, declares `cand` as `const long *` --
            // so an unseeded chain had to re-widen between two launches that are one edge. The
            // selection writes what ATen returns now, and this row is what says so.
            s("idx", "topk.1.indices", Same),
            s(
                "vals",
                "topk.1.values",
                Widened(
                    "ATen returns the input dtype; prod's bf16 values \
               are widened to the f32 this kernel writes",
                ),
            ),
        ],
    },
    // ---- the drafter's OWN q/k norm + rope (F28) ---------------------------------------------
    //
    // `qkv` IS bound, because r28's eager shadow taps it (`chain.l{l}.qkv`) -- it is an
    // intermediate of a compiled piece and no earlier capture carries it. The one this kernel
    // could most easily get wrong is `pos`: `int64`, as every position table the reference server hands a kernel
    // is, and the rope reads it directly rather than through an index conversion.
    Case {
        dir: "capture-dflash-proposal-r28",
        manifest: "{tag}.json",
        op: DflashOp::QkNormRope,
        layers: 5,
        slots: &[
            s("qkv", "chain.l{l}.qkv", Same),
            s("cos_sin", "ctx.rope_cos_sin", Same),
            s("pos", "q.positions", Same),
            s("q_out", "attn.l{l}.q", Same),
            s("k_out", "attn.l{l}.k", Same),
        ],
    },
    // ---- the standalone selector, re-captured at prod's widths by `--only sel` ---------------
    Case {
        dir: "capture-dflash-sel-std",
        manifest: "selector.json",
        op: DflashOp::SelectorScore,
        layers: 0,
        slots: &[
            s("cand", "candidate_ids", Same),
            s("unary", "unary_logits", Same),
            s("anchor", "anchor_token_ids", Same),
            s("out", "out", Same),
        ],
    },
];

/// Bytes per element of a torch dtype NAME. Deliberately total: an unknown name is an error, not
/// a zero, because a zero would pass every comparison.
fn itemsize(dtype: &str) -> Option<u32> {
    Some(match dtype {
        "float64" | "int64" | "uint64" => 8,
        "float32" | "int32" | "uint32" => 4,
        "bfloat16" | "float16" | "int16" | "uint16" => 2,
        "int8" | "uint8" | "bool" | "float8_e4m3fn" | "float8_e5m2" | "float8_e4m3fnuz" => 1,
        _ => return None,
    })
}

fn is_int(dtype: &str) -> bool {
    dtype.starts_with("int") || dtype.starts_with("uint") || dtype == "bool"
}

fn is_float(dtype: &str) -> bool {
    dtype.starts_with("float") || dtype.starts_with("bfloat")
}

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

/// The dtype a manifest records for a key, if it records one.
fn dtype_of<'a>(man: &'a Value, key: &str) -> Option<&'a str> {
    man.get(key)?.get("dtype")?.as_str()
}

/// The element width a manifest RECORDS for a key. The capture scripts write `itemsize` (F25);
/// older captures have only the dtype name, which is why [`itemsize`] exists at all.
fn recorded_itemsize(man: &Value, key: &str) -> Option<u32> {
    u32::try_from(man.get(key)?.get("itemsize")?.as_u64()?).ok()
}

/// Check one op's slots against one manifest. Returns (checked, violations).
fn check(
    op: DflashOp,
    sym: &str,
    sig: &[SigArg],
    man: &Value,
    slots: &[Slot],
    layers: u32,
    where_: &str,
) -> (usize, Vec<String>) {
    let mut checked = 0usize;
    let mut bad = Vec::new();
    for sl in slots {
        let arg = sig.iter().find(|a| a.name == sl.slot);
        let Some(arg) = arg else {
            bad.push(format!(
                "{}: `{sym}` has no kernarg `{}` ({where_})",
                op.as_str(),
                sl.slot
            ));
            continue;
        };
        if arg.kind != ArgKind::Buffer {
            bad.push(format!(
                "{}: `{}` is a {:?} kernarg, not a pointer ({where_})",
                op.as_str(),
                sl.slot,
                arg.kind
            ));
            continue;
        }
        let Some(want) = arg.elem_bytes() else {
            bad.push(format!(
                "{}: `{sym}` declares NO element width for `{}`; every buffer slot a capture \
                 binds must declare one ({where_})",
                op.as_str(),
                sl.slot
            ));
            continue;
        };
        for l in 0..layers.max(1) {
            let key = sl.key.replace("{l}", &l.to_string());
            let Some(dt) = dtype_of(man, &key) else {
                continue; // this capture does not carry the tensor; another one may.
            };
            let Some(got) = itemsize(dt) else {
                bad.push(format!(
                    "{}: `{key}` has dtype `{dt}`, which this test does not know the width of \
                     ({where_})",
                    op.as_str()
                ));
                continue;
            };
            // Where the capture RECORDS its itemsize, the two derivations must agree. Two
            // independent statements of the same fact is the whole point of the field: a dtype
            // name that this table maps to the wrong width would otherwise be invisible.
            if let Some(rec) = recorded_itemsize(man, &key) {
                if rec != got {
                    bad.push(format!(
                        "{}: `{key}` records itemsize {rec} and its dtype `{dt}` is {got} bytes \
                         ({where_})",
                        op.as_str()
                    ));
                    continue;
                }
            }
            checked += 1;
            match sl.bind {
                Same if got == want => {}
                Same => bad.push(format!(
                    "{} slot `{}`: the capture binds `{key}` ({dt}, {got} bytes per element) and \
                     `{sym}` declares {want}. A pointer of the wrong element width reads the \
                     right addresses and the wrong bytes ({where_})",
                    op.as_str(),
                    sl.slot
                )),
                Narrowed(why) => {
                    if !(got == 8 && want == 4 && is_int(dt)) {
                        bad.push(format!(
                            "{} slot `{}`: `{key}` is {dt} ({got}) and the slot is {want} -- a \
                             `Narrowed` binding is only for an i64 INDEX table converted to i32. \
                             A float narrowed to a narrower float is a ROUNDING, and declaring it \
                             here is how the selector's `unary` and `out` hid for three rounds \
                             ({where_}); the reason given was: {why}",
                            op.as_str(),
                            sl.slot
                        ));
                    }
                }
                Widened(why) => {
                    if !(want > got && is_float(dt)) {
                        bad.push(format!(
                            "{} slot `{}`: `{key}` is {dt} ({got}) and the slot is {want} -- a \
                             `Widened` binding is only for a float output compared after a \
                             lossless widening ({where_}); the reason given was: {why}",
                            op.as_str(),
                            sl.slot
                        ));
                    }
                }
            }
        }
    }
    (checked, bad)
}

/// The manifests one case covers: `{tag}` over `index.json`'s prompts, or the file itself.
fn manifests(dir: &Path, pattern: &str) -> Vec<(String, Value)> {
    let mut out = Vec::new();
    let names: Vec<String> = if pattern.contains("{tag}") {
        let idx = std::fs::read_to_string(dir.join("index.json")).ok();
        let tags: Vec<String> = idx
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
            .and_then(|v| {
                v.get("prompts")?.as_array().map(|a| {
                    a.iter()
                        .filter_map(|s| s.as_str().map(String::from))
                        .collect()
                })
            })
            .unwrap_or_default();
        tags.iter().map(|t| pattern.replace("{tag}", t)).collect()
    } else {
        vec![pattern.to_string()]
    };
    for n in names {
        if let Ok(txt) = std::fs::read_to_string(dir.join(&n)) {
            if let Ok(v) = serde_json::from_str::<Value>(&txt) {
                out.push((n, v));
            }
        }
    }
    out
}

/// THE RULE. Every `Buffer` kernarg a capture binds is held against that capture's own dtype.
#[test]
fn every_buffer_kernarg_binds_the_element_width_the_capture_records() {
    let rules = DflashRules::prod_dflash().expect("the dflash rules parse");
    let cache = repo().join(".cache");
    let mut checked = 0usize;
    let mut cases = 0usize;
    let mut bad: Vec<String> = Vec::new();
    let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
    for case in CASES {
        let dir = cache.join(case.dir);
        if !dir.is_dir() {
            continue;
        }
        let sym = rules.symbol_for(case.op).expect("a bound symbol");
        let sig = abi::signature_for(sym).expect("a declared signature");
        for (name, man) in manifests(&dir, case.manifest) {
            let where_ = format!("{}/{name}", case.dir);
            let (n, mut v) = check(case.op, sym, sig, &man, case.slots, case.layers, &where_);
            if n > 0 {
                cases += 1;
            }
            checked += n;
            *seen.entry(case.dir).or_default() += n;
            bad.append(&mut v);
        }
    }
    println!("== kernarg element widths against the captures on this box");
    for (d, n) in &seen {
        println!("   {d:<34} {n} (slot, tensor) pairs");
    }
    println!(
        "   {checked} pairs over {cases} (op, manifest) cases, {} refused",
        bad.len()
    );
    if cache.is_dir() {
        let mut unknown: Vec<String> = Vec::new();
        for e in std::fs::read_dir(&cache).into_iter().flatten().flatten() {
            let n = e.file_name().to_string_lossy().to_string();
            if n.starts_with("capture-") && e.path().is_dir() && !seen.contains_key(n.as_str()) {
                unknown.push(n);
            }
        }
        if !unknown.is_empty() {
            println!(
                "   capture dirs this table says nothing about: {}",
                unknown.join(", ")
            );
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// THE CONTROL, and it is the two defects themselves.
///
/// The same walk over the same captures, with the signature put back the way it was before F22
/// and F24 — `cand`/`candidates` as `int *`, `unary` as `__bf16 *`, `out` as `__bf16 *` — must
/// refuse, and must name the op and the slot. A rule that cannot fail on the bug it was written
/// for is a line of output, not a rule (and this test is inert where the captures are absent, so
/// the control also proves the walk ran at all).
#[test]
fn the_widths_that_shipped_broken_are_refused_by_op_and_slot() {
    let cache = repo().join(".cache");
    let dir = cache.join("capture-dflash-proposal-r28");
    if !dir.is_dir() {
        println!("no proposal capture on this box; the control cannot run");
        return;
    }
    let rules = DflashRules::prod_dflash().expect("the dflash rules parse");
    let broken = |sym: &str, slot: &str, elem: u32| -> Vec<SigArg> {
        abi::signature_for(sym)
            .expect("a declared signature")
            .iter()
            .map(|a| {
                let mut a = a.clone();
                if a.name == slot {
                    a.elem = elem;
                }
                a
            })
            .collect()
    };
    let mut fired = 0;
    for (op, slot, elem, want) in [
        (DflashOp::SelectorScore, "cand", 4, "cand"),
        (DflashOp::SelectorScore, "unary", 2, "unary"),
        (DflashOp::SelectorScore, "out", 2, "out"),
        (DflashOp::SelectorWalk, "candidates", 4, "candidates"),
    ] {
        let sym = rules.symbol_for(op).expect("a bound symbol");
        let sig = broken(sym, slot, elem);
        let case = CASES
            .iter()
            .find(|c| c.dir == "capture-dflash-proposal-r28" && c.op == op)
            .expect("a case for this op");
        let (_n, v) = manifests(&dir, case.manifest)
            .into_iter()
            .map(|(name, man)| check(op, sym, &sig, &man, case.slots, case.layers, &name))
            .fold((0usize, Vec::new()), |(an, mut av), (n, mut v)| {
                av.append(&mut v);
                (an + n, av)
            });
        assert!(
            !v.is_empty(),
            "{}: a {elem}-byte `{slot}` was NOT refused -- the rule has no teeth",
            op.as_str()
        );
        assert!(v
            .iter()
            .all(|m| m.contains(op.as_str()) && m.contains(want)));
        fired += 1;
    }
    assert_eq!(fired, 4, "every one of the two rounds' defects must fire");
}

/// Coverage: every buffer slot of every symbol the drafter's rules BIND declares a width, so the
/// table above cannot be silently outgrown by a new kernel.
#[test]
fn every_bound_dflash_kernel_declares_an_element_width_for_every_pointer() {
    let rules = DflashRules::prod_dflash().expect("the dflash rules parse");
    let mut buffers = 0usize;
    let mut bad = Vec::new();
    for sym in rules.symbols() {
        let sig = abi::signature_for(sym).expect("a declared signature");
        for a in sig {
            if a.kind != ArgKind::Buffer {
                continue;
            }
            buffers += 1;
            if a.elem_bytes().is_none() {
                bad.push(format!("{sym}: `{}` declares no element width", a.name));
            }
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
    assert!(
        buffers >= 60,
        "only {buffers} pointer slots over the bound drafter kernels -- this test could pass by \
         finding none"
    );
}
