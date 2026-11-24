//! INK1 host tests: the config's facts, its refusals (each with the field it names), the visit
//! schedule, the state ledger and the pack plan's coverage of the checkpoint.

use std::path::Path;

use super::config::{ExpertQuant, Inkling2Config};
use super::pack_plan::{ExpertForm, PackPlan, Split};

const CONFIG: &str = include_str!("testdata/config.json");
const CKPT: &str = "/mnt/nvme/models/Inkling-2-Mini-nvfp4";

fn cfg() -> Inkling2Config {
    Inkling2Config::from_config_json(CONFIG).expect("the checkpoint's config parses")
}

fn edited(f: impl FnOnce(&mut serde_json::Value)) -> Result<Inkling2Config, String> {
    let mut v: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
    f(&mut v);
    Inkling2Config::from_config_json(&v.to_string()).map_err(|e| e.to_string())
}

#[test]
fn the_checkpoint_config_parses_to_the_spec_facts() {
    let c = cfg();
    assert_eq!(
        (
            c.hidden_size,
            c.num_hidden_layers,
            c.vocab_size,
            c.unpadded_vocab_size
        ),
        (2304, 24, 201_024, 200_058)
    );
    assert_eq!(
        (c.num_attention_heads, c.num_key_value_heads, c.head_dim),
        (24, 8, 128)
    );
    assert_eq!(
        (
            c.n_routed_experts,
            c.num_experts_per_tok,
            c.intermediate_size
        ),
        (128, 8, 1152)
    );
    let globals: Vec<u32> = (0..24).filter(|&l| !c.is_local(l)).collect();
    assert_eq!(globals, vec![5, 11, 17, 23]);
    let nv = c
        .expert_quant
        .iter()
        .filter(|&&q| q == ExpertQuant::Nvfp4G16)
        .count();
    assert_eq!(nv, 20);
    assert!(c.expert_quant[20..]
        .iter()
        .all(|&q| q == ExpertQuant::Mxfp8G32));
    assert_eq!(c.drafter.target_layer_ids, vec![5, 23, 41, 48]);
    assert_eq!(c.eos_token_id, 200_006);
}

#[test]
fn the_visit_schedule_is_48_visits_with_two_message_passes() {
    let c = cfg();
    let v = c.visits();
    assert_eq!(v.len(), 48);
    let layers: Vec<u32> = v.iter().map(|x| x.layer).collect();
    let mut want: Vec<u32> = (0..6).collect();
    for _ in 0..3 {
        want.extend(6..18);
    }
    want.extend(18..24);
    assert_eq!(layers, want);
    assert_eq!(v.iter().filter(|x| x.messages).count(), 24);
    let resets: Vec<u32> = v
        .iter()
        .filter(|x| x.resets_to_saved)
        .map(|x| x.index)
        .collect();
    assert_eq!(resets, vec![18, 30]);
    let global: Vec<u32> = v.iter().filter(|x| !x.local).map(|x| x.index).collect();
    assert_eq!(global, vec![5, 11, 17, 23, 29, 35, 41, 47]);
    let bounds: Vec<u32> = v
        .iter()
        .filter(|x| x.block_boundary)
        .map(|x| x.index)
        .collect();
    assert_eq!(bounds, vec![0, 6, 12, 18, 24, 30, 36, 42]);
    assert_eq!(v[47].pass, 2);
    assert_eq!((v[17].pass, v[18].pass, v[41].pass), (0, 1, 2));
}

#[test]
fn the_state_ledger_matches_the_spec() {
    let c = cfg();
    let (g, l, conv) = c.state_bytes(1 << 20);
    assert_eq!(g, 32 << 30, "8 global visits x 32 KiB a token x 1M");
    assert_eq!(l, 160 << 20, "40 local rings of 1,024 rows x 4 KiB");
    assert_eq!(conv, 48 * 3 * (2 * 1024 + 2 * 2304) * 4);
    let (g2, l2, _) = c.state_bytes(100);
    assert_eq!((g2, l2), (8 * 100 * 4096, 40 * 100 * 4096));
}

#[test]
fn the_config_refuses_what_the_forward_cannot_run_by_name() {
    type Edit = Box<dyn FnOnce(&mut serde_json::Value)>;
    let cases: Vec<(Edit, &str)> = vec![
        (
            Box::new(|v| v["text_config"]["surprise"] = 1.into()),
            "text_config.surprise",
        ),
        (
            Box::new(|v| v["text_config"]["loop_config"]["is_varloop"] = true.into()),
            "is_varloop",
        ),
        (
            Box::new(|v| v["text_config"]["loop_config"]["core_start"] = 5.into()),
            "loop_config",
        ),
        (
            Box::new(|v| {
                v["quantization"]["language_model.model.layers.3.mlp.switch_mlp.up_proj"]["mode"] =
                    "mxfp4".into()
            }),
            "layers.3",
        ),
        (
            Box::new(|v| v["mtp_config"]["target_layer_ids"] = serde_json::json!([5, 23, 41, 24])),
            "target_layer_ids",
        ),
        (
            Box::new(|v| v["text_config"]["n_shared_experts"] = 1.into()),
            "n_shared_experts",
        ),
        (
            Box::new(|v| {
                v["text_config"].as_object_mut().unwrap().remove("d_rel");
            }),
            "d_rel",
        ),
        (
            Box::new(|v| v["model_type"] = "inkling2".into()),
            "model_type",
        ),
    ];
    for (f, needle) in cases {
        let e = edited(f).expect_err(needle);
        assert!(e.contains(needle), "{needle}: {e}");
    }
}

#[test]
fn the_pack_plan_bills_the_spec_bytes() {
    let c = cfg();
    let p = PackPlan::new(&c, ExpertForm::Mxfp4);
    assert!(p.kinds[..20].iter().all(|&k| k == "nvfp4_requant_mxfp4"));
    assert!(p.kinds[20..].iter().all(|&k| k == "mxfp8_requant_mxfp4"));
    let (ex, split, rep) = p.rank_bytes();
    assert_eq!(ex, 24 * 64 * p.geometry.mxfp4_stride);
    // ~6.5 GB of experts, ~1.1 GB split dense, ~1.1 GB replicated a rank.
    assert!((6_000_000_000..7_000_000_000).contains(&ex), "{ex}");
    assert!((1_000_000_000..1_300_000_000).contains(&split), "{split}");
    assert!((1_000_000_000..1_200_000_000).contains(&rep), "{rep}");
    let n = PackPlan::new(&c, ExpertForm::Nvfp4Native);
    assert!(n.kinds[..20].iter().all(|&k| k == "nvfp4_native"));
    assert_eq!(n.kinds[20], "mxfp8_requant_mxfp4");
    // INK6: every layer native; the MXFP8 record = the checkpoint's bytes (e4m3 + e8m0 / 32).
    let v = PackPlan::new(&c, ExpertForm::Native);
    assert!(v.kinds[..20].iter().all(|&k| k == "nvfp4_native"));
    assert!(v.kinds[20..].iter().all(|&k| k == "mxfp8_native"));
    let m8 = super::pack_plan::mxfp8_layout(2304, 1152);
    assert_eq!(
        m8.gate_up_w.1 + m8.gate_up_ws.1 + m8.down_w.1 + m8.down_ws.1,
        8_211_456
    );
    assert_eq!(m8.stride % 4096, 0);
    assert_eq!(v.stride(21), m8.stride);
    assert_eq!(v.stride(3), v.geometry.nvfp4_stride);
    let (ex, _, _) = v.rank_bytes();
    assert_eq!(ex, 64 * (20 * v.geometry.nvfp4_stride + 4 * m8.stride));
}

#[test]
fn the_pack_plan_covers_every_text_tensor_of_the_checkpoint_once() {
    if !Path::new(CKPT)
        .join("model.safetensors.index.json")
        .exists()
    {
        eprintln!("skip: {CKPT} absent");
        return;
    }
    let mut st = crate::qwen4_exp::st::ShardSet::open(Path::new(CKPT)).unwrap();
    let c = cfg();
    let p = PackPlan::new(&c, ExpertForm::Mxfp4);
    let mut planned = std::collections::HashSet::new();
    for d in &p.dense {
        let info = st
            .info(&d.tensor)
            .unwrap_or_else(|| panic!("{} not in the checkpoint", d.tensor));
        assert_eq!(info.dtype, d.dtype, "{}", d.tensor);
        assert_eq!(info.shape, d.shape, "{}", d.tensor);
        if let Split::Half { dim } = d.split {
            assert_eq!(d.shape[dim as usize] % 2, 0, "{}", d.tensor);
        }
        assert!(planned.insert(d.tensor.clone()), "{} twice", d.tensor);
    }
    let names: Vec<String> = st.names().cloned().collect();
    for n in names {
        if n.starts_with("vision_tower.") || n.contains(".mlp.switch_mlp.") {
            continue;
        }
        assert!(planned.contains(&n), "{n} has no record");
    }
}

#[test]
fn the_layer_graph_declares_every_visit_and_the_new_bodies() {
    use super::layer_graph::{new_bodies, step_launches, visit_launches};
    let c = cfg();
    let v = c.visits();
    // visit 0: no depth mix; local pass-1: one merged projection; global / message visits: three.
    assert_eq!(visit_launches(&v[0]), 16);
    assert_eq!(visit_launches(&v[1]), 17);
    assert_eq!(visit_launches(&v[5]), 19, "global: q, k, v inputs");
    assert_eq!(
        visit_launches(&v[18]),
        22,
        "message visit: msg attend + 2 gate ops + 3 projections"
    );
    let (all, new) = step_launches(&c);
    // 1 visit-0 + 19 local pass-1 + 4 global non-message + 24 message visits + embed/head.
    assert_eq!(all, 5 + 16 + 19 * 17 + 4 * 19 + 24 * 22);
    assert!(
        new > all / 2,
        "most launches are new bodies: {new} of {all}"
    );
    assert_eq!(new_bodies(&c).len(), 13);
}

fn geom(tp: u32) -> super::engine_plan::EngineGeom {
    super::engine_plan::EngineGeom {
        tp,
        max_seqs: 8,
        max_ctx: 32_768,
        max_chunk: 2_048,
        page_rows: 64,
        decode_splits: 16,
    }
}

#[test]
fn ink2_the_rank_plan_bills_the_spec_state_per_rank() {
    use super::engine_plan::RankPlan;
    let c = cfg();
    let p = RankPlan::new(&c, geom(2));
    assert_eq!(
        (p.global_visits, p.local_visits, p.conv_states),
        (8, 40, 192)
    );
    let by = |n: &str| p.classes.iter().find(|x| x.name == n).unwrap().bytes;
    // global: 16 KiB a token a rank (32 KiB over the pair), 512 pages a sequence + 1 spare
    assert_eq!(p.pages_per_seq, 512);
    assert_eq!(by("ink.kv.global"), 8 * (8 * 512 + 1) * 64 * 2048);
    // local: 40 rings x 1,024 rows x 2 KiB a rank = 80 MiB a slot (the spec's 160 MiB over the pair)
    assert_eq!(by("ink.kv.local"), 9 * (80 << 20));
    // conv: k / v split (512 a rank), attn / mlp replicated (2,304): 48 x 3 x 5,632 x 4 B a slot
    assert_eq!(by("ink.conv"), 9 * 48 * 3 * 5632 * 4);
    // TP=1 is the spec's whole-model numbers
    let p1 = RankPlan::new(&c, geom(1));
    let by1 = |n: &str| p1.classes.iter().find(|x| x.name == n).unwrap().bytes;
    assert_eq!(by1("ink.kv.local"), 9 * (160 << 20));
    assert_eq!(by1("ink.conv") / 9, 3_833_856); // 3.66 MiB a sequence (§5)
    assert!(
        p1.classes.iter().all(|x| x.name != "ink.ar"),
        "no reduce at TP=1"
    );
    // the state ledger of the config agrees with the pools at one sequence (no spare)
    let (g, l, cv) = c.state_bytes(32_768);
    assert_eq!(g, 8 * 32_768 * 4_096); // 32 KiB a token (§5)
    assert_eq!(by1("ink.kv.global"), 8 * (8 * 512 + 1) * 64 * 4_096);
    assert_eq!((l, cv), (160 << 20, 3_833_856));
}

#[test]
fn ink2_the_arena_is_the_widest_visit_and_names_the_rel_bias() {
    use super::engine_plan::{Pass, RankPlan};
    let p = RankPlan::new(&cfg(), geom(2));
    assert_eq!((p.prefill.pass, p.prefill.rows), (Pass::Prefill, 2_048));
    assert_eq!((p.decode.pass, p.decode.rows), (Pass::Decode, 8));
    let rel = p
        .prefill
        .bufs
        .iter()
        .find(|b| b.0 == "visit.rel_bias")
        .unwrap()
        .1;
    assert_eq!(
        rel,
        2_048 * 12 * 1_024 * 4,
        "r . P: [rows, 12 heads, 1,024] f32"
    );
    // the widest visit kind is a message visit of a global layer (3 inputs + 4 gates): 11, 17
    let v = cfg().visits()[p.prefill.widest_visit as usize];
    assert!(v.messages && !v.local, "{v:?}");
    assert!(p.decode.bufs.iter().any(|b| b.0 == "visit.attn_partials"));
    assert!(p.prefill.bufs.iter().all(|b| b.0 != "visit.attn_partials"));
    assert_eq!(
        p.prefill.bytes,
        p.prefill.bufs.iter().map(|b| b.1).sum::<u64>()
    );
    assert!(p.prefill.bytes > p.decode.bytes);
}

#[test]
fn ink2_the_plan_text_is_the_built_packs_plan() {
    use sha2::{Digest, Sha256};
    let pack = Path::new("/mnt/nvme/models/ink2-pack/manifest.json");
    if !pack.is_file() || !Path::new(CKPT).join("config.json").is_file() {
        eprintln!("skip: the full pack or its checkpoint is absent");
        return;
    }
    let m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(pack).unwrap()).unwrap();
    let c = Inkling2Config::from_config_json(
        &std::fs::read_to_string(Path::new(CKPT).join("config.json")).unwrap(),
    )
    .unwrap();
    let text = PackPlan::new(&c, ExpertForm::Mxfp4).plan_text(&c, CKPT);
    let got = hex::encode(Sha256::digest(text.as_bytes()));
    assert_eq!(got, m["plan_sha256"].as_str().unwrap());
    // a control: the other form's plan is not this pack's
    let other = PackPlan::new(&c, ExpertForm::Nvfp4Native).plan_text(&c, CKPT);
    assert_ne!(hex::encode(Sha256::digest(other.as_bytes())), got);
}
