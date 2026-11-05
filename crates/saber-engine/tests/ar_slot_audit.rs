//! **T10: the all-reduce protocol-state audit** -- which `ar_scratch` / `ar_flags` / `ar_seq`
//! region every reduce of the eager prefill path and of every captured decode graph resolves to.
//!
//! The launcher keys a peer-visible region by `(plugin, op, the route's bound shape, name)`
//! (`Launcher::ipc_key`). The one-shot kernels pick the scratch slot per BLOCK from that region's
//! per-block counter (`slot = ++seq_ctrs[b] & 1`) and push block `b`'s share of the message --
//! a share that depends on the call's grid and message size. So a region is safe by construction
//! only if every launch through it has ONE map (grid, message units): with two maps the counters
//! drift apart across blocks and, at a map change, the peer's first push of the new map can land
//! in the slot a still-reducing block of the old call reads (LEDGER T10).
//!
//! The audit: every reduce of every context (the prefill rungs, each decode bucket under
//! `1,2,4,8` and `1..8`, the drafter's shapes), its region key and its map; a key shared across
//! contexts, and a key with more than one map, are named. Negative control (rule 8): two
//! synthetic contexts with disjoint keys audit clean, the same two sharing one key do not.

use std::collections::BTreeMap;

use saber_engine::draft_plan::planned_shapes;
use saber_exec::layer::{BlockPlan, Collectives, PlanOptions, Route, RuntimeParams};
use saber_kernels::rules::{ArgValue, LaunchDescriptor, LaunchRules, Shape};
use saber_kernels::saber::{self, SaberRules};
use saber_model::dflash_graph::DflashConfig;
use saber_model::graph::{Bucket, Graph, ModelConfig};

const LADDER: [u32; 11] = [96, 128, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048];
const POW2: [u32; 4] = [1, 2, 4, 8];
const BATCH_LADDER: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

/// One reduce launch: its context, its region key, its map.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Use {
    key: String,
    map: (u32, u64),
    ctx: String,
    eager: bool,
}

fn units(d: &LaunchDescriptor) -> u64 {
    let v = |n: &str| match d.arg(n).map(|s| &s.value) {
        Some(ArgValue::I32 { ival } | ArgValue::I64 { ival }) => *ival as u64,
        _ => 0,
    };
    v("n_groups") + v("n16") + v("k16")
}

fn uses_of(descs: &[LaunchDescriptor], shape: Shape, ctx: &str, eager: bool) -> Vec<Use> {
    descs
        .iter()
        .filter(|d| d.workspaces.iter().any(|w| w.name == "ar_seq"))
        .map(|d| Use {
            key: format!(
                "{}/{} m={} n={} k={}",
                d.plugin, d.op, shape.m, shape.n, shape.k
            ),
            map: (d.grid[0], units(d)),
            ctx: ctx.to_owned(),
            eager,
        })
        .collect()
}

fn plan_uses(g: &Graph, bucket: Bucket, opts: PlanOptions, ctx: &str, eager: bool) -> Vec<Use> {
    let layers: Vec<u32> = (0..g.config.num_hidden_layers).collect();
    let p = BlockPlan::with_options(
        g,
        bucket,
        &layers,
        RuntimeParams {
            max_ctx: 4096,
            gdn_slots: 65,
        },
        &LaunchRules::prod_mxfp4().expect("rules"),
        &SaberRules::prod_saber().expect("saber"),
        opts,
    )
    .expect("a plan");
    p.steps
        .iter()
        .flat_map(|s| match &s.route {
            Route::Kernel { descs, shape } => uses_of(descs, *shape, ctx, eager),
            _ => Vec::new(),
        })
        .collect()
}

/// The findings: keys used by more than one context (and whether an eager and a graph context are
/// among them), and keys with more than one map.
fn audit(uses: &[Use]) -> (Vec<String>, Vec<String>) {
    let mut by_key: BTreeMap<&str, Vec<&Use>> = BTreeMap::new();
    for u in uses {
        by_key.entry(&u.key).or_default().push(u);
    }
    let mut shared = Vec::new();
    let mut multimap = Vec::new();
    for (k, us) in &by_key {
        let mut ctxs: Vec<&str> = us.iter().map(|u| u.ctx.as_str()).collect();
        ctxs.sort_unstable();
        ctxs.dedup();
        let mixed = us.iter().any(|u| u.eager) && us.iter().any(|u| !u.eager);
        if ctxs.len() > 1 {
            shared.push(format!(
                "{k}: {} contexts{} ({} .. {})",
                ctxs.len(),
                if mixed { ", EAGER AND GRAPH" } else { "" },
                ctxs[0],
                ctxs[ctxs.len() - 1]
            ));
        }
        let mut maps: Vec<(u32, u64)> = us.iter().map(|u| u.map).collect();
        maps.sort_unstable();
        maps.dedup();
        if maps.len() > 1 {
            multimap.push(format!(
                "{k}: {} maps (grid, units) {:?}{}",
                maps.len(),
                &maps[..maps.len().min(6)],
                if maps.len() > 6 { " ..." } else { "" }
            ));
        }
    }
    (shared, multimap)
}

fn every_use(set: &[u32]) -> Vec<Use> {
    let g = Graph::qwen38_27b(ModelConfig::qwen38_27b(), 2).expect("graph");
    let body = || {
        PlanOptions::BODY
            .rank(0)
            .with_collectives(Collectives::Real)
            .with_batch_invariant_attn()
    };
    let mut uses = Vec::new();
    for rows in LADDER {
        uses.extend(plan_uses(
            &g,
            Bucket::Prefill {
                tokens: rows,
                seqs: 1,
            },
            body(),
            &format!("prefill p{rows}"),
            true,
        ));
    }
    for b in set {
        for w in [1u32, 5, 8] {
            uses.extend(plan_uses(
                &g,
                Bucket::Decode {
                    batch: *b,
                    verify_width: w,
                },
                body().with_table_blocks(257),
                &format!("decode b{b}s{w}"),
                false,
            ));
        }
    }
    // The drafter binds every reduce at `Shape::new(0, 0, 0)` too (`DraftPlan::all_reduce`), one
    // message of `rows x hidden` bf16 per call; its backbone runs `reqs x block` rows.
    let r4 = SaberRules::prod_saber().expect("saber");
    let cfg = DflashConfig::qwen38_dflash2(2, 7);
    for s in planned_shapes(&cfg, 16, 4096, &[1, 5, 8], set, 8) {
        let rows = u64::from(s.reqs) * 8;
        let descs = saber::resolve_ar_banded(
            rows,
            5120,
            2,
            saber::ArPayload::Bf16,
            2,
            &r4,
            &saber::Registry::new(&r4),
        )
        .expect("a drafter reduce");
        uses.extend(uses_of(
            &descs,
            Shape::new(0, 0, 0),
            &format!("drafter r{}c{}", s.reqs, s.ctx_rows),
            false,
        ));
    }
    uses
}

/// The census under both bucket sets: every reduce of the eager prefill path and of every
/// captured decode graph resolves to ONE region per kernel family (the route binds
/// `Shape::new(0, 0, 0)`), so the prefill rungs, the decode graphs and the drafter share one
/// `ar_seq` counter array, one flag array and one scratch -- at many maps. This test pins the
/// finding (T10); it is the one to rewrite when the regions are keyed per map.
#[test]
fn the_reduce_regions_are_shared_across_the_prefill_path_and_every_decode_graph() {
    for (name, set) in [("1,2,4,8", &POW2[..]), ("1..8", &BATCH_LADDER[..])] {
        let uses = every_use(set);
        let (shared, multimap) = audit(&uses);
        println!("== bucket set {name}: {} reduce launches", uses.len());
        for s in &shared {
            println!("   shared   {s}");
        }
        for m in &multimap {
            println!("   maps     {m}");
        }
        assert!(
            shared
                .iter()
                .any(|s| s.contains("allreduce_wht") && s.contains("EAGER AND GRAPH")),
            "{name}: the lossy family's region is expected shared by the prefill rungs and the \
             decode graphs (the T10 finding); if this fails the regions were re-keyed -- update \
             the LEDGER and turn this census into the disjointness law"
        );
        assert!(
            multimap.iter().any(|m| m.contains("allreduce_wht")),
            "{name}: the shared lossy region is expected to carry several maps"
        );
        // Every key is the unkeyed shape: nothing distinguishes one message from another.
        assert!(
            uses.iter()
                .all(|u| u.key.ends_with("m=0 n=0 k=0") || u.key.contains("gemm")),
            "{name}: a reduce keyed by its shape: {:?}",
            uses.iter().find(|u| !u.key.ends_with("m=0 n=0 k=0"))
        );
    }
}

/// Negative control (rule 8): the audit itself. Two contexts on disjoint keys, one map each, are
/// clean; the same two on ONE deliberately shared key are named as shared and multi-map.
#[test]
fn the_audit_names_a_deliberately_shared_slot() {
    let u = |key: &str, map: (u32, u64), ctx: &str, eager: bool| Use {
        key: key.to_owned(),
        map,
        ctx: ctx.to_owned(),
        eager,
    };
    let clean = vec![
        u(
            "saber/allreduce_wht m=384 n=5120 k=0",
            (96, 960),
            "prefill p384",
            true,
        ),
        u(
            "saber/allreduce_wht m=384 n=5120 k=0",
            (96, 960),
            "prefill p384",
            true,
        ),
        u(
            "saber/allreduce_wht m=24 n=5120 k=0",
            (60, 60),
            "decode b3s8",
            false,
        ),
    ];
    let (s, m) = audit(&clean);
    assert!(s.is_empty() && m.is_empty(), "{s:?} {m:?}");
    let shared = vec![
        u(
            "saber/allreduce_wht m=0 n=0 k=0",
            (96, 960),
            "prefill p384",
            true,
        ),
        u(
            "saber/allreduce_wht m=0 n=0 k=0",
            (60, 60),
            "decode b3s8",
            false,
        ),
    ];
    let (s, m) = audit(&shared);
    assert_eq!(s.len(), 1, "{s:?}");
    assert!(s[0].contains("EAGER AND GRAPH"), "{s:?}");
    assert_eq!(m.len(), 1, "{m:?}");
}
