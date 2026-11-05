//! Shared fixtures: a real [`ExecutionPlan`] (resolved through `saber-plan`, not hand-built), a
//! ledger with declared pools, and a scheduler over one of the mock executors.
//!
//! The autotune cache here is fabricated, which is legal in a test and nowhere else — `saber-plan`
//! ships no such helper on purpose. Nothing else is fabricated: the buckets, the GDN state budget
//! and the Infinity-Cache gate are the planner's own.

#![allow(dead_code)]

use std::fmt::Write as _;

use saber_mem::ledger::{Ledger, Pool};
use saber_plan::{
    plan, reference, resolve, AutotuneCache, AutotuneRecord, DeviceProfile, ExecutionPlan,
    Manifest, ModelSpec, Provenance, VariantParams,
};
use saber_sched::{ArenaBytes, KvGeometry, Scheduler, SchedulerConfig, StepExecutor, WidthRule};
use saber_serve::engine::{EngineRequest, ModelInfo, StepEvent, Submission, TokenId};
use saber_serve::sampling::SamplingParams;

pub const MIB: u64 = 1024 * 1024;
pub const GIB: u64 = 1024 * MIB;

/// One EOS id, so the stop path is exercised without a tokenizer.
pub const EOS: TokenId = 151_645;

/// Builds a manifest whose `[[buckets]]` are exactly `buckets` (batch, spec_width, ctx_band).
pub fn manifest_with(buckets: &[(u32, u32, u32)], tp: u32) -> Manifest {
    let captured: Vec<(u32, u32, u32, bool)> =
        buckets.iter().map(|&(b, s, c)| (b, s, c, true)).collect();
    manifest_with_capture(&captured, tp)
}

/// Every plugin the reference registry names, in a stable order.
///
/// Derived from the registry rather than transcribed: a route whose plugin the manifest does not
/// declare is [`saber_plan::PlanError::UnknownPlugin`] at bind, and that is exactly how this
/// fixture broke when `saber-plan` grew the reference's attention / GDN / all-reduce rows. Reading the
/// list off the registry means the next plugin added there needs no edit here.
pub fn reference_plugins() -> Vec<String> {
    let registry = reference::reference_registry();
    let mut names: Vec<String> = registry
        .implementations()
        .iter()
        .map(|imp| imp.plugin.clone())
        .collect();
    names.sort();
    names.dedup();
    names
}

/// The same, with each bucket's `capture` flag stated.
pub fn manifest_with_capture(buckets: &[(u32, u32, u32, bool)], tp: u32) -> Manifest {
    let toml = manifest_text(buckets, tp);
    Manifest::from_str(&toml, "fixture.toml").expect("fixture manifest is valid")
}

/// The manifest body: model, device, every plugin the registry names, and the decode buckets.
fn manifest_text(buckets: &[(u32, u32, u32, bool)], tp: u32) -> String {
    let mut toml = format!(
        r#"
[model]
path = "/mnt/nvme/models/Qwen3.8-27B-NVFP4"
tier_id = "tier:qwen38-27b-nvfp4-mxfp4-dflash2-v22.3"

[device]
target_gfx = "gfx1201"
power_cap_w = 210
tp = {tp}

[plan]
capture = true
"#
    );
    for (i, plugin) in reference_plugins().into_iter().enumerate() {
        let _ = write!(
            toml,
            "\n[[plugins]]\nname = \"{plugin}\"\npath = \"plugins/gfx1201/{plugin}.hsaco\"\n\
             build_hash = \"fixture{i:04x}\"\n"
        );
    }
    for (batch, spec_width, ctx_band, capture) in buckets {
        let _ = write!(
            toml,
            "\n[[buckets]]\nbatch = {batch}\nspec_width = {spec_width}\nctx_band = {ctx_band}\n\
             capture = {capture}\n"
        );
    }
    toml
}

/// A manifest with decode buckets **and** prefill buckets: `(tokens, ctx_band)` each.
pub fn manifest_with_prefill(
    decode: &[(u32, u32, u32)],
    prefill: &[(u32, u32)],
    tp: u32,
) -> Manifest {
    let captured: Vec<(u32, u32, u32, bool)> =
        decode.iter().map(|&(b, s, c)| (b, s, c, true)).collect();
    let mut toml = manifest_text(&captured, tp);
    for (tokens, ctx_band) in prefill {
        let _ = write!(
            toml,
            "\n[[buckets]]\nphase = \"prefill\"\ntokens = {tokens}\nctx_band = {ctx_band}\n"
        );
    }
    Manifest::from_str(&toml, "fixture.toml").expect("fixture manifest is valid")
}

/// The same plan as [`plan_with`], with prefill buckets in it.
pub fn plan_with_prefill(
    decode: &[(u32, u32, u32)],
    prefill: &[(u32, u32)],
    tp: u32,
) -> ExecutionPlan {
    bind(manifest_with_prefill(decode, prefill, tp), tp)
}

/// Resolves and binds a plan for the reference model on the reference device.
pub fn plan_with(buckets: &[(u32, u32, u32)], tp: u32) -> ExecutionPlan {
    let captured: Vec<(u32, u32, u32, bool)> =
        buckets.iter().map(|&(b, s, c)| (b, s, c, true)).collect();
    plan_with_capture(&captured, tp)
}

/// The same, with each bucket's `capture` flag stated — used to prove an eager bucket is never
/// replayed.
pub fn plan_with_capture(buckets: &[(u32, u32, u32, bool)], tp: u32) -> ExecutionPlan {
    bind(manifest_with_capture(buckets, tp), tp)
}

/// Resolve a fixture manifest, fabricate a record for every route it needs, and bind.
fn bind(manifest: Manifest, tp: u32) -> ExecutionPlan {
    let device = DeviceProfile::gfx1201_reference();
    let model = ModelSpec::qwen38_27b_dense(tp).expect("model spec");
    let registry = reference::reference_registry();
    let resolved = resolve(&manifest, &device, &model, &registry).expect("resolve");

    // One record per requirement, whatever the requirements turn out to be: the GEMM family, the
    // saber attention / GDN / all-reduce rows, and both GDN routes where their batch bands apply.
    // Enumerating them from `requirements()` rather than naming them is what keeps this fixture
    // from going stale the next time `ModelSpec` grows a row.
    let mut cache = AutotuneCache::new();
    for r in resolved.requirements() {
        cache.insert(
            r.key.clone(),
            AutotuneRecord {
                impl_name: r.candidates[0].clone(),
                variant: VariantParams::new(12).param("split_k", 4).param("tm", 4),
                measured_us: 31.2,
                samples_kept: 4,
                runner_up: None,
                unstable: None,
                provenance: Provenance::new("fixture", "2026-09-18", "none"),
            },
        );
    }
    assert!(
        resolved.missing(&cache).is_empty(),
        "the fixture cache must answer every route the bind needs: {:?}",
        resolved
            .missing(&cache)
            .iter()
            .map(|r| r.op_name.clone())
            .collect::<Vec<_>>()
    );
    plan(&manifest, &device, &model, &registry, &cache).expect("bind")
}

/// The decode bucket ladder the scheduling tests use: batches 1/2/4/8 x widths 1/3/5, two bands.
/// `8 x 5 x 3.0 MiB / tp 2` = 60 MiB, inside the 64 MiB Infinity Cache — the 09-14 gate passes.
pub fn decode_buckets() -> Vec<(u32, u32, u32)> {
    let mut v = Vec::new();
    for batch in [1u32, 2, 4, 8] {
        for width in [1u32, 3, 5] {
            for band in [8192u32, 65536] {
                v.push((batch, width, band));
            }
        }
    }
    v
}

/// A ledger with every pool declared and room to spare, so a refusal in a test is the scheduler's,
/// never the ledger's.
pub fn boot_ledger() -> Ledger {
    let mut l = Ledger::new();
    l.declare(Pool::Vram(0), 32 * GIB).expect("rank 0");
    l.declare(Pool::Vram(1), 32 * GIB).expect("rank 1");
    l.declare(Pool::HostPinned, 4 * GIB).expect("pinned");
    l.declare(Pool::HostPageable, 48 * GIB).expect("pageable");
    l.declare(Pool::Nvme, 64 * GIB).expect("nvme");
    l
}

/// The model info the scheduler serves: one EOS id, prod's context length.
pub fn model_info() -> ModelInfo {
    ModelInfo {
        served_model_name: "Qwen3.8".to_owned(),
        aliases: Vec::new(),
        max_model_len: 262_144,
        root: "/mnt/nvme/models/Qwen3.8-27B-NVFP4".to_owned(),
        eos_token_ids: vec![EOS],
    }
}

/// A config over `buckets` with explicit arena sizes.
pub fn config_with(buckets: &[(u32, u32, u32)], tp: u32, arenas: ArenaBytes) -> SchedulerConfig {
    SchedulerConfig::from_plan(
        &plan_with(buckets, tp),
        KvGeometry::PROD_FP8,
        arenas,
        262_144,
    )
    .expect("config")
}

/// The SPEC=0-shaped ladder used by the admission tests: one draft token, batches to 32, one wide
/// band. `32 x 1 x 3.0 MiB / tp 2` = 48 MiB, inside the Infinity Cache.
pub fn wide_batch_buckets() -> Vec<(u32, u32, u32)> {
    [1u32, 2, 4, 8, 16, 32]
        .into_iter()
        .map(|batch| (batch, 1, 32768))
        .collect()
}

/// The default fixture under prod's **acceptance-EMA** verify-width rule.
///
/// The serving default is the batch table (ws-t2), so a test that is asking what an acceptance EMA
/// does has to say so — and a test that forgets gets the table's answer, which is a width that
/// does not move with acceptance at all.
pub fn scheduler_ema<E: StepExecutor>(executor: E) -> Scheduler<E> {
    let mut config = config_with(
        &decode_buckets(),
        2,
        ArenaBytes {
            kv_per_rank: 8 * GIB,
            gdn_state_per_rank: 8 * GIB,
        },
    );
    config.width = config.width.with_rule(WidthRule::AcceptanceEma);
    Scheduler::new(config, executor, model_info(), boot_ledger()).expect("the arenas fit")
}

/// The default fixture under an explicit batch table.
///
/// The fixture's captured widths are `1,3,5`, so a table is written in those terms
/// (`1:5,4:3` = the ceiling below four running requests and the middle width at or above it) and
/// the scheduler's own answer is then observable rather than rounded to the ceiling.
pub fn scheduler_by_batch<E: StepExecutor>(table: &str, executor: E) -> Scheduler<E> {
    let mut config = config_with(
        &decode_buckets(),
        2,
        ArenaBytes {
            kv_per_rank: 8 * GIB,
            gdn_state_per_rank: 8 * GIB,
        },
    );
    config.width = config
        .width
        .with_rule(WidthRule::parse(&format!("batch={table}")).expect("a batch table"));
    Scheduler::new(config, executor, model_info(), boot_ledger()).expect("the arenas fit")
}

/// The default scheduling fixture: the decode ladder, generous arenas.
pub fn scheduler<E: StepExecutor>(executor: E) -> Scheduler<E> {
    scheduler_with(
        &decode_buckets(),
        2,
        ArenaBytes {
            kv_per_rank: 8 * GIB,
            gdn_state_per_rank: 8 * GIB,
        },
        executor,
    )
}

/// A scheduler over an explicit bucket set and arena sizing.
pub fn scheduler_with<E: StepExecutor>(
    buckets: &[(u32, u32, u32)],
    tp: u32,
    arenas: ArenaBytes,
    executor: E,
) -> Scheduler<E> {
    Scheduler::new(
        config_with(buckets, tp, arenas),
        executor,
        model_info(),
        boot_ledger(),
    )
    .expect("the arenas fit")
}

/// A scheduler over an already-built plan (used where a bucket's `capture` flag matters).
pub fn scheduler_from_plan<E: StepExecutor>(
    plan: &ExecutionPlan,
    arenas: ArenaBytes,
    executor: E,
) -> Scheduler<E> {
    let config =
        SchedulerConfig::from_plan(plan, KvGeometry::PROD_FP8, arenas, 262_144).expect("config");
    Scheduler::new(config, executor, model_info(), boot_ledger()).expect("the arenas fit")
}

/// Sampling params with a token budget and no EOS suppression.
pub fn params(max_tokens: usize) -> SamplingParams {
    SamplingParams {
        max_tokens: Some(max_tokens),
        ..SamplingParams::default()
    }
}

/// A request.
pub fn request(prompt: &[TokenId], max_tokens: usize) -> EngineRequest {
    EngineRequest {
        prompt_token_ids: prompt.to_vec(),
        params: params(max_tokens),
    }
}

/// Drains a submission's stream. The scheduler is stepped by the test, so by the time this is
/// called every event has already been sent.
pub async fn drain(submission: Submission) -> Vec<StepEvent> {
    use futures_util::StreamExt;
    submission.events.collect().await
}

/// The tokens a request's events committed, in order.
pub fn tokens_of(events: &[StepEvent]) -> Vec<TokenId> {
    events.iter().flat_map(|e| e.new_tokens.clone()).collect()
}
