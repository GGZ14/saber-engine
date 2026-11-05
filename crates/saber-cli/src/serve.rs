//! `saber serve --mock` — the OpenAI `/v1` plane over the real scheduler, with a scripted
//! executor instead of a GPU.
//!
//! Everything above the [`StepExecutor`](saber_sched::StepExecutor) seam is production code: the
//! manifest is resolved the way `saber explain` resolves it, the scheduler admits against the real
//! paged-KV and GDN-state allocators through the real byte ledger, the verify width is prod's EMA
//! cap, the prompt is rendered by `qwen-fixed-v22.3.jinja` and tokenized by the checkpoint's own
//! `tokenizer.json`, and the stream is `saber-serve`'s — **one SSE update per engine step**, with
//! the `qwen3_coder` tool parser and the `qwen3` reasoning parser on. Only the tokens are fake.
//!
//! That makes it the harness the serving plane's parity is measured with before P3 has a kernel:
//! BetterBench, `pi-bench` and a browser all see the endpoint they will see later, and what they
//! classify — `batched`, one update per step, exact `usage` — is a property of this stack rather
//! than of the model.
//!
//! ```text
//! saber serve --mock examples/mock-script.toml \
//!             --manifest examples/qwen38-27b-nvfp4-tp2.toml \
//!             --autotune examples/autotune-gfx1201-210w.json \
//!             --port 8099
//! ```
//!
//! There is no environment variable on this path: the script, the manifest, the tokenizer, the
//! template and the port are all arguments.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context};
use saber_mem::ledger::{Ledger, Pool};
use saber_plan::{
    check_plugins_declared, registry_from_prebuilt, AutotuneCache, Bucket, DeviceProfile, Manifest,
    ModelSpec,
};
use saber_sched::{
    attach_park_tier, build_state_with_parsers, mock_model_info, spawn_step_loop, ArenaBytes,
    KvGeometry, ParkPolicy, Scheduler, SchedulerConfig, ScriptExecutor, ServeParkOptions,
};
use saber_serve::detok::ModelCodec;
use saber_serve::engine::TokenId;
use saber_serve::state::TemplateRenderer;
use serde::Deserialize;

/// Arguments of `saber serve`.
pub struct ServeArgs {
    /// The mock script: what the engine "generates", and how many tokens a step commits.
    pub mock: PathBuf,
    /// The Saber manifest, resolved exactly as `saber explain` resolves it.
    pub manifest: PathBuf,
    /// The prebuilt plugin directory the registry is derived from.
    pub prebuilt: PathBuf,
    /// An autotune cache. When it covers every route the plan **binds** and the scheduler is
    /// configured from it; when it does not, the mock falls back to the manifest's bucket
    /// geometry and says so, loudly, because no kernel is selected on this path anyway.
    pub autotune: Option<PathBuf>,
    /// The checkpoint's `tokenizer.json`.
    pub tokenizer: PathBuf,
    /// The chat template (`qwen-fixed-v22.3.jinja`).
    pub chat_template: PathBuf,
    /// Address to bind. Port 0 binds an ephemeral port and prints the one it got.
    pub host: String,
    /// Port to bind.
    pub port: u16,
    /// `--max-logprobs` (V7).
    pub max_logprobs: i32,
    /// The parser flags, resolved (V9).
    pub parsers: saber_serve::state::ResolvedParsers,
    /// KV arena per rank, GiB.
    pub kv_gib: u64,
    /// GDN state arena per rank, GiB.
    pub state_gib: u64,
    /// Milliseconds a step takes. A mock step is microseconds and a GPU step is ~22 ms, so a
    /// client that wants a stream paced like a server's asks for one here.
    pub step_delay_ms: u64,
    /// Host RAM the session-parking tier may hold, GiB. 0 means no parking.
    pub park_ram_gib: f64,
    /// Parking-file bytes, GiB. Needs `park_nvme_path`. 0 means no spill tier.
    pub park_nvme_gib: f64,
    /// Where the parking file lives.
    pub park_nvme_path: Option<PathBuf>,
    /// Let the scheduler evict on its own when an admission is refused for KV or GDN state.
    /// Without it a tier is an operator hook and nothing else.
    pub park_on_pressure: bool,
}

/// One scripted response.
#[derive(Debug, Deserialize)]
struct ScriptResponse {
    /// The text the engine "generates". Tokenized with the real tokenizer, so a tool call or a
    /// `<think>` block in here reaches the parsers exactly as a model's output would.
    text: String,
}

/// The `--mock` file.
#[derive(Debug, Deserialize)]
struct Script {
    /// Tokens a **decode** step commits. The prefill step always commits one, as prod's does.
    #[serde(default = "default_tokens_per_step")]
    tokens_per_step: u32,
    /// The model name the endpoint serves and clients must ask for.
    #[serde(default = "default_model_name")]
    served_model_name: String,
    /// Responses, cycled in admission order.
    response: Vec<ScriptResponse>,
}

fn default_tokens_per_step() -> u32 {
    5
}

fn default_model_name() -> String {
    "Qwen3.8".to_owned()
}

const GIB: u64 = 1 << 30;

/// The end-of-turn token, resolved from the tokenizer rather than hard-coded: `<|im_end|>` is what
/// the chat template emits and what the served model stops on.
pub(crate) fn eos_id(tokenizer: &saber_model::Tokenizer) -> anyhow::Result<TokenId> {
    let ids = tokenizer
        .encode("<|im_end|>")
        .context("tokenizing <|im_end|>")?;
    match ids.as_slice() {
        [one] => Ok(*one),
        other => bail!(
            "<|im_end|> tokenizes to {other:?}, not a single id; this tokenizer is not the \
             checkpoint's"
        ),
    }
}

/// Every id that ends generation, from the checkpoint's `generation_config.json`.
///
/// `<|im_end|>` alone is not the answer, and taking it by tokenizing a literal was a guess that
/// happened to be mostly right: this checkpoint declares `"eos_token_id": [248046, 248044]` —
/// `<|im_end|>` **and** `<|endoftext|>` — and a request that ends on the one the server does not
/// know about runs to `max_tokens`, emitting whatever the model produces after the conversation is
/// over. The file is the checkpoint's own statement of the fact; `<|im_end|>` is the fallback for a
/// pack that does not ship one, and it is announced rather than assumed.
pub(crate) fn eos_ids(
    tokenizer: &saber_model::Tokenizer,
    pack_root: &Path,
) -> anyhow::Result<Vec<TokenId>> {
    let fallback = eos_id(tokenizer)?;
    let path = pack_root.join("generation_config.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Ok(vec![fallback]);
    };
    let v: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    let mut out: Vec<TokenId> = match &v["eos_token_id"] {
        serde_json::Value::Number(n) => n.as_u64().map(|x| vec![x as TokenId]).unwrap_or_default(),
        serde_json::Value::Array(a) => a
            .iter()
            .filter_map(|x| x.as_u64().map(|x| x as TokenId))
            .collect(),
        _ => Vec::new(),
    };
    if !out.contains(&fallback) {
        out.push(fallback);
    }
    Ok(out)
}

fn read_script(path: &Path) -> anyhow::Result<Script> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading the mock script {}", path.display()))?;
    let script: Script = toml::from_str(&text)
        .with_context(|| format!("parsing the mock script {}", path.display()))?;
    if script.response.is_empty() {
        bail!("{}: no [[response]] entries", path.display());
    }
    if script.tokens_per_step == 0 {
        bail!("{}: tokens_per_step must be >= 1", path.display());
    }
    Ok(script)
}

pub fn run(args: &ServeArgs) -> anyhow::Result<()> {
    // ── the plan, exactly as `saber explain` builds it ──────────────────────────────────────
    let manifest = Manifest::from_path(&args.manifest)?;
    let registry = registry_from_prebuilt(&args.prebuilt)?;
    let device = DeviceProfile::gfx1201_reference();
    let model = ModelSpec::qwen38_27b_dense(manifest.device.tp)?;
    check_plugins_declared(&registry, &manifest, &model)?;
    let resolved = saber_plan::resolve(&manifest, &device, &model, &registry)?;
    let cache = match &args.autotune {
        Some(p) => AutotuneCache::load(p)?,
        None => AutotuneCache::new(),
    };
    let missing = resolved.missing(&cache);

    let arenas = ArenaBytes {
        kv_per_rank: args.kv_gib * GIB,
        gdn_state_per_rank: args.state_gib * GIB,
    };
    let max_model_len = manifest
        .buckets
        .iter()
        .map(|b| b.ctx_band as usize)
        .max()
        .unwrap_or(8192);

    let config = if missing.is_empty() {
        let plan = resolved.bind(&cache)?;
        println!("plan           bound, hash {}", plan.plan_hash());
        SchedulerConfig::from_plan(&plan, KvGeometry::PROD_FP8, arenas, max_model_len)?
    } else {
        // Fail loud rather than quietly: this is the one branch on this path, it is announced,
        // and it is legal only because a mock executor selects no implementation and launches no
        // kernel. `saber explain --autotune <cache>` prints the work list this names.
        println!(
            "plan           NOT BOUND: {} of {} routes are untuned, so no implementation is \
             selected.\n\
             \x20              The mock serve continues on the manifest's bucket geometry alone \
             (no kernel is launched\n\
             \x20              on this path). Run `saber explain {} --autotune <cache>` for the \
             work list.",
            missing.len(),
            resolved.requirements().len(),
            args.manifest.display(),
        );
        let buckets: Vec<Bucket> = manifest
            .buckets
            .iter()
            .map(|b| manifest.bucket(b))
            .collect();
        let gdn_bytes_per_slot = model
            .gdn_state_bytes_per_layer_per_slot
            .div_ceil(u64::from(model.tp))
            * u64::from(model.gdn_layers);
        SchedulerConfig::for_mock(
            buckets,
            model.tp,
            gdn_bytes_per_slot,
            KvGeometry::PROD_FP8,
            arenas,
            max_model_len,
            format!("mock:resolved:{}", resolved.input_hash()),
        )?
    };

    // ── the model plane: the real template and the real tokenizer ───────────────────────────
    let tokenizer = Arc::new(
        saber_model::Tokenizer::from_file(&args.tokenizer)
            .with_context(|| format!("loading {}", args.tokenizer.display()))?,
    );
    let renderer = Arc::new(
        TemplateRenderer::from_file(&args.chat_template)
            .with_context(|| format!("loading {}", args.chat_template.display()))?,
    );
    let eos = eos_id(&tokenizer)?;

    // ── the script, as token ids ────────────────────────────────────────────────────────────
    let script = read_script(&args.mock)?;
    let mut responses: Vec<Vec<TokenId>> = Vec::with_capacity(script.response.len());
    for r in &script.response {
        let mut ids = tokenizer
            .encode(&r.text)
            .with_context(|| "tokenizing a scripted response")?;
        ids.push(eos);
        responses.push(ids);
    }
    let executor = ScriptExecutor::new(responses.clone(), script.tokens_per_step, eos)
        .with_step_delay(std::time::Duration::from_millis(args.step_delay_ms));

    // ── the byte ledger: the arenas are reserved before a request is accepted ───────────────
    let mut ledger = Ledger::new();
    for rank in 0..config.tp {
        ledger.declare(Pool::Vram(rank), device.vram_bytes)?;
    }
    ledger.declare(Pool::HostPinned, 4 * GIB)?;
    ledger.declare(Pool::HostPageable, 48 * GIB)?;
    ledger.declare(Pool::Nvme, 64 * GIB)?;

    let model_info = mock_model_info(
        &script.served_model_name,
        &manifest.model.path.display().to_string(),
        eos,
        config.max_context_tokens,
    );
    // The parking tier, if one was asked for. The *options* live in `saber_sched::serve` so the
    // in-process gate composes the server exactly as this binary does; these flags only fill them
    // in. `image_bytes` is the whole session — KV for the longest context this mock will serve
    // plus one GDN state slot set — because a slot holds one session and nothing less is useful.
    let park = ServeParkOptions::from_gib(
        args.park_ram_gib,
        args.park_nvme_gib,
        args.park_nvme_path.clone(),
    )
    .with_image_bytes(
        config.kv.blocks_for(config.max_context_tokens as u64) * config.kv.block_bytes()
            + config.gdn_bytes_per_seq(),
    )
    .with_policy(if args.park_on_pressure {
        ParkPolicy::on_pressure()
    } else {
        ParkPolicy::default()
    });
    let scheduler = attach_park_tier(Scheduler::new(config, executor, model_info, ledger)?, &park)?;
    let scheduler = Arc::new(scheduler);
    let (free_kv, free_state) = scheduler.arena_free();
    let cfg = scheduler.config();

    if park.enabled() {
        println!(
            "parking        ram {} MiB, nvme {} MiB at {}, image {} MiB, on-pressure {}",
            park.ram_bytes / (1024 * 1024),
            park.nvme_bytes / (1024 * 1024),
            park.nvme_path
                .as_ref()
                .map_or_else(|| "-".to_owned(), |p| p.display().to_string()),
            park.image_bytes / (1024 * 1024),
            cfg.park.on_pressure,
        );
    }
    println!("manifest       {}", args.manifest.display());
    println!("script         {}", args.mock.display());
    println!(
        "model          {} (tp {}), eos {}, context {}",
        script.served_model_name, cfg.tp, eos, cfg.max_context_tokens
    );
    println!(
        "buckets        {} captured, widths {:?}, max batch {}",
        cfg.buckets.len(),
        cfg.width.widths,
        cfg.max_batch()
    );
    println!(
        "arenas         kv {} blocks of {} tokens, {} GDN state slot sets of {} slots",
        free_kv,
        cfg.kv.block_tokens,
        free_state,
        cfg.gdn_slots_per_seq()
    );
    println!(
        "script         {} response(s), {} tokens per decode step ({} ms), {} steps for the \
         first one",
        responses.len(),
        script.tokens_per_step,
        args.step_delay_ms,
        scheduler.with_executor(|e| e.steps_for(responses[0].len())),
    );

    // The step loop is a plain thread on purpose; tokio serves HTTP and never steps.
    let steps = spawn_step_loop(Arc::clone(&scheduler));

    let state = build_state_with_parsers(
        Arc::clone(&scheduler),
        Arc::new(ModelCodec::new(Arc::clone(&tokenizer))),
        renderer,
        args.max_logprobs,
        &args.parsers,
    );

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building the tokio runtime")?;
    let addr = format!("{}:{}", args.host, args.port);
    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .with_context(|| format!("binding {addr}"))?;
        let bound = listener.local_addr().context("local addr")?;
        println!("listening      http://{bound}/v1  (SIGINT or SIGTERM to stop)");
        saber_serve::server::serve_with_shutdown(state, listener, crate::engine::shutdown_signal())
            .await
            .context("serving")
    })?;

    match saber_sched::stop_with_deadline(scheduler, steps, std::time::Duration::from_secs(30)) {
        Ok(ran) => {
            println!("stopped        {ran} engine steps");
            Ok(())
        }
        Err(e) => {
            eprintln!("stuck          {e}");
            std::process::abort();
        }
    }
}
