//! Standing the serving plane up over the real scheduler, with a mock executor.
//!
//! This is the composition `saber serve --mock` performs and that `tests/e2e_http.rs` performs
//! in-process: **one** place where the serving plane, the scheduler and an executor are bolted
//! together, so the binary and the gate cannot drift apart.
//!
//! Three shapes are fixed here because they are prod's, not preferences:
//!
//! * **The step loop is a plain `std::thread`.** `ARCHITECTURE.md` § Rust-specific hazards: the
//!   step loop wants fixed cadence, not throughput, and a work-stealing scheduler adds variance to
//!   exactly that path. tokio owns HTTP; [`spawn_step_loop`] owns steps; they meet at the event
//!   channel.
//! * **Both parsers are on.** `--enable-auto-tool-choice` with `qwen3_coder` and the `qwen3`
//!   reasoning parser whose field is `reasoning`, because that is the parity checklist
//!   (`ARCHITECTURE.md` § Scope) and a server that answers without them is not the thing being
//!   measured.
//! * **One SSE update per engine step** falls out of the [`Engine`](saber_serve::Engine) contract
//!   the scheduler already implements; [`ScriptExecutor`] exists so a step's *width* can be set
//!   deliberately and a client's `completion_tokens / n_chunks` can be predicted exactly.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use saber_serve::detok::TextCodec;
use saber_serve::engine::{ModelInfo, TokenId};
use saber_serve::sampling::DefaultSampling;
use saber_serve::state::{ChatRenderer, ResolvedParsers, ServerState};

use crate::executor::{
    AdmittedSeq, Forked, SeqId, SeqState, SeqStep, StepBatch, StepExecutor, StepResult,
};
use crate::mock::{record, RecordedBatch};
use crate::park::{ParkError, ParkPolicy, TieredParkConfig, TieredParkStore};
use crate::sched::{Scheduler, StepOutcome};

/// A gate that decides when a step may run, so a test can bound on **state** rather than on time.
///
/// A step's pacing is otherwise the only handle a test has on the engine, and wall-clock bounds
/// under a loaded machine are how a gate becomes a coin flip: the disconnect test in
/// `tests/e2e_http.rs` failed 4 runs in 5 of a full `cargo test --workspace` that way. With a gate
/// the test releases steps one at a time and asserts against how many it released, so a slow
/// machine makes the test slower and never wrong.
///
/// This works because the executor call no longer holds the engine lock (see [`Scheduler`]):
/// blocking inside `execute` stops the *device*, not `stats()`, `submit` or `abort`. Before that
/// split this type would have deadlocked every question the serving plane asks.
#[derive(Debug, Default)]
pub struct StepGate {
    state: Mutex<GateState>,
    permit: Condvar,
}

#[derive(Debug, Default)]
struct GateState {
    permits: u64,
    taken: u64,
}

impl StepGate {
    /// A gate with no permits: the first step blocks until [`StepGate::release`].
    pub fn closed() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Lets `n` more steps run.
    pub fn release(&self, n: u64) {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        st.permits += n;
        drop(st);
        self.permit.notify_all();
    }

    /// Steps that have run through this gate.
    pub fn taken(&self) -> u64 {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).taken
    }

    /// Blocks until a permit is available, then consumes it.
    ///
    /// # Panics
    ///
    /// After 30 s without a permit. A test that forgets to release one would otherwise hang, and a
    /// test that hangs is a test nobody runs.
    fn wait(&self) {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        while st.permits == 0 {
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            assert!(!left.is_zero(), "StepGate: no permit within 30 s");
            let (guard, _) = self
                .permit
                .wait_timeout(st, left)
                .unwrap_or_else(|e| e.into_inner());
            st = guard;
        }
        st.permits -= 1;
        st.taken += 1;
    }
}

/// An executor that replays fixed token sequences, `tokens_per_step` at a time.
///
/// It is the mock a *serving* test needs rather than the one a scheduling test needs: responses
/// are unbounded in number (they cycle), so a benchmark client that sends hundreds of requests is
/// answered without a script per request, and the step width is fixed, so
/// `completion_tokens / n_chunks` — the ratio BetterBench classifies a stream by — is arithmetic
/// rather than an observation.
///
/// The shape it emits is prod's:
///
/// * the **prefill** step commits exactly one token (it verifies nothing, so it can commit nothing
///   else), which is why a real stream's first content chunk carries one token and the measured
///   tokens-per-update sits just under the step width;
/// * every later step commits `min(tokens_per_step, verify_width + 1)` tokens, reporting
///   `accepted = tokens - 1` against a scheduled width it never exceeds;
/// * the last token of a response is the EOS id, so the request finishes with `stop` through the
///   scheduler's own stop rule rather than by the executor deciding it is done.
#[derive(Debug)]
pub struct ScriptExecutor {
    responses: Vec<Vec<TokenId>>,
    tokens_per_step: u32,
    eos: TokenId,
    next: usize,
    live: BTreeMap<SeqId, Live>,
    batches: Vec<RecordedBatch>,
    admits: u64,
    releases: u64,
    step_delay: Duration,
    gate: Option<Arc<StepGate>>,
}

#[derive(Debug)]
struct Live {
    tokens: Vec<TokenId>,
    cursor: usize,
}

impl ScriptExecutor {
    /// Cycles `responses`, committing `tokens_per_step` tokens per decode step and ending each
    /// one with `eos`.
    ///
    /// # Panics
    ///
    /// If `responses` is empty or `tokens_per_step` is zero — both are configuration errors the
    /// caller is expected to have refused with a message.
    pub fn new(responses: Vec<Vec<TokenId>>, tokens_per_step: u32, eos: TokenId) -> Self {
        assert!(!responses.is_empty(), "ScriptExecutor needs a response");
        assert!(tokens_per_step >= 1, "tokens_per_step must be >= 1");
        Self {
            responses,
            tokens_per_step,
            eos,
            next: 0,
            live: BTreeMap::new(),
            batches: Vec::new(),
            admits: 0,
            releases: 0,
            step_delay: Duration::ZERO,
            gate: None,
        }
    }

    /// Wall-clock time a step takes, so a stream paces like a real one.
    ///
    /// A GPU step is ~22 ms on this box; a mock step is ~2 µs, which makes a whole response
    /// arrive inside one client read and leaves nothing running to observe. The delay is spent in
    /// [`StepExecutor::execute`] — where the kernel time would be — and never on a tokio thread.
    #[must_use]
    pub fn with_step_delay(mut self, delay: Duration) -> Self {
        self.step_delay = delay;
        self
    }

    /// Runs steps only when `gate` says so. See [`StepGate`].
    #[must_use]
    pub fn with_gate(mut self, gate: Arc<StepGate>) -> Self {
        self.gate = Some(gate);
        self
    }

    /// Every batch executed, for a test that wants to compare chunks against steps.
    pub fn batches(&self) -> &[RecordedBatch] {
        &self.batches
    }

    /// Admits and releases seen. Equal once nothing is running.
    pub fn admit_release_counts(&self) -> (u64, u64) {
        (self.admits, self.releases)
    }

    /// Sequences whose state the executor still holds.
    pub fn live_sequences(&self) -> usize {
        self.live.len()
    }

    /// How many **decode** steps a response of `n` tokens takes: the step that carries the first
    /// token (the last prefill chunk, or the one prefill-shaped decode step before P2), then
    /// `tokens_per_step` at a time. The EOS token is one of the `n`. Prefill chunks before it are
    /// counted by [`crate::SchedulerConfig::prefill`], not here.
    pub fn steps_for(&self, n: usize) -> usize {
        if n <= 1 {
            return n.max(1);
        }
        1 + (n - 1).div_ceil(self.tokens_per_step as usize)
    }
}

impl StepExecutor for ScriptExecutor {
    fn execute(&mut self, batch: &StepBatch<'_>) -> StepResult {
        if let Some(gate) = &self.gate {
            gate.wait();
        }
        if !self.step_delay.is_zero() {
            thread::sleep(self.step_delay);
        }
        self.batches.push(record(batch));
        let (per_step, eos) = (self.tokens_per_step, self.eos);
        let mut seqs = Vec::with_capacity(batch.slots.len());
        for slot in batch.slots {
            let live = self
                .live
                .get_mut(&slot.seq_id)
                .expect("ScriptExecutor: a step for a sequence that was never admitted");
            // A prefill chunk feeds prompt tokens and samples nothing until the last one, which
            // yields the sequence's first token -- the same token an unchunked prefill would.
            if let Some(chunk) = slot.chunk {
                let tokens = if chunk.last {
                    let t = live.tokens.get(live.cursor).copied().unwrap_or(eos);
                    live.cursor = (live.cursor + 1).min(live.tokens.len());
                    vec![t]
                } else {
                    Vec::new()
                };
                seqs.push(SeqStep {
                    seq_id: slot.seq_id,
                    tokens,
                    draft_tokens: 0,
                    accepted_tokens: 0,
                    logprobs: None,
                    prompt_logprobs: crate::mock::synthetic_prompt_logprobs(slot),
                });
                continue;
            }
            // A prefill step verifies nothing, so it can only commit the one token the target
            // sampled; a decode step commits what the scheduler scheduled, never more.
            let want = if slot.first_step {
                1
            } else {
                per_step.min(slot.verify_width + 1) as usize
            };
            let remaining = live.tokens.len() - live.cursor;
            let n = want.min(remaining).max(1);
            let tokens: Vec<TokenId> = if remaining == 0 {
                // The scheduler stops on the EOS the response ends with, so this is unreachable
                // unless a stop rule changed underneath. Emit the EOS again rather than an empty
                // commit, which would break `tokens == accepted + 1`.
                vec![eos]
            } else {
                live.tokens[live.cursor..live.cursor + n].to_vec()
            };
            live.cursor += tokens.len().min(remaining);
            let accepted = tokens.len() as u32 - 1;
            seqs.push(SeqStep {
                seq_id: slot.seq_id,
                tokens,
                // The scheduled width, which for this executor is exactly what it consumed.
                draft_tokens: accepted,
                accepted_tokens: accepted,
                logprobs: None,
                prompt_logprobs: None,
            });
        }
        StepResult { seqs }
    }

    fn admit(&mut self, seq: &AdmittedSeq) {
        let mut tokens = self.responses[self.next % self.responses.len()].clone();
        self.next += 1;
        if tokens.last() != Some(&self.eos) {
            tokens.push(self.eos);
        }
        self.live.insert(seq.seq_id, Live { tokens, cursor: 0 });
        self.admits += 1;
    }

    fn release(&mut self, seq: SeqId) {
        self.live.remove(&seq);
        self.releases += 1;
    }

    /// A branch takes the next scripted response, exactly as a fresh admission does. That is the
    /// whole of what a host-side mock can do about a fork, and it is what makes `n = 3` comparable
    /// to three single submits token for token.
    fn can_fork(&self) -> bool {
        true
    }

    /// The mock answers prompt logprobs with the synthetic seam the real law runs over, so the
    /// serve plane's shape is exercised end to end without a card.
    fn max_prompt_logprobs(&self) -> Option<u32> {
        Some(crate::mock::SYNTHETIC_PROMPT_K as u32)
    }

    /// Admit the branch and hand back the first token of its response — the token the last prefill
    /// chunk used to commit, which a branch no longer runs because it never prefills.
    fn fork(&mut self, _leader: SeqId, branch: &AdmittedSeq) -> Option<Forked> {
        self.admit(branch);
        let live = self.live.get_mut(&branch.seq_id)?;
        let t = live.tokens.get(live.cursor).copied().unwrap_or(self.eos);
        live.cursor = (live.cursor + 1).min(live.tokens.len());
        Some(Forked::token(t))
    }

    /// The cursor and the response it is walking — this executor's whole per-sequence state.
    ///
    /// It exists so that `saber serve --mock` can exercise parking end to end. A mock that
    /// refused to park would make `--park-ram-gib` a flag that configures a tier nothing ever
    /// uses, which is worse than not having it.
    fn park_state(&mut self, seq: SeqId) -> Option<SeqState> {
        let live = self.live.get(&seq)?;
        let mut bytes = Vec::with_capacity(8 + live.tokens.len() * 4);
        bytes.extend_from_slice(&(live.cursor as u64).to_le_bytes());
        for t in &live.tokens {
            bytes.extend_from_slice(&t.to_le_bytes());
        }
        Some(SeqState(bytes))
    }

    fn restore_state(&mut self, seq: &AdmittedSeq, state: &SeqState) -> bool {
        if state.0.len() < 8 || (state.0.len() - 8) % 4 != 0 {
            return false;
        }
        let cursor = u64::from_le_bytes(state.0[0..8].try_into().expect("8 bytes")) as usize;
        let tokens: Vec<TokenId> = state.0[8..]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().expect("4 bytes")))
            .collect();
        if cursor > tokens.len() {
            return false;
        }
        // `admit` ran first and handed this slot the next response in the cycle; the restore
        // replaces it, which is what makes the resumed stream identical to the unparked one.
        self.live.insert(seq.seq_id, Live { tokens, cursor });
        true
    }
}

/// What a mock serve needs to stand a parking tier up: `--park-ram-gib` and `--park-nvme-gib`, in
/// the one place the binary and the gate both read.
///
/// The flags live here rather than in `saber-cli` on purpose — the binary's argument file is
/// shared with several workstreams, and a serving option that only the binary knows about is one
/// the in-process gate cannot exercise. `saber serve --mock` builds one of these from its flags
/// and passes it to [`attach_park_tier`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServeParkOptions {
    /// Host RAM the parking tier may hold. Zero means no RAM tier.
    pub ram_bytes: u64,
    /// Parking-file bytes. Zero means no NVMe tier.
    pub nvme_bytes: u64,
    /// Where the parking file lives.
    pub nvme_path: Option<PathBuf>,
    /// The largest session image a slot must hold. Prod's is ~136 MiB (64 MiB of fp8 KV at 4k
    /// tokens plus 72 MiB of GDN state per rank); a mock's is a few hundred bytes, and the
    /// default below is the prod figure so the slot arithmetic a flag produces is the real one.
    pub image_bytes: u64,
    /// When the scheduler may evict on its own.
    pub policy: ParkPolicy,
}

impl Default for ServeParkOptions {
    fn default() -> Self {
        Self {
            ram_bytes: 0,
            nvme_bytes: 0,
            nvme_path: None,
            image_bytes: 136 * 1024 * 1024,
            policy: ParkPolicy::default(),
        }
    }
}

impl ServeParkOptions {
    /// Bytes in one GiB, so a `--park-ram-gib` flag has one conversion and not three.
    pub const GIB: u64 = 1024 * 1024 * 1024;

    /// From the two flags, in GiB. Fractions are honoured: `--park-ram-gib 0.5` is 512 MiB.
    #[must_use]
    pub fn from_gib(ram_gib: f64, nvme_gib: f64, nvme_path: Option<PathBuf>) -> Self {
        let bytes = |g: f64| (g.max(0.0) * Self::GIB as f64) as u64;
        Self {
            ram_bytes: bytes(ram_gib),
            nvme_bytes: bytes(nvme_gib),
            nvme_path,
            ..Self::default()
        }
    }

    /// Set the eviction policy.
    #[must_use]
    pub fn with_policy(mut self, policy: ParkPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Set the per-session image size a slot must hold.
    #[must_use]
    pub fn with_image_bytes(mut self, bytes: u64) -> Self {
        self.image_bytes = bytes;
        self
    }

    /// Whether any tier was asked for.
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.ram_bytes > 0 || self.nvme_bytes > 0
    }

    /// The store these options describe, or `None` when no tier was asked for.
    ///
    /// Fails loud: a budget smaller than one session slot, or an NVMe budget with no path, is a
    /// refusal to start rather than a tier that silently is not there.
    pub fn build(&self) -> Result<Option<TieredParkStore>, ParkError> {
        if !self.enabled() {
            return Ok(None);
        }
        if self.nvme_bytes > 0 && self.nvme_path.is_none() {
            return Err(ParkError::Ledger(
                "an NVMe parking budget needs a path for the parking file".to_owned(),
            ));
        }
        let mut cfg = TieredParkConfig::ram(self.image_bytes, self.ram_bytes);
        if let (Some(path), true) = (self.nvme_path.as_ref(), self.nvme_bytes > 0) {
            cfg = cfg.with_nvme(path.clone(), self.nvme_bytes);
        }
        TieredParkStore::host(&cfg).map(Some)
    }
}

/// Attach the parking tier these options describe, if any.
///
/// The scheduler is taken and returned because [`Scheduler::with_park_tier`] consumes it: a tier
/// is attached once, at composition, and never swapped underneath a running engine.
pub fn attach_park_tier<E: StepExecutor>(
    scheduler: Scheduler<E>,
    opts: &ServeParkOptions,
) -> Result<Scheduler<E>, ParkError> {
    // THE POLICY TRAVELS WITH THE TIER (ws-k17). The options carry both, and this used to attach
    // the tier alone: the scheduler kept `ParkPolicy::default()` (`on_pressure: false`), so the
    // serve plane's under-pressure arm refused 1,320 admissions for KV and parked nothing, while
    // the boot line printed the OPTIONS' `on-pressure true`. With no tier the policy is inert
    // (`park_under_pressure` needs one) but it is still the scheduler's, so what the scheduler
    // reports is what it runs.
    let scheduler = scheduler.with_park_policy(opts.policy);
    match opts.build()? {
        Some(store) => Ok(scheduler.with_park_tier(Box::new(store))),
        None => Ok(scheduler),
    }
}

/// The serving plane over a scheduler, with prod's parser flags **and prod's sampling
/// defaults**.
///
/// `served_model_name`, the aliases and the EOS ids come from the [`ModelInfo`] the scheduler was
/// built with, so `/v1/models` and the stop rule cannot disagree.
///
/// [`DefaultSampling::PROD`] is the second half of that and it is not cosmetic: prod launches with
/// `--override-generation-config '{"temperature":0.7,"top_p":0.95,"top_k":20}'`
/// (`serve-mxfp4.sh:1072`), so a request that sends no `temperature`/`top_p`/`top_k` is sampled at
/// **0.7 / 0.95 / 20**, not at OpenAI's 1.0 / 1.0 / off. A serving plane that took the OpenAI
/// defaults would sample from a different distribution than prod on every unparameterised request
/// and no wire-level gate would see it (workstream V1, 2026-09-21).
pub fn build_state<E: StepExecutor>(
    scheduler: Arc<Scheduler<E>>,
    codec: Arc<dyn TextCodec>,
    renderer: Arc<dyn ChatRenderer>,
) -> Arc<ServerState> {
    build_state_with(scheduler, codec, renderer, PROD_MAX_LOGPROBS)
}

/// The reference server's default `--max-logprobs`, which the prod unit does not override.
pub const PROD_MAX_LOGPROBS: i32 = 20;

/// [`build_state`] with `--max-logprobs` (V7): the cap on `logprobs`, `top_logprobs` and
/// `prompt_logprobs`, enforced at validation with the reference server's 400 text. `-1` is the whole vocabulary.
pub fn build_state_with<E: StepExecutor>(
    scheduler: Arc<Scheduler<E>>,
    codec: Arc<dyn TextCodec>,
    renderer: Arc<dyn ChatRenderer>,
    max_logprobs: i32,
) -> Arc<ServerState> {
    build_state_with_parsers(
        scheduler,
        codec,
        renderer,
        max_logprobs,
        &ResolvedParsers::prod(),
    )
}

/// [`build_state_with`] with the parser flags (V9): `--enable-auto-tool-choice`,
/// `--tool-call-parser`, `--reasoning-parser`, resolved by `ParserFlags::resolve`. Prod's profile
/// (`ResolvedParsers::prod()`) is what every other builder passes.
pub fn build_state_with_parsers<E: StepExecutor>(
    scheduler: Arc<Scheduler<E>>,
    codec: Arc<dyn TextCodec>,
    renderer: Arc<dyn ChatRenderer>,
    max_logprobs: i32,
    parsers: &ResolvedParsers,
) -> Arc<ServerState> {
    build_state_with_vision(scheduler, codec, renderer, max_logprobs, parsers, 0, 999)
}

/// [`build_state_with_parsers`] plus the image policy (D3): `vision_max_patches` (0 = text-only)
/// and `--limit-mm-per-prompt`'s image count.
pub fn build_state_with_vision<E: StepExecutor>(
    scheduler: Arc<Scheduler<E>>,
    codec: Arc<dyn TextCodec>,
    renderer: Arc<dyn ChatRenderer>,
    max_logprobs: i32,
    parsers: &ResolvedParsers,
    vision_max_patches: u32,
    image_limit: u32,
) -> Arc<ServerState> {
    ServerState::builder(scheduler, codec, renderer)
        .parsers(parsers)
        .default_sampling(DefaultSampling::PROD)
        .max_logprobs(max_logprobs)
        .vision(vision_max_patches, image_limit)
        // FN27 ("ok, align 27b"): tool-call `arguments` reach the template as prod 0.27.1 made
        // them -- a JSON string parsed, malformed JSON its 400.
        .tool_arguments(saber_serve::state::ToolArguments::ObjectsStrict)
        // FN28 ("ok, align that also for 27b"): the template renders the conversation prod
        // 0.27.1 rebuilds (`reasoning_content` -> `reasoning`, both keys on an assistant turn,
        // unknown keys dropped, content in the `openai` form) -- `saber_serve::conversation`.
        .conversation_form(saber_serve::state::ConversationForm::Prod0271)
        .build()
}

/// A running step loop. Dropping it asks the thread to stop; [`StepLoopHandle::stop`] waits for it.
#[derive(Debug)]
pub struct StepLoopHandle {
    stop: Arc<AtomicBool>,
    join: Option<thread::JoinHandle<u64>>,
}

impl StepLoopHandle {
    /// Stops the loop and returns how many steps it ran.
    pub fn stop(mut self) -> u64 {
        self.stop.store(true, Ordering::Relaxed);
        self.join
            .take()
            .map(|j| j.join().unwrap_or_default())
            .unwrap_or_default()
    }
}

impl Drop for StepLoopHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

/// A teardown that did not finish in time.
///
/// The caller is a binary that must now leave without running a destructor, because whatever is
/// stuck is stuck inside the device runtime: see [`stop_with_deadline`].
#[derive(Debug, thiserror::Error)]
#[error(
    "the engine did not shut down within {deadline:?} after {steps} step(s): something is \
         still inside the device runtime"
)]
pub struct StopTimeout {
    /// The deadline that passed.
    pub deadline: std::time::Duration,
    /// Steps the loop had run.
    pub steps: u64,
}

/// Stop the step loop and drop everything the engine owns, **against a deadline**.
///
/// # Why this exists
///
/// Dropping a scheduler drops its executor, which for the real engine joins the HIP thread. If that
/// thread is inside a blocking device call that never returns, the drop never returns either — and
/// a process that cannot exit holds whatever it holds. On 2026-09-19 that was the shared GPU lock
/// and 18.7 GiB of a card, for 2 h 55 m, because a gate script's `saber serve` took a signal,
/// printed its result, and then sat in `wait`.
///
/// So the drop happens on a thread of its own and this function waits on it for `deadline`. On time,
/// everything is released in order and the caller returns normally. Past the deadline the caller is
/// told, and its only correct move is to leave **without** running destructors: `std::process::exit`
/// with a live HIP thread segfaults (`docs/LEDGER.md`, 2026-09-18), so the binary aborts.
///
/// `owned` is everything whose drop must happen there: the scheduler, the server state, anything
/// else holding an `Arc` of either. Passing them here rather than dropping them at the call site is
/// what makes "the teardown is on its own thread" a fact rather than an intention.
pub fn stop_with_deadline<T: Send + 'static>(
    owned: T,
    steps: StepLoopHandle,
    deadline: std::time::Duration,
) -> Result<u64, StopTimeout> {
    // The loop first: it holds an `Arc` of the scheduler, and it is the one thing guaranteed to
    // stop, because its own check is a flag it reads between steps.
    let ran = steps.stop();
    let (tx, rx) = std::sync::mpsc::sync_channel::<()>(1);
    // Detached on purpose: if it never finishes, joining it here would be the very hang this
    // function exists to bound.
    std::thread::Builder::new()
        .name("saber-teardown".to_owned())
        .spawn(move || {
            drop(owned);
            let _ = tx.send(());
        })
        .map_err(|_| StopTimeout {
            deadline,
            steps: ran,
        })?;
    match rx.recv_timeout(deadline) {
        Ok(()) => Ok(ran),
        Err(_) => Err(StopTimeout {
            deadline,
            steps: ran,
        }),
    }
}

/// How long the step thread sleeps when there is nothing to do.
///
/// Short enough that it is not a latency term worth measuring (it bounds the extra TTFT a request
/// can wait before its first step) and long enough that an idle server does not spin a core.
pub const IDLE_POLL: Duration = Duration::from_micros(200);

/// Runs [`Scheduler::step`] on a plain `std::thread` until the handle is dropped or stopped.
pub fn spawn_step_loop<E: StepExecutor>(scheduler: Arc<Scheduler<E>>) -> StepLoopHandle {
    let stop = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&stop);
    // FN73: a fatal (a failed rank) ends every in-flight and queued request with its reason --
    // from the hooks thread, so a step stuck in the device does not hold the clients -- and the
    // loop stops stepping.
    {
        let weak = Arc::downgrade(&scheduler);
        saber_serve::fatal::on_fatal(move |f| {
            if let Some(s) = weak.upgrade() {
                let n = s.fail_all(&f.line());
                eprintln!(
                    "fatal          {n} request(s) ended with the reason; new requests get 503"
                );
            }
        });
    }
    let join = thread::Builder::new()
        .name("saber-step".to_owned())
        .spawn(move || {
            let mut steps = 0u64;
            while !flag.load(Ordering::Relaxed) {
                if let Some(f) = saber_serve::fatal::get() {
                    scheduler.fail_all(&f.line());
                    break;
                }
                match scheduler.step() {
                    StepOutcome::Ran(_) => {
                        steps += 1;
                        // The step holds the engine lock for as long as the executor runs -- ~22 ms
                        // with a real one. `std::sync::Mutex` is not fair, so a loop that releases
                        // and immediately re-acquires can starve `/metrics` and `submit` for the
                        // length of a whole response. Yielding between steps hands the lock over.
                        // The structural fix (not holding the lock across the device call) is a P3
                        // decision and is recorded in docs/LEDGER.md.
                        thread::yield_now();
                    }
                    StepOutcome::Idle => thread::sleep(IDLE_POLL),
                    // Fail loud, and keep serving the requests that do have a shape: a bucket
                    // that does not exist is a plan problem, not a reason to stop stepping.
                    StepOutcome::NoBucket(e) => {
                        tracing::error!(error = %e, "step loop: no captured bucket");
                        thread::sleep(IDLE_POLL);
                    }
                }
            }
            steps
        })
        .expect("spawn the step thread");
    StepLoopHandle {
        stop,
        join: Some(join),
    }
}

/// The model identity a mock serve reports. Kept here so the binary and the gate agree on it.
pub fn mock_model_info(
    served_model_name: &str,
    root: &str,
    eos: TokenId,
    max_len: usize,
) -> ModelInfo {
    ModelInfo {
        served_model_name: served_model_name.to_owned(),
        aliases: Vec::new(),
        max_model_len: max_len,
        root: root.to_owned(),
        eos_token_ids: vec![eos],
    }
}
