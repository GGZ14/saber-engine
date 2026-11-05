//! Fixtures shared by the engine's parking gates: a scheduler over [`RealExecutor`] with a real
//! parking tier under it, and the request/drain helpers.
//!
//! Deliberately separate from `engine_seam.rs`'s own helpers: that file is the P3 seam gate and
//! belongs to its workstream, and a parking gate should not make it recompile. The scheduler it
//! builds is the same shape — the engine's own facts decide the KV geometry and the arenas, so the
//! scheduler's pages are the engine's pages and not a second guess at them.

#![allow(dead_code)]

use std::sync::{Arc, Mutex};

use saber_engine::{ForwardRunner, NullForward, RealExecutor};
use saber_mem::ledger::{Ledger, Pool};
use saber_plan::Bucket;
use saber_sched::{
    ArenaBytes, KvGeometry, ParkPolicy, Parkable, Scheduler, SchedulerConfig, SessionImage,
    TieredParkConfig, TieredParkStore,
};
use saber_serve::engine::{EngineRequest, StepEvent, Submission, TokenId};
use saber_serve::sampling::SamplingParams;

const GIB: u64 = 1 << 30;
/// The prefill shape ladder, as `saber-sched`'s chunking rule would emit it.
pub const PREFILL: u32 = 2048;
/// Decode batches the stub plans. Batch > 1 is what lets a second sequence be live at the same
/// time, which is what a different state slot and the pressure policy both need.
pub const BATCHES: &[u32] = &[1, 2, 4];

/// A prompt of `n` distinct ids, none of them 0 or an EOS.
pub fn prompt_ids(n: usize) -> Vec<TokenId> {
    (0..n as TokenId).map(|i| 1000 + i).collect()
}

pub fn request(prompt: &[TokenId], max_tokens: usize) -> EngineRequest {
    EngineRequest {
        prompt_token_ids: prompt.to_vec(),
        params: SamplingParams {
            max_tokens: Some(max_tokens),
            // TEMPERATURE 0. Every test built on this helper compares IDS — a parked session's
            // stream against the unparked one, a batched sequence's against its solo run — and
            // `SamplingParams::default()` carries the reference server's own 1.0, which since the sampler was
            // wired into the step loop means a DRAW. Two draws are not a comparison.
            temperature: 0.0,
            ..SamplingParams::default()
        },
    }
}

pub async fn drain(sub: Submission) -> Vec<StepEvent> {
    use futures_util::StreamExt;
    sub.events.collect().await
}

pub fn tokens_of(events: &[StepEvent]) -> Vec<TokenId> {
    events.iter().flat_map(|e| e.new_tokens.clone()).collect()
}

/// A stub with `max_seqs` sequence slots and the batch ladder above.
pub fn stub(max_seqs: u32) -> NullForward {
    NullForward::new(&[PREFILL], BATCHES, max_seqs)
}

/// The same, parking **off the step loop**: every transfer takes `polls` polls to finish.
pub fn async_stub(max_seqs: u32, polls: u32) -> NullForward {
    stub(max_seqs).with_async_park(polls)
}

/// A scheduler whose engine parks asynchronously, with a real RAM tier under it.
pub fn scheduler_async_park(gdn_slots: u64, polls: u32) -> Scheduler<RealExecutor<NullForward>> {
    build(async_stub(4, polls), gdn_slots).with_park_tier(Box::new(store()))
}

/// The same with **two** staging slots, so a park and a restore overlap instead of taking turns.
pub fn scheduler_two_slots(gdn_slots: u64, polls: u32) -> Scheduler<RealExecutor<NullForward>> {
    build(async_stub(4, polls).with_two_slots(), gdn_slots).with_park_tier(Box::new(store()))
}

fn ledger() -> Ledger {
    let mut l = Ledger::new();
    l.declare(Pool::Vram(0), 32 * GIB).expect("rank 0");
    l.declare(Pool::HostPinned, 4 * GIB).expect("pinned");
    l.declare(Pool::HostPageable, 48 * GIB).expect("pageable");
    l.declare(Pool::Nvme, 16 * GIB).expect("nvme");
    l
}

/// The scheduler the engine gets. `gdn_slots` is the state arena's size in **slots**, and a
/// sequence costs `max_spec_width + 1` of them (2 here), so it bounds how many sequences can be
/// resident at once — the pressure gate needs that to be smaller than the requests in flight.
fn build(forward: NullForward, gdn_slots: u64) -> Scheduler<RealExecutor<NullForward>> {
    build_with(forward, gdn_slots, ParkPolicy::default())
}

fn build_with(
    forward: NullForward,
    gdn_slots: u64,
    policy: ParkPolicy,
) -> Scheduler<RealExecutor<NullForward>> {
    let f = forward.facts();
    let pool_blocks = f.kv_pool_blocks;
    let kv = KvGeometry {
        block_tokens: 16,
        bytes_per_token_per_rank: f.kv_block_bytes / 16,
    };
    let gdn_bytes_per_slot = f.gdn_bytes_per_slot;
    let mut buckets: Vec<Bucket> = BATCHES
        .iter()
        .map(|b| Bucket::decode(*b, 1, 8192, true))
        .collect();
    buckets.push(Bucket::prefill(PREFILL, 8192, true));
    let config = SchedulerConfig::for_mock(
        buckets,
        1,
        gdn_bytes_per_slot,
        kv,
        ArenaBytes {
            kv_per_rank: u64::from(pool_blocks) * kv.block_bytes(),
            gdn_state_per_rank: gdn_slots * gdn_bytes_per_slot,
        },
        65_536,
        "test:park-seam".to_owned(),
    )
    .expect("config")
    .with_park_policy(policy);
    Scheduler::new(
        config,
        RealExecutor::new(forward),
        saber_serve::engine::ModelInfo {
            served_model_name: "Qwen3.8".to_owned(),
            aliases: Vec::new(),
            max_model_len: 65_536,
            root: "/packs/qwen38-27b-nvfp4-devfinal-tp1".to_owned(),
            eos_token_ids: vec![151_645],
        },
        ledger(),
    )
    .expect("arenas fit")
}

/// A scratch directory on a filesystem that supports `O_DIRECT`.
///
/// Under `target/` rather than `std::env::temp_dir()`: `/tmp` is tmpfs here and tmpfs has no
/// `O_DIRECT`, so a test that used it would exercise the buffered path instead of the one the tier
/// ships (`docs/LEDGER.md`, 2026-09-18).
pub struct Scratch(pub std::path::PathBuf);

impl Scratch {
    pub fn new(tag: &str) -> Self {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let root = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../target"))
            .join("saber-engine-park");
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = root.join(format!("{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Scratch(dir)
    }

    pub fn file(&self, name: &str) -> std::path::PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A tiered store with room for **one** image in RAM and three on NVMe.
///
/// One RAM slot is the whole point: the second park has nowhere to go until the first spills, so the
/// demote path is on the ordinary route rather than behind a knob. `promoting_on_slack` lets the
/// scheduler pull a spilled image back on an idle step, which is the other half of the tier that had
/// never run under the real engine.
pub fn nvme_store(path: &std::path::Path) -> TieredParkStore {
    TieredParkStore::host(
        &TieredParkConfig::ram(1 << 20, 1 << 20)
            .with_nvme(path, 3 << 20)
            .hashing_on(2)
            .promoting_on_slack(),
    )
    .expect("a RAM+NVMe store")
}

/// A scheduler over the real engine with a RAM+NVMe store under it.
pub fn scheduler_with_nvme(
    gdn_slots: u64,
    polls: u32,
    path: &std::path::Path,
) -> Scheduler<RealExecutor<NullForward>> {
    let f = if polls == 0 {
        stub(4)
    } else {
        async_stub(4, polls)
    };
    build(f, gdn_slots).with_park_tier(Box::new(nvme_store(path)))
}

fn store() -> TieredParkStore {
    // One session image is a handful of stub pages plus a slot; a 1 MiB slot holds one with room to
    // spare. The shipped tier, not a mock — the point of these gates is that the engine, the
    // scheduler and the tier meet.
    TieredParkStore::host(&TieredParkConfig::ram(1 << 20, 16 << 20).hashing_on(2))
        .expect("ram store")
}

/// A scheduler with **no** parking tier: `park` is refused with `NoTier`, as before this branch.
pub fn scheduler_no_park() -> Scheduler<RealExecutor<NullForward>> {
    build(stub(4), 16)
}

/// A scheduler with a real RAM parking tier under it and `gdn_slots` state slots.
pub fn scheduler_with_park(gdn_slots: u64) -> Scheduler<RealExecutor<NullForward>> {
    build(stub(4), gdn_slots).with_park_tier(Box::new(store()))
}

/// The same, with the eviction policy armed: an admission refused for KV or GDN state parks the
/// longest-idle sequence instead of waiting.
pub fn scheduler_with_pressure_policy(gdn_slots: u64) -> Scheduler<RealExecutor<NullForward>> {
    build_with(
        stub(4),
        gdn_slots,
        // K5's victim order: these tests name A (the longest) as the victim and are about the
        // engine seam, not the order; prod's order has its own tests in saber-sched (ws-k18).
        ParkPolicy::on_pressure()
            .min_resident_steps(0)
            .victim(saber_sched::VictimOrder::MostWorkLeft),
    )
    .with_park_tier(Box::new(store()))
}

/// The same policy over an engine that parks **off the step loop** (`polls` polls per transfer),
/// which is the serve plane's path on the card (ws-k17).
pub fn scheduler_async_pressure_policy(
    gdn_slots: u64,
    polls: u32,
) -> Scheduler<RealExecutor<NullForward>> {
    build_with(
        async_stub(4, polls),
        gdn_slots,
        // K5's victim order: these tests name A (the longest) as the victim and are about the
        // engine seam, not the order; prod's order has its own tests in saber-sched (ws-k18).
        ParkPolicy::on_pressure()
            .min_resident_steps(0)
            .victim(saber_sched::VictimOrder::MostWorkLeft),
    )
    .with_park_tier(Box::new(store()))
}

/// A scheduler whose tier damages one byte of the image's **payload** on the way back.
///
/// The flip has to land after the tier has verified its own copy, or the tier catches it and the
/// engine never sees it — which would test the tier (ws-k5 already does) instead of the engine's
/// own chunk check on the live path.
pub fn scheduler_with_damaged_park() -> (Scheduler<RealExecutor<NullForward>>, Arc<Mutex<DamageLog>>)
{
    let log = Arc::new(Mutex::new(DamageLog::default()));
    let tier = DamagingTier {
        inner: store(),
        log: Arc::clone(&log),
    };
    (build(stub(4), 16).with_park_tier(Box::new(tier)), log)
}

/// What the damaging tier did.
#[derive(Debug, Default)]
pub struct DamageLog {
    /// Flip the next image that comes back.
    pub arm: bool,
    /// Images it actually damaged.
    pub damaged: usize,
}

#[derive(Debug)]
struct DamagingTier {
    inner: TieredParkStore,
    log: Arc<Mutex<DamageLog>>,
}

impl Parkable for DamagingTier {
    fn pool(&self) -> Pool {
        self.inner.pool()
    }

    fn park(
        &mut self,
        image: SessionImage,
        ledger: &mut Ledger,
    ) -> Result<saber_sched::ParkTicket, saber_sched::ParkError> {
        self.inner.park(image, ledger)
    }

    fn restore(
        &mut self,
        ticket: &saber_sched::ParkTicket,
        ledger: &mut Ledger,
    ) -> Result<SessionImage, saber_sched::ParkError> {
        let mut image = self.inner.restore(ticket, ledger)?;
        let mut log = self.log.lock().expect("damage log");
        if log.arm {
            log.arm = false;
            log.damaged += 1;
            // Late in the payload, past the header and the digest list, so the image still decodes
            // and it is the chunk check that has to catch it.
            let n = image.state.0.len();
            image.state.0[n - 17] ^= 0x01;
        }
        Ok(image)
    }

    fn discard(&mut self, ticket: &saber_sched::ParkTicket, ledger: &mut Ledger) {
        self.inner.discard(ticket, ledger);
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn counters(&self) -> saber_sched::ParkCounters {
        self.inner.counters()
    }
}
