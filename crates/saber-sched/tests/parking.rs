//! Session parking: KV and GDN state leave together, come back byte-identical, or do not leave.
//!
//! `ARCHITECTURE.md` § Memory plane: parking evicts *"an idle session's KV **and** its 72 MiB/rank
//! GDN state as one unit — they are useless apart"*, ~5 ms from RAM and ~12 ms from the Gen5 drive.
//! The gate here is the correctness half of that: a resumed session must continue the same stream
//! it would have produced had it never left, and the image that comes back must be the image that
//! went in. The 09-17 lazy-snapshot result is what this is guarding against — a state transform
//! that every single-shot gate reads null on and that ruins multi-turn chat.

mod common;

use std::sync::{Arc, Mutex};

use common::{drain, request, scheduler, scheduler_with, tokens_of, GIB, MIB};
use saber_mem::ledger::{Ledger, Pool};
use saber_sched::{
    Acceptance, ArenaBytes, HostTier, MockExecutor, ParkError, ParkTicket, Parkable,
    RandomAcceptanceExecutor, SeqScript, SessionImage,
};
use saber_serve::engine::{Engine, FinishReason, TokenId};

/// What the tier saw, so a test can compare the image in with the image out.
#[derive(Debug, Default)]
struct Spy {
    parked: Vec<SessionImage>,
    restored: Vec<SessionImage>,
    /// Adds a token to the restored image, which the scheduler must catch.
    corrupt: bool,
}

/// A [`Parkable`] built outside the crate — which also proves the trait is implementable from
/// outside it.
#[derive(Debug)]
struct SpyTier {
    inner: HostTier,
    log: Arc<Mutex<Spy>>,
}

impl SpyTier {
    fn ram() -> (Self, Arc<Mutex<Spy>>) {
        let log = Arc::new(Mutex::new(Spy::default()));
        (
            Self {
                inner: HostTier::ram(),
                log: Arc::clone(&log),
            },
            log,
        )
    }
}

impl Parkable for SpyTier {
    fn pool(&self) -> Pool {
        self.inner.pool()
    }

    fn park(&mut self, image: SessionImage, ledger: &mut Ledger) -> Result<ParkTicket, ParkError> {
        self.log.lock().expect("spy").parked.push(image.clone());
        self.inner.park(image, ledger)
    }

    fn restore(
        &mut self,
        ticket: &ParkTicket,
        ledger: &mut Ledger,
    ) -> Result<SessionImage, ParkError> {
        let mut image = self.inner.restore(ticket, ledger)?;
        if self.log.lock().expect("spy").corrupt {
            image.tokens.push(999_999);
        }
        self.log.lock().expect("spy").restored.push(image.clone());
        Ok(image)
    }

    fn discard(&mut self, ticket: &ParkTicket, ledger: &mut Ledger) {
        self.inner.discard(ticket, ledger);
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

fn prompt() -> Vec<TokenId> {
    vec![11, 22, 33, 44]
}

/// The same request, start to finish, with nothing parked.
async fn control_tokens(max_tokens: usize) -> Vec<TokenId> {
    let sched = scheduler(RandomAcceptanceExecutor::new(0xB0A7, Acceptance::Fixed(2)));
    let sub = sched
        .submit(request(&prompt(), max_tokens))
        .expect("submit");
    sched.run_steps(500);
    tokens_of(&drain(sub).await)
}

#[tokio::test]
async fn a_parked_session_resumes_byte_identically() {
    let control = control_tokens(60).await;
    assert_eq!(control.len(), 60);

    let (tier, log) = SpyTier::ram();
    let sched = scheduler(RandomAcceptanceExecutor::new(0xB0A7, Acceptance::Fixed(2)))
        .with_park_tier(Box::new(tier));
    let (kv_free, gdn_free) = sched.arena_free();

    let sub = sched.submit(request(&prompt(), 60)).expect("submit");
    sched.run_steps(4);
    let before_park = sched.stats().spec_decode_num_accepted_tokens;

    sched.park(sub.id).expect("park");
    assert_eq!(sched.parked_ids(), vec![sub.id]);
    assert_eq!(
        sched.arena_free(),
        (kv_free, gdn_free),
        "parking frees the KV blocks AND the state slot set, as one unit"
    );
    assert_eq!(sched.stats().num_requests_running, 0);

    // The tier is billed while the session is away, and the move is counted as write traffic.
    let snap = sched.ledger_snapshot();
    let host = snap
        .pools
        .iter()
        .find(|p| p.pool == Pool::HostPageable)
        .expect("host pool");
    // WHAT THE TIER HOLDS, not what the session held on the card (ws-k15): a host pool is billed
    // the encoded image's own length, which is the same number `TieredParkStore` bills.
    let parked_bytes = saber_sched::park_image_bytes(&log.lock().expect("spy").parked[0]) as u64;
    assert_eq!(host.used_bytes, parked_bytes);
    assert_eq!(host.written_bytes, parked_bytes);
    assert_eq!(host.reservations.len(), 1);
    assert!(host.reservations[0].name.starts_with("session:"));

    sched.unpark(sub.id).expect("unpark");
    sched.run_steps(500);

    let events = drain(sub).await;
    assert_eq!(
        tokens_of(&events),
        control,
        "a parked session must continue the stream it would have produced"
    );
    assert_eq!(events.last().unwrap().finished, Some(FinishReason::Length));
    // step_index continues across the park: one event per step, no gap and no repeat.
    for (i, e) in events.iter().enumerate() {
        assert_eq!(e.step_index, i as u32);
    }

    let spy = log.lock().expect("spy");
    assert_eq!(spy.parked.len(), 1);
    assert_eq!(spy.restored.len(), 1);
    assert_eq!(
        spy.parked[0].state, spy.restored[0].state,
        "the state image is byte-identical across the park"
    );
    assert_eq!(spy.parked[0].tokens, spy.restored[0].tokens);
    assert_eq!(
        spy.parked[0].gdn_slots, 6,
        "spec width 5 + 1 committed state"
    );
    assert!(before_park > 0);

    assert_eq!(sched.counters().parks, 1);
    assert_eq!(sched.counters().restores, 1);
    assert_eq!(sched.counters().restore_failures, 0);

    // Everything the tier held is released again.
    let snap = sched.ledger_snapshot();
    let host = snap
        .pools
        .iter()
        .find(|p| p.pool == Pool::HostPageable)
        .expect("host pool");
    assert_eq!(host.used_bytes, 0);
    assert_eq!(
        host.written_bytes, parked_bytes,
        "write traffic is monotone"
    );
}

#[tokio::test]
async fn an_executor_that_cannot_park_is_refused_not_evicted() {
    // MockExecutor does not implement the parking hooks; the default is "cannot park", and the
    // scheduler must refuse rather than evict state it cannot bring back.
    let mut exec = MockExecutor::new();
    exec.push_script(SeqScript::per_token([10, 11, 12, 13]));
    let (tier, _log) = SpyTier::ram();
    let sched = scheduler(exec).with_park_tier(Box::new(tier));

    let sub = sched.submit(request(&[1, 2], 4)).expect("submit");
    sched.run_steps(1);
    let held = sched.arena_free();

    let err = sched
        .park(sub.id)
        .expect_err("the executor has no state image");
    assert!(matches!(err, ParkError::ExecutorCannotPark(_)), "{err}");
    assert_eq!(sched.counters().park_refusals, 1);
    assert_eq!(sched.counters().parks, 0);
    assert_eq!(sched.arena_free(), held, "a refused park changes nothing");
    assert_eq!(sched.stats().num_requests_running, 1);

    sched.run_steps(10);
    let events = drain(sub).await;
    assert_eq!(tokens_of(&events), vec![10, 11, 12, 13]);
}

#[test]
fn parking_without_a_tier_is_refused() {
    let sched = scheduler(RandomAcceptanceExecutor::new(1, Acceptance::Fixed(1)));
    let sub = sched.submit(request(&[1], 8)).expect("submit");
    sched.run_steps(1);
    assert!(matches!(sched.park(sub.id), Err(ParkError::NoTier)));
    assert_eq!(sched.counters().park_refusals, 1);
    assert!(matches!(
        sched.park(saber_serve::engine::RequestId(999)),
        Err(ParkError::NotRunning(_))
    ));
}

#[tokio::test]
async fn a_corrupted_image_fails_the_restore_loud_and_the_session_stays_parked() {
    let (tier, log) = SpyTier::ram();
    let sched = scheduler(RandomAcceptanceExecutor::new(0xBAD, Acceptance::Fixed(1)))
        .with_park_tier(Box::new(tier));
    let sub = sched.submit(request(&prompt(), 40)).expect("submit");
    sched.run_steps(3);
    sched.park(sub.id).expect("park");

    log.lock().expect("spy").corrupt = true;
    sched.unpark(sub.id).expect("queued for restore");
    sched.run_steps(2);

    assert_eq!(sched.counters().restore_failures, 1);
    assert_eq!(sched.counters().restores, 0);
    assert_eq!(sched.stats().num_requests_running, 0);
    assert_eq!(
        sched.parked_ids(),
        vec![sub.id],
        "the session is still parked"
    );
    // The allocations taken for the attempt went straight back.
    let fresh = scheduler(RandomAcceptanceExecutor::new(0, Acceptance::Fixed(1)));
    assert_eq!(sched.arena_free(), fresh.arena_free());

    sched.abort(sub.id);
    assert_eq!(sched.counters().aborts_parked, 1);
    let events = drain(sub).await;
    assert_eq!(events.last().unwrap().finished, Some(FinishReason::Abort));
}

#[tokio::test]
async fn an_aborted_parked_session_releases_the_tier() {
    let (tier, _log) = SpyTier::ram();
    let sched = scheduler(RandomAcceptanceExecutor::new(2, Acceptance::Fixed(1)))
        .with_park_tier(Box::new(tier));
    let sub = sched.submit(request(&prompt(), 40)).expect("submit");
    sched.run_steps(3);
    sched.park(sub.id).expect("park");
    assert!(sched
        .ledger_snapshot()
        .pools
        .iter()
        .any(|p| { p.pool == Pool::HostPageable && p.used_bytes > 0 }));

    sched.abort(sub.id);
    assert!(sched.parked_ids().is_empty());
    let host_used = sched
        .ledger_snapshot()
        .pools
        .iter()
        .find(|p| p.pool == Pool::HostPageable)
        .expect("host pool")
        .used_bytes;
    assert_eq!(
        host_used, 0,
        "an aborted parked session is discarded from the tier"
    );

    let events = drain(sub).await;
    assert_eq!(events.last().unwrap().finished, Some(FinishReason::Abort));
}

#[tokio::test]
async fn a_restore_waits_for_the_space_it_needs() {
    // One state slot set in the whole engine. Parking A frees it, B takes it, and A's restore has
    // to wait -- the same admission rule as a new request, not a privileged path that could OOM.
    let (tier, _log) = SpyTier::ram();
    let sched = scheduler_with(
        &[(1, 1, 8192), (2, 1, 8192)],
        2,
        ArenaBytes {
            kv_per_rank: 2 * GIB,
            gdn_state_per_rank: 144 * MIB,
        },
        RandomAcceptanceExecutor::new(9, Acceptance::Fixed(1)),
    )
    .with_park_tier(Box::new(tier));
    assert_eq!(sched.arena_free().1, 1, "exactly one state slot set");

    let a = sched.submit(request(&[1, 2], 40)).expect("submit");
    sched.run_steps(3);
    sched.park(a.id).expect("park");

    let b = sched.submit(request(&[3, 4], 6)).expect("submit");
    sched.run_steps(1);
    assert_eq!(sched.stats().num_requests_running, 1, "B has the slot set");

    sched.unpark(a.id).expect("queued");
    sched.step();
    assert_eq!(
        sched.stats().num_requests_running,
        1,
        "A cannot restore while B holds the only slot set"
    );
    assert_eq!(sched.stats().num_requests_waiting, 1);
    assert_eq!(sched.counters().restores, 0);
    assert!(sched.admission_counters().refused_gdn >= 1);

    sched.run_steps(200);
    assert_eq!(sched.counters().restores, 1, "A resumes once B finishes");
    let (ea, eb) = (drain(a).await, drain(b).await);
    assert_eq!(tokens_of(&ea).len(), 40);
    assert_eq!(tokens_of(&eb).len(), 6);
    assert_eq!(ea.last().unwrap().finished, Some(FinishReason::Length));
}

// =================================================================================================
// The real tiers and the eviction policy (ws-k5)
// =================================================================================================

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use saber_sched::{
    choose_victim, KvGeometry, ParkCandidate, ParkCounters, ParkPolicy, Scheduler, SchedulerConfig,
    TieredParkConfig, TieredParkStore, VictimOrder,
};
use saber_serve::engine::RequestId;

/// A scratch directory under `target/`: `/tmp` is tmpfs here and tmpfs has no `O_DIRECT`.
struct Scratch(PathBuf);

static SEQ: AtomicU64 = AtomicU64::new(0);

impl Scratch {
    fn new(tag: &str) -> Self {
        let root = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../target"))
            .join("saber-sched-park");
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = root.join(format!("{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Scratch(dir)
    }

    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Every mock session image is well under a page, so one 4 KiB slot holds one session.
const IMAGE_BYTES: u64 = 4096;

fn ram_store(slots: u64) -> TieredParkStore {
    TieredParkStore::host(&TieredParkConfig::ram(IMAGE_BYTES, slots * IMAGE_BYTES).hashing_on(2))
        .expect("ram store")
}

/// The real store with a bit flipped in the image just before it is asked for.
///
/// The fault has to be injected from **inside** the tier, because the whole point is that nothing
/// outside it can tell: the scheduler holds a ticket, not bytes.
#[derive(Debug)]
struct Corruptible {
    inner: TieredParkStore,
    corrupt_next: Arc<AtomicBool>,
}

impl Corruptible {
    fn ram(slots: u64) -> (Self, Arc<AtomicBool>) {
        let flag = Arc::new(AtomicBool::new(false));
        (
            Self {
                inner: ram_store(slots),
                corrupt_next: Arc::clone(&flag),
            },
            flag,
        )
    }
}

impl Parkable for Corruptible {
    fn pool(&self) -> Pool {
        self.inner.pool()
    }

    fn park(&mut self, image: SessionImage, ledger: &mut Ledger) -> Result<ParkTicket, ParkError> {
        self.inner.park(image, ledger)
    }

    fn restore(
        &mut self,
        ticket: &ParkTicket,
        ledger: &mut Ledger,
    ) -> Result<SessionImage, ParkError> {
        if self.corrupt_next.swap(false, Ordering::SeqCst) {
            assert!(
                self.inner.corrupt_for_test(ticket, 24),
                "the fault injector must find the image it is asked to damage"
            );
        }
        self.inner.restore(ticket, ledger)
    }

    fn discard(&mut self, ticket: &ParkTicket, ledger: &mut Ledger) {
        self.inner.discard(ticket, ledger);
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn counters(&self) -> ParkCounters {
        self.inner.counters()
    }
}

// -------------------------------------------------------------------------------------------
// The victim rule, on its own
// -------------------------------------------------------------------------------------------

fn candidate(id: u64, admitted: u64, active: u64, remaining: u64, parks: u32) -> ParkCandidate {
    ParkCandidate {
        request_id: RequestId(id),
        admitted_at_step: admitted,
        last_active_step: active,
        remaining_tokens: remaining,
        parks_so_far: parks,
        // Admission order is id order unless a test says otherwise: a later request is further
        // back in the running queue.
        queue_pos: id as u32,
    }
}

#[test]
fn the_victim_rule_is_a_function_of_the_numbers() {
    // K5..K17's order, which every assertion below was written against; prod's is the next test.
    let policy = ParkPolicy::on_pressure().victim(VictimOrder::MostWorkLeft);
    let c = [
        candidate(1, 0, 10, 100, 0),
        candidate(2, 0, 10, 400, 0),
        candidate(3, 0, 10, 50, 0),
    ];
    // All equally idle, so the most remaining work wins: that is the sequence which would
    // otherwise hold its KV blocks and its state slot set for longest.
    assert_eq!(choose_victim(&c, &policy, 20).map(|r| r.0), Some(2));

    // Idleness outranks remaining work.
    let c2 = [candidate(1, 0, 3, 10, 0), candidate(2, 0, 10, 400, 0)];
    assert_eq!(choose_victim(&c2, &policy, 20).map(|r| r.0), Some(1));

    // The floor: never park below `min_running`.
    assert_eq!(choose_victim(&c2[..1], &policy, 20), None);
    assert_eq!(
        choose_victim(&c2, &policy.min_running(2), 20),
        None,
        "two running against a floor of two leaves nothing to evict"
    );

    // The residency guard: a sequence admitted this step has just paid for its whole prefill.
    let fresh = [candidate(1, 20, 20, 400, 0), candidate(2, 0, 20, 10, 0)];
    assert_eq!(
        choose_victim(&fresh, &policy, 20).map(|r| r.0),
        Some(2),
        "the freshly admitted sequence is spared even though it has the most work left"
    );

    // The per-session cap. ONE by default since ws-k18 (the card's re-park starvation); unbounded,
    // a session parked before is a candidate again, on the same numbers as everybody else...
    assert_eq!(ParkPolicy::default().max_parks_per_session, 1);
    let policy = policy.max_parks_per_session(u32::MAX);
    let once = [candidate(1, 0, 10, 400, 1), candidate(2, 0, 10, 10, 0)];
    assert_eq!(
        choose_victim(&once, &policy, 20).map(|r| r.0),
        Some(1),
        "a session parked once is still the one with the most work left"
    );
    // ...and an operator's finite cap binds: one park per session, then it runs to completion.
    let capped = policy.max_parks_per_session(1);
    assert_eq!(choose_victim(&once, &capped, 20).map(|r| r.0), Some(2));
    assert_eq!(
        choose_victim(&once[..1], &capped.min_running(0), 20),
        None,
        "under a cap of one, a session that has been parked once is not a candidate again"
    );

    // The idle threshold, which today can only ever exclude. Asserted so the documentation on
    // `ParkPolicy::idle_steps` stays true.
    assert_eq!(choose_victim(&c2, &policy.idle_steps(50), 20), None);
}

// -------------------------------------------------------------------------------------------
// The real store, through the scheduler
// -------------------------------------------------------------------------------------------

#[tokio::test]
async fn the_real_tier_round_trips_a_session_byte_identically() {
    let control = control_tokens(60).await;

    let sched = scheduler(RandomAcceptanceExecutor::new(0xB0A7, Acceptance::Fixed(2)))
        .with_park_tier(Box::new(ram_store(2)));
    let (kv_free, gdn_free) = sched.arena_free();
    let sub = sched.submit(request(&prompt(), 60)).expect("submit");
    sched.run_steps(4);

    sched.park(sub.id).expect("park");
    assert_eq!(
        sched.arena_free(),
        (kv_free, gdn_free),
        "KV blocks and the state slot set leave together"
    );

    // Billed to pinned host memory while it is away, and counted as write traffic.
    let snap = sched.ledger_snapshot();
    let pinned = snap
        .pools
        .iter()
        .find(|p| p.pool == Pool::HostPinned)
        .expect("pinned pool");
    assert!(pinned.used_bytes > 0, "{pinned:?}");
    assert_eq!(pinned.written_bytes, pinned.used_bytes);
    assert_eq!(sched.park_counters().parked_to_ram, 1);
    assert_eq!(sched.park_counters().demoted_to_nvme, 0);

    sched.unpark(sub.id).expect("unpark");
    sched.run_steps(500);

    let events = drain(sub).await;
    assert_eq!(
        tokens_of(&events),
        control,
        "a session that went through the real tier continues the stream it would have produced"
    );
    assert_eq!(events.last().unwrap().finished, Some(FinishReason::Length));
    assert_eq!(sched.park_counters().restored, 1);
    assert_eq!(sched.park_counters().refused_corrupt, 0);
    assert_eq!(sched.counters().restore_failures, 0);

    // Everything the tier held is released again.
    let snap = sched.ledger_snapshot();
    let pinned = snap
        .pools
        .iter()
        .find(|p| p.pool == Pool::HostPinned)
        .expect("pinned pool");
    assert_eq!(pinned.used_bytes, 0);
    assert!(pinned.reservations.is_empty());
}

#[tokio::test]
async fn a_corrupt_image_in_the_real_tier_is_refused_loudly() {
    // `CLAUDE.md` rule 8: the negative control for the whole parking path. One flipped bit, and
    // the sha256 the tier took on the way in is the only thing that knows.
    let (tier, corrupt) = Corruptible::ram(2);
    let sched = scheduler(RandomAcceptanceExecutor::new(0xBAD, Acceptance::Fixed(1)))
        .with_park_tier(Box::new(tier));
    let sub = sched.submit(request(&prompt(), 40)).expect("submit");
    sched.run_steps(3);
    sched.park(sub.id).expect("park");

    corrupt.store(true, Ordering::SeqCst);
    sched.unpark(sub.id).expect("queued for restore");
    sched.run_steps(2);

    assert_eq!(sched.counters().restore_failures, 1);
    assert_eq!(sched.counters().restores, 0);
    assert_eq!(sched.park_counters().refused_corrupt, 1);
    assert_eq!(sched.park_counters().restored, 0);
    assert_eq!(
        sched.parked_ids(),
        vec![sub.id],
        "a refused restore leaves the session parked, never half-restored"
    );
    let fresh = scheduler(RandomAcceptanceExecutor::new(0, Acceptance::Fixed(1)));
    assert_eq!(
        sched.arena_free(),
        fresh.arena_free(),
        "the allocations taken for the attempt went straight back"
    );

    sched.abort(sub.id);
    let snap = sched.ledger_snapshot();
    let pinned = snap
        .pools
        .iter()
        .find(|p| p.pool == Pool::HostPinned)
        .expect("pinned pool");
    assert_eq!(pinned.used_bytes, 0, "an aborted image is released");
    let events = drain(sub).await;
    assert_eq!(events.last().unwrap().finished, Some(FinishReason::Abort));
}

#[tokio::test]
async fn a_full_ram_tier_spills_the_oldest_image_to_nvme() {
    let scratch = Scratch::new("spill");
    let cfg = TieredParkConfig::ram(IMAGE_BYTES, IMAGE_BYTES)
        .with_nvme(scratch.file("park.bin"), 4 * IMAGE_BYTES)
        .hashing_on(2);
    assert_eq!((cfg.ram_slots(), cfg.nvme_slots()), (1, 4));
    let store = TieredParkStore::host(&cfg)
        .expect("store")
        .deleting_nvme_on_drop();

    let sched = scheduler(RandomAcceptanceExecutor::new(7, Acceptance::Fixed(2)))
        .with_park_tier(Box::new(store));
    let a = sched.submit(request(&prompt(), 40)).expect("a");
    sched.run_steps(1);
    let b = sched.submit(request(&[7, 7, 7], 40)).expect("b");
    sched.run_steps(2);

    sched.park(a.id).expect("park a");
    sched.park(b.id).expect("park b");

    let c = sched.park_counters();
    assert_eq!(c.parked_to_ram, 2, "both parks went for the RAM slot first");
    assert_eq!(
        c.demoted_to_nvme, 1,
        "the second park spilled the first image to make room"
    );
    let snap = sched.ledger_snapshot();
    let nvme = snap
        .pools
        .iter()
        .find(|p| p.pool == Pool::Nvme)
        .expect("nvme pool");
    assert!(nvme.used_bytes > 0 && nvme.written_bytes > 0, "{nvme:?}");

    // Both come back, one off the drive and one out of RAM.
    sched.unpark(a.id).expect("unpark a");
    sched.unpark(b.id).expect("unpark b");
    sched.run_steps(500);
    assert_eq!(sched.park_counters().restored, 2);
    assert_eq!(sched.counters().restore_failures, 0);

    let (ea, eb) = (drain(a).await, drain(b).await);
    assert_eq!(tokens_of(&ea).len(), 40);
    assert_eq!(tokens_of(&eb).len(), 40);

    // Every byte, on both pools, handed back.
    let snap = sched.ledger_snapshot();
    for pool in [Pool::HostPinned, Pool::Nvme] {
        let p = snap
            .pools
            .iter()
            .find(|p| p.pool == pool)
            .unwrap_or_else(|| panic!("{pool} pool"));
        assert_eq!(p.used_bytes, 0, "{pool} still holds bytes: {p:?}");
        assert!(p.reservations.is_empty(), "{pool}: {:?}", p.reservations);
    }
}

// -------------------------------------------------------------------------------------------
// The eviction policy, through the scheduler
// -------------------------------------------------------------------------------------------

/// A scheduler with `sets` state slot sets, a batch ladder that reaches four, and `policy`.
fn pressured(sets: u64, policy: ParkPolicy) -> Scheduler<RandomAcceptanceExecutor> {
    let buckets = [(1u32, 1u32, 8192u32), (2, 1, 8192), (4, 1, 8192)];
    let config = SchedulerConfig::from_plan(
        &common::plan_with(&buckets, 2),
        KvGeometry::PROD_FP8,
        ArenaBytes {
            kv_per_rank: 2 * GIB,
            gdn_state_per_rank: sets * 144 * MIB,
        },
        262_144,
    )
    .expect("config")
    .with_park_policy(policy);
    Scheduler::new(
        config,
        RandomAcceptanceExecutor::new(0x5EED, Acceptance::Fixed(1)),
        common::model_info(),
        common::boot_ledger(),
    )
    .expect("scheduler")
}

#[tokio::test]
async fn admission_under_pressure_parks_instead_of_refusing() {
    let sched = pressured(
        2,
        ParkPolicy::on_pressure().victim(VictimOrder::MostWorkLeft),
    )
    .with_park_tier(Box::new(ram_store(2)));
    assert_eq!(sched.arena_free().1, 2, "exactly two state slot sets");

    // A has a lot of work left and B very little; both are admitted on the first step.
    let a = sched.submit(request(&[1, 2], 60)).expect("a");
    let b = sched.submit(request(&[3, 4], 6)).expect("b");
    sched.run_steps(2);
    assert_eq!(sched.stats().num_requests_running, 2);

    // C arrives and there is no state slot set left. Before the policy this was a refusal that
    // simply waited.
    let c = sched.submit(request(&[5, 6], 6)).expect("c");
    sched.step();

    assert_eq!(sched.counters().policy_parks, 1, "the policy evicted one");
    assert_eq!(sched.counters().parks, 1);
    assert_eq!(sched.park_counters().parked_to_ram, 1);
    assert_eq!(
        sched.parked_ids(),
        vec![a.id],
        "the victim is the sequence with the most work left: it holds the arena longest"
    );
    assert_eq!(sched.stats().num_requests_running, 2, "B and C");
    assert!(sched.admission_counters().refused_gdn >= 1);

    // ...and A comes back on its own once something finishes. Nothing sends an explicit unpark.
    sched.run_steps(500);
    assert!(sched.counters().policy_restores_queued >= 1);
    assert_eq!(sched.park_counters().restored, 1);
    assert_eq!(sched.counters().restore_failures, 0);
    assert!(sched.parked_ids().is_empty());

    let (ea, eb, ec) = (drain(a).await, drain(b).await, drain(c).await);
    assert_eq!(
        tokens_of(&ea).len(),
        60,
        "the evicted request still finishes"
    );
    assert_eq!(tokens_of(&eb).len(), 6);
    assert_eq!(tokens_of(&ec).len(), 6);
    assert_eq!(ea.last().unwrap().finished, Some(FinishReason::Length));

    // One park, no ping-pong, and the ledger is flat again.
    assert_eq!(sched.counters().parks, 1);
    let snap = sched.ledger_snapshot();
    let pinned = snap
        .pools
        .iter()
        .find(|p| p.pool == Pool::HostPinned)
        .expect("pinned pool");
    assert_eq!(pinned.used_bytes, 0);
}

/// Drives the pressured scheduler until `a` has been through `rounds` park/restore round trips
/// (or the rounds run out), each forced the same way: two short arrivals while `a` holds one of
/// the two state slot sets. Returns every submission so the caller can drain them, and how many
/// of the rounds evicted A itself (a round can evict one of the short arrivals instead).
fn park_a_repeatedly(
    sched: &Scheduler<RandomAcceptanceExecutor>,
    a: RequestId,
    rounds: u32,
) -> (Vec<saber_serve::engine::Submission>, u32) {
    let mut others = Vec::new();
    let mut a_parked = 0;
    for round in 0..rounds {
        // Wait for A to be resident again (the first round it already is), then give it the one
        // step of residency the guard asks for.
        for _ in 0..500 {
            if !sched.parked_ids().contains(&a) {
                break;
            }
            sched.step();
        }
        sched.step();
        let base = 100 + u64::from(round) * 2;
        others.push(
            sched
                .submit(request(&[base as TokenId, 1], 6))
                .expect("short"),
        );
        others.push(
            sched
                .submit(request(&[base as TokenId + 1, 2], 6))
                .expect("short"),
        );
        sched.step();
        a_parked += u32::from(sched.parked_ids().contains(&a));
    }
    (others, a_parked)
}

#[tokio::test]
async fn one_session_is_parked_twice_and_still_answers_exactly() {
    // THE CONTROL: A alone, start to finish, on the same executor seed and the same geometry.
    let control = {
        let sched = pressured(2, ParkPolicy::on_pressure());
        let a = sched.submit(request(&[1, 2], 60)).expect("a");
        sched.run_steps(500);
        tokens_of(&drain(a).await)
    };
    assert_eq!(control.len(), 60);

    // THE CASE: no per-session cap (ws-k17's default; ws-k18 made the default one, measured).
    let sched = pressured(2, ParkPolicy::on_pressure().max_parks_per_session(u32::MAX))
        .with_park_tier(Box::new(ram_store(2)));
    let a = sched.submit(request(&[1, 2], 60)).expect("a");
    sched.run_steps(2);
    let (others, a_parked) = park_a_repeatedly(&sched, a.id, 2);
    sched.run_steps(1000);

    assert_eq!(a_parked, 2, "both rounds evicted A itself");
    let c = sched.counters();
    assert_eq!(c.policy_parks, 2, "A was evicted twice: {c:?}");
    assert_eq!(c.parks, 2);
    assert_eq!(
        c.reparks, 1,
        "the second park is a RE-park, and the census counts it"
    );
    assert_eq!(sched.park_counters().restored, 2, "and restored twice");
    assert_eq!(c.restore_failures, 0);
    assert!(sched.parked_ids().is_empty());
    let got = tokens_of(&drain(a).await);
    assert_eq!(
        got, control,
        "a session parked TWICE answers exactly what it answers unparked"
    );
    for sub in others {
        assert_eq!(tokens_of(&drain(sub).await).len(), 6);
    }

    // THE CAP'S CONTROL: the same pressure under a cap of one parks A once; the second round's
    // eviction has to take somebody else, which is the pre-ws-k17 behaviour.
    let sched = pressured(2, ParkPolicy::on_pressure().max_parks_per_session(1))
        .with_park_tier(Box::new(ram_store(2)));
    let a = sched.submit(request(&[1, 2], 60)).expect("a");
    sched.run_steps(2);
    let (others, a_parked) = park_a_repeatedly(&sched, a.id, 2);
    sched.run_steps(1000);
    assert_eq!(
        a_parked, 1,
        "the cap of one binds: A is evicted once and only once"
    );
    assert_eq!(sched.counters().reparks, 0, "and nobody is parked twice");
    assert_eq!(tokens_of(&drain(a).await), control);
    for sub in others {
        assert_eq!(tokens_of(&drain(sub).await).len(), 6);
    }
}

/// THE SERVE PLANE'S COMPOSITION carries the policy (ws-k17). `attach_park_tier` is what
/// `saber serve` builds its scheduler with, and it attached the tier and dropped the policy: the
/// first under-pressure run on the card refused 1,320 admissions for KV and parked nothing. The
/// case is the options with `on_pressure`; the control is the same options with the default
/// policy, which must park nothing — and the pre-fix composition was indistinguishable from it.
#[tokio::test]
async fn the_serve_composition_hands_the_policy_to_the_scheduler() {
    let run = |policy: ParkPolicy| {
        let opts = saber_sched::ServeParkOptions::from_gib(
            (2 * IMAGE_BYTES) as f64 / saber_sched::ServeParkOptions::GIB as f64,
            0.0,
            None,
        )
        .with_image_bytes(IMAGE_BYTES)
        .with_policy(policy);
        // The scheduler is built with the DEFAULT policy, as the serve plane's config is.
        let sched = saber_sched::attach_park_tier(pressured(2, ParkPolicy::default()), &opts)
            .expect("attach");
        assert_eq!(
            sched.config().park,
            policy,
            "the scheduler runs the options' policy"
        );
        let a = sched.submit(request(&[1, 2], 60)).expect("a");
        let b = sched.submit(request(&[3, 4], 6)).expect("b");
        sched.run_steps(2);
        let c = sched.submit(request(&[5, 6], 6)).expect("c");
        sched.run_steps(500);
        (sched.counters().policy_parks, [a, b, c])
    };
    let (parks, subs) = run(ParkPolicy::on_pressure());
    assert_eq!(parks, 1, "on-pressure through the serve composition parks");
    for sub in subs {
        assert!(drain(sub).await.last().unwrap().finished.is_some());
    }
    let (parks, subs) = run(ParkPolicy::default());
    assert_eq!(parks, 0, "the control: the default policy parks nothing");
    for sub in subs {
        assert!(drain(sub).await.last().unwrap().finished.is_some());
    }
}

#[tokio::test]
async fn the_policy_is_off_until_it_is_turned_on() {
    // The same shape with the default policy: the admission is refused and C waits, exactly as
    // before eviction existed. A tier is an operator hook until a policy says otherwise.
    let sched = pressured(2, ParkPolicy::default()).with_park_tier(Box::new(ram_store(2)));
    let a = sched.submit(request(&[1, 2], 60)).expect("a");
    let b = sched.submit(request(&[3, 4], 6)).expect("b");
    sched.run_steps(2);
    let c = sched.submit(request(&[5, 6], 6)).expect("c");
    sched.step();

    assert_eq!(sched.counters().policy_parks, 0);
    assert_eq!(sched.counters().policy_park_misses, 0);
    assert_eq!(sched.counters().parks, 0);
    assert_eq!(sched.stats().num_requests_waiting, 1, "C waits");

    sched.run_steps(500);
    for sub in [a, b, c] {
        assert!(drain(sub).await.last().unwrap().finished.is_some());
    }
}

#[tokio::test]
async fn pressure_with_nothing_eligible_is_counted_never_silent() {
    // One state slot set, so there is only ever one running sequence — and `min_running` refuses
    // to park it, because swapping the only sequence for the next one is a swap loop, not
    // progress. The policy must say out loud that it wanted to act and could not.
    let sched = pressured(1, ParkPolicy::on_pressure()).with_park_tier(Box::new(ram_store(1)));
    assert_eq!(sched.arena_free().1, 1);

    let a = sched.submit(request(&[1, 2], 20)).expect("a");
    sched.run_steps(2);
    let b = sched.submit(request(&[3, 4], 4)).expect("b");
    sched.step();

    assert_eq!(sched.counters().parks, 0);
    assert!(
        sched.counters().policy_park_misses >= 1,
        "a policy that wanted to evict and could not says so: {:?}",
        sched.counters()
    );
    sched.run_steps(500);
    assert_eq!(tokens_of(&drain(a).await).len(), 20);
    assert_eq!(tokens_of(&drain(b).await).len(), 4);
}

#[tokio::test]
async fn a_spilled_image_is_promoted_back_on_an_idle_step() {
    // The fifth path: an image that had to go to the drive is pulled back into RAM once there is
    // a slot for it, so its restore is a 5 ms copy rather than a 15 ms drive read. It happens on
    // an idle step and nowhere else — the promotion costs a read plus a copy, and the only moment
    // that is free is a step with nothing running.
    let scratch = Scratch::new("promote");
    let cfg = TieredParkConfig::ram(IMAGE_BYTES, IMAGE_BYTES)
        .with_nvme(scratch.file("park.bin"), 4 * IMAGE_BYTES)
        .hashing_on(2)
        .promoting_on_slack();
    let store = TieredParkStore::host(&cfg)
        .expect("store")
        .deleting_nvme_on_drop();
    let sched = scheduler(RandomAcceptanceExecutor::new(31, Acceptance::Fixed(2)))
        .with_park_tier(Box::new(store));

    let a = sched.submit(request(&prompt(), 40)).expect("a");
    sched.run_steps(1);
    let b = sched.submit(request(&[7, 7, 7], 12)).expect("b");
    sched.run_steps(2);
    sched.park(a.id).expect("park a");
    sched.park(b.id).expect("park b");
    assert_eq!(sched.park_counters().demoted_to_nvme, 1, "A spilled");
    assert_eq!(sched.park_counters().promoted, 0);

    // B comes back and finishes, which frees the RAM slot; the engine then goes idle, and that is
    // when A is pulled up off the drive.
    sched.unpark(b.id).expect("unpark b");
    sched.run_steps(500);
    assert_eq!(tokens_of(&drain(b).await).len(), 12);
    assert_eq!(
        sched.park_counters().promoted,
        1,
        "the spilled image was promoted on the idle step"
    );

    sched.unpark(a.id).expect("unpark a");
    sched.run_steps(500);
    assert_eq!(sched.park_counters().restored, 2);
    assert_eq!(sched.counters().restore_failures, 0);
    assert_eq!(tokens_of(&drain(a).await).len(), 40);

    let snap = sched.ledger_snapshot();
    for pool in [Pool::HostPinned, Pool::Nvme] {
        let p = snap
            .pools
            .iter()
            .find(|p| p.pool == pool)
            .unwrap_or_else(|| panic!("{pool} pool"));
        assert_eq!(p.used_bytes, 0, "{pool} still holds bytes: {p:?}");
    }
}

// -------------------------------------------------------------------------------------------
// Parking against the three-lock step (ws-l6)
// -------------------------------------------------------------------------------------------

#[tokio::test]
async fn a_park_is_refused_rather_than_waiting_on_the_device() {
    // Parking needs the executor, to copy the state image out. What it must never do is *wait*
    // for it: the executor lock is held for the device call, ~22 ms in production, and a session
    // layer blocking there would be blocking on the GPU. So a park attempted while a step is in
    // flight is refused with `StepInFlight` and the caller retries.
    //
    // `with_executor` holds exactly the lock a step in flight holds, which makes this
    // deterministic rather than a race against a background stepper. If `park` ever went back to
    // blocking on that lock this test would deadlock instead of failing — which is the honest
    // signal, because that is what the change would do to a server.
    let sched = scheduler(RandomAcceptanceExecutor::new(5, Acceptance::Fixed(1)))
        .with_park_tier(Box::new(ram_store(2)));
    let sub = sched.submit(request(&prompt(), 40)).expect("submit");
    sched.run_steps(2);
    let held = sched.arena_free();

    let err = sched
        .with_executor(|_| sched.park(sub.id))
        .expect_err("a step is in flight");
    assert!(matches!(err, ParkError::StepInFlight(_)), "{err}");
    assert_eq!(sched.counters().parks, 0);
    assert_eq!(sched.counters().park_refusals, 1);
    assert!(sched.parked_ids().is_empty(), "nothing was evicted");
    assert_eq!(sched.arena_free(), held, "a refused park changes nothing");
    assert_eq!(sched.park_counters().parked_to_ram, 0);

    // The contract is that the caller retries, and the retry is an ordinary park.
    sched.park(sub.id).expect("park once no step is in flight");
    assert_eq!(sched.parked_ids(), vec![sub.id]);
    sched.unpark(sub.id).expect("unpark");
    sched.run_steps(500);
    assert_eq!(tokens_of(&drain(sub).await).len(), 40);
    assert_eq!(sched.park_counters().restored, 1);
}

#[tokio::test]
async fn an_abort_brings_back_a_policy_parked_session_too() {
    // A running sequence that is aborted frees the same space as one that finishes, and under the
    // three-lock step it frees it through `retiring` rather than immediately. The session the
    // policy evicted must come back either way, or an aborted neighbour strands it forever.
    let sched = pressured(2, ParkPolicy::on_pressure()).with_park_tier(Box::new(ram_store(2)));
    let a = sched.submit(request(&[1, 2], 60)).expect("a");
    let b = sched.submit(request(&[3, 4], 60)).expect("b");
    sched.run_steps(2);
    let c = sched.submit(request(&[5, 6], 6)).expect("c");
    sched.step();

    assert_eq!(sched.counters().policy_parks, 1);
    let parked = sched.parked_ids();
    assert_eq!(parked.len(), 1);
    let victim = parked[0];
    let victim_is_a = victim == a.id;
    let survivor = if victim_is_a { b.id } else { a.id };

    sched.abort(survivor);
    assert!(
        sched.counters().policy_restores_queued >= 1,
        "the abort freed the space the eviction was for: {:?}",
        sched.counters()
    );

    sched.run_steps(500);
    assert!(
        sched.parked_ids().is_empty(),
        "the evicted session came back"
    );
    assert_eq!(sched.park_counters().restored, 1);
    assert_eq!(sched.counters().restore_failures, 0);

    let (ea, eb, ec) = (drain(a).await, drain(b).await, drain(c).await);
    let (evicted, killed) = if victim_is_a { (ea, eb) } else { (eb, ea) };
    assert_eq!(
        tokens_of(&evicted).len(),
        60,
        "the evicted request finished"
    );
    assert_eq!(killed.last().unwrap().finished, Some(FinishReason::Abort));
    assert_eq!(tokens_of(&ec).len(), 6);

    let snap = sched.ledger_snapshot();
    let pinned = snap
        .pools
        .iter()
        .find(|p| p.pool == Pool::HostPinned)
        .expect("pinned pool");
    assert_eq!(pinned.used_bytes, 0);
}

// -------------------------------------------------------------------------------------------
// ws-k18: prod's victim order, the discard hook, and the bounded restore
// -------------------------------------------------------------------------------------------

/// PROD'S RULE, as a function of the numbers: the back of the running queue. Read from the prod
/// image's `upstream/v1/core/sched/scheduler.py` (`self.running.pop()` under FCFS); see
/// `VictimOrder`'s docs for the source and for the two deviations Saber keeps.
#[test]
fn prods_victim_is_the_back_of_the_running_queue() {
    // Unbounded per session so the restored candidates below stay eligible (the default cap of one
    // would exclude them, which is the next assertion's subject, not this test's).
    let policy = ParkPolicy::on_pressure().max_parks_per_session(u32::MAX);
    assert_eq!(
        policy.victim,
        VictimOrder::Newest,
        "prod's order is the default"
    );
    // Same idleness; the one furthest back goes, whatever its work left.
    let c = [
        candidate(1, 0, 10, 400, 0),
        candidate(2, 0, 10, 10, 0),
        candidate(3, 0, 10, 50, 0),
    ];
    assert_eq!(choose_victim(&c, &policy, 20).map(|r| r.0), Some(3));
    // THE CONTROL: K5's order picks the most work left on the very same candidates.
    assert_eq!(
        choose_victim(&c, &policy.victim(VictimOrder::MostWorkLeft), 20).map(|r| r.0),
        Some(1)
    );
    // A RESTORED group is at the back, so prod would take it first; Saber's residency guard spares
    // it for the step of its restore, and after that it is the victim again exactly as prod's is.
    let mut r = [candidate(1, 0, 20, 400, 1), candidate(2, 20, 20, 10, 1)];
    r[1].queue_pos = 9;
    assert_eq!(
        choose_victim(&r, &policy, 20).map(|r| r.0),
        Some(1),
        "restored at step 20: spared at step 20"
    );
    assert_eq!(
        choose_victim(&r, &policy, 21).map(|r| r.0),
        Some(2),
        "and the back of the queue again at step 21"
    );
    assert_eq!(VictimOrder::parse("newest"), Ok(VictimOrder::Newest));
    assert_eq!(
        VictimOrder::parse("most-work"),
        Ok(VictimOrder::MostWorkLeft)
    );
    assert!(VictimOrder::parse("oldest").is_err());
}

#[tokio::test]
async fn under_prods_order_the_newest_admission_is_the_victim() {
    // The same scene as `admission_under_pressure_parks_instead_of_refusing`, under prod's order:
    // A (60 tokens) then B (6), both resident; C arrives with no state slot set left. prod pops B —
    // the back of the running queue — where K5's order took A for its work left.
    let sched = pressured(2, ParkPolicy::on_pressure()).with_park_tier(Box::new(ram_store(2)));
    let a = sched.submit(request(&[1, 2], 60)).expect("a");
    let b = sched.submit(request(&[3, 4], 6)).expect("b");
    sched.run_steps(2);
    let c = sched.submit(request(&[5, 6], 6)).expect("c");
    sched.step();
    assert_eq!(
        sched.parked_ids(),
        vec![b.id],
        "the newest admission is the victim"
    );
    sched.run_steps(500);
    assert!(sched.parked_ids().is_empty());
    assert_eq!(sched.counters().restore_failures, 0);
    assert_eq!(tokens_of(&drain(a).await).len(), 60);
    assert_eq!(tokens_of(&drain(b).await).len(), 6);
    assert_eq!(tokens_of(&drain(c).await).len(), 6);
}

/// THE DISCARD HOOK (ws-k18). An image dropped unrestored owes the executor a `discard_parked`
/// under the name it was parked as — at TP=2 that is what frees the follower's half — and an image
/// that WAS restored owes nothing (the restore consumed it).
#[tokio::test]
async fn an_image_dropped_unrestored_is_discarded_on_the_executor_too() {
    // THE CONTROL: park, restore, finish. Nothing is discarded.
    let sched = scheduler(RandomAcceptanceExecutor::new(3, Acceptance::Fixed(1)))
        .with_park_tier(Box::new(ram_store(2)));
    let sub = sched.submit(request(&prompt(), 20)).expect("submit");
    sched.run_steps(3);
    sched.park(sub.id).expect("park");
    sched.unpark(sub.id).expect("unpark");
    sched.run_steps(500);
    assert_eq!(tokens_of(&drain(sub).await).len(), 20);
    assert_eq!(sched.counters().parked_discards, 0);
    assert!(sched.with_executor(|e| e.discarded_parked().is_empty()));

    // THE CASE: park, abort while parked.
    let sched = scheduler(RandomAcceptanceExecutor::new(3, Acceptance::Fixed(1)))
        .with_park_tier(Box::new(ram_store(2)));
    let sub = sched.submit(request(&prompt(), 20)).expect("submit");
    sched.run_steps(3);
    sched.park(sub.id).expect("park");
    assert_eq!(
        sched.with_executor(|e| e.live_sequences()),
        0,
        "the parked sequence's pages and slot went back"
    );
    sched.abort(sub.id);
    assert_eq!(sched.counters().parked_discards, 1);
    let discarded = sched.with_executor(|e| e.discarded_parked().to_vec());
    assert_eq!(discarded.len(), 1, "one discard, for the one branch");
    assert!(sched.parked_ids().is_empty());
    assert_eq!(
        drain(sub).await.last().unwrap().finished,
        Some(FinishReason::Abort)
    );
}

/// Drive a policy park of A under pressure with a tier that (optionally) corrupts every restored
/// image, and return the scheduler, A's events and the tier's log.
async fn pressure_with_failing_restores(
    attempts: u32,
    corrupt: bool,
) -> (
    Scheduler<RandomAcceptanceExecutor>,
    Vec<saber_serve::engine::StepEvent>,
    Arc<Mutex<Spy>>,
) {
    let (tier, log) = SpyTier::ram();
    let sched = pressured(
        2,
        ParkPolicy::on_pressure()
            .victim(VictimOrder::MostWorkLeft)
            .max_restore_attempts(attempts),
    )
    .with_park_tier(Box::new(tier));
    let a = sched.submit(request(&[1, 2], 60)).expect("a");
    let b = sched.submit(request(&[3, 4], 6)).expect("b");
    sched.run_steps(2);
    let c = sched.submit(request(&[5, 6], 6)).expect("c");
    sched.step();
    assert_eq!(sched.parked_ids(), vec![a.id], "A is the victim");
    log.lock().expect("spy").corrupt = corrupt;
    sched.run_steps(500);
    assert_eq!(tokens_of(&drain(b).await).len(), 6);
    assert_eq!(tokens_of(&drain(c).await).len(), 6);
    let events = drain(a).await;
    (sched, events, log)
}

/// THE BOUNDED RESTORE (ws-k18). A restore that fails every time used to be retried for ever (and a
/// policy-parked session whose ASYNC restore failed was put back as an explicit park, so nothing
/// retried it and the client waited out its own timeout). Now it is retried up to
/// `max_restore_attempts` and then the request ENDS with a named error on its last event.
#[tokio::test]
async fn a_restore_that_always_fails_ends_the_request_with_a_named_error() {
    let (sched, events, log) = pressure_with_failing_restores(3, true).await;
    let c = sched.counters();
    assert_eq!(c.restore_failures, 3, "three attempts: {c:?}");
    assert_eq!(c.restores_given_up, 1);
    assert_eq!(c.restores, 0);
    assert_eq!(
        c.parked_discards, 1,
        "the given-up image is discarded on the executor too"
    );
    assert!(sched.parked_ids().is_empty(), "nothing is left parked");
    assert_eq!(
        log.lock().expect("spy").restored.len(),
        3,
        "three real attempts, not spent tickets"
    );
    let last = events.last().expect("an event");
    assert_eq!(last.finished, Some(FinishReason::Abort));
    let error = last.error.as_deref().expect("a named error");
    assert!(
        error.contains("could not be restored after 3 attempt(s)"),
        "{error}"
    );
    let host = sched
        .ledger_snapshot()
        .pools
        .into_iter()
        .find(|p| p.pool == Pool::HostPageable)
        .expect("host pool");
    assert_eq!(host.used_bytes, 0, "the tier holds nothing afterwards");

    // The bound is the number: one attempt gives up after one.
    let (sched, events, _log) = pressure_with_failing_restores(1, true).await;
    assert_eq!(sched.counters().restore_failures, 1);
    assert!(events.last().unwrap().error.is_some());

    // THE CONTROL: the same scene with a tier that does not corrupt restores A and finishes it.
    let (sched, events, _log) = pressure_with_failing_restores(3, false).await;
    assert_eq!(sched.counters().restores_given_up, 0);
    assert_eq!(sched.counters().restore_failures, 0);
    assert_eq!(tokens_of(&events).len(), 60);
    assert!(events.iter().all(|e| e.error.is_none()));
}
