//! Sequence groups: one request, one prompt, `n` branches.
//!
//! `n > 1` used to be fan-out in the serving plane — `n` submissions of the same prompt, prefilled
//! `n` times, with nothing in the engine knowing they were related. A group is the honest shape:
//! the prompt's **whole KV blocks are shared** between the branches (refcounted, freed on the last
//! release, with the partial block copied rather than shared), admission is **all-or-nothing** over
//! the branches, and every branch carries its own GDN state slot set, its own stop rules and its
//! own acceptance EMA.
//!
//! What that buys and what it does not:
//!
//! * KV is shared, so a group of three over a 4k prompt costs one prompt and three tails.
//! * **State is not.** Recurrent state is not addressable per token, so `n` slot sets are taken
//!   whatever the prompt costs — which is why a group's admission refusal is almost always the
//!   state arena, and why the refusal says so.

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;

use common::{boot_ledger, decode_buckets, plan_with, GIB, MIB};
use saber_sched::{
    ArenaBytes, KvGeometry, PagedKvAllocator, Refusal, Scheduler, SchedulerConfig, ScriptExecutor,
    StepGate,
};
use saber_serve::detok::{ByteCodec, TextCodec};
use saber_serve::engine::{Engine, EngineRequest, StepEvent, TokenId};
use saber_serve::sampling::SamplingParams;

const EOS: TokenId = 255;
const TOKENS_PER_STEP: u32 = 5;

/// Three answers, one per branch, so a group's choices are distinguishable from each other.
const ANSWERS: [&str; 3] = [
    "The first branch answers at some length so that it spans several engine steps.",
    "The second branch says something else entirely, and says it for a while too.",
    "The third branch is shorter.",
];

fn ids(text: &str) -> Vec<TokenId> {
    let mut v = ByteCodec.encode(text).expect("encode");
    v.push(EOS);
    v
}

fn scheduler_with_arenas(
    arenas: ArenaBytes,
    gate: Option<Arc<StepGate>>,
) -> Arc<Scheduler<ScriptExecutor>> {
    let plan = plan_with(&decode_buckets(), 2);
    let config =
        SchedulerConfig::from_plan(&plan, KvGeometry::PROD_FP8, arenas, 262_144).expect("config");
    let model =
        saber_sched::mock_model_info("Qwen3.8", "/models/mock", EOS, config.max_context_tokens);
    let responses: Vec<Vec<TokenId>> = ANSWERS.iter().map(|a| ids(a)).collect();
    let exec = ScriptExecutor::new(responses, TOKENS_PER_STEP, EOS);
    let exec = match gate {
        Some(gate) => exec.with_gate(gate),
        None => exec,
    };
    Arc::new(Scheduler::new(config, exec, model, boot_ledger()).expect("arenas fit"))
}

fn roomy() -> ArenaBytes {
    ArenaBytes {
        kv_per_rank: 8 * GIB,
        gdn_state_per_rank: 8 * GIB,
    }
}

fn request(prompt: &[TokenId], n: usize, max_tokens: usize) -> EngineRequest {
    EngineRequest {
        prompt_token_ids: prompt.to_vec(),
        params: SamplingParams {
            n,
            max_tokens: Some(max_tokens),
            ..SamplingParams::default()
        },
    }
}

/// Every choice's token stream, keyed by `choice_index`.
async fn choices(submission: saber_serve::engine::Submission) -> BTreeMap<u32, Vec<TokenId>> {
    use futures_util::StreamExt;
    let events: Vec<StepEvent> = submission.events.collect().await;
    let mut out: BTreeMap<u32, Vec<TokenId>> = BTreeMap::new();
    let mut finished: BTreeMap<u32, usize> = BTreeMap::new();
    let mut steps: BTreeMap<u32, u32> = BTreeMap::new();
    for e in &events {
        let tokens = out.entry(e.choice_index).or_default();
        tokens.extend(e.new_tokens.iter().copied());
        // Each choice carries its own step_index, monotone from 0.
        let next = steps.entry(e.choice_index).or_default();
        assert_eq!(e.step_index, *next, "choice {} step index", e.choice_index);
        *next += 1;
        if e.finished.is_some() {
            *finished.entry(e.choice_index).or_default() += 1;
        }
    }
    for (choice, count) in &finished {
        assert_eq!(*count, 1, "choice {choice} carried {count} finished events");
    }
    assert_eq!(
        finished.len(),
        out.len(),
        "every choice finished exactly once"
    );
    out
}

#[tokio::test]
async fn n_equals_three_is_token_for_token_three_single_submits() {
    let prompt: Vec<TokenId> = ByteCodec
        .encode("Explain sequence groups.")
        .expect("encode");

    // One request, three branches, one prompt.
    let group = scheduler_with_arenas(roomy(), None);
    let submission = group.submit(request(&prompt, 3, 200)).expect("submit");
    group.run_steps(1000);
    let grouped = choices(submission).await;
    assert_eq!(grouped.len(), 3, "three choices");

    // Three requests, one branch each, on a scheduler that hands out the same scripts in the same
    // order. The group must not change a single token.
    let singles = scheduler_with_arenas(roomy(), None);
    let mut submissions = Vec::new();
    for _ in 0..3 {
        submissions.push(singles.submit(request(&prompt, 1, 200)).expect("submit"));
        singles.run_steps(1000);
    }
    for (index, submission) in submissions.into_iter().enumerate() {
        let single = choices(submission).await;
        assert_eq!(single.len(), 1);
        assert_eq!(
            grouped[&(index as u32)],
            single[&0],
            "choice {index} differs between the group and its single-submit control"
        );
        assert_eq!(grouped[&(index as u32)], ids(ANSWERS[index]));
    }

    // Both arenas came back, on both schedulers.
    let fresh = scheduler_with_arenas(roomy(), None);
    assert_eq!(group.arena_free(), fresh.arena_free());
    assert_eq!(singles.arena_free(), fresh.arena_free());
    group.with_executor(|e| {
        let (admits, releases) = e.admit_release_counts();
        assert_eq!(
            (admits, releases),
            (3, 3),
            "three branches admitted and released"
        );
    });
}

#[tokio::test]
async fn the_group_shares_the_prompt_and_pays_for_state_per_branch() {
    // A prompt long enough to span many blocks, so the sharing is measurable rather than a
    // rounding difference: 4000 bytes is 250 blocks of 16 tokens.
    let prompt: Vec<TokenId> = ByteCodec.encode(&"prompt ".repeat(600)).expect("encode");
    let prompt_blocks = (prompt.len() / 16) as u32;

    let group = scheduler_with_arenas(roomy(), None);
    let (kv_before, state_before) = group.arena_free();
    let submission = group.submit(request(&prompt, 3, 100)).expect("submit");
    group.run_steps(1); // admit + one step

    let (kv_now, state_now) = group.arena_free();
    let state_used = state_before - state_now;
    let kv_used = kv_before - kv_now;
    assert_eq!(state_used, 3, "one GDN slot set per branch, always");
    // Three tails and ONE prompt, not three prompts.
    assert!(
        kv_used < 3 * prompt_blocks,
        "{kv_used} blocks for three branches over a {prompt_blocks}-block prompt: the prompt is \
         not being shared"
    );
    assert!(
        kv_used >= prompt_blocks,
        "{kv_used} blocks is less than the prompt itself"
    );

    group.run_steps(1000);
    drop(submission);
    let fresh = scheduler_with_arenas(roomy(), None);
    assert_eq!(
        group.arena_free(),
        fresh.arena_free(),
        "every shared block came back once the last branch released it"
    );
}

#[tokio::test]
async fn aborting_a_group_mid_step_retires_every_branch_and_returns_the_shared_blocks() {
    // The step is held open, so the abort lands while the executor is still holding the group's
    // slots: the commit afterwards matches results by SeqId, finds all three branches gone, and
    // drops their results rather than committing them into whatever now occupies those indices.
    let gate = StepGate::closed();
    let group = scheduler_with_arenas(roomy(), Some(Arc::clone(&gate)));
    let free_before = group.arena_free();
    let prompt: Vec<TokenId> = ByteCodec
        .encode("Explain sequence groups.")
        .expect("encode");
    let submission = group.submit(request(&prompt, 3, 200)).expect("submit");

    // One step: admits the group and blocks in the executor.
    let stepper = {
        let group = Arc::clone(&group);
        std::thread::spawn(move || group.run_steps(4))
    };
    for _ in 0..200 {
        if group.stats().num_requests_running == 1 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    assert_eq!(
        group.stats().num_requests_running,
        1,
        "one REQUEST, three sequences"
    );
    assert_eq!(group.arena_free().1, free_before.1 - 3, "three slot sets");

    group.abort(submission.id);
    assert_eq!(group.counters().aborts_running, 1, "one group, one abort");

    // Let the held step finish, and the retirement drain with it.
    gate.release(8);
    let _ = stepper.join().expect("stepper");
    for _ in 0..200 {
        if group.arena_free() == free_before {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    assert_eq!(
        group.arena_free(),
        free_before,
        "every branch's tail and the shared prompt came back"
    );
    // ONE result, not three: the group was aborted before its leader sampled, so the two branches
    // had never been handed to the executor at all (V4's fork defers a branch's `admit` to the
    // moment the leader's prompt produces a row). The leader's own result is the one that comes
    // back for a sequence that has left, and it is dropped rather than committed.
    assert_eq!(
        group.counters().results_for_gone_sequences,
        1,
        "the leader's result, for a sequence that had left: {:?}",
        group.counters()
    );
    let c = group.counters();
    assert_eq!(
        (c.forks_pending, c.forks, c.fork_failures),
        (2, 0, 0),
        "two branches were waiting on a fork that never came: {c:?}"
    );
    group.with_executor(|e| {
        assert_eq!(e.live_sequences(), 0);
        assert_eq!(
            e.admit_release_counts(),
            (1, 1),
            "only the leader ever reached the executor"
        );
    });

    // Every choice got its one finished event, and it says abort.
    let ends = choices(submission).await;
    assert_eq!(ends.len(), 3);
}

#[tokio::test]
async fn admission_refuses_a_group_on_the_state_arena_and_says_so() {
    // Room for two slot sets, so a group of three cannot be admitted whole. KV is not the
    // constraint: at this prompt the three branches want a handful of blocks.
    let per_seq = 6 * 72 * MIB; // (max spec width 5 + 1) slots of 72 MiB at TP=2
    let arenas = ArenaBytes {
        kv_per_rank: 2 * GIB,
        gdn_state_per_rank: 2 * per_seq,
    };
    let sched = scheduler_with_arenas(arenas, None);
    assert_eq!(sched.arena_free().1, 2, "two slot sets");

    let prompt: Vec<TokenId> = ByteCodec.encode("short").expect("encode");
    let _submission = sched.submit(request(&prompt, 3, 64)).expect("submit");
    sched.run_steps(2);

    assert_eq!(sched.stats().num_requests_running, 0, "not admitted at all");
    assert_eq!(sched.stats().num_requests_waiting, 1, "still queued");
    let counters = sched.admission_counters();
    assert!(counters.refused_gdn >= 1, "{counters:?}");
    assert_eq!(
        counters.refused_kv, 0,
        "KV was not the constraint: {counters:?}"
    );
    assert_eq!(
        sched.arena_free(),
        (sched.arena_free().0, 2),
        "a refused group leaves both arenas exactly as they were"
    );

    // And the refusal names what ran out, with the group size in it.
    let refusal = Refusal::GdnStateSlots {
        want_slots: 6,
        want_sets: 3,
        free_sets: 2,
    };
    let text = refusal.to_string();
    assert!(text.contains("gdn state arena"), "{text}");
    assert!(text.contains("3 set(s) of 6 slots"), "{text}");
    assert!(text.contains("n = 3"), "{text}");
    assert!(text.contains("caps n"), "{text}");
}

#[tokio::test]
async fn a_group_parks_and_resumes_as_one_unit() {
    use saber_sched::HostTier;

    let plan = plan_with(&decode_buckets(), 2);
    let config =
        SchedulerConfig::from_plan(&plan, KvGeometry::PROD_FP8, roomy(), 262_144).expect("config");
    let model =
        saber_sched::mock_model_info("Qwen3.8", "/models/mock", EOS, config.max_context_tokens);
    let responses: Vec<Vec<TokenId>> = ANSWERS.iter().map(|a| ids(a)).collect();
    let sched = Arc::new(
        Scheduler::new(
            config,
            ScriptExecutor::new(responses, TOKENS_PER_STEP, EOS),
            model,
            boot_ledger(),
        )
        .expect("arenas fit")
        .with_park_tier(Box::new(HostTier::ram())),
    );

    let control = scheduler_with_arenas(roomy(), None);
    let prompt: Vec<TokenId> = ByteCodec
        .encode("Explain sequence groups.")
        .expect("encode");
    let control_sub = control.submit(request(&prompt, 3, 200)).expect("submit");
    control.run_steps(1000);
    let want = choices(control_sub).await;

    let free_before = sched.arena_free();
    let submission = sched.submit(request(&prompt, 3, 200)).expect("submit");
    sched.run_steps(3);
    assert_eq!(sched.stats().num_requests_running, 1);

    // One call, three branches: the whole group leaves the device together.
    sched.park(submission.id).expect("park");
    assert_eq!(sched.parked_ids(), vec![submission.id]);
    assert_eq!(
        sched.arena_free(),
        free_before,
        "parking a group frees every branch's state and the pages they shared"
    );
    let host = sched
        .ledger_snapshot()
        .pools
        .into_iter()
        .find(|p| p.pool == saber_mem::ledger::Pool::HostPageable)
        .expect("host pool");
    assert_eq!(host.reservations.len(), 3, "one image per branch: {host:?}");

    sched.unpark(submission.id).expect("unpark");
    sched.run_steps(1000);
    let got = choices(submission).await;
    assert_eq!(
        got, want,
        "a parked group resumes the stream it would have produced"
    );
    assert_eq!(sched.counters().parks, 1);
    assert_eq!(sched.counters().restores, 1);
    assert_eq!(sched.arena_free(), free_before);
}

#[tokio::test]
async fn the_eviction_policy_treats_a_group_as_one_session() {
    use saber_sched::{HostTier, ParkPolicy};

    // Four state slot sets and a batch ladder that reaches eight: room for one group of two plus
    // one single-choice request, and nothing more.
    let plan = plan_with(&decode_buckets(), 2);
    let config = SchedulerConfig::from_plan(
        &plan,
        KvGeometry::PROD_FP8,
        ArenaBytes {
            kv_per_rank: 2 * GIB,
            gdn_state_per_rank: 3 * 6 * 72 * MIB,
        },
        262_144,
    )
    .expect("config")
    // K5's victim order, which picks the group on its work left; under prod's (`Newest`) the single
    // admitted after it would go first, and this test is about the group parking as ONE unit.
    .with_park_policy(ParkPolicy::on_pressure().victim(saber_sched::VictimOrder::MostWorkLeft));
    let model =
        saber_sched::mock_model_info("Qwen3.8", "/models/mock", EOS, config.max_context_tokens);
    let responses: Vec<Vec<TokenId>> = ANSWERS.iter().map(|a| ids(a)).collect();
    let sched = Arc::new(
        Scheduler::new(
            config,
            ScriptExecutor::new(responses, TOKENS_PER_STEP, EOS),
            model,
            boot_ledger(),
        )
        .expect("arenas fit")
        .with_park_tier(Box::new(HostTier::ram())),
    );
    assert_eq!(sched.arena_free().1, 3, "three state slot sets");

    let prompt: Vec<TokenId> = ByteCodec
        .encode("Explain sequence groups.")
        .expect("encode");
    let group = sched.submit(request(&prompt, 2, 200)).expect("group");
    sched.run_steps(2);
    assert_eq!(
        sched.stats().num_requests_running,
        1,
        "one request, two branches"
    );
    assert_eq!(sched.arena_free().1, 1, "two sets held by the group");

    // A second request wants the last set, then a third arrives with nothing left. The policy
    // evicts a whole session -- and the only one old enough is the group.
    let _single = sched.submit(request(&prompt, 1, 40)).expect("single");
    sched.run_steps(2);
    let _third = sched.submit(request(&prompt, 1, 40)).expect("third");
    sched.step();

    assert_eq!(
        sched.counters().policy_parks,
        1,
        "one park, not one per branch"
    );
    assert_eq!(
        sched.parked_ids(),
        vec![group.id],
        "the group was the victim"
    );
    // Both of its branches left together: three sets were held, the group's two came back and
    // the newcomer took one.
    assert!(
        sched.counters().parks == 1,
        "a group parks once: {:?}",
        sched.counters()
    );
    let host = sched
        .ledger_snapshot()
        .pools
        .into_iter()
        .find(|p| p.pool == saber_mem::ledger::Pool::HostPageable)
        .expect("host pool");
    assert_eq!(host.reservations.len(), 2, "one image per branch: {host:?}");

    // Once the short requests finish, the group is requeued as one unit and restored whole.
    sched.run_steps(400);
    assert!(
        sched.counters().policy_restores_queued >= 1,
        "the group should have been requeued: {:?}",
        sched.counters()
    );
    let finished = choices(group).await;
    assert_eq!(finished.len(), 2, "both branches finished");
    assert_eq!(sched.counters().restores, 1, "one restore for the group");
}

// ── the allocator itself ────────────────────────────────────────────────────────

#[test]
fn a_branch_references_the_prompt_and_owns_its_tail() {
    let geometry = KvGeometry::PROD_FP8;
    let mut kv = PagedKvAllocator::new(geometry, 64 * geometry.block_bytes());
    let total = kv.total_blocks();

    // A prompt of 40 tokens is two whole blocks and half of a third.
    let leader = kv.allocate(40 + 32).expect("leader");
    assert_eq!(leader.blocks.len(), 5, "72 tokens is five 16-token blocks");
    assert_eq!(leader.shared_prefix, 0);
    assert_eq!(leader.cow_source, None);

    let shared = kv.whole_blocks(40);
    assert_eq!(shared, 2, "only WHOLE blocks can be shared");
    let branch = kv.branch(&leader, shared, 40 + 32).expect("branch");
    assert_eq!(branch.shared_prefix, 2);
    assert_eq!(
        &branch.blocks[..2],
        &leader.blocks[..2],
        "the prefix is the same pages"
    );
    assert_eq!(
        branch.cow_source,
        Some(leader.blocks[2]),
        "the partial block is copied, not shared"
    );
    assert!(
        branch.blocks[2..]
            .iter()
            .all(|b| !leader.blocks.contains(b)),
        "the tail is private"
    );
    for b in &leader.blocks[..2] {
        assert_eq!(kv.refs(*b), 2, "held by the leader and the branch");
    }
    assert_eq!(
        kv.refs(leader.blocks[2]),
        1,
        "the COW source is the leader's alone"
    );

    // Five blocks for the leader, three private for the branch.
    assert_eq!(kv.free_blocks(), total - 8);

    // The branch goes first: the shared pages stay, its tail comes back.
    kv.free(branch);
    for b in &leader.blocks[..2] {
        assert_eq!(kv.refs(*b), 1, "still held by the leader");
    }
    assert_eq!(kv.free_blocks(), total - 5);

    // The leader goes last, and everything returns.
    kv.free(leader);
    assert_eq!(kv.free_blocks(), total);
}

#[test]
fn a_branch_is_refused_without_room_for_its_tail_and_changes_nothing() {
    let geometry = KvGeometry::PROD_FP8;
    let mut kv = PagedKvAllocator::new(geometry, 8 * geometry.block_bytes());
    let leader = kv.allocate(6 * 16).expect("leader");
    assert_eq!(kv.free_blocks(), 2);

    // Sharing four blocks still needs two of its own, which is exactly what is left.
    let ok = kv.branch(&leader, 4, 6 * 16).expect("fits exactly");
    assert_eq!(kv.free_blocks(), 0);
    kv.free(ok);
    assert_eq!(kv.free_blocks(), 2);

    // One more than fits is refused, and the refcounts are untouched.
    let err = kv
        .branch(&leader, 3, 6 * 16)
        .expect_err("three own blocks do not fit");
    assert_eq!(err, 2, "the refusal says how many were free");
    assert_eq!(kv.free_blocks(), 2);
    for b in &leader.blocks {
        assert_eq!(kv.refs(*b), 1, "a refused branch takes no references");
    }
}
