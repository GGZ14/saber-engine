//! The multi-turn gate (test tier 3b), in the shape the ledger demands.
//!
//! `docs/LEDGER.md`, 09-17: base+stash+replay GDN snapshots *"corrupt multi-turn chat at
//! temperature 0 (lazy 10/50 healthy vs eager 49/50, only variable) while single-shot, needle and
//! token-equality gates all read null."* The instrument that catches that class of bug is a
//! **conversation**, not a prompt: state that survives a sequence's release, or a slot that is
//! recycled without being re-initialised, only shows up on the turn that lands on the recycled
//! slot.
//!
//! So the gate is: five turns through **one** scheduler must be token-for-token identical to the
//! same five turns each run through a **fresh** scheduler. Prefix caching is off — a non-goal, not
//! a gap (`ARCHITECTURE.md` § Scope: it is disabled in prod deliberately, for correctness against
//! the speculative scheme being kept) — so every turn re-prefills the whole conversation and the
//! two runs are comparable by construction.
//!
//! And, per `CLAUDE.md` rule 8, the gate ships with a negative control that is known to fail:
//! [`RandomAcceptanceExecutor::leaky`] keeps a slot's state across admissions, which is exactly the
//! bug shape above, and the same comparison must catch it.

mod common;

use common::{drain, params, scheduler, tokens_of};
use saber_sched::{Acceptance, RandomAcceptanceExecutor, Scheduler};
use saber_serve::engine::{Engine, EngineRequest, TokenId};

const SEED: u64 = 0x5AB3_2026;
const TURNS: usize = 5;
const TOKENS_PER_TURN: usize = 12;

/// What the user "says" on turn `n`.
fn user_turn(n: usize) -> Vec<TokenId> {
    vec![90_000 + n as u32, 91_000 + n as u32, 92_000 + n as u32]
}

fn executor(leaky: bool) -> RandomAcceptanceExecutor {
    let e = RandomAcceptanceExecutor::new(SEED, Acceptance::Fixed(2));
    if leaky {
        e.leaky()
    } else {
        e
    }
}

/// Runs one turn on `sched` and returns the tokens it generated.
async fn run_turn(sched: &Scheduler<RandomAcceptanceExecutor>, prompt: &[TokenId]) -> Vec<TokenId> {
    let sub = sched
        .submit(EngineRequest {
            prompt_token_ids: prompt.to_vec(),
            params: params(TOKENS_PER_TURN),
        })
        .expect("submit");
    sched.run_steps(500);
    let tokens = tokens_of(&drain(sub).await);
    assert_eq!(tokens.len(), TOKENS_PER_TURN);
    tokens
}

/// Five turns through one scheduler. Returns each turn's prompt and its output.
async fn conversation(leaky: bool) -> Vec<(Vec<TokenId>, Vec<TokenId>)> {
    let sched = scheduler(executor(leaky));
    let mut context: Vec<TokenId> = Vec::new();
    let mut turns = Vec::new();
    for n in 0..TURNS {
        // KV reuse is OFF: every turn resubmits the whole conversation as a fresh prompt.
        context.extend(user_turn(n));
        let prompt = context.clone();
        let out = run_turn(&sched, &prompt).await;
        context.extend(out.iter().copied());
        turns.push((prompt, out));
    }
    assert_eq!(sched.stats().num_requests_running, 0);
    turns
}

#[tokio::test]
async fn five_turns_through_one_scheduler_match_five_single_turn_runs() {
    let turns = conversation(false).await;

    for (i, (prompt, multi)) in turns.iter().enumerate() {
        // The control: a scheduler that has never seen anything else.
        let fresh = scheduler(executor(false));
        let single = run_turn(&fresh, prompt).await;
        assert_eq!(
            *multi, single,
            "turn {i} differs between the multi-turn run and its single-turn control:\n\
             multi  = {multi:?}\n single = {single:?}"
        );
    }

    // The turns are genuinely different from each other -- otherwise the comparison above would
    // pass on a degenerate fixture.
    assert_ne!(turns[0].1, turns[1].1);
    assert_eq!(
        turns.iter().map(|t| t.0.len()).collect::<Vec<_>>(),
        vec![3, 18, 33, 48, 63],
        "each turn re-prefills prompt + history: 3 user tokens + 12 generated per turn"
    );
}

#[tokio::test]
async fn the_gate_catches_state_that_survives_a_released_slot() {
    // The negative control. A leaky executor does not re-initialise a recycled slot's state, which
    // is the 09-17 failure shape: single-shot behaviour is perfect, the conversation is wrong.
    let turns = conversation(true).await;

    // Turn 0 lands on a fresh slot in both runs, so it must still match -- that is what makes the
    // failure below a leak rather than a broken fixture.
    let fresh = scheduler(executor(true));
    let single0 = run_turn(&fresh, &turns[0].0).await;
    assert_eq!(
        turns[0].1, single0,
        "turn 0 cannot leak: nothing ran before it"
    );

    let mut differing = 0;
    for (prompt, multi) in turns.iter().skip(1) {
        let fresh = scheduler(executor(true));
        let single = run_turn(&fresh, prompt).await;
        if *multi != single {
            differing += 1;
        }
    }
    assert_eq!(
        differing,
        TURNS - 1,
        "the leaky executor must fail this gate on every turn after the first; \
         a gate that passes with the bug present is measuring nothing"
    );
}

#[tokio::test]
async fn interleaved_conversations_do_not_contaminate_each_other() {
    // Two conversations in flight at once, sharing the batch. Each must still match its own
    // single-turn control: a slot's neighbours in the batch are not part of its state either.
    let sched = scheduler(executor(false));
    let (p_a, p_b) = (user_turn(0), user_turn(3));

    let sub_a = sched
        .submit(EngineRequest {
            prompt_token_ids: p_a.clone(),
            params: params(TOKENS_PER_TURN),
        })
        .expect("submit a");
    // B arrives a step later, so the two are never in step with each other.
    sched.run_steps(2);
    let sub_b = sched
        .submit(EngineRequest {
            prompt_token_ids: p_b.clone(),
            params: params(TOKENS_PER_TURN),
        })
        .expect("submit b");
    sched.run_steps(500);

    let a = tokens_of(&drain(sub_a).await);
    let b = tokens_of(&drain(sub_b).await);

    let fresh = scheduler(executor(false));
    assert_eq!(a, run_turn(&fresh, &p_a).await);
    let fresh = scheduler(executor(false));
    assert_eq!(b, run_turn(&fresh, &p_b).await);
}
