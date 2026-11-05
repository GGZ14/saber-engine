//! FN27 ("ok, align 27b", the user, 2026-09-29): the 27B's plane builders run prod the reference 0.27.1's
//! tool-call `arguments` law (`ToolArguments::ObjectsStrict`), not the builder's default
//! (`Verbatim`, the pre-FN27 27B plane and FN27's control) and not the reference 0.29's (`Objects`, the
//! Flash-Next plane). Every 27B launcher reaches `build_state_with_vision` through one of these.
//! FN28 ("ok, align that also for 27b"): the same builders render prod 0.27.1's rebuilt
//! conversation (`ConversationForm::Prod0271`); the builder's default stays `AsSent`.

mod common;

use std::sync::Arc;

use common::{boot_ledger, decode_buckets, plan_with, GIB};
use saber_sched::{
    build_state, build_state_with, build_state_with_parsers, build_state_with_vision,
    mock_model_info, ArenaBytes, KvGeometry, Scheduler, SchedulerConfig, ScriptExecutor,
};
use saber_serve::detok::ByteCodec;
use saber_serve::state::{
    ConversationForm, MinimalChatRenderer, ResolvedParsers, ServerState, ToolArguments,
};

#[test]
fn every_27b_plane_builder_runs_the_0271_law() {
    let eos = 255;
    let plan = plan_with(&decode_buckets(), 2);
    let config = SchedulerConfig::from_plan(
        &plan,
        KvGeometry::PROD_FP8,
        ArenaBytes {
            kv_per_rank: 8 * GIB,
            gdn_state_per_rank: 8 * GIB,
        },
        262_144,
    )
    .expect("config");
    let model = mock_model_info("Qwen3.8", "/models/mock", eos, config.max_context_tokens);
    let sched = Arc::new(
        Scheduler::new(
            config,
            ScriptExecutor::new(vec![vec![eos]], 5, eos),
            model,
            boot_ledger(),
        )
        .expect("arenas fit"),
    );
    let states = [
        build_state(
            Arc::clone(&sched),
            Arc::new(ByteCodec),
            Arc::new(MinimalChatRenderer),
        ),
        build_state_with(
            Arc::clone(&sched),
            Arc::new(ByteCodec),
            Arc::new(MinimalChatRenderer),
            20,
        ),
        build_state_with_parsers(
            Arc::clone(&sched),
            Arc::new(ByteCodec),
            Arc::new(MinimalChatRenderer),
            20,
            &ResolvedParsers::prod(),
        ),
        build_state_with_vision(
            Arc::clone(&sched),
            Arc::new(ByteCodec),
            Arc::new(MinimalChatRenderer),
            20,
            &ResolvedParsers::prod(),
            0,
            999,
        ),
    ];
    for state in &states {
        assert_eq!(state.tool_arguments, ToolArguments::ObjectsStrict);
        assert_eq!(state.conversation_form, ConversationForm::Prod0271);
    }
    // The control: the builder's own default is still the pre-FN27 law.
    let bare =
        ServerState::builder(sched, Arc::new(ByteCodec), Arc::new(MinimalChatRenderer)).build();
    assert_eq!(bare.tool_arguments, ToolArguments::Verbatim);
    assert_eq!(bare.conversation_form, ConversationForm::AsSent);
}
