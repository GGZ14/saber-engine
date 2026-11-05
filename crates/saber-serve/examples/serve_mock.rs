//! `cargo run -p saber-serve --example serve_mock` — the serving plane on an
//! ephemeral port, backed by [`MockEngine`], with no GPU, no model and no HIP.
//!
//! It prints the base URL and the exact `curl` lines to drive it, so the OpenAI
//! surface can be poked by hand (and by `tests/http_openai.rs`, which does the same
//! thing in-process) long before an engine exists.

use std::sync::Arc;
use std::time::Duration;

use saber_serve::detok::ByteCodec;
use saber_serve::engine::{FinishReason, MockEngine, MockScript, ModelInfo};
use saber_serve::state::{MinimalChatRenderer, ReasoningParserKind, ServerState, ToolParserKind};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = ModelInfo {
        served_model_name: "Qwen3.8".to_owned(),
        aliases: vec![],
        max_model_len: 4096,
        root: "/models/Qwen3.8-27B-NVFP4".to_owned(),
        eos_token_ids: vec![255],
    };
    let engine = Arc::new(MockEngine::new(model).with_step_delay(Duration::from_millis(20)));

    // Five scripted generations: a plain answer, a reasoning-then-answer, a tool
    // call, and two more plain answers so several requests can be made by hand.
    let reply = "Hi! How can I help you today?";
    let thinking = "The user is greeting me</think>Hello!";
    let tool_call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
    for text in [reply, thinking, tool_call, reply, reply] {
        // Three tokens per step: the speculative shape, which is what makes the
        // stream one-update-per-step rather than one-per-token.
        engine.push_script(MockScript::in_steps_of(
            text.bytes().map(u32::from),
            3,
            FinishReason::Stop,
        ));
    }

    let state = ServerState::builder(engine, Arc::new(ByteCodec), Arc::new(MinimalChatRenderer))
        .reasoning_parser(ReasoningParserKind::Qwen3)
        .tool_parser(ToolParserKind::Qwen3Coder)
        .enable_auto_tool_choice(true)
        .default_enable_thinking(false)
        .build();

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    println!("saber-serve (mock engine) listening on http://{addr}");
    println!(
        "  curl -sN http://{addr}/v1/chat/completions -H 'Content-Type: application/json' \\\n\
         \x20   -d '{{\"model\":\"Qwen3.8\",\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}],\
         \"stream\":true,\"stream_options\":{{\"include_usage\":true}}}}'"
    );
    println!("  curl -s http://{addr}/metrics | grep spec_decode");
    saber_serve::server::serve(state, listener).await?;
    Ok(())
}
