//! The qwen3 state machine — a port of the reference server's `StreamingParserEngine` restricted to
//! the `qwen3_config` declaration (`parser/qwen3.py`), which is what production runs
//! for `qwen3_coder` / `qwen3_xml` / reasoning `qwen3`.
//!
//! Two branches of the original are deliberately absent, because the qwen3 config
//! cannot reach them, and reaching them silently would be worse than not compiling:
//!
//! * `tool_args_json` is `false`, so the JSON-aware argument buffer
//!   (`_feed_args_text`) never runs — argument text is emitted verbatim and parsed
//!   later by the `<parameter=…>` converter.
//! * `MESSAGE_HEADER` has no transition into it, so the header buffer never fills.
//!
//! Token-id pre-lexing (`TokenIDScanner`) is also not ported: it only fires when the
//! caller feeds token ids alongside the text, and Saber feeds text that its own
//! detokenizer has already assembled.

use super::lexer::{IncrementalLexer, LexToken, LexerShape, Terminal};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParserState {
    Content,
    Reasoning,
    ToolPreamble,
    ToolName,
    ToolArgs,
    ToolBetween,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventType {
    TextChunk,
    ReasoningChunk,
    ReasoningEnd,
    ToolCallStart,
    ToolName,
    ArgValueChunk,
    ToolCallEnd,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SemanticEvent {
    pub ty: EventType,
    pub value: String,
    pub tool_index: i32,
}

impl SemanticEvent {
    fn new(ty: EventType, value: &str, tool_index: i32) -> Self {
        Self {
            ty,
            value: value.to_owned(),
            tool_index,
        }
    }
}

/// `(next_state, events)` for one `(state, terminal)` pair.
fn transition(
    state: ParserState,
    terminal: Terminal,
) -> Option<(ParserState, &'static [EventType])> {
    use EventType as E;
    use ParserState as S;
    use Terminal as T;
    let table: (ParserState, &'static [EventType]) = match (state, terminal) {
        // Reasoning
        (S::Reasoning, T::ThinkStart) => (S::Reasoning, &[]),
        (S::Reasoning, T::ThinkEnd) => (S::Content, &[E::ReasoningEnd]),
        // A duplicate </think> after the transition is absorbed silently.
        (S::Content, T::ThinkEnd) => (S::Content, &[]),
        (S::Reasoning, T::ToolStart) => (S::ToolPreamble, &[E::ReasoningEnd, E::ToolCallStart]),
        // Tool calls
        (S::Content, T::ToolStart) => (S::ToolPreamble, &[E::ReasoningEnd, E::ToolCallStart]),
        // `<function=` without a preceding `<tool_call>`.
        (S::Content, T::FuncPrefix) => (S::ToolName, &[E::ToolCallStart]),
        (S::ToolPreamble, T::ToolEnd) => (S::Content, &[E::ToolCallEnd]),
        (S::ToolPreamble, T::FuncPrefix) => (S::ToolName, &[]),
        (S::ToolName, T::CloseAngle) => (S::ToolArgs, &[]),
        // Malformed: `</function>` while still reading the name (no closing `>`).
        (S::ToolName, T::FuncEnd) => (S::ToolBetween, &[E::ToolCallEnd]),
        (S::ToolArgs, T::FuncEnd) => (S::ToolBetween, &[E::ToolCallEnd]),
        (S::ToolArgs, T::ParamStart) => (S::ToolArgs, &[E::ArgValueChunk]),
        (S::ToolArgs, T::ParamEnd) => (S::ToolArgs, &[E::ArgValueChunk]),
        (S::ToolBetween, T::ToolEnd) => (S::Content, &[]),
        // Consecutive tool call with no closing `</tool_call>`.
        (S::ToolBetween, T::ToolStart) => (S::ToolPreamble, &[E::ToolCallStart]),
        (S::ToolBetween, T::FuncPrefix) => (S::ToolName, &[E::ToolCallStart]),
        _ => return None,
    };
    Some(table)
}

/// Terminals that take part in a tool-call transition. With `skip_tool_parsing`
/// these are surfaced as ordinary text instead of driving the machine, which is how
/// a server with a reasoning parser and no `--enable-auto-tool-choice` leaves tool
/// markup in `content`.
fn is_tool_terminal(terminal: Terminal) -> bool {
    matches!(
        terminal,
        Terminal::ToolStart
            | Terminal::ToolEnd
            | Terminal::FuncPrefix
            | Terminal::FuncEnd
            | Terminal::ParamStart
            | Terminal::ParamEnd
            | Terminal::CloseAngle
    )
}

/// What plain (non-terminal) text means in each state. `ToolPreamble` and
/// `ToolBetween` map to nothing, which is how stray text between the tags is dropped.
fn content_event(state: ParserState) -> Option<EventType> {
    match state {
        ParserState::Content => Some(EventType::TextChunk),
        ParserState::Reasoning => Some(EventType::ReasoningChunk),
        ParserState::ToolName => Some(EventType::ToolName),
        ParserState::ToolArgs => Some(EventType::ArgValueChunk),
        ParserState::ToolPreamble | ParserState::ToolBetween => None,
    }
}

/// The terminal literals of the qwen3 grammar, plus the special tokens to drop.
pub fn qwen3_lexer_shape(drop_tokens: &[String]) -> LexerShape {
    let mut literals = vec![
        ("<think>".to_owned(), Terminal::ThinkStart),
        ("</think>".to_owned(), Terminal::ThinkEnd),
        ("<tool_call>".to_owned(), Terminal::ToolStart),
        ("</tool_call>".to_owned(), Terminal::ToolEnd),
        ("<function=".to_owned(), Terminal::FuncPrefix),
        ("</function>".to_owned(), Terminal::FuncEnd),
        ("<parameter=".to_owned(), Terminal::ParamStart),
        ("</parameter>".to_owned(), Terminal::ParamEnd),
        (">".to_owned(), Terminal::CloseAngle),
    ];
    let configured: Vec<String> = literals.iter().map(|(lit, _)| lit.clone()).collect();
    for token in drop_tokens {
        if !configured.contains(token) {
            literals.push((token.clone(), Terminal::Drop));
        }
    }
    LexerShape::new(literals)
}

#[derive(Debug, Clone)]
pub struct Machine {
    lexer: IncrementalLexer,
    pub state: ParserState,
    pub tool_index: i32,
    initial_state: ParserState,
    /// The reference server's `skip_tool_parsing`, set by the reasoning-parser adapter.
    pub skip_tool_parsing: bool,
}

impl Machine {
    pub fn new(shape: LexerShape, initial_state: ParserState) -> Self {
        Self {
            lexer: IncrementalLexer::new(shape),
            state: initial_state,
            tool_index: -1,
            initial_state,
            skip_tool_parsing: false,
        }
    }

    pub fn reset(&mut self, initial_state: Option<ParserState>) {
        self.state = initial_state.unwrap_or(self.initial_state);
        self.tool_index = -1;
        self.lexer.reset();
    }

    pub fn feed(&mut self, text: &str) -> Vec<SemanticEvent> {
        let tokens = self.lexer.feed(text);
        self.process(tokens)
    }

    /// End of generation: flush the lexer, then close whatever state we are in.
    pub fn finish(&mut self) -> Vec<SemanticEvent> {
        let tokens = self.lexer.flush();
        let mut events = self.process(tokens);
        match self.state {
            ParserState::ToolPreamble
            | ParserState::ToolArgs
            | ParserState::ToolName
            | ParserState::ToolBetween => {
                if self.tool_index >= 0 {
                    events.push(SemanticEvent::new(
                        EventType::ToolCallEnd,
                        "",
                        self.tool_index,
                    ));
                }
                self.state = ParserState::Content;
            }
            ParserState::Reasoning => {
                events.push(SemanticEvent::new(
                    EventType::ReasoningEnd,
                    "",
                    self.tool_index,
                ));
                self.state = ParserState::Content;
            }
            ParserState::Content => {}
        }
        events
    }

    fn process(&mut self, tokens: Vec<LexToken>) -> Vec<SemanticEvent> {
        let mut events = Vec::new();
        for token in tokens {
            if token.terminal == Terminal::Content {
                if !token.value.is_empty() {
                    events.extend(self.emit_for_state(&token.value));
                }
            } else {
                events.extend(self.on_terminal(token.terminal, &token.value));
            }
        }
        events
    }

    fn on_terminal(&mut self, terminal: Terminal, value: &str) -> Vec<SemanticEvent> {
        match transition(self.state, terminal) {
            None => {
                if terminal == Terminal::Drop {
                    // A special token outside the grammar: dropped, never surfaced.
                    return Vec::new();
                }
                self.emit_for_state(value)
            }
            Some((next_state, events)) => {
                if self.skip_tool_parsing && is_tool_terminal(terminal) {
                    return self.emit_tool_terminal_as_text(events, value);
                }
                self.apply_transition(next_state, events, value)
            }
        }
    }

    /// `skip_tool_parsing`: the terminal is not applied, only surfaced. A terminal
    /// that would have ended reasoning still does, and its text follows as content.
    fn emit_tool_terminal_as_text(
        &mut self,
        event_types: &[EventType],
        value: &str,
    ) -> Vec<SemanticEvent> {
        if event_types.contains(&EventType::ReasoningEnd) {
            self.state = ParserState::Content;
            return vec![
                SemanticEvent::new(EventType::ReasoningEnd, value, self.tool_index),
                SemanticEvent::new(EventType::TextChunk, value, self.tool_index),
            ];
        }
        self.emit_for_state(value)
    }

    fn emit_for_state(&self, text: &str) -> Vec<SemanticEvent> {
        match content_event(self.state) {
            Some(ty) => vec![SemanticEvent::new(ty, text, self.tool_index)],
            None => Vec::new(),
        }
    }

    fn apply_transition(
        &mut self,
        next_state: ParserState,
        event_types: &[EventType],
        value: &str,
    ) -> Vec<SemanticEvent> {
        self.state = next_state;
        let mut events = Vec::with_capacity(event_types.len());
        for ty in event_types {
            if *ty == EventType::ToolCallStart {
                self.tool_index += 1;
            }
            if *ty == EventType::TextChunk && value.is_empty() {
                continue;
            }
            events.push(SemanticEvent::new(*ty, value, self.tool_index));
        }
        events
    }
}
