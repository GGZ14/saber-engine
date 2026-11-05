//! Incremental literal lexer — a port of the reference server's `parser/engine/incremental_lexer.py`.
//!
//! The behaviour that matters is **prefix-match buffering**: text that could still
//! become a terminal (`"<tool_"`) is held rather than emitted, so a tag split across
//! two engine steps is still recognised. Everything the qwen3 config declares is a
//! literal, so the regex branch of the original is not ported; a non-literal terminal
//! would need it and is rejected loudly at construction.

use std::collections::{HashMap, HashSet};

/// Terminal names of the qwen3 grammar, plus the two engine-level pseudo-terminals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Terminal {
    ThinkStart,
    ThinkEnd,
    ToolStart,
    ToolEnd,
    FuncPrefix,
    FuncEnd,
    ParamStart,
    ParamEnd,
    CloseAngle,
    /// A special token that is not part of the grammar: dropped silently.
    Drop,
    /// Ordinary text.
    Content,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LexToken {
    pub terminal: Terminal,
    pub value: String,
}

/// Immutable lexer tables, shared by every request.
#[derive(Debug, Clone)]
pub struct LexerShape {
    literals: Vec<(String, Terminal)>,
    first_chars: HashSet<u8>,
    prefix_set: HashSet<String>,
    by_first: HashMap<u8, Vec<(String, Terminal)>>,
}

impl LexerShape {
    /// `literals` are `(text, terminal)` pairs; every text must be ASCII (all of the
    /// qwen3 terminals and every Qwen special token are).
    pub fn new(literals: Vec<(String, Terminal)>) -> Self {
        assert!(
            literals
                .iter()
                .all(|(lit, _)| lit.is_ascii() && !lit.is_empty()),
            "lexer literals must be non-empty ASCII"
        );
        let mut first_chars = HashSet::new();
        let mut prefix_set = HashSet::new();
        let mut by_first: HashMap<u8, Vec<(String, Terminal)>> = HashMap::new();
        for (lit, name) in &literals {
            let first = lit.as_bytes()[0];
            first_chars.insert(first);
            for i in 1..lit.len() {
                prefix_set.insert(lit[..i].to_owned());
            }
            by_first
                .entry(first)
                .or_default()
                .push((lit.clone(), *name));
        }
        Self {
            literals,
            first_chars,
            prefix_set,
            by_first,
        }
    }
}

/// Streaming text → terminal tokens.
#[derive(Debug, Clone)]
pub struct IncrementalLexer {
    shape: LexerShape,
    buffer: String,
}

impl IncrementalLexer {
    pub fn new(shape: LexerShape) -> Self {
        Self {
            shape,
            buffer: String::new(),
        }
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
    }

    pub fn buffer_is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn feed(&mut self, text: &str) -> Vec<LexToken> {
        if self.buffer.is_empty() && !self.shape.first_chars.is_empty() {
            // Fast path: nothing in this chunk can start a terminal.
            if !text.bytes().any(|b| self.shape.first_chars.contains(&b)) {
                if text.is_empty() {
                    return Vec::new();
                }
                return vec![LexToken {
                    terminal: Terminal::Content,
                    value: text.to_owned(),
                }];
            }
        }
        self.buffer.push_str(text);
        self.drain(false)
    }

    /// End of stream: emit whatever is buffered, terminals first, then raw text.
    pub fn flush(&mut self) -> Vec<LexToken> {
        let mut tokens = Vec::new();
        if !self.buffer.is_empty() {
            tokens.extend(self.drain(true));
        }
        if !self.buffer.is_empty() {
            tokens.push(LexToken {
                terminal: Terminal::Content,
                value: std::mem::take(&mut self.buffer),
            });
        }
        tokens
    }

    fn drain(&mut self, final_pass: bool) -> Vec<LexToken> {
        let mut tokens = Vec::new();
        while !self.buffer.is_empty() {
            if !self
                .buffer
                .bytes()
                .any(|b| self.shape.first_chars.contains(&b))
            {
                tokens.push(LexToken {
                    terminal: Terminal::Content,
                    value: std::mem::take(&mut self.buffer),
                });
                break;
            }

            let first = self.buffer.as_bytes()[0];
            let candidates = self.shape.by_first.get(&first);
            let mut best: Option<(Terminal, usize)> = None;
            if let Some(candidates) = candidates {
                for (lit, name) in candidates {
                    if self.buffer.starts_with(lit.as_str())
                        && best.is_none_or(|(_, len)| lit.len() > len)
                    {
                        best = Some((*name, lit.len()));
                    }
                }
            }

            // A complete literal that is also the prefix of a longer one waits for
            // more input, unless this is the final pass.
            if !final_pass && self.shape.prefix_set.contains(&self.buffer) {
                match best {
                    Some((name, len)) => {
                        let longer = candidates.is_some_and(|c| {
                            c.iter().any(|(lit, _)| {
                                lit.len() > len && lit.starts_with(self.buffer.as_str())
                            })
                        });
                        if longer {
                            break;
                        }
                        tokens.push(LexToken {
                            terminal: name,
                            value: self.buffer[..len].to_owned(),
                        });
                        self.buffer.drain(..len);
                        continue;
                    }
                    None => break,
                }
            }

            if let Some((name, len)) = best {
                tokens.push(LexToken {
                    terminal: name,
                    value: self.buffer[..len].to_owned(),
                });
                self.buffer.drain(..len);
            } else {
                let boundary = self.content_boundary();
                if boundary > 0 {
                    tokens.push(LexToken {
                        terminal: Terminal::Content,
                        value: self.buffer[..boundary].to_owned(),
                    });
                    self.buffer.drain(..boundary);
                } else {
                    let step = self.buffer.chars().next().map(char::len_utf8).unwrap_or(1);
                    tokens.push(LexToken {
                        terminal: Terminal::Content,
                        value: self.buffer[..step].to_owned(),
                    });
                    self.buffer.drain(..step);
                }
            }
        }
        tokens
    }

    /// Byte offset of the earliest position after 0 where a literal could begin.
    ///
    /// Byte-wise rather than char-wise (the original is char-wise): every literal is
    /// ASCII, so a match forces the compared region to be ASCII, where the two agree.
    fn content_boundary(&self) -> usize {
        let bytes = self.buffer.as_bytes();
        for i in 1..bytes.len() {
            if !self.shape.first_chars.contains(&bytes[i]) {
                continue;
            }
            let remaining = bytes.len() - i;
            for (lit, _) in &self.shape.literals {
                let check = remaining.min(lit.len());
                if bytes[i..i + check] == lit.as_bytes()[..check] {
                    return i;
                }
            }
        }
        bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> LexerShape {
        LexerShape::new(vec![
            ("<tool_call>".to_owned(), Terminal::ToolStart),
            ("</tool_call>".to_owned(), Terminal::ToolEnd),
            ("<think>".to_owned(), Terminal::ThinkStart),
            ("</think>".to_owned(), Terminal::ThinkEnd),
            (">".to_owned(), Terminal::CloseAngle),
        ])
    }

    #[test]
    fn holds_a_tag_split_across_feeds() {
        let mut lexer = IncrementalLexer::new(shape());
        assert!(lexer.feed("<tool_").is_empty());
        let tokens = lexer.feed("call>x");
        assert_eq!(tokens[0].terminal, Terminal::ToolStart);
        assert_eq!(tokens[1].terminal, Terminal::Content);
        assert_eq!(tokens[1].value, "x");
    }

    #[test]
    fn unterminated_prefix_flushes_as_content() {
        let mut lexer = IncrementalLexer::new(shape());
        assert!(lexer.feed("<tool_c").is_empty());
        let tokens = lexer.flush();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].terminal, Terminal::Content);
        assert_eq!(tokens[0].value, "<tool_c");
    }

    #[test]
    fn multibyte_content_is_not_split_mid_character() {
        let mut lexer = IncrementalLexer::new(shape());
        let tokens = lexer.feed("é€🌤️<think>");
        let text: String = tokens
            .iter()
            .filter(|t| t.terminal == Terminal::Content)
            .map(|t| t.value.as_str())
            .collect();
        assert_eq!(text, "é€🌤️");
        assert_eq!(tokens.last().unwrap().terminal, Terminal::ThinkStart);
    }
}
