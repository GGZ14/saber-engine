//! Incremental detokenization.
//!
//! A step commits several token ids at once, and a UTF-8 character can straddle that
//! boundary: `é` is two tokens on a byte-level BPE vocabulary. Decoding each step in
//! isolation would emit `U+FFFD` and the client would see a permanent replacement
//! character where a letter belongs. The fix is the HuggingFace/the reference server incremental
//! algorithm: keep a small window of already-decoded ids, decode the window with and
//! without the new ids, and emit the difference **only when it does not end in a
//! replacement character**. Otherwise hold the bytes until the next step completes
//! the character.
//!
//! The tokenizer is behind [`TextCodec`] so the serving tests can run without a
//! checkpoint: [`ByteCodec`] is a one-byte-per-token vocabulary that reproduces the
//! split-character case exactly, and [`ModelCodec`] wraps `saber-model`'s real one.

use std::sync::Arc;

use crate::engine::TokenId;
use crate::protocol::ApiError;

/// The tokenizer, as the serving plane needs it.
pub trait TextCodec: Send + Sync + std::fmt::Debug {
    /// Encode a rendered prompt. The chat path never adds special tokens: the
    /// template already emits `<|im_start|>` / `<|im_end|>`.
    fn encode(&self, text: &str) -> Result<Vec<TokenId>, ApiError>;
    fn decode(&self, ids: &[TokenId], skip_special_tokens: bool) -> String;
    /// The literal piece for one id, for the `token` field of a logprob entry.
    fn id_to_piece(&self, id: TokenId) -> String {
        self.decode(&[id], false)
    }
}

/// `saber-model`'s tokenizer — the production implementation.
#[derive(Debug)]
pub struct ModelCodec {
    tokenizer: Arc<saber_model::Tokenizer>,
}

impl ModelCodec {
    pub fn new(tokenizer: Arc<saber_model::Tokenizer>) -> Self {
        Self { tokenizer }
    }
}

impl TextCodec for ModelCodec {
    fn encode(&self, text: &str) -> Result<Vec<TokenId>, ApiError> {
        self.tokenizer
            .encode(text)
            .map_err(|e| ApiError::internal(format!("tokenizer: {e}")))
    }

    fn decode(&self, ids: &[TokenId], skip_special_tokens: bool) -> String {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .unwrap_or_default()
    }

    fn id_to_piece(&self, id: TokenId) -> String {
        self.tokenizer
            .id_to_token(id)
            .unwrap_or_else(|| self.decode(&[id], false))
    }
}

/// One byte per token: ids `0..=255` are raw bytes, anything above has no text.
///
/// Not a toy for its own sake — it is the counterpart of [`crate::engine::MockEngine`]
/// and it makes the split-character case reproducible without a 27B checkpoint.
#[derive(Debug, Default, Clone, Copy)]
pub struct ByteCodec;

impl TextCodec for ByteCodec {
    fn encode(&self, text: &str) -> Result<Vec<TokenId>, ApiError> {
        Ok(text.bytes().map(u32::from).collect())
    }

    fn decode(&self, ids: &[TokenId], _skip_special_tokens: bool) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .filter(|id| **id < 256)
            .map(|id| *id as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

const REPLACEMENT: char = '\u{fffd}';

/// Streaming detokenizer over one request's output.
#[derive(Debug)]
pub struct IncrementalDetokenizer {
    codec: Arc<dyn TextCodec>,
    ids: Vec<TokenId>,
    prefix_offset: usize,
    read_offset: usize,
    skip_special_tokens: bool,
    /// Steps whose text was withheld because a character was still incomplete.
    held_steps: u64,
    /// Decodes where the longer window did not extend the shorter one. The fallback
    /// is "emit nothing this step"; the counter exists so the route is never silent
    /// (CLAUDE.md rule 5).
    nonprefix_decodes: u64,
}

impl IncrementalDetokenizer {
    pub fn new(codec: Arc<dyn TextCodec>, skip_special_tokens: bool) -> Self {
        Self {
            codec,
            ids: Vec::new(),
            prefix_offset: 0,
            read_offset: 0,
            skip_special_tokens,
            held_steps: 0,
            nonprefix_decodes: 0,
        }
    }

    /// Feed one step's tokens; returns the text that is now safe to emit, which is
    /// `""` while a multi-byte character is still incomplete.
    pub fn push(&mut self, new_ids: &[TokenId]) -> String {
        self.ids.extend_from_slice(new_ids);
        let prefix_text = self.codec.decode(
            &self.ids[self.prefix_offset..self.read_offset],
            self.skip_special_tokens,
        );
        let new_text = self
            .codec
            .decode(&self.ids[self.prefix_offset..], self.skip_special_tokens);

        if new_text.len() <= prefix_text.len() {
            self.held_steps += 1;
            return String::new();
        }
        if !new_text.starts_with(&prefix_text) {
            self.nonprefix_decodes += 1;
            self.held_steps += 1;
            return String::new();
        }
        if new_text.ends_with(REPLACEMENT) {
            // An incomplete character: hold everything until the next step closes it.
            self.held_steps += 1;
            return String::new();
        }
        let delta = new_text[prefix_text.len()..].to_owned();
        self.prefix_offset = self.read_offset;
        self.read_offset = self.ids.len();
        delta
    }

    /// End of generation: emit whatever is left, replacement characters included —
    /// nothing more is coming that could complete them.
    pub fn finish(&mut self) -> String {
        let prefix_text = self.codec.decode(
            &self.ids[self.prefix_offset..self.read_offset],
            self.skip_special_tokens,
        );
        let new_text = self
            .codec
            .decode(&self.ids[self.prefix_offset..], self.skip_special_tokens);
        self.prefix_offset = self.read_offset;
        self.read_offset = self.ids.len();
        if new_text.len() > prefix_text.len() && new_text.starts_with(&prefix_text) {
            new_text[prefix_text.len()..].to_owned()
        } else {
            self.nonprefix_decodes += u64::from(!new_text.starts_with(&prefix_text));
            String::new()
        }
    }

    /// How many steps withheld their text (diagnostic, exported by tests).
    pub fn held_steps(&self) -> u64 {
        self.held_steps
    }

    pub fn nonprefix_decodes(&self) -> u64 {
        self.nonprefix_decodes
    }

    pub fn token_count(&self) -> usize {
        self.ids.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detok() -> IncrementalDetokenizer {
        IncrementalDetokenizer::new(Arc::new(ByteCodec), false)
    }

    #[test]
    fn a_character_split_across_steps_never_emits_a_replacement_char() {
        // "é" is 0xC3 0xA9; the steps split it. A step that ends mid-character emits
        // nothing at all — the reference server withholds the whole delta rather than the tail, and
        // matching that keeps step boundaries comparable with prod.
        let mut d = detok();
        assert_eq!(d.push(&[b'a' as u32, 0xC3]), "");
        assert_eq!(d.push(&[0xA9, b'b' as u32]), "aéb");
        assert_eq!(d.finish(), "");
        assert_eq!(d.held_steps(), 1);
        assert_eq!(d.nonprefix_decodes(), 0);
    }

    #[test]
    fn a_character_split_three_ways_holds_until_complete() {
        // "🌤" is four bytes; feed them one step at a time.
        let mut d = detok();
        assert_eq!(d.push(&[0xF0]), "");
        assert_eq!(d.push(&[0x9F]), "");
        assert_eq!(d.push(&[0x8C]), "");
        assert_eq!(d.push(&[0xA4]), "🌤");
        assert_eq!(d.held_steps(), 3);
    }

    #[test]
    fn a_truncated_character_at_end_of_stream_is_surfaced_once() {
        let mut d = detok();
        assert_eq!(d.push(&[b'h' as u32, b'i' as u32]), "hi");
        assert_eq!(d.push(&[0xC3]), "", "held: could still complete");
        assert_eq!(d.finish(), "\u{fffd}", "end of stream: surface it");
    }

    #[test]
    fn concatenated_steps_reconstruct_the_whole_text() {
        let text = "héllo, 世界 🌤️ — done";
        let ids = ByteCodec.encode(text).expect("encode");
        for width in [1usize, 2, 3, 5, 8] {
            let mut d = detok();
            let mut out = String::new();
            for chunk in ids.chunks(width) {
                out.push_str(&d.push(chunk));
            }
            out.push_str(&d.finish());
            assert_eq!(out, text, "width {width}");
        }
    }
}
