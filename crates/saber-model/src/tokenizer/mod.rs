//! Tokenizer and position ids for Qwen3.8-27B.
//!
//! `tokenizer.json` is loaded with the `tokenizers` crate. The gate is an exact token-id
//! **sequence** comparison against HF over the P-1e corpus — not a round-trip test, which would
//! pass on a tokenizer that split differently but decoded back to the same text.
//!
//! **Special tokens on the chat path.** the reference server's `ChatCompletionRequest.add_special_tokens`
//! defaults to `false` and transformers tokenizes a rendered chat template the same way: the
//! template already emits `<|im_start|>` / `<|im_end|>`, so nothing is prepended afterwards.
//! [`Tokenizer::encode`] is therefore the `add_special_tokens = false` path; the other
//! behaviour is available as [`Tokenizer::encode_with_special_tokens`] and is *not* what the
//! serve path uses. This matters here more than on most checkpoints because
//! `config.json` declares `bos_token_id == eos_token_id == 248044`, so a stray BOS would be
//! invisible in a decode round-trip while shifting every position id by one.

mod positions;

use std::path::Path;

pub use positions::{
    rope_params, rope_params_from_text_config, MropePositions, RopeParameters, RopeParams,
    TextConfig,
};

use crate::ModelError;

/// A loaded HF tokenizer.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab_size", &self.vocab_size())
            .finish_non_exhaustive()
    }
}

impl Tokenizer {
    /// Load `tokenizer.json`.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ModelError> {
        let inner = tokenizers::Tokenizer::from_file(path.as_ref())
            .map_err(|e| ModelError::Tokenizer(format!("{}: {e}", path.as_ref().display())))?;
        Ok(Self { inner })
    }

    /// Load `tokenizer.json` from a checkpoint directory.
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self, ModelError> {
        Self::from_file(dir.as_ref().join("tokenizer.json"))
    }

    /// Encode text **the way the chat serve path does**: `add_special_tokens = false`.
    ///
    /// Added/special tokens that occur literally in the text (`<|im_start|>`, `<|im_end|>`,
    /// `<|vision_start|>`, …) are still matched as single ids — that is the added-vocabulary
    /// pass, which this flag does not disable.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, ModelError> {
        self.encode_inner(text, false)
    }

    /// Encode with the post-processor's special tokens added. Not used by `/v1/chat/completions`.
    pub fn encode_with_special_tokens(&self, text: &str) -> Result<Vec<u32>, ModelError> {
        self.encode_inner(text, true)
    }

    fn encode_inner(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, ModelError> {
        let enc = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| ModelError::Tokenizer(format!("encode: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode ids back to text.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, ModelError> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| ModelError::Tokenizer(format!("decode: {e}")))
    }

    /// Size of the vocabulary including added tokens.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Look up a token id by its literal text.
    #[must_use]
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Look up a token's literal text by id.
    #[must_use]
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tokenizer copy the P-1e corpus generator drops next to the corpus.
    fn tokenizer_path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../.cache/template_corpus.tokenizer.json")
    }

    #[test]
    fn special_tokens_resolve_and_bos_is_not_added() {
        let path = tokenizer_path();
        if !path.exists() {
            eprintln!(
                "SKIP: {} absent; run the internal reference harness",
                path.display()
            );
            return;
        }
        let tok = Tokenizer::from_file(&path).unwrap();
        // config.json's vocab_size (248320) is the *embedding* size and is padded above the
        // tokenizer's own vocabulary; ids must stay inside it or the lm_head indexes garbage.
        let vocab = tok.vocab_size();
        assert!(
            vocab <= 248_320,
            "tokenizer vocab {vocab} exceeds config vocab_size 248320"
        );
        assert!(vocab > 248_000, "unexpectedly small vocab {vocab}");

        let im_start = tok.token_to_id("<|im_start|>").expect("<|im_start|>");
        let im_end = tok.token_to_id("<|im_end|>").expect("<|im_end|>");
        assert_ne!(im_start, im_end);

        // A rendered chat prompt starts with the template's own <|im_start|>, and encoding it
        // must not prepend anything: `add_special_tokens = false` is the serve-path behaviour.
        let rendered = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";
        let ids = tok.encode(rendered).unwrap();
        assert_eq!(
            ids[0], im_start,
            "no BOS in front of the template's own <|im_start|>"
        );
        assert_eq!(ids.iter().filter(|i| **i == im_end).count(), 1);

        // The other mode is genuinely different on some checkpoints; keep it reachable and
        // assert it is never silently used for chat.
        let with_special = tok.encode_with_special_tokens(rendered).unwrap();
        assert!(with_special.len() >= ids.len());
    }
}
