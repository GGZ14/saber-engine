//! Sampling parameters, validated the way the reference server validates them.
//!
//! The messages are the reference 0.27.1's own (`upstream/sampling_params.py`) wrapped in the
//! envelope prod returns, because harnesses on this box match on them; they were
//! captured from the live server rather than guessed.

use crate::engine::TokenId;
use crate::protocol::ApiError;

/// Everything the engine needs to sample, in the shape the request actually sent.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    pub n: usize,
    pub temperature: f32,
    pub top_p: f32,
    /// The reference server 0.27 semantics: `0` disables.
    pub top_k: i32,
    pub min_p: f32,
    pub max_tokens: Option<usize>,
    pub stop: Vec<String>,
    pub stop_token_ids: Vec<TokenId>,
    pub seed: Option<u64>,
    /// Number of top logprobs per sampled token; `Some(0)` means the sampled token
    /// only and `Some(-1)` means all of them, as in the reference server.
    pub logprobs: Option<i32>,
    pub prompt_logprobs: Option<i32>,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repetition_penalty: f32,
    pub ignore_eos: bool,
    pub include_stop_str_in_output: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            n: 1,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            max_tokens: None,
            stop: Vec::new(),
            stop_token_ids: Vec::new(),
            seed: None,
            logprobs: None,
            prompt_logprobs: None,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            ignore_eos: false,
            include_stop_str_in_output: false,
        }
    }
}

/// The default floor of the `prompt_logprobs` cap: the device seam's K (`saber_kernels::head::DEVICE_ROW_K`),
/// the widest top-k a prompt row can be answered with.
pub const DEFAULT_MAX_PROMPT_LOGPROBS: i32 = 64;

/// Server-side limits a request is validated against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplingLimits {
    /// `--max-logprobs` (the reference server's default is 20): the cap on the sampled token's
    /// `logprobs` / `top_logprobs`, and on `prompt_logprobs` when it is above
    /// [`Self::max_prompt_logprobs`].
    pub max_logprobs: i32,
    /// The floor of the `prompt_logprobs` cap: [`DEFAULT_MAX_PROMPT_LOGPROBS`] = 64, the
    /// seam's width, so a teacher-forced scorer (QualityBench's `logprobs` task reads the top 64
    /// of every prompt position) is answered without lifting the sampled-token cap. The cap a
    /// request meets is the larger of this and `--max-logprobs` ([`Self::max_prompt_logprobs_resolved`]);
    /// above it the refusal is the reference server's own 400, text and shape.
    pub max_prompt_logprobs: i32,
    pub max_model_len: usize,
    /// the reference server's max-n-sequences environment knob, read off the prod image: 16384. A plan field rather than
    /// a `getenv` (CLAUDE.md, structural rule 4).
    pub max_n: usize,
    /// `model_config.get_vocab_size()`: what `-1` means, for `logprobs`, `prompt_logprobs` and
    /// `--max-logprobs -1`. Qwen3.8's is 248,320.
    pub vocab_size: usize,
}

impl SamplingLimits {
    /// `--max-logprobs`, resolved the way `_validate_logprobs` resolves it: `-1` is the whole
    /// vocabulary.
    #[must_use]
    pub fn max_logprobs_resolved(&self) -> i64 {
        if self.max_logprobs == -1 {
            self.vocab_size as i64
        } else {
            i64::from(self.max_logprobs)
        }
    }

    /// The `prompt_logprobs` cap: the larger of `--max-logprobs` and [`Self::max_prompt_logprobs`],
    /// each resolved (`-1` = the whole vocabulary).
    #[must_use]
    pub fn max_prompt_logprobs_resolved(&self) -> i64 {
        self.max_logprobs_resolved()
            .max(self.resolve(self.max_prompt_logprobs))
    }

    /// A request's `logprobs` / `prompt_logprobs`, resolved the same way.
    #[must_use]
    pub fn resolve(&self, requested: i32) -> i64 {
        if requested == -1 {
            self.vocab_size as i64
        } else {
            i64::from(requested)
        }
    }
}

impl Default for SamplingLimits {
    fn default() -> Self {
        Self {
            max_logprobs: 20,
            max_prompt_logprobs: DEFAULT_MAX_PROMPT_LOGPROBS,
            max_model_len: 262_144,
            max_n: 16_384,
            vocab_size: 248_320,
        }
    }
}

/// What a request that sends no `temperature` / `top_p` / `top_k` / `min_p` /
/// `repetition_penalty` actually gets.
///
/// This is **not** the OpenAI default, and getting it wrong is invisible on the wire:
/// The reference server merges the checkpoint's `generation_config.json` into every request
/// (`--generation-config auto` is the default), and prod goes further and passes
/// `--override-generation-config '{"temperature":0.7,"top_p":0.95,"top_k":20}'`
/// (`serve-mxfp4.sh:1072`). Measured in the prod image with prod's own
/// `ModelConfig.get_diff_sampling_param()` on 2026-09-21: `{'temperature': 0.7,
/// 'top_k': 20, 'top_p': 0.95}`. A Saber that defaulted to `1.0 / 1.0 / 0` would sample
/// from a **different distribution than prod on every unparameterised request**, and
/// every gate in this tree that sends `temperature: 0` would still be green.
///
/// Only the five keys `ChatCompletionRequest.to_sampling_params` consults are here;
/// `presence_penalty` and `frequency_penalty` are taken from the request alone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DefaultSampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub repetition_penalty: f32,
}

impl DefaultSampling {
    /// The reference server's own defaults, for a server started with the reference server's stock generation config.
    pub const STOCK: Self = Self {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        repetition_penalty: 1.0,
    };

    /// What the `qwen_prod_nvfp4` unit serves with.
    pub const PROD: Self = Self {
        temperature: 0.7,
        top_p: 0.95,
        top_k: 20,
        min_p: 0.0,
        repetition_penalty: 1.0,
    };
}

impl Default for DefaultSampling {
    fn default() -> Self {
        Self::STOCK
    }
}

/// CPython's `repr` of a float, which is what lands in the error message prod returns:
/// `-1.0`, not `-1`; `1e-05`, not `0.00001`. The messages are matched on by harnesses on
/// this box, so the formatting is part of the contract.
fn py_float(v: f32) -> String {
    let d = f64::from(v);
    if d.is_nan() {
        return "nan".to_owned();
    }
    if d.is_infinite() {
        return if d > 0.0 {
            "inf".to_owned()
        } else {
            "-inf".to_owned()
        };
    }
    let mut s = format!("{d:?}");
    if let Some(e) = s.find('e') {
        // Rust writes `1e-5`, CPython writes `1e-05`.
        let (mant, exp) = s.split_at(e);
        let exp = &exp[1..];
        let (sign, digits) = if let Some(rest) = exp.strip_prefix('-') {
            ("-", rest)
        } else {
            ("+", exp.trim_start_matches('+'))
        };
        let digits = if digits.len() < 2 {
            format!("0{digits}")
        } else {
            digits.to_owned()
        };
        s = format!("{mant}e{sign}{digits}");
    }
    s
}

impl SamplingParams {
    /// The reference server's `SamplingParams._verify_args`, **in its order**, with its messages.
    ///
    /// Two details that are not cosmetic:
    ///
    /// * the order is the order, because a request with two bad parameters gets the
    ///   message for whichever the reference server checks first. The reference server checks `n`, then the three
    ///   penalties, then `temperature`, then `top_p`, `top_k`, `min_p`, `max_tokens`
    ///   and the logprob pair — not the order the struct happens to declare them in;
    /// * only some of them carry the ` (parameter=…, value=…)` suffix.
    ///   `ValidationError.__str__` appends it **only when the raise passed
    ///   `parameter=`**, and `_verify_args` passes it for `temperature`, `top_p`,
    ///   `max_tokens`, `logprobs` and `prompt_logprobs` and for nothing else. A client
    ///   matching on the message sees the difference.
    ///
    /// `min_tokens`, `stream_interval`, `bad_words` and `detokenize` are not on this
    /// struct and are not checked here; their absence is stated rather than implied.
    pub fn validate(&self, limits: SamplingLimits) -> Result<(), ApiError> {
        if self.n < 1 {
            return Err(ApiError::bad_request(
                format!("n must be at least 1, got {}.", self.n),
                None,
            ));
        }
        if self.n > limits.max_n {
            return Err(ApiError::bad_request(
                format!("n must be at most {}, got {}.", limits.max_n, self.n),
                None,
            ));
        }
        if !(-2.0..=2.0).contains(&self.presence_penalty) {
            return Err(ApiError::bad_request(
                format!(
                    "presence_penalty must be in [-2, 2], got {}.",
                    py_float(self.presence_penalty)
                ),
                None,
            ));
        }
        if !(-2.0..=2.0).contains(&self.frequency_penalty) {
            return Err(ApiError::bad_request(
                format!(
                    "frequency_penalty must be in [-2, 2], got {}.",
                    py_float(self.frequency_penalty)
                ),
                None,
            ));
        }
        if !self.repetition_penalty.is_finite() {
            return Err(ApiError::bad_request(
                format!(
                    "repetition_penalty must be a finite number, got {}.",
                    py_float(self.repetition_penalty)
                ),
                None,
            ));
        }
        if self.repetition_penalty <= 0.0 {
            return Err(ApiError::bad_request(
                format!(
                    "repetition_penalty must be greater than zero, got {}.",
                    py_float(self.repetition_penalty)
                ),
                None,
            ));
        }
        if !self.temperature.is_finite() {
            return Err(ApiError::invalid_param(
                format!(
                    "temperature must be a finite number, got {}.",
                    py_float(self.temperature)
                ),
                "temperature",
                py_float(self.temperature),
            ));
        }
        if self.temperature < 0.0 {
            return Err(ApiError::invalid_param(
                format!(
                    "temperature must be non-negative, got {}.",
                    py_float(self.temperature)
                ),
                "temperature",
                py_float(self.temperature),
            ));
        }
        if self.temperature > 2.0 {
            return Err(ApiError::invalid_param(
                format!(
                    "temperature must be in [0, 2], got {}.",
                    py_float(self.temperature)
                ),
                "temperature",
                py_float(self.temperature),
            ));
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err(ApiError::invalid_param(
                format!("top_p must be in (0, 1], got {}.", py_float(self.top_p)),
                "top_p",
                py_float(self.top_p),
            ));
        }
        // The reference server "quietly accepts -1 as disabled, but prefers 0": only `< -1` is refused.
        if self.top_k < -1 {
            return Err(ApiError::bad_request(
                format!(
                    "top_k must be 0 (disable), or at least 1, got {}.",
                    self.top_k
                ),
                None,
            ));
        }
        if !(0.0..=1.0).contains(&self.min_p) {
            return Err(ApiError::bad_request(
                format!("min_p must be in [0, 1], got {}.", py_float(self.min_p)),
                None,
            ));
        }
        if let Some(max_tokens) = self.max_tokens {
            if max_tokens == 0 {
                return Err(ApiError::invalid_param(
                    "max_tokens must be at least 1, got 0.",
                    "max_tokens",
                    0,
                ));
            }
        }
        if let Some(logprobs) = self.logprobs {
            if logprobs < -1 {
                return Err(ApiError::invalid_param(
                    format!("logprobs must be non-negative or -1, got {logprobs}."),
                    "logprobs",
                    logprobs,
                ));
            }
            // `_validate_logprobs`: `-1` is the whole vocabulary, on both sides of the test, and
            // the message quotes the RESOLVED numbers.
            let want = limits.resolve(logprobs);
            let max = limits.max_logprobs_resolved();
            if logprobs != 0 && want > max {
                return Err(ApiError::invalid_param(
                    format!(
                        "Requested sample logprobs of {want}, which is greater than \
                         max allowed: {max}"
                    ),
                    "logprobs",
                    want,
                ));
            }
        }
        if let Some(prompt_logprobs) = self.prompt_logprobs {
            if prompt_logprobs < -1 {
                return Err(ApiError::invalid_param(
                    format!("prompt_logprobs must be non-negative or -1, got {prompt_logprobs}."),
                    "prompt_logprobs",
                    prompt_logprobs,
                ));
            }
            let want = limits.resolve(prompt_logprobs);
            let max = limits.max_prompt_logprobs_resolved();
            if prompt_logprobs != 0 && want > max {
                return Err(ApiError::invalid_param(
                    format!(
                        "Requested prompt logprobs of {want}, which is greater \
                         than max allowed: {max}"
                    ),
                    "prompt_logprobs",
                    want,
                ));
            }
        }
        if self.stop.iter().any(String::is_empty) {
            return Err(ApiError::bad_request(
                "stop cannot contain an empty string.",
                None,
            ));
        }
        // AFTER `_verify_args`, and only then: `__post_init__` collapses a greedy
        // request's truncations and calls `_verify_greedy_sampling`, which is the only
        // place `n > 1` is refused. A request with `temperature: 0, n: 2` is a 400.
        if f64::from(self.temperature) < crate::sampler::SAMPLING_EPS && self.n > 1 {
            return Err(ApiError::bad_request(
                format!("n must be 1 when using greedy sampling, got {}.", self.n),
                None,
            ));
        }
        Ok(())
    }

    /// The reference server's context check, message included, once the prompt length is known.
    pub fn check_context(
        &self,
        prompt_tokens: usize,
        limits: SamplingLimits,
    ) -> Result<(), ApiError> {
        let completion = self.max_tokens.unwrap_or(0);
        if prompt_tokens + completion > limits.max_model_len {
            return Err(ApiError::bad_request(
                format!(
                    "This model's maximum context length is {} tokens. However, you \
                     requested {} tokens ({} in the messages, {} in the completion). \
                     Please reduce the length of the messages or completion.",
                    limits.max_model_len,
                    prompt_tokens + completion,
                    prompt_tokens,
                    completion
                ),
                None,
            ));
        }
        Ok(())
    }

    /// The effective generation budget: the request's `max_tokens`, or what is left
    /// of the context window.
    pub fn effective_max_tokens(&self, prompt_tokens: usize, limits: SamplingLimits) -> usize {
        self.max_tokens
            .unwrap_or_else(|| limits.max_model_len.saturating_sub(prompt_tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejections_match_prod_messages() {
        let limits = SamplingLimits::default();
        let mut p = SamplingParams {
            temperature: -1.0,
            ..Default::default()
        };
        let err = p.validate(limits).expect_err("negative temperature");
        assert_eq!(
            err.body.message,
            "temperature must be non-negative, got -1.0. (parameter=temperature, value=-1.0)"
        );
        assert_eq!(err.body.param.as_deref(), Some("temperature"));
        assert_eq!(err.body.code, 400);

        p = SamplingParams {
            top_p: 3.0,
            ..Default::default()
        };
        assert!(p
            .validate(limits)
            .expect_err("top_p")
            .body
            .message
            .starts_with("top_p must be in (0, 1], got 3.0."));

        p = SamplingParams {
            logprobs: Some(99),
            ..Default::default()
        };
        assert_eq!(
            p.validate(limits).expect_err("logprobs").body.message,
            "Requested sample logprobs of 99, which is greater than max allowed: 20 \
             (parameter=logprobs, value=99)"
        );
    }

    /// `prompt_logprobs` is capped at the seam's 64 by default (QualityBench's
    /// teacher-forced top-64), above it with the reference server's own 400; the sampled token's
    /// `logprobs` keeps the reference server's 20; `--max-logprobs` above 64 lifts both.
    #[test]
    fn prompt_logprobs_cap_is_the_seam_and_sample_cap_stays_twenty() {
        let limits = SamplingLimits::default();
        assert_eq!(limits.max_prompt_logprobs_resolved(), 64);
        for ok in [1, 20, 21, 64] {
            SamplingParams {
                prompt_logprobs: Some(ok),
                ..Default::default()
            }
            .validate(limits)
            .unwrap_or_else(|e| panic!("prompt_logprobs {ok}: {}", e.body.message));
        }
        let err = SamplingParams {
            prompt_logprobs: Some(65),
            ..Default::default()
        }
        .validate(limits)
        .expect_err("65 is over the seam");
        assert_eq!(
            err.body.message,
            "Requested prompt logprobs of 65, which is greater than max allowed: 64 \
             (parameter=prompt_logprobs, value=65)"
        );
        assert_eq!(err.body.param.as_deref(), Some("prompt_logprobs"));
        assert_eq!(err.body.code, 400);
        let err = SamplingParams {
            prompt_logprobs: Some(-1),
            ..Default::default()
        }
        .validate(limits)
        .expect_err("-1 is the whole vocabulary");
        assert!(err.body.message.starts_with(
            "Requested prompt logprobs of 248320, which is greater than max allowed: 64"
        ));
        // The sampled token's cap is untouched.
        let err = SamplingParams {
            logprobs: Some(21),
            ..Default::default()
        }
        .validate(limits)
        .expect_err("sample cap 20");
        assert!(err
            .body
            .message
            .starts_with("Requested sample logprobs of 21, which is greater than max allowed: 20"));
        // `--max-logprobs` above 64 lifts the prompt cap with it; -1 is the vocabulary.
        let wide = SamplingLimits {
            max_logprobs: 100,
            ..Default::default()
        };
        assert_eq!(wide.max_prompt_logprobs_resolved(), 100);
        let all = SamplingLimits {
            max_logprobs: -1,
            ..Default::default()
        };
        assert_eq!(all.max_prompt_logprobs_resolved(), 248_320);
    }

    #[test]
    fn defaults_are_accepted_and_context_is_checked() {
        let limits = SamplingLimits {
            max_model_len: 32,
            ..Default::default()
        };
        let p = SamplingParams {
            max_tokens: Some(8),
            ..Default::default()
        };
        p.validate(limits).expect("defaults valid");
        p.check_context(24, limits).expect("fits exactly");
        let err = p.check_context(25, limits).expect_err("one over");
        assert!(err
            .body
            .message
            .contains("maximum context length is 32 tokens"));
        assert!(err
            .body
            .message
            .contains("you requested 33 tokens (25 in the messages, 8 in the completion)"));
    }
}
