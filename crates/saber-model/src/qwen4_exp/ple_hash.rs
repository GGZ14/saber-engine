//! The PLE n-gram hash on the HOST (workstream FN6): token ids -> the 16 table rows per token.
//!
//! Upstream's `Qwen4ExpNGramEmbedding.forward` (`qwen4_exp/amd/ple_layer.py`), written down:
//!
//! * the step's tokens are laid after the sequence's previous TWO tokens (`ngram_context`,
//!   eos-padded at the start of a sequence): `ctx = [c0, c1, t0, t1, ...]`;
//! * `seg(i) = i - prev_eos(i) - 1` where `prev_eos(i)` is the last index `< i` holding
//!   `eos_token_id` (else -1): the n-gram resets after an eos;
//! * `shift_s(i) = ctx[i - s]` if `i - s >= 0` and `seg(i) >= s`, else `eos_token_id` (s = 1, 2;
//!   `shift_0 = ctx[i]`);
//! * for n = 2, 3: `mixed = shift_0 * m0 ^ shift_1 * m1 [^ shift_2 * m2]` in WRAPPING int64 (torch
//!   long multiply), and heads `h in [(n-2) * 8, (n-1) * 8)` take
//!   `id_h = remainder(mixed, size_h) + offset_h` (torch's floored remainder: never negative for a
//!   positive size).
//!
//! The multipliers, sizes and offsets are the CHECKPOINT's stored buffers
//! (`ple_embedding.{layer_multipliers, ngram_heads_vocab_sizes, ngram_heads_offsets}`, referenced
//! by the pack's `shared/dense.json` raw records) -- [`PleHash::derive`] is upstream's derivation
//! (splitmix64 multipliers, the next primes above `ngram_vocab_size_base`), kept to CHECK the
//! stored buffers, never to replace them.
//!
//! Only the previous two tokens matter, so a sequence's hash state across chunks and decode steps
//! is [`PleContext`] (its last two tokens), and ids are identical however the sequence is cut.

use super::config::{nth_prime_after, PleConfig};

/// splitmix64's constants and the per-PLE-layer seed prime (upstream's class attributes).
const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const M1: u64 = 0xBF58_476D_1CE4_E5B9;
const M2: u64 = 0x94D0_49BB_1331_11EB;
const LAYER_PRIME: u64 = 10_007;

fn splitmix64(v: u64) -> u64 {
    let mut v = v.wrapping_add(GAMMA);
    v = (v ^ (v >> 30)).wrapping_mul(M1);
    v = (v ^ (v >> 27)).wrapping_mul(M2);
    v ^ (v >> 31)
}

/// The hash tables of one PLE layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PleHash {
    /// `layer_multipliers`, `ngram_size` entries (3).
    pub multipliers: Vec<i64>,
    /// `ngram_heads_vocab_sizes`, 16 entries.
    pub sizes: Vec<i64>,
    /// `ngram_heads_offsets`, 16 entries.
    pub offsets: Vec<i64>,
    /// `heads_per_ngram` (8).
    pub heads_per_ngram: usize,
    /// The segment token (`eos_token_id`, 248044).
    pub eos: u32,
}

/// A sequence's hash state: its last two tokens (eos at the start of a sequence).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PleContext(pub [u32; 2]);

impl PleContext {
    /// The state of a new sequence.
    #[must_use]
    pub const fn start(eos: u32) -> Self {
        Self([eos, eos])
    }
    /// Advance past `tokens` (what the next chunk / step needs).
    pub fn advance(&mut self, tokens: &[u32]) {
        for &t in tokens {
            self.0 = [self.0[1], t];
        }
    }
}

impl PleHash {
    /// From the checkpoint's stored buffers (little-endian i64 bytes, as the safetensors hold
    /// them). Refuses a length that does not match the config.
    ///
    /// # Errors
    /// On a buffer length off the config's `ngram_size` / `ngram_heads`, or a non-positive size.
    pub fn from_buffers(
        cfg: &PleConfig,
        eos: u32,
        multipliers: &[u8],
        sizes: &[u8],
        offsets: &[u8],
    ) -> Result<Self, String> {
        let rd = |b: &[u8], n: usize, what: &str| -> Result<Vec<i64>, String> {
            if b.len() != n * 8 {
                return Err(format!(
                    "ple hash: {what} has {} bytes, want {}",
                    b.len(),
                    n * 8
                ));
            }
            Ok(b.chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().expect("8 bytes")))
                .collect())
        };
        if !(2..=3).contains(&cfg.ngram_size) {
            return Err(format!(
                "ple hash: ngram_size {} (the host hash holds the 2 previous tokens: 2..=3)",
                cfg.ngram_size
            ));
        }
        let heads = cfg.ngram_heads() as usize;
        let h = Self {
            multipliers: rd(multipliers, cfg.ngram_size as usize, "layer_multipliers")?,
            sizes: rd(sizes, heads, "ngram_heads_vocab_sizes")?,
            offsets: rd(offsets, heads, "ngram_heads_offsets")?,
            heads_per_ngram: cfg.heads_per_ngram as usize,
            eos,
        };
        if h.sizes.iter().any(|&s| s <= 0) {
            return Err("ple hash: a non-positive head size".into());
        }
        Ok(h)
    }

    /// Upstream's DERIVATION (`_make_layer_multipliers`, `_make_vocab_layout`) for PLE layer
    /// `dense_id` (the index among the PLE layers, 0 here). Used to check the stored buffers.
    #[must_use]
    pub fn derive(cfg: &PleConfig, vocab_size: u32, eos: u32, dense_id: u64) -> Self {
        let max_mult = (i64::MAX as u64) / u64::from(vocab_size);
        let half = (max_mult / 2).max(1);
        let base = cfg
            .hash_seed
            .wrapping_add(LAYER_PRIME.wrapping_mul(dense_id));
        let multipliers = (0..u64::from(cfg.ngram_size))
            .map(|i| {
                let v = base.wrapping_add(GAMMA.wrapping_mul(i + 1));
                (2 * (splitmix64(v) % half) + 1) as i64
            })
            .collect();
        let heads = u64::from(cfg.ngram_heads());
        let mut sizes = Vec::new();
        let mut offsets = Vec::new();
        let mut off = 0i64;
        for local in 0..heads {
            let global = dense_id * heads + local;
            let size = nth_prime_after(cfg.ngram_vocab_size_base - 1, global + 1) as i64;
            sizes.push(size);
            offsets.push(off);
            off += size;
        }
        Self {
            multipliers,
            sizes,
            offsets,
            heads_per_ngram: cfg.heads_per_ngram as usize,
            eos,
        }
    }

    /// Rows per token (16).
    #[must_use]
    pub fn heads(&self) -> usize {
        self.sizes.len()
    }

    /// The table rows of `tokens`, laid after `ctx` (the sequence's previous two tokens), as a
    /// flat `[tokens.len() * 16]` in head order -- column h of FN3's `ple_ids`.
    #[must_use]
    pub fn ids(&self, ctx: PleContext, tokens: &[u32]) -> Vec<u64> {
        let heads = self.heads();
        let mut out = Vec::with_capacity(tokens.len() * heads);
        let eos = self.eos;
        // the two previous tokens and how many non-eos tokens directly precede each position
        let mut prev = ctx.0;
        // seg of the first new token: count back over the context (index 0, 1 of the array)
        // seg(i) >= s  <=>  the s tokens before i exist and none of them is eos.
        for &t in tokens {
            let ok1 = prev[1] != eos;
            let ok2 = ok1 && prev[0] != eos;
            let sh = [
                t,
                if ok1 { prev[1] } else { eos },
                if ok2 { prev[0] } else { eos },
            ];
            for n in 2..=self.multipliers.len() {
                let mut mixed = i64::from(sh[0]).wrapping_mul(self.multipliers[0]);
                for (k, &m) in self.multipliers.iter().enumerate().take(n).skip(1) {
                    mixed ^= i64::from(sh[k]).wrapping_mul(m);
                }
                let h0 = (n - 2) * self.heads_per_ngram;
                for h in h0..h0 + self.heads_per_ngram {
                    out.push((mixed.rem_euclid(self.sizes[h]) + self.offsets[h]) as u64);
                }
            }
            prev = [prev[1], t];
        }
        out
    }

    /// [`Self::ids`] and the context advanced past `tokens` (a chunk or a decode step).
    pub fn ids_step(&self, ctx: &mut PleContext, tokens: &[u32]) -> Vec<u64> {
        let v = self.ids(*ctx, tokens);
        ctx.advance(tokens);
        v
    }

    /// Upstream's own formulation, transcribed literally (the eos-segmented shift over the
    /// concatenated `[ctx | tokens]` array with `cummax`), kept as the independent reference the
    /// host tests hold [`Self::ids`] against.
    #[must_use]
    pub fn ids_reference(&self, ctx: PleContext, tokens: &[u32]) -> Vec<u64> {
        let eos = self.eos;
        let arr: Vec<i64> = ctx.0.iter().chain(tokens).map(|&t| i64::from(t)).collect();
        let n = arr.len();
        let mut seg = vec![0i64; n];
        let mut last_eos = -1i64; // cummax of eos positions, EXCLUSIVE of i
        for (i, s) in seg.iter_mut().enumerate() {
            *s = i as i64 - last_eos - 1;
            if arr[i] == i64::from(eos) {
                last_eos = i as i64;
            }
        }
        let shifted = |s: usize, i: usize| -> i64 {
            if s == 0 {
                arr[i]
            } else if i >= s && seg[i] >= s as i64 {
                arr[i - s]
            } else {
                i64::from(eos)
            }
        };
        let mut out = Vec::new();
        for i in 2..n {
            for ng in 2..=self.multipliers.len() {
                let mut mixed = shifted(0, i).wrapping_mul(self.multipliers[0]);
                for idx in 1..ng {
                    mixed ^= shifted(idx, i).wrapping_mul(self.multipliers[idx]);
                }
                let h0 = (ng - 2) * self.heads_per_ngram;
                for h in h0..h0 + self.heads_per_ngram {
                    out.push((mixed.rem_euclid(self.sizes[h]) + self.offsets[h]) as u64);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> PleConfig {
        PleConfig {
            layer_ids_one_based: vec![2],
            embed_dim: 2560,
            conv_kernel: 4,
            ngram_size: 3,
            heads_per_ngram: 8,
            ngram_vocab_size_base: 20_000_000,
            vocab_divisible_by: 128,
            split_parts: 128,
            hash_seed: 1234,
        }
    }

    #[test]
    fn the_fast_path_is_upstreams_formulation() {
        let h = PleHash::derive(&cfg(), 248_320, 248_044, 0);
        assert_eq!(h.heads(), 16);
        let mut seed = 0x1234_5678_u64;
        let mut rnd = || {
            seed = splitmix64(seed);
            seed
        };
        for _ in 0..200 {
            let n = (rnd() % 40) as usize + 1;
            // eos often, so the segment law is exercised
            let toks: Vec<u32> = (0..n)
                .map(|_| {
                    if rnd() % 5 == 0 {
                        248_044
                    } else {
                        (rnd() % 248_320) as u32
                    }
                })
                .collect();
            let ctx = PleContext([
                if rnd() % 3 == 0 { 248_044 } else { 7 },
                if rnd() % 3 == 0 { 248_044 } else { 9 },
            ]);
            assert_eq!(h.ids(ctx, &toks), h.ids_reference(ctx, &toks));
        }
    }

    /// The previous-2-tokens law: any cut of a sequence gives the ids of the whole; a context NOT
    /// carried across the cut (the negative control) differs.
    #[test]
    fn chunking_is_invisible_and_a_dropped_context_is_not() {
        let h = PleHash::derive(&cfg(), 248_320, 248_044, 0);
        let toks: Vec<u32> = (0..64u32).map(|i| (i * 7919 + 13) % 248_000).collect();
        let whole = h.ids(PleContext::start(248_044), &toks);
        for cut in 1..toks.len() {
            let mut c = PleContext::start(248_044);
            let mut v = h.ids_step(&mut c, &toks[..cut]);
            v.extend(h.ids_step(&mut c, &toks[cut..]));
            assert_eq!(v, whole, "cut at {cut}");
            let mut bad = h.ids(PleContext::start(248_044), &toks[..cut]);
            bad.extend(h.ids(PleContext::start(248_044), &toks[cut..]));
            assert_ne!(bad, whole, "control: a fresh context at {cut} must differ");
        }
        // one token at a time (decode)
        let mut c = PleContext::start(248_044);
        let v: Vec<u64> = toks
            .iter()
            .flat_map(|t| h.ids_step(&mut c, &[*t]))
            .collect();
        assert_eq!(v, whole);
    }

    #[test]
    fn ids_stay_in_their_heads_ranges() {
        let h = PleHash::derive(&cfg(), 248_320, 248_044, 0);
        let rows: i64 = h.sizes.iter().sum();
        // the pack's table is these rows padded to a multiple of 128 (320,001,536)
        assert!(rows <= 320_001_536 && 320_001_536 - rows < 128, "{rows}");
        let toks: Vec<u32> = (0..500u32).map(|i| i * 491 % 248_320).collect();
        let ids = h.ids(PleContext::start(248_044), &toks);
        for (j, id) in ids.iter().enumerate() {
            let hd = j % 16;
            let id = *id as i64;
            assert!(id >= h.offsets[hd] && id < h.offsets[hd] + h.sizes[hd]);
        }
    }
}
