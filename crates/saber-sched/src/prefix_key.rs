//! The KV block key of a prompt with images (D4; FN125 builds Flash-Next's prefix tree and prompt
//! cache on it).
//!
//! A key over token ids alone cannot name a block that holds image tokens: a block of
//! `<|image_pad|>` ids is the same ids for every image, and the reference server folds the image's content hash
//! into the hash of every block its tokens touch (`mm_hashes` in `hash_block_tokens`' extra keys).
//! Here the same:
//!
//! `key(i) = H(key(i - 1), ids of block i, hash of every image whose run overlaps block i)` —
//! chained, so a block's key names its whole prefix, and a text block AFTER an image carries the
//! image through the chain (its K/V attended to the image). A block no image run overlaps is
//! keyed by its ids alone: a text prompt's keys are the text law's.
//!
//! The overlap rule is ONE function, [`ImageKey::overlaps`]: these block keys and Flash-Next's
//! piece keys (`saber-cli`'s `flashnext::worker::pcache::piece_key`, FN125: the prefix tree's and the
//! prompt cache's chained sha256 over a PIECE of the prompt -- a run of blocks; an image run that
//! straddles pieces carries its hash into every piece it overlaps) both call it.

use saber_serve::engine::TokenId;

/// One image's run in a prompt and its content hash (`ImageBatch::content_hashes`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageKey {
    /// First prompt index of the run.
    pub start: u32,
    /// Tokens in the run.
    pub tokens: u32,
    /// The image's content hash.
    pub hash: u64,
}

impl ImageKey {
    /// Whether the image's run overlaps the prompt positions `lo..hi` (half-open; an empty range
    /// overlaps nothing).
    #[must_use]
    pub fn overlaps(&self, lo: u32, hi: u32) -> bool {
        lo < hi && self.start < hi && self.start + self.tokens > lo
    }
}

const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const PRIME: u64 = 0x0000_0100_0000_01b3;

fn mix(h: u64, v: u64) -> u64 {
    let mut h = h;
    for b in v.to_le_bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// The chained key of every FULL block of `prompt` (a partial last block is never shareable).
#[must_use]
pub fn prefix_block_keys(prompt: &[TokenId], block: u32, images: &[ImageKey]) -> Vec<u64> {
    let block = block.max(1) as usize;
    let mut keys = Vec::with_capacity(prompt.len() / block);
    let mut prev = OFFSET;
    for (i, ids) in prompt.chunks_exact(block).enumerate() {
        let (lo, hi) = ((i * block) as u32, ((i + 1) * block) as u32);
        let mut h = mix(OFFSET, prev);
        for &t in ids {
            h = mix(h, u64::from(t));
        }
        for im in images {
            if im.overlaps(lo, hi) {
                h = mix(mix(h, 0x1a6e), im.hash);
            }
        }
        keys.push(h);
        prev = h;
    }
    keys
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAD: TokenId = 248_056;

    /// 20 text ids, a 40-token image run, 12 text ids: blocks of 16.
    fn prompt() -> Vec<TokenId> {
        let mut v: Vec<TokenId> = (1..=20).collect();
        v.extend(std::iter::repeat_n(PAD, 40));
        v.extend(100..112);
        v
    }

    fn image(hash: u64) -> [ImageKey; 1] {
        [ImageKey {
            start: 20,
            tokens: 40,
            hash,
        }]
    }

    #[test]
    fn the_same_image_shares_every_block_and_a_different_one_only_the_text_before_it() {
        let p = prompt();
        let a = prefix_block_keys(&p, 16, &image(0xAAAA));
        let same = prefix_block_keys(&p, 16, &image(0xAAAA));
        let other = prefix_block_keys(&p, 16, &image(0xBBBB));
        assert_eq!(a.len(), 4);
        assert_eq!(a, same, "the same image: every block shared");
        // Block 0 (ids 0..16) is text before the image: shared. Block 1 (16..32) holds the image's
        // first tokens, and every later block chains through it: none shared.
        assert_eq!(a[0], other[0]);
        for i in 1..4 {
            assert_ne!(a[i], other[i], "block {i} of a different image");
        }
    }

    /// The control: the key WITHOUT the image hash is exactly the collision this module exists
    /// to prevent -- two different images, identical keys.
    #[test]
    fn a_key_over_token_ids_alone_collides_two_images() {
        let p = prompt();
        let a = prefix_block_keys(&p, 16, &[]);
        let b = prefix_block_keys(&p, 16, &[]);
        assert_eq!(a, b);
        assert_ne!(a, prefix_block_keys(&p, 16, &image(0xAAAA)));
    }

    /// FN125: the one overlap rule -- half-open on both sides, an empty range overlaps nothing.
    #[test]
    fn the_overlap_rule_is_half_open() {
        let im = ImageKey {
            start: 20,
            tokens: 40,
            hash: 1,
        };
        assert!(im.overlaps(0, 21) && im.overlaps(59, 100) && im.overlaps(30, 31));
        assert!(!im.overlaps(0, 20) && !im.overlaps(60, 100) && !im.overlaps(30, 30));
    }

    /// End to end over the plane's own hash: one pixel differs, the image blocks' keys differ.
    #[test]
    fn one_pixel_moves_the_content_hash_and_the_image_blocks_keys() {
        use saber_serve::engine::ImageBatch;
        let a = ImageBatch {
            grids: vec![[1, 4, 10]],
            pixel_values: vec![0.25; 40 * 1536],
        };
        let mut b = a.clone();
        b.pixel_values[1234] = 0.5;
        let (ha, hb) = (a.content_hashes()[0], b.content_hashes()[0]);
        assert_eq!(ha, a.clone().content_hashes()[0]);
        assert_ne!(ha, hb);
        let p = prompt();
        let ka = prefix_block_keys(&p, 16, &image(ha));
        let kb = prefix_block_keys(&p, 16, &image(hb));
        assert_eq!(ka[0], kb[0]);
        assert_ne!(ka[1], kb[1]);
    }
}
