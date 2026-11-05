//! Request and tool-call identifiers, in the shapes OpenAI clients expect and the reference server
//! emits: `chatcmpl-<16 hex>` for a response, `chatcmpl-tool-<32 hex>` for a tool call.
//!
//! No `rand` dependency: ids are cosmetic, not cryptographic, and a seeded
//! xorshift keeps the serve path free of another crate.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static STATE: AtomicU64 = AtomicU64::new(0);

fn next_u64() -> u64 {
    let mut x = STATE.load(Ordering::Relaxed);
    if x == 0 {
        // Seed once, from the clock and the address of a stack slot.
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15);
        let local = 0u8;
        x = nanos ^ (&local as *const u8 as u64).rotate_left(17) ^ 0x2545_F491_4F6C_DD1D;
        if x == 0 {
            x = 0x9E37_79B9_7F4A_7C15;
        }
    }
    // xorshift64*
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    STATE.store(x, Ordering::Relaxed);
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

fn hex(bytes: usize) -> String {
    let mut out = String::with_capacity(bytes * 2);
    let mut left = bytes;
    while left > 0 {
        let chunk = next_u64();
        for i in 0..8.min(left) {
            let byte = (chunk >> (i * 8)) as u8;
            out.push_str(&format!("{byte:02x}"));
        }
        left = left.saturating_sub(8);
    }
    out
}

/// 32 hex characters — the shape of `uuid4().hex`, which the reference server uses for tool-call ids.
pub fn random_hex32() -> String {
    hex(16)
}

/// 16 hex characters — the shape of the reference server's `chatcmpl-…` request ids.
pub fn random_hex16() -> String {
    hex(8)
}

/// `chatcmpl-…`, the id a chat completion response carries.
pub fn chat_completion_id() -> String {
    format!("chatcmpl-{}", random_hex16())
}

/// `cmpl-…`, the id a legacy completion response carries.
pub fn completion_id() -> String {
    format!("cmpl-{}", random_hex16())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_have_the_shapes_clients_parse() {
        let tool = random_hex32();
        assert_eq!(tool.len(), 32);
        assert!(tool.chars().all(|c| c.is_ascii_hexdigit()));
        assert!(chat_completion_id().starts_with("chatcmpl-"));
        assert_ne!(random_hex32(), random_hex32());
    }
}
