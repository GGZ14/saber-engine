//! Canonical hashing.
//!
//! Every hash in this crate — the plan hash, the plan *input* hash, the registry fingerprint, the
//! autotune key digest — is the sha256 of the same canonical serialization: `serde_json` over types
//! whose every map is a `BTreeMap` and whose every list is in declaration order. That makes the
//! hash a pure function of the value, stable across runs and across processes, which is what lets
//! it key a compiled-artifact cache.

use serde::Serialize;
use sha2::{Digest, Sha256};

/// The sha256 of a byte string, lower-case hex.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

/// The canonical JSON of a value: the exact bytes every digest in this crate is taken over.
///
/// # Panics
///
/// Only if a type in this crate grows a map with non-string keys or a non-finite float, neither of
/// which serialize to JSON. The plan types are all `BTreeMap<String, _>` and finite `f64`.
pub fn canonical_json<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).expect("saber-plan types serialize to JSON infallibly")
}

/// The sha256 of a value's canonical JSON, lower-case hex.
pub fn digest_of<T: Serialize>(value: &T) -> String {
    sha256_hex(canonical_json(value).as_bytes())
}
