//! Errors from parsing, verifying and streaming a pack.
//!
//! Every variant names what was wrong and what the reader expected: a pack fault is a load error
//! that a human has to act on offline (rebuild the pack, re-resolve the plan), never something the
//! engine can work around at runtime (I3).

#![forbid(unsafe_code)]

use std::path::PathBuf;

use thiserror::Error;

/// A pack could not be parsed, verified or streamed.
#[derive(Debug, Error)]
pub enum PackError {
    /// The file could not be read.
    #[error("{path}: {context}: {source}")]
    Io {
        /// File the I/O was issued against.
        path: PathBuf,
        /// What the reader was doing.
        context: String,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },

    /// The first eight bytes are not `SBRPACK1`.
    #[error("not a saber pack: magic is {found:02x?}, expected \"SBRPACK1\"")]
    BadMagic {
        /// The bytes that were there.
        found: Vec<u8>,
    },

    /// Format version this build cannot read.
    #[error("pack format version {found}, this build reads {expected}")]
    Version {
        /// Version on disk.
        found: u32,
        /// Version this build understands.
        expected: u32,
    },

    /// Header or record entry size this build cannot read.
    #[error("pack declares {what}_bytes {found}, this build reads {expected}")]
    StructSize {
        /// `"header"` or `"record"`.
        what: &'static str,
        /// Size on disk.
        found: u32,
        /// Size this build understands.
        expected: u32,
    },

    /// A structural rule from `docs/abi/PACK.md` was violated.
    #[error("malformed pack: {0}")]
    Malformed(String),

    /// A record's data offset is not 4096-aligned, so it cannot be read with `O_DIRECT`.
    #[error("record {index} ({name}): data_off {off} is not {align}-aligned")]
    UnalignedRecord {
        /// Table index.
        index: usize,
        /// Tensor name.
        name: String,
        /// The offending offset.
        off: u64,
        /// Required alignment.
        align: u64,
    },

    /// Dtype tag not in the ABI table.
    #[error("unknown dtype tag {0}")]
    UnknownDtype(u16),

    /// Shard-kind tag not in the ABI table.
    #[error("unknown shard kind tag {0}")]
    UnknownShardKind(u8),

    /// Layout id not in the ABI table (`docs/abi/PACK.md` § Layout ids).
    ///
    /// A loader that met a record it cannot interpret says so rather than guessing: an id it does
    /// not know describes bytes it does not know how to read.
    #[error("unknown layout id {0}; this build reads the ids in docs/abi/PACK.md")]
    UnknownLayout(u32),

    /// A record's layout is not the one the caller is about to read it as.
    #[error("record `{record}` is {found}, but this use needs {expected}")]
    LayoutMismatch {
        /// The record's name.
        record: String,
        /// What the pack says it is.
        found: String,
        /// What the caller needs it to be.
        expected: &'static str,
    },

    /// The record table's own hash does not match the header.
    #[error("record table hash mismatch: header says {expected}, table hashes to {found}")]
    TableHashMismatch {
        /// Hex digest from the header.
        expected: String,
        /// Hex digest of the bytes on disk.
        found: String,
    },

    /// A record's content hash does not match. The bytes landed wrong; the pack is refused.
    #[error("record {index} ({name}): content hash mismatch, table says {expected}, bytes hash to {found}")]
    RecordHashMismatch {
        /// Table index.
        index: usize,
        /// Tensor name.
        name: String,
        /// Hex digest from the table.
        expected: String,
        /// Hex digest of the bytes read.
        found: String,
    },

    /// One chunk of a record does not match the chunk digest the pack carries for it.
    ///
    /// The same fault as [`PackError::RecordHashMismatch`], found by the chunk-digest path — which
    /// says *where* in a 1.18 GiB record the bytes went wrong instead of only that they did.
    #[error(
        "record {index} ({name}): chunk {chunk} of {chunks} (bytes {lo}..{hi}) does not match the \
         pack's chunk digest {expected}, it hashes to {found}"
    )]
    RecordChunkMismatch {
        /// Table index.
        index: usize,
        /// Tensor name.
        name: String,
        /// Chunk index within the record.
        chunk: usize,
        /// Chunks the record has.
        chunks: usize,
        /// First payload byte of the chunk.
        lo: u64,
        /// One past its last payload byte.
        hi: u64,
        /// Hex digest from the chunk blob.
        expected: String,
        /// Hex digest of the bytes read.
        found: String,
    },

    /// The pack was built for a different plan. It is refused, never re-derived.
    #[error(
        "plan hash mismatch: pack was built for {found}, the resolved plan is {expected}; \
             a pack is refused, never re-derived — rebuild it offline with tools/pack"
    )]
    PlanHashMismatch {
        /// Hex digest in the pack header.
        found: String,
        /// Hex digest of the resolved plan.
        expected: String,
    },

    /// A stream configuration that cannot work.
    #[error("invalid stream configuration: {0}")]
    Config(String),
}

impl PackError {
    /// Attach a path and a context string to an I/O error.
    pub fn io(path: &std::path::Path, context: impl Into<String>, source: std::io::Error) -> Self {
        PackError::Io {
            path: path.to_path_buf(),
            context: context.into(),
            source,
        }
    }
}
