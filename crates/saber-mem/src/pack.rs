//! Tier-pack reader and streaming loader.
//!
//! A pack is the on-disk form of a **model tier** (`docs/ARCHITECTURE.md` § "The model tier"): the
//! release unit that binds a checkpoint, its quantization, its drafter, its template and its tuned
//! serving config under one content hash, written in device-final layout so that loading is one
//! sequential `O_DIRECT` stream with **zero transforms on the load path**.
//!
//! The byte layout is specified in `docs/abi/PACK.md` and mirrored in `include/saber_abi.h`. The
//! rules this module enforces, in one list:
//!
//! * magic `SBRPACK1`, format version 1, 256-byte header, 160-byte record entries;
//! * every record 4096-aligned, the file length a multiple of 4096 (so any record is directly
//!   readable);
//! * the table sorted by `data_off`, records non-overlapping after padding;
//! * `length` exactly what `shape` and `dtype` imply;
//! * per-record `sha256` checked after landing — a bad record is a load error, not a garbage weight;
//! * `plan_hash` must equal the resolved plan's hash. **A pack that does not match is refused, never
//!   re-derived.**
//!
//! ```no_run
//! use saber_mem::pack::{
//!     Borrowed, DeliveryOrder, PackStream, PageAlignedPool, RecordRef, StreamConfig,
//! };
//! # fn main() -> Result<(), saber_mem::pack::PackError> {
//! let stream = PackStream::with_config(
//!     std::path::Path::new("/mnt/nvme/saber-packs/qwen38-27b-nvfp4/rank0.pack"),
//!     StreamConfig { queue_depth: 8, order: DeliveryOrder::Completion, ..Default::default() },
//! )?;
//! // Where the buffers come from is a policy: this one reads into pageable memory. The upload
//! // path passes its pinned staging ring instead and the drive lands records where the H2D
//! // will read them.
//! let pool = PageAlignedPool::new(2 << 30);
//! let stats = stream.run(&pool, &mut Borrowed(|rec: RecordRef<'_>| {
//!     // hash-verified bytes in a 4096-aligned host buffer
//!     let _ = (rec.name, rec.bytes.len());
//!     Ok(())
//! }))?;
//! println!("{:.2} GB/s", stats.gigabytes_per_second());
//! # Ok(()) }
//! ```

#![forbid(unsafe_code)]

pub mod buffers;
mod error;
mod format;
pub mod layout;
mod reader;
mod stream;
pub mod writer;

pub use buffers::{BufferPool, PageAlignedPool, PinnedView, PoolBuf, PoolStats};
pub use error::PackError;
pub use format::{
    align_up, ChunkBlob, Dtype, PackHeader, PackRecord, ShardKind, ALIGN, CHUNK_ENTRY_HEADER,
    FLAGS_KNOWN, FLAG_CHUNK_DIGESTS, FORMAT_VERSION, FORMAT_VERSION_MIN, HEADER_BYTES, LAYOUT_RAW,
    MAGIC, MAX_NDIM, RECORD_BYTES,
};
pub use layout::{check as check_layout, LayoutId};
pub use reader::{metadata_end, PackReader};
pub use stream::{
    Borrowed, DeliveryOrder, PackStream, RecordOutcome, RecordRef, RecordView, StreamConfig,
    StreamConsumer, StreamStats,
};

/// Canonical location of built packs on this box: `<PACK_ROOT>/<tier>/rank<r>.pack`.
///
/// A constant rather than an environment variable: behaviour comes from the plan, never from
/// `getenv` (CLAUDE.md rule 4). The planner will carry the real path.
pub const PACK_ROOT: &str = "/mnt/nvme/saber-packs";

#[cfg(test)]
mod tests;
