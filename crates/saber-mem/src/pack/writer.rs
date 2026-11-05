//! A minimal pack writer.
//!
//! The tier builder is `the internal reference harness` — it runs in the prod image, reads safetensors and
//! shards for TP. This writer exists so that Rust tests can build a pack (including a deliberately
//! broken one) without a container, and so the encode side of the format has a second
//! implementation that the reader is tested against.

#![forbid(unsafe_code)]

use std::fs::File;
use std::io::{self, Seek, SeekFrom, Write};
use std::path::Path;

use sha2::{Digest, Sha256};

use super::format::{
    align_up, ChunkBlob, Dtype, PackHeader, PackRecord, ShardKind, ALIGN, FLAG_CHUNK_DIGESTS,
    FORMAT_VERSION, HEADER_BYTES, LAYOUT_RAW,
};

/// One tensor to write.
pub struct PackItem {
    /// Tensor name (UTF-8).
    pub name: String,
    /// Element type.
    pub dtype: Dtype,
    /// Rank-local shape.
    pub shape: Vec<u64>,
    /// How it was sharded.
    pub shard_kind: ShardKind,
    /// Layout id; 0 = raw checkpoint bytes.
    pub layout_id: u32,
    /// Payload.
    pub data: Vec<u8>,
}

impl PackItem {
    /// A raw 1-D `u8` record, the shortest way to state a test fixture.
    #[must_use]
    pub fn raw_u8(name: &str, data: Vec<u8>) -> Self {
        Self {
            name: name.to_string(),
            dtype: Dtype::U8,
            shape: vec![data.len() as u64],
            shard_kind: ShardKind::Replicated,
            layout_id: LAYOUT_RAW,
            data,
        }
    }
}

/// Knobs that produce a pack the reader must reject. Tests only: the builder never sets these.
#[derive(Debug, Clone, Copy, Default)]
pub struct WriterFaults {
    /// Shift the first record's `data_off` by one byte, and rewrite the table hash over the bad
    /// table so the parser fails on the alignment rule and not on the hash.
    pub unaligned_first_record: bool,
    /// Flip one bit of the first chunk digest of the first flagged record, leaving the blob's own
    /// `sha256` correct over the tampered blob. The record's `sha256` still matches its bytes, so
    /// this is the case that asks whether a chunk-verified read can be fooled.
    pub tamper_chunk_digest: bool,
    /// Leave the blob's `sha256` as the digest of a *different* blob. The entries then parse and
    /// bind correctly and only the blob hash disagrees.
    pub wrong_chunk_blob_hash: bool,
    /// Write a record whose **two descriptions disagree**: the payload is flipped after the record
    /// `sha256` is taken, and the chunk list is computed over the flipped bytes. Everything is
    /// internally consistent except that the record digest describes bytes the file does not hold.
    ///
    /// This is the one fault the load path cannot see and the offline verifier must.
    pub chunks_describe_other_bytes: bool,
}

/// How a written pack carries chunk digests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkPolicy {
    /// No chunk blob; the pack is byte-for-byte a v1 pack apart from `format_version`.
    None,
    /// A digest per `bytes` for every record **longer** than `bytes`.
    ///
    /// Records that fit one chunk are not flagged: their chunk digest would be a second hash of
    /// the same bytes the record digest already covers, and one hash per record is the point.
    Over {
        /// Chunk size; must be a non-zero multiple of [`ALIGN`].
        bytes: u32,
    },
}

impl ChunkPolicy {
    /// The load-path default: 4 MiB, the same chunk the parking digest uses.
    #[must_use]
    pub fn default_chunking() -> Self {
        ChunkPolicy::Over { bytes: 4 << 20 }
    }

    fn bytes(self) -> u32 {
        match self {
            ChunkPolicy::None => 0,
            ChunkPolicy::Over { bytes } => bytes,
        }
    }
}

/// Write `items` as `path`. Returns the file length.
pub fn write_pack(
    path: &Path,
    tp_size: u32,
    rank: u32,
    plan_hash: [u8; 32],
    tier_id: [u8; 32],
    items: &[PackItem],
) -> io::Result<u64> {
    write_pack_chunked(
        path,
        tp_size,
        rank,
        plan_hash,
        tier_id,
        items,
        ChunkPolicy::None,
        WriterFaults::default(),
    )
}

/// As [`write_pack`], with chunk digests.
pub fn write_pack_chunks(
    path: &Path,
    tp_size: u32,
    rank: u32,
    plan_hash: [u8; 32],
    tier_id: [u8; 32],
    items: &[PackItem],
    chunks: ChunkPolicy,
) -> io::Result<u64> {
    write_pack_chunked(
        path,
        tp_size,
        rank,
        plan_hash,
        tier_id,
        items,
        chunks,
        WriterFaults::default(),
    )
}

/// As [`write_pack`], with deliberate faults for negative controls.
pub fn write_pack_with_faults(
    path: &Path,
    tp_size: u32,
    rank: u32,
    plan_hash: [u8; 32],
    tier_id: [u8; 32],
    items: &[PackItem],
    faults: WriterFaults,
) -> io::Result<u64> {
    write_pack_chunked(
        path,
        tp_size,
        rank,
        plan_hash,
        tier_id,
        items,
        ChunkPolicy::None,
        faults,
    )
}

/// The one writer the others call.
#[allow(clippy::too_many_arguments)] // seven of them are the pack's own identity and contents
pub fn write_pack_chunked(
    path: &Path,
    tp_size: u32,
    rank: u32,
    plan_hash: [u8; 32],
    tier_id: [u8; 32],
    items: &[PackItem],
    chunks: ChunkPolicy,
    faults: WriterFaults,
) -> io::Result<u64> {
    // Name blob: u16 LE length prefix + UTF-8 bytes, in table order.
    let mut blob = Vec::new();
    let mut name_offs = Vec::with_capacity(items.len());
    for it in items {
        let len = u16::try_from(it.name.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "name longer than 65535"))?;
        blob.extend_from_slice(&len.to_le_bytes());
        name_offs.push(blob.len() as u32);
        blob.extend_from_slice(it.name.as_bytes());
    }

    let record_count = u32::try_from(items.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "too many records"))?;
    let table_off = HEADER_BYTES as u64;
    let name_blob_off = table_off + u64::from(record_count) * super::format::RECORD_BYTES as u64;
    let name_blob_bytes = blob.len() as u64;
    let chunk_bytes = chunks.bytes();

    // The chunk blob's length is known before any offset is chosen, because it depends only on the
    // record lengths -- which is why the data section can still start at a fixed offset.
    let mut flagged: Vec<usize> = Vec::new();
    let mut chunk_blob_bytes = 0u64;
    if chunk_bytes > 0 {
        for (i, it) in items.iter().enumerate() {
            if it.data.len() as u64 > u64::from(chunk_bytes) {
                let n = (it.data.len() as u64).div_ceil(u64::from(chunk_bytes));
                flagged.push(i);
                chunk_blob_bytes += super::format::CHUNK_ENTRY_HEADER as u64 + 32 * n;
            }
        }
    }
    let chunk_blob_off = if chunk_blob_bytes == 0 {
        0
    } else {
        name_blob_off + name_blob_bytes
    };
    let meta_end = if chunk_blob_bytes == 0 {
        name_blob_off + name_blob_bytes
    } else {
        chunk_blob_off + chunk_blob_bytes
    };
    let data_off = align_up(meta_end).max(ALIGN);

    let mut records = Vec::with_capacity(items.len());
    let mut cursor = data_off;
    for (i, it) in items.iter().enumerate() {
        let mut off = cursor;
        if faults.unaligned_first_record && i == 0 {
            off += 1;
        }
        records.push(PackRecord {
            name_off: name_offs[i],
            name_len: it.name.len() as u32,
            rank,
            layout_id: it.layout_id,
            dtype: it.dtype,
            shard_kind: it.shard_kind,
            data_off: off,
            length: it.data.len() as u64,
            shape: it.shape.clone(),
            sha256: Sha256::digest(&it.data).into(),
            flags: if flagged.contains(&i) {
                FLAG_CHUNK_DIGESTS
            } else {
                0
            },
        });
        cursor = align_up(cursor + it.data.len() as u64);
    }
    let file_bytes = cursor.max(data_off);

    let mut table_bytes = Vec::with_capacity(items.len() * super::format::RECORD_BYTES);
    for r in &records {
        table_bytes.extend_from_slice(&r.encode());
    }
    let mut hasher = Sha256::new();
    hasher.update(&table_bytes);
    hasher.update(&blob);
    let table_sha256: [u8; 32] = hasher.finalize().into();

    // Per-record chunk digests, taken from the same bytes the record digest was taken from.
    let mut digests: std::collections::BTreeMap<usize, Vec<[u8; 32]>> =
        std::collections::BTreeMap::new();
    // The payload as it will be WRITTEN, which is the same bytes the record digest was taken
    // over -- except under the fault below, which is the point of it.
    let mut payloads: Vec<&[u8]> = items.iter().map(|it| it.data.as_slice()).collect();
    let flipped: Vec<u8> = match (faults.chunks_describe_other_bytes, flagged.first()) {
        (true, Some(&i)) => {
            let mut v = items[i].data.clone();
            v[0] ^= 0x01;
            v
        }
        _ => Vec::new(),
    };
    if faults.chunks_describe_other_bytes {
        if let Some(&i) = flagged.first() {
            payloads[i] = &flipped;
        }
    }
    for &i in &flagged {
        let step = chunk_bytes as usize;
        digests.insert(
            i,
            payloads[i]
                .chunks(step)
                .map(|c| -> [u8; 32] { Sha256::digest(c).into() })
                .collect(),
        );
    }
    if faults.tamper_chunk_digest {
        if let Some(list) = flagged.first().and_then(|i| digests.get_mut(i)) {
            list[0][0] ^= 0x01;
        }
    }
    let mut chunk_blob = ChunkBlob::encode(&records, &digests);
    assert_eq!(
        chunk_blob.len() as u64,
        chunk_blob_bytes,
        "the chunk blob's length was computed before it was built and must agree"
    );
    let chunk_blob_sha256: [u8; 32] = if chunk_blob_bytes == 0 {
        [0u8; 32]
    } else if faults.wrong_chunk_blob_hash {
        Sha256::digest(b"not this blob").into()
    } else {
        Sha256::digest(&chunk_blob).into()
    };

    let header = PackHeader {
        format_version: FORMAT_VERSION,
        record_count,
        table_off,
        name_blob_off,
        name_blob_bytes,
        data_off,
        file_bytes,
        tp_size,
        rank,
        build_unix_seconds: 0,
        plan_hash,
        tier_id,
        table_sha256,
        chunk_blob_off,
        chunk_blob_bytes,
        chunk_blob_sha256,
        chunk_bytes,
    };

    let mut f = File::create(path)?;
    f.write_all(&header.encode())?;
    f.write_all(&table_bytes)?;
    f.write_all(&blob)?;
    if !chunk_blob.is_empty() {
        f.write_all(&chunk_blob)?;
    }
    chunk_blob.clear();
    for (i, r) in records.iter().enumerate() {
        f.seek(SeekFrom::Start(r.data_off))?;
        f.write_all(payloads[i])?;
    }
    f.set_len(file_bytes)?;
    f.sync_all()?;
    Ok(file_bytes)
}
