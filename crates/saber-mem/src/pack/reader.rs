//! Parse and verify a pack. See `docs/abi/PACK.md`.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::dio::{self, AlignedBuf, OpenMode};

use super::error::PackError;
use super::format::{
    align_up, ChunkBlob, PackHeader, PackRecord, ALIGN, HEADER_BYTES, RECORD_BYTES,
};

/// A parsed, validated pack: header, record table and names. Holds the file open; reads no data
/// until asked.
#[derive(Debug)]
pub struct PackReader {
    path: PathBuf,
    file: File,
    mode: OpenMode,
    header: PackHeader,
    records: Vec<PackRecord>,
    names: Vec<String>,
    by_name: HashMap<String, usize>,
    chunks: ChunkBlob,
}

impl PackReader {
    /// Open and validate a pack with `O_DIRECT`.
    pub fn open(path: &Path) -> Result<Self, PackError> {
        Self::open_with(path, OpenMode::Direct)
    }

    /// Open and validate a pack in the given mode.
    pub fn open_with(path: &Path, mode: OpenMode) -> Result<Self, PackError> {
        let file = dio::open_read(path, mode).map_err(|e| PackError::io(path, "open", e))?;
        let file_len = file
            .metadata()
            .map_err(|e| PackError::io(path, "stat", e))?
            .len();

        if file_len < ALIGN {
            return Err(PackError::Malformed(format!(
                "file is {file_len} bytes; a pack is at least one {ALIGN}-byte block \
                 (header {HEADER_BYTES} + padding)"
            )));
        }

        // Step 1: the first aligned block always contains the whole header.
        let first = read_aligned(&file, path, 0, ALIGN)?;
        let header = PackHeader::decode(&first)?;
        if header.file_bytes != file_len {
            return Err(PackError::Malformed(format!(
                "header says file_bytes {} but the file is {file_len} bytes",
                header.file_bytes
            )));
        }

        // Step 2: the metadata region [0, meta_end) holds the table, the name blob and, in a v2
        // pack, the chunk-digest blob.
        let meta_end = header.meta_end_unpadded();
        let meta = if meta_end <= first.len() as u64 {
            first
        } else {
            read_aligned(&file, path, 0, align_up(meta_end))?
        };

        let table_start = header.table_off as usize;
        let blob_start = header.name_blob_off as usize;
        let blob_end = (header.name_blob_off + header.name_blob_bytes) as usize;

        let mut hasher = Sha256::new();
        let mut records = Vec::with_capacity(header.record_count as usize);
        for i in 0..header.record_count as usize {
            let off = table_start + i * RECORD_BYTES;
            let bytes = &meta[off..off + RECORD_BYTES];
            hasher.update(bytes);
            records.push(PackRecord::decode(bytes, i)?);
        }
        hasher.update(&meta[blob_start..blob_end]);
        let table_digest: [u8; 32] = hasher.finalize().into();

        // Names first: structural record checks quote them.
        let blob = &meta[blob_start..blob_end];
        let mut names = Vec::with_capacity(records.len());
        for (i, r) in records.iter().enumerate() {
            names.push(decode_name(blob, r, i)?);
        }

        // The chunk blob, when there is one. Validated against its own digest -- the blob is
        // trustworthy because of that hash plus the record `sha256` each entry repeats, which the
        // table hash already covers.
        let chunks = if header.chunk_blob_bytes == 0 {
            ChunkBlob::default()
        } else {
            let lo = header.chunk_blob_off as usize;
            let hi = lo + header.chunk_blob_bytes as usize;
            if hi > meta.len() {
                return Err(PackError::Malformed(format!(
                    "chunk blob [{lo}, {hi}) is outside the {}-byte metadata region",
                    meta.len()
                )));
            }
            let bytes = &meta[lo..hi];
            let digest: [u8; 32] = Sha256::digest(bytes).into();
            if digest != header.chunk_blob_sha256 {
                return Err(PackError::Malformed(format!(
                    "chunk blob hashes to {} but the header says {}",
                    hex::encode(digest),
                    hex::encode(header.chunk_blob_sha256)
                )));
            }
            ChunkBlob::parse(bytes, &records, header.chunk_bytes)?
        };

        let me = PackReader {
            path: path.to_path_buf(),
            file,
            mode,
            header,
            records,
            names,
            by_name: HashMap::new(),
            chunks,
        };
        me.check_records()?;

        // The table hash is checked last on purpose: a structurally broken table should report the
        // structural fault, not "hash mismatch", which tells a human nothing.
        if table_digest != me.header.table_sha256 {
            return Err(PackError::TableHashMismatch {
                expected: hex::encode(me.header.table_sha256),
                found: hex::encode(table_digest),
            });
        }

        let mut me = me;
        for (i, n) in me.names.iter().enumerate() {
            if me.by_name.insert(n.clone(), i).is_some() {
                return Err(PackError::Malformed(format!("duplicate record name {n:?}")));
            }
        }
        Ok(me)
    }

    /// Open, validate, and refuse the pack unless its plan hash is `expected`.
    ///
    /// A mismatch is an error. The loader never re-derives a layout to make a foreign pack fit.
    pub fn open_expecting(
        path: &Path,
        mode: OpenMode,
        expected_plan_hash: &[u8; 32],
    ) -> Result<Self, PackError> {
        let me = Self::open_with(path, mode)?;
        me.check_plan_hash(expected_plan_hash)?;
        Ok(me)
    }

    /// Error unless the pack's plan hash is `expected`.
    pub fn check_plan_hash(&self, expected: &[u8; 32]) -> Result<(), PackError> {
        if &self.header.plan_hash != expected {
            return Err(PackError::PlanHashMismatch {
                found: hex::encode(self.header.plan_hash),
                expected: hex::encode(expected),
            });
        }
        Ok(())
    }

    fn check_records(&self) -> Result<(), PackError> {
        let h = &self.header;
        let mut cursor = h.data_off;
        for (i, r) in self.records.iter().enumerate() {
            let name = &self.names[i];
            if r.rank != h.rank {
                return Err(PackError::Malformed(format!(
                    "record {i} ({name}): rank {} but the file is rank {}",
                    r.rank, h.rank
                )));
            }
            if r.data_off % ALIGN != 0 {
                return Err(PackError::UnalignedRecord {
                    index: i,
                    name: name.clone(),
                    off: r.data_off,
                    align: ALIGN,
                });
            }
            if r.data_off < cursor {
                return Err(PackError::Malformed(format!(
                    "record {i} ({name}): data_off {} is below {cursor}; the table must be sorted \
                     by data_off and records must not overlap after padding",
                    r.data_off
                )));
            }
            let end = r.data_off.checked_add(r.length).ok_or_else(|| {
                PackError::Malformed(format!("record {i} ({name}): length overflow"))
            })?;
            if align_up(end) > h.file_bytes {
                return Err(PackError::Malformed(format!(
                    "record {i} ({name}): [{}, {end}) runs past file_bytes {}",
                    r.data_off, h.file_bytes
                )));
            }
            let implied = r.implied_length();
            if implied != r.length {
                return Err(PackError::Malformed(format!(
                    "record {i} ({name}): length {} but shape {:?} of {} implies {implied}",
                    r.length, r.shape, r.dtype
                )));
            }
            cursor = align_up(end);
        }
        Ok(())
    }

    /// Path this pack was opened from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// How the file was opened.
    #[must_use]
    pub fn mode(&self) -> OpenMode {
        self.mode
    }

    /// The file header.
    #[must_use]
    pub fn header(&self) -> &PackHeader {
        &self.header
    }

    /// The record table, in ascending `data_off` order.
    #[must_use]
    pub fn records(&self) -> &[PackRecord] {
        &self.records
    }

    /// Name of record `index`.
    #[must_use]
    pub fn name(&self, index: usize) -> &str {
        &self.names[index]
    }

    /// Table index of a tensor by name.
    #[must_use]
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.by_name.get(name).copied()
    }

    /// Total payload bytes (unpadded) across all records.
    #[must_use]
    pub fn payload_bytes(&self) -> u64 {
        self.records.iter().map(|r| r.length).sum()
    }

    /// Largest padded record — the floor on a streaming loader's memory budget.
    #[must_use]
    pub fn max_padded_record(&self) -> u64 {
        self.records
            .iter()
            .map(super::format::PackRecord::padded_length)
            .max()
            .unwrap_or(0)
    }

    pub(crate) fn file(&self) -> &File {
        &self.file
    }

    /// One record's chunk digests, if the pack carries them for it.
    ///
    /// The list is what lets a verifier hash a big record on every core instead of one: each entry
    /// covers `header().chunk_bytes` payload bytes, the last one the remainder. `None` for a v1
    /// pack, and for any record short enough that a chunk digest would just be a second hash of
    /// what the record digest already covers.
    #[must_use]
    pub fn chunk_digests(&self, index: usize) -> Option<&[[u8; 32]]> {
        self.chunks.get(index)
    }

    /// Bytes each chunk digest covers, or 0 when the pack has none.
    #[must_use]
    pub fn chunk_bytes(&self) -> u32 {
        self.header.chunk_bytes
    }

    /// Records carrying chunk digests, and how many digests in total.
    #[must_use]
    pub fn chunk_summary(&self) -> (usize, usize) {
        (self.chunks.len(), self.chunks.total_chunks())
    }

    /// Read one record into a page-aligned buffer. The buffer's visible payload is exactly
    /// `record.length`; its capacity is the padded length that was transferred.
    pub fn read_record(&self, index: usize) -> Result<AlignedBuf, PackError> {
        let r = &self.records[index];
        let mut buf = read_aligned(&self.file, &self.path, r.data_off, r.padded_length())?;
        buf.truncate(r.length as usize);
        Ok(buf)
    }

    /// Read one record and check its content hash.
    pub fn verify_record(&self, index: usize) -> Result<(), PackError> {
        let buf = self.read_record(index)?;
        self.check_hash_strict(index, &buf)
    }

    /// Check a record's content hash against bytes already in hand.
    /// Check `bytes` against the record's chunk digests when the pack has them, and against its
    /// whole-record `sha256` when it does not.
    ///
    /// This is the **load** path's check: one hash pass over the payload either way, and with chunk
    /// digests that pass parallelises. Every byte is covered by a digest the builder wrote, so it is
    /// as strong as the whole-record hash against a drive that returned the wrong bytes.
    ///
    /// What it cannot see is a pack whose two descriptions of the same record disagree — a builder
    /// that wrote a chunk list over one version of the bytes and a record digest over another. No
    /// format can cross-check those without hashing the payload twice, so the offline verifier does
    /// exactly that: see [`PackReader::check_hash_strict`], which is what `verify_record` uses.
    pub fn check_hash(&self, index: usize, bytes: &[u8]) -> Result<(), PackError> {
        if let Some(list) = self.chunks.get(index) {
            let step = self.header.chunk_bytes as usize;
            for (k, (c, want)) in bytes.chunks(step).zip(list).enumerate() {
                let d: [u8; 32] = Sha256::digest(c).into();
                if d != *want {
                    let lo = (k * step) as u64;
                    return Err(PackError::RecordChunkMismatch {
                        index,
                        name: self.names[index].clone(),
                        chunk: k,
                        chunks: list.len(),
                        lo,
                        hi: lo + c.len() as u64,
                        expected: hex::encode(want),
                        found: hex::encode(d),
                    });
                }
            }
            return Ok(());
        }
        self.check_record_digest(index, bytes)
    }

    /// Check **both** descriptions of the record: its chunk digests and its whole-record `sha256`.
    ///
    /// Two hash passes over the payload for a record that carries chunk digests, which is why the
    /// load path does not do it. An offline verifier can afford it, and it is the only thing that
    /// catches a builder whose chunk list and record digest describe different bytes.
    pub fn check_hash_strict(&self, index: usize, bytes: &[u8]) -> Result<(), PackError> {
        self.check_hash(index, bytes)?;
        if self.chunks.get(index).is_some() {
            self.check_record_digest(index, bytes)?;
        }
        Ok(())
    }

    fn check_record_digest(&self, index: usize, bytes: &[u8]) -> Result<(), PackError> {
        let r = &self.records[index];
        let digest: [u8; 32] = Sha256::digest(bytes).into();
        if digest != r.sha256 {
            return Err(PackError::RecordHashMismatch {
                index,
                name: self.names[index].clone(),
                expected: hex::encode(r.sha256),
                found: hex::encode(digest),
            });
        }
        Ok(())
    }

    /// Verify every record, sequentially (one read in flight).
    ///
    /// This is the simple, obvious check; it runs at the drive's QD1 rate (10.18 GB/s measured,
    /// minus hashing). [`super::PackStream`] with `verify: true` does the same work at QD8.
    pub fn verify_all(&self) -> Result<(), PackError> {
        for i in 0..self.records.len() {
            self.verify_record(i)?;
        }
        Ok(())
    }
}

fn decode_name(blob: &[u8], r: &PackRecord, index: usize) -> Result<String, PackError> {
    let off = r.name_off as usize;
    let len = r.name_len as usize;
    if len == 0 {
        return Err(PackError::Malformed(format!("record {index}: empty name")));
    }
    if off < 2 || off + len > blob.len() {
        return Err(PackError::Malformed(format!(
            "record {index}: name [{off}, {}) outside the {}-byte name blob",
            off + len,
            blob.len()
        )));
    }
    let prefix = u16::from_le_bytes([blob[off - 2], blob[off - 1]]) as usize;
    if prefix != len {
        return Err(PackError::Malformed(format!(
            "record {index}: name length prefix {prefix} disagrees with name_len {len}"
        )));
    }
    std::str::from_utf8(&blob[off..off + len])
        .map(str::to_owned)
        .map_err(|e| PackError::Malformed(format!("record {index}: name is not UTF-8: {e}")))
}

/// Read `[off, off+len)` (both multiples of [`ALIGN`]) into a page-aligned buffer.
pub(crate) fn read_aligned(
    file: &File,
    path: &Path,
    off: u64,
    len: u64,
) -> Result<AlignedBuf, PackError> {
    debug_assert_eq!(off % ALIGN, 0);
    debug_assert_eq!(len % ALIGN, 0);
    let mut buf = AlignedBuf::zeroed(len as usize);
    if len > 0 {
        dio::preadv_exact(file, &mut buf, off)
            .map_err(|e| PackError::io(path, format!("read {len} bytes at {off}"), e))?;
    }
    Ok(buf)
}

/// Size of the metadata region (header + table + names) for `record_count` records whose names
/// total `name_blob_bytes`. Used by writers.
///
/// **v1's definition, kept.** A v2 pack's metadata also holds the chunk blob, so the region a
/// reader has to read is [`PackHeader::meta_end_unpadded`]; this helper answers the narrower
/// question — where the names end — which is exactly what `table_sha256` is taken over.
#[must_use]
pub fn metadata_end(record_count: u32, name_blob_bytes: u64) -> u64 {
    HEADER_BYTES as u64 + u64::from(record_count) * RECORD_BYTES as u64 + name_blob_bytes
}
