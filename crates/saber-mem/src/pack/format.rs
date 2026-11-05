//! The `SBRPACK1` on-disk structures. Byte layout: `docs/abi/PACK.md`.
//!
//! Everything here is little-endian and fixed size. Encoding and decoding are hand-written against
//! the table in that document rather than derived from `#[repr(C)]`, so a struct-layout surprise in
//! either language cannot change the bytes.

#![forbid(unsafe_code)]

use std::fmt;

use super::error::PackError;

/// Magic at offset 0.
pub const MAGIC: [u8; 8] = *b"SBRPACK1";
/// Format version this build **writes**.
///
/// v2 adds the chunk-digest blob (`docs/abi/PACK.md` § Chunk digests) in the header's reserved
/// space and in a section between the name blob and the data. Nothing that v1 defined moved or
/// changed meaning, so a v2 reader reads a v1 pack unchanged — that is what
/// [`FORMAT_VERSION_MIN`] is for.
pub const FORMAT_VERSION: u32 = 2;
/// Oldest format version this build **reads**. A v1 pack simply carries no chunk digests.
pub const FORMAT_VERSION_MIN: u32 = 1;
/// Bytes of a chunk-blob entry before its digest list: `n_chunks`, `flags`, the record's `sha256`.
pub const CHUNK_ENTRY_HEADER: usize = 40;
/// Record flag: this record has an entry in the chunk-digest blob.
pub const FLAG_CHUNK_DIGESTS: u32 = 1 << 0;
/// Record flags this build understands. A record with any other bit set is refused.
pub const FLAGS_KNOWN: u32 = FLAG_CHUNK_DIGESTS;
/// Size of the fixed header.
pub const HEADER_BYTES: usize = 256;
/// Size of one record table entry.
pub const RECORD_BYTES: usize = 160;
/// Alignment of every record's data and of the file length.
pub const ALIGN: u64 = 4096;
/// Maximum number of dimensions a record can carry.
pub const MAX_NDIM: usize = 8;

/// Round `x` up to the next multiple of [`ALIGN`].
#[inline]
#[must_use]
pub fn align_up(x: u64) -> u64 {
    x.div_ceil(ALIGN) * ALIGN
}

/// Element type of a record's payload.
///
/// The tag values are ABI: they are mirrored in `include/saber_abi.h` and must never be renumbered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum Dtype {
    /// Unsigned 8-bit. Also the dtype of NVFP4 `weight_packed` at `layout_id 0`.
    U8 = 0,
    /// Signed 8-bit.
    I8 = 1,
    /// Unsigned 16-bit.
    U16 = 2,
    /// Signed 16-bit.
    I16 = 3,
    /// Unsigned 32-bit.
    U32 = 4,
    /// Signed 32-bit.
    I32 = 5,
    /// Unsigned 64-bit.
    U64 = 6,
    /// Signed 64-bit.
    I64 = 7,
    /// IEEE binary16.
    F16 = 8,
    /// bfloat16.
    BF16 = 9,
    /// IEEE binary32.
    F32 = 10,
    /// IEEE binary64.
    F64 = 11,
    /// OCP `e4m3` 8-bit float.
    F8E4M3 = 12,
    /// OCP `e5m2` 8-bit float.
    F8E5M2 = 13,
    /// Two `e2m1` values per byte, low nibble = even element. Shape is the logical element count.
    FP4E2M1X2 = 14,
    /// One byte per boolean.
    Bool = 15,
    /// MX block-scale exponent (`e8m0`).
    E8M0 = 16,
}

impl Dtype {
    /// Bits per logical element.
    #[must_use]
    pub fn bits(self) -> u64 {
        match self {
            Dtype::U8 | Dtype::I8 | Dtype::F8E4M3 | Dtype::F8E5M2 | Dtype::Bool | Dtype::E8M0 => 8,
            Dtype::U16 | Dtype::I16 | Dtype::F16 | Dtype::BF16 => 16,
            Dtype::U32 | Dtype::I32 | Dtype::F32 => 32,
            Dtype::U64 | Dtype::I64 | Dtype::F64 => 64,
            Dtype::FP4E2M1X2 => 4,
        }
    }

    /// Decode a wire tag.
    pub fn from_tag(tag: u16) -> Result<Self, PackError> {
        Ok(match tag {
            0 => Dtype::U8,
            1 => Dtype::I8,
            2 => Dtype::U16,
            3 => Dtype::I16,
            4 => Dtype::U32,
            5 => Dtype::I32,
            6 => Dtype::U64,
            7 => Dtype::I64,
            8 => Dtype::F16,
            9 => Dtype::BF16,
            10 => Dtype::F32,
            11 => Dtype::F64,
            12 => Dtype::F8E4M3,
            13 => Dtype::F8E5M2,
            14 => Dtype::FP4E2M1X2,
            15 => Dtype::Bool,
            16 => Dtype::E8M0,
            other => return Err(PackError::UnknownDtype(other)),
        })
    }

    /// Wire tag.
    #[must_use]
    pub fn tag(self) -> u16 {
        self as u16
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// How a tensor was split across TP ranks. Informational: the record's bytes are already sharded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ShardKind {
    /// Identical bytes on every rank.
    Replicated = 0,
    /// Column-parallel: split along dim 0.
    Dim0 = 1,
    /// Row-parallel: split along dim 1 (the reduction dim).
    Dim1 = 2,
    /// Dim 0 is a concatenation of segments, each split independently (fused QKV).
    Dim0Segments = 3,
}

impl ShardKind {
    /// Decode a wire tag.
    pub fn from_tag(tag: u8) -> Result<Self, PackError> {
        Ok(match tag {
            0 => ShardKind::Replicated,
            1 => ShardKind::Dim0,
            2 => ShardKind::Dim1,
            3 => ShardKind::Dim0Segments,
            other => return Err(PackError::UnknownShardKind(other)),
        })
    }

    /// Wire tag.
    #[must_use]
    pub fn tag(self) -> u8 {
        self as u8
    }
}

/// `layout_id 0`: raw checkpoint bytes, no transform. See `docs/abi/PACK.md` § Layout ids.
pub const LAYOUT_RAW: u32 = 0;

/// The 256-byte file header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackHeader {
    /// Format version; 1 for this build.
    pub format_version: u32,
    /// Number of entries in the record table.
    pub record_count: u32,
    /// Offset of the record table (always 256 in v1).
    pub table_off: u64,
    /// Offset of the name blob.
    pub name_blob_off: u64,
    /// Length of the name blob.
    pub name_blob_bytes: u64,
    /// Offset of the data section, 4096-aligned.
    pub data_off: u64,
    /// Total file length, a multiple of 4096.
    pub file_bytes: u64,
    /// Number of ranks in the tier.
    pub tp_size: u32,
    /// Rank this file holds.
    pub rank: u32,
    /// Build time, informational only.
    pub build_unix_seconds: u64,
    /// Hash of the resolved plan this pack was built for.
    pub plan_hash: [u8; 32],
    /// Content hash of the tier (the release unit); identical across ranks.
    pub tier_id: [u8; 32],
    /// `sha256` of `[table_off, name_blob_off + name_blob_bytes)`.
    ///
    /// Deliberately **not** widened to cover the chunk blob: its definition is v1's, and the blob
    /// carries its own digest below. A v1 reader's table check therefore means the same thing on a
    /// v2 pack as it did on a v1 one.
    pub table_sha256: [u8; 32],
    /// Offset of the chunk-digest blob, or 0 when the pack has none.
    pub chunk_blob_off: u64,
    /// Length of the chunk-digest blob, or 0 when the pack has none.
    pub chunk_blob_bytes: u64,
    /// `sha256` over the whole chunk-digest blob, or all zero when there is none.
    ///
    /// This is the blob's single authority. A chunk digest is trusted because the blob hashes to
    /// this **and** because the blob entry repeats the record's own `sha256`, which the table hash
    /// already covers — the same rule the parking ticket uses, in the shape a plain per-record
    /// `sha256` allows. See `docs/abi/PACK.md` § Chunk digests for why it cannot be a fold.
    pub chunk_blob_sha256: [u8; 32],
    /// Bytes per chunk the digests were taken at, or 0 when there are none.
    pub chunk_bytes: u32,
}

fn u16_at(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn u64_at(b: &[u8], off: usize) -> u64 {
    let mut v = [0u8; 8];
    v.copy_from_slice(&b[off..off + 8]);
    u64::from_le_bytes(v)
}

fn h32_at(b: &[u8], off: usize) -> [u8; 32] {
    let mut v = [0u8; 32];
    v.copy_from_slice(&b[off..off + 32]);
    v
}

impl PackHeader {
    /// Decode and range-check the header. Does not check anything that needs the table.
    pub fn decode(b: &[u8]) -> Result<Self, PackError> {
        if b.len() < HEADER_BYTES {
            return Err(PackError::Malformed(format!(
                "header needs {HEADER_BYTES} bytes, got {}",
                b.len()
            )));
        }
        if b[0..8] != MAGIC {
            return Err(PackError::BadMagic {
                found: b[0..8].to_vec(),
            });
        }
        let format_version = u32_at(b, 8);
        if !(FORMAT_VERSION_MIN..=FORMAT_VERSION).contains(&format_version) {
            return Err(PackError::Version {
                found: format_version,
                expected: FORMAT_VERSION,
            });
        }
        let header_bytes = u32_at(b, 12);
        if header_bytes as usize != HEADER_BYTES {
            return Err(PackError::StructSize {
                what: "header",
                found: header_bytes,
                expected: HEADER_BYTES as u32,
            });
        }
        let record_bytes = u32_at(b, 16);
        if record_bytes as usize != RECORD_BYTES {
            return Err(PackError::StructSize {
                what: "record",
                found: record_bytes,
                expected: RECORD_BYTES as u32,
            });
        }
        // v1 put nothing here; v2 put the chunk-blob fields in the first 52 bytes of it. The
        // tail stays reserved, and a non-zero byte there is still a refusal.
        let reserved_from = if format_version >= 2 { 228 } else { 176 };
        if b[reserved_from..256].iter().any(|&x| x != 0) {
            return Err(PackError::Malformed(
                "header reserved bytes are not zero".into(),
            ));
        }
        let h = PackHeader {
            format_version,
            record_count: u32_at(b, 20),
            table_off: u64_at(b, 24),
            name_blob_off: u64_at(b, 32),
            name_blob_bytes: u64_at(b, 40),
            data_off: u64_at(b, 48),
            file_bytes: u64_at(b, 56),
            tp_size: u32_at(b, 64),
            rank: u32_at(b, 68),
            build_unix_seconds: u64_at(b, 72),
            plan_hash: h32_at(b, 80),
            tier_id: h32_at(b, 112),
            table_sha256: h32_at(b, 144),
            chunk_blob_off: if format_version >= 2 {
                u64_at(b, 176)
            } else {
                0
            },
            chunk_blob_bytes: if format_version >= 2 {
                u64_at(b, 184)
            } else {
                0
            },
            chunk_blob_sha256: if format_version >= 2 {
                h32_at(b, 192)
            } else {
                [0u8; 32]
            },
            chunk_bytes: if format_version >= 2 {
                u32_at(b, 224)
            } else {
                0
            },
        };
        h.check_self()?;
        Ok(h)
    }

    fn check_self(&self) -> Result<(), PackError> {
        let bad = |m: String| Err(PackError::Malformed(m));
        if self.table_off != HEADER_BYTES as u64 {
            return bad(format!(
                "table_off {} must be {HEADER_BYTES}",
                self.table_off
            ));
        }
        let want_blob = self.table_off + u64::from(self.record_count) * RECORD_BYTES as u64;
        if self.name_blob_off != want_blob {
            return bad(format!(
                "name_blob_off {} must be {want_blob} (table_off + record_count * {RECORD_BYTES})",
                self.name_blob_off
            ));
        }
        if self.data_off % ALIGN != 0 {
            return bad(format!("data_off {} is not {ALIGN}-aligned", self.data_off));
        }
        let names_end = self.name_blob_off + self.name_blob_bytes;
        if self.chunk_blob_bytes == 0 {
            if self.chunk_blob_off != 0
                || self.chunk_bytes != 0
                || self.chunk_blob_sha256 != [0u8; 32]
            {
                return bad(
                    "a pack with no chunk blob must have all four chunk header fields zero".into(),
                );
            }
        } else {
            if self.chunk_blob_off != names_end {
                return bad(format!(
                    "chunk_blob_off {} must be {names_end} (immediately after the name blob)",
                    self.chunk_blob_off
                ));
            }
            if self.chunk_bytes == 0 || u64::from(self.chunk_bytes) % ALIGN != 0 {
                return bad(format!(
                    "chunk_bytes {} must be a non-zero multiple of {ALIGN}",
                    self.chunk_bytes
                ));
            }
            if self.chunk_blob_bytes % CHUNK_ENTRY_HEADER as u64 != 0
                && self.chunk_blob_bytes < CHUNK_ENTRY_HEADER as u64
            {
                return bad(format!(
                    "chunk_blob_bytes {} is too small for one entry",
                    self.chunk_blob_bytes
                ));
            }
        }
        let meta_end = align_up(if self.chunk_blob_bytes == 0 {
            names_end
        } else {
            self.chunk_blob_off + self.chunk_blob_bytes
        });
        if self.data_off < meta_end {
            return bad(format!(
                "data_off {} overlaps the metadata (it ends at {meta_end} after padding)",
                self.data_off
            ));
        }
        if self.file_bytes % ALIGN != 0 {
            return bad(format!(
                "file_bytes {} is not {ALIGN}-aligned",
                self.file_bytes
            ));
        }
        if self.file_bytes < self.data_off {
            return bad(format!(
                "file_bytes {} is before data_off {}",
                self.file_bytes, self.data_off
            ));
        }
        if self.tp_size == 0 || self.rank >= self.tp_size {
            return bad(format!(
                "rank {} is not valid for tp_size {}",
                self.rank, self.tp_size
            ));
        }
        Ok(())
    }

    /// Encode to the 256 wire bytes.
    #[must_use]
    pub fn encode(&self) -> [u8; HEADER_BYTES] {
        let mut b = [0u8; HEADER_BYTES];
        b[0..8].copy_from_slice(&MAGIC);
        b[8..12].copy_from_slice(&self.format_version.to_le_bytes());
        b[12..16].copy_from_slice(&(HEADER_BYTES as u32).to_le_bytes());
        b[16..20].copy_from_slice(&(RECORD_BYTES as u32).to_le_bytes());
        b[20..24].copy_from_slice(&self.record_count.to_le_bytes());
        b[24..32].copy_from_slice(&self.table_off.to_le_bytes());
        b[32..40].copy_from_slice(&self.name_blob_off.to_le_bytes());
        b[40..48].copy_from_slice(&self.name_blob_bytes.to_le_bytes());
        b[48..56].copy_from_slice(&self.data_off.to_le_bytes());
        b[56..64].copy_from_slice(&self.file_bytes.to_le_bytes());
        b[64..68].copy_from_slice(&self.tp_size.to_le_bytes());
        b[68..72].copy_from_slice(&self.rank.to_le_bytes());
        b[72..80].copy_from_slice(&self.build_unix_seconds.to_le_bytes());
        b[80..112].copy_from_slice(&self.plan_hash);
        b[112..144].copy_from_slice(&self.tier_id);
        b[144..176].copy_from_slice(&self.table_sha256);
        if self.format_version >= 2 {
            b[176..184].copy_from_slice(&self.chunk_blob_off.to_le_bytes());
            b[184..192].copy_from_slice(&self.chunk_blob_bytes.to_le_bytes());
            b[192..224].copy_from_slice(&self.chunk_blob_sha256);
            b[224..228].copy_from_slice(&self.chunk_bytes.to_le_bytes());
        }
        b
    }

    /// End of the metadata region before padding: the table, the names and the chunk blob.
    #[must_use]
    pub fn meta_end_unpadded(&self) -> u64 {
        if self.chunk_blob_bytes == 0 {
            self.name_blob_off + self.name_blob_bytes
        } else {
            self.chunk_blob_off + self.chunk_blob_bytes
        }
    }

    /// Where the metadata ends and the data section may begin.
    #[must_use]
    pub fn meta_end(&self) -> u64 {
        align_up(if self.chunk_blob_bytes == 0 {
            self.name_blob_off + self.name_blob_bytes
        } else {
            self.chunk_blob_off + self.chunk_blob_bytes
        })
    }
}

/// One 160-byte record table entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackRecord {
    /// Offset of the name's first UTF-8 byte, relative to `name_blob_off`.
    pub name_off: u32,
    /// Length of the name in bytes.
    pub name_len: u32,
    /// TP rank; equals the header's rank.
    pub rank: u32,
    /// Layout of the payload. 0 = raw checkpoint bytes.
    pub layout_id: u32,
    /// Element type.
    pub dtype: Dtype,
    /// How this tensor was sharded.
    pub shard_kind: ShardKind,
    /// Absolute file offset, 4096-aligned.
    pub data_off: u64,
    /// Payload length in bytes, unpadded.
    pub length: u64,
    /// Rank-local shape, `ndim` entries.
    pub shape: Vec<u64>,
    /// `sha256` over exactly `length` bytes at `data_off`.
    pub sha256: [u8; 32],
    /// Record flags; see [`FLAG_CHUNK_DIGESTS`].
    pub flags: u32,
}

impl PackRecord {
    /// Decode one table entry. `index` only decorates errors.
    pub fn decode(b: &[u8], index: usize) -> Result<Self, PackError> {
        if b.len() < RECORD_BYTES {
            return Err(PackError::Malformed(format!(
                "record {index}: needs {RECORD_BYTES} bytes, got {}",
                b.len()
            )));
        }
        let flags = u32_at(b, 20);
        if flags & !FLAGS_KNOWN != 0 {
            return Err(PackError::Malformed(format!(
                "record {index}: flags {flags:#x} not understood by format version {FORMAT_VERSION}"
            )));
        }
        if b[136..160].iter().any(|&x| x != 0) {
            return Err(PackError::Malformed(format!(
                "record {index}: reserved bytes are not zero"
            )));
        }
        let ndim = b[18] as usize;
        if ndim == 0 || ndim > MAX_NDIM {
            return Err(PackError::Malformed(format!(
                "record {index}: ndim {ndim} outside 1..={MAX_NDIM}"
            )));
        }
        let mut shape = Vec::with_capacity(ndim);
        for d in 0..MAX_NDIM {
            let v = u64_at(b, 40 + 8 * d);
            if d < ndim {
                if v == 0 {
                    return Err(PackError::Malformed(format!(
                        "record {index}: shape[{d}] is zero but ndim is {ndim}"
                    )));
                }
                shape.push(v);
            } else if v != 0 {
                return Err(PackError::Malformed(format!(
                    "record {index}: shape[{d}] is {v} beyond ndim {ndim}"
                )));
            }
        }
        let rec = PackRecord {
            name_off: u32_at(b, 0),
            name_len: u32_at(b, 4),
            rank: u32_at(b, 8),
            layout_id: u32_at(b, 12),
            dtype: Dtype::from_tag(u16_at(b, 16))?,
            shard_kind: ShardKind::from_tag(b[19])?,
            data_off: u64_at(b, 24),
            length: u64_at(b, 32),
            shape,
            sha256: h32_at(b, 104),
            flags,
        };
        Ok(rec)
    }

    /// Encode to the 160 wire bytes.
    #[must_use]
    pub fn encode(&self) -> [u8; RECORD_BYTES] {
        let mut b = [0u8; RECORD_BYTES];
        b[0..4].copy_from_slice(&self.name_off.to_le_bytes());
        b[4..8].copy_from_slice(&self.name_len.to_le_bytes());
        b[8..12].copy_from_slice(&self.rank.to_le_bytes());
        b[12..16].copy_from_slice(&self.layout_id.to_le_bytes());
        b[16..18].copy_from_slice(&self.dtype.tag().to_le_bytes());
        b[18] = self.shape.len() as u8;
        b[19] = self.shard_kind.tag();
        b[20..24].copy_from_slice(&self.flags.to_le_bytes());
        b[24..32].copy_from_slice(&self.data_off.to_le_bytes());
        b[32..40].copy_from_slice(&self.length.to_le_bytes());
        for (d, &v) in self.shape.iter().enumerate() {
            b[40 + 8 * d..48 + 8 * d].copy_from_slice(&v.to_le_bytes());
        }
        b[104..136].copy_from_slice(&self.sha256);
        b
    }

    /// Number of logical elements.
    #[must_use]
    pub fn elements(&self) -> u64 {
        self.shape.iter().product()
    }

    /// The payload length the shape and dtype imply.
    #[must_use]
    pub fn implied_length(&self) -> u64 {
        (self.elements() * self.dtype.bits()).div_ceil(8)
    }

    /// Payload length rounded up to [`ALIGN`] — what a direct read transfers.
    #[must_use]
    pub fn padded_length(&self) -> u64 {
        align_up(self.length)
    }

    /// Whether this record has an entry in the chunk-digest blob.
    #[must_use]
    pub fn has_chunk_digests(&self) -> bool {
        self.flags & FLAG_CHUNK_DIGESTS != 0
    }

    /// Chunks of `chunk_bytes` this record's payload splits into.
    #[must_use]
    pub fn chunk_count(&self, chunk_bytes: u32) -> usize {
        if chunk_bytes == 0 {
            return 0;
        }
        self.length.div_ceil(u64::from(chunk_bytes)) as usize
    }
}

/// The chunk-digest blob: one entry per record whose [`FLAG_CHUNK_DIGESTS`] is set, in table order.
///
/// Entry layout, little-endian, no padding:
///
/// ```text
///   0  u32      n_chunks
///   4  u32      flags            zero in v2
///   8  u8[32]   record_sha256    must equal the table entry's sha256 -- the binding
///  40  u8[32] * n_chunks         the per-chunk digests, in order
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChunkBlob {
    /// Per record index, its chunk digests. Absent for records without the flag.
    lists: std::collections::BTreeMap<usize, Vec<[u8; 32]>>,
}

impl ChunkBlob {
    /// Parse the blob against the record table.
    ///
    /// Every check here is structural and happens at `open`, before any payload byte is read: the
    /// entry count, each entry's length against the record's own `length` and the header's
    /// `chunk_bytes`, and the repeated `sha256` against the table. The blob's own digest is checked
    /// by the caller, which is what makes these entries trustworthy at all.
    pub fn parse(
        bytes: &[u8],
        records: &[PackRecord],
        chunk_bytes: u32,
    ) -> Result<Self, PackError> {
        let mut lists = std::collections::BTreeMap::new();
        let mut off = 0usize;
        for (i, rec) in records.iter().enumerate() {
            if !rec.has_chunk_digests() {
                continue;
            }
            if off + CHUNK_ENTRY_HEADER > bytes.len() {
                return Err(PackError::Malformed(format!(
                    "chunk blob ends inside record {i}'s entry header"
                )));
            }
            let n = u32_at(bytes, off) as usize;
            let flags = u32_at(bytes, off + 4);
            if flags != 0 {
                return Err(PackError::Malformed(format!(
                    "chunk blob entry for record {i}: flags {flags:#x} not understood"
                )));
            }
            let want = rec.chunk_count(chunk_bytes);
            if n != want {
                return Err(PackError::Malformed(format!(
                    "chunk blob entry for record {i}: {n} digests for a {} byte record at \
                     {chunk_bytes} bytes per chunk, which is {want}",
                    rec.length
                )));
            }
            let sha = h32_at(bytes, off + 8);
            if sha != rec.sha256 {
                return Err(PackError::Malformed(format!(
                    "chunk blob entry for record {i} names a different record: its sha256 is not \
                     the table's"
                )));
            }
            let end = off + CHUNK_ENTRY_HEADER + 32 * n;
            if end > bytes.len() {
                return Err(PackError::Malformed(format!(
                    "chunk blob ends inside record {i}'s digest list"
                )));
            }
            let mut list = Vec::with_capacity(n);
            for k in 0..n {
                list.push(h32_at(bytes, off + CHUNK_ENTRY_HEADER + 32 * k));
            }
            lists.insert(i, list);
            off = end;
        }
        if off != bytes.len() {
            return Err(PackError::Malformed(format!(
                "chunk blob is {} bytes; its entries account for {off}",
                bytes.len()
            )));
        }
        Ok(ChunkBlob { lists })
    }

    /// Encode a blob for `records`, taking each flagged record's digests from `digests`.
    #[must_use]
    pub fn encode(
        records: &[PackRecord],
        digests: &std::collections::BTreeMap<usize, Vec<[u8; 32]>>,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        for (i, rec) in records.iter().enumerate() {
            if !rec.has_chunk_digests() {
                continue;
            }
            let list = digests.get(&i).map_or(&[][..], Vec::as_slice);
            out.extend_from_slice(&(list.len() as u32).to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&rec.sha256);
            for d in list {
                out.extend_from_slice(d);
            }
        }
        out
    }

    /// One record's chunk digests, if it has any.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&[[u8; 32]]> {
        self.lists.get(&index).map(Vec::as_slice)
    }

    /// Records that carry digests.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lists.len()
    }

    /// Whether no record carries digests — a v1 pack, or a v2 one built with none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lists.is_empty()
    }

    /// Chunk digests held in total.
    #[must_use]
    pub fn total_chunks(&self) -> usize {
        self.lists.values().map(Vec::len).sum()
    }
}
