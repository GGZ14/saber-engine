//! The Flash-Next expert-tier file, `rank<r>/experts.xtier` (docs/EXEC.md §10, the pack law).
//!
//! A 4 KiB header (`SBRXPRT1`, v1), a sha256 table (32 B per record), then `layers × experts`
//! FIXED-SIZE records: record `i = layer · experts + (expert − first_expert)` at
//! `data_off + i · stride`. One record is one routed expert of one layer (gate_up `w`/`ws`/`wref`
//! and down `w`/`ws`/`wref`, every sub-record 256-aligned); the stride is a multiple of 4 KiB, so
//! ONE aligned `O_DIRECT` read of `stride` bytes pulls one expert into a VRAM staging slot or a
//! pinned RAM slot with no transform. The record's sha256 covers the whole stride (payload + the
//! zero pad), exactly as `tools/pack/build_flashnext_pack.py` hashes it.
//!
//! **v2 (FN144)**: the same 4 KiB header with version 2 and a FORMAT TAG at bytes 256..260
//! ([`XtierFormat`]): `NVFP4_E4M3G16` = the FP4-Local checkpoint's NVFP4 experts carried
//! losslessly -- gate_up `w` / `ws` (E4M3 per 16, `[H/16, 2I]`) / `gs` (two f32 globals: gate, up)
//! and down `w` / `ws` (`[I/16, H]`) / `gs` (one f32), 676 pages a record. A v1 file reads exactly
//! as before (format `MXFP4_E8M0`, implied); an MXFP4 header is still WRITTEN as v1 (byte-identical
//! packs). Every engine boot path calls [`XtierHeader::require_mxfp4`] until the NVFP4 kernels
//! (FN133 P3) exist: an NVFP4 tier is refused by name, never read as MXFP4.
//!
//! This module parses and checks the header and the table (the geometry must be self-consistent,
//! the table must hash to the header's digest and the file must be exactly `file_bytes` long) and
//! reads records. It also WRITES a synthetic file of the same format ([`write_synthetic`]) so the
//! tier's host tests and the FN7 benches run over fake records that obey the real law.

#![forbid(unsafe_code)]

use std::fs::File;
use std::io;
use std::path::Path;

use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::dio::{self, OpenMode, DIO_ALIGN};

/// Magic bytes of a v1 expert-tier file.
pub const XTIER_MAGIC: &[u8; 8] = b"SBRXPRT1";
/// The v1 version (MXFP4, no format tag).
pub const XTIER_VERSION: u32 = 1;
/// FN144: the v2 version (a format tag at bytes 256..260).
pub const XTIER_VERSION_V2: u32 = 2;
/// FN144: byte offset of the v2 format tag.
pub const XTIER_FORMAT_OFF: usize = 256;
/// Header bytes.
pub const XTIER_HEADER: u64 = 4096;
/// Sub-record alignment inside a record.
pub const SUB_ALIGN: u64 = 256;

/// The six sub-records of one expert record, in record order.
pub const SUB_NAMES: [&str; 6] = [
    "gate_up.w",
    "gate_up.ws",
    "gate_up.wref",
    "down.w",
    "down.ws",
    "down.wref",
];

/// The six sub-records of a v2 NVFP4 record (FN144), in record order.
pub const SUB_NAMES_NVFP4: [&str; 6] = [
    "gate_up.w",
    "gate_up.ws",
    "gate_up.gs",
    "down.w",
    "down.ws",
    "down.gs",
];

/// FN144: a tier file's record format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XtierFormat {
    /// e2m1 + e8m0 per 32 (Quark), `wref` = the scales' row max. v1 (implied) or v2 tag 1.
    Mxfp4E8m0,
    /// e2m1 + E4M3 per 16 + f32 globals (ModelOpt NVFP4, the scales CSF-decoded). v2 tag 2.
    Nvfp4E4m3G16,
}

impl XtierFormat {
    /// The v2 tag.
    #[must_use]
    pub fn tag(self) -> u32 {
        match self {
            XtierFormat::Mxfp4E8m0 => 1,
            XtierFormat::Nvfp4E4m3G16 => 2,
        }
    }
    /// The manifest / log spelling.
    #[must_use]
    pub fn key(self) -> &'static str {
        match self {
            XtierFormat::Mxfp4E8m0 => "MXFP4_E8M0",
            XtierFormat::Nvfp4E4m3G16 => "NVFP4_E4M3G16",
        }
    }
    /// The format of a v2 tag.
    #[must_use]
    pub fn from_tag(t: u32) -> Option<Self> {
        match t {
            1 => Some(XtierFormat::Mxfp4E8m0),
            2 => Some(XtierFormat::Nvfp4E4m3G16),
            _ => None,
        }
    }
    /// The sub-record names of a record of this format.
    #[must_use]
    pub fn sub_names(self) -> [&'static str; 6] {
        match self {
            XtierFormat::Mxfp4E8m0 => SUB_NAMES,
            XtierFormat::Nvfp4E4m3G16 => SUB_NAMES_NVFP4,
        }
    }
    /// `(sub, payload, stride)` of a record of this format.
    #[must_use]
    pub fn record_layout(self, hidden: u64, inter: u64) -> ([(u64, u64); 6], u64, u64) {
        match self {
            XtierFormat::Mxfp4E8m0 => record_layout(hidden, inter),
            XtierFormat::Nvfp4E4m3G16 => record_layout_nvfp4(hidden, inter),
        }
    }
}

/// Why an expert-tier file was refused.
#[derive(Debug, Error)]
pub enum XtierError {
    /// I/O.
    #[error("{path}: {source}")]
    Io {
        /// The file.
        path: String,
        /// The error.
        source: io::Error,
    },
    /// The header or table is not a valid v1 expert tier.
    #[error("{path}: {what}")]
    Format {
        /// The file.
        path: String,
        /// What is wrong, with the numbers.
        what: String,
    },
    /// A record index outside the file.
    #[error("record {index} outside the tier's {records} records")]
    Range {
        /// Requested index.
        index: u64,
        /// Records in the file.
        records: u64,
    },
    /// A destination buffer that is not one aligned stride.
    #[error("record buffer: {0}")]
    Buffer(String),
}

/// The parsed v1 header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XtierHeader {
    /// Owning rank.
    pub rank: u32,
    /// Layers (48 for Flash-Next).
    pub layers: u32,
    /// First owned expert id (`256 · rank` at EP=2).
    pub first_expert: u32,
    /// Owned experts per layer (256 at EP=2).
    pub experts: u32,
    /// Record stride, a multiple of 4 KiB (2,617,344 B).
    pub stride: u64,
    /// Payload bytes of a record (2,615,040 B).
    pub payload: u64,
    /// `(offset, length)` of each sub-record inside a record, [`SUB_NAMES`] order.
    pub sub: [(u64, u64); 6],
    /// Model hidden size.
    pub hidden: u32,
    /// Expert intermediate size.
    pub inter: u32,
    /// Byte offset of the sha256 table.
    pub table_off: u64,
    /// Bytes of the sha256 table (32 per record).
    pub table_bytes: u64,
    /// Byte offset of record 0.
    pub data_off: u64,
    /// Total file bytes.
    pub file_bytes: u64,
    /// sha256 of the table.
    pub table_sha256: [u8; 32],
    /// sha256 of the pack plan the file was built from.
    pub plan_sha256: [u8; 32],
    /// FN80a: the split law (header bytes 248..256, zero in every pre-FN80a file): 0 = whole
    /// experts (EP); 2 = every record is the `split_index`-th half of its expert's intermediate
    /// dimension (`saber flashnext-pack --experts tp2`: gate / up rows and down K
    /// `[inter * index, inter * (index + 1))`, `wref` = the WHOLE row's max).
    pub split_parts: u32,
    /// FN80a: which part (`< split_parts`; 0 when whole).
    pub split_index: u32,
    /// FN144: the record format (v1 = `Mxfp4E8m0`).
    pub format: XtierFormat,
}

/// FN80a: the sha256 of one record stride (the table's digest law).
#[must_use]
pub fn record_digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// FN80a: the sha256 of a table (the header's `table_sha256`).
#[must_use]
pub fn table_digest(table: &[u8]) -> [u8; 32] {
    Sha256::digest(table).into()
}

/// The 4 KiB v1 header block of `h` (the inverse of [`XtierHeader::parse`]).
#[must_use]
pub fn header_block(h: &XtierHeader) -> Vec<u8> {
    let mut hb = vec![0u8; XTIER_HEADER as usize];
    hb[0..8].copy_from_slice(XTIER_MAGIC);
    let put32 = |hb: &mut [u8], o: usize, v: u32| hb[o..o + 4].copy_from_slice(&v.to_le_bytes());
    let put64 = |hb: &mut [u8], o: usize, v: u64| hb[o..o + 8].copy_from_slice(&v.to_le_bytes());
    // An MXFP4 tier stays a v1 file, byte for byte (FN144).
    let v2 = h.format != XtierFormat::Mxfp4E8m0;
    put32(
        &mut hb,
        8,
        if v2 { XTIER_VERSION_V2 } else { XTIER_VERSION },
    );
    put32(&mut hb, 12, XTIER_HEADER as u32);
    put32(&mut hb, 16, h.rank);
    put32(&mut hb, 20, h.layers);
    put32(&mut hb, 24, h.first_expert);
    put32(&mut hb, 28, h.experts);
    put64(&mut hb, 32, h.stride);
    put64(&mut hb, 40, h.payload);
    for (i, &(o, l)) in h.sub.iter().enumerate() {
        put64(&mut hb, 48 + 16 * i, o);
        put64(&mut hb, 56 + 16 * i, l);
    }
    put32(&mut hb, 144, h.hidden);
    put32(&mut hb, 148, h.inter);
    put64(&mut hb, 152, h.table_off);
    put64(&mut hb, 160, h.table_bytes);
    put64(&mut hb, 168, h.data_off);
    put64(&mut hb, 176, h.file_bytes);
    hb[184..216].copy_from_slice(&h.table_sha256);
    hb[216..248].copy_from_slice(&h.plan_sha256);
    put32(&mut hb, 248, h.split_parts);
    put32(&mut hb, 252, h.split_index);
    if v2 {
        put32(&mut hb, XTIER_FORMAT_OFF, h.format.tag());
    }
    hb
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn u64_at(b: &[u8], o: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(a)
}

impl XtierHeader {
    /// Records in the file.
    #[must_use]
    pub fn records(&self) -> u64 {
        u64::from(self.layers) * u64::from(self.experts)
    }

    /// Record index of (layer, GLOBAL expert id), `None` when this rank does not own it.
    #[must_use]
    pub fn index(&self, layer: u32, expert: u32) -> Option<u64> {
        if layer >= self.layers
            || expert < self.first_expert
            || expert >= self.first_expert + self.experts
        {
            return None;
        }
        Some(u64::from(layer) * u64::from(self.experts) + u64::from(expert - self.first_expert))
    }

    /// FN144: refuse a tier whose records are not MXFP4 (every engine boot path, until the NVFP4
    /// expert kernels exist).
    pub fn require_mxfp4(&self) -> Result<(), String> {
        self.require_format(XtierFormat::Mxfp4E8m0)
    }

    /// FN150: refuse a tier whose records are not the boot's `--fn-experts-fmt` (`want`): the
    /// format check that routes an NVFP4 tier to FN148's bodies and nothing else to them.
    pub fn require_format(&self, want: XtierFormat) -> Result<(), String> {
        if self.format == want {
            Ok(())
        } else {
            Err(format!(
                "rank {}: the tier's records are {} but this boot runs {} expert bodies \
                 (--fn-experts-fmt {}; a v2 NVFP4 tier needs --fn-experts-fmt nvfp4 = FN148's \
                 form (b), FN150)",
                self.rank,
                self.format.key(),
                want.key(),
                if want == XtierFormat::Mxfp4E8m0 {
                    "mxfp4"
                } else {
                    "nvfp4"
                }
            ))
        }
    }

    /// File offset of record `i`.
    #[must_use]
    pub fn record_offset(&self, i: u64) -> u64 {
        self.data_off + i * self.stride
    }

    /// Parse and check a header block (the first 4 KiB of the file).
    pub fn parse(path: &str, b: &[u8]) -> Result<XtierHeader, XtierError> {
        let bad = |what: String| XtierError::Format {
            path: path.to_string(),
            what,
        };
        if b.len() < XTIER_HEADER as usize {
            return Err(bad(format!("header is {} bytes, not 4096", b.len())));
        }
        if &b[0..8] != XTIER_MAGIC {
            return Err(bad(format!("magic {:02x?} is not SBRXPRT1", &b[0..8])));
        }
        let version = u32_at(b, 8);
        let format = match version {
            XTIER_VERSION => XtierFormat::Mxfp4E8m0,
            XTIER_VERSION_V2 => {
                let t = u32_at(b, XTIER_FORMAT_OFF);
                XtierFormat::from_tag(t)
                    .ok_or_else(|| bad(format!("v2 format tag {t}, this reader knows 1, 2")))?
            }
            _ => return Err(bad(format!("version {version}, this reader knows 1, 2"))),
        };
        let hbytes = u32_at(b, 12);
        if u64::from(hbytes) != XTIER_HEADER {
            return Err(bad(format!("header bytes {hbytes}, not 4096")));
        }
        let mut sub = [(0u64, 0u64); 6];
        for (i, s) in sub.iter_mut().enumerate() {
            *s = (u64_at(b, 48 + 16 * i), u64_at(b, 56 + 16 * i));
        }
        let mut table_sha256 = [0u8; 32];
        table_sha256.copy_from_slice(&b[184..216]);
        let mut plan_sha256 = [0u8; 32];
        plan_sha256.copy_from_slice(&b[216..248]);
        let h = XtierHeader {
            rank: u32_at(b, 16),
            layers: u32_at(b, 20),
            first_expert: u32_at(b, 24),
            experts: u32_at(b, 28),
            stride: u64_at(b, 32),
            payload: u64_at(b, 40),
            sub,
            hidden: u32_at(b, 144),
            inter: u32_at(b, 148),
            table_off: u64_at(b, 152),
            table_bytes: u64_at(b, 160),
            data_off: u64_at(b, 168),
            file_bytes: u64_at(b, 176),
            table_sha256,
            plan_sha256,
            split_parts: u32_at(b, 248),
            split_index: u32_at(b, 252),
            format,
        };
        h.check().map_err(bad)?;
        Ok(h)
    }

    /// The geometry law, every number named on failure.
    fn check(&self) -> Result<(), String> {
        let a = DIO_ALIGN as u64;
        if self.layers == 0 || self.experts == 0 {
            return Err(format!("{} layers x {} experts", self.layers, self.experts));
        }
        if self.stride == 0 || self.stride % a != 0 {
            return Err(format!(
                "stride {} is not a positive multiple of 4096",
                self.stride
            ));
        }
        if (self.split_parts == 0 && self.split_index != 0)
            || self.split_parts == 1
            || (self.split_parts > 1 && self.split_index >= self.split_parts)
        {
            return Err(format!(
                "split marker ({} parts, index {}) is neither whole (0, 0) nor a part of >= 2",
                self.split_parts, self.split_index
            ));
        }
        if self.payload > self.stride {
            return Err(format!("payload {} > stride {}", self.payload, self.stride));
        }
        let mut end = 0u64;
        for (i, &(off, len)) in self.sub.iter().enumerate() {
            if off % SUB_ALIGN != 0 || off < end || off + len > self.payload || len == 0 {
                return Err(format!(
                    "sub-record {} at ({off}, {len}) breaks the 256-aligned, ordered, inside-payload law",
                    self.format.sub_names()[i]
                ));
            }
            end = off + len;
        }
        if self.table_off < XTIER_HEADER || self.table_bytes != self.records() * 32 {
            return Err(format!(
                "table ({}, {}) for {} records",
                self.table_off,
                self.table_bytes,
                self.records()
            ));
        }
        if self.data_off % a != 0 || self.data_off < self.table_off + self.table_bytes {
            return Err(format!(
                "data_off {} is unaligned or overlaps the table",
                self.data_off
            ));
        }
        if self.file_bytes != self.data_off + self.records() * self.stride {
            return Err(format!(
                "file_bytes {} != data_off {} + {} records x {}",
                self.file_bytes,
                self.data_off,
                self.records(),
                self.stride
            ));
        }
        Ok(())
    }
}

/// An open expert-tier file: its header, its sha256 table and a descriptor for record reads.
#[derive(Debug)]
pub struct XtierFile {
    path: String,
    file: File,
    header: XtierHeader,
    table: Vec<[u8; 32]>,
    mode: OpenMode,
}

impl XtierFile {
    /// Open, parse the header, read the table and check it against the header's digest and the
    /// file length. `mode` is `Direct` in production (the measurement's honest mode) and
    /// `Buffered` only for filesystems without `O_DIRECT`.
    pub fn open(path: &Path, mode: OpenMode) -> Result<XtierFile, XtierError> {
        let p = path.display().to_string();
        let io = |source| XtierError::Io {
            path: p.clone(),
            source,
        };
        let file = dio::open_read(path, mode).map_err(io)?;
        let len = file.metadata().map_err(io)?.len();
        let mut hb = dio::AlignedBuf::zeroed(XTIER_HEADER as usize);
        dio::preadv_exact(&file, &mut hb, 0).map_err(io)?;
        let header = XtierHeader::parse(&p, &hb)?;
        if len != header.file_bytes {
            return Err(XtierError::Format {
                path: p,
                what: format!("file is {len} bytes, the header says {}", header.file_bytes),
            });
        }
        let tlen = dio::align_up(header.table_bytes) as usize;
        let toff = header.table_off;
        if toff % DIO_ALIGN as u64 != 0 {
            return Err(XtierError::Format {
                path: p,
                what: format!("table offset {toff} is not 4096-aligned"),
            });
        }
        let mut tb = dio::AlignedBuf::zeroed(tlen);
        dio::preadv_exact(&file, &mut tb, toff).map_err(io)?;
        let tbytes = &tb[..header.table_bytes as usize];
        let digest: [u8; 32] = Sha256::digest(tbytes).into();
        if digest != header.table_sha256 {
            return Err(XtierError::Format {
                path: p,
                what: "the sha256 table does not hash to the header's table digest".into(),
            });
        }
        let table = tbytes
            .chunks_exact(32)
            .map(|c| {
                let mut a = [0u8; 32];
                a.copy_from_slice(c);
                a
            })
            .collect();
        Ok(XtierFile {
            path: p,
            file,
            header,
            table,
            mode,
        })
    }

    /// The header.
    #[must_use]
    pub fn header(&self) -> &XtierHeader {
        &self.header
    }

    /// FN71c: the open file (`O_DIRECT` in production) -- a P2PDMA pool reader's source.
    #[must_use]
    pub fn file(&self) -> &std::fs::File {
        &self.file
    }

    /// The file's path, for messages.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// The open mode.
    #[must_use]
    pub fn mode(&self) -> OpenMode {
        self.mode
    }

    /// The table's sha256 of record `i` (whole stride).
    #[must_use]
    pub fn record_sha256(&self, i: u64) -> Option<&[u8; 32]> {
        self.table.get(i as usize)
    }

    /// Read record `i` into `dst`: exactly one stride, 4096-aligned, ONE positional read.
    pub fn read_record(&self, i: u64, dst: &mut [u8]) -> Result<(), XtierError> {
        let h = &self.header;
        if i >= h.records() {
            return Err(XtierError::Range {
                index: i,
                records: h.records(),
            });
        }
        if dst.len() as u64 != h.stride || dst.as_ptr() as usize % DIO_ALIGN != 0 {
            return Err(XtierError::Buffer(format!(
                "{} bytes at {:#x}; a record read needs exactly {} bytes at a 4096-aligned address",
                dst.len(),
                dst.as_ptr() as usize,
                h.stride
            )));
        }
        dio::preadv_exact(&self.file, dst, h.record_offset(i)).map_err(|source| XtierError::Io {
            path: self.path.clone(),
            source,
        })
    }

    /// Whether `bytes` (one stride) is record `i` by the table.
    #[must_use]
    pub fn verify(&self, i: u64, bytes: &[u8]) -> bool {
        match self.table.get(i as usize) {
            Some(want) => {
                let got: [u8; 32] = Sha256::digest(bytes).into();
                &got == want
            }
            None => false,
        }
    }
}

/// Geometry of a synthetic expert tier (the same format, small numbers).
#[derive(Debug, Clone, Copy)]
pub struct SyntheticTier {
    /// Owning rank.
    pub rank: u32,
    /// Layers.
    pub layers: u32,
    /// First owned expert id.
    pub first_expert: u32,
    /// Owned experts per layer.
    pub experts: u32,
    /// Hidden size (the sub-record lengths follow the real formulas).
    pub hidden: u32,
    /// Expert intermediate size.
    pub inter: u32,
}

/// The real record's sub-record layout for (hidden, inter): gate_up `w` `[2i, h/2]`, `ws`
/// `[h/32, 2i]`, `wref` `[2i]`, down `w` `[h, i/2]`, `ws` `[i/32, h]`, `wref` `[h]`, each
/// 256-aligned. Returns (sub, payload, stride). At (2560, 640) this is FN2's
/// (2,615,040 payload, 2,617,344 stride).
#[must_use]
pub fn record_layout(hidden: u64, inter: u64) -> ([(u64, u64); 6], u64, u64) {
    let lens = [
        2 * inter * hidden / 2,
        (hidden / 32) * 2 * inter,
        2 * inter,
        hidden * inter / 2,
        (inter / 32) * hidden,
        hidden,
    ];
    let mut sub = [(0u64, 0u64); 6];
    let mut off = 0u64;
    for (i, &len) in lens.iter().enumerate() {
        sub[i] = (off, len);
        off = (off + len).div_ceil(SUB_ALIGN) * SUB_ALIGN;
    }
    let payload = sub[5].0 + sub[5].1;
    let stride = payload.div_ceil(DIO_ALIGN as u64) * DIO_ALIGN as u64;
    (sub, payload, stride)
}

/// FN144: the v2 NVFP4 record layout for experts of `hidden` x `inter`: gate_up `w` `[2I, H/2]`,
/// `ws` E4M3 `[H/16, 2I]`, `gs` 2 x f32 (gate, up); down `w` `[H, I/2]`, `ws` `[I/16, H]`, `gs`
/// 1 x f32; each sub-record 256-aligned; payload = the end of the last; stride 4 KiB-rounded
/// (2560 x 640: 2,768,896 B = 676 pages, against MXFP4's 639).
#[must_use]
pub fn record_layout_nvfp4(hidden: u64, inter: u64) -> ([(u64, u64); 6], u64, u64) {
    let lens = [
        2 * inter * hidden / 2,
        (hidden / 16) * 2 * inter,
        8,
        hidden * inter / 2,
        (inter / 16) * hidden,
        4,
    ];
    let mut sub = [(0u64, 0u64); 6];
    let mut off = 0u64;
    for (i, &len) in lens.iter().enumerate() {
        sub[i] = (off, len);
        off = (off + len).div_ceil(SUB_ALIGN) * SUB_ALIGN;
    }
    let payload = sub[5].0 + sub[5].1;
    let stride = payload.div_ceil(DIO_ALIGN as u64) * DIO_ALIGN as u64;
    (sub, payload, stride)
}

/// The deterministic payload byte of record `i` at payload offset `o` of a synthetic tier: a
/// mix of (rank, record, offset) so that two records never agree and a wrong-record read is
/// caught by content as well as by the table.
#[must_use]
pub fn synthetic_byte(rank: u32, i: u64, o: u64) -> u8 {
    let x = (u64::from(rank) << 56)
        ^ (i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        ^ o.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let x = (x ^ (x >> 31)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (x >> 29) as u8
}

/// Write a synthetic v1 tier file at `path` (buffered I/O; the reader may open it `O_DIRECT`).
/// Returns the header written.
pub fn write_synthetic(path: &Path, t: SyntheticTier) -> io::Result<XtierHeader> {
    use std::io::{Seek, SeekFrom, Write};
    let (sub, payload, stride) = record_layout(u64::from(t.hidden), u64::from(t.inter));
    let records = u64::from(t.layers) * u64::from(t.experts);
    let table_off = XTIER_HEADER;
    let table_bytes = records * 32;
    let data_off = (table_off + table_bytes).div_ceil(DIO_ALIGN as u64) * DIO_ALIGN as u64;
    let file_bytes = data_off + records * stride;
    let mut f = File::create(path)?;
    let mut table = Vec::with_capacity(table_bytes as usize);
    f.seek(SeekFrom::Start(data_off))?;
    let mut rec = vec![0u8; stride as usize];
    for i in 0..records {
        for (o, b) in rec.iter_mut().enumerate().take(payload as usize) {
            *b = synthetic_byte(t.rank, i, o as u64);
        }
        f.write_all(&rec)?;
        let d: [u8; 32] = Sha256::digest(&rec).into();
        table.extend_from_slice(&d);
    }
    let h = XtierHeader {
        rank: t.rank,
        layers: t.layers,
        first_expert: t.first_expert,
        experts: t.experts,
        stride,
        payload,
        sub,
        hidden: t.hidden,
        inter: t.inter,
        table_off,
        table_bytes,
        data_off,
        file_bytes,
        table_sha256: table_digest(&table),
        plan_sha256: [0; 32],
        split_parts: 0,
        split_index: 0,
        format: XtierFormat::Mxfp4E8m0,
    };
    f.seek(SeekFrom::Start(0))?;
    f.write_all(&header_block(&h))?;
    f.seek(SeekFrom::Start(table_off))?;
    f.write_all(&table)?;
    f.set_len(file_bytes)?;
    f.sync_all()?;
    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fn144_v2_nvfp4_header_round_trips_and_v1_stays_v1() {
        let (sub, payload, stride) = record_layout_nvfp4(2560, 640);
        assert_eq!(stride, 676 * 4096);
        assert_eq!(payload, 2_765_060);
        assert_eq!(sub[1], (1_638_400, 204_800));
        assert_eq!(sub[2], (1_843_200, 8));
        assert_eq!(sub[3], (1_843_456, 819_200));
        assert_eq!(sub[4], (2_662_656, 102_400));
        assert_eq!(sub[5], (2_765_056, 4));
        let records = 2u64 * 256;
        let data_off = (XTIER_HEADER + records * 32).div_ceil(4096) * 4096;
        let mut h = XtierHeader {
            rank: 1,
            layers: 2,
            first_expert: 256,
            experts: 256,
            stride,
            payload,
            sub,
            hidden: 2560,
            inter: 640,
            table_off: XTIER_HEADER,
            table_bytes: records * 32,
            data_off,
            file_bytes: data_off + records * stride,
            table_sha256: [7; 32],
            plan_sha256: [9; 32],
            split_parts: 0,
            split_index: 0,
            format: XtierFormat::Nvfp4E4m3G16,
        };
        let b = header_block(&h);
        assert_eq!(u32_at(&b, 8), 2);
        assert_eq!(u32_at(&b, XTIER_FORMAT_OFF), 2);
        let back = XtierHeader::parse("v2", &b).unwrap();
        assert_eq!(back, h);
        assert!(back.require_mxfp4().unwrap_err().contains("NVFP4_E4M3G16"));
        // FN150: the format check routes it to the NVFP4 bodies, and only it
        assert!(back.require_format(XtierFormat::Nvfp4E4m3G16).is_ok());
        assert!(back
            .require_format(XtierFormat::Mxfp4E8m0)
            .unwrap_err()
            .contains("--fn-experts-fmt nvfp4"));
        // An unknown tag and an unknown version are refused by name.
        let mut bad = b.clone();
        bad[XTIER_FORMAT_OFF] = 9;
        assert!(XtierHeader::parse("t", &bad)
            .unwrap_err()
            .to_string()
            .contains("format tag 9"));
        bad[8] = 3;
        assert!(XtierHeader::parse("t", &bad)
            .unwrap_err()
            .to_string()
            .contains("version 3"));
        // MXFP4 is written as v1 with no tag: the pre-FN144 bytes.
        let (s1, p1, st1) = record_layout(2560, 640);
        h.sub = s1;
        h.payload = p1;
        h.stride = st1;
        h.file_bytes = data_off + records * st1;
        h.format = XtierFormat::Mxfp4E8m0;
        let b1 = header_block(&h);
        assert_eq!(u32_at(&b1, 8), 1);
        assert!(b1[XTIER_FORMAT_OFF..].iter().all(|&x| x == 0));
        let back1 = XtierHeader::parse("v1", &b1).unwrap();
        assert_eq!(back1.format, XtierFormat::Mxfp4E8m0);
        assert!(back1.require_mxfp4().is_ok());
        assert!(back1
            .require_format(XtierFormat::Nvfp4E4m3G16)
            .unwrap_err()
            .contains("MXFP4_E8M0"));
    }

    #[test]
    fn the_real_record_layout_is_fn2s() {
        let (sub, payload, stride) = record_layout(2560, 640);
        assert_eq!(payload, 2_615_040);
        assert_eq!(stride, 2_617_344);
        assert_eq!(stride, 639 * 4096);
        assert_eq!(
            sub,
            [
                (0, 1_638_400),
                (1_638_400, 102_400),
                (1_740_800, 1_280),
                (1_742_080, 819_200),
                (2_561_280, 51_200),
                (2_612_480, 2_560)
            ]
        );
    }

    /// Under `target/`, removed when the returned guard drops -- never `/tmp` (FN51h).
    fn tmp(name: &str) -> (saber_test_scratch::ScratchDir, std::path::PathBuf) {
        let d = saber_test_scratch::ScratchDir::new(&format!("saber-xtier-{name}"));
        let p = d.join("experts.xtier");
        (d, p)
    }

    #[test]
    fn synthetic_roundtrip_and_refusals() {
        let (_d, p) = tmp("rt");
        let t = SyntheticTier {
            rank: 1,
            layers: 3,
            first_expert: 16,
            experts: 16,
            hidden: 256,
            inter: 64,
        };
        let h = write_synthetic(&p, t).unwrap();
        let f = XtierFile::open(&p, OpenMode::Buffered).unwrap();
        assert_eq!(f.header(), &h);
        assert_eq!(h.index(2, 16 + 5), Some(2 * 16 + 5));
        assert_eq!(h.index(0, 15), None, "rank 1 does not own expert 15");
        let mut buf = dio::AlignedBuf::zeroed(h.stride as usize);
        for i in [0, 7, h.records() - 1] {
            f.read_record(i, &mut buf).unwrap();
            assert!(f.verify(i, &buf), "record {i}");
            assert!(
                !f.verify(i ^ 1, &buf),
                "record {i} must not verify as {}",
                i ^ 1
            );
            assert_eq!(buf[3], synthetic_byte(1, i, 3));
        }
        assert!(matches!(
            f.read_record(h.records(), &mut buf),
            Err(XtierError::Range { .. })
        ));
        // A corrupted table byte: refused at open.
        let mut raw = std::fs::read(&p).unwrap();
        raw[XTIER_HEADER as usize + 5] ^= 1;
        let (_dq, q) = tmp("bad");
        std::fs::write(&q, &raw).unwrap();
        assert!(matches!(
            XtierFile::open(&q, OpenMode::Buffered),
            Err(XtierError::Format { .. })
        ));
        // A wrong stride: refused by the geometry law.
        let mut raw = std::fs::read(&p).unwrap();
        raw[32] ^= 0x10;
        std::fs::write(&q, &raw).unwrap();
        let e = XtierFile::open(&q, OpenMode::Buffered).unwrap_err();
        assert!(
            format!("{e}").contains("stride") || format!("{e}").contains("file_bytes"),
            "{e}"
        );
    }
}
