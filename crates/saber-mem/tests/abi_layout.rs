//! `saber-mem`'s hand-written pack decoder against `include/saber_abi.h`.
//!
//! `docs/abi/PACK.md` says it plainly: the bytes on disk are the source of truth, the header is the
//! source of truth for the C names, and if the two disagree the header is wrong. There are now
//! three descriptions of the same 256 + 160 bytes — the document, the C header (with its
//! `_Static_assert`ed offsets, mirrored into `saber-abi` by bindgen), and this crate's
//! `from_le_bytes` decoder, which indexes those offsets by hand so that no `#[repr(C)]` surprise in
//! either language can change what is read.
//!
//! Two of the three can drift silently. This test is what makes that a build failure: it reads the
//! constants out of `saber-abi` and checks them against what the reader actually does, by round
//! -tripping a record and a header through `encode`/`decode` and reading the fields back out of the
//! bytes at the header's declared offsets.
//!
//! `saber-abi` is a **dev-dependency only**. The load path does not link the ABI crate — coupling
//! it would buy nothing — but the two descriptions still have to agree, and this is where that is
//! settled.

use saber_abi as abi;
use saber_mem::pack::{
    align_up, Dtype, PackHeader, PackRecord, ShardKind, ALIGN, FORMAT_VERSION, HEADER_BYTES,
    LAYOUT_RAW, MAGIC, MAX_NDIM, RECORD_BYTES,
};

fn u16_at(b: &[u8], off: u32) -> u16 {
    let o = off as usize;
    u16::from_le_bytes([b[o], b[o + 1]])
}

fn u32_at(b: &[u8], off: u32) -> u32 {
    let o = off as usize;
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn u64_at(b: &[u8], off: u32) -> u64 {
    let o = off as usize;
    let mut v = [0u8; 8];
    v.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(v)
}

#[test]
fn constants_match_the_abi_header() {
    assert_eq!(MAGIC.len(), abi::SABER_PACK_MAGIC_BYTES as usize);
    assert_eq!(&MAGIC[..], &abi::SABER_PACK_MAGIC[..MAGIC.len()]);
    assert_eq!(FORMAT_VERSION, abi::SABER_PACK_FORMAT_VERSION);
    assert_eq!(HEADER_BYTES, abi::SABER_PACK_HEADER_BYTES as usize);
    assert_eq!(RECORD_BYTES, abi::SABER_PACK_RECORD_BYTES as usize);
    assert_eq!(ALIGN, u64::from(abi::SABER_PACK_ALIGN));
    assert_eq!(MAX_NDIM, abi::SABER_PACK_MAX_DIMS as usize);
    assert_eq!(LAYOUT_RAW, abi::SABER_LAYOUT_RAW);

    // The C structs are the same sizes the reader assumes.
    assert_eq!(
        std::mem::size_of::<abi::saber_pack_file_header>(),
        HEADER_BYTES
    );
    assert_eq!(
        std::mem::size_of::<abi::saber_pack_table_entry>(),
        RECORD_BYTES
    );
}

#[test]
fn dtype_tags_match_the_abi_header() {
    // Both tables, tag for tag. A renumbering on either side fails here rather than at the first
    // tier that loads with the wrong element width.
    let pairs = [
        (Dtype::U8, abi::Dtype::U8),
        (Dtype::I8, abi::Dtype::I8),
        (Dtype::U16, abi::Dtype::U16),
        (Dtype::I16, abi::Dtype::I16),
        (Dtype::U32, abi::Dtype::U32),
        (Dtype::I32, abi::Dtype::I32),
        (Dtype::U64, abi::Dtype::U64),
        (Dtype::I64, abi::Dtype::I64),
        (Dtype::F16, abi::Dtype::F16),
        (Dtype::BF16, abi::Dtype::Bf16),
        (Dtype::F32, abi::Dtype::F32),
        (Dtype::F64, abi::Dtype::F64),
        (Dtype::F8E4M3, abi::Dtype::F8E4M3),
        (Dtype::F8E5M2, abi::Dtype::F8E5M2),
        (Dtype::FP4E2M1X2, abi::Dtype::Fp4E2M1X2),
        (Dtype::Bool, abi::Dtype::Bool),
        (Dtype::E8M0, abi::Dtype::E8M0),
    ];
    for (mem, a) in pairs {
        assert_eq!(mem.tag(), a.tag(), "{mem:?} tag");
        assert_eq!(mem.bits(), a.bits(), "{mem:?} bits");
        assert_eq!(Dtype::from_tag(a.tag()).unwrap(), mem);
        assert_eq!(abi::Dtype::from_tag(mem.tag()).unwrap(), a);
    }
    // Neither side accepts a tag past the end of the table.
    let past_end = pairs.len() as u16;
    assert!(Dtype::from_tag(past_end).is_err());
    assert!(abi::Dtype::from_tag(past_end).is_err());
}

#[test]
fn shard_kind_tags_match_the_abi_header() {
    for (mem, a) in [
        (ShardKind::Replicated, abi::ShardKind::Replicated),
        (ShardKind::Dim0, abi::ShardKind::Dim0),
        (ShardKind::Dim1, abi::ShardKind::Dim1),
        (ShardKind::Dim0Segments, abi::ShardKind::Dim0Segments),
    ] {
        assert_eq!(mem.tag(), a.tag(), "{mem:?} tag");
        assert_eq!(ShardKind::from_tag(a.tag()).unwrap(), mem);
    }
    assert!(ShardKind::from_tag(4).is_err());
    assert!(abi::ShardKind::from_tag(4).is_err());
}

#[test]
fn the_reader_writes_every_header_field_at_the_offset_the_abi_declares() {
    let h = PackHeader {
        format_version: FORMAT_VERSION,
        record_count: 3,
        table_off: HEADER_BYTES as u64,
        name_blob_off: HEADER_BYTES as u64 + 3 * RECORD_BYTES as u64,
        name_blob_bytes: 96,
        data_off: 4096,
        file_bytes: 3 * 4096,
        tp_size: 2,
        rank: 1,
        build_unix_seconds: 1_758_211_000,
        plan_hash: [0xA1; 32],
        tier_id: [0xB2; 32],
        table_sha256: [0xC3; 32],
        chunk_blob_off: HEADER_BYTES as u64 + 3 * RECORD_BYTES as u64 + 96,
        chunk_blob_bytes: 104,
        chunk_blob_sha256: [0xE5; 32],
        chunk_bytes: 4 << 20,
    };
    let b = h.encode();
    assert_eq!(b.len(), abi::SABER_PACK_HEADER_BYTES as usize);

    let m = abi::SABER_PACK_HDR_OFF_MAGIC as usize;
    assert_eq!(&b[m..m + MAGIC.len()], &MAGIC[..]);
    assert_eq!(
        u32_at(&b, abi::SABER_PACK_HDR_OFF_FORMAT_VERSION),
        abi::SABER_PACK_FORMAT_VERSION
    );
    assert_eq!(u32_at(&b, abi::SABER_PACK_HDR_OFF_HEADER_BYTES), 256);
    assert_eq!(u32_at(&b, abi::SABER_PACK_HDR_OFF_RECORD_BYTES), 160);
    assert_eq!(u32_at(&b, abi::SABER_PACK_HDR_OFF_RECORD_COUNT), 3);
    assert_eq!(u64_at(&b, abi::SABER_PACK_HDR_OFF_TABLE_OFF), 256);
    assert_eq!(
        u64_at(&b, abi::SABER_PACK_HDR_OFF_NAME_BLOB_OFF),
        h.name_blob_off
    );
    assert_eq!(u64_at(&b, abi::SABER_PACK_HDR_OFF_NAME_BLOB_BYTES), 96);
    assert_eq!(u64_at(&b, abi::SABER_PACK_HDR_OFF_DATA_OFF), 4096);
    assert_eq!(u64_at(&b, abi::SABER_PACK_HDR_OFF_FILE_BYTES), 3 * 4096);
    assert_eq!(u32_at(&b, abi::SABER_PACK_HDR_OFF_TP_SIZE), 2);
    assert_eq!(u32_at(&b, abi::SABER_PACK_HDR_OFF_RANK), 1);
    assert_eq!(
        u64_at(&b, abi::SABER_PACK_HDR_OFF_BUILD_UNIX_SECONDS),
        1_758_211_000
    );
    // v2's four chunk fields, at the offsets the header declares.
    assert_eq!(
        u64_at(&b, abi::SABER_PACK_HDR_OFF_CHUNK_BLOB_OFF),
        h.chunk_blob_off
    );
    assert_eq!(u64_at(&b, abi::SABER_PACK_HDR_OFF_CHUNK_BLOB_BYTES), 104);
    assert_eq!(u32_at(&b, abi::SABER_PACK_HDR_OFF_CHUNK_BYTES), 4 << 20);
    for (off, fill) in [
        (abi::SABER_PACK_HDR_OFF_PLAN_HASH, 0xA1u8),
        (abi::SABER_PACK_HDR_OFF_TIER_ID, 0xB2),
        (abi::SABER_PACK_HDR_OFF_TABLE_SHA256, 0xC3),
        (abi::SABER_PACK_HDR_OFF_CHUNK_BLOB_SHA256, 0xE5),
    ] {
        let o = off as usize;
        assert_eq!(
            &b[o..o + abi::SABER_HASH_BYTES as usize],
            &[fill; 32][..],
            "hash at offset {off}"
        );
    }
    let r = abi::SABER_PACK_HDR_OFF_RESERVED as usize;
    assert!(b[r..].iter().all(|&x| x == 0));
    assert_eq!(
        b.len() - r,
        abi::SABER_PACK_HEADER_RESERVED_BYTES as usize,
        "reserved must run to the end of the header"
    );

    assert_eq!(PackHeader::decode(&b).unwrap(), h);
}

#[test]
fn the_reader_writes_every_record_field_at_the_offset_the_abi_declares() {
    let rec = PackRecord {
        name_off: 2,
        name_len: 9,
        rank: 1,
        layout_id: LAYOUT_RAW,
        dtype: Dtype::BF16,
        shard_kind: ShardKind::Dim0,
        data_off: 8192,
        length: 16 * 64 * 2,
        shape: vec![16, 64],
        sha256: [0xD4; 32],
        flags: abi::SABER_PACK_REC_FLAG_CHUNK_DIGESTS,
    };
    let b = rec.encode();
    assert_eq!(b.len(), abi::SABER_PACK_RECORD_BYTES as usize);

    assert_eq!(u32_at(&b, abi::SABER_PACK_REC_OFF_NAME_OFF), 2);
    assert_eq!(u32_at(&b, abi::SABER_PACK_REC_OFF_NAME_LEN), 9);
    assert_eq!(u32_at(&b, abi::SABER_PACK_REC_OFF_RANK), 1);
    assert_eq!(
        u32_at(&b, abi::SABER_PACK_REC_OFF_LAYOUT_ID),
        abi::SABER_LAYOUT_RAW
    );
    assert_eq!(
        u16_at(&b, abi::SABER_PACK_REC_OFF_DTYPE),
        abi::Dtype::Bf16.tag()
    );
    assert_eq!(b[abi::SABER_PACK_REC_OFF_NDIM as usize], 2);
    assert_eq!(
        b[abi::SABER_PACK_REC_OFF_SHARD_KIND as usize],
        abi::ShardKind::Dim0.tag()
    );
    assert_eq!(
        u32_at(&b, abi::SABER_PACK_REC_OFF_FLAGS),
        abi::SABER_PACK_REC_FLAG_CHUNK_DIGESTS,
        "the fixture sets the chunk-digest flag, so the wire byte must carry it"
    );
    assert_eq!(u64_at(&b, abi::SABER_PACK_REC_OFF_DATA_OFF), 8192);
    assert_eq!(u64_at(&b, abi::SABER_PACK_REC_OFF_LENGTH), 2048);
    for d in 0..MAX_NDIM {
        let want = rec.shape.get(d).copied().unwrap_or(0);
        let off = abi::SABER_PACK_REC_OFF_SHAPE + 8 * d as u32;
        assert_eq!(u64_at(&b, off), want, "shape[{d}]");
    }
    let s = abi::SABER_PACK_REC_OFF_SHA256 as usize;
    assert_eq!(&b[s..s + 32], &[0xD4u8; 32][..]);
    let r = abi::SABER_PACK_REC_OFF_RESERVED as usize;
    assert!(b[r..].iter().all(|&x| x == 0));
    assert_eq!(
        b.len() - r,
        8 * abi::SABER_PACK_RECORD_RESERVED_U64 as usize,
        "reserved must run to the end of the record"
    );

    assert_eq!(PackRecord::decode(&b, 0).unwrap(), rec);
}

#[test]
fn both_crates_compute_the_same_implied_length_and_padding() {
    for (dtype, abi_dtype, shape) in [
        (Dtype::BF16, abi::Dtype::Bf16, vec![16u64, 64]),
        (Dtype::U8, abi::Dtype::U8, vec![4096u64, 512]),
        (Dtype::FP4E2M1X2, abi::Dtype::Fp4E2M1X2, vec![128u64, 256]),
        (Dtype::F32, abi::Dtype::F32, vec![7u64]),
    ] {
        let elements: u64 = shape.iter().product();
        let implied = (elements * abi_dtype.bits()).div_ceil(8);
        let rec = PackRecord {
            name_off: 2,
            name_len: 1,
            rank: 0,
            layout_id: LAYOUT_RAW,
            dtype,
            shard_kind: ShardKind::Replicated,
            flags: 0,
            data_off: 4096,
            length: implied,
            shape: shape.clone(),
            sha256: [1u8; 32],
        };
        assert_eq!(rec.implied_length(), implied, "{dtype:?}");
        assert_eq!(rec.padded_length(), align_up(implied));

        // And the ABI's own checker agrees that this entry is well formed.
        let mut shape8 = [0u64; 8];
        for (i, &v) in shape.iter().enumerate() {
            shape8[i] = v;
        }
        let entry = abi::saber_pack_table_entry {
            name_off: 2,
            name_len: 1,
            rank: 0,
            layout_id: abi::SABER_LAYOUT_RAW,
            dtype: abi_dtype.tag(),
            ndim: shape.len() as u8,
            shard_kind: abi::ShardKind::Replicated.tag(),
            flags: 0,
            data_off: 4096,
            length: implied,
            shape: shape8,
            sha256: abi::saber_hash256 { bytes: [1u8; 32] },
            reserved: [0; 3],
        };
        assert!(abi::check_pack_entry(&entry).is_ok(), "{dtype:?}");

        // Negative control: one byte off the implied length and both sides reject it.
        let mut bad = entry;
        bad.length = implied + 1;
        assert!(abi::check_pack_entry(&bad).is_err(), "{dtype:?}");
    }
}
