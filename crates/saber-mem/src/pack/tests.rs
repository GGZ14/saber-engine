//! Round trips and the negative controls.
//!
//! The synthetic packs are written under `<repo>/target/saber-mem-testpacks/` rather than
//! `std::env::temp_dir()` on purpose: `/tmp` is tmpfs on this box and tmpfs does not implement
//! `O_DIRECT`, so a test that used it would silently exercise a different read path from the one
//! the loader ships.

use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::buffers::PageAlignedPool;
use super::writer::{
    write_pack, write_pack_chunks, write_pack_with_faults, ChunkPolicy, PackItem, WriterFaults,
};
use super::*;
use crate::dio::OpenMode;

const PLAN_A: [u8; 32] = [0xA1; 32];
const PLAN_B: [u8; 32] = [0xB2; 32];
const TIER: [u8; 32] = [0x7C; 32];

static SEQ: AtomicU64 = AtomicU64::new(0);

/// A scratch directory on a filesystem that supports `O_DIRECT`.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let root = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../target"))
            .join("saber-mem-testpacks");
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = root.join(format!("{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Scratch(dir)
    }

    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn items() -> Vec<PackItem> {
    vec![
        PackItem {
            name: "model.layers.0.mlp.gate_proj.weight_packed".into(),
            dtype: Dtype::U8,
            shape: vec![64, 32],
            shard_kind: ShardKind::Dim0,
            layout_id: LAYOUT_RAW,
            data: (0..64u32 * 32).map(|i| (i % 251) as u8).collect(),
        },
        PackItem {
            name: "model.layers.0.mlp.gate_proj.weight_scale".into(),
            dtype: Dtype::F8E4M3,
            shape: vec![64, 4],
            shard_kind: ShardKind::Dim0,
            layout_id: LAYOUT_RAW,
            data: (0..64u32 * 4).map(|i| (i % 97) as u8).collect(),
        },
        PackItem {
            name: "model.layers.0.input_layernorm.weight".into(),
            dtype: Dtype::BF16,
            shape: vec![5120],
            shard_kind: ShardKind::Replicated,
            layout_id: LAYOUT_RAW,
            data: vec![0x3F; 5120 * 2],
        },
        PackItem::raw_u8("tiny", vec![7u8; 3]),
    ]
}

fn build(dir: &Scratch, name: &str) -> PathBuf {
    let path = dir.file(name);
    write_pack(&path, 1, 0, PLAN_A, TIER, &items()).expect("write pack");
    path
}

#[test]
fn header_and_record_round_trip() {
    let h = PackHeader {
        format_version: FORMAT_VERSION,
        record_count: 3,
        table_off: HEADER_BYTES as u64,
        name_blob_off: HEADER_BYTES as u64 + 3 * RECORD_BYTES as u64,
        name_blob_bytes: 42,
        data_off: 4096,
        file_bytes: 8192,
        tp_size: 2,
        rank: 1,
        build_unix_seconds: 1_789_000_000,
        plan_hash: PLAN_A,
        tier_id: TIER,
        table_sha256: [9u8; 32],
        chunk_blob_off: 0,
        chunk_blob_bytes: 0,
        chunk_blob_sha256: [0u8; 32],
        chunk_bytes: 0,
    };
    assert_eq!(PackHeader::decode(&h.encode()).unwrap(), h);

    let r = PackRecord {
        name_off: 2,
        name_len: 5,
        rank: 1,
        layout_id: 0,
        dtype: Dtype::FP4E2M1X2,
        shard_kind: ShardKind::Dim1,
        data_off: 4096,
        length: 2048,
        shape: vec![64, 64],
        sha256: [3u8; 32],
        flags: 0,
    };
    assert_eq!(PackRecord::decode(&r.encode(), 0).unwrap(), r);
    assert_eq!(r.implied_length(), 2048, "fp4: 4 bits per element");
    assert_eq!(r.padded_length(), 4096);
}

#[test]
fn round_trip_read_and_verify() {
    let dir = Scratch::new("roundtrip");
    let path = build(&dir, "rank0.pack");
    let pack = PackReader::open(&path).expect("open");

    assert_eq!(pack.header().record_count, 4);
    assert_eq!(pack.header().tp_size, 1);
    assert_eq!(pack.header().plan_hash, PLAN_A);
    assert_eq!(pack.records().len(), 4);
    assert_eq!(pack.name(0), "model.layers.0.mlp.gate_proj.weight_packed");
    assert_eq!(pack.index_of("tiny"), Some(3));
    assert_eq!(pack.records()[2].shape, vec![5120]);
    assert_eq!(pack.records()[2].length, 10240);

    for (i, it) in items().iter().enumerate() {
        assert_eq!(pack.records()[i].data_off % ALIGN, 0);
        let buf = pack.read_record(i).unwrap();
        assert_eq!(&buf[..], &it.data[..], "record {i} bytes");
        assert_eq!(
            buf.as_ptr() as usize % 4096,
            0,
            "record {i} buffer alignment"
        );
    }
    pack.verify_all().expect("verify_all");
    pack.check_plan_hash(&PLAN_A).expect("plan hash");
}

#[test]
fn stream_delivers_every_record_in_both_orders() {
    let dir = Scratch::new("stream");
    let path = build(&dir, "rank0.pack");
    let expect = items();

    for (order, qd) in [
        (DeliveryOrder::Table, 1),
        (DeliveryOrder::Table, 4),
        (DeliveryOrder::Completion, 4),
    ] {
        let cfg = StreamConfig {
            queue_depth: qd,
            order,
            expect_plan_hash: Some(PLAN_A),
            ..StreamConfig::default()
        };
        let stream = PackStream::with_config(&path, cfg).expect("open stream");
        let pool = PageAlignedPool::new(2 << 20);
        let mut seen = Vec::new();
        let stats = stream
            .run(
                &pool,
                &mut Borrowed(|rec: RecordRef<'_>| {
                    assert_eq!(rec.bytes, &expect[rec.index].data[..], "{}", rec.name);
                    assert_eq!(rec.bytes.as_ptr() as usize % 4096, 0);
                    assert_eq!(rec.record.length as usize, rec.bytes.len());
                    seen.push(rec.index);
                    Ok(())
                }),
            )
            .expect("stream");
        assert_eq!(stats.records, 4);
        assert_eq!(
            stats.payload_bytes,
            expect.iter().map(|i| i.data.len() as u64).sum::<u64>()
        );
        if order == DeliveryOrder::Table {
            assert_eq!(seen, vec![0, 1, 2, 3], "table order");
        } else {
            seen.sort_unstable();
            assert_eq!(seen, vec![0, 1, 2, 3], "completion order covers everything");
        }
    }
}

#[test]
fn stream_survives_a_budget_smaller_than_one_record() {
    // The budget is raised to fit the largest record; a too-small budget must not deadlock.
    let dir = Scratch::new("budget");
    let path = build(&dir, "rank0.pack");
    let cfg = StreamConfig {
        queue_depth: 4,
        budget_bytes: 1,
        ..StreamConfig::default()
    };
    let stream = PackStream::with_config(&path, cfg).expect("open stream");
    let pool = PageAlignedPool::new(1);
    let stats = stream
        .run(&pool, &mut Borrowed(|_: RecordRef<'_>| Ok(())))
        .expect("stream");
    assert_eq!(stats.records, 4);
    assert!(stats.budget_bytes >= stream.reader().max_padded_record());
    assert!(stats.peak_inflight_bytes <= stats.budget_bytes);
}

#[test]
fn callback_error_aborts_the_stream() {
    let dir = Scratch::new("cberr");
    let path = build(&dir, "rank0.pack");
    let stream = PackStream::new(&path, 2).expect("open stream");
    let pool = PageAlignedPool::new(2 << 20);
    let err = stream
        .run(
            &pool,
            &mut Borrowed(|rec: RecordRef<'_>| {
                if rec.index == 1 {
                    Err(PackError::Config("stop here".into()))
                } else {
                    Ok(())
                }
            }),
        )
        .expect_err("callback error must propagate");
    assert!(matches!(err, PackError::Config(_)), "{err}");
}

// ---------------------------------------------------------------------------------------------
// Negative controls. Each one must fail, and fail for the reason it claims.
// ---------------------------------------------------------------------------------------------

#[test]
fn negative_control_flipped_data_byte_fails_verification() {
    let dir = Scratch::new("flip");
    let path = build(&dir, "rank0.pack");

    // Sanity: the pack is good before the flip.
    PackReader::open(&path).unwrap().verify_all().unwrap();

    let (target_off, target_index) = {
        let pack = PackReader::open(&path).unwrap();
        (pack.records()[1].data_off + 17, 1usize)
    };
    let mut byte = [0u8; 1];
    {
        use std::io::Read;
        let mut r = std::fs::File::open(&path).unwrap();
        r.seek(SeekFrom::Start(target_off)).unwrap();
        r.read_exact(&mut byte).unwrap();
    }
    {
        let mut f = OpenOptions::new().write(true).open(&path).unwrap();
        f.seek(SeekFrom::Start(target_off)).unwrap();
        f.write_all(&[byte[0] ^ 0x01]).unwrap();
        f.sync_all().unwrap();
    }

    // The table is untouched, so the pack still parses: corruption is a *content* fault.
    let pack = PackReader::open(&path).expect("header and table are still intact");
    let err = pack
        .verify_record(target_index)
        .expect_err("a flipped byte must fail verification");
    match err {
        PackError::RecordHashMismatch { index, .. } => assert_eq!(index, target_index),
        other => panic!("wrong error: {other}"),
    }
    assert!(pack.verify_all().is_err());
    // And the streaming path refuses it too.
    let pool = PageAlignedPool::new(2 << 20);
    let err = PackStream::new(&path, 4)
        .unwrap()
        .run(&pool, &mut Borrowed(|_: RecordRef<'_>| Ok(())))
        .expect_err("stream must refuse a corrupt record");
    assert!(matches!(err, PackError::RecordHashMismatch { .. }), "{err}");
}

#[test]
fn negative_control_wrong_plan_hash_is_refused() {
    let dir = Scratch::new("plan");
    let path = build(&dir, "rank0.pack");

    let err = PackReader::open_expecting(&path, OpenMode::Direct, &PLAN_B)
        .expect_err("a pack built for another plan must be refused");
    match err {
        PackError::PlanHashMismatch { found, expected } => {
            assert_eq!(found, hex::encode(PLAN_A));
            assert_eq!(expected, hex::encode(PLAN_B));
        }
        other => panic!("wrong error: {other}"),
    }

    // Same through the stream, before a single data byte is read.
    let cfg = StreamConfig {
        expect_plan_hash: Some(PLAN_B),
        ..StreamConfig::default()
    };
    assert!(matches!(
        PackStream::with_config(&path, cfg),
        Err(PackError::PlanHashMismatch { .. })
    ));

    // The right hash still opens.
    PackReader::open_expecting(&path, OpenMode::Direct, &PLAN_A).expect("matching plan hash");
}

#[test]
fn negative_control_unaligned_record_offset_is_rejected() {
    let dir = Scratch::new("unaligned");
    let path = dir.file("rank0.pack");
    write_pack_with_faults(
        &path,
        1,
        0,
        PLAN_A,
        TIER,
        &items(),
        WriterFaults {
            tamper_chunk_digest: false,
            wrong_chunk_blob_hash: false,
            chunks_describe_other_bytes: false,
            unaligned_first_record: true,
        },
    )
    .expect("write faulty pack");

    let err = PackReader::open(&path).expect_err("an unaligned record must be rejected");
    match err {
        PackError::UnalignedRecord {
            index, off, align, ..
        } => {
            assert_eq!(index, 0);
            assert_eq!(off % align, 1);
        }
        other => panic!(
            "wrong error: {other} (the table hash is recomputed over the bad table, \
                         so this must be the alignment rule, not a hash mismatch)"
        ),
    }
}

#[test]
fn a_truncated_file_is_rejected() {
    let dir = Scratch::new("trunc");
    let path = build(&dir, "rank0.pack");
    let f = OpenOptions::new().write(true).open(&path).unwrap();
    let len = f.metadata().unwrap().len();
    f.set_len(len - ALIGN).unwrap();
    f.sync_all().unwrap();
    let err = PackReader::open(&path).expect_err("truncation must be rejected");
    assert!(matches!(err, PackError::Malformed(_)), "{err}");
}

#[test]
fn a_foreign_file_is_rejected() {
    let dir = Scratch::new("magic");
    let path = dir.file("not-a-pack.bin");
    std::fs::write(&path, vec![0u8; 8192]).unwrap();
    let err = PackReader::open(&path).expect_err("magic must be checked");
    assert!(matches!(err, PackError::BadMagic { .. }), "{err}");
}

// =================================================================================================
// Chunk digests (v2) — ws-k7
// =================================================================================================

/// A chunk size small enough that the fixtures above straddle several chunks. The load path uses
/// 4 MiB; the format does not care, and the header carries whichever was used.
const TEST_CHUNK: u32 = 4096;

fn chunked(dir: &Scratch, name: &str) -> PathBuf {
    let path = dir.file(name);
    write_pack_chunks(
        &path,
        1,
        0,
        PLAN_A,
        TIER,
        &items(),
        ChunkPolicy::Over { bytes: TEST_CHUNK },
    )
    .expect("write chunked pack");
    path
}

#[test]
fn a_v2_pack_carries_chunk_digests_only_where_they_earn_their_place() {
    let dir = Scratch::new("chunks");
    let path = chunked(&dir, "rank0.pack");
    let pack = PackReader::open(&path).expect("open");

    assert_eq!(pack.chunk_bytes(), TEST_CHUNK);
    for (i, r) in pack.records().iter().enumerate() {
        let want_flag = r.length > u64::from(TEST_CHUNK);
        assert_eq!(
            r.has_chunk_digests(),
            want_flag,
            "record {i} ({}) is {} bytes",
            pack.name(i),
            r.length
        );
        match pack.chunk_digests(i) {
            Some(list) => {
                assert!(want_flag);
                assert_eq!(list.len(), r.chunk_count(TEST_CHUNK));
            }
            None => assert!(!want_flag, "record {i} has the flag but no list"),
        }
    }
    let (records_with, total) = pack.chunk_summary();
    assert_eq!(
        records_with, 1,
        "exactly one fixture record is longer than one 4096-byte chunk"
    );
    assert_eq!(total, 3, "and it is 10240 bytes, which is three chunks");

    // Verification passes through the chunk path, and the stream delivers everything.
    pack.verify_all().expect("verify_all");
    let pool = PageAlignedPool::new(2 << 20);
    let mut seen = 0usize;
    PackStream::new(&path, 4)
        .unwrap()
        .run(
            &pool,
            &mut Borrowed(|_: RecordRef<'_>| {
                seen += 1;
                Ok(())
            }),
        )
        .expect("stream");
    assert_eq!(seen, items().len());
}

#[test]
fn a_v1_pack_still_opens_and_reports_no_chunk_digests() {
    // The compatibility direction that matters: this build reads a pack written before v2 existed,
    // and behaves exactly as it did then.
    let dir = Scratch::new("v1compat");
    let path = build(&dir, "rank0.pack");
    let pack = PackReader::open(&path).expect("open a pack with no chunk blob");
    assert_eq!(pack.chunk_bytes(), 0);
    assert_eq!(pack.chunk_summary(), (0, 0));
    for i in 0..pack.records().len() {
        assert!(pack.chunk_digests(i).is_none());
        assert!(!pack.records()[i].has_chunk_digests());
    }
    pack.verify_all().expect("verify_all");
}

#[test]
fn negative_control_a_flip_inside_a_chunk_names_the_chunk() {
    let dir = Scratch::new("chunkflip");
    let path = chunked(&dir, "rank0.pack");
    PackReader::open(&path).unwrap().verify_all().unwrap();

    // Record 0 is 2048 bytes... pick the record that actually spans chunks, and a byte in its
    // SECOND chunk, so the reported chunk index is not trivially zero.
    let (target_index, target_off, want_chunk) = {
        let pack = PackReader::open(&path).unwrap();
        let i = pack
            .records()
            .iter()
            .position(|r| r.length > 2 * u64::from(TEST_CHUNK))
            .expect("a fixture record longer than two chunks");
        let r = &pack.records()[i];
        (i, r.data_off + u64::from(TEST_CHUNK) + 5, 1usize)
    };
    let mut byte = [0u8; 1];
    {
        use std::io::Read;
        let mut r = std::fs::File::open(&path).unwrap();
        r.seek(SeekFrom::Start(target_off)).unwrap();
        r.read_exact(&mut byte).unwrap();
    }
    {
        let mut f = OpenOptions::new().write(true).open(&path).unwrap();
        f.seek(SeekFrom::Start(target_off)).unwrap();
        f.write_all(&[byte[0] ^ 0x01]).unwrap();
        f.sync_all().unwrap();
    }

    let pack = PackReader::open(&path).expect("the table is untouched");
    let err = pack
        .verify_record(target_index)
        .expect_err("a flipped byte must fail verification");
    match err {
        PackError::RecordChunkMismatch { index, chunk, .. } => {
            assert_eq!(index, target_index);
            assert_eq!(chunk, want_chunk, "the fault must name the chunk it is in");
        }
        other => panic!("wrong error: {other}"),
    }
    // ...and the streaming path, which checks the chunks in parallel, refuses it the same way.
    let pool = PageAlignedPool::new(2 << 20);
    let err = PackStream::new(&path, 4)
        .unwrap()
        .run(&pool, &mut Borrowed(|_: RecordRef<'_>| Ok(())))
        .expect_err("stream must refuse a corrupt record");
    assert!(
        matches!(err, PackError::RecordChunkMismatch { .. }),
        "{err}"
    );
}

#[test]
fn negative_control_a_chunk_digest_that_does_not_match_its_bytes_is_refused() {
    // A chunk list that disagrees with the payload is caught by the chunk path itself -- the blob
    // hash covers the list, so what is inconsistent is the list against the bytes, and that is what
    // every verifier checks.
    let dir = Scratch::new("tamper");
    let path = dir.file("rank0.pack");
    write_pack_with_faults_chunked(
        &path,
        WriterFaults {
            tamper_chunk_digest: true,
            ..WriterFaults::default()
        },
    );
    let pack = PackReader::open(&path).expect("the blob is internally consistent, so it opens");
    let err = pack
        .verify_all()
        .expect_err("the list does not match the bytes");
    assert!(
        matches!(err, PackError::RecordChunkMismatch { .. }),
        "{err}"
    );
    let pool = PageAlignedPool::new(2 << 20);
    let err = PackStream::new(&path, 4)
        .unwrap()
        .run(&pool, &mut Borrowed(|_: RecordRef<'_>| Ok(())))
        .expect_err("the load path must refuse it too");
    assert!(
        matches!(err, PackError::RecordChunkMismatch { .. }),
        "{err}"
    );

    // ...and the control's control: the same writer with the fault off verifies clean.
    let path = dir.file("good.pack");
    write_pack_with_faults_chunked(&path, WriterFaults::default());
    PackReader::open(&path)
        .expect("open")
        .verify_all()
        .expect("verify");
}

#[test]
fn negative_control_two_descriptions_that_disagree_are_caught_by_the_offline_verifier() {
    // The one fault the load path cannot see, stated as a test rather than as a hope. The chunk
    // list describes the bytes in the file; the record `sha256` describes bytes the file does not
    // hold. Every chunk checks out, so the load path is happy -- and `verify_record`, which hashes
    // the payload a second time on purpose, is not.
    let dir = Scratch::new("disagree");
    let path = dir.file("rank0.pack");
    write_pack_with_faults_chunked(
        &path,
        WriterFaults {
            chunks_describe_other_bytes: true,
            ..WriterFaults::default()
        },
    );
    let pack = PackReader::open(&path).expect("structurally fine");

    let flagged = pack
        .records()
        .iter()
        .position(super::format::PackRecord::has_chunk_digests)
        .expect("a flagged record");
    let buf = pack.read_record(flagged).expect("read");
    pack.check_hash(flagged, &buf)
        .expect("the chunk list matches the bytes that are there");
    let err = pack
        .check_hash_strict(flagged, &buf)
        .expect_err("the record digest does not");
    assert!(matches!(err, PackError::RecordHashMismatch { .. }), "{err}");

    // `verify_record` is the strict one, so the offline verifier refuses the pack.
    let err = pack.verify_all().expect_err("verify_all must refuse it");
    assert!(matches!(err, PackError::RecordHashMismatch { .. }), "{err}");
}

#[test]
fn negative_control_a_blob_hash_that_names_another_blob_is_refused() {
    let dir = Scratch::new("bloghash");
    let path = dir.file("rank0.pack");
    write_pack_with_faults_chunked(
        &path,
        WriterFaults {
            wrong_chunk_blob_hash: true,
            ..WriterFaults::default()
        },
    );
    let err = PackReader::open(&path).expect_err("a wrong blob hash must be refused");
    assert!(
        matches!(&err, PackError::Malformed(m) if m.contains("chunk blob hashes to")),
        "{err}"
    );
}

fn write_pack_with_faults_chunked(path: &std::path::Path, faults: WriterFaults) {
    super::writer::write_pack_chunked(
        path,
        1,
        0,
        PLAN_A,
        TIER,
        &items(),
        ChunkPolicy::Over { bytes: TEST_CHUNK },
        faults,
    )
    .expect("write");
}
