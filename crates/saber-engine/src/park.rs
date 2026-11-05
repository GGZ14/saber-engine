//! The engine's parkable session image: one sequence's KV pages and GDN state slot, as bytes.
//!
//! # What a session image is here
//!
//! A sequence's device state is two things and nothing else: the **KV pages** the scheduler
//! allocated it, and the **GDN state slot** it was given. Both are addresses into pools the engine
//! owns, and neither survives the sequence — a parked session hands both back, and a restored one
//! gets whichever pages and slot are free when it is re-admitted. So the image is the *contents* of
//! those regions, in a fixed order, plus what it takes to put them somewhere else:
//! [`saber_mem::park::ImageDigests`] (the layout and a `sha256` per 4 MiB chunk) and the payload.
//!
//! That ordering is the whole design. `ARCHITECTURE.md` § Memory plane: *"evict an idle session's
//! KV and its GDN state as one unit — they are useless apart."* One image, one digest, one
//! decision.
//!
//! # Why it is a byte blob at the seam
//!
//! [`saber_sched::StepExecutor::park_state`] returns a `SeqState(Vec<u8>)`, because the scheduler
//! must be able to hand a session to a tier without understanding it. So the metadata travels
//! *inside* the blob: [`ParkImage::encode`] writes a fixed header, the region layout and the chunk
//! digests ahead of the payload, and [`ParkImage::decode`] refuses anything it cannot account for
//! byte by byte. The scheduler's tier then wraps the whole blob in its own record with its own
//! digest (`saber_sched::TieredParkStore`), so the bytes are covered on both hops: VRAM↔host by
//! these digests, host↔tier by the tier's.
//!
//! # The invariant a restore holds
//!
//! A restore's destinations are a **different** set of pages and, usually, a different slot. What
//! must match is the *shape*: the same number of regions, in the same order, of the same kinds and
//! lengths. `saber_mem::park::publish_from` checks exactly that before it issues anything, and
//! verifies each chunk before sending it (I3, in the chunk-at-a-time form the K6 ledger entry
//! records).
//!
//! # ...and the one thing the shape does not say (ws-k14)
//!
//! **Which slot of the run each GDN region belongs in.** A sequence owns `max_width` consecutive
//! state slots; the conv cache lives in the run's base and the recurrent snapshot in
//! `base + naccept - 1`, because the decode kernels write row `t` into `sidx[n*stride + t]` and
//! open at `sidx[n*stride + naccept - 1]` (`docs/EXEC.md` § 2). Two images of two sessions have
//! the same SHAPE and different offsets, so the offset has to travel: [`ParkImage::committed`] is
//! that number, and it is why this format is v4.

use saber_mem::park::{ImageDigests, RegionKind, RegionSpec};

/// Why an image could not be encoded or decoded.
#[derive(Debug, thiserror::Error)]
pub enum ImageError {
    /// The blob is not one of ours, or not this version.
    #[error("not a session image: {0}")]
    NotAnImage(String),
    /// The blob's header and its length disagree.
    #[error("session image is malformed: {0}")]
    Malformed(String),
}

/// `SBRPARK1`, so a blob read from the wrong place is a refusal rather than a decode.
const MAGIC: [u8; 8] = *b"SBRPARK1";
// 3 since ws-s5: a session's image may now carry the DRAFTER's own pages as a third region kind,
// so a build that does not know `RegionKind::DraftKv` must refuse an image that has one rather
// than decode its tags as something else. The version check is the refusal.
// 4 since ws-k14: a session's GDN regions are its LIVE slots, which for the recurrent state is
// `base + committed - 1` and not the base, so the image has to say what `committed` was or a
// restore cannot place them. A v3 blob's regions are all at the base and would be published at the
// wrong offsets by a v4 reader; the version check is the refusal.
const VERSION: u32 = 4;
/// magic + version + flags + n_regions + n_chunks + chunk_bytes + resident + committed +
/// payload_bytes.
const HEADER: usize = 8 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 8;
/// One region: kind tag + length.
const REGION_BYTES: usize = 4 + 8;

/// One sequence's device state, ready to leave the engine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParkImage {
    /// The layout, the chunk digests and the folded value.
    pub digests: ImageDigests,
    /// Tokens this session already had resident when it was parked.
    ///
    /// **Part of the session, not metadata about it.** A restored sequence gets fresh pages and a
    /// fresh slot, and the engine's own count of what is in them starts at zero; without this the
    /// first step after a restore is refused with "the batch says N token(s) are resident and the
    /// engine has 0". Found on the card by `saber engine --park-staging-mib`, not on the host — the
    /// stub keeps its own count inside the payload, so the host gates could not see it.
    pub resident: u32,
    /// Tokens this session's LAST step committed when it was parked, `1..=max_width`.
    ///
    /// **Part of the session, and the one number that places its GDN regions** (ws-k14): the
    /// recurrent snapshot the next step opens at is `sidx[n * stride + committed - 1]`, so the
    /// image carries the slot at that offset and a restore has to publish it at the same offset
    /// into the NEW run. Without it a restored session reads a slot nothing wrote — green while
    /// the restore landed back on its own slots, and a different token the first time it moved
    /// (LEDGER, T2 item 2).
    pub committed: u32,
    /// The bytes, exactly `digests.bytes` of them.
    pub bytes: Vec<u8>,
}

fn kind_tag(k: RegionKind) -> u32 {
    match k {
        RegionKind::Kv => 0,
        RegionKind::GdnState => 1,
        RegionKind::DraftKv => 2,
    }
}

fn kind_of(tag: u32) -> Result<RegionKind, ImageError> {
    match tag {
        0 => Ok(RegionKind::Kv),
        1 => Ok(RegionKind::GdnState),
        2 => Ok(RegionKind::DraftKv),
        other => Err(ImageError::Malformed(format!("region kind {other}"))),
    }
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().expect("4 bytes"))
}

fn u64_at(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().expect("8 bytes"))
}

/// Bytes the wire form of an image with these digests and this payload length takes.
#[must_use]
pub fn encoded_len_of(digests: &ImageDigests, payload: usize) -> usize {
    HEADER + REGION_BYTES * digests.layout.len() + 32 * digests.chunks.len() + 32 + payload
}

/// Bytes of header — everything before the payload — for a layout of `regions` and `chunks`.
///
/// Known at the moment a park is **issued**, before a byte has been hashed: the layout comes from the
/// session and the chunk count is `ceil(total / DIGEST_CHUNK)`. That is what lets the blob be built
/// incrementally, payload first, with the header filled in at the end.
#[must_use]
pub fn header_len(regions: usize, chunks: usize) -> usize {
    HEADER + REGION_BYTES * regions + 32 * chunks + 32
}

/// Write the header of an image whose payload is **already** at `out[header_len(..)..]`.
///
/// The counterpart of [`encode_into`] for a caller that streamed the payload in as it landed. The
/// first cut built the whole blob in one `encode_into` at the end of the last poll, which is one
/// full-image host memcpy — 80.26 ms on a 403 MiB image, measured (K9), in a single poll on the step
/// thread. This writes only the front of it.
pub fn write_header(
    digests: &ImageDigests,
    resident: u32,
    committed: u32,
    payload: usize,
    out: &mut [u8],
) {
    let n = header_len(digests.layout.len(), digests.chunks.len());
    assert!(
        out.len() >= n,
        "an image header needs {n} bytes and it was given {}",
        out.len()
    );
    let mut at = 0usize;
    let mut put = |bytes: &[u8]| {
        out[at..at + bytes.len()].copy_from_slice(bytes);
        at += bytes.len();
    };
    put(&MAGIC);
    put(&VERSION.to_le_bytes());
    put(&0u32.to_le_bytes()); // flags
    put(&(digests.layout.len() as u32).to_le_bytes());
    put(&(digests.chunks.len() as u32).to_le_bytes());
    put(&digests.chunk_bytes.to_le_bytes());
    put(&resident.to_le_bytes());
    put(&committed.to_le_bytes());
    put(&(payload as u64).to_le_bytes());
    for r in &digests.layout {
        put(&kind_tag(r.kind).to_le_bytes());
        put(&r.len.to_le_bytes());
    }
    for d in &digests.chunks {
        put(d);
    }
    put(&digests.sha256);
    debug_assert_eq!(at, n, "the header writer and `header_len` disagree");
}

/// Write the wire form of `digests` + `payload` into `out`.
///
/// Takes the payload as a **borrow** so a caller that already holds the bytes somewhere — the
/// engine's pinned park-staging buffer — can build the blob in one copy instead of two. The first
/// cut of this path went staging -> `ParkImage::bytes` -> `encode()`, which is two full-image host
/// memcpys per park; at 403 MiB that is real time on the step thread for nothing.
pub fn encode_into(
    digests: &ImageDigests,
    resident: u32,
    committed: u32,
    payload: &[u8],
    out: &mut Vec<u8>,
) {
    out.clear();
    out.reserve(encoded_len_of(digests, payload.len()));
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // flags
    out.extend_from_slice(&(digests.layout.len() as u32).to_le_bytes());
    out.extend_from_slice(&(digests.chunks.len() as u32).to_le_bytes());
    out.extend_from_slice(&digests.chunk_bytes.to_le_bytes());
    out.extend_from_slice(&resident.to_le_bytes());
    out.extend_from_slice(&committed.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    for r in &digests.layout {
        out.extend_from_slice(&kind_tag(r.kind).to_le_bytes());
        out.extend_from_slice(&r.len.to_le_bytes());
    }
    for d in &digests.chunks {
        out.extend_from_slice(d);
    }
    out.extend_from_slice(&digests.sha256);
    out.extend_from_slice(payload);
}

/// The payload of an encoded image, checked the way [`ParkImage::decode`] checks it.
///
/// A restore copies the payload into pinned staging and then publishes from there, so it never
/// needs the owned `Vec<u8>` a full decode builds. Returns the digests and a borrow of the bytes.
pub fn split_encoded(b: &[u8]) -> Result<(ImageDigests, ImageCounts, &[u8]), ImageError> {
    let (digests, counts) = ParkImage::decode_header(b)?;
    let at = encoded_len_of(&digests, 0);
    Ok((digests, counts, &b[at..]))
}

/// The two per-session counts an image carries beside its bytes.
///
/// Both are facts about the SESSION and neither can be derived from the payload: `resident` is
/// what the engine's own token count must go back to, and `committed` is which slot of the new
/// run each GDN region belongs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageCounts {
    /// Tokens resident when the park was issued.
    pub resident: u32,
    /// Tokens the last step committed: the offset of the live recurrent snapshot in the run.
    pub committed: u32,
}

impl ParkImage {
    /// Bytes this image encodes to.
    #[must_use]
    pub fn encoded_len(&self) -> usize {
        encoded_len_of(&self.digests, self.bytes.len())
    }

    /// Serialise for [`saber_sched::SeqState`].
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        encode_into(
            &self.digests,
            self.resident,
            self.committed,
            &self.bytes,
            &mut out,
        );
        out
    }

    /// Parse one back, accounting for every byte.
    pub fn decode(b: &[u8]) -> Result<ParkImage, ImageError> {
        let (digests, counts) = ParkImage::decode_header(b)?;
        let at = encoded_len_of(&digests, 0);
        Ok(ParkImage {
            digests,
            resident: counts.resident,
            committed: counts.committed,
            bytes: b[at..].to_vec(),
        })
    }

    /// Everything but the payload, accounting for every byte of the whole blob.
    ///
    /// The checks are the same ones [`ParkImage::decode`] makes — including that the blob's length
    /// is exactly what its header accounts for — so a caller that publishes straight out of the
    /// blob is no less strict than one that copies it out first.
    pub fn decode_header(b: &[u8]) -> Result<(ImageDigests, ImageCounts), ImageError> {
        if b.len() < HEADER {
            return Err(ImageError::NotAnImage(format!(
                "{} bytes, the header is {HEADER}",
                b.len()
            )));
        }
        if b[..8] != MAGIC {
            return Err(ImageError::NotAnImage(
                "it does not start with the session-image magic".to_owned(),
            ));
        }
        let version = u32_at(b, 8);
        if version != VERSION {
            return Err(ImageError::NotAnImage(format!(
                "version {version}, this build writes {VERSION}"
            )));
        }
        let flags = u32_at(b, 12);
        if flags != 0 {
            return Err(ImageError::NotAnImage(format!("flags {flags:#x}")));
        }
        let n_regions = u32_at(b, 16) as usize;
        let n_chunks = u32_at(b, 20) as usize;
        let chunk_bytes = u32_at(b, 24);
        let resident = u32_at(b, 28);
        let committed = u32_at(b, 32);
        // Never 0: it is `naccept`, and `naccept - 1` indexes the run. A 0 here would place every
        // GDN region one slot BEFORE the session's base, which is the previous sequence's state.
        if committed == 0 {
            return Err(ImageError::Malformed(
                "the image says its session committed 0 token(s); `committed - 1` indexes the \
                 state-slot run, so 0 is the slot below its base"
                    .to_owned(),
            ));
        }
        let payload = u64_at(b, 36);
        let want = HEADER + REGION_BYTES * n_regions + 32 * n_chunks + 32 + payload as usize;
        if b.len() != want {
            return Err(ImageError::Malformed(format!(
                "{} bytes; its header says {n_regions} region(s), {n_chunks} digest(s) and \
                 {payload} payload bytes, which is {want}",
                b.len()
            )));
        }
        let mut off = HEADER;
        let mut layout = Vec::with_capacity(n_regions);
        let mut total = 0u64;
        for _ in 0..n_regions {
            let kind = kind_of(u32_at(b, off))?;
            let len = u64_at(b, off + 4);
            total += len;
            layout.push(RegionSpec { kind, len });
            off += REGION_BYTES;
        }
        if total != payload {
            return Err(ImageError::Malformed(format!(
                "the regions total {total} bytes and the payload is {payload}"
            )));
        }
        let mut chunks = Vec::with_capacity(n_chunks);
        for _ in 0..n_chunks {
            let mut d = [0u8; 32];
            d.copy_from_slice(&b[off..off + 32]);
            chunks.push(d);
            off += 32;
        }
        let mut sha256 = [0u8; 32];
        sha256.copy_from_slice(&b[off..off + 32]);
        Ok((
            ImageDigests {
                layout,
                bytes: payload,
                chunk_bytes,
                chunks,
                sha256,
            },
            ImageCounts {
                resident,
                committed,
            },
        ))
    }

    /// Regions of each kind, and the bytes in them — the line a report prints.
    ///
    /// Counted per kind rather than "the rest are GDN": with a third kind (`DraftKv`) that
    /// subtraction was already wrong, and a shape line that mis-attributes a region is the kind
    /// of thing a reader trusts.
    #[must_use]
    pub fn shape(&self) -> (usize, usize, u64) {
        let of = |k: RegionKind| self.digests.layout.iter().filter(|r| r.kind == k).count();
        (
            of(RegionKind::Kv),
            of(RegionKind::GdnState),
            self.digests.bytes,
        )
    }

    /// Regions of the DRAFTER's pages this image carries, and their bytes.
    ///
    /// Zero for a session with no resident drafter, which is what makes "did the park carry the
    /// drafter?" a question the image itself answers.
    #[must_use]
    pub fn draft_shape(&self) -> (usize, u64) {
        let rs = self
            .digests
            .layout
            .iter()
            .filter(|r| r.kind == RegionKind::DraftKv);
        (rs.clone().count(), rs.map(|r| r.len).sum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image() -> ParkImage {
        let bytes: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        ParkImage {
            digests: ImageDigests {
                layout: vec![
                    RegionSpec {
                        kind: RegionKind::Kv,
                        len: 3072,
                    },
                    RegionSpec {
                        kind: RegionKind::GdnState,
                        len: 1024,
                    },
                ],
                bytes: 4096,
                chunk_bytes: 4 << 20,
                chunks: vec![[7u8; 32]],
                sha256: [9u8; 32],
            },
            // Not round: a restore that ignored this field and left the engine's count at zero is
            // the bug the card gate caught, so the round trip has to carry it.
            resident: 1234,
            // Nor is this: a restore that ignored it would publish the recurrent state at the
            // run's BASE, which is right only for a session whose last step committed one token.
            committed: 6,
            bytes,
        }
    }

    #[test]
    fn an_image_round_trips_every_field() {
        let img = image();
        let blob = img.encode();
        assert_eq!(blob.len(), img.encoded_len());
        assert_eq!(ParkImage::decode(&blob).expect("decode"), img);
        assert_eq!(img.shape(), (1, 1, 4096));
        // ...and the borrowing split agrees with the owning decode, field for field.
        let (digests, counts, payload) = split_encoded(&blob).expect("split");
        assert_eq!(digests, img.digests);
        assert_eq!(counts.resident, img.resident);
        assert_eq!(counts.committed, img.committed);
        assert_eq!(payload, &img.bytes[..]);
    }

    /// The DRAFTER's pages are a third region kind and they survive the wire (ws-s5).
    ///
    /// A session's image used to be "KV plus GDN state", and `shape()` counted the GDN half by
    /// SUBTRACTION — so the moment a third kind existed that line mis-attributed it. Both halves
    /// are asserted here: the tag round trips, and each kind is counted by its own name.
    #[test]
    fn an_image_carries_the_drafters_pages_as_their_own_kind() {
        let mut img = image();
        img.digests.layout.push(RegionSpec {
            kind: RegionKind::DraftKv,
            len: 2048,
        });
        img.digests.bytes += 2048;
        img.bytes.extend((0..2048u32).map(|i| (i % 97) as u8));
        let blob = img.encode();
        let back = ParkImage::decode(&blob).expect("decode");
        assert_eq!(back, img);
        assert_eq!(
            back.digests.layout[2].kind,
            RegionKind::DraftKv,
            "the drafter's region came back as another kind"
        );
        // One KV region and one GDN region, counted by name and not by subtraction.
        assert_eq!(back.shape(), (1, 1, 4096 + 2048));
        assert_eq!(back.draft_shape(), (1, 2048));
        // The CONTROL: an image with no drafter half reports none, so "did the park carry the
        // drafter?" is a question the image answers rather than one the caller has to know.
        assert_eq!(image().draft_shape(), (0, 0));
    }

    #[test]
    fn a_damaged_blob_does_not_decode() {
        // `CLAUDE.md` rule 8: every truncation, a wrong magic and a wrong version are refused --
        // and the clean blob beside them is the control that must pass.
        let blob = image().encode();
        for cut in 0..blob.len() {
            assert!(
                ParkImage::decode(&blob[..cut]).is_err(),
                "a {cut}-byte prefix decoded"
            );
        }
        let mut bad = blob.clone();
        bad[0] ^= 0xFF;
        assert!(ParkImage::decode(&bad).is_err(), "wrong magic decoded");
        let mut bad = blob.clone();
        // Derived from VERSION rather than written out: the first cut hard-coded `2`, and when the
        // format went to v2 (the resident-token count) that line started asserting that the CURRENT
        // version is refused, which it is not.
        bad[8] = (VERSION + 1) as u8;
        assert!(ParkImage::decode(&bad).is_err(), "wrong version decoded");
        let mut bad = blob.clone();
        bad[8] = (VERSION - 1) as u8;
        assert!(
            ParkImage::decode(&bad).is_err(),
            "an OLDER version decoded; this format has no backward reader"
        );
        // A `committed` of 0 is a slot BELOW the session's base — the previous sequence's state —
        // so the header refuses it rather than placing a region there (ws-k14).
        let mut bad = blob.clone();
        bad[32] = 0;
        bad[33] = 0;
        bad[34] = 0;
        bad[35] = 0;
        assert!(
            ParkImage::decode(&bad).is_err(),
            "an image that committed 0 tokens decoded"
        );
        // A region table that does not add up to the payload.
        let mut bad = blob.clone();
        bad[HEADER + 4] = 0xFF;
        assert!(
            ParkImage::decode(&bad).is_err(),
            "bad region length decoded"
        );
        assert!(ParkImage::decode(&blob).is_ok());
    }
}
