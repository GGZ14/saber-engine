//! Session parking — evicting an idle session's KV **and** GDN state as one unit.
//!
//! `ARCHITECTURE.md` § Memory plane, *Session parking*: *"Evict an idle session's KV and its
//! 72 MiB/rank GDN state as one unit — they are useless apart. ~5 ms restore from RAM at PCIe
//! speed; ~12 ms from the CPU-attached Gen5 drive... Park to RAM first, spill to NVMe under
//! pressure. Parking is write traffic on NVMe: the byte ledger counts NVMe write bytes."*
//!
//! Three properties are structural here, not policy:
//!
//! 1. **One unit.** A [`SessionImage`] carries the KV blocks, the state slots and one opaque byte
//!    image. There is no API that parks half a session.
//! 2. **Billed.** A tier reserves against a [`Pool`] in the byte ledger before it accepts an image,
//!    and records the write traffic. A full tier refuses; it never over-commits.
//! 3. **Byte-identical resume.** The image that comes back out is compared against what went in —
//!    the token prefix in the scheduler, the bytes in the executor — because a park that silently
//!    perturbs state is the 09-17 lazy-snapshot failure in another costume (healthy 10/50 at
//!    temperature 0, with every single-shot gate reading null).
//!
//! [`HostTier`] is the RAM tier, and it is also the mock the tests run against: the same code path
//! serves both, so the tests exercise the interface `saber-mem`'s real tier will implement.

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use saber_mem::dio::{align_up, AlignedBuf, OpenMode};
use saber_mem::ledger::{Ledger, LedgerError, Pool};
use saber_mem::park::{
    NvmeTier, ParkError as MemParkError, ParkTicket as MemTicket, ParkTier, PinnedHostTier, Slab,
    Verification,
};
use saber_serve::engine::{RequestId, TokenId};

use crate::executor::{SeqId, SeqState};

/// A parked session, as the tier stores it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionImage {
    /// The request it belongs to.
    pub request_id: RequestId,
    /// The sequence id it had while running. A restored session gets a **new** id; this one is
    /// kept so a mismatch is visible.
    pub seq_id: SeqId,
    /// Prompt plus everything generated at park time — the logical content of the KV blocks.
    pub tokens: Vec<TokenId>,
    /// KV blocks the session held.
    pub kv_blocks: u32,
    /// GDN state slots it held.
    pub gdn_slots: u32,
    /// VRAM bytes the two allocations held on one rank. What the tier bills.
    pub resident_bytes: u64,
    /// The opaque KV + GDN image the executor handed over.
    pub state: SeqState,
}

/// A receipt for a parked session.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParkTicket {
    /// Tier-local id.
    pub id: u64,
    /// Which pool it is billed to.
    pub pool: Pool,
    /// Bytes billed.
    pub bytes: u64,
}

/// Why a park or restore failed.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ParkError {
    /// No parking tier was configured.
    #[error("no parking tier is configured")]
    NoTier,
    /// The request is not running (already finished, never admitted, or already parked).
    #[error("request {0} is not running")]
    NotRunning(RequestId),
    /// The request is not parked.
    #[error("request {0} is not parked")]
    NotParked(RequestId),
    /// The request's prompt is not all in yet. An image taken now carries what the ENGINE holds
    /// (nothing, or a partial prefill) while the group's bookkeeping says the prompt is resident,
    /// and a restore marks it prefilled — so its first decode step reads a context the engine
    /// never wrote (ws-k17, card: `the batch says 714 token(s) are resident and the engine has 0`).
    #[error("request {0} is still prefilling; a park waits until its prompt is in")]
    StillPrefilling(RequestId),
    /// A step is in flight, and parking needs the executor to copy the state image out. The
    /// engine refuses rather than waiting on the device: a caller blocked here would be a caller
    /// blocked for a whole step. Retry between steps.
    #[error("request {0} cannot be parked while a step is in flight; retry")]
    StepInFlight(RequestId),
    /// The executor already has a session transfer in flight and holds one staging buffer, so this
    /// one has to wait. Not a failure: the group is exactly as it was and the caller may ask again
    /// next step.
    #[error("request {0}: a session transfer is already in flight; retry")]
    TransferBusy(RequestId),
    /// The executor cannot hand over a state image, so there is nothing to bring back. Refused
    /// rather than evicted — the fail-loud default of [`crate::StepExecutor::park_state`].
    #[error("executor cannot park sequence {0}: no state image")]
    ExecutorCannotPark(SeqId),
    /// The executor refused to take a state image back.
    #[error("executor refused to restore sequence {0}")]
    ExecutorCannotRestore(SeqId),
    /// The tier is full.
    #[error("park tier {pool} is full: {want} bytes wanted, {free} free")]
    TierFull {
        /// The pool.
        pool: Pool,
        /// Bytes wanted.
        want: u64,
        /// Bytes free.
        free: u64,
    },
    /// The ticket names nothing the tier holds.
    #[error("park tier {pool} has no image {id}")]
    NoSuchImage {
        /// The pool.
        pool: Pool,
        /// The ticket id.
        id: u64,
    },
    /// What came back is not what went in.
    #[error("restored image for request {request_id} does not match the parked session: {detail}")]
    ImageMismatch {
        /// The request.
        request_id: RequestId,
        /// What differed.
        detail: String,
    },
    /// The ledger refused.
    #[error("park tier ledger error: {0}")]
    Ledger(String),
}

impl From<LedgerError> for ParkError {
    fn from(e: LedgerError) -> Self {
        match e {
            LedgerError::OverCommit {
                pool, want, free, ..
            } => ParkError::TierFull { pool, want, free },
            other => ParkError::Ledger(other.to_string()),
        }
    }
}

/// Which-path counters for a parking tier (`CLAUDE.md` rule 5).
///
/// Five numbers, because there are five paths an image can take and each one has a different cost:
/// a park that fitted in RAM (~5 ms to come back), one that had to spill (~15 ms), a spilled image
/// pulled back up, a restore, and a refusal because the bytes did not hash to what went in. A tier
/// that silently spilled everything to NVMe would look identical to one that never did, except in
/// the p99 — which is exactly the kind of thing that gets found six weeks later.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParkCounters {
    /// Images parked into the RAM tier.
    pub parked_to_ram: u64,
    /// Images that ended up on NVMe: spilled to make room in RAM, or written straight there
    /// because RAM had no slot to make room in.
    pub demoted_to_nvme: u64,
    /// Images pulled back from NVMe into RAM without being restored.
    pub promoted: u64,
    /// Images handed back to the scheduler.
    pub restored: u64,
    /// Restores refused because the stored image did not hash to what was parked. I3: never a
    /// silent restore of bytes that do not match.
    pub refused_corrupt: u64,
}

/// A tier a session can be parked to: host RAM today, NVMe under pressure.
///
/// The ledger is passed in rather than held, so that there is exactly one ledger in the engine and
/// the tier cannot bill a second one.
pub trait Parkable: fmt::Debug + Send + 'static {
    /// The byte pool this tier bills.
    fn pool(&self) -> Pool;

    /// Takes a session. The bytes are reserved before the image is stored, so a full tier refuses
    /// instead of over-committing.
    fn park(&mut self, image: SessionImage, ledger: &mut Ledger) -> Result<ParkTicket, ParkError>;

    /// Gives a session back and releases its reservation.
    fn restore(
        &mut self,
        ticket: &ParkTicket,
        ledger: &mut Ledger,
    ) -> Result<SessionImage, ParkError>;

    /// Drops a parked session (its request was aborted) and releases its reservation.
    fn discard(&mut self, ticket: &ParkTicket, ledger: &mut Ledger);

    /// Sessions currently held.
    fn len(&self) -> usize;

    /// Whether the tier holds nothing.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Which-path counters. Default-empty so a tier with one path need not invent five.
    fn counters(&self) -> ParkCounters {
        ParkCounters::default()
    }

    /// Housekeeping the tier may do when **nothing is running**.
    ///
    /// The scheduler calls this only on an idle step. A tiered store uses it to pull a spilled
    /// image back into RAM, which costs a drive read plus a copy — free while the engine has
    /// nothing to do, and a 15 ms hole in a step if it were done on the restore path instead.
    fn maintain(&mut self, _ledger: &mut Ledger) {}
}

/// Bytes a parking tier holds for one session image: its wire form's own length.
///
/// **One number, for both tiers** (ws-k15). [`TieredParkStore`] bills what it writes into RAM or
/// onto the drive; [`HostTier`] used to bill [`SessionImage::resident_bytes`], which is a VRAM
/// figure — the KV blocks and the whole `max_width`-slot GDN run the session held on one rank —
/// and it is several times the image at any speculative width. A ledger that answers "how much
/// host memory is parking using" two ways answers it wrongly once.
///
/// [`SessionImage::resident_bytes`] stays: it is a fact about the SESSION (what a restore frees on
/// the card) and it is what `choose_victim` weighs. It is simply not what a host tier holds.
#[must_use]
pub fn park_image_bytes(image: &SessionImage) -> usize {
    wire::encoded_len(image)
}

/// The RAM parking tier — and the mock the tests use, deliberately the same code.
#[derive(Debug)]
pub struct HostTier {
    pool: Pool,
    images: BTreeMap<u64, SessionImage>,
    next_id: u64,
    parks: u64,
    restores: u64,
    discards: u64,
}

impl HostTier {
    /// A tier billing `pool`. Capacity comes from the pool's declared capacity in the ledger, so
    /// there is one place a byte budget is stated.
    pub fn new(pool: Pool) -> Self {
        Self {
            pool,
            images: BTreeMap::new(),
            next_id: 1,
            parks: 0,
            restores: 0,
            discards: 0,
        }
    }

    /// Host RAM, the first parking tier.
    pub fn ram() -> Self {
        Self::new(Pool::HostPageable)
    }

    /// NVMe, the spill tier. Write traffic lands in the ledger's write counter, which is the
    /// number `/metrics` needs against a 2400 TBW-class drive.
    pub fn nvme() -> Self {
        Self::new(Pool::Nvme)
    }

    /// Sessions parked since construction.
    pub fn parks(&self) -> u64 {
        self.parks
    }

    /// Sessions restored since construction.
    pub fn restores(&self) -> u64 {
        self.restores
    }

    /// Sessions discarded (aborted while parked).
    pub fn discards(&self) -> u64 {
        self.discards
    }

    fn reservation_name(id: u64) -> String {
        format!("session:{id}")
    }
}

impl Parkable for HostTier {
    fn pool(&self) -> Pool {
        self.pool
    }

    fn park(&mut self, image: SessionImage, ledger: &mut Ledger) -> Result<ParkTicket, ParkError> {
        let id = self.next_id;
        // **THE BYTES THIS TIER HOLDS, which are the IMAGE's and not the session's** (ws-k15).
        // This billed `resident_bytes` — the VRAM the session's KV blocks and GDN slot SET held on
        // one rank — against a HOST pool, while [`TieredParkStore`] bills what it actually writes.
        // Two tiers billing the same park two ways is two answers to "how much host memory is
        // parking using", and the gap is large and in the wrong direction: at verify width 8 a
        // session's slot set is the whole `max_width`-slot run (1,227 MiB at TP=1) and the image
        // that leaves the card carries ONE slot per state (258 MiB, ws-k14). The number both tiers
        // now bill is the encoded image's own length.
        let bytes = wire::encoded_len(&image) as u64;
        ledger.reserve(self.pool, &Self::reservation_name(id), bytes)?;
        // Parking is write traffic. On NVMe this is the number the drive's endurance budget is
        // read from; on RAM it is still the honest count of bytes moved.
        ledger
            .record_written(self.pool, bytes)
            .map_err(|e| ParkError::Ledger(e.to_string()))?;
        self.next_id += 1;
        self.parks += 1;
        self.images.insert(id, image);
        Ok(ParkTicket {
            id,
            pool: self.pool,
            bytes,
        })
    }

    fn restore(
        &mut self,
        ticket: &ParkTicket,
        ledger: &mut Ledger,
    ) -> Result<SessionImage, ParkError> {
        let image = self
            .images
            .remove(&ticket.id)
            .ok_or(ParkError::NoSuchImage {
                pool: self.pool,
                id: ticket.id,
            })?;
        ledger
            .release(self.pool, &Self::reservation_name(ticket.id))
            .map_err(|e| ParkError::Ledger(e.to_string()))?;
        self.restores += 1;
        Ok(image)
    }

    fn discard(&mut self, ticket: &ParkTicket, ledger: &mut Ledger) {
        if self.images.remove(&ticket.id).is_some() {
            self.discards += 1;
            // A release that fails here means the ledger and the tier disagree; log it rather than
            // unwinding a cleanup path.
            if let Err(e) = ledger.release(self.pool, &Self::reservation_name(ticket.id)) {
                tracing::error!(pool = %self.pool, id = ticket.id, error = %e, "park tier release failed");
            }
        }
    }

    fn len(&self) -> usize {
        self.images.len()
    }

    /// The mock has one tier, so three of the five paths cannot happen and stay zero. Reporting
    /// them anyway keeps `/metrics` the same shape whichever tier is attached.
    fn counters(&self) -> ParkCounters {
        ParkCounters {
            parked_to_ram: if self.pool == Pool::Nvme {
                0
            } else {
                self.parks
            },
            demoted_to_nvme: if self.pool == Pool::Nvme {
                self.parks
            } else {
                0
            },
            promoted: 0,
            restored: self.restores,
            refused_corrupt: 0,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The real tiers, behind the trait the scheduler already speaks
// ---------------------------------------------------------------------------------------------

/// Serialised form of a [`SessionImage`]: a fixed header, then the tokens, then the state bytes.
///
/// The whole thing is what the tier hashes, so the token list is covered by the same sha256 as the
/// device state. That is the point: the scheduler's restore check compares tokens and the tier's
/// compares bytes, and if the two verified different things there would be a gap between them for
/// a corrupt image to fit through.
mod wire {
    use super::{RequestId, SeqId, SeqState, SessionImage, TokenId};

    /// `SABRPARK`, so a slot read at the wrong offset is a refusal rather than a decode.
    const MAGIC: [u8; 8] = *b"SABRPARK";
    const VERSION: u32 = 1;
    // Field offsets, named rather than counted: the first cut of this had `n_tokens` four bytes
    // out, every decode failed the length check, and the scheduler dutifully refused every
    // restore as a corrupt image. The tier was right and the reader was wrong.
    const OFF_VERSION: usize = 8;
    const OFF_REQUEST: usize = OFF_VERSION + 4;
    const OFF_SEQ: usize = OFF_REQUEST + 8;
    const OFF_KV_BLOCKS: usize = OFF_SEQ + 8;
    const OFF_GDN_SLOTS: usize = OFF_KV_BLOCKS + 4;
    const OFF_RESIDENT: usize = OFF_GDN_SLOTS + 4;
    const OFF_N_TOKENS: usize = OFF_RESIDENT + 8;
    const OFF_N_STATE: usize = OFF_N_TOKENS + 8;
    const HEADER: usize = OFF_N_STATE + 8;

    /// Bytes `image` encodes to.
    pub fn encoded_len(image: &SessionImage) -> usize {
        HEADER + image.tokens.len() * 4 + image.state.0.len()
    }

    /// Encode into `out`, which is cleared first.
    pub fn encode(image: &SessionImage, out: &mut Vec<u8>) {
        out.clear();
        out.reserve(encoded_len(image));
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&image.request_id.0.to_le_bytes());
        out.extend_from_slice(&image.seq_id.0.to_le_bytes());
        out.extend_from_slice(&image.kv_blocks.to_le_bytes());
        out.extend_from_slice(&image.gdn_slots.to_le_bytes());
        out.extend_from_slice(&image.resident_bytes.to_le_bytes());
        out.extend_from_slice(&(image.tokens.len() as u64).to_le_bytes());
        out.extend_from_slice(&(image.state.0.len() as u64).to_le_bytes());
        for t in &image.tokens {
            out.extend_from_slice(&t.to_le_bytes());
        }
        out.extend_from_slice(&image.state.0);
    }

    fn u32_at(b: &[u8], off: usize) -> u32 {
        u32::from_le_bytes(b[off..off + 4].try_into().expect("4 bytes"))
    }

    fn u64_at(b: &[u8], off: usize) -> u64 {
        u64::from_le_bytes(b[off..off + 8].try_into().expect("8 bytes"))
    }

    /// Decode. `Err` carries what was wrong, for the scheduler's `ImageMismatch` detail.
    pub fn decode(b: &[u8]) -> Result<SessionImage, String> {
        if b.len() < HEADER {
            return Err(format!(
                "image is {} bytes; the header is {HEADER}",
                b.len()
            ));
        }
        if b[..8] != MAGIC {
            return Err("image does not start with the parking magic".to_owned());
        }
        let version = u32_at(b, OFF_VERSION);
        if version != VERSION {
            return Err(format!(
                "image is version {version}, this build writes {VERSION}"
            ));
        }
        let n_tokens = u64_at(b, OFF_N_TOKENS) as usize;
        let n_state = u64_at(b, OFF_N_STATE) as usize;
        let want = HEADER + n_tokens * 4 + n_state;
        if b.len() != want {
            return Err(format!(
                "image is {} bytes; its header says {n_tokens} tokens and {n_state} state bytes, \
                 which is {want}",
                b.len()
            ));
        }
        let mut tokens: Vec<TokenId> = Vec::with_capacity(n_tokens);
        for i in 0..n_tokens {
            tokens.push(u32_at(b, HEADER + i * 4));
        }
        let state = SeqState(b[HEADER + n_tokens * 4..].to_vec());
        Ok(SessionImage {
            request_id: RequestId(u64_at(b, OFF_REQUEST)),
            seq_id: SeqId(u64_at(b, OFF_SEQ)),
            tokens,
            kv_blocks: u32_at(b, OFF_KV_BLOCKS),
            gdn_slots: u32_at(b, OFF_GDN_SLOTS),
            resident_bytes: u64_at(b, OFF_RESIDENT),
            state,
        })
    }
}

/// How a [`TieredParkStore`] is sized and what it is allowed to do.
///
/// Every number is stated once, here; nothing reads an environment variable (`CLAUDE.md` rule 4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TieredParkConfig {
    /// The largest session image the store accepts, before 4096-alignment. One slot holds one
    /// session — KV **and** GDN state, because they are useless apart.
    pub image_bytes: u64,
    /// Pinned (or, with no ROCm, page-aligned) RAM the store may hold. Rounded **down** to whole
    /// slots; zero means no RAM tier, which only makes sense with an NVMe tier.
    pub ram_bytes: u64,
    /// File bytes the NVMe tier may hold. Zero means no NVMe tier, and a full RAM tier then
    /// refuses instead of spilling.
    pub nvme_bytes: u64,
    /// Where the parking file lives. Required when `nvme_bytes > 0`.
    pub nvme_path: Option<std::path::PathBuf>,
    /// Threads the per-image digest runs on. 1 is ~2.6 GB/s and 8 is ~16-20 GB/s (09-18, K4): at
    /// 136 MiB an image that is the difference between a 53 ms park and a 7 ms one.
    pub hash_threads: usize,
    /// Pull a spilled image back into RAM on an **idle** step. Off by default: it is a drive read
    /// plus a copy, and the only moment it is genuinely free is a step with nothing running.
    pub promote_on_slack: bool,
    /// `O_DIRECT` for the parking file. `Buffered` is for filesystems that have none (tmpfs).
    pub nvme_mode: OpenMode,
}

impl Default for TieredParkConfig {
    fn default() -> Self {
        Self {
            image_bytes: 0,
            ram_bytes: 0,
            nvme_bytes: 0,
            nvme_path: None,
            hash_threads: 8,
            promote_on_slack: false,
            nvme_mode: OpenMode::Direct,
        }
    }
}

impl TieredParkConfig {
    /// A RAM-only store of `ram_bytes`, holding sessions of at most `image_bytes`.
    #[must_use]
    pub fn ram(image_bytes: u64, ram_bytes: u64) -> Self {
        Self {
            image_bytes,
            ram_bytes,
            ..Self::default()
        }
    }

    /// Add the NVMe spill tier.
    #[must_use]
    pub fn with_nvme(mut self, path: impl Into<std::path::PathBuf>, bytes: u64) -> Self {
        self.nvme_path = Some(path.into());
        self.nvme_bytes = bytes;
        self
    }

    /// Set the digest thread count.
    #[must_use]
    pub fn hashing_on(mut self, threads: usize) -> Self {
        self.hash_threads = threads.max(1);
        self
    }

    /// Turn on [`TieredParkConfig::promote_on_slack`].
    #[must_use]
    pub fn promoting_on_slack(mut self) -> Self {
        self.promote_on_slack = true;
        self
    }

    /// Slot size: one session image, rounded up to the `O_DIRECT` alignment.
    #[must_use]
    pub fn slot_bytes(&self) -> u64 {
        align_up(self.image_bytes.max(1))
    }

    /// RAM slots this configuration buys.
    #[must_use]
    pub fn ram_slots(&self) -> usize {
        (self.ram_bytes / self.slot_bytes()) as usize
    }

    /// NVMe slots this configuration buys.
    #[must_use]
    pub fn nvme_slots(&self) -> usize {
        (self.nvme_bytes / self.slot_bytes()) as usize
    }
}

/// Where one parked image currently lives.
#[derive(Clone, Debug)]
struct Placed {
    ticket: MemTicket,
    /// Park order, for choosing which RAM image spills when a new one needs the slot.
    seq: u64,
}

/// The production parking store: RAM first, an `O_DIRECT` file under pressure.
///
/// This is the adapter that meets the scheduler's [`Parkable`] contract with `saber_mem::park`'s
/// real tiers; [`HostTier`] stays as the in-memory mock. What it adds over the mock is every
/// property those tiers were built for — the image is hashed on the way in and verified on the way
/// out (I3, so a corrupt image is refused rather than restored), the slot pools are allocated once
/// and never move (I1), every byte is billed to the one [`Ledger`], and a full RAM tier spills its
/// oldest image to the drive instead of refusing.
///
/// Measured on this box (09-18, `saber park-bench`, 136 MiB images, GPU idle): 5.03 ms to restore
/// from RAM and 15.1 ms from NVMe, against the architecture's ~5 ms / ~12-20 ms estimates.
///
/// One thing it deliberately does **not** do is split the image: `SeqState` is opaque, the
/// executor packed it, and the store's job is to give back exactly those bytes. The region-wise
/// D2H/H2D path (`saber_mem::park::ParkTier::park`) is what a device-capable executor will use
/// once it hands out device addresses; the byte path is today's `StepExecutor::park_state`
/// contract and is the only one that works with an executor that has no device at all.
#[derive(Debug)]
pub struct TieredParkStore {
    ram: Option<PinnedHostTier>,
    nvme: Option<NvmeTier>,
    placed: BTreeMap<u64, Placed>,
    next_id: u64,
    park_seq: u64,
    counters: ParkCounters,
    promote_on_slack: bool,
    scratch: Vec<u8>,
    slot_bytes: u64,
}

impl TieredParkStore {
    /// Build a store over slabs the caller supplies.
    ///
    /// `ram_slab` is where the RAM tier's slots come from: [`saber_mem::park::PinnedSlab`] in the
    /// engine (page-locked, so a park is a DMA) or an [`AlignedBuf`] anywhere there is no ROCm.
    /// `nvme` is the parking file's path and its staging slab, which needs one whole slot.
    pub fn new(
        cfg: &TieredParkConfig,
        ram_slab: Option<Box<dyn Slab>>,
        nvme: Option<(&Path, Box<dyn Slab>)>,
    ) -> Result<Self, ParkError> {
        let slot_bytes = cfg.slot_bytes();
        let verify = Verification::on(cfg.hash_threads.max(1));
        if cfg.ram_bytes > 0 && cfg.ram_slots() == 0 {
            return Err(ParkError::Ledger(format!(
                "park RAM budget is {} bytes, which is less than one {slot_bytes}-byte session slot",
                cfg.ram_bytes
            )));
        }
        if cfg.nvme_bytes > 0 && cfg.nvme_slots() == 0 {
            return Err(ParkError::Ledger(format!(
                "park NVMe budget is {} bytes, which is less than one {slot_bytes}-byte session \
                 slot",
                cfg.nvme_bytes
            )));
        }
        let ram = match ram_slab {
            Some(slab) if cfg.ram_slots() > 0 => Some(
                PinnedHostTier::new(slab, cfg.ram_slots(), slot_bytes)
                    .map_err(|e| mem_err(e, slot_bytes))?
                    .verifying(verify),
            ),
            _ => None,
        };
        let nvme = match nvme {
            Some((path, staging)) if cfg.nvme_slots() > 0 => Some(
                NvmeTier::create(path, cfg.nvme_slots(), slot_bytes, staging, cfg.nvme_mode)
                    .map_err(|e| mem_err(e, slot_bytes))?
                    .verifying(verify),
            ),
            _ => None,
        };
        if ram.is_none() && nvme.is_none() {
            return Err(ParkError::Ledger(
                "a parking store needs at least one tier with at least one slot".to_owned(),
            ));
        }
        Ok(Self {
            ram,
            nvme,
            placed: BTreeMap::new(),
            next_id: 1,
            park_seq: 0,
            counters: ParkCounters::default(),
            promote_on_slack: cfg.promote_on_slack,
            scratch: Vec::new(),
            slot_bytes,
        })
    }

    /// A store with no ROCm: both slabs are ordinary 4096-aligned heap memory.
    ///
    /// This is what `saber serve --mock` and this crate's tests run on. It exercises the same
    /// tiers, the same hashing, the same spill rule and the same ledger accounting as the engine;
    /// the only thing it does not exercise is the DMA, which has its own gate (`saber park-bench`).
    pub fn host(cfg: &TieredParkConfig) -> Result<Self, ParkError> {
        let slot_bytes = cfg.slot_bytes();
        let ram: Option<Box<dyn Slab>> = if cfg.ram_slots() > 0 {
            Some(Box::new(AlignedBuf::zeroed(
                cfg.ram_slots() * slot_bytes as usize,
            )))
        } else {
            None
        };
        let staging: Option<Box<dyn Slab>> = if cfg.nvme_slots() > 0 {
            Some(Box::new(AlignedBuf::zeroed(slot_bytes as usize)))
        } else {
            None
        };
        let nvme = match (&cfg.nvme_path, staging) {
            (Some(path), Some(staging)) => Some((path.as_path(), staging)),
            _ => None,
        };
        Self::new(cfg, ram, nvme)
    }

    /// Delete the parking file when the store is dropped. For a benchmark or a test, never a
    /// service: a service wants the preallocated file to survive a restart.
    #[must_use]
    pub fn deleting_nvme_on_drop(mut self) -> Self {
        self.nvme = self.nvme.take().map(NvmeTier::deleting_on_drop);
        self
    }

    /// Slot size in bytes — one whole session.
    #[must_use]
    pub fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }

    /// Slots each tier has: `(ram, nvme)`.
    #[must_use]
    pub fn slots(&self) -> (usize, usize) {
        (
            self.ram.as_ref().map_or(0, ParkTier::slots),
            self.nvme.as_ref().map_or(0, ParkTier::slots),
        )
    }

    /// Images resident in RAM.
    #[must_use]
    pub fn ram_len(&self) -> usize {
        self.ram.as_ref().map_or(0, ParkTier::len)
    }

    /// Images spilled to NVMe.
    #[must_use]
    pub fn nvme_len(&self) -> usize {
        self.nvme.as_ref().map_or(0, ParkTier::len)
    }

    /// Flip one bit of a parked image, to prove the restore refuses it.
    ///
    /// `CLAUDE.md` rule 8's negative control for the whole parking path. Only the RAM tier: the
    /// NVMe side's control writes to the file itself and lives in `saber park-bench`, where it
    /// also proves nothing reached VRAM.
    #[doc(hidden)]
    pub fn corrupt_for_test(&mut self, ticket: &ParkTicket, offset: usize) -> bool {
        let Some(placed) = self.placed.get(&ticket.id) else {
            return false;
        };
        if placed.ticket.pool == Pool::Nvme {
            return false;
        }
        let mem = placed.ticket.clone();
        self.ram
            .as_mut()
            .is_some_and(|t| t.corrupt_for_test(&mem, offset))
    }

    /// Where a ticket's bytes are **now**.
    ///
    /// [`ParkTicket::pool`] records where they were placed at park time, which a later spill can
    /// make stale — the scheduler never reads it, and this is the live answer for anything that
    /// does.
    #[must_use]
    pub fn residency(&self, ticket: &ParkTicket) -> Option<Pool> {
        self.placed.get(&ticket.id).map(|p| p.ticket.pool)
    }

    /// Move the oldest RAM image to NVMe, freeing one RAM slot. Returns whether one moved.
    fn spill_one(&mut self, ledger: &mut Ledger) -> Result<bool, ParkError> {
        let slot_bytes = self.slot_bytes;
        let Some(victim) = self
            .placed
            .iter()
            .filter(|(_, p)| p.ticket.pool == Pool::HostPinned)
            .min_by_key(|(_, p)| p.seq)
            .map(|(id, _)| *id)
        else {
            return Ok(false);
        };
        let old = self.placed[&victim].ticket.clone();
        let (Some(ram), Some(nvme)) = (self.ram.as_mut(), self.nvme.as_mut()) else {
            return Ok(false);
        };
        let moved =
            saber_mem::park::demote(ram, nvme, &old, ledger).map_err(|e| mem_err(e, slot_bytes))?;
        self.placed.get_mut(&victim).expect("victim").ticket = moved;
        self.counters.demoted_to_nvme += 1;
        Ok(true)
    }

    fn park_encoded(&mut self, session: u64, ledger: &mut Ledger) -> Result<MemTicket, ParkError> {
        // The scratch buffer is moved out so the tiers can be borrowed mutably, and moved back on
        // every path. One buffer for the life of the store; nothing is allocated per park.
        let bytes = std::mem::take(&mut self.scratch);
        let result = self.park_encoded_inner(session, &bytes, ledger);
        self.scratch = bytes;
        result
    }

    fn park_encoded_inner(
        &mut self,
        session: u64,
        bytes: &[u8],
        ledger: &mut Ledger,
    ) -> Result<MemTicket, ParkError> {
        let slot_bytes = self.slot_bytes;
        // RAM first. A full tier spills its oldest image and tries once more; only if there is
        // nothing to spill into does the new image go straight to the drive.
        for attempt in 0..2 {
            let Some(ram) = self.ram.as_mut() else { break };
            match ram.park_bytes(session, bytes, ledger) {
                Ok(t) => {
                    self.counters.parked_to_ram += 1;
                    return Ok(t);
                }
                // Both shapes of "RAM is full": no free slot, or the ledger's HostPinned budget is
                // spent. A spill releases both, so the retry is worth taking.
                Err(MemParkError::TierFull { .. } | MemParkError::Ledger(_)) if attempt == 0 => {
                    if !self.spill_one(ledger)? {
                        break;
                    }
                }
                Err(e) => return Err(mem_err(e, slot_bytes)),
            }
        }
        let Some(nvme) = self.nvme.as_mut() else {
            return Err(ParkError::TierFull {
                pool: Pool::HostPinned,
                want: bytes.len() as u64,
                free: ledger.free(Pool::HostPinned).unwrap_or(0),
            });
        };
        let t = nvme
            .park_bytes(session, bytes, ledger)
            .map_err(|e| mem_err(e, slot_bytes))?;
        self.counters.demoted_to_nvme += 1;
        Ok(t)
    }
}

/// `saber_mem`'s park errors, in the scheduler's vocabulary.
///
/// The one that matters is [`MemParkError::ImageCorrupt`]: it becomes `ImageMismatch`, which is
/// the error the scheduler already treats as *the session stays parked and nothing is restored*.
/// `TierFull` keeps its pool and reports one slot's worth of bytes, because whole slots — not
/// loose bytes — are what these tiers run out of.
fn mem_err(e: MemParkError, slot_bytes: u64) -> ParkError {
    match e {
        MemParkError::TierFull { pool, .. } => ParkError::TierFull {
            pool,
            want: slot_bytes,
            free: 0,
        },
        MemParkError::NoSuchImage { pool, id } => ParkError::NoSuchImage { pool, id },
        MemParkError::ImageCorrupt {
            session,
            expected,
            found,
            ..
        } => ParkError::ImageMismatch {
            request_id: RequestId(session),
            detail: format!("stored image hashes to {found}, not {expected}"),
        },
        other => ParkError::Ledger(other.to_string()),
    }
}

impl Parkable for TieredParkStore {
    fn pool(&self) -> Pool {
        if self.ram.is_some() {
            Pool::HostPinned
        } else {
            Pool::Nvme
        }
    }

    fn park(&mut self, image: SessionImage, ledger: &mut Ledger) -> Result<ParkTicket, ParkError> {
        let id = self.next_id;
        let encoded = wire::encoded_len(&image) as u64;
        if encoded > self.slot_bytes {
            // Fail loud and early: a session bigger than a slot is a sizing error, and half of it
            // on the drive would be worse than refusing.
            return Err(ParkError::TierFull {
                pool: self.pool(),
                want: encoded,
                free: self.slot_bytes,
            });
        }
        wire::encode(&image, &mut self.scratch);
        let mem = self.park_encoded(id, ledger)?;
        self.next_id += 1;
        self.park_seq += 1;
        let ticket = ParkTicket {
            id,
            pool: mem.pool,
            bytes: mem.bytes,
        };
        self.placed.insert(
            id,
            Placed {
                ticket: mem,
                seq: self.park_seq,
            },
        );
        Ok(ticket)
    }

    fn restore(
        &mut self,
        ticket: &ParkTicket,
        ledger: &mut Ledger,
    ) -> Result<SessionImage, ParkError> {
        let slot_bytes = self.slot_bytes;
        let placed = self
            .placed
            .get(&ticket.id)
            .ok_or(ParkError::NoSuchImage {
                pool: self.pool(),
                id: ticket.id,
            })?
            .clone();
        let mut buf = vec![0u8; placed.ticket.bytes as usize];
        let res = if placed.ticket.pool == Pool::Nvme {
            self.nvme
                .as_mut()
                .ok_or(ParkError::NoSuchImage {
                    pool: Pool::Nvme,
                    id: ticket.id,
                })?
                .restore_bytes(&placed.ticket, &mut buf, ledger)
        } else {
            self.ram
                .as_mut()
                .ok_or(ParkError::NoSuchImage {
                    pool: Pool::HostPinned,
                    id: ticket.id,
                })?
                .restore_bytes(&placed.ticket, &mut buf, ledger)
        };
        if let Err(e) = res {
            if matches!(e, MemParkError::ImageCorrupt { .. }) {
                self.counters.refused_corrupt += 1;
            }
            // The image is still in the tier and still billed: the session stays parked, which is
            // exactly what the scheduler's restore-failure path expects.
            return Err(mem_err(e, slot_bytes));
        }
        self.placed.remove(&ticket.id);
        match wire::decode(&buf) {
            Ok(image) => {
                self.counters.restored += 1;
                Ok(image)
            }
            Err(detail) => {
                // Bytes that hash correctly but do not decode mean the header and the digest
                // disagree, which is a bug in this file rather than a bit flip. Counted the same
                // way and just as loud.
                self.counters.refused_corrupt += 1;
                Err(ParkError::ImageMismatch {
                    request_id: RequestId(ticket.id),
                    detail,
                })
            }
        }
    }

    fn discard(&mut self, ticket: &ParkTicket, ledger: &mut Ledger) {
        let Some(placed) = self.placed.remove(&ticket.id) else {
            return;
        };
        if placed.ticket.pool == Pool::Nvme {
            if let Some(t) = self.nvme.as_mut() {
                t.discard(&placed.ticket, ledger);
            }
        } else if let Some(t) = self.ram.as_mut() {
            t.discard(&placed.ticket, ledger);
        }
    }

    fn len(&self) -> usize {
        self.placed.len()
    }

    fn counters(&self) -> ParkCounters {
        self.counters
    }

    fn maintain(&mut self, ledger: &mut Ledger) {
        if !self.promote_on_slack {
            return;
        }
        let slot_bytes = self.slot_bytes;
        // Newest first: the most recently spilled image is the one most likely to be asked for.
        let Some(id) = self
            .placed
            .iter()
            .filter(|(_, p)| p.ticket.pool == Pool::Nvme)
            .max_by_key(|(_, p)| p.seq)
            .map(|(id, _)| *id)
        else {
            return;
        };
        let old = self.placed[&id].ticket.clone();
        let (Some(ram), Some(nvme)) = (self.ram.as_mut(), self.nvme.as_mut()) else {
            return;
        };
        if ram.len() >= ram.slots() {
            return;
        }
        match saber_mem::park::promote(nvme, ram, &old, ledger) {
            Ok(moved) => {
                self.placed.get_mut(&id).expect("id").ticket = moved;
                self.counters.promoted += 1;
            }
            Err(e) => {
                let _ = slot_bytes;
                tracing::error!(id, error = %e, "promotion from NVMe failed; the image stays spilled");
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// When to park: the eviction policy, as data
// ---------------------------------------------------------------------------------------------

/// When the scheduler may park a running session on its own.
///
/// A policy, expressed as numbers rather than as a ladder of `if`s (`CLAUDE.md` rule 6). All of it
/// is off by default: attaching a tier gives the operator [`crate::Scheduler::park`], and nothing
/// evicts anything until this says it may.
///
/// The trigger is **ledger pressure**, not a timer: an admission that was refused for KV blocks or
/// for a GDN state slot set is the moment when a parked session is worth more than a resident one.
/// A `BatchFull` refusal is deliberately not a trigger — the running set is at the widest captured
/// bucket, and swapping one sequence for another there buys nothing but two transfers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParkPolicy {
    /// Park a victim when an admission is refused for KV or GDN state. Off by default.
    pub on_pressure: bool,
    /// Steps a sequence must have gone without committing a token before it may be evicted.
    ///
    /// **This cannot fire today and is not pretending to.** Every running sequence commits at
    /// least one token every step, so idleness is 0 or 1 and any value above 1 disables eviction
    /// outright. It is the hook for the case the architecture's parking section is actually about
    /// — a chat session that stays resident between turns — which needs sessions that outlive a
    /// request, and this engine has none (prefix caching is off, `tests/multi_turn.rs`). The knob
    /// that binds today is [`ParkPolicy::min_resident_steps`].
    pub idle_steps: u32,
    /// Steps a sequence must have been resident before it may be evicted.
    ///
    /// The guard that matters: a sequence admitted one step ago has just paid for its whole
    /// prefill, and parking it throws that away and then pays a restore to get back where it was.
    pub min_resident_steps: u32,
    /// Never park below this many running sequences. One is the floor that matters: parking the
    /// only running sequence to admit the next one is a swap loop, not progress.
    pub min_running: u32,
    /// Parks one step may perform. Bounded because a park is a synchronous transfer on the step
    /// thread — 5 ms from RAM, 15 ms to the drive.
    pub max_parks_per_step: u32,
    /// Times one session may be evicted over its life. **1 by default** — a MEASURED deviation
    /// from prod, which preempts a request as often as pressure demands (ws-k18).
    ///
    /// ws-k17 made it unbounded, reasoning that the residency guard (one step after a restore) and
    /// the restored group's place at the FRONT of the queue buy every round trip forward progress.
    /// One step is not enough: on the card (`multiturn.sh --tp 1 --pressure 3`, pool 150 blocks)
    /// the unbounded policy parked 23 times and 22 of them were RE-parks of one session — the one
    /// just restored is again the back of the running queue (prod's order) AND again the one with
    /// the most work left (K5's), so under BOTH orders it was evicted at the next arrival, and one
    /// pressure request took **163-167 s** against a p50 of 3 s. The default (prod's order, cap
    /// one) on the same command: 29 parks, 0 re-parks, the slowest request of either kind
    /// **12.4 s**, 60/60 and 15/15 — paid for by the conversation, whose turns were no longer
    /// sharing the card with a starved neighbour: turn p90 5.71 s against 4.47-4.93, decode p50
    /// 29.7 tok/s against 37.0-40.0, TTFT p50 1.45 s against 1.40-1.44 (`docs/LEDGER.md`, K18).
    /// A cap of one under K5's order moved more of the bill onto the conversation (turn p90
    /// 10.36 s). What the cap trades away — a session that would need a second eviction waits for
    /// its admission like any request — is what prod does to EVERY waiting request.
    /// `u32::MAX` is unbounded (`--park-max-per-session 0`).
    pub max_parks_per_session: u32,
    /// Bring a policy-parked session back on its own once a sequence finishes and frees the space
    /// it was evicted for. Without this, an evicted session waits for an explicit `unpark` that
    /// nothing is going to send.
    pub auto_restore: bool,
    /// WHICH running group is the victim, as data (`CLAUDE.md` rule 6). See [`VictimOrder`].
    pub victim: VictimOrder,
    /// Restore attempts one parked session may FAIL before the scheduler gives up on it and ends
    /// the request with a named error the client can read (ws-k18).
    ///
    /// Before this a restore that failed every time was retried on every freed slot for ever —
    /// `restore_failures` counting up in the census while the client waited out its own timeout,
    /// and a server that could not stop because a request never finished (the ws-k17 TP=2 run).
    /// A `TransferBusy` answer is not a failure and never counts. `0` behaves as `1`: a session is
    /// always given its first attempt.
    pub max_restore_attempts: u32,
}

/// Which running group [`choose_victim`] evicts, once the guards have filtered the candidates.
///
/// **prod's rule is [`VictimOrder::Newest`]**, read from the prod image's own scheduler
/// (`upstream/v1/core/sched/scheduler.py`, `Scheduler.schedule`, FCFS — `serve-mxfp4.sh` sets no
/// policy): when a RUNNING request cannot get the KV blocks for its next step, prod preempts
/// `self.running.pop()` — the BACK of the running queue, the request admitted (or re-admitted)
/// most recently — and repeats until the allocation fits or it has popped the requester itself.
/// A preempted request goes to the FRONT of the waiting queue (`prepend_request`), and a resumed
/// one is `self.running.append`-ed like any admission, so it is at the back again: **prod has no
/// guard that spares a request it has just brought back** — the one it resumed last is its first
/// victim next time. Two further facts, and Saber deviates from both by construction:
///
/// * prod never preempts to ADMIT a waiting request — a waiting request whose allocation fails
///   simply waits (`if new_blocks is None: break`), and nothing waiting is admitted in a step that
///   preempted. Saber reserves a sequence's whole `prompt + max_tokens + width + 1` at admission,
///   so a running sequence never needs to grow and admission is the only place pressure exists;
///   evicting FOR an admission is the deviation that makes parking do anything at all.
/// * prod RECOMPUTES a preempted request (`num_computed_tokens = 0`); Saber parks its state and
///   restores it, which is what the image law and the park gates are about.
///
/// And one guard Saber keeps that prod does not have: [`ParkPolicy::min_resident_steps`] spares a
/// group for its first step after an admission or a restore.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VictimOrder {
    /// prod's: the group furthest back in the running queue — admitted or restored most recently.
    Newest,
    /// K5..K17's: the group with the MOST work left (`max_tokens - generated`), because it would
    /// hold its KV blocks and its state slot set longest. Beside short streams it picks the
    /// long conversation at every arrival: ws-k17's card run parked 24 times and 23 of those were
    /// RE-parks of a session already parked once.
    MostWorkLeft,
}

impl VictimOrder {
    /// The flag spelling.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            VictimOrder::Newest => "newest",
            VictimOrder::MostWorkLeft => "most-work",
        }
    }

    /// Parse the flag spelling.
    ///
    /// # Errors
    /// Anything that is not one of the two labels.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "newest" => Ok(VictimOrder::Newest),
            "most-work" => Ok(VictimOrder::MostWorkLeft),
            other => Err(format!(
                "park victim order `{other}`: the orders are `newest` (prod's) and `most-work`"
            )),
        }
    }
}

impl Default for ParkPolicy {
    fn default() -> Self {
        Self {
            on_pressure: false,
            idle_steps: 0,
            min_resident_steps: 1,
            min_running: 1,
            max_parks_per_step: 1,
            max_parks_per_session: 1,
            auto_restore: true,
            victim: VictimOrder::Newest,
            max_restore_attempts: 3,
        }
    }
}

impl ParkPolicy {
    /// Park under KV/GDN pressure, with the defaults for everything else.
    #[must_use]
    pub fn on_pressure() -> Self {
        Self {
            on_pressure: true,
            ..Self::default()
        }
    }

    /// Set [`ParkPolicy::min_resident_steps`].
    #[must_use]
    pub fn min_resident_steps(mut self, steps: u32) -> Self {
        self.min_resident_steps = steps;
        self
    }

    /// Set [`ParkPolicy::idle_steps`].
    #[must_use]
    pub fn idle_steps(mut self, steps: u32) -> Self {
        self.idle_steps = steps;
        self
    }

    /// Set [`ParkPolicy::min_running`].
    #[must_use]
    pub fn min_running(mut self, running: u32) -> Self {
        self.min_running = running;
        self
    }

    /// Set [`ParkPolicy::max_parks_per_session`].
    #[must_use]
    pub fn max_parks_per_session(mut self, parks: u32) -> Self {
        self.max_parks_per_session = parks;
        self
    }

    /// Set [`ParkPolicy::victim`].
    #[must_use]
    pub fn victim(mut self, order: VictimOrder) -> Self {
        self.victim = order;
        self
    }

    /// Set [`ParkPolicy::max_restore_attempts`].
    #[must_use]
    pub fn max_restore_attempts(mut self, attempts: u32) -> Self {
        self.max_restore_attempts = attempts;
        self
    }
}

/// One running sequence, as the eviction policy sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParkCandidate {
    /// The request.
    pub request_id: RequestId,
    /// Engine step at which it was admitted.
    pub admitted_at_step: u64,
    /// Engine step at which it last committed a token.
    pub last_active_step: u64,
    /// Tokens it still has to generate before `max_tokens`.
    pub remaining_tokens: u64,
    /// Times this session has already been parked.
    pub parks_so_far: u32,
    /// Where the group stands in the RUNNING QUEUE: larger is further back, i.e. admitted or
    /// restored more recently. prod's victim is the largest ([`VictimOrder::Newest`]).
    pub queue_pos: u32,
}

/// Which running sequence to park, or `None` if none is eligible.
///
/// Pure, so the rule can be tested without a scheduler around it. Longest idle first; among
/// equals — which, today, is all of them — [`ParkPolicy::victim`] decides: prod's
/// [`VictimOrder::Newest`] (the back of the running queue) or K5's [`VictimOrder::MostWorkLeft`].
/// Ties break on the request id so the choice is deterministic and a test can assert it.
///
/// A restored session is spared for [`ParkPolicy::min_resident_steps`] after its restore and is
/// requeued ahead of new arrivals, so each eviction still leaves it further on;
/// [`ParkPolicy::max_parks_per_session`] (1 by default since ws-k18; prod's preemption is unbounded) is the
/// operator's hard cap on how many round trips one session may pay for.
#[must_use]
pub fn choose_victim(
    candidates: &[ParkCandidate],
    policy: &ParkPolicy,
    step: u64,
) -> Option<RequestId> {
    if candidates.len() as u32 <= policy.min_running {
        return None;
    }
    candidates
        .iter()
        .filter(|c| step.saturating_sub(c.last_active_step) >= u64::from(policy.idle_steps))
        .filter(|c| step.saturating_sub(c.admitted_at_step) >= u64::from(policy.min_resident_steps))
        .filter(|c| c.parks_so_far < policy.max_parks_per_session)
        .min_by_key(|c| {
            // Idleness first under either order (it cannot bind today: see `idle_steps`), then the
            // order's own key, then the request id so the choice is deterministic.
            let order = match policy.victim {
                VictimOrder::Newest => std::cmp::Reverse(u64::from(c.queue_pos)),
                VictimOrder::MostWorkLeft => std::cmp::Reverse(c.remaining_tokens),
            };
            (c.last_active_step, order, c.request_id.0)
        })
        .map(|c| c.request_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::SeqState;

    fn image() -> SessionImage {
        SessionImage {
            request_id: RequestId(0x0102_0304_0506_0708),
            seq_id: SeqId(9),
            tokens: vec![11, 22, 33, 151_645],
            kv_blocks: 7,
            gdn_slots: 6,
            resident_bytes: 136 * 1024 * 1024,
            state: SeqState(vec![0xAA; 37]),
        }
    }

    /// Every field, both ways. The first cut of `decode` read two lengths from the wrong offset,
    /// which the tiers reported as a corrupt image — correctly, and unhelpfully.
    #[test]
    fn the_image_wire_format_round_trips_every_field() {
        let img = image();
        let mut buf = Vec::new();
        wire::encode(&img, &mut buf);
        assert_eq!(buf.len(), wire::encoded_len(&img));
        assert_eq!(wire::decode(&buf).expect("decode"), img);
    }

    /// The negative control: every single-byte truncation, a wrong magic and a wrong version are
    /// all refused rather than decoded into something plausible.
    #[test]
    fn a_damaged_image_does_not_decode() {
        let img = image();
        let mut buf = Vec::new();
        wire::encode(&img, &mut buf);

        for cut in 0..buf.len() {
            assert!(
                wire::decode(&buf[..cut]).is_err(),
                "a {cut}-byte prefix decoded"
            );
        }
        let mut bad = buf.clone();
        bad[0] ^= 0xFF;
        assert!(wire::decode(&bad).is_err(), "wrong magic decoded");
        let mut bad = buf.clone();
        bad[8] = 2;
        assert!(wire::decode(&bad).is_err(), "wrong version decoded");

        // ...and the control that must NOT fail, or the three above prove nothing.
        assert!(wire::decode(&buf).is_ok());
    }
}
