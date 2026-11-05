//! saber-abi — the generated Rust view of `include/saber_abi.h`, plus the small amount of safe
//! typing that keeps every consumer from restating the ABI's semantics in its own words.
//!
//! **This crate is generated.** [`bindings`] is `bindgen` output, committed, and CI regenerates it
//! and requires an empty diff. Never hand-edit it; edit the header, bump
//! `SABER_ABI_VERSION_MINOR`, run [`regen.sh`](../../../crates/saber-abi/regen.sh), and add a
//! `docs/abi/CHANGELOG.md` entry in the same commit.
//!
//! What is hand-written lives here and is deliberately thin:
//!
//! * newtypes for the enumerated `u32` fields, each with a **fallible** conversion — an unknown
//!   value from a manifest is an error, never a silent default;
//! * [`ConstraintOp::holds`], the one implementation of the constraint predicates. The reference's header
//!   states the reason the registry exists at all: *"without it every integration ends up
//!   restating the constraints in its own gate, and the copies drift"*. Two copies of `DIV` is
//!   exactly that drift, so there is one;
//! * [`WsSize::resolve`], the one implementation of the workspace size law, with checked
//!   arithmetic;
//! * [`ImplFlags`] and friends, including the malformed-manifest checks the loader owes.
//!
//! There is no `unsafe` in this crate and no pointer dereferencing: the raw structs carry
//! `*const c_char` and `*const T` because that is the C ABI, but turning those into Rust values is
//! the manifest *loader's* job (saber-plan), not the schema's.

#![forbid(unsafe_code)]

mod bindings;

pub use bindings::*;

use core::fmt;

// ------------------------------------------------------------------------------------------- //
// Errors                                                                                        //
// ------------------------------------------------------------------------------------------- //

/// Something in a manifest is not representable in this ABI version.
///
/// Every variant is a refusal, not a warning: the design rule is that a manifest the host does not
/// fully understand is rejected rather than partially honoured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbiError {
    /// An enumerated field held a value this ABI version does not define.
    UnknownValue {
        /// Which enumeration, e.g. `"saber_constraint_op"`.
        what: &'static str,
        /// The value that was read.
        value: u32,
    },
    /// The manifest's `abi_version` is incompatible with this build's.
    Version {
        /// The version the manifest declares.
        found: AbiVersion,
        /// The version this build understands.
        expected: AbiVersion,
    },
    /// A field combination this ABI declares malformed (documented per check).
    Malformed(&'static str),
    /// The workspace size law overflowed `u64` for the given shape.
    SizeOverflow,
}

impl fmt::Display for AbiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AbiError::UnknownValue { what, value } => {
                write!(f, "unknown {what} value {value}")
            }
            AbiError::Version { found, expected } => write!(
                f,
                "manifest ABI version {found} is not accepted by a host built for {expected}"
            ),
            AbiError::Malformed(why) => write!(f, "malformed manifest: {why}"),
            AbiError::SizeOverflow => write!(f, "workspace size law overflowed u64"),
        }
    }
}

impl std::error::Error for AbiError {}

// ------------------------------------------------------------------------------------------- //
// Version                                                                                       //
// ------------------------------------------------------------------------------------------- //

/// A `(major, minor)` ABI version, unpacked from the `u32` a manifest carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AbiVersion {
    /// Incompatible revision: a different major is refused outright.
    pub major: u32,
    /// Additive revision: fields and enumerators appended.
    pub minor: u32,
}

impl AbiVersion {
    /// The version this build of `saber-abi` was generated from.
    pub const CURRENT: AbiVersion = AbiVersion {
        major: SABER_ABI_VERSION_MAJOR,
        minor: SABER_ABI_VERSION_MINOR,
    };

    /// Unpack the `(major << 16) | minor` form carried in every manifest root.
    #[must_use]
    pub const fn unpack(packed: u32) -> AbiVersion {
        AbiVersion {
            major: packed >> 16,
            minor: packed & 0xFFFF,
        }
    }

    /// Pack back into the wire form.
    #[must_use]
    pub const fn pack(self) -> u32 {
        (self.major << 16) | (self.minor & 0xFFFF)
    }

    /// Whether a host at `self` accepts a manifest at `other`: same major, and a minor no greater
    /// than the host's — a host must not read fields it does not know about.
    #[must_use]
    pub const fn accepts(self, other: AbiVersion) -> bool {
        self.major == other.major && other.minor <= self.minor
    }

    /// [`Self::accepts`] as a fallible check, so a caller can propagate the refusal.
    pub fn check(self, other: AbiVersion) -> Result<(), AbiError> {
        if self.accepts(other) {
            Ok(())
        } else {
            Err(AbiError::Version {
                found: other,
                expected: self,
            })
        }
    }
}

impl fmt::Display for AbiVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

// ------------------------------------------------------------------------------------------- //
// Hashes                                                                                        //
// ------------------------------------------------------------------------------------------- //

impl saber_hash256 {
    /// The all-zero digest. Used as "not yet computed"; a manifest that ships it is malformed, and
    /// [`saber_hash256::check_set`] is how a loader says so.
    pub const ZERO: saber_hash256 = saber_hash256 {
        bytes: [0u8; SABER_HASH_BYTES as usize],
    };

    /// Lowercase hex, the form that appears in `saber explain` and in autotune cache keys.
    #[must_use]
    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(self.bytes.len() * 2);
        for b in self.bytes {
            // Deliberately not `format!` per byte: this is called once per record in a pack of
            // thousands.
            const HEX: &[u8; 16] = b"0123456789abcdef";
            s.push(HEX[(b >> 4) as usize] as char);
            s.push(HEX[(b & 0xF) as usize] as char);
        }
        s
    }

    /// First `n` hex characters, for logs and tables where the full digest is noise.
    #[must_use]
    pub fn to_hex_short(&self, n: usize) -> String {
        let mut s = self.to_hex();
        s.truncate(n.min(self.bytes.len() * 2));
        s
    }

    /// Refuse an unset (all-zero) digest, naming what was being checked.
    pub fn check_set(&self, what: &'static str) -> Result<(), AbiError> {
        if *self == saber_hash256::ZERO {
            Err(AbiError::Malformed(what))
        } else {
            Ok(())
        }
    }
}

impl Default for saber_hash256 {
    fn default() -> Self {
        saber_hash256::ZERO
    }
}

impl fmt::Display for saber_hash256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

// ------------------------------------------------------------------------------------------- //
// Constraints                                                                                   //
// ------------------------------------------------------------------------------------------- //

/// The constraint predicates, typed.
///
/// One implementation of each, here, for the reason the reference's header gives: every integration that
/// restates them ends up with a copy that drifts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConstraintOp {
    /// `param == ival`
    Eq,
    /// `param <= ival`
    Le,
    /// `param >= ival`
    Ge,
    /// `param % ival == 0`
    Div,
    /// `param`, as a word, is one of the space-separated words in `sval`
    In,
}

impl ConstraintOp {
    /// Convert from the raw `u32` a manifest carries. An undefined operator is an error: a
    /// constraint that cannot be evaluated must not be treated as satisfied.
    pub fn from_raw(v: u32) -> Result<Self, AbiError> {
        match v {
            SABER_C_EQ => Ok(ConstraintOp::Eq),
            SABER_C_LE => Ok(ConstraintOp::Le),
            SABER_C_GE => Ok(ConstraintOp::Ge),
            SABER_C_DIV => Ok(ConstraintOp::Div),
            SABER_C_IN => Ok(ConstraintOp::In),
            other => Err(AbiError::UnknownValue {
                what: "saber_constraint_op",
                value: other,
            }),
        }
    }

    /// The raw `u32` for this operator.
    #[must_use]
    pub const fn to_raw(self) -> u32 {
        match self {
            ConstraintOp::Eq => SABER_C_EQ,
            ConstraintOp::Le => SABER_C_LE,
            ConstraintOp::Ge => SABER_C_GE,
            ConstraintOp::Div => SABER_C_DIV,
            ConstraintOp::In => SABER_C_IN,
        }
    }

    /// Evaluate a numeric constraint: does `value` satisfy `self` against `ival`?
    ///
    /// `Div` by zero is a malformed manifest, not "false" — a zero modulus would otherwise make an
    /// implementation silently unselectable with no error anywhere.
    pub fn holds(self, value: i64, ival: i64) -> Result<bool, AbiError> {
        Ok(match self {
            ConstraintOp::Eq => value == ival,
            ConstraintOp::Le => value <= ival,
            ConstraintOp::Ge => value >= ival,
            ConstraintOp::Div => {
                if ival == 0 {
                    return Err(AbiError::Malformed("SABER_C_DIV constraint with ival == 0"));
                }
                value % ival == 0
            }
            ConstraintOp::In => {
                return Err(AbiError::Malformed(
                    "SABER_C_IN is a string-set predicate; use ConstraintOp::holds_word",
                ))
            }
        })
    }

    /// Evaluate the string-set constraint: is `word` one of the space-separated words in `sval`?
    ///
    /// Splitting on ASCII whitespace, exactly as the reference's registry documents it.
    pub fn holds_word(self, word: &str, sval: &str) -> Result<bool, AbiError> {
        match self {
            ConstraintOp::In => Ok(sval.split_ascii_whitespace().any(|w| w == word)),
            _ => Err(AbiError::Malformed(
                "only SABER_C_IN takes a string-set operand",
            )),
        }
    }
}

// ------------------------------------------------------------------------------------------- //
// Workspaces                                                                                    //
// ------------------------------------------------------------------------------------------- //

/// What a workspace region is for, typed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WsKind {
    /// Aliasable between ops and between steps; contents undefined at entry.
    Scratch,
    /// Survives replay, not aliasable, zeroed per its flags.
    Persistent,
    /// Persistent and exported to the peer rank over IPC.
    IpcPeerVisible,
}

impl WsKind {
    /// Convert from the raw `u32` a manifest carries.
    pub fn from_raw(v: u32) -> Result<Self, AbiError> {
        match v {
            SABER_WS_SCRATCH => Ok(WsKind::Scratch),
            SABER_WS_PERSISTENT => Ok(WsKind::Persistent),
            SABER_WS_IPC_PEER_VISIBLE => Ok(WsKind::IpcPeerVisible),
            other => Err(AbiError::UnknownValue {
                what: "saber_ws_kind",
                value: other,
            }),
        }
    }

    /// The raw `u32` for this kind.
    #[must_use]
    pub const fn to_raw(self) -> u32 {
        match self {
            WsKind::Scratch => SABER_WS_SCRATCH,
            WsKind::Persistent => SABER_WS_PERSISTENT,
            WsKind::IpcPeerVisible => SABER_WS_IPC_PEER_VISIBLE,
        }
    }

    /// Whether the host may hand the same bytes to another op that never overlaps this one.
    #[must_use]
    pub const fn aliasable(self) -> bool {
        matches!(self, WsKind::Scratch)
    }
}

/// The affine workspace size law, detached from the raw struct so that a planner can hold it,
/// print it and resolve it without carrying `*const c_char`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WsSize {
    /// Constant term, bytes per slot.
    pub bytes: u64,
    /// Coefficient of M.
    pub per_m: u64,
    /// Coefficient of N.
    pub per_n: u64,
    /// Coefficient of K.
    pub per_k: u64,
    /// Coefficient of M*N.
    pub per_mn: u64,
}

impl WsSize {
    /// Read the law out of a raw requirement record.
    #[must_use]
    pub const fn from_req(req: &saber_ws_req) -> WsSize {
        WsSize {
            bytes: req.bytes,
            per_m: req.bytes_per_m,
            per_n: req.bytes_per_n,
            per_k: req.bytes_per_k,
            per_mn: req.bytes_per_mn,
        }
    }

    /// Resolve to bytes **per slot** for a bucket shape.
    ///
    /// Checked throughout: a manifest whose coefficients overflow at the bucket's shape is an
    /// error, because the alternative is an allocation that silently wraps.
    pub fn resolve(&self, m: u64, n: u64, k: u64) -> Result<u64, AbiError> {
        let mul = |a: u64, b: u64| a.checked_mul(b).ok_or(AbiError::SizeOverflow);
        let mn = mul(m, n)?;
        let mut total = self.bytes;
        for term in [
            mul(self.per_m, m)?,
            mul(self.per_n, n)?,
            mul(self.per_k, k)?,
            mul(self.per_mn, mn)?,
        ] {
            total = total.checked_add(term).ok_or(AbiError::SizeOverflow)?;
        }
        Ok(total)
    }
}

/// Resolve one requirement to its **total** byte size across all slots, and validate the fields
/// this ABI declares malformed: `slots == 0`, a non-power-of-two alignment, or an alignment below
/// the 16-byte floor.
pub fn ws_total_bytes(req: &saber_ws_req, m: u64, n: u64, k: u64) -> Result<u64, AbiError> {
    if req.slots == 0 {
        return Err(AbiError::Malformed("saber_ws_req.slots == 0"));
    }
    if req.align < 16 || !req.align.is_power_of_two() {
        return Err(AbiError::Malformed(
            "saber_ws_req.align must be a power of two >= 16",
        ));
    }
    WsSize::from_req(req)
        .resolve(m, n, k)?
        .checked_mul(u64::from(req.slots))
        .ok_or(AbiError::SizeOverflow)
}

// ------------------------------------------------------------------------------------------- //
// Kernarg                                                                                       //
// ------------------------------------------------------------------------------------------- //

/// Scalar type of one explicit kernel argument, typed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArgKind {
    /// Device-visible global pointer.
    Ptr,
    /// `i32`
    I32,
    /// `u32`
    U32,
    /// `i64`
    I64,
    /// `u64`
    U64,
    /// `f32`
    F32,
    /// `f64`
    F64,
    /// A struct passed by value, whose size and alignment are declared, not implied.
    ///
    /// The reference's two paged-attention kernels take `const SABERArgs a` by value: the code object
    /// reports ONE 136-byte `by_value` argument and the host writes nine pointers and eleven
    /// scalars into it. There is nothing natural about its size, which is why [`ArgKind::size`]
    /// and [`ArgKind::align`] answer `None` here and [`check_kernarg`] validates the declared
    /// pair instead.
    Struct,
}

impl ArgKind {
    /// Convert from the raw `u32` a manifest carries.
    pub fn from_raw(v: u32) -> Result<Self, AbiError> {
        match v {
            SABER_ARG_PTR => Ok(ArgKind::Ptr),
            SABER_ARG_I32 => Ok(ArgKind::I32),
            SABER_ARG_U32 => Ok(ArgKind::U32),
            SABER_ARG_I64 => Ok(ArgKind::I64),
            SABER_ARG_U64 => Ok(ArgKind::U64),
            SABER_ARG_F32 => Ok(ArgKind::F32),
            SABER_ARG_F64 => Ok(ArgKind::F64),
            SABER_ARG_STRUCT => Ok(ArgKind::Struct),
            other => Err(AbiError::UnknownValue {
                what: "saber_arg_kind",
                value: other,
            }),
        }
    }

    /// The raw `u32` for this kind.
    #[must_use]
    pub const fn to_raw(self) -> u32 {
        match self {
            ArgKind::Ptr => SABER_ARG_PTR,
            ArgKind::I32 => SABER_ARG_I32,
            ArgKind::U32 => SABER_ARG_U32,
            ArgKind::I64 => SABER_ARG_I64,
            ArgKind::U64 => SABER_ARG_U64,
            ArgKind::F32 => SABER_ARG_F32,
            ArgKind::F64 => SABER_ARG_F64,
            ArgKind::Struct => SABER_ARG_STRUCT,
        }
    }

    /// Natural size in bytes, or `None` for [`ArgKind::Struct`], which has none.
    ///
    /// The loader checks a declared argument against this and against the code object's metadata;
    /// a disagreement is memory corruption waiting to happen, so it is refused rather than
    /// adjusted. `Option` rather than a sentinel because "this kind has no natural size" is a
    /// different statement from "this kind is zero bytes", and a caller that ignores the
    /// difference should not compile.
    #[must_use]
    pub const fn size(self) -> Option<u32> {
        match self {
            ArgKind::I32 | ArgKind::U32 | ArgKind::F32 => Some(4),
            ArgKind::Ptr | ArgKind::I64 | ArgKind::U64 | ArgKind::F64 => Some(8),
            ArgKind::Struct => None,
        }
    }

    /// Natural alignment in bytes; equal to [`ArgKind::size`] for every scalar kind in this ABI,
    /// and `None` for [`ArgKind::Struct`].
    #[must_use]
    pub const fn align(self) -> Option<u32> {
        self.size()
    }
}

/// Validate one declared argument against its own kind: size, alignment, and that the offset is
/// naturally aligned. Cross-checking against the code object's msgpack metadata is the loader's
/// job and needs the `.hsaco`; this is the part that needs nothing.
///
/// [`ArgKind::Struct`] is the one kind whose size and alignment are declared rather than implied,
/// so it is checked against the C layout rules instead of against a constant: a non-zero `size`
/// that is a multiple of 4 and of `align`, an `align` that is a power of two no greater than
/// [`SABER_ARG_STRUCT_MAX_ALIGN`], and an `offset` that is a multiple of `align`.
pub fn check_kernarg(arg: &saber_kernarg) -> Result<(), AbiError> {
    let kind = ArgKind::from_raw(arg.kind)?;
    let align = match (kind.size(), kind.align()) {
        (Some(size), Some(align)) => {
            if arg.size != size {
                return Err(AbiError::Malformed(
                    "saber_kernarg.size does not match its kind",
                ));
            }
            if arg.align != align {
                return Err(AbiError::Malformed(
                    "saber_kernarg.align does not match its kind",
                ));
            }
            align
        }
        // The struct case. Every rule below is a property the AMDGPU C ABI guarantees of a real
        // by-value aggregate, so a declaration that breaks one describes a layout the compiler
        // cannot have produced.
        _ => {
            if arg.align == 0
                || !arg.align.is_power_of_two()
                || arg.align > SABER_ARG_STRUCT_MAX_ALIGN
            {
                return Err(AbiError::Malformed(
                    "saber_kernarg.align of a struct must be a power of two up to 16",
                ));
            }
            if arg.size == 0 || arg.size % 4 != 0 {
                return Err(AbiError::Malformed(
                    "saber_kernarg.size of a struct must be a non-zero multiple of 4",
                ));
            }
            if arg.size % arg.align != 0 {
                return Err(AbiError::Malformed(
                    "saber_kernarg.size of a struct must be a multiple of its alignment",
                ));
            }
            arg.align
        }
    };
    if arg.offset % align != 0 {
        return Err(AbiError::Malformed(
            "saber_kernarg.offset is not naturally aligned",
        ));
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------- //
// Implementation flags                                                                          //
// ------------------------------------------------------------------------------------------- //

/// The per-implementation flag word, typed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ImplFlags(pub u32);

impl ImplFlags {
    /// Legal inside a stream capture. The planner refuses to bind anything without this into a
    /// captured bucket.
    #[must_use]
    pub const fn graph_safe(self) -> bool {
        self.0 & SABER_IMPL_GRAPH_SAFE != 0
    }

    /// May synchronise (device-wide barrier, host sync, spin on a host flag).
    #[must_use]
    pub const fn may_sync(self) -> bool {
        self.0 & SABER_IMPL_MAY_SYNC != 0
    }

    /// Bit-reproducible for identical inputs and launch geometry. Split-K with an atomic float
    /// reduction is not, and must leave this clear.
    #[must_use]
    pub const fn deterministic(self) -> bool {
        self.0 & SABER_IMPL_DETERMINISTIC != 0
    }

    /// Touches peer-rank memory; needs IPC workspaces mapped and peer access enabled first.
    #[must_use]
    pub const fn peer_access(self) -> bool {
        self.0 & SABER_IMPL_PEER_ACCESS != 0
    }

    /// Refuse the combinations this ABI declares malformed.
    ///
    /// `GRAPH_SAFE | MAY_SYNC` is the one that matters: the two bits are different claims, and an
    /// implementation asserting both is telling the planner it is capture-legal *and* that it may
    /// synchronise, which is how a bucket ends up captured with a device-wide barrier in it.
    pub fn check(self) -> Result<(), AbiError> {
        if self.graph_safe() && self.may_sync() {
            return Err(AbiError::Malformed(
                "implementation asserts both SABER_IMPL_GRAPH_SAFE and SABER_IMPL_MAY_SYNC",
            ));
        }
        let known = SABER_IMPL_GRAPH_SAFE
            | SABER_IMPL_MAY_SYNC
            | SABER_IMPL_DETERMINISTIC
            | SABER_IMPL_PEER_ACCESS;
        if self.0 & !known != 0 {
            return Err(AbiError::Malformed(
                "implementation sets a flag bit this ABI version does not define",
            ));
        }
        Ok(())
    }
}

// ------------------------------------------------------------------------------------------- //
// Pack — the `SBRPACK1` on-disk format                                                          //
// ------------------------------------------------------------------------------------------- //

/// Element type of a pack record's payload, typed.
///
/// The discriminants are the wire tags in `docs/abi/PACK.md` § "Dtype tags" and **are never
/// renumbered**; a new type is appended. `saber-mem` parses the bytes with its own hand-written
/// decoder, so this type is the ABI's statement of the same table and the two are cross-checked in
/// `crates/saber-mem/tests/abi_layout.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum Dtype {
    /// 8 bits. Also the dtype of NVFP4 `weight_packed` at layout id 0.
    U8 = SABER_DT_U8 as u16,
    /// 8 bits, signed.
    I8 = SABER_DT_I8 as u16,
    /// 16 bits.
    U16 = SABER_DT_U16 as u16,
    /// 16 bits, signed.
    I16 = SABER_DT_I16 as u16,
    /// 32 bits.
    U32 = SABER_DT_U32 as u16,
    /// 32 bits, signed.
    I32 = SABER_DT_I32 as u16,
    /// 64 bits.
    U64 = SABER_DT_U64 as u16,
    /// 64 bits, signed.
    I64 = SABER_DT_I64 as u16,
    /// IEEE binary16.
    F16 = SABER_DT_F16 as u16,
    /// bfloat16.
    Bf16 = SABER_DT_BF16 as u16,
    /// IEEE binary32.
    F32 = SABER_DT_F32 as u16,
    /// IEEE binary64.
    F64 = SABER_DT_F64 as u16,
    /// OCP e4m3.
    F8E4M3 = SABER_DT_F8E4M3 as u16,
    /// OCP e5m2.
    F8E5M2 = SABER_DT_F8E5M2 as u16,
    /// Two e2m1 per byte, low nibble = even element; `shape` is the logical element count.
    Fp4E2M1X2 = SABER_DT_FP4E2M1X2 as u16,
    /// One byte per boolean.
    Bool = SABER_DT_BOOL as u16,
    /// MX block-scale exponent.
    E8M0 = SABER_DT_E8M0 as u16,
}

impl Dtype {
    /// Decode a wire tag. An unknown tag is an error, never a default.
    pub fn from_tag(tag: u16) -> Result<Self, AbiError> {
        Ok(match u32::from(tag) {
            SABER_DT_U8 => Dtype::U8,
            SABER_DT_I8 => Dtype::I8,
            SABER_DT_U16 => Dtype::U16,
            SABER_DT_I16 => Dtype::I16,
            SABER_DT_U32 => Dtype::U32,
            SABER_DT_I32 => Dtype::I32,
            SABER_DT_U64 => Dtype::U64,
            SABER_DT_I64 => Dtype::I64,
            SABER_DT_F16 => Dtype::F16,
            SABER_DT_BF16 => Dtype::Bf16,
            SABER_DT_F32 => Dtype::F32,
            SABER_DT_F64 => Dtype::F64,
            SABER_DT_F8E4M3 => Dtype::F8E4M3,
            SABER_DT_F8E5M2 => Dtype::F8E5M2,
            SABER_DT_FP4E2M1X2 => Dtype::Fp4E2M1X2,
            SABER_DT_BOOL => Dtype::Bool,
            SABER_DT_E8M0 => Dtype::E8M0,
            other => {
                return Err(AbiError::UnknownValue {
                    what: "saber_pack_dtype",
                    value: other,
                })
            }
        })
    }

    /// The wire tag.
    #[must_use]
    pub const fn tag(self) -> u16 {
        self as u16
    }

    /// Bits per **logical** element. `length == ceil(prod(shape) * bits / 8)` is computed from
    /// this, which is why `Fp4E2M1X2` is 4 and not 8.
    #[must_use]
    pub const fn bits(self) -> u64 {
        match self {
            Dtype::Fp4E2M1X2 => 4,
            Dtype::U8 | Dtype::I8 | Dtype::F8E4M3 | Dtype::F8E5M2 | Dtype::Bool | Dtype::E8M0 => 8,
            Dtype::U16 | Dtype::I16 | Dtype::F16 | Dtype::Bf16 => 16,
            Dtype::U32 | Dtype::I32 | Dtype::F32 => 32,
            Dtype::U64 | Dtype::I64 | Dtype::F64 => 64,
        }
    }
}

/// How a tensor was split across TP ranks, typed.
///
/// Informational: the record's bytes are already the rank's bytes and nothing on the load path
/// re-derives a split.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ShardKind {
    /// Identical bytes on every rank.
    Replicated = SABER_SHARD_REPLICATED as u8,
    /// Column-parallel: split along dim 0.
    Dim0 = SABER_SHARD_DIM0 as u8,
    /// Row-parallel: split along dim 1 (the reduction dim).
    Dim1 = SABER_SHARD_DIM1 as u8,
    /// Dim 0 is a concatenation of segments, each split independently (fused QKV).
    Dim0Segments = SABER_SHARD_DIM0_SEGMENTS as u8,
}

impl ShardKind {
    /// Decode a wire tag.
    pub fn from_tag(tag: u8) -> Result<Self, AbiError> {
        Ok(match u32::from(tag) {
            SABER_SHARD_REPLICATED => ShardKind::Replicated,
            SABER_SHARD_DIM0 => ShardKind::Dim0,
            SABER_SHARD_DIM1 => ShardKind::Dim1,
            SABER_SHARD_DIM0_SEGMENTS => ShardKind::Dim0Segments,
            other => {
                return Err(AbiError::UnknownValue {
                    what: "saber_pack_shard_kind",
                    value: other,
                })
            }
        })
    }

    /// The wire tag.
    #[must_use]
    pub const fn tag(self) -> u8 {
        self as u8
    }
}

/// A pack record's layout id.
///
/// Open-ended by design: `Raw` is the only id defined so far, device-final layouts are allocated in
/// `docs/abi/PACK.md` § "Layout ids" as they are implemented, and the private range is for
/// experiments that must never be loadable by a released engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayoutId {
    /// Raw checkpoint bytes, no transform (`layout_id == 0`).
    Raw,
    /// A device-final layout allocated in PACK.md's table.
    DeviceFinal(u32),
    /// A private id at or above [`SABER_LAYOUT_PRIVATE_BASE`].
    Private(u32),
}

impl LayoutId {
    /// Classify a raw layout id. Unlike every other enumeration here this one cannot fail: the
    /// range is open by specification. Whether *this build* can consume the layout is the
    /// planner's question, not the format's.
    #[must_use]
    pub fn from_raw(v: u32) -> Self {
        match v {
            SABER_LAYOUT_RAW => LayoutId::Raw,
            v if v >= SABER_LAYOUT_PRIVATE_BASE => LayoutId::Private(v),
            v => LayoutId::DeviceFinal(v),
        }
    }

    /// The raw `u32`.
    #[must_use]
    pub const fn to_raw(self) -> u32 {
        match self {
            LayoutId::Raw => SABER_LAYOUT_RAW,
            LayoutId::DeviceFinal(v) | LayoutId::Private(v) => v,
        }
    }
}

/// Round `x` up to the next multiple of [`SABER_PACK_ALIGN`].
///
/// Every record starts on such a boundary and a streaming loader transfers `align_up(length)`
/// bytes, which is what lets it read with `O_DIRECT` into an aligned buffer with no bounce.
#[must_use]
pub fn pack_align_up(x: u64) -> u64 {
    let a = u64::from(SABER_PACK_ALIGN);
    x.div_ceil(a) * a
}

/// Validate one on-disk record entry against the rules in PACK.md that need only the entry itself.
///
/// Deliberately **not** the whole rule set: `data_off` ordering, non-overlap after padding, name
/// uniqueness and `table_sha256` need the table and the file, and they belong to
/// `saber_mem::pack::PackReader`. What is checked here is what a single 160-byte entry can be
/// judged on — and every one of these is a refusal, because a record that cannot be judged cannot
/// be loaded transform-free, which is the entire point of the pack.
pub fn check_pack_entry(e: &saber_pack_table_entry) -> Result<(), AbiError> {
    if e.flags != 0 {
        return Err(AbiError::Malformed(
            "pack record flags must be zero in format version 1",
        ));
    }
    if e.reserved.iter().any(|&w| w != 0) {
        return Err(AbiError::Malformed(
            "pack record reserved words must be zero",
        ));
    }
    if e.name_off < SABER_PACK_NAME_PREFIX_BYTES || e.name_len == 0 {
        return Err(AbiError::Malformed(
            "pack record name_off must be >= 2 (the u16 prefix precedes it) and name_len >= 1",
        ));
    }
    let ndim = u32::from(e.ndim);
    if ndim == 0 || ndim > SABER_PACK_MAX_DIMS {
        return Err(AbiError::Malformed(
            "pack record ndim must be 1..=SABER_PACK_MAX_DIMS",
        ));
    }
    for (d, &v) in e.shape.iter().enumerate() {
        let within = (d as u32) < ndim;
        if within && v == 0 {
            return Err(AbiError::Malformed(
                "pack record shape has a zero extent within ndim",
            ));
        }
        if !within && v != 0 {
            return Err(AbiError::Malformed(
                "pack record shape is non-zero beyond ndim",
            ));
        }
    }
    if e.data_off % u64::from(SABER_PACK_ALIGN) != 0 {
        return Err(AbiError::Malformed(
            "pack record data_off is not SABER_PACK_ALIGN-aligned",
        ));
    }
    if e.length == 0 {
        return Err(AbiError::Malformed("pack record length is zero"));
    }
    let dtype = Dtype::from_tag(e.dtype)?;
    ShardKind::from_tag(e.shard_kind)?;
    let elements = e
        .shape
        .iter()
        .take(ndim as usize)
        .try_fold(1u64, |acc, &v| acc.checked_mul(v))
        .ok_or(AbiError::SizeOverflow)?;
    let implied = elements
        .checked_mul(dtype.bits())
        .ok_or(AbiError::SizeOverflow)?
        .div_ceil(8);
    if implied != e.length {
        return Err(AbiError::Malformed(
            "pack record length does not equal ceil(prod(shape) * dtype_bits / 8)",
        ));
    }
    e.sha256
        .check_set("pack record sha256 is all zero; the load-time integrity check would be a no-op")
}

/// Validate a decoded file header against the rules in PACK.md that need only the header.
///
/// `table_sha256` is **not** checked here: it needs the table and the name blob, and PACK.md
/// requires it to be checked *after* the structural rules so that a malformed table reports the
/// structural fault rather than a hash mismatch that explains nothing.
pub fn check_pack_header(h: &saber_pack_file_header) -> Result<(), AbiError> {
    if h.magic != *b"SBRPACK1" {
        return Err(AbiError::Malformed("pack magic is not SBRPACK1"));
    }
    if h.format_version < SABER_PACK_FORMAT_VERSION_MIN
        || h.format_version > SABER_PACK_FORMAT_VERSION
    {
        return Err(AbiError::Malformed(
            "pack format_version is outside the range this build reads; it refuses rather than \
             guesses",
        ));
    }
    if h.header_bytes != SABER_PACK_HEADER_BYTES || h.record_bytes != SABER_PACK_RECORD_BYTES {
        return Err(AbiError::Malformed(
            "pack header_bytes/record_bytes are not 256/160",
        ));
    }
    if h.reserved.iter().any(|&b| b != 0) {
        return Err(AbiError::Malformed(
            "pack header reserved bytes must be zero",
        ));
    }
    if h.table_off != u64::from(SABER_PACK_HEADER_BYTES) {
        return Err(AbiError::Malformed("pack table_off must equal 256"));
    }
    let want_blob = u64::from(h.record_count)
        .checked_mul(u64::from(SABER_PACK_RECORD_BYTES))
        .and_then(|t| t.checked_add(h.table_off))
        .ok_or(AbiError::SizeOverflow)?;
    if h.name_blob_off != want_blob {
        return Err(AbiError::Malformed(
            "pack name_blob_off must equal table_off + record_count * record_bytes",
        ));
    }
    let align = u64::from(SABER_PACK_ALIGN);
    if h.data_off % align != 0 || h.file_bytes % align != 0 {
        return Err(AbiError::Malformed(
            "pack data_off and file_bytes must be SABER_PACK_ALIGN-aligned",
        ));
    }
    // v2's chunk blob sits between the name blob and the data; v1 has none and all four of its
    // header fields must be zero.
    let names_end = h
        .name_blob_off
        .checked_add(h.name_blob_bytes)
        .ok_or(AbiError::SizeOverflow)?;
    if h.chunk_blob_bytes == 0 {
        if h.chunk_blob_off != 0
            || h.chunk_bytes != 0
            || h.chunk_blob_sha256.bytes.iter().any(|&b| b != 0)
        {
            return Err(AbiError::Malformed(
                "a pack with no chunk blob must have all four chunk header fields zero",
            ));
        }
    } else {
        if h.format_version < 2 {
            return Err(AbiError::Malformed(
                "a chunk blob needs format_version 2 or later",
            ));
        }
        if h.chunk_blob_off != names_end {
            return Err(AbiError::Malformed(
                "pack chunk_blob_off must be immediately after the name blob",
            ));
        }
        if h.chunk_bytes == 0 || u64::from(h.chunk_bytes) % u64::from(SABER_PACK_ALIGN) != 0 {
            return Err(AbiError::Malformed(
                "pack chunk_bytes must be a non-zero multiple of SABER_PACK_ALIGN",
            ));
        }
    }
    let meta_end = pack_align_up(if h.chunk_blob_bytes == 0 {
        names_end
    } else {
        h.chunk_blob_off
            .checked_add(h.chunk_blob_bytes)
            .ok_or(AbiError::SizeOverflow)?
    });
    if h.data_off < meta_end {
        return Err(AbiError::Malformed("pack data_off overlaps the metadata"));
    }
    if h.file_bytes < h.data_off {
        return Err(AbiError::Malformed("pack file_bytes is before data_off"));
    }
    if h.tp_size == 0 || h.rank >= h.tp_size {
        return Err(AbiError::Malformed(
            "pack rank must be less than tp_size, and tp_size at least 1",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_packs_and_unpacks() {
        assert_eq!(AbiVersion::CURRENT.pack(), SABER_ABI_VERSION);
        assert_eq!(AbiVersion::unpack(SABER_ABI_VERSION), AbiVersion::CURRENT);
        let v = AbiVersion { major: 3, minor: 7 };
        assert_eq!(AbiVersion::unpack(v.pack()), v);
    }

    #[test]
    fn version_accepts_equal_and_older_minor_only() {
        let host = AbiVersion { major: 1, minor: 4 };
        assert!(host.accepts(host));
        assert!(host.accepts(AbiVersion { major: 1, minor: 0 }));
        // Newer minor: the host would be reading fields it does not know.
        assert!(!host.accepts(AbiVersion { major: 1, minor: 5 }));
        // Different major: refused outright, in both directions.
        assert!(!host.accepts(AbiVersion { major: 2, minor: 0 }));
        assert!(!host.accepts(AbiVersion { major: 0, minor: 4 }));
        assert!(host.check(AbiVersion { major: 1, minor: 5 }).is_err());
    }

    #[test]
    fn constraint_ops_round_trip_and_reject_unknown() {
        for op in [
            ConstraintOp::Eq,
            ConstraintOp::Le,
            ConstraintOp::Ge,
            ConstraintOp::Div,
            ConstraintOp::In,
        ] {
            assert_eq!(ConstraintOp::from_raw(op.to_raw()).unwrap(), op);
        }
        assert_eq!(
            ConstraintOp::from_raw(5),
            Err(AbiError::UnknownValue {
                what: "saber_constraint_op",
                value: 5
            })
        );
    }

    #[test]
    fn constraint_predicates_match_the_header() {
        assert!(ConstraintOp::Eq.holds(256, 256).unwrap());
        assert!(!ConstraintOp::Eq.holds(255, 256).unwrap());
        assert!(ConstraintOp::Le.holds(64, 64).unwrap());
        assert!(!ConstraintOp::Le.holds(65, 64).unwrap());
        assert!(ConstraintOp::Ge.holds(64, 64).unwrap());
        assert!(!ConstraintOp::Ge.holds(63, 64).unwrap());
        assert!(ConstraintOp::Div.holds(4096, 128).unwrap());
        assert!(!ConstraintOp::Div.holds(4097, 128).unwrap());
    }

    #[test]
    fn div_by_zero_is_malformed_not_false() {
        // The negative control for "a zero modulus silently makes a kernel unselectable".
        assert!(ConstraintOp::Div.holds(4096, 0).is_err());
    }

    #[test]
    fn in_is_a_string_predicate_and_the_others_are_not() {
        assert!(ConstraintOp::In
            .holds_word("bf16", "bf16 fp16 fp32")
            .unwrap());
        assert!(!ConstraintOp::In
            .holds_word("fp8", "bf16 fp16 fp32")
            .unwrap());
        // Leading/trailing/multiple spaces are not words.
        assert!(ConstraintOp::In
            .holds_word("fp32", "  bf16   fp32 ")
            .unwrap());
        assert!(!ConstraintOp::In.holds_word("", "bf16 fp16").unwrap());
        assert!(ConstraintOp::In.holds(1, 1).is_err());
        assert!(ConstraintOp::Eq.holds_word("bf16", "bf16").is_err());
    }

    fn ws(bytes: u64, per_m: u64, per_n: u64, per_k: u64, per_mn: u64) -> saber_ws_req {
        saber_ws_req {
            name: core::ptr::null(),
            bytes,
            bytes_per_m: per_m,
            bytes_per_n: per_n,
            bytes_per_k: per_k,
            bytes_per_mn: per_mn,
            kind: SABER_WS_SCRATCH,
            align: 256,
            slots: 1,
            flags: 0,
        }
    }

    #[test]
    fn workspace_size_law_covers_the_real_cases() {
        // A persistent block counter: constant only.
        let counter = ws(4096, 0, 0, 0, 0);
        assert_eq!(ws_total_bytes(&counter, 2048, 5120, 4096).unwrap(), 4096);

        // Split-K partials at 4 splits, fp32: 4 * 4 bytes per (m, n) element.
        let partials = ws(0, 0, 0, 0, 16);
        assert_eq!(
            ws_total_bytes(&partials, 24, 5120, 4096).unwrap(),
            24 * 5120 * 16
        );

        // Two slots of an all-reduce staging buffer sized per row.
        let mut ar = ws(128, 8192, 0, 0, 0);
        ar.slots = 2;
        assert_eq!(
            ws_total_bytes(&ar, 40, 0, 0).unwrap(),
            (128 + 40 * 8192) * 2
        );
    }

    #[test]
    fn workspace_size_law_is_checked_not_wrapped() {
        let huge = ws(0, 0, 0, 0, u64::MAX / 2);
        assert_eq!(
            ws_total_bytes(&huge, 1 << 20, 1 << 20, 0),
            Err(AbiError::SizeOverflow)
        );
    }

    #[test]
    fn workspace_zero_slots_and_bad_align_are_malformed() {
        let mut w = ws(64, 0, 0, 0, 0);
        w.slots = 0;
        assert!(ws_total_bytes(&w, 1, 1, 1).is_err());

        let mut w = ws(64, 0, 0, 0, 0);
        w.align = 24; // not a power of two
        assert!(ws_total_bytes(&w, 1, 1, 1).is_err());

        let mut w = ws(64, 0, 0, 0, 0);
        w.align = 8; // below the floor
        assert!(ws_total_bytes(&w, 1, 1, 1).is_err());
    }

    #[test]
    fn ws_kinds_round_trip_and_only_scratch_aliases() {
        for k in [WsKind::Scratch, WsKind::Persistent, WsKind::IpcPeerVisible] {
            assert_eq!(WsKind::from_raw(k.to_raw()).unwrap(), k);
        }
        assert!(WsKind::Scratch.aliasable());
        assert!(!WsKind::Persistent.aliasable());
        assert!(!WsKind::IpcPeerVisible.aliasable());
        assert!(WsKind::from_raw(3).is_err());
    }

    #[test]
    fn arg_kinds_have_the_sizes_the_kernarg_packer_assumes() {
        assert_eq!(ArgKind::Ptr.size(), Some(8));
        assert_eq!(ArgKind::I32.size(), Some(4));
        assert_eq!(ArgKind::F64.size(), Some(8));
        for k in [
            ArgKind::Ptr,
            ArgKind::I32,
            ArgKind::U32,
            ArgKind::I64,
            ArgKind::U64,
            ArgKind::F32,
            ArgKind::F64,
            ArgKind::Struct,
        ] {
            assert_eq!(ArgKind::from_raw(k.to_raw()).unwrap(), k);
            assert_eq!(k.align(), k.size());
        }
        // A struct has no natural size, and says so rather than answering 0.
        assert_eq!(ArgKind::Struct.size(), None);
        assert_eq!(ArgKind::Struct.align(), None);
        assert!(ArgKind::from_raw(8).is_err());
    }

    #[test]
    fn a_by_value_struct_is_checked_against_the_c_layout_rules_not_a_constant() {
        // The reference's `SABERArgs`: 136 bytes, 8-aligned, first argument of both attention kernels.
        let saber_args = saber_kernarg {
            name: core::ptr::null(),
            kind: SABER_ARG_STRUCT,
            offset: 0,
            size: 136,
            align: 8,
        };
        assert!(check_kernarg(&saber_args).is_ok());

        // Each negative control below is a layout the AMDGPU C ABI cannot have produced.
        let mut odd_size = saber_args;
        odd_size.size = 134; // not a multiple of 4
        assert!(check_kernarg(&odd_size).is_err());

        let mut not_a_multiple_of_align = saber_args;
        not_a_multiple_of_align.size = 132; // 4-aligned but sizeof is not a multiple of alignof
        assert!(check_kernarg(&not_a_multiple_of_align).is_err());

        let mut empty = saber_args;
        empty.size = 0;
        assert!(check_kernarg(&empty).is_err());

        let mut align_not_pow2 = saber_args;
        align_not_pow2.align = 12;
        assert!(check_kernarg(&align_not_pow2).is_err());

        let mut align_too_big = saber_args;
        align_too_big.align = 32; // past the kernarg segment's own 16-byte base alignment
        align_too_big.size = 160;
        assert!(check_kernarg(&align_too_big).is_err());

        let mut misplaced = saber_args;
        misplaced.offset = 4; // 8-aligned struct cannot start at 4
        assert!(check_kernarg(&misplaced).is_err());

        // Positive control for the alignment rule: the same struct at a legal offset.
        let mut later = saber_args;
        later.offset = 8;
        assert!(check_kernarg(&later).is_ok());
    }

    #[test]
    fn kernarg_checks_catch_the_layout_mistakes_that_corrupt_memory() {
        let ok = saber_kernarg {
            name: core::ptr::null(),
            kind: SABER_ARG_PTR,
            offset: 16,
            size: 8,
            align: 8,
        };
        assert!(check_kernarg(&ok).is_ok());

        let mut wrong_size = ok;
        wrong_size.size = 4;
        assert!(check_kernarg(&wrong_size).is_err());

        let mut unaligned = ok;
        unaligned.offset = 12;
        assert!(check_kernarg(&unaligned).is_err());

        let mut wrong_align = ok;
        wrong_align.align = 4;
        assert!(check_kernarg(&wrong_align).is_err());
    }

    #[test]
    fn impl_flags_read_the_bits_and_refuse_the_contradiction() {
        let g = ImplFlags(SABER_IMPL_GRAPH_SAFE | SABER_IMPL_DETERMINISTIC);
        assert!(g.graph_safe());
        assert!(g.deterministic());
        assert!(!g.may_sync());
        assert!(!g.peer_access());
        assert!(g.check().is_ok());

        let contradiction = ImplFlags(SABER_IMPL_GRAPH_SAFE | SABER_IMPL_MAY_SYNC);
        assert!(contradiction.check().is_err());

        let undefined = ImplFlags(1 << 31);
        assert!(undefined.check().is_err());
    }

    #[test]
    fn dtype_tags_are_the_table_in_pack_md_and_are_never_renumbered() {
        // Written out as literals on purpose: this is the one place where a renumbering of the
        // on-disk tags shows up as a test failure rather than as a tier that loads wrong.
        for (tag, d, bits) in [
            (0u16, Dtype::U8, 8u64),
            (1, Dtype::I8, 8),
            (2, Dtype::U16, 16),
            (3, Dtype::I16, 16),
            (4, Dtype::U32, 32),
            (5, Dtype::I32, 32),
            (6, Dtype::U64, 64),
            (7, Dtype::I64, 64),
            (8, Dtype::F16, 16),
            (9, Dtype::Bf16, 16),
            (10, Dtype::F32, 32),
            (11, Dtype::F64, 64),
            (12, Dtype::F8E4M3, 8),
            (13, Dtype::F8E5M2, 8),
            (14, Dtype::Fp4E2M1X2, 4),
            (15, Dtype::Bool, 8),
            (16, Dtype::E8M0, 8),
        ] {
            assert_eq!(d.tag(), tag, "{d:?} must keep tag {tag}");
            assert_eq!(Dtype::from_tag(tag).unwrap(), d);
            assert_eq!(d.bits(), bits, "{d:?} bits");
        }
        // One past the end of the table, and a wild value: both refuse.
        assert!(Dtype::from_tag(17).is_err());
        assert!(Dtype::from_tag(9999).is_err());
    }

    #[test]
    fn shard_kinds_are_the_table_in_pack_md() {
        for (tag, k) in [
            (0u8, ShardKind::Replicated),
            (1, ShardKind::Dim0),
            (2, ShardKind::Dim1),
            (3, ShardKind::Dim0Segments),
        ] {
            assert_eq!(k.tag(), tag);
            assert_eq!(ShardKind::from_tag(tag).unwrap(), k);
        }
        assert!(ShardKind::from_tag(4).is_err());
    }

    #[test]
    fn layout_ids_split_raw_from_device_final_from_private() {
        assert_eq!(LayoutId::from_raw(0), LayoutId::Raw);
        assert_eq!(LayoutId::from_raw(0).to_raw(), SABER_LAYOUT_RAW);
        assert_eq!(LayoutId::from_raw(1), LayoutId::DeviceFinal(1));
        assert_eq!(
            LayoutId::from_raw(SABER_LAYOUT_PRIVATE_BASE - 1),
            LayoutId::DeviceFinal(SABER_LAYOUT_PRIVATE_BASE - 1)
        );
        assert_eq!(
            LayoutId::from_raw(SABER_LAYOUT_PRIVATE_BASE),
            LayoutId::Private(SABER_LAYOUT_PRIVATE_BASE)
        );
        assert_eq!(LayoutId::from_raw(u32::MAX), LayoutId::Private(u32::MAX));
        // The range is open by specification, so classification cannot fail; whether this build
        // can consume a given layout is the planner's question, not the format's.
        for v in [0, 1, 7, SABER_LAYOUT_PRIVATE_BASE, u32::MAX] {
            assert_eq!(LayoutId::from_raw(v).to_raw(), v);
        }
    }

    #[test]
    fn align_up_matches_the_4096_rule() {
        assert_eq!(pack_align_up(0), 0);
        assert_eq!(pack_align_up(1), 4096);
        assert_eq!(pack_align_up(4096), 4096);
        assert_eq!(pack_align_up(4097), 8192);
    }

    fn entry() -> saber_pack_table_entry {
        saber_pack_table_entry {
            name_off: 2,
            name_len: 4,
            rank: 0,
            layout_id: SABER_LAYOUT_RAW,
            dtype: Dtype::Bf16.tag(),
            ndim: 2,
            shard_kind: ShardKind::Dim0.tag(),
            flags: 0,
            data_off: 4096,
            // 16 * 64 bf16 = 2048 bytes.
            length: 2048,
            shape: [16, 64, 0, 0, 0, 0, 0, 0],
            sha256: saber_hash256 { bytes: [7u8; 32] },
            reserved: [0; 3],
        }
    }

    fn header() -> saber_pack_file_header {
        saber_pack_file_header {
            magic: *b"SBRPACK1",
            format_version: SABER_PACK_FORMAT_VERSION,
            header_bytes: SABER_PACK_HEADER_BYTES,
            record_bytes: SABER_PACK_RECORD_BYTES,
            record_count: 2,
            table_off: 256,
            name_blob_off: 256 + 2 * 160,
            name_blob_bytes: 64,
            data_off: 4096,
            file_bytes: 8192,
            tp_size: 2,
            rank: 1,
            build_unix_seconds: 1_758_000_000,
            plan_hash: saber_hash256 { bytes: [1u8; 32] },
            tier_id: saber_hash256 { bytes: [2u8; 32] },
            table_sha256: saber_hash256 { bytes: [3u8; 32] },
            chunk_blob_off: 256 + 2 * 160 + 64,
            chunk_blob_bytes: 136,
            chunk_blob_sha256: saber_hash256 { bytes: [4u8; 32] },
            chunk_bytes: 4 << 20,
            reserved: [0; SABER_PACK_HEADER_RESERVED_BYTES as usize],
        }
    }

    #[test]
    fn a_well_formed_entry_and_header_pass() {
        assert!(check_pack_entry(&entry()).is_ok());
        assert!(check_pack_header(&header()).is_ok());
    }

    #[test]
    fn entry_checks_are_the_negative_controls() {
        let bad = |f: fn(&mut saber_pack_table_entry)| {
            let mut e = entry();
            f(&mut e);
            assert!(check_pack_entry(&e).is_err());
        };
        bad(|e| e.flags = 1); // v1 says zero
        bad(|e| e.reserved[2] = 1); // reserved must be zero
        bad(|e| e.name_off = 1); // the u16 prefix does not fit before it
        bad(|e| e.name_len = 0);
        bad(|e| e.ndim = 0);
        bad(|e| e.ndim = (SABER_PACK_MAX_DIMS + 1) as u8);
        bad(|e| e.shape[1] = 0); // zero extent inside ndim
        bad(|e| e.shape[4] = 3); // non-zero extent beyond ndim
        bad(|e| e.data_off = 4095); // not 4096-aligned
        bad(|e| e.length = 0);
        bad(|e| e.length = 2049); // != ceil(prod(shape) * bits / 8)
        bad(|e| e.dtype = 17); // one past the dtype table
        bad(|e| e.shard_kind = 4); // one past the shard table
        bad(|e| e.sha256 = saber_hash256::ZERO); // an unset hash makes the load check a no-op
    }

    #[test]
    fn sub_byte_lengths_follow_the_logical_element_count() {
        // FP4E2M1X2: shape is the LOGICAL element count and length is half of it.
        let mut e = entry();
        e.dtype = Dtype::Fp4E2M1X2.tag();
        e.shape = [16, 64, 0, 0, 0, 0, 0, 0];
        e.length = 16 * 64 / 2;
        assert!(check_pack_entry(&e).is_ok());
        e.length = 16 * 64; // the byte count you get by forgetting the packing
        assert!(check_pack_entry(&e).is_err());
    }

    #[test]
    fn header_checks_are_the_negative_controls() {
        let bad = |f: fn(&mut saber_pack_file_header)| {
            let mut h = header();
            f(&mut h);
            assert!(check_pack_header(&h).is_err());
        };
        bad(|h| h.magic = *b"SABRPACK"); // the 0.1 spelling, which is now wrong
        bad(|h| h.format_version = SABER_PACK_FORMAT_VERSION + 1);
        bad(|h| h.format_version = 0);
        bad(|h| h.header_bytes = 128);
        bad(|h| h.record_bytes = 128);
        bad(|h| h.reserved[SABER_PACK_HEADER_RESERVED_BYTES as usize - 1] = 1);
        bad(|h| h.table_off = 0);
        bad(|h| h.name_blob_off = 256); // must be table_off + count * 160
        bad(|h| h.data_off = 4095);
        bad(|h| h.file_bytes = 8191);
        bad(|h| h.data_off = 0); // overlaps the metadata
        bad(|h| h.file_bytes = 0); // before data_off
        bad(|h| h.tp_size = 0);
        bad(|h| h.rank = 2); // == tp_size
    }

    #[test]
    fn hashes_render_as_lowercase_hex() {
        let mut h = saber_hash256::ZERO;
        h.bytes[0] = 0xde;
        h.bytes[1] = 0xad;
        h.bytes[31] = 0x0f;
        let hex = h.to_hex();
        assert_eq!(hex.len(), 64);
        assert!(hex.starts_with("dead"));
        assert!(hex.ends_with("0f"));
        assert_eq!(h.to_hex_short(8), "dead0000");
        assert_eq!(saber_hash256::ZERO.to_hex(), "0".repeat(64));
        assert_eq!(h.to_string(), hex);
    }

    #[test]
    fn the_wire_structs_are_exactly_the_sizes_pack_md_fixes() {
        // bindgen emits its own layout assertions for these; asserting the two documented
        // constants here as well is what makes a size change a failure in *this* crate's suite,
        // where the reader's cross-check in saber-mem reads them from.
        assert_eq!(
            core::mem::size_of::<saber_pack_file_header>(),
            SABER_PACK_HEADER_BYTES as usize
        );
        assert_eq!(
            core::mem::size_of::<saber_pack_table_entry>(),
            SABER_PACK_RECORD_BYTES as usize
        );
        assert_eq!(SABER_PACK_HEADER_BYTES, 256);
        assert_eq!(SABER_PACK_RECORD_BYTES, 160);
        assert_eq!(SABER_PACK_ALIGN, 4096);
        assert_eq!(SABER_PACK_MAX_DIMS, 8);
        assert_eq!(SABER_PACK_FORMAT_VERSION, 2);
        assert_eq!(SABER_PACK_FORMAT_VERSION_MIN, 1);
        assert_eq!(SABER_PACK_CHUNK_ENTRY_HEADER, 40);
        assert_eq!(
            core::mem::size_of::<saber_pack_chunk_entry>(),
            SABER_PACK_CHUNK_ENTRY_HEADER as usize
        );
    }

    #[test]
    fn pack_magic_is_sbrpack1() {
        // The 0.2 change that breaks every 0.1 reader, spelled out so a revert is loud.
        assert_eq!(SABER_PACK_MAGIC, b"SBRPACK1\0");
        assert_eq!(SABER_PACK_MAGIC_BYTES as usize, SABER_PACK_MAGIC.len() - 1);
    }

    #[test]
    fn the_declared_offsets_tile_both_structs_with_no_holes() {
        // Offsets are what saber-mem's hand-written decoder indexes with, so "they add up" is a
        // property worth asserting rather than eyeballing against the table in PACK.md.
        let header_fields: &[(u32, u32)] = &[
            (SABER_PACK_HDR_OFF_MAGIC, SABER_PACK_MAGIC_BYTES),
            (SABER_PACK_HDR_OFF_FORMAT_VERSION, 4),
            (SABER_PACK_HDR_OFF_HEADER_BYTES, 4),
            (SABER_PACK_HDR_OFF_RECORD_BYTES, 4),
            (SABER_PACK_HDR_OFF_RECORD_COUNT, 4),
            (SABER_PACK_HDR_OFF_TABLE_OFF, 8),
            (SABER_PACK_HDR_OFF_NAME_BLOB_OFF, 8),
            (SABER_PACK_HDR_OFF_NAME_BLOB_BYTES, 8),
            (SABER_PACK_HDR_OFF_DATA_OFF, 8),
            (SABER_PACK_HDR_OFF_FILE_BYTES, 8),
            (SABER_PACK_HDR_OFF_TP_SIZE, 4),
            (SABER_PACK_HDR_OFF_RANK, 4),
            (SABER_PACK_HDR_OFF_BUILD_UNIX_SECONDS, 8),
            (SABER_PACK_HDR_OFF_PLAN_HASH, SABER_HASH_BYTES),
            (SABER_PACK_HDR_OFF_TIER_ID, SABER_HASH_BYTES),
            (SABER_PACK_HDR_OFF_TABLE_SHA256, SABER_HASH_BYTES),
            (SABER_PACK_HDR_OFF_CHUNK_BLOB_OFF, 8),
            (SABER_PACK_HDR_OFF_CHUNK_BLOB_BYTES, 8),
            (SABER_PACK_HDR_OFF_CHUNK_BLOB_SHA256, SABER_HASH_BYTES),
            (SABER_PACK_HDR_OFF_CHUNK_BYTES, 4),
            (
                SABER_PACK_HDR_OFF_RESERVED,
                SABER_PACK_HEADER_RESERVED_BYTES,
            ),
        ];
        let record_fields: &[(u32, u32)] = &[
            (SABER_PACK_REC_OFF_NAME_OFF, 4),
            (SABER_PACK_REC_OFF_NAME_LEN, 4),
            (SABER_PACK_REC_OFF_RANK, 4),
            (SABER_PACK_REC_OFF_LAYOUT_ID, 4),
            (SABER_PACK_REC_OFF_DTYPE, 2),
            (SABER_PACK_REC_OFF_NDIM, 1),
            (SABER_PACK_REC_OFF_SHARD_KIND, 1),
            (SABER_PACK_REC_OFF_FLAGS, 4),
            (SABER_PACK_REC_OFF_DATA_OFF, 8),
            (SABER_PACK_REC_OFF_LENGTH, 8),
            (SABER_PACK_REC_OFF_SHAPE, 8 * SABER_PACK_MAX_DIMS),
            (SABER_PACK_REC_OFF_SHA256, SABER_HASH_BYTES),
            (
                SABER_PACK_REC_OFF_RESERVED,
                8 * SABER_PACK_RECORD_RESERVED_U64,
            ),
        ];
        for (fields, total, what) in [
            (header_fields, SABER_PACK_HEADER_BYTES, "header"),
            (record_fields, SABER_PACK_RECORD_BYTES, "record"),
        ] {
            let mut cursor = 0u32;
            for &(off, size) in fields {
                assert_eq!(
                    off, cursor,
                    "{what}: field at {off} leaves a hole at {cursor}"
                );
                cursor += size;
            }
            assert_eq!(cursor, total, "{what}: fields do not sum to {total}");
        }
    }
}
