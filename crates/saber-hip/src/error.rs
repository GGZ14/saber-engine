//! HIP status codes and this crate's error type.
//!
//! The rule this module exists to enforce: **an unknown HIP error code logs and returns `Err`; it
//! never panics the step loop and it is never matched exhaustively.** This box runs a 7.15.0-alpha
//! ROCm userspace in some images and a 7.14 one in others, new codes appear between versions, and
//! the engine's job when it meets one is to report it, not to fall over.
//!
//! Two mechanisms make that structural rather than advisory:
//!
//! * [`HipError`] is `#[non_exhaustive]`, so no crate outside this one can write a `match` on it
//!   without a wildcard arm; and
//! * it carries [`HipError::Unknown`], so a code this build has never heard of round-trips through
//!   the type with its number intact instead of being flattened into "some error".
//!
//! The variant list, its discriminants, [`HipError::from_raw`] and [`HipError::code`] are all
//! generated from ONE table by the `hip_errors!` macro below. Three hand-maintained copies of
//! eighty numbers is exactly the drift this codebase keeps paying for elsewhere.

use std::ffi::c_int;

macro_rules! hip_errors {
    ($( $(#[$attr:meta])* $name:ident = $code:literal ),* $(,)?) => {
        /// A HIP status code.
        ///
        /// The named variants are the codes in `hip_runtime_api.h` of HIP 7.14.60850 (the prod
        /// image), with their numeric values; the enum is `#[repr(i32)]` so a variant *is* its
        /// code. Several HIP names are aliases of one value (`hipErrorMemoryAllocation` ==
        /// `hipErrorOutOfMemory` == 2, `hipErrorInitializationError` == `hipErrorNotInitialized`
        /// == 3, `hipErrorMapBufferObjectFailed` == `hipErrorMapFailed` == 205,
        /// `hipErrorInvalidResourceHandle` == `hipErrorInvalidHandle` == 400); one name is kept
        /// per value and the alias is noted on the variant.
        ///
        /// [`HipError::Unknown`] carries any code this build does not name. Its own discriminant is
        /// `i32::MIN`, outside the range HIP uses, so it can never collide with a real code.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(i32)]
        #[non_exhaustive]
        pub enum HipError {
            $( $(#[$attr])* $name = $code, )*

            /// A code this build does not name, carried verbatim.
            ///
            /// Reaching this is not a bug in the caller and must not be treated as one: a newer
            /// ROCm userspace is allowed to return codes this binary predates.
            /// [`HipError::from_raw`] logs at `warn` when it constructs one, so the number appears
            /// in the log rather than being guessed at from a symptom.
            Unknown(i32) = i32::MIN,
        }

        impl HipError {
            /// Convert a raw `hipError_t`.
            ///
            /// An unrecognised code logs at `warn` and becomes [`HipError::Unknown`]. It is never a
            /// panic and never a silent mapping onto a neighbouring code.
            #[must_use]
            pub fn from_raw(code: c_int) -> HipError {
                match code {
                    $( $code => HipError::$name, )*
                    other => {
                        tracing::warn!(
                            code = other,
                            "HIP returned a status code this build does not name; carrying it \
                             verbatim. Expected against a newer ROCm userspace; not a crash \
                             condition."
                        );
                        HipError::Unknown(other)
                    }
                }
            }

            /// The raw `hipError_t` value, including for [`HipError::Unknown`].
            #[must_use]
            pub fn code(self) -> c_int {
                match self {
                    $( HipError::$name => $code, )*
                    HipError::Unknown(c) => c,
                }
            }
        }
    };
}

hip_errors! {
    /// `hipSuccess` -- not an error. Present so that every raw code converts.
    Success = 0,
    /// `hipErrorInvalidValue`
    InvalidValue = 1,
    /// `hipErrorOutOfMemory` (alias `hipErrorMemoryAllocation`)
    OutOfMemory = 2,
    /// `hipErrorNotInitialized` (alias `hipErrorInitializationError`)
    NotInitialized = 3,
    /// `hipErrorDeinitialized`
    Deinitialized = 4,
    /// `hipErrorProfilerDisabled`
    ProfilerDisabled = 5,
    /// `hipErrorProfilerNotInitialized`
    ProfilerNotInitialized = 6,
    /// `hipErrorProfilerAlreadyStarted`
    ProfilerAlreadyStarted = 7,
    /// `hipErrorProfilerAlreadyStopped`
    ProfilerAlreadyStopped = 8,
    /// `hipErrorInvalidConfiguration`
    InvalidConfiguration = 9,
    /// `hipErrorInvalidPitchValue`
    InvalidPitchValue = 12,
    /// `hipErrorInvalidSymbol`
    InvalidSymbol = 13,
    /// `hipErrorInvalidDevicePointer`
    InvalidDevicePointer = 17,
    /// `hipErrorInvalidMemcpyDirection`
    InvalidMemcpyDirection = 21,
    /// `hipErrorInsufficientDriver`
    InsufficientDriver = 35,
    /// `hipErrorMissingConfiguration`
    MissingConfiguration = 52,
    /// `hipErrorPriorLaunchFailure`
    PriorLaunchFailure = 53,
    /// `hipErrorInvalidDeviceFunction`
    InvalidDeviceFunction = 98,
    /// `hipErrorNoDevice`
    NoDevice = 100,
    /// `hipErrorInvalidDevice`
    InvalidDevice = 101,
    /// `hipErrorInvalidImage` -- a code object the runtime will not accept.
    InvalidImage = 200,
    /// `hipErrorInvalidContext`
    InvalidContext = 201,
    /// `hipErrorContextAlreadyCurrent`
    ContextAlreadyCurrent = 202,
    /// `hipErrorMapFailed` (alias `hipErrorMapBufferObjectFailed`)
    MapFailed = 205,
    /// `hipErrorUnmapFailed`
    UnmapFailed = 206,
    /// `hipErrorArrayIsMapped`
    ArrayIsMapped = 207,
    /// `hipErrorAlreadyMapped`
    AlreadyMapped = 208,
    /// `hipErrorNoBinaryForGpu` -- the code object holds nothing for this target.
    NoBinaryForGpu = 209,
    /// `hipErrorAlreadyAcquired`
    AlreadyAcquired = 210,
    /// `hipErrorNotMapped`
    NotMapped = 211,
    /// `hipErrorNotMappedAsArray`
    NotMappedAsArray = 212,
    /// `hipErrorNotMappedAsPointer`
    NotMappedAsPointer = 213,
    /// `hipErrorECCNotCorrectable`
    EccNotCorrectable = 214,
    /// `hipErrorUnsupportedLimit`
    UnsupportedLimit = 215,
    /// `hipErrorContextAlreadyInUse`
    ContextAlreadyInUse = 216,
    /// `hipErrorPeerAccessUnsupported`
    PeerAccessUnsupported = 217,
    /// `hipErrorInvalidKernelFile`
    InvalidKernelFile = 218,
    /// `hipErrorInvalidGraphicsContext`
    InvalidGraphicsContext = 219,
    /// `hipErrorInvalidSource`
    InvalidSource = 300,
    /// `hipErrorFileNotFound`
    FileNotFound = 301,
    /// `hipErrorSharedObjectSymbolNotFound`
    SharedObjectSymbolNotFound = 302,
    /// `hipErrorSharedObjectInitFailed`
    SharedObjectInitFailed = 303,
    /// `hipErrorOperatingSystem`
    OperatingSystem = 304,
    /// `hipErrorInvalidHandle` (alias `hipErrorInvalidResourceHandle`)
    InvalidHandle = 400,
    /// `hipErrorIllegalState`
    IllegalState = 401,
    /// `hipErrorNotFound` -- e.g. `hipModuleGetFunction` on a symbol the module does not export.
    NotFound = 500,
    /// `hipErrorNotReady` -- not an error for a query: an event or stream is still outstanding.
    NotReady = 600,
    /// `hipErrorIllegalAddress` -- the memory fault. The GPU is wedged; the process is finished.
    IllegalAddress = 700,
    /// `hipErrorLaunchOutOfResources` -- too many registers, too much LDS, or too large a block.
    LaunchOutOfResources = 701,
    /// `hipErrorLaunchTimeOut`
    LaunchTimeOut = 702,
    /// `hipErrorPeerAccessAlreadyEnabled`
    PeerAccessAlreadyEnabled = 704,
    /// `hipErrorPeerAccessNotEnabled`
    PeerAccessNotEnabled = 705,
    /// `hipErrorSetOnActiveProcess`
    SetOnActiveProcess = 708,
    /// `hipErrorContextIsDestroyed`
    ContextIsDestroyed = 709,
    /// `hipErrorAssert` -- a device-side assert fired.
    Assert = 710,
    /// `hipErrorHostMemoryAlreadyRegistered`
    HostMemoryAlreadyRegistered = 712,
    /// `hipErrorHostMemoryNotRegistered`
    HostMemoryNotRegistered = 713,
    /// `hipErrorLaunchFailure`
    LaunchFailure = 719,
    /// `hipErrorCooperativeLaunchTooLarge`
    CooperativeLaunchTooLarge = 720,
    /// `hipErrorNotSupported`
    NotSupported = 801,
    /// `hipErrorStreamCaptureUnsupported` -- an op that is not capture-legal ran under capture.
    StreamCaptureUnsupported = 900,
    /// `hipErrorStreamCaptureInvalidated`
    StreamCaptureInvalidated = 901,
    /// `hipErrorStreamCaptureMerge`
    StreamCaptureMerge = 902,
    /// `hipErrorStreamCaptureUnmatched`
    StreamCaptureUnmatched = 903,
    /// `hipErrorStreamCaptureUnjoined`
    StreamCaptureUnjoined = 904,
    /// `hipErrorStreamCaptureIsolation`
    StreamCaptureIsolation = 905,
    /// `hipErrorStreamCaptureImplicit` -- the failure mode that `hipStreamCaptureModeThreadLocal`
    /// plus a single owning thread exist to make impossible.
    StreamCaptureImplicit = 906,
    /// `hipErrorCapturedEvent`
    CapturedEvent = 907,
    /// `hipErrorStreamCaptureWrongThread`
    StreamCaptureWrongThread = 908,
    /// `hipErrorGraphExecUpdateFailure`
    GraphExecUpdateFailure = 910,
    /// `hipErrorInvalidChannelDescriptor`
    InvalidChannelDescriptor = 911,
    /// `hipErrorInvalidTexture`
    InvalidTexture = 912,
    /// `hipErrorInvalidResourceType`
    InvalidResourceType = 914,
    /// `hipErrorInvalidResourceConfiguration`
    InvalidResourceConfiguration = 915,
    /// `hipErrorStreamDetached`
    StreamDetached = 916,
    /// `hipErrorUnknown` -- HIP's own catch-all, which is a *named* code and therefore distinct
    /// from [`HipError::Unknown`].
    HipUnknown = 999,
    /// `hipErrorRuntimeMemory`
    RuntimeMemory = 1052,
    /// `hipErrorRuntimeOther`
    RuntimeOther = 1053,
    /// `hipErrorInvalidClusterSize`
    InvalidClusterSize = 1054,
}

impl HipError {
    /// `true` for [`HipError::Success`] only.
    #[must_use]
    pub fn is_success(self) -> bool {
        self == HipError::Success
    }

    /// `true` if this build does not name the code.
    #[must_use]
    pub fn is_unknown_to_this_build(self) -> bool {
        matches!(self, HipError::Unknown(_))
    }
}

impl std::fmt::Display for HipError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HipError::Unknown(c) => write!(f, "unnamed HIP status {c}"),
            other => write!(f, "{other:?} ({})", other.code()),
        }
    }
}

/// Everything that can go wrong in this crate.
///
/// [`Error::NotAvailable`] is the one a host build returns on a machine with no ROCm: loading is
/// lazy, so constructing a [`crate::HipThread`] is the first thing that can fail, and it fails with
/// this rather than at link time. The unit tests in this crate depend on that -- they run on a host
/// with no `libamdhip64.so.7` anywhere.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The HIP shared object could not be loaded. On this host that is the normal outcome: there
    /// is no ROCm outside the containers.
    #[error("HIP runtime '{soname}' is not available: {reason}")]
    NotAvailable {
        /// The soname that was attempted.
        soname: String,
        /// The dynamic loader's message.
        reason: String,
    },

    /// The library loaded but does not export a symbol this crate needs -- a ROCm too old or too
    /// exotic. A refusal, not a fallback: a partially bound runtime is how a route silently
    /// degrades.
    #[error("HIP runtime '{soname}' does not export '{symbol}': {reason}")]
    MissingSymbol {
        /// The soname that was loaded.
        soname: String,
        /// The symbol that was missing.
        symbol: &'static str,
        /// The dynamic loader's message.
        reason: String,
    },

    /// A HIP call returned a non-success status.
    #[error("{op} failed: {status}{}", .message.as_deref().map(|m| format!(" -- {m}")).unwrap_or_default())]
    Hip {
        /// The HIP entry point that failed, e.g. `"hipMalloc"`.
        op: &'static str,
        /// The status it returned.
        status: HipError,
        /// `hipGetErrorString`'s text, when it could be read.
        message: Option<String>,
    },

    /// The [`crate::HipThread`] is gone: it panicked, or it was dropped while a handle was still
    /// alive. Every HIP call runs on that thread, so there is nowhere to send the work.
    #[error("the HipThread is gone; no further HIP calls are possible")]
    ThreadGone,

    /// Caller error, caught before any HIP call was made.
    #[error("{0}")]
    Invalid(String),

    /// A bounded host wait ([`crate::HipThread::sync_event_within`]) passed its deadline: the work
    /// never completed (a peer that never pushed, a kernel spinning on a flag that is not coming).
    /// The HIP thread is still inside the wait and is LEFT there -- the handle is wedged and every
    /// later call reports [`Error::ThreadGone`]-like silence; the caller must give the card back.
    #[error("{what}: the device did not finish within {seconds} s")]
    Timeout {
        /// What was being waited for.
        what: String,
        /// The deadline, in seconds.
        seconds: u64,
    },
}

impl Error {
    /// Build an [`Error::Hip`].
    pub(crate) fn hip(op: &'static str, status: HipError, message: Option<String>) -> Error {
        Error::Hip {
            op,
            status,
            message,
        }
    }

    /// The HIP status, if this error came from a HIP call.
    #[must_use]
    pub fn hip_status(&self) -> Option<HipError> {
        match self {
            Error::Hip { status, .. } => Some(*status),
            _ => None,
        }
    }

    /// Whether this is the "no ROCm here" case, which the host test suite treats as expected.
    #[must_use]
    pub fn is_not_available(&self) -> bool {
        matches!(self, Error::NotAvailable { .. })
    }
}

/// This crate's result alias.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_codes_round_trip() {
        for code in [0, 1, 2, 98, 400, 600, 700, 906, 999, 1054] {
            let e = HipError::from_raw(code);
            assert!(!e.is_unknown_to_this_build(), "code {code} should be named");
            assert_eq!(e.code(), code, "code {code} did not round-trip");
        }
    }

    #[test]
    fn unknown_codes_survive_with_their_number() {
        // The whole point: a 7.15-alpha code this build predates must not panic and must not be
        // flattened. 31337 is not a HIP status in any released header.
        let e = HipError::from_raw(31337);
        assert_eq!(e, HipError::Unknown(31337));
        assert!(e.is_unknown_to_this_build());
        assert_eq!(e.code(), 31337);
        assert!(e.to_string().contains("31337"));
    }

    #[test]
    fn negative_and_absurd_codes_are_carried_too() {
        for code in [-1, i32::MIN + 1, i32::MAX] {
            let e = HipError::from_raw(code);
            assert_eq!(e.code(), code);
            assert!(e.is_unknown_to_this_build());
        }
    }

    #[test]
    fn hip_own_catch_all_is_not_our_catch_all() {
        // hipErrorUnknown (999) is a named code; it must not be confused with a code we do not
        // recognise, or "ROCm said Unknown" and "we have never heard of this" become one bug.
        let named = HipError::from_raw(999);
        assert_eq!(named, HipError::HipUnknown);
        assert!(!named.is_unknown_to_this_build());
    }

    #[test]
    fn success_is_not_an_error_value() {
        assert!(HipError::from_raw(0).is_success());
        assert!(!HipError::from_raw(1).is_success());
    }

    #[test]
    fn display_names_the_code() {
        assert!(HipError::OutOfMemory.to_string().contains("OutOfMemory"));
        assert!(HipError::OutOfMemory.to_string().contains('2'));
    }

    #[test]
    fn error_display_includes_the_op_and_message() {
        let e = Error::hip(
            "hipMalloc",
            HipError::OutOfMemory,
            Some("out of memory".into()),
        );
        let s = e.to_string();
        assert!(s.contains("hipMalloc"), "{s}");
        assert!(s.contains("out of memory"), "{s}");
        assert_eq!(e.hip_status(), Some(HipError::OutOfMemory));
        assert!(!e.is_not_available());
    }
}
