/* saber_abi.h -- THE Saber kernel-plugin ABI. Version 0.
 *
 * This header is the single source of truth for two schemas:
 *
 *   1. the KERNEL PLUGIN MANIFEST -- what a plugin publishes about its ops, its implementations,
 *      the geometries each one serves, the workspaces it needs, and the exact kernarg layout its
 *      code object expects; and
 *   2. the TIER PACK -- the device-final weight container the loader streams, and the file header
 *      that keys it to a resolved ExecutionPlan.
 *
 * ON THE SECOND SCHEMA, READ THIS FIRST. `docs/abi/PACK.md` is the source of truth for the BYTES
 * of a pack; this header is the source of truth for the C NAMES. If the two disagree, the bytes
 * win and this header is wrong. The pack structs below are therefore not a design -- they are a
 * transcription, field for field and offset for offset, and every offset is asserted at compile
 * time so a transcription error cannot survive a build.
 *
 * `crates/saber-abi` is GENERATED from this file by bindgen (`crates/saber-abi/regen.sh`); the
 * generated Rust is committed and CI diffs a fresh regeneration against it. Never hand-edit the
 * generated file; edit this header, bump the version below, add a CHANGELOG entry in
 * `docs/abi/CHANGELOG.md`, and regenerate.
 *
 * RULES THIS HEADER OBEYS, AND WHY
 *
 * - Plain C11 (C99 plus `_Static_assert`; the assertions degrade to nothing on an older compiler
 *   and everything else here is C99). No HIP types, no C++, no function declarations. A plugin
 *   artifact is `.hsaco` code
 *   objects plus a manifest; there is no host code in it (ARCHITECTURE.md, "Kernel plane"), so
 *   there is nothing to dlopen and nothing to declare. The structs below are the in-memory form of
 *   manifest data the host has already parsed or generated. This is also what deletes the
 *   runtime-identity, `getenv` and `noexcept` hazard classes structurally rather than by review.
 *
 * - Fixed-width integers only, one explicit `uint32_t` per enumerated field. Enumerations are
 *   declared as C enums for documentation and for bindgen constants; STRUCT FIELDS ARE NEVER OF
 *   ENUM TYPE, because the width of an enum is implementation-defined and an unknown value read
 *   from a manifest must be representable rather than UB (the same reason HIP error codes are
 *   `#[repr(i32)]` with a catch-all: this box runs a 7.15-alpha userspace).
 *
 * - Strings are `const char *`, NUL-terminated, UTF-8, and owned by whoever built the manifest.
 *   Arrays are (pointer, count) pairs, never NULL-terminated lists.
 *
 * - Sizes and offsets that describe device memory are `uint64_t`. Sizes that describe a kernarg
 *   segment are `uint32_t` (the AMDGPU kernarg segment is bounded well under 4 GiB).
 *
 * - Nothing here is a control-flow ladder. A tile rule, a split-K rule, a per-M optimum: those are
 *   `variant_id` plus autotune data, never a field the plugin branches on (ARCHITECTURE.md section
 *   1: "the plan only kills the env-var bug class if plugins cannot re-derive behaviour").
 *
 * - Comments on declarations are doxygen-style so that bindgen carries them into the generated
 *   Rust docs. Every field is documented; that is a review rule, not a style preference.
 */

#ifndef SABER_ABI_H
#define SABER_ABI_H

#include <stddef.h>
#include <stdint.h>

/** Compile-time assertion, used below to prove the on-disk pack structs are laid out exactly as
 * `docs/abi/PACK.md` specifies. Expands to nothing before C11, where the guarantee is carried by
 * the generated bindings' own layout tests instead. */
#if defined(__cplusplus)
#define SABER_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define SABER_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#else
#define SABER_STATIC_ASSERT(cond, msg)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------------------------- */
/* Versioning                                                                                      */
/* --------------------------------------------------------------------------------------------- */

/** Major ABI version.
 *
 * MAJOR changes when an existing field changes meaning, type, or position, or when a field is
 * removed: a manifest built against an older MAJOR is REFUSED, never reinterpreted. */
#define SABER_ABI_VERSION_MAJOR 0u

/** Minor ABI version.
 *
 * MINOR changes when fields or enumerators are appended: a host at (M, m) accepts a manifest at
 * (M, m') for any m' <= m, and refuses m' > m rather than reading fields it does not know. There
 * is no PATCH: a documentation-only edit changes no number. */
#define SABER_ABI_VERSION_MINOR 4u

/** Packed version carried in every manifest root: (MAJOR << 16) | MINOR. */
#define SABER_ABI_VERSION ((SABER_ABI_VERSION_MAJOR << 16) | SABER_ABI_VERSION_MINOR)

#define SABER_ABI_VERSION_GET_MAJOR(v) (((uint32_t)(v)) >> 16)
#define SABER_ABI_VERSION_GET_MINOR(v) (((uint32_t)(v)) & 0xFFFFu)

/* --------------------------------------------------------------------------------------------- */
/* Hashes                                                                                          */
/* --------------------------------------------------------------------------------------------- */

/** Length in bytes of every hash in this ABI. */
#define SABER_HASH_BYTES 32u

/** A raw SHA-256 digest.
 *
 * Every hash in this ABI is a raw SHA-256 digest, 32 bytes, in the order the digest function
 * produces them (i.e. exactly what `sha256sum` prints, hex-decoded). Text forms (`saber explain`,
 * the autotune cache key) are lowercase hex of these bytes. It is a named struct rather than a
 * bare array so that the generated Rust gets one type with one Debug/Display impl instead of N
 * anonymous arrays. */
struct saber_hash256 {
    uint8_t bytes[SABER_HASH_BYTES]; /**< the raw digest */
};

/* --------------------------------------------------------------------------------------------- */
/* Constraints -- the kernel registry's selection predicates                                       */
/* --------------------------------------------------------------------------------------------- */

/** Predicate operators for ::saber_constraint.
 *
 * Taken from the reference's `SABERConstraint` (`ref/kernels/ref.h`), including its discipline, which is
 * the part that matters:
 *
 *   A constraint is one predicate on one named parameter. It is a NECESSARY condition, not a
 *   sufficient one: it settles whether an implementation EXISTS for a geometry, never whether a
 *   particular call is safe. Strides, contiguity, alignment and buffer sizes are per-call and stay
 *   with the launcher, which still rejects anything it cannot run.
 *
 * Constraints discriminate OPS, never TILES. Tile choice -- block shape, split-K, fragment order,
 * A-tiled vs LDS-staged -- is an opaque `variant_id` plus an autotune record. That single split is
 * what stops the macro explosion (`ref_paro.hip:324-333` is ks x tm x wperm x nt x bits
 * today) and what keeps the planner's tile fields from being decorative. */
enum saber_constraint_op {
    SABER_C_EQ  = 0, /**< param == ival */
    SABER_C_LE  = 1, /**< param <= ival */
    SABER_C_GE  = 2, /**< param >= ival */
    SABER_C_DIV = 3, /**< param % ival == 0; ival > 0, and ival == 0 is a malformed manifest */
    SABER_C_IN  = 4  /**< param, as a word, is one of the space-separated words in sval */
};

/** One necessary condition on one named op parameter.
 *
 * The operand used depends on the operator: `ival` for EQ/LE/GE/DIV, `sval` for IN. The unused
 * operand is ignored -- do not assume it is zero or NULL. */
struct saber_constraint {
    const char *key;  /**< parameter name the planner binds, e.g. "head_dim", "K", "dtype_a" */
    const char *sval; /**< string-set operand for ::SABER_C_IN, space separated; NULL otherwise */
    int64_t     ival; /**< numeric operand for EQ/LE/GE/DIV; unspecified for IN */
    uint32_t    op;   /**< one of ::saber_constraint_op */
    uint32_t    _pad; /**< explicit padding, so the layout is identical everywhere */
};

/* --------------------------------------------------------------------------------------------- */
/* Workspaces                                                                                      */
/* --------------------------------------------------------------------------------------------- */

/** What a workspace region is for, and therefore how the host may treat it.
 *
 * "N bytes" cannot express what prod already needs: split-K partials sized by the shape, a block
 * counter that must survive between launches and be zeroed exactly once, and an all-reduce scratch
 * the peer rank must be able to map. So a requirement carries a KIND, a SIZE LAW and a SLOT COUNT,
 * and the host -- not the plugin -- owns the allocation. This is what retires the
 * `set_decode_scratch` global (ARCHITECTURE.md, "Kernel plane"). */
enum saber_ws_kind {
    /** Aliasable between ops and between steps. Contents are undefined at entry; the kernel must
     * write before it reads. The host may hand the same bytes to two ops that never overlap. */
    SABER_WS_SCRATCH = 0,
    /** Survives graph replay and is NOT aliasable. Zeroed according to ::saber_ws_flags. This is
     * the split-K block counter and anything else carrying state between launches of a bucket. */
    SABER_WS_PERSISTENT = 1,
    /** Persistent AND exported to the peer rank via an IPC memory handle: the 2-rank one-shot
     * all-reduce pushes into the peer's copy. Allocation is startup-only and the host owns the
     * handle exchange; `slots` expresses the parity alternation the fused-AR family needs. */
    SABER_WS_IPC_PEER_VISIBLE = 2
};

/** Bits for ::saber_ws_req::flags. */
enum saber_ws_flags {
    /** Zero once, at allocation, and never again. A persistent counter the kernel leaves at a
     * known value wants this. */
    SABER_WS_F_ZERO_ONCE = 1u << 0,
    /** Zero before every replay of the bucket that owns it. Costs a memset node in the graph; the
     * host budgets it in the launch census, which is why it is declared here rather than done
     * inside the kernel. */
    SABER_WS_F_ZERO_PER_REPLAY = 1u << 1,
    /** The allocation must be fine-grained (host-coherent) device memory. Required for the
     * all-reduce flag and sequence-counter regions. */
    SABER_WS_F_FINE_GRAINED = 1u << 2
};

/** One workspace region an implementation requires, with the law that sizes it.
 *
 * The resolved requirement is
 *
 * ```text
 * bytes + bytes_per_m * M + bytes_per_n * N + bytes_per_k * K + bytes_per_mn * M * N
 * ```
 *
 * evaluated PER SLOT, with M, N, K the bucket's resolved problem shape. It is affine on purpose:
 * it covers every workspace in the current stack (a fixed counter: the constant term only; split-K
 * partials: `bytes_per_mn = 4 * splits`; an all-reduce staging buffer: `bytes_per_m` = message
 * bytes per row) without becoming an expression language a plugin can hide behaviour in. A
 * workspace that cannot be written this way is a planner-owned buffer, not a plugin workspace. */
struct saber_ws_req {
    /** Stable identifier, unique within the implementation, used by `saber explain` and by the
     * autotune and launch-census records. */
    const char *name;
    /** Constant term of the size law, in bytes, per slot. */
    uint64_t    bytes;
    /** Coefficient of M in the size law. */
    uint64_t    bytes_per_m;
    /** Coefficient of N in the size law. */
    uint64_t    bytes_per_n;
    /** Coefficient of K in the size law. */
    uint64_t    bytes_per_k;
    /** Coefficient of M*N in the size law. */
    uint64_t    bytes_per_mn;
    /** One of ::saber_ws_kind. */
    uint32_t    kind;
    /** Required byte alignment of each slot's base; a power of two, at least 16. */
    uint32_t    align;
    /** Number of independent, identically sized regions to allocate; at least 1. The host passes
     * the slot index a launch is to use. The fused all-reduce family alternates slots per launch
     * so that a rank never writes the region its peer is still reading. 0 is malformed. */
    uint32_t    slots;
    /** Bitwise OR of ::saber_ws_flags. */
    uint32_t    flags;
};

/* --------------------------------------------------------------------------------------------- */
/* Kernarg ABI descriptor                                                                          */
/* --------------------------------------------------------------------------------------------- */

/** Scalar type of one explicit kernel argument.
 *
 * A kernarg layout the host and the code object disagree about is memory corruption, not an error
 * -- the code-object equivalent of the partition-mismatched tuple that silently hangs this GPU. So
 * the layout is declared, VERSIONED, and validated at load against the `.hsaco`'s own msgpack
 * kernel metadata (`.args`: `.offset`, `.size`, `.value_kind`), and the loader REFUSES on any
 * mismatch. */
enum saber_arg_kind {
    /** Global address-space pointer; 8 bytes, 8-aligned. Every pointer argument is a
     * device-visible address produced by saber-mem; a host pointer is not expressible here. */
    SABER_ARG_PTR = 0,
    SABER_ARG_I32 = 1, /**< int32_t */
    SABER_ARG_U32 = 2, /**< uint32_t */
    SABER_ARG_I64 = 3, /**< int64_t */
    SABER_ARG_U64 = 4, /**< uint64_t */
    SABER_ARG_F32 = 5, /**< IEEE binary32 */
    SABER_ARG_F64 = 6, /**< IEEE binary64 */
    /** A struct passed BY VALUE, whose size and alignment are DECLARED rather than implied by the
     * kind.
     *
     * This is not a convenience. The reference's two paged-attention kernels take `const SABERArgs a` by
     * value, so the struct's own layout IS most of the kernarg segment: the code object reports
     * ONE `by_value` argument of `sizeof(SABERArgs)` = 136 bytes, while the host has to write nine
     * pointers and eleven scalars into it. Without this kind the loader would have to either
     * refuse the kernel or stop checking the segment at all.
     *
     * For this kind, and only for this kind, ::saber_kernarg::size and ::saber_kernarg::align do
     * not have to match a natural scalar size: `size` is `sizeof(T)` and must be a non-zero
     * multiple of 4 and a multiple of `align`; `align` is `alignof(T)`, a power of two, at most
     * ::SABER_ARG_STRUCT_MAX_ALIGN. `offset` must still be a multiple of `align`.
     *
     * The FIELDS inside the struct are not described here. They are a host-side signature, and a
     * host that writes them lays them over `[offset, offset + size)` with the same natural-
     * alignment packing the AMDGPU C ABI uses. The code object cannot describe them: its metadata
     * has no `.align` per argument and no nested layout, which is exactly why this enumerator
     * carries the alignment explicitly. */
    SABER_ARG_STRUCT = 7
};

/** The largest alignment a ::SABER_ARG_STRUCT argument may declare.
 *
 * The AMDGPU kernarg segment is 16-aligned at its base, so an argument cannot usefully ask for
 * more; a manifest that declares more is describing a layout the runtime cannot produce. */
#define SABER_ARG_STRUCT_MAX_ALIGN 16u

/** One explicit kernel argument in the kernarg segment. */
struct saber_kernarg {
    /** The kernel parameter's name, for `explain` and for the loader's mismatch message. Not used
     * for matching -- offsets and kinds are. */
    const char *name;
    /** One of ::saber_arg_kind. */
    uint32_t    kind;
    /** Byte offset from the start of the EXPLICIT kernarg segment. */
    uint32_t    offset;
    /** Byte size. Equals the natural size of `kind`, except for ::SABER_ARG_STRUCT, where it is
     * the declared `sizeof(T)`. */
    uint32_t    size;
    /** Byte alignment. Equals the natural alignment of `kind`, except for ::SABER_ARG_STRUCT,
     * where it is the declared `alignof(T)`. */
    uint32_t    align;
};

/** The versioned kernarg ABI of one kernel symbol. */
struct saber_kernarg_desc {
    /** Version of the kernarg DESCRIPTOR schema, not of the kernel. Packed like
     * ::SABER_ABI_VERSION. It is separate from the plugin's ABI version so that a kernarg-layout
     * change is visible to a host that otherwise understands the manifest. */
    uint32_t abi_version;
    /** Bytes of the explicit argument segment: the arguments listed in `args`, including trailing
     * padding to `segment_align`. This is what the host packs. */
    uint32_t explicit_bytes;
    /** Bytes of the FULL kernarg segment the code object declares (`.kernarg_segment_size`),
     * including the hidden arguments the language runtime appends (block counts, grid sizes,
     * multigrid sync, printf and heap pointers). Hidden arguments are NOT listed in `args`: HIP's
     * `hipModuleLaunchKernel` fills them. The loader checks `explicit_bytes <= segment_bytes`. */
    uint32_t segment_bytes;
    /** Alignment of the segment base (`.kernarg_segment_align`); a power of two. */
    uint32_t segment_align;
    /** Number of entries in `args`. */
    uint32_t n_args;
    /** The explicit arguments, IN THE ORDER THE KERNEL DECLARES THEM. Offsets ascend and must not
     * overlap; gaps are padding and are zero-filled by the host. */
    const struct saber_kernarg *args;
    /** SHA-256 of the code object's kernel metadata note for this symbol, as validated when the
     * manifest was built. A mismatch at load is refused (invariant I3, fail loud). */
    struct saber_hash256 metadata_hash;
};

/* --------------------------------------------------------------------------------------------- */
/* Code objects                                                                                    */
/* --------------------------------------------------------------------------------------------- */

/** One `.hsaco` in the plugin.
 *
 * Wave size is a TRANSLATION-UNIT property, not a device property: `-mwavefrontsize64` is TU-wide
 * and the per-function attribute is silently ignored, so a plugin that mixes wave sizes is two
 * code objects. That is why wave size lives here as well as on ::saber_impl -- the loader
 * validates that an implementation's `wave_size` equals its code object's, and refuses on
 * disagreement rather than trusting either copy. */
struct saber_code_object {
    /** Path of the `.hsaco`, relative to the plugin directory. */
    const char *file;
    /** SHA-256 of the `.hsaco` bytes, checked before `hipModuleLoadData`. */
    struct saber_hash256 hash;
    /** The offload target it was compiled for, e.g. "gfx1201". */
    const char *target;
    /** 32 or 64: the translation unit's wavefront size. */
    uint32_t    wave_size;
    /** Explicit padding. */
    uint32_t    _pad;
};

/* --------------------------------------------------------------------------------------------- */
/* Implementations                                                                                 */
/* --------------------------------------------------------------------------------------------- */

/** Bits for ::saber_impl::flags. */
enum saber_impl_flags {
    /** The implementation is legal inside a stream capture: it allocates nothing, synchronises
     * nothing, and reads no host-visible state that changes between capture and replay. The
     * planner REFUSES to bind an implementation without this bit into a captured bucket, and
     * `saber explain` prints what forced a bucket eager. */
    SABER_IMPL_GRAPH_SAFE = 1u << 0,
    /** The implementation may synchronise: a device-wide barrier, a host sync, a spin on a host
     * flag. Carried separately from ::SABER_IMPL_GRAPH_SAFE because "does not synchronise" and "is
     * capture-legal" are different claims; a manifest asserting both bits is malformed. */
    SABER_IMPL_MAY_SYNC = 1u << 1,
    /** Output is bit-reproducible for identical inputs and identical launch geometry. Split-K with
     * an atomic float reduction is NOT, and must leave this clear -- that is what lets the
     * bit-exact gate know which routes it may assert byte equality on. */
    SABER_IMPL_DETERMINISTIC = 1u << 2,
    /** Reads or writes peer-rank memory; requires the IPC workspaces to be mapped and peer access
     * enabled before the first launch. */
    SABER_IMPL_PEER_ACCESS = 1u << 3
};

/** One implementation of one op: a kernel symbol plus everything the host must know to bind,
 * budget, allocate for and launch it. */
struct saber_impl {
    /** Kernel symbol in the code object, exactly as `hipModuleGetFunction` expects it. */
    const char *symbol;
    /** Human-readable name for `explain` and for autotune records; unique within the plugin. */
    const char *name;

    /** Index into `saber_plugin.ops` of the op this implements. Manifest-local; see
     * ::saber_op::id. */
    uint32_t op;
    /** Index into `saber_plugin.code_objects` of the code object holding `symbol`. */
    uint32_t code_object;

    /** OPAQUE tile/parameter identity. The planner carries it, the autotune cache keys on it, and
     * `explain` prints it; NOTHING interprets it. Two implementations of the same op differing
     * only in block shape, split-K factor, fragment order (WPERM) or staging strategy (A-tiled vs
     * LDS) differ only in this number plus their geometry. It is stable across rebuilds of the
     * same source because it is part of the autotune cache key. */
    uint32_t variant_id;
    /** 32 or 64; must equal the code object's `wave_size`. */
    uint32_t wave_size;

    /** The workgroup shape the kernel requires (`reqd_work_group_size` when the code object
     * declares one, otherwise the shape the launcher must use). Grid size is per-call and is
     * deliberately not here. */
    uint32_t block_dim[3];
    /** Static LDS bytes the kernel declares (`.group_segment_fixed_size`). Dynamic LDS, if any, is
     * added by the host at launch and is part of the variant's parameters, not of this field. */
    uint32_t lds_bytes;
    /** Private (scratch) bytes per work-item (`.private_segment_fixed_size`). Non-zero means the
     * kernel spills; it is reported so a spilling variant is visible in `explain` rather than
     * discovered in a profile. */
    uint32_t private_bytes;
    /** VGPRs the kernel uses, from code-object metadata; 0 if the toolchain did not report it. */
    uint32_t vgpr_count;
    /** SGPRs the kernel uses, from code-object metadata; 0 if the toolchain did not report it. */
    uint32_t sgpr_count;
    /** Maximum work-items per workgroup the kernel supports; 0 if unconstrained. */
    uint32_t max_threads_per_block;
    /** Bitwise OR of ::saber_impl_flags. */
    uint32_t flags;

    /** THE DECLARED LAUNCH COUNT: how many kernel launches ONE call of this implementation costs,
     * including its own reduce or epilogue launches.
     *
     * It is a constant and not a function of shape because launch count per bucket is declared
     * statically and asserted at capture (CLAUDE.md rule 7). An implementation whose launch count
     * varies with shape is TWO implementations with different `variant_id`s, not one with a
     * formula: a split-K variant needing a separate reduce declares 2; the same kernel at split-K 1
     * declares 1. This is what makes launch count budgetable before running rather than detectable
     * after the fact. */
    uint32_t n_launches;

    /** NECESSARY conditions on the op's named parameters; all must hold. An empty list means the
     * implementation serves every geometry of its op, which for a specialised kernel is almost
     * always a manifest bug. */
    const struct saber_constraint *constraints;
    /** Number of entries in `constraints`. */
    uint32_t n_constraints;
    /** Number of entries in `workspaces`. */
    uint32_t n_workspaces;
    /** Workspace requirements, resolved by the host against the bucket's shape. */
    const struct saber_ws_req *workspaces;

    /** Exact kernarg layout for `symbol`. */
    struct saber_kernarg_desc kernarg;
};

/* --------------------------------------------------------------------------------------------- */
/* Ops and the plugin manifest                                                                     */
/* --------------------------------------------------------------------------------------------- */

/** One operation the planner can select an implementation for.
 *
 * An op is the unit the planner selects on: "w4a8 GEMM", "paged decode attention", "exact 2-rank
 * all-reduce". Implementations of one op are interchangeable at the plan level and differ in the
 * geometries they serve and in their variant parameters.
 *
 * There is deliberately NO central numeric op registry in v0. `id` is the op's index within THIS
 * manifest and exists so implementations can reference it in four bytes; `name` is the identity
 * the planner matches on and the string that appears in the plan, in `explain` and in autotune
 * keys. A registry can be added in a MINOR bump once two plugins must agree on more than a
 * string. */
struct saber_op {
    /** Index of this entry in `saber_plugin.ops`; `id` equals the position. */
    uint32_t    id;
    /** Number of entries in `impls`. */
    uint32_t    n_impls;
    /** Selection key, e.g. "gemm.w4a8.nt" -- stable, lowercase, dotted. */
    const char *name;
    /** Coarse grouping for reports: "gemm" | "attn" | "gdn" | "ar" | "ew". */
    const char *family;
    /** What it computes, for a human reading `explain`. */
    const char *computes;
    /** The implementations of this op. */
    const struct saber_impl *impls;
};

/** The root of a plugin manifest. */
struct saber_plugin {
    /** ::SABER_ABI_VERSION this manifest was produced against. A host refuses a different MAJOR,
     * and refuses a MINOR above its own. */
    uint32_t abi_version;
    /** Number of entries in `ops`. */
    uint32_t n_ops;
    /** Number of entries in `code_objects`. */
    uint32_t n_code_objects;
    /** Explicit padding. */
    uint32_t _pad;

    /** Plugin identity that a plan binds to, e.g. "saber-kernels" or "saber". Stable. */
    const char *name;
    /** Human-facing version string. */
    const char *version;

    /** THE BUILD HASH: SHA-256 over the plugin's sources, the compiler identity and the exact
     * flags -- everything that could change generated code.
     *
     * It is part of every autotune cache key (a record measured against a different build is
     * invalid) and it is what `explain` prints, so that "which kernels was this number measured
     * with" has an answer that is not a guess. */
    struct saber_hash256 build_hash;

    /** The offload target every code object in this plugin was built for, e.g. "gfx1201". A plugin
     * spanning targets is several plugins. */
    const char *target;

    /** The code objects this plugin ships. */
    const struct saber_code_object *code_objects;
    /** The ops this plugin implements. */
    const struct saber_op *ops;
};

/* --------------------------------------------------------------------------------------------- */
/* Tier pack -- the `SBRPACK1` on-disk format                                                      */
/* --------------------------------------------------------------------------------------------- */

/* A pack is the on-disk form of a MODEL TIER, written in device-final layout so that loading is one
 * sequential O_DIRECT stream with ZERO transforms on the load path. Every transform the current
 * stack pays at load time -- NVFP4 -> MXFP4 requant, WPERM fragment order, the A-tiled prefill
 * layout, the 48 GDN in_proj merges, TP sharding -- is applied once at pack-build time.
 *
 * `docs/abi/PACK.md` OWNS THE BYTES. This section owns the C names for them and nothing else. The
 * structs, the constants and the offsets below are a transcription of that document's two tables;
 * where they disagree with it, they are the ones that are wrong. Each struct is followed by
 * SABER_STATIC_ASSERT of its size and of every field offset, so a transcription error is a build
 * failure rather than a corrupt read, and the generated Rust carries bindgen's own layout tests
 * for the same facts.
 *
 * Everything on disk is LITTLE-ENDIAN, fixed size, with no varints and no padding holes. Note that
 * there is deliberately no `#pragma pack`: every field in both structs is already naturally
 * aligned and both sizes are multiples of 8, so the natural C layout IS the wire layout. Packing
 * the structs would only hide a future field that is not -- the static assertions catch that
 * instead of silently absorbing it.
 *
 * File layout of one pack file:
 *
 * ```text
 *   0                      header        SABER_PACK_HEADER_BYTES (256), fixed
 *   256                    record table  record_count * SABER_PACK_RECORD_BYTES (160)
 *   256 + 160*n            name blob     name_blob_bytes
 *      ... zero padding ...
 *   data_off (4096-aligned) data section records, each 4096-aligned, zero padding between
 *   file_bytes (4096 mult.) EOF
 * ```
 *
 * A pack is ONE FILE PER RANK: `<packdir>/rank<r>.pack`. Every record in `rank<r>.pack` carries
 * `rank == r`; the rank is in both the header and the record so a table entry stays
 * self-describing when it is copied into a log line or a sidecar manifest.
 */

/** Magic in the first 8 bytes of a pack file: ASCII `SBRPACK1`, no NUL terminator.
 *
 * The trailing `1` is part of the magic, not a version field -- `format_version` is separate and
 * follows it. Comparing 8 bytes is the cheapest possible rejection of a truncated, wrong or
 * byte-swapped file, and it happens at byte 0 rather than at the first tensor. */
#define SABER_PACK_MAGIC "SBRPACK1"

/** Length of ::SABER_PACK_MAGIC on disk, excluding the C string's NUL. */
#define SABER_PACK_MAGIC_BYTES 8u

/** The on-disk format version this header describes, and the newest a reader here writes.
 *
 * v2 added the chunk-digest blob: four header fields in what v1 called `reserved`, one record flag,
 * and a section between the name blob and the data. Nothing v1 defined moved or changed meaning, so
 * a v2 reader reads a v1 pack; a v1 reader refuses a v2 one, which is the correct loud failure. */
#define SABER_PACK_FORMAT_VERSION 2u

/** Oldest on-disk format version a reader here accepts. A v1 pack has no chunk digests. */
#define SABER_PACK_FORMAT_VERSION_MIN 1u

/** Size of the fixed file header. A reader rejects a `header_bytes` that is not this. */
#define SABER_PACK_HEADER_BYTES 256u

/** Size of one record table entry. A v1 reader rejects a `record_bytes` that is not this. */
#define SABER_PACK_RECORD_BYTES 160u

/** Alignment of every record's data offset and of the file length.
 *
 * 4096 so that any record can be read with O_DIRECT and DMA'd without a bounce buffer. This is what
 * makes the measured 12.4 GB/s streaming load reachable at all; the same checkpoint through the reference server's
 * loader on the same drive gets 2.3 GB/s. */
#define SABER_PACK_ALIGN 4096u

/** Maximum tensor rank a pack record can describe; `ndim` is 1..=8.
 *
 * Raising it is a format change, not an ABI convenience: the shape array is inline in the 160-byte
 * record and changing its length moves every field after it. */
#define SABER_PACK_MAX_DIMS 8u

/** Bytes of the little-endian `u16` length prefix that precedes each name in the name blob.
 *
 * A record's `name_off` points at the FIRST UTF-8 BYTE, so the prefix sits at
 * `name_blob_off + name_off - SABER_PACK_NAME_PREFIX_BYTES` and `name_off >= 2` always. A reader
 * must check that the prefix equals `name_len`: the redundancy is deliberate and is the cheapest
 * available check that a table entry and the name blob came from the same build. */
#define SABER_PACK_NAME_PREFIX_BYTES 2u

/** Bytes of zero reserved at the end of the file header. A reader rejects a non-zero byte. */
#define SABER_PACK_HEADER_RESERVED_BYTES 28u

/** Number of reserved `uint64_t` at the end of a record entry. A reader rejects a non-zero one. */
#define SABER_PACK_RECORD_RESERVED_U64 3u

/** Element type of a record's payload, as a wire tag.
 *
 * THESE TAGS ARE ABI AND ARE NEVER RENUMBERED. They are the table in `docs/abi/PACK.md` section
 * "Dtype tags"; a new type is appended, never inserted. The comment on each variant gives its bits
 * per LOGICAL element, which is what `length == ceil(prod(shape) * bits / 8)` is computed from. */
enum saber_pack_dtype {
    SABER_DT_U8 = 0,  /**< 8 bits. Also the dtype of NVFP4 `weight_packed` at layout id 0, which the
                       *   checkpoint stores as `u8` with its packed shape `[N, K/2]`. */
    SABER_DT_I8 = 1,  /**< 8 bits, signed. */
    SABER_DT_U16 = 2, /**< 16 bits. */
    SABER_DT_I16 = 3, /**< 16 bits, signed. */
    SABER_DT_U32 = 4, /**< 32 bits. */
    SABER_DT_I32 = 5, /**< 32 bits, signed. */
    SABER_DT_U64 = 6, /**< 64 bits. */
    SABER_DT_I64 = 7, /**< 64 bits, signed. */
    SABER_DT_F16 = 8, /**< 16 bits, IEEE binary16. */
    SABER_DT_BF16 = 9,   /**< 16 bits, bfloat16. */
    SABER_DT_F32 = 10,   /**< 32 bits, IEEE binary32. */
    SABER_DT_F64 = 11,   /**< 64 bits, IEEE binary64. */
    SABER_DT_F8E4M3 = 12, /**< 8 bits, OCP e4m3 (`F8_E4M3` in safetensors). */
    SABER_DT_F8E5M2 = 13, /**< 8 bits, OCP e5m2. */
    /** 4 bits. Two e2m1 values per byte, LOW NIBBLE = EVEN ELEMENT. `shape` is then the LOGICAL
     * element count and `length = prod(shape) / 2`. Layout id 0 never uses this tag: a raw pack
     * reproduces the checkpoint's own dtype and shape byte for byte. */
    SABER_DT_FP4E2M1X2 = 14,
    SABER_DT_BOOL = 15, /**< 8 bits, one byte per boolean. */
    SABER_DT_E8M0 = 16  /**< 8 bits, MX block-scale exponent; reserved for the MXFP4 layouts. */
};

/** How a tensor was split across TP ranks.
 *
 * Purely informational: the bytes in the record are ALREADY the rank's bytes, and nothing on the
 * load path re-derives a split. It exists so `saber explain` and the sidecar manifest can say how a
 * tensor was divided without inferring it from shapes. */
enum saber_pack_shard_kind {
    SABER_SHARD_REPLICATED = 0, /**< Identical bytes on every rank. */
    SABER_SHARD_DIM0 = 1,       /**< Column-parallel: split along dim 0 (output features, vocab). */
    SABER_SHARD_DIM1 = 2,       /**< Row-parallel: split along dim 1 (the reduction dim K). */
    /** Dim 0 is a concatenation of segments (fused QKV): each segment is split independently and
     * the pieces re-concatenated rank-locally. A contiguous half-split would be WRONG. */
    SABER_SHARD_DIM0_SEGMENTS = 3
};

/** Layout of a record's payload.
 *
 * Unlike the dtype table this range is open, and it is allocated in `docs/abi/PACK.md` section
 * "Layout ids" -- not here. A new device-final layout is an ABI change: it gets an id in that
 * table in the same change that teaches `tools/pack` to emit it and names the kernel variant that
 * consumes it. `layout_id` is per record, not per pack, so a tier mixes raw and device-final
 * records during the port. */
enum saber_pack_layout {
    /** Raw checkpoint bytes: checkpoint dtype, rank-local shape, no transform whatsoever. */
    SABER_LAYOUT_RAW = 0,
    /** First id in the private range. Ids at or above this are never written by `tools/pack` and
     * are free for experiments that must not be loadable by a released engine. Everything below it
     * and above ::SABER_LAYOUT_RAW is reserved for device-final layouts allocated in PACK.md. */
    SABER_LAYOUT_PRIVATE_BASE = 0x10000000u
};

/** The 256-byte pack file header, exactly as it appears on disk.
 *
 * Field order and offsets are `docs/abi/PACK.md` section "Header (256 bytes)". `header_bytes` and
 * `record_bytes` are stored so that a reader built for v1 can DIAGNOSE a future layout instead of
 * silently misparsing it. */
struct saber_pack_file_header {
    /** Offset 0. Exactly ::SABER_PACK_MAGIC, 8 ASCII bytes, no NUL. */
    uint8_t magic[SABER_PACK_MAGIC_BYTES];
    /** Offset 8. ::SABER_PACK_FORMAT_VERSION. */
    uint32_t format_version;
    /** Offset 12. Must equal ::SABER_PACK_HEADER_BYTES. */
    uint32_t header_bytes;
    /** Offset 16. Must equal ::SABER_PACK_RECORD_BYTES. */
    uint32_t record_bytes;
    /** Offset 20. Number of entries in the record table. */
    uint32_t record_count;
    /** Offset 24. Offset of the record table; must equal ::SABER_PACK_HEADER_BYTES. */
    uint64_t table_off;
    /** Offset 32. Must equal `table_off + record_count * record_bytes`. */
    uint64_t name_blob_off;
    /** Offset 40. Length of the name blob in bytes. */
    uint64_t name_blob_bytes;
    /** Offset 48. Start of the data section: a multiple of ::SABER_PACK_ALIGN, and at or after
     * `align_up(name_blob_off + name_blob_bytes)`. */
    uint64_t data_off;
    /** Offset 56. Total file length: a multiple of ::SABER_PACK_ALIGN, and equal to the actual
     * size of the file. */
    uint64_t file_bytes;
    /** Offset 64. Number of ranks in the tier; at least 1. */
    uint32_t tp_size;
    /** Offset 68. The TP rank this file holds; less than `tp_size`. */
    uint32_t rank;
    /** Offset 72. Build time in Unix seconds. INFORMATIONAL ONLY -- never compared, never part of
     * any hash, so two rebuilds of the same inputs differ here and nowhere else. */
    uint64_t build_unix_seconds;
    /** Offset 80. Hash of the resolved ExecutionPlan this pack's layouts were produced for.
     *
     * The loader requires `header.plan_hash == plan.hash()` and on mismatch REFUSES THE PACK. It
     * does not renegotiate, does not fall back to a raw layout, and does not re-derive the missing
     * transform at load time -- that is the entire point of the pack (I3, and the zero-transform
     * property the cold-start budget depends on). Rebuilding is an offline step. */
    struct saber_hash256 plan_hash;
    /** Offset 112. Content hash of the release unit: checkpoint and quantization, the drafter
     * trained against it, merged adapters, the tuned serving config, the chat template, the
     * tokenizer.
     *
     * IDENTICAL ACROSS THE RANKS of one pack set, and deliberately excludes `tp_size`: TP is a plan
     * property, not a property of the release unit, so one tier can have a TP=1 and a TP=2 pack and
     * a drafter binds to both. The planner refuses to bind a drafter whose tier id does not match
     * the target's. */
    struct saber_hash256 tier_id;
    /** Offset 144. `sha256` over the byte range `[table_off, name_blob_off + name_blob_bytes)` --
     * that is, the record table AND the name blob, and nothing else: not this header, not the data.
     *
     * Checked AFTER the structural rules, so that a malformed table reports the structural fault
     * rather than a hash mismatch that says nothing about what is wrong. */
    struct saber_hash256 table_sha256;
    /** Offset 176. Byte offset of the chunk-digest blob, or 0 when the pack has none (v2). */
    uint64_t chunk_blob_off;
    /** Offset 184. Length of the chunk-digest blob, or 0 when the pack has none (v2). */
    uint64_t chunk_blob_bytes;
    /** Offset 192. `sha256` over the whole chunk-digest blob, or all zero when there is none.
     *
     * This is the blob's single authority. A chunk digest is trusted because the blob hashes to
     * this AND because the entry repeats the record's own `sha256`, which `table_sha256` already
     * covers. It is not a fold of the chunk digests, because the record digest is a plain `sha256`
     * of the payload and no fold can reproduce one -- see `docs/abi/PACK.md` § Chunk digests. */
    struct saber_hash256 chunk_blob_sha256;
    /** Offset 224. Payload bytes each chunk digest covers; 0 when there are none. A non-zero value
     * is a multiple of ::SABER_PACK_ALIGN. */
    uint32_t chunk_bytes;
    /** Offset 228. ::SABER_PACK_HEADER_RESERVED_BYTES bytes, all zero. A reader rejects a non-zero
     * byte rather than ignoring it: that is what keeps the range usable later. */
    uint8_t reserved[SABER_PACK_HEADER_RESERVED_BYTES];
};

/** Byte offset of `saber_pack_file_header::magic`. */
#define SABER_PACK_HDR_OFF_MAGIC 0u
/** Byte offset of `saber_pack_file_header::format_version`. */
#define SABER_PACK_HDR_OFF_FORMAT_VERSION 8u
/** Byte offset of `saber_pack_file_header::header_bytes`. */
#define SABER_PACK_HDR_OFF_HEADER_BYTES 12u
/** Byte offset of `saber_pack_file_header::record_bytes`. */
#define SABER_PACK_HDR_OFF_RECORD_BYTES 16u
/** Byte offset of `saber_pack_file_header::record_count`. */
#define SABER_PACK_HDR_OFF_RECORD_COUNT 20u
/** Byte offset of `saber_pack_file_header::table_off`. */
#define SABER_PACK_HDR_OFF_TABLE_OFF 24u
/** Byte offset of `saber_pack_file_header::name_blob_off`. */
#define SABER_PACK_HDR_OFF_NAME_BLOB_OFF 32u
/** Byte offset of `saber_pack_file_header::name_blob_bytes`. */
#define SABER_PACK_HDR_OFF_NAME_BLOB_BYTES 40u
/** Byte offset of `saber_pack_file_header::data_off`. */
#define SABER_PACK_HDR_OFF_DATA_OFF 48u
/** Byte offset of `saber_pack_file_header::file_bytes`. */
#define SABER_PACK_HDR_OFF_FILE_BYTES 56u
/** Byte offset of `saber_pack_file_header::tp_size`. */
#define SABER_PACK_HDR_OFF_TP_SIZE 64u
/** Byte offset of `saber_pack_file_header::rank`. */
#define SABER_PACK_HDR_OFF_RANK 68u
/** Byte offset of `saber_pack_file_header::build_unix_seconds`. */
#define SABER_PACK_HDR_OFF_BUILD_UNIX_SECONDS 72u
/** Byte offset of `saber_pack_file_header::plan_hash`. */
#define SABER_PACK_HDR_OFF_PLAN_HASH 80u
/** Byte offset of `saber_pack_file_header::tier_id`. */
#define SABER_PACK_HDR_OFF_TIER_ID 112u
/** Byte offset of `saber_pack_file_header::table_sha256`. */
#define SABER_PACK_HDR_OFF_TABLE_SHA256 144u
/** Byte offset of `saber_pack_file_header::chunk_blob_off`. */
#define SABER_PACK_HDR_OFF_CHUNK_BLOB_OFF 176u
/** Byte offset of `saber_pack_file_header::chunk_blob_bytes`. */
#define SABER_PACK_HDR_OFF_CHUNK_BLOB_BYTES 184u
/** Byte offset of `saber_pack_file_header::chunk_blob_sha256`. */
#define SABER_PACK_HDR_OFF_CHUNK_BLOB_SHA256 192u
/** Byte offset of `saber_pack_file_header::chunk_bytes`. */
#define SABER_PACK_HDR_OFF_CHUNK_BYTES 224u
/** Byte offset of `saber_pack_file_header::reserved`. */
#define SABER_PACK_HDR_OFF_RESERVED 228u

SABER_STATIC_ASSERT(sizeof(struct saber_pack_file_header) == SABER_PACK_HEADER_BYTES,
                    "pack file header must be 256 bytes (docs/abi/PACK.md)");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_file_header, magic) == SABER_PACK_HDR_OFF_MAGIC, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_file_header, format_version)
                        == SABER_PACK_HDR_OFF_FORMAT_VERSION, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_file_header, header_bytes)
                        == SABER_PACK_HDR_OFF_HEADER_BYTES, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_file_header, record_bytes)
                        == SABER_PACK_HDR_OFF_RECORD_BYTES, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_file_header, record_count)
                        == SABER_PACK_HDR_OFF_RECORD_COUNT, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_file_header, table_off)
                        == SABER_PACK_HDR_OFF_TABLE_OFF, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_file_header, name_blob_off)
                        == SABER_PACK_HDR_OFF_NAME_BLOB_OFF, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_file_header, name_blob_bytes)
                        == SABER_PACK_HDR_OFF_NAME_BLOB_BYTES, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_file_header, data_off)
                        == SABER_PACK_HDR_OFF_DATA_OFF, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_file_header, file_bytes)
                        == SABER_PACK_HDR_OFF_FILE_BYTES, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_file_header, tp_size)
                        == SABER_PACK_HDR_OFF_TP_SIZE, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_file_header, rank) == SABER_PACK_HDR_OFF_RANK, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_file_header, build_unix_seconds)
                        == SABER_PACK_HDR_OFF_BUILD_UNIX_SECONDS, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_file_header, plan_hash)
                        == SABER_PACK_HDR_OFF_PLAN_HASH, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_file_header, tier_id)
                        == SABER_PACK_HDR_OFF_TIER_ID, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_file_header, table_sha256)
                        == SABER_PACK_HDR_OFF_TABLE_SHA256, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_file_header, chunk_blob_off)
                        == SABER_PACK_HDR_OFF_CHUNK_BLOB_OFF, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_file_header, chunk_blob_bytes)
                        == SABER_PACK_HDR_OFF_CHUNK_BLOB_BYTES, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_file_header, chunk_blob_sha256)
                        == SABER_PACK_HDR_OFF_CHUNK_BLOB_SHA256, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_file_header, chunk_bytes)
                        == SABER_PACK_HDR_OFF_CHUNK_BYTES, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_file_header, reserved)
                        == SABER_PACK_HDR_OFF_RESERVED, "");

/** One 160-byte record table entry, exactly as it appears on disk.
 *
 * Field order and offsets are `docs/abi/PACK.md` section "Record table entry (160 bytes)". One
 * entry per (tensor, rank); records publish readiness individually, so early layers warm while
 * later ones are still landing.
 *
 * NAMING, because "rank" is overloaded and the ambiguity is expensive: `rank` is the TENSOR-
 * PARALLEL rank that owns this shard (the on-disk field name, so it is not renamed here), and
 * `ndim` is the number of tensor dimensions. Nothing in this ABI uses "rank" for a tensor's
 * dimensionality. */
struct saber_pack_table_entry {
    /** Offset 0. Offset of the name's first UTF-8 byte, measured FROM `name_blob_off`; at least
     * ::SABER_PACK_NAME_PREFIX_BYTES, because the `u16` length prefix precedes it. */
    uint32_t name_off;
    /** Offset 4. Length of the UTF-8 name in bytes; at least 1, and equal to the `u16` prefix that
     * precedes the name in the blob. Names are unique within a file and are the checkpoint's tensor
     * names verbatim -- a sharded record keeps the SAME name, because the rank is the
     * discriminator, not the name. */
    uint32_t name_len;
    /** Offset 8. TP rank owning this shard; equals the file header's `rank`. */
    uint32_t rank;
    /** Offset 12. One of ::saber_pack_layout, or an id allocated in PACK.md's layout table. */
    uint32_t layout_id;
    /** Offset 16. One of ::saber_pack_dtype. */
    uint16_t dtype;
    /** Offset 18. Number of meaningful entries in `shape`; 1..=::SABER_PACK_MAX_DIMS. */
    uint8_t ndim;
    /** Offset 19. One of ::saber_pack_shard_kind. */
    uint8_t shard_kind;
    /** Offset 20. Zero in format version 1; a reader rejects a non-zero value rather than ignoring
     * bits it does not understand. */
    uint32_t flags;
    /** Offset 24. Absolute file offset of the payload: a multiple of ::SABER_PACK_ALIGN and at or
     * after the header's `data_off`. */
    uint64_t data_off;
    /** Offset 32. Exact payload length, unpadded. A streaming loader transfers `align_up(length)`
     * bytes, which is why records must not overlap after padding. */
    uint64_t length;
    /** Offset 40. Rank-local shape: `shape[i] > 0` for `i < ndim` and `0` for the rest. The
     * payload length it implies, `ceil(prod(shape) * dtype_bits / 8)`, must equal `length`. */
    uint64_t shape[SABER_PACK_MAX_DIMS];
    /** Offset 104. `sha256` over exactly `length` bytes at `data_off`, verified after landing. This
     * is what turns "weights read as garbage" into a load error instead of fluent-looking nonsense
     * (invariant I3). */
    struct saber_hash256 sha256;
    /** Offset 136. ::SABER_PACK_RECORD_RESERVED_U64 reserved words, all zero; a reader rejects a
     * non-zero one. */
    uint64_t reserved[SABER_PACK_RECORD_RESERVED_U64];
};

/** Byte offset of `saber_pack_table_entry::name_off`. */
#define SABER_PACK_REC_OFF_NAME_OFF 0u
/** Byte offset of `saber_pack_table_entry::name_len`. */
#define SABER_PACK_REC_OFF_NAME_LEN 4u
/** Byte offset of `saber_pack_table_entry::rank`. */
#define SABER_PACK_REC_OFF_RANK 8u
/** Byte offset of `saber_pack_table_entry::layout_id`. */
#define SABER_PACK_REC_OFF_LAYOUT_ID 12u
/** Byte offset of `saber_pack_table_entry::dtype`. */
#define SABER_PACK_REC_OFF_DTYPE 16u
/** Byte offset of `saber_pack_table_entry::ndim`. */
#define SABER_PACK_REC_OFF_NDIM 18u
/** Byte offset of `saber_pack_table_entry::shard_kind`. */
#define SABER_PACK_REC_OFF_SHARD_KIND 19u
/** Byte offset of `saber_pack_table_entry::flags`. */
#define SABER_PACK_REC_OFF_FLAGS 20u
/** Byte offset of `saber_pack_table_entry::data_off`. */
#define SABER_PACK_REC_OFF_DATA_OFF 24u
/** Byte offset of `saber_pack_table_entry::length`. */
#define SABER_PACK_REC_OFF_LENGTH 32u
/** Byte offset of `saber_pack_table_entry::shape`. */
#define SABER_PACK_REC_OFF_SHAPE 40u
/** Byte offset of `saber_pack_table_entry::sha256`. */
#define SABER_PACK_REC_OFF_SHA256 104u
/** Byte offset of `saber_pack_table_entry::reserved`. */
#define SABER_PACK_REC_OFF_RESERVED 136u

/** `saber_pack_table_entry::flags` bit 0: this record has an entry in the chunk-digest blob. */
#define SABER_PACK_REC_FLAG_CHUNK_DIGESTS 1u

/** Every `saber_pack_table_entry::flags` bit this version defines. A reader refuses any other. */
#define SABER_PACK_REC_FLAGS_KNOWN SABER_PACK_REC_FLAG_CHUNK_DIGESTS

/** One entry of the chunk-digest blob (v2).
 *
 * Entries appear in table order for exactly the records whose
 * ::SABER_PACK_REC_FLAG_CHUNK_DIGESTS is set, packed with no padding between them, and the blob's
 * length must account for all of them exactly. An entry is variable length, so this struct
 * describes its fixed head only. */
struct saber_pack_chunk_entry {
    /** Offset 0. Digests that follow. Must equal `ceil(record.length / header.chunk_bytes)`. */
    uint32_t n_chunks;
    /** Offset 4. Zero in v2; a reader refuses a non-zero value. */
    uint32_t flags;
    /** Offset 8. The record's own `sha256`, repeated. This is the binding: an entry that names a
     * different digest than the table's is refused, so a digest list cannot be moved between
     * records or substituted wholesale. */
    struct saber_hash256 record_sha256;
    /* Then n_chunks * SABER_HASH_BYTES of chunk digests, in payload order. */
};

/** Bytes of a chunk-blob entry before its digest list. */
#define SABER_PACK_CHUNK_ENTRY_HEADER 40u

SABER_STATIC_ASSERT(sizeof(struct saber_pack_chunk_entry) == SABER_PACK_CHUNK_ENTRY_HEADER,
                    "chunk blob entry head must be 40 bytes (docs/abi/PACK.md)");

SABER_STATIC_ASSERT(sizeof(struct saber_pack_table_entry) == SABER_PACK_RECORD_BYTES,
                    "pack record entry must be 160 bytes (docs/abi/PACK.md)");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_table_entry, name_off)
                        == SABER_PACK_REC_OFF_NAME_OFF, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_table_entry, name_len)
                        == SABER_PACK_REC_OFF_NAME_LEN, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_table_entry, rank) == SABER_PACK_REC_OFF_RANK, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_table_entry, layout_id)
                        == SABER_PACK_REC_OFF_LAYOUT_ID, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_table_entry, dtype) == SABER_PACK_REC_OFF_DTYPE, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_table_entry, ndim) == SABER_PACK_REC_OFF_NDIM, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_table_entry, shard_kind)
                        == SABER_PACK_REC_OFF_SHARD_KIND, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_table_entry, flags) == SABER_PACK_REC_OFF_FLAGS, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_table_entry, data_off)
                        == SABER_PACK_REC_OFF_DATA_OFF, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_table_entry, length)
                        == SABER_PACK_REC_OFF_LENGTH, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_table_entry, shape) == SABER_PACK_REC_OFF_SHAPE, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_table_entry, sha256)
                        == SABER_PACK_REC_OFF_SHA256, "");
SABER_STATIC_ASSERT(offsetof(struct saber_pack_table_entry, reserved)
                        == SABER_PACK_REC_OFF_RESERVED, "");

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SABER_ABI_H */
