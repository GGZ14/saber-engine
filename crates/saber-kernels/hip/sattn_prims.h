// Saber paged attention, CLEANROOM-2 (CR2-2c): register-level primitives.
//
// Written from the CR2-2a specification (saber-cache/cr2-2/spec/attention.md), not from any earlier
// attention source. Every helper below names the one instruction the arithmetic contract (spec section 2)
// pins; nothing here may be swapped for a library routine or a "mathematically equal" form.
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace sattn {

using f16x8 = _Float16 __attribute__((ext_vector_type(8)));
using f16x2 = _Float16 __attribute__((ext_vector_type(2)));
using f32x8 = float __attribute__((ext_vector_type(8)));
using f32x2 = float __attribute__((ext_vector_type(2)));
using i32x2 = int __attribute__((ext_vector_type(2)));
using i16x8 = short __attribute__((ext_vector_type(8)));
using u32x4 = uint32_t __attribute__((ext_vector_type(4)));
using u32x2 = uint32_t __attribute__((ext_vector_type(2)));

// Kernel argument block, by value. Only its layout is ABI (include/saber_abi.h is the record): 9 pointers,
// 7 ints, 4 bytes of padding, 2 longs, a float and 2 ints = 136 bytes; the field names here are ours.
using cbuf = const void*;
using cints = const int*;
using cflts = const float*;
struct Params {
  cbuf query, cache;        // bf16 rows [token][q head][256]; paged KV (fp8 or bf16 elements)
  cints page_map, ctx_len;  // [seq][map_width] page ids; [seq] context length (QSA decode: key count)
  void* dst;                // bf16 rows, same layout as query
  cflts kscale, vscale;     // [seq][kv head] descales, or null (= 1)
  cflts qscale_unused;
  void* work;               // decode partials: f16 O, then f32 m, then f32 l
  int nseq, qrows, nqh, nkvh, hdim, page_slots, map_width;
  long page_stride, head_stride;  // elements between pages / between kv heads in a page
  float sm_scale;
  int split_hint, ctx_bound;
};
static_assert(sizeof(Params) == 136, "Params is the 136-byte attention kernarg");

// The QSA union list (32 bytes, by value): slot per listed key, visibility bytes, key count, row stride.
struct UnionKeys {
  cints kslot;             // [cta][k] page * 16 + slot, ascending key position
  const uint8_t* kbits;    // [cta][k / 8][row] visibility byte, bit e = key 8 b + e
  cints nkeys;             // [cta] listed keys
  long kstride;            // keys per cta row of kslot (kbits: kstride * 4 bytes per cta)
};
static_assert(sizeof(UnionKeys) == 32, "UnionKeys is the 32-byte QSA kernarg");

template <bool B, class T, class F> struct pick { typedef T type; };
template <class T, class F> struct pick<false, T, F> { typedef F type; };

constexpr int kHead = 256;      // head dim
constexpr int kPage = 16;       // KV slots per page
constexpr int kSlotElems = 512; // K (256) then V (256) per slot
constexpr float kLog2e = 1.44269504089f;
constexpr float kFloor = -3.0e38f;

// ---- scalar arithmetic the contract names

// raw v_exp_f32
__device__ __forceinline__ float ex2(float x) { return __builtin_amdgcn_exp2f(x); }

// f32 -> bf16 bits, round to nearest even by the integer rule
__device__ __forceinline__ uint32_t bf16_bits(float x) {
  uint32_t u = __builtin_bit_cast(uint32_t, x);
  return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
}

__device__ __forceinline__ uint32_t bf16_pair(float lo, float hi) {
  return bf16_bits(lo) | (bf16_bits(hi) << 16);
}

// two f32 -> packed f16, rounding toward zero (v_cvt_pkrtz_f16_f32)
__device__ __forceinline__ uint32_t rtz_pair(float lo, float hi) {
  return __builtin_bit_cast(uint32_t, __builtin_amdgcn_cvt_pkrtz(lo, hi));
}

// four f32 -> four e4m3 bytes, round to nearest even (two v_cvt_pk_fp8_f32)
__device__ __forceinline__ uint32_t e4m3_quad(float a, float b, float c, float d) {
  int w = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
  return (uint32_t)__builtin_amdgcn_cvt_pk_fp8_f32(c, d, w, true);
}

// bf16 bits (low / high half of a dword) -> f32
__device__ __forceinline__ float bf_lo(uint32_t w) { return __builtin_bit_cast(float, w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __builtin_bit_cast(float, w & 0xFFFF0000u); }

// e4m3 pair -> f32 pair (v_cvt_pk_f32_fp8). Issued in its VOP3 encoding on a full 32-bit VGPR: the builtin
// selects the VOP1 encoding, whose 16-bit source can only name v0..v127, and in the prefill those are the
// accumulators -- the allocator then spills the raw bytes and reloads them before every convert.
__device__ __forceinline__ f32x2 e4m3_pair_lo(uint32_t w) {
  f32x2 r;
  asm("v_cvt_pk_f32_fp8_e64 %0, %1" : "=v"(r) : "v"(w));
  return r;
}
__device__ __forceinline__ f32x2 e4m3_pair_hi(uint32_t w) {
  f32x2 r;
  asm("v_cvt_pk_f32_fp8_e64 %0, %1 op_sel:[1,0]" : "=v"(r) : "v"(w));
  return r;
}

// 4 e4m3 bytes -> 4 f16 (exact: f32 widen, then the packed RTZ convert)
__device__ __forceinline__ u32x2 e4m3x4_to_f16(uint32_t w) {
  f32x2 a = e4m3_pair_lo(w);
  f32x2 b = e4m3_pair_hi(w);
  u32x2 r;
  r.x = rtz_pair(a.x, a.y);
  r.y = rtz_pair(b.x, b.y);
  return r;
}

// 8 e4m3 bytes -> one f16x8 operand
__device__ __forceinline__ f16x8 e4m3x8_to_f16(u32x2 raw) {
  u32x2 lo = e4m3x4_to_f16(raw.x);
  u32x2 hi = e4m3x4_to_f16(raw.y);
  u32x4 r = {lo.x, lo.y, hi.x, hi.y};
  return __builtin_bit_cast(f16x8, r);
}

// 8 bf16 -> one f16x8 operand (RTZ)
__device__ __forceinline__ f16x8 bf16x8_to_f16(u32x4 raw) {
  u32x4 r;
  r.x = rtz_pair(bf_lo(raw.x), bf_hi(raw.x));
  r.y = rtz_pair(bf_lo(raw.y), bf_hi(raw.y));
  r.z = rtz_pair(bf_lo(raw.z), bf_hi(raw.z));
  r.w = rtz_pair(bf_lo(raw.w), bf_hi(raw.w));
  return __builtin_bit_cast(f16x8, r);
}

// denominator step: l + pa + pb in one fused v_dot2_f32_f16 against (1, 1)
__device__ __forceinline__ float dot2_ones(uint32_t pw, float l) {
  const f16x2 ones = {(_Float16)1.0f, (_Float16)1.0f};
  return __builtin_amdgcn_fdot2(__builtin_bit_cast(f16x2, pw), ones, l, false);
}

// ---- WMMA (gfx12 wave32). Operand roles are part of the bit contract: see each call site.
__device__ __forceinline__ f32x8 mma16(f16x8 a, f16x8 b, f32x8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}
__device__ __forceinline__ f32x8 mma8(u32x2 a, u32x2 b, f32x8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(__builtin_bit_cast(i32x2, a),
                                                               __builtin_bit_cast(i32x2, b), c);
}

// ---- lanes and barriers

// value of the same lane index in the other 16-lane half (identity selectors)
__device__ __forceinline__ float other_half(float v) {
  int r = __builtin_amdgcn_permlanex16(__builtin_bit_cast(int, v), __builtin_bit_cast(int, v), 0x76543210,
                                       (int)0xFEDCBA98, false, false);
  return __builtin_bit_cast(float, r);
}

__device__ __forceinline__ bool wave_any(bool p) { return __builtin_amdgcn_ballot_w32(p) != 0u; }

// a workgroup barrier that orders LDS only (no vector-cache invalidate)
__device__ __forceinline__ void lds_sync() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

__device__ __forceinline__ void sched_fence() { __builtin_amdgcn_sched_barrier(0); }

// ---- memory

// LDS pointers carried as 32-bit address-space-3 values (a generic pointer costs a 64-bit register pair)
#define SATTN_LDS __attribute__((address_space(3)))
template <class T>
__device__ __forceinline__ SATTN_LDS T* lds_ptr(T* p) {
  return (SATTN_LDS T*)p;
}

// 16-byte transposing global load: lane j of each 8-lane group receives the j-th 16-bit element of the
// 16 bytes addressed by each lane of its group (value i <- lane i of the group).
__device__ __forceinline__ u32x4 load_tr16(const void* p) {
  typedef i16x8 __attribute__((address_space(1))) * gptr;
  i16x8 v = __builtin_amdgcn_global_load_tr_b128_v8i16((gptr)(uintptr_t)p);
  return __builtin_bit_cast(u32x4, v);
}

__device__ __forceinline__ u32x2 ld8(const void* p) { return *(const u32x2*)p; }
__device__ __forceinline__ u32x4 ld16(const void* p) { return *(const u32x4*)p; }

// split 4 transposed dwords of e4m3 pairs (value i = {even dim, odd dim} of key i) into the even-dim
// row and the odd-dim row, 8 key bytes each
__device__ __forceinline__ void unzip_pairs(u32x4 v, u32x2& even, u32x2& odd) {
  even.x = __builtin_amdgcn_perm(v.y, v.x, 0x06040200u);
  even.y = __builtin_amdgcn_perm(v.w, v.z, 0x06040200u);
  odd.x = __builtin_amdgcn_perm(v.y, v.x, 0x07050301u);
  odd.y = __builtin_amdgcn_perm(v.w, v.z, 0x07050301u);
}

}  // namespace sattn
