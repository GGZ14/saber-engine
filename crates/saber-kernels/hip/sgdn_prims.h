// Saber gated-delta-net kernels, CLEANROOM-2 (CR2-3c): register-level primitives.
//
// Written from the CR2-3a specification (saber-cache/cr2-3/spec/gdn.md), not from any earlier GDN source.
// Each helper is the one instruction or operation sequence the arithmetic contract (spec section 2.1) pins:
// nothing here may be replaced by a library routine or by a "mathematically equal" form.
//
// Includability (spec 1.5.1): two other objects paste the GDN translation units in as text after remapping
// __global__ / blockIdx / gridDim / blockDim (and, in one of them, __syncthreads). Everything below lives in
// namespace sgdn, reads work indices only through those names, and is guarded by #pragma once.
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace sgdn {

using u16 = unsigned short;
using u32 = unsigned int;
using bf16x8 = short __attribute__((ext_vector_type(8)));   // one bf16 WMMA operand (8 elements a lane)
using f32x8 = float __attribute__((ext_vector_type(8)));    // one WMMA accumulator
using u32x2 = u32 __attribute__((ext_vector_type(2)));
using u32x4 = u32 __attribute__((ext_vector_type(4)));

// kernel parameter types: the kernarg layout (offsets, sizes, names) is the ABI, these spellings are ours
typedef const u16* __restrict__ cbf;              // read-only 16-bit elements (bf16 / raw)
typedef u16* __restrict__ wbf;                    // written 16-bit elements
typedef const float* __restrict__ cf32;
typedef float* __restrict__ wf32;
typedef const int* __restrict__ ci32;
typedef int* __restrict__ wi32;
typedef const void* __restrict__ cany;
typedef void* __restrict__ wany;
typedef const unsigned char* __restrict__ cb8;
typedef unsigned char* __restrict__ wb8;
typedef u32* __restrict__ wu32;
typedef unsigned long long* __restrict__ wu64;

constexpr int kDim = 128;   // head_k = head_v
constexpr int kTaps = 4;    // conv width
constexpr int kHist = 3;    // conv taps carried between steps
constexpr int kHalf = 64;   // k elements a thread owns in the decode recurrence (two threads a v row)
constexpr int kChunk = 64;  // prefill chunk

// ---- conversions (spec 2.1.1)

// bf16 bits -> f32 (exact)
__device__ __forceinline__ float bf(u32 bits16) { return __uint_as_float(bits16 << 16); }
// high bf16 of a dword -> f32 (exact)
__device__ __forceinline__ float bf_hi(u32 w) { return __uint_as_float(w & 0xFFFF0000u); }
__device__ __forceinline__ float bf_lo(u32 w) { return __uint_as_float(w << 16); }

// f32 -> bf16 by adding half an ulp of the bf16 grid to the bit pattern and truncating (ties away from zero)
__device__ __forceinline__ u16 away16(float x) { return (u16)((__float_as_uint(x) + 0x8000u) >> 16); }

// two away16 roundings packed {hi = b, lo = a}: the high halves of the two biased words, one byte permute
__device__ __forceinline__ u32 away16x2(float a, float b) {
  return __builtin_amdgcn_perm(__float_as_uint(b) + 0x8000u, __float_as_uint(a) + 0x8000u, 0x07060302u);
}

// f32 -> bf16, round to nearest even (the compiler's __bf16 conversion; NaN -> quiet NaN)
__device__ __forceinline__ u16 rne16(float x) { return __builtin_bit_cast(u16, (__bf16)x); }

// f32 -> f16 bits (v_cvt_f16_f32: RNE, subnormals kept) and back (exact)
__device__ __forceinline__ u32 to_h(float x) { return (u32)__builtin_bit_cast(u16, (_Float16)x); }
__device__ __forceinline__ float from_h(u32 bits16) {
  return (float)__builtin_bit_cast(_Float16, (u16)bits16);
}
// two f16 roundings packed {hi = b, lo = a}
__device__ __forceinline__ u32 to_h2(float a, float b) { return __builtin_amdgcn_perm(to_h(b), to_h(a), 0x05040100u); }

// ---- transcendental instructions (spec 2.1.2 - 2.1.4)

// raw v_exp_f32
__device__ __forceinline__ float ex2(float y) { return __builtin_amdgcn_exp2f(y); }
// E(x) = ex2(f32(x * log2 e)), the HIP fast exponential
__device__ __forceinline__ float E(float x) { return __expf(x); }
// L(x): the HIP fast logarithm, compiled to the accurate natural-log sequence
__device__ __forceinline__ float L(float x) { return __logf(x); }
// R(x): v_rsq_f32
__device__ __forceinline__ float R(float x) { return __frsqrt_rn(x); }

// ---- gate functions (spec 2.1.5)

// softplus with its linear tail past thr (ordered compares: a NaN takes the last arm)
__device__ __forceinline__ float softplus(float x, float thr) {
  if (x > thr) return x;
  if (x > 0.0f) return x + L(1.0f + E(-x));
  return L(1.0f + E(x));
}
// beta = 1 / (1 + E(-b))
__device__ __forceinline__ float sigmoid_div(float b) { return 1.0f / (1.0f + E(-b)); }
// conv activation z / (1 + E(-z))
__device__ __forceinline__ float silu_div(float z) { return z / (1.0f + E(-z)); }

// ---- lanes (spec 2.1.7): every shuffle is a ds_bpermute; the order of each reduction is the contract

__device__ __forceinline__ float pair_add(float x) { return x + __shfl_xor(x, 1, 32); }

// xor butterfly over the lane distances FIRST, FIRST/2, ..., 1 (every lane ends with the same sum)
template <int FIRST>
__device__ __forceinline__ float butterfly_sum(float s) {
#pragma unroll
  for (int d = FIRST; d > 0; d >>= 1) s = s + __shfl_xor(s, d, 32);
  return s;
}
__device__ __forceinline__ float wave_sum(float s) { return butterfly_sum<16>(s); }
__device__ __forceinline__ float half_wave_sum(float s) { return butterfly_sum<8>(s); }

// ---- state storage (spec 2.1.6): f32 = 64 floats a thread as 16-byte vectors; f16 = RNE narrow, exact widen

enum StateKind { kStateF32 = 0, kStateF16 = 2 };

template <int KIND> struct StateElem;
template <> struct StateElem<kStateF32> { typedef float type; };
template <> struct StateElem<kStateF16> { typedef u16 type; };

template <int KIND, int N>
__device__ __forceinline__ void state_load(float* h, const typename StateElem<KIND>::type* src) {
  if constexpr (KIND == kStateF32) {
#pragma unroll
    for (int j = 0; j < N; j += 4) {
      const float4 f = *(const float4*)(src + j);
      h[j] = f.x; h[j + 1] = f.y; h[j + 2] = f.z; h[j + 3] = f.w;
    }
  } else {
#pragma unroll
    for (int j = 0; j < N; j += 8) {
      const u32x4 w = *(const u32x4*)(src + j);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        h[j + 2 * e] = from_h(w[e] & 0xFFFFu);
        h[j + 2 * e + 1] = from_h(w[e] >> 16);
      }
    }
  }
}

template <int KIND, int N>
__device__ __forceinline__ void state_store(typename StateElem<KIND>::type* dst, const float* h) {
  if constexpr (KIND == kStateF32) {
#pragma unroll
    for (int j = 0; j < N; j += 4) *(float4*)(dst + j) = make_float4(h[j], h[j + 1], h[j + 2], h[j + 3]);
  } else {
#pragma unroll
    for (int j = 0; j < N; j += 8) {
      u32x4 w;
#pragma unroll
      for (int e = 0; e < 4; ++e) w[e] = to_h2(h[j + 2 * e], h[j + 2 * e + 1]);
      *(u32x4*)(dst + j) = w;
    }
  }
}

// h <- hf(h(h)): exactly what an f16 store and reload would give (the f16 carry, spec 2.4.4)
template <int N>
__device__ __forceinline__ void round_through_f16(float* h) {
#pragma unroll
  for (int j = 0; j < N; j += 2) {
    const u32 w = to_h2(h[j], h[j + 1]);
    h[j] = from_h(w & 0xFFFFu);
    h[j + 1] = from_h(w >> 16);
  }
}

// ---- bf16 WMMA (spec 2.1.8): D[m][n] = C[m][n] + sum_k A[m][k] B[n][k], one v_wmma_f32_16x16x16_bf16.
// Operand convention (A and B alike): lane (c = lane & 15, h = lane >> 4) holds row c, and its element e carries
// logical k = 8 (e >> 2) + 4 h + (e & 3) -- two runs of four, at k 4h.. and 8+4h... Accumulator: lane (c, h)
// element e = D[8h + e][c]. Chains are issued in the order written; never reassociated.
__device__ __forceinline__ f32x8 wmma(bf16x8 a, bf16x8 b, f32x8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

__device__ __forceinline__ f32x8 zero8() { return f32x8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}; }

// one operand from a k-contiguous bf16 row: the lane's two 8-byte runs (row + 4h, row + 8 + 4h)
__device__ __forceinline__ bf16x8 frag_row(const u16* row, int h) {
  const u32x2 a = *(const u32x2*)(row + 4 * h);
  const u32x2 b = *(const u32x2*)(row + 8 + 4 * h);
  return __builtin_bit_cast(bf16x8, u32x4{a.x, a.y, b.x, b.y});
}

// one operand whose logical k runs down a column: element e = base[kk(e) * pitch], eight scalar reads
__device__ __forceinline__ bf16x8 frag_col(const u16* base, int pitch, int h) {
  bf16x8 r;
#pragma unroll
  for (int e = 0; e < 8; ++e) r[e] = (short)base[(8 * (e >> 2) + 4 * h + (e & 3)) * pitch];
  return r;
}

// ---- barriers

// a workgroup barrier that orders LDS only (does not wait for this workgroup's global stores)
__device__ __forceinline__ void lds_sync() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

}  // namespace sgdn
