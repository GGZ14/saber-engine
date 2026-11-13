// Saber source (2026-11-11): the MACaQ-W4A8 activation transform shared by every `*_had32`
// producer twin. NOT a reference port: the arithmetic is DEFINED here and reproduced bit for bit by the
// torch reference (`had32_ref`, not shipped).
//
// A producer's final pre-quant values v (the bf16-rounded value the shipped producer quantizes) become
//   u = v * t[k]                                  (one f32 multiply, the sim's `x.float() * t`)
//   x = H32 u  per 32-block along K               (Sylvester order, unnormalised butterfly, f32 adds)
//   x = x * (1/sqrt(32))                          (one f32 multiply, applied LAST; 0x3E3504F3)
// and the producer's own per-token e4m3 law then runs on x (amax over x, its scale, its rounding).
//
// THE LANE MAP. Every producer twin keeps its parent's ownership: thread `tid` owns the 8-element groups
// g = tid + i * T (T a multiple of 4), so a 32-block = 4 adjacent lanes (lane bits 0-1 = element bits 3-4)
// x 8 registers (element bits 0-2). Every K (5,120 / 8,704 / 3,072 / 6,144 / 17,408) is a multiple of 32
// and a TP=2 shard boundary is too, so a quad is either fully active or fully idle in each pass.
//
// THE ORDER (fixed; the reference does the same): stages on element bit 0, 1, 2 (in-register), then bit 3
// (lane xor 1), then bit 4 (lane xor 2); each stage maps the pair (lo, hi) -> (lo + hi, lo - hi). The
// cross-lane moves are DPP quad_perm (exact register moves). Contraction is OFF in this header: no
// multiply here may fuse into a following add (u = v * t then lo + hi would otherwise become an FMA).
#pragma once
#include <hip/hip_runtime.h>

#define MACAQ_HAD32_INV 0x1.6a09e6p-3f   // (float)(1 / sqrt(32)) = 0x3E3504F3, the sim's h32 entry

// quad_perm controls: [1,0,3,2] (lane xor 1) and [2,3,0,1] (lane xor 2)
#define MACAQ_DPP_XOR1 0xB1
#define MACAQ_DPP_XOR2 0x4E

__device__ __forceinline__ float macaq_dpp_xor1(float v) {
  return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), MACAQ_DPP_XOR1, 0xF, 0xF, false));
}
__device__ __forceinline__ float macaq_dpp_xor2(float v) {
  return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), MACAQ_DPP_XOR2, 0xF, 0xF, false));
}

// v[8]: this lane's 8 values of its group (element order); t8: the group's 8 entries of t (f32, 32-byte
// aligned: t + 8 * g); lane: this thread's lane id (only bits 0-1 are read). In place.
#ifdef MACAQ_PROBE
// ORACLE BUILDS ONLY (an internal oracle compiles a scratch object with -DMACAQ_PROBE; never
// shipped): the pre-transform values v land at probe_v[row * probe_k + (t8 - probe_t)], so the oracle holds
// the twin's own v against its parent's quant and the torch transform against the twin's output.
__device__ float *macaq_probe_v;
__device__ const float *macaq_probe_t;
__device__ int macaq_probe_k;
#endif

__device__ __forceinline__ void macaq_had32(float (&v)[8], const float *__restrict__ t8, int lane) {
#pragma clang fp contract(off)
#ifdef MACAQ_PROBE
  {
    const long e = (long)(t8 - macaq_probe_t);
#pragma unroll
    for (int j = 0; j < 8; ++j) macaq_probe_v[(long)blockIdx.x * macaq_probe_k + e + j] = v[j];
  }
#endif
  const float4 ta = reinterpret_cast<const float4 *>(t8)[0];
  const float4 tb = reinterpret_cast<const float4 *>(t8)[1];
  v[0] = v[0] * ta.x; v[1] = v[1] * ta.y; v[2] = v[2] * ta.z; v[3] = v[3] * ta.w;
  v[4] = v[4] * tb.x; v[5] = v[5] * tb.y; v[6] = v[6] * tb.z; v[7] = v[7] * tb.w;
#pragma unroll
  for (int s = 1; s < 8; s <<= 1) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (j & s) continue;
      const float lo = v[j], hi = v[j | s];
      v[j] = lo + hi;
      v[j | s] = lo - hi;
    }
  }
  const bool b0 = (lane & 1) != 0, b1 = (lane & 2) != 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float p = macaq_dpp_xor1(v[j]);
    v[j] = b0 ? (p - v[j]) : (v[j] + p);
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float p = macaq_dpp_xor2(v[j]);
    v[j] = b1 ? (p - v[j]) : (v[j] + p);
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = v[j] * MACAQ_HAD32_INV;
}
