// Saber source (2026-11-11): the `--macaq-had on` twins of `attn_glue`'s `attn_gate_quant_t256_tiled{0,1}`
// (the attention o_proj's A producer). The parent's lines up to its bf16-rounded p, then `macaq_had32`
// (macaq_had.h), then the parent's own scale law on the transformed values (DIVIDE by 448, divide by the
// scale in the convert). `had_t` (f32 [n] of this rank's columns) is appended LAST. Built like `attn_glue`
// (-ffp-contract=off, saber-kbuild's `attn_glue` prefix row). The parent code object is untouched.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "macaq_had.h"

typedef unsigned int u32;
union Vec8 { uint4 u4; __bf16 h[8]; };
__device__ __forceinline__ float bf16r(float v) { return (float)(__bf16)v; }
__device__ __forceinline__ u32 e4m3_pair(float a, float b, float scale) {
  a = fminf(fmaxf(a / scale, -448.f), 448.f);
  b = fminf(fmaxf(b / scale, -448.f), 448.f);
  return __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu;
}

template <int T, bool TILED>
__device__ __forceinline__ void macaq_attn_gate_quant_impl(
    const __bf16 *__restrict__ core, const __bf16 *__restrict__ gate, unsigned char *__restrict__ q,
    float *__restrict__ scale, int n, int hd, int gs, const float *__restrict__ had_t) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  core += (long)row * n;
  gate += (long)row * gs;
  if (!TILED) q += (long)row * n;
  const int mt = row >> 4, lrow = row & 15, kst = n >> 4;
  const int nv = n >> 3;
  constexpr int MAXG = 3;
  float sv[MAXG][8];
  float amax = 0.f;
#pragma unroll
  for (int ng = 0; ng < MAXG; ++ng) {
    const int g = tid + ng * T;
    if (g >= nv) break;
    Vec8 vc;
    vc.u4 = ((const uint4 *)core)[g];
    const int e = g << 3, h = e / hd, o = e - h * hd;
    Vec8 vg;
    vg.u4 = ((const uint4 *)(gate + (long)h * 2 * hd + hd + o))[0];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float sig = bf16r(1.f / (1.f + expf(-(float)vg.h[j])));
      sv[ng][j] = bf16r((float)vc.h[j] * sig);
    }
    macaq_had32(sv[ng], had_t + 8 * g, tid);
#pragma unroll
    for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(sv[ng][j]));
  }
  __shared__ float lds[T / 32];
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_down(amax, o));
  if ((tid & 31) == 0) lds[tid >> 5] = amax;
  __syncthreads();
  if (tid < T / 32) {
    float v = lds[tid];
    for (int o = T / 64; o; o >>= 1) v = fmaxf(v, __shfl_down(v, o));
    if (tid == 0) lds[0] = v;
  }
  __syncthreads();
  const float sc = fmaxf(lds[0] / 448.f, 1.f / (448.f * 512.f));
  if (tid == 0) scale[row] = sc;
#pragma unroll
  for (int mg = 0; mg < MAXG; ++mg) {
    const int g = tid + mg * T;
    if (g >= nv) break;
    u32 lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 4; j += 2) lo |= e4m3_pair(sv[mg][j], sv[mg][j + 1], sc) << (j * 8);
#pragma unroll
    for (int j = 4; j < 8; j += 2) hi |= e4m3_pair(sv[mg][j], sv[mg][j + 1], sc) << ((j - 4) * 8);
    if (TILED) ((uint2 *)q)[(mt * kst + (g >> 1)) * 32 + lrow + 16 * (g & 1)] = make_uint2(lo, hi);
    else ((uint2 *)q)[g] = make_uint2(lo, hi);
  }
}

#define AGQ(name, TILED)                                                                          \
  extern "C" __global__ __launch_bounds__(256) void name(                                         \
      const __bf16 *core, const __bf16 *gate, unsigned char *q, float *scale, int n, int hd,      \
      int gs, const float *had_t) {                                                               \
    macaq_attn_gate_quant_impl<256, TILED>(core, gate, q, scale, n, hd, gs, had_t);               \
  }
AGQ(attn_gate_quant_had32_t256_tiled0, false)
AGQ(attn_gate_quant_had32_t256_tiled1, true)
#undef AGQ
