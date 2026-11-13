// Saber source (2026-11-11): the `--macaq-had on` twins of `mxfp4_quant`'s three producers.
//
// Each `*_had32_*` symbol is its parent in `mxfp4_quant.hip` (+ `mxfp4_gdn_norm_tiled.h`) line for line up to
// the producer's final bf16-rounded values, then `macaq_had32` (macaq_had.h: * t, the 32-point Hadamard,
// * 1/sqrt(32)), then the parent's own amax / scale / e4m3 store on the transformed values. The ONLY new
// kernel argument is `had_t` (f32 [K] of this rank's columns), appended LAST. The parents are untouched (a
// separate code object): with the arm off nothing here is loaded. Built like `mxfp4_quant` (the default
// contraction): the copied lines contract exactly as the parent's do; the transform has contraction off.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "macaq_had.h"

typedef unsigned int uint32_tt;
union ArnqVec8 { uint4 u4; __bf16 h[8]; };
#define ARNQ_T 256

// the parents' amax -> scale -> e4m3 tail, shared (multiply by the reciprocal, clamp, RNE convert)
template <int ART, int MAXG, bool TILED>
__device__ __forceinline__ void macaq_q8_tail(float (&sv)[MAXG][8], int NV, unsigned char *__restrict__ q,
                                              float *__restrict__ scale, int row, int mt, int lrow, int kst) {
  const int tid = threadIdx.x;
  float amax = 0.f;
#pragma unroll
  for (int mg = 0; mg < MAXG; ++mg) {
    const int g = tid + mg * ART;
    if (g >= NV) break;
#pragma unroll
    for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(sv[mg][j]));
  }
  __shared__ float lds[ART / 32];
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_down(amax, o));
  if ((tid & 31) == 0) lds[tid >> 5] = amax;
  __syncthreads();
  if (tid < ART / 32) {
    float v = lds[tid];
    for (int o = ART / 64; o; o >>= 1) v = fmaxf(v, __shfl_down(v, o));
    if (tid == 0) lds[0] = v;
  }
  __syncthreads();
  const float sc = fmaxf(lds[0] * (1.f / 448.f), 1.f / (448.f * 512.f));
  const float rs = 1.f / sc;
  if (tid == 0) scale[row] = sc;
#pragma unroll
  for (int mg = 0; mg < MAXG; ++mg) {
    const int g = tid + mg * ART;
    if (g >= NV) break;
    uint32_tt lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 4; j += 2) {
      float a = fminf(fmaxf(sv[mg][j] * rs, -448.f), 448.f);
      float b = fminf(fmaxf(sv[mg][j + 1] * rs, -448.f), 448.f);
      lo |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu) << (j * 8);
    }
#pragma unroll
    for (int j = 4; j < 8; j += 2) {
      float a = fminf(fmaxf(sv[mg][j] * rs, -448.f), 448.f);
      float b = fminf(fmaxf(sv[mg][j + 1] * rs, -448.f), 448.f);
      hi |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu) << ((j - 4) * 8);
    }
    if (TILED) ((uint2 *)q)[(mt * kst + (g >> 1)) * 32 + lrow + 16 * (g & 1)] = make_uint2(lo, hi);
    else ((uint2 *)q)[(long)row * (NV * 8 / 8) + g] = make_uint2(lo, hi);
  }
}

// ---- add_rms_quant (parent: saber_add_rms_quant_impl) ----------------------------------------
template <int ART, bool TILED>
__device__ __forceinline__ void macaq_add_rms_quant_impl(
    const __bf16 *__restrict__ y, const __bf16 *__restrict__ res, const __bf16 *__restrict__ w,
    unsigned char *__restrict__ q, float *__restrict__ scale, __bf16 *__restrict__ res_out,
    int K, float eps, const float *__restrict__ had_t) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  y += (long)row * K; res += (long)row * K; res_out += (long)row * K;
  const int mt = row >> 4, lrow = row & 15, kst = K >> 4;
  constexpr int MAXG = 5;
  const int KV = K >> 3;
  float sv[MAXG][8];
  float ssq = 0.f;
#pragma unroll
  for (int ng = 0; ng < MAXG; ++ng) {
    const int g = tid + ng * ART;
    if (g >= KV) break;
    ArnqVec8 vy, vr;
    vy.u4 = ((const uint4 *)y)[g];
    vr.u4 = ((const uint4 *)res)[g];
    ArnqVec8 vo;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float v = (float)vy.h[j] + (float)vr.h[j];
      sv[ng][j] = v;
      vo.h[j] = (__bf16)v;
      ssq = __builtin_fmaf(v, v, ssq);
    }
    ((uint4 *)res_out)[g] = vo.u4;
  }
  __shared__ float lds[ART / 32];
  for (int o = 16; o; o >>= 1) ssq += __shfl_down(ssq, o);
  if ((tid & 31) == 0) lds[tid >> 5] = ssq;
  __syncthreads();
  if (tid < ART / 32) {
    float v = lds[tid];
    for (int o = ART / 64; o; o >>= 1) v += __shfl_down(v, o);
    if (tid == 0) lds[0] = v;
  }
  __syncthreads();
  const float inv = rsqrtf(lds[0] / (float)K + eps);
  __syncthreads();
#pragma unroll
  for (int mg = 0; mg < MAXG; ++mg) {
    const int g = tid + mg * ART;
    if (g >= KV) break;
    ArnqVec8 vw;
    vw.u4 = ((const uint4 *)w)[g];
#pragma unroll
    for (int j = 0; j < 8; ++j) sv[mg][j] = (float)(__bf16)(sv[mg][j] * inv * ((float)vw.h[j] + 1.f));
    macaq_had32(sv[mg], had_t + 8 * g, tid);
  }
  __syncthreads();   // lds is re-declared by the tail; the sums above are all read
  macaq_q8_tail<ART, MAXG, TILED>(sv, KV, q, scale, row, mt, lrow, kst);
}

// ---- silu_mul_quant (parent: saber_silu_mul_quant_impl) --------------------------------------
template <int ART, bool TILED>
__device__ __forceinline__ void macaq_silu_mul_quant_impl(
    const __bf16 *__restrict__ gu, unsigned char *__restrict__ q, float *__restrict__ scale, int N,
    const float *__restrict__ had_t) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  gu += (long)row * 2 * N;
  const int mt = row >> 4, lrow = row & 15, kst = N >> 4;
  constexpr int MAXG = 5;
  const int NV = N >> 3;
  float pv[MAXG][8];
#pragma unroll
  for (int ng = 0; ng < MAXG; ++ng) {
    const int g = tid + ng * ART;
    if (g >= NV) break;
    ArnqVec8 vg, vu;
    vg.u4 = ((const uint4 *)gu)[g];
    vu.u4 = ((const uint4 *)(gu + N))[g];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float x = (float)vg.h[j];
      float t = (float)(__bf16)(x / (1.f + expf(-x)));
      float p = (float)(__bf16)(t * (float)vu.h[j]);
      pv[ng][j] = p;
    }
    macaq_had32(pv[ng], had_t + 8 * g, tid);
  }
  macaq_q8_tail<ART, MAXG, TILED>(pv, NV, q, scale, row, mt, lrow, kst);
}

// ---- gdn_norm_quant (parents: saber_gdn_norm_quant_impl, saber_gdn_norm_quant_tiled_impl) ----
template <int ART, bool TILED>
__device__ __forceinline__ void macaq_gdn_norm_quant_impl(
    const __bf16 *__restrict__ x, const __bf16 *__restrict__ z, long zs,
    const __bf16 *__restrict__ w, unsigned char *__restrict__ q, float *__restrict__ scale,
    int N, float eps, const float *__restrict__ had_t) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  x += (long)row * N; z += (long)row * zs;
  const int mt = row >> 4, lrow = row & 15, kst = N >> 4;
  constexpr int MAXG = 5;
  const int NV = N >> 3;
  float sv[MAXG][8];
#pragma unroll
  for (int ng = 0; ng < MAXG; ++ng) {
    const int g = tid + ng * ART;
    if (g >= NV) break;
    ArnqVec8 vx, vz, vw;
    vx.u4 = ((const uint4 *)x)[g];
#pragma unroll
    for (int j = 0; j < 8; ++j) sv[ng][j] = (float)vx.h[j];
    float ssq = sv[ng][1] * sv[ng][1];
    ssq = __builtin_fmaf(sv[ng][0], sv[ng][0], ssq);
#pragma unroll
    for (int j = 2; j < 8; ++j) ssq = __builtin_fmaf(sv[ng][j], sv[ng][j], ssq);
    for (int o = 8; o; o >>= 1) ssq += __shfl_xor(ssq, o, 16);
    const float inv = rsqrtf(ssq * (1.f / 128.f) + eps);
    vz.u4 = ((const uint4 *)z)[g];
    vw.u4 = ((const uint4 *)w)[g & 15];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float zz = (float)vz.h[j];
      const float sg = zz / (1.f + expf(-zz));
      sv[ng][j] = (float)(__bf16)(((sv[ng][j] * inv) * (float)vw.h[j]) * sg);
    }
    macaq_had32(sv[ng], had_t + 8 * g, tid);
  }
  macaq_q8_tail<ART, MAXG, TILED>(sv, NV, q, scale, row, mt, lrow, kst);
}

#define ARQ(name, T, TL)                                                                              \
  extern "C" __global__ __launch_bounds__(T) void name(                                               \
      const __bf16 *__restrict__ y, const __bf16 *__restrict__ res, const __bf16 *__restrict__ w,     \
      unsigned char *__restrict__ q, float *__restrict__ scale, __bf16 *__restrict__ res_out, int K,  \
      float eps, const float *__restrict__ had_t) {                                                   \
    macaq_add_rms_quant_impl<T, TL>(y, res, w, q, scale, res_out, K, eps, had_t);                     \
  }
ARQ(add_rms_quant_had32_t256_tiled0, 256, false)
ARQ(add_rms_quant_had32_t256_tiled1, 256, true)
ARQ(add_rms_quant_had32_t512_tiled0, 512, false)
ARQ(add_rms_quant_had32_t512_tiled1, 512, true)
#undef ARQ

#define SMQ(name, T, TL)                                                                              \
  extern "C" __global__ __launch_bounds__(T) void name(                                               \
      const __bf16 *__restrict__ gu, unsigned char *__restrict__ q, float *__restrict__ scale, int N, \
      const float *__restrict__ had_t) {                                                              \
    macaq_silu_mul_quant_impl<T, TL>(gu, q, scale, N, had_t);                                         \
  }
SMQ(silu_mul_quant_had32_t256_tiled0, 256, false)
SMQ(silu_mul_quant_had32_t256_tiled1, 256, true)
SMQ(silu_mul_quant_had32_t512_tiled0, 512, false)
SMQ(silu_mul_quant_had32_t512_tiled1, 512, true)
#undef SMQ

#define GNQ(name, TL)                                                                                 \
  extern "C" __global__ __launch_bounds__(ARNQ_T) void name(                                          \
      const __bf16 *__restrict__ x, const __bf16 *__restrict__ z, long zs,                            \
      const __bf16 *__restrict__ w, unsigned char *__restrict__ q, float *__restrict__ scale, int N,  \
      float eps, const float *__restrict__ had_t) {                                                   \
    macaq_gdn_norm_quant_impl<ARNQ_T, TL>(x, z, zs, w, q, scale, N, eps, had_t);                      \
  }
GNQ(gdn_norm_quant_had32_t256, false)
GNQ(gdn_norm_quant_had32_t256_tiled1, true)
#undef GNQ
