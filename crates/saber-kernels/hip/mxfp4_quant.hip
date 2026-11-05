// Saber source (CLEANROOM-1, 2026-10-25: frozen from a retired generator's output; edit this file directly)
//   reference sha256 79a7eb6f261c2939386a4c231cf5727a0a1f2325d8a391419178778da1fd2ae2
//   device code sliced VERBATIM by line range; only the __global__ signatures became
//   __device__ __forceinline__ *_impl and the extern "C" wrappers at the end are new.
// Do not edit by hand: re-run the generator. No host code, no pybind, no getenv in this file --
// every tile parameter the prod launcher chose from an env var is an explicit symbol here.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "mxfp4_common.h"

// ---- fused residual-add + Gemma RMSNorm + per-token fp8 quant -------------------------------
//
// The decode-step epilogue of every RowParallel linear: after the all-reduce, the traced graph
// runs (add+rms_norm) and the traced per-token quant -- 2-3 inductor kernels per site whose
// tiling inductor chooses for the M=8192 compile hint and replays at M=8 (XBLOCK=1, one row per
// workgroup, ~4.4 us of kernels + 2 dispatch gaps per site). This kernel does the whole epilogue
// in ONE launch. Semantics mirror upstream.ir.ops.fused_add_rms_norm with the GemmaRMSNorm weight
// convention plus the reference traced quant, in exactly its precision:
//   s      = f32(y) + f32(residual)         (fp32 sum feeds BOTH residual_out and the variance)
//   res_o  = bf16(s)
//   n      = bf16( s * rsqrt(mean(s^2)+eps) * (f32(w)+1) )   <- Gemma: weight is (1+w), fp32
//   scale  = max(amax|f32(n)|/448, 1/(448*512))               <- amax over the BF16-ROUNDED n
//   q      = e4m3_rne(clamp(f32(n)/scale, +-448))
// One workgroup per row; latency-bound at decode M<=8 like everything else in the step, and at
// prefill M=chunk there are thousands of rows so occupancy takes care of itself.
#define ARNQ_T 256
typedef unsigned int uint32_tt;
union ArnqVec8 { uint4 u4; __bf16 h[8]; };

template <int ART, bool TILED = false>
__device__ __forceinline__ void saber_add_rms_quant_impl(
    const __bf16 *__restrict__ y, const __bf16 *__restrict__ res, const __bf16 *__restrict__ w,
    unsigned char *__restrict__ q, float *__restrict__ scale, __bf16 *__restrict__ res_out,
    int K, float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  y += (long)row * K; res += (long)row * K; res_out += (long)row * K;
  if (!TILED) q += (long)row * K;
  // TILED: q is the fragment-tiled layout the atiled GEMM reads (see there): 8-group g of this
  // row lands in fragment (row/16, g/2) at lane (row%16) + 16*(g%2). Same single uint2 store;
  // the scatter across fragments measured +0.3..+3.8% on this kernel (tier7/aq.hip).
  const int mt = row >> 4, lrow = row & 15, kst = K >> 4;
  // 16-byte vectorized: thread t owns 8-element groups t, t+T, t+2T, ... The scalar version of
  // this kernel measured 12.5 us at M=8 under graph replay -- 20 independent 2-byte loads per
  // thread per tensor is pure latency serialization at 8 workgroups. Vectorized it is one
  // uint4 load per tensor per group. K must be a multiple of 8 (5120/8704/2048 all are).
  constexpr int MAXG = 5;                       // 8704 / (256*8) rounds up to 5
  const int KV = K >> 3;                        // groups of 8
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
      ssq = __builtin_fmaf(v, v, ssq);   // ws-h45: main's chain, pinned (EXEC section 8)
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
  __syncthreads();                       // lds[0] is reused for the amax reduction below
  float amax = 0.f;
#pragma unroll
  for (int mg = 0; mg < MAXG; ++mg) {
    const int g = tid + mg * ART;
    if (g >= KV) break;
    ArnqVec8 vw;
    vw.u4 = ((const uint4 *)w)[g];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float nv = (float)(__bf16)(sv[mg][j] * inv * ((float)vw.h[j] + 1.f));
      sv[mg][j] = nv;
      amax = fmaxf(amax, fabsf(nv));
    }
  }
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
    if (g >= KV) break;
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
    else ((uint2 *)q)[g] = make_uint2(lo, hi);
  }
}

// ---- fused SiLU-mul + per-token fp8 quant ----------------------------------------------------
// The MLP's activation epilogue: gate_up output [M, 2N] -> silu(gate) * up -> per-token e4m3.
// Same skeleton and precision contract as saber_add_rms_quant: every intermediate is rounded
// exactly where the traced path rounds (silu to bf16, product to bf16, amax over the rounded
// product). One workgroup per row.

template <int ART, bool TILED = false>
__device__ __forceinline__ void saber_silu_mul_quant_impl(
    const __bf16 *__restrict__ gu, unsigned char *__restrict__ q, float *__restrict__ scale,
    int N) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  gu += (long)row * 2 * N;
  if (!TILED) q += (long)row * N;
  const int mt = row >> 4, lrow = row & 15, kst = N >> 4;   // TILED: as in add_rms_quant
  constexpr int MAXG = 5;                  // N <= 10240 (8704 in this model)
  const int NV = N >> 3;
  float pv[MAXG][8];
  float amax = 0.f;
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
      float t = (float)(__bf16)(x / (1.f + expf(-x)));       // silu, rounded to bf16
      float p = (float)(__bf16)(t * (float)vu.h[j]);         // product, rounded to bf16
      pv[ng][j] = p;
      amax = fmaxf(amax, fabsf(p));
    }
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
      float a = fminf(fmaxf(pv[mg][j] * rs, -448.f), 448.f);
      float b = fminf(fmaxf(pv[mg][j + 1] * rs, -448.f), 448.f);
      lo |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu) << (j * 8);
    }
#pragma unroll
    for (int j = 4; j < 8; j += 2) {
      float a = fminf(fmaxf(pv[mg][j] * rs, -448.f), 448.f);
      float b = fminf(fmaxf(pv[mg][j + 1] * rs, -448.f), 448.f);
      hi |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu) << ((j - 4) * 8);
    }
    if (TILED) ((uint2 *)q)[(mt * kst + (g >> 1)) * 32 + lrow + 16 * (g & 1)] = make_uint2(lo, hi);
    else ((uint2 *)q)[g] = make_uint2(lo, hi);
  }
}

// ---- fused per-head gated RMSNorm + per-token fp8 quant (GDN out_proj input) ----------------
// Replaces RMSNormGated(norm_before_gate=True, group per 128-wide head) + the traced per-token
// quant in _output_projection, which inductor emits as two kernels per GDN layer (a variance
// reduction and a normalize+gate+quant). Precision contract mirrors saber_add_rms_quant and
// the traced reference: fp32 throughout, ((x * rsqrt(mean(x^2) + eps)) * w) * silu(z) rounded
// to bf16 once, amax over the rounded values, e4m3 with the reference min-scale clamp. The per-head
// sum of squares is a 16-lane xor-shuffle: 16 consecutive 8-groups make one head and, with ART a
// multiple of 16 and N a multiple of 128, every 16-lane segment is either fully active or fully
// idle in each pass. x: [M, N] contiguous bf16; z: [M, N] bf16 with row stride zs (a column
// slice of the fused qkvz projection, no copy); w: [128] bf16 shared by all heads.

template <int ART>
__device__ __forceinline__ void saber_gdn_norm_quant_impl(
    const __bf16 *__restrict__ x, const __bf16 *__restrict__ z, long zs,
    const __bf16 *__restrict__ w, unsigned char *__restrict__ q, float *__restrict__ scale,
    int N, float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  x += (long)row * N; z += (long)row * zs; q += (long)row * N;
  constexpr int MAXG = 5;                       // N <= ART*8*5 = 10240 (3072 in this model)
  const int NV = N >> 3;                        // 8-element groups; 16 per head
  float sv[MAXG][8];
  float amax = 0.f;
#pragma unroll
  for (int ng = 0; ng < MAXG; ++ng) {
    const int g = tid + ng * ART;
    if (g >= NV) break;
    ArnqVec8 vx, vz, vw;
    vx.u4 = ((const uint4 *)x)[g];
#pragma unroll
    for (int j = 0; j < 8; ++j) sv[ng][j] = (float)vx.h[j];
    // ws-h45: main's compiled chain, pinned (EXEC section 8): x1^2 rounded, x0^2 fused onto it,
    // then x2..x7 fused in order -- the zero seed folded the way the runtime loop folds it.
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
      const float sg = zz / (1.f + expf(-zz));                       // silu, fp32
      const float nv = (float)(__bf16)(((sv[ng][j] * inv) * (float)vw.h[j]) * sg);
      sv[ng][j] = nv;
      amax = fmaxf(amax, fabsf(nv));
    }
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
    ((uint2 *)q)[g] = make_uint2(lo, hi);
  }
}


extern "C" __global__ __launch_bounds__(256) void add_rms_quant_t256_tiled0(
    const __bf16 *__restrict__ y, const __bf16 *__restrict__ res, const __bf16 *__restrict__ w,
    unsigned char *__restrict__ q, float *__restrict__ scale, __bf16 *__restrict__ res_out,
    int K, float eps) {
  saber_add_rms_quant_impl<256, false>(y, res, w, q, scale, res_out, K, eps);
}

extern "C" __global__ __launch_bounds__(256) void add_rms_quant_t256_tiled1(
    const __bf16 *__restrict__ y, const __bf16 *__restrict__ res, const __bf16 *__restrict__ w,
    unsigned char *__restrict__ q, float *__restrict__ scale, __bf16 *__restrict__ res_out,
    int K, float eps) {
  saber_add_rms_quant_impl<256, true>(y, res, w, q, scale, res_out, K, eps);
}

extern "C" __global__ __launch_bounds__(512) void add_rms_quant_t512_tiled0(
    const __bf16 *__restrict__ y, const __bf16 *__restrict__ res, const __bf16 *__restrict__ w,
    unsigned char *__restrict__ q, float *__restrict__ scale, __bf16 *__restrict__ res_out,
    int K, float eps) {
  saber_add_rms_quant_impl<512, false>(y, res, w, q, scale, res_out, K, eps);
}

extern "C" __global__ __launch_bounds__(512) void add_rms_quant_t512_tiled1(
    const __bf16 *__restrict__ y, const __bf16 *__restrict__ res, const __bf16 *__restrict__ w,
    unsigned char *__restrict__ q, float *__restrict__ scale, __bf16 *__restrict__ res_out,
    int K, float eps) {
  saber_add_rms_quant_impl<512, true>(y, res, w, q, scale, res_out, K, eps);
}

extern "C" __global__ __launch_bounds__(256) void silu_mul_quant_t256_tiled0(
    const __bf16 *__restrict__ gu, unsigned char *__restrict__ q, float *__restrict__ scale, int N) {
  saber_silu_mul_quant_impl<256, false>(gu, q, scale, N);
}

extern "C" __global__ __launch_bounds__(256) void silu_mul_quant_t256_tiled1(
    const __bf16 *__restrict__ gu, unsigned char *__restrict__ q, float *__restrict__ scale, int N) {
  saber_silu_mul_quant_impl<256, true>(gu, q, scale, N);
}

extern "C" __global__ __launch_bounds__(512) void silu_mul_quant_t512_tiled0(
    const __bf16 *__restrict__ gu, unsigned char *__restrict__ q, float *__restrict__ scale, int N) {
  saber_silu_mul_quant_impl<512, false>(gu, q, scale, N);
}

extern "C" __global__ __launch_bounds__(512) void silu_mul_quant_t512_tiled1(
    const __bf16 *__restrict__ gu, unsigned char *__restrict__ q, float *__restrict__ scale, int N) {
  saber_silu_mul_quant_impl<512, true>(gu, q, scale, N);
}

extern "C" __global__ __launch_bounds__(ARNQ_T) void gdn_norm_quant_t256(
    const __bf16 *__restrict__ x, const __bf16 *__restrict__ z, long zs,
    const __bf16 *__restrict__ w, unsigned char *__restrict__ q, float *__restrict__ scale,
    int N, float eps) {
  saber_gdn_norm_quant_impl<ARNQ_T>(x, z, zs, w, q, scale, N, eps);
}

// ---------------------------------------------------------------- SABER-AUTHORED (G2)
// Everything below is NOT from the reference: see mxfp4_gdn_norm_tiled.h.
#include "mxfp4_gdn_norm_tiled.h"

extern "C" __global__ __launch_bounds__(ARNQ_T) void gdn_norm_quant_t256_tiled1(
    const __bf16 *__restrict__ x, const __bf16 *__restrict__ z, long zs,
    const __bf16 *__restrict__ w, unsigned char *__restrict__ q, float *__restrict__ scale,
    int N, float eps) {
  saber_gdn_norm_quant_tiled_impl<ARNQ_T>(x, z, zs, w, q, scale, N, eps);
}
