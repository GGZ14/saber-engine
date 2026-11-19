// Saber source (2026-11-18): the W4A8-iu8 DFlash2 drafter's activation producers (README, "W4A8-iu8").
// Each is the int8 TWIN of a shipped drafter producer: the parent's arithmetic line for line up to its final
// pre-quant value v, then `macaq_had32` (macaq_had.h: * t, the 32-point Hadamard, * 1/sqrt(32)), then the iu8 pack's
// per-token INT8 law (macaq_i8.h: absmax / 127 over the WHOLE row, `__fdiv_rn`, rint half-even, clamp +-127) stored
// A-TILED (the macaq_iu8 GEMMs' A operand at every M) with one f32 scale a row.
//
//   dflash_hadi8_act_t{T}g{G}    parent `dflash_act_quant_g128` / `_recip` (the aux projection's input and every
//                                layer's o_proj input): v = f32(x), the materialised bf16 tensor itself
//   dflash_hadi8_conv_t{T}g{G}   parent `dflash_grouped_conv_quant_t2g16` (the `prepare` conv feeding attn.qkv and
//                                mlp.gate_up): v = bf16(conv) -- the parent quantises the bf16-ROUNDED conv output
//   dflash_hadi8_silu_t{T}g{G}   parent `dflash_silu_quant_g128` (mlp.down's input): v = silu(gate) * up in FP32 (the
//                                parent's activation is never rounded: the quant is its only consumer)
//
// ONE WORKGROUP A ROW (the per-token scale is a reduction over the row's K, which the parents' per-128-group waves
// never needed): thread `tid` owns the 8-element groups g = tid + i * T (i < G), so a 32-block = 4 adjacent lanes
// (macaq_had.h's lane map; every K here is a multiple of 32, so a quad is wholly active or wholly idle). The parents
// are untouched (another code object). Built at the default contraction like the parents (`dflash_conv` /
// `dflash_mlp`): the copied conv / silu lines contract exactly as theirs do; the transform has contraction off.
// One launch a site, as the parent: the launch count of a drafter proposal does not move.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "macaq_had.h"
#include "macaq_i8.h"

typedef __bf16 bf16x4 __attribute__((ext_vector_type(4)));
union DhVec8 { uint4 u4; __bf16 h[8]; };

// ---- the int8 tail (= mxfp4_hadi8.hip's macaq_i8_tail, A-tiled): amax over the row, sc, codes ----------------------
template <int T, int G>
__device__ __forceinline__ void dh_i8_tail(float (&sv)[G][8], int NV, unsigned char *__restrict__ q,
                                           float *__restrict__ scale, int row, int kst) {
  const int tid = threadIdx.x;
  const int mt = row >> 4, lrow = row & 15;
  float amax = 0.f;
#pragma unroll
  for (int mg = 0; mg < G; ++mg) {
    const int g = tid + mg * T;
    if (g >= NV) break;
#pragma unroll
    for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(sv[mg][j]));
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
  const float sc = fmaxf(__fdiv_rn(lds[0], 127.f), 1e-30f);
  if (tid == 0) scale[row] = sc;
#pragma unroll
  for (int mg = 0; mg < G; ++mg) {
    const int g = tid + mg * T;
    if (g >= NV) break;
    unsigned lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) lo |= macaq_i8_code(sv[mg][j], sc) << (j * 8);
#pragma unroll
    for (int j = 4; j < 8; ++j) hi |= macaq_i8_code(sv[mg][j], sc) << ((j - 4) * 8);
    ((uint2 *)q)[(mt * kst + (g >> 1)) * 32 + lrow + 16 * (g & 1)] = make_uint2(lo, hi);
  }
}

// ---- act quant (parent: dflash_act_quant_g128 / _recip; v = f32(x)) -------------------------------------------------
template <int T, int G>
__device__ __forceinline__ void dh_act_impl(const __bf16 *__restrict__ x, unsigned char *__restrict__ q,
                                            float *__restrict__ scale, int n, const float *__restrict__ had_t) {
  const int row = blockIdx.x, tid = threadIdx.x;
  x += (long)row * n;
  const int NV = n >> 3;
  float sv[G][8];
#pragma unroll
  for (int mg = 0; mg < G; ++mg) {
    const int g = tid + mg * T;
    if (g >= NV) break;
    DhVec8 v;
    v.u4 = ((const uint4 *)x)[g];
#pragma unroll
    for (int j = 0; j < 8; ++j) sv[mg][j] = (float)v.h[j];
    macaq_had32(sv[mg], had_t + 8 * g, tid);
  }
  dh_i8_tail<T, G>(sv, NV, q, scale, row, n >> 4);
}

// ---- grouped conv quant (parent: dflash_grouped_conv_quant_t2g16; v = bf16(conv)) -----------------------------------
// `dh_conv` is dflash_conv.hip's `dc_conv` VERBATIM (TAPS 2, GS 16, VEC 4), called twice for a thread's 8 elements:
// every element's taps are summed in the parent's order with the parent's expression, so the contraction is its own.
template <int TAPS, int GS, int VEC>
__device__ __forceinline__ void dh_conv(const __bf16 *__restrict__ x, const __bf16 *__restrict__ coeff,
                                        const __bf16 *__restrict__ base, int t, int h, int H, int groups,
                                        int block_size, int side, float *o) {
  const int g = h / GS;
  const int pos = t & (block_size - 1);
#pragma unroll
  for (int e = 0; e < VEC; ++e) o[e] = 0.f;
#pragma unroll
  for (int tap = 0; tap < TAPS; ++tap) {
    if (tap > 0 && pos < tap) continue;
    const bf16x4 xv = *(const bf16x4 *)(x + (long)(t - tap) * H + h);
    const bf16x4 bv = *(const bf16x4 *)(base + ((long)side * TAPS + tap) * H + h);
    const float d = (float)(coeff[(((long)t * 2 + side) * TAPS + tap) * groups + g]);
#pragma unroll
    for (int e = 0; e < VEC; ++e) o[e] += ((float)(bv[e]) + d) * (float)(xv[e]);
  }
}

template <int T, int G>
__device__ __forceinline__ void dh_conv_impl(const __bf16 *__restrict__ x, const __bf16 *__restrict__ coeff,
                                             const __bf16 *__restrict__ base, unsigned char *__restrict__ q,
                                             float *__restrict__ scale, int h, int groups, int block_size, int side,
                                             const float *__restrict__ had_t) {
  const int row = blockIdx.x, tid = threadIdx.x;
  const int NV = h >> 3;
  float sv[G][8];
#pragma unroll
  for (int mg = 0; mg < G; ++mg) {
    const int g = tid + mg * T;
    if (g >= NV) break;
    float o[8];
    dh_conv<2, 16, 4>(x, coeff, base, row, g * 8, h, groups, block_size, side, o);
    dh_conv<2, 16, 4>(x, coeff, base, row, g * 8 + 4, h, groups, block_size, side, o + 4);
#pragma unroll
    for (int j = 0; j < 8; ++j) sv[mg][j] = (float)(__bf16)o[j];
    macaq_had32(sv[mg], had_t + 8 * g, tid);
  }
  dh_i8_tail<T, G>(sv, NV, q, scale, row, h >> 4);
}

// ---- silu quant (parent: dflash_silu_quant_g128; v = silu(gate) * up in f32) ----------------------------------------
extern "C" __device__ float __ocml_exp_f32(float);
__device__ __forceinline__ float dh_silu(float x) { return x / (1.f + __ocml_exp_f32(-x)); }

template <int T, int G>
__device__ __forceinline__ void dh_silu_impl(const __bf16 *__restrict__ gate_up, unsigned char *__restrict__ q,
                                             float *__restrict__ scale, int n, const float *__restrict__ had_t) {
  const int row = blockIdx.x, tid = threadIdx.x;
  gate_up += (long)row * 2 * n;
  const int NV = n >> 3;
  float sv[G][8];
#pragma unroll
  for (int mg = 0; mg < G; ++mg) {
    const int g = tid + mg * T;
    if (g >= NV) break;
    DhVec8 gv, uv;
    gv.u4 = ((const uint4 *)gate_up)[g];
    uv.u4 = ((const uint4 *)(gate_up + n))[g];
#pragma unroll
    for (int j = 0; j < 8; ++j) sv[mg][j] = dh_silu((float)gv.h[j]) * (float)uv.h[j];
    macaq_had32(sv[mg], had_t + 8 * g, tid);
  }
  dh_i8_tail<T, G>(sv, NV, q, scale, row, n >> 4);
}

#define DH_ACT(T, G)                                                                                         \
  extern "C" __global__ __launch_bounds__(T) void dflash_hadi8_act_t##T##g##G(                               \
      const __bf16 *__restrict__ x, unsigned char *__restrict__ q, float *__restrict__ scales, int n,        \
      const float *__restrict__ had_t) {                                                                     \
    dh_act_impl<T, G>(x, q, scales, n, had_t);                                                               \
  }
#define DH_CONV(T, G)                                                                                        \
  extern "C" __global__ __launch_bounds__(T) void dflash_hadi8_conv_t##T##g##G(                              \
      const __bf16 *__restrict__ x, const __bf16 *__restrict__ coeff, const __bf16 *__restrict__ base,       \
      unsigned char *__restrict__ q, float *__restrict__ scales, int h, int groups, int block_size, int side, \
      const float *__restrict__ had_t) {                                                                     \
    dh_conv_impl<T, G>(x, coeff, base, q, scales, h, groups, block_size, side, had_t);                       \
  }
#define DH_SILU(T, G)                                                                                        \
  extern "C" __global__ __launch_bounds__(T) void dflash_hadi8_silu_t##T##g##G(                              \
      const __bf16 *__restrict__ gate_up, unsigned char *__restrict__ q, float *__restrict__ scales, int n,   \
      const float *__restrict__ had_t) {                                                                     \
    dh_silu_impl<T, G>(gate_up, q, scales, n, had_t);                                                        \
  }

// The cells (crate::dflash_iu8::cell): T 256 while ceil(K / 8 / 256) <= 5, else T 512.
//   act : K 2,048 (TP=2 o_proj) g1, 4,096 (TP=1 o_proj) g2, 25,600 (the aux projection) t512 g7
//   conv: K 5,120 g3          silu: N 8,704 (TP=2) g5, 17,408 (TP=1) t512 g5
DH_ACT(256, 1)
DH_ACT(256, 2)
DH_ACT(256, 3)
DH_ACT(256, 4)
DH_ACT(256, 5)
DH_ACT(512, 7)
DH_CONV(256, 3)
DH_SILU(256, 5)
DH_SILU(512, 5)
