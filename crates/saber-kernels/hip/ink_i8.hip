// Inkling 2 (inkling2_loop) prefill dense trunk on the matrix cores, INK25 (docs/INKLING2.md
// section 9 INK25). Saber's own. An ARM (`--ink-prefill-trunk iu8`; `f32` = the byte-law column-lane
// / pf bodies = the default): the prefill rows' numerics change (int8 activations), the decode rows
// do not (the GEMVs never route here).
//
// THE OPERANDS
//   W   INK14's trunk8 record as served: int8 codes [N, ldw] row-major (round-half-even a 32-k block)
//       + bf16 scales [N, ldsc = K / 32] (the smallest bf16 >= amax / 127). Read in place: no copy.
//   qa  int8 activation codes [M, K] row-major and sa f32 [M, K / 32]: ONE scale a (row, 32-k block),
//       written by `ink_act_q8_b32` from the f32 activations: sa = amax / 127 (correctly rounded; 0 ->
//       1), code = clamp(rne(x / sa), -127, 127) (macaq_i8.h's law, `__fdiv_rn`).
//
// THE CONTRACT (the epilogue order; tools/capture/ink25/ink25_twin.py `ref` reproduces it in numpy f32)
//   for every output (m, n): f = 0.f; for b = 0 .. K/32 - 1 in order:
//       i = the exact int32 sum over the block's 32 k of qa[m, k] * qw[n, k]     (|i| <= 516,129 < 2^24)
//       s = sa[m, b] * f32(sw[n, b]);  p = f32(i) * s;  f = f + p               (each op rounded, no FMA)
//   then C[m, n] = acc ? C[m, n] + f : f.
// The k order INSIDE a block is not numerics (int32 adds are exact), so the two 16-k WMMAs of a block
// may take any k split: here lane half h of a row loads the block's bytes [16h, 16h + 16) and the
// first WMMA sums bytes {0..7, 16..23}, the second {8..15, 24..31}.
//
// THE BODY (`ink_gemm_i8_w8`): the WMMA's A operand = 16 WEIGHT rows (output columns n0..n0+15), B =
// 16 activation rows (tokens) -> lane l of the 16 x 16 result holds token mt*16 + l%16 and the eight
// output columns n0 + 8(l/16) + e, so a lane needs ONE activation scale a block (a float4 = 4 blocks)
// and eight weight scales (8 x 8-byte loads a 4-block group). A workgroup = one 16-column n-tile x WV
// 16-row m-tiles (one a wave, blockDim = 32 WV, WV <= 16: the waves read the same weight bytes at
// the same time -> the CU's caches serve all but the first); grid [ceil(N / 16), ceil(ceil(M / 16) /
// WV)]. 128 k a group, the next group's loads in registers while the current one is summed. Clamped
// rows / columns are read, never stored. Needs K % 128 == 0, ldw % 16 == 0, 16-byte aligned W and qa
// (the host checks; otherwise the f32 bodies run).
#include <hip/hip_runtime.h>

#include "macaq_i8.h"

typedef int i8v2 __attribute__((ext_vector_type(2)));
typedef int i8v4 __attribute__((ext_vector_type(4)));
typedef int i8v8 __attribute__((ext_vector_type(8)));
typedef unsigned u8v2 __attribute__((ext_vector_type(2)));
typedef float f8v4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ i8v8 i8_wmma(i8v2 a, i8v2 b, i8v8 c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
}

// f32 activations [M, lda] -> int8 codes [M, K] + f32 scales [M, K / 32]; one thread a (row, block).
extern "C" __global__ __launch_bounds__(256) void ink_act_q8_b32(const float* __restrict__ a, int lda,
                                                                 unsigned char* __restrict__ q,
                                                                 float* __restrict__ s, int M, int K) {
  const int nb = K >> 5;
  const long g = (long)blockIdx.x * 256 + threadIdx.x;
  if (g >= (long)M * nb) return;
  const int m = (int)(g / nb), b = (int)(g % nb);
  const f8v4* p = (const f8v4*)(a + (long)m * lda + b * 32);
  f8v4 v[8];
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    v[i] = p[i];
#pragma unroll
    for (int j = 0; j < 4; ++j) amax = fmaxf(amax, fabsf(v[i][j]));
  }
  float sc = __fdiv_rn(amax, 127.f);
  if (!(sc > 0.f)) sc = 1.f;
  unsigned w[8];
#pragma unroll
  for (int i = 0; i < 8; ++i)
    w[i] = macaq_i8_code(v[i][0], sc) | (macaq_i8_code(v[i][1], sc) << 8) |
           (macaq_i8_code(v[i][2], sc) << 16) | (macaq_i8_code(v[i][3], sc) << 24);
  i8v4* o = (i8v4*)(q + (long)m * K + b * 32);
  o[0] = i8v4{(int)w[0], (int)w[1], (int)w[2], (int)w[3]};
  o[1] = i8v4{(int)w[4], (int)w[5], (int)w[6], (int)w[7]};
  s[(long)m * nb + b] = sc;
}

// the same law, 4 lanes a (row, block) (8 values a lane, the amax over the 4 lanes by xor shuffles:
// max is exact in any order) -> the same bytes as ink_act_q8_b32.
extern "C" __global__ __launch_bounds__(256) void ink_act_q8_b32_v2(const float* __restrict__ a, int lda,
                                                                    unsigned char* __restrict__ q,
                                                                    float* __restrict__ s, int M, int K) {
  const int nb = K >> 5;
  const long g = (long)blockIdx.x * 256 + threadIdx.x;
  const long blk = g >> 2;
  const int part = threadIdx.x & 3;
  const bool live = blk < (long)M * nb;
  const long bl = live ? blk : 0;
  const int m = (int)(bl / nb), b = (int)(bl % nb);
  const f8v4* p = (const f8v4*)(a + (long)m * lda + b * 32 + part * 8);
  const f8v4 v0 = p[0], v1 = p[1];
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) amax = fmaxf(amax, fmaxf(fabsf(v0[j]), fabsf(v1[j])));
  amax = fmaxf(amax, __shfl_xor(amax, 1));
  amax = fmaxf(amax, __shfl_xor(amax, 2));
  float sc = __fdiv_rn(amax, 127.f);
  if (!(sc > 0.f)) sc = 1.f;
  if (!live) return;
  const unsigned w0 = macaq_i8_code(v0[0], sc) | (macaq_i8_code(v0[1], sc) << 8) |
                      (macaq_i8_code(v0[2], sc) << 16) | (macaq_i8_code(v0[3], sc) << 24);
  const unsigned w1 = macaq_i8_code(v1[0], sc) | (macaq_i8_code(v1[1], sc) << 8) |
                      (macaq_i8_code(v1[2], sc) << 16) | (macaq_i8_code(v1[3], sc) << 24);
  *(u8v2*)(q + (long)m * K + b * 32 + part * 8) = u8v2{w0, w1};
  if (part == 0) s[(long)m * nb + b] = sc;
}

struct I8Grp {
  i8v4 w[4], a[4];
  f8v4 sa;
  u8v2 sw[8];
};

__device__ __forceinline__ void i8_load(I8Grp& g, const unsigned char* __restrict__ wp,
                                        const unsigned char* __restrict__ ap,
                                        const float* __restrict__ sap,
                                        const unsigned short* const* scp, int grp) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    g.w[j] = *(const i8v4*)(wp + grp * 128 + 32 * j);
    g.a[j] = *(const i8v4*)(ap + grp * 128 + 32 * j);
  }
  g.sa = *(const f8v4*)(sap + grp * 4);
#pragma unroll
  for (int e = 0; e < 8; ++e) g.sw[e] = *(const u8v2*)(scp[e] + grp * 4);
}

__device__ __forceinline__ void i8_sum(const I8Grp& g, float (&f)[8]) {
#pragma clang fp contract(off)
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    i8v8 z = {0, 0, 0, 0, 0, 0, 0, 0};
    z = i8_wmma(g.w[j].xy, g.a[j].xy, z);
    z = i8_wmma(g.w[j].zw, g.a[j].zw, z);
    const float sa = g.sa[j];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const unsigned d = g.sw[e][j >> 1];
      const float sw = __builtin_bit_cast(float, (j & 1) ? (d & 0xFFFF0000u) : (d << 16));
      const float s = sa * sw;
      const float p = (float)z[e] * s;
      f[e] = f[e] + p;
    }
  }
}

// PF groups of loads in flight (a ring of PF + 1 register groups; needs (K / 128) % (PF + 1) == 0 for
// PF > 1). The ring depth is not numerics. INK25 measured: PF 2 = 256 VGPRs + 188 B scratch -> only PF 1
// is instantiated.
template <int PF>
__device__ __forceinline__ void i8_body(const unsigned char* __restrict__ qa, const float* __restrict__ sa,
                                        const unsigned char* __restrict__ w, int ldw,
                                        const unsigned short* __restrict__ sc, int ldsc,
                                        float* __restrict__ c, int ldc, int M, int N, int K, int acc) {
#pragma clang fp contract(off)
  const int lane = threadIdx.x & 31;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const int mt = blockIdx.y * (blockDim.x >> 5) + wave;
  if (mt * 16 >= M) return;
  const int n0 = blockIdx.x * 16;
  const int l16 = lane & 15, h = lane >> 4;
  const int nb = K >> 5;
  const int m = min(mt * 16 + l16, M - 1);
  const unsigned char* ap = qa + (long)m * K + 16 * h;
  const unsigned char* wp = w + (long)min(n0 + l16, N - 1) * ldw + 16 * h;
  const float* sap = sa + (long)m * nb;
  const unsigned short* scp[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) scp[e] = sc + (long)min(n0 + 8 * h + e, N - 1) * ldsc;
  float f[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) f[e] = 0.f;
  const int G = K >> 7;
  if constexpr (PF == 1) {
    I8Grp cur, nxt;
    i8_load(cur, wp, ap, sap, scp, 0);
    for (int grp = 0; grp < G; ++grp) {
      if (grp + 1 < G) i8_load(nxt, wp, ap, sap, scp, grp + 1);
      i8_sum(cur, f);
      cur = nxt;
    }
  } else {
    constexpr int R = PF + 1;
    I8Grp buf[R];
#pragma unroll
    for (int r = 0; r < PF; ++r) i8_load(buf[r], wp, ap, sap, scp, r);
    for (int g0 = 0; g0 < G; g0 += R) {
#pragma unroll
      for (int r = 0; r < R; ++r) {
        if (g0 + r + PF < G) i8_load(buf[(r + PF) % R], wp, ap, sap, scp, g0 + r + PF);
        i8_sum(buf[r], f);
      }
    }
  }
  const int mo = mt * 16 + l16;
  if (mo >= M) return;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int n = n0 + 8 * h + e;
    if (n < N) {
      float* o = c + (long)mo * ldc + n;
      *o = acc ? *o + f[e] : f[e];
    }
  }
}

#define I8_KERNEL(NAME, PF)                                                                            \
  extern "C" __global__ __launch_bounds__(512) void NAME(                                              \
      const unsigned char* __restrict__ qa, const float* __restrict__ sa,                             \
      const unsigned char* __restrict__ w, int ldw, const unsigned short* __restrict__ sc, int ldsc,  \
      float* __restrict__ c, int ldc, int M, int N, int K, int acc) {                                  \
    i8_body<PF>(qa, sa, w, ldw, sc, ldsc, c, ldc, M, N, K, acc);                                       \
  }
I8_KERNEL(ink_gemm_i8_w8, 1)
