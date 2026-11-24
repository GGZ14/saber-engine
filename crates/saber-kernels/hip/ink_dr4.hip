// INK21: the batched drafter's GEMMs on the matrix cores (docs/INKLING2.md section 9 INK21). The
// batched pass (`ink_dr3`, INK20) runs B blocks of 8 rows through the drafter's three layers at
// M = 8 B (16..64 rows); at M > 8 those GEMMs ran on the target's f32 column-lane cell (one FMA a
// weight element a row, ~5 TFLOP/s at M 64). The drafter's numerics are FREE under the drawn-row
// walk (a draft only moves acceptance; the target's verify decides every committed token), so
// these bodies are gated on the MTP law + acceptance, not on bytes.
//
//   ink_dr4_mm_{w8,bf16}  C[m][n] (+)= sum_k A[m][k] W[n][k], A f32 [m, lda] rounded to bf16 at
//                         the load, W [n, k] bf16 or INK14's 8-bit record (int8 codes + one bf16
//                         scale a 32-k block; a code is exact in bf16, so a 32-k block is two
//                         16x16x16 bf16 WMMAs into a block accumulator, then x scale into the
//                         column's f32 sum -- every accumulator element of a lane is ONE column).
//                         Grid (ceil(n / 32), ks), 128 threads = 4 waves, wave w = rows
//                         16w..16w+15 (a wave past m returns: no LDS, no barrier), two 16-column
//                         fragments a wave. ks = 1 stores C; ks > 1 stores the K slice's partials
//                         `part[s][m][n]` and `ink_dr4_ksum` sums them in slice order.
//   ink_dr4_ksum          C[r][c] = (acc ? C : 0) + sum_s part[s][r][c], s ascending.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float floatx8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ floatx8 dr4_wmma(bf16x8 a, bf16x8 b, floatx8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

__device__ __forceinline__ float dr4_bf2f(unsigned short b) { return __uint_as_float(((unsigned)b) << 16); }

// 8 consecutive f32 -> bf16x8 (round to nearest even)
__device__ __forceinline__ bf16x8 dr4_a8(const float* __restrict__ p) {
  const float4 x = *(const float4*)p, y = *(const float4*)(p + 4);
  bf16x8 r;
  r[0] = (__bf16)x.x; r[1] = (__bf16)x.y; r[2] = (__bf16)x.z; r[3] = (__bf16)x.w;
  r[4] = (__bf16)y.x; r[5] = (__bf16)y.y; r[6] = (__bf16)y.z; r[7] = (__bf16)y.w;
  return r;
}

// 8 int8 codes -> bf16x8 (exact)
__device__ __forceinline__ bf16x8 dr4_w8(const signed char* __restrict__ p) {
  const int2 v = *(const int2*)p;
  bf16x8 r;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    r[i] = (__bf16)(float)(signed char)((v.x >> (8 * i)) & 0xff);
    r[4 + i] = (__bf16)(float)(signed char)((v.y >> (8 * i)) & 0xff);
  }
  return r;
}

template <bool W8>
__device__ __forceinline__ void dr4_mm(const float* __restrict__ a, int lda, const void* __restrict__ w,
                                       const unsigned short* __restrict__ ws, float* __restrict__ c, int ldc,
                                       int m, int n, int k, int acc, int ks, float* __restrict__ part) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int m0 = wave * 16;
  if (m0 >= m) return;
  const int h = lane >> 4;
  const int ar = min(m0 + (lane & 15), m - 1);
  const int n0 = blockIdx.x * 32;
  const int kl = k / ks, kb0 = blockIdx.y * kl;
  int col[2];
#pragma unroll
  for (int f = 0; f < 2; ++f) col[f] = min(n0 + f * 16 + (lane & 15), n - 1);
  const float* ap = a + (long)ar * lda + kb0 + h * 8;
  floatx8 s[2] = {};
  if (W8) {
    const signed char* wp[2];
    const unsigned short* sp[2];
#pragma unroll
    for (int f = 0; f < 2; ++f) {
      wp[f] = (const signed char*)w + (long)col[f] * k + kb0 + h * 8;
      sp[f] = ws + (long)col[f] * (k / 32) + kb0 / 32;
    }
#pragma unroll 2
    for (int kk = 0; kk < kl; kk += 32) {
      const bf16x8 a0 = dr4_a8(ap + kk), a1 = dr4_a8(ap + kk + 16);
#pragma unroll
      for (int f = 0; f < 2; ++f) {
        const bf16x8 b0 = dr4_w8(wp[f] + kk), b1 = dr4_w8(wp[f] + kk + 16);
        const float sc = dr4_bf2f(sp[f][kk / 32]);
        floatx8 t = {};
        t = dr4_wmma(a0, b0, t);
        t = dr4_wmma(a1, b1, t);
#pragma unroll
        for (int e = 0; e < 8; ++e) s[f][e] += sc * t[e];
      }
    }
  } else {
    const __bf16* wp[2];
#pragma unroll
    for (int f = 0; f < 2; ++f) wp[f] = (const __bf16*)w + (long)col[f] * k + kb0 + h * 8;
#pragma unroll 2
    for (int kk = 0; kk < kl; kk += 16) {
      const bf16x8 a0 = dr4_a8(ap + kk);
#pragma unroll
      for (int f = 0; f < 2; ++f) s[f] = dr4_wmma(a0, *(const bf16x8*)(wp[f] + kk), s[f]);
    }
  }
#pragma unroll
  for (int f = 0; f < 2; ++f) {
    const int cc = n0 + f * 16 + (lane & 15);
    if (cc >= n) continue;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int r = m0 + h * 8 + e;
      if (r >= m) continue;
      if (ks == 1) {
        float* o = c + (long)r * ldc + cc;
        *o = acc ? *o + s[f][e] : s[f][e];
      } else {
        part[((long)blockIdx.y * m + r) * n + cc] = s[f][e];
      }
    }
  }
}

extern "C" __global__ __launch_bounds__(128) void ink_dr4_mm_w8(const float* __restrict__ a, int lda,
                                                                const signed char* __restrict__ w,
                                                                const unsigned short* __restrict__ ws,
                                                                float* __restrict__ c, int ldc, int m, int n,
                                                                int k, int acc, int ks, float* __restrict__ part) {
  dr4_mm<true>(a, lda, w, ws, c, ldc, m, n, k, acc, ks, part);
}

extern "C" __global__ __launch_bounds__(128) void ink_dr4_mm_bf16(const float* __restrict__ a, int lda,
                                                                  const unsigned short* __restrict__ w,
                                                                  const unsigned short* __restrict__ ws,
                                                                  float* __restrict__ c, int ldc, int m, int n,
                                                                  int k, int acc, int ks, float* __restrict__ part) {
  dr4_mm<false>(a, lda, w, ws, c, ldc, m, n, k, acc, ks, part);
}

extern "C" __global__ __launch_bounds__(256) void ink_dr4_ksum(const float* __restrict__ part, int ks, int m,
                                                               int n, float* __restrict__ c, int ldc, int acc) {
  const long i = (long)blockIdx.x * 256 + threadIdx.x;
  if (i >= (long)m * n) return;
  const int r = (int)(i / n), cc = (int)(i % n);
  float v = 0.f;
  for (int s = 0; s < ks; ++s) v += part[((long)s * m + r) * n + cc];
  float* o = c + (long)r * ldc + cc;
  *o = acc ? *o + v : v;
}
