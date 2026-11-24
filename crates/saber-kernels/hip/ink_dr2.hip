// INK18: the Inkling 2 block drafter's own bodies (docs/INKLING2.md section 9 INK18). The drafter's
// numerics are FREE under the drawn-row walk: a draft only moves acceptance, never a committed
// token (the target's verify decides every one), so these bodies are not held to the target's
// byte law -- they are gated on the MTP law (committed ids == the non-speculative stream) and on
// acceptance (tokens a verify).
//
//   ink_dr_argmax_map   ink_dr_argmax over a pruned vocabulary's partials, the winner mapped to
//                       its token id (*tok = map[best row]; first row on ties)
//   ink_dr_gemv_sk_{bf16,w8}
//                       C[m][n] (+)= sum_k A[m][k] W[n][k] for M <= 8 rows, split over K: grid
//                       (ceil(N / (32 np)), ks); workgroup (x, s) sums its 32 columns over the K slice
//                       s (A's slice staged in LDS), writes its partials, and the LAST workgroup of
//                       a column block to arrive (a counter by value, reset by it) sums the ks
//                       partials in slice order and stores C. W8 = int8 codes + one bf16 scale a
//                       32-k block (INK14's record), code x scale exact in f32.
#include <hip/hip_runtime.h>

#define SK_MAXKL 768

__device__ __forceinline__ float bf2f(unsigned short b) { return __uint_as_float(((unsigned)b) << 16); }

extern "C" __global__ __launch_bounds__(256) void ink_dr_argmax_map(const float* __restrict__ part, int n,
                                                                    const unsigned* __restrict__ map,
                                                                    int* __restrict__ tok) {
  __shared__ float bv[256];
  __shared__ int bi[256];
  const int tid = threadIdx.x;
  float b = -INFINITY;
  int ib = 0x7fffffff;
  for (int i = tid; i < n; i += 256) {
    const float v = part[2 * i];
    const int k = __float_as_int(part[2 * i + 1]);
    if (v > b || (v == b && k < ib)) {
      b = v;
      ib = k;
    }
  }
  bv[tid] = b;
  bi[tid] = ib;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (tid < s) {
      const float v = bv[tid + s];
      const int k = bi[tid + s];
      if (v > bv[tid] || (v == bv[tid] && k < bi[tid])) {
        bv[tid] = v;
        bi[tid] = k;
      }
    }
    __syncthreads();
  }
  if (tid == 0) *tok = (int)map[bi[0]];
}

template <bool W8>
struct SkW;
template <>
struct SkW<false> {
  uint4 q;
};
template <>
struct SkW<true> {
  uint2 q;
  float s;
};

template <bool W8>
__device__ __forceinline__ void sk_fetch(const void* __restrict__ w, const unsigned short* __restrict__ ws,
                                         long row, int k, int kdim, SkW<W8>& r) {
  if constexpr (W8) {
    r.q = *(const uint2*)((const signed char*)w + row * kdim + k);
    r.s = bf2f(ws[row * (kdim / 32) + k / 32]);
  } else {
    r.q = *(const uint4*)((const unsigned short*)w + row * kdim + k);
  }
}

template <bool W8>
__device__ __forceinline__ void sk_unpack(const SkW<W8>& r, float* f) {
  if constexpr (W8) {
    const unsigned u[2] = {r.q.x, r.q.y};
#pragma unroll
    for (int j = 0; j < 8; ++j) f[j] = (float)(signed char)((u[j >> 2] >> (8 * (j & 3))) & 0xff) * r.s;
  } else {
    const unsigned u[4] = {r.q.x, r.q.y, r.q.z, r.q.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) f[j] = __uint_as_float((j & 1) ? (u[j >> 1] & 0xffff0000u) : (u[j >> 1] << 16));
  }
}

// The lane's weight loads (4 columns x (K slice / 256) steps) are ISSUED before A's slice is staged;
// a workgroup walks np passes of 32 columns over the ONE staged slice (32 x np columns a
// workgroup), the next pass's loads issued before the current pass is summed (two register sets).
template <bool W8>
__device__ __forceinline__ void sk_fetch_pass(const void* __restrict__ w, const unsigned short* __restrict__ ws,
                                              int c0, int n, int k0, int lane, int k, int ni,
                                              SkW<W8> (&wr)[SK_MAXKL / 256][4]) {
#pragma unroll
  for (int it = 0; it < SK_MAXKL / 256; ++it) {
    if (it >= ni) break;
#pragma unroll
    for (int cc = 0; cc < 4; ++cc) sk_fetch<W8>(w, ws, min(c0 + cc, n - 1), k0 + it * 256 + lane * 8, k, wr[it][cc]);
  }
}

// sum one pass; lane j < 32 of the wave ends holding output (column c0 + j / 8, row j % 8)
template <bool W8>
__device__ __forceinline__ float sk_sum_pass(const float* la, int kl, int ni, int m, int lane,
                                             const SkW<W8> (&wr)[SK_MAXKL / 256][4]) {
  float av[4][8];
#pragma unroll
  for (int cc = 0; cc < 4; ++cc)
#pragma unroll
    for (int r = 0; r < 8; ++r) av[cc][r] = 0.f;
#pragma unroll
  for (int it = 0; it < SK_MAXKL / 256; ++it) {
    if (it >= ni) break;
    const int kk = it * 256 + lane * 8;
    float wf[4][8];
#pragma unroll
    for (int cc = 0; cc < 4; ++cc) sk_unpack<W8>(wr[it][cc], wf[cc]);
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      if (r >= m) break;
      const float4 x0 = *(const float4*)&la[r * kl + kk];
      const float4 x1 = *(const float4*)&la[r * kl + kk + 4];
      const float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
      for (int cc = 0; cc < 4; ++cc) {
        float d = av[cc][r];
#pragma unroll
        for (int j = 0; j < 8; ++j) d = fmaf(wf[cc][j], xv[j], d);
        av[cc][r] = d;
      }
    }
  }
  float mine = 0.f;
#pragma unroll
  for (int cc = 0; cc < 4; ++cc)
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      float v = av[cc][r];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
      if (lane == cc * 8 + r) mine = v;
    }
  return mine;
}

template <bool W8>
__device__ __forceinline__ void dr_gemv_sk(const float* __restrict__ a, int lda, const void* __restrict__ w,
                                           const unsigned short* __restrict__ ws, float* __restrict__ c,
                                           int ldc, int m, int n, int k, int acc, int ks, int np,
                                           float* __restrict__ part, unsigned* __restrict__ ctr) {
  __shared__ float la[8 * SK_MAXKL];
  __shared__ unsigned last;
  const int kl = k / ks, s = blockIdx.y, k0 = s * kl, ni = kl / 256;
  const int tid = threadIdx.x, lane = tid & 31, wv = tid >> 5;
  const int cb = 32 * np, cw = blockIdx.x * cb + wv * 4;
  SkW<W8> wa[SK_MAXKL / 256][4], wb[SK_MAXKL / 256][4];
  sk_fetch_pass<W8>(w, ws, cw, n, k0, lane, k, ni, wa);
  for (int i = tid; i < m * (kl / 4); i += 256) {
    const int r = i / (kl / 4), j = (i - r * (kl / 4)) * 4;
    *(float4*)&la[r * kl + j] = *(const float4*)&a[(long)r * lda + k0 + j];
  }
  __syncthreads();
  const int oc = lane >> 3, orr = lane & 7;
  auto put = [&](int p, float v) {
    const int col = cw + p * 32 + oc;
    if (lane < 32 && orr < m && col < n) {
      if (ks == 1) {
        float* o = c + (long)orr * ldc + col;
        *o = acc ? *o + v : v;
      } else {
        part[((long)s * m + orr) * n + col] = v;
      }
    }
  };
  for (int p = 0; p < np; p += 2) {
    if (p + 1 < np) sk_fetch_pass<W8>(w, ws, cw + (p + 1) * 32, n, k0, lane, k, ni, wb);
    put(p, sk_sum_pass<W8>(la, kl, ni, m, lane, wa));
    if (p + 2 < np) sk_fetch_pass<W8>(w, ws, cw + (p + 2) * 32, n, k0, lane, k, ni, wa);
    if (p + 1 < np) put(p + 1, sk_sum_pass<W8>(la, kl, ni, m, lane, wb));
  }
  if (ks == 1) return;
  __threadfence();
  __syncthreads();
  if (tid == 0) last = atomicAdd(&ctr[blockIdx.x], 1u);
  __syncthreads();
  if (last != (unsigned)(ks - 1)) return;
  __threadfence();
  for (int i = tid; i < m * cb; i += 256) {
    const int r = i / cb, col = blockIdx.x * cb + (i - r * cb);
    if (col < n) {
      float v = 0.f;
      for (int q = 0; q < ks; ++q)
        v += __hip_atomic_load(&part[((long)q * m + r) * n + col], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      float* o = c + (long)r * ldc + col;
      *o = acc ? *o + v : v;
    }
  }
  if (tid == 0) ctr[blockIdx.x] = 0u;
}

extern "C" __global__ __launch_bounds__(256) void ink_dr_gemv_sk_bf16(
    const float* __restrict__ a, int lda, const unsigned short* __restrict__ w, const unsigned short* __restrict__ ws,
    float* __restrict__ c, int ldc, int m, int n, int k, int acc, int ks, int np, float* __restrict__ part,
    unsigned* __restrict__ ctr) {
  dr_gemv_sk<false>(a, lda, w, ws, c, ldc, m, n, k, acc, ks, np, part, ctr);
}

extern "C" __global__ __launch_bounds__(256) void ink_dr_gemv_sk_w8(
    const float* __restrict__ a, int lda, const signed char* __restrict__ w, const unsigned short* __restrict__ ws,
    float* __restrict__ c, int ldc, int m, int n, int k, int acc, int ks, int np, float* __restrict__ part,
    unsigned* __restrict__ ctr) {
  dr_gemv_sk<true>(a, lda, w, ws, c, ldc, m, n, k, acc, ks, np, part, ctr);
}
