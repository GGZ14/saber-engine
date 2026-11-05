// Saber source (CLEANROOM-1, 2026-10-25: frozen from a retired generator's output; edit this file directly)
//   reference sha256 79a7eb6f261c2939386a4c231cf5727a0a1f2325d8a391419178778da1fd2ae2
//   device code sliced VERBATIM by line range; only the __global__ signatures became
//   __device__ __forceinline__ *_impl and the extern "C" wrappers at the end are new.
// Do not edit by hand: re-run the generator. No host code, no pybind, no getenv in this file --
// every tile parameter the prod launcher chose from an env var is an explicit symbol here.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "mxfp4_common.h"

#define BM 128
#define BN 64
#define NWAVE 8

__device__ __forceinline__ void saber_mxfp4_fp8_gemm_impl(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const float *__restrict__ As,
    __bf16 *__restrict__ C, int M, int N, int K) {
  __shared__ unsigned char sA[BM * ASTR];
  __shared__ unsigned char sW[BN * WSTR];
  __shared__ unsigned char sS[BN * (BK / 32)];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave >> 1, wn = wave & 1;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int nblk = K / 32;

  floatx8 acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    {   // A tile: 32 B per thread. CLAMP, NEVER PREDICATE: a bounds-predicated staging load
        // lands in its own s_and_saveexec region and the compiler cannot carry a counted
        // s_wait_loadcnt across an exec-mask merge, so it emits s_wait_loadcnt 0x0 after every
        // load -- one full memory round trip each where one per slab would do. Clamped, all the
        // slab's loads issue back to back under a single wait. Safe because the epilogue already
        // drops rows past M and columns past N, and a clamped read returns real data. Measured on
        // the int4 twin (autoround-tests/RESULTS.md, 2026-08-26): worth 7-26% at decode and
        // 5-20% at prefill. Applies to every staging load below.
      const int r = tid >> 1, off = (tid & 1) * 32;
      const int rc = r < M - 1 - m0 ? r : M - 1 - m0;
      const unsigned char *src = A + (size_t)(m0 + rc) * K + k0 + off;
      const uint4_t v0 = *(const uint4_t *)(src);
      const uint4_t v1 = *(const uint4_t *)(src + 16);
      *(uint4_t *)(&sA[r * ASTR + off]) = v0;
      *(uint4_t *)(&sA[r * ASTR + off + 16]) = v1;
    }
    {   // W tile: 8 packed bytes per thread -> 16 e4m3 bytes, guarded on the N tail
      const int r = tid >> 2, off = (tid & 3) * 8;
      const int rc = r < N - 1 - n0 ? r : N - 1 - n0;              // clamp, never predicate
      const unsigned long long packed =
          *(const unsigned long long *)(W + ((size_t)(n0 + rc) * K + k0) / 2 + off);
      unsigned char out[16];
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        const unsigned char byte = (packed >> (8 * t)) & 0xFF;
        out[2 * t] = kLUT[byte & 0xF];
        out[2 * t + 1] = kLUT[(byte >> 4) & 0xF];
      }
      *(uint4_t *)(&sW[r * WSTR + off * 2]) = *(const uint4_t *)out;
    }
    if (tid < BN * (BK / 32)) {   // scales are [K/32, N]: coalesced across n
      const int r = tid / (BK / 32), b = tid % (BK / 32);
      const int gn = n0 + r;
      const int gc = gn < N - 1 ? gn : N - 1;                      // clamp, never predicate
      sS[tid] = Ws[(size_t)(k0 / 32 + b) * N + gc];
    }
    __syncthreads();

#pragma unroll
    for (int blk = 0; blk < BK / 32; ++blk) {
      floatx8 tmp[2][2];
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e) tmp[i][j][e] = 0.f;
#pragma unroll
      for (int step = 0; step < 2; ++step) {     // two WMMA steps span one 32-wide MX block
        const int kk = blk * 32 + step * 16 + kb8;
        int2_t af[2], wf[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const unsigned char *p = &sA[(wm * 32 + i * 16 + col) * ASTR + kk];
          af[i][0] = *(const int *)p; af[i][1] = *(const int *)(p + 4);
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const unsigned char *p = &sW[(wn * 32 + j * 16 + col) * WSTR + kk];
          wf[j][0] = *(const int *)p; wf[j][1] = *(const int *)(p + 4);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int j = 0; j < 2; ++j)
            tmp[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf[j], tmp[i][j]);
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        // E8M0 is a power of two by definition, so 2^(e-127) is exactly the IEEE-754 float
        // whose exponent field is e: bits = e << 23. One integer shift instead of a
        // transcendental, worth +4% (exp2f ran 4x per lane per K-slab). e=0 would mean 2^-127
        // (denormal) and yields 0.0f here; it only arises for all-zero blocks, where the
        // product is 0 either way.
        const float s = __int_as_float(
            (int)sS[(wn * 32 + j * 16 + col) * (BK / 32) + blk] << 23);
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[i][j][e] += tmp[i][j][e] * s;
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + wm * 32 + i * 16 + kb8 + e;
        const int n = n0 + wn * 32 + j * 16 + col;
        if (m < M && n < N) C[(size_t)m * N + n] = (__bf16)(acc[i][j][e] * As[m]);
      }
}


extern "C" __global__ __launch_bounds__(NWAVE * 32) void mxfp4_gemm_base(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const float *__restrict__ As,
    __bf16 *__restrict__ C, int M, int N, int K) {
  saber_mxfp4_fp8_gemm_impl(A, W, Ws, As, C, M, N, K);
}
