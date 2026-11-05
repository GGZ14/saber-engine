// FN61e: the fp8 W8 dense trunk's MULTI-ROW tiles, its own code object so `dflash_gemm` and
// `flashnext_trunk8` (FN61c) keep their bytes. Every symbol here computes, for each output element,
// EXACTLY the parent's arithmetic (`dflash_gemm_a8w8_bs128_mtn128k128` / FN61c's ldc twin; for the
// bf16 cell `fn_t8_bgemm_m16_ldc`): per 128-deep k tile a fresh fp32 WMMA chain over the eight
// 16-deep k steps in order (A = the activation rows, B = the weight columns, the parent's operand
// order), then `acc += tile * (a_scale * b_scale)`, one round to bf16 at the store. What changes is
// only WHO computes which element and how the operands reach the WMMA:
//
//   * the parent re-reads the whole weight tile for every 16 rows (blockIdx.y = 16-row slices), so a
//     24-row verify step reads the weights twice and a 2,048-row prefill chunk 128 times (FN61b/d);
//   * here one workgroup owns BM = 16 * MF rows x BN = 16 * NW columns: wave w owns ONE 16-column
//     weight fragment and every one of the MF row fragments, so the preshuffled weight fragment goes
//     from global STRAIGHT into the WMMA operand registers once per workgroup (non-temporal on the
//     decode tiles: a once-read stream) and is reused MF times; the activation rows + their per-128
//     scales go through LDS (double-buffered, one LDS-only barrier a k tile: `__syncthreads` on
//     gfx12 drains the register prefetch -- FN31's law), staged with CLAMPED rows (no predicated
//     staging -- gfx1201 serializes it) one k tile ahead.
//
// Symbols (signature = FN61c's `fn_t8_a8w8_bs128_ldc(a, w, as, ws, c, m, n, k, ldc)`):
//   fn_t8p_a8w8_d{MF}w{NW}_ldc   decode / verify tiles (17..64 rows: one row tile, grid N / BN)
//   fn_t8p_a8w8_p{MF}w{NW}_ldc   prefill tiles (grid x = row tiles FASTEST so the workgroups in
//                                flight share a weight tile through L2; y = column tiles)
//   fn_t8p_bgemm_m{16*MF}_ldc    the bf16 suffix (in_proj_ba / the indexer) at MF row fragments a
//                                wave: FN61c's one-split chain, the weight fragment reused MF times
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

typedef float floatx8 __attribute__((ext_vector_type(8)));
typedef unsigned int uint2_t __attribute__((ext_vector_type(2)));
typedef unsigned int uint4_t __attribute__((ext_vector_type(4)));
typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));

#define P_BK 128
#define P_LDA (P_BK + 8)

__device__ __forceinline__ floatx8 p_wmma(uint2_t a, uint2_t b, floatx8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}

static __device__ __forceinline__ void p_lds_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

template <bool NT>
__device__ __forceinline__ uint2_t p_ldb(const unsigned char *p) {
  if constexpr (NT) return __builtin_nontemporal_load((const uint2_t *)p);
  else return *(const uint2_t *)p;
}

template <int MF, int MH, int NW, bool MFAST, bool NT>
__device__ __forceinline__ void t8p_impl(
    const unsigned char *__restrict__ A,   // [M, K] e4m3, row-major
    const unsigned char *__restrict__ W,   // [N/16, K*16] e4m3, preshuffled
    const float *__restrict__ As,          // [M, K/128] fp32
    const float *__restrict__ Ws,          // [N/128, K/128] fp32
    __bf16 *__restrict__ C,                // [M, ldc] bf16 (columns 0..N)
    int M, int N, int K, long LDC) {
  constexpr int BM = 16 * MF;
  constexpr int NTH = NW * 32;
  constexpr int KSTEP = P_BK / 16;
  constexpr int ACH = BM * (P_BK / 16) / NTH;   // uint4 A chunks a thread a k tile
  static_assert(ACH * NTH == BM * (P_BK / 16), "A staging divides");
  static_assert(NTH >= BM, "one scale a thread");
  __shared__ unsigned char sA[2][BM * P_LDA];
  __shared__ float sS[2][BM];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int mt = MFAST ? (int)blockIdx.x : (int)blockIdx.y;
  const int nt = MFAST ? (int)blockIdx.y : (int)blockIdx.x;
  const int m0 = mt * BM;
  const int nf = nt * NW * 16 + wave * 16;   // this wave's 16 columns
  const int kb = K / P_BK;

  floatx8 acc[MF];
#pragma unroll
  for (int i = 0; i < MF; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;

  const unsigned char *bbase =
      W + (long)(nf / 16) * K * 16 + (long)(lane & 15) * 16 + (long)(lane >> 4) * 8;
  const float *wsrow = Ws + (long)(nf / 128) * kb;
  uint2_t b0[KSTEP], b1[KSTEP];
  uint4_t abuf[ACH];
  float sbuf = 0.f;
  const int srow = min(m0 + (tid < BM ? tid : 0), M - 1);

  auto fetch_b = [&](uint2_t (&dst)[KSTEP], int kt) {
#pragma unroll
    for (int ks = 0; ks < KSTEP; ++ks)
      dst[ks] = p_ldb<NT>(bbase + ((long)kt * KSTEP + ks) * 256);
  };
  auto fetch_a = [&](int kt) {
#pragma unroll
    for (int i = 0; i < ACH; ++i) {
      const int off = (tid + i * NTH) * 16;
      const int r = off / P_BK, c = off % P_BK;
      const int rc = min(m0 + r, M - 1);
      abuf[i] = *(const uint4_t *)(A + (long)rc * K + kt * P_BK + c);
    }
    sbuf = As[(long)srow * kb + kt];
  };
  auto put_a = [&](int buf) {
#pragma unroll
    for (int i = 0; i < ACH; ++i) {
      const int off = (tid + i * NTH) * 16;
      *(uint4_t *)(&sA[buf][(off / P_BK) * P_LDA + off % P_BK]) = abuf[i];
    }
    if (tid < BM) sS[buf][tid] = sbuf;
  };
  auto step = [&](int kt, uint2_t (&cur)[KSTEP], uint2_t (&nxt)[KSTEP]) {
    const int buf = kt & 1;
    const bool more = kt + 1 < kb;
    if (more) {
      fetch_a(kt + 1);
      fetch_b(nxt, kt + 1);
    }
    const float bs = wsrow[kt];
    // MH row fragments a pass (the register bound): each fragment's chain is the parent's
#pragma unroll
    for (int h = 0; h < MF; h += MH) {
      floatx8 tile[MH];
#pragma unroll
      for (int i = 0; i < MH; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) tile[i][e] = 0.f;
#pragma unroll
      for (int ks = 0; ks < KSTEP; ++ks)
#pragma unroll
        for (int i = 0; i < MH; ++i) {
          const uint2_t av = *(const uint2_t *)(
              &sA[buf][((h + i) * 16 + (lane & 15)) * P_LDA + ks * 16 + (lane >> 4) * 8]);
          tile[i] = p_wmma(av, cur[ks], tile[i]);
        }
#pragma unroll
      for (int i = 0; i < MH; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float as = sS[buf][(h + i) * 16 + (lane >> 4) * 8 + e];
          acc[h + i][e] += tile[i][e] * (as * bs);
        }
    }
    if (more) put_a(buf ^ 1);
    p_lds_barrier();
  };

  fetch_a(0);
  fetch_b(b0, 0);
  put_a(0);
  __syncthreads();
  int kt = 0;
  for (; kt + 1 < kb; kt += 2) {
    step(kt, b0, b1);
    step(kt + 1, b1, b0);
  }
  if (kt < kb) step(kt, b0, b1);

  const int col = nf + (lane & 15);
#pragma unroll
  for (int i = 0; i < MF; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int row = m0 + i * 16 + (lane >> 4) * 8 + e;
      if (row < M) C[(long)row * LDC + col] = (__bf16)acc[i][e];
    }
}

#define T8P_SIG                                                                                 \
  (const unsigned char *__restrict__ a, const unsigned char *__restrict__ w,                    \
   const float *__restrict__ as, const float *__restrict__ ws, __bf16 *__restrict__ c, int m,   \
   int n, int k, int ldc)
#define T8P_DEF(NAME, MF, MH, NW, MFAST, NT)                                                    \
  extern "C" __global__ __launch_bounds__(NW * 32) void NAME T8P_SIG {                          \
    t8p_impl<MF, MH, NW, MFAST, NT>(a, w, as, ws, c, m, n, k, (long)ldc);                       \
  }

// decode / verify (one row tile: grid N / BN; the weights a once-read stream)
T8P_DEF(fn_t8p_a8w8_d2w2_ldc, 2, 2, 2, false, true)
T8P_DEF(fn_t8p_a8w8_d4w2_ldc, 4, 2, 2, false, true)
T8P_DEF(fn_t8p_a8w8_d2w4_ldc, 2, 2, 4, false, true)
T8P_DEF(fn_t8p_a8w8_d4w4_ldc, 4, 2, 4, false, true)
// prefill (row tiles fastest; the weight tile shared through L2 by the workgroups in flight)
// (MF 8 = 128-row tiles spill on gfx1201 -- 596-660 B of scratch at 256 VGPRs, measured at build:
// not shipped; the 64-row tile is the prefill cell)
T8P_DEF(fn_t8p_a8w8_p4w8_ldc, 4, 4, 8, true, false)
T8P_DEF(fn_t8p_a8w8_p4w4_ldc, 4, 4, 4, true, false)

// ============================================================ bf16 suffix, one split, MF rows
__device__ __forceinline__ floatx8 pb_wmma(bf16x8 a, bf16x8 b, floatx8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}
// FN61c's `fn_t8_bgemm_m16_ldc` (FN5's `fn_bgemm_sk_m16` at one split + ldc) with MF row fragments
// a wave: each output element's chain is k steps 0..K/16 in order from 0 (the parent's), the weight
// fragment loaded once and used MF times. grid (cdiv(n, 64), cdiv(m, 16 * MF)), 128 threads.
template <int MF>
__device__ __forceinline__ void t8p_bgemm_impl(const __bf16 *__restrict__ a,
                                               const __bf16 *__restrict__ w,
                                               __bf16 *__restrict__ c, int m, int n, int k,
                                               int ldc) {
  constexpr int NWB = 4, UNR = 4;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int n0 = ((int)blockIdx.x * NWB + wave) * 16;
  const int m0 = (int)blockIdx.y * 16 * MF;
  const int row = lane & 15, kh = (lane >> 4) * 8;
  const bool live = n0 < n;
  const __bf16 *ap[MF];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf) ap[mf] = a + (long)min(m0 + mf * 16 + row, m - 1) * k + kh;
  const __bf16 *wp = w + (long)min(n0 + row, n - 1) * k + kh;
  floatx8 acc[MF];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[mf][e] = 0.f;
  const int ksteps = live ? k / 16 : 0;
  int ks = 0;
  for (; ks + UNR <= ksteps; ks += UNR) {
    bf16x8 af[MF][UNR], wf[UNR];
#pragma unroll
    for (int u = 0; u < UNR; ++u) {
#pragma unroll
      for (int mf = 0; mf < MF; ++mf) af[mf][u] = *(const bf16x8 *)(ap[mf] + (ks + u) * 16);
      wf[u] = *(const bf16x8 *)(wp + (ks + u) * 16);
    }
#pragma unroll
    for (int u = 0; u < UNR; ++u)
#pragma unroll
      for (int mf = 0; mf < MF; ++mf) acc[mf] = pb_wmma(af[mf][u], wf[u], acc[mf]);
  }
  for (; ks < ksteps; ++ks) {
    const bf16x8 wv = *(const bf16x8 *)(wp + ks * 16);
#pragma unroll
    for (int mf = 0; mf < MF; ++mf)
      acc[mf] = pb_wmma(*(const bf16x8 *)(ap[mf] + ks * 16), wv, acc[mf]);
  }
  const int col = n0 + (lane & 15);
  if (!live || col >= n) return;
#pragma unroll
  for (int mf = 0; mf < MF; ++mf)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int r = m0 + mf * 16 + (lane >> 4) * 8 + e;
      if (r < m) c[(long)r * ldc + col] = (__bf16)acc[mf][e];
    }
}
extern "C" __global__ __launch_bounds__(128) void fn_t8p_bgemm_m64_ldc(
    const __bf16 *__restrict__ a, const __bf16 *__restrict__ w, __bf16 *__restrict__ c, int m,
    int n, int k, int ldc) {
  t8p_bgemm_impl<4>(a, w, c, m, n, k, ldc);
}
