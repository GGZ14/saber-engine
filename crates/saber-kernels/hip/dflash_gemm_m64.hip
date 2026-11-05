// CMP27K: the DFlash2 drafter's fp8 W8A8 blockscale GEMM over 64 ROWS IN ONE WEIGHT PASS.
//
// THE PROBLEM (LEDGER CMP27J): at a b8 proposal (8 sequences x the block of 8 = 64 rows) the drafter's
// gate_up runs `dflash_gemm_a8w8_bs128_mtn128k128`, which gives each 16-row m-tile its OWN workgroup
// over the whole weight slab -- four workgroups read every weight byte, gate_up 145 -> 317 us a layer.
//
// THE TWIN: one workgroup owns a 64 x 128 tile of C. Its A tile (64 x 128 e4m3 a k tile) goes through
// LDS as before; each wave's preshuffled B fragments go from global straight into registers ONCE and
// feed FOUR m-fragments (four sets of accumulators) before the next k tile. ARITHMETIC UNCHANGED,
// element for element, against the m-tiled kernel (`dflash_gemm.hip`):
//   * each output's per-k-tile dot is the same eight WMMA k-steps in ascending order into a tile
//     accumulator that starts at zero (the 16x16x16 fp8 WMMA on the same A row fragment and the same
//     B fragment bytes);
//   * the same `acc += tile * (a_scale * b_scale)` after every k tile, in k order, from zero;
//   * the same single round to bf16 at the store, the same A-row clamp (rows >= M read row M-1 with
//     a_scale 0, never stored).
// What moves is only which workgroup computes which rows -- the head_m64 precedent (CMP27F). The
// gate (`tools/capture/cmp27k/gateup_probe.py`) compares bytes against the served m-tiled kernel:
// 0 bytes differing at M 1/8/16/24/32/40/48/56/64 x 3 seeds x the pack's five gate_up layers, the
// one-byte control fires every time; 311.7 -> 159.7 us a layer at M 64 (CMP27K p1, card 1, IC-cold).
//
// Built with the dflash default contraction (no `-ffp-contract=off`, like `dflash_gemm`), and the
// scale-accumulate line is written exactly as there, so it contracts the same way.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

typedef float floatx8 __attribute__((ext_vector_type(8)));
typedef unsigned int uint2_t __attribute__((ext_vector_type(2)));
typedef unsigned int uint4_t __attribute__((ext_vector_type(4)));

#define DG64_BN 128
#define DG64_BK 128
#define DG64_MF 4                  // m-fragments of 16 a workgroup: 64 rows
#define DG64_BM (DG64_MF * 16)

__device__ __forceinline__ floatx8 dg64_wmma(uint2_t a, uint2_t b, floatx8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}

// The gfx12 w32 accumulator: element e of lane l is row (l >> 4) * 8 + e, column l & 15.
__device__ __forceinline__ int dg64_acc_row(int lane, int e) { return (lane >> 4) * 8 + e; }

__device__ __forceinline__ void dg64_store_acc(const floatx8 &acc, __bf16 *__restrict__ c, int n0,
                                               int m0, int M, int N, long ldc, int lane) {
  const int col = n0 + (lane & 15);
  if (col >= N) return;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int row = m0 + dg64_acc_row(lane, e);
    if (row < M) c[(long)row * ldc + col] = (__bf16)acc[e];
  }
}

template <int NW>
__device__ __forceinline__ void dflash_gemm_m64_impl(
    const unsigned char *__restrict__ A,   // [M, K] e4m3, row-major
    const unsigned char *__restrict__ W,   // [N/16, K*16] e4m3, preshuffled
    const float *__restrict__ As,          // [M, K/128] fp32 block scales
    const float *__restrict__ Ws,          // [N/128, K/128] fp32 block scales
    __bf16 *__restrict__ C,                // [M, N] bf16
    int M, int N, int K) {
  constexpr int BM = DG64_BM, BN = DG64_BN, BK = DG64_BK, MF = DG64_MF;
  constexpr int NT = NW * 32;
  constexpr int NFRAG = BN / 16;          // 8 n-fragments of 16
  constexpr int KSTEP = BK / 16;          // 8 WMMA k-steps of 16
  constexpr int MYF = NFRAG / NW;         // wave w owns fragments w, w + NW, ...
  constexpr int ACH = BM * BK / 16 / NT;  // 16-byte A chunks per thread per k tile
  constexpr int LDA = BK + 8;             // the m16 kernel's row pad
  __shared__ unsigned char sA[BM * LDA];
  // The k tile's 64 row a_scales, staged with A (rows >= M hold 0: the m16 kernel's `row < M ? .. : 0`).
  __shared__ float sAs[BM];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int n0 = blockIdx.x * BN;
  const int m0 = (int)blockIdx.y * BM;
  if (m0 >= M) return;
  const int ktiles = K / BK;

  floatx8 acc[MF][MYF];
#pragma unroll
  for (int f = 0; f < MF; ++f)
#pragma unroll
    for (int i = 0; i < MYF; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[f][i][e] = 0.f;

  const unsigned char *bbase = W + ((long)(n0 / 16) + wave) * (long)K * 16
                               + (long)(lane & 15) * 16 + (long)(lane >> 4) * 8;
  typedef uint2_t bfrag_t[MYF][KSTEP];
  bfrag_t b0, b1;
  uint4_t abuf[ACH];
  float asbuf = 0.f;
  auto fetch_b = [&](bfrag_t &dst, int kt) {
#pragma unroll
    for (int i = 0; i < MYF; ++i)
#pragma unroll
      for (int ks = 0; ks < KSTEP; ++ks)
        dst[i][ks] = *(const uint2_t *)(bbase + (long)i * NW * K * 16
                                        + ((long)kt * KSTEP + ks) * 256);
  };
  auto fetch_a = [&](int kt) {
#pragma unroll
    for (int i = 0; i < ACH; ++i) {
      const int off = (tid + i * NT) * 16;
      const int r = off / BK, c = off % BK;
      const int gr = m0 + r;
      const int rc = gr < M ? gr : M - 1;   // clamp, as the m16 kernel does
      abuf[i] = *(const uint4_t *)(A + (long)rc * K + kt * BK + c);
    }
    if (tid < BM) {
      const int gr = m0 + tid;
      asbuf = gr < M ? As[(long)gr * (K / 128) + kt] : 0.f;
    }
  };
  auto step = [&](int kt, bfrag_t &cur, bfrag_t &nxt) {
#pragma unroll
    for (int i = 0; i < ACH; ++i) {
      const int off = (tid + i * NT) * 16;
      *(uint4_t *)(&sA[(off / BK) * LDA + off % BK]) = abuf[i];
    }
    if (tid < BM) sAs[tid] = asbuf;
    __syncthreads();
    if (kt + 1 < ktiles) {
      fetch_b(nxt, kt + 1);
      fetch_a(kt + 1);
    }
    const float bs = Ws[(long)(n0 / 128) * (K / 128) + kt];
    // One m-fragment at a time: its tile's dot over the k tile (the same ascending k-steps into a
    // zeroed tile), then its scale-accumulate. The B fragments in `cur` are reused by all four.
#pragma unroll
    for (int f = 0; f < MF; ++f) {
      floatx8 tile[MYF];
#pragma unroll
      for (int i = 0; i < MYF; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) tile[i][e] = 0.f;
#pragma unroll
      for (int ks = 0; ks < KSTEP; ++ks) {
        const uint2_t av = *(const uint2_t *)(&sA[(f * 16 + (lane & 15)) * LDA + ks * 16
                                                 + (lane >> 4) * 8]);
#pragma unroll
        for (int i = 0; i < MYF; ++i) tile[i] = dg64_wmma(av, cur[i][ks], tile[i]);
      }
#pragma unroll
      for (int i = 0; i < MYF; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float as = sAs[f * 16 + dg64_acc_row(lane, e)];
          acc[f][i][e] += tile[i][e] * (as * bs);
        }
    }
    __syncthreads();
  };

  fetch_b(b0, 0);
  fetch_a(0);
  int kt = 0;
  for (; kt + 1 < ktiles; kt += 2) {
    step(kt, b0, b1);
    step(kt + 1, b1, b0);
  }
  if (kt < ktiles) step(kt, b0, b1);

#pragma unroll
  for (int f = 0; f < MF; ++f)
#pragma unroll
    for (int i = 0; i < MYF; ++i)
      dg64_store_acc(acc[f][i], C, n0 + (wave + i * NW) * 16, m0 + f * 16, M, N, N, lane);
}

// THE ROUTED INSTANTIATION: 8 waves, each wave one n-fragment x four m-fragments (224 VGPRs, no
// spill, 8.5 KiB LDS). A 4-wave form (the m16 kernel's launch width, two n-fragments a wave) is also
// bit-exact but spills 18 VGPRs and is 5 % slower (CMP27K p1: 167.3 vs 159.7 us at M 64), so it is
// not compiled here.
extern "C" __global__ __launch_bounds__(256) void dflash_gemm_a8w8_bs128_m64n128k128_nw8(
    const unsigned char *__restrict__ a, const unsigned char *__restrict__ w,
    const float *__restrict__ as, const float *__restrict__ ws, __bf16 *__restrict__ c,
    int m, int n, int k) {
  dflash_gemm_m64_impl<8>(a, w, as, ws, c, m, n, k);
}
