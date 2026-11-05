// FN61g: the drafter's EXACT RERANK (FN33) on the W8 head (`--fn-trunk w8-head`), its own code
// object so `dflash_gemm` / `flashnext_trunk8` / `flashnext_trunk8p` keep their bytes.
//
//   fn_t8r_a8w8_gather_m16n64   grid (ncols / 64), block 128: the rerank's candidate columns
//                               scored straight from the rank's PRESHUFFLED W8 head half
//                               (`lm_head.weight.w8.{w,ws}`, the bytes the verify step's head
//                               reads). Output column o = r * kk + j (o < rows * kk) is vocabulary
//                               column s = cand[r * cstride + j]; columns past rows * kk score 0.
//                               c[i * ldc + o] = the W8 head GEMM's logit (row i, column s).
//
// THE BYTE LAW (gated on the card by tools/capture/fn61g/t8r_gate.py and the engine's
// `--fn-mtp-chain-check` score diffs): each output element is `dflash_gemm_a8w8_bs128_m16n128k128`'s
// arithmetic for that (row, column) -- the same per-128 k tile chain (a fresh fp32 WMMA tile over 8
// fp8 k steps, then `acc += tile * (as * bs)`, one bf16 round at the store). The only differences
// are WHERE the operands come from: lane (l & 15)'s B fragment is row s of the preshuffled head
// (16 rows a 256-byte block: byte (s / 16) * K * 16 + (k / 16) * 256 + (s % 16) * 16 + k % 16 --
// the parent reads row n0 + wave * 16 + (l & 15) of the same layout at the same in-block offsets)
// and its block scale is Ws[(s / 128) * (K / 128) + kt] per lane instead of one per 128-wide tile.
// A WMMA element depends only on its own A row and B column, so the gathered column is the
// parent's bit for bit. M <= 16 (one row tile: the drafter's selected rows, zero-padded).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

typedef float floatx8 __attribute__((ext_vector_type(8)));
typedef unsigned int uint2_t __attribute__((ext_vector_type(2)));
typedef unsigned int uint4_t __attribute__((ext_vector_type(4)));

#define TR_BM 16
#define TR_BK 128
#define TR_NW 4
#define TR_NT (TR_NW * 32)
#define TR_BN (TR_NW * 16)

__device__ __forceinline__ floatx8 tr_wmma(uint2_t a, uint2_t b, floatx8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}
__device__ __forceinline__ int tr_acc_row(int lane, int e) { return (lane >> 4) * 8 + e; }

extern "C" __global__ __launch_bounds__(TR_NT) void fn_t8r_a8w8_gather_m16n64(
    const unsigned char *__restrict__ A,   // [M, K] e4m3 (the act quant of the rerank rows)
    const unsigned char *__restrict__ W,   // [N/16, K*16] e4m3, preshuffled (the head half)
    const float *__restrict__ As,          // [M, K/128]
    const float *__restrict__ Ws,          // [N/128, K/128]
    const int *__restrict__ cand,          // [rows, cstride] vocabulary columns of the half
    __bf16 *__restrict__ C,                // [M, ldc]
    int M, int ncols, int K, int rows, int kk, int cstride, int ldc) {
  constexpr int BM = TR_BM, BK = TR_BK, NT = TR_NT;
  constexpr int KSTEP = BK / 16;
  constexpr int ACH = BM * BK / 16 / NT;
  constexpr int LDA = BK + 8;
  __shared__ unsigned char sA[BM * LDA];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int o = (int)blockIdx.x * TR_BN + wave * 16 + (lane & 15);  // this lane's output column
  const int rr = o / kk, j = o - rr * kk;
  const int s = (o < ncols && rr < rows) ? cand[rr * cstride + j] : -1;
  const bool live = s >= 0;
  const int sr = live ? s : 0;
  const int ktiles = K / BK;

  floatx8 acc;
#pragma unroll
  for (int e = 0; e < 8; ++e) acc[e] = 0.f;

  const unsigned char *bbase = W + (long)(sr / 16) * (long)K * 16 + (long)(sr % 16) * 16
                               + (long)(lane >> 4) * 8;
  const float *wsrow = Ws + (long)(sr / 128) * (K / 128);
  typedef uint2_t bfrag_t[KSTEP];
  bfrag_t b0, b1;
  uint4_t abuf[ACH];
  auto fetch_b = [&](bfrag_t &dst, int kt) {
#pragma unroll
    for (int ks = 0; ks < KSTEP; ++ks) {
      const uint2_t v = *(const uint2_t *)(bbase + ((long)kt * KSTEP + ks) * 256);
      dst[ks] = live ? v : uint2_t{0u, 0u};
    }
  };
  auto fetch_a = [&](int kt) {
#pragma unroll
    for (int i = 0; i < ACH; ++i) {
      const int off = (tid + i * NT) * 16;
      const int r = off / BK, c = off % BK;
      const int rc = r < M ? r : M - 1;
      abuf[i] = *(const uint4_t *)(A + (long)rc * K + kt * BK + c);
    }
  };
  auto step = [&](int kt, bfrag_t &cur, bfrag_t &nxt) {
#pragma unroll
    for (int i = 0; i < ACH; ++i) {
      const int off = (tid + i * NT) * 16;
      *(uint4_t *)(&sA[(off / BK) * LDA + off % BK]) = abuf[i];
    }
    __syncthreads();
    if (kt + 1 < ktiles) {
      fetch_b(nxt, kt + 1);
      fetch_a(kt + 1);
    }
    floatx8 tile;
#pragma unroll
    for (int e = 0; e < 8; ++e) tile[e] = 0.f;
#pragma unroll
    for (int ks = 0; ks < KSTEP; ++ks) {
      const uint2_t av = *(const uint2_t *)(&sA[(lane & 15) * LDA + ks * 16 + (lane >> 4) * 8]);
      tile = tr_wmma(av, cur[ks], tile);
    }
    __syncthreads();
    const float bs = live ? wsrow[kt] : 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int row = tr_acc_row(lane, e);
      const float as = row < M ? As[(long)row * (K / 128) + kt] : 0.f;
      acc[e] += tile[e] * (as * bs);
    }
  };

  fetch_b(b0, 0);
  fetch_a(0);
  int kt = 0;
  for (; kt + 1 < ktiles; kt += 2) {
    step(kt, b0, b1);
    step(kt + 1, b1, b0);
  }
  if (kt < ktiles) step(kt, b0, b1);

  if (o >= ncols) return;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int row = tr_acc_row(lane, e);
    if (row < M) C[(long)row * ldc + o] = (__bf16)acc[e];
  }
}
