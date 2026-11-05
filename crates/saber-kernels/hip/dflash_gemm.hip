// The DFlash2 drafter's fp8 W8A8 blockscale GEMM.
//
// NOT a port: prod runs the reference's Triton `_gemm_a8w8_blockscale_preshuffle_kernel`, so there is no
// HIP source to slice. This is written from that kernel's semantics and gated bit-exact against
// its output, the way `attn_glue` was written from prod's Triton and ATen. The census found it at
// 21 launches and 1567 us per decode step -- 7.9% of the step, and the largest single thing in
// the drafter path that Saber had no kernel for at all.
//
// THE SHAPE PROD RUNS, read off the Triton specialisation names in the trace (they carry every
// constexpr, which is how the census identified them):
//   GROUP_K=128 GROUP_N=128 BLOCK_SIZE_M=16 BLOCK_SIZE_N=128 BLOCK_SIZE_K=128
//   GROUP_SIZE_M=8 NUM_KSPLIT=1 EVEN_K=1 cache_modifier=NONE
// Five (N, K) pairs, all at M=8 (1 + num_speculative_tokens at batch 1; BLOCK_SIZE_M is 16, so
// every M in 1..16 launches the same grid):
//   qkv 3072x5120, o_proj 5120x2048, gate_up 17408x5120, down_proj 5120x8704,
//   aux_fc 5120x25600  <- the census's "234 us outlier": the aux-hidden-state projection over the
//                         five target layers (5 x 5120), simply the biggest GEMM in the drafter.
//
// ARITHMETIC, and every rounding in it is the Triton kernel's:
//   * GROUP_N >= BLOCK_SIZE_N and GROUP_K >= BLOCK_SIZE_K, so the whole 128x128 B tile shares ONE
//     scalar b_scale and the kernel takes the `(a_scale * b_scale)[:, None]` arm.
//   * per K tile of 128: acc[m][n] += dot_fp32(a_tile, b_tile)[m][n] * (a_scale[m] * b_scale)
//     -- the fp8 dot accumulates in fp32 inside the WMMA, the SCALE is applied to the tile's dot
//     product, and only then is it added to the running fp32 accumulator. Scaling after the
//     accumulate, or folding the two scales differently, is a different answer.
//   * the store is one round to bf16.
//
// B IS PRESHUFFLED, and this is the layout the kernel un-shuffles rather than a guess:
//   shuffled[n / 16][ (k / 32) * 512 + ((k % 32) / 16) * 256 + (n % 16) * 16 + (k % 16) ] = W[n][k]
// derived by inverting
//   b.reshape(1, BN/16, BK/32, 2, 16, 16).permute(0, 1, 4, 2, 3, 5).reshape(BN, BK).trans(1, 0)
// and checked independently in the capture: the same operands through the reference's NON-preshuffled
// entry point have to give the same answer.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

typedef float floatx8 __attribute__((ext_vector_type(8)));
typedef unsigned int uint2_t __attribute__((ext_vector_type(2)));
typedef unsigned int uint4_t __attribute__((ext_vector_type(4)));

#define DG_BM   16     // BLOCK_SIZE_M: one WMMA M tile, and the reason M in 1..16 is one grid
#define DG_BN  128     // BLOCK_SIZE_N
#define DG_BK  128     // BLOCK_SIZE_K == GROUP_K, which the Triton wrapper asserts
#define DG_NW    4     // waves per workgroup; 4 x 32 = 128 threads, as the Triton launch uses
#define DG_NT  (DG_NW * 32)

// One 16x16x16 fp8 WMMA. The same builtin the Paro producers use (par_kernels.h:173), so the
// accumulate order inside a k-step is the hardware's and not this file's choice.
__device__ __forceinline__ floatx8 dg_wmma(uint2_t a, uint2_t b, floatx8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}

// The gfx12 w32 fragment and accumulator layout, taken from this tree's own WMMA code rather
// than from memory (par_kernels.h:1844 `col = lane & 15, kb8 = (lane >> 4) * 8`, and its epilogue
// at :2045 `m = ... + kb8 + e`):
//   * an A or B fragment of a 16x16x16 step: lane l holds row (l & 15), k bytes
//     (l >> 4) * 8 .. +8 -- eight bytes, one uint2_t.
//   * the f32 accumulator: element e of lane l is row (l >> 4) * 8 + e, column l & 15.
// Getting either wrong is silent: the kernel still runs and still produces numbers.
__device__ __forceinline__ int dg_acc_row(int lane, int e) { return (lane >> 4) * 8 + e; }

__device__ __forceinline__ void dg_store_acc(const floatx8 &acc, __bf16 *__restrict__ c,
                                             int n0, int m0, int M, int N, long ldc, int lane) {
  const int col = n0 + (lane & 15);
  if (col >= N) return;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int row = m0 + dg_acc_row(lane, e);
    if (row < M) c[(long)row * ldc + col] = (__bf16)acc[e];
  }
}

// -------------------------------------------------------------------------------------------
// One workgroup computes one 16x128 tile of C. Grid is cdiv(N, 128) blocks -- M <= 16 so there is
// exactly one M tile, which is what the trace's GRID_MN = cdiv(N, 128) says.
//
// Per k tile: A (16 x 128 e4m3) goes through LDS, B (128 x 128 e4m3) goes from global STRAIGHT
// into the WMMA operand registers -- the preshuffled block IS the fragment -- both prefetched one
// tile ahead; then 8 n-fragments x 8 k-steps of WMMA into eight fp32 accumulators, then scale and
// accumulate.
template <int BM, int BN, int BK, int NW, bool MT>
__device__ __forceinline__ void dflash_gemm_a8w8_blockscale_impl(
    const unsigned char *__restrict__ A,   // [M, K] e4m3, row-major
    const unsigned char *__restrict__ W,   // [N/16, K*16] e4m3, preshuffled
    const float *__restrict__ As,          // [M, K/128] fp32 block scales
    const float *__restrict__ Ws,          // [N/128, K/128] fp32 block scales
    __bf16 *__restrict__ C,                // [M, N] bf16
    int M, int N, int K) {
  constexpr int NT = NW * 32;
  constexpr int NFRAG = BN / 16;                 // 8 n-fragments of 16
  constexpr int KSTEP = BK / 16;                 // 8 WMMA k-steps of 16
  constexpr int MYF = NFRAG / NW;                // wave w owns fragments w, w + NW
  constexpr int ACH = BM * BK / 16 / NT;         // 16-byte A chunks per thread per k tile
  constexpr int LDA = BK + 8;                    // +8 bytes = +2 banks per row: see below
  __shared__ unsigned char sA[BM * LDA];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int n0 = blockIdx.x * BN;
  // WHICH SIXTEEN ROWS THIS WORKGROUP OWNS. `MT` is a compile-time constant, so the untiled
  // instantiation's `m0` is the literal 0 and every expression below folds back to exactly the
  // arithmetic it had before the tile existed -- which is what makes "bit-identical per 16-row
  // slice" a property of the SOURCE rather than of a measurement.
  const int m0 = MT ? (int)blockIdx.y * BM : 0;
  if (MT && m0 >= M) return;
  const int ktiles = K / BK;

  floatx8 acc[MYF];
#pragma unroll
  for (int i = 0; i < MYF; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;

  // ---- B NEVER GOES THROUGH LDS, AND THAT IS WHAT THE PRESHUFFLE IS FOR ----------------------
  // Un-shuffling B into LDS and reading fragments back out of it cost 2.5x the reference's time. The
  // layout already IS the fragment layout: for one n-group g = n / 16 and one k-run r = k / 16,
  // the shuffled row g holds a contiguous 256-byte block at byte r * 256 whose element (n % 16,
  // k % 16) sits at (n % 16) * 16 + (k % 16). A 16x16x16 w32 B fragment wants lane l to hold row
  // (l & 15), k bytes (l >> 4) * 8 .. +8 -- which is byte (l & 15) * 16 + (l >> 4) * 8 of exactly
  // that block. So one wave reading one contiguous 256 bytes, 8 per lane, IS the fragment: no LDS
  // round trip, no un-shuffle arithmetic, no barrier, and none of the 16-way ds_read bank conflict
  // that a BK-strided LDS tile has (row stride 128 B = 32 banks, so all 16 rows land on one bank
  // pair). Every byte of the weight stream -- 92% of the traffic -- now goes straight from L2 into
  // the WMMA operand register.
  //
  // Both operand streams are then prefetched one k tile ahead, so the WMMA of tile kt runs
  // underneath the loads of tile kt+1. ARITHMETIC UNCHANGED: this moves bytes earlier and moves
  // them through a different unit, it does not reorder one add, which is what keeps the bit-exact
  // gate the acceptance test for a performance change.
  const unsigned char *bbase = W + ((long)(n0 / 16) + wave) * (long)K * 16
                               + (long)(lane & 15) * 16 + (long)(lane >> 4) * 8;
  // Two NAMED fragment buffers, and the k loop unrolled by two so each step names the one it
  // reads. `bf[kt & 1]` cost 272 bytes of scratch -- a dynamically indexed register array is a
  // stack array -- which put the weight stream through memory twice over.
  typedef uint2_t bfrag_t[MYF][KSTEP];
  bfrag_t b0, b1;
  uint4_t abuf[ACH];
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
      const int gr = m0 + r;                      // this tile's row, in the matrix
      const int rc = gr < M ? gr : M - 1;         // clamp: a masked load is a round trip
      abuf[i] = *(const uint4_t *)(A + (long)rc * K + kt * BK + c);
    }
  };
  auto step = [&](int kt, bfrag_t &cur, bfrag_t &nxt) {
#pragma unroll
    for (int i = 0; i < ACH; ++i) {          // commit A: the s_waitcnt for tile kt lands here
      const int off = (tid + i * NT) * 16;
      *(uint4_t *)(&sA[(off / BK) * LDA + off % BK]) = abuf[i];
    }
    __syncthreads();
    if (kt + 1 < ktiles) {        // issued, not waited: the WMMA below runs underneath these
      fetch_b(nxt, kt + 1);
      fetch_a(kt + 1);
    }

    // ---- the dot: this wave's n-fragments x 8 k-steps ------------------------------------------
    floatx8 tile[MYF];
#pragma unroll
    for (int i = 0; i < MYF; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) tile[i][e] = 0.f;
#pragma unroll
    for (int i = 0; i < MYF; ++i)
#pragma unroll
      for (int ks = 0; ks < KSTEP; ++ks) {
        const uint2_t av =
            *(const uint2_t *)(&sA[(lane & 15) * LDA + ks * 16 + (lane >> 4) * 8]);
        tile[i] = dg_wmma(av, cur[i][ks], tile[i]);
      }
    __syncthreads();

    // ---- scale, then accumulate ----------------------------------------------------------------
    // ONE scalar b_scale for the whole 128-wide N tile (GROUP_N == BN is why the Triton kernel
    // takes its `(a_scale * b_scale)[:, None]` arm), and one a_scale per row. The scale multiplies
    // the TILE's dot product; folding it after the outer accumulate is a different answer.
    const float bs = Ws[(long)(n0 / 128) * (K / 128) + kt];
#pragma unroll
    for (int i = 0; i < MYF; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int row = m0 + dg_acc_row(lane, e);
        const float as = row < M ? As[(long)row * (K / 128) + kt] : 0.f;
        acc[i][e] += tile[i][e] * (as * bs);
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

#pragma unroll
  for (int i = 0; i < MYF; ++i)
    dg_store_acc(acc[i], C, n0 + (wave + i * NW) * 16, m0, M, N, N, lane);
}

extern "C" __global__ __launch_bounds__(DG_NT) void dflash_gemm_a8w8_bs128_m16n128k128(
    const unsigned char *__restrict__ a, const unsigned char *__restrict__ w,
    const float *__restrict__ as, const float *__restrict__ ws, __bf16 *__restrict__ c,
    int m, int n, int k) {
  dflash_gemm_a8w8_blockscale_impl<DG_BM, DG_BN, DG_BK, DG_NW, false>(a, w, as, ws, c, m, n, k);
}

// ---- THE M-TILED INSTANTIATION, for a BATCHED proposal ---------------------------------------
//
// 21 of the proposal's launches are this kernel, and at four requests every one of them is asked
// for 32 rows: `BLOCK_SIZE_M` is 16 and past it "the tile is wrong, not slow", so the committed
// rules refused the whole batch by name (LEDGER, workstreams S3/S4 open item 6, S6's census).
//
// The tile does not change and nothing about the reduction moves. `blockIdx.y` picks which
// sixteen rows a workgroup owns; the k loop, the eight n-fragments, the per-tile
// `acc += tile * (a_scale * b_scale)` and the single round to bf16 at the store are the kernel
// above, instruction for instruction, because `MT` is a template parameter and `m0` is a
// compile-time 0 in the other instantiation. So **every output row's arithmetic is the m16
// kernel's, element for element** -- which is the claim the gate tests from both ends: bit-identical
// to `dflash_gemm_a8w8_bs128_m16n128k128` on each 16-row slice, and against the reference's own kernel at
// prod's specialisation at M = 32 and M = 64 (which handles any M: its GRID_MN is
// `cdiv(M, 16) * cdiv(N, 128)`, so the reference at 32 rows is the same program on two m tiles).
//
// A SECOND SYMBOL rather than a widened first one, for the reason every second symbol in this tree
// is one: the shape the whole proposal has been gated at keeps the code object it was gated with.
extern "C" __global__ __launch_bounds__(DG_NT) void dflash_gemm_a8w8_bs128_mtn128k128(
    const unsigned char *__restrict__ a, const unsigned char *__restrict__ w,
    const float *__restrict__ as, const float *__restrict__ ws, __bf16 *__restrict__ c,
    int m, int n, int k) {
  dflash_gemm_a8w8_blockscale_impl<DG_BM, DG_BN, DG_BK, DG_NW, true>(a, w, as, ws, c, m, n, k);
}
