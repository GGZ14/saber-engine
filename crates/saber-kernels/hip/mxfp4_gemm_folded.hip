// Saber source (CLEANROOM-1, 2026-10-25: frozen from a retired generator's output; edit this file directly)
//   reference sha256 79a7eb6f261c2939386a4c231cf5727a0a1f2325d8a391419178778da1fd2ae2
//   device code sliced VERBATIM by line range; only the __global__ signatures became
//   __device__ __forceinline__ *_impl and the extern "C" wrappers at the end are new.
// Do not edit by hand: re-run the generator. No host code, no pybind, no getenv in this file --
// every tile parameter the prod launcher chose from an env var is an explicit symbol here.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "mxfp4_common.h"

// Autotuned folded-scale kernel. The MX block exponent is pushed into the weight byte during the
// LDS upconvert (kMag), so the inner loop is pure WMMA -- no temp accumulator, no per-32-block
// rescale, no LDS scale reads. Lossless while the shifted value stays an e4m3 normal (d <= 5):
// 99.998% of the checkpoint's 761M blocks qualify, worst case 10, 282 blocks flush.
//
// Tile config from a correctness-gated sweep of 28 shapes (BM/BN/BK/TM/TN/wave grid). BM=256 via
// TM=4 won at EVERY production shape and batch size -- a taller tile buys more A reuse per weight
// read. It also removed the large-M cliff the earlier hand-written BM=128 version had.
#define TM 4
#define WM 4
#define WN 2
#define NWAVE (WM * WN)
#define NTHREADS (NWAVE * 32)
#define BMF (WM * TM * 16)
#define BNF_OF(tn) (WN * (tn) * 16)
#define BNF BNF_OF(2)

// TN is templated because the winner depends on M. Widening the N tile (TN=4 -> BNF=128) amortises
// the A-tile staging, which an ablation prices at 24% of runtime -- A is BMF*BK = 16 KB per K-slab
// against W's 2 KB, and with BNF=64 it is re-staged across twice as many column blocks. Measured on
// gate_up 17408x5120, tight A/B, 10 alternations with no overlap between the distributions:
//   M=8192 +10.0%  M=6144 +9.7%  M=4096 +8.5%  M=3072 +6.9%
//   M=2048 +1.3%   M=1536 -1.6%  M=1024 -4.2%  M=512 -8.8%
// Below the crossover the wide tile cannot fill: it halves the block count in N while BMF still
// masks most of a short M. Hence TN=4 only from M >= 2048; SABER_MXFP4_TN4_MIN_M overrides.
// WPERM selects the weight layout. false is the checkpoint's own [N, K/2]; true is fragment
// order, one 32-lane uint32 slot per (n-tile, k-step), which is what the reference's
// saber_gemm_mxfp4a8_nt_m64 reads and what makes a wave's weight read 128 contiguous bytes instead
// of sixteen rows K/2 bytes apart. Both paths produce the SAME sW tile, so everything downstream
// of the staging loop is untouched.
template <int TN, bool WPERM, bool EPIFAST = true>
__device__ __forceinline__ void saber_mxfp4_fp8_gemm_folded_impl(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride) {
  __shared__ unsigned char sA[BMF * ASTR];
  __shared__ unsigned char sW[WN * TN * 16 * WSTR];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN, wn = wave % WN;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  constexpr int BNF_T = WN * TN * 16;
  const int m0 = blockIdx.y * BMF, n0 = blockIdx.x * BNF_T;
  // Merged-linear partition select (paro: one launch over the whole N, partition p's n-blocks
  // read the p-th rotated copy of A and its scales). Boundaries are multiples of every n-tile
  // (launcher-checked), so a block never straddles one. pb1 = pb2 = 1<<30 (the default) is p = 0.
  {
    const int p_ = n0 >= pb2 ? 2 : (n0 >= pb1 ? 1 : 0);
    A += (size_t)p_ * astride;
    As += (size_t)p_ * M;
  }

  floatx8 acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    // Wave-uniform 64-bit bases, hoisted out of the per-thread address arithmetic. Written as
    // `base + int_offset`, the compiler keeps the base in SGPRs and emits the SADDR form of
    // global_load, so a lane pays one 32-bit add instead of the v_add_co_u32 / v_add_co_ci_u32
    // pair a full 64-bit address costs. The matmul block was spending ~71 of its 565 instructions
    // on that arithmetic while using only 29 of the ~102 available SGPRs.
    // r*K fits an int by a wide margin: r < BMF = 256 and K <= 17408, so r*K < 4.5e6.
    const unsigned char *__restrict__ Ab = A + (size_t)m0 * K + k0;
    const unsigned char *__restrict__ Wb = W + ((size_t)n0 * K + k0) / 2;
    // s_prefetch_data on the next slab's weight runs: measured and REJECTED 2026-08-28 --
    // -0.3 to -1% on every production shape (tier7 spfd mode). With branchless staging at
    // occupancy 10 the loads are already fully overlapped; early L2 arrival buys nothing.
    const unsigned char *__restrict__ Wsb = Ws + (size_t)(k0 / 32) * N + n0;
    const unsigned char *__restrict__ Wrefb = Wref + n0;
    // ---- A tile: BM*BK bytes, 16 B per thread per iteration ----
#pragma unroll
    for (int off = 0; off < BMF * BK; off += NTHREADS * 16) {
      const int idx = off + tid * 16;
      const int r = idx / BK, c = idx % BK;
      const int rc = r < M - 1 - m0 ? r : M - 1 - m0;              // clamp, never predicate
      *(uint4_t *)(&sA[r * ASTR + c]) = *(const uint4_t *)(Ab + (rc * K + c));
    }
    // ---- W tile: BN*BK/2 packed bytes, 8 B per thread per iteration -> 16 e4m3 bytes ----
    if constexpr (WPERM) {
      // Fragment order: slot s of the tile is one lane's four packed bytes (eight elements) for
      // one (n-tile, k-step), so consecutive threads read consecutive uint32 and a wave takes 128
      // contiguous bytes. WSLOTS adjacent lane slots per thread; adjacent slots differ only in
      // the row (lane index is half*16 + row), so one vector read covers WSLOTS adjacent rows at
      // the same k and the wave still reads its bytes contiguously.
      //
      // WSLOTS is 4 (a 16 B request) whenever the tile has the slots for it, else 2 (8 B). The
      // original 2-slot trade was priced against PREDICATED staging: one slot (4 B) cost 14-25%
      // of prefill, two recovered most of it, and at TN=4 the remaining narrow-request penalty
      // was ~11% against checkpoint order (tier7, 2026-08-28). A 4-slot group never straddles
      // the k-half boundary because 16 % 4 == 0.
      constexpr int KSTEPS_T = BK / 16, NTILES_T = BNF_T / 16;
      constexpr int TOT_SLOTS = NTILES_T * KSTEPS_T * 32;
      constexpr int WSLOTS = (TOT_SLOTS >= NTHREADS * 4) ? 4 : 2;
      const unsigned int *__restrict__ Wp = (const unsigned int *)W;
      const int ksteps_g = K / 16, kstep0 = k0 / 16;
#pragma unroll
      for (int off = 0; off < TOT_SLOTS; off += NTHREADS * WSLOTS) {
        const int sl = off + tid * WSLOTS;              // first of the group; WSLOTS-aligned
        const int lane_ = sl & 31, rest = sl >> 5;
        const int kst = rest % KSTEPS_T, ntl = rest / KSTEPS_T;
        const int r = ntl * 16 + (lane_ & 15), kloc = kst * 16 + (lane_ >> 4) * 8;
        const int gn = n0 + r;
        // Clamp, never predicate (see the A-tile note in the first kernel). gn is always
        // WSLOTS-aligned here and N % 16 == 0 under WPERM, so clamping to the last WSLOTS-aligned
        // row keeps the vector read aligned and every gc + q in bounds; the fragment address is
        // recomputed from the clamped row.
        const int gc = gn < N - WSLOTS ? gn : N - WSLOTS;
        const int lanec = (lane_ & 16) | (gc & 15);
        const unsigned int *src = &Wp[((size_t)(gc >> 4) * ksteps_g + kstep0 + kst) * 32 + lanec];
        unsigned int wq[WSLOTS];
        if constexpr (WSLOTS == 4) {
          const uint4_t v = *(const uint4_t *)src;
          wq[0] = v[0]; wq[1] = v[1]; wq[2] = v[2]; wq[3] = v[3];
        } else {
          const uint2_t v = *(const uint2_t *)src;
          wq[0] = v[0]; wq[1] = v[1];
        }
        const int blk = (k0 + kst * 16) / 32;
        // One dword read each for the WSLOTS reference and block exponents instead of 2*WSLOTS
        // byte reads: gc is WSLOTS-aligned and N % 16 == 0, so both reads are aligned.
        unsigned int refw, wsw;
        if constexpr (WSLOTS == 4) {
          refw = *(const unsigned int *)(Wref + gc);
          wsw = *(const unsigned int *)(Ws + (size_t)blk * N + gc);
        } else {
          refw = *(const unsigned short *)(Wref + gc);
          wsw = *(const unsigned short *)(Ws + (size_t)blk * N + gc);
        }
#pragma unroll
        for (int q = 0; q < WSLOTS; ++q) {
          int d = (int)((refw >> (8 * q)) & 0xFF) - (int)((wsw >> (8 * q)) & 0xFF);
          d = d < 0 ? 0 : (d > 15 ? 15 : d);
          const unsigned int t0 = kMag[d][0], t1 = kMag[d][1];
          const unsigned int ev = wq[q] & 0x0F0F0F0Fu;
          const unsigned int od = (wq[q] >> 4) & 0x0F0F0F0Fu;
          const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u)
                                | ((ev & 0x08080808u) << 4);
          const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u)
                                | ((od & 0x08080808u) << 4);
          *(uint2_t *)(&sW[(r + q) * WSTR + kloc]) =
              uint2_t{__builtin_amdgcn_perm(bo, be, 0x05010400u),
                      __builtin_amdgcn_perm(bo, be, 0x07030602u)};
        }
      }
    } else
#pragma unroll
    for (int off = 0; off < BNF_T * BK / 2; off += NTHREADS * 8) {
      const int idx = off + tid * 8;
      const int r = idx / (BK / 2), c = idx % (BK / 2);
      const int rc = r < N - 1 - n0 ? r : N - 1 - n0;          // clamp, never predicate
      // Same hoist as A. rc*(K/2) < 1.2e6 and the scale offset < 3*N, both comfortably int.
      const unsigned long long packed = *(const unsigned long long *)(Wb + (rc * (K / 2) + c));
      const int blk = (c * 2) / 32;                            // k0's share is in Wsb
      int dsh = (int)Wrefb[rc] - (int)Wsb[blk * N + rc];       // scales are [K/32, N]
      dsh = dsh < 0 ? 0 : (dsh > 15 ? 15 : dsh);
      // v_perm_b32 resolves 4 table lookups per instruction from the 8-byte magnitude table;
      // the sign is separable because entries 8-15 are entries 0-7 with bit 7 set. The upconvert
      // measured 21% of runtime with per-byte constant loads (156 -> 188 TF with it removed
      // entirely), and this recovers +9-12% of that.
      const unsigned int t0 = kMag[dsh][0], t1 = kMag[dsh][1];
      unsigned int outw[4];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const unsigned int w = (unsigned int)(packed >> (32 * h));
        const unsigned int ev = w & 0x0F0F0F0Fu;
        const unsigned int od = (w >> 4) & 0x0F0F0F0Fu;
        const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u)
                              | ((ev & 0x08080808u) << 4);
        const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u)
                              | ((od & 0x08080808u) << 4);
        outw[2 * h]     = __builtin_amdgcn_perm(bo, be, 0x05010400u);
        outw[2 * h + 1] = __builtin_amdgcn_perm(bo, be, 0x07030602u);
      }
      *(uint4_t *)(&sW[r * WSTR + c * 2]) = uint4_t{outw[0], outw[1], outw[2], outw[3]};
    }
    __syncthreads();

#pragma unroll
    for (int step = 0; step < BK / 16; ++step) {
      const int kk = step * 16 + kb8;
      int2_t af[TM], wf[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        const unsigned char *p = &sA[(wm * TM * 16 + i * 16 + col) * ASTR + kk];
        af[i][0] = *(const int *)p; af[i][1] = *(const int *)(p + 4);
      }
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const unsigned char *p = &sW[(wn * TN * 16 + j * 16 + col) * WSTR + kk];
        wf[j][0] = *(const int *)p; wf[j][1] = *(const int *)(p + 4);
      }
      // Hard scheduling barrier per k-step. Without it the compiler hoists EVERY step's fragment
      // loads above the first WMMA -- the ISA shows 346 instructions and 140 v_dual_mov_b32 ahead
      // of a 64-WMMA run that itself contains zero moves. Those moves are the register shuffling
      // that holds four k-steps of fragments live at once. Fencing each step keeps one step's
      // fragments live instead of four. Same idea as the attention kernel's SGB bit, which notes
      // that without the barrier its prefetch is inert.
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf[j], acc[i][j]);
    }
    __syncthreads();
  }

  float rf[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int n = n0 + wn * TN * 16 + j * 16 + col;
    rf[j] = (n < N) ? __int_as_float((int)Wref[n] << 23) : 0.f;
  }
  // Epilogue addressing and predication. `C[(size_t)m * N + n]` is a full 64-bit address per
  // element and there are TM*TN*8 of them, each in its own s_and_saveexec region from the bounds
  // test. Measured in the ISA: 145 v_add_co_u32/v_add_co_ci pairs, 103 s_and_saveexec and 130
  // 64-bit shifts in a kernel with 32 WMMA -- about 15% of the instruction stream spent getting
  // to memory rather than computing.
  //
  // Hoisting a wave-uniform base and indexing with a 32-bit offset gets the SADDR form, so a lane
  // pays one 32-bit add instead of a 64-bit add pair -- the same reasoning as the staging bases
  // above. And a block lying entirely inside M and N needs no bounds test at all; on a real
  // prefill only the last row-block and column-block are ragged. The offset cannot overflow an
  // int: at most (kb8 + (TM-1)*16 + 7) * N + N <= 64 * 34816.
  //
  // BIT-IDENTICAL to the path below by construction -- same operands, same order, same rounding.
  // It is gated that way rather than on a tolerance, because nothing here may change numerics.
  if constexpr (EPIFAST) {
    __bf16 *__restrict__ Cb = C + (size_t)(m0 + wm * TM * 16) * N;
    const float *__restrict__ Asb = As + m0 + wm * TM * 16;
    const bool full = (m0 + wm * TM * 16 + (TM - 1) * 16 + kb8 + 7 < M) &&
                      (n0 + wn * TN * 16 + (TN - 1) * 16 + col < N);
    if (full) {
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int r = i * 16 + kb8 + e;
            const int n = n0 + wn * TN * 16 + j * 16 + col;
            Cb[r * N + n] = (__bf16)(acc[i][j][e] * rf[j] * Asb[r]);
          }
      return;
    }
  }
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + wm * TM * 16 + i * 16 + kb8 + e;
        const int n = n0 + wn * TN * 16 + j * 16 + col;
        if (m < M && n < N) C[(size_t)m * N + n] = (__bf16)(acc[i][j][e] * rf[j] * As[m]);
      }
}


extern "C" __global__ __launch_bounds__(NTHREADS) void mxfp4_gemm_folded_tn2_wperm0_epifast1(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride) {
  saber_mxfp4_fp8_gemm_folded_impl<2, false, true>(
      A, W, Ws, Wref, As, C, M, N, K, pb1, pb2, astride);
}

extern "C" __global__ __launch_bounds__(NTHREADS) void mxfp4_gemm_folded_tn2_wperm0_epifast0(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride) {
  saber_mxfp4_fp8_gemm_folded_impl<2, false, false>(
      A, W, Ws, Wref, As, C, M, N, K, pb1, pb2, astride);
}

extern "C" __global__ __launch_bounds__(NTHREADS) void mxfp4_gemm_folded_tn2_wperm1_epifast1(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride) {
  saber_mxfp4_fp8_gemm_folded_impl<2, true, true>(
      A, W, Ws, Wref, As, C, M, N, K, pb1, pb2, astride);
}

extern "C" __global__ __launch_bounds__(NTHREADS) void mxfp4_gemm_folded_tn2_wperm1_epifast0(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride) {
  saber_mxfp4_fp8_gemm_folded_impl<2, true, false>(
      A, W, Ws, Wref, As, C, M, N, K, pb1, pb2, astride);
}

extern "C" __global__ __launch_bounds__(NTHREADS) void mxfp4_gemm_folded_tn4_wperm0_epifast1(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride) {
  saber_mxfp4_fp8_gemm_folded_impl<4, false, true>(
      A, W, Ws, Wref, As, C, M, N, K, pb1, pb2, astride);
}

extern "C" __global__ __launch_bounds__(NTHREADS) void mxfp4_gemm_folded_tn4_wperm0_epifast0(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride) {
  saber_mxfp4_fp8_gemm_folded_impl<4, false, false>(
      A, W, Ws, Wref, As, C, M, N, K, pb1, pb2, astride);
}

extern "C" __global__ __launch_bounds__(NTHREADS) void mxfp4_gemm_folded_tn4_wperm1_epifast1(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride) {
  saber_mxfp4_fp8_gemm_folded_impl<4, true, true>(
      A, W, Ws, Wref, As, C, M, N, K, pb1, pb2, astride);
}

extern "C" __global__ __launch_bounds__(NTHREADS) void mxfp4_gemm_folded_tn4_wperm1_epifast0(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride) {
  saber_mxfp4_fp8_gemm_folded_impl<4, true, false>(
      A, W, Ws, Wref, As, C, M, N, K, pb1, pb2, astride);
}
