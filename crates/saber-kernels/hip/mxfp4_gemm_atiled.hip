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


// ---------------------------------------------------------------- A-tiled prefill kernel
// Same tile, same W staging, same epilogue as the folded kernel above, but the activation
// arrives in WMMA-FRAGMENT-TILED layout: each 16m x 16k fp8 fragment is 256 contiguous bytes in
// lane order (lane l owns bytes 8l..8l+7 = A[mt*16 + l%16][ks*16 + (l/16)*8 .. +7]), the layout
// saber_add_rms_quant / saber_silu_mul_quant emit under SABER_MXFP4_A_TILED_MIN_M. One
// coalesced global_load_b64 per fragment per wave lands straight in the af[] register the WMMA
// reads, so sA and its staging (the largest single cost of the folded kernel: ablated at 24-32%)
// are gone; only W goes through LDS (9-18 KB), so 3 blocks fit a CU instead of 2.
//
// Measured 2026-09-01 (~/mxfp4_work/tier7/lf.hip, DRAM-fed, prod server up, bit-identical on all
// six shapes x M tails x TN): vs the shipped folded kernel on the checkpoint layout, 0.84-0.87x
// at M=2048 (TN=4), 0.87-0.91x at M=1024 (TN=2), 0.878x at out M=8192 and 0.868x at down M=4096
// -- 215-220 TF/s against 190. The fragment-order (WPERM) layout is within 1% of that, so this
// kernel also removes the old "wp1 costs prefill" trade. Rejected on the way, all measured:
// W direct-to-register with an in-register fold (no LDS at all) 1.10-1.30x SLOWER at every
// shape -- the fold is then done by all WM=4 waves and the 4 B/lane W loads add instructions;
// a register-pipelined A prefetch (next slab in flight) 1.05-1.10x slower than this plain
// form at 250 VGPRs, and it spills at TN=4 with a 64-k slab (93 VGPRs, 2.2-4.9x).
// LBK: K per slab. With no A tile in LDS a 128-deep W slab is affordable (17 KB at TN=4, 3
// blocks/CU) and halves the barriers per K. Measured 2026-09-02 (tier7 lf.hip, bit-identical):
// at TN=2 (M < 2048) LBK=128 is 2-4% faster than 64 on every shape (gate48 -22%); at TN=4 it
// costs registers (246 VGPRs, occupancy 5) and loses 12-17%, so the launcher keys LBK on TN.
//
// ROUND 2 (2026-09-03, ~/mxfp4_work/tier7/at2.{h,hip}, abl.hip). An ablation of this kernel
// (WMMA count held, one piece removed at a time, TN=4 M=4096) priced its non-WMMA time: A
// fragment loads 28% (of which ~18% is the L2->L0 stream and ~10% issue/latency), W staging
// 14% (8% of it the fold VALU), LDS fragment reads 11%, barriers 1%; WMMA-only runs at 310
// TF/s = 55% of the full time, and WMMA-only + everything-else sum to ~90%, i.e. the memory
// work barely overlaps the matrix work. Three codegen fixes, each bit-identical, together
// 0.94-0.96x paired vs the 09-02 kernel on every shape at BOTH bands and both W layouts:
//   * SGPR bases for the A fragments (readfirstlane on the tile offset) with an UNSIGNED 32-bit
//     lane offset, so the 16 loads compile to one s_clause of SADDR global_load_b64 with
//     immediate offsets instead of eight 64-bit v_add_co/v_add_co_ci chains and their
//     s_wait_alu stalls (main loop VALU 63 -> ~47 per 64 WMMA).
//   * kMag served from LDS. As a global (constant) load it is a DEPENDENT load issued after the
//     A fragments, and loadcnt is in order, so `s_wait_loadcnt 0x0` before the fold drained the
//     whole A batch every slab. A ds_load has its own counter.
//   * LDS-only barrier fences. __syncthreads() is a workgroup-scope acq/rel on ALL address
//     spaces: `global_inv` plus a full loadcnt drain, twice per slab. Nothing in global memory
//     is shared between waves here, so the fence is scoped to LDS (s_wait_dscnt + s_barrier).
// Priced and REJECTED in the same round (all gated bit-identical): A fragments software-
// pipelined one slab ahead (loop unrolled by two, 230 VGPRs at TN=4): neutral on top of the
// three fixes at TN=4 and -2..-5% at TN=2 from the occupancy loss (the A cost is the L2 stream,
// not latency -- a cache-hot A ablation is still 10% off no-A); prefetching next slab's W into
// registers: +-1%. Every rebalancing of the tile trades one memory stream for another (TN=8 is
// the accumulator wall; A shared via LDS is the folded kernel; TM=3/TN=6 nets ~0), so the
// remaining ~40% over the WMMA floor is structural at this tile.
// Served 2026-09-04 (serve-mxfp4.sh defaults, BetterBench quick --prefill, same-day paired vs
// the 09-02 kernel): PP t/s 5141->5274 @2k, 5283->5385 @8k, 5109->5203 @16k, 4937->5046 @32k,
// 4686->4794 @64k (+1.8..+2.6%); decode untouched (this kernel only runs at M >= 513).
static __device__ __forceinline__ void saber_lds_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

template <int TN, bool WPERM, int LBK = BK>
__device__ __forceinline__ void saber_mxfp4_fp8_gemm_atiled_impl(
    const unsigned char *__restrict__ AT, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride) {
  constexpr int NS = LBK / 16;
  constexpr int LWSTR = LBK + PAD;
  constexpr int BNF_T = WN * TN * 16;
  constexpr int NIT = BNF_T * LBK / 2 / (NTHREADS * 8);   // row-major W staging iterations per slab
  static_assert(NIT >= 1 && NIT * NTHREADS * 8 == BNF_T * LBK / 2, "W tile must be 8 B/thread multiples");
  __shared__ unsigned char sW[BNF_T * LWSTR];
  __shared__ unsigned int sMag[32];   // kMag, 128 B: ds_load keeps the lookup off the loadcnt chain

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN, wn = wave % WN;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int m0 = blockIdx.y * BMF, n0 = blockIdx.x * BNF_T;
  // Merged-linear partition select (paro: one launch over the whole N, partition p's n-blocks
  // read the p-th rotated copy of A and its scales). Boundaries are multiples of every n-tile
  // (launcher-checked), so a block never straddles one. pb1 = pb2 = 1<<30 (the default) is p = 0.
  {
    const int p_ = n0 >= pb2 ? 2 : (n0 >= pb1 ? 1 : 0);
    AT += (size_t)p_ * astride;
    As += (size_t)p_ * M;
  }
  const int ksteps_g = K / 16;
  const int Mt = (M + 15) >> 4;

  if (tid < 32) sMag[tid] = ((const unsigned int *)kMag)[tid];
  saber_lds_barrier();

  // A tile bases, one per M-fragment: wave-uniform, forced into SGPRs; the lane part is an
  // unsigned 32-bit offset so the loads take the SADDR form (max offset < 2^31 for all prod).
  // Tile-granular clamp, never predicate: rows of a clamped tile only reach accumulators the
  // epilogue drops.
  const unsigned char *abase[TM];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int mt = (m0 >> 4) + wm * TM + i; mt = mt < Mt - 1 ? mt : Mt - 1;
    abase[i] = AT + (size_t)__builtin_amdgcn_readfirstlane(mt * ksteps_g * 256);
  }
  const unsigned int aoff = lane * 8;

  // W staging geometry, hoisted out of the K loop.
  //   Row-major (checkpoint) layout: row r, packed-byte column c, clamped row rc; 8 B/thread.
  //   Fragment order (WPERM): slot s of the tile is one lane's four packed bytes (eight
  //   elements) for one (n-tile, k-step), so consecutive threads read consecutive uint32 and a
  //   wave takes 128 contiguous bytes. WSLOTS adjacent lane slots per thread; adjacent slots
  //   differ only in the row (lane index is half*16 + row), so one vector read covers WSLOTS
  //   adjacent rows at the same k and the wave still reads its bytes contiguously. WSLOTS is 4
  //   (16 B) whenever the tile has the slots for it, else 2 (8 B); a 4-slot group never
  //   straddles the k-half boundary because 16 % 4 == 0. Clamp to the last WSLOTS-aligned row
  //   (N % 16 == 0 under WPERM) so the vector read stays aligned and every gc + q is in bounds.
  constexpr int KSTEPS_T = LBK / 16, NTILES_T = BNF_T / 16;
  constexpr int TOT_SLOTS = NTILES_T * KSTEPS_T * 32;
  constexpr int WSLOTS = (TOT_SLOTS >= NTHREADS * 4) ? 4 : 2;
  constexpr int NITW = TOT_SLOTS / (NTHREADS * WSLOTS);
  static_assert(!WPERM || NITW * NTHREADS * WSLOTS == TOT_SLOTS, "WPERM slot groups must tile");
  constexpr int NW = WPERM ? NITW : NIT;
  int wrow[NW], wcol[NW], wrc[NW], wkst[NW], wlanec[NW];
#pragma unroll
  for (int it = 0; it < NW; ++it) {
    if constexpr (WPERM) {
      const int sl = it * NTHREADS * WSLOTS + tid * WSLOTS;
      const int lane_ = sl & 31, rest = sl >> 5;
      const int kst = rest % KSTEPS_T, ntl = rest / KSTEPS_T;
      wrow[it] = ntl * 16 + (lane_ & 15);
      wcol[it] = kst * 16 + (lane_ >> 4) * 8;          // kloc
      const int gn = n0 + wrow[it];
      wrc[it] = gn < N - WSLOTS ? gn : N - WSLOTS;      // global clamped row
      wlanec[it] = (lane_ & 16) | (wrc[it] & 15);
      wkst[it] = kst;
    } else {
      const int idx = it * NTHREADS * 8 + tid * 8;
      wrow[it] = idx / (LBK / 2); wcol[it] = idx % (LBK / 2);
      wrc[it] = wrow[it] < N - 1 - n0 ? wrow[it] : N - 1 - n0;
      wkst[it] = 0; wlanec[it] = 0;
    }
  }
  unsigned int wref[NW];   // per-row max exponent(s): one byte, or WSLOTS packed under WPERM
#pragma unroll
  for (int it = 0; it < NW; ++it) {
    if constexpr (WPERM) {
      if constexpr (WSLOTS == 4) wref[it] = *(const unsigned int *)(Wref + wrc[it]);
      else                       wref[it] = *(const unsigned short *)(Wref + wrc[it]);
    } else wref[it] = Wref[n0 + wrc[it]];
  }

  floatx8 acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

  for (int k0 = 0; k0 < K; k0 += LBK) {
    // A fragments straight from global: one SADDR clause, consumed only at the WMMAs.
    int2_t af[TM][NS];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int st = 0; st < NS; ++st)
        af[i][st] = *(const int2_t *)(abase[i] + (aoff + (unsigned int)(k0 * 16) + (unsigned int)(st * 256)));

    // ---- W tile -> LDS, folded to e4m3 (kMag[d] via v_perm, see the folded kernel) ----
    uint4_t pk[NW]; unsigned int wsv[NW];
    if constexpr (WPERM) {
      const unsigned int *__restrict__ Wp = (const unsigned int *)W;
      const int kstep0 = k0 / 16;
#pragma unroll
      for (int it = 0; it < NW; ++it) {
        const unsigned int *src = &Wp[((size_t)(wrc[it] >> 4) * ksteps_g + kstep0 + wkst[it]) * 32 + wlanec[it]];
        if constexpr (WSLOTS == 4) pk[it] = *(const uint4_t *)src;
        else { const uint2_t v = *(const uint2_t *)src; pk[it] = uint4_t{v[0], v[1], 0u, 0u}; }
        const int blk = (k0 + wkst[it] * 16) / 32;
        if constexpr (WSLOTS == 4) wsv[it] = *(const unsigned int *)(Ws + (size_t)blk * N + wrc[it]);
        else                       wsv[it] = *(const unsigned short *)(Ws + (size_t)blk * N + wrc[it]);
      }
    } else {
      const unsigned char *__restrict__ Wb = W + (size_t)__builtin_amdgcn_readfirstlane((int)(((size_t)n0 * K + k0) / 2));
      const unsigned char *__restrict__ Wsb = Ws + (size_t)__builtin_amdgcn_readfirstlane((k0 / 32) * N + n0);
#pragma unroll
      for (int it = 0; it < NW; ++it) {
        const uint2_t v = *(const uint2_t *)(Wb + (unsigned int)(wrc[it] * (K / 2) + wcol[it]));
        pk[it] = uint4_t{v[0], v[1], 0u, 0u};
        wsv[it] = Wsb[(unsigned int)(((wcol[it] * 2) / 32) * N + wrc[it])];
      }
    }
    if constexpr (WPERM) {
#pragma unroll
      for (int it = 0; it < NW; ++it)
#pragma unroll
        for (int q = 0; q < WSLOTS; ++q) {
          int d = (int)((wref[it] >> (8 * q)) & 0xFF) - (int)((wsv[it] >> (8 * q)) & 0xFF);
          d = d < 0 ? 0 : (d > 15 ? 15 : d);
          const uint2_t t = *(const uint2_t *)&sMag[d * 2];
          const unsigned int wq = pk[it][q];
          const unsigned int ev = wq & 0x0F0F0F0Fu;
          const unsigned int od = (wq >> 4) & 0x0F0F0F0Fu;
          const unsigned int be = __builtin_amdgcn_perm(t[1], t[0], ev & 0x07070707u)
                                | ((ev & 0x08080808u) << 4);
          const unsigned int bo = __builtin_amdgcn_perm(t[1], t[0], od & 0x07070707u)
                                | ((od & 0x08080808u) << 4);
          *(uint2_t *)(&sW[(wrow[it] + q) * LWSTR + wcol[it]]) =
              uint2_t{__builtin_amdgcn_perm(bo, be, 0x05010400u),
                      __builtin_amdgcn_perm(bo, be, 0x07030602u)};
        }
    } else {
#pragma unroll
      for (int it = 0; it < NW; ++it) {
        int dsh = (int)wref[it] - (int)wsv[it];
        dsh = dsh < 0 ? 0 : (dsh > 15 ? 15 : dsh);
        const uint2_t t = *(const uint2_t *)&sMag[dsh * 2];
        unsigned int outw[4];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const unsigned int w = pk[it][h];
          const unsigned int ev = w & 0x0F0F0F0Fu;
          const unsigned int od = (w >> 4) & 0x0F0F0F0Fu;
          const unsigned int be = __builtin_amdgcn_perm(t[1], t[0], ev & 0x07070707u)
                                | ((ev & 0x08080808u) << 4);
          const unsigned int bo = __builtin_amdgcn_perm(t[1], t[0], od & 0x07070707u)
                                | ((od & 0x08080808u) << 4);
          outw[2 * h]     = __builtin_amdgcn_perm(bo, be, 0x05010400u);
          outw[2 * h + 1] = __builtin_amdgcn_perm(bo, be, 0x07030602u);
        }
        *(uint4_t *)(&sW[wrow[it] * LWSTR + wcol[it] * 2]) = uint4_t{outw[0], outw[1], outw[2], outw[3]};
      }
    }
    saber_lds_barrier();

#pragma unroll
    for (int step = 0; step < NS; ++step) {
      const int kk = step * 16 + kb8;
      int2_t wf[TN];
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const unsigned char *p = &sW[(wn * TN * 16 + j * 16 + col) * LWSTR + kk];
        wf[j][0] = *(const int *)p; wf[j][1] = *(const int *)(p + 4);
      }
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i][step], wf[j], acc[i][j]);
    }
    saber_lds_barrier();
  }

  // Epilogue: identical to the folded kernel's EPIFAST path (bit-identical by construction).
  float rf[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int n = n0 + wn * TN * 16 + j * 16 + col;
    rf[j] = (n < N) ? __int_as_float((int)Wref[n] << 23) : 0.f;
  }
  {
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


extern "C" __global__ __launch_bounds__(NTHREADS) void mxfp4_gemm_atiled_tn2_lbk128_wperm0(
    const unsigned char *__restrict__ AT, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride) {
  saber_mxfp4_fp8_gemm_atiled_impl<2, false, 128>(
      AT, W, Ws, Wref, As, C, M, N, K, pb1, pb2, astride);
}

extern "C" __global__ __launch_bounds__(NTHREADS) void mxfp4_gemm_atiled_tn2_lbk128_wperm1(
    const unsigned char *__restrict__ AT, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride) {
  saber_mxfp4_fp8_gemm_atiled_impl<2, true, 128>(
      AT, W, Ws, Wref, As, C, M, N, K, pb1, pb2, astride);
}

extern "C" __global__ __launch_bounds__(NTHREADS) void mxfp4_gemm_atiled_tn4_lbk64_wperm0(
    const unsigned char *__restrict__ AT, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride) {
  saber_mxfp4_fp8_gemm_atiled_impl<4, false, 64>(
      AT, W, Ws, Wref, As, C, M, N, K, pb1, pb2, astride);
}

extern "C" __global__ __launch_bounds__(NTHREADS) void mxfp4_gemm_atiled_tn4_lbk64_wperm1(
    const unsigned char *__restrict__ AT, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride) {
  saber_mxfp4_fp8_gemm_atiled_impl<4, true, 64>(
      AT, W, Ws, Wref, As, C, M, N, K, pb1, pb2, astride);
}

// ---------------------------------------------------------------- SABER-AUTHORED (G1)
// Everything below is NOT from the reference: see mxfp4_atiled_wm.h.
#include "mxfp4_atiled_wm.h"

extern "C" __global__ __launch_bounds__(512) void mxfp4_gemm_atiled_tn4_lbk64_wperm1_wm8(
    const unsigned char *__restrict__ AT, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride) {
  saber_mxfp4_atiled_wm_impl<4, 64, 8>(
      AT, W, Ws, Wref, As, C, M, N, K, pb1, pb2, astride);
}

extern "C" __global__ __launch_bounds__(512)
__attribute__((amdgpu_waves_per_eu(8, 8))) void mxfp4_gemm_atiled_tn4_lbk64_wperm1_wm8s(
    const unsigned char *__restrict__ AT, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride) {
  saber_mxfp4_atiled_wm_s_impl<4, 64, 8, false>(
      AT, W, Ws, Wref, As, C, M, N, K, pb1, pb2, astride);
}

extern "C" __global__ __launch_bounds__(512)
__attribute__((amdgpu_waves_per_eu(8, 8))) void mxfp4_gemm_atiled_tn4_lbk64_wperm1_wm8sp(
    const unsigned char *__restrict__ AT, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride) {
  saber_mxfp4_atiled_wm_s_impl<4, 64, 8, true>(
      AT, W, Ws, Wref, As, C, M, N, K, pb1, pb2, astride);
}
