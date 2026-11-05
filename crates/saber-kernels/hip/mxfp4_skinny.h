// SABER-AUTHORED (workstream G6, 2026-09-24) -- NOT sliced from the reference. Included after
// mxfp4_common.h (kMag, the fragment typedefs).
//
// The skinny-N MXFP4 W4A8 prefill GEMM: the 48-wide GDN gate (`in_proj_ba`, N = 48 per TP=2 shard,
// K = 5120). Every prefill tile the reference has for it is ONE workgroup per 256 rows (the folded
// and A-tiled tiles cover 64 columns, the gate has 48), so the whole K walk runs serially in one or
// two workgroups: 65-68 us at every rung, 3.1 ms a chunk (G5).
//
// This body spreads the SAME work over one wave per (16-row fragment, TNS 16-column tiles): M / 16
// waves instead of one workgroup. Each wave walks all of K itself, in the reference's order, so no
// partial sum is ever formed and no reduce exists: per 16x16 output tile the arithmetic is the
// reference's byte for byte --
//   * A: the lane's 8-byte WMMA operand for k-step ks (fragment order: `AT + (mt*ksteps + ks)*256 +
//     lane*8`, the A-tiled kernel's `abase + k0*16 + st*256 + lane*8`; row-major: row mt*16 +
//     (lane & 15) clamped to M - 1 as the folded stage clamps, bytes ks*16 + (lane >> 4)*8 .. +7);
//   * W: the fragment-order word the reference's WPERM staging puts in the same lane
//     (`((ntile*ksteps + ks)*32 + lane)*4`), folded by the SAME kMag / v_perm sequence against the
//     SAME per-32-block exponent `Ws[(ks/2)*N + n]` and row reference `Wref[n]` -- in registers
//     instead of through an LDS slab, which moves the bytes' path and not their values;
//   * the WMMA chain: acc = wmma(a, w, acc) over k ascending in 16-deep steps from zero;
//   * the EPIFAST epilogue: C = bf16(acc * 2^(Wref - 127) * As[m]).
// So it is one summation class with the folded and A-tiled tiles (gated 0 bytes:
// tools/capture/g6/g6_bench.cpp, with k-reversal / no-fold / W-flip controls).
//
// TILED: A in fragment order (a rung at or above the layout law's edge) or row-major (the rungs
// below it). TNS: 16-column tiles per wave (1 or 3). WPB: waves per workgroup (independent rows).
// UNR: k-steps per load batch; the next batch's loads are issued before this batch's WMMAs.
// CTRL (lab only; 0 in every shipped symbol): 1 = each batch's k-steps accumulated in reverse,
// 2 = W raw without the fold.
#pragma once

template <int TNS, bool TILED, int WPB, int UNR, int CTRL = 0>
__device__ __forceinline__ void saber_mxfp4_skinny_impl(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride) {
  __shared__ unsigned int sMag[32];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  if (tid < 32) sMag[tid] = ((const unsigned int *)kMag)[tid];
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");

  const int Mt = (M + 15) >> 4;
  const int mt = blockIdx.y * WPB + wave;
  if (mt >= Mt) return;
  const int n0 = blockIdx.x * TNS * 16;
  {
    const int p_ = n0 >= pb2 ? 2 : (n0 >= pb1 ? 1 : 0);
    A += (size_t)p_ * astride;
    As += (size_t)p_ * M;
  }
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int ksteps = K / 16;
  const int ntiles = N >> 4;

  const unsigned char *abase;
  if constexpr (TILED) {
    abase = A + (size_t)mt * ksteps * 256 + lane * 8;
  } else {
    int r = mt * 16 + col;
    r = r < M - 1 ? r : M - 1;
    abase = A + (size_t)r * K + kb8;
  }
  const unsigned int *wbase[TNS];
  const unsigned char *wsbase[TNS];
  unsigned int wref[TNS];
  int ncol[TNS];
#pragma unroll
  for (int j = 0; j < TNS; ++j) {
    int nt = (n0 >> 4) + j;
    nt = nt < ntiles - 1 ? nt : ntiles - 1;
    ncol[j] = nt * 16 + col;
    wbase[j] = (const unsigned int *)W + (size_t)nt * ksteps * 32 + lane;
    wsbase[j] = Ws + ncol[j];
    wref[j] = Wref[ncol[j]];
  }

  floatx8 acc[TNS];
#pragma unroll
  for (int j = 0; j < TNS; ++j)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[j][e] = 0.f;

  int2_t av[2][UNR];
  unsigned int wv[2][UNR][TNS], sv[2][UNR][TNS];
  auto load = [&](int b, int ks0) {
#pragma unroll
    for (int u = 0; u < UNR; ++u) {
      const int ks = ks0 + u;
      if constexpr (TILED) av[b][u] = *(const int2_t *)(abase + (size_t)ks * 256);
      else                 av[b][u] = *(const int2_t *)(abase + (size_t)ks * 16);
#pragma unroll
      for (int j = 0; j < TNS; ++j) {
        wv[b][u][j] = wbase[j][(size_t)ks * 32];
        sv[b][u][j] = wsbase[j][(size_t)(ks >> 1) * N];
      }
    }
  };
  auto compute = [&](int b) {
#pragma unroll
    for (int u_ = 0; u_ < UNR; ++u_) {
      const int u = CTRL == 1 ? (UNR - 1 - u_) : u_;
#pragma unroll
      for (int j = 0; j < TNS; ++j) {
        int d = (int)wref[j] - (int)sv[b][u][j];
        d = d < 0 ? 0 : (d > 15 ? 15 : d);
        const unsigned int t0 = sMag[d * 2], t1 = sMag[d * 2 + 1];
        const unsigned int wq = wv[b][u][j];
        const unsigned int ev = wq & 0x0F0F0F0Fu;
        const unsigned int od = (wq >> 4) & 0x0F0F0F0Fu;
        const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u)
                              | ((ev & 0x08080808u) << 4);
        const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u)
                              | ((od & 0x08080808u) << 4);
        int2_t wf;
        if constexpr (CTRL == 2) {
          wf[0] = (int)wq; wf[1] = (int)(wq ^ 0x5a5a5a5au);
        } else {
          wf[0] = (int)__builtin_amdgcn_perm(bo, be, 0x05010400u);
          wf[1] = (int)__builtin_amdgcn_perm(bo, be, 0x07030602u);
        }
        acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(av[b][u], wf, acc[j]);
      }
    }
  };

  // K % (16 * UNR) == 0 is the launcher's precondition (K = 5120 at every gate shape).
  // Two buffers, each named by a constant (a runtime buffer index would put them in scratch).
  load(0, 0);
  for (int ks0 = 0; ks0 < ksteps; ks0 += 2 * UNR) {
    if (ks0 + UNR < ksteps) load(1, ks0 + UNR);
    compute(0);
    if (ks0 + UNR >= ksteps) break;
    if (ks0 + 2 * UNR < ksteps) load(0, ks0 + 2 * UNR);
    compute(1);
  }

  float rf[TNS];
#pragma unroll
  for (int j = 0; j < TNS; ++j) {
    const int n = n0 + j * 16 + col;
    rf[j] = (n < N) ? __int_as_float((int)Wref[n] << 23) : 0.f;
  }
#pragma unroll
  for (int j = 0; j < TNS; ++j)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int m = mt * 16 + kb8 + e;
      const int n = n0 + j * 16 + col;
      if (m < M && n < N) C[(size_t)m * N + n] = (__bf16)(acc[j][e] * rf[j] * As[m]);
    }
}
