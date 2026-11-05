// SABER-AUTHORED (workstream G6, 2026-09-24) -- NOT sliced from the reference. The G5 lab's A-tiled
// body with the M tile as a parameter (tools/capture/g5/g5_atiled_mt.h, itself G3's `_wm8s` /
// `_wm8sp` body with TM free), promoted to a shipped header. Included after mxfp4_common.h and
// mxfp4_atiled_wm.h (kMag, PAD, the fragment typedefs, `saber_atiled_lds_barrier`).
//
// The A-tiled MXFP4 W4A8 prefill GEMM (fragment-order A read straight into the WMMA operands,
// fragment-order W folded once per slab through kMag / v_perm into LDS) with the tile TM_ 16-row
// fragments per wave x WM_ waves down M, so BMF_ = WM_ * TM_ * 16 rows (the reference: TM 4 x WM 4).
// Per 16x16 output tile the arithmetic is the reference's, byte for byte: the same fragments, the
// same fold, the same WMMA chain over k ascending in 16-deep steps, the same EPIFAST epilogue. So it
// is one summation class with `mxfp4_gemm_atiled_tn2_lbk128_wperm1` AND (G2) with the folded
// kernel over the same A row-major (gated: tools/capture/g6/g6_bench.cpp).
// PIPE double-buffers the W slab's GLOBAL loads (the next slab's words and scales are in flight
// while this slab's WMMAs run).
//
// Why at mid M (G6): below 513 rows the rung ran the folded family (A staged through LDS), and the
// A-tiled body with no A stage is the fastest at 129..512 rows (G5's lab: 0.56-0.78x the folded
// reference at 320/384). The layout law (`a_tiled_mid_min_m`, rules/mxfp4_gfx1201.toml) lets the
// producers write the fragment layout at those rungs.
//
// CTRL (lab only; 0 in every shipped symbol): 1 = each slab's k-steps accumulated in reverse (an
// order change, must differ), 2 = W stored raw without the fold (must differ everywhere).
#pragma once

template <int TN, int LBK, int WM_, bool PIPE, int TM_, int CTRL = 0>
__device__ __forceinline__ void saber_mxfp4_atiled_mt_impl(
    const unsigned char *__restrict__ AT, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride) {
  constexpr int WN_ = 2;
  constexpr int NT_ = WM_ * WN_ * 32;
  constexpr int BMF_ = WM_ * TM_ * 16;
  constexpr int NS = LBK / 16;
  constexpr int LWSTR = LBK + PAD;
  constexpr int BNF_T = WN_ * TN * 16;
  constexpr int NBUF = PIPE ? 2 : 1;
  __shared__ unsigned char sW[NBUF * BNF_T * LWSTR];
  __shared__ unsigned int sMag[32];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN_, wn = wave % WN_;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int m0 = blockIdx.y * BMF_, n0 = blockIdx.x * BNF_T;
  {
    const int p_ = n0 >= pb2 ? 2 : (n0 >= pb1 ? 1 : 0);
    AT += (size_t)p_ * astride;
    As += (size_t)p_ * M;
  }
  const int ksteps_g = K / 16;
  const int Mt = (M + 15) >> 4;

  if (tid < 32) sMag[tid] = ((const unsigned int *)kMag)[tid];
  saber_atiled_lds_barrier();

  const unsigned char *abase[TM_];
#pragma unroll
  for (int i = 0; i < TM_; ++i) {
    int mt = (m0 >> 4) + wm * TM_ + i; mt = mt < Mt - 1 ? mt : Mt - 1;
    abase[i] = AT + (size_t)__builtin_amdgcn_readfirstlane(mt * ksteps_g * 256);
  }
  const unsigned int aoff = lane * 8;

  constexpr int KSTEPS_T = LBK / 16, NTILES_T = BNF_T / 16;
  constexpr int TOT_SLOTS = NTILES_T * KSTEPS_T * 32;
  constexpr int WSLOTS = (TOT_SLOTS >= NT_ * 4) ? 4 : 2;
  constexpr int NW = TOT_SLOTS / (NT_ * WSLOTS);
  static_assert(NW * NT_ * WSLOTS == TOT_SLOTS, "WPERM slot groups must tile");
  int wrow[NW], wcol[NW], wrc[NW], wkst[NW];
  unsigned int wofs[NW];  // byte offset of the slot group's W word at k-step 0 (fragment order)
#pragma unroll
  for (int it = 0; it < NW; ++it) {
    const int sl = it * NT_ * WSLOTS + tid * WSLOTS;
    const int lane_ = sl & 31, rest = sl >> 5;
    const int kst = rest % KSTEPS_T, ntl = rest / KSTEPS_T;
    wrow[it] = ntl * 16 + (lane_ & 15);
    wcol[it] = kst * 16 + (lane_ >> 4) * 8;
    const int gn = n0 + wrow[it];
    wrc[it] = gn < N - WSLOTS ? gn : N - WSLOTS;
    const int wlanec = (lane_ & 16) | (wrc[it] & 15);
    wkst[it] = kst;
    wofs[it] = (unsigned)((((wrc[it] >> 4) * ksteps_g + kst) * 32 + wlanec) * 4);
  }
  unsigned int wref[NW];
#pragma unroll
  for (int it = 0; it < NW; ++it) {
    if constexpr (WSLOTS == 4) wref[it] = *(const unsigned int *)(Wref + wrc[it]);
    else                       wref[it] = *(const unsigned short *)(Wref + wrc[it]);
  }

  floatx8 acc[TM_][TN];
#pragma unroll
  for (int i = 0; i < TM_; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

  unsigned int pk[NW][WSLOTS], wsv[NW];
  auto wload = [&](int k0) {
#pragma unroll
    for (int it = 0; it < NW; ++it) {
      const unsigned int off = wofs[it] + (unsigned)(k0 / 16) * 128u;
      if constexpr (WSLOTS == 4) {
        const uint4_t v = *(const uint4_t *)(W + off);
        pk[it][0] = v[0]; pk[it][1] = v[1]; pk[it][2] = v[2]; pk[it][3] = v[3];
      } else {
        const uint2_t v = *(const uint2_t *)(W + off);
        pk[it][0] = v[0]; pk[it][1] = v[1];
      }
      const unsigned int wso = (unsigned)(((k0 + wkst[it] * 16) / 32) * N + wrc[it]);
      if constexpr (WSLOTS == 4) wsv[it] = *(const unsigned int *)(Ws + wso);
      else                       wsv[it] = *(const unsigned short *)(Ws + wso);
    }
  };
  auto wstore = [&](unsigned char *sWb) {
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
        if constexpr (CTRL == 2) {
          *(uint2_t *)(&sWb[(wrow[it] + q) * LWSTR + wcol[it]]) = uint2_t{wq, wq ^ 0x5a5a5a5au};
        } else {
          *(uint2_t *)(&sWb[(wrow[it] + q) * LWSTR + wcol[it]]) =
              uint2_t{__builtin_amdgcn_perm(bo, be, 0x05010400u),
                      __builtin_amdgcn_perm(bo, be, 0x07030602u)};
        }
      }
  };

  if constexpr (PIPE) wload(0);
  int buf = 0;
  for (int k0 = 0; k0 < K; k0 += LBK) {
    __builtin_amdgcn_s_setprio(2);
    unsigned char *sWb = sW + (PIPE ? buf * (BNF_T * LWSTR) : 0);
    if constexpr (!PIPE) wload(k0);
    wstore(sWb);
    __builtin_amdgcn_sched_barrier(0);
    int2_t af[TM_][NS];
#pragma unroll
    for (int i = 0; i < TM_; ++i)
#pragma unroll
      for (int st = 0; st < NS; ++st)
        af[i][st] = *(const int2_t *)(abase[i] + (aoff + (unsigned int)(k0 * 16) + (unsigned int)(st * 256)));
    if constexpr (PIPE) { if (k0 + LBK < K) wload(k0 + LBK); }
    saber_atiled_lds_barrier();

#pragma unroll
    for (int step_ = 0; step_ < NS; ++step_) {
      const int step = CTRL == 1 ? (NS - 1 - step_) : step_;
      const int kk = step * 16 + kb8;
      int2_t wf[TN];
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const unsigned char *p = &sWb[(wn * TN * 16 + j * 16 + col) * LWSTR + kk];
        wf[j][0] = *(const int *)p; wf[j][1] = *(const int *)(p + 4);
      }
      __builtin_amdgcn_sched_barrier(0);
      if (step_ == 0) __builtin_amdgcn_s_setprio(0);
#pragma unroll
      for (int i = 0; i < TM_; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i][step], wf[j], acc[i][j]);
    }
    if constexpr (!PIPE) saber_atiled_lds_barrier();
    buf ^= 1;
  }

  float rf[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int n = n0 + wn * TN * 16 + j * 16 + col;
    rf[j] = (n < N) ? __int_as_float((int)Wref[n] << 23) : 0.f;
  }
  {
    __bf16 *__restrict__ Cb = C + (size_t)(m0 + wm * TM_ * 16) * N;
    const float *__restrict__ Asb = As + m0 + wm * TM_ * 16;
    const bool full = (m0 + wm * TM_ * 16 + (TM_ - 1) * 16 + kb8 + 7 < M) &&
                      (n0 + wn * TN * 16 + (TN - 1) * 16 + col < N);
    if (full) {
#pragma unroll
      for (int i = 0; i < TM_; ++i)
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
  for (int i = 0; i < TM_; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + wm * TM_ * 16 + i * 16 + kb8 + e;
        const int n = n0 + wn * TN * 16 + j * 16 + col;
        if (m < M && n < N) C[(size_t)m * N + n] = (__bf16)(acc[i][j][e] * rf[j] * As[m]);
      }
}
