// SABER-AUTHORED (workstream G1, 2026-09-23) -- NOT sliced from the reference. Included at the end
// of the generated mxfp4_gemm_atiled.hip by tools/gen_hip_sources.py, after the reference's own
// A-tiled kernel, so kMag, PAD and the fragment typedefs are the reference's.
//
// The A-tiled prefill GEMM (fragment-order W, WPERM=1 only) with WM waves down M instead of the
// reference's four: WM=8 is a 512-row tile of 16 waves / 512 threads. Per 16x16 output tile the
// arithmetic is the reference's, byte for byte -- the same fragments, the same kMag/v_perm fold of
// the same W slab, the same WMMA chain over k in the same order, the same EPIFAST epilogue -- so the
// output is bit-identical (gated: tools/capture/g1/lab_bench.cpp, 13 shapes incl. ragged M).
// What changes is only how the work is spread:
//
//   * the W slab (128 rows x 64 k) is folded ONCE per 512 rows of A instead of once per 256, so
//     the fold's VALU per WMMA halves. On gfx1201 the WMMA is itself a VALU op and the fold does
//     not hide behind it (G1's ladder: removing the fold alone is -9.4% of the loop);
//   * the A fragment loads are issued AFTER the fold and the LDS store (a scheduling barrier
//     between them), so the fold's temporaries are dead before the 32 A registers go live:
//     191 VGPRs (the reference's tile: 205), 8 waves/SIMD, i.e. 2 of these blocks per WGP;
//   * `s_setprio 2` across the staging, back to 0 at the first WMMA of the slab, so a wave that is
//     issuing loads and the fold is not starved behind a neighbour's WMMA run.
//
// Measured (G1 hold 2, card 1, 210 W, IC-cold rotate + 256 MiB scrub, 100 interleaved reps x 2
// passes, M = 2048 TP=2 shards): 0.943-0.950x the reference's time on gate_up 17408x5120,
// 0.948-0.964 down 5120x8704, 0.955 in_proj 8192x5120, 0.964-0.967 qkv 7168x5120, 0.976-0.991
// o_proj 5120x3072. G2 (card 1, 2 passes) reads it 0.972-0.986x at M = 4096 / 8192 too, so the
// rules bind it on M 2048..8192 ([[atiled_wm]] in rules/mxfp4_gfx1201.toml; LEDGER G2).
static __device__ __forceinline__ void saber_atiled_lds_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

template <int TN, int LBK, int WM_>
__device__ __forceinline__ void saber_mxfp4_atiled_wm_impl(
    const unsigned char *__restrict__ AT, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride) {
  constexpr int TM_ = 4, WN_ = 2;
  constexpr int NT_ = WM_ * WN_ * 32;
  constexpr int BMF_ = WM_ * TM_ * 16;
  constexpr int NS = LBK / 16;
  constexpr int LWSTR = LBK + PAD;
  constexpr int BNF_T = WN_ * TN * 16;
  __shared__ unsigned char sW[BNF_T * LWSTR];
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

  // W staging geometry, fragment order: the reference's WPERM arithmetic at this block's width.
  constexpr int KSTEPS_T = LBK / 16, NTILES_T = BNF_T / 16;
  constexpr int TOT_SLOTS = NTILES_T * KSTEPS_T * 32;
  constexpr int WSLOTS = (TOT_SLOTS >= NT_ * 4) ? 4 : 2;
  constexpr int NW = TOT_SLOTS / (NT_ * WSLOTS);
  static_assert(NW * NT_ * WSLOTS == TOT_SLOTS, "WPERM slot groups must tile");
  int wrow[NW], wcol[NW], wrc[NW], wkst[NW], wlanec[NW];
#pragma unroll
  for (int it = 0; it < NW; ++it) {
    const int sl = it * NT_ * WSLOTS + tid * WSLOTS;
    const int lane_ = sl & 31, rest = sl >> 5;
    const int kst = rest % KSTEPS_T, ntl = rest / KSTEPS_T;
    wrow[it] = ntl * 16 + (lane_ & 15);
    wcol[it] = kst * 16 + (lane_ >> 4) * 8;
    const int gn = n0 + wrow[it];
    wrc[it] = gn < N - WSLOTS ? gn : N - WSLOTS;
    wlanec[it] = (lane_ & 16) | (wrc[it] & 15);
    wkst[it] = kst;
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

  const unsigned int *__restrict__ Wp = (const unsigned int *)W;

  for (int k0 = 0; k0 < K; k0 += LBK) {
    __builtin_amdgcn_s_setprio(2);
    // ---- W slab -> LDS, folded to e4m3 (the reference's kMag / v_perm fold, unchanged) ----
    uint4_t pk[NW]; unsigned int wsv[NW];
    {
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
    }
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
    // ---- A fragments straight from global, issued only now (the fold's registers are dead) ----
    __builtin_amdgcn_sched_barrier(0);
    int2_t af[TM_][NS];
#pragma unroll
    for (int i = 0; i < TM_; ++i)
#pragma unroll
      for (int st = 0; st < NS; ++st)
        af[i][st] = *(const int2_t *)(abase[i] + (aoff + (unsigned int)(k0 * 16) + (unsigned int)(st * 256)));
    saber_atiled_lds_barrier();

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
      if (step == 0) __builtin_amdgcn_s_setprio(0);
#pragma unroll
      for (int i = 0; i < TM_; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i][step], wf[j], acc[i][j]);
    }
    saber_atiled_lds_barrier();
  }

  // Epilogue: the reference's EPIFAST path (bit-identical by construction).
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

// SABER-AUTHORED (workstream G3, 2026-09-23). The same 512-row tile with the W / Ws addressing
// taken off the VALU's carry chain: both loads use the SADDR form (the kernarg base in SGPRs plus a
// loop-invariant 32-bit lane offset that advances by one `v_add_nc_u32` per slab), where the body
// above builds a 64-bit VGPR pointer per load and advances it with `v_add_co_u32` +
// `v_add_co_ci_u32`, each followed by an `s_wait_alu depctr_va_vcc(0)` at the end of every slab.
// The loop drops from 33 non-WMMA VALU (158 instructions) to 31 (155) and loses both VCC stalls.
// PIPE = 1 double-buffers the W slab in LDS (the next slab's W / Ws loaded before this slab's
// WMMAs, one barrier per slab); the wrapper caps the tile at 192 VGPRs (8 waves/SIMD, 2 blocks a
// WGP) with amdgpu_waves_per_eu, which G1's double-buffered arm (205 VGPRs) did not have.
// Per 16x16 output tile the arithmetic is the body above's -- same fragments, same kMag / v_perm
// fold, same WMMA chain in the same k order, same epilogue -- so the bytes are the same (gated:
// tools/capture/g3/g3_bench.cpp, 13 shapes; `rules::summation_class` names it with the reference).
template <int TN, int LBK, int WM_, bool PIPE>
__device__ __forceinline__ void saber_mxfp4_atiled_wm_s_impl(
    const unsigned char *__restrict__ AT, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride) {
  constexpr int TM_ = 4, WN_ = 2;
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
        *(uint2_t *)(&sWb[(wrow[it] + q) * LWSTR + wcol[it]]) =
            uint2_t{__builtin_amdgcn_perm(bo, be, 0x05010400u),
                    __builtin_amdgcn_perm(bo, be, 0x07030602u)};
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
    for (int step = 0; step < NS; ++step) {
      const int kk = step * 16 + kb8;
      int2_t wf[TN];
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const unsigned char *p = &sWb[(wn * TN * 16 + j * 16 + col) * LWSTR + kk];
        wf[j][0] = *(const int *)p; wf[j][1] = *(const int *)(p + 4);
      }
      __builtin_amdgcn_sched_barrier(0);
      if (step == 0) __builtin_amdgcn_s_setprio(0);
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
