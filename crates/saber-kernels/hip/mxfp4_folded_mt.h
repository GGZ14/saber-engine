// SABER-AUTHORED (workstream G5, 2026-09-23) -- NOT sliced from the reference. Included after the
// reference's folded kernel (so kMag, BK, PAD, ASTR, WSTR and the fragment typedefs are the
// reference's), or by the G5 lab bench.
//
// The folded MXFP4 W4A8 prefill GEMM (row-major fp8 A staged through LDS, fragment-order W folded
// once per slab through kMag / v_perm) with the M tile as a template: TM_ 16-row fragments per wave
// and WM_ waves down M, so the tile is BMF_ = WM_ * TM_ * 16 rows (the reference: TM 4 x WM 4 =
// 256). Per 16x16 output tile the arithmetic is the reference's, byte for byte -- the same A and W
// fragments out of the same LDS layout, the same kMag fold of the same W slab, the same WMMA chain
// over k ascending in 16-deep steps, the same EPIFAST epilogue -- so the bytes are the same at any
// tile height (gated: tools/capture/g5/g5_bench.cpp against the shipped
// `mxfp4_gemm_folded_tn2_wperm1_epifast1`, with k-reversal and no-fold controls that must fail).
//
// Why a tile height at all: a chunk of 257..384 rows planned at the C5 rungs 320 / 384 takes the
// reference's tile TWICE (512 rows of WMMA for 320 live), and the folded loop is WMMA-issue-bound
// (G1's ladder), so its cost is the tile count, not the live rows. A 320- or 384-row tile (or two
// 192-row ones) computes the rung's rows once.
//
// CTRL (lab only; 0 in every shipped symbol): 1 = each slab's four k-steps accumulated in reverse
// (an order change, must differ), 2 = W stored raw without the fold (must differ everywhere).
template <int TN, int TM_, int WM_, int CTRL = 0>
__device__ __forceinline__ void saber_mxfp4_folded_mt_impl(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride) {
  constexpr int WN_ = 2;
  constexpr int NT_ = WM_ * WN_ * 32;
  constexpr int BMF_ = WM_ * TM_ * 16;
  constexpr int BNF_T = WN_ * TN * 16;
  static_assert((BMF_ * BK) % (NT_ * 16) == 0, "the A stage must tile the thread block");
  __shared__ unsigned char sA[BMF_ * ASTR];
  __shared__ unsigned char sW[BNF_T * WSTR];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN_, wn = wave % WN_;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int m0 = blockIdx.y * BMF_, n0 = blockIdx.x * BNF_T;
  {
    const int p_ = n0 >= pb2 ? 2 : (n0 >= pb1 ? 1 : 0);
    A += (size_t)p_ * astride;
    As += (size_t)p_ * M;
  }

  floatx8 acc[TM_][TN];
#pragma unroll
  for (int i = 0; i < TM_; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    const unsigned char *__restrict__ Ab = A + (size_t)m0 * K + k0;
    // ---- A tile: BMF_*BK bytes, 16 B per thread per iteration (clamp, never predicate) ----
#pragma unroll
    for (int off = 0; off < BMF_ * BK; off += NT_ * 16) {
      const int idx = off + tid * 16;
      const int r = idx / BK, c = idx % BK;
      const int rc = r < M - 1 - m0 ? r : M - 1 - m0;
      *(uint4_t *)(&sA[r * ASTR + c]) = *(const uint4_t *)(Ab + (rc * K + c));
    }
    // ---- W tile, fragment order (the reference's WPERM staging at this block's thread count) ----
    {
      constexpr int KSTEPS_T = BK / 16, NTILES_T = BNF_T / 16;
      constexpr int TOT_SLOTS = NTILES_T * KSTEPS_T * 32;
      constexpr int WSLOTS = (TOT_SLOTS >= NT_ * 4) ? 4 : 2;
      static_assert(TOT_SLOTS % (NT_ * WSLOTS) == 0, "WPERM slot groups must tile");
      const unsigned int *__restrict__ Wp = (const unsigned int *)W;
      const int ksteps_g = K / 16, kstep0 = k0 / 16;
#pragma unroll
      for (int off = 0; off < TOT_SLOTS; off += NT_ * WSLOTS) {
        const int sl = off + tid * WSLOTS;
        const int lane_ = sl & 31, rest = sl >> 5;
        const int kst = rest % KSTEPS_T, ntl = rest / KSTEPS_T;
        const int r = ntl * 16 + (lane_ & 15), kloc = kst * 16 + (lane_ >> 4) * 8;
        const int gn = n0 + r;
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
          if constexpr (CTRL == 2) {
            *(uint2_t *)(&sW[(r + q) * WSTR + kloc]) = uint2_t{wq[q], wq[q] ^ 0x5a5a5a5au};
          } else {
            *(uint2_t *)(&sW[(r + q) * WSTR + kloc]) =
                uint2_t{__builtin_amdgcn_perm(bo, be, 0x05010400u),
                        __builtin_amdgcn_perm(bo, be, 0x07030602u)};
          }
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int step_ = 0; step_ < BK / 16; ++step_) {
      const int step = CTRL == 1 ? (BK / 16 - 1 - step_) : step_;
      const int kk = step * 16 + kb8;
      int2_t af[TM_], wf[TN];
#pragma unroll
      for (int i = 0; i < TM_; ++i) {
        const unsigned char *p = &sA[(wm * TM_ * 16 + i * 16 + col) * ASTR + kk];
        af[i][0] = *(const int *)p; af[i][1] = *(const int *)(p + 4);
      }
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const unsigned char *p = &sW[(wn * TN * 16 + j * 16 + col) * WSTR + kk];
        wf[j][0] = *(const int *)p; wf[j][1] = *(const int *)(p + 4);
      }
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < TM_; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf[j], acc[i][j]);
    }
    __syncthreads();
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
