// The LDS-staged bf16 WMMA GEMM body (workstream D6), shared by the vision tower (`vision.hip`)
// and the Flash-Next bf16 instantiations (`flashnext_hc.hip`, FN5). Moved here VERBATIM from
// vision.hip by FN5 so a second code object can instantiate it; the one addition is the
// epilogue EPI 4 (no bias: the linear's output is `bf(acc)`, the Flash-Next dense projections
// carry no bias), which every other EPI constant-folds away -- vision.hsaco's existing symbols
// are ISA-identical after the move (FN5, codeobj_diff).
//
// The includer defines: bf16x8, floatx8, `bf(float)` (one RNE rounding), `ld(const __bf16 *)`
// and `vg_wmma(a, b, c)` (the gfx12 bf16 WMMA) -- vision.hip's own definitions.
#pragma once

#define VL_BK 32
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
#define VL_STR (VL_BK + 8)  // 80 B rows

template <int EPI, int BM, int BN, int MF, int NF>
__device__ __forceinline__ void vis_gemm_lds_body(const __bf16 *__restrict__ a,
                                                  const __bf16 *__restrict__ w,
                                                  const __bf16 *__restrict__ bias,
                                                  const __bf16 *res, __bf16 *c, int m, int n,
                                                  int k) {
  constexpr int WROWS = BM / (MF * 16), WCOLS = BN / (NF * 16);
  constexpr int NT = WROWS * WCOLS * 32;
  constexpr int CHUNKS = (BM + BN) * (VL_BK / 8);  // 16-byte pieces of one A + W tile
  static_assert(CHUNKS % NT == 0, "every thread stages the same number of pieces");
  constexpr int PER = CHUNKS / NT;
  __shared__ __attribute__((aligned(16))) __bf16 lds[2][(BM + BN) * VL_STR];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int bm0 = (int)blockIdx.y * BM, bn0 = (int)blockIdx.x * BN;
  const int wm = (wave / WCOLS) * MF * 16, wn = (wave % WCOLS) * NF * 16;

  // The staging map: piece p = tid + i * NT is row p / 4 of the stacked [A; W] tile, k offset
  // (p % 4) * 8. Row and column clamps are fixed per thread except the column's tile base.
  const __bf16 *src[PER];
  int dst[PER];
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int p = tid + i * NT, r = p >> 2, kc = (p & 3) * 8;
    if (r < BM) {
      int ar = bm0 + r;
      ar = ar < m ? ar : m - 1;
      src[i] = a + (long)ar * k + kc;
    } else {
      int wr = bn0 + (r - BM);
      wr = wr < n ? wr : n - 1;
      src[i] = w + (long)wr * k + kc;
    }
    dst[i] = r * VL_STR + kc;
  }
  // A 32-k tile's piece at k offset kc reads columns k0 + kc .. +8; past K it is clamped to the
  // last eight columns of its own row (the half tile's upper k-step is never multiplied).
#define VL_GLOAD(T)                                                                            \
  do {                                                                                         \
    const int k0_ = (T) * VL_BK;                                                               \
    _Pragma("unroll") for (int i = 0; i < PER; ++i) {                                          \
      const int kc_ = ((tid + i * NT) & 3) * 8;                                                \
      const int off_ = k0_ + kc_ + 8 > k ? k - 8 - kc_ : k0_;                                  \
      v[i] = *(const u32x4 *)(src[i] + off_);                                                  \
    }                                                                                          \
  } while (0)
#define VL_SSTORE(B)                                                                           \
  do {                                                                                         \
    _Pragma("unroll") for (int i = 0; i < PER; ++i) *(u32x4 *)(&lds[B][dst[i]]) = v[i];        \
  } while (0)
  const int row = lane & 15, kh = (lane >> 4) * 8;
#define VL_FRAGS(B, S)                                                                         \
  do {                                                                                         \
    _Pragma("unroll") for (int i = 0; i < MF; ++i) af[i] =                                     \
        *(const bf16x8 *)(&lds[B][(wm + i * 16 + row) * VL_STR + (S) * 16 + kh]);              \
    _Pragma("unroll") for (int j = 0; j < NF; ++j) wf[j] =                                     \
        *(const bf16x8 *)(&lds[B][(BM + wn + j * 16 + row) * VL_STR + (S) * 16 + kh]);         \
  } while (0)

  floatx8 acc[MF][NF], part[MF][NF];
#pragma unroll
  for (int i = 0; i < MF; ++i)
#pragma unroll
    for (int j = 0; j < NF; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

  const int ksteps = k / 16, nchunk = ksteps / 4, ntile = (k + VL_BK - 1) / VL_BK;
  u32x4 v[PER];
  VL_GLOAD(0);
  VL_SSTORE(0);
  __syncthreads();

  // Two tiles per chunk: tile 2c in buffer 0, tile 2c + 1 in buffer 1.
  for (int ch = 0; ch < nchunk; ++ch) {
#pragma unroll
    for (int i = 0; i < MF; ++i)
#pragma unroll
      for (int j = 0; j < NF; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) part[i][j][e] = 0.f;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int t = 2 * ch + h;
      // The next tile, always: past the last one it re-reads the last (clamped), and the buffer
      // it lands in is read by nobody (the tail below only reads what a real prefetch staged).
      VL_GLOAD(t + 1 < ntile ? t + 1 : ntile - 1);
      // Keep the loads HERE, in flight under the WMMAs: left alone, the compiler sinks them to
      // the ds_write and issues them one at a time, each followed by its own wait (read in the
      // ISA: four serialized round trips per tile).
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        bf16x8 af[MF], wf[NF];
        VL_FRAGS(h, s);
#pragma unroll
        for (int i = 0; i < MF; ++i)
#pragma unroll
          for (int j = 0; j < NF; ++j) part[i][j] = vg_wmma(af[i], wf[j], part[i][j]);
      }
      // ... and the ds_write (whose wait is on those loads) AFTER the WMMAs, not hoisted above them.
      __builtin_amdgcn_sched_barrier(0);
      VL_SSTORE(h ^ 1);
      __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < MF; ++i)
#pragma unroll
      for (int j = 0; j < NF; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[i][j][e] += part[i][j][e];
  }
  // The k-steps past the last whole chunk (0..3 of them; K = 4304 has one): each its own zeroed
  // WMMA and a VALU add, exactly `vis_gemm_body`'s tail. They sit in tile 2 * nchunk (buffer 0,
  // already staged by the loop's last prefetch) and, for a third step, tile 2 * nchunk + 1.
  const int rem = ksteps - 4 * nchunk;
  for (int g = 0; g < rem; ++g) {
    const int b = g >> 1, s = g & 1;
    if (g == 2) {  // the second tail tile: stage it (nothing else reads buffer 1 now)
      VL_GLOAD(2 * nchunk + 1);
      VL_SSTORE(1);
      __syncthreads();
    }
    bf16x8 af[MF], wf[NF];
    VL_FRAGS(b, s);
#pragma unroll
    for (int i = 0; i < MF; ++i)
#pragma unroll
      for (int j = 0; j < NF; ++j) {
        floatx8 z;
#pragma unroll
        for (int e = 0; e < 8; ++e) z[e] = 0.f;
        z = vg_wmma(af[i], wf[j], z);
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[i][j][e] += z[e];
      }
  }

#undef VL_GLOAD
#undef VL_SSTORE
#undef VL_FRAGS
  // The epilogue, `vis_gemm_body`'s: bias in fp32, ONE rounding to bf16, then the ATen op.
#pragma unroll
  for (int j = 0; j < NF; ++j) {
    const int col = bn0 + wn + j * 16 + (lane & 15);
    if (col >= n) continue;
    const float bb = EPI == 4 ? 0.f : ld(bias + col);
#pragma unroll
    for (int i = 0; i < MF; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int r = bm0 + wm + i * 16 + (lane >> 4) * 8 + e;
        if (r >= m) continue;
        const float y = EPI == 4 ? bf(acc[i][j][e]) : bf(acc[i][j][e] + bb);
        float o;
        if (EPI == 1) {
          o = 0.5f * y * (1.0f + tanhf(0.7978845608028654f * (y + 0.044715f * y * y * y)));
        } else if (EPI == 2) {
          o = 0.5f * y * (1.0f + erff(y * 0.7071067811865476f));
        } else if (EPI == 3) {
          o = y + ld(res + (long)r * n + col);
        } else {
          o = y;
        }
        c[(long)r * n + col] = (__bf16)o;
      }
  }
}
