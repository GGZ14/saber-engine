// INK60 (docs/INKLING2.md INK60): the f16 SCALE-FOLDED MoE prefill GEMM -- a NUMERICS arm (`--ink-p60 f16`).
// Saber's own, written from the record laws INKLING2 states (INK6 / FN148 / FN153); the f64 twin and the
// kernel-vs-kernel oracle are tools/capture/ink60/ink60_twin.py.
//
// The shipped prefill cells (FN153's `fn153_w4a8b_pf_tm4_g*` for the NVFP4 layers 0..19, INK6's
// `ink_mx8_w8a8_g*` for the MXFP8 layers 20..23) run an fp8 x fp8 WMMA a 16-k (NVFP4) / 32-k (MXFP8) group
// into a ZERO accumulator and fold the group's scale into the f32 sum with one FMA an element
// (acc = fma(t, s, acc): 8 VALU FMAs a 16 x 16 tile a group). Here the block scale is folded into the
// WEIGHT in f16 once, when the 64-deep slab is staged in LDS, and the WMMA (f16 x f16, f32 accumulate)
// accumulates the whole K in its own C operand: no per-tile scale FMA.
//   NVFP4: w16 = e2m1 x E4M3(ws[g16, n]) -- EXACT in f16 (a 2-bit by 4-bit significand product, magnitude
//          2^-10 .. 2,688: always an f16 normal or zero); C = bf16(acc x gs x As[row]) as FN153.
//   MXFP8: w16 = e4m3 x 2^(s - 127 + 12) -- EXACT in f16 iff -5 <= s - 127 + 12 <= 7 (e4m3's smallest
//          subnormal 2^-9 stays an f16 normal, 448 x 2^7 < 65,504); C = bf16((acc x 2^-12) x As[row]).
//          A block outside the range cannot be folded exactly: it is COUNTED (atomicAdd on `cnt[0]`, the
//          engine reads it at every end line and refuses a run whose count is not 0); the checkpoint's
//          layers 20..23 have s - 127 in [-14, -9] (0 of 127,401,984 blocks out of range, LEDGER INK60).
//   A: e4m3 rows (one f32 scale a row) -> f16 exactly (e4m3's range is inside f16's normals).
// WHAT CHANGES (the numerics): every product a x w is the same real number as before (both exact); the
// SUMMATION is one f32 chain over K inside the WMMA accumulator (16 products + C a WMMA, K ascending) instead
// of a zero-started WMMA a group + one rounded FMA a group; the epilogue is the parents'.
// Tile: 64 rows (one expert, the INK37 / FN153 tile law, dead 16-row fragments skipped) x 128 columns,
// 8 waves (4 x 2), a wave = one 16-row fragment x 4 column fragments; K % 64 == 0, N % 128 == 0.
// Symbols: ink60_f16_nv_g{1,0} (NVFP4 v2 record, SIG_FN148_GROUPED; W staged in LDS, 64 x 128 tiles -- the
// first form, slower than FN153 at >= 1,024 rows, kept for the twin), ink60_f16_nv2_g{1,0} (THE ENGINE'S NVFP4
// body: fragments folded in registers, 64 x 256 tiles, see below), ink60_f16_mx_g{1,0} (INK6's MXFP8 record,
// the same signature: `p` / `pmax` / `gsplit` unused, `cnt` = the range counter).
// Built with -ffp-contract=off (the `ink_` row of saber-kbuild).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

typedef float i60_f8 __attribute__((ext_vector_type(8)));
typedef _Float16 i60_h8 __attribute__((ext_vector_type(8)));
typedef _Float16 i60_h2 __attribute__((ext_vector_type(2)));
typedef float i60_f2 __attribute__((ext_vector_type(2)));
typedef unsigned int i60_u4 __attribute__((ext_vector_type(4)));

#define I60_BK 64
#define I60_STR (I60_BK + 8)   // halfs a staged row (144 B: 16-B aligned fragment reads)
#define I60_MX_BIAS 12

static __device__ __forceinline__ unsigned int i60_pk(float a, float b) {
  // exact here (every value is an f16): the round mode never applies
  return __builtin_bit_cast(unsigned int, __builtin_amdgcn_cvt_pkrtz(a, b));
}

// 16 e4m3 bytes -> 16 f16 (exact), optionally x 2^e (exact while e keeps them f16 normals)
static __device__ __forceinline__ void i60_e4m3x16(const i60_u4 v, float mul, i60_u4 *o) {
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    const i60_f2 lo = __builtin_amdgcn_cvt_pk_f32_fp8((int)v[w], false);
    const i60_f2 hi = __builtin_amdgcn_cvt_pk_f32_fp8((int)v[w], true);
    o[w >> 1][(w & 1) * 2 + 0] = i60_pk(lo[0] * mul, lo[1] * mul);
    o[w >> 1][(w & 1) * 2 + 1] = i60_pk(hi[0] * mul, hi[1] * mul);
  }
}

template <bool NV, bool GATHER>
__device__ __forceinline__ void ink60_f16_impl(
    const unsigned char *__restrict__ A, const unsigned long long *__restrict__ xptr,
    const int *__restrict__ tile_expert, const int *__restrict__ tile_row0,
    const int *__restrict__ tile_rows, const int *__restrict__ row_tok,
    const int *__restrict__ meta, const float *__restrict__ As, int *__restrict__ cnt,
    __bf16 *__restrict__ C, int N, int K, int gsplit) {
  constexpr int TN = 4, WN = 2, NT = 256, BM = 64, BN = WN * TN * 16;
  __shared__ __attribute__((aligned(16))) _Float16 sA[BM * I60_STR];
  __shared__ __attribute__((aligned(16))) _Float16 sW[BN * I60_STR];
  __shared__ int s_arow[BM];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN, wn = wave % WN;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int tile = blockIdx.y;
  if (tile >= meta[0]) return;
  const int texp = tile_expert[tile], row0 = tile_row0[tile];
  const int M = tile_rows[tile];
  const unsigned char *__restrict__ W = (const unsigned char *)(uintptr_t)xptr[3 * (size_t)texp + 0];
  const unsigned char *__restrict__ Ws = (const unsigned char *)(uintptr_t)xptr[3 * (size_t)texp + 1];
  // this wave's 16-row fragment is live (wave-uniform)
  const bool live = __builtin_amdgcn_readfirstlane(wm * 16 < M ? 1 : 0) != 0;
  if (tid < BM) {
    const int rr = tid < M - 1 ? tid : M - 1;   // clamp, never predicate
    s_arow[tid] = GATHER ? row_tok[row0 + rr] : row0 + rr;
  }
  __syncthreads();
  const int n0 = blockIdx.x * BN;
  // A staging: thread -> (row tid / 4, 16 k at (tid % 4) * 16)
  const int ar = tid >> 2, ac = (tid & 3) * 16;
  const unsigned char *__restrict__ abase = A + (size_t)s_arow[ar] * K + ac;
  i60_f8 acc[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[j][e] = 0.f;
  int bad = 0;

  for (int k0 = 0; k0 < K; k0 += I60_BK) {
    {
      const i60_u4 v = *(const i60_u4 *)(abase + k0);
      i60_u4 o[2];
      i60_e4m3x16(v, 1.f, o);
      *(i60_u4 *)&sA[ar * I60_STR + ac] = o[0];
      *(i60_u4 *)&sA[ar * I60_STR + ac + 8] = o[1];
    }
    if constexpr (NV) {
      // the v2 record's permuted e2m1 slab (FN153's staging map: one 16-B load = 4 columns x 8 k)
      const unsigned int *__restrict__ Wp = (const unsigned int *)W;
      const int ksteps_g = K / 16, kstep0 = k0 / 16;
      const int sl = tid * 4;
      const int lane_ = sl & 31, rest = sl >> 5;
      const int kst = rest % (I60_BK / 16), ntl = rest / (I60_BK / 16);
      const int r = ntl * 16 + (lane_ & 15), kloc = kst * 16 + (lane_ >> 4) * 8;
      const int gn = n0 + r;
      const int gc = gn < N - 4 ? gn : N - 4;
      const int lanec = (lane_ & 16) | (gc & 15);
      const i60_u4 wq = *(const i60_u4 *)&Wp[((size_t)(gc >> 4) * ksteps_g + kstep0 + kst) * 32 + lanec];
      const unsigned int sw4 = *(const unsigned int *)(Ws + (size_t)(kstep0 + kst) * N + gc);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        // e2m1 -> the f16's high byte (low byte 0): 0, .5, 1, 1.5 | 2, 3, 4, 6; the sign to bit 7
        const unsigned int t0 = 0x3e3c3800u, t1 = 0x46444240u;
        const unsigned int ev = wq[q] & 0x0F0F0F0Fu, od = (wq[q] >> 4) & 0x0F0F0F0Fu;
        const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u) | ((ev & 0x08080808u) << 4);
        const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u) | ((od & 0x08080808u) << 4);
        const _Float16 s16 = (_Float16)__builtin_amdgcn_cvt_f32_fp8((int)(sw4 >> (8 * q)), 0);
        const i60_h2 ss = {s16, s16};
        i60_u4 o;
        // k order be0 bo0 be1 bo1 be2 bo2 be3 bo3 (FN153's interleave), as f16 pairs x the block scale
        o[0] = __builtin_bit_cast(unsigned int, __builtin_bit_cast(i60_h2, __builtin_amdgcn_perm(bo, be, 0x040C000Cu)) * ss);
        o[1] = __builtin_bit_cast(unsigned int, __builtin_bit_cast(i60_h2, __builtin_amdgcn_perm(bo, be, 0x050C010Cu)) * ss);
        o[2] = __builtin_bit_cast(unsigned int, __builtin_bit_cast(i60_h2, __builtin_amdgcn_perm(bo, be, 0x060C020Cu)) * ss);
        o[3] = __builtin_bit_cast(unsigned int, __builtin_bit_cast(i60_h2, __builtin_amdgcn_perm(bo, be, 0x070C030Cu)) * ss);
        *(i60_u4 *)&sW[(r + q) * I60_STR + kloc] = o;
      }
    } else {
      // INK6's row-major e4m3 [N, K] slab, the e8m0 [K / 32, N] scale folded (x 2^(s - 127 + bias))
#pragma unroll
      for (int i = 0; i < BN * I60_BK / (NT * 16); ++i) {
        const int idx = tid + i * NT;
        const int r = idx >> 2, c = (idx & 3) * 16;
        const int gn = n0 + r < N - 1 ? n0 + r : N - 1;
        const i60_u4 v = *(const i60_u4 *)(W + (size_t)gn * K + k0 + c);
        const int e = (int)Ws[(size_t)((k0 + c) / 32) * N + gn] - 127 + I60_MX_BIAS;
        bad += (e < -5 || e > 7) ? 1 : 0;
        i60_u4 o[2];
        i60_e4m3x16(v, __builtin_ldexpf(1.f, e), o);
        *(i60_u4 *)&sW[r * I60_STR + c] = o[0];
        *(i60_u4 *)&sW[r * I60_STR + c + 8] = o[1];
      }
    }
    __syncthreads();
    if (live) {
#pragma unroll
      for (int step = 0; step < I60_BK / 16; ++step) {
        const int kk = step * 16 + kb8;
        const i60_h8 af = *(const i60_h8 *)&sA[(wm * 16 + col) * I60_STR + kk];
        i60_h8 wf[TN];
#pragma unroll
        for (int j = 0; j < TN; ++j) wf[j] = *(const i60_h8 *)&sW[(wn * TN * 16 + j * 16 + col) * I60_STR + kk];
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af, wf[j], acc[j]);
      }
    }
    __syncthreads();
  }
  if constexpr (!NV) {
    if (bad) atomicAdd(cnt, bad);
  }
  if (!live) return;
  const float *__restrict__ Gs = (const float *)(uintptr_t)xptr[3 * (size_t)texp + 2];
  float rf[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int n = n0 + wn * TN * 16 + j * 16 + col;
    if constexpr (NV) {
      rf[j] = (n < N) ? Gs[n < gsplit ? 0 : 1] : 0.f;
    } else {
      rf[j] = 0x1p-12f;   // 2^-I60_MX_BIAS
    }
  }
#pragma unroll
  for (int j = 0; j < TN; ++j)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int m = wm * 16 + kb8 + e;
      const int n = n0 + wn * TN * 16 + j * 16 + col;
      if (m < M && n < N) C[(size_t)(row0 + m) * N + n] = (__bf16)(acc[j][e] * rf[j] * As[s_arow[m]]);
    }
}

#define INK60_F16(name, NV, G)                                                                       \
  extern "C" __global__ __launch_bounds__(256) void name(                                            \
      const unsigned char *__restrict__ A, const unsigned long long *__restrict__ xptr,             \
      const int *__restrict__ tile_expert, const int *__restrict__ tile_row0,                       \
      const int *__restrict__ tile_rows, const int *__restrict__ row_tok,                           \
      const int *__restrict__ meta, const float *__restrict__ As, float *__restrict__ P,            \
      int *__restrict__ cnt, __bf16 *__restrict__ C, int N, int K, int Pmax, int gsplit) {          \
    (void)P;                                                                                        \
    (void)Pmax;                                                                                     \
    ink60_f16_impl<NV, G>(A, xptr, tile_expert, tile_row0, tile_rows, row_tok, meta, As, cnt, C, N, \
                          K, gsplit);                                                               \
  }
INK60_F16(ink60_f16_nv_g1, true, true)
INK60_F16(ink60_f16_nv_g0, true, false)
INK60_F16(ink60_f16_mx_g1, false, true)
INK60_F16(ink60_f16_mx_g0, false, false)
#undef INK60_F16

// ---- nv2: the NVFP4 record's fragments read STRAIGHT into registers (the v2 record is stored in WMMA
// B-fragment order: 32 dwords a (16-column block, 16-k step), lane L = column L & 15, k half L >> 4), folded
// to f16 there; only A is staged in LDS. A wave = all four 16-row fragments x 2 column fragments (8 WMMAs
// a 16-k step on 4 + 0 LDS fragment reads); a workgroup = 64 rows x 256 columns; the next slab's A and W
// dwords are loaded into registers while the current slab computes. N % 256 == 0. Same numerics as
// ink60_f16_nv (each output element's chain is the same f32 WMMA accumulation over K ascending).
static __device__ __forceinline__ void i60_nv_frag(unsigned int wq, _Float16 s16, i60_h8 &o) {
  const unsigned int t0 = 0x3e3c3800u, t1 = 0x46444240u;
  const unsigned int ev = wq & 0x0F0F0F0Fu, od = (wq >> 4) & 0x0F0F0F0Fu;
  const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u) | ((ev & 0x08080808u) << 4);
  const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u) | ((od & 0x08080808u) << 4);
  const i60_h2 ss = {s16, s16};
  i60_u4 r;
  r[0] = __builtin_bit_cast(unsigned int, __builtin_bit_cast(i60_h2, __builtin_amdgcn_perm(bo, be, 0x040C000Cu)) * ss);
  r[1] = __builtin_bit_cast(unsigned int, __builtin_bit_cast(i60_h2, __builtin_amdgcn_perm(bo, be, 0x050C010Cu)) * ss);
  r[2] = __builtin_bit_cast(unsigned int, __builtin_bit_cast(i60_h2, __builtin_amdgcn_perm(bo, be, 0x060C020Cu)) * ss);
  r[3] = __builtin_bit_cast(unsigned int, __builtin_bit_cast(i60_h2, __builtin_amdgcn_perm(bo, be, 0x070C030Cu)) * ss);
  o = __builtin_bit_cast(i60_h8, r);
}

template <bool GATHER>
__device__ __forceinline__ void ink60_nv2_impl(
    const unsigned char *__restrict__ A, const unsigned long long *__restrict__ xptr,
    const int *__restrict__ tile_expert, const int *__restrict__ tile_row0,
    const int *__restrict__ tile_rows, const int *__restrict__ row_tok,
    const int *__restrict__ meta, const float *__restrict__ As, __bf16 *__restrict__ C, int N, int K,
    int gsplit) {
  constexpr int TM = 4, TN = 2, BM = 64, BN = 8 * TN * 16, KS = I60_BK / 16;
  __shared__ __attribute__((aligned(16))) _Float16 sA[BM * I60_STR];
  __shared__ int s_arow[BM];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int tile = blockIdx.y;
  if (tile >= meta[0]) return;
  const int texp = tile_expert[tile], row0 = tile_row0[tile];
  const int M = tile_rows[tile];
  const unsigned int *__restrict__ Wp = (const unsigned int *)(uintptr_t)xptr[3 * (size_t)texp + 0];
  const unsigned char *__restrict__ Ws = (const unsigned char *)(uintptr_t)xptr[3 * (size_t)texp + 1];
  const int nlf = __builtin_amdgcn_readfirstlane(min(TM, (M + 15) / 16));
  if (tid < BM) {
    const int rr = tid < M - 1 ? tid : M - 1;   // clamp, never predicate
    s_arow[tid] = GATHER ? row_tok[row0 + rr] : row0 + rr;
  }
  __syncthreads();
  const int nw0 = blockIdx.x * BN + wave * TN * 16;
  const int ar = tid >> 2, ac = (tid & 3) * 16;
  const unsigned char *__restrict__ abase = A + (size_t)s_arow[ar] * K + ac;
  const int ksteps_g = K / 16;
  const unsigned int *wsrc[TN];
  const unsigned char *ssrc[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int n = min(nw0 + j * 16 + col, N - 1);
    wsrc[j] = Wp + (size_t)((nw0 + j * 16) >> 4) * ksteps_g * 32 + lane;
    ssrc[j] = Ws + n;
  }
  i60_f8 acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;
  // the slab in registers: A's 16 bytes, the W dwords + their scale bytes
  i60_u4 araw = *(const i60_u4 *)abase;
  unsigned int wraw[TN][KS], sraw[TN][KS];
#pragma unroll
  for (int j = 0; j < TN; ++j)
#pragma unroll
    for (int s = 0; s < KS; ++s) {
      wraw[j][s] = wsrc[j][s * 32];
      sraw[j][s] = ssrc[j][(size_t)s * N];
    }
  for (int k0 = 0; k0 < K; k0 += I60_BK) {
    {
      i60_u4 o[2];
      i60_e4m3x16(araw, 1.f, o);
      *(i60_u4 *)&sA[ar * I60_STR + ac] = o[0];
      *(i60_u4 *)&sA[ar * I60_STR + ac + 8] = o[1];
    }
    i60_h8 wf[TN][KS];
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int s = 0; s < KS; ++s)
        i60_nv_frag(wraw[j][s], (_Float16)__builtin_amdgcn_cvt_f32_fp8((int)sraw[j][s], 0), wf[j][s]);
    __syncthreads();
    if (k0 + I60_BK < K) {
      const int kn = k0 + I60_BK, ksn = kn / 16;
      araw = *(const i60_u4 *)(abase + kn);
#pragma unroll
      for (int j = 0; j < TN; ++j)
#pragma unroll
        for (int s = 0; s < KS; ++s) {
          wraw[j][s] = wsrc[j][(ksn + s) * 32];
          sraw[j][s] = ssrc[j][(size_t)(ksn + s) * N];
        }
    }
#pragma unroll
    for (int s = 0; s < KS; ++s) {
      const int kk = s * 16 + kb8;
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        if (i < nlf) {
          const i60_h8 af = *(const i60_h8 *)&sA[(i * 16 + col) * I60_STR + kk];
#pragma unroll
          for (int j = 0; j < TN; ++j) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af, wf[j][s], acc[i][j]);
        }
      }
    }
    __syncthreads();
  }
  const float *__restrict__ Gs = (const float *)(uintptr_t)xptr[3 * (size_t)texp + 2];
  float rf[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int n = nw0 + j * 16 + col;
    rf[j] = (n < N) ? Gs[n < gsplit ? 0 : 1] : 0.f;
  }
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = i * 16 + kb8 + e;
        const int n = nw0 + j * 16 + col;
        if (i < nlf && m < M && n < N) C[(size_t)(row0 + m) * N + n] = (__bf16)(acc[i][j][e] * rf[j] * As[s_arow[m]]);
      }
}

#define INK60_NV2(name, G)                                                                           \
  extern "C" __global__ __launch_bounds__(256) void name(                                            \
      const unsigned char *__restrict__ A, const unsigned long long *__restrict__ xptr,             \
      const int *__restrict__ tile_expert, const int *__restrict__ tile_row0,                       \
      const int *__restrict__ tile_rows, const int *__restrict__ row_tok,                           \
      const int *__restrict__ meta, const float *__restrict__ As, float *__restrict__ P,            \
      int *__restrict__ cnt, __bf16 *__restrict__ C, int N, int K, int Pmax, int gsplit) {          \
    (void)P;                                                                                        \
    (void)cnt;                                                                                      \
    (void)Pmax;                                                                                     \
    ink60_nv2_impl<G>(A, xptr, tile_expert, tile_row0, tile_rows, row_tok, meta, As, C, N, K, gsplit); \
  }
INK60_NV2(ink60_f16_nv2_g1, true)
INK60_NV2(ink60_f16_nv2_g0, false)
#undef INK60_NV2
