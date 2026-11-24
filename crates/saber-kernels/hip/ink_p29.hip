// Inkling 2 (inkling2_loop) prefill levers, INK29 (docs/INKLING2.md section 9 INK29). Saber's own.
// Every body here is an ARM behind `--ink29 <items>` (empty = INK26's defaults):
//
//   ink29_gemm_i8_kx{2,3}   INK25's iu8 contract (hip/ink_i8.hip) BYTE FOR BYTE, the K range cut into
//                           S slices run by S waves of one workgroup (one 16-row m-tile x 16 columns):
//                           slice s >= 1 writes its per-block products p = f32(i) * (sa * sw) to LDS,
//                           slice 0 sums its own blocks then the stored ones IN BLOCK ORDER -> the
//                           same f = f + p chain as ink_gemm_i8_w8 (item `kx`). The short-chunk floor
//                           was one 128-k group in flight a wave (INK25): S slices = S groups in flight.
//   ink29_q8_ls             int8 activations a (row, 32-k block) with a least-squares scale: codes
//                           from amax / 127, then sa' = sum(x q) / sum(q q), codes re-rounded at sa'
//                           (clamped +-127). A NUMERICS arm (item `ls`); the GEMM is unchanged.
//   ink29_q8_b16 + ink29_gemm_i8b16   activations a (row, 16-k block) (sa = amax / 127, macaq's code
//                           law); the GEMM's two WMMAs of a 32-k block each take ONE 16-k half, so
//                           f = f + f32(i_lo) * (sa_lo * sw); f = f + f32(i_hi) * (sa_hi * sw), blocks in
//                           order (item `b16`, a NUMERICS arm).
//   ink29_gemm_bw_s{1,2,3}  the 4 GLOBAL layers' bf16 projections of a prefill chunk on
//                           v_wmma_f32_16x16x16_bf16: activations f32 -> bf16 (RNE) in registers, the
//                           weight bf16 as stored; S = K / 768 slices (a law of the SHAPE, never of M),
//                           each one WMMA chain over its 768 k from 0, then f = (c0 + c1) + c2 in slice
//                           order; C = acc ? C + f : f. A row's bytes depend on that row alone (item
//                           `bw`, a NUMERICS arm: the bf16 rounding of the activations).
//   ink29_route_sort_e128_v2  ink_route_sort_e128's outputs BYTE FOR BYTE (row_tok / row_slot / tiles /
//                           meta / pos), one workgroup an expert: the stable t-major rank by a ballot
//                           prefix instead of the single workgroup's O(256) scans (item `rs`).
//   ink29_route_sort_e128_v2_t{32,64} + moe_fn148's tm2 / tm4 cells: the same per-row bytes on 32 /
//                           64-row expert tiles (item `mt`; the weight staged once a TR rows).
//   ink29_out_gate_q8       ink_out_gate's y (the same bytes) + INK25's q8 of y in the same launch
//                           (item `fq`, byte-equal to out_gate then ink_act_q8_b32_v2).
#include <hip/hip_runtime.h>

#include "macaq_i8.h"

typedef int i8v2 __attribute__((ext_vector_type(2)));
typedef int i8v4 __attribute__((ext_vector_type(4)));
typedef int i8v8 __attribute__((ext_vector_type(8)));
typedef unsigned u8v2 __attribute__((ext_vector_type(2)));
typedef float f8v4 __attribute__((ext_vector_type(4)));
typedef float f8v8 __attribute__((ext_vector_type(8)));
typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ i8v8 i8_wmma(i8v2 a, i8v2 b, i8v8 c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
}

// ---------------------------------------------------------------------------------------- quants
// LS: 4 lanes a (row, 32-k block), 8 values a lane (ink_act_q8_b32_v2's map).
extern "C" __global__ __launch_bounds__(256) void ink29_q8_ls(const float* __restrict__ a, int lda,
                                                              unsigned char* __restrict__ q,
                                                              float* __restrict__ s, int M, int K) {
#pragma clang fp contract(off)
  const int nb = K >> 5;
  const long g = (long)blockIdx.x * 256 + threadIdx.x;
  const long blk = g >> 2;
  const int part = threadIdx.x & 3;
  const bool live = blk < (long)M * nb;
  const long bl = live ? blk : 0;
  const int m = (int)(bl / nb), b = (int)(bl % nb);
  const f8v4* p = (const f8v4*)(a + (long)m * lda + b * 32 + part * 8);
  const f8v4 v0 = p[0], v1 = p[1];
  float x[8] = {v0[0], v0[1], v0[2], v0[3], v1[0], v1[1], v1[2], v1[3]};
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(x[j]));
  amax = fmaxf(amax, __shfl_xor(amax, 1));
  amax = fmaxf(amax, __shfl_xor(amax, 2));
  float sc = __fdiv_rn(amax, 127.f);
  if (!(sc > 0.f)) sc = 1.f;
  // one least-squares refinement for the codes at sc (the sums are commutative butterflies: every
  // lane of the block ends with the same bits)
  float xq = 0.f, qq = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float c = (float)(signed char)macaq_i8_code(x[j], sc);
    xq = xq + x[j] * c;
    qq = qq + c * c;
  }
  xq = xq + __shfl_xor(xq, 1);
  qq = qq + __shfl_xor(qq, 1);
  xq = xq + __shfl_xor(xq, 2);
  qq = qq + __shfl_xor(qq, 2);
  float s2 = qq > 0.f ? __fdiv_rn(xq, qq) : 1.f;
  if (!(s2 > 0.f)) s2 = sc;
  if (!live) return;
  unsigned w0 = 0, w1 = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    w0 |= macaq_i8_code(x[j], s2) << (8 * j);
    w1 |= macaq_i8_code(x[4 + j], s2) << (8 * j);
  }
  *(u8v2*)(q + (long)m * K + b * 32 + part * 8) = u8v2{w0, w1};
  if (part == 0) s[(long)m * nb + b] = s2;
}

// B16: 2 lanes a (row, 16-k block), 8 values a lane; scales [M, K / 16].
extern "C" __global__ __launch_bounds__(256) void ink29_q8_b16(const float* __restrict__ a, int lda,
                                                               unsigned char* __restrict__ q,
                                                               float* __restrict__ s, int M, int K) {
  const int nb = K >> 4;
  const long g = (long)blockIdx.x * 256 + threadIdx.x;
  const long blk = g >> 1;
  const int part = threadIdx.x & 1;
  const bool live = blk < (long)M * nb;
  const long bl = live ? blk : 0;
  const int m = (int)(bl / nb), b = (int)(bl % nb);
  const f8v4* p = (const f8v4*)(a + (long)m * lda + b * 16 + part * 8);
  const f8v4 v0 = p[0], v1 = p[1];
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) amax = fmaxf(amax, fmaxf(fabsf(v0[j]), fabsf(v1[j])));
  amax = fmaxf(amax, __shfl_xor(amax, 1));
  float sc = __fdiv_rn(amax, 127.f);
  if (!(sc > 0.f)) sc = 1.f;
  if (!live) return;
  const unsigned w0 = macaq_i8_code(v0[0], sc) | (macaq_i8_code(v0[1], sc) << 8) |
                      (macaq_i8_code(v0[2], sc) << 16) | (macaq_i8_code(v0[3], sc) << 24);
  const unsigned w1 = macaq_i8_code(v1[0], sc) | (macaq_i8_code(v1[1], sc) << 8) |
                      (macaq_i8_code(v1[2], sc) << 16) | (macaq_i8_code(v1[3], sc) << 24);
  *(u8v2*)(q + (long)m * K + b * 16 + part * 8) = u8v2{w0, w1};
  if (part == 0) s[(long)m * nb + b] = sc;
}

// out_gate + q8 in one launch: 4 lanes a (row, 32-k block) of y (N % 32 == 0); y written as
// ink_out_gate writes it (expf, 1 / (1 + e), o * sg), the codes / scales as ink_act_q8_b32_v2.
extern "C" __global__ __launch_bounds__(256) void ink29_out_gate_q8(
    const float* __restrict__ o, int ld_o, const float* __restrict__ g, int ld_g,
    float* __restrict__ y, int ld_y, int T, int N, unsigned char* __restrict__ q,
    float* __restrict__ s) {
#pragma clang fp contract(off)
  const int nb = N >> 5;
  const long gi = (long)blockIdx.x * 256 + threadIdx.x;
  const long blk = gi >> 2;
  const int part = threadIdx.x & 3;
  const bool live = blk < (long)T * nb;
  const long bl = live ? blk : 0;
  const int t = (int)(bl / nb), b = (int)(bl % nb);
  float v[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const long c = b * 32 + part * 8 + j;
    const float sg = 1.0f / (1.0f + expf(-g[(long)t * ld_g + c]));
    v[j] = o[(long)t * ld_o + c] * sg;
  }
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(v[j]));
  amax = fmaxf(amax, __shfl_xor(amax, 1));
  amax = fmaxf(amax, __shfl_xor(amax, 2));
  float sc = __fdiv_rn(amax, 127.f);
  if (!(sc > 0.f)) sc = 1.f;
  if (!live) return;
#pragma unroll
  for (int j = 0; j < 8; ++j) y[(long)t * ld_y + b * 32 + part * 8 + j] = v[j];
  unsigned w0 = 0, w1 = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    w0 |= macaq_i8_code(v[j], sc) << (8 * j);
    w1 |= macaq_i8_code(v[4 + j], sc) << (8 * j);
  }
  *(u8v2*)(q + (long)t * N + b * 32 + part * 8) = u8v2{w0, w1};
  if (part == 0) s[(long)t * nb + b] = sc;
}

// ---------------------------------------------------------------------------------- iu8 GEMMs
// One 128-k group of a lane: weight + activation bytes, the 4 (8 at B16) activation scales, the 8
// columns' weight scales (4 bf16 a column).
template <bool B16>
struct G29 {
  i8v4 w[4], a[4];
  f8v4 sa[B16 ? 2 : 1];
  u8v2 sw[8];
};

template <bool B16>
__device__ __forceinline__ void g29_load(G29<B16>& g, const unsigned char* __restrict__ wp,
                                         const unsigned char* __restrict__ ap,
                                         const float* __restrict__ sap,
                                         const unsigned short* const* scp, int grp) {
  if constexpr (!B16) {
    // lane half h: bytes [16h, 16h + 16) of each 32-k block (ink_i8's map)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      g.w[j] = *(const i8v4*)(wp + grp * 128 + 32 * j);
      g.a[j] = *(const i8v4*)(ap + grp * 128 + 32 * j);
    }
    g.sa[0] = *(const f8v4*)(sap + grp * 4);
  } else {
    // lane half h: bytes [8h, 8h + 8) (.xy: the block's first 16-k half) and [16 + 8h, 24 + 8h)
    // (.zw: the second half) -> WMMA 1 sums k 0..15 of the block, WMMA 2 k 16..31
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const i8v2 w0 = *(const i8v2*)(wp + grp * 128 + 32 * j);
      const i8v2 w1 = *(const i8v2*)(wp + grp * 128 + 32 * j + 16);
      const i8v2 a0 = *(const i8v2*)(ap + grp * 128 + 32 * j);
      const i8v2 a1 = *(const i8v2*)(ap + grp * 128 + 32 * j + 16);
      g.w[j] = i8v4{w0.x, w0.y, w1.x, w1.y};
      g.a[j] = i8v4{a0.x, a0.y, a1.x, a1.y};
    }
    g.sa[0] = *(const f8v4*)(sap + grp * 8);
    g.sa[1] = *(const f8v4*)(sap + grp * 8 + 4);
  }
#pragma unroll
  for (int e = 0; e < 8; ++e) g.sw[e] = *(const u8v2*)(scp[e] + grp * 4);
}

__device__ __forceinline__ float g29_sw(const u8v2 (&sw)[8], int e, int j) {
  const unsigned d = sw[e][j >> 1];
  return __builtin_bit_cast(float, (j & 1) ? (d & 0xFFFF0000u) : (d << 16));
}

// the group's per-block products, block by block: f = f + p (SUM) or p stored to `st` (store
// target: block j of this group at st + j * 256 (+ 256 * 4 for B16's second half), lane-major)
template <bool B16, bool SUM>
__device__ __forceinline__ void g29_do(const G29<B16>& g, float (&f)[8], float* __restrict__ st,
                                       int lane) {
#pragma clang fp contract(off)
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    if constexpr (!B16) {
      i8v8 z = {0, 0, 0, 0, 0, 0, 0, 0};
      z = i8_wmma(g.w[j].xy, g.a[j].xy, z);
      z = i8_wmma(g.w[j].zw, g.a[j].zw, z);
      const float sa = g.sa[0][j];
      float pv[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float s = sa * g29_sw(g.sw, e, j);
        pv[e] = (float)z[e] * s;
        if constexpr (SUM) f[e] = f[e] + pv[e];
      }
      if constexpr (!SUM) {
        f8v4* o = (f8v4*)(st + j * 256 + lane * 8);
        o[0] = f8v4{pv[0], pv[1], pv[2], pv[3]};
        o[1] = f8v4{pv[4], pv[5], pv[6], pv[7]};
      }
    } else {
      const i8v8 z0 = i8_wmma(g.w[j].xy, g.a[j].xy, i8v8{0, 0, 0, 0, 0, 0, 0, 0});
      const i8v8 z1 = i8_wmma(g.w[j].zw, g.a[j].zw, i8v8{0, 0, 0, 0, 0, 0, 0, 0});
      const float s0 = g.sa[j >> 1][(j & 1) * 2], s1 = g.sa[j >> 1][(j & 1) * 2 + 1];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float sw = g29_sw(g.sw, e, j);
        const float p0 = (float)z0[e] * (s0 * sw);
        const float p1 = (float)z1[e] * (s1 * sw);
        f[e] = f[e] + p0;
        f[e] = f[e] + p1;
      }
    }
  }
}

// WV m-tiles x S slices waves (S > 1 -> WV == 1); grid [ceil(N / 16), ceil(ceil(M / 16) / WV)].
// LDS: (G - gs) groups x 4 blocks x 256 floats (S > 1), gs = ceil(G / S), G = K / 128.
#define G29_LDS_BLOCKS 48
template <int S, bool B16>
__device__ __forceinline__ void g29_body(const unsigned char* __restrict__ qa, const float* __restrict__ sa,
                                         const unsigned char* __restrict__ w, int ldw,
                                         const unsigned short* __restrict__ sc, int ldsc,
                                         float* __restrict__ c, int ldc, int M, int N, int K, int acc,
                                         float* lp) {
#pragma clang fp contract(off)
  const int lane = threadIdx.x & 31;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const int slice = wave % S;
  const int mt = blockIdx.y * ((blockDim.x >> 5) / S) + wave / S;
  const bool mlive = mt * 16 < M;   // a dead m-tile still meets the barrier (S > 1: whole WG alike)
  const int n0 = blockIdx.x * 16;
  const int l16 = lane & 15, h = lane >> 4;
  const int nbs = B16 ? (K >> 4) : (K >> 5);
  const int m = min(mt * 16 + l16, M - 1);
  const unsigned char* ap = qa + (long)m * K + (B16 ? 8 : 16) * h;
  const unsigned char* wp = w + (long)min(n0 + l16, N - 1) * ldw + (B16 ? 8 : 16) * h;
  const float* sap = sa + (long)m * nbs;
  const unsigned short* scp[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) scp[e] = sc + (long)min(n0 + 8 * h + e, N - 1) * ldsc;
  float f[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) f[e] = 0.f;
  const int G = K >> 7;
  const int gs = (G + S - 1) / S;
  const int g_lo = slice * gs, g_hi = min(G, g_lo + gs);
  if (mlive && g_lo < g_hi) {
    G29<B16> cur, nxt;
    g29_load<B16>(cur, wp, ap, sap, scp, g_lo);
    for (int grp = g_lo; grp < g_hi; ++grp) {
      if (grp + 1 < g_hi) g29_load<B16>(nxt, wp, ap, sap, scp, grp + 1);
      if (slice == 0) {
        g29_do<B16, true>(cur, f, nullptr, lane);
      } else {
        if constexpr (!B16) g29_do<B16, false>(cur, f, lp + (long)(grp - gs) * 4 * 256, lane);
      }
      cur = nxt;
    }
  }
  if constexpr (S > 1) {
    __syncthreads();
    if (slice != 0 || !mlive) return;
#pragma unroll 4
    for (int bb = 0; bb < (G - gs) * 4; ++bb) {
      const f8v4* o = (const f8v4*)(lp + (long)bb * 256 + lane * 8);
      const f8v4 x0 = o[0], x1 = o[1];
      f[0] = f[0] + x0[0]; f[1] = f[1] + x0[1]; f[2] = f[2] + x0[2]; f[3] = f[3] + x0[3];
      f[4] = f[4] + x1[0]; f[5] = f[5] + x1[1]; f[6] = f[6] + x1[2]; f[7] = f[7] + x1[3];
    }
  } else {
    if (!mlive) return;
  }
  const int mo = mt * 16 + l16;
  if (mo >= M) return;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int n = n0 + 8 * h + e;
    if (n < N) {
      float* op = c + (long)mo * ldc + n;
      *op = acc ? *op + f[e] : f[e];
    }
  }
}

#define G29_KERNEL(NAME, S, B16)                                                                       \
  extern "C" __global__ __launch_bounds__(512) void NAME(                                              \
      const unsigned char* __restrict__ qa, const float* __restrict__ sa,                             \
      const unsigned char* __restrict__ w, int ldw, const unsigned short* __restrict__ sc, int ldsc,  \
      float* __restrict__ c, int ldc, int M, int N, int K, int acc) {                                  \
    __shared__ float lp[S > 1 ? G29_LDS_BLOCKS * 256 : 1];                                             \
    g29_body<S, B16>(qa, sa, w, ldw, sc, ldsc, c, ldc, M, N, K, acc, lp);                              \
  }
G29_KERNEL(ink29_gemm_i8_kx2, 2, false)
G29_KERNEL(ink29_gemm_i8_kx3, 3, false)
G29_KERNEL(ink29_gemm_i8b16, 1, true)

// ---------------------------------------------------------------------------------- bf16 GEMM
__device__ __forceinline__ unsigned short f2bf_rne(float x) {
  const unsigned u = __builtin_bit_cast(unsigned, x);
  return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// one 16-k step of a lane: 8 weight bf16 (row l16, k 8h..8h+8) + 8 activation f32 -> bf16
struct BwStep {
  i8v4 w;      // 8 bf16
  f8v4 a0, a1;  // 8 f32
};

__device__ __forceinline__ bf16x8 bw_cvt(const f8v4& a0, const f8v4& a1) {
  unsigned short u[8] = {f2bf_rne(a0[0]), f2bf_rne(a0[1]), f2bf_rne(a0[2]), f2bf_rne(a0[3]),
                         f2bf_rne(a1[0]), f2bf_rne(a1[1]), f2bf_rne(a1[2]), f2bf_rne(a1[3])};
  i8v4 p = {(int)(u[0] | ((unsigned)u[1] << 16)), (int)(u[2] | ((unsigned)u[3] << 16)),
            (int)(u[4] | ((unsigned)u[5] << 16)), (int)(u[6] | ((unsigned)u[7] << 16))};
  return __builtin_bit_cast(bf16x8, p);
}

// grid [ceil(N / 16), ceil(M / 16)], block 32 * S (one m-tile, S = K / 768 slices of 48 steps)
template <int S>
__device__ __forceinline__ void bw_body(const float* __restrict__ a, int lda,
                                        const unsigned short* __restrict__ w, int ldw,
                                        float* __restrict__ c, int ldc, int M, int N, int K, int acc) {
#pragma clang fp contract(off)
  __shared__ float red[S > 1 ? (S - 1) * 256 : 1];
  const int lane = threadIdx.x & 31;
  const int slice = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const int mt = blockIdx.y;
  const int n0 = blockIdx.x * 16;
  const int l16 = lane & 15, h = lane >> 4;
  const int m = min(mt * 16 + l16, M - 1);
  const float* ap = a + (long)m * lda + 8 * h;
  const unsigned short* wp = w + (long)min(n0 + l16, N - 1) * ldw + 8 * h;
  const int ks = K / S;   // 768
  const int k_lo = slice * ks;
  f8v8 acc8 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  constexpr int U = 4;   // steps a register group
  BwStep cur[U], nxt[U];
  auto ld = [&](BwStep (&g)[U], int k0) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int k = k0 + 16 * u;
      g[u].w = *(const i8v4*)(wp + k);
      g[u].a0 = *(const f8v4*)(ap + k);
      g[u].a1 = *(const f8v4*)(ap + k + 4);
    }
  };
  ld(cur, k_lo);
  for (int k0 = k_lo; k0 < k_lo + ks; k0 += 16 * U) {
    if (k0 + 16 * U < k_lo + ks) ld(nxt, k0 + 16 * U);
#pragma unroll
    for (int u = 0; u < U; ++u)
      acc8 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(bf16x8, cur[u].w),
                                                              bw_cvt(cur[u].a0, cur[u].a1), acc8);
#pragma unroll
    for (int u = 0; u < U; ++u) cur[u] = nxt[u];
  }
  if constexpr (S > 1) {
    if (slice > 0) {
      f8v4* o = (f8v4*)(red + (slice - 1) * 256 + lane * 8);
      o[0] = f8v4{acc8[0], acc8[1], acc8[2], acc8[3]};
      o[1] = f8v4{acc8[4], acc8[5], acc8[6], acc8[7]};
    }
    __syncthreads();
    if (slice != 0) return;
#pragma unroll
    for (int s = 1; s < S; ++s) {
      const f8v4* o = (const f8v4*)(red + (s - 1) * 256 + lane * 8);
      const f8v4 x0 = o[0], x1 = o[1];
      acc8[0] = acc8[0] + x0[0]; acc8[1] = acc8[1] + x0[1]; acc8[2] = acc8[2] + x0[2];
      acc8[3] = acc8[3] + x0[3]; acc8[4] = acc8[4] + x1[0]; acc8[5] = acc8[5] + x1[1];
      acc8[6] = acc8[6] + x1[2]; acc8[7] = acc8[7] + x1[3];
    }
  }
  const int mo = mt * 16 + l16;
  if (mo >= M) return;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int n = n0 + 8 * h + e;
    if (n < N) {
      float* op = c + (long)mo * ldc + n;
      *op = acc ? *op + acc8[e] : acc8[e];
    }
  }
}

#define BW_KERNEL(NAME, S)                                                                         \
  extern "C" __global__ __launch_bounds__(256) void NAME(                                          \
      const float* __restrict__ a, int lda, const unsigned short* __restrict__ w, int ldw,         \
      float* __restrict__ c, int ldc, int M, int N, int K, int acc) {                              \
    bw_body<S>(a, lda, w, ldw, c, ldc, M, N, K, acc);                                              \
  }
BW_KERNEL(ink29_gemm_bw_s1, 1)
BW_KERNEL(ink29_gemm_bw_s2, 2)
BW_KERNEL(ink29_gemm_bw_s3, 3)
BW_KERNEL(ink29_gemm_bw_s6, 6)

// ------------------------------------------------------------------------------- the route sort
#define RS_E 128
// TR = the tile's rows (16 = ink_route_sort_e128's tables; 32 / 64 = the tables of the fn148 tm2 /
// tm4 cells, item `mt`: an expert's rows padded to TR, tiles of TR rows)
template <int TR>
__device__ __forceinline__ void rs_v2(
    const int* __restrict__ ids, int T, int lo, int nloc,
    int* __restrict__ row_tok, int* __restrict__ row_slot, int* __restrict__ tile_e,
    int* __restrict__ tile_r0, int* __restrict__ tile_n, int* __restrict__ meta,
    int* __restrict__ pos) {
  __shared__ int cnt[RS_E];
  __shared__ int wsum[8];
  __shared__ int s_seg, s_tb, s_last;
  const int tid = threadIdx.x, lane = tid & 31, wv = tid >> 5, N = T * 8;
  const int e = blockIdx.x;
  if (tid < RS_E) cnt[tid] = 0;
  __syncthreads();
  for (int i = tid; i < N; i += 256) {
    const int x = ids[i] - lo;
    if (x >= 0 && x < nloc) atomicAdd(&cnt[x], 1);
  }
  __syncthreads();
  if (tid == 0) {
    int r = 0, tb = 0;
    for (int x = 0; x < nloc; ++x) {
      if (x == e) { s_seg = r; s_tb = tb; }
      const int nt = (cnt[x] + TR - 1) / TR;
      r += nt * TR;
      tb += nt;
    }
    if (e == 0) { meta[0] = tb; meta[1] = r; }
    s_last = -1;
  }
  __syncthreads();
  const int c = cnt[e], seg = s_seg, tb = s_tb;
  const int nt = (c + TR - 1) / TR;
  for (int k = tid; k < nt; k += 256) {
    tile_e[tb + k] = e;
    tile_r0[tb + k] = seg + TR * k;
    tile_n[tb + k] = min(TR, c - TR * k);
  }
  int run = 0;
  for (int b = 0; b < N; b += 256) {
    const int i = b + tid;
    int x = -1;
    if (i < N) {
      x = ids[i] - lo;
      if (x < 0 || x >= nloc) x = -1;
    }
    if (e == 0 && i < N && x < 0) pos[i] = -1;
    const bool hit = x == e;
    const unsigned long long bal = __ballot(hit);
    const unsigned lm = (unsigned)bal & ((1u << lane) - 1u);
    if (lane == 0) wsum[wv] = __popc((unsigned)bal);
    __syncthreads();
    int off = 0, tot = 0;
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      off += q < wv ? wsum[q] : 0;
      tot += wsum[q];
    }
    if (hit) {
      const int row = seg + run + off + __popc(lm);
      row_tok[row] = i >> 3;
      row_slot[row] = i;
      pos[i] = row;
      if (run + off + __popc(lm) == c - 1) s_last = i >> 3;
    }
    run += tot;
    __syncthreads();
  }
  __syncthreads();
  const int padded = nt * TR, lt = s_last;
  for (int k = c + tid; k < padded; k += 256) {
    row_tok[seg + k] = lt;
    row_slot[seg + k] = -1;
  }
}
#define RS_KERNEL(NAME, TR)                                                                         \
  extern "C" __global__ __launch_bounds__(256) void NAME(                                           \
      const int* __restrict__ ids, int T, int lo, int nloc, int* __restrict__ row_tok,              \
      int* __restrict__ row_slot, int* __restrict__ tile_e, int* __restrict__ tile_r0,              \
      int* __restrict__ tile_n, int* __restrict__ meta, int* __restrict__ pos) {                    \
    rs_v2<TR>(ids, T, lo, nloc, row_tok, row_slot, tile_e, tile_r0, tile_n, meta, pos);             \
  }
RS_KERNEL(ink29_route_sort_e128_v2, 16)
RS_KERNEL(ink29_route_sort_e128_v2_t32, 32)
RS_KERNEL(ink29_route_sort_e128_v2_t64, 64)

// ------------------------------------------------------------------------- the message-gate z
// ink_msg_z (hip/ink_fast.hip) BYTE FOR BYTE (the same norm sums, the same per-lane k walk, the
// same butterfly; -ffp-contract=off as every ink_ object), the next 512-wide Wd window's loads in
// registers while the current one is summed, the first window's loads issued before the norms
// (item `mz`): the parent waits on one global round trip a window (9 a row) in series.
#define M29_D 2304
#define M29_PER 9
#define M29_R 64
__device__ __forceinline__ float m29_bf(unsigned short u) {
  return __builtin_bit_cast(float, ((unsigned)u) << 16);
}
__device__ __forceinline__ float m29_sum256(float v, float* red) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
  __syncthreads();
  if (lane == 0) red[w] = v;
  __syncthreads();
  float s = red[0];
#pragma unroll
  for (int i = 1; i < 8; ++i) s = s + red[i];
  return s;
}
extern "C" __global__ __launch_bounds__(256) void ink29_msg_z_p(
    const float* __restrict__ h, int ld_h, const float* __restrict__ m, int ld_m,
    const unsigned short* __restrict__ wd, float* __restrict__ z, float eps) {
  __shared__ float in[2 * M29_D];
  __shared__ float red[8];
  __shared__ unsigned short win[8][512];
  const int t = blockIdx.x, b = blockIdx.y, tid = threadIdx.x, lane = tid & 31, wv = tid >> 5;
  const int r = b * 8 + wv;
  const unsigned short* wr = wd + (long)r * 2 * M29_D;
  uint4 x0 = *reinterpret_cast<const uint4*>(wr + 16 * lane);
  uint4 x1 = *reinterpret_cast<const uint4*>(wr + 16 * lane + 8);
  float hv[M29_PER], mv[M29_PER], sh = 0.f, sm = 0.f;
#pragma unroll
  for (int i = 0; i < M29_PER; ++i) {
    hv[i] = h[(long)t * ld_h + tid + 256 * i];
    mv[i] = m[(long)t * ld_m + tid + 256 * i];
    sh = sh + hv[i] * hv[i];
    sm = sm + mv[i] * mv[i];
  }
  const float rh = 1.0f / sqrtf(m29_sum256(sh, red) / (float)M29_D + eps);
  const float rm = 1.0f / sqrtf(m29_sum256(sm, red) / (float)M29_D + eps);
#pragma unroll
  for (int i = 0; i < M29_PER; ++i) {
    in[tid + 256 * i] = hv[i] * rh;
    in[M29_D + tid + 256 * i] = mv[i] * rm;
  }
  float d = 0.f;
  for (int c0 = 0; c0 < 2 * M29_D; c0 += 512) {
    __syncthreads();   // the previous window's readers are done (and, the first time, `in`)
    *reinterpret_cast<uint4*>(&win[wv][16 * lane]) = x0;
    *reinterpret_cast<uint4*>(&win[wv][16 * lane + 8]) = x1;
    if (c0 + 512 < 2 * M29_D) {
      x0 = *reinterpret_cast<const uint4*>(wr + c0 + 512 + 16 * lane);
      x1 = *reinterpret_cast<const uint4*>(wr + c0 + 512 + 16 * lane + 8);
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int i = c0 + lane + 32 * j;
      d = d + in[i] * m29_bf(win[wv][lane + 32 * j]);
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) d += __shfl_xor(d, off, 32);
  if (lane == 0) z[(long)t * M29_R + r] = d;
}

// out_gate + the b16 quant (item `fq` under `b16`): 2 lanes a (row, 16-k block), y as ink_out_gate,
// codes / scales as ink29_q8_b16.
extern "C" __global__ __launch_bounds__(256) void ink29_out_gate_q8_b16(
    const float* __restrict__ o, int ld_o, const float* __restrict__ g, int ld_g,
    float* __restrict__ y, int ld_y, int T, int N, unsigned char* __restrict__ q,
    float* __restrict__ s) {
#pragma clang fp contract(off)
  const int nb = N >> 4;
  const long gi = (long)blockIdx.x * 256 + threadIdx.x;
  const long blk = gi >> 1;
  const int part = threadIdx.x & 1;
  const bool live = blk < (long)T * nb;
  const long bl = live ? blk : 0;
  const int t = (int)(bl / nb), b = (int)(bl % nb);
  float v[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const long c = b * 16 + part * 8 + j;
    const float sg = 1.0f / (1.0f + expf(-g[(long)t * ld_g + c]));
    v[j] = o[(long)t * ld_o + c] * sg;
  }
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(v[j]));
  amax = fmaxf(amax, __shfl_xor(amax, 1));
  float sc = __fdiv_rn(amax, 127.f);
  if (!(sc > 0.f)) sc = 1.f;
  if (!live) return;
#pragma unroll
  for (int j = 0; j < 8; ++j) y[(long)t * ld_y + b * 16 + part * 8 + j] = v[j];
  unsigned w0 = 0, w1 = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    w0 |= macaq_i8_code(v[j], sc) << (8 * j);
    w1 |= macaq_i8_code(v[4 + j], sc) << (8 * j);
  }
  *(u8v2*)(q + (long)t * N + b * 16 + part * 8) = u8v2{w0, w1};
  if (part == 0) s[(long)t * nb + b] = sc;
}

// the message gate's z on the bf16 matrix cores (item `mzb`, a NUMERICS arm): the normed pair
// [rmsnorm(h) ; rmsnorm(m)] a row (ink_msg_z's norm arithmetic) -> f32 [T, 2D], then
// ink29_gemm_bw_s6 (K 4,608 = 6 slices) x Wd [64, 2D] -> z [T, 64].
extern "C" __global__ __launch_bounds__(256) void ink29_msg_norm2(
    const float* __restrict__ h, int ld_h, const float* __restrict__ m, int ld_m,
    float* __restrict__ out, float eps) {
  __shared__ float red[8];
  const int t = blockIdx.x, tid = threadIdx.x;
  float hv[M29_PER], mv[M29_PER], sh = 0.f, sm = 0.f;
#pragma unroll
  for (int i = 0; i < M29_PER; ++i) {
    hv[i] = h[(long)t * ld_h + tid + 256 * i];
    mv[i] = m[(long)t * ld_m + tid + 256 * i];
    sh = sh + hv[i] * hv[i];
    sm = sm + mv[i] * mv[i];
  }
  const float rh = 1.0f / sqrtf(m29_sum256(sh, red) / (float)M29_D + eps);
  const float rm = 1.0f / sqrtf(m29_sum256(sm, red) / (float)M29_D + eps);
#pragma unroll
  for (int i = 0; i < M29_PER; ++i) {
    out[(long)t * 2 * M29_D + tid + 256 * i] = hv[i] * rh;
    out[(long)t * 2 * M29_D + M29_D + tid + 256 * i] = mv[i] * rm;
  }
}
