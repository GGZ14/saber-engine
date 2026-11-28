// Inkling 2 (inkling2_loop) prefill trunk re-tiles, INK58 (docs/INKLING2.md INK58). Saber's own.
// Every body here is BYTE-EQUAL to the shipped body it replaces (`--ink-p58 <items>`, `none` = off):
//
//   ink58_i8l_m<WM>n<WN>_g<GM>x<GN>  == ink29_gemm_i8b16 (hip/ink_p29.hip) at every output (item `i8`):
//                       an LDS-staged tile (below). Per element the program is the parent's: f = 0; per
//                       32-k block b in order: p0 = f32(z0) * (sa[m,2b] * sw[n,b]); p1 = f32(z1) *
//                       (sa[m,2b+1] * sw[n,b]); f = f + p0; f = f + p1; C = acc ? C + f : f (z0 / z1 = the
//                       exact int sums of the block's two 16-k halves).
//   ink58_bw_t<TN>_s<S> == ink29_gemm_bw_s<S> (item `bw`): the activation fragment converted to bf16 (RNE)
//                       once a 16-k step for TN n-subtiles; the same S slices of 768 k (one WMMA chain each
//                       from 0, the same fragment layout), f = (c0 + c1) + c2 in slice order, the same
//                       epilogue.
// The lab forms INK58 measured and dropped (LEDGER INK58): register-only re-tiles (16 x 64 a wave, no
// LDS) and the magic-number i32 -> f32 (VOPD-pairable) were not faster -- the parent's bound was the
// gathered fragment loads (16-32 cache lines a load), then the scale chain's VALU.
// -ffp-contract=off (saber-kbuild's `ink_` row): no product may fuse into the next add.
#include <hip/hip_runtime.h>

typedef int i8v2 __attribute__((ext_vector_type(2)));
typedef int i8v4 __attribute__((ext_vector_type(4)));
typedef int i8v8 __attribute__((ext_vector_type(8)));
typedef float f8v4 __attribute__((ext_vector_type(4)));
typedef float f8v8 __attribute__((ext_vector_type(8)));
typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ i8v8 p58_wmma(i8v2 a, i8v2 b, i8v8 c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
}

// ------------------------------------------------------------------------- iu8 b16, LDS-staged
// ink58_i8l_m<WM>n<WN>_g<GM>x<GN>: the same per-element program as ink29_gemm_i8b16 on a workgroup tile
// of BM = 16 WM GM rows x BN = 16 WN GN columns (GM x GN waves, each WM x WN 16 x 16 subtiles), the
// activation codes and the weights staged through LDS a 64-k slab at a time (double-buffered, coalesced
// 32-byte block loads), each 32-byte block stored with its 8-byte chunks in the order c0 c2 c1 c3 so a
// lane's two WMMA halves are one 16-byte LDS read (k order inside a 16-k half is the WMMA's exact
// integer sum: unchanged bytes), the weight scales of the slab as f32 in LDS. grid [N / BN, ceil(M / BM)].
// N % BN == 0, K % 64 == 0.
#define P58_LP 80   // LDS row pitch (bytes) of a 64-byte slab row
template <int WM, int WN, int GM, int GN>
__device__ __forceinline__ void p58_i8l_body(const unsigned char* __restrict__ qa, const float* __restrict__ sa,
                                             const unsigned char* __restrict__ w, int ldw,
                                             const unsigned short* __restrict__ sc, int ldsc,
                                             float* __restrict__ c, int ldc, int M, int N, int K, int acc,
                                             unsigned char* lds) {
#pragma clang fp contract(off)
  constexpr int BM = 16 * WM * GM, BN = 16 * WN * GN, NT = 32 * GM * GN;
  constexpr int AS = BM * P58_LP, WS = BN * P58_LP, SS = 2 * BN * 4;
  constexpr int STG = AS + WS + SS;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int wm = wave / GN, wn = wave - wm * GN;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int l16 = lane & 15, h = lane >> 4;
  const int nst = K >> 6;
  // global -> registers: A / W 32-byte blocks (thread u < 2 BM: A block u; 2 BM <= u < 2 (BM + BN): W),
  // the slab's weight scales (thread u < BN: column u's two bf16)
  constexpr int NBLK = 2 * (BM + BN);
  constexpr int PER = (NBLK + NT - 1) / NT;
  i8v4 rx[PER], ry[PER];
  unsigned rs = 0;
  auto gload = [&](int st) {
    const int k0 = st * 64;
#pragma unroll
    for (int q = 0; q < PER; ++q) {
      const int u = tid + q * NT;
      if (u < 2 * BM) {
        const int row = u >> 1, blk = u & 1;
        const unsigned char* src = qa + (long)min(m0 + row, M - 1) * K + k0 + blk * 32;
        rx[q] = *(const i8v4*)src;
        ry[q] = *(const i8v4*)(src + 16);
      } else if (u < NBLK) {
        const int v = u - 2 * BM, col = v >> 1, blk = v & 1;
        const unsigned char* src = w + (long)(n0 + col) * ldw + k0 + blk * 32;
        rx[q] = *(const i8v4*)src;
        ry[q] = *(const i8v4*)(src + 16);
      }
    }
    if (tid < BN) rs = *(const unsigned*)(sc + (long)(n0 + tid) * ldsc + (k0 >> 5));
  };
  auto lstore = [&](int so) {
#pragma unroll
    for (int q = 0; q < PER; ++q) {
      const int u = tid + q * NT;
      if (u < NBLK) {
        const bool isa = u < 2 * BM;
        const int v = isa ? u : u - 2 * BM;
        const int d = so + (isa ? 0 : AS) + (v >> 1) * P58_LP + (v & 1) * 32;
        *(i8v4*)(lds + d) = i8v4{rx[q].x, rx[q].y, ry[q].x, ry[q].y};
        *(i8v4*)(lds + d + 16) = i8v4{rx[q].z, rx[q].w, ry[q].z, ry[q].w};
      }
    }
    if (tid < BN) {
      float* sl = (float*)(lds + so + AS + WS);
      sl[tid] = __builtin_bit_cast(float, rs << 16);
      sl[BN + tid] = __builtin_bit_cast(float, rs & 0xFFFF0000u);
    }
  };
  float f[WM][WN][8];
#pragma unroll
  for (int i = 0; i < WM; ++i)
#pragma unroll
    for (int j = 0; j < WN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) f[i][j][e] = 0.f;
  const float* sap[WM];
#pragma unroll
  for (int i = 0; i < WM; ++i)
    sap[i] = sa + (long)min(m0 + (wm * WM + i) * 16 + l16, M - 1) * (K >> 4);
  gload(0);
  lstore(0);
  __syncthreads();
  for (int st = 0; st < nst; ++st) {
    const int so = (st & 1) * STG;
    if (st + 1 < nst) gload(st + 1);
    f8v4 s4[WM];
#pragma unroll
    for (int i = 0; i < WM; ++i) s4[i] = *(const f8v4*)(sap[i] + st * 4);
    const float* sl = (const float*)(lds + so + AS + WS);
#pragma unroll
    for (int blk = 0; blk < 2; ++blk) {
      i8v4 af[WM], wf[WN];
      float sw[WN][8];
#pragma unroll
      for (int i = 0; i < WM; ++i)
        af[i] = *(const i8v4*)(lds + so + ((wm * WM + i) * 16 + l16) * P58_LP + blk * 32 + 16 * h);
#pragma unroll
      for (int j = 0; j < WN; ++j) {
        wf[j] = *(const i8v4*)(lds + so + AS + ((wn * WN + j) * 16 + l16) * P58_LP + blk * 32 + 16 * h);
        const f8v4 x0 = *(const f8v4*)(sl + blk * BN + (wn * WN + j) * 16 + 8 * h);
        const f8v4 x1 = *(const f8v4*)(sl + blk * BN + (wn * WN + j) * 16 + 8 * h + 4);
        sw[j][0] = x0[0]; sw[j][1] = x0[1]; sw[j][2] = x0[2]; sw[j][3] = x0[3];
        sw[j][4] = x1[0]; sw[j][5] = x1[1]; sw[j][6] = x1[2]; sw[j][7] = x1[3];
      }
#pragma unroll
      for (int i = 0; i < WM; ++i) {
        const float s0 = s4[i][2 * blk], s1 = s4[i][2 * blk + 1];
#pragma unroll
        for (int j = 0; j < WN; ++j) {
          const i8v8 z0 = p58_wmma(wf[j].xy, af[i].xy, i8v8{0, 0, 0, 0, 0, 0, 0, 0});
          const i8v8 z1 = p58_wmma(wf[j].zw, af[i].zw, i8v8{0, 0, 0, 0, 0, 0, 0, 0});
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const float p0 = (float)z0[e] * (s0 * sw[j][e]);
            const float p1 = (float)z1[e] * (s1 * sw[j][e]);
            f[i][j][e] = f[i][j][e] + p0;
            f[i][j][e] = f[i][j][e] + p1;
          }
        }
      }
    }
    if (st + 1 < nst) lstore(((st + 1) & 1) * STG);
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < WM; ++i) {
    const int mo = m0 + (wm * WM + i) * 16 + l16;
    if (mo >= M) continue;
#pragma unroll
    for (int j = 0; j < WN; ++j) {
      float* op = c + (long)mo * ldc + n0 + (wn * WN + j) * 16 + 8 * h;
#pragma unroll
      for (int e = 0; e < 8; ++e) op[e] = acc ? op[e] + f[i][j][e] : f[i][j][e];
    }
  }
}

#define P58_I8L(WM, WN, GM, GN)                                                             \
  extern "C" __global__ __launch_bounds__(32 * GM * GN) void ink58_i8l_m##WM##n##WN##_g##GM##x##GN( \
      const unsigned char* __restrict__ qa, const float* __restrict__ sa,                             \
      const unsigned char* __restrict__ w, int ldw, const unsigned short* __restrict__ sc, int ldsc,  \
      float* __restrict__ c, int ldc, int M, int N, int K, int acc) {                                  \
    __shared__ __attribute__((aligned(16))) unsigned char lds[2 * ((16 * WM * GM + 16 * WN * GN) * P58_LP \
                                                               + 2 * 16 * WN * GN * 4)];               \
    p58_i8l_body<WM, WN, GM, GN>(qa, sa, w, ldw, sc, ldsc, c, ldc, M, N, K, acc, lds);                 \
  }
P58_I8L(2, 1, 4, 4)
P58_I8L(1, 2, 4, 2)

// ------------------------------------------------------------------------------------- bf16 bw
__device__ __forceinline__ unsigned short p58_bf(float x) {
  const unsigned u = __builtin_bit_cast(unsigned, x);
  return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ bf16x8 p58_cvt(const f8v4& a0, const f8v4& a1) {
  unsigned short u[8] = {p58_bf(a0[0]), p58_bf(a0[1]), p58_bf(a0[2]), p58_bf(a0[3]),
                         p58_bf(a1[0]), p58_bf(a1[1]), p58_bf(a1[2]), p58_bf(a1[3])};
  i8v4 p = {(int)(u[0] | ((unsigned)u[1] << 16)), (int)(u[2] | ((unsigned)u[3] << 16)),
            (int)(u[4] | ((unsigned)u[5] << 16)), (int)(u[6] | ((unsigned)u[7] << 16))};
  return __builtin_bit_cast(bf16x8, p);
}

// grid [N / (16 TN), ceil(M / 16)], block 32 S (wave = slice s of one m-tile, S = K / 768 slices of
// 48 16-k steps); N % (16 TN) == 0. LDS: (S - 1) x TN x 256 f32 (the slice partials).
template <int TN, int S>
__device__ __forceinline__ void p58_bw_body(const float* __restrict__ a, int lda,
                                            const unsigned short* __restrict__ w, int ldw,
                                            float* __restrict__ c, int ldc, int M, int N, int K, int acc) {
#pragma clang fp contract(off)
  __shared__ float red[S > 1 ? (S - 1) * TN * 256 : 1];
  const int lane = threadIdx.x & 31;
  const int slice = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const int mt = blockIdx.y;
  const int n0 = blockIdx.x * 16 * TN;
  const int l16 = lane & 15, h = lane >> 4;
  const int m = min(mt * 16 + l16, M - 1);
  const float* ap = a + (long)m * lda + 8 * h;
  const unsigned short* wp[TN];
#pragma unroll
  for (int t = 0; t < TN; ++t) wp[t] = w + (long)(n0 + 16 * t + l16) * ldw + 8 * h;
  const int ks = K / S;
  const int k_lo = slice * ks;
  f8v8 acc8[TN];
#pragma unroll
  for (int t = 0; t < TN; ++t) acc8[t] = f8v8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  constexpr int U = 2;
  f8v4 ca0[U], ca1[U], na0[U], na1[U];
  i8v4 cw[U][TN], nw[U][TN];
  auto ld = [&](f8v4 (&x0)[U], f8v4 (&x1)[U], i8v4 (&ww)[U][TN], int k0) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int k = k0 + 16 * u;
      x0[u] = *(const f8v4*)(ap + k);
      x1[u] = *(const f8v4*)(ap + k + 4);
#pragma unroll
      for (int t = 0; t < TN; ++t) ww[u][t] = *(const i8v4*)(wp[t] + k);
    }
  };
  ld(ca0, ca1, cw, k_lo);
  for (int k0 = k_lo; k0 < k_lo + ks; k0 += 16 * U) {
    if (k0 + 16 * U < k_lo + ks) ld(na0, na1, nw, k0 + 16 * U);
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const bf16x8 av = p58_cvt(ca0[u], ca1[u]);
#pragma unroll
      for (int t = 0; t < TN; ++t)
        acc8[t] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(bf16x8, cw[u][t]), av,
                                                                   acc8[t]);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      ca0[u] = na0[u];
      ca1[u] = na1[u];
#pragma unroll
      for (int t = 0; t < TN; ++t) cw[u][t] = nw[u][t];
    }
  }
  if constexpr (S > 1) {
    if (slice > 0) {
#pragma unroll
      for (int t = 0; t < TN; ++t) {
        f8v4* o = (f8v4*)(red + ((slice - 1) * TN + t) * 256 + lane * 8);
        o[0] = f8v4{acc8[t][0], acc8[t][1], acc8[t][2], acc8[t][3]};
        o[1] = f8v4{acc8[t][4], acc8[t][5], acc8[t][6], acc8[t][7]};
      }
    }
    __syncthreads();
    if (slice != 0) return;
#pragma unroll
    for (int s = 1; s < S; ++s) {
#pragma unroll
      for (int t = 0; t < TN; ++t) {
        const f8v4* o = (const f8v4*)(red + ((s - 1) * TN + t) * 256 + lane * 8);
        const f8v4 x0 = o[0], x1 = o[1];
        acc8[t][0] = acc8[t][0] + x0[0]; acc8[t][1] = acc8[t][1] + x0[1];
        acc8[t][2] = acc8[t][2] + x0[2]; acc8[t][3] = acc8[t][3] + x0[3];
        acc8[t][4] = acc8[t][4] + x1[0]; acc8[t][5] = acc8[t][5] + x1[1];
        acc8[t][6] = acc8[t][6] + x1[2]; acc8[t][7] = acc8[t][7] + x1[3];
      }
    }
  }
  const int mo = mt * 16 + l16;
  if (mo >= M) return;
#pragma unroll
  for (int t = 0; t < TN; ++t) {
    float* op = c + (long)mo * ldc + n0 + 16 * t + 8 * h;
#pragma unroll
    for (int e = 0; e < 8; ++e) op[e] = acc ? op[e] + acc8[t][e] : acc8[t][e];
  }
}

#define P58_BW(NAME, TN, S)                                                                        \
  extern "C" __global__ __launch_bounds__(32 * S) void NAME(                                       \
      const float* __restrict__ a, int lda, const unsigned short* __restrict__ w, int ldw,         \
      float* __restrict__ c, int ldc, int M, int N, int K, int acc) {                              \
    p58_bw_body<TN, S>(a, lda, w, ldw, c, ldc, M, N, K, acc);                                      \
  }
P58_BW(ink58_bw_t4_s2, 4, 2)
P58_BW(ink58_bw_t4_s3, 4, 3)
