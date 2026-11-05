// Saber gated-delta-net kernels, CLEANROOM-2 (CR2-3c): the per-chunk solve A = (I + N)^-1 of the WY prefill
// (spec 2.7), N = strict_lower(beta_i e^(g_i - g_j) k_i . k_j) over a 64-token chunk, in bf16.
//
// One workgroup = (chunk, k head), four waves. The k head's Gram K K^T is formed once (10 upper 16x16 tiles,
// bf16 WMMA chains straight from global) and reused by every v head of the group (R = H / Hg): per v head the
// gated, masked N is written into one 64 x 68 bf16 LDS matrix, which is then inverted in place -- the four
// diagonal 16-blocks by forward substitution in f32, two 32-merges, one 64-merge (each a pair of WMMA chains
// through a 32 x 36 bf16 scratch) -- and stored.
#include "sgdn_prims.h"

namespace sgdn {

constexpr int kKktPitch = 68;    // LDS row pitch of the chunk matrix (bf16)
constexpr int kKktTmp = 36;      // LDS row pitch of the merge scratch

// flat chunk id -> (sequence, chunk start); false past the last chunk
__device__ __forceinline__ bool chunk_of(int cid, sgdn::ci32 cu, int nseq, int& bos, int& t0,
                                         int& rows) {
  for (int n = 0; n < nseq; ++n) {
    const int b = cu[n], len = cu[n + 1] - b;
    const int nch = (len + kChunk - 1) / kChunk;
    if (cid < nch) {
      bos = b;
      t0 = cid * kChunk;
      rows = min(kChunk, len - t0);
      return true;
    }
    cid -= nch;
  }
  return false;
}

template <int R>
__device__ __forceinline__ void kkt_solve(const u16* __restrict__ k, sgdn::cf32 beta,
                                          sgdn::cf32 g, u16* __restrict__ a_out,
                                          sgdn::ci32 cu, int nseq, int H, int Hg) {
  __shared__ __attribute__((aligned(16))) u16 sA[kChunk][kKktPitch];
  __shared__ __attribute__((aligned(16))) u16 sT[32][kKktTmp];
  __shared__ float sg[kChunk], sb[kChunk];
  int bos, t0, rows;
  if (!chunk_of(blockIdx.x, cu, nseq, bos, t0, rows)) return;
  const int hg = blockIdx.y;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int c = lane & 15, h = lane >> 4;
  const long tok0 = (long)bos + t0;

  // the chunk matrix starts at zero: the blocks above the diagonal are never written
  {
    u32x4* z = (u32x4*)&sA[0][0];
    for (int i = tid; i < (int)(sizeof(sA) / 16); i += 128) z[i] = u32x4{0u, 0u, 0u, 0u};
  }

  // Gram tiles (mt, nt), nt >= mt, row-major over the upper triangle; tile ti belongs to wave ti mod 4
  const u16* kh = k + tok0 * Hg * kDim + (long)hg * kDim;
  const long kpitch = (long)Hg * kDim;
  f32x8 P[3];
  int tm[3], tn[3];
#pragma unroll
  for (int s = 0; s < 3; ++s) {
    const int ti = wave + 4 * s;
    tm[s] = ti < 4 ? 0 : (ti < 7 ? 1 : (ti < 9 ? 2 : 3));
    tn[s] = tm[s] + ti - (4 * tm[s] - (tm[s] * (tm[s] - 1)) / 2);
    P[s] = zero8();
    if (ti < 10) {
      const u16* ra = kh + (long)min(16 * tm[s] + c, rows - 1) * kpitch;
      const u16* rb = kh + (long)min(16 * tn[s] + c, rows - 1) * kpitch;
#pragma unroll
      for (int sk = 0; sk < 8; ++sk) P[s] = wmma(frag_row(ra + 16 * sk, h), frag_row(rb + 16 * sk, h), P[s]);
    }
  }
  const int ntiles = wave < 2 ? 3 : 2;

  // the gates of v head hv for the chunk's rows (threads < 64), loaded one head ahead
  const long gsrc = (tok0 + min(tid, rows - 1)) * H;
  float ng = 0.0f, nb = 0.0f;
  if (tid < kChunk) {
    ng = g[gsrc + hg * R];
    nb = beta[gsrc + hg * R];
  }
  for (int hh = 0; hh < R; ++hh) {
    const int hv = hg * R + hh;
    if (tid < kChunk) {
      sg[tid] = ng;
      sb[tid] = nb;
      if (hh + 1 < R) {
        ng = g[gsrc + hv + 1];
        nb = beta[gsrc + hv + 1];
      }
    }
    __syncthreads();

    // N: A[i][j] = (beta_i * E(g_i - g_j)) * (k_j . k_i), masked to i > j inside the live rows
#pragma unroll
    for (int s = 0; s < 3; ++s) {
      if (s < ntiles) {
        const int i = 16 * tn[s] + c;
        const bool live = i < rows;
        const float bi = live ? sb[i] : 0.0f;
        const float gi = live ? sg[i] : 0.0f;
        float val[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int j = 16 * tm[s] + 8 * h + e;
          const float d = (live && i > j && j < rows) ? gi - sg[j] : -INFINITY;
          val[e] = (bi * E(d)) * P[s][e];
        }
        u16* dst = &sA[i][16 * tm[s] + 8 * h];
        *(u32x2*)dst = u32x2{away16x2(val[0], val[1]), away16x2(val[2], val[3])};
        *(u32x2*)(dst + 4) = u32x2{away16x2(val[4], val[5]), away16x2(val[6], val[7])};
      }
    }
    __syncthreads();

    // diagonal blocks: wave p, lane c computes column c of (I + N_p)^-1 by forward substitution
    {
      const int p = wave;
      float d[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < i; ++j) acc = acc + bf(sA[16 * p + i][16 * p + j]) * d[j];
        d[i] = (i == c ? 1.0f : 0.0f) - acc;
      }
      if (lane < 16) {
#pragma unroll
        for (int i = 0; i < 16; ++i) sA[16 * p + i][16 * p + c] = away16(d[i]);
      }
    }
    __syncthreads();

    // 32-merges: block blk's lower-left = -D1 (N10 D0)
    if (wave < 2) {
      const int r0 = 32 * wave;
      const f32x8 X = wmma(frag_row(&sA[r0 + 16 + c][r0], h), frag_col(&sA[r0][r0 + c], kKktPitch, h), zero8());
      u16* t = &sT[16 * wave + c][8 * h];
      *(u32x2*)t = u32x2{away16x2(X[0], X[1]), away16x2(X[2], X[3])};
      *(u32x2*)(t + 4) = u32x2{away16x2(X[4], X[5]), away16x2(X[6], X[7])};
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
      __builtin_amdgcn_wave_barrier();
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
      const f32x8 Y = wmma(frag_row(&sA[r0 + 16 + c][r0 + 16], h), frag_row(&sT[16 * wave + c][0], h), zero8());
#pragma unroll
      for (int e = 0; e < 8; ++e) sA[r0 + 16 + 8 * h + e][r0 + c] = away16(-Y[e]);
    }
    __syncthreads();

    // 64-merge: lower-left 32 x 32 = -Q^-1 (R P^-1), one 16 x 16 tile a wave
    {
      const int mt = wave >> 1, nt = wave & 1;
      f32x8 Y = zero8();
#pragma unroll 1
      for (int sk = 0; sk < 2; ++sk)   // a rolled two-step chain (the census reads it as the head loop's inner loop)
        Y = wmma(frag_row(&sA[32 + 16 * mt + c][16 * sk], h), frag_col(&sA[16 * sk][16 * nt + c], kKktPitch, h), Y);
      u16* t = &sT[16 * nt + c][16 * mt + 8 * h];
      *(u32x2*)t = u32x2{away16x2(Y[0], Y[1]), away16x2(Y[2], Y[3])};
      *(u32x2*)(t + 4) = u32x2{away16x2(Y[4], Y[5]), away16x2(Y[6], Y[7])};
      __syncthreads();
      f32x8 Z = zero8();
#pragma unroll
      for (int sk = 0; sk < 2; ++sk)
        Z = wmma(frag_row(&sA[32 + 16 * mt + c][32 + 16 * sk], h), frag_row(&sT[16 * nt + c][16 * sk], h), Z);
#pragma unroll
      for (int e = 0; e < 8; ++e) sA[32 + 16 * mt + 8 * h + e][16 * nt + c] = away16(-Z[e]);
    }
    __syncthreads();

    // rows r < rows out, eight threads a row, 16 bytes each, in passes of 16 rows (a uniform trip count)
    const int npass = (rows + 15) >> 4;
#pragma unroll 1
    for (int pass = 0; pass < npass; ++pass) {
      const int r = (tid >> 3) + 16 * pass;
      if (r < rows) {
        const int q8 = 8 * (tid & 7);
        const u32x2 lo = *(const u32x2*)&sA[r][q8];
        const u32x2 hi = *(const u32x2*)&sA[r][q8 + 4];
        *(u32x4*)(a_out + ((tok0 + r) * H + hv) * kChunk + q8) = u32x4{lo.x, lo.y, hi.x, hi.y};
      }
    }
    __syncthreads();
  }
}

}  // namespace sgdn

#define SGDN_KKT(R)                                                                                              \
  extern "C" __global__ __launch_bounds__(128) void saber_gdn_kkt_solve_k128_c64_r##R(                           \
      sgdn::cbf k, sgdn::cf32 beta, sgdn::cf32 g,         \
      sgdn::wbf a_out, sgdn::ci32 cu, int n_seqs, int h, int hg) {              \
    sgdn::kkt_solve<R>(k, beta, g, a_out, cu, n_seqs, h, hg);                                                    \
  }

SGDN_KKT(1)
SGDN_KKT(2)
SGDN_KKT(3)
SGDN_KKT(4)
SGDN_KKT(6)
SGDN_KKT(8)
