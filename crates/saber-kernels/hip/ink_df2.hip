// INK48: the public Inkling-2-Mini DFlash2 block drafter's glue on the card (rank 0; docs/INKLING2.md
// INK47 / INK48). Saber's own arithmetic, f32, built -ffp-contract=off; gated against
// tools/capture/ink47/dflash2_ref.py (the host f32 oracle: argmax-equal drafts). The GEMVs are the
// Inkling drafter's (`ink_dr_gemv_sk_*`, `ink_dr4_mm_*`, the target's GEMV); the selector edge scores
// are the 27B's `dflash_selector_score_r256_k16` (R 256, K 16). The math is our 27B DFlash2 op list
// (`dflash_graph.rs`) at H 2,304 / 144 conv groups / 32 q heads / 8 KV heads / window 2,048.
//
//   ink_df2_cat4     out[r] = [taps0[r] ; taps1[r] ; taps2[r] ; taps3[r]]   (the fc input, 4 x D)
//   ink_df2_rms      out[r] = rmsnorm_g(x[r]) over D
//   ink_df2_hnr      per (row, head): rmsnorm_g over 128 then rope (half split, theta given) at
//                    position ppos[row / div] + row % div + add, in place
//   ink_df2_ring     a K or V row into its slot's 2,048-row ring at (position & 2047)
//   ink_df2_conv     the grouped two-tap block conv: out[t] (=|+=) (b0 + c0[g]) x[t] + (b1 + c1[g]) x[t-1]
//   ink_df2_attn     one (block row, head): softmax over [ring keys of |i - j| < 2048 ; the 8 block
//                    keys] (no causal mask), GQA 4
//   ink_df2_rows     dst[r] = src[(r / k) * 8 + 1 + r % k] (the draft rows of every block)
//   ink_df2_top16    per row: the 16 largest logits (ties -> the lowest index), ids through `map`
//   ink_df2_f2bf     dst = bf16(src) (round to nearest even)
//   ink_df2_pick     dst[i] = src[i * stride] (i32)
//   ink_df2_walk     the greedy walk over the softcapped (after-correction) edge scores + each step's
//                    16-way top-1 probability -> ptoks words 1..k (ids), 9..8+k (f32 bits)
//   ink_df2_i8_w8    INK50: the W8A8 (int8 activation, matrix-core) projection of the iu8 drafter
//                    form (`--ink-draft-layers iu8`): the weight = INK14's 8-bit record AS IT IS
//                    (int8 codes [N, ldw] + bf16 scales a 32-k block, read in place), the activation
//                    = `ink_act_q8_b32_v2`'s codes + f32 scales a (row, 32-k block) (ink_i8.hip's
//                    law). A workgroup = one 16-row m-tile x one 16-column n-tile; its KW waves split
//                    the K / 128 groups into KW CONTIGUOUS runs (KW | K / 128). CONTRACT (numerics;
//                    tools/capture/ink50/i8_oracle.py reproduces it in numpy f32): wave w's partial
//                    f_w = 0; for b in w's blocks in order: i = the exact int32 sum of the block's 32
//                    products; f_w = f_w + f32(i) * (sa[m, b] * f32(sw[n, b])) (each op rounded, no
//                    FMA); then C[m, n] (=|+=) ((f_0 + f_1) + f_2) + ... + f_{KW-1}. KW = 1 is
//                    ink_gemm_i8_w8's contract.
#include <hip/hip_runtime.h>

#include "macaq_i8.h"

#define DF_D 2304
#define DF_HD 128
#define DF_W 2048
#define DF_G 144
#define DF_CP 576
#define DF_MAXS (DF_W + 16)

__device__ __forceinline__ float df_bf2f(unsigned short b) { return __uint_as_float(((unsigned)b) << 16); }

__device__ __forceinline__ float df_wsum32(float v) {
  for (int o = 16; o > 0; o >>= 1) v = v + __shfl_xor(v, o, 32);
  return v;
}

__device__ __forceinline__ float df_bsum256(float v, float* red) {
  v = df_wsum32(v);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  __syncthreads();
  if (lane == 0) red[w] = v;
  __syncthreads();
  float s = 0.f;
  for (int i = 0; i < 8; ++i) s = s + red[i];
  return s;
}

__device__ __forceinline__ float df_bmax256(float v, float* red) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  __syncthreads();
  if (lane == 0) red[w] = v;
  __syncthreads();
  float s = red[0];
  for (int i = 1; i < 8; ++i) s = fmaxf(s, red[i]);
  return s;
}

// grid (rows, 4 * D / 256)
extern "C" __global__ __launch_bounds__(256) void ink_df2_cat4(const float* __restrict__ taps, long plane,
                                                               int row0, float* __restrict__ out) {
  const int r = blockIdx.x;
  const int c = blockIdx.y * 256 + threadIdx.x;  // 0 .. 4D
  const int q = c / DF_D, h = c - q * DF_D;
  out[(long)r * 4 * DF_D + c] = taps[q * plane + (long)(row0 + r) * DF_D + h];
}

// grid (rows), 256 lanes
extern "C" __global__ __launch_bounds__(256) void ink_df2_rms(const float* __restrict__ x, int ldx,
                                                              const unsigned short* __restrict__ gain,
                                                              float* __restrict__ out, int ldo, float eps) {
  __shared__ float red[8];
  const float* xr = x + (long)blockIdx.x * ldx;
  float ss = 0.f;
  for (int i = threadIdx.x; i < DF_D; i += 256) ss = ss + xr[i] * xr[i];
  const float r = 1.0f / sqrtf(df_bsum256(ss, red) / (float)DF_D + eps);
  for (int i = threadIdx.x; i < DF_D; i += 256) out[(long)blockIdx.x * ldo + i] = (xr[i] * r) * df_bf2f(gain[i]);
}

// grid (rows, heads), 64 lanes: lane j owns elements j and j + 64 (the rope pair)
extern "C" __global__ __launch_bounds__(64) void ink_df2_hnr(float* __restrict__ x, int ld,
                                                             const unsigned short* __restrict__ gain,
                                                             const int* __restrict__ ppos, int div, int add,
                                                             float theta, float eps) {
  __shared__ float red[2];
  const int j = threadIdx.x;
  const int row = blockIdx.x;
  float* p = x + (long)row * ld + blockIdx.y * DF_HD;
  const float v1 = p[j], v2 = p[j + 64];
  float s = df_wsum32(v1 * v1 + v2 * v2);
  if ((j & 31) == 0) red[j >> 5] = s;
  __syncthreads();
  s = red[0] + red[1];
  const float r = 1.0f / sqrtf(s / 128.0f + eps);
  const float y1 = (v1 * r) * df_bf2f(gain[j]);
  const float y2 = (v2 * r) * df_bf2f(gain[j + 64]);
  const double pos = (double)(ppos[row / div] + row % div + add);
  const double inv = pow((double)theta, -(double)j / 64.0);
  const double ang = pos * inv;
  const float c = (float)cos(ang), sn = (float)sin(ang);
  p[j] = y1 * c - y2 * sn;
  p[j + 64] = y2 * c + y1 * sn;
}

// grid (rows, n / 256): row r -> slot slots[r] (null: slot 0 of `ring`; < 0: skipped) at position
// ppos[r / div] + r % div + add
extern "C" __global__ __launch_bounds__(256) void ink_df2_ring(const float* __restrict__ src, int ld,
                                                               float* __restrict__ ring, long sstride, int n,
                                                               const int* __restrict__ ppos, int div, int add,
                                                               const int* __restrict__ slots) {
  const int r = blockIdx.x;
  const int slot = slots ? slots[r] : 0;
  if (slot < 0) return;
  const int pos = ppos[r / div] + r % div + add;
  const int c = blockIdx.y * 256 + threadIdx.x;
  if (c >= n) return;
  ring[slot * sstride + (long)(pos & (DF_W - 1)) * n + c] = src[(long)r * ld + c];
}

// grid (rows, D / 256). cf [rows][576] = [side][tap][144 groups]; base bf16 [side][tap][D]
extern "C" __global__ __launch_bounds__(256) void ink_df2_conv(const float* __restrict__ x, int ldx,
                                                               const float* __restrict__ cf,
                                                               const unsigned short* __restrict__ base, int side,
                                                               float* __restrict__ out, int ldo, int acc) {
  const int r = blockIdx.x, t = r & 7;
  const int h = blockIdx.y * 256 + threadIdx.x;
  const int g = h >> 4;
  const float* c = cf + (long)r * DF_CP + side * 2 * DF_G;
  const float k0 = df_bf2f(base[(side * 2 + 0) * DF_D + h]) + c[g];
  float v = k0 * x[(long)r * ldx + h];
  if (t >= 1) {
    const float k1 = df_bf2f(base[(side * 2 + 1) * DF_D + h]) + c[DF_G + g];
    v = v + k1 * x[(long)(r - 1) * ldx + h];
  }
  float* o = out + (long)r * ldo + h;
  *o = acc ? *o + v : v;
}

// grid (rows, 32 heads), 256 lanes. Row r = block row r % 8 of sequence r / 8 at position
// ppos[r / 8] + r % 8; the sequence's ring = kr / vr + slot x sstride (slots null: slot 0); the
// context keys j in [max(0, i - 2047), p - 1], ring row j & 2047; the block keys bk / bv [rows][ldb]
extern "C" __global__ __launch_bounds__(256) void ink_df2_attn(
    const float* __restrict__ q, int ldq, const float* __restrict__ kr, const float* __restrict__ vr,
    long sstride, const float* __restrict__ bk, const float* __restrict__ bv, int ldb,
    const int* __restrict__ ppos, const int* __restrict__ slots, float* __restrict__ o, int ldo,
    float scale) {
  __shared__ float qs[DF_HD];
  __shared__ float sc[DF_MAXS];
  __shared__ float red[8];
  __shared__ float acc2[2][DF_HD];
  const int row = blockIdx.x, h = blockIdx.y, kvh = h >> 2, tid = threadIdx.x;
  const int sq = row >> 3, t = row & 7;
  const int p = ppos[sq];
  const long sb = (long)(slots ? slots[sq] : 0) * sstride;
  const int i = p + t;
  const int lo = max(0, i - (DF_W - 1));
  const int nctx = p - lo, S = nctx + 8;
  const float* bkr = bk + (long)(sq * 8) * ldb + kvh * DF_HD;
  const float* bvr = bv + (long)(sq * 8) * ldb + kvh * DF_HD;
  if (tid < DF_HD) qs[tid] = q[(long)row * ldq + h * DF_HD + tid];
  __syncthreads();
  float mx = -INFINITY;
  for (int j = tid; j < S; j += 256) {
    const float* k = j < nctx ? kr + sb + ((long)((lo + j) & (DF_W - 1)) * 8 + kvh) * DF_HD
                              : bkr + (long)(j - nctx) * ldb;
    float d = 0.f;
    for (int e = 0; e < DF_HD; ++e) d = d + qs[e] * k[e];
    d = d * scale;
    sc[j] = d;
    mx = fmaxf(mx, d);
  }
  mx = df_bmax256(mx, red);
  float sum = 0.f;
  for (int j = tid; j < S; j += 256) {
    const float e = expf(sc[j] - mx);
    sc[j] = e;
    sum = sum + e;
  }
  sum = df_bsum256(sum, red);
  __syncthreads();
  const int d = tid & 127, half = tid >> 7;
  float a = 0.f;
  for (int j = half; j < S; j += 2) {
    const float* v = j < nctx ? vr + sb + ((long)((lo + j) & (DF_W - 1)) * 8 + kvh) * DF_HD
                              : bvr + (long)(j - nctx) * ldb;
    a = a + sc[j] * v[d];
  }
  acc2[half][d] = a;
  __syncthreads();
  if (tid < DF_HD) o[(long)row * ldo + h * DF_HD + tid] = (acc2[0][tid] + acc2[1][tid]) / sum;
}

// grid (B * k, D / 256)
extern "C" __global__ __launch_bounds__(256) void ink_df2_rows(const float* __restrict__ src,
                                                               float* __restrict__ dst, int k) {
  const int r = blockIdx.x;
  const int h = blockIdx.y * 256 + threadIdx.x;
  dst[(long)r * DF_D + h] = src[(long)((r / k) * 8 + 1 + r % k) * DF_D + h];
}

__device__ __forceinline__ bool df_better(float a, int ia, float b, int ib) {
  return a > b || (a == b && ia < ib);
}

// grid (rows), 256 lanes: each lane keeps its 16 best of a strided slice, then 16 rounds of a
// block arg-max over the 4,096 survivors. cand [rows][16] i64 ids (map ? map[i] : i), unary f32.
extern "C" __global__ __launch_bounds__(256) void ink_df2_top16(const float* __restrict__ lg, int ld, int n,
                                                                const unsigned* __restrict__ map,
                                                                long long* __restrict__ cand,
                                                                float* __restrict__ unary) {
  __shared__ float sv[256 * 16];
  __shared__ int si[256 * 16];
  __shared__ float rv[8];
  __shared__ int ri[8];
  const int tid = threadIdx.x;
  const float* x = lg + (long)blockIdx.x * ld;
  float tv[16];
  int ti[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    tv[j] = -INFINITY;
    ti[j] = 0x7fffffff;
  }
  float mv = -INFINITY;
  int mi = 0x7fffffff, ms = 0;
  for (int i = tid; i < n; i += 256) {
    const float v = x[i];
    if (df_better(v, i, mv, mi)) {
#pragma unroll
      for (int j = 0; j < 16; ++j)
        if (j == ms) {
          tv[j] = v;
          ti[j] = i;
        }
      // the new worst of the 16
      mv = tv[0];
      mi = ti[0];
      ms = 0;
#pragma unroll
      for (int j = 1; j < 16; ++j)
        if (df_better(mv, mi, tv[j], ti[j])) {
          mv = tv[j];
          mi = ti[j];
          ms = j;
        }
    }
  }
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    sv[j * 256 + tid] = tv[j];
    si[j * 256 + tid] = ti[j];
  }
  __syncthreads();
  const int lane = tid & 31, w = tid >> 5;
  for (int round = 0; round < 16; ++round) {
    float bv = -INFINITY;
    int bi = 0x7fffffff, bs = -1;
    for (int j = 0; j < 16; ++j) {
      const int s = j * 256 + tid;
      if (si[s] != 0x7fffffff && df_better(sv[s], si[s], bv, bi)) {
        bv = sv[s];
        bi = si[s];
        bs = s;
      }
    }
    float cv = bv;
    int ci = bi;
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor(cv, o, 32);
      const int oi = __shfl_xor(ci, o, 32);
      if (df_better(ov, oi, cv, ci)) {
        cv = ov;
        ci = oi;
      }
    }
    if (lane == 0) {
      rv[w] = cv;
      ri[w] = ci;
    }
    __syncthreads();
    float gv = rv[0];
    int gi = ri[0];
    for (int k = 1; k < 8; ++k)
      if (df_better(rv[k], ri[k], gv, gi)) {
        gv = rv[k];
        gi = ri[k];
      }
    if (bs >= 0 && bi == gi) si[bs] = 0x7fffffff;  // the winner leaves its lane's list
    if (tid == 0) {
      const long o = (long)blockIdx.x * 16 + round;
      cand[o] = (long long)(map ? map[gi] : (unsigned)gi);
      unary[o] = gv;
    }
    __syncthreads();
  }
}

extern "C" __global__ __launch_bounds__(256) void ink_df2_f2bf(const float* __restrict__ src,
                                                               unsigned short* __restrict__ dst, int n) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  const unsigned u = __float_as_uint(src[i]);
  dst[i] = (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

extern "C" __global__ __launch_bounds__(64) void ink_df2_pick(const int* __restrict__ src, int stride,
                                                              int* __restrict__ dst, int n) {
  const int i = threadIdx.x;
  if (i < n) dst[i] = src[i * stride];
}

// grid (B), 32 lanes (lane c < 16 = candidate c). scores [B][k][16 (pred)][16 (cand)] f32, cand
// [B][k][16] i64; sequence s's words at ptoks + s * pstride: 1..k ids, 9..8+k the top-1 probability
extern "C" __global__ __launch_bounds__(32) void ink_df2_walk(const float* __restrict__ scores,
                                                              const long long* __restrict__ cand, int k,
                                                              float cap, int* __restrict__ ptoks, int pstride) {
  const int s = blockIdx.x, c = threadIdx.x;
  int prev = 0;
  for (int l = 0; l < k; ++l) {
    const long pos = (long)s * k + l;
    float v = -INFINITY;
    if (c < 16) v = cap * tanhf(scores[(pos * 16 + prev) * 16 + c] / cap);
    float bv = v;
    int bi = c < 16 ? c : 64;
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor(bv, o, 32);
      const int oi = __shfl_xor(bi, o, 32);
      if (ov > bv || (ov == bv && oi < bi)) {
        bv = ov;
        bi = oi;
      }
    }
    float e = c < 16 ? expf(v - bv) : 0.f;
    e = df_wsum32(e);
    if (c == 0) {
      ptoks[s * pstride + 1 + l] = (int)cand[pos * 16 + bi];
      ptoks[s * pstride + 9 + l] = __float_as_int(1.0f / e);
    }
    prev = bi;
  }
}

// ---------------------------------------------------------------------------------------------------
// INK50: ink_df2_i8_w8 (the contract is in the header). Lane l of a wave's 16 x 16 int32 WMMA result
// holds token mt * 16 + l % 16 and the eight output columns n0 + 8 (l / 16) + e; lane half h loads the
// bytes [16h, 16h + 16) of each 32-k block (the first WMMA sums bytes {0..7, 16..23}, the second
// {8..15, 24..31}: the order inside a block is exact int32). Rows / columns past M / N are clamped on
// the load and never stored. Needs K % 128 == 0, (K / 128) % KW == 0, ldw % 16 == 0, 16-byte W / qa.
typedef int df_i8v2 __attribute__((ext_vector_type(2)));
typedef int df_i8v4 __attribute__((ext_vector_type(4)));
typedef int df_i8v8 __attribute__((ext_vector_type(8)));
typedef unsigned df_u8v2 __attribute__((ext_vector_type(2)));
typedef float df_f8v4 __attribute__((ext_vector_type(4)));

struct DfI8Grp {
  df_i8v4 w[4], a[4];
  df_f8v4 sa;
  df_u8v2 sw[8];
};

__device__ __forceinline__ void df_i8_load(DfI8Grp& g, const unsigned char* __restrict__ wp,
                                           const unsigned char* __restrict__ ap, const float* __restrict__ sap,
                                           const unsigned short* const* scp, int grp) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    g.w[j] = *(const df_i8v4*)(wp + grp * 128 + 32 * j);
    g.a[j] = *(const df_i8v4*)(ap + grp * 128 + 32 * j);
  }
  g.sa = *(const df_f8v4*)(sap + grp * 4);
#pragma unroll
  for (int e = 0; e < 8; ++e) g.sw[e] = *(const df_u8v2*)(scp[e] + grp * 4);
}

__device__ __forceinline__ void df_i8_sum(const DfI8Grp& g, float (&f)[8]) {
#pragma clang fp contract(off)
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    df_i8v8 z = {0, 0, 0, 0, 0, 0, 0, 0};
    z = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, g.w[j].xy, true, g.a[j].xy, z, false);
    z = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, g.w[j].zw, true, g.a[j].zw, z, false);
    const float sa = g.sa[j];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const unsigned d = g.sw[e][j >> 1];
      const float sw = __builtin_bit_cast(float, (j & 1) ? (d & 0xFFFF0000u) : (d << 16));
      const float s = sa * sw;
      const float p = (float)z[e] * s;
      f[e] = f[e] + p;
    }
  }
}

extern "C" __global__ __launch_bounds__(512) void ink_df2_i8_w8(
    const unsigned char* __restrict__ qa, const float* __restrict__ sa, const unsigned char* __restrict__ w,
    int ldw, const unsigned short* __restrict__ sc, int ldsc, float* __restrict__ c, int ldc, int M, int N,
    int K, int acc) {
#pragma clang fp contract(off)
  __shared__ float red[15 * 32 * 8];
  const int lane = threadIdx.x & 31;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const int kw = blockDim.x >> 5;
  const int mt = blockIdx.y, n0 = blockIdx.x * 16;
  const int l16 = lane & 15, h = lane >> 4;
  const int nb = K >> 5;
  const int m = min(mt * 16 + l16, M - 1);
  const unsigned char* ap = qa + (long)m * K + 16 * h;
  const unsigned char* wp = w + (long)min(n0 + l16, N - 1) * ldw + 16 * h;
  const float* sap = sa + (long)m * nb;
  const unsigned short* scp[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) scp[e] = sc + (long)min(n0 + 8 * h + e, N - 1) * ldsc;
  float f[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) f[e] = 0.f;
  const int per = (K >> 7) / kw, g0 = wave * per;
  DfI8Grp cur, nxt;
  df_i8_load(cur, wp, ap, sap, scp, g0);
  for (int g = 0; g < per; ++g) {
    if (g + 1 < per) df_i8_load(nxt, wp, ap, sap, scp, g0 + g + 1);
    df_i8_sum(cur, f);
    cur = nxt;
  }
  if (wave > 0) {
#pragma unroll
    for (int e = 0; e < 8; ++e) red[((wave - 1) * 32 + lane) * 8 + e] = f[e];
  }
  __syncthreads();
  if (wave != 0) return;
  for (int v = 0; v < kw - 1; ++v) {
#pragma unroll
    for (int e = 0; e < 8; ++e) f[e] = f[e] + red[(v * 32 + lane) * 8 + e];
  }
  const int mo = mt * 16 + l16;
  if (mo >= M) return;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int n = n0 + 8 * h + e;
    if (n < N) {
      float* o = c + (long)mo * ldc + n;
      *o = acc ? *o + f[e] : f[e];
    }
  }
}
