// INK49: split-key decode attention for long context (docs/INKLING2.md section 9 INK49). Saber's own;
// its own object so ink_fast / ink_draft / ink_dr3 keep their bytes. Built -ffp-contract=off.
//
// A row's key range is cut into S contiguous segments (S from the key count n, the launch's
// segment cap `smax` = the grid's z and the minimum segment `segmin`; S = 1 at n <= `thresh`):
// one workgroup a (row, head, segment) walks its segment in the parent's per-tile order and
// leaves (max, sum of exp, the weighted V sum) -> a combine workgroup a (row, head) merges the S
// partials (+ the sink). At S = 1 the part kernel runs the parent's whole body and epilogue (the
// combine exits) -> BYTE-EQUAL to the parent; at S > 1 the sums run in another order (a numerics
// arm: `--ink-p49 ks`, gated on the INK7 envelope).
//
//   ink_ks49_part     ink_fast.hip's ink_attn_bias_sink_v1 on one key segment, grid (T, H, smax)
//   ink_ks49_comb     the segments' merge + the sink, grid (T, H), 128 lanes
//   ink_ks49_dr_part  ink_draft.hip's ink_dr_attn / ink_dr3.hip's ink_dr3_attn on one segment of
//                     [ctx keys lo..p-1 ; the 8 block keys], grid (rows, 24, smax); `slots` null =
//                     the one-sequence form (kc0 / vc0 = the slot's own planes)
//   ink_ks49_dr_comb  the drafter segments' merge, grid (rows, 24), 128 lanes
//
// The partials: part[((row * H + h) * smax + s) * KS_LD + {0..127 acc, 128 max, 129 sum}].
#include <hip/hip_runtime.h>

#define FA_HD 128
#define FA_PAGE 64
#define FA_TILE 256
#define KS_LD 132
#define DR_HD 128
#define DR_MAXS 4224

__device__ __forceinline__ float lo16(unsigned w) { return __builtin_bit_cast(float, w << 16); }
__device__ __forceinline__ float hi16(unsigned w) {
  return __builtin_bit_cast(float, w & 0xFFFF0000u);
}
__device__ __forceinline__ float fbf(unsigned short u) {
  return __builtin_bit_cast(float, ((unsigned)u) << 16);
}

// The split law (host twin: inkling2_exec::ks49_split): S segments of `seg` keys (a multiple of
// `tile`), S = 1 at n <= thresh or smax <= 1.
__device__ __forceinline__ void ks_split(int n, int smax, int thresh, int segmin, int tile, int cap,
                                         int& S, int& seg) {
  if (n <= thresh || smax <= 1) {
    S = 1;
    seg = n;
    return;
  }
  int s = min(smax, (n + segmin - 1) / segmin);
  seg = ((n + s - 1) / s + tile - 1) / tile * tile;
  if (seg > cap) seg = cap;
  S = (n + seg - 1) / seg;
}

__device__ __forceinline__ float fa_max(float v, float* red) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
  if (lane == 0) red[w] = v;
  __syncthreads();
  float m = red[0];
#pragma unroll
  for (int i = 1; i < FA_TILE / 32; ++i) m = fmaxf(m, red[i]);
  __syncthreads();
  return m;
}
__device__ __forceinline__ float fa_sum(float v, float* red) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
  if (lane == 0) red[w] = v;
  __syncthreads();
  float s = red[0];
#pragma unroll
  for (int i = 1; i < FA_TILE / 32; ++i) s += red[i];
  __syncthreads();
  return s;
}

// ------------------------------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256) void ink_ks49_part(
    const unsigned short* __restrict__ q, int ld_q,
    const unsigned short* __restrict__ k_pool, const unsigned short* __restrict__ v_pool,
    const int* __restrict__ table, int table_ld,
    const int* __restrict__ row_seq, const int* __restrict__ row_pos,
    const float* __restrict__ rp, int ld_rp,
    const unsigned short* __restrict__ P,
    const float* __restrict__ sink, int ld_sink,
    float* __restrict__ o, int ld_o,
    int H, int KV, int extent, int window,
    float* __restrict__ part, int smax, int thresh, int segmin) {
  __shared__ float qs[FA_HD];
  __shared__ float ps[FA_TILE];
  __shared__ float red[FA_TILE / 32];
  __shared__ float oacc[2][FA_HD];
  __shared__ float rv[16];
  const int t = blockIdx.x, h = blockIdx.y, sg = blockIdx.z, tid = threadIdx.x;
  const int kvh = h / (H / KV);
  const int C = KV * FA_HD;
  const int sq = row_seq[t], pos = row_pos[t];
  const int lo = window > 0 ? max(0, pos - window + 1) : 0;
  int S, seg;
  ks_split(pos - lo + 1, smax, thresh, segmin, FA_TILE, 1 << 30, S, seg);
  if (sg >= S) return;
  const int b0 = lo + sg * seg;
  const int b1 = S == 1 ? pos : min(pos, b0 + seg - 1);   // the segment's last key
  if (tid < FA_HD) qs[tid] = fbf(q[(long)t * ld_q + h * FA_HD + tid]);
  if (tid < 16) rv[tid] = rp[(long)t * ld_rp + h * 16 + tid];
  __syncthreads();
  float m = -INFINITY, l = 0.f, acc = 0.f;
  const int d = tid & (FA_HD - 1), half = tid >> 7;
  for (int base = b0; base <= b1; base += FA_TILE) {
    const int j = base + tid;
    float s = -INFINITY;
    if (j <= b1) {
      const long row = (long)table[(long)sq * table_ld + j / FA_PAGE] * FA_PAGE + j % FA_PAGE;
      const uint4* kr = reinterpret_cast<const uint4*>(k_pool + row * C + kvh * FA_HD);
      uint4 kv4[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) kv4[i] = kr[i];
      float dot = 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const unsigned wv[4] = {kv4[i].x, kv4[i].y, kv4[i].z, kv4[i].w};
#pragma unroll
        for (int qd = 0; qd < 4; ++qd) {
          dot = dot + qs[8 * i + 2 * qd] * lo16(wv[qd]);
          dot = dot + qs[8 * i + 2 * qd + 1] * hi16(wv[qd]);
        }
      }
      s = dot * 0.08838834764831845f;
      const int dist = pos - j;
      if (dist < extent) {
        float rb = 0.f;
#pragma unroll
        for (int k = 0; k < 16; ++k) rb = rb + rv[k] * fbf(P[(long)k * extent + dist]);
        s = s + rb;
      }
    }
    const float tm = fa_max(s, red);
    const float mn = fmaxf(m, tm);
    const float p = (j <= b1) ? expf(s - mn) : 0.f;
    ps[tid] = p;
    const float corr = (m == -INFINITY) ? 0.f : expf(m - mn);
    const float ts = fa_sum(p, red);
    l = l * corr + ts;
    float a = 0.f;
    const int jn = min(FA_TILE / 2, b1 - base - half * (FA_TILE / 2) + 1);
    const int j0 = base + half * (FA_TILE / 2);
    int i = 0;
    for (; i + 8 <= jn; i += 8) {
      float vv[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const int jj = j0 + i + u;
        const long row = (long)table[(long)sq * table_ld + jj / FA_PAGE] * FA_PAGE + jj % FA_PAGE;
        vv[u] = fbf(v_pool[row * C + kvh * FA_HD + d]);
      }
#pragma unroll
      for (int u = 0; u < 8; ++u) a = a + ps[half * (FA_TILE / 2) + i + u] * vv[u];
    }
    for (; i < jn; ++i) {
      const int jj = j0 + i;
      const long row = (long)table[(long)sq * table_ld + jj / FA_PAGE] * FA_PAGE + jj % FA_PAGE;
      a = a + ps[half * (FA_TILE / 2) + i] * fbf(v_pool[row * C + kvh * FA_HD + d]);
    }
    acc = acc * corr + a;
    m = mn;
    __syncthreads();
  }
  oacc[half][d] = acc;
  __syncthreads();
  if (S == 1) {
    // the parent's epilogue, op for op
    const float sk = sink[(long)t * ld_sink + h];
    const float mf = fmaxf(m, sk);
    const float c = expf(m - mf);
    const float lf = l * c + expf(sk - mf);
    if (tid < FA_HD) o[(long)t * ld_o + h * FA_HD + d] = ((oacc[0][d] + oacc[1][d]) * c) / lf;
    return;
  }
  float* pr = part + (((long)t * H + h) * smax + sg) * KS_LD;
  if (tid < FA_HD) pr[d] = oacc[0][d] + oacc[1][d];
  if (tid == 0) {
    pr[128] = m;
    pr[129] = l;
  }
}

extern "C" __global__ __launch_bounds__(128) void ink_ks49_comb(
    const int* __restrict__ row_pos, const float* __restrict__ sink, int ld_sink,
    float* __restrict__ o, int ld_o, int H, int window,
    const float* __restrict__ part, int smax, int thresh, int segmin) {
  const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const int pos = row_pos[t];
  const int lo = window > 0 ? max(0, pos - window + 1) : 0;
  int S, seg;
  ks_split(pos - lo + 1, smax, thresh, segmin, FA_TILE, 1 << 30, S, seg);
  if (S == 1) return;
  const float* pr = part + ((long)t * H + h) * smax * KS_LD;
  const float sk = sink[(long)t * ld_sink + h];
  float mf = sk;
  for (int s = 0; s < S; ++s) mf = fmaxf(mf, pr[s * KS_LD + 128]);
  float lf = 0.f, a = 0.f;
  for (int s = 0; s < S; ++s) {
    const float c = expf(pr[s * KS_LD + 128] - mf);
    lf = lf + pr[s * KS_LD + 129] * c;
    a = a + pr[s * KS_LD + d] * c;
  }
  lf = lf + expf(sk - mf);
  o[(long)t * ld_o + h * FA_HD + d] = a / lf;
}

// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ float wsum32(float v) {
  for (int o = 16; o > 0; o >>= 1) v = v + __shfl_xor(v, o, 32);
  return v;
}
__device__ __forceinline__ float bsum256(float v, float* red) {
  v = wsum32(v);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  __syncthreads();
  if (lane == 0) red[w] = v;
  __syncthreads();
  float s = 0.f;
  for (int i = 0; i < 8; ++i) s = s + red[i];
  return s;
}
__device__ __forceinline__ float bmax256(float v, float* red) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  __syncthreads();
  if (lane == 0) red[w] = v;
  __syncthreads();
  float s = red[0];
  for (int i = 1; i < 8; ++i) s = fmaxf(s, red[i]);
  return s;
}

// the drafter: S = 1 only when the whole [ctx ; block] range fits the parent's score plane
extern "C" __global__ __launch_bounds__(256) void ink_ks49_dr_part(
    const float* __restrict__ q, int ldq, const float* __restrict__ kc0, const float* __restrict__ vc0,
    long cstride, const int* __restrict__ slots, const float* __restrict__ bk,
    const float* __restrict__ bv, int ldb, const int* __restrict__ ppos, int window,
    float* __restrict__ o, int ldo, float scale,
    float* __restrict__ part, int smax, int thresh, int segmin) {
  __shared__ float qs[DR_HD];
  __shared__ float sc[DR_MAXS];
  __shared__ float red[8];
  __shared__ float acc2[2][DR_HD];
  const int row = blockIdx.x, h = blockIdx.y, sg = blockIdx.z, kvh = h / 3, tid = threadIdx.x;
  const int H = gridDim.y;
  const int b = row / 8;
  const float* kc = slots ? kc0 + (long)slots[b] * cstride : kc0;
  const float* vc = slots ? vc0 + (long)slots[b] * cstride : vc0;
  const int p = ppos[b];
  const int lo = window > 0 ? max(0, p - window) : 0;
  const int nctx = p - lo, Stot = nctx + 8;
  int S, seg;
  ks_split(Stot, smax, min(thresh, DR_MAXS), segmin, 256, DR_MAXS, S, seg);
  if (sg >= S) return;
  if (tid < DR_HD) qs[tid] = q[(long)row * ldq + h * DR_HD + tid];
  __syncthreads();
  if (S == 1) {
    // the parent's body, op for op
    float mx = -INFINITY;
    for (int j = tid; j < Stot; j += 256) {
      const float* kr = j < nctx ? kc + ((long)(lo + j) * 8 + kvh) * DR_HD
                                 : bk + (long)(b * 8 + j - nctx) * ldb + kvh * DR_HD;
      float dd = 0.f;
      for (int e = 0; e < DR_HD; ++e) dd = dd + qs[e] * kr[e];
      dd = dd * scale;
      sc[j] = dd;
      mx = fmaxf(mx, dd);
    }
    mx = bmax256(mx, red);
    float sum = 0.f;
    for (int j = tid; j < Stot; j += 256) {
      const float e = expf(sc[j] - mx);
      sc[j] = e;
      sum = sum + e;
    }
    sum = bsum256(sum, red);
    __syncthreads();
    const int d = tid & 127, half = tid >> 7;
    float a = 0.f;
    for (int j = half; j < Stot; j += 2) {
      const float* vr = j < nctx ? vc + ((long)(lo + j) * 8 + kvh) * DR_HD
                                 : bv + (long)(b * 8 + j - nctx) * ldb + kvh * DR_HD;
      a = a + sc[j] * vr[d];
    }
    acc2[half][d] = a;
    __syncthreads();
    if (tid < DR_HD) o[(long)row * ldo + h * DR_HD + tid] = (acc2[0][tid] + acc2[1][tid]) / sum;
    return;
  }
  // one segment [j0, j1) of the combined index (scores in the plane: seg <= DR_MAXS)
  const int j0 = sg * seg, j1 = min(Stot, j0 + seg), n = j1 - j0;
  float mx = -INFINITY;
  for (int jl = tid; jl < n; jl += 256) {
    const int j = j0 + jl;
    const float* kr = j < nctx ? kc + ((long)(lo + j) * 8 + kvh) * DR_HD
                               : bk + (long)(b * 8 + j - nctx) * ldb + kvh * DR_HD;
    const float4* k4 = reinterpret_cast<const float4*>(kr);
    float dd = 0.f;
#pragma unroll 4
    for (int e = 0; e < DR_HD / 4; ++e) {
      const float4 kk = k4[e];
      dd = dd + qs[4 * e] * kk.x;
      dd = dd + qs[4 * e + 1] * kk.y;
      dd = dd + qs[4 * e + 2] * kk.z;
      dd = dd + qs[4 * e + 3] * kk.w;
    }
    dd = dd * scale;
    sc[jl] = dd;
    mx = fmaxf(mx, dd);
  }
  mx = bmax256(mx, red);
  float sum = 0.f;
  for (int jl = tid; jl < n; jl += 256) {
    const float e = expf(sc[jl] - mx);
    sc[jl] = e;
    sum = sum + e;
  }
  sum = bsum256(sum, red);
  __syncthreads();
  const int d = tid & 127, half = tid >> 7;
  float a = 0.f;
  for (int jl = half; jl < n; jl += 2) {
    const int j = j0 + jl;
    const float* vr = j < nctx ? vc + ((long)(lo + j) * 8 + kvh) * DR_HD
                               : bv + (long)(b * 8 + j - nctx) * ldb + kvh * DR_HD;
    a = a + sc[jl] * vr[d];
  }
  acc2[half][d] = a;
  __syncthreads();
  float* pr = part + (((long)row * H + h) * smax + sg) * KS_LD;
  if (tid < DR_HD) pr[tid] = acc2[0][tid] + acc2[1][tid];
  if (tid == 0) {
    pr[128] = mx;
    pr[129] = sum;
  }
}

extern "C" __global__ __launch_bounds__(128) void ink_ks49_dr_comb(
    const int* __restrict__ ppos, int window, float* __restrict__ o, int ldo,
    const float* __restrict__ part, int smax, int thresh, int segmin) {
  const int row = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const int H = gridDim.y;
  const int p = ppos[row / 8];
  const int lo = window > 0 ? max(0, p - window) : 0;
  int S, seg;
  ks_split(p - lo + 8, smax, min(thresh, DR_MAXS), segmin, 256, DR_MAXS, S, seg);
  if (S == 1) return;
  const float* pr = part + ((long)row * H + h) * smax * KS_LD;
  float mf = -INFINITY;
  for (int s = 0; s < S; ++s) mf = fmaxf(mf, pr[s * KS_LD + 128]);
  float lf = 0.f, a = 0.f;
  for (int s = 0; s < S; ++s) {
    const float c = expf(pr[s * KS_LD + 128] - mf);
    lf = lf + pr[s * KS_LD + 129] * c;
    a = a + pr[s * KS_LD + d] * c;
  }
  o[(long)row * ldo + h * DR_HD + d] = a / lf;
}
