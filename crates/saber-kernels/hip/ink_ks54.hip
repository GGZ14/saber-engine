// INK54: the row-shared split-key VERIFY attention for the target's global layers (docs/INKLING2.md
// section 9 INK54). Saber's own; its own object so ink_ks49 / ink_fast keep their bytes. Built
// -ffp-contract=off.
//
// INK49's `ink_ks49_part` runs one workgroup a (row, head, segment): a T-row verify of one sequence
// re-reads every key and value of a segment once per row and per query head of the KV head. Here ONE
// workgroup a (run of <= 8 consecutive rows of one sequence, KV head [, query-head group], segment)
// loads each key / value once and serves every (row, query head) pair of the run:
//   * every row keeps ITS OWN split (`ks_split` of its own key count, the same law as ks49_part), so
//     a row's segment s covers the same keys as in INK49's kernel; with the global layer's window 0
//     every segment starts on the 256-key tile grid, so the rows' tiles coincide where they overlap;
//   * a row takes part in a tile only when the tile lies inside its own segment, and inside a tile
//     every logit, max, sum of exp and V accumulation is INK49's expression in INK49's order (the
//     reductions are the same shuffle tree + the same sequential 8-wave fold, the V sum the same
//     sequential walk per half) -> every row's partial (and its S = 1 output) is BYTE-EQUAL to
//     ink_ks49_part's, and ink_ks49_comb merges them unchanged.
//
//   ink_ks54_rv1  grid (>= groups, KV * (H / KV), smax): one query head a workgroup, R <= 8 rows
//   ink_ks54_rv3  grid (>= groups, KV, smax): the three query heads of a KV head, R <= 8 rows
//   ink_ks54_rv1r4  rv1 at R <= 4 rows a workgroup
//   ink_ks54_short  launched beside each: the groups whose rows are all at S = 1 run INK49's per-row
//                   body there (one workgroup a row), the row-shared kernels skip them
// Workgroup x serves group x (the rows' runs of one sequence cut every R rows; `rows` = T); a grid
// wider than the group count exits. The partials' layout is INK49's:
// part[((row * H + h) * smax + s) * KS_LD + {0..127 acc, 128 max, 129 sum}].
#include <hip/hip_runtime.h>

#define FA_HD 128
#define FA_PAGE 64
#define FA_TILE 256
#define KS_LD 132

__device__ __forceinline__ float lo16(unsigned w) { return __builtin_bit_cast(float, w << 16); }
__device__ __forceinline__ float hi16(unsigned w) {
  return __builtin_bit_cast(float, w & 0xFFFF0000u);
}
__device__ __forceinline__ float fbf(unsigned short u) {
  return __builtin_bit_cast(float, ((unsigned)u) << 16);
}

// INK49's split law (ink_ks49.hip `ks_split`, host twin inkling2_exec::ks49_split), op for op.
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

// INK49's `ink_ks49_part` body, op for op: `ink_ks54_short` runs it for the rows of every group whose
// rows are all at S = 1 (<= thresh keys: short context), where one workgroup a row has more
// parallelism than a shared walk (and the row-shared kernel's registers would cost it occupancy).
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

__device__ __forceinline__ void ks54_row_body(
    const unsigned short* __restrict__ q, int ld_q,
    const unsigned short* __restrict__ k_pool, const unsigned short* __restrict__ v_pool,
    const int* __restrict__ table, int table_ld,
    const int* __restrict__ row_seq, const int* __restrict__ row_pos,
    const float* __restrict__ rp, int ld_rp,
    const unsigned short* __restrict__ P,
    const float* __restrict__ sink, int ld_sink,
    float* __restrict__ o, int ld_o,
    int H, int KV, int extent, int window,
    float* __restrict__ part, int smax, int thresh, int segmin, int h) {
  __shared__ float qs[FA_HD];
  __shared__ float ps[FA_TILE];
  __shared__ float red[FA_TILE / 32];
  __shared__ float oacc[2][FA_HD];
  __shared__ float rv[16];
  const int t = blockIdx.x, sg = blockIdx.z, tid = threadIdx.x;   // h: the caller's
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

// the group of <= R rows of one sequence starting at row g0: every row at S = 1 (<= thresh keys)
__device__ __forceinline__ bool ks54_all1(const int* __restrict__ row_seq, const int* __restrict__ row_pos,
                                          int T, int g0, int R, int sq, int smax, int thresh) {
  if (smax <= 1) return true;
  for (int r = g0; r < T && r < g0 + R && row_seq[r] == sq; ++r) {
    if (row_pos[r] + 1 > thresh) return false;
  }
  return true;
}

// V staging: pass q holds keys base + half * 128 + q * VK + k (k < VK) of both halves; thread tid
// moves VK / 8 of the pass's 32 * VK 16-byte pieces (key row = piece / 16, column = piece % 16).
template <int VK>
__device__ __forceinline__ void ks54_vload(uint4 (&vr)[VK / 8], const unsigned short* __restrict__ v_pool,
                                           const long* prow, int base, int q, int jmax, int C,
                                           int kvh, int tid) {
#pragma unroll
  for (int m = 0; m < VK / 8; ++m) {
    const int e = tid + FA_TILE * m, r = e / 16, c = e % 16;
    const int k = (r / VK) * (FA_TILE / 2) + q * VK + r % VK;   // the key in the tile
    vr[m] = make_uint4(0, 0, 0, 0);
    if (base + k <= jmax) {
      const long row = prow[k / FA_PAGE] + k % FA_PAGE;
      vr[m] = reinterpret_cast<const uint4*>(v_pool + row * C + kvh * FA_HD)[c];
    }
  }
}
template <int VK>
__device__ __forceinline__ void ks54_vstore(unsigned short* vs, const uint4 (&vr)[VK / 8], int tid) {
#pragma unroll
  for (int m = 0; m < VK / 8; ++m) {
    const int e = tid + FA_TILE * m;
    reinterpret_cast<uint4*>(vs)[e] = vr[m];   // [r][c * 8 .. +8] = r * 128 + c * 8 halves
  }
}

template <int G, int VK, int RV_R>
__device__ __forceinline__ void ks54_rows(
    const unsigned short* __restrict__ q, int ld_q,
    const unsigned short* __restrict__ k_pool, const unsigned short* __restrict__ v_pool,
    const int* __restrict__ table, int table_ld,
    const int* __restrict__ row_seq, const int* __restrict__ row_pos,
    const float* __restrict__ rp, int ld_rp,
    const unsigned short* __restrict__ P,
    const float* __restrict__ sink, int ld_sink,
    float* __restrict__ o, int ld_o,
    int H, int KV, int extent,
    float* __restrict__ part, int smax, int thresh, int segmin, int T) {
  constexpr int NP = RV_R * G;
  __shared__ float qs[NP][FA_HD];
  __shared__ __attribute__((aligned(16))) float ps[NP][FA_TILE];   // the tile's p; after the walk the two halves' accumulators
  __shared__ float redm[NP][FA_TILE / 32];
  __shared__ float reds[NP][FA_TILE / 32];
  __shared__ float rv[NP][16];
  // a pair's running max / sum and the tile's new max / rescale (uniform over the workgroup: every
  // value is INK49's expression, computed once by lane `pair` instead of by every thread)
  __shared__ float sm[NP], sl[NP], smn[NP], scorr[NP];
  // a row's position, segment bounds [b0, b1] and segment count
  __shared__ int rpos[RV_R], rb0[RV_R], rb1[RV_R], rS[RV_R];
  // the tile's 4 page rows (one table read a page, not a key) and one V pass: [2 halves][64 keys][128]
  __shared__ long prow[FA_TILE / FA_PAGE];
  __shared__ __attribute__((aligned(16))) unsigned short vs[2 * VK * FA_HD];
  const int sg = blockIdx.z, tid = threadIdx.x;
  const int GQ = H / KV, ng = GQ / G;
  const int kvh = blockIdx.y / ng, h0 = kvh * GQ + (blockIdx.y % ng) * G;
  const int C = KV * FA_HD;
  // blockIdx.x = the group: the runs of consecutive rows of one sequence, cut every R rows, in row
  // order (a uniform scan over <= T row words); a grid past the group count exits
  int t = -1, nr = 0;
  {
    int r = 0, k = 0;
    while (r < T && t < 0) {
      const int s0 = row_seq[r];
      int e = r;
      while (e < T && row_seq[e] == s0) ++e;
      for (int g = r; g < e; g += RV_R) {
        if (k == (int)blockIdx.x) {
          t = g;
          nr = min(RV_R, e - g);
        }
        ++k;
      }
      r = e;
    }
  }
  if (t < 0) return;
  const int sq = row_seq[t];
  // a group whose rows are all at S = 1 is `ink_ks54_short`'s (INK49's per-row body, launched beside)
  if (ks54_all1(row_seq, row_pos, T, t, RV_R, sq, smax, thresh)) return;
  if (tid < RV_R) {
    int p = 0, S = 1, lo = 0, hi = -1;
    if (tid < nr) {
      p = row_pos[t + tid];
      int seg;
      ks_split(p + 1, smax, thresh, segmin, FA_TILE, 1 << 30, S, seg);
      if (sg < S) {
        lo = sg * seg;
        hi = S == 1 ? p : min(p, lo + seg - 1);
      }
    }
    rpos[tid] = p;
    rb0[tid] = lo;
    rb1[tid] = hi;
    rS[tid] = S;
  }
  for (int i = tid; i < nr * G * FA_HD; i += FA_TILE) {
    const int pi = i / FA_HD, d = i % FA_HD;
    qs[pi][d] = fbf(q[(long)(t + pi / G) * ld_q + (h0 + pi % G) * FA_HD + d]);
  }
  for (int i = tid; i < nr * G * 16; i += FA_TILE) {
    const int pi = i / 16, k = i % 16;
    rv[pi][k] = rp[(long)(t + pi / G) * ld_rp + (h0 + pi % G) * 16 + k];
  }
  if (tid < NP) {
    sm[tid] = -INFINITY;
    sl[tid] = 0.f;
  }
  __syncthreads();
  // act: rows with a segment sg (bit r); the tile range over them
  int act = 0, tlo = 1 << 30, thi = -1;
  for (int r = 0; r < nr; ++r) {
    if (rb1[r] >= 0) {
      act |= 1 << r;
      tlo = min(tlo, rb0[r]);
      thi = max(thi, rb1[r]);
    }
  }
  if (act == 0) return;
  float acc[NP];
#pragma unroll
  for (int pi = 0; pi < NP; ++pi) acc[pi] = 0.f;
  const int d = tid & (FA_HD - 1), half = tid >> 7;
  const int lane = tid & 31, w = tid >> 5;
  const int* trow = table + (long)sq * table_ld;
  for (int base = tlo; base <= thi; base += FA_TILE) {
    // in: the rows whose own segment holds this tile; b1[r] = -1 for a row outside it
    int b1[RV_R];
    int in = 0, jmax = -1;
#pragma unroll
    for (int r = 0; r < RV_R; ++r) {
      const bool ir = ((act >> r) & 1) && base >= rb0[r] && base <= rb1[r];
      b1[r] = ir ? rb1[r] : -1;
      in |= ir ? 1 << r : 0;
      jmax = max(jmax, b1[r]);
    }
    if (tid < FA_TILE / FA_PAGE) {
      const int pg = base / FA_PAGE + tid;
      prow[tid] = pg * FA_PAGE <= jmax ? (long)trow[pg] * FA_PAGE : 0;
    }
    __syncthreads();
    // V pass 0's loads go out now (they land while the scores run)
    uint4 vr[VK / 8];
    ks54_vload<VK>(vr, v_pool, prow, base, 0, jmax, C, kvh, tid);
    const int j = base + tid;
    float s[NP];
#pragma unroll
    for (int pi = 0; pi < NP; ++pi) s[pi] = -INFINITY;
    if (j <= jmax) {
      const long row = prow[tid / FA_PAGE] + tid % FA_PAGE;
      const uint4* kr = reinterpret_cast<const uint4*>(k_pool + row * C + kvh * FA_HD);
      uint4 kv4[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) kv4[i] = kr[i];
#pragma unroll
      for (int r = 0; r < RV_R; ++r) {
        if (j > b1[r]) continue;
        const int dist = rpos[r] - j;
#pragma unroll
        for (int g = 0; g < G; ++g) {
          const int pi = r * G + g;
          float dot = 0.f;
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            const unsigned wv[4] = {kv4[i].x, kv4[i].y, kv4[i].z, kv4[i].w};
#pragma unroll
            for (int qd = 0; qd < 4; ++qd) {
              dot = dot + qs[pi][8 * i + 2 * qd] * lo16(wv[qd]);
              dot = dot + qs[pi][8 * i + 2 * qd + 1] * hi16(wv[qd]);
            }
          }
          float sc = dot * 0.08838834764831845f;
          if (dist < extent) {
            float rb = 0.f;
#pragma unroll
            for (int k = 0; k < 16; ++k) rb = rb + rv[pi][k] * fbf(P[(long)k * extent + dist]);
            sc = sc + rb;
          }
          s[pi] = sc;
        }
      }
    }
    // the tile max of every pair: INK49's shuffle tree, then the sequential 8-wave fold
#pragma unroll
    for (int pi = 0; pi < NP; ++pi) {
      float v = s[pi];
#pragma unroll
      for (int of = 16; of > 0; of >>= 1) v = fmaxf(v, __shfl_xor(v, of, 32));
      if (lane == 0) redm[pi][w] = v;
    }
    __syncthreads();
    if (tid < NP) {
      float tm = redm[tid][0];
#pragma unroll
      for (int i = 1; i < FA_TILE / 32; ++i) tm = fmaxf(tm, redm[tid][i]);
      const float mo = sm[tid];
      const float mn = fmaxf(mo, tm);
      smn[tid] = mn;
      scorr[tid] = (mo == -INFINITY) ? 0.f : expf(mo - mn);
    }
    __syncthreads();
#pragma unroll
    for (int pi = 0; pi < NP; ++pi) {
      const float p = (j <= b1[pi / G]) ? expf(s[pi] - smn[pi]) : 0.f;
      ps[pi][tid] = p;
      float v = p;
#pragma unroll
      for (int of = 16; of > 0; of >>= 1) v += __shfl_xor(v, of, 32);
      if (lane == 0) reds[pi][w] = v;
    }
    __syncthreads();
    if (tid < NP && ((in >> (tid / G)) & 1)) {
      float ts = reds[tid][0];
#pragma unroll
      for (int i = 1; i < FA_TILE / 32; ++i) ts += reds[tid][i];
      sl[tid] = sl[tid] * scorr[tid] + ts;
      sm[tid] = smn[tid];
    }
    // the V walk: each half's keys in order, every value staged once in LDS for all pairs (a row
    // past its own last key adds nothing: INK49's walk stops at jn), 128 / VK passes of VK keys a half
    const int j0 = base + half * (FA_TILE / 2);
    const int jnm = min(FA_TILE / 2, jmax - j0 + 1);
    float a[NP];
#pragma unroll
    for (int pi = 0; pi < NP; ++pi) a[pi] = 0.f;
#pragma unroll
    for (int pass = 0; pass < FA_TILE / 2 / VK; ++pass) {
      ks54_vstore<VK>(vs, vr, tid);
      if (pass + 1 < FA_TILE / 2 / VK) ks54_vload<VK>(vr, v_pool, prow, base, pass + 1, jmax, C, kvh, tid);
      __syncthreads();
      const int i0 = pass * VK;
      const int n = min(VK, jnm - i0);
      for (int i = 0; i < n; i += 4) {
        float vv[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) vv[u] = fbf(vs[(half * VK + i + u) * FA_HD + d]);
#pragma unroll
        for (int pi = 0; pi < NP; ++pi) {
          const float4 p4 =
              *reinterpret_cast<const float4*>(&ps[pi][half * (FA_TILE / 2) + i0 + i]);
          const int lim = b1[pi / G] - (j0 + i0 + i);   // keys u <= lim count
          if (0 <= lim) a[pi] = a[pi] + p4.x * vv[0];
          if (1 <= lim) a[pi] = a[pi] + p4.y * vv[1];
          if (2 <= lim) a[pi] = a[pi] + p4.z * vv[2];
          if (3 <= lim) a[pi] = a[pi] + p4.w * vv[3];
        }
      }
      __syncthreads();
    }
#pragma unroll
    for (int pi = 0; pi < NP; ++pi) {
      if ((in >> (pi / G)) & 1) acc[pi] = acc[pi] * scorr[pi] + a[pi];
    }
    __syncthreads();
  }
  // the two halves' accumulators (ps reused as [pair][half][d])
#pragma unroll
  for (int pi = 0; pi < NP; ++pi) ps[pi][half * FA_HD + d] = acc[pi];
  __syncthreads();
  if (tid >= FA_HD) return;
  for (int pi = 0; pi < nr * G; ++pi) {
    const int r = pi / G, h = h0 + pi % G;
    if (!((act >> r) & 1)) continue;
    const int row = t + r;
    const float mm = sm[pi], ll = sl[pi];
    if (rS[r] == 1) {
      // INK49's S = 1 epilogue (= the parent's), op for op
      const float sk = sink[(long)row * ld_sink + h];
      const float mf = fmaxf(mm, sk);
      const float c = expf(mm - mf);
      const float lf = ll * c + expf(sk - mf);
      o[(long)row * ld_o + h * FA_HD + d] = ((ps[pi][d] + ps[pi][FA_HD + d]) * c) / lf;
    } else {
      float* pr = part + (((long)row * H + h) * smax + sg) * KS_LD;
      pr[d] = ps[pi][d] + ps[pi][FA_HD + d];
      if (d == 0) {
        pr[128] = mm;
        pr[129] = ll;
      }
    }
  }
}

#define KS54_ARGS                                                                               \
  const unsigned short *__restrict__ q, int ld_q, const unsigned short *__restrict__ k_pool,     \
      const unsigned short *__restrict__ v_pool, const int *__restrict__ table, int table_ld,    \
      const int *__restrict__ row_seq, const int *__restrict__ row_pos,                          \
      const float *__restrict__ rp, int ld_rp, const unsigned short *__restrict__ P,             \
      const float *__restrict__ sink, int ld_sink, float *__restrict__ o, int ld_o, int H,       \
      int KV, int extent, int window, float *__restrict__ part, int smax, int thresh, int segmin
#define KS54_PASS                                                                         \
  q, ld_q, k_pool, v_pool, table, table_ld, row_seq, row_pos, rp, ld_rp, P, sink, ld_sink, o, \
      ld_o, H, KV, extent, part, smax, thresh, segmin, rows

// `window` is ink_ks49_part's argument; the row-shared body is the global layers' (window 0) only:
// the host routes nothing else here (a windowed row's segments would not share the tile grid).
extern "C" __global__ __launch_bounds__(256) void ink_ks54_rv1(KS54_ARGS, int rows) {
  (void)window;
  ks54_rows<1, 64, 8>(KS54_PASS);
}
extern "C" __global__ __launch_bounds__(256) void ink_ks54_rv3(KS54_ARGS, int rows) {
  (void)window;
  ks54_rows<3, 32, 8>(KS54_PASS);
}

// R 4: a run's rows in workgroups of 4 (twice the workgroups of rv1 a run, half the pairs each)
extern "C" __global__ __launch_bounds__(256) void ink_ks54_rv1r4(KS54_ARGS, int rows) {
  (void)window;
  ks54_rows<1, 64, 4>(KS54_PASS);
}

// beside every row-shared launch, grid (T, H, smax): row t when its group of R rows is all at S = 1
// (else the row-shared kernel has it); R = the row-shared kernel's (8: rv1 / rv3, 4: rv1r4)
extern "C" __global__ __launch_bounds__(256) void ink_ks54_short(KS54_ARGS, int R) {
  const int t = blockIdx.x;
  const int sq = row_seq[t];
  int rs = t;
  while (rs > 0 && row_seq[rs - 1] == sq) --rs;
  if (!ks54_all1(row_seq, row_pos, gridDim.x, rs + (t - rs) / R * R, R, sq, smax, thresh)) return;
  ks54_row_body(q, ld_q, k_pool, v_pool, table, table_ld, row_seq, row_pos, rp, ld_rp, P, sink, ld_sink,
                o, ld_o, H, KV, extent, window, part, smax, thresh, segmin, blockIdx.y);
}

// INK54 task B (`--ink-p54 wk`): one system-scope store into a scratch word of the PEER's fuse words
// right before a pushing launch, so the idle P2P path is awake when the push's first store goes out
// (wake_probe: the first store after >= 20 us idle lands ~8-9 us late; a store 2 / 10 us earlier
// removes 70 % / all of it). Not a flag, no data: no reader ever reads the word.
extern "C" __global__ __launch_bounds__(32) void ink_wake54(unsigned* __restrict__ peer_word) {
  if (threadIdx.x == 0) {
    __hip_atomic_store(peer_word, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}
