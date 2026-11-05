// SABER-AUTHORED (CMP27Y, 2026-10-29): `saber_device_row_split_g{2,4,8}_bf16`, the sampler seam's
// row kernel (`head.hip`'s `saber_device_row_bf16`, CMP27X's `saber_device_row_ilp_bf16`) with each
// row split over G workgroups of 256 / G threads. The shipped kernel's 256 threads become 256
// VIRTUAL threads: workgroup g of a row runs virtual threads [g * 256 / G, (g + 1) * 256 / G), and
// each walks exactly the ids the shipped thread of that index walks (vt, vt + 256, ..), in the same
// order, with the same arithmetic. So:
//   * both log-sum-exp partials of every virtual thread are the shipped thread's bit for bit; they
//     go to the row's `sync` slab and the LAST workgroup of the row to finish runs the shipped
//     256-wide tree in the shipped pairing order (`red[i] += red[i + s]`, s = 128 .. 1);
//   * the row maximum is order-free (a max of non-NaN floats; only the sign of a zero maximum can
//     depend on the order, and `v - mx`, `v / t - mx / t` and `mx + log(sum)` give the same bits
//     for either sign of a zero `mx`), published per workgroup and combined after ONE per-row
//     barrier (a monotonic arrival counter: no reset, graph-replay safe);
//   * the top-K is the top K of the total order (key descending, id ascending) -- the shipped
//     selection: every key above the threshold plus the lowest-id ties -- so each workgroup selects
//     ITS ids' top K (the shipped radix select + ordered tie scan over its own ids) and the last
//     workgroup ranks the union (a superset of the row's top K) by that order, then orders the K
//     winners by (value descending, id ascending), the shipped insertion sort's unique result;
//   * the penalty gather is a gather.
// Deadlock freedom of the barrier: workgroups dispatch in index order and a row's G are adjacent, so
// at most G - 1 resident workgroups wait on undispatched ones; any machine that holds G of them
// progresses. A slab serves ONE G (the counters count in G-sized rounds from 0). Bound only under `--engine-step-arms rowsplit` (an ARM, default off); card-1 probe
// `tools/capture/cmp27y/probe.py rowsplit` (bytes vs the shipped symbol, + a control).

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define RS_T 256  // virtual threads a row = the shipped kernel's block
#define RS_K 64
#define RS_W 32
#define RS_U 8
// The row's slab in `sync` (u32 words): arrivals, completions, the G maxima, the 2 x 256 lse
// partials, the G x K candidates (values, ids) and the G candidate counts.
#define RS_SLAB 2048
#define RS_ARRIVE 0
#define RS_DONE 1
#define RS_MAX 8
#define RS_S1 64
#define RS_ST 320
#define RS_CV 576
#define RS_CI 1088
#define RS_CN 1600

__device__ __forceinline__ unsigned rs_key(unsigned short b) {
  return (b & 0x8000u) ? (unsigned)(unsigned short)(~b) : (unsigned)(unsigned short)(b | 0x8000u);
}

__device__ __forceinline__ bool rs_is_nan(unsigned short b) {
  return (b & 0x7F80u) == 0x7F80u && (b & 0x007Fu) != 0u;
}

__device__ __forceinline__ unsigned short rs_bits(const __bf16 *p) {
  return *(const unsigned short *)p;
}

__device__ __forceinline__ float rs_f(unsigned short b) {
  return __uint_as_float((unsigned)b << 16);
}

__device__ __forceinline__ unsigned rs_ld(const unsigned *p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ void rs_load(const __bf16 *__restrict__ p, int j0, int cols,
                                        unsigned short (&bb)[RS_U]) {
#pragma unroll
  for (int u = 0; u < RS_U; ++u) {
    const int j = j0 + u * RS_T;
    bb[u] = j < cols ? rs_bits(p + j) : (unsigned short)0;
  }
}

/// The top-K bucket scan over a 256-bin histogram (one thread): the shipped kernel's loop.
__device__ __forceinline__ void rs_scan(const unsigned *h, long need, int &bin, long &above) {
  long cum = 0;
  bin = -1;
  above = 0;
  for (int b = 255; b >= 0; --b) {
    const unsigned n = h[b];
    if (n == 0u) continue;
    if (cum + (long)n >= need) {
      bin = b;
      above = cum;
      return;
    }
    cum += (long)n;
    bin = b;
    above = cum - (long)n;
  }
}

template <int G>
__device__ __forceinline__ void rs_body(const __bf16 *__restrict__ logits,
                                        const float *__restrict__ temps,
                                        const int *__restrict__ pen_ids,
                                        float *__restrict__ top_values, int *__restrict__ top_ids,
                                        float *__restrict__ lse, float *__restrict__ pen_values,
                                        int cols, long stride, int pen_max,
                                        unsigned *__restrict__ sync) {
  constexpr int NT = RS_T / G;      // real threads a workgroup
  constexpr int NW = NT / RS_W;     // waves a workgroup
  constexpr int E = NT / RS_W;      // consecutive ids of a block a lane owns in the tie scan
  constexpr int NC = G * RS_K;      // candidates the last workgroup ranks
  __shared__ unsigned hist[NW][256];
  __shared__ float red[RS_T];
  __shared__ float cand_v[RS_K];
  __shared__ int cand_i[RS_K];
  __shared__ float m_v[NC];
  __shared__ int m_i[NC];
  __shared__ unsigned char m_sel[NC];
  __shared__ int wave_ties[NW];
  __shared__ int s_bucket, s_above, s_thr, s_cnt_gt, s_take, s_ties, s_n_gt, s_n_tie, s_last;
  __shared__ float s_max;

  const int row = blockIdx.x / G;
  const int g = blockIdx.x % G;
  const int tid = threadIdx.x;
  const int vt = g * NT + tid;
  const int wave = tid / RS_W;
  const int lane = tid % RS_W;
  const float temperature = temps[row];
  float *tv = top_values + (long)row * RS_K;
  int *ti = top_ids + (long)row * RS_K;
  unsigned *slab = sync + (long)row * RS_SLAB;

  if (!(temperature > 0.0f)) {
    if (g == 0 && tid == 0) ti[0] = -1;
    return;
  }
  const __bf16 *p = logits + (long)row * stride;

  for (int i = vt; i < pen_max; i += RS_T) {
    const int id = pen_ids[(long)row * pen_max + i];
    pen_values[(long)row * pen_max + i] = (id >= 0 && id < cols) ? (float)p[id] : 0.0f;
  }

  // ---- pass 1 (this workgroup's ids): the key's high byte and the maximum --------------------
  for (int i = tid; i < 256 * NW; i += NT) ((unsigned *)hist)[i] = 0u;
  __syncthreads();
  float mx = -__builtin_inff();
  for (int j0 = vt; j0 < cols; j0 += RS_T * RS_U) {
    unsigned short bb[RS_U];
    rs_load(p, j0, cols, bb);
#pragma unroll
    for (int u = 0; u < RS_U; ++u) {
      if (j0 + u * RS_T >= cols) break;
      const unsigned short b = bb[u];
      if (rs_is_nan(b)) continue;
      atomicAdd(&hist[wave][rs_key(b) >> 8], 1u);
      const float v = rs_f(b);
      if (v > mx) mx = v;
    }
  }
  red[tid] = mx;
  __syncthreads();
  for (int s = NT / 2; s > 0; s >>= 1) {
    if (tid < s && red[tid + s] > red[tid]) red[tid] = red[tid + s];
    __syncthreads();
  }
  unsigned folded = 0u;
  for (int b = tid; b < 256; b += NT) {
    folded = 0u;
    for (int w = 0; w < NW; ++w) folded += hist[w][b];
    hist[0][b] = folded;  // wave 0's copy is read only by this thread for this bin
  }
  __syncthreads();
  if (tid == 0) {
    // Publish this workgroup's maximum and wait for the row's other G - 1.
    __hip_atomic_store(slab + RS_MAX + g, __float_as_uint(red[0]), __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
    __threadfence();
    const unsigned old = atomicAdd(slab + RS_ARRIVE, 1u);
    const unsigned target = (old / G + 1u) * G;
    while ((int)(rs_ld(slab + RS_ARRIVE) - target) < 0) __builtin_amdgcn_s_sleep(1);
    __threadfence();
    float m = -__builtin_inff();
    for (int k = 0; k < G; ++k) {
      const float v = __uint_as_float(rs_ld(slab + RS_MAX + k));
      if (v > m) m = v;
    }
    s_max = m;
    int bucket;
    long above;
    rs_scan(hist[0], (long)RS_K, bucket, above);
    s_bucket = bucket;
    s_above = (int)above;
  }
  __syncthreads();

  // ---- pass 2: the low byte in this workgroup's bucket, and both lse partials ----------------
  for (int i = tid; i < 256 * NW; i += NT) ((unsigned *)hist)[i] = 0u;
  __syncthreads();
  const float mxv = s_max;
  const float mxt = mxv / temperature;
  const int bucket = s_bucket;
  float s1 = 0.0f, st = 0.0f;
  for (int j0 = vt; j0 < cols; j0 += RS_T * RS_U) {
    unsigned short bb[RS_U];
    rs_load(p, j0, cols, bb);
#pragma unroll
    for (int u = 0; u < RS_U; ++u) {
      if (j0 + u * RS_T >= cols) break;
      const unsigned short b = bb[u];
      if (rs_is_nan(b)) continue;
      const unsigned key = rs_key(b);
      if ((int)(key >> 8) == bucket) atomicAdd(&hist[wave][key & 0xFFu], 1u);
      const float v = rs_f(b);
      s1 += __builtin_expf(v - mxv);
      st += __builtin_expf(v / temperature - mxt);
    }
  }
  __hip_atomic_store(slab + RS_S1 + vt, __float_as_uint(s1), __ATOMIC_RELAXED,
                     __HIP_MEMORY_SCOPE_AGENT);
  __hip_atomic_store(slab + RS_ST + vt, __float_as_uint(st), __ATOMIC_RELAXED,
                     __HIP_MEMORY_SCOPE_AGENT);
  __syncthreads();
  for (int b = tid; b < 256; b += NT) {
    folded = 0u;
    for (int w = 0; w < NW; ++w) folded += hist[w][b];
    hist[0][b] = folded;
  }
  __syncthreads();
  if (tid == 0) {
    if (bucket >= 0) {
      int low;
      long above;
      rs_scan(hist[0], (long)RS_K - (long)s_above, low, above);
      if (low < 0) low = 0;
      s_thr = (bucket << 8) | low;
      s_cnt_gt = s_above + (int)above;
      s_ties = (int)hist[0][low];
      const int want = RS_K - s_cnt_gt;
      s_take = want < s_ties ? want : s_ties;
    } else {
      s_thr = 0x10000;  // no key reaches it: this workgroup has no finite id
      s_cnt_gt = 0;
      s_ties = 0;
      s_take = 0;
    }
    s_n_gt = 0;
    s_n_tie = 0;
  }
  __syncthreads();

  // ---- pass 3: this workgroup's top K ----------------------------------------------------------
  const unsigned thr = (unsigned)s_thr;
  const int cnt_gt = s_cnt_gt;
  const int take = s_take;
  if (bucket >= 0) {
    for (int j0 = vt; j0 < cols; j0 += RS_T * RS_U) {
      unsigned short bb[RS_U];
      rs_load(p, j0, cols, bb);
#pragma unroll
      for (int u = 0; u < RS_U; ++u) {
        const int j = j0 + u * RS_T;
        if (j >= cols) break;
        const unsigned short b = bb[u];
        if (rs_is_nan(b)) continue;
        if (rs_key(b) > thr) {
          const int slot = atomicAdd(&s_n_gt, 1);
          if (slot < RS_K) {
            cand_v[slot] = rs_f(b);
            cand_i[slot] = j;
          }
        }
      }
    }
  }
  __syncthreads();
  if (bucket >= 0 && take == s_ties) {
    for (int j0 = vt; j0 < cols; j0 += RS_T * RS_U) {
      unsigned short bb[RS_U];
      rs_load(p, j0, cols, bb);
#pragma unroll
      for (int u = 0; u < RS_U; ++u) {
        const int j = j0 + u * RS_T;
        if (j >= cols) break;
        const unsigned short b = bb[u];
        if (rs_is_nan(b)) continue;
        if (rs_key(b) == thr) {
          const int slot = cnt_gt + atomicAdd(&s_n_tie, 1);
          if (slot < RS_K) {
            cand_v[slot] = rs_f(b);
            cand_i[slot] = j;
          }
        }
      }
    }
  } else if (bucket >= 0) {
    // More ties than room: the lowest-id `take` of THIS workgroup's ties. Its ids in ascending
    // order are (block b = j / 256, then j % 256); a group = E consecutive ids of a block (lane l
    // of group gi owns ids b * 256 + g * NT + l * E + e). Waves take contiguous group ranges,
    // count, one barrier + prefix, then walk in id order with a wave scan (CMP27X's tie scan).
    const int nb = (cols + RS_T - 1) / RS_T;
    const int ngrp = nb * RS_W;
    const int gpw = (ngrp + NW - 1) / NW;
    const int g_lo = wave * gpw;
    const int g_hi = g_lo + gpw < ngrp ? g_lo + gpw : ngrp;
    auto gid = [&](int gi, int e) -> int { return (gi / RS_W) * RS_T + g * NT + (gi % RS_W) * E + e; };
    auto tie_mask = [&](int gi) -> unsigned {
      unsigned m = 0u;
      if (gi < g_hi) {
        unsigned short bb[E];
#pragma unroll
        for (int e = 0; e < E; ++e) {
          const int j = gid(gi, e);
          bb[e] = j < cols ? rs_bits(p + j) : (unsigned short)0x7FC1u;
        }
#pragma unroll
        for (int e = 0; e < E; ++e)
          if (!rs_is_nan(bb[e]) && rs_key(bb[e]) == thr) m |= 1u << e;
      }
      return m;
    };
    int mine = 0;
    for (int gi = g_lo + lane; gi < g_hi; gi += RS_W) mine += __builtin_popcount(tie_mask(gi));
#pragma unroll
    for (int off = RS_W / 2; off > 0; off >>= 1) mine += __shfl_xor(mine, off, RS_W);
    if (lane == 0) wave_ties[wave] = mine;
    __syncthreads();
    int base = 0;
#pragma unroll
    for (int w = 0; w < NW; ++w)
      if (w < wave) base += wave_ties[w];
    if (base < take) {
      int done = base;
      unsigned mnext = tie_mask(g_lo + lane);
      for (int gi = g_lo + lane; gi - lane < g_hi && done < take; gi += RS_W) {
        unsigned m = mnext;
        mnext = tie_mask(gi + RS_W);
        const int c = __builtin_popcount(m);
        int x = c;
#pragma unroll
        for (int off = 1; off < RS_W; off <<= 1) {
          const int y = __shfl_up(x, off, RS_W);
          if (lane >= off) x += y;
        }
        int rank = done + x - c;
        while (m != 0u && rank < take) {
          const int e = __builtin_ctz(m);
          m &= m - 1u;
          const int slot = cnt_gt + rank;
          if (slot < RS_K) {
            const int j = gid(gi, e);
            cand_v[slot] = rs_f(rs_bits(p + j));
            cand_i[slot] = j;
          }
          ++rank;
        }
        done += __shfl(x, RS_W - 1, RS_W);
      }
    }
    if (tid == 0) s_n_tie = take;
  }
  __syncthreads();
  const int got0 = (cnt_gt < RS_K ? cnt_gt : RS_K) + s_n_tie;
  const int got = bucket >= 0 ? (got0 < RS_K ? got0 : RS_K) : 0;
  for (int i = tid; i < got; i += NT) {
    __hip_atomic_store(slab + RS_CV + g * RS_K + i, __float_as_uint(cand_v[i]), __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_store(slab + RS_CI + g * RS_K + i, (unsigned)cand_i[i], __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
  }
  if (tid == 0)
    __hip_atomic_store(slab + RS_CN + g, (unsigned)got, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    const unsigned old = atomicAdd(slab + RS_DONE, 1u);
    s_last = (old % G) == (unsigned)(G - 1);
    __threadfence();
  }
  __syncthreads();
  if (!s_last) return;

  // ---- the last workgroup of the row: the shipped lse tree, then the top K of the union -------
  int n = 0;
  int cnt[G];
#pragma unroll
  for (int k = 0; k < G; ++k) {
    cnt[k] = (int)rs_ld(slab + RS_CN + k);
    n += cnt[k];
  }
  if (n == 0) {
    for (int i = tid; i < RS_K; i += NT) {
      tv[i] = -__builtin_inff();
      ti[i] = cols;
    }
    if (tid == 0) {
      lse[(long)row * 2] = -__builtin_inff();
      lse[(long)row * 2 + 1] = -__builtin_inff();
    }
    return;
  }
  for (int pass = 0; pass < 2; ++pass) {
    for (int i = tid; i < RS_T; i += NT)
      red[i] = __uint_as_float(rs_ld(slab + (pass == 0 ? RS_S1 : RS_ST) + i));
    __syncthreads();
    for (int s = RS_T / 2; s > 0; s >>= 1) {
      for (int i = tid; i < s; i += NT) red[i] += red[i + s];
      __syncthreads();
    }
    if (tid == 0) lse[(long)row * 2 + pass] = (pass == 0 ? mxv : mxt) + __builtin_logf(red[0]);
    __syncthreads();
  }
  {
    int off = 0;
#pragma unroll
    for (int k = 0; k < G; ++k) {
      for (int i = tid; i < cnt[k]; i += NT) {
        m_v[off + i] = __uint_as_float(rs_ld(slab + RS_CV + k * RS_K + i));
        m_i[off + i] = (int)rs_ld(slab + RS_CI + k * RS_K + i);
      }
      off += cnt[k];
    }
  }
  __syncthreads();
  // Selection: rank by (key descending, id ascending), keep rank < K.
  for (int c = tid; c < n; c += NT) {
    const unsigned kc = rs_key((unsigned short)(__float_as_uint(m_v[c]) >> 16));
    const int ic = m_i[c];
    int r = 0;
    for (int d = 0; d < n; ++d) {
      const unsigned kd = rs_key((unsigned short)(__float_as_uint(m_v[d]) >> 16));
      r += (kd > kc) || (kd == kc && m_i[d] < ic);
    }
    m_sel[c] = r < RS_K;
  }
  __syncthreads();
  // Order: (value descending, id ascending) among the kept.
  for (int c = tid; c < n; c += NT) {
    if (!m_sel[c]) continue;
    const float vc = m_v[c];
    const int ic = m_i[c];
    int r = 0;
    for (int d = 0; d < n; ++d)
      r += m_sel[d] && ((m_v[d] > vc) || (m_v[d] == vc && m_i[d] < ic));
    tv[r] = vc;
    ti[r] = ic;
  }
  const int have = n < RS_K ? n : RS_K;
  for (int i = have + tid; i < RS_K; i += NT) {
    tv[i] = -__builtin_inff();
    ti[i] = cols;
  }
}

#define RS_KERNEL(G)                                                                              \
  extern "C" __global__ void __launch_bounds__(RS_T / G) saber_device_row_split_g##G##_bf16(     \
      const __bf16 *__restrict__ logits, const float *__restrict__ temps,                         \
      const int *__restrict__ pen_ids, float *__restrict__ top_values, int *__restrict__ top_ids, \
      float *__restrict__ lse, float *__restrict__ pen_values, int cols, long stride,             \
      int pen_max, unsigned *__restrict__ sync) {                                                 \
    rs_body<G>(logits, temps, pen_ids, top_values, top_ids, lse, pen_values, cols, stride,        \
               pen_max, sync);                                                                    \
  }

RS_KERNEL(2)
RS_KERNEL(4)
RS_KERNEL(8)
