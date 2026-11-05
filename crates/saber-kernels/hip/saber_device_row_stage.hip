// SABER-AUTHORED (CMP27AJ, 2026-10-30): `saber_device_row_stage_t{512,1024}_bf16`, the sampler
// seam's row kernel (`head.hip`'s `saber_device_row_bf16`, CMP27X's `saber_device_row_ilp_bf16`) in
// the STAGED-TERM form: one workgroup of NT threads a row, the shipped kernel's 256 threads become
// 256 OWNERS (threads 0..255) of the two log-sum-exp chains.
//   * Order-dependent (kept): owner t folds s1 += e^(v - mx) and st += e^(v/T - mx/T) over ids
//     t, t + 256, t + 512, .. in that order (the shipped thread's walk), then the shipped 256-wide
//     tree (`red[i] += red[i + s]`, s = 128 .. 1) and `mx + log(sum)`. The TERMS are a pure
//     function of (v, mx, T) and are computed here by ALL NT lanes over contiguous ids (the same
//     source expressions, the same object flags) into LDS; a NaN id (and an id past `cols`) stores
//     a signalling-NaN sentinel that no arithmetic produces and the owner SKIPS it (the shipped
//     `continue` / loop end), so no extra add ever enters a chain.
//   * Order-free: the two radix histograms (integer counts), the row max (a max of non-NaN floats;
//     only a zero max's sign can depend on order and it moves no output bit -- CMP27Y), the
//     above-threshold collection (a set, ordered later), the all-ties collection (a set), the
//     ordered tie scan (the lowest `take` tied ids, ranked in id order: a unique answer), the final
//     order (value desc, id asc over unique ids = the insertion sort's unique result, computed here
//     as a rank sort) and the penalty gather.
// Card-1 probe `tools/capture/cmp27aj/probe.py` (bytes vs the shipped symbol + a control, phase cuts via
// -DRG_PHASE=N). Bound only under `--engine-step-arms rowstage` (an ARM, default off): t1024.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define RG_K 64
#define RG_O 256
#define RG_WV 32
#define RG_HC 8
#define RG_E 8
#define RG_SENT 0x7FA0DEADu

#ifndef RG_PHASE
#define RG_PHASE 9  // probe phase cut: stop after phase N (1 hist+max, 2 terms, 3 gt, 4 ties, 9 all)
#endif

__device__ __forceinline__ unsigned rg_key(unsigned short b) {
  return (b & 0x8000u) ? (unsigned)(unsigned short)(~b) : (unsigned)(unsigned short)(b | 0x8000u);
}
__device__ __forceinline__ bool rg_is_nan(unsigned short b) {
  return (b & 0x7F80u) == 0x7F80u && (b & 0x007Fu) != 0u;
}
__device__ __forceinline__ float rg_f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
__device__ __forceinline__ unsigned short rg_bits(const __bf16 *p) { return *(const unsigned short *)p; }

/// N consecutive ids from j (j a multiple of N): one vector load when the whole group is in range
/// and the row is aligned, else per-id loads; an id past `cols` reads as a NaN (skipped everywhere).
template <int N>
__device__ __forceinline__ void rg_load(const __bf16 *__restrict__ p, int j, int cols, bool vec,
                                        unsigned short (&bb)[N]) {
  if (vec && j + N <= cols) {
    if constexpr (N == 8) {
      const uint4 w = *(const uint4 *)(p + j);
      const unsigned ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        bb[2 * e] = (unsigned short)(ww[e] & 0xFFFFu);
        bb[2 * e + 1] = (unsigned short)(ww[e] >> 16);
      }
    } else if constexpr (N == 4) {
      const uint2 w = *(const uint2 *)(p + j);
      bb[0] = (unsigned short)(w.x & 0xFFFFu);
      bb[1] = (unsigned short)(w.x >> 16);
      bb[2] = (unsigned short)(w.y & 0xFFFFu);
      bb[3] = (unsigned short)(w.y >> 16);
    } else if constexpr (N == 16) {
      const uint4 w0 = *(const uint4 *)(p + j);
      const uint4 w1 = *(const uint4 *)(p + j + 8);
      const unsigned ww[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        bb[2 * e] = (unsigned short)(ww[e] & 0xFFFFu);
        bb[2 * e + 1] = (unsigned short)(ww[e] >> 16);
      }
    } else {
#pragma unroll
      for (int e = 0; e < N; ++e) bb[e] = rg_bits(p + j + e);
    }
  } else {
#pragma unroll
    for (int e = 0; e < N; ++e) bb[e] = j + e < cols ? rg_bits(p + j + e) : (unsigned short)0x7FC1u;
  }
}

/// The shipped serial bucket scan (from 255 down, skip empty; the first bucket whose running count
/// reaches `need`; none -> the lowest non-empty bucket and the count above it; nothing -> -1), run
/// by wave 0 as a lane-blocked scan. Writes *ob / *oa. Every lane of wave 0 must call it.
__device__ __forceinline__ void rg_select(const unsigned *h, int need, int lane, int *ob, int *oa) {
  unsigned n8[8];
  int sum = 0, lowest = -1;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    n8[i] = h[255 - (lane * 8 + i)];
    sum += (int)n8[i];
    if (n8[i] != 0u) lowest = 255 - (lane * 8 + i);
  }
  int x = sum;  // inclusive scan over lanes (lane 0 = the highest buckets)
#pragma unroll
  for (int off = 1; off < RG_WV; off <<= 1) {
    const int y = __shfl_up(x, off, RG_WV);
    if (lane >= off) x += y;
  }
  const int total = __shfl(x, RG_WV - 1, RG_WV);
  int cum = x - sum;
  int fb = -1, fa = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int n = (int)n8[i];
    if (n == 0 || fb >= 0) continue;
    if (cum + n >= need) {
      fb = 255 - (lane * 8 + i);
      fa = cum;
    } else {
      cum += n;
    }
  }
  const unsigned long long fm = __ballot(fb >= 0);
  if (fm != 0ull) {
    const int src = __builtin_ctzll(fm);
    const int b = __shfl(fb, src, RG_WV), a = __shfl(fa, src, RG_WV);
    if (lane == 0) {
      *ob = b;
      *oa = a;
    }
    return;
  }
  const unsigned long long nm = __ballot(lowest >= 0);
  if (nm == 0ull) {
    if (lane == 0) {
      *ob = -1;
      *oa = 0;
    }
    return;
  }
  const int src = 63 - __builtin_clzll(nm);
  const int b = __shfl(lowest, src, RG_WV);
  if (lane == 0) {
    *ob = b;
    *oa = total - (int)h[b];
  }
}

template <int NT, int IPL>
__device__ __forceinline__ void rg_body(const __bf16 *__restrict__ logits, const float *__restrict__ temps,
                                        const int *__restrict__ pen_ids, float *__restrict__ top_values,
                                        int *__restrict__ top_ids, float *__restrict__ lse,
                                        float *__restrict__ pen_values, int cols, long stride, int pen_max) {
  constexpr int NW = NT / RG_WV;
  constexpr int TILE = NT * IPL;
  constexpr int R = TILE / RG_O;
  static_assert(TILE % RG_O == 0, "a tile holds whole owner rounds");
  __shared__ unsigned hist[RG_HC][256];
  __shared__ float e1[TILE];
  __shared__ float et[TILE];
  __shared__ float red[RG_O];
  __shared__ float wmax[NW];
  __shared__ float cand_v[RG_K];
  __shared__ int cand_i[RG_K];
  __shared__ int wave_ties[NW];
  __shared__ int s_bucket, s_above, s_low, s_above2, s_n_gt, s_n_tie;
  __shared__ float s_max;

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int wave = tid / RG_WV;
  const int lane = tid % RG_WV;
  const float temperature = temps[row];
  float *tv = top_values + (long)row * RG_K;
  int *ti = top_ids + (long)row * RG_K;
  if (!(temperature > 0.0f)) {
    if (tid == 0) ti[0] = -1;
    return;
  }
  const __bf16 *p = logits + (long)row * stride;
  const bool vec8 = ((reinterpret_cast<unsigned long>(p) & 15ul) == 0ul);
  const bool vecI = ((reinterpret_cast<unsigned long>(p) & (unsigned long)(2 * IPL > 16 ? 15 : 2 * IPL - 1)) == 0ul);

  for (int i = tid; i < pen_max; i += NT) {
    const int id = pen_ids[(long)row * pen_max + i];
    pen_values[(long)row * pen_max + i] = (id >= 0 && id < cols) ? (float)p[id] : 0.0f;
  }

  // ---- pass 1: high-byte histogram + the row max (order-free), contiguous 8-id groups ----------
  for (int i = tid; i < 256 * RG_HC; i += NT) ((unsigned *)hist)[i] = 0u;
  __syncthreads();
  const int ng = (cols + RG_E - 1) / RG_E;
  unsigned *hw = hist[wave % RG_HC];
  float mx = -__builtin_inff();
  for (int g0 = tid; g0 < ng; g0 += 2 * NT) {
    unsigned short ba[RG_E], bb2[RG_E];
    rg_load<RG_E>(p, g0 * RG_E, cols, vec8, ba);
    const bool two = g0 + NT < ng;
    if (two) rg_load<RG_E>(p, (g0 + NT) * RG_E, cols, vec8, bb2);
#pragma unroll
    for (int e = 0; e < RG_E; ++e) {
      const unsigned short b = ba[e];
      if (rg_is_nan(b)) continue;
      atomicAdd(&hw[rg_key(b) >> 8], 1u);
      const float v = rg_f(b);
      if (v > mx) mx = v;
    }
    if (two) {
#pragma unroll
      for (int e = 0; e < RG_E; ++e) {
        const unsigned short b = bb2[e];
        if (rg_is_nan(b)) continue;
        atomicAdd(&hw[rg_key(b) >> 8], 1u);
        const float v = rg_f(b);
        if (v > mx) mx = v;
      }
    }
  }
#pragma unroll
  for (int off = RG_WV / 2; off > 0; off >>= 1) {
    const float o = __shfl_xor(mx, off, RG_WV);
    if (o > mx) mx = o;
  }
  if (lane == 0) wmax[wave] = mx;
  // Every wave's histogram atomics land before any copy is folded (p1's race: without this barrier
  // a fast wave folded a slow wave's copy early, the threshold came out low and the above-threshold
  // set overflowed the 64 slots in atomic order).
  __syncthreads();
  unsigned folded = 0u;
  if (tid < 256) {
#pragma unroll
    for (int w = 0; w < RG_HC; ++w) folded += hist[w][tid];
  }
  __syncthreads();
  if (tid < 256) hist[0][tid] = folded;
  if (tid == 0) {
    float m = -__builtin_inff();
    for (int w = 0; w < NW; ++w)
      if (wmax[w] > m) m = wmax[w];
    s_max = m;
  }
  __syncthreads();
  if (wave == 0) rg_select(hist[0], RG_K, lane, &s_bucket, &s_above);
  __syncthreads();
  if (s_bucket < 0) {
    for (int i = tid; i < RG_K; i += NT) {
      tv[i] = -__builtin_inff();
      ti[i] = cols;
    }
    if (tid == 0) {
      lse[(long)row * 2] = -__builtin_inff();
      lse[(long)row * 2 + 1] = -__builtin_inff();
    }
    return;
  }
#if RG_PHASE < 2
  if (tid == 0) ti[0] = s_bucket;
  return;
#endif

  // ---- pass 2: low-byte histogram (order-free) + the staged terms, folded in the owners' order ---
  for (int i = tid; i < 256 * RG_HC; i += NT) ((unsigned *)hist)[i] = 0u;
  __syncthreads();
  const float mxv = s_max;
  const float mxt = mxv / temperature;
  const int bucket = s_bucket;
  float s1 = 0.0f, st = 0.0f;
  for (int base = 0; base < cols; base += TILE) {
    {
      const int j = base + tid * IPL;
      unsigned short bb[IPL];
      rg_load<IPL>(p, j, cols, vecI, bb);
      float t1[IPL], t2[IPL];
#pragma unroll
      for (int e = 0; e < IPL; ++e) {
        const unsigned short b = bb[e];
        if (rg_is_nan(b)) {
          t1[e] = __uint_as_float(RG_SENT);
          t2[e] = __uint_as_float(RG_SENT);
          continue;
        }
        const unsigned key = rg_key(b);
        if ((int)(key >> 8) == bucket) atomicAdd(&hw[key & 0xFFu], 1u);
        const float v = rg_f(b);
        t1[e] = __builtin_expf(v - mxv);
        t2[e] = __builtin_expf(v / temperature - mxt);
      }
#pragma unroll
      for (int e = 0; e < IPL; ++e) {
        e1[tid * IPL + e] = t1[e];
        et[tid * IPL + e] = t2[e];
      }
    }
    __syncthreads();
    if (tid < RG_O) {
#pragma unroll
      for (int k = 0; k < R; ++k) {
        const float a = e1[k * RG_O + tid];
        const float c = et[k * RG_O + tid];
        if (__float_as_uint(a) != RG_SENT) s1 += a;
        if (__float_as_uint(c) != RG_SENT) st += c;
      }
    }
    __syncthreads();
  }
  if (tid < RG_O) red[tid] = s1;
  __syncthreads();
  for (int s = RG_O / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  if (tid == 0) lse[(long)row * 2] = mxv + __builtin_logf(red[0]);
  __syncthreads();
  if (tid < RG_O) red[tid] = st;
  __syncthreads();
  for (int s = RG_O / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  if (tid == 0) lse[(long)row * 2 + 1] = mxt + __builtin_logf(red[0]);
  folded = 0u;
  if (tid < 256) {
#pragma unroll
    for (int w = 0; w < RG_HC; ++w) folded += hist[w][tid];
  }
  __syncthreads();
  if (tid < 256) hist[0][tid] = folded;
  __syncthreads();
  if (wave == 0) rg_select(hist[0], RG_K - s_above, lane, &s_low, &s_above2);
  if (tid == 0) {
    s_n_gt = 0;
    s_n_tie = 0;
  }
  __syncthreads();
#if RG_PHASE < 3
  if (tid == 0) ti[0] = s_low;
  return;
#endif
  const unsigned thr = ((unsigned)bucket << 8) | (unsigned)s_low;
  const int cnt_gt = s_above + s_above2;
  const int ties = (int)hist[0][s_low];
  const int want = RG_K - cnt_gt;
  const int take = want < ties ? want : ties;

  // ---- pass 3: the candidates above the threshold (a set) --------------------------------------
  for (int g0 = tid; g0 < ng; g0 += NT) {
    unsigned short bb[RG_E];
    rg_load<RG_E>(p, g0 * RG_E, cols, vec8, bb);
#pragma unroll
    for (int e = 0; e < RG_E; ++e) {
      const unsigned short b = bb[e];
      if (rg_is_nan(b)) continue;
      if (rg_key(b) > thr) {
        const int slot = atomicAdd(&s_n_gt, 1);
        if (slot < RG_K) {
          cand_v[slot] = rg_f(b);
          cand_i[slot] = g0 * RG_E + e;
        }
      }
    }
  }
  __syncthreads();
#if RG_PHASE < 4
  if (tid == 0) ti[0] = s_n_gt;
  return;
#endif
  if (take == ties) {
    for (int g0 = tid; g0 < ng; g0 += NT) {
      unsigned short bb[RG_E];
      rg_load<RG_E>(p, g0 * RG_E, cols, vec8, bb);
#pragma unroll
      for (int e = 0; e < RG_E; ++e) {
        const unsigned short b = bb[e];
        if (rg_is_nan(b)) continue;
        if (rg_key(b) == thr) {
          const int slot = cnt_gt + atomicAdd(&s_n_tie, 1);
          if (slot < RG_K) {
            cand_v[slot] = rg_f(b);
            cand_i[slot] = g0 * RG_E + e;
          }
        }
      }
    }
  } else {
    // The lowest `take` tied ids in id order at slots cnt_gt + rank (CMP27X's wave-contiguous scan
    // over NW waves).
    const int gpw = (ng + NW - 1) / NW;
    const int g_lo = wave * gpw;
    const int g_hi = g_lo + gpw < ng ? g_lo + gpw : ng;
    auto tie_mask = [&](int g) -> unsigned {
      unsigned m = 0u;
      if (g < g_hi) {
        unsigned short bb[RG_E];
        rg_load<RG_E>(p, g * RG_E, cols, vec8, bb);
#pragma unroll
        for (int e = 0; e < RG_E; ++e)
          if (!rg_is_nan(bb[e]) && rg_key(bb[e]) == thr) m |= 1u << e;
      }
      return m;
    };
    int mine = 0;
    for (int g0 = g_lo + lane; g0 < g_hi; g0 += RG_WV) mine += __builtin_popcount(tie_mask(g0));
#pragma unroll
    for (int off = RG_WV / 2; off > 0; off >>= 1) mine += __shfl_xor(mine, off, RG_WV);
    if (lane == 0) wave_ties[wave] = mine;
    __syncthreads();
    int basec = 0;
    for (int w = 0; w < NW; ++w)
      if (w < wave) basec += wave_ties[w];
    if (basec < take) {
      int done = basec;
      unsigned mnext = tie_mask(g_lo + lane);
      for (int g = g_lo + lane; g - lane < g_hi && done < take; g += RG_WV) {
        unsigned m = mnext;
        mnext = tie_mask(g + RG_WV);
        const int c = __builtin_popcount(m);
        int x = c;
#pragma unroll
        for (int off = 1; off < RG_WV; off <<= 1) {
          const int y = __shfl_up(x, off, RG_WV);
          if (lane >= off) x += y;
        }
        int rank = done + x - c;
        while (m != 0u && rank < take) {
          const int e = __builtin_ctz(m);
          m &= m - 1u;
          const int slot = cnt_gt + rank;
          if (slot < RG_K) {
            cand_v[slot] = rg_f(rg_bits(p + g * RG_E + e));
            cand_i[slot] = g * RG_E + e;
          }
          ++rank;
        }
        done += __shfl(x, RG_WV - 1, RG_WV);
      }
    }
    if (tid == 0) s_n_tie = take;
  }
  __syncthreads();

  // ---- the order: value descending, id ascending (a rank sort = the insertion sort's answer) ----
  const int got = (cnt_gt < RG_K ? cnt_gt : RG_K) + s_n_tie;
  const int have = got < RG_K ? got : RG_K;
  float ov = -__builtin_inff();
  int oi = cols, orank = tid;
  if (tid < have) {
    ov = cand_v[tid];
    oi = cand_i[tid];
    int r = 0;
    for (int k = 0; k < have; ++k) {
      const float w = cand_v[k];
      const int wi = cand_i[k];
      r += (w > ov || (w == ov && wi < oi)) ? 1 : 0;
    }
    orank = r;
  }
  __syncthreads();
  if (tid < RG_K) {
    cand_v[orank] = ov;
    cand_i[orank] = oi;
  }
  __syncthreads();
  for (int i = tid; i < RG_K; i += NT) {
    tv[i] = cand_v[i];
    ti[i] = cand_i[i];
  }
}

#define RG_KERNEL(NT, IPL)                                                                          \
  extern "C" __global__ __launch_bounds__(NT) void saber_device_row_stage_t##NT##_bf16(            \
      const __bf16 *__restrict__ logits, const float *__restrict__ temps,                          \
      const int *__restrict__ pen_ids, float *__restrict__ top_values, int *__restrict__ top_ids,   \
      float *__restrict__ lse, float *__restrict__ pen_values, int cols, long stride, int pen_max) { \
    rg_body<NT, IPL>(logits, temps, pen_ids, top_values, top_ids, lse, pen_values, cols, stride,    \
                     pen_max);                                                                      \
  }


RG_KERNEL(512, 8)
RG_KERNEL(1024, 4)
