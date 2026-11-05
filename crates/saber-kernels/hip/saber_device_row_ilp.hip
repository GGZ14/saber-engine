// SABER-AUTHORED (CMP27X, 2026-10-29): `saber_device_row_ilp_bf16`, the sampler seam's row kernel
// (`head.hip`'s `saber_device_row_bf16`) with (1) its full-row passes issuing DEVROW_U loads of a
// thread's strided walk before using any and (2) its ordered tie scan over wave-contiguous ranges
// (one block barrier instead of two a 2,048-id tile). Every thread visits the same ids in the same
// order and does the same arithmetic (the float of a bf16 is its bits << 16, as `(float)p[j]`), so
// the log-sum-exps, the histograms, the candidates and the sorted top-K are the shipped bytes
// (card-1 probe `tools/capture/cmp27x/probe.py devrow`). Its own code object: `head` keeps its
// bytes. Bound only under `--engine-step-arms devrow` (an ARM, default off).

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define DEVROW_T 256
#define DEVROW_K 64
#define DEVROW_WAVE 32
#define DEVROW_WAVES (DEVROW_T / DEVROW_WAVE)
#define DEVROW_E 8  // consecutive ids a thread owns in the ordered tie scan (P27-W2-1)

/// bf16 bits -> a 16-bit key with the same total order as the float value.
__device__ __forceinline__ unsigned devrow_key(unsigned short b) {
  return (b & 0x8000u) ? (unsigned)(unsigned short)(~b) : (unsigned)(unsigned short)(b | 0x8000u);
}

__device__ __forceinline__ bool devrow_is_nan(unsigned short b) {
  return (b & 0x7F80u) == 0x7F80u && (b & 0x007Fu) != 0u;
}

__device__ __forceinline__ unsigned short devrow_bits(const __bf16 *p) {
  return *(const unsigned short *)p;
}

#define DEVROW_U 8  // CMP27X: loads in flight a thread (the strided walk's order is unchanged)
#define DEVROW_TU 2  // CMP27X: 8-id groups in flight a lane in the tie count

/// CMP27X: bf16 bits -> the float `(float)p[j]` gives (exact: bf16 is the float's top half).
__device__ __forceinline__ float devrow_f(unsigned short b) {
  return __uint_as_float((unsigned)b << 16);
}

/// CMP27X: the next `DEVROW_U` ids of this thread's strided walk (`j0`, `j0 + T`, ..), loaded
/// before any is used, so a pass waits on one round trip a `DEVROW_U` ids instead of one a id.
__device__ __forceinline__ void devrow_ilp_load(const __bf16 *__restrict__ p, int j0, int cols,
                                                unsigned short (&bb)[DEVROW_U]) {
#pragma unroll
  for (int u = 0; u < DEVROW_U; ++u) {
    const int j = j0 + u * DEVROW_T;
    bb[u] = j < cols ? devrow_bits(p + j) : (unsigned short)0;
  }
}

extern "C" __global__ void saber_device_row_ilp_bf16(const __bf16 *__restrict__ logits,
                                                 const float *__restrict__ temps,
                                                 const int *__restrict__ pen_ids,
                                                 float *__restrict__ top_values,
                                                 int *__restrict__ top_ids,
                                                 float *__restrict__ lse,
                                                 float *__restrict__ pen_values, int cols,
                                                 long stride, int pen_max) {
  // Per-WAVE histograms, not one per block: at 248,320 logits from one distribution most of the
  // mass lands in a handful of exponent buckets, and one shared counter per bucket serialises
  // every lane that hits it. Eight copies cost 8 KiB of LDS and cut that by the wave count.
  __shared__ unsigned hist[DEVROW_WAVES][256];
  __shared__ float red[DEVROW_T];
  __shared__ float cand_v[DEVROW_K];
  __shared__ int cand_i[DEVROW_K];
  __shared__ int wave_ties[DEVROW_WAVES];
  __shared__ int s_bucket, s_above, s_thr, s_cnt_gt, s_take, s_ties, s_n_gt, s_n_tie;
  __shared__ float s_max;

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int wave = tid / DEVROW_WAVE;
  const float temperature = temps[row];
  float *tv = top_values + (long)row * DEVROW_K;
  int *ti = top_ids + (long)row * DEVROW_K;

  if (!(temperature > 0.0f)) {
    // A greedy row: the argmax already answered it. `-1` is not a token and not `cols`, so a host
    // that read this slot anyway fails loudly rather than sampling a plausible id.
    if (tid == 0) ti[0] = -1;
    return;
  }
  const __bf16 *p = logits + (long)row * stride;

  // ---- the penalty gather --------------------------------------------------------------------
  // FIRST, because it depends on nothing else: the host knows which ids the penalties can move --
  // the prompt's and the output's tokens -- and this is their raw logits, whatever those are. `-1`
  // pads the list; its slot is written 0 and the host reads only as many entries as it asked for.
  //
  // It used to sit at the END, after the selection, and the all-NaN row of the gate's edge case
  // caught that: the "nothing won" path returned before it and wrote zeros where the row's own
  // (NaN) logits belonged, six bytes of 4,352. A gather is a gather; it does not have an opinion
  // about whether the row has a winner.
  for (int i = tid; i < pen_max; i += DEVROW_T) {
    const int id = pen_ids[(long)row * pen_max + i];
    pen_values[(long)row * pen_max + i] = (id >= 0 && id < cols) ? (float)p[id] : 0.0f;
  }

  // ---- pass 1: the key's high byte, and the row's maximum ------------------------------------
  for (int i = tid; i < 256 * DEVROW_WAVES; i += DEVROW_T) ((unsigned *)hist)[i] = 0u;
  __syncthreads();
  float mx = -__builtin_inff();
  for (int j0 = tid; j0 < cols; j0 += DEVROW_T * DEVROW_U) {
    unsigned short bb[DEVROW_U];
    devrow_ilp_load(p, j0, cols, bb);
#pragma unroll
    for (int u = 0; u < DEVROW_U; ++u) {
      if (j0 + u * DEVROW_T >= cols) break;
      const unsigned short b = bb[u];
      if (devrow_is_nan(b)) continue;
      atomicAdd(&hist[wave][devrow_key(b) >> 8], 1u);
      const float v = devrow_f(b);
      if (v > mx) mx = v;
    }
  }
  red[tid] = mx;
  __syncthreads();
  for (int s = DEVROW_T / 2; s > 0; s >>= 1) {
    if (tid < s && red[tid + s] > red[tid]) red[tid] = red[tid + s];
    __syncthreads();
  }
  if (tid == 0) s_max = red[0];
  // Fold the per-wave histograms into wave 0's copy: every copy is read before any is written.
  unsigned folded = 0u;
  if (tid < 256) {
    for (int w = 0; w < DEVROW_WAVES; ++w) folded += hist[w][tid];
  }
  __syncthreads();
  if (tid < 256) hist[0][tid] = folded;
  __syncthreads();
  if (tid == 0) {
    // The bucket the K-th largest key lies in, and how many keys sit above that bucket. When the
    // row holds fewer than K finite values the scan runs out and keeps the LOWEST non-empty
    // bucket, which is the "take everything" answer.
    long cum = 0;
    int bucket = -1;
    long above = 0;
    for (int b = 255; b >= 0; --b) {
      const unsigned n = hist[0][b];
      if (n == 0u) continue;
      if (cum + (long)n >= (long)DEVROW_K) {
        bucket = b;
        above = cum;
        break;
      }
      cum += (long)n;
      bucket = b;
      above = cum - (long)n;
    }
    s_bucket = bucket;
    s_above = (int)above;
  }
  __syncthreads();

  if (s_bucket < 0) {
    // Every element was NaN. `cols` is what `saber_argmax_bf16` writes when nothing won.
    for (int i = tid; i < DEVROW_K; i += DEVROW_T) {
      tv[i] = -__builtin_inff();
      ti[i] = cols;
    }
    if (tid == 0) {
      lse[(long)row * 2] = -__builtin_inff();
      lse[(long)row * 2 + 1] = -__builtin_inff();
    }
    return;
  }

  // ---- pass 2: the key's low byte inside that bucket, and both log-sum-exps -------------------
  for (int i = tid; i < 256 * DEVROW_WAVES; i += DEVROW_T) ((unsigned *)hist)[i] = 0u;
  __syncthreads();
  const float mxv = s_max;
  const float mxt = mxv / temperature;
  const int bucket = s_bucket;
  float s1 = 0.0f, st = 0.0f;
  for (int j0 = tid; j0 < cols; j0 += DEVROW_T * DEVROW_U) {
    unsigned short bb[DEVROW_U];
    devrow_ilp_load(p, j0, cols, bb);
#pragma unroll
    for (int u = 0; u < DEVROW_U; ++u) {
      if (j0 + u * DEVROW_T >= cols) break;
      const unsigned short b = bb[u];
      if (devrow_is_nan(b)) continue;
      const unsigned key = devrow_key(b);
      if ((int)(key >> 8) == bucket) atomicAdd(&hist[wave][key & 0xFFu], 1u);
      const float v = devrow_f(b);
      s1 += __builtin_expf(v - mxv);
      st += __builtin_expf(v / temperature - mxt);
    }
  }
  // Two tree reductions over the same array, in the order the host reference folds them.
  red[tid] = s1;
  __syncthreads();
  for (int s = DEVROW_T / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  if (tid == 0) lse[(long)row * 2] = mxv + __builtin_logf(red[0]);
  __syncthreads();
  red[tid] = st;
  __syncthreads();
  for (int s = DEVROW_T / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  if (tid == 0) lse[(long)row * 2 + 1] = mxt + __builtin_logf(red[0]);
  folded = 0u;
  if (tid < 256) {
    for (int w = 0; w < DEVROW_WAVES; ++w) folded += hist[w][tid];
  }
  __syncthreads();
  if (tid < 256) hist[0][tid] = folded;
  __syncthreads();
  if (tid == 0) {
    long cum = 0;
    int low = 0;
    long above = 0;
    const long need = (long)DEVROW_K - (long)s_above;
    for (int b = 255; b >= 0; --b) {
      const unsigned n = hist[0][b];
      if (n == 0u) continue;
      if (cum + (long)n >= need) {
        low = b;
        above = cum;
        break;
      }
      cum += (long)n;
      low = b;
      above = cum - (long)n;
    }
    s_thr = (bucket << 8) | low;
    s_cnt_gt = s_above + (int)above;
    s_ties = (int)hist[0][low];
    const int want = DEVROW_K - s_cnt_gt;
    s_take = want < s_ties ? want : s_ties;
    s_n_gt = 0;
    s_n_tie = 0;
  }
  __syncthreads();

  // ---- pass 3: collect the candidates --------------------------------------------------------
  const unsigned thr = (unsigned)s_thr;
  const int cnt_gt = s_cnt_gt;
  const int take = s_take;
  for (int j0 = tid; j0 < cols; j0 += DEVROW_T * DEVROW_U) {
    unsigned short bb[DEVROW_U];
    devrow_ilp_load(p, j0, cols, bb);
#pragma unroll
    for (int u = 0; u < DEVROW_U; ++u) {
      const int j = j0 + u * DEVROW_T;
      if (j >= cols) break;
      const unsigned short b = bb[u];
      if (devrow_is_nan(b)) continue;
      const unsigned key = devrow_key(b);
      if (key > thr) {
        const int slot = atomicAdd(&s_n_gt, 1);
        if (slot < DEVROW_K) {
          cand_v[slot] = devrow_f(b);
          cand_i[slot] = j;
        }
      }
    }
  }
  __syncthreads();
  if (take == s_ties) {
    // Every element at the threshold is kept, so which of them lands in which slot cannot change
    // the answer and the collection is unordered.
    for (int j0 = tid; j0 < cols; j0 += DEVROW_T * DEVROW_U) {
      unsigned short bb[DEVROW_U];
      devrow_ilp_load(p, j0, cols, bb);
#pragma unroll
      for (int u = 0; u < DEVROW_U; ++u) {
        const int j = j0 + u * DEVROW_T;
        if (j >= cols) break;
        const unsigned short b = bb[u];
        if (devrow_is_nan(b)) continue;
        if (devrow_key(b) == thr) {
          const int slot = cnt_gt + atomicAdd(&s_n_tie, 1);
          if (slot < DEVROW_K) {
            cand_v[slot] = devrow_f(b);
            cand_i[slot] = j;
          }
        }
      }
    }
  } else {
    // The threshold value has more members than the seam has room for, so the tie is broken by
    // LOWEST ID -- the argmax's own rule (the first `take` tied ids in ascending id order land at
    // slots `cnt_gt + rank`, as in the shipped P27-W2-1 scan). CMP27X computes the same ranks over
    // WAVE-CONTIGUOUS ranges instead of 2,048-id tiles with two block barriers each: (1) every wave
    // counts the ties in its own contiguous range of 8-id groups (loads issued ahead, no barrier),
    // (2) one barrier and a prefix over the 8 wave counts, (3) a wave whose prefix is below `take`
    // walks its range in id order (group = range start + step x 32 + lane, ids ascending inside a
    // group) with a wave scan only. Same ids, same slots, same values.
    const int lane = tid % DEVROW_WAVE;
    const int ngrp = (cols + DEVROW_E - 1) / DEVROW_E;
    const int gpw = (ngrp + DEVROW_WAVES - 1) / DEVROW_WAVES;
    const int g_lo = wave * gpw;
    const int g_hi = g_lo + gpw < ngrp ? g_lo + gpw : ngrp;
    auto tie_mask = [&](int g) -> unsigned {
      unsigned m = 0u;
      if (g < g_hi) {
        unsigned short bb[DEVROW_E];
#pragma unroll
        for (int e = 0; e < DEVROW_E; ++e) {
          const int j = g * DEVROW_E + e;
          bb[e] = j < cols ? devrow_bits(p + j) : (unsigned short)0x7FC1u;  // a NaN pad: never a tie
        }
#pragma unroll
        for (int e = 0; e < DEVROW_E; ++e)
          if (!devrow_is_nan(bb[e]) && devrow_key(bb[e]) == thr) m |= 1u << e;
      }
      return m;
    };
    int mine = 0;
    for (int g0 = g_lo + lane; g0 < g_hi; g0 += DEVROW_WAVE * DEVROW_TU) {
      unsigned mm[DEVROW_TU];
#pragma unroll
      for (int u = 0; u < DEVROW_TU; ++u) mm[u] = tie_mask(g0 + u * DEVROW_WAVE);
#pragma unroll
      for (int u = 0; u < DEVROW_TU; ++u) mine += __builtin_popcount(mm[u]);
    }
#pragma unroll
    for (int off = DEVROW_WAVE / 2; off > 0; off >>= 1) mine += __shfl_xor(mine, off, DEVROW_WAVE);
    if (lane == 0) wave_ties[wave] = mine;
    __syncthreads();
    int base = 0;
#pragma unroll
    for (int w = 0; w < DEVROW_WAVES; ++w)
      if (w < wave) base += wave_ties[w];
    if (base < take) {
      int done = base;  // ties before this step in id order (wave-uniform)
      unsigned mnext = tie_mask(g_lo + lane);
      for (int g = g_lo + lane; g - lane < g_hi && done < take; g += DEVROW_WAVE) {
        unsigned m = mnext;
        mnext = tie_mask(g + DEVROW_WAVE);
        const int c = __builtin_popcount(m);
        int x = c;  // inclusive scan of c across the wave
#pragma unroll
        for (int off = 1; off < DEVROW_WAVE; off <<= 1) {
          const int y = __shfl_up(x, off, DEVROW_WAVE);
          if (lane >= off) x += y;
        }
        int rank = done + x - c;
        while (m != 0u && rank < take) {
          const int e = __builtin_ctz(m);
          m &= m - 1u;
          const int slot = cnt_gt + rank;
          if (slot < DEVROW_K) {
            cand_v[slot] = devrow_f(devrow_bits(p + g * DEVROW_E + e));
            cand_i[slot] = g * DEVROW_E + e;
          }
          ++rank;
        }
        done += __shfl(x, DEVROW_WAVE - 1, DEVROW_WAVE);
      }
    }
    if (tid == 0) s_n_tie = take;
  }
  __syncthreads();

  // ---- the order: value descending, id ascending ----------------------------------------------
  if (tid == 0) {
    const int got = (cnt_gt < DEVROW_K ? cnt_gt : DEVROW_K) + s_n_tie;
    const int have = got < DEVROW_K ? got : DEVROW_K;
    for (int i = 1; i < have; ++i) {
      const float v = cand_v[i];
      const int id = cand_i[i];
      int k = i - 1;
      while (k >= 0 && (cand_v[k] < v || (cand_v[k] == v && cand_i[k] > id))) {
        cand_v[k + 1] = cand_v[k];
        cand_i[k + 1] = cand_i[k];
        --k;
      }
      cand_v[k + 1] = v;
      cand_i[k + 1] = id;
    }
    for (int i = have; i < DEVROW_K; ++i) {
      cand_v[i] = -__builtin_inff();
      cand_i[i] = cols;
    }
  }
  __syncthreads();
  for (int i = tid; i < DEVROW_K; i += DEVROW_T) {
    tv[i] = cand_v[i];
    ti[i] = cand_i[i];
  }
}
