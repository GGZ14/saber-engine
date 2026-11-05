// Saber paged attention, CLEANROOM-2 (CR2-2c): the split-KV combine (spec section 2.8).
//
// One block of 256 threads per job = (token, q head); thread d owns output dim d. The weights of the used
// segments go to a dynamic LDS row (splits floats); their total is a 256-leaf descending tree; the weighted
// sum per dim runs four explicit-fma chains over the segments in steps of four.
#pragma once
#include "sattn_prims.h"

namespace sattn {

template <bool RAGGED, bool HALF_PARTIALS>
__device__ __forceinline__ void combine_job(const Params& a, int splits, int tile, const int* __restrict__ cu_q) {
  extern __shared__ float wrow[];
  __shared__ float tree[256];
  const int d = threadIdx.x;
  const int job = blockIdx.x;
  const int tok = (int)((unsigned)job / (unsigned)a.nqh);
  const int qh = job - tok * a.nqh;
  uint16_t* out = (uint16_t*)a.dst + (long)job * kHead + d;

  int seq;
  if constexpr (RAGGED) {
    if (tok >= cu_q[a.nseq]) {
      *out = 0;
      return;
    }
    seq = 0;
    while (tok >= cu_q[seq + 1]) ++seq;
  } else {
    seq = (int)((unsigned)tok / (unsigned)a.qrows);
  }
  const int nctx = a.ctx_len[seq];
  if (nctx <= 0) {
    *out = 0;
    return;
  }
  const int ntl = (int)((unsigned)(nctx + tile - 1) / (unsigned)tile);
  const int tps = (int)((unsigned)(ntl + splits - 1) / (unsigned)splits);
  const int used = (int)((unsigned)(ntl + tps - 1) / (unsigned)tps);
  const long nrow = (long)a.nseq * a.qrows * a.nqh * splits;
  const long base = (long)job * splits;
  const float* pm;
  const float* pl;
  const _Float16* p16 = (const _Float16*)a.work;
  const float* p32 = (const float*)a.work;
  if constexpr (HALF_PARTIALS) {
    pm = (const float*)(p16 + nrow * kHead);
  } else {
    pm = p32 + nrow * kHead;
  }
  pl = pm + nrow;

  // the max is order-free (every m >= the floor): each wave reads the segments lane-strided and reduces
  // them itself (no LDS, no barrier)
  const int lane = d & 31;
  float gmax = kFloor;
  for (int s = lane; s < used; s += 32) gmax = fmaxf(gmax, pm[base + s]);
#pragma unroll
  for (int x = 16; x >= 1; x >>= 1)
    gmax = fmaxf(gmax, __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute((lane ^ x) << 2, __builtin_bit_cast(int, gmax))));

  float t = 0.0f;
  for (int s = d; s < used; s += 256) {
    const float m = pm[base + s];
    float w = (m <= kFloor) ? 0.0f : ex2(m - gmax);
    if constexpr (HALF_PARTIALS) {
      w = w * pl[base + s];
      t = t + w;
    } else {
      t = t + w * pl[base + s];
    }
    wrow[s] = w;
  }
  // 256-leaf descending tree (leaf d += leaf d + step for d < step): the two wide levels through LDS,
  // the last six inside every wave (lane L holds leaf L), so no barrier follows level 64
  tree[d] = t;
  lds_sync();
  if (d < 128) tree[d] = tree[d] + tree[d + 128];
  lds_sync();
  if (d < 64) tree[d] = tree[d] + tree[d + 64];
  lds_sync();
  float r32 = tree[lane] + tree[lane + 32];
#pragma unroll
  for (int step = 16; step >= 1; step >>= 1) {
    const int src = lane + step < 32 ? lane + step : lane;
    r32 = r32 + __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute(src << 2, __builtin_bit_cast(int, r32)));
  }
  const float tot = __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, r32)));

  float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
  int s = 0;
  if constexpr (HALF_PARTIALS) {
    const _Float16* col = p16 + base * kHead + d;
    // twelve segments a step (12 loads in flight, four weights per LDS read), then the 4-step form; within
    // each chain the segments stay in increasing order
    typedef float f32x4 __attribute__((ext_vector_type(4)));
    constexpr int kQuads = 3;
#pragma unroll 1
    for (; s + 4 * kQuads <= used; s += 4 * kQuads) {
      f16x2 h[2 * kQuads];
#pragma unroll
      for (int j = 0; j < 2 * kQuads; ++j) h[j] = (f16x2){col[(s + 2 * j) * kHead], col[(s + 2 * j + 1) * kHead]};
      sched_fence();
#pragma unroll
      for (int j = 0; j < kQuads; ++j) {
        const f32x4 w4 = *(const SATTN_LDS f32x4*)lds_ptr(&wrow[s + 4 * j]);
        c0 = fmaf(w4.x, (float)h[2 * j].x, c0);
        c1 = fmaf(w4.y, (float)h[2 * j].y, c1);
        c2 = fmaf(w4.z, (float)h[2 * j + 1].x, c2);
        c3 = fmaf(w4.w, (float)h[2 * j + 1].y, c3);
        sched_fence();
      }
    }
    for (; s + 4 <= used; s += 4) {
      c0 = fmaf(wrow[s + 0], (float)col[(s + 0) * kHead], c0);
      c1 = fmaf(wrow[s + 1], (float)col[(s + 1) * kHead], c1);
      c2 = fmaf(wrow[s + 2], (float)col[(s + 2) * kHead], c2);
      c3 = fmaf(wrow[s + 3], (float)col[(s + 3) * kHead], c3);
    }
    for (; s < used; ++s) c0 = fmaf(wrow[s], (float)col[s * kHead], c0);
  } else {
    const float* col = p32 + base * kHead + d;
    for (; s + 4 <= used; s += 4) {
      c0 = fmaf(wrow[s + 0], col[(s + 0) * kHead], c0);
      c1 = fmaf(wrow[s + 1], col[(s + 1) * kHead], c1);
      c2 = fmaf(wrow[s + 2], col[(s + 2) * kHead], c2);
      c3 = fmaf(wrow[s + 3], col[(s + 3) * kHead], c3);
    }
    for (; s < used; ++s) c0 = fmaf(wrow[s], col[s * kHead], c0);
  }
  float av = 0.0f;
  av = av + c0;
  av = av + c1;
  av = av + c2;
  av = av + c3;
  const int kvh = (int)((unsigned)qh / ((unsigned)a.nqh / (unsigned)a.nkvh));
  const float vd = a.vscale ? a.vscale[seq * a.nkvh + kvh] : 1.0f;
  const float r = tot > 0.0f ? (av * vd) / tot : 0.0f;
  *out = (uint16_t)bf16_bits(r);
}

}  // namespace sattn
