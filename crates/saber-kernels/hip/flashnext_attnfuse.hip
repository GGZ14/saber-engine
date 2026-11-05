// Flash-Next (ATTN-OPT-3): attention-neighbour launch fusion at the decode / verify forms -- ADDED symbols in
// their own object; `flashnext_glue`, `flashnext_glue_kv8`, `flashnext_attn*` and `flashnext_hc` keep their bytes.
// Each symbol writes the SAME bytes as the chain it replaces (the oracle is byte equality with the chain,
// tools/capture/attn3/attn3_twin.py); only launches go.
//
//   fn_attnfuse_front_h256_{bf16,fp8}   `--fn-fuse f6`: the QSA layer's front (3 launches -> 1) =
//        fn_qk_norm_rope_h256_rot64_mi1_v2 (workgroups [0, nb1): eight (row, head) waves each, its body line
//        for line) + fn_idx_q_compress_h128_v2 (workgroups [nb1, nb1 + nb2): four rows each, its body line
//        for line) + fn_kv_append_ring_h256_{bf16,fp8} folded into the K head's wave: the wave that rounds a
//        row's K to bf16 (the bytes it still stores to k_out) writes those same bf16 values -- or their
//        e4m3 codes by the append's convert (value / scale, v_cvt_pk_fp8_f32, no clamp) -- into the page
//        slot, copies / converts the row's V, and (kv head 0) copies the indexer's raw key + its three
//        positions into the ring. The indexer reads the ring only at positions BEFORE the step's first
//        row and the append writes only the step's rows, so the ring slots are disjoint whenever a
//        sequence's rows + 3 <= the ring's 8 rows (the host enables the form at decode only, rows <= 5).
//   fn_attnfuse_combine_gate_q8_h256    `--fn-fuse f7`: the split-KV combine (sattn_combine.h's
//        combine_job<false, true>, its arithmetic copied op for op) + family 1's gate + the W8 o_proj's
//        act quant (fn_attn_gate_q8_t256): the block of job (token, q head) holds that head's 256 merged
//        values, rounds each to bf16 (the core it still stores), gates it, and -- a 128-element quant
//        group being half a head -- takes each group's amax over its four waves (a max: order-free).
//
// Built with -ffp-contract=off (every fp32 multiply and add rounds on its own), as its parents.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include "sattn_prims.h"

typedef unsigned int u32;
union G4 { uint2 u2; __bf16 h[4]; };

__device__ __forceinline__ float bf(float x) { return (float)(__bf16)x; }
__device__ __forceinline__ float rsqrt_exact(float v) { return (float)(1.0 / sqrt((double)v)); }
__device__ __forceinline__ float bfmul(float a, float b) {
  return __builtin_bit_cast(float, __builtin_bit_cast(unsigned, a * b) & 0xffff0000u);
}
#define AF_EPS 1e-6f

// ------------------------------------------------------------------------------- the glue helpers
// (flashnext_glue.hip's fg_rope_ops / fg_apply4 / fg_tree, the same operations in the same order)
__device__ __forceinline__ void af_rope_ops(const __bf16 *__restrict__ cos_sin, const long p3[3], int lane,
                                            int rd, int sec_h, int sec_w, float c[4], float sn[4]) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int i = (lane & 7) * 4 + j;
    const int r = i % 3;
    const int sec = (r == 1 && i <= 3 * sec_h) ? 1 : ((r == 2 && i <= 3 * sec_w) ? 2 : 0);
    const long p = sec == 0 ? p3[0] : (sec == 1 ? p3[1] : p3[2]);
    c[j] = (float)cos_sin[p * rd + i];
    sn[j] = (float)cos_sin[p * rd + rd / 2 + i];
  }
}
__device__ __forceinline__ void af_apply4(float x[4], float inv, const float w1[4], const float c[4],
                                          const float sn[4], int lane) {
  float n[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) n[j] = bf((x[j] * inv) * w1[j]);
  float other[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) other[j] = __shfl_xor(n[j], 8);
  if (lane < 16) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float t = bfmul(other[j], sn[j]);
      n[j] = bfmul(n[j], c[j]) + (lane < 8 ? -t : t);
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) x[j] = n[j];
}
__device__ __forceinline__ float af_tree(float sum) {
  for (int o = 1; o < 32; o <<= 1) sum += __shfl_down(sum, o);
  return sum;
}

// the append's e4m3 pair (flashnext_glue_kv8.hip): divide, hardware convert, no clamp
__device__ __forceinline__ u32 af_e4m3_pair(float a, float b, float scale) {
  return __builtin_amdgcn_cvt_pk_fp8_f32(a / scale, b / scale, 0u, false) & 0xffffu;
}

struct FrontArgs {
  // fn_qk_norm_rope_h256_rot64_mi1_v2
  const __bf16 *qkv, *qw, *kw, *cos_sin;
  const long *pos;
  __bf16 *q_out, *k_out;
  int m, nq, nkv, qkv_n;
  float eps;
  int sec_h, sec_w;
  // fn_idx_q_compress_h128_v2
  const __bf16 *iqk;
  long iqk_stride;
  const __bf16 *iqw, *ikw;
  const int *row_seq;
  const long *row_lpos;
  const int *seq_row0;
  const long *seq_pos0;
  const int *ring_row0;
  __bf16 *ring;
  long *ring_pos;
  int ring_rows;
  const int *block_table;
  int max_blocks;
  __bf16 *iq_out, *comp;
  // fn_kv_append_ring_h256_*
  const __bf16 *value;
  long value_stride;
  void *kv;
  const long *slot;
  long block_stride, page_stride, head_stride;
  const __bf16 *ikey;
  long ikey_stride;
  const long *ring_slot;
  float k_scale, v_scale;
};

// The norm/rope job of fn_qk_norm_rope_h256_rot64_mi1_v2 (workgroup `bx` of that grid), with the K wave's
// append.
template <bool KV8>
__device__ __forceinline__ void af_norm_rope_append(const FrontArgs &a, long bx) {
  constexpr int HD = 256, RD = 64, W = 32;
  __shared__ float s_sum[8], s_inv[8];
  const int lane = (int)(threadIdx.x & 31), wave = (int)(threadIdx.x >> 5);
  const int heads = a.nq + a.nkv;
  const long npair = (long)a.m * heads;
  const long pr = bx * 8 + wave;
  const bool live = pr < npair;
  const long pair = live ? pr : npair - 1;
  const int tok = (int)(pair / heads), head = (int)(pair - (long)tok * heads);
  const bool is_k = head >= a.nq;
  const int lh = is_k ? head - a.nq : head;
  const __bf16 *in = a.qkv + (long)tok * a.qkv_n + (is_k ? (long)a.nq * 2 * HD + (long)lh * HD : (long)lh * 2 * HD);
  const __bf16 *w = is_k ? a.kw : a.qw;
  __bf16 *out = is_k ? a.k_out + (long)tok * a.nkv * HD + (long)lh * HD : a.q_out + (long)tok * a.nq * HD + (long)lh * HD;
  const long p3[3] = {a.pos[tok], a.pos[(long)a.m + tok], a.pos[2 * (long)a.m + tok]};
  G4 vin[2], vw[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    vin[r].u2 = ((const uint2 *)(in + (r * W + lane) * 4))[0];
    vw[r].u2 = ((const uint2 *)(w + (r * W + lane) * 4))[0];
  }
  float c[4], sn[4];
  af_rope_ops(a.cos_sin, p3, lane, RD, a.sec_h, a.sec_w, c, sn);
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const float v0 = (float)vin[r].h[0], v1 = (float)vin[r].h[1];
    const float v2 = (float)vin[r].h[2], v3 = (float)vin[r].h[3];
    a0 += v0 * v0;
    a1 += v1 * v1;
    a2 += v2 * v2;
    a3 += v3 * v3;
  }
  const float sum = af_tree(((a0 + a1) + a2) + a3);
  if (lane == 0) s_sum[wave] = sum;
  __syncthreads();
  if (wave == 0 && lane < 8) s_inv[lane] = rsqrt_exact(s_sum[lane] / (float)HD + a.eps);
  __syncthreads();
  const float inv = s_inv[wave];
  float x[2][4], w1[2][4];
#pragma unroll
  for (int r = 0; r < 2; ++r)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      x[r][j] = (float)vin[r].h[j];
      w1[r][j] = (float)vw[r].h[j] + 1.f;
    }
  af_apply4(x[0], inv, w1[0], c, sn, lane);
  float n1[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) n1[j] = bf((x[1][j] * inv) * w1[1][j]);
  if (!live) return;
  G4 o0, o1;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    o0.h[j] = (__bf16)x[0][j];
    o1.h[j] = (__bf16)n1[j];
  }
  ((uint2 *)(out + lane * 4))[0] = o0.u2;
  ((uint2 *)(out + (W + lane) * 4))[0] = o1.u2;
  if (!is_k) return;
  // ---- the append of row `tok`, kv head `lh` (fn_kv_append_ring_h256_*'s job for (tok, lh))
  const long s = a.slot[tok];
  if (s >= 0) {
    const long blk = s / 16, off = s % 16;
    const __bf16 *vp = a.value + (long)tok * a.value_stride + (long)lh * HD;
    G4 v0, v1;
    v0.u2 = ((const uint2 *)(vp + lane * 4))[0];
    v1.u2 = ((const uint2 *)(vp + (W + lane) * 4))[0];
    const long base = blk * a.block_stride + (long)lh * a.head_stride + off * a.page_stride;
    if constexpr (KV8) {
      unsigned char *dst = (unsigned char *)a.kv + base;
      const u32 k0 = af_e4m3_pair((float)o0.h[0], (float)o0.h[1], a.k_scale) |
                     (af_e4m3_pair((float)o0.h[2], (float)o0.h[3], a.k_scale) << 16);
      const u32 k1 = af_e4m3_pair((float)o1.h[0], (float)o1.h[1], a.k_scale) |
                     (af_e4m3_pair((float)o1.h[2], (float)o1.h[3], a.k_scale) << 16);
      const u32 q0 = af_e4m3_pair((float)v0.h[0], (float)v0.h[1], a.v_scale) |
                     (af_e4m3_pair((float)v0.h[2], (float)v0.h[3], a.v_scale) << 16);
      const u32 q1 = af_e4m3_pair((float)v1.h[0], (float)v1.h[1], a.v_scale) |
                     (af_e4m3_pair((float)v1.h[2], (float)v1.h[3], a.v_scale) << 16);
      *(u32 *)(dst + lane * 4) = k0;
      *(u32 *)(dst + (W + lane) * 4) = k1;
      *(u32 *)(dst + HD + lane * 4) = q0;
      *(u32 *)(dst + HD + (W + lane) * 4) = q1;
    } else {
      __bf16 *dst = (__bf16 *)a.kv + base;
      ((uint2 *)(dst + lane * 4))[0] = o0.u2;
      ((uint2 *)(dst + (W + lane) * 4))[0] = o1.u2;
      ((uint2 *)(dst + HD + lane * 4))[0] = v0.u2;
      ((uint2 *)(dst + HD + (W + lane) * 4))[0] = v1.u2;
    }
  }
  if (lh != 0) return;
  const long rs = a.ring_slot[tok];
  if (rs < 0) return;
  ((uint2 *)(a.ring + rs * 128))[lane] = ((const uint2 *)(a.ikey + (long)tok * a.ikey_stride))[lane];
  if (lane < 3) a.ring_pos[rs * 3 + lane] = a.pos[(long)lane * a.m + tok];
}

// The indexer job of fn_idx_q_compress_h128_v2 (workgroup `bx` of that grid), line for line.
__device__ __forceinline__ void af_idx(const FrontArgs &a, int bx) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int t = bx * 4 + (wave >> 1);
  const int m = a.m;
  if (t >= m) return;
  const __bf16 *qk = a.iqk;
  const long qk_stride = a.iqk_stride;
  const long *pos = a.pos;
  if ((wave & 1) == 0) {
    const long p3[3] = {pos[t], pos[(long)m + t], pos[2 * (long)m + t]};
    G4 v[4], vw;
#pragma unroll
    for (int h = 0; h < 4; ++h) v[h].u2 = *(const uint2 *)(qk + (long)t * qk_stride + h * 128 + 4 * lane);
    vw.u2 = *(const uint2 *)(a.iqw + 4 * lane);
    float c[4], sn[4];
    af_rope_ops(a.cos_sin, p3, lane, 64, a.sec_h, a.sec_w, c, sn);
    float x[4][4], sum[4];
#pragma unroll
    for (int h = 0; h < 4; ++h) {
#pragma unroll
      for (int j = 0; j < 4; ++j) x[h][j] = (float)v[h].h[j];
      const float a0 = x[h][0] * x[h][0], a1 = x[h][1] * x[h][1];
      const float a2 = x[h][2] * x[h][2], a3 = x[h][3] * x[h][3];
      sum[h] = af_tree(((a0 + a1) + a2) + a3);
    }
    const float s0 = __shfl(sum[0], 0), s1 = __shfl(sum[1], 0);
    const float s2 = __shfl(sum[2], 0), s3 = __shfl(sum[3], 0);
    const int hl = lane & 3;
    const float mine = hl == 0 ? s0 : (hl == 1 ? s1 : (hl == 2 ? s2 : s3));
    const float invl = rsqrt_exact(mine / 128.f + AF_EPS);
    float w1[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) w1[j] = (float)vw.h[j] + 1.f;
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      af_apply4(x[h], __shfl(invl, h), w1, c, sn, lane);
      G4 o;
#pragma unroll
      for (int j = 0; j < 4; ++j) o.h[j] = (__bf16)x[h][j];
      *(uint2 *)(a.iq_out + ((long)t * 4 + h) * 128 + 4 * lane) = o.u2;
    }
    return;
  }
  const long lp = a.row_lpos[t];
  if (lp < 3 || ((lp + 1) & 3) != 0) return;
  const int s = a.row_seq[t];
  G4 vw;
  vw.u2 = *(const uint2 *)(a.ikw + 4 * lane);
  const long g = lp >> 2;
  const long page = a.block_table[(long)s * a.max_blocks + (g >> 2)];
  const long p0 = a.seq_pos0[s];
  const int r0 = a.seq_row0[s];
  const long rbase = (long)a.ring_row0[s] * a.ring_rows;
  G4 mv[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const long pj = lp - 3 + j;
    const __bf16 *src = pj >= p0 ? qk + (long)(r0 + (pj - p0)) * qk_stride + 512 : a.ring + (rbase + pj % a.ring_rows) * 128;
    mv[j].u2 = *(const uint2 *)(src + 4 * lane);
  }
  const long pf = lp - 3;
  long p3[3];
  if (pf >= p0) {
    const long rr = r0 + (pf - p0);
    p3[0] = pos[rr]; p3[1] = pos[(long)m + rr]; p3[2] = pos[2 * (long)m + rr];
  } else {
    const long rr = rbase + pf % a.ring_rows;
    p3[0] = a.ring_pos[rr * 3]; p3[1] = a.ring_pos[rr * 3 + 1]; p3[2] = a.ring_pos[rr * 3 + 2];
  }
  float c[4], sn[4];
  af_rope_ops(a.cos_sin, p3, lane, 64, a.sec_h, a.sec_w, c, sn);
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[e] = acc[e] + (float)mv[j].h[e];
  float x[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) x[e] = bf(acc[e] * 0.25f);
  const float a0 = x[0] * x[0], a1 = x[1] * x[1], a2 = x[2] * x[2], a3 = x[3] * x[3];
  const float sum = af_tree(((a0 + a1) + a2) + a3);
  const float inv = rsqrt_exact(__shfl(sum, 0) / 128.f + AF_EPS);
  float w1[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) w1[j] = (float)vw.h[j] + 1.f;
  af_apply4(x, inv, w1, c, sn, lane);
  G4 o;
#pragma unroll
  for (int e = 0; e < 4; ++e) o.h[e] = (__bf16)x[e];
  *(uint2 *)(a.comp + (page * 4 + (g & 3)) * 128 + 4 * lane) = o.u2;
}

// grid (ceil(m (nq + nkv) / 8) + ceil(m / 4)), 256 threads; the first part's workgroups run the norm/rope
// (+ append) jobs, the rest the indexer's (a workgroup-uniform branch: the norm job's barriers are met by
// all eight of its waves).
template <bool KV8>
__device__ __forceinline__ void af_front(const FrontArgs &a) {
  const long nb1 = ((long)a.m * (a.nq + a.nkv) + 7) / 8;
  const long bx = blockIdx.x;
  if (bx < nb1) {
    af_norm_rope_append<KV8>(a, bx);
  } else {
    af_idx(a, (int)(bx - nb1));
  }
}

// The kernargs: qk_norm_rope's (sec_t dropped: unused by its body), then the indexer's not already named
// (its cos_sin / pos / m / sec_h / sec_w are the norm's: the engine passes the same values), then the
// append's not already named (its ring / ring_pos / pos / m likewise); the bf16 form ignores the scales.
#define AF_PARAMS                                                                                          \
  const __bf16 *__restrict__ qkv, const __bf16 *__restrict__ qw, const __bf16 *__restrict__ kw,            \
      const __bf16 *__restrict__ cos_sin, const long *__restrict__ pos, __bf16 *__restrict__ q_out,        \
      __bf16 *__restrict__ k_out, int m, int nq, int nkv, int qkv_n, float eps, int sec_h, int sec_w,      \
      const __bf16 *__restrict__ iqk, long iqk_stride, const __bf16 *__restrict__ iqw,                     \
      const __bf16 *__restrict__ ikw, const int *__restrict__ row_seq, const long *__restrict__ row_lpos,  \
      const int *__restrict__ seq_row0, const long *__restrict__ seq_pos0,                                 \
      const int *__restrict__ ring_row0, __bf16 *__restrict__ ring, long *__restrict__ ring_pos,           \
      int ring_rows, const int *__restrict__ block_table, int max_blocks, __bf16 *__restrict__ iq_out,     \
      __bf16 *__restrict__ comp, const __bf16 *__restrict__ value, long value_stride, KVT *__restrict__ kv, \
      const long *__restrict__ slot, long block_stride, long page_stride, long head_stride,                \
      const __bf16 *__restrict__ ikey, long ikey_stride, const long *__restrict__ ring_slot, float k_scale, \
      float v_scale
#define AF_FILL                                                                                             \
  FrontArgs a{qkv,       qw,       kw,          cos_sin,     pos,          q_out,       k_out,     m,         \
              nq,        nkv,      qkv_n,       eps,         sec_h,        sec_w,       iqk,       iqk_stride, \
              iqw,       ikw,      row_seq,     row_lpos,    seq_row0,     seq_pos0,    ring_row0, ring,       \
              ring_pos,  ring_rows, block_table, max_blocks, iq_out,       comp,        value,     value_stride, \
              (void *)kv, slot,    block_stride, page_stride, head_stride, ikey,        ikey_stride, ring_slot, \
              k_scale,   v_scale}

#define KVT __bf16
extern "C" __global__ __launch_bounds__(256) void fn_attnfuse_front_h256_bf16(AF_PARAMS) {
  AF_FILL;
  af_front<false>(a);
}
#undef KVT
#define KVT unsigned char
extern "C" __global__ __launch_bounds__(256) void fn_attnfuse_front_h256_fp8(AF_PARAMS) {
  AF_FILL;
  af_front<true>(a);
}
#undef KVT

// ------------------------------------------------------------------------- combine + gate + quant
// sattn::combine_job<false, true> (sattn_combine.h) up to the merged value, op for op; then the core's
// bf16 bits are stored as the combine stores them and the gate + quant run on them.
extern "C" __global__ __launch_bounds__(256) void fn_attnfuse_combine_gate_q8_h256(
    sattn::Params a, int splits, int tile, const __bf16 *__restrict__ gate, __bf16 *__restrict__ gated,
    unsigned char *__restrict__ q, float *__restrict__ scales, int n, int hd, int gs) {
  using namespace sattn;
  extern __shared__ float wrow[];
  __shared__ float tree[256];
  __shared__ float gmaxs[8];
  const int d = threadIdx.x;
  const int job = blockIdx.x;
  const int tok = (int)((unsigned)job / (unsigned)a.nqh);
  const int qh = job - tok * a.nqh;
  uint16_t *out = (uint16_t *)a.dst + (long)job * kHead + d;
  const int seq = (int)((unsigned)tok / (unsigned)a.qrows);
  const int nctx = a.ctx_len[seq];
  uint32_t bits = 0;
  if (nctx > 0) {
    const int ntl = (int)((unsigned)(nctx + tile - 1) / (unsigned)tile);
    const int tps = (int)((unsigned)(ntl + splits - 1) / (unsigned)splits);
    const int used = (int)((unsigned)(ntl + tps - 1) / (unsigned)tps);
    const long nrow = (long)a.nseq * a.qrows * a.nqh * splits;
    const long base = (long)job * splits;
    const _Float16 *p16 = (const _Float16 *)a.work;
    const float *pm = (const float *)(p16 + nrow * kHead);
    const float *pl = pm + nrow;
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
      w = w * pl[base + s];
      t = t + w;
      wrow[s] = w;
    }
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
    const _Float16 *col = p16 + base * kHead + d;
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
        const f32x4 w4 = *(const SATTN_LDS f32x4 *)lds_ptr(&wrow[s + 4 * j]);
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
    float av = 0.0f;
    av = av + c0;
    av = av + c1;
    av = av + c2;
    av = av + c3;
    const int kvh = (int)((unsigned)qh / ((unsigned)a.nqh / (unsigned)a.nkvh));
    const float vd = a.vscale ? a.vscale[seq * a.nkvh + kvh] : 1.0f;
    const float r = tot > 0.0f ? (av * vd) / tot : 0.0f;
    bits = bf16_bits(r);
  }
  *out = (uint16_t)bits;
  // ---- fn_attn_gate_q8_t256 on this head's 256 values: sig = bf16(1 / (1 + expf(-g))), p = bf16(core sig),
  // the 128-group amax, sc = max(amax / 448, 1 / (448 x 512)), q = e4m3(clamp(p / sc, +-448))
  const float core = __builtin_bit_cast(float, bits << 16);
  const int e = qh * hd + d;
  const __bf16 g = gate[(long)tok * gs + (long)qh * 2 * hd + hd + d];
  const float sig = bf(1.f / (1.f + expf(-(float)g)));
  const __bf16 o = (__bf16)(core * sig);
  gated[(long)tok * n + e] = o;
  const float ob = (float)o;
  float am = fmaxf(0.f, fabsf(ob));   // the gate's amax starts at 0 (fmaxf drops a NaN either way)
  for (int x = 1; x < 32; x <<= 1) am = fmaxf(am, __shfl_xor(am, x));
  if ((d & 31) == 0) gmaxs[d >> 5] = am;
  __syncthreads();
  const int g4 = (d >> 7) * 4;
  am = fmaxf(fmaxf(gmaxs[g4], gmaxs[g4 + 1]), fmaxf(gmaxs[g4 + 2], gmaxs[g4 + 3]));
  const float sc = fmaxf(am / 448.f, 1.f / (448.f * 512.f));
  if ((e & 127) == 0) scales[(long)tok * (n >> 7) + (e >> 7)] = sc;
  const float u = fminf(fmaxf(ob / sc, -448.f), 448.f);
  q[(long)tok * n + e] = (unsigned char)(__builtin_amdgcn_cvt_pk_fp8_f32(u, u, 0u, false) & 0xffu);
}
