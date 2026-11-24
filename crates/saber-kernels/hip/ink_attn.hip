// Inkling 2 (inkling2_loop) attention, INK3 v0 (docs/INKLING2.md §3.4). Saber's own body, written
// from the architecture; bound-gated against the numpy twin in tools/capture/ink3/ink3_probe.py.
//
//   ink_attn_bias_sink_v0   one query row x one q head a workgroup (256 lanes): GQA (q head h reads
//       KV head h / group), no RoPE; logits = q . k * 128^-1/2 + bias, bias = rel[t, h, i - j] for
//       0 <= i - j < extent, 0 beyond; masked j > i, and i - j >= window when window > 0 (local);
//       a per-row per-head SINK logit with a zero value enlarges the denominator only; f32
//       online softmax over 256-key tiles; o f32 [T, H, 128].
//   K / V come from the paged pools (block table [seq][pos / 64] * 64 + pos % 64, bf16 rows of
//   KV * 128); the rows this step wrote are already in the pools (kv_conv_post_append runs first).
//   A correctness-first body (one key a lane, a 128-wide dot from global memory): the INK3 fast
//   prefill / decode forms are re-tiles of it, gated against it.
//
// Built with -ffp-contract=off.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define INK_HD 128
#define INK_PAGE 64
#define INK_TILE 256

__device__ __forceinline__ float bf2f(unsigned short u) {
  return __builtin_bit_cast(float, ((unsigned)u) << 16);
}

__device__ __forceinline__ float wg_max(float v, float* red) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
  if (lane == 0) red[w] = v;
  __syncthreads();
  float m = red[0];
#pragma unroll
  for (int i = 1; i < INK_TILE / 32; ++i) m = fmaxf(m, red[i]);
  __syncthreads();
  return m;
}

__device__ __forceinline__ float wg_sum(float v, float* red) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
  if (lane == 0) red[w] = v;
  __syncthreads();
  float s = red[0];
#pragma unroll
  for (int i = 1; i < INK_TILE / 32; ++i) s += red[i];
  __syncthreads();
  return s;
}

extern "C" __global__ __launch_bounds__(256) void ink_attn_bias_sink_v0(
    const unsigned short* __restrict__ q, int ld_q,        // [T][H * 128] bf16 (post clip / tau)
    const unsigned short* __restrict__ k_pool, const unsigned short* __restrict__ v_pool,  // [rows][KV * 128]
    const int* __restrict__ table, int table_ld,
    const int* __restrict__ row_seq, const int* __restrict__ row_pos,
    const float* __restrict__ rel, int ld_rel,             // [T][H * extent] f32 (r . P)
    const float* __restrict__ sink, int ld_sink,           // [T][H] f32
    float* __restrict__ o, int ld_o,                       // [T][H * 128] f32
    int H, int KV, int extent, int window) {
  __shared__ float qs[INK_HD];
  __shared__ float ps[INK_TILE];
  __shared__ float red[INK_TILE / 32];
  __shared__ float oacc[2][INK_HD];
  const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
  const int kvh = h / (H / KV);
  const int C = KV * INK_HD;
  const int sq = row_seq[t], pos = row_pos[t];
  const int lo = window > 0 ? max(0, pos - window + 1) : 0;
  if (tid < INK_HD) qs[tid] = bf2f(q[(long)t * ld_q + h * INK_HD + tid]);
  __syncthreads();
  float m = -INFINITY, l = 0.f, acc = 0.f;   // acc: dim (tid & 127) over this lane's key half
  const int d = tid & (INK_HD - 1), half = tid >> 7;
  for (int base = lo; base <= pos; base += INK_TILE) {
    const int j = base + tid;
    float s = -INFINITY;
    if (j <= pos) {
      const long row = (long)table[(long)sq * table_ld + j / INK_PAGE] * INK_PAGE + j % INK_PAGE;
      const unsigned short* kr = k_pool + row * C + kvh * INK_HD;
      float dot = 0.f;
      for (int i = 0; i < INK_HD; ++i) dot = dot + qs[i] * bf2f(kr[i]);
      s = dot * 0.08838834764831845f;
      const int dist = pos - j;
      if (dist < extent) s = s + rel[(long)t * ld_rel + h * extent + dist];
    }
    const float tm = wg_max(s, red);
    const float mn = fmaxf(m, tm);
    const float p = (j <= pos) ? expf(s - mn) : 0.f;
    ps[tid] = p;
    const float corr = (m == -INFINITY) ? 0.f : expf(m - mn);
    const float ts = wg_sum(p, red);   // its barriers also publish ps
    l = l * corr + ts;
    // o: lanes 0..127 take keys 0..127 of the tile, lanes 128..255 keys 128..255, dim d
    float a = 0.f;
    const int jn = min(INK_TILE / 2, pos - base - half * (INK_TILE / 2) + 1);
    for (int i = 0; i < jn; ++i) {
      const int jj = base + half * (INK_TILE / 2) + i;
      const long row = (long)table[(long)sq * table_ld + jj / INK_PAGE] * INK_PAGE + jj % INK_PAGE;
      a = a + ps[half * (INK_TILE / 2) + i] * bf2f(v_pool[row * C + kvh * INK_HD + d]);
    }
    acc = acc * corr + a;
    m = mn;
    __syncthreads();
  }
  // the sink: one more logit with a zero value
  const float sk = sink[(long)t * ld_sink + h];
  const float mf = fmaxf(m, sk);
  const float c = expf(m - mf);
  const float lf = l * c + expf(sk - mf);
  oacc[half][d] = acc;
  __syncthreads();
  if (tid < INK_HD) o[(long)t * ld_o + h * INK_HD + d] = ((oacc[0][d] + oacc[1][d]) * c) / lf;
}
