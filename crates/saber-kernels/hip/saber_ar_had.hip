// Saber source (2026-11-11): the `--macaq-had on` twin of the served `_nq` epilogue
// `saber_ar_oneshot_2rank_exact_nq_ssqfma_bf16` (sar_nq.h, SSQ = fma, no check). Its lines up to the bf16-rounded
// normed value nv, then `macaq_had32` (macaq_had.h) per word, then the parent's amax / scale / e4m3 tail on
// the transformed values. `had_t` (f32 [K]) is appended LAST. Built -ffp-contract=off (saber-kbuild's
// `saber_ar_` row) like the parent; `saber_ar_exact` is untouched.
#include "sar_exact.h"
#include "sar_nq.h"
#include "macaq_had.h"

namespace sar {

__device__ __forceinline__ void nq_row_had(const int4* in, int4* peer_scratch, int4* my_scratch, const int4* residual,
                                           const int4* norm_w, unsigned char* q_out, float* scale_out, int4* res_out,
                                           u32* peer_flags, u32* my_flags, u32* seq, int k16, int stride16, float eps,
                                           int drain, int acq, const float* had_t) {
  __shared__ u32 bcast;
  __shared__ float part[kNqWaves];
  const int b = (int)blockIdx.x;
  const int tid = (int)threadIdx.x;
  const int lane = tid & 31, wave = tid >> 5;
  const u32 s = take_seq(seq, b, &bcast);
  const long so = (long)(s & 1u) * stride16;
  int4* peer = peer_scratch + so;
  const int4* mine = my_scratch + so;
  const long row = (long)b * k16;

  int4 x[kNqMaxWords];
  bool ok[kNqMaxWords];
  long at[kNqMaxWords];
#pragma unroll
  for (int i = 0; i < kNqMaxWords; ++i) {
    const int w = tid + i * kNqThreads;
    ok[i] = w < k16;
    at[i] = row + (ok[i] ? w : 0);
    x[i] = make_int4(0, 0, 0, 0);
    if (ok[i]) x[i] = in[at[i]];
  }
#pragma unroll
  for (int i = 0; i < kNqMaxWords; ++i) {
    if (ok[i]) peer[at[i]] = x[i];
  }

  drain_and_sync(drain);
  if (tid == 0) {
    (void)signal_wait(peer_flags + b, my_flags + b, s);
  }
  __syncthreads();
  acquire_opt(acq);

  float v[kNqMaxWords][8];
  float ssq = 0.0f;
#pragma unroll
  for (int i = 0; i < kNqMaxWords; ++i) {
    int4 p = make_int4(0, 0, 0, 0), r = p;
    if (ok[i]) {
      p = mine[at[i]];
      r = residual[at[i]];
    }
    const u32 sum[4] = {add_lane<kBf16>((u32)x[i].x, (u32)p.x), add_lane<kBf16>((u32)x[i].y, (u32)p.y),
                        add_lane<kBf16>((u32)x[i].z, (u32)p.z), add_lane<kBf16>((u32)x[i].w, (u32)p.w)};
    const u32 rr[4] = {(u32)r.x, (u32)r.y, (u32)r.z, (u32)r.w};
    u32 ro[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float lo = bf_lo(sum[e]) + bf_lo(rr[e]);
      const float hi = bf_hi(sum[e]) + bf_hi(rr[e]);
      v[i][2 * e] = ok[i] ? lo : 0.0f;
      v[i][2 * e + 1] = ok[i] ? hi : 0.0f;
      ro[e] = rne16x2(lo, hi);
    }
    if (ok[i]) res_out[at[i]] = make_int4((int)ro[0], (int)ro[1], (int)ro[2], (int)ro[3]);
#pragma unroll
    for (int j = 0; j < 8; ++j) ssq = __builtin_fmaf(v[i][j], v[i][j], ssq);
  }

#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ssq = ssq + __shfl_xor(ssq, o, 32);
  if (lane == 0) part[wave] = ssq;
  __syncthreads();
  const float total = ((part[0] + part[4]) + (part[2] + part[6])) + ((part[1] + part[5]) + (part[3] + part[7]));
  const float inv = rsq(total / (float)(8 * k16) + eps);

  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < kNqMaxWords; ++i) {
    int4 g = make_int4(0, 0, 0, 0);
    if (ok[i]) g = norm_w[at[i] - row];
    const u32 gw[4] = {(u32)g.x, (u32)g.y, (u32)g.z, (u32)g.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      v[i][2 * e] = bf(rne16((v[i][2 * e] * inv) * (bf_lo(gw[e]) + 1.0f)));
      v[i][2 * e + 1] = bf(rne16((v[i][2 * e + 1] * inv) * (bf_hi(gw[e]) + 1.0f)));
    }
    // a slot past the row holds zeros and reads t's first group (in bounds): its quad is all past the row
    macaq_had32(v[i], had_t + 8L * (at[i] - row), tid);
    if (ok[i]) {
#pragma unroll
      for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(v[i][j]));
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o, 32));
  __syncthreads();
  if (lane == 0) part[wave] = amax;
  __syncthreads();
  float m = part[0];
#pragma unroll
  for (int i = 1; i < kNqWaves; ++i) m = fmaxf(m, part[i]);
  const float sc = fmaxf(m * (1.0f / 448.0f), 1.0f / 229376.0f);
  const float rs = 1.0f / sc;
  if (tid == 0) scale_out[b] = sc;

  const long K = 8L * k16;
#pragma unroll
  for (int i = 0; i < kNqMaxWords; ++i) {
    float c[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) c[j] = __builtin_amdgcn_fmed3f(v[i][j] * rs, -448.0f, 448.0f);
    const uint2 q = make_uint2(e4m3x4(c[0], c[1], c[2], c[3]), e4m3x4(c[4], c[5], c[6], c[7]));
    if (ok[i]) *(uint2*)(q_out + (long)b * K + 8L * (at[i] - row)) = q;
  }
}

}  // namespace sar

extern "C" __global__ void __launch_bounds__(256) saber_ar_oneshot_2rank_exact_nq_ssqfma_had32_bf16(
    SAR_NQ_PARAMS, const float* had_t) {
  sar::nq_row_had(SAR_NQ_ARGS, had_t);
}
