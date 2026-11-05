// Saber two-rank all-reduce kernels, CLEANROOM-2 (CR2-4c): the exact reduce of one row + the decoder epilogue
// (spec 2.6): residual add, Gemma RMS norm, bf16 round, per-row e4m3 quantisation.
//
// Grid = rows, 256 threads, row b = block b; thread t owns the row's 16-byte words t, t + 256, ... (< k16, at most
// five). Push and hand-off are the exact one-shot's on the row's counter / flag word b. Per element:
//   red = rne16(f32(mine) + f32(peer)); v = bf(red) + bf(res); res_out = rne16(v)
//   ssq: the served chain fma(v, v, ssq) (one rounding) or the control's ssq + v * v (two), word then lane order
//   total = the fixed tree (wave butterflies, then ((W0+W4)+(W2+W6))+((W1+W5)+(W3+W7)))
//   inv = rsq(total / K + eps); nv = bf(rne16((v * inv) * (bf(w) + 1)))
//   sc = max(amax(|nv|) / 448, 1 / (448 * 512)); q = e4m3(med3(nv * (1 / sc), -448, 448))
#pragma once
#include "sar_prims.h"
#include "sar_exact.h"

namespace sar {

enum SsqChain { kSsqTwoRound = 0, kSsqFma = 1 };

constexpr int kNqThreads = 256;
constexpr int kNqWaves = kNqThreads / 32;
constexpr int kNqMaxWords = 5;   // K <= 8 x 256 x 5 = 10,240

__device__ __forceinline__ u32 e4m3x4(float a, float b, float c, float d) {
  int w = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
  w = __builtin_amdgcn_cvt_pk_fp8_f32(c, d, w, true);
  return (u32)w;
}

template <int SSQ, int CHK>
__device__ __forceinline__ void nq_row(const int4* in, int4* peer_scratch, int4* my_scratch, const int4* residual,
                                       const int4* norm_w, unsigned char* q_out, float* scale_out, int4* res_out,
                                       u32* peer_flags, u32* my_flags, u32* seq, int k16, int stride16, float eps,
                                       int drain, int acq) {
  __shared__ u32 bcast;
  __shared__ float part[kNqWaves];
  const int b = (int)blockIdx.x;
  const int tid = (int)threadIdx.x;
  const int lane = tid & 31, wave = tid >> 5;
  const u32 s = take_seq(seq, b, &bcast);
  const long so = (long)(s & 1u) * stride16;
  int4* peer = peer_scratch + so;
  const int4* mine = my_scratch + so;
  const long slot_bytes = (long)stride16 * 16;
  const long row = (long)b * k16;
  unsigned h = 0u;

  // a thread's slots: word tid + 256 i of the row; a slot past the row reads the row's first word (in bounds) and
  // computes with zeros, so every slot runs the same straight-line code and only the stores are guarded
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
    if (ok[i]) {
      peer[at[i]] = x[i];
      if constexpr (CHK != kNoCheck) h ^= chk_word(x[i], (unsigned)at[i]);
    }
  }
  if constexpr (CHK != kNoCheck) chk_send(h, peer, slot_bytes, (unsigned)b, 0u, s);

  drain_and_sync(drain);
  if (tid == 0) {
    const u64 z = signal_wait(peer_flags + b, my_flags + b, s);
    if constexpr (CHK != kNoCheck) chk_after_spin(my_flags + b, s, z, seq + CHK_DIAG);
  }
  __syncthreads();
  acquire_opt(acq);

  // the reduced row + residual (zeros in a slot past the row), res_out, the per-thread sum of squares
  float v[kNqMaxWords][8];
  float ssq = 0.0f;
  h = 0u;
#pragma unroll
  for (int i = 0; i < kNqMaxWords; ++i) {
    int4 p = make_int4(0, 0, 0, 0), r = p;
    if (ok[i]) {
      p = mine[at[i]];
      r = residual[at[i]];
    }
    if constexpr (CHK != kNoCheck) {
      if (ok[i]) h ^= chk_word(p, (unsigned)at[i]);
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
    // a zero slot adds exactly nothing (fma(0, 0, ssq) == ssq, ssq >= +0)
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if constexpr (SSQ == kSsqFma) ssq = __builtin_fmaf(v[i][j], v[i][j], ssq);
      else ssq = ssq + v[i][j] * v[i][j];
    }
  }
  if constexpr (CHK != kNoCheck)
    chk_recv(h, mine, (int)row, (int)row + k16, (unsigned)b, 0u, s, slot_bytes, seq + CHK_DIAG, false);

  // the fixed tree: lane l + o into lane l (o = 16 .. 1) inside each wave, then the eight wave sums
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
      const float n0 = bf(rne16((v[i][2 * e] * inv) * (bf_lo(gw[e]) + 1.0f)));
      const float n1 = bf(rne16((v[i][2 * e + 1] * inv) * (bf_hi(gw[e]) + 1.0f)));
      v[i][2 * e] = n0;
      v[i][2 * e + 1] = n1;
      if (ok[i]) {
        amax = fmaxf(amax, fabsf(n0));
        amax = fmaxf(amax, fabsf(n1));
      }
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o, 32));
  __syncthreads();   // every wave has read the sums
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

#define SAR_NQ_PARAMS                                                                                         \
  const int4 *in, int4 *peer_scratch, int4 *my_scratch, const int4 *residual, const int4 *norm_w,               \
      unsigned char *q_out, float *scale_out, int4 *res_out, unsigned *peer_flags, unsigned *my_flags,           \
      unsigned *seq_ctrs, int k16, int slot_stride16, float eps, int drain, int acq
#define SAR_NQ_ARGS                                                                                           \
  in, peer_scratch, my_scratch, residual, norm_w, q_out, scale_out, res_out, peer_flags, my_flags, seq_ctrs, k16, \
      slot_stride16, eps, drain, acq
