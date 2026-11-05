// Saber gated-delta-net kernels, CLEANROOM-2 (CR2-3c): the fused decode (f16 state) with a ticketed gated-norm +
// fp8 quant tail over the step's rows (the 27B's fusion row 5, spec 2.6.3).
//
// After the fused update's three phases every workgroup takes one ticket from a monotonic 64-bit counter (launch c
// holds tickets [c nt, (c + 1) nt), nt = the grid); the last `rows` ticket holders of a launch each own one row:
// they wait until every ticket of the launch is taken (so every o row is final), then normalise and quantise the
// whole row: per 8-element group the sum of squares (explicit fmas), a 16-lane butterfly, the silu-by-division
// gate, one amax and one scale for the row, e4m3 by the reciprocal scale.
#include "sgdn_rec.h"

namespace sgdn {

struct FnqTail {
  const u16* z; long zs; const u16* w; unsigned char* q; float* scale; int n; float eps;
  unsigned long long* tiles; int ep_rows;
};

__device__ __forceinline__ void fnq_tail(const u16* __restrict__ o, const FnqTail& t) {
  __shared__ int s_row;
  __shared__ float s_wmax[8];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    const unsigned long long nt = gridDim.x;
    const unsigned long long tk = atomicAdd(t.tiles, 1ull);
    const unsigned long long i = tk % nt;
    const unsigned long long rows = (unsigned long long)min(t.ep_rows, (int)nt);
    int row = -1;
    if (i + rows >= nt) {
      row = (int)(i + rows - nt);
      const unsigned long long target = (tk / nt + 1) * nt;
      for (unsigned long long it = 0; it < 4000000000ull; ++it)
        if (__hip_atomic_load(t.tiles, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) >= target) break;
    }
    s_row = row;
  }
  __syncthreads();
  const int row = s_row;
  if (row < 0) return;
  __threadfence();

  constexpr int kMaxGroups = 5;   // N <= 10,240: 1,280 groups of 8 over 256 threads
  float nv[kMaxGroups][8];
  float amax = 0.0f;
  const long rbase = (long)row * t.n;
#pragma unroll
  for (int i = 0; i < kMaxGroups; ++i) {
    const int g = tid + 256 * i;
    if (8 * g < t.n) {
      const u32x4 ow = *(const u32x4*)(o + rbase + 8 * g);
      const u32x4 zw = *(const u32x4*)(t.z + (long)row * t.zs + 8 * g);
      const u32x4 ww = *(const u32x4*)(t.w + 8 * (g & 15));
      float s[8], z[8], w[8];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        s[2 * e] = bf_lo(ow[e]); s[2 * e + 1] = bf_hi(ow[e]);
        z[2 * e] = bf_lo(zw[e]); z[2 * e + 1] = bf_hi(zw[e]);
        w[2 * e] = bf_lo(ww[e]); w[2 * e + 1] = bf_hi(ww[e]);
      }
      const float ssq = half_wave_sum(sumsq8_fma(s));
      const float inv = R(fmaf(ssq, 0.0078125f, t.eps));
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        nv[i][e] = bf(rne16(((s[e] * inv) * w[e]) * (z[e] / (1.0f + expf(-z[e])))));
        amax = fmaxf(amax, fabsf(nv[i][e]));
      }
    }
  }
  // block amax (max is order-free)
#pragma unroll
  for (int d = 16; d >= 1; d >>= 1) amax = fmaxf(amax, __shfl_xor(amax, d, 32));
  if (lane == 0) s_wmax[wave] = amax;
  __syncthreads();
  amax = s_wmax[0];
#pragma unroll
  for (int w = 1; w < 8; ++w) amax = fmaxf(amax, s_wmax[w]);
  const float sc = fmaxf(amax * (1.0f / 448.0f), 1.0f / (448.0f * 512.0f));
  const float rs = 1.0f / sc;
#pragma unroll
  for (int i = 0; i < kMaxGroups; ++i) {
    const int g = tid + 256 * i;
    if (8 * g < t.n) {
      float qv[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) qv[e] = clamp448(nv[i][e] * rs);
      *(u32x2*)(t.q + rbase + 8 * g) = e4m3x8(qv);
    }
  }
  if (tid == 0) t.scale[row] = sc;
}

}  // namespace sgdn

extern "C" __global__ __launch_bounds__(256) void saber_gdn_fnq_f16s_w4k128v128_ab1(
    SGDN_FUSED_PARAMS, sgdn::cbf nq_z, long nq_zs, sgdn::cbf nq_w, sgdn::wb8 nq_q, sgdn::wf32 nq_scale, int nq_n,
    float nq_eps, sgdn::wu64 tiles, int ep_rows) {
  sgdn::fused_update<sgdn::kStateF16, true, false>(SGDN_FUSED_ARGS);
  sgdn::fnq_tail(o, sgdn::FnqTail{nq_z, nq_zs, nq_w, nq_q, nq_scale, nq_n, nq_eps, tiles, ep_rows});
}
