// Saber gated-delta-net kernels, CLEANROOM-2 (CR2-3c): the standalone gated RMS norm (spec 2.6.1).
// One wave a row of 128 channels, eight rows a workgroup; lane l owns channels 4l .. 4l+3.
//   ss = sum of squares (4 terms from 0, then the full-wave butterfly); sc = R(ss * 2^-7 + eps);
//   out = away16(((x * sc) * w) * act(z)), act = z * sigmoid(z) ("silu") or sigmoid(z); w f32, used as is.
#include "sgdn_prims.h"

namespace sgdn {

template <bool SILU>
__device__ __forceinline__ void gated_rmsnorm(const u16* __restrict__ x, const u16* __restrict__ z,
                                              sgdn::cf32 w, u16* __restrict__ o, long rows,
                                              long xrow, long zrow, long orow, float eps) {
  const int lane = threadIdx.x & 31;
  // the row is wave-uniform: held in SGPRs, every address is a scalar base + the lane's offset
  const long row = (long)blockIdx.x * 8 + __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  if (row >= rows) return;
  const float4 wr = *(const float4*)(w + 4 * lane);
  const u32x2 xr = *(const u32x2*)(x + row * xrow + 4 * lane);
  const u32x2 zr = *(const u32x2*)(z + row * zrow + 4 * lane);
  const float xv[4] = {bf_lo(xr.x), bf_hi(xr.x), bf_lo(xr.y), bf_hi(xr.y)};
  const float zv[4] = {bf_lo(zr.x), bf_hi(zr.x), bf_lo(zr.y), bf_hi(zr.y)};
  const float wv[4] = {wr.x, wr.y, wr.z, wr.w};
  float ss = 0.0f;
#pragma unroll
  for (int c = 0; c < 4; ++c) ss = ss + xv[c] * xv[c];
  // the gate does not depend on the row sum: it is issued under the butterfly's LDS round trips
  float act[4];
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const float sg = 1.0f / (1.0f + E(-zv[c]));
    act[c] = SILU ? zv[c] * sg : sg;
  }
  ss = wave_sum(ss);
  const float sc = R(ss * 0.0078125f + eps);
  float out[4];
#pragma unroll
  for (int c = 0; c < 4; ++c) out[c] = ((xv[c] * sc) * wv[c]) * act[c];
  *(u32x2*)(o + row * orow + 4 * lane) = u32x2{away16x2(out[0], out[1]), away16x2(out[2], out[3])};
}

}  // namespace sgdn

#define SGDN_NORM_PARAMS                                                                                 \
  sgdn::cbf x, sgdn::cbf z, sgdn::cf32 w, \
      sgdn::wbf o, long rows, long xrow, long zrow, long orow, float eps

extern "C" __global__ __launch_bounds__(256) void saber_gdn_gated_rmsnorm_h128_silu(SGDN_NORM_PARAMS) {
  sgdn::gated_rmsnorm<true>(x, z, w, o, rows, xrow, zrow, orow, eps);
}

extern "C" __global__ __launch_bounds__(256) void saber_gdn_gated_rmsnorm_h128_sigmoid(SGDN_NORM_PARAMS) {
  sgdn::gated_rmsnorm<false>(x, z, w, o, rows, xrow, zrow, orow, eps);
}
