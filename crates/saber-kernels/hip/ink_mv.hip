// Inkling 2 (inkling2_loop) multi-sequence verify glue, INK20 (docs/INKLING2.md section 9 INK20).
// Saber's own. Its own object so ink_loop / ink_draft keep their bytes.
//
// A multi-sequence verify chunk = B sequences x W chained rows each (row t = sequence t / W, its
// row t % W), every sequence at its own slot (row_seq[t], data) and position (row_pos[t], data).
// INK9's chain runs it; the three per-sequence pieces it needs are here:
//
//   ink_mv_sconv   ink_loop's ink_sconv_resid for B sequences at once: row t's conv reads
//                  x_{t-3..t} of ITS sequence (rows b W .. b W + W - 1), a lag before the
//                  sequence's first row reads its state row 3 + s at slot row_seq[t]; the same
//                  expression in the same order (acc = 0 + v0 w0 + v1 w1 + v2 w2 + v3 w3, then
//                  (acc + x_t) / sqrt 5, + resid). It writes NO state: a verify's commit owns it.
//   ink_mv_hist    ink_draft's ink_dr_hist per sequence: entry b = [state of slot row_seq[b W]
//                  (3 rows) ; x rows b W .. b W + W - 1] at hist + b (3 + W) C.
//   ink_mv_commit  ink_draft's ink_dr_commit per sequence: state(slot row_seq[b W]) = entry b's
//                  rows keep_b .. keep_b + 2, keep_b = keeps[b] (1..W).
//
// Built with -ffp-contract=off (saber-kbuild's `ink_` row).
#include <hip/hip_runtime.h>

__device__ __forceinline__ float mv_bf(unsigned short u) {
  return __builtin_bit_cast(float, ((unsigned)u) << 16);
}

// grid (ceil(C / 256), B * W)
extern "C" __global__ __launch_bounds__(256) void ink_mv_sconv(
    const float* __restrict__ x, int ld_x, const unsigned short* __restrict__ w,   // [C, 4]
    const float* __restrict__ st,   // [slots][3][C], oldest first
    const int* __restrict__ row_seq, const float* __restrict__ resid, int ld_r,
    float* __restrict__ o, int ld_o, int W, int C) {
  const int c = blockIdx.x * 256 + threadIdx.x;
  if (c >= C) return;
  const int t = blockIdx.y;
  const int j = t % W, r0 = t - j;
  const float* sb = st + (long)row_seq[t] * 3 * C;
  float wt[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) wt[i] = mv_bf(w[(long)c * 4 + i]);
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int s = j - 3 + i;
    const float v = s >= 0 ? x[(long)(r0 + s) * ld_x + c] : sb[(long)(3 + s) * C + c];
    acc = acc + v * wt[i];
  }
  float y = (acc + x[(long)t * ld_x + c]) / 2.2360679774997898f;
  if (resid) y = y + resid[(long)t * ld_r + c];
  o[(long)t * ld_o + c] = y;
}

// grid (ceil(C / 256), 3 + W, B)
extern "C" __global__ __launch_bounds__(256) void ink_mv_hist(const float* __restrict__ st,
                                                              const int* __restrict__ row_seq,
                                                              const float* __restrict__ x, int ldx,
                                                              int C, int W, float* __restrict__ hist) {
  const int c = blockIdx.x * 256 + threadIdx.x, r = blockIdx.y, b = blockIdx.z;
  if (c >= C) return;
  float* h = hist + (long)b * (3 + W) * C;
  const float* sb = st + (long)row_seq[b * W] * 3 * C;
  h[(long)r * C + c] = r < 3 ? sb[(long)r * C + c] : x[(long)(b * W + r - 3) * ldx + c];
}

// grid (ceil(maxC / 256), n, B): entry e = (hist, state base, C) as three u64 words
extern "C" __global__ __launch_bounds__(256) void ink_mv_commit(
    const unsigned long long* __restrict__ tab, const int* __restrict__ keeps,
    const int* __restrict__ row_seq, int W) {
  const unsigned long long* e = tab + 3 * blockIdx.y;
  const int C = (int)e[2], b = blockIdx.z;
  const int c = blockIdx.x * 256 + threadIdx.x;
  if (c >= C) return;
  const float* hist = (const float*)e[0] + (long)b * (3 + W) * C;
  float* dst = (float*)e[1] + (long)row_seq[b * W] * 3 * C;
  const int keep = keeps[b];
  for (int r = 0; r < 3; ++r) dst[(long)r * C + c] = hist[(long)(keep + r) * C + c];
}
