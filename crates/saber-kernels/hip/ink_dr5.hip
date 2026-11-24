// INK22: the batched drafter's plain-step append as fixed-address work (docs/INKLING2.md section 9
// INK22). Saber's own; its own object so ink_draft / ink_dr2 / ink_dr3 / ink_dr4 keep their bytes.
// A plain decode step of B rows appends each row's context K / V (+ the head's f_rnn) to ITS slot at
// ITS position: the per-row arithmetic runs on rows 0..B of fixed scratch (the parents' bodies, one
// launch for all rows), then this kernel moves row b to its slot's row -- a copy, no arithmetic.
//
//   ink_dr5_scatter   dst[slots[b] * sstride + pos[b] * pstride + i] = src[b * ld + i], i < n;
//                     slots[b] < 0 = row b is skipped (a pad row / a context not in sync)
#include <hip/hip_runtime.h>

// grid (B, ceil(n / 256)), 256 lanes
extern "C" __global__ __launch_bounds__(256) void ink_dr5_scatter(const float* __restrict__ src, int ld,
                                                                  float* __restrict__ dst, long sstride,
                                                                  long pstride, int n,
                                                                  const int* __restrict__ slots,
                                                                  const int* __restrict__ pos) {
  const int b = blockIdx.x;
  const int s = slots[b];
  if (s < 0) return;
  const int i = blockIdx.y * 256 + threadIdx.x;
  if (i >= n) return;
  dst[(long)s * sstride + (long)pos[b] * pstride + i] = src[(long)b * ld + i];
}
