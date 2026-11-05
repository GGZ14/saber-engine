// Flash-Next (FN119c, `--fn-device-step ahead` at b > 1): the drafter chain's inputs COMPOSED per
// sequence on the card from the walk's accepted counts. Its own code object (`flashnext_ahead`
// keeps its bytes).
//
// The host builds the chain inputs once per accepted count a (every sequence at a: variant a) and
// learns, by building each sequence alone at every other count, which 4-byte words belong to which
// sequence (a word two sequences move refuses the step's round-ahead); the card then writes
//
//   fn119b_copy     grid (ceil(n16 / 256)), block 256: variant 0 -> the chain's input region
//   fn119b_compose  grid (nruns), block 256: run r = (sequence s, first word, words): the words of
//                   variant acc_s (acc_s = the walk's result block, sequence s) over the copy
//
// == the inputs the host builds for the walk's counts, word for word (the commit re-builds them on
// the host and compares: a difference fails the boot).
#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ __launch_bounds__(256) void fn119b_copy(const uint4* __restrict__ src,
                                                             uint4* __restrict__ dst, int n16) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (i < n16) dst[i] = src[i];
}

extern "C" __global__ __launch_bounds__(256) void fn119b_compose(
    const uint32_t* __restrict__ out, const uint32_t* __restrict__ var, uint32_t* __restrict__ dst,
    const uint32_t* __restrict__ runs, int stride_words) {
  const uint32_t* run = runs + (size_t)blockIdx.x * 3;
  const uint32_t s = run[0], w0 = run[1], nw = run[2];
  const uint32_t acc = out[(size_t)s * 16 + 1];
  const uint32_t* src = var + (size_t)acc * (size_t)stride_words;
  for (uint32_t i = threadIdx.x; i < nw; i += 256) dst[w0 + i] = src[w0 + i];
}
