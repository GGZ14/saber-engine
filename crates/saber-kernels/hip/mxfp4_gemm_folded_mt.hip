// SABER-AUTHORED (workstream G5, 2026-09-23) -- NOT sliced from the reference and NOT generated.
// The folded MXFP4 W4A8 prefill GEMM at mid-M tile heights (`mxfp4_folded_mt.h`): the reference's
// arithmetic per 16x16 output tile, byte for byte, with the M tile TM x WM x 16 rows instead of 256.
// Its own code object, like mxfp4_gemm_decode_arnq: the declared `mxfp4_gemm_folded` keeps its
// bytes (and every autotune key its digest) while no committed registry row binds these symbols;
// `rules/mxfp4_gfx1201.toml`'s `[[folded_mt]]` rows bind them when the `folded_mt` knob is on
// (`saber serve --engine-mid-chunk-gemm`).
//
// Symbols: mxfp4_gemm_folded_mt_tn2_wperm1_t{TM}w{WM} (fragment-order W, EPIFAST; the only layout
// the served packs carry). Gated 0 bytes differing against mxfp4_gemm_folded_tn2_wperm1_epifast1
// on the TP=2 shards at M 129..512 incl. ragged tails (tools/capture/g5/unit_inner.sh).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "mxfp4_common.h"
#include "mxfp4_folded_mt.h"

#define G5_FOLDED_MT(TM_, WM_)                                                                     \
  extern "C" __global__ __launch_bounds__(WM_ * 64) void mxfp4_gemm_folded_mt_tn2_wperm1_t##TM_##w##WM_( \
      const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,                   \
      const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,               \
      const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K,                  \
      int pb1, int pb2, long astride) {                                                          \
    saber_mxfp4_folded_mt_impl<2, TM_, WM_, 0>(A, W, Ws, Wref, As, C, M, N, K, pb1, pb2, astride); \
  }

// 320 rows (the p320 rung, N >= 8192), 160 rows (the p320 rung, N < 8192), 192 rows (p192, p384).
G5_FOLDED_MT(5, 4)
G5_FOLDED_MT(5, 2)
G5_FOLDED_MT(3, 4)
