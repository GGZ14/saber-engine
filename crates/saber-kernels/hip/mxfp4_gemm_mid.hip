// SABER-AUTHORED (workstream G6, 2026-09-24) -- NOT sliced from the reference and NOT generated.
// Two MXFP4 W4A8 prefill GEMM bodies for the rungs the layout law's mid band and the 48-wide GDN gate
// open, each the reference's arithmetic per 16x16 output tile byte for byte:
//   * mxfp4_gemm_atiled_mt_tn{TN}_lbk{LBK}_wperm1_t{TM}w{WM}[p]  the A-tiled tile at TM x WM x 16 rows
//     (`mxfp4_atiled_mt.h`), bound by `[[atiled_mt]]` under the `atiled_mid` knob;
//   * mxfp4_gemm_skinny_n{TNS}{t|r}_w{WPB}_u{UNR}_wperm1  the skinny-N body (`mxfp4_skinny.h`), one
//     wave per 16-row fragment, A in fragment order (t) or row-major (r), bound by `[[skinny]]` under
//     the `skinny` knob.
// Its own code object, like mxfp4_gemm_folded_mt: `mxfp4_gemm_atiled` keeps its bytes (and every
// autotune key its digest) while no committed registry row binds these symbols.
//
// Gated 0 bytes differing against the shipped `mxfp4_gemm_folded_tn2_wperm1_epifast1` over the same A
// (tools/capture/g6/unit_inner.sh; k-reversal / no-fold controls failing, the W-byte flip moving).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "mxfp4_common.h"
#include "mxfp4_atiled_wm.h"
#include "mxfp4_atiled_mt.h"
#include "mxfp4_skinny.h"

#define G6_SIG                                                                                     \
  const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,                       \
      const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,               \
      const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K, int pb1, int pb2, \
      long astride
#define G6_ARGS A, W, Ws, Wref, As, C, M, N, K, pb1, pb2, astride

#define G6_ATILED_MT(TN_, LBK_, TM_, WM_, PIPE_, SUF_)                                             \
  extern "C" __global__ __launch_bounds__(WM_ * 64) void                                         \
      mxfp4_gemm_atiled_mt_tn##TN_##_lbk##LBK_##_wperm1_t##TM_##w##WM_##SUF_(G6_SIG) {             \
    saber_mxfp4_atiled_mt_impl<TN_, LBK_, WM_, PIPE_, TM_, 0>(G6_ARGS);                            \
  }

#define G6_SKINNY(TNS_, L_, TILED_, WPB_, UNR_)                                                    \
  extern "C" __global__ __launch_bounds__(WPB_ * 32) void                                        \
      mxfp4_gemm_skinny_n##TNS_##L_##_w##WPB_##_u##UNR_##_wperm1(G6_SIG) {                         \
    saber_mxfp4_skinny_impl<TNS_, TILED_, WPB_, UNR_, 0>(G6_ARGS);                                 \
  }

// The bound cells (rules/mxfp4_gfx1201.toml `[[atiled_mt]]` / `[[skinny]]`).
G6_ATILED_MT(2, 128, 3, 4, false, )
G6_ATILED_MT(2, 128, 4, 4, true, p)
G6_ATILED_MT(2, 64, 5, 4, true, p)
G6_ATILED_MT(2, 64, 5, 2, true, p)
G6_ATILED_MT(2, 64, 6, 4, true, p)
G6_ATILED_MT(2, 64, 6, 2, true, p)
G6_SKINNY(1, t, true, 4, 8)
G6_SKINNY(1, r, false, 4, 8)
