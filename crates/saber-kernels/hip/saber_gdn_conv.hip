// Saber gated-delta-net kernels, CLEANROOM-2 (CR2-3c): the conv object -- prefill conv + gates (conv_prep) and
// the decode conv (conv_update). Bodies in sgdn_conv.h; this file is the instantiation table (names, parameter
// lists and launch bounds are the ABI).
#include "sgdn_conv.h"

#define SGDN_PREP_PARAMS                                                                                         \
  sgdn::cbf x, long xpitch, sgdn::cbf wgt,                      \
      sgdn::cbf bias, sgdn::wbf cstate, long cs_seq, long cs_dim,     \
      long cs_tok, sgdn::ci32 cache_idx, long ci_stride, sgdn::cb8 has_init,         \
      sgdn::cany av, sgdn::cany bv, long ab_stride, sgdn::cf32 a_log,  \
      sgdn::cf32 dt_bias, sgdn::wbf q, sgdn::wbf k,          \
      sgdn::wbf v, sgdn::wf32 g, sgdn::wf32 beta, sgdn::ci32 cu, \
      int n_seqs, int h, int hg, float sp_thr

#define SGDN_PREP(HPB, AB)                                                                                        \
  extern "C" __global__ __launch_bounds__(32 * HPB) void saber_gdn_conv_prep_w4_h128_hpb##HPB##_ab##AB(          \
      SGDN_PREP_PARAMS) {                                                                                         \
    const sgdn::PrepArgs a{x,  xpitch, wgt,   bias,  cstate, cs_seq, cs_dim, cs_tok, cache_idx, ci_stride,      \
                           has_init, av, bv, ab_stride, a_log, dt_bias, q, k, v, g, beta, cu, n_seqs, h, hg,     \
                           sp_thr};                                                                               \
    sgdn::conv_prep<HPB, AB != 0>(a);                                                                             \
  }

SGDN_PREP(1, 0)
SGDN_PREP(1, 1)
SGDN_PREP(2, 0)
SGDN_PREP(2, 1)
SGDN_PREP(4, 0)
SGDN_PREP(4, 1)
SGDN_PREP(8, 0)
SGDN_PREP(8, 1)

#define SGDN_CONVUPD(HPB)                                                                                          \
  extern "C" __global__ __launch_bounds__(32 * HPB) void saber_gdn_conv_update_w4_h128_hpb##HPB(                  \
      sgdn::cbf x, long xpitch, sgdn::cbf wgt,                  \
      sgdn::cbf bias, sgdn::wbf cstate, long cs_seq, long cs_dim,     \
      long cs_tok, int state_len_max, sgdn::ci32 cache_idx, long ci_stride,                         \
      sgdn::ci32 naccept, sgdn::wbf q, sgdn::wbf k,            \
      sgdn::wbf v, sgdn::ci32 cu, int h, int hg, int maxq) {                      \
    sgdn::conv_update_unit(x, xpitch, wgt, bias, cstate, cs_seq, cs_dim, cs_tok, state_len_max, cache_idx,       \
                           ci_stride, naccept, q, k, v, cu, h, hg, maxq, blockIdx.x,                              \
                           blockIdx.y * HPB + (threadIdx.x >> 5), threadIdx.x & 31);                              \
  }

SGDN_CONVUPD(1)
SGDN_CONVUPD(2)
SGDN_CONVUPD(4)
SGDN_CONVUPD(8)
