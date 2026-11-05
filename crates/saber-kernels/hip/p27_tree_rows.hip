// Saber gated-delta-net kernels, CLEANROOM-2 (CR2-3c): the 27B tree verify's conv with the rows in parallel (one
// workgroup a unit, TM waves, rows w, w + TM, ...; one barrier; the cache rewrite split over the waves), its own
// object so p27_tree_conv keeps its bytes (spec 2.3.3). Body in sgdn_conv.h; this file is the instantiation table.
#include "sgdn_conv.h"

#define SGDN_TREE_ROWS(TM)                                                                                       \
  extern "C" __global__ __launch_bounds__(32 * TM) void p27_gdn_conv_tree_rows_w4_h128_t##TM(                    \
      sgdn::cbf x, long xpitch, sgdn::cbf wgt,                  \
      sgdn::cbf bias, sgdn::wbf cstate, long cs_seq, long cs_dim,     \
      long cs_tok, int state_len_max, sgdn::ci32 cache_idx, long ci_stride,                         \
      sgdn::ci32 hsrc, sgdn::ci32 tsrc, sgdn::wbf q,                \
      sgdn::wbf k, sgdn::wbf v, sgdn::ci32 cu, int h, int hg,  \
      int maxq) {                                                                                                \
    const sgdn::TreeConvArgs a{x,         xpitch, wgt,  bias, cstate, cs_seq, cs_dim, cs_tok, state_len_max,     \
                               cache_idx, ci_stride, hsrc, tsrc, q,   k,      v,      cu,     h,      hg, maxq}; \
    sgdn::tree_conv<TM>(a, blockIdx.x, blockIdx.y, threadIdx.x >> 5, threadIdx.x & 31);                          \
  }

SGDN_TREE_ROWS(10)
SGDN_TREE_ROWS(5)
