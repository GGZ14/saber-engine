// Saber gated-delta-net kernels, CLEANROOM-2 (CR2-3c): the 27B tree verify's conv, one wave walking a unit's
// rows serially (spec 2.3.3). Body in sgdn_conv.h; this file is the instantiation table.
#include "sgdn_conv.h"

#define SGDN_TREE_CONV_PARAMS                                                                                    \
  sgdn::cbf x, long xpitch, sgdn::cbf wgt,                      \
      sgdn::cbf bias, sgdn::wbf cstate, long cs_seq, long cs_dim,     \
      long cs_tok, int state_len_max, sgdn::ci32 cache_idx, long ci_stride,                         \
      sgdn::ci32 hsrc, sgdn::ci32 tsrc, sgdn::wbf q,                \
      sgdn::wbf k, sgdn::wbf v, sgdn::ci32 cu, int h, int hg,  \
      int maxq
#define SGDN_TREE_CONV_ARGS                                                                                      \
  sgdn::TreeConvArgs{x,         xpitch, wgt, bias, cstate, cs_seq, cs_dim, cs_tok, state_len_max, cache_idx,     \
                     ci_stride, hsrc,   tsrc, q,   k,      v,      cu,     h,      hg,            maxq}

#define SGDN_TREE_CONV(HPB)                                                                                      \
  extern "C" __global__ __launch_bounds__(32 * HPB) void p27_gdn_conv_tree_w4_h128_hpb##HPB(                     \
      SGDN_TREE_CONV_PARAMS) {                                                                                   \
    sgdn::tree_conv<0>(SGDN_TREE_CONV_ARGS, blockIdx.x, blockIdx.y * HPB + (threadIdx.x >> 5), 0,               \
                       threadIdx.x & 31);                                                                        \
  }

SGDN_TREE_CONV(1)
SGDN_TREE_CONV(2)
SGDN_TREE_CONV(4)
SGDN_TREE_CONV(8)
