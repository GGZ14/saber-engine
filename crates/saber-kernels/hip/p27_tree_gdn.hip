// Saber gated-delta-net kernels, CLEANROOM-2 (CR2-3c): the 27B WY tree verify (up to 10 rows) and the standalone
// advance (spec 2.9). Body in sgdn_tree.h; this file is the instantiation table.
#include "sgdn_tree.h"

#define SGDN_TREE_PARAMS                                                                                         \
  sgdn::cbf q, sgdn::cbf k,                                    \
      sgdn::cbf v, sgdn::cbf av,                               \
      sgdn::cbf bv, long ab_stride, sgdn::cf32 a_log,                    \
      sgdn::cf32 dt_bias, sgdn::wbf state, long st_slot, long st_head,         \
      sgdn::ci32 slot, sgdn::wbf o, sgdn::ci32 cu,                  \
      sgdn::cbf anc, sgdn::cf32 ts_in, sgdn::wf32 ts_out,       \
      long ts_seq, long ts_head, sgdn::ci32 acc, sgdn::cbf zg,             \
      sgdn::cf32 nw, float neps, int nact, int h, int hg, float scale, float sp_thr
#define SGDN_TREE_ARGS                                                                                           \
  sgdn::TreeArgs{q,      k,      v,   av,  bv,     ab_stride, a_log,   dt_bias, state, st_slot,                  \
                 st_head, slot,  o,   cu,  anc,    ts_in,     ts_out,  ts_seq,  ts_head, acc,                    \
                 zg,     nw,     neps, nact, h,    hg,        scale,   sp_thr}

#define SGDN_TREE(TM, NAME, VS, NORM, ADV)                                                                       \
  extern "C" __global__ __launch_bounds__(256 / VS) void NAME(SGDN_TREE_PARAMS) {                               \
    sgdn::tree_update<TM, VS, NORM != 0, ADV != 0>(SGDN_TREE_ARGS);                                              \
  }

#define SGDN_TREE_ADVANCE(TM, NAME)                                                                              \
  extern "C" __global__ __launch_bounds__(256) void NAME(                                                        \
      sgdn::wbf state, long st_slot, long st_head, sgdn::ci32 slot,              \
      sgdn::ci32 slot_out, sgdn::cf32 ts, long ts_seq, long ts_head,                 \
      sgdn::ci32 acc) {                                                                             \
    sgdn::tree_advance_kernel<TM>(state, st_slot, st_head, slot, slot_out, ts, ts_seq, ts_head, acc, blockIdx.x, \
                                  blockIdx.y);                                                                   \
  }

#ifndef SGDN_TREE_ROWS16
SGDN_TREE(10, p27_gdn_tree_f16s_k128_v128_ab1_norm0_vs1_adv0, 1, 0, 0)
SGDN_TREE(10, p27_gdn_tree_f16s_k128_v128_ab1_norm0_vs1_adv1, 1, 0, 1)
SGDN_TREE(10, p27_gdn_tree_f16s_k128_v128_ab1_norm0_vs2_adv0, 2, 0, 0)
SGDN_TREE(10, p27_gdn_tree_f16s_k128_v128_ab1_norm0_vs2_adv1, 2, 0, 1)
SGDN_TREE(10, p27_gdn_tree_f16s_k128_v128_ab1_norm0_vs4_adv0, 4, 0, 0)
SGDN_TREE(10, p27_gdn_tree_f16s_k128_v128_ab1_norm0_vs4_adv1, 4, 0, 1)
SGDN_TREE(10, p27_gdn_tree_f16s_k128_v128_ab1_norm1_vs1_adv0, 1, 1, 0)
SGDN_TREE(10, p27_gdn_tree_f16s_k128_v128_ab1_norm1_vs1_adv1, 1, 1, 1)
SGDN_TREE_ADVANCE(10, p27_gdn_tree_advance_f16s_k128_v128)
#endif
