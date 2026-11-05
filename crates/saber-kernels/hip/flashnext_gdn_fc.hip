// Saber gated-delta-net kernels, CLEANROOM-2 (CR2-3c): Flash-Next's width-1 verify class (spec 2.4.4, 2.6.2) --
// the f16-carry recurrent / fused updates (each row's state rounded through f16 before the next, so a w-row
// window equals w single-token steps), and the same with the gated norm + fp8 quant tail after each (sequence,
// head) (FN86, `_rc` = the reciprocal-scale arm). Bodies in sgdn_rec.h; this file names the symbols.
#include "sgdn_rec.h"

extern "C" __global__ __launch_bounds__(256) void fn_gdn_recurrent_update_fc_f16s_k128_v128_ab1_norm0_vs1(
    SGDN_REC_PARAMS) {
  sgdn::recurrent_rows<sgdn::kStateF16, true, false, 1, true>(SGDN_REC_ARGS, blockIdx.x, blockIdx.y, 0, nullptr);
}

extern "C" __global__ __launch_bounds__(256) void fn_gdn_fused_update_fc_f16s_w4k128v128_ab1(SGDN_FUSED_PARAMS) {
  sgdn::fused_update<sgdn::kStateF16, true, true>(SGDN_FUSED_ARGS);
}

#define SGDN_FN86_REC(NAME, MODE)                                                                                \
  extern "C" __global__ __launch_bounds__(256) void NAME(SGDN_REC_PARAMS, SGDN_NQ8_PARAMS) {                    \
    sgdn::recurrent_nq8<sgdn::kStateF16, true, true, MODE>(SGDN_REC_ARGS, SGDN_NQ8_ARGS, blockIdx.x, blockIdx.y); \
  }

#define SGDN_FN86_FUSED(NAME, MODE)                                                                              \
  extern "C" __global__ __launch_bounds__(256) void NAME(SGDN_FUSED_PARAMS, SGDN_NQ8_PARAMS) {                  \
    const sgdn::Nq8Args tail = SGDN_NQ8_ARGS;                                                                    \
    sgdn::fused_update<sgdn::kStateF16, true, true, MODE>(SGDN_FUSED_ARGS, &tail);                               \
  }

SGDN_FN86_REC(fn86_gdn_recurrent_nq8_fc_f16s_k128_v128_ab1_vs1, 1)
SGDN_FN86_REC(fn86_gdn_recurrent_nq8_fc_f16s_k128_v128_ab1_vs1_rc, 2)
SGDN_FN86_FUSED(fn86_gdn_fused_update_nq8_fc_f16s_w4k128v128_ab1, 1)
SGDN_FN86_FUSED(fn86_gdn_fused_update_nq8_fc_f16s_w4k128v128_ab1_rc, 2)
