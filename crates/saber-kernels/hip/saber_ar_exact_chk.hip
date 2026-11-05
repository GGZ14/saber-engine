// Saber two-rank exact all-reduce, race-check twins (CLEANROOM-2, CR2-4c): the instantiation table of code object
// `saber_ar_exact_chk`. The served bodies (sar_exact.h, sar_nq.h) at the checked level: the pushes, hand-off,
// reduce and epilogue are the served ones; the check words, flag audit and probe re-read are `ar_race_check.h`.
#include "sar_exact.h"
#include "sar_nq.h"

extern "C" __global__ void __launch_bounds__(1024) saber_ar_oneshot_2rank_exact_bf16_chk(SAR_EXACT_PARAMS) {
  sar::exact_oneshot<sar::kBf16, sar::kPerBlock, sar::kChecked>(SAR_EXACT_ARGS);
}

extern "C" __global__ void __launch_bounds__(256) saber_ar_oneshot_2rank_exact_nq_ssqfma_bf16_chk(SAR_NQ_PARAMS) {
  sar::nq_row<sar::kSsqFma, sar::kChecked>(SAR_NQ_ARGS);
}
