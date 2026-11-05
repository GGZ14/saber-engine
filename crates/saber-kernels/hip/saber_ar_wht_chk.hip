// Saber two-rank rotated all-reduce, race-check twins (CLEANROOM-2, CR2-4c): the instantiation table of code
// object `saber_ar_wht_chk`. The served one-shot (sar_rot.h) at the checked level.
#include "sar_rot.h"

extern "C" __global__ void __launch_bounds__(1024) saber_ar_oneshot_2rank_wht4_bf16_chk(SAR_ROT_PARAMS) {
  sar::rot_oneshot<4, sar::kBf16, sar::kPerBlock, sar::kChecked>(SAR_ROT_ARGS);
}

extern "C" __global__ void __launch_bounds__(1024) saber_ar_oneshot_2rank_wht6_bf16_chk(SAR_ROT_PARAMS) {
  sar::rot_oneshot<6, sar::kBf16, sar::kPerBlock, sar::kChecked>(SAR_ROT_ARGS);
}
