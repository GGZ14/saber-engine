// Saber two-rank rotated all-reduce (CLEANROOM-2, CR2-4c): the instantiation table of code object `saber_ar_wht`.
// Body: sar_rot.h (the one-shot). bits 4 / 6 x bf16 / f16 in and out. Built -ffp-contract=off.
#include "sar_rot.h"

extern "C" __global__ void __launch_bounds__(1024) saber_ar_oneshot_2rank_wht4_bf16(SAR_ROT_PARAMS) {
  sar::rot_oneshot<4, sar::kBf16, sar::kPerBlock, sar::kNoCheck>(SAR_ROT_ARGS);
}

extern "C" __global__ void __launch_bounds__(1024) saber_ar_oneshot_2rank_wht4_fp16(SAR_ROT_PARAMS) {
  sar::rot_oneshot<4, sar::kF16, sar::kPerBlock, sar::kNoCheck>(SAR_ROT_ARGS);
}

extern "C" __global__ void __launch_bounds__(1024) saber_ar_oneshot_2rank_wht6_bf16(SAR_ROT_PARAMS) {
  sar::rot_oneshot<6, sar::kBf16, sar::kPerBlock, sar::kNoCheck>(SAR_ROT_ARGS);
}

extern "C" __global__ void __launch_bounds__(1024) saber_ar_oneshot_2rank_wht6_fp16(SAR_ROT_PARAMS) {
  sar::rot_oneshot<6, sar::kF16, sar::kPerBlock, sar::kNoCheck>(SAR_ROT_ARGS);
}
