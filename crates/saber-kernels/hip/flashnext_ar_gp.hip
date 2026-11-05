// Saber two-rank all-reduce, global-parity forms (CLEANROOM-2, CR2-4c): the instantiation table of code object
// `flashnext_ar_gp` (Flash-Next's `--fn-ar gp`, the default). The one-shot bodies of sar_rot.h / sar_exact.h with
// the global-parity line: every one of the 96 counters (ar_seq / ar_flags words 128..223, shared by the two forms)
// advances once a call whatever the grid, so the slot is the call's parity. `_chk` = the race-check level + the
// sender's s in check-row piece 1; `_chkctl` = the checked body with the receiver's wait removed (a control the
// checker must catch). The wht6 body is placed first (code placement of the large body: spec 5.3).
#include "sar_rot.h"
#include "sar_exact.h"

extern "C" __global__ void __launch_bounds__(1024) saber_ar_oneshot_2rank_wht6_bf16_gp(SAR_ROT_PARAMS) {
  sar::rot_oneshot<6, sar::kBf16, sar::kGlobal, sar::kNoCheck>(SAR_ROT_ARGS);
}

extern "C" __global__ void __launch_bounds__(1024) saber_ar_oneshot_2rank_wht6_bf16_gp_chk(SAR_ROT_PARAMS) {
  sar::rot_oneshot<6, sar::kBf16, sar::kGlobal, sar::kChecked>(SAR_ROT_ARGS);
}

extern "C" __global__ void __launch_bounds__(1024) saber_ar_oneshot_2rank_exact_bf16_gp(SAR_EXACT_PARAMS) {
  sar::exact_oneshot<sar::kBf16, sar::kGlobal, sar::kNoCheck>(SAR_EXACT_ARGS);
}

extern "C" __global__ void __launch_bounds__(1024) saber_ar_oneshot_2rank_exact_bf16_gp_chk(SAR_EXACT_PARAMS) {
  sar::exact_oneshot<sar::kBf16, sar::kGlobal, sar::kChecked>(SAR_EXACT_ARGS);
}

extern "C" __global__ void __launch_bounds__(1024) saber_ar_oneshot_2rank_exact_bf16_gp_chkctl(SAR_EXACT_PARAMS) {
  sar::exact_oneshot<sar::kBf16, sar::kGlobal, sar::kCheckedNoWait>(SAR_EXACT_ARGS);
}
