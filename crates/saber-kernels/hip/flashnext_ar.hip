// Saber two-rank flag-in-data exact all-reduce (CLEANROOM-2, CR2-4c): the instantiation table of code object
// `flashnext_ar` (Flash-Next `--fn-ar ll`, an arm). Body: sar_ll.h. `_wt` = system-scope (write-through) stores
// instead of the write-back; `_chk` = the race-check level; `_chkctl` = the checked body that accepts the first read
// of every word (the data-tag check off: a control the checker must catch). `drain` and `acq` are unused by this form.
#include "sar_ll.h"

extern "C" __global__ void __launch_bounds__(1024) fn_ar_ll_exact_bf16(SAR_LL_PARAMS) {
  (void)drain;
  (void)acq;
  sar::ll_exact<sar::kNoCheck, sar::kLlWriteBack>(SAR_LL_ARGS);
}

extern "C" __global__ void __launch_bounds__(1024) fn_ar_ll_exact_bf16_chk(SAR_LL_PARAMS) {
  (void)drain;
  (void)acq;
  sar::ll_exact<sar::kChecked, sar::kLlWriteBack>(SAR_LL_ARGS);
}

extern "C" __global__ void __launch_bounds__(1024) fn_ar_ll_exact_bf16_chkctl(SAR_LL_PARAMS) {
  (void)drain;
  (void)acq;
  sar::ll_exact<sar::kCheckedNoWait, sar::kLlWriteBack>(SAR_LL_ARGS);
}

extern "C" __global__ void __launch_bounds__(1024) fn_ar_ll_exact_bf16_wt(SAR_LL_PARAMS) {
  (void)drain;
  (void)acq;
  sar::ll_exact<sar::kNoCheck, sar::kLlWriteThrough>(SAR_LL_ARGS);
}
