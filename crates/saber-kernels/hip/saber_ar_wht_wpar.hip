// Saber two-rank rotated all-reduce, wave-parallel body (CLEANROOM-2, CR2-4c): the instantiation table of code
// object `saber_ar_wht_wpar` (and, with SAR_WPAR_CHECKED defined, of `saber_ar_wht_wpar_chk`). Body: sar_wpar.h.
// The host routes only messages whose chunks a block fit the LDS stream (SABER_WPAR_CPB_MAX = the host's
// AR_WPAR_MAX_CHUNKS_PER_BLOCK); the kernel traps past it.
#include "sar_wpar.h"

#define SABER_WPAR_CPB_MAX 8

#ifndef SAR_WPAR_CHECKED
extern "C" __global__ void __launch_bounds__(1024) saber_ar_wpar_2rank_wht6_bf16(SAR_ROT_PARAMS) {
  sar::rot_wpar<SABER_WPAR_CPB_MAX, sar::kNoCheck>(SAR_ROT_ARGS);
}
#else
extern "C" __global__ void __launch_bounds__(1024) saber_ar_wpar_2rank_wht6_bf16_chk(SAR_ROT_PARAMS) {
  sar::rot_wpar<SABER_WPAR_CPB_MAX, sar::kChecked>(SAR_ROT_ARGS);
}
#endif
