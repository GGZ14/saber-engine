// Saber two-rank exact all-reduce (CLEANROOM-2, CR2-4c): the instantiation table of code object `saber_ar_exact`.
// Bodies: sar_exact.h (the one-shot), sar_nq.h (one row + the decoder epilogue). Built -ffp-contract=off.
//
// Includable as text (spec 1.5.1): another translation unit may paste this file in with __global__ / __launch_bounds__ /
// blockIdx / gridDim / blockDim / __syncthreads remapped and calls saber_ar_oneshot_2rank_exact_bf16 as a device
// function with this argument list.
#include "sar_exact.h"
#include "sar_nq.h"

extern "C" __global__ void __launch_bounds__(1024) saber_ar_oneshot_2rank_exact_bf16(SAR_EXACT_PARAMS) {
  sar::exact_oneshot<sar::kBf16, sar::kPerBlock, sar::kNoCheck>(SAR_EXACT_ARGS);
}

extern "C" __global__ void __launch_bounds__(1024) saber_ar_oneshot_2rank_exact_fp16(SAR_EXACT_PARAMS) {
  sar::exact_oneshot<sar::kF16, sar::kPerBlock, sar::kNoCheck>(SAR_EXACT_ARGS);
}

extern "C" __global__ void __launch_bounds__(1024) saber_ar_oneshot_2rank_exact_fp32(SAR_EXACT_PARAMS) {
  sar::exact_oneshot<sar::kF32, sar::kPerBlock, sar::kNoCheck>(SAR_EXACT_ARGS);
}

// the control: the sum of squares with two roundings a term
extern "C" __global__ void __launch_bounds__(256) saber_ar_oneshot_2rank_exact_nq_bf16(SAR_NQ_PARAMS) {
  sar::nq_row<sar::kSsqTwoRound, sar::kNoCheck>(SAR_NQ_ARGS);
}

// served: the sum of squares as an fma chain
extern "C" __global__ void __launch_bounds__(256) saber_ar_oneshot_2rank_exact_nq_ssqfma_bf16(SAR_NQ_PARAMS) {
  sar::nq_row<sar::kSsqFma, sar::kNoCheck>(SAR_NQ_ARGS);
}
