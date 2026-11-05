// Saber gated-delta-net kernels, CLEANROOM-2 (CR2-3c): the decode recurrence, f32 state (spec 2.4).
// Body in sgdn_rec.h; this file names the symbols (names, parameter lists, launch bounds = the ABI).
#include "sgdn_rec.h"

SGDN_REC_KERNEL(saber_gdn_recurrent_update_k128_v128_ab0_norm0_vs1, sgdn::kStateF32, 0, 0, 1)
SGDN_REC_KERNEL(saber_gdn_recurrent_update_k128_v128_ab0_norm0_vs2, sgdn::kStateF32, 0, 0, 2)
SGDN_REC_KERNEL(saber_gdn_recurrent_update_k128_v128_ab0_norm0_vs4, sgdn::kStateF32, 0, 0, 4)
SGDN_REC_KERNEL(saber_gdn_recurrent_update_k128_v128_ab0_norm1_vs1, sgdn::kStateF32, 0, 1, 1)
SGDN_REC_KERNEL(saber_gdn_recurrent_update_k128_v128_ab1_norm0_vs1, sgdn::kStateF32, 1, 0, 1)
SGDN_REC_KERNEL(saber_gdn_recurrent_update_k128_v128_ab1_norm0_vs2, sgdn::kStateF32, 1, 0, 2)
SGDN_REC_KERNEL(saber_gdn_recurrent_update_k128_v128_ab1_norm0_vs4, sgdn::kStateF32, 1, 0, 4)
SGDN_REC_KERNEL(saber_gdn_recurrent_update_k128_v128_ab1_norm1_vs1, sgdn::kStateF32, 1, 1, 1)
