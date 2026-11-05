// Saber gated-delta-net kernels, CLEANROOM-2 (CR2-3c): the one-launch fused decode, f16 state (spec 2.4.3): conv
// barrier, recurrence (VS 1), pad rows.
// Body in sgdn_rec.h; this file names the symbols (names, parameter lists, launch bounds = the ABI).
#include "sgdn_rec.h"

SGDN_FUSED_KERNEL(saber_gdn_fused_update_f16s_w4k128v128_ab0, sgdn::kStateF16, 0)
SGDN_FUSED_KERNEL(saber_gdn_fused_update_f16s_w4k128v128_ab1, sgdn::kStateF16, 1)
