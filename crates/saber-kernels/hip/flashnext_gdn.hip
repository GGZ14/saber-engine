// Saber gated-delta-net kernels, CLEANROOM-2 (CR2-3c): Flash-Next's exact-gate chunk scan (spec 2.8.3), f32 and
// f16 h0 / ht. xg0 weights the state operand as it is read; xg and xg2 (one instantiation under two names) write
// it once into LDS. Body in sgdn_scan.h; this file is the instantiation table.
#include "sgdn_scan.h"

SGDN_SCAN_KERNEL(fn_gdn_chunk_scan_xg0_k128_v128_c64, sgdn::kStateF32, sgdn::kGateExact)
SGDN_SCAN_KERNEL(fn_gdn_chunk_scan_xg0_f16s_k128_v128_c64, sgdn::kStateF16, sgdn::kGateExact)
SGDN_SCAN_KERNEL(fn_gdn_chunk_scan_xg_k128_v128_c64, sgdn::kStateF32, sgdn::kGateExactW)
SGDN_SCAN_KERNEL(fn_gdn_chunk_scan_xg_f16s_k128_v128_c64, sgdn::kStateF16, sgdn::kGateExactW)
SGDN_SCAN_KERNEL(fn_gdn_chunk_scan_xg2_k128_v128_c64, sgdn::kStateF32, sgdn::kGateExactW)
SGDN_SCAN_KERNEL(fn_gdn_chunk_scan_xg2_f16s_k128_v128_c64, sgdn::kStateF16, sgdn::kGateExactW)
