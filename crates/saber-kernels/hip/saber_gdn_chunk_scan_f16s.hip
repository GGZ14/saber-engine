// Saber gated-delta-net kernels, CLEANROOM-2 (CR2-3c): the certified chunk scan, f16 h0 / ht (spec 2.8.2, 2.1.6).
// Body in sgdn_scan.h; this file names the symbol (name, parameter list, launch bound = the ABI).
#include "sgdn_scan.h"

SGDN_SCAN_KERNEL(saber_gdn_chunk_scan_f16s_k128_v128_c64, sgdn::kStateF16, sgdn::kGateSplit)
