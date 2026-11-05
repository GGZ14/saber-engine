// Saber gated-delta-net kernels, CLEANROOM-2 (CR2-3c): the certified chunk scan, f32 h0 / ht (spec 2.8.2).
// Body in sgdn_scan.h; this file names the symbol (name, parameter list, launch bound = the ABI).
#include "sgdn_scan.h"

SGDN_SCAN_KERNEL(saber_gdn_chunk_scan_k128_v128_c64, sgdn::kStateF32, sgdn::kGateSplit)
