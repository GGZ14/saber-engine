// Saber gated-delta-net kernels, CLEANROOM-2 (CR2-3c): the 27B batched decode in replay form with every shared
// operand staged once in LDS (spec 2.10.2). Body in sgdn_replay.h; this file is the instantiation table.
#include "sgdn_replay.h"

#define SGDN_REPLAY_LDS(VS)                                                                                    \
  extern "C" __global__ __launch_bounds__(256 / VS) void p27w2_gdn_replay_lds_f16s_k128_v128_ab1_vs##VS(       \
      SGDN_REPLAY_PARAMS) {                                                                                    \
    sgdn::replay_lds<VS>(SGDN_REPLAY_ARGS);                                                                    \
  }

SGDN_REPLAY_LDS(1)
SGDN_REPLAY_LDS(2)
SGDN_REPLAY_LDS(4)
