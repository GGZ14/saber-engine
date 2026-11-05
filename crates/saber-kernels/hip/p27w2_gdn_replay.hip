// Saber gated-delta-net kernels, CLEANROOM-2 (CR2-3c): the 27B batched decode in replay form, operands read from
// global per row (spec 2.10.1): the law (round after the replay), the no-round control, and materialize. Body in
// sgdn_replay.h; this file is the instantiation table.
#include "sgdn_replay.h"

#define SGDN_REPLAY(NAME, VS, MODE)                                                                              \
  extern "C" __global__ __launch_bounds__(256 / VS) void NAME(SGDN_REPLAY_PARAMS) {                             \
    sgdn::replay_global<VS, MODE>(SGDN_REPLAY_ARGS);                                                             \
  }

SGDN_REPLAY(p27w2_gdn_replay_f16s_k128_v128_ab1_vs1, 1, 0)
SGDN_REPLAY(p27w2_gdn_replay_f16s_k128_v128_ab1_vs2, 2, 0)
SGDN_REPLAY(p27w2_gdn_replay_f16s_k128_v128_ab1_vs4, 4, 0)
SGDN_REPLAY(p27w2_gdn_replay_noround_f16s_k128_v128_ab1_vs1, 1, 1)
SGDN_REPLAY(p27w2_gdn_replay_materialize_f16s_k128_v128_ab1_vs1, 1, 2)
