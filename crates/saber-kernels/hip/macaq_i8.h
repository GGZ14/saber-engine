// Saber source (2026-11-17): the per-token symmetric INT8 activation law of the W4A8-iu8 pack, shared by
// every `*_hadi8_*` producer (tools/gen_macaq_iu8.py). The law: sc = max(amax / 127, 1e-30) (the producer
// computes it), q = clamp(round_half_even(x / sc), -127, 127). Both divisions are correctly rounded
// (`__fdiv_rn`), so a numpy float32 program reproduces the codes bit for bit (the round is IEEE
// round-half-even, `rint`).
// Clamping before rint is the same as after (the bounds are integers). The code's two's-complement byte.
#pragma once
#include <hip/hip_runtime.h>

__device__ __forceinline__ unsigned macaq_i8_code(float x, float sc) {
  const float y = fminf(fmaxf(__fdiv_rn(x, sc), -127.f), 127.f);
  return ((unsigned)(int)__builtin_rintf(y)) & 0xffu;
}
