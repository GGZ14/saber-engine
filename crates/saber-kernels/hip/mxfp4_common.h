// Saber source (CLEANROOM-1, 2026-10-25: frozen from a retired generator's output; edit this file directly)
//   reference sha256 79a7eb6f261c2939386a4c231cf5727a0a1f2325d8a391419178778da1fd2ae2
//   device code sliced VERBATIM by line range; only the __global__ signatures became
//   __device__ __forceinline__ *_impl and the extern "C" wrappers at the end are new.
// Do not edit by hand: re-run the generator. No host code, no pybind, no getenv in this file --
// every tile parameter the prod launcher chose from an env var is an explicit symbol here.
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

typedef float floatx8 __attribute__((ext_vector_type(8)));
typedef int int2_t __attribute__((ext_vector_type(2)));
typedef unsigned int uint4_t __attribute__((ext_vector_type(4)));
typedef unsigned int uint2_t __attribute__((ext_vector_type(2)));

__device__ __constant__ unsigned char kLUT[16] = {
    0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C,
    0x80, 0xB0, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC};

// kMag[d]: the 8 e2m1 magnitudes, each already shifted DOWN by d binades and encoded as an e4m3
// byte, packed 4 per word so `v_perm_b32` can resolve four lookups per instruction. Folding the MX
// block exponent into the WEIGHT here removes the per-32-block rescale from the inner loop
// entirely -- the scale becomes one per-row factor applied in the epilogue. The sign is separable
// (a negative code is the positive byte with bit 7 set), which is why only magnitudes are tabulated.
//
// The table runs into e4m3 SUBNORMALS, and that is deliberate. Stopping at the smallest e4m3 NORMAL
// (2^-6) makes the fold exact only while d <= 5, and this checkpoint exceeds that: mlp.down_proj
// reaches d=10 and holds 11,354 of the model's 14,433 out-of-range blocks. On layer 58 that zeroed
// 6.7% of the weights on 36 output channels, for per-channel errors up to 18.7% and a 1.18%
// whole-layer norm error -- squarely inside the range CHECKALL flags. Extending into subnormals
// (down to 2^-9) makes every d <= 8 exact and leaves d=9..12 rounding rather than flushing:
// measured, every affected layer with max d <= 8 goes to exactly zero error, and layer 58 (the only
// d=10 one) drops 4x to 0.31%.
//
// This is only sound because the gfx12 fp8 WMMA HONOURS e4m3 subnormals rather than flushing them
// -- verified on hardware, exact for every m*2^-9, m=1..7 (~/mxfp4_work/tier6/subnorm.hip).
__device__ __constant__ unsigned int kMag[16][2] = {
  {0x3c383000u, 0x4c484440u},
  {0x34302800u, 0x44403c38u},
  {0x2c282000u, 0x3c383430u},
  {0x24201800u, 0x34302c28u},
  {0x1c181000u, 0x2c282420u},
  {0x14100800u, 0x24201c18u},
  {0x0c080400u, 0x1c181410u},
  {0x06040200u, 0x14100c08u},
  {0x03020100u, 0x0c080604u},
  {0x01010000u, 0x06040302u},
  {0x01000000u, 0x03020101u},
  {0x00000000u, 0x01010100u},
  {0x00000000u, 0x01000000u},
  {0x00000000u, 0x00000000u},
  {0x00000000u, 0x00000000u},
  {0x00000000u, 0x00000000u},
};

#define BK 64
#define PAD 8
#define ASTR (BK + PAD)
#define WSTR (BK + PAD)
