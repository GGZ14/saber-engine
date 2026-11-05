#!/usr/bin/env python3
"""Derive `mxfp4_gemm_decode_silu.hip` from the decode kernel it is a variant of.

The fused kernel is the decode GEMM with `silu_mul_quant` folded into its A staging. Its GEMM half
-- the W staging, the fold, the WMMA loop, the split-K partials and the fused reduce -- must be
the decode kernel's, character for character, or the measurement is comparing two kernels rather
than one kernel with and without a fusion. So it is not transcribed, it is DERIVED: this script
takes `mxfp4_gemm_decode.hip`'s own impl and applies five named edits.

`mxfp4_gemm_decode.hip` is a hand-owned source (CLEANROOM-1 froze its retired generator's output), so the chain is
decode source -> derived fused, and the link is re-runnable.

  tools/gen_fused_silu.py [--check]
"""
import argparse
import os
import sys

HIP = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "crates", "saber-kernels", "hip"))
SRC = os.path.join(HIP, "mxfp4_gemm_decode.hip")
OUT = os.path.join(HIP, "mxfp4_gemm_decode_silu.hip")

HEADER = '''// The fusion lever, second measurement: `silu_mul_quant` folded into `down_proj`'s A staging.
//
// NOT generated, and NOT a shipped route until the numbers say so. The GEMM half of this file is
// DERIVED MECHANICALLY from `mxfp4_gemm_decode.hip`'s own impl by
// `tools/gen_fused_silu.py`, which renames the template, swaps the e4m3 A operand for the bf16
// gate_up, replaces the staging lambda and points the epilogue at the scale computed here. Every
// other line -- the W staging, the fold, the WMMA loop, the split-K partials and the fused reduce
// -- is the decode kernel's, character for character. Re-run the deriver rather than editing this
// by hand, and the bit-exact gate against the two-kernel composition is what proves the copy has
// not drifted.
//
// WHAT THIS COSTS, stated before the numbers: `silu_mul_quant` computes the per-token scale from
// a full-row amax. A split-K workgroup sees only K/DKS of the row, so it cannot get that scale
// from its own slice -- and the decode band for `down_proj` is DKS=4 at M<=16, where the kernel's
// own ledger says DKS=1 is 2.4x worse. So the fused kernel keeps split-K and pays for it: EVERY
// workgroup reads the WHOLE gate_up row in phase 1. At M=5 that is 160 workgroups x 174 KiB =
// 27.8 MB of reads on top of a 23.7 MB weight stream, all of it Infinity-Cache resident after the
// first reader. Whether that is cheaper than one deleted node is the question.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "mxfp4_common.h"

#define DEC_DWN 8   // waves per decode block; the N tile is DEC_DWN*16 = 128 wide
#define DEC_MTILE 16
#define DEC_PAD 8
static __device__ __forceinline__ unsigned int wq_dec(uint2_t v, int q) { return q ? v[1] : v[0]; }
template <bool NT, typename T>
static __device__ __forceinline__ T ld_w(const T *p) {
  if constexpr (NT) return __builtin_nontemporal_load(p);
  else return *p;
}

union SiluVec8 { uint4_t u4; __bf16 h[8]; };

// `silu_mul_quant`'s arithmetic, to the rounding: silu to bf16, product to bf16, and only then
// the amax. Anything else here is a different answer, however small.
static __device__ __forceinline__ float silu_mul_bf16(__bf16 g, __bf16 u) {
  const float x = (float)g;
  const float t = (float)(__bf16)(x / (1.f + expf(-x)));
  return (float)(__bf16)(t * (float)u);
}

'''

FOOTER = '''
// The decode band's own tile parameters for `down_proj` (N=5120, K=8704) at M in {1,5,10,20,40}:
// the plain rules resolve those to ks4/tm1, ks4/tm2 and ks2/tm3, so the fused route offers exactly
// those three and nothing else. A shape the band would send elsewhere has no fused symbol, which
// is a refusal rather than a slower path.
#define FSILU(name, DBK, DKS, DTM)                                                                \\
  extern "C" __global__ __launch_bounds__(DEC_DWN * 32) void name(                                \\
      const __bf16 *__restrict__ gu, const unsigned char *__restrict__ W,                         \\
      const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,               \\
      float *__restrict__ P, int *__restrict__ cnt, __bf16 *__restrict__ C,                       \\
      int M, int N, int K) {                                                                      \\
    mxfp4_decode_fused_silu_impl<DEC_DWN, DBK, DKS, DTM, true, true>(gu, W, Ws, Wref, P, cnt, C,  \\
                                                                     M, N, K);                    \\
  }
FSILU(mxfp4_gemm_decode_fused_silu_bk128_ks4_tm1_wperm1_nt1, 128, 4, 1)
FSILU(mxfp4_gemm_decode_fused_silu_bk128_ks4_tm2_wperm1_nt1, 128, 4, 2)
FSILU(mxfp4_gemm_decode_fused_silu_bk128_ks2_tm3_wperm1_nt1, 128, 2, 3)
#undef FSILU
'''

PHASE1 = '''  // ---- phase 1: the per-token scale, over the WHOLE row ---------------------------------------
  // `silu_mul_quant` computes it from a full-row amax, so a split-K workgroup -- which sees only
  // K/DKS -- cannot get it from its own slice. Every workgroup therefore reads the whole gate_up
  // row. That redundancy IS the cost of this fusion and it is the thing being measured.
  //
  // It is bit-exact for a reason that does not hold in general: the reduction is a MAX, which is
  // associative and commutative exactly, so any order gives the producer's answer. A fused norm
  // would not have that luxury -- there the fp32 sum order IS the answer (LEDGER, attn_glue).
  for (int r = tid; r < DEC_MTILE * DTM; r += DNTHREADS) s_sc[r] = 0.f;
  __syncthreads();
  {
    const int NV = K >> 3;
    for (int r = 0; r < M; ++r) {
      const __bf16 *g = gu + (size_t)r * 2 * K;
      float amax = 0.f;
      for (int v = tid; v < NV; v += DNTHREADS) {
        SiluVec8 vg, vu;
        vg.u4 = ((const uint4_t *)g)[v];
        vu.u4 = ((const uint4_t *)(g + K))[v];
#pragma unroll
        for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(silu_mul_bf16(vg.h[j], vu.h[j])));
      }
      for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_down(amax, o));
      if (lane == 0) s_red[wave] = amax;
      __syncthreads();
      if (tid == 0) {
        float v = s_red[0];
#pragma unroll
        for (int w = 1; w < DWN; ++w) v = fmaxf(v, s_red[w]);
        // The producer's own constants: `sc = max(amax * (1/448), 1/(448*512))`, and the quant
        // multiplies by the RECIPROCAL. Both are load-bearing -- a divide here is a different
        // last bit (LEDGER, attn_glue, where the reference server's quantiser divides and the reference's does not).
        s_sc[r] = fmaxf(v * (1.f / 448.f), 1.f / (448.f * 512.f));
        s_rs[r] = 1.f / s_sc[r];
      }
      __syncthreads();
    }
  }

  floatx8 acc[DTM];'''

STAGE = '''  // ---- phase 2: stage A by RECOMPUTING silu(g)*u and quantising, instead of reading it --------
  // Same 16-byte store into sA as the plain kernel; the 16 e4m3 bytes are produced here rather
  // than copied. The products are recomputed rather than kept from phase 1 because keeping them
  // would need M*K bf16 of LDS -- 87 KB at M=5, against a 64 KB budget.
  auto stage_a = [&](int k0) {
#pragma unroll
    for (int off = 0; off < DEC_MTILE * DTM * DBK; off += DNTHREADS * 16) {
      const int idx = off + tid * 16;
      if (idx < DEC_MTILE * DTM * DBK) {
        const int r = idx / DBK, c = idx % DBK;
        const int rc = r < M - 1 ? r : M - 1;                  // clamp, never predicate
        const __bf16 *g = gu + (size_t)rc * 2 * K + k0 + c;
        const float rs = s_rs[rc];
        unsigned int outq[4];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          SiluVec8 vg, vu;
          vg.u4 = ((const uint4_t *)g)[h];
          vu.u4 = ((const uint4_t *)(g + K))[h];
          unsigned int lo = 0, hi = 0;
#pragma unroll
          for (int j = 0; j < 4; j += 2) {
            const float a = fminf(fmaxf(silu_mul_bf16(vg.h[j], vu.h[j]) * rs, -448.f), 448.f);
            const float b = fminf(fmaxf(silu_mul_bf16(vg.h[j + 1], vu.h[j + 1]) * rs, -448.f),
                                  448.f);
            lo |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu) << (j * 8);
          }
#pragma unroll
          for (int j = 4; j < 8; j += 2) {
            const float a = fminf(fmaxf(silu_mul_bf16(vg.h[j], vu.h[j]) * rs, -448.f), 448.f);
            const float b = fminf(fmaxf(silu_mul_bf16(vg.h[j + 1], vu.h[j + 1]) * rs, -448.f),
                                  448.f);
            hi |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu) << ((j - 4) * 8);
          }
          outq[2 * h] = lo;
          outq[2 * h + 1] = hi;
        }
        *(uint4_t *)(&sA[r * DASTR + c]) = uint4_t{outq[0], outq[1], outq[2], outq[3]};
      }
    }
  };
'''


def derive():
    src = open(SRC).read().splitlines(True)
    start = next(i for i, l in enumerate(src)
                 if l.startswith("template <int DWN, int DBK, int DKS, int DTM, bool WPERM"))
    end = next(i for i, l in enumerate(src) if l.startswith('extern "C" __global__'))
    impl = "".join(src[start:end])

    def sub(old, new, what):
        if old not in impl_ref[0]:
            sys.exit(f"gen_fused_silu: anchor for `{what}` not found in {SRC};\n"
                     f"  the decode kernel changed shape -- re-derive by hand, do not guess")
        impl_ref[0] = impl_ref[0].replace(old, new, 1)

    impl_ref = [impl]
    # 1. the signature: an e4m3 A and its scales become the bf16 gate_up, and the partition
    #    select's operands go with them.
    sub('template <int DWN, int DBK, int DKS, int DTM, bool WPERM, bool DIRECT = true, '
        'bool NT = false>\n'
        '__device__ __forceinline__ void saber_mxfp4_fp8_gemm_decode_impl(\n'
        '    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,\n'
        '    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,\n'
        '    const float *__restrict__ As, float *__restrict__ P, int *__restrict__ cnt,\n'
        '    __bf16 *__restrict__ C, int M, int N, int K, int pb1, int pb2, long astride) {',
        'template <int DWN, int DBK, int DKS, int DTM, bool WPERM, bool NT>\n'
        '__device__ __forceinline__ void mxfp4_decode_fused_silu_impl(\n'
        '    const __bf16 *__restrict__ gu, const unsigned char *__restrict__ W,\n'
        '    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,\n'
        '    float *__restrict__ P, int *__restrict__ cnt,\n'
        '    __bf16 *__restrict__ C, int M, int N, int K) {', "signature")
    # 2. the shared state the two phases need.
    sub('  __shared__ int s_last;',
        '  __shared__ int s_last;\n'
        '  __shared__ float s_sc[DEC_MTILE * DTM], s_rs[DEC_MTILE * DTM], s_red[DWN];',
        "shared state")
    # 3. the merged-linear partition select has no meaning on this edge.
    t = impl_ref[0]
    part = t[t.index("  // Merged-linear partition select"):t.index("  const int slabs =")]
    sub(part,
        "  // No partition select: the fused route exists only where the graph edge says the A "
        "producer is\n  // silu_mul, which is a single-partition MLP edge. A merged linear would "
        "need one rotated copy of\n  // the gate_up output per partition, which no producer "
        "emits.\n", "partition select")
    # 4. phase 1 goes in front of the accumulator; 5. the staging lambda is replaced.
    sub("  floatx8 acc[DTM];", PHASE1, "accumulator")
    t = impl_ref[0]
    old_stage = t[t.index("  auto stage_a = [&](int k0) {"):
                  t.index("  // Row-order (checkpoint layout) W staging")]
    sub(old_stage, STAGE, "stage_a")
    # 6. the epilogue's per-token factor is the scale this kernel computed, not a producer's.
    t = impl_ref[0].replace("acc[i][e] * rf * As[m]", "acc[i][e] * rf * s_sc[m]")
    t = t.replace("sum * rf * As[m]", "sum * rf * s_sc[m]")
    # 7. DIRECT is not a knob here: there is no indirect arm to compare against.
    t = t.replace("  if constexpr (DKS == 1 && DIRECT) {",
                  "  if constexpr (DKS == 1) {   // DIRECT is not a knob here: the fused route "
                  "has no indirect arm")
    # The only surviving "DIRECT" is in the decode kernel's own comment about the control that
    # proves the DKS==1 arm; there is no such control here, and the comment is worth keeping.
    if "As[" in t or "DIRECT)" in t or "bool DIRECT" in t:
        sys.exit("gen_fused_silu: an A-scale or a live DIRECT knob survived the derivation")
    return HEADER + t + FOOTER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    want = derive()
    if a.check:
        have = open(OUT).read() if os.path.exists(OUT) else ""
        if have != want:
            sys.exit(f"gen_fused_silu: {OUT} is not what the deriver produces; re-run it")
        print(f"gen_fused_silu: {os.path.basename(OUT)} matches the deriver")
        return
    open(OUT, "w").write(want)
    print(f"gen_fused_silu: wrote {OUT} ({len(want.splitlines())} lines)")


if __name__ == "__main__":
    main()
