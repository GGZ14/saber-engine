// The fusion lever, second measurement: `silu_mul_quant` folded into `down_proj`'s A staging.
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

template <int DWN, int DBK, int DKS, int DTM, bool WPERM, bool NT>
__device__ __forceinline__ void mxfp4_decode_fused_silu_impl(
    const __bf16 *__restrict__ gu, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    float *__restrict__ P, int *__restrict__ cnt,
    __bf16 *__restrict__ C, int M, int N, int K) {
  constexpr int BND = DWN * 16;
  constexpr int DASTR = DBK + DEC_PAD, DWSTR = DBK + DEC_PAD;
  constexpr int DNTHREADS = DWN * 32;
  __shared__ unsigned char sA[DEC_MTILE * DTM * DASTR];
  __shared__ unsigned char sW[BND * DWSTR];
  __shared__ int s_last;
  __shared__ float s_sc[DEC_MTILE * DTM], s_rs[DEC_MTILE * DTM], s_red[DWN];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int n0 = blockIdx.x * BND;
  const int ks = blockIdx.z;
  // No partition select: the fused route exists only where the graph edge says the A producer is
  // silu_mul, which is a single-partition MLP edge. A merged linear would need one rotated copy of
  // the gate_up output per partition, which no producer emits.
  const int slabs = (K + DBK - 1) / DBK;
  const int spb = (slabs + DKS - 1) / DKS;
  const int s_lo = ks * spb, s_hi = min(slabs, s_lo + spb);

  // ---- phase 1: the per-token scale, over the WHOLE row ---------------------------------------
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

  floatx8 acc[DTM];
#pragma unroll
  for (int i = 0; i < DTM; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;

  // ---- per-slab pieces, shared by the three slab loops below ----
  // ---- phase 2: stage A by RECOMPUTING silu(g)*u and quantising, instead of reading it --------
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
  // Row-order (checkpoint layout) W staging from a packed 8-byte word per (iteration, thread).
  // getw(it, gc, c) supplies the word so the plain and line-group loops share the fold.
  constexpr int NITW = BND * DBK / 2 / (DNTHREADS * 8);
  auto stage_w_row = [&](int k0, auto &&getw) {
#pragma unroll
    for (int it = 0; it < NITW; ++it) {
      const int idx = it * DNTHREADS * 8 + tid * 8;
      const int r = idx / (DBK / 2), c = idx % (DBK / 2), gn = n0 + r;
      const int gc = gn < N - 1 ? gn : N - 1;                  // clamp, never predicate
      const unsigned long long packed = getw(it, gc, c);
      const int blk = (k0 + c * 2) / 32;
      int dsh = (int)Wref[gc] - (int)Ws[(size_t)blk * N + gc];
      dsh = dsh < 0 ? 0 : (dsh > 15 ? 15 : dsh);
      const unsigned int t0 = kMag[dsh][0], t1 = kMag[dsh][1];
      unsigned int outw[4];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const unsigned int wv = (unsigned int)(packed >> (32 * h));
        const unsigned int ev = wv & 0x0F0F0F0Fu;
        const unsigned int od = (wv >> 4) & 0x0F0F0F0Fu;
        const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u)
                              | ((ev & 0x08080808u) << 4);
        const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u)
                              | ((od & 0x08080808u) << 4);
        outw[2 * h]     = __builtin_amdgcn_perm(bo, be, 0x05010400u);
        outw[2 * h + 1] = __builtin_amdgcn_perm(bo, be, 0x07030602u);
      }
      *(uint4_t *)(&sW[r * DWSTR + c * 2]) = uint4_t{outw[0], outw[1], outw[2], outw[3]};
    }
  };
  auto stage_w_wperm = [&](int k0) {
    if constexpr (WPERM) {
      // Fragment order, as in the folded kernel: consecutive threads take consecutive uint32, so
      // a wave's weight read is 128 contiguous bytes instead of sixteen rows K/2 bytes apart.
      // TWO lane slots per thread (an 8 B request), the width the int4 twin measured optimal
      // (4 B was 11% slower there; 16 B identical). Adjacent slots are adjacent rows at the same
      // k, exactly as in the folded kernel above.
      constexpr int KSTEPS_T = DBK / 16, NTILES_T = BND / 16;
      const unsigned int *__restrict__ Wp = (const unsigned int *)W;
      const int ksteps_g = K / 16, kstep0 = k0 / 16;
#pragma unroll
      for (int off = 0; off < NTILES_T * KSTEPS_T * 32; off += DNTHREADS * 2) {
        const int sl = off + tid * 2;                   // first of the pair; always even
        const int lane_ = sl & 31, rest = sl >> 5;
        const int kst = rest % KSTEPS_T, ntl = rest / KSTEPS_T;
        const int r = ntl * 16 + (lane_ & 15), kloc = kst * 16 + (lane_ >> 4) * 8;
        const int gn = n0 + r;
        // Clamp, never predicate; gn is always even here and N % 16 == 0 under WPERM, so
        // clamping to the last even row keeps the uint2 read 8-byte aligned and gc + 1 in
        // bounds; the fragment address is recomputed from the clamped row.
        const int gc = gn < N - 2 ? gn : N - 2;
        const int lanec = (lane_ & 16) | (gc & 15);
        const uint2_t wv = ld_w<NT>(
            (const uint2_t *)&Wp[((size_t)(gc >> 4) * ksteps_g + kstep0 + kst) * 32 + lanec]);
        const int blk = (k0 + kst * 16) / 32;
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          int d = (int)Wref[gc + q] - (int)Ws[(size_t)blk * N + gc + q];
          d = d < 0 ? 0 : (d > 15 ? 15 : d);
          const unsigned int t0 = kMag[d][0], t1 = kMag[d][1];
          const unsigned int ev = wq_dec(wv, q) & 0x0F0F0F0Fu;
          const unsigned int od = (wq_dec(wv, q) >> 4) & 0x0F0F0F0Fu;
          const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u)
                                | ((ev & 0x08080808u) << 4);
          const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u)
                                | ((od & 0x08080808u) << 4);
          *(uint2_t *)(&sW[(r + q) * DWSTR + kloc]) =
              uint2_t{__builtin_amdgcn_perm(bo, be, 0x05010400u),
                      __builtin_amdgcn_perm(bo, be, 0x07030602u)};
        }
      }
    }
  };
  auto mma_slab = [&]() {
    __syncthreads();
#pragma unroll
    for (int step = 0; step < DBK / 16; ++step) {
      const int kk = step * 16 + kb8;
      int2_t af[DTM], wf;
#pragma unroll
      for (int i = 0; i < DTM; ++i) {
        const unsigned char *pa = &sA[(i * 16 + col) * DASTR + kk];
        af[i][0] = *(const int *)pa; af[i][1] = *(const int *)(pa + 4);
      }
      const unsigned char *pw = &sW[(wave * 16 + col) * DWSTR + kk];
      wf[0] = *(const int *)pw; wf[1] = *(const int *)(pw + 4);
#pragma unroll
      for (int i = 0; i < DTM; ++i)
        acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf, acc[i]);
    }
    __syncthreads();
  };

  if constexpr (WPERM) {
    for (int s = s_lo; s < s_hi; ++s) {
      const int k0 = s * DBK;
      stage_a(k0);
      stage_w_wperm(k0);
      mma_slab();
    }
  } else {
    for (int s = s_lo; s < s_hi; ++s) {
      const int k0 = s * DBK;
      stage_a(k0);
      stage_w_row(k0, [&](int, int gc, int c) {
        return ld_w<NT>((const unsigned long long *)(W + ((size_t)gc * K + k0) / 2 + c));
      });
      mma_slab();
    }
  }

  const int n = n0 + wave * 16 + col;

  // At DKS==1 the accumulator already holds the whole K range, so the partial buffer, the
  // threadfence, the atomic and the reduction pass are all pure overhead -- and that overhead is
  // not small. Partial traffic is DKS*M*N floats: constant in the weight stream but LINEAR IN M,
  // 11.1 MB against gate_up's 44.6 MB of weights at M=40 and 17.8 MB at M=64. Picking DKS by shape
  // and by M is the launcher's job (split_k_for below); this is the path that makes DKS==1 free.
  // Bit-identical to the DKS==1 reduction path, which summed a single term -- the DIRECT=false
  // instantiation exists so that equivalence can be tested rather than asserted.
  if constexpr (DKS == 1) {   // DIRECT is not a knob here: the fused route has no indirect arm
    if (n < N) {
      const float rf = __int_as_float((int)Wref[n] << 23);
#pragma unroll
      for (int i = 0; i < DTM; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int m = i * 16 + kb8 + e;
          if (m < M) C[(size_t)m * N + n] = (__bf16)(acc[i][e] * rf * s_sc[m]);
        }
    }
    return;
  }

  if (n < N) {
#pragma unroll
    for (int i = 0; i < DTM; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = i * 16 + kb8 + e;
        if (m < M) P[((size_t)ks * M + m) * N + n] = acc[i][e];
      }
  }

  // Fused reduction. The DKS blocks covering one n-range race on an atomic counter and whoever
  // arrives last reduces in place, so there is no second kernel launch -- 13,376 of them per
  // decode run in the profile, at a 4.5 us mean host gap. It is also faster in pure kernel time
  // (-5.5% on gate_up to -29% on gdn.out) because the partials are still in cache when the
  // reduction reads them. The last block resets the counter, leaving it zeroed for the next
  // launch; launches on one stream are serialised, so no separate clearing pass is needed.
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    s_last = (atomicAdd(&cnt[blockIdx.x], 1) == DKS - 1);
  }
  __syncthreads();
  if (!s_last) return;
  if (tid == 0) cnt[blockIdx.x] = 0;

  const int nhi = min(n0 + BND, N);
  for (int nn = n0 + tid; nn < nhi; nn += DNTHREADS) {
    const float rf = __int_as_float((int)Wref[nn] << 23);
    for (int m = 0; m < M; ++m) {
      float sum = 0.f;
      for (int k = 0; k < DKS; ++k) sum += P[((size_t)k * M + m) * N + nn];
      C[(size_t)m * N + nn] = (__bf16)(sum * rf * s_sc[m]);
    }
  }
}



// The decode band's own tile parameters for `down_proj` (N=5120, K=8704) at M in {1,5,10,20,40}:
// the plain rules resolve those to ks4/tm1, ks4/tm2 and ks2/tm3, so the fused route offers exactly
// those three and nothing else. A shape the band would send elsewhere has no fused symbol, which
// is a refusal rather than a slower path.
#define FSILU(name, DBK, DKS, DTM)                                                                \
  extern "C" __global__ __launch_bounds__(DEC_DWN * 32) void name(                                \
      const __bf16 *__restrict__ gu, const unsigned char *__restrict__ W,                         \
      const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,               \
      float *__restrict__ P, int *__restrict__ cnt, __bf16 *__restrict__ C,                       \
      int M, int N, int K) {                                                                      \
    mxfp4_decode_fused_silu_impl<DEC_DWN, DBK, DKS, DTM, true, true>(gu, W, Ws, Wref, P, cnt, C,  \
                                                                     M, N, K);                    \
  }
FSILU(mxfp4_gemm_decode_fused_silu_bk128_ks4_tm1_wperm1_nt1, 128, 4, 1)
FSILU(mxfp4_gemm_decode_fused_silu_bk128_ks4_tm2_wperm1_nt1, 128, 4, 2)
FSILU(mxfp4_gemm_decode_fused_silu_bk128_ks2_tm3_wperm1_nt1, 128, 2, 3)
#undef FSILU
