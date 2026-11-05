// The DFlash2 drafter's QUERY q/k norm + rope -- its own kernel, and NOT an `attn_glue` band row.
//
// WHY THIS FILE EXISTS, which is the finding of workstream F round 28. The drafter's plan routed
// `layer.<i>.attn.qk_norm_rope` to `attn_glue::qk_norm_rope_h128_rot128_mi0_red5` -- the right
// head width, the right plain rope, the right ungated `[q|k|v]` stride -- and every one of those
// is a SHAPE. The ARITHMETIC underneath them is the TARGET's and the drafter does not share it:
//
//   * `attn_glue`'s norm is a **GemmaRMSNorm**: `x * rsqrt(var + eps) * (f32(w) + 1)`. Qwen3.8's
//     q/k norm is one. The drafter's is `RMSNorm(self.head_dim, eps=rms_norm_eps)` --
//     `qwen3_dflash.py:239` -- which is the reference server's PLAIN norm, `x * rsqrt(var + eps) * w`, with no
//     `1 +`. Applying the target's term to the drafter's weight is a ~2x error on every element
//     of q and k in all five layers, and it is what made `--no-seed`'s `chain.attn.l0.q` differ
//     in 44,612 of 65,536 bytes while every kernel in that prefix gated green on its own.
//   * the variance REDUCTION is a different program (ATen's `mean` there, Triton's `tl.sum` here),
//   * the normed value is NOT rounded to bf16 before the rope reads it (see below), and
//   * the rope CONTRACTS into FMAs, while `attn_glue` is compiled `-ffp-contract=off` for its own
//     Triton reference and the flag is translation-unit wide.
//
// The last one settles where this kernel can live all by itself: it cannot be an `attn_glue`
// instantiation at any number of template parameters. `dflash_ctx.hip` made the same argument for
// the context store's rope and it is the same argument here.
//
// THE REFERENCE, read off prod's OWN generated code rather than off the model source, because the
// model source is eager and the program that runs is inductor's. `DFlashQwen3Model` is
// `@support_torch_compile`, so this seam is two Triton kernels in the drafter's compiled piece --
// `triton_per_fused_9` (the sums of squares, q and k as one combo kernel) and `triton_poi_fused_10`
// (the epilogue: scale, weight, rope, store) -- and both were read out of the inductor cache and
// the Triton cache a capture leaves behind (`.cache/inductor-prop*`, `.cache/triton/*/*.amdgcn`).
// What they do, in order:
//
//     var = sum over the 128 elements of the head of x*x                (fp32, see ORD)
//     inv = rsqrt(var * (1/128) + eps)                                  (v_rsq_f32, see RSQ)
//     n   = x * inv                                                     (fp32)
//     nw  = n * w                                                       (fp32)
//     o1  = fma(nw1, cos, -(nw2 * sin))                                 (i < 64)
//     o2  = fma(nw2, cos,   nw1 * sin )                                 (i >= 64)
//     store bf16, round to nearest even
//
// TWO THINGS IN THAT LIST ARE NOT IN THE PYTHON. `ir.ops.rms_norm` writes
// `x.to(weight.dtype) * weight` -- a bf16 round between the scale and the weight -- and the
// generated Triton has `tmp9 = tmp8.to(tl.float32)` in its place: a no-op. There is not one
// `tl.bfloat16` in the whole generated module, because inductor computes this region in fp32 and
// rounds only at the store. So the drafter's compiled q/k norm rounds ONCE, where the eager op
// rounds twice, and a port that follows the source rather than the code object is wrong in the
// last bit of a few elements per row. (The LEDGER records the same trap for `silu_quant`, from
// the other direction.)
//
// And the rope's two outputs do NOT round the same product. `o1` keeps `nw2 * sin` as a separate
// `v_mul_f32` and fuses `nw1 * cos`; `o2` keeps `nw1 * sin` and fuses `nw2 * cos`. The C++ rope
// `dflash_ctx.hip` implements -- the one the CONTEXT store calls -- rounds `cos * y` in `o2` and
// fuses `sin * x`, which is the mirror image. Two ropes, one model, a rounding apart: the context
// store calls `torch.ops._C.rotary_embedding` directly and this one is inductor's.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

typedef __bf16 bf16x4_qk __attribute__((ext_vector_type(4)));

#define DQ_HD   128    // the drafter's head_dim, and its FULL rotary dim (no partial factor)
#define DQ_VEC    4    // elements per lane; 32 lanes x 4 covers one head
#define DQ_WAVE  32

__device__ __forceinline__ float dq_f32(__bf16 v) { return (float)v; }

// `v_add_f32_dpp ... row_shr:N bound_ctrl:1`, which is what Triton's `tl.sum` lowers to here:
// lane `i` adds lane `i - N` of its own 16-lane ROW, and a lane with no source adds an exact zero.
// `bound_ctrl` is the whole reason this is not `__shfl_up`: at N = 8 lanes 16..23 would otherwise
// read lanes 8..15, which is a different sum for every lane in the second row (it happens not to
// reach lane 31, which is the lane the result is read from -- but "happens not to" is not a
// contract).
template <int CTRL>
__device__ __forceinline__ float dq_row_shr_add(float v) {
  const int sh = __builtin_amdgcn_update_dpp(0, __float_as_int(v), CTRL, 0xf, 0xf, true);
  return v + __int_as_float(sh);
}

// The sum of squares of ONE head, in the reference's own order.
//
// ORD is the per-lane order and it is a measured choice, not a preference. Each lane owns four
// CONTIGUOUS elements (`global_load_b64` at `lane * 4`), and the ISA opens the accumulator with a
// plain `v_dual_mul_f32` on the SECOND of them and then fmas the first, the third and the fourth:
//
//     acc = e1*e1 ; acc = fma(e0,e0,acc) ; acc = fma(e2,e2,acc) ; acc = fma(e3,e3,acc)
//
// which is ORD=0. ORD=1 is the same four products in index order -- the transcription anyone
// would write -- and it is this kernel's negative control: two orders of four fp32 fmas differ
// in the last bit often enough to move a bf16 result, and if the control does NOT differ on the
// captured rows then the gate has not proved the order.
//
// The cross-lane fold is then the DPP ladder above at 8, 4, 2, 1 followed by `permlanex16` and
// one add, with the answer in LANE 31 (`v_readlane_b32 s10, v1, 31`).
template <int ORD>
__device__ __forceinline__ float dq_head_sumsq(const __bf16 *__restrict__ in, int lane) {
  const bf16x4_qk v = *(const bf16x4_qk *)(in + lane * DQ_VEC);
  const float e0 = dq_f32(v[0]), e1 = dq_f32(v[1]), e2 = dq_f32(v[2]), e3 = dq_f32(v[3]);
  float acc;
  if constexpr (ORD == 0) {
    acc = e1 * e1;
    acc = fmaf(e0, e0, acc);
    acc = fmaf(e2, e2, acc);
    acc = fmaf(e3, e3, acc);
  } else {
    // A PAIRWISE TREE, and it has to be a tree rather than a permutation of the chain. The first
    // draft of this control summed the same four squares starting from `e0` instead of `e1` and
    // was IDENTICAL to the route on every row of every shape, synthetic ones included -- because
    // `fma(e0, e0, fl(e1*e1))` and `fma(e1, e1, fl(e0*e0))` are `round(e0^2 + e1^2)` either way
    // and fp32 addition is commutative. A control that is the same function is not a control; it
    // was caught by the gate's own "DID NOT FIRE" rule, which is the second time that rule has
    // paid for itself this round.
    acc = fmaf(e0, e0, e1 * e1) + fmaf(e2, e2, e3 * e3);
  }
  acc = dq_row_shr_add<0x118>(acc);   // row_shr:8
  acc = dq_row_shr_add<0x114>(acc);   // row_shr:4
  acc = dq_row_shr_add<0x112>(acc);   // row_shr:2
  acc = dq_row_shr_add<0x111>(acc);   // row_shr:1
  acc = acc + __shfl_xor(acc, 16, DQ_WAVE);
  return __shfl(acc, DQ_WAVE - 1, DQ_WAVE);
}

// RSQ=0 is the reference's: `tl.rsqrt` lowers to `llvm.amdgcn.rsq.f32` and the ISA carries a bare
// `v_rsq_f32_e32`, a ~1-ULP approximation. RSQ=1 is ATen's correctly-rounded one (fp64), which is
// what `attn_glue` needs for the TARGET because that norm's reference is ATen -- and which is this
// kernel's second control. The LEDGER records what the difference is worth on a rope: 15 of 40
// cases, by 1 to 29 bytes, always a rotary PAIR whose shared `inv` crossed a rounding boundary.
template <int RSQ>
__device__ __forceinline__ float dq_rsqrt(float v) {
  if constexpr (RSQ == 0) {
    return __builtin_amdgcn_rsqf(v);
  } else {
    return (float)(1.0 / sqrt((double)v));
  }
}

// One WAVE per (token, head). `head < nq` is a q head and reads `qw`; the rest are k heads,
// `head - nq` into `kw`, and the k block starts at `nq * HD` because the drafter's projection is a
// plain `[q | k | v]` -- there is no output gate to double the q stride, which is the one thing
// `attn_glue`'s `GATED` parameter already got right.
// NRM=0 is the compiled program's: fp32 from the scale through the weight to the rope, one
// rounding at the store. NRM=1 puts back the bf16 round `ir.ops.rms_norm` writes in Python
// (`x = x.to(weight.dtype) * weight`) and inductor elides -- the mistake a port that reads the
// SOURCE instead of the generated code makes, and the one this kernel's whole third paragraph is
// about. It is a control that cannot silently agree: it changes a value's width before two
// multiplies and a store.
template <int ORD, int RSQ, int NRM = 0>
__device__ __forceinline__ void dq_norm_rope_impl(
    const __bf16 *__restrict__ qkv,      // [m, qkv_n] : q | k | v
    const __bf16 *__restrict__ qw,       // [HD]
    const __bf16 *__restrict__ kw,       // [HD]
    const __bf16 *__restrict__ cos_sin,  // [max_pos, HD] : cos(HD/2) | sin(HD/2)
    const long *__restrict__ pos,        // [m] int64, as the reference server stores positions
    __bf16 *__restrict__ q_out,          // [m, nq * HD]
    __bf16 *__restrict__ k_out,          // [m, nkv * HD]
    int m, int nq, int nkv, int qkv_n, float eps) {
  const int tok = (int)blockIdx.x;
  const int head = (int)blockIdx.y;
  const int lane = (int)threadIdx.x;
  if (tok >= m || head >= nq + nkv) return;
  const bool is_k = head >= nq;
  const int lh = is_k ? head - nq : head;
  const __bf16 *in = qkv + (long)tok * qkv_n + (long)(is_k ? nq + lh : lh) * DQ_HD;
  const __bf16 *w = is_k ? kw : qw;
  __bf16 *out = is_k ? k_out + ((long)tok * nkv + lh) * DQ_HD
                     : q_out + ((long)tok * nq + lh) * DQ_HD;

  const float var = dq_head_sumsq<ORD>(in, lane);
  // `var * (1/128) + eps` as one fma, which is the reference's `v_fmaak_f32 v2, s2, v2, 0x358637bd`
  // -- and identical to the divide it came from, because 128 is a power of two and `var / 128` is
  // exact. The eps is the drafter's own `rms_norm_eps`; `_build_fused_kv_buffers` asserts every
  // layer's `q_norm.variance_epsilon` is that one value.
  const float inv = dq_rsqrt<RSQ>(fmaf(var, 1.f / (float)DQ_HD, eps));

  const int half = DQ_HD / 2;
  const __bf16 *cache = cos_sin + pos[tok] * (long)DQ_HD;
#pragma unroll
  for (int e = 0; e < DQ_VEC; ++e) {
    const int i = lane * DQ_VEC + e;             // this lane's element of the head
    const int p = (i < half) ? i + half : i - half;   // its NEOX partner
    const int ro = (i < half) ? i : i - half;    // the rotary index the pair shares
    const float ni = dq_f32(in[i]) * inv;
    const float np = dq_f32(in[p]) * inv;
    const float nwi = (NRM == 0 ? ni : (float)(__bf16)ni) * dq_f32(w[i]);
    const float nwp = (NRM == 0 ? np : (float)(__bf16)np) * dq_f32(w[p]);
    const float c = dq_f32(cache[ro]);
    const float s = dq_f32(cache[half + ro]);
    // One product rounded, the other fused -- and WHICH one differs between the two halves.
    out[i] = (__bf16)(i < half ? fmaf(nwi, c, -(nwp * s)) : fmaf(nwi, c, nwp * s));
  }
}

#define DQ_KERNEL(NAME, ORD, RSQ, NRM)                                                             \
  extern "C" __global__ __launch_bounds__(DQ_WAVE) void NAME(                                      \
      const __bf16 *__restrict__ qkv, const __bf16 *__restrict__ qw,                               \
      const __bf16 *__restrict__ kw, const __bf16 *__restrict__ cos_sin,                           \
      const long *__restrict__ pos, __bf16 *__restrict__ q_out, __bf16 *__restrict__ k_out,        \
      int m, int nq, int nkv, int qkv_n, float eps) {                                              \
    dq_norm_rope_impl<ORD, RSQ, NRM>(qkv, qw, kw, cos_sin, pos, q_out, k_out, m, nq, nkv, qkv_n,   \
                                     eps);                                                         \
  }

// The route, and its two controls. A control here is one arithmetic choice changed and nothing
// else -- same grid, same block, same kernarg layout -- so the gate can dispatch it through the
// routed descriptor with the symbol replaced (`dispatch_as`).
DQ_KERNEL(dflash_qk_norm_rope_h128, 0, 0, 0)
DQ_KERNEL(dflash_qk_norm_rope_h128_ord1, 1, 0, 0)
DQ_KERNEL(dflash_qk_norm_rope_h128_rsqf64, 0, 1, 0)
DQ_KERNEL(dflash_qk_norm_rope_h128_bf16n, 0, 0, 1)
#undef DQ_KERNEL
