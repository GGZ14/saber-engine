// The producers that stand between the drafter's PLAN and a real proposal.
//
// None of them is a port: prod runs all three as inductor-compiled Triton, fused out of the reference server's own
// Python. The census counted them at ~46 launches with the conv family, and the drafter's plan
// (`saber-model::dflash_graph`) declares them as the last shapes that are not code objects apart
// from the ATen selections it will never own and the rope the rules cannot yet select.
//
//   silu_quant    `SiluAndMul.forward_native` -- `F.silu(x[..., :d]) * x[..., d:]` -- followed by
//                 `QuantFP8._quantize_group_native` at GroupShape(1, 128). Five launches a step.
//   conv_add_rms  `_grouped_conv` (the `finish` half) followed by `ir.ops.fused_add_rms_norm`,
//                 which is ALREADY one kernel in prod: the census's
//                 `triton_red_fused_add_arange_bitwise_and_constant_pad_nd_*` carries the conv's
//                 arange/pad ops AND a reduction, and that reduction is this norm's variance. Ten
//                 launches a step, and fusing them here removes ten more from the plan than a
//                 separate norm would.
//   rms_norm      `ir.ops.rms_norm` on its own, which happens exactly once: layer 0 is the only
//                 layer whose residual is None, so it is the only norm with nothing to fuse into.
//
// THE THREE ROUNDINGS the conv's LEDGER entry records apply to all three, because they are
// properties of how inductor compiles a bf16 graph and not of any one kernel:
//   * eager is NOT the reference -- eager bf16 rounds at every op, inductor keeps fp32 across a
//     fused region and rounds at a real store;
//   * a quant's amax is over the BF16-ROUNDED producer output WHEN that bf16 tensor is
//     materialised -- which is a condition, not a rule, and `silu_quant` below is the case where
//     it fails: the conv's output is stored as a residual and is rounded, the MLP activation has
//     no consumer but the quant and stays fp32;
//   * `absmax / 448` compiles to a RECIPROCAL MULTIPLY -- the source divides and the program does
//     not.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

typedef __bf16 bf16x4 __attribute__((ext_vector_type(4)));
typedef float f32x4 __attribute__((ext_vector_type(4)));

#define DM_QG   128    // the activation-quant group, GroupShape(1, 128)
#define DM_VEC    4    // bf16 per thread; 32 lanes cover one quant group
#define DM_NW     4    // waves per workgroup
#define DM_NT   (DM_NW * 32)
#define DM_ROW  256    // threads per row in the norm, which owns a whole row's reduction

__device__ __forceinline__ float dm_f32(__bf16 v) { return (float)v; }

// `exp`, and NOT the one the drafter's attention matched. There are two lowerings of exp in this
// codebase and they are different functions:
//
//   * a HAND-WRITTEN Triton kernel's `tl.exp` becomes `llvm.exp.f32`, whose AMDGPU expansion is
//     `v_exp_f32` plus `v_ldexp_f32`. `dflash_attn.hip` matches that with `__builtin_expf`.
//   * an INDUCTOR-GENERATED kernel calls the ROCm math library instead. The generated IR for this
//     fusion carries `tt.extern_elementwise ... symbol = "__ocml_exp_f32"` -- an extern call, not
//     an intrinsic, and a different implementation.
//
// Read off the reference's own IR, and it is worth the 31 scales it was: with `__builtin_expf`
// this kernel matched every quantised BYTE and still missed 31 of 544 scales, because a 1-ULP
// activation moves a group's amax without moving what that group rounds to.
extern "C" __device__ float __ocml_exp_f32(float);
__device__ __forceinline__ float dm_exp(float x) { return __ocml_exp_f32(x); }

// `F.silu(x)` is ONE DIVISION, not a sigmoid and a multiply. The source reads `x * sigmoid(x)` and
// the obvious transcription is `x * (1 / (1 + exp(-x)))` -- a reciprocal and then a multiply, two
// roundings. `silu` is a single ATen op, and inductor lowers it directly to `x / (1 + exp(-x))`,
// which rounds once. Measured over the captured inputs against the compiled program's scales: one
// division 36 of 544, reciprocal-then-multiply 155 -- and the first GPU gate of the multiply form
// scored 147, which is what identified it. The same class as the `absmax / 448` entry and the
// opposite direction: there the source divides and the program multiplies, here the source's shape
// suggests a multiply and the program divides. Neither is a habit; both are read off the reference.
__device__ __forceinline__ float dm_silu(float x) { return x / (1.f + dm_exp(-x)); }

// ---- silu(gate) * up, then the per-128-group e4m3 quant --------------------------------------
// One WAVE owns one quant group, so the amax is a single cross-lane reduction with no LDS and no
// barrier -- the shape `dflash_conv.hip` already uses, because it is the same quant.
extern "C" __global__ __launch_bounds__(DM_NT) void dflash_silu_quant_g128(
    const __bf16 *__restrict__ gate_up,   // [T, 2 * N], gate then up
    unsigned char *__restrict__ q,        // [T, N] e4m3
    float *__restrict__ scales,           // [T, N / 128] fp32
    int t_rows, int n) {
  const int wave = (int)(blockIdx.x * DM_NW + (threadIdx.x >> 5));
  const int lane = threadIdx.x & 31;
  const int ngq = n / DM_QG;
  const int t = wave / ngq, qg = wave - t * ngq;
  if (t >= t_rows) return;
  const int j = qg * DM_QG + lane * DM_VEC;
  const bf16x4 gv = *(const bf16x4 *)(gate_up + (long)t * 2 * n + j);
  const bf16x4 uv = *(const bf16x4 *)(gate_up + (long)t * 2 * n + n + j);

  // The activation is NOT rounded here, and this is the one place the conv's rule inverts. The
  // amax sees bf16 only when the producer's bf16 tensor is MATERIALISED in the graph; in the conv
  // it is, because `fused_add_rms_norm` stores the residual. Here the quant is the activation's
  // only consumer, so inductor keeps it in fp32 across the fused region and the amax -- and the
  // division below -- see the FP32 value. Measured, not assumed: over the captured inputs, fp32
  // reproduces the compiled program's bytes exactly and bf16 misses 1511 of 69632.
  float ob[DM_VEC];
  float a = 0.f;
#pragma unroll
  for (int e = 0; e < DM_VEC; ++e) {
    ob[e] = dm_silu(dm_f32(gv[e])) * dm_f32(uv[e]);
    a = fmaxf(a, fabsf(ob[e]));
  }
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) a = fmaxf(a, __shfl_xor(a, d, 32));
  // TWO SCALES FROM ONE AMAX, and they are not the same number. Prod's `_6` stores the group's
  // fp32 amax into a **bf16** buffer (`buf18 = empty_strided_cuda((s72, 136, 1), ...,
  // torch.bfloat16)`) as well as into the fp32 `buf21`; `_7` re-reads the BF16 one to build the
  // divisor, while the GEMM multiplies by the fp32 one. So the quantised bytes and the scale the
  // epilogue applies come from amaxes that differ by up to a bf16 ULP.
  //
  // IT MATTERS HERE AND NOWHERE ELSE. `_0`/`_1` (`act_quant`) and `_4`/`_5` (the conv's quant)
  // take their amax over a MATERIALISED bf16 tensor, so `bf16(amax) == amax` exactly and the two
  // scales coincide. This one's amax is over the fp32 `silu(gate) * up` that the fused region
  // never rounds, so the round trip is lossy and the asymmetry is real.
  //
  // Attributed on the host before any GPU time (`tools/capture/projection_attribution.py`, the
  // body table): with this line and `act_quant`'s reciprocal, row 0 of `attn.l1.v` -- the one row
  // that exercises every op of the body and no conv tap-1 -- is BIT-EXACT against prod's own on
  // 3/3 prompts, and taking either of them back out breaks it on at least 2 of 3.
  const float amax_bf = (float)(__bf16)a;
  const float sc = fmaxf(a * (1.f / 448.f), 1.f / (448.f * 512.f));
  const float sc_div = fmaxf(amax_bf * (1.f / 448.f), 1.f / (448.f * 512.f));
  if (lane == 0) scales[(long)t * ngq + qg] = sc;
  unsigned int packed = 0;
#pragma unroll
  for (int e = 0; e < DM_VEC; e += 2) {
    const float u = fminf(fmaxf(ob[e] / sc_div, -448.f), 448.f);
    const float v = fminf(fmaxf(ob[e + 1] / sc_div, -448.f), 448.f);
    packed |= (__builtin_amdgcn_cvt_pk_fp8_f32(u, v, 0u, false) & 0xffffu) << (e * 8);
  }
  *(unsigned int *)(q + (long)t * n + j) = packed;
}

// ---- the conv's `finish`, then the residual add and the RMS norm ------------------------------
//
// TWO SYMBOLS, and the pair IS prod's fp32 residual. The drafter's compiled piece spans from one
// attention to the next, and inside it the residual between the layer's two halves is an FP32
// buffer -- `buf6 = empty_strided_cuda((s72, 5120), ..., torch.float32)` written by the attention
// half and handed to the MLP half as `in_out_ptr0` (`buf24 = buf6; del buf6  # reuse`). Only at
// the region's edge, where the eager attention reads it, is it rounded to bf16 (`buf25`, a
// graph output). So:
//
//   `_open`   resid bf16 -> res_out FP32   the ATTENTION half: the conv after `o_proj`
//   `_close`  resid FP32 -> res_out bf16   the MLP half: the conv after `down_proj`
//
// Saber stored bf16 at both until F29, which rounded the residual once a layer more than prod.
//
// THE ARITHMETIC, read off the drafter's OWN generated Triton and not off the Python
// (`triton_red_fused_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_*_2` and `_8`,
// kept at `an internal evidence capture (not shipped)`):
//
//   o       = the conv, in FP32                `tmp22 = tmp21.to(tl.float32)` is a NO-OP: the
//                                              bf16 round a MATERIALISED conv output takes
//                                              (`dflash_conv.hip`, whose consumer is a quant) is
//                                              elided here, because the fused region's only real
//                                              store is the residual
//   x       = o + f32(resid)                   fp32, and the residual store
//   res_out = x  (`_open`) | bf16(x) (`_close`)
//   var     = mean over H of x*x               fp32
//   out     = bf16((x * rsqrt(var + eps)) * f32(weight))     ONE rounding, at the store
//
// **The epilogue rounds ONCE.** `ir.ops.fused_add_rms_norm`'s Python is
// `x.to(weight.dtype) * weight` and the reference server's C++ `fused_add_rms_norm` kernel writes
// `((scalar_t)(x * s_variance)) * weight[i]`, so both round twice -- and this kernel did too,
// which is why a bench against an ISOLATED `torch.compile` of the pair read identical while the
// chain read 47.5% of `chain.l0.v`'s bytes. In the drafter's own region the custom op is
// decomposed (the reference server's piecewise compilation leaves `CustomOp.enabled()` false), inductor keeps the
// region in fp32 and rounds at the store, and there is not one `tl.bfloat16` in the whole
// generated module. Attributed on the host to the BYTE before a second of GPU time:
// `tools/capture/projection_attribution.py` reproduces the gate's own 7,784 / 7,699 / 7,816 from
// the double rounding alone and gives 0 / 0 / 1 with it removed.
//
// One workgroup per row, because the variance is a reduction over the whole row and nothing smaller
// owns it. The row is read twice -- once to reduce, once to scale -- which is what the reference
// does too; at 5120 bf16 it is 10 KiB and lands in cache.

// The residual's two widths, as typed vector accesses. `bf16 -> f32` is exact and `f32 -> f32` is
// a copy, so these change WHICH BYTES the residual occupies and never what the norm computes.
__device__ __forceinline__ void dm_load4(const __bf16 *__restrict__ p, float *o) {
  const bf16x4 v = *(const bf16x4 *)p;
#pragma unroll
  for (int i = 0; i < 4; ++i) o[i] = dm_f32(v[i]);
}
__device__ __forceinline__ void dm_load4(const float *__restrict__ p, float *o) {
  const f32x4 v = *(const f32x4 *)p;
#pragma unroll
  for (int i = 0; i < 4; ++i) o[i] = v[i];
}
__device__ __forceinline__ void dm_store4(__bf16 *__restrict__ p, const float *o) {
  bf16x4 v;
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = (__bf16)o[i];
  *(bf16x4 *)p = v;
}
__device__ __forceinline__ void dm_store4(float *__restrict__ p, const float *o) {
  f32x4 v;
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = o[i];
  *(f32x4 *)p = v;
}

template <typename RIN, typename ROUT>
__device__ __forceinline__ void dm_conv_add_rms(
    const __bf16 *__restrict__ x,          // [T, H] the conv's input
    const __bf16 *__restrict__ coeff,      // [T, 2, taps, G]
    const __bf16 *__restrict__ base,       // [2, taps, H]
    const RIN *__restrict__ resid,         // [T, H] bf16 (`_open`) or fp32 (`_close`)
    const __bf16 *__restrict__ weight,     // [H]
    __bf16 *__restrict__ out,              // [T, H]
    ROUT *__restrict__ res_out,            // [T, H] fp32 (`_open`) or bf16 (`_close`)
    int t_rows, int h, int groups, int block_size, int side, float eps) {
  const int t = (int)blockIdx.x;
  if (t >= t_rows) return;
  const int tid = (int)threadIdx.x;
  constexpr int TAPS = 2, GS = 16;
  const int pos = t & (block_size - 1);
  __shared__ float red[DM_ROW];

  // Pass 1: the conv, the residual add, the residual store, and the sum of squares. The FP32 sum
  // is kept in registers, because pass 2 needs THAT and `_close` does not store it: the reference
  // scales `x` -- the fp32 value -- and rounds only at the store. (`_open`'s fp32 `res_out` IS
  // that value, and prod's own kernel re-reads it in pass 2; an fp32 round trip is a copy, so the
  // register array is the same arithmetic with one fewer load.)
  constexpr int MAXV = 32;                  // H <= DM_ROW * 4 * MAXV / 4; the rules refuse wider
  float xv[MAXV];
  int nv = 0;
  float acc = 0.f;
  for (int e = tid * 4; e < h; e += DM_ROW * 4) {
    const int g = e / GS;
    float o[4] = {0.f, 0.f, 0.f, 0.f};
    // NO CONTRACTION IN THE CONV, and here it is load-bearing where `dflash_conv.hip` can let the
    // default stand. The reference multiplies and then adds (`tmp13 = tmp7 * tmp12` feeds
    // `tmp20 = tmp13 * mask` feeds `tmp21 = tmp4 + tmp20`), so the tap-1 product is ROUNDED
    // before it reaches the sum; `o[i] += c * x` under hipcc's HIP default fuses that into an FMA
    // and keeps the product at full width. The standalone conv gets away with it because its
    // output is stored bf16 and a 1-ULP fp32 difference survives that with probability 2^-16.
    // This one's output stays FP32 all the way into the next layer's residual.
    {
#pragma clang fp contract(off)
#pragma unroll
      for (int tap = 0; tap < TAPS; ++tap) {
        if (tap > 0 && pos < tap) continue;
        const bf16x4 xin = *(const bf16x4 *)(x + (long)(t - tap) * h + e);
        const bf16x4 bv = *(const bf16x4 *)(base + ((long)side * TAPS + tap) * h + e);
        const float d = dm_f32(coeff[(((long)t * 2 + side) * TAPS + tap) * groups + g]);
#pragma unroll
        for (int i = 0; i < 4; ++i) o[i] += (dm_f32(bv[i]) + d) * dm_f32(xin[i]);
      }
    }
    float rv[4];
    dm_load4(resid + (long)t * h + e, rv);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      // The conv's output is NOT rounded here: inductor keeps the fused region in fp32.
      const float sv = o[i] + rv[i];
      xv[nv + i] = sv;
      acc += sv * sv;
    }
    nv += 4;
    dm_store4(res_out + (long)t * h + e, &xv[nv - 4]);
  }
  red[tid] = acc;
  __syncthreads();
  for (int s = DM_ROW / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  // `torch.rsqrt` of the MEAN. Which rsqrt is a gate question: ATen's is correctly rounded (the
  // attn_glue LEDGER entry), Triton's `tl.rsqrt` is the ~1-ULP `v_rsq_f32`, and this is the
  // inductor path -- so the hardware one, with the correctly-rounded form one edit away.
  const float r = __builtin_amdgcn_rsqf(red[0] / (float)h + eps);
  __syncthreads();

  // Pass 2: scale the FP32 x by the weight, rounding where the reference rounds -- ONCE.
  nv = 0;
  for (int e = tid * 4; e < h; e += DM_ROW * 4) {
    const bf16x4 wv = *(const bf16x4 *)(weight + e);
    bf16x4 ov;
#pragma unroll
    for (int i = 0; i < 4; ++i) ov[i] = (__bf16)((xv[nv + i] * r) * dm_f32(wv[i]));
    nv += 4;
    *(bf16x4 *)(out + (long)t * h + e) = ov;
  }
}

// The ATTENTION half: it OPENS the fp32 residual prod carries across the layer.
extern "C" __global__ __launch_bounds__(DM_ROW) void dflash_conv_add_rms_t2g16_open(
    const __bf16 *__restrict__ x, const __bf16 *__restrict__ coeff,
    const __bf16 *__restrict__ base, const __bf16 *__restrict__ resid,
    const __bf16 *__restrict__ weight, __bf16 *__restrict__ out, float *__restrict__ res_out,
    int t_rows, int h, int groups, int block_size, int side, float eps) {
  dm_conv_add_rms(x, coeff, base, resid, weight, out, res_out, t_rows, h, groups, block_size,
                  side, eps);
}

// The MLP half: it CLOSES it, at the seam where prod's compiled region ends and the eager
// attention reads the residual back as bf16.
extern "C" __global__ __launch_bounds__(DM_ROW) void dflash_conv_add_rms_t2g16_close(
    const __bf16 *__restrict__ x, const __bf16 *__restrict__ coeff,
    const __bf16 *__restrict__ base, const float *__restrict__ resid,
    const __bf16 *__restrict__ weight, __bf16 *__restrict__ out, __bf16 *__restrict__ res_out,
    int t_rows, int h, int groups, int block_size, int side, float eps) {
  dm_conv_add_rms(x, coeff, base, resid, weight, out, res_out, t_rows, h, groups, block_size,
                  side, eps);
}

// ---- the ONE standalone RMS norm: layer 0's `input_layernorm` ----------------------------------
//
// `qwen3_dflash2.py:164` -- `if residual is None: residual = hidden_states; hidden_states =
// self.input_layernorm(hidden_states)`. Layer 0 is the only layer that takes that branch, so the
// drafter runs exactly one norm with no residual to add and no conv in front of it to fuse into;
// the census sees it as `triton_red_fused_mul_rms_norm_1`, n=1, grid 2048 x 256, 2.24 us. Every
// other norm in the drafter is the tail of a `conv_add_rms` above.
//
// `ir.ops.rms_norm` is `fused_add_rms_norm` with the add and the residual store deleted, and this
// kernel is the one above with the same two lines deleted -- the same reduction, the same rsqrt,
// and the SAME SINGLE ROUNDING at the store. Prod's own generated program for this one is
// `triton_red_fused__to_copy_embedding_mul_rms_norm_0` (the embedding is fused into it, which is
// also why `chain.embed` is the bf16 tensor the chain compares against): `tmp22 = tmp16 * tmp21`,
// `tmp23 = tmp22.to(tl.float32)` -- a NO-OP -- `tmp25 = tmp23 * tmp24`, and one bf16 store.
// THE DIFFERENCE THAT MATTERS: the
// conv's version had to keep the fp32 row in registers because its fp32 sum is not recoverable
// from the bf16 residual it stores. Here the fp32 row IS recoverable -- `x` is an unmodified bf16
// input and bf16 -> fp32 is exact -- so pass 2 re-reads it for 32 fewer VGPRs, which is also the
// shape of inductor's own `_red_` kernel: it loops the row twice and reloads.
extern "C" __global__ __launch_bounds__(DM_ROW) void dflash_rms_norm(
    const __bf16 *__restrict__ x,          // [T, H]
    const __bf16 *__restrict__ weight,     // [H]
    __bf16 *__restrict__ out,              // [T, H]
    int t_rows, int h, float eps) {
  const int t = (int)blockIdx.x;
  if (t >= t_rows) return;
  const int tid = (int)threadIdx.x;
  __shared__ float red[DM_ROW];

  // Pass 1: the sum of squares of the FP32 row. `x.to(torch.float32)` is exact from bf16.
  float acc = 0.f;
  for (int e = tid * 4; e < h; e += DM_ROW * 4) {
    const bf16x4 iv = *(const bf16x4 *)(x + (long)t * h + e);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float sv = dm_f32(iv[i]);
      acc += sv * sv;
    }
  }
  red[tid] = acc;
  __syncthreads();
  for (int s = DM_ROW / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  const float r = __builtin_amdgcn_rsqf(red[0] / (float)h + eps);
  __syncthreads();

  // Pass 2: the same epilogue the fused version runs, with the same one rounding.
  for (int e = tid * 4; e < h; e += DM_ROW * 4) {
    const bf16x4 iv = *(const bf16x4 *)(x + (long)t * h + e);
    const bf16x4 wv = *(const bf16x4 *)(weight + e);
    bf16x4 ov;
#pragma unroll
    for (int i = 0; i < 4; ++i) ov[i] = (__bf16)((dm_f32(iv[i]) * r) * dm_f32(wv[i]));
    *(bf16x4 *)(out + (long)t * h + e) = ov;
  }
}

// ---- the same quant, with the amax exposed -----------------------------------------------------
// A DIAGNOSTIC entry point, not a route: identical arithmetic to `dflash_silu_quant_g128` with one
// extra store, so the question "which amax did each side compute?" can be asked without inferring
// it backwards through `a * (1/448)`, which is lossy. `scales` still gets the scale; `amax_out`
// gets the raw pre-scale maximum. The rules never select this.
extern "C" __global__ __launch_bounds__(DM_NT) void dflash_silu_quant_g128_amax(
    const __bf16 *__restrict__ gate_up,
    unsigned char *__restrict__ q,
    float *__restrict__ scales,
    float *__restrict__ amax_out,
    int t_rows, int n) {
  const int wave = (int)(blockIdx.x * DM_NW + (threadIdx.x >> 5));
  const int lane = threadIdx.x & 31;
  const int ngq = n / DM_QG;
  const int t = wave / ngq, qg = wave - t * ngq;
  if (t >= t_rows) return;
  const int j = qg * DM_QG + lane * DM_VEC;
  const bf16x4 gv = *(const bf16x4 *)(gate_up + (long)t * 2 * n + j);
  const bf16x4 uv = *(const bf16x4 *)(gate_up + (long)t * 2 * n + n + j);
  float ob[DM_VEC];
  float a = 0.f;
#pragma unroll
  for (int e = 0; e < DM_VEC; ++e) {
    ob[e] = dm_silu(dm_f32(gv[e])) * dm_f32(uv[e]);
    a = fmaxf(a, fabsf(ob[e]));
  }
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) a = fmaxf(a, __shfl_xor(a, d, 32));
  const float sc = fmaxf(a * (1.f / 448.f), 1.f / (448.f * 512.f));
  const float sc_div = fmaxf((float)(__bf16)a * (1.f / 448.f), 1.f / (448.f * 512.f));
  if (lane == 0) {
    scales[(long)t * ngq + qg] = sc;
    amax_out[(long)t * ngq + qg] = a;
  }
  unsigned int packed = 0;
#pragma unroll
  for (int e = 0; e < DM_VEC; e += 2) {
    const float u = fminf(fmaxf(ob[e] / sc_div, -448.f), 448.f);
    const float v = fminf(fmaxf(ob[e + 1] / sc_div, -448.f), 448.f);
    packed |= (__builtin_amdgcn_cvt_pk_fp8_f32(u, v, 0u, false) & 0xffffu) << (e * 8);
  }
  *(unsigned int *)(q + (long)t * n + j) = packed;
}

// ---- the blockscale GEMM's OWN activation quant -------------------------------------------------
//
// The launch nobody declared for three rounds, and the reason is worth keeping: prod runs it inside
// the layer's quant method, so a census that counts GEMMs never sees it, and a STAGED replay seeds
// past it. It showed the first time a replay tried to CHAIN the proposal (workstream F round 21).
//
// Six launches a decode step, ONE kernel at two shapes:
//   `aux_hidden_projection.quant`   [8, 25600]  the five tapped target layers, concatenated
//   `layer.{0..4}.attn.o_proj.quant` [8, 4096]  the attention's output, per layer
// Every other blockscale GEMM in the drafter is fed by a declared producer -- `grouped_conv_quant`
// feeds `attn.qkv` and `mlp.gate_up`, `silu_quant` feeds `mlp.down` -- and these two are the sites
// where the producer is on the other side of an op that writes bf16.
//
// THE REFERENCE IS `QuantFP8.forward_native` -> `_quantize_group_native`, not the reference's per-group
// quant and not `ops.scaled_fp8_quant`. Which one runs is decided by `CustomOp.enabled()`, and with
// The reference server's inductor-piecewise compilation the custom ops default OFF, so BOTH the eager site
// (`combine_hidden_states` calls `model.fc` outside the compiled region) and the compiled one
// (`o_proj`, inside the drafter's graph) take the native path. `dflash_silu_quant_g128` above is
// written against the same function and its LEDGER entry is the evidence for the two roundings
// below.
//
//   absmax = x_grouped.abs().max(-1).float()    over the BF16 tensor -- exact, both sites
//   scales = max(absmax / 448, 1/(448*512))     a REAL DIVISION -- see below
//   q      = clamp(f32(x) / scales, +-448).to(e4m3)      divide by a TENSOR, so a real division
//
// **`absmax / 448` IS A DIVISION HERE AND A RECIPROCAL MULTIPLY IN `silu_quant`, and the two are
// not in conflict.** The LEDGER records `a * (1/448)` for the MLP's quant, and that entry is about
// an INDUCTOR-GENERATED Triton kernel, which folds a constant divisor into a multiply. This site's
// reference is the same source line reached a different way: `QuantFP8.forward_native` run by ATen,
// where `div` by a python float is a real `v_div` and keeps the extra rounding. Measured on the
// host with no card against prod's own captured `quant.*.q` / `quant.*.s` (2026-09-21, F26, r26
// capture, three prompts x six sites): the division is IDENTICAL of 204800 + 6400 bytes at the aux
// shape and 32768 + 1024 at each layer's, and the reciprocal multiply differs in 124-165 of the
// quantised bytes and 868-944 of the scales. Both arms are here; the measurement is what picks.
//
// THE ONE DIFFERENCE FROM `silu_quant` IN THE OTHER DIRECTION, and it is the conv's rule rather
// than the MLP's: here the producer's bf16 tensor IS materialised (the target's hidden states; the
// attention's output), so the amax sees BF16 and not the fp32 an unmaterialised fused region would
// carry. Reading the input as bf16 is therefore not an approximation of prod, it is prod.
//
// `_recip` and `_scmul` below are the CONTROLS for the two roundings, and they are not routes:
// measured 95 / 26 / 15 / 10 / 13 of the quantised bytes for `_scmul` (which leaves the SCALES
// identical, because it changes only how the scale is applied) and the numbers above for `_recip`.
#define DQ_BODY(SCALE_EXPR, APPLY)                                                                 \
  const int wave = (int)(blockIdx.x * DM_NW + (threadIdx.x >> 5));                                 \
  const int lane = threadIdx.x & 31;                                                               \
  const int ngq = n / DM_QG;                                                                       \
  const int t = wave / ngq, qg = wave - t * ngq;                                                   \
  if (t >= t_rows) return;                                                                         \
  const int j = qg * DM_QG + lane * DM_VEC;                                                        \
  const bf16x4 xv = *(const bf16x4 *)(x + (long)t * n + j);                                        \
  float ob[DM_VEC];                                                                                \
  float a = 0.f;                                                                                   \
  _Pragma("unroll") for (int e = 0; e < DM_VEC; ++e) {                                             \
    ob[e] = dm_f32(xv[e]);                                                                         \
    a = fmaxf(a, fabsf(ob[e]));                                                                    \
  }                                                                                                \
  _Pragma("unroll") for (int d = 1; d < 32; d <<= 1) a = fmaxf(a, __shfl_xor(a, d, 32));           \
  const float sc = fmaxf(SCALE_EXPR, 1.f / (448.f * 512.f));                                       \
  if (lane == 0) scales[(long)t * ngq + qg] = sc;                                                  \
  unsigned int packed = 0;                                                                         \
  _Pragma("unroll") for (int e = 0; e < DM_VEC; e += 2) {                                          \
    const float u = fminf(fmaxf(APPLY(ob[e]), -448.f), 448.f);                                     \
    const float v = fminf(fmaxf(APPLY(ob[e + 1]), -448.f), 448.f);                                 \
    packed |= (__builtin_amdgcn_cvt_pk_fp8_f32(u, v, 0u, false) & 0xffffu) << (e * 8);             \
  }                                                                                                \
  *(unsigned int *)(q + (long)t * n + j) = packed;

#define DQ_DIV_BY_SC(V) ((V) / sc)

// The route: `absmax / 448` exactly as the source writes it.
extern "C" __global__ __launch_bounds__(DM_NT) void dflash_act_quant_g128(
    const __bf16 *__restrict__ x,         // [T, N] bf16, the producer's own materialised tensor
    unsigned char *__restrict__ q,        // [T, N] e4m3
    float *__restrict__ scales,           // [T, N / 128] fp32
    int t_rows, int n) {
  DQ_BODY(a / 448.f, DQ_DIV_BY_SC)
}

// CONTROL 1: the reciprocal multiply an inductor-compiled version of this line uses, and which
// `dflash_silu_quant_g128` above is written with. Taking that lowering on faith here would have
// cost 124-165 quantised bytes and 868-944 scales per prompt -- the LEDGER's entry is about a
// Triton kernel and this reference is ATen, and the flag is a property of the REFERENCE.
extern "C" __global__ __launch_bounds__(DM_NT) void dflash_act_quant_g128_recip(
    const __bf16 *__restrict__ x, unsigned char *__restrict__ q, float *__restrict__ scales,
    int t_rows, int n) {
  DQ_BODY(a * (1.f / 448.f), DQ_DIV_BY_SC)
}

// CONTROL 2: the scale APPLIED as a reciprocal multiply. `x_grouped / scales` divides by a TENSOR,
// which neither eager nor inductor turns into a reciprocal, so this one must differ.
extern "C" __global__ __launch_bounds__(DM_NT) void dflash_act_quant_g128_scmul(
    const __bf16 *__restrict__ x, unsigned char *__restrict__ q, float *__restrict__ scales,
    int t_rows, int n) {
#define APPLY_RSC(V) ((V) * rsc)
  const int wave = (int)(blockIdx.x * DM_NW + (threadIdx.x >> 5));
  const int lane = threadIdx.x & 31;
  const int ngq = n / DM_QG;
  const int t = wave / ngq, qg = wave - t * ngq;
  if (t >= t_rows) return;
  const int j = qg * DM_QG + lane * DM_VEC;
  const bf16x4 xv = *(const bf16x4 *)(x + (long)t * n + j);
  float ob[DM_VEC];
  float a = 0.f;
#pragma unroll
  for (int e = 0; e < DM_VEC; ++e) {
    ob[e] = dm_f32(xv[e]);
    a = fmaxf(a, fabsf(ob[e]));
  }
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) a = fmaxf(a, __shfl_xor(a, d, 32));
  const float sc = fmaxf(a / 448.f, 1.f / (448.f * 512.f));
  const float rsc = 1.f / sc;
  if (lane == 0) scales[(long)t * ngq + qg] = sc;
  unsigned int packed = 0;
#pragma unroll
  for (int e = 0; e < DM_VEC; e += 2) {
    const float u = fminf(fmaxf(APPLY_RSC(ob[e]), -448.f), 448.f);
    const float v = fminf(fmaxf(APPLY_RSC(ob[e + 1]), -448.f), 448.f);
    packed |= (__builtin_amdgcn_cvt_pk_fp8_f32(u, v, 0u, false) & 0xffffu) << (e * 8);
  }
  *(unsigned int *)(q + (long)t * n + j) = packed;
#undef APPLY_RSC
}
