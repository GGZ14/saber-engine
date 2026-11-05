// The DFlash2 drafter's grouped convolution, and the per-group fp8 quant fused into it.
//
// NOT a port, again, and this time not even from one reference function: prod runs these as
// INDUCTOR-COMPILED Triton, fused out of `_grouped_conv` (`qwen3_dflash2.py:26`) and
// `QuantFP8._quantize_group_native` (`input_quant_fp8.py:228`). The census counted the family at
// ~46 launches and ~110 us per decode step:
//   triton_red_fused__to_copy_abs_add_arange_bitwise_and_constant_pad_nd_*   9 + 1   13.2 us
//   triton_poi_fused__to_copy_add_arange_bitwise_and_constant_pad_nd_*       9 + 1   17.3 us
//   triton_poi_fused__to_copy_clamp_div_preshuffle_ge_*                      4 + 1    7.1 us
//   triton_red_fused_add_arange_bitwise_and_constant_pad_nd_*               5 + 5    44.4 us
//   triton_poi_fused_add_arange_bitwise_and_constant_pad_nd_*                8 + 2   16.4 us
//
// THE CENSUS'S OWN KERNEL NAMES ARE THE EVIDENCE FOR FUSING THEM. `abs` and a reduction appear in
// the SAME kernel as the conv's `arange`/`bitwise_and`/`constant_pad_nd`, so inductor already found
// that the conv and the quant's amax share an axis. What it cannot do is finish the job: the scale
// is a reduction over a group and the quantised store needs it, so inductor computes the conv twice
// -- once to reduce, once to divide -- across two kernels. One workgroup that owns a whole
// 128-wide group does both in one pass, which is the fusion rule exactly: it owns the reduction
// axis, and it introduces no global sync the pair did not already pay.
//
// THE ARITHMETIC, and every dtype in it is load-bearing:
//   c[tap]      = f32(base[side][tap][h]) + f32(delta[t][side][tap][g])   -- NOT rounded to bf16
//   o           = sum over taps of c[tap] * x[t - tap][h], the tap > 0 terms gated by
//                 (t & (block_size - 1)) >= tap, which is what makes the conv restart per block
//   out         = bf16(o)                                                 -- one round, at the store
// and then, for the quant variant, over each group of 128 of THOSE bf16 values:
//   absmax      = f32(max |out|)              -- the max is over the BF16-ROUNDED conv output
//   scale       = max(absmax * (1/448), 1/(448*512))   -- a RECIPROCAL MULTIPLY: see below
//   q           = e4m3_rne(clamp(f32(out) / scale, +-448))
// Intermediates stay fp32 because inductor propagates fp32 through a fused region and rounds only
// at a real bf16 store; eager torch would round at every op and give a different answer, and the
// capture dumps both so the gate can say which prod runs. The two DIVISIONS are the ones
// `the reference`'s own quant does as multiplies by a reciprocal (mxfp4_quant.hip:99) -- the same trap
// the LEDGER records for the W4A8 producers, in the other direction.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

typedef __bf16 bf16x4 __attribute__((ext_vector_type(4)));

#define DC_TAPS   2     // conv_kernel_size
#define DC_GS    16     // conv_group_size
#define DC_QG   128     // the activation-quant group: GroupShape(1, 128)
#define DC_VEC    4     // bf16 per thread: one 8-byte load, and 32 lanes cover one quant group
#define DC_NW     4     // waves per workgroup, so one workgroup covers DC_NW quant groups
#define DC_NT   (DC_NW * 32)

__device__ __forceinline__ float dc_f32(__bf16 v) { return (float)v; }

// One wave's amax over its own 128 values. `max` is order-independent and exact, so unlike a SUM
// this needs no tree matching -- any reduction shape gives the reference's answer bit for bit.
__device__ __forceinline__ float dc_wave_amax(float v) {
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) v = fmaxf(v, __shfl_xor(v, d, 32));
  return v;
}

// The conv itself: `DC_VEC` consecutive outputs of row t, all in one conv group because
// DC_GS % DC_VEC == 0.
template <int TAPS, int GS, int VEC>
__device__ __forceinline__ void dc_conv(const __bf16 *__restrict__ x,
                                        const __bf16 *__restrict__ coeff,
                                        const __bf16 *__restrict__ base, int t, int h, int H,
                                        int groups, int block_size, int side, float *o) {
  const int g = h / GS;
  const int pos = t & (block_size - 1);          // prod's `arange` & `bitwise_and`: a power-of-two
                                                 // block_size is the branch `_grouped_conv` takes
#pragma unroll
  for (int e = 0; e < VEC; ++e) o[e] = 0.f;
#pragma unroll
  for (int tap = 0; tap < TAPS; ++tap) {
    if (tap > 0 && pos < tap) continue;          // the `(position >= tap)` mask AND the zero pad
    const bf16x4 xv = *(const bf16x4 *)(x + (long)(t - tap) * H + h);
    const bf16x4 bv = *(const bf16x4 *)(base + ((long)side * TAPS + tap) * H + h);
    const float d = dc_f32(coeff[(((long)t * 2 + side) * TAPS + tap) * groups + g]);
#pragma unroll
    for (int e = 0; e < VEC; ++e) o[e] += (dc_f32(bv[e]) + d) * dc_f32(xv[e]);
  }
}

// ---- conv only: the `finish` calls, whose output feeds a norm rather than a GEMM -------------
extern "C" __global__ __launch_bounds__(DC_NT) void dflash_grouped_conv_t2g16(
    const __bf16 *__restrict__ x, const __bf16 *__restrict__ coeff,
    const __bf16 *__restrict__ base, __bf16 *__restrict__ out,
    int t_rows, int h, int groups, int block_size, int side) {
  const int idx = (int)(blockIdx.x * DC_NT + threadIdx.x) * DC_VEC;
  const int t = idx / h, hh = idx - t * h;
  if (t >= t_rows) return;
  float o[DC_VEC];
  dc_conv<DC_TAPS, DC_GS, DC_VEC>(x, coeff, base, t, hh, h, groups, block_size, side, o);
  bf16x4 ov;
#pragma unroll
  for (int e = 0; e < DC_VEC; ++e) ov[e] = (__bf16)o[e];
  *(bf16x4 *)(out + (long)t * h + hh) = ov;
}

// ---- conv + per-group fp8 quant: the `prepare` calls, whose output feeds a W8A8 GEMM ----------
// One WAVE owns one 128-wide quant group, so the amax is a single cross-lane reduction with no
// LDS and no barrier, and the quantised store happens in the same pass.
extern "C" __global__ __launch_bounds__(DC_NT) void dflash_grouped_conv_quant_t2g16(
    const __bf16 *__restrict__ x, const __bf16 *__restrict__ coeff,
    const __bf16 *__restrict__ base, unsigned char *__restrict__ q,
    float *__restrict__ scales, int t_rows, int h, int groups, int block_size, int side) {
  const int wave = (int)(blockIdx.x * DC_NW + (threadIdx.x >> 5));
  const int lane = threadIdx.x & 31;
  const int ngq = h / DC_QG;                     // quant groups per row
  const int t = wave / ngq, qg = wave - t * ngq;
  if (t >= t_rows) return;
  const int hh = qg * DC_QG + lane * DC_VEC;
  float o[DC_VEC];
  dc_conv<DC_TAPS, DC_GS, DC_VEC>(x, coeff, base, t, hh, h, groups, block_size, side, o);

  // The quant reads the conv's BF16 output, not its fp32 intermediate: the graph has a bf16 tensor
  // between the two ops and inductor's `_to_copy` keeps it. Measured both ways -- skipping the
  // round leaves 319 of 320 group scales differing from the compiled reference, keeping it leaves
  // none.
  __bf16 ob[DC_VEC];
  float a = 0.f;
#pragma unroll
  for (int e = 0; e < DC_VEC; ++e) {
    ob[e] = (__bf16)o[e];
    a = fmaxf(a, fabsf(dc_f32(ob[e])));
  }
  const float absmax = dc_wave_amax(a);
  // A RECIPROCAL MULTIPLY, not the division prod's source writes, and this one was measured rather
  // than reasoned: `_quantize_group_native` says `absmax / _FP8_MAX`, and dividing leaves 194 of
  // 320 group scales one ULP off the compiled reference while multiplying by 1/448 leaves none.
  // Inductor strength-reduces the constant divisor; the source and the program prod actually runs
  // disagree, and the program wins. (The LEDGER records the same trap in the other direction for
  // the W4A8 producers, where the reference multiplies and the reference server's source divides.)
  const float sc = fmaxf(absmax * (1.f / 448.f), 1.f / (448.f * 512.f));
  if (lane == 0) scales[(long)t * ngq + qg] = sc;
  unsigned int packed = 0;
#pragma unroll
  for (int e = 0; e < DC_VEC; e += 2) {
    // The SECOND division is a true one: with a runtime divisor the two forms are
    // indistinguishable on this data (both give the reference's e4m3 codes exactly), so this keeps
    // the source's operation.
    const float u = fminf(fmaxf(dc_f32(ob[e]) / sc, -448.f), 448.f);
    const float v = fminf(fmaxf(dc_f32(ob[e + 1]) / sc, -448.f), 448.f);
    packed |= (__builtin_amdgcn_cvt_pk_fp8_f32(u, v, 0u, false) & 0xffffu) << (e * 8);
  }
  *(unsigned int *)(q + (long)t * h + hh) = packed;
}
