// SABER-AUTHORED (workstream G2, 2026-09-23) -- NOT sliced from the reference. Included at the end
// of the generated mxfp4_quant.hip by the internal reference harness, after the reference's own
// `saber_gdn_norm_quant_impl`, so ArnqVec8, uint32_tt and ARNQ_T are the reference's.
//
// The GDN out_proj's A producer writing the FRAGMENT-TILED layout the A-tiled prefill GEMM reads.
// The reference has one instantiation of this kernel (templated on the thread count alone, store
// `q += row * N`) and prod never registers its output as tiled (`ref_arnq.py`), so in prod
// and on Saber's main the GDN out_proj is the ONE prefill GEMM on the folded (row-major A) kernel:
// 163 TF/s in chain against 198-230 for the A-tiled family (H34).
//
// The arithmetic below is the reference's line for line -- the same loads, the same 16-lane
// per-head sum of squares, the same fp32 expression and bf16 rounding, the same amax reduction,
// scale and e4m3 conversion. The ONLY change is where each 8-byte group lands: 8-group g of row r
// goes to fragment (r/16, g/2) at lane (r%16) + 16*(g%2), the single uint2 store add_rms_quant and
// silu_mul_quant use for their `tiled1` instantiations. So the bytes are a permutation of the
// reference's (gated: tile(reference q) == this q, scale identical, tools/capture/g2/).
template <int ART>
__device__ __forceinline__ void saber_gdn_norm_quant_tiled_impl(
    const __bf16 *__restrict__ x, const __bf16 *__restrict__ z, long zs,
    const __bf16 *__restrict__ w, unsigned char *__restrict__ q, float *__restrict__ scale,
    int N, float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  x += (long)row * N; z += (long)row * zs;
  const int mt = row >> 4, lrow = row & 15, kst = N >> 4;   // as in add_rms_quant's TILED arm
  constexpr int MAXG = 5;                       // N <= ART*8*5 = 10240 (3072 in this model)
  const int NV = N >> 3;                        // 8-element groups; 16 per head
  float sv[MAXG][8];
  float amax = 0.f;
#pragma unroll
  for (int ng = 0; ng < MAXG; ++ng) {
    const int g = tid + ng * ART;
    if (g >= NV) break;
    ArnqVec8 vx, vz, vw;
    vx.u4 = ((const uint4 *)x)[g];
#pragma unroll
    for (int j = 0; j < 8; ++j) sv[ng][j] = (float)vx.h[j];
    // ws-h45: main's compiled chain, pinned (EXEC section 8): x1^2 rounded, x0^2 fused onto it,
    // then x2..x7 fused in order -- the zero seed folded the way the runtime loop folds it.
    float ssq = sv[ng][1] * sv[ng][1];
    ssq = __builtin_fmaf(sv[ng][0], sv[ng][0], ssq);
#pragma unroll
    for (int j = 2; j < 8; ++j) ssq = __builtin_fmaf(sv[ng][j], sv[ng][j], ssq);
    for (int o = 8; o; o >>= 1) ssq += __shfl_xor(ssq, o, 16);
    const float inv = rsqrtf(ssq * (1.f / 128.f) + eps);
    vz.u4 = ((const uint4 *)z)[g];
    vw.u4 = ((const uint4 *)w)[g & 15];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float zz = (float)vz.h[j];
      const float sg = zz / (1.f + expf(-zz));                       // silu, fp32
      const float nv = (float)(__bf16)(((sv[ng][j] * inv) * (float)vw.h[j]) * sg);
      sv[ng][j] = nv;
      amax = fmaxf(amax, fabsf(nv));
    }
  }
  __shared__ float lds[ART / 32];
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_down(amax, o));
  if ((tid & 31) == 0) lds[tid >> 5] = amax;
  __syncthreads();
  if (tid < ART / 32) {
    float v = lds[tid];
    for (int o = ART / 64; o; o >>= 1) v = fmaxf(v, __shfl_down(v, o));
    if (tid == 0) lds[0] = v;
  }
  __syncthreads();
  const float sc = fmaxf(lds[0] * (1.f / 448.f), 1.f / (448.f * 512.f));
  const float rs = 1.f / sc;
  if (tid == 0) scale[row] = sc;
#pragma unroll
  for (int mg = 0; mg < MAXG; ++mg) {
    const int g = tid + mg * ART;
    if (g >= NV) break;
    uint32_tt lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 4; j += 2) {
      float a = fminf(fmaxf(sv[mg][j] * rs, -448.f), 448.f);
      float b = fminf(fmaxf(sv[mg][j + 1] * rs, -448.f), 448.f);
      lo |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu) << (j * 8);
    }
#pragma unroll
    for (int j = 4; j < 8; j += 2) {
      float a = fminf(fmaxf(sv[mg][j] * rs, -448.f), 448.f);
      float b = fminf(fmaxf(sv[mg][j + 1] * rs, -448.f), 448.f);
      hi |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu) << ((j - 4) * 8);
    }
    ((uint2 *)q)[(mt * kst + (g >> 1)) * 32 + lrow + 16 * (g & 1)] = make_uint2(lo, hi);
  }
}
