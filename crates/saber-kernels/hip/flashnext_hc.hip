// Flash-Next (qwen4_exp) hyper-connections and the bf16 dense GEMM family (workstream FN5).
//
// NOT sliced from a reference: upstream runs these ops in Triton (`qwen4_exp/amd/ops/hc.py`) and
// hipBLASLt. The math is upstream's, written down in `tools/capture/fn5/hc_ref.py` (the numpy
// reference every kernel here is gated against), with Saber's own fixed order inside each op:
//
//   mix           xn = bf16(gemma_rms_per_stream(x));  d = bf16(xn @ [Wdown; Winject]^T)  (N 324)
//                 lora = bf16(silu(d[:320] / 4));  g = bf16(lora @ Wup^T)                 (K 320)
//                 bi = bf16(sum_s sigmoid(g_s) * xn_s / 4)  -> the block input, + its e4m3 quant
//   combine       x' = bf16(res + blk * 2 * sigmoid(inj / 4))   (fused into the next mix's norm)
//
// TWO launches per mix (the HC launch law, docs/EXEC.md): `fn_hc_down_*` (norm, fused combine or
// expand, the down+inject GEMM split over K slices, and -- in its last-arriving workgroup per row
// block -- the slice sum, silu and the injection logits) and `fn_hc_up_*` (the K = 320 up GEMM, the
// sigmoid gate and gated mean, and -- in its last-arriving workgroup per row block -- the per-token
// e4m3 quant of the block input). Four launches per decoder layer (two mixes).
//
// Built with -ffp-contract=off (saber-kbuild's EXTRA_FLAGS): every fp32 multiply and add rounds on
// its own, so the host reference reproduces them one IEEE op at a time. The one region that needs
// the DEFAULT contraction (the bf16-output twin of `gdn_norm_quant`, which must round exactly where
// the contracted mxfp4_quant producer does) says so with a pragma.
//
// Loads past M / N are CLAMPED to the last real row and never stored (the predicated-staging law).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float floatx8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ float bf(float x) { return (float)(__bf16)x; }  // one RNE rounding
__device__ __forceinline__ float ld(const __bf16 *p) { return (float)*p; }
__device__ __forceinline__ floatx8 vg_wmma(bf16x8 a, bf16x8 b, floatx8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}
union FnVec8 { uint4 u4; __bf16 h[8]; bf16x8 v; };

// ------------------------------------------------------------------------------ constants
#define FN_H 2560                  // hidden per stream
#define FN_S 4                     // hc_count
#define FN_W (FN_H * FN_S)         // 10,240
#define FN_R 320                   // hc_lowrank
#define FN_NI 4                    // injection logits
#define FN_PSTR 336                // partial row stride (21 column tiles x 16)
#define FN_A_WAVES 7               // down: 21 tiles = 3 per wave
#define FN_A_TPW 3

// ------------------------------------------------------------------------------ exp / sigmoid
// hc_ref.hexp, op for op: clamp, Cody-Waite with a 16-bit ln2_hi, degree-7 Horner, ldexpf. The
// constants are DOUBLE literals rounded once to float, as numpy's F32(double) rounds them.
__device__ __forceinline__ float fn_hexp(float t) {
  t = fminf(fmaxf(t, -80.f), 80.f);
  const float kn = rintf(t * (float)1.4426950408889634);
  float r = t - kn * (float)0.693145751953125;
  r = r - kn * (float)1.4286068203094173e-06;
  float p = (float)(1.0 / 5040);
  p = p * r + (float)(1.0 / 720);
  p = p * r + (float)(1.0 / 120);
  p = p * r + (float)(1.0 / 24);
  p = p * r + (float)(1.0 / 6);
  p = p * r + 0.5f;
  p = p * r + 1.0f;
  p = p * r + 1.0f;
  return ldexpf(p, (int)kn);
}
__device__ __forceinline__ float fn_sigmoid(float v) { return 1.f / (1.f + fn_hexp(-v)); }
__device__ __forceinline__ float fn_silu(float v) { return v * fn_sigmoid(v); }

// FN49b: `__shfl_xor(v, o, 32)` for o in 1..16 on DPP16 `row_xmask` (lane ^ 1/2/4/8 inside a
// 16-lane row) or `v_permlanex16` with the identity select (lane ^ 16) instead of `ds_bpermute`:
// the same partner lane, so the same value (FN49a's `mf_dpp_xor`, the MoE route's). Every lane of
// the wave must be active (the callers' chains are wave-uniform).
__device__ __forceinline__ float fn_dpp_xor(float v, int o) {
  const int x = __builtin_bit_cast(int, v);
  int y;
  switch (o) {
    case 16: y = __builtin_amdgcn_permlanex16(x, x, 0x76543210u, 0xfedcba98u, false, false); break;
    case 8: y = __builtin_amdgcn_update_dpp(x, x, 0x168, 0xf, 0xf, false); break;
    case 4: y = __builtin_amdgcn_update_dpp(x, x, 0x164, 0xf, 0xf, false); break;
    case 2: y = __builtin_amdgcn_update_dpp(x, x, 0x162, 0xf, 0xf, false); break;
    default: y = __builtin_amdgcn_update_dpp(x, x, 0x161, 0xf, 0xf, false); break;
  }
  return __builtin_bit_cast(float, y);
}

// The combine's per-stream gain, 2*sigmoid(inj/4), from the bf16 logit.
__device__ __forceinline__ float fn_gain(const __bf16 *inj, int row, int s) {
  return 2.f * fn_sigmoid(ld(inj + (long)row * FN_NI + s) * 0.25f);
}

// ------------------------------------------------------------------------------ the stream x
// MODE 0 MIX      x is the materialized stream `src` [M, W]
// MODE 1 EXPAND   x_s = the embedding row `src` [M, H] for every stream (layer 0's repeat(1, 4))
// MODE 2 COMBINE  x_s = bf16(res + blk * gain_s): `src` = res [M, W], `blk` [M, H]
// MODE 3 RR       (FN37) the combine's x' and rr were written by the block-output PRODUCER:
//                 `src` = x' [M, W] (read as MODE 0), rr [M, 4] read, no norm prologue
// Eight consecutive elements of stream s, row `row`, at in-stream offset `i` (a multiple of 8).
template <int MODE>
__device__ __forceinline__ void fn_x8(const __bf16 *__restrict__ src, const __bf16 *__restrict__ blk,
                                      float gain, int row, int s, int i, float x[8]) {
  FnVec8 v;
  if (MODE == 1) {
    v.u4 = *(const uint4 *)(src + (long)row * FN_H + i);
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = (float)v.h[j];
  } else if (MODE == 0 || MODE == 3) {
    v.u4 = *(const uint4 *)(src + (long)row * FN_W + s * FN_H + i);
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = (float)v.h[j];
  } else {
    FnVec8 b;
    v.u4 = *(const uint4 *)(src + (long)row * FN_W + s * FN_H + i);
    b.u4 = *(const uint4 *)(blk + (long)row * FN_H + i);
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = bf((float)v.h[j] + (float)b.h[j] * gain);
  }
}

// ---- FN37: B2 + B1 for ONE (row, stream) by ONE wave -- the single source of truth of the stream
// norm, expanded (as text: a device-function call re-schedules the shipped down's phase 1, measured)
// in FN11's v2 down (phase 1) and in the block-output producer's epilogue (`fn_ar_exact_hcrr_bf16`).
// Lane l holds the ten 8-element groups c * 256 + l * 8 of the raw stream (`xa`: res, or the stream
// itself) and, for the combine, of the block output (`xb`); x = bf(res + blk * gain) (B2) is stored
// to row r, stream s of `hid` [M, W] when `wr`; the chain sums x * x in (c, j) order, the xor
// butterfly runs 16..1 (REV: 1..16, the gate's CONTROL only), mean = acc / 2560, and it declares
// `f` = rr = 1 / sqrt(mean + eps). Needs NC, xa, xb, gain, lane, eps in scope.
#define FN_HC_RR_WAVE(MODE_, REV_, wr, hid, r, s, f)                                              \
  float acc = 0.f;                                                                                \
  _Pragma("unroll") for (int c = 0; c < NC; ++c) {                                                \
    float x[8];                                                                                   \
    _Pragma("unroll") for (int j = 0; j < 8; ++j)                                                 \
      x[j] = MODE_ == 2 ? bf((float)xa[c].h[j] + (float)xb[c].h[j] * gain) : (float)xa[c].h[j];   \
    _Pragma("unroll") for (int j = 0; j < 8; ++j) acc = acc + x[j] * x[j];                        \
    if (MODE_ != 0 && wr) {                                                                       \
      FnVec8 o;                                                                                   \
      _Pragma("unroll") for (int j = 0; j < 8; ++j) o.h[j] = (__bf16)x[j];                        \
      *(uint4 *)(hid + (long)r * FN_W + s * FN_H + c * 256 + lane * 8) = o.u4;                    \
    }                                                                                             \
  }                                                                                               \
  if ((REV_) == 2) { /* FN49b: the shipped order 16..1 on DPP / permlanex16 */                 \
    _Pragma("unroll") for (int o = 16; o; o >>= 1) acc = acc + fn_dpp_xor(acc, o);                \
  } else if (REV_) {                                                                              \
    _Pragma("unroll") for (int o = 1; o <= 16; o <<= 1) acc = acc + __shfl_xor(acc, o, 32);       \
  } else {                                                                                        \
    _Pragma("unroll") for (int o = 16; o; o >>= 1) acc = acc + __shfl_xor(acc, o, 32);            \
  }                                                                                               \
  const float mean = acc / (float)FN_H;                                                           \
  const float f = 1.f / sqrtf(mean + eps);

// ==================================================================================== DOWN
// fn_hc_down: grid (4 * sps, ceil(M / (16 * MF))), block FN_A_WAVES waves.
//   blockIdx.x = K slice q: stream s = q / sps, in-stream offset (q % sps) * (H / sps).
//   Phase 1  every row of the block: the stream's sum of squares (one wave per row; lane l owns
//            elements c*256 + l*8 + j, summed in (c, j) order, then a 32-lane xor butterfly),
//            rr = 1 / sqrt(ss / 2560 + eps) -> LDS (+ `rr` [M, 4] from slice 0 of the stream);
//            EXPAND / COMBINE: slice 0 of each stream also writes the materialized x to `hid`.
//   Phase 2  the slice's WMMA chains: A = bf16(t + t * w) with t = x * rr, formed in register from
//            x (never stored), B = Wdown rows (tiles 0..19) / Winject rows (tile 20, rows clamped
//            to 3); 16-k steps ascending into one fp32 accumulator per fragment -> `part`.
//   Phase 3  the LAST-ARRIVING workgroup of the row block (a self-resetting counter, the split-K
//            decode GEMM's protocol): d = bf16(sum of the slices in slice order), lora =
//            bf16(silu(d / 4)) for n < 320, the injection logits bf16(d) for n >= 320.
// NINJ = 4 (a layer's mix) or 0 (the final mixer: Wdown only, no logits).
// `hid` NEVER aliases `src` (COMBINE): the slice-0 workgroup of a stream writes x' while the other
// slices of the same rows still read `res` -- the engine ping-pongs two stream buffers.
template <int MODE, int MF, int NINJ>
__device__ __forceinline__ void fn_hc_down_body(
    const __bf16 *__restrict__ src, const __bf16 *__restrict__ blk, const __bf16 *__restrict__ inj,
    __bf16 *__restrict__ hid, const __bf16 *__restrict__ nw, const __bf16 *__restrict__ wdn,
    const __bf16 *__restrict__ wij, float *__restrict__ part, float *__restrict__ rr,
    int *__restrict__ cnt, __bf16 *__restrict__ lora, __bf16 *__restrict__ inj_out, int m,
    int sps, float eps) {
  constexpr int MB = 16 * MF;
  constexpr int NCOL = FN_R + NINJ;
  constexpr int NTILE = (NCOL + 15) / 16;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int q = blockIdx.x, s = q / sps, qi = q - s * sps;
  const int kw = FN_H / sps, k0 = qi * kw;          // this slice, in-stream
  const int m0 = (int)blockIdx.y * MB;
  __shared__ float s_rr[MB];
  __shared__ int s_last;

  // ---- phase 1: per-row stream norm factor
  for (int rl = wave; rl < MB; rl += FN_A_WAVES) {
    const int r = m0 + rl;
    if (r >= m) break;
    const float gain = MODE == 2 ? fn_gain(inj, r, s) : 0.f;
    float acc = 0.f;
#pragma unroll 2
    for (int c = 0; c < FN_H / 256; ++c) {
      float x[8];
      fn_x8<MODE>(src, blk, gain, r, s, c * 256 + lane * 8, x);
#pragma unroll
      for (int j = 0; j < 8; ++j) acc = acc + x[j] * x[j];
      if (MODE != 0 && qi == 0) {
        FnVec8 o;
#pragma unroll
        for (int j = 0; j < 8; ++j) o.h[j] = (__bf16)x[j];
        *(uint4 *)(hid + (long)r * FN_W + s * FN_H + c * 256 + lane * 8) = o.u4;
      }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) acc = acc + __shfl_xor(acc, o, 32);
    const float mean = acc / (float)FN_H;
    const float f = 1.f / sqrtf(mean + eps);
    if (lane == 0) {
      s_rr[rl] = f;
      if (qi == 0) rr[(long)r * FN_S + s] = f;
    }
  }
  __syncthreads();

  // ---- phase 2: the slice's chains
  const int row = lane & 15, kh = (lane >> 4) * 8;
  int arow[MF];
  float af_rr[MF], af_gain[MF];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf) {
    const int ar = min(m0 + mf * 16 + row, m - 1);
    arow[mf] = ar;
    af_rr[mf] = s_rr[ar - m0];
    af_gain[mf] = MODE == 2 ? fn_gain(inj, ar, s) : 0.f;
  }
  const __bf16 *bp[FN_A_TPW];
#pragma unroll
  for (int t = 0; t < FN_A_TPW; ++t) {
    const int j = wave + FN_A_WAVES * t;
    const int n = min(j * 16 + row, NCOL - 1);
    bp[t] = (NINJ > 0 && n >= FN_R ? wij + (long)(n - FN_R) * FN_W : wdn + (long)n * FN_W) +
            s * FN_H + kh;
  }
  floatx8 acc[MF][FN_A_TPW];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf)
#pragma unroll
    for (int t = 0; t < FN_A_TPW; ++t)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[mf][t][e] = 0.f;

  const int ksteps = kw / 16;
  constexpr int UNR = 4;                      // every sps the rules admit gives ksteps % 4 == 0
  for (int ks = 0; ks < ksteps; ks += UNR) {
    bf16x8 af[UNR][MF], wf[UNR][FN_A_TPW];
#pragma unroll
    for (int u = 0; u < UNR; ++u) {
      const int i = k0 + (ks + u) * 16 + kh;  // in-stream element offset of this lane's 8
#pragma unroll
      for (int t = 0; t < FN_A_TPW; ++t) wf[u][t] = *(const bf16x8 *)(bp[t] + k0 + (ks + u) * 16);
      FnVec8 wv;
      wv.u4 = *(const uint4 *)(nw + s * FN_H + i);
#pragma unroll
      for (int mf = 0; mf < MF; ++mf) {
        float x[8];
        fn_x8<MODE>(src, blk, af_gain[mf], arow[mf], s, i, x);
        FnVec8 a;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float tt = x[j] * af_rr[mf];
          a.h[j] = (__bf16)(tt + tt * (float)wv.h[j]);
        }
        af[u][mf] = a.v;
      }
    }
#pragma unroll
    for (int u = 0; u < UNR; ++u)
#pragma unroll
      for (int t = 0; t < FN_A_TPW; ++t) {
        if (wave + FN_A_WAVES * t >= NTILE) continue;
#pragma unroll
        for (int mf = 0; mf < MF; ++mf) acc[mf][t] = vg_wmma(af[u][mf], wf[u][t], acc[mf][t]);
      }
  }
#pragma unroll
  for (int t = 0; t < FN_A_TPW; ++t) {
    const int col = (wave + FN_A_WAVES * t) * 16 + (lane & 15);
    if (col >= NCOL) continue;
#pragma unroll
    for (int mf = 0; mf < MF; ++mf)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int r = m0 + mf * 16 + (lane >> 4) * 8 + e;
        if (r < m) part[((long)q * m + r) * FN_PSTR + col] = acc[mf][t][e];
      }
  }

  // ---- phase 3: the last arrival of this row block reduces
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    s_last = atomicAdd(&cnt[blockIdx.y], 1) == (int)gridDim.x - 1;
  }
  __syncthreads();
  if (!s_last) return;
  if (tid == 0) cnt[blockIdx.y] = 0;
  __threadfence();
  const int nrow = min(MB, m - m0), ks_n = (int)gridDim.x;
  for (int idx = tid; idx < nrow * NCOL; idx += FN_A_WAVES * 32) {
    const int rl = idx / NCOL, n = idx - rl * NCOL, r = m0 + rl;
    float a = part[(long)r * FN_PSTR + n];
    for (int k = 1; k < ks_n; ++k) a = a + part[((long)k * m + r) * FN_PSTR + n];
    const float d = bf(a);
    if (n < FN_R) lora[(long)r * FN_R + n] = (__bf16)fn_silu(d * 0.25f);
    else inj_out[(long)r * FN_NI + (n - FN_R)] = (__bf16)d;
  }
}

#define FN_DOWN(NAME, MODE, MF, NINJ)                                                             \
  extern "C" __global__ __launch_bounds__(FN_A_WAVES * 32) void NAME(                             \
      const __bf16 *__restrict__ src, const __bf16 *__restrict__ blk,                             \
      const __bf16 *__restrict__ inj, __bf16 *__restrict__ hid, const __bf16 *__restrict__ nw,    \
      const __bf16 *__restrict__ wdn, const __bf16 *__restrict__ wij, float *__restrict__ part,   \
      float *__restrict__ rr, int *__restrict__ cnt, __bf16 *__restrict__ lora,                   \
      __bf16 *__restrict__ inj_out, int m, int sps, float eps) {                                  \
    fn_hc_down_body<MODE, MF, NINJ>(src, blk, inj, hid, nw, wdn, wij, part, rr, cnt, lora,        \
                                    inj_out, m, sps, eps);                                        \
  }
#define FN_DOWN_MF(SUF, MF)                                                                       \
  FN_DOWN(fn_hc_down_mix_mf##SUF, 0, MF, 4)                                                       \
  FN_DOWN(fn_hc_down_expand_mf##SUF, 1, MF, 4)                                                    \
  FN_DOWN(fn_hc_down_combine_mf##SUF, 2, MF, 4)                                                   \
  FN_DOWN(fn_hc_down_final_mf##SUF, 2, MF, 0)
FN_DOWN_MF(1, 1)
FN_DOWN_MF(2, 2)

// ====================================================================================== UP
// fn_hc_up: grid (H / (16 * NWB), ceil(M / (16 * MF))), block NWB waves.
//   Each wave owns 16 block-input columns c0..c0+15 and forms the up GEMM for the four stream
//   columns s*H + c (four accumulators per fragment), A = lora rows (K = 320: 20 k-steps, no
//   padding -- a WMMA band walks 16-k steps, so K % 128 is not a law here), then per element:
//   g_s = bf16(acc_s), xn_s = bf16(t + t*w) recomputed from the stream (bit-identical to the
//   down's A), bi = bf16((((0 + sig(g_0)*xn_0) + sig(g_1)*xn_1) + ...) * 0.25).
//   QUANT 0: bf16 only. 1: + e4m3 row-major. 2: + e4m3 fragment-tiled (the A-tiled GEMM's layout).
//   The per-token scale needs the whole row's amax: each workgroup writes its per-row partial max
//   (`amax` [ncg, M]; max is exact and order-free) and the LAST arrival of the row block (a
//   self-resetting counter) quantizes the block's rows with the MXFP4 producers' law.
//   DBG 1 (gate only, not a route): also writes g [M, W].
#define FN_B_WAVES 4
template <int MF, int QUANT, int DBG>
__device__ __forceinline__ void fn_hc_up_body(
    const __bf16 *__restrict__ hid, const float *__restrict__ rr, const __bf16 *__restrict__ nw,
    const __bf16 *__restrict__ lora, const __bf16 *__restrict__ wup, __bf16 *__restrict__ bi,
    unsigned char *__restrict__ q8, float *__restrict__ scale, float *__restrict__ amax,
    int *__restrict__ cnt, __bf16 *__restrict__ gdbg, int m) {
  constexpr int MB = 16 * MF;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int c0 = ((int)blockIdx.x * FN_B_WAVES + wave) * 16;
  const int m0 = (int)blockIdx.y * MB;
  const int row = lane & 15, kh = (lane >> 4) * 8;
  __shared__ float s_amax[FN_B_WAVES][MB];
  __shared__ int s_last;

  const __bf16 *ap[MF];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf) ap[mf] = lora + (long)min(m0 + mf * 16 + row, m - 1) * FN_R + kh;
  const __bf16 *bp[FN_S];
#pragma unroll
  for (int s = 0; s < FN_S; ++s) bp[s] = wup + (long)(s * FN_H + c0 + row) * FN_R + kh;
  floatx8 acc[MF][FN_S];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf)
#pragma unroll
    for (int s = 0; s < FN_S; ++s)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[mf][s][e] = 0.f;
  constexpr int UNR = 4;
#pragma unroll 1
  for (int ks = 0; ks < FN_R / 16; ks += UNR) {
    bf16x8 af[UNR][MF], wf[UNR][FN_S];
#pragma unroll
    for (int u = 0; u < UNR; ++u) {
#pragma unroll
      for (int mf = 0; mf < MF; ++mf) af[u][mf] = *(const bf16x8 *)(ap[mf] + (ks + u) * 16);
#pragma unroll
      for (int s = 0; s < FN_S; ++s) wf[u][s] = *(const bf16x8 *)(bp[s] + (ks + u) * 16);
    }
#pragma unroll
    for (int u = 0; u < UNR; ++u)
#pragma unroll
      for (int s = 0; s < FN_S; ++s)
#pragma unroll
        for (int mf = 0; mf < MF; ++mf) acc[mf][s] = vg_wmma(af[u][mf], wf[u][s], acc[mf][s]);
  }

  // the gate and the gated mean, element by element (lane: column c0 + (lane & 15), 8 rows)
  const int col = c0 + (lane & 15);
  float wcol[FN_S];
#pragma unroll
  for (int s = 0; s < FN_S; ++s) wcol[s] = ld(nw + s * FN_H + col);
#pragma unroll
  for (int mf = 0; mf < MF; ++mf) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int rl = mf * 16 + (lane >> 4) * 8 + e, r = m0 + rl;
      const int rc = min(r, m - 1);
      float a = 0.f;
#pragma unroll
      for (int s = 0; s < FN_S; ++s) {
        const float g = bf(acc[mf][s][e]);
        if (DBG && r < m) gdbg[(long)r * FN_W + s * FN_H + col] = (__bf16)g;
        const float tt = ld(hid + (long)rc * FN_W + s * FN_H + col) * rr[(long)rc * FN_S + s];
        const float xn = bf(tt + tt * wcol[s]);
        a = a + fn_sigmoid(g) * xn;
      }
      const float y = bf(a * 0.25f);
      if (r < m) bi[(long)r * FN_H + col] = (__bf16)y;
      if (QUANT) {
        float v = fabsf(y);
#pragma unroll
        for (int o = 8; o; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 16));
        if ((lane & 15) == 0) s_amax[wave][rl] = v;
      }
    }
  }
  if (!QUANT) return;
  __syncthreads();
  for (int rl = tid; rl < MB; rl += FN_B_WAVES * 32) {
    float v = s_amax[0][rl];
#pragma unroll
    for (int w = 1; w < FN_B_WAVES; ++w) v = fmaxf(v, s_amax[w][rl]);
    if (m0 + rl < m) amax[(long)blockIdx.x * m + m0 + rl] = v;
  }
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    s_last = atomicAdd(&cnt[blockIdx.y], 1) == (int)gridDim.x - 1;
  }
  __syncthreads();
  if (!s_last) return;
  if (tid == 0) cnt[blockIdx.y] = 0;
  __threadfence();
  __shared__ float s_rs[MB];
  const int nrow = min(MB, m - m0);
  for (int rl = tid; rl < nrow; rl += FN_B_WAVES * 32) {
    const int r = m0 + rl;
    float v = 0.f;
    for (int g = 0; g < (int)gridDim.x; ++g) v = fmaxf(v, amax[(long)g * m + r]);
    const float sc = fmaxf(v * (1.f / 448.f), 1.f / (448.f * 512.f));
    scale[r] = sc;
    s_rs[rl] = 1.f / sc;
  }
  __syncthreads();
  constexpr int GPR = FN_H / 8;                  // 8-groups per row
  for (int idx = tid; idx < nrow * GPR; idx += FN_B_WAVES * 32) {
    const int rl = idx / GPR, g = idx - rl * GPR, r = m0 + rl;
    FnVec8 v;
    v.u4 = *(const uint4 *)(bi + (long)r * FN_H + g * 8);
    const float rs = s_rs[rl];
    unsigned lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 4; j += 2) {
      const float a = fminf(fmaxf((float)v.h[j] * rs, -448.f), 448.f);
      const float b = fminf(fmaxf((float)v.h[j + 1] * rs, -448.f), 448.f);
      lo |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu) << (j * 8);
    }
#pragma unroll
    for (int j = 4; j < 8; j += 2) {
      const float a = fminf(fmaxf((float)v.h[j] * rs, -448.f), 448.f);
      const float b = fminf(fmaxf((float)v.h[j + 1] * rs, -448.f), 448.f);
      hi |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu) << ((j - 4) * 8);
    }
    if (QUANT == 2) {
      const int mt = r >> 4, lrow = r & 15, kst = FN_H >> 4;
      ((uint2 *)q8)[((long)mt * kst + (g >> 1)) * 32 + lrow + 16 * (g & 1)] = make_uint2(lo, hi);
    } else {
      ((uint2 *)(q8 + (long)r * FN_H))[g] = make_uint2(lo, hi);
    }
  }
}

#define FN_UP(NAME, MF, QUANT, DBG)                                                               \
  extern "C" __global__ __launch_bounds__(FN_B_WAVES * 32) void NAME(                             \
      const __bf16 *__restrict__ hid, const float *__restrict__ rr,                               \
      const __bf16 *__restrict__ nw, const __bf16 *__restrict__ lora,                             \
      const __bf16 *__restrict__ wup, __bf16 *__restrict__ bi, unsigned char *__restrict__ q8,    \
      float *__restrict__ scale, float *__restrict__ amax, int *__restrict__ cnt,                 \
      __bf16 *__restrict__ gdbg, int m) {                                                         \
    fn_hc_up_body<MF, QUANT, DBG>(hid, rr, nw, lora, wup, bi, q8, scale, amax, cnt, gdbg, m);     \
  }
#define FN_UP_MF(SUF, MF)                                                                         \
  FN_UP(fn_hc_up_bf16_mf##SUF, MF, 0, 0)                                                          \
  FN_UP(fn_hc_up_q8_mf##SUF, MF, 1, 0)                                                            \
  FN_UP(fn_hc_up_q8t_mf##SUF, MF, 2, 0)                                                           \
  FN_UP(fn_hc_up_q8_dbg_mf##SUF, MF, 1, 1)
FN_UP_MF(1, 1)
FN_UP_MF(2, 2)

// ============================================================================ FN11: v2 bodies
// BIT-EXACT restructurings of the down and the up above (workstream FN11). The shipped symbols
// stay as they are (the control arm and the byte oracle); these are NEW symbols beside them, and
// the rules choose. What fixes the bytes, and what each v2 body therefore keeps EXACTLY:
//   B1  the stream norm: lane l's chain over (c, j) of x*x (each product and add rounded), the
//       32-lane xor butterfly 16..1, mean = acc / 2560.f, rr = 1.f / sqrtf(mean + eps);
//   B2  the combine's x = bf(res + blk * gain), gain = 2 * sigmoid(bf16 inj * 0.25) (hexp);
//   B3  A = bf16(tt + tt * w), tt = x * rr (two roundings, no contraction);
//   B4  per K slice, per output tile: ONE fp32 accumulator from zero, 16-k WMMA steps ascending,
//       each step's operands in the WMMA lane layout (row = lane & 15, k half = lane >> 4);
//       a row's output depends on its own A row only, so rows past M may hold anything finite;
//   B5  the slice sum: a = part[0]; a = a + part[k] for k = 1.. in slice order; d = bf(a);
//       lora = bf16(silu(d * 0.25)), logits = bf16(d);
//   B6  the up: one accumulator per (row, stream column) from zero over the 20 k-steps ascending,
//       g = bf(acc); xn recomputed by B3 from the stored stream; the gated mean
//       a = 0; a = a + sigmoid(g_s) * xn_s for s = 0..3 (each product rounded); bi = bf(a * 0.25);
//   B7  the quant: amax = max |bi| over the row (max is exact and order-free for values >= +0),
//       sc = fmaxf(amax * (1/448), 1/(448*512)), q = cvt_pk_fp8(clamp(v * (1/sc), +-448)).
// What the v2 bodies change is WHERE and HOW OFTEN these run, never their order:
//   * the norm loop issues all ten 256-element loads of a row at once (one round trip, not five);
//   * A is formed ONCE per workgroup into an LDS slab (B3, rows past M zeroed) and every wave
//     reads its fragments from it -- the shipped body re-forms it in every wave, per fragment;
//   * the weight stream runs through a register ring PF k-steps deep, its first PF steps issued
//     before the norm (they do not depend on it);
//   * the 21 column tiles can be split over gridDim.z column groups (G): more workgroups at the
//     same K slices; each group has its own arrival counter [row blocks][G] and its last arrival
//     reduces its own columns (B5 unchanged) -- SCRATCH: down counters = row blocks x G;
//   * the slice sum and the up's amax / quant tails batch their loads (the shipped tails issue
//     one load and wait, per slice / per column group / per 8-group);
//   * the up can put one stream per wave (the four products meet in LDS and are summed in s
//     order, B6) -- 4x the waves at decode; grid H/16 column groups -- SCRATCH: amax = 160 x M;
//   * the wide up can stage the lora rows in LDS and take 64-row blocks (prefill).
// The argument lists are the shipped symbols', unchanged.

// ---- FN85: THE REDUCE IN THE CONSUMER'S PROLOGUE (RP != 0; the combine down only). At TP = 2 the
// combine's block output `blk` is the two ranks' partials summed by the exact one-shot reduce
// (`saber_ar_oneshot_2rank_exact_bf16`: bf16(float(mine) + float(peer's)), one rounding). An RP down
// takes THIS rank's partial as `blk` and does the exchange itself: workgroups 0..P-1 push the
// partial (all rows) into the peer's scratch slot and post their flag (the one-shot's drain + system
// release); every workgroup waits for the peer's P flags and forms blk = the SAME vadd from its own
// partial and the peer's (read from its own scratch slot) wherever MODE 2 reads blk. The call's
// sequence is a per-channel counter every workgroup reads at entry; the last workgroup to read it
// advances it (`ctl` [calls, readers], plain device memory). The weight ring's head is issued
// before the exchange (it does not depend on it). RP 2 / 3 are GATE CONTROLS: 2 = the vadd's
// operands swapped (peer + mine: fp32 addition commutes, so it must NOT differ), 3 = the two
// partials folded into the residual in another grouping, bf(bf(res + peer * g) + mine * g) (must
// differ). The pushers are the lowest workgroup ids (dispatched first, so resident while later
// workgroups spin).
#define FN_RPW (FN_H / 8)                  // 16-byte words a row
#define FN_RP_PUSH 8                       // pusher workgroups (<= the grid)
__device__ __forceinline__ uint4 fn_rp_vadd(const uint4 &a, const uint4 &b) {  // the reference's vadd<bf16, 8>
  uint4 r;
  const __hip_bfloat16 *pa = reinterpret_cast<const __hip_bfloat16 *>(&a);
  const __hip_bfloat16 *pb = reinterpret_cast<const __hip_bfloat16 *>(&b);
  __hip_bfloat16 *pr = reinterpret_cast<__hip_bfloat16 *>(&r);
#pragma unroll
  for (int j = 0; j < 8; ++j) pr[j] = (__hip_bfloat16)((float)pa[j] + (float)pb[j]);
  return r;
}
// fn_x8<2> with blk formed from the two partials (RP 1 / 2) or the RP 3 control's grouping
template <int RP>
__device__ __forceinline__ void fn_x8rp(const __bf16 *__restrict__ src,
                                        const __bf16 *__restrict__ mine, const uint4 *peer,
                                        float gain, int row, int s, int i, float x[8]) {
  FnVec8 v, a, b, k;
  v.u4 = *(const uint4 *)(src + (long)row * FN_W + s * FN_H + i);
  a.u4 = *(const uint4 *)(mine + (long)row * FN_H + i);
  b.u4 = peer[((long)row * FN_H + i) >> 3];
  if (RP == 3) {
#pragma unroll
    for (int j = 0; j < 8; ++j)
      x[j] = bf(bf((float)v.h[j] + (float)b.h[j] * gain) + (float)a.h[j] * gain);
    return;
  }
  k.u4 = RP == 2 ? fn_rp_vadd(b.u4, a.u4) : fn_rp_vadd(a.u4, b.u4);
#pragma unroll
  for (int j = 0; j < 8; ++j) x[j] = bf((float)v.h[j] + (float)k.h[j] * gain);
}

// ---- DOWN v2: grid (4 * sps, ceil(M / MB), G), NW waves, each wave T tiles:
//      tile j = z + G * (wave + NW * t) < NTILE; needs G * NW * T >= NTILE and kw % KC == 0.
template <int MODE, int MF, int NINJ, int NW, int T, int KC, int PF, int DPP = 0, int RP = 0>
__device__ __forceinline__ void fn_hc_down2_body(
    const __bf16 *__restrict__ src, const __bf16 *__restrict__ blk, const __bf16 *__restrict__ inj,
    __bf16 *__restrict__ hid, const __bf16 *__restrict__ nw, const __bf16 *__restrict__ wdn,
    const __bf16 *__restrict__ wij, float *__restrict__ part, float *__restrict__ rr,
    int *__restrict__ cnt, __bf16 *__restrict__ lora, __bf16 *__restrict__ inj_out, int m,
    int sps, float eps, uint4 *rp_peer = nullptr, const uint4 *rp_my = nullptr,
    __bf16 *rp_out = nullptr, unsigned int *rp_peer_flags = nullptr,
    unsigned int *rp_my_flags = nullptr, unsigned int *rp_ctl = nullptr, int rp_stride16 = 0,
    int rp_drain = 0, int rp_acq = 0) {
  constexpr int MB = 16 * MF;
  constexpr int NCOL = FN_R + NINJ;
  constexpr int NTILE = (NCOL + 15) / 16;
  constexpr int NT = NW * 32;
  constexpr int LDA = KC + 8;                  // bf16 per LDS row: 16 B of pad
  constexpr int SPC = KC / 16;                 // k-steps per chunk
  constexpr int GR = KC / 8;                   // 8-groups per row per chunk
  constexpr int FB = 4;                        // formation batch per thread
  constexpr int NC = FN_H / 256;               // the norm's 256-element chunks
  static_assert(SPC % PF == 0, "the ring must tile a chunk");
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int q = blockIdx.x, s = q / sps, qi = q - s * sps;
  const int kw = FN_H / sps, k0 = qi * kw;
  const int m0 = (int)blockIdx.y * MB;
  const int G = (int)gridDim.z, z = (int)blockIdx.z;
  const int nrow = min(MB, m - m0);
  const int nsteps = kw / 16, nchunk = kw / KC;
  const bool writer = qi == 0 && z == 0;       // one workgroup per stream writes hid / rr
  __shared__ float s_rr[MB], s_gain[MB];
  __shared__ __attribute__((aligned(16))) __bf16 s_a[2][MB * LDA];
  __shared__ int s_last;

  // ---- this wave's tiles and the head of the weight ring (independent of the norm)
  const int row = lane & 15, kh = (lane >> 4) * 8;
  const __bf16 *bp[T];
  bool tl[T];
#pragma unroll
  for (int t = 0; t < T; ++t) {
    const int j = z + G * (wave + NW * t);
    tl[t] = j < NTILE;
    const int n = min(min(j, NTILE - 1) * 16 + row, NCOL - 1);
    bp[t] = (NINJ > 0 && n >= FN_R ? wij + (long)(n - FN_R) * FN_W : wdn + (long)n * FN_W) +
            s * FN_H + k0 + kh;
  }
  bf16x8 wr[PF][T];
#pragma unroll
  for (int p = 0; p < PF; ++p)
#pragma unroll
    for (int t = 0; t < T; ++t) wr[p][t] = *(const bf16x8 *)(bp[t] + min(p, nsteps - 1) * 16);

  // ---- FN85 (RP): the exchange (push, release flag, wait for the peer's pushers)
  const uint4 *rp_ms = nullptr;
  if (RP) {
    __shared__ unsigned int s_rsq;
    const int nwg = (int)(gridDim.x * gridDim.y * gridDim.z);
    const int lin = (int)(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z));
    const int npush = min(nwg, FN_RP_PUSH);
    if (tid == 0) {
      const unsigned int c =
          __hip_atomic_load(&rp_ctl[0], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
      s_rsq = c + 1u;
      if (__hip_atomic_fetch_add(&rp_ctl[1], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) ==
          (unsigned int)nwg - 1u) {
        __hip_atomic_store(&rp_ctl[1], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        __hip_atomic_store(&rp_ctl[0], c + 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
    __syncthreads();
    const unsigned int sq = s_rsq;
    const int slot = (int)(sq & 1u);
    rp_ms = rp_my + (size_t)slot * rp_stride16;
    if (lin < npush) {
      uint4 *ps = rp_peer + (size_t)slot * rp_stride16;
      const uint4 *mine4 = (const uint4 *)blk;
      const int n16 = m * FN_RPW;
      for (int k = lin * NT + tid; k < n16; k += npush * NT) ps[k] = mine4[k];
      if (rp_drain == 1) __threadfence_system();
      else if (rp_drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
      __syncthreads();
      if (tid == 0) {
        if (rp_drain == 2) __threadfence_system();
        __hip_atomic_store(&rp_peer_flags[lin], sq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
      }
    }
    if (tid < npush) {
      unsigned long long zz = 0;
      while (__hip_atomic_load(&rp_my_flags[tid], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) <
             sq) {
        if (++zz > 4000000000ULL) break;
      }
    }
    __syncthreads();
    if (rp_acq) __threadfence_system();
  }

  // ---- phase 1: the stream norm factor, one wave per row, all ten loads in flight (B1, B2);
  // MODE 3 (FN37): the producer computed it (the same FN_HC_RR_WAVE), read it
  if (MODE == 3) {
    for (int rl = tid; rl < nrow; rl += NT) {
      s_rr[rl] = rr[(long)(m0 + rl) * FN_S + s];
      s_gain[rl] = 0.f;
    }
  }
  for (int rl = wave; rl < nrow; rl += NW) {
    if (MODE == 3) break;
    const int r = m0 + rl;
    // the logit and the raw row issue together, then the gain: one round trip; fn_x8's
    // arithmetic (B2) and the (c, j) order of the chain (B1) unchanged
    const __bf16 ginj = MODE == 2 ? inj[(long)r * FN_NI + s] : (__bf16)0.f;
    const __bf16 *xs = src + (MODE == 1 ? (long)r * FN_H : (long)r * FN_W + s * FN_H) + lane * 8;
    FnVec8 xa[NC], xb[NC];
    FnVec8 xp[RP ? NC : 1];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      xa[c].u4 = *(const uint4 *)(xs + c * 256);
      if (MODE == 2) xb[c].u4 = *(const uint4 *)(blk + (long)r * FN_H + c * 256 + lane * 8);
      if (RP) xp[RP ? c : 0].u4 = rp_ms[(long)r * FN_RPW + c * 32 + lane];
    }
    __builtin_amdgcn_sched_barrier(0);         // nothing waits before every load is issued
    const float gain = MODE == 2 ? 2.f * fn_sigmoid((float)ginj * 0.25f) : 0.f;
    if (RP) {
#pragma unroll
      for (int c = 0; c < NC; ++c) {
        if (RP == 3) {                         // CONTROL: res + peer first, mine after
#pragma unroll
          for (int j = 0; j < 8; ++j)
            xa[c].h[j] = (__bf16)bf((float)xa[c].h[j] + (float)xp[RP ? c : 0].h[j] * gain);
        } else {
          xb[c].u4 = RP == 2 ? fn_rp_vadd(xp[RP ? c : 0].u4, xb[c].u4)
                             : fn_rp_vadd(xb[c].u4, xp[RP ? c : 0].u4);
        }
      }
      if (RP == 1 && rp_out != nullptr && writer && s == 0) {
#pragma unroll
        for (int c = 0; c < NC; ++c)
          *(uint4 *)(rp_out + (long)r * FN_H + c * 256 + lane * 8) = xb[c].u4;
      }
    }
    FN_HC_RR_WAVE(MODE, DPP ? 2 : 0, writer, hid, r, s, f)
    if (lane == 0) {
      s_rr[rl] = f;
      s_gain[rl] = gain;
      if (writer) rr[(long)r * FN_S + s] = f;
    }
  }
  // rows past M: zeros in every slab buffer this slice uses (B4: they never reach a stored row)
  if (nrow < MB) {
    const int nb = min(nchunk, 2), nz = (MB - nrow) * GR;
    for (int idx = tid; idx < nb * nz; idx += NT) {
      const int b = idx / nz, e = idx - b * nz, rl = nrow + e / GR, g = e - (e / GR) * GR;
      *(uint4 *)(&s_a[b][rl * LDA + g * 8]) = make_uint4(0u, 0u, 0u, 0u);
    }
  }
  __syncthreads();

  // ---- the A slab of chunk c, formed ONCE per workgroup (B3), loads batched FB deep
  auto form = [&](int c, int buf) {
    const int ng = nrow * GR, kb = k0 + c * KC;
    for (int base = 0; base < ng; base += NT * FB) {
      float xv[FB][8];
      FnVec8 wv[FB];
      int rls[FB], gs[FB];
#pragma unroll
      for (int u = 0; u < FB; ++u) {
        const int idx = min(base + u * NT + tid, ng - 1);
        const int rl = idx / GR, g = idx - rl * GR, i = kb + g * 8;
        rls[u] = rl;
        gs[u] = g;
        if (RP) fn_x8rp<RP>(src, blk, rp_ms, s_gain[rl], m0 + rl, s, i, xv[u]);
        else fn_x8<MODE>(src, blk, s_gain[rl], m0 + rl, s, i, xv[u]);
        wv[u].u4 = *(const uint4 *)(nw + s * FN_H + i);
      }
#pragma unroll
      for (int u = 0; u < FB; ++u) {
        if (base + u * NT + tid >= ng) continue;
        const float f = s_rr[rls[u]];
        FnVec8 a;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float tt = xv[u][j] * f;
          a.h[j] = (__bf16)(tt + tt * (float)wv[u].h[j]);
        }
        *(uint4 *)(&s_a[buf][rls[u] * LDA + gs[u] * 8]) = a.u4;
      }
    }
  };

  floatx8 acc[MF][T];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf)
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[mf][t][e] = 0.f;

  form(0, 0);
  __syncthreads();
  // ---- phase 2: the slice's chains (B4), A from the slab, W through the ring
  for (int c = 0; c < nchunk; ++c) {
    const __bf16 *ab = s_a[c & 1];
    for (int b = 0; b < SPC; b += PF) {
#pragma unroll
      for (int p = 0; p < PF; ++p) {
        const int ls = b + p, st = c * SPC + ls;
        bf16x8 af[MF];
#pragma unroll
        for (int mf = 0; mf < MF; ++mf)
          af[mf] = *(const bf16x8 *)(ab + (mf * 16 + row) * LDA + ls * 16 + kh);
#pragma unroll
        for (int t = 0; t < T; ++t) {
          if (!tl[t]) continue;
#pragma unroll
          for (int mf = 0; mf < MF; ++mf) acc[mf][t] = vg_wmma(af[mf], wr[p][t], acc[mf][t]);
        }
        const int nx = min(st + PF, nsteps - 1);
#pragma unroll
        for (int t = 0; t < T; ++t) wr[p][t] = *(const bf16x8 *)(bp[t] + nx * 16);
      }
    }
    if (c + 1 < nchunk) {
      form(c + 1, (c + 1) & 1);
      __syncthreads();
    }
  }
#pragma unroll
  for (int t = 0; t < T; ++t) {
    if (!tl[t]) continue;
    const int col = (z + G * (wave + NW * t)) * 16 + (lane & 15);
    if (col >= NCOL) continue;
#pragma unroll
    for (int mf = 0; mf < MF; ++mf)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int r = m0 + mf * 16 + (lane >> 4) * 8 + e;
        if (r < m) part[((long)q * m + r) * FN_PSTR + col] = acc[mf][t][e];
      }
  }

  // ---- phase 3: the last arrival of (row block, column group) reduces that group's columns (B5)
  __syncthreads();
  const int ci = (int)blockIdx.y * G + z;
  if (tid == 0) {
    __threadfence();
    s_last = atomicAdd(&cnt[ci], 1) == (int)gridDim.x - 1;
  }
  __syncthreads();
  if (!s_last) return;
  if (tid == 0) cnt[ci] = 0;
  __threadfence();
  constexpr int RE = 4;                        // elements per thread per pass
  const int ks_n = (int)gridDim.x;
  const int ntz = (NTILE - z + G - 1) / G;     // tiles of group z
  const int nel = nrow * ntz * 16;
  for (int base = 0; base < nel; base += NT * RE) {
    int ro[RE], no[RE], nr[RE];
    float a[RE];
#pragma unroll
    for (int e = 0; e < RE; ++e) {
      const int idx = min(base + e * NT + tid, nel - 1);
      const int rl = idx / (ntz * 16), rem = idx - rl * (ntz * 16);
      nr[e] = (z + G * (rem >> 4)) * 16 + (rem & 15);
      no[e] = min(nr[e], NCOL - 1);
      ro[e] = m0 + rl;
      a[e] = part[(long)ro[e] * FN_PSTR + no[e]];
    }
    for (int kb = 1; kb < ks_n; kb += 8) {
      float v[RE][8];
#pragma unroll
      for (int e = 0; e < RE; ++e)
#pragma unroll
        for (int u = 0; u < 8; ++u)
          v[e][u] = part[((long)min(kb + u, ks_n - 1) * m + ro[e]) * FN_PSTR + no[e]];
#pragma unroll
      for (int e = 0; e < RE; ++e)
#pragma unroll
        for (int u = 0; u < 8; ++u)
          if (kb + u < ks_n) a[e] = a[e] + v[e][u];
    }
#pragma unroll
    for (int e = 0; e < RE; ++e) {
      if (base + e * NT + tid >= nel || nr[e] >= NCOL) continue;
      const float d = bf(a[e]);
      if (nr[e] < FN_R) lora[(long)ro[e] * FN_R + nr[e]] = (__bf16)fn_silu(d * 0.25f);
      else inj_out[(long)ro[e] * FN_NI + (nr[e] - FN_R)] = (__bf16)d;
    }
  }
}

#define FN_DOWN2(NAME, MODE, MF, NINJ, NW, T, KC, PF)                                             \
  extern "C" __global__ __launch_bounds__(NW * 32) void NAME(                                    \
      const __bf16 *__restrict__ src, const __bf16 *__restrict__ blk,                             \
      const __bf16 *__restrict__ inj, __bf16 *__restrict__ hid, const __bf16 *__restrict__ nw,    \
      const __bf16 *__restrict__ wdn, const __bf16 *__restrict__ wij, float *__restrict__ part,   \
      float *__restrict__ rr, int *__restrict__ cnt, __bf16 *__restrict__ lora,                   \
      __bf16 *__restrict__ inj_out, int m, int sps, float eps) {                                  \
    fn_hc_down2_body<MODE, MF, NINJ, NW, T, KC, PF>(src, blk, inj, hid, nw, wdn, wij, part, rr,   \
                                                    cnt, lora, inj_out, m, sps, eps);             \
  }
#define FN_DOWN2_CFG(CFG, MF, NW, T, KC, PF)                                                      \
  FN_DOWN2(fn_hc_down_mix_v2_##CFG, 0, MF, 4, NW, T, KC, PF)                                        \
  FN_DOWN2(fn_hc_down_expand_v2_##CFG, 1, MF, 4, NW, T, KC, PF)                                     \
  FN_DOWN2(fn_hc_down_combine_v2_##CFG, 2, MF, 4, NW, T, KC, PF)                                    \
  FN_DOWN2(fn_hc_down_final_v2_##CFG, 2, MF, 0, NW, T, KC, PF)                                      \
  FN_DOWN2(fn_hc_down_combinerr_v2_##CFG, 3, MF, 4, NW, T, KC, PF)                                  \
  FN_DOWN2(fn_hc_down_finalrr_v2_##CFG, 3, MF, 0, NW, T, KC, PF)
// cfg m<MF>w<NW>t<T>: G = 21 / (NW * T) column groups; KC = the slab chunk (kw % KC == 0).
// FN11 hold 1 also measured (and dropped): m1w7t3, m1w1t1, m4w7t3, m4w21t1 (LEDGER FN11).
FN_DOWN2_CFG(m1w7t1, 1, 7, 1, 640, 8)
FN_DOWN2_CFG(m1w3t1, 1, 3, 1, 320, 10)
FN_DOWN2_CFG(m2w7t3, 2, 7, 3, 320, 4)
FN_DOWN2_CFG(m2w21t1, 2, 21, 1, 320, 10)
// FN49b: the decode cfgs with the norm's 5-hop butterfly on DPP / permlanex16 (`d`): the same
// partners in the same order -- byte-identical to the cfg without `d` by construction (gated).
#define FN_DOWN2D(NAME, MODE, MF, NINJ, NW, T, KC, PF)                                            \
  extern "C" __global__ __launch_bounds__(NW * 32) void NAME(                                    \
      const __bf16 *__restrict__ src, const __bf16 *__restrict__ blk,                             \
      const __bf16 *__restrict__ inj, __bf16 *__restrict__ hid, const __bf16 *__restrict__ nw,    \
      const __bf16 *__restrict__ wdn, const __bf16 *__restrict__ wij, float *__restrict__ part,   \
      float *__restrict__ rr, int *__restrict__ cnt, __bf16 *__restrict__ lora,                   \
      __bf16 *__restrict__ inj_out, int m, int sps, float eps) {                                  \
    fn_hc_down2_body<MODE, MF, NINJ, NW, T, KC, PF, 1>(src, blk, inj, hid, nw, wdn, wij, part,    \
                                                       rr, cnt, lora, inj_out, m, sps, eps);      \
  }
#define FN_DOWN2D_CFG(CFG, MF, NW, T, KC, PF)                                                     \
  FN_DOWN2D(fn_hc_down_mix_v2_##CFG, 0, MF, 4, NW, T, KC, PF)                                       \
  FN_DOWN2D(fn_hc_down_expand_v2_##CFG, 1, MF, 4, NW, T, KC, PF)                                    \
  FN_DOWN2D(fn_hc_down_combine_v2_##CFG, 2, MF, 4, NW, T, KC, PF)                                   \
  FN_DOWN2D(fn_hc_down_final_v2_##CFG, 2, MF, 0, NW, T, KC, PF)
FN_DOWN2D_CFG(m1w7t1d, 1, 7, 1, 640, 8)
FN_DOWN2D_CFG(m1w3t1d, 1, 3, 1, 320, 10)
// FN85: the combine downs of the decode bands with the exact two-rank reduce in their prologue
// (`_rp`; `blk` = this rank's partial + the reduce's channel). `_rps` / `_rpo` = GATE CONTROLS.
#define FN_DOWN2RP(NAME, DPP, RP, MF, NW, T, KC, PF)                                              \
  extern "C" __global__ __launch_bounds__(NW * 32) void NAME(                                    \
      const __bf16 *__restrict__ src, const __bf16 *__restrict__ blk,                             \
      const __bf16 *__restrict__ inj, __bf16 *__restrict__ hid, const __bf16 *__restrict__ nw,    \
      const __bf16 *__restrict__ wdn, const __bf16 *__restrict__ wij, float *__restrict__ part,   \
      float *__restrict__ rr, int *__restrict__ cnt, __bf16 *__restrict__ lora,                   \
      __bf16 *__restrict__ inj_out, int m, int sps, float eps, uint4 *peer_base,                  \
      const uint4 *my_base, __bf16 *out, unsigned int *peer_flags, unsigned int *my_flags,        \
      unsigned int *ctl, int slot_stride16, int drain, int acq) {                                 \
    fn_hc_down2_body<2, MF, 4, NW, T, KC, PF, DPP, RP>(                                           \
        src, blk, inj, hid, nw, wdn, wij, part, rr, cnt, lora, inj_out, m, sps, eps, peer_base,   \
        my_base, out, peer_flags, my_flags, ctl, slot_stride16, drain, acq);                      \
  }
FN_DOWN2RP(fn_hc_down_combine_v2_m1w3t1d_rp, 1, 1, 1, 3, 1, 320, 10)
FN_DOWN2RP(fn_hc_down_combine_v2_m1w3t1_rp, 0, 1, 1, 3, 1, 320, 10)
FN_DOWN2RP(fn_hc_down_combine_v2_m1w7t1_rp, 0, 1, 1, 7, 1, 640, 8)
FN_DOWN2RP(fn_hc_down_combine_v2_m1w3t1d_rps, 1, 2, 1, 3, 1, 320, 10)
FN_DOWN2RP(fn_hc_down_combine_v2_m1w3t1d_rpo, 1, 3, 1, 3, 1, 320, 10)

// ---- the up's quant tail (B7), shared by the v2 ups: the last arrival of the row block.
// The row's amax over every column group's partial: all loads in flight, combined with an LDS
// atomicMax on the bit patterns (every partial is >= +0, where uint order == float order).
template <int MB, int NT, int QUANT>
__device__ __forceinline__ void fn_hc_up2_tail(const __bf16 *__restrict__ bi,
                                               unsigned char *__restrict__ q8,
                                               float *__restrict__ scale,
                                               const float *__restrict__ amax, int m, int m0,
                                               int nrow, unsigned *s_mx, float *s_rs) {
  const int tid = threadIdx.x;
  for (int i = tid; i < MB; i += NT) s_mx[i] = 0u;
  __syncthreads();
  const int ng = (int)gridDim.x, tot = nrow * ng;
  constexpr int AU = 4;
  for (int base = 0; base < tot; base += NT * AU) {
    float v[AU];
    int rl[AU];
#pragma unroll
    for (int u = 0; u < AU; ++u) {
      const int idx = min(base + u * NT + tid, tot - 1);
      rl[u] = idx / ng;
      v[u] = amax[(long)(idx - rl[u] * ng) * m + m0 + rl[u]];
    }
#pragma unroll
    for (int u = 0; u < AU; ++u)
      if (base + u * NT + tid < tot) atomicMax(&s_mx[rl[u]], __float_as_uint(v[u]));
  }
  __syncthreads();
  for (int rl = tid; rl < nrow; rl += NT) {
    const float v = __uint_as_float(s_mx[rl]);
    const float sc = fmaxf(v * (1.f / 448.f), 1.f / (448.f * 512.f));
    scale[m0 + rl] = sc;
    s_rs[rl] = 1.f / sc;
  }
  __syncthreads();
  constexpr int GPR = FN_H / 8;
  constexpr int QU = 4;
  const int nq = nrow * GPR;
  for (int base = 0; base < nq; base += NT * QU) {
    FnVec8 v[QU];
#pragma unroll
    for (int u = 0; u < QU; ++u) {
      const int idx = min(base + u * NT + tid, nq - 1);
      const int rl = idx / GPR, g = idx - rl * GPR;
      v[u].u4 = *(const uint4 *)(bi + (long)(m0 + rl) * FN_H + g * 8);
    }
#pragma unroll
    for (int u = 0; u < QU; ++u) {
      const int idx = base + u * NT + tid;
      if (idx >= nq) continue;
      const int rl = idx / GPR, g = idx - rl * GPR, r = m0 + rl;
      const float rs = s_rs[rl];
      unsigned lo = 0, hi = 0;
#pragma unroll
      for (int j = 0; j < 4; j += 2) {
        const float a = fminf(fmaxf((float)v[u].h[j] * rs, -448.f), 448.f);
        const float b = fminf(fmaxf((float)v[u].h[j + 1] * rs, -448.f), 448.f);
        lo |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu) << (j * 8);
      }
#pragma unroll
      for (int j = 4; j < 8; j += 2) {
        const float a = fminf(fmaxf((float)v[u].h[j] * rs, -448.f), 448.f);
        const float b = fminf(fmaxf((float)v[u].h[j + 1] * rs, -448.f), 448.f);
        hi |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu) << ((j - 4) * 8);
      }
      if (QUANT == 2) {
        const int mt = r >> 4, lrow = r & 15, kst = FN_H >> 4;
        ((uint2 *)q8)[((long)mt * kst + (g >> 1)) * 32 + lrow + 16 * (g & 1)] = make_uint2(lo, hi);
      } else {
        ((uint2 *)(q8 + (long)r * FN_H))[g] = make_uint2(lo, hi);
      }
    }
  }
}

// ---- UP v2 "split": grid (H / 16, ceil(M / MB)), 4 waves = ONE STREAM PER WAVE for 16 block-
// input columns: each wave's 20-step chain for its stream (B6, all loads in flight), the stream
// operands of the epilogue loaded before the chain; the four gated products meet in LDS and are
// summed in stream order.
template <int MF, int QUANT, int DBG, int DPP = 0>
__device__ __forceinline__ void fn_hc_up2s_body(
    const __bf16 *__restrict__ hid, const float *__restrict__ rr, const __bf16 *__restrict__ nw,
    const __bf16 *__restrict__ lora, const __bf16 *__restrict__ wup, __bf16 *__restrict__ bi,
    unsigned char *__restrict__ q8, float *__restrict__ scale, float *__restrict__ amax,
    int *__restrict__ cnt, __bf16 *__restrict__ gdbg, int m) {
  constexpr int MB = 16 * MF, NT = FN_S * 32, KS = FN_R / 16;
  constexpr int UH = MF == 1 ? KS : KS / 2;
  const int tid = threadIdx.x, lane = tid & 31, s = tid >> 5;
  const int c0 = (int)blockIdx.x * 16, m0 = (int)blockIdx.y * MB;
  const int row = lane & 15, kh = (lane >> 4) * 8;
  const int nrow = min(MB, m - m0);
  __shared__ float s_p[FN_S][MB][16];
  __shared__ unsigned s_mx[MB];
  __shared__ float s_rs[MB];
  __shared__ int s_last;

  const __bf16 *ap[MF];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf) ap[mf] = lora + (long)min(m0 + mf * 16 + row, m - 1) * FN_R + kh;
  const __bf16 *bp = wup + (long)(s * FN_H + c0 + row) * FN_R + kh;
  const int col = c0 + (lane & 15);
  const float wcol = ld(nw + s * FN_H + col);
  float hv[MF][8], rv[MF][8];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int rc = min(m0 + mf * 16 + (lane >> 4) * 8 + e, m - 1);
      hv[mf][e] = ld(hid + (long)rc * FN_W + s * FN_H + col);
      rv[mf][e] = rr[(long)rc * FN_S + s];
    }
  floatx8 acc[MF];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[mf][e] = 0.f;
#pragma unroll
  for (int h = 0; h < KS; h += UH) {
    bf16x8 af[UH][MF], wf[UH];
#pragma unroll
    for (int u = 0; u < UH; ++u) {
#pragma unroll
      for (int mf = 0; mf < MF; ++mf) af[u][mf] = *(const bf16x8 *)(ap[mf] + (h + u) * 16);
      wf[u] = *(const bf16x8 *)(bp + (h + u) * 16);
    }
#pragma unroll
    for (int u = 0; u < UH; ++u)
#pragma unroll
      for (int mf = 0; mf < MF; ++mf) acc[mf] = vg_wmma(af[u][mf], wf[u], acc[mf]);
  }
#pragma unroll
  for (int mf = 0; mf < MF; ++mf)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int rl = mf * 16 + (lane >> 4) * 8 + e, r = m0 + rl;
      const float g = bf(acc[mf][e]);
      if (DBG && r < m) gdbg[(long)r * FN_W + s * FN_H + col] = (__bf16)g;
      const float tt = hv[mf][e] * rv[mf][e];
      const float xn = bf(tt + tt * wcol);
      s_p[s][rl][lane & 15] = fn_sigmoid(g) * xn;
    }
  __syncthreads();
#pragma unroll
  for (int it = 0; it < MB * 16 / NT; ++it) {
    const int idx = it * NT + tid, rl = idx >> 4, cc = idx & 15, r = m0 + rl;
    float a = 0.f;
#pragma unroll
    for (int s2 = 0; s2 < FN_S; ++s2) a = a + s_p[s2][rl][cc];
    const float y = bf(a * 0.25f);
    if (r < m) bi[(long)r * FN_H + c0 + cc] = (__bf16)y;
    if (QUANT) {
      float v = fabsf(y);
      if constexpr (DPP) {
#pragma unroll
        for (int o = 8; o; o >>= 1) v = fmaxf(v, fn_dpp_xor(v, o));
      } else {
#pragma unroll
      for (int o = 8; o; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 16));
      }
      if (cc == 0 && r < m) amax[(long)blockIdx.x * m + r] = v;
    }
  }
  if (!QUANT) return;
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    s_last = atomicAdd(&cnt[blockIdx.y], 1) == (int)gridDim.x - 1;
  }
  __syncthreads();
  if (!s_last) return;
  if (tid == 0) cnt[blockIdx.y] = 0;
  __threadfence();
  fn_hc_up2_tail<MB, NT, QUANT>(bi, q8, scale, amax, m, m0, nrow, s_mx, s_rs);
}

// ---- UP v2 "wide": the shipped up's layout (grid (40, ceil(M / MB)), 4 waves x 16 columns x
// 4 streams, B6 as shipped), with the lora rows optionally staged ONCE in LDS (LDSA) for tall
// row blocks, and the batched tail.
template <int MF, int QUANT, int DBG, int LDSA, int UNR>
__device__ __forceinline__ void fn_hc_up2w_body(
    const __bf16 *__restrict__ hid, const float *__restrict__ rr, const __bf16 *__restrict__ nw,
    const __bf16 *__restrict__ lora, const __bf16 *__restrict__ wup, __bf16 *__restrict__ bi,
    unsigned char *__restrict__ q8, float *__restrict__ scale, float *__restrict__ amax,
    int *__restrict__ cnt, __bf16 *__restrict__ gdbg, int m) {
  constexpr int MB = 16 * MF, NT = FN_B_WAVES * 32;
  constexpr int LDL = FN_R + 8;                // bf16 per staged lora row
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int c0 = ((int)blockIdx.x * FN_B_WAVES + wave) * 16;
  const int m0 = (int)blockIdx.y * MB;
  const int row = lane & 15, kh = (lane >> 4) * 8;
  const int nrow = min(MB, m - m0);
  __shared__ float s_amax[FN_B_WAVES][MB];
  __shared__ __attribute__((aligned(16))) __bf16 s_l[LDSA ? MB * LDL : 8];
  __shared__ unsigned s_mx[MB];
  __shared__ float s_rs[MB];
  __shared__ int s_last;

  if (LDSA) {
    constexpr int GR = FN_R / 8, NG = MB * GR, SB = 8;
    for (int base = 0; base < NG; base += NT * SB) {
      uint4 v[SB];
#pragma unroll
      for (int u = 0; u < SB; ++u) {
        const int idx = min(base + u * NT + tid, NG - 1), rl = idx / GR, g = idx - rl * GR;
        v[u] = *(const uint4 *)(lora + (long)min(m0 + rl, m - 1) * FN_R + g * 8);
      }
#pragma unroll
      for (int u = 0; u < SB; ++u) {
        const int idx = base + u * NT + tid;
        if (idx >= NG) continue;
        const int rl = idx / GR, g = idx - rl * GR;
        *(uint4 *)(&s_l[rl * LDL + g * 8]) = v[u];
      }
    }
    __syncthreads();
  }
  const __bf16 *ap[MF];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf)
    ap[mf] = LDSA ? s_l + (mf * 16 + row) * LDL + kh
                  : lora + (long)min(m0 + mf * 16 + row, m - 1) * FN_R + kh;
  const __bf16 *bp[FN_S];
#pragma unroll
  for (int s = 0; s < FN_S; ++s) bp[s] = wup + (long)(s * FN_H + c0 + row) * FN_R + kh;
  floatx8 acc[MF][FN_S];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf)
#pragma unroll
    for (int s = 0; s < FN_S; ++s)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[mf][s][e] = 0.f;
#pragma unroll 1
  for (int ks = 0; ks < FN_R / 16; ks += UNR) {
    bf16x8 af[UNR][MF], wf[UNR][FN_S];
#pragma unroll
    for (int u = 0; u < UNR; ++u) {
#pragma unroll
      for (int mf = 0; mf < MF; ++mf) af[u][mf] = *(const bf16x8 *)(ap[mf] + (ks + u) * 16);
#pragma unroll
      for (int s = 0; s < FN_S; ++s) wf[u][s] = *(const bf16x8 *)(bp[s] + (ks + u) * 16);
    }
#pragma unroll
    for (int u = 0; u < UNR; ++u)
#pragma unroll
      for (int s = 0; s < FN_S; ++s)
#pragma unroll
        for (int mf = 0; mf < MF; ++mf) acc[mf][s] = vg_wmma(af[u][mf], wf[u][s], acc[mf][s]);
  }
  const int col = c0 + (lane & 15);
  float wcol[FN_S];
#pragma unroll
  for (int s = 0; s < FN_S; ++s) wcol[s] = ld(nw + s * FN_H + col);
#pragma unroll
  for (int mf = 0; mf < MF; ++mf) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int rl = mf * 16 + (lane >> 4) * 8 + e, r = m0 + rl;
      const int rc = min(r, m - 1);
      float a = 0.f;
#pragma unroll
      for (int s = 0; s < FN_S; ++s) {
        const float g = bf(acc[mf][s][e]);
        if (DBG && r < m) gdbg[(long)r * FN_W + s * FN_H + col] = (__bf16)g;
        const float tt = ld(hid + (long)rc * FN_W + s * FN_H + col) * rr[(long)rc * FN_S + s];
        const float xn = bf(tt + tt * wcol[s]);
        a = a + fn_sigmoid(g) * xn;
      }
      const float y = bf(a * 0.25f);
      if (r < m) bi[(long)r * FN_H + col] = (__bf16)y;
      if (QUANT) {
        float v = fabsf(y);
#pragma unroll
        for (int o = 8; o; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 16));
        if ((lane & 15) == 0) s_amax[wave][rl] = v;
      }
    }
  }
  if (!QUANT) return;
  __syncthreads();
  for (int rl = tid; rl < MB; rl += NT) {
    float v = s_amax[0][rl];
#pragma unroll
    for (int w = 1; w < FN_B_WAVES; ++w) v = fmaxf(v, s_amax[w][rl]);
    if (m0 + rl < m) amax[(long)blockIdx.x * m + m0 + rl] = v;
  }
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    s_last = atomicAdd(&cnt[blockIdx.y], 1) == (int)gridDim.x - 1;
  }
  __syncthreads();
  if (!s_last) return;
  if (tid == 0) cnt[blockIdx.y] = 0;
  __threadfence();
  fn_hc_up2_tail<MB, NT, QUANT>(bi, q8, scale, amax, m, m0, nrow, s_mx, s_rs);
}

#define FN_UP2(NAME, BODY, NTH, ...)                                                              \
  extern "C" __global__ __launch_bounds__(NTH) void NAME(                                         \
      const __bf16 *__restrict__ hid, const float *__restrict__ rr,                               \
      const __bf16 *__restrict__ nw, const __bf16 *__restrict__ lora,                             \
      const __bf16 *__restrict__ wup, __bf16 *__restrict__ bi, unsigned char *__restrict__ q8,    \
      float *__restrict__ scale, float *__restrict__ amax, int *__restrict__ cnt,                 \
      __bf16 *__restrict__ gdbg, int m) {                                                         \
    BODY<__VA_ARGS__>(hid, rr, nw, lora, wup, bi, q8, scale, amax, cnt, gdbg, m);                 \
  }
#define FN_UP2S(CFG, MF)                                                                          \
  FN_UP2(fn_hc_up_bf16_v2_##CFG, fn_hc_up2s_body, 128, MF, 0, 0)                                    \
  FN_UP2(fn_hc_up_q8_v2_##CFG, fn_hc_up2s_body, 128, MF, 1, 0)                                      \
  FN_UP2(fn_hc_up_q8t_v2_##CFG, fn_hc_up2s_body, 128, MF, 2, 0)                                     \
  FN_UP2(fn_hc_up_q8_dbg_v2_##CFG, fn_hc_up2s_body, 128, MF, 1, 1)
#define FN_UP2W(CFG, MF, LDSA, UNR)                                                               \
  FN_UP2(fn_hc_up_bf16_v2_##CFG, fn_hc_up2w_body, 128, MF, 0, 0, LDSA, UNR)                         \
  FN_UP2(fn_hc_up_q8_v2_##CFG, fn_hc_up2w_body, 128, MF, 1, 0, LDSA, UNR)                           \
  FN_UP2(fn_hc_up_q8t_v2_##CFG, fn_hc_up2w_body, 128, MF, 2, 0, LDSA, UNR)                          \
  FN_UP2(fn_hc_up_q8_dbg_v2_##CFG, fn_hc_up2w_body, 128, MF, 1, 1, LDSA, UNR)
// s<MF>: one stream per wave, grid (160, .); w<MF>: the shipped layout, lora in LDS, grid (40, .).
// FN11 hold 1 also measured (and dropped): s2, g1 (the shipped layout + the batched tail), w4.
FN_UP2S(s1, 1)
// FN49b: s1 with the q8 amax's 4-hop butterfly on DPP row_xmask (`d`; byte-identical by
// construction, gated against s1 / FN5's up).
#define FN_UP2SD(CFG, MF)                                                                         \
  FN_UP2(fn_hc_up_bf16_v2_##CFG, fn_hc_up2s_body, 128, MF, 0, 0, 1)                                 \
  FN_UP2(fn_hc_up_q8_v2_##CFG, fn_hc_up2s_body, 128, MF, 1, 0, 1)                                   \
  FN_UP2(fn_hc_up_q8t_v2_##CFG, fn_hc_up2s_body, 128, MF, 2, 0, 1)                                  \
  FN_UP2(fn_hc_up_q8_dbg_v2_##CFG, fn_hc_up2s_body, 128, MF, 1, 1, 1)
FN_UP2SD(s1d, 1)
FN_UP2W(w2, 2, 1, 4)

// ================================================================================= COMBINE
// The standalone combine (layer 1: the PLE adds into the materialized stream between the combine
// and the mix, so there the combine cannot fuse into the norm): x' = bf16(res + blk * gain_s).
// One thread per 8 elements; grid ceil(M * W / 8 / 256).
extern "C" __global__ __launch_bounds__(256) void fn_hc_combine(
    const __bf16 *__restrict__ res, const __bf16 *__restrict__ blk,
    const __bf16 *__restrict__ inj, __bf16 *__restrict__ out, int m) {
  const long g = (long)blockIdx.x * 256 + threadIdx.x;
  if (g >= (long)m * (FN_W / 8)) return;
  const int r = (int)(g / (FN_W / 8)), e = (int)(g - (long)r * (FN_W / 8)) * 8;
  const int s = e / FN_H, i = e - s * FN_H;
  float x[8];
  fn_x8<2>(res, blk, fn_gain(inj, r, s), r, s, i, x);
  FnVec8 o;
#pragma unroll
  for (int j = 0; j < 8; ++j) o.h[j] = (__bf16)x[j];
  *(uint4 *)(out + (long)r * FN_W + e) = o.u4;
}

// ============================================================================ FN37: THE PRODUCER
// The block output's PRODUCER at TP = 2 is the two-rank exact all-reduce (the o_proj / MoE partials
// of both ranks summed; the reference's `saber_ar_oneshot_2rank_exact_bf16`, every call below 128 KiB). This
// is that reduce restructured ONE BLOCK PER ROW (as the reference's own `_nq` epilogue kernel) with the HC
// combine + the stream norm of the NEXT mix in its epilogue:
//   * the push / release flag / spin / reduce are the exact kernel's, word for word: blk = the
//     same vadd (bf16(float(mine) + float(peer's)), in that order) -> `out` (unchanged bytes);
//   * waves 0..3 (stream s = wave) load their ten res groups and the logit BEFORE the spin (they do
//     not depend on it), then fn_hc_rr_wave (B2 + B1, the down's own function) over res and the
//     reduced row from LDS -> x' `[M, W]` (hid) and rr `[M, 4]`: exactly what the down's combine
//     mode writes, so the down runs its norm-free twin (MODE 3, `fn_hc_down_{combine,final}rr_*`).
// Grid M blocks x 320 threads (one 16-byte word each: H = 2,560 = 320 x 8). The flag / sequence
// slots and the scratch window are the CALLER's (a private channel: its own sequence per block).
typedef __hip_bfloat16 fn_arbf;
__device__ __forceinline__ void fn_store_sys_rel(unsigned int *p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned int fn_load_sys_acq(const unsigned int *p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ int4 fn_ar_vadd(const int4 &a, const int4 &b) {  // the reference's vadd<bf16, 8>
  int4 r;
  const fn_arbf *pa = reinterpret_cast<const fn_arbf *>(&a);
  const fn_arbf *pb = reinterpret_cast<const fn_arbf *>(&b);
  fn_arbf *pr = reinterpret_cast<fn_arbf *>(&r);
#pragma unroll
  for (int j = 0; j < 8; ++j) pr[j] = (fn_arbf)((float)pa[j] + (float)pb[j]);
  return r;
}
#define FN_ARW (FN_H / 8)                  // 320 words a row
template <int REV>
__device__ __forceinline__ void fn_ar_hcrr_body(
    const int4 *__restrict__ in4, int4 *peer_base, const int4 *my_base, int4 *__restrict__ out4,
    unsigned int *peer_flags, unsigned int *my_flags, unsigned int *seq_ctrs,
    const __bf16 *__restrict__ res, const __bf16 *__restrict__ inj, __bf16 *__restrict__ xo,
    float *__restrict__ rro, int slot_stride16, float eps, int drain, int acq) {
  constexpr int NC = FN_H / 256;
  const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  __shared__ unsigned int s_seq;
  __shared__ __attribute__((aligned(16))) int4 s_blk[FN_ARW];
  if (tid == 0) s_seq = atomicAdd(&seq_ctrs[b], 1u) + 1u;
  __syncthreads();
  const unsigned int sq = s_seq;
  const int slot = (int)(sq & 1u);
  int4 *ps = peer_base + (size_t)slot * slot_stride16;
  const int4 *ms = my_base + (size_t)slot * slot_stride16;
  const int k = b * FN_ARW + tid;
  const int4 mine = in4[k];
  ps[k] = mine;
  // the epilogue's inputs that do not depend on the peer: in flight across the handshake
  FnVec8 xa[NC], xb[NC];
  __bf16 ginj = (__bf16)0.f;
  if (wave < FN_S) {
    ginj = inj[(long)b * FN_NI + wave];
    const __bf16 *xs = res + (long)b * FN_W + wave * FN_H + lane * 8;
#pragma unroll
    for (int c = 0; c < NC; ++c) xa[c].u4 = *(const uint4 *)(xs + c * 256);
  }
  if (drain == 1) __threadfence_system();
  else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    if (drain == 2) __threadfence_system();
    fn_store_sys_rel(&peer_flags[b], sq);
    unsigned long long z = 0;
    while (fn_load_sys_acq(&my_flags[b]) < sq) { if (++z > 4000000000ULL) break; }
  }
  __syncthreads();
  if (acq) __threadfence_system();
  const int4 red = fn_ar_vadd(mine, ms[k]);
  out4[k] = red;
  s_blk[tid] = red;
  __syncthreads();
  if (wave >= FN_S) return;
#pragma unroll
  for (int c = 0; c < NC; ++c) {
    const int4 v = s_blk[c * 32 + lane];
    xb[c].u4 = make_uint4((unsigned)v.x, (unsigned)v.y, (unsigned)v.z, (unsigned)v.w);
  }
  const float gain = 2.f * fn_sigmoid((float)ginj * 0.25f);
  FN_HC_RR_WAVE(2, REV, true, xo, b, wave, f)
  if (lane == 0) rro[(long)b * FN_S + wave] = f;
}
#define FN_AR_HCRR(NAME, REV)                                                                     \
  extern "C" __global__ __launch_bounds__(FN_ARW) void NAME(                                      \
      const int4 *__restrict__ in4, int4 *peer_base, const int4 *my_base,                         \
      int4 *__restrict__ out4, unsigned int *peer_flags, unsigned int *my_flags,                  \
      unsigned int *seq_ctrs, const __bf16 *__restrict__ res, const __bf16 *__restrict__ inj,     \
      __bf16 *__restrict__ xo, float *__restrict__ rro, int slot_stride16, float eps, int drain,  \
      int acq) {                                                                                  \
    fn_ar_hcrr_body<REV>(in4, peer_base, my_base, out4, peer_flags, my_flags, seq_ctrs, res, inj, \
                         xo, rro, slot_stride16, eps, drain, acq);                                \
  }
FN_AR_HCRR(fn_ar_exact_hcrr_bf16, 0)
// GATE ONLY: the butterfly reversed (1..16) -- the byte gate's control must see it.
FN_AR_HCRR(fn_ar_exact_hcrr_rev_bf16, 1)

// ============================================================================ WMMA PROBE
// GATE ONLY. One WMMA per 16x16 tile: D = C + A.B^T, A/B [T][16][16] bf16 row-major, C/D
// [T][16][16] fp32. The host compares D with each rounding model of `hc_ref.wmma_step`.
extern "C" __global__ __launch_bounds__(32) void fn_wmma_probe(
    const __bf16 *__restrict__ a, const __bf16 *__restrict__ b, const float *__restrict__ c,
    float *__restrict__ d) {
  const int t = blockIdx.x, lane = threadIdx.x, row = lane & 15, kh = (lane >> 4) * 8;
  const bf16x8 av = *(const bf16x8 *)(a + (long)t * 256 + row * 16 + kh);
  const bf16x8 bv = *(const bf16x8 *)(b + (long)t * 256 + row * 16 + kh);
  floatx8 acc;
#pragma unroll
  for (int e = 0; e < 8; ++e) acc[e] = c[(long)t * 256 + ((lane >> 4) * 8 + e) * 16 + row];
  acc = vg_wmma(av, bv, acc);
#pragma unroll
  for (int e = 0; e < 8; ++e) d[(long)t * 256 + ((lane >> 4) * 8 + e) * 16 + row] = acc[e];
}

// ===================================================================== bf16 GEMM, SPLIT-K band
// The drafter's band (`dflash_bgemm.hip` `db_mtile`: one wave per 16 output columns walks its K in
// ascending 16-k steps into one fp32 accumulator per 16-row fragment, 8 steps of both operands in
// flight) with K cut into `splits` equal slices on blockIdx.z -- for the SKINNY-N decode shapes the
// band alone cannot fill the card with (router N 512: 32 waves; out_proj / o_proj N 2,560 at
// K 6,144: 160 waves walking 384 steps each). Each slice's chain starts from zero; the LAST
// arrival for an output tile sums the slices in slice order and rounds once to bf16 (the class
// `hc_ref.gemm_split`). A [M, K], W [N, K] bf16 row-major, C [M, N] bf16; K % (16 * splits) == 0.
#define FB_NW 4
#define FB_UNR 8
template <int MROWS>
__device__ __forceinline__ void fn_bgemm_sk_body(const __bf16 *__restrict__ a,
                                                 const __bf16 *__restrict__ w,
                                                 float *__restrict__ part, int *__restrict__ cnt,
                                                 __bf16 *__restrict__ c, int m, int n, int k) {
  constexpr int MF = MROWS / 16;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int n0 = ((int)blockIdx.x * FB_NW + wave) * 16;
  const int m0 = (int)blockIdx.y * MROWS;
  const int sp = (int)blockIdx.z, nsp = (int)gridDim.z;
  const int kw = k / nsp, kb = sp * kw;
  const int row = lane & 15, kh = (lane >> 4) * 8;
  __shared__ int s_last;
  const bool live = n0 < n;
  const __bf16 *ap[MF];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf) ap[mf] = a + (long)min(m0 + mf * 16 + row, m - 1) * k + kb + kh;
  const __bf16 *wp = w + (long)min(n0 + row, n - 1) * k + kb + kh;
  floatx8 acc[MF];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[mf][e] = 0.f;
  const int ksteps = live ? kw / 16 : 0;
  int ks = 0;
  for (; ks + FB_UNR <= ksteps; ks += FB_UNR) {
    bf16x8 af[MF][FB_UNR], wf[FB_UNR];
#pragma unroll
    for (int u = 0; u < FB_UNR; ++u) {
#pragma unroll
      for (int mf = 0; mf < MF; ++mf) af[mf][u] = *(const bf16x8 *)(ap[mf] + (ks + u) * 16);
      wf[u] = *(const bf16x8 *)(wp + (ks + u) * 16);
    }
#pragma unroll
    for (int u = 0; u < FB_UNR; ++u)
#pragma unroll
      for (int mf = 0; mf < MF; ++mf) acc[mf] = vg_wmma(af[mf][u], wf[u], acc[mf]);
  }
  for (; ks < ksteps; ++ks) {
    const bf16x8 wv = *(const bf16x8 *)(wp + ks * 16);
#pragma unroll
    for (int mf = 0; mf < MF; ++mf)
      acc[mf] = vg_wmma(*(const bf16x8 *)(ap[mf] + ks * 16), wv, acc[mf]);
  }
  const int col = n0 + (lane & 15);
  if (nsp == 1) {                                 // one slice: the band itself, no partials
    if (!live || col >= n) return;
#pragma unroll
    for (int mf = 0; mf < MF; ++mf)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int r = m0 + mf * 16 + (lane >> 4) * 8 + e;
        if (r < m) c[(long)r * n + col] = (__bf16)acc[mf][e];
      }
    return;
  }
  if (live && col < n) {
#pragma unroll
    for (int mf = 0; mf < MF; ++mf)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int r = m0 + mf * 16 + (lane >> 4) * 8 + e;
        if (r < m) part[((long)sp * m + r) * n + col] = acc[mf][e];
      }
  }
  __syncthreads();
  const int tile = (int)blockIdx.y * (int)gridDim.x + (int)blockIdx.x;
  if (tid == 0) {
    __threadfence();
    s_last = atomicAdd(&cnt[tile], 1) == nsp - 1;
  }
  __syncthreads();
  if (!s_last) return;
  if (tid == 0) cnt[tile] = 0;
  __threadfence();
  const int nb = (int)blockIdx.x * FB_NW * 16;
  const int ncols = min(FB_NW * 16, n - nb), nrows = min(MROWS, m - m0);
  for (int idx = tid; idx < nrows * ncols; idx += FB_NW * 32) {
    const int rl = idx / ncols, cc = nb + (idx - rl * ncols), r = m0 + rl;
    float v = part[(long)r * n + cc];
    for (int p = 1; p < nsp; ++p) v = v + part[((long)p * m + r) * n + cc];
    c[(long)r * n + cc] = (__bf16)v;
  }
}

#define FN_BGEMM_SK(NAME, MROWS)                                                                  \
  extern "C" __global__ __launch_bounds__(FB_NW * 32) void NAME(                                  \
      const __bf16 *__restrict__ a, const __bf16 *__restrict__ w, float *__restrict__ part,       \
      int *__restrict__ cnt, __bf16 *__restrict__ c, int m, int n, int k) {                       \
    fn_bgemm_sk_body<MROWS>(a, w, part, cnt, c, m, n, k);                                         \
  }
FN_BGEMM_SK(fn_bgemm_sk_m16, 16)
FN_BGEMM_SK(fn_bgemm_sk_m32, 32)
FN_BGEMM_SK(fn_bgemm_sk_m64, 64)

// ===================================================================== bf16 GEMM, LDS tile (prefill)
// The vision tower's D6 body (`bf16_gemm_lds.h`), EPI 4: no bias. Same tiles as the tower's.
#include "bf16_gemm_lds.h"
#define FN_GEMM_LDS(NAME, BM, BN, MF, NF)                                                         \
  extern "C" __global__ __launch_bounds__((BM / (MF * 16)) * (BN / (NF * 16)) * 32) void NAME(   \
      const __bf16 *__restrict__ a, const __bf16 *__restrict__ w, __bf16 *c, int m, int n,        \
      int k) {                                                                                    \
    vis_gemm_lds_body<4, BM, BN, MF, NF>(a, w, nullptr, nullptr, c, m, n, k);                     \
  }
FN_GEMM_LDS(fn_bgemm_lds_l128x128, 128, 128, 4, 2)
FN_GEMM_LDS(fn_bgemm_lds_l128x64, 128, 64, 4, 2)
FN_GEMM_LDS(fn_bgemm_lds_l64x64, 64, 64, 2, 2)

// ============================================================== bf16-OUTPUT NORM / GATE twins
// While out_proj and o_proj are bf16 GEMMs their A side is bf16, not e4m3: these are the two
// producers' pre-quant values, stored. Each computes EXACTLY what its e4m3 twin computes up to its
// `(float)(__bf16)` rounding -- the gate quantizes these bytes with the twin's law and compares
// with the twin's q and scale byte for byte.
union FnArnq { uint4 u4; __bf16 h[8]; };

// `gdn_norm_quant_t256`'s first loop (mxfp4_quant.hip, saber_gdn_norm_quant_impl), line for
// line. mxfp4_quant is built WITH the HIP default contraction (saber-kbuild's table says why), so
// this function restores it: `rsqrtf(ssq * (1.f / 128.f) + eps)` contracts there and must here.
extern "C" __global__ __launch_bounds__(256) void fn_gdn_norm_bf16_t256(
    const __bf16 *__restrict__ x, const __bf16 *__restrict__ z, long zs,
    const __bf16 *__restrict__ w, __bf16 *__restrict__ out, int N, float eps) {
#pragma clang fp contract(fast)
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  x += (long)row * N; z += (long)row * zs; out += (long)row * N;
  constexpr int MAXG = 5;
  const int NV = N >> 3;
  float sv[MAXG][8];
#pragma unroll
  for (int ng = 0; ng < MAXG; ++ng) {
    const int g = tid + ng * 256;
    if (g >= NV) break;
    FnArnq vx, vz, vw;
    vx.u4 = ((const uint4 *)x)[g];
#pragma unroll
    for (int j = 0; j < 8; ++j) sv[ng][j] = (float)vx.h[j];
    float ssq = sv[ng][1] * sv[ng][1];
    ssq = __builtin_fmaf(sv[ng][0], sv[ng][0], ssq);
#pragma unroll
    for (int j = 2; j < 8; ++j) ssq = __builtin_fmaf(sv[ng][j], sv[ng][j], ssq);
    for (int o = 8; o; o >>= 1) ssq += __shfl_xor(ssq, o, 16);
    const float inv = rsqrtf(ssq * (1.f / 128.f) + eps);
    vz.u4 = ((const uint4 *)z)[g];
    vw.u4 = ((const uint4 *)w)[g & 15];
    FnArnq o;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float zz = (float)vz.h[j];
      const float sg = zz / (1.f + expf(-zz));
      o.h[j] = (__bf16)(((sv[ng][j] * inv) * (float)vw.h[j]) * sg);
    }
    ((uint4 *)out)[g] = o.u4;
  }
}

// FN9: Flash-Next's GDN gated norm gates with SIGMOID, not the 27B's silu (upstream
// `RMSNormGated.forward_static(..., activation="sigmoid")`, FN3's reference line 567; the numpy
// forward's `o * sigmoid(z)`). `fn_gdn_norm_bf16_t256` is the 27B twin (silu) and stays as FN5
// gated it; this is the same body with the gate `1 / (1 + exp(-z))`, reading z IN PLACE at the
// merged in_proj row stride (the saber gated norm's sigmoid variant needs z in compact 128-wide rows).
extern "C" __global__ __launch_bounds__(256) void fn_gdn_norm_bf16_sig_t256(
    const __bf16 *__restrict__ x, const __bf16 *__restrict__ z, long zs,
    const __bf16 *__restrict__ w, __bf16 *__restrict__ out, int N, float eps) {
#pragma clang fp contract(fast)
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  x += (long)row * N; z += (long)row * zs; out += (long)row * N;
  constexpr int MAXG = 5;
  const int NV = N >> 3;
  float sv[MAXG][8];
#pragma unroll
  for (int ng = 0; ng < MAXG; ++ng) {
    const int g = tid + ng * 256;
    if (g >= NV) break;
    FnArnq vx, vz, vw;
    vx.u4 = ((const uint4 *)x)[g];
#pragma unroll
    for (int j = 0; j < 8; ++j) sv[ng][j] = (float)vx.h[j];
    float ssq = sv[ng][1] * sv[ng][1];
    ssq = __builtin_fmaf(sv[ng][0], sv[ng][0], ssq);
#pragma unroll
    for (int j = 2; j < 8; ++j) ssq = __builtin_fmaf(sv[ng][j], sv[ng][j], ssq);
    for (int o = 8; o; o >>= 1) ssq += __shfl_xor(ssq, o, 16);
    const float inv = rsqrtf(ssq * (1.f / 128.f) + eps);
    vz.u4 = ((const uint4 *)z)[g];
    vw.u4 = ((const uint4 *)w)[g & 15];
    FnArnq o;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float zz = (float)vz.h[j];
      const float sg = 1.f / (1.f + expf(-zz));
      o.h[j] = (__bf16)(((sv[ng][j] * inv) * (float)vw.h[j]) * sg);
    }
    ((uint4 *)out)[g] = o.u4;
  }
}

// FN86: the GDN gated norm + the W8 out_proj's activation quant as ONE launch. The norm is
// `fn_gdn_norm_bf16_sig_t256` line for line (same thread -> 8-element map, same sum order, the
// same contraction pragma); its bf16 output is still stored. Each 16-lane half-wave holds one
// 128-element quant group (one head: N = heads x 128) of that bf16 output in registers, so the
// quant is `dflash_act_quant_g128`'s law on the same bf16 values: amax (max is exact, so the
// butterfly order is free), sc = max(amax / 448, 1 / (448 x 512)) -- a REAL division, q =
// clamp(x / sc, +-448) -> e4m3 by the same convert. `_rc` = the GATE CONTROL (the scale as a
// reciprocal multiply, dflash's own control 1): must differ.
template <bool RC>
__device__ __forceinline__ void fn_gdn_norm_q8_sig_body(
    const __bf16 *__restrict__ x, const __bf16 *__restrict__ z, long zs,
    const __bf16 *__restrict__ w, __bf16 *__restrict__ out, unsigned char *__restrict__ q,
    float *__restrict__ scales, int N, float eps) {
#pragma clang fp contract(fast)
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  x += (long)row * N; z += (long)row * zs; out += (long)row * N;
  q += (long)row * N; scales += (long)row * (N >> 7);
  constexpr int MAXG = 5;
  const int NV = N >> 3;
  float sv[MAXG][8];
#pragma unroll
  for (int ng = 0; ng < MAXG; ++ng) {
    const int g = tid + ng * 256;
    if (g >= NV) break;
    FnArnq vx, vz, vw;
    vx.u4 = ((const uint4 *)x)[g];
#pragma unroll
    for (int j = 0; j < 8; ++j) sv[ng][j] = (float)vx.h[j];
    float ssq = sv[ng][1] * sv[ng][1];
    ssq = __builtin_fmaf(sv[ng][0], sv[ng][0], ssq);
#pragma unroll
    for (int j = 2; j < 8; ++j) ssq = __builtin_fmaf(sv[ng][j], sv[ng][j], ssq);
    for (int o = 8; o; o >>= 1) ssq += __shfl_xor(ssq, o, 16);
    const float inv = rsqrtf(ssq * (1.f / 128.f) + eps);
    vz.u4 = ((const uint4 *)z)[g];
    vw.u4 = ((const uint4 *)w)[g & 15];
    FnArnq o;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float zz = (float)vz.h[j];
      const float sg = 1.f / (1.f + expf(-zz));
      o.h[j] = (__bf16)(((sv[ng][j] * inv) * (float)vw.h[j]) * sg);
    }
    ((uint4 *)out)[g] = o.u4;
    // the quant of this half-wave's group, from the stored bf16 values
    float ob[8];
    float a = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      ob[j] = (float)o.h[j];
      a = fmaxf(a, fabsf(ob[j]));
    }
    for (int d = 1; d < 16; d <<= 1) a = fmaxf(a, __shfl_xor(a, d, 16));
    const float sc = fmaxf(RC ? a * (1.f / 448.f) : a / 448.f, 1.f / (448.f * 512.f));
    if ((g & 15) == 0) scales[g >> 4] = sc;
    unsigned int pk[2] = {0u, 0u};
#pragma unroll
    for (int e = 0; e < 8; e += 2) {
      const float u = fminf(fmaxf(ob[e] / sc, -448.f), 448.f);
      const float v = fminf(fmaxf(ob[e + 1] / sc, -448.f), 448.f);
      pk[e >> 2] |= (__builtin_amdgcn_cvt_pk_fp8_f32(u, v, 0u, false) & 0xffffu) << ((e & 3) * 8);
    }
    ((uint2 *)q)[g] = make_uint2(pk[0], pk[1]);
  }
}

extern "C" __global__ __launch_bounds__(256) void fn_gdn_norm_q8_sig_t256(
    const __bf16 *__restrict__ x, const __bf16 *__restrict__ z, long zs,
    const __bf16 *__restrict__ w, __bf16 *__restrict__ out, unsigned char *__restrict__ q,
    float *__restrict__ scales, int N, float eps) {
  fn_gdn_norm_q8_sig_body<false>(x, z, zs, w, out, q, scales, N, eps);
}

extern "C" __global__ __launch_bounds__(256) void fn_gdn_norm_q8_sig_t256_rc(
    const __bf16 *__restrict__ x, const __bf16 *__restrict__ z, long zs,
    const __bf16 *__restrict__ w, __bf16 *__restrict__ out, unsigned char *__restrict__ q,
    float *__restrict__ scales, int N, float eps) {
  fn_gdn_norm_q8_sig_body<true>(x, z, zs, w, out, q, scales, N, eps);
}

// `attn_gate_quant_t256`'s first loop (attn_glue.hip, attn_gate_quant_impl): attn_glue is built
// -ffp-contract=off, as this file is. sig = bf16(1 / (1 + expf(-g))), p = bf16(core * sig).
extern "C" __global__ __launch_bounds__(256) void fn_attn_gate_bf16_t256(
    const __bf16 *__restrict__ core, const __bf16 *__restrict__ gate, __bf16 *__restrict__ out,
    int n, int hd, int gs) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  core += (long)row * n;
  gate += (long)row * gs;
  out += (long)row * n;
  const int nv = n >> 3;
  constexpr int MAXG = 3;
#pragma unroll
  for (int ng = 0; ng < MAXG; ++ng) {
    const int g = tid + ng * 256;
    if (g >= nv) break;
    FnArnq vc, vg, o;
    vc.u4 = ((const uint4 *)core)[g];
    const int e = g << 3, h = e / hd, off = e - h * hd;
    vg.u4 = ((const uint4 *)(gate + (long)h * 2 * hd + hd + off))[0];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float sig = bf(1.f / (1.f + expf(-(float)vg.h[j])));
      o.h[j] = (__bf16)((float)vc.h[j] * sig);
    }
    ((uint4 *)out)[g] = o.u4;
  }
}

// FN21: the same gate, every load issued before any arithmetic (the shipped body's `break` per
// 8-group serializes a load round trip per group). Same arguments, grid (m), 256 threads; the
// arithmetic line for line (sig = bf16(1 / (1 + expf(-g))), p = bf16(core * sig)).
extern "C" __global__ __launch_bounds__(256) void fn_attn_gate_bf16_v2_t256(
    const __bf16 *__restrict__ core, const __bf16 *__restrict__ gate, __bf16 *__restrict__ out,
    int n, int hd, int gs) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  core += (long)row * n;
  gate += (long)row * gs;
  out += (long)row * n;
  const int nv = n >> 3;
  constexpr int MAXG = 3;
  FnArnq vc[MAXG], vg[MAXG];
#pragma unroll
  for (int ng = 0; ng < MAXG; ++ng) {
    const int g = min(tid + ng * 256, nv - 1);
    const int e = g << 3, h = e / hd, off = e - h * hd;
    vc[ng].u4 = ((const uint4 *)core)[g];
    vg[ng].u4 = ((const uint4 *)(gate + (long)h * 2 * hd + hd + off))[0];
  }
#pragma unroll
  for (int ng = 0; ng < MAXG; ++ng) {
    const int g = tid + ng * 256;
    if (g >= nv) break;
    FnArnq o;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float sig = bf(1.f / (1.f + expf(-(float)vg[ng].h[j])));
      o.h[j] = (__bf16)((float)vc[ng].h[j] * sig);
    }
    ((uint4 *)out)[g] = o.u4;
  }
}

// FN86b (family 1, the QSA half): the attention gate + the W8 o_proj's activation quant as ONE
// launch. The gate is `fn_attn_gate_bf16_t256` line for line (same thread -> 8-element map, same
// arithmetic, this file's -ffp-contract=off); its bf16 output is still stored. A 16-lane half-wave
// holds one 128-element quant group of that output (n % 128 == 0, so a half-wave never straddles
// the row's end), so the quant is `dflash_act_quant_g128`'s law on the same bf16 values (FN86's
// twin): amax by a max butterfly (order-free), sc = max(amax / 448, 1 / (448 x 512)) a REAL
// division, q = clamp(x / sc, +-448) -> e4m3 by the same convert. `_rc` = the GATE CONTROL (the
// scale as a reciprocal multiply): must differ.
template <bool RC>
__device__ __forceinline__ void fn_attn_gate_q8_body(
    const __bf16 *__restrict__ core, const __bf16 *__restrict__ gate, __bf16 *__restrict__ out,
    unsigned char *__restrict__ q, float *__restrict__ scales, int n, int hd, int gs) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  core += (long)row * n;
  gate += (long)row * gs;
  out += (long)row * n;
  q += (long)row * n;
  scales += (long)row * (n >> 7);
  const int nv = n >> 3;
  constexpr int MAXG = 3;
#pragma unroll
  for (int ng = 0; ng < MAXG; ++ng) {
    const int g = tid + ng * 256;
    if (g >= nv) break;
    FnArnq vc, vg, o;
    vc.u4 = ((const uint4 *)core)[g];
    const int e = g << 3, h = e / hd, off = e - h * hd;
    vg.u4 = ((const uint4 *)(gate + (long)h * 2 * hd + hd + off))[0];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float sig = bf(1.f / (1.f + expf(-(float)vg.h[j])));
      o.h[j] = (__bf16)((float)vc.h[j] * sig);
    }
    ((uint4 *)out)[g] = o.u4;
    float ob[8];
    float a = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      ob[j] = (float)o.h[j];
      a = fmaxf(a, fabsf(ob[j]));
    }
    for (int d = 1; d < 16; d <<= 1) a = fmaxf(a, __shfl_xor(a, d, 16));
    const float sc = fmaxf(RC ? a * (1.f / 448.f) : a / 448.f, 1.f / (448.f * 512.f));
    if ((g & 15) == 0) scales[g >> 4] = sc;
    unsigned int pk[2] = {0u, 0u};
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
      const float u = fminf(fmaxf(ob[j] / sc, -448.f), 448.f);
      const float v = fminf(fmaxf(ob[j + 1] / sc, -448.f), 448.f);
      pk[j >> 2] |= (__builtin_amdgcn_cvt_pk_fp8_f32(u, v, 0u, false) & 0xffffu) << ((j & 3) * 8);
    }
    ((uint2 *)q)[g] = make_uint2(pk[0], pk[1]);
  }
}

extern "C" __global__ __launch_bounds__(256) void fn_attn_gate_q8_t256(
    const __bf16 *__restrict__ core, const __bf16 *__restrict__ gate, __bf16 *__restrict__ out,
    unsigned char *__restrict__ q, float *__restrict__ scales, int n, int hd, int gs) {
  fn_attn_gate_q8_body<false>(core, gate, out, q, scales, n, hd, gs);
}

extern "C" __global__ __launch_bounds__(256) void fn_attn_gate_q8_t256_rc(
    const __bf16 *__restrict__ core, const __bf16 *__restrict__ gate, __bf16 *__restrict__ out,
    unsigned char *__restrict__ q, float *__restrict__ scales, int n, int hd, int gs) {
  fn_attn_gate_q8_body<true>(core, gate, out, q, scales, n, hd, gs);
}

// ============================================================================ FN21: prefill up, M=1 down
// BIT-EXACT restructurings beside FN11's v2 symbols (workstream FN21; constraints B1-B7 above, kept
// EXACTLY -- the new bodies change where operands live and how often an op runs, never an order).
//
// THE SIGMOID TABLE. B6's gate is fn_sigmoid(g) of g = bf16(acc): a function of 16 bits. The table
// holds fn_sigmoid of every bf16 value whose result is not a constant of its region, computed on the
// card by fn_sigmoid ITSELF (same instructions, same IEEE ops: an entry is the gated value by
// construction), and the index folds the constant regions (host-exhaustive over all 65,536 bf16
// values with hc_ref.sigmoid, and on the card by `fn_hc_sig_probe` vs fn_sigmoid, 0 differing):
//   |g| bits <= 0x33c0         sigmoid == 0.5   (both signs: 1 + hexp(-g) rounds to 2)
//   |g| bits >= 0x42a0 (80)    hexp's clamp: the value at +-80
//   g   bits >= +0x4186        sigmoid == 1.0f  (1 + hexp(-g) rounds to 1)
//   NaN (either sign)          hexp's clamp maps t = NaN to -80: the value at +80 (= 1.0f)
// Entries [0, 3809): -(0x33c0 + i); [3809, 7336): +(0x33c0 + i - 3809). 29,344 bytes of LDS, built
// once per workgroup (~29 sigmoids a thread at 256 threads), replacing ~40 VALU ops per gated
// element with one index and one LDS read.
#define FN_SIG_LO 0x33c0u
#define FN_SIG_NHI 0x42a0u
#define FN_SIG_PHI 0x4186u
#define FN_SIG_NNEG ((int)(FN_SIG_NHI - FN_SIG_LO + 1))
#define FN_SIG_N (FN_SIG_NNEG + (int)(FN_SIG_PHI - FN_SIG_LO + 1))
__device__ __forceinline__ unsigned fn_sig_bits(int i) {
  return i < FN_SIG_NNEG ? (0x8000u | ((unsigned)i + FN_SIG_LO))
                         : ((unsigned)(i - FN_SIG_NNEG) + FN_SIG_LO);
}
__device__ __forceinline__ int fn_sig_index(unsigned b) {
  const unsigned mag = b & 0x7fffu;
  const unsigned c = min(max(mag, FN_SIG_LO), FN_SIG_NHI) - FN_SIG_LO;
  const bool neg = (b & 0x8000u) != 0u && mag <= 0x7f80u;
  return neg ? (int)c : FN_SIG_NNEG + (int)min(c, FN_SIG_PHI - FN_SIG_LO);
}
template <int NT>
__device__ __forceinline__ void fn_sig_build(float *tab) {
  for (int i = (int)threadIdx.x; i < FN_SIG_N; i += NT)
    tab[i] = fn_sigmoid(__uint_as_float(fn_sig_bits(i) << 16));
}
__device__ __forceinline__ unsigned fn_bf_bits(__bf16 h) {
  return (unsigned)__builtin_bit_cast(unsigned short, h);
}

// GATE ONLY: all 65,536 bf16 values g -> direct[g] = fn_sigmoid(g), viatab[g] = the table's entry.
// grid 256, 256 threads (each workgroup builds its own table, as the up does).
extern "C" __global__ __launch_bounds__(256) void fn_hc_sig_probe(float *__restrict__ direct,
                                                                  float *__restrict__ viatab) {
  __shared__ float s_tab[FN_SIG_N];
  fn_sig_build<256>(s_tab);
  __syncthreads();
  const unsigned b = blockIdx.x * 256u + threadIdx.x;
  direct[b] = fn_sigmoid(__uint_as_float(b << 16));
  viatab[b] = s_tab[fn_sig_index(b)];
}

// ---- UP v2 "p" (prefill): grid (H / (16 NW), ceil(M / 16 MF)), NW waves. Wave w owns 16 block-input
// columns c0 = (bx * NW + w) * 16 of all four streams and the workgroup's 16 MF rows: 4 MF WMMA
// accumulators, lora (A) and Wup (B) fragments straight from global through a 2-deep register ring,
// the 20 k-steps ascending (B6). The workgroup's Wup slice is read once per 16 MF rows (w2: 32).
//   SWAP  the chain runs as D^T = Wup . lora^T (the WMMA's A operand = the Wup fragment): the same
//         products in the same k order per output element, and the accumulator lands ROW-MAJOR --
//         lane (r = lane & 15, h = lane >> 4) holds row r's 8 consecutive columns c0 + 8h .. +8.
//         (Gated byte for byte against the shipped up: no host model of the WMMA's rounding exists,
//         so the byte oracle, not an argument, is what says the transposed chain is the same class.)
//   !SWAP the shipped operand order; each stream's g tile goes through a per-wave 16 x 16 LDS
//         transpose to the same row-major lane layout.
// The epilogue is then vectorized: per (row, 8 columns) one 16-byte load of hid per stream, one of
// rr, the norm weights once per lane, bi stored 16 bytes at a time; B6's gated mean per element in
// stream order (a = 0; a = a + sigma(g_s) * xn_s), sigma from the table (TAB) or fn_sigmoid.
// QUANT: the per-row partial amax over the workgroup's 16 NW columns -> amax[bx][M]; the last
// arrival of the row block quantizes (fn_hc_up2_tail, B7).
template <int MF, int NW, int QUANT, int DBG, bool SWAP, bool TAB>
__device__ __forceinline__ void fn_hc_up2p_body(
    const __bf16 *__restrict__ hid, const float *__restrict__ rr, const __bf16 *__restrict__ nw,
    const __bf16 *__restrict__ lora, const __bf16 *__restrict__ wup, __bf16 *__restrict__ bi,
    unsigned char *__restrict__ q8, float *__restrict__ scale, float *__restrict__ amax,
    int *__restrict__ cnt, __bf16 *__restrict__ gdbg, int m) {
  constexpr int MB = 16 * MF, NT = NW * 32, KS = FN_R / 16, LDT = 24;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int c0 = ((int)blockIdx.x * NW + wave) * 16, m0 = (int)blockIdx.y * MB;
  const int row = lane & 15, kh = (lane >> 4) * 8;
  const int nrow = min(MB, m - m0);
  __shared__ float s_tab[TAB ? FN_SIG_N : 1];
  __shared__ float s_amax[NW][MB];
  __shared__ __attribute__((aligned(16))) __bf16 s_t[SWAP ? 8 : NW * 16 * LDT];
  __shared__ unsigned s_mx[MB];
  __shared__ float s_rs[MB];
  __shared__ int s_last;

  const __bf16 *ap[MF];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf) ap[mf] = lora + (long)min(m0 + mf * 16 + row, m - 1) * FN_R + kh;
  const __bf16 *bp[FN_S];
#pragma unroll
  for (int s = 0; s < FN_S; ++s) bp[s] = wup + (long)(s * FN_H + c0 + row) * FN_R + kh;
  floatx8 acc[MF][FN_S];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf)
#pragma unroll
    for (int s = 0; s < FN_S; ++s)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[mf][s][e] = 0.f;
  // a 2-deep fragment ring in a NOT unrolled loop (a fully unrolled chain lets the scheduler hoist
  // every step's loads: 256 VGPRs and spills at MF 4)
  bf16x8 a0[MF], w0[FN_S], a1[MF], w1[FN_S];
  auto fl = [&](int ks, bf16x8 *av, bf16x8 *wv) {
#pragma unroll
    for (int mf = 0; mf < MF; ++mf) av[mf] = *(const bf16x8 *)(ap[mf] + ks * 16);
#pragma unroll
    for (int s = 0; s < FN_S; ++s) wv[s] = *(const bf16x8 *)(bp[s] + ks * 16);
  };
  auto fm = [&](const bf16x8 *av, const bf16x8 *wv) {
#pragma unroll
    for (int s = 0; s < FN_S; ++s)
#pragma unroll
      for (int mf = 0; mf < MF; ++mf)
        acc[mf][s] = SWAP ? vg_wmma(wv[s], av[mf], acc[mf][s]) : vg_wmma(av[mf], wv[s], acc[mf][s]);
  };
  fl(0, a0, w0);
  if (TAB) fn_sig_build<NT>(s_tab);            // VALU while the first fragments fly
#pragma unroll 1
  for (int ks = 0; ks < KS; ks += 2) {         // KS = 20: even; step ks + 2 clamps at the end
    fl(ks + 1, a1, w1);
    fm(a0, w0);
    fl(min(ks + 2, KS - 1), a0, w0);
    fm(a1, w1);
  }
  if (TAB) __syncthreads();
  __builtin_amdgcn_sched_barrier(0);           // the epilogue's loads stay out of the chain

  // ---- the epilogue, row-major: lane (row, h) -> row m0 + 16 mf + row, columns col8 .. col8 + 7;
  // the stream operands of fragment mf + 1 are loaded while fragment mf is gated (a 2-deep ring)
  const int col8 = c0 + kh;
  FnVec8 wv[FN_S];
#pragma unroll
  for (int s = 0; s < FN_S; ++s) wv[s].u4 = *(const uint4 *)(nw + s * FN_H + col8);
  FnVec8 hv[2][FN_S];
  float4 rv[2];
  auto hload = [&](int mf, int b) {
    const int rc = min(m0 + mf * 16 + row, m - 1);
#pragma unroll
    for (int s = 0; s < FN_S; ++s)
      hv[b][s].u4 = *(const uint4 *)(hid + (long)rc * FN_W + s * FN_H + col8);
    rv[b] = *(const float4 *)(rr + (long)rc * FN_S);
  };
  hload(0, 0);
  __bf16 *st = s_t + (SWAP ? 0 : wave * 16 * LDT);
#pragma unroll
  for (int mf = 0; mf < MF; ++mf) {
    const int rl = mf * 16 + row, r = m0 + rl, hb = mf & 1;
    if (mf + 1 < MF) hload(mf + 1, hb ^ 1);
    const float rs4[FN_S] = {rv[hb].x, rv[hb].y, rv[hb].z, rv[hb].w};
    float a[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) a[e] = 0.f;
#pragma unroll
    for (int s = 0; s < FN_S; ++s) {
      FnVec8 gv;
      if (SWAP) {
#pragma unroll
        for (int e = 0; e < 8; ++e) gv.h[e] = (__bf16)acc[mf][s][e];
      } else {
        // acc lane layout: column (lane & 15), rows kh + e -> st[row][col], read back row-major
#pragma unroll
        for (int e = 0; e < 8; ++e) st[(kh + e) * LDT + row] = (__bf16)acc[mf][s][e];
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        gv.u4 = *(const uint4 *)(st + row * LDT + kh);
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
      }
      if (DBG && r < m) *(uint4 *)(gdbg + (long)r * FN_W + s * FN_H + col8) = gv.u4;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float g = (float)gv.h[e];
        const float sg = TAB ? s_tab[fn_sig_index(fn_bf_bits(gv.h[e]))] : fn_sigmoid(g);
        const float tt = (float)hv[hb][s].h[e] * rs4[s];
        const float xn = bf(tt + tt * (float)wv[s].h[e]);
        a[e] = a[e] + sg * xn;
      }
    }
    FnVec8 y;
    float v = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      y.h[e] = (__bf16)(a[e] * 0.25f);
      v = fmaxf(v, fabsf((float)y.h[e]));
    }
    if (r < m) *(uint4 *)(bi + (long)r * FN_H + col8) = y.u4;
    if (QUANT) {
      v = fmaxf(v, __shfl_xor(v, 16, 32));
      if (lane < 16) s_amax[wave][rl] = v;
    }
  }
  if (!QUANT) return;
  __syncthreads();
  for (int rl = tid; rl < MB; rl += NT) {
    float v = s_amax[0][rl];
#pragma unroll
    for (int w = 1; w < NW; ++w) v = fmaxf(v, s_amax[w][rl]);
    if (m0 + rl < m) amax[(long)blockIdx.x * m + m0 + rl] = v;
  }
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    s_last = atomicAdd(&cnt[blockIdx.y], 1) == (int)gridDim.x - 1;
  }
  __syncthreads();
  if (!s_last) return;
  if (tid == 0) cnt[blockIdx.y] = 0;
  __threadfence();
  fn_hc_up2_tail<MB, NT, QUANT>(bi, q8, scale, amax, m, m0, nrow, s_mx, s_rs);
}
#define FN_UP2P(CFG, MF, NW, SWAP, TAB)                                                           \
  FN_UP2(fn_hc_up_bf16_v2_##CFG, fn_hc_up2p_body, NW * 32, MF, NW, 0, 0, SWAP, TAB)               \
  FN_UP2(fn_hc_up_q8_v2_##CFG, fn_hc_up2p_body, NW * 32, MF, NW, 1, 0, SWAP, TAB)                 \
  FN_UP2(fn_hc_up_q8t_v2_##CFG, fn_hc_up2p_body, NW * 32, MF, NW, 2, 0, SWAP, TAB)                \
  FN_UP2(fn_hc_up_q8_dbg_v2_##CFG, fn_hc_up2p_body, NW * 32, MF, NW, 1, 1, SWAP, TAB)
// cfg p<MF><s|t|n>: s = SWAP + table, t = the LDS transpose + table, n = SWAP without the table
// (the table's control); all 8 waves (grid_x 20).
FN_UP2P(p4s, 4, 8, true, true)
FN_UP2P(p4t, 4, 8, false, true)
FN_UP2P(p2n, 2, 8, true, false)
FN_UP2P(p2s, 2, 8, true, true)

// ---- DOWN v3 (decode): FN11's down v2 (B1-B5 unchanged, the same geometry and grid) with the
// critical path at a few rows shortened:
//   * the A slab's operands (the raw stream / blk / norm-weight vectors of chunk 0) are loaded
//     BEFORE the norm, beside the weight ring's head, and chunk c+1's before chunk c's chain -- the
//     formation after the norm is register arithmetic (B2 / B3 as fn_x8 and form(): the same
//     roundings); one pass per chunk needs rows x KC/8 <= threads x FB, else the v2 form() runs;
//   * the weight ring can be the whole slab chunk deep (PF = KC / 16);
//   * the last arrival's slice sum loads SB slices per round (16 = every slice at sps 4 in ONE round
//     trip; B5's order is the add loop's, not the load's).
template <int MODE, int MF, int NINJ, int NW, int T, int KC, int PF, int SB, int FB>
__device__ __forceinline__ void fn_hc_down3_body(
    const __bf16 *__restrict__ src, const __bf16 *__restrict__ blk, const __bf16 *__restrict__ inj,
    __bf16 *__restrict__ hid, const __bf16 *__restrict__ nw, const __bf16 *__restrict__ wdn,
    const __bf16 *__restrict__ wij, float *__restrict__ part, float *__restrict__ rr,
    int *__restrict__ cnt, __bf16 *__restrict__ lora, __bf16 *__restrict__ inj_out, int m,
    int sps, float eps) {
  constexpr int MB = 16 * MF;
  constexpr int NCOL = FN_R + NINJ;
  constexpr int NTILE = (NCOL + 15) / 16;
  constexpr int NT = NW * 32;
  constexpr int LDA = KC + 8;
  constexpr int SPC = KC / 16;
  constexpr int GR = KC / 8;
  constexpr int FBV = 4;                       // the v2 form()'s batch (the fallback)
  constexpr int NC = FN_H / 256;
  static_assert(SPC % PF == 0, "the ring must tile a chunk");
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int q = blockIdx.x, s = q / sps, qi = q - s * sps;
  const int kw = FN_H / sps, k0 = qi * kw;
  const int m0 = (int)blockIdx.y * MB;
  const int G = (int)gridDim.z, z = (int)blockIdx.z;
  const int nrow = min(MB, m - m0);
  const int nsteps = kw / 16, nchunk = kw / KC;
  const bool writer = qi == 0 && z == 0;
  const int ng = nrow * GR;
  const bool pre = ng <= NT * FB;
  __shared__ float s_rr[MB], s_gain[MB];
  __shared__ __attribute__((aligned(16))) __bf16 s_a[2][MB * LDA];
  __shared__ int s_last;

  const int row = lane & 15, kh = (lane >> 4) * 8;
  const __bf16 *bp[T];
  bool tl[T];
#pragma unroll
  for (int t = 0; t < T; ++t) {
    const int j = z + G * (wave + NW * t);
    tl[t] = j < NTILE;
    const int n = min(min(j, NTILE - 1) * 16 + row, NCOL - 1);
    bp[t] = (NINJ > 0 && n >= FN_R ? wij + (long)(n - FN_R) * FN_W : wdn + (long)n * FN_W) +
            s * FN_H + k0 + kh;
  }
  bf16x8 wr[PF][T];
#pragma unroll
  for (int p = 0; p < PF; ++p)
#pragma unroll
    for (int t = 0; t < T; ++t) wr[p][t] = *(const bf16x8 *)(bp[t] + min(p, nsteps - 1) * 16);

  // the slab operands of chunk c, raw (item u = u * NT + tid, clamped; stored only if < ng)
  FnVec8 fa[FB], fb[FB], fw[FB];
  auto fload = [&](int c) {
    const int kb = k0 + c * KC;
#pragma unroll
    for (int u = 0; u < FB; ++u) {
      const int idx = min(u * NT + tid, ng - 1);
      const int rl = idx / GR, g = idx - rl * GR, i = kb + g * 8, r = m0 + rl;
      fa[u].u4 = *(const uint4 *)(src + (MODE == 1 ? (long)r * FN_H : (long)r * FN_W + s * FN_H) + i);
      if (MODE == 2) fb[u].u4 = *(const uint4 *)(blk + (long)r * FN_H + i);
      fw[u].u4 = *(const uint4 *)(nw + s * FN_H + i);
    }
  };
  auto fstore = [&](int buf) {
#pragma unroll
    for (int u = 0; u < FB; ++u) {
      const int idx = u * NT + tid;
      if (idx >= ng) continue;
      const int rl = idx / GR, g = idx - rl * GR;
      const float f = s_rr[rl], gain = s_gain[rl];
      FnVec8 a;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float x = MODE == 2 ? bf((float)fa[u].h[j] + (float)fb[u].h[j] * gain) : (float)fa[u].h[j];
        const float tt = x * f;
        a.h[j] = (__bf16)(tt + tt * (float)fw[u].h[j]);
      }
      *(uint4 *)(&s_a[buf][rl * LDA + g * 8]) = a.u4;
    }
  };
  if (pre) fload(0);

  // ---- phase 1: the stream norm factor (B1, B2), as v2
  for (int rl = wave; rl < nrow; rl += NW) {
    const int r = m0 + rl;
    const __bf16 ginj = MODE == 2 ? inj[(long)r * FN_NI + s] : (__bf16)0.f;
    const __bf16 *xs = src + (MODE == 1 ? (long)r * FN_H : (long)r * FN_W + s * FN_H) + lane * 8;
    FnVec8 xa[NC], xb[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      xa[c].u4 = *(const uint4 *)(xs + c * 256);
      if (MODE == 2) xb[c].u4 = *(const uint4 *)(blk + (long)r * FN_H + c * 256 + lane * 8);
    }
    __builtin_amdgcn_sched_barrier(0);
    const float gain = MODE == 2 ? 2.f * fn_sigmoid((float)ginj * 0.25f) : 0.f;
    float acc = 0.f;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      float x[8];
#pragma unroll
      for (int j = 0; j < 8; ++j)
        x[j] = MODE == 2 ? bf((float)xa[c].h[j] + (float)xb[c].h[j] * gain) : (float)xa[c].h[j];
#pragma unroll
      for (int j = 0; j < 8; ++j) acc = acc + x[j] * x[j];
      if (MODE != 0 && writer) {
        FnVec8 o;
#pragma unroll
        for (int j = 0; j < 8; ++j) o.h[j] = (__bf16)x[j];
        *(uint4 *)(hid + (long)r * FN_W + s * FN_H + c * 256 + lane * 8) = o.u4;
      }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) acc = acc + __shfl_xor(acc, o, 32);
    const float mean = acc / (float)FN_H;
    const float f = 1.f / sqrtf(mean + eps);
    if (lane == 0) {
      s_rr[rl] = f;
      s_gain[rl] = gain;
      if (writer) rr[(long)r * FN_S + s] = f;
    }
  }
  if (nrow < MB) {
    const int nb = min(nchunk, 2), nz = (MB - nrow) * GR;
    for (int idx = tid; idx < nb * nz; idx += NT) {
      const int b = idx / nz, e = idx - b * nz, rl = nrow + e / GR, g = e - (e / GR) * GR;
      *(uint4 *)(&s_a[b][rl * LDA + g * 8]) = make_uint4(0u, 0u, 0u, 0u);
    }
  }
  __syncthreads();

  // the v2 formation (fallback when one pass cannot hold the chunk), B3
  auto form = [&](int c, int buf) {
    const int kb = k0 + c * KC;
    for (int base = 0; base < ng; base += NT * FBV) {
      float xv[FBV][8];
      FnVec8 wv[FBV];
      int rls[FBV], gs[FBV];
#pragma unroll
      for (int u = 0; u < FBV; ++u) {
        const int idx = min(base + u * NT + tid, ng - 1);
        const int rl = idx / GR, g = idx - rl * GR, i = kb + g * 8;
        rls[u] = rl;
        gs[u] = g;
        fn_x8<MODE>(src, blk, s_gain[rl], m0 + rl, s, i, xv[u]);
        wv[u].u4 = *(const uint4 *)(nw + s * FN_H + i);
      }
#pragma unroll
      for (int u = 0; u < FBV; ++u) {
        if (base + u * NT + tid >= ng) continue;
        const float f = s_rr[rls[u]];
        FnVec8 a;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float tt = xv[u][j] * f;
          a.h[j] = (__bf16)(tt + tt * (float)wv[u].h[j]);
        }
        *(uint4 *)(&s_a[buf][rls[u] * LDA + gs[u] * 8]) = a.u4;
      }
    }
  };

  floatx8 acc[MF][T];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf)
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[mf][t][e] = 0.f;

  if (pre) fstore(0);
  else form(0, 0);
  __syncthreads();
  for (int c = 0; c < nchunk; ++c) {
    if (pre && c + 1 < nchunk) fload(c + 1);
    const __bf16 *ab = s_a[c & 1];
    for (int b = 0; b < SPC; b += PF) {
#pragma unroll
      for (int p = 0; p < PF; ++p) {
        const int ls = b + p, st = c * SPC + ls;
        bf16x8 af[MF];
#pragma unroll
        for (int mf = 0; mf < MF; ++mf)
          af[mf] = *(const bf16x8 *)(ab + (mf * 16 + row) * LDA + ls * 16 + kh);
#pragma unroll
        for (int t = 0; t < T; ++t) {
          if (!tl[t]) continue;
#pragma unroll
          for (int mf = 0; mf < MF; ++mf) acc[mf][t] = vg_wmma(af[mf], wr[p][t], acc[mf][t]);
        }
        const int nx = min(st + PF, nsteps - 1);
#pragma unroll
        for (int t = 0; t < T; ++t) wr[p][t] = *(const bf16x8 *)(bp[t] + nx * 16);
      }
    }
    if (c + 1 < nchunk) {
      if (pre) fstore((c + 1) & 1);
      else form(c + 1, (c + 1) & 1);
      __syncthreads();
    }
  }
#pragma unroll
  for (int t = 0; t < T; ++t) {
    if (!tl[t]) continue;
    const int col = (z + G * (wave + NW * t)) * 16 + (lane & 15);
    if (col >= NCOL) continue;
#pragma unroll
    for (int mf = 0; mf < MF; ++mf)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int r = m0 + mf * 16 + (lane >> 4) * 8 + e;
        if (r < m) part[((long)q * m + r) * FN_PSTR + col] = acc[mf][t][e];
      }
  }

  // ---- phase 3 (B5): the last arrival of (row block, column group), SB slices a round
  __syncthreads();
  const int ci = (int)blockIdx.y * G + z;
  if (tid == 0) {
    __threadfence();
    s_last = atomicAdd(&cnt[ci], 1) == (int)gridDim.x - 1;
  }
  __syncthreads();
  if (!s_last) return;
  if (tid == 0) cnt[ci] = 0;
  __threadfence();
  constexpr int RE = 4;
  const int ks_n = (int)gridDim.x;
  const int ntz = (NTILE - z + G - 1) / G;
  const int nel = nrow * ntz * 16;
  for (int base = 0; base < nel; base += NT * RE) {
    int ro[RE], no[RE], nr[RE];
    float a[RE];
#pragma unroll
    for (int e = 0; e < RE; ++e) {
      const int idx = min(base + e * NT + tid, nel - 1);
      const int rl = idx / (ntz * 16), rem = idx - rl * (ntz * 16);
      nr[e] = (z + G * (rem >> 4)) * 16 + (rem & 15);
      no[e] = min(nr[e], NCOL - 1);
      ro[e] = m0 + rl;
      a[e] = part[(long)ro[e] * FN_PSTR + no[e]];
    }
    for (int kb = 1; kb < ks_n; kb += SB) {
      float v[RE][SB];
#pragma unroll
      for (int e = 0; e < RE; ++e)
#pragma unroll
        for (int u = 0; u < SB; ++u)
          v[e][u] = part[((long)min(kb + u, ks_n - 1) * m + ro[e]) * FN_PSTR + no[e]];
#pragma unroll
      for (int e = 0; e < RE; ++e)
#pragma unroll
        for (int u = 0; u < SB; ++u)
          if (kb + u < ks_n) a[e] = a[e] + v[e][u];
    }
#pragma unroll
    for (int e = 0; e < RE; ++e) {
      if (base + e * NT + tid >= nel || nr[e] >= NCOL) continue;
      const float d = bf(a[e]);
      if (nr[e] < FN_R) lora[(long)ro[e] * FN_R + nr[e]] = (__bf16)fn_silu(d * 0.25f);
      else inj_out[(long)ro[e] * FN_NI + (nr[e] - FN_R)] = (__bf16)d;
    }
  }
}

#define FN_DOWN3(NAME, MODE, MF, NINJ, NW, T, KC, PF, SB, FB)                                     \
  extern "C" __global__ __launch_bounds__(NW * 32) void NAME(                                    \
      const __bf16 *__restrict__ src, const __bf16 *__restrict__ blk,                             \
      const __bf16 *__restrict__ inj, __bf16 *__restrict__ hid, const __bf16 *__restrict__ nw,    \
      const __bf16 *__restrict__ wdn, const __bf16 *__restrict__ wij, float *__restrict__ part,   \
      float *__restrict__ rr, int *__restrict__ cnt, __bf16 *__restrict__ lora,                   \
      __bf16 *__restrict__ inj_out, int m, int sps, float eps) {                                  \
    fn_hc_down3_body<MODE, MF, NINJ, NW, T, KC, PF, SB, FB>(src, blk, inj, hid, nw, wdn, wij,     \
                                                            part, rr, cnt, lora, inj_out, m, sps, \
                                                            eps);                                 \
  }
#define FN_DOWN3_CFG(CFG, MF, NW, T, KC, PF, SB, FB)                                              \
  FN_DOWN3(fn_hc_down_mix_v2_##CFG, 0, MF, 4, NW, T, KC, PF, SB, FB)                              \
  FN_DOWN3(fn_hc_down_expand_v2_##CFG, 1, MF, 4, NW, T, KC, PF, SB, FB)                           \
  FN_DOWN3(fn_hc_down_combine_v2_##CFG, 2, MF, 4, NW, T, KC, PF, SB, FB)                          \
  FN_DOWN3(fn_hc_down_final_v2_##CFG, 2, MF, 0, NW, T, KC, PF, SB, FB)
// cfg m1w3t1<p|q> (FN11's M = 1 geometry): p = prefetched slab + 16-slice tail, ring 10 deep;
// q = the same with the ring the whole chunk deep (20). m1w7t1p: FN11's M 2..64 geometry with the
// prefetch (one pass up to 4 rows at kc 640 with FB 2) + the 16-slice tail.
FN_DOWN3_CFG(m1w3t1p, 1, 3, 1, 320, 10, 16, 1)
FN_DOWN3_CFG(m1w3t1q, 1, 3, 1, 320, 20, 16, 1)
FN_DOWN3_CFG(m1w7t1p, 1, 7, 1, 640, 8, 16, 2)
