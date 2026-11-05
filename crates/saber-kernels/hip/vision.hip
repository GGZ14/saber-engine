// The vision tower (D1): Qwen3.8-27B's `model.visual` -- `Qwen3_VisionTransformer` in prod
// (qwen3_vl.py), which `Qwen3_5ForConditionalGeneration` instantiates as `self.visual`.
//
// THE OPS, and where each one's arithmetic comes from (docs/VISION.md has the whole map):
//
//   patch embed   Conv3d(3 -> 1152, kernel = stride = (2, 16, 16)) over pixel_values [L, 1536]
//                 == a GEMM against the weight viewed [1152, 1536] (kernel = stride, so every
//                 output reads exactly one input row). pixel_values arrive fp32 and are cast to
//                 bf16 first (`x.to(dtype)`): `vis_cast_f32_bf16`.
//   pos embed     bilinear interpolation of the 48x48 learned table onto the (h, w) grid, in
//                 spatial-merge order, ADDED to the patch rows: `vis_pos_add`. The gather table
//                 (4 indices + 4 fp32 weights per row) is host metadata, like rope's cos/sin.
//   27 blocks     x += proj(attn(rope(qkv(LN1(x)))));  x += fc2(gelu_tanh(fc1(LN2(x))))
//                 width 1152, 16 heads x 72, MLP 4304, LayerNorm WITH bias (eps 1e-6), 2-D rope
//                 (18 frequencies per axis, neox halves of 72), non-causal attention per image.
//   merger        LN(1152) -> view [L/4, 4608] -> fc1 4608 -> GELU(erf) -> fc2 5120.
//
// ROUNDING POINTS. `tools/capture/vision_preflight.py` pins every elementwise one against prod's
// own bytes on the host before any card time: the pos-embed products and sum, the rope products
// and sums, and both residual adds are each rounded to bf16 exactly where ATen rounds them, and
// reproduce the CPU reference 0 of N differing. The GEMMs, LayerNorm, GELU and attention are
// reductions or transcendental: those are gated by ACCUMULATION CLASS (docs/GATES.md, the tower
// row), not bit-exact, because prod's own two paths (CPU oneDNN / GPU hipBLASLt + SDPA) do not
// agree bit-for-bit either.
//
// Built with -ffp-contract=off (saber-kbuild's EXTRA_FLAGS): an FMA would fuse `x1*c - x2*s`
// across the rounding points ATen has there.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float floatx8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ float bf(float x) { return (float)(__bf16)x; }  // one RNE rounding
__device__ __forceinline__ float ld(const __bf16 *p) { return (float)*p; }

// ------------------------------------------------------------------------------------ cast
extern "C" __global__ __launch_bounds__(256) void vis_cast_f32_bf16(
    const float *__restrict__ x, __bf16 *__restrict__ y, long n) {
  const long i = (long)blockIdx.x * 256 + threadIdx.x;
  if (i < n) y[i] = (__bf16)x[i];
}

// ------------------------------------------------------------------------------------ GEMM
// C[m][n] = epi( sum_k A[m][k] * W[n][k] + bias[n] ),  A [M, K], W [N, K] bf16 row-major.
// One fp32 accumulator per output, 16-wide WMMA k-steps in ascending k, the WMMA summing one 64-k
// chunk at a time into a zeroed fragment that a VALU add folds into the accumulator (D2), bias
// added in fp32, ONE rounding to bf16 -- that is the linear's output, and the epilogue then works
// on that bf16 value exactly as the next ATen op would:
//   EPI 0  store
//   EPI 1  gelu_pytorch_tanh(y)           (the checkpoint's name for the block act; the CONTROL)
//   EPI 2  gelu(y) with erf               (the merger's nn.GELU(), and the block act prod runs on
//                                          ROCm: the reference server's `_get_gelu_pytorch_tanh` falls back to it)
//   EPI 3  y + res                        (the residual add, fused; res may alias c)
// Geometry: 4 waves per workgroup in 2x2, each wave a 32x32 tile (2x2 WMMA fragments), so the
// workgroup covers 64x64. Rows/cols past M/N are clamped for the loads and never stored (a masked
// load on gfx1201 is a full round trip -- the LEDGER's predicated-staging finding).
#define VG_T 64
#define VG_UNROLL 4

__device__ __forceinline__ floatx8 vg_wmma(bf16x8 a, bf16x8 b, floatx8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

template <int EPI>
__device__ __forceinline__ void vis_gemm_body(const __bf16 *__restrict__ a,
                                              const __bf16 *__restrict__ w,
                                              const __bf16 *__restrict__ bias,
                                              const __bf16 *res, __bf16 *c, int m, int n,
                                              int k) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int m0 = (int)blockIdx.y * VG_T + (wave >> 1) * 32;
  const int n0 = (int)blockIdx.x * VG_T + (wave & 1) * 32;
  if (m0 >= m || n0 >= n) return;
  const int row = lane & 15, kh = (lane >> 4) * 8;
  const __bf16 *ap[2], *wp[2];
#pragma unroll
  for (int f = 0; f < 2; ++f) {
    int ar = m0 + f * 16 + row;
    ar = ar < m ? ar : m - 1;
    int wr = n0 + f * 16 + row;
    wr = wr < n ? wr : n - 1;
    ap[f] = a + (long)ar * k + kh;
    wp[f] = w + (long)wr * k + kh;
  }
  floatx8 acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

  const int ksteps = k / 16;
  int ks = 0;
  for (; ks + VG_UNROLL <= ksteps; ks += VG_UNROLL) {
    bf16x8 af[VG_UNROLL][2], wf[VG_UNROLL][2];
#pragma unroll
    for (int u = 0; u < VG_UNROLL; ++u)
#pragma unroll
      for (int f = 0; f < 2; ++f) {
        af[u][f] = *(const bf16x8 *)(ap[f] + (ks + u) * 16);
        wf[u][f] = *(const bf16x8 *)(wp[f] + (ks + u) * 16);
      }
    // D2: the WMMA accumulates only the 64-k CHUNK, into a zeroed fragment, and the chunk is
    // added into the running fp32 sum by a VALU add (RNE). With the WMMA carrying the whole
    // K-long sum, the block MLP's fc2 (K = 4304) left 613 of 884,736 outputs of block 0 off the
    // correctly-rounded result where prod-CPU leaves 71 -- the one class row D1's gate could not
    // see (its fc2 input was not prod's) and D2's, with prod's act captured, failed.
    floatx8 part[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) part[i][j][e] = 0.f;
#pragma unroll
    for (int u = 0; u < VG_UNROLL; ++u)
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) part[i][j] = vg_wmma(af[u][i], wf[u][j], part[i][j]);
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[i][j][e] += part[i][j][e];
  }
  for (; ks < ksteps; ++ks) {
    bf16x8 af[2], wf[2];
#pragma unroll
    for (int f = 0; f < 2; ++f) {
      af[f] = *(const bf16x8 *)(ap[f] + ks * 16);
      wf[f] = *(const bf16x8 *)(wp[f] + ks * 16);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        floatx8 z;
#pragma unroll
        for (int e = 0; e < 8; ++e) z[e] = 0.f;
        z = vg_wmma(af[i], wf[j], z);
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[i][j][e] += z[e];
      }
  }

#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int col = n0 + j * 16 + (lane & 15);
    if (col >= n) continue;
    const float b = ld(bias + col);
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int r = m0 + i * 16 + (lane >> 4) * 8 + e;
        if (r >= m) continue;
        const float y = bf(acc[i][j][e] + b);
        float o;
        if (EPI == 1) {
          o = 0.5f * y * (1.0f + tanhf(0.7978845608028654f * (y + 0.044715f * y * y * y)));
        } else if (EPI == 2) {
          o = 0.5f * y * (1.0f + erff(y * 0.7071067811865476f));
        } else if (EPI == 3) {
          o = y + ld(res + (long)r * n + col);
        } else {
          o = y;
        }
        c[(long)r * n + col] = (__bf16)o;
      }
  }
}

#define VIS_GEMM(NAME, EPI)                                                                    \
  extern "C" __global__ __launch_bounds__(128) void NAME(                                      \
      const __bf16 *__restrict__ a, const __bf16 *__restrict__ w,                              \
      const __bf16 *__restrict__ bias, const __bf16 *res, __bf16 *c, int m, int n, int k) {    \
    vis_gemm_body<EPI>(a, w, bias, res, c, m, n, k);                                           \
  }
VIS_GEMM(vis_gemm_bf16_bias, 0)
VIS_GEMM(vis_gemm_bf16_bias_gelu_tanh, 1)
VIS_GEMM(vis_gemm_bf16_bias_gelu_erf, 2)
VIS_GEMM(vis_gemm_bf16_bias_residual, 3)

// --------------------------------------------------------------------- GEMM, LDS-staged (D6)
// The same arithmetic as `vis_gemm_body` -- every output element's sum is formed in the same
// order from the same WMMA instructions -- on a larger, LDS-staged tile:
//
//   * each 64-k CHUNK is summed by the WMMA into a zeroed fragment (four 16-k steps, ascending)
//     and folded into the fp32 accumulator with one VALU add; the k-steps past the last whole
//     chunk each go through their own zeroed WMMA and add (`vis_gemm_body`'s tail loop), so the
//     result is the shipped kernel's BIT FOR BIT, and the gate says so shape by shape before any
//     time is taken (`tools/capture/d6_gemm_bench.py`);
//   * A and W both go through LDS, in 32-k tiles, double-buffered: the next tile's global loads
//     are issued before the current tile's WMMAs and land in the other buffer after them, one
//     barrier per tile. gfx1201 has no LDS-DMA, so the staging is global -> VGPR -> ds_write;
//   * rows past M / N are CLAMPED for the loads and never stored (a masked load on gfx1201 is a
//     full round trip -- the predicated-staging finding), and so are the k columns of a half tile
//     (K = 4304 ends 16 into a 32-k tile): their products are never formed.
//
// Which in-tree pattern this follows: neither `db_mtile` (the drafter's band walks K from global
// with no LDS at all -- an M <= 64 kernel, one wave per 16 columns) nor the MXFP4 A-tiled kernel
// (A read DIRECTLY in fragment order from a pre-tiled activation layout, W through LDS). The
// tower's activations are row-major [L, C] and are read by LayerNorm, rope and attention in that
// layout, so an A-tiled operand would need a producer change in five kernels; both operands go
// through LDS here instead, in the row-major fragment order the WMMA reads (row `l & 15`, eight
// k from `(l >> 4) * 8`), padded 8 bf16 per row so a 16-lane b128 read touches distinct banks.
//
// Geometry: BM x BN per workgroup, waves of MF x NF fragments (MF*16 rows, NF*16 columns).
#include "bf16_gemm_lds.h"

#define VIS_GEMM_LDS(NAME, EPI, BM, BN, MF, NF)                                                \
  extern "C" __global__ __launch_bounds__((BM / (MF * 16)) * (BN / (NF * 16)) * 32) void NAME( \
      const __bf16 *__restrict__ a, const __bf16 *__restrict__ w,                              \
      const __bf16 *__restrict__ bias, const __bf16 *res, __bf16 *c, int m, int n, int k) {    \
    vis_gemm_lds_body<EPI, BM, BN, MF, NF>(a, w, bias, res, c, m, n, k);                       \
  }
// Three tiles, one arithmetic: 128 x 128 (8 waves of 64 x 32), 128 x 64 (4 waves of 64 x 32) and
// 64 x 64 (4 waves of 32 x 32, the shipped kernel's wave tile). The rules pick one per launch by
// how many workgroups it yields (`gemm_tiles`, `vision_gfx1201.toml`).
#define VIS_GEMM_LDS_ALL(SUF, BM, BN, MF, NF)                                                  \
  VIS_GEMM_LDS(vis_gemm_bf16_bias##SUF, 0, BM, BN, MF, NF)                                     \
  VIS_GEMM_LDS(vis_gemm_bf16_bias_gelu_tanh##SUF, 1, BM, BN, MF, NF)                           \
  VIS_GEMM_LDS(vis_gemm_bf16_bias_gelu_erf##SUF, 2, BM, BN, MF, NF)                            \
  VIS_GEMM_LDS(vis_gemm_bf16_bias_residual##SUF, 3, BM, BN, MF, NF)
VIS_GEMM_LDS_ALL(_l128x128, 128, 128, 4, 2)
VIS_GEMM_LDS_ALL(_l128x64, 128, 64, 4, 2)
VIS_GEMM_LDS_ALL(_l64x64, 64, 64, 2, 2)

// ------------------------------------------------------------------------------- LayerNorm
// nn.LayerNorm(C, eps) with weight and bias over bf16 rows: fp32 mean, then fp32 variance of
// (x - mean) (two passes, biased), y = (x - mean) * rstd * w + b, one rounding. One workgroup
// per row; the per-thread partials are reduced in a fixed tree, so the result is deterministic.
#define LN_T 256
__device__ __forceinline__ float ln_block_sum(float v, float *sh) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  __syncthreads();
  if (lane == 0) sh[wave] = v;
  __syncthreads();
  float t = 0.f;
  for (int i = 0; i < LN_T / 32; ++i) t += sh[i];
  return t;
}

extern "C" __global__ __launch_bounds__(LN_T) void vis_layernorm_bf16(
    const __bf16 *__restrict__ x, const __bf16 *__restrict__ w, const __bf16 *__restrict__ b,
    __bf16 *__restrict__ y, int m, int c, float eps) {
  __shared__ float sh[LN_T / 32];
  const int r = blockIdx.x;
  if (r >= m) return;
  const __bf16 *xr = x + (long)r * c;
  float s = 0.f;
  for (int i = threadIdx.x; i < c; i += LN_T) s += ld(xr + i);
  const float mean = ln_block_sum(s, sh) / (float)c;
  float q = 0.f;
  for (int i = threadIdx.x; i < c; i += LN_T) {
    const float d = ld(xr + i) - mean;
    q += d * d;
  }
  const float var = ln_block_sum(q, sh) / (float)c;
  const float rstd = 1.0f / sqrtf(var + eps);
  for (int i = threadIdx.x; i < c; i += LN_T)
    y[(long)r * c + i] = (__bf16)((ld(xr + i) - mean) * rstd * ld(w + i) + ld(b + i));
}

// ------------------------------------------------------------------------------ pos embed
// out[i][c] = bf16( patch[i][c] + P[i][c] ),
// P[i][c]   = bf16( ((p0 + p1) + p2) + p3 ),  p_j = bf16( table[idx[j][i]][c] * bf16(wts[j][i]) )
// -- `pos_embed_interpolate_native`'s rounding points (bf16 weights, `embeds *= weights` in
// bf16, `.sum(0)`), pinned 0 of 884,736 by the pre-flight.
extern "C" __global__ __launch_bounds__(256) void vis_pos_add(
    const __bf16 *__restrict__ patch, const __bf16 *__restrict__ table,
    const int *__restrict__ idx, const float *__restrict__ wts, __bf16 *__restrict__ out, int l,
    int c) {
  const long t = (long)blockIdx.x * 256 + threadIdx.x;
  if (t >= (long)l * c) return;
  const int i = (int)(t / c), ch = (int)(t % c);
  float acc = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float wj = bf(wts[(long)j * l + i]);
    acc += bf(ld(table + (long)idx[(long)j * l + i] * c + ch) * wj);
  }
  out[t] = (__bf16)(ld(patch + t) + bf(acc));
}

// ------------------------------------------------------------------------------- 2-D rope
// In place on the q and k thirds of qkv [L, 3 * heads * hd]. Neox halves: x1 = x[:hd/2],
// x2 = x[hd/2:], cos/sin [L, hd/2] bf16 (the first hd/4 columns are the h-axis frequencies, the
// last hd/4 the w-axis ones -- the host table builds that). ATen's bf16 rounding points:
//   o1 = bf16( bf16(x1*c) - bf16(x2*s) ),  o2 = bf16( bf16(x2*c) + bf16(x1*s) )
extern "C" __global__ __launch_bounds__(256) void vis_rope_2d(
    __bf16 *qkv, const __bf16 *__restrict__ cosv, const __bf16 *__restrict__ sinv, int l,
    int heads, int hd) {
  const int half = hd / 2;
  const long t = (long)blockIdx.x * 256 + threadIdx.x;
  const long total = (long)l * 2 * heads * half;
  if (t >= total) return;
  const int d = (int)(t % half);
  long u = t / half;
  const int h = (int)(u % heads);
  u /= heads;
  const int which = (int)(u % 2);  // 0 = q, 1 = k
  const int i = (int)(u / 2);
  __bf16 *base = qkv + (long)i * 3 * heads * hd + (long)which * heads * hd + (long)h * hd;
  const float x1 = ld(base + d), x2 = ld(base + d + half);
  const float cc = ld(cosv + (long)i * half + d), ss = ld(sinv + (long)i * half + d);
  const float o1 = bf(bf(x1 * cc) - bf(x2 * ss));
  const float o2 = bf(bf(x2 * cc) + bf(x1 * ss));
  base[d] = (__bf16)o1;
  base[d + half] = (__bf16)o2;
}

// ------------------------------------------------------------------------------ attention
// Non-causal softmax(q k^T * scale) v per image (cu_seqlens), 16 heads x 72, reading q/k/v in
// place from the qkv buffer. VA_QR query rows per workgroup; their scores for the WHOLE sequence
// sit in LDS (VA_MAXL keys), so this first slice serves an image of up to VA_MAXL patches per
// sequence and the planner refuses a larger one by name. fp32 dot products (d ascending), fp32
// max / exp / sum, P*V summed over keys in ascending order, divided by the sum, one rounding.
#define VA_T 256
#define VA_QR 4
#define VA_MAXL 3072
#define VA_HD 72
extern "C" __global__ __launch_bounds__(VA_T) void vis_attn_bf16_hd72(
    const __bf16 *__restrict__ qkv, __bf16 *__restrict__ out, const int *__restrict__ cu,
    int heads, float scale) {
  __shared__ float S[VA_QR][VA_MAXL];
  __shared__ float Q[VA_QR][VA_HD];
  __shared__ float red[VA_QR][VA_T / 32];
  const int seq = blockIdx.z, h = blockIdx.y;
  const int start = cu[seq], len = cu[seq + 1] - start;
  const int q0 = blockIdx.x * VA_QR;
  if (q0 >= len) return;
  const int nq = len - q0 < VA_QR ? len - q0 : VA_QR;
  const long rs = 3L * heads * VA_HD;  // qkv row stride
  for (int t = threadIdx.x; t < VA_QR * VA_HD; t += VA_T) {
    const int r = t / VA_HD, d = t % VA_HD;
    const int qi = start + q0 + (r < nq ? r : nq - 1);
    Q[r][d] = ld(qkv + (long)qi * rs + (long)h * VA_HD + d);
  }
  __syncthreads();
  for (int j = threadIdx.x; j < len; j += VA_T) {
    const __bf16 *kr = qkv + (long)(start + j) * rs + (long)heads * VA_HD + (long)h * VA_HD;
    float acc[VA_QR];
#pragma unroll
    for (int r = 0; r < VA_QR; ++r) acc[r] = 0.f;
    for (int d = 0; d < VA_HD; ++d) {
      const float kv = ld(kr + d);
#pragma unroll
      for (int r = 0; r < VA_QR; ++r) acc[r] += Q[r][d] * kv;
    }
#pragma unroll
    for (int r = 0; r < VA_QR; ++r) S[r][j] = acc[r] * scale;
  }
  __syncthreads();
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  float mx[VA_QR], sm[VA_QR];
#pragma unroll
  for (int r = 0; r < VA_QR; ++r) {
    float v = -INFINITY;
    for (int j = threadIdx.x; j < len; j += VA_T) v = fmaxf(v, S[r][j]);
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
    if (lane == 0) red[r][wave] = v;
  }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < VA_QR; ++r) {
    float v = red[r][0];
    for (int i = 1; i < VA_T / 32; ++i) v = fmaxf(v, red[r][i]);
    mx[r] = v;
  }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < VA_QR; ++r) {
    float v = 0.f;
    for (int j = threadIdx.x; j < len; j += VA_T) {
      const float p = expf(S[r][j] - mx[r]);
      S[r][j] = p;
      v += p;
    }
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    if (lane == 0) red[r][wave] = v;
  }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < VA_QR; ++r) {
    float v = 0.f;
    for (int i = 0; i < VA_T / 32; ++i) v += red[r][i];
    sm[r] = v;
  }
  for (int t = threadIdx.x; t < VA_QR * VA_HD; t += VA_T) {
    const int r = t / VA_HD, d = t % VA_HD;
    if (r >= nq) continue;
    const __bf16 *vc = qkv + (long)start * rs + 2L * heads * VA_HD + (long)h * VA_HD + d;
    float acc = 0.f;
    for (int j = 0; j < len; ++j) acc += S[r][j] * ld(vc + (long)j * rs);
    out[(long)(start + q0 + r) * heads * VA_HD + (long)h * VA_HD + d] = (__bf16)(acc / sm[r]);
  }
}

// ------------------------------------------------------------------- tiled attention (D2)
// Flash-style non-causal attention per image, 16 heads x 72, NO per-image cap: the sequence is
// walked in 64-key tiles with an online softmax, so LDS holds one K tile and one V tile whatever
// the image. Round D2's replacement for `vis_attn_bf16_hd72` (kept as the second reference).
//
// GEOMETRY. One workgroup = VT_NW waves x 16 query rows (64 queries) of one head of one image;
// grid (ceil(max_seq / 64), heads, images). Per 64-key tile: K staged row-major [key][d] and V
// staged TRANSPOSED [d][key] in LDS (both from 16-byte global loads of the qkv rows), then per
// wave, per 16-key subtile, S^T = K . Q^T on bf16 WMMA 16x16x16 (A = K rows, B = the wave's Q,
// held in VGPRs for the whole walk), and O^T += V^T . P (A = V^T rows, B = P). The C layout of
// the first WMMA (lane = query, 8 keys) IS the B layout of the second, so P never moves between
// lanes -- the same property the reference's prefill kernel builds on (CONTIG).
//
// HEAD 72 IS PADDED TO 80 (five 16-wide k-steps), not split 64 + 8: the WMMA's k-step is 16, so an
// 8-wide tail is either a padded WMMA anyway or a VALU dot product with its own accumulation
// order. The pad columns of Q (in VGPRs) and K / V (in LDS, zeroed once) are exact zeros, so the
// padded dot product is the 72-term one plus exact zeros, and output rows 72..79 are never
// stored. Cost: 80/72 = +11% WMMA issue, which is not what bounds this kernel.
//
// ARITHMETIC CLASS. q.k in fp32 (WMMA), scaled by scale*log2(e) and exponentiated with exp2 --
// the fold is one fp32 multiply per score; the online max is taken per 64-key tile (rescale the
// accumulator and the running sum when it moves). The running sum is fp32 over the UNROUNDED p.
// P . V: WMMA takes bf16 operands, so P is either rounded once to bf16 (PSPLIT 0, what a bf16
// flash kernel does) or carried as hi + lo, two bf16 WMMAs whose sum represents p to ~2^-16
// relative (PSPLIT 1, twice the P.V issue). Output = acc / sum, one rounding. The order of the
// fp32 sums differs from D1's kernel (tiles, WMMA's internal order), so the gate is the
// accumulation class (docs/GATES.md), not bytes.
#define VT_NW 4
#define VT_KT 64
#define VT_HD 72
#define VT_HP 80
#define VT_NKS (VT_HP / 16)
#define VT_KSTR (VT_HP + 8)  // 176 B rows: 16-byte aligned fragment reads, banks spread
#define VT_VSTR (VT_KT + 8)  // 144 B rows
#define VT_QV (VT_HD / 8)    // uint4 per 72-wide head row

typedef short shortx8 __attribute__((ext_vector_type(8)));

template <int PSPLIT>
__device__ __forceinline__ void vis_attn_tiled_body(const __bf16 *__restrict__ qkv,
                                                    __bf16 *__restrict__ out,
                                                    const int *__restrict__ cu, int heads,
                                                    float scale) {
  __shared__ __attribute__((aligned(16))) unsigned short sK[VT_KT * VT_KSTR];
  __shared__ __attribute__((aligned(16))) unsigned short sV[VT_HP * VT_VSTR];
  const int seq = blockIdx.z, h = blockIdx.y;
  const int start = cu[seq], len = cu[seq + 1] - start;
  const int q0 = (int)blockIdx.x * (VT_NW * 16);
  if (q0 >= len) return;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int c = lane & 15, hh = lane >> 4;
  const long rs = 3L * heads * VT_HD;  // qkv row stride (elements)
  const __bf16 *kbase = qkv + (long)heads * VT_HD + (long)h * VT_HD;
  const __bf16 *vbase = qkv + 2L * heads * VT_HD + (long)h * VT_HD;

  // zero the pad once: K columns 72..79 of every key row, V rows 72..79
  for (int i = tid; i < VT_KT * (VT_HP - VT_HD); i += VT_NW * 32)
    sK[(i / (VT_HP - VT_HD)) * VT_KSTR + VT_HD + i % (VT_HP - VT_HD)] = 0;
  for (int i = tid; i < (VT_HP - VT_HD) * VT_VSTR; i += VT_NW * 32) sV[VT_HD * VT_VSTR + i] = 0;

  // this lane's query row, as the B operand of S^T = K . Q^T: column c, k elements 8hh..8hh+7
  const int qrow = q0 + wave * 16 + c;
  const bool live = qrow < len;
  const long qr = start + (live ? qrow : len - 1);
  bf16x8 qf[VT_NKS];
#pragma unroll
  for (int t = 0; t < VT_NKS; ++t) {
    const int d0 = 16 * t + 8 * hh;
    if (d0 < VT_HD) {
      qf[t] = *(const bf16x8 *)(qkv + qr * rs + (long)h * VT_HD + d0);
    } else {
      const shortx8 z = {0, 0, 0, 0, 0, 0, 0, 0};
      qf[t] = __builtin_bit_cast(bf16x8, z);
    }
  }
  const float mul = scale * 1.4426950408889634f;
  float m_run = -INFINITY, l_run = 0.f;
  floatx8 acc[VT_NKS];
#pragma unroll
  for (int t = 0; t < VT_NKS; ++t)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[t][e] = 0.f;

  const int ntiles = (len + VT_KT - 1) / VT_KT;
  for (int ti = 0; ti < ntiles; ++ti) {
    const int k0 = ti * VT_KT;
    __syncthreads();  // the previous tile's fragments are consumed
    for (int i = tid; i < VT_KT * VT_QV; i += VT_NW * 32) {
      const int j = i / VT_QV, part = i - j * VT_QV;
      const int key = k0 + j < len ? k0 + j : len - 1;  // clamp: masked below, finite here
      const long roff = (long)(start + key) * rs + part * 8;
      const uint4 kv = *(const uint4 *)(kbase + roff);
      const uint4 vv = *(const uint4 *)(vbase + roff);
      *(uint4 *)&sK[j * VT_KSTR + part * 8] = kv;
      const unsigned short *vs = (const unsigned short *)&vv;
#pragma unroll
      for (int e = 0; e < 8; ++e) sV[(part * 8 + e) * VT_VSTR + j] = vs[e];
    }
    __syncthreads();

    // S^T for the four 16-key subtiles: lane (c, hh) holds query c, keys 16*mt + 8hh + e
    floatx8 s[VT_KT / 16];
#pragma unroll
    for (int mt = 0; mt < VT_KT / 16; ++mt) {
#pragma unroll
      for (int e = 0; e < 8; ++e) s[mt][e] = 0.f;
      const unsigned short *kr = &sK[(mt * 16 + c) * VT_KSTR + 8 * hh];
#pragma unroll
      for (int t = 0; t < VT_NKS; ++t) {
        const bf16x8 kf = *(const bf16x8 *)(kr + 16 * t);
        s[mt] = vg_wmma(kf, qf[t], s[mt]);
      }
    }
    float tmax = -INFINITY;
#pragma unroll
    for (int mt = 0; mt < VT_KT / 16; ++mt)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int key = k0 + mt * 16 + 8 * hh + e;
        const float x = key < len ? s[mt][e] * mul : -INFINITY;
        s[mt][e] = x;
        tmax = fmaxf(tmax, x);
      }
    tmax = fmaxf(tmax, __shfl_xor(tmax, 16, 32));
    const float m_new = fmaxf(m_run, tmax);  // finite: every tile holds a live key
    if (m_new > m_run) {
      const float alpha = exp2f(m_run - m_new);  // 0 on the first tile
#pragma unroll
      for (int t = 0; t < VT_NKS; ++t)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[t][e] *= alpha;
      l_run *= alpha;
      m_run = m_new;
    }
#pragma unroll
    for (int mt = 0; mt < VT_KT / 16; ++mt) {
      float p[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        p[e] = exp2f(s[mt][e] - m_run);
        l_run += p[e];
      }
      bf16x8 phi, plo;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const __bf16 b = (__bf16)p[e];
        phi[e] = b;
        plo[e] = PSPLIT ? (__bf16)(p[e] - (float)b) : b;
      }
      const unsigned short *vr = &sV[c * VT_VSTR + mt * 16 + 8 * hh];
#pragma unroll
      for (int t = 0; t < VT_NKS; ++t) {
        const bf16x8 vf = *(const bf16x8 *)(vr + t * 16 * VT_VSTR);
        acc[t] = vg_wmma(vf, phi, acc[t]);
        if (PSPLIT) acc[t] = vg_wmma(vf, plo, acc[t]);
      }
    }
  }
  if (!live) return;
  const float l = l_run + __shfl_xor(l_run, 16, 32);
  __bf16 *op = out + (long)(start + qrow) * heads * VT_HD + (long)h * VT_HD;
#pragma unroll
  for (int t = 0; t < VT_NKS; ++t) {
    const int d0 = 16 * t + 8 * hh;
    if (d0 >= VT_HD) continue;
    bf16x8 o;
#pragma unroll
    for (int e = 0; e < 8; ++e) o[e] = (__bf16)(acc[t][e] / l);
    *(bf16x8 *)(op + d0) = o;
  }
}

extern "C" __global__ __launch_bounds__(VT_NW * 32) void vis_attn_tiled_bf16_hd72(
    const __bf16 *__restrict__ qkv, __bf16 *__restrict__ out, const int *__restrict__ cu,
    int heads, float scale) {
  vis_attn_tiled_body<1>(qkv, out, cu, heads, scale);
}

// The PSPLIT 0 arm: P rounded once to bf16. Built and gated beside the default so the class cost
// of the cheaper P.V is a measured row, not an assumption.
extern "C" __global__ __launch_bounds__(VT_NW * 32) void vis_attn_tiled_p16_bf16_hd72(
    const __bf16 *__restrict__ qkv, __bf16 *__restrict__ out, const int *__restrict__ cu,
    int heads, float scale) {
  vis_attn_tiled_body<0>(qkv, out, cu, heads, scale);
}
