// Flash-Next (FN22): the MTP drafter's DEVICE-SIDE CHAIN -- argmax -> the next token's embedding
// row, on the card, between two drafter passes, with no host round trip. Saber's own integer law;
// no floating-point arithmetic touches a logit (keys are the bf16 bit patterns' total order).
//
// THE LAW (== the host chain's `worker::row_argmax` + `worker::merge_argmax`, bit for bit):
//   per rank   the largest bf16 over the rank's vocabulary half by the order-preserving key
//              key(b) = b & 0x8000 ? ~b : b | 0x8000   (-0 < +0; +NaN above +inf, -NaN below -inf),
//              the LOWEST local index on an exact key tie;
//   the pair   the larger key; on an exact tie the LOWER global id -- always rank 0's candidate
//              (rank 0 holds ids 0..half, rank 1 half..2*half).
//   `tie_high` (the tie law's negative control, never served) flips both ties to the highest index.
//
//   fn_draft_argmax_part  grid (FN22_SPLITS, rows), block 256: split s of row sel[r] of the logits
//                         [*, half] bf16 -> part[r * FN22_SPLITS + s] = key << 32 | ~index (u64; the
//                         max over u64 IS the law: larger key, then lower index). sel[r] < 0 (a pad
//                         row) writes 0 and reads nothing.
//   fn_draft_pack         grid (rows), block 256: the row's winner over its splits; the exchange
//                         payload row r [2 * H + 16] bf16 = [slot rank 0 | slot rank 1 | lanes]:
//                         this rank's slot = embh[local winner] (the checkpoint's row, copied), its
//                         8 lanes = (key hi, key lo, gid b0, b1, b2, 0, 0, 0) as bf16 small integers
//                         (exact: <= 255), and the OTHER rank's slot + lanes = -0.0 (0x8000). A pad
//                         row forces rank 0's candidate to id 0 at key 0xffff (rank 1: key 0).
//   (the certified 2-rank exact all-reduce, saber_ar_oneshot_2rank_exact_bf16, sums the payloads:
//    f32(x) + f32(-0.0) == x EXACTLY for every non-NaN x, +0 and -0 included, in either order, and
//    the bf16 round of an exact value is itself -- so both ranks hold both candidates' bytes)
//   fn_draft_select       grid (1), block 256: per row the pair law over the two lane sets; the
//                         winner's slot row -> emb[r] (the next pass's embedding input), the id ->
//                         ids[*ctr * FN22_IDS_STRIDE + r]; then *ctr += 1 (one pass = one row block).
//
// Built with -ffp-contract=off (saber-kbuild's EXTRA_FLAGS; no contraction exists here anyway).
#include <hip/hip_runtime.h>
#include <stdint.h>

#define FN22_SPLITS 32
#define FN22_H 2560
#define FN22_LANES 16
#define FN22_IDS_STRIDE 8

__device__ __forceinline__ uint32_t fn22_key(uint32_t b) {
  return (b & 0x8000u) ? (~b & 0xffffu) : (b | 0x8000u);
}

__device__ __forceinline__ unsigned long long fn22_max(unsigned long long a, unsigned long long b) {
  return a > b ? a : b;
}

__device__ __forceinline__ unsigned long long fn22_wave_max(unsigned long long v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const uint32_t lo = __shfl_xor((uint32_t)v, o, 32);
    const uint32_t hi = __shfl_xor((uint32_t)(v >> 32), o, 32);
    v = fn22_max(v, ((unsigned long long)hi << 32) | lo);
  }
  return v;
}

// a small integer (<= 255) as a bf16 bit pattern: exact (the f32's low 16 bits are zero)
__device__ __forceinline__ uint16_t fn22_enc(uint32_t v) {
  return (uint16_t)(__builtin_bit_cast(uint32_t, (float)v) >> 16);
}

__device__ __forceinline__ uint32_t fn22_dec(uint16_t b) {
  return (uint32_t)__builtin_bit_cast(float, (uint32_t)b << 16);
}

__device__ __forceinline__ unsigned long long fn22_pack(uint32_t key, uint32_t idx, int tie_high) {
  return ((unsigned long long)key << 32) | (tie_high ? idx : ~idx);
}

// ------------------------------------------------------------------------------------ argmax part
extern "C" __global__ __launch_bounds__(256) void fn_draft_argmax_part(
    const uint16_t *__restrict__ lg, const int *__restrict__ sel, unsigned long long *__restrict__ part,
    int half, int tie_high) {
  __shared__ unsigned long long red[8];
  const int r = blockIdx.y, s = blockIdx.x, tid = threadIdx.x;
  const int row = sel[r];
  unsigned long long best = 0;
  if (row >= 0) {  // workgroup-uniform
    const int nv = half >> 3;  // 16-byte vectors (half % 8 == 0: asserted by the host)
    const int per = (nv + FN22_SPLITS - 1) / FN22_SPLITS;
    const int v0 = s * per;
    const int v1 = min(nv, v0 + per);
    const uint4 *p = (const uint4 *)(lg + (long)row * half);
    for (int v = v0 + tid; v < v1; v += 256) {
      const uint4 x = p[v];
      const uint32_t w[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t i0 = (uint32_t)(v * 8 + j * 2);
        best = fn22_max(best, fn22_pack(fn22_key(w[j] & 0xffffu), i0, tie_high));
        best = fn22_max(best, fn22_pack(fn22_key(w[j] >> 16), i0 + 1, tie_high));
      }
    }
  }
  best = fn22_wave_max(best);
  if ((tid & 31) == 0) red[tid >> 5] = best;
  __syncthreads();
  if (tid == 0) {
    unsigned long long b = red[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) b = fn22_max(b, red[i]);
    part[r * FN22_SPLITS + s] = b;
  }
}

// ------------------------------------------------------------------------------------------- pack
extern "C" __global__ __launch_bounds__(256) void fn_draft_pack(
    const unsigned long long *__restrict__ part, const int *__restrict__ sel,
    const uint16_t *__restrict__ embh, uint16_t *__restrict__ pay, int half, int rank,
    int tie_high) {
  __shared__ uint32_t win[2];
  const int r = blockIdx.x, tid = threadIdx.x;
  if (tid < 32) {
    unsigned long long v = part[r * FN22_SPLITS + tid];  // FN22_SPLITS == the wave's 32 lanes
    v = fn22_wave_max(v);
    if (tid == 0) {
      uint32_t key = (uint32_t)(v >> 32);
      uint32_t lidx = tie_high ? (uint32_t)v : ~(uint32_t)v;
      if (sel[r] < 0) {  // a pad row: token 0 (rank 0's row 0) wins the pair
        key = rank == 0 ? 0xffffu : 0u;
        lidx = 0;
      }
      win[0] = key;
      win[1] = lidx;
    }
  }
  __syncthreads();
  const uint32_t key = win[0], lidx = win[1];
  uint16_t *row = pay + (long)r * (2 * FN22_H + FN22_LANES);
  uint4 *mine = (uint4 *)(row + rank * FN22_H);
  uint4 *other = (uint4 *)(row + (1 - rank) * FN22_H);
  const uint4 *src = (const uint4 *)(embh + (long)lidx * FN22_H);
  const uint4 nz = make_uint4(0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u);
  for (int i = tid; i < FN22_H / 8; i += 256) {
    mine[i] = src[i];
    other[i] = nz;
  }
  if (tid < FN22_LANES) {
    const int owner = tid >> 3, j = tid & 7;
    uint16_t b = 0x8000;
    if (owner == rank) {
      const uint32_t gid = (uint32_t)rank * (uint32_t)half + lidx;
      const uint32_t byte = j == 0 ? key >> 8 : j == 1 ? key & 0xffu : j <= 4 ? (gid >> (8 * (j - 2))) & 0xffu : 0u;
      b = fn22_enc(byte);
    }
    row[2 * FN22_H + tid] = b;
  }
}

// ----------------------------------------------------------------------------------------- select
extern "C" __global__ __launch_bounds__(256) void fn_draft_select(
    const uint16_t *__restrict__ red, uint16_t *__restrict__ emb, uint32_t *__restrict__ ids,
    uint32_t *__restrict__ ctr, int rows, int tie_high) {
  const int tid = threadIdx.x;
  const uint32_t c = *ctr;
  for (int r = 0; r < rows; ++r) {
    const uint16_t *row = red + (long)r * (2 * FN22_H + FN22_LANES);
    const uint16_t *ln = row + 2 * FN22_H;
    const uint32_t k0 = (fn22_dec(ln[0]) << 8) | fn22_dec(ln[1]);
    const uint32_t k1 = (fn22_dec(ln[8]) << 8) | fn22_dec(ln[9]);
    const int w = tie_high ? (k1 >= k0) : (k1 > k0);
    const uint16_t *l = ln + 8 * w;
    const uint32_t gid = fn22_dec(l[2]) | (fn22_dec(l[3]) << 8) | (fn22_dec(l[4]) << 16);
    const uint4 *src = (const uint4 *)(row + w * FN22_H);
    uint4 *dst = (uint4 *)(emb + (long)r * FN22_H);
    for (int i = tid; i < FN22_H / 8; i += 256) dst[i] = src[i];
    if (tid == 0) ids[c * FN22_IDS_STRIDE + r] = gid;
  }
  __syncthreads();
  if (tid == 0) *ctr = c + 1;
}

// =============================================================================================
// FN33: the drafter's head at DRAFT precision -- the 27B's int2 scorer (dflash_head.hip) picks
// top-K candidates of the rank's vocabulary half, the target head's own bf16 GEMM re-scores them
// exactly, and the candidates' argmax feeds FN22's pack unchanged. The four kernels below are the
// glue; none of them does floating-point arithmetic on a logit (the quantiser runs once at boot).
//
//   fn_draft_x_gather       grid (xrows), block 256: xp[r] = x[sel[r]] for r < rows with
//                           sel[r] >= 0, zeros otherwise (the scorer's padded 16-row input and the
//                           rerank GEMM's A).
//   fn_draft_head_int2_quant grid (ceil(n * 5 / 256)), block 256: one thread per (row, group j of
//                           a quarter): groups j, j+5, j+10, j+15 (one per quarter) -> their scale
//                           pairs and the 128 packed bytes j*128 .. +128 (prod's
//                           `_quantize_head_now`, fp32, round-half-even; saber_kernels::fn33::quant_row
//                           is the host law).
//   fn_draft_wgather        grid (ncols), block 256: wc[o] = head[cand[r * cstride + j]] for
//                           o = r * kk + j < rows * kk, zeros past it.
//   fn_draft_argmax_cand    grid (rows), block 32: part[r * FN22_SPLITS + 0] = max over j < kk of
//                           key(sc[r * ldc + r * kk + j]) << 32 | ~cand[r * cstride + j] (FN22's
//                           law: the larger key, then the LOWER index), part[.. + 1..31] = 0; a pad
//                           row (sel[r] < 0) writes 0 everywhere (fn_draft_pack forces its token).
#define FN33_K 2560
#define FN33_G 128
#define FN33_Q (FN33_K / 4)
#define FN33_NQG (FN33_Q / FN33_G)   // groups per quarter: 5
#define FN33_NG (FN33_K / FN33_G)    // 20

extern "C" __global__ __launch_bounds__(256) void fn_draft_x_gather(
    const uint16_t *__restrict__ x, const int *__restrict__ sel, uint16_t *__restrict__ xp,
    int rows) {
  const int r = blockIdx.x, tid = threadIdx.x;
  const int s = r < rows ? sel[r] : -1;
  uint4 *dst = (uint4 *)(xp + (long)r * FN22_H);
  if (s >= 0) {
    const uint4 *src = (const uint4 *)(x + (long)s * FN22_H);
    for (int i = tid; i < FN22_H / 8; i += 256) dst[i] = src[i];
  } else {
    for (int i = tid; i < FN22_H / 8; i += 256) dst[i] = make_uint4(0u, 0u, 0u, 0u);
  }
}

__device__ __forceinline__ float fn33_f(uint16_t b) {
  return __builtin_bit_cast(float, (uint32_t)b << 16);
}

__device__ __forceinline__ uint16_t fn33_bf16(float v) {
  uint32_t a = __builtin_bit_cast(uint32_t, v);
  a += 0x7fffu + ((a >> 16) & 1u);
  return (uint16_t)(a >> 16);
}

extern "C" __global__ __launch_bounds__(256) void fn_draft_head_int2_quant(
    const uint16_t *__restrict__ w, unsigned char *__restrict__ wq, uint16_t *__restrict__ s,
    uint16_t *__restrict__ zs, int n) {
  const long t = (long)blockIdx.x * 256 + threadIdx.x;
  if (t >= (long)n * FN33_NQG) return;
  const long row = t / FN33_NQG;
  const int j = (int)(t - row * FN33_NQG);
  const uint16_t *wr = w + row * FN33_K;
  float sc[4], zp[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int g = q * FN33_NQG + j;
    const uint16_t *x = wr + g * FN33_G;
    float lo = INFINITY, hi = -INFINITY;
    for (int i = 0; i < FN33_G; ++i) {
      const float v = fn33_f(x[i]);
      lo = fminf(lo, v);
      hi = fmaxf(hi, v);
    }
    float c = (hi - lo) / 3.0f;
    c = c < 1e-8f ? 1e-8f : c;
    float z = rintf(-lo / c);
    z = z < 0.0f ? 0.0f : (z > 3.0f ? 3.0f : z);
    sc[q] = c;
    zp[q] = z;
    const float s4 = 4.0f * c;
    const float zz = zp[q] * c;
    s[row * FN33_NG + g] = fn33_bf16(s4);
    zs[row * FN33_NG + g] = fn33_bf16(s4 + zz);
  }
  unsigned char *out = wq + row * FN33_Q + j * FN33_G;
  for (int i = 0; i < FN33_G; ++i) {
    uint32_t b = 0;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float v = fn33_f(wr[q * FN33_Q + j * FN33_G + i]);
      float c = rintf(v / sc[q] + zp[q]);
      c = c < 0.0f ? 0.0f : (c > 3.0f ? 3.0f : c);
      b |= (uint32_t)c << (2 * q);
    }
    out[i] = (unsigned char)b;
  }
}

extern "C" __global__ __launch_bounds__(256) void fn_draft_wgather(
    const uint16_t *__restrict__ head, const int *__restrict__ cand, uint16_t *__restrict__ wc,
    int rows, int kk, int cstride) {
  const int o = blockIdx.x, tid = threadIdx.x;
  const int r = o / kk, j = o - r * kk;
  uint4 *dst = (uint4 *)(wc + (long)o * FN22_H);
  if (r < rows) {
    const uint4 *src = (const uint4 *)(head + (long)cand[r * cstride + j] * FN22_H);
    for (int i = tid; i < FN22_H / 8; i += 256) dst[i] = src[i];
  } else {
    for (int i = tid; i < FN22_H / 8; i += 256) dst[i] = make_uint4(0u, 0u, 0u, 0u);
  }
}

extern "C" __global__ __launch_bounds__(32) void fn_draft_argmax_cand(
    const uint16_t *__restrict__ sc, const int *__restrict__ cand, const int *__restrict__ sel,
    unsigned long long *__restrict__ part, int kk, int ldc, int cstride, int tie_high) {
  const int r = blockIdx.x, lane = threadIdx.x;
  unsigned long long best = 0;
  if (sel[r] >= 0) {  // workgroup-uniform
    for (int j = lane; j < kk; j += 32) {
      const uint32_t b = sc[(long)r * ldc + r * kk + j];
      best = fn22_max(best, fn22_pack(fn22_key(b), (uint32_t)cand[r * cstride + j], tie_high));
    }
  }
  best = fn22_wave_max(best);
  part[r * FN22_SPLITS + lane] = lane == 0 ? best : 0ull;
}

// =============================================================================================
// FN86d family 10 (F10): the drafter round's int2-tail glue in fewer launches -- ADDED symbols,
// each byte-equal to the chain it replaces (card-1 oracle `tools/capture/fn86d/fn86d_twin.py`);
// the shipped kernels above are untouched; the 27B helpers they reproduce (dflash_topk's dt80_*,
// dflash_head's group sum, dflash_mlp's act quant) are COPIED under fn86d_ names, and FN22's
// shared helpers are called with the shipped callers' arguments (width 32).
//
//   fn86_draft_xg_sum_q8        grid (fn33::XBUF_ROWS), block 256 == fn_draft_x_gather (xp) ->
//                               dflash_head_group_sum_g128 (xs: rows XROWS, stride NG) ->
//                               dflash_act_quant_g128 (q / qs of xp's first `rows` rows, n = H):
//                               workgroup r gathers row r; wave w owns its 128-groups w, w + 8,
//                               w + 16, lane l the elements l*4 .. +3 -- the group sum's ORDER 0
//                               lanes and the quant's lanes are the same elements.
//   fn86_topk_gather_cand       grid (rows), block 1024 == dflash_topk_f32_gather_k80 ->
//                               dflash_head_cand_gather (one workgroup a row in both; the row's
//                               candidate columns read `pos` after a barrier); the threshold
//                               (dflash_topk_f32_thresh_k80) stays the shipped launch.
//   fn86_draft_argmax_cand_pack grid (rows), block 256 == fn_draft_argmax_cand -> fn_draft_pack:
//                               wave 0 takes the candidates' argmax (part[.. 0] = best, 1..31 = 0
//                               as the chain stores it); the max of those 32 partials IS best.
//   _rc controls: the scale as a reciprocal multiply; the gather's key without the index (the
//   _notie order); the candidates' tie law flipped.

#define FN86D_NG 20      // FN33_NG: 128-groups of a 2,560 row
#define FN86D_XROWS 16   // fn33::XROWS: the scorer's padded rows (the group sums' rows)

template <bool RC>
__device__ __forceinline__ void fn86d_xg_sum_q8(
    const uint16_t *__restrict__ x, const int *__restrict__ sel, uint16_t *__restrict__ xp,
    int rows, float *__restrict__ xs, unsigned char *__restrict__ q, float *__restrict__ qs) {
  const int r = blockIdx.x, tid = threadIdx.x;
  const int s = r < rows ? sel[r] : -1;
  uint4 *dst = (uint4 *)(xp + (long)r * FN22_H);
  if (s >= 0) {
    const uint4 *src = (const uint4 *)(x + (long)s * FN22_H);
    for (int i = tid; i < FN22_H / 8; i += 256) dst[i] = src[i];
  } else {
    for (int i = tid; i < FN22_H / 8; i += 256) dst[i] = make_uint4(0u, 0u, 0u, 0u);
  }
  const bool sum = r < FN86D_XROWS, quant = r < rows;
  if (!sum && !quant) return;  // workgroup-uniform
  const int lane = tid & 31, wave = tid >> 5;
  for (int g = wave; g < FN86D_NG; g += 8) {
    const int j = g * 128 + lane * 4;
    uint2 raw = make_uint2(0u, 0u);
    if (s >= 0) raw = *(const uint2 *)(x + (long)s * FN22_H + j);
    const uint32_t w2[2] = {raw.x, raw.y};
    float ob[4];
#pragma unroll
    for (int e = 0; e < 4; ++e)
      ob[e] = __builtin_bit_cast(float, ((w2[e >> 1] >> (16 * (e & 1))) & 0xffffu) << 16);
    if (sum) {
      // dflash_head_group_sum_impl<0>: four accumulators from 0 + v, folded in order, then the
      // ascending-offset tree in lane order
      float v[4];
#pragma unroll
      for (int i = 0; i < 4; i++) v[i] = 0.0f;
      if (RC) {
        // the control also sums the group in the strided lane order (`_strided`, ORDER 1)
        const uint16_t *gb = x + (long)(s >= 0 ? s : 0) * FN22_H + g * 128;
#pragma unroll
        for (int i = 0; i < 4; i++)
          v[i] = v[i] + (s >= 0 ? __builtin_bit_cast(float, (uint32_t)gb[lane + i * 32] << 16)
                                : 0.0f);
      } else {
#pragma unroll
        for (int i = 0; i < 4; i++) v[i] = v[i] + ob[i];
      }
#pragma unroll
      for (int i = 1; i < 4; i++) v[0] = v[0] + v[i];
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) v[0] = v[0] + __shfl_down(v[0], off, 32);
      if (lane == 0) xs[(long)r * FN86D_NG + g] = v[0];
    }
    if (quant) {
      // dflash_act_quant_g128 (DQ_BODY: a / 448, the value / sc)
      float a = 0.f;
#pragma unroll
      for (int e = 0; e < 4; ++e) a = fmaxf(a, fabsf(ob[e]));
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) a = fmaxf(a, __shfl_xor(a, d, 32));
      const float sc = fmaxf(a / 448.f, 1.f / (448.f * 512.f));
      if (lane == 0) qs[(long)r * FN86D_NG + g] = sc;
      unsigned int packed = 0;
      if (RC) {
        const float rsc = 1.f / sc;
#pragma unroll
        for (int e = 0; e < 4; e += 2) {
          const float u = fminf(fmaxf(ob[e] * rsc, -448.f), 448.f);
          const float v = fminf(fmaxf(ob[e + 1] * rsc, -448.f), 448.f);
          packed |= (__builtin_amdgcn_cvt_pk_fp8_f32(u, v, 0u, false) & 0xffffu) << (e * 8);
        }
      } else {
#pragma unroll
        for (int e = 0; e < 4; e += 2) {
          const float u = fminf(fmaxf(ob[e] / sc, -448.f), 448.f);
          const float v = fminf(fmaxf(ob[e + 1] / sc, -448.f), 448.f);
          packed |= (__builtin_amdgcn_cvt_pk_fp8_f32(u, v, 0u, false) & 0xffffu) << (e * 8);
        }
      }
      *(unsigned int *)(q + (long)r * FN22_H + j) = packed;
    }
  }
}

extern "C" __global__ __launch_bounds__(256) void fn86_draft_xg_sum_q8(
    const uint16_t *__restrict__ x, const int *__restrict__ sel, uint16_t *__restrict__ xp,
    int rows, float *__restrict__ xs, unsigned char *__restrict__ q, float *__restrict__ qs) {
  fn86d_xg_sum_q8<false>(x, sel, xp, rows, xs, q, qs);
}

extern "C" __global__ __launch_bounds__(256) void fn86_draft_xg_sum_q8_rc(
    const uint16_t *__restrict__ x, const int *__restrict__ sel, uint16_t *__restrict__ xp,
    int rows, float *__restrict__ xs, unsigned char *__restrict__ q, float *__restrict__ qs) {
  fn86d_xg_sum_q8<true>(x, sel, xp, rows, xs, q, qs);
}

// ---- the top-80 gather + the candidate columns (dflash_topk.hip's dt80_key / dt80_mono COPIED) --
// t2 measured the three-kernel form (threshold + gather + columns in one workgroup) at 52 vs 27 us
// for the chain (not attributed): the threshold stays the shipped launch, the gather and the
// candidate-column gather are ONE (both one workgroup a row already).
#define FN86D_NT 1024
#define FN86D_K 80

__device__ __forceinline__ unsigned int fn86d_mono(float f) {
  unsigned int b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ unsigned long long fn86d_key(float v, int i) {
  return ((unsigned long long)fn86d_mono(v) << 32) | (unsigned int)(~(unsigned int)i);
}

template <bool NOTIE>
__device__ __forceinline__ void fn86d_gather_cand(
    const float *__restrict__ x, const unsigned long long *__restrict__ thresh,
    float *__restrict__ vals, int *__restrict__ idx, const int *__restrict__ bidx,
    int *__restrict__ cand, int rows, int n, int k) {
  const int row = (int)blockIdx.x;
  if (row >= rows || k <= 0) return;
  const int tid = (int)threadIdx.x;
  __shared__ unsigned long long selk[FN86D_K];
  __shared__ int nsel;
  // dflash_topk_f32_gather_k80 (the control = `_notie`'s order), line for line
  if (k <= FN86D_K) {
    if (tid == 0) nsel = 0;
    __syncthreads();
    const float *xr = x + (long)row * n;
    if (NOTIE) {
      const unsigned int thv = (unsigned int)(thresh[row] >> 32);
      for (int e = tid; e < n; e += FN86D_NT) {
        const unsigned int mv = fn86d_mono(xr[e]);
        if (mv >= thv) {
          const int p = atomicAdd(&nsel, 1);
          if (p < FN86D_K) selk[p] = ((unsigned long long)mv << 32) | (unsigned int)e;
        }
      }
    } else {
      const unsigned long long th = thresh[row];
      for (int e = tid; e < n; e += FN86D_NT) {
        const unsigned long long key = fn86d_key(xr[e], e);
        if (key >= th) {
          const int p = atomicAdd(&nsel, 1);
          if (p < FN86D_K) selk[p] = key;
        }
      }
    }
    __syncthreads();
    const int m = nsel < k ? nsel : k;
    if (tid < m) {
      const unsigned long long mine = selk[tid];
      int rank = 0;
      for (int j = 0; j < m; ++j) rank += (selk[j] > mine) ? 1 : 0;
      const int e = NOTIE ? (int)(unsigned int)(mine & 0xffffffffu)
                          : (int)(~(unsigned int)(mine & 0xffffffffu));
      vals[(long)row * k + rank] = xr[e];
      idx[(long)row * k + rank] = e;
    }
  }
  // dflash_head_cand_gather: the row's k columns, `pos` read back after the barrier
  __syncthreads();
  for (int i = tid; i < k; i += FN86D_NT) {
    const long o = (long)row * k + i;
    const int p = idx[o];
    cand[o] = (p >= 0 && p < n) ? bidx[(long)row * n + p] : 0;
  }
}

extern "C" __global__ __launch_bounds__(FN86D_NT) void fn86_topk_gather_cand(
    const float *__restrict__ x, const unsigned long long *__restrict__ thresh,
    float *__restrict__ vals, int *__restrict__ idx, const int *__restrict__ bidx,
    int *__restrict__ cand, int rows, int n, int k) {
  fn86d_gather_cand<false>(x, thresh, vals, idx, bidx, cand, rows, n, k);
}

extern "C" __global__ __launch_bounds__(FN86D_NT) void fn86_topk_gather_cand_rc(
    const float *__restrict__ x, const unsigned long long *__restrict__ thresh,
    float *__restrict__ vals, int *__restrict__ idx, const int *__restrict__ bidx,
    int *__restrict__ cand, int rows, int n, int k) {
  fn86d_gather_cand<true>(x, thresh, vals, idx, bidx, cand, rows, n, k);
}

// ---- the candidates' argmax + the pack --------------------------------------------------------
template <bool RC>
__device__ __forceinline__ void fn86d_argmax_cand_pack(
    const uint16_t *__restrict__ sc, const int *__restrict__ cand, const int *__restrict__ sel,
    unsigned long long *__restrict__ part, int kk, int ldc, int cstride,
    const uint16_t *__restrict__ embh, uint16_t *__restrict__ pay, int half, int rank,
    int tie_high) {
  __shared__ uint32_t win[2];
  const int r = blockIdx.x, tid = threadIdx.x;
  if (tid < 32) {
    // fn_draft_argmax_cand (the control flips its tie law only)
    const int th = RC ? !tie_high : tie_high;
    unsigned long long best = 0;
    if (sel[r] >= 0) {  // workgroup-uniform
      for (int j = tid; j < kk; j += 32) {
        const uint32_t b = sc[(long)r * ldc + r * kk + j];
        best = fn22_max(best, fn22_pack(fn22_key(b), (uint32_t)cand[r * cstride + j], th));
      }
    }
    best = fn22_wave_max(best);
    if (RC && sel[r] >= 0) {  // the control's winner, re-encoded in the served tie law's index form
      const uint32_t id = th ? (uint32_t)best : ~(uint32_t)best;
      best = (best & 0xffffffff00000000ull) | (tie_high ? id : ~id);
    }
    part[r * FN22_SPLITS + tid] = tid == 0 ? best : 0ull;
    // fn_draft_pack's winner: the max over the 32 partials {best, 0, ..} is best
    if (tid == 0) {
      uint32_t key = (uint32_t)(best >> 32);
      uint32_t lidx = tie_high ? (uint32_t)best : ~(uint32_t)best;
      if (sel[r] < 0) {  // a pad row: token 0 (rank 0's row 0) wins the pair
        key = rank == 0 ? 0xffffu : 0u;
        lidx = 0;
      }
      win[0] = key;
      win[1] = lidx;
    }
  }
  __syncthreads();
  const uint32_t key = win[0], lidx = win[1];
  uint16_t *row = pay + (long)r * (2 * FN22_H + FN22_LANES);
  uint4 *mine = (uint4 *)(row + rank * FN22_H);
  uint4 *other = (uint4 *)(row + (1 - rank) * FN22_H);
  const uint4 *src = (const uint4 *)(embh + (long)lidx * FN22_H);
  const uint4 nz = make_uint4(0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u);
  for (int i = tid; i < FN22_H / 8; i += 256) {
    mine[i] = src[i];
    other[i] = nz;
  }
  if (tid < FN22_LANES) {
    const int owner = tid >> 3, j = tid & 7;
    uint16_t b = 0x8000;
    if (owner == rank) {
      const uint32_t gid = (uint32_t)rank * (uint32_t)half + lidx;
      const uint32_t byte = j == 0 ? key >> 8 : j == 1 ? key & 0xffu : j <= 4 ? (gid >> (8 * (j - 2))) & 0xffu : 0u;
      b = fn22_enc(byte);
    }
    row[2 * FN22_H + tid] = b;
  }
}

extern "C" __global__ __launch_bounds__(256) void fn86_draft_argmax_cand_pack(
    const uint16_t *__restrict__ sc, const int *__restrict__ cand, const int *__restrict__ sel,
    unsigned long long *__restrict__ part, int kk, int ldc, int cstride,
    const uint16_t *__restrict__ embh, uint16_t *__restrict__ pay, int half, int rank,
    int tie_high) {
  fn86d_argmax_cand_pack<false>(sc, cand, sel, part, kk, ldc, cstride, embh, pay, half, rank,
                                tie_high);
}

extern "C" __global__ __launch_bounds__(256) void fn86_draft_argmax_cand_pack_rc(
    const uint16_t *__restrict__ sc, const int *__restrict__ cand, const int *__restrict__ sel,
    unsigned long long *__restrict__ part, int kk, int ldc, int cstride,
    const uint16_t *__restrict__ embh, uint16_t *__restrict__ pay, int half, int rank,
    int tie_high) {
  fn86d_argmax_cand_pack<true>(sc, cand, sel, part, kk, ldc, cstride, embh, pay, half, rank,
                               tie_high);
}
