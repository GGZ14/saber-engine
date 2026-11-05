// CMP27Z: the drafter's int2 coarse head at 64 rows in ONE weight pass (CMP27J candidate (b1)).
//
// The served route at a b5..b8 drafter round (m_pad 64) is the 32-row `_lx` band on a y extent of
// 2: two workgroups a 128-column pair, each decoding the SAME int2 weight bytes for its own 32
// rows. This twin is the band at MROWS = 64 -- four m-fragments fed from each decoded B fragment,
// the quarter's [64, 128] A slice staged once in LDS -- and nothing else: every output element's
// eight WMMA k-steps run in the same ascending order into a fresh tile, `acc += dot * S;
// acc -= XS * ZS` are the same two expressions (contracted exactly as the band's: this object is
// built at the default contraction, like `dflash_head`), every y / bmax / bidx is the band's
// bytes by construction and gated anyway (`tools/capture/cmp27z/head_probe.py`). The emit's
// per-row scan is the band's, run over two 32-row halves through the band's own tile map.
//
// Its own code object so `dflash_head` keeps its bytes. The helpers below (geometry, WMMA, the
// int2 dequant bit pattern, the emit's tile map and tie law) are `dflash_head.hip`'s, verbatim.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

typedef __bf16 bf16x4 __attribute__((ext_vector_type(4)));
typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float floatx8 __attribute__((ext_vector_type(8)));
typedef unsigned int uint2_t __attribute__((ext_vector_type(2)));
typedef unsigned int uint4_t __attribute__((ext_vector_type(4)));

#define DH_M      16   // BLOCK_M: one WMMA M tile, and `tl.dot`'s minimum
#define DH_MB     32   // the SECOND band's BLOCK_M: two M tiles in one workgroup's accumulator
#define DH_N      64   // BLOCK_N: prod's do_bench optimum, and the width of one block-max entry
#define DH_G     128   // the quantisation group along K, and the kernel's k tile
#define DH_KC      8   // candidates emitted per block; KCAND
#define DH_NW      2   // waves per workgroup: prod's num_warps at the M=16 band
#define DH_NT   (DH_NW * 32)
#define DH_NF   (DH_N / 16 / DH_NW)   // n-fragments per wave: 2

// One 16x16x16 bf16 WMMA, the instruction Triton's `tl.dot` lowers to on gfx1201.
__device__ __forceinline__ floatx8 dh_wmma(bf16x8 a, bf16x8 b, floatx8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

// The gfx12 w32 fragment and accumulator layout, from this tree's own WMMA code (par_kernels.h):
// an A or B fragment holds row `l & 15`, k elements `(l >> 4) * 8 .. +8`; accumulator element `e`
// of lane `l` is row `(l >> 4) * 8 + e`, column `l & 15`. At BLOCK_M = 16 the two half-waves carry
// rows 0..7 and 8..15, so one tile is every row.
__device__ __forceinline__ int dh_acc_row(int lane, int e) { return (lane >> 4) * 8 + e; }

// ---- the int2 dequant, as a bit pattern ------------------------------------------------------
// Eight packed bytes -> eight bf16, the exact values `0x3F80 | ((b << (5 - 2q)) & 0x60)` names, in
// four instructions per input dword and with nothing held between quarters.
//
// The shape of it: the wanted 16-bit result is `0x3F` in its high byte and `0x80 | (v << 5)` in its
// low byte, so the low bytes of FOUR results can be made at once on the packed dword --
// `((d << (5 - 2q)) & 0x60606060) | 0x80808080`, where the mask is what confines each byte to its
// own two bits after a shift that crosses byte boundaries -- and then two `v_perm_b32` interleave
// those four bytes with a constant `0x3F` to give the four 16-bit values. One shift, one `v_and_or`,
// two perms.
//
// The first draft instead spread the bytes into 16-bit lanes ONCE per group and kept the result
// across the four quarters, mirroring prod's own hoist. It is the cheaper instruction count on
// paper and it was the wrong trade here: the spread tile is 64 VGPRs, which took the kernel to 256
// registers and 204 bytes of scratch -- the weight stream through memory twice. Recomputing four
// times from 32 registers of raw bytes is faster than spilling once.
template <int Q>
__device__ __forceinline__ uint4_t dh_decode(uint2_t raw) {
  uint4_t o;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const unsigned int t =
        (((Q < 3) ? (raw[h] << (5 - 2 * Q)) : (raw[h] >> 1)) & 0x60606060u) | 0x80808080u;
    o[h * 2 + 0] = __builtin_amdgcn_perm(0x3F3F3F3Fu, t, 0x04010400u);
    o[h * 2 + 1] = __builtin_amdgcn_perm(0x3F3F3F3Fu, t, 0x04030402u);
  }
  return o;
}

// ---- the emit: top-KC of this block's 64 columns, per row ------------------------------------
// Prod's `_emit`, which is a successive max-and-mask rather than a token-level top-k: the exact
// winner is always the maximum of its own block, so block maxima carry the rerank candidates at
// 1/BLOCK_N the selection width (65 us against 105 for a full-row topk). `tl.argmax` breaks ties to
// the LOWEST index, so the combine below prefers the lower index on an exact tie -- which makes the
// reduction order-independent as well as tie-compatible.
//
// The accumulator tile goes to LDS once so that one wave can see all 64 columns of a row. The
// address map is `c * 17 + row + (c >> 5) * 16`, and both halves of it are load-bearing: the
// stride of 17 makes a read of 16 different rows at one column hit 16 different banks, and the
// `+16` per 32-column half makes the two halves of the reading wave hit the other 16 -- so the
// scan, which is the hot part, is conflict-free across all 32 lanes.
//
// PARAMETERISED ON THE BAND'S ROW COUNT, and the 16-row map above is what it prints at MROWS = 16:
// the column stride is `MROWS + 1` (17 there, 33 at the 32-row band, both odd so one column's rows
// cover consecutive banks) and the `+16` per 32-column half is unchanged, because the reading wave
// is 16 rows x 2 halves whatever the band is -- at MROWS = 32 there are simply TWO such waves, one
// per 16-row group, and each of them wants its own 16 banks against the other half.
#define DH_TILE_MR(MR, row, c) ((c) * ((MR) + 1) + (row) + ((c) >> 5) * 16)
#define DH_TILE_MR_WORDS(MR) (63 * ((MR) + 1) + (MR) - 1 + 16 + 1)
#define DH_TILE(row, c) DH_TILE_MR(DH_M, row, c)
#define DH_TILE_WORDS DH_TILE_MR_WORDS(DH_M)

// Branchless: prod's `tl.argmax` breaks ties to the LOWEST index, and preferring the lower index on
// an exact tie makes the combine order-independent as well, so the two halves can be reduced in
// either direction. Written as compares and selects rather than an `if`, which on gfx12 would be an
// exec-mask save/restore per candidate.
__device__ __forceinline__ void dh_better(float &bv, int &bi, float v, int i) {
  const bool take = (v > bv) | ((v == bv) & (i < bi));
  bv = take ? v : bv;
  bi = take ? i : bi;
}

// The `_lx` tile (dflash_head.hip defines these beside the band).
#define DH_LXB    2
#define DH_LXNT   (DH_LXB * DH_NT)
#define DH_LXS    (DH_G + 8)

template <bool TOPK_ONLY, int MROWS>
__device__ __forceinline__ void dflash_draft_head_int2_m64_impl(
    const __bf16 *__restrict__ x, const float *__restrict__ xs,
    const unsigned char *__restrict__ wq, const __bf16 *__restrict__ s,
    const __bf16 *__restrict__ zs, __bf16 *__restrict__ y, float *__restrict__ bmax,
    int *__restrict__ bidx, int m, int n, int k, int stride_wq, int stride_s, int stride_xs,
    int nblk) {
  constexpr int NG = DH_G / 16;
  constexpr int MF = MROWS / DH_M;
  constexpr int CPR = DH_G / 8;                          // 16-byte chunks per staged row: 16
  constexpr int CH = MROWS * CPR / DH_LXNT;              // chunks per thread per quarter: 2 or 4
  static_assert(CH * DH_LXNT == MROWS * CPR, "the stage must split evenly over the workgroup");
  constexpr int STAGE_BYTES = MROWS * DH_LXS * 2;
  constexpr int ER = 32;                                 // the emit runs in two 32-row halves
  constexpr int TILE_WORDS = DH_TILE_MR_WORDS(ER);
  constexpr int TILES_BYTES = DH_LXB * TILE_WORDS * 4;
  constexpr int LDS_BYTES = STAGE_BYTES > TILES_BYTES ? STAGE_BYTES : TILES_BYTES;
  __shared__ __attribute__((aligned(16))) unsigned char lds[LDS_BYTES];
  // The group's XS for all four quarters and all MROWS rows, staged once a group (raw bits).
  __shared__ float xsl[4 * MROWS];
  __bf16 *const stage = (__bf16 *)lds;

  const int m0 = (int)blockIdx.y * MROWS;
  if (m0 >= m) return;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int bw = wave >> 1, lw = wave & 1;               // block within the workgroup, wave within it
  const int pid = (int)blockIdx.x * DH_LXB + bw;         // the shipped band's block index
  const bool blive = pid < nblk;
  const int n0 = pid * DH_N;
  const int q4 = k / 4;
  const int ngrp = q4 / DH_G;
  const int col0 = n0 + lw * 32;
  const int mycol = col0 + (lane & 15);
  float *const tile = (float *)lds + bw * TILE_WORDS;

  floatx8 acc[MF][DH_NF];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf)
#pragma unroll
    for (int f = 0; f < DH_NF; ++f)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[mf][f][e] = 0.f;

  const bool live = blive && mycol < n;
  const int cbase = live ? mycol : 0;
  const unsigned char *wp0 = wq + (long)cbase * stride_wq + (lane >> 4) * 8;
  const unsigned char *wp1 = wp0 + 16L * stride_wq;
  static_assert(4 * MROWS == 2 * DH_LXNT, "two XS words a thread a group");
  const float *const xsg = xs + (long)(m0 + (tid % MROWS)) * stride_xs + (tid / MROWS) * 2 * ngrp;
  float *const xso = xsl + (tid / MROWS) * 2 * MROWS + (tid % MROWS);
  const float *const xsr = xsl + (lane >> 4) * 8;
  const __bf16 *sp = s + (long)cbase * stride_s;
  const __bf16 *zp = zs + (long)cbase * stride_s;

  // The staging map: chunk c = tid + i * DH_LXNT is row c / 16, k chunk c % 16 of the quarter,
  // i.e. row `tid / 16 + 8 * i` at chunk `tid % 16`: one base pointer and constant row steps.
  constexpr int RSTEP = DH_LXNT / CPR;                   // 8 rows per chunk round
  const __bf16 *const xg = x + (long)(m0 + tid / CPR) * k + (tid % CPR) * 8;
  const long xstep = (long)RSTEP * k;
  __bf16 *const so = stage + (tid / CPR) * DH_LXS + (tid % CPR) * 8;
  // Where this lane's A fragment sits in the stage (m-fragment 0).
  const int ab = (lane & 15) * DH_LXS + (lane >> 4) * 8;

  for (int g = 0; g < ngrp; ++g) {
    uint2_t raw[DH_NF][NG];
#pragma unroll
    for (int t = 0; t < NG; ++t) {
      raw[0][t] = *(const uint2_t *)(wp0 + t * 16);
      raw[1][t] = *(const uint2_t *)(wp1 + t * 16);
    }
    const float xs0 = xsg[g], xs1 = xsg[ngrp + g];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      // This quarter's chunks: loaded, then (once every wave is done reading the previous
      // quarter) stored. No prefetch across the k-steps: holding the next quarter's chunks
      // through the WMMAs spilled (152-224 B of scratch); the other resident workgroups cover
      // the load's latency instead.
      // The stage in two halves of CH / 2 chunks (16 VGPRs of x in flight, not 32).
      __syncthreads();
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        uint4_t xr[CH / 2];
#pragma unroll
        for (int i = 0; i < CH / 2; ++i)
          xr[i] = *(const uint4_t *)(xg + (hh * (CH / 2) + i) * xstep + q * q4 + g * DH_G);
#pragma unroll
        for (int i = 0; i < CH / 2; ++i)
          *(uint4_t *)(so + (hh * (CH / 2) + i) * RSTEP * DH_LXS) = xr[i];
      }
      if (q == 0) {
        xso[0] = xs0;
        xso[MROWS] = xs1;
      }
      __syncthreads();
      // The band's k loop: per k-step the four A fragments from the stage, each decoded B
      // fragment fed to the four of them (both n-fragments' dot tiles live: the CMP27Z probe's
      // faster form; an n-fragment-at-a-time loop measured tied with the served pair).
      const int gq = q * ngrp;
      floatx8 dot[MF][DH_NF];
#pragma unroll
      for (int mf = 0; mf < MF; ++mf)
#pragma unroll
        for (int f = 0; f < DH_NF; ++f)
#pragma unroll
          for (int e = 0; e < 8; ++e) dot[mf][f][e] = 0.f;
#pragma unroll
      for (int t = 0; t < NG; ++t) {
        bf16x8 xv[MF];
#pragma unroll
        for (int mf = 0; mf < MF; ++mf)
          xv[mf] = *(const bf16x8 *)(stage + mf * DH_M * DH_LXS + ab + t * 16);
#pragma unroll
        for (int f = 0; f < DH_NF; ++f) {
          const uint4_t w4 = (q == 0)   ? dh_decode<0>(raw[f][t])
                             : (q == 1) ? dh_decode<1>(raw[f][t])
                             : (q == 2) ? dh_decode<2>(raw[f][t])
                                        : dh_decode<3>(raw[f][t]);
          const bf16x8 wv = __builtin_bit_cast(bf16x8, w4);
#pragma unroll
          for (int mf = 0; mf < MF; ++mf) dot[mf][f] = dh_wmma(xv[mf], wv, dot[mf][f]);
        }
      }
#pragma unroll
      for (int f = 0; f < DH_NF; ++f) {
        const float sv = (float)sp[f * 16 * stride_s + gq];
        const float zv = (float)zp[f * 16 * stride_s + gq];
#pragma unroll
        for (int mf = 0; mf < MF; ++mf)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const float xsv = xsr[q * MROWS + mf * DH_M + e];
            acc[mf][f][e] += dot[mf][f][e] * sv;
            acc[mf][f][e] -= xsv * zv;
          }
      }
    }
    wp0 += DH_G;
    wp1 += DH_G;
    sp += 1;
    zp += 1;
  }

  // THE EMIT IN TWO 32-ROW HALVES, each through the 32-row band's own [32, 64] tile map (half the
  // LDS a 64-row tile would hold): the first half's y stores and tile, the band's scan over those
  // 32 rows, then the second half's. Rows are independent, so which pass scans which rows moves
  // nothing; every row's scan is the band's, column for column.
#pragma unroll
  for (int hf = 0; hf < MF / 2; ++hf) {
    __syncthreads();                        // the tiles alias the stage / the previous half's scan
#pragma unroll
    for (int ml = 0; ml < 2; ++ml) {
      const int mf = hf * 2 + ml;
#pragma unroll
      for (int f = 0; f < DH_NF; ++f) {
        const int c = mycol + f * 16;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int row = mf * DH_M + dh_acc_row(lane, e);
          if (live)
            y[(long)(m0 + row) * n + c] = TOPK_ONLY ? (__bf16)(-INFINITY) : (__bf16)acc[mf][f][e];
          tile[DH_TILE_MR(ER, ml * DH_M + dh_acc_row(lane, e), lw * 32 + (lane & 15) + f * 16)] =
              live ? acc[mf][f][e] : -INFINITY;
        }
      }
    }
    __syncthreads();
    if (blive) {
      const int lrow = lw * DH_M + (lane & 15), half = lane >> 4;
      const int row = hf * ER + lrow;
      for (int c = 0; c < DH_KC; ++c) {
        float bv = -INFINITY;
        int bi = -1;
#pragma unroll 8
        for (int j = 0; j < 32; ++j)
          dh_better(bv, bi, tile[DH_TILE_MR(ER, lrow, half * 32 + j)], half * 32 + j);
        const float ov = __builtin_bit_cast(
            float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, bv), 0x401F));
        const int oi = __builtin_amdgcn_ds_swizzle(bi, 0x401F);
        float wv = bv;
        int wi = bi;
        dh_better(wv, wi, ov, oi);
        if (half == 0) {
          bmax[(long)(m0 + row) * (nblk * DH_KC) + pid * DH_KC + c] = wv;
          bidx[(long)(m0 + row) * (nblk * DH_KC) + pid * DH_KC + c] = n0 + wi;
        }
        if (wi >> 5 == half) tile[DH_TILE_MR(ER, lrow, wi)] = -INFINITY;
      }
    }
  }
}

extern "C" __global__ __launch_bounds__(DH_LXNT) void dflash_draft_head_int2_m64n64x2g128_lx(
    const __bf16 *__restrict__ x, const float *__restrict__ xs,
    const unsigned char *__restrict__ wq, const __bf16 *__restrict__ s,
    const __bf16 *__restrict__ zs, __bf16 *__restrict__ y, float *__restrict__ bmax,
    int *__restrict__ bidx, int m, int n, int k, int stride_wq, int stride_s, int stride_xs,
    int nblk) {
  dflash_draft_head_int2_m64_impl<false, 64>(x, xs, wq, s, zs, y, bmax, bidx, m, n, k,
                                             stride_wq, stride_s, stride_xs, nblk);
}

