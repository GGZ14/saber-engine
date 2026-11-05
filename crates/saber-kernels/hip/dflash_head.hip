// The DFlash2 drafter's int2 draft head and its exact rerank.
//
// NOT a port. Prod's `ref/ref_drafthead.py` is a `@triton.jit` module, so there is
// no device source to slice by anchor the way `saber_*` and `paro_*` are sliced. These two kernels are
// written from that module's semantics and gated bit-exact against its output on captured inputs,
// the method `attn_glue` used against prod's Triton + ATen. The census put `_draft_head_int2` at
// 397.90 us x 2 = 795.8 us, 4.0% of a decode step -- the largest single kernel in the drafter path.
//
// PROD'S GEOMETRY, out of the trace and confirmed by the capture's own launch line:
//   grid 124160x1x1 at 64 threads  =  1940 workgroups x 2 waves
//   N 124160 (the 248320 vocab halved by TP=2), K 5120, M 16, G 128, BLOCK_N 64, KCAND 8
// M is 16 because `_pow2_at_least` pads the draft rows up to `tl.dot`'s 16-row minimum, which is
// also exactly one WMMA M tile -- so one accumulator tile per workgroup covers every row.
//
// THE WEIGHT FORMAT, from `_quantize_head_now`: int2, asymmetric, one (row, group-of-128) scale
// pair, QUARTER-SPLIT packed -- byte j of a row carries k = j, K/4+j, K/2+j and 3K/4+j -- so one
// byte load serves four contiguous k ranges. The dequant is a bit pattern, not a convert:
// `0x3F80 | ((b << (5 - 2q)) & 0x60)` reads as the bf16 value 1 + v/4. The 1.0 bias is exact rather
// than approximate: the dot then returns sum_k x_k + dot(x, v)/4, and the group's sum of x is
// already on hand for the zero point, so prod folds both into premultiplied scales
// (`S = 4*sc`, `ZS = 4*sc + zp*sc`) and the per-group correction is two accumulator ops.
//
// THE ARITHMETIC ORDER IS THE GATE. Per (quarter q, group g), in this order and no other:
//   tile  = sum over 8 WMMA k-steps of dot_f32(x[16,16], w[16,16])   -- a FRESH fp32 tile
//   acc  += tile * S[n][gi]                                          -- multiply, then add
//   acc  -= XS[m][gi] * ZS[n][gi]                                    -- multiply, then subtract
// Folding the scale after the outer accumulate, or contracting either multiply-add into an FMA, is
// a different answer -- which is why this file is built with `-ffp-contract=off` (the flag the
// manifest records, and the same one the MXFP4 kernels rely on).
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

// THE EPILOGUE, ABSORBED. For a TOP-K caller -- which is what DFlash2's
// `candidate_logits_processor` is -- prod's `_apply_head_int2` does
//
//     if self._ref_topk_only: y.fill_(float("-inf"))
//     y.scatter_(1, idx.long(), ex.to(torch.bfloat16))
//
// so every coarse logit this kernel computes is OVERWRITTEN before anything reads it: the coarse
// pass exists only to produce the block maxima the rerank draws its candidates from. `TOPK_ONLY`
// therefore stores -inf where the unfused path stores a coarse value -- the same bytes, and the
// separate 248320-wide `fill_` launch disappears. The scatter cannot live here (it needs the
// reranked values, which do not exist yet); it is absorbed into the rerank instead, which already
// visits exactly the positions the scatter writes.
//
// MROWS IS THE BAND, and it is the one thing prod carries as a different instantiation. At 16 it is
// the traced launch and every expression below folds to the one it had before this parameter
// existed; at 32 one workgroup holds a SECOND accumulator fragment and reads each weight byte once
// for thirty-two rows instead of once per sixteen -- which is the whole point, because the head's
// 159 MiB int2 weight is the launch's cost and a row-slice grid reads it once per m tile. What
// MROWS does NOT change is the arithmetic: `acc[mf][f][e]` accumulates `dot * S - XS * ZS` over the
// same ten groups of each of the same four quarters in the same order, and the `tl.dot` inside a
// group is the same eight ascending WMMA k-steps, so each output element is computed exactly as the
// 16-row band computes it. That is what makes "bit-identical per 16-row slice" a property of this
// SOURCE rather than a hope about a rewrite, and the gate measures it from both ends.
template <bool TOPK_ONLY, bool MT, int MROWS>
__device__ __forceinline__ void dflash_draft_head_int2_impl(
    const __bf16 *__restrict__ x,           // [M, K] bf16 hidden states
    const float *__restrict__ xs,           // [M, K/G] fp32 per-group sums of x
    const unsigned char *__restrict__ wq,   // [N, K/4] int2, quarter-split packed
    const __bf16 *__restrict__ s,           // [N, K/G] bf16 premultiplied scale  (4*sc)
    const __bf16 *__restrict__ zs,          // [N, K/G] bf16 premultiplied zero   (4*sc + zp*sc)
    __bf16 *__restrict__ y,                 // [M, N] bf16 coarse logits
    float *__restrict__ bmax,               // [M, NBLK*KC] fp32 block maxima
    int *__restrict__ bidx,                 // [M, NBLK*KC] i32 their column indices
    int m, int n, int k, int stride_wq, int stride_s, int stride_xs, int nblk) {
  constexpr int NG = DH_G / 16;             // WMMA k-steps per group: 8
  constexpr int MF = MROWS / DH_M;          // accumulator m-fragments: 1 at the traced band, 2
  // `m` is carried for the descriptor's sake and used only as a bound: prod pads the draft rows up
  // to BLOCK_M with `_pow2_at_least` and allocates y/bmax/bidx at that padded height, so every one
  // of a tile's 16 rows is a real row. `dflash::resolve` refuses an m the tiles do not cover
  // exactly -- the untiled symbol takes only m == 16, the M-tiled one only a whole number of tiles.
  //
  // WHICH SIXTEEN ROWS THIS WORKGROUP OWNS. `MT` is a template parameter, so `m0` is the literal 0
  // in the untiled instantiation and every address below folds back to the expression it had
  // before the tile existed: the k loop, the four quarters, the two accumulator ops per group and
  // the emit's scan are the same program on the same sixteen rows, which is what makes
  // "bit-identical per 16-row slice" a property of the SOURCE.
  const int m0 = MT ? (int)blockIdx.y * MROWS : 0;
  if (MT && m0 >= m) return;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int pid = blockIdx.x;
  const int n0 = pid * DH_N;                // this block's 64 columns
  const int q4 = k / 4;                     // Q: one quarter of K
  const int ngrp = q4 / DH_G;               // groups per quarter (10 at K = 5120)
  const int col0 = n0 + wave * 32;          // this wave's 32 columns
  // Column this lane owns in fragment nf, and the row-major offsets that follow from it.
  const int mycol = col0 + (lane & 15);
  __shared__ float tile[DH_TILE_MR_WORDS(MROWS)];

  floatx8 acc[MF][DH_NF];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf)
#pragma unroll
    for (int f = 0; f < DH_NF; ++f)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[mf][f][e] = 0.f;

  const bool live = mycol < n;              // prod's `mask_n`; N % 64 == 0 in prod, kept anyway

  // ---- hoisted addresses ---------------------------------------------------------------------
  // Every access in the group loop below is an immediate or 32-bit offset from one of these, and
  // each advances by exactly one group per iteration. The first draft wrote the same indices as
  // `(long)` expressions inside the loops and compiled to ~200 sixty-four-bit address instructions
  // per group (`v_mad_co_u64_u32` plus add-carry pairs), which turned a kernel that should be
  // bandwidth-bound into a VALU-bound one: 490 us against Triton's 398 on identical bytes. A dead
  // column's VALUE is never masked here, only its ADDRESS clamped and its stores suppressed --
  // prod masks the loads to 0.0 instead, and nothing observes the difference because nothing reads
  // a dead column's accumulator.
  const int cbase = live ? mycol : 0;
  const unsigned char *wp0 = wq + (long)cbase * stride_wq + (lane >> 4) * 8;
  const unsigned char *wp1 = wp0 + 16L * stride_wq;   // the wave's second n-fragment
  // One A base per m-fragment, and at MF = 1 this is the single expression it replaces: fragment
  // `mf` covers rows `m0 + mf * 16 ..+16`, which is the row slice the 16-row symbol is launched on.
  const __bf16 *xp[MF];
  const float *xsp[MF];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf) {
    xp[mf] = x + (long)(m0 + mf * DH_M + (lane & 15)) * k + (lane >> 4) * 8;
    xsp[mf] = xs + (long)(m0 + mf * DH_M + (lane >> 4) * 8) * stride_xs;
  }
  const __bf16 *sp = s + (long)cbase * stride_s;
  const __bf16 *zp = zs + (long)cbase * stride_s;
  for (int g = 0; g < ngrp; ++g) {
    // One group's packed bytes for both fragments: 8 k-steps x 8 bytes, issued together so a
    // tile's worth of the weight stream is in flight at once rather than one load at a time.
    uint2_t raw[DH_NF][NG];
#pragma unroll
    for (int t = 0; t < NG; ++t) {
      raw[0][t] = *(const uint2_t *)(wp0 + t * 16);
      raw[1][t] = *(const uint2_t *)(wp1 + t * 16);
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      floatx8 dot[MF][DH_NF];
#pragma unroll
      for (int mf = 0; mf < MF; ++mf)
#pragma unroll
        for (int f = 0; f < DH_NF; ++f)
#pragma unroll
          for (int e = 0; e < 8; ++e) dot[mf][f][e] = 0.f;
      // The k range this quarter reads: q * Q + g * G, the same 128 elements prod's `offs_k` names.
#pragma unroll
      for (int t = 0; t < NG; ++t) {
        bf16x8 xv[MF];
#pragma unroll
        for (int mf = 0; mf < MF; ++mf) xv[mf] = *(const bf16x8 *)(xp[mf] + q * q4 + t * 16);
#pragma unroll
        for (int f = 0; f < DH_NF; ++f) {
          // ONE decode, MF WMMAs: the 8 bytes this lane holds are the same B fragment for every
          // m-fragment, which is the reuse a real BLOCK_M = 32 band buys and a row-slice grid
          // cannot (its two m tiles are two workgroups and read the weight twice).
          const uint4_t w4 = (q == 0)   ? dh_decode<0>(raw[f][t])
                             : (q == 1) ? dh_decode<1>(raw[f][t])
                             : (q == 2) ? dh_decode<2>(raw[f][t])
                                        : dh_decode<3>(raw[f][t]);
          const bf16x8 wv = __builtin_bit_cast(bf16x8, w4);
#pragma unroll
          for (int mf = 0; mf < MF; ++mf) dot[mf][f] = dh_wmma(xv[mf], wv, dot[mf][f]);
        }
      }
      const int gq = q * ngrp;                // the scale group is (q * NG + g); g is in the bases
      float xsv[MF][8];
#pragma unroll
      for (int mf = 0; mf < MF; ++mf)
#pragma unroll
        for (int e = 0; e < 8; ++e) xsv[mf][e] = xsp[mf][e * stride_xs + gq];
#pragma unroll
      for (int f = 0; f < DH_NF; ++f) {
        const int so = f * 16 * stride_s + gq;
        const float sv = (float)sp[so];
        const float zv = (float)zp[so];
#pragma unroll
        for (int mf = 0; mf < MF; ++mf)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            acc[mf][f][e] += dot[mf][f][e] * sv;
            acc[mf][f][e] -= xsv[mf][e] * zv;
          }
      }
    }
    wp0 += DH_G;
    wp1 += DH_G;
    sp += 1;
    zp += 1;
#pragma unroll
    for (int mf = 0; mf < MF; ++mf) {
      xp[mf] += DH_G;
      xsp[mf] += 1;
    }
  }

  // ---- the coarse logits, and the tile the emit scans ----------------------------------------
#pragma unroll
  for (int mf = 0; mf < MF; ++mf)
#pragma unroll
    for (int f = 0; f < DH_NF; ++f) {
      const int c = mycol + f * 16;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int row = mf * DH_M + dh_acc_row(lane, e);
        // Prod stores all BLOCK_M rows and masks only on N -- `m` is padded up to the band by
        // `_pow2_at_least` and the buffers are allocated at that padded height -- so a `row < m`
        // guard here would be one exec-mask branch per row for a condition the reference does not
        // have.
        if (live)
          y[(long)(m0 + row) * n + c] = TOPK_ONLY ? (__bf16)(-INFINITY) : (__bf16)acc[mf][f][e];
        // Prod masks the emit's input, not the accumulator: a column past N enters the scan as
        // -inf.
        tile[DH_TILE_MR(MROWS, row, wave * 32 + (lane & 15) + f * 16)] =
            live ? acc[mf][f][e] : -INFINITY;
      }
    }
  __syncthreads();

  // One wave scans: lane l takes row `l & 15` and the 32 columns of half `l >> 4`, then the two
  // halves combine with a single cross-lane swap. Masking writes -inf back into the winner's own
  // LDS slot, which is read only by the lane that wrote it.
  if (MF > 1 || wave == 0) {
    // 16 rows x 2 halves is one wave, so a band of MROWS rows is MF scanning waves and wave `w`
    // takes rows `w * 16 ..+16`. At MF = 1 this is `wave == 0` and `row = lane & 15`, the scan the
    // traced band has always run.
    const int row = (MF > 1 ? wave * DH_M : 0) + (lane & 15), half = lane >> 4;
    for (int c = 0; c < DH_KC; ++c) {
      float bv = -INFINITY;
      int bi = -1;
#pragma unroll 8
      for (int j = 0; j < 32; ++j)
        dh_better(bv, bi, tile[DH_TILE_MR(MROWS, row, half * 32 + j)], half * 32 + j);
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
      if (wi >> 5 == half) tile[DH_TILE_MR(MROWS, row, wi)] = -INFINITY;
    }
  }
}

#define DH_HEAD_ENTRY(name, TOPK, MT, MROWS)                                                      \
  extern "C" __global__ __launch_bounds__(DH_NT) void name(                                       \
      const __bf16 *__restrict__ x, const float *__restrict__ xs,                                 \
      const unsigned char *__restrict__ wq, const __bf16 *__restrict__ s,                         \
      const __bf16 *__restrict__ zs, __bf16 *__restrict__ y, float *__restrict__ bmax,            \
      int *__restrict__ bidx, int m, int n, int k, int stride_wq, int stride_s, int stride_xs,     \
      int nblk) {                                                                                 \
    dflash_draft_head_int2_impl<TOPK, MT, MROWS>(x, xs, wq, s, zs, y, bmax, bidx, m, n, k,        \
                                     stride_wq, stride_s, stride_xs, nblk);                       \
  }

// The MTP caller reads the coarse logits (an argmax over the whole row), so its variant stores them.
DH_HEAD_ENTRY(dflash_draft_head_int2_m16n64g128, false, false, DH_M)
// DFlash2's candidate caller does not: `_ref_topk_only` blanks the row first.
DH_HEAD_ENTRY(dflash_draft_head_int2_m16n64g128_topk, true, false, DH_M)

// ---- THE M-TILED INSTANTIATIONS, for a BATCHED proposal ---------------------------------------
//
// The head scores the sampled rows of the whole batch: seven per request at prod's block of eight,
// padded by `_pow2_at_least` to 32 at four requests and 256 at `max_seqs`. Prod carries M = 32 and
// M = 64 as DIFFERENT instantiations -- `_cfg_for` picks num_warps 2 / 4 / 8 by band and BLOCK_M is
// the padded row count -- so the committed rules refused anything but 16 by name (LEDGER, S6's
// census of the batched proposal).
//
// The band is not an arithmetic fact, it is a launch-config fact. Per output element the reduction
// is the same one whatever BLOCK_M is: `acc[m][n]` accumulates `dot * S - XS * ZS` over the same
// ten groups of each of the same four quarters, in the same order, and the `tl.dot` inside a group
// is the same eight ascending WMMA k-steps. What BLOCK_M changes in the reference is how many rows
// one program covers and how many warps split them -- so a GRID with `blockIdx.y` picking the
// sixteen rows a workgroup owns computes prod's own numbers at any padded M that is a whole number
// of tiles, and the gate tests exactly that: bit-identical to the m16 symbol on each 16-row slice,
// and against prod's Triton head at BLOCK_M = 32 and 64.
DH_HEAD_ENTRY(dflash_draft_head_int2_mtn64g128, false, true, DH_M)
DH_HEAD_ENTRY(dflash_draft_head_int2_mtn64g128_topk, true, true, DH_M)

// ---- THE 32-ROW BAND, which is what a batched proposal should actually launch -----------------
//
// The row slice above makes the shape LEGAL and costs 17% MORE than two launches of the 16-row
// symbol (868.54 us against 743.20, F33's per-launch table), and the reason is in the grid: at a y
// extent of 2 the two m tiles of one 64-column block are two different workgroups, so the head's
// 159 MiB int2 weight is read once per tile exactly as two separate launches read it, and the tile
// buys no reuse at all while costing the extra workgroups. Prod's own BLOCK_M = 32 band does the
// thing that pays -- ONE workgroup holds a [32, 64] accumulator and reads each weight byte once for
// thirty-two rows -- which is why prod's Triton at BLOCK_M 32 is 592.64 us, 0.80x of two m16
// launches where the row slice is 1.17x.
//
// This is that band, and it is the same program: MROWS = 32 gives each lane a SECOND accumulator
// fragment (rows 16..31 of the tile) driven by a second A fragment against the SAME decoded B
// fragment, the emit writes 32 rows of a [32, 64] LDS tile at a 33-word column stride, and the scan
// is the same 16-rows-x-2-halves wave twice, one per 16-row group. Every output element's
// arithmetic is the 16-row band's, element for element, which is what the gate checks per 16-row
// slice and against prod's Triton at BLOCK_M = 32.
//
// The y extent is `M / 32`, so this serves 32, 64, 128 and 256 -- every M `_pow2_at_least` produces
// above the traced one. An M that is a whole number of 16-row tiles but NOT of 32 (48 rows, say) is
// the row slice's, and the rules name it as exactly that fallback.
DH_HEAD_ENTRY(dflash_draft_head_int2_m32n64g128, false, true, DH_MB)
DH_HEAD_ENTRY(dflash_draft_head_int2_m32n64g128_topk, true, true, DH_MB)

// ---- H43: THE ACTIVATIONS STAGED ONCE IN LDS, two 64-column blocks a workgroup ----------------
//
// ws-h42's reading of the shipped band: per 512-k group a lane issues 16 `b64` weight loads but 32
// `b128` loads of `x`, because every one of a workgroup's two waves reads the WHOLE [16, 128] A
// slice of every quarter for itself -- 1,940 workgroups x 2 waves x 160 KB = ~620 MB of L2 -> CU
// traffic per launch against 165 MB of weight. This variant changes the TILE and nothing else:
//
//   * a workgroup is FOUR waves covering TWO adjacent 64-column blocks (`pid = 2 * blockIdx.x +
//     (wave >> 1)`; each pair of waves is the shipped band's workgroup, same columns, same emit);
//   * each quarter's [MROWS, 128] A slice is loaded ONCE by all 128 threads (16-byte chunks) into a
//     272-byte-stride LDS stage,
//     and every wave reads its A fragments from there -- so `x` crosses L2 once per 128 columns
//     instead of four times;
//   * the emit's tiles alias the stage after the k loop.
//
// THE ARITHMETIC IS THE SHIPPED BAND'S, element for element: the A fragment a lane feeds each WMMA
// holds the same eight bf16 of the same row and k (only their route changed, global -> LDS -> VGPR
// as raw bits), the eight k-steps of a group run in the same ascending order into a fresh tile, and
// `acc += dot * S; acc -= XS * ZS` are the same two expressions in the same order. So every y, bmax
// and bidx is the shipped symbol's bytes BY CONSTRUCTION -- and gated anyway (`h43_head_ab.py`).
// A workgroup whose second block is past `nblk` (odd block counts) stages and syncs like the others
// and stores nothing for it.
#define DH_LXB    2                         // 64-column blocks per workgroup
#define DH_LXNT   (DH_LXB * DH_NT)          // 128 threads = 4 waves
#define DH_LXS    (DH_G + 8)                // the stage's row stride in bf16: 272 B, 16 rows over 64 banks

template <bool TOPK_ONLY, bool MT, int MROWS>
__device__ __forceinline__ void dflash_draft_head_int2_lx_impl(
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
  constexpr int TILE_WORDS = DH_TILE_MR_WORDS(MROWS);
  constexpr int TILES_BYTES = DH_LXB * TILE_WORDS * 4;
  constexpr int LDS_BYTES = STAGE_BYTES > TILES_BYTES ? STAGE_BYTES : TILES_BYTES;
  __shared__ __attribute__((aligned(16))) unsigned char lds[LDS_BYTES];
  __bf16 *const stage = (__bf16 *)lds;

  const int m0 = MT ? (int)blockIdx.y * MROWS : 0;
  if (MT && m0 >= m) return;
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
  const float *xsp[MF];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf)
    xsp[mf] = xs + (long)(m0 + mf * DH_M + (lane >> 4) * 8) * stride_xs;
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
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      // This quarter's chunks: loaded, then (once every wave is done reading the previous
      // quarter) stored. No prefetch across the k-steps: holding the next quarter's chunks
      // through the WMMAs spilled (152-224 B of scratch); the other resident workgroups cover
      // the load's latency instead.
      uint4_t xr[CH];
#pragma unroll
      for (int i = 0; i < CH; ++i) xr[i] = *(const uint4_t *)(xg + i * xstep + q * q4 + g * DH_G);
      __syncthreads();
#pragma unroll
      for (int i = 0; i < CH; ++i) *(uint4_t *)(so + i * RSTEP * DH_LXS) = xr[i];
      __syncthreads();
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
      const int gq = q * ngrp;
      float xsv[MF][8];
#pragma unroll
      for (int mf = 0; mf < MF; ++mf)
#pragma unroll
        for (int e = 0; e < 8; ++e) xsv[mf][e] = xsp[mf][e * stride_xs + gq];
#pragma unroll
      for (int f = 0; f < DH_NF; ++f) {
        const int so = f * 16 * stride_s + gq;
        const float sv = (float)sp[so];
        const float zv = (float)zp[so];
#pragma unroll
        for (int mf = 0; mf < MF; ++mf)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            acc[mf][f][e] += dot[mf][f][e] * sv;
            acc[mf][f][e] -= xsv[mf][e] * zv;
          }
      }
    }
    wp0 += DH_G;
    wp1 += DH_G;
    sp += 1;
    zp += 1;
#pragma unroll
    for (int mf = 0; mf < MF; ++mf) xsp[mf] += 1;
  }

  __syncthreads();                          // the tiles alias the stage
#pragma unroll
  for (int mf = 0; mf < MF; ++mf)
#pragma unroll
    for (int f = 0; f < DH_NF; ++f) {
      const int c = mycol + f * 16;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int row = mf * DH_M + dh_acc_row(lane, e);
        if (live)
          y[(long)(m0 + row) * n + c] = TOPK_ONLY ? (__bf16)(-INFINITY) : (__bf16)acc[mf][f][e];
        tile[DH_TILE_MR(MROWS, row, lw * 32 + (lane & 15) + f * 16)] =
            live ? acc[mf][f][e] : -INFINITY;
      }
    }
  __syncthreads();

  if (blive && (MF > 1 || lw == 0)) {
    const int row = (MF > 1 ? lw * DH_M : 0) + (lane & 15), half = lane >> 4;
    for (int c = 0; c < DH_KC; ++c) {
      float bv = -INFINITY;
      int bi = -1;
#pragma unroll 8
      for (int j = 0; j < 32; ++j)
        dh_better(bv, bi, tile[DH_TILE_MR(MROWS, row, half * 32 + j)], half * 32 + j);
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
      if (wi >> 5 == half) tile[DH_TILE_MR(MROWS, row, wi)] = -INFINITY;
    }
  }
}

#define DH_HEAD_LX_ENTRY(name, TOPK, MT, MROWS)                                                   \
  extern "C" __global__ __launch_bounds__(DH_LXNT) void name(                                     \
      const __bf16 *__restrict__ x, const float *__restrict__ xs,                                 \
      const unsigned char *__restrict__ wq, const __bf16 *__restrict__ s,                         \
      const __bf16 *__restrict__ zs, __bf16 *__restrict__ y, float *__restrict__ bmax,            \
      int *__restrict__ bidx, int m, int n, int k, int stride_wq, int stride_s, int stride_xs,     \
      int nblk) {                                                                                 \
    dflash_draft_head_int2_lx_impl<TOPK, MT, MROWS>(x, xs, wq, s, zs, y, bmax, bidx, m, n, k,     \
                                                    stride_wq, stride_s, stride_xs, nblk);        \
  }

// Grid x = ceil(nblk / 2); y = 1 at the 16-row band, m / 32 at the 32-row one.
DH_HEAD_LX_ENTRY(dflash_draft_head_int2_m16n64x2g128_lx, false, false, DH_M)
DH_HEAD_LX_ENTRY(dflash_draft_head_int2_m16n64x2g128_lx_topk, true, false, DH_M)
DH_HEAD_LX_ENTRY(dflash_draft_head_int2_m32n64x2g128_lx, false, true, DH_MB)
DH_HEAD_LX_ENTRY(dflash_draft_head_int2_m32n64x2g128_lx_topk, true, true, DH_MB)

// -----------------------------------------------------------------------------------------------
// The exact rerank. One program per (draft row, candidate): an exact fp32 logit for candidate
// column `idx[m][j]`, read straight off the head's own e4m3 weight -- a few hundred KB against the
// coarse pass's 0.167 GiB, which is what makes the head exact at 2 bits.
//
// THE REDUCTION SHAPE IS THE GATE, and it is Triton's rather than a choice. `acc` is a [BLOCK_K]
// vector of 512 fp32 partials, so element i accumulates k = i, 512+i, 1024+i, ... in that order and
// `tl.sum` then reduces the 512. With num_warps=4 the blocked layout gives thread t the four
// CONTIGUOUS elements 4t..4t+3 (`v_lshl_or_b32 v5, v0, 3` then `buffer_load_b64` in the reference's
// own disassembly), and the tree that follows was READ OUT OF THAT DISASSEMBLY, because guessing it
// cost 923 of 1280 values at 1-4 ULP:
//
//   1. four sequential adds inside the lane:            ((a0 + a1) + a2) + a3
//   2. `v_add_f32_dpp row_shr:8, 4, 2, 1` -- DESCENDING strides, so the 16 lanes of a row pair as
//      (x_i, x_{i-8}) FIRST and adjacent lanes LAST. A butterfly over 1, 2, 4, 8 is the same
//      number of adds in the same balanced shape and a DIFFERENT last bit.
//   3. `v_permlanex16` and one add, so the wave's total lands in LANE 31, not lane 0.
//   4. the four waves' partials through LDS, combined `(p0 + p2) + (p1 + p3)` -- quad_perm xor 2
//      then xor 1, descending again.
//
// The fp8 decode is explicit bit arithmetic, as in prod, and not a cast: e4m3 with bias 7, the
// subnormal arm multiplying by 2^-9 exactly. `v_exp_f32` is what both sides use for the exponent.
#define DH_RK 512

__device__ __forceinline__ float dh_e4m3(unsigned int b, float sc) {
  const int e = (int)((b >> 3) & 15u);
  const float mant = (float)(b & 7u);
  const float mag = (e == 0) ? mant * 0.001953125f
                             : (1.0f + mant * 0.125f) * exp2f((float)(e - 7));
  return ((b & 128u) != 0u ? -mag : mag) * sc;
}

// One `v_add_f32_dpp vN, vN, vN row_shr:S bound_ctrl:1`: the DPP operand reads 0 outside the
// 16-lane row, so the shifted-in lanes contribute nothing. `old = 0` makes that explicit rather
// than relying on which of the two bound-control conventions the builtin takes.
template <int S>
__device__ __forceinline__ float dh_row_shr_add(float v) {
  const int sh = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), 0x110 | S, 0xf, 0xf,
                                             true);
  return v + __builtin_bit_cast(float, sh);
}

// The rerank, in two variants times two weight dtypes. The DENSE one writes `out[m][j]`, which is
// what prod's `_rerank_exact` does and what the gate compares against. The SCATTER one writes
// `y[m][idx[m][j]]` instead -- absorbing prod's `y.scatter_(1, idx.long(), ex.to(bf16))`, whose
// positions are exactly the ones this kernel already visits, one per program. The rounding is the
// scatter's: an fp32 sum, then one bf16 round.
//
// THE WEIGHT DTYPE IS THE REFERENCE'S `FP8` CONSTEXPR, and which arm runs is a property of the
// SERVE and not of this kernel. `_rerank_exact` takes `FP8: tl.constexpr` and `_apply_head_int2`
// picks the arm from `_head_matrix`: an e4m3 `lm_head` with a per-row scale takes `FP8=True`, and
// a plain bf16 one takes `FP8=False` and passes `x` for the unused scale argument. Under the unit
// prod serves, `SABER_NVFP4_LMHEAD=bf16` dequantises the head at load
// (`ref_nvfp4.RefFp8ChannelToBf16`, `bf16(f32(w8) * f32(scale))` one row at a time), so
// `_head_matrix` returns `(bf16 rows, None)` and the FP8=False arm is the one that runs. The
// bf16 entry points below are that arm; they take no `ws` at all, because the reference does not
// read one and a bound-but-unread scale would be a record the pack does not need.
//
// A THIRD weight form (workstream N6), and the one the default serve runs since 2026-09-22: the
// checkpoint's own e4m3 head and its per-row scale (`head.rerank.w8` + `head.rerank.ws`, the SAME
// record bytes as the target's `lm_head.w8` + `lm_head.ws`, so the card holds one head record), expanded
// IN REGISTER to the bf16 arm's weight: `bf16(f32(w8) * f32(scale))`, one exact multiply (4 x 8
// significant bits fit f32's 24) and the round-to-nearest-even (N4's `db_w8_expand_rne`). The value
// the dot then reads is the bf16 record's element bit for bit, in the same k order into the same
// four accumulators and the same reduction tree, so the output is the bf16 arm's, not close to it.
// It is NOT the FP8=True arm above: that one multiplies the scale in f32 and never rounds to bf16,
// which is prod's matrix minus one rounding (6 / 8 / 9 of 448 bytes off on the corpus) -- and is
// therefore the unit gate's negative control. The integer RNE is the cast for every finite product;
// the checkpoint has no e4m3 NaN code (N4's census over the whole head).
enum DhWeight { kDhFp8Scaled = 0, kDhBf16 = 1, kDhW8Bf16 = 2 };

__device__ __forceinline__ float dh_bf16_rne(float v) {
  unsigned int a = __float_as_uint(v);
  a += 0x7fffu + ((a >> 16) & 1u);
  return __uint_as_float(a & 0xffff0000u);
}

template <bool SCATTER, int WK>
__device__ __forceinline__ void dflash_rerank_exact_impl(
    const __bf16 *__restrict__ x,           // [M, K] bf16, the same hidden states
    const void *__restrict__ wv_,           // [rows, K] e4m3 bytes (FP8) or bf16 (else)
    const float *__restrict__ ws,           // [rows] fp32 per-row scale; null when !FP8
    const int *__restrict__ idx,            // [M, R] candidate columns
    void *__restrict__ out,                 // [M, R] fp32, or [M, N] bf16 when SCATTER
    int k, int stride_w, int r, int n) {
  const int mrow = blockIdx.x, j = blockIdx.y;
  const int lane = threadIdx.x & 31, wv = threadIdx.x >> 5;
  const int nrow = idx[mrow * r + j];
  const float sc = WK != kDhBf16 ? ws[nrow] : 1.0f;
  const int base = (int)threadIdx.x * 4;    // Triton's blocked layout at sizePerThread = 4
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  // The four elements a lane owns are contiguous, so they are ONE 8-byte load of x and ONE 4-byte
  // (e4m3) or 8-byte (bf16) load of the weight -- which is what the reference does
  // (`buffer_load_b64` + `buffer_load_b32`). Written as four scalar subscripts the compiler
  // emitted eight `global_load_d16`, and eight narrow loads per k step cost the kernel 6% against
  // Triton on identical bytes.
  const __bf16 *xr = x + (long)mrow * k + base;
  const unsigned char *wr = (const unsigned char *)wv_ + (long)nrow * stride_w + base;
  const __bf16 *wb16 = (const __bf16 *)wv_ + (long)nrow * stride_w + base;
  for (int k0 = 0; k0 < k; k0 += DH_RK) {
    const bf16x4 xv = *(const bf16x4 *)(xr + k0);
    if (WK == kDhFp8Scaled) {
      const unsigned int wb = *(const unsigned int *)(wr + k0);
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[i] += (float)xv[i] * dh_e4m3((wb >> (8 * i)) & 0xFFu, sc);
    } else if (WK == kDhW8Bf16) {
      typedef float floatx2 __attribute__((ext_vector_type(2)));
      const unsigned int wb = *(const unsigned int *)(wr + k0);
      const floatx2 lo = __builtin_amdgcn_cvt_pk_f32_fp8((int)wb, false);
      const floatx2 hi = __builtin_amdgcn_cvt_pk_f32_fp8((int)wb, true);
      const float w4[4] = {dh_bf16_rne(lo[0] * sc), dh_bf16_rne(lo[1] * sc),
                           dh_bf16_rne(hi[0] * sc), dh_bf16_rne(hi[1] * sc)};
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[i] += (float)xv[i] * w4[i];
    } else {
      const bf16x4 wv4 = *(const bf16x4 *)(wb16 + k0);
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[i] += (float)xv[i] * (float)wv4[i];
    }
  }
  float v = ((acc[0] + acc[1]) + acc[2]) + acc[3];
  v = dh_row_shr_add<8>(v);
  v = dh_row_shr_add<4>(v);
  v = dh_row_shr_add<2>(v);
  v = dh_row_shr_add<1>(v);
  v += __shfl_xor(v, 16, 32);               // permlanex16: the two 16-lane halves
  __shared__ float part[4];
  if (lane == 31) part[wv] = v;             // ... and the wave's total is in lane 31
  __syncthreads();
  if (threadIdx.x == 0) {
    const float tot = (part[0] + part[2]) + (part[1] + part[3]);
    if (SCATTER) {
      ((__bf16 *)out)[(long)mrow * n + nrow] = (__bf16)tot;
    } else {
      ((float *)out)[mrow * r + j] = tot;
    }
  }
}

#define DH_RERANK_ENTRY(name, SCAT)                                                               \
  extern "C" __global__ __launch_bounds__(128) void name(                                         \
      const __bf16 *__restrict__ x, const unsigned char *__restrict__ w,                          \
      const float *__restrict__ ws, const int *__restrict__ idx, void *__restrict__ out, int k,    \
      int stride_w, int r, int n) {                                                               \
    dflash_rerank_exact_impl<SCAT, kDhFp8Scaled>(x, w, ws, idx, out, k, stride_w, r, n);           \
  }

DH_RERANK_ENTRY(dflash_rerank_exact_fp8_k512, false)
DH_RERANK_ENTRY(dflash_rerank_exact_fp8_k512_scatter, true)

// ... and the FP8=False arm, which is the one the DFlash2 serve runs. No `ws` in the signature:
// the reference passes `x` for that argument and never loads it.
#define DH_RERANK_BF16_ENTRY(name, SCAT)                                                          \
  extern "C" __global__ __launch_bounds__(128) void name(                                         \
      const __bf16 *__restrict__ x, const __bf16 *__restrict__ w,                                 \
      const int *__restrict__ idx, void *__restrict__ out, int k, int stride_w, int r, int n) {   \
    dflash_rerank_exact_impl<SCAT, kDhBf16>(x, w, nullptr, idx, out, k, stride_w, r, n);           \
  }

DH_RERANK_BF16_ENTRY(dflash_rerank_exact_bf16_k512, false)
DH_RERANK_BF16_ENTRY(dflash_rerank_exact_bf16_k512_scatter, true)

// ... and the bf16 arm over the fp8 RECORD (N6): the same kernargs as the FP8=True entry, named
// `w8` / `ws` after the records they bind (`head.rerank.w8`, `head.rerank.ws`).
#define DH_RERANK_W8_ENTRY(name, SCAT)                                                            \
  extern "C" __global__ __launch_bounds__(128) void name(                                         \
      const __bf16 *__restrict__ x, const unsigned char *__restrict__ w8,                         \
      const float *__restrict__ ws, const int *__restrict__ idx, void *__restrict__ out, int k,    \
      int stride_w, int r, int n) {                                                               \
    dflash_rerank_exact_impl<SCAT, kDhW8Bf16>(x, w8, ws, idx, out, k, stride_w, r, n);             \
  }

DH_RERANK_W8_ENTRY(dflash_rerank_exact_w8_k512, false)
DH_RERANK_W8_ENTRY(dflash_rerank_exact_w8_k512_scatter, true)

// -----------------------------------------------------------------------------------------------
// The int2 head's OWN INPUT PREP: the two launches `_apply_head_int2` makes before the head and
// that the plan treated as inputs until F27 wrote the chain.
//
//   x = torch.cat([x, x.new_zeros(M - m, k)])          # _pow2_at_least's row pad
//   xs = x.reshape(M, ng, GROUP).float().sum(-1)       # the per-group fp32 column sum
//
// Neither is a port and neither has a Triton source: both are ATen, so the reference is ATen's
// own arithmetic and the gate is prod's captured `xs` (`head.x_pad` / `head.xs`, r26+).
//
// THE SUM'S ORDER IS THE GATE, and it was READ OUT OF `ATen/native/hip/Reduce.cuh` in the serving
// image rather than guessed. For `[640, 128] f32` reduced over the fastest dimension:
//
//   * `setReduceConfig` takes the "vectorize along input" branch (`dim0 >= 128`, one reduce dim),
//     so `input_vec_size = vt0 = 4` and `dim0` becomes 32;
//   * `set_block_dimension(32, 640)` gives `block_width = 32` (the wave) and `block_height = 16`,
//     and `input_mult[0] = split_input(32)` makes `step_input` 32;
//   * `values_per_thread()` is 4, below the 256 `warp_split_threshold`, so each WAVE owns one
//     output and there is no cross-wave stage;
//   * `input_vectorized_thread_reduce_impl` therefore gives lane `t` exactly one `float4` --
//     elements `4t .. 4t+3` -- into four accumulators combined `((v0 + v1) + v2) + v3`;
//   * and `block_x_reduce`'s intra-wave fold is ASCENDING on ROCm (`#if defined(USE_ROCM)`:
//     `for (offset = 1; offset < dim_x; offset <<= 1)`), which is a balanced binary tree over the
//     32 lane partials in lane order -- NOT the descending `row_shr` fold the rerank above needs.
//
// So: four contiguous elements per lane, summed left to right, then a balanced 32-lane tree. Any
// other association is a different last bit, which is what the LEDGER's rerank entry cost.
#define DHP_G     128        // the quantisation group; also 32 lanes x 4 elements
#define DHP_VEC     4        // input_vec_size = vt0
#define DHP_WAVES   8        // waves per workgroup: one output group each
#define DHP_NT    (DHP_WAVES * 32)


// ORDER selects the lane decomposition. 0 is the route -- ATen's vectorized branch, one float4
// per lane. 1 and 2 are CONTROLS in the same code object, each one arithmetic choice away:
//   1 (`_strided`) is the NON-vectorized branch of the same file (`thread_reduce_impl`), where
//     lane t takes elements t, t+32, t+64, t+96. It is the other reading of `setReduceConfig`
//     and it is the one a reader who missed `dim0 >= 128` would write;
//   2 (`_seq`) drops the lane decomposition entirely and sums the 128 left to right in one
//     accumulator, which is what "a group sum" means before anyone looks at the reference.
template <int ORDER>
__device__ __forceinline__ void dflash_head_group_sum_impl(
    const __bf16 *__restrict__ x, float *__restrict__ xs, int rows, int k, int stride_xs) {
  const int ng = k / DHP_G;
  const int lane = (int)(threadIdx.x & 31u);
  const int wave = (int)(threadIdx.x >> 5);
  const int out = (int)blockIdx.x * DHP_WAVES + wave;   // a (row, group) pair
  if (out >= rows * ng) return;
  const int row = out / ng;
  const int grp = out - row * ng;
  const __bf16 *base = x + (long)row * k + (long)grp * DHP_G;

  if (ORDER == 2) {
    if (lane == 0) {
      float acc = 0.0f;
      for (int i = 0; i < DHP_G; i++) acc = acc + (float)base[i];
      xs[(long)row * stride_xs + grp] = acc;
    }
    return;
  }

  // The four accumulators, each started from the reduction's identity exactly as ATen starts
  // them: `0 + v` and not `v`, because the two differ when v is -0.0.
  float v[DHP_VEC];
#pragma unroll
  for (int i = 0; i < DHP_VEC; i++) v[i] = 0.0f;
#pragma unroll
  for (int i = 0; i < DHP_VEC; i++) {
    const int e = (ORDER == 0) ? (lane * DHP_VEC + i) : (lane + i * 32);
    v[i] = v[i] + (float)base[e];
  }
#pragma unroll
  for (int i = 1; i < DHP_VEC; i++) v[0] = v[0] + v[i];

  // `block_x_reduce`'s ROCm fold: ascending offsets, a balanced tree in lane order.
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) v[0] = v[0] + __shfl_down(v[0], off, 32);
  if (lane == 0) xs[(long)row * stride_xs + grp] = v[0];
}

#define DHP_SUM_ENTRY(name, ORDER)                                                                \
  extern "C" __global__ __launch_bounds__(DHP_NT) void name(                                      \
      const __bf16 *__restrict__ x, float *__restrict__ xs, int rows, int k, int stride_xs) {     \
    dflash_head_group_sum_impl<ORDER>(x, xs, rows, k, stride_xs);                                 \
  }

DHP_SUM_ENTRY(dflash_head_group_sum_g128, 0)
DHP_SUM_ENTRY(dflash_head_group_sum_g128_strided, 1)
DHP_SUM_ENTRY(dflash_head_group_sum_g128_seq, 2)

// ... and the row pad. `torch.cat([x, x.new_zeros(M - m, k)])` is two ATen launches -- the fill
// and the concatenation -- and this is one kernel that writes both halves, the same absorption
// `dflash_draft_head_int2_m16n64g128_topk` already makes of prod's `y.fill_(-inf)`. A copy has no
// arithmetic, so the only way it can be wrong is WHICH ROWS it writes -- which is what the
// `_fillfirst` control puts the other way round.
template <bool FILL_FIRST>
__device__ __forceinline__ void dflash_head_row_pad_impl(
    const __bf16 *__restrict__ x, __bf16 *__restrict__ out, int m, int rows, int k) {
  const long n = (long)rows * k;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < n; i += (long)gridDim.x * 256) {
    const long r = i / k;
    const bool live = FILL_FIRST ? (r >= (long)(rows - m)) : (r < (long)m);
    const long src = FILL_FIRST ? (i - (long)(rows - m) * k) : i;
    out[i] = live ? x[src] : (__bf16)0.0f;
  }
}

#define DHP_PAD_ENTRY(name, FF)                                                                   \
  extern "C" __global__ __launch_bounds__(256) void name(                                         \
      const __bf16 *__restrict__ x, __bf16 *__restrict__ out, int m, int rows, int k) {           \
    dflash_head_row_pad_impl<FF>(x, out, m, rows, k);                                             \
  }

DHP_PAD_ENTRY(dflash_head_row_pad_bf16, false)
DHP_PAD_ENTRY(dflash_head_row_pad_bf16_fillfirst, true)

// -----------------------------------------------------------------------------------------------
// The THIRD launch of the head's selection, and F27 found it the same way F26 found the other two
// -- by writing the chain and asking what feeds the rerank.
//
//     idx = bi.gather(1, bm.topk(RERANK, dim=1).indices).contiguous()
//
// `dflash_topk_f32_gather_k80` returns POSITIONS in the block-maxima row (that is what prod's own
// `topk.indices` are, and what F26's gate held it against). The exact rerank scores VOCABULARY
// COLUMNS, and `bi` is the map between them: `bidx[row][p]` is the column the head's coarse pass
// put at position p. The gather is an ATen launch prod makes and no plan row named it, so the
// declared total is 110 and not F26's 107 + 2.
//
// The control writes the POSITION instead of the column, which is the defect this op exists to
// prevent: an `idx` of positions reranks 80 arbitrary vocabulary entries and still produces a
// plausible-looking block.
template <bool AS_POSITION>
__device__ __forceinline__ void dflash_head_cand_gather_impl(
    const int *__restrict__ bidx, const int *__restrict__ pos, int *__restrict__ out, int rows,
    int n, int k) {
  const long total = (long)rows * k;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < total; i += (long)gridDim.x * 256) {
    const int row = (int)(i / k);
    const int p = pos[i];
    out[i] = AS_POSITION ? p : ((p >= 0 && p < n) ? bidx[(long)row * n + p] : 0);
  }
}

#define DHP_GATHER_ENTRY(name, AP)                                                                \
  extern "C" __global__ __launch_bounds__(256) void name(                                         \
      const int *__restrict__ bidx, const int *__restrict__ pos, int *__restrict__ out, int rows, \
      int n, int k) {                                                                             \
    dflash_head_cand_gather_impl<AP>(bidx, pos, out, rows, n, k);                                 \
  }

DHP_GATHER_ENTRY(dflash_head_cand_gather, false)
DHP_GATHER_ENTRY(dflash_head_cand_gather_pos, true)

// -----------------------------------------------------------------------------------------------
// The two ROW GATHERS `_generate_draft` makes, which are the fourth and fifth launches a chain
// finds inside this seam (`dflash2/speculator.py:196-206`):
//
//     hidden_states = last_hidden_states[self.sample_indices[:num_sample]]
//     anchor_token_ids = self.input_buffers.input_ids[self._anchor_indices[:num_reqs]]
//
// They are the drafter's 8 query rows narrowed to the 7 the head scores, and the one token per
// request the selector anchors on. ONE kernel at two shapes, the way `act_quant` is one kernel at
// two shapes: a row gather has no dtype, only a row WIDTH IN BYTES, so bf16 rows of 5120 and i32
// rows of 1 are the same program. A copy, so the only way it can be wrong is which rows it reads
// -- which is what the `_noidx` control puts back to the identity.
template <bool NO_IDX>
__device__ __forceinline__ void dflash_row_gather_impl(
    const unsigned char *__restrict__ in, const long *__restrict__ idx,
    unsigned char *__restrict__ out, int rows, int row_bytes, int src_rows) {
  const long n = (long)rows * row_bytes;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < n; i += (long)gridDim.x * 256) {
    const int r = (int)(i / row_bytes);
    const long off = i - (long)r * row_bytes;
    long src = NO_IDX ? r : idx[r];
    if (src < 0) src += src_rows;                   // ATen's negative-index rule
    out[i] = (src >= 0 && src < src_rows) ? in[src * row_bytes + off] : (unsigned char)0;
  }
}

#define DHP_ROWGATHER_ENTRY(name, NI)                                                             \
  extern "C" __global__ __launch_bounds__(256) void name(                                         \
      const unsigned char *__restrict__ in, const long *__restrict__ idx,                         \
      unsigned char *__restrict__ out, int rows, int row_bytes, int src_rows) {                   \
    dflash_row_gather_impl<NI>(in, idx, out, rows, row_bytes, src_rows);                           \
  }

DHP_ROWGATHER_ENTRY(dflash_row_gather, false)
DHP_ROWGATHER_ENTRY(dflash_row_gather_noidx, true)
