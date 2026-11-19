#!/usr/bin/env python3
"""Generate the W4A8-iu8 pack's FUSED decode bodies and band tiles (`hip/macaq_iu8f.hip`, code object
`macaq_iu8f`) from the plain iu8 sources, by anchored slicing + substitution (every anchor must occur exactly once).

    tools/gen_macaq_iu8f.py            write the source
    tools/gen_macaq_iu8f.py --check    exit 1 if the committed source differs from what this would write
    tools/gen_macaq_iu8f.py --lab F    write a tuning sweep object's source (every LAB_PFX tile) to F
    tools/gen_macaq_iu8f.py --lab27 F  the same for every LAB27_PFX tile (the p96 / p128 / p384 sweep)

(a) THE DECODE PAIR `macaq_iu8f_pair_w4_tm{1..4}` (fusion-map row 2 for iu8: two W4 GEMMs that read ONE A edge,
    gdn in_proj_qkvz + in_proj_ba): the two unpaired grids laid end to end (problem 0's `nb0 * ks0` blocks first,
    then problem 1's) under one dispatch. Each block runs `macaq_iu8.hip`'s `iu8_dec_impl` VERBATIM with its
    `blockIdx.x` / `blockIdx.z` / `gridDim.z` made arguments (the block's own problem, n-block, split, split
    count), so each half is byte for byte the dispatch it replaces (its own split count, partial slab and
    counters; int32 adds make the split not numerics anyway).
(b) ROW 3 `macaq_iu8f_arnq_hadi8_{w4,w8}_tm1` (a row-parallel decode GEMM + the exact two-rank all-reduce + the
    `_nq` epilogue, out / o / down): the same verbatim decode body (its last-arriving split block holds the
    exact C tile), then `mxfp4_gemm_decode_arnq_had.hip`'s row-3 protocol block VERBATIM (push the tile's piece
    of every row to the peer, tickets, the last M arrivals own one row each), then its `arnq_epilogue_impl`
    VERBATIM up to the transformed values (exact reduce, residual add, Gemma RMS norm, `macaq_had32`), with the
    e4m3 tail replaced by `saber_ar_hadi8.hip`'s per-token INT8 tail (sc = max(amax / 127, 1e-30) by `__fdiv_rn`,
    `macaq_i8_code`, the A-TILED store). The unfused path it replaces is `macaq_iu8_dec_*` + the
    `saber_ar_oneshot_2rank_exact_nq_ssqfma_hadi8_bf16` epilogue; the e4m3 forms of the same epilogue match the
    same `_nq` body bit for bit (same word ownership, same ssq / amax trees).

The small-chunk band + the ba-first pair (same object; `macaq_iu8.hsaco` untouched):
(c) THE BA-FIRST PAIR `macaq_iu8f_pairb_w4_tm{1..4}`: (a)'s body with the dispatch order swapped -- problem 1's
    (the ba half's) `nb1 * ks1` blocks FIRST, then problem 0's -- so the small split-K half is not queued
    behind the qkvz grid. Same per-block work (each block is still its unpaired dispatch's block).
(d) THE MID DECODE BAND `macaq_iu8f_dec_{w4,w8}_tm{5..8}` (M 65..128, prefill rungs p96 / p128): the decode
    body sliced for (a), instantiated at DTM 5..8 with the coordinates from the grid (blockIdx.x / .z /
    gridDim.z) -- the iu8 analogue of the MXFP4 small-chunk route (prefill rungs <= 128 rows on the split-K
    decode GEMM). The same arguments as `macaq_iu8_dec_*`.
(e) THE SMALL PREFILL TILES `macaq_iu8f_pf_{w4,w8}_bm{128,64}`: `macaq_iu8.hip`'s `iu8_pf_impl` VERBATIM with
    its wave tile height PTM a template argument (4 = the plain BM 256; 2 = BM 128; 1 = BM 64): more
    workgroups at the 129..512-row rungs. The same arguments as `macaq_iu8_pf_*`.
The 129..384-row band (same object):
(f) THE PARAMETRIC PREFILL TILES `macaq_iu8f_pf_{w4,w8}_t{PTM}w{PWM}n{TN}k{LBK}[p]`: `iu8_pf_impl` with the whole
    tile a template argument (PTM x PWM down M, TN across N, the K slab LBK) and PIPE (`p`: the next slab's W words
    in registers while this slab's WMMAs run, the W stage double-buffered in LDS, no trailing barrier). The loads,
    the W4 unpack, the WMMA chain and the epilogue are the plain body's own lines; the shipped object carries
    `SHIP_PFX` (the tuning sweep's winners), `--lab FILE` / `--lab27 FILE` write every `LAB_PFX` / `LAB27_PFX`
    candidate for a timing sweep (the sweep harness is not shipped).
Every form is integer-exact (int32 accumulators over all of K; the epilogue is the plain `iu8_epi`), so
any band choice gives the same bytes.

Contraction: the object is built with the default (like `macaq_iu8` and `mxfp4_gemm_decode_arnq_had`); the
GEMM epilogue (`iu8_epi`, `iu8_wsu`) and the row-3 epilogue carry `#pragma clang fp contract(off)` in their
bodies and `macaq_had.h` turns it off in its own; the int8 tail has no multiply-add.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIP = os.path.join(REPO, 'crates', 'saber-kernels', 'hip')
OUT = os.path.join(HIP, 'macaq_iu8f.hip')


def sub(s, a, b, what):
    n = s.count(a)
    if n != 1:
        raise SystemExit(f'{what}: anchor occurs {n} times: {a[:90]!r}')
    return s.replace(a, b)


def between(s, start, end, what):
    """the text from `start` (inclusive) to `end` (exclusive), both unique."""
    i, j = s.find(start), s.find(end)
    if s.count(start) != 1 or s.count(end) != 1 or j < i:
        raise SystemExit(f'{what}: slice anchors {start[:60]!r} .. {end[:60]!r}')
    return s[i:j]


def dec_body():
    """macaq_iu8.hip's helpers + `iu8_dec_impl`, with the block coordinates as arguments and a bool result
    (true = this block holds the exact C tile of its n-block: the last split arriver, or every block at ks 1)."""
    s = open(os.path.join(HIP, 'macaq_iu8.hip')).read()
    s = between(s, 'typedef int iu8_int2', '#define IU8_DEC(name, DTM, W8)', 'iu8 decode slice')
    s = sub(s, '__device__ __forceinline__ void iu8_dec_impl(const unsigned char *__restrict__ AT,',
            '__device__ __forceinline__ bool iu8f_dec_impl(const int bx, const int bz, const int nz,\n'
            '                                              const unsigned char *__restrict__ AT,', 'signature')
    s = sub(s, '  const int nt = blockIdx.x * 8 + wave;\n', '  const int nt = bx * 8 + wave;\n', 'nt')
    s = sub(s, '  const int ks = blockIdx.z, DKS = gridDim.z;\n', '  const int ks = bz, DKS = nz;\n', 'ks')
    s = sub(s, '    s_last = (atomicAdd(&cnt[blockIdx.x], 1) == DKS - 1);\n',
            '    s_last = (atomicAdd(&cnt[bx], 1) == DKS - 1);\n', 'counter')
    s = sub(s, '  if (tid == 0) cnt[blockIdx.x] = 0;', '  if (tid == 0) cnt[bx] = 0;', 'reset')
    s = sub(s, '  const int nb = blockIdx.x * 128, nw = min(128, N - nb);\n',
            '  const int nb = bx * 128, nw = min(128, N - nb);\n', 'nb')
    s = sub(s, '    if (!live) return;\n', '    if (!live) return true;\n', 'dead wave ks1')
    s = sub(s, '        if (m < M) C[(size_t)m * N + n] = (__bf16)iu8_epi(acc[i][e], as[m], wsu);\n      }\n    return;\n',
            '        if (m < M) C[(size_t)m * N + n] = (__bf16)iu8_epi(acc[i][e], as[m], wsu);\n      }\n    return true;\n',
            'ks1 return')
    s = sub(s, '  if (!s_last) return;\n', '  if (!s_last) return false;\n', 'not last')
    s = sub(s, '    C[(size_t)m * N + nn] = (__bf16)iu8_epi(sum, as[m], iu8_wsu<W8>(ws, nn));\n  }\n}\n',
            '    C[(size_t)m * N + nn] = (__bf16)iu8_epi(sum, as[m], iu8_wsu<W8>(ws, nn));\n  }\n  return true;\n}\n',
            'last return')
    code = '\n'.join(l.split('//')[0] for l in s.splitlines())
    assert 'blockIdx' not in code and 'gridDim' not in code, 'a block coordinate left in the decode slice'
    return s


I8_TAIL = '''  // saber_ar_hadi8.hip's per-token INT8 tail (codes in the A-TILED fragment order, row b)
  const float sc = fmaxf(__fdiv_rn(lds[0], 127.0f), 1e-30f);
  if (tid == 0) scale[b] = sc;
  const int mt = b >> 4, lrow = b & 15, kst = k16 >> 1;
#pragma unroll
  for (int mg = 0; mg < SABER_ARNQ_MAXG; ++mg) {
    const int k = start + tid + mg * SABER_ARNQ_T;
    if (k >= end) break;
    unsigned int lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) lo |= macaq_i8_code(sv[mg][j], sc) << (j * 8);
#pragma unroll
    for (int j = 4; j < 8; ++j) hi |= macaq_i8_code(sv[mg][j], sc) << ((j - 4) * 8);
    const long g = k - start;
    ((uint2*)q)[((long)mt * kst + (g >> 1)) * 32 + lrow + 16 * (g & 1)] = make_uint2(lo, hi);
  }
}

'''


def pf_body():
    """macaq_iu8.hip's `iu8_pf_impl` with PTM (m-fragments a wave) a template argument."""
    s = open(os.path.join(HIP, 'macaq_iu8.hip')).read()
    s = between(s, 'template <bool W8>\n__device__ __forceinline__ void iu8_pf_impl(', '#define IU8_PF(name, W8)',
                'iu8 prefill slice')
    s = sub(s, 'template <bool W8>\n__device__ __forceinline__ void iu8_pf_impl(',
            'template <bool W8, int PTM>\n__device__ __forceinline__ void iu8f_pf_impl(', 'pf signature')
    s = sub(s, '  constexpr int TN = 4, LBK = 64, PWM = 4, PWN = 2, PTM = 4, NS = LBK / 16, NT = PWM * PWN * 32;\n',
            '  constexpr int TN = 4, LBK = 64, PWM = 4, PWN = 2, NS = LBK / 16, NT = PWM * PWN * 32;\n', 'pf PTM')
    return s


PFX_LOOP = """  for (int k0 = 0; k0 < K; k0 += LBK) {
    const int kk0 = k0 >> 4;
"""


def pfx_body():
    """`iu8_pf_impl` with the whole tile a template argument (PTM m-fragments a wave, PWM waves down M,
    TN n-fragments a wave, the K slab LBK) and PIPE: the next slab's W words loaded into registers while this
    slab's WMMAs run, the W stage double-buffered in LDS (no trailing barrier). The af / W-load / W-store /
    WMMA / epilogue lines are the plain body's own, sliced; only the loop around them is new at PIPE."""
    s = open(os.path.join(HIP, 'macaq_iu8.hip')).read()
    s = between(s, 'template <bool W8>\n__device__ __forceinline__ void iu8_pf_impl(', '#define IU8_PF(name, W8)',
                'iu8 prefill slice (pfx)')
    s = sub(s, 'template <bool W8>\n__device__ __forceinline__ void iu8_pf_impl(',
            'template <bool W8, int PTM, int PWM, int TN, int LBK, bool PIPE>\n'
            '__device__ __forceinline__ void iu8f_pfx_impl(', 'pfx signature')
    s = sub(s, '  constexpr int TN = 4, LBK = 64, PWM = 4, PWN = 2, PTM = 4, NS = LBK / 16, NT = PWM * PWN * 32;\n',
            '  constexpr int PWN = 2, NS = LBK / 16, NT = PWM * PWN * 32;\n', 'pfx tile')
    s = sub(s, '  __shared__ unsigned char sW[BN / 16 * NS * 256];\n',
            '  constexpr int WBUF = BN / 16 * NS * 256;\n  __shared__ unsigned char sW[(PIPE ? 2 : 1) * WBUF];\n',
            'pfx lds')
    loop = between(s, '  for (int k0 = 0; k0 < K; k0 += LBK) {\n', '  float wsu[TN];\n', 'pfx loop')
    af = between(loop, '    iu8_int2 af[PTM][NS];\n', '    iu8_uint4 v[PIECES];\n', 'pfx af')
    wl = between(loop, '#pragma unroll\n    for (int p = 0; p < PIECES; ++p) {\n      const int o =',
                 '#pragma unroll\n    for (int p = 0; p < PIECES; ++p) {\n      unsigned char *dst', 'pfx wload')
    ws_ = between(loop, '#pragma unroll\n    for (int p = 0; p < PIECES; ++p) {\n      unsigned char *dst',
                  '    iu8_bar();\n#pragma unroll\n    for (int st = 0; st < NS; ++st) {', 'pfx wstore')
    mm = between(loop, '#pragma unroll\n    for (int st = 0; st < NS; ++st) {', '    iu8_bar();\n  }\n', 'pfx wmma')
    ws_ = sub(ws_, '      unsigned char *dst = &sW[vdst[p]];\n', '      unsigned char *dst = &sWb[vdst[p]];\n',
              'pfx wstore buf')
    mm = sub(mm, '&sW[(wn * TN + j) * NS * 256', '&sWb[(wn * TN + j) * NS * 256', 'pfx wmma buf')
    piped = ('  if constexpr (PIPE) {\n'
             '    iu8_uint4 v[PIECES];\n    int vdst[PIECES];\n'
             '    auto wload = [&](const int kk0) {\n' + wl + '    };\n'
             '    wload(0);\n'
             '    int buf = 0;\n'
             '    for (int k0 = 0; k0 < K; k0 += LBK) {\n      const int kk0 = k0 >> 4;\n'
             '      unsigned char *sWb = sW + buf * WBUF;\n' + ws_ + af +
             '      if (k0 + LBK < K) wload(kk0 + (LBK >> 4));\n'
             '      iu8_bar();\n' + mm + '      buf ^= 1;\n    }\n'
             '  } else {\n' + loop + '  }\n')
    s = s.replace(loop, piped, 1)
    return s


def arnq_pieces():
    s = open(os.path.join(HIP, 'mxfp4_gemm_decode_arnq_had.hip')).read()
    head = between(s, '#define SABER_AR_SPIN_MAX 4000000000ULL\n', 'static_assert(SABER_ARNQ_T == DEC_DWN * 32',
                   'arnq epilogue slice')
    head = sub(head, '__device__ __forceinline__ void arnq_epilogue_impl(',
               '__device__ __forceinline__ void iu8f_arnq_epilogue(', 'epilogue name')
    head = sub(head, '  res += (size_t)b * K; q += (size_t)b * K; res_out += (size_t)b * K;\n',
               '  res += (size_t)b * K; res_out += (size_t)b * K;  // q is the A-tiled edge base\n',
               'q base')
    i = head.find('  const float sc = fmaxf(lds[0] * (1.f / 448.f), 1.f / (448.f * 512.f));\n')
    if head.count('  const float sc = fmaxf(lds[0] * (1.f / 448.f), 1.f / (448.f * 512.f));\n') != 1:
        raise SystemExit('e4m3 tail anchor')
    head = head[:i] + I8_TAIL
    proto = between(s, '  __shared__ int s_row;\n', '  arnq_epilogue_impl(row, reinterpret_cast<const int4 *>(C)',
                    'row-3 protocol slice')
    return head, proto


PRE = '''// GENERATED by tools/gen_macaq_iu8f.py from crates/saber-kernels/hip/macaq_iu8.hip,
// mxfp4_gemm_decode_arnq_had.hip and saber_ar_hadi8.hip -- do not edit; regenerate.
// Saber source (2026-11-17): the W4A8-iu8 pack's fused decode bodies and band tiles:
//   macaq_iu8f_pair_w4_tm{1..4}        the decode PAIR: two W4 GEMMs on ONE A edge, one dispatch
//   macaq_iu8f_arnq_hadi8_{w4,w8}_tm1  fusion-map ROW 3: decode GEMM + exact 2-rank all-reduce + `_nq` (int8)
//   macaq_iu8f_pairb_w4_tm{1..4}       the decode pair, the ba half's blocks first
//   macaq_iu8f_dec_{w4,w8}_tm{5..8}    the decode body at M 65..128 (small prefill chunks)
//   macaq_iu8f_pf_{w4,w8}_bm{128,64}   the prefill tile at BM 128 / 64
//   macaq_iu8f_pf_{w4,w8}_t*w*n*k*[p]  the parametric prefill tiles at the p96..p384 rungs
// Every arithmetic line is a plain iu8 source's own (see the generator's docstring for the slices).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include "macaq_had.h"
#include "macaq_i8.h"

'''

PAIR = '''
// ============================================================================ (a) the decode pair
// Problem 0's grid (nb x ks blocks, n-block fastest, as `macaq_iu8_dec_*` dispatches it) then problem 1's;
// both halves W4, the same M and K, the shared A tile and per-token scale.
template <int DTM>
__device__ __forceinline__ void iu8f_pair_impl(const unsigned char *__restrict__ a, const float *__restrict__ as,
                                               const unsigned char *__restrict__ w, const float *__restrict__ ws,
                                               int *__restrict__ P, int *__restrict__ cnt, __bf16 *__restrict__ c,
                                               const unsigned char *__restrict__ w_b, const float *__restrict__ ws_b,
                                               int *__restrict__ P_b, int *__restrict__ cnt_b,
                                               __bf16 *__restrict__ c_b, int M, int K, int N0, int ks0, int nb0,
                                               int N1, int ks1, int nb1) {
  const int b0 = nb0 * ks0;
  const bool p1 = (int)blockIdx.x >= b0;
  const int bl = p1 ? (int)blockIdx.x - b0 : (int)blockIdx.x;
  const int nb = p1 ? nb1 : nb0;
  (void)iu8f_dec_impl<DTM, false>(bl % nb, bl / nb, p1 ? ks1 : ks0, a, p1 ? w_b : w, as, p1 ? ws_b : ws,
                                  p1 ? P_b : P, p1 ? cnt_b : cnt, p1 ? c_b : c, M, p1 ? N1 : N0, K);
}

#define IU8F_PAIR(name, DTM)                                                                                   \\
  extern "C" __global__ __launch_bounds__(256) void name(                                                     \\
      const unsigned char *__restrict__ a, const float *__restrict__ as, const unsigned char *__restrict__ w, \\
      const float *__restrict__ ws, int *__restrict__ decode_partial, int *__restrict__ decode_counter,        \\
      __bf16 *__restrict__ c, const unsigned char *__restrict__ w_b, const float *__restrict__ ws_b,           \\
      int *__restrict__ decode_partial_b, int *__restrict__ decode_counter_b, __bf16 *__restrict__ c_b,        \\
      int m, int k, int n, int ks, int nb, int n_b, int ks_b, int nb_b) {                                     \\
    iu8f_pair_impl<DTM>(a, as, w, ws, decode_partial, decode_counter, c, w_b, ws_b, decode_partial_b,          \\
                        decode_counter_b, c_b, m, k, n, ks, nb, n_b, ks_b, nb_b);                              \\
  }
IU8F_PAIR(macaq_iu8f_pair_w4_tm1, 1)
IU8F_PAIR(macaq_iu8f_pair_w4_tm2, 2)
IU8F_PAIR(macaq_iu8f_pair_w4_tm3, 3)
IU8F_PAIR(macaq_iu8f_pair_w4_tm4, 4)
#undef IU8F_PAIR

// ============================================================================ (b) row 3
'''

PAIRB = '''
// ============================================================================ (c) the ba-first pair
// Problem 1's grid (the ba half: nb1 x ks1 blocks) FIRST, then problem 0's; otherwise (a).
template <int DTM>
__device__ __forceinline__ void iu8f_pairb_impl(const unsigned char *__restrict__ a, const float *__restrict__ as,
                                                const unsigned char *__restrict__ w, const float *__restrict__ ws,
                                                int *__restrict__ P, int *__restrict__ cnt, __bf16 *__restrict__ c,
                                                const unsigned char *__restrict__ w_b, const float *__restrict__ ws_b,
                                                int *__restrict__ P_b, int *__restrict__ cnt_b,
                                                __bf16 *__restrict__ c_b, int M, int K, int N0, int ks0, int nb0,
                                                int N1, int ks1, int nb1) {
  const int b1 = nb1 * ks1;
  const bool p1 = (int)blockIdx.x < b1;
  const int bl = p1 ? (int)blockIdx.x : (int)blockIdx.x - b1;
  const int nb = p1 ? nb1 : nb0;
  (void)iu8f_dec_impl<DTM, false>(bl % nb, bl / nb, p1 ? ks1 : ks0, a, p1 ? w_b : w, as, p1 ? ws_b : ws,
                                  p1 ? P_b : P, p1 ? cnt_b : cnt, p1 ? c_b : c, M, p1 ? N1 : N0, K);
}

#define IU8F_PAIRB(name, DTM)                                                                                  \\
  extern "C" __global__ __launch_bounds__(256) void name(                                                     \\
      const unsigned char *__restrict__ a, const float *__restrict__ as, const unsigned char *__restrict__ w, \\
      const float *__restrict__ ws, int *__restrict__ decode_partial, int *__restrict__ decode_counter,        \\
      __bf16 *__restrict__ c, const unsigned char *__restrict__ w_b, const float *__restrict__ ws_b,           \\
      int *__restrict__ decode_partial_b, int *__restrict__ decode_counter_b, __bf16 *__restrict__ c_b,        \\
      int m, int k, int n, int ks, int nb, int n_b, int ks_b, int nb_b) {                                     \\
    iu8f_pairb_impl<DTM>(a, as, w, ws, decode_partial, decode_counter, c, w_b, ws_b, decode_partial_b,         \\
                         decode_counter_b, c_b, m, k, n, ks, nb, n_b, ks_b, nb_b);                             \\
  }
IU8F_PAIRB(macaq_iu8f_pairb_w4_tm1, 1)
IU8F_PAIRB(macaq_iu8f_pairb_w4_tm2, 2)
IU8F_PAIRB(macaq_iu8f_pairb_w4_tm3, 3)
IU8F_PAIRB(macaq_iu8f_pairb_w4_tm4, 4)
#undef IU8F_PAIRB

// ============================================================================ (d) the mid decode band
// M 65..128 (DTM = ceil(M/16) 5..8) on the decode body; grid (nb, 1, ks) as `macaq_iu8_dec_*`.
#define IU8F_DEC(name, DTM, W8)                                                                                \\
  extern "C" __global__ __launch_bounds__(256) void name(                                                     \\
      const unsigned char *__restrict__ a, const unsigned char *__restrict__ w, const float *__restrict__ as, \\
      const float *__restrict__ ws, int *__restrict__ decode_partial, int *__restrict__ decode_counter,       \\
      __bf16 *__restrict__ c, int m, int n, int k) {                                                           \\
    (void)iu8f_dec_impl<DTM, W8>((int)blockIdx.x, (int)blockIdx.z, (int)gridDim.z, a, w, as, ws, decode_partial, \\
                                 decode_counter, c, m, n, k);                                                 \\
  }
IU8F_DEC(macaq_iu8f_dec_w4_tm5, 5, false)
IU8F_DEC(macaq_iu8f_dec_w4_tm6, 6, false)
IU8F_DEC(macaq_iu8f_dec_w4_tm7, 7, false)
IU8F_DEC(macaq_iu8f_dec_w4_tm8, 8, false)
IU8F_DEC(macaq_iu8f_dec_w8_tm5, 5, true)
IU8F_DEC(macaq_iu8f_dec_w8_tm6, 6, true)
IU8F_DEC(macaq_iu8f_dec_w8_tm7, 7, true)
IU8F_DEC(macaq_iu8f_dec_w8_tm8, 8, true)
#undef IU8F_DEC

// ============================================================================ (e) the small prefill tiles
// `iu8_pf_impl` with PTM m-fragments a wave: BM = 4 * PTM * 16; grid (nb, ceil(M / BM), 1) as `macaq_iu8_pf_*`.
'''

PF_TAIL = '''
#define IU8F_PF(name, W8, PTM)                                                                                 \\
  extern "C" __global__ __launch_bounds__(256) void name(                                                     \\
      const unsigned char *__restrict__ a, const unsigned char *__restrict__ w, const float *__restrict__ as, \\
      const float *__restrict__ ws, __bf16 *__restrict__ c, int m, int n, int k) {                             \\
    iu8f_pf_impl<W8, PTM>(a, w, as, ws, c, m, n, k);                                                           \\
  }
IU8F_PF(macaq_iu8f_pf_w4_bm128, false, 2)
IU8F_PF(macaq_iu8f_pf_w4_bm64, false, 1)
IU8F_PF(macaq_iu8f_pf_w8_bm128, true, 2)
IU8F_PF(macaq_iu8f_pf_w8_bm64, true, 1)
#undef IU8F_PF
'''

ROW3_OPEN = '''
// The decode GEMM (tm1: M <= 16 rows, the exact band's M <= 12) whose split-K last arriver per 128-column
// n-block holds the exact C tile, then row 3's protocol (sliced) and the int8 `_nq` epilogue on the last M
// arrivals. N must be a whole number of 128-column n-blocks (every row-parallel N is 5,120).
template <bool W8>
__device__ __forceinline__ void iu8f_arnq_impl(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W, const float *__restrict__ as,
    const float *__restrict__ ws, int *__restrict__ P, int *__restrict__ cnt, __bf16 *__restrict__ C, int M, int N,
    int K, int4 *peer_base, const int4 *my_base, const __hip_bfloat16 *__restrict__ res,
    const __hip_bfloat16 *__restrict__ nw, unsigned char *__restrict__ q, float *__restrict__ scale,
    __hip_bfloat16 *__restrict__ res_out, unsigned int *peer_flags, unsigned int *my_flags,
    unsigned int *seq_ctrs, unsigned long long *tiles, int k16, int slot_stride16, float eps, int drain, int acq,
    int ep_rows, const float *__restrict__ had_t) {
  if (!iu8f_dec_impl<1, W8>((int)blockIdx.x, (int)blockIdx.z, (int)gridDim.z, A, W, as, ws, P, cnt, C, M, N, K))
    return;  // not this n-block's last split arriver (uniform over the block)
  const int tid = threadIdx.x;
  const int n0 = blockIdx.x * 128;
'''

ROW3_CLOSE = '''  iu8f_arnq_epilogue(row, reinterpret_cast<const int4 *>(C), my_base, res, nw, q, scale, res_out, peer_flags,
                     my_flags, seq_ctrs, k16, slot_stride16, eps, drain, acq, had_t);
}

#define IU8F_ARNQ(name, W8)                                                                                     \\
  extern "C" __global__ __launch_bounds__(256) void name(                                                       \\
      const unsigned char *__restrict__ a, const unsigned char *__restrict__ w, const float *__restrict__ as,   \\
      const float *__restrict__ ws, int *__restrict__ decode_partial, int *__restrict__ decode_counter,          \\
      __bf16 *__restrict__ c, int m, int n, int k, int4 *ar_peer_scratch, const int4 *ar_scratch,               \\
      const __hip_bfloat16 *__restrict__ residual, const __hip_bfloat16 *__restrict__ norm_w,                    \\
      unsigned char *__restrict__ q_out, float *__restrict__ scale_out, __hip_bfloat16 *__restrict__ res_out,    \\
      unsigned int *ar_peer_flags, unsigned int *ar_flags, unsigned int *ar_seq, unsigned long long *ar_tiles,   \\
      int k16, int slot_stride16, float eps, int drain, int acq, int ep_rows, const float *__restrict__ had_t) { \\
    iu8f_arnq_impl<W8>(a, w, as, ws, decode_partial, decode_counter, c, m, n, k, ar_peer_scratch, ar_scratch,   \\
                       residual, norm_w, q_out, scale_out, res_out, ar_peer_flags, ar_flags, ar_seq, ar_tiles,  \\
                       k16, slot_stride16, eps, drain, acq, ep_rows, had_t);                                    \\
  }
IU8F_ARNQ(macaq_iu8f_arnq_hadi8_w4_tm1, false)
IU8F_ARNQ(macaq_iu8f_arnq_hadi8_w8_tm1, true)
#undef IU8F_ARNQ
'''


def pfx_name(w8, ptm, pwm, tn, lbk, pipe):
    """the symbol of a parametric prefill tile (BM = PWM * PTM * 16, BN = 2 * TN * 16, PWM * 64 threads)."""
    return f'macaq_iu8f_pf_{"w8" if w8 else "w4"}_t{ptm}w{pwm}n{tn}k{lbk}{"p" if pipe else ""}'


def pfx_ok(w8, ptm, pwm, tn, lbk):
    """the W stage splits into whole 16-byte pieces a thread (the body's static_assert)."""
    pslab = 2 * tn * (lbk // 16) * (256 if w8 else 128)
    return pslab % (pwm * 64 * 16) == 0


# the tuning sweep's candidates (`--lab`) as (PTM, PWM, TN, LBK, PIPE); each is built for W4 and W8 where
# `pfx_ok`.
LAB_PFX = [(ptm, pwm, tn, lbk, pipe)
           for pwm, tn, lbk in ((4, 4, 64), (4, 2, 128), (2, 4, 64), (2, 2, 64))
           for ptm in (1, 2, 3, 4, 5) for pipe in (False, True) if not (pwm == 2 and ptm == 1)]
# the p96 / p128 / p384 sweep's candidates (`--lab27`) = LAB_PFX + the
# 384-row-exact heights (PTM 6 at PWM 4 / 2, PTM 3 at PWM 8), the 96-row-exact ones (PTM 6 / 3 at PWM 1 / 2) and the
# BN 64 x LBK 128 tiles at PWM 2 / 1, as (PTM, PWM, TN, LBK, PIPE); each built for W4 and W8 where `pfx_ok`.
LAB27_PFX = LAB_PFX + [(ptm, pwm, tn, lbk, pipe) for pipe in (False, True) for ptm, pwm, tn, lbk in (
    (6, 4, 2, 128), (6, 4, 2, 64), (6, 2, 2, 64), (6, 2, 2, 128), (6, 2, 4, 64), (3, 8, 4, 128), (3, 8, 4, 64),
    (3, 4, 2, 64), (6, 4, 4, 64), (2, 2, 2, 128), (3, 2, 2, 128), (4, 2, 2, 128), (5, 2, 2, 128),
    (6, 1, 2, 64), (6, 1, 2, 128), (3, 1, 2, 64), (4, 1, 2, 128), (2, 4, 4, 128), (3, 4, 4, 128))]
# the tiles the shipped object carries (the sweep's winners at the 129..320-row rungs; routed by
# `macaq_iu8::PF_BAND`), as (W8, PTM, PWM, TN, LBK, PIPE).
SHIP_PFX = [
    (False, 1, 4, 2, 128, True),   # the ba gate (N <= 128) at 129..320
    (False, 3, 4, 4, 64, True),    # the wide ops at p192
    (False, 4, 4, 4, 64, True),    # the wide ops at p256
    (False, 5, 4, 4, 64, True),    # the wide ops at p320
    (False, 4, 4, 2, 128, True),   # the 5,120-column W4 ops at p256
    (False, 5, 2, 2, 64, True),    # the 5,120-column W4 ops at p320
    (True, 1, 4, 2, 128, False),   # the 5,120-column W8 ops at p320
    # the p96 / p128 / p384 rungs (the `--lab27` sweep). The CARRIED rows stay in the object so the
    # shipped build is the one whose timing chose the routing (the placement law: removing a kernel re-places the rest).
    (False, 3, 2, 2, 128, True),   # p96: every W4 op but the ba gate
    (False, 3, 2, 4, 64, True),    # CARRIED, UNROUTED: the lab's p96 TP=1 gate_up winner, lost in the shipped build
    (False, 2, 2, 2, 128, True),   # CARRIED, UNROUTED: the lab's p128 5,120-column winner, lost in the shipped build
    (False, 2, 4, 4, 128, True),   # p128: the wide ops to 8,192 columns
    (True, 3, 4, 2, 64, True),     # CARRIED, UNROUTED: the lab's p384 W8 winner, lost in the shipped build
    (False, 3, 8, 4, 128, True),   # p384: every wide op (384 rows, PWM 8)
]

PFX_MACRO = """
// ============================================================================ (f) the parametric prefill tiles
// `iu8_pf_impl` with the tile free: PTM m-fragments x PWM waves down M (BM = PWM * PTM * 16), TN n-fragments x 2
// waves across N (BN = 2 * TN * 16), K slab LBK; `p` = PIPE (the next slab's W in registers during this slab's
// WMMAs, the W stage double-buffered). Grid (ceil(N / BN), ceil(M / BM), 1), PWM * 64 threads.
#define IU8F_PFX(name, W8, PTM, PWM, TN, LBK, PIPE)                                                             \\
  extern "C" __global__ __launch_bounds__(PWM * 64) void name(                                                  \\
      const unsigned char *__restrict__ a, const unsigned char *__restrict__ w, const float *__restrict__ as, \\
      const float *__restrict__ ws, __bf16 *__restrict__ c, int m, int n, int k) {                             \\
    iu8f_pfx_impl<W8, PTM, PWM, TN, LBK, PIPE>(a, w, as, ws, c, m, n, k);                                       \\
  }
"""


def pfx_tail(cells):
    lines = [f'IU8F_PFX({pfx_name(w8, *c)}, {"true" if w8 else "false"}, {c[0]}, {c[1]}, {c[2]}, {c[3]}, '
             f'{"true" if c[4] else "false"})' for w8, *c in cells]
    return PFX_MACRO + '\n'.join(lines) + '\n#undef IU8F_PFX\n'


def gen(pfx=None):
    dec = dec_body()
    head, proto = arnq_pieces()
    cells = SHIP_PFX if pfx is None else pfx
    out = PRE + dec + PAIR + head + ROW3_OPEN + proto + ROW3_CLOSE + PAIRB + pf_body() + PF_TAIL
    if cells:
        out += pfx_body() + pfx_tail(cells)
    return out


def main():
    for flag, fam in (('--lab', LAB_PFX), ('--lab27', LAB27_PFX)):
        if flag not in sys.argv[1:]:   # a sweep object's source (never committed as the shipped one)
            continue
        dst = sys.argv[sys.argv.index(flag) + 1]
        cells = [(w8, *c) for c in fam for w8 in (False, True) if pfx_ok(w8, c[0], c[1], c[2], c[3])]
        open(dst, 'w').write(gen(cells))
        print(f'wrote {dst}: {len(cells)} parametric tiles')
        for w8, *c in cells:
            print(pfx_name(w8, *c))
        return
    text = gen()
    if '--check' in sys.argv[1:]:
        cur = open(OUT).read() if os.path.exists(OUT) else None
        if cur != text:
            print(f'{OUT}: STALE (regenerate with tools/gen_macaq_iu8f.py)')
            sys.exit(1)
        print(f'{OUT}: up to date')
        return
    open(OUT, 'w').write(text)
    print(f'wrote {OUT} ({len(text)} B)')


if __name__ == '__main__':
    main()
