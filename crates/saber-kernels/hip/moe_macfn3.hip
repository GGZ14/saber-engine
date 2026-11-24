// Saber source (MACFN3, 2026-11-21): the MACaQ int4 routed-expert bodies for Flash-Next in the SHARED-t form
// (docs/MACFN-EXPERTS.md; the record = xtier v2 tag 3 `MACAQ_I4ROW_TSH`, saber_mem::xtier::record_layout_macaq).
// Saber's own arithmetic, DEFINED so a numpy float32 program reproduces every byte
// (tools/capture/macfn3/macfn3_ref.py on top of tools/capture/macaq21/iu8_ref.py).
//
//   macfn3_tok_hadi8_t256   the MoE input's per-token int8 row: v = f32(bf16 x), u = v * t_layer[k] (the
//                           layer-shared gate_up t), x = had32(u) (macaq_had.h's butterfly, * 1/sqrt 32 last),
//                           sc = max(amax / 127, 1e-30), q = clamp(rint(x / sc), +-127) (macaq_i8.h). ONE row a
//                           token, read by every routed expert's gate_up (the shared-t form's point).
//   macfn3_silu_hadi8_t64   the down input a ROUTE row: v = bf16(silu(g) * u) from the gate_up output's bf16 gate /
//                           up halves, then the same transform with the row's expert's OWN t_dn (record slot 5)
//                           and the same int8 law; one row a route (the expert-major compact order).
//   macfn3_i4_bk*_ks*_tm*_g* the grouped GEMM: int8 A rows (gathered by token for gate_up, g1; compact for down,
//                           g0) x int4 W codes in pack layout 17 (i4_row_frag) unpacked to the exact int8 16 * w,
//                           `v_wmma_i32_16x16x16_iu8`, an exact int32 accumulator over ALL of K (split-K partials
//                           are int32 too, so the split count is not numerics), then MACAQ21's contract:
//                             y = bf16_rne( (f32(acc) * as[row]) * (ws[n] * 1/16) ).
//
// THE ROUTE (FN4's tables, as moe_fn148): tile -> (expert, first compact row, rows); row_tok[compact row] = token.
// xptr holds 3 device pointers an expert: [codes, f32 row scales, f32 t_dn (down only; 0 for gate_up)].
// Contraction is OFF in this object (saber-kbuild's EXTRA_FLAGS row) and in every helper it includes.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "macaq_had.h"
#include "macaq_i8.h"

typedef int m3_int2 __attribute__((ext_vector_type(2)));
typedef int m3_intx8 __attribute__((ext_vector_type(8)));
typedef unsigned int m3_uint4 __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ m3_intx8 m3_wmma(m3_int2 a, m3_int2 b, m3_intx8 c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
}
static __device__ __forceinline__ float m3_epi(int acc, float a_scale, float w_scale) {
#pragma clang fp contract(off)
  const float t = (float)acc * a_scale;
  return t * (w_scale * 0.0625f);
}
static __device__ __forceinline__ float m3_bf(__bf16 b) { return (float)b; }

// ------------------------------------------------------------------------------------- the producers
// Block-wide amax of a per-thread value (blockDim.x threads, wave32).
template <int NT>
static __device__ __forceinline__ float m3_block_amax(float v, float *red) {
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
  if (lane == 0) red[wave] = v;
  __syncthreads();
  float m = red[0];
#pragma unroll
  for (int w = 1; w < NT / 32; ++w) m = fmaxf(m, red[w]);
  __syncthreads();
  return m;
}

// One token row a block: K = 2,560 (320 groups of 8; thread tid owns groups tid and tid + 256 when < K/8).
static __device__ __forceinline__ void macfn3_tok_row(const __bf16 *__restrict__ x, const float *__restrict__ t,
                                                      signed char *__restrict__ q, float *__restrict__ sc, int K) {
#pragma clang fp contract(off)
  __shared__ float red[8];
  const int row = blockIdx.x, tid = threadIdx.x;
  const int G = K >> 3;
  float v[2][8];
  bool own[2];
  float am = 0.f;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int g = tid + i * 256;
    own[i] = g < G;
    const int gg = own[i] ? g : 0;
    const uint4 raw = *reinterpret_cast<const uint4 *>(x + (size_t)row * K + gg * 8);
    const unsigned w[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[i][2 * j] = __uint_as_float(w[j] << 16);
      v[i][2 * j + 1] = __uint_as_float(w[j] & 0xFFFF0000u);
    }
    macaq_had32(v[i], t + gg * 8, tid);
    if (own[i])
#pragma unroll
      for (int j = 0; j < 8; ++j) am = fmaxf(am, fabsf(v[i][j]));
  }
  const float amax = m3_block_amax<256>(am, red);
  const float s = fmaxf(__fdiv_rn(amax, 127.f), 1e-30f);
  if (tid == 0) sc[row] = s;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    if (!own[i]) continue;
    const int g = tid + i * 256;
    unsigned lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) lo |= macaq_i8_code(v[i][j], s) << (8 * j);
#pragma unroll
    for (int j = 0; j < 4; ++j) hi |= macaq_i8_code(v[i][4 + j], s) << (8 * j);
    *reinterpret_cast<uint2 *>(q + (size_t)row * K + g * 8) = make_uint2(lo, hi);
  }
}

extern "C" __global__ __launch_bounds__(256) void macfn3_tok_hadi8_t256(
    const __bf16 *__restrict__ x, const float *__restrict__ t, signed char *__restrict__ q,
    float *__restrict__ sc, int K) {
  macfn3_tok_row(x, t, q, sc, K);
}
// MACFN3b (the engine route): the token row (the kernel above, byte for byte) + the SHARED-ONLY route tables the
// shipped MXFP4 cells read for the shared expert: the route's last `sh_tiles` tiles (route law: the shared expert's
// tiles follow every routed tile; a tile of another expert there = a broken law -> trap, never a wrong byte).
// Block 0 copies; meta_sh = {sh_tiles, meta[1..5]}.
extern "C" __global__ __launch_bounds__(256) void macfn3_tok_hadi8_sh_t256(
    const __bf16 *__restrict__ x, const float *__restrict__ t, signed char *__restrict__ q,
    float *__restrict__ sc, int K, const int *__restrict__ meta, const int *__restrict__ tile_expert,
    const int *__restrict__ tile_row0, const int *__restrict__ tile_rows, int *__restrict__ meta_sh,
    int *__restrict__ te_sh, int *__restrict__ t0_sh, int *__restrict__ tr_sh, int sh_tiles, int e_shared) {
  if (blockIdx.x == 0) {
    const int base = meta[0] - sh_tiles;
    for (int j = threadIdx.x; j < sh_tiles; j += 256) {
      const int tt = base + j;
      if (tt < 0 || tile_expert[tt] != e_shared || meta[2] < 0) __builtin_trap();
      te_sh[j] = tile_expert[tt];
      t0_sh[j] = tile_row0[tt];
      tr_sh[j] = tile_rows[tt];
    }
    if (threadIdx.x < 6) meta_sh[threadIdx.x] = threadIdx.x == 0 ? sh_tiles : meta[threadIdx.x];
  }
  macfn3_tok_row(x, t, q, sc, K);
}

// One TILE a block (<= 64 rows), its rows in order: I = the half intermediate (320 at TP=2; I/8 <= 64 groups,
// one group a thread). C = the gate_up output [rows, 2I] (gate cols [0, I), up [I, 2I)). vdbg (nullable) = the
// bf16 v the transform starts from (the oracle's handle on the silu: numpy continues from these bytes).
// MX (MACFN3b, the engine route): a tile of the route's SHARED expert (rows from meta[2] on) takes the SHIPPED
// expert-major silu-quant's law instead (`moe_silu_mul_quant_t256`'s arithmetic, character for character: bf16 silu,
// bf16 product, e4m3 at amax / 448 floored at 1 / (448 * 512)) -- the shared expert keeps its MXFP4 record and the
// shipped down cell reads those e4m3 rows; the routed tiles are the MX = false body byte for byte.
static __device__ __forceinline__ void m3_mx_silu_rows(const __bf16 *__restrict__ C, int row0, int M, int I,
                                                       unsigned char *__restrict__ q, float *__restrict__ sc,
                                                       float *red) {
  const int tid = threadIdx.x;
  const int NV = I >> 3;   // <= 64 groups (I <= 512): one group a thread
  const bool own = tid < NV;
  const int g = own ? tid : 0;
  for (int m = 0; m < M; ++m) {
    const int r = row0 + m;
    const __bf16 *gu = C + (size_t)r * 2 * I;
    float pv[8];
    float amax = 0.f;
    const uint4 rg = ((const uint4 *)gu)[g];
    const uint4 ru = ((const uint4 *)(gu + I))[g];
    const unsigned wg[4] = {rg.x, rg.y, rg.z, rg.w}, wu[4] = {ru.x, ru.y, ru.z, ru.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const unsigned gw = wg[j >> 1], uw = wu[j >> 1];
      const float x = __uint_as_float((j & 1) ? (gw & 0xFFFF0000u) : (gw << 16));
      const float uu = __uint_as_float((j & 1) ? (uw & 0xFFFF0000u) : (uw << 16));
      const float t = (float)(__bf16)(x / (1.f + expf(-x)));
      const float pp = (float)(__bf16)(t * uu);
      pv[j] = pp;
      if (own) amax = fmaxf(amax, fabsf(pp));
    }
    const float am = m3_block_amax<64>(amax, red);
    const float s = fmaxf(am * (1.f / 448.f), 1.f / (448.f * 512.f));
    const float rs = 1.f / s;
    if (tid == 0) sc[r] = s;
    if (own) {
      unsigned lo = 0, hi = 0;
#pragma unroll
      for (int j = 0; j < 4; j += 2) {
        const float a = fminf(fmaxf(pv[j] * rs, -448.f), 448.f);
        const float b = fminf(fmaxf(pv[j + 1] * rs, -448.f), 448.f);
        lo |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu) << (j * 8);
      }
#pragma unroll
      for (int j = 4; j < 8; j += 2) {
        const float a = fminf(fmaxf(pv[j] * rs, -448.f), 448.f);
        const float b = fminf(fmaxf(pv[j + 1] * rs, -448.f), 448.f);
        hi |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu) << ((j - 4) * 8);
      }
      ((uint2 *)(q + (size_t)r * I))[g] = make_uint2(lo, hi);
    }
  }
}

template <bool MX>
static __device__ __forceinline__ void macfn3_silu_impl(
    const __bf16 *__restrict__ C, const unsigned long long *__restrict__ xptr, const int *__restrict__ tile_expert,
    const int *__restrict__ tile_row0, const int *__restrict__ tile_rows, const int *__restrict__ meta,
    signed char *__restrict__ q, float *__restrict__ sc, __bf16 *__restrict__ vdbg, int I) {
#pragma clang fp contract(off)
  __shared__ float red[2];
  const int tile = blockIdx.x;
  if (tile >= meta[0]) return;
  const int tid = threadIdx.x;
  const int texp = tile_expert[tile], row0 = tile_row0[tile], M = tile_rows[tile];
  if (MX && meta[2] >= 0 && row0 >= meta[2]) {
    m3_mx_silu_rows(C, row0, M, I, (unsigned char *)q, sc, red);
    return;
  }
  const float *__restrict__ tdn = (const float *)(uintptr_t)xptr[3 * (size_t)texp + 2];
  const int G = I >> 3;
  const bool own = tid < G;
  const int g = own ? tid : 0;
  for (int m = 0; m < M; ++m) {
    const int r = row0 + m;
    float v[8];
    const uint4 rg = *reinterpret_cast<const uint4 *>(C + (size_t)r * 2 * I + g * 8);
    const uint4 ru = *reinterpret_cast<const uint4 *>(C + (size_t)r * 2 * I + I + g * 8);
    const unsigned wg[4] = {rg.x, rg.y, rg.z, rg.w}, wu[4] = {ru.x, ru.y, ru.z, ru.w};
    unsigned vb[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float gv = __uint_as_float(h ? (wg[j] & 0xFFFF0000u) : (wg[j] << 16));
        const float uv = __uint_as_float(h ? (wu[j] & 0xFFFF0000u) : (wu[j] << 16));
        const float e = __expf(-gv);
        const float sv = __fdiv_rn(gv, 1.f + e);
        const __bf16 b = (__bf16)(sv * uv);
        v[2 * j + h] = m3_bf(b);
        const unsigned short bb = __builtin_bit_cast(unsigned short, b);
        if (h) vb[j] |= (unsigned)bb << 16; else vb[j] = bb;
      }
    }
    if (vdbg != nullptr && own)
      *reinterpret_cast<uint4 *>(vdbg + (size_t)r * I + g * 8) = make_uint4(vb[0], vb[1], vb[2], vb[3]);
    macaq_had32(v, tdn + g * 8, tid);
    float am = 0.f;
    if (own)
#pragma unroll
      for (int j = 0; j < 8; ++j) am = fmaxf(am, fabsf(v[j]));
    const float amax = m3_block_amax<64>(am, red);
    const float s = fmaxf(__fdiv_rn(amax, 127.f), 1e-30f);
    if (tid == 0) sc[r] = s;
    if (own) {
      unsigned lo = 0, hi = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) lo |= macaq_i8_code(v[j], s) << (8 * j);
#pragma unroll
      for (int j = 0; j < 4; ++j) hi |= macaq_i8_code(v[4 + j], s) << (8 * j);
      *reinterpret_cast<uint2 *>(q + (size_t)r * I + g * 8) = make_uint2(lo, hi);
    }
  }
}


extern "C" __global__ __launch_bounds__(64) void macfn3_silu_hadi8_t64(
    const __bf16 *__restrict__ C, const unsigned long long *__restrict__ xptr, const int *__restrict__ tile_expert,
    const int *__restrict__ tile_row0, const int *__restrict__ tile_rows, const int *__restrict__ meta,
    signed char *__restrict__ q, float *__restrict__ sc, __bf16 *__restrict__ vdbg, int I) {
  macfn3_silu_impl<false>(C, xptr, tile_expert, tile_row0, tile_rows, meta, q, sc, vdbg, I);
}
extern "C" __global__ __launch_bounds__(64) void macfn3_silu_hadi8_mx_t64(
    const __bf16 *__restrict__ C, const unsigned long long *__restrict__ xptr, const int *__restrict__ tile_expert,
    const int *__restrict__ tile_row0, const int *__restrict__ tile_rows, const int *__restrict__ meta,
    signed char *__restrict__ q, float *__restrict__ sc, __bf16 *__restrict__ vdbg, int I) {
  macfn3_silu_impl<true>(C, xptr, tile_expert, tile_row0, tile_rows, meta, q, sc, vdbg, I);
}

// ------------------------------------------------------------------------------------- the grouped GEMM
// 8 waves a block (BN = 128 output columns, one 16-column n-tile a wave), one route tile a blockIdx.y, a split of
// the K slabs a blockIdx.z. A: the tile's int8 rows staged in LDS once a slab (row-major source, clamped rows).
// W: each wave's fragment dwords nontemporal straight into the B operand (layout 17: 128 contiguous bytes a wave a
// 16-k step). DBK = 16 * NS bytes a slab; K % DBK == 0 (gate_up K 2,560 at NS 8; down K 320 at NS 4).
// an LDS-only workgroup barrier (FN31's law: a __syncthreads() is a release fence = it drains the W prefetch)
static __device__ __forceinline__ void m3_bar() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
// PF = the register double buffer: the next slab's W dwords are in flight while this slab's WMMAs run, behind
// LDS-only slab barriers (byte-identical to PF = false by construction: the same int32 sums).
// SX (MACFN3b, the engine route): a tile of the route's SHARED expert (its rows from meta[2] on, meta[2] >= 0) exits
// at entry -- the shared expert keeps its served MXFP4 record and runs on the shipped cell over the shared-only tables
// `macfn3_tok_hadi8_sh_t256` copies; every other tile is the SX = false body byte for byte.
template <int NS, int DKS, int DTM, bool GATHER, bool PF = false, bool SX = false>
__device__ __forceinline__ void macfn3_i4_impl(
    const signed char *__restrict__ A, const unsigned long long *__restrict__ xptr,
    const int *__restrict__ tile_expert, const int *__restrict__ tile_row0, const int *__restrict__ tile_rows,
    const int *__restrict__ row_tok, const int *__restrict__ meta, const float *__restrict__ As,
    int *__restrict__ P, int *__restrict__ cnt, __bf16 *__restrict__ C, int N, int K, int Pmax) {
  constexpr int DBK = 16 * NS, DASTR = DBK + 8, NTH = 256;
  __shared__ signed char sA[16 * DTM * DASTR];
  __shared__ int s_arow[16 * DTM];
  __shared__ int s_last;
  const int tile = blockIdx.y;
  if (tile >= meta[0]) return;
  if (SX && meta[2] >= 0 && tile_row0[tile] >= meta[2]) return;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int texp = tile_expert[tile], row0 = tile_row0[tile], M = tile_rows[tile];
  const unsigned char *__restrict__ W = (const unsigned char *)(uintptr_t)xptr[3 * (size_t)texp + 0];
  const float *__restrict__ Ws = (const float *)(uintptr_t)xptr[3 * (size_t)texp + 1];
  for (int r = tid; r < 16 * DTM; r += NTH) {
    const int rr = r < M - 1 ? r : M - 1;
    s_arow[r] = GATHER ? row_tok[row0 + rr] : row0 + rr;
  }
  __syncthreads();
  const int ntiles = N >> 4;
  const int nt = blockIdx.x * 8 + wave;
  const bool live = nt < ntiles;
  const int ntl = live ? nt : ntiles - 1;   // a dead wave streams a live tile's bytes and stores nothing
  const int ksg = K >> 4;
  const int slabs = K / DBK, spb = (slabs + DKS - 1) / DKS;
  const int ks = blockIdx.z;
  const int s_lo = ks * spb, s_hi = min(slabs, s_lo + spb);
  const unsigned char *wbase = W + (size_t)ntl * ksg * 128;
  constexpr int ACH = 16 * DTM * DBK / 16;   // 16-byte A chunks a slab
  m3_intx8 acc[DTM];
#pragma unroll
  for (int i = 0; i < DTM; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0;
  auto load_w = [&](int s, unsigned (&d)[NS]) {
#pragma unroll
    for (int st = 0; st < NS; ++st)
      d[st] = __builtin_nontemporal_load(
          (const unsigned *)(wbase + ((unsigned)lane * 4 + (unsigned)((s * NS + st) * 128))));
  };
  // the A loads go out BEFORE the next slab's W prefetch: loadcnt retires in order, so an A load issued after the
  // prefetch would wait for it (and drain the double buffer)
  auto slab = [&](int s, const unsigned (&d)[NS], int pf_s, unsigned (&dn)[NS]) {
    const int kk0 = s * NS;
    for (int c = tid; c < ACH; c += NTH) {
      const int r = c / (DBK / 16), o = (c % (DBK / 16)) * 16;
      *(m3_uint4 *)&sA[r * DASTR + o] = *(const m3_uint4 *)(A + (size_t)s_arow[r] * K + kk0 * 16 + o);
    }
    if constexpr (PF) {
      m3_bar();
      if (pf_s < s_hi) load_w(pf_s, dn);
    } else {
      __syncthreads();
    }
#pragma unroll
    for (int st = 0; st < NS; ++st) {
      m3_int2 wv;
      wv[0] = (int)((d[st] << 4) & 0xF0F0F0F0u);
      wv[1] = (int)(d[st] & 0xF0F0F0F0u);
#pragma unroll
      for (int i = 0; i < DTM; ++i) {
        const signed char *pa = &sA[(i * 16 + col) * DASTR + st * 16 + kb8];
        m3_int2 af;
        af[0] = *(const int *)pa;
        af[1] = *(const int *)(pa + 4);
        acc[i] = m3_wmma(af, wv, acc[i]);
      }
    }
    if constexpr (PF) m3_bar(); else __syncthreads();
  };
  if constexpr (PF) {
    unsigned d0[NS], d1[NS];
    if (s_lo < s_hi) load_w(s_lo, d0);
    for (int s = s_lo; s < s_hi; s += 2) {
      slab(s, d0, s + 1, d1);
      if (s + 1 >= s_hi) break;
      slab(s + 1, d1, s + 2, d0);
    }
  } else {
#pragma unroll 2
    for (int s = s_lo; s < s_hi; ++s) {
      unsigned d[NS];
      load_w(s, d);
      slab(s, d, s_hi, d);
    }
  }
  const int n = ntl * 16 + col;
  if constexpr (DKS == 1) {
    if (!live) return;
    const float wsn = Ws[n];
#pragma unroll
    for (int i = 0; i < DTM; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = i * 16 + kb8 + e;
        if (m < M) C[(size_t)(row0 + m) * N + n] = (__bf16)m3_epi(acc[i][e], As[s_arow[m]], wsn);
      }
    return;
  }
  if (live) {
#pragma unroll
    for (int i = 0; i < DTM; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = i * 16 + kb8 + e;
        if (m < M) P[((size_t)ks * Pmax + row0 + m) * N + n] = acc[i][e];
      }
  }
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    s_last = (atomicAdd(&cnt[(size_t)tile * gridDim.x + blockIdx.x], 1) == DKS - 1);
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  if (tid == 0) cnt[(size_t)tile * gridDim.x + blockIdx.x] = 0;   // self-resetting
  const int nb = blockIdx.x * 128, nw = min(128, N - nb);
  for (int idx = tid; idx < nw * M; idx += NTH) {
    const int m = idx / nw, nn = nb + idx % nw;
    int sum = 0;
#pragma unroll
    for (int k = 0; k < DKS; ++k) sum += __builtin_nontemporal_load(&P[((size_t)k * Pmax + row0 + m) * N + nn]);
    C[(size_t)(row0 + m) * N + nn] = (__bf16)m3_epi(sum, As[s_arow[m]], Ws[nn]);
  }
}

#define MACFN3_I4(name, NS, KS, TM, G, ...)                                                                        \
  extern "C" __global__ __launch_bounds__(256) void name(                                                       \
      const signed char *__restrict__ a, const unsigned long long *__restrict__ xptr,                          \
      const int *__restrict__ tile_expert, const int *__restrict__ tile_row0, const int *__restrict__ tile_rows, \
      const int *__restrict__ row_tok, const int *__restrict__ meta, const float *__restrict__ as,             \
      int *__restrict__ p, int *__restrict__ cnt, __bf16 *__restrict__ c, int n, int k, int pmax) {          \
    macfn3_i4_impl<NS, KS, TM, G, ##__VA_ARGS__>(a, xptr, tile_expert, tile_row0, tile_rows, row_tok, meta, as, p, cnt, c, n, \
                                  k, pmax);                                                                     \
  }
// gate_up (gathered by token): K 2,560, slab 128
MACFN3_I4(macfn3_i4_bk128_ks1_tm1_g1, 8, 1, 1, true)
MACFN3_I4(macfn3_i4_bk128_ks2_tm1_g1, 8, 2, 1, true)
MACFN3_I4(macfn3_i4_bk128_ks4_tm1_g1, 8, 4, 1, true)
MACFN3_I4(macfn3_i4_bk128_ks1_tm2_g1, 8, 1, 2, true)
MACFN3_I4(macfn3_i4_bk128_ks2_tm2_g1, 8, 2, 2, true)
MACFN3_I4(macfn3_i4_bk128_ks1_tm4_g1, 8, 1, 4, true)
// down (compact rows): K 320 at TP=2 (slab 64) / 640 whole (slab 128)
MACFN3_I4(macfn3_i4_bk64_ks1_tm1_g0, 4, 1, 1, false)
MACFN3_I4(macfn3_i4_bk64_ks2_tm1_g0, 4, 2, 1, false)
MACFN3_I4(macfn3_i4_bk64_ks1_tm2_g0, 4, 1, 2, false)
MACFN3_I4(macfn3_i4_bk64_ks1_tm4_g0, 4, 1, 4, false)
MACFN3_I4(macfn3_i4_bk128_ks1_tm1_g0, 8, 1, 1, false)
MACFN3_I4(macfn3_i4_bk128_ks2_tm1_g0, 8, 2, 1, false)
// PF: the register double buffer (byte-identical twins)
MACFN3_I4(macfn3_i4_bk128_ks1_tm1_g1_pf, 8, 1, 1, true, true)
MACFN3_I4(macfn3_i4_bk128_ks2_tm1_g1_pf, 8, 2, 1, true, true)
MACFN3_I4(macfn3_i4_bk128_ks4_tm1_g1_pf, 8, 4, 1, true, true)
MACFN3_I4(macfn3_i4_bk64_ks1_tm1_g0_pf, 4, 1, 1, false, true)
MACFN3_I4(macfn3_i4_bk64_ks2_tm1_g0_pf, 4, 2, 1, false, true)
// SX: the engine route's cells (the route's shared tiles exit; ks 1 = the fastest split, exact int32 at any split)
MACFN3_I4(macfn3_i4_bk128_ks1_tm1_g1_pf_sx, 8, 1, 1, true, true, true)
MACFN3_I4(macfn3_i4_bk128_ks1_tm2_g1_sx, 8, 1, 2, true, false, true)
MACFN3_I4(macfn3_i4_bk128_ks1_tm4_g1_sx, 8, 1, 4, true, false, true)
MACFN3_I4(macfn3_i4_bk64_ks1_tm1_g0_pf_sx, 4, 1, 1, false, true, true)
MACFN3_I4(macfn3_i4_bk64_ks1_tm2_g0_sx, 4, 1, 2, false, false, true)
MACFN3_I4(macfn3_i4_bk64_ks1_tm4_g0_sx, 4, 1, 4, false, false, true)
#undef MACFN3_I4
