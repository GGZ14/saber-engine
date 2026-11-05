// Saber gated-delta-net kernels, CLEANROOM-2 (CR2-3c): the decode recurrence (spec 2.4) and the one-launch
// fused decode (conv + grid barrier + recurrence + pad rows, spec 2.4.3).
//
// Written from the CR2-3a specification. Two threads own one v row of the [V][K] state: thread `tid` holds
// row tid >> 1 and the 64 k elements [64 * (tid & 1), +64) in registers (f32 accumulators, whatever the storage).
// One row (token) is, per thread, in this order:
//   q, k widened; per-half sums of squares (sequential from 0) + one pair exchange; q *= R(sq + 1e-6) * scale,
//   k *= R(sk + 1e-6); h = h * eg, dot = dot + h * k (from 0) + pair; u = (v - dot) * beta;
//   h = h + u * k, on = on + h * q (from 0) + pair; [gated row norm]; o = away16(on); [state snapshot].
#pragma once
#include "sgdn_conv.h"

namespace sgdn {

// ---- gates of one token (uniform over the workgroup)
template <bool AB_BF16>
__device__ __forceinline__ void token_gates(const void* av, const void* bv, long idx, float alog, float dtb,
                                            float thr, float& eg, float& bt) {
  const float a = gate_in<AB_BF16>(av, idx);
  const float b = gate_in<AB_BF16>(bv, idx);
  const float g = -alog * softplus(a + dtb, thr);
  eg = E(g);
  bt = sigmoid_div(b);
}

// 64 bf16 at p -> f32
__device__ __forceinline__ void load_row64(float* dst, const u16* __restrict__ p) {
#pragma unroll
  for (int j = 0; j < kHalf; j += 8) {
    const u32x4 w = *(const u32x4*)(p + j);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      dst[j + 2 * e] = bf_lo(w[e]);
      dst[j + 2 * e + 1] = bf_hi(w[e]);
    }
  }
}

// sequential sum of squares from 0, then the pair exchange
__device__ __forceinline__ float sumsq_pair(const float* x) {
  float s = 0.0f;
#pragma unroll
  for (int j = 0; j < kHalf; ++j) s = s + x[j] * x[j];
  return pair_add(s);
}

// q, k of one token, normalised (q also scaled)
__device__ __forceinline__ void qk_norm(float* qq, float* kk, float scale) {
  const float sq = sumsq_pair(qq);
  const float sk = sumsq_pair(kk);
  const float qs = R(sq + 1e-6f) * scale;
  const float ks = R(sk + 1e-6f);
#pragma unroll
  for (int j = 0; j < kHalf; ++j) {
    qq[j] = qq[j] * qs;
    kk[j] = kk[j] * ks;
  }
}

// decay + k.h; returns the pair-summed dot
__device__ __forceinline__ float decay_dot(float* h, const float* kk, float eg) {
  float dot = 0.0f;
#pragma unroll
  for (int j = 0; j < kHalf; ++j) {
    h[j] = h[j] * eg;
    dot = dot + h[j] * kk[j];
  }
  return pair_add(dot);
}

// rank-1 update + q.h; returns the pair-summed output
__device__ __forceinline__ float update_out(float* h, const float* kk, const float* qq, float u) {
  float on = 0.0f;
#pragma unroll
  for (int j = 0; j < kHalf; ++j) {
    h[j] = h[j] + u * kk[j];
    on = on + h[j] * qq[j];
  }
  return pair_add(on);
}

// ---- the gated row norm epilogue (spec 2.6.1, recurrent / tree form): every wave's sum through LDS
template <int NW>
__device__ __forceinline__ float row_norm(float on, int half, int tid, float* wsum, float nwr, float zf, float neps,
                                          int nact) {
  float s = (half == 0) ? on * on : 0.0f;
  s = wave_sum(s);
  if ((tid & 31) == 0) wsum[tid >> 5] = s;
  __syncthreads();
  float tot = 0.0f;
#pragma unroll
  for (int w = 0; w < NW; ++w) tot = tot + wsum[w];
  const float sc = R(tot * 0.0078125f + neps);
  const float sg = 1.0f / (1.0f + E(-zf));
  const float act = (nact == 0) ? zf * sg : sg;
  const float r = on * ((sc * nwr) * act);
  __syncthreads();   // every wave has read the sums before the next row writes them
  return r;
}

// ============================================================================ recurrent update (spec 2.4)
struct RecArgs {
  const u16* q; const u16* k; const u16* v; const void* av; const void* bv; long ab_stride;
  const float* a_log; const float* dt_bias; void* state; long st_slot, st_head; u16* o; const int* cu;
  const int* sidx; long sidx_stride; const int* naccept; const u16* zg; const float* nw; float neps; int nact;
  int H, Hg; float scale, sp_thr;
};

// The rows of one (sequence n, v head hv, slice z): VS slices split the 128 rows. CARRY = the f16 carry between
// rows (spec 2.4.4).
template <int KIND, bool AB_BF16, bool NORM, int VS, bool CARRY>
__device__ __forceinline__ void recurrent_rows(const RecArgs& a, int n, int hv, int z, float* wsum) {
  typedef typename StateElem<KIND>::type ST;
  constexpr int NR = kDim / VS;
  constexpr int NW = (2 * NR) / 32;
  const int bos = a.cu[n];
  const int T = a.cu[n + 1] - bos;
  if (T <= 0) return;
  const long srow = (long)n * a.sidx_stride;
  const int si = a.sidx[srow + (a.naccept ? a.naccept[n] - 1 : 0)];
  if (si <= 0) return;
  const int tid = threadIdx.x;
  const int row = (tid >> 1) + z * NR, half = tid & 1, k0 = half * kHalf;
  const int hk = hv / (a.H / a.Hg);
  ST* S = (ST*)a.state;
  const long soff = (long)hv * a.st_head + (long)row * kDim + k0;
  float h[kHalf];
  state_load<KIND, kHalf>(h, S + (long)si * a.st_slot + soff);
  const float alog = E(a.a_log[hv]);
  const float dtb = a.dt_bias[hv];
  for (int t = 0; t < T; ++t) {
    const long tok = bos + t;
    // the row's scalar operands first, so their waits fold into the q / k one
    const u32 vraw = a.v[(tok * a.H + hv) * kDim + row];
    const float graw = gate_in<AB_BF16>(a.av, tok * a.ab_stride + hv);
    const float braw = gate_in<AB_BF16>(a.bv, tok * a.ab_stride + hv);
    const int so = a.sidx[srow + t];
    u32 zraw = 0;
    float nwr = 0.0f;
    if constexpr (NORM) {
      zraw = a.zg[(tok * a.H + hv) * kDim + row];
      nwr = a.nw[row];
    }
    // q stays packed (32 registers, not 64): each element is widened where it is used, the same value each time
    u32x4 qraw[kHalf / 8];
    const u16* qp = a.q + (tok * a.Hg + hk) * kDim + k0;
#pragma unroll
    for (int j = 0; j < kHalf / 8; ++j) qraw[j] = *(const u32x4*)(qp + 8 * j);
    float kk[kHalf];
    load_row64(kk, a.k + (tok * a.Hg + hk) * kDim + k0);
    auto qel = [&](int j) { const u32 w = qraw[j >> 3][(j >> 1) & 3]; return (j & 1) ? bf_hi(w) : bf_lo(w); };
    float sq = 0.0f;
#pragma unroll
    for (int j = 0; j < kHalf; ++j) sq = sq + qel(j) * qel(j);
    sq = pair_add(sq);
    const float sk = sumsq_pair(kk);
    const float qs = R(sq + 1e-6f) * a.scale;
    const float ks = R(sk + 1e-6f);
#pragma unroll
    for (int j = 0; j < kHalf; ++j) kk[j] = kk[j] * ks;
    const float eg = E(-alog * softplus(graw + dtb, a.sp_thr));
    const float bt = sigmoid_div(braw);
    const float dot = decay_dot(h, kk, eg);
    const float u = (bf(vraw) - dot) * bt;
    float on = 0.0f;
#pragma unroll
    for (int j = 0; j < kHalf; ++j) {
      h[j] = h[j] + u * kk[j];
      on = on + h[j] * (qel(j) * qs);
    }
    on = pair_add(on);
    if constexpr (NORM)
      on = row_norm<NW>(on, half, tid, wsum, nwr, bf(zraw), a.neps, a.nact);
    if (half == 0) a.o[(tok * a.H + hv) * kDim + row] = away16(on);
    if (so > 0) state_store<KIND, kHalf>(S + (long)so * a.st_slot + soff, h);
    if constexpr (CARRY) {
      if (t + 1 < T) round_through_f16<kHalf>(h);
    }
  }
}

// ============================================================================ norm + fp8 quant tails
// The gated norm of one 128-wide head row of o (16 lanes, eight elements each), its bf16 output and its e4m3
// quantisation with one scale a (row, head) (spec 2.6.2). The sum of squares is written with explicit fmas (the
// contraction the spec records): ssq = s1^2, then fma(s0, s0, .), fma(s_j, s_j, .) for j = 2..7.
struct Nq8Args {
  const u16* nz; long nzs; const u16* nw8; u16* nout; unsigned char* nq; float* nsc; float neps8;
};

__device__ __forceinline__ float sumsq8_fma(const float* s) {
  float ssq = s[1] * s[1];
  ssq = fmaf(s[0], s[0], ssq);
#pragma unroll
  for (int j = 2; j < 8; ++j) ssq = fmaf(s[j], s[j], ssq);
  return ssq;
}

__device__ __forceinline__ float half_wave_max(float a) {
  a = fmaxf(a, __shfl_xor(a, 1, 32));
  a = fmaxf(a, __shfl_xor(a, 2, 32));
  a = fmaxf(a, __shfl_xor(a, 4, 32));
  a = fmaxf(a, __shfl_xor(a, 8, 32));
  return a;
}

__device__ __forceinline__ float clamp448(float x) { return fminf(fmaxf(x, -448.0f), 448.0f); }

// eight values -> eight e4m3 bytes in element order
__device__ __forceinline__ u32x2 e4m3x8(const float* x) {
  int lo = __builtin_amdgcn_cvt_pk_fp8_f32(x[0], x[1], 0, false);
  lo = __builtin_amdgcn_cvt_pk_fp8_f32(x[2], x[3], lo, true);
  int hi = __builtin_amdgcn_cvt_pk_fp8_f32(x[4], x[5], 0, false);
  hi = __builtin_amdgcn_cvt_pk_fp8_f32(x[6], x[7], hi, true);
  return u32x2{(u32)lo, (u32)hi};
}

// RC = scale by the reciprocal constant (the _rc arm) instead of the division
template <bool RC>
__device__ __forceinline__ void nq8_head_row(const u16* __restrict__ o, int H, const Nq8Args& t, long r8, int hv,
                                             int l) {
  const long base = (r8 * H + hv) * kDim + 8 * l;
  const u32x4 ow = *(const u32x4*)(o + base);
  const u32x4 zw = *(const u32x4*)(t.nz + r8 * t.nzs + (long)hv * kDim + 8 * l);
  const u32x4 ww = *(const u32x4*)(t.nw8 + 8 * l);
  float s[8], z[8], w[8];
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    s[2 * e] = bf_lo(ow[e]); s[2 * e + 1] = bf_hi(ow[e]);
    z[2 * e] = bf_lo(zw[e]); z[2 * e + 1] = bf_hi(zw[e]);
    w[2 * e] = bf_lo(ww[e]); w[2 * e + 1] = bf_hi(ww[e]);
  }
  float ssq = half_wave_sum(sumsq8_fma(s));
  const float inv = R(fmaf(ssq, 0.0078125f, t.neps8));
  float nb[8];
  u32 nw[4];
#pragma unroll
  for (int e = 0; e < 8; e += 2) {
    const u32 lo = rne16(((s[e] * inv) * w[e]) * (1.0f / (1.0f + expf(-z[e]))));
    const u32 hi = rne16(((s[e + 1] * inv) * w[e + 1]) * (1.0f / (1.0f + expf(-z[e + 1]))));
    nw[e / 2] = lo | (hi << 16);
    nb[e] = bf(lo);
    nb[e + 1] = bf(hi);
  }
  *(u32x4*)(t.nout + base) = u32x4{nw[0], nw[1], nw[2], nw[3]};
  float amax = 0.0f;
#pragma unroll
  for (int e = 0; e < 8; ++e) amax = fmaxf(amax, fabsf(nb[e]));
  amax = half_wave_max(amax);
  const float sc = fmaxf(RC ? amax * (1.0f / 448.0f) : amax / 448.0f, 1.0f / (448.0f * 512.0f));
  if (l == 0) t.nsc[r8 * H + hv] = sc;
  float qv[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) qv[e] = clamp448(nb[e] / sc);
  *(u32x2*)(t.nq + base) = e4m3x8(qv);
}

// the tail over a (sequence, head)'s rows, one 16-lane half-wave a row
template <bool RC>
__device__ __forceinline__ void nq8_tail(const u16* o, int H, const Nq8Args& t, int bos, int T, int hv) {
  const int tid = threadIdx.x;
  for (int t8 = tid >> 4; t8 < T; t8 += 16) nq8_head_row<RC>(o, H, t, (long)bos + t8, hv, tid & 15);
}

// ============================================================================ fused update (spec 2.4.3)
struct FusedArgs {
  const u16* x; long xpitch; const u16* wgt; const u16* bias; u16* cstate; long cs_seq, cs_dim, cs_tok;
  int state_len_max; const int* cache_idx; long ci_stride; const int* naccept; const int* cu; int nseq, H, Hg, maxq;
  u16* q; u16* k; u16* v; const void* av; const void* bv; long ab_stride; const float* a_log; const float* dt_bias;
  void* state; long st_slot, st_head; u16* o; const int* sidx; long sidx_stride; float scale, sp_thr; int* bar;
  int num_wg, o_rows;
};

// Every workgroup of the launch meets here once (all are co-resident: num_wg <= 32). The last arriver resets
// the counter, so it is zero again for the next launch.
__device__ __forceinline__ void grid_barrier(int* bar, int num_wg) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const int old = atomicAdd(bar, 1);
    if (old == num_wg - 1) {
      __threadfence();
      atomicExch(bar, 0);
    } else {
      while (atomicAdd(bar, 0) != 0) __builtin_amdgcn_s_sleep(8);
    }
  }
  __syncthreads();
  __threadfence();
}

// phase 1: the decode conv, one wave an item (sequence, unit)
__device__ __forceinline__ void fused_conv_phase(const FusedArgs& a) {
  const int units = 2 * a.Hg + a.H;
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int items = a.nseq * units;
  for (int item = blockIdx.x * 8 + wave; item < items; item += a.num_wg * 8) {
    const int n = item / units, unit = item - n * units;
    conv_update_unit(a.x, a.xpitch, a.wgt, a.bias, a.cstate, a.cs_seq, a.cs_dim, a.cs_tok, a.state_len_max,
                     a.cache_idx, a.ci_stride, a.naccept, a.q, a.k, a.v, a.cu, a.H, a.Hg, a.maxq, n, unit, lane);
  }
}

// phase 3: zero the o rows past the last sequence
__device__ __forceinline__ void fused_pad_phase(const FusedArgs& a) {
  const int first = a.cu[a.nseq];
  const int per_row = a.H * kDim / 4;   // 8-byte stores a row
  for (int r = first + blockIdx.x; r < a.o_rows; r += a.num_wg) {
    u32x2* dst = (u32x2*)(a.o + (long)r * a.H * kDim);
    for (int e = threadIdx.x; e < per_row; e += 256) dst[e] = u32x2{0u, 0u};
  }
}

// NQ8: 0 = none, 1 = the norm + quant tail (scale by division), 2 = the same with the reciprocal scale (_rc)
template <int KIND, bool AB_BF16, bool CARRY, int NQ8 = 0>
__device__ __forceinline__ void fused_update(const FusedArgs& a, const Nq8Args* tail = nullptr) {
  fused_conv_phase(a);
  grid_barrier(a.bar, a.num_wg);
  RecArgs r;
  r.q = a.q; r.k = a.k; r.v = a.v; r.av = a.av; r.bv = a.bv; r.ab_stride = a.ab_stride; r.a_log = a.a_log;
  r.dt_bias = a.dt_bias; r.state = a.state; r.st_slot = a.st_slot; r.st_head = a.st_head; r.o = a.o; r.cu = a.cu;
  r.sidx = a.sidx; r.sidx_stride = a.sidx_stride; r.naccept = a.naccept; r.zg = nullptr; r.nw = nullptr;
  r.neps = 0.0f; r.nact = 0; r.H = a.H; r.Hg = a.Hg; r.scale = a.scale; r.sp_thr = a.sp_thr;
  for (int item = blockIdx.x; item < a.nseq * a.H; item += a.num_wg) {
    const int n = item / a.H, hv = item - n * a.H;
    recurrent_rows<KIND, AB_BF16, false, 1, CARRY>(r, n, hv, 0, nullptr);
    if constexpr (NQ8 != 0) {
      const int bos = a.cu[n], T = a.cu[n + 1] - bos;
      __syncthreads();
      nq8_tail<NQ8 == 2>(a.o, a.H, *tail, bos, T, hv);
    }
  }
  fused_pad_phase(a);
  if constexpr (NQ8 != 0) {
    // the zeroed pad rows are normed and quantised too (every head)
    __syncthreads();
    const int first = a.cu[a.nseq];
    for (int row = first + blockIdx.x; row < a.o_rows; row += a.num_wg)
      for (int hv = threadIdx.x >> 4; hv < a.H; hv += 16)
        nq8_head_row<NQ8 == 2>(a.o, a.H, *tail, row, hv, threadIdx.x & 15);
  }
}

// the recurrent update + the norm + quant tail (the tail runs whatever the slot: on the o the rows left)
template <int KIND, bool AB_BF16, bool CARRY, int NQ8>
__device__ __forceinline__ void recurrent_nq8(const RecArgs& a, const Nq8Args& tail, int n, int hv) {
  recurrent_rows<KIND, AB_BF16, false, 1, CARRY>(a, n, hv, 0, nullptr);
  const int bos = a.cu[n], T = a.cu[n + 1] - bos;
  if (T <= 0) return;
  __syncthreads();
  nq8_tail<NQ8 == 2>(a.o, a.H, tail, bos, T, hv);
}

}  // namespace sgdn

// ---- the decode objects' parameter lists (the kernarg ABI: names, order, sizes) and argument blocks
#define SGDN_REC_PARAMS                                                                                        \
  sgdn::cbf q, sgdn::cbf k, sgdn::cbf v, sgdn::cany av, sgdn::cany bv, long ab_stride, sgdn::cf32 a_log,       \
      sgdn::cf32 dt_bias, sgdn::wany state, long st_slot, long st_head, sgdn::wbf o, sgdn::ci32 cu,            \
      sgdn::ci32 sidx, long sidx_stride, sgdn::ci32 naccept, sgdn::cbf zg, sgdn::cf32 nw, float neps, int nact, \
      int h, int hg, float scale, float sp_thr
#define SGDN_REC_ARGS                                                                                          \
  sgdn::RecArgs {                                                                                              \
    q, k, v, av, bv, ab_stride, a_log, dt_bias, state, st_slot, st_head, o, cu, sidx, sidx_stride, naccept,    \
        zg, nw, neps, nact, h, hg, scale, sp_thr                                                               \
  }
#define SGDN_FUSED_PARAMS                                                                                      \
  sgdn::cbf x, long xpitch, sgdn::cbf wgt, sgdn::cbf bias, sgdn::wbf cstate, long cs_seq, long cs_dim,         \
      long cs_tok, int state_len_max, sgdn::ci32 cache_idx, long ci_stride, sgdn::ci32 naccept, sgdn::ci32 cu, \
      int n_seqs, int h, int hg, int maxq, sgdn::wbf q, sgdn::wbf k, sgdn::wbf v, sgdn::cany av,               \
      sgdn::cany bv, long ab_stride, sgdn::cf32 a_log, sgdn::cf32 dt_bias, sgdn::wany state, long st_slot,     \
      long st_head, sgdn::wbf o, sgdn::ci32 sidx, long sidx_stride, float scale, float sp_thr, sgdn::wi32 bar, \
      int num_wg, int o_rows
#define SGDN_FUSED_ARGS                                                                                        \
  sgdn::FusedArgs {                                                                                            \
    x, xpitch, wgt, bias, cstate, cs_seq, cs_dim, cs_tok, state_len_max, cache_idx, ci_stride, naccept, cu,    \
        n_seqs, h, hg, maxq, q, k, v, av, bv, ab_stride, a_log, dt_bias, state, st_slot, st_head, o, sidx,     \
        sidx_stride, scale, sp_thr, bar, num_wg, o_rows                                                        \
  }
#define SGDN_NQ8_PARAMS \
  sgdn::cbf nz, long nzs, sgdn::cbf nw8, sgdn::wbf nout, sgdn::wb8 nq, sgdn::wf32 nsc, float neps8
#define SGDN_NQ8_ARGS sgdn::Nq8Args{nz, nzs, nw8, nout, nq, nsc, neps8}

// one recurrent symbol (VS slices of the rows; NORM adds the gated row norm and its per-wave sums in LDS)
#define SGDN_REC_KERNEL(NAME, KIND, AB, NORM, VS)                                                              \
  extern "C" __global__ __launch_bounds__(256 / VS) void NAME(SGDN_REC_PARAMS) {                              \
    __shared__ float wsum[NORM ? 8 : 1];                                                                       \
    sgdn::recurrent_rows<KIND, AB != 0, NORM != 0, VS, false>(SGDN_REC_ARGS, blockIdx.x, blockIdx.y,          \
                                                              VS > 1 ? (int)blockIdx.z : 0, wsum);             \
  }

// one fused symbol
#define SGDN_FUSED_KERNEL(NAME, KIND, AB)                                                                      \
  extern "C" __global__ __launch_bounds__(256) void NAME(SGDN_FUSED_PARAMS) {                                 \
    sgdn::fused_update<KIND, AB != 0, false>(SGDN_FUSED_ARGS);                                                 \
  }

