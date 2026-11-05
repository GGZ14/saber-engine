// Saber gated-delta-net kernels, CLEANROOM-2 (CR2-3c): the 27B batched decode in replay form (spec 2.10).
//
// Written from the CR2-3a specification. One BASE state a sequence (slot sidx[n][0]) plus a stash of the
// previous step's inputs: a launch first replays the r rows the previous step committed (k, gates, v from the
// stash; the decode row without q), stores the base at f16 and continues from its rounded value (the law that
// makes it byte-equal to the snapshot chain), then answers the step's own rows with no state store, stashing
// their inputs for the next launch. Stash region (bytes): header {magic, tag, slot, count} | a[8] u16 @64 |
// b[8] u16 @80 | k[8][128] bf16 @128 | v[8][128] bf16 @2176; two regions (parity) per (slot set, v head).
// Two forms: the operands read from global per row (replay), or staged once in LDS by a prologue (replay_lds).
#pragma once
#include "sgdn_rec.h"

namespace sgdn {

constexpr u32 kStashMagic = 0x52504C59u;
constexpr int kStashRows = 8;
constexpr int kSxA = 64, kSxB = 80, kSxK = 128, kSxV = kSxK + kStashRows * kDim * 2;

struct ReplayArgs {
  const u16* q; const u16* k; const u16* v; const u16* av; const u16* bv; long ab_stride; const float* a_log;
  const float* dt_bias; u16* state; long st_slot, st_head; u16* o; const int* cu; const int* sidx;
  long sidx_stride; const int* rp; unsigned char* stash; long sx_div, sx_head; u32* err; int H, Hg;
  float scale, sp_thr;
};

// The refusal word of a sequence (bit 0 stash mismatch / r out of range, bit 1 too many rows, bit 2 base off the
// slot-set grid).
__device__ __forceinline__ u32 replay_refusal(const u32* hdr, int r, u32 tag, int si, int T, long sx_div) {
  u32 bad = 0;
  if (r < 0 || r > kStashRows) bad |= 1u;
  if (r > 0 && !(hdr[0] == kStashMagic && hdr[1] == tag && hdr[2] == (u32)si && hdr[3] >= (u32)r)) bad |= 1u;
  if (T > kStashRows) bad |= 2u;
  if ((long)(si - 1) % sx_div != 0) bad |= 4u;
  return bad;
}

// One replayed row from operands already widened: decay + dot, u, rank-1 update (no output).
__device__ __forceinline__ void replay_row(float* h, const float* kk, float vval, float eg, float bt) {
  const float dot = decay_dot(h, kk, eg);
  const float u = (vval - dot) * bt;
#pragma unroll
  for (int j = 0; j < kHalf; ++j) h[j] = h[j] + u * kk[j];
}

// ============================================================================ global-operand form (2.10.1)
// MODE 0 = the law (round after the replay), 1 = no rounding (the control), 2 = materialize (replay + base store,
// then an empty header in the other region; no step rows).
template <int VS, int MODE>
__device__ __forceinline__ void replay_global(const ReplayArgs& a) {
  constexpr int NR = kDim / VS;
  const int n = blockIdx.x, hv = blockIdx.y, z = (VS > 1) ? (int)blockIdx.z : 0;
  const int bos = a.cu[n];
  const int T = (MODE == 2) ? 0 : a.cu[n + 1] - bos;
  if (MODE != 2 && T <= 0) return;
  const int si = a.sidx[(long)n * a.sidx_stride];
  if (si <= 0) return;
  const int tid = threadIdx.x;
  const int lrow = tid >> 1, row = lrow + z * NR, half = tid & 1, k0 = half * kHalf;
  const int hk = hv / (a.H / a.Hg);
  const int r = a.rp[3 * n], par = a.rp[3 * n + 2] & 1;
  const u32 tg = (u32)a.rp[3 * n + 1];
  const long sxi = (long)(si - 1) / a.sx_div;
  unsigned char* rgn = a.stash + ((sxi * a.H + hv) * 2) * a.sx_head;
  const unsigned char* rx = rgn + (long)par * a.sx_head;
  unsigned char* wx = rgn + (long)(par ^ 1) * a.sx_head;
  const u32 bad = replay_refusal((const u32*)rx, r, tg, si, T, a.sx_div);
  if (bad) {
    if (tid == 0) atomicOr(a.err, bad);
    if (half == 0)
      for (int t = 0; t < T; ++t) a.o[((long)(bos + t) * a.H + hv) * kDim + row] = 0xFFFFu;
    return;
  }
  u16* base = a.state + (long)si * a.st_slot + (long)hv * a.st_head + (long)row * kDim + k0;
  float h[kHalf];
  state_load<kStateF16, kHalf>(h, base);
  const float alog = E(a.a_log[hv]), dtb = a.dt_bias[hv];
  const u16* r_ab = (const u16*)(rx + kSxA);
  const u16* r_k = (const u16*)(rx + kSxK);
  const u16* r_v = (const u16*)(rx + kSxV);
  // each row's gates and v value are loaded one row ahead (with the row's k)
  u32 na = 0, nb = 0, nv = 0;
  if (r > 0) { na = r_ab[0]; nb = r_ab[kStashRows]; nv = r_v[row]; }
  for (int t = 0; t < r; ++t) {
    float kk[kHalf];
    load_row64(kk, r_k + t * kDim + k0);
    const u32 ca = na, cb = nb, cv = nv;
    if (t + 1 < r) { na = r_ab[t + 1]; nb = r_ab[kStashRows + t + 1]; nv = r_v[(t + 1) * kDim + row]; }
    const float ks = R(sumsq_pair(kk) + 1e-6f);
#pragma unroll
    for (int j = 0; j < kHalf; ++j) kk[j] = kk[j] * ks;
    const float g = -alog * softplus(bf(ca) + dtb, a.sp_thr);
    const float eg = E(g);
    const float bt = sigmoid_div(bf(cb));
    replay_row(h, kk, bf(cv), eg, bt);
  }
  if (r > 0) {
    state_store<kStateF16, kHalf>(base, h);
    if constexpr (MODE != 1) round_through_f16<kHalf>(h);
  }
  u32* wh = (u32*)wx;
  if constexpr (MODE == 2) {
    if (tid == 0 && z == 0) { wh[0] = kStashMagic; wh[1] = tg; wh[2] = (u32)si; wh[3] = 0u; }
    return;
  }
  u16* w_ab = (u16*)(wx + kSxA);
  u16* w_k = (u16*)(wx + kSxK);
  u16* w_v = (u16*)(wx + kSxV);
  for (int t = 0; t < T; ++t) {
    const long tok = bos + t;
    const u16* krow = a.k + (tok * a.Hg + hk) * kDim + k0;
    float qq[kHalf], kk[kHalf];
    load_row64(qq, a.q + (tok * a.Hg + hk) * kDim + k0);
    load_row64(kk, krow);
    if (z == 0 && lrow == 0) {
#pragma unroll
      for (int j = 0; j < kHalf; j += 8) *(u32x4*)(w_k + t * kDim + k0 + j) = *(const u32x4*)(krow + j);
    }
    qk_norm(qq, kk, a.scale);
    const u16 ra = a.av[tok * a.ab_stride + hv], rb = a.bv[tok * a.ab_stride + hv];
    if (z == 0 && tid == 0) { w_ab[t] = ra; w_ab[kStashRows + t] = rb; }
    const float g = -alog * softplus(bf(ra) + dtb, a.sp_thr);
    const float eg = E(g);
    const float bt = sigmoid_div(bf(rb));
    const u16 rv = a.v[(tok * a.H + hv) * kDim + row];
    if (half == 0) w_v[t * kDim + row] = rv;
    const float dot = decay_dot(h, kk, eg);
    const float u = (bf(rv) - dot) * bt;
    const float on = update_out(h, kk, qq, u);
    if (half == 0) a.o[(tok * a.H + hv) * kDim + row] = away16(on);
  }
  if (tid == 0 && z == 0) { wh[0] = kStashMagic; wh[1] = tg; wh[2] = (u32)si; wh[3] = (u32)T; }
}

// ============================================================================ LDS-staged form (2.10.2)
template <int VS>
__device__ __forceinline__ void replay_lds(const ReplayArgs& a) {
  constexpr int NR = kDim / VS;
  constexpr int NT = 2 * NR;
  constexpr int RV = (kStashRows * NR + NT - 1) / NT;   // stash v values a thread loads
  static_assert(3 * kStashRows <= NR, "the prologue is one pass: r + 2T vectors fit the lane pairs");
  __shared__ __attribute__((aligned(16))) float s_k[2 * kStashRows][kDim];
  __shared__ __attribute__((aligned(16))) float s_q[kStashRows][kDim];
  __shared__ float s_v[2 * kStashRows][NR];
  __shared__ float s_eg[2 * kStashRows];
  __shared__ float s_bt[2 * kStashRows];
  const int n = blockIdx.x, hv = blockIdx.y, z = (VS > 1) ? (int)blockIdx.z : 0;
  const int bos = a.cu[n];
  const int T = a.cu[n + 1] - bos;
  if (T <= 0) return;
  const int si = a.sidx[(long)n * a.sidx_stride];
  if (si <= 0) return;
  const int tid = threadIdx.x;
  const int lrow = tid >> 1, row = lrow + z * NR, half = tid & 1, k0 = half * kHalf;
  const int hk = hv / (a.H / a.Hg);
  const int r = a.rp[3 * n], par = a.rp[3 * n + 2] & 1;
  const u32 tg = (u32)a.rp[3 * n + 1];
  const long sxi = (long)(si - 1) / a.sx_div;
  unsigned char* rgn = a.stash + ((sxi * a.H + hv) * 2) * a.sx_head;
  const unsigned char* rx = rgn + (long)par * a.sx_head;
  unsigned char* wx = rgn + (long)(par ^ 1) * a.sx_head;
  const u32 bad = replay_refusal((const u32*)rx, r, tg, si, T, a.sx_div);
  if (bad) {
    if (tid == 0) atomicOr(a.err, bad);
    if (half == 0)
      for (int t = 0; t < T; ++t) a.o[((long)(bos + t) * a.H + hv) * kDim + row] = 0xFFFFu;
    return;
  }
  const u16* r_ab = (const u16*)(rx + kSxA);
  const u16* r_k = (const u16*)(rx + kSxK);
  const u16* r_v = (const u16*)(rx + kSxV);
  u16* w_ab = (u16*)(wx + kSxA);
  u16* w_k = (u16*)(wx + kSxK);
  u16* w_v = (u16*)(wx + kSxV);

  u16* base = a.state + (long)si * a.st_slot + (long)hv * a.st_head + (long)row * kDim + k0;
  float h[kHalf];
  state_load<kStateF16, kHalf>(h, base);
  const float alog = E(a.a_log[hv]), dtb = a.dt_bias[hv];

  // ---- prologue: vector p = lrow (both halves): p < r the stash's k row; r <= p < r + T the step's k row;
  // r + T <= p < r + 2T its q row -- normalised (q also scaled) into LDS
  {
    const int p = lrow;
    const int nvec = r + 2 * T;
    if (p < nvec) {
      const bool isq = p >= r + T;
      const u16* src;
      if (p < r) src = r_k + p * kDim;
      else if (!isq) src = a.k + ((long)(bos + p - r) * a.Hg + hk) * kDim;
      else src = a.q + ((long)(bos + p - r - T) * a.Hg + hk) * kDim;
      float xv[kHalf];
      load_row64(xv, src + k0);
      float ss = 0.0f;
#pragma unroll
      for (int j = 0; j < kHalf; ++j) ss = ss + xv[j] * xv[j];
      ss = pair_add(ss);
      const float sc = isq ? R(ss + 1e-6f) * a.scale : R(ss + 1e-6f);
      float* dst = isq ? &s_q[p - r - T][k0] : &s_k[p][k0];
#pragma unroll
      for (int j = 0; j < kHalf; j += 4)
        *(float4*)(dst + j) = make_float4(xv[j] * sc, xv[j + 1] * sc, xv[j + 2] * sc, xv[j + 3] * sc);
    } else {
      // pairs past the vectors (none at VS 4 with full rows) idle; the exchange above is skipped uniformly
    }
  }
  // gates of every row; the stash's v column and the step's (stashing the step's inputs)
  if (tid < r + T) {
    const int t = tid;
    u16 ga, gb;
    if (t < r) {
      ga = r_ab[t];
      gb = r_ab[kStashRows + t];
    } else {
      const long tok = bos + t - r;
      ga = a.av[tok * a.ab_stride + hv];
      gb = a.bv[tok * a.ab_stride + hv];
      if (z == 0) { w_ab[t - r] = ga; w_ab[kStashRows + t - r] = gb; }
    }
    const float g = -alog * softplus(bf(ga) + dtb, a.sp_thr);
    s_eg[t] = E(g);
    s_bt[t] = sigmoid_div(bf(gb));
  }
#pragma unroll
  for (int j = 0; j < RV; ++j) {
    const int i = tid + j * NT;
    if (i < r * NR) {
      const int t = i / NR, lr = i - t * NR;
      s_v[t][lr] = bf(r_v[t * kDim + lr + z * NR]);
    }
  }
  for (int i = tid; i < T * NR; i += NT) {
    const int t = i / NR, lr = i - t * NR;
    const u16 rv = a.v[((long)(bos + t) * a.H + hv) * kDim + z * NR + lr];
    s_v[r + t][lr] = bf(rv);
    w_v[t * kDim + z * NR + lr] = rv;
  }
  if (z == 0) {
    for (int i = tid; i < T * (kDim / 8); i += NT) {
      const int t = i / (kDim / 8), c8 = (i % (kDim / 8)) * 8;
      *(u32x4*)(w_k + t * kDim + c8) = *(const u32x4*)(a.k + ((long)(bos + t) * a.Hg + hk) * kDim + c8);
    }
  }
  lds_sync();

  // ---- replay the committed rows
  for (int t = 0; t < r; ++t) {
    const float eg = s_eg[t], bt = s_bt[t];
    const float* kr = &s_k[t][k0];
    float dot = 0.0f;
#pragma unroll
    for (int j = 0; j < kHalf; j += 4) {
      const float4 kv = *(const float4*)(kr + j);
      h[j] = h[j] * eg;         dot = dot + h[j] * kv.x;
      h[j + 1] = h[j + 1] * eg; dot = dot + h[j + 1] * kv.y;
      h[j + 2] = h[j + 2] * eg; dot = dot + h[j + 2] * kv.z;
      h[j + 3] = h[j + 3] * eg; dot = dot + h[j + 3] * kv.w;
    }
    dot = pair_add(dot);
    const float u = (s_v[t][lrow] - dot) * bt;
#pragma unroll
    for (int j = 0; j < kHalf; j += 4) {
      const float4 kv = *(const float4*)(kr + j);
      h[j] = h[j] + u * kv.x;
      h[j + 1] = h[j + 1] + u * kv.y;
      h[j + 2] = h[j + 2] + u * kv.z;
      h[j + 3] = h[j + 3] + u * kv.w;
    }
  }
  if (r > 0) {
    state_store<kStateF16, kHalf>(base, h);
    round_through_f16<kHalf>(h);
  }
  // ---- the step's rows: outputs, no state store
  for (int t = 0; t < T; ++t) {
    const long tok = bos + t;
    const float eg = s_eg[r + t], bt = s_bt[r + t];
    const float* kr = &s_k[r + t][k0];
    const float* qr = &s_q[t][k0];
    float dot = 0.0f;
#pragma unroll
    for (int j = 0; j < kHalf; j += 4) {
      const float4 kv = *(const float4*)(kr + j);
      h[j] = h[j] * eg;         dot = dot + h[j] * kv.x;
      h[j + 1] = h[j + 1] * eg; dot = dot + h[j + 1] * kv.y;
      h[j + 2] = h[j + 2] * eg; dot = dot + h[j + 2] * kv.z;
      h[j + 3] = h[j + 3] * eg; dot = dot + h[j + 3] * kv.w;
    }
    dot = pair_add(dot);
    const float u = (s_v[r + t][lrow] - dot) * bt;
    float on = 0.0f;
#pragma unroll
    for (int j = 0; j < kHalf; j += 4) {
      const float4 kv = *(const float4*)(kr + j);
      const float4 qv = *(const float4*)(qr + j);
      h[j] = h[j] + u * kv.x;         on = on + h[j] * qv.x;
      h[j + 1] = h[j + 1] + u * kv.y; on = on + h[j + 1] * qv.y;
      h[j + 2] = h[j + 2] + u * kv.z; on = on + h[j + 2] * qv.z;
      h[j + 3] = h[j + 3] + u * kv.w; on = on + h[j + 3] * qv.w;
    }
    on = pair_add(on);
    if (half == 0) a.o[(tok * a.H + hv) * kDim + row] = away16(on);
  }
  if (tid == 0 && z == 0) {
    u32* wh = (u32*)wx;
    wh[0] = kStashMagic; wh[1] = tg; wh[2] = (u32)si; wh[3] = (u32)T;
  }
}

}  // namespace sgdn

// The replay family's parameter list (the kernarg ABI) and its argument block.
#define SGDN_REPLAY_PARAMS                                                                                     \
  sgdn::cbf q, sgdn::cbf k, sgdn::cbf v, sgdn::cbf av, sgdn::cbf bv, long ab_stride, sgdn::cf32 a_log,         \
      sgdn::cf32 dt_bias, sgdn::wbf state, long st_slot, long st_head, sgdn::wbf o, sgdn::ci32 cu,             \
      sgdn::ci32 sidx, long sidx_stride, sgdn::ci32 rp, sgdn::wb8 stash, long sx_div, long sx_head,            \
      sgdn::wu32 err, int h, int hg, float scale, float sp_thr
#define SGDN_REPLAY_ARGS                                                                                       \
  sgdn::ReplayArgs {                                                                                           \
    q, k, v, av, bv, ab_stride, a_log, dt_bias, state, st_slot, st_head, o, cu, sidx, sidx_stride, rp, stash,  \
        sx_div, sx_head, err, h, hg, scale, sp_thr                                                             \
  }

