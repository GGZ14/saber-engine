// Saber gated-delta-net kernels, CLEANROOM-2 (CR2-3c): the depthwise causal convolution (4 taps + SiLU) in its
// prefill, decode and tree forms, and the prefill gates (spec 2.2, 2.3).
//
// Written from the CR2-3a specification. One wave owns one 128-channel unit of the mixed projection; lane l owns
// the four consecutive channels 4l .. 4l+3. The per-channel arithmetic is the spec's tap step:
//   acc = bias; acc = acc + w_i * win_i (i = 0..3, tap 0 oldest, two roundings each); out = away16(silu_div(acc)).
#pragma once
#include "sgdn_prims.h"

namespace sgdn {

// Four channels' weights (4 taps each) and biases, widened to f32.
struct ConvCoef {
  float w[4][kTaps];
  float b[4];
};

__device__ __forceinline__ void conv_coef(ConvCoef& cf, const u16* __restrict__ wgt, const u16* __restrict__ bias,
                                          long d0) {
  const u32x4 lo = *(const u32x4*)(wgt + d0 * kTaps);       // channels d0, d0+1
  const u32x4 hi = *(const u32x4*)(wgt + d0 * kTaps + 8);   // channels d0+2, d0+3
  const u32 raw[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    cf.w[c][0] = bf_lo(raw[2 * c]);
    cf.w[c][1] = bf_hi(raw[2 * c]);
    cf.w[c][2] = bf_lo(raw[2 * c + 1]);
    cf.w[c][3] = bf_hi(raw[2 * c + 1]);
  }
  if (bias) {
    const u32x2 bb = *(const u32x2*)(bias + d0);
    cf.b[0] = bf_lo(bb.x); cf.b[1] = bf_hi(bb.x); cf.b[2] = bf_lo(bb.y); cf.b[3] = bf_hi(bb.y);
  } else {
#pragma unroll
    for (int c = 0; c < 4; ++c) cf.b[c] = 0.0f;
  }
}

// One output of one channel: the tap sum in order, then the SiLU by division.
__device__ __forceinline__ float conv_tap(const float* w, float b, float x0, float x1, float x2, float x3) {
  float acc = b;
  acc = acc + w[0] * x0;
  acc = acc + w[1] * x1;
  acc = acc + w[2] * x2;
  acc = acc + w[3] * x3;
  return silu_div(acc);
}

// x row of four channels as two dwords -> four raw bf16 values
__device__ __forceinline__ void split4(u32x2 r, u32* v) {
  v[0] = r.x & 0xFFFFu; v[1] = r.x >> 16; v[2] = r.y & 0xFFFFu; v[3] = r.y >> 16;
}

// ============================================================================ prefill (spec 2.2)
struct PrepArgs {
  const u16* x; long xpitch; const u16* wgt; const u16* bias; u16* cstate; long cs_seq, cs_dim, cs_tok;
  const int* cache_idx; long ci_stride; const unsigned char* has_init; const void* av; const void* bv; long ab_stride;
  const float* a_log; const float* dt_bias; u16* q; u16* k; u16* v; float* g; float* beta; const int* cu;
  int nseq, H, Hg; float sp_thr;
};

template <bool AB_BF16>
__device__ __forceinline__ float gate_in(const void* p, long i) {
  if constexpr (AB_BF16) return bf(((const u16*)p)[i]);
  else return ((const float*)p)[i];
}

// The chunk-local gates of one v head: g (cumulated over the chunk) and beta, two tokens a lane.
template <bool AB_BF16>
__device__ __forceinline__ void prep_gates(const PrepArgs& a, int hv, long tok0, int rows, int lane) {
  const float nal = -E(a.a_log[hv]);
  const float dtb = a.dt_bias[hv];
  const int i0 = 2 * lane, i1 = 2 * lane + 1;
  float g0 = 0.0f, g1 = 0.0f, b0 = 0.0f, b1 = 0.0f;
  if (i0 < rows) {
    g0 = nal * softplus(gate_in<AB_BF16>(a.av, (tok0 + i0) * a.ab_stride + hv) + dtb, a.sp_thr);
    b0 = gate_in<AB_BF16>(a.bv, (tok0 + i0) * a.ab_stride + hv);
  }
  if (i1 < rows) {
    g1 = nal * softplus(gate_in<AB_BF16>(a.av, (tok0 + i1) * a.ab_stride + hv) + dtb, a.sp_thr);
    b1 = gate_in<AB_BF16>(a.bv, (tok0 + i1) * a.ab_stride + hv);
  }
  // inclusive Hillis-Steele scan of the pair sums over the lanes
  float p = g0 + g1;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const float t = __shfl_up(p, d, 32);
    if (lane >= d) p = p + t;
  }
  const float pre = p - (g0 + g1);
  const float c0 = pre + g0;
  const float c1 = c0 + g1;
  if (i0 < rows) {
    a.g[(tok0 + i0) * a.H + hv] = c0;
    a.beta[(tok0 + i0) * a.H + hv] = sigmoid_div(b0);
  }
  if (i1 < rows) {
    a.g[(tok0 + i1) * a.H + hv] = c1;
    a.beta[(tok0 + i1) * a.H + hv] = sigmoid_div(b1);
  }
}

// One workgroup = one 64-token chunk of one sequence x HPB units (one wave a unit); the y == 0 workgroup also
// computes the chunk's gates for every v head.
template <int HPB, bool AB_BF16>
__device__ __forceinline__ void conv_prep(const PrepArgs& a) {
  // flat chunk id -> (sequence, chunk start)
  int cid = blockIdx.x;
  int n = 0, bos = 0, slen = 0;
  bool found = false;
  for (; n < a.nseq; ++n) {
    bos = a.cu[n];
    slen = a.cu[n + 1] - bos;
    const int nch = (slen + kChunk - 1) / kChunk;
    if (cid < nch) { found = true; break; }
    cid -= nch;
  }
  if (!found) return;
  const int t0 = cid * kChunk;
  const int rows = min(kChunk, slen - t0);
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const long tok0 = (long)bos + t0;

  if (blockIdx.y == 0)
    for (int hv = wave; hv < a.H; hv += HPB) prep_gates<AB_BF16>(a, hv, tok0, rows, lane);

  const int unit = blockIdx.y * HPB + wave;
  const long d0 = (long)unit * kDim + 4 * lane;
  ConvCoef cf;
  conv_coef(cf, a.wgt, a.bias, d0);

  // the window before the chunk's first token: x when inside the sequence, else the conv cache (first chunk
  // with an initial state), else zero
  const bool init = t0 == 0 && a.has_init && a.has_init[n] != 0;
  const bool own_cache = t0 == 0 && a.cache_idx;
  u32 old[4][kHist];
  u16* cs = nullptr;
  if (init || own_cache) {
    const int slot = a.cache_idx[(long)n * a.ci_stride];
    cs = a.cstate + (long)slot * a.cs_seq;
  }
  float win[4][kHist];
#pragma unroll
  for (int i = 0; i < kHist; ++i) {
    const int idx = t0 - kHist + i;
    if (idx >= 0) {
      u32 r[4];
      split4(*(const u32x2*)(a.x + (long)(bos + idx) * a.xpitch + d0), r);
#pragma unroll
      for (int c = 0; c < 4; ++c) win[c][i] = bf(r[c]);
    } else if (init) {
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        old[c][i] = cs[(d0 + c) * a.cs_dim + (long)(kHist + idx) * a.cs_tok];
        win[c][i] = bf(old[c][i]);
      }
    } else {
#pragma unroll
      for (int c = 0; c < 4; ++c) win[c][i] = 0.0f;
    }
  }

  const int region = unit < a.Hg ? 0 : (unit < 2 * a.Hg ? 1 : 2);
  u16* dst;
  long dstride;
  if (region == 0) { dst = a.q + tok0 * a.Hg * kDim + (long)unit * kDim + 4 * lane; dstride = (long)a.Hg * kDim; }
  else if (region == 1) { dst = a.k + tok0 * a.Hg * kDim + (long)(unit - a.Hg) * kDim + 4 * lane; dstride = (long)a.Hg * kDim; }
  else { dst = a.v + tok0 * a.H * kDim + (long)(unit - 2 * a.Hg) * kDim + 4 * lane; dstride = (long)a.H * kDim; }

  const u16* xr = a.x + tok0 * a.xpitch + d0;
  for (int t = 0; t < rows; ++t) {
    u32 r[4];
    split4(*(const u32x2*)(xr + (long)t * a.xpitch), r);
    float o[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const float xc = bf(r[c]);
      o[c] = bf(away16(conv_tap(cf.w[c], cf.b[c], win[c][0], win[c][1], win[c][2], xc)));
      win[c][0] = win[c][1];
      win[c][1] = win[c][2];
      win[c][2] = xc;
    }
    u32x2 st;
    if (region < 2) {
      float ss = 0.0f;
      ss = ss + o[0] * o[0];
      ss = ss + o[1] * o[1];
      ss = ss + o[2] * o[2];
      ss = ss + o[3] * o[3];
      ss = wave_sum(ss);
      const float inv = 1.0f / sqrtf(ss + 1e-6f);
      st.x = away16x2(o[0] * inv, o[1] * inv);
      st.y = away16x2(o[2] * inv, o[3] * inv);
    } else {
      st.x = away16x2(o[0], o[1]);
      st.y = away16x2(o[2], o[3]);
    }
    *(u32x2*)(dst + (long)t * dstride) = st;
  }

  // the conv cache keeps the sequence's last three raw inputs (first chunk only; it read the old entries)
  if (own_cache) {
#pragma unroll
    for (int i = 0; i < kHist; ++i) {
      const int src = slen - kHist + i;
      u32 val[4];
      if (src >= 0) {
        split4(*(const u32x2*)(a.x + (long)(bos + src) * a.xpitch + d0), val);
      } else if (init) {
        // old entry kHist + src (< kHist), read above as the window seed
#pragma unroll
        for (int c = 0; c < 4; ++c) val[c] = cs[(d0 + c) * a.cs_dim + (long)(kHist + src) * a.cs_tok];
      } else {
#pragma unroll
        for (int c = 0; c < 4; ++c) val[c] = 0;
      }
#pragma unroll
      for (int c = 0; c < 4; ++c) cs[(d0 + c) * a.cs_dim + (long)i * a.cs_tok] = (u16)val[c];
    }
  }
  (void)old;
}

// ============================================================================ decode (spec 2.3.1 / 2.3.2)
struct ConvUpdArgs {
  const u16* x; long xpitch; const u16* wgt; const u16* bias; u16* cstate; long cs_seq, cs_dim, cs_tok;
  int state_len_max; const int* cache_idx; long ci_stride; const int* naccept; u16* q; u16* k; u16* v;
  const int* cu; int H, Hg, maxq;
};

// Where a unit's outputs go: q / k [token][Hg][128], v [token][H][128].
__device__ __forceinline__ u16* conv_dst(u16* q, u16* k, u16* v, int H, int Hg, int unit, long tok, int lane,
                                         long& stride) {
  if (unit < Hg) { stride = (long)Hg * kDim; return q + tok * stride + (long)unit * kDim + 4 * lane; }
  if (unit < 2 * Hg) { stride = (long)Hg * kDim; return k + tok * stride + (long)(unit - Hg) * kDim + 4 * lane; }
  stride = (long)H * kDim;
  return v + tok * stride + (long)(unit - 2 * Hg) * kDim + 4 * lane;
}

// The decode conv of one (sequence, unit) by one wave: window from the cache at the accepted offset, the step's
// tokens, then the cache rewritten in place (valid domain: state_len_max = 3 + maxq - 1).
__device__ __forceinline__ void conv_update_unit(const u16* __restrict__ x, long xpitch, const u16* __restrict__ wgt,
                                                 const u16* __restrict__ bias, u16* __restrict__ cstate,
                                                 long cs_seq, long cs_dim, long cs_tok, int state_len_max,
                                                 const int* __restrict__ cache_idx, long ci_stride,
                                                 const int* __restrict__ naccept, u16* q, u16* k, u16* v,
                                                 const int* __restrict__ cu, int H, int Hg, int maxq, int n,
                                                 int unit, int lane) {
  const int bos = cu[n];
  const int slen = cu[n + 1] - bos;
  if (slen <= 0) return;
  const int slot = cache_idx[(long)n * ci_stride];
  if (slot <= 0) return;
  const int off = naccept ? naccept[n] - 1 : 0;
  const long d0 = (long)unit * kDim + 4 * lane;
  ConvCoef cf;
  conv_coef(cf, wgt, bias, d0);
  u16* cs = cstate + (long)slot * cs_seq + d0 * cs_dim;
  u32 hist[4][kHist];
  float win[4][kHist];
#pragma unroll
  for (int c = 0; c < 4; ++c)
#pragma unroll
    for (int i = 0; i < kHist; ++i) {
      hist[c][i] = cs[(long)c * cs_dim + (long)(off + i) * cs_tok];
      win[c][i] = bf(hist[c][i]);
    }
  long ostride;
  u16* dst = conv_dst(q, k, v, H, Hg, unit, bos, lane, ostride);
  const u16* xr = x + (long)bos * xpitch + d0;
  for (int t = 0; t < slen; ++t) {
    u32 r[4];
    split4(*(const u32x2*)(xr + (long)t * xpitch), r);
    float o[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const float xc = bf(r[c]);
      o[c] = conv_tap(cf.w[c], cf.b[c], win[c][0], win[c][1], win[c][2], xc);
      win[c][0] = win[c][1];
      win[c][1] = win[c][2];
      win[c][2] = xc;
    }
    u32x2 st;
    st.x = away16x2(o[0], o[1]);
    st.y = away16x2(o[2], o[3]);
    *(u32x2*)(dst + (long)t * ostride) = st;
  }
  // rewrite: entry i <- hist[i + 1] while i + slen < slen_eff, else the raw input row i - 2
  const int slen_eff = state_len_max - (maxq - slen);
  for (int i = 0; i < slen_eff; ++i) {
    u32 val[4];
    if (i + slen < slen_eff) {
#pragma unroll
      for (int c = 0; c < 4; ++c) val[c] = (i == 0) ? hist[c][1] : hist[c][2];
    } else {
      split4(*(const u32x2*)(xr + (long)(i - 2) * xpitch), val);
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) cs[(long)c * cs_dim + (long)i * cs_tok] = (u16)val[c];
  }
}

// ============================================================================ tree conv (spec 2.3.3)
// The window is gathered per row along its parent path: hsrc[n][0..2] name the cache entries of the committed
// window (hist), tsrc[row][i] names tap i's source: < 3 = hist[s], >= 3 = the sequence's input row s - 3.
struct TreeConvArgs {
  const u16* x; long xpitch; const u16* wgt; const u16* bias; u16* cstate; long cs_seq, cs_dim, cs_tok;
  int state_len_max; const int* cache_idx; long ci_stride; const int* hsrc; const int* tsrc; u16* q; u16* k;
  u16* v; const int* cu; int H, Hg, maxq;
};

__device__ __forceinline__ u32 pick3(const u32* hc, int s) { return s == 0 ? hc[0] : (s == 1 ? hc[1] : hc[2]); }

// one tree row t of four channels
__device__ __forceinline__ void tree_conv_row(const TreeConvArgs& a, const ConvCoef& cf, const u32 (*hist)[kHist],
                                              int bos, int t, long d0, u16* dst, long ostride) {
  float win[kHist][4];
#pragma unroll
  for (int i = 0; i < kHist; ++i) {
    const int s = a.tsrc[(long)(bos + t) * 4 + i];
    if (s >= kHist) {
      u32 r[4];
      split4(*(const u32x2*)(a.x + (long)(bos + s - kHist) * a.xpitch + d0), r);
#pragma unroll
      for (int c = 0; c < 4; ++c) win[i][c] = bf(r[c]);
    } else {
#pragma unroll
      for (int c = 0; c < 4; ++c) win[i][c] = bf(pick3(hist[c], s));
    }
  }
  u32 r[4];
  split4(*(const u32x2*)(a.x + (long)(bos + t) * a.xpitch + d0), r);
  float o[4];
#pragma unroll
  for (int c = 0; c < 4; ++c) o[c] = conv_tap(cf.w[c], cf.b[c], win[0][c], win[1][c], win[2][c], bf(r[c]));
  *(u32x2*)(dst + (long)t * ostride) = u32x2{away16x2(o[0], o[1]), away16x2(o[2], o[3])};
}

// cache entry i of four channels after the step (the decode law with the gathered window)
__device__ __forceinline__ void tree_conv_rewrite(const TreeConvArgs& a, const u32 (*hist)[kHist], u16* cs,
                                                  int bos, int slen, int slen_eff, int i, long d0) {
  u32 val[4];
  if (i + slen < slen_eff) {
#pragma unroll
    for (int c = 0; c < 4; ++c) val[c] = (i == 0) ? hist[c][1] : hist[c][2];
  } else {
    split4(*(const u32x2*)(a.x + (long)(bos + i - 2) * a.xpitch + d0), val);
  }
#pragma unroll
  for (int c = 0; c < 4; ++c) cs[(long)c * a.cs_dim + (long)i * a.cs_tok] = (u16)val[c];
}

// rows serially by one wave (the p27_tree_conv form); ROWS_PAR > 0: rows in parallel over ROWS_PAR waves of one
// workgroup, then one workgroup barrier, then the rewrite split over the waves by entry (the p27_tree_rows form)
template <int ROWS_PAR>
__device__ __forceinline__ void tree_conv(const TreeConvArgs& a, int n, int unit, int wave, int lane) {
  const int bos = a.cu[n];
  const int slen = a.cu[n + 1] - bos;
  if (slen <= 0) return;
  const int slot = a.cache_idx[(long)n * a.ci_stride];
  if (slot <= 0) return;
  const long d0 = (long)unit * kDim + 4 * lane;
  ConvCoef cf;
  conv_coef(cf, a.wgt, a.bias, d0);
  u16* cs = a.cstate + (long)slot * a.cs_seq + d0 * a.cs_dim;
  u32 hist[4][kHist];
#pragma unroll
  for (int i = 0; i < kHist; ++i) {
    const int e = a.hsrc[(long)n * 4 + i];
#pragma unroll
    for (int c = 0; c < 4; ++c) hist[c][i] = cs[(long)c * a.cs_dim + (long)e * a.cs_tok];
  }
  long ostride;
  u16* dst = conv_dst(a.q, a.k, a.v, a.H, a.Hg, unit, bos, lane, ostride);
  const int slen_eff = a.state_len_max - (a.maxq - slen);
  if constexpr (ROWS_PAR == 0) {
    for (int t = 0; t < slen; ++t) tree_conv_row(a, cf, hist, bos, t, d0, dst, ostride);
    for (int i = 0; i < slen_eff; ++i) tree_conv_rewrite(a, hist, cs, bos, slen, slen_eff, i, d0);
  } else {
    for (int t = wave; t < slen; t += ROWS_PAR) tree_conv_row(a, cf, hist, bos, t, d0, dst, ostride);
    __syncthreads();
    for (int i = wave; i < slen_eff; i += ROWS_PAR) tree_conv_rewrite(a, hist, cs, bos, slen, slen_eff, i, d0);
  }
}

}  // namespace sgdn
