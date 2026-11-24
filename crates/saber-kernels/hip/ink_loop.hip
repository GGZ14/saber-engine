// Inkling 2 (inkling2_loop) loop / residual / MoE-plumbing bodies, INK3b (docs/INKLING2.md §2-§3,
// §9). Saber's own, written from the architecture as docs/INKLING2.md states it; the numpy twins in
// tools/capture/ink3/ink3b_probe.py are the spec every body here is gated against (bound or byte),
// each with a control that must fail. Correctness-first bodies: the INK4 eager prefill runs on them
// and the fast forms are re-tiles gated against them.
//
//   ink_gemm_f32_bf16       C[M, N] f32 (+= C when acc) = A[M, K] f32 . W[N, K]^T bf16; 64 x 64
//                           tiles, K % 32 == 0, every output a sequential f32 sum over k (the dense
//                           projections, the dynamic-conv taps, the head).
//   ink_depth_attend        attention over depth (§2): S states (an array of S row pointers), nq
//                           queries; keys = values = rms(state) (no gain); softmax over S at D^-1/2;
//                           each query's output optionally gain-RMS-normed (depth_attend_in + in_norm).
//   ink_rmsnorm             x * (1 / sqrt(mean x^2 + eps)) [* gain bf16] (embed / final / in norms).
//   ink_msg_gate            the message gate (§3.6): z = Wd [rms h ; rms m], out = (1 - tanh Wf z) h
//                           + tanh(Wg z) m (Wd [64, 2D], Wf / Wg [D, 64], bf16).
//   ink_sconv_resid         static short conv over D channels with the 3-row state (tap 3 = current,
//                           (sum + x) / sqrt 5) + an optional residual (sconv_resid_mlp_in's first
//                           half, mlp_sconv_resid); the next state = the last 3 inputs.
//   ink_out_gate            o * sigmoid(g) (the attention's output gate, o_proj's A).
//   ink_rel_bias            rel[t, h, d] = r_post[t, h, :] . P[:, d] (P bf16 [16, extent]).
//   ink_route_sort_e128     the routes of one rank's experts expert-major (t-major inside an expert),
//                           16-row tiles (Flash-Next's grouped tile law), pos[t, j] = the row or -1,
//                           row_slot[row] = t * 8 + j or -1 (pad); ONE workgroup, deterministic.
//   ink_act_quant_rows      the W4A8 cells' A producer: e4m3 rows + one f32 scale a row (amax / 448).
//   ink_swiglu_q8           the softcapped SwiGLU over the gate_up cell's bf16 rows fused with the
//                           down cell's A quant (pad rows -> zeros, scale 0).
//   ink_moe_combine         y[t] (+)= sum_j wts[t, j] * C[pos[t, j]] (f32, j in order 0..7).
//   ink_softcap             x = tanh(x / cap) * cap.
//
// Built with -ffp-contract=off: every f32 multiply and add rounds on its own.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

__device__ __forceinline__ float bf2f(unsigned short u) {
  return __builtin_bit_cast(float, ((unsigned)u) << 16);
}

// block-wide sum over 256 lanes (8 waves of 32); red >= 8 floats
__device__ __forceinline__ float bsum256(float v, float* red) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
  __syncthreads();
  if (lane == 0) red[w] = v;
  __syncthreads();
  float s = red[0];
#pragma unroll
  for (int i = 1; i < 8; ++i) s = s + red[i];
  return s;
}
__device__ __forceinline__ float bmax256(float v, float* red) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
  __syncthreads();
  if (lane == 0) red[w] = v;
  __syncthreads();
  float s = red[0];
#pragma unroll
  for (int i = 1; i < 8; ++i) s = fmaxf(s, red[i]);
  return s;
}

// ------------------------------------------------------------------------------------------------
#define GT 64
#define GK 32
extern "C" __global__ __launch_bounds__(256) void ink_gemm_f32_bf16(
    const float* __restrict__ a, int lda, const unsigned short* __restrict__ w, int ldw,
    float* __restrict__ c, int ldc, int M, int N, int K, int acc) {
  __shared__ float as[GK][GT + 1];
  __shared__ float ws[GK][GT + 1];
  const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
  const int m0 = blockIdx.y * GT, n0 = blockIdx.x * GT;
  float s[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) s[i][j] = 0.f;
  for (int k0 = 0; k0 < K; k0 += GK) {
    // stage: 64 rows x 32 k of A and of W, 8 elements a thread each
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int p = tid + i * 256, r = p >> 5, kk = p & 31;
      const int gm = m0 + r, gn = n0 + r;
      as[kk][r] = gm < M ? a[(long)gm * lda + k0 + kk] : 0.f;
      ws[kk][r] = gn < N ? bf2f(w[(long)gn * ldw + k0 + kk]) : 0.f;
    }
    __syncthreads();
#pragma unroll 4
    for (int kk = 0; kk < GK; ++kk) {
      float av[4], wv[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) av[i] = as[kk][ty * 4 + i];
#pragma unroll
      for (int j = 0; j < 4; ++j) wv[j] = ws[kk][tx * 4 + j];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) s[i][j] = s[i][j] + av[i] * wv[j];
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int gm = m0 + ty * 4 + i;
    if (gm >= M) continue;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int gn = n0 + tx * 4 + j;
      if (gn >= N) continue;
      float* p = c + (long)gm * ldc + gn;
      *p = acc ? *p + s[i][j] : s[i][j];
    }
  }
}

// ------------------------------------------------------------------------------------------------
#define DA_D 2304
#define DA_PER 9   // 2304 / 256
#define DA_SMAX 8
#define DA_QMAX 4
struct InkPtrs {
  const float* s[DA_SMAX];          // states, each [T, ld_s] f32
  const float* q[DA_QMAX];          // queries, each [D] f32
  float* o[DA_QMAX];                // outputs, each [T, ld_o] f32
  const unsigned short* g[DA_QMAX]; // optional gains (bf16 [D]); null = the raw attend output
};
extern "C" __global__ __launch_bounds__(256) void ink_depth_attend(InkPtrs P, int S, int nq, int ld_s,
                                                                   int ld_o, float eps) {
  __shared__ float red[8];
  const int t = blockIdx.x, tid = threadIdx.x;
  float k[DA_SMAX][DA_PER];
  float sc[DA_QMAX][DA_SMAX];
  // every loop over states / queries is unrolled to its compile-time cap with a uniform exit, so
  // k[][] and sc[][] stay in registers (a runtime index put them in 432 B of scratch)
#pragma unroll
  for (int si = 0; si < DA_SMAX; ++si) {
    if (si >= S) break;
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < DA_PER; ++i) {
      const float x = P.s[si][(long)t * ld_s + tid + 256 * i];
      k[si][i] = x;
      ss = ss + x * x;
    }
    const float tot = bsum256(ss, red);
    const float r = 1.0f / sqrtf(tot / (float)DA_D + eps);
#pragma unroll
    for (int i = 0; i < DA_PER; ++i) k[si][i] = k[si][i] * r;
#pragma unroll
    for (int qi = 0; qi < DA_QMAX; ++qi) {
      if (qi >= nq) break;
      float d = 0.f;
#pragma unroll
      for (int i = 0; i < DA_PER; ++i) d = d + k[si][i] * P.q[qi][tid + 256 * i];
      sc[qi][si] = bsum256(d, red) * 0.020833333333333332f;   // 2304^-1/2 = 1/48
    }
  }
#pragma unroll
  for (int qi = 0; qi < DA_QMAX; ++qi) {
    if (qi >= nq) break;
    float mx = sc[qi][0];
#pragma unroll
    for (int si = 1; si < DA_SMAX; ++si)
      if (si < S) mx = fmaxf(mx, sc[qi][si]);
    float e[DA_SMAX], den = 0.f;
#pragma unroll
    for (int si = 0; si < DA_SMAX; ++si) {
      if (si >= S) break;
      e[si] = expf(sc[qi][si] - mx);
      den = den + e[si];
    }
    float out[DA_PER];
#pragma unroll
    for (int i = 0; i < DA_PER; ++i) out[i] = 0.f;
#pragma unroll
    for (int si = 0; si < DA_SMAX; ++si) {
      if (si >= S) break;
      const float wgt = e[si] / den;
#pragma unroll
      for (int i = 0; i < DA_PER; ++i) out[i] = out[i] + wgt * k[si][i];
    }
    if (P.g[qi]) {
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < DA_PER; ++i) ss = ss + out[i] * out[i];
      const float r = 1.0f / sqrtf(bsum256(ss, red) / (float)DA_D + eps);
#pragma unroll
      for (int i = 0; i < DA_PER; ++i) out[i] = (out[i] * r) * bf2f(P.g[qi][tid + 256 * i]);
    }
#pragma unroll
    for (int i = 0; i < DA_PER; ++i) P.o[qi][(long)t * ld_o + tid + 256 * i] = out[i];
  }
}

// ------------------------------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256) void ink_rmsnorm(const float* __restrict__ x, int ld_x,
                                                              const unsigned short* __restrict__ gain,
                                                              float* __restrict__ o, int ld_o, int D,
                                                              float eps) {
  __shared__ float red[8];
  const int t = blockIdx.x;
  float ss = 0.f;
  for (int i = threadIdx.x; i < D; i += 256) {
    const float v = x[(long)t * ld_x + i];
    ss = ss + v * v;
  }
  const float r = 1.0f / sqrtf(bsum256(ss, red) / (float)D + eps);
  for (int i = threadIdx.x; i < D; i += 256) {
    float v = x[(long)t * ld_x + i] * r;
    if (gain) v = v * bf2f(gain[i]);
    o[(long)t * ld_o + i] = v;
  }
}

// ------------------------------------------------------------------------------------------------
#define MG_R 64
extern "C" __global__ __launch_bounds__(256) void ink_msg_gate(
    const float* __restrict__ h, int ld_h, const float* __restrict__ m, int ld_m,
    const unsigned short* __restrict__ wd,   // [64, 2D]
    const unsigned short* __restrict__ wf,   // [D, 64]
    const unsigned short* __restrict__ wg,   // [D, 64]
    float* __restrict__ o, int ld_o, float eps) {
  __shared__ float in[2 * DA_D];
  __shared__ float z[MG_R];
  __shared__ float red[8];
  const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wv = tid >> 5;
  float hv[DA_PER], mv[DA_PER], sh = 0.f, sm = 0.f;
#pragma unroll
  for (int i = 0; i < DA_PER; ++i) {
    hv[i] = h[(long)t * ld_h + tid + 256 * i];
    mv[i] = m[(long)t * ld_m + tid + 256 * i];
    sh = sh + hv[i] * hv[i];
    sm = sm + mv[i] * mv[i];
  }
  const float rh = 1.0f / sqrtf(bsum256(sh, red) / (float)DA_D + eps);
  const float rm = 1.0f / sqrtf(bsum256(sm, red) / (float)DA_D + eps);
#pragma unroll
  for (int i = 0; i < DA_PER; ++i) {
    in[tid + 256 * i] = hv[i] * rh;
    in[DA_D + tid + 256 * i] = mv[i] * rm;
  }
  __syncthreads();
  // z: wave wv takes rows wv, wv + 8, ..; a row's 4608 products strided over the 32 lanes
  for (int r = wv; r < MG_R; r += 8) {
    const unsigned short* wr = wd + (long)r * 2 * DA_D;
    float d = 0.f;
    for (int i = lane; i < 2 * DA_D; i += 32) d = d + in[i] * bf2f(wr[i]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) d += __shfl_xor(d, off, 32);
    if (lane == 0) z[r] = d;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < DA_PER; ++i) {
    const int c = tid + 256 * i;
    float f = 0.f, g = 0.f;
    for (int j = 0; j < MG_R; ++j) {
      f = f + z[j] * bf2f(wf[(long)c * MG_R + j]);
      g = g + z[j] * bf2f(wg[(long)c * MG_R + j]);
    }
    o[(long)t * ld_o + c] = (1.0f - tanhf(f)) * hv[i] + tanhf(g) * mv[i];
  }
}

// ------------------------------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256) void ink_sconv_resid(
    const float* __restrict__ x, int ld_x, const unsigned short* __restrict__ w,   // [C, 4]
    const float* __restrict__ st_in,   // [3, C], oldest first
    float* __restrict__ st_out,        // [3, C] (may be null)
    const float* __restrict__ resid, int ld_r,   // may be null
    float* __restrict__ o, int ld_o, int T, int C) {
  const int c = blockIdx.x * 256 + threadIdx.x;
  if (c >= C) return;
  const int t0 = blockIdx.y * 64, t1 = min(T, t0 + 64);
  float wt[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) wt[j] = bf2f(w[(long)c * 4 + j]);
  for (int t = t0; t < t1; ++t) {
    // ctx[t + j] for j = 0..3 = x_{t - 3 + j}; index < 0 reads the state row 3 + (t - 3 + j)
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int s = t - 3 + j;
      const float v = s >= 0 ? x[(long)s * ld_x + c] : st_in[(long)(3 + s) * C + c];
      acc = acc + v * wt[j];
    }
    float y = (acc + x[(long)t * ld_x + c]) / 2.2360679774997898f;
    if (resid) y = y + resid[(long)t * ld_r + c];
    o[(long)t * ld_o + c] = y;
  }
  if (st_out && blockIdx.y == 0) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const int s = T - 3 + j;
      st_out[(long)j * C + c] = s >= 0 ? x[(long)s * ld_x + c] : st_in[(long)(3 + s) * C + c];
    }
  }
}

// ------------------------------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256) void ink_out_gate(const float* __restrict__ o, int ld_o,
                                                               const float* __restrict__ g, int ld_g,
                                                               float* __restrict__ y, int ld_y, int T,
                                                               int N) {
  const long n = (long)T * N;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < n; i += (long)gridDim.x * 256) {
    const long t = i / N, c = i % N;
    const float sg = 1.0f / (1.0f + expf(-g[t * ld_g + c]));
    y[t * ld_y + c] = o[t * ld_o + c] * sg;
  }
}

// ------------------------------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256) void ink_rel_bias(const float* __restrict__ r, int ld_r,
                                                               const unsigned short* __restrict__ P,
                                                               float* __restrict__ rel, int ld_rel,
                                                               int extent) {
  __shared__ float rv[16];
  const int t = blockIdx.x, h = blockIdx.y;
  if (threadIdx.x < 16) rv[threadIdx.x] = r[(long)t * ld_r + h * 16 + threadIdx.x];
  __syncthreads();
  for (int d = threadIdx.x; d < extent; d += 256) {
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < 16; ++k) s = s + rv[k] * bf2f(P[(long)k * extent + d]);
    rel[(long)t * ld_rel + (long)h * extent + d] = s;
  }
}

// ------------------------------------------------------------------------------------------------
#define RS_E 128
extern "C" __global__ __launch_bounds__(256) void ink_route_sort_e128(
    const int* __restrict__ ids, int T, int lo, int nloc,
    int* __restrict__ row_tok, int* __restrict__ row_slot, int* __restrict__ tile_e,
    int* __restrict__ tile_r0, int* __restrict__ tile_n, int* __restrict__ meta,
    int* __restrict__ pos) {
  __shared__ int cnt[RS_E], seg[RS_E], run[RS_E], tbase[RS_E];
  __shared__ int ce[256];
  const int tid = threadIdx.x, N = T * 8;
  if (tid < RS_E) { cnt[tid] = 0; run[tid] = 0; }
  __syncthreads();
  for (int i = tid; i < N; i += 256) {
    const int e = ids[i] - lo;
    if (e >= 0 && e < nloc) atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  if (tid == 0) {
    int r = 0, tb = 0;
    for (int e = 0; e < nloc; ++e) {
      seg[e] = r;
      tbase[e] = tb;
      const int nt = (cnt[e] + 15) >> 4;
      r += nt * 16;
      tb += nt;
    }
    meta[0] = tb;
    meta[1] = r;
  }
  __syncthreads();
  // tiles + pad rows
  for (int e = 0; e < nloc; ++e) {
    const int nt = (cnt[e] + 15) >> 4;
    for (int k = tid; k < nt; k += 256) {
      tile_e[tbase[e] + k] = e;
      tile_r0[tbase[e] + k] = seg[e] + 16 * k;
      tile_n[tbase[e] + k] = min(16, cnt[e] - 16 * k);
    }
  }
  // ranks, 256 entries a round in entry order (t-major): rank = run[e] + earlier lanes of e
  for (int b = 0; b < N; b += 256) {
    const int i = b + tid;
    int e = -1;
    if (i < N) {
      e = ids[i] - lo;
      if (e < 0 || e >= nloc) e = -1;
    }
    ce[tid] = e;
    __syncthreads();
    int rk = 0, last = 1;
    if (e >= 0) {
      for (int j = 0; j < tid; ++j) rk += ce[j] == e;
      for (int j = tid + 1; j < 256; ++j) last &= ce[j] != e;
      const int row = seg[e] + run[e] + rk;
      row_tok[row] = i >> 3;
      row_slot[row] = i;
      pos[i] = row;
    } else if (i < N) {
      pos[i] = -1;
    }
    __syncthreads();
    if (e >= 0 && last) run[e] += rk + 1;
    __syncthreads();
  }
  // pad rows: the tile's last live token, slot -1
  for (int e = 0; e < nloc; ++e) {
    const int c = cnt[e], padded = ((c + 15) >> 4) * 16;
    for (int k = c + tid; k < padded; k += 256) {
      row_tok[seg[e] + k] = row_tok[seg[e] + c - 1];
      row_slot[seg[e] + k] = -1;
    }
    __syncthreads();
  }
}

// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ unsigned char f2e4m3(float v) {
  // OCP e4m3 (fn), RNE, the card's convert (saturating at 448)
  const int p = __builtin_amdgcn_cvt_pk_fp8_f32(v, 0.f, 0, false);
  return (unsigned char)(p & 0xFF);
}

extern "C" __global__ __launch_bounds__(256) void ink_act_quant_rows(const float* __restrict__ x, int ld_x,
                                                                     unsigned char* __restrict__ q, int ld_q,
                                                                     float* __restrict__ scale, int K) {
  __shared__ float red[8];
  const int t = blockIdx.x;
  float mx = 0.f;
  for (int i = threadIdx.x; i < K; i += 256) mx = fmaxf(mx, fabsf(x[(long)t * ld_x + i]));
  mx = bmax256(mx, red);
  const float s = fmaxf(mx / 448.0f, 1e-30f);
  for (int i = threadIdx.x; i < K; i += 256) q[(long)t * ld_q + i] = f2e4m3(x[(long)t * ld_x + i] / s);
  if (threadIdx.x == 0) scale[t] = s;
}

extern "C" __global__ __launch_bounds__(256) void ink_swiglu_q8(
    const unsigned short* __restrict__ gu, int ld_gu,   // [rows, 2 I] bf16 (gate, then up)
    const int* __restrict__ row_slot, const int* __restrict__ meta,
    float* __restrict__ act, int ld_act,                // optional f32 copy (the gate's probe)
    unsigned char* __restrict__ q, int ld_q, float* __restrict__ scale, int I, float gc, float uc) {
  __shared__ float red[8];
  __shared__ float av[2048];
  const int r = blockIdx.x;
  if (r >= meta[1]) return;
  const int live = row_slot[r] >= 0;
  float mx = 0.f;
  for (int c = threadIdx.x; c < I; c += 256) {
    float a = 0.f;
    if (live) {
      const float g = bf2f(gu[(long)r * ld_gu + c]), up = bf2f(gu[(long)r * ld_gu + I + c]);
      const float sg = 1.0f / (1.0f + expf(-g));
      a = ((tanhf(g / gc) * gc) * sg) * (tanhf(up / uc) * uc);
    }
    av[c] = a;
    if (act) act[(long)r * ld_act + c] = a;
    mx = fmaxf(mx, fabsf(a));
  }
  mx = bmax256(mx, red);
  const float s = live ? fmaxf(mx / 448.0f, 1e-30f) : 0.f;
  for (int c = threadIdx.x; c < I; c += 256) q[(long)r * ld_q + c] = live ? f2e4m3(av[c] / s) : 0;
  if (threadIdx.x == 0) scale[r] = s;
}

extern "C" __global__ __launch_bounds__(256) void ink_moe_combine(
    const unsigned short* __restrict__ C, int ld_c,   // [rows, D] bf16, expert-major
    const int* __restrict__ pos, const float* __restrict__ wts, float* __restrict__ y, int ld_y, int D,
    int acc) {
  const int t = blockIdx.x;
  int p[8];
  float wt[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    p[j] = pos[t * 8 + j];
    wt[j] = wts[t * 8 + j];
  }
  for (int c = threadIdx.x; c < D; c += 256) {
    float s = acc ? y[(long)t * ld_y + c] : 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (p[j] >= 0) s = s + wt[j] * bf2f(C[(long)p[j] * ld_c + c]);
    y[(long)t * ld_y + c] = s;
  }
}

extern "C" __global__ __launch_bounds__(256) void ink_softcap(float* __restrict__ x, long n, float cap) {
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < n; i += (long)gridDim.x * 256)
    x[i] = tanhf(x[i] / cap) * cap;
}
