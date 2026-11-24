// Inkling 2 (inkling2_loop) fused decode bodies, INK11 (docs/INKLING2.md section 9 INK11). Saber's
// own. Its own object so ink_fuse keeps its bytes (INK10's `all` = the INK11 control runs the INK10
// code object). The helpers below are ink_fuse.hip's (verbatim).
//
// Built with -ffp-contract=off (saber-kbuild's `ink_` row).
#include <hip/hip_runtime.h>

__device__ __forceinline__ float lo16(unsigned w) { return __builtin_bit_cast(float, w << 16); }
__device__ __forceinline__ float hi16(unsigned w) {
  return __builtin_bit_cast(float, w & 0xFFFF0000u);
}
__device__ __forceinline__ float fbf(unsigned short u) {
  return __builtin_bit_cast(float, ((unsigned)u) << 16);
}

#define GV_N 64
#define GV_K 256
#define GV_M 8
#define GV_LDW (GV_K / 2 + 1)
#define GV_LDS_BARRIER()                                                    \
  do {                                                                      \
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");         \
    __builtin_amdgcn_s_barrier();                                           \
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");         \
  } while (0)

template <int MM>
__device__ __forceinline__ void gv_step(const uint4 (&p)[8], const float (&ap)[MM], unsigned* ws,
                                        float (*as)[GV_K], float (&s)[GV_M], int tid) {
  GV_LDS_BARRIER();
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int e = tid + 256 * i, r = e >> 5, u = e & 31;
    unsigned* d = ws + r * GV_LDW + u * 4;
    d[0] = p[i].x;
    d[1] = p[i].y;
    d[2] = p[i].z;
    d[3] = p[i].w;
  }
#pragma unroll
  for (int m = 0; m < MM; ++m) as[m][tid] = ap[m];
  GV_LDS_BARRIER();
}

template <int MM>
__device__ __forceinline__ void gv_sum(const unsigned* ws, float (*as)[GV_K], float (&s)[GV_M],
                                       int tid) {
  if (tid < GV_N) {
    const unsigned* row = ws + tid * GV_LDW;
#pragma unroll 16
    for (int kk = 0; kk < GV_K; kk += 2) {
      const unsigned wd = row[kk >> 1];
      const float w0 = lo16(wd), w1 = hi16(wd);
#pragma unroll
      for (int m = 0; m < MM; ++m) s[m] = s[m] + as[m][kk] * w0;
#pragma unroll
      for (int m = 0; m < MM; ++m) s[m] = s[m] + as[m][kk + 1] * w1;
    }
  }
}

#define DA_D 2304
#define DA_PER 9
#define DA_SMAX 8
#define DA_QMAX 4
struct InkPtrs {
  const float* s[DA_SMAX];
  const float* q[DA_QMAX];
  float* o[DA_QMAX];
  const unsigned short* g[DA_QMAX];
};
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


// ================================================================================================
// INK11 (docs/INKLING2.md section 9 INK11): three more launch folds, each a fusion of byte-equal
// parents (every value the parent's expression in the parent's order).
//
//   ink_depth_mq     ink_loop's ink_depth_attend (verbatim) for 4 x gridDim.y queries over ONE
//                    state list: workgroup (t, g) runs the parent's nq = 4 body for queries
//                    4 g .. 4 g + 3 (q_j = q0 + j ldq, o_j = o0 + j o_stride). A pass's 48 messages
//                    (12 core layers x q / k / v / mlp, all over [*P_blocks, P_x]) in ONE launch
//                    instead of 12.
//   ink_gemv3_f32_bf16  ink_fast's ink_gemv_f32_bf16 over up to THREE (A, W, C, N) segments in one
//                    grid (a workgroup's 64 columns lie in one segment; each output ONE sequential
//                    f32 sum over k in the parent's order): a visit's q / k / v GEMVs of three
//                    distinct input rows in one launch.
//   ink_attn_kvc     ink_glue's ink_kv_conv_post_append at ONE row a sequence (decode / batched
//                    decode rows) folded into ink_fast's ink_attn_bias_sink_v1: workgroup (t, h)
//                    forms its q head's post (scale, l2 clip, tau -> bf16) and r post (clip, tau),
//                    and its KV head's k / v conv rows for the current token (static or dynamic
//                    taps, the state rows of slot row_seq[t], then the k clip / v RMS -> bf16), all
//                    in the parent's lane / butterfly order; the key / value at j == pos come from
//                    those registers (LDS), every older row from the pools. The workgroup of the
//                    KV head's FIRST q head (h % (H / KV) == 0) writes the pool row and the next
//                    conv state. THE STATE PARITY LAW: a KV conv state has two buffers, the state
//                    "before position p" lives in buffer p & 1 -- a row at pos reads st[pos & 1]
//                    and writes st[(pos + 1) & 1], so the 3 q heads of a KV head never read a state
//                    another workgroup is writing (the host's kv_conv chunks follow the same law:
//                    in = st[pos0 & 1], out = st[(pos0 + t) & 1]).
struct InkStates {
  const float* s[DA_SMAX];
};
extern "C" __global__ __launch_bounds__(256) void ink_depth_mq(InkStates SS, int S, const float* q0,
                                                               int ldq, float* o0, long o_stride,
                                                               int ld_s, int ld_o, float eps) {
  __shared__ float red[8];
  const int t = blockIdx.x, tid = threadIdx.x, g = blockIdx.y;
  const int nq = DA_QMAX;
  InkPtrs P;
#pragma unroll
  for (int i = 0; i < DA_SMAX; ++i) P.s[i] = SS.s[i];
#pragma unroll
  for (int i = 0; i < DA_QMAX; ++i) {
    P.q[i] = q0 + (long)(4 * g + i) * ldq;
    P.o[i] = o0 + (long)(4 * g + i) * o_stride;
    P.g[i] = nullptr;
  }
  float k[DA_SMAX][DA_PER];
  float sc[DA_QMAX][DA_SMAX];
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
      sc[qi][si] = bsum256(d, red) * 0.020833333333333332f;
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
#pragma unroll
    for (int i = 0; i < DA_PER; ++i) P.o[qi][(long)t * ld_o + tid + 256 * i] = out[i];
  }
}

// ------------------------------------------------------------------------------------------------
template <int MM>
__device__ __forceinline__ void g3_load(uint4 (&p)[8], float (&ap)[MM], const float* __restrict__ a,
                                        int lda, const unsigned short* __restrict__ w, int ldw,
                                        int n0, int N, int k0, int tid) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int e = tid + 256 * i, r = e >> 5, u = e & 31;
    const int gn = n0 + r;
    p[i] = gn < N ? *reinterpret_cast<const uint4*>(w + (long)gn * ldw + k0 + u * 8)
                  : make_uint4(0u, 0u, 0u, 0u);
  }
#pragma unroll
  for (int m = 0; m < MM; ++m) ap[m] = a[(long)m * lda + k0 + tid];
}
template <int MM>
__device__ __forceinline__ void g3_body(const float* __restrict__ a, int lda,
                                        const unsigned short* __restrict__ w, int ldw,
                                        float* __restrict__ c, int ldc, int N, int K, int n0,
                                        unsigned* ws, float (*as)[GV_K]) {
  const int tid = threadIdx.x;
  uint4 pa[8], pb[8];
  float aa[MM], ab[MM];
  float s[GV_M];
#pragma unroll
  for (int m = 0; m < GV_M; ++m) s[m] = 0.f;
  g3_load<MM>(pa, aa, a, lda, w, ldw, n0, N, 0, tid);
  if (GV_K < K) g3_load<MM>(pb, ab, a, lda, w, ldw, n0, N, GV_K, tid);
  for (int k0 = 0; k0 < K; k0 += 2 * GV_K) {
    gv_step<MM>(pa, aa, ws, as, s, tid);
    if (k0 + 2 * GV_K < K) g3_load<MM>(pa, aa, a, lda, w, ldw, n0, N, k0 + 2 * GV_K, tid);
    gv_sum<MM>(ws, as, s, tid);
    if (k0 + GV_K < K) {
      gv_step<MM>(pb, ab, ws, as, s, tid);
      if (k0 + 3 * GV_K < K) g3_load<MM>(pb, ab, a, lda, w, ldw, n0, N, k0 + 3 * GV_K, tid);
      gv_sum<MM>(ws, as, s, tid);
    }
  }
  const int gn = n0 + tid;
  if (tid < GV_N && gn < N) {
#pragma unroll
    for (int m = 0; m < MM; ++m) c[(long)m * ldc + gn] = s[m];
  }
}
struct Gemv3 {
  const float* a[3];
  const unsigned short* w[3];
  float* c[3];
  int n[3];
  int ldc[3];
};
extern "C" __global__ __launch_bounds__(256) void ink_gemv3_f32_bf16(Gemv3 G, int lda, int M,
                                                                     int K) {
  __shared__ unsigned ws[GV_N * GV_LDW];
  __shared__ float as[GV_M][GV_K];
  // the segment of this workgroup (uniform)
  int b = blockIdx.x, sg = 0;
  const int t0 = (G.n[0] + GV_N - 1) / GV_N, t1 = (G.n[1] + GV_N - 1) / GV_N;
  if (b >= t0) {
    b -= t0;
    sg = 1;
    if (b >= t1) {
      b -= t1;
      sg = 2;
    }
  }
  const float* a = sg == 0 ? G.a[0] : (sg == 1 ? G.a[1] : G.a[2]);
  const unsigned short* w = sg == 0 ? G.w[0] : (sg == 1 ? G.w[1] : G.w[2]);
  float* c = sg == 0 ? G.c[0] : (sg == 1 ? G.c[1] : G.c[2]);
  const int N = sg == 0 ? G.n[0] : (sg == 1 ? G.n[1] : G.n[2]);
  const int ldc = sg == 0 ? G.ldc[0] : (sg == 1 ? G.ldc[1] : G.ldc[2]);
  const int n0 = b * GV_N;
  switch (M) {
    case 1: g3_body<1>(a, lda, w, K, c, ldc, N, K, n0, ws, as); break;
    case 2: g3_body<2>(a, lda, w, K, c, ldc, N, K, n0, ws, as); break;
    case 3: g3_body<3>(a, lda, w, K, c, ldc, N, K, n0, ws, as); break;
    case 4: g3_body<4>(a, lda, w, K, c, ldc, N, K, n0, ws, as); break;
    case 5: g3_body<5>(a, lda, w, K, c, ldc, N, K, n0, ws, as); break;
    case 6: g3_body<6>(a, lda, w, K, c, ldc, N, K, n0, ws, as); break;
    case 7: g3_body<7>(a, lda, w, K, c, ldc, N, K, n0, ws, as); break;
    default: g3_body<8>(a, lda, w, K, c, ldc, N, K, n0, ws, as); break;
  }
}

// ------------------------------------------------------------------------------------------------
// ink_fast's attention constants and block reductions (fa_max / fa_sum, verbatim)
#define FA_HD_K 128
#define FA_PAGE_K 64
#define FA_TILE_K 256
__device__ __forceinline__ float fk_max(float v, float* red) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
  if (lane == 0) red[w] = v;
  __syncthreads();
  float m = red[0];
#pragma unroll
  for (int i = 1; i < FA_TILE_K / 32; ++i) m = fmaxf(m, red[i]);
  __syncthreads();
  return m;
}
__device__ __forceinline__ float fk_sum(float v, float* red) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
  if (lane == 0) red[w] = v;
  __syncthreads();
  float s = red[0];
#pragma unroll
  for (int i = 1; i < FA_TILE_K / 32; ++i) s += red[i];
  __syncthreads();
  return s;
}
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ unsigned short kc_f2bf(float x) {
  return __builtin_bit_cast(unsigned short, (__bf16)x);   // RNE (ink_glue's f2bf)
}
__device__ __forceinline__ float kc_tau(int pos, int global_layer, float n_floor, float alpha) {
  if (!global_layer) return 1.0f;
  const float p = (float)(pos + 1);
  return 1.0f + alpha * logf(fmaxf(p / n_floor, 1.0f));
}
// ink_glue's sum128 for ONE row over a 4-wave group (waves wb .. wb + 3): the xor butterfly a wave,
// then ((r0 + r1) + r2) + r3; every thread of the workgroup reaches the barriers
__device__ __forceinline__ float kc_sum128(float v, float* red4, int in_group) {
  const int lane = threadIdx.x & 31, w = (threadIdx.x >> 5) & 3;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
  if (in_group && lane == 0) red4[w] = v;
  __syncthreads();
  const float s = ((red4[0] + red4[1]) + red4[2]) + red4[3];
  __syncthreads();
  return s;
}

extern "C" __global__ __launch_bounds__(256) void ink_attn_kvc(
    // projection rows (f32)
    const float* __restrict__ kraw, const float* __restrict__ vraw, int ld_kv,
    const float* __restrict__ qraw, int ld_q, const float* __restrict__ rraw, int ld_r,
    const float* __restrict__ dyn_k, const float* __restrict__ dyn_v, int ld_dyn,
    const unsigned short* __restrict__ wk, const unsigned short* __restrict__ wv, int c_stride,
    int lag0, int lag_stride, const unsigned short* __restrict__ q_scale,
    const unsigned short* __restrict__ rel_scale,
    // the conv states by parity: [slots][3][C] f32 each
    float* st_k0, float* st_k1, float* st_v0, float* st_v1,
    unsigned short* k_pool, unsigned short* v_pool, const int* __restrict__ table, int table_ld,
    const int* __restrict__ row_seq, const int* __restrict__ row_pos,
    const unsigned short* __restrict__ P, const float* __restrict__ sink, int ld_sink,
    float* __restrict__ o, int ld_o, int H, int KV, int extent, int window, int global_layer,
    float qk_clip, float rel_clip, float n_floor, float alpha) {
  __shared__ float qs[FA_HD_K];
  __shared__ float ps[FA_TILE_K];
  __shared__ float red[FA_TILE_K / 32];
  __shared__ float oacc[2][FA_HD_K];
  __shared__ float rv[16];
  __shared__ float vcur[FA_HD_K];
  __shared__ __attribute__((aligned(16))) unsigned short kcb[FA_HD_K];   // the current key, bf16
  __shared__ float redk[4], redv[4], redq[4];
  const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
  const int kvh = h / (H / KV);
  const int C = KV * FA_HD_K;
  const int sq = row_seq[t], pos = row_pos[t];
  const int writer = (h % (H / KV)) == 0;
  // 1. the KV head's k (waves 0..3) and v (waves 4..7) conv rows (ink_kv_conv_post_append at one
  //    row of its sequence: have = 0, every lag > 0 from the state)
  {
    const int is_v = tid >= FA_HD_K;
    const int c0 = tid & (FA_HD_K - 1);
    const int c = kvh * FA_HD_K + c0;
    const float* raw = is_v ? vraw : kraw;
    const unsigned short* w = is_v ? wv : wk;
    const float* dyn = is_v ? dyn_v : dyn_k;
    const float* sin_ = is_v ? ((pos & 1) ? st_v1 : st_v0) : ((pos & 1) ? st_k1 : st_k0);
    float* sout = is_v ? ((pos & 1) ? st_v0 : st_v1) : ((pos & 1) ? st_k0 : st_k1);
    float tap[4];
#pragma unroll
    for (int L = 0; L < 4; ++L) tap[L] = fbf(w[(long)c * c_stride + lag0 + L * lag_stride]);
    const float x0 = raw[(long)t * ld_kv + c];
    float acc = 0.f;
#pragma unroll
    for (int L = 3; L >= 0; --L) {
      const float xv = L == 0 ? x0 : sin_[((long)sq * 3 + (3 - L)) * C + c];
      float wl = tap[L];
      if (global_layer) wl = wl + dyn[(long)t * ld_dyn + kvh * 4 + L];
      acc = acc + xv * wl;
    }
    const float v = global_layer ? (acc + x0) / 3.0f : (acc + x0) / 2.2360679774997898f;
    // the two groups reduce in their own LDS words (each is ink_glue's sum128 of its 4 waves)
    float s = v * v;
    const int lane = tid & 31, w4 = (tid >> 5) & 3;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor(s, off, 32);
    if (lane == 0) (is_v ? redv : redk)[w4] = s;
    // the next state's rows are read before anyone writes (the writer writes the OTHER parity)
    const float n0 = sin_[((long)sq * 3 + 1) * C + c], n1 = sin_[((long)sq * 3 + 2) * C + c];
    __syncthreads();
    const float* rr = is_v ? redv : redk;
    const float ss = ((rr[0] + rr[1]) + rr[2]) + rr[3];
    float ov;
    if (is_v) {
      ov = v * (1.0f / sqrtf(ss / 128.0f + 1e-6f));
    } else {
      const float nrm = sqrtf(ss);
      ov = v * fminf(qk_clip / (nrm + 1e-6f), 1.0f);
    }
    const unsigned short bf = kc_f2bf(ov);
    if (is_v) {
      vcur[c0] = fbf(bf);
    } else {
      kcb[c0] = bf;
    }
    if (writer) {
      const long prow = (long)table[(long)sq * table_ld + pos / FA_PAGE_K] * FA_PAGE_K + pos % FA_PAGE_K;
      (is_v ? v_pool : k_pool)[prow * C + c] = bf;
      sout[((long)sq * 3 + 0) * C + c] = n0;
      sout[((long)sq * 3 + 1) * C + c] = n1;
      sout[((long)sq * 3 + 2) * C + c] = x0;
    }
  }
  // 2. the q head's post (waves 0..3) and its r post (wave 0, after)
  {
    const int c0 = tid & (FA_HD_K - 1);
    float qv = 0.f;
    if (tid < FA_HD_K) qv = qraw[(long)t * ld_q + h * FA_HD_K + c0] * fbf(q_scale[h * FA_HD_K + c0]);
    const float ss = kc_sum128(qv * qv, redq, tid < FA_HD_K);
    if (tid < FA_HD_K) {
      const float tau = kc_tau(pos, global_layer, n_floor, alpha);
      float ov = qv * fminf(qk_clip / (sqrtf(ss) + 1e-6f), 1.0f);
      if (global_layer) ov = ov * tau;
      qs[c0] = fbf(kc_f2bf(ov));
    }
    if (tid < 32) {
      const int d = tid & 15;
      const float rs = fbf(rel_scale[h * 16 + d]);
      const float r0 = rraw[(long)t * ld_r + h * 16 + d] * rs;
      float s = r0 * r0;
#pragma unroll
      for (int off = 8; off > 0; off >>= 1) s += __shfl_xor(s, off, 16);
      float ov = r0 * fminf(rel_clip / (sqrtf(s) + 1e-6f), 1.0f);
      if (global_layer) ov = ov * kc_tau(pos, 1, n_floor, alpha);
      if (tid < 16) rv[tid] = ov;
    }
  }
  __syncthreads();
  // 3. ink_attn_bias_sink_v1 (the row at j == pos from kcur / vcur)
  const int lo = window > 0 ? max(0, pos - window + 1) : 0;
  float m = -INFINITY, l = 0.f, acc = 0.f;
  const int d = tid & (FA_HD_K - 1), half = tid >> 7;
  for (int base = lo; base <= pos; base += FA_TILE_K) {
    const int j = base + tid;
    float s = -INFINITY;
    if (j <= pos) {
      float dot = 0.f;
      uint4 kv4[16];
      if (j == pos) {
        const uint4* kr = reinterpret_cast<const uint4*>(kcb);
#pragma unroll
        for (int i = 0; i < 16; ++i) kv4[i] = kr[i];
      } else {
        const long row = (long)table[(long)sq * table_ld + j / FA_PAGE_K] * FA_PAGE_K + j % FA_PAGE_K;
        const uint4* kr = reinterpret_cast<const uint4*>(k_pool + row * C + kvh * FA_HD_K);
#pragma unroll
        for (int i = 0; i < 16; ++i) kv4[i] = kr[i];
      }
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const unsigned wv4[4] = {kv4[i].x, kv4[i].y, kv4[i].z, kv4[i].w};
#pragma unroll
        for (int qd = 0; qd < 4; ++qd) {
          dot = dot + qs[8 * i + 2 * qd] * lo16(wv4[qd]);
          dot = dot + qs[8 * i + 2 * qd + 1] * hi16(wv4[qd]);
        }
      }
      s = dot * 0.08838834764831845f;
      const int dist = pos - j;
      if (dist < extent) {
        float rb = 0.f;
#pragma unroll
        for (int k = 0; k < 16; ++k) rb = rb + rv[k] * fbf(P[(long)k * extent + dist]);
        s = s + rb;
      }
    }
    const float tm = fk_max(s, red);
    const float mn = fmaxf(m, tm);
    const float p = (j <= pos) ? expf(s - mn) : 0.f;
    ps[tid] = p;
    const float corr = (m == -INFINITY) ? 0.f : expf(m - mn);
    const float ts = fk_sum(p, red);
    l = l * corr + ts;
    float a = 0.f;
    const int jn = min(FA_TILE_K / 2, pos - base - half * (FA_TILE_K / 2) + 1);
    const int j0 = base + half * (FA_TILE_K / 2);
    int i = 0;
    for (; i + 8 <= jn; i += 8) {
      float vv[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const int jj = j0 + i + u;
        if (jj == pos) {
          vv[u] = vcur[d];
        } else {
          const long row = (long)table[(long)sq * table_ld + jj / FA_PAGE_K] * FA_PAGE_K + jj % FA_PAGE_K;
          vv[u] = fbf(v_pool[row * C + kvh * FA_HD_K + d]);
        }
      }
#pragma unroll
      for (int u = 0; u < 8; ++u) a = a + ps[half * (FA_TILE_K / 2) + i + u] * vv[u];
    }
    for (; i < jn; ++i) {
      const int jj = j0 + i;
      float vx;
      if (jj == pos) {
        vx = vcur[d];
      } else {
        const long row = (long)table[(long)sq * table_ld + jj / FA_PAGE_K] * FA_PAGE_K + jj % FA_PAGE_K;
        vx = fbf(v_pool[row * C + kvh * FA_HD_K + d]);
      }
      a = a + ps[half * (FA_TILE_K / 2) + i] * vx;
    }
    acc = acc * corr + a;
    m = mn;
    __syncthreads();
  }
  const float sk = sink[(long)t * ld_sink + h];
  const float mf = fmaxf(m, sk);
  const float cc = expf(m - mf);
  const float lf = l * cc + expf(sk - mf);
  oacc[half][d] = acc;
  __syncthreads();
  if (tid < FA_HD_K) o[(long)t * ld_o + h * FA_HD_K + d] = ((oacc[0][d] + oacc[1][d]) * cc) / lf;
}

// ================================================================================================
// INK11 serve: the sampler's device seam (saber_serve::sampler::DeviceRow) over a decode step's
// gathered logits rows -- the K = 64 largest raw logits of the served vocabulary in the argmax
// kernel's total order (value descending, id ascending on a tie) + the row's log-sum-exp at T = 1
// and at the row's own temperature. Two launches:
//   ink_seam_p1  grid [SEAM_NCH, rows], block 256: chunk c of row r (ceil(ncols / SEAM_NCH)
//                columns, <= 4,096) sorted in LDS (bitonic, the total order; pad = -inf / id max),
//                its top 64 + (max, sum exp(x - max)) at T = 1 and at T -> part[r][c]
//   ink_seam_p2  grid [rows], block 1024: the SEAM_NCH x 64 candidates sorted -> the row's top 64;
//                lse = M + log(sum_c s_c exp(m_c - M)) in chunk order (both temperatures)
// out[r] = [64 f32 values | 64 i32 ids | lse | lse_t | 0 0] (132 words a row). The seam's numbers are
// NOT the dense sampler's sums bit for bit (a log-sum-exp in another order): only the DRAW law is
// exact inside the seam (top_k <= 64 cuts above the tail; greedy).
#define SEAM_K 64
#define SEAM_NCH 64
#define SEAM_SORT 4096
#define SEAM_W 132
__device__ __forceinline__ bool seam_before(float va, int ia, float vb, int ib) {
  return va > vb || (va == vb && ia < ib);
}
// bitonic sort of SEAM_SORT (value, id) pairs into the total order, nt threads
__device__ __forceinline__ void seam_sort(float* v, int* id, int nt) {
  for (int k = 2; k <= SEAM_SORT; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = threadIdx.x; i < SEAM_SORT; i += nt) {
        const int p = i ^ j;
        if (p > i) {
          const bool up = (i & k) == 0;   // this run sorts into the total order
          const float a = v[i], b = v[p];
          const int ia = id[i], ib = id[p];
          const bool sw = up ? seam_before(b, ib, a, ia) : seam_before(a, ia, b, ib);
          if (sw) {
            v[i] = b;
            v[p] = a;
            id[i] = ib;
            id[p] = ia;
          }
        }
      }
      __syncthreads();
    }
  }
}
extern "C" __global__ __launch_bounds__(256) void ink_seam_p1(const float* __restrict__ lg, long ld,
                                                              int ncols,
                                                              const float* __restrict__ temp,
                                                              float* __restrict__ part) {
  __shared__ float v[SEAM_SORT];
  __shared__ int id[SEAM_SORT];
  __shared__ float red[4][8];
  const int c = blockIdx.x, r = blockIdx.y, tid = threadIdx.x;
  const int per = (ncols + SEAM_NCH - 1) / SEAM_NCH;
  const int c0 = c * per, c1 = min(ncols, c0 + per);
  const float* row = lg + (long)r * ld;
  const float T = temp[r];
  float m = -INFINITY, mt = -INFINITY;
  for (int i = tid; i < SEAM_SORT; i += 256) {
    const int col = c0 + i;
    const float x = col < c1 ? row[col] : -INFINITY;
    v[i] = x;
    id[i] = col < c1 ? col : 0x7FFFFFFF;
    m = fmaxf(m, x);
    if (col < c1) mt = fmaxf(mt, x / T);
  }
  // block max (both)
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    m = fmaxf(m, __shfl_xor(m, o, 32));
    mt = fmaxf(mt, __shfl_xor(mt, o, 32));
  }
  if ((tid & 31) == 0) {
    red[0][tid >> 5] = m;
    red[1][tid >> 5] = mt;
  }
  __syncthreads();
  m = red[0][0];
  mt = red[1][0];
  for (int i = 1; i < 8; ++i) {
    m = fmaxf(m, red[0][i]);
    mt = fmaxf(mt, red[1][i]);
  }
  float s = 0.f, st = 0.f;
  for (int i = tid; i < SEAM_SORT; i += 256) {
    const int col = c0 + i;
    if (col < c1) {
      s += expf(v[i] - m);
      st += expf(v[i] / T - mt);
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    s += __shfl_xor(s, o, 32);
    st += __shfl_xor(st, o, 32);
  }
  __syncthreads();
  if ((tid & 31) == 0) {
    red[2][tid >> 5] = s;
    red[3][tid >> 5] = st;
  }
  __syncthreads();
  seam_sort(v, id, 256);
  float* o = part + ((long)r * SEAM_NCH + c) * (2 * SEAM_K + 4);
  if (tid < SEAM_K) {
    o[tid] = v[tid];
    o[SEAM_K + tid] = __builtin_bit_cast(float, id[tid]);
  }
  if (tid == 0) {
    float ss = 0.f, sst = 0.f;
    for (int i = 0; i < 8; ++i) {
      ss += red[2][i];
      sst += red[3][i];
    }
    o[2 * SEAM_K + 0] = m;
    o[2 * SEAM_K + 1] = ss;
    o[2 * SEAM_K + 2] = mt;
    o[2 * SEAM_K + 3] = sst;
  }
}
extern "C" __global__ __launch_bounds__(1024) void ink_seam_p2(const float* __restrict__ part,
                                                               float* __restrict__ out) {
  __shared__ float v[SEAM_SORT];
  __shared__ int id[SEAM_SORT];
  const int r = blockIdx.x, tid = threadIdx.x;
  const float* p = part + (long)r * SEAM_NCH * (2 * SEAM_K + 4);
  for (int i = tid; i < SEAM_SORT; i += 1024) {
    const int c = i / SEAM_K, k = i % SEAM_K;
    const float* q = p + (long)c * (2 * SEAM_K + 4);
    v[i] = q[k];
    id[i] = __builtin_bit_cast(int, q[SEAM_K + k]);
  }
  __syncthreads();
  seam_sort(v, id, 1024);
  float* o = out + (long)r * SEAM_W;
  if (tid < SEAM_K) {
    o[tid] = v[tid];
    o[SEAM_K + tid] = __builtin_bit_cast(float, id[tid]);
  }
  if (tid == 0) {
    float M = -INFINITY, Mt = -INFINITY;
    for (int c = 0; c < SEAM_NCH; ++c) {
      M = fmaxf(M, p[(long)c * (2 * SEAM_K + 4) + 2 * SEAM_K]);
      Mt = fmaxf(Mt, p[(long)c * (2 * SEAM_K + 4) + 2 * SEAM_K + 2]);
    }
    float S = 0.f, St = 0.f;
    for (int c = 0; c < SEAM_NCH; ++c) {
      const float* q = p + (long)c * (2 * SEAM_K + 4) + 2 * SEAM_K;
      S += q[1] * expf(q[0] - M);
      St += q[3] * expf(q[2] - Mt);
    }
    o[2 * SEAM_K] = M + logf(S);
    o[2 * SEAM_K + 1] = Mt + logf(St);
    o[2 * SEAM_K + 2] = 0.f;
    o[2 * SEAM_K + 3] = 0.f;
  }
}
