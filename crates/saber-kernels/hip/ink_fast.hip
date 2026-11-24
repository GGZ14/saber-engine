// Inkling 2 (inkling2_loop) fast decode bodies, INK9 (docs/INKLING2.md section 9). Saber's own.
// Each body here is a re-tile of an INK3b correctness-first body (ink_loop.hip / ink_glue.hip /
// ink_attn.hip) that keeps that body's reduction order element for element, so its outputs are
// BYTE-equal to the parent's (gated in situ: tools/capture/ink9/cmp9.py over the two-card tf +
// decode rows, every row == INK8 h2's).
//
//   ink_gemv_f32_bf16   C[M, N] f32 (+= C when acc) = A[M, K] f32 . W[N, K]^T bf16 at M <= 8 (the
//                       decode rows): ink_gemm_f32_bf16's order (each output ONE sequential f32 sum
//                       over k = 0..K-1, products a * w rounded on their own) with the W rows staged
//                       through LDS by the whole workgroup in 16-byte coalesced loads (64 outputs x
//                       256 k a tile, the next tile in registers while the 64 summing lanes walk
//                       the current one; two tiles in flight ahead, an LDS-only barrier). Requires
//                       K % 256 == 0 and 16-byte aligned rows; M is a template case 1..8.
//
// Built with -ffp-contract=off (saber-kbuild's `ink_` row): every f32 multiply and add rounds on
// its own, as in the parents.
#include <hip/hip_runtime.h>

__device__ __forceinline__ float lo16(unsigned w) { return __builtin_bit_cast(float, w << 16); }
__device__ __forceinline__ float hi16(unsigned w) {
  return __builtin_bit_cast(float, w & 0xFFFF0000u);
}

#define GV_N 64
#define GV_K 256
#define GV_M 8
#define GV_LDW (GV_K / 2 + 1)   // u32 words a staged row (129: the 64 summing lanes hit 64 banks)
// an LDS-only workgroup barrier: a __syncthreads() on gfx12 is a release fence that drains the
// register prefetch (s_wait_loadcnt 0); this one orders LDS only
#define GV_LDS_BARRIER()                                                    \
  do {                                                                      \
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");         \
    __builtin_amdgcn_s_barrier();                                           \
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");         \
  } while (0)

template <int MM>
__device__ __forceinline__ void gv_load(uint4 (&p)[8], float (&ap)[MM], const float* __restrict__ a,
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
__device__ __forceinline__ void gv_step(const uint4 (&p)[8], const float (&ap)[MM], unsigned* ws,
                                        float (*as)[GV_K], float (&s)[GV_M], int tid) {
  GV_LDS_BARRIER();   // every lane is done reading the previous tile
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

// K % 256 == 0 (the decode shapes: 2,304 / 1,536); two tiles in flight (registers) ahead of the
// tile the summing lanes walk
template <int MM>
__device__ __forceinline__ void gv_body(const float* __restrict__ a, int lda,
                                        const unsigned short* __restrict__ w, int ldw,
                                        float* __restrict__ c, int ldc, int N, int K, int acc,
                                        unsigned* ws, float (*as)[GV_K]) {
  const int tid = threadIdx.x;
  const int n0 = blockIdx.x * GV_N;
  uint4 pa[8], pb[8];
  float aa[MM], ab[MM];
  float s[GV_M];
#pragma unroll
  for (int m = 0; m < GV_M; ++m) s[m] = 0.f;
  gv_load<MM>(pa, aa, a, lda, w, ldw, n0, N, 0, tid);
  if (GV_K < K) gv_load<MM>(pb, ab, a, lda, w, ldw, n0, N, GV_K, tid);
  for (int k0 = 0; k0 < K; k0 += 2 * GV_K) {
    gv_step<MM>(pa, aa, ws, as, s, tid);
    if (k0 + 2 * GV_K < K) gv_load<MM>(pa, aa, a, lda, w, ldw, n0, N, k0 + 2 * GV_K, tid);
    gv_sum<MM>(ws, as, s, tid);
    if (k0 + GV_K < K) {
      gv_step<MM>(pb, ab, ws, as, s, tid);
      if (k0 + 3 * GV_K < K) gv_load<MM>(pb, ab, a, lda, w, ldw, n0, N, k0 + 3 * GV_K, tid);
      gv_sum<MM>(ws, as, s, tid);
    }
  }
  const int gn = n0 + tid;
  if (tid < GV_N && gn < N) {
#pragma unroll
    for (int m = 0; m < MM; ++m) {
      float* p = c + (long)m * ldc + gn;
      *p = acc ? *p + s[m] : s[m];
    }
  }
}

extern "C" __global__ __launch_bounds__(256) void ink_gemv_f32_bf16(
    const float* __restrict__ a, int lda, const unsigned short* __restrict__ w, int ldw,
    float* __restrict__ c, int ldc, int M, int N, int K, int acc) {
  __shared__ unsigned ws[GV_N * GV_LDW];
  __shared__ float as[GV_M][GV_K];
  switch (M) {
    case 1: gv_body<1>(a, lda, w, ldw, c, ldc, N, K, acc, ws, as); break;
    case 2: gv_body<2>(a, lda, w, ldw, c, ldc, N, K, acc, ws, as); break;
    case 3: gv_body<3>(a, lda, w, ldw, c, ldc, N, K, acc, ws, as); break;
    case 4: gv_body<4>(a, lda, w, ldw, c, ldc, N, K, acc, ws, as); break;
    case 5: gv_body<5>(a, lda, w, ldw, c, ldc, N, K, acc, ws, as); break;
    case 6: gv_body<6>(a, lda, w, ldw, c, ldc, N, K, acc, ws, as); break;
    case 7: gv_body<7>(a, lda, w, ldw, c, ldc, N, K, acc, ws, as); break;
    default: gv_body<8>(a, lda, w, ldw, c, ldc, N, K, acc, ws, as); break;
  }
}

// ------------------------------------------------------------------------------------------------
// The message gate as two launches (ink_loop.hip's ink_msg_gate, one workgroup a row, split by
// what it computes; every value in the parent's order):
//   ink_msg_z    grid [T, 8], block 256: every workgroup computes the parent's rh / rm (the same
//                256-lane partials + bsum256) and its 8 z rows, wave w row 8 b + w with the parent's
//                lane walk (lane L sums i = L, L + 32, .. in order, then the xor butterfly); the
//                Wd row comes through a wave-private LDS window in 16-byte loads. z -> [T, 64] f32.
//   ink_msg_out  grid [ceil(D / 64), T], block 64: out[c] = (1 - tanh f) h + tanh(g) m with
//                f = sum_j z[j] Wf[c, j] in j order (the row in 16-byte loads), as the parent.
#define FM_D 2304
#define FM_PER 9
#define FM_R 64
__device__ __forceinline__ float fsum256(float v, float* red) {
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
__device__ __forceinline__ float fbf(unsigned short u) {
  return __builtin_bit_cast(float, ((unsigned)u) << 16);
}

extern "C" __global__ __launch_bounds__(256) void ink_msg_z(
    const float* __restrict__ h, int ld_h, const float* __restrict__ m, int ld_m,
    const unsigned short* __restrict__ wd,   // [64, 2D]
    float* __restrict__ z, float eps) {
  __shared__ float in[2 * FM_D];
  __shared__ float red[8];
  __shared__ unsigned short win[8][512];
  const int t = blockIdx.x, b = blockIdx.y, tid = threadIdx.x, lane = tid & 31, wv = tid >> 5;
  float hv[FM_PER], mv[FM_PER], sh = 0.f, sm = 0.f;
#pragma unroll
  for (int i = 0; i < FM_PER; ++i) {
    hv[i] = h[(long)t * ld_h + tid + 256 * i];
    mv[i] = m[(long)t * ld_m + tid + 256 * i];
    sh = sh + hv[i] * hv[i];
    sm = sm + mv[i] * mv[i];
  }
  const float rh = 1.0f / sqrtf(fsum256(sh, red) / (float)FM_D + eps);
  const float rm = 1.0f / sqrtf(fsum256(sm, red) / (float)FM_D + eps);
#pragma unroll
  for (int i = 0; i < FM_PER; ++i) {
    in[tid + 256 * i] = hv[i] * rh;
    in[FM_D + tid + 256 * i] = mv[i] * rm;
  }
  __syncthreads();
  const int r = b * 8 + wv;
  const unsigned short* wr = wd + (long)r * 2 * FM_D;
  float d = 0.f;
  for (int c0 = 0; c0 < 2 * FM_D; c0 += 512) {
    // lane L stages elements c0 + 16 L .. + 15 (two 16-byte loads), then walks its own
    // L, L + 32, .. of the window
    const uint4 x0 = *reinterpret_cast<const uint4*>(wr + c0 + 16 * lane);
    const uint4 x1 = *reinterpret_cast<const uint4*>(wr + c0 + 16 * lane + 8);
    *reinterpret_cast<uint4*>(&win[wv][16 * lane]) = x0;
    *reinterpret_cast<uint4*>(&win[wv][16 * lane + 8]) = x1;
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int i = c0 + lane + 32 * j;
      d = d + in[i] * fbf(win[wv][lane + 32 * j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) d += __shfl_xor(d, off, 32);
  if (lane == 0) z[(long)t * FM_R + r] = d;
}

extern "C" __global__ __launch_bounds__(64) void ink_msg_out(
    const float* __restrict__ h, int ld_h, const float* __restrict__ m, int ld_m,
    const float* __restrict__ z,
    const unsigned short* __restrict__ wf,   // [D, 64]
    const unsigned short* __restrict__ wg,   // [D, 64]
    float* __restrict__ o, int ld_o) {
  __shared__ float zs[FM_R];
  const int t = blockIdx.y, c = blockIdx.x * 64 + threadIdx.x;
  zs[threadIdx.x] = z[(long)t * FM_R + threadIdx.x];
  __syncthreads();
  if (c >= FM_D) return;
  uint4 f4[8], g4[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    f4[i] = *reinterpret_cast<const uint4*>(wf + (long)c * FM_R + 8 * i);
    g4[i] = *reinterpret_cast<const uint4*>(wg + (long)c * FM_R + 8 * i);
  }
  float f = 0.f, g = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const unsigned fw[4] = {f4[i].x, f4[i].y, f4[i].z, f4[i].w};
    const unsigned gw[4] = {g4[i].x, g4[i].y, g4[i].z, g4[i].w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int j = 8 * i + 2 * q;
      f = f + zs[j] * lo16(fw[q]);
      g = g + zs[j] * lo16(gw[q]);
      f = f + zs[j + 1] * hi16(fw[q]);
      g = g + zs[j + 1] * hi16(gw[q]);
    }
  }
  const float hv = h[(long)t * ld_h + c], mv = m[(long)t * ld_m + c];
  o[(long)t * ld_o + c] = (1.0f - tanhf(f)) * hv + tanhf(g) * mv;
}

// ------------------------------------------------------------------------------------------------
// The post-attention norm + router + top-8 (ink_glue.hip's ink_post_norm_route_e128, one
// 128-lane workgroup a row) as two launches, every value in the parent's order:
//   ink_route_logits  grid [T, 16], block 128: every workgroup computes the parent's norm (the
//                     same 128-lane partials, butterfly, ((r0 + r1) + r2) + r3) and u; workgroup
//                     0 writes u; workgroup b scores experts 8 b .. 8 b + 7, each ONE sequential
//                     dot over i (the parent's), its 8 router rows staged in LDS in 16-byte loads.
//                     -> sk [T, 128, 2] = (s + bias, s).
//   ink_route_top8    grid [T], block 128: the parent's rank / select / renorm tail.
extern "C" __global__ __launch_bounds__(128) void ink_route_logits(
    const float* __restrict__ h, int ld_h, const unsigned short* __restrict__ gain,
    const unsigned short* __restrict__ wr, const float* __restrict__ bias,
    float* __restrict__ u, int ld_u, float* __restrict__ sk, int D) {
  __shared__ float us[2304];
  __shared__ float red[4];
  __shared__ unsigned short wsr[8][2304];
  const int t = blockIdx.x, b = blockIdx.y, e = threadIdx.x, lane = e & 31, w = e >> 5;
  // stage the 8 router rows first (independent of the norm)
  for (int p = e; p < 8 * (D / 8); p += 128) {
    const int r = p / (D / 8), q = p % (D / 8);
    *reinterpret_cast<uint4*>(&wsr[r][8 * q]) =
        *reinterpret_cast<const uint4*>(wr + (long)(8 * b + r) * D + 8 * q);
  }
  float ss = 0.f;
  for (int i = e; i < D; i += 128) {
    const float x = h[(long)t * ld_h + i];
    us[i] = x;
    ss = ss + x * x;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o, 32);
  if (lane == 0) red[w] = ss;
  __syncthreads();
  const float tot = ((red[0] + red[1]) + red[2]) + red[3];
  const float r = 1.0f / sqrtf(tot / (float)D + 1e-6f);
  __syncthreads();
  for (int i = e; i < D; i += 128) {
    const float v = (us[i] * r) * fbf(gain[i]);
    us[i] = v;
    if (b == 0) u[(long)t * ld_u + i] = v;
  }
  __syncthreads();
  if (e < 8) {
    float dot = 0.f;
    // the parent's sequential dot, the row read 8 bf16 (16 bytes) at a time from LDS
    const uint4* wr8 = reinterpret_cast<const uint4*>(&wsr[e][0]);
#pragma unroll 4
    for (int i8 = 0; i8 < 2304 / 8; ++i8) {
      const uint4 x = wr8[i8];
      const unsigned wv[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        dot = dot + us[8 * i8 + 2 * q] * lo16(wv[q]);
        dot = dot + us[8 * i8 + 2 * q + 1] * hi16(wv[q]);
      }
    }
    const int ex = 8 * b + e;
    const float s = 1.0f / (1.0f + expf(-dot));
    sk[((long)t * 128 + ex) * 2 + 0] = s + bias[ex];
    sk[((long)t * 128 + ex) * 2 + 1] = s;
  }
}

extern "C" __global__ __launch_bounds__(128) void ink_route_top8(
    const float* __restrict__ sk, int* __restrict__ ids, float* __restrict__ wts, float route_scale) {
  __shared__ float key[128];
  __shared__ float sel[8];
  __shared__ int selid[8];
  const int t = blockIdx.x, e = threadIdx.x;
  const float mk = sk[((long)t * 128 + e) * 2 + 0], s = sk[((long)t * 128 + e) * 2 + 1];
  key[e] = mk;
  __syncthreads();
  int rank = 0;
  for (int j = 0; j < 128; ++j) {
    const float kj = key[j];
    rank += (kj > mk) || (kj == mk && j < e);
  }
  if (rank < 8) {
    sel[rank] = s;
    selid[rank] = e;
  }
  __syncthreads();
  if (e < 8) {
    float sum = 0.f;
    for (int j = 0; j < 8; ++j) sum = sum + sel[j];
    ids[t * 8 + e] = selid[e];
    wts[t * 8 + e] = sel[e] / (sum + 1e-20f) * route_scale;
  }
}

// ------------------------------------------------------------------------------------------------
//   ink_gemv_gate_f32_bf16  ink_gemv_f32_bf16 with A = o * sigmoid(g) formed while staging (the
//       attention's output gate, ink_loop.hip's ink_out_gate: sg = 1 / (1 + exp(-g)), o * sg) --
//       the o projection without the gated row's round trip; byte-equal to out_gate + the GEMV.
template <int MM>
__device__ __forceinline__ void gvg_load(uint4 (&p)[8], float (&ap)[MM], const float* __restrict__ a,
                                         int lda, const float* __restrict__ g, int ldg,
                                         const unsigned short* __restrict__ w, int ldw, int n0,
                                         int N, int k0, int tid) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int e = tid + 256 * i, r = e >> 5, u = e & 31;
    const int gn = n0 + r;
    p[i] = gn < N ? *reinterpret_cast<const uint4*>(w + (long)gn * ldw + k0 + u * 8)
                  : make_uint4(0u, 0u, 0u, 0u);
  }
#pragma unroll
  for (int m = 0; m < MM; ++m) {
    const float sg = 1.0f / (1.0f + expf(-g[(long)m * ldg + k0 + tid]));
    ap[m] = a[(long)m * lda + k0 + tid] * sg;
  }
}

template <int MM>
__device__ __forceinline__ void gvg_body(const float* __restrict__ a, int lda,
                                         const float* __restrict__ g, int ldg,
                                         const unsigned short* __restrict__ w, int ldw,
                                         float* __restrict__ c, int ldc, int N, int K, int acc,
                                         unsigned* ws, float (*as)[GV_K]) {
  const int tid = threadIdx.x;
  const int n0 = blockIdx.x * GV_N;
  uint4 pa[8], pb[8];
  float aa[MM], ab[MM];
  float s[GV_M];
#pragma unroll
  for (int m = 0; m < GV_M; ++m) s[m] = 0.f;
  gvg_load<MM>(pa, aa, a, lda, g, ldg, w, ldw, n0, N, 0, tid);
  if (GV_K < K) gvg_load<MM>(pb, ab, a, lda, g, ldg, w, ldw, n0, N, GV_K, tid);
  for (int k0 = 0; k0 < K; k0 += 2 * GV_K) {
    gv_step<MM>(pa, aa, ws, as, s, tid);
    if (k0 + 2 * GV_K < K) gvg_load<MM>(pa, aa, a, lda, g, ldg, w, ldw, n0, N, k0 + 2 * GV_K, tid);
    gv_sum<MM>(ws, as, s, tid);
    if (k0 + GV_K < K) {
      gv_step<MM>(pb, ab, ws, as, s, tid);
      if (k0 + 3 * GV_K < K)
        gvg_load<MM>(pb, ab, a, lda, g, ldg, w, ldw, n0, N, k0 + 3 * GV_K, tid);
      gv_sum<MM>(ws, as, s, tid);
    }
  }
  const int gn = n0 + tid;
  if (tid < GV_N && gn < N) {
#pragma unroll
    for (int m = 0; m < MM; ++m) {
      float* p = c + (long)m * ldc + gn;
      *p = acc ? *p + s[m] : s[m];
    }
  }
}

extern "C" __global__ __launch_bounds__(256) void ink_gemv_gate_f32_bf16(
    const float* __restrict__ a, int lda, const float* __restrict__ g, int ldg,
    const unsigned short* __restrict__ w, int ldw, float* __restrict__ c, int ldc, int M, int N,
    int K, int acc) {
  __shared__ unsigned ws[GV_N * GV_LDW];
  __shared__ float as[GV_M][GV_K];
  switch (M) {
    case 1: gvg_body<1>(a, lda, g, ldg, w, ldw, c, ldc, N, K, acc, ws, as); break;
    case 2: gvg_body<2>(a, lda, g, ldg, w, ldw, c, ldc, N, K, acc, ws, as); break;
    case 3: gvg_body<3>(a, lda, g, ldg, w, ldw, c, ldc, N, K, acc, ws, as); break;
    case 4: gvg_body<4>(a, lda, g, ldg, w, ldw, c, ldc, N, K, acc, ws, as); break;
    case 5: gvg_body<5>(a, lda, g, ldg, w, ldw, c, ldc, N, K, acc, ws, as); break;
    case 6: gvg_body<6>(a, lda, g, ldg, w, ldw, c, ldc, N, K, acc, ws, as); break;
    case 7: gvg_body<7>(a, lda, g, ldg, w, ldw, c, ldc, N, K, acc, ws, as); break;
    default: gvg_body<8>(a, lda, g, ldg, w, ldw, c, ldc, N, K, acc, ws, as); break;
  }
}

// ------------------------------------------------------------------------------------------------
//   ink_route_tail  ONE launch for the route's tail and the expert cells' A quant: block 0 =
//       ink_route_top8 over every row (the parent's rank / select / renorm, ids + wts to global
//       and LDS) then ink_route_sort_e128's body over the LDS ids for one rank's experts; blocks
//       1..T = ink_act_quant_rows of row b - 1 (the parent's bmax256 + e4m3 convert). T <= 32.
__device__ __forceinline__ float fmax256(float v, float* red) {
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
__device__ __forceinline__ unsigned char ff2e4m3(float v) {
  const int p = __builtin_amdgcn_cvt_pk_fp8_f32(v, 0.f, 0, false);
  return (unsigned char)(p & 0xFF);
}
#define RT_E 128
extern "C" __global__ __launch_bounds__(256) void ink_route_tail(
    const float* __restrict__ sk, int* __restrict__ ids, float* __restrict__ wts, float route_scale,
    const float* __restrict__ x, int ld_x, unsigned char* __restrict__ q, int ld_q,
    float* __restrict__ scale, int K, int T, int lo, int nloc, int* __restrict__ row_tok,
    int* __restrict__ row_slot, int* __restrict__ tile_e, int* __restrict__ tile_r0,
    int* __restrict__ tile_n, int* __restrict__ meta, int* __restrict__ pos) {
  const int tid = threadIdx.x;
  if (blockIdx.x > 0) {
    __shared__ float red[8];
    const int t = blockIdx.x - 1;
    float mx = 0.f;
    for (int i = tid; i < K; i += 256) mx = fmaxf(mx, fabsf(x[(long)t * ld_x + i]));
    mx = fmax256(mx, red);
    const float s = fmaxf(mx / 448.0f, 1e-30f);
    for (int i = tid; i < K; i += 256) q[(long)t * ld_q + i] = ff2e4m3(x[(long)t * ld_x + i] / s);
    if (tid == 0) scale[t] = s;
    return;
  }
  __shared__ float key[RT_E];
  __shared__ float sel[8];
  __shared__ int selid[8];
  __shared__ int sids[256];
  for (int t = 0; t < T; ++t) {
    float mk = 0.f, sv = 0.f;
    if (tid < RT_E) {
      mk = sk[((long)t * 128 + tid) * 2 + 0];
      sv = sk[((long)t * 128 + tid) * 2 + 1];
      key[tid] = mk;
    }
    __syncthreads();
    if (tid < RT_E) {
      int rank = 0;
      for (int j = 0; j < 128; ++j) {
        const float kj = key[j];
        rank += (kj > mk) || (kj == mk && j < tid);
      }
      if (rank < 8) {
        sel[rank] = sv;
        selid[rank] = tid;
      }
    }
    __syncthreads();
    if (tid < 8) {
      float sum = 0.f;
      for (int j = 0; j < 8; ++j) sum = sum + sel[j];
      ids[t * 8 + tid] = selid[tid];
      sids[t * 8 + tid] = selid[tid];
      wts[t * 8 + tid] = sel[tid] / (sum + 1e-20f) * route_scale;
    }
    __syncthreads();
  }
  // ink_route_sort_e128 over sids
  __shared__ int cnt[RT_E], seg[RT_E], run[RT_E], tbase[RT_E];
  __shared__ int ce[256];
  const int N = T * 8;
  if (tid < RT_E) { cnt[tid] = 0; run[tid] = 0; }
  __syncthreads();
  for (int i = tid; i < N; i += 256) {
    const int e = sids[i] - lo;
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
  for (int e = 0; e < nloc; ++e) {
    const int nt = (cnt[e] + 15) >> 4;
    for (int k = tid; k < nt; k += 256) {
      tile_e[tbase[e] + k] = e;
      tile_r0[tbase[e] + k] = seg[e] + 16 * k;
      tile_n[tbase[e] + k] = min(16, cnt[e] - 16 * k);
    }
  }
  for (int b = 0; b < N; b += 256) {
    const int i = b + tid;
    int e = -1;
    if (i < N) {
      e = sids[i] - lo;
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
  for (int e = 0; e < nloc; ++e) {
    const int c = cnt[e], padded = ((c + 15) >> 4) * 16;
    __syncthreads();
    for (int k = c + tid; k < padded; k += 256) {
      row_tok[seg[e] + k] = row_tok[seg[e] + c - 1];
      row_slot[seg[e] + k] = -1;
    }
    __syncthreads();
  }
}

// ------------------------------------------------------------------------------------------------
//   ink_attn_bias_sink_v1  ink_attn.hip's ink_attn_bias_sink_v0 with the band bias formed in the
//       body (ink_loop.hip's ink_rel_bias: rel = sum_k r_post[t, h, k] P[k, dist] in k order, the
//       same expression) instead of a [T, H, 1024] round trip, the key rows read in 16-byte loads
//       and the value walk unrolled (loads ahead of the sequential sum); every logit, max, sum and
//       accumulator in v0's order -> byte-equal to rel_bias + v0.
#define FA_HD 128
#define FA_PAGE 64
#define FA_TILE 256
__device__ __forceinline__ float fa_max(float v, float* red) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
  if (lane == 0) red[w] = v;
  __syncthreads();
  float m = red[0];
#pragma unroll
  for (int i = 1; i < FA_TILE / 32; ++i) m = fmaxf(m, red[i]);
  __syncthreads();
  return m;
}
__device__ __forceinline__ float fa_sum(float v, float* red) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
  if (lane == 0) red[w] = v;
  __syncthreads();
  float s = red[0];
#pragma unroll
  for (int i = 1; i < FA_TILE / 32; ++i) s += red[i];
  __syncthreads();
  return s;
}

extern "C" __global__ __launch_bounds__(256) void ink_attn_bias_sink_v1(
    const unsigned short* __restrict__ q, int ld_q,
    const unsigned short* __restrict__ k_pool, const unsigned short* __restrict__ v_pool,
    const int* __restrict__ table, int table_ld,
    const int* __restrict__ row_seq, const int* __restrict__ row_pos,
    const float* __restrict__ rp, int ld_rp,               // [T][H * 16] f32 (r_post)
    const unsigned short* __restrict__ P,                  // [16][extent] bf16 (rel_proj)
    const float* __restrict__ sink, int ld_sink,
    float* __restrict__ o, int ld_o,
    int H, int KV, int extent, int window) {
  __shared__ float qs[FA_HD];
  __shared__ float ps[FA_TILE];
  __shared__ float red[FA_TILE / 32];
  __shared__ float oacc[2][FA_HD];
  __shared__ float rv[16];
  const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
  const int kvh = h / (H / KV);
  const int C = KV * FA_HD;
  const int sq = row_seq[t], pos = row_pos[t];
  const int lo = window > 0 ? max(0, pos - window + 1) : 0;
  if (tid < FA_HD) qs[tid] = fbf(q[(long)t * ld_q + h * FA_HD + tid]);
  if (tid < 16) rv[tid] = rp[(long)t * ld_rp + h * 16 + tid];
  __syncthreads();
  float m = -INFINITY, l = 0.f, acc = 0.f;
  const int d = tid & (FA_HD - 1), half = tid >> 7;
  for (int base = lo; base <= pos; base += FA_TILE) {
    const int j = base + tid;
    float s = -INFINITY;
    if (j <= pos) {
      const long row = (long)table[(long)sq * table_ld + j / FA_PAGE] * FA_PAGE + j % FA_PAGE;
      const uint4* kr = reinterpret_cast<const uint4*>(k_pool + row * C + kvh * FA_HD);
      uint4 kv4[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) kv4[i] = kr[i];
      float dot = 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const unsigned wv[4] = {kv4[i].x, kv4[i].y, kv4[i].z, kv4[i].w};
#pragma unroll
        for (int qd = 0; qd < 4; ++qd) {
          dot = dot + qs[8 * i + 2 * qd] * lo16(wv[qd]);
          dot = dot + qs[8 * i + 2 * qd + 1] * hi16(wv[qd]);
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
    const float tm = fa_max(s, red);
    const float mn = fmaxf(m, tm);
    const float p = (j <= pos) ? expf(s - mn) : 0.f;
    ps[tid] = p;
    const float corr = (m == -INFINITY) ? 0.f : expf(m - mn);
    const float ts = fa_sum(p, red);
    l = l * corr + ts;
    float a = 0.f;
    const int jn = min(FA_TILE / 2, pos - base - half * (FA_TILE / 2) + 1);
    const int j0 = base + half * (FA_TILE / 2);
    int i = 0;
    for (; i + 8 <= jn; i += 8) {
      float vv[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const int jj = j0 + i + u;
        const long row = (long)table[(long)sq * table_ld + jj / FA_PAGE] * FA_PAGE + jj % FA_PAGE;
        vv[u] = fbf(v_pool[row * C + kvh * FA_HD + d]);
      }
#pragma unroll
      for (int u = 0; u < 8; ++u) a = a + ps[half * (FA_TILE / 2) + i + u] * vv[u];
    }
    for (; i < jn; ++i) {
      const int jj = j0 + i;
      const long row = (long)table[(long)sq * table_ld + jj / FA_PAGE] * FA_PAGE + jj % FA_PAGE;
      a = a + ps[half * (FA_TILE / 2) + i] * fbf(v_pool[row * C + kvh * FA_HD + d]);
    }
    acc = acc * corr + a;
    m = mn;
    __syncthreads();
  }
  const float sk = sink[(long)t * ld_sink + h];
  const float mf = fmaxf(m, sk);
  const float c = expf(m - mf);
  const float lf = l * c + expf(sk - mf);
  oacc[half][d] = acc;
  __syncthreads();
  if (tid < FA_HD) o[(long)t * ld_o + h * FA_HD + d] = ((oacc[0][d] + oacc[1][d]) * c) / lf;
}

// ------------------------------------------------------------------------------------------------
//   ink_sconv_rows  ink_loop.hip's ink_sconv_resid at ONE token a row for T rows of DIFFERENT
//       sequences (batched decode): row t reads and writes its own 3-row state, slot rseq[t] of
//       st [slots][3][C] (in place: the row's thread reads its state before it writes it), the
//       parent's T = 1 arithmetic (acc over the taps j = 0..3 in order: the state rows then x;
//       (acc + x) / sqrt 5 [+ resid]) -> byte-equal to the parent run per sequence.
extern "C" __global__ __launch_bounds__(256) void ink_sconv_rows(
    const float* __restrict__ x, int ld_x, const unsigned short* __restrict__ w,
    float* st, const int* __restrict__ rseq, const float* __restrict__ resid, int ld_r,
    float* __restrict__ o, int ld_o, int T, int C) {
  const int c = blockIdx.x * 256 + threadIdx.x, t = blockIdx.y;
  if (c >= C || t >= T) return;
  float* s = st + (long)rseq[t] * 3 * C;
  float wt[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) wt[j] = fbf(w[(long)c * 4 + j]);
  const float xt = x[(long)t * ld_x + c];
  const float s0 = s[c], s1 = s[(long)C + c], s2 = s[2L * C + c];
  float acc = 0.f;
  acc = acc + s0 * wt[0];
  acc = acc + s1 * wt[1];
  acc = acc + s2 * wt[2];
  acc = acc + xt * wt[3];
  float y = (acc + xt) / 2.2360679774997898f;
  if (resid) y = y + resid[(long)t * ld_r + c];
  o[(long)t * ld_o + c] = y;
  s[c] = s1;
  s[(long)C + c] = s2;
  s[2L * C + c] = xt;
}
