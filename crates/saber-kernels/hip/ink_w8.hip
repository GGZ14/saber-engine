// Inkling 2 (inkling2_loop) 8-bit dense trunk bodies, INK14 (docs/INKLING2.md section 9 INK14).
// Saber's own.
//
// THE RECORD (`ink14-trunk8/1`, tools/pack/build_inkling2_pack.py --trunk8): a bf16 [N, K] dense
// weight as int8 codes [N, K] row-major (ldw = K bytes) + one bf16 scale a row and 32-k block
// [N, K / 32] (ldsc = K / 32); the weight the bodies multiply is w = (float)code * (float)scale,
// EXACT in f32 (|code| <= 127 = 7 bits times an 8-bit bf16 mantissa). Every body here is its bf16
// parent's order with that w in place of bf16(W): each output ONE sequential f32 sum over
// k = 0..K-1 of products a * w rounded on their own (-ffp-contract=off) -> the host twin is the
// parent's twin on the dequantized f32 weight, BYTE for byte (tools/capture/ink14/ink14_twin.py).
// The dequant is fused into the summing lane's walk (no dequant pass, no f32 weight copy).
//
//   ink_gemv_f32_w8        ink_fast's ink_gemv_f32_bf16 (M <= 8, K % 256 == 0), 64 outputs a
//                          workgroup; 256 k a tile staged through LDS (16 B a row: half the bf16
//                          tile's bytes), FOUR tiles in flight in registers (= the bf16 body's
//                          bytes in flight: the GEMV is latency x bytes-in-flight bound).
//   ink_gemv_n16_f32_w8    the same on a 16-column tile (N <= 1,024; eight tiles in flight).
//   ink_gemv_gate_f32_w8   ink_gemv_gate_f32_bf16: A = o * sigmoid(g) formed while staging.
//   ink_gemv_gate_push_w8 / _n16_w8  ink_fuse's push form (the o GEMV writes its partial into the
//                          PEER's window + raises its workgroup's flag; ink_fuse's slot law).
//   ink_gemv3_f32_w8       ink_fuse11's ink_gemv3_f32_bf16 (three (A, W, C, N) segments).
//   ink_gemm_f32_w8        ink_loop's ink_gemm_f32_bf16 (64 x 64 tiles, K % 32 == 0; prefill).
#include <hip/hip_runtime.h>

__device__ __forceinline__ float w8_bf(unsigned short u) {
  return __builtin_bit_cast(float, ((unsigned)u) << 16);
}
__device__ __forceinline__ float w8_code(unsigned wd, int b) {
  return (float)((int)(wd << (24 - 8 * b)) >> 24);
}
__device__ __forceinline__ void w8_rel(unsigned* p, unsigned v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

#define W8_K 256
#define W8_M 8
#define W8_LDW (W8_K / 4 + 1)   // u32 words a staged row (65: the summing lanes walk 65 banks apart)
#define W8_LDS (W8_K / 32 + 1)  // f32 scales a staged row
#define W8_LDS_BARRIER()                                                    \
  do {                                                                      \
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");         \
    __builtin_amdgcn_s_barrier();                                           \
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");         \
  } while (0)

// one 256-k tile of GN rows: GN x 16 uint4 of codes (GN / 16 a thread), GN uint4 of scales
template <int GN>
struct W8Tile {
  uint4 p[GN / 16];
  uint4 s;
};

template <int GN>
__device__ __forceinline__ void w8_wload(W8Tile<GN>& t, const unsigned char* __restrict__ w, int ldw,
                                         const unsigned short* __restrict__ sc, int ldsc, int n0,
                                         int N, int k0, int tid) {
#pragma unroll
  for (int i = 0; i < GN / 16; ++i) {
    const int e = tid + 256 * i, r = e >> 4, u = e & 15;
    const int gn = n0 + r;
    t.p[i] = gn < N ? *reinterpret_cast<const uint4*>(w + (long)gn * ldw + k0 + u * 16)
                    : make_uint4(0u, 0u, 0u, 0u);
  }
  if (tid < GN) {
    const int gn = n0 + tid;
    t.s = gn < N ? *reinterpret_cast<const uint4*>(sc + (long)gn * ldsc + (k0 >> 5))
                 : make_uint4(0u, 0u, 0u, 0u);
  }
}
template <int MM, int GATE>
__device__ __forceinline__ void w8_aload(float (&ap)[MM], const float* __restrict__ a, int lda,
                                         const float* __restrict__ g, int ldg, int k0, int tid) {
#pragma unroll
  for (int m = 0; m < MM; ++m) {
    if (GATE) {
      const float sg = 1.0f / (1.0f + expf(-g[(long)m * ldg + k0 + tid]));
      ap[m] = a[(long)m * lda + k0 + tid] * sg;
    } else {
      ap[m] = a[(long)m * lda + k0 + tid];
    }
  }
}

template <int GN, int MM>
__device__ __forceinline__ void w8_step(const W8Tile<GN>& t, const float (&ap)[MM], unsigned* ws,
                                        float* ss, float (*as)[W8_K], int tid) {
  W8_LDS_BARRIER();   // every lane is done reading the previous tile
#pragma unroll
  for (int i = 0; i < GN / 16; ++i) {
    const int e = tid + 256 * i, r = e >> 4, u = e & 15;
    unsigned* d = ws + r * W8_LDW + u * 4;
    d[0] = t.p[i].x;
    d[1] = t.p[i].y;
    d[2] = t.p[i].z;
    d[3] = t.p[i].w;
  }
  if (tid < GN) {
    float* d = ss + tid * W8_LDS;
    d[0] = __builtin_bit_cast(float, t.s.x << 16);
    d[1] = __builtin_bit_cast(float, t.s.x & 0xFFFF0000u);
    d[2] = __builtin_bit_cast(float, t.s.y << 16);
    d[3] = __builtin_bit_cast(float, t.s.y & 0xFFFF0000u);
    d[4] = __builtin_bit_cast(float, t.s.z << 16);
    d[5] = __builtin_bit_cast(float, t.s.z & 0xFFFF0000u);
    d[6] = __builtin_bit_cast(float, t.s.w << 16);
    d[7] = __builtin_bit_cast(float, t.s.w & 0xFFFF0000u);
  }
#pragma unroll
  for (int m = 0; m < MM; ++m) as[m][tid] = ap[m];
  W8_LDS_BARRIER();
}

// the summing lane: its row's 256 k in order, w = code * scale (exact), s += a * w
template <int GN, int MM>
__device__ __forceinline__ void w8_sum(const unsigned* ws, const float* ss, float (*as)[W8_K],
                                       float (&s)[W8_M], int tid) {
  if (tid < GN) {
    const unsigned* row = ws + tid * W8_LDW;
    const float* srow = ss + tid * W8_LDS;
#pragma unroll 1
    for (int blk = 0; blk < W8_K / 32; ++blk) {
      const float sc = srow[blk];
#pragma unroll 2
      for (int u = 0; u < 8; ++u) {
        const unsigned wd = row[blk * 8 + u];
        const int k4 = blk * 32 + u * 4;
        const float w0 = w8_code(wd, 0) * sc, w1 = w8_code(wd, 1) * sc;
        const float w2 = w8_code(wd, 2) * sc, w3 = w8_code(wd, 3) * sc;
        // the four activations of these k as one LDS read a row (broadcast); each output's sum
        // still walks k in order
#pragma unroll
        for (int m = 0; m < MM; ++m) {
          const float4 av = *reinterpret_cast<const float4*>(&as[m][k4]);
          s[m] = s[m] + av.x * w0;
          s[m] = s[m] + av.y * w1;
          s[m] = s[m] + av.z * w2;
          s[m] = s[m] + av.w * w3;
        }
      }
    }
  }
}

// NB tiles in flight in registers (a ring); K % 256 == 0
template <int GN, int NB, int MM, int GATE, int PUSH>
__device__ __forceinline__ void w8_body(const float* __restrict__ a, int lda,
                                        const float* __restrict__ g, int ldg,
                                        const unsigned char* __restrict__ w, int ldw,
                                        const unsigned short* __restrict__ sc, int ldsc,
                                        float* __restrict__ c, int ldc, int N, int K, int acc,
                                        int n0, float* pwin, int ld_win, unsigned slot,
                                        unsigned* ws, float* ss, float (*as)[W8_K]) {
  const int tid = threadIdx.x;
  // the weight tiles: an NB ring in registers; the activations: two tiles ahead (NB is even, so
  // tile kt's activation buffer is kt & 1 == j & 1)
  W8Tile<GN> t[NB];
  float ap[2][MM];
  float s[W8_M];
#pragma unroll
  for (int m = 0; m < W8_M; ++m) s[m] = 0.f;
  const int KT = K / W8_K;
#pragma unroll
  for (int j = 0; j < NB; ++j)
    if (j < KT) w8_wload<GN>(t[j], w, ldw, sc, ldsc, n0, N, j * W8_K, tid);
  w8_aload<MM, GATE>(ap[0], a, lda, g, ldg, 0, tid);
  if (KT > 1) w8_aload<MM, GATE>(ap[1], a, lda, g, ldg, W8_K, tid);
  for (int kt0 = 0; kt0 < KT; kt0 += NB) {
#pragma unroll
    for (int j = 0; j < NB; ++j) {
      const int kt = kt0 + j;
      if (kt < KT) {
        w8_step<GN, MM>(t[j], ap[j & 1], ws, ss, as, tid);
        if (kt + NB < KT) w8_wload<GN>(t[j], w, ldw, sc, ldsc, n0, N, (kt + NB) * W8_K, tid);
        if (kt + 2 < KT) w8_aload<MM, GATE>(ap[j & 1], a, lda, g, ldg, (kt + 2) * W8_K, tid);
        w8_sum<GN, MM>(ws, ss, as, s, tid);
      }
    }
  }
  const int gn = n0 + tid;
  if (tid < GN && gn < N) {
#pragma unroll
    for (int m = 0; m < MM; ++m) {
      float* p = c + (long)m * ldc + gn;
      const float v = acc ? *p + s[m] : s[m];
      *p = v;
      if (PUSH) pwin[((long)slot * W8_M + m) * ld_win + gn] = v;
    }
  }
}

template <int GN, int NB, int GATE, int PUSH>
__device__ __forceinline__ void w8_entry(const float* __restrict__ a, int lda,
                                         const float* __restrict__ g, int ldg,
                                         const unsigned char* __restrict__ w, int ldw,
                                         const unsigned short* __restrict__ sc, int ldsc,
                                         float* __restrict__ c, int ldc, int M, int N, int K,
                                         int acc, float* pwin, int ld_win, unsigned* pflags,
                                         unsigned* ctr) {
  __shared__ unsigned ws[GN * W8_LDW];
  __shared__ float ss[GN * W8_LDS];
  __shared__ __attribute__((aligned(16))) float as[W8_M][W8_K];
  __shared__ unsigned s_call;
  const int tid = threadIdx.x;
  unsigned s = 0, slot = 0;
  if (PUSH) {
    if (tid == 0) {
      const unsigned v = ctr[blockIdx.x] + 1u;
      ctr[blockIdx.x] = v;
      s_call = v;
    }
    __syncthreads();
    s = s_call;
    slot = s & 1u;
  }
  const int n0 = blockIdx.x * GN;
#define W8_CASE(MM)                                                                             \
  w8_body<GN, NB, MM, GATE, PUSH>(a, lda, g, ldg, w, ldw, sc, ldsc, c, ldc, N, K, acc, n0, pwin, \
                                  ld_win, slot, ws, ss, as)
  switch (M) {
    case 1: W8_CASE(1); break;
    case 2: W8_CASE(2); break;
    case 3: W8_CASE(3); break;
    case 4: W8_CASE(4); break;
    case 5: W8_CASE(5); break;
    case 6: W8_CASE(6); break;
    case 7: W8_CASE(7); break;
    default: W8_CASE(8); break;
  }
#undef W8_CASE
  if (PUSH) {
    asm volatile("s_wait_storecnt 0x0" ::: "memory");
    __syncthreads();
    if (tid == 0) w8_rel(&pflags[blockIdx.x], s);
  }
}

extern "C" __global__ __launch_bounds__(256) void ink_gemv_f32_w8(
    const float* __restrict__ a, int lda, const unsigned char* __restrict__ w, int ldw,
    const unsigned short* __restrict__ sc, int ldsc, float* __restrict__ c, int ldc, int M, int N,
    int K, int acc) {
  w8_entry<64, 4, 0, 0>(a, lda, nullptr, 0, w, ldw, sc, ldsc, c, ldc, M, N, K, acc, nullptr, 0,
                        nullptr, nullptr);
}
extern "C" __global__ __launch_bounds__(256) void ink_gemv_n16_f32_w8(
    const float* __restrict__ a, int lda, const unsigned char* __restrict__ w, int ldw,
    const unsigned short* __restrict__ sc, int ldsc, float* __restrict__ c, int ldc, int M, int N,
    int K, int acc) {
  w8_entry<16, 8, 0, 0>(a, lda, nullptr, 0, w, ldw, sc, ldsc, c, ldc, M, N, K, acc, nullptr, 0,
                        nullptr, nullptr);
}
extern "C" __global__ __launch_bounds__(256) void ink_gemv_gate_f32_w8(
    const float* __restrict__ a, int lda, const float* __restrict__ g, int ldg,
    const unsigned char* __restrict__ w, int ldw, const unsigned short* __restrict__ sc, int ldsc,
    float* __restrict__ c, int ldc, int M, int N, int K, int acc) {
  w8_entry<64, 4, 1, 0>(a, lda, g, ldg, w, ldw, sc, ldsc, c, ldc, M, N, K, acc, nullptr, 0,
                        nullptr, nullptr);
}
extern "C" __global__ __launch_bounds__(256) void ink_gemv_gate_push_w8(
    const float* __restrict__ a, int lda, const float* __restrict__ g, int ldg,
    const unsigned char* __restrict__ w, int ldw, const unsigned short* __restrict__ sc, int ldsc,
    float* __restrict__ c, int ldc, int M, int N, int K, int acc, float* pwin, int ld_win,
    unsigned* pflags, unsigned* ctr) {
  w8_entry<64, 4, 1, 1>(a, lda, g, ldg, w, ldw, sc, ldsc, c, ldc, M, N, K, acc, pwin, ld_win,
                        pflags, ctr);
}
extern "C" __global__ __launch_bounds__(256) void ink_gemv_gate_push_n16_w8(
    const float* __restrict__ a, int lda, const float* __restrict__ g, int ldg,
    const unsigned char* __restrict__ w, int ldw, const unsigned short* __restrict__ sc, int ldsc,
    float* __restrict__ c, int ldc, int M, int N, int K, int acc, float* pwin, int ld_win,
    unsigned* pflags, unsigned* ctr) {
  w8_entry<16, 8, 1, 1>(a, lda, g, ldg, w, ldw, sc, ldsc, c, ldc, M, N, K, acc, pwin, ld_win,
                        pflags, ctr);
}

// ------------------------------------------------------------------------------------------------
struct Gemv3W8 {
  const float* a[3];
  const unsigned char* w[3];
  const unsigned short* sc[3];
  float* c[3];
  int n[3];
  int ldc[3];
};
extern "C" __global__ __launch_bounds__(256) void ink_gemv3_f32_w8(Gemv3W8 G, int lda, int M, int K) {
  __shared__ unsigned ws[64 * W8_LDW];
  __shared__ float ss[64 * W8_LDS];
  __shared__ __attribute__((aligned(16))) float as[W8_M][W8_K];
  int b = blockIdx.x, sg = 0;
  const int t0 = (G.n[0] + 63) / 64, t1 = (G.n[1] + 63) / 64;
  if (b >= t0) {
    b -= t0;
    sg = 1;
    if (b >= t1) {
      b -= t1;
      sg = 2;
    }
  }
  const float* a = sg == 0 ? G.a[0] : (sg == 1 ? G.a[1] : G.a[2]);
  const unsigned char* w = sg == 0 ? G.w[0] : (sg == 1 ? G.w[1] : G.w[2]);
  const unsigned short* sc = sg == 0 ? G.sc[0] : (sg == 1 ? G.sc[1] : G.sc[2]);
  float* c = sg == 0 ? G.c[0] : (sg == 1 ? G.c[1] : G.c[2]);
  const int N = sg == 0 ? G.n[0] : (sg == 1 ? G.n[1] : G.n[2]);
  const int ldc = sg == 0 ? G.ldc[0] : (sg == 1 ? G.ldc[1] : G.ldc[2]);
  const int n0 = b * 64;
#define W3_CASE(MM)                                                                                \
  w8_body<64, 4, MM, 0, 0>(a, lda, nullptr, 0, w, K, sc, K / 32, c, ldc, N, K, 0, n0, nullptr, 0, \
                           0u, ws, ss, as)
  switch (M) {
    case 1: W3_CASE(1); break;
    case 2: W3_CASE(2); break;
    case 3: W3_CASE(3); break;
    case 4: W3_CASE(4); break;
    case 5: W3_CASE(5); break;
    case 6: W3_CASE(6); break;
    case 7: W3_CASE(7); break;
    default: W3_CASE(8); break;
  }
#undef W3_CASE
}

// ------------------------------------------------------------------------------------------------
// ink_loop's ink_gemm_f32_bf16 with the W stage dequantizing (64 x 64 tiles, 32 k a stage = one
// scale a row and stage); every output a sequential f32 sum over k
#define WG_T 64
#define WG_K 32
extern "C" __global__ __launch_bounds__(256) void ink_gemm_f32_w8(
    const float* __restrict__ a, int lda, const unsigned char* __restrict__ w, int ldw,
    const unsigned short* __restrict__ sc, int ldsc, float* __restrict__ c, int ldc, int M, int N,
    int K, int acc) {
  __shared__ float as[WG_K][WG_T + 1];
  __shared__ float ws[WG_K][WG_T + 1];
  const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
  const int m0 = blockIdx.y * WG_T, n0 = blockIdx.x * WG_T;
  float s[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) s[i][j] = 0.f;
  for (int k0 = 0; k0 < K; k0 += WG_K) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int p = tid + i * 256, r = p >> 5, kk = p & 31;
      const int gm = m0 + r, gn = n0 + r;
      as[kk][r] = gm < M ? a[(long)gm * lda + k0 + kk] : 0.f;
      ws[kk][r] = gn < N ? (float)(int)(signed char)w[(long)gn * ldw + k0 + kk] *
                               w8_bf(sc[(long)gn * ldsc + (k0 >> 5)])
                         : 0.f;
    }
    __syncthreads();
#pragma unroll 4
    for (int kk = 0; kk < WG_K; ++kk) {
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
