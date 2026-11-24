// Inkling 2 (inkling2_loop) native MXFP8 expert bodies, INK6 (docs/INKLING2.md section 4 / 9).
// Saber's own, written from the record law docs/INKLING2.md states; the numpy twins in
// tools/capture/ink6/ink6_moe.py are the spec (bound-gated vs f64, controls that must fail).
//
// Layers 20..23 of the checkpoint store their routed experts as MXFP8: e4m3 weights with one e8m0
// scale per 32 k (w = e4m3 * 2^(s - 127), no global). The INK6 record keeps those bytes as stored
// (no rounding): `w` e4m3 [N, K] row-major (gate_up = gate rows then up rows), `ws` e8m0 [K/32, N].
// xptr[3 e + {0, 1}] = slot e's w / ws (slot 2 unused: the record has no global).
//
//   ink_mx8_w8a8_g{1,0}     the W8A8 grouped cell over Flash-Next's tile law (tile_expert / row0 /
//                           rows, meta[0] = live tiles; g1 gathers A rows through row_tok, g0 reads
//                           the expert-major rows): A = e4m3 rows + one f32 scale a row (`as`), each
//                           32-k group's two 16 x 16 x 16 fp8 WMMAs summed in the WMMA accumulator,
//                           then acc = fma(t, 2^(s - 127), acc) (the scale is a power of two: one
//                           rounding a group); C bf16 = acc * as[row]. 16-row tiles, 128-column
//                           blocks (8 waves x 16), K % 128 == 0, N % 128 == 0.
//   ink_mx8_a16_ref_g{1,0}  the pricing ORACLE of the activation class: bf16 A x the exact weight
//                           (e4m3 * 2^(s - 127) in f32), K in order, one bf16 rounding (FN148's
//                           `fn148_a16_nvfp4_ref` law for the MXFP8 record).
//
// Built with -ffp-contract=off (the `ink_` row of saber-kbuild).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

typedef float ink_floatx8 __attribute__((ext_vector_type(8)));
typedef int ink_int2 __attribute__((ext_vector_type(2)));
typedef unsigned int ink_uint4 __attribute__((ext_vector_type(4)));

#define INK_MX8_NW 8
#define INK_MX8_BN (INK_MX8_NW * 16)
#define INK_MX8_BK 128
#define INK_MX8_PAD 16
#define INK_MX8_STR (INK_MX8_BK + INK_MX8_PAD)

static __device__ __forceinline__ float ink_e8m0(unsigned char s) {
  return __builtin_ldexpf(1.f, (int)s - 127);
}

template <bool GATHER>
__device__ __forceinline__ void ink_mx8_w8a8_impl(
    const unsigned char *__restrict__ A, const unsigned long long *__restrict__ xptr,
    const int *__restrict__ tile_expert, const int *__restrict__ tile_row0,
    const int *__restrict__ tile_rows, const int *__restrict__ row_tok, const int *__restrict__ meta,
    const float *__restrict__ As, __bf16 *__restrict__ C, int N, int K) {
  __shared__ unsigned char sA[16 * INK_MX8_STR];
  __shared__ unsigned char sW[INK_MX8_BN * INK_MX8_STR];
  __shared__ int s_arow[16];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int tile = blockIdx.y;
  if (tile >= meta[0]) return;
  const int n0 = blockIdx.x * INK_MX8_BN;
  const int texp = tile_expert[tile], row0 = tile_row0[tile], M = tile_rows[tile];
  const unsigned char *__restrict__ W = (const unsigned char *)(uintptr_t)xptr[3 * (size_t)texp + 0];
  const unsigned char *__restrict__ Ws = (const unsigned char *)(uintptr_t)xptr[3 * (size_t)texp + 1];
  if (tid < 16) {
    const int rr = tid < M - 1 ? tid : M - 1;
    s_arow[tid] = GATHER ? row_tok[row0 + rr] : row0 + rr;
  }
  __syncthreads();
  const int ncol = n0 + wave * 16 + col < N - 1 ? n0 + wave * 16 + col : N - 1;
  ink_floatx8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const ink_floatx8 z = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int k0 = 0; k0 < K; k0 += INK_MX8_BK) {
    if (tid < 16 * (INK_MX8_BK / 16)) {
      const int r = tid / (INK_MX8_BK / 16), c = (tid % (INK_MX8_BK / 16)) * 16;
      *(ink_uint4 *)&sA[r * INK_MX8_STR + c] = *(const ink_uint4 *)(A + (size_t)s_arow[r] * K + k0 + c);
    }
#pragma unroll
    for (int i = 0; i < INK_MX8_BN * (INK_MX8_BK / 16) / (INK_MX8_NW * 32); ++i) {
      const int idx = tid + i * INK_MX8_NW * 32;
      const int r = idx / (INK_MX8_BK / 16), c = (idx % (INK_MX8_BK / 16)) * 16;
      const int gn = n0 + r < N - 1 ? n0 + r : N - 1;
      *(ink_uint4 *)&sW[r * INK_MX8_STR + c] = *(const ink_uint4 *)(W + (size_t)gn * K + k0 + c);
    }
    float sc[INK_MX8_BK / 32];
#pragma unroll
    for (int g = 0; g < INK_MX8_BK / 32; ++g) sc[g] = ink_e8m0(Ws[(size_t)(k0 / 32 + g) * N + ncol]);
    __syncthreads();
#pragma unroll
    for (int g = 0; g < INK_MX8_BK / 32; ++g) {
      ink_floatx8 t = z;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int kk = g * 32 + h * 16 + kb8;
        const unsigned char *pa = &sA[col * INK_MX8_STR + kk];
        const unsigned char *pw = &sW[(wave * 16 + col) * INK_MX8_STR + kk];
        ink_int2 af, wf;
        af[0] = *(const int *)pa;
        af[1] = *(const int *)(pa + 4);
        wf[0] = *(const int *)pw;
        wf[1] = *(const int *)(pw + 4);
        t = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af, wf, t);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[e] = __builtin_fmaf(t[e], sc[g], acc[e]);
    }
    __syncthreads();
  }
  const int n = n0 + wave * 16 + col;
  if (n < N) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int m = kb8 + e;
      if (m < M) C[(size_t)(row0 + m) * N + n] = (__bf16)(acc[e] * As[s_arow[m]]);
    }
  }
}

#define INK_MX8_W8A8(name, G)                                                                       \
  extern "C" __global__ __launch_bounds__(INK_MX8_NW * 32) void name(                               \
      const unsigned char *__restrict__ A, const unsigned long long *__restrict__ xptr,             \
      const int *__restrict__ tile_expert, const int *__restrict__ tile_row0,                       \
      const int *__restrict__ tile_rows, const int *__restrict__ row_tok,                           \
      const int *__restrict__ meta, const float *__restrict__ As, __bf16 *__restrict__ C, int N,    \
      int K) {                                                                                      \
    ink_mx8_w8a8_impl<G>(A, xptr, tile_expert, tile_row0, tile_rows, row_tok, meta, As, C, N, K);   \
  }
INK_MX8_W8A8(ink_mx8_w8a8_g1, true)
INK_MX8_W8A8(ink_mx8_w8a8_g0, false)
#undef INK_MX8_W8A8

// The A16 oracle: grid (N / 16, tiles, 1), 256 threads = 16 rows x 16 columns.
template <bool GATHER>
__device__ __forceinline__ void ink_mx8_a16_impl(
    const __bf16 *__restrict__ A, const unsigned long long *__restrict__ xptr,
    const int *__restrict__ tile_expert, const int *__restrict__ tile_row0,
    const int *__restrict__ tile_rows, const int *__restrict__ row_tok, const int *__restrict__ meta,
    __bf16 *__restrict__ C, int N, int K) {
  const int tile = blockIdx.y;
  if (tile >= meta[0]) return;
  const int texp = tile_expert[tile], row0 = tile_row0[tile], M = tile_rows[tile];
  const int n = blockIdx.x * 16 + (threadIdx.x & 15);
  const int m = 16 * blockIdx.z + (threadIdx.x >> 4);
  if (m >= M || n >= N) return;
  const unsigned char *W = (const unsigned char *)(uintptr_t)xptr[3 * (size_t)texp + 0];
  const unsigned char *Ws = (const unsigned char *)(uintptr_t)xptr[3 * (size_t)texp + 1];
  const int arow = GATHER ? row_tok[row0 + m] : row0 + m;
  const __bf16 *a = A + (size_t)arow * K;
  const unsigned char *w = W + (size_t)n * K;
  float acc = 0.f, sc = 0.f;
  for (int k = 0; k < K; ++k) {
    if ((k & 31) == 0) sc = ink_e8m0(Ws[(size_t)(k >> 5) * N + n]);
    const float wv = __builtin_amdgcn_cvt_f32_fp8((int)w[k], 0) * sc;
    acc = acc + (float)a[k] * wv;
  }
  C[(size_t)(row0 + m) * N + n] = (__bf16)acc;
}

#define INK_MX8_A16(name, G)                                                                        \
  extern "C" __global__ __launch_bounds__(256) void name(                                           \
      const __bf16 *__restrict__ A, const unsigned long long *__restrict__ xptr,                    \
      const int *__restrict__ tile_expert, const int *__restrict__ tile_row0,                       \
      const int *__restrict__ tile_rows, const int *__restrict__ row_tok,                           \
      const int *__restrict__ meta, __bf16 *__restrict__ C, int N, int K) {                         \
    ink_mx8_a16_impl<G>(A, xptr, tile_expert, tile_row0, tile_rows, row_tok, meta, C, N, K);        \
  }
INK_MX8_A16(ink_mx8_a16_ref_g1, true)
INK_MX8_A16(ink_mx8_a16_ref_g0, false)
#undef INK_MX8_A16
