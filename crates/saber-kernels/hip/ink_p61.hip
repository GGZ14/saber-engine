// INK61 (docs/INKLING2.md INK61): byte-equal prefill levers, Saber's own, every body BYTE-EQUAL to its control
// (`--ink-p61 <letters>`; the kernel-vs-kernel oracle is tools/capture/ink61/ink61_twin.py).
//
//   ink61_mx8_tm4_n{1,2}_g{1,0}   item `a`: INK6's MXFP8 W8A8 grouped cell (`ink_mx8_w8a8_g*`, hip/ink_moe.hip)
//       on INK37's 64-row expert tile (the route sort's 64-row tables, dead 16-row fragments skipped). Per
//       output element the parent's program exactly: each 32-k group's two 16 x 16 x 16 fp8 WMMAs summed
//       into a ZERO-started accumulator (k ascending, the same lane -> k map), acc = fma(t, 2^(s - 127), acc)
//       over the groups ascending, C bf16 = acc * as[row]. gfx1201's WMMA gives an output element the same
//       bits whichever fragment carries it (FN21's law), so the tile height never moves a byte. A workgroup =
//       64 rows x (128 x n) columns, 8 waves; a wave = the four 16-row fragments x n 16-column fragments (the
//       W fragment read once from LDS for four row fragments); the 128-deep slab staged through LDS (the
//       parent's 144-byte rows), the next slab's A / W / scale bytes loaded into registers while the current
//       one computes. K % 128 == 0, N % (128 n) == 0. Signature = SIG_INK_MX8_W8A8.
//   ink61_route_q8_e128_r4        item `b`: INK37's `ink37_post_norm_route_e128_r4` BYTE FOR BYTE (u, ids,
//       wts) + `ink_act_quant_rows` of the u it writes (the MoE cells' A: e4m3 rows, s = max(amax / 448,
//       1e-30) a row) in the same launch: amax is order-free (fmaxf of |v|); the second pass recomputes each
//       v with the same expression ((x * r) * gain) and writes f2e4m3(v / s) -- the parent's bytes.
//
// Built with -ffp-contract=off (saber-kbuild's `ink_` rule): every f32 multiply and add rounds on its own.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

typedef float i61_f8 __attribute__((ext_vector_type(8)));
typedef int i61_i2 __attribute__((ext_vector_type(2)));
typedef unsigned int i61_u4 __attribute__((ext_vector_type(4)));

#define I61_BK 128
#define I61_STR (I61_BK + 16)   // the parent's staged row (144 B)
#define I61_BM 64

static __device__ __forceinline__ float i61_e8m0(unsigned char s) {
  return __builtin_ldexpf(1.f, (int)s - 127);
}

template <int TN, bool GATHER>
__device__ __forceinline__ void ink61_mx8_impl(
    const unsigned char *__restrict__ A, const unsigned long long *__restrict__ xptr,
    const int *__restrict__ tile_expert, const int *__restrict__ tile_row0,
    const int *__restrict__ tile_rows, const int *__restrict__ row_tok, const int *__restrict__ meta,
    const float *__restrict__ As, __bf16 *__restrict__ C, int N, int K) {
  constexpr int NW = 8, BN = NW * 16 * TN, CH = I61_BK / 16;   // 16-byte chunks a staged row
  constexpr int AL = I61_BM * CH / (NW * 32), WL = BN * CH / (NW * 32);
  __shared__ __attribute__((aligned(16))) unsigned char sA[I61_BM * I61_STR];
  __shared__ __attribute__((aligned(16))) unsigned char sW[BN * I61_STR];
  __shared__ int s_arow[I61_BM];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int tile = blockIdx.y;
  if (tile >= meta[0]) return;
  const int n0 = blockIdx.x * BN;
  const int texp = tile_expert[tile], row0 = tile_row0[tile], M = tile_rows[tile];
  const unsigned char *__restrict__ W = (const unsigned char *)(uintptr_t)xptr[3 * (size_t)texp + 0];
  const unsigned char *__restrict__ Ws = (const unsigned char *)(uintptr_t)xptr[3 * (size_t)texp + 1];
  if (tid < I61_BM) {
    const int rr = tid < M - 1 ? tid : M - 1;   // clamp, never predicate
    s_arow[tid] = GATHER ? row_tok[row0 + rr] : row0 + rr;
  }
  __syncthreads();
  const int nlf = __builtin_amdgcn_readfirstlane(min(4, (M + 15) / 16));
  const unsigned char *abase[AL];
  int aoff[AL];
#pragma unroll
  for (int i = 0; i < AL; ++i) {
    const int idx = tid + i * NW * 32, r = idx / CH, c = (idx % CH) * 16;
    abase[i] = A + (size_t)s_arow[r] * K + c;
    aoff[i] = r * I61_STR + c;
  }
  const unsigned char *wbase[WL];
  int woff[WL];
#pragma unroll
  for (int i = 0; i < WL; ++i) {
    const int idx = tid + i * NW * 32, r = idx / CH, c = (idx % CH) * 16;
    const int gn = n0 + r < N - 1 ? n0 + r : N - 1;
    wbase[i] = W + (size_t)gn * K + c;
    woff[i] = r * I61_STR + c;
  }
  const unsigned char *sbase[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int n = n0 + wave * 16 * TN + j * 16 + col;
    sbase[j] = Ws + (n < N - 1 ? n : N - 1);
  }
  i61_f8 acc[4][TN];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;
  const i61_f8 z = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  i61_u4 ar[AL], wr[WL];
  unsigned char sr[TN][I61_BK / 32];
#pragma unroll
  for (int i = 0; i < AL; ++i) ar[i] = *(const i61_u4 *)abase[i];
#pragma unroll
  for (int i = 0; i < WL; ++i) wr[i] = *(const i61_u4 *)wbase[i];
#pragma unroll
  for (int j = 0; j < TN; ++j)
#pragma unroll
    for (int g = 0; g < I61_BK / 32; ++g) sr[j][g] = sbase[j][(size_t)g * N];
  for (int k0 = 0; k0 < K; k0 += I61_BK) {
#pragma unroll
    for (int i = 0; i < AL; ++i) *(i61_u4 *)&sA[aoff[i]] = ar[i];
#pragma unroll
    for (int i = 0; i < WL; ++i) *(i61_u4 *)&sW[woff[i]] = wr[i];
    float sc[TN][I61_BK / 32];
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int g = 0; g < I61_BK / 32; ++g) sc[j][g] = i61_e8m0(sr[j][g]);
    __syncthreads();
    if (k0 + I61_BK < K) {
      const int kn = k0 + I61_BK;
#pragma unroll
      for (int i = 0; i < AL; ++i) ar[i] = *(const i61_u4 *)(abase[i] + kn);
#pragma unroll
      for (int i = 0; i < WL; ++i) wr[i] = *(const i61_u4 *)(wbase[i] + kn);
#pragma unroll
      for (int j = 0; j < TN; ++j)
#pragma unroll
        for (int g = 0; g < I61_BK / 32; ++g) sr[j][g] = sbase[j][(size_t)(kn / 32 + g) * N];
    }
#pragma unroll
    for (int g = 0; g < I61_BK / 32; ++g) {
      i61_i2 wf[TN][2];
#pragma unroll
      for (int j = 0; j < TN; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h)
          wf[j][h] = *(const i61_i2 *)&sW[(wave * 16 * TN + j * 16 + col) * I61_STR + g * 32 + h * 16 + kb8];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (i < nlf) {
          i61_i2 af[2];
#pragma unroll
          for (int h = 0; h < 2; ++h) af[h] = *(const i61_i2 *)&sA[(i * 16 + col) * I61_STR + g * 32 + h * 16 + kb8];
#pragma unroll
          for (int j = 0; j < TN; ++j) {
            i61_f8 t = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[0], wf[j][0], z);
            t = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[1], wf[j][1], t);
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = __builtin_fmaf(t[e], sc[j][g], acc[i][j][e]);
          }
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if (i < nlf) {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int n = n0 + wave * 16 * TN + j * 16 + col;
        if (n < N) {
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int m = i * 16 + kb8 + e;
            if (m < M) C[(size_t)(row0 + m) * N + n] = (__bf16)(acc[i][j][e] * As[s_arow[m]]);
          }
        }
      }
    }
  }
}

#define INK61_MX8(name, TN, G)                                                                      \
  extern "C" __global__ __launch_bounds__(256) void name(                                           \
      const unsigned char *__restrict__ A, const unsigned long long *__restrict__ xptr,             \
      const int *__restrict__ tile_expert, const int *__restrict__ tile_row0,                       \
      const int *__restrict__ tile_rows, const int *__restrict__ row_tok,                           \
      const int *__restrict__ meta, const float *__restrict__ As, __bf16 *__restrict__ C, int N,    \
      int K) {                                                                                      \
    ink61_mx8_impl<TN, G>(A, xptr, tile_expert, tile_row0, tile_rows, row_tok, meta, As, C, N, K);  \
  }
INK61_MX8(ink61_mx8_tm4_n1_g1, 1, true)
INK61_MX8(ink61_mx8_tm4_n1_g0, 1, false)
INK61_MX8(ink61_mx8_tm4_n2_g1, 2, true)
INK61_MX8(ink61_mx8_tm4_n2_g0, 2, false)
#undef INK61_MX8

// ------------------------------------------------------------------------------- b: router + act quant
__device__ __forceinline__ float i61_bf2f(unsigned short u) {
  return __builtin_bit_cast(float, ((unsigned)u) << 16);
}
__device__ __forceinline__ unsigned char i61_f2e4m3(float v) {
  // OCP e4m3 (fn), RNE, the card's convert (saturating at 448) -- ink_loop's f2e4m3
  const int p = __builtin_amdgcn_cvt_pk_fp8_f32(v, 0.f, 0, false);
  return (unsigned char)(p & 0xFF);
}
#define I61_CH 256   // columns a chunk of the normed rows in LDS (INK37's RT_CH)

template <int R>
__device__ __forceinline__ void route_q8(
    const float *__restrict__ h, int ld_h, const unsigned short *__restrict__ gain,
    const unsigned short *__restrict__ wr, const float *__restrict__ bias,
    float *__restrict__ u, int ld_u, int *__restrict__ ids, float *__restrict__ wts, int D,
    float route_scale, int T, unsigned char *__restrict__ q, int ld_q, float *__restrict__ qs) {
  __shared__ float us[R][I61_CH];
  __shared__ float red[R][4];
  __shared__ float key[R][128];
  __shared__ float sel[R][8];
  __shared__ int selid[R][8];
  __shared__ float mxs[R][4];
  const int t0 = blockIdx.x * R, e = threadIdx.x, lane = e & 31, w = e >> 5;
  int tr[R];
#pragma unroll
  for (int r = 0; r < R; ++r) tr[r] = t0 + r < T ? t0 + r : T - 1;   // clamp, writes guarded
#pragma unroll
  for (int r = 0; r < R; ++r) {
    float ss = 0.f;
    for (int i = e; i < D; i += 128) {
      const float x = h[(long)tr[r] * ld_h + i];
      ss = ss + x * x;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o, 32);
    if (lane == 0) red[r][w] = ss;
  }
  __syncthreads();
  float rn[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const float tot = ((red[r][0] + red[r][1]) + red[r][2]) + red[r][3];
    rn[r] = 1.0f / sqrtf(tot / (float)D + 1e-6f);
  }
  float dot[R], mx[R];
#pragma unroll
  for (int r = 0; r < R; ++r) dot[r] = 0.f, mx[r] = 0.f;
  const unsigned short *wrow = wr + (long)e * D;
  for (int c0 = 0; c0 < D; c0 += I61_CH) {
    const int nc = D - c0 < I61_CH ? D - c0 : I61_CH;
    __syncthreads();
#pragma unroll
    for (int r = 0; r < R; ++r) {
      for (int i = e; i < nc; i += 128) {
        const float v = (h[(long)tr[r] * ld_h + c0 + i] * rn[r]) * i61_bf2f(gain[c0 + i]);
        us[r][i] = v;
        mx[r] = fmaxf(mx[r], fabsf(v));
        if (t0 + r < T) u[(long)(t0 + r) * ld_u + c0 + i] = v;
      }
    }
    __syncthreads();
    if (nc == I61_CH && ((((unsigned long)(wrow + c0)) & 15) == 0)) {
      for (int i = 0; i < I61_CH; i += 8) {
        const uint4 qq4 = *(const uint4 *)(wrow + c0 + i);
        const unsigned int qq[4] = {qq4.x, qq4.y, qq4.z, qq4.w};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float wv = i61_bf2f((unsigned short)(qq[j >> 1] >> (16 * (j & 1))));
#pragma unroll
          for (int r = 0; r < R; ++r) dot[r] = dot[r] + us[r][i + j] * wv;
        }
      }
    } else {
      for (int i = 0; i < nc; ++i) {
        const float wv = i61_bf2f(wrow[c0 + i]);
#pragma unroll
        for (int r = 0; r < R; ++r) dot[r] = dot[r] + us[r][i] * wv;
      }
    }
  }
  // the act quant's amax (order-free) -> the row scale, then the codes from the same v
#pragma unroll
  for (int r = 0; r < R; ++r) {
    float m = mx[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
    if (lane == 0) mxs[r][w] = m;
  }
  float s[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    s[r] = 1.0f / (1.0f + expf(-dot[r]));
    key[r][e] = s[r] + bias[e];
  }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const float m = fmaxf(fmaxf(mxs[r][0], mxs[r][1]), fmaxf(mxs[r][2], mxs[r][3]));
    const float sq = fmaxf(m / 448.0f, 1e-30f);
    if (t0 + r < T) {
      for (int i = e; i < D; i += 128) {
        const float v = (h[(long)tr[r] * ld_h + i] * rn[r]) * i61_bf2f(gain[i]);
        q[(long)(t0 + r) * ld_q + i] = i61_f2e4m3(v / sq);
      }
      if (e == 0) qs[t0 + r] = sq;
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const float mk = key[r][e];
    int rank = 0;
    for (int j = 0; j < 128; ++j) {
      const float kj = key[r][j];
      rank += (kj > mk) || (kj == mk && j < e);
    }
    if (rank < 8) {
      sel[r][rank] = s[r];
      selid[r][rank] = e;
    }
  }
  __syncthreads();
  if (e < 8) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      if (t0 + r >= T) continue;
      float sum = 0.f;
      for (int j = 0; j < 8; ++j) sum = sum + sel[r][j];
      ids[(t0 + r) * 8 + e] = selid[r][e];
      wts[(t0 + r) * 8 + e] = sel[r][e] / (sum + 1e-20f) * route_scale;
    }
  }
}

extern "C" __global__ __launch_bounds__(128) void ink61_route_q8_e128_r4(
    const float *__restrict__ h, int ld_h, const unsigned short *__restrict__ gain,
    const unsigned short *__restrict__ wr, const float *__restrict__ bias, float *__restrict__ u,
    int ld_u, int *__restrict__ ids, float *__restrict__ wts, int D, float route_scale, int T,
    unsigned char *__restrict__ q, int ld_q, float *__restrict__ qs) {
  route_q8<4>(h, ld_h, gain, wr, bias, u, ld_u, ids, wts, D, route_scale, T, q, ld_q, qs);
}
