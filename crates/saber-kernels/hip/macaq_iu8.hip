// Saber source (2026-11-17): the W4A8-iu8 pack's GEMMs (README, "W4A8-iu8"). Saber's own: the decode band
// and the prefill tile are integer-WMMA (`v_wmma_i32_16x16x16_iu8`) bodies with N guards for
// n-tile counts that are not a multiple of 8 (the GDN `in_proj_ba` 48 / 96 rows), an M clamp for the last
// A tile, a runtime split count, the W4 (int4 per-row) and W8 (int8 per-row) operands, and the epilogue
// order of the contract below.
//
// THE OPERANDS.
//   A   int8 activations in the A-TILED fragment order (the MXFP4 path's `a_tiled_fp8` order with int8
//       codes): fragment (mt, kk) = 256 B at (mt * K/16 + kk) * 256; lane l's 8 bytes = row mt*16 + l%16,
//       k = kk*16 + (l/16)*8 + 0..7. Rows past M inside the last 16-row tile are never stored from.
//   as  f32 [M]: the per-token scale (absmax / 127) the producer wrote.
//   W4  int4 [-8, 7] two's-complement nibbles, fragment (nt, kk) = 128 B at (nt * K/16 + kk) * 128; lane
//       l's dword at l*4, byte b: low nibble = W[nt*16 + l%16][kk*16 + (l/16)*8 + b], high nibble = the
//       same row at k + 4. Unpacked at staging to the exact int8 16 * w (lo = (d << 4) & 0xF0F0F0F0,
//       hi = d & 0xF0F0F0F0), so the int32 accumulator holds 16 * sum(a * w).
//   W8  int8 [-127, 127], fragment (nt, kk) = 256 B: lane l's 8 bytes = row nt*16 + l%16, k = kk*16 +
//       (l/16)*8 + 0..7 (the A order, by rows of W).
//   ws  f32 [N]: one scale an output row (the pack's row scale s, the sim's `s`).
//
// THE CONTRACT (the epilogue, every kernel, every split): acc = the exact int32 sum over ALL of K (int32
// adds are associative, so the split count, the k walk and the tile order are NOT numerics);
//   y = bf16_rne( (f32(acc) * as[m]) * (ws[n] * U) ),   U = 1/16 for W4 (the unpack's x16), 1 for W8,
// each `*` one f32 IEEE multiply (contraction OFF in the epilogue), so a numpy float32 reference
// reproduces every output bit for bit.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

typedef int iu8_int2 __attribute__((ext_vector_type(2)));
typedef int iu8_intx8 __attribute__((ext_vector_type(8)));
typedef unsigned int iu8_uint4 __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ iu8_intx8 iu8_wmma(iu8_int2 a, iu8_int2 b, iu8_intx8 c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
}
// an LDS-only workgroup barrier (a __syncthreads() would also drain the register prefetch)
static __device__ __forceinline__ void iu8_bar() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
static __device__ __forceinline__ float iu8_epi(int acc, float a_scale, float w_scale_u) {
#pragma clang fp contract(off)
  const float t = (float)acc * a_scale;
  return t * w_scale_u;
}
template <bool W8>
static __device__ __forceinline__ float iu8_wsu(const float *__restrict__ ws, int n) {
#pragma clang fp contract(off)
  return W8 ? ws[n] : ws[n] * 0.0625f;
}

// ============================================================================ decode band (M <= 64)
// 8 waves a block, one 16-column n-tile a wave (a block = 128 columns), A staged in LDS once a slab,
// W nontemporal straight into the B operand, K slab 128, split-K over blockIdx.z (gridDim.z = ks) with
// the last-arriving split reducing the int32 partials (exact) and running the epilogue. DTM = ceil(M/16).
template <int DTM, bool W8>
__device__ __forceinline__ void iu8_dec_impl(const unsigned char *__restrict__ AT, const unsigned char *__restrict__ W,
                                             const float *__restrict__ as, const float *__restrict__ ws,
                                             int *__restrict__ P, int *__restrict__ cnt, __bf16 *__restrict__ C,
                                             int M, int N, int K) {
  constexpr int NS = 8;
  constexpr int WFB = W8 ? 256 : 128;  // bytes a W fragment
  __shared__ int s_last;
  __shared__ unsigned char sA[DTM * NS * 256];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int ntiles = N >> 4;
  const int nt = blockIdx.x * 8 + wave;
  const bool live = nt < ntiles;
  const int ntl = live ? nt : ntiles - 1;  // a dead wave streams a live tile's bytes and stores nothing
  const int ks = blockIdx.z, DKS = gridDim.z;
  const int ksg = K >> 4;
  const int slabs = K >> 7, spb = (slabs + DKS - 1) / DKS;
  const int s_lo = ks * spb, s_hi = min(slabs, s_lo + spb);
  const unsigned char *wbase = W + (size_t)ntl * ksg * WFB;
  const int n = ntl * 16 + col;
  iu8_intx8 acc[DTM];
#pragma unroll
  for (int i = 0; i < DTM; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0;
#pragma unroll 2
  for (int s = s_lo; s < s_hi; ++s) {
    const int kk0 = s * NS;
    iu8_int2 wv[NS];
#pragma unroll
    for (int st = 0; st < NS; ++st) {
      if constexpr (W8) {
        const unsigned *p = (const unsigned *)(wbase + ((unsigned)lane * 8 + (unsigned)((kk0 + st) * 256)));
        wv[st][0] = (int)__builtin_nontemporal_load(p);
        wv[st][1] = (int)__builtin_nontemporal_load(p + 1);
      } else {
        const unsigned d = __builtin_nontemporal_load((const unsigned *)(wbase + ((unsigned)lane * 4 + (unsigned)((kk0 + st) * 128))));
        wv[st][0] = (int)((d << 4) & 0xF0F0F0F0u);
        wv[st][1] = (int)(d & 0xF0F0F0F0u);
      }
    }
    for (int o = tid * 16; o < DTM * NS * 256; o += 256 * 16) {
      const int i = o / (NS * 256), r = o % (NS * 256);
      *(iu8_uint4 *)&sA[o] = *(const iu8_uint4 *)(AT + (size_t)(i * ksg + kk0) * 256 + r);
    }
    iu8_bar();
    iu8_int2 af[DTM][NS];
#pragma unroll
    for (int i = 0; i < DTM; ++i)
#pragma unroll
      for (int st = 0; st < NS; ++st) af[i][st] = *(const iu8_int2 *)&sA[(i * NS + st) * 256 + lane * 8];
#pragma unroll
    for (int st = 0; st < NS; ++st)
#pragma unroll
      for (int i = 0; i < DTM; ++i) acc[i] = iu8_wmma(af[i][st], wv[st], acc[i]);
    iu8_bar();
  }
  if (DKS == 1) {
    if (!live) return;
    const float wsu = iu8_wsu<W8>(ws, n);
#pragma unroll
    for (int i = 0; i < DTM; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = i * 16 + kb8 + e;
        if (m < M) C[(size_t)m * N + n] = (__bf16)iu8_epi(acc[i][e], as[m], wsu);
      }
    return;
  }
  if (live) {
#pragma unroll
    for (int i = 0; i < DTM; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = i * 16 + kb8 + e;
        if (m < M) P[((size_t)ks * M + m) * N + n] = acc[i][e];
      }
  }
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    s_last = (atomicAdd(&cnt[blockIdx.x], 1) == DKS - 1);
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  if (tid == 0) cnt[blockIdx.x] = 0;  // self-resetting: zeroed once at construction, never again
  const int nb = blockIdx.x * 128, nw = min(128, N - nb);
  for (int idx = tid; idx < nw * M; idx += 256) {
    const int m = idx / nw, nn = nb + idx % nw;
    int sum = 0;
    for (int k = 0; k < DKS; ++k) sum += __builtin_nontemporal_load(&P[((size_t)k * M + m) * N + nn]);
    C[(size_t)m * N + nn] = (__bf16)iu8_epi(sum, as[m], iu8_wsu<W8>(ws, nn));
  }
}

#define IU8_DEC(name, DTM, W8)                                                                                     \
  extern "C" __global__ __launch_bounds__(256) void name(                                                         \
      const unsigned char *__restrict__ a, const unsigned char *__restrict__ w, const float *__restrict__ as,     \
      const float *__restrict__ ws, int *__restrict__ decode_partial, int *__restrict__ decode_counter,           \
      __bf16 *__restrict__ c, int m, int n, int k) {                                                               \
    iu8_dec_impl<DTM, W8>(a, w, as, ws, decode_partial, decode_counter, c, m, n, k);                               \
  }
IU8_DEC(macaq_iu8_dec_w4_tm1, 1, false)
IU8_DEC(macaq_iu8_dec_w4_tm2, 2, false)
IU8_DEC(macaq_iu8_dec_w4_tm3, 3, false)
IU8_DEC(macaq_iu8_dec_w4_tm4, 4, false)
IU8_DEC(macaq_iu8_dec_w8_tm1, 1, true)
IU8_DEC(macaq_iu8_dec_w8_tm2, 2, true)
IU8_DEC(macaq_iu8_dec_w8_tm3, 3, true)
IU8_DEC(macaq_iu8_dec_w8_tm4, 4, true)
#undef IU8_DEC

// ============================================================================ prefill tile (M > 64)
// The prefill tile: 4 x 2 waves, a wave 4 m-fragments x TN 4 n-fragments (BM 256, BN 128), K slab
// LBK 64 (NS 4 fragments), A straight from the A-tiled edge into registers, W staged through LDS (W4
// unpacked to int8 on the way), one int32 accumulator set over all of K, the epilogue as the decode band.
template <bool W8>
__device__ __forceinline__ void iu8_pf_impl(const unsigned char *__restrict__ AT, const unsigned char *__restrict__ W,
                                            const float *__restrict__ as, const float *__restrict__ ws,
                                            __bf16 *__restrict__ C, int M, int N, int K) {
  constexpr int TN = 4, LBK = 64, PWM = 4, PWN = 2, PTM = 4, NS = LBK / 16, NT = PWM * PWN * 32;
  constexpr int BM = PWM * PTM * 16, BN = PWN * TN * 16;
  constexpr int WFB = W8 ? 256 : 128;
  constexpr int PSLAB = BN / 16 * NS * WFB;  // bytes of W a slab
  constexpr int PIECES = PSLAB / (NT * 16);
  static_assert(PSLAB % (NT * 16) == 0, "whole pieces");
  __shared__ unsigned char sW[BN / 16 * NS * 256];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / PWN, wn = wave % PWN;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int ksg = K >> 4;
  const int mtiles = (M + 15) >> 4, ntiles = N >> 4;
  const unsigned aoff = lane * 8;
  const unsigned char *abase[PTM];
#pragma unroll
  for (int i = 0; i < PTM; ++i) {
    const int mt = min(m0 / 16 + wm * PTM + i, mtiles - 1);  // past the last tile: re-read it, store nothing
    abase[i] = AT + (size_t)__builtin_amdgcn_readfirstlane(mt * ksg * 256);
  }
  int nj[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) nj[j] = n0 + wn * TN * 16 + j * 16 + col;
  iu8_intx8 acc[PTM][TN];
#pragma unroll
  for (int i = 0; i < PTM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0;
  for (int k0 = 0; k0 < K; k0 += LBK) {
    const int kk0 = k0 >> 4;
    iu8_int2 af[PTM][NS];
#pragma unroll
    for (int i = 0; i < PTM; ++i)
#pragma unroll
      for (int st = 0; st < NS; ++st)
        af[i][st] = *(const iu8_int2 *)(abase[i] + (aoff + (unsigned)(kk0 * 256) + (unsigned)(st * 256)));
    iu8_uint4 v[PIECES];
    int vdst[PIECES];
#pragma unroll
    for (int p = 0; p < PIECES; ++p) {
      const int o = (p * NT + tid) * 16;
      const int ntl = o / (NS * WFB), rem = o % (NS * WFB);
      const int ntg = min(n0 / 16 + ntl, ntiles - 1);  // a tile past N re-reads the last one (stored nowhere)
      v[p] = *(const iu8_uint4 *)(W + (size_t)ntg * ksg * WFB + (unsigned)(kk0 * WFB + rem));
      vdst[p] = ntl * NS * 256 + (rem / WFB) * 256 + (rem % WFB) * (W8 ? 1 : 2);
    }
#pragma unroll
    for (int p = 0; p < PIECES; ++p) {
      unsigned char *dst = &sW[vdst[p]];
      if constexpr (W8) {
        *(iu8_uint4 *)dst = v[p];
      } else {
        unsigned lo[4], hi[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          lo[q] = (v[p][q] << 4) & 0xF0F0F0F0u;
          hi[q] = v[p][q] & 0xF0F0F0F0u;
        }
        *(iu8_uint4 *)dst = iu8_uint4{lo[0], hi[0], lo[1], hi[1]};
        *(iu8_uint4 *)(dst + 16) = iu8_uint4{lo[2], hi[2], lo[3], hi[3]};
      }
    }
    iu8_bar();
#pragma unroll
    for (int st = 0; st < NS; ++st) {
      iu8_int2 wf[TN];
#pragma unroll
      for (int j = 0; j < TN; ++j) wf[j] = *(const iu8_int2 *)&sW[(wn * TN + j) * NS * 256 + st * 256 + lane * 8];
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < PTM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = iu8_wmma(af[i][st], wf[j], acc[i][j]);
    }
    iu8_bar();
  }
  float wsu[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) wsu[j] = nj[j] < N ? iu8_wsu<W8>(ws, nj[j]) : 0.f;
#pragma unroll
  for (int i = 0; i < PTM; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int m = m0 + wm * PTM * 16 + i * 16 + kb8 + e;
      if (m >= M) continue;
      const float am = as[m];
#pragma unroll
      for (int j = 0; j < TN; ++j)
        if (nj[j] < N) C[(size_t)m * N + nj[j]] = (__bf16)iu8_epi(acc[i][j][e], am, wsu[j]);
    }
}

#define IU8_PF(name, W8)                                                                                       \
  extern "C" __global__ __launch_bounds__(256) void name(                                                     \
      const unsigned char *__restrict__ a, const unsigned char *__restrict__ w, const float *__restrict__ as, \
      const float *__restrict__ ws, __bf16 *__restrict__ c, int m, int n, int k) {                             \
    iu8_pf_impl<W8>(a, w, as, ws, c, m, n, k);                                                                 \
  }
IU8_PF(macaq_iu8_pf_w4_tn4_lbk64, false)
IU8_PF(macaq_iu8_pf_w8_tn4_lbk64, true)
#undef IU8_PF
