// Saber's fp8 lm_head at 33..64 rows in ONE weight pass (CMP27F, 2026-10-26).
//
//   C[m][n] = bf16( sum over k-steps ascending of WMMA(A[m][k..k+16], bf16(f32(W8[n][k..]) * ws[n])) )
//
// Why a new tile: the 27B's b8 verify asks the head for 64 rows. The 32-row band reads the 0.636 GB
// head twice (2.2 ms); a 64-row band with every wave loading its own A fragments from memory reads A
// (64 x K bf16) once PER WAVE -- 7,760 waves x 640 KB -- and that, not the weight, is its wall (the
// one-pass band's time grows with the number of distinct A rows). Here a workgroup of 8 waves (128
// output columns) stages A once per 128-k chunk in LDS (double-buffered) and every wave reads its
// four m-fragments from there; the weight is read once, 16 bytes a lane a k-step pair.
//
// Arithmetic: every output element is the served bands' -- the same fragments (A bytes, the weight
// expanded with the integer round-to-nearest-even, `saber_rne_pair`), the same WMMA, k-steps in
// ascending order into one f32 accumulator per fragment, one bf16 rounding at the store. Requires
// K % 128 == 0 (the router refuses otherwise).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

namespace shw {
typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float floatx8 __attribute__((ext_vector_type(8)));
typedef unsigned u4v __attribute__((ext_vector_type(4)));
constexpr int MR = 64;          // rows a workgroup holds (four 16-row WMMA fragments)

__device__ __forceinline__ floatx8 wmma(bf16x8 a, bf16x8 b, floatx8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}
__device__ __forceinline__ unsigned rne_pair(float lo, float hi) {
  unsigned a = __float_as_uint(lo), b = __float_as_uint(hi);
  a += 0x7fffu + ((a >> 16) & 1u);
  b += 0x7fffu + ((b >> 16) & 1u);
  return __builtin_amdgcn_perm(b, a, 0x07060302u);
}
__device__ __forceinline__ bf16x8 expand(uint2 r, float s) {
  typedef float floatx2 __attribute__((ext_vector_type(2)));
  const floatx2 f0 = __builtin_amdgcn_cvt_pk_f32_fp8((int)r.x, false);
  const floatx2 f1 = __builtin_amdgcn_cvt_pk_f32_fp8((int)r.x, true);
  const floatx2 f2 = __builtin_amdgcn_cvt_pk_f32_fp8((int)r.y, false);
  const floatx2 f3 = __builtin_amdgcn_cvt_pk_f32_fp8((int)r.y, true);
  union { uint4 u; bf16x8 h; } o;
  o.u.x = rne_pair(f0[0] * s, f0[1] * s);
  o.u.y = rne_pair(f1[0] * s, f1[1] * s);
  o.u.z = rne_pair(f2[0] * s, f2[1] * s);
  o.u.w = rne_pair(f3[0] * s, f3[1] * s);
  return o.h;
}
__device__ __forceinline__ uint4 ld16_nt(const void* p) {
  const u4v v = __builtin_nontemporal_load((const u4v*)p);
  return make_uint4(v.x, v.y, v.z, v.w);
}
__device__ __forceinline__ void lds_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
}  // namespace shw

// NW waves a workgroup (each owns 16 output columns), KC k a chunk staged in LDS (double-buffered).
template <int NW, int KC>
__device__ __forceinline__ void head_m64(const __bf16* __restrict__ a,
                                         const unsigned char* __restrict__ w,
                                         const float* __restrict__ ws, __bf16* __restrict__ c,
                                         int m, int n, int k) {
  using namespace shw;
  constexpr int T = NW * 32;
  constexpr int AS = KC + 8;                 // LDS row stride in bf16 (16 B pad)
  constexpr int RV = KC / 8;                 // 16-byte vectors a row of a chunk
  constexpr int PT = MR * RV / T;            // vectors a thread stages a chunk
  constexpr int RS = T / RV;                 // row stride between a thread's vectors
  constexpr int KSC = KC / 16;               // k-steps a chunk
  static_assert(PT * T == MR * RV && KSC % 2 == 0, "staging geometry");
  __shared__ __attribute__((aligned(16))) __bf16 sA[2][MR * AS];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int n0 = (int)(blockIdx.x * NW + wave) * 16;
  const int m0 = (int)blockIdx.y * MR;
  const int row = lane & 15, hh = lane >> 4;
  const bool wl = n0 < n;                         // a wave past N still helps stage A
  const int nrow = wl ? (n0 + row < n ? n0 + row : n - 1) : 0;
  const unsigned char* wp = w + (long)nrow * k + hh * 16;
  const float s = ws[nrow];
  const int srow0 = tid / RV, scol = (tid % RV) * 8;
  // rows past m clamp to the last real row (never stored)
#define SHW_AROW(i) (a + (long)((m0 + srow0 + RS * (i)) < m ? (m0 + srow0 + RS * (i)) : m - 1) * k + scol)
  // the staged vectors as named registers (an indexed array here went to scratch)
  uint4 ra0, ra1, ra2, ra3, ra4, ra5, ra6, ra7, ra8, ra9, raa, rab, rac, rad, rae, raf;
#define SHW_EACH(X) X(0, ra0) X(1, ra1) X(2, ra2) X(3, ra3) X(4, ra4) X(5, ra5) X(6, ra6) X(7, ra7) X(8, ra8) X(9, ra9) X(10, raa) X(11, rab) X(12, rac) X(13, rad) X(14, rae) X(15, raf)
#define SHW_LD0(i, r) if constexpr ((i) < PT) r = *(const uint4*)SHW_AROW(i);
#define SHW_LD(i, r) if constexpr ((i) < PT) r = *(const uint4*)(SHW_AROW(i) + off);
#define SHW_ST(i, r) if constexpr ((i) < PT) *(uint4*)&dstb[(srow0 + RS * (i)) * AS + scol] = r;
  SHW_EACH(SHW_LD0)
  uint4 wc[KSC / 2], wn[KSC / 2];
#pragma unroll
  for (int u = 0; u < KSC / 2; ++u) wc[u] = ld16_nt(wp + (2 * u) * 16);
  {
    __bf16* dstb = &sA[0][0];
    SHW_EACH(SHW_ST)
  }
  lds_barrier();

  floatx8 acc[4];
#pragma unroll
  for (int f = 0; f < 4; ++f)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[f][e] = 0.f;

  const int nc = k / KC;
  for (int ch = 0; ch < nc; ++ch) {
    const int cur = ch & 1;
    const bool more = ch + 1 < nc;
    if (more) {
      const int off = (ch + 1) * KC;
      SHW_EACH(SHW_LD)
#pragma unroll
      for (int u = 0; u < KSC / 2; ++u) wn[u] = ld16_nt(wp + ((ch + 1) * KSC + 2 * u) * 16);
    }
    const __bf16* ab = &sA[cur][0];
#pragma unroll
    for (int u = 0; u < KSC / 2; ++u) {
      // the lane's 16-byte load covers k-step 2u (low half-wave) or 2u+1 (high): swap the 8-byte
      // halves with the lane sixteen away so each lane holds its own fragment of both k-steps
      const unsigned sx = hh ? wc[u].x : wc[u].z, sy = hh ? wc[u].y : wc[u].w;
      const unsigned rx = __builtin_amdgcn_permlanex16(sx, sx, 0x76543210u, 0xfedcba98u, false, false);
      const unsigned ry = __builtin_amdgcn_permlanex16(sy, sy, 0x76543210u, 0xfedcba98u, false, false);
      const uint2 e0 = hh ? make_uint2(rx, ry) : make_uint2(wc[u].x, wc[u].y);
      const uint2 e1 = hh ? make_uint2(wc[u].z, wc[u].w) : make_uint2(rx, ry);
      const bf16x8 w0 = expand(e0, s), w1 = expand(e1, s);
#pragma unroll
      for (int f = 0; f < 4; ++f) {
        const bf16x8 af = *(const bf16x8*)(ab + (f * 16 + row) * AS + (2 * u) * 16 + hh * 8);
        acc[f] = wmma(af, w0, acc[f]);
      }
#pragma unroll
      for (int f = 0; f < 4; ++f) {
        const bf16x8 af = *(const bf16x8*)(ab + (f * 16 + row) * AS + (2 * u + 1) * 16 + hh * 8);
        acc[f] = wmma(af, w1, acc[f]);
      }
    }
    if (more) {
      __bf16* dstb = &sA[cur ^ 1][0];
      SHW_EACH(SHW_ST)
#pragma unroll
      for (int u = 0; u < KSC / 2; ++u) wc[u] = wn[u];
    }
    lds_barrier();
  }

  const int col = n0 + (lane & 15);
  if (!wl || col >= n) return;
#pragma unroll
  for (int f = 0; f < 4; ++f)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int r = m0 + f * 16 + hh * 8 + e;
      if (r < m) c[(long)r * n + col] = (__bf16)acc[f][e];
    }
}

#define SABER_HEAD_M64_ENTRY(NAME, NW, KC)                                                         \
  extern "C" __global__ __launch_bounds__(NW * 32) void NAME(                                      \
      const __bf16* __restrict__ a, const unsigned char* __restrict__ w,                           \
      const float* __restrict__ ws, __bf16* __restrict__ c, int m, int n, int k) {                 \
    head_m64<NW, KC>(a, w, ws, c, m, n, k);                                                        \
  }

SABER_HEAD_M64_ENTRY(saber_head_w8_m64_lds, 8, 128)
SABER_HEAD_M64_ENTRY(saber_head_w8_m64_lds_nw4_kc128, 4, 128)
SABER_HEAD_M64_ENTRY(saber_head_w8_m64_lds_nw8_kc64, 8, 64)
SABER_HEAD_M64_ENTRY(saber_head_w8_m64_lds_nw4_kc64, 4, 64)
SABER_HEAD_M64_ENTRY(saber_head_w8_m64_lds_nw2_kc128, 2, 128)
