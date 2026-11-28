// INK62 (docs/INKLING2.md INK62): byte-equal re-forms of the NVFP4 MoE PREFILL GEMM, Saber's own, every body
// BYTE-EQUAL per row to FN153's grouped prefill tile `fn153_w4a8b_pf_tm4_g{1,0}` (hip/moe_fn153.hip; INK37's
// `--ink-p37 moe` cell); `--ink-p62 <letters>`, the kernel-vs-kernel oracle is tools/capture/ink62/ink62_twin.py.
//
// The per-element program is the parent's, unchanged: acc = fma(wmma_fp8(a8, fp8(e2m1), 0)[e], E4M3(ws[g, n]),
// acc) over the 16-k groups ascending (one zero-started 16 x 16 x 16 WMMA a group, the same lane -> k map), C =
// bf16((acc * gs) * As[row]). gfx1201's WMMA gives an output element the same bits whichever fragment carries
// it (FN21's law), so the wave / tile shape never moves a byte. What the arms change is only the schedule:
//
//   ink62_nv_p_g{1,0}   item `p`: the parent's 64 x 128 tile (8 waves, a wave = one 16-row fragment x four
//       16-column fragments) with the NEXT 64-deep slab's A / W / scale words loaded into registers while the
//       current slab computes (the parent loads, waits and converts serially: two exposed memory latencies a
//       slab), global (not flat) loads, LDS-only barriers (the parent's `__syncthreads` is a release fence =
//       `s_wait_loadcnt 0` + a `global_inv` a barrier).
//   ink62_nv_w_g{1,0}   item `w`: `p` on a 64 x 256 tile, a wave = the four 16-row fragments x two 16-column
//       fragments (INK61 a's shape: the W fragment read once for four row fragments, a dead 16-row fragment
//       skipped by every wave alike).
//   ink62_nv_t_g{1,0}   item `t`: `p` on a 128 x 128 tile (the route sort's 128-row tables, INK37's `moe8`), a
//       wave = 4 row x 2 column fragments (the e2m1 -> fp8 convert a W element shared by 128 rows, not 64).
//
// The record = the xtier v2 NVFP4 one (w permute_w e2m1, ws E4M3 g16 [K/16, N], gs f32); ABI SIG_FN148_GROUPED
// (P / cnt / Pmax unused). K % 64 == 0, N % BN == 0. Built with -ffp-contract=off (saber-kbuild's `ink_` rule).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

typedef float i62_f8 __attribute__((ext_vector_type(8)));
typedef int i62_i2 __attribute__((ext_vector_type(2)));
typedef unsigned int i62_u4 __attribute__((ext_vector_type(4)));
typedef unsigned int i62_u2 __attribute__((ext_vector_type(2)));

#define I62_BK 64
#define I62_STR (I62_BK + 8)   // the parent's staged row (ASTR = WSTR = 72 B)
#define I62_KG (I62_BK / 16)   // NVFP4 16-k groups a slab

typedef __attribute__((address_space(1))) const unsigned char g_u8;
typedef __attribute__((address_space(1))) const unsigned int g_u32;
typedef __attribute__((address_space(1))) const i62_u4 g_u4;

__device__ __forceinline__ void i62_lds_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// e2m1 (permute_w nibbles) -> e4m3 bytes in k order: the parent's kMag[0] LUT + sign, byte for byte.
__device__ __forceinline__ i62_u2 i62_e2m1_fp8(unsigned int wq) {
  const unsigned int t0 = 0x3c383000u, t1 = 0x4c484440u;
  const unsigned int ev = wq & 0x0F0F0F0Fu;
  const unsigned int od = (wq >> 4) & 0x0F0F0F0Fu;
  const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u) | ((ev & 0x08080808u) << 4);
  const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u) | ((od & 0x08080808u) << 4);
  return i62_u2{__builtin_amdgcn_perm(bo, be, 0x05010400u), __builtin_amdgcn_perm(bo, be, 0x07030602u)};
}

// `e` (CV 1): the e2m1 nibble placed in an e4m3 byte as-is (bits 4..2 = e1 e0 m, the sign at bit 7) = EXACTLY
// 2^-6 x the e2m1 value (code 1 = 0.5 -> the e4m3 subnormal 2^-7): 7 + 2 ops a W dword instead of the LUT's 8 + 2.
// The WMMA's 16-product sum, each FMA and the accumulator then carry exactly 2^-6 x the parent's values (every
// f32 here stays far above the subnormal range), and the epilogue's gs x 2^6 restores the parent's product --
// byte-equal IFF the WMMA treats an e4m3 subnormal B element exactly: it does (INK62's oracle, 84 / 84 cases), but
// the convert is not the limit (twin tied), so no symbol instantiates it; kept as the measured law.
__device__ __forceinline__ i62_u2 i62_e2m1_fp8_x64(unsigned int wq) {
  const unsigned int be = ((wq << 2) & 0x1C1C1C1Cu) | ((wq << 4) & 0x80808080u);   // even nibbles (k even)
  const unsigned int bo = ((wq >> 2) & 0x1C1C1C1Cu) | (wq & 0x80808080u);          // odd nibbles, sign in place
  return i62_u2{__builtin_amdgcn_perm(bo, be, 0x05010400u), __builtin_amdgcn_perm(bo, be, 0x07030602u)};
}

// DG: diagnostic bits for the census twin ONLY (never instantiated in this object): 1 = the scale FMA chain
// replaced by the WMMA's own C (no VALU), 2 = W staged without the e2m1 -> fp8 convert, 4 = no WMMA (the FMAs
// read the A fragment), 8 = no global loads past the first slab, 16 = the PRICE of
// a numerics form (the weight requantized to e4m3 by a per-slab ratio at staging + the slab's four WMMAs chained in
// C + one FMA a slab). None is a law.
template <int WM, int WN, int TMW, int TNW, bool GATHER, int DG, int CV = 0>
__device__ __forceinline__ void ink62_nv_impl(
    const unsigned char *__restrict__ A, const unsigned long long *__restrict__ xptr,
    const int *__restrict__ tile_expert, const int *__restrict__ tile_row0, const int *__restrict__ tile_rows,
    const int *__restrict__ row_tok, const int *__restrict__ meta, const float *__restrict__ As,
    __bf16 *__restrict__ C, int N, int K, int gsplit) {
  constexpr int NT = WM * WN * 32;
  static_assert(NT == 256, "eight waves");
  constexpr int BM = WM * TMW * 16, BN = WN * TNW * 16;
  constexpr int NA = BM * I62_BK / (NT * 16);            // 16-byte A chunks a thread a slab
  constexpr int NWG = (BN / 16) * I62_KG * 32 / (NT * 4); // 4-slot W groups a thread a slab
  static_assert(NA >= 1 && NWG >= 1, "the staging must tile the block");
  __shared__ __attribute__((aligned(16))) unsigned char sA[BM * I62_STR];
  __shared__ __attribute__((aligned(16))) unsigned char sW[BN * I62_STR];
  __shared__ __attribute__((aligned(16))) unsigned char sS[BN * I62_KG];
  __shared__ int s_arow[BM];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN, wn = wave % WN;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int tile = blockIdx.y;
  if (tile >= meta[0]) return;
  const int texp = tile_expert[tile], row0 = tile_row0[tile];
  const int M = tile_rows[tile];
  g_u32 *__restrict__ Wp = (g_u32 *)(uintptr_t)xptr[3 * (size_t)texp + 0];
  g_u8 *__restrict__ Ws = (g_u8 *)(uintptr_t)xptr[3 * (size_t)texp + 1];
  const float *__restrict__ Gs = (const float *)(uintptr_t)xptr[3 * (size_t)texp + 2];
  // this wave's live 16-row fragments (wave-uniform), the parent's SC bit 0
  const int nlf = __builtin_amdgcn_readfirstlane(min(TMW, max(0, (M - wm * TMW * 16 + 15) / 16)));
  for (int r = tid; r < BM; r += NT) {
    const int rr = r < M - 1 ? r : M - 1;   // clamp, never predicate
    s_arow[r] = GATHER ? row_tok[row0 + rr] : row0 + rr;
  }
  __syncthreads();
  const int n0 = blockIdx.x * BN;
  const int ksteps_g = K / 16;
  // A: thread chunk i = rows (i * NT + tid) * 16 / BK, the parent's staging map
  g_u4 *abase[NA];
  int aoff[NA];
#pragma unroll
  for (int i = 0; i < NA; ++i) {
    const int idx = i * NT * 16 + tid * 16, r = idx / I62_BK, c = idx % I62_BK;
    abase[i] = (g_u4 *)((g_u8 *)A + (size_t)s_arow[r] * K + c);
    aoff[i] = r * I62_STR + c;
  }
  // W: slot group g = 4 consecutive columns at one (kstep, k half), the parent's WPERM map
  g_u32 *wsrc[NWG];
  g_u8 *ssrc[NWG];
  int woff[NWG], soff[NWG];
#pragma unroll
  for (int g = 0; g < NWG; ++g) {
    const int sl = g * NT * 4 + tid * 4;
    const int lane_ = sl & 31, rest = sl >> 5;
    const int kst = rest % I62_KG, ntl = rest / I62_KG;
    const int r = ntl * 16 + (lane_ & 15), kloc = kst * 16 + (lane_ >> 4) * 8;
    const int gn = n0 + r;
    const int gc = gn < N - 4 ? gn : N - 4;
    const int lanec = (lane_ & 16) | (gc & 15);
    wsrc[g] = Wp + ((size_t)(gc >> 4) * ksteps_g + kst) * 32 + lanec;
    ssrc[g] = Ws + (size_t)kst * N + gc;
    woff[g] = r * I62_STR + kloc;
    soff[g] = r * I62_KG + kst;
  }
  i62_f8 acc[TMW][TNW];
#pragma unroll
  for (int i = 0; i < TMW; ++i)
#pragma unroll
    for (int j = 0; j < TNW; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

  i62_u4 aR[NA], wR[NWG];
  unsigned int sR[NWG];
#pragma unroll
  for (int i = 0; i < NA; ++i) aR[i] = *abase[i];
#pragma unroll
  for (int g = 0; g < NWG; ++g) {
    wR[g] = *(g_u4 *)wsrc[g];
    sR[g] = *(g_u32 *)ssrc[g];   // 4 columns' E4M3 bytes at this kstep (N % 4 == 0: aligned)
  }
  for (int k0 = 0; k0 < K; k0 += I62_BK) {
    // ---- stage the slab held in registers
#pragma unroll
    for (int i = 0; i < NA; ++i) *(i62_u4 *)&sA[aoff[i]] = aR[i];
#pragma unroll
    for (int g = 0; g < NWG; ++g) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        sS[soff[g] + q * I62_KG] = (unsigned char)(sR[g] >> (8 * q));
        i62_u2 v = (DG & 2) ? i62_u2{wR[g][q], wR[g][q] >> 4}
                   : CV ? i62_e2m1_fp8_x64(wR[g][q]) : i62_e2m1_fp8(wR[g][q]);
        if constexpr ((DG & 16) != 0) {   // the price of a requant at staging: fp8 -> f32 x ratio -> fp8
          const float r = __builtin_amdgcn_cvt_f32_fp8((int)(sR[g] >> (8 * q)), 0);
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const float f[4] = {__builtin_amdgcn_cvt_f32_fp8((int)v[h], 0) * r, __builtin_amdgcn_cvt_f32_fp8((int)v[h], 1) * r,
                                __builtin_amdgcn_cvt_f32_fp8((int)v[h], 2) * r, __builtin_amdgcn_cvt_f32_fp8((int)v[h], 3) * r};
            int o = __builtin_amdgcn_cvt_pk_fp8_f32(f[0], f[1], 0, false);
            o = __builtin_amdgcn_cvt_pk_fp8_f32(f[2], f[3], o, true);
            v[h] = (unsigned int)o;
          }
        }
        *(i62_u2 *)&sW[woff[g] + q * I62_STR] = v;
      }
    }
    i62_lds_barrier();
    // ---- the next slab into registers while this one computes
    if (k0 + I62_BK < K && ((DG & 8) == 0 || k0 == 0)) {
      const int kn = k0 + I62_BK, ksn = kn / 16;
#pragma unroll
      for (int i = 0; i < NA; ++i) aR[i] = *(g_u4 *)((g_u8 *)abase[i] + kn);
#pragma unroll
      for (int g = 0; g < NWG; ++g) {
        wR[g] = *(g_u4 *)(wsrc[g] + ksn * 32);
        sR[g] = *(g_u32 *)(ssrc[g] + (size_t)ksn * N);
      }
    }
    i62_f8 slab[TMW][TNW];   // DG 16: the slab's WMMA chain (one FMA a slab)
    unsigned int scw[TNW];
#pragma unroll
    for (int j = 0; j < TNW; ++j) scw[j] = *(const unsigned int *)&sS[(wn * TNW * 16 + j * 16 + col) * I62_KG];
#pragma unroll
    for (int step = 0; step < I62_KG; ++step) {
      const int kk = step * 16 + kb8;
      i62_i2 af[TMW], wf[TNW];
      float sv[TNW];
#pragma unroll
      for (int i = 0; i < TMW; ++i) {
        if (i < nlf) {
          const unsigned char *p = &sA[(wm * TMW * 16 + i * 16 + col) * I62_STR + kk];
          af[i][0] = *(const int *)p;
          af[i][1] = *(const int *)(p + 4);
        }
      }
#pragma unroll
      for (int j = 0; j < TNW; ++j) {
        const unsigned char *p = &sW[(wn * TNW * 16 + j * 16 + col) * I62_STR + kk];
        wf[j][0] = *(const int *)p;
        wf[j][1] = *(const int *)(p + 4);
        sv[j] = __builtin_amdgcn_cvt_f32_fp8((int)(scw[j] >> (8 * step)), 0);   // step is unrolled
      }
      const i62_f8 z = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int i = 0; i < TMW; ++i) {
        if (i < nlf) {
#pragma unroll
          for (int j = 0; j < TNW; ++j) {
            if constexpr ((DG & 16) != 0) {
              slab[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf[j], step == 0 ? z : slab[i][j]);
              if (step == I62_KG - 1) {
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[i][j][e] = __builtin_fmaf(slab[i][j][e], sv[j], acc[i][j][e]);
              }
            } else if constexpr ((DG & 1) != 0) {
              acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf[j], acc[i][j]);
            } else if constexpr ((DG & 4) != 0) {
              const float a0 = __builtin_bit_cast(float, af[i][0] ^ wf[j][1]);
#pragma unroll
              for (int e = 0; e < 8; ++e) acc[i][j][e] = __builtin_fmaf(a0, sv[j], acc[i][j][e]);
            } else {
              const i62_f8 tt = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf[j], z);
#pragma unroll
              for (int e = 0; e < 8; ++e) acc[i][j][e] = __builtin_fmaf(tt[e], sv[j], acc[i][j][e]);
            }
          }
        }
      }
    }
    i62_lds_barrier();
  }

  // Epilogue: the parent's (bf16((acc * gs) * As[row]), rows < M only).
  float rf[TNW];
#pragma unroll
  for (int j = 0; j < TNW; ++j) {
    const int n = n0 + wn * TNW * 16 + j * 16 + col;
    rf[j] = (n < N) ? Gs[n < gsplit ? 0 : 1] : 0.f;
    if constexpr (CV != 0) rf[j] = rf[j] * 64.f;   // exact (a power of two): acc here = 2^-6 x the parent's
  }
#pragma unroll
  for (int i = 0; i < TMW; ++i) {
    if (i < nlf) {
#pragma unroll
      for (int j = 0; j < TNW; ++j) {
        const int n = n0 + wn * TNW * 16 + j * 16 + col;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int m = wm * TMW * 16 + i * 16 + kb8 + e;
          if (m < M && n < N) C[(size_t)(row0 + m) * N + n] = (__bf16)(acc[i][j][e] * rf[j] * As[s_arow[m]]);
        }
      }
    }
  }
}

#define INK62_NV(name, WM, WN, TMW, TNW, G, DG, ...)                                                        \
  extern "C" __global__ __launch_bounds__(256) void name(                                                \
      const unsigned char *__restrict__ A, const unsigned long long *__restrict__ xptr,                  \
      const int *__restrict__ tile_expert, const int *__restrict__ tile_row0,                            \
      const int *__restrict__ tile_rows, const int *__restrict__ row_tok, const int *__restrict__ meta,  \
      const float *__restrict__ As, float *__restrict__ P, int *__restrict__ cnt, __bf16 *__restrict__ C, \
      int N, int K, int Pmax, int gsplit) {                                                              \
    (void)P; (void)cnt; (void)Pmax;                                                                      \
    ink62_nv_impl<WM, WN, TMW, TNW, G, DG, ##__VA_ARGS__>(A, xptr, tile_expert, tile_row0, tile_rows, row_tok, meta,    \
                                           As, C, N, K, gsplit);                                         \
  }
#ifndef INK62_DIAG
INK62_NV(ink62_nv_p_g1, 4, 2, 1, 4, true, 0)
INK62_NV(ink62_nv_p_g0, 4, 2, 1, 4, false, 0)
INK62_NV(ink62_nv_w_g1, 1, 8, 4, 2, true, 0)
INK62_NV(ink62_nv_w_g0, 1, 8, 4, 2, false, 0)
INK62_NV(ink62_nv_t_g1, 2, 4, 4, 2, true, 0)
INK62_NV(ink62_nv_t_g0, 2, 4, 4, 2, false, 0)
#endif
