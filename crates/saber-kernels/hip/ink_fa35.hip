// Inkling 2 (inkling2_loop) prefill attention on the matrix cores, INK35 (docs/INKLING2.md section 9
// INK35). Saber's own.
//
//   ink_fa35_wm  the prefill chunk's attention (local window or global, the band bias, the sink) as a
//       flash attention on v_wmma: 64 consecutive query rows of ONE sequence a workgroup (4 waves x 16
//       rows), keys in tiles of 32 staged once in LDS for the 4 waves.
//         * S = q . k on v_wmma_f32_16x16x16_bf16 (q and k are bf16 as stored: every product exact,
//           f32 accumulation) x 1/sqrt(128);
//         * the band bias r . P[:, dist] (dist = pos - j < extent) as G = R . P over the 48 distances a
//           (16-row, 32-key) block can see, on the same WMMA with R split hi + lo bf16 (P is bf16 as
//           stored: ~f32 accuracy), staged through LDS and gathered per (row, key);
//         * the online softmax in f32 (log2 domain, exp2), the row stats across the lane pair;
//         * PV on v_wmma_f32_16x16x16_f16: p in f16 (|p| <= 1), v bf16 -> f16 (exact in v's range);
//         * the sink epilogue as v1's: o = acc * c / (l c + exp(sink - mf)).
//       A NUMERICS arm (the summation order and p's rounding differ from v1; the first-token KL table
//       is its gate). Grid [ceil(T / 64), H], 128 threads, the shipped v1 argument list.
#include <hip/hip_runtime.h>

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));

#define WM_HD 128
#define WM_PAGE 64
#define WM_ROWS 64
#define WM_KT 32
#define KS_LD 136   // bf16 stride of a staged K row (128 + 8)
#define VT_LD 40    // f16 stride of a staged V^T row (32 keys + 8)
#define G_LD 52     // f32 stride of a G row (48 distances + 4)

__device__ __forceinline__ unsigned short wm_bf_rne(float x) {
  const unsigned u = __builtin_bit_cast(unsigned, x);
  return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float wm_bf(unsigned short u) {
  return __builtin_bit_cast(float, static_cast<unsigned>(u) << 16);
}

// LEAN (ISA51, wm32's form): two address forms with the same values, so the same loads, the same LDS
// tile and the same arithmetic (byte-equal by construction):
//   * a staged tile's 32 keys start at a multiple of 32 (jbeg and the step are 32-aligned), so they
//     sit in ONE 64-key page -- its row is read once a tile, not once a chunk;
//   * the band-bias gather index dd = l16 - 16 s - 8 hh - e + 31 lies in [0, 46] for every lane, s
//     and e, so the min / max clamp never binds and the 16 reads are one per-lane base + constant
//     offsets.
// Without it the compiler hoisted 16 gather addresses + 8 page rows out of the tile loop and wm32
// (8 staging chunks a thread) spilled 7 VGPRs (256 VGPRs, 32 B scratch; INK35 / CONS40).
template <int NW, bool LEAN>
__device__ __forceinline__ void fa35_body(
    const unsigned short* __restrict__ q, int ld_q, const unsigned short* __restrict__ k_pool,
    const unsigned short* __restrict__ v_pool, const int* __restrict__ table, int table_ld,
    const int* __restrict__ row_seq, const int* __restrict__ row_pos, const float* __restrict__ rp,
    int ld_rp, const unsigned short* __restrict__ P, const float* __restrict__ sink, int ld_sink,
    float* __restrict__ o, int ld_o, int T, int H, int KV, int extent, int window) {
  __shared__ __attribute__((aligned(16))) unsigned short Ks[WM_KT][KS_LD];
  __shared__ __attribute__((aligned(16))) _Float16 Vt[WM_HD][VT_LD];
  __shared__ __attribute__((aligned(16))) float Gs[NW][16][G_LD];
  constexpr int ROWS = 16 * NW, NT = 32 * NW, NC = (WM_KT * 16) / NT;
  const int tid = threadIdx.x, lane = tid & 31;
  const int w = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int l16 = lane & 15, hh = lane >> 4;
  const int h = blockIdx.y, kvh = h / (H / KV);
  const int C = KV * WM_HD;
  const int t0 = blockIdx.x * ROWS;
  const int tl = min(t0 + ROWS, T) - 1;   // the workgroup's last row
  const int sq = row_seq[t0];
  const int pos_first = row_pos[t0], pos_last = row_pos[tl];
  // this wave's rows
  const int tw0 = t0 + 16 * w;
  const bool wlive = tw0 <= tl;
  const int r = min(tw0 + l16, tl);
  const int prow = row_pos[r];
  const int plo = window > 0 ? max(0, prow - window + 1) : 0;
  const int pw0 = row_pos[min(tw0, tl)], pw15 = row_pos[min(tw0 + 15, tl)];
  const int lw0 = window > 0 ? max(0, pw0 - window + 1) : 0;
  // q fragments (Y operand: row l16, k = 16 ks + 8 hh .. + 7)
  bf16x8 qf[8];
  {
    const uint4* qp = reinterpret_cast<const uint4*>(q + (long)r * ld_q + h * WM_HD + 8 * hh);
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) qf[ks] = __builtin_bit_cast(bf16x8, qp[2 * ks]);
  }
  // R fragments (hi + lo bf16 of the f32 r_post, k = feature 8 hh .. + 7)
  bf16x8 rhi, rlo;
  {
    unsigned short a[8], b[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float x = rp[(long)r * ld_rp + h * 16 + 8 * hh + e];
      a[e] = wm_bf_rne(x);
      b[e] = wm_bf_rne(x - wm_bf(a[e]));
    }
    rhi = __builtin_bit_cast(bf16x8, a);
    rlo = __builtin_bit_cast(bf16x8, b);
  }
  const float LOG2E = 1.4426950408889634f;
  const float scale = 0.08838834764831845f;
  float m = -INFINITY, l = 0.f;
  f32x8 oacc[8];
#pragma unroll
  for (int dt = 0; dt < 8; ++dt) oacc[dt] = f32x8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  const int lo0 = window > 0 ? max(0, pos_first - window + 1) : 0;
  const int jbeg = lo0 & ~(WM_KT - 1);
  // staging: chunk c = tid + 128 i (i 0..3): key c >> 4, part c & 15 (8 values)
  uint4 kr[NC], vr[NC];
  static_assert(WM_PAGE % WM_KT == 0, "a staged tile must sit in one page (LEAN)");
  // LEAN: chunk i's key = (tid >> 4) + (NT / 16) i and part = tid & 15 (NT % 16 == 0), so a chunk's
  // element offset is the thread's tile base + i (NT / 16) C, a wave-uniform step
  static_assert(NT % 16 == 0, "LEAN chunk map");
  const int kb = tid >> 4, pb = tid & 15;
  auto load = [&](int j0) {
    if (LEAN) {
      const long prow0 = (long)table[(long)sq * table_ld + j0 / WM_PAGE] * WM_PAGE + j0 % WM_PAGE;
      const unsigned short* kp = k_pool + (prow0 + kb) * C + kvh * WM_HD + pb * 8;
      const unsigned short* vp = v_pool + (prow0 + kb) * C + kvh * WM_HD + pb * 8;
#pragma unroll
      for (int i = 0; i < NC; ++i) {
        if (j0 + kb + (NT / 16) * i <= pos_last) {
          const long step = (long)((NT / 16) * i) * C;
          kr[i] = *reinterpret_cast<const uint4*>(kp + step);
          vr[i] = *reinterpret_cast<const uint4*>(vp + step);
        } else {
          kr[i] = uint4{0u, 0u, 0u, 0u};
          vr[i] = uint4{0u, 0u, 0u, 0u};
        }
      }
      return;
    }
#pragma unroll
    for (int i = 0; i < NC; ++i) {
      const int c = tid + NT * i, key = c >> 4, part = c & 15;
      const int j = j0 + key;
      if (j <= pos_last) {
        const long row = (long)table[(long)sq * table_ld + j / WM_PAGE] * WM_PAGE + j % WM_PAGE;
        const long off = row * C + kvh * WM_HD + part * 8;
        kr[i] = *reinterpret_cast<const uint4*>(k_pool + off);
        vr[i] = *reinterpret_cast<const uint4*>(v_pool + off);
      } else {
        kr[i] = uint4{0u, 0u, 0u, 0u};
        vr[i] = uint4{0u, 0u, 0u, 0u};
      }
    }
  };
  auto store = [&]() {
#pragma unroll
    for (int i = 0; i < NC; ++i) {
      const int c = tid + NT * i;
      const int key = LEAN ? kb + (NT / 16) * i : c >> 4, part = LEAN ? pb : c & 15;
      *reinterpret_cast<uint4*>(&Ks[key][part * 8]) = kr[i];
      const unsigned wv[4] = {vr[i].x, vr[i].y, vr[i].z, vr[i].w};
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        Vt[part * 8 + 2 * u][key] = (_Float16)wm_bf((unsigned short)(wv[u] & 0xFFFFu));
        Vt[part * 8 + 2 * u + 1][key] = (_Float16)wm_bf((unsigned short)(wv[u] >> 16));
      }
    }
  };
  load(jbeg);
  for (int j0 = jbeg; j0 <= pos_last; j0 += WM_KT) {
    __syncthreads();
    store();
    __syncthreads();
    if (j0 + WM_KT <= pos_last) load(j0 + WM_KT);
    // a wave whose rows see no key of this tile skips it (wave-uniform)
    if (!wlive || j0 > pw15 || j0 + WM_KT - 1 < lw0) continue;
    // S (D: row l16, key 16 s + 8 hh + e)
    f32x8 sc[2];
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      f32x8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const bf16x8 kf =
            *reinterpret_cast<const bf16x8*>(&Ks[16 * s + l16][16 * ks + 8 * hh]);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(kf, qf[ks], acc);
      }
      sc[s] = acc;
    }
    // the band bias: G[row][dd] = r(row) . P[:, dbase + dd], dd 0..47 (wave-uniform test)
    const int dbase = pw0 - j0 - (WM_KT - 1);
    const bool bias = dbase < extent;
    if (bias) {
#pragma unroll
      for (int g = 0; g < 3; ++g) {
        const int dist = dbase + 16 * g + l16;
        unsigned short pv[8];
#pragma unroll
        for (int e = 0; e < 8; ++e)
          pv[e] = (dist >= 0 && dist < extent) ? P[(long)(8 * hh + e) * extent + dist] : 0;
        const bf16x8 pf = __builtin_bit_cast(bf16x8, pv);
        f32x8 gacc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        gacc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(pf, rhi, gacc);
        gacc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(pf, rlo, gacc);
        float4* gp = reinterpret_cast<float4*>(&Gs[w][l16][16 * g + 8 * hh]);
        gp[0] = float4{gacc[0], gacc[1], gacc[2], gacc[3]};
        gp[1] = float4{gacc[4], gacc[5], gacc[6], gacc[7]};
      }
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
      __builtin_amdgcn_wave_barrier();
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    }
    float mx = -INFINITY;
#pragma unroll
    for (int s = 0; s < 2; ++s) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int kj = j0 + 16 * s + 8 * hh + e;
        float v = sc[s][e] * scale;
        if (bias) {
          if (LEAN) {
            // dd = (l16 - 8 hh + 8) + (23 - 16 s - e), both terms >= 0
            v = v + (&Gs[w][l16][l16 - 8 * hh + 8])[23 - 16 * s - e];
          } else {
            const int dd = min(max(l16 - 16 * s - 8 * hh - e + (WM_KT - 1), 0), 47);
            v = v + Gs[w][l16][dd];
          }
        }
        v = (kj <= prow && kj >= plo) ? v * LOG2E : -INFINITY;
        sc[s][e] = v;
        mx = fmaxf(mx, v);
      }
    }
    mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
    const float mn = fmaxf(m, mx);
    const float mu = (mn == -INFINITY) ? 0.f : mn;
    const float corr = exp2f(m - mu);
    float rs = 0.f;
    f16x8 pf[2];
#pragma unroll
    for (int s = 0; s < 2; ++s) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float p = exp2f(sc[s][e] - mu);
        rs += p;
        pf[s][e] = (_Float16)p;
      }
    }
    rs += __shfl_xor(rs, 16, 32);
    l = l * corr + rs;
    m = mn;
#pragma unroll
    for (int dt = 0; dt < 8; ++dt) oacc[dt] *= corr;
    // PV (D: row l16, d = 16 dt + 8 hh + e)
#pragma unroll
    for (int s = 0; s < 2; ++s) {
#pragma unroll
      for (int dt = 0; dt < 8; ++dt) {
        const f16x8 vf = *reinterpret_cast<const f16x8*>(&Vt[16 * dt + l16][16 * s + 8 * hh]);
        oacc[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vf, pf[s], oacc[dt]);
      }
    }
  }
  const int row = tw0 + l16;
  if (!wlive || row > tl) return;
  const float sk = sink[(long)row * ld_sink + h] * LOG2E;
  const float mf = fmaxf(m, sk);
  const float c = exp2f(m - mf);
  const float lf = l * c + exp2f(sk - mf);
  const float f = c / lf;
  float* op = o + (long)row * ld_o + h * WM_HD + 8 * hh;
#pragma unroll
  for (int dt = 0; dt < 8; ++dt) {
    float4* p4 = reinterpret_cast<float4*>(op + 16 * dt);
    p4[0] = float4{oacc[dt][0] * f, oacc[dt][1] * f, oacc[dt][2] * f, oacc[dt][3] * f};
    p4[1] = float4{oacc[dt][4] * f, oacc[dt][5] * f, oacc[dt][6] * f, oacc[dt][7] * f};
  }
}

#define FA35_KERNEL(NAME, NW, LN)                                                                    \
  extern "C" __global__ __launch_bounds__(32 * NW) void NAME(                                      \
      const unsigned short* __restrict__ q, int ld_q, const unsigned short* __restrict__ k_pool,   \
      const unsigned short* __restrict__ v_pool, const int* __restrict__ table, int table_ld,      \
      const int* __restrict__ row_seq, const int* __restrict__ row_pos, const float* __restrict__ rp, \
      int ld_rp, const unsigned short* __restrict__ P, const float* __restrict__ sink, int ld_sink, \
      float* __restrict__ o, int ld_o, int T, int H, int KV, int extent, int window) {             \
    fa35_body<NW, LN>(q, ld_q, k_pool, v_pool, table, table_ld, row_seq, row_pos, rp, ld_rp, P, sink,  \
                  ld_sink, o, ld_o, T, H, KV, extent, window);                                     \
  }
FA35_KERNEL(ink_fa35_wm, 4, false)
// ISA51: wm32 runs the LEAN address forms (was 256 VGPRs + 7 spilled, 32 B scratch).
FA35_KERNEL(ink_fa35_wm32, 2, true)
// CONS40 `--ink-p40 fa128`: the same body on 128 rows a workgroup (8 waves): every row's tiles, order and
// arithmetic are wm's (byte-equal by construction), the K / V tiles staged once for twice the rows.
FA35_KERNEL(ink_fa35_wm128, 8, false)

// ------------------------------------------------------------------------------------------------
//   ink35_gemm_bws_s6  ink_p29's bw GEMM (K = 6 slices of 768, slices summed in order) with the f32
//       activations split hi + lo bf16 (two WMMAs a 16-k step: x = hi + lo to ~2^-16 relative; the
//       weight is bf16 as stored, every product exact) -> ~f32 accuracy on the matrix cores. INK35's
//       message-gate z (item `mzs`, a NUMERICS arm: the f32 order changes, not the operand rounding).
//       grid [ceil(N / 16), ceil(M / 16)], block 32 * 6.
__device__ __forceinline__ void bws_cvt(const float4& a0, const float4& a1, bf16x8& hi, bf16x8& lo) {
  const float v[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
  unsigned short h8[8], l8[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    h8[e] = wm_bf_rne(v[e]);
    l8[e] = wm_bf_rne(v[e] - wm_bf(h8[e]));
  }
  hi = __builtin_bit_cast(bf16x8, h8);
  lo = __builtin_bit_cast(bf16x8, l8);
}

extern "C" __global__ __launch_bounds__(192) void ink35_gemm_bws_s6(
    const float* __restrict__ a, int lda, const unsigned short* __restrict__ w, int ldw,
    float* __restrict__ c, int ldc, int M, int N, int K, int acc) {
  constexpr int S = 6;
  __shared__ float red[(S - 1) * 256];
  const int lane = threadIdx.x & 31;
  const int slice = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const int mt = blockIdx.y;
  const int n0 = blockIdx.x * 16;
  const int l16 = lane & 15, hh = lane >> 4;
  const int m = min(mt * 16 + l16, M - 1);
  const float* ap = a + (long)m * lda + 8 * hh;
  const unsigned short* wp = w + (long)min(n0 + l16, N - 1) * ldw + 8 * hh;
  const int ks = K / S;
  const int k_lo = slice * ks;
  f32x8 acc8 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int k = k_lo; k < k_lo + ks; k += 16) {
    const uint4 wv = *reinterpret_cast<const uint4*>(wp + k);
    const float4 a0 = *reinterpret_cast<const float4*>(ap + k);
    const float4 a1 = *reinterpret_cast<const float4*>(ap + k + 4);
    bf16x8 hi, lo;
    bws_cvt(a0, a1, hi, lo);
    const bf16x8 wf = __builtin_bit_cast(bf16x8, wv);
    acc8 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(wf, hi, acc8);
    acc8 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(wf, lo, acc8);
  }
  if (slice > 0) {
    float4* o = reinterpret_cast<float4*>(red + (slice - 1) * 256 + lane * 8);
    o[0] = float4{acc8[0], acc8[1], acc8[2], acc8[3]};
    o[1] = float4{acc8[4], acc8[5], acc8[6], acc8[7]};
  }
  __syncthreads();
  if (slice != 0) return;
#pragma unroll
  for (int s = 1; s < S; ++s) {
    const float4* o = reinterpret_cast<const float4*>(red + (s - 1) * 256 + lane * 8);
    const float4 x0 = o[0], x1 = o[1];
    acc8[0] += x0.x; acc8[1] += x0.y; acc8[2] += x0.z; acc8[3] += x0.w;
    acc8[4] += x1.x; acc8[5] += x1.y; acc8[6] += x1.z; acc8[7] += x1.w;
  }
  const int mo = mt * 16 + l16;
  if (mo >= M) return;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int n = n0 + 8 * hh + e;
    if (n < N) {
      float* op = c + (long)mo * ldc + n;
      *op = acc ? *op + acc8[e] : acc8[e];
    }
  }
}
