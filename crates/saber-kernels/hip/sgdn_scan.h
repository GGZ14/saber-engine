// Saber gated-delta-net kernels, CLEANROOM-2 (CR2-3c): the WY chunk scan of the prefill (spec 2.8), one body
// with three compile-time gate policies:
//   kGateSplit  -- the certified form: gates split about the chunk midpoint and clamped at 80 nats;
//   kGateExact  -- exact gates, the state operand weighted as each fragment is read ("xg0");
//   kGateExactW -- exact gates, the weighted state operand written once into LDS ("xg" / "xg2").
//
// Workgroup = (v half, v head, sequence), eight waves; the state of the 64-wide v half is held in registers
// transposed (C[m = k][n = v]), four 16x16 tiles a wave. Per 64-token chunk: U = K S^T, O = S Q^T and P = K Q^T
// (bf16 WMMA chains over the eight k tiles), W = beta (V - e^g U), the causal sA, V' = A W, O continued by
// V' sA^T, then S = e^(..) (S + K^T V'). Every chain is issued in the order the spec gives, operands in the
// spec's fragment convention; every bf16 pack is away16 except the exact forms' state operand (rne16).
#pragma once
#include "sgdn_prims.h"

namespace sgdn {

enum ScanGate { kGateSplit = 0, kGateExact = 1, kGateExactW = 2 };

constexpr int kScanKP = 132;   // LDS pitch of the [64][128] k and state tiles (bf16)
constexpr int kScanCP = 68;    // LDS pitch of the [64][64] chunk tiles (bf16)

struct ScanLds {
  u16 k[kChunk][kScanKP];   // the chunk's K, [token][k]
  u16 a[kChunk][kScanCP];   // the causal sA, [i][j]
  u16 s[kChunk][kScanKP];   // the state, [v][k]
  u16 w[kChunk][kScanCP];   // W, [v][token]; the exact-W form's weighted V' after use
  u16 d[kChunk][kScanCP];   // V as [token][v], then V' as [v][token]
  float ge[kChunk], gv[kChunk], rs[kChunk], bt[kChunk];
};
static_assert(sizeof(ScanLds) == 60928, "the scan's LDS budget");

template <int KIND, int GATE>
__device__ __forceinline__ void chunk_scan(const u16* __restrict__ q, const u16* __restrict__ k,
                                           const u16* __restrict__ v, const u16* __restrict__ a_mat,
                                           const float* __restrict__ gcs, const float* __restrict__ beta,
                                           const void* __restrict__ h0v, u16* __restrict__ o, void* __restrict__ htv,
                                           const int* __restrict__ cu, int H, int Hg, float scale) {
  typedef typename StateElem<KIND>::type ST;
  __shared__ __attribute__((aligned(16))) ScanLds L;
  const int bv = blockIdx.x, hv = blockIdx.y, nq = blockIdx.z;
  const int v0 = 64 * bv;
  const int hq = hv / (H / Hg);
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int c = lane & 15, h = lane >> 4;
  const int mt = wave >> 1;            // token tile (U, O, W, sA, V')
  const int ntb = 2 * (wave & 1);      // first of the wave's two v tiles
  const int kt0 = 2 * (wave >> 1);     // state tiles: k tiles kt0, kt0+1 x v tiles vt0, vt0+1
  const int vt0 = 2 * (wave & 1);
  const int tb = cu[nq], te = cu[nq + 1];

  // ---- prologue: the state from h0, transposed into registers, and its bf16 copy in LDS
  const ST* h0 = (const ST*)h0v + ((long)nq * H + hv) * kDim * kDim;
  f32x8 St[4];
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    const int KT = kt0 + t / 2, VT = vt0 + t % 2;
    float tmp[8];
    state_load<KIND, 8>(tmp, h0 + (long)(v0 + 16 * VT + c) * kDim + 16 * KT + 8 * h);
#pragma unroll
    for (int e = 0; e < 8; ++e) St[t][e] = tmp[e];
    u16* sp = &L.s[16 * VT + c][16 * KT + 8 * h];
    *(u32x2*)sp = u32x2{away16x2(St[t][0], St[t][1]), away16x2(St[t][2], St[t][3])};
    *(u32x2*)(sp + 4) = u32x2{away16x2(St[t][4], St[t][5]), away16x2(St[t][6], St[t][7])};
  }

  // ---- staging of the chunk's K (4 x 16 B a thread) and V (2 x 16 B), issued one chunk ahead
  const long kpitch = (long)Hg * kDim, vpitch = (long)H * kDim;
  const u16* kbase = k + (long)hq * kDim;
  const u16* vbase = v + (long)hv * kDim + v0;
  u32x4 pk[4], pv[2];
  float pg = 0.0f, pb = 0.0f, pg0 = 0.0f, pgl = 0.0f;   // the chunk's gates (thread t < 64: token t), first, last
  auto stage_load = [&](int cs) {
    const int lim = min(kChunk, te - cs) - 1;
    pg0 = gcs[(long)cs * H + hv];
    pgl = gcs[(long)(cs + lim) * H + hv];
    if (tid <= lim) {
      pg = gcs[(long)(cs + tid) * H + hv];
      pb = beta[(long)(cs + tid) * H + hv];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int s = tid + 256 * i, r = min(s >> 4, lim), sg = s & 15;
      pk[i] = *(const u32x4*)(kbase + (long)(cs + r) * kpitch + 8 * sg);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int s = tid + 256 * i, r = min(s >> 3, lim), sg = s & 7;
      pv[i] = *(const u32x4*)(vbase + (long)(cs + r) * vpitch + 8 * sg);
    }
  };
  if (tb < te) stage_load(tb);
  lds_sync();

  for (int c0 = tb; c0 < te; c0 += kChunk) {
    const int nval = min(kChunk, te - c0), lim = nval - 1;
    // ---- step 1: gates and the staged chunk into LDS
    const float gl = pgl;
    const float cref = 0.5f * (pg0 + gl);
    const bool gok = tid < nval;
    const float ggr = gok ? pg : 0.0f;
    const float btr = gok ? pb : 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int s = tid + 256 * i, r = s >> 4, sg = s & 15;
      u16* dst = &L.k[r][8 * sg];
      *(u32x2*)dst = u32x2{pk[i].x, pk[i].y};
      *(u32x2*)(dst + 4) = u32x2{pk[i].z, pk[i].w};
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int s = tid + 256 * i, r = s >> 3, sg = s & 7;
      u16* dst = &L.d[r][8 * sg];
      *(u32x2*)dst = u32x2{pv[i].x, pv[i].y};
      *(u32x2*)(dst + 4) = u32x2{pv[i].z, pv[i].w};
    }
    if (c0 + kChunk < te) stage_load(c0 + kChunk);
    if (tid < kChunk) {
      const bool ok = gok;
      L.bt[tid] = btr;
      L.ge[tid] = E(ggr);
      if constexpr (GATE == kGateSplit) {
        L.gv[tid] = ok ? E(fminf(cref - ggr, 80.0f)) : 0.0f;
        L.rs[tid] = ok ? scale * E(fminf(ggr - cref, 80.0f)) : 0.0f;
      } else {
        L.gv[tid] = ok ? E(gl - ggr) : 0.0f;
        L.rs[tid] = ggr;
      }
    }
    lds_sync();   // (1)

    // ---- step 2: the three chains over the k tiles
    const int qi = min(16 * mt + c, lim);
    const u16* qrow = q + ((long)(c0 + qi) * Hg + hq) * kDim;
    f32x8 U[2], O[2], P[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) { U[i] = zero8(); O[i] = zero8(); P[i] = zero8(); }
#pragma unroll
    for (int sk = 0; sk < 8; ++sk) {
      const bf16x8 fq = frag_row(qrow + 16 * sk, h);
      const bf16x8 fk = frag_row(&L.k[16 * mt + c][16 * sk], h);
      const bf16x8 fs0 = frag_row(&L.s[16 * ntb + c][16 * sk], h);
      const bf16x8 fs1 = frag_row(&L.s[16 * (ntb + 1) + c][16 * sk], h);
      const bf16x8 fk0 = frag_row(&L.k[16 * ntb + c][16 * sk], h);
      const bf16x8 fk1 = frag_row(&L.k[16 * (ntb + 1) + c][16 * sk], h);
      U[0] = wmma(fk, fs0, U[0]);
      U[1] = wmma(fk, fs1, U[1]);
      O[0] = wmma(fs0, fq, O[0]);
      O[1] = wmma(fs1, fq, O[1]);
      P[0] = wmma(fk0, fq, P[0]);
      P[1] = wmma(fk1, fq, P[1]);
      // issue order (the scheduler's groups): per two k steps the four q loads, the ten LDS fragment loads, then
      // the twelve WMMAs, so independent chains interleave while the next step's operands are in flight
      if (sk & 1) {
        __builtin_amdgcn_sched_group_barrier(0x020, 4, 0);    // VMEM reads
        __builtin_amdgcn_sched_group_barrier(0x100, 10, 0);   // DS reads
        __builtin_amdgcn_sched_group_barrier(0x008, 12, 0);   // WMMA
      }
    }

    // the V' chain's A fragments (global), issued before the VALU work of steps 3-5 so it covers their latency
    bf16x8 fa6[4];
    {
      const int ai = min(16 * mt + c, lim);
      const u16* arow = a_mat + ((long)(c0 + ai) * H + hv) * kChunk;
#pragma unroll
      for (int sk = 0; sk < 4; ++sk) fa6[sk] = frag_row(arow + 16 * sk, h);
    }

    // ---- steps 3-5: O scaled, W, sA
    const float og = scale * L.ge[16 * mt + c];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int e = 0; e < 8; ++e) O[i][e] = O[i][e] * og;
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int vv = 16 * (ntb + i) + c;
      float wv[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int t = 16 * mt + 8 * h + e;
        wv[e] = L.bt[t] * (bf(L.d[t][vv]) - L.ge[t] * U[i][e]);
      }
      u16* wp = &L.w[vv][16 * mt + 8 * h];
      *(u32x2*)wp = u32x2{away16x2(wv[0], wv[1]), away16x2(wv[2], wv[3])};
      *(u32x2*)(wp + 4) = u32x2{away16x2(wv[4], wv[5]), away16x2(wv[6], wv[7])};
    }
    {
      const int ii = 16 * mt + c;
      float av[2][8];
      if constexpr (GATE == kGateSplit) {
        const float ri = L.rs[ii];
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int j = 16 * (ntb + i) + 8 * h + e;
            av[i][e] = (j <= ii) ? P[i][e] * ri : 0.0f;
          }
      } else {
        const float ri = L.rs[ii];
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int j = 16 * (ntb + i) + 8 * h + e;
            av[i][e] = (j <= ii) ? P[i][e] * (scale * E(fminf(ri - L.rs[j], 0.0f))) : 0.0f;
          }
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        u16* ap = &L.a[ii][16 * (ntb + i) + 8 * h];
        *(u32x2*)ap = u32x2{away16x2(av[i][0], av[i][1]), away16x2(av[i][2], av[i][3])};
        *(u32x2*)(ap + 4) = u32x2{away16x2(av[i][4], av[i][5]), away16x2(av[i][6], av[i][7])};
      }
    }
    lds_sync();   // (2)

    // ---- step 6: V' = A W (A from global), weighted (split form) or not (exact forms), into D as [v][token]
    {
      f32x8 Vp[2] = {zero8(), zero8()};
#pragma unroll
      for (int sk = 0; sk < 4; ++sk) {
#pragma unroll
        for (int i = 0; i < 2; ++i) Vp[i] = wmma(fa6[sk], frag_row(&L.w[16 * (ntb + i) + c][16 * sk], h), Vp[i]);
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        float dv[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          if constexpr (GATE == kGateSplit) dv[e] = Vp[i][e] * L.gv[16 * mt + 8 * h + e];
          else dv[e] = Vp[i][e];
        }
        u16* dp = &L.d[16 * (ntb + i) + c][16 * mt + 8 * h];
        *(u32x2*)dp = u32x2{away16x2(dv[0], dv[1]), away16x2(dv[2], dv[3])};
        *(u32x2*)(dp + 4) = u32x2{away16x2(dv[4], dv[5]), away16x2(dv[6], dv[7])};
      }
    }
    lds_sync();   // (3)

    // ---- step 7: O continued by V' sA^T, stored
#pragma unroll
    for (int sk = 0; sk < 4; ++sk) {
      const bf16x8 fa = frag_row(&L.a[16 * mt + c][16 * sk], h);
#pragma unroll
      for (int i = 0; i < 2; ++i) O[i] = wmma(frag_row(&L.d[16 * (ntb + i) + c][16 * sk], h), fa, O[i]);
    }
    {
      const int ti = 16 * mt + c;
      if (ti < nval) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          u16* op = o + ((long)(c0 + ti) * H + hv) * kDim + v0 + 16 * (ntb + i) + 8 * h;
          *(u32x4*)op = u32x4{away16x2(O[i][0], O[i][1]), away16x2(O[i][2], O[i][3]),
                              away16x2(O[i][4], O[i][5]), away16x2(O[i][6], O[i][7])};
        }
      }
    }

    // ---- step 8: the state
    if constexpr (GATE == kGateExactW) {
      // each lane's own 16 V' elements, weighted, into W at V''s [v][token] layout
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int vv = 16 * (ntb + i) + c;
        const u16* dp = &L.d[vv][16 * mt + 8 * h];
        const u32x2 r0 = *(const u32x2*)dp, r1 = *(const u32x2*)(dp + 4);
        const u32 raw[4] = {r0.x, r0.y, r1.x, r1.y};
        u32 wv[4];
#pragma unroll
        for (int e2 = 0; e2 < 4; ++e2) {
          const int t = 16 * mt + 8 * h + 2 * e2;
          const u32 lo = rne16(bf_lo(raw[e2]) * L.gv[t]);
          const u32 hi = rne16(bf_hi(raw[e2]) * L.gv[t + 1]);
          wv[e2] = lo | (hi << 16);
        }
        u16* wp = &L.w[vv][16 * mt + 8 * h];
        *(u32x2*)wp = u32x2{wv[0], wv[1]};
        *(u32x2*)(wp + 4) = u32x2{wv[2], wv[3]};
      }
      lds_sync();
    }
    {
      float ec, lam;
      if constexpr (GATE == kGateSplit) { ec = E(cref); lam = E(gl - cref); }
      else { ec = E(gl); lam = 1.0f; }
#pragma unroll
      for (int t = 0; t < 4; ++t)
#pragma unroll
        for (int e = 0; e < 8; ++e) St[t][e] = St[t][e] * ec;
#pragma unroll
      for (int sk = 0; sk < 4; ++sk) {
        bf16x8 fv[2];
#pragma unroll
        for (int vt = 0; vt < 2; ++vt) {
          const int vv = 16 * (vt0 + vt) + c;
          if constexpr (GATE == kGateSplit) {
            fv[vt] = frag_row(&L.d[vv][16 * sk], h);
          } else if constexpr (GATE == kGateExactW) {
            fv[vt] = frag_row(&L.w[vv][16 * sk], h);
          } else {
            const bf16x8 raw = frag_row(&L.d[vv][16 * sk], h);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              const int t = 16 * sk + 8 * (e >> 2) + 4 * h + (e & 3);
              fv[vt][e] = (short)rne16(bf((u16)raw[e]) * L.gv[t]);
            }
          }
        }
#pragma unroll
        for (int kt = 0; kt < 2; ++kt) {
          const bf16x8 fk = frag_col(&L.k[16 * sk][16 * (kt0 + kt) + c], kScanKP, h);
#pragma unroll
          for (int vt = 0; vt < 2; ++vt) St[2 * kt + vt] = wmma(fk, fv[vt], St[2 * kt + vt]);
        }
      }
      if constexpr (GATE == kGateSplit) {
#pragma unroll
        for (int t = 0; t < 4; ++t)
#pragma unroll
          for (int e = 0; e < 8; ++e) St[t][e] = St[t][e] * lam;
      }
      (void)lam;
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int KT = kt0 + t / 2, VT = vt0 + t % 2;
        u16* sp = &L.s[16 * VT + c][16 * KT + 8 * h];
        *(u32x2*)sp = u32x2{away16x2(St[t][0], St[t][1]), away16x2(St[t][2], St[t][3])};
        *(u32x2*)(sp + 4) = u32x2{away16x2(St[t][4], St[t][5]), away16x2(St[t][6], St[t][7])};
      }
    }
    lds_sync();   // (4)
  }

  // ---- the final state
  ST* ht = (ST*)htv + ((long)nq * H + hv) * kDim * kDim;
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    const int KT = kt0 + t / 2, VT = vt0 + t % 2;
    float tmp[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) tmp[e] = St[t][e];
    state_store<KIND, 8>(ht + (long)(v0 + 16 * VT + c) * kDim + 16 * KT + 8 * h, tmp);
  }
}

}  // namespace sgdn

// One scan symbol: the parameter list is the kernarg ABI (names, order, sizes), shared by every object.
#define SGDN_SCAN_KERNEL(NAME, KIND, GATE)                                                                     \
  extern "C" __global__ __launch_bounds__(256) void NAME(                                                     \
      sgdn::cbf q, sgdn::cbf k, sgdn::cbf v, sgdn::cbf a_mat, sgdn::cf32 gcs, sgdn::cf32 beta, sgdn::cany h0,  \
      sgdn::wbf o, sgdn::wany ht, sgdn::ci32 cu, int h, int hg, float scale) {                                \
    sgdn::chunk_scan<KIND, GATE>(q, k, v, a_mat, gcs, beta, h0, o, ht, cu, h, hg, scale);                      \
  }
