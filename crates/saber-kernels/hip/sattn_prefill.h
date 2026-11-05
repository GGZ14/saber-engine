// Saber paged attention, CLEANROOM-2 (CR2-2c): the query-tiled prefill body.
//
// One workgroup = 24 waves = 384 rows (BLOCK_Q queries x GQA heads) x one KV head (spec sections 0, 2, 4.1).
// The causal key range is walked in 48-key tiles staged in LDS (K row-major, V transposed); each wave runs
// the lazy-rescale online softmax for its 16 rows over the three 16-key m-tiles of every tile and writes
// bf16 rows directly. Compile-time legs: QK8 (Q and K as e4m3 in the score WMMA), PV8 (P and V as e4m3 in
// the PV WMMA), the key source (block table or QSA list) and the FN124 sub-tile skip.
#pragma once
#include "sattn_prims.h"

namespace sattn {

enum class Skip { None, Wave, HalfVote };

template <int GQA_, bool KV8_, bool QK8_, bool PV8_, bool LIST_, Skip SK_, bool VAHEAD_ = true>
struct PrefillSpec {
  static constexpr int GQA = GQA_;
  static constexpr bool KV8 = KV8_, QK8 = QK8_, PV8 = PV8_, LIST = LIST_;
  static constexpr Skip SK = SK_;
  static constexpr int kWaves = 24, kRows = 384, kBQ = 384 / GQA, kTile = 48;
  static constexpr int kEB = KV8 ? 1 : 2;
  static constexpr int kKRow = 264, kVRow = 56;            // LDS row strides (elements)
  static constexpr int kVJobs = KV8 ? 24 : 48;             // transposing loads per tile
  static constexpr int kVPerWave = kVJobs / kWaves;
  static constexpr float kGrow = PV8 ? 8.0f : 14.0f;
  static constexpr bool kVAhead = VAHEAD_;                 // prefetch the next tile's V during the m-tiles
  typedef typename pick<QK8, uint8_t, _Float16>::type KElem;
  typedef typename pick<PV8, uint8_t, _Float16>::type VElem;
};

template <class S>
struct PrefillShared {
  typename S::KElem k[S::kTile * S::kKRow];
  typename S::VElem v[kHead * S::kVRow];
};

template <class S>
__device__ __forceinline__ void prefill_body(const Params& a, const UnionKeys& uk) {
  constexpr int GQA = S::GQA, BQ = S::kBQ, EB = S::kEB, KR = S::kKRow, VR = S::kVRow;
  __shared__ PrefillShared<S> sm;

  const int qb = blockIdx.x, kvh = blockIdx.y, seq = blockIdx.z;
  const int tid = threadIdx.x, lane = tid & 31;
  const int wv = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int c = lane & 15, hh = lane >> 4;
  const int ql = a.qrows;

  const int nctx = a.ctx_len[seq];
  if (nctx <= 0) return;
  int ntiles, nkeys, cta = 0;
  const int* slots = nullptr;
  if constexpr (S::LIST) {
    cta = seq * (int)gridDim.x + qb;
    nkeys = uk.nkeys[cta];
    ntiles = (nkeys + 47) / 48;
    slots = uk.kslot + (long)cta * uk.kstride;
  } else {
    const int qmax = min(qb * BQ + BQ - 1, ql - 1);
    nkeys = nctx;
    ntiles = (min(nctx - ql + qmax + 1, nctx) + 47) / 48;
  }
  const int last = nkeys - 1;
  const int* btrow = a.page_map + (long)seq * a.map_width;

  // ---- this lane's row
  const int r = wv * 16 + c;
  const int qil = r / GQA, hi = r - (r / GQA) * GQA;
  const int qpos = qb * BQ + qil;
  const bool live = qpos < ql;
  const int qrow = live ? qpos : ql - 1;
  const int qhead = kvh * GQA + hi;
  const long tok = (long)seq * ql + qrow;
  const int klimit = nctx - ql + qpos;  // unclamped row
  const uint8_t* bits = nullptr;
  if constexpr (S::LIST) bits = uk.kbits + (long)cta * uk.kstride * 4 + qil;

  const float kd = a.kscale ? a.kscale[seq * a.nkvh + kvh] : 1.0f;
  const float mul = (a.sm_scale * kd) * kLog2e;

  // Q as the B operand of the score WMMA (row c, dims 16 t + 8 h ..)
  f16x8 qf[S::QK8 ? 1 : 16];
  u32x2 q8[S::QK8 ? 16 : 1];
  {
    const uint16_t* qp = (const uint16_t*)a.query + (tok * a.nqh + qhead) * kHead + 8 * hh;
#pragma unroll
    for (int t = 0; t < 16; ++t) {
      u32x4 w = ld16(qp + 16 * t);
      if constexpr (S::QK8) {
        q8[t].x = e4m3_quad(bf_lo(w.x) * mul, bf_hi(w.x) * mul, bf_lo(w.y) * mul, bf_hi(w.y) * mul);
        q8[t].y = e4m3_quad(bf_lo(w.z) * mul, bf_hi(w.z) * mul, bf_lo(w.w) * mul, bf_hi(w.w) * mul);
      } else {
        u32x4 o;
        o.x = rtz_pair(bf_lo(w.x) * mul, bf_hi(w.x) * mul);
        o.y = rtz_pair(bf_lo(w.y) * mul, bf_hi(w.y) * mul);
        o.z = rtz_pair(bf_lo(w.z) * mul, bf_hi(w.z) * mul);
        o.w = rtz_pair(bf_lo(w.w) * mul, bf_hi(w.w) * mul);
        qf[t] = __builtin_bit_cast(f16x8, o);
      }
    }
  }

  // KV addresses: a wave-uniform base (kv head + page, SGPRs) and a 32-bit lane offset. In the block-table
  // form every 8-key run a wave touches lies in one page, so the page index is wave-uniform; keys past the
  // last are clamped to the last key (their scores are masked, their bytes must still be real).
  const uint8_t* hb = (const uint8_t*)a.cache + (long)kvh * a.head_stride * EB;
  const int last_page = last >> 4;
  auto run_addr = [&](int tile, int page_in_tile, int k, uint32_t elem) -> const uint8_t* {
    const int kc = min(k, last);
    if constexpr (S::LIST) {
      const int s = slots[kc];
      return hb + (long)(s >> 4) * a.page_stride * EB + (uint32_t)(((s & 15) * kSlotElems + elem) * EB);
    } else {
      const int pg = btrow[min(3 * tile + page_in_tile, last_page)];
      return hb + (long)pg * a.page_stride * EB + (uint32_t)(((kc & 15) * kSlotElems + elem) * EB);
    }
  };

  // K staging: one 16-dim chunk of one key per thread; key 2 (i >> 5) + (i & 1), chunk (i & 31) >> 1
  const int kk = 2 * (tid >> 5) + (tid & 1), kch = (tid & 31) >> 1;
  // V staging: wave w's n-th job = key group w % 6 (8 keys, one page) x dim block w / 6 + 4 n; lane (g, i)
  const int vg = lane >> 3, vi = lane & 7;
  const int vkg = wv % 6, vdb = wv / 6;
  auto v_src = [&](int tile, int n) -> const uint8_t* {
    const int db = vdb + 4 * n;
    const int d0 = S::KV8 ? 64 * db + 16 * vg : 32 * db + 8 * vg;
    return run_addr(tile, vkg >> 1, tile * 48 + 8 * vkg + vi, kHead + d0);
  };

  f32x8 acc[16];
#pragma unroll
  for (int g = 0; g < 16; ++g) acc[g] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
  float m_ref = kFloor, off = 0.0f - kFloor, l = 0.0f;

  u32x4 vreg[S::kVPerWave];
  if constexpr (S::kVAhead) {
    if (ntiles > 0) {
#pragma unroll
      for (int n = 0; n < S::kVPerWave; ++n) vreg[n] = load_tr16(v_src(0, n));
    }
  }

  for (int tile = 0; tile < ntiles; ++tile) {
    lds_sync();
    // K fetch (global -> registers)
    const uint8_t* ksrc = run_addr(tile, wv >> 3, tile * 48 + kk, 16 * kch);
    u32x4 kraw0, kraw1;
    kraw0 = ld16(ksrc);
    if constexpr (!S::KV8) kraw1 = ld16(ksrc + 16);
    if constexpr (!S::kVAhead) {
#pragma unroll
      for (int n = 0; n < S::kVPerWave; ++n) vreg[n] = load_tr16(v_src(tile, n));
    }
    sched_fence();
    // V -> LDS (transposed)
#pragma unroll
    for (int n = 0; n < S::kVPerWave; ++n) {
      const int kg = vkg, db = vdb + 4 * n;
      if constexpr (S::KV8) {
        const int d = 64 * db + 16 * vg + 2 * vi;
        u32x2 ev, od;
        unzip_pairs(vreg[n], ev, od);
        if constexpr (S::PV8) {
          *lds_ptr((u32x2*)&sm.v[d * VR + 8 * kg]) = ev;
          *lds_ptr((u32x2*)&sm.v[(d + 1) * VR + 8 * kg]) = od;
        } else {
          *lds_ptr((f16x8*)&sm.v[d * VR + 8 * kg]) = e4m3x8_to_f16(ev);
          sched_fence();
          *lds_ptr((f16x8*)&sm.v[(d + 1) * VR + 8 * kg]) = e4m3x8_to_f16(od);
        }
      } else {
        const int d = 32 * db + 8 * vg + vi;
        *lds_ptr((f16x8*)&sm.v[d * VR + 8 * kg]) = bf16x8_to_f16(vreg[n]);
      }
    }
    sched_fence();
    // K -> LDS
    if constexpr (S::QK8) {
      *lds_ptr((u32x4*)&sm.k[kk * KR + 16 * kch]) = kraw0;
    } else if constexpr (S::KV8) {
      // two 8-dim halves, fenced apart: keeps the f32 temporaries of the widen to one half at a time
      *lds_ptr((f16x8*)&sm.k[kk * KR + 16 * kch]) = e4m3x8_to_f16((u32x2){kraw0.x, kraw0.y});
      sched_fence();
      *lds_ptr((f16x8*)&sm.k[kk * KR + 16 * kch + 8]) = e4m3x8_to_f16((u32x2){kraw0.z, kraw0.w});
    } else {
      *lds_ptr((f16x8*)&sm.k[kk * KR + 16 * kch]) = bf16x8_to_f16(kraw0);
      *lds_ptr((f16x8*)&sm.k[kk * KR + 16 * kch + 8]) = bf16x8_to_f16(kraw1);
    }
    lds_sync();
    if constexpr (S::kVAhead) {
      if (tile + 1 < ntiles) {
#pragma unroll
        for (int n = 0; n < S::kVPerWave; ++n) vreg[n] = load_tr16(v_src(tile + 1, n));
      }
    }

#pragma unroll 1
    for (int mt = 0; mt < 3; ++mt) {
      const int kb = tile * 48 + 16 * mt + 8 * hh;
      uint32_t mbyte = 0;
      if constexpr (S::LIST) {
        mbyte = bits[(kb >> 3) * 32];
        if constexpr (S::SK == Skip::Wave) {
          if (!wave_any(mbyte != 0u)) continue;
        } else if constexpr (S::SK == Skip::HalfVote) {
          if (!wave_any(hh == 0 && mbyte != 0u)) continue;
        }
      }
      // scores: A = K (row = key 16 mt + c, k = dims), B = Q
      f32x8 sc = {0, 0, 0, 0, 0, 0, 0, 0};
      const SATTN_LDS typename S::KElem* krow = lds_ptr(&sm.k[(16 * mt + c) * KR + 8 * hh]);
#pragma unroll
      for (int t0 = 0; t0 < 16; t0 += 4) {
        if constexpr (S::QK8) {
          u32x2 kf[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) kf[i] = *(const SATTN_LDS u32x2*)(krow + 16 * (t0 + i));
          sched_fence();
#pragma unroll
          for (int i = 0; i < 4; ++i) sc = mma8(kf[i], q8[t0 + i], sc);
          sched_fence();
        } else {
          f16x8 kf[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) kf[i] = *(const SATTN_LDS f16x8*)(krow + 16 * (t0 + i));
          sched_fence();
#pragma unroll
          for (int i = 0; i < 4; ++i) sc = mma16(kf[i], qf[t0 + i], sc);
          sched_fence();
        }
      }

      // mask
      if constexpr (S::LIST) {
#pragma unroll
        for (int e = 0; e < 8; ++e)
          if (!((mbyte >> e) & 1u)) sc[e] = -__builtin_inff();
      } else {
        if (wave_any(kb + 7 > klimit)) {
#pragma unroll
          for (int e = 0; e < 8; ++e)
            if (kb + e > klimit) sc[e] = -__builtin_inff();
        }
      }

      float smax = sc[0];
#pragma unroll
      for (int e = 1; e < 8; ++e) smax = fmaxf(smax, sc[e]);
      if (wave_any(smax > m_ref + S::kGrow)) {
        float rmax = fmaxf(smax, other_half(smax));
        float mnew = fmaxf(fmaxf(m_ref, rmax), kFloor);
        float alpha = ex2(m_ref - mnew);
#pragma unroll
        for (int g = 0; g < 16; ++g) acc[g] *= alpha;
        l *= alpha;
        m_ref = mnew;
        off = 0.0f - mnew;
      }

      float p[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) p[e] = ex2(sc[e] + off);
      const SATTN_LDS typename S::VElem* vrow = lds_ptr(&sm.v[c * VR + 16 * mt + 8 * hh]);
      if constexpr (S::PV8) {
        float ls = 0.0f;
#pragma unroll
        for (int e = 0; e < 8; ++e) ls = ls + p[e];
        l = l + ls;
        const u32x2 pb = {e4m3_quad(p[0], p[1], p[2], p[3]), e4m3_quad(p[4], p[5], p[6], p[7])};
#pragma unroll
        for (int g0 = 0; g0 < 16; g0 += 4) {
          u32x2 vf[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) vf[i] = *(const SATTN_LDS u32x2*)(vrow + (g0 + i) * 16 * VR);
          sched_fence();
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[g0 + i] = mma8(vf[i], pb, acc[g0 + i]);
          sched_fence();
        }
      } else {
        uint32_t pw[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) pw[i] = rtz_pair(p[2 * i], p[2 * i + 1]);
#pragma unroll
        for (int i = 0; i < 4; ++i) l = dot2_ones(pw[i], l);
        const f16x8 pb = __builtin_bit_cast(f16x8, (u32x4){pw[0], pw[1], pw[2], pw[3]});
#pragma unroll
        for (int g0 = 0; g0 < 16; g0 += 4) {
          f16x8 vf[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) vf[i] = *(const SATTN_LDS f16x8*)(vrow + (g0 + i) * 16 * VR);
          sched_fence();
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[g0 + i] = mma16(vf[i], pb, acc[g0 + i]);
          sched_fence();
        }
      }
    }
  }

  // ---- epilogue: bf16 rows
  const float lt = l + other_half(l);
  if (!live) return;
  const float vd = a.vscale ? a.vscale[seq * a.nkvh + kvh] : 1.0f;
  const float inv = lt > 0.0f ? vd / lt : 0.0f;
  uint16_t* dst = (uint16_t*)a.dst + (tok * a.nqh + qhead) * kHead + 8 * hh;
#pragma unroll
  for (int g = 0; g < 16; ++g) {
    f32x8 v = acc[g];
    u32x4 o;
    o.x = bf16_pair(v[0] * inv, v[1] * inv);
    o.y = bf16_pair(v[2] * inv, v[3] * inv);
    o.z = bf16_pair(v[4] * inv, v[5] * inv);
    o.w = bf16_pair(v[6] * inv, v[7] * inv);
    *(u32x4*)(dst + 16 * g) = o;
  }
}

}  // namespace sattn
