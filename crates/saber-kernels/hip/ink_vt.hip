// Inkling 2 (inkling2_loop) fused T-row verify body, INK17 (docs/INKLING2.md section 9 INK17).
// Saber's own. Its own object so ink_fuse keeps its bytes.
//
// The fused decode form (INK10 / INK11) takes ONE row a sequence: ink_xsd reads and writes row t's
// 3-row sconv state at slot rseq[t] in place, so two rows of one sequence (a verify chunk) would
// both read the state before the chunk. A verify chunk therefore ran INK9's chain (pulled reduce +
// ink_sconv_resid + ink_depth_attend + the history stash as their own launches: T 3 18.1 ms vs the
// b1 step's 12.35).
//
//   ink_xsdc   ink_xsd for T rows of ONE sequence (chained rows). Workgroup t forms the reduced rows
//              x_r = own_r + peer_r (ink_p2p_xchg's expression) for r = t-3..t ITSELF (redundant
//              reads of the pushed words: no workgroup waits on another), the conv of row t over
//              [state ; x_0 .. x_t] in ink_sconv_resid's order (acc = 0 + v_{t-3} w0 + v_{t-2} w1 +
//              v_{t-1} w2 + x_t w3, v_s < 0 = state row 3 + s), (+ resid), then ink_xsd's depth
//              attend verbatim. It writes the verify's history entry [state (3 rows) ; x (T rows)]
//              (ink_dr_hist's layout, the rollback's input) and the LAST workgroup to arrive (one
//              agent-scope counter chosen by value, reset by it) writes the new state = history rows
//              T .. T+2 (ink_sconv_resid's st_out) after every workgroup read the old one.
//              -> every output byte = the chain's (gated in situ: tf / verify rows, the MTP law).
//
// Built with -ffp-contract=off (saber-kbuild's `ink_` row).
#include <hip/hip_runtime.h>

__device__ __forceinline__ float fbf(unsigned short u) {
  return __builtin_bit_cast(float, ((unsigned)u) << 16);
}
__device__ __forceinline__ float lo16(unsigned w) { return __builtin_bit_cast(float, w << 16); }
__device__ __forceinline__ float hi16(unsigned w) {
  return __builtin_bit_cast(float, w & 0xFFFF0000u);
}
__device__ __forceinline__ unsigned xf_ld_sys(const unsigned* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
// bounded spin (40 s of the 100 MHz realtime clock: a KFD queue eviction is a wait, not a hang)
__device__ __forceinline__ void xf_spin_ge(const unsigned* p, unsigned s, unsigned* diag) {
  const unsigned long long t0 = wall_clock64();
  while ((int)(xf_ld_sys(p) - s) < 0) {
    __builtin_amdgcn_s_sleep(1);
    if (wall_clock64() - t0 > 4000000000ull) {
      atomicAdd(diag, 1u);
      break;
    }
  }
}

#define DA_D 2304
#define DA_PER 9
#define DA_SMAX 8
#define DA_QMAX 4
struct InkPtrs {
  const float* s[DA_SMAX];
  const float* q[DA_QMAX];
  float* o[DA_QMAX];
  const unsigned short* g[DA_QMAX];
};
__device__ __forceinline__ float bsum256(float v, float* red) {
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

extern "C" __global__ __launch_bounds__(256) void ink_xsdc(
    const float* __restrict__ own, int ld_own, const float* win, int ld_win, const unsigned* flags,
    const unsigned* ctr, int nfl, int per_row, int xmode, const unsigned short* __restrict__ w,
    float* st, const int* __restrict__ rseq, const float* __restrict__ resid, int ld_r,
    float* o, int ld_o, InkPtrs P, int S, int nq, int ld_s, int ld_qo, float eps, unsigned* diag,
    float* hist, unsigned* arrive) {
  __shared__ float red[8];
  __shared__ unsigned s_slot[256];
  __shared__ unsigned s_last;
  const int t = blockIdx.x, tid = threadIdx.x, T = gridDim.x;
  const int r0 = t >= 3 ? t - 3 : 0;
  // 1. the exchange: wave 0 waits for the flags rows r0..t need (wave-uniform branch)
  if (xmode) {
    if (__builtin_amdgcn_readfirstlane(tid >> 5) == 0) {
      const int lane = tid & 31;
      const int j0 = per_row ? r0 : 0, nj = per_row ? t - r0 + 1 : nfl;
      for (int j = lane; j < nj; j += 32) {
        const unsigned s = ctr[j0 + j];
        xf_spin_ge(&flags[j0 + j], s, diag);
        s_slot[j] = s & 1u;
      }
    }
    __syncthreads();
  }
  // 2. x_r = own + peer for r = t-3..t, the conv of row t over [state ; x] (ink_sconv_resid)
  const float* sb = st + (long)rseq[t] * 3 * DA_D;
#pragma unroll
  for (int i = 0; i < DA_PER; ++i) {
    const int c = tid + 256 * i;
    float xv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int s = t - 3 + j;
      if (s >= 0) {
        float xs = own[(long)s * ld_own + c];
        if (xmode) {
          const unsigned slot = per_row ? s_slot[s - r0] : s_slot[c / (DA_D / nfl)];
          const float pv = __builtin_bit_cast(
              float, xf_ld_sys((const unsigned*)&win[((long)slot * 8 + s) * ld_win + c]));
          xs = xs + pv;
        }
        xv[j] = xs;
      } else {
        xv[j] = sb[(long)(3 + s) * DA_D + c];
      }
    }
    float wt[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) wt[j] = fbf(w[(long)c * 4 + j]);
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) acc = acc + xv[j] * wt[j];
    float y = (acc + xv[3]) / 2.2360679774997898f;
    if (resid) y = y + resid[(long)t * ld_r + c];
    o[(long)t * ld_o + c] = y;
    hist[(long)(3 + t) * DA_D + c] = xv[3];
    if (t == 0) {
#pragma unroll
      for (int j = 0; j < 3; ++j) hist[(long)j * DA_D + c] = sb[(long)j * DA_D + c];
    }
  }
  // 3. ink_depth_attend (ink_xsd's, verbatim; state S - 1 is `o`, each element re-read by its writer)
  if (nq > 0) {
    float k[DA_SMAX][DA_PER];
    float sc[DA_QMAX][DA_SMAX];
#pragma unroll
    for (int si = 0; si < DA_SMAX; ++si) {
      if (si >= S) break;
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < DA_PER; ++i) {
        const float x = P.s[si][(long)t * ld_s + tid + 256 * i];
        k[si][i] = x;
        ss = ss + x * x;
      }
      const float tot = bsum256(ss, red);
      const float r = 1.0f / sqrtf(tot / (float)DA_D + eps);
#pragma unroll
      for (int i = 0; i < DA_PER; ++i) k[si][i] = k[si][i] * r;
#pragma unroll
      for (int qi = 0; qi < DA_QMAX; ++qi) {
        if (qi >= nq) break;
        float d = 0.f;
#pragma unroll
        for (int i = 0; i < DA_PER; ++i) d = d + k[si][i] * P.q[qi][tid + 256 * i];
        sc[qi][si] = bsum256(d, red) * 0.020833333333333332f;
      }
    }
#pragma unroll
    for (int qi = 0; qi < DA_QMAX; ++qi) {
      if (qi >= nq) break;
      float mx = sc[qi][0];
#pragma unroll
      for (int si = 1; si < DA_SMAX; ++si)
        if (si < S) mx = fmaxf(mx, sc[qi][si]);
      float e[DA_SMAX], den = 0.f;
#pragma unroll
      for (int si = 0; si < DA_SMAX; ++si) {
        if (si >= S) break;
        e[si] = expf(sc[qi][si] - mx);
        den = den + e[si];
      }
      float out[DA_PER];
#pragma unroll
      for (int i = 0; i < DA_PER; ++i) out[i] = 0.f;
#pragma unroll
      for (int si = 0; si < DA_SMAX; ++si) {
        if (si >= S) break;
        const float wgt = e[si] / den;
#pragma unroll
        for (int i = 0; i < DA_PER; ++i) out[i] = out[i] + wgt * k[si][i];
      }
      if (P.g[qi]) {
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < DA_PER; ++i) ss = ss + out[i] * out[i];
        const float r = 1.0f / sqrtf(bsum256(ss, red) / (float)DA_D + eps);
#pragma unroll
        for (int i = 0; i < DA_PER; ++i) out[i] = (out[i] * r) * fbf(P.g[qi][tid + 256 * i]);
      }
#pragma unroll
      for (int i = 0; i < DA_PER; ++i) P.o[qi][(long)t * ld_qo + tid + 256 * i] = out[i];
    }
  }
  // 4. the last arriver writes the new state (history rows T .. T+2) after every read of the old;
  // a verify chunk passes arrive = null: its commit (ink_dr_commit from the history) owns the state
  if (!arrive) return;
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    const unsigned d = __hip_atomic_fetch_add(arrive, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    s_last = (d == (unsigned)T - 1u) ? 1u : 0u;
  }
  __syncthreads();
  if (!s_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  if (tid == 0) __hip_atomic_store(arrive, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  float* so = st + (long)rseq[0] * 3 * DA_D;
#pragma unroll
  for (int i = 0; i < DA_PER; ++i) {
    const int c = tid + 256 * i;
#pragma unroll
    for (int j = 0; j < 3; ++j) so[(long)j * DA_D + c] = hist[(long)(T + j) * DA_D + c];
  }
}

// ------------------------------------------------------------------------------------------------
// INK17: ink_route_fused_rt -- ink_fuse's ink_route_fused (route logits grid [T, 16], verbatim) with
// the tail run PER ROW: the last of row t's 16 workgroups (a per-row counter chosen by value, reset
// by it) runs row t's top-8 + weights + act quant (the serial tail's row body, verbatim), and the
// last row to finish runs the route sort over every row (ids read back from global after the
// acquire). Every value = the parent's expression in the parent's order -> byte-equal; T rows no
// longer queue behind one workgroup.
#define RT_E 128
__device__ __forceinline__ unsigned char ff2e4m3(float v) {
  const int p = __builtin_amdgcn_cvt_pk_fp8_f32(v, 0.f, 0, false);
  return (unsigned char)(p & 0xFF);
}
extern "C" __global__ __launch_bounds__(128) void ink_route_fused_rt(
    const float* __restrict__ h, int ld_h, const unsigned short* __restrict__ gain,
    const unsigned short* __restrict__ wr, const float* __restrict__ bias, float* u, int ld_u,
    float* sk, int D, int* __restrict__ ids, float* __restrict__ wts, float route_scale,
    unsigned char* __restrict__ q, int ld_q, float* __restrict__ scale, int T, int lo, int nloc,
    int* __restrict__ row_tok, int* __restrict__ row_slot, int* __restrict__ tile_e,
    int* __restrict__ tile_r0, int* __restrict__ tile_n, int* __restrict__ meta,
    int* __restrict__ pos, unsigned* arrive, unsigned* rarrive) {
  __shared__ float us[2304];
  __shared__ float red[4];
  __shared__ unsigned short wsr[8][2304];
  __shared__ unsigned s_last;
  {
    const int t = blockIdx.x, b = blockIdx.y, e = threadIdx.x, lane = e & 31, w = e >> 5;
    for (int p = e; p < 8 * (D / 8); p += 128) {
      const int r = p / (D / 8), qq = p % (D / 8);
      *reinterpret_cast<uint4*>(&wsr[r][8 * qq]) =
          *reinterpret_cast<const uint4*>(wr + (long)(8 * b + r) * D + 8 * qq);
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
      const uint4* wr8 = reinterpret_cast<const uint4*>(&wsr[e][0]);
#pragma unroll 4
      for (int i8 = 0; i8 < 2304 / 8; ++i8) {
        const uint4 x = wr8[i8];
        const unsigned wv[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
          dot = dot + us[8 * i8 + 2 * qq] * lo16(wv[qq]);
          dot = dot + us[8 * i8 + 2 * qq + 1] * hi16(wv[qq]);
        }
      }
      const int ex = 8 * b + e;
      const float s = 1.0f / (1.0f + expf(-dot));
      sk[((long)t * 128 + ex) * 2 + 0] = s + bias[ex];
      sk[((long)t * 128 + ex) * 2 + 1] = s;
    }
  }
  // INK17: the LAST of row t's 16 workgroups runs row t's top-8 + act quant (ink_route_tail's
  // block 0 row body + block 1 + t, verbatim); the last ROW to finish runs the route sort
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned d =
        __hip_atomic_fetch_add(&rarrive[blockIdx.x], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    s_last = (d == gridDim.y - 1u) ? 1u : 0u;
  }
  __syncthreads();
  if (!s_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  const int tid = threadIdx.x;
  if (tid == 0) __hip_atomic_store(&rarrive[blockIdx.x], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  __shared__ float key[RT_E];
  __shared__ float sel[8];
  __shared__ int selid[8];
  __shared__ int sids[256];
  {
    const int t = blockIdx.x;
    const float mk = sk[((long)t * 128 + tid) * 2 + 0];
    const float sv = sk[((long)t * 128 + tid) * 2 + 1];
    key[tid] = mk;
    __syncthreads();
    int rank = 0;
    for (int j = 0; j < 128; ++j) {
      const float kj = key[j];
      rank += (kj > mk) || (kj == mk && j < tid);
    }
    if (rank < 8) {
      sel[rank] = sv;
      selid[rank] = tid;
    }
    __syncthreads();
    if (tid < 8) {
      float sum = 0.f;
      for (int j = 0; j < 8; ++j) sum = sum + sel[j];
      ids[t * 8 + tid] = selid[tid];
      wts[t * 8 + tid] = sel[tid] / (sum + 1e-20f) * route_scale;
    }
    __syncthreads();
  }
  {
    __shared__ float mred[4];
    const int t = blockIdx.x;
    float mx = 0.f;
    for (int i = tid; i < D; i += 128) mx = fmaxf(mx, fabsf(u[(long)t * ld_u + i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
    __syncthreads();
    if ((tid & 31) == 0) mred[tid >> 5] = mx;
    __syncthreads();
    mx = fmaxf(fmaxf(mred[0], mred[1]), fmaxf(mred[2], mred[3]));
    const float s = fmaxf(mx / 448.0f, 1e-30f);
    for (int i = tid; i < D; i += 128) q[(long)t * ld_q + i] = ff2e4m3(u[(long)t * ld_u + i] / s);
    if (tid == 0) scale[t] = s;
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    const unsigned d = __hip_atomic_fetch_add(arrive, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    s_last = (d == gridDim.x - 1u) ? 1u : 0u;
  }
  __syncthreads();
  if (!s_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  if (tid == 0) __hip_atomic_store(arrive, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  for (int i = tid; i < T * 8; i += 128) sids[i] = ids[i];
  __syncthreads();
  __shared__ int cnt[RT_E], seg[RT_E], run[RT_E], tbase[RT_E];
  __shared__ int ce[128];
  const int N = T * 8;   // <= 64 < 128: one pass of the stable rank below
  cnt[tid] = 0;
  run[tid] = 0;
  __syncthreads();
  for (int i = tid; i < N; i += 128) {
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
    for (int k = tid; k < nt; k += 128) {
      tile_e[tbase[e] + k] = e;
      tile_r0[tbase[e] + k] = seg[e] + 16 * k;
      tile_n[tbase[e] + k] = min(16, cnt[e] - 16 * k);
    }
  }
  {
    const int i = tid;
    int e = -1;
    if (i < N) {
      e = sids[i] - lo;
      if (e < 0 || e >= nloc) e = -1;
    }
    ce[tid] = e;
    __syncthreads();
    int rk = 0;
    if (e >= 0) {
      for (int j = 0; j < tid; ++j) rk += ce[j] == e;
      const int row = seg[e] + run[e] + rk;
      row_tok[row] = i >> 3;
      row_slot[row] = i;
      pos[i] = row;
    } else if (i < N) {
      pos[i] = -1;
    }
    __syncthreads();
  }
  for (int e = 0; e < nloc; ++e) {
    const int c = cnt[e], padded = ((c + 15) >> 4) * 16;
    __syncthreads();
    for (int k = c + tid; k < padded; k += 128) {
      row_tok[seg[e] + k] = row_tok[seg[e] + c - 1];
      row_slot[seg[e] + k] = -1;
    }
    __syncthreads();
  }
}

// ------------------------------------------------------------------------------------------------
