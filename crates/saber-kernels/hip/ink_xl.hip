// INK18: the fused consumers' window loads in one acquire (docs/INKLING2.md section 9 INK18).
// Saber's own; its own object so ink_fuse / ink_vt keep their bytes.
//
//   ink_xsd_l / ink_xsdc_l   ink_fuse's ink_xsd / ink_vt's ink_xsdc, every byte of arithmetic
//              verbatim; the ONE change: the pushed peer words are read with plain loads after ONE
//              system-scope acquire fence a thread (after the flag spin + the barrier), instead of
//              one system-scope load an element (ink_xsdc re-reads rows t-3..t: 4 x 9 a thread).
//              The producer publishes with a system-scope release before raising the flag (the
//              fence-fence law ink_p2p_xchg_v4 rests on, INK15). Gated: decode == tf rows, every
//              verify round's rows sha256-equal, the MTP law.
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

extern "C" __global__ __launch_bounds__(256) void ink_xsd_l(
    const float* __restrict__ own, int ld_own, const float* win, int ld_win, const unsigned* flags,
    const unsigned* ctr, int nfl, int per_row, int xmode, const unsigned short* __restrict__ w,
    float* st, const int* __restrict__ rseq, const float* __restrict__ resid, int ld_r,
    float* o, int ld_o, InkPtrs P, int S, int nq, int ld_s, int ld_qo, float eps, unsigned* diag) {
  __shared__ float red[8];
  __shared__ unsigned s_slot[256];
  const int t = blockIdx.x, tid = threadIdx.x;
  // 1. the exchange: wave 0 waits for the flags this row needs (wave-uniform branch)
  if (xmode) {
    if (__builtin_amdgcn_readfirstlane(tid >> 5) == 0) {
      const int lane = tid & 31;
      const int j0 = per_row ? t : 0, nj = per_row ? 1 : nfl;
      for (int j = lane; j < nj; j += 32) {
        const unsigned s = ctr[j0 + j];
        xf_spin_ge(&flags[j0 + j], s, diag);
        s_slot[j] = s & 1u;
      }
    }
    __syncthreads();
    // INK18: one system-scope acquire a thread after the flags (the producer raised them with a
    // system-scope release after its pushed words): plain loads of the window observe the words
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  }
  // 2. x = own + peer (ink_p2p_xchg's expression), the sconv (ink_sconv_rows' arithmetic)
  float* sb = st + (long)rseq[t] * 3 * DA_D;
#pragma unroll
  for (int i = 0; i < DA_PER; ++i) {
    const int c = tid + 256 * i;
    float xt = own[(long)t * ld_own + c];
    if (xmode) {
      // the attention site's producer workgroup of column c is c / (D / nfl) (its slot)
      const unsigned slot = per_row ? s_slot[0] : s_slot[c / (DA_D / nfl)];
      const float pv =
          win[((long)slot * 8 + t) * ld_win + c];
      xt = xt + pv;
    }
    float wt[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) wt[j] = fbf(w[(long)c * 4 + j]);
    const float s0 = sb[c], s1 = sb[(long)DA_D + c], s2 = sb[2L * DA_D + c];
    float acc = 0.f;
    acc = acc + s0 * wt[0];
    acc = acc + s1 * wt[1];
    acc = acc + s2 * wt[2];
    acc = acc + xt * wt[3];
    float y = (acc + xt) / 2.2360679774997898f;
    if (resid) y = y + resid[(long)t * ld_r + c];
    o[(long)t * ld_o + c] = y;
    sb[c] = s1;
    sb[(long)DA_D + c] = s2;
    sb[2L * DA_D + c] = xt;
  }
  if (nq == 0) return;
  // 3. ink_depth_attend (verbatim; state S - 1 is `o`, each element re-read by its own writer)
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

extern "C" __global__ __launch_bounds__(256) void ink_xsdc_l(
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
    // INK18: one system-scope acquire a thread after the flags (the producer raised them with a
    // system-scope release after its pushed words): plain loads of the window observe the words
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
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
          const float pv = win[((long)slot * 8 + s) * ld_win + c];
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
