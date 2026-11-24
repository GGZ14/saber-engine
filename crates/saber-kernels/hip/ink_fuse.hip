// Inkling 2 (inkling2_loop) fused decode bodies, INK10 (docs/INKLING2.md section 9). Saber's own.
// Each body is a FUSION of byte-equal parents (ink_fast.hip / ink_loop.hip / ink_p2p.hip): every
// value is computed by the parent's expression in the parent's order, only the launch boundaries
// and the reduce's transport move -> the outputs are BYTE-equal to the unfused chain (gated in situ:
// tools/capture/ink9/cmp9.py over the two-card tf + decode rows vs INK8 h2's).
//
// THE PUSH REDUCE (`--ink-fuse x`): the two reduces of a visit (o_proj's partial, the MoE combine's
// partial) no longer run ink_p2p_xchg (two PCIe crossings: the peer's operand READ over P2P + the
// depart handshake). The PRODUCER (ink_gemv_gate_push / ink_combine_push) writes its f32 partial
// into the PEER's receive window (a posted P2P write), drains its stores (s_wait_storecnt 0), and
// raises its workgroup's flag in the peer's flag array with a system-scope release; the CONSUMER
// (ink_xsd, the next launch on the same stream) spins on the flags it needs (system scope), reads
// the pushed words with system-scope loads (no stale cache line can serve them) and forms
// own + peer -- f32 addition commutes, so both ranks hold rank 0 + rank 1's bytes, as the xchg.
// The slot law (FN114d / FN164): every producer workgroup keeps its own CALL COUNTER on its own card
// (`ctr[wg]`, advanced once a call whatever the grid); the call s writes slot s & 1 and flags s;
// the consumer expects the local producer's counter (both ranks run the same calls a site). A slot
// is rewritten only at call s + 2, which the peer can issue only after it consumed our call s + 1,
// which we pushed only after our consumer of call s finished (stream order) -> no reuse hazard.
// One window + counter set a SITE (attention / MoE): a site's grid never changes between calls.
//
//   ink_gemv_gate_push  ink_fast's ink_gemv_gate_f32_bf16 (the gated o GEMV, same body) + the push
//                       of its 64 columns x M rows; flag + counter by workgroup.
//   ink_combine_push    ink_loop's ink_moe_combine (same body) + the push of its row; flag + counter
//                       by row.
//   ink_xsd             the consumer: (spin) x = own + peer (xchg's expression), then ink_fast's
//                       ink_sconv_rows (row t's 3-row state at slot rseq[t], in place, + resid),
//                       then ink_loop's ink_depth_attend over S states (the last = the sconv output,
//                       re-read by the thread that wrote it) for nq queries (+ optional gains).
//                       xmode 0 = no exchange (x = own), nq 0 = no depth (exchange + sconv only).
//   ink_gemv_n{16,32}_f32_bf16 / ink_gemv_gate_push_n{16,32}  INK10 task 2: ink_fast's M <= 8 GEMV
//                       (and its gated + push form) on a NARROW column tile (16 / 32 outputs a
//                       workgroup instead of 64: 4x / 2x the workgroups -- the 64-wide tile left
//                       N 512 at 8 workgroups) with THREE 256-k tiles in flight in registers (a
//                       ring, K / 256 % 3 == 0: 2,304 and 1,536); every output is still ONE
//                       sequential f32 sum over k = 0..K-1 in the parent's order -> byte-equal.
//   ink_route_fused     ink_fast's ink_route_logits (grid [T, 16], same body) + ink_route_tail's
//                       work run by the LAST workgroup to finish (a counter chosen by value, reset
//                       by that workgroup): the top-8 of every row, the route sort and the act quant
//                       of every row (max / convert are exact and order-free; the sort is stable by
//                       construction at N = 8 T <= 64).
//
// Built with -ffp-contract=off (saber-kbuild's `ink_` row).
#include <hip/hip_runtime.h>

__device__ __forceinline__ float lo16(unsigned w) { return __builtin_bit_cast(float, w << 16); }
__device__ __forceinline__ float hi16(unsigned w) {
  return __builtin_bit_cast(float, w & 0xFFFF0000u);
}
__device__ __forceinline__ float fbf(unsigned short u) {
  return __builtin_bit_cast(float, ((unsigned)u) << 16);
}
__device__ __forceinline__ unsigned xf_ld_sys(const unsigned* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void xf_rel(unsigned* p, unsigned v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
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

// ------------------------------------------------------------------------------------------------
// the gated o GEMV (ink_fast.hip's gvg_* / gv_* helpers, verbatim)
#define GV_N 64
#define GV_K 256
#define GV_M 8
#define GV_LDW (GV_K / 2 + 1)
#define GV_LDS_BARRIER()                                                    \
  do {                                                                      \
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");         \
    __builtin_amdgcn_s_barrier();                                           \
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");         \
  } while (0)

template <int MM>
__device__ __forceinline__ void gv_step(const uint4 (&p)[8], const float (&ap)[MM], unsigned* ws,
                                        float (*as)[GV_K], float (&s)[GV_M], int tid) {
  GV_LDS_BARRIER();
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int e = tid + 256 * i, r = e >> 5, u = e & 31;
    unsigned* d = ws + r * GV_LDW + u * 4;
    d[0] = p[i].x;
    d[1] = p[i].y;
    d[2] = p[i].z;
    d[3] = p[i].w;
  }
#pragma unroll
  for (int m = 0; m < MM; ++m) as[m][tid] = ap[m];
  GV_LDS_BARRIER();
}

template <int MM>
__device__ __forceinline__ void gv_sum(const unsigned* ws, float (*as)[GV_K], float (&s)[GV_M],
                                       int tid) {
  if (tid < GV_N) {
    const unsigned* row = ws + tid * GV_LDW;
#pragma unroll 16
    for (int kk = 0; kk < GV_K; kk += 2) {
      const unsigned wd = row[kk >> 1];
      const float w0 = lo16(wd), w1 = hi16(wd);
#pragma unroll
      for (int m = 0; m < MM; ++m) s[m] = s[m] + as[m][kk] * w0;
#pragma unroll
      for (int m = 0; m < MM; ++m) s[m] = s[m] + as[m][kk + 1] * w1;
    }
  }
}

template <int MM>
__device__ __forceinline__ void gvg_load(uint4 (&p)[8], float (&ap)[MM], const float* __restrict__ a,
                                         int lda, const float* __restrict__ g, int ldg,
                                         const unsigned short* __restrict__ w, int ldw, int n0,
                                         int N, int k0, int tid) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int e = tid + 256 * i, r = e >> 5, u = e & 31;
    const int gn = n0 + r;
    p[i] = gn < N ? *reinterpret_cast<const uint4*>(w + (long)gn * ldw + k0 + u * 8)
                  : make_uint4(0u, 0u, 0u, 0u);
  }
#pragma unroll
  for (int m = 0; m < MM; ++m) {
    const float sg = 1.0f / (1.0f + expf(-g[(long)m * ldg + k0 + tid]));
    ap[m] = a[(long)m * lda + k0 + tid] * sg;
  }
}

template <int MM>
__device__ __forceinline__ void gvgp_body(const float* __restrict__ a, int lda,
                                          const float* __restrict__ g, int ldg,
                                          const unsigned short* __restrict__ w, int ldw,
                                          float* __restrict__ c, int ldc, int N, int K, int acc,
                                          float* pwin, int ld_win, unsigned slot, unsigned* ws,
                                          float (*as)[GV_K]) {
  const int tid = threadIdx.x;
  const int n0 = blockIdx.x * GV_N;
  uint4 pa[8], pb[8];
  float aa[MM], ab[MM];
  float s[GV_M];
#pragma unroll
  for (int m = 0; m < GV_M; ++m) s[m] = 0.f;
  gvg_load<MM>(pa, aa, a, lda, g, ldg, w, ldw, n0, N, 0, tid);
  if (GV_K < K) gvg_load<MM>(pb, ab, a, lda, g, ldg, w, ldw, n0, N, GV_K, tid);
  for (int k0 = 0; k0 < K; k0 += 2 * GV_K) {
    gv_step<MM>(pa, aa, ws, as, s, tid);
    if (k0 + 2 * GV_K < K) gvg_load<MM>(pa, aa, a, lda, g, ldg, w, ldw, n0, N, k0 + 2 * GV_K, tid);
    gv_sum<MM>(ws, as, s, tid);
    if (k0 + GV_K < K) {
      gv_step<MM>(pb, ab, ws, as, s, tid);
      if (k0 + 3 * GV_K < K)
        gvg_load<MM>(pb, ab, a, lda, g, ldg, w, ldw, n0, N, k0 + 3 * GV_K, tid);
      gv_sum<MM>(ws, as, s, tid);
    }
  }
  const int gn = n0 + tid;
  if (tid < GV_N && gn < N) {
#pragma unroll
    for (int m = 0; m < MM; ++m) {
      float* p = c + (long)m * ldc + gn;
      const float v = acc ? *p + s[m] : s[m];
      *p = v;
      pwin[((long)slot * GV_M + m) * ld_win + gn] = v;
    }
  }
}

extern "C" __global__ __launch_bounds__(256) void ink_gemv_gate_push(
    const float* __restrict__ a, int lda, const float* __restrict__ g, int ldg,
    const unsigned short* __restrict__ w, int ldw, float* __restrict__ c, int ldc, int M, int N,
    int K, int acc, float* pwin, int ld_win, unsigned* pflags, unsigned* ctr) {
  __shared__ unsigned ws[GV_N * GV_LDW];
  __shared__ float as[GV_M][GV_K];
  __shared__ unsigned s_call;
  if (threadIdx.x == 0) {
    const unsigned s = ctr[blockIdx.x] + 1u;
    ctr[blockIdx.x] = s;
    s_call = s;
  }
  __syncthreads();
  const unsigned s = s_call, slot = s & 1u;
  switch (M) {
    case 1: gvgp_body<1>(a, lda, g, ldg, w, ldw, c, ldc, N, K, acc, pwin, ld_win, slot, ws, as); break;
    case 2: gvgp_body<2>(a, lda, g, ldg, w, ldw, c, ldc, N, K, acc, pwin, ld_win, slot, ws, as); break;
    case 3: gvgp_body<3>(a, lda, g, ldg, w, ldw, c, ldc, N, K, acc, pwin, ld_win, slot, ws, as); break;
    case 4: gvgp_body<4>(a, lda, g, ldg, w, ldw, c, ldc, N, K, acc, pwin, ld_win, slot, ws, as); break;
    case 5: gvgp_body<5>(a, lda, g, ldg, w, ldw, c, ldc, N, K, acc, pwin, ld_win, slot, ws, as); break;
    case 6: gvgp_body<6>(a, lda, g, ldg, w, ldw, c, ldc, N, K, acc, pwin, ld_win, slot, ws, as); break;
    case 7: gvgp_body<7>(a, lda, g, ldg, w, ldw, c, ldc, N, K, acc, pwin, ld_win, slot, ws, as); break;
    default: gvgp_body<8>(a, lda, g, ldg, w, ldw, c, ldc, N, K, acc, pwin, ld_win, slot, ws, as); break;
  }
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) xf_rel(&pflags[blockIdx.x], s);
}

// ------------------------------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256) void ink_combine_push(
    const unsigned short* __restrict__ C, int ld_c, const int* __restrict__ pos,
    const float* __restrict__ wts, float* __restrict__ y, int ld_y, int D, int acc, float* pwin,
    int ld_win, unsigned* pflags, unsigned* ctr) {
  __shared__ unsigned s_call;
  const int t = blockIdx.x;
  if (threadIdx.x == 0) {
    const unsigned s = ctr[t] + 1u;
    ctr[t] = s;
    s_call = s;
  }
  __syncthreads();
  const unsigned s = s_call, slot = s & 1u;
  int p[8];
  float wt[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    p[j] = pos[t * 8 + j];
    wt[j] = wts[t * 8 + j];
  }
  for (int c = threadIdx.x; c < D; c += 256) {
    float sm = acc ? y[(long)t * ld_y + c] : 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (p[j] >= 0) sm = sm + wt[j] * fbf(C[(long)p[j] * ld_c + c]);
    y[(long)t * ld_y + c] = sm;
    pwin[((long)slot * 8 + t) * ld_win + c] = sm;
  }
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) xf_rel(&pflags[t], s);
}

// ------------------------------------------------------------------------------------------------
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

extern "C" __global__ __launch_bounds__(256) void ink_xsd(
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
          __builtin_bit_cast(float, xf_ld_sys((const unsigned*)&win[((long)slot * 8 + t) * ld_win + c]));
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

// ------------------------------------------------------------------------------------------------
// the route: ink_route_logits (verbatim, 128 lanes) + the last workgroup's ink_route_tail
__device__ __forceinline__ unsigned char ff2e4m3(float v) {
  const int p = __builtin_amdgcn_cvt_pk_fp8_f32(v, 0.f, 0, false);
  return (unsigned char)(p & 0xFF);
}
#define RT_E 128
extern "C" __global__ __launch_bounds__(128) void ink_route_fused(
    const float* __restrict__ h, int ld_h, const unsigned short* __restrict__ gain,
    const unsigned short* __restrict__ wr, const float* __restrict__ bias, float* u, int ld_u,
    float* sk, int D, int* __restrict__ ids, float* __restrict__ wts, float route_scale,
    unsigned char* __restrict__ q, int ld_q, float* __restrict__ scale, int T, int lo, int nloc,
    int* __restrict__ row_tok, int* __restrict__ row_slot, int* __restrict__ tile_e,
    int* __restrict__ tile_r0, int* __restrict__ tile_n, int* __restrict__ meta,
    int* __restrict__ pos, unsigned* arrive) {
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
  // the last arriver (agent-scope release / acquire around one counter, chosen by value)
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned d = __hip_atomic_fetch_add(arrive, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    s_last = (d == gridDim.x * gridDim.y - 1u) ? 1u : 0u;
  }
  __syncthreads();
  if (!s_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  const int tid = threadIdx.x;
  if (tid == 0) __hip_atomic_store(arrive, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  // ink_route_tail block 0 (top-8 of every row, then the route sort)
  __shared__ float key[RT_E];
  __shared__ float sel[8];
  __shared__ int selid[8];
  __shared__ int sids[256];
  for (int t = 0; t < T; ++t) {
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
      sids[t * 8 + tid] = selid[tid];
      wts[t * 8 + tid] = sel[tid] / (sum + 1e-20f) * route_scale;
    }
    __syncthreads();
  }
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
  // ink_route_tail blocks 1..T: the act quant of every row (an exact max, an elementwise convert)
  __shared__ float mred[4];
  for (int t = 0; t < T; ++t) {
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
    __syncthreads();
  }
}

// ------------------------------------------------------------------------------------------------
// INK10 task 2: the narrow-tile GEMV (GN outputs a workgroup, NB = 3 k-tiles in registers)
template <int GN>
__device__ __forceinline__ void gn_load(uint4 (&p)[GN / 8], const unsigned short* __restrict__ w,
                                        int ldw, int n0, int N, int k0, int tid) {
#pragma unroll
  for (int i = 0; i < GN / 8; ++i) {
    const int e = tid + 256 * i, r = e >> 5, u = e & 31;
    const int gn = n0 + r;
    p[i] = gn < N ? *reinterpret_cast<const uint4*>(w + (long)gn * ldw + k0 + u * 8)
                  : make_uint4(0u, 0u, 0u, 0u);
  }
}
template <int MM, int GATE>
__device__ __forceinline__ void gn_aload(float (&ap)[MM], const float* __restrict__ a, int lda,
                                         const float* __restrict__ g, int ldg, int k0, int tid) {
#pragma unroll
  for (int m = 0; m < MM; ++m) {
    if (GATE) {
      const float sg = 1.0f / (1.0f + expf(-g[(long)m * ldg + k0 + tid]));
      ap[m] = a[(long)m * lda + k0 + tid] * sg;
    } else {
      ap[m] = a[(long)m * lda + k0 + tid];
    }
  }
}
template <int MM, int GN>
__device__ __forceinline__ void gn_step(const uint4 (&p)[GN / 8], const float (&ap)[MM],
                                        unsigned* ws, float (*as)[GV_K], int tid) {
  GV_LDS_BARRIER();
#pragma unroll
  for (int i = 0; i < GN / 8; ++i) {
    const int e = tid + 256 * i, r = e >> 5, u = e & 31;
    unsigned* d = ws + r * GV_LDW + u * 4;
    d[0] = p[i].x;
    d[1] = p[i].y;
    d[2] = p[i].z;
    d[3] = p[i].w;
  }
#pragma unroll
  for (int m = 0; m < MM; ++m) as[m][tid] = ap[m];
  GV_LDS_BARRIER();
}
template <int MM, int GN>
__device__ __forceinline__ void gn_sum(const unsigned* ws, float (*as)[GV_K], float (&s)[GV_M],
                                       int tid) {
  if (tid < GN) {
    const unsigned* row = ws + tid * GV_LDW;
#pragma unroll 8
    for (int kk = 0; kk < GV_K; kk += 2) {
      const unsigned wd = row[kk >> 1];
      const float w0 = lo16(wd), w1 = hi16(wd);
#pragma unroll
      for (int m = 0; m < MM; ++m) s[m] = s[m] + as[m][kk] * w0;
#pragma unroll
      for (int m = 0; m < MM; ++m) s[m] = s[m] + as[m][kk + 1] * w1;
    }
  }
}
template <int MM, int GN, int GATE, int PUSH>
__device__ __forceinline__ void gn_body(const float* __restrict__ a, int lda,
                                        const float* __restrict__ g, int ldg,
                                        const unsigned short* __restrict__ w, int ldw,
                                        float* __restrict__ c, int ldc, int N, int K, int acc,
                                        float* pwin, int ld_win, unsigned slot, unsigned* ws,
                                        float (*as)[GV_K]) {
  const int tid = threadIdx.x;
  const int n0 = blockIdx.x * GN;
  uint4 p0[GN / 8], p1[GN / 8], p2[GN / 8];
  float a0[MM], a1[MM], a2[MM];
  float s[GV_M];
#pragma unroll
  for (int m = 0; m < GV_M; ++m) s[m] = 0.f;
  const int KT = K / GV_K;   // a multiple of 3 (the caller checks)
  gn_load<GN>(p0, w, ldw, n0, N, 0, tid);
  gn_aload<MM, GATE>(a0, a, lda, g, ldg, 0, tid);
  gn_load<GN>(p1, w, ldw, n0, N, GV_K, tid);
  gn_aload<MM, GATE>(a1, a, lda, g, ldg, GV_K, tid);
  gn_load<GN>(p2, w, ldw, n0, N, 2 * GV_K, tid);
  gn_aload<MM, GATE>(a2, a, lda, g, ldg, 2 * GV_K, tid);
  for (int kt = 0; kt < KT; kt += 3) {
    gn_step<MM, GN>(p0, a0, ws, as, tid);
    if (kt + 3 < KT) {
      gn_load<GN>(p0, w, ldw, n0, N, (kt + 3) * GV_K, tid);
      gn_aload<MM, GATE>(a0, a, lda, g, ldg, (kt + 3) * GV_K, tid);
    }
    gn_sum<MM, GN>(ws, as, s, tid);
    gn_step<MM, GN>(p1, a1, ws, as, tid);
    if (kt + 4 < KT) {
      gn_load<GN>(p1, w, ldw, n0, N, (kt + 4) * GV_K, tid);
      gn_aload<MM, GATE>(a1, a, lda, g, ldg, (kt + 4) * GV_K, tid);
    }
    gn_sum<MM, GN>(ws, as, s, tid);
    gn_step<MM, GN>(p2, a2, ws, as, tid);
    if (kt + 5 < KT) {
      gn_load<GN>(p2, w, ldw, n0, N, (kt + 5) * GV_K, tid);
      gn_aload<MM, GATE>(a2, a, lda, g, ldg, (kt + 5) * GV_K, tid);
    }
    gn_sum<MM, GN>(ws, as, s, tid);
  }
  const int gn = n0 + tid;
  if (tid < GN && gn < N) {
#pragma unroll
    for (int m = 0; m < MM; ++m) {
      float* p = c + (long)m * ldc + gn;
      const float v = acc ? *p + s[m] : s[m];
      *p = v;
      if (PUSH) pwin[((long)slot * GV_M + m) * ld_win + gn] = v;
    }
  }
}

template <int GN, int GATE, int PUSH>
__device__ __forceinline__ void gn_entry(const float* __restrict__ a, int lda,
                                         const float* __restrict__ g, int ldg,
                                         const unsigned short* __restrict__ w, int ldw,
                                         float* __restrict__ c, int ldc, int M, int N, int K,
                                         int acc, float* pwin, int ld_win, unsigned* pflags,
                                         unsigned* ctr) {
  __shared__ unsigned ws[GN * GV_LDW];
  __shared__ float as[GV_M][GV_K];
  __shared__ unsigned s_call;
  const int tid = threadIdx.x;
  unsigned s = 0, slot = 0;
  if (PUSH) {
    if (tid == 0) {
      const unsigned v = ctr[blockIdx.x] + 1u;
      ctr[blockIdx.x] = v;
      s_call = v;
    }
    __syncthreads();
    s = s_call;
    slot = s & 1u;
  }
#define GN_CASE(MM)                                                                              \
  gn_body<MM, GN, GATE, PUSH>(a, lda, g, ldg, w, ldw, c, ldc, N, K, acc, pwin, ld_win, slot, ws, \
                              as)
  switch (M) {
    case 1: GN_CASE(1); break;
    case 2: GN_CASE(2); break;
    case 3: GN_CASE(3); break;
    case 4: GN_CASE(4); break;
    case 5: GN_CASE(5); break;
    case 6: GN_CASE(6); break;
    case 7: GN_CASE(7); break;
    default: GN_CASE(8); break;
  }
#undef GN_CASE
  if (PUSH) {
    asm volatile("s_wait_storecnt 0x0" ::: "memory");
    __syncthreads();
    if (tid == 0) xf_rel(&pflags[blockIdx.x], s);
  }
}

#define INK_GEMV_N(GN)                                                                            \
  extern "C" __global__ __launch_bounds__(256) void ink_gemv_n##GN##_f32_bf16(                    \
      const float* __restrict__ a, int lda, const unsigned short* __restrict__ w, int ldw,        \
      float* __restrict__ c, int ldc, int M, int N, int K, int acc) {                             \
    gn_entry<GN, 0, 0>(a, lda, nullptr, 0, w, ldw, c, ldc, M, N, K, acc, nullptr, 0, nullptr,     \
                       nullptr);                                                                  \
  }                                                                                               \
  extern "C" __global__ __launch_bounds__(256) void ink_gemv_gate_push_n##GN(                     \
      const float* __restrict__ a, int lda, const float* __restrict__ g, int ldg,                 \
      const unsigned short* __restrict__ w, int ldw, float* __restrict__ c, int ldc, int M, int N, \
      int K, int acc, float* pwin, int ld_win, unsigned* pflags, unsigned* ctr) {                 \
    gn_entry<GN, 1, 1>(a, lda, g, ldg, w, ldw, c, ldc, M, N, K, acc, pwin, ld_win, pflags, ctr);  \
  }
INK_GEMV_N(16)
INK_GEMV_N(32)

// ------------------------------------------------------------------------------------------------
// INK10: the message gates of a visit in two launches (ink_fast.hip's ink_msg_z / ink_msg_out
// bodies, verbatim, one gate a blockIdx.z) + ink_loop.hip's ink_rmsnorm of each normed gate's row
// run by the LAST of that row's 36 out workgroups (a per (gate, row) counter, reset by it); the
// 64-lane workgroup forms ink_rmsnorm's 256 lane partials as 8 virtual waves (lane L of virtual
// wave w sums i = 32 w + L, + 256, .. in order, the same xor butterfly, then red[0..7] in order).
#define FM_D 2304
#define FM_PER 9
#define FM_R 64
struct MsgZSet {
  const float* h[3];
  const float* m[3];
  const unsigned short* wd[3];
  float* z[3];
};
struct MsgOSet {
  const float* h[3];
  const float* m[3];
  const float* z[3];
  const unsigned short* wf[3];
  const unsigned short* wg[3];
  float* o[3];
  const unsigned short* gain[3];   // null = no norm (the raw gate output in o)
  float* no[3];                    // the normed row
};
__device__ __forceinline__ float fsum256(float v, float* red) {
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

extern "C" __global__ __launch_bounds__(256) void ink_msg_z3(MsgZSet S, int ld, float eps) {
  __shared__ float in[2 * FM_D];
  __shared__ float red[8];
  __shared__ unsigned short win[8][512];
  const int t = blockIdx.x, b = blockIdx.y, gi = blockIdx.z, tid = threadIdx.x, lane = tid & 31,
            wv = tid >> 5;
  const float* h = S.h[gi];
  const float* m = S.m[gi];
  const unsigned short* wd = S.wd[gi];
  float* z = S.z[gi];
  float hv[FM_PER], mv[FM_PER], sh = 0.f, sm = 0.f;
#pragma unroll
  for (int i = 0; i < FM_PER; ++i) {
    hv[i] = h[(long)t * ld + tid + 256 * i];
    mv[i] = m[(long)t * ld + tid + 256 * i];
    sh = sh + hv[i] * hv[i];
    sm = sm + mv[i] * mv[i];
  }
  const float rh = 1.0f / sqrtf(fsum256(sh, red) / (float)FM_D + eps);
  const float rm = 1.0f / sqrtf(fsum256(sm, red) / (float)FM_D + eps);
#pragma unroll
  for (int i = 0; i < FM_PER; ++i) {
    in[tid + 256 * i] = hv[i] * rh;
    in[FM_D + tid + 256 * i] = mv[i] * rm;
  }
  __syncthreads();
  const int r = b * 8 + wv;
  const unsigned short* wr = wd + (long)r * 2 * FM_D;
  float d = 0.f;
  for (int c0 = 0; c0 < 2 * FM_D; c0 += 512) {
    const uint4 x0 = *reinterpret_cast<const uint4*>(wr + c0 + 16 * lane);
    const uint4 x1 = *reinterpret_cast<const uint4*>(wr + c0 + 16 * lane + 8);
    *reinterpret_cast<uint4*>(&win[wv][16 * lane]) = x0;
    *reinterpret_cast<uint4*>(&win[wv][16 * lane + 8]) = x1;
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int i = c0 + lane + 32 * j;
      d = d + in[i] * fbf(win[wv][lane + 32 * j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) d += __shfl_xor(d, off, 32);
  if (lane == 0) z[(long)t * FM_R + r] = d;
}

extern "C" __global__ __launch_bounds__(64) void ink_msg_out3n(MsgOSet S, int ld, float eps,
                                                                unsigned* arrive) {
  __shared__ float zs[FM_R];
  __shared__ float red[8];
  __shared__ unsigned s_last;
  const int t = blockIdx.y, gi = blockIdx.z, c = blockIdx.x * 64 + threadIdx.x;
  const float* z = S.z[gi];
  const unsigned short* wf = S.wf[gi];
  const unsigned short* wg = S.wg[gi];
  float* o = S.o[gi];
  zs[threadIdx.x] = z[(long)t * FM_R + threadIdx.x];
  __syncthreads();
  if (c < FM_D) {
    uint4 f4[8], g4[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      f4[i] = *reinterpret_cast<const uint4*>(wf + (long)c * FM_R + 8 * i);
      g4[i] = *reinterpret_cast<const uint4*>(wg + (long)c * FM_R + 8 * i);
    }
    float f = 0.f, g = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const unsigned fw[4] = {f4[i].x, f4[i].y, f4[i].z, f4[i].w};
      const unsigned gw[4] = {g4[i].x, g4[i].y, g4[i].z, g4[i].w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int j = 8 * i + 2 * q;
        f = f + zs[j] * lo16(fw[q]);
        g = g + zs[j] * lo16(gw[q]);
        f = f + zs[j + 1] * hi16(fw[q]);
        g = g + zs[j + 1] * hi16(gw[q]);
      }
    }
    const float hv = S.h[gi][(long)t * ld + c], mv = S.m[gi][(long)t * ld + c];
    o[(long)t * ld + c] = (1.0f - tanhf(f)) * hv + tanhf(g) * mv;
  }
  const unsigned short* gain = S.gain[gi];
  if (!gain) return;
  // the last of the row's out workgroups norms the row (ink_rmsnorm's arithmetic)
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned* ctr = &arrive[gi * 8 + t];
    const unsigned d = __hip_atomic_fetch_add(ctr, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    s_last = (d == gridDim.x - 1u) ? 1u : 0u;
    if (s_last) __hip_atomic_store(ctr, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  __syncthreads();
  if (!s_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  const float* x = o + (long)t * ld;
  const int lane = threadIdx.x & 31, rw = threadIdx.x >> 5;
#pragma unroll
  for (int vw = 0; vw < 8; vw += 2) {
    const int w = vw + rw, T = 32 * w + lane;
    float ss = 0.f;
    for (int i = T; i < FM_D; i += 256) {
      const float v = x[i];
      ss = ss + v * v;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor(ss, off, 32);
    if (lane == 0) red[w] = ss;
  }
  __syncthreads();
  float tot = red[0];
#pragma unroll
  for (int i = 1; i < 8; ++i) tot = tot + red[i];
  const float r = 1.0f / sqrtf(tot / (float)FM_D + eps);
  float* no = S.no[gi] + (long)t * ld;
  for (int i = threadIdx.x; i < FM_D; i += 64) no[i] = (x[i] * r) * fbf(gain[i]);
}
