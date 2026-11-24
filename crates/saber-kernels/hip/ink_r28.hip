// INK28 (docs/INKLING2.md section 9 INK28): the decode route on ONE workgroup a row. Saber's own.
//
// ink_route_r1   ink_fuse's ink_route_fused / ink_vt's ink_route_fused_rt (the same outputs, the same
//                arguments; `rarrive` unused) on a grid [T, 1] instead of [T, 16]: the 128 lanes of a
//                row's workgroup are the 128 experts, so every router logit is ONE lane's sequential
//                f32 sum over k = 0..D-1 in the parent's order (dot = dot + us[k] * w[e][k], each op
//                rounded: -ffp-contract=off) -> byte-equal. The parents ran 8 of 128 lanes a workgroup
//                (16 workgroups each re-normalising the row, staging 8 router rows into LDS at a
//                4,608-byte stride = one bank for all eight lanes) and a last-arriver tail; here the
//                lane e streams router row e straight from memory (three batches of 8 x 16 B in
//                flight, pinned ahead of the multiply-adds by scheduling barriers: an LDS-tiled form
//                left each tile's global loads behind its compute, 37.7 us at one row). The row's rmsnorm, top-8, weights and act quant are
//                the parents' bodies verbatim (order-free max / convert, the same partial-sum shape:
//                128 lanes, a 32-wide butterfly, ((r0 + r1) + r2) + r3); at T > 1 the last row to
//                finish runs the route sort (rt's body verbatim, ids read back after the acquire).
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
__device__ __forceinline__ unsigned char ff2e4m3(float v) {
  const int p = __builtin_amdgcn_cvt_pk_fp8_f32(v, 0.f, 0, false);
  return (unsigned char)(p & 0xFF);
}
#define R1_E 128
#define R1_D 2304
__device__ __forceinline__ void r1_ld8(uint4 (&q)[8], const uint4* __restrict__ row, int c) {
#pragma unroll
  for (int j = 0; j < 8; ++j) q[j] = row[c + j];
}
// 8 x 8 bf16 weights of chunk c.. against us[8 c ..]: dot = dot + u * w, k ascending
__device__ __forceinline__ void r1_dot8(float& dot, const uint4 (&q)[8], const float* us, int c) {
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const unsigned wv[4] = {q[j].x, q[j].y, q[j].z, q[j].w};
    const float* ua = us + 8 * (c + j);
#pragma unroll
    for (int qq = 0; qq < 4; ++qq) {
      dot = dot + ua[2 * qq] * lo16(wv[qq]);
      dot = dot + ua[2 * qq + 1] * hi16(wv[qq]);
    }
  }
}

extern "C" __global__ __launch_bounds__(128) void ink_route_r1(
    const float* __restrict__ h, int ld_h, const unsigned short* __restrict__ gain,
    const unsigned short* __restrict__ wr, const float* __restrict__ bias, float* u, int ld_u,
    float* sk, int D, int* __restrict__ ids, float* __restrict__ wts, float route_scale,
    unsigned char* __restrict__ q, int ld_q, float* __restrict__ scale, int T, int lo, int nloc,
    int* __restrict__ row_tok, int* __restrict__ row_slot, int* __restrict__ tile_e,
    int* __restrict__ tile_r0, int* __restrict__ tile_n, int* __restrict__ meta,
    int* __restrict__ pos, unsigned* arrive, unsigned* rarrive) {
  (void)D;
  (void)rarrive;
  __shared__ float us[R1_D];
  __shared__ float red[4];
  __shared__ unsigned s_last;
  __shared__ float key[R1_E];
  __shared__ float sel[8];
  __shared__ int selid[8];
  __shared__ int sids[256];
  __shared__ float mred[4];
  __shared__ int cnt[R1_E], seg[R1_E], tbase[R1_E];
  __shared__ int ce[256];
  const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  // lane tid streams router row tid straight from memory: three batches of 8 x 16 bytes in flight
  // (the loads are pinned ahead of the FMAs-that-are-not-FMAs by scheduling barriers)
  const uint4* wrow = reinterpret_cast<const uint4*>(wr + (long)tid * R1_D);
  uint4 qa[8], qb[8], qc[8];
  r1_ld8(qa, wrow, 0);
  r1_ld8(qb, wrow, 8);
  r1_ld8(qc, wrow, 16);
  // the row's rmsnorm (the parents' body: 128 lanes, strided partials, a 32-wide butterfly)
  float ss = 0.f;
  for (int i = tid; i < R1_D; i += 128) {
    const float x = h[(long)t * ld_h + i];
    us[i] = x;
    ss = ss + x * x;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o, 32);
  if (lane == 0) red[w] = ss;
  __syncthreads();
  const float tot = ((red[0] + red[1]) + red[2]) + red[3];
  const float r = 1.0f / sqrtf(tot / (float)R1_D + 1e-6f);
  __syncthreads();
  float mx = 0.f;
  for (int i = tid; i < R1_D; i += 128) {
    const float v = (us[i] * r) * fbf(gain[i]);
    us[i] = v;
    u[(long)t * ld_u + i] = v;
    mx = fmaxf(mx, fabsf(v));
  }
  __syncthreads();
  // expert tid's logit: one sequential sum over k in the parent's order
  float dot = 0.f;
  for (int c = 0; c < R1_D / 8; c += 24) {
    __builtin_amdgcn_sched_barrier(0);
    r1_dot8(dot, qa, us, c);
    r1_ld8(qa, wrow, min(c + 24, R1_D / 8 - 8));
    __builtin_amdgcn_sched_barrier(0);
    r1_dot8(dot, qb, us, c + 8);
    r1_ld8(qb, wrow, min(c + 32, R1_D / 8 - 8));
    __builtin_amdgcn_sched_barrier(0);
    r1_dot8(dot, qc, us, c + 16);
    r1_ld8(qc, wrow, min(c + 40, R1_D / 8 - 8));
  }
  const float s = 1.0f / (1.0f + expf(-dot));
  const float mk = s + bias[tid];
  sk[((long)t * 128 + tid) * 2 + 0] = mk;
  sk[((long)t * 128 + tid) * 2 + 1] = s;
  // top-8 + weights (ink_route_tail's row body, verbatim)
  key[tid] = mk;
  __syncthreads();
  {
    int rank = 0;
    for (int j = 0; j < 128; ++j) {
      const float kj = key[j];
      rank += (kj > mk) || (kj == mk && j < tid);
    }
    if (rank < 8) {
      sel[rank] = s;
      selid[rank] = tid;
    }
  }
  __syncthreads();
  if (tid < 8) {
    float sum = 0.f;
    for (int j = 0; j < 8; ++j) sum = sum + sel[j];
    ids[t * 8 + tid] = selid[tid];
    sids[tid] = selid[tid];
    wts[t * 8 + tid] = sel[tid] / (sum + 1e-20f) * route_scale;
  }
  // the act quant of the row (an exact max, an elementwise convert)
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
  if (lane == 0) mred[w] = mx;
  __syncthreads();
  mx = fmaxf(fmaxf(mred[0], mred[1]), fmaxf(mred[2], mred[3]));
  {
    const float sc = fmaxf(mx / 448.0f, 1e-30f);
    for (int i = tid; i < R1_D; i += 128) q[(long)t * ld_q + i] = ff2e4m3(us[i] / sc);
    if (tid == 0) scale[t] = sc;
  }
  if (T > 1) {
    // the last row to finish runs the sort (ink_route_fused_rt's form)
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
  }
  __syncthreads();
  // the route sort (ink_route_tail's body: the same tables; INK30: N = T x 8 <= 256 entries, two
  // a lane, every entry's stable rank over all N at once -- the tail's 256-wide pass)
  const int N = T * 8;
  cnt[tid] = 0;
  __syncthreads();
  for (int i = tid; i < N; i += 128) {
    const int e = sids[i] - lo;
    if (e >= 0 && e < nloc) atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  if (tid == 0) {
    int rr = 0, tb = 0;
    for (int e = 0; e < nloc; ++e) {
      seg[e] = rr;
      tbase[e] = tb;
      const int nt = (cnt[e] + 15) >> 4;
      rr += nt * 16;
      tb += nt;
    }
    meta[0] = tb;
    meta[1] = rr;
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
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int i = tid + 128 * h;
      int e = -1;
      if (i < N) {
        e = sids[i] - lo;
        if (e < 0 || e >= nloc) e = -1;
      }
      ce[i] = e;
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int i = tid + 128 * h;
      if (i >= N) break;
      const int e = ce[i];
      int rk = 0;
      if (e >= 0) {
        for (int j = 0; j < i; ++j) rk += ce[j] == e;
        const int row = seg[e] + rk;
        row_tok[row] = i >> 3;
        row_slot[row] = i;
        pos[i] = row;
      } else {
        pos[i] = -1;
      }
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
// ink_xsd2 -- ink_fuse's ink_xsd (the consumer of the push reduce: spin, x = own + peer, the sconv,
// the depth attend) with the depth attend's block reductions BATCHED: ink_xsd ran one bsum256 (a
// 32-wide butterfly + two barriers + an 8-term sum) per state's norm, per (query, state) score and
// per query's output norm -- S + S nq + nq dependent rounds of a single workgroup. Here every
// reduction of a phase shares one butterfly pass and one barrier pair: each value is still its own
// tree (the same per-thread partial, the same xor order, the same red[0] + .. + red[7] sum), so the
// bytes are ink_xsd's. Spin + sconv verbatim.
__device__ __forceinline__ unsigned xf_ld_sys(const unsigned* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
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
// n block sums of 256 lanes at once (n <= NMAX), each = bsum256's tree
template <int NMAX>
__device__ __forceinline__ void bsum256_n(float (&v)[NMAX], int n, float (*red)[8]) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
    for (int j = 0; j < NMAX; ++j)
      if (j < n) v[j] += __shfl_xor(v[j], o, 32);
  }
  __syncthreads();
  if (lane == 0) {
#pragma unroll
    for (int j = 0; j < NMAX; ++j)
      if (j < n) red[j][w] = v[j];
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < NMAX; ++j) {
    if (j < n) {
      float s = red[j][0];
#pragma unroll
      for (int i = 1; i < 8; ++i) s = s + red[j][i];
      v[j] = s;
    }
  }
}
extern "C" __global__ __launch_bounds__(256) void ink_xsd2(
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
  __shared__ float redb[DA_SMAX * DA_QMAX][8];
  // 3. ink_depth_attend, the reductions batched by phase (each value its own bsum256 tree)
  float k[DA_SMAX][DA_PER];
  float ssn[DA_SMAX];
#pragma unroll
  for (int si = 0; si < DA_SMAX; ++si) {
    ssn[si] = 0.f;
    if (si >= S) continue;
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < DA_PER; ++i) {
      const float x = P.s[si][(long)t * ld_s + tid + 256 * i];
      k[si][i] = x;
      ss = ss + x * x;
    }
    ssn[si] = ss;
  }
  bsum256_n<DA_SMAX>(ssn, S, redb);
  float dq[DA_SMAX * DA_QMAX];
#pragma unroll
  for (int si = 0; si < DA_SMAX; ++si) {
    if (si >= S) break;
    const float r = 1.0f / sqrtf(ssn[si] / (float)DA_D + eps);
#pragma unroll
    for (int i = 0; i < DA_PER; ++i) k[si][i] = k[si][i] * r;
  }
#pragma unroll
  for (int qi = 0; qi < DA_QMAX; ++qi) {
#pragma unroll
    for (int si = 0; si < DA_SMAX; ++si) {
      dq[qi * DA_SMAX + si] = 0.f;
      if (qi >= nq || si >= S) continue;
      float d = 0.f;
#pragma unroll
      for (int i = 0; i < DA_PER; ++i) d = d + k[si][i] * P.q[qi][tid + 256 * i];
      dq[qi * DA_SMAX + si] = d;
    }
  }
  // the scores of query qi are slots qi * DA_SMAX + si: reduce the first nq rows of 8 (unused = 0)
  bsum256_n<DA_SMAX * DA_QMAX>(dq, nq * DA_SMAX, redb);
  float sc[DA_QMAX][DA_SMAX];
#pragma unroll
  for (int qi = 0; qi < DA_QMAX; ++qi)
#pragma unroll
    for (int si = 0; si < DA_SMAX; ++si) sc[qi][si] = dq[qi * DA_SMAX + si] * 0.020833333333333332f;
  float out[DA_QMAX][DA_PER];
  float sso[DA_QMAX];
#pragma unroll
  for (int qi = 0; qi < DA_QMAX; ++qi) {
    sso[qi] = 0.f;
    if (qi >= nq) continue;
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
#pragma unroll
    for (int i = 0; i < DA_PER; ++i) out[qi][i] = 0.f;
#pragma unroll
    for (int si = 0; si < DA_SMAX; ++si) {
      if (si >= S) break;
      const float wgt = e[si] / den;
#pragma unroll
      for (int i = 0; i < DA_PER; ++i) out[qi][i] = out[qi][i] + wgt * k[si][i];
    }
    if (P.g[qi]) {
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < DA_PER; ++i) ss = ss + out[qi][i] * out[qi][i];
      sso[qi] = ss;
    }
  }
  bsum256_n<DA_QMAX>(sso, nq, redb);
#pragma unroll
  for (int qi = 0; qi < DA_QMAX; ++qi) {
    if (qi >= nq) break;
    if (P.g[qi]) {
      const float r = 1.0f / sqrtf(sso[qi] / (float)DA_D + eps);
#pragma unroll
      for (int i = 0; i < DA_PER; ++i) out[qi][i] = (out[qi][i] * r) * fbf(P.g[qi][tid + 256 * i]);
    }
#pragma unroll
    for (int i = 0; i < DA_PER; ++i) P.o[qi][(long)t * ld_qo + tid + 256 * i] = out[qi][i];
  }
}

// ------------------------------------------------------------------------------------------------
// INK30 (`--ink-fuse p30`): latency twins of the decode / verify train's single-workgroup kernels.
// Every twin computes the parent's values in the parent's order (byte-equal by construction); only
// the schedule moves:
//   * the cross-lane butterflies run on DPP16 `row_xmask` (lane ^ 1/2/4/8) and `v_permlanex16`
//     (lane ^ 16) -- the same partner lane as `__shfl_xor`, so the same (commutative) sum, without
//     the `ds_bpermute` LDS round trip a step (FN49b's form, `fn_dpp_xor`);
//   * a block sum's partials go to one of TWO LDS slots in turn, so one barrier a sum instead of
//     two (sum k writes slot k & 1 only after every lane passed sum k - 1's barrier, i.e. after
//     every read of sum k - 2's slot);
//   * every global load a phase needs is issued before its first dependent use (the parents left
//     them behind loop-carried stores / barriers: one memory latency an iteration).
__device__ __forceinline__ float p30_xor(float v, int o) {
  const int x = __builtin_bit_cast(int, v);
  int y;
  switch (o) {
    case 16: y = __builtin_amdgcn_permlanex16(x, x, 0x76543210u, 0xfedcba98u, false, false); break;
    case 8: y = __builtin_amdgcn_update_dpp(x, x, 0x168, 0xf, 0xf, false); break;
    case 4: y = __builtin_amdgcn_update_dpp(x, x, 0x164, 0xf, 0xf, false); break;
    case 2: y = __builtin_amdgcn_update_dpp(x, x, 0x162, 0xf, 0xf, false); break;
    default: y = __builtin_amdgcn_update_dpp(x, x, 0x161, 0xf, 0xf, false); break;
  }
  return __builtin_bit_cast(float, y);
}
// v += shfl_xor(v, o) for o = 16, 8, 4, 2, 1 (every lane of the wave active)
__device__ __forceinline__ float p30_wsum(float v) {
  v += p30_xor(v, 16);
  v += p30_xor(v, 8);
  v += p30_xor(v, 4);
  v += p30_xor(v, 2);
  v += p30_xor(v, 1);
  return v;
}
// bsum256's value (the same butterfly, red[0] + .. + red[7] in order) on one barrier
__device__ __forceinline__ float p30_bsum256(float v, float (*red)[8], int& ph) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  v = p30_wsum(v);
  if (lane == 0) red[ph][w] = v;
  __syncthreads();
  float s = red[ph][0];
#pragma unroll
  for (int i = 1; i < 8; ++i) s = s + red[ph][i];
  ph ^= 1;
  return s;
}
__device__ __forceinline__ void p30_rel(unsigned* p, unsigned v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// the depth attend (ink_fuse's ink_depth_attend body) of row t: every state row loaded before the
// first block sum (state S - 1 = `o`, this lane's own values yv when the pointer is o), the sums on
// DPP + one barrier. All lanes reach it (S, nq uniform).
__device__ __forceinline__ void p30_depth(const InkPtrs& P, int S, int nq, int ld_s, int ld_qo,
                                          float eps, int t, const float* o, int ld_o,
                                          const float (&yv)[DA_PER], float (*red)[8]) {
  const int tid = threadIdx.x;
  float k[DA_SMAX][DA_PER];
#pragma unroll
  for (int si = 0; si < DA_SMAX; ++si) {
    if (si >= S) break;
    if (P.s[si] == o && ld_s == ld_o) {
#pragma unroll
      for (int i = 0; i < DA_PER; ++i) k[si][i] = yv[i];
    } else {
#pragma unroll
      for (int i = 0; i < DA_PER; ++i) k[si][i] = P.s[si][(long)t * ld_s + tid + 256 * i];
    }
  }
  float qv[DA_QMAX][DA_PER];
#pragma unroll
  for (int qi = 0; qi < DA_QMAX; ++qi) {
    if (qi >= nq) break;
#pragma unroll
    for (int i = 0; i < DA_PER; ++i) qv[qi][i] = P.q[qi][tid + 256 * i];
  }
  int ph = 0;
  float sc[DA_QMAX][DA_SMAX];
#pragma unroll
  for (int si = 0; si < DA_SMAX; ++si) {
    if (si >= S) break;
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < DA_PER; ++i) ss = ss + k[si][i] * k[si][i];
    const float tot = p30_bsum256(ss, red, ph);
    const float r = 1.0f / sqrtf(tot / (float)DA_D + eps);
#pragma unroll
    for (int i = 0; i < DA_PER; ++i) k[si][i] = k[si][i] * r;
#pragma unroll
    for (int qi = 0; qi < DA_QMAX; ++qi) {
      if (qi >= nq) break;
      float d = 0.f;
#pragma unroll
      for (int i = 0; i < DA_PER; ++i) d = d + k[si][i] * qv[qi][i];
      sc[qi][si] = p30_bsum256(d, red, ph) * 0.020833333333333332f;
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
      float gv[DA_PER];
#pragma unroll
      for (int i = 0; i < DA_PER; ++i) gv[i] = fbf(P.g[qi][tid + 256 * i]);
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < DA_PER; ++i) ss = ss + out[i] * out[i];
      const float r = 1.0f / sqrtf(p30_bsum256(ss, red, ph) / (float)DA_D + eps);
#pragma unroll
      for (int i = 0; i < DA_PER; ++i) out[i] = (out[i] * r) * gv[i];
    }
#pragma unroll
    for (int i = 0; i < DA_PER; ++i) P.o[qi][(long)t * ld_qo + tid + 256 * i] = out[i];
  }
}
// INK30 task 4 (`--ink-fuse ...,wt`, MEASUREMENT ONLY): the _w twins add wall_clock64 (100 MHz)
// stamps -- entry, after the exchange's spin (the exposed peer wait), exit -- summed by thread 0
// into wt[0] (spin ticks), wt[1] (kernel ticks), wt[2] (launch rows); the bytes are the p30 twin's.
__device__ __forceinline__ void p30_stamp(unsigned long long* wt, unsigned long long a,
                                          unsigned long long b, unsigned long long c) {
  if (wt && threadIdx.x == 0) {
    atomicAdd(&wt[0], b - a);
    atomicAdd(&wt[1], c - a);
    atomicAdd(&wt[2], 1ull);
  }
}

// ink_fuse's ink_xsd (the push-reduce consumer + sconv + depth attend): the sconv's own inputs
// loaded before the spin, the peer's window right after it, the depth attend on p30_depth.
template <bool TS>
__device__ __forceinline__ void xsd_p_body(
    const float* __restrict__ own, int ld_own, const float* win, int ld_win, const unsigned* flags,
    const unsigned* ctr, int nfl, int per_row, int xmode, const unsigned short* __restrict__ w,
    float* st, const int* __restrict__ rseq, const float* __restrict__ resid, int ld_r,
    float* o, int ld_o, const InkPtrs& P, int S, int nq, int ld_s, int ld_qo, float eps,
    unsigned* diag, unsigned long long* wtk) {
  __shared__ float red[2][8];
  __shared__ unsigned s_slot[256];
  const unsigned long long c0 = TS ? wall_clock64() : 0ull;
  const int t = blockIdx.x, tid = threadIdx.x;
  float* sb = st + (long)rseq[t] * 3 * DA_D;
  float xo[DA_PER], s0v[DA_PER], s1v[DA_PER], s2v[DA_PER], rv[DA_PER], wt[DA_PER][4];
#pragma unroll
  for (int i = 0; i < DA_PER; ++i) {
    const int c = tid + 256 * i;
    xo[i] = own[(long)t * ld_own + c];
#pragma unroll
    for (int j = 0; j < 4; ++j) wt[i][j] = fbf(w[(long)c * 4 + j]);
    s0v[i] = sb[c];
    s1v[i] = sb[(long)DA_D + c];
    s2v[i] = sb[2L * DA_D + c];
    rv[i] = resid ? resid[(long)t * ld_r + c] : 0.f;
  }
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
  const unsigned long long c1 = TS ? wall_clock64() : 0ull;
  float pv[DA_PER];
  if (xmode) {
#pragma unroll
    for (int i = 0; i < DA_PER; ++i) {
      const int c = tid + 256 * i;
      const unsigned slot = per_row ? s_slot[0] : s_slot[c / (DA_D / nfl)];
      pv[i] = __builtin_bit_cast(
          float, xf_ld_sys((const unsigned*)&win[((long)slot * 8 + t) * ld_win + c]));
    }
  }
#pragma unroll
  for (int i = 0; i < DA_PER; ++i) {
    const int c = tid + 256 * i;
    float xt = xo[i];
    if (xmode) xt = xt + pv[i];
    float acc = 0.f;
    acc = acc + s0v[i] * wt[i][0];
    acc = acc + s1v[i] * wt[i][1];
    acc = acc + s2v[i] * wt[i][2];
    acc = acc + xt * wt[i][3];
    float y = (acc + xt) / 2.2360679774997898f;
    if (resid) y = y + rv[i];
    o[(long)t * ld_o + c] = y;
    sb[c] = s1v[i];
    sb[(long)DA_D + c] = s2v[i];
    sb[2L * DA_D + c] = xt;
    xo[i] = y;
  }
  if (nq > 0) p30_depth(P, S, nq, ld_s, ld_qo, eps, t, o, ld_o, xo, red);
  if (TS) p30_stamp(wtk, c0, c1, wall_clock64());
}
extern "C" __global__ __launch_bounds__(256) void ink_xsd_p(
    const float* __restrict__ own, int ld_own, const float* win, int ld_win, const unsigned* flags,
    const unsigned* ctr, int nfl, int per_row, int xmode, const unsigned short* __restrict__ w,
    float* st, const int* __restrict__ rseq, const float* __restrict__ resid, int ld_r,
    float* o, int ld_o, InkPtrs P, int S, int nq, int ld_s, int ld_qo, float eps, unsigned* diag) {
  xsd_p_body<false>(own, ld_own, win, ld_win, flags, ctr, nfl, per_row, xmode, w, st, rseq, resid,
                    ld_r, o, ld_o, P, S, nq, ld_s, ld_qo, eps, diag, nullptr);
}
extern "C" __global__ __launch_bounds__(256) void ink_xsd_pw(
    const float* __restrict__ own, int ld_own, const float* win, int ld_win, const unsigned* flags,
    const unsigned* ctr, int nfl, int per_row, int xmode, const unsigned short* __restrict__ w,
    float* st, const int* __restrict__ rseq, const float* __restrict__ resid, int ld_r,
    float* o, int ld_o, InkPtrs P, int S, int nq, int ld_s, int ld_qo, float eps, unsigned* diag,
    unsigned long long* wtk) {
  xsd_p_body<true>(own, ld_own, win, ld_win, flags, ctr, nfl, per_row, xmode, w, st, rseq, resid,
                   ld_r, o, ld_o, P, S, nq, ld_s, ld_qo, eps, diag, wtk);
}

// ink_vt's ink_xsdc (the chained T-row form: row t's conv over rows t-3..t, the history entry, the
// last arriver's state write): the own / state / history inputs loaded before the spin, the peer's
// window rows right after it, the depth attend on p30_depth; the tail verbatim.
template <bool TS>
__device__ __forceinline__ void xsdc_p_body(
    const float* __restrict__ own, int ld_own, const float* win, int ld_win, const unsigned* flags,
    const unsigned* ctr, int nfl, int per_row, int xmode, const unsigned short* __restrict__ w,
    float* st, const int* __restrict__ rseq, const float* __restrict__ resid, int ld_r,
    float* o, int ld_o, const InkPtrs& P, int S, int nq, int ld_s, int ld_qo, float eps,
    unsigned* diag, float* hist, unsigned* arrive, unsigned long long* wtk) {
  __shared__ float red[2][8];
  __shared__ unsigned s_slot[256];
  __shared__ unsigned s_last;
  const unsigned long long c0 = TS ? wall_clock64() : 0ull;
  const int t = blockIdx.x, tid = threadIdx.x, T = gridDim.x;
  const int r0 = t >= 3 ? t - 3 : 0;
  const float* sb = st + (long)rseq[t] * 3 * DA_D;
  // xv[i][j] = row t - 3 + j's x (own now, + the peer after the spin) or the carried state row
  float xv[DA_PER][4], wt[DA_PER][4], rv[DA_PER], h0[DA_PER][3];
#pragma unroll
  for (int i = 0; i < DA_PER; ++i) {
    const int c = tid + 256 * i;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int s = t - 3 + j;
      xv[i][j] = s >= 0 ? own[(long)s * ld_own + c] : sb[(long)(3 + s) * DA_D + c];
      wt[i][j] = fbf(w[(long)c * 4 + j]);
    }
    rv[i] = resid ? resid[(long)t * ld_r + c] : 0.f;
    if (t == 0) {
#pragma unroll
      for (int j = 0; j < 3; ++j) h0[i][j] = sb[(long)j * DA_D + c];
    }
  }
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
  const unsigned long long c1 = TS ? wall_clock64() : 0ull;
  if (xmode) {
    float pv[DA_PER][4];
#pragma unroll
    for (int i = 0; i < DA_PER; ++i) {
      const int c = tid + 256 * i;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int s = t - 3 + j;
        pv[i][j] = 0.f;
        if (s >= 0) {
          const unsigned slot = per_row ? s_slot[s - r0] : s_slot[c / (DA_D / nfl)];
          pv[i][j] = __builtin_bit_cast(
              float, xf_ld_sys((const unsigned*)&win[((long)slot * 8 + s) * ld_win + c]));
        }
      }
    }
#pragma unroll
    for (int i = 0; i < DA_PER; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        if (t - 3 + j >= 0) xv[i][j] = xv[i][j] + pv[i][j];
  }
  float yv[DA_PER];
#pragma unroll
  for (int i = 0; i < DA_PER; ++i) {
    const int c = tid + 256 * i;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) acc = acc + xv[i][j] * wt[i][j];
    float y = (acc + xv[i][3]) / 2.2360679774997898f;
    if (resid) y = y + rv[i];
    o[(long)t * ld_o + c] = y;
    yv[i] = y;
    hist[(long)(3 + t) * DA_D + c] = xv[i][3];
    if (t == 0) {
#pragma unroll
      for (int j = 0; j < 3; ++j) hist[(long)j * DA_D + c] = h0[i][j];
    }
  }
  if (nq > 0) p30_depth(P, S, nq, ld_s, ld_qo, eps, t, o, ld_o, yv, red);
  if (TS) p30_stamp(wtk, c0, c1, wall_clock64());
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
    float hv[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) hv[j] = hist[(long)(T + j) * DA_D + c];
#pragma unroll
    for (int j = 0; j < 3; ++j) so[(long)j * DA_D + c] = hv[j];
  }
}
#define P30_XSDC_ARGS                                                                              \
  const float *__restrict__ own, int ld_own, const float *win, int ld_win, const unsigned *flags,  \
      const unsigned *ctr, int nfl, int per_row, int xmode, const unsigned short *__restrict__ w, \
      float *st, const int *__restrict__ rseq, const float *__restrict__ resid, int ld_r,         \
      float *o, int ld_o, InkPtrs P, int S, int nq, int ld_s, int ld_qo, float eps,               \
      unsigned *diag, float *hist, unsigned *arrive
extern "C" __global__ __launch_bounds__(256) void ink_xsdc_p(P30_XSDC_ARGS) {
  xsdc_p_body<false>(own, ld_own, win, ld_win, flags, ctr, nfl, per_row, xmode, w, st, rseq, resid,
                     ld_r, o, ld_o, P, S, nq, ld_s, ld_qo, eps, diag, hist, arrive, nullptr);
}
extern "C" __global__ __launch_bounds__(256) void ink_xsdc_pw(P30_XSDC_ARGS,
                                                               unsigned long long* wtk) {
  xsdc_p_body<true>(own, ld_own, win, ld_win, flags, ctr, nfl, per_row, xmode, w, st, rseq, resid,
                    ld_r, o, ld_o, P, S, nq, ld_s, ld_qo, eps, diag, hist, arrive, wtk);
}

// ink_fuse's ink_combine_push: the row's 9 columns a lane loaded at once (the parent's loop
// carried its stores ahead of the next column's loads). D == 2304 (the exec's only D).
extern "C" __global__ __launch_bounds__(256) void ink_combine_push_p(
    const unsigned short* __restrict__ C, int ld_c, const int* __restrict__ pos,
    const float* __restrict__ wts, float* __restrict__ y, int ld_y, int D, int acc, float* pwin,
    int ld_win, unsigned* pflags, unsigned* ctr) {
  (void)D;
  __shared__ unsigned s_call;
  const int t = blockIdx.x;
  if (threadIdx.x == 0) {
    const unsigned s = ctr[t] + 1u;
    ctr[t] = s;
    s_call = s;
  }
  int p[8];
  float wt[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    p[j] = pos[t * 8 + j];
    wt[j] = wts[t * 8 + j];
  }
  unsigned short cv[DA_PER][8];
  float yv[DA_PER];
#pragma unroll
  for (int i = 0; i < DA_PER; ++i) {
    const int c = threadIdx.x + 256 * i;
    yv[i] = acc ? y[(long)t * ld_y + c] : 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) cv[i][j] = p[j] >= 0 ? C[(long)p[j] * ld_c + c] : (unsigned short)0;
  }
  __syncthreads();
  const unsigned s = s_call, slot = s & 1u;
#pragma unroll
  for (int i = 0; i < DA_PER; ++i) {
    const int c = threadIdx.x + 256 * i;
    float sm = yv[i];
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (p[j] >= 0) sm = sm + wt[j] * fbf(cv[i][j]);
    y[(long)t * ld_y + c] = sm;
    pwin[((long)slot * 8 + t) * ld_win + c] = sm;
  }
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) p30_rel(&pflags[t], s);
}

// ink_fuse's message gates (ink_msg_z3 + ink_msg_out3n), the same structs
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
  const unsigned short* gain[3];
  float* no[3];
};
// one 512-element chunk of ink_msg_z3's dot: the wave's own LDS row (ordered by the wave alone)
__device__ __forceinline__ void p30_z3_chunk(float& d, uint4 a, uint4 b, unsigned short* wrow,
                                             const float* in, int c0, int lane) {
  *reinterpret_cast<uint4*>(&wrow[16 * lane]) = a;
  *reinterpret_cast<uint4*>(&wrow[16 * lane + 8]) = b;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#pragma unroll
  for (int j = 0; j < 16; ++j) d = d + in[c0 + lane + 32 * j] * fbf(wrow[lane + 32 * j]);
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}
// ink_msg_z3: the wave's down-projection row loaded whole before the norms (18 x 16 B a lane), its
// per-wave LDS transpose ordered by the wave alone (win[wv] is that wave's own), the norms on DPP.
extern "C" __global__ __launch_bounds__(256) void ink_msg_z3_p(MsgZSet S, int ld, float eps) {
  __shared__ float in[2 * FM_D];
  __shared__ float red[2][8];
  __shared__ unsigned short win[8][512];
  const int t = blockIdx.x, b = blockIdx.y, gi = blockIdx.z, tid = threadIdx.x, lane = tid & 31,
            wv = tid >> 5;
  const float* h = S.h[gi];
  const float* m = S.m[gi];
  const unsigned short* wd = S.wd[gi];
  float* z = S.z[gi];
  const int r = b * 8 + wv;
  const unsigned short* wr = wd + (long)r * 2 * FM_D;
  uint4 x0[FM_PER], x1[FM_PER];
#pragma unroll
  for (int c = 0; c < FM_PER; ++c) {
    x0[c] = *reinterpret_cast<const uint4*>(wr + 512 * c + 16 * lane);
    x1[c] = *reinterpret_cast<const uint4*>(wr + 512 * c + 16 * lane + 8);
  }
  float hv[FM_PER], mv[FM_PER], sh = 0.f, sm = 0.f;
#pragma unroll
  for (int i = 0; i < FM_PER; ++i) {
    hv[i] = h[(long)t * ld + tid + 256 * i];
    mv[i] = m[(long)t * ld + tid + 256 * i];
    sh = sh + hv[i] * hv[i];
    sm = sm + mv[i] * mv[i];
  }
  int ph = 0;
  const float rh = 1.0f / sqrtf(p30_bsum256(sh, red, ph) / (float)FM_D + eps);
  const float rm = 1.0f / sqrtf(p30_bsum256(sm, red, ph) / (float)FM_D + eps);
#pragma unroll
  for (int i = 0; i < FM_PER; ++i) {
    in[tid + 256 * i] = hv[i] * rh;
    in[FM_D + tid + 256 * i] = mv[i] * rm;
  }
  __syncthreads();
  float d = 0.f;
  // nine explicit chunk steps (a loop over x0[c] left the register arrays dynamically indexed)
#define P30_Z3_STEP(C) p30_z3_chunk(d, x0[C], x1[C], win[wv], in, 512 * (C), lane)
  P30_Z3_STEP(0); P30_Z3_STEP(1); P30_Z3_STEP(2); P30_Z3_STEP(3); P30_Z3_STEP(4);
  P30_Z3_STEP(5); P30_Z3_STEP(6); P30_Z3_STEP(7); P30_Z3_STEP(8);
#undef P30_Z3_STEP
  d = p30_wsum(d);
  if (lane == 0) z[(long)t * FM_R + r] = d;
}

// ink_msg_out3n: the last arriver's row norm with the row's 36 values a lane and their gains loaded
// at once (the parent: a strided loop a virtual wave, then a 36-step write loop that re-read the
// row); each lane writes the normed elements it summed. Partial-sum shape unchanged.
extern "C" __global__ __launch_bounds__(64) void ink_msg_out3n_p(MsgOSet S, int ld, float eps,
                                                                  unsigned* arrive) {
  __shared__ float zs[FM_R];
  __shared__ float red[8];
  __shared__ unsigned s_last;
  const int t = blockIdx.y, gi = blockIdx.z, c = blockIdx.x * 64 + threadIdx.x;
  const float* z = S.z[gi];
  const unsigned short* wf = S.wf[gi];
  const unsigned short* wg = S.wg[gi];
  float* o = S.o[gi];
  uint4 f4[8], g4[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    f4[i] = *reinterpret_cast<const uint4*>(wf + (long)c * FM_R + 8 * i);
    g4[i] = *reinterpret_cast<const uint4*>(wg + (long)c * FM_R + 8 * i);
  }
  const float hv = S.h[gi][(long)t * ld + c], mv = S.m[gi][(long)t * ld + c];
  zs[threadIdx.x] = z[(long)t * FM_R + threadIdx.x];
  __syncthreads();
  {
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
    o[(long)t * ld + c] = (1.0f - tanhf(f)) * hv + tanhf(g) * mv;
  }
  const unsigned short* gain = S.gain[gi];
  if (!gain) return;
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
  // virtual wave w = vw + rw covers i = 32 w + lane + 256 k (k = 0..8): the parent's partials
  float xv[4][FM_PER], gv[4][FM_PER];
#pragma unroll
  for (int a = 0; a < 4; ++a) {
    const int T = 32 * (2 * a + rw) + lane;
#pragma unroll
    for (int kk = 0; kk < FM_PER; ++kk) {
      xv[a][kk] = x[T + 256 * kk];
      gv[a][kk] = fbf(gain[T + 256 * kk]);
    }
  }
#pragma unroll
  for (int a = 0; a < 4; ++a) {
    float ss = 0.f;
#pragma unroll
    for (int kk = 0; kk < FM_PER; ++kk) ss = ss + xv[a][kk] * xv[a][kk];
    ss = p30_wsum(ss);
    if (lane == 0) red[2 * a + rw] = ss;
  }
  __syncthreads();
  float tot = red[0];
#pragma unroll
  for (int i = 1; i < 8; ++i) tot = tot + red[i];
  const float r = 1.0f / sqrtf(tot / (float)FM_D + eps);
  float* no = S.no[gi] + (long)t * ld;
#pragma unroll
  for (int a = 0; a < 4; ++a) {
    const int T = 32 * (2 * a + rw) + lane;
#pragma unroll
    for (int kk = 0; kk < FM_PER; ++kk) no[T + 256 * kk] = (xv[a][kk] * r) * gv[a][kk];
  }
}
