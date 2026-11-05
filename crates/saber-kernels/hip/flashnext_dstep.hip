// Flash-Next (FN119): THE DEVICE-RESIDENT STEP's sampler + acceptance -- the host sampler's draw
// and F40's drawn-row walk on the card, BIT FOR BIT, so the next pass can be issued one step ahead
// (round 2) without a numerics change. Its own code object: every shipped object keeps its bytes.
//
// WHAT THE HOST LAW IS (saber_serve::sampler, the served Flash-Next walk `verify::walk_slot`):
// a row's draw reads ONLY the merged seam top-64 (ids + bf16 values), the request's knobs and its
// noise key -- never the row's log-sum-exp, unless `top_p < 1` meets a tail that survived `top_k`
// / `min_p` (then the tail's mass needs lse_t: this kernel REFUSES the row, flag LIVE_TAIL, and the
// host decides it). So the device draw needs no log-sum-exp at all, and every float op below is
// the host's op, in the host's order, rounded where the host rounds:
//
//   merge   the two ranks' seams (each: 64 x (global id, f32 of a bf16), value desc, id asc) ->
//           the union's top 64 by (bf16 key desc, id asc) == pick2's `total_cmp` merge
//   greedy  temperature < 1e-5 (the host resolved it): the token is merged[0]
//   sampled v = value / T (IEEE div); min_p: softmax (max, glibc expf terms, a SEQUENTIAL f32 sum
//           in index order, IEEE div), pmax, mask p < pmax * min_p; top_k: threshold v[k-1]
//           (v is non-increasing here), mask v < threshold; top_p: ascending (value, index)
//           order, its own sequential f32 sum, cumulative <= 1 - p masked except the last; the
//           tail's life by the host's own comparisons (ceiling = floor / T); a survivor at or
//           below the ceiling = the host's refusal (flag CEIL); the draw: softmax again, score
//           = p / E(id), the FIRST strict maximum.
//   noise   E(id) = -(ln u) as f32, u = ((splitmix64(key ^ splitmix64(id)) >> 11) + 1) * 2^-53:
//           ln in f64 (ocml); the f32 rounding of -ln u is exact unless the f64 value lies within
//           FN119_GUARD f64 ulps of an f32 rounding midpoint -- then flag GUARD (host decides;
//           ~1e-7 a candidate): never a different E than glibc's by construction
//   expf    glibc 2.39's `__expf_fma` (ARM optimized-routines: 32-entry table, cubic in f64, FMA
//           contraction at the three a*b+c sites AND r = fma(InvLn2N, x, -kd)) -- gated EXHAUSTIVELY on all 2^32 inputs against
//           the container's libm (`saber fn119-oracle expf`) and on the card (`fn119_expf_probe`)
//   walk    rows j = 0..w-1 of a sequence decide positions pos + j (the host passes each row's
//           key); targets[j] = draw; the walk stops at j == ndraft or draw != draft[j]; acc =
//           targets - 1, tokens = targets (== saber_engine::drawn_rows + Greedy.accept)
//
// Flags (per row; a sequence with any flagged row in its walk is the host's): NAN 1 (a NaN or inf
// value in the merged seam), HOST 2 (the host marked the row: penalties / a dense row), TAIL 4
// (lse_t needed), CEIL 8 (a survivor at or below the ceiling), GUARD 16 (the noise rounding
// guard).
//
//   fn119_draw_walk  grid (nseq), block 32 x FN119_WMAX: wave j = row j of sequence b; lane l holds
//                    candidates l and l + 32; the sequential sums run on lane 0 over LDS.
//   fn119_expf_probe grid (n / 256), block 256: XOR-of-mix hash of expf(x) over a range of bit
//                    patterns, one u64 a workgroup (the exhaustive card gate).
//
// Built with -ffp-contract=off: every a*b+c below is written as the host writes it (`__builtin_fma`
// exactly where glibc's FMA build contracts).
#include <hip/hip_runtime.h>
#include <stdint.h>

#define FN119_K 64
#define FN119_WMAX 5
#define FN119_GUARD 64.0

#define F119_NAN 1u
#define F119_HOST 2u
#define F119_TAIL 4u
#define F119_CEIL 8u
#define F119_GUARDF 16u

__constant__ uint64_t fn119_tab[32] = {
    0x3ff0000000000000ull, 0x3fefd9b0d3158574ull, 0x3fefb5586cf9890full, 0x3fef9301d0125b51ull,
    0x3fef72b83c7d517bull, 0x3fef54873168b9aaull, 0x3fef387a6e756238ull, 0x3fef1e9df51fdee1ull,
    0x3fef06fe0a31b715ull, 0x3feef1a7373aa9cbull, 0x3feedea64c123422ull, 0x3feece086061892dull,
    0x3feebfdad5362a27ull, 0x3feeb42b569d4f82ull, 0x3feeab07dd485429ull, 0x3feea47eb03a5585ull,
    0x3feea09e667f3bcdull, 0x3fee9f75e8ec5f74ull, 0x3feea11473eb0187ull, 0x3feea589994cce13ull,
    0x3feeace5422aa0dbull, 0x3feeb737b0cdc5e5ull, 0x3feec49182a3f090ull, 0x3feed503b23e255dull,
    0x3feee89f995ad3adull, 0x3feeff76f2fb5e47ull, 0x3fef199bdd85529cull, 0x3fef3720dcef9069ull,
    0x3fef5818dcfba487ull, 0x3fef7c97337b9b5full, 0x3fefa4afa2a490daull, 0x3fefd0765b6e4540ull,
};

// glibc's expf (FMA build), op for op.
__device__ __noinline__ float fn119_expf(float x) {
  const uint32_t ux = __float_as_uint(x);
  const uint32_t abstop = (ux >> 20) & 0x7ffu;
  if (abstop >= (0x42b00000u >> 20)) {  // top12(88.0f)
    if (ux == 0xff800000u) return 0.0f;
    if (abstop >= (0x7f800000u >> 20)) return x + x;
    if (x > __uint_as_float(0x42b17217u)) return __uint_as_float(0x7f800000u);
    if (x < __uint_as_float(0xc2cff1b4u)) return 0.0f;
  }
  const double n = 32.0;
  const double inv = __longlong_as_double(0x3ff71547652b82fell) * n;
  const double c0 = __longlong_as_double(0x3fac6af84b912394ll) / n / n / n;
  const double c1 = __longlong_as_double(0x3fcebfce50fac4f3ll) / n / n;
  const double c2 = __longlong_as_double(0x3fe62e42ff0c52d6ll) / n;
  const double shift = __longlong_as_double(0x4338000000000000ll);
  const double xd = (double)x;
  const double z = inv * xd;
  double kd = z + shift;
  const uint64_t ki = (uint64_t)__double_as_longlong(kd);
  kd = kd - shift;
  const double r = __builtin_fma(inv, xd, -kd);  // glibc's FMA build contracts z - kd
  const uint64_t t = fn119_tab[ki & 31u] + (ki << 47);
  const double s = __longlong_as_double((long long)t);
  const double zz = __builtin_fma(c0, r, c1);
  const double r2 = r * r;
  const double y0 = __builtin_fma(c2, r, 1.0);
  const double y = __builtin_fma(zz, r2, y0);
  return (float)(y * s);
}

__device__ __forceinline__ uint64_t fn119_mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// E(id) = -(ln u) as f32; *guard set when the f64 value sits within FN119_GUARD ulps of an f32
// rounding midpoint (the only place an ocml-vs-glibc f64 difference could move the f32).
__device__ float fn119_noise(uint64_t key, uint32_t id, uint32_t* guard) {
  const uint64_t bits = fn119_mix(key ^ fn119_mix((uint64_t)id));
  const double u = (double)((bits >> 11) + 1ull) * __longlong_as_double(0x3ca0000000000000ll);
  const double L = -log(u);
  const float f = (float)L;
  if (L != 0.0) {
    // the f32 neighbours' midpoints around L
    const double fd = (double)f;
    const double up = (double)__uint_as_float(__float_as_uint(f) + 1u);
    const double dn = f > 0.0f ? (double)__uint_as_float(__float_as_uint(f) - 1u) : 0.0;
    const double m_hi = 0.5 * (fd + up), m_lo = 0.5 * (fd + dn);
    const double ulp64 = fabs(L) * 2.220446049250313e-16;
    if (fabs(L - m_hi) <= FN119_GUARD * ulp64 || fabs(L - m_lo) <= FN119_GUARD * ulp64) *guard = 1u;
  }
  return f;
}

// the bf16 key of an f32 that holds a bf16 (larger key = larger value; NaNs at the ends)
__device__ __forceinline__ uint32_t fn119_key(uint32_t fbits) {
  const uint32_t b = fbits >> 16;
  return (b & 0x8000u) ? (~b & 0xffffu) : (b | 0x8000u);
}

__device__ __forceinline__ float fn119_wave_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float w = __shfl_xor(v, o, 32);
    v = fmaxf(v, w);
  }
  return v;
}

__device__ __forceinline__ uint32_t fn119_wave_or(uint32_t v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v |= __shfl_xor(v, o, 32);
  return v;
}

// The host's `softmax(v)` into p (64 entries in LDS; lane 0 sums sequentially in index order).
__device__ void fn119_softmax(const float* v, float* e, float* p, float* s_sum, int lane) {
  float m = -INFINITY;
  for (int i = 0; i < FN119_K; ++i) m = fmaxf(m, v[i]);  // fold(NEG_INF, f32::max), every lane
  if (!(m - m == 0.0f)) {  // !is_finite: the host returns zeros
    p[lane] = 0.0f;
    p[lane + 32] = 0.0f;
    __builtin_amdgcn_wave_barrier();
    return;
  }
  e[lane] = fn119_expf(v[lane] - m);
  e[lane + 32] = fn119_expf(v[lane + 32] - m);
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
  if (lane == 0) {
    float s = -0.0f;
    for (int i = 0; i < FN119_K; ++i) s = s + e[i];
    *s_sum = s;
  }
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
  const float s = *s_sum;
  p[lane] = __fdiv_rn(e[lane], s);
  p[lane + 32] = __fdiv_rn(e[lane + 32], s);
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
}

#define WSYNC()                       \
  do {                                \
    __builtin_amdgcn_wave_barrier();  \
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront"); \
  } while (0)

// rp row (8 u32): [0] T f32, [1] top_p f32, [2] min_p f32, [3] top_k (>= 64 or vocab = no cut),
// [4] flags (bit0 greedy, bit1 host, bit2 top_k active), [5] seam row, [6..7] key u64
// sq seq (8 u32): [0] row0, [1] w, [2] ndraft, [3..7] drafts
// out seq (16 u32): [0] flags (OR over the walked rows), [1] acc, [2] ntok, [3..7] tokens,
//                   [8..12] row flags, [13..15] 0
// rowout row (2 u32): token, flags
extern "C" __global__ __launch_bounds__(32 * FN119_WMAX) void fn119_draw_walk(
    const uint32_t* __restrict__ seam_a, const uint32_t* __restrict__ seam_b,
    const uint32_t* __restrict__ rp, const uint32_t* __restrict__ sq, uint32_t* __restrict__ out,
    uint32_t* __restrict__ rowout, int nseq) {
  __shared__ uint32_t s_id[FN119_WMAX][FN119_K];
  __shared__ float s_v[FN119_WMAX][FN119_K];
  __shared__ float s_e[FN119_WMAX][FN119_K];
  __shared__ float s_p[FN119_WMAX][FN119_K];
  __shared__ uint32_t s_ck[FN119_WMAX][2 * FN119_K];
  __shared__ uint32_t s_ci[FN119_WMAX][2 * FN119_K];
  __shared__ float s_sum[FN119_WMAX];
  __shared__ uint32_t s_tok[FN119_WMAX], s_fl[FN119_WMAX];
  const int b = blockIdx.x;
  if (b >= nseq) return;
  const int j = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint32_t* q = sq + (size_t)b * 8;
  const int w = (int)q[1];
  if (j < w) {
    const int row = (int)q[0] + j;
    const uint32_t* prm = rp + (size_t)row * 8;
    uint32_t* ids = s_id[j];
    float* v = s_v[j];
    float* e = s_e[j];
    float* p = s_p[j];
    uint32_t* ck = s_ck[j];
    uint32_t* ci = s_ci[j];
    // 1. the merge: 128 candidates, 4 a lane
    const size_t srow = (size_t)prm[5];  // the row's seam (the oracle shares one across keys)
    const uint32_t* ha = seam_a + srow * 2 * FN119_K;
    const uint32_t* hb = seam_b + srow * 2 * FN119_K;
    for (int c = lane; c < 2 * FN119_K; c += 32) {
      const uint32_t* h = c < FN119_K ? ha + 2 * c : hb + 2 * (c - FN119_K);
      ci[c] = h[0];
      ck[c] = h[1];  // f32 bits for now
    }
    WSYNC();
    for (int c = lane; c < 2 * FN119_K; c += 32) {
      const uint32_t kc = fn119_key(ck[c]), ic = ci[c];
      uint32_t rank = 0;
      for (int d = 0; d < 2 * FN119_K; ++d) {
        const uint32_t kd = fn119_key(ck[d]), id = ci[d];
        rank += (kd > kc || (kd == kc && id < ic)) ? 1u : 0u;
      }
      if (rank < FN119_K) {
        ids[rank] = ic;
        v[rank] = __uint_as_float(ck[c]);
      }
    }
    WSYNC();
    uint32_t fl = prm[4] & 2u ? F119_HOST : 0u;
    {
      const float a = v[lane], c = v[lane + 32];
      const bool bad = !(a - a == 0.0f) || !(c - c == 0.0f);  // NaN or inf
      fl |= fn119_wave_or(bad ? F119_NAN : 0u);
    }
    uint32_t tok = ids[0];
    if (!(prm[4] & 1u) && fl == 0u) {
      const float T = __uint_as_float(prm[0]);
      const float top_p = __uint_as_float(prm[1]);
      const float min_p = __uint_as_float(prm[2]);
      const uint32_t top_k = prm[3];
      const float floor_v = v[FN119_K - 1];
      WSYNC();
      // temperature
      v[lane] = __fdiv_rn(v[lane], T);
      v[lane + 32] = __fdiv_rn(v[lane + 32], T);
      WSYNC();
      const float ceiling = __fdiv_rn(floor_v, T);
      int tail = 0;  // 0 whole, 1 partial, 2 dead
      if (min_p > 0.0f) {
        float mx = -INFINITY;
        for (int i = 0; i < FN119_K; ++i) mx = fmaxf(mx, v[i]);
        tail = fn119_expf(ceiling - mx) < min_p ? 2 : (tail == 2 ? 2 : 1);
        fn119_softmax(v, e, p, &s_sum[j], lane);
        float pm = 0.0f;
        for (int i = 0; i < FN119_K; ++i) pm = fmaxf(pm, p[i]);
        const float thr = pm * min_p;
        if (p[lane] < thr) v[lane] = -INFINITY;
        if (p[lane + 32] < thr) v[lane + 32] = -INFINITY;
        WSYNC();
      }
      if (prm[4] & 4u) {  // top_k < vocab (the host's resolved knob)
        float thr;
        if (top_k == 0u || top_k >= (uint32_t)FN119_K) {
          thr = INFINITY;
          for (int i = 0; i < FN119_K; ++i) thr = fminf(thr, v[i]);
        } else {
          thr = v[top_k - 1];
        }
        tail = thr > ceiling ? 2 : (tail == 2 ? 2 : 1);
        if (top_k != 0u && top_k < (uint32_t)FN119_K) {
          WSYNC();
          if (v[lane] < thr) v[lane] = -INFINITY;
          if (v[lane + 32] < thr) v[lane + 32] = -INFINITY;
          WSYNC();
        }
      }
      if (top_p < 1.0f) {
        if (tail != 2) {
          fl |= F119_TAIL;
        } else {
          // ascending (value, index) ranks
          uint32_t* rk = ci;  // reuse: rank of entry i in the ascending order
          for (int c = lane; c < FN119_K; c += 32) {
            const float vc = v[c];
            uint32_t r = 0;
            for (int d = 0; d < FN119_K; ++d) r += (v[d] < vc || (v[d] == vc && d < c)) ? 1u : 0u;
            rk[c] = r;
          }
          WSYNC();
          uint32_t* ord = ck;  // ord[r] = i
          for (int c = lane; c < FN119_K; c += 32) ord[rk[c]] = (uint32_t)c;
          WSYNC();
          const float mx = v[ord[FN119_K - 1]];
          e[lane] = fn119_expf(v[lane] - mx);
          e[lane + 32] = fn119_expf(v[lane + 32] - mx);
          WSYNC();
          if (lane == 0) {
            float s = 0.0f;
            for (int r = 0; r < FN119_K; ++r) s += e[ord[r]];
            for (int i = 0; i < FN119_K; ++i) p[i] = __fdiv_rn(e[i], s);
            const float cut = 1.0f - top_p;
            float cum = 0.0f;  // tail.mass (dead) = 0
            const float scale = 1.0f - 0.0f;
            for (int r = 0; r < FN119_K; ++r) {
              const uint32_t i = ord[r];
              cum += p[i] * scale;
              if (r + 1 != FN119_K && cum <= cut) v[i] = -INFINITY;
            }
          }
          WSYNC();
        }
      }
      if (!(fl & F119_TAIL)) {
        const bool under = (v[lane] - v[lane] == 0.0f && v[lane] <= ceiling) ||
                           (v[lane + 32] - v[lane + 32] == 0.0f && v[lane + 32] <= ceiling);
        fl |= fn119_wave_or(under ? F119_CEIL : 0u);
      }
      if (fl == 0u) {
        fn119_softmax(v, e, p, &s_sum[j], lane);
        const uint64_t key = (uint64_t)prm[6] | ((uint64_t)prm[7] << 32);
        uint32_t g = 0;
        const float p0 = p[lane], p1 = p[lane + 32];
        const float e0 = fn119_noise(key, ids[lane], &g);
        uint32_t g0 = p0 > 0.0f ? g : 0u;
        g = 0;
        const float e1 = fn119_noise(key, ids[lane + 32], &g);
        uint32_t g1 = p1 > 0.0f ? g : 0u;
        fl |= fn119_wave_or((g0 | g1) ? F119_GUARDF : 0u);
        e[lane] = __fdiv_rn(p0, e0);
        e[lane + 32] = __fdiv_rn(p1, e1);
        WSYNC();
        if (lane == 0) {
          int best = 0;
          float bs = -INFINITY;
          for (int i = 0; i < FN119_K; ++i) {
            const float sc = e[i];
            if (sc > bs) {
              bs = sc;
              best = i;
            }
          }
          s_tok[j] = ids[best];
        }
        WSYNC();
        tok = s_tok[j];
      }
    }
    if (lane == 0) {
      s_tok[j] = tok;
      s_fl[j] = fl;
      rowout[(size_t)row * 2] = tok;
      rowout[(size_t)row * 2 + 1] = fl;
    }
  }
  __syncthreads();
  // 2. the walk (drawn_rows + Greedy.accept)
  if (threadIdx.x == 0) {
    uint32_t* o = out + (size_t)b * 16;
    const int nd = (int)q[2];
    uint32_t sf = 0, n = 0;
    for (int r = 0; r < w; ++r) {
      sf |= s_fl[r];
      o[3 + r] = s_tok[r];
      ++n;
      if (r == nd || s_tok[r] != q[3 + r]) break;
    }
    for (int r = 0; r < FN119_WMAX; ++r) o[8 + r] = r < w ? s_fl[r] : 0u;
    for (int r = (int)n; r < FN119_WMAX; ++r) o[3 + r] = 0u;
    o[0] = sf;
    o[1] = n - 1u;
    o[2] = n;
    o[13] = o[14] = o[15] = 0u;
  }
}

// The exhaustive expf gate: workgroup g hashes expf over bit patterns [base + g*256*per,
// + 256*per): per thread `per` consecutive patterns, XOR of mix(bits(expf(x)) ^ (pattern << 32))
// over the block, one u64 a workgroup (the host computes the same table with glibc's expf).
extern "C" __global__ __launch_bounds__(256) void fn119_expf_probe(uint64_t* __restrict__ out,
                                                                    uint32_t base, int per) {
  __shared__ uint64_t s[256];
  const uint32_t t = threadIdx.x;
  const uint32_t g0 = base + (uint32_t)blockIdx.x * 256u * (uint32_t)per + t * (uint32_t)per;
  uint64_t h = 0;
  for (int i = 0; i < per; ++i) {
    const uint32_t u = g0 + (uint32_t)i;
    uint32_t r = __float_as_uint(fn119_expf(__uint_as_float(u)));
    if ((r & 0x7f800000u) == 0x7f800000u && (r & 0x7fffffu)) r = 0x7fc00000u;  // NaN class
    h ^= fn119_mix(((uint64_t)u << 32) | r);
  }
  s[t] = h;
  __syncthreads();
  for (int o = 128; o > 0; o >>= 1) {
    if (t < (uint32_t)o) s[t] ^= s[t + o];
    __syncthreads();
  }
  if (t == 0) out[blockIdx.x] = s[0];
}
