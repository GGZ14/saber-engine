// Inkling 2 (inkling2_loop) prefill attention, row-tiled, INK27 (docs/INKLING2.md section 9 INK27).
// Saber's own.
//
//   ink_fa_rt{8,16}  ink_fast.hip's ink_attn_bias_sink_v1 for R consecutive query rows of ONE
//       sequence a workgroup (a prefill chunk: row_seq equal, row_pos consecutive) instead of one
//       row: every key row (K and V) is loaded ONCE for the R rows (v1 reloads it for every row: the
//       long-prompt global layers' traffic is rows x keys x 512 B a head). Every value is v1's,
//       op for op, so the output is BYTE-equal to v1 on every row:
//         * each row keeps its OWN key tiles (base_r = lo_r + 256 i, lo_r = max(0, pos_r - window
//           + 1) for a local layer, 0 for a global one); the workgroup's key range is the union of
//           its rows' tile i (<= 256 + R - 1 keys);
//         * the logit s = (sequential d = 0..127 sum of q * k) * 0.08838834764831845 [+ the band
//           bias rb = sequential sum over the 16 features], -INF past the row's position;
//         * the tile max (exact), p = expf(s - mn) (0 past pos), corr = expf(m - mn) (0 at the
//           first tile), the tile sum in v1's tree: a 32-lane xor butterfly (o = 16, 8, 4, 2, 1)
//           over v1's lane / wave map of the tile (lane = u % 32, wave = u / 32), lane 0's value,
//           then the eight wave partials summed in wave order;
//         * PV: v1's thread (half, d) sum, sequential over the half's keys in key order, a fresh
//           sum a tile, acc = acc * corr + a; the final ((acc0 + acc1) * c) / lf with the sink.
//       Built with -ffp-contract=off (saber-kbuild's `ink_` row), as v1: every multiply and add
//       rounds on its own. Grid [ceil(T / R), H], 256 threads, the shipped v1 argument list.
#include <hip/hip_runtime.h>

__device__ __forceinline__ float fa_lo16(unsigned w) { return __builtin_bit_cast(float, w << 16); }
__device__ __forceinline__ float fa_hi16(unsigned w) {
  return __builtin_bit_cast(float, w & 0xFFFF0000u);
}
__device__ __forceinline__ float fa_bf(unsigned short u) {
  return __builtin_bit_cast(float, static_cast<unsigned>(u) << 16);
}

#define RT_HD 128
#define RT_PAGE 64
#define RT_TILE 256

template <int R, bool TL>
__device__ __forceinline__ void ink_fa_rt_body(
    const unsigned short* __restrict__ q, int ld_q, const unsigned short* __restrict__ k_pool,
    const unsigned short* __restrict__ v_pool, const int* __restrict__ table, int table_ld,
    const int* __restrict__ row_seq, const int* __restrict__ row_pos, const float* __restrict__ rp,
    int ld_rp, const unsigned short* __restrict__ P, const float* __restrict__ sink, int ld_sink,
    float* __restrict__ o, int ld_o, int T, int H, int KV, int extent, int window) {
  __shared__ float qR[R][RT_HD];          // the rows' q (f32)
  __shared__ float sc[R][RT_TILE];        // the tile's logits, then its p; at the end half 1's acc
  __shared__ float rvs[R][16];
  // the tile's p by (half, key - the half's union base, row): every row's p of one key in one
  // 16-byte run, 0 where the key is outside the row's half range (0 x v adds +0: the identity of
  // every reachable partial sum, the key rows inside the union range are all written positions)
  __shared__ __attribute__((aligned(16))) float P2[2][RT_TILE / 2 + R][R];
  __shared__ float corr_s[R], m_s[R], l_s[R];
  __shared__ int pos_s[R], lo_s[R];       // pos -1 = a row past T
  const int t0 = blockIdx.x * R, h = blockIdx.y, tid = threadIdx.x;
  const int lane = tid & 31, wv = tid >> 5;
  const int kvh = h / (H / KV);
  const int C = KV * RT_HD;
  const int sq = row_seq[t0];
  if (tid < R) {
    const int pr = t0 + tid < T ? row_pos[t0 + tid] : -1;
    pos_s[tid] = pr;
    lo_s[tid] = window > 0 ? max(0, pr - window + 1) : 0;
    m_s[tid] = -INFINITY;
    l_s[tid] = 0.f;
  }
  for (int e = tid; e < RT_HD * R; e += 256) {
    const int r = e / RT_HD, dd = e % RT_HD;
    qR[r][dd] = t0 + r < T ? fa_bf(q[(long)(t0 + r) * ld_q + h * RT_HD + dd]) : 0.f;
  }
  for (int e = tid; e < 16 * R; e += 256) {
    const int r = e / 16, kk = e % 16;
    rvs[r][kk] = t0 + r < T ? rp[(long)(t0 + r) * ld_rp + h * 16 + kk] : 0.f;
  }
  __syncthreads();
  // the most tiles any row has (rows are consecutive positions: the last live row's lo / pos
  // bound every row's count, but take the max to stay general)
  int ntile = 0;
#pragma unroll
  for (int r = 0; r < R; ++r)
    if (pos_s[r] >= 0) ntile = max(ntile, (pos_s[r] - lo_s[r]) / RT_TILE + 1);
  float acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = 0.f;
  const int d = tid & (RT_HD - 1), half = tid >> 7;
  for (int it = 0; it < ntile; ++it) {
    // a row is ACTIVE in tile it when its tile base is <= its position
    int kmin = 0x7fffffff, kmax = -1;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int pr = pos_s[r], br = lo_s[r] + it * RT_TILE;
      if (pr >= 0 && br <= pr) {
        kmin = min(kmin, br);
        kmax = max(kmax, min(br + RT_TILE - 1, pr));
      }
    }
    // A: the logits of every (row, key) of the union range
    for (int e = tid; e < R * RT_TILE; e += 256) (&sc[0][0])[e] = -INFINITY;
    for (int e = tid; e < 2 * (RT_TILE / 2 + R) * R; e += 256) (&P2[0][0][0])[e] = 0.f;
    __syncthreads();
    // INK31 (TL): the main pass walks at most 256 keys (one key a thread); a local layer's union
    // of R rows' tiles is up to 256 + R - 1 keys, whose tail is spread one (row, key) a thread
    // below instead of R serial dots on R - 1 threads (same dot, same order: byte-equal)
    const int kend = TL ? min(kmax, kmin + RT_TILE - 1) : kmax;
    for (int k = kmin + tid; k <= kend; k += 256) {
      const long row = (long)table[(long)sq * table_ld + k / RT_PAGE] * RT_PAGE + k % RT_PAGE;
      const uint4* kr = reinterpret_cast<const uint4*>(k_pool + row * C + kvh * RT_HD);
      uint4 kv4[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) kv4[i] = kr[i];
#pragma unroll 1
      for (int r = 0; r < R; ++r) {
        const int pr = pos_s[r], br = lo_s[r] + it * RT_TILE;
        const int u = k - br;
        if (pr < 0 || u < 0 || u >= RT_TILE || k > pr) continue;
        const float* qr = qR[r];
        float dot = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const unsigned wv4[4] = {kv4[i].x, kv4[i].y, kv4[i].z, kv4[i].w};
#pragma unroll
          for (int qd = 0; qd < 4; ++qd) {
            dot = dot + qr[8 * i + 2 * qd] * fa_lo16(wv4[qd]);
            dot = dot + qr[8 * i + 2 * qd + 1] * fa_hi16(wv4[qd]);
          }
        }
        float sv = dot * 0.08838834764831845f;
        const int dist = pr - k;
        if (dist < extent) {
          float rb = 0.f;
#pragma unroll
          for (int kk = 0; kk < 16; ++kk) rb = rb + rvs[r][kk] * fa_bf(P[(long)kk * extent + dist]);
          sv = sv + rb;
        }
        sc[r][u] = sv;
      }
    }
    if constexpr (TL) {
      const int ntail = kmax - kend;
      for (int e = tid; e < ntail * R; e += 256) {
        const int k = kend + 1 + e / R, r = e % R;
        const int pr = pos_s[r], br = lo_s[r] + it * RT_TILE;
        const int u = k - br;
        if (pr < 0 || u < 0 || u >= RT_TILE || k > pr) continue;
        const long row = (long)table[(long)sq * table_ld + k / RT_PAGE] * RT_PAGE + k % RT_PAGE;
        const uint4* kr = reinterpret_cast<const uint4*>(k_pool + row * C + kvh * RT_HD);
        const float* qr = qR[r];
        float dot = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const uint4 kq = kr[i];
          const unsigned wv4[4] = {kq.x, kq.y, kq.z, kq.w};
#pragma unroll
          for (int qd = 0; qd < 4; ++qd) {
            dot = dot + qr[8 * i + 2 * qd] * fa_lo16(wv4[qd]);
            dot = dot + qr[8 * i + 2 * qd + 1] * fa_hi16(wv4[qd]);
          }
        }
        float sv = dot * 0.08838834764831845f;
        const int dist = pr - k;
        if (dist < extent) {
          float rb = 0.f;
#pragma unroll
          for (int kk = 0; kk < 16; ++kk) rb = rb + rvs[r][kk] * fa_bf(P[(long)kk * extent + dist]);
          sv = sv + rb;
        }
        sc[r][u] = sv;
      }
    }
    __syncthreads();
    // B: per row (wave w takes rows w, w + 8, ..): max, p, corr, the tile sum in v1's tree
    for (int r = wv; r < R; r += 8) {
      const int pr = pos_s[r], br = lo_s[r] + it * RT_TILE;
      if (pr < 0 || br > pr) continue;
      float v[8];
      float tm = -INFINITY;
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        v[w] = sc[r][w * 32 + lane];
        tm = fmaxf(tm, v[w]);
      }
#pragma unroll
      for (int ofs = 16; ofs > 0; ofs >>= 1) tm = fmaxf(tm, __shfl_xor(tm, ofs, 32));
      const float mo = m_s[r];
      const float mn = fmaxf(mo, tm);
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        const int u = w * 32 + lane;
        v[w] = (br + u <= pr) ? expf(v[w] - mn) : 0.f;
        if (br + u <= pr) P2[u >> 7][br - kmin + (u & 127)][r] = v[w];
      }
      float ts = 0.f;
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        float x = v[w];
#pragma unroll
        for (int ofs = 16; ofs > 0; ofs >>= 1) x += __shfl_xor(x, ofs, 32);
        x = __shfl(x, 0, 32);   // lane 0's value: v1's red[w]
        ts = w == 0 ? x : ts + x;
      }
      const float cr = (mo == -INFINITY) ? 0.f : expf(mo - mn);
      const float lo_ = l_s[r];
      // every lane of the wave read m_s / l_s above; one writes the new state after the wave's
      // reads (a wave executes in lockstep, the LDS ops in order)
      if (lane == 0) {
        corr_s[r] = cr;
        l_s[r] = lo_ * cr + ts;
        m_s[r] = mn;
      }
    }
    __syncthreads();
    // C: PV, thread (half, d), keys in key order over the union of the rows' half ranges
    float a[R];
#pragma unroll
    for (int r = 0; r < R; ++r) a[r] = 0.f;
    int jmin = 0x7fffffff, jmax = -1;
    int j0[R], jn[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int pr = pos_s[r], br = lo_s[r] + it * RT_TILE;
      j0[r] = br + half * (RT_TILE / 2);
      jn[r] = (pr < 0 || br > pr) ? 0 : min(RT_TILE / 2, pr - j0[r] + 1);
      if (jn[r] > 0) {
        jmin = min(jmin, j0[r]);
        jmax = max(jmax, j0[r] + jn[r] - 1);
      }
    }
    const int kb = kmin + half * (RT_TILE / 2);
    for (int k = jmin; k <= jmax; ++k) {
      const long row = (long)table[(long)sq * table_ld + k / RT_PAGE] * RT_PAGE + k % RT_PAGE;
      const float vv = fa_bf(v_pool[row * C + kvh * RT_HD + d]);
      const float4* pp = reinterpret_cast<const float4*>(P2[half][k - kb]);
#pragma unroll
      for (int q4 = 0; q4 < R / 4; ++q4) {
        const float4 p4 = pp[q4];
        a[4 * q4] = a[4 * q4] + p4.x * vv;
        a[4 * q4 + 1] = a[4 * q4 + 1] + p4.y * vv;
        a[4 * q4 + 2] = a[4 * q4 + 2] + p4.z * vv;
        a[4 * q4 + 3] = a[4 * q4 + 3] + p4.w * vv;
      }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int pr = pos_s[r], br = lo_s[r] + it * RT_TILE;
      if (pr >= 0 && br <= pr) acc[r] = acc[r] * corr_s[r] + a[r];
    }
    __syncthreads();
  }
  // oacc through the logits buffer: half 1 writes, half 0 combines (v1: oacc[0] + oacc[1])
  float* oacc = &sc[0][0];   // [R][128]
  if (half == 1) {
#pragma unroll
    for (int r = 0; r < R; ++r) oacc[r * RT_HD + d] = acc[r];
  }
  __syncthreads();
  if (half == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      if (pos_s[r] < 0) continue;
      const float mr = m_s[r], lr = l_s[r];
      const float sk = sink[(long)(t0 + r) * ld_sink + h];
      const float mf = fmaxf(mr, sk);
      const float c = expf(mr - mf);
      const float lf = lr * c + expf(sk - mf);
      o[(long)(t0 + r) * ld_o + h * RT_HD + d] = ((acc[r] + oacc[r * RT_HD + d]) * c) / lf;
    }
  }
}

#define INK_FA_RT(RR, NAME, TL)                                                                     \
  extern "C" __global__ __launch_bounds__(256) void NAME(                                           \
      const unsigned short* __restrict__ q, int ld_q, const unsigned short* __restrict__ k_pool,    \
      const unsigned short* __restrict__ v_pool, const int* __restrict__ table, int table_ld,       \
      const int* __restrict__ row_seq, const int* __restrict__ row_pos,                             \
      const float* __restrict__ rp, int ld_rp, const unsigned short* __restrict__ P,                \
      const float* __restrict__ sink, int ld_sink, float* __restrict__ o, int ld_o, int T, int H,   \
      int KV, int extent, int window) {                                                             \
    ink_fa_rt_body<RR, TL>(q, ld_q, k_pool, v_pool, table, table_ld, row_seq, row_pos, rp, ld_rp, P, \
                       sink, ld_sink, o, ld_o, T, H, KV, extent, window);                           \
  }
INK_FA_RT(8, ink_fa_rt8, false)
INK_FA_RT(16, ink_fa_rt16, false)
// INK31: the union tail spread one (row, key) a thread (byte-equal to ink_fa_rt8 / v1)
INK_FA_RT(8, ink_fa_rt8t, true)
