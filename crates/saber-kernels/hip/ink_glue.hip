// Inkling 2 (inkling2_loop) glue kernels, INK3 (docs/INKLING2.md §3.4-§3.5, §9). Saber's own,
// written from the architecture as docs/INKLING2.md states it; the numpy twins in
// tools/capture/ink3/ink3_probe.py (built from tools/capture/ink1/ink2_ref.py's formulas) are the
// spec every body here is bound-gated against.
//
//   ink_kv_conv_post_append   one visit's post-projection glue at one rank (KV heads 4, q heads 12):
//       k / v: the short conv over the projection rows (static: taps [C, 4], tap 3 = the current
//              token, (sum + x) / sqrt(5); dynamic (global layers): taps static[lag, C] + the row's
//              dyn[head, lag], lag 0 = the current token, (sum + x) / 3), the conv state (the last
//              3 INPUTS of each sequence, f32) read for the rows before a sequence's first row of
//              this launch and written by its last row; then k: per-head l2 clip at qk_clip; v:
//              per-head RMS (no gain, eps 1e-6); written bf16 to the K / V pools at the row's page
//              (block table [seq][pos / 64] * 64 + pos % 64).
//       q: x q_scale[h, d], l2 clip, x tau (global) -> bf16 q_post; r: x rel_scale, clip at
//              rel_clip, x tau -> f32 r_post (the rel-bias GEMM's A).
//   grid (ceil(T / INK_R), 2 KV + H), block 128 (one channel a lane); y < KV: a K head, < 2 KV: a V
//   head, else q head y - 2 KV (+ its r rows on lanes 0..15). Every row's window comes from its own
//   sequence (rows of a sequence are contiguous and in position order; row_seq[t] = the slot).
//   IN-PLACE state (state_out == state_in) is legal only when every sequence's rows sit inside one
//   R-row tile (the host's law, ink3::conv_inplace_ok): the thread that writes a channel's new
//   state is the one that read its old state.
//
// Built with -ffp-contract=off: every f32 multiply and add rounds on its own.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define INK_HD 128
#define INK_R 8       // rows a workgroup (INK3b: 16 spilled SGPRs; per-row sums do not depend on it)
#define INK_PAGE 64   // rows a KV page
#define INK_DREL 16

__device__ __forceinline__ float bf2f(unsigned short u) {
  return __builtin_bit_cast(float, ((unsigned)u) << 16);
}
__device__ __forceinline__ unsigned short f2bf(float x) {
  return __builtin_bit_cast(unsigned short, (__bf16)x);   // RNE
}

// sum over the workgroup's 128 lanes (4 waves of 32) of INK_R values a lane; `red` [4][INK_R]
__device__ __forceinline__ void sum128(float (&v)[INK_R], float (*red)[INK_R]) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
  for (int r = 0; r < INK_R; ++r) {
    float s = v[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor(s, o, 32);
    if (lane == 0) red[w][r] = s;
  }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < INK_R; ++r) v[r] = ((red[0][r] + red[1][r]) + red[2][r]) + red[3][r];
  __syncthreads();
}

__device__ __forceinline__ float tau_of(int pos, int global_layer, float n_floor, float alpha) {
  if (!global_layer) return 1.0f;
  const float p = (float)(pos + 1);
  return 1.0f + alpha * logf(fmaxf(p / n_floor, 1.0f));
}

extern "C" __global__ __launch_bounds__(128) void ink_kv_conv_post_append(
    // projection rows (f32), row strides in elements
    const float* __restrict__ kraw, const float* __restrict__ vraw, int ld_kv,
    const float* __restrict__ qraw, int ld_q, const float* __restrict__ rraw, int ld_r,
    const float* __restrict__ dyn_k, const float* __restrict__ dyn_v, int ld_dyn,   // global only
    // conv taps (bf16): coefficient of lag L at channel c = w[c * c_stride + lag0 + L * lag_stride]
    const unsigned short* __restrict__ wk, const unsigned short* __restrict__ wv, int c_stride,
    int lag0, int lag_stride,
    const unsigned short* __restrict__ q_scale,   // [H, 128] bf16
    const unsigned short* __restrict__ rel_scale, // [H, 16] bf16
    // conv states [slots][3][C] f32 (k, v), read / written
    const float* __restrict__ st_k_in, const float* __restrict__ st_v_in, float* __restrict__ st_k_out,
    float* __restrict__ st_v_out,
    // pools + tables
    unsigned short* __restrict__ k_pool, unsigned short* __restrict__ v_pool,   // [rows][KV * 128] bf16
    const int* __restrict__ table, int table_ld,   // [seq][table_ld] physical page
    const int* __restrict__ row_seq, const int* __restrict__ row_pos,
    unsigned short* __restrict__ q_post, int ld_qp,   // [T][H * 128] bf16
    float* __restrict__ r_post, int ld_rp,            // [T][H * 16] f32
    int T, int KV, int global_layer, float qk_clip, float rel_clip, float n_floor, float alpha) {
  __shared__ float red[4][INK_R];
  const int c0 = threadIdx.x;   // channel within the head
  const int t0 = blockIdx.x * INK_R;
  const int y = blockIdx.y;
  const int nrows = min(INK_R, T - t0);
  // per-row metadata staged in LDS once (INK3b: 16 unrolled rows of scalar-loaded tables spilled
  // SGPRs): the slot, the in-launch rows of its sequence behind the row (cap 3), the KV pool row,
  // the position and whether it is its sequence's last row here
  __shared__ int m_sq[INK_R], m_have[INK_R], m_prow[INK_R], m_pos[INK_R], m_last[INK_R];
  if (threadIdx.x < nrows) {
    const int t = t0 + threadIdx.x;
    const int sq = row_seq[t], pos = row_pos[t];
    const int b1 = (t >= 1) && row_seq[max(t - 1, 0)] == sq;
    const int b2 = b1 && (t >= 2) && row_seq[max(t - 2, 0)] == sq;
    const int b3 = b2 && (t >= 3) && row_seq[max(t - 3, 0)] == sq;
    m_sq[threadIdx.x] = sq;
    m_have[threadIdx.x] = b1 + b2 + b3;
    m_pos[threadIdx.x] = pos;
    m_prow[threadIdx.x] = table[(long)sq * table_ld + pos / INK_PAGE] * INK_PAGE + pos % INK_PAGE;
    m_last[threadIdx.x] = !(t + 1 < T && row_seq[t + 1] == sq);
  }
  __syncthreads();
  if (y < 2 * KV) {
    const int is_v = y >= KV;
    const int head = is_v ? y - KV : y;
    const int C = KV * INK_HD;
    const int c = head * INK_HD + c0;
    const float* raw = is_v ? vraw : kraw;
    const unsigned short* w = is_v ? wv : wk;
    const float* dyn = is_v ? dyn_v : dyn_k;
    const float* sin_ = is_v ? st_v_in : st_k_in;
    float* sout = is_v ? st_v_out : st_k_out;
    float tap[4];
#pragma unroll
    for (int L = 0; L < 4; ++L) tap[L] = bf2f(w[(long)c * c_stride + lag0 + L * lag_stride]);
    float y_[INK_R], ss[INK_R];
#pragma unroll
    for (int r = 0; r < INK_R; ++r) {
      y_[r] = 0.f;
      ss[r] = 0.f;
      if (r < nrows) {
        const int t = t0 + r;
        const int sq = m_sq[r];
        const float x0 = raw[(long)t * ld_kv + c];
        // the conv: x_{t - L} from this launch's rows of the same sequence, else the state.
        // `have` = this launch's rows of the sequence behind t (capped at 3), counted ONCE a row
        // (INK3b: the per-lag walk's nested divergent loops spilled 37 SGPRs; same values).
        const int have = m_have[r];
        float acc = 0.f;
#pragma unroll
        for (int L = 3; L >= 0; --L) {   // oldest first: lag 3 .. 0
          const int need = L - have;          // > 0: 1..3 rows into the state, newest = row 2
          const float* src = need <= 0 ? raw + (long)(t - L) * ld_kv + c
                                       : sin_ + ((long)sq * 3 + (3 - need)) * C + c;
          const float xv = *src;
          float wl = tap[L];
          if (global_layer) wl = wl + dyn[(long)t * ld_dyn + head * 4 + L];
          acc = acc + xv * wl;
        }
        const float v = global_layer ? (acc + x0) / 3.0f : (acc + x0) / 2.2360679774997898f;
        y_[r] = v;
        ss[r] = v * v;
      }
    }
    sum128(ss, red);
    // the post op + the KV write
    unsigned short* pool = is_v ? v_pool : k_pool;
#pragma unroll
    for (int r = 0; r < INK_R; ++r) {
      if (r < nrows) {
        const int t = t0 + r;
        float o;
        if (is_v) {
          o = y_[r] * (1.0f / sqrtf(ss[r] / 128.0f + 1e-6f));
        } else {
          const float nrm = sqrtf(ss[r]);
          o = y_[r] * fminf(qk_clip / (nrm + 1e-6f), 1.0f);
        }
        pool[(long)m_prow[r] * C + c] = f2bf(o);
      }
    }
    // the next state: the last 3 inputs of each sequence, by the thread that read the old one
    for (int r = 0; r < nrows; ++r) {
      const int t = t0 + r;
      const int sq = m_sq[r];
      if (!m_last[r]) continue;   // not the sequence's last row here
      float nx[3];
      const int have = min(m_have[r], 2);
#pragma unroll
      for (int j = 0; j < 3; ++j) {   // nx[j] = x_{t - 2 + j}
        const int L = 2 - j;
        const int need = L - have;
        const float* src = need <= 0 ? raw + (long)(t - L) * ld_kv + c
                                     : sin_ + ((long)sq * 3 + (3 - need)) * C + c;
        nx[j] = *src;
      }
#pragma unroll
      for (int j = 0; j < 3; ++j) sout[((long)sq * 3 + j) * C + c] = nx[j];
    }
  } else {
    // a q head (+ its r rows)
    const int h = y - 2 * KV;
    const float qs = bf2f(q_scale[h * INK_HD + c0]);
    float qv[INK_R], ss[INK_R];
#pragma unroll
    for (int r = 0; r < INK_R; ++r) {
      qv[r] = 0.f;
      ss[r] = 0.f;
      if (r < nrows) {
        qv[r] = qraw[(long)(t0 + r) * ld_q + h * INK_HD + c0] * qs;
        ss[r] = qv[r] * qv[r];
      }
    }
    sum128(ss, red);
#pragma unroll
    for (int r = 0; r < INK_R; ++r) {
      if (r < nrows) {
        const int t = t0 + r;
        const float tau = tau_of(m_pos[r], global_layer, n_floor, alpha);
        float o = qv[r] * fminf(qk_clip / (sqrtf(ss[r]) + 1e-6f), 1.0f);
        if (global_layer) o = o * tau;
        q_post[(long)t * ld_qp + h * INK_HD + c0] = f2bf(o);
      }
    }
    // r: 16 lanes of wave 0, one row at a time (a 16-lane xor reduce stays inside the wave)
    if (threadIdx.x < 32) {
      const int d = threadIdx.x & 15;
      const float rs = bf2f(rel_scale[h * INK_DREL + d]);
      for (int r = 0; r < nrows; ++r) {
        const int t = t0 + r;
        const float rv = rraw[(long)t * ld_r + h * INK_DREL + d] * rs;
        float s = rv * rv;
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) s += __shfl_xor(s, o, 16);
        float o = rv * fminf(rel_clip / (sqrtf(s) + 1e-6f), 1.0f);
        if (global_layer) o = o * tau_of(m_pos[r], 1, n_floor, alpha);
        if (threadIdx.x < 16) r_post[(long)t * ld_rp + h * INK_DREL + d] = o;
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
//   ink_post_norm_route_e128   the MLP input's gain RMS norm (eps 1e-6) -> u f32 [T, D], and the
//       router on it: s = sigmoid(u . Wr[e]) for the 128 experts (f32 dot, bf16 weights), top-8 by
//       s + bias[e] (ties: the lower expert index), weights = s_sel / (sum s_sel + 1e-20) x scale.
//       grid T, block 128 (one expert a lane; the norm's D elements strided over the lanes).
//       Outputs: u, ids [T, 8] i32 (descending by s + bias), wts [T, 8] f32.
extern "C" __global__ __launch_bounds__(128) void ink_post_norm_route_e128(
    const float* __restrict__ h, int ld_h, const unsigned short* __restrict__ gain,   // [D] bf16
    const unsigned short* __restrict__ wr,   // [128, D] bf16
    const float* __restrict__ bias,          // [128] f32
    float* __restrict__ u, int ld_u, int* __restrict__ ids, float* __restrict__ wts, int D,
    float route_scale) {
  __shared__ float us[2304];
  __shared__ float red[4];
  __shared__ float key[128];
  const int t = blockIdx.x, e = threadIdx.x, lane = e & 31, w = e >> 5;
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
    const float v = (us[i] * r) * bf2f(gain[i]);
    us[i] = v;
    u[(long)t * ld_u + i] = v;
  }
  __syncthreads();
  float dot = 0.f;
  const unsigned short* wrow = wr + (long)e * D;
  for (int i = 0; i < D; ++i) dot = dot + us[i] * bf2f(wrow[i]);
  const float s = 1.0f / (1.0f + expf(-dot));
  key[e] = s + bias[e];
  __syncthreads();
  // rank of this expert: how many beat it (greater key, or equal key at a lower index)
  const float mk = key[e];
  int rank = 0;
  for (int j = 0; j < 128; ++j) {
    const float kj = key[j];
    rank += (kj > mk) || (kj == mk && j < e);
  }
  __shared__ float sel[8];
  __shared__ int selid[8];
  if (rank < 8) {
    sel[rank] = s;
    selid[rank] = e;
  }
  __syncthreads();
  if (e < 8) {
    float sum = 0.f;
    for (int j = 0; j < 8; ++j) sum = sum + sel[j];
    ids[t * 8 + e] = selid[e];
    wts[t * 8 + e] = sel[e] / (sum + 1e-20f) * route_scale;
  }
}

// ------------------------------------------------------------------------------------------------
//   ink_swiglu_softcap   a = tanh(g / gc) gc sigmoid(g) tanh(u / uc) uc over the gate_up rows
//       [rows, 2 I] f32 (gate columns first, then up), f32 out [rows, I].
extern "C" __global__ __launch_bounds__(256) void ink_swiglu_softcap(
    const float* __restrict__ gu, int ld_gu, float* __restrict__ a, int ld_a, int rows, int I, float gc,
    float uc) {
  const long n = (long)rows * I;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < n; i += (long)gridDim.x * 256) {
    const long r = i / I, c = i % I;
    const float g = gu[r * ld_gu + c], up = gu[r * ld_gu + I + c];
    const float sg = 1.0f / (1.0f + expf(-g));
    a[r * ld_a + c] = ((tanhf(g / gc) * gc) * sg) * (tanhf(up / uc) * uc);
  }
}
