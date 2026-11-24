// INK37 (docs/INKLING2.md section 9 INK37): prefill levers, Saber's own, every body BYTE-EQUAL to its control.
//
//   ink37_route_sort_e128_v2_t128   ink_p29's route sort (one workgroup an expert, the stable t-major
//       rank by a ballot prefix) on 128-row expert tiles: the tables of moe_fn153's pf_tm8 cell (item
//       `moe`, `--ink-p37 moe`). The per-row bytes of the expert GEMMs depend on the row alone (FN153:
//       fn153_w4a8b_pf_tm* == fn148_w4a8b_*_ks1_* byte for byte), so the tile height never moves a byte.
//   ink37_post_norm_route_e128_r{4,8}   ink_glue's ink_post_norm_route_e128 BYTE FOR BYTE on R rows a
//       workgroup (item `glue`): the router's bf16 rows are read once per R rows instead of once a row
//       (1,024 x 590 KB a layer of L2 traffic -> 1,024 / R), the normed row recomputed per 256-column
//       chunk into LDS with the parent's expression ((x * r) * gain), every dot in the parent's order
//       (i ascending, one rounding a multiply and an add), the parent's norm reduction (lane-strided
//       sums, xor tree, ((r0 + r1) + r2) + r3), sigmoid, rank and weights. grid ceil(T / R), block 128.
//
// Built with -ffp-contract=off (saber-kbuild's `ink_` rule): every f32 multiply and add rounds on its own.
#include <hip/hip_runtime.h>

// ------------------------------------------------------------------------------- the route sort
#define RS_E 128
template <int TR>
__device__ __forceinline__ void rs37(
    const int* __restrict__ ids, int T, int lo, int nloc,
    int* __restrict__ row_tok, int* __restrict__ row_slot, int* __restrict__ tile_e,
    int* __restrict__ tile_r0, int* __restrict__ tile_n, int* __restrict__ meta,
    int* __restrict__ pos) {
  __shared__ int cnt[RS_E];
  __shared__ int wsum[8];
  __shared__ int s_seg, s_tb, s_last;
  const int tid = threadIdx.x, lane = tid & 31, wv = tid >> 5, N = T * 8;
  const int e = blockIdx.x;
  if (tid < RS_E) cnt[tid] = 0;
  __syncthreads();
  for (int i = tid; i < N; i += 256) {
    const int x = ids[i] - lo;
    if (x >= 0 && x < nloc) atomicAdd(&cnt[x], 1);
  }
  __syncthreads();
  if (tid == 0) {
    int r = 0, tb = 0;
    for (int x = 0; x < nloc; ++x) {
      if (x == e) { s_seg = r; s_tb = tb; }
      const int nt = (cnt[x] + TR - 1) / TR;
      r += nt * TR;
      tb += nt;
    }
    if (e == 0) { meta[0] = tb; meta[1] = r; }
    s_last = -1;
  }
  __syncthreads();
  const int c = cnt[e], seg = s_seg, tb = s_tb;
  const int nt = (c + TR - 1) / TR;
  for (int k = tid; k < nt; k += 256) {
    tile_e[tb + k] = e;
    tile_r0[tb + k] = seg + TR * k;
    tile_n[tb + k] = min(TR, c - TR * k);
  }
  int run = 0;
  for (int b = 0; b < N; b += 256) {
    const int i = b + tid;
    int x = -1;
    if (i < N) {
      x = ids[i] - lo;
      if (x < 0 || x >= nloc) x = -1;
    }
    if (e == 0 && i < N && x < 0) pos[i] = -1;
    const bool hit = x == e;
    const unsigned long long bal = __ballot(hit);
    const unsigned lm = (unsigned)bal & ((1u << lane) - 1u);
    if (lane == 0) wsum[wv] = __popc((unsigned)bal);
    __syncthreads();
    int off = 0, tot = 0;
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      off += q < wv ? wsum[q] : 0;
      tot += wsum[q];
    }
    if (hit) {
      const int row = seg + run + off + __popc(lm);
      row_tok[row] = i >> 3;
      row_slot[row] = i;
      pos[i] = row;
      if (run + off + __popc(lm) == c - 1) s_last = i >> 3;
    }
    run += tot;
    __syncthreads();
  }
  __syncthreads();
  const int padded = nt * TR, lt = s_last;
  for (int k = c + tid; k < padded; k += 256) {
    row_tok[seg + k] = lt;
    row_slot[seg + k] = -1;
  }
}
extern "C" __global__ __launch_bounds__(256) void ink37_route_sort_e128_v2_t128(
    const int* __restrict__ ids, int T, int lo, int nloc, int* __restrict__ row_tok,
    int* __restrict__ row_slot, int* __restrict__ tile_e, int* __restrict__ tile_r0,
    int* __restrict__ tile_n, int* __restrict__ meta, int* __restrict__ pos) {
  rs37<128>(ids, T, lo, nloc, row_tok, row_slot, tile_e, tile_r0, tile_n, meta, pos);
}

// ------------------------------------------------------------------------------- the router on R rows
__device__ __forceinline__ float bf2f37(unsigned short u) {
  return __builtin_bit_cast(float, ((unsigned)u) << 16);
}
#define RT_CH 256   // columns a chunk of the normed rows in LDS

template <int R>
__device__ __forceinline__ void route_r(
    const float* __restrict__ h, int ld_h, const unsigned short* __restrict__ gain,
    const unsigned short* __restrict__ wr, const float* __restrict__ bias,
    float* __restrict__ u, int ld_u, int* __restrict__ ids, float* __restrict__ wts, int D,
    float route_scale, int T) {
  __shared__ float us[R][RT_CH];
  __shared__ float red[R][4];
  __shared__ float key[R][128];
  __shared__ float sel[R][8];
  __shared__ int selid[R][8];
  const int t0 = blockIdx.x * R, e = threadIdx.x, lane = e & 31, w = e >> 5;
  int tr[R];
#pragma unroll
  for (int r = 0; r < R; ++r) tr[r] = t0 + r < T ? t0 + r : T - 1;   // clamp, writes guarded
  // the norm: the parent's lane-strided sum, xor tree and wave order, a row at a time
#pragma unroll
  for (int r = 0; r < R; ++r) {
    float ss = 0.f;
    for (int i = e; i < D; i += 128) {
      const float x = h[(long)tr[r] * ld_h + i];
      ss = ss + x * x;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o, 32);
    if (lane == 0) red[r][w] = ss;
  }
  __syncthreads();
  float rn[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const float tot = ((red[r][0] + red[r][1]) + red[r][2]) + red[r][3];
    rn[r] = 1.0f / sqrtf(tot / (float)D + 1e-6f);
  }
  float dot[R];
#pragma unroll
  for (int r = 0; r < R; ++r) dot[r] = 0.f;
  const unsigned short* wrow = wr + (long)e * D;
  for (int c0 = 0; c0 < D; c0 += RT_CH) {
    const int nc = D - c0 < RT_CH ? D - c0 : RT_CH;
    __syncthreads();
#pragma unroll
    for (int r = 0; r < R; ++r) {
      for (int i = e; i < nc; i += 128) {
        const float v = (h[(long)tr[r] * ld_h + c0 + i] * rn[r]) * bf2f37(gain[c0 + i]);
        us[r][i] = v;
        if (t0 + r < T) u[(long)(t0 + r) * ld_u + c0 + i] = v;
      }
    }
    __syncthreads();
    if (nc == RT_CH && ((((unsigned long)(wrow + c0)) & 15) == 0)) {
      for (int i = 0; i < RT_CH; i += 8) {
        const uint4 q = *(const uint4*)(wrow + c0 + i);
        const unsigned int qq[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float wv = bf2f37((unsigned short)(qq[j >> 1] >> (16 * (j & 1))));
#pragma unroll
          for (int r = 0; r < R; ++r) dot[r] = dot[r] + us[r][i + j] * wv;
        }
      }
    } else {
      for (int i = 0; i < nc; ++i) {
        const float wv = bf2f37(wrow[c0 + i]);
#pragma unroll
        for (int r = 0; r < R; ++r) dot[r] = dot[r] + us[r][i] * wv;
      }
    }
  }
  float s[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    s[r] = 1.0f / (1.0f + expf(-dot[r]));
    key[r][e] = s[r] + bias[e];
  }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < R; ++r) {
    // rank of this expert: how many beat it (greater key, or equal key at a lower index)
    const float mk = key[r][e];
    int rank = 0;
    for (int j = 0; j < 128; ++j) {
      const float kj = key[r][j];
      rank += (kj > mk) || (kj == mk && j < e);
    }
    if (rank < 8) {
      sel[r][rank] = s[r];
      selid[r][rank] = e;
    }
  }
  __syncthreads();
  if (e < 8) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      if (t0 + r >= T) continue;
      float sum = 0.f;
      for (int j = 0; j < 8; ++j) sum = sum + sel[r][j];
      ids[(t0 + r) * 8 + e] = selid[r][e];
      wts[(t0 + r) * 8 + e] = sel[r][e] / (sum + 1e-20f) * route_scale;
    }
  }
}
#define RT_KERNEL(NAME, R)                                                                          \
  extern "C" __global__ __launch_bounds__(128) void NAME(                                           \
      const float* __restrict__ h, int ld_h, const unsigned short* __restrict__ gain,               \
      const unsigned short* __restrict__ wr, const float* __restrict__ bias,                        \
      float* __restrict__ u, int ld_u, int* __restrict__ ids, float* __restrict__ wts, int D,       \
      float route_scale, int T) {                                                                   \
    route_r<R>(h, ld_h, gain, wr, bias, u, ld_u, ids, wts, D, route_scale, T);                      \
  }
RT_KERNEL(ink37_post_norm_route_e128_r4, 4)
RT_KERNEL(ink37_post_norm_route_e128_r8, 8)

// ------------------------------------------------------------------------------- the message out on R rows
//   ink37_msg_out_r8   ink_fast's ink_msg_out BYTE FOR BYTE on 8 rows a workgroup (item `glue`): a column's
//       forget / gain weights (2 x 64 bf16) are loaded once per 8 rows instead of once a row; every row's
//       f / g sums in the parent's order, o = (1 - tanh f) h + tanh g m. grid (D / 64, ceil(T / 8)), block 64.
#define MO_R 8
#define MO_Z 64
__device__ __forceinline__ float lo16_37(unsigned w) { return __builtin_bit_cast(float, w << 16); }
__device__ __forceinline__ float hi16_37(unsigned w) { return __builtin_bit_cast(float, w & 0xFFFF0000u); }
extern "C" __global__ __launch_bounds__(64) void ink37_msg_out_r8(
    const float* __restrict__ h, int ld_h, const float* __restrict__ m, int ld_m,
    const float* __restrict__ z,
    const unsigned short* __restrict__ wf,   // [D, 64]
    const unsigned short* __restrict__ wg,   // [D, 64]
    float* __restrict__ o, int ld_o, int D, int T) {
  __shared__ float zs[MO_R][MO_Z];
  const int t0 = blockIdx.y * MO_R, c = blockIdx.x * 64 + threadIdx.x;
#pragma unroll
  for (int r = 0; r < MO_R; ++r) {
    const int t = t0 + r < T ? t0 + r : T - 1;
    zs[r][threadIdx.x] = z[(long)t * MO_Z + threadIdx.x];
  }
  __syncthreads();
  if (c >= D) return;
  uint4 f4[8], g4[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    f4[i] = *reinterpret_cast<const uint4*>(wf + (long)c * MO_Z + 8 * i);
    g4[i] = *reinterpret_cast<const uint4*>(wg + (long)c * MO_Z + 8 * i);
  }
  for (int r = 0; r < MO_R; ++r) {
    const int t = t0 + r;
    if (t >= T) break;
    float f = 0.f, g = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const unsigned fw[4] = {f4[i].x, f4[i].y, f4[i].z, f4[i].w};
      const unsigned gw[4] = {g4[i].x, g4[i].y, g4[i].z, g4[i].w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int j = 8 * i + 2 * q;
        f = f + zs[r][j] * lo16_37(fw[q]);
        g = g + zs[r][j] * lo16_37(gw[q]);
        f = f + zs[r][j + 1] * hi16_37(fw[q]);
        g = g + zs[r][j + 1] * hi16_37(gw[q]);
      }
    }
    const float hv = h[(long)t * ld_h + c], mv = m[(long)t * ld_m + c];
    o[(long)t * ld_o + c] = (1.0f - tanhf(f)) * hv + tanhf(g) * mv;
  }
}
