// Flash-Next MoE routing, combine and the A16 control (workstream FN4). NOT sliced from a reference
// .hip: upstream runs the reference server's `topk_softmax` + `moe_align_block_size` + fused_moe's own sum, so the
// arithmetic here is Saber's, DEFINED so that a numpy float32 program reproduces it bit for bit
// (tools/capture/fn4/moe_ref.py) -- built with `-ffp-contract=off` (saber-kbuild's flag table) so
// every product and sum rounds where the source says, and with an exp made of IEEE ops only.
//
//   moe_route_topk10_sort_e512   router logits [T, 512] bf16 -> softmax (fp32) -> top-10 with
//                                upstream's tie rule (the LOWER expert id wins a tie) and
//                                renormalisation (`norm_topk_prob`: w = p * (1 / max(sum, 0 -> 1)),
//                                The reference server's `scale /= denom` with scale 1) -> the expert-major sort of
//                                the rank's LOCAL experts [e_lo, e_lo + e_loc) into 16-row blocks,
//                                cut into tiles of `tile_blocks` blocks -> (optionally) the shared
//                                expert's rows (every token, one more "expert" at index e_loc).
//                                ONE launch: every workgroup routes 32 tokens and counts its routes
//                                per expert; the LAST workgroup to arrive (an arrival counter, the
//                                split-K reduce's pattern) scans and places every route. No atomic
//                                decides an output byte: counts are sums, positions are
//                                (expert offset + earlier workgroups' count + earlier tokens in this
//                                workgroup), i.e. routes of one expert in TOKEN order.
//   moe_combine_bf16             out[t] = sum over t's routes k = 0..9 IN TOP-K ORDER of
//                                w[t,k] * Y[pos(t,k)] (a non-local route contributes nothing; the
//                                other rank's all-reduce sums it) + sigmoid(g[t]) * Y[shared row t]
//                                when the shared rows exist, fp32, ONE bf16 rounding.
//   moe_grouped_a16_ref          the A16 control arm's grouped GEMM: bf16 activations times the
//                                exactly dequantised MXFP4 weight (code * 2^(e8m0 - 127)) in fp32,
//                                K in order, one bf16 rounding. A reference, not a fast path.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

#define MOE_EXPERTS 512   // router width (config num_experts)
#define MOE_TOPK 10       // num_experts_per_tok
#define MOE_TPB 32        // tokens per routing workgroup
#define MOE_THREADS 256   // 8 waves, 4 tokens each

static __device__ __forceinline__ float f32_bits(unsigned int u) { return __uint_as_float(u); }

// e^x for x <= 0, from IEEE-rounded mul/add/rint/ldexp only (bit-reproducible on the host):
//   n = rint(x * log2e); r = (x - n*C1) - n*C2; e^r by Horner (Taylor, degree 7, |r| <= 0.347);
//   ldexp(p, n). C1 = ln2's top 16 bits (n*C1 exact for |n| < 256). x <= -80 (or NaN) -> 0: a
// probability below e^-80 relative to the row's max cannot reach a top-10 of 512 real logits and
// cannot move a sum that holds the max's 1.0.
static __device__ __forceinline__ float saber_exp_nonpos(float x) {
  if (!(x > -80.f)) return 0.f;
  const float n = __builtin_rintf(x * f32_bits(0x3FB8AA3Bu));      // log2(e)
  float r = x - n * f32_bits(0x3F317200u);                         // C1 = 0.693145751953125
  r = r - n * f32_bits(0x35BFBE8Eu);                               // C2 = ln2 - C1
  float p = f32_bits(0x39500D01u);                                 // 1/5040
  p = p * r + f32_bits(0x3AB60B61u);                               // 1/720
  p = p * r + f32_bits(0x3C088889u);                               // 1/120
  p = p * r + f32_bits(0x3D2AAAABu);                               // 1/24
  p = p * r + f32_bits(0x3E2AAAABu);                               // 1/6
  p = p * r + 0.5f;
  p = p * r + 1.f;
  p = p * r + 1.f;
  return __builtin_ldexpf(p, (int)n);
}

static __device__ __forceinline__ float saber_sigmoid(float g) {
  const float e = saber_exp_nonpos(-fabsf(g));
  const float d = 1.f + e;
  return g >= 0.f ? 1.f / d : e / d;
}

// Block-wide exclusive scan of one int per thread (MOE_THREADS threads); returns the prefix and
// writes the total to *total. Integer, so order-free; two barriers.
static __device__ __forceinline__ int block_exscan(int v, int *s_wave, int *total) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  int x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up(x, o, 32);
    if (lane >= o) x += y;
  }
  if (lane == 31) s_wave[wave] = x;
  __syncthreads();
  int base = 0, all = 0;
#pragma unroll
  for (int w = 0; w < MOE_THREADS / 32; ++w) {
    const int s = s_wave[w];
    if (w < wave) base += s;
    all += s;
  }
  __syncthreads();
  *total = all;
  return base + x - v;
}

extern "C" __global__ __launch_bounds__(MOE_THREADS) void moe_route_topk10_sort_e512(
    const __bf16 *__restrict__ logits, int T, int e_lo, int e_loc, int tile_blocks,
    int with_shared, int *__restrict__ topk_ids, float *__restrict__ topk_w,
    int *__restrict__ route_pos, int *__restrict__ row_tok, int *__restrict__ row_route,
    int *__restrict__ tile_expert, int *__restrict__ tile_row0, int *__restrict__ tile_rows,
    int *__restrict__ expert_cnt, int *__restrict__ meta, int *__restrict__ blk_cnt,
    int *__restrict__ rank, int *__restrict__ arrive) {
  __shared__ int s_ids[MOE_TPB * MOE_TOPK];
  __shared__ int s_cnt[MOE_EXPERTS];
  __shared__ int s_off[MOE_EXPERTS + 1], s_toff[MOE_EXPERTS + 1];
  __shared__ int s_wave[MOE_THREADS / 32];
  __shared__ int s_last;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int nblk = gridDim.x, blk = blockIdx.x, t0 = blk * MOE_TPB;

  for (int e = tid; e < MOE_EXPERTS; e += MOE_THREADS) s_cnt[e] = 0;
  for (int i = tid; i < MOE_TPB * MOE_TOPK; i += MOE_THREADS) s_ids[i] = -1;
  __syncthreads();

  // ---- phase 1: softmax + top-10 + renormalise, one wave per token; lane l owns experts
  // [16l, 16l + 16). Sum order: each lane's 16 in index order, then the xor butterfly 16,8,4,2,1.
  for (int i = 0; i < MOE_TPB / (MOE_THREADS / 32); ++i) {
    const int tl = wave * (MOE_TPB / (MOE_THREADS / 32)) + i, t = t0 + tl;
    if (t >= T) break;
    float p[16];
    const uint4 *src = (const uint4 *)(logits + (size_t)t * MOE_EXPERTS + lane * 16);
    union { uint4 u; __bf16 h[8]; } v0, v1;
    v0.u = src[0];
    v1.u = src[1];
#pragma unroll
    for (int j = 0; j < 8; ++j) { p[j] = (float)v0.h[j]; p[8 + j] = (float)v1.h[j]; }
    float mx = p[0];
#pragma unroll
    for (int j = 1; j < 16; ++j) mx = fmaxf(mx, p[j]);
#pragma unroll
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) { p[j] = saber_exp_nonpos(p[j] - mx); s = s + p[j]; }
#pragma unroll
    for (int o = 16; o; o >>= 1) s = s + __shfl_xor(s, o, 32);
    const float inv = 1.f / s;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      float q = p[j] * inv;
      if (!__builtin_isfinite(q)) q = 0.f;   // upstream's NaN/Inf clamp
      p[j] = q;
    }
    float sel = 0.f, wk[MOE_TOPK];
    int ik[MOE_TOPK];
#pragma unroll
    for (int k = 0; k < MOE_TOPK; ++k) {
      float bv = p[0];
      int bj = 0;
#pragma unroll
      for (int j = 1; j < 16; ++j)
        if (p[j] > bv) { bv = p[j]; bj = j; }    // strict: the lower index keeps a tie
      int bi = lane * 16 + bj;
#pragma unroll
      for (int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor(bv, o, 32);
        const int oi = __shfl_xor(bi, o, 32);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
      if ((bi >> 4) == lane) {
#pragma unroll
        for (int j = 0; j < 16; ++j)
          if (j == (bi & 15)) p[j] = -1.f;       // probabilities are >= 0
      }
      wk[k] = bv;
      ik[k] = bi;
      sel = sel + bv;                            // k order, as upstream's selected_sum
    }
    const float scale = 1.f / (sel > 0.f ? sel : 1.f);
    if (lane < MOE_TOPK) {
      float wv = 0.f;
      int iv = 0;
#pragma unroll
      for (int k = 0; k < MOE_TOPK; ++k)
        if (k == lane) { wv = wk[k]; iv = ik[k]; }
      topk_ids[(size_t)t * MOE_TOPK + lane] = iv;
      topk_w[(size_t)t * MOE_TOPK + lane] = wv * scale;
      s_ids[tl * MOE_TOPK + lane] = iv;
    }
  }
  __syncthreads();

  // ---- phase 2: this workgroup's per-expert counts and each route's rank among the earlier
  // tokens of this workgroup routed to the same expert (a token's ten experts are distinct).
  for (int i = tid; i < MOE_TPB * MOE_TOPK; i += MOE_THREADS) {
    const int e = s_ids[i];
    const int le = e - e_lo;
    if (e < 0 || le < 0 || le >= e_loc) continue;
    atomicAdd(&s_cnt[le], 1);
    const int tl = i / MOE_TOPK;
    int rk = 0;
    for (int u = 0; u < tl * MOE_TOPK; ++u) rk += s_ids[u] == e;
    rank[(size_t)t0 * MOE_TOPK + i] = rk;
  }
  __syncthreads();
  for (int e = tid; e < e_loc; e += MOE_THREADS) blk_cnt[(size_t)blk * e_loc + e] = s_cnt[e];
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    s_last = atomicAdd(arrive, 1) == nblk - 1;
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();   // acquire: every workgroup's topk_ids, rank and blk_cnt
  if (tid == 0) *arrive = 0;

  // ---- phase 3 (the last workgroup): per-expert totals and each workgroup's base (in place).
  for (int e = tid; e < e_loc; e += MOE_THREADS) {
    int run = 0;
    for (int b = 0; b < nblk; ++b) {
      const int c = blk_cnt[(size_t)b * e_loc + e];
      blk_cnt[(size_t)b * e_loc + e] = run;
      run += c;
    }
    s_cnt[e] = run;
    expert_cnt[e] = run;
  }
  __syncthreads();
  // Padded rows and tiles per expert, scanned: two experts per thread (e_loc <= 512).
  const int G = tile_blocks, TR = 16 * tile_blocks;
  int r2 = 0, t2 = 0, touched = 0, routes = 0;
  for (int h = 0; h < 2; ++h) {
    const int e = 2 * tid + h;
    if (e < e_loc) {
      const int c = s_cnt[e], nb = (c + 15) / 16;
      r2 += nb * 16;
      t2 += (nb + G - 1) / G;
      touched += c > 0;
      routes += c;
    }
  }
  int rows_total, tiles_total, touched_total, routes_total;
  const int rbase = block_exscan(r2, s_wave, &rows_total);
  const int tbase = block_exscan(t2, s_wave, &tiles_total);
  block_exscan(touched, s_wave, &touched_total);
  block_exscan(routes, s_wave, &routes_total);
  {
    int ro = rbase, to = tbase;
    for (int h = 0; h < 2; ++h) {
      const int e = 2 * tid + h;
      if (e < e_loc) {
        const int c = s_cnt[e], nb = (c + 15) / 16;
        s_off[e] = ro;
        s_toff[e] = to;
        ro += nb * 16;
        to += (nb + G - 1) / G;
      }
    }
  }
  __syncthreads();
  const int sh_row0 = rows_total;
  const int sh_blocks = with_shared ? (T + 15) / 16 : 0;
  const int sh_tiles = (sh_blocks + G - 1) / G;
  if (tid == 0) {
    meta[0] = tiles_total + sh_tiles;
    meta[1] = rows_total + sh_blocks * 16;
    meta[2] = with_shared ? sh_row0 : -1;
    meta[3] = routes_total;
    meta[4] = touched_total;
    meta[5] = rows_total;
  }
  // Tiles and pad rows.
  for (int e = tid; e < e_loc; e += MOE_THREADS) {
    const int c = s_cnt[e], nb = (c + 15) / 16, nt = (nb + G - 1) / G;
    for (int j = 0; j < nt; ++j) {
      tile_expert[s_toff[e] + j] = e;
      tile_row0[s_toff[e] + j] = s_off[e] + j * TR;
      tile_rows[s_toff[e] + j] = min(TR, c - j * TR);
    }
    for (int r = s_off[e] + c; r < s_off[e] + nb * 16; ++r) { row_tok[r] = -1; row_route[r] = -1; }
  }
  for (int j = tid; j < sh_tiles; j += MOE_THREADS) {
    tile_expert[tiles_total + j] = e_loc;
    tile_row0[tiles_total + j] = sh_row0 + j * TR;
    tile_rows[tiles_total + j] = min(TR, T - j * TR);
  }
  for (int r = tid; r < sh_blocks * 16; r += MOE_THREADS) {
    row_tok[sh_row0 + r] = r < T ? r : -1;
    row_route[sh_row0 + r] = -1;
  }
  // Placement: route r of token t at s_off[e] + (earlier workgroups' count) + rank.
  for (int r = tid; r < T * MOE_TOPK; r += MOE_THREADS) {
    const int e = topk_ids[r], le = e - e_lo;
    if (le < 0 || le >= e_loc) { route_pos[r] = -1; continue; }
    const int t = r / MOE_TOPK, b = t / MOE_TPB;
    const int p = s_off[le] + blk_cnt[(size_t)b * e_loc + le] + rank[r];
    row_tok[p] = t;
    row_route[p] = r;
    route_pos[r] = p;
  }
}

// The weighted combine. One workgroup per token, 8 bf16 per thread-step.
extern "C" __global__ __launch_bounds__(256) void moe_combine_bf16(
    const __bf16 *__restrict__ Y, const float *__restrict__ topk_w, const int *__restrict__ route_pos,
    const __bf16 *__restrict__ sgate, const int *__restrict__ meta, __bf16 *__restrict__ out,
    int H) {
  const int t = blockIdx.x;
  __shared__ int s_pos[MOE_TOPK];
  __shared__ float s_w[MOE_TOPK];
  if (threadIdx.x < MOE_TOPK) {
    s_pos[threadIdx.x] = route_pos[(size_t)t * MOE_TOPK + threadIdx.x];
    s_w[threadIdx.x] = topk_w[(size_t)t * MOE_TOPK + threadIdx.x];
  }
  __syncthreads();
  const int sh0 = meta[2];
  const bool shared = sh0 >= 0 && sgate != nullptr;
  const float sg = shared ? saber_sigmoid((float)sgate[t]) : 0.f;
  union V { uint4 u; __bf16 h[8]; };
  for (int g = threadIdx.x; g < H / 8; g += blockDim.x) {
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = 0.f;
    for (int k = 0; k < MOE_TOPK; ++k) {
      const int p = s_pos[k];
      if (p < 0) continue;
      const float w = s_w[k];
      V y;
      y.u = ((const uint4 *)(Y + (size_t)p * H))[g];
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] = acc[j] + w * (float)y.h[j];
    }
    if (shared) {
      V y;
      y.u = ((const uint4 *)(Y + (size_t)(sh0 + t) * H))[g];
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] = acc[j] + sg * (float)y.h[j];
    }
    V o;
#pragma unroll
    for (int j = 0; j < 8; ++j) o.h[j] = (__bf16)acc[j];
    ((uint4 *)(out + (size_t)t * H))[g] = o.u;
  }
}

// The A16 control: C[row0 + m, n] = bf16( sum_k A[arow(m), k] * code(n, k) * 2^(ws[k/32, n] - 127) )
// over the tile table the W4A8 kernel reads, K in order, fp32. W in the pack's fragment order
// (`permute_w`: [N/16][K/16][2 k-halves][16 rows][4 bytes], low nibble = the even element).
// 16 x 16 outputs per workgroup (thread = one (m, n)); GATHER as in the W4A8 kernel.
__device__ __constant__ float kE2M1[16] = {0.f,  0.5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
                                           -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
template <bool GATHER>
__device__ __forceinline__ void moe_grouped_a16_impl(
    const __bf16 *__restrict__ A, const unsigned long long *__restrict__ xptr,
    const int *__restrict__ tile_expert, const int *__restrict__ tile_row0,
    const int *__restrict__ tile_rows, const int *__restrict__ row_tok,
    const int *__restrict__ meta, __bf16 *__restrict__ C, int N, int K) {
  const int tile = blockIdx.y;
  if (tile >= meta[0]) return;
  const int texp = tile_expert[tile], row0 = tile_row0[tile], M = tile_rows[tile];
  const int n = blockIdx.x * 16 + (threadIdx.x & 15);
  const int m = 16 * blockIdx.z + (threadIdx.x >> 4);   // grid z = the tile's 16-row blocks
  if (m >= M) return;
  {
    const unsigned char *W = (const unsigned char *)(uintptr_t)xptr[3 * (size_t)texp + 0];
    const unsigned char *Ws = (const unsigned char *)(uintptr_t)xptr[3 * (size_t)texp + 1];
    const int arow = GATHER ? row_tok[row0 + m] : row0 + m;
    const __bf16 *a = A + (size_t)arow * K;
    const int KS = K / 16, nt = n >> 4, nr = n & 15;
    float acc = 0.f, sc = 0.f;
    for (int k = 0; k < K; ++k) {
      if ((k & 31) == 0) sc = __int_as_float((int)Ws[(size_t)(k >> 5) * N + n] << 23);
      const size_t off = ((((size_t)nt * KS + (k >> 4)) * 2 + ((k >> 3) & 1)) * 16 + nr) * 4 + ((k & 7) >> 1);
      const int code = (W[off] >> ((k & 1) * 4)) & 0xF;
      acc = acc + (float)a[k] * (kE2M1[code] * sc);
    }
    C[(size_t)(row0 + m) * N + n] = (__bf16)acc;
  }
}

#define A16(name, G)                                                                              \
  extern "C" __global__ __launch_bounds__(256) void name(                                         \
      const __bf16 *__restrict__ A, const unsigned long long *__restrict__ xptr,                  \
      const int *__restrict__ tile_expert, const int *__restrict__ tile_row0,                     \
      const int *__restrict__ tile_rows, const int *__restrict__ row_tok,                         \
      const int *__restrict__ meta, __bf16 *__restrict__ C, int N, int K) {                       \
    moe_grouped_a16_impl<G>(A, xptr, tile_expert, tile_row0, tile_rows, row_tok, meta, C, N, K);  \
  }
A16(moe_grouped_a16_ref_g1, true)
A16(moe_grouped_a16_ref_g0, false)
#undef A16

// ---- FN26 v2 (GENERATED by tools/gen_moe_route_v2.py -- do not edit) ----
extern "C" __global__ __launch_bounds__(MOE_THREADS) void moe_route_topk10_sort_e512_v2(
    const __bf16 *__restrict__ logits, int T, int e_lo, int e_loc, int tile_blocks,
    int with_shared, int *__restrict__ topk_ids, float *__restrict__ topk_w,
    int *__restrict__ route_pos, int *__restrict__ row_tok, int *__restrict__ row_route,
    int *__restrict__ tile_expert, int *__restrict__ tile_row0, int *__restrict__ tile_rows,
    int *__restrict__ expert_cnt, int *__restrict__ meta, int *__restrict__ blk_cnt,
    int *__restrict__ rank, int *__restrict__ arrive) {
  __shared__ int s_ids[MOE_TPB * MOE_TOPK];
  __shared__ int s_cnt[MOE_EXPERTS];
  __shared__ int s_off[MOE_EXPERTS + 1], s_toff[MOE_EXPERTS + 1];
  __shared__ int s_wave[MOE_THREADS / 32];
  __shared__ int s_last;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int nblk = gridDim.x, blk = blockIdx.x, t0 = blk * MOE_TPB;

  for (int e = tid; e < MOE_EXPERTS; e += MOE_THREADS) s_cnt[e] = 0;
  for (int i = tid; i < MOE_TPB * MOE_TOPK; i += MOE_THREADS) s_ids[i] = -1;
  __syncthreads();

  // ---- phase 1: softmax + top-10 + renormalise, one wave per token; lane l owns experts
  // [16l, 16l + 16). Sum order: each lane's 16 in index order, then the xor butterfly 16,8,4,2,1.
  for (int i = 0; i < MOE_TPB / (MOE_THREADS / 32); ++i) {
    const int tl = i * (MOE_THREADS / 32) + wave, t = t0 + tl;   // FN26: interleaved
    if (t >= T) break;
    float p[16];
    const uint4 *src = (const uint4 *)(logits + (size_t)t * MOE_EXPERTS + lane * 16);
    union { uint4 u; __bf16 h[8]; } v0, v1;
    v0.u = src[0];
    v1.u = src[1];
#pragma unroll
    for (int j = 0; j < 8; ++j) { p[j] = (float)v0.h[j]; p[8 + j] = (float)v1.h[j]; }
    float mx = p[0];
#pragma unroll
    for (int j = 1; j < 16; ++j) mx = fmaxf(mx, p[j]);
#pragma unroll
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) { p[j] = saber_exp_nonpos(p[j] - mx); s = s + p[j]; }
#pragma unroll
    for (int o = 16; o; o >>= 1) s = s + __shfl_xor(s, o, 32);
    const float inv = 1.f / s;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      float q = p[j] * inv;
      if (!__builtin_isfinite(q)) q = 0.f;   // upstream's NaN/Inf clamp
      p[j] = q;
    }
    float sel = 0.f, wk[MOE_TOPK];
    int ik[MOE_TOPK];
#pragma unroll
    for (int k = 0; k < MOE_TOPK; ++k) {
      float bv = p[0];
      int bj = 0;
#pragma unroll
      for (int j = 1; j < 16; ++j)
        if (p[j] > bv) { bv = p[j]; bj = j; }    // strict: the lower index keeps a tie
      int bi = lane * 16 + bj;
#pragma unroll
      for (int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor(bv, o, 32);
        const int oi = __shfl_xor(bi, o, 32);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
      if ((bi >> 4) == lane) {
#pragma unroll
        for (int j = 0; j < 16; ++j)
          if (j == (bi & 15)) p[j] = -1.f;       // probabilities are >= 0
      }
      wk[k] = bv;
      ik[k] = bi;
      sel = sel + bv;                            // k order, as upstream's selected_sum
    }
    const float scale = 1.f / (sel > 0.f ? sel : 1.f);
    if (lane < MOE_TOPK) {
      float wv = 0.f;
      int iv = 0;
#pragma unroll
      for (int k = 0; k < MOE_TOPK; ++k)
        if (k == lane) { wv = wk[k]; iv = ik[k]; }
      topk_ids[(size_t)t * MOE_TOPK + lane] = iv;
      topk_w[(size_t)t * MOE_TOPK + lane] = wv * scale;
      s_ids[tl * MOE_TOPK + lane] = iv;
    }
  }
  __syncthreads();

  // ---- phase 2: this workgroup's per-expert counts and each route's rank among the earlier
  // tokens of this workgroup routed to the same expert (a token's ten experts are distinct).
  for (int i = tid; i < MOE_TPB * MOE_TOPK; i += MOE_THREADS) {
    const int e = s_ids[i];
    const int le = e - e_lo;
    if (e < 0 || le < 0 || le >= e_loc) continue;
    atomicAdd(&s_cnt[le], 1);
    const int tl = i / MOE_TOPK;
    int rk = 0;
    for (int u = 0; u < tl * MOE_TOPK; ++u) rk += s_ids[u] == e;
    rank[(size_t)t0 * MOE_TOPK + i] = rk;
  }
  __syncthreads();
  for (int e = tid; e < e_loc; e += MOE_THREADS) blk_cnt[(size_t)blk * e_loc + e] = s_cnt[e];
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    s_last = atomicAdd(arrive, 1) == nblk - 1;
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();   // acquire: every workgroup's topk_ids, rank and blk_cnt
  if (tid == 0) *arrive = 0;

  // ---- phase 3 (the last workgroup): per-expert totals and each workgroup's base (in place).
  for (int e = tid; e < e_loc; e += MOE_THREADS) {
    int run = 0;
    for (int b = 0; b < nblk; ++b) {
      const int c = blk_cnt[(size_t)b * e_loc + e];
      blk_cnt[(size_t)b * e_loc + e] = run;
      run += c;
    }
    s_cnt[e] = run;
    expert_cnt[e] = run;
  }
  __syncthreads();
  // Padded rows and tiles per expert, scanned: two experts per thread (e_loc <= 512).
  const int G = tile_blocks, TR = 16 * tile_blocks;
  int r2 = 0, t2 = 0, touched = 0, routes = 0;
  for (int h = 0; h < 2; ++h) {
    const int e = 2 * tid + h;
    if (e < e_loc) {
      const int c = s_cnt[e], nb = (c + 15) / 16;
      r2 += nb * 16;
      t2 += (nb + G - 1) / G;
      touched += c > 0;
      routes += c;
    }
  }
  int rows_total, tiles_total, touched_total, routes_total;
  const int rbase = block_exscan(r2, s_wave, &rows_total);
  const int tbase = block_exscan(t2, s_wave, &tiles_total);
  block_exscan(touched, s_wave, &touched_total);
  block_exscan(routes, s_wave, &routes_total);
  {
    int ro = rbase, to = tbase;
    for (int h = 0; h < 2; ++h) {
      const int e = 2 * tid + h;
      if (e < e_loc) {
        const int c = s_cnt[e], nb = (c + 15) / 16;
        s_off[e] = ro;
        s_toff[e] = to;
        ro += nb * 16;
        to += (nb + G - 1) / G;
      }
    }
  }
  __syncthreads();
  const int sh_row0 = rows_total;
  const int sh_blocks = with_shared ? (T + 15) / 16 : 0;
  const int sh_tiles = (sh_blocks + G - 1) / G;
  if (tid == 0) {
    meta[0] = tiles_total + sh_tiles;
    meta[1] = rows_total + sh_blocks * 16;
    meta[2] = with_shared ? sh_row0 : -1;
    meta[3] = routes_total;
    meta[4] = touched_total;
    meta[5] = rows_total;
  }
  // Tiles and pad rows.
  for (int e = tid; e < e_loc; e += MOE_THREADS) {
    const int c = s_cnt[e], nb = (c + 15) / 16, nt = (nb + G - 1) / G;
    for (int j = 0; j < nt; ++j) {
      tile_expert[s_toff[e] + j] = e;
      tile_row0[s_toff[e] + j] = s_off[e] + j * TR;
      tile_rows[s_toff[e] + j] = min(TR, c - j * TR);
    }
    for (int r = s_off[e] + c; r < s_off[e] + nb * 16; ++r) { row_tok[r] = -1; row_route[r] = -1; }
  }
  for (int j = tid; j < sh_tiles; j += MOE_THREADS) {
    tile_expert[tiles_total + j] = e_loc;
    tile_row0[tiles_total + j] = sh_row0 + j * TR;
    tile_rows[tiles_total + j] = min(TR, T - j * TR);
  }
  for (int r = tid; r < sh_blocks * 16; r += MOE_THREADS) {
    row_tok[sh_row0 + r] = r < T ? r : -1;
    row_route[sh_row0 + r] = -1;
  }
  // Placement: route r of token t at s_off[e] + (earlier workgroups' count) + rank.
  for (int r = tid; r < T * MOE_TOPK; r += MOE_THREADS) {
    const int e = topk_ids[r], le = e - e_lo;
    if (le < 0 || le >= e_loc) { route_pos[r] = -1; continue; }
    const int t = r / MOE_TOPK, b = t / MOE_TPB;
    const int p = s_off[le] + blk_cnt[(size_t)b * e_loc + le] + rank[r];
    row_tok[p] = t;
    row_route[p] = r;
    route_pos[r] = p;
  }
}
// ---- end FN26 v2 ----

// ---- FN31 front (GENERATED by tools/gen_moe_front.py -- do not edit) ----
// FN31: the front half of the MoE block. See tools/gen_moe_front.py for the law; every fp32 line
// below is the shipped band's or FN4's route's text (anchor-checked by the deriver).
#define MF_TMAX 64                    // tokens the LDS route holds (moe::BackTrain::MAX_TOKENS)
typedef __bf16 mf_bf16x8 __attribute__((ext_vector_type(8)));
typedef float mf_floatx8 __attribute__((ext_vector_type(8)));
__device__ __forceinline__ mf_floatx8 mf_vg_wmma(mf_bf16x8 a, mf_bf16x8 b, mf_floatx8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}
// FN49a: `__shfl_xor(v, o, 32)` (ds_bpermute: an LDS round trip a hop) as the SAME partner lane read
// by DPP16 row_xmask (lane ^ 1 / 2 / 4 / 8 inside a 16-lane row) or v_permlanex16_b32 with the
// identity lane select (lane ^ 16, the other row): pure data movement, so every value -- and every
// byte the route writes -- is the shipped one. Needs the whole wave active (the route's token loop
// is wave-uniform) and width 32. RDNA4 ISA ch. 7.9 (DPP16), ch. 16 (V_PERMLANEX16_B32).
template <class V>
__device__ __forceinline__ V mf_dpp_xor(V v, int o, int width) {
  static_assert(sizeof(V) == 4, "mf_dpp_xor moves 32-bit values");
  (void)width;
  const int x = __builtin_bit_cast(int, v);
  int y;
  switch (o) {
    case 16: y = __builtin_amdgcn_permlanex16(x, x, 0x76543210u, 0xfedcba98u, false, false); break;
    case 8: y = __builtin_amdgcn_update_dpp(x, x, 0x168, 0xf, 0xf, false); break;
    case 4: y = __builtin_amdgcn_update_dpp(x, x, 0x164, 0xf, 0xf, false); break;
    case 2: y = __builtin_amdgcn_update_dpp(x, x, 0x162, 0xf, 0xf, false); break;
    default: y = __builtin_amdgcn_update_dpp(x, x, 0x161, 0xf, 0xf, false); break;
  }
  return __builtin_bit_cast(V, y);
}
template <int NT>
static __device__ __forceinline__ int mf_exscan(int v, int *s_wave, int *total) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  int x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up(x, o, 32);
    if (lane >= o) x += y;
  }
  if (lane == 31) s_wave[wave] = x;
  __syncthreads();
  int base = 0, all = 0;
#pragma unroll
  for (int w = 0; w < NT / 32; ++w) {
    const int s = s_wave[w];
    if (w < wave) base += s;
    all += s;
  }
  __syncthreads();
  *total = all;
  return base + x - v;
}

// FN5's split-K band body (flashnext_hc.hip), derived: waves / unroll as parameters, the
// partial slice stored at every split count.
template <int MROWS, int FB_NW, int FB_UNR>
__device__ __forceinline__ void mf_bgemm_sk_part(const __bf16 *__restrict__ a,
                                                 const __bf16 *__restrict__ w,
                                                 float *__restrict__ part, int m, int n, int k) {
  constexpr int MF = MROWS / 16;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int n0 = ((int)blockIdx.x * FB_NW + wave) * 16;
  const int m0 = (int)blockIdx.y * MROWS;
  const int sp = (int)blockIdx.z, nsp = (int)gridDim.z;
  const int kw = k / nsp, kb = sp * kw;
  const int row = lane & 15, kh = (lane >> 4) * 8;
  const bool live = n0 < n;
  const __bf16 *ap[MF];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf) ap[mf] = a + (long)min(m0 + mf * 16 + row, m - 1) * k + kb + kh;
  const __bf16 *wp = w + (long)min(n0 + row, n - 1) * k + kb + kh;
  mf_floatx8 acc[MF];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[mf][e] = 0.f;
  const int ksteps = live ? kw / 16 : 0;
  int ks = 0;
  for (; ks + FB_UNR <= ksteps; ks += FB_UNR) {
    mf_bf16x8 af[MF][FB_UNR], wf[FB_UNR];
#pragma unroll
    for (int u = 0; u < FB_UNR; ++u) {
#pragma unroll
      for (int mf = 0; mf < MF; ++mf) af[mf][u] = *(const mf_bf16x8 *)(ap[mf] + (ks + u) * 16);
      wf[u] = *(const mf_bf16x8 *)(wp + (ks + u) * 16);
    }
#pragma unroll
    for (int u = 0; u < FB_UNR; ++u)
#pragma unroll
      for (int mf = 0; mf < MF; ++mf) acc[mf] = mf_vg_wmma(af[mf][u], wf[u], acc[mf]);
  }
  for (; ks < ksteps; ++ks) {
    const mf_bf16x8 wv = *(const mf_bf16x8 *)(wp + ks * 16);
#pragma unroll
    for (int mf = 0; mf < MF; ++mf)
      acc[mf] = mf_vg_wmma(*(const mf_bf16x8 *)(ap[mf] + ks * 16), wv, acc[mf]);
  }
  const int col = n0 + (lane & 15);
  if (live && col < n) {
#pragma unroll
    for (int mf = 0; mf < MF; ++mf)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int r = m0 + mf * 16 + (lane >> 4) * 8 + e;
        if (r < m) part[((long)sp * m + r) * n + col] = acc[mf][e];
      }
  }
}

// The same with the slice fully unrolled.
template <int MROWS, int FB_NW, int KST>
__device__ __forceinline__ void mf_bgemm_sk_full(const __bf16 *__restrict__ a,
                                                 const __bf16 *__restrict__ w,
                                                 float *__restrict__ part, int m, int n, int k) {
  constexpr int MF = MROWS / 16;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int n0 = ((int)blockIdx.x * FB_NW + wave) * 16;
  const int m0 = (int)blockIdx.y * MROWS;
  const int sp = (int)blockIdx.z, nsp = (int)gridDim.z;
  const int kw = k / nsp, kb = sp * kw;
  const int row = lane & 15, kh = (lane >> 4) * 8;
  const bool live = n0 < n;
  const __bf16 *ap[MF];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf) ap[mf] = a + (long)min(m0 + mf * 16 + row, m - 1) * k + kb + kh;
  const __bf16 *wp = w + (long)min(n0 + row, n - 1) * k + kb + kh;
  mf_floatx8 acc[MF];
#pragma unroll
  for (int mf = 0; mf < MF; ++mf)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[mf][e] = 0.f;
  const int ksteps = live ? kw / 16 : 0;
  if (live && ksteps != KST) __builtin_trap();
  if (ksteps > 0) {
#pragma unroll
    for (int ks = 0; ks < KST; ++ks) {
      mf_bf16x8 af[MF], wf;
#pragma unroll
      for (int mf = 0; mf < MF; ++mf) af[mf] = *(const mf_bf16x8 *)(ap[mf] + ks * 16);
      wf = *(const mf_bf16x8 *)(wp + ks * 16);
#pragma unroll
      for (int mf = 0; mf < MF; ++mf) acc[mf] = mf_vg_wmma(af[mf], wf, acc[mf]);
    }
  }
  const int col = n0 + (lane & 15);
  if (live && col < n) {
#pragma unroll
    for (int mf = 0; mf < MF; ++mf)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int r = m0 + mf * 16 + (lane >> 4) * 8 + e;
        if (r < m) part[((long)sp * m + r) * n + col] = acc[mf][e];
      }
  }
}

// The route of a whole step (T <= MF_TMAX) in ONE workgroup of NT threads, every table in LDS.
// FROM_PART: the logits are the band's partial slices (summed + rounded here, written to lg_out);
// else the bf16 logits `lg_in` (FN4's input).
template <int NT, bool FROM_PART, bool DPP = false>
__device__ __forceinline__ void mf_route_lds(
    const float *__restrict__ part, const int nsp, const __bf16 *__restrict__ logits,
    __bf16 *__restrict__ lg_out, int T, int e_lo, int e_loc, int tile_blocks, int with_shared,
    int *__restrict__ topk_ids, float *__restrict__ topk_w, int *__restrict__ route_pos,
    int *__restrict__ row_tok, int *__restrict__ row_route, int *__restrict__ tile_expert,
    int *__restrict__ tile_row0, int *__restrict__ tile_rows, int *__restrict__ expert_cnt,
    int *__restrict__ meta) {
  constexpr int MF_EPT = MOE_EXPERTS / NT;   // experts a thread scans
  __shared__ int s_ids[MF_TMAX * MOE_TOPK];
  __shared__ unsigned int s_mask[MOE_EXPERTS][2];   // bit t: token t routes to the expert
  __shared__ int s_cnt[MOE_EXPERTS];
  __shared__ int s_off[MOE_EXPERTS + 1], s_toff[MOE_EXPERTS + 1];
  __shared__ int s_wave[NT / 32];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  if (T > MF_TMAX) __builtin_trap();   // fail loud: the planner bands this kernel to T <= 64

  for (int e = tid; e < MOE_EXPERTS; e += NT) { s_mask[e][0] = 0u; s_mask[e][1] = 0u; }

  // ---- phase 1: FN4's softmax + top-10 + renormalise, one wave per token, v2's interleaved map.
  for (int tl = wave; tl < T; tl += NT / 32) {
    const int t = tl;
    float p[16];
    if constexpr (FROM_PART) {
      // The band's reduce, per logit: `v = part[0]; v = v + part[p]` for p = 1 .. nsp - 1 in slice
      // order, one bf16 rounding (the band's expression, anchor-checked below) -- with the lane's
      // 16 x nsp partials loaded FIRST, 16-byte, unconditional at a clamped slice (FN31 hold 1: a
      // load inside the runtime slice loop serialized 16 x nsp L2 round trips, ~29 us a token).
      union { uint4 u; __bf16 h[8]; } v0, v1;
      const int m = T, n = MOE_EXPERTS, r = t;
      float sl[4][16];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int qc = q < nsp ? q : nsp - 1;
        const float4 *s4 = (const float4 *)(part + ((long)qc * m + r) * n + lane * 16);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float4 x = s4[i];
          sl[q][4 * i] = x.x; sl[q][4 * i + 1] = x.y; sl[q][4 * i + 2] = x.z; sl[q][4 * i + 3] = x.w;
        }
      }
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        float v = sl[0][j];
#pragma unroll
        for (int p = 1; p < 4; ++p)
          if (p < nsp) v = v + sl[p][j];
        const __bf16 b = (__bf16)v;
        if (j < 8) v0.h[j] = b; else v1.h[j - 8] = b;
      }
      uint4 *dst = (uint4 *)(lg_out + (size_t)t * MOE_EXPERTS + lane * 16);
      dst[0] = v0.u;
      dst[1] = v1.u;
#pragma unroll
      for (int j = 0; j < 8; ++j) { p[j] = (float)v0.h[j]; p[8 + j] = (float)v1.h[j]; }
    } else {
      const uint4 *src = (const uint4 *)(logits + (size_t)t * MOE_EXPERTS + lane * 16);
      union { uint4 u; __bf16 h[8]; } v0, v1;
      v0.u = src[0];
      v1.u = src[1];
  #pragma unroll
      for (int j = 0; j < 8; ++j) { p[j] = (float)v0.h[j]; p[8 + j] = (float)v1.h[j]; }
    }
    if constexpr (!DPP) {
    float mx = p[0];
#pragma unroll
    for (int j = 1; j < 16; ++j) mx = fmaxf(mx, p[j]);
#pragma unroll
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) { p[j] = saber_exp_nonpos(p[j] - mx); s = s + p[j]; }
#pragma unroll
    for (int o = 16; o; o >>= 1) s = s + __shfl_xor(s, o, 32);
    const float inv = 1.f / s;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      float q = p[j] * inv;
      if (!__builtin_isfinite(q)) q = 0.f;   // upstream's NaN/Inf clamp
      p[j] = q;
    }
    float sel = 0.f, wk[MOE_TOPK];
    int ik[MOE_TOPK];
#pragma unroll
    for (int k = 0; k < MOE_TOPK; ++k) {
      float bv = p[0];
      int bj = 0;
#pragma unroll
      for (int j = 1; j < 16; ++j)
        if (p[j] > bv) { bv = p[j]; bj = j; }    // strict: the lower index keeps a tie
      int bi = lane * 16 + bj;
#pragma unroll
      for (int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor(bv, o, 32);
        const int oi = __shfl_xor(bi, o, 32);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
      if ((bi >> 4) == lane) {
#pragma unroll
        for (int j = 0; j < 16; ++j)
          if (j == (bi & 15)) p[j] = -1.f;       // probabilities are >= 0
      }
      wk[k] = bv;
      ik[k] = bi;
      sel = sel + bv;                            // k order, as upstream's selected_sum
    }
    const float scale = 1.f / (sel > 0.f ? sel : 1.f);
    if (lane < MOE_TOPK) {
      float wv = 0.f;
      int iv = 0;
#pragma unroll
      for (int k = 0; k < MOE_TOPK; ++k)
        if (k == lane) { wv = wk[k]; iv = ik[k]; }
      topk_ids[(size_t)t * MOE_TOPK + lane] = iv;
      topk_w[(size_t)t * MOE_TOPK + lane] = wv * scale;
      s_ids[tl * MOE_TOPK + lane] = iv;
    }
    } else {
    float mx = p[0];
#pragma unroll
    for (int j = 1; j < 16; ++j) mx = fmaxf(mx, p[j]);
#pragma unroll
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, mf_dpp_xor(mx, o, 32));
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) { p[j] = saber_exp_nonpos(p[j] - mx); s = s + p[j]; }
#pragma unroll
    for (int o = 16; o; o >>= 1) s = s + mf_dpp_xor(s, o, 32);
    const float inv = 1.f / s;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      float q = p[j] * inv;
      if (!__builtin_isfinite(q)) q = 0.f;   // upstream's NaN/Inf clamp
      p[j] = q;
    }
    float sel = 0.f, wk[MOE_TOPK];
    int ik[MOE_TOPK];
#pragma unroll
    for (int k = 0; k < MOE_TOPK; ++k) {
      float bv = p[0];
      int bj = 0;
#pragma unroll
      for (int j = 1; j < 16; ++j)
        if (p[j] > bv) { bv = p[j]; bj = j; }    // strict: the lower index keeps a tie
      int bi = lane * 16 + bj;
#pragma unroll
      for (int o = 16; o; o >>= 1) {
        const float ov = mf_dpp_xor(bv, o, 32);
        const int oi = mf_dpp_xor(bi, o, 32);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
      if ((bi >> 4) == lane) {
#pragma unroll
        for (int j = 0; j < 16; ++j)
          if (j == (bi & 15)) p[j] = -1.f;       // probabilities are >= 0
      }
      wk[k] = bv;
      ik[k] = bi;
      sel = sel + bv;                            // k order, as upstream's selected_sum
    }
    const float scale = 1.f / (sel > 0.f ? sel : 1.f);
    if (lane < MOE_TOPK) {
      float wv = 0.f;
      int iv = 0;
#pragma unroll
      for (int k = 0; k < MOE_TOPK; ++k)
        if (k == lane) { wv = wk[k]; iv = ik[k]; }
      topk_ids[(size_t)t * MOE_TOPK + lane] = iv;
      topk_w[(size_t)t * MOE_TOPK + lane] = wv * scale;
      s_ids[tl * MOE_TOPK + lane] = iv;
    }
    }
  }
  __syncthreads();

  // ---- phase 2: the token mask per local expert (a token's ten experts are distinct).
  for (int i = tid; i < T * MOE_TOPK; i += NT) {
    const int le = s_ids[i] - e_lo;
    if (le < 0 || le >= e_loc) continue;
    const int t = i / MOE_TOPK;
    atomicOr(&s_mask[le][t >> 5], 1u << (t & 31));
  }
  __syncthreads();

  // ---- phase 3: per-expert totals (popcount), then FN4's scans, tiles, pads and placement.
  for (int e = tid; e < e_loc; e += NT) {
    const int c = __popc(s_mask[e][0]) + __popc(s_mask[e][1]);
    s_cnt[e] = c;
    expert_cnt[e] = c;
  }
  __syncthreads();
  // Padded rows and tiles per expert, scanned: MF_EPT experts per thread.
  const int G = tile_blocks, TR = 16 * tile_blocks;
  int r2 = 0, t2 = 0, touched = 0, routes = 0;
  for (int h = 0; h < MF_EPT; ++h) {
    const int e = MF_EPT * tid + h;
    if (e < e_loc) {
      const int c = s_cnt[e], nb = (c + 15) / 16;
      r2 += nb * 16;
      t2 += (nb + G - 1) / G;
      touched += c > 0;
      routes += c;
    }
  }
  int rows_total, tiles_total, touched_total, routes_total;
  const int rbase = mf_exscan<NT>(r2, s_wave, &rows_total);
  const int tbase = mf_exscan<NT>(t2, s_wave, &tiles_total);
  mf_exscan<NT>(touched, s_wave, &touched_total);
  mf_exscan<NT>(routes, s_wave, &routes_total);
  {
    int ro = rbase, to = tbase;
    for (int h = 0; h < MF_EPT; ++h) {
      const int e = MF_EPT * tid + h;
      if (e < e_loc) {
        const int c = s_cnt[e], nb = (c + 15) / 16;
        s_off[e] = ro;
        s_toff[e] = to;
        ro += nb * 16;
        to += (nb + G - 1) / G;
      }
    }
  }
  __syncthreads();
  const int sh_row0 = rows_total;
  const int sh_blocks = with_shared ? (T + 15) / 16 : 0;
  const int sh_tiles = (sh_blocks + G - 1) / G;
  if (tid == 0) {
    meta[0] = tiles_total + sh_tiles;
    meta[1] = rows_total + sh_blocks * 16;
    meta[2] = with_shared ? sh_row0 : -1;
    meta[3] = routes_total;
    meta[4] = touched_total;
    meta[5] = rows_total;
  }
  // Tiles and pad rows.
  for (int e = tid; e < e_loc; e += NT) {
    const int c = s_cnt[e], nb = (c + 15) / 16, nt = (nb + G - 1) / G;
    for (int j = 0; j < nt; ++j) {
      tile_expert[s_toff[e] + j] = e;
      tile_row0[s_toff[e] + j] = s_off[e] + j * TR;
      tile_rows[s_toff[e] + j] = min(TR, c - j * TR);
    }
    for (int r = s_off[e] + c; r < s_off[e] + nb * 16; ++r) { row_tok[r] = -1; row_route[r] = -1; }
  }
  for (int j = tid; j < sh_tiles; j += NT) {
    tile_expert[tiles_total + j] = e_loc;
    tile_row0[tiles_total + j] = sh_row0 + j * TR;
    tile_rows[tiles_total + j] = min(TR, T - j * TR);
  }
  for (int r = tid; r < sh_blocks * 16; r += NT) {
    row_tok[sh_row0 + r] = r < T ? r : -1;
    row_route[sh_row0 + r] = -1;
  }
  // Placement: route r of token t at s_off[e] + (routes of e from tokens < t), FN4's position.
  for (int r = tid; r < T * MOE_TOPK; r += NT) {
    const int le = s_ids[r] - e_lo;
    if (le < 0 || le >= e_loc) { route_pos[r] = -1; continue; }
    const int t = r / MOE_TOPK;
    const unsigned int lo = s_mask[le][0], hi = s_mask[le][1];
    const int below = t < 32 ? __popc(lo & ((1u << t) - 1u))
                             : __popc(lo) + __popc(hi & ((1u << (t - 32)) - 1u));
    const int p = s_off[le] + below;
    row_tok[p] = t;
    row_route[p] = r;
    route_pos[r] = p;
  }
}

// FN4's route, drop-in (SIG_MOE_ROUTE, grid cdiv(T, 32) x 256): workgroup 0 routes the whole step
// in LDS; `blk_cnt`, `rank` and `arrive` are not touched.
extern "C" __global__ __launch_bounds__(MOE_THREADS) void moe_route_topk10_sort_e512_lds(
    const __bf16 *__restrict__ logits, int T, int e_lo, int e_loc, int tile_blocks,
    int with_shared, int *__restrict__ topk_ids, float *__restrict__ topk_w,
    int *__restrict__ route_pos, int *__restrict__ row_tok, int *__restrict__ row_route,
    int *__restrict__ tile_expert, int *__restrict__ tile_row0, int *__restrict__ tile_rows,
    int *__restrict__ expert_cnt, int *__restrict__ meta, int *__restrict__ blk_cnt,
    int *__restrict__ rank, int *__restrict__ arrive) {
  if (blockIdx.x != 0) return;
  mf_route_lds<MOE_THREADS, false>(nullptr, 1, logits, nullptr, T, e_lo, e_loc, tile_blocks,
                                   with_shared, topk_ids, topk_w, route_pos, row_tok, row_route,
                                   tile_expert, tile_row0, tile_rows, expert_cnt, meta);
}

// FN49a: the same route with its xor butterflies on DPP / v_permlanex16 (`mf_xor<true>`).
extern "C" __global__ __launch_bounds__(MOE_THREADS) void moe_route_topk10_sort_e512_lds_dpp(
    const __bf16 *__restrict__ logits, int T, int e_lo, int e_loc, int tile_blocks,
    int with_shared, int *__restrict__ topk_ids, float *__restrict__ topk_w,
    int *__restrict__ route_pos, int *__restrict__ row_tok, int *__restrict__ row_route,
    int *__restrict__ tile_expert, int *__restrict__ tile_row0, int *__restrict__ tile_rows,
    int *__restrict__ expert_cnt, int *__restrict__ meta, int *__restrict__ blk_cnt,
    int *__restrict__ rank, int *__restrict__ arrive) {
  if (blockIdx.x != 0) return;
  mf_route_lds<MOE_THREADS, false, true>(nullptr, 1, logits, nullptr, T, e_lo, e_loc, tile_blocks,
                                         with_shared, topk_ids, topk_w, route_pos, row_tok,
                                         row_route, tile_expert, tile_row0, tile_rows, expert_cnt,
                                         meta);
}

// The router GEMM + the route as ONE launch. Grid [512 / (16 NW), cdiv(T, 16), split] x 32 NW,
// split = the rules' `moe.router` count at T (4 at T 1, 2 at T 2..16, 1 above); `part` =
// [split][T][512] f32; `sync` = one i32, zero at allocation and zero again when the launch ends.
template <int NW, bool FULL, int UNR, bool DPP = false>
__device__ __forceinline__ void mf_front(
    const __bf16 *__restrict__ bi, const __bf16 *__restrict__ wr, float *__restrict__ part,
    int *__restrict__ sync, __bf16 *__restrict__ logits, int k, int T, int e_lo, int e_loc,
    int tile_blocks, int with_shared, int *__restrict__ topk_ids, float *__restrict__ topk_w,
    int *__restrict__ route_pos, int *__restrict__ row_tok, int *__restrict__ row_route,
    int *__restrict__ tile_expert, int *__restrict__ tile_row0, int *__restrict__ tile_rows,
    int *__restrict__ expert_cnt, int *__restrict__ meta) {
  __shared__ int s_last;
  if constexpr (FULL) mf_bgemm_sk_full<16, NW, 2560 / 16 / UNR>(bi, wr, part, T, MOE_EXPERTS, k);
  else mf_bgemm_sk_part<16, NW, UNR>(bi, wr, part, T, MOE_EXPERTS, k);
  __syncthreads();
  const int nwg = (int)(gridDim.x * gridDim.y * gridDim.z);
  if (threadIdx.x == 0) {
    __threadfence();
    s_last = atomicAdd(sync, 1) == nwg - 1;
  }
  __syncthreads();
  if (!s_last) return;
  if (threadIdx.x == 0) *sync = 0;
  __threadfence();   // acquire: every workgroup's partial slice
  mf_route_lds<NW * 32, true, DPP>(part, (int)gridDim.z, nullptr, logits, T, e_lo, e_loc,
                                   tile_blocks, with_shared, topk_ids, topk_w, route_pos, row_tok,
                                   row_route, tile_expert, tile_row0, tile_rows, expert_cnt, meta);
}

extern "C" __global__ __launch_bounds__(128) void moe_front_e512_nw4_u8(
    const __bf16 *__restrict__ bi, const __bf16 *__restrict__ wr, float *__restrict__ part,
    int *__restrict__ sync, __bf16 *__restrict__ logits, int k, int T, int e_lo, int e_loc,
    int tile_blocks, int with_shared, int *__restrict__ topk_ids, float *__restrict__ topk_w,
    int *__restrict__ route_pos, int *__restrict__ row_tok, int *__restrict__ row_route,
    int *__restrict__ tile_expert, int *__restrict__ tile_row0, int *__restrict__ tile_rows,
    int *__restrict__ expert_cnt, int *__restrict__ meta) {
  mf_front<4, false, 8>(bi, wr, part, sync, logits, k, T, e_lo, e_loc, tile_blocks, with_shared,
                  topk_ids, topk_w, route_pos, row_tok, row_route, tile_expert, tile_row0, tile_rows,
                  expert_cnt, meta);
}

extern "C" __global__ __launch_bounds__(128) void moe_front_e512_nw4_u20(
    const __bf16 *__restrict__ bi, const __bf16 *__restrict__ wr, float *__restrict__ part,
    int *__restrict__ sync, __bf16 *__restrict__ logits, int k, int T, int e_lo, int e_loc,
    int tile_blocks, int with_shared, int *__restrict__ topk_ids, float *__restrict__ topk_w,
    int *__restrict__ route_pos, int *__restrict__ row_tok, int *__restrict__ row_route,
    int *__restrict__ tile_expert, int *__restrict__ tile_row0, int *__restrict__ tile_rows,
    int *__restrict__ expert_cnt, int *__restrict__ meta) {
  mf_front<4, false, 20>(bi, wr, part, sync, logits, k, T, e_lo, e_loc, tile_blocks, with_shared,
                  topk_ids, topk_w, route_pos, row_tok, row_route, tile_expert, tile_row0, tile_rows,
                  expert_cnt, meta);
}

extern "C" __global__ __launch_bounds__(128) void moe_front_e512_nw4_s4(
    const __bf16 *__restrict__ bi, const __bf16 *__restrict__ wr, float *__restrict__ part,
    int *__restrict__ sync, __bf16 *__restrict__ logits, int k, int T, int e_lo, int e_loc,
    int tile_blocks, int with_shared, int *__restrict__ topk_ids, float *__restrict__ topk_w,
    int *__restrict__ route_pos, int *__restrict__ row_tok, int *__restrict__ row_route,
    int *__restrict__ tile_expert, int *__restrict__ tile_row0, int *__restrict__ tile_rows,
    int *__restrict__ expert_cnt, int *__restrict__ meta) {
  mf_front<4, true, 4>(bi, wr, part, sync, logits, k, T, e_lo, e_loc, tile_blocks, with_shared,
                  topk_ids, topk_w, route_pos, row_tok, row_route, tile_expert, tile_row0, tile_rows,
                  expert_cnt, meta);
}

extern "C" __global__ __launch_bounds__(256) void moe_front_e512_nw8_s4(
    const __bf16 *__restrict__ bi, const __bf16 *__restrict__ wr, float *__restrict__ part,
    int *__restrict__ sync, __bf16 *__restrict__ logits, int k, int T, int e_lo, int e_loc,
    int tile_blocks, int with_shared, int *__restrict__ topk_ids, float *__restrict__ topk_w,
    int *__restrict__ route_pos, int *__restrict__ row_tok, int *__restrict__ row_route,
    int *__restrict__ tile_expert, int *__restrict__ tile_row0, int *__restrict__ tile_rows,
    int *__restrict__ expert_cnt, int *__restrict__ meta) {
  mf_front<8, true, 4>(bi, wr, part, sync, logits, k, T, e_lo, e_loc, tile_blocks, with_shared,
                  topk_ids, topk_w, route_pos, row_tok, row_route, tile_expert, tile_row0, tile_rows,
                  expert_cnt, meta);
}

extern "C" __global__ __launch_bounds__(128) void moe_front_e512_nw4_s2(
    const __bf16 *__restrict__ bi, const __bf16 *__restrict__ wr, float *__restrict__ part,
    int *__restrict__ sync, __bf16 *__restrict__ logits, int k, int T, int e_lo, int e_loc,
    int tile_blocks, int with_shared, int *__restrict__ topk_ids, float *__restrict__ topk_w,
    int *__restrict__ route_pos, int *__restrict__ row_tok, int *__restrict__ row_route,
    int *__restrict__ tile_expert, int *__restrict__ tile_row0, int *__restrict__ tile_rows,
    int *__restrict__ expert_cnt, int *__restrict__ meta) {
  mf_front<4, true, 2>(bi, wr, part, sync, logits, k, T, e_lo, e_loc, tile_blocks, with_shared,
                  topk_ids, topk_w, route_pos, row_tok, row_route, tile_expert, tile_row0, tile_rows,
                  expert_cnt, meta);
}

extern "C" __global__ __launch_bounds__(256) void moe_front_e512_nw8_s2(
    const __bf16 *__restrict__ bi, const __bf16 *__restrict__ wr, float *__restrict__ part,
    int *__restrict__ sync, __bf16 *__restrict__ logits, int k, int T, int e_lo, int e_loc,
    int tile_blocks, int with_shared, int *__restrict__ topk_ids, float *__restrict__ topk_w,
    int *__restrict__ route_pos, int *__restrict__ row_tok, int *__restrict__ row_route,
    int *__restrict__ tile_expert, int *__restrict__ tile_row0, int *__restrict__ tile_rows,
    int *__restrict__ expert_cnt, int *__restrict__ meta) {
  mf_front<8, true, 2>(bi, wr, part, sync, logits, k, T, e_lo, e_loc, tile_blocks, with_shared,
                  topk_ids, topk_w, route_pos, row_tok, row_route, tile_expert, tile_row0, tile_rows,
                  expert_cnt, meta);
}

extern "C" __global__ __launch_bounds__(128) void moe_front_e512_nw4_s1(
    const __bf16 *__restrict__ bi, const __bf16 *__restrict__ wr, float *__restrict__ part,
    int *__restrict__ sync, __bf16 *__restrict__ logits, int k, int T, int e_lo, int e_loc,
    int tile_blocks, int with_shared, int *__restrict__ topk_ids, float *__restrict__ topk_w,
    int *__restrict__ route_pos, int *__restrict__ row_tok, int *__restrict__ row_route,
    int *__restrict__ tile_expert, int *__restrict__ tile_row0, int *__restrict__ tile_rows,
    int *__restrict__ expert_cnt, int *__restrict__ meta) {
  mf_front<4, true, 1>(bi, wr, part, sync, logits, k, T, e_lo, e_loc, tile_blocks, with_shared,
                  topk_ids, topk_w, route_pos, row_tok, row_route, tile_expert, tile_row0, tile_rows,
                  expert_cnt, meta);
}

extern "C" __global__ __launch_bounds__(256) void moe_front_e512_nw8_s1(
    const __bf16 *__restrict__ bi, const __bf16 *__restrict__ wr, float *__restrict__ part,
    int *__restrict__ sync, __bf16 *__restrict__ logits, int k, int T, int e_lo, int e_loc,
    int tile_blocks, int with_shared, int *__restrict__ topk_ids, float *__restrict__ topk_w,
    int *__restrict__ route_pos, int *__restrict__ row_tok, int *__restrict__ row_route,
    int *__restrict__ tile_expert, int *__restrict__ tile_row0, int *__restrict__ tile_rows,
    int *__restrict__ expert_cnt, int *__restrict__ meta) {
  mf_front<8, true, 1>(bi, wr, part, sync, logits, k, T, e_lo, e_loc, tile_blocks, with_shared,
                  topk_ids, topk_w, route_pos, row_tok, row_route, tile_expert, tile_row0, tile_rows,
                  expert_cnt, meta);
}

extern "C" __global__ __launch_bounds__(256) void moe_front_e512_nw8_s4_dpp(
    const __bf16 *__restrict__ bi, const __bf16 *__restrict__ wr, float *__restrict__ part,
    int *__restrict__ sync, __bf16 *__restrict__ logits, int k, int T, int e_lo, int e_loc,
    int tile_blocks, int with_shared, int *__restrict__ topk_ids, float *__restrict__ topk_w,
    int *__restrict__ route_pos, int *__restrict__ row_tok, int *__restrict__ row_route,
    int *__restrict__ tile_expert, int *__restrict__ tile_row0, int *__restrict__ tile_rows,
    int *__restrict__ expert_cnt, int *__restrict__ meta) {
  mf_front<8, true, 4, true>(bi, wr, part, sync, logits, k, T, e_lo, e_loc, tile_blocks, with_shared,
                  topk_ids, topk_w, route_pos, row_tok, row_route, tile_expert, tile_row0, tile_rows,
                  expert_cnt, meta);
}

extern "C" __global__ __launch_bounds__(256) void moe_front_e512_nw8_s2_dpp(
    const __bf16 *__restrict__ bi, const __bf16 *__restrict__ wr, float *__restrict__ part,
    int *__restrict__ sync, __bf16 *__restrict__ logits, int k, int T, int e_lo, int e_loc,
    int tile_blocks, int with_shared, int *__restrict__ topk_ids, float *__restrict__ topk_w,
    int *__restrict__ route_pos, int *__restrict__ row_tok, int *__restrict__ row_route,
    int *__restrict__ tile_expert, int *__restrict__ tile_row0, int *__restrict__ tile_rows,
    int *__restrict__ expert_cnt, int *__restrict__ meta) {
  mf_front<8, true, 2, true>(bi, wr, part, sync, logits, k, T, e_lo, e_loc, tile_blocks, with_shared,
                  topk_ids, topk_w, route_pos, row_tok, row_route, tile_expert, tile_row0, tile_rows,
                  expert_cnt, meta);
}

extern "C" __global__ __launch_bounds__(256) void moe_front_e512_nw8_s1_dpp(
    const __bf16 *__restrict__ bi, const __bf16 *__restrict__ wr, float *__restrict__ part,
    int *__restrict__ sync, __bf16 *__restrict__ logits, int k, int T, int e_lo, int e_loc,
    int tile_blocks, int with_shared, int *__restrict__ topk_ids, float *__restrict__ topk_w,
    int *__restrict__ route_pos, int *__restrict__ row_tok, int *__restrict__ row_route,
    int *__restrict__ tile_expert, int *__restrict__ tile_row0, int *__restrict__ tile_rows,
    int *__restrict__ expert_cnt, int *__restrict__ meta) {
  mf_front<8, true, 1, true>(bi, wr, part, sync, logits, k, T, e_lo, e_loc, tile_blocks, with_shared,
                  topk_ids, topk_w, route_pos, row_tok, row_route, tile_expert, tile_row0, tile_rows,
                  expert_cnt, meta);
}
// ---- end FN31 front ----

// ---- FN111: the LDS route at 16 waves (T 17..64) ----
// The three-launch train's route above T 16 is FN4's `moe_route_topk10_sort_e512` (one workgroup,
// 27.4 us a b8 w3 verify layer = 1.31 ms a step, LEDGER FN111 census). This is FN49a's LDS + DPP
// route (`mf_route_lds<.., false, true>`, byte-identical to FN4's at every T <= 64: FN31 / FN49a
// oracles incl. synthetic T 33 / 64) at NT 512: phase 1's per-token wave work is unchanged (one wave a
// token, its own lanes), 16 waves take 24 tokens in 2 rounds instead of 3; phases 2-3 are integer
// (atomicOr masks, popcounts, exact scans) -> the tables are the same bytes. Drop-in signature; grid
// cdiv(T, 32) x 512, workgroups past the first exit.
extern "C" __global__ __launch_bounds__(512) void fn111_moe_route_lds_dpp_nt512(
    const __bf16 *__restrict__ logits, int T, int e_lo, int e_loc, int tile_blocks,
    int with_shared, int *__restrict__ topk_ids, float *__restrict__ topk_w,
    int *__restrict__ route_pos, int *__restrict__ row_tok, int *__restrict__ row_route,
    int *__restrict__ tile_expert, int *__restrict__ tile_row0, int *__restrict__ tile_rows,
    int *__restrict__ expert_cnt, int *__restrict__ meta, int *__restrict__ blk_cnt,
    int *__restrict__ rank, int *__restrict__ arrive) {
  if (blockIdx.x != 0) return;
  mf_route_lds<512, false, true>(nullptr, 1, logits, nullptr, T, e_lo, e_loc, tile_blocks,
                                 with_shared, topk_ids, topk_w, route_pos, row_tok, row_route,
                                 tile_expert, tile_row0, tile_rows, expert_cnt, meta);
}
