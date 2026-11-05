// FN163: byte-equal twins of the MoE route (`fn111_moe_route_lds_dpp_nt512` / `_lds_dpp`, the b8
// band T 9..64) and the fused front half (`moe_front_e512_nw8_s{4,2}_dpp`, the b1 band T 1..8).
// ITS OWN OBJECT so `moe_route` keeps its bytes; built `-ffp-contract=off` (the `moe_route` prefix
// row of saber-kbuild's flag table) like its parents. Every fp32 line below is the shipped text
// (moe_route.hip's FN4 / FN31 / FN49a bodies: the softmax, its DPP butterflies, the reduce of the
// router's partial slices, the renormalisation); what changes is WHO does the integer work and the
// order of the exact selection:
//
//  * the b8 route (`fn163_route_nt1024`): the shipped phase-1 text (softmax + 10 selection rounds)
//    at 32 waves -- one token a wave, so T <= 32 runs ONE round (the shipped 16 waves take two at
//    T 17..32) -- and phase 3's four exclusive scans as ONE packed pass (rows | tiles << 16,
//    touched | routes << 16: integer sums of fields whose totals stay < 2^16 at T <= 64, e_loc <=
//    512 -> exact). Threads past expert 512 own no expert.
//  * the b1 front (`fn163_front_e512_nw8_s4`): the shipped router GEMM text + a register-rank
//    top-10 (FAST): the threshold L = the 10th largest of the 32 lane maxima (rank counting over
//    v_readlane: exact compares), every value >= L compacted one a lane (ballot + mbcnt through
//    LDS), each candidate's rank under the shipped total order (value desc, LOWER index first)
//    counted over the 32 lanes -> the candidate of rank k is the shipped k-th pick. Exact: the top
//    10 all hold values >= L (10 distinct elements >= L exist) and every element ranked above a
//    candidate is a candidate; the weights are the same p values, `sel` is summed in k order from
//    0.f as shipped. More than 32 candidates (degenerate ties: all-zero / NaN rows) runs the
//    shipped rounds verbatim (wave-uniform branch). It wins on a latency-bound wave (one token a
//    wave, the front's T <= 8) and loses when the SIMDs are full (FN163 p2, the b8 route).
//  * FN163 p1 measured a register-block weight prefetch + LDS-staged rows for the front's GEMM
//    0.4-0.6 us SLOWER: not built.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

#define MOE_EXPERTS 512
#define MOE_TOPK 10
#define MF_TMAX 64
#define R163_CAP 32       // candidates a wave's list holds = one a lane (else the shipped rounds)

static __device__ __forceinline__ float f32_bits(unsigned int u) { return __uint_as_float(u); }

// moe_route.hip's saber_exp_nonpos, verbatim.
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

typedef __bf16 r163_bf16x8 __attribute__((ext_vector_type(8)));
typedef float r163_floatx8 __attribute__((ext_vector_type(8)));
__device__ __forceinline__ r163_floatx8 r163_wmma(r163_bf16x8 a, r163_bf16x8 b, r163_floatx8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

// moe_route.hip's mf_dpp_xor (FN49a), verbatim.
template <class V>
__device__ __forceinline__ V r163_dpp_xor(V v, int o) {
  static_assert(sizeof(V) == 4, "r163_dpp_xor moves 32-bit values");
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

static __device__ __forceinline__ void r163_wave_sync() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

// Two ints scanned at once (exclusive), NT threads; totals to *t0 / *t1. Integer: order-free.
template <int NT>
static __device__ __forceinline__ int2 r163_exscan2(int a, int b, int2 *s_wave, int *t0, int *t1) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  int x = a, y = b;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int u = __shfl_up(x, o, 32);
    const int v = __shfl_up(y, o, 32);
    if (lane >= o) { x += u; y += v; }
  }
  if (lane == 31) s_wave[wave] = make_int2(x, y);
  __syncthreads();
  int b0 = 0, b1 = 0, a0 = 0, a1 = 0;
#pragma unroll
  for (int w = 0; w < NT / 32; ++w) {
    const int2 s = s_wave[w];
    if (w < wave) { b0 += s.x; b1 += s.y; }
    a0 += s.x;
    a1 += s.y;
  }
  __syncthreads();
  *t0 = a0;
  *t1 = a1;
  return make_int2(b0 + x - a, b1 + y - b);
}

struct R163Cand {
  float v;
  int x;
};

// Phase 1 for ONE token on one wave: p[16] = the lane's probabilities (lane l owns experts
// [16l, 16l + 16)), the shipped softmax (DPP butterflies) then the exact top-10.
// FAST = false: the shipped 10 rounds verbatim (the attribution / control body).
template <bool FAST>
static __device__ __forceinline__ void r163_token(float (&p)[16], int t, int *__restrict__ topk_ids,
                                                  float *__restrict__ topk_w, int *s_ids_row,
                                                  R163Cand *cl) {
  const int lane = threadIdx.x & 31;
  {
    float mx = p[0];
#pragma unroll
    for (int j = 1; j < 16; ++j) mx = fmaxf(mx, p[j]);
#pragma unroll
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, r163_dpp_xor(mx, o));
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) { p[j] = saber_exp_nonpos(p[j] - mx); s = s + p[j]; }
#pragma unroll
    for (int o = 16; o; o >>= 1) s = s + r163_dpp_xor(s, o);
    const float inv = 1.f / s;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      float q = p[j] * inv;
      if (!__builtin_isfinite(q)) q = 0.f;   // upstream's NaN/Inf clamp
      p[j] = q;
    }
  }
  bool done = false;
  if constexpr (FAST) {
    // the threshold: the 10th largest lane maximum (rank counting, value desc / lane asc)
    float lm = p[0];
#pragma unroll
    for (int j = 1; j < 16; ++j) lm = fmaxf(lm, p[j]);   // values >= 0, finite: exact max
    int rk = 0;
#pragma unroll
    for (int l2 = 0; l2 < 32; ++l2) {
      const float o = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(lm), l2));
      rk += (o > lm) || (o == lm && l2 < lane);
    }
    const unsigned int m9 = __builtin_amdgcn_ballot_w32(rk == 9);
    const float L = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(lm), __builtin_ctz(m9)));
    // compaction of every value >= L (j-major, lane-minor; the order does not enter a byte)
    int base = 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const bool c = p[j] >= L;
      const unsigned int m = __builtin_amdgcn_ballot_w32(c);
      const int off = base + (int)__builtin_amdgcn_mbcnt_lo(m, 0u);
      if (c && off < R163_CAP) cl[off] = R163Cand{p[j], lane * 16 + j};
      base += __builtin_popcount(m);
    }
    if (base <= 32) {              // wave-uniform: one candidate a lane
      r163_wave_sync();
      const int C = base;
      const bool valid = lane < C;
      const R163Cand me = cl[valid ? lane : 0];
      // rank under the shipped total order (value desc, LOWER index first), in registers
      int r = 0;
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        const float ov = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(me.v), j));
        const int ox = __builtin_amdgcn_readlane(me.x, j);
        r += (j < C) && ((ov > me.v) || (ov == me.v && ox < me.x));
      }
      const bool top = valid && r < MOE_TOPK;
      float sel = 0.f;
#pragma unroll
      for (int k = 0; k < MOE_TOPK; ++k) {             // k order, from 0.f
        const unsigned int mk = __builtin_amdgcn_ballot_w32(top && r == k);
        sel = sel + __int_as_float(__builtin_amdgcn_readlane(__float_as_int(me.v), __builtin_ctz(mk)));
      }
      const float scale = 1.f / (sel > 0.f ? sel : 1.f);
      if (top) {
        topk_ids[(size_t)t * MOE_TOPK + r] = me.x;
        topk_w[(size_t)t * MOE_TOPK + r] = me.v * scale;
        s_ids_row[r] = me.x;
      }
      r163_wave_sync();            // cl is reused by this wave's next token
      done = true;
    }
  }
  if (!done) {
    // the shipped rounds (moe_route.hip mf_route_lds<.., DPP = true>), verbatim
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
        const float ov = r163_dpp_xor(bv, o);
        const int oi = r163_dpp_xor(bi, o);
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
      s_ids_row[lane] = iv;
    }
  }
}

// The route of a whole step (T <= 64) in ONE workgroup of NT threads (mf_route_lds's tables).
// STOP 1 = phase 1 only (FN163 p1/p2's timing attribution arms; none is built now).
template <int NT, bool FROM_PART, bool FAST, int STOP = 0>
__device__ __forceinline__ void r163_route(
    const float *__restrict__ part, const int nsp, const __bf16 *__restrict__ logits,
    __bf16 *__restrict__ lg_out, int T, int e_lo, int e_loc, int tile_blocks, int with_shared,
    int *__restrict__ topk_ids, float *__restrict__ topk_w, int *__restrict__ route_pos,
    int *__restrict__ row_tok, int *__restrict__ row_route, int *__restrict__ tile_expert,
    int *__restrict__ tile_row0, int *__restrict__ tile_rows, int *__restrict__ expert_cnt,
    int *__restrict__ meta) {
  constexpr int NWV = NT / 32;
  constexpr int EPT = MOE_EXPERTS / NT > 0 ? MOE_EXPERTS / NT : 1;   // experts a thread scans
  __shared__ int s_ids[MF_TMAX * MOE_TOPK];
  __shared__ unsigned int s_mask[MOE_EXPERTS][2];   // bit t: token t routes to the expert
  __shared__ int s_cnt[MOE_EXPERTS];
  __shared__ int s_off[MOE_EXPERTS + 1], s_toff[MOE_EXPERTS + 1];
  __shared__ int2 s_wave[NWV];
  __shared__ R163Cand s_cl[FAST ? NWV : 1][FAST ? R163_CAP : 1];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  if (T > MF_TMAX) __builtin_trap();

  for (int e = tid; e < MOE_EXPERTS; e += NT) { s_mask[e][0] = 0u; s_mask[e][1] = 0u; }

  for (int tl = wave; tl < T; tl += NWV) {
    const int t = tl;
    float p[16];
    if constexpr (FROM_PART) {
      // moe_route.hip mf_route_lds<.., FROM_PART = true>'s reduce, verbatim
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
        for (int p2 = 1; p2 < 4; ++p2)
          if (p2 < nsp) v = v + sl[p2][j];
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
    r163_token<FAST>(p, t, topk_ids, topk_w, &s_ids[tl * MOE_TOPK], s_cl[FAST ? wave : 0]);
  }
  if constexpr (STOP == 1) return;
  __syncthreads();

  // ---- phase 2: the token mask per local expert (a token's ten experts are distinct).
  for (int i = tid; i < T * MOE_TOPK; i += NT) {
    const int le = s_ids[i] - e_lo;
    if (le < 0 || le >= e_loc) continue;
    const int t = i / MOE_TOPK;
    atomicOr(&s_mask[le][t >> 5], 1u << (t & 31));
  }
  __syncthreads();

  // ---- phase 3: per-expert totals, ONE packed scan, tiles, pads and placement.
  for (int e = tid; e < e_loc; e += NT) {
    const int c = __popc(s_mask[e][0]) + __popc(s_mask[e][1]);
    s_cnt[e] = c;
    expert_cnt[e] = c;
  }
  __syncthreads();
  const int G = tile_blocks, TR = 16 * tile_blocks;
  int r2 = 0, t2 = 0, touched = 0, routes = 0;
#pragma unroll
  for (int h = 0; h < EPT; ++h) {
    const int e = EPT * tid + h;
    if (e < e_loc) {
      const int c = s_cnt[e], nb = (c + 15) / 16;
      r2 += nb * 16;
      t2 += (nb + G - 1) / G;
      touched += c > 0;
      routes += c;
    }
  }
  int pk0, pk1;
  const int2 bs = r163_exscan2<NT>(r2 | (t2 << 16), touched | (routes << 16), s_wave, &pk0, &pk1);
  const int rows_total = pk0 & 0xffff, tiles_total = pk0 >> 16;
  const int touched_total = pk1 & 0xffff, routes_total = pk1 >> 16;
  {
    int ro = bs.x & 0xffff, to = bs.x >> 16;
#pragma unroll
    for (int h = 0; h < EPT; ++h) {
      const int e = EPT * tid + h;
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

#define R163_ROUTE_SIG                                                                         \
  const __bf16 *__restrict__ logits, int T, int e_lo, int e_loc, int tile_blocks,              \
      int with_shared, int *__restrict__ topk_ids, float *__restrict__ topk_w,                 \
      int *__restrict__ route_pos, int *__restrict__ row_tok, int *__restrict__ row_route,     \
      int *__restrict__ tile_expert, int *__restrict__ tile_row0, int *__restrict__ tile_rows, \
      int *__restrict__ expert_cnt, int *__restrict__ meta, int *__restrict__ blk_cnt,         \
      int *__restrict__ rank, int *__restrict__ arrive
#define R163_ROUTE_ARGS                                                                        \
  nullptr, 1, logits, nullptr, T, e_lo, e_loc, tile_blocks, with_shared, topk_ids, topk_w,     \
      route_pos, row_tok, row_route, tile_expert, tile_row0, tile_rows, expert_cnt, meta

// The route twins: SIG_MOE_ROUTE, grid cdiv(T, 32) x NT; workgroups past the first exit;
// `blk_cnt` / `rank` / `arrive` untouched (the shipped LDS route's law).
#define R163_ROUTE(name, NT, FAST, STOP)                                             \
  extern "C" __global__ __launch_bounds__(NT) void name(R163_ROUTE_SIG) {           \
    if (blockIdx.x != 0) return;                                                     \
    r163_route<NT, false, FAST, STOP>(R163_ROUTE_ARGS);                               \
  }
// THE ROUTED b8 twin (`--fn-route163 b8`, T 9..64): the shipped selection rounds at 32 waves (one
// token a wave: T <= 32 in ONE round) + the packed scan.
R163_ROUTE(fn163_route_nt1024, 1024, false, 0)
// timing / oracle arms (never routed): the register-rank selection at 16 / 32 waves (FN163 p2: it
// loses to the shipped rounds when a WGP's SIMDs are full -- more instructions a token)
R163_ROUTE(fn163_attr_route_nt512_fast, 512, true, 0)
R163_ROUTE(fn163_attr_route_nt1024_fast, 1024, true, 0)

// ---- the front half ----
// moe_route.hip mf_bgemm_sk_full (the shipped split-K band body at a fully unrolled slice).
template <int MROWS, int FB_NW, int KST>
__device__ __forceinline__ void r163_bgemm(const __bf16 *__restrict__ a, const __bf16 *__restrict__ w,
                                           float *__restrict__ part, int m, int n, int k) {
  constexpr int MF = MROWS / 16;
  static_assert(MF == 1, "the front half is the 16-row band");
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int n0 = ((int)blockIdx.x * FB_NW + wave) * 16;
  const int m0 = (int)blockIdx.y * MROWS;
  const int sp = (int)blockIdx.z, nsp = (int)gridDim.z;
  const int kw = k / nsp, kb = sp * kw;
  const int row = lane & 15, kh = (lane >> 4) * 8;
  const bool live = n0 < n;
  const __bf16 *wp = w + (long)min(n0 + row, n - 1) * k + kb + kh;
  r163_floatx8 acc;
#pragma unroll
  for (int e = 0; e < 8; ++e) acc[e] = 0.f;
  const int ksteps = live ? kw / 16 : 0;
  if (live && ksteps != KST) __builtin_trap();
  {
    const __bf16 *ap = a + (long)min(m0 + row, m - 1) * k + kb + kh;
    if (ksteps > 0) {
#pragma unroll
      for (int ks = 0; ks < KST; ++ks) {
        r163_bf16x8 af, wf;
        af = *(const r163_bf16x8 *)(ap + ks * 16);
        wf = *(const r163_bf16x8 *)(wp + ks * 16);
        acc = r163_wmma(af, wf, acc);
      }
    }
  }
  const int col = n0 + (lane & 15);
  if (live && col < n) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int r = m0 + (lane >> 4) * 8 + e;
      if (r < m) part[((long)sp * m + r) * n + col] = acc[e];
    }
  }
}

// moe_route.hip mf_front's law (the last-arriving workgroup routes) on the twins' bodies.
template <int NW, int SPLIT, bool FAST>
__device__ __forceinline__ void r163_front(
    const __bf16 *__restrict__ bi, const __bf16 *__restrict__ wr, float *__restrict__ part,
    int *__restrict__ sync, __bf16 *__restrict__ logits, int k, int T, int e_lo, int e_loc,
    int tile_blocks, int with_shared, int *__restrict__ topk_ids, float *__restrict__ topk_w,
    int *__restrict__ route_pos, int *__restrict__ row_tok, int *__restrict__ row_route,
    int *__restrict__ tile_expert, int *__restrict__ tile_row0, int *__restrict__ tile_rows,
    int *__restrict__ expert_cnt, int *__restrict__ meta) {
  __shared__ int s_last;
  r163_bgemm<16, NW, 2560 / 16 / SPLIT>(bi, wr, part, T, MOE_EXPERTS, k);
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
  r163_route<NW * 32, true, FAST>(part, (int)gridDim.z, nullptr, logits, T, e_lo, e_loc,
                                  tile_blocks, with_shared, topk_ids, topk_w, route_pos, row_tok,
                                  row_route, tile_expert, tile_row0, tile_rows, expert_cnt, meta);
}

#define R163_FRONT(name, NW, SPLIT, FAST)                                                     \
  extern "C" __global__ __launch_bounds__(NW * 32) void name(                                    \
      const __bf16 *__restrict__ bi, const __bf16 *__restrict__ wr, float *__restrict__ part,    \
      int *__restrict__ sync, __bf16 *__restrict__ logits, int k, int T, int e_lo, int e_loc,    \
      int tile_blocks, int with_shared, int *__restrict__ topk_ids, float *__restrict__ topk_w,  \
      int *__restrict__ route_pos, int *__restrict__ row_tok, int *__restrict__ row_route,       \
      int *__restrict__ tile_expert, int *__restrict__ tile_row0, int *__restrict__ tile_rows,   \
      int *__restrict__ expert_cnt, int *__restrict__ meta) {                                    \
    r163_front<NW, SPLIT, FAST>(bi, wr, part, sync, logits, k, T, e_lo, e_loc, tile_blocks,  \
                                    with_shared, topk_ids, topk_w, route_pos, row_tok, row_route, \
                                    tile_expert, tile_row0, tile_rows, expert_cnt, meta);        \
  }
// the twins of moe_front_e512_nw8_s{4,2}_dpp (grid [512 / (16 NW), cdiv(T, 16), SPLIT] x 32 NW):
// the shipped GEMM text + the register-rank selection. s4 = the b1 band's (ROUTED under
// `--fn-route163 b1`); s2 is built and oracle-gated but not routed (FN163 p2: mixed, +0.66 us at T 4).
R163_FRONT(fn163_front_e512_nw8_s4, 8, 4, true)
R163_FRONT(fn163_front_e512_nw8_s2, 8, 2, true)
