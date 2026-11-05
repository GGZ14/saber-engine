// Flash-Next (FN128): the QSA selection RE-TILED for the decode / verify forms (a handful of rows a
// step) -- ADDED symbols in their own object; `flashnext_qsa` keeps its bytes. Each symbol computes
// the SAME output words as its shipped parent (flashnext_qsa.hip; the law is
// tools/capture/fn19/qsa_spec.py): the oracle is byte equality with the parent
// (tools/capture/fn128/fn128_twin.py).
//
//   fn128_qsa_topk512_k<K>  THE SELECTION LAW of `fn_qsa_topk512`, one 512-thread workgroup a row
//                           with the row's keys in REGISTERS (thread t holds the contiguous chunk
//                           [t * C, t * C + C), C <= K a multiple of 4, b128 loads): the 512th largest
//                           key T is built bit by bit from the top set bit down, 2 bits a round, by
//                           COUNTING the keys >= each candidate (every wave, one barrier a round) until
//                           the live band holds <= 512 keys (the greedy maximum v with
//                           count(>= v) >= 512 IS the 512th largest), then the band in LDS gives T
//                           by RANK in one pass (a thread a band key); then ONE
//                           block scan of (count > T, count == T) a thread writes every key > T and
//                           the first 512 - count(> T) keys == T in index order, ascending. The parent
//                           walks every visible group four times on 256 threads with LDS atomics and
//                           a serial 256-entry scan a level; the set is the same by definition (keys
//                           compared as u32 bit patterns in both: the parent's radix order).
//   fn128_qsa_score_s1      `fn_qsa_score_r32` for a tile of <= 8 rows: ONE thread a group holds
//                           the key once for every row of the tile (NS = 2 / 4 spilled: not built), the query tile
//                           only the tile's 8 rows (8 KiB, not 32), the parent's arithmetic op for op
//                           (the same expression, the same order); grid (ceil(gcap / 64), row tiles).
//   fn128_qsa_union_s       `fn_qsa_union` for a CTA of <= 8 rows (decode / verify): the union of the
//                           rows' ascending 512-lists BY RANK (binary searches in LDS) instead of a
//                           scan over every group to the CTA's last: an element owned by the lowest
//                           list holding it (or by the CTA's span [p_first / 4, p_last / 4]) lands at
//                           (# owned elements below it, every list) + its span offset; the same kslot /
//                           kmask / kbits / nkeys words. `who` is never touched (it stays zero).
//
// Built with -ffp-contract=off (saber-kbuild's `flashnext_qsa` row); no fp arithmetic here.
// Loads past a live range are CLAMPED, never predicated (the predicated-staging law).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

#define QSA_TOPK 512
#define QSA_IDX_D 128
#define QSA_IDX_H 4

__device__ __forceinline__ float qsa_bf2f(uint32_t w, int hi) {
  return __builtin_bit_cast(float, hi ? (w & 0xffff0000u) : (w << 16));
}

// ------------------------------------------------------------------------------------------ score
// grid (ceil(gcap / 64), ceil(m / rt)), block 64 * NS, rt <= 8 (trap above). Thread (g = tid & 63,
// rs = tid >> 6) holds group g's key and walks rows rs, rs + NS, ...
template <int NS>
__device__ __forceinline__ void fn128_score_body(const __bf16 *__restrict__ q,
                                                 const __bf16 *__restrict__ comp,
                                                 const int *__restrict__ bt, int max_blocks,
                                                 const long *__restrict__ pos,
                                                 float *__restrict__ scores, long sstride, int m,
                                                 int rows_per_seq, int rt) {
  __shared__ uint4 sq[8 * QSA_IDX_H * QSA_IDX_D / 8];  // 8 KiB: [row][head][128] bf16
  const int tid = threadIdx.x;
  if (rt > 8) __builtin_trap();
  const int r0 = blockIdx.y * rt;
  const int nr = min(rt, m - r0);
  if (nr <= 0) return;
  const int g0 = blockIdx.x * 64;
  long pmax = pos[r0];
  for (int i = 1; i < nr; ++i) pmax = max(pmax, pos[r0 + i]);
  const int gvis = (int)((pmax + 1) >> 2);
  if (g0 >= gvis) return;  // workgroup-uniform
  const int seq = r0 / rows_per_seq;
  for (int i = tid; i < nr * QSA_IDX_H * QSA_IDX_D / 8; i += 64 * NS)
    sq[i] = *(const uint4 *)(q + (long)r0 * QSA_IDX_H * QSA_IDX_D + (long)i * 8);
  const int gl = tid & 63, rs = tid >> 6;
  const int g = min(g0 + gl, gvis - 1);
  const long page = bt[(long)seq * max_blocks + (g >> 2)];
  const uint4 *kp = (const uint4 *)(comp + (page * 4 + (g & 3)) * QSA_IDX_D);
  uint4 kr[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) kr[i] = kp[i];
  __syncthreads();
#pragma unroll 1
  for (int rr = rs; rr < nr; rr += NS) {
    const int r = r0 + rr;
    const int vis = (int)((pos[r] + 1) >> 2);
    float s = 0.f;
#pragma unroll 1
    for (int h = 0; h < QSA_IDX_H; ++h) {
      const uint4 *qp = &sq[(rr * QSA_IDX_H + h) * (QSA_IDX_D / 8)];
      float acc = 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const uint4 qv = qp[i];
        const uint4 kv = kr[i];
        const uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w};
        const uint32_t kw[4] = {kv.x, kv.y, kv.z, kv.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          acc = acc + qsa_bf2f(qw[j], 0) * qsa_bf2f(kw[j], 0);
          acc = acc + qsa_bf2f(qw[j], 1) * qsa_bf2f(kw[j], 1);
        }
      }
      s = s + fmaxf(acc, 0.f);
    }
    s = __fdiv_rn(s, 11.3137083053588867f);  // float32(sqrt(128))
    if (g0 + gl < vis) scores[(long)r * sstride + g0 + gl] = s;
  }
}

#define FN128_SCORE(NS)                                                                        \
  extern "C" __global__ __launch_bounds__(64 * NS) void fn128_qsa_score_s##NS(                 \
      const __bf16 *__restrict__ q, const __bf16 *__restrict__ comp, const int *__restrict__ bt, \
      int max_blocks, const long *__restrict__ pos, float *__restrict__ scores, long sstride,  \
      int m, int rows_per_seq, int rt) {                                                       \
    fn128_score_body<NS>(q, comp, bt, max_blocks, pos, scores, sstride, m, rows_per_seq, rt);  \
  }
FN128_SCORE(1)

// ------------------------------------------------------------------------------------------ top-k
// A wave's integer sum on DPP16 row_xmask (lane ^ 8/4/2/1) + v_permlanex16 (lane ^ 16): every lane
// holds the total (FN49b's register form of the xor butterfly; integer adds: order-free).
__device__ __forceinline__ int fn128_wsum(int x) {
  x += __builtin_amdgcn_update_dpp(x, x, 0x168, 0xf, 0xf, false);
  x += __builtin_amdgcn_update_dpp(x, x, 0x164, 0xf, 0xf, false);
  x += __builtin_amdgcn_update_dpp(x, x, 0x162, 0xf, 0xf, false);
  x += __builtin_amdgcn_update_dpp(x, x, 0x161, 0xf, 0xf, false);
  x += __builtin_amdgcn_permlanex16(x, x, 0x76543210u, 0xfedcba98u, false, false);
  return x;
}
__device__ __forceinline__ unsigned fn128_wor(unsigned x) {
  x |= (unsigned)__builtin_amdgcn_update_dpp((int)x, (int)x, 0x168, 0xf, 0xf, false);
  x |= (unsigned)__builtin_amdgcn_update_dpp((int)x, (int)x, 0x164, 0xf, 0xf, false);
  x |= (unsigned)__builtin_amdgcn_update_dpp((int)x, (int)x, 0x162, 0xf, 0xf, false);
  x |= (unsigned)__builtin_amdgcn_update_dpp((int)x, (int)x, 0x161, 0xf, 0xf, false);
  x |= (unsigned)__builtin_amdgcn_permlanex16((int)x, (int)x, 0x76543210u, 0xfedcba98u, false,
                                              false);
  return x;
}
// a wave's exclusive prefix of n (lane order) and its total
__device__ __forceinline__ int fn128_wexcl(int n, int lane, int &tot) {
  int incl = n;
#pragma unroll
  for (int s = 1; s < 32; s <<= 1) {
    const int u = __shfl_up(incl, s, 32);
    if (lane >= s) incl += u;
  }
  tot = __shfl(incl, 31, 32);
  return incl - n;
}

// THE SEARCH. T (the 512th largest key) is built bit by bit from the top set bit down, RB bits a
// round: a candidate's count is count(key >= candidate); the greedy maximum v with
// count(>= v) >= 512 IS the 512th largest. The LIVE BAND: after a round every key >= T + 2^(b + 1)
// counts (cu of them) and a key below T never will; the band [T, T + 2^(b + 1)) holds ct - cu keys.
// Phase 1 (every wave, keys in registers, one barrier a round) runs while the band holds more than
// BAND keys; then the band is COMPACTED into LDS and T is its (512 - cu)-th largest key, by rank.
template <int K, int NT>
__device__ __forceinline__ void fn128_topk_body(const float *__restrict__ scores, long sstride,
                                                const long *__restrict__ pos, int *__restrict__ sel,
                                                int m) {
  constexpr int NW = NT / 32;
  constexpr int RB = 2;
  constexpr int NC = (1 << RB) - 1;
  constexpr int BAND = 512;
  __shared__ int cnt[2][NC][NW];
  __shared__ unsigned orw[NW];
  __shared__ int wsum[2][NW];
  __shared__ __attribute__((aligned(16))) uint32_t band[BAND + 8];
  __shared__ uint32_t tfin;
  __shared__ int obuf[QSA_TOPK];
  const int r = blockIdx.x, tid = threadIdx.x;
  const int lane = tid & 31, wave = tid >> 5;
  if (r >= m) return;
  const int vis = (int)((pos[r] + 1) >> 2);
  int *out = sel + (long)r * QSA_TOPK;
  if (vis <= QSA_TOPK) {  // upstream's shortcut: every visible group (the parent's words)
    for (int j = tid; j < QSA_TOPK; j += NT) out[j] = j < vis ? j : -1;
    return;
  }
  // the chunk: C keys a thread (a multiple of 4: b128 loads at 16 B alignment)
  const int C = (((vis + NT - 1) / NT) + 3) & ~3;
  if (C > K) __builtin_trap();  // the host sizes K from the row stride (>= every visible count)
  const int base = tid * C;
  const uint4 *row = (const uint4 *)(scores + (long)r * sstride);
  const long lastq = (sstride >> 2) - 1;  // the row's last b128: a clamp inside the row
  uint32_t x[K];
#pragma unroll
  for (int i = 0; i < K; i += 4) {
    const long qi = min((long)((base + i) >> 2), lastq);
    const uint4 v = row[qi];
    x[i] = v.x;
    x[i + 1] = v.y;
    x[i + 2] = v.z;
    x[i + 3] = v.w;
  }
  const int nv = max(0, min(C, vis - base));  // keys [base, base + nv) are live
  // a dead key reads 0: every candidate is >= 1, so a dead key never counts in a round (the
  // band and the ordered pass test i < nv themselves: T may be +0)
  unsigned o = 0;
#pragma unroll
  for (int i = 0; i < K; ++i) {
    x[i] = (i < nv) ? x[i] : 0u;
    o |= x[i];
  }
  o = fn128_wor(o);
  if (lane == 0) orw[wave] = o;
  __syncthreads();
  const unsigned ob = fn128_wor(lane < NW ? orw[lane] : 0u);
  uint32_t T = 0;
  int b = ob ? 31 - __clz(ob) : -1;  // the highest bit still to decide
  int ct = vis, cu = 0;  // count(>= T) / count(>= T + 2^(b + 1)) (T = 0: every visible key)
  int par = 0;
  // ---- phase 1: every wave, the keys in registers
  while (b >= 0 && ct - cu > BAND) {  // block-uniform
    const int lo = max(b - RB + 1, 0);
    const int nc = (1 << (b - lo + 1)) - 1;
    int c[NC];
#pragma unroll
    for (int jj = 0; jj < NC; ++jj) {
      const uint32_t cand = T | ((uint32_t)(jj + 1) << lo);
      int n = 0;
#pragma unroll
      for (int i = 0; i < K; ++i) n += x[i] >= cand ? 1 : 0;
      c[jj] = fn128_wsum(n);
    }
    if (lane == 0) {
#pragma unroll
      for (int jj = 0; jj < NC; ++jj) cnt[par][jj][wave] = c[jj];
    }
    __syncthreads();
    int jmax = 0, v[NC];
#pragma unroll
    for (int jj = 0; jj < NC; ++jj) {
      v[jj] = fn128_wsum(lane < NW ? cnt[par][jj][lane] : 0);
      jmax += (jj < nc && v[jj] >= QSA_TOPK) ? 1 : 0;  // counts fall as j grows
    }
    int nct = ct, ncu = cu;
#pragma unroll
    for (int jj = 0; jj < NC; ++jj) {
      if (jj == jmax - 1) nct = v[jj];
      if (jj == jmax && jj < nc) ncu = v[jj];
    }
    ct = nct;
    cu = ncu;
    T |= (uint32_t)jmax << lo;
    b = lo - 1;
    par ^= 1;
  }
  if (b >= 0) {
    // ---- the band [T, U) into LDS (ct - cu <= BAND keys), in thread order
    const uint64_t U = (uint64_t)T + (2ull << b);
    int n = 0;
#pragma unroll
    for (int i = 0; i < K; ++i) n += (i < nv && x[i] >= T && (uint64_t)x[i] < U) ? 1 : 0;
    int wt;
    const int ex = fn128_wexcl(n, lane, wt);
    if (lane == 0) wsum[par][wave] = wt;
    __syncthreads();
    const int wb = fn128_wsum(lane < wave ? wsum[par][lane] : 0);  // the waves before this one
    int at = wb + ex;
#pragma unroll
    for (int i = 0; i < K; ++i)
      if (i < nv && x[i] >= T && (uint64_t)x[i] < U) band[at++] = x[i];
    __syncthreads();
    // ---- phase 2: T = the (512 - cu)-th largest key of the band, by RANK in one pass: thread t
    // takes band key t and counts the band keys > it and >= it (LDS broadcast reads); the key
    // with #greater < k <= #greater-or-equal is T (every thread that finds it writes the same
    // bits). One barrier instead of a round a remaining bit (t16 / t17: ~1-3 us a round).
    {
      const int nl = ct - cu;
      const int kk = QSA_TOPK - cu;  // 1 <= kk <= nl (ct >= 512)
      // the band padded to a multiple of 8 with +0 (a pad is never > a key; it is >= a key only
      // when the key is +0, and those pads are taken back out)
      const int nl8 = (nl + 7) & ~7;
      if (tid < nl8 - nl) band[nl + tid] = 0u;
      __syncthreads();
      if (tid < nl) {
        const uint32_t key = band[tid];
        int gt = 0, ge = 0;
        const uint4 *b4 = (const uint4 *)band;
        for (int e = 0; e < nl8 / 4; e += 2) {
          const uint4 v0 = b4[e], v1 = b4[e + 1];
          gt += (v0.x > key) + (v0.y > key) + (v0.z > key) + (v0.w > key) + (v1.x > key) +
                (v1.y > key) + (v1.z > key) + (v1.w > key);
          ge += (v0.x >= key) + (v0.y >= key) + (v0.z >= key) + (v0.w >= key) + (v1.x >= key) +
                (v1.y >= key) + (v1.z >= key) + (v1.w >= key);
        }
        if (key == 0u) ge -= nl8 - nl;
        if (gt < kk && ge >= kk) tfin = key;
      }
    }
    __syncthreads();
    T = tfin;
  }
  // ---- the ordered pass: every key > T, and the first `need` keys == T in index order
  int ngt = 0, neq = 0;
#pragma unroll
  for (int i = 0; i < K; ++i) {
    ngt += (i < nv && x[i] > T) ? 1 : 0;
    neq += (i < nv && x[i] == T) ? 1 : 0;
  }
  int tg, te;
  const int xg = fn128_wexcl(ngt, lane, tg), xe = fn128_wexcl(neq, lane, te);
  if (lane == 0) {
    cnt[par][0][wave] = tg;
    cnt[par][1][wave] = te;
  }
  __syncthreads();
  const int gt_tot = fn128_wsum(lane < NW ? cnt[par][0][lane] : 0);
  const int gb = fn128_wsum(lane < wave ? cnt[par][0][lane] : 0);
  const int eb = fn128_wsum(lane < wave ? cnt[par][1][lane] : 0);
  const int gt_base = gb + xg, eq_base = eb + xe;
  const int need = QSA_TOPK - gt_tot;  // >= 1
  int at = gt_base + min(eq_base, need);
  int er = eq_base;
#pragma unroll
  for (int i = 0; i < K; ++i) {
    if (i < nv) {
      const bool eq = x[i] == T;
      const bool take = x[i] > T || (eq && er < need);
      er += eq ? 1 : 0;
      if (take) obuf[at++] = base + i;
    }
  }
  // the row written from LDS in one coalesced pass
  __syncthreads();
  for (int j = tid; j < QSA_TOPK; j += NT) out[j] = obuf[j];
}

#define FN128_TOPK(K)                                                                       \
  extern "C" __global__ __launch_bounds__(512) void fn128_qsa_topk512_k##K(                 \
      const float *__restrict__ scores, long sstride, const long *__restrict__ pos,         \
      int *__restrict__ sel, int m) {                                                       \
    fn128_topk_body<K, 512>(scores, sstride, pos, sel, m);                                  \
  }
FN128_TOPK(8)
FN128_TOPK(16)
FN128_TOPK(32)
// the 131,072 band (32,768 groups): 32 keys a thread at 1,024 threads (64 at 512 held its
// predicates in SGPRs and spilled)
extern "C" __global__ __launch_bounds__(1024) void fn128_qsa_topk512_k32_t1024(
    const float *__restrict__ scores, long sstride, const long *__restrict__ pos,
    int *__restrict__ sel, int m) {
  fn128_topk_body<32, 1024>(scores, sstride, pos, sel, m);
}

// ------------------------------------------------------------------------------------------ union
#define U128_ROWS 8

// lower_bound over the first n entries of an ascending LDS list
__device__ __forceinline__ int fn128_lb(const int *L, int n, int x) {
  int lo = 0, hi = n;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (L[mid] < x)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

// grid (ncta), block U128_NT. The parent's arguments; a CTA of more than 8 rows is refused (trap:
// the host routes those to the parent).
template <int U128_NT>
__device__ __forceinline__ void fn128_union_body(
    const int *__restrict__ sel, const long *__restrict__ pos, int m, int R, int cps,
    const int *__restrict__ bt, int max_blocks, unsigned *__restrict__ who, long wstride,
    int *__restrict__ kslot, unsigned *__restrict__ kmask, unsigned char *__restrict__ kbits,
    int *__restrict__ nkeys, long kstride) {
  __shared__ int L[U128_ROWS][QSA_TOPK];
  __shared__ int op[U128_ROWS][QSA_TOPK + 1];  // owned prefix: op[r][j] = owned among L[r][0, j)
  __shared__ unsigned char own[U128_ROWS][QSA_TOPK];
  __shared__ int len[U128_ROWS];
  __shared__ long spos[U128_ROWS];
  __shared__ unsigned char kmb[4 * (U128_ROWS * QSA_TOPK + 16) + 48];  // a key's row mask (<= 8 bits)
  (void)who;
  (void)wstride;
  const int c = blockIdx.x, tid = threadIdx.x;
  const int lane = tid & 31, wave = tid >> 5;
  const int r0 = c * R, nr = min(R, m - r0);
  if (nr <= 0) {
    if (tid == 0) nkeys[c] = 0;
    return;
  }
  if (nr > U128_ROWS) __builtin_trap();
  const int seq = c / cps;
  if (tid < nr) spos[tid] = pos[r0 + tid];
  if (tid < U128_ROWS) len[tid] = 0;
  for (int i = tid; i < nr * QSA_TOPK; i += U128_NT) {
    const int rr = i / QSA_TOPK, j = i - rr * QSA_TOPK;
    L[rr][j] = sel[(long)(r0 + rr) * QSA_TOPK + j];
  }
  __syncthreads();
  // a list = its valid (>= 0) ascending prefix; -1 only pads its tail
  for (int i = tid; i < nr * QSA_TOPK; i += U128_NT) {
    const int rr = i / QSA_TOPK, j = i - rr * QSA_TOPK;
    if (L[rr][j] >= 0 && (j == QSA_TOPK - 1 || L[rr][j + 1] < 0)) len[rr] = j + 1;
  }
  __syncthreads();
  const long p_first = spos[0], p_last = spos[nr - 1];
  const int g_span = (int)(p_first >> 2), g_last = (int)(p_last >> 2);
  const int ns = g_last - g_span + 1;
  if (ns > 16) __builtin_trap();  // a decode CTA's rows are consecutive positions: ns <= 3
  // phase 1: ownership -- a list element below the span is owned by the lowest list holding it
  for (int i = tid; i < nr * QSA_TOPK; i += U128_NT) {
    const int rr = i / QSA_TOPK, j = i - rr * QSA_TOPK;
    unsigned char ow = 0;
    if (j < len[rr]) {
      const int x = L[rr][j];
      ow = x < g_span ? 1 : 0;
      for (int q = 0; q < rr && ow; ++q) {
        const int k = fn128_lb(L[q], len[q], x);
        if (k < len[q] && L[q][k] == x) ow = 0;
      }
    }
    own[rr][j] = ow;
  }
  __syncthreads();
  // phase 2: the owned prefix of each list (one wave a list, 16 ballots of 32)
  if (wave < nr) {
    int run = 0;
    for (int b = 0; b < QSA_TOPK; b += 32) {
      const int f = own[wave][b + lane];
      const unsigned bal = (unsigned)__ballot(f);
      op[wave][b + lane] = run + __popc(bal & ((1u << lane) - 1u));
      run += __popc(bal);
    }
    if (lane == 0) op[wave][QSA_TOPK] = run;
  }
  __syncthreads();
  int nown = 0;
  for (int q = 0; q < nr; ++q) nown += op[q][len[q]];
  const int nu = nown + ns;
  const int nk = 4 * (nu - 1) + (int)(p_last & 3) + 1;
  const int nkp = (nk + 47) / 48 * 48;
  int *ks = kslot + (long)c * kstride;
  unsigned *km = kmask + (long)c * kstride;
  const int *btr = bt + (long)seq * max_blocks;
  // phase 3a: every owned list element -> its union position, row mask and four keys
  for (int i = tid; i < nr * QSA_TOPK; i += U128_NT) {
    const int rr = i / QSA_TOPK, j = i - rr * QSA_TOPK;
    if (j < len[rr] && own[rr][j]) {
      const int x = L[rr][j];
      int u = 0;
      unsigned wv = 0;
      for (int q = 0; q < nr; ++q) {
        const int k = fn128_lb(L[q], len[q], x);
        u += op[q][k];
        if (k < len[q] && L[q][k] == x) wv |= 1u << q;
      }
      const int page = btr[x >> 2];
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {  // x < g_span: every token of the group is <= p_last
        ks[4 * u + jj] = page * 16 + 4 * (x & 3) + jj;
        km[4 * u + jj] = wv;
        kmb[4 * u + jj] = (unsigned char)wv;
      }
    }
  }
  // phase 3b: the span's groups (always kept) at nown + their offset; the open-group tails
  for (int s = tid; s < ns; s += U128_NT) {
    const int x = g_span + s;
    unsigned wv = 0;
    for (int q = 0; q < nr; ++q) {
      const int k = fn128_lb(L[q], len[q], x);
      if (k < len[q] && L[q][k] == x) wv |= 1u << q;
    }
    const int u = nown + s;
    const int page = btr[x >> 2];
    for (int jj = 0; jj < 4; ++jj) {
      const long t = 4L * x + jj;
      if (t > p_last) break;
      unsigned msk = wv;
      for (int q = 0; q < nr; ++q) {
        const long pq = spos[q];
        const long ts = ((pq + 1) >> 2) << 2;
        if (t >= ts && t <= pq) msk |= 1u << q;
      }
      ks[4 * u + jj] = page * 16 + 4 * (x & 3) + jj;
      km[4 * u + jj] = msk;
      kmb[4 * u + jj] = (unsigned char)msk;
    }
  }
  for (int k = nk + tid; k < nkp; k += U128_NT) {
    ks[k] = 0;
    km[k] = 0u;
    kmb[k] = 0;
  }
  if (tid == 0) nkeys[c] = nk;
  __syncthreads();
  // the attention's mask bytes [nkp / 8][32]: byte (b, q) bit e = row q may attend key 8b + e; as
  // words (q = 4w .. 4w + 3), rows past 7 carry no bit
  unsigned *kb = (unsigned *)(kbits + (long)c * kstride * 4);
  for (int i = tid; i < (nkp / 8) * 8; i += U128_NT) {
    const int b = i >> 3, w = i & 7;
    unsigned v = 0u;
    if (w < 2) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const unsigned mb = kmb[8 * b + e];
#pragma unroll
        for (int qq = 0; qq < 4; ++qq) v |= ((mb >> (4 * w + qq)) & 1u) << (8 * qq + e);
      }
    }
    kb[i] = v;
  }
}

#define FN128_UNION(NAME, NT)                                                                   \
  extern "C" __global__ __launch_bounds__(NT) void NAME(                                        \
      const int *__restrict__ sel, const long *__restrict__ pos, int m, int R, int cps,         \
      const int *__restrict__ bt, int max_blocks, unsigned *__restrict__ who, long wstride,     \
      int *__restrict__ kslot, unsigned *__restrict__ kmask, unsigned char *__restrict__ kbits, \
      int *__restrict__ nkeys, long kstride) {                                                  \
    fn128_union_body<NT>(sel, pos, m, R, cps, bt, max_blocks, who, wstride, kslot, kmask, kbits,  \
                         nkeys, kstride);                                                       \
  }
FN128_UNION(fn128_qsa_union_s, 512)
FN128_UNION(fn128_qsa_union_s1k, 1024)
