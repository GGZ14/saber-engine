// Flash-Next (FN19): the QSA selection -- Saber's own arithmetic, the spec is
// tools/capture/fn19/qsa_spec.py (a numpy float32 program that reproduces every op bit for bit).
//
//   fn_qsa_score_r32        the indexer's block scores: for each query row r and each VISIBLE
//                           compressed group g < (pos_r + 1) / 4,
//                             dot_h  = sum over d = 0..127, IN ORDER, of q[r, h, d] * k[g, d]
//                                      (each product exact: bf16 x bf16; each add rounded)
//                             score  = (((0 + relu(dot_0)) + relu(dot_1)) + relu(dot_2)) + relu(dot_3)
//                             score  = score / sqrt(128)      (fp32, correctly rounded)
//                           = upstream `_qsa_mqa_paged_kernel`'s formula (its sum order is Triton's).
//   fn_qsa_topk512          THE SELECTION LAW, exact: a row with visible <= 512 takes every
//                           visible group; above, the 512 largest scores, EXACT ties at the 512th
//                           value resolved to the LOWER group index; the list is written ascending
//                           (-1 padded). Radix select on the fp32 bits (scores are >= +0: the bit
//                           order is the value order) 11 + 11 + 10 bits, then one ordered pass.
//   fn_qsa_union            the attention's key list per attention CTA (a prefill CTA = 32 query
//                           positions, a decode CTA = one sequence's verify rows): the ascending
//                           union of the CTA's rows' selected groups + the groups of the CTA's own
//                           positions, expanded to tokens; per key its KV slot and a row mask
//                           (bit q: row q may attend it = its group was selected by q, or it is in
//                           q's open-group tail [4 * ((pos + 1) / 4), pos]); the key count.
//
// Built with -ffp-contract=off (saber-kbuild's EXTRA_FLAGS): every fp32 multiply and add rounds on
// its own. Loads past a live range are CLAMPED, never predicated (the predicated-staging law).
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
// grid (ceil(gcap / 64), ceil(m / rt)), block 256. A workgroup scores 64 groups x rt (<= 32) rows of
// ONE sequence (the host cuts row tiles inside a sequence). Thread (g = tid & 63, rs = tid >> 6)
// holds group g's key (128 bf16 = 64 VGPRs) and walks rows rs, rs + 4, ...; the tile's queries sit
// in LDS (read as broadcasts: every lane of a wave takes the same row).
extern "C" __global__ __launch_bounds__(256) void fn_qsa_score_r32(
    const __bf16 *__restrict__ q, const __bf16 *__restrict__ comp, const int *__restrict__ bt,
    int max_blocks, const long *__restrict__ pos, float *__restrict__ scores, long sstride, int m,
    int rows_per_seq, int rt) {
  __shared__ uint4 sq[32 * QSA_IDX_H * QSA_IDX_D / 8];  // 32 KiB: [row][head][128] bf16
  const int tid = threadIdx.x;
  const int r0 = blockIdx.y * rt;
  const int nr = min(rt, m - r0);
  if (nr <= 0) return;
  const int g0 = blockIdx.x * 64;
  long pmax = pos[r0];
  for (int i = 1; i < nr; ++i) pmax = max(pmax, pos[r0 + i]);
  const int gvis = (int)((pmax + 1) >> 2);
  if (g0 >= gvis) return;  // workgroup-uniform
  const int seq = r0 / rows_per_seq;
  for (int i = tid; i < nr * QSA_IDX_H * QSA_IDX_D / 8; i += 256)
    sq[i] = *(const uint4 *)(q + (long)r0 * QSA_IDX_H * QSA_IDX_D + (long)i * 8);
  const int gl = tid & 63, rs = tid >> 6;
  const int g = min(g0 + gl, gvis - 1);  // clamped: a lane past the tile's visible range loads a
                                         // real row and writes nothing
  const long page = bt[(long)seq * max_blocks + (g >> 2)];
  const uint4 *kp = (const uint4 *)(comp + (page * 4 + (g & 3)) * QSA_IDX_D);
  uint4 kr[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) kr[i] = kp[i];
  __syncthreads();
  for (int rr = rs; rr < nr; rr += 4) {
    const int r = r0 + rr;
    const int vis = (int)((pos[r] + 1) >> 2);
    float s = 0.f;
#pragma unroll
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

// ------------------------------------------------------------------------------------------ top-k
// Block-wide exclusive prefix of a 0/1 flag over 256 threads (8 waves of 32), + the total.
__device__ __forceinline__ int qsa_scan1(int f, int *wsum, int &total) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const unsigned bal = (unsigned)__ballot(f);
  const int in_wave = __popc(bal & ((1u << lane) - 1u));
  if (lane == 0) wsum[wave] = __popc(bal);
  __syncthreads();
  int off = 0, tot = 0;
#pragma unroll
  for (int w = 0; w < 8; ++w) {
    const int v = wsum[w];
    off += (w < wave) ? v : 0;
    tot += v;
  }
  __syncthreads();
  total = tot;
  return off + in_wave;
}

// The bin of the k-th largest key among the histogram's `nb` bins (descending walk): returns the bin
// and, through `above`, the count of keys in strictly higher bins. Thread t owns bins
// [nb - 1 - t * per .. nb - per - t * per] (descending).
__device__ __forceinline__ int qsa_kth_bin(const int *hist, int nb, int k, int *red, int &above) {
  const int tid = threadIdx.x, per = nb / 256;
  const int top = nb - 1 - tid * per;
  int mine = 0;
  for (int i = 0; i < per; ++i) mine += hist[top - i];
  red[tid] = mine;
  __syncthreads();
  // exclusive prefix over threads (descending bins): a serial 256-entry scan by wave 0 is cheap
  if (tid == 0) {
    int run = 0;
    for (int t = 0; t < 256; ++t) {
      const int v = red[t];
      red[t] = run;
      run += v;
    }
  }
  __syncthreads();
  int run = red[tid];
  int found = -1, fab = 0;
  for (int i = 0; i < per; ++i) {
    const int c = hist[top - i];
    if (run < k && run + c >= k) {
      found = top - i;
      fab = run;
    }
    run += c;
  }
  __syncthreads();
  if (found >= 0) {
    red[0] = found;
    red[1] = fab;
  }
  __syncthreads();
  const int b = red[0];
  above = red[1];
  __syncthreads();
  return b;
}

// grid (m), block 256. scores [m, sstride] f32 (only g < visible read); sel [m, 512] i32.
extern "C" __global__ __launch_bounds__(256) void fn_qsa_topk512(
    const float *__restrict__ scores, long sstride, const long *__restrict__ pos,
    int *__restrict__ sel, int m) {
  __shared__ int hist[2048];
  __shared__ int red[256];
  __shared__ int wsum[8];
  const int r = blockIdx.x, tid = threadIdx.x;
  if (r >= m) return;
  const int vis = (int)((pos[r] + 1) >> 2);
  int *out = sel + (long)r * QSA_TOPK;
  if (vis <= QSA_TOPK) {  // upstream's shortcut: every visible group
    for (int j = tid; j < QSA_TOPK; j += 256) out[j] = j < vis ? j : -1;
    return;
  }
  const uint32_t *key = (const uint32_t *)(scores + (long)r * sstride);
  // ---- three radix levels: bits 31..21 (11), 20..10 (11), 9..0 (10)
  uint32_t prefix = 0;
  int k = QSA_TOPK, above_total = 0;
  const int shifts[3] = {21, 10, 0};
  const int widths[3] = {11, 11, 10};
#pragma unroll
  for (int lv = 0; lv < 3; ++lv) {
    const int nb = 1 << widths[lv];
    for (int i = tid; i < 2048; i += 256) hist[i] = 0;
    __syncthreads();
    const int hs = shifts[lv] + widths[lv];  // bits above this level must equal `prefix`
    for (int g = tid; g < vis; g += 256) {
      const uint32_t kv = key[g];
      const bool match = lv == 0 ? true : ((kv >> hs) == prefix);
      if (match) atomicAdd(&hist[(kv >> shifts[lv]) & (nb - 1)], 1);
    }
    __syncthreads();
    int above = 0;
    const int b = qsa_kth_bin(hist, nb, k, red, above);
    prefix = (prefix << widths[lv]) | (uint32_t)b;
    k -= above;
    above_total += above;
  }
  const uint32_t T = prefix;         // the 512th largest key
  const int need = QSA_TOPK - above_total;  // how many keys == T are taken (>= 1)
  // ---- ordered pass: every key > T, and the first `need` keys == T in index order
  int eq_run = 0, out_run = 0;
  for (int base = 0; base < vis; base += 256) {
    const int g = base + tid;
    const uint32_t kv = key[min(g, vis - 1)];
    const bool valid = g < vis;
    const int eq = (valid && kv == T) ? 1 : 0;
    int eq_tot = 0;
    const int eq_rank = eq_run + qsa_scan1(eq, wsum, eq_tot);
    const int take = (valid && (kv > T || (eq && eq_rank < need))) ? 1 : 0;
    int take_tot = 0;
    const int at = out_run + qsa_scan1(take, wsum, take_tot);
    if (take) out[at] = g;
    eq_run += eq_tot;
    out_run += take_tot;
  }
  for (int j = out_run + tid; j < QSA_TOPK; j += 256) out[j] = -1;  // never at vis > 512 (a guard)
}

// ------------------------------------------------------------------------------------------ union
// grid (ncta), block 256. CTA c covers rows [c * R, min(c * R + R, m)) of sequence c / cps (the rows
// of a CTA are position-ascending). `who` [ncta, wstride] u32 must be ZERO on entry (a seeded region;
// this kernel leaves it zero again). Writes kslot / kmask [ncta, kstride] for the keys
// [0, round_up(nk, 48)) -- kslot = the key's KV slot (page * 16 + slot in page), kmask its row
// mask (the pad keys: slot 0, mask 0) --, the attention's per-row mask bytes
// kbits [ncta, kstride / 8, 32] and nkeys[c] = nk.
extern "C" __global__ __launch_bounds__(256) void fn_qsa_union(
    const int *__restrict__ sel, const long *__restrict__ pos, int m, int R, int cps,
    const int *__restrict__ bt, int max_blocks, unsigned *__restrict__ who, long wstride,
    int *__restrict__ kslot, unsigned *__restrict__ kmask,
    unsigned char *__restrict__ kbits,
    int *__restrict__ nkeys, long kstride) {
  __shared__ long spos[32];
  __shared__ int wsum[8];
  const int c = blockIdx.x, tid = threadIdx.x;
  const int r0 = c * R, nr = min(R, m - r0);
  if (nr <= 0) {
    if (tid == 0) nkeys[c] = 0;
    return;
  }
  const int seq = c / cps;
  if (tid < nr) spos[tid] = pos[r0 + tid];
  __syncthreads();
  unsigned *w = who + (long)c * wstride;
  for (int i = tid; i < nr * QSA_TOPK; i += 256) {
    const int rr = i / QSA_TOPK;
    const int g = sel[(long)(r0 + rr) * QSA_TOPK + (i - rr * QSA_TOPK)];
    if (g >= 0) atomicOr(&w[g], 1u << rr);
  }
  __threadfence_block();
  __syncthreads();
  const long p_first = spos[0], p_last = spos[nr - 1];
  const int g_span = (int)(p_first >> 2), g_last = (int)(p_last >> 2);
  int *ks = kslot + (long)c * kstride;
  unsigned *km = kmask + (long)c * kstride;
  const int *btr = bt + (long)seq * max_blocks;
  int u_run = 0;
  for (int base = 0; base <= g_last; base += 256) {
    const int g = base + tid;
    const bool valid = g <= g_last;
    // read-and-clear at L2 (the scatter's atomics live there; a plain load could see a stale L0 line)
    const unsigned wv = valid ? atomicExch(&w[g], 0u) : 0u;
    const int keep = (valid && (wv != 0u || g >= g_span)) ? 1 : 0;
    int tot = 0;
    const int u = u_run + qsa_scan1(keep, wsum, tot);
    if (keep) {
      const int page = btr[g >> 2];
      for (int j = 0; j < 4; ++j) {
        const long t = 4L * g + j;
        if (t > p_last) break;
        unsigned msk = wv;
        if (g >= g_span) {
          for (int q = 0; q < nr; ++q) {
            const long pq = spos[q];
            const long ts = ((pq + 1) >> 2) << 2;
            if (t >= ts && t <= pq) msk |= 1u << q;
          }
        }
        ks[4 * u + j] = page * 16 + 4 * (g & 3) + j;
        km[4 * u + j] = msk;
      }
    }
    u_run += tot;
  }
  const int nk = 4 * (u_run - 1) + (int)(p_last & 3) + 1;
  const int nkp = (nk + 47) / 48 * 48;
  for (int k = nk + tid; k < nkp; k += 256) {
    ks[k] = 0;
    km[k] = 0u;
  }
  if (tid == 0) nkeys[c] = nk;
  // the attention's form of the mask: per 8-key block b and CTA row q ONE byte, bit e = key 8b + e
  // (a lane of the attention reads its own row's byte for its 8 keys: one load, one VGPR)
  __syncthreads();
  unsigned char *kb = kbits + (long)c * kstride * 4;
  for (int i = tid; i < (nkp / 8) * 32; i += 256) {
    const int b = i >> 5, q = i & 31;
    unsigned v = 0u;
#pragma unroll
    for (int e = 0; e < 8; ++e) v |= ((km[8 * b + e] >> q) & 1u) << e;
    kb[i] = (unsigned char)v;
  }
}
