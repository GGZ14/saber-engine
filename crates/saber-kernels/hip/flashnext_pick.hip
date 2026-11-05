// Flash-Next (FN57): the step's SEAM PICK on the card -- each logits row's top-64 over the rank's
// vocabulary half, in the host seam's total order, so the host reads 512 B a row instead of the
// row (248 KB). Saber's own integer law; no floating-point arithmetic touches a logit.
//
// THE LAW (== `worker::seam_top_k(row, 64)` + the frame's (id + base, value) pairs, bit for bit):
//   key(b) = b & 0x8000 ? ~b : b | 0x8000 over the bf16 bit pattern (larger key = larger value;
//   -0 < +0; +NaN above +inf, -NaN below -inf); the 64 largest keys, value descending, the LOWER
//   index first on an exact key tie. Indices are unique, so (key desc, index asc) is a total order
//   and the answer does not depend on how the row is cut across threads.
//
//   fn_pick_top64_bf16  grid (rows), block 256, one row a workgroup:
//     1. the row's maximum key M (a block max);
//     2. a 4,096-bin histogram of the keys in the WINDOW [M - 4095, M] (key units: one bf16 ulp
//        each, 32 binades) -- nearly every element is below it and costs no atomic;
//     3. the threshold T: the 64th largest key (a suffix scan of the window's bins); when fewer
//        than 64 keys lie in the window (a pathological row) the FULL two-level radix select
//        (high byte, then low byte) finds T instead -- exact for every input;
//     4. every key > T (fewer than 64; appended in any order) and the LOWEST-index keys == T up
//        to 64 (each thread owns a contiguous index range; an exclusive scan of the per-thread
//        tie counts orders them);
//     5. the 64 sorted by (key desc, index asc): out[r][rank] = (index + base, bf16 -> f32 bits).
//   out row r = 64 x (u32 id, u32 f32 bits) = 512 B, the host frame's seam bytes exactly.
//
// Built with -ffp-contract=off (saber-kbuild's EXTRA_FLAGS; nothing here contracts anyway).
#include <hip/hip_runtime.h>
#include <stdint.h>

#define FNP_NT 256
#define FNP_K 64
#define FNP_WIN 4096
#define FNP_PER 488  // contiguous elements a thread (61 x 8): 256 x 488 = 124,928 >= 124,160

__device__ __forceinline__ uint32_t fnp_key(uint32_t b) {
  return (b & 0x8000u) ? (~b & 0xffffu) : (b | 0x8000u);
}

__device__ __forceinline__ uint32_t fnp_wave_max(uint32_t v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const uint32_t w = __shfl_xor(v, o, 32);
    v = v > w ? v : w;
  }
  return v;
}

// the block's exclusive prefix of v (thread order) and the total, through LDS
__device__ __forceinline__ uint32_t fnp_scan(uint32_t v, uint32_t* sc, uint32_t* total) {
  const int t = threadIdx.x;
  sc[t] = v;
  __syncthreads();
  for (int o = 1; o < FNP_NT; o <<= 1) {
    const uint32_t a = t >= o ? sc[t - o] : 0u;
    __syncthreads();
    sc[t] += a;
    __syncthreads();
  }
  const uint32_t incl = sc[t];
  *total = sc[FNP_NT - 1];
  __syncthreads();
  return incl - v;
}

// this thread's range [i0, i1) of the row (8-aligned: n and FNP_PER are multiples of 8)
__device__ __forceinline__ void fnp_range(int n, int* i0, int* i1) {
  const int a = threadIdx.x * FNP_PER;
  *i0 = a < n ? a : n;
  *i1 = a + FNP_PER < n ? a + FNP_PER : n;
}

__device__ __forceinline__ void fnp_load8(const uint16_t* x, int i, uint32_t k[8]) {
  const uint4 q = *reinterpret_cast<const uint4*>(x + i);
  const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    k[2 * j] = fnp_key(w[j] & 0xffffu);
    k[2 * j + 1] = fnp_key(w[j] >> 16);
  }
}

// the key T whose suffix count (keys >= T) first reaches FNP_K over `nb` bins of `hist` (bin b =
// key lo + b); bins are summed from the top. Returns T and the count of keys > T in *above.
__device__ uint32_t fnp_find(const uint32_t* hist, int nb, uint32_t lo, uint32_t need,
                             uint32_t* sc, uint32_t* above, uint32_t* found) {
  // thread t owns bins [nb - (t+1)*per, nb - t*per) counted from the top
  const int t = threadIdx.x;
  const int per = (nb + FNP_NT - 1) / FNP_NT;
  const int hi = nb - t * per;
  const int lo_b = hi - per > 0 ? hi - per : 0;
  uint32_t s = 0;
  for (int b = lo_b; b < hi; ++b) s += hist[b];
  uint32_t total;
  const uint32_t before = fnp_scan(s, sc, &total);  // keys in the bins above this thread's
  if (t == 0) *found = 0u;
  __syncthreads();
  if (before < need && before + s >= need) {
    uint32_t c = before;
    for (int b = hi - 1; b >= lo_b; --b) {
      if (c + hist[b] >= need) {
        *found = lo + (uint32_t)b;
        *above = c;
        break;
      }
      c += hist[b];
    }
  }
  __syncthreads();
  return *found;
}

extern "C" __global__ __launch_bounds__(256) void fn_pick_top64_bf16(
    const uint16_t* __restrict__ lg, uint32_t* __restrict__ out, int rows, int n, int ld,
    int base) {
  __shared__ uint32_t hist[FNP_WIN];
  __shared__ uint32_t sc[FNP_NT];
  __shared__ uint32_t wmax[FNP_NT / 32];
  __shared__ uint32_t sel_key[FNP_K];
  __shared__ uint32_t sel_idx[FNP_K];
  __shared__ uint32_t s_m, s_t, s_above, s_found, s_ngt, s_win;
  const int r = blockIdx.x;
  if (r >= rows) return;
  const int t = threadIdx.x;
  const uint16_t* x = lg + (size_t)r * (size_t)ld;
  int i0, i1;
  fnp_range(n, &i0, &i1);

  // 1. the maximum key
  uint32_t m = 0;
  for (int i = i0; i < i1; i += 8) {
    uint32_t k[8];
    fnp_load8(x, i, k);
#pragma unroll
    for (int j = 0; j < 8; ++j) m = m > k[j] ? m : k[j];
  }
  m = fnp_wave_max(m);
  if ((t & 31) == 0) wmax[t >> 5] = m;
  for (int b = t; b < FNP_WIN; b += FNP_NT) hist[b] = 0u;
  __syncthreads();
  if (t == 0) {
    uint32_t a = 0;
    for (int w = 0; w < FNP_NT / 32; ++w) a = a > wmax[w] ? a : wmax[w];
    s_m = a;
  }
  __syncthreads();
  const uint32_t M = s_m;
  const uint32_t lo = M >= FNP_WIN - 1 ? M - (FNP_WIN - 1) : 0u;

  // 2. the window's histogram
  for (int i = i0; i < i1; i += 8) {
    uint32_t k[8];
    fnp_load8(x, i, k);
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (k[j] >= lo) atomicAdd(&hist[k[j] - lo], 1u);
  }
  __syncthreads();
  const int nwin = (int)(M - lo) + 1;
  {
    uint32_t c = 0;
    for (int b = t; b < nwin; b += FNP_NT) c += hist[b];
    uint32_t total;
    fnp_scan(c, sc, &total);
    if (t == 0) s_win = total;
    __syncthreads();
  }
  const uint32_t need = (uint32_t)(n < FNP_K ? n : FNP_K);
  uint32_t T, above;
  if (s_win >= need) {
    // 3. the threshold inside the window
    T = fnp_find(hist, nwin, lo, need, sc, &s_above, &s_found);
    above = s_above;
  } else {
    // 3'. the full radix select: the high byte, then the low byte within it
    for (int b = t; b < 256; b += FNP_NT) hist[b] = 0u;
    __syncthreads();
    for (int i = i0; i < i1; i += 8) {
      uint32_t k[8];
      fnp_load8(x, i, k);
#pragma unroll
      for (int j = 0; j < 8; ++j) atomicAdd(&hist[k[j] >> 8], 1u);
    }
    __syncthreads();
    const uint32_t hb = fnp_find(hist, 256, 0u, need, sc, &s_above, &s_found);
    const uint32_t above_hb = s_above;
    __syncthreads();
    for (int b = t; b < 256; b += FNP_NT) hist[b] = 0u;
    __syncthreads();
    for (int i = i0; i < i1; i += 8) {
      uint32_t k[8];
      fnp_load8(x, i, k);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if ((k[j] >> 8) == hb) atomicAdd(&hist[k[j] & 0xffu], 1u);
    }
    __syncthreads();
    const uint32_t lb = fnp_find(hist, 256, 0u, need - above_hb, sc, &s_above, &s_found);
    T = (hb << 8) | lb;
    above = above_hb + s_above;
  }

  // 4. the keys above T (any order) and the lowest-index keys equal to T
  if (t == 0) s_ngt = 0u;
  uint32_t eq = 0;
  for (int i = i0; i < i1; i += 8) {
    uint32_t k[8];
    fnp_load8(x, i, k);
#pragma unroll
    for (int j = 0; j < 8; ++j) eq += k[j] == T ? 1u : 0u;
  }
  __syncthreads();
  uint32_t tot;
  const uint32_t before = fnp_scan(eq, sc, &tot);
  const uint32_t take = need - above;  // >= 1: how many ties at T make the cut
  uint32_t e = before;
  for (int i = i0; i < i1; i += 8) {
    uint32_t k[8];
    fnp_load8(x, i, k);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (k[j] > T) {
        const uint32_t at = atomicAdd(&s_ngt, 1u);
        sel_key[at] = k[j];
        sel_idx[at] = (uint32_t)(i + j);
      } else if (k[j] == T) {
        if (e < take) {
          sel_key[above + e] = T;
          sel_idx[above + e] = (uint32_t)(i + j);
        }
        ++e;
      }
    }
  }
  __syncthreads();

  // 5. the cut, sorted: rank = how many entries precede this one in (key desc, index asc)
  if (t < (int)need) {
    const uint32_t kt = sel_key[t], it = sel_idx[t];
    uint32_t rank = 0;
    for (int j = 0; j < (int)need; ++j) {
      const uint32_t kj = sel_key[j], ij = sel_idx[j];
      rank += (kj > kt || (kj == kt && ij < it)) ? 1u : 0u;
    }
    const uint32_t b = (kt & 0x8000u) ? (kt & 0x7fffu) : (~kt & 0xffffu);  // key -> bf16 bits
    uint32_t* o = out + (size_t)r * (2 * FNP_K) + 2 * rank;
    o[0] = it + (uint32_t)base;
    o[1] = b << 16;
  }
}
