// T6's SHADOW DIGEST: the per-proposal stage digests computed ON THE CARD, so a determinism
// monitor needs no read-back and no sync of its own (`--engine-race-check shadow`).
//
// T5's instrument (`--engine-race-check`, the `full` mode) digests every stage of a proposal on
// the HOST: ~25 blocking read-backs a proposal, a sync before the launch, the `_chk` all-reduce
// twins. 168 checked boots saw 0 events against 1 in 56 unchecked boots -- the shape of a timing
// race the checks' own syncs hide. These two kernels are the other way round: the engine enqueues
// them on the proposal's stream AFTER its last kernel, they write one small record of partial
// digests, and the record rides the pinned batch the proposal's tokens are already read with
// (`DraftResident::read_many`). The host adds zero waits; the card adds two short dispatches after
// the work it digests has finished (stream order), so they cannot reorder anything inside it.
//
// THE HASH is order-free by construction: a region of `bytes` bytes is read as little-endian
// 64-bit words `w_i` (a short tail zero-padded) and every word contributes `mix(w_i, i)` by XOR, so
// the partials of any partition of the words XOR to the same value, whatever the grid, the wave
// order or the workgroup count -- the digest is a function of the bytes and their positions only.
// `mix(w, i) = sm64(w ^ sm64(i))` (splitmix64's finaliser), so a swapped pair of words, a moved
// word and a flipped bit all change it. The host folds a region's partials and finalises with the
// length (`saber_kernels::race_digest::fold`), and `race_digest::host_digest` is the same function
// on the host, which is what the unit tests hold the device's partitioning against.
//
// Both kernels: one workgroup of 256 threads per PART, `parts` parts per region; each part walks
// the region's words at a stride of `parts * 256`; the workgroup XOR-reduces in LDS and thread 0
// writes one u64. No atomics, no uninitialised LDS read (every slot is written before the
// barrier), no dependence on launch order.

#include <hip/hip_runtime.h>

#define RD_BLOCK 256

__device__ __forceinline__ unsigned long long rd_sm64(unsigned long long x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

__device__ __forceinline__ unsigned long long rd_mix(unsigned long long w, unsigned long long i) {
  return rd_sm64(w ^ rd_sm64(i));
}

// Little-endian word of `n` (< 8) bytes, zero-padded.
__device__ __forceinline__ unsigned long long rd_tail(const unsigned char *p, unsigned long long n) {
  unsigned long long w = 0;
  for (unsigned long long b = 0; b < n; ++b) w |= (unsigned long long)p[b] << (8 * b);
  return w;
}

// XOR-reduce `h` over the workgroup; thread 0 holds the result.
__device__ __forceinline__ unsigned long long rd_reduce(unsigned long long h) {
  __shared__ unsigned long long s[RD_BLOCK];
  s[threadIdx.x] = h;
  __syncthreads();
  for (unsigned int w = RD_BLOCK / 2; w > 0; w >>= 1) {
    if (threadIdx.x < w) s[threadIdx.x] ^= s[threadIdx.x + w];
    __syncthreads();
  }
  return s[0];
}

// The words `[0, nw)` of `p`, this part's share, keyed from `key0`.
__device__ __forceinline__ unsigned long long rd_walk(const unsigned char *p, unsigned long long nw,
                                                      unsigned long long key0, unsigned int part,
                                                      unsigned int parts) {
  unsigned long long h = 0;
  const unsigned long long stride = (unsigned long long)parts * RD_BLOCK;
  unsigned long long i = (unsigned long long)part * RD_BLOCK + threadIdx.x;
  if ((((unsigned long long)p) & 7ull) == 0) {
    const unsigned long long *q = (const unsigned long long *)p;
    for (; i < nw; i += stride) h ^= rd_mix(q[i], key0 + i);
  } else {
    for (; i < nw; i += stride) h ^= rd_mix(rd_tail(p + 8 * i, 8), key0 + i);
  }
  return h;
}

// REGIONS: `desc` is `n` pairs (device address, bytes), region `n` is `extra` / `extra_bytes` (the
// proposal's aux window, bound per launch). Grid `(n + 1) * parts`; `out[r * parts + part]`.
extern "C" __global__ void __launch_bounds__(RD_BLOCK)
saber_race_digest_regions(const unsigned long long *__restrict__ desc,
                          const unsigned char *__restrict__ extra,
                          unsigned long long *__restrict__ out, long extra_bytes, int n,
                          int parts) {
  const unsigned int r = blockIdx.x / (unsigned int)parts;
  const unsigned int part = blockIdx.x % (unsigned int)parts;
  const unsigned char *p;
  unsigned long long bytes;
  if (r < (unsigned int)n) {
    p = (const unsigned char *)desc[2 * r];
    bytes = desc[2 * r + 1];
  } else {
    p = extra;
    bytes = (unsigned long long)extra_bytes;
  }
  const unsigned long long nw = bytes >> 3;
  unsigned long long h = p ? rd_walk(p, nw, 0, part, (unsigned int)parts) : 0;
  if (p && part == 0 && threadIdx.x == 0 && (bytes & 7ull))
    h ^= rd_mix(rd_tail(p + 8 * nw, bytes & 7ull), nw);
  h = rd_reduce(h);
  if (threadIdx.x == 0) out[blockIdx.x] = h;
}

// THE DRAFTER'S K/V PAGES the proposal read (and its own tail tile wrote): for every request row
// `r < reqs` of the shape, the pages `j < ceil(seq_lens[r] / page_tokens)` of its block-table row,
// in both caches and every layer. A page's words are keyed by its LOGICAL place `(r, j)`, not by
// the physical page id (the tables' own digest carries that), so the digest is the context the
// attention saw. Grid `layers * 2 * parts` (layer-major, K then V); `out[(l * 2 + c) * parts +
// part]`. `block_bytes` is a multiple of 8 (a page of bf16/e4m3 rows).
extern "C" __global__ void __launch_bounds__(RD_BLOCK)
saber_race_digest_kv(const int *__restrict__ block_table, const int *__restrict__ seq_lens,
                     const unsigned char *__restrict__ kbase,
                     const unsigned char *__restrict__ vbase,
                     unsigned long long *__restrict__ out, long per_layer, long block_bytes,
                     int reqs, int bt_stride, int page_tokens, int parts, int blocks) {
  const unsigned int lc = blockIdx.x / (unsigned int)parts;
  const unsigned int part = blockIdx.x % (unsigned int)parts;
  const unsigned int layer = lc >> 1;
  const unsigned char *base = ((lc & 1u) ? vbase : kbase) + (unsigned long long)layer * per_layer;
  const unsigned long long wpp = (unsigned long long)block_bytes >> 3;
  unsigned long long h = 0;
  for (int r = 0; r < reqs; ++r) {
    int len = seq_lens[r];
    if (len <= 0) continue;
    int np = (len + page_tokens - 1) / page_tokens;
    if (np > bt_stride) np = bt_stride;
    for (int j = (int)part; j < np; j += parts) {
      const int pg = block_table[r * bt_stride + j];
      if (pg < 0 || pg >= blocks) continue;
      const unsigned long long *q =
          (const unsigned long long *)(base + (unsigned long long)pg * block_bytes);
      const unsigned long long key0 = ((unsigned long long)r * bt_stride + j) * wpp;
      for (unsigned long long i = threadIdx.x; i < wpp; i += RD_BLOCK) h ^= rd_mix(q[i], key0 + i);
    }
  }
  h = rd_reduce(h);
  if (threadIdx.x == 0) out[blockIdx.x] = h;
}
