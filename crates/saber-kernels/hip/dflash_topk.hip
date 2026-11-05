// The selector's candidate top-k: `torch.topk(logits, 16)` over this rank's vocabulary shard.
//
// WHICH top-k this is, because the census's ~160 us ATen family is three different operations and
// only one of them is a reduction a workgroup can own:
//
//   mbtopk::{computeBlockDigitCounts, computeBlockwiseWithinKCounts, gatherTopK, fill}
//       ~40 us over 5 launches, grid 125440 -- `get_top_k_tokens` doing `torch.topk(logits, 16)`
//       over the 124160-wide vocab SHARD for the drafter's seven mask positions. THIS KERNEL.
//   sbtopk::gatherTopK
//       98.1 us over 2 launches, grid 16384 = 16 rows x 1024 threads -- the int2 draft head's
//       `bm.topk(RERANK)`, a top-80 of 15520. NOT ported; see the LEDGER for why k=80 is not
//       ownable the way k=16 is.
//   warpMergeSortTopK / bitonicSortKVInPlace / warpMergeSortKVInPlace
//       ~10 us, grids 224 / 112 / 512 -- the top-16 of the 32 values the TP all-gather produces,
//       and two small key-value sorts. Too small to be worth a launch of their own.
//
// WHY ONE WORKGROUP OWNS IT, exactly rather than approximately: a per-thread local top-k is exact
// if and only if each thread's capacity is k, because a single thread's slice may contain all k
// winners. k = 16 fits (16 entries per thread in LDS, 32 KiB for 256 threads), so one pass over the
// row plus a merge of the 256 local lists is the whole algorithm and nothing is approximated. At
// k = 80 the same argument needs 80 entries per thread and stops fitting, which is why the draft
// head's selection stays with ATen's multi-pass radix select.
//
// TIE ORDER IS THE GATE, AND **ATen's TIE ORDER DEPENDS ON WHICH ALGORITHM IT PICKED**. That is the
// finding, and it is why this has to be read out of the output at the RIGHT SHAPE rather than from
// the source or from any one experiment. With exact duplicates planted, the same `torch.topk` call
// gives:
//
//   n = 124160, k = 16, bf16  (mbtopk)      value at [22437, 84517] -> rank 0 = 22437, rank 1 = 84517
//   n = 15520,  k = 80, fp32  (sbtopk)      value at [471, 8231]    -> rank 0 = 471
//   n = 16,     k = 16, fp32  (warp sort)   value at [4, 12]        -> rank 0 = **12**, rank 1 = 4
//
// So the two selection algorithms emit ties in ASCENDING index order and the small full-sort emits
// them DESCENDING. This kernel serves the selector's shape, which is the first row, so equal values
// go to the LOWER index first. A port that had read the rule off the 16-wide case would have had it
// exactly backwards -- and the logits are bf16, 8 mantissa bits over a 124160-wide row, so ties are
// ordinary rather than a corner case.
//
// Preferring the lower index also keeps the comparator a total order, so the merge tree stays
// order-independent and its shape is free.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// THE INDEX WIDTH IS i64, AND IT IS THE CONSUMER'S LAW RATHER THAN A PREFERENCE.
//
// `torch.topk` returns `int64` indices, and `dflash_selector_score_r256_k16` -- the only consumer
// of this selection -- declares `cand` as `const long *`, because F24's finding was that
// declaring that slot narrower reads the right addresses and the wrong bytes. Through F27 this
// kernel wrote `int`, so the producer and the consumer of one edge disagreed about an element
// width and the unseeded chain had to re-widen on the HOST between them -- a round trip in a
// proposal whose whole claim is `host_round_trips() == 0`, and a lie about the launch set if it
// had been done with a kernel instead. The staged gate could never see it: it uploads prod's own
// i64 `walk.candidate_ids` into that slot.
//
// So the selection writes what its consumer reads and what ATen returns. The value is a column of
// a 124,160-wide vocabulary shard either way; the WIDTH is the fact.
#define DTK_NT   256   // threads per workgroup; one workgroup per row
#define DTK_K     16   // the selector's `selector_top_k`, and the per-thread capacity

struct DtkEntry {
  float v;
  int i;
};

typedef __bf16 dtk_bf16x8 __attribute__((ext_vector_type(8)));

#define DTK_VEC    8   // bf16 per thread per iteration: one 16-byte load, 4 KiB a wave

// Strictly greater wins; `LOWTIE` picks which index wins an exact tie. Both are TOTAL orders over
// (value, index) because indices are unique, which is what makes the answer independent of how
// the row is partitioned -- the property the wide load below relies on.
template <bool LOWTIE>
__device__ __forceinline__ bool dtk_cmp(float av, int ai, float bv, int bi) {
  return LOWTIE ? (av > bv || (av == bv && ai < bi)) : (av > bv || (av == bv && ai > bi));
}

// One workgroup, one row. `LOWTIE` is the tie rule; everything else is shared between the route
// and its control, because a control that does not carry the route's LAYOUT stops being a control
// for the route.
//
// THE WIDE LOAD, and why it cannot change the answer. Through F29 each thread walked the row with
// a STRIDED scalar 2-byte load (`for (e = tid; e < n; e += 256)`), ~970 of them with a dependent
// compare behind each: 272.6 us for a 3.48 MB read is 12.8 GB/s of a 635 GB/s card, and the F29
// table named it and the top-80 below as the only two launches in the whole proposal that are not
// within 15% of the DRAM roofline. Seven rows is seven workgroups of 256 threads -- 1,792 of the
// card's 65,536 -- so it is LATENCY-bound at 1/16 occupancy rather than bandwidth-bound, and the
// lever is bytes in flight per thread, not bytes moved.
//
// Each thread now owns whole 8-element GROUPS (`g = tid; g += 256`), one 16-byte `dwordx4` per
// iteration, with the NEXT group's load issued before the current group's eight compares so there
// is always one fetch outstanding. 970 scalar loads become 121 vector ones.
//
// The per-thread SETS change -- thread 0 used to hold elements {0, 256, 512, ...} and now holds
// {0..7, 2048..2055, ...} -- and the ANSWER DOES NOT, exactly rather than approximately: the
// comparator is a total order, so the top-k of the row is a unique SEQUENCE, and a per-thread top-k
// is exact iff each thread's capacity is k (k = 16 = DTK_K here). Any partition of the row into
// slices of any shape therefore merges to the same k entries in the same order. That is the same
// argument the per-thread list itself rests on, applied to a different partition; it is the reason
// this is a layout change and not an arithmetic one, and the gate checks it rather than trusting
// it.
template <bool LOWTIE>
__device__ __forceinline__ void dtk_row(const __bf16 *__restrict__ x, float *__restrict__ vals,
                                        long *__restrict__ idx, int rows, int n, int k) {
  const int row = (int)blockIdx.x;
  if (row >= rows || k > DTK_K) return;
  const int tid = (int)threadIdx.x;
  __shared__ DtkEntry local[DTK_NT * DTK_K];
  __shared__ DtkEntry best[DTK_NT];

  // The lists are INTERLEAVED (`local[j * DTK_NT + tid]`): eight-byte entries at a 16-entry stride
  // put every lane of a wave on one LDS bank pair, a 32-way conflict twice per input element.
  // The THRESHOLD (`local[k-1]`) is a register pair, because the common case is "this element does
  // not beat the k-th" and that must not touch LDS at all.
#define DTK_MINE(j) local[(j) * DTK_NT + tid]
  const int sent = LOWTIE ? 0x7fffffff : -1;     // loses every tie under this rule
  for (int j = 0; j < k; ++j) {
    DTK_MINE(j).v = -INFINITY;
    DTK_MINE(j).i = sent;
  }
  float thr_v = -INFINITY;
  int thr_i = sent;
  const __bf16 *xr = x + (long)row * n;

#define DTK_OFFER(vv, ee)                                                                          \
  do {                                                                                             \
    const float v_ = (vv);                                                                         \
    const int e_ = (ee);                                                                           \
    if (dtk_cmp<LOWTIE>(v_, e_, thr_v, thr_i)) {                                                   \
      int p = k - 1;                                                                               \
      while (p > 0 && dtk_cmp<LOWTIE>(v_, e_, DTK_MINE(p - 1).v, DTK_MINE(p - 1).i)) {             \
        DTK_MINE(p) = DTK_MINE(p - 1);                                                             \
        --p;                                                                                       \
      }                                                                                            \
      DTK_MINE(p).v = v_;                                                                          \
      DTK_MINE(p).i = e_;                                                                          \
      thr_v = DTK_MINE(k - 1).v;                                                                    \
      thr_i = DTK_MINE(k - 1).i;                                                                    \
    }                                                                                              \
  } while (0)

  // TWO groups an iteration, both loads issued before either is looked at, so there are two
  // 16-byte fetches outstanding per thread instead of one. At 1,792 threads that is the only lever
  // there is: the row is read once and the arithmetic is eight compares that almost never fire, so
  // what the launch costs is how many bytes a thread has asked for and not yet needed.
  const int vt = n / DTK_VEC;                    // whole 8-element groups; the tail is scalar
  for (int g = tid; g < vt; g += 2 * DTK_NT) {
    const int g1 = g + DTK_NT;
    const bool two = g1 < vt;
    const dtk_bf16x8 v0 = *(const dtk_bf16x8 *)(xr + (long)g * DTK_VEC);
    dtk_bf16x8 v1;
    if (two) v1 = *(const dtk_bf16x8 *)(xr + (long)g1 * DTK_VEC);
#pragma unroll
    for (int q = 0; q < DTK_VEC; ++q) DTK_OFFER((float)v0[q], g * DTK_VEC + q);
    if (two) {
#pragma unroll
      for (int q = 0; q < DTK_VEC; ++q) DTK_OFFER((float)v1[q], g1 * DTK_VEC + q);
    }
  }
  for (int e = vt * DTK_VEC + tid; e < n; e += DTK_NT) DTK_OFFER((float)xr[e], e);
#undef DTK_OFFER
  __syncthreads();

  // ---- the merge: k rounds of a block-wide best-and-consume over the 256 sorted lists --------
  // Each thread tracks how far into its own list the merge has eaten, so a round costs one
  // comparison per thread rather than a re-scan.
  int head = 0;
  for (int j = 0; j < k; ++j) {
    DtkEntry cand = head < k ? DTK_MINE(head) : DtkEntry{-INFINITY, sent};
    best[tid] = cand;
    __syncthreads();
    // Tree reduction; the comparator is order-independent so this needs no fixed shape.
    for (int s = DTK_NT / 2; s > 0; s >>= 1) {
      if (tid < s) {
        const DtkEntry o = best[tid + s];
        if (dtk_cmp<LOWTIE>(o.v, o.i, best[tid].v, best[tid].i)) best[tid] = o;
      }
      __syncthreads();
    }
    const DtkEntry win = best[0];
    if (tid == 0) {
      vals[(long)row * k + j] = win.v;
      idx[(long)row * k + j] = (long)win.i;
    }
    // Whichever thread owned the winner advances its cursor. Comparing the index is enough: an
    // index appears in exactly one thread's list.
    if (head < k && DTK_MINE(head).i == win.i) ++head;
    __syncthreads();
  }
#undef DTK_MINE
}

extern "C" __global__ __launch_bounds__(DTK_NT) void dflash_topk_bf16_k16(
    const __bf16 *__restrict__ x,     // [rows, n] bf16 logits
    float *__restrict__ vals,         // [rows, k] fp32 -- ATen returns the INPUT dtype; the caller
                                      // converts, so this writes fp32 and the gate compares after
                                      // the same conversion
    long *__restrict__ idx,           // [rows, k] **i64**, see above
    int rows, int n, int k) {
  dtk_row<true>(x, vals, idx, rows, n, k);
}

// CONTROL for the tie rule above, and the reason it has to exist is a MEASUREMENT rather than a
// worry. Against prod's own `torch.topk(logits, 16)` on the r26 capture (2026-09-21, F26, three
// prompts, host, no card): the selected SET is identical, the VALUE sequence is identical, and the
// INDICES differ in 11 / 8 / 13 of 112 -- every one of them a pair of EXACTLY EQUAL bf16 logits.
// Neither rule reproduces prod: ascending index differs in 11 and descending in 13 on the same
// row, so ATen's order at this shape is not one rule at all (`mbtopk` merges per-block results and
// the order among equal values falls out of the merge). This arm is the other rule, and it exists
// so the claim "the difference is ties and only ties" is falsifiable: it must move exactly the
// tied positions and nothing else. It is the SAME function as the route with one template
// argument flipped, so it cannot drift away from the route's layout.
extern "C" __global__ __launch_bounds__(DTK_NT) void dflash_topk_bf16_k16_hightie(
    const __bf16 *__restrict__ x, float *__restrict__ vals, long *__restrict__ idx,
    int rows, int n, int k) {
  dtk_row<false>(x, vals, idx, rows, n, k);
}

// ---- the INT2 DRAFT HEAD's candidate selection: top-80 of the block maxima ---------------------
//
// `ref_drafthead._apply_head_int2` runs `bm.topk(RERANK, dim=1)` over `[M, nblk * KCAND]`
// fp32 block maxima -- 31040 columns at TP=1 (3880 blocks of 64 columns, 8 candidates each) and
// 15520 at TP=2 -- and RERANK is the serve's 80. The census sees it as `sbtopk::gatherTopK`, two
// launches and 98.1 us, the top-k family's largest item.
//
// THE LEDGER SAID THIS ONE WAS NOT OWNABLE, and it was right about the algorithm it had in mind.
// `dflash_topk_bf16_k16` above is exact because a per-thread local top-k is exact iff the
// per-thread capacity is k: at k = 16 that is 16 entries per thread and it fits, at k = 80 it does
// not, and the entry concluded "exactness would require a multi-pass radix select, which is what
// ATen already runs". That is the conclusion this kernel takes up rather than the refusal: a radix
// select IS what runs here, and it is two launches because the selection and the gather are two
// passes over the row with a per-row threshold between them -- the same shape ATen's own two
// launches have, and the count the plan declares.
//
// TIE ORDER IS NOT A RULE HERE, IT IS DESIGNED AWAY. The ledger's finding -- that ATen's tie order
// depends on which algorithm it picked, ascending index for the two SELECTIONS and descending for
// the small full sort -- is a hazard only while the ordering is a partial order. The key this
// kernel selects on is the 64-bit pair
//
//     composite = (u64)monotone(value) << 32 | (u32)~index
//
// which is a TOTAL order: value descending, then index ASCENDING, which is `sbtopk`'s own order at
// this shape (measured, see the header of this file). No two elements compare equal, so the
// threshold is a single element, the selected set is exactly k, and the gather needs no
// tie-breaking pass and no capacity guess. `monotone` is the standard order-preserving fp32 -> u32
// map (flip the sign bit for positives, invert everything for negatives), which is exact for
// -inf too -- and -inf is most of this row, because `_ref_topk_only` fills every entry the
// rerank did not touch.
//
// One workgroup owns one row in both passes. The row is 124 KB at TP=1, so the four byte-passes of
// pass 1 stream it four times out of L2; nothing here is bandwidth-shaped at 16 rows.
// ONE WORKGROUP PER ROW AND SIXTEEN ROWS, which is 1/16 of the card at 256 threads and is the
// other half of this kernel's F29 reading: 213.0 us for a 1.99 MB row is 9.3 GB/s, and a launch
// that reads its operand eight times at 4,096 threads is latency-bound, not bandwidth-bound. The
// histogram has 256 bins and the suffix scan below is written for 256 lanes, so the workgroup
// widens in whole multiples of that and the first 256 threads keep the scan; `head.topk_block` in
// `rules/dflash_gfx1201.toml` is where the number lives, because it is the launch geometry and
// launch geometry is rules data.
#define DT80_NT 1024
#define DT80_K    80

__device__ __forceinline__ unsigned int dt80_mono(float f) {
  unsigned int b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ unsigned long long dt80_key(float v, int i) {
  return ((unsigned long long)dt80_mono(v) << 32) | (unsigned int)(~(unsigned int)i);
}

typedef float dt80_f32x4 __attribute__((ext_vector_type(4)));

// THE HISTOGRAM'S ATOMIC, AND THE BUCKET IT COLLIDES ON IS NOT THE ONE THE LEDGER NAMED.
//
// F29 recorded "the radix histogram's `atomicAdd` collides on the -inf bucket, because `-inf` is
// most of the row (`_ref_topk_only` fills every entry the rerank did not touch)". Measured on
// prod's own `topk.0.x` (2026-09-21, F30, host, no card): **there is not one -inf in it.** What is
// there is worse and is the same shape:
//
//   rows 0-6 (the drafter's seven live mask positions): 18-19 distinct top bytes over 31,040
//     columns, largest bucket 26-43% of the row -- because the block maxima of one row share an
//     exponent, and the composite's top byte is the top byte of `dt80_mono(value)`.
//   rows 7-15 (the padded rest of the M = 16 tile): **+0.0 everywhere, so ONE bucket holds 100%
//     of the row**, on all eight byte passes, in nine of the sixteen workgroups.
//
// So the collision is real, it is on whichever bucket the row's dominant exponent lands in, and
// -inf was a plausible reading of the source rather than a reading of the data.
//
// THE AGGREGATION IS ONE STEP, not a loop over distinct buckets, and that was measured too. A
// leader-election loop (elect, ballot, mask off, repeat) is optimal for the 100% rows and costs up
// to nineteen ballots per element on the live ones: 213.0 -> 163.2 us, a fifth of what the same
// kernel gives with the version below. This one tests a SINGLE candidate bucket -- whatever lane 0
// holds -- and aggregates it only when more than one lane shares it; every other lane adds its own
// as before. O(1) per element either way, exact for any distribution, and the 100% case collapses
// to one atomic per wave.
__device__ __forceinline__ void dt80_hist_add(int *__restrict__ hist, int bucket, bool active) {
  const int b0 = __shfl(bucket, 0, 32);
  const unsigned long long same = __ballot(active && bucket == b0);
  const int n = (int)__popcll(same);
  if (n > 1) {
    if ((int)(threadIdx.x & 31) == __ffsll(same) - 1) atomicAdd(&hist[b0], n);
    if (active && bucket != b0) atomicAdd(&hist[bucket], 1);
  } else if (active) {
    atomicAdd(&hist[bucket], 1);
  }
}

// PASS 1 -- the threshold: the k-th largest composite in the row, by a radix select over its
// EIGHT bytes, most significant first. Each pass counts only the elements whose higher bytes
// already match the prefix, so the work after the first byte is proportional to what survives.
//
// THREE CHANGES SINCE F29, none of them arithmetic:
//
//   * the histogram's `atomicAdd` is wave-aggregated (above);
//   * the row is read as `float4` -- 16 bytes a thread an iteration instead of 4 -- with every
//     lane of the wave iterating the same number of times, because the aggregation's ballots are
//     whole-wave operations and a lane that left the loop early would corrupt them;
//   * the 256-bin scan that picks the byte is a PARALLEL suffix sum instead of one thread walking
//     255 LDS entries, which it did eight times per row.
//
// AND THE PASSES THAT NEVER SHRINK, WHICH ARE NINE OF THE SIXTEEN WORKGROUPS (F34).
//
// The measurement above says rows 7-15 -- the padding `_pow2_at_least` puts under the M = 16 tile
// -- are `+0.0` everywhere, so one bucket holds 100% of such a row on all eight byte passes and the
// early exit below never fires for them: the chosen bin holds the whole row, never exactly `sh_k`,
// until the LAST byte separates the indices one apiece. Nine of the sixteen workgroups therefore
// scanned the row eight times to learn something that is decided before the first pass ends.
//
// It is decided by the composite, and exactly. If every element of the row carries the SAME 32 bits
// -- which is what a padded row is, and bitwise equality is the right test because `+0.0` and
// `-0.0` are different composites -- then `key(e) = (mono(v) << 32) | ~e` is strictly DECREASING in
// `e`, so the top-k are the first k indices and the k-th largest composite is `key(v, k - 1)`.
// That is the same u64 the eight passes converge to, bit for bit, so `thresh` is unchanged and the
// gather's `vals` / `idx` are unchanged; what disappears is seven passes over a 124 KB row.
//
// The test rides the FIRST pass rather than adding one: byte 7 already reads every element, so each
// thread also compares its four words against `xr[0]` and a single shared flag survives. A live row
// pays one `v_cmp` per element for it, a padded row pays seven eighths less work, and the wall this
// kernel sits at is the SLOWEST of its sixteen workgroups -- which is exactly what the padded rows
// were.
//
// And one EARLY EXIT, which is exact rather than an approximation and is worth four of the eight
// passes on this data. If the bin chosen for a byte holds exactly `sh_k` matching elements, then
// every element matching the prefix so far is in the answer, so the PARTIAL prefix (the resolved
// bytes, zeros below) selects exactly the same set as the full composite would: an element is at
// or above it iff its prefix is strictly higher (there are `k - sh_k` of those) or equal (there are
// `sh_k`). Pass 2 ranks by the FULL key, so the order is unaffected. On this row the top-80 finite
// values resolve after the composite's four VALUE bytes, and the four ~index bytes are skipped.
extern "C" __global__ __launch_bounds__(DT80_NT) void dflash_topk_f32_thresh_k80(
    const float *__restrict__ x,            // [rows, n] fp32 block maxima
    unsigned long long *__restrict__ thresh, // [rows] the k-th largest composite
    int rows, int n, int k) {
  static_assert(DT80_NT >= 256 && DT80_NT % 256 == 0,
                "the 256 histogram bins are zeroed and scanned by the FIRST 256 threads, and every"
                " thread has to reach the barriers inside that scan");
  const int row = (int)blockIdx.x;
  if (row >= rows || k <= 0) return;
  const int tid = (int)threadIdx.x;
  __shared__ int hist[256];
  __shared__ int scan[256];
  __shared__ unsigned long long sh_prefix;
  __shared__ int sh_k;
  __shared__ int sh_bin;
  __shared__ int sh_done;
  // 1 until some thread sees a word that differs from the row's first. Every writer writes the same
  // value, so the race is benign and no atomic is needed.
  __shared__ int sh_same;
  if (tid == 0) {
    sh_prefix = 0ull;
    sh_k = k;
    sh_done = 0;
    sh_same = 1;
  }
  __syncthreads();
  const float *xr = x + (long)row * n;
  const int vt = n >> 2;                                   // whole float4 groups
  const int viters = (vt + DT80_NT - 1) / DT80_NT;

  // Uniform across the workgroup, so the two `if (probe)` below are scalar branches and not an
  // exec mask: the first pass carries the constant-row probe and the other seven do not.
  const unsigned int w0 = __float_as_uint(xr[0]);
  for (int byte = 7; byte >= 0; --byte) {
    const int shift = byte * 8;
    const bool probe = (byte == 7);
    // Everything above this byte must already match the prefix.
    const unsigned long long hi_mask =
        (shift + 8 >= 64) ? 0ull : (~0ull << (shift + 8));
    if (tid < 256) hist[tid] = 0;
    __syncthreads();
    for (int it = 0; it < viters; ++it) {
      const int g = it * DT80_NT + tid;
      const bool ok = g < vt;
      dt80_f32x4 v;
      if (ok) v = *(const dt80_f32x4 *)(xr + (long)g * 4);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        bool act = ok;
        int bucket = 0;
        if (act) {
          const unsigned long long key = dt80_key(v[q], g * 4 + q);
          act = (key & hi_mask) == (sh_prefix & hi_mask);
          bucket = (int)((key >> shift) & 0xffu);
          if (probe && __float_as_uint(v[q]) != w0) sh_same = 0;
        }
        dt80_hist_add(hist, bucket, act);
      }
    }
    for (int e = (vt << 2) + tid; e < (vt << 2) + DT80_NT; e += DT80_NT) {
      bool act = e < n;
      int bucket = 0;
      if (act) {
        const unsigned long long key = dt80_key(xr[e], e);
        act = (key & hi_mask) == (sh_prefix & hi_mask);
        bucket = (int)((key >> shift) & 0xffu);
        if (probe && __float_as_uint(xr[e]) != w0) sh_same = 0;
      }
      dt80_hist_add(hist, bucket, act);
    }
    __syncthreads();

    // THE CONSTANT ROW, decided here and not by seven more passes. Uniform: `sh_same` is read after
    // the barrier the histogram already needed, so every thread takes the same branch. `n >= k` is
    // the guard the argument needs -- with fewer than k elements the k-th largest does not exist and
    // the byte passes' clamp is what defines the answer, so a short row keeps them.
    if (probe && sh_same && n >= k) {
      if (tid == 0) thresh[row] = dt80_key(xr[0], k - 1);
      return;
    }

    // A suffix sum over the 256 bins: `scan[b]` is the population at or above bin `b`, so the bin
    // the select wants is the LARGEST `b` with `scan[b] >= sh_k` -- exactly the bin the serial
    // walk from 255 down used to break on, and the accumulated count above it is `scan[b + 1]`.
    if (tid < 256) scan[tid] = hist[tid];
    __syncthreads();
    for (int off = 1; off < 256; off <<= 1) {
      int add = 0;
      if (tid < 256) add = (tid + off < 256) ? scan[tid + off] : 0;
      __syncthreads();
      if (tid < 256) scan[tid] += add;
      __syncthreads();
    }
    if (tid == 0) sh_bin = 0;                    // fewer than k entries: clamp, never wrap
    __syncthreads();
    if (tid < 256 && scan[tid] >= sh_k && (tid == 255 || scan[tid + 1] < sh_k)) sh_bin = tid;
    __syncthreads();
    if (tid == 0) {
      const int bin = sh_bin;
      const int acc = (bin == 255) ? 0 : scan[bin + 1];
      sh_prefix |= (unsigned long long)(unsigned)bin << shift;
      sh_k -= acc;
      // The partial prefix already selects exactly k -- but only from byte 4, where the composite's
      // four VALUE bytes are resolved and only the ~index half is left. Stopping earlier would
      // leave `thresh`'s HIGH 32 bits partial, and `dflash_topk_f32_gather_k80_notie` reads
      // exactly those 32 bits: a control that stops meaning "the index half of the key" is not a
      // control for the tie order any more.
      if (byte <= 4 && hist[bin] == sh_k) sh_done = 1;
    }
    __syncthreads();
    if (sh_done) break;
  }
  if (tid == 0) thresh[row] = sh_prefix;
}

// PASS 2 -- the gather: every element whose composite is at least the threshold is in the answer,
// and there are exactly k of them because the order is total. Then rank them against each other,
// which is `k * k` comparisons and one thread per entry.
extern "C" __global__ __launch_bounds__(DT80_NT) void dflash_topk_f32_gather_k80(
    const float *__restrict__ x,                   // [rows, n]
    const unsigned long long *__restrict__ thresh, // [rows]
    float *__restrict__ vals,                      // [rows, k] fp32, descending
    int *__restrict__ idx,                         // [rows, k] i32
    int rows, int n, int k) {
  const int row = (int)blockIdx.x;
  if (row >= rows || k <= 0 || k > DT80_K) return;
  const int tid = (int)threadIdx.x;
  __shared__ unsigned long long sel[DT80_K];
  __shared__ int nsel;
  if (tid == 0) nsel = 0;
  __syncthreads();
  const unsigned long long th = thresh[row];
  const float *xr = x + (long)row * n;
  for (int e = tid; e < n; e += DT80_NT) {
    const unsigned long long key = dt80_key(xr[e], e);
    if (key >= th) {
      const int p = atomicAdd(&nsel, 1);
      if (p < DT80_K) sel[p] = key;
    }
  }
  __syncthreads();
  const int m = nsel < k ? nsel : k;
  if (tid < m) {
    const unsigned long long mine = sel[tid];
    int rank = 0;
    for (int j = 0; j < m; ++j) rank += (sel[j] > mine) ? 1 : 0;
    const int e = (int)(~(unsigned int)(mine & 0xffffffffu));
    vals[(long)row * k + rank] = xr[e];
    idx[(long)row * k + rank] = e;
  }
}

// CONTROL: the same selection with the index DROPPED from the key, so equal values are separated
// by whichever thread's atomic won -- the partial order the ledger's tie-order entry is about. It
// must differ from the route on any row that carries a tie and agree on every row that does not,
// which is what makes it a control for the ORDER rather than for the values.
extern "C" __global__ __launch_bounds__(DT80_NT) void dflash_topk_f32_gather_k80_notie(
    const float *__restrict__ x, const unsigned long long *__restrict__ thresh,
    float *__restrict__ vals, int *__restrict__ idx, int rows, int n, int k) {
  const int row = (int)blockIdx.x;
  if (row >= rows || k <= 0 || k > DT80_K) return;
  const int tid = (int)threadIdx.x;
  __shared__ unsigned long long sel[DT80_K];
  __shared__ int nsel;
  if (tid == 0) nsel = 0;
  __syncthreads();
  const unsigned int thv = (unsigned int)(thresh[row] >> 32);
  const float *xr = x + (long)row * n;
  for (int e = tid; e < n; e += DT80_NT) {
    const unsigned int mv = dt80_mono(xr[e]);
    if (mv >= thv) {
      const int p = atomicAdd(&nsel, 1);
      if (p < DT80_K) sel[p] = ((unsigned long long)mv << 32) | (unsigned int)e;
    }
  }
  __syncthreads();
  const int m = nsel < k ? nsel : k;
  if (tid < m) {
    const unsigned long long mine = sel[tid];
    int rank = 0;
    for (int j = 0; j < m; ++j) rank += (sel[j] > mine) ? 1 : 0;
    const int e = (int)(unsigned int)(mine & 0xffffffffu);
    vals[(long)row * k + rank] = xr[e];
    idx[(long)row * k + rank] = e;
  }
}
