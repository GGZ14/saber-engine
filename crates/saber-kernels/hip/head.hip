// The forward pass's ends: the embedding lookup that starts it and the two head kernels that
// finish it. Three small, memory-bound kernels that prod does not have as kernels at all --
// The reference server's `VocabParallelEmbedding` is a torch index_select, the row selection before `lm_head` is
// `index_select` inside `LogitsProcessor`, and greedy sampling is `torch.argmax` -- so the
// reference for all three is torch, and the gate is byte equality against it.
//
// They are together in one code object because they are the same KIND of work (a strided copy or
// a row reduction over bf16) and because three separate blobs would be three `hipModuleLoad`s for
// four launches per forward.
//
// Every kernel here is ONE WORKGROUP PER OUTPUT ROW. That is not a tile choice: the rows are
// independent, the work per row is a copy of `hidden` (5120) or a reduction over `vocab` (248320
// at TP=1), and a grid of `rows` workgroups is what lets the scheduler fill the machine at
// prefill and still be a single launch at M=1.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// 16 bytes = 8 bf16. Every width this model uses (5120, 8192, 3072, 248320) is a multiple of 8,
// but the tail loop is still here: a width that is not is a refusal in `head::resolve`, and a
// kernel that silently dropped the tail would be worse than one that is merely slower.
#define HEAD_VEC 8

/// Vocab-parallel embedding lookup: `out[t] = table[ids[t] - vocab_begin]`, or **zero** when this
/// rank does not own the row.
///
/// The zero is the vocab-parallel contract, not a guard against bad input: at TP > 1 each rank
/// holds `vocab/tp` rows, writes zeros for the ids it does not own, and the all-reduce that
/// follows sums the one non-zero contribution per token. At TP = 1 the range covers everything
/// and the branch is never taken.
extern "C" __global__ void saber_embed_bf16(const __bf16 *__restrict__ table,
                                            const int *__restrict__ ids,
                                            __bf16 *__restrict__ out, int hidden,
                                            int vocab_begin, int vocab_end) {
  const int row = blockIdx.x;
  const int id = ids[row];
  __bf16 *o = out + (long)row * hidden;
  const int tid = threadIdx.x, nt = blockDim.x;
  const int nv = hidden / HEAD_VEC;
  if (id < vocab_begin || id >= vocab_end) {
    // Not this rank's row: write zeros, and never read the table.
    for (int g = tid; g < nv; g += nt) ((uint4 *)o)[g] = make_uint4(0u, 0u, 0u, 0u);
    for (int j = nv * HEAD_VEC + tid; j < hidden; j += nt) o[j] = (__bf16)0.0f;
    return;
  }
  const __bf16 *src = table + (long)(id - vocab_begin) * hidden;
  for (int g = tid; g < nv; g += nt) ((uint4 *)o)[g] = ((const uint4 *)src)[g];
  for (int j = nv * HEAD_VEC + tid; j < hidden; j += nt) o[j] = src[j];
}

/// Select rows: `out[r] = src[idx[r]]`, `cols` wide, `src` rows `src_stride` apart.
///
/// This is the selection the reference server's `LogitsProcessor` makes BEFORE `lm_head`: only the last position
/// of each sequence needs logits, so a 2048-token prefill does one `[1, 5120] x [5120, 248320]`
/// GEMM instead of 2048 of them. Doing it after the head instead would compute 2048 x 248320
/// logits and throw 99.95% of them away -- 1 GiB of bf16 written to be ignored.
extern "C" __global__ void saber_row_gather_bf16(const __bf16 *__restrict__ src,
                                                 const int *__restrict__ idx,
                                                 __bf16 *__restrict__ out, int cols,
                                                 long src_stride) {
  const int r = blockIdx.x;
  const __bf16 *s = src + (long)idx[r] * src_stride;
  __bf16 *o = out + (long)r * cols;
  const int tid = threadIdx.x, nt = blockDim.x;
  const int nv = cols / HEAD_VEC;
  for (int g = tid; g < nv; g += nt) ((uint4 *)o)[g] = ((const uint4 *)s)[g];
  for (int j = nv * HEAD_VEC + tid; j < cols; j += nt) o[j] = s[j];
}

/// Greedy sampling: the index of the largest logit in each row, **ties to the lowest index**.
///
/// The tie rule is the whole reason this is a kernel and not a `reduce(max)`: `torch.argmax`
/// returns the first maximum, and a sampler that returned a different one on a tie would produce
/// a different token from the reference on inputs where nothing is wrong. The comparison below is
/// a total order on `(value descending, index ascending)` and the tree reduction applies exactly
/// it at every step, so the answer does not depend on how the work was split.
///
/// NaN never wins: every comparison against a NaN is false. A row that is ALL NaN therefore
/// yields `cols` (no index won), which is out of range on purpose -- a forward pass that produced
/// NaN logits is broken, and a sampler that quietly returned token 0 would hide it.
#define ARGMAX_T 256
extern "C" __global__ void saber_argmax_bf16(const __bf16 *__restrict__ logits,
                                             int *__restrict__ out, int cols, long stride) {
  __shared__ float sv[ARGMAX_T];
  __shared__ int si[ARGMAX_T];
  const int row = blockIdx.x;
  const __bf16 *p = logits + (long)row * stride;
  const int tid = threadIdx.x;

  float best = -__builtin_inff();
  int bi = cols;  // "nothing won yet"
  for (int j = tid; j < cols; j += ARGMAX_T) {
    const float v = (float)p[j];
    if (v > best || (v == best && j < bi)) {
      best = v;
      bi = j;
    }
  }
  sv[tid] = best;
  si[tid] = bi;
  __syncthreads();
  for (int s = ARGMAX_T / 2; s > 0; s >>= 1) {
    if (tid < s) {
      const float ov = sv[tid + s];
      const int oi = si[tid + s];
      if (ov > sv[tid] || (ov == sv[tid] && oi < si[tid])) {
        sv[tid] = ov;
        si[tid] = oi;
      }
    }
    __syncthreads();
  }
  if (tid == 0) out[row] = si[0];
}

// ---------------------------------------------------------------------------------------------
// The sampler's seam: the K largest logits of a row, the row's two log-sum-exps, and the raw
// logits at the ids the penalties can move.
//
// WHY A KERNEL AND NOT A READBACK. Everything the reference server's sampler does to a row is a function of four
// things (`crates/saber-serve/src/sampler.rs`, `DeviceRow`): the K largest values and their ids in
// one total order, the row's log-sum-exp at T = 1 (logprobs are the RAW logits' log_softmax), the
// row's log-sum-exp at its own temperature (only `top_p` needs it -- its cumulative sum runs over
// the WHOLE distribution), and the raw logits at the penalty ids (the only ids penalties move).
// Reading the dense row back instead is 248,320 bf16 = 485 KiB per sampled row per step; this
// writes 520 bytes plus the penalty gather.
//
// THE TOTAL ORDER IS `saber_argmax_bf16`'s: value descending, id ascending. `top_ids[0]` is
// therefore the argmax kernel's own answer on the same row, which is what the gate asserts.
//
// THE SELECTION IS A RADIX SELECT ON THE bf16 BIT PATTERN, so it is exact and independent of how
// the work was split. A bf16 maps to a 16-bit key with the same total order (`devrow_key`), so the
// K-th largest value is found by two 256-bin histogram passes -- the key's high byte, then its low
// byte inside the one bucket that straddles K -- and the threshold that comes out is an exact bf16
// value rather than an estimate. A comparison ladder would need one pass per bit; a 65,536-bin
// histogram would need 256 KiB of LDS.
//
// NaN NEVER WINS, the same rule the argmax reduces with: a NaN element is skipped entirely (it
// enters neither histogram nor either sum), and a row with no finite element writes `cols` for
// every id -- the argmax kernel's own "nothing won" answer -- and -inf for both log-sum-exps.
//
// A GREEDY ROW IS SKIPPED. `temps[row] <= 0` means the host will take this row's argmax and
// nothing else, so the kernel writes `-1` into `top_ids[row * K]` and returns. That is what keeps
// a temperature-0 step what it was before this kernel existed: the node is in the captured graph
// either way, and at temperature 0 it costs one dispatch and an early-out.
#define DEVROW_T 256
#define DEVROW_K 64
#define DEVROW_WAVE 32
#define DEVROW_WAVES (DEVROW_T / DEVROW_WAVE)
#define DEVROW_E 8  // consecutive ids a thread owns in the ordered tie scan (P27-W2-1)

/// bf16 bits -> a 16-bit key with the same total order as the float value.
__device__ __forceinline__ unsigned devrow_key(unsigned short b) {
  return (b & 0x8000u) ? (unsigned)(unsigned short)(~b) : (unsigned)(unsigned short)(b | 0x8000u);
}

__device__ __forceinline__ bool devrow_is_nan(unsigned short b) {
  return (b & 0x7F80u) == 0x7F80u && (b & 0x007Fu) != 0u;
}

__device__ __forceinline__ unsigned short devrow_bits(const __bf16 *p) {
  return *(const unsigned short *)p;
}

extern "C" __global__ void saber_device_row_bf16(const __bf16 *__restrict__ logits,
                                                 const float *__restrict__ temps,
                                                 const int *__restrict__ pen_ids,
                                                 float *__restrict__ top_values,
                                                 int *__restrict__ top_ids,
                                                 float *__restrict__ lse,
                                                 float *__restrict__ pen_values, int cols,
                                                 long stride, int pen_max) {
  // Per-WAVE histograms, not one per block: at 248,320 logits from one distribution most of the
  // mass lands in a handful of exponent buckets, and one shared counter per bucket serialises
  // every lane that hits it. Eight copies cost 8 KiB of LDS and cut that by the wave count.
  __shared__ unsigned hist[DEVROW_WAVES][256];
  __shared__ float red[DEVROW_T];
  __shared__ float cand_v[DEVROW_K];
  __shared__ int cand_i[DEVROW_K];
  __shared__ int wave_ties[DEVROW_WAVES];
  __shared__ int s_bucket, s_above, s_thr, s_cnt_gt, s_take, s_ties, s_n_gt, s_n_tie;
  __shared__ float s_max;

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int wave = tid / DEVROW_WAVE;
  const float temperature = temps[row];
  float *tv = top_values + (long)row * DEVROW_K;
  int *ti = top_ids + (long)row * DEVROW_K;

  if (!(temperature > 0.0f)) {
    // A greedy row: the argmax already answered it. `-1` is not a token and not `cols`, so a host
    // that read this slot anyway fails loudly rather than sampling a plausible id.
    if (tid == 0) ti[0] = -1;
    return;
  }
  const __bf16 *p = logits + (long)row * stride;

  // ---- the penalty gather --------------------------------------------------------------------
  // FIRST, because it depends on nothing else: the host knows which ids the penalties can move --
  // the prompt's and the output's tokens -- and this is their raw logits, whatever those are. `-1`
  // pads the list; its slot is written 0 and the host reads only as many entries as it asked for.
  //
  // It used to sit at the END, after the selection, and the all-NaN row of the gate's edge case
  // caught that: the "nothing won" path returned before it and wrote zeros where the row's own
  // (NaN) logits belonged, six bytes of 4,352. A gather is a gather; it does not have an opinion
  // about whether the row has a winner.
  for (int i = tid; i < pen_max; i += DEVROW_T) {
    const int id = pen_ids[(long)row * pen_max + i];
    pen_values[(long)row * pen_max + i] = (id >= 0 && id < cols) ? (float)p[id] : 0.0f;
  }

  // ---- pass 1: the key's high byte, and the row's maximum ------------------------------------
  for (int i = tid; i < 256 * DEVROW_WAVES; i += DEVROW_T) ((unsigned *)hist)[i] = 0u;
  __syncthreads();
  float mx = -__builtin_inff();
  for (int j = tid; j < cols; j += DEVROW_T) {
    const unsigned short b = devrow_bits(p + j);
    if (devrow_is_nan(b)) continue;
    atomicAdd(&hist[wave][devrow_key(b) >> 8], 1u);
    const float v = (float)p[j];
    if (v > mx) mx = v;
  }
  red[tid] = mx;
  __syncthreads();
  for (int s = DEVROW_T / 2; s > 0; s >>= 1) {
    if (tid < s && red[tid + s] > red[tid]) red[tid] = red[tid + s];
    __syncthreads();
  }
  if (tid == 0) s_max = red[0];
  // Fold the per-wave histograms into wave 0's copy: every copy is read before any is written.
  unsigned folded = 0u;
  if (tid < 256) {
    for (int w = 0; w < DEVROW_WAVES; ++w) folded += hist[w][tid];
  }
  __syncthreads();
  if (tid < 256) hist[0][tid] = folded;
  __syncthreads();
  if (tid == 0) {
    // The bucket the K-th largest key lies in, and how many keys sit above that bucket. When the
    // row holds fewer than K finite values the scan runs out and keeps the LOWEST non-empty
    // bucket, which is the "take everything" answer.
    long cum = 0;
    int bucket = -1;
    long above = 0;
    for (int b = 255; b >= 0; --b) {
      const unsigned n = hist[0][b];
      if (n == 0u) continue;
      if (cum + (long)n >= (long)DEVROW_K) {
        bucket = b;
        above = cum;
        break;
      }
      cum += (long)n;
      bucket = b;
      above = cum - (long)n;
    }
    s_bucket = bucket;
    s_above = (int)above;
  }
  __syncthreads();

  if (s_bucket < 0) {
    // Every element was NaN. `cols` is what `saber_argmax_bf16` writes when nothing won.
    for (int i = tid; i < DEVROW_K; i += DEVROW_T) {
      tv[i] = -__builtin_inff();
      ti[i] = cols;
    }
    if (tid == 0) {
      lse[(long)row * 2] = -__builtin_inff();
      lse[(long)row * 2 + 1] = -__builtin_inff();
    }
    return;
  }

  // ---- pass 2: the key's low byte inside that bucket, and both log-sum-exps -------------------
  for (int i = tid; i < 256 * DEVROW_WAVES; i += DEVROW_T) ((unsigned *)hist)[i] = 0u;
  __syncthreads();
  const float mxv = s_max;
  const float mxt = mxv / temperature;
  const int bucket = s_bucket;
  float s1 = 0.0f, st = 0.0f;
  for (int j = tid; j < cols; j += DEVROW_T) {
    const unsigned short b = devrow_bits(p + j);
    if (devrow_is_nan(b)) continue;
    const unsigned key = devrow_key(b);
    if ((int)(key >> 8) == bucket) atomicAdd(&hist[wave][key & 0xFFu], 1u);
    const float v = (float)p[j];
    s1 += __builtin_expf(v - mxv);
    st += __builtin_expf(v / temperature - mxt);
  }
  // Two tree reductions over the same array, in the order the host reference folds them.
  red[tid] = s1;
  __syncthreads();
  for (int s = DEVROW_T / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  if (tid == 0) lse[(long)row * 2] = mxv + __builtin_logf(red[0]);
  __syncthreads();
  red[tid] = st;
  __syncthreads();
  for (int s = DEVROW_T / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  if (tid == 0) lse[(long)row * 2 + 1] = mxt + __builtin_logf(red[0]);
  folded = 0u;
  if (tid < 256) {
    for (int w = 0; w < DEVROW_WAVES; ++w) folded += hist[w][tid];
  }
  __syncthreads();
  if (tid < 256) hist[0][tid] = folded;
  __syncthreads();
  if (tid == 0) {
    long cum = 0;
    int low = 0;
    long above = 0;
    const long need = (long)DEVROW_K - (long)s_above;
    for (int b = 255; b >= 0; --b) {
      const unsigned n = hist[0][b];
      if (n == 0u) continue;
      if (cum + (long)n >= need) {
        low = b;
        above = cum;
        break;
      }
      cum += (long)n;
      low = b;
      above = cum - (long)n;
    }
    s_thr = (bucket << 8) | low;
    s_cnt_gt = s_above + (int)above;
    s_ties = (int)hist[0][low];
    const int want = DEVROW_K - s_cnt_gt;
    s_take = want < s_ties ? want : s_ties;
    s_n_gt = 0;
    s_n_tie = 0;
  }
  __syncthreads();

  // ---- pass 3: collect the candidates --------------------------------------------------------
  const unsigned thr = (unsigned)s_thr;
  const int cnt_gt = s_cnt_gt;
  const int take = s_take;
  for (int j = tid; j < cols; j += DEVROW_T) {
    const unsigned short b = devrow_bits(p + j);
    if (devrow_is_nan(b)) continue;
    const unsigned key = devrow_key(b);
    if (key > thr) {
      const int slot = atomicAdd(&s_n_gt, 1);
      if (slot < DEVROW_K) {
        cand_v[slot] = (float)p[j];
        cand_i[slot] = j;
      }
    }
  }
  __syncthreads();
  if (take == s_ties) {
    // Every element at the threshold is kept, so which of them lands in which slot cannot change
    // the answer and the collection is unordered.
    for (int j = tid; j < cols; j += DEVROW_T) {
      const unsigned short b = devrow_bits(p + j);
      if (devrow_is_nan(b)) continue;
      if (devrow_key(b) == thr) {
        const int slot = cnt_gt + atomicAdd(&s_n_tie, 1);
        if (slot < DEVROW_K) {
          cand_v[slot] = (float)p[j];
          cand_i[slot] = j;
        }
      }
    }
  } else {
    // The threshold value has more members than the seam has room for, so the tie is broken by
    // LOWEST ID -- the argmax's own rule. P27-W2-1: an ORDERED BLOCK SCAN, not a walk on thread 0
    // (P27-14 measured that walk at ~1.7 ms of rank 1's sampled step: rank 1's shard holds the
    // vocabulary's tail and its tied ids sit at ~51 % of the shard). The row is cut into tiles of
    // DEVROW_T x DEVROW_E ids; thread t owns the DEVROW_E CONSECUTIVE ids at base + t * DEVROW_E,
    // so (tile, thread, element) is exactly id order. Each thread counts its tied ids, a wave
    // inclusive scan + the per-wave totals give its exclusive rank among the tile's ties, and the
    // tie of global rank r < take lands in slot cnt_gt + r -- the same SET of ids the serial walk
    // kept (the lowest `take` tied ids) in the same slots, so the bytes cannot move. Two barriers a
    // tile, and the scan stops at the first tile that completes `take`.
    const int lane = tid % DEVROW_WAVE;
    int done = 0;  // tied ids in every tile before this one (block-uniform)
    for (int base = 0; base < cols && done < take; base += DEVROW_T * DEVROW_E) {
      const int j0 = base + tid * DEVROW_E;
      unsigned m = 0u;
#pragma unroll
      for (int e = 0; e < DEVROW_E; ++e) {
        const int j = j0 + e;
        if (j < cols) {
          const unsigned short b = devrow_bits(p + j);
          if (!devrow_is_nan(b) && devrow_key(b) == thr) m |= 1u << e;
        }
      }
      const int c = __builtin_popcount(m);
      int x = c;  // inclusive scan of c across the wave
#pragma unroll
      for (int off = 1; off < DEVROW_WAVE; off <<= 1) {
        const int y = __shfl_up(x, off, DEVROW_WAVE);
        if (lane >= off) x += y;
      }
      if (lane == DEVROW_WAVE - 1) wave_ties[wave] = x;
      __syncthreads();
      int rank = done + x - c;
      int total = done;
#pragma unroll
      for (int w = 0; w < DEVROW_WAVES; ++w) {
        const int n = wave_ties[w];
        if (w < wave) rank += n;
        total += n;
      }
      while (m != 0u && rank < take) {
        const int e = __builtin_ctz(m);
        m &= m - 1u;
        const int slot = cnt_gt + rank;
        if (slot < DEVROW_K) {
          cand_v[slot] = (float)p[j0 + e];
          cand_i[slot] = j0 + e;
        }
        ++rank;
      }
      done = total;
      __syncthreads();  // wave_ties is rewritten by the next tile
    }
    // The serial walk stopped at exactly `take`; the order pass reads the count from here.
    if (tid == 0) s_n_tie = done < take ? done : take;
  }
  __syncthreads();

  // ---- the order: value descending, id ascending ----------------------------------------------
  if (tid == 0) {
    const int got = (cnt_gt < DEVROW_K ? cnt_gt : DEVROW_K) + s_n_tie;
    const int have = got < DEVROW_K ? got : DEVROW_K;
    for (int i = 1; i < have; ++i) {
      const float v = cand_v[i];
      const int id = cand_i[i];
      int k = i - 1;
      while (k >= 0 && (cand_v[k] < v || (cand_v[k] == v && cand_i[k] > id))) {
        cand_v[k + 1] = cand_v[k];
        cand_i[k + 1] = cand_i[k];
        --k;
      }
      cand_v[k + 1] = v;
      cand_i[k + 1] = id;
    }
    for (int i = have; i < DEVROW_K; ++i) {
      cand_v[i] = -__builtin_inff();
      cand_i[i] = cols;
    }
  }
  __syncthreads();
  for (int i = tid; i < DEVROW_K; i += DEVROW_T) {
    tv[i] = cand_v[i];
    ti[i] = cand_i[i];
  }
}

// ---------------------------------------------------------------------------------------------
// The DFlash2 AUX TAP, and the one width seam the drafter's own chain could not close.
//
// Two more strided copies, here rather than in `dflash` because they are the head plugin's own
// kind of work -- one workgroup per row, a copy and nothing else -- and because putting them in
// `dflash` would make every engine that taps a residual stream load the drafter's code object.
//
// WHY THE TAP NEEDS ITS OWN KERNEL AND `saber_row_gather_bf16` WILL NOT DO. The gather writes its
// output rows CONTIGUOUSLY (`out + r * cols`): it selects rows out of a wide source into a packed
// destination, which is what the row selection before `lm_head` wants. The aux tap is the other
// shape of the same idea -- one whole slab of `rows x hidden` copied into a COLUMN BLOCK of a
// `rows x 5 x hidden` concatenation -- so its destination rows are `5 * hidden` apart while its
// source rows are `hidden` apart, and there is no index list at all. A stride the kernel cannot
// be told is a stride the caller has to fake with five separate destination buffers and a sixth
// launch to concatenate them.
//
// The tap must copy AT the tap point, not after the step: the body plan runs with slot reuse ON,
// so `l5.resid.a` and `l61.resid.a` are the same arena buffer and at the end of a step it holds
// layer 63's value five times over (`docs/EXEC.md` section 6, and `probe_step` reads its dump
// points the same way for the same reason).
extern "C" __global__ void saber_row_copy_bf16(const __bf16 *__restrict__ src,
                                               __bf16 *__restrict__ out, int cols,
                                               long src_stride, long dst_stride) {
  const int r = blockIdx.x;
  const __bf16 *s = src + (long)r * src_stride;
  __bf16 *o = out + (long)r * dst_stride;
  const int tid = threadIdx.x, nt = blockDim.x;
  const int nv = cols / HEAD_VEC;
  for (int g = tid; g < nv; g += nt) ((uint4 *)o)[g] = ((const uint4 *)s)[g];
  for (int j = nv * HEAD_VEC + tid; j < cols; j += nt) o[j] = s[j];
}

/// `out[r][j] = (long)src[r][j]`: a sign-extending widen, rows `src_stride` / `dst_stride` apart.
///
/// THE SEAM THIS CLOSES, stated because it is a deviation and not a kernel prod runs. The DFlash2
/// selector's `dflash_selector_score_r256_k16` and `dflash_selector_walk_greedy` both take their
/// candidate table as `const long *` -- prod's candidates are `torch.topk`'s int64, and ws-f24's
/// whole finding was that declaring that slot narrower reads the right addresses and the wrong
/// bytes. Saber's own producer, `dflash_topk_bf16_k16`, writes i32. In a STAGED replay the two
/// never meet (the capture supplies prod's own i64 table); in a CHAIN they do, and
/// `draft_proposal.rs` bridges them with a host round trip, which a captured graph cannot have.
///
/// So the engine's captured proposal carries ONE launch prod does not make, declared and counted
/// as its own op rather than hidden inside another. It disappears the day the selection writes
/// i64 -- which is the real fix and belongs to that kernel's owner (ws-f27's open item).
extern "C" __global__ void saber_widen_i32_i64(const int *__restrict__ src,
                                               long *__restrict__ out, int cols,
                                               long src_stride, long dst_stride) {
  const int r = blockIdx.x;
  const int *s = src + (long)r * src_stride;
  long *o = out + (long)r * dst_stride;
  const int tid = threadIdx.x, nt = blockDim.x;
  for (int j = tid; j < cols; j += nt) o[j] = (long)s[j];
}

// ---- the bf16 head's final norm (workstream N2) -------------------------------------------------
//
// `model.norm` when `lm_head` is a bf16 GEMM rather than a W4A8 one: the same residual add and the
// same Gemma RMS norm `saber_add_rms_quant` computes -- the sum kept in fp32 (NOT rounded before
// the norm), `rsqrt(mean(x^2) + eps)`, `bf16(x * inv * (w + 1))` rounded once -- and then the bf16
// value itself is the output instead of an e4m3 quant of it. That bf16 row is exactly what prod's
// `lm_head` reads under `SABER_NVFP4_LMHEAD=bf16`: a bf16 activation into a bf16 matrix.
// `res_out` is written for symmetry with the fused op (the head's residual is consumed by nothing).
//
// One workgroup per row, `blockDim.x` threads (a multiple of 32, at most 1024), K a multiple of 8.
// The reduction tree is this kernel's own; it is not claimed bit-identical to `add_rms_quant`'s
// pre-quant value at every row (a different partial-sum order can move one rsqrt ULP).
#define ARB_MAXG 4  // K <= 4 * 8 * blockDim.x: 8192 at 256 threads (hidden is 5120)
extern "C" __global__ __launch_bounds__(256) void saber_add_rms_bf16(
    const __bf16 *__restrict__ y, const __bf16 *__restrict__ res, const __bf16 *__restrict__ w,
    __bf16 *__restrict__ out, __bf16 *__restrict__ res_out, int k, float eps) {
  const long row = blockIdx.x;
  const int tid = threadIdx.x, nt = blockDim.x;
  y += row * k;
  res += row * k;
  out += row * k;
  res_out += row * k;
  const int kv = k / HEAD_VEC;
  union V8 { uint4 u; __bf16 h[8]; };
  float sv[ARB_MAXG][8];
  float ssq = 0.f;
  // Unrolled with a constant group index, so `sv` stays in registers (a dynamically indexed
  // array spills to scratch).
#pragma unroll
  for (int ng = 0; ng < ARB_MAXG; ++ng) {
    const int g = tid + ng * nt;
    if (g >= kv) break;
    V8 vy, vr, vo;
    vy.u = ((const uint4 *)y)[g];
    vr.u = ((const uint4 *)res)[g];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float v = (float)vy.h[j] + (float)vr.h[j];
      sv[ng][j] = v;
      vo.h[j] = (__bf16)v;
      ssq += v * v;
    }
    ((uint4 *)res_out)[g] = vo.u;
  }
  __shared__ float lds[32];
  for (int o = 16; o; o >>= 1) ssq += __shfl_down(ssq, o);
  if ((tid & 31) == 0) lds[tid >> 5] = ssq;
  __syncthreads();
  if (tid == 0) {
    float v = 0.f;
    for (int i = 0; i < nt / 32; ++i) v += lds[i];
    lds[0] = v;
  }
  __syncthreads();
  const float inv = rsqrtf(lds[0] / (float)k + eps);
#pragma unroll
  for (int mg = 0; mg < ARB_MAXG; ++mg) {
    const int g = tid + mg * nt;
    if (g >= kv) break;
    V8 vw, vo;
    vw.u = ((const uint4 *)w)[g];
#pragma unroll
    for (int j = 0; j < 8; ++j) vo.h[j] = (__bf16)(sv[mg][j] * inv * ((float)vw.h[j] + 1.f));
    ((uint4 *)out)[g] = vo.u;
  }
}
