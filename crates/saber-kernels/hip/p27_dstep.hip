// P27-W2-2: THE 27B's DEVICE CANDIDATE SEAM (`--engine-device-step cand`).
//
// The DFlash2 drafter at TP=2 selects the best `top_k` (16) candidates of ITS HALF of the
// vocabulary after the backbone (`selector.candidate_topk` -> `sel.vals` f32 + `sel.cand64` i64,
// shard-local ids), and the tail scores the top-16 of the CONCATENATION. Today the host bridges
// the two graphs: wait for the backbone, read both regions, swap them over the rendezvous socket
// (`engine.draftcand`), merge (`device.rs::merge_candidates`: a STABLE descending sort of rank 0's
// 16 then rank 1's 16), seed the merge back, replay the tail, wait again.
//
// This kernel does that bridge on the card, so the tail is enqueued behind the backbone with no
// host wait between them (FN119b's P2P window law, ported):
//
//   p27d_cand_seam  grid (rows), block 32: lane l < top_k pushes this rank's candidate l of row
//                   `r` (value bits + GLOBAL id = local + vocab_base) into the PEER's window as
//                   three TAGGED 64-bit words ((call << 32) | 32 data bits, one aligned store
//                   each, system scope), then a system fence; then it polls THIS rank's window
//                   for the peer's three words of candidate l until each carries `call` (a word
//                   not landed after P27D_SPIN_MAX polls = TIMEOUT in the error word: fail
//                   loud). No separate flag: FN114b measured a flag stored after its data landing
//                   BEFORE it over the link (h1 of this round reproduced it: the first proposal's
//                   peer shard read stale). Then the 32 entries are merged by
//                   the host's own law: entry i (rank 0's 16 first, then rank 1's) lands at
//                   #{j : v_j > v_i} + #{j < i : v_j == v_i} -- exactly where a stable descending
//                   sort puts it; positions < top_k are written over `sel.vals` / `sel.cand64`.
//                   A NaN value makes the host's comparator inconsistent (`partial_cmp` -> Equal),
//                   so a row holding one sets NAN in the error word (the host refuses the step).
//
// SLOTS: the window holds two slots (`call & 1`), each `rows_max` rows of P27D_RS 64-bit words. Both ranks
// call in lockstep with the same `call` (a per-rank host counter, both advanced once a proposal).
// A slot is rewritten at `call + 2`, which this rank issues only after its `call + 1` launch saw
// the peer's `call + 1` flag -- pushed by the peer's `call + 1` launch, which follows the peer's
// `call` launch (the one that read the slot) in the peer's stream order. A word from an older call
// never carries `call`, so no slot is zeroed.
//
// SCRATCH (one region a rank, read back by the host only under the shadow digest): this rank's
// shard as the host read it (values, then LOCAL ids) and the peer's shard as the socket carried it
// (values, then GLOBAL ids) and the merge as the host seeded it (values, then GLOBAL ids), so the
// shadow's `cand.mine` / `cand.theirs` / `cand.merged` stages are the same bytes; the error word is
// at word 0.
//
// No float arithmetic: comparisons only (built with -ffp-contract=off like every exact law).

#include <hip/hip_runtime.h>
#include <stdint.h>

#define P27D_K 16
#define P27D_RS 64
#define P27D_SPIN_MAX 400000000ULL
#define P27D_ERR_TIMEOUT 1u
#define P27D_ERR_NAN 2u
#define P27D_ERR_GEOM 4u

extern "C" __global__ __launch_bounds__(32) void p27d_cand_seam(
    float* vals, long long* ids, unsigned long long* win_mine, unsigned long long* win_peer,
    uint32_t* scratch, long long vocab_base, int top_k, int rank, int call, int rows,
    int rows_max) {
  __shared__ float s_v[2 * P27D_K];
  __shared__ long long s_i[2 * P27D_K];
  __shared__ uint32_t s_to;
  const int r = blockIdx.x;
  const int l = threadIdx.x;
  uint32_t* err = scratch;
  if (top_k != P27D_K || r >= rows || rows > rows_max) {
    if (l == 0) atomicOr(err, P27D_ERR_GEOM);
    return;
  }
  if (l == 0) s_to = 0u;
  const size_t n = (size_t)rows * P27D_K;
  const size_t slot = (size_t)((uint32_t)call & 1u) * (size_t)rows_max * P27D_RS;
  unsigned long long* dst = win_peer + slot + (size_t)r * P27D_RS;
  unsigned long long* src = win_mine + slot + (size_t)r * P27D_RS;
  const unsigned long long tag = (unsigned long long)(uint32_t)call << 32;
  // the scratch's six arrays (byte offsets 8 .. ): mine vals | mine local ids | theirs vals |
  // theirs global ids | merged vals | merged global ids
  float* sc_mv = (float*)(scratch + 2);
  long long* sc_mi = (long long*)(scratch + 2 + n);
  float* sc_tv = (float*)(scratch + 2 + 3 * n);
  long long* sc_ti = (long long*)(scratch + 2 + 4 * n);
  float* sc_gv = (float*)(scratch + 2 + 6 * n);
  long long* sc_gi = (long long*)(scratch + 2 + 7 * n);
  __syncthreads();

  float mv = 0.0f;
  long long mi = 0;
  if (l < P27D_K) {
    mv = vals[(size_t)r * P27D_K + l];
    const long long local = ids[(size_t)r * P27D_K + l];
    mi = local + vocab_base;
    sc_mv[(size_t)r * P27D_K + l] = mv;
    sc_mi[(size_t)r * P27D_K + l] = local;
    // EVERY WORD CARRIES ITS OWN TAG (FN114b's law: a flag stored after the data can still land
    // before it over the link, so no word is trusted on another word's word): one aligned 64-bit
    // store = (call << 32) | the 32 data bits, landed whole or not at all.
    const unsigned long long w0 = tag | (unsigned long long)__float_as_uint(mv);
    const unsigned long long w1 = tag | ((unsigned long long)mi & 0xffffffffull);
    const unsigned long long w2 = tag | ((unsigned long long)mi >> 32);
    __hip_atomic_store(&dst[l], w0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    __hip_atomic_store(&dst[P27D_K + 2 * l], w1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    __hip_atomic_store(&dst[P27D_K + 2 * l + 1], w2, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
  // a system-scope write-back: a plain store to the peer's memory sits in L2 without it (FN114b)
  __threadfence_system();
  float pv = 0.0f;
  long long pi = 0;
  if (l < P27D_K) {
    unsigned long long w[3] = {0ull, 0ull, 0ull};
    const int at[3] = {l, P27D_K + 2 * l, P27D_K + 2 * l + 1};
    bool ok = true;
    for (int q = 0; q < 3; ++q) {
      unsigned long long spins = 0;
      unsigned long long v = 0;
      do {
        v = __hip_atomic_load(&src[at[q]], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        ++spins;
      } while ((v & 0xffffffff00000000ull) != tag && spins < P27D_SPIN_MAX);
      if ((v & 0xffffffff00000000ull) != tag) ok = false;
      w[q] = v;
    }
    if (!ok) atomicOr(&s_to, 1u);
    pv = __uint_as_float((uint32_t)w[0]);
    pi = (long long)(((w[2] & 0xffffffffull) << 32) | (w[1] & 0xffffffffull));
    sc_tv[(size_t)r * P27D_K + l] = pv;
    sc_ti[(size_t)r * P27D_K + l] = pi;
    // rank 0's shard is the merge's first half on both ranks
    const int a = (rank == 0) ? l : P27D_K + l;
    const int b = (rank == 0) ? P27D_K + l : l;
    s_v[a] = mv;
    s_i[a] = mi;
    s_v[b] = pv;
    s_i[b] = pi;
  }
  __syncthreads();
  if (s_to != 0u) {
    if (l == 0) atomicOr(err, P27D_ERR_TIMEOUT);
    return;
  }
  // lane i owns entry i of the 32
  const float v = s_v[l];
  int pos = 0;
  bool nan = (v != v);
  for (int j = 0; j < 2 * P27D_K; ++j) {
    const float o = s_v[j];
    nan = nan || (o != o);
    pos += (o > v) ? 1 : 0;
    pos += (j < l && o == v) ? 1 : 0;
  }
  if (nan) {
    if (l == 0) atomicOr(err, P27D_ERR_NAN);
    return;
  }
  if (pos < P27D_K) {
    vals[(size_t)r * P27D_K + pos] = v;
    ids[(size_t)r * P27D_K + pos] = s_i[l];
    sc_gv[(size_t)r * P27D_K + pos] = v;
    sc_gi[(size_t)r * P27D_K + pos] = s_i[l];
  }
}
