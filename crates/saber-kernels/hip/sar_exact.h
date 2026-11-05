// Saber two-rank all-reduce kernels, CLEANROOM-2 (CR2-4c): the exact one-shot (spec 2.2 / 3.3 / 3.5 / 3.9).
//
// One body for every exact form: the element type (bf16 / f16 / f32), the parity law (each block's own counter,
// or the global-parity line that advances all 96 counters once a call) and the race-check level (none, checked,
// checked with the receiver's wait removed = the control). A message of n16 16-byte words: block b owns the
// contiguous run [b w, min((b + 1) w, n16)), w = ceil(n16 / nb); it pushes its run raw into the peer's slot
// (s & 1), drains, hands off through its flag word, then sums its run against what the peer pushed into its
// own slot: out = conv(f32(in) + f32(peer)) lane by lane. `in` is re-read for the sum (in == out is allowed).
#pragma once
#include "sar_prims.h"
#include "ar_race_check.h"

namespace sar {

enum Parity { kPerBlock = 0, kGlobal = 1 };
enum Check { kNoCheck = 0, kChecked = 1, kCheckedNoWait = 2 };

constexpr int kGpCounters = 96;   // the global-parity forms' counters / flags (shared with the rotated gp form)
#define SAR_CHK_D_GP_PARITY 14    // receiver's s != the sender's s (check row piece 1)

// the contiguous run of 16-byte words a block owns
struct Run {
  int start, end;
};
__device__ __forceinline__ Run block_run(int n16) {
  const int nb = (int)gridDim.x;
  const int w = (n16 + nb - 1) / nb;
  const int start = (int)blockIdx.x * w;
  return Run{start, min(start + w, n16)};
}

template <int E, int PAR, int CHK>
__device__ __forceinline__ void exact_oneshot(const int4* in, int4* peer_scratch, int4* my_scratch, int4* out,
                                              u32* peer_flags, u32* my_flags, u32* seq, int n16, int stride16,
                                              int drain, int acq) {
  __shared__ u32 bcast;
  const int b = (int)blockIdx.x;
  const int nt = (int)blockDim.x;
  const int tid = (int)threadIdx.x;
  const Run run = block_run(n16);
  const u32 s = take_seq(seq, b, &bcast);
  if constexpr (PAR == kGlobal) advance_idle(seq, b, (int)gridDim.x, kGpCounters);
  const long so = (long)(s & 1u) * stride16;
  int4* peer = peer_scratch + so;
  const int4* mine = my_scratch + so;
  const long slot_bytes = (long)stride16 * 16;
  unsigned h = 0u;

  // push: four words in flight a thread
  int k = run.start + tid;
  for (; k + 3 * nt < run.end; k += 4 * nt) {
    const int4 v0 = in[k], v1 = in[k + nt], v2 = in[k + 2 * nt], v3 = in[k + 3 * nt];
    peer[k] = v0;
    peer[k + nt] = v1;
    peer[k + 2 * nt] = v2;
    peer[k + 3 * nt] = v3;
    if constexpr (CHK != kNoCheck)
      h ^= chk_word(v0, (unsigned)k) ^ chk_word(v1, (unsigned)(k + nt)) ^ chk_word(v2, (unsigned)(k + 2 * nt)) ^
           chk_word(v3, (unsigned)(k + 3 * nt));
  }
  for (; k < run.end; k += nt) {
    const int4 v = in[k];
    peer[k] = v;
    if constexpr (CHK != kNoCheck) h ^= chk_word(v, (unsigned)k);
  }
  if constexpr (CHK != kNoCheck) {
    if constexpr (PAR == kGlobal) {
      if (tid == 0) chk_ck(peer, slot_bytes)[b * CHK_PIECES + 1] = s;
    }
    chk_send(h, peer, slot_bytes, (unsigned)b, 0u, s);
  }

  drain_and_sync(drain);
  if (tid == 0) {
    u64 z = 0;
    if constexpr (CHK == kCheckedNoWait) {
      flag_release(peer_flags + b, s);
    } else {
      z = signal_wait(peer_flags + b, my_flags + b, s);
    }
    if constexpr (CHK != kNoCheck) {
      u32* diag = seq + CHK_DIAG;
      chk_after_spin(my_flags + b, s, z, diag);
      if constexpr (PAR == kGlobal) {
        if (chk_ck(mine, slot_bytes)[b * CHK_PIECES + 1] != s) atomicAdd(&diag[SAR_CHK_D_GP_PARITY], 1u);
      }
    }
  }
  __syncthreads();
  acquire_opt(acq);

  // reduce: my words + what the peer pushed, four words in flight a thread
  h = 0u;
  k = run.start + tid;
  for (; k + 3 * nt < run.end; k += 4 * nt) {
    const int4 a0 = in[k], a1 = in[k + nt], a2 = in[k + 2 * nt], a3 = in[k + 3 * nt];
    const int4 p0 = mine[k], p1 = mine[k + nt], p2 = mine[k + 2 * nt], p3 = mine[k + 3 * nt];
    out[k] = add_word<E>(a0, p0);
    out[k + nt] = add_word<E>(a1, p1);
    out[k + 2 * nt] = add_word<E>(a2, p2);
    out[k + 3 * nt] = add_word<E>(a3, p3);
    if constexpr (CHK != kNoCheck)
      h ^= chk_word(p0, (unsigned)k) ^ chk_word(p1, (unsigned)(k + nt)) ^ chk_word(p2, (unsigned)(k + 2 * nt)) ^
           chk_word(p3, (unsigned)(k + 3 * nt));
  }
  for (; k < run.end; k += nt) {
    const int4 a = in[k];
    const int4 p = mine[k];
    out[k] = add_word<E>(a, p);
    if constexpr (CHK != kNoCheck) h ^= chk_word(p, (unsigned)k);
  }
  if constexpr (CHK != kNoCheck)
    chk_recv(h, mine, run.start, run.end, (unsigned)b, 0u, s, slot_bytes, seq + CHK_DIAG, false);
}

}  // namespace sar

// the exact one-shot's kernarg list (spec 4): the names and order are the ABI
#define SAR_EXACT_PARAMS                                                                                  \
  const int4 *in, int4 *peer_scratch, int4 *my_scratch, int4 *out, unsigned *peer_flags, unsigned *my_flags, \
      unsigned *seq_ctrs, int n16, int slot_stride16, int drain, int acq
#define SAR_EXACT_ARGS \
  in, peer_scratch, my_scratch, out, peer_flags, my_flags, seq_ctrs, n16, slot_stride16, drain, acq
