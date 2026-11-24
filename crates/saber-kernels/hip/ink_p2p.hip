// Inkling 2 (inkling2_loop) two-card exchange body, INK8 (docs/INKLING2.md section 9). Saber's own.
// One process, two HIP threads, card r = rank r, peer access on: each rank passes its own buffer
// and the PEER's buffer address (read over P2P), its own link words and the peer's.
//
//   ink_p2p_xchg   sum = 1: out[r, c] = own[r, c] + peer[r, c], c < n0 (the visit's two reduces:
//                  f32 addition commutes, so both ranks hold the same bytes = rank 0 + rank 1);
//                  sum = 0: out[r, 0..n0) = own[r, ..], out[r, n0..n0 + n1) = peer[r, ..] (the
//                  head's vocab halves gathered on the card).
//
// The link (4 u32 words a rank, zeroed once at boot): [0] calls (this rank's completed calls),
// [1] done (workgroups finished in the current call), [2] arrive (written by the PEER: its call
// number whose operand is published), [3] depart (written by the PEER: its call number whose reads
// of OUR operand are complete). A call s = calls + 1:
//   1. workgroup 0 makes our operand visible at system scope (release: L2 write-back) and stores
//      s into the peer's arrive word;
//   2. every workgroup spins (one wave, s_sleep) until our arrive word >= s, then every thread
//      takes the system-scope acquire; peer elements are read with system-scope loads;
//   3. the LAST workgroup to finish (a counter, chosen by value) bumps calls, stores s into the
//      peer's depart word and spins until ours >= s -- the kernel ends only after the peer has
//      read our operand, so the stream may reuse it.
// The call number lives on the card, so a captured graph replays it (no host sequence number).
// Built with -ffp-contract=off.
#include <hip/hip_runtime.h>

__device__ __forceinline__ unsigned ink_ld_sys(const unsigned* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void ink_spin_ge(const unsigned* p, unsigned s) {
  while ((int)(ink_ld_sys(p) - s) < 0) __builtin_amdgcn_s_sleep(2);
}

extern "C" __global__ __launch_bounds__(256) void ink_p2p_xchg(
    const float* __restrict__ own, int ld_own, const float* peer, int ld_peer, float* __restrict__ out,
    int ld_out, int rows, int n0, int n1, int sum, unsigned* link, unsigned* peer_link) {
  __shared__ unsigned s_seq;
  const int tid = threadIdx.x;
  // wave 0 does the handshake under a wave-uniform branch (FN120c)
  if (__builtin_amdgcn_readfirstlane(tid >> 5) == 0) {
    const unsigned s = __hip_atomic_load(&link[0], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1u;
    if (blockIdx.x == 0) {
      __atomic_thread_fence(__ATOMIC_RELEASE);  // order our operand's stores before the flag
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");  // system scope: write back L2
      if (tid == 0) __hip_atomic_store(&peer_link[2], s, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    ink_spin_ge(&link[2], s);
    if (tid == 0) s_seq = s;
  }
  __syncthreads();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");  // every thread: system-scope acquire
  const unsigned s = s_seq;
  const long cols = sum ? n0 : (long)n0 + n1;
  const long total = (long)rows * cols;
  for (long i = (long)blockIdx.x * 256 + tid; i < total; i += (long)gridDim.x * 256) {
    const long r = i / cols, c = i - r * cols;
    float v;
    if (sum) {
      const float p = __builtin_bit_cast(float, ink_ld_sys((const unsigned*)&peer[r * ld_peer + c]));
      v = own[r * ld_own + c] + p;
    } else if (c < n0) {
      v = own[r * ld_own + c];
    } else {
      v = __builtin_bit_cast(float, ink_ld_sys((const unsigned*)&peer[r * ld_peer + (c - n0)]));
    }
    out[r * ld_out + c] = v;
  }
  __syncthreads();
  if (__builtin_amdgcn_readfirstlane(tid >> 5) == 0) {
    unsigned last = 0;
    if (tid == 0) {
      const unsigned d = __hip_atomic_fetch_add(&link[1], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
      last = (d == gridDim.x - 1) ? 1u : 0u;
    }
    last = __builtin_amdgcn_readfirstlane(last);
    if (last) {
      if (tid == 0) {
        __hip_atomic_store(&link[1], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        __hip_atomic_store(&link[0], s, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        __hip_atomic_store(&peer_link[3], s, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
      }
      ink_spin_ge(&link[3], s);
    }
  }
}
