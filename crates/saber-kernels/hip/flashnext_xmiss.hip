// FN53c: THE VALVE'S DEVICE-SIDE MISS FLAG (Flash-Next, `--fn-mem-valve on` + `--fn-valve-graphs
// on`). One launch a target layer, between the route and the grouped experts, in the captured
// graphs of a decode / verify step whose expert tier is INCOMPLETE (records demoted to the drive
// by the pressure valve; their xptr entries are 0).
//
// The law (one workgroup):
//   1. every routed id of THIS rank (`lo <= id < lo + 256`) whose gate_up xptr entry is 0 is a
//      MISS (a bitmap in LDS: each record once, in expert order);
//   2. no miss: return (the common case -- the open set is the coldest records);
//   3. a miss: thread 0 writes the list, the layer and the count into the coherent HOST mailbox,
//      bumps the mailbox's sequence and RELEASES it at system scope, then spins (s_sleep) on the
//      host's ACK == that sequence with a system-scope ACQUIRE (FN71b's fence law: a kernel
//      already running when the host's word lands must acquire at system scope). The host --
//      polling the mailbox while it waits for the step -- has read each missing record from the
//      drive into the valve's bounce (the P2PDMA pool, or pinned RAM) and written the record's six
//      sub-record addresses (gate_up x3, down x3) before the ACK;
//   4. thread 0 patches the layer's xptr entries with those addresses. The grouped GEMMs that
//      follow in stream order start after this kernel ends, so they read the patched tables and
//      the drive's bytes (a kernel launched after the word sees them).
// A host that never answers is a hang, not a wrong byte: the spin traps after ~2^26 sleeps.
//
// Mailbox (u32 words; addresses as u64 at byte 2048): [0] REQ, [1] LAYER, [2] N, [3] DSEQ (the
// device's own counter), [4] ACK (the host's), [16 ..) LIST, ADDR[6 j + k].
#include <hip/hip_runtime.h>
#include <stdint.h>

#define MB_REQ 0
#define MB_LAYER 1
#define MB_N 2
#define MB_DSEQ 3
#define MB_ACK 4
#define MB_LIST 16
#define MB_ADDR_BYTES 2048

extern "C" __global__ __launch_bounds__(256) void fn_xmiss_e256(const int* __restrict__ ids,
                                                                 int n, uint64_t* gu, uint64_t* dn,
                                                                 int lo, uint32_t* mbox,
                                                                 int layer) {
  __shared__ uint32_t bits[8];
  __shared__ uint32_t any;
  const int t = threadIdx.x;
  if (t < 8) bits[t] = 0u;
  if (t == 0) any = 0u;
  __syncthreads();
  for (int i = t; i < n; i += blockDim.x) {
    const int e = ids[i] - lo;
    if (e >= 0 && e < 256 && gu[3 * e] == 0ull) {
      atomicOr(&bits[e >> 5], 1u << (e & 31));
      any = 1u;
    }
  }
  __syncthreads();
  if (any == 0u || t != 0) return;
  uint32_t cnt = 0;
  for (int w = 0; w < 8; ++w) {
    uint32_t b = bits[w];
    while (b != 0u) {
      const int k = __builtin_ctz(b);
      b &= b - 1u;
      __hip_atomic_store(&mbox[MB_LIST + cnt], (uint32_t)(w * 32 + k), __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_SYSTEM);
      ++cnt;
    }
  }
  const uint32_t seq =
      __hip_atomic_load(&mbox[MB_DSEQ], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) + 1u;
  __hip_atomic_store(&mbox[MB_DSEQ], seq, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  __hip_atomic_store(&mbox[MB_LAYER], (uint32_t)layer, __ATOMIC_RELAXED,
                     __HIP_MEMORY_SCOPE_SYSTEM);
  __hip_atomic_store(&mbox[MB_N], cnt, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  __hip_atomic_store(&mbox[MB_REQ], seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  uint32_t spins = 0;
  while (__hip_atomic_load(&mbox[MB_ACK], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != seq) {
    __builtin_amdgcn_s_sleep(8);
    if (++spins > (1u << 26)) __builtin_trap();
  }
  uint64_t* addr = (uint64_t*)((char*)mbox + MB_ADDR_BYTES);
  for (uint32_t j = 0; j < cnt; ++j) {
    const uint32_t e =
        __hip_atomic_load(&mbox[MB_LIST + j], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    for (int k = 0; k < 3; ++k) {
      gu[3 * e + k] =
          __hip_atomic_load(&addr[6 * j + k], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      dn[3 * e + k] =
          __hip_atomic_load(&addr[6 * j + 3 + k], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
  }
}

// FN98: THE IN-GRAPH LAYER GATE (`--fn-graph-form one`). One single-wave launch before each gate
// block's first layer, INSIDE the step form's one captured graph: the block's entry of the gate
// table (written by the host into device memory before every launch of the graph: the word's
// address and the value the block's LAST issued expert-copy mark writes there; value 0 = no copy
// in flight, the gate passes at once) is read, and thread 0 spins (s_sleep) until the mover's
// SDMA-written landed word reaches the value with a SYSTEM-scope ACQUIRE (FN71b's law; the slot
// bytes the SDMA wrote before the word are then visible to the launches after this one: the
// card-1 probe's oracle, 40 of 40 new with the old bytes primed into the caches, 10 of 10 stale
// with the gate open). The word is the one `hipStreamWaitValue32` waited outside the graph (FN72b);
// a captured wait node holds its value at capture and ROCm 10.1 refuses to update it in the exec
// (`hipGraphExecBatchMemOpNodeSetParams` -> invalid argument), so the value is DATA here.
// A word that never lands is a hang, not a wrong byte: the spin traps after ~2^26 sleeps.
struct Gate98 {
  uint64_t word;
  uint32_t value;
  uint32_t pad;
};

extern "C" __global__ __launch_bounds__(32) void fn_gate98(const Gate98* __restrict__ tab,
                                                           int idx) {
  if (threadIdx.x != 0) return;
  const Gate98 g = tab[idx];
  if (g.value == 0u) return;
  const uint32_t* w = (const uint32_t*)g.word;
  uint32_t spins = 0;
  while (__hip_atomic_load(w, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < g.value) {
    __builtin_amdgcn_s_sleep(2);
    if (++spins > (1u << 26)) __builtin_trap();
  }
}
