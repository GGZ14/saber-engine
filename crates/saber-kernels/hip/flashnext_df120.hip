// FN120 (2026-10-13): THE MEGAKERNEL PROGRAM, PHASE A -- the persistent DATAFLOW primitive measured on
// gfx1201, NOT a product (no plan binds it; nothing in the engine launches it; its own code object, so
// every shipped object keeps its bytes).
//
// Three families of kernels:
//
//  1. RESIDENCY (`fn120_res_v<V>_l<L>_w<W>`): a check-in kernel compiled at a FORCED budget -- V VGPRs
//     (an asm clobber of v<V-1>), L bytes of static LDS (touched), W waves a workgroup. Every workgroup
//     arrives on one device-scope counter and spins until all G arrived OR a REALTIME timeout passes;
//     a timeout raises the refusal word. G is co-resident iff nothing refused: the host bisects G to the
//     MEASURED co-residency of each budget (the occupancy law's check on the silicon, not HIP's table).
//
//  2. THE RUNTIME (`fn120_df_run`): a persistent grid of G workgroups (128 threads) that first PROVES its
//     co-residency (all G check in before any proceeds; a REALTIME timeout refuses and every workgroup
//     exits), then pulls tasks from a device task table by ATOMIC TICKET (ticket order = a topological
//     order, so a ticket only waits on earlier tickets, which are held by resident workgroups: no
//     deadlock once co-residency is proven). A task names at most one dependency COUNTER and the count
//     it needs; lane 0 spins on it with `s_sleep` backoff, then EVERY thread takes a device-scope
//     ACQUIRE; the body runs; every thread takes a device-scope RELEASE, the workgroup barrier, and lane
//     0 adds 1 to the task's output counter. Bodies: NOP, PAYLOAD write (a run-tagged pattern) and
//     PAYLOAD check (pre-reads the producer's block BEFORE waiting, so a consumer without the acquire
//     reads the stale lines it pulled in: the race checker, in wave 0 with no barrier between the
//     pre-read, the spin and the check; flag DROP_ACQ is its CONTROL and must be caught). Every task stamps REALTIME at pickup, at dependency seen and at release.
//
//  3. THE BODY PAIR (`fn120_df_pair`: the runtime's loop over ROW / TILE tasks): FN86b's family-1 QSA pair -- the attention gate + the W8
//     o_proj's activation quant (`fn_attn_gate_q8_t256`, flashnext_hc) -> the W8 o_proj GEMM on 64-column
//     tiles (`fn111_t8_a8w8_n64_ldc`, flashnext_trunk8) -- as a DATAFLOW pair: one task a producer row,
//     one task a GEMM tile; a tile waits on its pair's row counter (== M), the next pair's rows wait on
//     the previous pair's tile counter (chain mode: a train of dependent layers). THE BIT-EXACT LAW BY
//     CONSTRUCTION (FN23's): there is NO copy of either body -- both certified sources are `#include`d
//     verbatim under the macros FN23 proved (`__global__` -> `__device__`, `__launch_bounds__` dropped,
//     `blockIdx` -> the virtual block from LDS through readfirstlane) plus ONE more, `threadIdx` -> the
//     VIRTUAL thread: the persistent workgroup is the GEMM's native 128 threads, and a 256-thread producer
//     row runs as two passes of 128 virtual threads (the producer has no barrier; a 16-lane quant group
//     stays inside one half-wave because the pass offset is a multiple of 32). Each include sits under its
//     code object's fp-contract state (flashnext_hc `off`; flashnext_trunk8 the hipcc default = fast).
//     `vg_wmma` is renamed in the trunk8 include only (both sources define the same helper text).
//     The twin oracle (tools/capture/fn120/fn120_probe.py `pair`) compares every output byte with the
//     SHIPPED symbols run from their prebuilt objects. The ordering CONTROL is a table whose tiles carry
//     no edge to their rows (must differ).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

// ------------------------------------------------------------------------------------- the clock
__device__ __forceinline__ uint64_t fn120_now() { return wall_clock64(); }   // REALTIME, 100 MHz

// ============================================================================== 1. RESIDENCY sweep
// out[0] = refusals (workgroups that timed out), out[1] = max arrivals any workgroup saw.
template <int V, int L>
__device__ __forceinline__ void fn120_res_body(unsigned *__restrict__ ctr, unsigned *__restrict__ out,
                                               unsigned G, unsigned long long timeout) {
  if constexpr (L > 0) {
    __shared__ unsigned char pad[L];
    volatile unsigned char *p = pad;
    if (threadIdx.x == 0) { p[0] = 1; p[L - 1] = 2; }
  }
  if (threadIdx.x == 0) {
    __hip_atomic_fetch_add(ctr, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    const uint64_t t0 = fn120_now();
    unsigned seen;
    while ((seen = __hip_atomic_load(ctr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) < G) {
      if (fn120_now() - t0 > timeout) {
        __hip_atomic_fetch_add(&out[0], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        break;
      }
      __builtin_amdgcn_s_sleep(1);
    }
    __hip_atomic_fetch_max(&out[1], seen, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
}

#define FN120_RES(VV, LL, WW)                                                                      \
  extern "C" __global__ __launch_bounds__(WW * 32) void fn120_res_v##VV##_l##LL##_w##WW(           \
      unsigned *__restrict__ ctr, unsigned *__restrict__ out, unsigned G,                          \
      unsigned long long timeout) {                                                                \
    asm volatile("" ::: FN120_CLOB_V);                                                             \
    fn120_res_body<VV, LL * 1024>(ctr, out, G, timeout);                                           \
  }

// The clobber names the budget's LAST VGPR so the allocation is at least V (the manifest records it).
#define FN120_CLOB_V "v63"
FN120_RES(64, 0, 4) FN120_RES(64, 16, 4) FN120_RES(64, 32, 4) FN120_RES(64, 48, 4) FN120_RES(64, 64, 4)
FN120_RES(64, 0, 8) FN120_RES(64, 16, 8) FN120_RES(64, 32, 8) FN120_RES(64, 48, 8) FN120_RES(64, 64, 8)
#undef FN120_CLOB_V
#define FN120_CLOB_V "v95"
FN120_RES(96, 0, 4) FN120_RES(96, 16, 4) FN120_RES(96, 32, 4) FN120_RES(96, 48, 4) FN120_RES(96, 64, 4)
FN120_RES(96, 0, 8) FN120_RES(96, 16, 8) FN120_RES(96, 32, 8) FN120_RES(96, 48, 8) FN120_RES(96, 64, 8)
#undef FN120_CLOB_V
#define FN120_CLOB_V "v127"
FN120_RES(128, 0, 4) FN120_RES(128, 16, 4) FN120_RES(128, 32, 4) FN120_RES(128, 48, 4) FN120_RES(128, 64, 4)
FN120_RES(128, 0, 8) FN120_RES(128, 16, 8) FN120_RES(128, 32, 8) FN120_RES(128, 48, 8) FN120_RES(128, 64, 8)
#undef FN120_CLOB_V
#define FN120_CLOB_V "v191"
FN120_RES(192, 0, 4) FN120_RES(192, 16, 4) FN120_RES(192, 32, 4) FN120_RES(192, 48, 4) FN120_RES(192, 64, 4)
FN120_RES(192, 0, 8) FN120_RES(192, 16, 8) FN120_RES(192, 32, 8) FN120_RES(192, 48, 8) FN120_RES(192, 64, 8)
#undef FN120_CLOB_V
#define FN120_CLOB_V "v255"
FN120_RES(256, 0, 4) FN120_RES(256, 16, 4) FN120_RES(256, 32, 4) FN120_RES(256, 48, 4) FN120_RES(256, 64, 4)
FN120_RES(256, 0, 8) FN120_RES(256, 16, 8) FN120_RES(256, 32, 8) FN120_RES(256, 48, 8) FN120_RES(256, 64, 8)
#undef FN120_CLOB_V

// ============================================================================== 2. THE RUNTIME
#define FN120_NT 128
#define FN120_NONE 0xffffffffu
enum : unsigned { FN120_NOP = 0, FN120_PUT = 1, FN120_CHECK = 2 };
enum : unsigned { FN120_DROP_ACQ = 1u, FN120_SPIN0 = 2u, FN120_SLEEP2 = 4u };

struct Fn120Task {          // 32 bytes
  unsigned kind;            // FN120_NOP / PUT / CHECK
  unsigned dep;             // the counter it waits on (FN120_NONE = none)
  unsigned need;            // the count it needs
  unsigned out;             // the counter it releases (+1)
  unsigned block;           // PUT / CHECK: the payload block (words at block * words)
  unsigned words;           // payload words a block
  unsigned tag;             // the run's tag (the pattern's seed)
  unsigned pad;
};

__device__ __forceinline__ unsigned fn120_pat(unsigned tag, unsigned block, unsigned w) {
  unsigned x = tag * 0x9E3779B1u ^ block * 0x85EBCA77u ^ w * 0xC2B2AE3Du;
  x ^= x >> 15;
  x *= 0x2C1B3C6Du;
  x ^= x >> 12;
  return x | 1u;            // never 0: the buffer's reset value
}

// Wait for `*c >= need` on lane 0 with s_sleep backoff; returns false on the timeout (refusal).
__device__ __forceinline__ bool fn120_wait(const unsigned *c, unsigned need, unsigned flags,
                                           unsigned long long timeout) {
  const uint64_t t0 = fn120_now();
  unsigned spins = 0;
  while (__hip_atomic_load(c, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < need) {
    if (flags & FN120_SPIN0) {
      // pure spin
    } else if (flags & FN120_SLEEP2) {
      __builtin_amdgcn_s_sleep(2);
    } else {
      __builtin_amdgcn_s_sleep(1);
    }
    if ((++spins & 63u) == 0 && fn120_now() - t0 > timeout) return false;
  }
  return true;
}

__shared__ unsigned fn120_s_ticket, fn120_s_ok;

// stats[0] = refusals (co-residency), [1] = dependency timeouts, [2] = payload mismatches,
//      [3] = tasks run, [4] = check-in arrivals seen by the last workgroup.
// stamps[t * 4 + {0,1,2,3}] = pickup, dependency seen, release, the workgroup id.
extern "C" __global__ __launch_bounds__(FN120_NT) void fn120_df_run(
    const Fn120Task *__restrict__ tab, unsigned ntasks, unsigned *__restrict__ ticket,
    unsigned *__restrict__ ctr, unsigned *__restrict__ checkin, unsigned *__restrict__ payload,
    unsigned *__restrict__ stats, unsigned long long *__restrict__ stamps, unsigned G,
    unsigned flags, unsigned long long timeout) {
  const unsigned tid = threadIdx.x;
  // ---- the co-residency proof: every workgroup checks in before any proceeds
  if (tid == 0) {
    __hip_atomic_fetch_add(checkin, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    const uint64_t t0 = fn120_now();
    unsigned ok = 1, seen;
    while ((seen = __hip_atomic_load(checkin, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) < G) {
      if (__hip_atomic_load(&stats[0], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 0u ||
          fn120_now() - t0 > timeout) {
        __hip_atomic_fetch_add(&stats[0], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        ok = 0;
        break;
      }
      __builtin_amdgcn_s_sleep(1);
    }
    __hip_atomic_fetch_max(&stats[4], seen, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    fn120_s_ok = ok;
  }
  __syncthreads();
  if (!fn120_s_ok) return;
  unsigned ran = 0, bad = 0;
  for (;;) {
    if (tid == 0) {
      fn120_s_ticket = __hip_atomic_fetch_add(ticket, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
    const unsigned t = __builtin_amdgcn_readfirstlane(fn120_s_ticket);
    if (t >= ntasks) break;
    const uint64_t tp = fn120_now();
    const Fn120Task k = tab[t];
    unsigned *blk = payload + (size_t)k.block * k.words;
    unsigned pre = 0;
    uint64_t ts = tp;
    if (k.kind == FN120_CHECK) {
      // THE RACE CHECKER, all in wave 0 with NO barrier between its reads (a workgroup barrier in
      // gfx12 WGP mode is itself a `global_inv scope:SCOPE_SE`, which hid the control in h1): pull
      // the PRODUCER's (old) lines into this CU's caches, spin on its counter (every lane, relaxed),
      // take the device-scope ACQUIRE (DROP_ACQ = the control: none), then check every word. The
      // producer's block = this block - 1 (the table pairs PUT block b with CHECK block b + 1).
      // h2: the order is FORCED -- the CHECK's ticket comes first and it raises its READY counter
      // (k.out) after the pre-read; the PUT waits for READY before it writes. So the consumer's
      // caches hold the old lines when the producer's new ones land (the control's premise).
      if (tid < 32) {
        const unsigned pb = k.block - 1;
        const unsigned *src = payload + (size_t)pb * k.words;
        for (unsigned w = tid; w < k.words; w += 32) pre ^= src[w];
        asm volatile("" ::"v"(pre));
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");      // the pre-read has landed
        if (tid == 0 && k.out != FN120_NONE)
          __hip_atomic_fetch_add(&ctr[k.out], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        const bool ok = fn120_wait(&ctr[k.dep], k.need, flags, timeout);
        if (!(flags & FN120_DROP_ACQ)) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        ts = fn120_now();
        unsigned miss = 0;
        for (unsigned w = tid; w < k.words; w += 32) miss += src[w] != fn120_pat(k.tag, pb, w);
        bad += miss;
        if (!ok && tid == 0)
          __hip_atomic_fetch_add(&stats[1], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    } else {
      if (k.dep != FN120_NONE) {
        if (tid == 0) fn120_s_ok = fn120_wait(&ctr[k.dep], k.need, flags, timeout) ? 1u : 0u;
        __syncthreads();
        if (!fn120_s_ok) {
          if (tid == 0) __hip_atomic_fetch_add(&stats[1], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
          break;
        }
        if (!(flags & FN120_DROP_ACQ)) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
      }
      ts = fn120_now();
      if (k.kind == FN120_PUT) {
        for (unsigned w = tid; w < k.words; w += FN120_NT) blk[w] = fn120_pat(k.tag, k.block, w);
      }
    }
    asm volatile("" ::"v"(pre));
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (tid == 0) {
      if (k.out != FN120_NONE && k.kind != FN120_CHECK)
        __hip_atomic_fetch_add(&ctr[k.out], 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      const uint64_t tr = fn120_now();
      stamps[(size_t)t * 4 + 0] = tp;
      stamps[(size_t)t * 4 + 1] = ts;
      stamps[(size_t)t * 4 + 2] = tr;
      stamps[(size_t)t * 4 + 3] = blockIdx.x;
    }
    ++ran;
  }
  // the mismatch count is per thread: sum it
  if (bad) __hip_atomic_fetch_add(&stats[2], bad, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  if (tid == 0) __hip_atomic_fetch_add(&stats[3], ran, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// The launch train's comparison node: every workgroup stamps REALTIME at entry and exit (min / max).
// st[node * 2 + 0] = min start (init ~0), st[node * 2 + 1] = max end (init 0).
extern "C" __global__ __launch_bounds__(256) void fn120_stamp(unsigned long long *__restrict__ st,
                                                              unsigned node) {
  if (threadIdx.x == 0) {
    const uint64_t t = fn120_now();
    __hip_atomic_fetch_min(&st[node * 2 + 0], t, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    const uint64_t e = fn120_now();
    __hip_atomic_fetch_max(&st[node * 2 + 1], e, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
}

// ============================================================================== 3. THE BODY PAIR
struct fn120_dim3 {
  unsigned int x, y, z;
};
__shared__ fn120_dim3 fn120_vbid;
__shared__ unsigned fn120_vtoff;

__device__ __forceinline__ fn120_dim3 fn120_rfl_b() {
  fn120_dim3 r;
  r.x = __builtin_amdgcn_readfirstlane(fn120_vbid.x);
  r.y = __builtin_amdgcn_readfirstlane(fn120_vbid.y);
  r.z = __builtin_amdgcn_readfirstlane(fn120_vbid.z);
  return r;
}
__device__ __forceinline__ fn120_dim3 fn120_vtid() {
  fn120_dim3 r;
  r.x = __builtin_amdgcn_workitem_id_x() + __builtin_amdgcn_readfirstlane(fn120_vtoff);
  r.y = 0;
  r.z = 0;
  return r;
}

#pragma push_macro("__global__")
#pragma push_macro("__launch_bounds__")
#undef __global__
#undef __launch_bounds__
#define __global__ __device__
#define __launch_bounds__(...)
#define blockIdx (fn120_rfl_b())
#define threadIdx (fn120_vtid())

#pragma clang fp contract(off)
#include "flashnext_hc.hip"
#pragma clang fp contract(fast)
#define vg_wmma fn120_t8_vg_wmma
#include "flashnext_trunk8.hip"
#undef vg_wmma
#pragma clang fp contract(fast)

#undef threadIdx
#undef blockIdx
#undef __launch_bounds__
#undef __global__
#pragma pop_macro("__launch_bounds__")
#pragma pop_macro("__global__")

// One pair's buffers (the shipped symbols' arguments).
struct Fn120Pair {
  const __bf16 *core;       // [M, n] attention core
  const __bf16 *gate;       // [M, gs] the merged q|gate projection rows
  __bf16 *out;              // [M, n] the gate's bf16 output (stored, as shipped)
  unsigned char *q;         // [M, n] e4m3 = the GEMM's A
  float *scales;            // [M, n / 128] = the GEMM's As
  const unsigned char *w;   // [N / 16, n * 16] e4m3 preshuffled
  const float *ws;          // [N / 128, n / 128]
  __bf16 *c;                // [M, ldc]
};

enum : unsigned { FN120_PAIR_PREFETCH = 4u };

// Each body behind a call boundary: the persistent loop's own control flow stays the runtime's.
__device__ __attribute__((noinline)) void fn120_row_call(const __bf16 *core, const __bf16 *gate, __bf16 *out,
                                                         unsigned char *q, float *scales, int n, int hd,
                                                         int gs) {
  fn_attn_gate_q8_body<false>(core, gate, out, q, scales, n, hd, gs);
}
__device__ __attribute__((noinline)) void fn120_tile_call(const unsigned char *a, const unsigned char *w,
                                                          const float *as, const float *ws, __bf16 *c,
                                                          int M, int N, int K, int ldc) {
  t8_gemm_a8w8_impl<DG_BM, 64, DG_BK, DG_NW>(a, w, as, ws, c, M, N, K, (long)ldc);
}

// h3-h6 (LEDGER FN120): a first form that DECODED its tasks in-kernel from the ticket (p = t / per,
// row / tile by index, the edges computed) hung at G = 1 even with no body compiled in, while
// `fn120_df_run` over a HOST-BUILT task table runs; that form is removed. So the pair runs on
// fn120_df_run's loop text unchanged, with two more task kinds (the dependency edges come from the
// table the host builds: a tile needs its pair's M rows, a CHAIN row needs the previous pair's T
// tiles; the control's table simply drops the tiles' edge):
//   ROW  (kind 3): block = the pair, words = the row   -> fn_attn_gate_q8_body, two virtual passes
//   TILE (kind 4): block = the pair, words = the tile  -> t8_gemm_a8w8_impl<16, 64, 128, 4>
// dims = {M, N, K, n, hd, gs, ldc}; flags bit 4 (PREFETCH): a tile touches its weight slab first.
enum : unsigned { FN120_ROW = 3, FN120_TILE = 4 };
extern "C" __global__ __launch_bounds__(FN120_NT) void fn120_df_pair(
    const Fn120Task *__restrict__ tab, unsigned ntasks, unsigned *__restrict__ ticket,
    unsigned *__restrict__ ctr, unsigned *__restrict__ checkin, const Fn120Pair *__restrict__ pd,
    const int *__restrict__ dims, unsigned *__restrict__ stats, unsigned long long *__restrict__ stamps,
    unsigned G, unsigned flags, unsigned long long timeout) {
  const unsigned tid = threadIdx.x;
  if (tid == 0) {
    __hip_atomic_fetch_add(checkin, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    const uint64_t t0 = fn120_now();
    unsigned ok = 1, seen;
    while ((seen = __hip_atomic_load(checkin, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) < G) {
      if (__hip_atomic_load(&stats[0], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 0u ||
          fn120_now() - t0 > timeout) {
        __hip_atomic_fetch_add(&stats[0], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        ok = 0;
        break;
      }
      __builtin_amdgcn_s_sleep(1);
    }
    __hip_atomic_fetch_max(&stats[4], seen, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    fn120_s_ok = ok;
  }
  __syncthreads();
  if (!fn120_s_ok) return;
  const int M = dims[0], N = dims[1], K = dims[2], n = dims[3], hd = dims[4], gs = dims[5],
            ldc = dims[6];
  const unsigned nt = (unsigned)((N + 63) / 64);
  unsigned ran = 0;
  for (;;) {
    if (tid == 0) {
      fn120_s_ticket = __hip_atomic_fetch_add(ticket, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
    const unsigned t = __builtin_amdgcn_readfirstlane(fn120_s_ticket);
    if (t >= ntasks) break;
    const uint64_t tp = fn120_now();
    const Fn120Task k = tab[t];
    const Fn120Pair P = pd[k.block];
    uint64_t ts = tp;
    if (k.kind == FN120_TILE && (flags & FN120_PAIR_PREFETCH)) {
      const unsigned j = k.words % nt;
      const unsigned *wb = (const unsigned *)(P.w + (size_t)(j * 64 / 16) * (size_t)K * 16);
      const unsigned nw = 4u * (unsigned)K * 16u / 4u;
      unsigned acc = 0;
      for (unsigned w = tid * 32u; w < nw; w += FN120_NT * 32u) acc ^= wb[w];
      asm volatile("" ::"v"(acc));
    }
    if (k.dep != FN120_NONE) {
      if (tid == 0) fn120_s_ok = fn120_wait(&ctr[k.dep], k.need, 0u, timeout) ? 1u : 0u;
      __syncthreads();
      if (!fn120_s_ok) {
        if (tid == 0) __hip_atomic_fetch_add(&stats[1], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        break;
      }
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    ts = fn120_now();
    if (k.kind == FN120_ROW) {
      if (tid == 0) { fn120_vbid.x = k.words; fn120_vbid.y = 0; fn120_vbid.z = 0; }
#pragma unroll 1
      for (unsigned pass = 0; pass < 2; ++pass) {
        if (tid == 0) fn120_vtoff = pass * FN120_NT;
        __syncthreads();
        fn120_row_call(P.core, P.gate, P.out, P.q, P.scales, n, hd, gs);
        __syncthreads();
      }
      if (tid == 0) fn120_vtoff = 0;
    } else if (k.kind == FN120_TILE) {
      if (tid == 0) {
        fn120_vbid.x = k.words % nt;
        fn120_vbid.y = k.words / nt;
        fn120_vbid.z = 0;
        fn120_vtoff = 0;
      }
      __syncthreads();
      fn120_tile_call(P.q, P.w, P.scales, P.ws, P.c, M, N, K, ldc);
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (tid == 0) {
      if (k.out != FN120_NONE)
        __hip_atomic_fetch_add(&ctr[k.out], 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      const uint64_t tr = fn120_now();
      stamps[(size_t)t * 4 + 0] = tp;
      stamps[(size_t)t * 4 + 1] = ts;
      stamps[(size_t)t * 4 + 2] = tr;
      stamps[(size_t)t * 4 + 3] = blockIdx.x;
    }
    ++ran;
  }
  if (tid == 0) __hip_atomic_fetch_add(&stats[3], ran, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
