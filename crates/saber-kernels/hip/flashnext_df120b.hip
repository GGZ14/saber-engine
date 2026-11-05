// FN120c (2026-10-13): THE MEGAKERNEL PROGRAM, PHASE B -- the GDN mixer half's TAIL as ONE dataflow
// launch beside the train: the merged projection (`fn86_t8_q8w8_sfx_ldc`, flashnext_trunk8), the fused
// GDN update + gated norm + act quant (`fn86_gdn_fused_update_nq8_fc_f16s_w4k128v128_ab1`,
// flashnext_gdn_fc) and the out projection (`fn111_t8_a8w8_n64_ldc`, flashnext_trunk8) of a b1 verify
// step's GDN layer run as TASKS of one persistent grid (96 workgroups x 256 threads: the measured
// co-residency of a 256-VGPR body at <= 32 KiB LDS, LEDGER FN120), each body's virtual blocks one
// task, a stage's tasks released on striped counters its consumer stage waits for. Its own code object
// (every shipped object keeps its bytes); `--fn-dataflow gdn` substitutes it for the three launches
// inside a captured run (`flashnext/dev.rs`), `off` = the train (the control).
//
// THE BIT-EXACT LAW BY CONSTRUCTION (FN23's, LEDGER FN23 / FN120): there is NO copy of any body. The
// shipped sources are `#include`d verbatim under FN23's macros (`__global__` -> `__device__`,
// `__launch_bounds__` dropped, `blockIdx` / `gridDim` / `blockDim` -> the VIRTUAL block from LDS
// through readfirstlane) + FN120's `threadIdx` -> the virtual thread, each include under its code
// object's fp-contract state (trunk8: the hipcc default = fast; gdn_fc: off), and every body is
// called with the SAME kernarg image the train launches (FN23's `fn23_call` unpack at the ABI's
// offsets). The two 128-thread GEMM bodies run MIRRORED on the 256-thread workgroup: waves 4..7 are
// threads 0..127 again (virtual tid = tid & 127), so every `__syncthreads()` stays the hardware
// barrier over all eight waves and both halves compute and store the SAME bytes (the bodies hold no
// atomic: checked in the source); the fused update runs at its native 256 threads.
//
// THE DATAFLOW RULES (CLAUDE.md, the persistent-kernel law): the task list and every edge come from a
// HOST-BUILT table (stage descriptors, tasks in topological order, kernarg images); tickets are taken
// in order (a ticket waits only on earlier tickets, so no deadlock while >= the fused update's 24
// virtual blocks run at once -- its own grid barrier needs them co-resident); a consumer task spins on
// its producer stage's 8 striped counters (lane 0, `s_sleep`), every thread then takes the agent
// acquire; a producer releases (fence + barrier + one atomic on stripe vb & 7). The LAST workgroup to
// leave the loop resets the ticket, the exit count and the counters (every other workgroup has left
// its loop), so a graph replays the launch with no memset node.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include <cstdint>
#include <utility>

// ------------------------------------------------------------------------------ the virtual block
struct fn120b_dim3 {
  unsigned int x, y, z;
};
__shared__ fn120b_dim3 fn120b_vbid, fn120b_vgdim;
__shared__ unsigned fn120b_vthr, fn120b_ticket, fn120b_ok, fn120b_par;

__device__ __forceinline__ fn120b_dim3 fn120b_rfl(const fn120b_dim3 &d) {
  fn120b_dim3 r;
  r.x = __builtin_amdgcn_readfirstlane(d.x);
  r.y = __builtin_amdgcn_readfirstlane(d.y);
  r.z = __builtin_amdgcn_readfirstlane(d.z);
  return r;
}
__device__ __forceinline__ fn120b_dim3 fn120b_bdim() {
  fn120b_dim3 r;
  r.x = __builtin_amdgcn_readfirstlane(fn120b_vthr);
  r.y = 1;
  r.z = 1;
  return r;
}
// the virtual thread: a 128-thread body is MIRRORED on the 256-thread workgroup (tid & 127)
__device__ __forceinline__ fn120b_dim3 fn120b_vtid() {
  fn120b_dim3 r;
  r.x = __builtin_amdgcn_workitem_id_x() & (__builtin_amdgcn_readfirstlane(fn120b_vthr) - 1u);
  r.y = 0;
  r.z = 0;
  return r;
}

// ------------------------------------------------------------------ the certified sources, verbatim
#pragma push_macro("__global__")
#pragma push_macro("__launch_bounds__")
#undef __global__
#undef __launch_bounds__
#define __global__ __device__
#define __launch_bounds__(...)
#define blockIdx (fn120b_rfl(fn120b_vbid))
#define gridDim (fn120b_rfl(fn120b_vgdim))
#define blockDim (fn120b_bdim())
#define threadIdx (fn120b_vtid())

#pragma clang fp contract(fast)
#include "flashnext_trunk8.hip"
#pragma clang fp contract(off)
#include "flashnext_gdn_fc.hip"
#pragma clang fp contract(fast)

#undef threadIdx
#undef blockDim
#undef gridDim
#undef blockIdx
#undef __launch_bounds__
#undef __global__
#pragma pop_macro("__launch_bounds__")
#pragma pop_macro("__global__")

// ------------------------------------------------------------------------------ argument unpacking
// FN23's: a kernarg image is the ABI's -- each argument at the next multiple of its own alignment.
template <typename... A> struct fn120b_layout {
  static constexpr unsigned n = sizeof...(A);
  struct offs_t {
    unsigned v[n > 0 ? n : 1];
  };
  static constexpr offs_t offs() {
    offs_t r{};
    unsigned o = 0, i = 0;
    ((o = (o + alignof(A) - 1) / alignof(A) * alignof(A), r.v[i++] = o, o += sizeof(A)), ...);
    return r;
  }
};
// Every argument is UNIFORM (the train passes it in SGPRs): each dword is read through
// readfirstlane so the body keeps its arguments in scalar registers, as the shipped kernel does
// (a vector copy of a 43-argument image costs VGPRs the bodies need: scratch spills).
template <typename T> __device__ __forceinline__ T fn120b_ld(const unsigned char *p) {
  static_assert(sizeof(T) == 4 || sizeof(T) == 8, "scalars and pointers only");
  unsigned w[sizeof(T) / 4];
#pragma unroll
  for (unsigned i = 0; i < sizeof(T) / 4; ++i)
    w[i] = __builtin_amdgcn_readfirstlane(*(const unsigned *)(p + 4 * i));
  T v;
  __builtin_memcpy(&v, w, sizeof(T));
  return v;
}
template <auto F> struct fn120b_call;
template <typename... A, void (*F)(A...)> struct fn120b_call<F> {
  template <unsigned... I>
  __device__ __forceinline__ static void go_(const unsigned char *img,
                                             std::integer_sequence<unsigned, I...>) {
    constexpr auto o = fn120b_layout<A...>::offs();
    F(fn120b_ld<A>(img + o.v[I])...);
  }
  __device__ __forceinline__ static void go(const unsigned char *img) {
    go_(img, std::make_integer_sequence<unsigned, sizeof...(A)>{});
  }
  __device__ __forceinline__ static void offsets(unsigned *out) {
    constexpr auto o = fn120b_layout<A...>::offs();
    out[0] = sizeof...(A);
    for (unsigned i = 0; i < sizeof...(A); ++i) out[1 + i] = o.v[i];
  }
};
// one body as its own function: the persistent kernel's VGPRs are the widest body's
template <auto F> __device__ __attribute__((noinline)) void fn120b_body_n(const unsigned char *img) {
  fn120b_call<F>::go(img);
}
// inlined: no call-boundary spills (the uniform-control law makes inlining safe; c12 measured the
// noinline form's 560 B of scratch a thread as the cost)
template <auto F> __device__ __attribute__((always_inline)) inline void fn120b_body_i(const unsigned char *img) {
  fn120b_call<F>::go(img);
}

// ------------------------------------------------------------------------------ the tables
enum : unsigned { DF_SFX = 1, DF_GDN = 2, DF_N64 = 3 };
#define DF_NONE 0xffffffffu
#define DF_STRIPES 8u

struct Df120Stage {        // 32 bytes
  unsigned kind;           // DF_*
  unsigned thr;            // the body's threads (128 = mirrored, 256 = native)
  unsigned gx, gy, gz;     // the train's grid
  unsigned img;            // byte offset of the kernarg image in `imgs`
  unsigned pad[2];
};
struct Df120Task {         // 16 bytes
  unsigned stage;          // index into the stages
  unsigned vb;             // the virtual block (linear: x fastest)
  unsigned dep;            // the stage it waits for (DF_NONE = none)
  unsigned need;           // that stage's task count
};

__device__ __forceinline__ uint64_t fn120b_now() { return wall_clock64(); }   // REALTIME, 100 MHz

// ctl (one 256-byte block a site, zero at boot; nothing resets it on the host):
//   u64 [0]       check-ins (monotonic: launch e's workgroups arrive as e x G .. (e + 1) x G - 1)
//   u32 [2], [3]  the ticket of the launches of even / odd epoch
//   u32 [4] refusals, [5] dependency timeouts (STICKY, read by the host), [6] launches begun
//   u32 [16 + (b x 3 + s) x 8 + j]  epoch-parity b's stage s counter stripe j
// THE EPOCH LAW (no reset after the loop): launch e uses the parity-(e & 1) ticket and counters;
// after its check-in completes (launch e - 1 has ended, e + 1 cannot start before e ends: one
// stream), its FIRST arriver zeroes the other parity's words for launch e + 1.
#define DF_CTR0 16u
__device__ __forceinline__ bool fn120b_wait(const unsigned *c, unsigned need,
                                            unsigned long long timeout) {
  const uint64_t t0 = fn120b_now();
  unsigned spins = 0;
  for (;;) {
    unsigned sum = 0;
#pragma unroll
    for (unsigned j = 0; j < DF_STRIPES; ++j)
      sum += __hip_atomic_load(&c[j], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    if (sum >= need) return true;
    __builtin_amdgcn_s_sleep(1);
    if ((++spins & 63u) == 0 && fn120b_now() - t0 > timeout) return false;
  }
}

// MODE 0 = the engine form (all three bodies), 1 = the GEMM bodies only, 2 = no body (probes).
// THE UNIFORM-CONTROL LAW (c4-c11, LEDGER FN120c): no branch of this loop is per lane. A
// `if (tid == 0) { atomic }` inside a loop with barriers let the structurizer park lane 0 at a
// divergent loop exit while the other lanes looped back to the ticket barrier and re-read the
// stale ticket forever (G 1, no body: the Phase A in-kernel loop hang). So WAVE 0 does the
// single-lane work under a wave-uniform branch (readfirstlane(tid >> 5) == 0); an atomic's single
// effective lane is chosen by VALUE (increment lane == 0 ? 1 : 0, read back through readfirstlane);
// every exit / dispatch value is readfirstlane'd.
template <int MODE, bool INL = true>
__device__ __forceinline__ void fn120b_run(
    const Df120Stage *__restrict__ st, const Df120Task *__restrict__ tk, unsigned ntasks,
    const unsigned char *__restrict__ imgs, unsigned *__restrict__ ctl, unsigned nstages,
    unsigned long long timeout) {
  const unsigned tid = threadIdx.x;
  const unsigned lane = tid & 31u;
  const bool w0 = __builtin_amdgcn_readfirstlane(tid >> 5) == 0u;
  const unsigned one = lane == 0u ? 1u : 0u;
  const unsigned G = gridDim.x;
  unsigned long long *checkin = (unsigned long long *)ctl;
  // ---- the co-residency proof (the persistent-kernel law): every workgroup checks in first
  if (w0) {
    const unsigned long long r = __hip_atomic_fetch_add(checkin, (unsigned long long)one, __ATOMIC_RELAXED,
                                                        __HIP_MEMORY_SCOPE_AGENT);
    const unsigned long long ar =
        ((unsigned long long)__builtin_amdgcn_readfirstlane((unsigned)(r >> 32)) << 32) |
        __builtin_amdgcn_readfirstlane((unsigned)r);
    const unsigned long long e = ar / G, need = (e + 1) * G;
    const uint64_t t0 = fn120b_now();
    unsigned ok = 1;
    for (;;) {
      const unsigned long long c = __hip_atomic_load(checkin, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      if (__builtin_amdgcn_readfirstlane((unsigned)(c >= need)) != 0u) break;
      const unsigned bad = __hip_atomic_load(&ctl[4], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 0u ||
                           fn120b_now() - t0 > timeout;
      if (__builtin_amdgcn_readfirstlane(bad) != 0u) {
        __hip_atomic_fetch_add(&ctl[4], one, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        ok = 0;
        break;
      }
      __builtin_amdgcn_s_sleep(1);
    }
    const unsigned b = (unsigned)(e & 1ull);
    if (ok && ar == e * G) {
      // the first arriver: the other parity's words for launch e + 1 (lane j zeroes word j)
      const unsigned o = b ^ 1u;
      if (lane == 0u) __hip_atomic_store(&ctl[2 + o], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      if (lane < 3u * DF_STRIPES)
        __hip_atomic_store(&ctl[DF_CTR0 + o * 3u * DF_STRIPES + lane], 0u, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_fetch_add(&ctl[6], one, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    fn120b_ok = ok;
    fn120b_par = b;
  }
  __syncthreads();
  if (__builtin_amdgcn_readfirstlane(fn120b_ok) == 0u) return;
  const unsigned b = __builtin_amdgcn_readfirstlane(fn120b_par);
  unsigned *ticket = &ctl[2 + b];
  unsigned *ctr = ctl + DF_CTR0 + b * 3u * DF_STRIPES;
  for (;;) {
    if (w0) {
      const unsigned v = __hip_atomic_fetch_add(ticket, one, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      fn120b_ticket = __builtin_amdgcn_readfirstlane(v);
    }
    __syncthreads();
    const unsigned t = __builtin_amdgcn_readfirstlane(fn120b_ticket);
    if (t >= ntasks) break;
    const unsigned kst = __builtin_amdgcn_readfirstlane(tk[t].stage);
    const unsigned kvb = __builtin_amdgcn_readfirstlane(tk[t].vb);
    const unsigned kdep = __builtin_amdgcn_readfirstlane(tk[t].dep);
    const unsigned kneed = __builtin_amdgcn_readfirstlane(tk[t].need);
    if (kdep != DF_NONE) {
      if (w0) fn120b_ok = __builtin_amdgcn_readfirstlane(
                  fn120b_wait(&ctr[kdep * DF_STRIPES], kneed, timeout) ? 1u : 0u);
      __syncthreads();
      if (__builtin_amdgcn_readfirstlane(fn120b_ok) == 0u) {
        if (w0) __hip_atomic_fetch_add(&ctl[5], one, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        break;
      }
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    const unsigned skind = __builtin_amdgcn_readfirstlane(st[kst].kind);
    const unsigned sthr = __builtin_amdgcn_readfirstlane(st[kst].thr);
    const unsigned sgx = __builtin_amdgcn_readfirstlane(st[kst].gx);
    const unsigned sgy = __builtin_amdgcn_readfirstlane(st[kst].gy);
    const unsigned sgz = __builtin_amdgcn_readfirstlane(st[kst].gz);
    const unsigned simg = __builtin_amdgcn_readfirstlane(st[kst].img);
    if (w0) {
      fn120b_vbid.x = kvb % sgx;
      fn120b_vbid.y = (kvb / sgx) % sgy;
      fn120b_vbid.z = kvb / (sgx * sgy);
      fn120b_vgdim.x = sgx;
      fn120b_vgdim.y = sgy;
      fn120b_vgdim.z = sgz;
      fn120b_vthr = sthr;
    }
    __syncthreads();
    const unsigned char *img = imgs + simg;
    if constexpr (MODE != 2) {
      if (skind == DF_SFX) {
        if constexpr (INL) fn120b_body_i<fn86_t8_q8w8_sfx_ldc>(img); else fn120b_body_n<fn86_t8_q8w8_sfx_ldc>(img);
      } else if (skind == DF_N64) {
        if constexpr (INL) fn120b_body_i<fn111_t8_a8w8_n64_ldc>(img); else fn120b_body_n<fn111_t8_a8w8_n64_ldc>(img);
      } else if (MODE == 0 && skind == DF_GDN) {
        if constexpr (MODE == 0) if constexpr (INL) fn120b_body_i<fn86_gdn_fused_update_nq8_fc_f16s_w4k128v128_ab1>(img); else fn120b_body_n<fn86_gdn_fused_update_nq8_fc_f16s_w4k128v128_ab1>(img);
      }
    }
    // the release: every thread's stores, then one atomic on the task's stripe
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (w0)
      __hip_atomic_fetch_add(&ctr[kst * DF_STRIPES + (kvb & (DF_STRIPES - 1u))], one, __ATOMIC_RELEASE,
                             __HIP_MEMORY_SCOPE_AGENT);
  }
}

extern "C" __global__ __launch_bounds__(256) void fn120_df_gdn_b1(
    const Df120Stage *__restrict__ st, const Df120Task *__restrict__ tk, unsigned ntasks,
    const unsigned char *__restrict__ imgs, unsigned *__restrict__ ctl, unsigned nstages,
    unsigned long long timeout) {
  fn120b_run<0>(st, tk, ntasks, imgs, ctl, nstages, timeout);
}

// The PROBE's form (card 1, tools/capture/fn120/fn120c_probe.py): the same runtime with the two
// GEMM bodies only (a DF_GDN task runs nothing) -- residency and the GEMM bodies without the
// update's registers / scratch. Nothing in the engine launches it.
extern "C" __global__ __launch_bounds__(256) void fn120_df_gemm_probe(
    const Df120Stage *__restrict__ st, const Df120Task *__restrict__ tk, unsigned ntasks,
    const unsigned char *__restrict__ imgs, unsigned *__restrict__ ctl, unsigned nstages,
    unsigned long long timeout) {
  fn120b_run<1>(st, tk, ntasks, imgs, ctl, nstages, timeout);
}

// The PROBE's no-body form (any block size up to 256): the runtime's loop alone.
extern "C" __global__ __launch_bounds__(256) void fn120_df_nobody_probe(
    const Df120Stage *__restrict__ st, const Df120Task *__restrict__ tk, unsigned ntasks,
    const unsigned char *__restrict__ imgs, unsigned *__restrict__ ctl, unsigned nstages,
    unsigned long long timeout) {
  fn120b_run<2>(st, tk, ntasks, imgs, ctl, nstages, timeout);
}
