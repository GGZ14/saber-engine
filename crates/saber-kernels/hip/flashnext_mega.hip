// FN23 (2026-09-29): the PERSISTENT DECODE MEGAKERNEL feasibility prototype -- ONE Flash-Next
// GDN + MoE + HC decoder layer (TP=2 rank 0 at b1 / b8, SPEC 0) as persistent kernels that walk a
// device-side stage table, NOT a product (no plan binds it; nothing in the engine launches it).
//
// THE BIT-EXACT LAW, BY CONSTRUCTION: there is NO copy of any certified body here. Every stage calls
// the SHIPPED kernel entry point itself -- the same function text the launch train's code objects
// were compiled from -- by `#include`-ing the committed sources verbatim (flashnext_hc.hip,
// dflash_bgemm.hip, moe_route.hip, moe_grouped.hip, saber_gdn_fused_update_f16s.hip, saber_gdn_conv.hip,
// saber_gdn_recurrent_update_f16s.hip, saber_ar_exact.hip) with FOUR macros in force around the includes:
//   * `__global__`        -> `__device__`: every shipped kernel becomes a callable device function;
//   * `__launch_bounds__` -> nothing (it only binds a kernel's register budget, never arithmetic);
//   * `blockIdx` / `gridDim` / `blockDim` -> the VIRTUAL block the persistent workgroup is running
//     (an LDS triple written by thread 0, read back through readfirstlane: uniform, scalar);
//   * `__syncthreads()`   -> `fn23_sync()`: the hardware barrier when the stage uses every wave of
//     the persistent workgroup, an LDS generation barrier over the stage's own waves otherwise
//     (a 96/128/224-thread stage runs on waves 0..n-1 of a 256-thread workgroup).
// Each include sits under the fp-contract state its code object is built with (saber-kbuild's
// EXTRA_FLAGS: `-ffp-contract=off` for flashnext_hc / moe_route / saber_*, the hipcc default for
// moe_grouped / dflash_bgemm), so a product is contracted exactly where the launch train's is.
// The kernel arguments are unpacked from the SAME kernarg image the launch train hands
// hipModuleLaunchKernel (`fn23_call<F>`: offsets by the kernarg ABI's natural alignment, checked
// against the manifests on the card before any gate by `fn23_offsets`).
//
// THE SCHEDULE: a stage = (kind, the train's grid + block, kernarg image, flags). Workgroup w runs
// virtual blocks w, w + G, ... of the stage; a stage with the BAR flag ends in a grid-wide barrier
// (arrive counter + generation word, `s_sleep` spin, agent-scope release/acquire). A stage whose
// body carries its own grid barrier (the fused GDN update, 24 workgroups) needs G >= its grid: the
// host asserts it. SEED stages replace the train's memset nodes (zero / 0xFF regions).
//
// WHY FOUR KERNELS AND NOT ONE (FN23's first finding): static LDS of every inlined body is SUMMED
// by the AMDGPU backend (no liveness overlap across stages; `lds_test.hip` in the LEDGER): the b1
// layer's bodies hold 76,368 B > the 65,536 B workgroup limit. So a layer is TWO persistent kernels
// here -- `_a` (HC mixes, the GDN mixer, out_proj, the attention reduce, the router + route: 36.9 KB
// b1 / 57.4 KB b8) and `_b` (grouped gate_up, silu-quant, grouped down, combine, the MoE reduce:
// 39.4 KB). One kernel per step needs the bodies' LDS as a union (a mechanical
// `__shared__` -> arena-pointer refactor of six certified sources, byte-gated): a productization item.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include <cstdint>

// ------------------------------------------------------------------------------ the virtual block
struct fn23_dim3 {
  unsigned int x, y, z;
};
__shared__ fn23_dim3 fn23_vbid, fn23_vgdim, fn23_vbdim;
__shared__ unsigned int fn23_sync_cnt, fn23_sync_gen, fn23_nwaves;

__device__ __forceinline__ fn23_dim3 fn23_rfl(const fn23_dim3 &d) {
  fn23_dim3 r;
  r.x = __builtin_amdgcn_readfirstlane(d.x);
  r.y = __builtin_amdgcn_readfirstlane(d.y);
  r.z = __builtin_amdgcn_readfirstlane(d.z);
  return r;
}

// The hardware barrier with the workgroup-scope fences `__syncthreads` carries.
__device__ __forceinline__ void fn23_hw_sync() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

// `__syncthreads()` inside a stage body: every wave of the workgroup -> the hardware barrier; a
// narrower stage -> a generation barrier over its own waves in LDS (lane 0 of each wave arrives).
__device__ __forceinline__ void fn23_sync() {
  const unsigned nw = __builtin_amdgcn_readfirstlane(fn23_nwaves);
  if (nw * 32u == blockDim.x) {
    fn23_hw_sync();
    return;
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  if ((threadIdx.x & 31u) == 0) {
    const unsigned gen = __hip_atomic_load(&fn23_sync_gen, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
    if (__hip_atomic_fetch_add(&fn23_sync_cnt, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_WORKGROUP) == nw - 1u) {
      __hip_atomic_store(&fn23_sync_cnt, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
      __hip_atomic_store(&fn23_sync_gen, gen + 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
    } else {
      while (__hip_atomic_load(&fn23_sync_gen, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP) == gen)
        __builtin_amdgcn_s_sleep(1);
    }
  }
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

// ------------------------------------------------------------------ the certified sources, verbatim
#pragma push_macro("__global__")
#pragma push_macro("__launch_bounds__")
#undef __global__
#undef __launch_bounds__
#define __global__ __device__
#define __launch_bounds__(...)
#define blockIdx (fn23_rfl(fn23_vbid))
#define gridDim (fn23_rfl(fn23_vgdim))
#define blockDim (fn23_rfl(fn23_vbdim))
#define __syncthreads() fn23_sync()

#pragma clang fp contract(off)
#include "flashnext_hc.hip"
#pragma clang fp contract(off)
#include "moe_route.hip"
#pragma clang fp contract(off)
#include "saber_gdn_fused_update_f16s.hip"
#pragma clang fp contract(off)
#include "saber_gdn_conv.hip"
#pragma clang fp contract(off)
#include "saber_gdn_recurrent_update_f16s.hip"
#pragma clang fp contract(off)
#include "saber_ar_exact.hip"
#pragma clang fp contract(fast)
#include "dflash_bgemm.hip"
#pragma clang fp contract(fast)
#include "moe_grouped.hip"
#pragma clang fp contract(off)

#undef __syncthreads
#undef blockDim
#undef gridDim
#undef blockIdx
#undef __launch_bounds__
#undef __global__
#pragma pop_macro("__launch_bounds__")
#pragma pop_macro("__global__")

// ------------------------------------------------------------------------------ argument unpacking
// A kernarg image is the ABI's: each argument at the next multiple of its own alignment.
template <typename... A> struct fn23_layout {
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

template <typename T> __device__ __forceinline__ T fn23_ld(const unsigned char *p) {
  T v;
  __builtin_memcpy(&v, p, sizeof(T));
  return v;
}

template <auto F> struct fn23_call;
template <typename... A, void (*F)(A...)> struct fn23_call<F> {
  template <unsigned... I>
  __device__ __forceinline__ static void go_(const unsigned char *img, std::integer_sequence<unsigned, I...>) {
    constexpr auto o = fn23_layout<A...>::offs();
    F(fn23_ld<A>(img + o.v[I])...);
  }
  __device__ __forceinline__ static void go(const unsigned char *img) {
    go_(img, std::make_integer_sequence<unsigned, sizeof...(A)>{});
  }
  // The offsets, for the card-side pre-flight against the manifests.
  __device__ __forceinline__ static void offsets(unsigned *out) {
    constexpr auto o = fn23_layout<A...>::offs();
    out[0] = sizeof...(A);
    for (unsigned i = 0; i < sizeof...(A); ++i) out[1 + i] = o.v[i];
  }
};

// ------------------------------------------------------------------------------ the stage table
// Kinds. The host (tools/capture/fn23/fn23_bench.cpp, crates/saber-kernels/src/fn23.rs) holds the
// same numbering and the symbol of each kind.
enum : unsigned {
  K_NOP = 0,          // no body (the schedule-only control: barriers + virtual blocks)
  K_SEED = 1,         // aux0 = address lo/hi in img (8 B), aux1 = bytes, flags FILL_ONES
  K_HC_DOWN_M1W3 = 2, // fn_hc_down_combine_v2_m1w3t1
  K_HC_DOWN_M1W7 = 3, // fn_hc_down_combine_v2_m1w7t1
  K_HC_UP_BF16 = 4,   // fn_hc_up_bf16_v2_s1
  K_HC_UP_Q8 = 5,     // fn_hc_up_q8_v2_s1
  K_BGEMM_SK16 = 6,   // fn_bgemm_sk_m16
  K_DBGEMM_16 = 7,    // dflash_bgemm_bf16_m16n16
  K_GDN_FUSED = 8,    // saber_gdn_fused_update_f16s_w4k128v128_ab1
  K_GDN_CONV = 9,     // saber_gdn_conv_update_w4_h128_hpb8
  K_GDN_REC = 10,     // saber_gdn_recurrent_update_f16s_k128_v128_ab1_norm0_vs1
  K_GDN_NORM = 11,    // fn_gdn_norm_bf16_sig_t256
  K_AR_EXACT = 12,    // saber_ar_oneshot_2rank_exact_bf16
  K_ROUTE = 13,       // moe_route_topk10_sort_e512
  K_GU_KS4 = 14,      // moe_grouped_w4a8_bk128_ks4_tm1_g1
  K_GU_KS1 = 15,      // moe_grouped_w4a8_bk128_ks1_tm1_g1
  K_SILU = 16,        // moe_silu_mul_quant_t256
  K_DN_KS2 = 17,      // moe_grouped_w4a8_bk128_ks2_tm1_g0
  K_DN_KS1 = 18,      // moe_grouped_w4a8_bk128_ks1_tm1_g0
  K_COMBINE = 19,     // moe_combine_bf16
  K_COUNT = 20
};
enum : unsigned { F_BAR = 1u, F_ONES = 2u };

struct fn23_stage {
  unsigned kind, threads;
  unsigned grid[3];
  unsigned img;  // byte offset of the kernarg image (SEED: of the {addr u64, bytes u64} pair)
  unsigned flags;
  unsigned pad;
};

// The grid barrier: every wave releases its stores at agent scope, lane 0 of wave 0 arrives; the
// last arrival resets the count and bumps the generation; the others spin on it with s_sleep.
__device__ __forceinline__ void fn23_grid_sync(unsigned *gbar, unsigned G) {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  fn23_hw_sync();
  if (threadIdx.x == 0) {
    const unsigned gen = __hip_atomic_load(&gbar[1], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    if (__hip_atomic_fetch_add(&gbar[0], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) == G - 1u) {
      __hip_atomic_store(&gbar[0], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(&gbar[1], gen + 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    } else {
      while (__hip_atomic_load(&gbar[1], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == gen)
        __builtin_amdgcn_s_sleep(1);
    }
  }
  fn23_hw_sync();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}

__device__ __forceinline__ void fn23_seed(const unsigned char *img, unsigned flags, unsigned G) {
  const uint64_t addr = fn23_ld<uint64_t>(img), bytes = fn23_ld<uint64_t>(img + 8);
  const uint32_t v = (flags & F_ONES) ? 0xFFFFFFFFu : 0u;
  uint32_t *p = reinterpret_cast<uint32_t *>(addr);
  const uint64_t n = bytes / 4, stride = (uint64_t)G * blockDim.x;
  for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) p[i] = v;
}

// One stage body as its OWN function (noinline): the register allocator sees one certified body at
// a time, so the persistent kernel's VGPR count is the widest stage's, not the sum of the live
// ranges of every body inlined into one switch (which spilled 79 VGPRs in kernel `_a` at b1).
template <auto F> __device__ __attribute__((noinline)) void fn23_stage_call(const unsigned char *img) {
  fn23_call<F>::go(img);
}

// The dispatch of one kernel SET (the set's kinds only: an uncalled body's LDS is never allocated):
// `FN23_CASE` = each body its own function, `FN23_CASE_INL` = every body inlined into the walk.
#define FN23_CASE(K, SYM)                                                                           \
  case K:                                                                                           \
    fn23_stage_call<SYM>(img);                                                                      \
    break;
#define FN23_CASE_INL(K, SYM)                                                                       \
  case K:                                                                                           \
    fn23_call<SYM>::go(img);                                                                        \
    break;

#define FN23_SET_A_B1(X)                                                                            \
  X(K_HC_DOWN_M1W3, fn_hc_down_combine_v2_m1w3t1)                                                   \
  X(K_HC_UP_BF16, fn_hc_up_bf16_v2_s1)                                                              \
  X(K_HC_UP_Q8, fn_hc_up_q8_v2_s1)                                                                  \
  X(K_BGEMM_SK16, fn_bgemm_sk_m16)                                                                  \
  X(K_GDN_FUSED, saber_gdn_fused_update_f16s_w4k128v128_ab1)                                          \
  X(K_GDN_NORM, fn_gdn_norm_bf16_sig_t256)                                                          \
  X(K_AR_EXACT, saber_ar_oneshot_2rank_exact_bf16)                                                    \
  X(K_ROUTE, moe_route_topk10_sort_e512)
#define FN23_SET_B_B1(X)                                                                            \
  X(K_GU_KS4, moe_grouped_w4a8_bk128_ks4_tm1_g1)                                                    \
  X(K_SILU, moe_silu_mul_quant_t256)                                                                \
  X(K_DN_KS2, moe_grouped_w4a8_bk128_ks2_tm1_g0)                                                    \
  X(K_COMBINE, moe_combine_bf16)                                                                    \
  X(K_AR_EXACT, saber_ar_oneshot_2rank_exact_bf16)
#define FN23_SET_A_B8(X)                                                                            \
  X(K_HC_DOWN_M1W7, fn_hc_down_combine_v2_m1w7t1)                                                   \
  X(K_HC_UP_BF16, fn_hc_up_bf16_v2_s1)                                                              \
  X(K_HC_UP_Q8, fn_hc_up_q8_v2_s1)                                                                  \
  X(K_BGEMM_SK16, fn_bgemm_sk_m16)                                                                  \
  X(K_DBGEMM_16, dflash_bgemm_bf16_m16n16)                                                          \
  X(K_GDN_CONV, saber_gdn_conv_update_w4_h128_hpb8)                                                   \
  X(K_GDN_REC, saber_gdn_recurrent_update_f16s_k128_v128_ab1_norm0_vs1)                               \
  X(K_GDN_NORM, fn_gdn_norm_bf16_sig_t256)                                                          \
  X(K_AR_EXACT, saber_ar_oneshot_2rank_exact_bf16)                                                    \
  X(K_ROUTE, moe_route_topk10_sort_e512)
#define FN23_SET_B_B8(X)                                                                            \
  X(K_GU_KS1, moe_grouped_w4a8_bk128_ks1_tm1_g1)                                                    \
  X(K_SILU, moe_silu_mul_quant_t256)                                                                \
  X(K_DN_KS1, moe_grouped_w4a8_bk128_ks1_tm1_g0)                                                    \
  X(K_COMBINE, moe_combine_bf16)                                                                    \
  X(K_AR_EXACT, saber_ar_oneshot_2rank_exact_bf16)

// The persistent walk: stages [s0, s1) of the table, G = gridDim.x resident workgroups of 256.
// `ts` (optional): workgroup 0 stamps the constant wall clock (`wall_clock64`, hipDeviceAttributeWallClockRate) at entry (ts[s1]) and after
// each stage (ts[s], after its grid barrier when it has one) -- the per-stage device breakdown.
#define FN23_MEGA(NAME, SET, CASE)                                                                  \
  extern "C" __global__ __launch_bounds__(256) void NAME(                                           \
      const fn23_stage *__restrict__ st, int s0, int s1, const unsigned char *__restrict__ imgs,   \
      unsigned *__restrict__ gbar, unsigned long long *__restrict__ ts) {                           \
    const unsigned G = gridDim.x, w = blockIdx.x;                                                   \
    if (ts && w == 0 && threadIdx.x == 0) ts[s1] = wall_clock64();                \
    for (int s = s0; s < s1; ++s) {                                                                 \
      const unsigned kind = __builtin_amdgcn_readfirstlane(st[s].kind);                            \
      const unsigned thr = __builtin_amdgcn_readfirstlane(st[s].threads);                          \
      const unsigned gx = st[s].grid[0], gy = st[s].grid[1], gz = st[s].grid[2];                    \
      const unsigned flags = __builtin_amdgcn_readfirstlane(st[s].flags);                          \
      const unsigned char *img = imgs + st[s].img;                                                  \
      if (kind == K_SEED) {                                                                         \
        fn23_seed(img, flags, G);                                                                   \
      } else if (kind != K_NOP) {                                                                   \
        const unsigned nvb = gx * gy * gz;                                                          \
        for (unsigned vb = w; vb < nvb; vb += G) {                                                  \
          if (threadIdx.x == 0) {                                                                   \
            fn23_vbid.x = vb % gx;                                                                  \
            fn23_vbid.y = (vb / gx) % gy;                                                           \
            fn23_vbid.z = vb / (gx * gy);                                                           \
            fn23_vgdim.x = gx;                                                                      \
            fn23_vgdim.y = gy;                                                                      \
            fn23_vgdim.z = gz;                                                                      \
            fn23_vbdim.x = thr;                                                                     \
            fn23_vbdim.y = 1;                                                                       \
            fn23_vbdim.z = 1;                                                                       \
            fn23_nwaves = (thr + 31u) / 32u;                                                        \
            fn23_sync_cnt = 0;                                                                      \
          }                                                                                         \
          fn23_hw_sync();                                                                           \
          if (threadIdx.x < thr) {                                                                  \
            switch (kind) { SET(CASE) default: break; }                                             \
          }                                                                                         \
          fn23_hw_sync();                                                                           \
        }                                                                                           \
      }                                                                                             \
      if (flags & F_BAR) fn23_grid_sync(gbar, G);                                                   \
      if (ts && w == 0 && threadIdx.x == 0) ts[s] = wall_clock64();               \
    }                                                                                               \
  }

FN23_MEGA(fn23_mega_a_b1, FN23_SET_A_B1, FN23_CASE)
FN23_MEGA(fn23_mega_a_b1_inl, FN23_SET_A_B1, FN23_CASE_INL)
FN23_MEGA(fn23_mega_b_b1, FN23_SET_B_B1, FN23_CASE)
FN23_MEGA(fn23_mega_b_b1_inl, FN23_SET_B_B1, FN23_CASE_INL)
FN23_MEGA(fn23_mega_a_b8, FN23_SET_A_B8, FN23_CASE)
FN23_MEGA(fn23_mega_b_b8, FN23_SET_B_B8, FN23_CASE)
FN23_MEGA(fn23_mega_b_b8_inl, FN23_SET_B_B8, FN23_CASE_INL)

// The card-side pre-flight: the argument offsets `fn23_call` unpacks for every kind, [kind][0] =
// argument count, [kind][1 + i] = offset of argument i (64 words a kind). The bench compares them
// with the manifests' offsets before any gate runs.
#define FN23_OFFS(K, SYM)                                                                           \
  fn23_call<SYM>::offsets(out + 64 * (K));
extern "C" __global__ __launch_bounds__(64) void fn23_offsets(unsigned *__restrict__ out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  FN23_SET_A_B1(FN23_OFFS)
  FN23_SET_B_B1(FN23_OFFS)
  FN23_SET_A_B8(FN23_OFFS)
  FN23_SET_B_B8(FN23_OFFS)
}
