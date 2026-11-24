// Inkling 2 (inkling2_loop) prefill GEMM re-tiles, INK15 (docs/INKLING2.md section 9 INK15).
// Saber's own.
//
// The INK15 prefill census (tools/capture/ink15/pcensus.py) put 63 % of a short prompt's prefill in
// the correctness-first dense GEMMs (ink_loop's ink_gemm_f32_bf16, ink_w8's ink_gemm_f32_w8): one
// 64 x 64 tile a workgroup, so a 26-row chunk at N 1,536 ran 24 workgroups (of 64 CUs) through 72
// serial k-stages each, ~225 us a launch whatever the shape. These are the SAME arithmetic on a
// tile sized to the shape (the host picks TM x TN so the grid fills the card), branchless staging
// (clamped indices; a clamped row's outputs are never stored) and the next k-stage's loads in
// registers while the current one is summed:
//
//   every output ONE sequential f32 sum over k = 0..K-1 of a * w, each product and each add
//   rounded on its own (-ffp-contract=off), from 0.f, then C = acc ? C + s : s
//
// == the parents' law, so every output is BYTE-equal to the parent's (and to the decode GEMVs',
// whose order is the parent's): gated on the card by the engine's tf rows vs the control.
//
//   ink_gemm_pf_f32_bf16_t<TM>x<TN>   A [M, K] f32 . W [N, K]^T bf16
//   ink_gemm_pf_f32_w8_t<TM>x<TN>     the 8-bit record: w = (float)code * (float)scale, exact
//
// Signatures = the parents' (SIG_INK_GEMM / SIG_INK_W8_GEMM). Tiles: 16x32, 16x64, 32x32, 32x64,
// 64x64; 256 threads; RM x RN outputs a thread; 32 k a stage.
#include <hip/hip_runtime.h>

__device__ __forceinline__ float pf_bf(unsigned short u) {
  return __builtin_bit_cast(float, ((unsigned)u) << 16);
}

#define PF_K 32

template <int TM, int TN, bool W8>
__device__ __forceinline__ void pf_body(const float* __restrict__ a, int lda,
                                        const void* __restrict__ wv, int ldw,
                                        const unsigned short* __restrict__ sc, int ldsc,
                                        float* __restrict__ c, int ldc, int M, int N, int K,
                                        int acc) {
  constexpr int NX = (TM * TN >= 4096) ? 16 : (TN / ((TM * TN) / 256 >= 4 ? 4 : 2));
  constexpr int RN = TN / NX;
  constexpr int NY = 256 / NX;
  constexpr int RM = TM / NY;
  static_assert(NX * NY == 256 && RM * NY == TM && RN * NX == TN, "tile");
  constexpr int LA = TM * PF_K / 256;  // A elements a thread a stage
  constexpr int LW = TN * PF_K / 256;  // W elements a thread a stage
  __shared__ float as[PF_K][TM + 1];
  __shared__ float ws[PF_K][TN + 1];
  const int tid = threadIdx.x, tx = tid % NX, ty = tid / NX;
  const int m0 = blockIdx.y * TM, n0 = blockIdx.x * TN;
  const unsigned short* wb = (const unsigned short*)wv;
  const signed char* w8 = (const signed char*)wv;
  float s[RM][RN];
#pragma unroll
  for (int i = 0; i < RM; ++i)
#pragma unroll
    for (int j = 0; j < RN; ++j) s[i][j] = 0.f;
  // the clamped row / column of each staged element (a clamped row's sums are never stored)
  long aoff[LA], woff[LW];
  long soff[LW];
#pragma unroll
  for (int i = 0; i < LA; ++i) {
    const int p = tid + i * 256, r = p >> 5, kk = p & 31;
    const int gm = min(m0 + r, M - 1);
    aoff[i] = (long)gm * lda + kk;
  }
#pragma unroll
  for (int i = 0; i < LW; ++i) {
    const int p = tid + i * 256, r = p >> 5, kk = p & 31;
    const int gn = min(n0 + r, N - 1);
    woff[i] = (long)gn * ldw + kk;
    soff[i] = (long)gn * ldsc;
  }
  float ra[LA], rw[LW];
  auto load = [&](int k0) {
#pragma unroll
    for (int i = 0; i < LA; ++i) ra[i] = a[aoff[i] + k0];
#pragma unroll
    for (int i = 0; i < LW; ++i) {
      if constexpr (W8) {
        rw[i] = (float)(int)w8[woff[i] + k0] * pf_bf(sc[soff[i] + (k0 >> 5)]);
      } else {
        rw[i] = pf_bf(wb[woff[i] + k0]);
      }
    }
  };
  load(0);
  for (int k0 = 0; k0 < K; k0 += PF_K) {
#pragma unroll
    for (int i = 0; i < LA; ++i) {
      const int p = tid + i * 256;
      as[p & 31][p >> 5] = ra[i];
    }
#pragma unroll
    for (int i = 0; i < LW; ++i) {
      const int p = tid + i * 256;
      ws[p & 31][p >> 5] = rw[i];
    }
    __syncthreads();
    if (k0 + PF_K < K) load(k0 + PF_K);
#pragma unroll 4
    for (int kk = 0; kk < PF_K; ++kk) {
      float av[RM], wq[RN];
#pragma unroll
      for (int i = 0; i < RM; ++i) av[i] = as[kk][ty * RM + i];
#pragma unroll
      for (int j = 0; j < RN; ++j) wq[j] = ws[kk][tx * RN + j];
#pragma unroll
      for (int i = 0; i < RM; ++i)
#pragma unroll
        for (int j = 0; j < RN; ++j) s[i][j] = s[i][j] + av[i] * wq[j];
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < RM; ++i) {
    const int gm = m0 + ty * RM + i;
    if (gm >= M) continue;
#pragma unroll
    for (int j = 0; j < RN; ++j) {
      const int gn = n0 + tx * RN + j;
      if (gn >= N) continue;
      float* p = c + (long)gm * ldc + gn;
      *p = acc ? *p + s[i][j] : s[i][j];
    }
  }
}

#define PF_BF16(TM, TN)                                                                          \
  extern "C" __global__ __launch_bounds__(256) void ink_gemm_pf_f32_bf16_t##TM##x##TN(          \
      const float* __restrict__ a, int lda, const unsigned short* __restrict__ w, int ldw,      \
      float* __restrict__ c, int ldc, int M, int N, int K, int acc) {                           \
    pf_body<TM, TN, false>(a, lda, w, ldw, nullptr, 0, c, ldc, M, N, K, acc);                    \
  }
#define PF_W8(TM, TN)                                                                            \
  extern "C" __global__ __launch_bounds__(256) void ink_gemm_pf_f32_w8_t##TM##x##TN(            \
      const float* __restrict__ a, int lda, const unsigned char* __restrict__ w, int ldw,       \
      const unsigned short* __restrict__ sc, int ldsc, float* __restrict__ c, int ldc, int M,   \
      int N, int K, int acc) {                                                                  \
    pf_body<TM, TN, true>(a, lda, w, ldw, sc, ldsc, c, ldc, M, N, K, acc);                       \
  }

PF_BF16(16, 32)
PF_BF16(16, 64)
PF_BF16(32, 32)
PF_BF16(32, 64)
PF_BF16(64, 64)
PF_W8(16, 32)
PF_W8(16, 64)
PF_W8(32, 32)
PF_W8(32, 64)
PF_W8(64, 64)

// ------------------------------------------------------------------------------------------------
// ink_p2p_xchg_v4: ink_p2p's ink_p2p_xchg (the same link protocol, word for word: arrive / spin /
// system-scope acquire / last-arriver depart) with the operand moved 16 bytes a load. The census:
// the parent reads the peer one 4-byte system-scope load an element (a 26-row reduce = 60 k
// PCIe read requests, ~90 us; request-count bound). After the system-scope acquire fence (the
// peer published its operand with a system-scope release before raising our arrive word), plain
// loads observe the published operand (the memory model's fence-fence synchronisation), so each
// thread reads float4s. Values are the parent's: sum = own + peer per element (f32, one add),
// else a copy. Needs cols % 4 == 0, every row stride % 4 == 0 and 16-byte aligned bases (the host
// checks; otherwise it launches the parent).
extern "C" __global__ __launch_bounds__(256) void ink_p2p_xchg_v4(
    const float* __restrict__ own, int ld_own, const float* peer, int ld_peer, float* __restrict__ out,
    int ld_out, int rows, int n0, int n1, int sum, unsigned* link, unsigned* peer_link) {
  __shared__ unsigned s_seq;
  const int tid = threadIdx.x;
  if (__builtin_amdgcn_readfirstlane(tid >> 5) == 0) {
    const unsigned s = __hip_atomic_load(&link[0], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1u;
    if (blockIdx.x == 0) {
      __atomic_thread_fence(__ATOMIC_RELEASE);
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
      if (tid == 0) __hip_atomic_store(&peer_link[2], s, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    while ((int)(__hip_atomic_load(&link[2], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) - s) < 0)
      __builtin_amdgcn_s_sleep(2);
    if (tid == 0) s_seq = s;
  }
  __syncthreads();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  const unsigned s = s_seq;
  const long cols = sum ? n0 : (long)n0 + n1;
  const long c4 = cols >> 2;
  const long total = (long)rows * c4;
  for (long i = (long)blockIdx.x * 256 + tid; i < total; i += (long)gridDim.x * 256) {
    const long r = i / c4, c = (i - r * c4) << 2;
    float4 v;
    if (sum) {
      const float4 p = *(const float4*)&peer[r * ld_peer + c];
      const float4 o = *(const float4*)&own[r * ld_own + c];
      v.x = o.x + p.x;
      v.y = o.y + p.y;
      v.z = o.z + p.z;
      v.w = o.w + p.w;
    } else if (c < n0) {
      v = *(const float4*)&own[r * ld_own + c];
    } else {
      v = *(const float4*)&peer[r * ld_peer + (c - n0)];
    }
    *(float4*)&out[r * ld_out + c] = v;
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
      while ((int)(__hip_atomic_load(&link[3], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) - s) < 0)
        __builtin_amdgcn_s_sleep(2);
    }
  }
}
