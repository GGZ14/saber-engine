// Inkling 2 (inkling2_loop) prefill GEMM, column-lane form, INK16 (docs/INKLING2.md section 9
// INK16). Saber's own.
//
// INK15's re-tiles (ink_pf) left a short prompt's prefill instruction-bound: a 16 x 32 tile gives
// each thread 1 x 2 outputs, so every product pays ~1.5 LDS reads, and a 26-row chunk at N 1,536
// is 96 workgroups on 64 CUs (half of them run two). This form keeps the parents' arithmetic
// exactly --
//
//   every output ONE sequential f32 sum over k = 0..K-1 of a * w, each product and each add
//   rounded on its own (-ffp-contract=off), from 0.f, then C = acc ? C + s : s
//
// -- and changes only who sums what: a workgroup owns 32 output COLUMNS (one a lane) and 8 x RM
// rows (wave w sums rows w * RM .. w * RM + RM - 1 of every column), so a lane walks k in order for
// RM outputs of ONE column. Per four k a lane reads its column's four weights as one 16-byte LDS
// read and each row's four activations as one 16-byte read the whole wave shares (a broadcast);
// the weight is converted (bf16 -> f32, or code x scale for the 8-bit record) ONCE when staged.
// Staging: 64 k a stage, two LDS buffers (one LDS-only barrier a stage, no load drain), NB stages'
// global loads in a register ring; clamped indices (a clamped
// row / column is never stored). Rows past M: a wave whose rows all lie past M skips the sums.
//
//   ink_gemm_cl_f32_bf16_r<RM>n<RN>   A [M, K] f32 . W [N, K]^T bf16
//   ink_gemm_cl_f32_w8_r<RM>n<RN>     the 8-bit record: w = (float)code * (float)scale, exact
// (RN columns a lane, 32 RN a workgroup; four stages of global loads in flight in registers)
//
// Signatures = the parents' (SIG_INK_GEMM / SIG_INK_W8_GEMM). Needs K % 64 == 0, lda % 4 == 0,
// 16-byte aligned A, and for the weight: ldw % 8 (w8) / % 8 elements (bf16) with 8-/16-byte
// aligned rows (the host checks; otherwise it launches ink_pf). Grid [ceil(N / (32 RN)),
// ceil(M / (8 RM))], 256 threads.
#include <hip/hip_runtime.h>

typedef float cl_f4 __attribute__((ext_vector_type(4)));

#define CL_K 64
#define CL_LD (CL_K + 4)  // f32 a staged row (272 B: 16-byte aligned rows)
#define CL_LDS_BARRIER()                                                    \
  do {                                                                      \
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");         \
    __builtin_amdgcn_s_barrier();                                           \
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");         \
  } while (0)

__device__ __forceinline__ float cl_lo(unsigned u) { return __builtin_bit_cast(float, u << 16); }
__device__ __forceinline__ float cl_hi(unsigned u) {
  return __builtin_bit_cast(float, u & 0xFFFF0000u);
}
__device__ __forceinline__ float cl_code(unsigned wd, int b) {
  return (float)((int)(wd << (24 - 8 * b)) >> 24);
}

template <int RM, int RN, int NB, bool W8>
__device__ __forceinline__ void cl_body(const float* __restrict__ a, int lda,
                                        const void* __restrict__ wv, int ldw,
                                        const unsigned short* __restrict__ sc, int ldsc,
                                        float* __restrict__ c, int ldc, int M, int N, int K,
                                        int acc) {
  constexpr int TM = 8 * RM;
  constexpr int NA = (TM * CL_K / 4 + 255) / 256;  // A float4s a thread a stage
  __shared__ __attribute__((aligned(16))) float as[2][TM][CL_LD];
  __shared__ __attribute__((aligned(16))) float ws[2][32 * RN][CL_LD];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int m0 = blockIdx.y * TM, n0 = blockIdx.x * (32 * RN);
  // A staging: float4 e = tid + 256 i of the TM x 64 tile (row e / 16, k (e % 16) * 4)
  long aoff[NA];
  bool aon[NA];
#pragma unroll
  for (int i = 0; i < NA; ++i) {
    const int e = tid + 256 * i, r = e >> 4, kk = (e & 15) * 4;
    aon[i] = e < TM * 16;
    const int gm = min(m0 + min(r, TM - 1), M - 1);
    aoff[i] = (long)gm * lda + kk;
  }
  // W staging: 8 k of RN rows a thread (rows tid / 8 + 32 q, k (tid % 8) * 8)
  const int wr = tid >> 3, wk = (tid & 7) * 8;
  const unsigned char* w8p[RN];
  const unsigned short* wbp[RN];
  const unsigned short* scp[RN];
#pragma unroll
  for (int q = 0; q < RN; ++q) {
    const int gn = min(n0 + wr + 32 * q, N - 1);
    w8p[q] = (const unsigned char*)wv + (long)gn * ldw + wk;
    wbp[q] = (const unsigned short*)wv + (long)gn * ldw + wk;
    scp[q] = sc + (long)gn * ldsc + (wk >> 5);
  }
  // the register ring: NB stages' global loads in flight (stage kt's loads were issued NB - 1
  // stages earlier; INK16 h1: one stage ahead left every stage waiting on the load latency)
  cl_f4 ra[NB][NA];
  uint4 rw[NB][RN];
  unsigned rs[NB][RN];
  // CL_LOAD / CL_STAGE / CL_STEP take the ring slot J as a LITERAL (macros, not lambdas: a lambda
  // left the ring in scratch), so the ring never leaves registers
#define CL_LOAD(J, K0)                                                                \
  do {                                                                                \
    _Pragma("unroll") for (int i = 0; i < NA; ++i) ra[J][i] =                         \
        *reinterpret_cast<const cl_f4*>(a + aoff[i] + (K0));                         \
    _Pragma("unroll") for (int q = 0; q < RN; ++q) {                                  \
      if constexpr (W8) {                                                             \
        const uint2 v_ = *reinterpret_cast<const uint2*>(w8p[q] + (K0));              \
        rw[J][q] = make_uint4(v_.x, v_.y, 0u, 0u);                                    \
        rs[J][q] = scp[q][(K0) >> 5];                                                 \
      } else {                                                                        \
        rw[J][q] = *reinterpret_cast<const uint4*>(wbp[q] + (K0));                    \
      }                                                                               \
    }                                                                                 \
  } while (0)
#define CL_STAGE(J, B)                                                                \
  do {                                                                                \
    _Pragma("unroll") for (int i = 0; i < NA; ++i) if (aon[i]) {                      \
      const int e_ = tid + 256 * i;                                                   \
      *reinterpret_cast<cl_f4*>(&as[B][e_ >> 4][(e_ & 15) * 4]) = ra[J][i];          \
    }                                                                                 \
    _Pragma("unroll") for (int q = 0; q < RN; ++q) {                                  \
      float4 w0_, w1_;                                                                \
      if constexpr (W8) {                                                             \
        const float s_ = __builtin_bit_cast(float, rs[J][q] << 16);                   \
        w0_ = make_float4(cl_code(rw[J][q].x, 0) * s_, cl_code(rw[J][q].x, 1) * s_,   \
                          cl_code(rw[J][q].x, 2) * s_, cl_code(rw[J][q].x, 3) * s_);  \
        w1_ = make_float4(cl_code(rw[J][q].y, 0) * s_, cl_code(rw[J][q].y, 1) * s_,   \
                          cl_code(rw[J][q].y, 2) * s_, cl_code(rw[J][q].y, 3) * s_);  \
      } else {                                                                        \
        w0_ = make_float4(cl_lo(rw[J][q].x), cl_hi(rw[J][q].x), cl_lo(rw[J][q].y),    \
                          cl_hi(rw[J][q].y));                                         \
        w1_ = make_float4(cl_lo(rw[J][q].z), cl_hi(rw[J][q].z), cl_lo(rw[J][q].w),    \
                          cl_hi(rw[J][q].w));                                         \
      }                                                                               \
      *reinterpret_cast<float4*>(&ws[B][wr + 32 * q][wk]) = w0_;                      \
      *reinterpret_cast<float4*>(&ws[B][wr + 32 * q][wk + 4]) = w1_;                  \
    }                                                                                 \
  } while (0)
  float s[RM][RN];
#pragma unroll
  for (int i = 0; i < RM; ++i)
#pragma unroll
    for (int q = 0; q < RN; ++q) s[i][q] = 0.f;
  const bool live = m0 + wave * RM < M;  // wave-uniform
  const int KT = K / CL_K;
#define CL_PRO(J) \
  if ((J) < NB && (J) < KT) CL_LOAD(J, (J) * CL_K)
  CL_PRO(0); CL_PRO(1); CL_PRO(2); CL_PRO(3);
  if constexpr (NB > 4) { CL_PRO(4); CL_PRO(5); CL_PRO(6); CL_PRO(7); }
  // one LDS-only barrier a stage on two buffers: a wave writes buffer kt & 1 only after every
  // wave passed stage kt - 1's barrier, i.e. finished summing stage kt - 2 (the same buffer)
#define CL_STEP(J)                                                                    \
  do {                                                                                \
    const int kt = kt0 + (J);                                                         \
    if (kt < KT) {                                                                    \
      const int b = kt & 1;                                                           \
      CL_STAGE(J, b);                                                                 \
      if (kt + NB < KT) CL_LOAD(J, (kt + NB) * CL_K);                                 \
      CL_LDS_BARRIER();                                                               \
      if (live) {                                                                     \
        _Pragma("unroll 4") for (int k4 = 0; k4 < CL_K; k4 += 4) {                    \
          float4 w4[RN];                                                              \
          _Pragma("unroll") for (int q = 0; q < RN; ++q) w4[q] =                      \
              *reinterpret_cast<const float4*>(&ws[b][lane + 32 * q][k4]);            \
          _Pragma("unroll") for (int i = 0; i < RM; ++i) {                            \
            const float4 a4 = *reinterpret_cast<const float4*>(&as[b][wave * RM + i][k4]); \
            _Pragma("unroll") for (int q = 0; q < RN; ++q) {                          \
              s[i][q] = s[i][q] + a4.x * w4[q].x;                                     \
              s[i][q] = s[i][q] + a4.y * w4[q].y;                                     \
              s[i][q] = s[i][q] + a4.z * w4[q].z;                                     \
              s[i][q] = s[i][q] + a4.w * w4[q].w;                                     \
            }                                                                         \
          }                                                                           \
        }                                                                             \
      }                                                                               \
    }                                                                                 \
  } while (0)
  for (int kt0 = 0; kt0 < KT; kt0 += NB) {
    CL_STEP(0); CL_STEP(1); CL_STEP(2); CL_STEP(3);
    if constexpr (NB > 4) { CL_STEP(4); CL_STEP(5); CL_STEP(6); CL_STEP(7); }
  }
#undef CL_STEP
#undef CL_PRO
#undef CL_STAGE
#undef CL_LOAD
  if (live) {
#pragma unroll
    for (int q = 0; q < RN; ++q) {
      const int on = n0 + lane + 32 * q;
      if (on >= N) continue;
#pragma unroll
      for (int i = 0; i < RM; ++i) {
        const int gm = m0 + wave * RM + i;
        if (gm < M) {
          float* p = c + (long)gm * ldc + on;
          *p = acc ? *p + s[i][q] : s[i][q];
        }
      }
    }
  }
}

#define CL_BF16(RM, RN)                                                                  \
  extern "C" __global__ __launch_bounds__(256) void ink_gemm_cl_f32_bf16_r##RM##n##RN(         \
      const float* __restrict__ a, int lda, const unsigned short* __restrict__ w, int ldw,      \
      float* __restrict__ c, int ldc, int M, int N, int K, int acc) {                           \
    cl_body<RM, RN, 4, false>(a, lda, w, ldw, nullptr, 0, c, ldc, M, N, K, acc);                 \
  }
#define CL_W8(RM, RN)                                                                    \
  extern "C" __global__ __launch_bounds__(256) void ink_gemm_cl_f32_w8_r##RM##n##RN(           \
      const float* __restrict__ a, int lda, const unsigned char* __restrict__ w, int ldw,       \
      const unsigned short* __restrict__ sc, int ldsc, float* __restrict__ c, int ldc, int M,   \
      int N, int K, int acc) {                                                                  \
    cl_body<RM, RN, 4, true>(a, lda, w, ldw, sc, ldsc, c, ldc, M, N, K, acc);                    \
  }

CL_BF16(1, 1)
CL_BF16(2, 1)
CL_BF16(4, 1)
CL_BF16(8, 1)
CL_BF16(2, 2)
CL_BF16(4, 2)
CL_W8(1, 1)
CL_W8(2, 1)
CL_W8(4, 1)
CL_W8(8, 1)
CL_W8(2, 2)
CL_W8(4, 2)

// ------------------------------------------------------------------------------------------------
// ink_p2p_push_v5: the prefill reduce as a PUSH (INK16). INK15's ink_p2p_xchg_v4 PULLS the peer's
// partial over the link (16-byte reads: a read is a round trip, so the reduce ran ~3x its bytes'
// time at 85 rows). Here each rank WRITES its partial into the PEER's receive window (posted
// writes), raises the peer's arrive word once every workgroup's stores are released at system
// scope (a last-arriver count), waits for the peer's arrive word, takes the system-scope acquire in
// every reading thread (FN71b: remote writes into our VRAM are stale behind our caches
// otherwise), and sums own + received locally: out = own + peer per element (f32, one add) == the
// v4 value (an IEEE add is commutative, so both ranks hold the same bytes, and == v4's).
// The window has two slots chosen by the call's GLOBAL count s (FN114d: never by blockIdx): our
// write of slot s & 1 at call s lands after the peer arrived at call s - 1, i.e. after it finished
// reading that slot at call s - 2 (kernel order on its stream). The link words are v4's: s =
// link[0] + 1, the arrive word link[2] (the peer writes ours), link[1] the local count; no depart
// phase (no one reads our operand). Grid <= 128 workgroups (co-resident: every workgroup spins
// on the peer only after its own stores), grid-stride. Needs cols % 4, 16-byte bases / strides.
extern "C" __global__ __launch_bounds__(256) void ink_p2p_push_v5(
    const float* __restrict__ own, int ld_own, float* mywin, float* peerwin, long slot_elems,
    float* __restrict__ out, int ld_out, int rows, int cols, unsigned* link, unsigned* peer_link) {
  __shared__ unsigned s_last;
  const int tid = threadIdx.x;
  const unsigned s = __hip_atomic_load(&link[0], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1u;
  const long slot = (long)(s & 1u) * slot_elems;
  const long c4 = cols >> 2;
  const long total = (long)rows * c4;
  // phase A: our partial into the peer's window (row-major, ld = cols)
  for (long i = (long)blockIdx.x * 256 + tid; i < total; i += (long)gridDim.x * 256) {
    const long r = i / c4, c = (i - r * c4) << 2;
    *(float4*)&peerwin[slot + r * cols + c] = *(const float4*)&own[r * ld_own + c];
  }
  __threadfence_system();
  __syncthreads();
  if (__builtin_amdgcn_readfirstlane(tid >> 5) == 0) {
    unsigned last = 0;
    if (tid == 0) {
      const unsigned d = __hip_atomic_fetch_add(&link[1], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
      last = (d == gridDim.x - 1) ? 1u : 0u;
    }
    last = __builtin_amdgcn_readfirstlane(last);
    if (last && tid == 0) {
      __hip_atomic_store(&link[1], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __atomic_thread_fence(__ATOMIC_RELEASE);
      __hip_atomic_store(&peer_link[2], s, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    if (tid == 0) s_last = last;
    // the peer's partial landed in our window
    while ((int)(__hip_atomic_load(&link[2], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) - s) < 0)
      __builtin_amdgcn_s_sleep(2);
  }
  __syncthreads();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  for (long i = (long)blockIdx.x * 256 + tid; i < total; i += (long)gridDim.x * 256) {
    const long r = i / c4, c = (i - r * c4) << 2;
    const float4 p = *(const float4*)&mywin[slot + r * cols + c];
    const float4 o = *(const float4*)&own[r * ld_own + c];
    float4 v;
    v.x = o.x + p.x;
    v.y = o.y + p.y;
    v.z = o.z + p.z;
    v.w = o.w + p.w;
    *(float4*)&out[r * ld_out + c] = v;
  }
  // the next call's s: the last arriver's workgroup stores link[0] once every workgroup is past
  // its read of link[0] (it read it before counting in)
  if (s_last && tid == 0) __hip_atomic_store(&link[0], s, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
