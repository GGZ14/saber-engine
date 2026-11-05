// Saber source (CMP27AH, 2026-10-29): the 27B's batched GDN decode PAIR as ONE launch -- the conv
// update (`saber_gdn_conv_update_w4_h128_hpb*`) fused into the LDS-staged replay update
// (`p27w2_gdn_replay_lds_f16s_k128_v128_ab1_vs*`, CMP27T), race-free by a last-arriver counter
// (`--engine-step-arms cuf`, an ARM).
//
// One workgroup a (sequence, v head, v slice). Each computes, in its prologue, the conv outputs it
// consumes: the whole q and k rows of its k head (all 256 channels, every step row) and its own
// slice of the v head's channels. The v channels are owned by ONE workgroup, which rewrites their
// conv window in place right after reading it (the shipped conv's own law). The q / k channels of a
// k head are needed by all H / Hg x VS workgroups of the head's group; the rewrite of their window
// (shifted by the accepted count) is done by the LAST of them to have read it: each workgroup, once
// every one of its window reads has returned (the values are consumed before the barrier), adds one
// to the group's counter `cnt[n * Hg + hk]` (device scope, release); the one that sees the count
// reach the group's size resets the word to 0 and rewrites the window at its end. The counter is a
// zeroed region (one per GDN layer, boot-zeroed) and every launch leaves it zero: every workgroup of
// a live sequence arrives, refused or not, so no state crosses launches.
//
// Arithmetic: per output, the shipped pair's, operation for operation (both objects and this one are
// compiled -ffp-contract=off): the conv's `acc = bias; acc += w[i] * win[i]` (i ascending), the bf16
// rounding of silu(acc) that the pair stored and re-read, then the LDS replay's prologue on those
// values (the shipped per-half j-ascending sum of squares on bf16->f32 values, one exchange, the
// same frsqrt / scale products), its gates, its replay and its row loop, verbatim. Every byte the
// pair wrote is written: q / k (by the group's first workgroup), v, the conv window, o, the base
// state, both stash regions and the refusal word. Gated against the pair by
// `tools/capture/cmp27ah/probe.py` (0 bytes in every output; a no-counter control must differ under
// the stress delays).
#include <hip/hip_runtime.h>
#include "sgdn_prims.h"

#define CF_K     128
#define CF_V     128
#define CF_HK    64
#define CF_TM    8
#define CF_W     4
#define CF_ST    (CF_W - 1)
#define CF_MAGIC 0x52504c59u
#define CF_SX_AB 64
#define CF_SX_K  128
#define CF_SX_V  (CF_SX_K + CF_TM * CF_K * 2)

__device__ __forceinline__ float cf_softplus(float x, float thr) {
  if (x > thr) return x;
  return (x > 0.0f) ? (x + __logf(1.0f + __expf(-x))) : __logf(1.0f + __expf(x));
}

// the conv's silu, verbatim (saber_gdn_conv.hip `cp_silu`)
__device__ __forceinline__ float cf_silu(float z) {
  return z / (1.0f + __expf(-z));
}

#ifdef CUF_STRESS
// PROBE ONLY (never in the shipped object): a per-workgroup pseudo-random stall, so the group's
// workgroups reach their window reads and their arrival in a different order every launch.
__device__ unsigned cuf_stress_seed;
__device__ __forceinline__ void cf_delay(unsigned salt) {
  const unsigned s = cuf_stress_seed;
  if (!s) return;
  // the clocks make every launch's order its own (no host write between launches)
  unsigned h = s ^ (unsigned)wall_clock64() ^ (unsigned)(clock64() >> 3) ^ (blockIdx.x * 0x9E3779B1u) ^ (blockIdx.y * 0x85EBCA77u)
             ^ (blockIdx.z * 0xC2B2AE3Du) ^ (salt * 0x27D4EB2Fu);
  h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
  const int m = (int)(h & 127u);
  for (int i = 0; i < m; ++i) __builtin_amdgcn_s_sleep(16);
}
#define CF_DELAY(s) cf_delay(s)
#else
#define CF_DELAY(s) ((void)0)
#endif

// An LDS barrier (no wait on this workgroup's global stores, which no lane of the launch reads back),
// with compiler barriers on both sides so no memory operation -- the arrival's atomic in particular
// -- is moved across it: the arrival must issue after every wave's window loads were consumed.
#define CF_LDS_BARRIER()                                                     \
  do {                                                                       \
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");          \
    asm volatile("" ::: "memory");                                           \
    __builtin_amdgcn_s_barrier();                                            \
    asm volatile("" ::: "memory");                                           \
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");          \
  } while (0)

// THE ARGUMENTS, READ WHERE THEY ARE USED (CMP27AI). The launch ABI is CMP27AH's (the kernels'
// parameter lists below are unchanged: the same 268-byte kernarg segment); the body never names a
// parameter. Each phase reads the segment through `cf_ka()`, whose pointer is laundered through an
// empty asm, so the phase issues its own scalar loads of the fields it uses and no argument (nor a
// stash pointer derived from one) is held across phases. CMP27AH's object held every argument +
// the derived stash pointers from the prologue to the last arriver's rewrite: 107 SGPRs and
// 79 / 80 SGPRs spilled to VGPR lanes (structural check 9). Only WHERE an address is loaded from
// changed; no arithmetic did.
struct CfArgs {
  const unsigned short* x; long xpitch; const unsigned short* wgt; const unsigned short* bias;
  unsigned short* cstate; long cs_seq; long cs_dim; long cs_tok; int state_len_max;
  const int* naccept; int maxq; unsigned* cnt; unsigned short* q; unsigned short* k;
  unsigned short* v; const unsigned short* av; const unsigned short* bv; long ab_stride;
  const float* A_log; const float* dt_bias; unsigned short* state; long st_slot; long st_head;
  unsigned short* o; const int* cu; const int* sidx; long sidx_stride; const int* rp;
  unsigned char* stash; long sx_div; long sx_head; unsigned* err; int H; int Hg; float scale;
  float sp_thr;
};
static_assert(__builtin_offsetof(CfArgs, naccept) == 72, "the CMP27AH kernarg layout");
static_assert(__builtin_offsetof(CfArgs, cnt) == 88, "the CMP27AH kernarg layout");
static_assert(__builtin_offsetof(CfArgs, stash) == 224, "the CMP27AH kernarg layout");
static_assert(__builtin_offsetof(CfArgs, err) == 248, "the CMP27AH kernarg layout");
static_assert(__builtin_offsetof(CfArgs, sp_thr) == 268, "the CMP27AH kernarg layout");
typedef const __attribute__((address_space(4))) CfArgs* CfKa;
__device__ __forceinline__ CfKa cf_ka() {
  CfKa p = (CfKa)__builtin_amdgcn_kernarg_segment_ptr();
  asm volatile("" : "+s"(p));
  return p;
}

// A uniform value the compiler may not hoist out of the conv's task loop (CMP27AI): everything
// derived from it is recomputed in the iteration (a few SALU ops) instead of held across the loop.
__device__ __forceinline__ int cf_pin(int v) {
  v = __builtin_amdgcn_readfirstlane(v);   // uniform by construction: the same value, in an SGPR
  asm volatile("" : "+s"(v));
  return v;
}
#define CF_PIN_UNIFORMS()                                                                        \
  const int cf_T_ = cf_pin(T), cf_bos_ = cf_pin(bos), cf_si_ = cf_pin(si), cf_off_ = cf_pin(off); \
  const int cf_r_ = cf_pin(r), cf_hk_ = cf_pin(hk);                                              \
  const int T = cf_T_, bos = cf_bos_, si = cf_si_, off = cf_off_, r = cf_r_, hk = cf_hk_

// One phase's view of the arguments (the CMP27AH parameter names, shadowing the previous phase's)
#define CF_LOAD_ARGS()                                                                           \
  const CfKa cf_a_ = cf_ka();                                                                    \
  [[maybe_unused]] const unsigned short* __restrict__ x = cf_a_->x;                              \
  [[maybe_unused]] const long xpitch = cf_a_->xpitch;                                            \
  [[maybe_unused]] const unsigned short* __restrict__ wgt = cf_a_->wgt;                          \
  [[maybe_unused]] const unsigned short* __restrict__ bias = cf_a_->bias;                        \
  [[maybe_unused]] unsigned short* __restrict__ cstate = cf_a_->cstate;                          \
  [[maybe_unused]] const long cs_seq = cf_a_->cs_seq;                                            \
  [[maybe_unused]] const long cs_dim = cf_a_->cs_dim;                                            \
  [[maybe_unused]] const long cs_tok = cf_a_->cs_tok;                                            \
  [[maybe_unused]] const int state_len_max = cf_a_->state_len_max;                               \
  [[maybe_unused]] const int* __restrict__ naccept = cf_a_->naccept;                             \
  [[maybe_unused]] const int maxq = cf_a_->maxq;                                                 \
  [[maybe_unused]] unsigned* __restrict__ cnt = cf_a_->cnt;                                      \
  [[maybe_unused]] unsigned short* __restrict__ q = cf_a_->q;                                    \
  [[maybe_unused]] unsigned short* __restrict__ k = cf_a_->k;                                    \
  [[maybe_unused]] unsigned short* __restrict__ v = cf_a_->v;                                    \
  [[maybe_unused]] const unsigned short* __restrict__ av = cf_a_->av;                            \
  [[maybe_unused]] const unsigned short* __restrict__ bv = cf_a_->bv;                            \
  [[maybe_unused]] const long ab_stride = cf_a_->ab_stride;                                      \
  [[maybe_unused]] const float* __restrict__ A_log = cf_a_->A_log;                               \
  [[maybe_unused]] const float* __restrict__ dt_bias = cf_a_->dt_bias;                           \
  [[maybe_unused]] unsigned short* __restrict__ state = cf_a_->state;                            \
  [[maybe_unused]] const long st_slot = cf_a_->st_slot;                                          \
  [[maybe_unused]] const long st_head = cf_a_->st_head;                                          \
  [[maybe_unused]] unsigned short* __restrict__ o = cf_a_->o;                                    \
  [[maybe_unused]] const int* __restrict__ cu = cf_a_->cu;                                       \
  [[maybe_unused]] const int* __restrict__ sidx = cf_a_->sidx;                                   \
  [[maybe_unused]] const long sidx_stride = cf_a_->sidx_stride;                                  \
  [[maybe_unused]] const int* __restrict__ rp = cf_a_->rp;                                       \
  [[maybe_unused]] unsigned char* __restrict__ stash = cf_a_->stash;                             \
  [[maybe_unused]] const long sx_div = cf_a_->sx_div;                                            \
  [[maybe_unused]] const long sx_head = cf_a_->sx_head;                                          \
  [[maybe_unused]] unsigned* __restrict__ err = cf_a_->err;                                      \
  [[maybe_unused]] const int H = cf_a_->H;                                                       \
  [[maybe_unused]] const int Hg = cf_a_->Hg;                                                     \
  [[maybe_unused]] const float scale = cf_a_->scale;                                             \
  [[maybe_unused]] const float sp_thr = cf_a_->sp_thr

// The write half of the sequence's stash pair, recomputed at its point of use (CMP27AH held
// `rgn` / `wx` and its four sub-pointers from the prologue): the same address, by the same
// integer arithmetic (rgn + (par ^ 1) * sx_head).
#define CF_WX()                                                                                  \
  unsigned char* wx = stash + (((size_t)sxi * H + hv) * 2) * (size_t)sx_head                    \
                      + (size_t)(par ^ 1) * sx_head;                                             \
  [[maybe_unused]] unsigned* wh = (unsigned*)wx;                                                 \
  [[maybe_unused]] unsigned short* w_ab = (unsigned short*)(wx + CF_SX_AB);                      \
  [[maybe_unused]] unsigned short* w_k = (unsigned short*)(wx + CF_SX_K);                        \
  [[maybe_unused]] unsigned short* w_v = (unsigned short*)(wx + CF_SX_V)

template <int VS>
__device__ __forceinline__ void cf_impl()
{
  constexpr int NR = CF_V / VS;          // v rows (= v channels) this workgroup owns
  constexpr int NT = 2 * NR;             // threads
  constexpr int QK_CH = 2 * CF_K;        // the k head's q | k conv channels (every workgroup of it)
  constexpr int CH = QK_CH + NR;         // + this slice's v channels: one conv task each
  constexpr int RV = (CF_TM * NR + NT - 1) / NT;   // stash v values a thread prefetches
  static_assert(CF_TM + 2 * CF_TM <= NR, "prologue 1 is one pass: r + 2T vectors fit NR lane pairs");
  __shared__ float s_k[2 * CF_TM][CF_K];
  __shared__ float s_q[CF_TM][CF_K];
  __shared__ float s_v[2 * CF_TM][NR];
  __shared__ float s_eg[2 * CF_TM];
  __shared__ float s_bt[2 * CF_TM];
  __shared__ unsigned short s_hist[CF_ST][QK_CH];   // the q | k window, raw, for the rewrite
  __shared__ unsigned s_old;

  CF_LOAD_ARGS();
  const int n = blockIdx.x, hv = blockIdx.y, z = (VS > 1) ? (int)blockIdx.z : 0;
  const int bos = cu[n];
  const int T = cu[n + 1] - bos;
  if (T <= 0) return;
  // The pair's one slot law: the conv's `cache_idx` IS the recurrent's `sidx` (the same binding,
  // the same stride -- `saber_engine::gdn_cuf` refuses a plan where they differ), so one read
  // decides both halves, uniformly over the group.
  const int si = sidx[(size_t)n * sidx_stride];
  if (si <= 0) return;                                   // NULL_BLOCK_ID: padding
  const int tid = threadIdx.x;
  const int lrow = tid >> 1, row = lrow + z * NR, half = tid & 1, k0 = half * CF_HK;
  const int ratio = H / Hg;
  const int hk = hv / ratio;
  const int r = rp[3 * n], par = rp[3 * n + 2] & 1;
  const unsigned tg = (unsigned)rp[3 * n + 1];
  const int off = naccept ? (naccept[n] - 1) : 0;
  const long sxi = (long)(si - 1) / sx_div;
  unsigned char* rgn = stash + (((size_t)sxi * H + hv) * 2) * (size_t)sx_head;
  unsigned char* rx = rgn + (size_t)par * sx_head;
  const unsigned* rh = (const unsigned*)rx;
  const unsigned short* r_ab = (const unsigned short*)(rx + CF_SX_AB);
  const unsigned short* r_k = (const unsigned short*)(rx + CF_SX_K);
  const unsigned short* r_v = (const unsigned short*)(rx + CF_SX_V);
  // r and T in range is every address below's guard; the header check only decides the refusal
  const bool shape_ok = r >= 0 && r <= CF_TM && T <= CF_TM;

  // ---- EVERY LOAD OF THE LAUNCH, ISSUED BEFORE ANY IS WAITED ON ---------------------------------
  // the base state (raw f16), the replay's stash rows / gates, then the conv taps below
  unsigned short* base = state + (size_t)si * st_slot + (size_t)hv * st_head
                         + (size_t)row * CF_K + k0;
  uint4 hraw[CF_HK / 8];
#pragma unroll
  for (int j = 0; j < CF_HK / 8; ++j) hraw[j] = *(const uint4*)(base + 8 * j);
  const bool krow = shape_ok && lrow < r;                 // this lane pair's vector is a stash k row
  uint4 kraw[CF_HK / 8];
  if (krow) {
#pragma unroll
    for (int j = 0; j < CF_HK / 8; ++j)
      kraw[j] = *(const uint4*)(r_k + (size_t)lrow * CF_K + k0 + 8 * j);
  }
  unsigned short ga = 0, gb = 0;                          // this row's gates, raw
  const bool grow = shape_ok && tid < r + T;
  if (grow) {
    if (tid < r) {
      ga = r_ab[tid];
      gb = r_ab[CF_TM + tid];
    } else {
      const size_t tok = (size_t)(bos + tid - r);
      ga = av[tok * ab_stride + hv];
      gb = bv[tok * ab_stride + hv];
    }
  }
  unsigned short rv[RV];
#pragma unroll
  for (int j = 0; j < RV; ++j) {
    const int i = tid + j * NT;
    if (shape_ok && i < r * NR) {
      const int t = i / NR, lr = i - t * NR;
      rv[j] = r_v[(size_t)t * CF_V + lr + z * NR];
    }
  }

  // ---- the replay's refusals (verbatim); a refused workgroup still runs the conv half and arrives
  unsigned bad = 0;
  if (r < 0 || r > CF_TM) bad |= 1u;
  if (r > 0 && !(rh[0] == CF_MAGIC && rh[1] == tg && rh[2] == (unsigned)si &&
                 rh[3] >= (unsigned)r))
    bad |= 1u;
  if (T > CF_TM) bad |= 2u;
  if ((long)(si - 1) % sx_div != 0) bad |= 4u;

  // ---- THE CONV HALF (saber_gdn_conv_update's per-channel law), ONE CHANNEL A TASK ------------------
  CF_DELAY(1u);
  const int slen_eff = state_len_max - (maxq - T);
  const bool qk_writer = (hv % ratio == 0) && z == 0;   // the group's writer of the q / k edges
  for (int task = tid; task < CH; task += NT) {
    CF_PIN_UNIFORMS();
    CF_LOAD_ARGS();
    CF_WX();
    const bool isv = task >= QK_CH;
    const int c = isv ? task - QK_CH : task;            // column: q|k 0..255, or v slice 0..NR-1
    const bool isk = !isv && c >= CF_K;
    const int d = isv ? ((2 * Hg + hv) * CF_V + z * NR + c)
                      : (isk ? ((Hg + hk) * CF_K + (c - CF_K)) : (hk * CF_K + c));
    unsigned short xb[CF_TM];
#pragma unroll
    for (int t = 0; t < CF_TM; ++t)
      if (t < T) xb[t] = x[(size_t)(bos + t) * xpitch + d];
    const ushort4 w4 = *(const ushort4*)(wgt + (size_t)d * CF_W);
    const unsigned short* ws = (const unsigned short*)&w4;
    float wt[CF_W];
#pragma unroll
    for (int i = 0; i < CF_W; ++i) wt[i] = sgdn::bf(ws[i]);
    const float bs = bias ? sgdn::bf(bias[d]) : 0.0f;
    unsigned short* sp = cstate + (size_t)si * cs_seq + (size_t)d * cs_dim;
    float win[CF_W];
    unsigned short hist[CF_ST];
#pragma unroll
    for (int i = 0; i < CF_ST; ++i) {
      hist[i] = sp[(size_t)(off + i) * cs_tok];
      win[i] = sgdn::bf(hist[i]);
    }
    if (!isv) {
#pragma unroll
      for (int i = 0; i < CF_ST; ++i) s_hist[i][c] = hist[i];
    }
    // one row: the conv's tap order, the bf16 output, its destinations, the window shift
    auto row_step = [&](int t, unsigned short raw) {
      win[CF_W - 1] = sgdn::bf(raw);
      float acc = bs;
#pragma unroll
      for (int i = 0; i < CF_W; ++i) acc += wt[i] * win[i];
      const unsigned short ob = sgdn::away16(cf_silu(acc));
      if (isv) v[((size_t)(bos + t) * H + hv) * CF_V + z * NR + c] = ob;
      else if (qk_writer) (isk ? k : q)[((size_t)(bos + t) * Hg + hk) * CF_K + (c & (CF_K - 1))] = ob;
      if (!bad) {
        const float f = sgdn::bf(ob);
        if (isv) {
          s_v[r + t][c] = f;
          w_v[(size_t)t * CF_V + z * NR + c] = ob;
        } else if (isk) {
          s_k[r + t][c - CF_K] = f;
          if (z == 0) w_k[(size_t)t * CF_K + (c - CF_K)] = ob;
        } else {
          s_q[t][c] = f;
        }
      }
#pragma unroll
      for (int i = 0; i < CF_W - 1; ++i) win[i] = win[i + 1];
    };
#pragma unroll
    for (int t = 0; t < CF_TM; ++t)
      if (t < T) row_step(t, xb[t]);
    for (int t = CF_TM; t < T; ++t)                      // a refused longer step only
      row_step(t, x[(size_t)(bos + t) * xpitch + d]);
    if (isv) {
      // single owner: the shipped conv's in-place rewrite, right after this workgroup's own reads.
      // hist[i + 1] by a register select: i + T < slen_eff holds only at i = 0, 1 (the host passes
      // state_len_max = (W - 1) + (maxq - 1), so slen_eff - T = W - 2)
      for (int i = 0; i < slen_eff; ++i)
        sp[(size_t)i * cs_tok] = (i + T < slen_eff) ? (i == 0 ? hist[1] : hist[2])
                                                    : x[(size_t)(bos + i - (CF_W - 2)) * xpitch + d];
    }
#ifdef CUF_NAIVE
    else {
      // PROBE CONTROL ONLY: the race CMP27L named -- every workgroup of the group rewrites the
      // shared q / k window right after its own reads, with no counter.
      for (int i = 0; i < slen_eff; ++i)
        sp[(size_t)i * cs_tok] = (i + T < slen_eff) ? (i == 0 ? hist[1] : hist[2])
                                                    : x[(size_t)(bos + i - (CF_W - 2)) * xpitch + d];
    }
#endif
  }
  // An LDS barrier: every window read of this workgroup has returned (each value was consumed by
  // the conv above) and the conv's LDS rows are visible. The global stores stay in flight: no lane
  // of this launch reads them.
  CF_LDS_BARRIER();
#ifndef CUF_NAIVE
  // THE ARRIVAL. Relaxed is enough and is the point: what the last arriver must follow is the other
  // workgroups' window LOADS, and a load whose value was consumed before this workgroup's barrier
  // has returned -- no store of this launch can reach it. The returned count is read at the end.
  unsigned old = 0;
  if (tid == 0) {
    CF_LOAD_ARGS();
    CF_DELAY(2u);
    old = __hip_atomic_fetch_add(&cnt[(size_t)n * Hg + hk], 1u, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT);
  }
#endif

  if (!bad) {
    CF_LOAD_ARGS();
    CF_WX();
    // ---- the base state (gdn_state_load's f16 -> f32, exact)
    float h[CF_HK];
#pragma unroll
    for (int j = 0; j < CF_HK / 8; ++j) {
      const unsigned short* sw = (const unsigned short*)&hraw[j];
#pragma unroll
      for (int e = 0; e < 8; ++e) h[8 * j + e] = sgdn::from_h(sw[e]);
    }
    const float alog = __expf(A_log[hv]), dtb = dt_bias[hv];

    // ---- PROLOGUE 1 (one pass): the shared rows, normalised. Vector p = lrow (both halves): p < r =
    // the stash's k row p (prefetched); r <= p < r + T = the step's k row (in s_k, raw);
    // r + T <= p < r + 2T = its q row (in s_q, raw). Normalised in place: each half its own 64.
    {
      const int nvec = r + 2 * T;
      const int p = lrow;
      const bool live = p < nvec;
      const bool isq = live && p >= r + T;
      float xv[CF_HK];
      float ss = 0.0f;
      if (live) {
        if (p < r) {
#pragma unroll
          for (int j = 0; j < CF_HK / 8; ++j) {
            const unsigned short* sw = (const unsigned short*)&kraw[j];
#pragma unroll
            for (int e = 0; e < 8; ++e) xv[8 * j + e] = sgdn::bf(sw[e]);
          }
        } else {
          const float* src = isq ? &s_q[p - r - T][k0] : &s_k[p][k0];
#pragma unroll
          for (int j = 0; j < CF_HK; j += 4) {
            const float4 f = *(const float4*)(src + j);
            xv[j] = f.x; xv[j + 1] = f.y; xv[j + 2] = f.z; xv[j + 3] = f.w;
          }
        }
#pragma unroll
        for (int j = 0; j < CF_HK; ++j) ss += xv[j] * xv[j];
      }
      ss += __shfl_xor(ss, 1, 32);
      if (live) {
        const float sc = isq ? __frsqrt_rn(ss + 1e-6f) * scale : __frsqrt_rn(ss + 1e-6f);
        float* dst = isq ? &s_q[p - r - T][k0] : &s_k[p][k0];
#pragma unroll
        for (int j = 0; j < CF_HK; j += 4) {
          float4 f;
          f.x = xv[j] * sc; f.y = xv[j + 1] * sc; f.z = xv[j + 2] * sc; f.w = xv[j + 3] * sc;
          *(float4*)(dst + j) = f;
        }
      }
    }
    // ---- PROLOGUE 2: the gates of every row (prefetched); the stash's v column (prefetched; the
    // step's own came from the conv)
    if (grow) {
      const int t = tid;
      if (t >= r && z == 0) { w_ab[t - r] = ga; w_ab[CF_TM + t - r] = gb; }
      const float ra = sgdn::bf(ga), rb = sgdn::bf(gb);
      const float g  = -alog * cf_softplus(ra + dtb, sp_thr);
      s_bt[t] = 1.0f / (1.0f + __expf(-rb));
      s_eg[t] = __expf(g);
    }
#pragma unroll
    for (int j = 0; j < RV; ++j) {
      const int i = tid + j * NT;
      if (i < r * NR) {
        const int t = i / NR, lr = i - t * NR;
        s_v[t][lr] = sgdn::bf(rv[j]);
      }
    }
    CF_LDS_BARRIER();

    // ---- REPLAY the previous step's committed rows (no outputs)
    for (int t = 0; t < r; ++t) {
      const float eg = s_eg[t], bt = s_bt[t];
      const float* kr = &s_k[t][k0];
      float dot = 0.0f;
#pragma unroll
      for (int j = 0; j < CF_HK; j += 4) {
        const float4 kv = *(const float4*)(kr + j);
        h[j] *= eg;     dot += h[j] * kv.x;
        h[j + 1] *= eg; dot += h[j + 1] * kv.y;
        h[j + 2] *= eg; dot += h[j + 2] * kv.z;
        h[j + 3] *= eg; dot += h[j + 3] * kv.w;
      }
      dot += __shfl_xor(dot, 1, 32);
      const float u = (s_v[t][lrow] - dot) * bt;
#pragma unroll
      for (int j = 0; j < CF_HK; j += 4) {
        const float4 kv = *(const float4*)(kr + j);
        h[j] += u * kv.x;
        h[j + 1] += u * kv.y;
        h[j + 2] += u * kv.z;
        h[j + 3] += u * kv.w;
      }
    }
    if (r > 0) {
      sgdn::state_store<sgdn::kStateF16, CF_HK>(base, h);         // the chain's row a-1 snapshot bytes
#pragma unroll
      for (int j = 0; j < CF_HK; j += 2) {
        const unsigned pk = sgdn::to_h2(h[j], h[j + 1]);
        h[j] = sgdn::from_h((unsigned short)(pk & 0xFFFFu));
        h[j + 1] = sgdn::from_h((unsigned short)(pk >> 16));
      }
    }

    // ---- the step's own rows: outputs, no state store
    for (int t = 0; t < T; ++t) {
      const int tok = bos + t;
      const float eg = s_eg[r + t], bt = s_bt[r + t];
      const float* kr = &s_k[r + t][k0];
      const float* qr = &s_q[t][k0];
      float dot = 0.0f;
#pragma unroll
      for (int j = 0; j < CF_HK; j += 4) {
        const float4 kv = *(const float4*)(kr + j);
        h[j] *= eg;     dot += h[j] * kv.x;
        h[j + 1] *= eg; dot += h[j + 1] * kv.y;
        h[j + 2] *= eg; dot += h[j + 2] * kv.z;
        h[j + 3] *= eg; dot += h[j + 3] * kv.w;
      }
      dot += __shfl_xor(dot, 1, 32);
      const float u = (s_v[r + t][lrow] - dot) * bt;
      float on = 0.0f;
#pragma unroll
      for (int j = 0; j < CF_HK; j += 4) {
        const float4 kv = *(const float4*)(kr + j);
        const float4 qv = *(const float4*)(qr + j);
        h[j] += u * kv.x;     on += h[j] * qv.x;
        h[j + 1] += u * kv.y; on += h[j + 1] * qv.y;
        h[j + 2] += u * kv.z; on += h[j + 2] * qv.z;
        h[j + 3] += u * kv.w; on += h[j + 3] * qv.w;
      }
      on += __shfl_xor(on, 1, 32);
      if (half == 0) o[((size_t)tok * H + hv) * CF_V + row] = sgdn::away16(on);
    }
    if (tid == 0 && z == 0) { wh[0] = CF_MAGIC; wh[1] = tg; wh[2] = (unsigned)si; wh[3] = (unsigned)T; }
  } else {
    CF_LOAD_ARGS();
    // ---- refused (the shipped replay's poison, verbatim): FATAL on the host at the next step
    if (tid == 0) atomicOr(err, bad);
    if (half == 0)
      for (int t = 0; t < T; ++t) o[((size_t)(bos + t) * H + hv) * CF_V + row] = 0xFFFFu;
  }

#ifndef CUF_NAIVE
  // ---- THE LAST ARRIVER rewrites the group's q / k window (shifted, the shipped conv's law)
  if (tid == 0) s_old = old;
  CF_LDS_BARRIER();
  if (s_old == (unsigned)(ratio * VS - 1)) {
    CF_LOAD_ARGS();
    CF_DELAY(3u);
    if (tid == 0) cnt[(size_t)n * Hg + hk] = 0u;           // every arrival is in: reset for the next launch
    for (int c = tid; c < QK_CH; c += NT) {
      const int d = (c >= CF_K) ? ((Hg + hk) * CF_K + (c - CF_K)) : (hk * CF_K + c);
      unsigned short* dst = cstate + (size_t)si * cs_seq + (size_t)d * cs_dim;
      for (int i = 0; i < slen_eff; ++i) {
        const unsigned short val = (i + T < slen_eff)
            ? s_hist[i + 1][c]
            : x[(size_t)(bos + i - (CF_W - 2)) * xpitch + d];
        dst[(size_t)i * cs_tok] = val;
      }
    }
  }
#endif
}

#define CF_ARGS                                                                                  \
    const unsigned short* __restrict__ x, long xpitch, const unsigned short* __restrict__ wgt,   \
    const unsigned short* __restrict__ bias, unsigned short* __restrict__ cstate, long cs_seq,   \
    long cs_dim, long cs_tok, int state_len_max, const int* __restrict__ naccept, int maxq,      \
    unsigned* __restrict__ cnt, unsigned short* __restrict__ q, unsigned short* __restrict__ k,  \
    unsigned short* __restrict__ v, const unsigned short* __restrict__ av,                       \
    const unsigned short* __restrict__ bv, long ab_stride, const float* __restrict__ A_log,      \
    const float* __restrict__ dt_bias, unsigned short* __restrict__ state, long st_slot,         \
    long st_head, unsigned short* __restrict__ o, const int* __restrict__ cu,                    \
    const int* __restrict__ sidx, long sidx_stride, const int* __restrict__ rp,                  \
    unsigned char* __restrict__ stash, long sx_div, long sx_head, unsigned* __restrict__ err,    \
    int H, int Hg, float scale, float sp_thr
// The parameters declare the ABI (kernarg layout + metadata) only; the body reads them via cf_ka().

extern "C" __global__ __launch_bounds__(256)
void p27w2_gdn_cuf_f16s_k128_v128_ab1_vs1(CF_ARGS) { cf_impl<1>(); }

extern "C" __global__ __launch_bounds__(128)
void p27w2_gdn_cuf_f16s_k128_v128_ab1_vs2(CF_ARGS) { cf_impl<2>(); }

extern "C" __global__ __launch_bounds__(64)
void p27w2_gdn_cuf_f16s_k128_v128_ab1_vs4(CF_ARGS) { cf_impl<4>(); }
