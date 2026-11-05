// Saber two-rank all-reduce kernels, CLEANROOM-2 (CR2-4c): the flag-in-data exact reduce (spec 2.7, 3.7; Flash-Next
// `--fn-ar ll`, an arm).
//
// No flag hand-off: each 16-byte word travels as its bitwise complement, so an all-zero u32 means "not landed".
// A word whose complement holds a zero u32 (the true u32 was 0xFFFFFFFF) also gets a zero-mask word (0x80000000 |
// bit i for each such u32 i). The receiver polls exactly the words its twin thread pushed (relaxed system-scope 8-byte
// loads) until every u32 is non-zero, or the mask has landed and every u32 outside it is non-zero; sums; zeroes the
// word (and a used mask) for the slot's next use. Global parity over 32 counters (every one advances once a call).
// No acquire fence (`acq` and `drain` are unused by this form). The private window: LL_PAYLOAD16 words a slot, then one mask u32 a word; the tail (slot 0) holds the race check's
// flags (words 0..31) and check words (32 + slot 32 + block).
#pragma once
#include "sar_exact.h"

namespace sar {

constexpr int kLlPayload16 = 20480;   // 64 rows x 2,560 bf16 a slot
constexpr int kLlCounters = 32;
#define SAR_CHK_D_LL_MASKED 13        // words accepted through a mask

enum LlStore { kLlWriteBack = 0, kLlWriteThrough = 1 };

__device__ __forceinline__ u64 ld64_sys(const u64* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void st64_sys(u64* p, u64 v) {
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ u32 ld32_sys(const u32* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void st32_sys(u32* p, u32 v) {
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ u32 zero_lanes(u32 a, u32 b, u32 c, u32 d) {
  return (a == 0u ? 1u : 0u) | (b == 0u ? 2u : 0u) | (c == 0u ? 4u : 0u) | (d == 0u ? 8u : 0u);
}

// the exact bf16 sum with MY value as the first source operand of every add, pinned at the instruction level: the
// decoded complement otherwise reaches the add as its first source, and an add of two NaNs returns the first
// source's NaN (spec 7.1)
__device__ __forceinline__ float add_mine_first(float mine, float peer) {
  float r;
  asm("v_add_f32 %0, %1, %2" : "=v"(r) : "v"(mine), "v"(peer));
  return r;
}
__device__ __forceinline__ u32 ll_add_lane(u32 a, u32 b) {
  return rne16x2(add_mine_first(bf_lo(a), bf_lo(b)), add_mine_first(bf_hi(a), bf_hi(b)));
}
__device__ __forceinline__ int4 ll_add_word(int4 a, int4 b) {
  return make_int4((int)ll_add_lane((u32)a.x, (u32)b.x), (int)ll_add_lane((u32)a.y, (u32)b.y),
                   (int)ll_add_lane((u32)a.z, (u32)b.z), (int)ll_add_lane((u32)a.w, (u32)b.w));
}

// one word out: the complement (plain stores, or system-scope 8-byte stores in the write-through form) and, when
// the complement has a zero u32, the mask word
template <int ST>
__device__ __forceinline__ void ll_send(int4* pw, u32* pm, int k, int4 v) {
  const u32 e0 = ~(u32)v.x, e1 = ~(u32)v.y, e2 = ~(u32)v.z, e3 = ~(u32)v.w;
  const u32 z = zero_lanes(e0, e1, e2, e3);
  if constexpr (ST == kLlWriteThrough) {
    u64* q = (u64*)(pw + k);
    st64_sys(q, (u64)e0 | ((u64)e1 << 32));
    st64_sys(q + 1, (u64)e2 | ((u64)e3 << 32));
    if (z) st32_sys(pm + k, 0x80000000u | z);
  } else {
    pw[k] = make_int4((int)e0, (int)e1, (int)e2, (int)e3);
    if (z) pm[k] = 0x80000000u | z;
  }
}

// one word in: poll until it validates (every u32 non-zero, or the mask landed and covers every zero u32), sum it
// with my value, zero the slot word (and a used mask); returns the decoded peer word
template <int CHK>
__device__ __forceinline__ int4 ll_recv(int4* mw, u32* mm, int k, int4 a, int4* out, u32& masked) {
  u64* q = (u64*)(mw + k);
  u64 lo, hi;
  bool used_mask = false;
  u64 z = 0;
  for (;;) {
    lo = ld64_sys(q);
    hi = ld64_sys(q + 1);
    const u32 m = ld32_sys(mm + k);
    if constexpr (CHK == kCheckedNoWait) break;
    const u32 zl = zero_lanes((u32)lo, (u32)(lo >> 32), (u32)hi, (u32)(hi >> 32));
    if (zl == 0u) break;
    if ((m & 0x80000000u) && (zl & ~m) == 0u) {
      used_mask = true;
      break;
    }
    if (++z > SABER_AR_SPIN_MAX) break;
  }
  const int4 p = make_int4((int)~(u32)lo, (int)~(u32)(lo >> 32), (int)~(u32)hi, (int)~(u32)(hi >> 32));
  out[k] = ll_add_word(a, p);
  st64_sys(q, 0ull);
  st64_sys(q + 1, 0ull);
  if (used_mask) {
    st32_sys(mm + k, 0u);
    ++masked;
  }
  return p;
}

template <int CHK, int ST>
__device__ __forceinline__ void ll_exact(const int4* in, int4* peer_scratch, int4* my_scratch, int4* out,
                                         u32* peer_tail, u32* my_tail, u32* seq, int n16, int stride16) {
  __shared__ u32 bcast;
  const int b = (int)blockIdx.x, nb = (int)gridDim.x;
  const int nt = (int)blockDim.x, tid = (int)threadIdx.x;
  const Run run = block_run(n16);
  const u32 s = take_seq(seq, b, &bcast);
  advance_idle(seq, b, nb, kLlCounters);
  const u32 slot = s & 1u;
  const long so = (long)slot * stride16;
  int4* pw = peer_scratch + so;
  u32* pm = (u32*)(pw + kLlPayload16);
  int4* mw = my_scratch + so;
  u32* mm = (u32*)(mw + kLlPayload16);
  unsigned h = 0u;

  // send; a thread's first two words stay in registers for the sum
  const int k0 = run.start + tid, k1 = k0 + nt;
  const bool h0 = k0 < run.end, h1 = k1 < run.end;
  int4 v0 = make_int4(0, 0, 0, 0), v1 = v0;
  if (h0) v0 = in[k0];
  if (h1) v1 = in[k1];
  if (h0) {
    ll_send<ST>(pw, pm, k0, v0);
    if constexpr (CHK != kNoCheck) h ^= chk_word(v0, (unsigned)k0);
  }
  if (h1) {
    ll_send<ST>(pw, pm, k1, v1);
    if constexpr (CHK != kNoCheck) h ^= chk_word(v1, (unsigned)k1);
  }
  for (int k = k1 + nt; k < run.end; k += nt) {
    const int4 v = in[k];
    ll_send<ST>(pw, pm, k, v);
    if constexpr (CHK != kNoCheck) h ^= chk_word(v, (unsigned)k);
  }
  if constexpr (ST == kLlWriteBack) {
    // a wave that pushed anything waits for its stores to reach L2, then writes L2 back at system scope (not
    // waited for): without it the words sit dirty in this card's L2 and the peer never sees them
    if (__builtin_amdgcn_readfirstlane(run.start + (tid & ~31)) < run.end) {
      wait_stores();
      writeback_sys();
    }
  }
  if constexpr (CHK != kNoCheck) {
    const unsigned t = chk_block_xor(h);
    if (tid == 0) {
      peer_tail[32 + slot * 32 + b] = t ^ chk_seq(s);
      flag_release(peer_tail + b, s);
    }
  }

  // receive: poll the twin's words, sum, zero them
  h = 0u;
  u32 masked = 0u;
  int kr = k0;
  if constexpr (CHK != kCheckedNoWait) {   // (the control reads every word once, in one loop)
    if (h0) {
      const int4 p = ll_recv<CHK>(mw, mm, k0, v0, out, masked);
      if constexpr (CHK != kNoCheck) h ^= chk_word(p, (unsigned)k0);
    }
    if (h1) {
      const int4 p = ll_recv<CHK>(mw, mm, k1, v1, out, masked);
      if constexpr (CHK != kNoCheck) h ^= chk_word(p, (unsigned)k1);
    }
    kr = k1 + nt;
  }
  for (int k = kr; k < run.end; k += nt) {
    const int4 p = ll_recv<CHK>(mw, mm, k, in[k], out, masked);
    if constexpr (CHK != kNoCheck) h ^= chk_word(p, (unsigned)k);
  }
  if constexpr (CHK != kNoCheck) {
    u32* diag = seq + CHK_DIAG;
    if (masked) atomicAdd(&diag[SAR_CHK_D_LL_MASKED], masked);
    const unsigned t = chk_block_xor(h);
    if (tid == 0) {
      u64 z = 0;
      while ((int)(flag_acquire(my_tail + b) - s) < 0) {
        if (++z > SABER_AR_SPIN_MAX) break;
      }
      if (b == 0) diag[CHK_D_LAST_S] = s;
      atomicAdd(&diag[CHK_D_BLOCKS], 1u);
      if ((t ^ chk_seq(s)) != my_tail[32 + slot * 32 + b]) chk_mismatch(diag, s, (unsigned)b, 0u);
    }
  }
}

}  // namespace sar

#define SAR_LL_PARAMS                                                                                         \
  const int4 *in, int4 *peer_scratch, int4 *my_scratch, int4 *out, unsigned *peer_tail, unsigned *my_tail,      \
      unsigned *seq_ctrs, int n16, int slot_stride16, int drain, int acq
#define SAR_LL_ARGS in, peer_scratch, my_scratch, out, peer_tail, my_tail, seq_ctrs, n16, slot_stride16
