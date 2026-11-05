// Saber two-rank all-reduce kernels, CLEANROOM-2 (CR2-4c): the rotated, quantised family (spec 2.3 - 2.5, 3.5,
// 3.6, 3.9): the one-shot (bits 6 / 4, bf16 / f16 in and out), its global-parity twin, the wave-parallel body and
// the split halves.
//
// The message is cut into 64-element groups (8 lanes x 8 elements); a chunk is CG = 32 groups, walked by one wave in
// ITERS = 8 iterations of 4 groups. Per group: the orthonormal Walsh-Hadamard transform (radix-2 stages on index bits
// 0, 1, 2 inside a lane, then 3, 4, 5 across lanes xor 1 / 2 / 4, then x rsq(64)), amax, a bf16-rounded scale,
// RTNE symbols. A lane's symbols of a chunk form one LSB-first bit stream of NWORD u32 words, stored as NU4 rows of
// 32 lane-contiguous uint4s at chunk c's region of the peer's slot and, identically, of this rank's local pack.
// The reduce dequantises the peer's half and this rank's QUANTISED half (peer term first), adds once, transforms
// and converts.
#pragma once
#include "sar_prims.h"
#include "sar_exact.h"

namespace sar {

template <int BITS>
struct Rot {
  static constexpr int CG = (BITS % 2 == 0) ? 32 : 64;   // groups a chunk
  static constexpr int ITERS = CG * 64 / 256;             // a wave covers 4 groups an iteration
  static constexpr int NWORD = ITERS * 8 * BITS / 32;     // u32 a lane a chunk
  static constexpr int NU4 = NWORD / 4;
  static constexpr int U4PC = NU4 * 32;                   // uint4 a chunk
  static constexpr int QMIN = -(1 << (BITS - 1));
  static constexpr int QMAX = (1 << (BITS - 1)) - 1;
};

// 1 / qmax as the f32 the scale multiplies by
template <int BITS>
__device__ __forceinline__ float inv_qmax() {
  return 1.0f / (float)Rot<BITS>::QMAX;
}

// ---- element I/O: 8 consecutive elements (one 16-byte word) a lane

template <int E>
__device__ __forceinline__ void load8(const int4* src, float (&x)[8]) {
  const int4 v = *src;
  const u32 w[4] = {(u32)v.x, (u32)v.y, (u32)v.z, (u32)v.w};
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    if constexpr (E == kBf16) {
      x[2 * e] = bf_lo(w[e]);
      x[2 * e + 1] = bf_hi(w[e]);
    } else {
      x[2 * e] = hf(w[e] & 0xFFFFu);
      x[2 * e + 1] = hf(w[e] >> 16);
    }
  }
}

template <int E>
__device__ __forceinline__ int4 pack8(const float (&x)[8]) {
  u32 w[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    if constexpr (E == kBf16) w[e] = rne16x2(x[2 * e], x[2 * e + 1]);
    else w[e] = to_h2(x[2 * e], x[2 * e + 1]);
  }
  return make_int4((int)w[0], (int)w[1], (int)w[2], (int)w[3]);
}

// ---- the transform (spec 2.3.2): lower index x, upper y -> (x + y, x - y), one rounding each, in stage order

__device__ __forceinline__ void fwht(float (&x)[8], int j, float nrm) {
#pragma unroll
  for (int d = 1; d < 8; d <<= 1) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (i & d) continue;
      const float a = x[i], c = x[i + d];
      x[i] = a + c;
      x[i + d] = a - c;
    }
  }
#pragma unroll
  for (int d = 1; d < 8; d <<= 1) {
    const bool up = (j & d) != 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float p = __shfl_xor(x[i], d, 32);
      x[i] = up ? (p - x[i]) : (x[i] + p);
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) x[i] = x[i] * nrm;
}

// ---- amax, scale, symbols (spec 2.3.3): returns the scale; code[i] in [0, 2^BITS)

template <int BITS>
__device__ __forceinline__ float quantise(const float (&r)[8], u32 (&code)[8]) {
  float a = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) a = fmaxf(a, fabsf(r[i]));
  a = fmaxf(a, __shfl_xor(a, 1, 32));
  a = fmaxf(a, __shfl_xor(a, 2, 32));
  a = fmaxf(a, __shfl_xor(a, 4, 32));
  float sc = (a > 0.0f) ? scl16(a * inv_qmax<BITS>()) : 1.0f;
  if (!(sc > 0.0f)) sc = 1.0f;
  const float inv = 1.0f / sc;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int q = (int)rintf(r[i] * inv);
    code[i] = (u32)(min(max(q, Rot<BITS>::QMIN), Rot<BITS>::QMAX) - Rot<BITS>::QMIN);
  }
  return sc;
}

// ---- the lane's bit stream: symbol (t, i) at bits [(8t + i) BITS, + BITS) LSB-first

template <int BITS, int T>
__device__ __forceinline__ void put_symbols(u32 (&w)[Rot<BITS>::NWORD], const u32 (&code)[8]) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int pos = (8 * T + i) * BITS;
    const int wi = pos >> 5, bo = pos & 31;
    w[wi] |= code[i] << bo;
    if (bo + BITS > 32) w[wi + 1] |= code[i] >> (32 - bo);
  }
}

// iteration t's 8 symbols out of a window of three consecutive stream words, starting at bit bo of the first
template <int BITS>
__device__ __forceinline__ void get_symbols(u32 lo, u32 mid, u32 hi, int bo, u32 (&code)[8]) {
  // the 8 * BITS (<= 48) bits from bit bo of the 96-bit window lo | mid << 32 | hi << 64
  const u32 f0 = __builtin_amdgcn_alignbit(mid, lo, bo);
  const u32 f1 = __builtin_amdgcn_alignbit(hi, mid, bo);
  const u64 f = (u64)f0 | ((u64)f1 << 32);
#pragma unroll
  for (int i = 0; i < 8; ++i) code[i] = (u32)(f >> (BITS * i)) & ((1u << BITS) - 1u);
}

template <int BITS, int T>
__device__ __forceinline__ void get_symbols_at(const u32 (&w)[Rot<BITS>::NWORD], u32 (&code)[8]) {
  constexpr int pos = 8 * T * BITS;
  constexpr int wi = pos >> 5, bo = pos & 31;
  constexpr int N = Rot<BITS>::NWORD;
  const u32 lo = w[wi];
  u32 mid = 0u, hi = 0u;
  if constexpr (wi + 1 < N) mid = w[wi + 1];
  if constexpr (wi + 2 < N) hi = w[wi + 2];
  get_symbols<BITS>(lo, mid, hi, bo, code);
}

// dequantise both halves and add, peer term first (both products are exact)
template <int BITS>
__device__ __forceinline__ void dequant_add(const u32 (&cp)[8], float sp, const u32 (&cm)[8], float sm,
                                            float (&v)[8]) {
#pragma unroll
  for (int i = 0; i < 8; ++i)
    v[i] = (float)((int)cp[i] + Rot<BITS>::QMIN) * sp + (float)((int)cm[i] + Rot<BITS>::QMIN) * sm;
}

// ---- the regions of one call

struct RotRegions {
  char* peer;          // the peer's slot (payload from 0, bf16 scales at scale_off)
  const char* mine;    // my slot, as the peer filled it
  char* loc;           // my local pack (payload from 0, f32 scales at scale_off)
  long scale_off;
};

__device__ __forceinline__ uint4* u4_at(char* base, long idx) { return (uint4*)base + idx; }
__device__ __forceinline__ const uint4* u4_at(const char* base, long idx) { return (const uint4*)base + idx; }
__device__ __forceinline__ u16* sc16(char* base, long so) { return (u16*)(base + so); }
__device__ __forceinline__ const u16* sc16(const char* base, long so) { return (const u16*)(base + so); }
__device__ __forceinline__ float* sc32(char* base, long so) { return (float*)(base + so); }

// ---- one chunk's push by one wave: transform, quantise, scales, the symbol stream to the peer and the pack

template <int BITS, int E, int CHK>
__device__ __forceinline__ void push_chunk(const u16* in, const RotRegions& R, long c, int n_groups, float nrm,
                                           unsigned& h) {
  using G = Rot<BITS>;
  const int lane = (int)threadIdx.x & 31;
  const int j = lane & 7;
  u32 w[G::NWORD];
#pragma unroll
  for (int i = 0; i < G::NWORD; ++i) w[i] = 0u;
  float x[G::ITERS][8];
#pragma unroll
  for (int t = 0; t < G::ITERS; ++t) {
    const long g = c * G::CG + 4 * t + (lane >> 3);
    if (g < n_groups) {
      load8<E>((const int4*)(in + g * 64 + 8 * j), x[t]);
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) x[t][i] = 0.0f;
    }
  }
#pragma unroll
  for (int t = 0; t < G::ITERS; ++t) {
    const long g = c * G::CG + 4 * t + (lane >> 3);
    fwht(x[t], j, nrm);
    u32 code[8];
    const float sc = quantise<BITS>(x[t], code);
    if (j == 0 && g < n_groups) {
      const u16 s16 = (u16)(__float_as_uint(sc) >> 16);
      sc16(R.peer, R.scale_off)[g] = s16;
      sc32(R.loc, R.scale_off)[g] = sc;
      if constexpr (CHK != kNoCheck) h ^= chk_half(s16, (unsigned)g);
    }
    switch (t) {   // compile-time after unrolling
#define SAR_PUT(T) \
  case T: put_symbols<BITS, T>(w, code); break;
      SAR_PUT(0) SAR_PUT(1) SAR_PUT(2) SAR_PUT(3) SAR_PUT(4) SAR_PUT(5) SAR_PUT(6) SAR_PUT(7)
#undef SAR_PUT
    }
  }
#pragma unroll
  for (int u = 0; u < G::NU4; ++u) {
    const long idx = c * G::U4PC + u * 32 + lane;
    const uint4 v = make_uint4(w[4 * u], w[4 * u + 1], w[4 * u + 2], w[4 * u + 3]);
    *u4_at(R.peer, idx) = v;
    *u4_at(R.loc, idx) = v;
    if constexpr (CHK != kNoCheck) h ^= chk_word4(v, (unsigned)idx);
  }
}

// ---- one chunk's reduce by one wave

template <int BITS, int E, int CHK>
__device__ __forceinline__ void reduce_chunk(u16* out, const RotRegions& R, long c, int n_groups, float nrm,
                                             unsigned& h) {
  using G = Rot<BITS>;
  const int lane = (int)threadIdx.x & 31;
  const int j = lane & 7;
  u32 pw[G::NWORD], mw[G::NWORD];
#pragma unroll
  for (int u = 0; u < G::NU4; ++u) {
    const long idx = c * G::U4PC + u * 32 + lane;
    const uint4 a = *u4_at(R.mine, idx);
    const uint4 b = *u4_at((const char*)R.loc, idx);
    pw[4 * u] = a.x; pw[4 * u + 1] = a.y; pw[4 * u + 2] = a.z; pw[4 * u + 3] = a.w;
    mw[4 * u] = b.x; mw[4 * u + 1] = b.y; mw[4 * u + 2] = b.z; mw[4 * u + 3] = b.w;
    if constexpr (CHK != kNoCheck) h ^= chk_word4(a, (unsigned)idx);
  }
#pragma unroll
  for (int t = 0; t < G::ITERS; ++t) {
    const long g = c * G::CG + 4 * t + (lane >> 3);
    const bool live = g < n_groups;
    float sp = 0.0f, sm = 0.0f;
    if (live) {
      const u16 s16 = sc16(R.mine, R.scale_off)[g];
      sp = bf(s16);
      sm = sc32(R.loc, R.scale_off)[g];
      if constexpr (CHK != kNoCheck) {
        if (j == 0) h ^= chk_half(s16, (unsigned)g);
      }
    }
    u32 cp[8], cm[8];
    switch (t) {
#define SAR_GET(T)                          \
  case T:                                   \
    get_symbols_at<BITS, T>(pw, cp);        \
    get_symbols_at<BITS, T>(mw, cm);        \
    break;
      SAR_GET(0) SAR_GET(1) SAR_GET(2) SAR_GET(3) SAR_GET(4) SAR_GET(5) SAR_GET(6) SAR_GET(7)
#undef SAR_GET
    }
    float v[8];
    dequant_add<BITS>(cp, sp, cm, sm, v);
    fwht(v, j, nrm);
    if (live) *(int4*)(out + g * 64 + 8 * j) = pack8<E>(v);
  }
}

// the chunks a block owns
struct ChunkRun {
  int c0, c1, cpb;
};
__device__ __forceinline__ ChunkRun block_chunks(int n_chunks, int nb, int b) {
  const int cpb = (n_chunks + nb - 1) / nb;
  const int c0 = b * cpb;
  return ChunkRun{c0, min(c0 + cpb, n_chunks), cpb};
}

__device__ __forceinline__ RotRegions regions(char* peer_scratch, const char* my_scratch, char* loc_pack, u32 s,
                                              long stride_bytes, long scale_off) {
  const long so = (long)(s & 1u) * stride_bytes;
  return RotRegions{peer_scratch + so, my_scratch + so, loc_pack, scale_off};
}

// ---- the race check's receiver for a block's chunks: one piece (0) = the payload words + the bf16 scales it
// read; on a mismatch the block re-reads both at system scope until the hash matches or CHK_PROBE_MAX polls pass

template <int BITS>
__device__ __forceinline__ void rot_check_recv(unsigned h, const RotRegions& R, const ChunkRun& cr, int n_groups,
                                               int b, u32 s, long slot_bytes, u32* diag) {
  using G = Rot<BITS>;
  __shared__ int bad;
  const unsigned t = chk_block_xor(h);
  const unsigned* ck = chk_ck(R.mine, slot_bytes) + b * CHK_PIECES;
  if (threadIdx.x == 0) {
    if (b == 0) diag[CHK_D_LAST_S] = s;
    atomicAdd(&diag[CHK_D_BLOCKS], 1u);
    bad = (t ^ chk_seq(s)) != *ck;
    if (bad) chk_mismatch(diag, s, (unsigned)b, 0u);
  }
  __syncthreads();
  if (!bad) return;
  const long u0 = (long)cr.c0 * G::U4PC, u1 = (long)cr.c1 * G::U4PC;
  const long g0 = (long)cr.c0 * G::CG, g1 = min((long)cr.c1 * G::CG, (long)n_groups);
  unsigned polls = 0u;
  bool ok = false;
  while (!ok && polls < CHK_PROBE_MAX) {
    ++polls;
    __builtin_amdgcn_s_sleep(8);
    unsigned r = 0u;
    for (long k = u0 + threadIdx.x; k < u1; k += blockDim.x)
      r ^= chk_word(chk_ld_sys((const int4*)R.mine + k), (unsigned)k);
    const unsigned* sw = (const unsigned*)(R.mine + R.scale_off);
    for (long g = g0 + threadIdx.x; g < g1; g += blockDim.x) {
      const unsigned pair = chk_ld32_sys(sw + (g >> 1));
      r ^= chk_half((u16)((g & 1) ? (pair >> 16) : (pair & 0xFFFFu)), (unsigned)g);
    }
    const unsigned tt = chk_block_xor(r);
    if (threadIdx.x == 0) bad = (tt ^ chk_seq(s)) != chk_ld32_sys(ck);
    __syncthreads();
    ok = !bad;
    __syncthreads();
  }
  if (threadIdx.x == 0) chk_probe_done(diag, polls, ok);
}

// ---- the one-shot (and its global-parity twin)

template <int BITS, int E, int PAR, int CHK>
__device__ __forceinline__ void rot_oneshot(const u16* in, char* peer_scratch, char* my_scratch, char* loc_pack,
                                            u16* out, u32* peer_flags, u32* my_flags, u32* seq, int n_groups,
                                            int n_chunks, long stride_bytes, long scale_off, int drain, int acq) {
  __shared__ u32 bcast;
  const int b = (int)blockIdx.x;
  const int tid = (int)threadIdx.x;
  const int wave = tid >> 5, nwaves = (int)blockDim.x >> 5;
  const float nrm = rsq(64.0f);
  const ChunkRun cr = block_chunks(n_chunks, (int)gridDim.x, b);
  const u32 s = take_seq(seq, b, &bcast);
  if constexpr (PAR == kGlobal) advance_idle(seq, b, (int)gridDim.x, kGpCounters);
  const RotRegions R = regions(peer_scratch, my_scratch, loc_pack, s, stride_bytes, scale_off);
  unsigned h = 0u;
  for (int c = cr.c0 + wave; c < cr.c1; c += nwaves) push_chunk<BITS, E, CHK>(in, R, c, n_groups, nrm, h);
  if constexpr (CHK != kNoCheck) {
    if constexpr (PAR == kGlobal) {
      if (tid == 0) chk_ck(R.peer, stride_bytes)[b * CHK_PIECES + 1] = s;
    }
    chk_send(h, R.peer, stride_bytes, (unsigned)b, 0u, s);
  }
  drain_and_sync(drain);
  if (tid == 0) {
    const u64 z = signal_wait(peer_flags + b, my_flags + b, s);
    if constexpr (CHK != kNoCheck) {
      u32* diag = seq + CHK_DIAG;
      chk_after_spin(my_flags + b, s, z, diag);
      if constexpr (PAR == kGlobal) {
        if (chk_ck(R.mine, stride_bytes)[b * CHK_PIECES + 1] != s) atomicAdd(&diag[SAR_CHK_D_GP_PARITY], 1u);
      }
    }
  }
  __syncthreads();
  acquire_opt(acq);
  h = 0u;
  for (int c = cr.c0 + wave; c < cr.c1; c += nwaves) reduce_chunk<BITS, E, CHK>(out, R, c, n_groups, nrm, h);
  if constexpr (CHK != kNoCheck) rot_check_recv<BITS>(h, R, cr, n_groups, b, s, stride_bytes, seq + CHK_DIAG);
}

}  // namespace sar

#define SAR_ROT_PARAMS                                                                                         \
  const unsigned short *in, char *peer_scratch, char *my_scratch, char *loc_pack, unsigned short *out,           \
      unsigned *peer_flags, unsigned *my_flags, unsigned *seq_ctrs, int n_groups, int n_chunks,                   \
      long slot_stride_bytes, long scale_off_bytes, int drain, int acq
#define SAR_ROT_ARGS                                                                                           \
  in, peer_scratch, my_scratch, loc_pack, out, peer_flags, my_flags, seq_ctrs, n_groups, n_chunks,               \
      slot_stride_bytes, scale_off_bytes, drain, acq
