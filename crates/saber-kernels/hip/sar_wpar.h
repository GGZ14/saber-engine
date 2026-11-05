// Saber two-rank all-reduce kernels, CLEANROOM-2 (CR2-4c): the wave-parallel rotated body and the split halves
// (spec 2.4, 2.5, 3.6). Same arithmetic, payload, scales, local pack, flags and slot law as the one-shot of
// sar_rot.h; only WHO does the work (wpar) or WHERE the call is cut (split) differs.
//
// wpar: a block's items are (chunk-in-block, iteration) pairs, item it = cl ITERS + t, waves take items w, w + nwaves,
// ...; an item's 8 symbols a lane are OR-ed into the lane's word stream in LDS ([CPB_MAX][32][NWORD], zeroed at entry;
// two items can share a word), then every thread copies whole uint4s out to the one-shot's offsets. The reduce item
// reads only the one or two uint4s that hold its iteration's bits.
// split: the push half ends with the per-block flags and an arrival count; the block that completes it raises the
// peer's split flag (word 511) to the host's flag_val; the host's stream waits on it; the reduce half reads (does not
// advance) its block's counter.
#pragma once
#include "sar_rot.h"

namespace sar {

constexpr int kSplitWord = 511;

// ---- wpar

template <int CPB_MAX, int CHK>
__device__ __forceinline__ void rot_wpar(const u16* in, char* peer_scratch, char* my_scratch, char* loc_pack, u16* out,
                                         u32* peer_flags, u32* my_flags, u32* seq, int n_groups, int n_chunks,
                                         long stride_bytes, long scale_off, int drain, int acq) {
  constexpr int BITS = 6;
  using G = Rot<BITS>;
  constexpr int SB = 8 * BITS;   // stream bits an iteration
  __shared__ u32 bcast;
  __shared__ __attribute__((aligned(16))) u32 lw[CPB_MAX * 32 * G::NWORD];
  const int b = (int)blockIdx.x;
  const int tid = (int)threadIdx.x, nt = (int)blockDim.x;
  const int wave = tid >> 5, nwaves = nt >> 5;
  const int lane = tid & 31, j = lane & 7;
  const float nrm = rsq(64.0f);
  const ChunkRun cr = block_chunks(n_chunks, (int)gridDim.x, b);
  if (cr.cpb > CPB_MAX) __builtin_trap();
  for (int i = tid; i < CPB_MAX * 32 * G::NWORD / 4; i += nt) ((uint4*)lw)[i] = make_uint4(0u, 0u, 0u, 0u);
  const u32 s = take_seq(seq, b, &bcast);
  const RotRegions R = regions(peer_scratch, my_scratch, loc_pack, s, stride_bytes, scale_off);
  const int nci = max(cr.c1 - cr.c0, 0);
  const int items = nci * G::ITERS;
  unsigned h = 0u;

  // phase 1: transform + quantise an item, its scale, its symbols OR-ed into the lane's LDS stream
  for (int it = wave; it < items; it += nwaves) {
    const int cl = it / G::ITERS, t = it % G::ITERS;
    const long c = cr.c0 + cl;
    const long g = c * G::CG + 4 * t + (lane >> 3);
    float x[8];
    if (g < n_groups) {
      load8<kBf16>((const int4*)(in + g * 64 + 8 * j), x);
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) x[i] = 0.0f;
    }
    fwht(x, j, nrm);
    u32 code[8];
    const float sc = quantise<BITS>(x, code);
    if (j == 0 && g < n_groups) {
      const u16 s16 = (u16)(__float_as_uint(sc) >> 16);
      sc16(R.peer, R.scale_off)[g] = s16;
      sc32(R.loc, R.scale_off)[g] = sc;
      if constexpr (CHK != kNoCheck) h ^= chk_half(s16, (unsigned)g);
    }
    u64 f = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) f |= (u64)code[i] << (BITS * i);
    const int pos = SB * t;
    const int wi = pos >> 5, bo = pos & 31;
    u32* row = lw + (cl * 32 + lane) * G::NWORD;
    atomicOr(row + wi, (u32)(f << bo));
    atomicOr(row + wi + 1, (u32)(f >> (32 - bo)));
    if (bo + SB > 64) atomicOr(row + wi + 2, (u32)(f >> (64 - bo)));
  }
  __syncthreads();
  // phase 2: whole uint4s to the peer's slot and the local pack, at the one-shot's offsets
  const int total = nci * G::U4PC;
  for (int idx = tid; idx < total; idx += nt) {
    const int cl = idx / G::U4PC, r = idx - cl * G::U4PC;
    const int u = r >> 5, L = r & 31;
    const uint4 v = *(const uint4*)(lw + (cl * 32 + L) * G::NWORD + 4 * u);
    const long gi = (long)(cr.c0 + cl) * G::U4PC + r;
    *u4_at(R.peer, gi) = v;
    *u4_at(R.loc, gi) = v;
    if constexpr (CHK != kNoCheck) h ^= chk_word4(v, (unsigned)gi);
  }
  if constexpr (CHK != kNoCheck) chk_send(h, R.peer, stride_bytes, (unsigned)b, 0u, s);
  drain_and_sync(drain);
  if (tid == 0) {
    const u64 z = signal_wait(peer_flags + b, my_flags + b, s);
    if constexpr (CHK != kNoCheck) chk_after_spin(my_flags + b, s, z, seq + CHK_DIAG);
  }
  __syncthreads();
  acquire_opt(acq);

  // reduce: an item loads the one or two uint4s holding iteration t's bits (each peer uint4 hashed once, by the
  // first iteration that holds it)
  h = 0u;
  for (int it = wave; it < items; it += nwaves) {
    const int cl = it / G::ITERS, t = it % G::ITERS;
    const long c = cr.c0 + cl;
    const long g = c * G::CG + 4 * t + (lane >> 3);
    const bool live = g < n_groups;
    const int b0 = SB * t;
    const int u0 = b0 >> 7, u1 = (b0 + SB - 1) >> 7;
    const long i0 = c * G::U4PC + u0 * 32 + lane;
    const uint4 p0 = *u4_at(R.mine, i0);
    const uint4 m0 = *u4_at((const char*)R.loc, i0);
    uint4 p1 = make_uint4(0u, 0u, 0u, 0u), m1 = p1;
    if (u1 != u0) {
      p1 = *u4_at(R.mine, i0 + 32);
      m1 = *u4_at((const char*)R.loc, i0 + 32);
    }
    if constexpr (CHK != kNoCheck) {
      if (t == 0 || ((b0 - 1) >> 7) < u0) h ^= chk_word4(p0, (unsigned)i0);
      if (u1 != u0) h ^= chk_word4(p1, (unsigned)(i0 + 32));
    }
    const int rel = b0 - 128 * u0;
    const int wi = rel >> 5, bo = rel & 31;
    const u32 pa[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
    const u32 ma[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};
    u32 pl = pa[0], pm = pa[1], ph = pa[2], ml = ma[0], mm = ma[1], mh = ma[2];
#pragma unroll
    for (int q = 1; q < 4; ++q) {
      if (wi == q) {
        pl = pa[q]; pm = pa[q + 1]; ph = pa[q + 2];
        ml = ma[q]; mm = ma[q + 1]; mh = ma[q + 2];
      }
    }
    u32 cp[8], cm[8];
    get_symbols<BITS>(pl, pm, ph, bo, cp);
    get_symbols<BITS>(ml, mm, mh, bo, cm);
    float sp = 0.0f, sm = 0.0f;
    if (live) {
      const u16 s16 = sc16(R.mine, R.scale_off)[g];
      sp = bf(s16);
      sm = sc32(R.loc, R.scale_off)[g];
      if constexpr (CHK != kNoCheck) {
        if (j == 0) h ^= chk_half(s16, (unsigned)g);
      }
    }
    float v[8];
    dequant_add<BITS>(cp, sp, cm, sm, v);
    fwht(v, j, nrm);
    if (live) *(int4*)(out + g * 64 + 8 * j) = pack8<kBf16>(v);
  }
  if constexpr (CHK != kNoCheck) rot_check_recv<BITS>(h, R, cr, n_groups, b, s, stride_bytes, seq + CHK_DIAG);
}

// ---- split

// thread 0, after the drain: count this block's arrival; the last of P raises the peer's split flag
__device__ __forceinline__ void split_arrive(u32* peer_flags, u32* seq, u32 flag_val, int P) {
  fence_sys();
  const u32 old = __hip_atomic_fetch_add(seq + kSplitWord, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
  if (old + 1u == (u32)P) {
    __hip_atomic_store(seq + kSplitWord, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    fence_sys();
    flag_release(peer_flags + kSplitWord, flag_val);
  }
}

// the push half at the one-shot's grid
__device__ __forceinline__ void split_push(const u16* in, char* peer_scratch, char* my_scratch, char* loc_pack,
                                           u32* peer_flags, u32* seq, int n_groups, int n_chunks, long stride_bytes,
                                           long scale_off, int drain, u32 flag_val) {
  __shared__ u32 bcast;
  const int b = (int)blockIdx.x;
  const int wave = (int)threadIdx.x >> 5, nwaves = (int)blockDim.x >> 5;
  const float nrm = rsq(64.0f);
  const ChunkRun cr = block_chunks(n_chunks, (int)gridDim.x, b);
  const u32 s = take_seq(seq, b, &bcast);
  const RotRegions R = regions(peer_scratch, my_scratch, loc_pack, s, stride_bytes, scale_off);
  unsigned h = 0u;
  for (int c = cr.c0 + wave; c < cr.c1; c += nwaves) push_chunk<6, kBf16, kNoCheck>(in, R, c, n_groups, nrm, h);
  drain_and_sync(drain);
  if (threadIdx.x == 0) {
    flag_release(peer_flags + b, s);
    split_arrive(peer_flags, seq, flag_val, (int)gridDim.x);
  }
}

// the push half on P <= nbv blocks: physical block b walks the virtual blocks b, b + P, ... of the one-shot's grid
__device__ __forceinline__ void split_push_nb(const u16* in, char* peer_scratch, char* my_scratch, char* loc_pack,
                                              u32* peer_flags, u32* seq, int n_groups, int n_chunks,
                                              long stride_bytes, long scale_off, int drain, u32 flag_val, int nbv) {
  __shared__ u32 bcast;
  const int b = (int)blockIdx.x, P = (int)gridDim.x;
  const int wave = (int)threadIdx.x >> 5, nwaves = (int)blockDim.x >> 5;
  const float nrm = rsq(64.0f);
  unsigned h = 0u;
  for (int vb = b; vb < nbv; vb += P) {
    const ChunkRun cr = block_chunks(n_chunks, nbv, vb);
    const u32 s = take_seq(seq, vb, &bcast);
    const RotRegions R = regions(peer_scratch, my_scratch, loc_pack, s, stride_bytes, scale_off);
    for (int c = cr.c0 + wave; c < cr.c1; c += nwaves) push_chunk<6, kBf16, kNoCheck>(in, R, c, n_groups, nrm, h);
    __syncthreads();   // the broadcast word is reused by the next virtual block
  }
  drain_and_sync(drain);
  if (threadIdx.x == 0) {
    for (int vb = b; vb < nbv; vb += P)
      flag_release(peer_flags + vb, __hip_atomic_load(seq + vb, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
    split_arrive(peer_flags, seq, flag_val, P);
  }
}

// the reduce half: the block's counter as the push left it, no hand-off of its own
__device__ __forceinline__ void split_reduce(char* my_scratch, char* loc_pack, u16* out, u32* seq, int n_groups,
                                             int n_chunks, long stride_bytes, long scale_off, int acq) {
  const int b = (int)blockIdx.x;
  const int wave = (int)threadIdx.x >> 5, nwaves = (int)blockDim.x >> 5;
  const float nrm = rsq(64.0f);
  const ChunkRun cr = block_chunks(n_chunks, (int)gridDim.x, b);
  const u32 s = __hip_atomic_load(seq + b, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  const RotRegions R = regions(nullptr, my_scratch, loc_pack, s, stride_bytes, scale_off);
  acquire_opt(acq);
  unsigned h = 0u;
  for (int c = cr.c0 + wave; c < cr.c1; c += nwaves) reduce_chunk<6, kBf16, kNoCheck>(out, R, c, n_groups, nrm, h);
}

}  // namespace sar
